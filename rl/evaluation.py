import math
import numpy as np
import torch

from crowd_sim.envs.utils.info import *
from crowd_nav.policy.taga_safety import TAGASafetyController


def _sigmoid_alpha(d_group, d_switch, band):
    """Sigmoid blend weight: 1 when close to group, 0 when far.
    Centered at d_switch; band/3 sets steepness so alpha≈0.95 at d_switch-band
    and alpha≈0.05 at d_switch+band.
    """
    return 1.0 / (1.0 + math.exp((d_group - d_switch) / max(band / 3.0, 1e-6)))


def _tangent_side_cost(tangent_dir, robot_pos, goal_pos, obs, group_centroids,
                       group_radii, detected_groups, current_gid, taga_cfg, device):
    """
    Cost for choosing a given tangent direction.
    cost = w_goal * goal_angle_cost + w_obstacle * obstacle_density_cost

    goal_angle_cost  ∈ [0,1]: 0 = tangent aligned with goal, 1 = opposite
    obstacle_density_cost: sum of 1/(d+0.1) for humans/groups in a forward cone
    """
    tang_unit = tangent_dir / (torch.norm(tangent_dir) + 1e-9)
    goal_dir  = (goal_pos - robot_pos)
    goal_dir  = goal_dir / (torch.norm(goal_dir) + 1e-9)

    cos_to_goal  = torch.clamp(torch.dot(tang_unit, goal_dir), -1.0, 1.0)
    goal_cost    = (1.0 - cos_to_goal) / 2.0  # [0, 1]

    cos_thresh   = math.cos(math.radians(taga_cfg.cone_half_angle))
    look_ahead   = taga_cfg.look_ahead
    obs_cost     = 0.0

    # Individual humans from spatial_edges
    if obs is not None and 'spatial_edges' in obs:
        spatial = obs['spatial_edges'][0]   # [num_humans, features]
        masks   = obs.get('visible_masks', [None])[0]
        for i in range(len(spatial)):
            if masks is not None and not masks[i]:
                continue
            rel  = spatial[i, :2]
            d    = torch.norm(rel).item()
            if d < 1e-3 or d > look_ahead:
                continue
            rel_unit = rel / (torch.norm(rel) + 1e-9)
            if torch.dot(tang_unit, rel_unit).item() > cos_thresh:
                obs_cost += 1.0 / (d + 0.1)

    # Other groups
    for gid in detected_groups:
        if gid == current_gid:
            continue
        g_cent = group_centroids[gid]
        g_rad  = float(group_radii[gid])
        rel    = g_cent - robot_pos
        d      = torch.norm(rel).item()
        if d > look_ahead + g_rad:
            continue
        rel_unit = rel / (torch.norm(rel) + 1e-9)
        if torch.dot(tang_unit, rel_unit).item() > cos_thresh:
            obs_cost += 2.0 / (d + 0.1)   # groups weighted double vs. individuals

    return taga_cfg.w_goal * goal_cost.item() + taga_cfg.w_obstacle * obs_cost


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, visualize=False, group_avoid_action=False):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ['orca', 'social_force', 'zone_based', 'f_formation']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)
    # Add group intrusion tracking
    group_intrusion_count = []
    group_intrusion_time = []
    total_time_in_groups = 0
    episode_group_intrusions = 0
     
    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    grp_collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    grp_collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    grp_collision_cases = []
    timeout_cases = []

    # Initialize safety controller for TAGA
    if group_avoid_action:
        safety_controller = TAGASafetyController(config)
        logging.info("TAGA Safety Controller initialized")
    
    group_intrusion_ratios = []  # Percentage of time in groups per episode 
    all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit
    # time_step = baseEnv.time_step

    # v_pref = baseEnv.robot.v_pref

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        if group_avoid_action:
            safety_controller.reset()   # clear velocity history for accel filter
        episode_rew = 0
        obs = eval_envs.reset()
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()
        episode_group_intrusions = 0

        grp_obs = {}

        while not done:
            stepCounter = stepCounter + 1
            act = None
            
            if config.robot.policy not in ['orca', 'social_force', 'zone_based', 'f_formation']:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else:
                action = torch.zeros([1, 2], device=device)
            if not done:
                global_time = baseEnv.global_time
           
            if group_avoid_action:
                # Read group info directly from env (obs keys are not in obs_space so VecEnv filters them)
                env_grp = getattr(baseEnv, 'grp', [])
                env_hulls = getattr(baseEnv, 'group_hulls', {}) or {}
                if env_grp and env_hulls:
                    group_dict = {}
                    group_centroids_direct = {}
                    group_radii_direct = {}
                    for g in env_grp:
                        if g.members and g.id in env_hulls:
                            hull = env_hulls[g.id]
                            group_dict[g.id] = [h.id for h in g.members]
                            cx, cy = hull.centroid if hasattr(hull, 'centroid') else (g.centroid[0], g.centroid[1])
                            group_centroids_direct[g.id] = torch.tensor([cx, cy], device=device, dtype=torch.float32)
                            group_radii_direct[g.id] = hull.bounding_radius if hasattr(hull, 'bounding_radius') else g.radius
                    if group_dict:
                        grp_obs = {'group_members': group_dict,
                                   'group_centroids': group_centroids_direct,
                                   'group_radii': group_radii_direct}

                if grp_obs and grp_obs.get('group_members'):
                    detected_groups = grp_obs['group_members']
                    group_centroids = grp_obs['group_centroids']   # dict: gid -> tensor
                    group_radii     = grp_obs['group_radii']       # dict: gid -> scalar

                    robot_position = torch.tensor(
                        [obs['robot_node'][0, 0, 0], obs['robot_node'][0, 0, 1]], device=device)
                    goal_position = torch.tensor(
                        [obs['robot_node'][0, 0, 3], obs['robot_node'][0, 0, 4]], device=device)

                    taga_cfg = config.taga
                    v_pref = config.robot.v_pref

                    robot_to_goal = goal_position - robot_position
                    d_robot_goal  = torch.norm(robot_to_goal).item()

                    # Outer goal priority: skip TAGA entirely this step
                    goal_priority_fire = False
                    for group_id in detected_groups:
                        centroid        = group_centroids[group_id]
                        d_centroid_goal = torch.norm(centroid - goal_position).item()
                        if d_robot_goal < d_centroid_goal and d_robot_goal < taga_cfg.goal_threshold:
                            goal_priority_fire = True
                            break

                    if not goal_priority_fire:
                        # P2: collect ALL blocking groups sorted by proximity
                        blocking = []
                        for group_id in detected_groups:
                            centroid        = group_centroids[group_id]
                            radius          = group_radii[group_id]
                            robot_to_group  = centroid - robot_position
                            d_centroid_goal = torch.norm(centroid - goal_position).item()
                            d_group         = torch.norm(robot_to_group).item()

                            # Group must be ahead of robot
                            if torch.dot(robot_to_goal, robot_to_group).item() <= 0:
                                continue
                            # Inner goal priority per group
                            if d_robot_goal < d_centroid_goal:
                                continue

                            # P0: sigmoid alpha
                            d_switch = float(radius) + taga_cfg.safe_margin
                            if taga_cfg.smooth_switching:
                                alpha = _sigmoid_alpha(d_group, d_switch, taga_cfg.switch_band)
                            else:
                                alpha = 1.0 if d_group < d_switch else 0.0

                            if alpha <= 1e-3:
                                continue

                            # P1: cost-aware side selection
                            cw  = find_perpendi(robot_position, centroid, device, clockwise=True)
                            ccw = find_perpendi(robot_position, centroid, device, clockwise=False)
                            if taga_cfg.cost_aware_side:
                                cw_cost  = _tangent_side_cost(cw,  robot_position, goal_position,
                                                              obs, group_centroids, group_radii,
                                                              detected_groups, group_id,
                                                              taga_cfg, device)
                                ccw_cost = _tangent_side_cost(ccw, robot_position, goal_position,
                                                              obs, group_centroids, group_radii,
                                                              detected_groups, group_id,
                                                              taga_cfg, device)
                                tangent = cw if cw_cost <= ccw_cost else ccw
                            else:
                                _, cw_angle = angle_between_vectors(cw, robot_to_goal)
                                cw_angle  = cw_angle.item()
                                ccw_angle = 180.0 - cw_angle
                                tangent   = cw if cw_angle < ccw_angle else ccw

                            blocking.append((alpha, tangent, d_group))

                        if blocking:
                            if taga_cfg.multi_group:
                                # P2: weighted-average tangent over k-nearest groups
                                blocking.sort(key=lambda x: x[2])          # nearest first
                                blocking = blocking[:taga_cfg.max_groups]
                                agg = torch.zeros(2, device=device)
                                total_w = 0.0
                                for alpha, tangent, d_g in blocking:
                                    w = alpha / (d_g + 1e-9)               # closer = higher weight
                                    agg    += w * tangent
                                    total_w += w
                                agg_norm = torch.norm(agg)
                                tangent_final = agg / (agg_norm + 1e-9)
                                alpha_final   = min(sum(a for a, _, _ in blocking), 1.0)
                            else:
                                # Legacy: first (closest) group only
                                blocking.sort(key=lambda x: x[2])
                                alpha_final, tangent_final, _ = blocking[0]

                            base_action = action[0]
                            taga_scaled = tangent_final * v_pref
                            desired = alpha_final * taga_scaled + (1.0 - alpha_final) * base_action
                            act = safety_controller.get_safe_taga_action(obs, desired, device)

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'CrowdSimPredRealGST-v0' and config.env.use_wrapper:
                out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
                # send manager action to all processes
                ack = eval_envs.talk2Env(out_pred)
                assert all(ack)
            # render
            if visualize:
                eval_envs.render()

            if act is not None:
                action = act.unsqueeze(0)
            

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)
            # grp_obs = obs.pop('grp')

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)
            
            if isinstance(infos[0]['info'], GroupIntrusion):
                if k not in group_intrusion_count:  # Note: use 'k' not 'episode_k'
                    group_intrusion_count.append(k)
                total_time_in_groups += 1
                episode_group_intrusions += 1

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)

        if stepCounter > 0:
            episode_intrusion_ratio = (episode_group_intrusions / stepCounter) * 100
            group_intrusion_ratios.append(episode_intrusion_ratio)
        else:
            group_intrusion_ratios.append(0)

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision') 
        # elif isinstance(infos[0]['info'], GroupCollision):
        #     grp_collision += 1
        #     grp_collision_cases.append(k)
        #     grp_collision_times.append(global_time)
        #     print('Group Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            print('Time out')
        elif isinstance(infos[0]['info'] is None):
            pass
        else:
            raise ValueError('Invalid end signal from environment')
    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    grp_collision_rate = grp_collision / test_size
    timeout_rate = timeout / test_size
    # assert success + collision + grp_collision + timeout == test_size
    assert success + collision + timeout == test_size  # Remove grp_collision from sum
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit
    gcr_rate = np.mean(group_intrusion_ratios) if group_intrusion_ratios else 0  # Average GCR across episodes

    # Calculate GCR as a separate metric (percentage of time in groups)
    # Calculate GCR as percentage of steps spent in groups
    # total_steps = sum([steps for steps in stepCounter])  # Need to track all step counts
    # gcr_rate = (total_time_in_groups / (test_size * average_steps_per_episode)) * 100 if test_size > 0 else 0
    # total_steps = sum([len(ep) for ep in all_path_len])
    # gcr_rate = total_time_in_groups / total_steps if total_steps > 0 else 0

    # logging
    # logging.info(
    #     'Testing success rate: {:.2f}, collision rate: {:.2f}, group collision rate: {:.2f}, timeout rate: {:.2f}, '
    #     'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
    #     'average minimal distance during intrusions: {:.2f}'.
    #         format(success_rate, collision_rate, grp_collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
    #                np.mean(too_close_ratios), np.mean(min_dist)))
    logging.info(
    'Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
    'group intrusion rate (GCR): {:.2f}%, '  # Now a percentage, not part of the sum
    'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
    'average minimal distance during intrusions: {:.2f}'.
    format(success_rate, collision_rate, timeout_rate, gcr_rate,  # GCR as percentage
           avg_nav_time, np.mean(all_path_len),
           np.mean(too_close_ratios), np.mean(min_dist)))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    # logging.info('Group Collision cases: ' + ' '.join([str(x) for x in grp_collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    logging.info(f'Mean Reward: {np.mean(eval_episode_rewards)}' )
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    eval_envs.close()


def cal_vec(obs, action, centroid, device):
    # Robot's current position
    robot_position = torch.tensor([obs['robot_node'][0, 0, 0], obs['robot_node'][0, 0, 1]], device=device)

    # Intended position vector (from robot to intended position)
    intended_position = torch.tensor([obs['robot_node'][0, 0, 0] + action[0, 0] * 0.25,\
                                    obs['robot_node'][0, 0, 1] + action[0, 1] * 0.25], device=device)
                                    # obs['robot_node'][0, 0, 1] + action[0, 1] * 0.25],dtype=torch.float64, device=device)
    intended_vector = intended_position - robot_position  # Vector from robot to intended position
    # intended_vector = intended_vector.type(torch.float64)

    centroid_vector = centroid - robot_position  # Vector from robot to group centroid

    # Calculate dot product between intended_vector and centroid_vector
    dot_product = torch.dot(intended_vector, centroid_vector)

    # Calculate magnitudes (norms) of the two vectors
    intended_magnitude = torch.norm(intended_vector)
    centroid_magnitude = torch.norm(centroid_vector)

    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (intended_magnitude * centroid_magnitude)

    # Calculate the angle in radians
    angle_radians = torch.acos(cos_theta)  # Returns angle in radians

    # Convert the angle to degrees if needed
    # angle_degrees = angle_radians * 180 / torch.pi

    # Print or return the angle
    # print(f"Angle between the vectors: {angle_radians.item()} radians, {angle_degrees.item()} degrees")
    return angle_radians

def find_perpendi(robot_position, centroid, device, clockwise=True):
    centroid_vector = centroid - robot_position
    if clockwise:
        perpendicular_vector = torch.tensor([-centroid_vector[1], centroid_vector[0]], device=device)
    else:
        perpendicular_vector = torch.tensor([centroid_vector[1], -centroid_vector[0]], device=device)

    return perpendicular_vector / torch.norm(perpendicular_vector)

# Function to check if any human is near the goal
def check_humans_near_goal(obs, goal_position, threshold, device):
    visible_humans = obs['visible_masks'][0]
    for i, visible in enumerate(visible_humans):
        if visible:
            human_position = torch.tensor([obs['spatial_edges'][0, i, 0], obs['spatial_edges'][0, i, 1]], device=device)
            distance_to_goal = torch.norm(human_position - goal_position)
            # distance_to_goal = torch.norm(human_position.type(torch.float64) - goal_position)
            # print(distance_to_goal)
            if distance_to_goal < threshold:
                return True  # Human is near the goal
    return False

def angle_between_vectors(v1, v2):
    
    # Calculate the dot product
    dot_product = torch.dot(v1, v2)
    
    # Calculate the magnitudes (norms) of the vectors
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)
    
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Clamp the value to avoid numerical errors that lead to invalid arccos values
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = torch.acos(cos_theta)
    
    # Optionally, convert to degrees
    angle_degrees = angle_radians * 180 / torch.pi
    
    return angle_radians, angle_degrees