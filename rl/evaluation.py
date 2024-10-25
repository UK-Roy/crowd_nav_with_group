import numpy as np
import torch

from crowd_sim.envs.utils.info import *


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, visualize=False):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit
    time_step = baseEnv.time_step
    # v_pref = baseEnv.robot.v_pref

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()
        grp_obs = {}

        while not done:
            stepCounter = stepCounter + 1
            act = None
            if config.robot.policy not in ['orca', 'social_force']:
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
            
            if grp_obs:
                # Identify detected groups and calculate their positions
                detected_groups = grp_obs.get('group_members', {})

                if detected_groups:
                    group_centroids = grp_obs['group_centroids']
                    group_radii = grp_obs['group_radii']

                    # Robot's current position and goal position
                    robot_position = torch.tensor([obs['robot_node'][0, 0, 0], obs['robot_node'][0, 0, 1]], device=device)
                    goal_position = torch.tensor([obs['robot_node'][0, 0, 3], obs['robot_node'][0, 0, 4]], device=device)

                    # Now adjust the robot's velocity based on group avoidance
                    for centroid, radius in zip(group_centroids, group_radii):
                        # Calculate the vector from the robot to the goal
                        robot_to_goal = goal_position - robot_position
                        distance_robot_to_goal = torch.norm(robot_to_goal)
                        distance_centroid_to_goal = torch.norm(centroid - goal_position)

                        robot_to_goal = robot_to_goal.type(torch.float64)
                        robot_to_group = centroid - robot_position
                
                        # Check if the group is between the robot and the goal
                        # We do this by checking if the dot product of the direction to the goal and the direction to the group is positive (they are in the same direction)
                        if torch.dot(robot_to_goal, robot_to_group) > 0:
                            # Calculate the distance from the group to the goal path
                            distance_to_group = torch.norm(robot_to_group)

                            # If the group is too close, adjust the robot's action to avoid it
                            if distance_to_group < radius:
                                if distance_robot_to_goal < distance_centroid_to_goal:
                                    # Prioritize the goal if the robot is closer to the goal than the group centroid
                                    break  # Stop avoiding the group, and proceed toward the goal
                                angle = cal_vec(obs, action, centroid, device)
                                if angle < 1.56:
                                    act = find_perpendi(robot_position, centroid, device)
                                break  # Only adjust for the nearest group


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
            grp_obs = obs.pop('grp')

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

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

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision')
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
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    # logging
    logging.info(
        'Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist)))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    eval_envs.close()

def cal_vec(obs, action, centroid, device):
    # Robot's current position
    robot_position = torch.tensor([obs['robot_node'][0, 0, 0], obs['robot_node'][0, 0, 1]], device=device)

    # Intended position vector (from robot to intended position)
    intended_position = torch.tensor([obs['robot_node'][0, 0, 0] + action[0, 0] * 0.25,\
                                    obs['robot_node'][0, 0, 1] + action[0, 1] * 0.25],dtype=torch.float64, device=device)
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

def find_perpendi(robot_position, centroid, device):

    # Centroid vector (from robot to group centroid)
    centroid_vector = centroid - robot_position

    # To rotate the vector 90 degrees, swap the x and y components and negate one of them
    # Rotation by +90 degrees:
    perpendicular_vector = torch.tensor([-centroid_vector[1], centroid_vector[0]], device=device)

    # Normalize the perpendicular vector to get a direction vector
    perpendicular_direction = perpendicular_vector / torch.norm(perpendicular_vector)

    # print(f"Intended position (90 degrees to centroid vector): {perpendicular_direction}")

    return perpendicular_direction