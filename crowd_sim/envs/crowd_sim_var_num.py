import gym
import numpy as np
from numpy.linalg import norm
import copy
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
# import networkx as nx
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import *
from crowd_sim.envs.utils.utils import *
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.group import Group
from crowd_sim.envs.utils.state import JointState


class CrowdSimVarNum(CrowdSim):
    """
    The environment for our model with no trajectory prediction, or the baseline models with no prediction
    The number of humans at each timestep can change within a range
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.id_counter = None
        self.observed_human_ids = None
        self.pred_method = None
        self.group_centroids = None
        self.group_radii = None
        
    def configure(self, config):
        """ read the config to the environment variables """
        super(CrowdSimVarNum, self).configure(config)
        self.action_type=config.action_space.kinematics

    # set observation space and action space
    def set_robot(self, robot):
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d={}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype = np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_human_num, 2), dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.max_human_num,),
                                            dtype=bool)
        self.observation_space=gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if self.record:
            px, py = 0, 0
            gx, gy = 0, -1.5
            self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
            # generate a dummy human
            for i in range(self.max_human_num):
                human = Human(self.config, 'humans')
                human.set(15, 15, 15, 15, 0, 0, 0)
                human.isObstacle = True
                self.humans.append(human)

        else:
            # for sim2real
            if self.robot.kinematics == 'unicycle':
                # generate robot
                angle = np.random.uniform(0, np.pi * 2)
                px = self.arena_size * np.cos(angle)
                py = self.arena_size * np.sin(angle)
                while True:
                    gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= 4:  # 1 was 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2 * np.pi))  # randomize init orientation
                # 1 to 4 humans
                self.human_num = np.random.randint(1, self.config.sim.human_num + self.human_num_range + 1)
                # print('human_num:', self.human_num)
                # self.human_num = 4


            # for sim exp
            else:
                # generate robot
                while True:
                    px, py, gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 4)
                    if np.linalg.norm([px - gx, py - gy]) >= 8: # 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
                # generate humans
                self.human_num = np.random.randint(low=self.config.sim.human_num - self.human_num_range,
                                                   high=self.config.sim.human_num + self.human_num_range + 1)

            self.generate_random_human_position(human_num=self.human_num)
            self.last_human_states = np.zeros((self.human_num, 5))

            # set human ids
            # for i in range(self.human_num):
            #     self.humans[i].id = i


    # generate a human that starts on a circle, and its goal is on the opposite side of the circle
    def generate_circle_crossing_human(self, human):

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 2
            px_noise = np.random.uniform(0, 1) * noise_range
            py_noise = np.random.uniform(0, 1) * noise_range
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            for i, agent in enumerate([self.robot] + self.humans):
                # keep human at least 3 meters away from robot
                if self.robot.kinematics == 'unicycle' and i == 0:
                    min_dist = self.circle_radius / 2  # Todo: if circle_radius <= 4, it will get stuck here
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)

        return human

    # calculate the ground truth future trajectory of humans
    # if robot is visible: assume linear motion for robot
    # ret val: [self.predict_steps + 1, self.human_num, 4]
    # method: 'truth' or 'const_vel' or 'inferred'
    def calc_human_future_traj(self, method):
        # if the robot is invisible, it won't affect human motions
        # else it will
        human_num = self.human_num + 1 if self.robot.visible else self.human_num
        # buffer to store predicted future traj of all humans [px, py, vx, vy]
        # [time, human id, features]
        if method == 'truth':
            self.human_future_traj = np.zeros((self.buffer_len + 1, human_num, 4))
        elif method == 'const_vel':
            self.human_future_traj = np.zeros((self.predict_steps + 1, human_num, 4))
        else:
            raise NotImplementedError

        # initialize the 0-th position with current states
        for i in range(self.human_num):
            # use true states for now, to count for invisible humans' influence on visible humans
            # take px, py, vx, vy, remove radius
            self.human_future_traj[0, i] = np.array(self.humans[i].get_observable_state_list()[:-1])

        # if we are using constant velocity model, we need to use displacement to approximate velocity (pos_t - pos_t-1)
        # we shouldn't use true velocity for fair comparison with GST inferred pred
        if method == 'const_vel':
            self.human_future_traj[0, :, 2:4] = self.prev_human_pos[:, 2:4]

        # add robot to the end of the array
        if self.robot.visible:
            self.human_future_traj[0, -1] = np.array(self.robot.get_observable_state_list()[:-1])

        if method == 'truth':
            for i in range(1, self.buffer_len + 1):
                for j in range(self.human_num):
                    # prepare joint state for all humans
                    full_state = np.concatenate(
                        (self.human_future_traj[i - 1, j], self.humans[j].get_full_state_list()[4:]))
                    observable_states = []
                    for k in range(self.human_num):
                        if j == k:
                            continue
                        observable_states.append(
                            np.concatenate((self.human_future_traj[i - 1, k], [self.humans[k].radius])))

                    # use joint states to get actions from the states in the last step (i-1)
                    action = self.humans[j].act_joint_state(JointState(full_state, observable_states))

                    # step all humans with action
                    self.human_future_traj[i, j] = self.humans[j].one_step_lookahead(
                        self.human_future_traj[i - 1, j, :2], action)

                if self.robot.visible:
                    action = ActionXY(*self.human_future_traj[i - 1, -1, 2:])
                    # update px, py, vx, vy
                    self.human_future_traj[i, -1] = self.robot.one_step_lookahead(self.human_future_traj[i - 1, -1, :2],
                                                                                  action)
            # only take predictions every self.pred_interval steps
            self.human_future_traj = self.human_future_traj[::self.pred_interval]
        # for const vel model
        elif method == 'const_vel':
            # [self.pred_steps+1, human_num, 4]
            self.human_future_traj = np.tile(self.human_future_traj[0].reshape(1, human_num, 4), (self.predict_steps+1, 1, 1))
            # [self.pred_steps+1, human_num, 2]
            pred_timestep = np.tile(np.arange(0, self.predict_steps+1, dtype=float).reshape((self.predict_steps+1, 1, 1)) * self.time_step * self.pred_interval,
                                    [1, human_num, 2])
            pred_disp = pred_timestep * self.human_future_traj[:, :, 2:]
            self.human_future_traj[:, :, :2] = self.human_future_traj[:, :, :2] + pred_disp
        else:
            raise NotImplementedError

        # remove the robot if it is visible
        if self.robot.visible:
            self.human_future_traj = self.human_future_traj[:, :-1]


        # remove invisible humans
        self.human_future_traj[:, np.logical_not(self.human_visibility), :2] = 15
        self.human_future_traj[:, np.logical_not(self.human_visibility), 2:] = 0

        return self.human_future_traj

 
    def generate_ob(self, reset, sort=False):
        """Generate observation with added direction consistency for group detection."""
        ob = {}
        eps_value = 1.0  # Adjust this to control clustering sensitivity
        min_samples_value = 2  # Minimum number of points to form a group
        spatial_weight = 0.8
        velocity_weight = 1
        directional_weight = 1
        distance_weight = 1  # Weight for the pairwise distance feature

        # Nodes (robot and human states)
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        ob['robot_node'] = self.robot.get_full_state_list_noV()

        prev_human_pos = copy.deepcopy(self.last_human_states)
        self.update_last_human_states(self.human_visibility, reset=reset)

        # Edges (robot velocity)
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # Initialize spatial, velocity, and direction consistency arrays
        all_spatial_edges = np.ones((self.max_human_num, 2)) * np.inf  # [relative_px, relative_py]
        all_velocity_edges = np.ones((self.max_human_num, 2)) * np.inf  # [relative_vx, relative_vy]
        direction_consistency_edges = np.ones((self.max_human_num, )) * np.inf  # Direction consistency score

        grp_centroids = np.ones((self.num_groups, 2)) * np.inf  # [relative_vx, relative_vy]
        grp_radii = np.ones((self.num_groups,)) * np.inf  # Direction consistency score
        
        # Collect positions, velocities, and IDs of visible humans for clustering
        visible_positions = []
        visible_velocities = []
        visible_human_ids = []
        
        cluster_dict = {}
        # for i in range(self.human_num):
        #     group_id = self.humans[i].group_id
        #     if group_id is not None:
        #         cluster_label = group_id
        #         if cluster_label not in cluster_dict:
        #             cluster_dict[group_id] = []
        #         cluster_dict[group_id].append(self.humans[i].id)

        for i in range(self.human_num):
            if self.human_visibility[i]:
                # Position relative to the robot
                relative_pos = np.array([self.last_human_states[i, 0] - self.robot.px, 
                                        self.last_human_states[i, 1] - self.robot.py])

                # Calculate relative velocity (human velocity relative to robot velocity)
                relative_velocity = np.array([self.last_human_states[i, 2] - self.robot.vx, 
                                            self.last_human_states[i, 3] - self.robot.vy])

                # Store spatial and velocity features
                all_spatial_edges[self.humans[i].id, :] = relative_pos
                all_velocity_edges[self.humans[i].id, :] = relative_velocity

                visible_positions.append(relative_pos)
                visible_velocities.append(relative_velocity)
                visible_human_ids.append(self.humans[i].id)  # Track human_id
                
                # For Ground Truth Grouping 
                if self.group_ground_truth:
                    group_id = self.humans[i].group_id
                    if group_id is not None:
                        cluster_label = group_id
                        if cluster_label not in cluster_dict:
                            cluster_dict[group_id] = []
                        cluster_dict[group_id].append(self.humans[i].id) 
        
        # print(cluster_dict)

        # Compute direction consistency using cosine similarity
        def cosine_similarity(vec1, vec2):
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                # If either of the vectors has zero magnitude, return 0 (no similarity)
                return 0
            
            # Compute cosine similarity
            return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

        if len(visible_positions) > 0:
            visible_positions = np.array(visible_positions)
            visible_velocities = np.array(visible_velocities)

            
            # My code for human Group Detection

            # Normalize spatial and velocity features (Z-Score)
            # spatial_mean = visible_positions.mean(axis=0)
            # spatial_std = visible_positions.std(axis=0) + 1e-6  # To avoid division by zero
            # velocity_mean = visible_velocities.mean(axis=0)
            # velocity_std = visible_velocities.std(axis=0) + 1e-6

            # norm_spatial = (visible_positions - spatial_mean) / spatial_std
            # norm_velocity = (visible_velocities - velocity_mean) / velocity_std

            # # Calculate direction consistency for each human
            # robot_velocity = np.array([self.robot.vx, self.robot.vy])
            # for idx, vel in enumerate(visible_velocities):
            #     direction_consistency = cosine_similarity(robot_velocity, vel)
            #     direction_consistency_edges[visible_human_ids[idx], 0] = direction_consistency

            # # Calculate pairwise distances among visible humans
            # visible_human_distances = np.zeros((len(visible_positions), len(visible_positions)))
            # for i in range(len(visible_positions)):
            #     for j in range(i + 1, len(visible_positions)):
            #         # Calculate Euclidean distance between visible humans i and j
            #         distance = np.linalg.norm(visible_positions[i] - visible_positions[j])
            #         visible_human_distances[i, j] = distance
            #         visible_human_distances[j, i] = distance

            # # Add pairwise distances to the observation for use in clustering
            # ob['visible_human_distances'] = visible_human_distances

            # # Normalize pairwise distances
            # distance_mean = visible_human_distances.mean()
            # distance_std = visible_human_distances.std() + 1e-6
            # normalized_distances = (visible_human_distances - distance_mean) / distance_std
            
            # # Combine normalized spatial, velocity, and direction consistency features for clustering
            # # combined_features = np.hstack([normalized_distances * distance_weight, norm_velocity * velocity_weight, \
            # #     direction_consistency_edges[visible_human_ids] * directional_weight])
            # combined_features = np.hstack([norm_spatial * spatial_weight, norm_velocity * velocity_weight, \
            #     direction_consistency_edges[visible_human_ids] * directional_weight])

            # # Perform clustering using DBSCAN on the combined features
            # # clu = self.cluster_by_distance(normalized_distances, 1.0, visualize=False)
            # # print(f"Another Clustering: {clu}")
            # clustering = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(combined_features)
            # cluster_labels = clustering.labels_  # Cluster labels for each human (noise is labeled -1)

            # # Create a dictionary to track which human_ids belong to each group (cluster)
            # cluster_dict = {}
            # for idx, cluster_label in enumerate(cluster_labels):
            #     human_id = visible_human_ids[idx]
            #     if cluster_label != -1:  # Ignore noise (-1)
            #         if cluster_label not in cluster_dict:
            #             cluster_dict[cluster_label] = []
            #         cluster_dict[cluster_label].append(human_id)

            # # Add group/cluster information to the observation
            # ob['clusters'] = cluster_labels  # Cluster labels for each human
            # ob['group_members'] = cluster_dict  # Dictionary mapping group_id (cluster_label) to human_ids
            # print(f"Groups: {ob['group_members']}\n")
            
            # My code Ended
            
            ob['clusters'] = np.array([])  # No clusters if no humans are visible
            ob['group_members'] = cluster_dict           
            # print(cluster_dict)
            
        else:
            ob['clusters'] = np.array([])  # No clusters if no humans are visible
            ob['group_members'] = {}  # No group members detected

        ob['clusters'] = update_cluster(self.max_human_num, cluster_dict) 
        # print(f"Groups: {ob['group_members']}\n")
       
        # Store other observation data
        ob['visible_masks'] = np.zeros(self.max_human_num, dtype=bool)
        ob['spatial_edges'] = all_spatial_edges
        ob['velocity_edges'] = all_velocity_edges
        ob['direction_consistency'] = direction_consistency_edges

        # Sort humans by distance if needed
        if sort:
            ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x)))
            if num_visibles > 0:
                ob['visible_masks'][:num_visibles] = True
        else:
            ob['visible_masks'][:self.human_num] = self.human_visibility

        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        ob['velocity_edges'][np.isinf(ob['velocity_edges'])] = 15
        ob['direction_consistency'][np.isinf(ob['direction_consistency'])] = 15
        ob['detected_human_num'] = num_visibles
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # Update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]

        ob['grp'] = False
        self.ob = ob
        
        # Identify detected groups and calculate their positions
        detected_groups = self.ob.get('group_members', {})
        
        if detected_groups:
            grp_detected = True
            self.group_centroids = []  # To store group centroids
            self.group_radii = []      # To store group safety radii

            for group_id, human_ids in detected_groups.items():
                group_positions = np.array([[self.last_human_states[i, 0], self.last_human_states[i, 1]] 
                                            for i in human_ids])

                # Calculate the centroid of the group
                centroid = np.mean(group_positions, axis=0)
                self.group_centroids.append(centroid)
                
                grp_centroids[group_id, 0] = centroid[0]
                grp_centroids[group_id, 1] = centroid[1]

                # Calculate the group's bounding radius (or convex hull radius)
                max_distance = np.max(np.linalg.norm(group_positions - centroid, axis=1))
                radii = max_distance + self.group_safety_buffer
                self.group_radii.append(radii)  # Add safety buffer
                grp_radii[group_id] = radii 

            ob['grp'] = grp_detected
            # if ob['grp']:
            #     ob['group_members'] = detected_groups
            # ob['group_centroids'] = self.group_centroids
            # ob['group_radii'] = self.group_radii
        
             
        ob['group_centroids'] = grp_centroids
        ob['group_radii'] = grp_radii

        ob['group_centroids'][np.isinf(ob['group_centroids'])] = 15
        ob['group_radii'][np.isinf(ob['group_radii'])] = 15
        self.ob = ob

        return ob

    # Update the specified human's end goals in the environment randomly
    def update_human_pos_goal(self, human):
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            gx_noise = (np.random.random() - 0.5) * v_pref
            gy_noise = (np.random.random() - 0.5) * v_pref
            gx = self.circle_radius * np.cos(angle) + gx_noise
            gy = self.circle_radius * np.sin(angle) + gy_noise
            collide = False

            if not collide:
                break

        # Give human new goal
        human.gx = gx
        human.gy = gy


    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0

        self.humans = []
        self.group_counter = 0
        self.leader = {}
        self.leader_act = {}
        self.grp = []

        for i in range(self.num_groups):
            self.grp.append(Group(i, self.min_size, self.max_size))
        # self.human_num = self.config.sim.human_num
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        np.random.seed(self.rand_seed)

        self.generate_robot_humans(phase)

        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # initialize potential and angular potential
        rob_goal_vec = np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py])
        self.potential = -abs(np.linalg.norm(rob_goal_vec))
        self.angle = np.arctan2(rob_goal_vec[1], rob_goal_vec[0]) - self.robot.theta
        if self.angle > np.pi:
            # self.abs_angle = np.pi * 2 - self.abs_angle
            self.angle = self.angle - 2 * np.pi
        elif self.angle < -np.pi:
            self.angle = self.angle + 2 * np.pi

        # get robot observation
        ob = self.generate_ob(reset=True, sort=self.config.args.sort_humans)

        return ob


    def step(self, action, update=True):
        """
        Step the environment forward for one timestep
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.robot.policy.name in ['ORCA', 'social_force']:
            # assemble observation for orca: px, py, vx, vy, r
            human_states = copy.deepcopy(self.last_human_states)
            # get orca action
            action = self.robot.act(human_states.tolist())
        else:
            action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
            action = ActionRot(self.desiredVelocity[0], action.r)

        human_actions = self.get_human_actions()

        # need to update self.human_future_traj in testing to calculate number of intrusions
        if self.phase == 'test':
            # use ground truth future positions of humans
            self.calc_human_future_traj(method='truth')

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action, danger_zone='future')


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter =self.step_counter+1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # print('before:', self.human_num,', self.min_human_num:', self.min_human_num)
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - self.min_human_num
                    # print('max_remove_num, invisible', max_remove_num)
                else:
                    max_remove_num = min(self.human_num - self.min_human_num, (self.human_num - 1) - max(self.observed_human_ids))
                    # print('max_remove_num, viscluster_dictible', max_remove_num)
                remove_num = np.random.randint(low=0, high=max_remove_num + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                # print('after:', self.human_num)
                self.last_human_states = self.last_human_states[:self.human_num]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.generate_random_human_position(human_num=1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num
                    if true_add_num > 0:
                        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]]*true_add_num)), axis=0)

        assert self.min_human_num <= self.human_num <= self.max_human_num

        # compute the observation
        ob = self.generate_ob(reset=False, sort=self.config.args.sort_humans)


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    if self.robot.kinematics == 'holonomic':
                        self.humans[i] = self.generate_circle_crossing_human(self.initialize_human(i))
                    else:
                        self.update_human_goal(human)

        return ob, reward, done, info

    # find R(s, a)
    # danger_zone: how to define the personal_zone (if the robot intrudes into this zone, the info will be Danger)
    # circle (traditional) or future (based on true future traj of humans)
    def calc_reward(self, action, danger_zone='circle'):
        # collision detection
        dmin = float('inf')
        dmingrp = float('inf')

        danger_dists = []
        collision = False

        # collision check with humans
        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
        
        danger_dists = []
        grp_collision = False
        
        # collision check with groups
        if self.group_centroids is not None:
            for i, center in enumerate(self.group_centroids):
                dx = center[0] - self.robot.px
                dy = center[1] - self.robot.py
                # closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.group_radii[i] - self.group_safety_buffer - self.robot.radius
                closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.group_radii[i] - 0.05

                if closest_dist < self.discomfort_group_dist:
                    danger_dists.append(closest_dist)
                if closest_dist < 0:
                    grp_collision = True
                    # print(f"Group Id: {i}")
                    break
                elif closest_dist < dmingrp:
                    dmingrp = closest_dist


        # check if reaching the goal
        if self.robot.kinematics == 'unicycle':
            goal_radius = 0.6
        else:
            goal_radius = self.robot.radius
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < goal_radius
        

        # use danger_zone to determine the condition for Danger
        if danger_zone == 'circle' or self.phase == 'train':
            danger_cond_grp = dmingrp < self.discomfort_group_dist
            danger_cond = dmin < self.discomfort_dist
            min_danger_dist = 0
        else:
            if self.phase == 'test':
                danger_cond_grp = False
            # if the robot collides with future states, give it a collision penalty
            relative_pos = self.human_future_traj[1:, :, :2] - np.array([self.robot.px, self.robot.py])
            relative_dist = np.linalg.norm(relative_pos, axis=-1)

            collision_idx = relative_dist < self.robot.radius + self.config.humans.radius  # [predict_steps, human_num]

            danger_cond = np.any(collision_idx)
            # if robot is dangerously close to any human, calculate the min distance between robot and its closest human
            if danger_cond:
                min_danger_dist = np.amin(relative_dist[collision_idx])
            else:
                min_danger_dist = 0

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif grp_collision:
            reward = self.grp_collision_penalty
            done = True
            episode_info = GroupCollision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()
        
        elif danger_cond_grp:
            # only penalize agent for getting too close to grp if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmingrp - self.discomfort_group_dist) * self.discomfort_grp_penalty_factor * self.time_step
            done = False
            episode_info = Danger(min_danger_dist)

        elif danger_cond:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(min_danger_dist)

        else:
            # potential reward
            if self.robot.kinematics == 'holonomic':
                pot_factor = 2
            else:
                pot_factor = 3
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = pot_factor * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()

        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            r_spin = -4.5 * action.r ** 2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.

            reward = reward + r_spin + r_back

        return reward, done, episode_info


    def render(self, mode='human'):
        """ Render the current status of the environment using matplotlib """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        # plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range + self.robot.radius+self.config.humans.radius, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.humans]

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            # if self.detect_visible(self.robot, self.humans[i], robot1=True):
            if self.human_visibility[i]:
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')
            if self.humans[i].id in self.observed_human_ids:
                human_circles[i].set_color(c='b')

            plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)

        plt.pause(0.01)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)

