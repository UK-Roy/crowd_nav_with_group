import numpy as np
from arguments import get_args

class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # for now, import all args from arguments.py
    args = get_args()

    training = BaseConfig()
    training.device = "cuda:0" if args.cuda else "cpu"

    # general configs for OpenAI gym env
    env = BaseConfig()
    env.time_limit = 49.25
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 100
    # if randomize human behaviors, set to True, else set to False
    env.randomize_attributes = True
    env.num_processes = args.num_processes
    # record robot states and actions an episode for system identification in sim2real
    env.record = False
    env.load_act = False

    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99
    
    reward.group_safety_buffer = 0.1
    reward.discomfort_group_dist = 0.35
    reward.discomfort_grp_penalty_factor = 10
    reward.grp_collision_penalty = -5
    reward.use_garn_reward = False
    
    # config for Groups
    group = BaseConfig()
    group.num_groups = 3
    group.min_size = 3
    group.max_size = 3
    group.min_distance = 2.0
    group.min_radius = 1.0
    group.max_radius = 1.3
    group.dynamic = True
    group.ground_truth = True
    group.types = ['static_f', 'dynamic_lf', 'dynamic_free']
    group.avoid_action = False
    group.num_on_path = 2

    # config for realistic pedestrian / group modeling
    realistic = BaseConfig()
    realistic.enabled = True
    realistic.use_speed_variation = True
    realistic.use_group_speed_factor = True
    realistic.use_f_formations = True
    realistic.use_leader_follower = True
    realistic.use_convex_hull = True
    realistic.individual_speed_mean = 1.34
    realistic.individual_speed_std  = 0.26
    realistic.individual_speed_min  = 0.80
    realistic.individual_speed_max  = 1.80
    realistic.group_speed_factor = 0.85
    realistic.f_formation_radius = 0.65
    realistic.leader_follower_spacing = 0.70
    realistic.leader_follower_gain = 1.20
    realistic.hull_degenerate_buffer = 0.30

    # config for TAGA (Tangent Action for Group Avoidance)
    taga = BaseConfig()
    taga.smooth_switching = True
    taga.switch_band = 0.5
    taga.safe_margin = 0.6
    taga.goal_threshold = 2.5
    taga.use_scaled_zones = True
    taga.emergency_factor = 0.5
    taga.danger_factor = 0.833
    taga.caution_factor = 2.0
    taga.emergency_zone = 0.4
    taga.danger_zone = 0.6
    taga.caution_zone = 1.0
    taga.use_accel_limit = True
    taga.max_accel_factor = 0.5
    taga.cost_aware_side = True
    taga.look_ahead = 5.0
    taga.cone_half_angle = 60.0
    taga.w_goal = 0.4
    taga.w_obstacle = 0.6
    taga.multi_group = False
    taga.max_groups = 3
    taga.intent_based = True
    taga.intent_lookahead = 0.7
    taga.intent_margin = 0.0
    taga.safety_filter = True
    taga.safety_lookahead = 1.0
    taga.safety_radius = 0.55
    taga.safety_damping = 0.0
    taga.safety_horizons = [0.3, 0.7, 1.0, 1.5, 2.0]
    taga.safety_alphas = [1.0, 0.7, 0.4, 0.2, 0.0]
    taga.debug_log = True
    taga.hull_safety_filter = False
    taga.hull_safety_margin = 0.15
    taga.hull_safety_horizons = [0.3, 0.7, 1.0]
    taga.direction_guard = False
    taga.direction_max_angle = 30.0
    taga.anti_vel_dynamic = False
    taga.anti_vel_min_speed = 0.1
    taga.anti_vel_max_with_goal = 0.5
    taga.anti_vel_w_anti = 0.3
    taga.anti_vel_w_goal = 0.7
    taga.anti_vel_cone_angle = 45.0
    taga.anti_vel_pause_radius = 1.2
    taga.cone_ttc_check = True
    taga.cone_ttc_horizon = 0.7
    taga.cone_ttc_radius = 0.55
    taga.max_consecutive_pause = 3
     
    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 8.5
    sim.arena_size = 8.5
    sim.human_num = 20
    sim.has_individuals = True
    sim.has_groups = True
    # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    sim.human_num_range = 0
    sim.predict_steps = 5
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'inferred'
    # render the simulation during training or not
    sim.render = False

    # for save_traj only
    render_traj = False
    save_slides = False
    save_path = None

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # human config
    humans = BaseConfig()
    humans.visible = True
    # orca or social_force for now
    # hybrid_orca_social_force
    # humans.policy = "hybrid_orca_flocking"
    humans.policy = "orca"
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False
    humans.displacement = True

    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = False
    # For baseline: srnn; our method: selfAttn_merge_srnn
    # robot.policy = 'selfAttn_merge_srnn'
    # social_force
    # zone_based
    robot.policy = 'selfAttn_merge_srnn'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2
    # radius of perception range
    robot.sensor_range = 5

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 0.25

    # config for the GST predictor
    pred = BaseConfig()
    # see 'gst_updated/results/README.md' for how to set this variable
    # If randomized humans: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
    # else: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj
    pred.model_dir = 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'

    # LIDAR config
    lidar = BaseConfig()
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 5
    # range in meters
    lidar.range = 10

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = True
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")
