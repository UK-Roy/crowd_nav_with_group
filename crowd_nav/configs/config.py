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
    env.time_limit = 49.25   # 49.25 / 0.25 = 197 steps
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
    # Stage 1–2: 0.07 (nearly touching — groups off-path, penalty factor is 0 anyway)
    # Stage 3+:  0.35 (35 cm early warning, same scale as individual discomfort_dist)
    reward.discomfort_group_dist = 0.35
    # Stage 1–2: 0   (group avoidance off; robot learns individual navigation first)
    # Stage 3+: 10   (same scale as individual discomfort_penalty_factor)
    reward.discomfort_grp_penalty_factor = 10

    # Stage 1–2: 0   (group intrusion not penalised during early curriculum)
    # Stage 3+: -5   (softer than collision_penalty=-20; doesn't end episode)
    reward.grp_collision_penalty = -5

    # whether to use GARN's group-related reward (R_grp) instead of default group reward
    reward.use_garn_reward = False

    # config for GARN reward function (Lu et al. RA-L 2025)
    garn = BaseConfig()
    garn.intrusion_penalty = -0.25     # Eq. 5: penalty per group intrusion per timestep
    garn.c1 = 1.0                      # Eq. 6: overtaking/following weight
    garn.c2 = 1.0                      # Eq. 8: cooperative passing weight
    garn.d_t1 = 3.0                    # Eq. 7: consideration range for groups (meters)

    # config for GRAM-v2 Phase 4 (end-to-end perception-aware navigation)
    # Activate by setting: robot.policy='gram_v2', sim.predict_method='none',
    # --env-name CrowdSimVarNum-v0, --human_node_rnn_size 256,
    # --human_human_edge_rnn_size 14
    gram_v2 = BaseConfig()
    gram_v2.phase2_checkpoint = 'trained_models/gram_v2/phase2_v2/best.pt'
    gram_v2.phase3_checkpoint = 'trained_models/gram_v2/phase3/best.pt'
    # False = train backbone end-to-end with navigation policy (Stage 1-2, 3a)
    # True  = freeze backbone, only train cross-attn+GRU+heads (Stage 3b+)
    gram_v2.freeze_backbone = False
    # False = skip SlotAttention, cross-attn only over human embeddings (Stage 1-2, 3a)
    # True  = include group slot prototypes in cross-attn kv (Stage 3b+)
    gram_v2.use_slots = False

    # config for GRACE (end-to-end cost-map navigation, rl/networks/grace_network.py)
    # Activate by setting: robot.policy='grace', same env flags as gram_v2
    grace = BaseConfig()
    grace.phase2_checkpoint = 'trained_models/gram_v2/phase2_v2/best.pt'
    grace.phase3_checkpoint = 'trained_models/gram_v2/phase3/best.pt'
    # Freeze backbone during Stage B (initial PPO); unfreeze for Stage C fine-tuning
    grace.freeze_backbone   = False
    # BEV grid: grid_size × grid_size cells covering 2×grid_range metres square
    # Default 32×32 at 6m → 0.375 m/cell.  Increase to 48 or 64 if VRAM allows.
    grace.grid_size  = 32
    grace.grid_range = 6.0
    # Temporal horizons for trajectory cost layers (seconds)
    grace.horizons   = [0.3, 0.7, 1.0, 1.5]
    # Self-supervised auxiliary occupancy loss
    # Stage B: False (frozen cost map, pure PPO)
    # Stage C: True  (joint PPO + aux loss)
    grace.use_aux_loss    = True
    grace.aux_loss_weight = 0.1      # λ: weight of aux loss relative to PPO loss

    # config for Groups
    group = BaseConfig()
    group.num_groups = 3        # benchmark: 3 groups

    group.min_size = 3
    group.max_size = 3          # Stage 2: slightly larger groups

    group.min_distance = 2.0

    group.min_radius = 1.0
    group.max_radius = 1.3

    group.dynamic = True
    group.ground_truth = True
    # Pool of group types assigned randomly per group at each reset.
    # 'static_f'    — stationary F-formation (Kendon 1990)
    # 'dynamic_lf'  — moving, followers track leader (Helbing & Molnar 1995)
    # 'dynamic_free'— moving, each member navigates independently (ORCA)
    group.types = ['static_f', 'dynamic_lf', 'dynamic_free']  # benchmark: all group types

    group.avoid_action = False

    # How many of the groups are placed along the robot→goal path to guarantee
    # the robot encounters them. Remaining groups are placed randomly.
    group.num_on_path = 2       # benchmark: 2 groups on robot's path

    # config for realistic pedestrian / group modeling (shared benchmark env)
    # Every sub-flag gates a discrete feature; defaults are *off* so trained
    # checkpoints load bit-exactly. Flip on individually as phases validate.
    realistic = BaseConfig()
    realistic.enabled = True                    # benchmark: all realistic phases on
    realistic.use_speed_variation = True        # Phase A: Weidmann 1992 v_pref
    realistic.use_group_speed_factor = True     # Phase B: Moussaid 2010 slowdown
    realistic.use_f_formations = True           # Phase C: Kendon 1990 (for static_f groups)
    realistic.use_leader_follower = True        # Phase D: Helbing & Molnar 1995 (for dynamic_lf groups)
    realistic.use_convex_hull = True            # Phase E: group geometry

    # Phase A — individual preferred-speed distribution (Weidmann 1992)
    realistic.individual_speed_mean = 1.34
    realistic.individual_speed_std  = 0.26
    realistic.individual_speed_min  = 0.80
    realistic.individual_speed_max  = 1.80

    # Phase B
    realistic.group_speed_factor = 0.85

    # Phase C
    realistic.f_formation_radius = 0.65

    # Phase D
    realistic.leader_follower_spacing = 0.70
    realistic.leader_follower_gain = 1.20

    # Phase E
    realistic.hull_degenerate_buffer = 0.30

    # config for TAGA (Tangent Action for Group Avoidance)
    taga = BaseConfig()
    # smooth (sigmoid) switching between base policy and tangent action
    # Set False to recover original hard-threshold behaviour
    taga.smooth_switching = True
    # half-width of the blending band around the activation threshold (m)
    taga.switch_band = 0.5
    # extra margin added to group radius to trigger TAGA (m) — center of the band
    taga.safe_margin = 0.6
    # robot ignores group avoidance within this distance to its goal (m)
    taga.goal_threshold = 2.5

    # Safety controller zones — expressed as multiples of (robot_radius + human_radius)
    # so they scale automatically with agent sizes.
    # With default radii (r_robot=r_human=0.3, sum=0.6):
    #   emergency = 0.5 × 0.6 = 0.30 m  ← matches paper's d_critical = 0.3 m
    #   danger    = 0.83 × 0.6 = 0.50 m ← matches paper's d_personal  = 0.5 m
    #   caution   = 2.0 × 0.6 = 1.20 m  ← outer awareness boundary
    # Set use_scaled_zones=False to fall back to the fixed metre values below.
    taga.use_scaled_zones  = True
    taga.emergency_factor  = 0.5     # × (r_robot + r_human) → d_critical (paper)
    taga.danger_factor     = 0.833   # × (r_robot + r_human) → d_personal  (paper)
    taga.caution_factor    = 2.0     # × (r_robot + r_human) → outer boundary
    # Fallback fixed values (used when use_scaled_zones=False)
    taga.emergency_zone = 0.4
    taga.danger_zone    = 0.6
    taga.caution_zone   = 1.0

    # Bounded-acceleration safety filter (P1)
    # Max velocity change per time_step as a fraction of v_pref.
    # Replaces hard emergency override with smooth velocity clipping.
    # Set use_accel_limit=False to restore old hard-override behaviour.
    taga.use_accel_limit   = True
    taga.max_accel_factor  = 0.5     # max |Δv| per step = factor × v_pref

    # cost-aware tangent side selection (P1)
    taga.cost_aware_side = True       # False = legacy smaller-angle rule
    taga.look_ahead = 5.0             # metres ahead to scan for obstacles
    taga.cone_half_angle = 60.0       # degrees: cone half-angle for obstacle scan
    taga.w_goal = 0.4                 # weight for goal-alignment cost
    taga.w_obstacle = 0.6             # weight for obstacle-density cost
    # multi-group aggregation (P2)
    # True  = weighted-average tangent across ALL blocking groups
    # False = legacy first-match-wins (break after first triggering group)
    taga.multi_group = False
    taga.max_groups  = 3              # max blocking groups to consider per step

    # Intent gate ON: TAGA only fires when the base policy's action would enter a
    # group hull. This is the "base fails due to groups" case your design targets.
    # Keeps the base-success guarantee — TAGA can only help, not hurt.
    # GCR reduction via soft proximity activation is a separate feature (TODO).
    taga.intent_based        = True
    taga.intent_lookahead    = 0.7
    taga.intent_margin       = 0.0

    # P3: Safety filter — validate the TAGA-blended action against individual humans.
    # If a collision is predicted, damp TAGA's influence (alpha) toward the base policy.
    taga.safety_filter       = True
    taga.safety_lookahead    = 1.0   # (legacy) single-horizon lookahead — kept for compat
    taga.safety_radius       = 0.55  # metres: collision threshold (r_robot + r_human + buffer)
    taga.safety_damping      = 0.0   # (legacy) multiply alpha when collision predicted

    # P3 Option A: multi-horizon safety filter — checks collision at each time below
    # Captures both fast (short horizon) and slow (long horizon) approaching humans.
    taga.safety_horizons     = [0.3, 0.7, 1.0, 1.5, 2.0]

    # P3 Option B: iterative alpha search — tries these alpha values in order (high→low),
    # picks the highest one that doesn't collide. Preserves TAGA benefit when possible,
    # falls back to 0.0 (pure base policy) only when all higher alphas fail.
    taga.safety_alphas       = [1.0, 0.7, 0.4, 0.2, 0.0]

    # P3: Debug logging — prints per-episode TAGA activity summary
    taga.debug_log           = True

    # Hull-aware safety filter (Exp 06): reject TAGA if the blended action
    # would enter/graze any group hull MORE than the base action would.
    # Goal: drive GCR down without sacrificing SR — TAGA only commits when it
    # strictly reduces hull intrusion vs base. Translates each hull by
    # V_group * t to predict where it will be at each horizon.
    taga.hull_safety_filter   = False
    taga.hull_safety_margin   = 0.15   # m: extra clearance from hull boundary
    taga.hull_safety_horizons = [0.3, 0.7, 1.0]

    # Direction-shift guard (Exp 08): reject TAGA's blended action if its
    # direction differs from base_action by more than direction_max_angle.
    # Rationale: at base SR = 0.84, most failures TAGA could fix happen when
    # base is already aiming somewhat in the right direction — small tangent
    # corrections are useful, big redirections derail otherwise-OK trajectories.
    taga.direction_guard      = False
    taga.direction_max_angle  = 30.0   # degrees

    # Anti-velocity tangent for dynamic groups (new):
    # For dynamic_lf / dynamic_free groups, pick the CW/CCW tangent that best
    # combines (a) alignment with -V_group (escape opposite to group travel) and
    # (b) alignment with goal_dir (still progress toward the goal). Goal-blended
    # so we don't backpedal when a group walks toward the same goal.
    # Then scan a cone in that direction: if any individual is on a collision
    # course (TTC-based, not just in-cone), stand still instead of colliding.
    # Bounded pause budget prevents the robot from freezing forever.
    taga.anti_vel_dynamic     = False  # OFF: regressed ORCA; cost-aware tangent works better
    taga.anti_vel_min_speed   = 0.1    # m/s: min group speed to apply anti-vel rule
    # If V_group · goal_dir > this threshold, the group is walking with us toward
    # the goal — anti-velocity would point backward, so fall back to cost-aware.
    taga.anti_vel_max_with_goal = 0.5
    taga.anti_vel_w_anti      = 0.3    # weight on -V_group alignment in tangent score
    taga.anti_vel_w_goal      = 0.7    # weight on goal_dir alignment in tangent score
    taga.anti_vel_cone_angle  = 45.0   # degrees: half-angle of escape-path cone scan
    taga.anti_vel_pause_radius = 1.2   # m: gate distance for cone scan
    # TTC-aware cone-pause: pause only when individual is on a collision course.
    taga.cone_ttc_check       = True   # use TTC instead of just-in-cone presence
    taga.cone_ttc_horizon     = 0.7    # s: lookahead horizon for closest-approach
    taga.cone_ttc_radius      = 0.55   # m: collision-course distance threshold
    # Bounded pause budget — robot never pauses more than N consecutive steps.
    taga.max_consecutive_pause = 3

    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 8.5
    sim.arena_size = 8.5
    sim.human_num = 20      # benchmark: full 20-human setting
    # Composition toggles. True/True = mixed (default, legacy behaviour).
    # True/False = individuals only. False/True = groups only (forces every
    # human into a group; clip human_num to total group capacity).
    sim.has_individuals = True
    sim.has_groups = True       # must be True — frozen backbone needs groups to produce useful embeddings
    # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    sim.human_num_range = 0
    sim.predict_steps = 5
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'none'
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
    humans.policy = "social_force"
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

    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = False
    # For baseline: srnn; another method: selfAttn_merge_srnn
    # our method robot.policy = 'selfAttn_merge_srnn'
    # GARN baseline: 'garn'; GRAM-v2: 'gram_v2'; GRACE: 'grace'
    robot.policy = 'grace'
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
