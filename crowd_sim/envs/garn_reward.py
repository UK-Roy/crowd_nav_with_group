"""
GARN group-related reward R_grp (Lu et al. RA-L 2025, Eqs. 4-8).

R_grp has three additive components:
  - R_grp^intra (Eq. 5): penalty when the robot is inside a group's spatial boundary.
  - R_grp^ot/fw (Eq. 6): shaping term for overtaking/following groups that are in
    front of the robot and moving in the same direction.
  - R_grp^coop (Eq. 8): shaping term for cooperatively passing groups that approach
    the robot from its front.

The reward is intended as an *additive* shaping term on top of the base reward
(goal + collision + proximity), controlled by `config.reward.use_garn_reward`.
"""

import numpy as np


def _group_states_at_t(env):
    """
    Build per-group state arrays at the current simulation time t using the
    env's most recent observation (populated by the last generate_ob).

    Returns:
        centroids:  (G, 2) np.float32
        radii:      (G,)   np.float32 (approx. convex hull radius = max member distance from centroid)
        velocities: (G, 2) np.float32 (mean member velocity)
        prev_centroids: (G, 2) np.float32, estimated at t-1 from member velocities
    Empty arrays are returned when no groups are present.
    """
    ob = getattr(env, 'ob', None) or {}
    detected = ob.get('group_members', {}) or {}

    if not detected:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), empty, empty

    dt = env.time_step
    centroids, radii, velocities, prev_centroids = [], [], [], []
    for human_ids in detected.values():
        pos = np.array([[env.humans[i].px, env.humans[i].py] for i in human_ids],
                       dtype=np.float32)
        vel = np.array([[env.humans[i].vx, env.humans[i].vy] for i in human_ids],
                       dtype=np.float32)
        centroid = pos.mean(axis=0)
        radius = np.linalg.norm(pos - centroid, axis=1).max()
        mean_vel = vel.mean(axis=0)
        prev_pos = pos - vel * dt
        prev_centroid = prev_pos.mean(axis=0)

        centroids.append(centroid)
        radii.append(radius)
        velocities.append(mean_vel)
        prev_centroids.append(prev_centroid)

    return (np.asarray(centroids, dtype=np.float32),
            np.asarray(radii, dtype=np.float32),
            np.asarray(velocities, dtype=np.float32),
            np.asarray(prev_centroids, dtype=np.float32))


def compute_group_reward(env):
    """
    Compute the group-related reward R_grp for the current step.

    Contract: called from env.calc_reward BEFORE positions are updated, so
    env.robot / env.humans reflect state at time t; previous positions at
    t-1 are recovered from velocity * time_step.

    Returns:
        float: R_grp = R_intra + R_ot_fw + R_coop (0.0 if no groups).
    """
    garn_cfg = env.config.garn
    intrusion_penalty = garn_cfg.intrusion_penalty    # default -0.25
    c1 = garn_cfg.c1                                   # 1.0
    c2 = garn_cfg.c2                                   # 1.0
    d_frt = garn_cfg.d_t1                              # 3.0 m (consideration range)

    centroids, radii, group_vels, prev_centroids = _group_states_at_t(env)
    if centroids.shape[0] == 0:
        return 0.0

    dt = env.time_step
    v_pref = max(env.robot.v_pref, 1e-6)

    robot_pos = np.array([env.robot.px, env.robot.py], dtype=np.float32)
    robot_vel = np.array([env.robot.vx, env.robot.vy], dtype=np.float32)
    robot_prev_pos = robot_pos - robot_vel * dt

    # Δ(p_t, p_m^v) = p_t − p_m^v
    delta_curr = robot_pos[None, :] - centroids          # (G, 2)
    delta_prev = robot_prev_pos[None, :] - prev_centroids  # (G, 2)
    dist_curr = np.linalg.norm(delta_curr, axis=1)       # (G,)

    # --- (a) R_grp^intra (Eq. 5) -----------------------------------------
    # Indicator 1_{d_m}: robot is inside the group's spatial boundary.
    # Approximation: use the bounding radius (consistent with the rest of
    # the codebase's group collision logic).
    inside = (dist_curr < radii).astype(np.float32)
    outside = 1.0 - inside
    R_intra = float((intrusion_penalty * inside).sum())

    # Directional components require a defined robot heading.
    robot_speed = np.linalg.norm(robot_vel)
    if robot_speed < 1e-6:
        return R_intra

    robot_dir = robot_vel / robot_speed                  # unit vector (2,)

    # Classify groups:
    #   in_front   : group centroid lies ahead of robot along robot_dir
    #   same_dir   : group moves roughly the same way as the robot
    #   opp_dir    : group moves roughly opposite to the robot
    group_rel = centroids - robot_pos                    # (G, 2)
    in_front = (group_rel @ robot_dir) > 0               # (G,)
    vel_alignment = group_vels @ robot_dir               # (G,)
    same_dir = vel_alignment > 0
    opp_dir = vel_alignment < 0

    # ρ^m (Eq. 7): only reward groups within consideration range d_frt.
    within_range = (dist_curr <= d_frt).astype(np.float32)

    # Projected displacement onto robot heading:
    #   Eq. 6: [Δ(p_{t-1}, p_m^v) − Δ(p_t, p_m^v)] / v_pref · v_l/|v_l|
    #   Eq. 8: [Δ(p_t, p_m^v) − Δ(p_{t-1}, p_m^v)] / v_pref · v_l/|v_l|
    disp_proj_ot = ((delta_prev - delta_curr) / v_pref) @ robot_dir   # (G,)
    disp_proj_cp = ((delta_curr - delta_prev) / v_pref) @ robot_dir   # (G,)

    # M_1: same direction AND in front (AND outside the group space)
    M1_mask = (in_front & same_dir).astype(np.float32) * outside
    # M_2: approaching from front AND in front (AND outside the group space)
    M2_mask = (in_front & opp_dir).astype(np.float32) * outside

    R_ot_fw = float((M1_mask * c1 * within_range * disp_proj_ot).sum())
    R_coop = float((M2_mask * c2 * within_range * disp_proj_cp).sum())

    return R_intra + R_ot_fw + R_coop
