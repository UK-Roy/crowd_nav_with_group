import torch


class TAGASafetyController:
    """Smooth safety layer for TAGA.

    Two improvements over the original 3-zone hard-switching controller:

    P1-a (bounded acceleration):
        Instead of hard-overriding TAGA with a repulsive action in the EMERGENCY
        zone (which causes velocity jumps), we clamp the per-step velocity change
        to max_accel = max_accel_factor × v_pref. This guarantees continuity while
        still steering away from imminent collisions.

    P1-b (scaled zones):
        Safety zone thresholds are expressed as multiples of (r_robot + r_human)
        so they automatically scale with agent sizes rather than being magic numbers.
    """

    def __init__(self, config):
        self.config     = config
        self.v_pref     = config.robot.v_pref
        self.time_step  = config.env.time_step
        taga_cfg        = config.taga

        r_sum = config.robot.radius + config.humans.radius

        if taga_cfg.use_scaled_zones:
            self.emergency_zone = taga_cfg.emergency_factor * r_sum
            self.danger_zone    = taga_cfg.danger_factor    * r_sum
            self.caution_zone   = taga_cfg.caution_factor   * r_sum
        else:
            self.emergency_zone = taga_cfg.emergency_zone
            self.danger_zone    = taga_cfg.danger_zone
            self.caution_zone   = taga_cfg.caution_zone

        self.use_accel_limit  = taga_cfg.use_accel_limit
        # max velocity change per step
        self.max_accel        = taga_cfg.max_accel_factor * self.v_pref

        # track previous action for acceleration clamping
        self._prev_action = None

    def _safety_weight(self, d_min):
        """Continuous w(d): 0 outside caution_zone, 1 at/below emergency_zone."""
        if d_min >= self.caution_zone:
            return 0.0
        if d_min <= self.emergency_zone:
            return 1.0
        return (self.caution_zone - d_min) / (self.caution_zone - self.emergency_zone)

    def _min_distance(self, obs):
        """Return min clearance to any visible human."""
        min_dist    = float('inf')
        closest_pos = None
        spatial     = obs['spatial_edges'][0]
        masks       = obs['visible_masks'][0]
        for i in range(len(masks)):
            if masks[i] and spatial[i, 0] != 15:
                rel = spatial[i, :2]
                d   = torch.norm(rel).item() - self.config.robot.radius - self.config.humans.radius
                if d < min_dist:
                    min_dist    = d
                    closest_pos = rel
        return min_dist, closest_pos

    def _repulsive_force(self, spatial, masks, device):
        """Aggregate repulsive force from every human inside caution_zone."""
        force = torch.zeros(2, device=device)
        for i in range(len(masks)):
            if masks[i] and spatial[i, 0] != 15:
                rel = spatial[i, :2]
                d   = torch.norm(rel).item() - self.config.robot.radius - self.config.humans.radius
                if d < self.caution_zone:
                    mag   = (self.caution_zone - d) / self.caution_zone
                    norm  = torch.norm(rel) + 1e-9
                    force = force + (-rel / norm) * mag
        return force

    def _safety_action(self, robot_pos, goal_pos, spatial, masks, w, device):
        """Blend repulsive force with goal direction. Partial goal kept even at w=1
        to avoid pushing robot into walls or other groups."""
        repulsive = self._repulsive_force(spatial, masks, device)
        goal_dir  = goal_pos - robot_pos
        goal_norm = torch.norm(goal_dir)
        if goal_norm > 0:
            goal_dir = goal_dir / goal_norm
        combined = w * repulsive + (1.0 - 0.5 * w) * goal_dir
        n = torch.norm(combined)
        if n > 0:
            return combined / n * self.v_pref
        return goal_dir * self.v_pref

    def _clamp_acceleration(self, desired, device):
        """Clamp velocity change to max_accel per step (bounded-acceleration filter).
        Replaces hard emergency override with a smooth velocity constraint."""
        if self._prev_action is None or not self.use_accel_limit:
            self._prev_action = desired
            return desired

        delta = desired - self._prev_action
        delta_norm = torch.norm(delta)
        if delta_norm > self.max_accel:
            delta = delta / delta_norm * self.max_accel

        clamped = self._prev_action + delta

        # Still respect v_pref speed limit
        speed = torch.norm(clamped)
        if speed > self.v_pref:
            clamped = clamped / speed * self.v_pref

        self._prev_action = clamped
        return clamped

    def reset(self):
        """Call at episode start to clear velocity history."""
        self._prev_action = None

    def get_safe_taga_action(self, obs, taga_action, device):
        """Blend TAGA with safety, then apply bounded-acceleration filter.

        w = 0  → pure TAGA (no nearby humans)
        w = 1  → full safety blend (human inside emergency zone)
        Acceleration clamp ensures no step-to-step velocity jumps.
        """
        if not isinstance(taga_action, torch.Tensor):
            taga_action = torch.tensor(taga_action, dtype=torch.float32, device=device)

        robot_pos = torch.tensor(
            [obs['robot_node'][0, 0, 0], obs['robot_node'][0, 0, 1]], device=device)
        goal_pos  = torch.tensor(
            [obs['robot_node'][0, 0, 3], obs['robot_node'][0, 0, 4]], device=device)

        min_dist, _ = self._min_distance(obs)
        w           = self._safety_weight(min_dist)

        if w == 0.0:
            result = taga_action
        else:
            safety = self._safety_action(
                robot_pos, goal_pos,
                obs['spatial_edges'][0], obs['visible_masks'][0],
                w, device)
            blended = (1.0 - w) * taga_action + w * safety
            n = torch.norm(blended)
            result = blended / n * self.v_pref if n > 0 else safety

        # P1-a: bounded-acceleration filter applied to final result
        result = self._clamp_acceleration(result, device)
        return result
