import torch


class TAGASafetyController:
    """Continuous-blend safety layer for TAGA.

    Unlike the original 3-zone hard-switching controller, this version smoothly
    interpolates between the TAGA tangent action and a repulsive safety action
    using a continuous weight w ∈ [0, 1] derived from the distance to the
    nearest individual human. This removes the step-change artefacts that
    caused the motion-discontinuity issues flagged in IROS review R2-Q4.
    """

    def __init__(self, config):
        self.config = config
        taga_cfg = config.taga
        self.emergency_zone = taga_cfg.emergency_zone
        self.danger_zone = taga_cfg.danger_zone
        self.caution_zone = taga_cfg.caution_zone
        self.robot_radius = config.robot.radius
        self.human_radius = config.humans.radius
        self.v_pref = config.robot.v_pref

    def _safety_weight(self, d_min):
        """Continuous w(d): 0 outside caution_zone, 1 at/below emergency_zone,
        linear ramp in between."""
        if d_min >= self.caution_zone:
            return 0.0
        if d_min <= self.emergency_zone:
            return 1.0
        return (self.caution_zone - d_min) / (self.caution_zone - self.emergency_zone)

    def _min_distance(self, obs):
        """Return min clearance to any visible human, and its relative position."""
        min_distance = float('inf')
        closest_pos = None
        spatial_edges = obs['spatial_edges'][0]
        visible_masks = obs['visible_masks'][0]

        for i in range(len(visible_masks)):
            if visible_masks[i] and spatial_edges[i, 0] != 15:
                rel = spatial_edges[i, :2]
                d = torch.norm(rel) - self.robot_radius - self.human_radius
                if d < min_distance:
                    min_distance = d.item()
                    closest_pos = rel
        return min_distance, closest_pos

    def _repulsive_force(self, spatial_edges, visible_masks, device):
        """Sum repulsive forces from every human inside caution_zone.
        Magnitude per human grows linearly as (caution - d) / caution."""
        force = torch.zeros(2, device=device)
        for i in range(len(visible_masks)):
            if visible_masks[i] and spatial_edges[i, 0] != 15:
                rel = spatial_edges[i, :2]
                d = torch.norm(rel) - self.robot_radius - self.human_radius
                if d < self.caution_zone:
                    mag = (self.caution_zone - d) / self.caution_zone
                    force = force + (-rel / torch.norm(rel)) * mag
        return force

    def _safety_action(self, robot_pos, goal_pos, spatial_edges, visible_masks, w, device):
        """Build a safety action = w·repulsive + (1 - 0.5·w)·goal_dir.

        Keeping partial goal attraction even at high w prevents the "push into
        wall / other group" failure mode of pure emergency_avoid.
        """
        repulsive = self._repulsive_force(spatial_edges, visible_masks, device)

        goal_dir = goal_pos - robot_pos
        goal_norm = torch.norm(goal_dir)
        if goal_norm > 0:
            goal_dir = goal_dir / goal_norm

        combined = w * repulsive + (1.0 - 0.5 * w) * goal_dir
        n = torch.norm(combined)
        if n > 0:
            return combined / n * self.v_pref
        return goal_dir * self.v_pref

    def get_safe_taga_action(self, obs, taga_action, device):
        """Blend TAGA with safety using a continuous weight.

        w = 0  → return pure TAGA (no nearby humans)
        w = 1  → return pure safety action (a human is at/inside emergency zone)
        0 < w < 1 → smooth interpolation, no step changes.
        """
        robot_pos = torch.tensor(
            [obs['robot_node'][0, 0, 0], obs['robot_node'][0, 0, 1]], device=device
        )
        goal_pos = torch.tensor(
            [obs['robot_node'][0, 0, 3], obs['robot_node'][0, 0, 4]], device=device
        )

        min_dist, _ = self._min_distance(obs)
        w = self._safety_weight(min_dist)

        if w == 0.0:
            return taga_action

        safety_action = self._safety_action(
            robot_pos, goal_pos,
            obs['spatial_edges'][0], obs['visible_masks'][0],
            w, device,
        )

        blended = (1.0 - w) * taga_action + w * safety_action
        n = torch.norm(blended)
        if n > 0:
            return blended / n * self.v_pref
        return safety_action
