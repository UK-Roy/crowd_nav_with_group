import numpy as np
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class HYBRID_ORCA_FLOCKING(Policy):
    def __init__(self, config):
        super().__init__(config)
        # self.orca_policy = policy_factory['orca'](config)
        # self.orca_policy = orca_policy
        # Hyperparameters for flocking
        self.w_c = 0.5  # cohesion
        self.w_a = 1.0  # alignment
        self.w_s = 1.0  # separation
        self.alpha = 0.5  # ORCA vs flocking blending

    def predict(self, state, orca_action, group_members):
        self_state = state.self_state
        # Get ORCA action using your ORCA policy
        # orca_action = self.orca_policy.predict(state)
        orca_vel = np.array([orca_action.vx, orca_action.vy])

        # If not in group, fallback to ORCA only
        # if getattr(self_state, 'group_id', None) is None:
        #     return orca_action

        # Identify group members (exclude self)
        # group_members = [
        #     h for h in state.human_states
        #     if getattr(h, 'group_id', None) == self_state.group_id and h.id != self_state.id
        # ]
        # if not group_members:
        #     return orca_action  # fallback: alone in group

        # --- Flocking calculations ---
        positions = np.array([[h.px, h.py] for h in group_members])
        velocities = np.array([[h.vx, h.vy] for h in group_members])

        # Cohesion: steer toward centroid
        centroid = positions.mean(axis=0)
        cohesion = centroid - np.array([self_state.px, self_state.py])
        if np.linalg.norm(cohesion) > 1e-6:
            cohesion /= np.linalg.norm(cohesion)
        else:
            cohesion = np.zeros(2)

        # Alignment: match mean velocity
        alignment = velocities.mean(axis=0)
        if np.linalg.norm(alignment) > 1e-6:
            alignment /= np.linalg.norm(alignment)
        else:
            alignment = np.zeros(2)

        # Separation: avoid crowding
        separation = np.zeros(2)
        min_dist = 0.6  # adjust as needed
        for h in group_members:
            diff = np.array([self_state.px, self_state.py]) - np.array([h.px, h.py])
            dist = np.linalg.norm(diff)
            if dist < min_dist and dist > 1e-3:
                separation += diff / dist

        if np.linalg.norm(separation) > 1e-6:
            separation /= np.linalg.norm(separation)

        # Weighted flocking vector
        flocking_vec = (
            self.w_c * cohesion +
            self.w_a * alignment +
            self.w_s * separation
        )
        if np.linalg.norm(flocking_vec) > 1e-6:
            flocking_vec /= np.linalg.norm(flocking_vec)

        # Blend ORCA and flocking
        blended_vel = self.alpha * orca_vel + (1 - self.alpha) * flocking_vec
        norm = np.linalg.norm(blended_vel)
        if norm > self_state.v_pref:
            blended_vel = blended_vel / norm * self_state.v_pref
        return ActionXY(*blended_vel)
