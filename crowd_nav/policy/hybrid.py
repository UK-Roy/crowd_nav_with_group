import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
import rvo2  # for ORCA

class HYBRID_ORCA_SOCIAL_FORCE(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'hybrid_orca_social_force'
        self.max_neighbors = None
        self.radius = None
        self.max_speed = 1  # The ego agent assumes that all other agents have this max speed
        self.sim = None
        self.safety_space = self.config.orca.safety_space
        self.A = self.config.sf.A  # Interaction strength for social force
        self.B = self.config.sf.B  # Interaction range for social force
        self.KI = self.config.sf.KI  # Inverse relocation time constant for social force

    def predict(self, state):
        """
        Produce action by combining ORCA for safety and Social Force for human-like interaction dynamics.
        """
        # ORCA part
        self_state = state.self_state
        self.max_neighbors = len(state.human_states)
        self.radius = state.self_state.radius
        params = self.config.orca.neighbor_dist, self.max_neighbors, self.config.orca.time_horizon, self.config.orca.time_horizon_obst

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self.sim.addAgent((self_state.px, self_state.py), *params, self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, (self_state.vx, self_state.vy))
            for human_state in state.human_states:
                self.sim.addAgent((human_state.px, human_state.py), *params,
                                  human_state.radius + 0.01 + self.safety_space, self.max_speed, (human_state.vx, human_state.vy))
        else:
            self.sim.setAgentPosition(0, (self_state.px, self_state.py))
            self.sim.setAgentVelocity(0, (self_state.vx, self_state.vy))
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, (human_state.px, human_state.py))
                self.sim.setAgentVelocity(i + 1, (human_state.vx, human_state.vy))

        # Set preferred velocity for ORCA
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))  # Unknown goals for other humans

        # Run one ORCA step
        self.sim.doStep()
        orca_action = np.array(self.sim.getAgentVelocity(0))

        # Social Force part
        # Pull force to goal
        delta_x = self_state.gx - self_state.px
        delta_y = self_state.gy - self_state.py
        dist_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2)
        desired_vx = (delta_x / dist_to_goal) * self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * self_state.v_pref
        curr_delta_vx = self.KI * (desired_vx - self_state.vx)
        curr_delta_vy = self.KI * (desired_vy - self_state.vy)

        # Push force from other agents
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            delta_x = self_state.px - other_human_state.px
            delta_y = self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x ** 2 + delta_y ** 2)
            interaction_vx += self.A * np.exp((self_state.radius + other_human_state.radius - dist_to_human) / self.B) * (delta_x / dist_to_human)
            interaction_vy += self.A * np.exp((self_state.radius + other_human_state.radius - dist_to_human) / self.B) * (delta_y / dist_to_human)

        # Combine ORCA velocity with Social Force adjustments
        combined_vx = orca_action[0] + (curr_delta_vx + interaction_vx) * self.config.env.time_step
        combined_vy = orca_action[1] + (curr_delta_vy + interaction_vy) * self.config.env.time_step

        # Clip speed to ensure it does not exceed the agent's preferred speed
        act_norm = np.linalg.norm([combined_vx, combined_vy])
        if act_norm > self_state.v_pref:
            return ActionXY(combined_vx / act_norm * self_state.v_pref, combined_vy / act_norm * self_state.v_pref)
        else:
            return ActionXY(combined_vx, combined_vy)
    
