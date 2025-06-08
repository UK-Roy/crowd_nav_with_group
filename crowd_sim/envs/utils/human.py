from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY

import numpy as np

class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        self.config = config
        self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent
        self.id = None # the human's ID, used for collecting traj data
        self.observed_id = -1 # if it is observed, give it a tracking ID
        self.group_id = None  # Group ID, None means the human is not part of a group

    # Edited by me
    def set_group(self, group_id):
        self.group_id = group_id

    # ob: a list of observable states
    def act(self, ob, members=None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)

        if self.isObstacle is True:
            return ActionXY(0, 0)
        
        # Edited by me
        # if self.group_id is not None:
        #     return self.grp_policy.predict(state)

        action = self.policy.predict(state)

        if self.group_id is not None:
            action = self.grp_policy.predict(state, action, members)

        # Check if the human is part of a group (Edited)
        # if self.group_id is not None:
        #     group_members = [human for human in ob if human.group_id == self.group_id]
        #     # Add group dynamics
        #     action = self.policy.predict(state)
        #     group_dynamics_forces = self.compute_group_dynamics_forces(group_members)
        #     action.vx += group_dynamics_forces[0]
        #     action.vy += group_dynamics_forces[1]
        # else:
        #     # Act individually
        #     action = self.policy.predict(state)
        return action

    # ob: a joint state (ego agent's full state + other agents' observable states)
    def act_joint_state(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        action = self.policy.predict(ob)
        return action  
    
    # Modified by me
    def compute_group_dynamics_forces(self, group_members):
        # Cohesion: move towards the center of the group
        group_center = np.mean([human.get_position() for human in group_members], axis=0)
        cohesion_force = self.config.C_cohesion * (group_center - self.get_position())

        # Alignment: align velocity with the group
        group_velocity = np.mean([human.get_velocity() for human in group_members], axis=0)
        alignment_force = self.config.C_alignment * (group_velocity - self.get_velocity())

        # Separation: avoid getting too close to other group members
        separation_force = np.zeros(2)
        for member in group_members:
            if np.linalg.norm(self.get_position() - member.get_position()) < self.config.separation_distance:
                separation_force += self.config.C_separation * (
                    self.get_position() - member.get_position()) / np.linalg.norm(self.get_position() - member.get_position()) ** 2

        return cohesion_force + alignment_force + separation_force
    
    def follow_leader_avoid_other(self, ob, leader):
        state = JointState(self.get_full_state(), ob)
        # Pull force to goal
        delta_x = leader.px - state.self_state.px
        delta_y = leader.py - state.self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref
        KI = self.config.sf.KI # Inverse of relocation time K_i
        curr_delta_vx = KI * (desired_vx - state.self_state.vx)
        curr_delta_vy = KI * (desired_vy - state.self_state.vy)
        
        # Push force(s) from other agents
        A = self.config.sf.A # Other observations' interaction strength: 1.5
        B = self.config.sf.B # Other observations' interaction range: 1.0
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            interaction_vx += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_x / dist_to_human)
            interaction_vy += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_y / dist_to_human)

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.config.env.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.config.env.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        new_vy = state.self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])

        if act_norm > state.self_state.v_pref:
            return ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(new_vx, new_vy)

# class Human(Agent):
#     # see Agent class in agent.py for details!!!
#     def __init__(self, config, section):
#         super().__init__(config, section)
#         self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent
#         self.id = None # the human's ID, used for collecting traj data
#         self.observed_id = -1 # if it is observed, give it a tracking ID
#         self.group_id = -1  # Added attribute: Group ID for group dynamics (-1 means no group)
#         self.group_cohesion_factor = config.sf.group_cohesion_factor  # Group cohesion factor for cohesion force

#     def act(self, ob):
#         """
#         The state for human is its full state and all other agents' observable states
#         Takes group dynamics into account if the human is part of a group.
#         :param ob: List of observable states (other agents).
#         :return: Action (movement) considering both personal goals and group dynamics.
#         """
#         state = JointState(self.get_full_state(), ob)

#         # Calculate the group cohesion force if the human is part of a group
#         if self.group_id != -1:
#             cohesion_force = self.calculate_group_cohesion_force(state)
#         else:
#             cohesion_force = np.array([0.0, 0.0])  # No group cohesion force

#         # Get the action based on policy prediction and add cohesion force
#         action = self.policy.predict(state)

#         # Modify the predicted action by adding the cohesion force (vector addition)
#         action.vx += cohesion_force[0]
#         action.vy += cohesion_force[1]

#         return action

#     def act_joint_state(self, ob):
#         """
#         The state for human is its full state and all other agents' observable states
#         Takes group dynamics into account if the human is part of a group.
#         :param ob: A joint state (ego agent's full state + other agents' observable states).
#         :return: Action (movement) considering both personal goals and group dynamics.
#         """
#         # Calculate the group cohesion force if the human is part of a group
#         if self.group_id != -1:
#             cohesion_force = self.calculate_group_cohesion_force(ob)
#         else:
#             cohesion_force = np.array([0.0, 0.0])  # No group cohesion force

#         # Get the action based on policy prediction and add cohesion force
#         action = self.policy.predict(ob)

#         # Modify the predicted action by adding the cohesion force (vector addition)
#         action.vx += cohesion_force[0]
#         action.vy += cohesion_force[1]

#         return action

#     def calculate_group_cohesion_force(self, state):
#         """
#         Calculate the group cohesion force to pull the human closer to other members of the same group.
#         This simulates a tendency for humans in the same group to stay together.
#         :param state: Joint state with information on other agents.
#         :return: Cohesion force (vx, vy) to adjust the agent's velocity.
#         """
#         cohesion_vx = 0.0
#         cohesion_vy = 0.0
#         group_members = [other for other in state.human_states if other.group_id == self.group_id]

#         # Sum of forces pulling towards each group member
#         for group_member in group_members:
#             delta_x = group_member.px - state.self_state.px
#             delta_y = group_member.py - state.self_state.py
#             dist_to_group_member = np.sqrt(delta_x ** 2 + delta_y ** 2)

#             # Apply the cohesion force (inverse distance to pull members closer)
#             if dist_to_group_member > 0:  # Avoid division by zero
#                 cohesion_vx += self.group_cohesion_factor * (delta_x / dist_to_group_member)
#                 cohesion_vy += self.group_cohesion_factor * (delta_y / dist_to_group_member)

#         # Average cohesion force (if more than one group member)
#         if len(group_members) > 0:
#             cohesion_vx /= len(group_members)
#             cohesion_vy /= len(group_members)

#         return np.array([cohesion_vx, cohesion_vy])