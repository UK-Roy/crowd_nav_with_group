import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class FFormationAvoidance(Policy):
    """F-formation aware navigation inspired by SEAN 2.0 (Tsoi et al., 2023)"""
    def __init__(self, config):
        super().__init__(config)
        self.name = 'f_formation'
        self.time_step = config.env.time_step
        self.v_pref = config.robot.v_pref
        
        # F-formation parameters
        self.o_space_radius = 1.5  # O-space (center) radius
        self.p_space_width = 0.5   # P-space (personal) width
        self.r_space_buffer = 1.0  # R-space (rear) buffer
        
    def predict(self, state):
        # print(f"Currently I'm in f_formation predict")
        if hasattr(state, 'self_state'):
            robot_state = state.self_state
            human_states = state.human_states
        else:
            return self.predict_from_list(state)
            
        robot_pos = np.array([robot_state.px, robot_state.py])
        goal_pos = np.array([robot_state.gx, robot_state.gy])
        
        # Goal force
        goal_vec = goal_pos - robot_pos
        if np.linalg.norm(goal_vec) > 0:
            goal_direction = goal_vec / np.linalg.norm(goal_vec)
        else:
            goal_direction = np.zeros(2)
        
        # Detect F-formations
        formations = self.detect_f_formations(human_states)
        
        # Calculate avoidance forces
        avoidance_force = np.zeros(2)
        
        for formation in formations:
            if len(formation['members']) > 1:
                # Calculate O-space (center of formation)
                o_space_center = formation['center']
                
                # Check if robot would intrude into O-space
                dist_to_center = np.linalg.norm(robot_pos - o_space_center)
                
                # F-formation has three zones:
                # 1. O-space (center) - never enter
                # 2. P-space (where people stand) - avoid
                # 3. R-space (behind people) - can pass through carefully
                
                if dist_to_center < self.o_space_radius:
                    # Strong repulsion from O-space
                    force = 3.0 * (self.o_space_radius - dist_to_center) / self.o_space_radius
                    if dist_to_center > 0:
                        avoidance_force += (robot_pos - o_space_center) / dist_to_center * force
                        
                elif dist_to_center < self.o_space_radius + self.p_space_width:
                    # Medium repulsion from P-space
                    force = 1.5
                    if dist_to_center > 0:
                        avoidance_force += (robot_pos - o_space_center) / dist_to_center * force
                        
                # Check if behind formation (R-space) - allow careful passage
                if self.is_behind_formation(robot_pos, formation):
                    # Reduce repulsion in R-space to allow passing
                    avoidance_force *= 0.3
        
        # Individual avoidance for non-formation humans
        for human in human_states:
            if not any(human in f['members'] for f in formations):
                human_pos = np.array([human.px, human.py])
                dist = np.linalg.norm(robot_pos - human_pos)
                if dist < 1.5 and dist > 0:
                    avoidance_force += (robot_pos - human_pos) / dist * (1.5 - dist)
        
        # Combine forces
        action = goal_direction * robot_state.v_pref + avoidance_force * 0.5
        
        # Normalize
        speed = np.linalg.norm(action)
        if speed > robot_state.v_pref:
            action = action / speed * robot_state.v_pref
            
        return ActionXY(action[0], action[1])
    
    def detect_f_formations(self, human_states):
        """Detect F-formations based on spatial arrangement and orientation"""
        formations = []
        assigned = []
        
        for i, h1 in enumerate(human_states):
            if i in assigned:
                continue
                
            # Check for potential F-formation partners
            formation = {'members': [h1], 'center': np.array([h1.px, h1.py])}
            
            for j, h2 in enumerate(human_states):
                if j in assigned or i == j:
                    continue
                    
                # F-formation criteria:
                # 1. Close proximity (< 3m)
                # 2. Facing toward common center (simplified here)
                dist = np.linalg.norm([h1.px - h2.px, h1.py - h2.py])
                
                if dist < 3.0:  # Potential F-formation
                    formation['members'].append(h2)
                    assigned.append(j)
            
            if len(formation['members']) > 1:
                # Calculate formation center (O-space)
                positions = np.array([[h.px, h.py] for h in formation['members']])
                formation['center'] = np.mean(positions, axis=0)
                formations.append(formation)
                assigned.append(i)
        
        return formations
    
    def is_behind_formation(self, robot_pos, formation):
        """Check if robot is in R-space (behind formation members)"""
        # Simplified: check if robot is opposite to where members are facing
        center = formation['center']
        members_positions = np.array([[h.px, h.py] for h in formation['members']])
        
        # If robot is further from center than members, likely in R-space
        robot_dist = np.linalg.norm(robot_pos - center)
        avg_member_dist = np.mean([np.linalg.norm(pos - center) for pos in members_positions])
        
        return robot_dist > avg_member_dist + self.p_space_width
    
    def clip_action(self, action, v_pref):
        """
        Clip action to respect velocity constraints
        Required for compatibility with the environment
        """
        if isinstance(action, np.ndarray):
            # Convert numpy array to ActionXY
            action = ActionXY(action[0], action[1])
        
        # Clip to preferred velocity
        velocity = np.array([action.vx, action.vy])
        speed = np.linalg.norm(velocity)
        
        if speed > v_pref:
            velocity = velocity / speed * v_pref
            return ActionXY(velocity[0], velocity[1])
        
        return action