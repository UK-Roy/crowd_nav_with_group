import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class ZoneBasedGroupAvoidance(Policy):
    """Zone-based group avoidance - creates repulsive zones around groups"""
    def __init__(self, config):
        super().__init__(config)
        self.name = 'zone_based'
        self.time_step = config.env.time_step
        self.v_pref = config.robot.v_pref
        
        # Zone parameters
        self.individual_zone = 1.5  # meters
        self.group_zone_multiplier = 2.0  # groups get 2x larger zones

        # print(f"Zone based is initialized")
        
    def predict(self, state):
        # print(f"Currently I'm in zone based predit")
        if hasattr(state, 'self_state'):
            robot_state = state.self_state
            human_states = state.human_states
        else:
            # For list input from ORCA-style
            
            return self.predict_from_list(state)
        
        robot_pos = np.array([robot_state.px, robot_state.py])
        goal_pos = np.array([robot_state.gx, robot_state.gy])
        
        # ADD DEBUG HERE - After getting positions
        debug = False  # Set to True for debugging
        if debug:
            print(f"\n=== Zone-Based Debug ===")
            print(f"Robot at ({robot_state.px:.2f}, {robot_state.py:.2f})")
            print(f"Goal at ({robot_state.gx:.2f}, {robot_state.gy:.2f})")
            print(f"Number of humans: {len(human_states)}")
        
        # Goal attractive force
        goal_vec = goal_pos - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 0:
            goal_direction = goal_vec / goal_dist
        else:
            goal_direction = np.zeros(2)
        
        # Repulsive forces
        repulsion = np.zeros(2)
        
        # Detect groups (simple clustering)
        groups = self.detect_groups_simple(human_states)
        
        if debug:
            print(f"Detected {len(groups)} groups")
        
        # Apply zone-based repulsion
        for group in groups:
            if len(group) > 1:  # It's a group
                # Calculate group center
                group_center = np.mean([np.array([h.px, h.py]) for h in group], axis=0)
                group_radius = max([np.linalg.norm([h.px - group_center[0], 
                                h.py - group_center[1]]) for h in group])
                
                # Distance from robot to group boundary
                dist_to_center = np.linalg.norm(robot_pos - group_center)
                dist_to_boundary = dist_to_center - group_radius
                
                # FIX 1: Reduce repulsion strength
                zone_size = self.individual_zone * self.group_zone_multiplier
                if dist_to_boundary < zone_size:
                    # CHANGED: Reduced force multiplier from 2.0 to 0.5
                    force_magnitude = (zone_size - dist_to_boundary) / zone_size * 0.5
                    if dist_to_center > 0:
                        repulsion += (robot_pos - group_center) / dist_to_center * force_magnitude
                        
                if debug and dist_to_boundary < zone_size:
                    print(f"Group repulsion: dist_to_boundary={dist_to_boundary:.2f}, force={force_magnitude:.2f}")
                    
            else:
                # Individual human
                human = group[0]
                human_pos = np.array([human.px, human.py])
                dist = np.linalg.norm(robot_pos - human_pos)
                
                if dist < self.individual_zone and dist > 0:
                    # CHANGED: Reduced individual force
                    force_magnitude = (self.individual_zone - dist) / self.individual_zone * 0.3
                    repulsion += (robot_pos - human_pos) / dist * force_magnitude
        
        if debug:
            print(f"Total repulsion: {repulsion}")
            print(f"Goal direction: {goal_direction}")
        
        # FIX 2: Better force combination - prioritize goal
        # CHANGED: Increased goal weight, decreased repulsion weight
        action = goal_direction * robot_state.v_pref * 0.8 + repulsion * 0.2
        
        # FIX 3: Ensure minimum forward progress
        # If action is too small, add more goal direction
        if np.linalg.norm(action) < 0.1:
            action = goal_direction * robot_state.v_pref * 0.5
        
        # Normalize to v_pref
        speed = np.linalg.norm(action)
        if speed > robot_state.v_pref:
            action = action / speed * robot_state.v_pref
        
        if debug:
            print(f"Final action: {action}")
            print("=" * 30)
        
        return ActionXY(action[0], action[1])

    def predict_from_list(self, state):
        """Handle list-based state input (for ORCA compatibility)"""
        # Create a simple state object
        from crowd_sim.envs.utils.state import ObservableState, FullState
        
        # state is a list of ObservableState objects
        # First element is the robot
        if len(state) == 0:
            return ActionXY(0, 0)
        
        # For simple testing, just move toward goal
        return ActionXY(0.5, 0)  # Default safe action
    
    def detect_groups_simple(self, human_states, threshold=2.0):
        """Simple spatial clustering to detect groups"""
        groups = []
        assigned = []
        
        for i, h1 in enumerate(human_states):
            if i in assigned:
                continue
            group = [h1]
            assigned.append(i)
            
            for j, h2 in enumerate(human_states):
                if j in assigned or i == j:
                    continue
                dist = np.linalg.norm([h1.px - h2.px, h1.py - h2.py])
                if dist < threshold:
                    group.append(h2)
                    assigned.append(j)
            
            groups.append(group)
        
        return groups

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