import numpy as np
import torch
from crowd_sim.envs.utils.action import ActionXY

class TAGASafetyController:
    def __init__(self, config):
        self.config = config
        # Safety zones (in meters)
        self.EMERGENCY_ZONE = 0.4  # Immediate collision risk
        self.DANGER_ZONE = 0.6     # High risk zone  
        self.CAUTION_ZONE = 1.0    # Safety buffer
        self.robot_radius = config.robot.radius
        self.human_radius = config.humans.radius
        self.v_pref = config.robot.v_pref
        
    def check_individual_safety(self, robot_pos, obs, device):
        """Check distances to all visible humans"""
        min_distance = float('inf')
        closest_human_idx = -1
        closest_human_pos = None
        
        # Check spatial edges for visible humans
        spatial_edges = obs['spatial_edges'][0]  # Remove batch dimension
        visible_masks = obs['visible_masks'][0]
        
        for i in range(len(visible_masks)):
            if visible_masks[i] and spatial_edges[i, 0] != 15:  # Not dummy human
                # Human position relative to robot
                human_rel_pos = spatial_edges[i, :2]
                distance = torch.norm(human_rel_pos) - self.robot_radius - self.human_radius
                
                if distance < min_distance:
                    min_distance = distance.item()
                    closest_human_idx = i
                    closest_human_pos = human_rel_pos
        
        return min_distance, closest_human_idx, closest_human_pos
    
    def emergency_avoid(self, closest_human_pos, device):
        """Emergency collision avoidance"""
        # Move directly away from closest human
        if closest_human_pos is not None:
            avoid_direction = -closest_human_pos / torch.norm(closest_human_pos)
            action = avoid_direction * self.v_pref
            return action
        return torch.zeros(2, device=device)
    
    def compute_safety_action(self, robot_pos, goal_pos, spatial_edges, visible_masks, min_dist, device):
        """Compute repulsive forces from nearby humans"""
        repulsive_force = torch.zeros(2, device=device)
        
        for i in range(len(visible_masks)):
            if visible_masks[i] and spatial_edges[i, 0] != 15:
                human_rel_pos = spatial_edges[i, :2]
                distance = torch.norm(human_rel_pos) - self.robot_radius - self.human_radius
                
                if distance < self.CAUTION_ZONE:
                    # Repulsive force inversely proportional to distance
                    force_magnitude = (self.CAUTION_ZONE - distance) / self.CAUTION_ZONE
                    repulsion_direction = -human_rel_pos / torch.norm(human_rel_pos)
                    repulsive_force += repulsion_direction * force_magnitude * 2.0
        
        # Add goal attraction
        goal_direction = goal_pos - robot_pos
        if torch.norm(goal_direction) > 0:
            goal_direction = goal_direction / torch.norm(goal_direction)
        
        # Blend forces (prioritize safety in danger zone)
        if min_dist < self.DANGER_ZONE:
            combined = 0.7 * repulsive_force + 0.3 * goal_direction
        else:
            combined = 0.4 * repulsive_force + 0.6 * goal_direction
        
        # Normalize to preferred speed
        if torch.norm(combined) > 0:
            return combined / torch.norm(combined) * self.v_pref
        return goal_direction * self.v_pref
    
    def get_safe_taga_action(self, obs, taga_action, device):
        """Apply safety layer to TAGA action"""
        # Get robot and goal positions
        robot_pos = torch.tensor([obs['robot_node'][0, 0, 0], 
                                  obs['robot_node'][0, 0, 1]], device=device)
        goal_pos = torch.tensor([obs['robot_node'][0, 0, 3], 
                                 obs['robot_node'][0, 0, 4]], device=device)
        
        # Check safety
        min_dist, closest_idx, closest_pos = self.check_individual_safety(robot_pos, obs, device)
        
        if min_dist < self.EMERGENCY_ZONE:
            # Override TAGA - emergency avoidance
            return self.emergency_avoid(closest_pos, device)
        
        elif min_dist < self.DANGER_ZONE:
            # Blend TAGA with safety
            safety_action = self.compute_safety_action(
                robot_pos, goal_pos, 
                obs['spatial_edges'][0], 
                obs['visible_masks'][0], 
                min_dist, device
            )
            # Weighted blend: prioritize safety
            blended = 0.3 * taga_action + 0.7 * safety_action
            return blended / torch.norm(blended) * self.v_pref if torch.norm(blended) > 0 else safety_action
        
        elif min_dist < self.CAUTION_ZONE:
            # Light safety blend with TAGA
            safety_action = self.compute_safety_action(
                robot_pos, goal_pos,
                obs['spatial_edges'][0],
                obs['visible_masks'][0],
                min_dist, device
            )
            blended = 0.8 * taga_action + 0.2 * safety_action
            return blended / torch.norm(blended) * self.v_pref if torch.norm(blended) > 0 else taga_action
        
        # Safe zone - use pure TAGA
        return taga_action
