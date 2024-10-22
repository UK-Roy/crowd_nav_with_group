import numpy as np
from crowd_sim.envs.utils.action import ActionXY

class Group:
    def __init__(self, members):
        self.members = members
        self.leader = members[0]  # First member is the leader by default
        self.followers = members[1:]  # The rest are followers

    def get_leader_action(self, ob):
        """
        Compute the leader's action based on the environment.
        """
        # Example action: Move towards goal or avoid obstacles
        return self.leader.act(ob)  # Use environment or policy-based action computation
    
    def get_follower_action(self, leader, follower, formation_offset):
        """
        Compute a follower's action based on the leader's action and formation constraints.
        
        Args:
            leader: The leader object.
            follower: The follower object.
            formation_offset: The desired position relative to the leader.
            
        Returns:
            ActionXY: The follower's action.
        """
        # Calculate the relative position the follower should maintain
        target_px = self.leader.px + formation_offset[0]
        target_py = self.leader.py + formation_offset[1]
        
        # Compute the action to move towards the desired position
        vx = target_px - follower.px
        vy = target_py - follower.py
        
        # Normalize the velocity to avoid large jumps
        norm_v = np.linalg.norm([vx, vy])
        if norm_v > follower.v_pref:
            vx = vx / norm_v * follower.v_pref
            vy = vy / norm_v * follower.v_pref
        
        return ActionXY(vx=vx, vy=vy)
    
    def maintain_group_cohesion(self, group):
        """
        Ensure that the group members stay cohesive and do not overlap.
        """
        group_center = np.mean([(member.px, member.py) for member in group.members], axis=0)
        max_distance = 1.5  # Max distance followers should stay from the leader
        
        for follower in group.followers:
            distance_to_leader = np.linalg.norm([follower.px - group.leader.px, follower.py - group.leader.py])
            if distance_to_leader > max_distance:
                # Adjust follower's action to move closer to the leader
                follower.vx, follower.vy = self.get_follower_action(group.leader, follower, [0.5, 0.5])
