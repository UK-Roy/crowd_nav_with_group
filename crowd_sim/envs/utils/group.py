import numpy as np

class Group:
    def __init__(self, id, min_mem, max_mem):
        self.id = id
        self.members = []
        self.leader = None
        self.centroid = None
        self.radius = None
        self.max_member = np.random.randint(min_mem, max_mem+1)
    
    def config_group(self):
        self.leader = self.members[0] if self.members else None

    def set_centroid(self, px, py):
        self.centroid = [px, py]
    
    def set_radius(self, radius):
        self.radius = radius
    
    def add_member(self, human):
        if not self.check_validity():
            print(f"Members are limited")
            return
        self.members.append(human)
        self.config_group()  # Configure leader after adding first member
    
    def remove_member(self, mem_id):
        if mem_id in self.members:
            self.members.remove(mem_id)
            if mem_id == self.leader and self.members:
                self.leader = self.members[0]  # Update leader if needed

    def check_validity(self):
        return len(self.members) < self.max_member

    def get_centroid(self):
        if self.centroid:
            return self.centroid[0], self.centroid[1]
        return None

    def get_group(self):
        return {
            "id": self.id,
            "leader": self.leader,
            "members": self.members,
            "centroid": self.centroid,
            "radius": self.radius
        }

    def get_max_group_number(self):
        return self.max_member

    # Formation methods
    def circle_formation(self):
        """Arrange members in a circular formation around the centroid."""
        num_members = len(self.members)
        angle_interval = 2 * np.pi / num_members
        positions = []
        for i in range(num_members):
            angle = i * angle_interval
            px = self.centroid[0] + self.radius * np.cos(angle)
            py = self.centroid[1] + self.radius * np.sin(angle)
            positions.append((px, py))
        return positions

    def line_formation(self):
        """Arrange members in a line formation with respect to the centroid."""
        num_members = len(self.members)
        positions = []
        for i in range(num_members):
            px = self.centroid[0] + (i - num_members // 2) * self.radius
            py = self.centroid[1]
            positions.append((px, py))
        return positions

    def v_shape_formation(self):
        """Arrange members in a V-shaped formation around the centroid."""
        num_members = len(self.members)
        half_num = num_members // 2
        positions = []
        for i in range(num_members):
            row = abs(i - half_num)
            col = i if i < half_num else i - half_num
            px = self.centroid[0] + col * self.radius
            py = self.centroid[1] + row * self.radius * (-1 if i < half_num else 1)
            positions.append((px, py))
        return positions

    def grid_formation(self):
        """Arrange members in a small grid around the centroid."""
        num_members = len(self.members)
        grid_size = int(np.ceil(np.sqrt(num_members)))
        positions = []
        for i in range(num_members):
            row = i // grid_size
            col = i % grid_size
            px = self.centroid[0] + col * self.radius
            py = self.centroid[1] + row * self.radius
            positions.append((px, py))
        return positions
    
    def select_formation(self):
        """Randomly select and apply a formation to group members."""
        formations = [self.circle_formation, self.line_formation, self.v_shape_formation, self.grid_formation]
        formation_function = np.random.choice(formations)
        return formation_function()

    # def position_members(self, human_list):
    #     """Apply selected formation and set positions for each group member in the simulation environment."""
    #     positions = self.select_formation()
    #     for pos, mem_id in zip(positions, self.members):
    #         human = human_list[mem_id]
    #         px, py = pos
    #         human.set(px, py, -px, -py, 0, 0, human.radius)
    
    def position_members(self, robot, humans, min_distance=0.7):
        """
        Apply selected formation, set positions for each group member, and check for collisions.
        
        :param human_list: List of all humans in the environment.
        :param all_positions: List of positions of other agents to avoid collisions.
        :param min_distance: Minimum distance to avoid collisions.
        """
        # Generate positions based on the selected formation
        positions = self.select_formation()
        # positions = self.grid_formation()
        
        # Adjust positions if necessary to avoid collisions
        # adjusted_positions = []
        for pos, mem in zip(positions, self.members):
            px, py = pos
            collision = True

            # Try adjusting the position until there is no collision
            while collision:
                collision = False
                for i, agent in enumerate([robot] + humans):
                    
                    if np.linalg.norm((px - agent.px, py - agent.py)) < min_distance:
                        collision = True
                        # Apply random offset to try a new position (e.g., small shift within radius)
                        px += np.random.uniform(-0.2, 0.2)
                        py += np.random.uniform(-0.2, 0.2)
                        break  # Exit inner loop and recheck with updated position

            
            mem.set(px, py, -px, -py, 0, 0, mem.radius)  # Assuming `set` positions the human
            humans.append(mem)
            # Set the position in the human object
            # human = humans[mem.id]
            # human.set(px, py, -px, -py, 0, 0, human.radius)  # Assuming `set` positions the human
        
        return humans


