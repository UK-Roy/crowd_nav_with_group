import numpy as np

class Group:
    def __init__(self, id, min_mem, max_mem):
        self.id = id
        self.members = []
        self.leader = None
        self.centroid = None
        self.radius = None
        self.max_member = np.random.randint(min_mem, max_mem+1)
        self.positioned = False  # True once position_members has been called
        # 'static_f', 'dynamic_lf', or 'dynamic_free' — assigned at reset time
        self.group_type = None
    
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
        """Randomly select and apply a formation to group members, with certain formations excluded for group 0."""
        formations = [self.circle_formation, self.v_shape_formation, self.grid_formation, self.line_formation]
        
        # Exclude line formation for group with ID 0
        if self.id == 0:
            formations.remove(self.line_formation)
        
        # Select and return the chosen formation function
        formation_function = np.random.choice(formations)
        return formation_function()
 
    def position_members(self, robot, humans, gap=0.15):
        """Position group members using a random formation, resolving spawn collisions.

        gap: extra clearance beyond sum-of-radii required between any two agents.
        """
        positions = self.select_formation()
        all_agents = [robot] + humans  # grows as each member is placed

        for pos, mem in zip(positions, self.members):
            px, py = pos
            for _ in range(300):
                collision = any(
                    np.linalg.norm([px - a.px, py - a.py]) < (mem.radius + a.radius + gap)
                    for a in all_agents if a.px is not None
                )
                if not collision:
                    break
                px += np.random.uniform(-0.4, 0.4)
                py += np.random.uniform(-0.4, 0.4)

            mem.set(px, py, -px, -py, 0, 0, mem.radius)
            humans.append(mem)
            all_agents.append(mem)

        self.positioned = True
            # Set the position in the human object
            # human = humans[mem.id]
            # human.set(px, py, -px, -py, 0, 0, human.radius)  # Assuming `set` positions the human
        
        return humans


