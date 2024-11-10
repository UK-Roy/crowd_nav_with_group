import numpy as np

class Group:
    def __init__(self, id, max_mem):
        self.id = id
        self.members = []
        self.leader = None
        self.centroid = None
        self.radius = None
        self.max_member = np.random.randint(2, max_mem+1)
    
    def config_group(self):
        if self.members:
            self.leader = self.members[0]  # First member is the leader

    def set_centroid(self, px, py):
        self.centroid = [px, py]
    
    def set_radius(self, radius):
        self.radius = radius
    
    def add_member(self, mem_id):
        if not self.check_validity():
            print(f"Members are limited")
            return
        self.members.append(mem_id)
        self.config_group()  # Configure leader after adding first member
    
    def remove_member(self, mem_id):
        if mem_id in self.members:
            self.members.remove(mem_id)
            if mem_id == self.leader and self.members:
                self.leader = self.members[0]  # Update leader if needed

    def check_validity(self):
        return len(self.members) < self.max_member
    
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

