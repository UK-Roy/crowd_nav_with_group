import numpy as np

class ScenarioConfig:
    """Configuration for different evaluation scenarios"""
    
    @staticmethod
    def get_scenario(scenario_type):
        """Get configuration for specific scenario type"""
        
        scenarios = {
            'dense_groups': {
                'num_groups': 4,
                'min_group_size': 3,
                'max_group_size': 5,
                'arena_size': 4,  # Smaller arena for density
                'human_num': 18,
                'group_dynamic': False,
                'description': 'Dense groups in narrow passages'
            },
            
            'mixed_50_50': {
                'num_groups': 2,
                'min_group_size': 4,
                'max_group_size': 5,
                'arena_size': 6,
                'human_num': 16,  # 8 in groups, 8 individual
                'group_dynamic': False,
                'description': '50% individuals, 50% in groups'
            },
            
            'dynamic_formations': {
                'num_groups': 3,
                'min_group_size': 2,
                'max_group_size': 4,
                'arena_size': 6,
                'human_num': 15,
                'group_dynamic': True,  # Groups can move
                'description': 'Dynamic group formations'
            },
            
            'crossing_groups': {
                'num_groups': 2,
                'min_group_size': 4,
                'max_group_size': 6,
                'arena_size': 6,
                'human_num': 10,
                'group_dynamic': True,
                'description': 'Groups crossing robot path'
            },
            
            'static_dynamic_mix': {
                'num_groups': 3,
                'min_group_size': 3,
                'max_group_size': 4,
                'arena_size': 6,
                'human_num': 14,
                'group_dynamic': 'mixed',  # Some static, some dynamic
                'description': 'Mix of static and dynamic groups'
            },
            'groups_only': {
                'num_groups': 5,
                'min_group_size': 3,
                'max_group_size': 4,
                'arena_size': 6,
                'human_num': 18,  # All will be in groups
                'group_dynamic': False,
                'individuals_allowed': False,  # New flag
                'description': 'Only groups, no individual humans'
            },
        }
        
        return scenarios.get(scenario_type, scenarios['mixed_50_50'])