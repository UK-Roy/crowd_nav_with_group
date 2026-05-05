from crowd_nav.policy.srnn import SRNN


class GRAMMap(SRNN):
    """
    Policy class for GRAM-Map (end-to-end cost-map navigation).
    Network lives in rl/networks/gram_map_network.py.
    predict() is never called during PPO training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = 'gram_map'
        self.trainable = True
        self.multiagent_training = True
