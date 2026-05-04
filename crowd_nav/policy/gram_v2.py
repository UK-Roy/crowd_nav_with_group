from crowd_nav.policy.srnn import SRNN


class GRAMV2(SRNN):
    """
    Policy class for GRAM-v2 (Phase 4 end-to-end navigation).
    The network lives in rl/networks/gram_v2_network.py.
    This class provides the policy name; predict() is never called during PPO training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = 'gram_v2'
        self.trainable = True
        self.multiagent_training = True
