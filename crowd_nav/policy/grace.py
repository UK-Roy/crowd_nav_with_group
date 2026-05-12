from crowd_nav.policy.srnn import SRNN


class GRACE(SRNN):
    """
    Policy class for GRACE (end-to-end group-aware cost-map navigation).
    Network lives in rl/networks/grace_network.py.
    predict() is never called during PPO training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = 'grace'
        self.trainable = True
        self.multiagent_training = True
