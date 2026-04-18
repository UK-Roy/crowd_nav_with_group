from crowd_nav.policy.srnn import SRNN


class GARN(SRNN):
    """
    Policy class for GARN (Group-Aware Robot Navigation, Lu et al. RA-L 2025).
    Implemented as a comparison baseline for TAGA.

    The network architecture (STGAN) lives in rl/networks/stgan_model.py.
    This class provides the policy name and inherits clip_action from SRNN.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = 'garn'
        self.trainable = True
        self.multiagent_training = True
