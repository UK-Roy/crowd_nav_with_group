policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.hybrid import HYBRID_ORCA_SOCIAL_FORCE
from crowd_nav.policy.hybrid_orca_flocking import HYBRID_ORCA_FLOCKING
from crowd_nav.policy.srnn import SRNN, selfAttn_merge_SRNN, selfAttn_merge_SRNN_GrpAttn
from crowd_nav.policy.zone_based import ZoneBasedGroupAvoidance
from crowd_nav.policy.f_formation import FFormationAvoidance
from crowd_nav.policy.garn import GARN
from crowd_nav.policy.gram_v2 import GRAMV2
from crowd_nav.policy.grace import GRACE

policy_factory['zone_based'] = ZoneBasedGroupAvoidance
policy_factory['f_formation'] = FFormationAvoidance

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN
policy_factory['selfAttn_merge_srnn'] = selfAttn_merge_SRNN
policy_factory['selfAttn_merge_srnn_grpAttn'] = selfAttn_merge_SRNN_GrpAttn

policy_factory['hybrid_orca_social_force'] = HYBRID_ORCA_SOCIAL_FORCE
policy_factory['hybrid_orca_flocking'] = HYBRID_ORCA_FLOCKING
policy_factory['garn'] = GARN
policy_factory['gram_v2'] = GRAMV2
policy_factory['grace'] = GRACE
policy_factory['gram_map'] = GRACE   # backward-compat alias for stageA/B/C checkpoints