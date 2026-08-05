"""
Online estimation of the group count K from the GroupDetector's pairwise
groupness scores.

The fixed K=3 slot budget is the sharpest criticism of the CoRL submission: with
exactly three groups in the benchmark, a reviewer can read K=3 as the group count
supplied in advance. This module removes the need to supply it. The detector
already emits W_ij, the probability that pedestrians i and j share a group, so
thresholding W and counting connected components yields a per-frame estimate that
uses no privileged information.

Singleton components are ungrouped individuals, not groups, so only components of
at least `min_group_size` members are counted. A scene the detector reads as
having no groups therefore returns 0, and the caller is expected to render no
group layers for it.
"""

import torch


def estimate_k(W: torch.Tensor,
               vmask: torch.Tensor,
               threshold: float = 0.40,
               min_group_size: int = 2) -> torch.Tensor:
    """
    W     : (B, N, N)  pairwise groupness probabilities in [0, 1]
    vmask : (B, N)     bool or float, nonzero = pedestrian visible this frame

    Returns k_hat : (B,) long, the number of detected groups per sample.
    Invisible pedestrians never join a component, so padding slots in the
    observation cannot inflate the count.
    """
    N = W.shape[1]
    vis = vmask.bool()

    # Adjacency over visible pairs only, symmetrised: the edge classifier is not
    # guaranteed to score (i,j) and (j,i) identically, and either direction
    # crossing the threshold is evidence of a shared group.
    pair_vis = vis.unsqueeze(1) & vis.unsqueeze(2)              # (B,N,N)
    adj = (W >= threshold) & pair_vis
    adj = adj | adj.transpose(1, 2)

    # Reflexive closure for visible nodes only, so that boolean matrix powers
    # accumulate reachability instead of alternating parity.
    eye = torch.eye(N, dtype=torch.bool, device=W.device).unsqueeze(0)
    reach = adj | (eye & vis.unsqueeze(2))

    # Transitive closure by repeated squaring: N nodes need ceil(log2(N)) steps
    # for the longest possible chain to close.
    n_steps = max(1, int(torch.ceil(torch.log2(torch.tensor(float(max(N, 2))))).item()))
    for _ in range(n_steps):
        reach = (reach.float().bmm(reach.float()) > 0)

    comp_size = reach.sum(-1)                                   # (B,N)

    # Count each component once by electing its lowest-indexed member. A node is
    # the representative when no lower-indexed node is reachable from it.
    lower = torch.tril(torch.ones(N, N, dtype=torch.bool, device=W.device),
                       diagonal=-1).unsqueeze(0)                # (1,N,N)
    has_lower = (reach & lower).any(-1)                         # (B,N)
    is_rep = vis & ~has_lower

    k_hat = (is_rep & (comp_size >= min_group_size)).sum(-1)    # (B,)
    return k_hat.long()
