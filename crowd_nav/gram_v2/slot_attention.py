"""
GRAM-v2 — Stage 4a: Slot Attention group pooling.

Compresses N per-human GNN-refined embeddings g_i into K=3 group-prototype
vectors via iterative cross-attention (Locatello et al., NeurIPS 2020).

Architecture (one forward pass):
  1. Sample K slot vectors from a learned Gaussian (mu, sigma).
  2. Repeat n_iters times:
       a. Compute dot-product attention between slots (queries) and humans (keys).
          Softmax over K  →  each human is "claimed" by the most compatible slot.
       b. Masked-mean readout from human values, weighted by attention.
       c. GRU slot update.
       d. Residual position-wise FF.
  3. Return final slots and the last-iteration attention map.

Usage:
  from crowd_nav.gram_v2.slot_attention import SlotAttention

  sa = SlotAttention()                # K=3, slot_dim=64, n_iters=3
  slots, attn = sa(g, mask)
    g     : (B, N, 64)   GNN-refined embeddings from GroupDetector Phase 2
    mask  : (B, N)       bool, True = visible human
    slots : (B, K, 64)   group prototype vectors  (fed to cross-attention in Phase 4)
    attn  : (B, K, N)    attention weights, softmax over K
                         (attn[b, k, n] ≈ probability human n belongs to slot k)

Training losses  (gram_v2_train_phase3.py):
  co_assignment_loss  : BCE on co-assignment prob vs GT pairwise labels — no
                        Hungarian matching needed, fully differentiable.
  slot_diversity_loss : entropy regulariser that prevents all humans collapsing
                        into a single slot.

Evaluation:
  compute_purity : Hungarian-matched purity over grouped humans.
                   Success criterion: > 0.85 on the test set.
"""

from typing import Optional

import torch
import torch.nn as nn
import numpy as np

EMBED_DIM = 64
SLOT_DIM  = 64
K_SLOTS   = 3
N_ITERS   = 3


class SlotAttention(nn.Module):
    """
    Slot attention pooling: N human embeddings → K group prototypes.

    Slots are re-initialised from a learned Gaussian at every forward call,
    so the module sees fresh competition at every timestep — no memory of
    previous scene state (that is handled by the GRU in Phase 4).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, slot_dim: int = SLOT_DIM,
                 K: int = K_SLOTS, n_iters: int = N_ITERS, eps: float = 1e-8):
        super().__init__()
        self.K        = K
        self.slot_dim = slot_dim
        self.n_iters  = n_iters
        self.eps      = eps
        self.scale    = slot_dim ** -0.5

        # Learned slot initialisation — one shared Gaussian for all K slots.
        # Different random noise at each forward call ensures slot diversity.
        self.slot_mu       = nn.Parameter(torch.empty(1, 1, slot_dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.normal_(self.slot_mu)

        # Input → keys / values
        self.norm_input = nn.LayerNorm(embed_dim)
        self.to_k = nn.Linear(embed_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, slot_dim, bias=False)

        # Slot → queries
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU slot update (GRUCell: input_size=slot_dim, hidden_size=slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Residual position-wise FF applied after GRU
        self.norm_ff = nn.LayerNorm(slot_dim)
        self.ff = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )

    def forward(self, g: torch.Tensor, mask: torch.Tensor):
        """
        g    : (B, N, embed_dim)   GNN-refined human embeddings
        mask : (B, N)              bool, True = visible human

        Returns
        -------
        slots : (B, K, slot_dim)
        attn  : (B, K, N)   final-iteration attention weights (softmax over K)
        """
        B, _, _ = g.shape

        # ── Sample slot initialisations ──────────────────────────────────────
        mu    = self.slot_mu.expand(B, self.K, -1)               # (B, K, D)
        sigma = self.slot_logsigma.exp().expand(B, self.K, -1)
        slots = mu + sigma * torch.randn(
            B, self.K, self.slot_dim, device=g.device, dtype=g.dtype)

        # ── Project inputs ───────────────────────────────────────────────────
        g_ln = self.norm_input(g)
        keys   = self.to_k(g_ln)   # (B, N, D)
        values = self.to_v(g_ln)   # (B, N, D)

        # Mask invisible humans: push logits to −∞ before softmax over K
        neg_inf = (~mask).float().unsqueeze(1) * -1e9   # (B, 1, N)

        attn = None
        for _ in range(self.n_iters):
            prev = slots

            # Queries from normalised slots
            q = self.to_q(self.norm_slots(slots))                     # (B, K, D)

            # Slot competition: softmax over K so each human picks one slot
            logits = torch.einsum('bkd,bnd->bkn', q, keys) * self.scale
            logits = logits + neg_inf
            attn   = logits.softmax(dim=1)                             # (B, K, N)

            # Weighted-mean readout, normalised over visible humans
            weights = attn / (attn.sum(-1, keepdim=True) + self.eps)  # (B, K, N)
            updates = torch.einsum('bkn,bnd->bkd', weights, values)   # (B, K, D)

            # GRU update (flatten batch × slots for GRUCell)
            slots = self.gru(
                updates.reshape(B * self.K, self.slot_dim),
                prev.reshape(   B * self.K, self.slot_dim),
            ).reshape(B, self.K, self.slot_dim)

            # Residual FF
            slots = slots + self.ff(self.norm_ff(slots))

        return slots, attn   # attn : (B, K, N)


# ── Training losses ───────────────────────────────────────────────────────────

def co_assignment_loss(attn: torch.Tensor, labels: torch.Tensor,
                       mask: torch.Tensor, pos_weight: float = 3.0) -> torch.Tensor:
    """
    Co-assignment BCE loss — fully differentiable, no Hungarian matching.

    co_assign[i, j] = Σ_k  attn[k, i] · attn[k, j]
      = probability that humans i and j are assigned to the same slot.

    BCE(co_assign, GT_labels) trains slots to pull same-group members together
    and push different-group members to different slots.

    attn       : (B, K, N)   slot attention weights (softmax over K)
    labels     : (B, N, N)   GT pairwise groupness — 1.0 iff same group
    mask       : (B, N)      bool visible mask
    pos_weight : scalar weight on positive (same-group) pairs
    """
    # Predicted co-assignment probability — (B, N, N)
    co = torch.einsum('bkn,bkm->bnm', attn, attn)

    _, N = mask.shape
    eye   = torch.eye(N, device=mask.device, dtype=torch.bool).unsqueeze(0)
    valid = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye   # (B, N, N)

    pred = co[valid].clamp(1e-6, 1.0 - 1e-6)
    gt   = labels[valid]

    # Weighted BCE: emphasise rare positive pairs
    bce = -(pos_weight * gt * pred.log() + (1.0 - gt) * (1.0 - pred).log())
    return bce.mean()


def slot_diversity_loss(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Entropy regulariser — prevents all humans collapsing into one slot.

    Computes the entropy of each slot's mean attention over visible humans.
    A high-entropy mean distribution means each slot is used roughly equally.
    We maximise entropy by minimising its negation.

    attn : (B, K, N)
    mask : (B, N)
    """
    vis       = mask.float().unsqueeze(1)                              # (B, 1, N)
    mean_attn = (attn * vis).sum(-1) / (vis.sum(-1) + 1e-8)           # (B, K)
    mean_attn = mean_attn / (mean_attn.sum(-1, keepdim=True) + 1e-8)
    entropy   = -(mean_attn * (mean_attn + 1e-8).log()).sum(-1)        # (B,)
    return -entropy.mean()   # minimise = maximise entropy


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_purity(attn_np: np.ndarray, labels_np: np.ndarray,
                   mask_np: np.ndarray) -> Optional[float]:
    """
    Hungarian-matched purity for one sample.

    For each grouped human (has at least one positive GT pair), we check
    whether its predicted slot matches the majority vote within its GT group.
    Hungarian matching finds the slot→group assignment that maximises correct
    count, then purity = correct / total_grouped.

    attn_np   : (K, N)  numpy float
    labels_np : (N, N)  numpy float  — GT pair matrix
    mask_np   : (N,)    numpy bool

    Returns purity ∈ [0, 1], or None if no grouped humans are visible.
    """
    try:
        from scipy.sparse.csgraph import connected_components
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None

    vis_idx = np.where(mask_np)[0]
    if len(vis_idx) < 2:
        return None

    # Connected components among visible humans
    sub = labels_np[np.ix_(vis_idx, vis_idx)]
    _, comp = connected_components(sub, directed=False, connection='weak')
    has_partner = sub.sum(axis=1) > 0

    # Only count grouped humans
    grouped_local = np.where(has_partner)[0]   # indices within vis_idx
    if len(grouped_local) == 0:
        return None

    global_idx   = vis_idx[grouped_local]
    slot_assign  = attn_np[:, global_idx].argmax(axis=0)   # (n_grouped,)
    gt_comp      = comp[grouped_local]                     # component IDs

    K = attn_np.shape[0]
    uniq_g   = np.unique(gt_comp)
    G        = len(uniq_g)
    g_map    = {g: i for i, g in enumerate(uniq_g)}

    # Confusion matrix C[k, g] = humans assigned to slot k in GT group g
    C = np.zeros((K, G), dtype=int)
    for s, g in zip(slot_assign, gt_comp):
        C[s, g_map[g]] += 1

    # Hungarian: maximise matched count (negate for minimisation)
    row_ind, col_ind = linear_sum_assignment(-C)
    matched = C[row_ind, col_ind].sum()
    return float(matched) / len(grouped_local)
