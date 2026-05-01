"""
GRAM-v2 — end-to-end group perception modules.

Stage 1: PedestrianEncoder   — per-human 7-d features → 64-d embeddings
Stage 2: PairwiseEdgeNetwork — pair of embeddings → groupness probability [0,1]
Stage 3: GroupGNN            — edge-weighted message passing (in gnn.py)

GroupDetector is the single model class for all phases:
  n_gnn_layers=0  →  Phase 1 (Encoder + EdgeNet only)
  n_gnn_layers=3  →  Phase 2 (Encoder + EdgeNet + GNN + EdgeNet, shared)

Forward always returns (W_final, W0, g, h0):
  W_final : post-GNN groupness  (= W0 when n_gnn_layers=0)
  W0      : pre-GNN groupness   (auxiliary loss target)
  g       : GNN-refined embeddings  (= h0 when n_gnn_layers=0)
  h0      : raw encoder embeddings

Input feature vector per human (7-d):
  [p_rel_x, p_rel_y,   relative position w.r.t. robot (m)
   v_rel_x, v_rel_y,   relative velocity w.r.t. robot (m/s)
   v_norm,             speed magnitude  (captures v_pref heterogeneity)
   sin_theta, cos_theta]

Invisible-human slots must be zeroed before calling forward().
mask[i] = True  means human i is visible.
"""

import torch
import torch.nn as nn

MAX_HUMANS = 20
FEAT_DIM   = 7
EMBED_DIM  = 64


class PedestrianEncoder(nn.Module):
    """Stage 1: encode each human's 7-d feature into a 64-d embedding."""

    def __init__(self, input_dim: int = FEAT_DIM, hidden_dim: int = 128,
                 output_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N, 7)   raw per-human features; invisible slots must be zero
        mask : (B, N)      bool, True = visible
        Returns h : (B, N, 64)  zero for invisible humans
        """
        m = mask.unsqueeze(-1).float()
        return self.net(x * m) * m


class PairwiseEdgeNetwork(nn.Module):
    """Stage 2: predict a symmetric groupness probability for every human pair.

    Edge feature for pair (i, j)  [134 dims total]:
      h_i, h_j      : 64 + 64   concatenated embeddings
      dp_x, dp_y    : 2         position diff (i relative to j)
      dv_x, dv_y    : 2         velocity diff
      cos_sim       : 1         cosine similarity of velocities
      dist          : 1         Euclidean distance between i and j

    Output is symmetrised: W = (W_raw + W_raw.T) / 2  before masking.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 6, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor,
                raw_feats: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        h         : (B, N, 64)
        raw_feats : (B, N, 7)
        mask      : (B, N)  bool
        Returns W : (B, N, N)  symmetric groupness probabilities; 0 for invalid pairs
        """
        N = h.shape[1]

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        p      = raw_feats[:, :, :2]
        v      = raw_feats[:, :, 2:4]
        v_norm = raw_feats[:, :, 4:5]

        dp   = p.unsqueeze(2) - p.unsqueeze(1)           # (B,N,N,2)
        dv   = v.unsqueeze(2) - v.unsqueeze(1)
        dist = torch.norm(dp, dim=-1, keepdim=True)       # (B,N,N,1)

        v_i  = v.unsqueeze(2).expand(-1, -1, N, -1)
        v_j  = v.unsqueeze(1).expand(-1, N, -1, -1)
        vn_i = v_norm.unsqueeze(2).expand(-1, -1, N, -1)
        vn_j = v_norm.unsqueeze(1).expand(-1, N, -1, -1)
        cos_sim = (v_i * v_j).sum(-1, keepdim=True) / (vn_i * vn_j + 1e-6)

        edge_feat = torch.cat([h_i, h_j, dp, dv, cos_sim, dist], dim=-1)
        logits    = self.net(edge_feat).squeeze(-1)        # (B,N,N)

        # symmetrise then sigmoid
        logits = (logits + logits.transpose(-2, -1)) / 2.0
        W      = torch.sigmoid(logits)

        # zero self-loops and invisible pairs
        valid   = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
        eye     = torch.eye(N, device=h.device).unsqueeze(0)
        return W * valid * (1.0 - eye)


class GroupDetector(nn.Module):
    """
    Single model class for all GRAM-v2 phases.

      n_gnn_layers=0  →  Phase 1: Encoder + EdgeNet
      n_gnn_layers=3  →  Phase 2: Encoder + EdgeNet + GNN + EdgeNet (shared)

    Always returns (W_final, W0, g, h0) so training scripts are identical.
    When n_gnn_layers=0: W_final == W0 and g == h0.
    """

    def __init__(self, input_dim: int = FEAT_DIM, embed_dim: int = EMBED_DIM,
                 enc_hidden: int = 128, gnn_hidden: int = 128,
                 n_gnn_layers: int = 0, dropout: float = 0.1):
        super().__init__()
        self.encoder  = PedestrianEncoder(input_dim, enc_hidden, embed_dim, dropout)
        self.edge_net = PairwiseEdgeNetwork(embed_dim)

        if n_gnn_layers > 0:
            from crowd_nav.gram_v2.gnn import GroupGNN
            self.gnn = GroupGNN(embed_dim, gnn_hidden, n_gnn_layers)
        else:
            self.gnn = None

    def forward(self, raw_feats: torch.Tensor, mask: torch.Tensor):
        """
        raw_feats : (B, N, 7)   invisible slots must be zeroed by caller
        mask      : (B, N)      bool

        Returns:
          W_final : (B, N, N)  main groupness prediction
          W0      : (B, N, N)  pre-GNN groupness (== W_final when no GNN)
          g       : (B, N, 64) GNN-refined embeddings (== h0 when no GNN)
          h0      : (B, N, 64) raw encoder embeddings
        """
        h0 = self.encoder(raw_feats, mask)
        W0 = self.edge_net(h0, raw_feats, mask)

        if self.gnn is not None:
            g       = self.gnn(h0, W0, mask)
            W_final = self.edge_net(g, raw_feats, mask)
        else:
            g       = h0
            W_final = W0

        return W_final, W0, g, h0
