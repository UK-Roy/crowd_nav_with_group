"""
GRAM-v2 — end-to-end group perception modules.

Stage 1: PedestrianEncoder   — per-human 7*T_WINDOW-d features → 64-d embeddings
Stage 2: PairwiseEdgeNetwork — pair of embeddings + current-frame geometry → groupness [0,1]
Stage 3: GroupGNN            — edge-weighted message passing (in gnn.py)

GroupDetector is the single model class for all phases:
  n_gnn_layers=0  →  Phase 1 (Encoder + EdgeNet only)
  n_gnn_layers=3  →  Phase 2 (Encoder + EdgeNet + GNN + EdgeNet, shared)

Forward always returns (W_final, W0, g, h0):
  W_final : post-GNN groupness  (= W0 when n_gnn_layers=0)
  W0      : pre-GNN groupness   (auxiliary loss target)
  g       : GNN-refined embeddings  (= h0 when n_gnn_layers=0)
  h0      : raw encoder embeddings

Input feature vector per human (7-d per frame, T_WINDOW=3 frames stacked → 21-d total):
  Frame layout (repeated T_WINDOW times, oldest first):
  [p_rel_x, p_rel_y,   relative position w.r.t. robot (m)
   v_rel_x, v_rel_y,   relative velocity w.r.t. robot (m/s)
   v_norm,             speed magnitude
   sin_theta, cos_theta]

Temporal context: stacking 3 consecutive frames lets the edge network observe
velocity trends (group members co-move; strangers drift apart over time).
The edge geometry features (dp, dv, dist) use only the most recent frame's 7 dims.

Invisible-human slots must be zeroed before calling forward().
mask[i] = True  means human i is visible.
"""

import torch
import torch.nn as nn

MAX_HUMANS = 20
FEAT_DIM   = 7      # dims per single frame
T_WINDOW   = 3      # consecutive frames stacked
INPUT_DIM  = FEAT_DIM * T_WINDOW   # 21 — encoder input width
EMBED_DIM  = 64


class PedestrianEncoder(nn.Module):
    """Stage 1: encode each human's 21-d (3-frame) feature into a 64-d embedding."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 256,
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
        x    : (B, N, INPUT_DIM)  raw per-human features; invisible slots must be zero
        mask : (B, N)             bool, True = visible
        Returns h : (B, N, 64)   zero for invisible humans
        """
        m = mask.unsqueeze(-1).float()
        return self.net(x * m) * m


class PairwiseEdgeNetwork(nn.Module):
    """Stage 2: predict a symmetric groupness probability for every human pair.

    Variant B edge feature [134 dims]:
      h_i, h_j   : 64+64  embeddings (carry 3-frame context via encoder)
      dp, dv     : 2+2    position/velocity diff — current frame
      cos_sim    : 1      velocity direction alignment — current frame
      dist       : 1      Euclidean distance — current frame

    Variant C adds explicit pairwise temporal features [+10 dims = 144 total]:
      dp_k       : 2+2    position diff at frames t-2 and t-1
      dist_k     : 1+1+1  distance at frames t-2, t-1, t
      delta_dist : 1      dist(t) - dist(t-2)  — stable≈group, changing≈stranger
      cos_sim_k  : 1+1    velocity alignment at frames t-2 and t-1

    Output is symmetrised: W = (W_raw + W_raw.T) / 2  before masking.
    """

    # Extra dims added by pairwise temporal features (Variant C)
    TEMPORAL_EXTRA = 10  # dp_0(2) + dp_1(2) + dist_0,1,2(3) + delta_dist(1) + cos_k0,1(2)

    def __init__(self, embed_dim: int = EMBED_DIM,
                 use_pairwise_temporal: bool = False):
        super().__init__()
        self.use_pairwise_temporal = use_pairwise_temporal
        extra = self.TEMPORAL_EXTRA if use_pairwise_temporal else 0
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 6 + extra, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _pairwise_temporal(self, full_feats: torch.Tensor) -> torch.Tensor:
        """Compute explicit pairwise temporal features from all T_WINDOW frames.
        full_feats : (B, N, 21)
        Returns    : (B, N, N, 10)
        """
        N = full_feats.shape[1]
        temporal = []
        dists = []
        for k in range(T_WINDOW):
            p_k  = full_feats[:, :, k * FEAT_DIM     : k * FEAT_DIM + 2]   # (B,N,2)
            v_k  = full_feats[:, :, k * FEAT_DIM + 2 : k * FEAT_DIM + 4]
            vn_k = full_feats[:, :, k * FEAT_DIM + 4 : k * FEAT_DIM + 5]

            dp_k   = p_k.unsqueeze(2) - p_k.unsqueeze(1)                   # (B,N,N,2)
            dist_k = torch.norm(dp_k, dim=-1, keepdim=True)                 # (B,N,N,1)
            dists.append(dist_k)

            if k < T_WINDOW - 1:    # t-2 and t-1 only (t already in curr_feats)
                temporal.append(dp_k)

                v_i  = v_k.unsqueeze(2).expand(-1, -1, N, -1)
                v_j  = v_k.unsqueeze(1).expand(-1, N, -1, -1)
                vn_i = vn_k.unsqueeze(2).expand(-1, -1, N, -1)
                vn_j = vn_k.unsqueeze(1).expand(-1, N, -1, -1)
                cos_k = (v_i * v_j).sum(-1, keepdim=True) / (vn_i * vn_j + 1e-6)
                temporal.append(cos_k)

        # dist at all 3 frames + delta_dist
        temporal += dists
        delta_dist = dists[-1] - dists[0]
        temporal.append(delta_dist)

        return torch.cat(temporal, dim=-1)   # (B,N,N,10)

    def forward(self, h: torch.Tensor,
                curr_feats: torch.Tensor,
                mask: torch.Tensor,
                full_feats: torch.Tensor = None) -> torch.Tensor:
        """
        h          : (B, N, 64)   encoder embeddings
        curr_feats : (B, N, 7)    most-recent frame (for dp, dv, cos_sim, dist)
        mask       : (B, N)       bool
        full_feats : (B, N, 21)   all frames — required when use_pairwise_temporal=True
        Returns W  : (B, N, N)    symmetric groupness probabilities
        """
        N = h.shape[1]

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

        p      = curr_feats[:, :, :2]
        v      = curr_feats[:, :, 2:4]
        v_norm = curr_feats[:, :, 4:5]

        dp   = p.unsqueeze(2) - p.unsqueeze(1)
        dv   = v.unsqueeze(2) - v.unsqueeze(1)
        dist = torch.norm(dp, dim=-1, keepdim=True)

        v_i  = v.unsqueeze(2).expand(-1, -1, N, -1)
        v_j  = v.unsqueeze(1).expand(-1, N, -1, -1)
        vn_i = v_norm.unsqueeze(2).expand(-1, -1, N, -1)
        vn_j = v_norm.unsqueeze(1).expand(-1, N, -1, -1)
        cos_sim = (v_i * v_j).sum(-1, keepdim=True) / (vn_i * vn_j + 1e-6)

        parts = [h_i, h_j, dp, dv, cos_sim, dist]

        if self.use_pairwise_temporal and full_feats is not None:
            parts.append(self._pairwise_temporal(full_feats))

        edge_feat = torch.cat(parts, dim=-1)
        logits    = self.net(edge_feat).squeeze(-1)

        logits = (logits + logits.transpose(-2, -1)) / 2.0
        W      = torch.sigmoid(logits)

        valid = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
        eye   = torch.eye(N, device=h.device).unsqueeze(0)
        return W * valid * (1.0 - eye)


class GroupDetector(nn.Module):
    """
    Single model class for all GRAM-v2 phases.

      n_gnn_layers=0  →  Phase 1: Encoder + EdgeNet
      n_gnn_layers=3  →  Phase 2: Encoder + EdgeNet + GNN + EdgeNet (shared)

    Always returns (W_final, W0, g, h0) so training scripts are identical.
    When n_gnn_layers=0: W_final == W0 and g == h0.
    """

    def __init__(self, input_dim: int = INPUT_DIM, embed_dim: int = EMBED_DIM,
                 enc_hidden: int = 256, gnn_hidden: int = 256,
                 n_gnn_layers: int = 0, dropout: float = 0.1,
                 use_pairwise_temporal: bool = False):
        super().__init__()
        self.feat_dim              = FEAT_DIM
        self.use_pairwise_temporal = use_pairwise_temporal
        self.encoder  = PedestrianEncoder(input_dim, enc_hidden, embed_dim, dropout)
        self.edge_net = PairwiseEdgeNetwork(embed_dim, use_pairwise_temporal)

        if n_gnn_layers > 0:
            from crowd_nav.gram_v2.gnn import GroupGNN
            self.gnn = GroupGNN(embed_dim, gnn_hidden, n_gnn_layers)
        else:
            self.gnn = None

    def forward(self, raw_feats: torch.Tensor, mask: torch.Tensor):
        """
        raw_feats : (B, N, INPUT_DIM)  invisible slots must be zeroed by caller
                    Last FEAT_DIM dims = most recent frame (used for edge geometry).
        mask      : (B, N)             bool

        Returns:
          W_final : (B, N, N)   main groupness prediction
          W0      : (B, N, N)   pre-GNN groupness (== W_final when no GNN)
          g       : (B, N, 64)  GNN-refined embeddings (== h0 when no GNN)
          h0      : (B, N, 64)  raw encoder embeddings
        """
        # Most recent frame for edge geometry (dp, dv, dist)
        curr_feats  = raw_feats[..., -self.feat_dim:]
        # full_feats passed to edge_net only when Variant C pairwise temporal is on
        full_feats  = raw_feats if self.use_pairwise_temporal else None

        h0 = self.encoder(raw_feats, mask)
        W0 = self.edge_net(h0, curr_feats, mask, full_feats)

        if self.gnn is not None:
            g       = self.gnn(h0, W0, mask)
            W_final = self.edge_net(g, curr_feats, mask, full_feats)
        else:
            g       = h0
            W_final = W0

        return W_final, W0, g, h0
