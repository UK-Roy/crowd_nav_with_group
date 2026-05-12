"""
GRAM-v2 Stage 3 — GNN message passing.

Each GNNLayer runs one round of edge-weighted message passing:
    h_i^(l+1) = LayerNorm( h_i + mean_j[ W[i,j] * MLP([h_i, h_j]) ] )

After n_layers rounds the embeddings carry community context: humans that are
spatially and kinematically grouped together look distinct from isolated pedestrians.

GroupGNN stacks multiple GNNLayers; used inside GroupDetector when n_gnn_layers > 0.
"""

import torch
import torch.nn as nn

EMBED_DIM = 64


class GNNLayer(nn.Module):
    """One round of edge-weighted message passing with residual + LayerNorm."""

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 128):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h: torch.Tensor,
                W: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        h    : (B, N, embed_dim)   input human embeddings
        W    : (B, N, N)           soft adjacency (groupness probabilities)
        mask : (B, N)              bool, True = visible
        Returns g : (B, N, embed_dim)  refined embeddings (invisible slots = 0)
        """
        B, N, D = h.shape

        # Build pairwise message inputs
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, D)
        msg = self.msg_mlp(torch.cat([h_i, h_j], dim=-1))  # (B, N, N, D)

        # Weighted aggregation: mean over neighbours
        # W[b, i, j] weights message from j to i
        W_exp = W.unsqueeze(-1)                         # (B, N, N, 1)
        agg   = (W_exp * msg).sum(dim=2)                # (B, N, D)
        denom = W.sum(dim=2, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
        agg   = agg / denom

        # Residual + LayerNorm + mask
        g = self.norm(h + agg)
        return g * mask.unsqueeze(-1).float()


class GroupGNN(nn.Module):
    """Stack of GNNLayers that iteratively refines human embeddings."""

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 128,
                 n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [GNNLayer(embed_dim, hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, h: torch.Tensor,
                W: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        h    : (B, N, 64)   initial embeddings from PedestrianEncoder
        W    : (B, N, N)    soft adjacency from PairwiseEdgeNetwork
        mask : (B, N)       bool
        Returns g : (B, N, 64)  group-aware embeddings
        """
        for layer in self.layers:
            h = layer(h, W, mask)
        return h

