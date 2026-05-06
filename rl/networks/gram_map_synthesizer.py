"""
GRAM-Map cost-map synthesis and planning components.

CostMapSynthesizer  — renders a (B, C, H, W) spatial cost stack from perception outputs.
OccupancyHead       — predicts future occupancy for the self-supervised auxiliary loss.
CostMapPlanner      — 2D CNN policy head over the cost stack.

Cost channels (C = 1 + T + 1 + 1 + 1 + 1 = T+5, default T=4 → C=9):
  0        individual occupancy  (current positions)
  1..T     trajectory layers     (predicted positions at each horizon)
  T+1      group cohesion        (soft hull around slot centroids)
  T+2      group repulsion       (early-warning halo outside group hull)
  T+3      goal attractor        (high cost far from goal)
  T+4      boundary              (high cost near arena edge)

Grid convention:
  - robot-centric, world-axis-aligned (simple translation, no heading rotation)
  - (0, 0) = robot position
  - grid_range metres in each direction → 2*grid_range × 2*grid_range window
  - default 32×32 at 6m range → 0.375 m/cell
"""

import torch
import torch.nn as nn


class CostMapSynthesizer(nn.Module):
    def __init__(self,
                 grid_size: int = 32,
                 grid_range: float = 6.0,
                 horizons=(0.3, 0.7, 1.0, 1.5),
                 sigma_indiv: float = 0.5,
                 sigma_traj_base: float = 0.5,
                 sigma_group_scale: float = 1.2):
        super().__init__()
        self.grid_size        = grid_size
        self.grid_range       = grid_range
        self.horizons         = list(horizons)
        self.sigma_indiv      = sigma_indiv
        self.sigma_traj_base  = sigma_traj_base
        self.sigma_group_scale = sigma_group_scale

        # Pre-compute fixed grid coordinates — registered as buffer (moves with .to(device))
        lin  = torch.linspace(-grid_range, grid_range, grid_size)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')      # (H, W)
        _grid = torch.stack([xx, yy], dim=-1)                  # (H, W, 2)
        self.grid: torch.Tensor                                 # type hint for Pyright
        self.register_buffer('grid', _grid)

    @property
    def n_channels(self) -> int:
        return 1 + len(self.horizons) + 1 + 1 + 1 + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _splat(self, pos: torch.Tensor, vmask: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Gaussian splat for each agent position onto the grid.
        pos   : (B, N, 2)
        vmask : (B, N) float, 1 = visible
        Returns (B, H, W) cost layer in [0, 1].
        """
        # (B, N, H, W, 2) via broadcasting
        diff   = self.grid[None, None] - pos[:, :, None, None]   # (B,N,H,W,2)
        dsq    = (diff ** 2).sum(-1)                              # (B,N,H,W)
        gauss  = torch.exp(-dsq / (2 * sigma ** 2))              # (B,N,H,W)
        gauss  = gauss * vmask[:, :, None, None]                  # mask invisible
        return gauss.sum(1).clamp(0, 1)                           # (B,H,W)

    # ------------------------------------------------------------------

    def forward(self,
                p_cur: torch.Tensor,
                v_cur: torch.Tensor,
                vmask: torch.Tensor,
                goal:  torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """
        p_cur : (B, N, 2)  human positions in robot-centric frame
        v_cur : (B, N, 2)  human velocities (robot-frame-aligned)
        vmask : (B, N)     bool, True = visible
        goal  : (B, 2)     goal vector in robot-centric frame
        alpha : (B, K, N)  soft slot assignments from SlotAttention

        Returns cost_stack : (B, C, H, W)
        """
        B = p_cur.shape[0]
        mask_f = vmask.float()           # (B, N)

        layers = []

        # ── L1: Individual occupancy (current step) ───────────────────────────
        layers.append(self._splat(p_cur, mask_f, self.sigma_indiv).unsqueeze(1))

        # ── L2: Trajectory layers — one per horizon ───────────────────────────
        for i, dt in enumerate(self.horizons):
            sigma_t = self.sigma_traj_base * (1.0 + i * 0.3)   # grows with horizon
            future  = p_cur + v_cur * dt                        # (B, N, 2)
            layers.append(self._splat(future, mask_f, sigma_t).unsqueeze(1))

        # ── L3: Group cohesion (soft centroid-based hull per slot) ────────────
        alpha_n   = alpha / (alpha.sum(-1, keepdim=True) + 1e-8)  # (B,K,N)
        centroid  = torch.bmm(alpha_n, p_cur)                      # (B,K,2)

        diff_sq_k = ((p_cur.unsqueeze(1) - centroid.unsqueeze(2)) ** 2).sum(-1)  # (B,K,N)
        spread    = (alpha_n * diff_sq_k).sum(-1, keepdim=True).sqrt().clamp(min=0.4)  # (B,K,1)
        sigma_g   = (spread * self.sigma_group_scale).unsqueeze(-1)  # (B,K,1,1)

        # (B, K, H, W, 2) centroid splat
        diff_grp   = self.grid[None, None] - centroid[:, :, None, None]  # (B,K,H,W,2)
        dsq_grp    = (diff_grp ** 2).sum(-1)                              # (B,K,H,W)
        grp_gauss  = torch.exp(-dsq_grp / (2 * sigma_g ** 2))            # (B,K,H,W)
        group_lyr  = grp_gauss.max(1).values.clamp(0, 1).unsqueeze(1)    # (B,1,H,W)
        layers.append(group_lyr)

        # ── L4: Group repulsion halo (wider sigma — early-warning zone) ───────
        repulsion  = torch.exp(-dsq_grp / (2 * (sigma_g * 2.0) ** 2))
        repulsion_lyr = repulsion.max(1).values.clamp(0, 1).unsqueeze(1)
        layers.append(repulsion_lyr)

        # ── L5: Goal attractor — high cost far from goal ──────────────────────
        diff_goal  = self.grid[None] - goal[:, None, None]             # (B,H,W,2)
        dist_goal  = (diff_goal ** 2).sum(-1).sqrt()                   # (B,H,W)
        max_dist   = self.grid_range * 2 ** 0.5                        # diagonal
        goal_lyr   = (dist_goal / max_dist).clamp(0, 1).unsqueeze(1)  # (B,1,H,W)
        layers.append(goal_lyr)

        # ── L6: Boundary — high cost near arena edge ──────────────────────────
        coords    = self.grid / self.grid_range                        # (H,W,2) in [-1,1]
        boundary  = coords.abs().max(dim=-1).values                    # (H,W) in [0,1]
        boundary  = boundary.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)  # (B,1,H,W)
        layers.append(boundary)

        return torch.cat(layers, dim=1)   # (B, C, H, W)


class OccupancyHead(nn.Module):
    """Lightweight head that decodes cost_stack → future occupancy predictions."""

    def __init__(self, in_channels: int, n_horizons: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, n_horizons, 1),
            # No Sigmoid here — output raw logits for binary_cross_entropy_with_logits
        )

    def forward(self, cost_stack: torch.Tensor) -> torch.Tensor:
        """cost_stack (B, C, H, W) → logits (B, T, H, W) for BCE with logits"""
        return self.decoder(cost_stack)

    @staticmethod
    def make_targets(p_cur: torch.Tensor,
                     v_cur: torch.Tensor,
                     vmask: torch.Tensor,
                     grid:  torch.Tensor,
                     horizons: list,
                     sigma: float = 0.5) -> torch.Tensor:
        """
        Compute self-supervised occupancy targets analytically from current obs.
        No environment labels needed — targets are predicted future positions.

        Returns (B, T, H, W) soft occupancy in [0, 1].
        """
        mask_f = vmask.float()
        targets = []
        for dt in horizons:
            future = p_cur + v_cur * dt                               # (B, N, 2)
            diff   = grid[None, None] - future[:, :, None, None]     # (B,N,H,W,2)
            dsq    = (diff ** 2).sum(-1)                              # (B,N,H,W)
            gauss  = torch.exp(-dsq / (2 * sigma ** 2))
            gauss  = gauss * mask_f[:, :, None, None]
            targets.append(gauss.sum(1).clamp(0, 1))                  # (B,H,W)
        return torch.stack(targets, dim=1)                             # (B,T,H,W)


class CostMapPlanner(nn.Module):
    """
    2D CNN that takes the cost stack and produces 256-d navigation features.

    Architecture: 4× Conv2d (stride-2 downsampling) → AdaptiveAvgPool2d(4) → Linear → 256.
    Input grid 32×32 → 16×16 → 8×8 → 4×4 → 4×4 pooled → 128×4×4=2048 → 256.
    """

    def __init__(self, in_channels: int, output_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, padding=1),            nn.ReLU(),
            nn.Conv2d(32,          64,  3, stride=2, padding=1),  nn.ReLU(),  # 16×16
            nn.Conv2d(64,          128, 3, stride=2, padding=1),  nn.ReLU(),  # 8×8
            nn.Conv2d(128,         128, 3, padding=1),             nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                            # 4×4
        )
        self.fc = nn.Linear(128 * 4 * 4, output_size)

    def forward(self, cost_stack: torch.Tensor) -> torch.Tensor:
        """cost_stack (B, C, H, W) → features (B, output_size)"""
        return self.fc(self.net(cost_stack).flatten(1))
