"""
GRAM-Map Phase 4 — end-to-end group-aware cost-map navigation policy.

Pipeline (per timestep):
  1. Build 7-d per-human features from the 2-frame rolling buffer (same as GRAM-v2)
  2. GroupDetector (Phase 2) → g : (B, N, 64)  GNN-refined embeddings
  3. SlotAttention  (Phase 3) → slots : (B, K, 64), alpha : (B, K, N)
  4. CostMapSynthesizer → cost_stack : (B, C, H, W)  spatial belief
  5. CostMapPlanner (CNN)   → map_feat : (B, 256)
  6. Robot-state MLP        → rob_feat : (B, 64)
  7. Fusion: Linear(256+64) → fused   : (B, 256)
  8. GRUCell(256 → 256)     → h_new   (temporal memory)
  9. Actor / Critic heads   → value, actor features

Auxiliary loss (Stage A / Stage C, optional):
  OccupancyHead predicts future occupancy from cost_stack.
  Targets are computed analytically from current obs (self-supervised).
  Stored as self._aux_loss (scalar tensor) for the PPO update to pick up.

rnn_hxs layout (identical to GRAM-v2 for compatibility):
  'human_node_rnn'       (B, 1, 256)   — GRU hidden
  'human_human_edge_rnn' (B, N+1, 14)  — 2-frame per-human feature buffer

Run flags:
  --env-name CrowdSimVarNum-v0
  --human_node_rnn_size 256
  --human_human_edge_rnn_size 14
"""

import torch
import torch.nn as nn

from crowd_nav.gram_v2.models import GroupDetector, FEAT_DIM, INPUT_DIM
from crowd_nav.gram_v2.slot_attention import SlotAttention
from rl.networks.gram_map_synthesizer import CostMapSynthesizer, OccupancyHead, CostMapPlanner

MAX_HUMANS  = 20
GRU_HIDDEN  = 256
EMBED_DIM   = 64
K_SLOTS     = 3
ROBOT_RAW   = 9       # temporal_edges(2) + robot_node(7)
OUTPUT_SIZE = 256


def _reshapeT(tensor, seq_length, nenv):
    shape = tensor.size()[1:]
    return tensor.unsqueeze(0).reshape((seq_length, nenv, *shape))


def _build_7d_frame(spatial, velocity, mask):
    """(B,N,2), (B,N,2), (B,N) bool → (B,N,7)"""
    vx     = velocity[..., 0:1]
    vy     = velocity[..., 1:2]
    v_norm = (vx ** 2 + vy ** 2).sqrt()
    sin_t  = vy / (v_norm + 1e-6)
    cos_t  = vx / (v_norm + 1e-6)
    frame  = torch.cat([spatial, velocity, v_norm, sin_t, cos_t], dim=-1)
    return frame * mask.unsqueeze(-1).float()


class GRAMMapNetwork(nn.Module):
    """
    Full GRAM-Map network.  Matches the Policy interface expected by model.py:
      __init__(obs_space_dict, args)
      forward(inputs, rnn_hxs, masks, infer=False) → (value, actor_feat, rnn_hxs)
    """

    def __init__(self, obs_space_dict, args):  # noqa: ARG002
        super().__init__()
        self.is_recurrent = True
        self.args         = args

        self.human_num  = MAX_HUMANS
        self.seq_length = args.num_steps
        self.nenv       = args.num_processes
        self.nminibatch = args.num_mini_batch
        self.output_size = OUTPUT_SIZE

        # Expected by evaluation.py
        self.human_node_rnn_size       = GRU_HIDDEN
        self.human_human_edge_rnn_size = FEAT_DIM * 2   # 14

        # Grid / cost-map config (read from gram_map config if present)
        cfg = getattr(args, '_gram_map_cfg', None)
        grid_size   = getattr(cfg, 'grid_size',  32)  if cfg else 32
        grid_range  = getattr(cfg, 'grid_range', 6.0) if cfg else 6.0
        horizons    = getattr(cfg, 'horizons',   [0.3, 0.7, 1.0, 1.5]) if cfg else [0.3, 0.7, 1.0, 1.5]
        use_aux     = getattr(cfg, 'use_aux_loss', False) if cfg else False
        self.use_aux_loss  = getattr(args, 'gram_map_use_aux_loss', use_aux)
        self.aux_lambda    = getattr(cfg, 'aux_loss_weight', 0.1) if cfg else 0.1

        # ── Perception backbones (same as GRAM-v2) ────────────────────────────
        self.detector  = GroupDetector(input_dim=INPUT_DIM, n_gnn_layers=3)
        self.slot_attn = SlotAttention(embed_dim=EMBED_DIM, K=K_SLOTS)
        self._detector_use_pt = False

        # ── Cost-map synthesis ────────────────────────────────────────────────
        self.synthesizer = CostMapSynthesizer(
            grid_size=grid_size,
            grid_range=grid_range,
            horizons=horizons,
        )
        n_cost_ch = self.synthesizer.n_channels

        # ── Planning head (CNN over cost stack) ───────────────────────────────
        self.planner = CostMapPlanner(in_channels=n_cost_ch, output_size=GRU_HIDDEN)

        # ── Robot state MLP ───────────────────────────────────────────────────
        self.robot_mlp = nn.Sequential(
            nn.Linear(ROBOT_RAW, 128), nn.ReLU(),
            nn.Linear(128, EMBED_DIM),
        )

        # ── Fusion: planner_feat(256) + robot_feat(64) → 256 ─────────────────
        self.fusion = nn.Sequential(
            nn.Linear(GRU_HIDDEN + EMBED_DIM, GRU_HIDDEN), nn.ReLU(),
        )

        # ── Temporal GRU ──────────────────────────────────────────────────────
        self.gru = nn.GRUCell(GRU_HIDDEN, GRU_HIDDEN)

        # ── Actor-Critic heads ────────────────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
        )
        self.critic_linear = nn.Linear(GRU_HIDDEN, 1)

        # ── Auxiliary occupancy head (self-supervised, optional) ──────────────
        if self.use_aux_loss:
            self.occ_head = OccupancyHead(
                in_channels=n_cost_ch,
                n_horizons=len(horizons),
            )
        self._aux_loss = None   # set during forward when use_aux_loss=True

    @property
    def recurrent_hidden_state_size(self) -> int:
        return GRU_HIDDEN

    def load_frozen_backbones(self, detector_path, slot_path, device, freeze=True):
        """Load Phase 2 + Phase 3 checkpoints (identical API to GRAM-v2)."""
        ckpt2 = torch.load(detector_path, map_location=device)
        use_pt = ckpt2.get('use_pairwise_temporal', False)
        if use_pt != self._detector_use_pt:
            self.detector = GroupDetector(
                input_dim=INPUT_DIM, n_gnn_layers=3,
                use_pairwise_temporal=use_pt,
            ).to(device)
            self._detector_use_pt = use_pt
        self.detector.load_state_dict(ckpt2['model_state'])
        for p in self.detector.parameters():
            p.requires_grad_(not freeze)

        ckpt3 = torch.load(slot_path, map_location=device)
        self.slot_attn.load_state_dict(ckpt3['model_state'])
        for p in self.slot_attn.parameters():
            p.requires_grad_(not freeze)

        if freeze:
            self.detector.eval();  self.slot_attn.eval()
        else:
            self.detector.train(); self.slot_attn.train()

        mode = "frozen" if freeze else "trainable"
        print(f"[GRAM-Map] GroupDetector ({mode}) loaded from {detector_path}")
        print(f"[GRAM-Map] SlotAttention  ({mode}) loaded from {slot_path}")

    # ------------------------------------------------------------------

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            seq_length = 1
            nenv       = self.nenv
        else:
            seq_length = self.seq_length
            nenv       = self.nenv // self.nminibatch

        device = masks.device

        # ── Unpack observations ───────────────────────────────────────────────
        robot_node     = _reshapeT(inputs['robot_node'],     seq_length, nenv)  # (T,B,1,7)
        temporal_edges = _reshapeT(inputs['temporal_edges'], seq_length, nenv)  # (T,B,1,2)
        spatial_edges  = _reshapeT(inputs['spatial_edges'],  seq_length, nenv)  # (T,B,N,2)
        velocity_edges = _reshapeT(inputs['velocity_edges'], seq_length, nenv)  # (T,B,N,2)
        vis_masks      = _reshapeT(inputs['visible_masks'].float(), seq_length, nenv).bool()
        masks_seq      = _reshapeT(masks, seq_length, nenv)                     # (T,B,1)

        N_actual  = spatial_edges.shape[2]
        h_init    = rnn_hxs['human_node_rnn'][:, 0, :]
        frame_buf = rnn_hxs['human_human_edge_rnn'][:, :N_actual, :]
        prev2     = frame_buf[:, :, :FEAT_DIM]
        prev1     = frame_buf[:, :, FEAT_DIM:]

        # ── Build per-human features and robot state ──────────────────────────
        all_feat21, all_vmask, all_robot, all_p, all_v, all_goal = [], [], [], [], [], []

        for t in range(seq_length):
            m_t = masks_seq[t]
            prev2 = prev2 * m_t.unsqueeze(-1)
            prev1 = prev1 * m_t.unsqueeze(-1)

            curr = _build_7d_frame(spatial_edges[t], velocity_edges[t], vis_masks[t])
            all_feat21.append(torch.cat([prev2, prev1, curr], dim=-1))
            all_vmask.append(vis_masks[t])

            robot_feat = torch.cat([
                temporal_edges[t, :, 0, :],   # (B, 2) robot velocity
                robot_node[t, :, 0, :],        # (B, 7) robot state
            ], dim=-1)
            all_robot.append(robot_feat)

            # Goal in robot-centric frame: [gx-px, gy-py]
            goal_rel = robot_node[t, :, 0, 3:5] - robot_node[t, :, 0, 0:2]  # (B, 2)
            all_goal.append(goal_rel)

            # Current human pos/vel for cost map (clamp inf sentinels → boundary)
            p_t = spatial_edges[t].clamp(-self.synthesizer.grid_range,
                                          self.synthesizer.grid_range)   # (B,N,2)
            v_t = velocity_edges[t].clamp(-5.0, 5.0)                    # (B,N,2)
            all_p.append(p_t)
            all_v.append(v_t)

            prev2 = prev1
            prev1 = curr

        feat21_all = torch.stack(all_feat21, dim=0)  # (T,B,N,21)
        vmask_all  = torch.stack(all_vmask,  dim=0)  # (T,B,N)
        robot_all  = torch.stack(all_robot,  dim=0)  # (T,B,9)
        p_all      = torch.stack(all_p,      dim=0)  # (T,B,N,2)
        v_all      = torch.stack(all_v,      dim=0)  # (T,B,N,2)
        goal_all   = torch.stack(all_goal,   dim=0)  # (T,B,2)

        # ── Batch GroupDetector + SlotAttention over T×B ─────────────────────
        TB         = seq_length * nenv
        feat_flat  = feat21_all.reshape(TB, N_actual, INPUT_DIM)
        vmask_flat = vmask_all.reshape( TB, N_actual)

        no_grad = not self.detector.training
        with torch.no_grad() if no_grad else torch.enable_grad():
            _, _, g, _ = self.detector(feat_flat, vmask_flat)     # (TB,N,64)
            _, alpha = self.slot_attn(g, vmask_flat)              # (TB,K,64),(TB,K,N)
        alpha = alpha.nan_to_num(0.0)   # guard: backbone NaN on first unfreeze steps

        # ── Cost-map synthesis ────────────────────────────────────────────────
        p_flat    = p_all.reshape(TB, N_actual, 2)
        v_flat    = v_all.reshape(TB, N_actual, 2)
        goal_flat = goal_all.reshape(TB, 2)

        cost_stack = self.synthesizer(p_flat, v_flat, vmask_flat, goal_flat, alpha)
        # (TB, C, H, W)

        # ── Auxiliary occupancy loss (self-supervised) ────────────────────────
        if self.use_aux_loss and hasattr(self, 'occ_head'):
            occ_logits = self.occ_head(cost_stack)             # (TB, T, H, W) logits
            occ_tgt    = OccupancyHead.make_targets(
                p_flat, v_flat, vmask_flat,
                self.synthesizer.grid,
                self.synthesizer.horizons,
            )
            loss_val = nn.functional.binary_cross_entropy_with_logits(
                occ_logits, occ_tgt.clamp(0, 1)
            ) * self.aux_lambda
            self._aux_loss = None if torch.isnan(loss_val) else loss_val
        else:
            self._aux_loss = None

        # ── Planning CNN ──────────────────────────────────────────────────────
        map_feat   = self.planner(cost_stack)                   # (TB, 256)

        # ── Robot state fusion ────────────────────────────────────────────────
        rob_feat   = self.robot_mlp(robot_all.reshape(TB, ROBOT_RAW))  # (TB, 64)
        fused      = self.fusion(torch.cat([map_feat, rob_feat], dim=-1))  # (TB, 256)

        # ── GRU unroll ────────────────────────────────────────────────────────
        fused_seq = fused.reshape(seq_length, nenv, GRU_HIDDEN)
        h         = h_init
        outputs   = []

        for t in range(seq_length):
            m_t = masks_seq[t, :, 0]
            h   = h * m_t.unsqueeze(-1)
            h   = self.gru(fused_seq[t], h)
            outputs.append(h)

        x = torch.stack(outputs, dim=0)   # (T, B, 256)

        # ── Actor-Critic ──────────────────────────────────────────────────────
        hidden_actor  = self.actor(x)
        hidden_critic = self.critic(x)

        # ── Update rnn_hxs ───────────────────────────────────────────────────
        new_node = torch.zeros(nenv, 1, GRU_HIDDEN, device=device, dtype=x.dtype)
        new_node[:, 0, :] = h

        buf_rows = rnn_hxs['human_human_edge_rnn'].shape[1]
        new_edge = torch.zeros(nenv, buf_rows, FEAT_DIM * 2, device=device, dtype=x.dtype)
        new_edge[:, :N_actual, :FEAT_DIM] = prev2
        new_edge[:, :N_actual, FEAT_DIM:] = prev1

        rnn_hxs['human_node_rnn']       = new_node
        rnn_hxs['human_human_edge_rnn'] = new_edge

        if infer:
            return (
                self.critic_linear(hidden_critic).squeeze(0),
                hidden_actor.squeeze(0),
                rnn_hxs,
            )
        return (
            self.critic_linear(hidden_critic).view(-1, 1),
            hidden_actor.view(-1, OUTPUT_SIZE),
            rnn_hxs,
        )
