"""
GRAM-v2 Phase 4 — end-to-end perception-aware navigation policy.

Pipeline (per timestep):
  1. Build 21-d per-human features from a 2-frame rolling buffer + current obs
  2. GroupDetector (Phase 2, frozen) → g : (B, N, 64)  GNN-refined embeddings
  3. SlotAttention  (Phase 3, frozen) → slots : (B, K, 64)  group prototypes
  4. Robot-query MLP                  → q : (B, 64)
  5. CrossAttention: robot attends to concat(g, slots)  → c : (B, 64)
  6. Context projection               → (B, 256)
  7. GRU(c, h)                        → h_new : (B, 256)  temporal memory
  8. Actor / Critic heads             → value, action-distribution features

rnn_hxs layout  (keys must match RolloutStorage allocation in train.py):
  'human_node_rnn'       (B, 1, GRU_HIDDEN=256)  — GRU hidden state
  'human_human_edge_rnn' (B, 21, FEAT_DIM*2=14)  — per-human 2-frame buffer
                          [:, :MAX_HUMANS, :7]  = frame t-2
                          [:, :MAX_HUMANS, 7:]  = frame t-1

Run with:
  --env-name CrowdSimVarNum-v0
  --human_node_rnn_size 256
  --human_human_edge_rnn_size 14
"""

import torch
import torch.nn as nn

from crowd_nav.gram_v2.models import GroupDetector, FEAT_DIM, INPUT_DIM
from crowd_nav.gram_v2.slot_attention import SlotAttention

MAX_HUMANS  = 20
GRU_HIDDEN  = 256
EMBED_DIM   = 64
K_SLOTS     = 3
ROBOT_RAW   = 9    # temporal_edges(2) + robot_node(7)
OUTPUT_SIZE = 256
GEO_DIM     = 7    # centroid(2)+group_vel(2)+spread(1)+approach_rate(1)+cohesion(1)
SPATIAL_DIM = 5    # range(1)+sin_bearing(1)+cos_bearing(1)+TCPA(1)+DCPA(1)


def _reshapeT(tensor: torch.Tensor, seq_length: int, nenv: int) -> torch.Tensor:
    """Reshape (seq*nenv, ...) → (seq, nenv, ...) for sequence processing."""
    shape = tensor.size()[1:]
    return tensor.unsqueeze(0).reshape((seq_length, nenv, *shape))


def _build_7d_frame(spatial: torch.Tensor,
                    velocity: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    Build 7-d per-human feature frame.
    spatial  : (B, N, 2)  relative position  [px, py]
    velocity : (B, N, 2)  relative velocity  [vx, vy]
    mask     : (B, N)     bool, True = visible
    Returns  : (B, N, 7)  [px, py, vx, vy, v_norm, sin_θ, cos_θ], 0 for invisible
    """
    vx     = velocity[..., 0:1]
    vy     = velocity[..., 1:2]
    v_norm = (vx ** 2 + vy ** 2).sqrt()
    sin_t  = vy / (v_norm + 1e-6)
    cos_t  = vx / (v_norm + 1e-6)
    frame  = torch.cat([spatial, velocity, v_norm, sin_t, cos_t], dim=-1)  # (B, N, 7)
    return frame * mask.unsqueeze(-1).float()


class GRAMV2Network(nn.Module):
    """
    Full GRAM-v2 network.  Matches the interface expected by model.py Policy:
      __init__(obs_space_dict, args)
      forward(inputs, rnn_hxs, masks, infer=False)  →  (value, actor_feat, rnn_hxs)
    """

    def __init__(self, obs_space_dict, args):
        super().__init__()
        self.is_recurrent = True
        self.args         = args

        self.human_num  = MAX_HUMANS
        self.seq_length = args.num_steps
        self.nenv       = args.num_processes
        self.nminibatch = args.num_mini_batch
        self.output_size = OUTPUT_SIZE
        # default True keeps old behaviour; set False in Stage 1-2 to skip slots
        self.use_slots  = getattr(args, 'gram_v2_use_slots', True)

        # Attributes expected by evaluation.py
        self.human_node_rnn_size       = GRU_HIDDEN      # 256
        self.human_human_edge_rnn_size = FEAT_DIM * 2    # 14

        # ── Frozen perception backbone (rebuilt in load_frozen_backbones) ────────
        self.detector  = GroupDetector(input_dim=INPUT_DIM, n_gnn_layers=3)
        self.slot_attn = SlotAttention(embed_dim=EMBED_DIM, K=K_SLOTS)
        self._detector_use_pt = False  # updated by load_frozen_backbones

        # Always-valid null token — prevents all-keys-masked NaN in cross-attn
        self.null_embed    = nn.Parameter(torch.zeros(EMBED_DIM))
        # Projects 7-d geometric group summary → slot embedding space (Stage 3+)
        self.slot_geo_proj = nn.Linear(GEO_DIM, EMBED_DIM)

        # Robot-centric spatial encoder: range, bearing, TCPA, DCPA → 64-d
        self.spatial_encoder = nn.Sequential(
            nn.Linear(SPATIAL_DIM, EMBED_DIM), nn.ReLU(),
        )

        # ── Robot-query MLP ───────────────────────────────────────────────────
        self.robot_query_mlp = nn.Sequential(
            nn.Linear(ROBOT_RAW, 128), nn.ReLU(),
            nn.Linear(128, EMBED_DIM),
        )

        # ── Cross-Attention (robot→humans+slots) ───────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=4, batch_first=True, dropout=0.0
        )

        # ── Temporal GRU ──────────────────────────────────────────────────────
        self.context_proj = nn.Linear(EMBED_DIM, GRU_HIDDEN)
        self.gru          = nn.GRUCell(GRU_HIDDEN, GRU_HIDDEN)

        # ── Actor-Critic heads ───────────────────────────────────────────────
        self.actor  = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN), nn.Tanh(),
        )
        self.critic_linear = nn.Linear(GRU_HIDDEN, 1)

    @property
    def recurrent_hidden_state_size(self) -> int:
        return GRU_HIDDEN

    def load_frozen_backbones(self, detector_path: str, slot_path: str,
                               device: torch.device, freeze: bool = True):
        """Load Phase 2 and Phase 3 checkpoints.  freeze=False trains them."""
        ckpt2 = torch.load(detector_path, map_location=device)
        use_pt = ckpt2.get('use_pairwise_temporal', False)
        if use_pt != self._detector_use_pt:
            self.detector = GroupDetector(input_dim=INPUT_DIM, n_gnn_layers=3,
                                          use_pairwise_temporal=use_pt).to(device)
            self._detector_use_pt = use_pt
        self.detector.load_state_dict(ckpt2['model_state'])
        for p in self.detector.parameters():
            p.requires_grad_(not freeze)

        ckpt3 = torch.load(slot_path, map_location=device)
        self.slot_attn.load_state_dict(ckpt3['model_state'])
        for p in self.slot_attn.parameters():
            p.requires_grad_(not freeze)

        if freeze:
            self.detector.eval()
            self.slot_attn.eval()
        else:
            self.detector.train()
            self.slot_attn.train()

        mode = "frozen" if freeze else "trainable"
        print(f"[GRAM-v2] Loaded GroupDetector ({mode}) from {detector_path}")
        print(f"[GRAM-v2] Loaded SlotAttention  ({mode}) from {slot_path}")

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            seq_length = 1
            nenv       = self.nenv
        else:
            seq_length = self.seq_length
            nenv       = self.nenv // self.nminibatch

        device = masks.device

        # ── Unpack and reshape observations → (T, B, ...) ────────────────────
        robot_node     = _reshapeT(inputs['robot_node'],     seq_length, nenv)  # (T,B,1,7)
        temporal_edges = _reshapeT(inputs['temporal_edges'], seq_length, nenv)  # (T,B,1,2)
        spatial_edges  = _reshapeT(inputs['spatial_edges'],  seq_length, nenv)  # (T,B,N,2)
        velocity_edges = _reshapeT(inputs['velocity_edges'], seq_length, nenv)  # (T,B,N,2)
        vis_masks      = _reshapeT(inputs['visible_masks'].float(), seq_length, nenv).bool()  # (T,B,N)
        masks_seq      = _reshapeT(masks, seq_length, nenv)                     # (T,B,1)

        # ── Unpack recurrent state ────────────────────────────────────────────
        # N_actual is derived from observations so the network works with any human_num
        N_actual  = spatial_edges.shape[2]
        h_init    = rnn_hxs['human_node_rnn'][:, 0, :]              # (B, 256)
        frame_buf = rnn_hxs['human_human_edge_rnn'][:, :N_actual, :]  # (B, N_actual, 14)
        prev2     = frame_buf[:, :, :FEAT_DIM]                       # (B, N_actual, 7)
        prev1     = frame_buf[:, :, FEAT_DIM:]                       # (B, N_actual, 7)

        # ── Build 21-d features and robot state for each timestep ─────────────
        all_feat21 = []
        all_vmask  = []
        all_robot  = []

        for t in range(seq_length):
            m_t = masks_seq[t]  # (B, 1): 0 = episode boundary → reset buffer

            prev2 = prev2 * m_t.unsqueeze(-1)   # (B, N, 7)  zero on new episode
            prev1 = prev1 * m_t.unsqueeze(-1)

            curr = _build_7d_frame(
                spatial_edges[t], velocity_edges[t], vis_masks[t]
            )  # (B, N, 7)

            all_feat21.append(torch.cat([prev2, prev1, curr], dim=-1))  # (B, N, 21)
            all_vmask.append(vis_masks[t])                               # (B, N)

            robot_feat = torch.cat([
                temporal_edges[t, :, 0, :],   # (B, 2)
                robot_node[t, :, 0, :],        # (B, 7)
            ], dim=-1)                         # (B, 9)
            all_robot.append(robot_feat)

            prev2 = prev1
            prev1 = curr

        feat21_all = torch.stack(all_feat21, dim=0)   # (T, B, N, 21)
        vmask_all  = torch.stack(all_vmask,  dim=0)   # (T, B, N)
        robot_all  = torch.stack(all_robot,  dim=0)   # (T, B, 9)

        # ── Batch GroupDetector + SlotAttention over T×B ─────────────────────
        TB = seq_length * nenv
        feat_flat  = feat21_all.reshape(TB, N_actual, INPUT_DIM)
        vmask_flat = vmask_all.reshape( TB, N_actual)

        no_grad = not self.detector.training
        with torch.no_grad() if no_grad else torch.enable_grad():
            W_final, _, g, _ = self.detector(feat_flat, vmask_flat)  # (TB,N,N), (TB,N,64)
            if self.use_slots:
                slots, alpha = self.slot_attn(g, vmask_flat)         # (TB,K,64), (TB,K,N)

        # ── Robot-centric spatial encoding (range, bearing, TCPA, DCPA) ────────
        p_cur = feat_flat[..., -FEAT_DIM:-FEAT_DIM+2]          # (TB, N, 2) rel pos
        v_cur = feat_flat[..., -FEAT_DIM+2:-FEAT_DIM+4]        # (TB, N, 2) rel vel
        r     = p_cur.norm(dim=-1, keepdim=True).clamp(min=1e-6)     # range
        sin_b = p_cur[..., 1:2] / r                                   # bearing sin
        cos_b = p_cur[..., 0:1] / r                                   # bearing cos
        v_sq  = (v_cur ** 2).sum(-1, keepdim=True).clamp(min=1e-6)
        pv    = (p_cur * v_cur).sum(-1, keepdim=True)
        tcpa  = (-pv / v_sq).clamp(0, 10)                             # seconds to CPA
        dcpa  = ((r ** 2 * v_sq - pv ** 2).clamp(min=0).sqrt() / v_sq.sqrt())  # dist at CPA
        spatial_enc = self.spatial_encoder(
            torch.cat([r, sin_b, cos_b, tcpa, dcpa], dim=-1)
        ) * vmask_flat.unsqueeze(-1).float()                           # (TB, N, 64)
        g = g + spatial_enc

        # ── Geometric group summaries augmenting slot embeddings ──────────────
        if self.use_slots:
            p = feat_flat[..., -FEAT_DIM:-FEAT_DIM+2]               # (TB, N, 2) relative pos
            v = feat_flat[..., -FEAT_DIM+2:-FEAT_DIM+4]             # (TB, N, 2) relative vel

            alpha_n = alpha / (alpha.sum(-1, keepdim=True) + 1e-8)  # (TB, K, N) normalised

            centroid  = torch.bmm(alpha_n, p)                        # (TB, K, 2)
            group_vel = torch.bmm(alpha_n, v)                        # (TB, K, 2)

            # Spread: weighted RMS distance from centroid
            diff_sq = ((p.unsqueeze(1) - centroid.unsqueeze(2)) ** 2).sum(-1)  # (TB,K,N)
            spread  = (alpha_n * diff_sq).sum(-1, keepdim=True).sqrt()         # (TB,K,1)

            # Approach rate: closing speed of group centroid toward robot (origin)
            c_norm        = centroid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            approach_rate = -(centroid * group_vel).sum(-1, keepdim=True) / c_norm  # (TB,K,1)

            # Cohesion: expected pairwise groupness within each slot
            coh_num  = (torch.bmm(alpha, W_final) * alpha).sum(-1)  # (TB, K)
            cohesion = (coh_num / (alpha * alpha).sum(-1).clamp(min=1e-8)).unsqueeze(-1)  # (TB,K,1)

            geo   = torch.cat([centroid, group_vel, spread, approach_rate, cohesion], dim=-1)  # (TB,K,7)
            slots = slots + self.slot_geo_proj(geo)

        # ── Cross-Attention: robot query attends to humans [+ slots] ─────────
        q = self.robot_query_mlp(robot_all.reshape(TB, ROBOT_RAW)).unsqueeze(1)  # (TB, 1, 64)

        if self.use_slots:
            kv        = torch.cat([g, slots], dim=1)                               # (TB, N+K, 64)
            slot_ok   = torch.ones(TB, K_SLOTS, dtype=torch.bool, device=device)
            key_valid = torch.cat([vmask_flat, slot_ok], dim=1)                    # (TB, N+K)
        else:
            kv        = g                                                           # (TB, N, 64)
            key_valid = vmask_flat                                                  # (TB, N)

        # Append null token — always valid, prevents all-keys-masked NaN
        null_tok  = self.null_embed.unsqueeze(0).unsqueeze(0).expand(TB, 1, -1)   # (TB, 1, 64)
        null_ok   = torch.ones(TB, 1, dtype=torch.bool, device=device)
        kv        = torch.cat([kv, null_tok], dim=1)
        key_valid = torch.cat([key_valid, null_ok], dim=1)

        key_pad = ~key_valid                                      # True = ignore

        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_pad)
        c = attn_out.squeeze(1)           # (TB, 64)
        c = self.context_proj(c)          # (TB, 256)

        # ── GRU unroll over T timesteps ───────────────────────────────────────
        c_seq    = c.reshape(seq_length, nenv, GRU_HIDDEN)   # (T, B, 256)
        h        = h_init                                     # (B, 256)
        outputs  = []

        for t in range(seq_length):
            m_t = masks_seq[t, :, 0]     # (B,)
            h   = h * m_t.unsqueeze(-1)  # reset hidden on episode boundary
            h   = self.gru(c_seq[t], h)
            outputs.append(h)

        x = torch.stack(outputs, dim=0)  # (T, B, 256)

        # ── Actor-Critic heads ────────────────────────────────────────────────
        hidden_actor  = self.actor(x)    # (T, B, 256)
        hidden_critic = self.critic(x)

        # ── Update rnn_hxs ───────────────────────────────────────────────────
        new_node = torch.zeros(nenv, 1, GRU_HIDDEN, device=device, dtype=x.dtype)
        new_node[:, 0, :] = h           # h = last GRU hidden state

        buf_rows = rnn_hxs['human_human_edge_rnn'].shape[1]  # = N_actual + 1
        new_edge = torch.zeros(nenv, buf_rows, FEAT_DIM * 2,
                               device=device, dtype=x.dtype)
        new_edge[:, :N_actual, :FEAT_DIM] = prev2  # frame t-1 → t-2 for next call
        new_edge[:, :N_actual, FEAT_DIM:] = prev1  # frame t   → t-1 for next call

        rnn_hxs['human_node_rnn']       = new_node
        rnn_hxs['human_human_edge_rnn'] = new_edge

        # ── Return ────────────────────────────────────────────────────────────
        if infer:
            return (
                self.critic_linear(hidden_critic).squeeze(0),  # (B, 1)
                hidden_actor.squeeze(0),                        # (B, 256)
                rnn_hxs,
            )
        else:
            return (
                self.critic_linear(hidden_critic).view(-1, 1),   # (T*B, 1)
                hidden_actor.view(-1, OUTPUT_SIZE),               # (T*B, 256)
                rnn_hxs,
            )
