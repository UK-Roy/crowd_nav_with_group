# GRAM-v2 Design Document

End-to-end, perception-aware redesign of GRAM for the realistic crowd-navigation environment.

> **Status:** design only. No code written yet. All decisions here are
> subject to revision once we start implementing. Update this file every
> time we change the design or finish a milestone.

---

## 1. Why redesign GRAM?

### 1.1 What the current GRAM does

`crowd_nav/policy/selfAttn_merge_srnn_grpAttn.py` + `rl/networks/selfAttn_srnn_temp_node_groupAttn.py`:

```
[obs]  →  spatial-self-attn (humans)  ─┐
       →  group-attn (robot ↔ groups) ─┴→ EndRNN (GRU) → actor + critic
       →  GT cluster labels from env
```

Group representation is built in `compute_group_embeddings()` by:
1. Reading **ground-truth `clusters`** from the env (one int per human, `-1` = no group).
2. Hard-masking spatial/velocity edges by cluster id, sum-averaging into per-group features.
3. Concatenating with `group_centroids` and `group_radii` (also from env GT).
4. Feeding the result through a 2-layer MLP.

### 1.2 Limitations

| # | Limitation | Why it matters |
|---|---|---|
| L1 | **Group detection uses ground truth.** `clusters` and `group_centroids` are read straight from `human.group_id` in the simulator. | Won't transfer to a real robot. The whole "group-aware" claim leans on a perfect sensor that doesn't exist. |
| L2 | **`direction_consistency` is dead.** The cosine-similarity computation in `crowd_sim_var_num.py:338-396` is commented out — the field is always `15.0` (sentinel). It's still concatenated into the group embedding. | The network is being fed pure noise as part of the "group" signal. |
| L3 | **Hard cluster assignment is non-differentiable.** `clusters[valid_mask].long()` then `clusters == group_id` masking. | Can't learn group structure at all — gradients don't flow through the clustering. |
| L4 | **Sum-pooling within a group destroys structure.** F-formation geometry, leader-follower asymmetry, member roles — all collapsed into a single mean vector. | Static F-groups vs dynamic LF-groups become indistinguishable after pooling. |
| L5 | **No human-human message passing.** `SpatialEdgeSelfAttn` is a single-layer multi-head attention over per-human embeddings. There is no GNN, no iterative refinement. | The model can't reason "humans i, j, k are walking together" — only "human i is close to robot." |
| L6 | **Heterogeneous velocity profiles ignored.** Phase A draws each individual's `v_pref` from `N(1.34, 0.26)`. The network only sees `vx, vy` — not `‖v‖` or `v_pref`. | Two pedestrians moving at the same direction but very different speeds look similar in the input. |
| L7 | **Static vs dynamic groups identical at the network's input.** Both produce `(centroid, radius)` features; static groups have `vx=vy=0` for all members but the network has to discover that. | Model wastes capacity learning a known categorical distinction. |

---

## 2. Goals for GRAM-v2

1. **End-to-end perception.** No `clusters` field, no `group_centroids`, no `group_radii` from the env. The network detects groups from raw per-human `(p_rel, v_rel, ‖v‖)` observations.
2. **Differentiable group inference.** Use a learned soft adjacency / GNN so gradients flow from the navigation reward back through group structure.
3. **Velocity-aware individual modeling.** Encode each pedestrian's speed magnitude as well as direction.
4. **Multi-stage attention.** Self-attention over humans → graph message passing → group pooling → cross-attention from robot to (individuals, groups).
5. **Auxiliary supervision.** Use GT `human.group_id` labels **only as auxiliary loss during training** (not as input). At test time, the network must run cold.

---

## 3. Proposed architecture

```
                                 ┌─────────────────────────────────────────┐
  per-human raw obs              │  Stage 1: Pedestrian Feature Encoder    │
  [p_rel, v_rel, ‖v_rel‖, mask]  │  MLP: 2 layers, output h_i ∈ R^64       │
                                 └────────────┬────────────────────────────┘
                                              │  H ∈ R^(N × 64)
                                              ▼
                  ┌────────────────────────────────────────────────────┐
                  │  Stage 2: Pairwise Edge Network (learned adjacency)│
                  │  For each pair (i, j):                             │
                  │    e_ij = MLP([h_i, h_j, p_i-p_j, v_i-v_j,         │
                  │                cos(v_i, v_j), ‖p_i-p_j‖])          │
                  │    w_ij = sigmoid(e_ij)  ← "groupness probability" │
                  └────────────┬───────────────────────────────────────┘
                               │  W ∈ R^(N × N)
                               ▼
                  ┌────────────────────────────────────────────────────┐
                  │  Stage 3: GNN Message Passing                      │
                  │  3 layers of:                                      │
                  │    h_i ← h_i + Σ_j w_ij · MLP([h_i, h_j])          │
                  │    LayerNorm + residual                            │
                  │  Output: group-aware embeddings g_i ∈ R^64         │
                  └────────────┬───────────────────────────────────────┘
                               │
                               ├──────────────────┐
                               ▼                  ▼
              ┌─────────────────────────┐   ┌──────────────────────────────┐
              │ Stage 4a: Soft Group    │   │ Stage 4b: Individual         │
              │ Pooling                 │   │ Embeddings (passthrough)     │
              │ K slot-attention heads  │   │ G_ind ∈ R^(N × 64)           │
              │ → K group prototypes    │   │                              │
              │ p_k ∈ R^(K × 64)        │   │                              │
              └─────────────┬───────────┘   └──────────────┬───────────────┘
                            │                              │
                            └──────────┬───────────────────┘
                                       ▼
                  ┌────────────────────────────────────────────────────┐
                  │  Stage 5: Cross-Attention (robot ↔ humans, groups) │
                  │  Q = robot_state, K/V = [G_ind ; p_k]              │
                  │  4-head MultiHeadAttention                         │
                  │  Output: c ∈ R^256                                 │
                  └────────────┬───────────────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────────────────────────────┐
                  │  Stage 6: Temporal Recurrence (GRU)                │
                  │  hidden_t = GRU(c_t, hidden_{t-1})                 │
                  └────────────┬───────────────────────────────────────┘
                               │
                  ┌────────────┴───────────────┐
                  ▼                            ▼
              actor head                 critic head
               (PPO)                      (PPO)
```

### 3.1 Stage details

**Stage 1 — Pedestrian Feature Encoder**
- Input per human: `[p_rel_x, p_rel_y, v_rel_x, v_rel_y, ‖v_rel‖, sin(θ_v), cos(θ_v)]` — 7 dims
- Mask out invalid humans (the `15.0` sentinel slots) before encoding so we never feed sentinels into the MLP.
- 2-layer MLP: `7 → 64 → 64`, ReLU, dropout 0.1.

**Stage 2 — Pairwise Edge Network**
- Compute pairwise edge features for all `N×N` pairs (N=20 max).
- Edge feature: `[h_i, h_j, p_i - p_j, v_i - v_j, cos(v_i, v_j), ‖p_i - p_j‖]` — 134 dims.
- 2-layer MLP: `134 → 64 → 1`, sigmoid.
- Mask self-loops to 0; mask invalid-human pairs to 0.
- Output `W ∈ [0,1]^(N×N)` — soft groupness adjacency.

**Stage 3 — GNN Message Passing**
- 3 graph-conv layers using `W` as the soft adjacency:
  ```
  h_i^(l+1) = h_i^(l) + LayerNorm( Σ_j W[i,j] · MLP_l([h_i^(l), h_j^(l)]) )
  ```
- Final output `g_i ∈ R^64` — the human embedding now carries community context.

**Stage 4a — Soft Group Pooling (Slot Attention)**
- Use **slot attention** (Locatello et al. 2020) with `K = max_groups = 3` slots.
- Each slot iteratively attends to humans, producing K group prototypes that compete for membership.
- Why slot attention: it naturally handles "variable number of groups", produces permutation-invariant slots, and is fully differentiable. Alternative: K-means via EM-style attention (DiffPool).

**Stage 4b — Individual Passthrough**
- The post-GNN human embeddings `g_i` are kept and used in cross-attention alongside group prototypes. This way the policy can attend to a single threatening individual even if it's not part of a detected group.

**Stage 5 — Robot ↔ (Humans, Groups) Cross-Attention**
- Robot state: `[p_robot, v_robot, p_goal, ‖p_robot - p_goal‖, heading]` → MLP → `q ∈ R^64`.
- Keys/values: concatenate individual embeddings (N) and group prototypes (K) → `(N+K) × 64`.
- 4-head multi-head cross-attention.
- Output `c ∈ R^256`.

**Stage 6 — Temporal Recurrence**
- Standard GRU. Maintains memory across timesteps. Hidden size 256.

**Stage 7 — Actor + Critic**
- 2-layer MLPs, same as current GRAM.

---

## 4. Auxiliary losses (training-time only)

Auxiliary supervision keeps the perception modules grounded while the RL signal is sparse.

| Loss | Target | Source | Weight |
|---|---|---|---|
| **L_group** | Pairwise edge weights `W[i,j]` should match GT membership | `crowd_sim_var_num.py:328-334` reads `human.group_id` — use it as the supervision target, NOT as input | 0.5 |
| **L_velocity** | Predict next-step relative velocity for each visible human | Self-supervised from rollout buffer | 0.1 |
| **L_PPO** | Standard PPO clipped surrogate + value loss | RL reward signal | 1.0 |

`L_group` is implemented as **binary cross-entropy** over all valid pairs:
```
L_group = BCE(W[i,j], 1[human.group_id_i == human.group_id_j])
```
where pairs with at least one human in `group_id == None` are masked out.

**Important:** `L_group` only runs during training. At test/deployment time, the env will not provide `clusters` or `group_id` — only the raw 7-d per-human features. This forces the network to truly learn to detect groups.

---

## 5. Environment-side changes

We need to **expose raw per-human features** and **stop relying on the GT cluster path at inference time**. Concrete edits:

| File | Change |
|---|---|
| `crowd_sim/envs/crowd_sim_var_num.py` | Add a flag `config.gram_v2.expose_raw_only`. When `True`, return only `[robot_node, temporal_edges, spatial_edges, velocity_edges, visible_masks]` plus `‖v_rel‖` per human. **Do not return `clusters`, `group_centroids`, `group_radii`, `group_members` at inference.** |
| `crowd_sim/envs/crowd_sim_var_num.py` | Add `gt_group_id_per_human` (shape `(max_human_num,)`) ONLY when in training mode for use by L_group. Mark it clearly so it never leaks into inference. |
| `crowd_nav/configs/config.py` | Add `gram_v2` config block: `enabled`, `K_max_groups`, `gnn_layers`, `aux_loss_weights`, `expose_raw_only`. |
| `rl/networks/__init__.py` (or factory) | Register the new policy key `'gram_v2'`. |
| `crowd_nav/policy/policy_factory.py` | Add `gram_v2` entry pointing to the new network class. |

We should keep the existing `selfAttn_merge_srnn_grpAttn` path **untouched** so the original GRAM remains a baseline.

---

## 6. Implementation roadmap

Each phase has a measurable acceptance criterion. We move to the next phase only when the current one passes.

### Phase 1 — Perception sanity check (offline, no RL)
- Implement Stage 1 (encoder) + Stage 2 (pairwise edges) only.
- Train with `L_group` alone using a fixed dataset of env rollouts (random-policy).
- **Success criterion:** F1 > 0.85 on per-pair group prediction across held-out test seeds.
- Expected effort: 2–3 days.

### Phase 2 — GNN refinement
- Add Stage 3 (GNN message passing).
- Re-train `L_group` end-to-end.
- **Success criterion:** F1 > 0.90 (i.e., GNN actually helps), plus ARI > 0.7 on the implied clustering vs GT.
- Expected effort: 2 days.

### Phase 3 — Slot attention pooling
- Add Stage 4a (slot attention, K=3 slots).
- Train slot attention on auxiliary group-classification loss (each slot should align with one GT group).
- **Success criterion:** Hungarian-matched purity > 0.85; visualize slot assignments and confirm they cluster spatially correctly.
- Expected effort: 2 days.

### Phase 4 — Full network + PPO training
- Wire Stage 5 (cross-attention), Stage 6 (GRU), Stage 7 (actor/critic).
- Train end-to-end with PPO + auxiliary losses on the realistic env.
- **Success criterion:** SR ≥ 0.85, GCR ≤ 0.020 at 100 seeds — beating both ORCA (B0 = 0.78/0.026 in Exp 09) and original GRAM.
- Expected effort: 3–5 days of GPU training + tuning.

### Phase 5 — Ablations + paper
- Ablate: no GNN, no slot attention, no aux loss, no velocity-magnitude features.
- Compare against ORCA, ORCA+TAGA, GARN (when its training finishes), original GRAM.
- Expected effort: 1 week.

---

## 7. Open design questions

1. **K = 3 slots or learnable K?** Slot attention requires fixed K. Realistically the env has `num_groups = 3`, so K=3 is fine. If we ever vary `group.num_groups`, switch to set transformer.
2. **Should we share the encoder MLP between Stage 1 and Stage 4b?** Yes — the post-GNN embedding `g_i` IS the individual representation; no need for two encoders.
3. **Slot attention iterations?** Default 3; increase to 5 if pooling underfits.
4. **Group prototype initialization?** Random vs learned-mean-init. Start with random + Gaussian.
5. **How to handle the `v_pref` heterogeneity (Phase A)?** Two options: (a) feed `‖v_rel‖` as the 5th dim of the per-human input (current plan), (b) also feed the inferred `v_pref` (max observed speed window) — but that requires a buffer. **Decision:** start with (a), add (b) only if needed.
6. **Where does TAGA fit?** GRAM-v2 is the trained policy; TAGA is the reactive add-on. By the [permanent rule](TAGA_EXPERIMENT_LOG.md#permanent-rules-learned), TAGA should NOT be applied to GRAM-v2 (learning-based policy). The two are alternative integration strategies for the same problem.

---

## 8. Risks / things that may go wrong

| Risk | Mitigation |
|---|---|
| Slot attention collapses (all slots learn the same prototype) | Add slot diversity loss; init slots from random Gaussian; limit iterations. |
| GNN with soft adjacency overfits to GT membership and ignores RL signal | Anneal `L_group` weight from 0.5 → 0.05 over training; use it as a regularizer, not a target. |
| Cold-start at test time fails because env still emits `clusters` and the policy comes to depend on them via the dataloader | Strict separation: at inference, the network sees ONLY raw features. Add an integration test that asserts no `clusters` access on inference path. |
| 7-d per-human input is too lean for the GNN to discover groups | Add `time-since-last-velocity-change`, or use a small temporal MLP over the last 4 frames of (p, v) per human. |
| End-to-end training is unstable with 3 losses | Train Stages 1–3 with only `L_group` first (Phase 1–2), then freeze partial weights and add RL loss. |

---

## 9. Milestones / log

| Date | Phase | Milestone | Outcome |
|---|---|---|---|
| 2026-04-30 | — | Design doc created | this file |
| | Phase 1 | encoder + edges + L_group offline | (pending) |
| | Phase 2 | GNN added | (pending) |
| | Phase 3 | slot attention pooling | (pending) |
| | Phase 4 | full network + PPO training | (pending) |
| | Phase 5 | ablations + paper writeup | (pending) |

> **Update this table after every phase completes.** Include the actual SR/GCR
> numbers and a one-line verdict.

---

## 10. References

- **Slot Attention** — Locatello et al., NeurIPS 2020. *Object-Centric Learning with Slot Attention.* — for soft group pooling.
- **DiffPool** — Ying et al., NeurIPS 2018. *Hierarchical Graph Representation Learning with Differentiable Pooling.* — alternative to slot attention.
- **GAT / GATv2** — Velickovic et al. 2018; Brody et al. 2022. — for the GNN message-passing layer.
- **Set Transformer** — Lee et al., ICML 2019. — alternative if K must be variable.
- **Original GRAM** — `selfAttn_srnn_temp_node_groupAttn.py` in this repo. The baseline we're replacing.
- **TAGA experiment log** — `TAGA_EXPERIMENT_LOG.md`. Establishes that reactive overrides hurt learning-based policies — motivation for "build group awareness into the policy itself."
- **Realistic env spec** — `env_spec.yaml`. Source of truth for observation shapes, action bounds, reward structure.
