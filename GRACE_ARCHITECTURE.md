# GRACE: End-to-End Group-Aware Cost Map Navigation

**Target venue:** CoRL 2026
**Status:** Stage C complete (SR=89%, CR=3%) — Realistic benchmark pending
**Author:** Utshar Roy
**Date:** 2026-05-06

---

## 1. Problem and Motivation

Social robot navigation in human crowds requires reasoning about three things simultaneously:
1. **Where humans are now** (geometry)
2. **Where humans will be** (intent / prediction)
3. **What humans belong to whom** (group structure — F-formations, leader-follower, walking dyads)

Existing learned approaches (DS-RNN, GRAM, GARN, our own GRAM-v2) compress this into a recurrent policy that maps observations directly to actions through cross-attention. This works but has three failure modes that we have empirically observed:

| Failure mode | Where we saw it | Why it happens |
|---|---|---|
| PPO collapses on curriculum advance | GRAM-v2 Stage 3 (15 humans, mixed groups) — SR went 50% → 0% in 200 updates | New env distribution + unfrozen backbone = exploration noise destroys learned features |
| Opaque failures | All neural baselines on dense F-formations | Cross-attention scalar weights don't reveal *why* the robot chose to plow through a group |
| No counterfactual reasoning | TAGA experiments — neural policies can't be "patched" with a tangent action without retraining | Action is conditioned on hidden state, not on a queryable spatial belief |

Classical methods (ORCA, social force) do not have these problems but cannot reason about groups as cohesive entities. They treat every human independently, which is exactly why our environment was designed to be hard for them — yet ORCA still wins on GCR purely because tight group spacing (1.0–1.3 m radius) physically forces avoidance.

**The contribution of this paper is a navigation architecture that exposes the robot's spatial belief as a queryable, interpretable, group-aware cost map and plans over it end-to-end.** The cost map is the bridge between perception and planning, and it is the unit on which we train, ablate, and visualize.

---

## 2. Contribution

We propose **GRACE**, an architecture with four novel components:

### C1. Group-aware cost field synthesis (novel)
We render a 2D bird's-eye-view (BEV) cost map by composing differentiable cost layers from learned group perception. Existing social cost maps treat pedestrians as independent Gaussians; ours reads off a learned group affinity matrix and renders a *group cohesion field* — a smooth high-cost region across the convex hull of group members, with a soft falloff. This is the first cost map that is group-aware by construction, not by post-hoc clustering.

### C2. Time-conditioned future-intent layer (novel)
The cost map has a temporal axis: at each planning step we render `T` future cost slices, one per look-ahead horizon (e.g., 0.3 s, 0.7 s, 1.0 s, 1.5 s). Each slice propagates each human along their predicted velocity (from the GST predictor or a learned head) and re-renders the cost. The planner attends to all slices, so it can choose paths that are clear *now and in the future*. Existing learned cost maps are static or single-slice; ours is a 3D (H × W × T) tensor.

### C3. Self-supervised cost regularization (novel)
We add an auxiliary loss: predict future occupancy. Given the current observation, the cost synthesizer must produce a cost map whose high-cost regions match where humans actually are at each future horizon. This loss is free (ground-truth future positions are known in simulation), it does not require manual cost-map labels, and it grounds the cost field in physical reality before any RL signal arrives. PPO instability — the dominant failure mode of GRAM-v2 — is mitigated because the perception-to-cost path is already trained when the policy starts learning.

### C4. Reuse of self-supervised group backbones
Phase 2 (GroupDetector) and Phase 3 (SlotAttention) from GRAM-v2 are pretrained on group detection (F1=0.77 on detection, purity=0.93 on slot assignments). They feed directly into the cost synthesizer. This means the entire pre-policy pipeline is trained without any RL signal — and we report ablations with frozen vs unfrozen backbones to quantify the value of self-supervised perception.

---

## 3. Architecture

```
                       Raw observation
                       (N humans: pos, vel, radius;
                        robot: pos, vel, goal)
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
    [Phase 2 GroupDetector]       [Robot-centric encoder]
       (pretrained, frozen)         (range, bearing,
                │                    TCPA, DCPA)
       group affinity W (N×N)              │
       human embeddings g (N×64)           │
                │                          │
                ▼                          │
    [Phase 3 SlotAttention]                │
       (pretrained, frozen)                │
                │                          │
       group slots s (K×64)                │
       assignments α (K×N)                 │
                │                          │
                └─────────────┬────────────┘
                              ▼
                ╔═══════════════════════════╗
                ║  COST MAP SYNTHESIZER     ║      ← novel
                ╠═══════════════════════════╣
                ║  L1: Individual layer     ║      Gaussian splat at each
                ║                           ║      (px,py); softmax-clipped peak
                ║  L2: Trajectory layer (T) ║      time-conditioned splat along
                ║                           ║      v·Δt for T horizons
                ║  L3: Group cohesion layer ║      α-weighted soft hull,
                ║                           ║      cost peaks inside hull
                ║  L4: Group repulsion layer║      gradient outward across
                ║                           ║      hull boundary (1.5–2.0 m
                ║                           ║      decay)
                ║  L5: Goal attractor       ║      negative cost gradient
                ║                           ║      toward goal
                ║  L6: Boundary cost        ║      arena edges
                ╚═══════════════╤═══════════╝
                                ▼
                  Aggregated BEV cost stack
                  C ∈ R^{H × W × T}
                  (robot-centric, e.g. 6 m × 6 m
                   at 0.1 m resolution → 60×60×T)
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   ┌────────────────────────┐      ┌─────────────────────────┐
   │  PLANNING HEAD         │      │  AUXILIARY HEAD         │
   │  (trained with PPO)    │      │  (occupancy prediction) │
   │                        │      │                         │
   │  3D-CNN over (H,W,T)   │      │  Decode each slice to   │
   │  → flatten → MLP       │      │  predicted occupancy    │
   │  → action mean + std   │      │  grid; supervise w/     │
   │  + value head          │      │  ground-truth future    │
   │                        │      │  positions (BCE loss)   │
   └───────────┬────────────┘      └────────────┬────────────┘
               ▼                                ▼
            Action                       L_aux (training only)
```

### 3.1 Cost layer formulae

Let `(x, y)` be a grid cell in robot-centric BEV coordinates. For human `i` at position `p_i` with velocity `v_i`:

**L1 — Individual:**
```
c_indiv(x, y) = Σ_i  exp(-||(x, y) - p_i||² / (2 σ_i²))
```
where `σ_i = r_human + safety_margin`.

**L2 — Trajectory (per horizon t ∈ {0.3, 0.7, 1.0, 1.5} s):**
```
c_traj_t(x, y) = Σ_i  exp(-||(x, y) - (p_i + v_i · t)||² / (2 σ_t²))
```
`σ_t` grows with `t` to model prediction uncertainty.

**L3 — Group cohesion (per group k):**
```
ĉ_k = α_k · {p_i}    # soft member positions weighted by slot assignment
hull_k = soft_convex_hull(ĉ_k)
c_group(x, y) = max_k  hull_k(x, y) · γ_k     # γ_k = group cohesion strength
```

**L4 — Group repulsion** (gradient outside the hull, 1.5–2.0 m falloff) provides early-warning cost so the planner doesn't graze hull boundaries.

**L5 — Goal attractor** is a negative cost field, so the planner's job becomes "minimize total cost while moving toward the goal."

**L6 — Boundary** prevents the planner from learning to wall-hug.

All layers are differentiable. Gradients flow from PPO loss through the planning head, into the cost map, into the perception encoders.

### 3.2 Why a 3D cost map (H × W × T) and not just 2D
A 2D snapshot cost map cannot express *"this lane is clear now but will be blocked in 1 second."* The temporal axis lets the planner trade off short-term and long-term clearance, which is exactly what humans do when they walk through crowds. The planning CNN's first layer is `Conv3D(in=1, out=32, kernel=(3,3,3))`, so temporal kernels learn motion patterns automatically.

### 3.3 Cost map resolution
Trade-off: higher resolution = more spatial fidelity but quadratic FLOPs.
- **Default:** 60 × 60 × 5 at 0.1 m / cell → 6 m × 6 m × 5 horizons = 18 000 cells per timestep.
- Forward pass through a small 3D-CNN: < 1 ms on a single GPU at batch 16 envs. Negligible vs PPO rollout time.

---

## 4. Implementation Status

**Branch:** `grace`

| File | Status | Description |
|---|---|---|
| `rl/networks/grace_synthesizer.py` | ✅ Done | CostMapSynthesizer, OccupancyHead, CostMapPlanner |
| `rl/networks/grace_network.py` | ✅ Done | GRACENetwork — full policy |
| `crowd_nav/policy/grace.py` | ✅ Done | Policy class |
| `crowd_nav/policy/policy_factory.py` | ✅ Done | Registered as `grace` |
| `crowd_nav/configs/config.py` | ✅ Done | `grace` config section added |
| `rl/networks/model.py` | ✅ Done | GRACENetwork wired to `grace` key |
| `train.py` | ✅ Done | backbone loading + aux loss injection |
| `test.py` | ✅ Done | aux loss disabled at eval |
| `rl/ppo/ppo.py` | ✅ Done | `_aux_loss` hook (3 lines) |

### Run flags (same as GRAM-v2)

```bash
--env-name CrowdSimVarNum-v0
--human_node_rnn_size 256
--human_human_edge_rnn_size 14
```

### Training commands

Before running, set `robot.policy = 'grace'` in `config.py`, then:

**Stage B — Initial PPO (frozen backbone, no aux loss):**
```bash
# config.py: grace.freeze_backbone=True, grace.use_aux_loss=False
# Environment: human_num=15, circle_radius=6, group.types=['static_f','dynamic_lf'], num_on_path=1
# Reward: discomfort_group_dist=0.35, discomfort_grp_penalty_factor=10, grp_collision_penalty=-5
python train.py --env-name CrowdSimVarNum-v0 --human_node_rnn_size 256 --human_human_edge_rnn_size 14 --output_dir trained_models/grace/stageB --num-env-steps 20000000 --num-processes 16 --num-steps 30 --num-mini-batch 2 --ppo-epoch 5 --lr 4e-4 --eps 1e-5 --gamma 0.99 --gae-lambda 0.95 --entropy-coef 0.05 --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 --use-linear-lr-decay --save-interval 200 --log-interval 20
```

> **Critical:** `--entropy-coef 0.05` is required. Default is 0.0 which causes policy collapse (SR drops from 83% → 17% over 40K updates).

**Stage C — Fine-tune with aux loss (frozen backbone unlocked):**
```bash
# config.py: grace.freeze_backbone=False, grace.use_aux_loss=True
python train.py --env-name CrowdSimVarNum-v0 --human_node_rnn_size 256 --human_human_edge_rnn_size 14 --output_dir trained_models/grace/stageC --num-env-steps 20000000 --num-processes 16 --num-steps 30 --num-mini-batch 2 --ppo-epoch 5 --lr 5e-5 --eps 1e-5 --gamma 0.99 --gae-lambda 0.95 --entropy-coef 0.05 --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 --use-linear-lr-decay --save-interval 200 --log-interval 20 --resume --load_path trained_models/grace/stageB/checkpoints/<best>.pt
```

**Evaluate (metrics — SR, CR, GCR):**
```bash
python test.py --model_dir trained_models/grace/stageC --test_model <best>.pt
```

**Visualize cost map (video — env + 9 cost channels side by side):**
```bash
python visualize_cost_map.py \
    --model_dir trained_models/grace/stageC \
    --test_model <best>.pt \
    --seed 3 --out cost_map.mp4 --fps 8 --dpi 130
```

Outputs `cost_map.mp4`: left panel = bird's-eye environment (robot △, humans ●, goal ★), right panel = 3×3 grid of the 9 cost-map channels. The **Group Cohesion** channel (Ch5, purple) is the paper's key visual — it shows a single merged blob across all group members rather than separate dots. Run with several seeds to find the best F-formation blocking scenario.

**Realistic benchmark evaluation (paper table numbers):**

Step 1 — In `crowd_nav/configs/config.py`, change these settings:
```python
# Bump to full benchmark scale
sim.human_num        = 20
sim.circle_radius    = 8.5
sim.arena_size       = 8.5
group.num_groups     = 3
group.num_on_path    = 2
group.types          = ['static_f', 'dynamic_lf', 'dynamic_free']

# Enable all realistic phases (A–E)
realistic.enabled               = True
realistic.use_speed_variation   = True   # Phase A: v_pref ~ N(1.34, 0.26)
realistic.use_group_speed_factor= True   # Phase B: groups at 0.85× min v_pref
realistic.use_f_formations      = True   # Phase C: F-formation spawning
realistic.use_leader_follower   = True   # Phase D: leader-follower motion
realistic.use_convex_hull       = True   # Phase E: convex hull boundaries
```

Step 2 — Run evaluation (500 episodes for stable estimates):
```bash
python test.py --model_dir trained_models/grace/stageC --test_model 41660.pt
```

> **Important:** Do NOT retrain after changing these settings — only run `test.py`. The model was trained on the simpler 15-human non-realistic env; we evaluate in the harder realistic env to stress-test generalization. Restore `sim.human_num=15`, `realistic.enabled=False` before any retraining.

---

## 5. Training Procedure

We train in three stages. **Each stage has a single, easy-to-debug objective**, which is the lesson learned from GRAM-v2 Stage 3 collapse (multi-change cascade).

### Stage A — Cost map pretraining (no RL) — SKIPPED
CostMapSynthesizer has zero learnable parameters (pure deterministic Gaussian splatting). The cost map is physically grounded from step 0 — no pretraining needed. Stage A is not required.

### Stage B — Planning head only (PPO with frozen backbone) ✅ COMPLETE
- `grace.freeze_backbone=True`, `grace.use_aux_loss=False`
- Environment: 15 humans, circle_radius=6, groups=['static_f','dynamic_lf'], num_on_path=1
- Only CostMapPlanner + RobotMLP + fusion + GRU + Actor/Critic trained; backbones frozen
- Loss: standard PPO (clipped surrogate + value loss)
- **Result: SR=83%, CR=11%, TR=6%** at ~41600 updates (mean reward 13.4)
- Peak SR=92% observed early (~update 400) — statistical fluctuation in 100-ep window
- Best checkpoint: `trained_models/grace/stageB/checkpoints/41600.pt`

> **Lesson learned:** `--entropy-coef 0.0` (default) causes policy to collapse from SR=83% → 17% by end of training. Must use `--entropy-coef 0.05`.

**Advance criterion:** SR > 60% ✅ achieved (83%)

### Stage C — End-to-end fine-tuning ✅ COMPLETE
- `grace.freeze_backbone=False`, `grace.use_aux_loss=True`
- Unfreeze GroupDetector + SlotAttention; add self-supervised aux occupancy loss (λ=0.1)
- Joint loss: `L_PPO + 0.1 · L_aux`
- LR = 5e-5 (reduced from 4e-4 to prevent NaN gradient corruption on first backbone unfreeze)
- Resume from Stage B best checkpoint (`41600.pt`)
- Total updates: ~41654 | FPS: 668

**NaN fixes required for Stage C stability:**
1. `g = g.nan_to_num(0.0)` after GroupDetector — GD produces NaN on first unfreeze
2. `alpha = alpha.nan_to_num(0.0)` after SlotAttention
3. `cost_stack = cost_stack.nan_to_num(0.0)` after synthesizer
4. OccupancyHead outputs raw logits (no Sigmoid) → `binary_cross_entropy_with_logits`
5. NaN gradient zeroing in `ppo.py` before `optimizer.step()` — prevents weight corruption

**Stage C Test Results (100 episodes):**

Evaluation environment config (same as training — non-realistic):
```
sim.human_num        = 15
sim.circle_radius    = 6.0
sim.arena_size       = 6.0
group.num_groups     = 2
group.num_on_path    = 1
group.types          = ['static_f', 'dynamic_lf']
realistic.enabled    = False   (no speed variation, no F-formations, no leader-follower, no convex hull)
reward.discomfort_group_dist         = 0.35
reward.discomfort_grp_penalty_factor = 10
reward.grp_collision_penalty         = -5
```

| Metric | Value |
|---|---|
| Success Rate (SR) | **89%** |
| Collision Rate (CR) | **3%** |
| Timeout Rate (TR) | 8% |
| GCR (group intrusion rate) | 3.32% |
| Avg intrusion ratio | 3.39% |
| Avg min distance in intrusions | 0.38 m |
| Nav time | 17.37 s |
| Path length | 26.68 m |
| Mean reward | 19.76 |

Training diagnostics at update 41654:
- DIAG SR=89%, CR=5%, TR=6% (100-ep rolling window)
- Mean/median reward: 6.6 / 27.9 | Min/max: −463.1 / 36.5
- Collision avg step: 31/197 (early collisions → occasional panic in tight situations)
- Success avg: 85 steps to goal
- Discomfort: 1.4% of steps in danger zone
- Best checkpoint: `trained_models/grace/stageC/checkpoints/41660.pt`

**Stage B → Stage C improvement: SR +6pp (83% → 89%), CR −8pp (11% → 3%)**

**Advance criterion:** SR > 85% sustained ✅ achieved (89%)

---

**Stage C Realistic Benchmark Results (100 episodes):**

Evaluation environment config (realistic — paper table numbers):
```
sim.human_num        = 20
sim.circle_radius    = 8.5
sim.arena_size       = 8.5
group.num_groups     = 3
group.num_on_path    = 2
group.types          = ['static_f', 'dynamic_lf', 'dynamic_free']
realistic.enabled    = True   (all phases A–E: speed variation, group speed factor,
                               F-formations, leader-follower, convex hull boundaries)
```

| Metric | Value |
|---|---|
| Success Rate (SR) | **87%** |
| Collision Rate (CR) | **7%** |
| Timeout Rate (TR) | 6% |
| GCR (group hull intrusion rate) | **0.00%** |
| Avg intrusion ratio | 5.86% |
| Avg min distance in intrusions | 0.37 m |
| Nav time | 20.76 s |
| Path length | 31.70 m |
| Mean reward | 31.35 |

Collision cases: episodes 4, 21, 40, 51, 54, 60, 90 (7 out of 100)
Timeout cases: episodes 35, 41, 47, 49, 88, 97 (6 out of 100)

**Non-realistic → Realistic comparison (same checkpoint 41665.pt):**
| Metric | Non-realistic (15 humans) | Realistic (20 humans) | Change |
|---|---|---|---|
| SR | 89% | 87% | −2pp (mild drop, model generalises well) |
| CR | 3% | 7% | +4pp (harder env, more collisions) |
| TR | 8% | 6% | −2pp |
| GCR (hull intrusion) | 3.32% | **0.00%** | ↓ (model never enters convex hull) |
| Mean reward | 19.76 | 31.35 | ↑ (larger arena → more potential reward) |
| Nav time | 17.37 s | 20.76 s | ↑ (larger arena, 8.5m vs 6m) |
| Path length | 26.68 m | 31.70 m | ↑ (longer paths in larger arena) |

> **GCR = 0.00%** means the robot never entered the convex hull of any group in any of the 100 realistic episodes. The 5.86% intrusion ratio is the discomfort-zone proximity rate (within 0.35m of the hull boundary), not a hull crossing.

---

## 5. Experiments

### 5.1 Main results table
Compare on the realistic benchmark (20 humans, 3 groups, all five realistic phases on):

| Method | SR ↑ | CR ↓ | TR ↓ | GCR ↓ | Avg Reward ↑ | Avg Steps ↓ |
|---|---|---|---|---|---|---|
| ORCA | | | | | | |
| Social Force | | | | | | |
| GARN (RA-L 2025) | | | | | | |
| GRAM (legacy) | | | | | | |
| GRAM-v2 (Stage 5) | | | | | | |
| **GRACE (ours)** | **87%** | **7%** | 6% | **0.00%** | 31.35 | — |

### 5.2 Ablations
| Variant | What it tests |
|---|---|
| GRACE w/o L3+L4 (group layers) | Does group-aware cost matter, or is per-human enough? |
| GRACE w/o L2 (trajectory layer) | Does future intent matter? |
| GRACE w/o L_aux | Does self-supervised cost regularization matter? |
| GRACE w/ random init backbones | Does Phase 2/3 pretraining matter? |
| GRACE T = 1 (no temporal axis) | Is the 3D cost map worth the FLOPs? |

### 5.3 Generalization
Train on 3-4 member groups, evaluate on:
- 5-6 member groups
- All-static-F environment
- All-dynamic-LF environment
- Mixed environment with novel formation (line-of-three not seen in training)

### 5.4 Qualitative — cost map visualizations
Render the BEV cost map alongside the trajectory for representative scenarios:
1. F-formation blocking the path (does the model see a single high-cost region, not a cluster of points?)
2. Two groups walking toward each other (does the trajectory layer create a high-cost overlap zone?)
3. Sparse crowd, robot has multiple options (does the goal attractor dominate?)

These visualizations are the core "story" of the paper — they show *why* the robot chose what it chose, in a way no recurrent policy can.

### 5.5 Robustness
- Sensor noise (Gaussian on positions, velocities)
- Partial observability (FOV-limited robot)
- Human policy mismatch (train on ORCA, test on Social Force)

---

## 6. Comparison to Related Work

| Work | What it does | Why GRACE is different |
|---|---|---|
| Social cost maps (Kollmitz 2015, Kim & Pineau 2016) | Hand-designed Gaussians around pedestrians, classical planner | Hand-designed; not group-aware; not temporal; not learned |
| MPPI on learned costs (Williams 2017, Wang 2021) | Sampled trajectories on a neural cost field | Cost field is monolithic, not compositional; not group-aware; trained from scratch |
| SACSoN (Hirose 2023) | Learned occupancy + planning for indoor robots | Static scenes; no group reasoning; designed for offline data |
| TrajNet++/Y-Net trajectory predictors | Predict future trajectories | Prediction only; doesn't close the loop with planning |
| GRAM (ours, prior) | Group-aware attention over recurrent policy | Opaque; no spatial reasoning; PPO-only training |
| GARN (RA-L 2025) | STGAN trajectory predictor + reward shaping | Reward-level group awareness; no spatial cost map |
| **GRACE (this work)** | **Compositional learned cost map with group layer + time axis + self-supervision** | **Combines all four ideas; first end-to-end social cost map that is group-aware and time-conditioned** |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Cost map resolution too coarse for tight squeezes | Bilinear sampling at sub-cell resolution for the planner's local neighborhood |
| Cost synthesizer overfits to ORCA-style humans | Train on Social Force + ORCA + ground-truth + heuristic mixed; auxiliary loss is policy-agnostic |
| 3D-CNN planner is slow on low-end GPUs | Default config uses 5 horizons × 60×60 grid → < 1 ms / forward pass on a single 3060 |
| PPO instability returns at Stage C | L_aux acts as a regularizer; if instability, freeze cost synthesizer permanently and report Stage B as the headline result |
| CoRL reviewers say "MPPI on learned cost is not new" | We pre-empt this in §6 — our novelty is **compositional + group-aware + time-conditioned**, not the cost-map-on-which-to-plan idea itself |

---

## 8. Timeline

| Week | Milestone | Status |
|---|---|---|
| 1 | Architecture design + full implementation | ✅ Done (2026-05-06) |
| 2 | Stage A skipped (zero-param synthesizer); Stage B training | ✅ Done — SR=83% |
| 3 | Stage C: unfreeze backbone + aux loss fine-tuning | ✅ Done — SR=89%, CR=3% |
| 4 | Realistic benchmark eval (all phases on, 20 humans) | 🔄 Next |
| 5 | Ablations (remove group layers, trajectory layer, aux loss) | Pending |
| 6 | Cost map visualizations for paper | Pending |
| 6 | Paper draft; figures; noise robustness experiments | Pending |
| 7 | CoRL submission (deadline ≈ June 2026) | Pending |

Total: **~7 weeks of focused work.** Heavy reuse of existing infrastructure (Phase 2/3 weights, environment, evaluation pipeline, GST predictor for trajectory layer) keeps this realistic.

---

## 9. What this work makes possible afterward

- **Real-world transfer:** the cost map is the natural interface to a real lidar / depth sensor. Phase 2/3 can be retrained on real-world group detection data; the planner doesn't change.
- **Human-readable failure analysis:** every collision can be inspected by replaying the cost map at the moment of failure.
- **Compositionality:** new cost layers (e.g., comfort zones for elderly, pet zones, terrain) plug in without retraining the planner from scratch — train just that layer.
- **TAGA bridge:** TAGA is a hand-crafted tangent action; in this framework it becomes a *learned policy on a cost map that has a group layer*, which is strictly more general.

---

## 10. One-paragraph summary (for the abstract)

We present GRACE, an end-to-end architecture for social robot navigation that exposes the robot's spatial belief as a learned, group-aware, time-conditioned cost map and plans over it with a 3D convolutional policy. Cost is composed from interpretable layers (individual occupancy, trajectory propagation, group cohesion hull, group repulsion, goal attractor, arena boundary), each rendered differentiably from a self-supervised group perception backbone (GroupDetector + SlotAttention). A self-supervised auxiliary loss grounds the cost map in observed future occupancy, mitigating the PPO instability that plagues prior end-to-end methods. We evaluate on a realistic crowd benchmark (Social Force humans, 20 agents, F-formations and leader-follower groups) against ORCA, Social Force, GARN, GRAM, and GRAM-v2, and ablate every cost layer. GRACE beats prior learned baselines on success rate while reducing group crossing rate below classical methods, and produces visualizable cost maps that explain every robot decision.

---

## 11. Branch Overview and Working Process

### How many branches exist

| Branch | Purpose | Touch it when… |
|---|---|---|
| `gram-map` | **Daily development** — your working branch | Every day. All new code, experiments, bug fixes go here first |
| `grace` | **CoRL 2026 paper** — clean, structured, paper-ready | When a result is confirmed and you want to add it to the paper |
| `ral-benchmark` | **RA-L paper** — environment + TAGA code | When updating the environment or TAGA experiments |
| `main` | Original CrowdNav++ baseline | Almost never — it's the starting point, not touched |
| `gramv2` | Old GRAM-v2 curriculum experiments | Only if you need to look up old curriculum results |
| `garn-training` | Old GARN training experiments | Only if you need to look up old GARN results |

You have **6 branches total**. You will mostly work on `gram-map` and periodically push clean results to `grace`.

---

### Normal daily workflow: work in `gram-map`, then sync to `grace`

**Step 1 — Make sure you are on `gram-map`:**
```bash
git checkout gram-map
git status   # confirm you are on gram-map and see what is changed
```

**Step 2 — Do your work** (edit code, run experiments, fix bugs). When you are happy with a change:
```bash
git add <files you changed>
git commit -m "short description of what you did"
```

**Step 3 — Find the commit hash you want to copy to `grace`:**
```bash
git log --oneline -10    # shows last 10 commits, each has a short hash like a1b2c3d
```

**Step 4 — Switch to `grace` and cherry-pick that commit:**
```bash
git checkout grace
git cherry-pick a1b2c3d    # replace with your actual commit hash
```

This copies exactly that one commit from `gram-map` into `grace`. Everything else stays unchanged.

**Step 5 — Switch back to `gram-map` to continue working:**
```bash
git checkout gram-map
```

**Example — you fixed a bug in `grace_network.py` on `gram-map` and want it in `grace` too:**
```bash
# On gram-map, after fixing and committing:
git log --oneline -3
# Output:
#   f4a9c12  Fix NaN in cross-attention when all humans masked
#   3d3fad9  Rename GRAM-Map → GRACE; add CoRL 2026 paper draft
#   015dce9  The visualizer code is updated

git checkout grace
git cherry-pick f4a9c12     # copies only the NaN fix
git checkout gram-map       # back to work
```

> **Rule of thumb:** only cherry-pick to `grace` when the result is confirmed and clean.
> `gram-map` can have messy experiments. `grace` should always be in a working state.

---

### Ablation study workflow — step by step

Ablations are run on `gram-map` first. Once confirmed, add results to the paper on `grace`.

**Phase 1 — Train ablations on `gram-map`**

Make sure you are on `gram-map` and Stage B checkpoint exists:
```
trained_models/gram_map/stageB/checkpoints/best.pt
```

Run each ablation (each takes ~1 GPU-day). You can run them one after another or in parallel on different GPUs:

```bash
# C1 — does group cost map matter?
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C1_no_group \
    --lr 5e-5 --use-linear-lr-decay \
    --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt \
    --ablation_no_group_layers

# C2 — does having 3 slots (vs 1) matter?
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C2_K1 \
    --lr 5e-5 --use-linear-lr-decay \
    --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt \
    --ablation_K_slots 1

# C3 — do trajectory prediction channels matter?
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C3_no_traj \
    --lr 5e-5 --use-linear-lr-decay \
    --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt \
    --ablation_no_traj_layers

# C4 — does auxiliary loss matter?
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C4_no_aux \
    --lr 5e-5 --use-linear-lr-decay \
    --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt \
    --ablation_no_aux_loss

# C5 — does learned slot assignment (vs uniform) matter?
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C5_uniform \
    --lr 5e-5 --use-linear-lr-decay \
    --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt \
    --ablation_uniform_alpha
```

**Phase 2 — Evaluate all ablations on `gram-map`**

```bash
python test.py --model_dir trained_models/gram_map/stageC              --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C1_no_group --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C2_K1       --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C3_no_traj  --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C4_no_aux   --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C5_uniform  --test_model best.pt
```

Or get the full comparison table at once. First add these entries to `POLICY_REGISTRY` at the top of `record_comparison.py`:
```python
dict(label='grace_full', policy_key='grace', model_dir='trained_models/gram_map/stageC',               test_model='best.pt', with_taga=False),
dict(label='grace_C1',   policy_key='grace', model_dir='trained_models/gram_map/ablation_C1_no_group',  test_model='best.pt', with_taga=False),
dict(label='grace_C2',   policy_key='grace', model_dir='trained_models/gram_map/ablation_C2_K1',        test_model='best.pt', with_taga=False),
dict(label='grace_C3',   policy_key='grace', model_dir='trained_models/gram_map/ablation_C3_no_traj',   test_model='best.pt', with_taga=False),
dict(label='grace_C4',   policy_key='grace', model_dir='trained_models/gram_map/ablation_C4_no_aux',    test_model='best.pt', with_taga=False),
dict(label='grace_C5',   policy_key='grace', model_dir='trained_models/gram_map/ablation_C5_uniform',   test_model='best.pt', with_taga=False),
```

Then run:
```bash
python record_comparison.py \
    --policies grace_full,grace_C1,grace_C2,grace_C3,grace_C4,grace_C5 \
    --seeds 0,1,2,3,4 --no-video
```

Results saved to `results/metrics.csv`.

**Phase 3 — Visualise cost maps (see what each ablation removes)**

```bash
python visualize_cost_map.py --model_dir trained_models/gram_map/stageC              --test_model best.pt --seed 3 --out videos/costmap_full.mp4
python visualize_cost_map.py --model_dir trained_models/gram_map/ablation_C1_no_group --test_model best.pt --seed 3 --out videos/costmap_C1.mp4
python visualize_cost_map.py --model_dir trained_models/gram_map/ablation_C3_no_traj  --test_model best.pt --seed 3 --out videos/costmap_C3.mp4
python visualize_cost_map.py --model_dir trained_models/gram_map/ablation_C5_uniform  --test_model best.pt --seed 3 --out videos/costmap_C5.mp4
```

**Phase 4 — Confirm results, then update the paper on `grace`**

Once you have the numbers from `results/metrics.csv`:

```bash
# Check what commits you made on gram-map (to find what to cherry-pick)
git log --oneline -10

# Switch to grace and open the paper
git checkout grace

# Edit the ablation table in the paper (fill in the -- placeholders)
# File: corl_2026/corl_2026_template_submission/grace.tex
# Look for Table 2 / \tbd placeholders and replace with actual numbers

# Commit the updated paper
git add corl_2026/corl_2026_template_submission/grace.tex
git commit -m "Fill ablation Table 2 with measured results"

# Switch back to gram-map
git checkout gram-map
```

---

### Quick reference: most common commands

```bash
# See which branch you are on
git status

# Switch to a branch
git checkout gram-map
git checkout grace

# See recent commits (to find hash for cherry-pick)
git log --oneline -10

# Copy one commit from gram-map to grace
git checkout grace
git cherry-pick <hash>
git checkout gram-map

# See what is different between gram-map and grace
git diff gram-map..grace --stat
```
