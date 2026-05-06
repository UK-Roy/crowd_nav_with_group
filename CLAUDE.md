# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep reinforcement learning project for robot crowd navigation with group-aware dynamics. The main contribution is **TAGA** (Tangent Action for Group Avoidance), a modular reactive mechanism that computes tangent trajectories around detected human groups and can be bolted onto any existing navigation policy. TAGA was submitted to IROS 2025 and is being strengthened for IEEE RA-L journal submission.

**GARN** (Lu et al., IEEE RA-L 2025 — `references/GARN.pdf`) is an external baseline implemented in this codebase for head-to-head evaluation against TAGA. GARN training is the current active work on the `garn-training` branch.

**GRAM** (prior published work) remains in the repo as a legacy baseline — its network and training infrastructure are intact but not the current focus.

**GRAM-v2** is the active new model on branch `gramv2`. It replaces GRAM's ground-truth group labels with a learned perception pipeline (GroupDetector + SlotAttention) and trains a full navigation policy via PPO using curriculum learning.

## Current Goals

1. **[ACTIVE] Train GRAM-v2** via curriculum on GPU (rmedu). Branch: `gramv2`. Stage 1 done (SR ~50%), Stage 2 in progress.
2. **Realistic benchmark environment** — completed. All policies (TAGA, GARN, GRAM, ORCA, etc.) are evaluated in this shared env.
3. **Strengthen TAGA for RA-L** — deferred.
4. **Train GARN** — deferred until GRAM-v2 curriculum completes.

## Setup

```bash
conda create -n crowdnav python=3.8 && conda activate crowdnav
pip install -r requirements.txt
# PyTorch with CUDA (match your CUDA version):
pip install torch==1.12.1+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116
# OpenAI Baselines:
git clone https://github.com/openai/baselines.git && cd baselines && pip install -e . && cd ..
# Python-RVO2:
cd Python-RVO2 && python setup.py install && cd ..
```

## Key Commands

**Train GARN** (default config: `robot.policy='garn'`, realistic env on):
```bash
python train.py
```

**Visualize environment** (interactive):
```bash
python visualize_env.py                        # mixed groups + individuals
python visualize_env.py --groups-only          # groups only
python visualize_env.py --individuals-only     # individuals only
```

**Record paper-quality video** (single policy):
```bash
python record_episode.py --seed 3 --dpi 150 --fps 10 --out episode.mp4
```

**Record and compare multiple policies** (produces videos + metrics CSV):
```bash
# All registered policies × N seeds
python record_comparison.py --seeds 0,1,2

# Subset by label
python record_comparison.py --policies orca,orca+taga,garn --seeds 0,1,2

# Metrics only (fast, no video rendering)
python record_comparison.py --seeds 0,1,2,3,4 --no-video
```

Outputs: `videos/<label>_seed<N>.mp4` and `results/metrics.csv`.
Summary table columns: **SR** (success rate), **CR** (collision rate), **TR** (timeout rate), Avg Steps, Avg GCR, Avg Reward.

**Evaluate trained model:**
```bash
python test.py                  # base policy only
python test.py --group_avoid    # base policy + TAGA on top
```
`model_dir` and `test_model` are set at the top of `test.py`. `--group_avoid` applies TAGA to any loaded policy (classical or neural) — GARN does not need it.

**Plot training curves:**
```bash
python plot.py
```

## Realistic Crowd Environment

The benchmark environment (`crowd_sim/envs/`) implements five realistic modeling phases, all enabled by default. Each is gated by a flag in `config.realistic` so disabling them reproduces the legacy env exactly.

| Phase | Flag | Description |
|---|---|---|
| A | `use_speed_variation` | Individual v_pref ~ N(1.34, 0.26) clipped [0.8, 1.8] (Weidmann 1992) |
| B | `use_group_speed_factor` | Dynamic groups walk at 0.85 × min(member v_pref) (Moussaid 2010) |
| C | `use_f_formations` | Static groups spawn in F-formations: vis-à-vis, L-shape, side-by-side, circle (Kendon 1990) |
| D | `use_leader_follower` | Dynamic LF groups: followers track leader with staggered lateral slots + inter-follower repulsion (Helbing & Molnar 1995) |
| E | `use_convex_hull` | Group boundaries are convex hulls (polygon/rectangle/circle); used for reward and GCR |

**Group types** — randomly assigned per group at each `reset()`:
- `static_f` — stationary F-formation, members are obstacles
- `dynamic_lf` — moving, followers track leader in staggered formation
- `dynamic_free` — moving, each member navigates independently (ORCA)

**Composition flags** in `config.sim`:
```python
sim.has_individuals = True   # False = groups only
sim.has_groups      = True   # False = individuals only
```

**Default composition**: 20 total humans, 3 groups × 3–4 members, 2 groups placed in robot's path, rest random.

---

## Human & Group Behaviour Reference

### Individual humans

| Property | Value | Source |
|---|---|---|
| Base policy | ORCA (reciprocal collision avoidance) | `config.humans.policy = 'orca'` |
| Preferred speed | 1.0 m/s (or N(1.34, 0.26) clipped [0.8,1.8] if phase A on) | `humans.v_pref` / `realistic_human_modeling.py` |
| Radius | 0.3 m | `humans.radius` |
| FOV | 360° (2π) | `humans.FOV = 2.` |
| Sensor range | full visibility | `humans.sensor = 'coordinates'` |

**Goal lifecycle (individuals only):**
1. **Mid-episode random goal change** — if `humans.random_goal_changing = True`, every 5 s there is a `goal_change_chance = 0.5` probability the human is redirected to a new random goal on the boundary circle (`crowd_sim.py:1045-1048`).
2. **End-goal change** — if `humans.end_goal_changing = True`, once a human reaches its goal it gets a new random goal drawn from the boundary circle with noise (`crowd_sim.py:1051-1055`, `crowd_sim_var_num.py:808-822`). `end_goal_change_chance = 1.0` means it always happens.
3. **Holonomic path (var_num env)** — on goal-reach the human is *fully regenerated* at a new spawn position (`generate_circle_crossing_human`), not just re-targeted.

> **Group members are exempt from all of the above.** When a group member reaches its goal it freezes in place: `gx/gy ← px/py`, `vx = vy = 0`, `v_pref = 0`. It remains visible as a static obstacle for the rest of the episode. (`crowd_sim_var_num.py:810-816`, `crowd_sim.py:1054-1061`)

---

### Groups

**Structure** (`crowd_sim/envs/utils/group.py`):
- Each `Group` holds: `id`, `members[]`, `leader` (= `members[0]`), `centroid [px, py]`, `radius`, `group_type`.
- Size drawn from `[group.min_size, group.max_size]` = `[3, 4]` at construction.
- `group_type` assigned randomly from `config.group.types` at every `reset()`.

**Spawn / placement:**
- `group.num_on_path = 2` groups are placed along the robot → goal axis to guarantee the robot encounters them.
- Remaining groups are placed at random positions.
- Within each group, members are positioned via `Group.position_members()` using a randomly chosen formation (circle, V-shape, grid, line — line excluded for group ID 0). Up to 300 jitter attempts resolve spawn collisions.
- Initial goal of every group member is set to the *opposite* of its spawn position (`gx = -px, gy = -py`).

**Group types and motion:**

| Type | Motion | Member control | Speed |
|---|---|---|---|
| `static_f` | Stationary — F-formation (vis-à-vis, L-shape, side-by-side, circle) | `isObstacle = True`, `v_pref = 0` | 0 |
| `dynamic_lf` | Moving — leader navigates with ORCA, followers track leader in staggered lateral slots with inter-follower repulsion (Helbing & Molnár 1995) | Leader: ORCA; followers: custom | 0.85 × min(member v_pref) if phase B on |
| `dynamic_free` | Moving — each member navigates independently | ORCA per-member | individual v_pref |

**Group hull (phase E, `realistic.use_convex_hull = True`):**
- Rebuilt every step from ground-truth positions via `_update_group_hulls()` → `build_group_hulls()` in `realistic_human_modeling.py`.
- Hull geometry is a convex polygon (or fallback circle/rectangle for degenerate cases with `hull_degenerate_buffer = 0.30` m).
- Used by the reward function (intrusion detection) and GCR metric.

**Group detection available to TAGA:**
- Currently **ground-truth only** (`config.group.ground_truth = True`). `generate_ob()` reads `human.group_id` directly from the simulator (`crowd_sim_var_num.py:328-334`).
- DBSCAN block exists but is commented out. Enabling it requires `group.ground_truth = False` and uncommenting `crowd_sim_var_num.py:338-399`.

**What happens when a group member reaches its goal:**
- `v_pref` is set to 0, velocity zeroed, goal snapped to current position → member stays as a static obstacle.
- Other group members continue their normal motion (ORCA / leader-follower) unaffected.
- The group's hull is still rebuilt from the remaining moving members each step.

---

## Configuration

Two files work together:
- `crowd_nav/configs/config.py` — simulation, reward, group, realistic, GARN, TAGA parameters
- `arguments.py` — network architecture, PPO hyperparameters, output directory

Key settings for GARN training (already set correctly):
```python
robot.policy = 'garn'
reward.use_garn_reward = True
sim.predict_method = 'inferred'   # uses GST predictor
sim.has_individuals = True
sim.has_groups = True
realistic.enabled = True          # all phases on
group.types = ['static_f', 'dynamic_lf', 'dynamic_free']
```

When testing, `test.py` loads config from the **model directory** (`{model_dir}/configs/config.py`), not the root.

## Architecture

```
crowd_nav/configs/config.py          ← simulation & reward config
arguments.py                          ← network hyperparameters & training config
     ↓
train.py / test.py
     ↓
rl/networks/envs.py                  ← creates vectorized gym environments
     ↓
crowd_sim/envs/crowd_sim_pred_real_gst.py   ← training env (GST predictor)
  └─ crowd_sim/envs/crowd_sim_pred.py
       └─ crowd_sim/envs/crowd_sim_var_num.py
            └─ crowd_sim/envs/crowd_sim.py  ← base env
     ↓
rl/networks/model.py                 ← policy network wrapper
crowd_nav/policy/garn.py             ← GARN (STGAN) policy
rl/ppo/ppo.py                        ← PPO optimizer
```

### Environment step() pipeline (all subclasses)

Every `step()` call runs in this order:
1. `get_human_actions()` — group-type-aware: static=zero, dynamic_lf=leader-follower, dynamic_free=ORCA
2. `_update_group_hulls()` — builds ConvexHullGeometry per group from current positions
3. `calc_reward()` — uses hulls for intrusion detection; adds GARN R_grp if enabled
4. `robot.step()` + `humans[i].step()` — advance positions
5. `_enforce_separation()` — hard push-apart for any overlapping pair

### Policy Registry (`crowd_nav/policy/policy_factory.py`)

| Key | Class | Network file | Notes |
|---|---|---|---|
| `orca` | `ORCA` | — | Classical baseline |
| `social_force` | `SOCIAL_FORCE` | — | Classical baseline |
| `srnn` | `SRNN` | `srnn_model.py` | DS-RNN baseline (neural) — checkpoint: `trained_models/srnn` (not yet trained) |
| `selfAttn_merge_srnn` | `selfAttn_merge_SRNN` | `selfAttn_srnn_temp_node.py` | Intention-aware RL (neural) |
| `selfAttn_merge_srnn_grpAttn` | `selfAttn_merge_SRNN_GrpAttn` | `selfAttn_srnn_temp_node_groupAttn.py` | GRAM — our model |
| `garn` | `GARN` | `stgan_model.py` | GARN: STGAN + group reward |
| `zone_based` | `ZoneBasedGroupAvoidance` | — | Heuristic group-aware |
| `f_formation` | `FFormationAvoidance` | — | Heuristic group-aware |

### Key files

| File | Purpose |
|---|---|
| `crowd_sim/envs/utils/realistic_human_modeling.py` | All realistic modeling utilities (phases A–E) |
| `crowd_sim/envs/garn_reward.py` | GARN R_grp reward (Eq. 4–8, Lu et al.) |
| `visualize_env.py` | Interactive environment viewer |
| `record_episode.py` | Paper-quality video recorder (single policy) |
| `record_comparison.py` | Multi-policy comparison recorder + metrics CSV |

## Evaluation Metrics

| Metric | Symbol | Description |
|---|---|---|
| Success Rate | SR | fraction of episodes where robot reached goal |
| Collision Rate | CR | fraction of episodes with any collision |
| Timeout Rate | TR | fraction of episodes that hit max_steps without success/collision |
| Group Crossing Rate | GCR | avg fraction of steps the robot was inside a group hull (lower = less disruptive) |
| Avg Steps | — | navigation efficiency |
| Avg Reward | — | cumulative reward; shaped by GARN R_grp if enabled |

Note: SR + CR + TR ≈ 1.0 is a sanity check across seeds.

## `record_comparison.py` Policy Registry

Edit `POLICY_REGISTRY` at the top of the file to add new policies — no other code changes needed. Each entry:

```python
dict(label='my_policy',
     policy_key='key_in_policy_factory',   # or policy_factory key
     model_dir='trained_models/my_model',  # None for classical
     with_taga=True)                        # False for GARN
```

`expand_registry()` automatically generates a `<label>+taga` variant for every entry where `with_taga=True`.

## Important Behavioral Notes

- **Convex hulls** are rebuilt every step from ground-truth `human.group_id` assignments — works in all env subclasses regardless of which `generate_ob` override is active.
- **Separation enforcement** (`_enforce_separation`) runs after every step in both `CrowdSimVarNum` and `CrowdSimPred` to prevent agents merging visually.
- **arguments.py** uses `parse_known_args` so custom CLI flags (e.g., `--steps`, `--seed`) don't clash with the training argument parser.
- **Parallel envs**: default 16 (`--num-processes`). Reduce to 8 if GPU OOM.
- **GST predictor**: pre-trained models in `gst_updated/results/`. Path set via `pred.model_dir` in config.
- **TAGA**: pass `--group_avoid` to `test.py`. Uses DBSCAN for runtime group detection.
- **Group type assignment**: randomly drawn from `config.group.types` at every `reset()` — each episode can have a different mix of static/dynamic_lf/dynamic_free groups.

## Pre-trained Models

| Directory | Checkpoint | Description |
|---|---|---|
| `trained_models/GST_predictor_rand` | `41665.pt` | Intention-aware RL, randomized humans |
| `trained_models/GST_predictor_non_rand` | `41200.pt` | Intention-aware RL, no randomization |
| `trained_models/my_model` | — | GRAM — legacy baseline |
| `trained_models/garn` | — | GARN in realistic env — deferred |
| `trained_models/gram_v2/phase2_v2` | `best.pt` | GroupDetector backbone (Phase 2) |
| `trained_models/gram_v2/phase3` | `best.pt` | SlotAttention backbone (Phase 3) |
| `trained_models/gram_v2/stage1_unfreeze` | `41660.pt` | GRAM-v2 Stage 1 nav policy (SR ~50%) |
| `trained_models/gram_v2/stage2` | (training) | GRAM-v2 Stage 2 — in progress |

## GRAM-v2 Curriculum Training

### Architecture (`rl/networks/gram_v2_network.py`)

Per-timestep pipeline:
1. 2-frame rolling buffer → 21-d per-human features
2. **GroupDetector** → `W_final` (N×N groupness), `g` (N×64 embeddings)
3. **Robot-centric spatial encoder** → range, sin/cos bearing, TCPA, DCPA → added to `g` (always active)
4. **SlotAttention** → `slots` (K×64) + `alpha` (K×N assignments) — Stage 3+ only
5. **Geometric group summaries** from `alpha`+`W_final` → centroid, group_vel, spread, approach_rate, cohesion → added to slots — Stage 3+ only
6. **Null token** (always-valid) appended to prevent all-keys-masked NaN in cross-attention
7. **CrossAttention**: robot query (9-d → 64-d) attends to `[g, (slots), null]`
8. **GRUCell** (256→256) temporal memory → Actor/Critic heads

### Key config flags (`config.gram_v2`)

| Flag | Stage 1–2 | Stage 3+ | Effect |
|---|---|---|---|
| `freeze_backbone` | `False` | `True` | Train backbone with nav policy vs lock it |
| `use_slots` | `False` | `True` | Skip SlotAttention vs include group prototypes |

### Curriculum stages

| Stage | humans | radius | groups | on-path | freeze | slots | realistic | Advance criterion | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 4m | 1×static_f | 0 | No | No | No | SR ≥ 40% | ✅ SR ~50% |
| 2 | 10 | 5m | 2×static_f | 0 | No | No | No | mean > 0 sustained | ✅ mean ~3, median ~9 |
| 3 | 15 | 6m | 2×mixed | 1 | **Yes** | **Yes** | No | SR > 40% | Pending |
| 4 | 20 | 8.5m | 3×mixed | 2 | Yes | Yes | No | SR > 35% | Pending |
| 5 | 20 | 8.5m | 3×mixed | 2 | Yes | Yes | **Yes** | SR > 30% | Pending |

**Mean reward ceiling per stage** — do not wait for mean 15+; the environment gets harder each stage:
- Stage 1: mean 3–7 is ceiling (5 humans, easy)
- Stage 2: mean 2–5 is ceiling (10 humans, some collisions expected)
- Stage 3+: mean can drop at start (new slots + frozen backbone need re-learning)

### Config changes required before each stage

**Before Stage 3** (in `config.py`):
```python
# Architecture
gram_v2.freeze_backbone = True
gram_v2.use_slots       = True
# Environment
sim.human_num = 15;  sim.circle_radius = 6;  sim.arena_size = 6
group.num_on_path = 1
group.types = ['static_f', 'dynamic_lf']
# Reward — enable group penalties now
reward.discomfort_group_dist         = 0.35   # was 0.07
reward.discomfort_grp_penalty_factor = 10     # was 0
reward.grp_collision_penalty         = -5     # was 0
```

**Before Stage 4** (in `config.py`):
```python
sim.human_num = 20;  sim.circle_radius = 8.5;  sim.arena_size = 8.5
group.num_groups  = 3
group.num_on_path = 2
group.types = ['static_f', 'dynamic_lf', 'dynamic_free']
```

**Before Stage 5** (in `config.py`):
```python
realistic.enabled = True   # turns on all phases A–E
```

### Training commands

```bash
# Stage 3 (update config first, then run)
python train.py --env-name CrowdSimVarNum-v0 \
  --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
  --output_dir trained_models/gram_v2/stage3 \
  --lr 5e-4 --use-linear-lr-decay \
  --resume --load_path trained_models/gram_v2/stage2/checkpoints/<best>.pt

# Stage N→N+1: update config, then resume (strict=False is automatic)
python train.py --env-name CrowdSimVarNum-v0 \
  --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
  --output_dir trained_models/gram_v2/stage<N> \
  --lr 5e-4 --use-linear-lr-decay \
  --resume --load_path trained_models/gram_v2/stage<N-1>/checkpoints/<best>.pt
```

### Warning signs during training

| Pattern | Cause | Fix |
|---|---|---|
| Mean peaks then declines (300+ updates) | LR too high, overshooting | Roll back to best checkpoint, reduce LR to 2e-4 |
| Mean stuck at −18 to −20 from start | Backbone frozen in wrong stage or cold-start LR | Check `freeze_backbone`, use 7e-4 for cold start |
| Mean flat > 3000 updates after initial recovery | Config change not applied or LR too low | Verify config and check LR |
| Discomfort > 30% + SR not rising | Robot oscillating between humans | Check reward balance; may need to reduce `discomfort_dist` |
| SR drops to 0% after stage advance | `strict=False` not loading new layers | Check train.py resume code |
| NaN in loss | All-keys-masked cross-attention | Null token must be present — check `gram_v2_network.py` |

### Testing

```bash
python test.py --model_dir trained_models/gram_v2/stage1_unfreeze --test_model 41660.pt
```

`test.py` auto-reads `env_name.txt` from the model dir (correct env), uses `strict=False` (handles new layers), and injects `gram_v2_use_slots` from the saved config.

### Important fixes applied

- Backbone unfrozen for Stages 1–2 (frozen backbone → 0% SR; trainable → ~50% SR)
- Null token in cross-attention prevents NaN when all humans invisible
- `N_actual` derived dynamically from observations (was hardcoded MAX_HUMANS=20)
- `env_name.txt` read by test.py (default was wrong CrowdSimPredRealGST-v0)
- `strict=False` in both train.py and test.py for curriculum resume compatibility
- **Reward shaping fixed** (`crowd_sim_var_num.py:calc_reward`): potential reward and `self.potential` update now happen unconditionally every step; discomfort penalties (individual and group) are additive on top of potential instead of replacing it. Previously the `elif` chain silently dropped the progress signal whenever any human was nearby, and the stale `self.potential` caused artifact reward spikes when the crowd finally cleared.

### Reward shaping design (`crowd_sim_var_num.py:calc_reward`)

```
pot_reward = pot_factor * (potential_prev - potential_cur)   # always computed
self.potential updated every step

Terminal events (override pot_reward):
  timeout    → reward = 0
  collision  → reward = collision_penalty
  reach_goal → reward = success_reward

Non-terminal events (additive):
  grp_intrusion → reward = pot_reward + grp_collision_penalty
  grp_discomfort→ reward = pot_reward + (dmingrp - grp_dist) * grp_factor * dt
  ind_discomfort→ reward = pot_reward + (dmin - dist) * factor * dt
  nothing       → reward = pot_reward
```

GARN R_grp is always additive at the end regardless of branch.

### Reward magnitude reference

| Term | Stage 1–2 | Stage 3+ | Notes |
|---|---|---|---|
| `success_reward` | +10 | +10 | terminal |
| `collision_penalty` | −20 | −20 | terminal |
| `pot_factor` (holonomic) | 2 | 2 | +0.50/step forward at v_pref=1; total ≈ 10 over 5m |
| `discomfort_dist` | 0.25 m | 0.25 m | individual early-warning zone |
| `discomfort_penalty_factor` | 10 | 10 | max −0.625/step at contact |
| `discomfort_group_dist` | 0.07 m | **0.35 m** | increase before Stage 3 (7 cm → 35 cm) |
| `discomfort_grp_penalty_factor` | 0 | **10** | enable before Stage 3 |
| `grp_collision_penalty` | 0 | **−5** | enable before Stage 3 (softer than −20; episode continues) |

**Before starting Stage 3**, update `config.py`:
```python
reward.discomfort_group_dist         = 0.35
reward.discomfort_grp_penalty_factor = 10
reward.grp_collision_penalty         = -5
```

Net reward while moving forward near a human (v_pref=1, holonomic):
- `dmin = 0.10 m`: +0.50 (potential) − 0.375 (discomfort) = **+0.125** → still profitable
- `dmin = 0.00 m`: +0.50 (potential) − 0.625 (discomfort) = **−0.125** → backs off
