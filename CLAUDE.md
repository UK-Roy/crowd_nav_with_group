# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep reinforcement learning project for robot crowd navigation with group-aware dynamics. The main contribution is **TAGA** (Tangent Action for Group Avoidance), a modular reactive mechanism that computes tangent trajectories around detected human groups and can be bolted onto any existing navigation policy. TAGA was submitted to IROS 2025 and is being strengthened for IEEE RA-L journal submission.

**GARN** (Lu et al., IEEE RA-L 2025 — `references/GARN.pdf`) is an external baseline implemented in this codebase for head-to-head evaluation against TAGA. GARN training is the current active work on the `garn-training` branch.

**GRAM** (prior published work) remains in the repo as a legacy baseline — its network and training infrastructure are intact but not the current focus.

## Current Goals

1. **[ACTIVE] Train GARN** in the realistic crowd environment on GPU. Branch: `garn-training`.
2. **Realistic benchmark environment** — completed. All policies (TAGA, GARN, GRAM, ORCA, etc.) are evaluated in this shared env.
3. **Strengthen TAGA for RA-L** — deferred until after GARN training completes.

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
python test.py   # edit model_dir and test_model at top of file
```

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

| Key | Class | Notes |
|---|---|---|
| `garn` | `GARN` | STGAN: spatio-temporal GCN + LSTM + attention |
| `selfAttn_merge_srnn_grpAttn` | `selfAttn_merge_SRNN_GrpAttn` | GRAM (legacy) |
| `orca` | `ORCA` | Classical baseline |
| `social_force` | `SOCIAL_FORCE` | Classical baseline |
| `zone_based` | `ZoneBasedGroupAvoidance` | Heuristic group-aware |
| `f_formation` | `FFormationAvoidance` | Heuristic group-aware |

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
| `trained_models/GST_predictor_rand` | `41665.pt` | GRAM, randomized humans |
| `trained_models/GST_predictor_non_rand` | `41200.pt` | GRAM, no randomization |
| `trained_models/garn_realistic` | (training) | GARN in realistic env — in progress |
