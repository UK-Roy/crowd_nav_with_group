# Realistic Group-Aware Crowd Navigation Benchmark

**Branch: `ral-benchmark` — RA-L environment & TAGA paper code**

This branch contains the realistic group-aware crowd simulation benchmark and the
**TAGA** (Tangent Action for Group Avoidance) reactive navigation method.
The benchmark defines five modeling phases (A–E) covering individual speed variation,
group speed coupling, F-formations, leader-follower dynamics, and convex-hull group
boundaries — used for fair head-to-head evaluation of all crowd navigation policies.

> The GRACE policy (CoRL 2026) is evaluated *in this benchmark environment* and cites
> this work. Its code lives on the `grace` branch.

---

## Branch overview

| Branch | Purpose |
|---|---|
| `ral-benchmark` | **This branch** — benchmark env + TAGA RA-L paper |
| `grace` | GRACE CoRL 2026 paper code (evaluated in this env) |
| `gram-map` | Active development branch |
| `main` | Original CrowdNav++ baseline |

---

## Benchmark environment (crowd_sim/)

Five realism phases, all enabled by default via `config.realistic`:

| Phase | Flag | Description |
|---|---|---|
| A | `use_speed_variation` | Individual v_pref ~ N(1.34, 0.26) clipped [0.8, 1.8] |
| B | `use_group_speed_factor` | Dynamic groups walk at 0.85 × min(member v_pref) |
| C | `use_f_formations` | Static groups spawn in F-formations (vis-à-vis, L-shape, side-by-side, circle) |
| D | `use_leader_follower` | Followers track leader with staggered lateral slots |
| E | `use_convex_hull` | Group boundaries are convex hulls; used for GCR metric |

**Group types**: `static_f`, `dynamic_lf`, `dynamic_free` — randomly assigned per episode.

---

## TAGA

`crowd_nav/policy/taga_safety.py` — reactive group avoidance via tangent steering.
Pass `--group_avoid` to `test.py` to apply TAGA on top of any loaded base policy.

```bash
python test.py --group_avoid   # ORCA + TAGA
```

---

## Setup

```bash
conda create -n crowdnav python=3.8 && conda activate crowdnav
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/openai/baselines.git && cd baselines && pip install -e . && cd ..
cd Python-RVO2 && python setup.py install && cd ..
```

---

## Evaluation

```bash
# Run all registered policies × 3 seeds (produces metrics.csv + videos)
python record_comparison.py --seeds 0,1,2

# Metrics only (faster)
python record_comparison.py --seeds 0,1,2,3,4 --no-video

# Single policy
python test.py
```

Summary table columns: **SR** (success), **CR** (collision), **TR** (timeout), Avg Steps, Avg GCR, Avg Reward.

## Visualize environment

```bash
python visualize_env.py                  # mixed groups + individuals
python visualize_env.py --groups-only    # groups only
python visualize_env.py --individuals-only
```

## Policies available for comparison

| Key | Description |
|---|---|
| `orca` | ORCA classical baseline |
| `social_force` | Social Force Model |
| `srnn` | DS-RNN (neural) |
| `selfAttn_merge_srnn` | Intention-aware RL |
| `selfAttn_merge_srnn_grpAttn` | GRAM (attention-based group-aware) |
| `garn` | GARN (group-aware neural) |
| `gram_v2` | GRAM-v2 (learned group perception) |
| `zone_based` | Zone-based group avoidance |
| `f_formation` | F-formation avoidance |
