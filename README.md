# GRACE: Group-Responsive Avoidance via Cost-Map Encoding

**Branch: `grace` — CoRL 2026 submission code**

GRACE is an end-to-end deep RL navigation policy for robots moving through crowds with human groups.
It renders an explicit 9-channel bird's-eye-view cost map from a learned perception backbone
(GroupDetector + SlotAttention), then plans through the cost map with a CNN planner and GRU temporal memory.

Paper draft: `corl_2026/corl_2026_template_submission/grace.tex`

---

## Branch overview

| Branch | Purpose |
|---|---|
| `grace` | **This branch** — GRACE CoRL 2026 paper code |
| `gram-map` | Active development branch for GRACE |
| `ral-benchmark` | RA-L benchmark paper: realistic group env + TAGA |
| `main` | Shared environment baseline |

---

## File overview

| File / Directory | What it does |
|---|---|
| `crowd_nav/grace_perception/` | **GRACE perception backbone** — GroupDetector + SlotAttention model definitions |
| `crowd_nav/grace_perception/models.py` | `GroupDetector`: pedestrian encoder + GNN → 64-d embeddings + N×N groupness matrix |
| `crowd_nav/grace_perception/slot_attention.py` | `SlotAttention`: K=3 group-prototype vectors from N embeddings via iterative cross-attention |
| `crowd_nav/grace_perception/gnn.py` | `GroupGNN`: graph neural network layer inside GroupDetector |
| `rl/networks/grace_network.py` | `GRACENetwork`: full navigation policy (Phase 4 end-to-end) |
| `rl/networks/grace_synthesizer.py` | `CostMapSynthesizer` + `CostMapPlanner`: 9-channel BEV cost map from perception output |
| `crowd_nav/policy/grace.py` | Policy class registered under key `'grace'` in the policy factory |
| `crowd_nav/configs/config.py` | All GRACE hyperparameters under `config.grace` (grid, horizons, checkpoints, aux loss) |
| `rl/networks/gram_v2_network.py` | `GRAMV2Network`: comparison baseline — same perception backbone, different planner |
| `arguments.py` | All CLI arguments for `train.py` (PPO, network sizes, ablation flags) |
| `train.py` | PPO training entry point |
| `test.py` | Evaluation — loads config from model directory |
| `plot.py` | Plot training reward / loss curves |
| `record_comparison.py` | Run all policies × N seeds; outputs `results/metrics.csv` + videos |
| `record_episode.py` | Record a single-policy episode MP4 |
| `visualize_grace.py` | Episode visualiser: env + slot assignment + group hulls |
| `visualize_cost_map.py` | Side-by-side video: environment + all 9 cost-map channels |
| `visualize_env.py` | Plain environment viewer (no policy needed) |
| `grace_perception_collect_data.py` | Collect simulation episodes for perception training |
| `grace_perception_collect_data_v2.py` | Same, with richer 21-feature format |
| `grace_perception_train_phase2.py` | Train GroupDetector (Phase 2) |
| `grace_perception_train_phase3.py` | Train SlotAttention (Phase 3) |
| `grace_perception_eval_detection.py` | Quantitative: GroupDetector vs DBSCAN (purity, ARI, F1) |
| `grace_perception_visualize_detection.py` | Qualitative: per-human slot assignments on an episode |
| `corl_2026/` | LaTeX paper source |

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

## GRACE Perception Module — Training Pipeline

Train the backbone **once** before any navigation training. These scripts are independent
of `train.py` and PPO — they have their own training loops and checkpoint formats.

### Step 1 — Data collection

Runs the simulator and saves pedestrian trajectories with ground-truth group labels.

```bash
# Standard format (7 features per human)
python grace_perception_collect_data.py \
    --train 3000 \          # training episodes         (default: 3000)
    --val   300  \          # validation episodes       (default: 300)
    --test  300  \          # test episodes             (default: 300)
    --out   gram_v2_data    # output directory          (default: gram_v2_data)

# v2 format (21 features per human — recommended for Phase 2 v2 checkpoint)
python grace_perception_collect_data_v2.py \
    --train 3000 \
    --val   300  \
    --test  300  \
    --out   gram_v2_data_v2
```

Output: `gram_v2_data/` or `gram_v2_data_v2/` (gitignored).
Each `.pkl` file stores `(feats, masks, labels)` for one episode.
Collecting 3000 episodes takes ~20 min on CPU.

---

### Step 2 — Train GroupDetector (Phase 2)

Supervised pairwise co-assignment BCE loss. Reads dataset from Step 1.

```bash
python grace_perception_train_phase2.py \
    --data       gram_v2_data          \   # dataset dir from Step 1     (default: gram_v2_data)
    --save       trained_models/gram_v2/phase2_v2 \  # checkpoint output (default: trained_models/gram_v2/phase2)
    --phase1     trained_models/gram_v2/phase1/B/best.pt \  # optional warm-start checkpoint
    --epochs     60     \   # training epochs                            (default: 60)
    --batch      256    \   # batch size                                 (default: 256)
    --lr         5e-4   \   # learning rate — AdamW + cosine schedule    (default: 5e-4)
    --lambda-aux 0.3    \   # weight of auxiliary spatial-smoothness loss (default: 0.3)
    --gnn-layers 3      \   # GNN message-passing layers in GroupDetector (default: 3)
    --workers    4      \   # DataLoader worker threads                  (default: 4)
    --no-cuda           # force CPU (omit to use GPU automatically)
```

Best checkpoint saved as `trained_models/gram_v2/phase2_v2/best.pt`.
Target: validation F1 > 0.85, ARI > 0.75 by epoch 40.

**What GroupDetector learns:** each pedestrian's [px, py, vx, vy, speed, sin_θ, cos_θ] is
encoded to 64-d, then 3 rounds of GNN message-passing pull group-mates together in embedding space.
`W_final[i,j]` ≈ probability that pedestrians i and j belong to the same group.

---

### Step 3 — Train SlotAttention (Phase 3)

GroupDetector is **frozen**. Only SlotAttention weights are trained.
Loss: co-assignment BCE + slot diversity entropy (prevents all pedestrians collapsing to one slot).

```bash
python grace_perception_train_phase3.py \
    --data       gram_v2_data           \  # dataset dir               (default: gram_v2_data)
    --phase2     trained_models/gram_v2/phase2_v2/best.pt \  # Phase 2 checkpoint (required)
    --save       trained_models/gram_v2/phase3 \  # checkpoint output  (default: trained_models/gram_v2/phase3)
    --epochs     40     \   # training epochs                           (default: 40)
    --batch      256    \   # batch size                                (default: 256)
    --lr         1e-3   \   # learning rate — Adam + cosine schedule    (default: 1e-3)
    --lambda-div 0.1    \   # slot diversity entropy loss weight        (default: 0.1)
    --K          3      \   # number of group-prototype slots           (default: 3)
    --n-iters    3      \   # slot attention refinement iterations      (default: 3)
    --workers    4      \
    --no-cuda
```

Best checkpoint saved as `trained_models/gram_v2/phase3/best.pt`.
Target: slot purity > 0.90 by epoch 20.

**What SlotAttention learns:** K=3 slot vectors iteratively attend to the N human embeddings
from GroupDetector. After convergence, `alpha[k,n]` ≈ probability human n belongs to group k.
This soft assignment drives group-layer synthesis (L6 cohesion, L7 repulsion) in the cost map.

---

### Evaluate and visualise perception quality

```bash
# Quantitative: purity / ARI / F1 comparing GroupDetector vs DBSCAN
python grace_perception_eval_detection.py \
    --data      gram_v2_data                         \   # dataset dir
    --phase1    trained_models/gram_v2/phase1        \   # Phase 1 dir (optional)
    --phase2    trained_models/gram_v2/phase2_v2     \   # Phase 2 dir
    --vel-scale 0.5   # velocity feature scale factor    (default: 0.5)

# Qualitative: frame-by-frame slot assignment on a saved episode
python grace_perception_visualize_detection.py \
    --data       gram_v2_data                         \
    --phase2     trained_models/gram_v2/phase2_v2/best.pt \
    --mode       random     \   # 'random'|'best'|'worst'|'idx'          (default: random)
    --idx        None       \   # episode index when --mode idx          (default: None)
    --n-frames   6          \   # frames to show in grid plot            (default: 6)
    --threshold  0.5        \   # W_final groupness threshold            (default: 0.5)
    --seed       0          \
    --gif               \       # save animated GIF instead of static grid
    --split-gif         \       # save a separate GIF per human
    --out        None   \       # output path (auto-named if None)
    --no-cuda
```

---

## GRACE Navigation — Training Pipeline

Navigation training reads the perception checkpoints from `trained_models/gram_v2/`.
Three curriculum stages. Edit `crowd_nav/configs/config.py` before each stage.

**These two args must always be set for GRACE (they don't match the defaults in `arguments.py`):**
```
--human_node_rnn_size 256        # GRU hidden state size
--human_human_edge_rnn_size 14   # 2-frame per-human feature buffer size
```

### Stage A — Navigation warm-up

Small env (5 humans, 4 m), trainable backbone, no slots, no group penalties.

**Config changes** (`crowd_nav/configs/config.py`):
```python
robot.policy                       = 'grace'
grace.freeze_backbone              = False
grace.use_slots                    = False
grace.use_aux_loss                 = False
sim.human_num, sim.circle_radius, sim.arena_size = 5, 4.0, 4.0
group.num_groups, group.num_on_path = 1, 0
group.types                        = ['static_f']
```

```bash
python train.py \
    --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 \
    --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/stageA \
    --lr 7e-4 \
    --use-linear-lr-decay \
    --num-processes 16 \
    --num-env-steps 20000000
```

Advance when SR ≥ 40% sustained over ~500 PPO updates.

---

### Stage B — Frozen backbone, planner learning

Larger env (15 humans, 6 m), perception frozen, slots on, group penalties added.

**Config changes**:
```python
grace.freeze_backbone                  = True
grace.use_slots                        = True
sim.human_num, sim.circle_radius, sim.arena_size = 15, 6.0, 6.0
group.num_groups, group.num_on_path    = 2, 1
group.types                            = ['static_f', 'dynamic_lf']
reward.discomfort_group_dist           = 0.35
reward.discomfort_grp_penalty_factor   = 10
reward.grp_collision_penalty           = -5
```

```bash
python train.py \
    --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 \
    --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/stageB \
    --lr 5e-4 \
    --use-linear-lr-decay \
    --resume \
    --load_path trained_models/gram_map/stageA/checkpoints/best.pt
```

Advance when SR > 40%, mean reward positive and stable.

---

### Stage C — Joint fine-tuning with auxiliary loss

Same env as Stage B. Unfreeze backbone, enable self-supervised occupancy loss.

**Config changes**:
```python
grace.freeze_backbone  = False
grace.use_aux_loss     = True
grace.aux_loss_weight  = 0.1
```

```bash
python train.py \
    --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 \
    --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/stageC \
    --lr 5e-5 \
    --use-linear-lr-decay \
    --resume \
    --load_path trained_models/gram_map/stageB/checkpoints/best.pt
```

`OccupancyHead` predicts future occupancy from the cost stack (self-supervised signal).
Its loss is stored as `self._aux_loss` in `GRACENetwork` and added by `rl/ppo/ppo.py` automatically.

---

### Full `train.py` parameter reference

**PPO / training:**

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `trained_models/garn` | Checkpoint and config save directory |
| `--resume` | `False` | Resume from `--load_path` |
| `--load_path` | — | `.pt` checkpoint to resume from |
| `--env-name` | `CrowdSimPredRealGST-v0` | Gym env — use `CrowdSimVarNum-v0` for GRACE |
| `--lr` | `4e-5` | Learning rate |
| `--use-linear-lr-decay` | `False` | Decay LR linearly to 0 over training |
| `--num-processes` | `16` | Parallel envs (reduce to 8 if GPU OOM) |
| `--num-env-steps` | `20e6` | Total environment steps |
| `--num-steps` | `30` | Rollout length per env per PPO update |
| `--num-mini-batch` | `2` | Mini-batches per PPO update |
| `--ppo-epoch` | `5` | PPO update epochs per rollout |
| `--clip-param` | `0.2` | PPO clip ε |
| `--value-loss-coef` | `0.5` | Value loss weight |
| `--entropy-coef` | `0.0` | Entropy bonus weight |
| `--gamma` | `0.99` | Reward discount factor |
| `--gae-lambda` | `0.95` | GAE λ |
| `--max-grad-norm` | `0.5` | Gradient clipping norm |
| `--seed` | `425` | Random seed |
| `--no-cuda` | `False` | Force CPU |
| `--save-interval` | `200` | Save checkpoint every N PPO updates |
| `--log-interval` | `20` | Print stats every N PPO updates |

**Network sizes (must match for GRACE):**

| Argument | GRACE value | Description |
|---|---|---|
| `--human_node_rnn_size` | **256** | GRU hidden state size (default 128 is wrong for GRACE) |
| `--human_human_edge_rnn_size` | **14** | 2-frame per-human buffer size (default 256 is wrong) |

**GRACE ablation flags** (all default `False` — normal training is unaffected):

| Argument | Ablation | What it removes |
|---|---|---|
| `--ablation_no_group_layers` | C1 | Group cohesion (L6) + repulsion (L7) cost channels |
| `--ablation_K_slots N` | C2 | Override K — e.g. `--ablation_K_slots 1` for single-slot |
| `--ablation_no_traj_layers` | C3 | Trajectory horizon channels (L2–L5) |
| `--ablation_no_aux_loss` | C4 | Self-supervised occupancy auxiliary loss |
| `--ablation_uniform_alpha` | C5 | SlotAttention — replaced with uniform group assignment |

---

## Testing & Evaluation

### `test.py`

Config is loaded from the **model directory**, not root `config.py`.

```bash
python test.py \
    --model_dir   trained_models/gram_map/stageC \  # dir with configs/ and checkpoints/ subdirs
    --test_model  best.pt       \  # checkpoint filename inside checkpoints/  (default: 41665.pt)
    --visualize                 \  # interactive matplotlib window            (default: False)
    --test_case   -1            \  # episode index; -1 = random              (default: -1)
    --render_traj               \  # overlay predicted future trajectories   (default: True)
    --save_slides               \  # save per-frame images to disk           (default: False)
    --group_avoid               # apply TAGA reactive avoidance on top (do NOT use for GRACE)
```

Results logged in `trained_models/<model_dir>/test/` and printed to terminal.

---

### `record_comparison.py`

Runs every registered policy × seeds and writes `results/metrics.csv` + MP4s.

```bash
python record_comparison.py \
    --policies   grace,orca,garn   \  # subset of POLICY_REGISTRY labels (omit for all)
    --seeds      0,1,2             \  # comma-separated seeds             (default: 0,1,2)
    --max-steps  300               \  # max steps per episode             (default: 300)
    --fps        10                \  # video frame rate                  (default: 10)
    --dpi        150               \  # video resolution                  (default: 150)
    --no-video                     \  # skip rendering — metrics only, much faster
    --group-types static_f,dynamic_lf,dynamic_free
```

Metrics: **SR** (success), **CR** (collision), **TR** (timeout), Avg Steps, **GCR** (group crossing rate), Avg Reward.

Add a policy by editing `POLICY_REGISTRY` at the top of `record_comparison.py`:
```python
dict(label='grace',
     policy_key='grace',
     model_dir='trained_models/gram_map/stageC',
     test_model='best.pt',
     with_taga=False)
```

---

### `record_episode.py`

```bash
python record_episode.py \
    --policy         orca         \  # policy key from policy_factory    (default: orca)
    --seed           0            \  # episode seed                      (default: 0)
    --out            episode.mp4  \  # output file                       (default: episode.mp4)
    --fps            10           \  # frame rate                        (default: 10)
    --dpi            150          \  # resolution                        (default: 150)
    --max-steps      300          \  # max episode length                (default: 300)
    --groups-only                 \  # groups only, no individuals
    --individuals-only            \  # individuals only, no groups
    --group-types    static_f,dynamic_lf,dynamic_free
```

---

### `plot.py`

```bash
python plot.py
```

Edit the `model_dirs` list inside `plot.py` to choose training runs. Reads `.csv` logs saved by `train.py`.

---

## Visualisation

### `visualize_grace.py` — episode with slot assignment overlay

```bash
python visualize_grace.py \
    --model_dir   trained_models/gram_map/stageC  \  # (required)
    --test_model  best.pt       \  # checkpoint filename         (default: 41665.pt)
    --seed        0             \  # episode seed                (default: 0)
    --out         grace_vis.mp4 \  # output video               (default: grace_vis.mp4)
    --fps         8             \  # frame rate                  (default: 8)
    --dpi         130           \  # resolution                  (default: 130)
    --max-steps   400           \  # max episode length          (default: 400)
    --device      auto          \  # 'auto'|'cuda'|'cpu'|'cuda:0' (default: auto)
    --show-hulls                \  # draw convex group hulls (default: on)
    --no-show-hulls             # hide group hulls
```

---

### `visualize_cost_map.py` — cost-map channel video

Left: environment. Right: 3×3 grid of the 9 cost channels.

```bash
python visualize_cost_map.py \
    --model_dir   trained_models/gram_map/stageC  \  # (required)
    --test_model  best.pt       \  # checkpoint filename         (default: best.pt)
    --seed        0             \  # episode seed                (default: 0)
    --out         cost_map.mp4  \  # output video               (default: cost_map.mp4)
    --fps         8             \  # frame rate                  (default: 8)
    --dpi         130           \  # resolution                  (default: 130)
    --max-steps   200           \  # max episode length          (default: 200)
    --device      auto
```

The 9 cost-map channels:

| Channel | Name | Colour map | Encodes |
|---|---|---|---|
| L1 | Individual occupancy | Reds | Current pedestrian Gaussian footprints |
| L2 | Trajectory t+0.3 s | Oranges | Predicted occupancy 0.3 s ahead |
| L3 | Trajectory t+0.7 s | YlOrBr | Predicted occupancy 0.7 s ahead |
| L4 | Trajectory t+1.0 s | YlOrRd | Predicted occupancy 1.0 s ahead |
| L5 | Trajectory t+1.5 s | OrRd | Predicted occupancy 1.5 s ahead |
| L6 | Group cohesion | Purples | Attractive potential around group centroid |
| L7 | Group repulsion | RdPu | Repulsive potential at group boundary |
| L8 | Goal attractor | Greens | Gradient toward robot goal |
| L9 | Boundary | Blues | Arena keep-out zone |

---

### `visualize_env.py` — plain environment viewer

```bash
python visualize_env.py                    # mixed (default)
python visualize_env.py --groups-only
python visualize_env.py --individuals-only
```

---

## Checkpoint layout

```
trained_models/
  gram_v2/
    phase2_v2/best.pt    ← GroupDetector  (grace_perception_train_phase2.py output)
    phase3/best.pt       ← SlotAttention  (grace_perception_train_phase3.py output)
  gram_map/
    stageA/checkpoints/  ← Stage A nav policy checkpoints
    stageB/checkpoints/  ← Stage B nav policy checkpoints
    stageC/checkpoints/  ← Stage C nav policy  ← use for final evaluation
      best.pt
      <step>.pt           periodic saves every 200 PPO updates
```

> `trained_models/` is gitignored — checkpoints live only on the training machine.
> Directory names (`gram_v2/`, `gram_map/`) are passed via `--model_dir` and are
> intentionally not renamed to avoid breaking existing checkpoints.

---

## Comparison baselines

| Policy key | Class | Description |
|---|---|---|
| `orca` | `ORCA` | Reciprocal collision avoidance (classical) |
| `social_force` | `SOCIAL_FORCE` | Social Force Model (classical) |
| `srnn` | `SRNN` | DS-RNN (neural) |
| `selfAttn_merge_srnn` | `selfAttn_merge_SRNN` | Intention-aware RL (neural) |
| `selfAttn_merge_srnn_grpAttn` | `selfAttn_merge_SRNN_GrpAttn` | GRAM (group-attention, neural) |
| `garn` | `GARN` | GARN — Lu et al. RA-L 2025 |
| `gram_v2` | `GRAMV2` | GRAM-v2 (grace_perception backbone + cross-attention planner) |
| `grace` | `GRACE` | **This work** |

Register policies for comparison runs in `POLICY_REGISTRY` at the top of `record_comparison.py`.
