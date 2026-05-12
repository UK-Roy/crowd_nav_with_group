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

## Module overview

```
crowd_nav/grace_perception/        ← GRACE perception backbone (independent module)
  models.py                          GroupDetector: pedestrian encoder + GNN → 64-d embeddings
  slot_attention.py                  SlotAttention: K=3 group-prototype vectors from N embeddings
  gnn.py                             GroupGNN: graph neural network layer used by GroupDetector

rl/networks/grace_network.py       ← GRACE navigation policy (Phase 4 end-to-end)
rl/networks/grace_synthesizer.py   ← 9-channel BEV cost-map synthesizer + CNN planner
crowd_nav/policy/grace.py          ← Policy factory registration (key: 'grace')
crowd_nav/configs/config.py        ← config.grace section: grid_size, horizons, checkpoints

rl/networks/gram_v2_network.py     ← GRAM-v2 navigation policy (comparison baseline,
                                     also uses grace_perception backbone)
```

---

## GRACE Perception Module — Training Pipeline

The perception backbone is trained in 3 phases **before** navigation training.
All scripts are prefixed `grace_perception_*`.

### Phase 1 — Data collection

Collect trajectory episodes from the simulator for backbone supervision:

```bash
python grace_perception_collect_data.py
# or the updated version with richer features:
python grace_perception_collect_data_v2.py
```

Output: `gram_v2_data/` (gitignored). Per-episode pedestrian tracks with ground-truth group labels.

### Phase 2 — GroupDetector training

Trains the pedestrian encoder + GNN to produce 64-d per-human embeddings capturing group membership.
Supervised with pairwise co-assignment BCE loss.

```bash
python grace_perception_train_phase2.py
```

Checkpoint: `trained_models/gram_v2/phase2_v2/best.pt`

**Architecture** (`crowd_nav/grace_perception/models.py → GroupDetector`):
- Input: (B, N, 7) per-human features [px, py, vx, vy, v_norm, sin_θ, cos_θ]
- `PedestrianEncoder` → `PairwiseEdgeNetwork` → `GroupGNN` (3 layers)
- Output: `W_final` (N×N groupness matrix), `g` (N×64 refined embeddings)

### Phase 3 — SlotAttention training

Trains K=3 slot vectors to compress N per-human embeddings into group prototypes.
No Hungarian matching needed — uses co-assignment BCE + slot diversity entropy loss.

```bash
python grace_perception_train_phase3.py
```

Checkpoint: `trained_models/gram_v2/phase3/best.pt`

**Architecture** (`crowd_nav/grace_perception/slot_attention.py → SlotAttention`):
- Input: `g` (B, N, 64), `mask` (B, N) bool
- Iterative cross-attention (n_iters=3): slots attend to humans → GRU slot update
- Output: `slots` (B, K, 64) group prototypes, `alpha` (B, K, N) assignment weights

### Evaluate and visualise detection

```bash
# Quantitative: compare GroupDetector vs DBSCAN (purity, ARI, F1)
python grace_perception_eval_detection.py

# Qualitative: visualise per-human slot assignment on a live episode
python grace_perception_visualize_detection.py
```

---

## GRACE Navigation — Training Pipeline

Uses the pre-trained perception checkpoints as backbone.
Three curriculum stages (A → B → C). Set config values in `crowd_nav/configs/config.py`
before each stage as described below.

### Stage A — Navigation warm-up (backbone trainable, small env)

Config: 5 humans, 4 m arena, 1 static group. `config.grace.freeze_backbone = False`, slots off, no aux loss.

```bash
python train.py --env-name CrowdSimVarNum-v0 \
  --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
  --output_dir trained_models/gram_map/stageA \
  --lr 7e-4 --use-linear-lr-decay
```

Advance when: SR ≥ 40% sustained over ~500 updates.

### Stage B — Frozen backbone, cost-map planner learning

Config: 15 humans, 6 m arena, 2 mixed groups (static_f + dynamic_lf).
Set `config.grace.freeze_backbone = True` and `config.grace.use_slots = True` in `config.py`.

```bash
python train.py --env-name CrowdSimVarNum-v0 \
  --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
  --output_dir trained_models/gram_map/stageB \
  --lr 5e-4 --use-linear-lr-decay \
  --resume --load_path trained_models/gram_map/stageA/checkpoints/best.pt
```

Advance when: SR > 40%, mean reward positive and sustained.

### Stage C — Joint fine-tuning with auxiliary loss

Config: same as Stage B. Set `config.grace.freeze_backbone = False`, `config.grace.use_aux_loss = True`.

```bash
python train.py --env-name CrowdSimVarNum-v0 \
  --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
  --output_dir trained_models/gram_map/stageC \
  --lr 5e-5 --use-linear-lr-decay \
  --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt
```

The auxiliary loss (`OccupancyHead`) predicts future occupancy from the cost stack (self-supervised).
It is stored as `self._aux_loss` in `GRACENetwork` and picked up by `rl/ppo/ppo.py` automatically.

---

## Testing & Evaluation

```bash
# Evaluate a single checkpoint
python test.py --model_dir trained_models/gram_map/stageC --test_model best.pt

# Multi-policy comparison table (produces results/metrics.csv + videos/)
python record_comparison.py --seeds 0,1,2

# Metrics only, no video rendering (much faster)
python record_comparison.py --seeds 0,1,2,3,4 --no-video
```

Output metrics: **SR** (success rate), **CR** (collision rate), **TR** (timeout rate),
Avg Steps (navigation efficiency), **GCR** (group crossing rate — lower is less disruptive), Avg Reward.

```bash
# Plot training reward curves
python plot.py
```

---

## Visualisation

```bash
# Side-by-side: environment view + all 9 cost-map channels (saves MP4)
python visualize_cost_map.py \
  --model_dir trained_models/gram_map/stageC --test_model best.pt \
  --seed 3 --out cost_map.mp4

# Single-episode overlay with slot assignment and group hulls
python visualize_grace.py \
  --model_dir trained_models/gram_map/stageC --test_model best.pt \
  --seed 3

# Plain environment viewer (no policy needed — useful for env debugging)
python visualize_env.py
python visualize_env.py --groups-only
python visualize_env.py --individuals-only
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

## Checkpoint directory layout

```
trained_models/
  gram_v2/
    phase2_v2/best.pt    ← GroupDetector weights  (grace_perception_train_phase2.py)
    phase3/best.pt       ← SlotAttention weights  (grace_perception_train_phase3.py)
  gram_map/
    stageA/checkpoints/  ← Stage A nav policy
    stageB/checkpoints/  ← Stage B nav policy
    stageC/checkpoints/  ← Stage C nav policy  ← use this for evaluation
```

> `trained_models/` is gitignored — checkpoints live only on the training machine.
> Directory names (`gram_v2/`, `gram_map/`) are filesystem paths passed via `--model_dir`
> and are intentionally not renamed to avoid breaking existing checkpoints.

---

## Comparison baselines

| Policy key | Description |
|---|---|
| `orca` | ORCA (classical, reciprocal collision avoidance) |
| `social_force` | Social Force Model (classical) |
| `srnn` | DS-RNN (neural) |
| `selfAttn_merge_srnn` | Intention-aware RL (neural) |
| `selfAttn_merge_srnn_grpAttn` | GRAM (attention-based group-aware, neural) |
| `garn` | GARN (group-aware, Lu et al. RA-L 2025) |
| `gram_v2` | GRAM-v2 (grace_perception backbone + cross-attention planner) |
| `grace` | **GRACE** (this work) |

Add policies to `POLICY_REGISTRY` at the top of `record_comparison.py` to include them in comparison runs.
