# GRACE: Group-Responsive Avoidance via Cost-Map Encoding

**Branch: `grace` — CoRL 2026 submission code**

GRACE is an end-to-end deep RL navigation policy for robots moving through crowds with human groups.
It renders an explicit 9-channel bird's-eye-view cost map from a learned GroupDetector + SlotAttention
perception backbone, then plans through the cost map with a CNN planner and GRU temporal memory.

Paper draft: `corl_2026/corl_2026_template_submission/grace.tex`

---

## Repository layout

```
rl/networks/grace_network.py      ← GRACE policy network
rl/networks/grace_synthesizer.py  ← 9-channel cost-map synthesizer + CNN planner
crowd_nav/policy/grace.py         ← policy factory registration
crowd_nav/configs/config.py       ← config.grace section controls all hyperparams
corl_2026/                        ← paper LaTeX source
train.py / test.py                ← training and evaluation entry points
visualize_grace.py                ← single-episode visualizer (cost map + env)
visualize_cost_map.py             ← side-by-side cost-map video recorder
record_comparison.py              ← multi-policy comparison (metrics CSV + videos)
```

## Branch overview

| Branch | Purpose |
|---|---|
| `grace` | **This branch** — GRACE CoRL 2026 paper code |
| `gram-map` | Active development branch for GRACE |
| `ral-benchmark` | RA-L benchmark paper: realistic group env + TAGA |
| `main` | Shared environment baseline |

> **Note:** `trained_models/gram_map/` holds GRACE checkpoints (Stage A/B/C).
> The directory name is intentionally not renamed — pass it via `--model_dir`.

---

## Setup

```bash
conda create -n crowdnav python=3.8 && conda activate crowdnav
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/openai/baselines.git && cd baselines && pip install -e . && cd ..
cd Python-RVO2 && python setup.py install && cd ..
```

## Training GRACE (Stage C fine-tuning)

```bash
python train.py --env-name CrowdSimVarNum-v0 \
  --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
  --output_dir trained_models/gram_map/stageC \
  --lr 5e-5 --use-linear-lr-decay \
  --resume --load_path trained_models/gram_map/stageB/checkpoints/best.pt
```

## Evaluation

```bash
# Single model
python test.py --model_dir trained_models/gram_map/stageC --test_model best.pt

# Multi-policy comparison (produces metrics.csv + videos)
python record_comparison.py --seeds 0,1,2
```

## Visualize cost map (final GRACE result)

```bash
python visualize_grace.py --model_dir trained_models/gram_map/stageC --test_model best.pt --seed 3
python visualize_cost_map.py --model_dir trained_models/gram_map/stageC --test_model best.pt --seed 3 --out cost_map.mp4
```

---

## Ablation Study

Five ablations of the Stage C architecture. All start from `stageC/checkpoints/best.pt`.

| # | What is removed | Output dir |
|---|---|---|
| C1 | Group layers (L3 + L4 zeroed) | `ablation_C1_no_group` |
| C2 | K=1 slot (single group prototype) | `ablation_C2_K1` |
| C3 | Trajectory layers (L2 zeroed) | `ablation_C3_no_traj` |
| C4 | Auxiliary occupancy loss | `ablation_C4_no_aux` |
| C5 | Uniform slot assignment (alpha) | `ablation_C5_uniform` |

> **C** = Stage **C** of the training curriculum. All ablations are variants of the Stage C model.

Full training commands and the results table to fill in are in [`ABLATION_RESULTS.md`](ABLATION_RESULTS.md).

### Running an ablation (example: C1)

```bash
# 1. Train
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C1_no_group \
    --num-env-steps 10000000 --num-processes 16 --num-steps 30 \
    --num-mini-batch 2 --ppo-epoch 5 \
    --lr 5e-5 --entropy-coef 0.005 --eps 1e-5 \
    --gamma 0.99 --gae-lambda 0.95 \
    --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 \
    --use-linear-lr-decay --save-interval 200 --log-interval 20 \
    --resume --load_path trained_models/gram_map/stageC/checkpoints/best.pt \
    --ablation_no_group_layers

# 2. Find best checkpoint (top-3 by training reward, then pick highest test SR)
python test.py --model_dir trained_models/gram_map/ablation_C1_no_group --test_model XXXXX.pt
cp trained_models/gram_map/ablation_C1_no_group/checkpoints/XXXXX.pt \
   trained_models/gram_map/ablation_C1_no_group/checkpoints/best.pt

# 3. Final evaluation
python test.py --model_dir trained_models/gram_map/ablation_C1_no_group --test_model best.pt
```

See [`ABLATION_RESULTS.md`](ABLATION_RESULTS.md) for the full 4-step best.pt selection workflow and copy-paste commands for all five ablations.

### Ablation visualizations

`visualize_ablation.py` generates figures and optional videos for any ablation.
`visualize_cost_map.py` is reserved for the final GRACE result only.

```bash
# Training curves for all ablations (no model needed — reads progress.csv)
python visualize_ablation.py --plot training_curves

# Results bar chart: SR and GCR for all ablations (no model needed — reads test logs)
python visualize_ablation.py --plot results_bar

# Robot trajectory: baseline vs C1 side-by-side
python visualize_ablation.py --ablation C1 --plot trajectory --seed 42

# Cost map comparison: shows which channels go dark when ablated
python visualize_ablation.py --ablation C1 --plot cost_map --seed 42

# Generate video instead of static figure
python visualize_ablation.py --ablation C1 --plot trajectory --seed 42 --video

# Full report for one ablation (all plots + video)
python visualize_ablation.py --ablation C1 --plot all --seed 42 --video

# Batch: all ablations, all plots
python visualize_ablation.py --ablation all --plot all --seed 42
```

All outputs are saved to `ablation_figures/`.

---

## Group Detection: DBSCAN vs Perception Module

GRACE uses a two-phase learned GroupDetector as its perception backbone. This section documents how to replicate the two detection tables in the CoRL paper.

| Table | CoRL location | Script |
|---|---|---|
| Table 1 — main comparison (4 rows) | §5.2, `tab:detection` in `grace.tex` | `bash scripts/run_dbscan_comparison.sh` |
| Table 2 — DBSCAN ε sweep (14 rows) | Appendix A.2, `tab:detection_full` in `grace_appendix.tex` | `bash scripts/run_eps_sweep.sh` |

Scripts live in `scripts/`. Results are saved to `results/perception_detection_results.txt` (auto-created if missing — no setup needed).

---

### Perception backbone: data collection → training

The GroupDetector is trained in two phases using data collected from the simulator.

**Step 1 — Collect training data**

```bash
python collect_group_data.py --output_dir gram_v2_data --n_episodes 5000
```

This generates `gram_v2_data/{train,val,test}.npz` — pairwise (feature, label) samples recording which human pairs belong to the same group. Each sample is a 21-d feature vector (position, velocity, distance, bearing, relative speed).

**Step 2 — Train Phase 1: encoder only (no GNN)**

```bash
python train_group_detector.py \
    --data_dir gram_v2_data \
    --output_dir trained_models/gram_v2/phase1_v2/B \
    --n_gnn_layers 0 --hidden_size 256 --variant B
```

Phase 1 is a lightweight encoder + PairwiseEdgeNetwork classifier (no graph convolution). Checkpoint saved to `phase1_v2/B/best.pt`.

**Step 3 — Train Phase 2: encoder + GNN (GRACE backbone)**

```bash
python train_group_detector.py \
    --data_dir gram_v2_data \
    --output_dir trained_models/gram_v2/phase2_v2 \
    --n_gnn_layers 3 --hidden_size 256 --variant B \
    --resume --load_path trained_models/gram_v2/phase1_v2/B/best.pt
```

Phase 2 adds a 3-layer GNN on top of Phase 1. This is the backbone used inside the full GRACE navigation policy. Checkpoint saved to `phase2_v2/best.pt`.

> **Pre-trained checkpoints** — if you have `trained_models/gram_v2/phase1_v2/B/best.pt` and `trained_models/gram_v2/phase2_v2/best.pt`, skip Steps 1–3 and go straight to evaluation below.

---

### Replicate Table 1 — main comparison

Compares DBSCAN (position-only and position+velocity) against both GroupDetector phases on the held-out test set.

```bash
bash scripts/run_dbscan_comparison.sh
```

- Auto-creates `results/perception_detection_results.txt` if it doesn't exist.
- Detects saved `results.pt` files; runs full model inference only if they are missing.
- Prints the comparison table to stdout and writes results into `results/perception_detection_results.txt` (Table 1).
- Re-run with `bash scripts/run_dbscan_comparison.sh --force` to force re-evaluation even when cached results exist.

To regenerate model `results.pt` files from existing checkpoints only:

```bash
bash scripts/run_perception_eval.sh            # both phases
bash scripts/run_perception_eval.sh --phase1-only
bash scripts/run_perception_eval.sh --phase2-only
bash scripts/run_perception_eval.sh --no-cuda  # CPU fallback
```

**Confirmed results (2026-05-12, test set ~8,549 frames)**

| Method | F1 | Prec | Recall | ARI | AUROC |
|---|---|---|---|---|---|
| DBSCAN pos. only (ε*=1.5) | 0.314 | 0.261 | 0.395 | 0.149 | — |
| DBSCAN pos.+vel. (ε*=2.5) | 0.379 | 0.285 | 0.566 | 0.310 | — |
| Phase 1: encoder only | 0.687 | 0.762 | 0.625 | 0.558 | 0.935 |
| **Phase 2: enc.+GNN (GRACE backbone)** | **0.770** | **0.709** | **0.843** | **0.631** | **0.974** |

ε* = best ε found by val-set sweep (300 samples).

---

### Replicate Table 2 — DBSCAN ε sweep

Sweeps ε ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5} for both position-only and position+velocity modes.

```bash
bash scripts/run_eps_sweep.sh
```

- Auto-creates `results/perception_detection_results.txt` if it doesn't exist.
- Runs each ε value sequentially (~30–60 s total) and writes each result into the matching row in `results/perception_detection_results.txt` (Table 2) as soon as it completes.
- To control test-set size: `bash scripts/run_eps_sweep.sh --n-test=1000` (default 2000).
- CPU fallback: `bash scripts/run_eps_sweep.sh --no-cuda`.

One row at a time (alternative):

```bash
python eval_detection_comparison.py --fixed-eps 1.0 --mode position --dbscan-only --n-test 2000
python eval_detection_comparison.py --fixed-eps 1.0 --mode pos+vel  --dbscan-only --n-test 2000
```

Full sweep table (print all ε rows, no recording):

```bash
python eval_detection_comparison.py --eps-sweep-only --n-test 2000
```

---

### `results/perception_detection_results.txt` — where results are stored

Both scripts write into `results/perception_detection_results.txt`. This file:

- Is automatically created (with blank TBD rows) by both scripts if it does not exist — **no manual setup needed**.
- Tracks Table 1 (main comparison) and Table 2 (ε sweep) in CoRL paper format.
- Contains copy-paste instructions for updating `grace.tex` and `grace_appendix.tex` once all rows are filled.

After running both scripts, open `results/perception_detection_results.txt` and copy the filled values into the LaTeX tables.

---

### Baseline results (confirmed 2026-05-14)

| Model | SR | CR | TR | GCR | Mean Reward |
|---|---|---|---|---|---|
| **GRACE full** (stageC/best.pt) | **0.90** | 0.07 | 0.03 | 0.00% | 32.08 |
| C1 No group layers | 0.23 | 0.17 | 0.60 | 1.14% | 8.17 |
| C2 K=1 slot | 0.01 | 0.36 | 0.63 | 5.32% | -27.36 |
| C2 K=2 slots | 0.90 | 0.03 | 0.07 | 3.50% | 20.99 |
| C2 K=5 slots | 0.86 | 0.04 | 0.10 | 4.29% | 9.90 |
| C3 No traj layers | 0.91 | 0.02 | 0.07 | 3.30% | 20.45 |
| C4 No aux loss | 0.89 | 0.04 | 0.07 | 3.86% | 19.14 |
| C5 Uniform alpha | 0.12 | 0.44 | 0.44 | 6.15% | -7.46 |
