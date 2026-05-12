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

Validates GRACE's GroupDetector backbone against DBSCAN in isolation, independent of the navigation policy.
Results feed into two CoRL tables. All values are recorded in `perception_detection_results.txt`.

| Table | Location in paper | What it shows |
|---|---|---|
| **Table 1** | `grace.tex` §5.2 | Best-eps DBSCAN vs GroupDetector Phase1 vs Phase2 |
| **Table 2** | `grace_appendix.tex` §A.2 | DBSCAN across all ε values (both feature modes) |

---

### Replicate Table 1 — Main comparison (grace.tex §5.2)

**Normal path** — checkpoints and `results.pt` files already exist:
```bash
bash run_dbscan_comparison.sh
```
Runs in ~30 s. Prints ASCII + LaTeX table. Copy LaTeX into `grace.tex` Table 2.

**Force re-evaluation** — re-run model inference from checkpoint (ignores saved results.pt):
```bash
bash run_dbscan_comparison.sh --force
```

**Full rebuild from scratch** — checkpoints missing or need to retrain:
```bash
# 1. Collect rollout data (only if gram_v2_data/ is missing)
python gram_v2_collect_data.py

# 2. Train Phase 1 (encoder, no GNN)
python gram_v2_train_phase1.py --variant B \
    --save trained_models/gram_v2/phase1_v2/B

# 3. Train Phase 2 (encoder + GNN = GRACE backbone)
python gram_v2_train_phase2.py \
    --phase1 trained_models/gram_v2/phase1_v2/B/best.pt \
    --save trained_models/gram_v2/phase2_v2

# 4. Regenerate results.pt from the trained checkpoints
bash run_perception_eval.sh

# 5. Run comparison
bash run_dbscan_comparison.sh
```

---

### Replicate Table 2 — DBSCAN ε sweep (grace_appendix.tex §A.2)

**Fastest — fill all rows at once** (~60 s, evaluates on 2000 test samples per eps):
```bash
python eval_detection_comparison.py --eps-sweep-only --n-test 2000
```
Prints a table of F1/Prec/Recall/ARI for every ε (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
for both position-only and pos+vel modes. Copy values into `perception_detection_results.txt`,
then paste into `grace_appendix.tex` Table A2.

**One row at a time** — position-only mode:
```bash
python eval_detection_comparison.py --fixed-eps 0.5  --mode position --dbscan-only
python eval_detection_comparison.py --fixed-eps 0.8  --mode position --dbscan-only
python eval_detection_comparison.py --fixed-eps 1.0  --mode position --dbscan-only
python eval_detection_comparison.py --fixed-eps 1.2  --mode position --dbscan-only
python eval_detection_comparison.py --fixed-eps 2.0  --mode position --dbscan-only
python eval_detection_comparison.py --fixed-eps 2.5  --mode position --dbscan-only
```

**One row at a time** — position + velocity mode:
```bash
python eval_detection_comparison.py --fixed-eps 0.5  --mode pos+vel --dbscan-only
python eval_detection_comparison.py --fixed-eps 0.8  --mode pos+vel --dbscan-only
python eval_detection_comparison.py --fixed-eps 1.0  --mode pos+vel --dbscan-only
python eval_detection_comparison.py --fixed-eps 1.2  --mode pos+vel --dbscan-only
python eval_detection_comparison.py --fixed-eps 1.5  --mode pos+vel --dbscan-only
python eval_detection_comparison.py --fixed-eps 2.0  --mode pos+vel --dbscan-only
```

After each run: copy the printed F1/Prec/Recall/ARI into `perception_detection_results.txt`
(Table 2 section), then replace the matching `\tbd{}` in `grace_appendix.tex`.

---

### Confirmed results (2026-05-12)

| Method | F1 | Prec | Recall | ARI |
|---|---|---|---|---|
| DBSCAN pos. (ε*=1.5) | 0.314 | 0.261 | 0.395 | 0.149 |
| DBSCAN pos+vel (ε*=2.5) | 0.379 | 0.285 | 0.566 | 0.310 |
| GroupDetector Phase1 (encoder) | 0.687 | 0.762 | 0.625 | 0.558 |
| **GroupDetector Phase2 (GNN — GRACE backbone)** | **0.770** | 0.709 | **0.843** | **0.631** |

---

### Baseline results (confirmed 2026-05-12)

| Model | SR | CR | TR | GCR | Mean Reward |
|---|---|---|---|---|---|
| **GRACE full** (stageC/best.pt) | **0.92** | 0.03 | 0.05 | 0.07% | 32.70 |
| C1 No group layers | — | — | — | — | — |
| C2 K=1 slot | — | — | — | — | — |
| C3 No traj layers | — | — | — | — | — |
| C4 No aux loss | — | — | — | — | — |
| C5 Uniform alpha | — | — | — | — | — |
