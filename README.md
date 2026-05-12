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

## Visualize cost map

```bash
python visualize_grace.py --model_dir trained_models/gram_map/stageC --test_model best.pt --seed 3
python visualize_cost_map.py --model_dir trained_models/gram_map/stageC --test_model best.pt --seed 3 --out cost_map.mp4
```
