# Training Guide

Training GARN, intention_rl (selfAttn_merge_srnn), and SRNN in the realistic group-aware environment.

## Environment Setup (run once on the GPU PC)

```bash
conda create -n crowdnav python=3.8
conda activate crowdnav
pip install -r requirements.txt

# PyTorch with CUDA — match your CUDA version:
pip install torch==1.12.1+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# OpenAI Baselines:
git clone https://github.com/openai/baselines.git
cd baselines && pip install -e . && cd ..

# Python-RVO2:
cd Python-RVO2 && python setup.py install && cd ..
```

---

## GST Predictor

Pre-trained GST models are included in `gst_updated/results/`. No retraining needed.

| Model | Path | Use when |
|---|---|---|
| Randomized humans | `gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj` | `randomize_attributes=True` (default) |
| Non-randomized | `gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj` | `randomize_attributes=False` |

`pred.model_dir` in `config.py` is already set to the `_rand` model. Do not change it.

---

## Training GARN

**Config changes in `crowd_nav/configs/config.py`:**
```python
robot.policy = 'garn'             # line ~228
reward.use_garn_reward = True     # already True
sim.predict_method = 'inferred'   # already 'inferred'
```

**Command:**
```bash
python train.py \
  --output_dir trained_models/garn \
  --env-name CrowdSimPredRealGST-v0 \
  --num-processes 16
```

Monitor progress:
```bash
python plot.py
```

---

## Training intention_rl (selfAttn_merge_srnn)

**Config changes in `crowd_nav/configs/config.py`:**
```python
robot.policy = 'selfAttn_merge_srnn'   # change from 'garn'
reward.use_garn_reward = False         # change from True
sim.predict_method = 'inferred'        # keep as 'inferred'
```

**Command:**
```bash
python train.py \
  --output_dir trained_models/intention_rl_realistic \
  --env-name CrowdSimPredRealGST-v0 \
  --num-processes 16
```

> Uses the GST predictor (`predict_method='inferred'`) — same env class as GARN.
> The `_rand` GST model at `pred.model_dir` is already correct.

---

## Training SRNN (DS-RNN baseline)

**Config changes in `crowd_nav/configs/config.py`:**
```python
robot.policy = 'srnn'           # change from 'garn'
reward.use_garn_reward = False  # change from True
sim.predict_method = 'none'     # change from 'inferred'
```

Also update the wrapper flag (required when predict_method != 'inferred'):
```python
# The config.py already handles this automatically via:
# if sim.predict_method == 'inferred': env.use_wrapper = True
# else: env.use_wrapper = False
# So no manual change needed here.
```

**Command:**
```bash
python train.py \
  --output_dir trained_models/srnn_realistic \
  --env-name CrowdSimVarNum-v0 \
  --num-processes 16
```

> SRNN does not use trajectory prediction — no GST model needed.

---

## What `train.py` Saves Per Model Directory

Each training run automatically saves:

| File | Purpose |
|---|---|
| `checkpoints/*.pt` | Model weights (one per save interval) |
| `configs/config.py` | Simulation config frozen at training time |
| `arguments.py` | Network hyperparameters |
| `env_name.txt` | Exact env used (`CrowdSimPredRealGST-v0` or `CrowdSimVarNum-v0`) |

`env_name.txt` is critical — `record_comparison.py` reads it to auto-detect whether to use the GST predictor at evaluation time. No manual changes needed.

---

## After Training: Bring Checkpoints Back

Copy the trained model dirs back to the evaluation machine:
```bash
scp -r user@gpu-pc:/path/to/crowd_nav_with_group/trained_models/intention_rl_realistic \
    /home/lenovo/crowd_nav_with_group/trained_models/

scp -r user@gpu-pc:/path/to/crowd_nav_with_group/trained_models/srnn_realistic \
    /home/lenovo/crowd_nav_with_group/trained_models/
```

Then update `POLICY_REGISTRY` in `record_comparison.py`:
```python
dict(label='srnn',
     policy_key='srnn',
     model_dir='trained_models/srnn_realistic',
     with_taga=True),
dict(label='intention_rl',
     policy_key='selfAttn_merge_srnn',
     model_dir='trained_models/intention_rl_realistic',
     with_taga=True),
```

Run full comparison (all policies, 100 seeds, metrics only):
```bash
python -u record_comparison.py \
  --policies orca,orca+taga,srnn,srnn+taga,intention_rl,intention_rl+taga,garn \
  --seeds $(seq -s, 0 99) \
  --no-video | tee results/summary_all.txt
```

Or SRNN + intention_rl only:
```bash
python -u record_comparison.py \
  --policies srnn,srnn+taga,intention_rl,intention_rl+taga \
  --seeds $(seq -s, 0 99) \
  --no-video | tee results/summary_srnn_intention.txt
```

> `intention_rl` and `intention_rl+taga` automatically use the GST predictor at eval time (detected from `env_name.txt`). No extra flags needed.
> `srnn` uses `CrowdSimVarNum-v0` without GST (also auto-detected).
> The `+taga` variants are generated automatically — no separate checkpoint needed.

---

## Config Quick-Reference

| Policy | `robot.policy` | `sim.predict_method` | `reward.use_garn_reward` | `--env-name` | GST during training | `use_wrapper` |
|---|---|---|---|---|:---:|:---:|
| GARN | `'garn'` | `'inferred'` | `True` | `CrowdSimPredRealGST-v0` | ✓ | True |
| intention_rl | `'selfAttn_merge_srnn'` | `'inferred'` | `False` | `CrowdSimPredRealGST-v0` | ✓ | True |
| SRNN | `'srnn'` | `'none'` | `False` | `CrowdSimVarNum-v0` | ✗ | False |

**`use_wrapper` is set automatically** by `config.py`:
- `predict_method = 'inferred'` → `env.use_wrapper = True` (VecPretextNormalize wraps the envs and runs GST every rollout step)
- `predict_method = 'none'` → `env.use_wrapper = False`

You never need to set `use_wrapper` manually — changing `predict_method` is enough.

**GST predictor**: pre-trained models in `gst_updated/results/`. Path already set in `config.py` (`pred.model_dir`). No retraining needed for any of the above policies.

> After each training run, restore `config.py` to the GARN defaults before committing.
