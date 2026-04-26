# Training Guide

End-to-end guide for training GARN, intention_rl (selfAttn_merge_srnn), and SRNN in the
realistic group-aware crowd navigation environment, followed by full multi-policy comparison.

---

## 1. Environment Setup (run once on the GPU PC)

```bash
conda create -n crowdnav python=3.8
conda activate crowdnav
pip install -r requirements.txt

# PyTorch with CUDA — match your CUDA version:
pip install torch==1.12.1+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# OpenAI Baselines (required by PPO trainer):
git clone https://github.com/openai/baselines.git
cd baselines && pip install -e . && cd ..

# Python-RVO2 (ORCA collision avoidance for humans):
cd Python-RVO2 && python setup.py install && cd ..
```

---

## 2. GST Predictor (no action needed)

Pre-trained GST models are already in `gst_updated/results/`. Do **not** retrain them.

`pred.model_dir` in `config.py` is already set to the randomized-humans model:
```
gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
```

GARN and intention_rl use this automatically during training. SRNN does not use it at all.

---

## 3. Training GARN

GARN (Lu et al., IEEE RA-L 2025) is a group-aware neural policy trained with a custom
group reward (R_grp). It uses the GST predictor to observe predicted human trajectories
during training.

**Edit `crowd_nav/configs/config.py`:**
```python
robot.policy          = 'garn'      # already the default
reward.use_garn_reward = True       # already True
sim.predict_method    = 'inferred'  # already 'inferred'
```

**Train:**
```bash
python train.py \
  --output_dir trained_models/garn \
  --env-name CrowdSimPredRealGST-v0 \
  --num-processes 16
```

**Monitor:**
```bash
python plot.py
```

> GST predictor fills 5-step predicted future positions into `spatial_edges` every rollout
> step via `VecPretextNormalize`. The network learns to use these predictions for
> group-aware navigation.

---

## 4. Training intention_rl (selfAttn_merge_srnn)

intention_rl is an intention-aware RL policy using self-attention over human states.
Like GARN, it uses the GST predictor during training but without GARN's group reward.

**Edit `crowd_nav/configs/config.py`:**
```python
robot.policy          = 'selfAttn_merge_srnn'   # change from 'garn'
reward.use_garn_reward = False                  # change from True
sim.predict_method    = 'inferred'              # keep — GST is used
```

**Train:**
```bash
python train.py \
  --output_dir trained_models/intention_rl_realistic \
  --env-name CrowdSimPredRealGST-v0 \
  --num-processes 16
```

> Same env class as GARN (`CrowdSimPredRealGST-v0`). GST predictor is active.
> `env.use_wrapper = True` is set automatically when `predict_method = 'inferred'`.

---

## 5. Training SRNN (DS-RNN baseline)

SRNN is a spatial-temporal graph RNN baseline. It does **not** use trajectory prediction —
it only sees current human positions. Simpler env, trains faster.

**Edit `crowd_nav/configs/config.py`:**
```python
robot.policy          = 'srnn'    # change from 'garn'
reward.use_garn_reward = False    # change from True
sim.predict_method    = 'none'    # change from 'inferred'
```

**Train:**
```bash
python train.py \
  --output_dir trained_models/srnn_realistic \
  --env-name CrowdSimVarNum-v0 \
  --num-processes 16
```

> No GST predictor, no wrapper. `env.use_wrapper = False` is set automatically
> when `predict_method = 'none'`.

---

## 6. Restore config.py After Training

After finishing all training runs, restore `config.py` to GARN defaults before committing:

```python
robot.policy          = 'garn'
reward.use_garn_reward = True
sim.predict_method    = 'inferred'
```

---

## 7. What train.py Saves Per Model Directory

Each training run automatically writes these files to `--output_dir`:

| File | Purpose |
|---|---|
| `checkpoints/*.pt` | Model weights saved every N updates |
| `configs/config.py` | Simulation config frozen at training time |
| `arguments.py` | Network architecture and PPO hyperparameters |
| `env_name.txt` | Exact env used (`CrowdSimPredRealGST-v0` or `CrowdSimVarNum-v0`) |

`env_name.txt` is read by `record_comparison.py` at eval time to auto-detect whether
to activate the GST predictor. No manual changes needed.

---

## 8. Bring Checkpoints to Evaluation Machine

```bash
scp -r user@gpu-pc:/path/to/crowd_nav_with_group/trained_models/garn \
    /home/lenovo/crowd_nav_with_group/trained_models/

scp -r user@gpu-pc:/path/to/crowd_nav_with_group/trained_models/intention_rl_realistic \
    /home/lenovo/crowd_nav_with_group/trained_models/

scp -r user@gpu-pc:/path/to/crowd_nav_with_group/trained_models/srnn_realistic \
    /home/lenovo/crowd_nav_with_group/trained_models/
```

---

## 9. Update POLICY_REGISTRY in record_comparison.py

Open `record_comparison.py` and update `POLICY_REGISTRY` at the top with the new model dirs:

```python
POLICY_REGISTRY = [
    dict(label='orca',         policy_key='orca',                       model_dir=None,
         with_taga=True),
    dict(label='social_force', policy_key='social_force',               model_dir=None,
         with_taga=True),
    dict(label='srnn',         policy_key='srnn',
         model_dir='trained_models/srnn_realistic',                     with_taga=True),
    dict(label='intention_rl', policy_key='selfAttn_merge_srnn',
         model_dir='trained_models/intention_rl_realistic',             with_taga=True),
    dict(label='garn',         policy_key='garn',
         model_dir='trained_models/garn',                               with_taga=False),
]
```

> `with_taga=True` auto-generates a `<label>+taga` variant. No separate checkpoint needed.
> `garn` uses `with_taga=False` because it has built-in group-awareness.

---

## 10. Full Comparison Commands

### All policies — full 100-seed evaluation (metrics only, no video)

```bash
python -u record_comparison.py \
  --policies orca,orca+taga,social_force,social_force+taga,srnn,srnn+taga,intention_rl,intention_rl+taga,garn \
  --seeds $(seq -s, 0 99) \
  --no-video | tee results/summary_all.txt
```

### Quick smoke-test (3 seeds, with video)

```bash
python record_comparison.py \
  --policies orca,orca+taga,social_force,social_force+taga,srnn,srnn+taga,intention_rl,intention_rl+taga,garn \
  --seeds 0,1,2
```

### Neural policies only

```bash
python -u record_comparison.py \
  --policies srnn,srnn+taga,intention_rl,intention_rl+taga,garn \
  --seeds $(seq -s, 0 99) \
  --no-video | tee results/summary_neural.txt
```

### Classical baselines only

```bash
python -u record_comparison.py \
  --policies orca,orca+taga,social_force,social_force+taga \
  --seeds $(seq -s, 0 99) \
  --no-video | tee results/summary_classical.txt
```

---

## 11. Policy Descriptions

| Policy | Type | GST at eval | TAGA | Description |
|---|---|:---:|:---:|---|
| `orca` | Classical | ✗ | ✗ | Reciprocal collision avoidance — strong baseline, no group awareness |
| `orca+taga` | Classical + reactive | ✗ | ✓ | ORCA with TAGA tangent override near groups |
| `social_force` | Classical | ✗ | ✗ | Social force model — physically motivated repulsion/attraction |
| `social_force+taga` | Classical + reactive | ✗ | ✓ | Social Force with TAGA tangent override near groups |
| `srnn` | Neural | ✗ | ✗ | DS-RNN: spatial-temporal graph RNN, no trajectory prediction |
| `srnn+taga` | Neural + reactive | ✗ | ✓ | SRNN with TAGA tangent override near groups |
| `intention_rl` | Neural | ✓ | ✗ | Self-attention RL with GST 5-step predicted futures |
| `intention_rl+taga` | Neural + reactive | ✓ | ✓ | intention_rl with TAGA tangent override near groups |
| `garn` | Neural | ✓ | ✗ | GARN (Lu et al. RA-L 2025): STGAN + group reward, built-in group awareness |

---

## 12. Evaluation Metrics

Output saved to `results/metrics.csv`. Summary printed to terminal.

| Metric | Symbol | Meaning |
|---|---|---|
| Success Rate | SR | Fraction of episodes where robot reached goal without collision |
| Collision Rate | CR | Fraction of episodes with any collision |
| Timeout Rate | TR | Fraction of episodes that hit max steps (SR + CR + TR ≈ 1.0) |
| Group Crossing Rate | GCR | Avg fraction of steps robot was inside a group hull — lower is better |
| Avg Steps | — | Navigation efficiency — lower means faster |
| Avg Reward | — | Cumulative reward per episode |

---

## Config Quick-Reference

| Policy | `robot.policy` | `sim.predict_method` | `reward.use_garn_reward` | `--env-name` | GST | `use_wrapper` |
|---|---|---|---|---|:---:|:---:|
| GARN | `'garn'` | `'inferred'` | `True` | `CrowdSimPredRealGST-v0` | ✓ | True |
| intention_rl | `'selfAttn_merge_srnn'` | `'inferred'` | `False` | `CrowdSimPredRealGST-v0` | ✓ | True |
| SRNN | `'srnn'` | `'none'` | `False` | `CrowdSimVarNum-v0` | ✗ | False |

`use_wrapper` is set automatically by `config.py` based on `predict_method`. Never set it manually.
