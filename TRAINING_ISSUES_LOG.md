# GRACE Training Issues Log

**File location:** `TRAINING_ISSUES_LOG.md` in the root of the repo, **`gram-map` branch only**
**Purpose:** Permanent record of every training failure, root cause, and fix applied during GRACE development. Check this file before starting a new training run to avoid repeating known mistakes.

---

## Why Training Results Were Not as Good as They Should Have Been

The GRACE model achieved SR=0.92 on evaluation, but the **training process itself was severely broken** due to multiple entropy-related bugs. This section explains the chain of problems clearly so it is never repeated.

### The short version

Three bugs combined to make the GRACE training much harder than it needed to be:

1. The entropy gradient in the PPO loss was **half its intended value** (typo in `FixedNormal.entrop()`).
2. Because of a missing upper clamp, `logstd` grew without bound, making the policy sample **completely random actions during training** (σ grew to 1082 m/s — far beyond any real speed).
3. NaN values from the unfrozen GroupDetector **silently corrupted gradients** early in stageC.

Together, these meant that during stageB and stageC, the robot was navigating with *random noise actions* during training — it only appeared to succeed because test evaluation uses the *mean* action (not a sample). The model learned **despite** the broken training, not because of it. The training took longer, was noisier, and converged to a lower training reward (mean ~10–20) than it would have with correct entropy handling.

### The full causal chain

```
entrop() typo (Issue 2)
    → entropy term in PPO loss is halved
    → entropy regularization is weaker than intended
    → logstd gradient from policy loss dominates
    → with entropy_coef=0.05 (too high), logstd is still pushed upward unchecked

No logstd clamp (Issue 3)
    → logstd grows: 0 → 7 over stageB (~42,000 updates)
    → σ grows: 1 → 1082 m/s
    → sampled training actions = random unit-circle direction (all clipped to v_pref)
    → gradient signal for the policy mean is extremely noisy (high-variance estimator)
    → training reward plateaus at 10–20 instead of converging toward 30+
    → model learns slowly through the noise — but the MEAN policy is still reasonable

NaN propagation (Issue 1)
    → at stageC backbone unfreeze, GroupDetector produces NaN
    → NaN flows to entropy → dist_entropy = NaN
    → PPO loss = NaN → all gradients = NaN → weights corrupted
    → training stalls / regresses in the first 200 updates of stageC
    → needed nan_to_num guards to patch through

Result: stageC reached SR=0.92 on test (deterministic mean actions),
        but training was inefficient and mean reward never exceeded ~25.
        With the entropy fixes now in place, ablation training should:
          - converge faster (lower-variance gradients)
          - reach higher training rewards (σ stays in [0.05, 1.65])
          - be more stable (no NaN risk, no runaway)
```

### What is fixed now (as of 2026-05-12)

| Bug | Fix | File |
|---|---|---|
| `entrop()` typo | Renamed to `entropy()` + `.sum(-1)` | `rl/networks/distributions.py` |
| No logstd clamp | `.clamp(-3.0, 0.5)` in `DiagGaussian.forward()` | `rl/networks/distributions.py` |
| NaN from backbone | `nan_to_num(0.0)` at g, alpha, cost_stack + gradients | `rl/networks/grace_network.py`, `rl/ppo/ppo.py` |

**For all future training (ablations, new stages):** use `--entropy-coef 0.005`. The clamp makes the old `0.05` value unnecessary and harmful.

---

## Issue 1 — NaN propagation through GroupDetector (stageC onset)

**Symptom:** Training loss became NaN immediately after stageC started (backbone unfrozen). Policy weights corrupted in the first few hundred updates. Mean reward dropped to −20.

**Root cause:** When `grace.freeze_backbone` flips from `True` → `False` at the start of stageC, the GroupDetector's GNN parameters are first exposed to the navigation gradient. On the first backward pass their gradients are ill-conditioned (the GNN was never trained with the navigation loss), producing NaN outputs in `g` (N×64 embeddings), which then propagated through SlotAttention `alpha` and into the cost stack.

**Fix (committed `1c01efd`, `3db7d67`):**
```python
# grace_network.py — three nan_to_num guards in forward()
g     = g.nan_to_num(0.0)          # guard GroupDetector output
alpha = alpha.nan_to_num(0.0)      # guard SlotAttention output
cost_stack = cost_stack.nan_to_num(0.0)  # guard before CNN planner

# ppo.py — zero NaN gradients before optimizer step
for p in self.actor_critic.parameters():
    if p.grad is not None:
        p.grad = p.grad.nan_to_num(0.0)
```

**Status:** Fixed. NaN no longer appears in stageC or ablation training.

---

## Issue 2 — FixedNormal.entropy() typo (silent bug since initial commit)

**Symptom:** Entropy term in PPO loss was computing per-element average (scalar = per-dim entropy, e.g. 1.42 for σ=1) instead of the correct action-summed entropy (scalar = 2×1.42 = 2.84 for σ=1, 2D action space). No crash — just miscalibrated entropy regularization. The bug existed from the very first commit and was never detected because training still converged.

**Root cause:** `distributions.py` had `def entrop(self)` (typo, missing 'y'). Since `FixedNormal` never overrode `entropy()`, all calls fell back to `torch.distributions.Normal.entropy()` which returns per-element shape `(B, N_actions)` instead of the desired summed shape `(B,)`. `Categorical` and `Bernoulli` did correctly sum, but `Normal` did not.

**Fix (committed `043a561`):**
```python
# distributions.py — FixedNormal
def entropy(self):
    return super().entropy().sum(-1)   # sum over action dims → shape (B,)
```

**Status:** Fixed. Entropy now correctly sums over action dimensions, matching Categorical and Bernoulli behavior.

---

## Issue 3 — logstd runaway (entropy grew from σ=1 to σ=1082 during stageB)

**Symptom:** `loss/policy_entropy` in `progress.csv` grew from 1.43 (start of stageB) to 8.41 (end of stageB) and then stayed frozen at 8.39 throughout stageC. This corresponds to `logstd` growing from ≈0 to ≈7, meaning the action distribution's standard deviation σ grew from 1 → 1082 m/s. During training, sampled actions were completely random noise, clipped to `v_pref` by `clip_action()`. Test SR remained 92% because evaluation uses `deterministic=True` (mean actions only, ignores σ).

**Root cause:** `entropy_coef=0.05` in the training command kept adding an entropy bonus to the PPO loss. Since there was no upper bound on `logstd`, the bonus drove it up indefinitely. The `FixedNormal.entrop()` typo (Issue 2) made the effective entropy bonus half its intended value, so it rose more slowly — but it still grew without bound.

**Fix (committed `043a561`):**
```python
# distributions.py — DiagGaussian.forward()
action_logstd = self.logstd(zeros).clamp(-3.0, 0.5)
# σ range: [e^-3, e^0.5] ≈ [0.05, 1.65]
```

**Status:** Fixed. logstd is now hard-clamped. Use `--entropy-coef 0.005` (not 0.05) for ablation training — the clamp makes the large coef unnecessary, and a small coef keeps mild exploration without runaway.

---

## Issue 4 — Policy entropy collapse during GRAM-v2 stage1 (predecessor model)

**Symptom:** During GRAM-v2 stage1 (`trained_models/gram_v2/stage1_unfreeze`), entropy collapsed to 0.25 around update 14760 (from min 0.25, starting near 1.42). The policy became near-deterministic prematurely, unable to explore its way out of local optima.

**Root cause:** Default `entropy_coef=0.0` (no entropy regularization) combined with the `FixedNormal.entrop()` typo — even if a non-zero coef was set, the gradient was halved. Without regularization, the policy gradient pushed `logstd` toward −∞ as soon as it found a locally good action.

**Fix:** The clamp added in Issue 3 (`logstd ≥ -3.0`) prevents collapse regardless of entropy_coef. A minimum σ of 0.05 ensures some exploration is always present.

**Status:** Fixed by the same logstd clamp from Issue 3.

---

## Issue 5 — Wrong environment config for ablation training

**Symptom (would-have-been):** The current `crowd_nav/configs/config.py` was left in Stage 3 state after stageC training ended (`human_num=15`, `circle_radius=6`, `realistic.enabled=False`). If ablation training started from stageC/best.pt with this config, the environment would be easier (fewer humans, no realistic phases) than what stageC was trained on. Ablation results would not be directly comparable to the stageC SR=92% baseline.

**Root cause:** `train.py` always reads from the live `crowd_nav/configs/config.py` and copies it to the output directory. The config was never updated after stageC completed.

**Fix (config updated before first ablation run):**
```python
# crowd_nav/configs/config.py — must match stageC training conditions
sim.human_num        = 20
sim.circle_radius    = 8.5
sim.arena_size       = 8.5
group.num_groups     = 3
group.num_on_path    = 2
group.types          = ['static_f', 'dynamic_lf', 'dynamic_free']
realistic.enabled    = True    # all phases A–E active
```

**Status:** Fixed in config. Ablation commands now use `--resume --load_path trained_models/gram_map/stageC/checkpoints/best.pt`.

---

## Issue 6 — gram_map policy alias missing (KeyError on stageA/B/C checkpoint load)

**Symptom:** `test.py` and `train.py --resume` crashed with `KeyError: 'gram_map'` when loading any stageA/B/C checkpoint. The saved configs use `robot.policy='gram_map'` (the old name before the rename to GRACE), but `policy_factory.py` and `model.py` only registered `'grace'`.

**Root cause:** The policy was renamed from `gram_map` → `grace` in commit `3d3fad9`, but saved checkpoints still carry the old policy name in their `configs/config.py`.

**Fix (committed `1f54ed3`):**
```python
# crowd_nav/policy/policy_factory.py
policy_factory['gram_map'] = GRACE   # backward-compat alias for stageA/B/C checkpoints

# rl/networks/model.py
elif base in ('grace', 'gram_map'):
    base = GRACENetwork

# test.py
if config.robot.policy in ('grace', 'gram_map'):
    grace_cfg = getattr(config, 'grace', None) or getattr(config, 'gram_map', None)
```

**Status:** Fixed. Both `policy='grace'` (new checkpoints) and `policy='gram_map'` (old checkpoints) load correctly.

---

## Issue 7 — best.pt checkpoints did not exist

**Symptom:** All ablation training commands and eval commands in `GRACE_ARCHITECTURE.md` referenced `checkpoints/best.pt`, but no such file existed in stageB or stageC. The per-update checkpoints are saved as `<update_number>.pt` only.

**Root cause:** The training script (`train.py`) saves periodic checkpoints by update number but does not track or save a `best.pt` based on mean reward.

**Fix:** Manually identified the best-reward available checkpoint in each stage from `progress.csv` and copied it:

| File | Source | Reward |
|---|---|---|
| `stageB/checkpoints/best.pt` | `35600.pt` (update 35600) | 19.74 |
| `stageC/checkpoints/best.pt` | `41000.pt` (update 41000) | 20.36 → **test SR=0.92** |

**Status:** Fixed. `best.pt` files exist in both stageB and stageC.

> **Future prevention:** Consider adding a `best.pt` saver to `train.py` that overwrites when mean reward improves.

---

## Issue 8 — CUDA BCE assertion error during aux loss computation

**Symptom:** `RuntimeError: CUDA error: device-side assert triggered` in the BCE aux loss during stageC training. Training crashed on GPU.

**Root cause:** `OccupancyHead` output logits were being passed to `torch.nn.functional.binary_cross_entropy` (not the `_with_logits` variant), which requires targets in [0,1] and inputs also in [0,1]. The logits (unbounded) triggered the assertion.

**Fix (committed `d951d73`):** Changed to `binary_cross_entropy_with_logits` throughout, which accepts unbounded logits. Also added `.clamp(0, 1)` on targets as a defensive guard.

**Status:** Fixed in `grace_network.py`.

---

## Ablation Training Checklist

Before starting any ablation run, verify all of these:

- [ ] **Branch:** `git branch` shows `* gram-map`
- [ ] **Config env matches stageC:** `human_num=20`, `circle_radius=8.5`, `realistic.enabled=True`, `num_on_path=2`, `group.types=['static_f','dynamic_lf','dynamic_free']`
- [ ] **Config policy:** `robot.policy = 'grace'`
- [ ] **Config freeze:** `grace.freeze_backbone = False` (ablations fine-tune end-to-end)
- [ ] **Config aux loss:** `grace.use_aux_loss = True` (except C4 which disables it via CLI flag)
- [ ] **Starting checkpoint:** `trained_models/gram_map/stageC/checkpoints/best.pt` exists
- [ ] **Entropy flag:** Use `--entropy-coef 0.005` (NOT 0.05 — logstd clamp is now in place)
- [ ] **Output dir:** Does NOT already exist (or use `--overwrite`)
- [ ] **GPU:** Check `nvidia-smi` for free VRAM (need ~4 GB for 16 processes)

---

## Ablation Commands (canonical, copy-paste ready)

All ablations resume from `stageC/best.pt`, full benchmark env (20 humans, realistic).

```bash
# ── Full GRACE (baseline, re-evaluate only — already trained) ─────────────
python test.py --model_dir trained_models/gram_map/stageC --test_model best.pt

# ── C1: No group cost layers (L3 + L4 zeroed) ────────────────────────────
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

# ── C2: K=1 slot (single group prototype instead of 3) ───────────────────
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C2_K1 \
    --num-env-steps 10000000 --num-processes 16 --num-steps 30 \
    --num-mini-batch 2 --ppo-epoch 5 \
    --lr 5e-5 --entropy-coef 0.005 --eps 1e-5 \
    --gamma 0.99 --gae-lambda 0.95 \
    --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 \
    --use-linear-lr-decay --save-interval 200 --log-interval 20 \
    --resume --load_path trained_models/gram_map/stageC/checkpoints/best.pt \
    --ablation_K_slots 1

# ── C3: No trajectory layers (L2 zeroed — no future prediction) ──────────
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C3_no_traj \
    --num-env-steps 10000000 --num-processes 16 --num-steps 30 \
    --num-mini-batch 2 --ppo-epoch 5 \
    --lr 5e-5 --entropy-coef 0.005 --eps 1e-5 \
    --gamma 0.99 --gae-lambda 0.95 \
    --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 \
    --use-linear-lr-decay --save-interval 200 --log-interval 20 \
    --resume --load_path trained_models/gram_map/stageC/checkpoints/best.pt \
    --ablation_no_traj_layers

# ── C4: No auxiliary occupancy loss ──────────────────────────────────────
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C4_no_aux \
    --num-env-steps 10000000 --num-processes 16 --num-steps 30 \
    --num-mini-batch 2 --ppo-epoch 5 \
    --lr 5e-5 --entropy-coef 0.005 --eps 1e-5 \
    --gamma 0.99 --gae-lambda 0.95 \
    --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 \
    --use-linear-lr-decay --save-interval 200 --log-interval 20 \
    --resume --load_path trained_models/gram_map/stageC/checkpoints/best.pt \
    --ablation_no_aux_loss

# ── C5: Uniform alpha (slot assignment replaced by uniform weights) ───────
python train.py --env-name CrowdSimVarNum-v0 \
    --human_node_rnn_size 256 --human_human_edge_rnn_size 14 \
    --output_dir trained_models/gram_map/ablation_C5_uniform \
    --num-env-steps 10000000 --num-processes 16 --num-steps 30 \
    --num-mini-batch 2 --ppo-epoch 5 \
    --lr 5e-5 --entropy-coef 0.005 --eps 1e-5 \
    --gamma 0.99 --gae-lambda 0.95 \
    --value-loss-coef 0.5 --clip-param 0.2 --max-grad-norm 0.5 \
    --use-linear-lr-decay --save-interval 200 --log-interval 20 \
    --resume --load_path trained_models/gram_map/stageC/checkpoints/best.pt \
    --ablation_uniform_alpha
```

## Evaluation Commands

After training completes, run `test.py` against each ablation's best checkpoint.
Identify best.pt from `progress.csv` (highest `eprewmean` among saved checkpoints).

```bash
python test.py --model_dir trained_models/gram_map/stageC                --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C1_no_group  --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C2_K1        --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C3_no_traj   --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C4_no_aux    --test_model best.pt
python test.py --model_dir trained_models/gram_map/ablation_C5_uniform   --test_model best.pt
```

Results go to `<model_dir>/test/test_best.pt.log`.

## Expected Ablation Results (what to look for)

| Ablation | Expected drop vs GRACE (SR=0.92) | Why |
|---|---|---|
| C1 No group layers | Large drop (−10 to −20 pp SR) | Group-aware channels are the core contribution |
| C2 K=1 slot | Moderate drop (−5 to −10 pp SR) | Single prototype can't distinguish overlapping groups |
| C3 No traj layers | Small-moderate drop (−3 to −8 pp SR) | Future prediction helps but individual layer still captures some info |
| C4 No aux loss | Small drop (−2 to −5 pp SR) | Aux loss stabilizes training but doesn't dominate test SR |
| C5 Uniform alpha | Moderate drop (−5 to −10 pp SR) | Learned slot assignments matter for group cohesion rendering |

If **C1 shows no drop** — suspect the cost map is not being used by the planner (check cost_stack values are non-zero). 
If **all ablations show no drop** — suspect the policy is ignoring the cost map entirely and relying only on the GRU hidden state.

## Warning Signs During Ablation Training

| Observation | Likely cause | Action |
|---|---|---|
| Mean reward collapses to −20 immediately | Config env mismatch (Issue 5), or wrong checkpoint | Check config vs stageC, verify checkpoint path |
| NaN in loss at startup | Backbone not loaded properly | Check `load_frozen_backbones` prints in train log |
| Entropy jumps to 8+ within 200 updates | `--entropy-coef` too high | Reduce to 0.001 or 0.0 |
| Entropy near 0 within 200 updates | logstd at clamp floor (−3) | Check optimizer; try `--entropy-coef 0.005` |
| SR never exceeds 40% after 5000 updates | Ablation is too destructive — the component is load-bearing | Report as "critical component" in paper |
| SR matches full GRACE (within 2 pp) | Component is redundant | Report as "ablated without loss" in paper |
