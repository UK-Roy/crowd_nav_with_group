# GRACE Ablation Study Results

**Branch:** `gram-map`
**Baseline:** `trained_models/gram_map/stageC/checkpoints/best.pt`
**Test episodes:** 100 per model
**Environment:** 20 humans, 3 groups, realistic phases A–E, `CrowdSimVarNum-v0`

---

## How to read results from the log

After running `test.py`, open the log file:
```bash
cat trained_models/gram_map/ablation_C1_no_group/test/test_best.pt.log
```
Find this line:
```
Testing success rate: X.XX, collision rate: X.XX, timeout rate: X.XX,
group intrusion rate (GCR): X.XX%, nav time: X.XX, path length: X.XX, ...
```
Also note `Mean Reward: X.XX` from the last line.

---

## Results Table

| # | Model | SR | CR | TR | GCR | Mean Reward | Best Checkpoint |
|---|---|---|---|---|---|---|---|
| — | **GRACE full** (baseline) | **0.92** | **0.05** | **0.03** | **0.00%** | **32.37** | stageC/41000.pt |
| C1 | No group layers (L3+L4 zeroed) | | | | | | |
| C2 | K=1 slot (single group prototype) | | | | | | |
| C3 | No trajectory layers (L2 zeroed) | | | | | | |
| C4 | No auxiliary occupancy loss | | | | | | |
| C5 | Uniform slot assignment (alpha) | | | | | | |

**SR** = Success Rate (higher is better, target ≥ 0.92 for full GRACE)
**CR** = Collision Rate (lower is better)
**TR** = Timeout Rate (lower is better)
**GCR** = Group Crossing Rate — % of steps inside a group hull (lower = less intrusive)
SR + CR + TR ≈ 1.0 is a sanity check.

---

## How to find best.pt for each ablation

After training finishes, run this to find the best checkpoint:
```bash
python3 -c "
import csv, os
stage = 'trained_models/gram_map/ablation_C1_no_group'  # change this
rows = []
with open(f'{stage}/progress.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((float(row['eprewmean']), int(row['misc/nupdates'])))
rows.sort(reverse=True)
avail = set(int(f.replace('.pt','')) for f in os.listdir(f'{stage}/checkpoints/') if f.endswith('.pt'))
print('Best available:')
for rew, upd in rows:
    if upd in avail:
        print(f'  update={upd}  reward={rew:.2f}')
        break
"
```
Then copy it to `best.pt`:
```bash
cp trained_models/gram_map/ablation_C1_no_group/checkpoints/<update>.pt \
   trained_models/gram_map/ablation_C1_no_group/checkpoints/best.pt
```
Then evaluate:
```bash
python test.py --model_dir trained_models/gram_map/ablation_C1_no_group --test_model best.pt
```

---

## Training Commands (copy-paste)

### C1 — No group layers
```bash
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
```

### C2 — K=1 slot
```bash
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
```

### C3 — No trajectory layers
```bash
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
```

### C4 — No auxiliary loss
```bash
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
```

### C5 — Uniform slot assignment
```bash
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

---

## Expected Result Ranges

| Ablation | Expected SR drop | Interpretation if drop is large |
|---|---|---|
| C1 No group layers | −10 to −20 pp | Group cost map is the **core contribution** — critical |
| C2 K=1 slot | −5 to −10 pp | Multiple group prototypes matter for overlapping groups |
| C3 No traj layers | −3 to −8 pp | Future prediction helps but individual layer partially compensates |
| C4 No aux loss | −2 to −5 pp | Aux loss stabilises training but is not the main driver |
| C5 Uniform alpha | −5 to −10 pp | Learned slot assignment matters for correct group rendering |

**pp** = percentage points (e.g. SR 0.92 → 0.80 = −12 pp drop)

---

## Warning Signs During Training

| What you see in progress.csv | Meaning | Action |
|---|---|---|
| Mean reward stays at −20 for 500+ updates | Wrong checkpoint or config mismatch | Stop, re-check config and `--load_path` |
| `loss/policy_entropy` jumps above 5.0 | `entropy-coef` too high | Reduce to 0.001 |
| `loss/policy_entropy` drops below 0.5 | logstd at clamp floor | Fine — clamp prevents collapse, keep training |
| NaN in `loss/value_loss` or `loss/policy_loss` | Backbone NaN (should be guarded) | Check `nan_to_num` guards exist in grace_network.py |
| Mean reward peaks then slowly declines | LR too high or overfitting | Roll back to best checkpoint, halve LR |

---

## Progress Tracking

| Ablation | Training Started | Training Done | best.pt copied | test.py run | Results recorded |
|---|---|---|---|---|---|
| C1 No group | | | | | |
| C2 K=1 slot | | | | | |
| C3 No traj | | | | | |
| C4 No aux | | | | | |
| C5 Uniform | | | | | |
