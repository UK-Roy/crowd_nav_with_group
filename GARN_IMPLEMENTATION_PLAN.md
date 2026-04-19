# GARN Implementation Plan

This document maps the GARN paper (Lu et al., IEEE RA-L 2025, `references/GARN.pdf`) to our
codebase and lays out a step-by-step plan for implementing it as a modular comparison baseline
for TAGA.

---

## 1. GARN Paper — Key Technical Details

### 1.1 High-Level Architecture (STGAN)

STGAN has three modules wired in series (see paper Fig. 3):

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| **Attention Extraction** | Produce per-agent attention weights | Agent feature vectors `x_i` | Attention matrix `A_t` |
| **Relation Modeling** | Capture spatio-temporal interactions via GCN + LSTM | `A_t`, agent states | Spatio-temporal relation features `C_t^L` |
| **Value Estimation** | Predict state value + policy (actor-critic) | `C_t^L` | Value `f_v`, action distribution `f_a` |

### 1.2 Attention Extraction Module

- Each agent's state is passed through **four independent MLPs** (paper Section IV-C-2,
  Fig. 4b). Each MLP outputs a feature vector of size `x_r`.
- Outputs are aggregated by element-wise product: `A_t = softmax(x_r^T W_r)`.
- `W_r` is a trainable weight matrix. The size of `W` depends only on the output feature
  vector length of the four MLPs, **not** on the number of agents `N` — so it scales to
  variable crowd sizes without retraining.
- The paper states the four MLPs map each agent to a fixed-length vector, which are then
  "appended to generate output features" of a fixed length, with a "similarity function"
  (Eq. 9) applied to extract attention weights.

**Paper dimensions (Section IV-D-2):** Hidden layer sizes for MLPs are
`(128), (128), (256), (256)` with output embedding dimension 128.

### 1.3 Relation Modeling Module (Spatio-Temporal GCN)

- **Spatial GCN layer (per timestep):**
  - Graph `(V_t, E_t)` where nodes = all agents (1 robot + K individuals + I group members + M groups).
  - Total node count `|V_t| = N = 1 + K + I + M`.
  - Edges `E_t` capture spatial relations; edge weight `e_{(i,j)}` is the attention from agent `i` to agent `j`.
  - Agent state is passed through an MLP `f_s`, then embedded into a latent space with a fixed length.
  - The latent matrix `E_t` is multiplied by attention matrix `A_t`, then fed into a GCN layer `f_g` which propagates information across nodes and produces spatial relation feature matrix `C_t^l`.
  - Layer-wise update rule (Eq. 10):
    `C^(l+1) = σ(A_t · C^l · W_g)`
    where `σ` is ReLU, `A_t` is the attention, `C^l` is input features, `W_g` is a learnable weight.

- **Temporal LSTM:**
  - After spatial GCN, the features are fed into an LSTM to capture temporal dependencies.
  - This encodes the temporal dependence of spatial relations with long short-term memory.
  - The LSTM hidden size = 128 (paper Section IV-D-2).

### 1.4 Value Estimation Module

- The output of the relation modeling (spatio-temporal features for the robot node, after
  `S` GCN layers) is fed into an MLP that serves as both actor (policy) and critic (value).
- This is the standard actor-critic structure used with PPO.

### 1.5 Agent and Group State Representation

**Individual pedestrian state** (Eq. in Section IV-A):

Each pedestrian `i` has two observation types: circular and rectangular.
- State: `P^i = {p^i, v^i, θ^i, r^i}` (position, velocity, orientation, radius)
  - For circular obstacle: position `p^i`, radius `r^i`
  - For rectangular obstacle: position `p^i`, orientation `θ^i`, width `w^i`, height `h^i`
- The robot's state at time `t`: `s_t = {p, v, θ, r, g}` (position, velocity, orientation, radius, goal)

**Group state** (Eq. 1 in Section IV-A):

- Group space `GP_m^{iv}` = ConvexHull({`P^i | i ∈ G·P_m^{iv}`}) — convex hull of personal spaces of group members.
- Group state `GP_m^{iv} = {p_g^m, v_m^{iv}, GP_m^{iv}, n_m^{iv}}` where:
  - `p_g^m` = gravity center (centroid) of group members
  - `v_m^{iv}` = mean velocity of group members
  - `GP_m^{iv}` = convex hull boundary
  - `n_m^{iv}` = number of group members

### 1.6 Reward Function (Eq. 3-8)

The total reward has four components (Eq. 3):

```
R(J_t, a_t) = R_gl + R_obs + R_prox + R_grp
```

Where:
- `R_gl` = goal reward (standard — success reward on reaching goal)
- `R_obs` = obstacle reward (standard — collision penalty)
- `R_prox` = proximity reward (standard — discomfort penalty when too close to individuals)
- `R_grp` = **novel group-related reward** (the GARN contribution)

**Group reward `R_grp` (Eq. 4)** has three sub-components:

```
R_grp = R_grp^intra + P_grp^{o/fw} + R_grp^coop
```

**(a) Group intrusion penalty `R_grp^intra` (Eq. 5):**

```
R_grp^intra = Σ_{m∈M} {-0.25 × 1_{d_m}}
```

- `d_m` = shortest distance from robot to convex hull of group `m` (as shown in Fig. 2)
- `1_{d_m}` = binary indicator: 1 if robot is inside the group's convex hull, 0 otherwise
- Penalty of -0.25 per group intruded per timestep

**(b) Overtaking/following penalty `R_grp^{o/fw}` (Eq. 6-7):**

```
R_grp^{o/fw} = Σ_{m∈M_1} 0_{d_o} · c_1 · p^v · [Δ(p_{t-1}, p_m^v) - Δ(p_t, p_m^v)] / |v_t|  ×  v_i / |v_i|
```

Where:
- `M_1` = set of groups walking in the same direction as the robot while being positioned in front of it
- `0_{d_o}` = logical negation of `1_{d_m}` (only active when robot is NOT inside the group)
- `c_1` = hyperparameter for weighting lagging behind and successful overtaking
- `p^v` = dot product with `v_t` projects the displacement difference onto the direction of the robot's velocity
- `g^v` = binary function
- `d_{t1}` = consideration range for surrounding groups
- Condition (Eq. 7): `||Δ(p_t, p_m^v)|| ≤ d_{t1}` — robot must be within consideration distance

This incentivizes overtaking without collision and intrusion while penalizing distant following.

**(c) Cooperative passing reward `R_grp^coop` (Eq. 8):**

```
R_grp^coop = Σ_{m∈M_2} 0_{d_o} · φ^m · [Δ(p_t, p_m^v) - Δ(p_{t-1}, p_m^v)] / |v_{pref}|  ×  v_i / |v_i|
```

Where:
- `M_2` = set of groups approaching the robot from its front
- `c_2` = hyperparameter for weighting cooperative passing
- Encourages cooperative passing with a group rather than waiting when encountering it

### 1.7 Training Details (Section IV-D)

- **Algorithm:** Model-free DRL with PPO (Double deep Q-learning also mentioned for action value estimation)
- **Training framework:** Adam optimizer, learning rate 0.0005, discount factor γ = 0.99
- **Attention extraction:** Pre-trained using model-free DRL (supervising attention weights with model-free value function)
- **Training episodes:** 200 episodes, 500 episodes per evaluation
- **Simulation:** 12m × 12m space, up to 20 dynamic humans, 5m sensor range
- **Initial states:** Robot starts at one side, goal on the opposite side (circle crossing)
- **Groups:** Static groups (2-3 members), dynamic groups (3-4 members) with leader-follower dynamics
- **Time limit:** 197 steps (Section IV-D-2, but this may be environment-specific)
- **GCN layers:** The paper mentions "LSTM hidden" of size 128
- **Evaluation scenarios:**
  - S1: 6 individuals, 2 groups (2 and 3 members), 4 dynamic groups (2, 3, and 4 members)
  - S2: 4 individuals, 2 groups (2 and 3 members), 3 dynamic groups (3, 3, and 4 members)

---

## 2. Mapping GARN → Our Codebase

### 2.1 What We Can Reuse Directly

| GARN Component | Our Codebase Equivalent | Notes |
|----------------|------------------------|-------|
| Simulation environment (12m×12m, 20 humans, groups) | `crowd_sim/envs/crowd_sim_var_num.py` + `crowd_sim_pred_real_gst.py` | Already has groups, centroids, radii |
| Group detection (ground truth labels) | `crowd_sim_var_num.py` `generate_ob()` — `cluster_dict` from `group_ground_truth` | Already implemented |
| Convex hull of group members | `crowd_sim_var_num.py` line 443-444 — uses max distance from centroid | Need to upgrade to actual ConvexHull |
| Observation: `robot_node`, `temporal_edges`, `spatial_edges` | Already in obs space | Same format |
| Observation: `clusters`, `group_centroids`, `group_radii`, `velocity_edges` | Already in `CrowdSimPredRealGST` obs space | GARN can use these |
| PPO training pipeline | `train.py` + `rl/ppo/ppo.py` | Reuse as-is |
| Rollout storage | `rl/networks/storage.py` | Reuse as-is |
| Environment vectorization | `rl/networks/envs.py` | Reuse as-is |
| Policy base class | `crowd_nav/policy/policy.py` → `crowd_nav/policy/srnn.py` | Same pattern |
| Actor-critic wrapper | `rl/networks/model.py` (`Policy` class) | Add GARN as another `base` option |

### 2.2 What We Must Build New

| GARN Component | New File | Description |
|----------------|----------|-------------|
| STGAN network (attention extraction + GCN + LSTM + value head) | `rl/networks/stgan_model.py` | The full GARN neural network |
| GARN policy wrapper | `crowd_nav/policy/garn.py` | Policy class (like `srnn.py`) with `name='garn'` |
| Group-related reward function `R_grp` | `crowd_sim/envs/garn_reward.py` | Modular reward calculator, called from env |
| Convex hull group representation | Inside `garn_reward.py` | `scipy.spatial.ConvexHull` for `1_{d_m}` check |

### 2.3 Existing Files That Need Minimal Modification

| File | Change | Scope |
|------|--------|-------|
| `crowd_nav/policy/policy_factory.py` | Add `policy_factory['garn'] = GARN` import + registration | 2 lines |
| `rl/networks/model.py` | Add `elif base == 'garn': base = STGAN` import + branch | 3 lines |
| `crowd_nav/configs/config.py` | Add `garn = BaseConfig()` section with GARN-specific hyperparams | ~10 lines |
| `arguments.py` | Add GARN-specific args (GCN layers, LSTM hidden size, etc.) | ~10 lines |
| `crowd_sim/envs/crowd_sim_var_num.py` | Call `garn_reward.calc_garn_reward()` when `config.reward.use_garn_reward` is True — **additive**, does not touch existing reward path | ~5 lines in `calc_reward()` |

**Total lines changed in existing files: ~30 lines. No TAGA or GRAM code is touched.**

---

## 3. New Files — Detailed Design

### 3.1 `rl/networks/stgan_model.py` — The STGAN Network

```
class AttentionExtraction(nn.Module):
    """Four parallel MLPs that produce per-agent attention weights (paper Section IV-C-2)"""
    - Input: agent features [batch, N, feat_dim]
    - Four MLPs: each maps feat_dim → hidden → x_r
    - Aggregate: element-wise product of four outputs
    - Attention: softmax(x_r^T · W_r) → attention matrix A_t [batch, N, N]

class SpatioTemporalGCN(nn.Module):
    """Spatial GCN layers + temporal LSTM (paper Section IV-C-1)"""
    - Input: agent states, attention matrix A_t
    - Spatial: MLP embed → GCN propagation: C^{l+1} = ReLU(A_t · C^l · W_g)
    - Temporal: LSTM over the sequence of spatial GCN outputs
    - Output: robot's spatio-temporal feature vector

class STGAN(nn.Module):
    """Full STGAN network — the nn.Module that model.py's Policy wraps"""
    - __init__(self, obs_space_dict, args):
        - self.attention_extraction = AttentionExtraction(args)
        - self.relation_modeling = SpatioTemporalGCN(args)
        - self.actor = nn.Sequential(...)  # Same pattern as GRAM
        - self.critic = nn.Sequential(...)
        - self.critic_linear = nn.Linear(hidden, 1)
        - self.is_recurrent = True
        - self.output_size = args.human_node_output_size
    - forward(self, inputs, rnn_hxs, masks, infer=False):
        - Extract obs: robot_node, spatial_edges, velocity_edges, clusters,
          group_centroids, group_radii, detected_human_num
        - Build agent feature matrix (robot + visible humans + group nodes)
        - Run attention extraction → A_t
        - Run spatio-temporal GCN → robot feature
        - Actor head → action features
        - Critic head → value
        - Return (value, actor_features, rnn_hxs)
    - @property recurrent_hidden_state_size: returns LSTM hidden size

Key dimensions (from paper Section IV-D-2):
    - MLP hidden sizes: (128), (128), (256), (256)
    - Embedding dimension: 128
    - GCN hidden size: 128
    - LSTM hidden size: 128
    - Actor/critic hidden: 256 (matching our existing pattern)
    - Output size: 256 (matching args.human_node_output_size)
```

### 3.2 `crowd_nav/policy/garn.py` — GARN Policy Class

```python
from crowd_nav.policy.srnn import SRNN

class GARN(SRNN):
    """Policy class for GARN — follows same pattern as selfAttn_merge_SRNN_GrpAttn"""
    def __init__(self, config):
        super().__init__(config)
        self.name = 'garn'
        self.trainable = True
        self.multiagent_training = True
```

This is minimal — the network architecture lives in `stgan_model.py`, the policy class
just provides the name and clip_action behavior (inherited from SRNN).

### 3.3 `crowd_sim/envs/garn_reward.py` — Group Reward Calculator

```python
def calc_garn_group_reward(robot, humans, groups, group_centroids, group_radii, config):
    """
    Calculate R_grp = R_grp^intra + R_grp^{o/fw} + R_grp^coop
    Called from crowd_sim_var_num.py's calc_reward() when use_garn_reward=True.

    Returns: float reward value (to be added to existing R_gl + R_obs + R_prox)
    """
    r_intra = calc_intrusion_penalty(robot, groups, group_centroids)
    r_ofw = calc_overtaking_following(robot, groups, group_centroids, config)
    r_coop = calc_cooperative_passing(robot, groups, group_centroids, config)
    return r_intra + r_ofw + r_coop

def calc_intrusion_penalty(robot, groups, group_centroids):
    """Eq. 5: -0.25 per group whose convex hull the robot is inside"""
    # Use scipy.spatial.ConvexHull + point-in-hull test
    ...

def calc_overtaking_following(robot, groups, group_centroids, config):
    """Eq. 6-7: Penalize/reward overtaking and following of same-direction groups"""
    ...

def calc_cooperative_passing(robot, groups, group_centroids, config):
    """Eq. 8: Reward cooperative passing with approaching groups"""
    ...

def point_in_convex_hull(point, hull_points):
    """Check if a 2D point is inside the convex hull of hull_points"""
    # Use scipy.spatial.Delaunay or ConvexHull
    ...
```

---

## 4. Step-by-Step Implementation Order

### Phase 1: Scaffolding (no training yet)

1. **Create `crowd_nav/policy/garn.py`** — minimal policy class
2. **Register in `policy_factory.py`** — add import and `policy_factory['garn'] = GARN`
3. **Add GARN args to `arguments.py`** — GCN layers, LSTM hidden, attention MLP sizes, `--use_garn_reward`
4. **Add GARN config section to `crowd_nav/configs/config.py`** — hyperparameters for the reward function (`c_1`, `c_2`, `d_t1`, intrusion penalty weight)
5. **Create `rl/networks/stgan_model.py`** — stub with correct `__init__` signature and `forward()` that returns dummy tensors in the right shapes
6. **Wire into `rl/networks/model.py`** — add the `elif base == 'garn'` branch
7. **Verify:** Set `robot.policy = 'garn'` in config, run `python train.py` — it should start (even if the network outputs garbage)

### Phase 1.5: Sanity Check (before real network implementation)

8. **Verify GPU/CUDA setup** — run the GPU check command from Section 7.1 to confirm PyTorch sees the GPU
9. **Train the dummy stub network for ~1000 steps** — run `python train.py --num-env-steps 50000 --num-processes 2` with `robot.policy = 'garn'`. This uses the stub STGAN from step 5 (which outputs random tensors in the correct shapes). The goal is to confirm the entire end-to-end loop works: env creation → obs dict → network forward → PPO update → checkpoint save. Fix any shape mismatches, missing dict keys, or device errors before proceeding.
10. **Check output** — verify that `trained_models/GARN/progress.csv` is created and contains rows, and that a checkpoint `.pt` file is saved. The reward values will be garbage (random policy) — that's expected.

### Phase 2: STGAN Network

11. **Implement `AttentionExtraction`** — four MLPs + attention weight computation
12. **Implement `SpatioTemporalGCN`** — GCN layer + LSTM temporal encoder
13. **Implement full `STGAN.forward()`** — wire everything together, handle variable human counts, handle group nodes
14. **Verify:** Run a short training (~1000 steps), check that loss decreases and shapes are correct

### Phase 3: Group Reward

12. **Create `crowd_sim/envs/garn_reward.py`** — implement all three reward components
13. **Add convex hull computation** — `scipy.spatial.ConvexHull` for point-in-hull test
14. **Integrate into `crowd_sim_var_num.py`** — add conditional call in `calc_reward()` gated by `config.reward.use_garn_reward`
15. **Verify:** Run a few episodes, print reward breakdown, sanity-check that `R_grp^intra` fires when robot is inside a group

### Phase 4: Training & Tuning

16. **Train GARN** — full 20M steps (or paper's recommended episode count)
17. **Monitor:** Check `progress.csv` for reward curves, verify convergence
18. **Tune hyperparameters** if needed (`c_1`, `c_2`, learning rate, GCN depth)

### Phase 5: Evaluation & Comparison

19. **Test GARN** — run `test.py` with `--model_dir trained_models/GARN`
20. **Compare vs TAGA** — run identical test scenarios for GARN, TAGA+ORCA, TAGA+AG-RL, ORCA, DS-RNN
21. **Compute GCR** — use the same Group Collision Rate metric from TAGA paper
22. **Generate comparison tables and plots**

---

## 5. Activation Mechanism

GARN will be activated the same way as any other policy — via the config file:

```python
# In crowd_nav/configs/config.py:
robot.policy = 'garn'

# GARN-specific reward:
reward.use_garn_reward = True

# GARN hyperparameters:
garn = BaseConfig()
garn.intrusion_penalty = -0.25      # Eq. 5: penalty per group intrusion
garn.c1 = 1.0                       # Eq. 6: overtaking/following weight
garn.c2 = 1.0                       # Eq. 8: cooperative passing weight
garn.d_t1 = 3.0                     # Eq. 7: consideration range for groups
garn.gcn_layers = 2                 # Number of GCN layers
garn.lstm_hidden = 128              # LSTM hidden size
garn.attn_mlp_hidden = [128, 128, 256, 256]  # Four MLP hidden sizes
```

For testing, no special flag is needed — just point `--model_dir` at the trained GARN model directory (which contains the saved config).

---

## 6. Uncertainties & Open Questions

These are ambiguities in the GARN paper that may require experimentation or clarification:

### Architecture Questions

1. **Attention Extraction — exact aggregation method:**
   The paper says four MLPs are used and their outputs are "appended to generate output features" with a "similarity function" applied (Eq. 9). The exact aggregation (concatenation? element-wise product? sum?) is not fully specified. The softmax attention formula `A_t = softmax(x_r^T W_r)` suggests `x_r` is already a single vector per agent. **Plan:** Start with concatenation of four MLP outputs → linear projection → softmax, which is the most standard interpretation.

2. **GCN input feature construction:**
   How exactly robot, individual humans, and groups are all represented as nodes in the same graph is somewhat implicit. The paper says `|V_t| = N = 1 + K + I + M` (robot + K individuals + I group-member individuals + M groups), meaning groups are separate nodes alongside their constituent members. **Plan:** Construct the node feature matrix with robot state, per-human states, and per-group aggregate states (centroid, mean velocity, radius, member count) all as rows.

3. **Number of GCN layers:**
   Not explicitly stated. The paper mentions "layer-wise update" (Eq. 10) and `C^(l+1)` notation suggesting multiple layers, but does not specify the depth. **Plan:** Default to 2 GCN layers, make it configurable.

4. **Weight matrix `W` scalability claim:**
   The paper emphasizes that `W`'s size doesn't depend on `N` (number of agents). This works because attention is computed per-agent and the GCN weight `W_g` is shared across all nodes. This is consistent with standard GCN — no issue here.

### Reward Questions

5. **`R_grp^{o/fw}` and `R_grp^coop` — exact velocity/displacement computation:**
   The notation in Eqs. 6-8 is somewhat dense. The terms `Δ(p_t, p_m^v)` appear to mean the displacement between robot position and group centroid projected onto velocity direction. **Plan:** Implement as: (a) compute vector from robot to group centroid, (b) project onto robot velocity direction, (c) compare between timesteps to get progress. Test with simple scenarios to validate sign/direction.

6. **Hyperparameter values for `c_1` and `c_2`:**
   Not given in the paper. **Plan:** Start with `c_1 = c_2 = 1.0`, tune via ablation.

7. **`d_t1` consideration distance:**
   The threshold for when overtaking/following reward kicks in. Not given. **Plan:** Start with `d_t1 = 3.0` meters (slightly larger than typical group radius + robot sensor range consideration).

8. **Convex hull for small groups (2 members):**
   ConvexHull is degenerate for collinear points. For 2-member groups, the "hull" is a line segment. **Plan:** Use a buffered convex hull (expand by personal space radius) or fall back to circle representation for groups with ≤ 2 members.

### Training Questions

9. **"Pre-trained attention extraction" (Section IV-D-1):**
   The paper says "the attention extraction subnetwork was pre-trained using supervised learning weights from the model-free DRL." This implies a two-phase training: (a) train base DRL without group reward, (b) use the learned value function to supervise attention weights, (c) then train full STGAN with group reward. **Plan:** Start with end-to-end training (simpler). **Decision criterion:** If after 5M training steps the success rate is below 50%, pause and implement the two-phase pre-training approach as described in the paper — first train without `R_grp` to get a working base policy, then freeze the base and supervise the attention extraction module, then fine-tune the full STGAN with `R_grp` enabled.

10. **Learning rate discrepancy:**
    The paper says 0.0005 (Section IV-D-2). Our codebase default is 4e-5. **Plan:** Use the paper's 0.0005 for GARN training, keep existing LR for other policies.

11. **Episode count vs step count:**
    The paper trains for "200 episodes" which seems very low. Our codebase trains for 20M env steps. This likely means 200 *updates* or the paper's episodes are much longer. **Plan:** Train for the same total env steps (20M) as our other policies for fair comparison.

---

## 7. Training and Testing Guide

> **IMPORTANT: Run the GPU check command below FIRST before any training.**
> If CUDA is unavailable, reduce `--num-env-steps` to 5M and use `--num-processes 4`.
> CPU training for full 20M-step runs is **not recommended** — it will take days.
> CPU is only suitable for short sanity checks and debugging.

### 7.1 Check GPU Availability (run this first!)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If this prints `CUDA available: False`, you must pass `--no-cuda` to all training commands and reduce the step count.

### 7.2 Train GARN on GPU

Before training, set the following in `crowd_nav/configs/config.py`:
```python
robot.policy = 'garn'
reward.use_garn_reward = True
```

And in `arguments.py`:
```python
parser.add_argument('--output_dir', type=str, default='trained_models/GARN')
parser.add_argument('--lr', type=float, default=5e-4)  # GARN paper: 0.0005
```

Then run:

```bash
# Single GPU (default)
python train.py

# Multi-GPU (if available — uses nn.DataParallel, already in train.py)
# Just run the same command; DataParallel is already applied in train.py line 114
python train.py
```

The model will save checkpoints to `trained_models/GARN/checkpoints/` every 200 updates.

### 7.3 Train GARN on CPU (fallback)

```bash
python train.py --no-cuda --num-processes 4 --num-env-steps 5000000
```

Reduce `--num-processes` (parallel envs) and `--num-env-steps` since CPU training is much slower.

### 7.4 Estimated Training Time

| Hardware | Processes | Steps | Estimated Time |
|----------|-----------|-------|----------------|
| Single GPU (RTX 3060+) | 16 | 20M | ~8-12 hours |
| Single GPU (RTX 3060+) | 16 | 5M | ~2-3 hours |
| CPU only | 4 | 5M | ~24-48 hours |
| CPU only | 4 | 20M | ~4-8 days |

These are rough estimates based on existing GRAM training times in this codebase. GARN's GCN may be slightly slower per step due to graph construction, but the bottleneck is usually the simulation.

### 7.5 Test Trained GARN Model

```bash
# Run 500 test episodes (no visualization)
python test.py --model_dir trained_models/GARN --test_model <checkpoint>.pt

# Run with visualization
python test.py --model_dir trained_models/GARN --test_model <checkpoint>.pt --visualize

# Run a specific test case
python test.py --model_dir trained_models/GARN --test_model <checkpoint>.pt --test_case 42 --visualize

# Run with trajectory rendering
python test.py --model_dir trained_models/GARN --test_model <checkpoint>.pt --render_traj
```

### 7.6 Compare GARN vs TAGA vs Other Baselines

Run each method on the same test scenarios and compare logs:

```bash
# 1. GARN (learned group-aware policy)
python test.py --model_dir trained_models/GARN --test_model <best>.pt

# 2. TAGA + ORCA (reactive group avoidance on top of ORCA)
python test.py --model_dir trained_models/ORCA_no_rand --test_model 00000.pt --group_avoid

# 3. TAGA + AG-RL (reactive group avoidance on top of learned policy)
python test.py --model_dir trained_models/GST_predictor_rand --test_model 41665.pt --group_avoid

# 4. Plain ORCA (no group awareness)
python test.py --model_dir trained_models/ORCA_no_rand --test_model 00000.pt

# 5. Plain AG-RL / GRAM (no group awareness beyond what was trained)
python test.py --model_dir trained_models/GST_predictor_rand --test_model 41665.pt

# 6. DS-RNN (if available)
# python test.py --model_dir trained_models/DS_RNN --test_model <best>.pt
```

Results are saved to `<model_dir>/test/test_<checkpoint>.log`. Compare:
- Success Rate (SR)
- Collision Rate (CR)
- Timeout Rate (TR)
- Group Collision Rate (GCR)
- Navigation Time (NT)
- Path Length (PL)

### 7.7 Plot Training Curves

```bash
# After training, plot progress from the CSV log
python plot.py
```

Edit `plot.py` to point at `trained_models/GARN/progress.csv` if needed. The CSV contains columns: `misc/nupdates`, `misc/total_timesteps`, `fps`, `eprewmean`, `loss/policy_entropy`, `loss/policy_loss`, `loss/value_loss`.

---

## 8. File Summary

### New Files (4)

| File | Purpose | Approx Lines |
|------|---------|-------------|
| `rl/networks/stgan_model.py` | STGAN neural network (attention + GCN + LSTM + actor-critic) | ~300-400 |
| `crowd_nav/policy/garn.py` | GARN policy class (name registration, clip_action) | ~15 |
| `crowd_sim/envs/garn_reward.py` | Group-related reward R_grp (intrusion + overtaking + cooperative) | ~150 |
| `GARN_IMPLEMENTATION_PLAN.md` | This document | — |

### Modified Files (4, minimal changes)

| File | Change | Lines Added |
|------|--------|------------|
| `crowd_nav/policy/policy_factory.py` | Import GARN + register | ~2 |
| `rl/networks/model.py` | Add `elif base == 'garn'` | ~3 |
| `crowd_nav/configs/config.py` | Add `garn` config section + `reward.use_garn_reward` | ~12 |
| `arguments.py` | Add GARN-specific network args | ~10 |
| `crowd_sim/envs/crowd_sim_var_num.py` | Conditional call to `garn_reward` in `calc_reward()` | ~5 |

### Untouched

- All TAGA code (`crowd_nav/policy/taga_safety.py`, test.py `--group_avoid` flag, etc.)
- All GRAM code (`rl/networks/selfAttn_srnn_temp_node_groupAttn.py`, etc.)
- Training pipeline (`train.py`, `rl/ppo/ppo.py`, `rl/networks/storage.py`)
- Environment registration (`crowd_sim/__init__.py`)
- Evaluation (`rl/evaluation.py`)
