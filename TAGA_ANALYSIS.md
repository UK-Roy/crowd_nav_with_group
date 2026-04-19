# TAGA Implementation Analysis

Working document capturing the current state of TAGA (Tangent Action for Group Avoidance) in this codebase, to guide the RA-L revision.

## 1. What TAGA Is (as implemented today)

TAGA is a **runtime, evaluation-only** reactive module layered on top of any base navigation policy. It activates when a detected human group sits between the robot and its goal, and steers the robot tangent to the group boundary until the group is no longer blocking.

**Two concerns, two modules:**

| Module | Purpose | Source |
|---|---|---|
| Tangent action selector | Computes an avoidance velocity perpendicular to the robot→group-centroid ray | `rl/evaluation.py:129-169` + helpers `find_perpendi`, `angle_between_vectors` |
| Safety controller | Prevents individual-human collisions while TAGA is active, by blending/overriding with repulsive forces | `crowd_nav/policy/taga_safety.py` |

Both are invoked **only at test time** (`rl/evaluation.py`) behind the `--group_avoid` CLI flag. Training is unaffected; TAGA never changes the base policy's learned weights.

---

## 2. The Tangent Action Selector (`rl/evaluation.py:99-169`)

Per-step pseudocode:

```
if obs['grp']:                              # at least one group detected
    build group_dict from obs['clusters']
    for each (centroid, radius) in detected_groups:
        robot_to_goal   = goal - robot
        robot_to_group  = centroid - robot
        d_robot_goal    = |robot_to_goal|
        d_centroid_goal = |centroid - goal|

        if d_robot_goal < d_centroid_goal AND d_robot_goal < 3:
            break                           # goal-priority shortcut

        if dot(robot_to_goal, robot_to_group) > 0:     # group in front
            if |robot_to_group| < radius + 0.25:       # close enough
                if d_robot_goal < d_centroid_goal:
                    break                   # goal still closer

                cw  = perpendicular(robot→centroid, clockwise)
                ccw = perpendicular(robot→centroid, counter-clockwise)

                pick whichever makes a smaller angle with robot_to_goal
                taga_action = chosen perpendicular      # unit vector

                act = safety_controller.get_safe_taga_action(obs, taga_action)
                break                       # only adjust for the nearest group
```

**Key properties / quirks:**

- **Tangent direction** is a 90° rotation of the robot→centroid vector. The implementation normalises to unit length and lets downstream code scale by `v_pref`.
- **Side selection** uses the smaller of the two tangent-vs-goal angles (greedy, not trajectory-optimal).
- **Goal priority** fires twice — once before the dot-product check and once inside the if-branch — which is defensive but redundant.
- **`break` on first match**: TAGA only considers the **nearest group in iteration order**, not the most blocking one. Iteration order comes from the dict `detected_groups.items()`, which is insertion-ordered (Python 3.7+) — tied to group ID, not proximity or threat.
- **Switching distance** is implicit: `|robot→group| < radius + 0.25`. The 0.25 m `safe_margin` is hard-coded in `evaluation.py:126`; not exposed in config.
- **Goal threshold** `goal_threshold = 3` (meters) is also hard-coded (`evaluation.py:125`).

---

## 3. The Safety Controller (`crowd_nav/policy/taga_safety.py`)

Three-zone graduated policy over the distance to the **nearest individual human**:

| Zone | Threshold (m) | Behaviour |
|---|---|---|
| EMERGENCY | `< 0.4` | **Override** TAGA: velocity points directly away from the closest human, at `v_pref` |
| DANGER | `0.4 – 0.6` | Blend: `0.3 · TAGA + 0.7 · safety_force` |
| CAUTION | `0.6 – 1.0` | Blend: `0.8 · TAGA + 0.2 · safety_force` |
| SAFE | `> 1.0` | Pure TAGA |

`compute_safety_action` builds a repulsive force from **every** visible human inside 1.0 m:

```
force_i = (CAUTION_ZONE - d_i) / CAUTION_ZONE · (-r̂_i)    # per human, magnitude ∈ [0,1], times 2.0
repulsive = Σ force_i
goal_dir  = unit(goal - robot)

if min_dist < DANGER:    combined = 0.7 · repulsive + 0.3 · goal_dir
else:                    combined = 0.4 · repulsive + 0.6 · goal_dir
```

`combined` is then normalised to `v_pref`.

**⚠ Divergence from paper:**
- IROS/RA-L draft cites `d_critical = 0.3 m`, `d_personal = 0.5 m`; code uses `0.4 / 0.6 / 1.0`. Three zones in the code vs. the paper's two explicit zones.
- Emergency response uses the **closest human alone**, not an aggregate — so two humans at equal distance cause direction flipping.

---

## 4. Group Detection — What's Actually Running

**Claim (CLAUDE.md, paper):** DBSCAN at runtime, ε = 1.5 m, `min_size = 2`.

**Reality (`crowd_sim/envs/crowd_sim_var_num.py:296-302`, `config.py:67`):**

```python
group.ground_truth = True                   # config.py
...
if self.group_ground_truth:                 # always True today
    group_id = self.humans[i].group_id      # ground-truth label from sim
    cluster_dict[group_id].append(...)
```

The DBSCAN block exists but is **fully commented out** (`crowd_sim_var_num.py:340-381`). Every evaluation today uses **simulator-provided group labels**. This is exactly what IROS reviewer R3 flagged, and the current implementation does not yet address that concern.

---

## 5. GCR Metric Computation (`rl/evaluation.py:199-261`)

Per episode:

```
episode_group_intrusions += 1   each step with info == GroupIntrusion
episode_intrusion_ratio = episode_group_intrusions / stepCounter * 100
```

Aggregated:

```
gcr_rate = mean(episode_intrusion_ratios)   # % of timesteps inside some group
```

- GCR is reported **separately** from SR/CR/TR (not summed into a probability). `assert success + collision + timeout == test_size` confirms GCR is decoupled. ✅ (matches the revised definition.)
- `GroupIntrusion` info is emitted by the env during `step` — so the semantics are "was the robot inside any group's bounding radius at this step".
- A commented-out "global GCR" variant using `total_time_in_groups / total_steps` remains in the file — not active.

---

## 6. Configuration Surface

| Knob | Location | Current Value | Used by |
|---|---|---|---|
| `group.ground_truth` | `config.py:67` | `True` | crowd_sim group detection |
| `group.num_groups` | `config.py:56` | `2` | sim init |
| `group.min_size` / `max_size` | `config.py:58-59` | `3 / 4` | sim init |
| `group.min_radius` / `max_radius` | `config.py:63-64` | `1.0 / 1.3` | sim init |
| `group.dynamic` | `config.py:66` | `False` | group motion |
| `reward.group_safety_buffer` | `config.py:38` | `0.1` | added to bounding radius |
| `reward.discomfort_group_dist` | `config.py:39` | `0.07` | group proximity penalty |
| `reward.grp_collision_penalty` | `config.py:42` | `-21` | env reward |
| `--group_avoid` (CLI) | `test.py:37` | off | toggles TAGA at test |
| `safe_margin` | hard-coded `evaluation.py:126` | `0.25` | tangent trigger distance |
| `goal_threshold` | hard-coded `evaluation.py:125` | `3.0` | goal-priority cutoff |
| `EMERGENCY/DANGER/CAUTION` | hard-coded `taga_safety.py:9-11` | `0.4 / 0.6 / 1.0` | safety controller |

Nothing in TAGA is exposed as tunable from `config.py` — every threshold is a magic number in code.

---

## 7. Mapping Code State → IROS Reviewer Concerns

| Reviewer concern | Current code handles it? | Notes |
|---|---|---|
| R2-Q1 Individual collisions when TAGA active | Partial | Safety controller exists but EMERGENCY fully overrides TAGA, causing jerks; repulsive force from *all* humans even when only one matters |
| R2-Q2 Group detection method | **No** | DBSCAN commented out; ground-truth labels used |
| R2-Q3 Why tangent, not goal | Partial | Two goal-priority shortcuts exist but are coarse (distance-only), no explicit trajectory cost |
| R2-Q4 Motion discontinuity at switch | **No** | Hard step change: base policy → tangent unit vector → safety override. No hysteresis, no velocity filtering |
| R2-Q7 GCR not independent of SR/CR/TR | **Yes** | `assert success + collision + timeout == test_size` — GCR is reported as a separate % |
| R3 Predefined group labels | **No** | Same as R2-Q2; sim ground truth is still the source |
| R3 Compare with zone-based / F-formation | Partial | `zone_based.py` and `f_formation.py` baselines exist as policies; head-to-head numbers not yet produced |
| AE Novelty / positioning | N/A | Writing-side fix |

---

## 8. Known Weaknesses (algorithmic, for the revision)

1. **No runtime group detection.** `group.ground_truth = True` in the shipped config; DBSCAN block is commented out. Highest-priority fix for reviewer response.
2. **Hard switch between modes.** Base policy → TAGA → emergency override are three disjoint regimes. The action vector can jump discontinuously at every boundary — exactly the "jerk" R2-Q4 flagged.
3. **Greedy tangent side selection.** The smaller-angle rule ignores what the robot will have to do *after* passing the group (e.g. the chosen side may put another group in the path).
4. **Single-group handling.** `break` after the first triggering group means a configuration of two groups side-by-side reduces to avoiding one and intruding into the other.
5. **Magic numbers in hot paths.** `safe_margin=0.25`, `goal_threshold=3.0`, safety zones `0.4/0.6/1.0`, repulsive force coefficient `2.0`, blend weights — all unexplained constants, not tied to kinematics, robot/human radius, or `v_pref`.
6. **Emergency override ignores goal.** `emergency_avoid` is purely repulsive from one human; can push the robot into a wall / other humans / another group.
7. **No temporal consistency.** Group membership recomputed every step; if DBSCAN is re-enabled, labels will flicker and the tangent will oscillate.
8. **Paper / code zone mismatch.** Paper claims `d_critical=0.3, d_personal=0.5`; code uses `0.4/0.6/1.0`. Either paper or code must be corrected.

---

## 9. Suggested Improvement Axes (ranked, not yet decided)

| Priority | Change | Addresses |
|---|---|---|
| **P0** | Re-enable DBSCAN detection; add hysteresis on group ID assignment to prevent flicker | R2-Q2, R3 |
| **P0** | Smooth switching via a sigmoid blend between base policy and TAGA over a transition band `[R+S-δ, R+S+δ]` | R2-Q4 |
| **P1** | Cost-aware tangent side: pick the side that minimises expected collision with *other* groups/humans in the near horizon, not just goal-angle | R2-Q1, R2-Q3 |
| **P1** | Replace EMERGENCY hard override with bounded-acceleration safety filter (velocity clipping, not swap) | R2-Q1, R2-Q4 |
| **P1** | Expose zones and `safe_margin` in config; tie them to `robot.v_pref` and `time_step` | clarity |
| **P2** | Multi-group aggregation: handle the k-nearest blocking groups, not only the first | robustness |
| **P2** | Reconcile paper's `d_critical/d_personal` with code's three-zone controller, or update the paper | paper integrity |

---

## 10. Entry Points When We Start Editing

- **Switching logic**: `rl/evaluation.py:99-169` (the main TAGA loop at eval time)
- **Safety controller**: `crowd_nav/policy/taga_safety.py` (the whole file)
- **Group detection**: `crowd_sim/envs/crowd_sim_var_num.py:260-393` — uncomment / replace DBSCAN block, set `config.group.ground_truth = False`
- **Metric**: `rl/evaluation.py:199-261` for GCR, already independent
- **Baselines for comparison**: `crowd_nav/policy/zone_based.py`, `crowd_nav/policy/f_formation.py`
