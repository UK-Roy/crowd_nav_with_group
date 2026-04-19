# Realistic Crowd Modeling — Implementation Plan

Planning document for the 5-part upgrade of human-group realism in the
simulator. **Goal: turn this environment into a canonical, realistic benchmark
in which every navigation policy — TAGA, GARN, GRAM, ORCA, social-force, etc.
— is evaluated under the same realistic pedestrian dynamics.** GARN reward
integration is a downstream detail, not the driver.

Everything is **gated behind config flags** so existing trained checkpoints
continue to load and run bit-exactly when flags are off. Once a phase is
validated, we flip its flag **on by default** so the "realistic" config
becomes the standard evaluation environment.

---

## 1. Scope Recap

| # | Feature | Reference |
|---|---|---|
| 1 | ConvexHull group geometry + point-in-hull intrusion test | — |
| 2 | Dynamic groups walk 15 % slower than individuals | Moussaid et al. 2010 |
| 3 | F-formations (vis-à-vis / L-shape / side-by-side / circle) for static groups | Kendon 1990 |
| 4 | Leader-follower dynamics for dynamic groups | Helbing & Molnar 1995 (force blend) |
| 5 | Individual preferred-speed sampling `v ~ N(1.34, 0.26)` clipped to `[0.8, 1.8]` | Weidmann 1992 |

---

## 2. Config Surface (new `realistic` section)

Append to `crowd_nav/configs/config.py` **after** the `taga` block. All defaults
chosen so the **off** state is strictly equivalent to today's environment.

```python
realistic = BaseConfig()
realistic.enabled = False              # master kill-switch
realistic.use_convex_hull = False      # ConvexHull geometry + point-in-hull
realistic.use_f_formations = False     # F-formation static groups
realistic.use_leader_follower = False  # leader-follower dynamic groups
realistic.use_speed_variation = False  # per-human v_pref sampling
realistic.use_group_speed_factor = False  # 0.85× slowdown for dynamic groups

realistic.group_speed_factor = 0.85
realistic.individual_speed_mean = 1.34
realistic.individual_speed_std  = 0.26
realistic.individual_speed_min  = 0.80
realistic.individual_speed_max  = 1.80

# Hull/formation robustness knobs
realistic.hull_degenerate_buffer = 0.30  # half-width for 2-member rectangle (m)
realistic.f_formation_radius     = 0.65  # member distance from o-space (m)
realistic.leader_follower_spacing = 0.70 # side-by-side offset from leader (m)
realistic.leader_follower_gain    = 1.20 # follow-force weight (dimensionless)
```

`enabled` is a top-level guard. Sub-flags compose, so we can validate each
feature independently, then flip `enabled=True` + per-feature flags on by
default once the canonical benchmark settles.

**Evaluation-first framing.** None of these changes are PPO-training-side.
They run inside `crowd_sim` — i.e. every policy (learned or classical) sees
them at test time just by changing the config. No retraining is required to
re-evaluate any existing checkpoint in the realistic environment.

---

## 3. Files to Create / Modify

### Create

- **`crowd_sim/envs/utils/realistic_human_modeling.py`** — new module, pure
  implementation, **no** imports back into the env (avoids circular import).
  Contents:
  - `sample_individual_speed(rng, cfg)` → float.
  - `group_target_speed(member_speeds, cfg)` → float = `cfg.group_speed_factor * min(member_speeds)`.
  - `class FFormation` — enum-like (`VIS_A_VIS`, `L_SHAPE`, `SIDE_BY_SIDE`, `CIRCLE`) plus a `compute_positions_orientations(centroid, radius, n_members, rng)` factory that returns a list of `(px, py, theta)`.
  - `class ConvexHullGeometry` — wraps `scipy.spatial.ConvexHull`; handles degenerate cases explicitly:
    - `n == 1` → circle fallback (radius = human_radius).
    - `n == 2` → oriented rectangle (segment ± `hull_degenerate_buffer` perpendicular); exposes `contains(point)` via half-plane tests.
    - `n >= 3` collinear → fall back to rectangle around bounding segment.
    - Generic: stores vertices (ordered CCW) and a `Delaunay` triangulation for `contains`.
    - Exposes `bounding_radius` (for back-compat where a scalar is needed) and `vertices_ccw`.
  - `class LeaderFollowerController` — given a follower's state, its leader, target offset, and nearby agents, returns a velocity command as a blend of follow-force, social force, and obstacle force. Uses the existing `crowd_nav.policy.social_force` primitives where possible (no re-implementation).
  - `class RealisticGroup` — **compositional** layer, **does not** subclass `Group`. Holds a reference to the underlying `Group` plus:
    - `formation: FFormation | None`
    - `leader: Human | None`
    - `follower_offsets: dict[human_id, np.ndarray]`
    - `hull: ConvexHullGeometry | None`
    - `group_v_pref: float | None`
    - `refresh_hull(humans)` — recompute each step from current member positions.
    - `advance(humans, robot)` — called from the env; if static → no-op (positions frozen); if dynamic → updates follower velocities via `LeaderFollowerController`.

### Modify (minimally)

- **`crowd_nav/configs/config.py`** — add the `realistic` section described above. No changes to existing values.
- **`crowd_sim/envs/utils/human.py`** — in `__init__` (or a small `sample_v_pref(cfg, rng)` method) sample `v_pref` when `cfg.realistic.enabled and cfg.realistic.use_speed_variation`. Otherwise keep the existing fixed value.
- **`crowd_sim/envs/utils/group.py`** — add **one** optional field `self.realistic = None` and leave everything else untouched. The existing `position_members` / `select_formation` remain the legacy path.
- **`crowd_sim/envs/crowd_sim.py`** — where groups are instantiated (around `assign_group_centroids` / `generate_humans`), branch on `cfg.realistic.enabled`:
  - If on: construct `RealisticGroup` wrappers, call the F-formation / leader-follower init instead of `position_members`.
  - If off: unchanged path.
- **`crowd_sim/envs/crowd_sim_var_num.py`**:
  - In the obs builder (the block around line 420 that writes `ob['group_centroids']`, `ob['group_radii']`), when `realistic.use_convex_hull` is on, additionally publish `ob['group_hulls']` = list of hull-vertex arrays, one per group_id (padded slots get a single-NaN sentinel, or we leave them absent in a dict keyed by group_id).
  - For dynamic groups, call `realistic_group.advance(...)` at each step before the existing per-human `act()` loop.
- **`crowd_sim/envs/garn_reward.py`** — secondary. When `cfg.realistic.use_convex_hull` is on, swap the `dist_curr < radii` indicator for a point-in-hull test so GARN's reward matches the realistic geometry. Behind the same flag; in-flight training on GPU remains bit-exact. This is a reward-consistency patch, not a core requirement of this task.

### Untouched (per ground rules)

- `crowd_nav/policy/taga_safety.py`, `rl/evaluation.py` — TAGA stays bit-exact.
- `rl/networks/stgan_model.py` — GARN policy net untouched.
- Training loops, PPO, arguments.py.

---

## 4. Detailed Design — Key Sub-components

### 4.1 ConvexHull with Degeneracy Handling

```
ConvexHullGeometry(positions: (n, 2) np.float32):
    if n == 1:               # singleton → circle
        kind = "circle"
        center, radius = positions[0], human_radius
    elif n == 2 or collinear(positions):
        kind = "rectangle"   # oriented; axis = segment direction
        build 4 corners = endpoints ± buffer * perpendicular_unit
    else:
        hull = ConvexHull(positions)   # scipy, QhullError-handled
        vertices_ccw = positions[hull.vertices]
        kind = "polygon"
        delaunay = Delaunay(vertices_ccw)
    contains(p):
        if kind == "circle":  return |p - center| <= radius
        if kind == "rectangle": four half-plane tests
        if kind == "polygon": return delaunay.find_simplex(p) >= 0
    bounding_radius():
        max_i |vertex_i - centroid|    # for legacy callers
```

### 4.2 F-Formations (static groups)

Kendon's 1990 taxonomy maps to initial *position + orientation*:

| Formation | Members | Geometry |
|---|---|---|
| `VIS_A_VIS` | 2 | Members at (+r, 0) and (−r, 0), orientations facing each other (θ = π and 0) |
| `L_SHAPE` | 2 | (+r, 0) facing −x; (0, +r) facing −y (90° between them) |
| `SIDE_BY_SIDE` | 2 | (+r/2, 0) and (−r/2, 0), both facing +y |
| `CIRCLE` | ≥ 3 | evenly spaced on circle of radius `r`, each facing centroid |

Centroid is the group's assigned spawn centroid; `r = f_formation_radius`.
Formation drawn uniformly at random from the applicable set; recorded so the
env can report it (for figures / logs).

### 4.3 Leader-Follower Dynamics

- **Leader**: `members[0]` (deterministic; we can swap to random-leader later if needed without a flag).
- **Leader v_pref** (if `use_group_speed_factor`): `0.85 * min(member v_pref)`.
- **Follower offsets**: precomputed at group creation — evenly spaced side-by-side around leader, spacing = `leader_follower_spacing`. Offsets are stored in the leader's frame (along leader's instantaneous heading).
- **Follower velocity** at each step:
  ```
  target = leader.pos + R(leader.heading) @ offset
  follow_force    = leader_follower_gain * (target - follower.pos)
  social_force    = existing SF primitives vs. other humans + robot
  obstacle_force  = existing SF primitives vs. obstacles (if any)
  v_desired       = clip(follow_force + social_force + obstacle_force, v_pref)
  ```
  This is applied by overriding the follower's `act()` result inside
  `RealisticGroup.advance` — the follower's *policy* is untouched; we just
  replace its action for this step.

### 4.4 Policy-Side Consumption (TAGA, GARN, etc.)

The environment publishes richer group state (`group_hulls`, formation labels,
follower offsets) **as additive obs keys**. Policies opt in:

- **TAGA** (runtime-only, `rl/evaluation.py`) — can use hull vertices for
  tangent-side selection instead of bounding-circle centroid. Out of scope
  for this task; will be a follow-up once the env is stable.
- **GARN** (`crowd_sim/envs/garn_reward.py`) — swap
  `inside = (dist_curr < radii)` for a point-in-hull test when
  `use_convex_hull` is on. The reward landscape then matches the realistic
  geometry the robot actually navigates.
- **GRAM / ORCA / social-force** — do not read any new keys, so they are
  unaffected.

Because everything is publish-and-opt-in, the environment becomes the
single source of realism and each policy can be patched independently.

---

## 5. Phase-by-Phase Rollout

After each phase: CPU sanity check (`python test.py --no-cuda --num-processes 2 --test-episodes 5 --visualize`) with **both** flags-off (regression) and flags-on (new behavior). Stop for user approval.

Order is chosen by **env-impact first, policy-coupling last**, so the
environment feels realistic as early as possible. GARN-reward wiring moves to
the very end — the environment work doesn't depend on it.

| Phase | Deliverable | New files | Flag to toggle | Risk |
|---|---|---|---|---|
| A | Individual speed variation (Weidmann) | `realistic_human_modeling.py` (speed sampler only) | `use_speed_variation` | Low — no geometry change |
| B | Group speed factor (Moussaid 15 % slowdown) | + group-speed util | `use_group_speed_factor` | Low |
| C | F-formations for static groups | + `FFormation` + `RealisticGroup.init_static` | `use_f_formations` | Medium — changes initial positions |
| D | Leader-follower dynamics for dynamic groups | + `LeaderFollowerController` + `RealisticGroup.advance` | `use_leader_follower` | High — new per-step force composition |
| E | ConvexHull geometry + obs publish (+ GARN patch) | + `ConvexHullGeometry` | `use_convex_hull` | Medium — obs schema grows; GARN reward indicator changes |

---

## 6. Edge Cases & Open Questions

1. **2-member groups + ConvexHull.** scipy raises `QhullError` for <3 non-collinear points. Fallback: oriented rectangle (segment axis ± buffer). Plan accepts this; alternative is to always inflate with `buffer` and use the Minkowski sum of the segment with a disk (capsule shape) — more accurate but costlier. Leaning toward the rectangle for speed; revisit if visual shows issues.

2. **Collinear ≥ 3-member groups.** scipy also fails. Same rectangle fallback applied.

3. **Re-sampling speed on episode reset.** `sample_individual_speed` is called once per episode (in human init/reset), not every step. Confirmed with `humans.random_v_pref` already in config.

4. **F-formation and ORCA / SF compatibility.** Static groups with `isObstacle=True` don't use policies — their positions are frozen. So F-formation only affects initial placement + recorded orientation; no policy conflict.

5. **Leader-follower vs ORCA.** If the follower's *policy* is ORCA, ORCA still runs but its output is discarded/overwritten by `RealisticGroup.advance`. If that's unacceptable, we'd need to expose "follower" as a policy type. Proposed: start with override; upgrade to proper policy later if reviewers push.

6. **Leader departure.** If a leader reaches its goal, we currently have no re-leader logic. Plan: on leader arrival, pick next un-arrived member as leader; if none, the group is effectively dissolved.

7. **GARN training/eval reward mismatch.** The in-flight GARN run learns against bounding-circle radii. Evaluating it with `use_convex_hull=True` shifts the *reward* landscape at test time only — but GARN is a policy evaluation here, we don't re-score it on training reward. So the mismatch only matters if we *retrain* GARN under the realistic env. Plan: evaluate the trained GARN checkpoint in the realistic env (fair apples-to-apples with TAGA), and schedule an optional realistic-env retraining run later if reviewers ask.

8. **Observation space change.** Adding `ob['group_hulls']` changes the dict schema. Policies that don't read this key are unaffected (they access specific keys by name). No breakage expected for TAGA/GARN/GRAM, but worth verifying by running each before declaring Phase A done.

9. **`group_id = 0` special case.** Today group 0 is always an obstacle even when `group.dynamic=True`. F-formations and leader-follower should respect this: group 0 stays static even under realistic modeling, unless the user explicitly removes the special case (out of scope for this task).

10. **Random seed determinism.** All new randomness routed through `np.random.default_rng(seed)` owned by the env, not the global numpy state, so reproducibility is preserved.

---

## 7. Backward Compatibility

- **Default**: `realistic.enabled = False` → every new code path short-circuits immediately. Zero changes to legacy behavior.
- **Trained checkpoints**: obs dict is only *augmented* (`group_hulls` added when flag on); existing models consume only the keys they know → they still load and run.
- **In-flight GARN training**: unaffected — flags off on the GPU branch. When we later want the hull variant, it's a separate retraining run.
- **TAGA / GRAM**: do not read any new key; unaffected.

---

## 8. Testing Strategy

Per phase:
1. **Flags-off regression** — run `python test.py --no-cuda --num-processes 2 --test-episodes 5 --visualize` and confirm SR / CR / TR identical to pre-change numbers (or within RNG noise with fixed seed).
2. **Flags-on smoke** — same command with the phase's flag set to `True`. Confirm no tracebacks, visualizer renders sensibly (hull drawn; formations look right; leader line visible in dynamic groups).
3. **Trained-model load** — `python test.py` pointed at an existing checkpoint (e.g. `trained_models/GST_predictor_rand/checkpoints/41665.pt`), flags-off. Must produce same numbers as before.
4. **GARN smoke (Phase A only)** — reload the sanity-test GARN checkpoint on CPU with `use_convex_hull=True`, confirm the reward path executes without crashing. We're not comparing numbers here, just path coverage.

---

## 9. Deliverables Checklist (After Approval)

- [ ] Phase A (individual speed variation) → sanity → approval
- [ ] Phase B (group speed factor) → sanity → approval
- [ ] Phase C (F-formations) → sanity → approval
- [ ] Phase D (leader-follower) → sanity → approval
- [ ] Phase E (ConvexHull + obs + optional GARN patch) → sanity → approval
- [ ] Flip `realistic.enabled=True` + validated sub-flags on by default; this becomes the canonical evaluation env
- [ ] Re-run every available baseline (TAGA, GARN, GRAM, ORCA, social-force, zone-based, F-formation heuristic) in the realistic env and produce the comparison table
- [ ] Update `CLAUDE.md` with a "Realistic Crowd Modeling" section
- [ ] Append a note to `GARN_IMPLEMENTATION_PLAN.md` about the optional hull-based intrusion indicator
- [ ] Short paper-section draft (Markdown) citing Helbing & Molnar 1995, Moussaid 2010, Kendon 1990, Weidmann 1992

---

## 10. Ready for Review

Reframed for **environment-realism-first**. The defaults answered below
assume the v1 choices that keep things simple; say "yes, go" and I'll start
Phase A (individual speed variation), or flag any redirect.

1. **Phase order**: speed → group speed → F-formations → leader-follower → hull. Env-first, GARN-patch last. OK?
2. **Leader selection**: deterministic `members[0]` for v1 (random-with-seed is a two-line swap later).
3. **Follower policy**: override the follower's action inside `RealisticGroup.advance` for v1 (avoids touching every policy class).
4. **2-member hull**: oriented rectangle. Upgrade to capsule only if visuals look off.
5. **Speed sampling**: once per episode (resets with human).
6. **Default-on after validation**: once a phase passes sanity + all-baseline eval, flip its flag on by default so the realistic env is canonical.

Green light → I start Phase A.
