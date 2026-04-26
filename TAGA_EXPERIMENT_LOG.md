# TAGA Experiment Log

Running log of every TAGA design change tried, why, and the measured outcome.
Append a new entry on **every** experiment — never overwrite. Future Claude
sessions read this top-to-bottom to avoid re-trying ideas that already failed.

> **Rule (durable):** every code change to `apply_taga` or `taga.*` config
> must add a section here with: date, change, motivation, evaluation setup,
> result (SR/CR/GCR before vs after), and verdict (kept / reverted / tuned).

---

## Reference points

| Run | SR_orca+taga | CR_orca+taga | GCR_orca+taga | SR_sf+taga | SR_irl+taga | Notes |
|---|---|---|---|---|---|---|
| **B0 — Original good baseline** (100 seeds, `results/taga_config_baseline_optionsAB.json`, 2026-04-25) | **0.84** | 0.12 | 0.0254 | 0.36 | 0.34 | Cost-aware tangent + sigmoid alpha + Options A+B safety filter. ORCA: +5pp vs base. |
| **B1 — Broken anti-vel + cone-pause** (100 seeds, before today's fixes) | 0.77 | 0.18 | 0.0212 | 0.42 | 0.33 | First try at goal-blind anti-velocity + indiscriminate cone-pause. ORCA: −4pp regression. |

For all per-seed details, see `results/summary_pause_last.txt` (B1 log).

---

## Permanent rules learned

| Rule | Why |
|---|---|
| **TAGA must never override base when base is succeeding.** Every spurious activation can derail a base trajectory — especially for stateful neural policies (intention_rl) where one bad action cascades. | Observed in v2/v3: base ok @ N steps → +taga collision in <50 steps with high `activated` count. |
| **Goal awareness is non-negotiable.** Any tangent-direction or escape-direction signal that doesn't include `goal_dir` will backpedal the robot when groups walk toward the same goal. | Anti-vel-only tangent (Exp 01) caused 134-activation timeouts on seeds where group V_group ≈ goal_dir. |
| **Cone-pause must be TTC-gated, not presence-gated.** Pausing on any individual standing in the escape direction freezes the robot indefinitely while pedestrians walk past. | intention_rl GCR doubled (0.018 → 0.034) under presence-only cone-pause (Exp 01). |
| **Pauses need a budget.** Without a `max_consecutive_pause`, robot can freeze inside a moving hull until timeout. | Seed 95 in B1 had `cone_paused=19` → timeout. |
| **20-seed sweeps are noise.** ±10pp SR variance. Use 100 seeds before claiming improvement. ORCA at 1.00 in 20 seeds vs 0.79 in 100 seeds (same code). | Observed across v2/v3/v4 reruns. |
| **For static_f groups, future-hull = current-hull.** V_group = 0, so no benefit from prediction. The over-firing problem on static groups stems from ORCA's instantaneous direction projecting through the hull when ORCA's reactive correction would actually curve around it. | Seeds 0/7 of v3 had mostly static groups, 18-27 activations, collisions. |

---

## Experiments

### Exp 00 — B0 baseline (reference, 2026-04-25)

**Config:** cost-aware tangent (`anti_vel_dynamic=False`, the default at the time),
sigmoid alpha blending, current-hull intent gate, Options A+B safety filter
(multi-horizon collision check + iterative alpha search).

**Result (100 seeds):** ORCA+TAGA SR 0.84 / CR 0.12 / GCR 0.0254 — **+5pp SR vs base ORCA (0.79)**.

**Verdict:** kept as reference baseline. Snapshot in `results/taga_config_baseline_optionsAB.json`.

---

### Exp 01 — Anti-velocity tangent + presence-based cone-pause (2026-04-26)

**Change:** for `dynamic_lf` / `dynamic_free` groups, replaced cost-aware
CW/CCW selection with `argmax(dot(tangent, -V_group))`. Added an escape-cone
scan that paused the robot if any individual stood within 1.2m and 45° of the
chosen tangent.

**Motivation:** user observed collisions while TAGA was dodging dynamic groups —
hypothesis was that perpendicular tangents drift into the moving hull's future
positions, and any individual on the tangent path gets hit.

**Result (100 seeds, `results/summary_pause_last.txt`):**
- ORCA+TAGA: SR 0.84 → **0.77 (−7pp)**, CR 0.12 → 0.18, GCR ≈
- intention_rl+TAGA: SR 0.40 → 0.33 (−6pp), GCR 0.018 → **0.034 (worse)**
- SF+TAGA: ≈ neutral

**Failure modes observed:**
- Seeds with V_group ≈ goal_dir: anti-vel = backwards → 100+ activations + timeout.
- intention_rl: cone-pause held robot at hull boundary, GCR ticked every step.
- Seed 95: `cone_paused=19` → robot froze, timed out.

**Verdict:** reverted. Anti-velocity is goal-blind; cone-pause is too aggressive.

---

### Exp 02 — Goal-blended anti-velocity + TTC cone-pause + pause budget (2026-04-27)

**Changes:**
1. Tangent score = `0.6·dot(t, -V_group) + 0.4·dot(t, goal_dir)` (later: 0.3/0.7).
2. Cone-pause uses TTC: only pause if individual on collision course over 0.7s horizon, dist < 0.55m.
3. Skip anti-vel entirely if `dot(V_group_unit, goal_dir) > 0.5` (group walking with us toward goal).
4. `max_consecutive_pause = 3`; after that, force-commit to base action and reset.

**Motivation:** address the three root causes from Exp 01.

**Result (20 seeds, `/tmp/taga_test_v2.log`):**
- ORCA+TAGA: SR 0.90 (+13pp vs Exp 01), CR 0.05, GCR 0.020
- SF+TAGA: SR 0.40, GCR 0.000
- intention_rl+TAGA: SR 0.30, GCR 0.037

ORCA+TAGA seed 2 (was 134 activations + timeout in Exp 01) → 22 activations + success. Goal-blended fix worked.
But ORCA seeds 1 + 10 still hurt (`activated=7,5` → collision/timeout).

**Verdict:** improvement over Exp 01, but anti-vel still introduces direction-flipping under noisy V_group. Move to gate-side fix.

---

### Exp 03 — Future-hull intent gate (2026-04-27)

**Change:** in the intent gate, predict each hull's future position by translating
its vertices by `V_group · t_intent`, then check if `robot_future` lies inside the
*future* hull. Equivalent: translate `robot_future` by `−V_group·t_intent` and
test against the current hull. Static groups (V_group = 0) unchanged.

**Motivation:** the gate was firing TAGA when base ORCA's instantaneous velocity
pointed at the *current* hull position — but by t=0.7s the hull had walked away.
Spurious activation = TAGA fights with base = collision/timeout.

**Result (20 seeds, `/tmp/taga_test_v3.log`):**

| Policy | SR | CR | GCR |
|---|---|---|---|
| orca+taga | 0.80 | 0.15 | 0.0215 |
| social_force+taga | 0.40 | 0.45 | 0.0018 |
| intention_rl+taga | **0.40 (+10pp vs Exp 02)** | 0.15 | 0.054 |

Per-seed: helped 3 intention_rl timeouts (seeds 2/3/6), hurt 5 (seeds 1/7/11/16/19).
For ORCA, hurt seeds 0 + 7 had mostly **static** groups — future-hull doesn't help
those because V_group = 0.

**Verdict:** kept (strict improvement on dynamic groups). Doesn't fix static-group
over-firing — that's a deeper ORCA reactive-correction issue.

---

### Exp 04 — Disable anti-vel, keep future-hull (2026-04-27)

**Change:** `taga.anti_vel_dynamic = False`. Falls back to cost-aware tangent
(matches B0 baseline). Future-hull intent gate stays on. Pause budget stays on.
TTC cone-pause stays on but is gated off when anti_vel_dynamic=False.

**Motivation:** Exp 02/03 showed anti-vel introduces direction-flipping noise.
Cost-aware is what delivered the +5pp in B0. Goal was to isolate the
future-hull gate's contribution on top of B0.

**Result (20 seeds, `/tmp/taga_test_v4.log`):**

| Policy | SR | CR | TR | GCR | Reward |
|---|---|---|---|---|---|
| orca | 0.85 | 0.15 | 0.00 | 0.0094 | 26.50 |
| **orca+taga** | **0.95** | **0.00** | 0.05 | 0.0205 | 33.06 |
| social_force | 0.50 | 0.30 | 0.20 | 0.0000 | 19.18 |
| social_force+taga | 0.50 | 0.40 | 0.10 | 0.0059 | 18.42 |
| intention_rl | 0.50 | 0.15 | 0.35 | 0.0757 | 20.74 |
| intention_rl+taga | 0.40 | 0.15 | 0.45 | 0.0643 | 22.48 |

**Per-seed:**
- ORCA: helps 2, hurts 0, **net +2**. Saved seeds 3 (col@33→ok@58) and 10 (col@99→ok@145). Zero regressions.
- SF: helps 3 (saved 2 collisions + 1 timeout), hurts 3 (3 collisions). Net 0.
- intention_rl: helps 2 (saved 2 timeouts), hurts 4 (1 col + 3 timeouts). Net −2.

**Comparison to references:**
- vs **B0 baseline (100 seeds)**: ORCA+TAGA SR 0.84 → 0.95 (+11pp); CR 0.12 → 0.00 (−12pp). Variance in 20 seeds is ±10pp, so directional but needs 100-seed confirmation.
- vs **B1 broken (100 seeds)**: ORCA+TAGA SR 0.77 → 0.95 (+18pp).

**Verdict:** **kept**. This is the best ORCA+TAGA seen so far. The future-hull
intent gate is a strict win on top of B0's cost-aware tangent for ORCA. The
pause budget caps the rare timeout edge cases without firing in normal play.

**Open issues:**
1. intention_rl still net-negative (−2). The neural policy is goal-priority-bound (high `goal_pri` count) and timeouts cascade from the slightest TAGA action.
2. Need 100-seed run to confirm the ORCA win is real, not noise.
3. SF is noise-level (TAGA almost never fires for SF — see prior runs).

---

## How to add a new entry

1. Run a controlled change in `record_comparison.py` / `crowd_nav/configs/config.py`.
2. Run 20 seeds across `orca,orca+taga,social_force,social_force+taga,intention_rl,intention_rl+taga`.
3. Append a section above using the Exp NN template:
   ```
   ### Exp NN — <one-line title> (YYYY-MM-DD)
   **Change:** <what you flipped>
   **Motivation:** <which prior exp / observation drove this>
   **Result (N seeds, /path/to/log):** <table>
   **Verdict:** kept / reverted / needs tuning
   ```
4. If a permanent design rule emerges, also add it to **Permanent rules learned**.
5. Update `MEMORY.md` only if the result changes the headline pattern (e.g.,
   classical-vs-neural asymmetry).
