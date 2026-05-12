#!/usr/bin/env bash
# run_dbscan_comparison.sh
#
# Runs the full DBSCAN vs GRACE GroupDetector comparison table.
# Outputs both an ASCII table and a LaTeX snippet for the CoRL paper.
#
# Prerequisites:
#   - gram_v2_data/ must contain val.npz and test.npz
#   - GroupDetector checkpoints must exist (trained_models/gram_v2/phase1_v2 and phase2_v2)
#
# If the model results are missing, run first:
#   bash run_perception_eval.sh
#
# Usage:
#   bash run_dbscan_comparison.sh              # uses saved results.pt (fast)
#   bash run_dbscan_comparison.sh --force      # re-runs model inference from scratch

set -euo pipefail

DATA_DIR="gram_v2_data"
PHASE1_CKPT="trained_models/gram_v2/phase1_v2/B/best.pt"
PHASE1_SAVE="trained_models/gram_v2/phase1_v2/B"
PHASE2_CKPT="trained_models/gram_v2/phase2_v2/best.pt"
PHASE2_SAVE="trained_models/gram_v2/phase2_v2"

RECORD="perception_detection_results.txt"

# ── Auto-create results record file if missing ────────────────────────────────
if [ ! -f "$RECORD" ]; then
    echo ">>> $RECORD not found — creating it now..."
    python - <<'PYEOF'
content = """\
================================================================================
GRACE — Group Detection Results Record
================================================================================
Test set: gram_v2_data/test.npz  (~8,549 frames)
Val set:  gram_v2_data/val.npz   (~8,133 frames, used for eps tuning only)
Metrics:
  F1      — pairwise same-group pair classification (threshold 0.5)
  Prec    — pairwise precision
  Recall  — pairwise recall
  ARI     — Adjusted Rand Index (whole-scene cluster quality)
  AUROC   — area under ROC curve (probabilistic methods only; — = N/A)
================================================================================


────────────────────────────────────────────────────────────────────────────────
TABLE 1 — MAIN COMPARISON  (CoRL main paper / grace.tex)
────────────────────────────────────────────────────────────────────────────────
Run command:  bash run_dbscan_comparison.sh

Method                              F1     Prec   Recall   ARI    AUROC
────────────────────────────────────────────────────────────────────────────────
[Classical baselines]
DBSCAN pos. only  (eps*=TBD)       TBD    TBD    TBD      TBD    —
DBSCAN pos.+vel.  (eps*=TBD)       TBD    TBD    TBD      TBD    —
────────────────────────────────────────────────────────────────────────────────
[Ours — GroupDetector]
Phase 1: encoder only              TBD    TBD    TBD      TBD    TBD
Phase 2: enc.+GNN (GRACE backbone) TBD    TBD    TBD      TBD    TBD
────────────────────────────────────────────────────────────────────────────────


────────────────────────────────────────────────────────────────────────────────
TABLE 2 — DBSCAN EPSILON SWEEP  (CoRL appendix / grace_appendix.tex)
────────────────────────────────────────────────────────────────────────────────
Run command:  bash run_eps_sweep.sh

                                     F1     Prec   Recall   ARI
────────────────────────────────────────────────────────────────────────────────
[DBSCAN — position only]
  eps = 0.5                         TBD    TBD    TBD      TBD
  eps = 0.8                         TBD    TBD    TBD      TBD
  eps = 1.0                         TBD    TBD    TBD      TBD
  eps = 1.2                         TBD    TBD    TBD      TBD
  eps = 1.5                         TBD    TBD    TBD      TBD
  eps = 2.0                         TBD    TBD    TBD      TBD
  eps = 2.5                         TBD    TBD    TBD      TBD

[DBSCAN — position + velocity (vel_scale=0.5)]
  eps = 0.5                         TBD    TBD    TBD      TBD
  eps = 0.8                         TBD    TBD    TBD      TBD
  eps = 1.0                         TBD    TBD    TBD      TBD
  eps = 1.2                         TBD    TBD    TBD      TBD
  eps = 1.5                         TBD    TBD    TBD      TBD
  eps = 2.0                         TBD    TBD    TBD      TBD
  eps = 2.5                         TBD    TBD    TBD      TBD

[Ours — GroupDetector (for reference)]
  Phase 1: encoder only             TBD    TBD    TBD      TBD
  Phase 2: enc.+GNN (GRACE)        TBD    TBD    TBD      TBD
────────────────────────────────────────────────────────────────────────────────


────────────────────────────────────────────────────────────────────────────────
HOW TO UPDATE THE CORL PAPER FROM THIS FILE
────────────────────────────────────────────────────────────────────────────────
Main paper (grace.tex):
  Table 2 (tab:detection) — update rows from TABLE 1 above.

Appendix (grace_appendix.tex):
  Table A2 (tab:detection_full) — update rows from TABLE 2 above.
  Replace each \\tbd{} with the numeric value from this file.


────────────────────────────────────────────────────────────────────────────────
HISTORY OF RUNS
────────────────────────────────────────────────────────────────────────────────
================================================================================
"""
with open("perception_detection_results.txt", "w") as f:
    f.write(content)
print("Created perception_detection_results.txt")
PYEOF
fi

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/test.npz" ]; then
    echo "ERROR: $DATA_DIR/test.npz not found."
    echo "       Run: python gram_v2_collect_data.py  to collect the dataset first."
    exit 1
fi

if [ ! -f "$PHASE2_CKPT" ]; then
    echo "ERROR: Phase2 checkpoint not found: $PHASE2_CKPT"
    echo "       Run: bash run_perception_eval.sh  to train / evaluate the model."
    exit 1
fi

# Check if results files exist (fast path)
FORCE_EVAL=""
if [ "${1:-}" = "--force" ]; then
    FORCE_EVAL="--force-eval"
    echo ">>> Force-eval mode: re-running model inference from scratch."
fi

RESULTS_EXIST=true
[ ! -f "$PHASE1_SAVE/phase1_results.pt" ] && RESULTS_EXIST=false
[ ! -f "$PHASE2_SAVE/phase2_results.pt" ] && RESULTS_EXIST=false

if [ "$RESULTS_EXIST" = false ] && [ -z "$FORCE_EVAL" ]; then
    echo ">>> Saved results.pt not found — running model eval automatically."
    echo "    (Use --force to always re-run inference regardless.)"
    FORCE_EVAL="--force-eval"
fi

# ── Run comparison ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "   GRACE Group Detection — DBSCAN vs Learned Module"
echo "════════════════════════════════════════════════════════════"
echo ""

python eval_detection_comparison.py \
    --data        "$DATA_DIR"   \
    --phase1-ckpt "$PHASE1_CKPT" \
    --phase1-save "$PHASE1_SAVE" \
    --phase2-ckpt "$PHASE2_CKPT" \
    --phase2-save "$PHASE2_SAVE" \
    $FORCE_EVAL

echo "Done."
