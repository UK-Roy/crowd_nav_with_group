#!/usr/bin/env bash
# scripts/run_eps_sweep.sh
#
# Runs the full DBSCAN epsilon sweep (all eps values, both feature modes) and
# records each result automatically into results/perception_detection_results.txt
# (replaces TBD rows as results come in).
#
# After this script completes, results/perception_detection_results.txt has all
# Table 2 (appendix) rows filled. Copy the values into grace_appendix.tex
# to replace the \tbd{} cells.
#
# Usage (run from anywhere in the repo):
#   bash scripts/run_eps_sweep.sh                    # normal run (~5-10 min)
#   bash scripts/run_eps_sweep.sh --n-test=1000      # fewer samples (faster)
#   bash scripts/run_eps_sweep.sh --n-test=0         # full test set (very slow)
#   bash scripts/run_eps_sweep.sh --no-cuda          # force CPU

set -euo pipefail

# Always run from repo root regardless of where the script is called from
cd "$(dirname "$0")/.."

RECORD="results/perception_detection_results.txt"
N_TEST=2000   # test samples per eps (increase for more precise results)
EXTRA_ARGS="" # pass through any extra flags (e.g. --no-cuda)

for arg in "$@"; do
    case "$arg" in
        --n-test=*)   N_TEST="${arg#*=}" ;;
        --n-test)     shift; N_TEST="$1" ;;
        --no-cuda)    EXTRA_ARGS="$EXTRA_ARGS --no-cuda" ;;
    esac
done

# ── Auto-create results record file if missing ────────────────────────────────
mkdir -p results
if [ ! -f "$RECORD" ]; then
    echo ">>> $RECORD not found — creating it now..."
    python - <<'PYEOF'
import os
os.makedirs("results", exist_ok=True)
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
Run command:  bash scripts/run_dbscan_comparison.sh

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
Run command:  bash scripts/run_eps_sweep.sh

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
with open("results/perception_detection_results.txt", "w") as f:
    f.write(content)
print("Created results/perception_detection_results.txt")
PYEOF
fi

if [ ! -f "gram_v2_data/test.npz" ]; then
    echo "ERROR: gram_v2_data/test.npz not found. Run: python gram_v2_collect_data.py"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "   DBSCAN Epsilon Sweep — results → $RECORD"
echo "   Test samples per eps: $N_TEST"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Position-only sweep ────────────────────────────────────────────────────────
echo "── Position-only (eps = 0.5 0.8 1.0 1.2 2.0 2.5) ──────────────"
for eps in 0.5 0.8 1.0 1.2 2.0 2.5; do
    echo ""
    echo "  [pos] eps=$eps ..."
    python eval_detection_comparison.py \
        --fixed-eps "$eps"       \
        --mode position          \
        --dbscan-only            \
        --no-latex               \
        --n-test "$N_TEST"       \
        --record-file "$RECORD"  \
        $EXTRA_ARGS
done

# ── Position + velocity sweep ──────────────────────────────────────────────────
echo ""
echo "── Position + velocity (eps = 0.5 0.8 1.0 1.2 1.5 2.0) ─────────"
for eps in 0.5 0.8 1.0 1.2 1.5 2.0; do
    echo ""
    echo "  [pos+vel] eps=$eps ..."
    python eval_detection_comparison.py \
        --fixed-eps "$eps"       \
        --mode pos+vel           \
        --dbscan-only            \
        --no-latex               \
        --n-test "$N_TEST"       \
        --record-file "$RECORD"  \
        $EXTRA_ARGS
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Done. All TBD rows filled in $RECORD."
echo ""
echo "Next: copy values from $RECORD (Table 2) into"
echo "      grace_appendix.tex → Table A2 (tab:detection_full)."
echo "════════════════════════════════════════════════════════════════"
