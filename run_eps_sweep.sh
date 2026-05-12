#!/usr/bin/env bash
# run_eps_sweep.sh
#
# Runs the full DBSCAN epsilon sweep (all eps values, both feature modes) and
# records each result automatically into perception_detection_results.txt
# (replaces TBD rows as results come in).
#
# After this script completes, perception_detection_results.txt has all
# Table 2 (appendix) rows filled. Copy the values into grace_appendix.tex
# to replace the \tbd{} cells.
#
# Usage:
#   bash run_eps_sweep.sh                    # normal run (~5-10 min)
#   bash run_eps_sweep.sh --n-test 0         # full test set (very slow)
#   bash run_eps_sweep.sh --no-cuda          # force CPU

set -euo pipefail

RECORD="perception_detection_results.txt"
N_TEST=2000   # test samples per eps (increase for more precise results)
EXTRA_ARGS="" # pass through any extra flags (e.g. --no-cuda)

for arg in "$@"; do
    case "$arg" in
        --n-test=*)   N_TEST="${arg#*=}" ;;
        --n-test)     shift; N_TEST="$1" ;;
        --no-cuda)    EXTRA_ARGS="$EXTRA_ARGS --no-cuda" ;;
    esac
done

if [ ! -f "$RECORD" ]; then
    echo "ERROR: $RECORD not found. Run from the project root directory."
    exit 1
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
