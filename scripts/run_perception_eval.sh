#!/usr/bin/env bash
# scripts/run_perception_eval.sh
#
# Runs eval-only passes for the GRACE GroupDetector (Phase1 + Phase2)
# and saves results.pt files alongside their checkpoints.
#
# Use this script when:
#   - results.pt files are missing (first time, or after re-training)
#   - you want to regenerate metrics from existing checkpoints
#
# After this script completes, run:
#   bash scripts/run_dbscan_comparison.sh      ← prints the full comparison table
#
# Usage (run from anywhere in the repo):
#   bash scripts/run_perception_eval.sh
#   bash scripts/run_perception_eval.sh --no-cuda       # force CPU
#   bash scripts/run_perception_eval.sh --phase1-only   # only Phase1 (encoder)
#   bash scripts/run_perception_eval.sh --phase2-only   # only Phase2 (GNN, GRACE backbone)

set -euo pipefail

# Always run from repo root regardless of where the script is called from
cd "$(dirname "$0")/.."

DATA_DIR="gram_v2_data"
PHASE1_CKPT="trained_models/gram_v2/phase1_v2/B/best.pt"
PHASE1_SAVE="trained_models/gram_v2/phase1_v2/B"
PHASE2_CKPT="trained_models/gram_v2/phase2_v2/best.pt"
PHASE2_SAVE="trained_models/gram_v2/phase2_v2"

NO_CUDA=""
PHASE_FILTER="both"

for arg in "$@"; do
    case "$arg" in
        --no-cuda)      NO_CUDA="--no-cuda" ;;
        --phase1-only)  PHASE_FILTER="phase1" ;;
        --phase2-only)  PHASE_FILTER="phase2" ;;
    esac
done

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/test.npz" ]; then
    echo "ERROR: $DATA_DIR/test.npz not found."
    echo "       Run: python gram_v2_collect_data.py  to collect the dataset first."
    exit 1
fi

# ── Phase 1 — encoder only ─────────────────────────────────────────────────────
if [ "$PHASE_FILTER" != "phase2" ]; then
    echo ""
    echo "── Phase 1 (GroupDetector encoder, no GNN) ──────────────────────────"
    if [ ! -f "$PHASE1_CKPT" ]; then
        echo "WARNING: Phase1 checkpoint not found: $PHASE1_CKPT"
        echo "         Skipping Phase1 evaluation."
    else
        python eval_detection_comparison.py \
            --data        "$DATA_DIR"    \
            --phase1-ckpt "$PHASE1_CKPT" \
            --phase1-save "$PHASE1_SAVE" \
            --phase2-ckpt "__skip__"     \
            --phase2-save "$PHASE2_SAVE" \
            --force-eval  \
            --save-results \
            --no-latex     \
            $NO_CUDA
        echo "Phase1 results saved to: $PHASE1_SAVE/phase1_results.pt"
    fi
fi

# ── Phase 2 — encoder + GNN (= GRACE backbone) ────────────────────────────────
if [ "$PHASE_FILTER" != "phase1" ]; then
    echo ""
    echo "── Phase 2 (GroupDetector encoder + GNN — GRACE backbone) ──────────"
    if [ ! -f "$PHASE2_CKPT" ]; then
        echo "ERROR: Phase2 checkpoint not found: $PHASE2_CKPT"
        echo "       Train Phase2 first with: python gram_v2_train_phase2.py"
        exit 1
    fi
    python eval_detection_comparison.py \
        --data        "$DATA_DIR"    \
        --phase1-ckpt "__skip__"     \
        --phase1-save "$PHASE1_SAVE" \
        --phase2-ckpt "$PHASE2_CKPT" \
        --phase2-save "$PHASE2_SAVE" \
        --force-eval  \
        --save-results \
        --no-latex     \
        $NO_CUDA
    echo "Phase2 results saved to: $PHASE2_SAVE/phase2_results.pt"
fi

echo ""
echo "Perception eval complete."
echo "Now run:  bash scripts/run_dbscan_comparison.sh"
