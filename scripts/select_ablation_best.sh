#!/usr/bin/env bash
# Usage: bash scripts/select_ablation_best.sh <model_dir> <ckpt1.pt> [ckpt2.pt ckpt3.pt ...]
# Example:
#   bash scripts/select_ablation_best.sh trained_models/gram_map/ablation_C1_no_group 5200.pt 5400.pt 5600.pt
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${1:?Usage: $0 <model_dir> <ckpt1.pt> [ckpt2.pt ...]}"
shift
CHECKPOINTS=("$@")

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: provide at least one checkpoint file."
    exit 1
fi

RESULTS_FILE="results/ablation_results.txt"
mkdir -p results

# Create results file with header if missing
if [ ! -f "$RESULTS_FILE" ]; then
    cat > "$RESULTS_FILE" << 'EOF'
================================================================================
GRACE — Ablation Study Results
================================================================================
Last updated: (auto-filled on each run)

Method                              SR     CR     TR     GCR       MeanReward   BestCkpt
────────────────────────────────────────────────────────────────────────────────────────
GRACE full (stageC/best.pt)        0.92   0.03   0.05   0.07%     32.70        best.pt
────────────────────────────────────────────────────────────────────────────────────────
EOF
fi

ABLATION_NAME=$(basename "$MODEL_DIR")

echo "================================================================"
echo "Ablation: $ABLATION_NAME"
echo "Model dir: $MODEL_DIR"
echo "Candidates: ${CHECKPOINTS[*]}"
echo "================================================================"

BEST_CKPT=""
BEST_SR=-1
BEST_CR="—"
BEST_TR="—"
BEST_GCR="—"
BEST_REWARD="—"

for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_PATH="$MODEL_DIR/checkpoints/$CKPT"
    if [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: $CKPT_PATH not found — skipping."
        continue
    fi

    echo ""
    echo "──── Testing $CKPT ────"
    python test.py --model_dir "$MODEL_DIR" --test_model "$CKPT"

    LOG_FILE="$MODEL_DIR/test/test_${CKPT}.log"
    if [ ! -f "$LOG_FILE" ]; then
        echo "WARNING: log file $LOG_FILE not found — skipping."
        continue
    fi

    # Parse metrics from log (format: "..., INFO: Testing success rate: X.XX, collision rate: ...")
    STATS_LINE=$(grep "Testing success rate:" "$LOG_FILE" | tail -1)
    SR=$(echo "$STATS_LINE"   | sed 's/.*Testing success rate: \([0-9.]*\).*/\1/')
    CR=$(echo "$STATS_LINE"   | sed 's/.*collision rate: \([0-9.]*\).*/\1/')
    TR=$(echo "$STATS_LINE"   | sed 's/.*timeout rate: \([0-9.]*\).*/\1/')
    GCR=$(echo "$STATS_LINE"  | sed 's/.*group intrusion rate (GCR): \([0-9.]*\)%.*/\1/')
    REWARD=$(grep "Mean Reward:" "$LOG_FILE" | tail -1 | sed 's/.*Mean Reward: \([0-9.-]*\).*/\1/')

    echo "  → SR=$SR  CR=$CR  TR=$TR  GCR=${GCR}%  Reward=$REWARD"

    # Float comparison via python
    IS_BETTER=$(python3 -c "print('yes' if float('${SR:-0}') > float('$BEST_SR') else 'no')")
    if [ "$IS_BETTER" = "yes" ]; then
        BEST_SR="$SR"
        BEST_CKPT="$CKPT"
        BEST_CR="$CR"
        BEST_TR="$TR"
        BEST_GCR="$GCR"
        BEST_REWARD="$REWARD"
    fi
done

if [ -z "$BEST_CKPT" ]; then
    echo ""
    echo "ERROR: no valid checkpoints were found or tested."
    exit 1
fi

echo ""
echo "================================================================"
echo "WINNER: $BEST_CKPT  →  SR=$BEST_SR  CR=$BEST_CR  TR=$BEST_TR  GCR=${BEST_GCR}%  Reward=$BEST_REWARD"
echo "================================================================"

# Copy winner to best.pt
cp "$MODEL_DIR/checkpoints/$BEST_CKPT" "$MODEL_DIR/checkpoints/best.pt"
echo "Saved: $MODEL_DIR/checkpoints/best.pt"

# Update timestamp in results file
TODAY=$(date +%Y-%m-%d)
sed -i "s/Last updated: .*/Last updated: $TODAY/" "$RESULTS_FILE"

# Append result row (skip if this ablation is already recorded)
if grep -q "$ABLATION_NAME" "$RESULTS_FILE"; then
    echo "NOTE: $ABLATION_NAME already in $RESULTS_FILE — skipping duplicate entry."
else
    printf "%-35s %-6s %-6s %-6s %-9s %-12s %s\n" \
        "$ABLATION_NAME" "$BEST_SR" "$BEST_CR" "$BEST_TR" "${BEST_GCR}%" "$BEST_REWARD" "$BEST_CKPT" \
        >> "$RESULTS_FILE"
fi

echo ""
echo "Results saved to $RESULTS_FILE"
echo "────────────────────────────────────────────────────────────────"
cat "$RESULTS_FILE"
