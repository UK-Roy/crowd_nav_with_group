#!/bin/bash
# Run test.py for all policies in individuals-only realistic environment (no groups).
# Realistic phases A-E remain on; groups disabled via --no_groups flag.
# All runs use seed=425 → same 100 episodes for every model.
# Logs saved to results/test_no_group/

set -e
OUTDIR="results/test_no_group"
mkdir -p "$OUTDIR"

run_test() {
    local label=$1
    local model_dir=$2
    local test_model=$3
    echo "============================================"
    echo "Running: $label"
    echo "============================================"
    python test.py \
        --model_dir "$model_dir" \
        --test_model "$test_model" \
        --no_groups \
        2>&1 | tee "$OUTDIR/${label}.log"
    echo "Done: $label"
    echo ""
}

# ── Classical ──────────────────────────────────────────────────────────
run_test "orca"         trained_models/ORCA_no_rand      00000.pt
run_test "social_force" trained_models/SF_no_rand        00000.pt

# ── Learning-based ─────────────────────────────────────────────────────
run_test "intention_rl" trained_models/GST_predictor_rand 41665.pt

# ── GRACE (ours) ───────────────────────────────────────────────────────
run_test "grace"        trained_models/gram_map/stageC   best.pt

echo "============================================"
echo "All tests complete. Logs in $OUTDIR/"
echo "============================================"

# Quick SR summary
echo ""
echo "=== SR Summary (individuals-only realistic env) ==="
for f in "$OUTDIR"/*.log; do
    label=$(basename "$f" .log)
    sr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "Testing success rate: \K[0-9.]+")
    cr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "collision rate: \K[0-9.]+")
    tr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "timeout rate: \K[0-9.]+")
    echo "$label: SR=$sr  CR=$cr  TR=$tr"
done
