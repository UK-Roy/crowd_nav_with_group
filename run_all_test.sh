#!/bin/bash
# Run test.py for all policies (with and without TAGA) and collect logs.
# All runs use seed=425 (default) → same 100 episodes for every model.
# Logs saved to results/test_benchmark/

set -e
OUTDIR="results/test_benchmark"
mkdir -p "$OUTDIR"

run_test() {
    local label=$1
    local model_dir=$2
    local test_model=$3
    local extra=$4
    echo "============================================"
    echo "Running: $label"
    echo "============================================"
    python test.py \
        --model_dir "$model_dir" \
        --test_model "$test_model" \
        $extra \
        2>&1 | tee "$OUTDIR/${label}.log"
    echo "Done: $label"
    echo ""
}

# ── Classical ──────────────────────────────────────────────────────────
run_test "orca"              trained_models/ORCA_no_rand       00000.pt ""
run_test "orca_taga"         trained_models/ORCA_no_rand       00000.pt "--group_avoid"
run_test "social_force"      trained_models/SF_no_rand         00000.pt ""
run_test "social_force_taga" trained_models/SF_no_rand         00000.pt "--group_avoid"

# ── Learning-based ─────────────────────────────────────────────────────
run_test "intention_rl"      trained_models/intention_rl_realistic  41665.pt ""
run_test "intention_rl_taga" trained_models/intention_rl_realistic  41665.pt "--group_avoid"

# ── GRACE (ours) ───────────────────────────────────────────────────────
run_test "grace"             trained_models/gram_map/stageC    best.pt  ""

echo "============================================"
echo "All tests complete. Logs in $OUTDIR/"
echo "============================================"

# Quick SR summary
echo ""
echo "=== SR Summary ==="
for f in "$OUTDIR"/*.log; do
    label=$(basename "$f" .log)
    sr=$(grep "Testing success rate" "$f" | tail -1 | awk '{print $NF}')
    cr=$(grep "Collision cases" "$f" | tail -1 | awk '{print $NF}')
    echo "$label: SR=$sr  Collisions=$cr"
done
