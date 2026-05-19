#!/bin/bash
# Run test.py for all policies in group environment with Social Force pedestrians.
# Groups present, realistic phases A-E on, human policy = social_force.
# TAGA versions included for classical and learning-based policies (not GRACE).
# All runs use seed=425 → same 100 episodes for every model.
# Logs saved to results/test_sf_group/

set -e
OUTDIR="results/test_sf_group"
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
        --sf_humans \
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
run_test "intention_rl"      trained_models/GST_predictor_rand 41665.pt ""
run_test "intention_rl_taga" trained_models/GST_predictor_rand 41665.pt "--group_avoid"

# ── GRACE (ours) — no TAGA version ─────────────────────────────────────
run_test "grace"             trained_models/gram_map/stageC    best.pt  ""

echo "============================================"
echo "All tests complete. Logs in $OUTDIR/"
echo "============================================"

# Quick SR summary
echo ""
echo "=== SR Summary (groups present, social force pedestrians) ==="
for f in "$OUTDIR"/*.log; do
    label=$(basename "$f" .log)
    sr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "Testing success rate: \K[0-9.]+")
    cr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "collision rate: \K[0-9.]+")
    tr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "timeout rate: \K[0-9.]+")
    gcr=$(grep "Testing success rate" "$f" | tail -1 | grep -oP "GCR\): \K[0-9.]+")
    echo "$label: SR=$sr  CR=$cr  TR=$tr  GCR=${gcr}%"
done
