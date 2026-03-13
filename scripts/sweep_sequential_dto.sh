#!/bin/bash
# Hyperparameter sweep for Sequential DTO.
#
# Sequential DTO 的计算量 = num_skill_tokens × max_iters_per_pos × num_outer_rounds
# (e.g. 180 tokens × 20 iters × 3 rounds = 10800 步), 所以 max_iters 不宜过大。
#
# Tier 1: Core grid (lr × scale × iters_per_pos)              — 80 configs
# Tier 2: commit_every sweep                                    — 48 configs
# Tier 3: Loss coefficient sweep (nll × fluency × reward)      — 80 configs
# Tier 4: Outer rounds sweep                                    — 30 configs
# Total: ~238 configs
#
# Usage:
#   bash scripts/sweep_sequential_dto.sh
#   bash scripts/sweep_sequential_dto.sh cuda:7
#
set -euo pipefail

DEVICE="${1:-cuda:6}"
OUTDIR="outputs/sweep_seqdto_$(date +%Y%m%d_%H%M%S)"
SCRIPT="scripts/example_physics.py"

mkdir -p "$OUTDIR"

COMMON_ARGS="--device $DEVICE --optimization_mode sequential_dto --max_outer_rounds 3 --seed 42 --verbose 2"

declare -A CONFIGS

# ============================================================================
# Tier 1: Core grid — lr × init_logit_scale × max_iters_per_pos
# commit_every=1 (default), default loss coefficients
# 4 lr × 5 scale × 4 iters = 80 configs
# ============================================================================
for LR in 0.001 0.005 0.01 0.05; do
  for SCALE in 0.5 1.0 2.0 3.0 5.0; do
    for ITERS in 5 10 20 50; do
      # Sanitize scale for config name (remove dot)
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t1_lr${LR}_scale${SCALE_TAG}_iter${ITERS}"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters $ITERS"
    done
  done
done

# ============================================================================
# Tier 2: commit_every sweep
# Fix a few (lr, scale, iters) combos, sweep commit_every
# 4 commit × 3 lr × 2 scale × 2 iters = 48 configs
# ============================================================================
for COMMIT in 3 5 10 20; do
  for LR in 0.005 0.01 0.05; do
    for SCALE in 1.0 3.0; do
      for ITERS in 10 20; do
        SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
        NAME="t2_commit${COMMIT}_lr${LR}_scale${SCALE_TAG}_iter${ITERS}"
        CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters $ITERS --sequential_commit_every $COMMIT"
      done
    done
  done
done

# ============================================================================
# Tier 3: Loss coefficient sweep
# Fix lr=0.01, scale=2.0, iters=20, commit=1
# 4 nll × 5 fluency × 4 reward = 80 configs
# ============================================================================
for NLL in 1e-4 1e-3 1e-2 0.1; do
  for FLUENCY in 0 1e-4 1e-3 1e-2 0.1; do
    for REWARD in 0.1 0.5 1.0 2.0; do
      NLL_TAG=$(echo "$NLL" | tr '-' 'n' | tr '.' 'p')
      FLUENCY_TAG=$(echo "$FLUENCY" | tr '-' 'n' | tr '.' 'p')
      REWARD_TAG=$(echo "$REWARD" | tr '.' 'p')
      NAME="t3_nll${NLL_TAG}_flu${FLUENCY_TAG}_rew${REWARD_TAG}"
      CONFIGS["$NAME"]="--lr 0.01 --init_logit_scale 2.0 --max_iters 20 --response_nll_coeff $NLL --skill_fluency_coeff $FLUENCY --reward_coeff $REWARD"
    done
  done
done

# ============================================================================
# Tier 4: Outer rounds sweep
# Fix loss coefficients, sweep outer_rounds × a few inner combos
# 5 rounds × 6 inner combos = 30 configs
# ============================================================================
for ROUNDS in 1 2 3 5 7; do
  for COMBO in \
    "0.01 2.0 20" \
    "0.01 3.0 20" \
    "0.005 1.0 20" \
    "0.05 2.0 10" \
    "0.01 2.0 50" \
    "0.05 3.0 20"; do
    LR=$(echo "$COMBO" | awk '{print $1}')
    SCALE=$(echo "$COMBO" | awk '{print $2}')
    ITERS=$(echo "$COMBO" | awk '{print $3}')
    SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
    NAME="t4_rounds${ROUNDS}_lr${LR}_scale${SCALE_TAG}_iter${ITERS}"
    CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters $ITERS --max_outer_rounds $ROUNDS"
  done
done

# ============================================================================
# Run sweep
# ============================================================================

echo "============================================"
echo "Sequential DTO Sweep"
echo "Output dir: $OUTDIR"
echo "Device: $DEVICE"
echo "Total configs: ${#CONFIGS[@]}"
echo "============================================"

cat > "$OUTDIR/sweep_meta.txt" << EOF
Sweep started: $(date)
Device: $DEVICE
Common args: $COMMON_ARGS
Total configs: ${#CONFIGS[@]}
EOF

RUN_IDX=0
TOTAL=${#CONFIGS[@]}

for NAME in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    EXTRA="${CONFIGS[$NAME]}"
    RUN_IDX=$((RUN_IDX + 1))
    OUTFILE="$OUTDIR/${NAME}.txt"

    echo ""
    echo "[$RUN_IDX/$TOTAL] Running: $NAME"
    echo "  Args: $COMMON_ARGS $EXTRA"
    echo "  Output: $OUTFILE"

    cat > "$OUTFILE" << EOF
========================================
Config: $NAME
Args: $COMMON_ARGS $EXTRA
Started: $(date)
========================================

EOF

    if python "$SCRIPT" $COMMON_ARGS $EXTRA >> "$OUTFILE" 2>&1; then
        echo "  Status: SUCCESS"
        echo "" >> "$OUTFILE"
        echo "Status: SUCCESS" >> "$OUTFILE"
    else
        EXIT_CODE=$?
        echo "  Status: FAILED (exit code $EXIT_CODE)"
        echo "" >> "$OUTFILE"
        echo "Status: FAILED (exit code $EXIT_CODE)" >> "$OUTFILE"
    fi

    echo "Finished: $(date)" >> "$OUTFILE"
done

echo ""
echo "============================================"
echo "Sweep complete! Results in: $OUTDIR/"
echo "============================================"
echo ""

# Summary table
echo "SUMMARY"
echo "----------------------------------------------------------------------"
printf "%-55s %10s %10s %10s\n" "Config" "RM_orig" "RM_final" "Delta"
echo "----------------------------------------------------------------------"
for NAME in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    FILE="$OUTDIR/${NAME}.txt"
    if [ -f "$FILE" ]; then
        RM_ORIG=$(grep "RM (orig):" "$FILE" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
        RM_FINAL=$(grep "RM (final):" "$FILE" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
        DELTA=$(grep "Delta:" "$FILE" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
        printf "%-55s %10s %10s %10s\n" "$NAME" "$RM_ORIG" "$RM_FINAL" "$DELTA"
    fi
done

# Save summary to file
{
    echo "SUMMARY"
    echo "----------------------------------------------------------------------"
    printf "%-55s %10s %10s %10s\n" "Config" "RM_orig" "RM_final" "Delta"
    echo "----------------------------------------------------------------------"
    for NAME in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
        FILE="$OUTDIR/${NAME}.txt"
        if [ -f "$FILE" ]; then
            RM_ORIG=$(grep "RM (orig):" "$FILE" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
            RM_FINAL=$(grep "RM (final):" "$FILE" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
            DELTA=$(grep "Delta:" "$FILE" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
            printf "%-55s %10s %10s %10s\n" "$NAME" "$RM_ORIG" "$RM_FINAL" "$DELTA"
        fi
    done
} > "$OUTDIR/summary.txt"

echo ""
echo "Summary saved to: $OUTDIR/summary.txt"
