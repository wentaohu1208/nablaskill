#!/bin/bash
# Hyperparameter sweep for TTSO DTO (~1000 configs).
#
# Tier 1: Core grid (lr × scale × iters)                      — 256 configs
# Tier 2a: Sweep response_nll_coeff                            — 160 configs
# Tier 2b: Sweep skill_fluency_coeff                           — 180 configs
# Tier 2c: Sweep reward_coeff                                  — 160 configs
# Tier 3: Combined extreme loss configs                        — 244 configs
#
# Usage:
#   bash scripts/sweep_hyperparams.sh
#   bash scripts/sweep_hyperparams.sh cuda:7
#
set -euo pipefail

DEVICE="${1:-cuda:6}"
OUTDIR="outputs/sweep_$(date +%Y%m%d_%H%M%S)"
SCRIPT="scripts/example_physics.py"

mkdir -p "$OUTDIR"

COMMON_ARGS="--device $DEVICE --optimization_mode dto --max_outer_rounds 3 --seed 42 --verbose 2"

declare -A CONFIGS

# ============================================================================
# Tier 1: Core grid — lr × init_logit_scale × max_iters
# Default loss coefficients, full factorial
# 8 lr × 8 scale × 4 iters = 256 configs
# ============================================================================
for LR in 0.001 0.005 0.01 0.05 0.1 0.3 0.5 1.0; do
  for SCALE in 0.1 0.5 1.0 2.0 3.0 5.0 7.0 10.0; do
    for ITERS in 50 100 200 500; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t1_lr${LR}_scale${SCALE_TAG}_iter${ITERS}"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters $ITERS"
    done
  done
done

# ============================================================================
# Tier 2a: response_nll_coeff sweep
# Fix a subset of (lr, scale, iters), sweep nll_coeff
# 8 nll × 5 base_combos × 4 lr_groups = 160 configs
# ============================================================================
# Helper: pick (scale, iters) combos for each lr range
# Low lr (0.001, 0.005): scale in {1.0, 3.0, 5.0}, iters=200
# Mid lr (0.01, 0.05):   scale in {0.5, 2.0, 3.0}, iters=100
# High lr (0.1, 0.3, 0.5, 1.0): scale in {1.0, 2.0}, iters=50
for NLL in 0 1e-05 0.0001 0.001 0.005 0.01 0.05 0.1; do
  NLL_TAG=$(echo "$NLL" | tr '-' 'n' | tr '.' 'p')
  for LR in 0.001 0.005; do
    for SCALE in 1.0 3.0 5.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2a_nll${NLL_TAG}_lr${LR}_scale${SCALE_TAG}_iter200"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 200 --response_nll_coeff $NLL"
    done
  done
  for LR in 0.01 0.05; do
    for SCALE in 0.5 2.0 3.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2a_nll${NLL_TAG}_lr${LR}_scale${SCALE_TAG}_iter100"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 100 --response_nll_coeff $NLL"
    done
  done
  for LR in 0.1 0.3 0.5 1.0; do
    for SCALE in 1.0 2.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2a_nll${NLL_TAG}_lr${LR}_scale${SCALE_TAG}_iter50"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 50 --response_nll_coeff $NLL"
    done
  done
done

# ============================================================================
# Tier 2b: skill_fluency_coeff sweep
# Same base combos as 2a, plus more fluency values
# 9 fluency × 5 base_combos × 4 lr_groups = 180 configs
# ============================================================================
for FLU in 0 1e-05 0.0001 0.001 0.01 0.05 0.1 0.5 1.0; do
  FLU_TAG=$(echo "$FLU" | tr '-' 'n' | tr '.' 'p')
  for LR in 0.001 0.005; do
    for SCALE in 1.0 3.0 5.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2b_flu${FLU_TAG}_lr${LR}_scale${SCALE_TAG}_iter200"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 200 --skill_fluency_coeff $FLU"
    done
  done
  for LR in 0.01 0.05; do
    for SCALE in 0.5 2.0 3.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2b_flu${FLU_TAG}_lr${LR}_scale${SCALE_TAG}_iter100"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 100 --skill_fluency_coeff $FLU"
    done
  done
  for LR in 0.1 0.3 0.5 1.0; do
    for SCALE in 1.0 2.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2b_flu${FLU_TAG}_lr${LR}_scale${SCALE_TAG}_iter50"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 50 --skill_fluency_coeff $FLU"
    done
  done
done

# ============================================================================
# Tier 2c: reward_coeff sweep
# Same base combos as 2a
# 8 reward × 5 base_combos × 4 lr_groups = 160 configs
# ============================================================================
for REW in 0 0.01 0.05 0.1 0.5 1.0 2.0 5.0; do
  REW_TAG=$(echo "$REW" | tr '.' 'p')
  for LR in 0.001 0.005; do
    for SCALE in 1.0 3.0 5.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2c_rew${REW_TAG}_lr${LR}_scale${SCALE_TAG}_iter200"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 200 --reward_coeff $REW"
    done
  done
  for LR in 0.01 0.05; do
    for SCALE in 0.5 2.0 3.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2c_rew${REW_TAG}_lr${LR}_scale${SCALE_TAG}_iter100"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 100 --reward_coeff $REW"
    done
  done
  for LR in 0.1 0.3 0.5 1.0; do
    for SCALE in 1.0 2.0; do
      SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
      NAME="t2c_rew${REW_TAG}_lr${LR}_scale${SCALE_TAG}_iter50"
      CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters 50 --reward_coeff $REW"
    done
  done
done

# ============================================================================
# Tier 3: Combined extreme loss configs
# 16 loss combos × (varying base params) ≈ 244 configs
# ============================================================================

# Loss combo presets: "name nll_coeff fluency_coeff reward_coeff"
LOSS_PRESETS=(
  "pure_reward        0       0     1.0"
  "no_fluency         0.001   0     1.0"
  "no_nll             0       0.05  1.0"
  "no_reward          0.001   0.05  0"
  "no_loss            0       0     0"
  "balanced_low       0.1     0.1   0.1"
  "reward_heavy       0.01    0.01  2.0"
  "fluency_heavy      0.01    0.5   0.5"
  "nll_heavy          0.1     0.05  1.0"
  "extreme_fluency    0.001   1.0   1.0"
  "nll_reward_only    0.1     0     2.0"
  "flu_reward_only    0       0.5   2.0"
  "all_small          0.01    0.01  0.01"
  "all_medium         0.5     0.5   0.5"
  "high_rew_balanced  0.1     0.1   2.0"
  "extreme_reward     0.001   0     5.0"
)

# Base param combos for Tier 3: "lr scale iters"
T3_BASES=(
  "0.001 0.5 100"
  "0.001 2.0 100"
  "0.001 5.0 100"
  "0.005 1.0 100"
  "0.005 3.0 100"
  "0.01  0.5 100"
  "0.01  2.0 100"
  "0.01  3.0 100"
  "0.01  5.0 100"
  "0.05  1.0 100"
  "0.05  2.0 100"
  "0.1   1.0 50"
  "0.3   1.0 50"
  "0.5   2.0 50"
  "1.0   2.0 50"
)

# Not all (preset × base) combinations — cycle through bases for each preset
# to keep total near 244: ~15 bases × 16 presets = 240, then add a few extras
for PRESET in "${LOSS_PRESETS[@]}"; do
  read -r PNAME NLL FLU REW <<< "$PRESET"
  for BASE in "${T3_BASES[@]}"; do
    read -r LR SCALE ITERS <<< "$BASE"
    SCALE_TAG=$(echo "$SCALE" | tr '.' '_')
    NAME="t3_${PNAME}_lr${LR}_scale${SCALE_TAG}_iter${ITERS}"
    CONFIGS["$NAME"]="--lr $LR --init_logit_scale $SCALE --max_iters $ITERS --response_nll_coeff $NLL --skill_fluency_coeff $FLU --reward_coeff $REW"
  done
done

# A few extra extreme combos with high lr + high iters
for PRESET in "pure_reward 0 0 1.0" "extreme_reward 0.001 0 5.0" "no_loss 0 0 0" "nll_heavy 0.1 0.05 1.0"; do
  read -r PNAME NLL FLU REW <<< "$PRESET"
  NAME="t3_${PNAME}_lr0.05_scale2_0_iter200"
  CONFIGS["$NAME"]="--lr 0.05 --init_logit_scale 2.0 --max_iters 200 --response_nll_coeff $NLL --skill_fluency_coeff $FLU --reward_coeff $REW"
done

# ============================================================================
# Run sweep
# ============================================================================

echo "============================================"
echo "DTO Hyperparameter Sweep"
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
