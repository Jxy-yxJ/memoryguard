#!/bin/bash
# B3: Run GSAM verification with before-frame DINO fix on ALL seeds
# Usage: bash run_b3_remaining_seeds.sh
set -euo pipefail

# NOTE: This script is kept for reference. Most seeds have already been collected
# using the FloorPlan3 capture dirs. See results/ai2thor_grounded_sam2_floorplan3_*_real_semantic_fix/
# For new seeds/scenes, collect fresh captures first via:
#   conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_rearrangement_benchmark --collect-scenes ...
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CKPT_DIR="${CKPT_DIR:-$REPO/checkpoints}"
BASE_CACHE="${BASE_CACHE:-$HOME/.cache/memoryguard-grounded-sam2-src-unpacked2}"
CONDA_ENV="${CONDA_ENV:-memoryguard-grounded-sam2}"
GD_CONFIG="$BASE_CACHE/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CKPT="$CKPT_DIR/groundingdino_swint_ogc.pth"
SAM2_CKPT="$CKPT_DIR/sam2.1_hiera_tiny.pt"

export HF_HUB_OFFLINE=1
TIMEOUT=600  # 10 minutes per seed

SEEDS=(31 37 41 43 67 71 73 79 83)

for seed in "${SEEDS[@]}"; do
    CAPTURE_DIR="$REPO/results/ai2thor_rearrangement_floorplan3_seed${seed}_capture"
    OUT_DIR="$REPO/results/ai2thor_grounded_sam2_floorplan3_seed${seed}_beforedino"

    if [[ ! -d "$CAPTURE_DIR" ]]; then
        echo "===== SEED $seed SKIPPED (no capture dir) ====="
        continue
    fi

    echo "===== SEED $seed ====="
    echo "CAPTURE: $CAPTURE_DIR"
    echo "OUT: $OUT_DIR"

    timeout $TIMEOUT conda run -n memoryguard-grounded-sam2 python -m embodied_memory_pilot.ai2thor_grounded_sam2_verifier \
        --probe "$CAPTURE_DIR/ai2thor_rearrangement_probe.json" \
        --image-dir "$CAPTURE_DIR/images" \
        --backend grounded-sam2 \
        --grounding-config "$GD_CONFIG" \
        --grounding-checkpoint "$GD_CKPT" \
        --sam2-config sam2.1_hiera_t \
        --sam2-checkpoint "$SAM2_CKPT" \
        --device cuda \
        --budgets 16 \
        --location-thresholds 0.02 0.05 0.1 \
        --out-dir "$OUT_DIR" \
        2>&1 | tail -5

    echo "===== SEED $seed DONE ====="
done

echo "ALL DONE"
