#!/bin/bash
# Task 2: Heuristic vs Random comparison on multi-scene multi-seed real probes.
# Runs random_active for seeds that have heuristic_active but not random_active results.
set -euo pipefail

CONDA_ENV=memoryguard-ai2thor
BASE="/home/jxy/MemoryGuard-v2"
RESULTS="${BASE}/results"

declare -A SEEDS_BY_SCENE
SEEDS_BY_SCENE[FloorPlan3]="7 11 29 37 67 72"
SEEDS_BY_SCENE[FloorPlan1]="29 37 41"
SEEDS_BY_SCENE[FloorPlan201]="29 37 41"

for SCENE in FloorPlan3 FloorPlan1 FloorPlan201; do
    for SEED in ${SEEDS_BY_SCENE[$SCENE]}; do
        PROBE="${RESULTS}/ai2thor_rearrangement_${SCENE}_seed${SEED}_capture/ai2thor_rearrangement_probe.json"
        if [ ! -f "$PROBE" ]; then
            PROBE="${RESULTS}/ai2thor_rearrangement_${SCENE}_seed${SEED}/ai2thor_rearrangement_probe.json"
        fi
        if [ ! -f "$PROBE" ]; then
            echo "SKIP ${SCENE} seed${SEED}: no probe found"
            continue
        fi
        OUT="${RESULTS}/ai2thor_live_maintenance_${SCENE,,}_seed${SEED}_random_active"
        if [ -d "$OUT" ] && [ -f "$OUT/live_maintenance_smoke.csv" ]; then
            echo "SKIP ${SCENE} seed${SEED}: already exists at $OUT"
            continue
        fi
        echo "RUN ${SCENE} seed${SEED} random_active -> $OUT"
        conda run --no-capture-output -n "$CONDA_ENV" python -m embodied_memory_pilot.ai2thor_live_maintenance \
            --probe "$PROBE" \
            --scene "$SCENE" \
            --platform CloudRendering \
            --random-seed "$SEED" \
            --maintenance-policy random_active \
            --maintenance-budget-per-task 1 \
            --verifier-backend crop_proxy \
            --out-dir "$OUT" \
            --task-budget 10 \
            2>&1 | tail -5
    done
done

echo "DONE: All random_active runs complete."
# TODO: Run comparison analysis script
