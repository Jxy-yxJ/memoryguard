"""Bridge GSAM post-hoc verification results into multiseed aggregator format.

Reads per-seed GSAM JSONs, extracts Oracle and GSAM metrics at budget=16
for threshold=0.02, and writes per-seed CSVs that the multiseed aggregator
can consume. Each per-seed CSV contains TWO rows:
  - policy="verify_rearrangement_risk" (Oracle)
  - policy="gsam_location" (Grounded-SAM2)

Then run:
    python -m embodied_memory_pilot.ai2thor_rearrangement_multiseed \
        results/ai2thor_gsam_bridge_csvs/seed029_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed031_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed037_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed041_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed043_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed067_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed071_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed073_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed079_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed083_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed101_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed103_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed107_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed109_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed113_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed127_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed131_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed137_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed139_bridge.csv \
        results/ai2thor_gsam_bridge_csvs/seed149_bridge.csv \
        --out-dir results/ai2thor_multiseed_gsam_20seed

to produce Oracle vs GSAM comparison table.
"""

import csv
import json
import sys
from pathlib import Path
from typing import cast

SEEDS = [
    29, 31, 37, 41, 43,
    67, 71, 73, 79, 83,
    101, 103, 107, 109, 113,
    127, 131, 137, 139, 149,
]

BUDGET = 16
THRESHOLD = 0.02

OUTPUT_DIR = Path("results/ai2thor_gsam_bridge_csvs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JsonRow = dict[str, object]


def to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Cannot convert {value!r} to int")


def to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Cannot convert {value!r} to float")


def read_gsam_json(seed: int) -> JsonRow:
    path = Path(f"results/ai2thor_grounded_sam2_location_seed{seed}/grounded_sam2_location_vs_oracle.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing GSAM JSON: {path}")
    with open(path) as f:
        return cast(JsonRow, json.load(f))


def find_best_row(data: JsonRow) -> JsonRow:
    rows = cast(list[JsonRow], data["results"])
    for row in rows:
        budget = to_int(row["budget"])
        threshold = to_float(row["location_threshold"])
        if budget == BUDGET and abs(threshold - THRESHOLD) < 0.001:
            return row
    raise ValueError(f"No row found for budget={BUDGET}, threshold={THRESHOLD}")


def write_per_seed_csv(seed: int, row: JsonRow, probe: JsonRow) -> Path:
    path = OUTPUT_DIR / f"seed{seed:03d}_bridge.csv"

    paired = to_int(probe.get("paired_objects", 0))
    moved = to_int(probe.get("moved_objects", 0))
    oracle_completion = to_float(row["oracle_completion_rate"])
    oracle_avg_cost = to_float(row["oracle_avg_cost"])
    oracle_stale_errors = to_int(row["oracle_stale_errors"])
    oracle_verifications = to_int(row["oracle_verifications"])
    oracle_catches = to_int(row["oracle_catches"])
    location_completion = to_float(row["location_completion_rate"])
    location_avg_cost = to_float(row["location_avg_cost"])
    location_stale_errors = to_int(row["location_stale_errors"])
    location_verifications = to_int(row["location_verifications"])
    location_catches = to_int(row["location_catches"])

    oracle_row: JsonRow = {
        "budget": BUDGET,
        "policy": "verify_rearrangement_risk",
        "rearrangement_tasks": paired,
        "moved_tasks": moved,
        "memory_items": paired,
        "completion_rate": oracle_completion,
        "avg_reachable_cost": oracle_avg_cost,
        "query_hit_rate": _compute_query_hit_rate(row, "oracle"),
        "stale_errors": oracle_stale_errors,
        "verifications": oracle_verifications,
        "verification_catches": oracle_catches,
    }

    gsam_row: JsonRow = {
        "budget": BUDGET,
        "policy": "gsam_location",
        "rearrangement_tasks": paired,
        "moved_tasks": moved,
        "memory_items": paired,
        "completion_rate": location_completion,
        "avg_reachable_cost": location_avg_cost,
        "query_hit_rate": _compute_query_hit_rate(row, "location"),
        "stale_errors": location_stale_errors,
        "verifications": location_verifications,
        "verification_catches": location_catches,
    }

    fieldnames = [
        "budget", "policy", "rearrangement_tasks", "moved_tasks", "memory_items",
        "completion_rate", "avg_reachable_cost", "query_hit_rate",
        "stale_errors", "verifications", "verification_catches",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(oracle_row)
        writer.writerow(gsam_row)

    return path


def _compute_query_hit_rate(row: JsonRow, prefix: str) -> float:
    """Compute query_hit_rate from verifications and catches."""
    verifications = to_int(row.get(f"{prefix}_verifications", 0))
    catches = to_int(row.get(f"{prefix}_catches", 0))
    if verifications == 0:
        return 0.0
    return catches / verifications


def main():
    written_paths: list[Path] = []
    for seed in SEEDS:
        try:
            data = read_gsam_json(seed)
            best = find_best_row(data)
            probe = cast(JsonRow, data.get("probe_summary", {}))
            path = write_per_seed_csv(seed, best, probe)
            completion = to_float(best["location_completion_rate"])
            oracle = to_float(best["oracle_completion_rate"])
            print(f"  seed {seed:3d}: Oracle={oracle:.4f}, GSAM={completion:.4f}, delta={oracle-completion:.4f} → {path}")
            written_paths.append(path)
        except FileNotFoundError as e:
            print(f"  seed {seed:3d}: SKIP — {e}", file=sys.stderr)
        except Exception as e:
            print(f"  seed {seed:3d}: ERROR — {e}", file=sys.stderr)

    print(f"\nWrote {len(written_paths)}/{len(SEEDS)} per-seed bridge CSVs to {OUTPUT_DIR.resolve()}")

    csvs = " ".join(str(path) for path in written_paths)
    print(f"\nRun multiseed aggregator:")
    print(f"  python -m embodied_memory_pilot.ai2thor_rearrangement_multiseed \\")
    print(f"      {csvs} \\")
    print(f"      --out-dir results/ai2thor_multiseed_gsam_20seed")


if __name__ == "__main__":
    main()
