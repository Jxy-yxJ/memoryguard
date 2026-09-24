from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    build_rearrangement_tasks,
    summarize_rearrangement_probe,
)
from embodied_memory_pilot.ai2thor_rearrangement_maintenance import (
    HeuristicActiveMaintenanceSelection,
    MetricRow,
    PassiveMaintenanceSelection,
    Probe,
    RandomActiveMaintenanceSelection,
    evaluate_policy_rearrangement_maintenance,
)


SummaryRow = dict[str, float | int | str]


@dataclass(frozen=True)
class ParsedArgs:
    random_seeds: list[int]
    budget: int
    maintenance_budget: int
    maintenance_cost: float
    out_dir: Path


def build_selection_pressure_probe(object_types: Sequence[str] = TARGET_OBJECTS) -> Probe:
    """Build a controlled saved-probe-like scene with many stale candidates.

    All target objects move, but the maintenance budget defaults to one item per
    task. This creates selection pressure: a policy must choose which stale
    memory to refresh before each query.
    """

    reachable_positions = [{"x": float(index), "y": 0.0, "z": 0.0} for index in range(12)]
    before_objects: list[dict[str, object]] = []
    after_objects: list[dict[str, object]] = []
    for index, object_type in enumerate(object_types):
        object_id = f"{object_type}|stress-{index}"
        before_objects.append(
            {
                "object_id": object_id,
                "object_type": object_type,
                "pickupable": True,
                "receptacle": False,
                "position": {"x": float(index + 1), "y": 1.0, "z": 0.0},
            }
        )
        after_objects.append(
            {
                "object_id": object_id,
                "object_type": object_type,
                "pickupable": True,
                "receptacle": False,
                "position": {"x": float(index + 6), "y": 1.0, "z": 0.0},
            }
        )

    return {
        "status": "ok",
        "mode": "synthetic_selection_pressure",
        "scenes": [
            {
                "scene": "SyntheticSelectionPressure",
                "status": "ok",
                "agent_position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "reachable_positions": reachable_positions,
                "before_visible_objects": before_objects,
                "after_visible_objects": after_objects,
            }
        ],
    }


def run_selection_pressure_stress(
    *,
    random_seeds: Iterable[int],
    budget: int = 5,
    maintenance_budget: int = 1,
    maintenance_cost: float = 0.5,
    object_types: Sequence[str] = TARGET_OBJECTS,
) -> tuple[Probe, list[MetricRow]]:
    probe = build_selection_pressure_probe(object_types)
    policies = (
        PassiveMaintenanceSelection(),
        RandomActiveMaintenanceSelection(),
        HeuristicActiveMaintenanceSelection(),
    )
    rows: list[MetricRow] = []
    for seed in random_seeds:
        for policy in policies:
            rows.append(
                evaluate_policy_rearrangement_maintenance(
                    probe,
                    policy=policy,
                    budget=budget,
                    maintenance_budget=maintenance_budget,
                    maintenance_cost=maintenance_cost,
                    random_seed=seed,
                )
            )
    return probe, rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_stress(rows: Sequence[MetricRow]) -> list[SummaryRow]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(str(row["maintenance_policy"]), []).append(row)

    summary: list[SummaryRow] = []
    for policy, group in sorted(grouped.items()):
        summary.append(
            {
                "maintenance_policy": policy,
                "runs": len(group),
                "completion_rate_mean": round(mean([float(row["completion_rate"]) for row in group]), 4),
                "avg_reachable_cost_mean": round(mean([float(row["avg_reachable_cost"]) for row in group]), 4),
                "stale_errors_mean": round(mean([float(row["stale_errors"]) for row in group]), 4),
                "maintenance_checks_mean": round(mean([float(row["maintenance_checks"]) for row in group]), 4),
                "maintenance_catches_mean": round(mean([float(row["maintenance_catches"]) for row in group]), 4),
            }
        )
    return summary


def selection_pressure_delta(summary: Sequence[SummaryRow]) -> SummaryRow:
    by_policy = {str(row["maintenance_policy"]): row for row in summary}
    heuristic = by_policy.get("heuristic_active")
    random_active = by_policy.get("random_active")
    passive = by_policy.get("passive")
    if heuristic is None or random_active is None or passive is None:
        raise ValueError("Stress summary requires passive, random_active, and heuristic_active rows.")
    return {
        "heuristic_vs_random_completion_delta": round(
            float(heuristic["completion_rate_mean"]) - float(random_active["completion_rate_mean"]), 4
        ),
        "heuristic_vs_random_stale_errors_delta": round(
            float(heuristic["stale_errors_mean"]) - float(random_active["stale_errors_mean"]), 4
        ),
        "heuristic_vs_passive_completion_delta": round(
            float(heuristic["completion_rate_mean"]) - float(passive["completion_rate_mean"]), 4
        ),
        "heuristic_vs_passive_stale_errors_delta": round(
            float(heuristic["stale_errors_mean"]) - float(passive["stale_errors_mean"]), 4
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        _ = path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in rows[0].keys()})


def write_outputs(probe: Probe, rows: Sequence[MetricRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_stress(rows)
    delta = selection_pressure_delta(summary)
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_stress_rows.csv", rows)
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_stress_summary.csv", summary)
    _ = (out_dir / "ai2thor_rearrangement_maintenance_stress.json").write_text(
        json.dumps(
            {
                "probe_summary": summarize_rearrangement_probe(probe),
                "task_count": len(build_rearrangement_tasks(probe)),
                "runs": list(rows),
                "summary": summary,
                "delta": delta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _ = (out_dir / "README.md").write_text(render_readme(probe, summary, delta), encoding="utf-8")


def render_readme(probe: Probe, summary: Sequence[SummaryRow], delta: SummaryRow) -> str:
    probe_summary = summarize_rearrangement_probe(probe)
    lines = [
        "# AI2-THOR Rearrangement Maintenance Selection-Pressure Stress",
        "",
        "Mode: `synthetic_selection_pressure`",
        f"Moved paired objects: `{probe_summary['moved_objects']}`",
        f"Heuristic vs random completion delta: `{delta['heuristic_vs_random_completion_delta']}`",
        f"Heuristic vs random stale-error delta: `{delta['heuristic_vs_random_stale_errors_delta']}`",
        "",
        "This is a controlled synthetic saved-probe-like stress benchmark. It creates multiple stale maintenance candidates with a maintenance budget of one item per task.",
        "It tests whether the tau-style heuristic can choose the task-relevant stale memory better than random under selection pressure.",
        "It does not replace the real saved-probe aggregate and does not constitute live closed-loop sensing evidence.",
        "",
        "## Summary",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['maintenance_policy']}`: completion `{row['completion_rate_mean']}`, stale errors `{row['stale_errors_mean']}`, avg cost `{row['avg_reachable_cost_mean']}`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> ParsedArgs:
    parser = argparse.ArgumentParser(description="Controlled stress test for proactive maintenance selection pressure.")
    _ = parser.add_argument("--random-seeds", type=int, nargs="+", default=list(range(10)))
    _ = parser.add_argument("--budget", type=int, default=5)
    _ = parser.add_argument("--maintenance-budget", type=int, default=1)
    _ = parser.add_argument("--maintenance-cost", type=float, default=0.5)
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_rearrangement_maintenance_stress"))
    namespace = parser.parse_args()
    return ParsedArgs(
        random_seeds=cast(list[int], namespace.random_seeds),
        budget=cast(int, namespace.budget),
        maintenance_budget=cast(int, namespace.maintenance_budget),
        maintenance_cost=cast(float, namespace.maintenance_cost),
        out_dir=cast(Path, namespace.out_dir),
    )


def main() -> None:
    args = parse_args()
    probe, rows = run_selection_pressure_stress(
        random_seeds=args.random_seeds,
        budget=args.budget,
        maintenance_budget=args.maintenance_budget,
        maintenance_cost=args.maintenance_cost,
    )
    write_outputs(probe, rows, args.out_dir)
    summary = summarize_stress(rows)
    print(
        json.dumps(
            {
                "probe_summary": summarize_rearrangement_probe(probe),
                "summary": summary,
                "delta": selection_pressure_delta(summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
