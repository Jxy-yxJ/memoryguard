from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast


CsvRow = dict[str, str]
SummaryRow = dict[str, float | int | str]


METRICS = (
    "completion_rate",
    "avg_reachable_cost",
    "query_hit_rate",
    "stale_errors",
    "maintenance_checks",
    "maintenance_catches",
    "saved_reachable_cost",
)

COUNT_FIELDS = (
    "rearrangement_tasks",
    "moved_tasks",
    "memory_items",
    "retained_items",
)


@dataclass(frozen=True)
class ParsedArgs:
    summary_paths: list[Path]
    out_dir: Path


def _read_rows(path: Path) -> list[CsvRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_summaries(summary_paths: Iterable[Path]) -> list[SummaryRow]:
    grouped: dict[tuple[int, int, str], list[CsvRow]] = {}
    for path in summary_paths:
        for row in _read_rows(path):
            key = (int(row["budget"]), int(row["maintenance_budget"]), row["maintenance_policy"])
            grouped.setdefault(key, []).append(row)

    rows: list[SummaryRow] = []
    for (budget, maintenance_budget, policy), group in sorted(grouped.items()):
        out: SummaryRow = {
            "budget": budget,
            "maintenance_budget": maintenance_budget,
            "maintenance_policy": policy,
            "seeds": len(group),
        }
        for field in COUNT_FIELDS:
            values = [float(row[field]) for row in group]
            out[f"{field}_mean"] = round(_mean(values), 4)
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = round(_mean(values), 4)
            out[f"{metric}_std"] = round(_sample_std(values), 4)
        rows.append(out)
    return rows


def delta_vs_passive(rows: Sequence[SummaryRow]) -> list[SummaryRow]:
    by_key = {
        (int(row["budget"]), int(row["maintenance_budget"]), str(row["maintenance_policy"])): row
        for row in rows
    }
    deltas: list[SummaryRow] = []
    for row in rows:
        policy = str(row["maintenance_policy"])
        if policy == "passive":
            continue
        budget = int(row["budget"])
        maintenance_budget = int(row["maintenance_budget"])
        baseline = by_key.get((budget, maintenance_budget, "passive"))
        if baseline is None:
            continue
        deltas.append(
            {
                "budget": budget,
                "maintenance_budget": maintenance_budget,
                "method_policy": policy,
                "baseline_policy": "passive",
                "seeds": min(int(row["seeds"]), int(baseline["seeds"])),
                "completion_rate_delta": round(float(row["completion_rate_mean"]) - float(baseline["completion_rate_mean"]), 4),
                "avg_reachable_cost_delta": round(float(row["avg_reachable_cost_mean"]) - float(baseline["avg_reachable_cost_mean"]), 4),
                "stale_errors_delta": round(float(row["stale_errors_mean"]) - float(baseline["stale_errors_mean"]), 4),
                "maintenance_checks_delta": round(float(row["maintenance_checks_mean"]) - float(baseline["maintenance_checks_mean"]), 4),
                "maintenance_catches_delta": round(float(row["maintenance_catches_mean"]) - float(baseline["maintenance_catches_mean"]), 4),
            }
        )
    return deltas


def pareto_front(rows: Sequence[SummaryRow]) -> list[SummaryRow]:
    front: list[SummaryRow] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                float(other["completion_rate_mean"]) >= float(row["completion_rate_mean"])
                and float(other["avg_reachable_cost_mean"]) <= float(row["avg_reachable_cost_mean"])
                and float(other["stale_errors_mean"]) <= float(row["stale_errors_mean"])
                and float(other["maintenance_checks_mean"]) <= float(row["maintenance_checks_mean"])
            )
            strictly_better = (
                float(other["completion_rate_mean"]) > float(row["completion_rate_mean"])
                or float(other["avg_reachable_cost_mean"]) < float(row["avg_reachable_cost_mean"])
                or float(other["stale_errors_mean"]) < float(row["stale_errors_mean"])
                or float(other["maintenance_checks_mean"]) < float(row["maintenance_checks_mean"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        _ = path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in rows[0].keys()})


def write_outputs(rows: Sequence[SummaryRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = delta_vs_passive(rows)
    front = pareto_front(rows)
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_multiseed_summary.csv", rows)
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_multiseed_delta.csv", deltas)
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_multiseed_pareto.csv", front)
    _ = (out_dir / "ai2thor_rearrangement_maintenance_multiseed.json").write_text(
        json.dumps({"summary": list(rows), "delta_vs_passive": deltas, "pareto_front": front}, indent=2),
        encoding="utf-8",
    )
    _ = (out_dir / "README.md").write_text(render_readme(rows, deltas, front), encoding="utf-8")


def render_readme(
    rows: Sequence[SummaryRow],
    deltas: Sequence[SummaryRow],
    front: Sequence[SummaryRow],
) -> str:
    best_delta = max(deltas, key=lambda row: float(row["completion_rate_delta"]), default=None)
    lines = [
        "# AI2-THOR Rearrangement Maintenance Multi-Seed Summary",
        "",
        f"Rows: `{len(rows)}`",
        f"Pareto frontier rows: `{len(front)}`",
        "",
    ]
    if best_delta:
        lines.extend(
            [
                f"Best completion delta vs passive: budget `{best_delta['budget']}` / maintenance budget `{best_delta['maintenance_budget']}`",
                f"Method policy: `{best_delta['method_policy']}`",
                f"Completion delta: `{best_delta['completion_rate_delta']}`",
                f"Stale-error delta: `{best_delta['stale_errors_delta']}`",
                f"Maintenance checks delta: `{best_delta['maintenance_checks_delta']}`",
                "",
            ]
        )
    lines.extend(
        [
            "Scope: this aggregates saved-probe proactive-maintenance summaries. It does not turn replay/oracle maintenance into live closed-loop sensing evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> ParsedArgs:
    parser = argparse.ArgumentParser(description="Aggregate AI2-THOR proactive maintenance summaries across seeds.")
    _ = parser.add_argument("summary_paths", type=Path, nargs="+")
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_rearrangement_maintenance_multiseed"))
    namespace = parser.parse_args()
    return ParsedArgs(
        summary_paths=cast(list[Path], namespace.summary_paths),
        out_dir=cast(Path, namespace.out_dir),
    )


def main() -> None:
    args = parse_args()
    rows = aggregate_summaries(args.summary_paths)
    write_outputs(rows, args.out_dir)
    print(json.dumps({"summary": rows, "delta_vs_passive": delta_vs_passive(rows), "pareto_front": pareto_front(rows)}, indent=2))


if __name__ == "__main__":
    main()
