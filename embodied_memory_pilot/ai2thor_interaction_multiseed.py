from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


METRICS = (
    "completion_rate",
    "avg_interaction_cost",
    "query_hit_rate",
    "interaction_stale_errors",
    "verifications",
    "verification_catches",
)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_summaries(summary_paths: Iterable[Path]) -> List[Dict[str, float | int | str]]:
    grouped: Dict[tuple[int, str], List[Dict[str, str]]] = {}
    for path in summary_paths:
        for row in _read_rows(path):
            grouped.setdefault((int(row["budget"]), row["policy"]), []).append(row)

    rows: List[Dict[str, float | int | str]] = []
    for (budget, policy), group in sorted(grouped.items()):
        out: Dict[str, float | int | str] = {
            "budget": budget,
            "policy": policy,
            "seeds": len(group),
            "interaction_tasks_mean": round(_mean([float(row["interaction_tasks"]) for row in group]), 4),
            "pickup_tasks_mean": round(_mean([float(row["pickup_tasks"]) for row in group]), 4),
            "open_tasks_mean": round(_mean([float(row["open_tasks"]) for row in group]), 4),
            "moved_tasks_mean": round(_mean([float(row["moved_tasks"]) for row in group]), 4),
            "memory_items_mean": round(_mean([float(row["memory_items"]) for row in group]), 4),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = round(_mean(values), 4)
            out[f"{metric}_std"] = round(_sample_std(values), 4)
        rows.append(out)
    return rows


def delta_vs_baseline(
    rows: Sequence[Dict[str, float | int | str]],
    method_policy: str = "verify_interaction_risk",
    baseline_policy: str = "salience",
) -> List[Dict[str, float | int | str]]:
    by_key = {(int(row["budget"]), str(row["policy"])): row for row in rows}
    deltas: List[Dict[str, float | int | str]] = []
    for row in rows:
        budget = int(row["budget"])
        if str(row["policy"]) != method_policy:
            continue
        baseline = by_key.get((budget, baseline_policy))
        if baseline is None:
            continue
        deltas.append(
            {
                "budget": budget,
                "method_policy": method_policy,
                "baseline_policy": baseline_policy,
                "seeds": min(int(row["seeds"]), int(baseline["seeds"])),
                "completion_rate_delta": round(float(row["completion_rate_mean"]) - float(baseline["completion_rate_mean"]), 4),
                "avg_interaction_cost_delta": round(float(row["avg_interaction_cost_mean"]) - float(baseline["avg_interaction_cost_mean"]), 4),
                "interaction_stale_errors_delta": round(float(row["interaction_stale_errors_mean"]) - float(baseline["interaction_stale_errors_mean"]), 4),
                "verifications_delta": round(float(row["verifications_mean"]) - float(baseline["verifications_mean"]), 4),
                "verification_catches_delta": round(float(row["verification_catches_mean"]) - float(baseline["verification_catches_mean"]), 4),
            }
        )
    return deltas


def _write_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = delta_vs_baseline(rows)
    _write_csv(out_dir / "ai2thor_interaction_multiseed_summary.csv", rows)
    _write_csv(out_dir / "ai2thor_interaction_multiseed_delta.csv", deltas)
    (out_dir / "ai2thor_interaction_multiseed.json").write_text(
        json.dumps({"summary": list(rows), "delta_vs_salience": deltas}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(render_readme(rows, deltas), encoding="utf-8")


def render_readme(rows: Sequence[Dict[str, float | int | str]], deltas: Sequence[Dict[str, float | int | str]]) -> str:
    best = max(deltas, key=lambda row: float(row["completion_rate_delta"]), default=None)
    lines = [
        "# AI2-THOR Interaction Memory Multi-Seed Summary",
        "",
        f"Rows: `{len(rows)}`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best completion delta vs salience: budget `{best['budget']}`",
                f"Completion delta: `{best['completion_rate_delta']}`",
                f"Interaction stale-error delta: `{best['interaction_stale_errors_delta']}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate AI2-THOR interaction memory summaries across seeds.")
    parser.add_argument("summary_paths", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_interaction_multiseed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = aggregate_summaries(args.summary_paths)
    write_outputs(rows, args.out_dir)
    print(json.dumps({"summary": rows, "delta_vs_salience": delta_vs_baseline(rows)}, indent=2))


if __name__ == "__main__":
    main()
