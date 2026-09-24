from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from embodied_memory_pilot.pilot import MemoryItem, MemoryPolicy, run_policy
from embodied_memory_pilot.symbolic_sim import run_symbolic_policy


class TunableFreshnessPolicy(MemoryPolicy):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"freshness_threshold_{self.threshold:.2f}"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Dict[str, int] | None = None,
    ) -> float:
        age = now - item.observed_at
        freshness = 1.0 / (1.0 + age)
        target_bonus = 1.0 if item.object_name == current_target else 0.0
        verification_need = item.volatility * math.log1p(age)
        utility = 3.0 * item.demand + 1.1 * target_bonus + 1.3 * freshness
        return utility - 2.8 * verification_need + 0.4 * item.failure_count

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        age = now - item.observed_at
        return item.volatility * math.log1p(age) > self.threshold


def score_row(row: Dict[str, float | int | str]) -> float:
    return round(
        float(row["success_rate"])
        - 0.04 * float(row["avg_cost"])
        - 0.01 * float(row["stale_errors"])
        - 0.002 * float(row["verifications"]),
        4,
    )


def aggregate_threshold_rows(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    grouped: Dict[tuple[float, int, str], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((float(row["threshold"]), int(row["budget"]), str(row["policy"])), []).append(row)

    summary: List[Dict[str, float | int | str]] = []
    for (threshold, budget, policy), group in sorted(grouped.items()):
        entry: Dict[str, float | int | str] = {
            "threshold": threshold,
            "budget": budget,
            "policy": policy,
            "runs": len(group),
            "success_rate": round(sum(float(r["success_rate"]) for r in group) / len(group), 4),
            "avg_cost": round(sum(float(r["avg_cost"]) for r in group) / len(group), 4),
            "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
            "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
            "verifications": round(sum(float(r["verifications"]) for r in group) / len(group), 2),
            "verification_catches": round(
                sum(float(r["verification_catches"]) for r in group) / len(group),
                2,
            ),
        }
        entry["score"] = score_row(entry)
        summary.append(entry)
    return summary


def find_pareto_front(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    front: List[Dict[str, float | int | str]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            at_least_as_good = (
                float(other["success_rate"]) >= float(row["success_rate"])
                and float(other["avg_cost"]) <= float(row["avg_cost"])
                and float(other["stale_errors"]) <= float(row["stale_errors"])
            )
            strictly_better = (
                float(other["success_rate"]) > float(row["success_rate"])
                or float(other["avg_cost"]) < float(row["avg_cost"])
                or float(other["stale_errors"]) < float(row["stale_errors"])
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda item: float(item["threshold"]))


def run_threshold_sweep(
    thresholds: Sequence[float],
    seeds: Iterable[int],
    budgets: Sequence[int],
    tasks: int,
    drift_scale: float = 3.0,
    task_budget: float | None = None,
    mode: str = "symbolic",
) -> Dict[str, object]:
    if mode not in {"offline", "symbolic"}:
        raise ValueError("mode must be either 'offline' or 'symbolic'")
    if task_budget is None:
        task_budget = 6.0 if mode == "symbolic" else 5.0

    rows: List[Dict[str, float | int | str]] = []
    seed_values = list(seeds)
    for threshold in thresholds:
        policy = TunableFreshnessPolicy(threshold=threshold)
        for seed in seed_values:
            for budget in budgets:
                if mode == "symbolic":
                    metrics = run_symbolic_policy(
                        seed=seed,
                        budget=budget,
                        tasks_n=tasks,
                        policy=policy,
                        drift_scale=drift_scale,
                        task_budget=task_budget,
                    )
                else:
                    metrics = run_policy(
                        seed=seed,
                        budget=budget,
                        tasks_n=tasks,
                        policy=policy,
                        drift_scale=drift_scale,
                        task_budget=task_budget,
                    )
                row = metrics.as_dict(seed=seed, budget=budget)
                row["threshold"] = threshold
                row["drift_scale"] = drift_scale
                row["task_budget"] = task_budget
                row["mode"] = mode
                rows.append(row)

    summary = aggregate_threshold_rows(rows)
    best = max(summary, key=lambda row: float(row["score"])) if summary else {}
    pareto_front = find_pareto_front(summary)
    return {"runs": rows, "summary": summary, "best": best, "pareto_front": pareto_front}


def write_outputs(result: Dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = list(result["summary"])  # type: ignore[arg-type]
    (out_dir / "threshold_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "threshold_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "threshold",
            "budget",
            "policy",
            "runs",
            "success_rate",
            "avg_cost",
            "query_hit_rate",
            "stale_errors",
            "verifications",
            "verification_catches",
            "score",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep freshness verification thresholds.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.18, 0.24, 0.30, 0.36, 0.42, 0.50])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--drift-scale", type=float, default=3.0)
    parser.add_argument("--task-budget", type=float, default=None)
    parser.add_argument("--mode", choices=("offline", "symbolic"), default="symbolic")
    parser.add_argument("--out-dir", type=Path, default=Path("results/threshold_sweep"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_threshold_sweep(
        thresholds=args.thresholds,
        seeds=range(args.seeds),
        budgets=args.budgets,
        tasks=args.tasks,
        drift_scale=args.drift_scale,
        task_budget=args.task_budget,
        mode=args.mode,
    )
    write_outputs(result, args.out_dir)
    print(json.dumps({"best": result["best"], "pareto_front": result["pareto_front"]}, indent=2))


if __name__ == "__main__":
    main()
