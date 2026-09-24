from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    RearrangementTask,
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    load_probe,
    summarize_rearrangement_probe,
)
from embodied_memory_pilot.pilot import MemoryItem, MemoryPolicy, MemoryStore, NoMemoryPolicy, PolicyMetrics, SaliencePolicy


class ThresholdRearrangementRiskPolicy(MemoryPolicy):
    name = "verify_threshold"

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold
        self.name = f"verify_threshold_{_label_float(threshold)}"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Dict[str, int] | None = None,
    ) -> float:
        return SaliencePolicy().score(item, now=now, current_target=current_target, future_counts=future_counts)

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        age = max(now - item.observed_at, 0)
        return item.volatility * (1.0 + age / 12.0) > self.threshold


class VerifyAllRearrangementPolicy(ThresholdRearrangementRiskPolicy):
    name = "verify_all"

    def __init__(self) -> None:
        super().__init__(threshold=-math.inf)
        self.name = "verify_all"

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        return True


class DetectOnlyRearrangementRiskPolicy(ThresholdRearrangementRiskPolicy):
    name = "detect_only_no_recovery"

    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__(threshold=threshold)
        self.name = f"detect_only_{_label_float(threshold)}"


@dataclass(frozen=True)
class AblationSpec:
    ablation: str
    policy: MemoryPolicy
    recover_caught_stale: bool = True


@dataclass(frozen=True)
class PreparedRearrangementProbe:
    items: Sequence[MemoryItem]
    tasks: Sequence[RearrangementTask]
    query_now: int
    task_by_old_location: Dict[str, RearrangementTask]


def _label_float(value: float) -> str:
    return str(value).replace("-", "neg").replace(".", "p")


def prepare_rearrangement_probe(
    probe: Dict[str, Any],
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> PreparedRearrangementProbe:
    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe, target_objects=target_objects)
    query_now = max((item.observed_at for item in items), default=0) + 1
    task_by_old_location: Dict[str, RearrangementTask] = {task.old_location: task for task in tasks}
    return PreparedRearrangementProbe(
        items=items,
        tasks=tasks,
        query_now=query_now,
        task_by_old_location=task_by_old_location,
    )


def _evaluate_prepared_rearrangement_ablation(
    prepared: PreparedRearrangementProbe,
    policy: MemoryPolicy,
    ablation: str,
    budget: int,
    verification_cost: float = 0.5,
    recover_caught_stale: bool = True,
) -> Dict[str, Any]:
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)
    saved_reachable_cost = 0.0

    for item in prepared.items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    for task in prepared.tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=prepared.query_now):
            metrics.verifications += 1
            metrics.total_cost += verification_cost
            if outcome.stale:
                metrics.verification_catches += 1
                if recover_caught_stale:
                    metrics.successes += 1
                    metrics.total_cost += task.scene_search_cost
                else:
                    metrics.total_cost += task.scene_search_cost
            else:
                metrics.successes += 1
                metrics.query_hits += 1
                metrics.total_cost += task.direct_action_cost
                saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost - verification_cost, 0.0)
        elif outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost
            saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost, 0.0)
        elif outcome.found and outcome.stale and outcome.item is not None:
            metrics.stale_errors += 1
            stale_task = prepared.task_by_old_location.get(outcome.item.location)
            old_cost = stale_task.old_action_cost if stale_task else task.old_action_cost
            metrics.total_cost += old_cost + task.scene_search_cost
        else:
            metrics.total_cost += task.scene_search_cost

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "ai2thor_rearrangement_ablation"
    row["ablation"] = ablation
    row["policy"] = policy.name
    row["memory_items"] = len(prepared.items)
    row["rearrangement_tasks"] = len(prepared.tasks)
    row["moved_tasks"] = sum(1 for task in prepared.tasks if task.rearranged)
    row["retained_items"] = len(store.items)
    row["avg_reachable_cost"] = round(metrics.avg_cost, 4)
    row["saved_reachable_cost"] = round(saved_reachable_cost, 4)
    row["completion_rate"] = row["success_rate"]
    row["verification_cost"] = verification_cost
    row["recover_caught_stale"] = int(recover_caught_stale)
    return row


def evaluate_rearrangement_ablation(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    ablation: str,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    verification_cost: float = 0.5,
    recover_caught_stale: bool = True,
) -> Dict[str, Any]:
    return _evaluate_prepared_rearrangement_ablation(
        prepare_rearrangement_probe(probe, target_objects=target_objects),
        policy=policy,
        ablation=ablation,
        budget=budget,
        verification_cost=verification_cost,
        recover_caught_stale=recover_caught_stale,
    )


def ablation_specs(thresholds: Sequence[float]) -> List[AblationSpec]:
    specs: List[AblationSpec] = [
        AblationSpec("salience_no_verification", SaliencePolicy(), True),
        AblationSpec("verify_all", VerifyAllRearrangementPolicy(), True),
    ]
    for threshold in thresholds:
        specs.append(AblationSpec(f"verify_threshold_{_label_float(threshold)}", ThresholdRearrangementRiskPolicy(threshold), True))
        specs.append(AblationSpec(f"detect_only_{_label_float(threshold)}", DetectOnlyRearrangementRiskPolicy(threshold), False))
    return specs


def run_ablation_grid(
    probes: Sequence[Dict[str, Any]],
    budgets: Sequence[int],
    verification_costs: Sequence[float],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = ablation_specs(thresholds)
    prepared_probes = [prepare_rearrangement_probe(probe) for probe in probes]
    for seed_index, prepared in enumerate(prepared_probes):
        for budget in budgets:
            for verification_cost in verification_costs:
                for spec in specs:
                    row = _evaluate_prepared_rearrangement_ablation(
                        prepared,
                        policy=spec.policy,
                        ablation=spec.ablation,
                        budget=budget,
                        verification_cost=verification_cost,
                        recover_caught_stale=spec.recover_caught_stale,
                    )
                    row["seed"] = seed_index
                    rows.append(row)
    return rows


def aggregate_ablation_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float | int | str]]:
    grouped: Dict[tuple[int, float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["budget"]), float(row["verification_cost"]), str(row["ablation"])), []).append(row)

    summary: List[Dict[str, float | int | str]] = []
    for (budget, verification_cost, ablation), group in sorted(grouped.items()):
        summary.append(
            {
                "budget": budget,
                "verification_cost": verification_cost,
                "ablation": ablation,
                "policy": str(group[0]["policy"]),
                "seeds": len(group),
                "rearrangement_tasks_mean": round(_mean([float(row["rearrangement_tasks"]) for row in group]), 4),
                "moved_tasks_mean": round(_mean([float(row["moved_tasks"]) for row in group]), 4),
                "completion_rate_mean": round(_mean([float(row["completion_rate"]) for row in group]), 4),
                "completion_rate_std": round(_sample_std([float(row["completion_rate"]) for row in group]), 4),
                "avg_reachable_cost_mean": round(_mean([float(row["avg_reachable_cost"]) for row in group]), 4),
                "avg_reachable_cost_std": round(_sample_std([float(row["avg_reachable_cost"]) for row in group]), 4),
                "stale_errors_mean": round(_mean([float(row["stale_errors"]) for row in group]), 4),
                "stale_errors_std": round(_sample_std([float(row["stale_errors"]) for row in group]), 4),
                "verifications_mean": round(_mean([float(row["verifications"]) for row in group]), 4),
                "verification_catches_mean": round(_mean([float(row["verification_catches"]) for row in group]), 4),
                "recover_caught_stale": int(group[0]["recover_caught_stale"]),
            }
        )
    return summary


def delta_vs_baseline(
    summary: Sequence[Dict[str, float | int | str]],
    baseline_ablation: str = "salience_no_verification",
) -> List[Dict[str, float | int | str]]:
    by_key = {
        (int(row["budget"]), float(row["verification_cost"]), str(row["ablation"])): row
        for row in summary
    }
    deltas: List[Dict[str, float | int | str]] = []
    for row in summary:
        ablation = str(row["ablation"])
        if ablation == baseline_ablation:
            continue
        baseline = by_key.get((int(row["budget"]), float(row["verification_cost"]), baseline_ablation))
        if baseline is None:
            continue
        deltas.append(
            {
                "budget": int(row["budget"]),
                "verification_cost": float(row["verification_cost"]),
                "ablation": ablation,
                "baseline_ablation": baseline_ablation,
                "seeds": min(int(row["seeds"]), int(baseline["seeds"])),
                "completion_rate_delta": round(float(row["completion_rate_mean"]) - float(baseline["completion_rate_mean"]), 4),
                "avg_reachable_cost_delta": round(float(row["avg_reachable_cost_mean"]) - float(baseline["avg_reachable_cost_mean"]), 4),
                "stale_errors_delta": round(float(row["stale_errors_mean"]) - float(baseline["stale_errors_mean"]), 4),
                "verifications_delta": round(float(row["verifications_mean"]) - float(baseline["verifications_mean"]), 4),
                "verification_catches_delta": round(float(row["verification_catches_mean"]) - float(baseline["verification_catches_mean"]), 4),
            }
        )
    return deltas


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _write_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_ablation_rows(rows)
    deltas = delta_vs_baseline(summary)
    (out_dir / "ai2thor_rearrangement_ablation.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary, "delta_vs_salience_no_verification": deltas}, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "ai2thor_rearrangement_ablation_summary.csv", summary)
    _write_csv(out_dir / "ai2thor_rearrangement_ablation_delta.csv", deltas)
    (out_dir / "README.md").write_text(render_readme(summary, deltas), encoding="utf-8")


def render_readme(summary: Sequence[Dict[str, float | int | str]], deltas: Sequence[Dict[str, float | int | str]]) -> str:
    best = max(deltas, key=lambda row: (float(row["completion_rate_delta"]), -float(row["stale_errors_delta"])), default=None)
    lines = [
        "# AI2-THOR Rearrangement Verification Ablations",
        "",
        f"Summary rows: `{len(summary)}`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best ablation vs salience-no-verification: `{best['ablation']}`",
                f"Budget: `{best['budget']}`",
                f"Verification cost: `{best['verification_cost']}`",
                f"Completion delta: `{best['completion_rate_delta']}`",
                f"Stale-error delta: `{best['stale_errors_delta']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This replay uses saved AI2-THOR rearrangement probes and does not start Unity.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR rearrangement verification ablation replay.")
    parser.add_argument("--probes", type=Path, nargs="+", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--verification-costs", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_rearrangement_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probes = [load_probe(path) for path in args.probes]
    rows = run_ablation_grid(
        probes,
        budgets=args.budgets,
        verification_costs=args.verification_costs,
        thresholds=args.thresholds,
    )
    write_outputs(rows, args.out_dir)
    print(json.dumps({"probe_summaries": [summarize_rearrangement_probe(probe) for probe in probes], "summary": aggregate_ablation_rows(rows)}, indent=2))


if __name__ == "__main__":
    main()
