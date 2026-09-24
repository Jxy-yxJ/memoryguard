from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, cast

from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    RearrangementTask,
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    load_probe,
    summarize_rearrangement_probe,
)
from embodied_memory_pilot.pilot import MemoryItem, MemoryStore, PolicyMetrics, SaliencePolicy


RowValue = int | float | str
MetricRow = dict[str, RowValue]
Probe = dict[str, object]


class MaintenanceSelection(Protocol):
    name: str

    def select(
        self,
        items: Sequence[MemoryItem],
        *,
        now: int,
        current_target: str,
        future_counts: dict[str, int],
        budget: int,
        rng: random.Random,
    ) -> Sequence[MemoryItem]:
        ...


@dataclass(frozen=True)
class ParsedArgs:
    probe: Path
    budgets: list[int]
    maintenance_budgets: list[int]
    maintenance_cost: float
    random_seed: int
    out_dir: Path


class PassiveMaintenanceSelection:
    name: str = "passive"

    def select(
        self,
        items: Sequence[MemoryItem],
        *,
        now: int,
        current_target: str,
        future_counts: dict[str, int],
        budget: int,
        rng: random.Random,
    ) -> Sequence[MemoryItem]:
        _ = (items, now, current_target, future_counts, budget, rng)
        return ()


class RandomActiveMaintenanceSelection:
    name: str = "random_active"

    def select(
        self,
        items: Sequence[MemoryItem],
        *,
        now: int,
        current_target: str,
        future_counts: dict[str, int],
        budget: int,
        rng: random.Random,
    ) -> Sequence[MemoryItem]:
        _ = (now, current_target, future_counts)
        if budget <= 0 or not items:
            return ()
        pool = list(items)
        rng.shuffle(pool)
        return tuple(pool[:budget])


class HeuristicActiveMaintenanceSelection:
    name: str = "heuristic_active"

    def select(
        self,
        items: Sequence[MemoryItem],
        *,
        now: int,
        current_target: str,
        future_counts: dict[str, int],
        budget: int,
        rng: random.Random,
    ) -> Sequence[MemoryItem]:
        _ = rng
        if budget <= 0 or not items:
            return ()
        ranked = sorted(
            items,
            key=lambda item: maintenance_tau(
                item,
                now=now,
                current_target=current_target,
                future_counts=future_counts,
            ),
        )
        return tuple(ranked[:budget])


def _maintenance_item_features(
    item: MemoryItem,
    *,
    now: int,
    current_target: str,
    future_counts: dict[str, int] | None = None,
) -> list[float]:
    age = max(now - item.observed_at, 0)
    freshness = 1.0 / (1.0 + age)
    risk = item.volatility * math.log1p(age)
    future = 0 if future_counts is None else future_counts.get(item.object_name, 0)
    target_bonus = 1.0 if item.object_name == current_target else 0.0
    return [
        freshness,
        float(item.salience),
        float(item.demand),
        risk,
        min(age / 100.0, 1.0),
        min(future / 10.0, 1.0) if future_counts is not None else 0.0,
        target_bonus,
        float(item.failure_count),
    ]


class MLPMaintenanceSelection:
    name: str = "mlp_active"

    def __init__(self, mlp: "SimpleMLP", threshold: float = 0.5):
        self.mlp = mlp
        self.threshold = threshold

    def select(
        self,
        items: Sequence[MemoryItem],
        *,
        now: int,
        current_target: str,
        future_counts: dict[str, int],
        budget: int,
        rng: random.Random,
    ) -> Sequence[MemoryItem]:
        _ = rng
        if budget <= 0 or not items:
            return ()
        ranked = sorted(
            items,
            key=lambda item: self.mlp.predict(
                _maintenance_item_features(
                    item,
                    now=now,
                    current_target=current_target,
                    future_counts=future_counts,
                )
            ),
            reverse=True,
        )
        return tuple(ranked[:budget])


def _train_mlp_for_maintenance(
    probes: Sequence[Path],
    *,
    hold_out_idx: int = 0,
    epochs: int = 200,
    budget: int = 16,
) -> Optional["SimpleMLP"]:
    from embodied_memory_pilot.ai2thor_learned_verification import SimpleMLP

    train_paths = [p for i, p in enumerate(probes) if i != hold_out_idx]
    examples: list[dict] = []

    for tp in train_paths:
        probe = load_probe(tp)
        items = build_rearrangement_memory_items(probe)
        tasks = build_rearrangement_tasks(probe)
        if not items:
            continue
        query_now = max((it.observed_at for it in items), default=0) + 1
        store = MemoryStore(policy=SaliencePolicy(), budget=budget)
        for it in items:
            store.observe(it, now=it.observed_at, current_target=it.object_name)

        for task in tasks:
            outcome = store.query(task.target, task.true_location)
            if outcome.found and outcome.item is not None:
                feat = _maintenance_item_features(
                    outcome.item,
                    now=query_now,
                    current_target=task.target,
                )
                examples.append({
                    "features": feat,
                    "stale": 1 if outcome.stale else 0,
                    "object_name": outcome.item.object_name,
                })

    if not examples or sum(1 for e in examples if e["stale"]) == 0:
        return None

    mlp = SimpleMLP(input_dim=len(examples[0]["features"]), hidden_dim=16)
    mlp.train(examples, epochs=epochs)
    return mlp


def maintenance_tau(
    item: MemoryItem,
    *,
    now: int,
    current_target: str,
    future_counts: dict[str, int] | None = None,
) -> float:
    """Trust score for proactive maintenance.

    Lower tau means the memory item is less trusted and should be revisited sooner.
    """

    age = max(now - item.observed_at, 0)
    freshness = 1.0 / (1.0 + age)
    risk = item.volatility * math.log1p(age)
    future = 0 if future_counts is None else future_counts.get(item.object_name, 0)
    target_bonus = 1.0 if item.object_name == current_target else 0.0
    return (
        0.9 * freshness
        + 0.2 * item.salience
        + 0.2 * item.demand
        - 1.1 * risk
        - 0.15 * future
        - 0.1 * target_bonus
        - 0.05 * item.failure_count
    )


def future_task_counts(tasks: Sequence[RearrangementTask], start: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks[start:]:
        counts[task.target] = counts.get(task.target, 0) + 1
    return counts


def _replace_item(store: MemoryStore, old_item: MemoryItem, new_item: MemoryItem) -> None:
    store.items = [
        existing
        for existing in store.items
        if not (existing.object_name == old_item.object_name and existing.location == old_item.location)
    ]
    store.observe(new_item, now=new_item.observed_at, current_target=new_item.object_name)


def _maintenance_candidates(
    store: MemoryStore,
    task_lookup: Mapping[str, RearrangementTask],
) -> list[MemoryItem]:
    return [item for item in store.items if item.location in task_lookup]


def evaluate_policy_rearrangement_maintenance(
    probe: Probe,
    policy: MaintenanceSelection,
    budget: int,
    maintenance_budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    maintenance_cost: float = 0.5,
    random_seed: int = 0,
) -> MetricRow:
    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe, target_objects=target_objects)
    task_lookup = {task.old_location: task for task in tasks}
    store = MemoryStore(policy=SaliencePolicy(), budget=budget)
    metrics = PolicyMetrics(policy=policy.name)
    rng = random.Random(random_seed)
    saved_reachable_cost = 0.0

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    base_now = max((item.observed_at for item in items), default=0) + 1
    for index, task in enumerate(tasks):
        now = base_now + index
        future_counts = future_task_counts(tasks, index + 1)
        candidates = _maintenance_candidates(store, task_lookup)
        selected = policy.select(
            candidates,
            now=now,
            current_target=task.target,
            future_counts=future_counts,
            budget=maintenance_budget,
            rng=rng,
        )

        for selected_item in selected:
            task_for_item = task_lookup.get(selected_item.location)
            if task_for_item is None:
                continue
            metrics.verifications += 1
            metrics.total_cost += maintenance_cost
            if selected_item.location == task_for_item.true_location:
                refreshed = MemoryItem(
                    object_name=selected_item.object_name,
                    location=selected_item.location,
                    observed_at=now,
                    salience=selected_item.salience,
                    volatility=selected_item.volatility,
                    demand=selected_item.demand,
                    failure_count=selected_item.failure_count,
                )
            else:
                metrics.verification_catches += 1
                refreshed = MemoryItem(
                    object_name=selected_item.object_name,
                    location=task_for_item.true_location,
                    observed_at=now,
                    salience=selected_item.salience,
                    volatility=selected_item.volatility,
                    demand=selected_item.demand,
                    failure_count=selected_item.failure_count + 1,
                )
            _replace_item(store, selected_item, refreshed)

        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost
            saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost, 0.0)
        elif outcome.found and outcome.stale and outcome.item is not None:
            metrics.stale_errors += 1
            stale_task = task_lookup.get(outcome.item.location)
            stale_cost = stale_task.old_action_cost if stale_task else task.old_action_cost
            metrics.total_cost += stale_cost + task.scene_search_cost
        else:
            metrics.total_cost += task.scene_search_cost

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=random_seed, budget=budget)
    row["mode"] = "ai2thor_rearrangement_active_maintenance"
    row["maintenance_policy"] = policy.name
    row["maintenance_budget"] = maintenance_budget
    row["maintenance_cost"] = maintenance_cost
    row["maintenance_checks"] = metrics.verifications
    row["maintenance_catches"] = metrics.verification_catches
    row["memory_items"] = len(items)
    row["rearrangement_tasks"] = len(tasks)
    row["moved_tasks"] = sum(1 for task in tasks if task.rearranged)
    row["retained_items"] = len(store.items)
    row["avg_reachable_cost"] = round(metrics.avg_cost, 4)
    row["saved_reachable_cost"] = round(saved_reachable_cost, 4)
    row["completion_rate"] = row["success_rate"]
    return row


def run_rearrangement_maintenance(
    probe: Probe,
    budgets: Sequence[int],
    maintenance_budgets: Sequence[int],
    policies: Iterable[MaintenanceSelection] | None = None,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    maintenance_cost: float = 0.5,
    random_seed: int = 0,
) -> list[MetricRow]:
    rows: list[MetricRow] = []
    selected_policies = tuple(
        policies
        if policies is not None
        else (
            PassiveMaintenanceSelection(),
            RandomActiveMaintenanceSelection(),
            HeuristicActiveMaintenanceSelection(),
        )
    )
    for budget in budgets:
        for maintenance_budget in maintenance_budgets:
            for policy in selected_policies:
                rows.append(
                    evaluate_policy_rearrangement_maintenance(
                        probe,
                        policy=policy,
                        budget=budget,
                        maintenance_budget=maintenance_budget,
                        target_objects=target_objects,
                        maintenance_cost=maintenance_cost,
                        random_seed=random_seed,
                    )
                )
    return rows


def summarize_rows(rows: Sequence[MetricRow]) -> list[MetricRow]:
    grouped: dict[tuple[int, int, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["budget"]), int(row["maintenance_budget"]), str(row["maintenance_policy"])),
            [],
        ).append(row)

    summary: list[MetricRow] = []
    for (budget, maintenance_budget, policy), group in sorted(grouped.items()):
        summary.append(
            {
                "budget": budget,
                "maintenance_budget": maintenance_budget,
                "maintenance_policy": policy,
                "runs": len(group),
                "rearrangement_tasks": int(group[0]["rearrangement_tasks"]) if group else 0,
                "moved_tasks": int(group[0]["moved_tasks"]) if group else 0,
                "memory_items": int(group[0]["memory_items"]) if group else 0,
                "retained_items": round(sum(float(r["retained_items"]) for r in group) / len(group), 2),
                "completion_rate": round(sum(float(r["completion_rate"]) for r in group) / len(group), 4),
                "avg_reachable_cost": round(sum(float(r["avg_reachable_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
                "maintenance_checks": round(sum(float(r["maintenance_checks"]) for r in group) / len(group), 2),
                "maintenance_catches": round(sum(float(r["maintenance_catches"]) for r in group) / len(group), 2),
                "maintenance_cost": round(sum(float(r["maintenance_cost"]) for r in group) / len(group), 4),
                "saved_reachable_cost": round(sum(float(r["saved_reachable_cost"]) for r in group) / len(group), 4),
            }
        )
    return summary


def pareto_front(rows: Sequence[MetricRow]) -> list[MetricRow]:
    front: list[MetricRow] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = (
                float(other["completion_rate"]) >= float(row["completion_rate"])
                and float(other["avg_reachable_cost"]) <= float(row["avg_reachable_cost"])
                and float(other["stale_errors"]) <= float(row["stale_errors"])
            )
            strictly_better = (
                float(other["completion_rate"]) > float(row["completion_rate"])
                or float(other["avg_reachable_cost"]) < float(row["avg_reachable_cost"])
                or float(other["stale_errors"]) < float(row["stale_errors"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(
        front,
        key=lambda row: (-float(row["completion_rate"]), float(row["avg_reachable_cost"]), float(row["stale_errors"])),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        _ = path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in rows[0].keys()})


def write_outputs(rows: Sequence[MetricRow], probe: Probe, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    front = pareto_front(summary)
    _ = (out_dir / "ai2thor_rearrangement_maintenance.json").write_text(
        json.dumps(
            {
                "probe_summary": summarize_rearrangement_probe(probe),
                "runs": list(rows),
                "summary": summary,
                "pareto_front": front,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_summary.csv", summary)
    _write_csv(out_dir / "ai2thor_rearrangement_maintenance_pareto.csv", front)
    _ = (out_dir / "README.md").write_text(render_readme(summary, front, probe), encoding="utf-8")


def render_readme(summary: Sequence[MetricRow], front: Sequence[MetricRow], probe: Probe) -> str:
    probe_summary = summarize_rearrangement_probe(probe)
    best = max(
        summary,
        key=lambda row: (
            float(row["completion_rate"]),
            -float(row["stale_errors"]),
            -float(row["avg_reachable_cost"]),
            float(row["maintenance_budget"]),
        ),
        default=None,
    )
    lines = [
        "# AI2-THOR Rearrangement Active Maintenance",
        "",
        "Mode: `ai2thor_rearrangement_active_maintenance`",
        f"OK scenes: `{probe_summary['ok_scenes']}`",
        f"Moved paired objects: `{probe_summary['moved_objects']}`",
        f"Pareto frontier points: `{len(front)}`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['maintenance_policy']}` at budget `{best['budget']}` / maintenance budget `{best['maintenance_budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Stale errors: `{best['stale_errors']}`",
                f"Maintenance checks: `{best['maintenance_checks']}`",
                f"Average reachable cost: `{best['avg_reachable_cost']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This benchmark reuses saved AI2-THOR rearrangement probes and compares passive, random-active, and heuristic-active maintenance policies.",
            "Maintenance uses replay/oracle access to saved after-state metadata; it is an MVP for proactive memory maintenance, not a live closed-loop controller.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> ParsedArgs:
    parser = argparse.ArgumentParser(description="AI2-THOR proactive memory-maintenance replay benchmark.")
    _ = parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_rearrangement_6scene_seed7/ai2thor_rearrangement_probe.json"))
    _ = parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    _ = parser.add_argument("--maintenance-budgets", type=int, nargs="+", default=[0, 1, 2])
    _ = parser.add_argument("--maintenance-cost", type=float, default=0.5)
    _ = parser.add_argument("--random-seed", type=int, default=0)
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_rearrangement_maintenance"))
    namespace = parser.parse_args()
    return ParsedArgs(
        probe=cast(Path, namespace.probe),
        budgets=cast(list[int], namespace.budgets),
        maintenance_budgets=cast(list[int], namespace.maintenance_budgets),
        maintenance_cost=cast(float, namespace.maintenance_cost),
        random_seed=cast(int, namespace.random_seed),
        out_dir=cast(Path, namespace.out_dir),
    )


def main() -> None:
    args = parse_args()
    probe = cast(Probe, load_probe(args.probe))
    rows = run_rearrangement_maintenance(
        probe,
        budgets=args.budgets,
        maintenance_budgets=args.maintenance_budgets,
        maintenance_cost=args.maintenance_cost,
        random_seed=args.random_seed,
    )
    write_outputs(rows, probe, args.out_dir)
    print(
        json.dumps(
            {
                "probe_summary": summarize_rearrangement_probe(probe),
                "summary": summarize_rows(rows),
                "pareto_front": pareto_front(summarize_rows(rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
