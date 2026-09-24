from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_memory_eval import (
    AI2ThorCalibratedPolicy,
    TARGET_OBJECTS,
    build_memory_items,
    load_probe,
)
from embodied_memory_pilot.ai2thor_reachable_benchmark import (
    ReachableTask,
    build_reachable_tasks,
)
from embodied_memory_pilot.ai2thor_weight_search import WeightedAI2ThorPolicy
from embodied_memory_pilot.pilot import (
    FreshnessAwarePolicy,
    MemoryItem,
    MemoryPolicy,
    MemoryStore,
    NoMemoryPolicy,
    PolicyMetrics,
    SaliencePolicy,
    TaskConditionedPolicy,
)


DEFAULT_STALE_AWARE_WEIGHTS = {
    "salience": 0.6,
    "demand": 2.0,
    "target": 1.2,
    "freshness": 2.5,
    "pickupable": 0.5,
    "stale": 3.0,
}


def _stale_location(task: ReachableTask, index: int) -> str:
    return f"{task.scene}:stale:{task.target}:{index}"


def select_probe_scenes(probe: Dict[str, Any], scenes: Sequence[str] | None) -> Dict[str, Any]:
    if not scenes:
        return dict(probe)
    scene_set = set(scenes)
    selected = dict(probe)
    selected_scenes = [scene for scene in probe.get("scenes", []) if str(scene.get("scene")) in scene_set]
    selected["scenes"] = selected_scenes
    selected["summary"] = {
        "scenes": len(selected_scenes),
        "ok_scenes": sum(1 for scene in selected_scenes if scene.get("status") == "ok"),
        "total_visible": sum(int(scene.get("visible_count", len(scene.get("visible_objects", [])))) for scene in selected_scenes),
        "total_reachable": sum(int(scene.get("reachable_count", len(scene.get("reachable_positions", [])))) for scene in selected_scenes),
    }
    return selected


def build_stale_stress_memory_items(
    probe: Dict[str, Any],
    target_objects: Sequence[str] = TARGET_OBJECTS,
    age_gap: int = 48,
    stale_salience: float = 1.4,
    stale_volatility: float = 0.9,
) -> List[MemoryItem]:
    base_items = build_memory_items(probe)
    tasks = build_reachable_tasks(probe, target_objects=target_objects)
    stale_items: List[MemoryItem] = []

    for index, task in enumerate(tasks):
        stale_items.append(
            MemoryItem(
                object_name=task.target,
                location=_stale_location(task, index),
                observed_at=-(age_gap + len(tasks) - index),
                salience=stale_salience,
                volatility=stale_volatility,
                demand=1.0,
                failure_count=0,
            )
        )

    return stale_items + base_items


def _evaluate_with_items(
    tasks: Sequence[ReachableTask],
    items: Sequence[MemoryItem],
    policy: MemoryPolicy,
    budget: int,
) -> Dict[str, Any]:
    task_by_location = {task.true_location: task for task in tasks}
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)
    saved_reachable_cost = 0.0

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    for task in tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost
            saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost, 0.0)
        elif outcome.found and outcome.stale and outcome.item is not None:
            metrics.stale_errors += 1
            stale_task = task_by_location.get(outcome.item.location)
            stale_cost = stale_task.direct_action_cost if stale_task else task.direct_action_cost
            metrics.total_cost += stale_cost + task.scene_search_cost
        else:
            metrics.total_cost += task.scene_search_cost

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "ai2thor_stale_reachable_stress"
    row["memory_items"] = len(items)
    row["reachable_tasks"] = len(tasks)
    row["retained_items"] = len(store.items)
    row["avg_reachable_cost"] = round(metrics.avg_cost, 4)
    row["saved_reachable_cost"] = round(saved_reachable_cost, 4)
    row["completion_rate"] = row["success_rate"]
    return row


def evaluate_policy_stale_stress(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    age_gap: int = 48,
    stale_salience: float = 1.4,
    stale_volatility: float = 0.9,
) -> Dict[str, Any]:
    tasks = build_reachable_tasks(probe, target_objects=target_objects)
    base_items = build_memory_items(probe)
    items = build_stale_stress_memory_items(
        probe,
        target_objects=target_objects,
        age_gap=age_gap,
        stale_salience=stale_salience,
        stale_volatility=stale_volatility,
    )
    row = _evaluate_with_items(tasks, items, policy=policy, budget=budget)
    row["stale_distractors"] = len(items) - len(base_items)
    return row


def policy_suite() -> tuple[MemoryPolicy, ...]:
    return (
        NoMemoryPolicy(),
        SaliencePolicy(),
        TaskConditionedPolicy(),
        FreshnessAwarePolicy(),
        AI2ThorCalibratedPolicy(),
        WeightedAI2ThorPolicy("stale_aware_weighted", DEFAULT_STALE_AWARE_WEIGHTS),
    )


def run_stale_stress(
    probe: Dict[str, Any],
    budgets: Sequence[int],
    policies: Iterable[MemoryPolicy] | None = None,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    age_gap: int = 48,
    stale_salience: float = 1.4,
    stale_volatility: float = 0.9,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for policy in policies or policy_suite():
            rows.append(
                evaluate_policy_stale_stress(
                    probe,
                    policy=policy,
                    budget=budget,
                    target_objects=target_objects,
                    age_gap=age_gap,
                    stale_salience=stale_salience,
                    stale_volatility=stale_volatility,
                )
            )
    return rows


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["budget"]), str(row["policy"])), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (budget, policy), group in sorted(grouped.items()):
        summary.append(
            {
                "budget": budget,
                "policy": policy,
                "runs": len(group),
                "reachable_tasks": int(group[0]["reachable_tasks"]) if group else 0,
                "memory_items": int(group[0]["memory_items"]) if group else 0,
                "stale_distractors": int(group[0]["stale_distractors"]) if group else 0,
                "retained_items": round(sum(float(r["retained_items"]) for r in group) / len(group), 2),
                "completion_rate": round(sum(float(r["completion_rate"]) for r in group) / len(group), 4),
                "avg_reachable_cost": round(sum(float(r["avg_reachable_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
                "saved_reachable_cost": round(sum(float(r["saved_reachable_cost"]) for r in group) / len(group), 4),
            }
        )
    return summary


def write_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (out_dir / "ai2thor_stale_stress.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_stale_stress_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "reachable_tasks",
            "memory_items",
            "stale_distractors",
            "retained_items",
            "completion_rate",
            "avg_reachable_cost",
            "query_hit_rate",
            "stale_errors",
            "saved_reachable_cost",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")


def render_readme(summary: Sequence[Dict[str, Any]]) -> str:
    best = max(summary, key=lambda row: (float(row["completion_rate"]), -float(row["stale_errors"]), -float(row["avg_reachable_cost"])), default=None)
    lines = [
        "# AI2-THOR Stale-Memory Stress Benchmark",
        "",
        "Mode: `ai2thor_stale_reachable_stress`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['policy']}` at budget `{best['budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Average reachable cost: `{best['avg_reachable_cost']}`",
                f"Stale errors: `{best['stale_errors']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This derived benchmark injects older same-target stale memory distractors into an AI2-THOR reachable-position probe.",
            "It is intended to test whether a policy can avoid high-salience but outdated memories under a tight memory budget.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR stale-memory reachable-position stress benchmark.")
    parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_reachable_benchmark_9scene/ai2thor_reachable_probe.json"))
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--age-gap", type=int, default=48)
    parser.add_argument("--stale-salience", type=float, default=1.4)
    parser.add_argument("--stale-volatility", type=float, default=0.9)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_stale_stress"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = select_probe_scenes(load_probe(args.probe), scenes=args.scenes)
    rows = run_stale_stress(
        probe,
        budgets=args.budgets,
        age_gap=args.age_gap,
        stale_salience=args.stale_salience,
        stale_volatility=args.stale_volatility,
    )
    write_outputs(rows, args.out_dir)
    print(json.dumps(summarize_rows(rows), indent=2))


if __name__ == "__main__":
    main()
