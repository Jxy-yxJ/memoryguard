from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_memory_eval import (
    TARGET_OBJECTS,
    build_memory_items,
    object_features,
    policy_suite,
)
from embodied_memory_pilot.pilot import MemoryPolicy, MemoryStore, NoMemoryPolicy, PolicyMetrics


DEFAULT_AGENT_POSITION = {"x": 0.0, "y": 0.0, "z": 0.0}


@dataclass(frozen=True)
class ActionTask:
    target: str
    true_location: str
    scene: str
    direct_action_cost: float
    scene_search_cost: float


def load_probe(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def navigation_distance(start: Dict[str, Any] | None, goal: Dict[str, Any] | None) -> float:
    start = start or DEFAULT_AGENT_POSITION
    goal = goal or DEFAULT_AGENT_POSITION
    dx = float(goal.get("x", 0.0)) - float(start.get("x", 0.0))
    dz = float(goal.get("z", 0.0)) - float(start.get("z", 0.0))
    return round(math.hypot(dx, dz), 4)


def object_location(scene_name: str, obj: Dict[str, Any]) -> str:
    object_type = str(obj.get("object_type") or "Unknown")
    object_id = str(obj.get("object_id") or obj.get("name") or object_type)
    return f"{scene_name}:{object_id}"


def scene_search_cost(scene: Dict[str, Any], agent_position: Dict[str, Any]) -> float:
    distances = [
        navigation_distance(agent_position, obj.get("position"))
        for obj in scene.get("visible_objects", [])
        if obj.get("position")
    ]
    if not distances:
        return 6.0
    # A miss means the agent must scan a substantial part of the observed scene.
    return round(max(distances) + 0.35 * len(distances) + 2.0, 4)


def build_action_tasks(
    probe: Dict[str, Any],
    target_objects: Sequence[str] = TARGET_OBJECTS,
    agent_position: Dict[str, Any] | None = None,
) -> List[ActionTask]:
    target_set = set(target_objects)
    agent = agent_position or DEFAULT_AGENT_POSITION
    tasks: List[ActionTask] = []
    seen_locations: set[str] = set()
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        search_cost = scene_search_cost(scene, agent)
        for obj in scene.get("visible_objects", []):
            object_type = str(obj.get("object_type") or "Unknown")
            if object_type not in target_set and not bool(obj.get("pickupable")):
                continue
            location = object_location(scene_name, obj)
            if location in seen_locations:
                continue
            seen_locations.add(location)
            direct_cost = round(1.0 + navigation_distance(agent, obj.get("position")), 4)
            tasks.append(
                ActionTask(
                    target=object_type,
                    true_location=location,
                    scene=scene_name,
                    direct_action_cost=direct_cost,
                    scene_search_cost=max(search_cost, direct_cost + 1.0),
                )
            )
    return tasks


def evaluate_policy_action_costs(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> Dict[str, Any]:
    items = build_memory_items(probe)
    tasks = build_action_tasks(probe, target_objects=target_objects)
    task_by_location = {task.true_location: task for task in tasks}
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)
    saved_action_cost = 0.0

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    for task in tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost
            saved_action_cost += max(task.scene_search_cost - task.direct_action_cost, 0.0)
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
    row["mode"] = "ai2thor_action_cost_replay"
    row["memory_items"] = len(items)
    row["action_tasks"] = len(tasks)
    row["retained_items"] = len(store.items)
    row["avg_action_cost"] = round(metrics.avg_cost, 4)
    row["saved_action_cost"] = round(saved_action_cost, 4)
    row["completion_rate"] = row["success_rate"]
    return row


def run_benchmark(
    probe: Dict[str, Any],
    budgets: Sequence[int],
    policies: Iterable[MemoryPolicy] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for policy in policies or policy_suite():
            rows.append(evaluate_policy_action_costs(probe, policy=policy, budget=budget))
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
                "action_tasks": int(group[0]["action_tasks"]) if group else 0,
                "memory_items": int(group[0]["memory_items"]) if group else 0,
                "retained_items": round(sum(float(r["retained_items"]) for r in group) / len(group), 2),
                "completion_rate": round(sum(float(r["completion_rate"]) for r in group) / len(group), 4),
                "avg_action_cost": round(sum(float(r["avg_action_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
                "saved_action_cost": round(sum(float(r["saved_action_cost"]) for r in group) / len(group), 4),
            }
        )
    return summary


def write_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (out_dir / "ai2thor_action_benchmark.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_action_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "action_tasks",
            "memory_items",
            "retained_items",
            "completion_rate",
            "avg_action_cost",
            "query_hit_rate",
            "stale_errors",
            "saved_action_cost",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")


def render_readme(summary: Sequence[Dict[str, Any]]) -> str:
    best = max(summary, key=lambda row: (float(row["completion_rate"]), -float(row["avg_action_cost"])), default=None)
    lines = [
        "# AI2-THOR Action-Cost Benchmark",
        "",
        "Mode: `ai2thor_action_cost_replay`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['policy']}` at budget `{best['budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Average action cost: `{best['avg_action_cost']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This benchmark estimates action-level search cost from AI2-THOR object coordinates.",
            "A memory hit pays direct navigation cost; a miss pays scene scan cost.",
            "It is a replay benchmark and should later be replaced with controller-level reachable-position navigation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate action-level AI2-THOR memory benchmark costs.")
    parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_scene_probe_360/ai2thor_scene_probe.json"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_action_benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = load_probe(args.probe)
    rows = run_benchmark(probe, budgets=args.budgets)
    write_outputs(rows, args.out_dir)
    print(json.dumps(summarize_rows(rows), indent=2))


if __name__ == "__main__":
    main()
