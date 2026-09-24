from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS, policy_suite
from embodied_memory_pilot.ai2thor_reachable_benchmark import (
    DEFAULT_AGENT_POSITION,
    reachable_path_distance,
    reachable_scene_search_cost,
)
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    _paired_objects,
    build_rearrangement_memory_items,
    load_probe,
    object_moved,
    rearrangement_location,
    summarize_rearrangement_probe,
)
from embodied_memory_pilot.ai2thor_rearrangement_verification import VerifyRearrangementRiskPolicy
from embodied_memory_pilot.pilot import MemoryItem, MemoryPolicy, MemoryStore, NoMemoryPolicy, PolicyMetrics


@dataclass(frozen=True)
class InteractionTask:
    target: str
    action: str
    true_location: str
    old_location: str
    scene: str
    direct_action_cost: float
    old_action_cost: float
    scene_search_cost: float
    rearranged: bool


class VerifyInteractionRiskPolicy(VerifyRearrangementRiskPolicy):
    name = "verify_interaction_risk"


def _interaction_action(obj: Dict[str, Any]) -> str | None:
    if bool(obj.get("pickupable")):
        return "PickupObject"
    if bool(obj.get("openable")):
        return "OpenObject"
    return None


def _scene_visible_after(scene: Dict[str, Any]) -> Dict[str, Any]:
    projected = dict(scene)
    projected["visible_objects"] = list(scene.get("after_visible_objects", []))
    return projected


def build_interaction_tasks(
    probe: Dict[str, Any],
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> List[InteractionTask]:
    target_set = set(target_objects)
    tasks: List[InteractionTask] = []
    seen: set[tuple[str, str, str]] = set()
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        agent_position = scene.get("agent_position") or DEFAULT_AGENT_POSITION
        reachable_positions = scene.get("reachable_positions", [])
        search_cost = reachable_scene_search_cost(_scene_visible_after(scene), agent_position)
        for before, after in _paired_objects(scene):
            object_type = str(after.get("object_type") or "Unknown")
            action = _interaction_action(after)
            if action is None:
                continue
            if action == "PickupObject" and object_type not in target_set and not bool(after.get("pickupable")):
                continue
            key = (scene_name, object_type, action)
            if key in seen:
                continue
            seen.add(key)
            moved = object_moved(before, after)
            direct_cost = round(1.0 + reachable_path_distance(agent_position, after.get("position", {}), reachable_positions), 4)
            old_cost = round(1.0 + reachable_path_distance(agent_position, before.get("position", {}), reachable_positions), 4)
            tasks.append(
                InteractionTask(
                    target=object_type,
                    action=action,
                    true_location=rearrangement_location(scene_name, after, "after") if moved else rearrangement_location(scene_name, before, "before"),
                    old_location=rearrangement_location(scene_name, before, "before"),
                    scene=scene_name,
                    direct_action_cost=direct_cost,
                    old_action_cost=old_cost,
                    scene_search_cost=max(search_cost, direct_cost + 1.0),
                    rearranged=moved,
                )
            )
    return tasks


def evaluate_policy_interaction(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    verification_cost: float = 0.5,
    action_cost: float = 0.5,
) -> Dict[str, Any]:
    items = build_rearrangement_memory_items(probe)
    tasks = build_interaction_tasks(probe, target_objects=target_objects)
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)
    saved_reachable_cost = 0.0
    interaction_stale_errors = 0

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    query_now = max((item.observed_at for item in items), default=0) + 1
    for task in tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=query_now):
            metrics.verifications += 1
            metrics.total_cost += verification_cost
            if outcome.stale:
                metrics.verification_catches += 1
                metrics.successes += 1
                metrics.total_cost += task.scene_search_cost + action_cost
            else:
                metrics.successes += 1
                metrics.query_hits += 1
                metrics.total_cost += task.direct_action_cost + action_cost
                saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost - verification_cost, 0.0)
        elif outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost + action_cost
            saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost, 0.0)
        elif outcome.found and outcome.stale:
            metrics.stale_errors += 1
            interaction_stale_errors += 1
            metrics.total_cost += task.old_action_cost + task.scene_search_cost + action_cost
        else:
            metrics.total_cost += task.scene_search_cost + action_cost

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "ai2thor_interaction_memory"
    row["memory_items"] = len(items)
    row["interaction_tasks"] = len(tasks)
    row["pickup_tasks"] = sum(1 for task in tasks if task.action == "PickupObject")
    row["open_tasks"] = sum(1 for task in tasks if task.action == "OpenObject")
    row["moved_tasks"] = sum(1 for task in tasks if task.rearranged)
    row["retained_items"] = len(store.items)
    row["avg_interaction_cost"] = round(metrics.avg_cost, 4)
    row["avg_reachable_cost"] = row["avg_interaction_cost"]
    row["saved_reachable_cost"] = round(saved_reachable_cost, 4)
    row["completion_rate"] = row["success_rate"]
    row["interaction_stale_errors"] = interaction_stale_errors
    row["verification_cost"] = verification_cost
    row["action_cost"] = action_cost
    return row


def run_interaction_benchmark(
    probe: Dict[str, Any],
    budgets: Sequence[int],
    policies: Iterable[MemoryPolicy] | None = None,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    verification_cost: float = 0.5,
    action_cost: float = 0.5,
) -> List[Dict[str, Any]]:
    selected_policies = tuple(policies) if policies is not None else (*policy_suite(), VerifyInteractionRiskPolicy())
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for policy in selected_policies:
            rows.append(
                evaluate_policy_interaction(
                    probe,
                    policy=policy,
                    budget=budget,
                    target_objects=target_objects,
                    verification_cost=verification_cost,
                    action_cost=action_cost,
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
                "interaction_tasks": int(group[0]["interaction_tasks"]) if group else 0,
                "pickup_tasks": int(group[0]["pickup_tasks"]) if group else 0,
                "open_tasks": int(group[0]["open_tasks"]) if group else 0,
                "moved_tasks": int(group[0]["moved_tasks"]) if group else 0,
                "memory_items": int(group[0]["memory_items"]) if group else 0,
                "retained_items": round(sum(float(r["retained_items"]) for r in group) / len(group), 2),
                "completion_rate": round(sum(float(r["completion_rate"]) for r in group) / len(group), 4),
                "avg_interaction_cost": round(sum(float(r["avg_interaction_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "interaction_stale_errors": round(sum(float(r["interaction_stale_errors"]) for r in group) / len(group), 2),
                "verifications": round(sum(float(r["verifications"]) for r in group) / len(group), 2),
                "verification_catches": round(sum(float(r["verification_catches"]) for r in group) / len(group), 2),
                "saved_reachable_cost": round(sum(float(r["saved_reachable_cost"]) for r in group) / len(group), 4),
            }
        )
    return summary


def write_outputs(rows: Sequence[Dict[str, Any]], probe: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (out_dir / "ai2thor_interaction_memory.json").write_text(
        json.dumps({"probe_summary": summarize_rearrangement_probe(probe), "runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_interaction_memory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "interaction_tasks",
            "pickup_tasks",
            "open_tasks",
            "moved_tasks",
            "memory_items",
            "retained_items",
            "completion_rate",
            "avg_interaction_cost",
            "query_hit_rate",
            "interaction_stale_errors",
            "verifications",
            "verification_catches",
            "saved_reachable_cost",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "README.md").write_text(render_readme(summary, probe), encoding="utf-8")


def render_readme(summary: Sequence[Dict[str, Any]], probe: Dict[str, Any]) -> str:
    probe_summary = summarize_rearrangement_probe(probe)
    best = max(
        summary,
        key=lambda row: (
            float(row["completion_rate"]),
            -float(row["interaction_stale_errors"]),
            -float(row["avg_interaction_cost"]),
            -float(row["verifications"]),
        ),
        default=None,
    )
    lines = [
        "# AI2-THOR Interaction Memory Benchmark",
        "",
        "Mode: `ai2thor_interaction_memory`",
        f"OK scenes: `{probe_summary['ok_scenes']}`",
        f"Moved paired objects: `{probe_summary['moved_objects']}`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['policy']}` at budget `{best['budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Interaction stale errors: `{best['interaction_stale_errors']}`",
                f"Verifications: `{best['verifications']}`",
                f"Average interaction cost: `{best['avg_interaction_cost']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This benchmark reuses real AI2-THOR rearrangement observations as interaction precondition memory tasks.",
            "A memory hit must localize the object before a lightweight PickupObject/OpenObject action can succeed.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR interaction-precondition memory benchmark.")
    parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_rearrangement_6scene_seed7/ai2thor_rearrangement_probe.json"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--verification-cost", type=float, default=0.5)
    parser.add_argument("--action-cost", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_interaction_memory"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = load_probe(args.probe)
    rows = run_interaction_benchmark(
        probe,
        budgets=args.budgets,
        verification_cost=args.verification_cost,
        action_cost=args.action_cost,
    )
    write_outputs(rows, probe, args.out_dir)
    print(json.dumps({"probe_summary": summarize_rearrangement_probe(probe), "summary": summarize_rows(rows)}, indent=2))


if __name__ == "__main__":
    main()
