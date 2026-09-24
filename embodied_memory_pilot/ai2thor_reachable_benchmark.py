from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from embodied_memory_pilot.ai2thor_action_benchmark import DEFAULT_AGENT_POSITION, object_location
from embodied_memory_pilot.ai2thor_adapter import (
    DEFAULT_PROBE_ACTIONS,
    ai2thor_capability,
    configure_build_mirror,
    merge_visible_objects,
    object_memory_summary,
    platform_class,
    visible_objects_from_event,
)
from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS, build_memory_items, policy_suite
from embodied_memory_pilot.pilot import MemoryPolicy, MemoryStore, NoMemoryPolicy, PolicyMetrics


@dataclass(frozen=True)
class ReachableTask:
    target: str
    true_location: str
    scene: str
    direct_action_cost: float
    scene_search_cost: float


def load_probe(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ground_point(position: Dict[str, Any]) -> Tuple[float, float]:
    return (float(position.get("x", 0.0)), float(position.get("z", 0.0)))


def ground_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax, az = ground_point(a)
    bx, bz = ground_point(b)
    return math.hypot(ax - bx, az - bz)


def nearest_position(target: Dict[str, Any], positions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not positions:
        return dict(target)
    return min(positions, key=lambda position: ground_distance(target, position))


def _node_key(position: Dict[str, Any]) -> Tuple[float, float]:
    x, z = ground_point(position)
    return (round(x, 4), round(z, 4))


def _step_size(nodes: Sequence[Dict[str, Any]]) -> float:
    distances: List[float] = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            dist = ground_distance(a, b)
            if dist > 1e-6:
                distances.append(dist)
    return min(distances) if distances else 0.25


def reachable_path_distance(
    start: Dict[str, Any],
    goal: Dict[str, Any],
    reachable_positions: Sequence[Dict[str, Any]],
) -> float:
    if not reachable_positions:
        return round(ground_distance(start, goal), 4)

    start_node = nearest_position(start, reachable_positions)
    goal_node = nearest_position(goal, reachable_positions)
    start_key = _node_key(start_node)
    goal_key = _node_key(goal_node)
    if start_key == goal_key:
        return 0.0

    step = _step_size(reachable_positions)
    neighbor_limit = step * 1.05
    nodes = {_node_key(position): position for position in reachable_positions}
    adjacency: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]] = {
        key: [] for key in nodes
    }
    keys = list(nodes.keys())
    for index, key_a in enumerate(keys):
        pos_a = nodes[key_a]
        for key_b in keys[index + 1 :]:
            pos_b = nodes[key_b]
            dist = ground_distance(pos_a, pos_b)
            if dist <= neighbor_limit + 1e-6:
                adjacency[key_a].append((key_b, dist))
                adjacency[key_b].append((key_a, dist))

    queue = deque([(start_key, 0.0)])
    seen = {start_key}
    while queue:
        current, distance = queue.popleft()
        if current == goal_key:
            return round(distance, 4)
        for neighbor, edge_cost in adjacency.get(current, []):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + edge_cost))

    return round(ground_distance(start_node, goal_node), 4)


def reachable_scene_search_cost(scene: Dict[str, Any], agent_position: Dict[str, Any]) -> float:
    reachable_positions = scene.get("reachable_positions", [])
    distances = [
        reachable_path_distance(agent_position, obj.get("position", {}), reachable_positions)
        for obj in scene.get("visible_objects", [])
        if obj.get("position")
    ]
    if not distances:
        return 6.0
    return round(max(distances) + 0.35 * len(distances) + 2.0, 4)


def build_reachable_tasks(
    probe: Dict[str, Any],
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> List[ReachableTask]:
    target_set = set(target_objects)
    tasks: List[ReachableTask] = []
    seen_locations: set[str] = set()
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        agent_position = scene.get("agent_position") or DEFAULT_AGENT_POSITION
        reachable_positions = scene.get("reachable_positions", [])
        search_cost = reachable_scene_search_cost(scene, agent_position)
        for obj in scene.get("visible_objects", []):
            object_type = str(obj.get("object_type") or "Unknown")
            if object_type not in target_set and not bool(obj.get("pickupable")):
                continue
            location = object_location(scene_name, obj)
            if location in seen_locations:
                continue
            seen_locations.add(location)
            direct_cost = round(1.0 + reachable_path_distance(agent_position, obj.get("position", {}), reachable_positions), 4)
            tasks.append(
                ReachableTask(
                    target=object_type,
                    true_location=location,
                    scene=scene_name,
                    direct_action_cost=direct_cost,
                    scene_search_cost=max(search_cost, direct_cost + 1.0),
                )
            )
    return tasks


def evaluate_policy_reachable_costs(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> Dict[str, Any]:
    items = build_memory_items(probe)
    tasks = build_reachable_tasks(probe, target_objects=target_objects)
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
    row["mode"] = "ai2thor_reachable_navigation_replay"
    row["memory_items"] = len(items)
    row["reachable_tasks"] = len(tasks)
    row["retained_items"] = len(store.items)
    row["avg_reachable_cost"] = round(metrics.avg_cost, 4)
    row["saved_reachable_cost"] = round(saved_reachable_cost, 4)
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
            rows.append(evaluate_policy_reachable_costs(probe, policy=policy, budget=budget))
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
    (out_dir / "ai2thor_reachable_benchmark.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_reachable_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "reachable_tasks",
            "memory_items",
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
    best = max(summary, key=lambda row: (float(row["completion_rate"]), -float(row["avg_reachable_cost"])), default=None)
    lines = [
        "# AI2-THOR Reachable-Position Benchmark",
        "",
        "Mode: `ai2thor_reachable_navigation_replay`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['policy']}` at budget `{best['budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Average reachable cost: `{best['avg_reachable_cost']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This benchmark uses AI2-THOR reachable positions when available.",
            "A memory hit pays grid-path navigation cost to the nearest reachable point by the target object.",
            "A memory miss pays an observed-scene scan cost.",
            "",
        ]
    )
    return "\n".join(lines)


def collect_reachable_scene(
    scene: str,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS,
) -> Dict[str, Any]:
    capability = ai2thor_capability()
    if not capability.available:
        return {
            "status": "blocked",
            "scene": scene,
            "capability": capability.as_dict(),
            "visible_objects": [],
            "reachable_positions": [],
        }

    from ai2thor.controller import Controller  # type: ignore  # pragma: no cover

    try:
        configure_build_mirror(build_base_url)
        controller_kwargs: Dict[str, Any] = {"scene": scene, "width": width, "height": height}
        selected_platform = platform_class(platform_name)
        if selected_platform is not None:
            controller_kwargs["platform"] = selected_platform
        controller = Controller(**controller_kwargs)
    except Exception as exc:  # pragma: no cover
        return {
            "status": "blocked",
            "scene": scene,
            "capability": capability.as_dict(),
            "blocker": f"AI2-THOR Controller failed to start: {exc}",
            "visible_objects": [],
            "reachable_positions": [],
        }

    try:
        reachable_event = controller.step(action="GetReachablePositions")
        reachable_positions = (getattr(reachable_event, "metadata", {}) or {}).get("actionReturn") or []
        agent_position = (getattr(reachable_event, "metadata", {}) or {}).get("agent", {}).get("position", DEFAULT_AGENT_POSITION)
        frames: List[List[Dict[str, Any]]] = []
        action_status: List[Dict[str, Any]] = []
        for action in actions:
            event = controller.step(action=action)
            frames.append(visible_objects_from_event(event))
            metadata = getattr(event, "metadata", {}) or {}
            agent_position = metadata.get("agent", {}).get("position", agent_position)
            action_status.append(
                {
                    "action": action,
                    "success": bool(metadata.get("lastActionSuccess", True)),
                    "error_message": metadata.get("errorMessage", ""),
                }
            )
        visible = merge_visible_objects(frames)
        return {
            "status": "ok",
            "scene": scene,
            "capability": capability.as_dict(),
            "agent_position": agent_position,
            "reachable_positions": reachable_positions,
            "reachable_count": len(reachable_positions),
            "visible_objects": visible,
            "visible_count": len(visible),
            "actions": list(actions),
            "action_status": action_status,
            "memory_summary": object_memory_summary(visible),
            "target_object_types": list(TARGET_OBJECTS),
        }
    finally:
        controller.stop()


def collect_reachable_probe(
    scenes: Sequence[str],
    out_dir: Path,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS,
) -> Dict[str, Any]:
    scene_results = [
        collect_reachable_scene(
            scene=scene,
            width=width,
            height=height,
            platform_name=platform_name,
            build_base_url=build_base_url,
            actions=actions,
        )
        for scene in scenes
    ]
    result = {
        "status": "ok" if all(scene.get("status") == "ok" for scene in scene_results) else "partial",
        "scenes": scene_results,
        "summary": {
            "scenes": len(scene_results),
            "ok_scenes": sum(1 for scene in scene_results if scene.get("status") == "ok"),
            "total_visible": sum(int(scene.get("visible_count", 0)) for scene in scene_results),
            "total_reachable": sum(int(scene.get("reachable_count", 0)) for scene in scene_results),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ai2thor_reachable_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR reachable-position memory benchmark.")
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--collect-scenes", nargs="+", default=None)
    parser.add_argument("--actions", nargs="+", default=None)
    parser.add_argument("--platform", dest="platform_name", default=None)
    parser.add_argument("--build-base-url", default=None)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_reachable_benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.collect_scenes:
        probe = collect_reachable_probe(
            scenes=args.collect_scenes,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            platform_name=args.platform_name,
            build_base_url=args.build_base_url,
            actions=tuple(args.actions or DEFAULT_PROBE_ACTIONS),
        )
    else:
        probe_path = args.probe or Path("results/ai2thor_reachable_benchmark/ai2thor_reachable_probe.json")
        probe = load_probe(probe_path)

    rows = run_benchmark(probe, budgets=args.budgets)
    write_outputs(rows, args.out_dir)
    print(json.dumps(summarize_rows(rows), indent=2))


if __name__ == "__main__":
    main()
