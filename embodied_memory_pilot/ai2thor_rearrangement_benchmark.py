from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from embodied_memory_pilot.ai2thor_adapter import (
    DEFAULT_PROBE_ACTIONS,
    ai2thor_capability,
    configure_build_mirror,
    merge_visible_objects,
    platform_class,
    visible_objects_from_event,
)
from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS, object_features, policy_suite
from embodied_memory_pilot.ai2thor_reachable_benchmark import (
    DEFAULT_AGENT_POSITION,
    reachable_path_distance,
    reachable_scene_search_cost,
)
from embodied_memory_pilot.pilot import MemoryItem, MemoryPolicy, MemoryStore, NoMemoryPolicy, PolicyMetrics


@dataclass(frozen=True)
class RearrangementTask:
    target: str
    true_location: str
    old_location: str
    scene: str
    direct_action_cost: float
    old_action_cost: float
    scene_search_cost: float
    rearranged: bool


def load_probe(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def object_key(obj: Dict[str, Any]) -> str:
    object_type = str(obj.get("object_type") or "Unknown")
    return str(obj.get("object_id") or obj.get("name") or object_type)


def rearrangement_location(scene_name: str, obj: Dict[str, Any], phase: str) -> str:
    return f"{scene_name}:{object_key(obj)}@{phase}"


def _ground_tuple(obj: Dict[str, Any]) -> Tuple[float, float]:
    position = obj.get("position") or {}
    return (round(float(position.get("x", 0.0)), 4), round(float(position.get("z", 0.0)), 4))


def object_moved(before: Dict[str, Any], after: Dict[str, Any], threshold: float = 0.05) -> bool:
    bx, bz = _ground_tuple(before)
    ax, az = _ground_tuple(after)
    return abs(bx - ax) > threshold or abs(bz - az) > threshold


def _paired_objects(scene: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    before_objects = list(scene.get("before_visible_objects", []))
    after_objects = list(scene.get("after_visible_objects", []))
    before_by_id = {object_key(obj): obj for obj in before_objects}
    after_by_id = {object_key(obj): obj for obj in after_objects}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    paired_before: set[str] = set()
    paired_after: set[str] = set()
    for key, before in before_by_id.items():
        after = after_by_id.get(key)
        if after is not None:
            pairs.append((before, after))
            paired_before.add(key)
            paired_after.add(key)

    before_by_type: Dict[str, List[Dict[str, Any]]] = {}
    after_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for obj in before_objects:
        if object_key(obj) in paired_before:
            continue
        before_by_type.setdefault(str(obj.get("object_type") or "Unknown"), []).append(obj)
    for obj in after_objects:
        if object_key(obj) in paired_after:
            continue
        after_by_type.setdefault(str(obj.get("object_type") or "Unknown"), []).append(obj)

    for object_type, before_candidates in before_by_type.items():
        after_candidates = after_by_type.get(object_type, [])
        if len(before_candidates) == 1 and len(after_candidates) == 1:
            pairs.append((before_candidates[0], after_candidates[0]))
    return pairs


def summarize_rearrangement_probe(probe: Dict[str, Any]) -> Dict[str, Any]:
    ok_scenes = 0
    moved = 0
    stable = 0
    paired = 0
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        ok_scenes += 1
        for before, after in _paired_objects(scene):
            paired += 1
            if object_moved(before, after):
                moved += 1
            else:
                stable += 1
    return {
        "scenes": len(probe.get("scenes", [])),
        "ok_scenes": ok_scenes,
        "paired_objects": paired,
        "moved_objects": moved,
        "stable_objects": stable,
    }


def build_rearrangement_memory_items(probe: Dict[str, Any]) -> List[MemoryItem]:
    items: List[MemoryItem] = []
    observed_at = 0
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        for obj in scene.get("before_visible_objects", []):
            object_type = str(obj.get("object_type") or "Unknown")
            features = object_features(obj)
            items.append(
                MemoryItem(
                    object_name=object_type,
                    location=rearrangement_location(scene_name, obj, "before"),
                    observed_at=observed_at,
                    salience=features["salience"],
                    volatility=features["volatility"],
                    demand=features["demand"],
                )
            )
            observed_at += 1
    return items


def _scene_visible_after(scene: Dict[str, Any]) -> Dict[str, Any]:
    projected = dict(scene)
    projected["visible_objects"] = list(scene.get("after_visible_objects", []))
    return projected


def build_rearrangement_tasks(
    probe: Dict[str, Any],
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> List[RearrangementTask]:
    target_set = set(target_objects)
    tasks: List[RearrangementTask] = []
    seen: set[str] = set()
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        agent_position = scene.get("agent_position") or DEFAULT_AGENT_POSITION
        reachable_positions = scene.get("reachable_positions", [])
        search_cost = reachable_scene_search_cost(_scene_visible_after(scene), agent_position)
        for before, after in _paired_objects(scene):
            object_type = str(after.get("object_type") or "Unknown")
            if object_type not in target_set and not bool(after.get("pickupable")):
                continue
            key = object_key(after)
            if key in seen:
                continue
            seen.add(key)
            direct_cost = round(1.0 + reachable_path_distance(agent_position, after.get("position", {}), reachable_positions), 4)
            old_cost = round(1.0 + reachable_path_distance(agent_position, before.get("position", {}), reachable_positions), 4)
            moved = object_moved(before, after)
            tasks.append(
                RearrangementTask(
                    target=object_type,
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


def evaluate_policy_rearrangement(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> Dict[str, Any]:
    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe, target_objects=target_objects)
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)
    saved_reachable_cost = 0.0

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    task_by_old_location = {task.old_location: task for task in tasks}
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
            stale_task = task_by_old_location.get(outcome.item.location)
            old_cost = stale_task.old_action_cost if stale_task else task.old_action_cost
            metrics.total_cost += old_cost + task.scene_search_cost
        else:
            metrics.total_cost += task.scene_search_cost

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "ai2thor_rearrangement_replay"
    row["memory_items"] = len(items)
    row["rearrangement_tasks"] = len(tasks)
    row["moved_tasks"] = sum(1 for task in tasks if task.rearranged)
    row["rearranged_tasks"] = row["moved_tasks"]
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
            rows.append(evaluate_policy_rearrangement(probe, policy=policy, budget=budget))
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
                "rearrangement_tasks": int(group[0]["rearrangement_tasks"]) if group else 0,
                "moved_tasks": int(group[0]["moved_tasks"]) if group else 0,
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


def write_outputs(rows: Sequence[Dict[str, Any]], probe: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (out_dir / "ai2thor_rearrangement_benchmark.json").write_text(
        json.dumps({"probe_summary": summarize_rearrangement_probe(probe), "runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_rearrangement_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "rearrangement_tasks",
            "moved_tasks",
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
    (out_dir / "README.md").write_text(render_readme(summary, probe), encoding="utf-8")


def render_readme(summary: Sequence[Dict[str, Any]], probe: Dict[str, Any]) -> str:
    probe_summary = summarize_rearrangement_probe(probe)
    best = max(summary, key=lambda row: (float(row["completion_rate"]), -float(row["stale_errors"]), -float(row["avg_reachable_cost"])), default=None)
    lines = [
        "# AI2-THOR Rearrangement Benchmark",
        "",
        "Mode: `ai2thor_rearrangement_replay`",
        f"OK scenes: `{probe_summary['ok_scenes']}`",
        f"Moved paired objects: `{probe_summary['moved_objects']}`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['policy']}` at budget `{best['budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Stale errors: `{best['stale_errors']}`",
                f"Average reachable cost: `{best['avg_reachable_cost']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _collect_visible(
    controller: Any,
    actions: Sequence[str],
    capture_images: bool = False,
    image_dir: Path | None = None,
    scene_name: str = "",
    phase: str = "before",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frames: List[List[Dict[str, Any]]] = []
    agent_position = DEFAULT_AGENT_POSITION
    for i, action in enumerate(actions):
        event = controller.step(action=action)
        frames.append(visible_objects_from_event(event))
        if capture_images and image_dir is not None:
            _save_object_images(event, image_dir, scene_name, phase, i)
        metadata = getattr(event, "metadata", {}) or {}
        agent_position = metadata.get("agent", {}).get("position", agent_position)
    return merge_visible_objects(frames), agent_position


def _save_object_images(
    event: Any,
    image_dir: Path,
    scene: str,
    phase: str,
    action_idx: int,
) -> None:
    """Save per-object cropped JPEGs using instance segmentation masks."""
    import numpy as np
    from PIL import Image

    frame = np.array(event.frame)
    if frame is None or frame.size == 0:
        return
    full_path = image_dir / scene / phase / f"{action_idx:02d}_frame.jpg"
    manifest_rows: List[Dict[str, Any]] = []
    masks = getattr(event, "instance_masks", None) or {}
    metadata = getattr(event, "metadata", {}) or {}
    objects = metadata.get("objects", [])
    for obj in objects:
        obj_id = str(obj.get("objectId") or "")
        if not obj_id or not obj.get("visible"):
            continue
        mask = masks.get(obj_id)
        if mask is not None:
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y1, y2 = int(ys.min()), int(ys.max())
                x1, x2 = int(xs.min()), int(xs.max())
                h, w = y2 - y1, x2 - x1
                pad_y, pad_x = max(int(h * 0.1), 2), max(int(w * 0.1), 2)
                y1 = max(0, y1 - pad_y); y2 = min(frame.shape[0], y2 + pad_y)
                x1 = max(0, x1 - pad_x); x2 = min(frame.shape[1], x2 + pad_x)
                if y2 > y1 and x2 > x1:
                    crop = frame[y1:y2, x1:x2]
                    out_path = image_dir / scene / phase / f"{action_idx:02d}_{obj_id}.jpg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(crop).save(out_path, quality=85)
                    manifest_rows.append(
                        {
                            "scene": scene,
                            "phase": phase,
                            "action_idx": action_idx,
                            "object_id": obj_id,
                            "object_type": str(obj.get("objectType") or obj.get("object_type") or ""),
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "frame_path": str(full_path.relative_to(image_dir)),
                            "crop_path": str(out_path.relative_to(image_dir)),
                        }
                    )
    # Also save full frame as fallback
    full_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(full_path, quality=85)
    if manifest_rows:
        manifest_path = image_dir / scene / phase / "image_manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def collect_rearrangement_scene(
    scene: str,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS,
    random_seed: int = 1,
    capture_images: bool = False,
    image_dir: Path | None = None,
) -> Dict[str, Any]:
    capability = ai2thor_capability()
    if not capability.available:
        return {
            "scene": scene,
            "status": "blocked",
            "capability": capability.as_dict(),
            "before_visible_objects": [],
            "after_visible_objects": [],
            "reachable_positions": [],
        }

    from ai2thor.controller import Controller  # type: ignore  # pragma: no cover

    try:
        configure_build_mirror(build_base_url)
        controller_kwargs: Dict[str, Any] = {"scene": scene, "width": width, "height": height}
        if capture_images:
            controller_kwargs["renderInstanceSegmentation"] = True
        selected_platform = platform_class(platform_name)
        if selected_platform is not None:
            controller_kwargs["platform"] = selected_platform
        controller = Controller(**controller_kwargs)
    except Exception as exc:  # pragma: no cover
        return {
            "scene": scene,
            "status": "blocked",
            "capability": capability.as_dict(),
            "blocker": f"AI2-THOR Controller failed to start: {exc}",
            "before_visible_objects": [],
            "after_visible_objects": [],
            "reachable_positions": [],
        }

    try:
        reachable_event = controller.step(action="GetReachablePositions")
        reachable_positions = (getattr(reachable_event, "metadata", {}) or {}).get("actionReturn") or []
        before_visible, agent_position = _collect_visible(controller, actions, capture_images=capture_images, image_dir=image_dir, scene_name=scene, phase="before")
        rearrange_event = controller.step(action="InitialRandomSpawn", randomSeed=random_seed, forceVisible=True, numPlacementAttempts=5)
        rearrange_metadata = getattr(rearrange_event, "metadata", {}) or {}
        after_visible, agent_position = _collect_visible(controller, actions, capture_images=capture_images, image_dir=image_dir, scene_name=scene, phase="after")
        result = {
            "scene": scene,
            "status": "ok",
            "capability": capability.as_dict(),
            "agent_position": agent_position,
            "reachable_positions": reachable_positions,
            "reachable_count": len(reachable_positions),
            "before_visible_objects": before_visible,
            "after_visible_objects": after_visible,
            "before_visible_count": len(before_visible),
            "after_visible_count": len(after_visible),
            "rearrange_action": {
                "action": "InitialRandomSpawn",
                "success": bool(rearrange_metadata.get("lastActionSuccess", True)),
                "error_message": rearrange_metadata.get("errorMessage", ""),
                "random_seed": random_seed,
            },
        }
        result["summary"] = summarize_rearrangement_probe({"scenes": [result]})
        return result
    finally:
        controller.stop()


def collect_rearrangement_probe(
    scenes: Sequence[str],
    out_dir: Path,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS,
    random_seed: int = 1,
    capture_images: bool = False,
) -> Dict[str, Any]:
    image_dir = out_dir / "images" if capture_images else None
    scene_results = [
        collect_rearrangement_scene(
            scene=scene,
            width=width,
            height=height,
            platform_name=platform_name,
            build_base_url=build_base_url,
            actions=actions,
            random_seed=random_seed,
            capture_images=capture_images,
            image_dir=image_dir,
        )
        for scene in scenes
    ]
    result = {
        "status": "ok" if all(scene.get("status") == "ok" for scene in scene_results) else "partial",
        "scenes": scene_results,
        "summary": summarize_rearrangement_probe({"scenes": scene_results}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ai2thor_rearrangement_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR before/after rearrangement memory benchmark.")
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--collect-scenes", nargs="+", default=None)
    parser.add_argument("--actions", nargs="+", default=None)
    parser.add_argument("--platform", dest="platform_name", default=None)
    parser.add_argument("--build-base-url", default=None)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--capture-images", action="store_true", default=False)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_rearrangement_benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.collect_scenes:
        probe = collect_rearrangement_probe(
            scenes=args.collect_scenes,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            platform_name=args.platform_name,
            build_base_url=args.build_base_url,
            actions=tuple(args.actions or DEFAULT_PROBE_ACTIONS),
            random_seed=args.random_seed,
            capture_images=args.capture_images,
        )
    else:
        probe_path = args.probe or Path("results/ai2thor_rearrangement_benchmark/ai2thor_rearrangement_probe.json")
        probe = load_probe(probe_path)

    rows = run_benchmark(probe, budgets=args.budgets)
    write_outputs(rows, probe, args.out_dir)
    print(json.dumps({"probe_summary": summarize_rearrangement_probe(probe), "summary": summarize_rows(rows)}, indent=2))


if __name__ == "__main__":
    main()
