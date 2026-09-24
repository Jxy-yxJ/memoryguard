from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from embodied_memory_pilot.ai2thor_adapter import CapabilityReport, ai2thor_capability, configure_build_mirror, platform_class

SCHEMA_VERSION = "ai2thor_navigation_smoke.v1"
CLAIM_BOUNDARY = (
    "single-case non-shortcut navigation mechanism instrumentation; measured path forbids TeleportFull; "
    "not SPL statistics, not navigation success rate, not broad active-maintenance evidence"
)

Json = dict[str, Any]
Position = dict[str, float]


class ControllerLike(Protocol):
    def step(self, action: str, **kwargs: object) -> object: ...

    def stop(self) -> None: ...


ControllerFactory = Callable[..., ControllerLike]


@dataclass(frozen=True)
class NavigationSmokeConfig:
    scene: str = "FloorPlan1"
    seed: int = 29
    target: str = "Apple"
    remembered_position: Position | None = None
    goal_position: Position | None = None
    max_steps: int = 24
    move_step_size: float = 0.25
    reach_tolerance: float = 0.13
    width: int = 300
    height: int = 300
    platform_name: str | None = "CloudRendering"
    build_base_url: str | None = None
    out_dir: Path = Path("results/0514_navigation_smoke")


def _default_remembered_position() -> Position:
    return {"x": -0.46516576409339905, "y": 1.151225209236145, "z": 0.47580063343048096}


def _default_goal_position() -> Position:
    return {"x": -0.05909985303878784, "y": 1.157825231552124, "z": -0.25377392768859863}


def _metadata(event: object) -> Mapping[str, Any]:
    raw = getattr(event, "metadata", {}) or {}
    return cast(Mapping[str, Any], raw)


def _agent_position(metadata: Mapping[str, Any]) -> Position:
    agent = metadata.get("agent")
    if isinstance(agent, Mapping):
        pos = agent.get("position")
        if isinstance(pos, Mapping):
            return {"x": float(pos.get("x", 0.0)), "y": float(pos.get("y", 0.9)), "z": float(pos.get("z", 0.0))}
    return {"x": 0.0, "y": 0.9, "z": 0.0}


def _object_position(obj: Mapping[str, Any]) -> Position | None:
    pos = obj.get("position")
    if not isinstance(pos, Mapping):
        return None
    return {"x": float(pos.get("x", 0.0)), "y": float(pos.get("y", 0.9)), "z": float(pos.get("z", 0.0))}


def _objects(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = metadata.get("objects", [])
    if not isinstance(raw, list):
        return []
    return [cast(Mapping[str, Any], item) for item in raw if isinstance(item, Mapping)]


def _target_visible(metadata: Mapping[str, Any], target: str) -> bool:
    return any(str(obj.get("objectType") or obj.get("name")) == target and bool(obj.get("visible")) for obj in _objects(metadata))


def _distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return math.sqrt((float(a.get("x", 0.0)) - float(b.get("x", 0.0))) ** 2 + (float(a.get("z", 0.0)) - float(b.get("z", 0.0))) ** 2)


def _round_position(pos: Mapping[str, float]) -> Position:
    return {"x": round(float(pos.get("x", 0.0)), 4), "y": round(float(pos.get("y", 0.9)), 4), "z": round(float(pos.get("z", 0.0)), 4)}


def _reachable_positions(raw: object) -> list[Position]:
    if not isinstance(raw, list):
        return []
    positions: list[Position] = []
    for item in raw:
        if isinstance(item, Mapping) and "x" in item and "z" in item:
            positions.append({"x": float(item.get("x", 0.0)), "y": float(item.get("y", 0.9)), "z": float(item.get("z", 0.0))})
    return positions


def _agent_heading(metadata: Mapping[str, Any]) -> float:
    agent = metadata.get("agent")
    if isinstance(agent, Mapping):
        rotation = agent.get("rotation")
        if isinstance(rotation, Mapping):
            return float(rotation.get("y", 0.0)) % 360.0
    return 0.0


def _goal_heading(current: Mapping[str, float], goal: Mapping[str, float]) -> float:
    dx = float(goal.get("x", 0.0)) - float(current.get("x", 0.0))
    dz = float(goal.get("z", 0.0)) - float(current.get("z", 0.0))
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _signed_heading_diff(current_heading: float, goal_heading: float) -> float:
    diff = (goal_heading - current_heading) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def nearest_reachable_point(goal: Mapping[str, float], reachable: Sequence[Mapping[str, float]]) -> tuple[Position, float]:
    if not reachable:
        raise ValueError("reachable positions are required")
    selected = min(reachable, key=lambda pos: _distance(goal, pos))
    point = {"x": float(selected.get("x", 0.0)), "y": float(selected.get("y", 0.9)), "z": float(selected.get("z", 0.0))}
    return point, _distance(goal, point)


def _default_controller_factory(**kwargs: object) -> ControllerLike:
    controller_module = importlib.import_module("ai2thor.controller")  # pragma: no cover
    controller_type = cast(Callable[..., ControllerLike], getattr(controller_module, "Controller"))  # pragma: no cover
    return controller_type(**kwargs)


def _make_controller_kwargs(config: NavigationSmokeConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {"scene": config.scene, "width": config.width, "height": config.height}
    try:
        selected_platform = cast(object, platform_class(config.platform_name))
    except Exception:
        selected_platform = None
    if selected_platform is not None:
        kwargs["platform"] = selected_platform
    return kwargs


def blocked_result(config: NavigationSmokeConfig, capability: CapabilityReport, blocker_type: str, error_message: str) -> Json:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "capability": capability.as_dict(),
        "blocker_type": blocker_type,
        "error_message": error_message,
        "config": _config_json(config),
        "row": None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _config_json(config: NavigationSmokeConfig) -> Json:
    raw = asdict(config)
    raw["out_dir"] = str(config.out_dir)
    return raw


def _navigation_step(controller: ControllerLike, current: Position, goal: Position, current_heading: float) -> tuple[object, float, bool, float, str]:
    goal_heading = _goal_heading(current, goal)
    diff = _signed_heading_diff(current_heading, goal_heading)
    if abs(diff) > 45.0:
        action = "RotateRight" if diff > 0 else "RotateLeft"
        event = controller.step(action)
        meta = _metadata(event)
        new_heading = _agent_heading(meta)
        success = bool(meta.get("lastActionSuccess", True))
        return event, 0.0, success, new_heading, action
    before = dict(current)
    event = controller.step("MoveAhead")
    meta = _metadata(event)
    after = _agent_position(meta)
    moved_distance = _distance(before, after)
    success = bool(meta.get("lastActionSuccess", True)) and moved_distance > 1e-9
    new_heading = _agent_heading(meta)
    return event, moved_distance, success, new_heading, "MoveAhead"


def _visibility_sweep(controller: ControllerLike, target: str) -> tuple[bool, list[str]]:
    actions = ["Pass", "RotateRight", "RotateRight", "RotateRight", "RotateRight"]
    used: list[str] = []
    for action in actions:
        event = controller.step(action)
        used.append(action)
        if _target_visible(_metadata(event), target):
            return True, used
    return False, used


def run_navigation_smoke(
    config: NavigationSmokeConfig,
    *,
    controller_factory: ControllerFactory | None = None,
    capability: CapabilityReport | None = None,
) -> Json:
    capability = capability or ai2thor_capability()
    if not capability.available:
        return blocked_result(config, capability, "ai2thor_unavailable", str(capability.blocker or "ai2thor unavailable"))

    make_controller = controller_factory or _default_controller_factory
    try:
        configure_build_mirror(config.build_base_url)
        controller = make_controller(**_make_controller_kwargs(config))
    except Exception as exc:  # pragma: no cover - depends on AI2-THOR runtime
        return blocked_result(config, capability, "controller_start_failed", str(exc))

    actions: list[str] = []
    collision_count = 0
    path_steps = 0
    path_length = 0.0
    failure_reason: str | None = None
    reached = False
    visible_at_destination = False
    start_pose: Position | None = None
    nearest_point: Position | None = None
    nearest_distance: float | None = None
    final_position: Position | None = None
    visibility_actions: list[str] = []

    try:
        reachable_event = controller.step("GetReachablePositions")
        actions.append("GetReachablePositions")
        reachable = _reachable_positions(_metadata(reachable_event).get("actionReturn"))
        goal = dict(config.goal_position or _default_goal_position())
        remembered = dict(config.remembered_position or _default_remembered_position())
        if not reachable:
            failure_reason = "no_reachable_positions"
            start_pose = _agent_position(_metadata(reachable_event))
        else:
            nearest_point, nearest_distance = nearest_reachable_point(goal, reachable)
            spawn_event = controller.step("InitialRandomSpawn", randomSeed=config.seed, forceVisible=True, numPlacementAttempts=5)
            actions.append("InitialRandomSpawn")
            start_meta = _metadata(spawn_event)
            start_pose = _agent_position(start_meta)
            current_heading = _agent_heading(start_meta)
            current = dict(start_pose)
            for _ in range(config.max_steps):
                if _distance(current, nearest_point) <= config.reach_tolerance:
                    reached = True
                    break
                event, moved, success, current_heading, action = _navigation_step(controller, current, nearest_point, current_heading)
                actions.append(action)
                if action == "MoveAhead":
                    path_steps += 1
                    path_length += moved
                    if not success:
                        collision_count += 1
                        failure_reason = "collision_stuck"
                        break
                    current = _agent_position(_metadata(event))
            if not reached and failure_reason is None:
                reached = _distance(current, nearest_point) <= config.reach_tolerance
            if not reached and failure_reason is None:
                failure_reason = "max_steps_exceeded"
            final_position = dict(current)
            visible_at_destination, visibility_actions = _visibility_sweep(controller, config.target)
            actions.extend(visibility_actions)
            if reached and not visible_at_destination:
                failure_reason = "target_not_visible"
        used_teleportfull = any(action == "TeleportFull" for action in actions)
        row: Json = {
            "scene": config.scene,
            "seed": config.seed,
            "target": config.target,
            "remembered_position": _round_position(remembered),
            "goal_position": _round_position(goal),
            "nearest_reachable_point": _round_position(nearest_point) if nearest_point is not None else None,
            "goal_to_nearest_reachable_distance": round(float(nearest_distance), 4) if nearest_distance is not None else None,
            "start_pose": _round_position(start_pose or {"x": 0.0, "y": 0.9, "z": 0.0}),
            "final_position": _round_position(final_position or start_pose or {"x": 0.0, "y": 0.9, "z": 0.0}),
            "reached": reached,
            "path_steps": path_steps,
            "path_length_euclidean": round(path_length, 4),
            "collision_count": collision_count,
            "visible_at_destination": visible_at_destination,
            "failure_reason": failure_reason,
            "visibility_probe_actions": visibility_actions,
            "used_teleportfull_for_measured_path": used_teleportfull,
            "measured_path_actions": actions,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "capability": capability.as_dict(),
            "config": _config_json(config),
            "row": row,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - defensive around simulator runtime
        return blocked_result(config, capability, "navigation_smoke_failed", str(exc))
    finally:
        controller.stop()


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "navigation_smoke.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    row = result.get("row")
    if isinstance(row, Mapping):
        fieldnames = sorted(str(key) for key in row.keys())
        with (out_dir / "navigation_smoke.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: row.get(key) for key in fieldnames})
    (out_dir / "README.md").write_text(
        "# 0514 Navigation Smoke\n\n"
        "Single-case non-shortcut navigation mechanism instrumentation. The measured path forbids TeleportFull and does not support SPL statistics, navigation success rate, or broad active-maintenance claims.\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal non-shortcut AI2-THOR navigation smoke.")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--target", default="Apple")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--platform-name", default="CloudRendering")
    parser.add_argument("--build-base-url", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/0514_navigation_smoke"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = NavigationSmokeConfig(
        scene=args.scene,
        seed=args.seed,
        target=args.target,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        platform_name=args.platform_name,
        build_base_url=args.build_base_url,
        out_dir=args.out_dir,
    )
    result = run_navigation_smoke(config)
    write_outputs(result, config.out_dir)


if __name__ == "__main__":
    main()
