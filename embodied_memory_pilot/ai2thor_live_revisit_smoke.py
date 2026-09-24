"""Minimal AI2-THOR live revisit smoke for proactive maintenance.

This module performs one controller-backed revisit action and writes an
auditable artifact. It is deliberately narrower than a task benchmark: an OK
run only proves that a controller could revisit one remembered location and
collect a post-revisit observation.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from embodied_memory_pilot.ai2thor_adapter import (
DEFAULT_OBJECTS,
    DEFAULT_PROBE_ACTIONS,
    CapabilityReport,
    ai2thor_capability,
    configure_build_mirror,
    merge_visible_objects,
    platform_class,
    visible_objects_from_event,
)


JsonValue = object
VisibleObject = dict[str, JsonValue]
ResultRow = dict[str, JsonValue]
Result = dict[str, JsonValue]

SCHEMA_VERSION = "ai2thor_live_revisit_smoke.v1"
NO_CONTROLLER_BOUNDARY = "AI2-THOR controller unavailable; no live revisit evidence"
LIVE_REVISIT_BOUNDARY = "single-scene AI2-THOR controller revisit smoke; not task success, not policy superiority, not perception-backed unless verifier_used=true"
ROW_BOUNDARY = "controller-level live revisit only unless verifier_used=true"
VISIBILITY_PROBE_ACTIONS = ("Pass", "RotateRight", "RotateRight", "RotateRight", "RotateRight")


class ControllerLike(Protocol):
    def step(self, action: str, **kwargs: object) -> object: ...

    def stop(self) -> None: ...


ControllerFactory = Callable[..., ControllerLike]


@dataclass(frozen=True)
class ParsedArgs:
    scene: str
    out_dir: Path
    width: int
    height: int
    platform_name: str | None
    build_base_url: str | None
    actions: tuple[str, ...]
    random_seed: int


def select_revisit_target(
    visible_objects: Sequence[Mapping[str, JsonValue]],
    target_object_types: tuple[str, ...] = DEFAULT_OBJECTS,
) -> VisibleObject | None:
    candidates: list[VisibleObject] = []
    for obj in visible_objects:
        position = obj.get("position")
        if not isinstance(position, Mapping):
            continue
        if not all(key in position for key in ("x", "y", "z")):
            continue
        candidates.append(dict(obj))

    if not candidates:
        return None

    target_rank = {object_type: i for i, object_type in enumerate(target_object_types)}

    def rank(obj: VisibleObject) -> tuple[int, int, str]:
        object_type = str(obj.get("object_type") or "")
        return (
            target_rank.get(object_type, len(target_rank)),
            0 if bool(obj.get("pickupable")) else 1,
            str(obj.get("object_id") or obj.get("name") or object_type),
        )

    return sorted(candidates, key=rank)[0]


def remembered_location(scene: str, obj: Mapping[str, JsonValue]) -> str:
    object_id = str(obj.get("object_id") or obj.get("name") or obj.get("object_type") or "Unknown")
    return f"{scene}:{object_id}@live-memory"


def _to_float(value: JsonValue, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _to_int(value: JsonValue, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def _coordinates(raw: Mapping[str, JsonValue], y_default: float = 0.0) -> dict[str, float]:
    return {
        "x": _to_float(raw.get("x", 0.0)),
        "y": _to_float(raw.get("y", y_default), y_default),
        "z": _to_float(raw.get("z", 0.0)),
    }


def _position(obj: Mapping[str, JsonValue]) -> dict[str, float]:
    raw = obj.get("position")
    if not isinstance(raw, Mapping):
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return _coordinates(cast(dict[str, JsonValue], raw))


def _reachable_positions(raw_positions: Sequence[object]) -> list[dict[str, float]]:
    positions: list[dict[str, float]] = []
    for raw in raw_positions:
        if isinstance(raw, Mapping) and all(key in raw for key in ("x", "z")):
            positions.append(_coordinates(cast(dict[str, JsonValue], raw)))
    return positions


def _ground_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def nearest_revisit_position(
    remembered_position: Mapping[str, float],
    reachable_positions: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], str, bool, float]:
    if not reachable_positions:
        return dict(remembered_position), "remembered_position", False, 0.0
    nearest = min(reachable_positions, key=lambda position: _ground_distance(remembered_position, position))
    distance = _ground_distance(remembered_position, nearest)
    return dict(nearest), "nearest_reachable", True, distance


def _round_distance(distance: float) -> float:
    return round(distance, 6)


def _metadata(event: object) -> Mapping[str, JsonValue]:
    metadata = getattr(event, "metadata", {}) or {}
    return cast(Mapping[str, JsonValue], metadata)


def _agent_metadata(metadata: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    agent = metadata.get("agent")
    return cast(dict[str, JsonValue], agent or {}) if isinstance(agent, Mapping) else {}


def _agent_position(agent: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    position = agent.get("position")
    return cast(dict[str, JsonValue], position or {}) if isinstance(position, Mapping) else {}


def _agent_rotation(agent: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    rotation = agent.get("rotation")
    return cast(dict[str, JsonValue], rotation or {"x": 0.0, "y": 0.0, "z": 0.0}) if isinstance(rotation, Mapping) else {"x": 0.0, "y": 0.0, "z": 0.0}


def _visible_types(objects: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    return sorted({str(obj.get("object_type") or "Unknown") for obj in objects})


def _target_visible(objects: Sequence[Mapping[str, JsonValue]], target: Mapping[str, JsonValue]) -> bool:
    target_id = str(target.get("object_id") or "")
    target_type = str(target.get("object_type") or "")
    for obj in objects:
        if target_id and str(obj.get("object_id") or "") == target_id:
            return True
        if target_type and str(obj.get("object_type") or "") == target_type:
            return True
    return False


def _find_object_by_type(
    objects: Sequence[Mapping[str, JsonValue]],
    object_type: str,
) -> VisibleObject | None:
    for obj in objects:
        if str(obj.get("object_type") or "") == object_type:
            return dict(obj)
    return None


def _probe_visibility(
    controller: ControllerLike,
    target: Mapping[str, JsonValue],
    actions: Sequence[str] = VISIBILITY_PROBE_ACTIONS,
) -> tuple[list[VisibleObject], list[str], bool, str, dict[str, float] | None]:
    collected_frames: list[list[VisibleObject]] = []
    executed_actions: list[str] = []
    for action in actions:
        event = controller.step(action=action)
        executed_actions.append(action)
        visible = [dict(obj) for obj in visible_objects_from_event(event)]
        collected_frames.append(visible)
        if _target_visible(visible, target):
            merged = merge_visible_objects(collected_frames)
            found = _find_object_by_type(visible, str(target.get("object_type") or ""))
            found_position = _position(found) if found else None
            return merged, executed_actions, True, "", found_position

    merged = merge_visible_objects(collected_frames)
    return merged, executed_actions, False, "target_not_visible_after_revisit_sweep", None


def blocked_result(
    scene: str,
    out_dir: Path,
    capability: CapabilityReport,
    platform_name: str | None,
    build_base_url: str | None,
    blocker_type: str,
    error_message: str,
) -> Result:
    result: Result = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "scene": scene,
        "platform_name": platform_name,
        "build_base_url": build_base_url,
        "capability": capability.as_dict(),
        "startup_attempts": [
            {
                "platform_name": platform_name or "default",
                "success": False,
                "blocker_type": blocker_type,
                "error_message": error_message,
            }
        ],
        "summary": {
            "records": 0,
            "blocked": True,
            "degraded": False,
            "live_controller_started": False,
            "initial_random_spawn_attempted": False,
            "initial_random_spawn_success": False,
            "teleport_attempted": False,
            "teleport_success": False,
            "post_revisit_observation_collected": False,
            "verifier_used": False,
            "perception_backed_success": False,
        },
        "rows": [],
        "bridge_rows": [],
        "claim_boundary": NO_CONTROLLER_BOUNDARY,
    }
    write_outputs(result, out_dir)
    return result


def _default_controller_factory(**kwargs: object) -> ControllerLike:
    controller_module = importlib.import_module("ai2thor.controller")  # pragma: no cover
    controller_type = cast(Callable[..., ControllerLike], getattr(controller_module, "Controller"))  # pragma: no cover
    return controller_type(**kwargs)


def _make_controller_kwargs(
    scene: str,
    width: int,
    height: int,
    platform_name: str | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"scene": scene, "width": width, "height": height}
    selected_platform = cast(object, platform_class(platform_name))
    if selected_platform is not None:
        kwargs["platform"] = selected_platform
    return kwargs


def _collect_memory(controller: ControllerLike, actions: Sequence[str]) -> tuple[list[VisibleObject], Mapping[str, JsonValue]]:
    frames: list[list[VisibleObject]] = []
    latest_metadata: Mapping[str, JsonValue] = {}
    for action in actions:
        event = controller.step(action=action)
        latest_metadata = _metadata(event)
        frames.append([dict(obj) for obj in visible_objects_from_event(event)])
    return merge_visible_objects(frames), latest_metadata


def _row(
    scene: str,
    target: Mapping[str, JsonValue],
    revisit_position: Mapping[str, float],
    revisit_position_source: str,
    reachable_positions_available: bool,
    remembered_to_revisit_distance: float,
    teleport_success: bool,
    teleport_error_message: str,
    post_visible: Sequence[Mapping[str, JsonValue]],
    visibility_probe_actions: Sequence[str],
    target_visible_after_revisit: bool,
    visibility_failure_reason: str,
    target_moved_by_spawn: bool = False,
    target_post_spawn_position: Mapping[str, float] | None = None,
    target_movement_distance: float | None = None,
    target_post_spawn_visible: bool = False,
    oracle_teleport_attempted: bool = False,
    oracle_teleport_success: bool = False,
    oracle_teleport_source: str | None = None,
    oracle_teleport_position: Mapping[str, float] | None = None,
    oracle_teleport_error: str = "",
    found_by_sweep_position: Mapping[str, float] | None = None,
    perception_claim: str = "not_attempted",
) -> ResultRow:
    target_type = str(target.get("object_type") or "Unknown")
    target_id = str(target.get("object_id") or target.get("name") or target_type)
    return {
        "trace_id": f"{scene}:{target_id}:live-revisit",
        "scene": scene,
        "target_object_id": target_id,
        "target_object_type": target_type,
        "remembered_location": remembered_location(scene, target),
        "remembered_position": _position(target),
        "revisit_position": dict(revisit_position),
        "revisit_position_source": revisit_position_source,
        "reachable_positions_available": reachable_positions_available,
        "remembered_to_revisit_distance": _round_distance(remembered_to_revisit_distance),
        "before_visible": True,
        "after_random_spawn_visible": target_post_spawn_visible,
        "revisit_action": "TeleportFull",
        "teleport_attempted": True,
        "teleport_success": teleport_success,
        "teleport_error_message": teleport_error_message,
        "target_moved_by_spawn": target_moved_by_spawn,
        "target_post_spawn_position": dict(target_post_spawn_position) if target_post_spawn_position else None,
        "target_movement_distance": _round_distance(target_movement_distance) if target_movement_distance is not None else None,
        "target_post_spawn_visible": target_post_spawn_visible,
        "visibility_probe_actions": list(visibility_probe_actions),
        "visibility_probe_count": len(visibility_probe_actions),
        "target_visible_after_revisit": target_visible_after_revisit,
        "visibility_failure_reason": visibility_failure_reason,
        "post_revisit_visible_object_types": _visible_types(post_visible),
        "post_revisit_target_visible_by_metadata": target_visible_after_revisit,
        "verifier_used": False,
        "decision_stale": None,
        "decision_confidence": None,
        "decision_reason": None,
        "policy_action": "revisit_memory_location",
        "policy_reason": "minimal proactive-maintenance live smoke",
        "perception_claim": perception_claim,
        "found_by_sweep_position": dict(found_by_sweep_position) if found_by_sweep_position else None,
        "oracle_teleport_attempted": oracle_teleport_attempted,
        "oracle_teleport_success": oracle_teleport_success,
        "oracle_teleport_source": oracle_teleport_source,
        "oracle_teleport_position": dict(oracle_teleport_position) if oracle_teleport_position else None,
        "oracle_teleport_error": oracle_teleport_error,
        "claim_boundary": ROW_BOUNDARY,
    }


def run_live_revisit_smoke(
    scene: str,
    out_dir: Path,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS,
    random_seed: int = 7,
    capability: CapabilityReport | None = None,
    controller_factory: ControllerFactory | None = None,
) -> Result:
    capability = capability or ai2thor_capability()
    if not capability.available:
        return blocked_result(
            scene=scene,
            out_dir=out_dir,
            capability=capability,
            platform_name=platform_name,
            build_base_url=build_base_url,
            blocker_type="ai2thor_unavailable",
            error_message=str(capability.blocker or "ai2thor is unavailable"),
        )

    controller_factory = controller_factory or _default_controller_factory
    try:
        configure_build_mirror(build_base_url)
        controller = controller_factory(**_make_controller_kwargs(scene, width, height, platform_name))
    except Exception as exc:  # pragma: no cover - depends on AI2-THOR runtime
        return blocked_result(
            scene=scene,
            out_dir=out_dir,
            capability=capability,
            platform_name=platform_name,
            build_base_url=build_base_url,
            blocker_type="controller_start_failed",
            error_message=str(exc),
        )

    startup_attempts: list[dict[str, JsonValue]] = [
        {"platform_name": platform_name or "default", "success": True, "blocker_type": "", "error_message": ""}
    ]
    try:
        reachable_event = controller.step(action="GetReachablePositions")
        reachable_metadata = _metadata(reachable_event)
        raw_reachable_positions = reachable_metadata.get("actionReturn")
        raw_reachable_objects: Sequence[object] = cast(list[object], raw_reachable_positions) if isinstance(raw_reachable_positions, list) else []
        reachable_positions = _reachable_positions(raw_reachable_objects)
        before_visible, before_metadata = _collect_memory(controller, actions)
        target = select_revisit_target(before_visible)
        if target is None:
            return blocked_result(
                scene=scene,
                out_dir=out_dir,
                capability=capability,
                platform_name=platform_name,
                build_base_url=build_base_url,
                blocker_type="no_revisit_target",
                error_message="No visible target object with a valid position was available.",
            )

        spawn_event = controller.step(action="InitialRandomSpawn", randomSeed=random_seed, forceVisible=True, numPlacementAttempts=5)
        spawn_metadata = _metadata(spawn_event)
        spawn_success = bool(spawn_metadata.get("lastActionSuccess", True))
        # Track where the target object actually went after spawn.
        spawn_visible = [dict(obj) for obj in visible_objects_from_event(spawn_event)]
        target_type_str = str(target.get("object_type") or "")
        target_post_spawn_obj = _find_object_by_type(spawn_visible, target_type_str)
        target_post_spawn_visible = target_post_spawn_obj is not None
        target_position = _position(target)
        if target_post_spawn_obj is not None:
            post_spawn_pos = _position(target_post_spawn_obj)
            movement_distance = _ground_distance(target_position, post_spawn_pos)
            target_moved = movement_distance > 0.01
        else:
            post_spawn_pos = None
            movement_distance = None
            target_moved = False

        oracle_position: dict[str, float] | None = None
        if not target_post_spawn_visible:
            _, _, found_by_sweep, _, sweep_pos = _probe_visibility(controller, target)
            if found_by_sweep and sweep_pos is not None:
                oracle_position = dict(sweep_pos)
                pre_teleport_sweep_found = True
                post_spawn_pos = oracle_position
                target_post_spawn_visible = True
                target_moved = _ground_distance(target_position, oracle_position) > 0.01
        elif post_spawn_pos is not None:
            oracle_position = dict(post_spawn_pos)
        agent = _agent_metadata(spawn_metadata) or _agent_metadata(before_metadata)
        agent_position = _agent_position(agent)
        rotation = _agent_rotation(agent)
        horizon = _to_float(agent.get("cameraHorizon", 30.0), 30.0)

        revisit_position: dict[str, float] = dict(target_position)
        revisit_position_source = "remembered_position"
        reachable_available = False
        revisit_distance = 0.0

        oracle_teleport_attempted: bool = False
        oracle_teleport_success: bool = False
        oracle_teleport_source: str | None = None
        oracle_teleport_position: dict[str, float] | None = None
        oracle_teleport_error: str = ""

        if oracle_position is not None:
            oracle_revisit, oracle_source, oracle_available, oracle_distance = nearest_revisit_position(
                oracle_position,
                reachable_positions,
            )
            oracle_teleport_event = controller.step(
                action="TeleportFull",
                x=oracle_revisit["x"],
                y=_to_float(agent_position.get("y", oracle_revisit["y"]), oracle_revisit["y"]),
                z=oracle_revisit["z"],
                rotation=rotation,
                horizon=horizon,
                standing=True,
            )
            oracle_metadata = _metadata(oracle_teleport_event)
            oracle_teleport_attempted = True
            oracle_teleport_success = bool(oracle_metadata.get("lastActionSuccess", True))
            oracle_teleport_source = "post_spawn_oracle"
            oracle_teleport_position = dict(oracle_revisit)
            oracle_teleport_error = str(oracle_metadata.get("errorMessage") or "")
            teleport_event = oracle_teleport_event
            revisit_position = dict(oracle_revisit)
            revisit_position_source = oracle_source
            reachable_available = oracle_available
            revisit_distance = oracle_distance
        else:
            revisit_position, revisit_position_source, reachable_available, revisit_distance = nearest_revisit_position(
                target_position,
                reachable_positions,
            )
            teleport_event = controller.step(
                action="TeleportFull",
                x=revisit_position["x"],
                y=_to_float(agent_position.get("y", revisit_position["y"]), revisit_position["y"]),
                z=revisit_position["z"],
                rotation=rotation,
                horizon=horizon,
                standing=True,
            )
        teleport_metadata = _metadata(teleport_event)
        teleport_success = bool(teleport_metadata.get("lastActionSuccess", True))
        teleport_error = str(teleport_metadata.get("errorMessage") or "")
        post_visible, visibility_actions, target_visible_after_revisit, visibility_failure_reason, sweep_found_position = _probe_visibility(controller, target)
        perception_claim: str = "not_attempted"
        if target_visible_after_revisit and sweep_found_position is not None:
            perception_claim = "metadata_found_target_at_revisit"
        row = _row(
            scene,
            target,
            revisit_position,
            revisit_position_source,
            reachable_available,
            revisit_distance,
            teleport_success,
            teleport_error,
            post_visible,
            visibility_actions,
            target_visible_after_revisit,
            visibility_failure_reason,
            target_moved_by_spawn=target_moved,
            target_post_spawn_position=post_spawn_pos,
            target_movement_distance=movement_distance,
            target_post_spawn_visible=target_post_spawn_visible,
            oracle_teleport_attempted=oracle_teleport_attempted,
            oracle_teleport_success=oracle_teleport_success,
            oracle_teleport_source=oracle_teleport_source,
            oracle_teleport_position=oracle_teleport_position,
            oracle_teleport_error=oracle_teleport_error,
            found_by_sweep_position=sweep_found_position,
            perception_claim=perception_claim,
        )
        status = "ok" if teleport_success else "degraded"
        result: Result = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "scene": scene,
            "platform_name": platform_name,
            "build_base_url": build_base_url,
            "random_seed": random_seed,
            "actions": list(actions),
            "capability": capability.as_dict(),
            "startup_attempts": startup_attempts,
            "reachable_count": len(reachable_positions),
            "summary": {
                "records": 1,
                "blocked": False,
                "degraded": status == "degraded",
                "live_controller_started": True,
                "initial_random_spawn_attempted": True,
                "initial_random_spawn_success": spawn_success,
                "target_moved_by_spawn": target_moved,
                "teleport_attempted": True,
                "teleport_success": teleport_success,
                "post_revisit_observation_collected": True,
                "target_visible_after_revisit": target_visible_after_revisit,
                "verifier_used": False,
                "perception_backed_success": False,
            },
            "rows": [row],
            "bridge_rows": [],
            "claim_boundary": LIVE_REVISIT_BOUNDARY,
        }
    finally:
        controller.stop()

    write_outputs(result, out_dir)
    return result


def render_readme(result: Mapping[str, JsonValue]) -> str:
    summary = cast(Mapping[str, JsonValue], result.get("summary", {}))
    lines = [
        "# AI2-THOR Live Revisit Smoke",
        "",
        f"Status: `{result.get('status')}`",
        f"Scene: `{result.get('scene')}`",
        f"Records: `{summary.get('records', 0)}`",
        f"Live controller started: `{summary.get('live_controller_started', False)}`",
        f"Target moved by spawn: `{summary.get('target_moved_by_spawn', False)}`",
        f"Teleport attempted: `{summary.get('teleport_attempted', False)}`",
        f"Teleport success: `{summary.get('teleport_success', False)}`",
        f"Verifier used: `{summary.get('verifier_used', False)}`",
        "",
        "This smoke executes one AI2-THOR controller revisit action toward one remembered object location.",
        "It also tracks whether InitialRandomSpawn moved the target and, if so, records its new position and movement distance.",
        "It is controller-level instrumentation only: it is not task completion, not policy superiority, and not perception-backed unless verifier_used=true.",
        f"Claim boundary: `{result.get('claim_boundary')}`",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: Mapping[str, JsonValue], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = (out_dir / "live_revisit_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = cast(Sequence[Mapping[str, JsonValue]], result.get("rows", []))
    with (out_dir / "live_revisit_smoke.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _ = (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def parse_args() -> ParsedArgs:
    parser = argparse.ArgumentParser(description="Run one minimal AI2-THOR live revisit smoke.")
    _ = parser.add_argument("--scene", default="FloorPlan1")
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_live_revisit_smoke"))
    _ = parser.add_argument("--width", type=int, default=300)
    _ = parser.add_argument("--height", type=int, default=300)
    _ = parser.add_argument("--platform", dest="platform_name", default="CloudRendering")
    _ = parser.add_argument("--build-base-url", default=None)
    _ = parser.add_argument("--actions", nargs="+", default=list(DEFAULT_PROBE_ACTIONS))
    _ = parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()
    namespace = cast(dict[str, object], vars(args))
    raw_actions = cast(Sequence[object], namespace["actions"])
    return ParsedArgs(
        scene=str(namespace["scene"]),
        out_dir=cast(Path, namespace["out_dir"]),
        width=_to_int(namespace["width"], 300),
        height=_to_int(namespace["height"], 300),
        platform_name=cast(str | None, namespace["platform_name"]),
        build_base_url=cast(str | None, namespace["build_base_url"]),
        actions=tuple(str(action) for action in raw_actions),
        random_seed=_to_int(namespace["random_seed"], 7),
    )


def main() -> None:
    args = parse_args()
    result = run_live_revisit_smoke(
        scene=args.scene,
        out_dir=args.out_dir,
        width=args.width,
        height=args.height,
        platform_name=args.platform_name,
        build_base_url=args.build_base_url,
        actions=args.actions,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
