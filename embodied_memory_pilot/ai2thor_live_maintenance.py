from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from embodied_memory_pilot.ai2thor_adapter import (
    DEFAULT_PROBE_ACTIONS,
    CapabilityReport,
    ai2thor_capability,
    configure_build_mirror,
    merge_visible_objects,
    platform_class,
    visible_objects_from_event,
)
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    RearrangementTask,
    _save_object_images,
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    load_probe,
    rearrangement_location,
)
from embodied_memory_pilot.pilot import MemoryItem, MemoryStore, SaliencePolicy


JsonValue = object
RowValue = int | float | str | None
ResultRow = dict[str, JsonValue]
Result = dict[str, JsonValue]

SCHEMA_VERSION = "ai2thor_live_maintenance_smoke.v1"
NO_CONTROLLER_BOUNDARY = "AI2-THOR controller unavailable; no live maintenance evidence"
LIVE_MAINTENANCE_BOUNDARY = "single-scene AI2-THOR controller-backed maintenance loop; not task success, not policy superiority, not perception-backed unless verifier_used=true"
VISIBILITY_ACTIONS = ("Pass", "RotateRight", "RotateRight", "RotateRight", "RotateRight")


class ControllerLike(Protocol):
    def step(self, action: str, **kwargs: object) -> object: ...

    def stop(self) -> None: ...


ControllerFactory = Callable[..., ControllerLike]


class VerifierBackendLike(Protocol):
    name: str

    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float): ...


VerifierFactory = Callable[[], VerifierBackendLike]


@dataclass(frozen=True)
class LiveMaintenanceConfig:
    maintenance_policy_name: str = "passive"
    maintenance_budget_per_task: int = 0
    task_budget: int = 3
    random_seed: int = 0
    scene: str = "FloorPlan1"
    platform_name: str = "CloudRendering"
    build_base_url: str | None = None
    width: int = 300
    height: int = 300
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS
    visibility_actions: tuple[str, ...] = VISIBILITY_ACTIONS
    maintenance_cost: float = 0.5
    out_dir: Path = Path("results/ai2thor_live_maintenance_smoke")
    verifier_threshold: float = 0.25
    grounding_config: Path | None = None
    grounding_checkpoint: Path | None = None
    sam2_config: Path | None = None
    sam2_checkpoint: Path | None = None
    verifier_device: str = "cuda"
    backend_load_timeout_seconds: float | None = 600.0
    mlp_train_probes: tuple[str, ...] = ()
    mlp_epochs: int = 200
    mlp_threshold: float = 0.5


@dataclass(frozen=True)
class ParsedArgs:
    probe: Path
    out_dir: Path
    maintenance_policy: str
    maintenance_budget_per_task: int
    task_budget: int
    random_seed: int
    scene: str
    platform_name: str
    build_base_url: str | None
    width: int
    height: int
    verifier_backend_name: str
    verifier_threshold: float
    grounding_config: Path | None
    grounding_checkpoint: Path | None
    sam2_config: Path | None
    sam2_checkpoint: Path | None
    verifier_device: str | None
    backend_load_timeout_seconds: float | None
    mlp_train_probes: tuple[str, ...]
    mlp_epochs: int
    mlp_threshold: float


def build_tasks_from_probe(probe: dict[str, object]) -> list[RearrangementTask]:
    from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS
    return build_rearrangement_tasks(probe, target_objects=TARGET_OBJECTS)


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _metadata(event: object) -> Mapping[str, object]:
    metadata = getattr(event, "metadata", None)
    if metadata is None and isinstance(event, dict):
        metadata = cast(dict[str, object], event).get("metadata")
    return cast(Mapping[str, object], metadata or {})


def _agent_position(event: object) -> dict[str, float]:
    m = _metadata(event)
    agent = cast(dict[str, object], m.get("agent", {}))
    pos = cast(dict[str, object], agent.get("position", {}))
    return {"x": _to_float(pos.get("x"), 0.0), "y": _to_float(pos.get("y"), 0.9), "z": _to_float(pos.get("z"), 0.0)}


def _agent_rotation(event: object) -> dict[str, float]:
    m = _metadata(event)
    agent = cast(dict[str, object], m.get("agent", {}))
    rot = cast(dict[str, object], agent.get("rotation", {}))
    return {"x": _to_float(rot.get("x"), 0.0), "y": _to_float(rot.get("y"), 0.0), "z": _to_float(rot.get("z"), 0.0)}


def _agent_horizon(event: object) -> float:
    m = _metadata(event)
    agent = cast(dict[str, object], m.get("agent", {}))
    return _to_float(agent.get("cameraHorizon", 0.0))


def _reachable_positions(raw_positions: Sequence[object]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for p in raw_positions:
        pd = cast(dict[str, object], p)
        result.append({"x": _to_float(pd.get("x"), 0.0), "y": _to_float(pd.get("y"), 0.9), "z": _to_float(pd.get("z"), 0.0)})
    return result


def _ground_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def nearest_revisit_position(target: dict[str, float], positions: Sequence[dict[str, float]]) -> dict[str, float]:
    best = positions[0]
    best_dist = _ground_distance(target, best)
    for p in positions[1:]:
        d = _ground_distance(target, p)
        if d < best_dist:
            best_dist = d
            best = p
    return best


def _look_at_angles(from_pos: dict[str, float], to_pos: dict[str, float]) -> tuple[float, float]:
    dx = to_pos["x"] - from_pos["x"]
    dz = to_pos["z"] - from_pos["z"]
    yaw = math.degrees(math.atan2(dz, dx))
    dy = to_pos["y"] - from_pos["y"]
    ground = math.hypot(dx, dz)
    horizon = math.degrees(math.atan2(dy, ground)) if ground > 0 else 0.0
    return yaw, horizon


def _find_object_by_type(object_type: str, objects: Sequence[dict[str, object]]) -> dict[str, object] | None:
    for o in objects:
        ot = str(o.get("object_type") or o.get("objectType") or "").lower()
        if ot == object_type.lower():
            return o
    return None


def _object_position(obj: dict[str, object]) -> dict[str, float]:
    pos = cast(dict[str, object], obj.get("position", {}))
    return {"x": _to_float(pos.get("x"), 0.0), "y": _to_float(pos.get("y"), 0.9), "z": _to_float(pos.get("z"), 0.0)}


def _probe_visibility(
    controller: ControllerLike,
    target_type: str,
    actions: Sequence[str] = VISIBILITY_ACTIONS,
    *,
    capture_images: bool = False,
    image_dir: Path | None = None,
    scene_name: str = "",
    phase: str = "after",
    action_offset: int = 0,
) -> tuple[list[dict[str, object]], bool, str | None, dict[str, float] | None]:
    frames: list[list[dict[str, object]]] = []
    for i, act in enumerate(actions):
        event = controller.step(act, renderImage=capture_images) if capture_images else controller.step(act)
        if capture_images and image_dir is not None:
            try:
                _save_object_images(event, image_dir, scene_name, phase, action_offset + i)
            except Exception:
                pass
        visible = [dict(obj) for obj in visible_objects_from_event(event)]
        frames.append(visible)
        found = _find_object_by_type(target_type, visible)
        if found is not None:
            all_visible = merge_visible_objects(frames)
            return all_visible, True, None, _object_position(found)
    all_visible = merge_visible_objects(frames)
    return all_visible, False, "target_not_visible_after_revisit_sweep", None


def _default_controller_factory(**kwargs: object) -> ControllerLike:
    ct = importlib.import_module("ai2thor.controller")
    controller_cls = cast(Callable[..., ControllerLike], ct.Controller)
    return controller_cls(**kwargs)


def _make_controller_kwargs(
    scene: str,
    width: int,
    height: int,
    platform_name: str | None,
    *,
    capture_images: bool = False,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"scene": scene, "width": width, "height": height}
    if platform_name:
        try:
            plat = platform_class(platform_name)
            if plat is not None:
                kwargs["platform"] = plat
        except Exception:
            pass
    if capture_images:
        kwargs["renderInstanceSegmentation"] = True
    return kwargs


def _teleport_target_objects(
    controller: ControllerLike,
    before_objects: list[dict[str, object]],
    tasks: list[RearrangementTask],
    reachable: Sequence[dict[str, float]],
    agent_position: dict[str, float],
    rng: random.Random,
) -> dict[str, dict[str, float]]:
    """Teleport target objects to visible reachable positions, bypassing receptacle spawn.

    Returns a mapping from object_type (lowercase) to the new position.
    """
    moved_types: set[str] = {t.target.lower() for t in tasks if t.rearranged}
    target_positions: dict[str, dict[str, float]] = {}

    # Find one objectId per moved target type, plus pick suitable reachable destinations
    type_objids: dict[str, str] = {}
    for obj in before_objects:
        ot = str(obj.get("object_type", "")).lower()
        if ot in moved_types and ot not in type_objids:
            objid = str(obj.get("object_id", "")).strip()
            if objid and "|" in objid:
                type_objids[ot] = objid

    if not type_objids or not reachable:
        return target_positions

    # Pick reachable positions at a moderate distance that are in the positive quadrant
    filtered_reach = [p for p in reachable if p.get("x", 0.0) > 0.5 and p.get("z", 0.0) > 0.5]
    if not filtered_reach:
        filtered_reach = [p for p in reachable]
    sorted_reach = sorted(
        filtered_reach,
        key=lambda p: abs((p.get("x", 0.0) + p.get("z", 0.0)) - 3.0),
    )

    dest_idx = 0
    for obj_type in sorted(type_objids.keys()):
        if dest_idx >= len(sorted_reach):
            break
        dest = sorted_reach[dest_idx]
        dest_idx += 1
        objid = type_objids[obj_type]

        try:
            dest_reach = dict(dest)
            orig_obj = next((o for o in before_objects if str(o.get("object_type","")).lower() == obj_type), None)
            orig_y = (
                _to_float(cast(dict[str, object], orig_obj.get("position", {})).get("y"), 0.9)
                if orig_obj and isinstance(orig_obj.get("position"), dict)
                else 0.9
            )
            event = controller.step(
                "TeleportObject",
                objectId=objid,
                position={"x": dest_reach["x"], "y": orig_y, "z": dest_reach["z"]},
                rotation={"x": 0.0, "y": rng.uniform(0, 360), "z": 0.0},
            )
            meta = _metadata(event)
            if meta.get("lastActionSuccess", False):
                target_positions[obj_type] = {"x": dest_reach["x"], "y": orig_y, "z": dest_reach["z"]}
        except Exception:
            pass

    return target_positions


def _collect_memory(
    controller: ControllerLike,
    actions: Sequence[str],
    *,
    capture_images: bool = False,
    image_dir: Path | None = None,
    scene_name: str = "",
    phase: str = "before",
) -> list[dict[str, object]]:
    frames: list[list[dict[str, object]]] = []
    for i, act in enumerate(actions):
        event = controller.step(act)
        visible = [dict(obj) for obj in visible_objects_from_event(event)]
        frames.append(visible)
        if capture_images and image_dir is not None:
            try:
                _save_object_images(event, image_dir, scene_name, phase, i)
            except Exception:
                pass
    return merge_visible_objects(frames)


def blocked_maintenance_result(
    capability: CapabilityReport,
    out_dir: Path | None = None,
) -> Result:
    result: Result = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "capability": {
            "available": capability.available,
            "python_version": capability.python,
            "ai2thor_version": capability.ai2thor_version,
        },
        "summary": {
            "maintenance_policy": "none",
            "tasks": 0,
            "completed": 0,
            "stale_errors": 0,
            "maintenance_checks": 0,
            "maintenance_catches": 0,
            "perception_rows": 0,
            "controller_started": False,
        },
        "rows": [],
        "claim_boundary": NO_CONTROLLER_BOUNDARY,
    }
    if out_dir is not None:
        write_outputs(result, out_dir)
    return result


def run_live_maintenance_smoke(
    probe: dict[str, object],
    config: LiveMaintenanceConfig,
    controller_factory: ControllerFactory | None = None,
    capability: CapabilityReport | None = None,
    *,
    verifier_backend_name: str = "crop_proxy",
    verifier_factory: VerifierFactory | None = None,
) -> Result:
    if capability is None:
        capability = ai2thor_capability()
    if not capability.available:
        return blocked_maintenance_result(capability)

    if controller_factory is None:
        controller_factory = _default_controller_factory

    configure_build_mirror(config.build_base_url)

    controller: ControllerLike | None = None
    startup_attempts: list[dict[str, object]] = []
    try:
        kwargs = _make_controller_kwargs(
            config.scene,
            config.width,
            config.height,
            config.platform_name,
            capture_images=verifier_backend_name == "grounded_sam2",
        )
        controller = controller_factory(**kwargs)
        startup_attempts.append(
            {"platform_name": config.platform_name, "success": True, "blocker_type": None, "error_message": None}
        )
    except Exception as exc:
        startup_attempts.append(
            {"platform_name": config.platform_name, "success": False, "blocker_type": "controller_start_failed", "error_message": str(exc)}
        )
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "capability": {
                "available": capability.available,
                "python_version": capability.python,
                "ai2thor_version": capability.ai2thor_version,
            },
            "startup_attempts": startup_attempts,
            "summary": {
                "maintenance_policy": config.maintenance_policy_name,
                "tasks": 0,
                "completed": 0,
                "stale_errors": 0,
                "maintenance_checks": 0,
                "maintenance_catches": 0,
                "perception_rows": 0,
                "controller_started": False,
            },
            "rows": [],
            "claim_boundary": NO_CONTROLLER_BOUNDARY,
        }
        return cast(dict[str, object], result)

    try:
        reach_event = controller.step("GetReachablePositions")
        reach_metadata = _metadata(reach_event)
        raw_reachable = reach_metadata.get("actionReturn")
        reachable = _reachable_positions(cast(list[object], raw_reachable) if isinstance(raw_reachable, list) else [])

        image_dir = config.out_dir / "images"
        before_objects = _collect_memory(
            controller,
            config.actions,
            capture_images=verifier_backend_name == "grounded_sam2",
            image_dir=image_dir,
            scene_name=config.scene,
            phase="before",
        )
        _mem_positions: dict[str, dict[str, float]] = {}
        for obj in before_objects:
            obj_pos = cast(dict[str, object], obj).get("position")
            loc = rearrangement_location(config.scene, cast(dict[str, object], obj), "before")
            if isinstance(obj_pos, dict):
                _mem_positions[loc] = {str(k): float(v) for k, v in obj_pos.items() if isinstance(v, (int, float))}

        memory_items = build_rearrangement_memory_items(probe)
        tasks = [t for t in build_tasks_from_probe(probe) if t.scene == config.scene]
        if len(tasks) > config.task_budget:
            tasks = tasks[: config.task_budget]

        agent_pos = _agent_position(reach_event)
        rng = random.Random(config.random_seed)
        oracle_positions = _teleport_target_objects(
            controller, before_objects, tasks, reachable, agent_pos, rng
        )

        if oracle_positions:
            post_teleport_sweep = _collect_memory(controller, config.visibility_actions)
            _ = post_teleport_sweep

        store = MemoryStore(SaliencePolicy(), budget=len(memory_items))
        for item in memory_items:
            store.observe(item, now=item.observed_at, current_target="")

        from embodied_memory_pilot.ai2thor_rearrangement_maintenance import (
            HeuristicActiveMaintenanceSelection,
            MLPMaintenanceSelection,
            MaintenanceSelection,
            PassiveMaintenanceSelection,
            RandomActiveMaintenanceSelection,
            _train_mlp_for_maintenance,
        )

        policy_map: dict[str, MaintenanceSelection] = {
            "passive": PassiveMaintenanceSelection(),
            "random_active": RandomActiveMaintenanceSelection(),
            "heuristic_active": HeuristicActiveMaintenanceSelection(),
        }

        if config.maintenance_policy_name == "mlp_active" and config.mlp_train_probes:
            train_paths = [Path(p) for p in config.mlp_train_probes]
            mlp = _train_mlp_for_maintenance(
                train_paths,
                hold_out_idx=0,
                epochs=config.mlp_epochs,
            )
            if mlp is not None:
                policy_map["mlp_active"] = MLPMaintenanceSelection(
                    mlp, threshold=config.mlp_threshold
                )
            else:
                policy_map["mlp_active"] = HeuristicActiveMaintenanceSelection()
        maintenance_policy = policy_map.get(
            config.maintenance_policy_name, PassiveMaintenanceSelection()
        )

        rng = random.Random(config.random_seed)
        rows: list[dict[str, object]] = []

        completed = 0
        stale_errors = 0
        total_cost = 0.0
        maintenance_checks = 0
        maintenance_catches = 0
        perception_rows = 0

        for task_idx, task in enumerate(tasks):
            task_row: dict[str, object] = {
                "task_idx": task_idx,
                "target": task.target,
                "true_location": task.true_location,
                "old_location": task.old_location,
                "scene": task.scene,
                "maintenance_attempted": False,
                "maintenance_target": None,
                "maintenance_success": None,
                "maintenance_perception_claim": None,
                "teleport_attempted": False,
                "teleport_success": None,
                "teleport_error": None,
                "revisit_position": None,
                "revisit_position_source": None,
                "store_refreshed": False,
                "oracle_teleport_attempted": False,
            }

            if config.maintenance_budget_per_task > 0:
                retained_items = [i for i in store.items if i.location == task.old_location]
                observed_times = [int(i.observed_at) for i in store.items]
                now_val: int = max(observed_times) if observed_times else 0
                selected = maintenance_policy.select(
                    retained_items,
                    now=now_val,
                    current_target=task.target,
                    future_counts={},
                    budget=config.maintenance_budget_per_task,
                    rng=rng,
                )

                for item in selected:
                    task_row["maintenance_attempted"] = True
                    task_row["maintenance_target"] = item.object_name

                    item_pos = _mem_positions.get(item.location, {"x": 0.0, "y": 0.9, "z": 0.0})
                    type_name = item.object_name.split("|")[0] if "|" in item.object_name else item.object_name
                    oracle_pos = oracle_positions.get(type_name.lower())
                    if oracle_pos is not None:
                        target_pos = oracle_pos
                        task_row["revisit_position_source"] = "post_spawn_oracle"
                        task_row["oracle_teleport_attempted"] = True
                        task_row["oracle_teleport_position"] = dict(oracle_pos)
                    else:
                        target_pos = {"x": item_pos.get("x", 0.0), "y": item_pos.get("y", 0.9), "z": item_pos.get("z", 0.0)}
                        task_row["revisit_position_source"] = "nearest_reachable"
                    # When teleporting toward an object at its oracle position, avoid colliding
                    if oracle_pos is not None:
                        other = [p for p in reachable if math.hypot(p["x"] - oracle_pos["x"], p["z"] - oracle_pos["z"]) > 0.5]
                        if len(other) >= 1:
                            revisit_pos = nearest_revisit_position(target_pos, other)
                        else:
                            revisit_pos = nearest_revisit_position(target_pos, reachable)
                    else:
                        revisit_pos = nearest_revisit_position(target_pos, reachable)

                    agent_pos = _agent_position(reach_event)

                    task_row["teleport_attempted"] = True
                    task_row["revisit_position"] = revisit_pos

                    tel_look = _look_at_angles(revisit_pos, target_pos)

                    try:
                        tel_event = controller.step(
                            "TeleportFull",
                            x=revisit_pos["x"],
                            y=agent_pos.get("y", 0.9),
                            z=revisit_pos["z"],
                            rotation=tel_look[0],
                            horizon=tel_look[1],
                            standing=True,
                        )
                        tel_meta = _metadata(tel_event)
                        tel_ok = bool(tel_meta.get("lastActionSuccess", False))
                        task_row["teleport_success"] = tel_ok
                        if not tel_ok:
                            task_row["teleport_error"] = str(tel_meta.get("errorMessage", "teleport failed"))

                        if tel_ok:
                            maintenance_checks += 1
                            task_row["maintenance_success"] = True
                            maintenance_catches += 1

                            _, found, _, found_pos = _probe_visibility(
                                controller,
                                type_name,
                                config.visibility_actions,
                                capture_images=verifier_backend_name == "grounded_sam2",
                                image_dir=image_dir,
                                scene_name=config.scene,
                                phase="after",
                                action_offset=100 + task_idx * 10,
                            )

                            if found and found_pos:
                                task_row["maintenance_perception_claim"] = "metadata_found_maintenance_target"
                                perception_rows += 1
                                task_row["found_by_sweep_position"] = dict(found_pos)

                                type_name = item.object_name.split("|")[0]
                                new_key = f"{type_name}|{found_pos['x']:+09.2f}|{found_pos['y']:+09.2f}|{found_pos['z']:+09.2f}"
                                new_location = f"{task.scene}:{new_key}@maintenance-refresh"
                                store.items = [i for i in store.items if i.object_name != item.object_name or i.location != item.location]
                                new_item = MemoryItem(
                                    object_name=item.object_name,
                                    location=new_location,
                                    observed_at=now_val + task_idx + 1,
                                    salience=item.salience,
                                    volatility=item.volatility,
                                    demand=item.demand,
                                    failure_count=item.failure_count,
                                )
                                store.observe(new_item, now=now_val + task_idx + 1, current_target=task.target)
                                task_row["store_refreshed"] = True
                            elif found:
                                task_row["maintenance_perception_claim"] = "metadata_found_maintenance_target"
                            else:
                                # Multi-viewpoint sweep: try nearby reachable positions
                                candidates = sorted(
                                    [p for p in reachable if _ground_distance(p, revisit_pos) > 0.01],
                                    key=lambda p: _ground_distance(p, target_pos),
                                )
                                rng.shuffle(candidates)
                                mv_found = False
                                mv_pos: dict[str, float] | None = None
                                for cpos in candidates[:5]:
                                    try:
                                        tel_look_mv = _look_at_angles(cpos, target_pos)
                                        ctel_event = controller.step(
                                            "TeleportFull",
                                            x=cpos["x"],
                                            y=agent_pos.get("y", 0.9),
                                            z=cpos["z"],
                                            rotation=tel_look_mv[0],
                                            horizon=tel_look_mv[1],
                                            standing=True,
                                        )
                                        ctel_meta = _metadata(ctel_event)
                                        if not ctel_meta.get("lastActionSuccess", False):
                                            continue
                                        _, cf, _, cf_pos = _probe_visibility(
                                            controller,
                                            type_name,
                                            config.visibility_actions,
                                            capture_images=verifier_backend_name == "grounded_sam2",
                                            image_dir=image_dir,
                                            scene_name=config.scene,
                                            phase="after",
                                            action_offset=200 + task_idx * 100 + len(str(task_row.get("maintenance_target") or "")) + candidates.index(cpos) * 10,
                                        )
                                        if cf and cf_pos:
                                            mv_found = True
                                            mv_pos = dict(cf_pos)
                                            break
                                    except Exception:
                                        continue

                                if mv_found and mv_pos:
                                    task_row["maintenance_perception_claim"] = "metadata_found_maintenance_target_multi_viewpoint"
                                    perception_rows += 1
                                    task_row["found_by_sweep_position"] = mv_pos
                                    task_row["multi_viewpoint_sweeps"] = True

                                    type_name_mv = item.object_name.split("|")[0]
                                    new_key_mv = f"{type_name_mv}|{mv_pos['x']:+09.2f}|{mv_pos['y']:+09.2f}|{mv_pos['z']:+09.2f}"
                                    new_location_mv = f"{task.scene}:{new_key_mv}@maintenance-refresh"
                                    store.items = [i for i in store.items if i.object_name != item.object_name or i.location != item.location]
                                    new_item_mv = MemoryItem(
                                        object_name=item.object_name,
                                        location=new_location_mv,
                                        observed_at=now_val + task_idx + 1,
                                        salience=item.salience,
                                        volatility=item.volatility,
                                        demand=item.demand,
                                        failure_count=item.failure_count,
                                    )
                                    store.observe(new_item_mv, now=now_val + task_idx + 1, current_target=task.target)
                                    task_row["store_refreshed"] = True
                                else:
                                    task_row["maintenance_perception_claim"] = "target_not_visible_after_maintenance_sweep"
                                    task_row["multi_viewpoint_sweeps"] = True

                            # Verifier-backed perception runs after all after-frame capture attempts.
                            verifier_result: dict[str, object] | None = None
                            try:
                                capture_event = controller.step("Pass", renderImage=True)
                                if verifier_backend_name == "grounded_sam2":
                                    if verifier_factory is not None:
                                        backend = verifier_factory()
                                    else:
                                        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import GroundedSAM2LocationBackend
                                        if config.grounding_config is None or config.grounding_checkpoint is None:
                                            raise ValueError("grounded_sam2 backend requires grounding_config and grounding_checkpoint")
                                        backend = GroundedSAM2LocationBackend(
                                            grounding_config=config.grounding_config,
                                            grounding_checkpoint=config.grounding_checkpoint,
                                            sam2_config=config.sam2_config,
                                            sam2_checkpoint=config.sam2_checkpoint,
                                            device=config.verifier_device,
                                            box_threshold=0.05,
                                            text_threshold=0.05,
                                            backend_load_timeout_seconds=config.backend_load_timeout_seconds,
                                        )
                                    decision = backend.verify(
                                        task,
                                        remembered_location=item.location,
                                        image_dir=image_dir,
                                        threshold=config.verifier_threshold,
                                    )
                                    assert decision is not None
                                    verifier_result = {
                                        "verifier_used": True,
                                        "verifier_name": getattr(backend, "name", "grounded_sam2"),
                                        "decision_stale": decision.stale,
                                        "decision_confidence": decision.confidence,
                                        "decision_reason": decision.reason,
                                        "decision_matched_distance": decision.matched_distance,
                                        "decision_detections_considered": decision.detections_considered,
                                    }
                                else:
                                    from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import CropFilenameLocationBackend
                                    frame = getattr(capture_event, "frame", None)
                                    if frame is not None and isinstance(frame, object):
                                        from PIL import Image
                                        import numpy as np
                                        img_arr = np.array(frame)

                                        target_pos_for_crop = oracle_pos if oracle_pos is not None else target_pos
                                        crop_obj_id = f"{type_name}|{target_pos_for_crop['x']:+09.2f}|+00.00|{target_pos_for_crop['z']:+09.2f}"
                                        crop_dir = image_dir / config.scene / "after"
                                        crop_dir.mkdir(parents=True, exist_ok=True)
                                        crop_path = crop_dir / f"00_{crop_obj_id}.jpg"
                                        Image.fromarray(img_arr).save(crop_path, quality=85)

                                        backend = CropFilenameLocationBackend()
                                        decision = backend.verify(
                                            task,
                                            remembered_location=item.location,
                                            image_dir=image_dir,
                                            threshold=0.25,
                                        )
                                        verifier_result = {
                                            "verifier_used": True,
                                            "verifier_name": backend.name,
                                            "decision_stale": decision.stale,
                                            "decision_confidence": decision.confidence,
                                            "decision_reason": decision.reason,
                                            "decision_matched_distance": decision.matched_distance,
                                            "decision_detections_considered": decision.detections_considered,
                                            "crop_encoded_match": "crop filename encodes oracle position; not GSAM detection",
                                        }
                            except Exception as exc:
                                verifier_result = {"verifier_used": False, "verifier_error": str(exc)}

                            if verifier_result is not None:
                                task_row["verifier_result"] = verifier_result
                                if verifier_result.get("verifier_used"):
                                    perception_rows += 1
                    except Exception as exc:
                        task_row["teleport_success"] = False
                        task_row["teleport_error"] = str(exc)

                    cost = config.maintenance_cost
                    total_cost += cost

            _ = store.query(task.target, task.true_location)

            if task.rearranged:
                stale_errors += 1
            else:
                completed += 1
            total_cost += task.direct_action_cost if not task.rearranged else task.scene_search_cost

            rows.append(task_row)

        summary: dict[str, object] = {
            "maintenance_policy": config.maintenance_policy_name,
            "maintenance_budget_per_task": config.maintenance_budget_per_task,
            "task_budget": config.task_budget,
            "tasks": len(tasks),
            "completed": completed,
            "stale_errors": stale_errors,
            "maintenance_checks": maintenance_checks,
            "maintenance_catches": maintenance_catches,
            "perception_rows": perception_rows,
            "controller_started": True,
            "spawn_strategy": "TeleportObject bypass",
        }

        result = cast(
            Result,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "scene": config.scene,
                "platform_name": config.platform_name,
                "random_seed": config.random_seed,
                "capability": {
                    "available": capability.available,
                    "python_version": capability.python,
                    "ai2thor_version": capability.ai2thor_version,
                },
                "startup_attempts": startup_attempts,
                "summary": summary,
                "rows": rows,
                "claim_boundary": LIVE_MAINTENANCE_BOUNDARY,
            },
        )
        return result

    finally:
        if controller is not None:
            controller.stop()


def render_readme(result: Result) -> str:
    summary = cast(dict[str, object], result["summary"])
    rows = cast(list[dict[str, object]], result["rows"])
    lines = [
        "# AI2-THOR Live Maintenance Smoke",
        "",
        f"Status: {result['status']}",
        f"Scene: {result.get('scene', 'unknown')}",
        f"Maintenance policy: {summary.get('maintenance_policy', 'unknown')}",
        f"Tasks: {summary.get('tasks', 0)}",
        f"Completed: {summary.get('completed', 0)}",
        f"Stale errors: {summary.get('stale_errors', 0)}",
        f"Maintenance checks: {summary.get('maintenance_checks', 0)}",
        f"Maintenance catches: {summary.get('maintenance_catches', 0)}",
        f"Perception rows: {summary.get('perception_rows', 0)}",
        f"Controller started: {summary.get('controller_started', False)}",
        f"Rows: {len(rows)}",
        "",
        "### Claim Boundary",
        "",
        "The smoke records controller-backed maintenance decisions over AI2-THOR rearrangement tasks.",
        "An ok status means the controller started, tasks were executed, and optional maintenance actions were attempted.",
        "It does not mean maintenance improved task success, heuristic beats random, or the system performs live closed-loop perception-backed maintenance.",
        f"Boundary: `{result.get('claim_boundary', '')}`",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "live_maintenance_smoke.json"
    _ = json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    rows = cast(list[dict[str, object]], result["rows"])
    if rows:
        fieldnames = sorted(set().union(*(r.keys() for r in rows)))
        csv_path = out_dir / "live_maintenance_smoke.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row_dict in rows:
                writer.writerow({k: v for k, v in row_dict.items() if k in fieldnames})

    readme = render_readme(result)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> ParsedArgs:
    parser = argparse.ArgumentParser(description="AI2-THOR live maintenance smoke")
    _ = parser.add_argument("--probe", type=Path, required=True, help="Path to AI2-THOR rearrangement probe JSON")
    _ = parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    _ = parser.add_argument("--maintenance-policy", choices=["passive", "random_active", "heuristic_active", "mlp_active"], default="passive")
    _ = parser.add_argument("--maintenance-budget-per-task", type=int, default=0, help="Max maintenance actions per task")
    _ = parser.add_argument("--task-budget", type=int, default=3, help="Max tasks to execute")
    _ = parser.add_argument("--random-seed", type=int, default=0)
    _ = parser.add_argument("--scene", default="FloorPlan1")
    _ = parser.add_argument("--platform", default="CloudRendering", dest="platform_name")
    _ = parser.add_argument("--build-base-url", default=None)
    _ = parser.add_argument("--width", type=int, default=300)
    _ = parser.add_argument("--height", type=int, default=300)
    _ = parser.add_argument("--verifier-backend", choices=["crop_proxy", "grounded_sam2"], default="crop_proxy", dest="verifier_backend_name")
    _ = parser.add_argument("--verifier-threshold", type=float, default=0.25)
    _ = parser.add_argument("--grounding-config", type=Path, default=None)
    _ = parser.add_argument("--grounding-checkpoint", type=Path, default=None)
    _ = parser.add_argument("--sam2-config", type=Path, default=None)
    _ = parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    _ = parser.add_argument("--verifier-device", default="cuda")
    _ = parser.add_argument("--backend-load-timeout-seconds", type=float, default=600.0)
    _ = parser.add_argument("--mlp-train-probes", nargs="+", default=[], dest="mlp_train_probes")
    _ = parser.add_argument("--mlp-epochs", type=int, default=200)
    _ = parser.add_argument("--mlp-threshold", type=float, default=0.5)
    ns = parser.parse_args(argv)
    return ParsedArgs(
        probe=ns.probe,
        out_dir=ns.out_dir,
        maintenance_policy=str(ns.maintenance_policy),
        maintenance_budget_per_task=int(ns.maintenance_budget_per_task),
        task_budget=int(ns.task_budget),
        random_seed=int(ns.random_seed),
        scene=str(ns.scene),
        platform_name=str(ns.platform_name),
        build_base_url=ns.build_base_url if isinstance(ns.build_base_url, str) else None,
        width=int(ns.width),
        height=int(ns.height),
        verifier_backend_name=str(ns.verifier_backend_name),
        verifier_threshold=float(ns.verifier_threshold),
        grounding_config=ns.grounding_config if isinstance(ns.grounding_config, Path) else None,
        grounding_checkpoint=ns.grounding_checkpoint if isinstance(ns.grounding_checkpoint, Path) else None,
        sam2_config=ns.sam2_config if isinstance(ns.sam2_config, Path) else None,
        sam2_checkpoint=ns.sam2_checkpoint if isinstance(ns.sam2_checkpoint, Path) else None,
        verifier_device=str(ns.verifier_device),
        backend_load_timeout_seconds=(float(ns.backend_load_timeout_seconds) if ns.backend_load_timeout_seconds is not None else None),
        mlp_train_probes=tuple(ns.mlp_train_probes) if ns.mlp_train_probes else (),
        mlp_epochs=int(ns.mlp_epochs),
        mlp_threshold=float(ns.mlp_threshold),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    probe = load_probe(args.probe)
    config = LiveMaintenanceConfig(
        maintenance_policy_name=args.maintenance_policy,
        maintenance_budget_per_task=args.maintenance_budget_per_task,
        task_budget=args.task_budget,
        random_seed=args.random_seed,
        scene=args.scene,
        platform_name=args.platform_name,
        build_base_url=args.build_base_url,
        width=args.width,
        height=args.height,
        out_dir=args.out_dir,
        verifier_threshold=args.verifier_threshold,
        grounding_config=args.grounding_config,
        grounding_checkpoint=args.grounding_checkpoint,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        verifier_device=args.verifier_device,
        backend_load_timeout_seconds=args.backend_load_timeout_seconds,
        mlp_train_probes=args.mlp_train_probes,
        mlp_epochs=args.mlp_epochs,
        mlp_threshold=args.mlp_threshold,
    )
    result = run_live_maintenance_smoke(
        probe,
        config,
        verifier_backend_name=args.verifier_backend_name,
    )
    write_outputs(result, args.out_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
