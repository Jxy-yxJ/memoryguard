"""Habitat-Sim geometry/image-hash stale-memory transfer smoke.

This runner is intentionally narrow. It checks whether Habitat-Sim can revisit
remembered agent poses in the Habitat test scene and reproduce RGB/depth image
hashes after moving away and returning. It does not use semantic annotations,
object labels, GSAM, ObjectNav, or task-success evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, SupportsFloat, SupportsInt, cast

SCHEMA_VERSION = "habitat_stale_memory_transfer.v1"
VERIFY_UPDATE_BOUNDARY = (
    "Habitat-Sim perception-backed verify/update smoke over pseudo visual landmarks; "
    "geometry/RGB-D hash consistency only; not object semantics; not ObjectNav; not manipulation; not task success."
)
DEFAULT_SCENE = Path("/home/jxy/coding/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")
DEFAULT_OUT_DIR = Path("results/habitat_stale_memory_transfer_smoke")
CLAIM_BOUNDARY = VERIFY_UPDATE_BOUNDARY
BLOCKED_BOUNDARY = "Habitat-Sim runtime unavailable or scene failed; no Habitat geometry/image-hash transfer evidence."
ROW_CLAIM_BOUNDARY = CLAIM_BOUNDARY
OBSERVATION_BOUNDARY = "rgb_depth_hash_only_no_semantics"
FALSE_BOUNDARY_FIELDS = {
    "semantic_annotations_used": False,
    "object_labels_used": False,
    "gsam_used": False,
    "objectnav_used": False,
    "task_success_evaluated": False,
}
JSON_NAME = "habitat_stale_memory_transfer.json"
CSV_NAME = "habitat_stale_memory_transfer.csv"

Json = dict[str, object]
Vector3 = tuple[float, float, float]
Rotation = tuple[float, float, float, float]


class PathfinderLike(Protocol):
    def find_path(self, path: object) -> bool: ...


class AgentLike(Protocol):
    def set_state(self, state: object) -> None: ...

    def get_state(self) -> object: ...


class SimulatorLike(Protocol):
    pathfinder: PathfinderLike

    def initialize_agent(self, agent_id: int) -> AgentLike: ...

    def get_sensor_observations(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class SimulatorConfigurationLike(Protocol):
    scene_id: str


class SensorTypeLike(Protocol):
    COLOR: object
    DEPTH: object


class SensorSpecLike(Protocol):
    uuid: str
    sensor_type: object
    resolution: list[int]
    position: list[float]


class AgentConfigurationLike(Protocol):
    sensor_specifications: list[SensorSpecLike]


class AgentStateLike(Protocol):
    position: list[float]
    rotation: list[float]


class ShortestPathLike(Protocol):
    requested_start: list[float]
    requested_end: list[float]
    geodesic_distance: float


class HabitatSimLike(Protocol):
    SimulatorConfiguration: type[SimulatorConfigurationLike]
    CameraSensorSpec: type[SensorSpecLike]
    SensorType: SensorTypeLike
    AgentConfiguration: type[AgentConfigurationLike]
    Configuration: Callable[[SimulatorConfigurationLike, list[AgentConfigurationLike]], object]
    Simulator: Callable[[object], SimulatorLike]
    AgentState: type[AgentStateLike]
    ShortestPath: type[ShortestPathLike]


@dataclass(frozen=True)
class HabitatSmokeConfig:
    scene: Path = DEFAULT_SCENE
    out_dir: Path = DEFAULT_OUT_DIR
    seed: int = 7
    num_rows: int = 5
    meters_forward: float = 1.0
    width: int = 160
    height: int = 120
    sensor_height: float = 1.5


def _habitat_sim_module() -> HabitatSimLike:
    return cast(HabitatSimLike, cast(object, importlib.import_module("habitat_sim")))


def observation_hash(observation: object) -> str:
    """Return a deterministic SHA256 hash for an observation array-like object."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to hash Habitat observations") from exc

    array = np.ascontiguousarray(observation)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("points must have the same dimensionality")
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(len(a))))


def capability_report(config: HabitatSmokeConfig, *, blocker: str | None = None) -> Json:
    habitat_sim_version: str | None = None
    habitat_version: str | None = None
    available = blocker is None
    try:
        habitat_sim_module = importlib.import_module("habitat_sim")
        raw_version = cast(object, getattr(habitat_sim_module, "__version__", None))
        habitat_sim_version = str(raw_version) if raw_version is not None else None
    except Exception as exc:
        available = False
        blocker = blocker or str(exc)
    try:
        habitat_module = importlib.import_module("habitat")
        raw_version = cast(object, getattr(habitat_module, "__version__", None))
        habitat_version = str(raw_version) if raw_version is not None else None
    except Exception:
        habitat_version = None
    return {
        "available": available,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "habitat_sim_version": habitat_sim_version,
        "habitat_version": habitat_version,
        "scene_path": str(config.scene),
        "blocker": blocker,
    }


def _rounded_vector(values: Sequence[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def _hash_observations(observations: Mapping[str, object]) -> tuple[str, str]:
    return observation_hash(observations["rgb"]), observation_hash(observations["depth"])


def _sensor_shape(observations: Mapping[str, object], sensor_name: str) -> list[int]:
    value = observations.get(sensor_name)
    shape = getattr(value, "shape", None)
    if not isinstance(shape, Sequence):
        return []
    return [int(cast(SupportsInt, dim)) for dim in shape]


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return round(sum(finite) / len(finite), 6)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(cast(SupportsFloat, value))


def _summary(scene: Path | str, seed: int, rows: Sequence[Mapping[str, object]], *, blocked: bool) -> Json:
    row_count = len(rows)
    return {
        "rows": row_count,
        "scene": str(scene),
        "seed": seed,
        "before_after_rows": sum(1 for row in rows if row.get("before_after_euclidean_distance") is not None),
        "revisit_rows": sum(1 for row in rows if row.get("before_revisit_euclidean_distance") is not None),
        "rgb_revisit_match_rows": sum(1 for row in rows if row.get("revisit_matches_before_rgb") is True),
        "depth_revisit_match_rows": sum(1 for row in rows if row.get("revisit_matches_before_depth") is True),
        "mean_euclidean_before_after": _mean([_optional_float(row.get("before_after_euclidean_distance")) for row in rows]),
        "mean_geodesic_before_after": _mean([_optional_float(row.get("before_after_geodesic_distance")) for row in rows]),
        "claim_boundary": CLAIM_BOUNDARY if not blocked else BLOCKED_BOUNDARY,
        **FALSE_BOUNDARY_FIELDS,
    }


def blocked_result(config: HabitatSmokeConfig, blocker_type: str, error_message: str) -> Json:
    capability = capability_report(config, blocker=f"{blocker_type}: {error_message}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "capability": capability,
        "claim_boundary": BLOCKED_BOUNDARY,
        "summary": _summary(config.scene, config.seed, [], blocked=True),
        "rows": [],
        "blocker": {"type": blocker_type, "message": error_message},
    }


def build_result(
    scene: Path | str,
    seed: int,
    rows: Sequence[Mapping[str, object]],
    *,
    capability: Mapping[str, object] | None = None,
    status: str = "ok",
) -> Json:
    row_list = [dict(row) for row in rows]
    config = HabitatSmokeConfig(scene=Path(scene), seed=seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "capability": dict(capability) if capability is not None else capability_report(config),
        "claim_boundary": CLAIM_BOUNDARY,
        "summary": _summary(scene, seed, row_list, blocked=False),
        "rows": row_list,
    }


def _make_simulator(config: HabitatSmokeConfig) -> SimulatorLike:
    habitat_sim = _habitat_sim_module()
    _ = importlib.import_module("habitat_sim.bindings")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(config.scene)

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [config.height, config.width]
    rgb_spec.position = [0.0, config.sensor_height, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [config.height, config.width]
    depth_spec.position = [0.0, config.sensor_height, 0.0]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def _agent_state(position: Vector3, rotation: Rotation) -> object:
    habitat_sim = _habitat_sim_module()
    state = habitat_sim.AgentState()
    state.position = list(position)
    state.rotation = list(rotation)
    return state


def _seed_runtime(config: HabitatSmokeConfig, sim: SimulatorLike) -> None:
    random.seed(config.seed)
    try:
        import numpy as np

        np.random.seed(config.seed)
    except ImportError:
        pass
    for target in (sim, getattr(sim, "pathfinder", None)):
        seed_fn = getattr(target, "seed", None)
        if callable(seed_fn):
            _ = seed_fn(config.seed)


def _shortest_path_distance(sim: SimulatorLike, start: Vector3, end: Vector3) -> tuple[float | None, str]:
    try:
        habitat_sim = _habitat_sim_module()
        path = habitat_sim.ShortestPath()
        path.requested_start = list(start)
        path.requested_end = list(end)
        found_path = bool(sim.pathfinder.find_path(path))
        if not found_path:
            return None, "no_path"
        return round(float(path.geodesic_distance), 6), "ok"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}"


def _as_vector3(raw: object) -> Vector3 | None:
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) >= 3:
        return (
            float(cast(SupportsFloat, raw[0])),
            float(cast(SupportsFloat, raw[1])),
            float(cast(SupportsFloat, raw[2])),
        )
    if all(hasattr(raw, attr) for attr in ("x", "y", "z")):
        return (
            float(cast(SupportsFloat, getattr(raw, "x"))),
            float(cast(SupportsFloat, getattr(raw, "y"))),
            float(cast(SupportsFloat, getattr(raw, "z"))),
        )
    return None


def _as_rotation(raw: object, fallback: Rotation) -> Rotation:
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) >= 4:
        return (
            float(cast(SupportsFloat, raw[0])),
            float(cast(SupportsFloat, raw[1])),
            float(cast(SupportsFloat, raw[2])),
            float(cast(SupportsFloat, raw[3])),
        )
    if all(hasattr(raw, attr) for attr in ("x", "y", "z", "w")):
        return (
            float(cast(SupportsFloat, getattr(raw, "x"))),
            float(cast(SupportsFloat, getattr(raw, "y"))),
            float(cast(SupportsFloat, getattr(raw, "z"))),
            float(cast(SupportsFloat, getattr(raw, "w"))),
        )
    return fallback


def _fallback_before_position(row_idx: int, config: HabitatSmokeConfig) -> Vector3:
    return (float(row_idx) * config.meters_forward, 0.0, 0.0)


def _before_position(sim: SimulatorLike, row_idx: int, config: HabitatSmokeConfig) -> Vector3:
    random_point = getattr(sim.pathfinder, "get_random_navigable_point", None)
    if callable(random_point):
        candidate = _as_vector3(random_point())
        if candidate is not None:
            return candidate
    return _fallback_before_position(row_idx, config)


def _after_position(sim: SimulatorLike, row_idx: int, before_position: Vector3, config: HabitatSmokeConfig) -> Vector3:
    random_point = getattr(sim.pathfinder, "get_random_navigable_point", None)
    if callable(random_point):
        for _candidate_idx in range(20):
            candidate = _as_vector3(random_point())
            if candidate is not None and euclidean_distance(before_position, candidate) >= config.meters_forward:
                return candidate
    angle = (config.seed + row_idx) * 0.6180339887498949
    dx = math.cos(angle) * config.meters_forward
    dz = math.sin(angle) * config.meters_forward
    return (before_position[0] + dx, before_position[1], before_position[2] + dz)


def _rotation(row_idx: int, config: HabitatSmokeConfig) -> Rotation:
    _ = config
    return (0.0, math.sin(row_idx * 0.05), 0.0, math.cos(row_idx * 0.05))


def _capture_hashes(agent: AgentLike, sim: SimulatorLike, position: Vector3, rotation: Rotation) -> tuple[Mapping[str, object], str, str, Vector3, Rotation]:
    state = _agent_state(position, rotation)
    agent.set_state(state)
    actual_state = agent.get_state()
    actual_position = _as_vector3(getattr(actual_state, "position", position)) or position
    actual_rotation = _as_rotation(getattr(actual_state, "rotation", rotation), rotation)
    observations = sim.get_sensor_observations()
    rgb_hash, depth_hash = _hash_observations(observations)
    return observations, rgb_hash, depth_hash, actual_position, actual_rotation


def _row(row_idx: int, scene: Path, seed: int, before_position: Vector3, after_position: Vector3, before_rotation: Rotation, after_rotation: Rotation, revisit_rotation: Rotation, observations: Mapping[str, object], hashes: Mapping[str, str], distances: Mapping[str, float | None], geodesics: Mapping[str, float | None], geodesic_statuses: Mapping[str, str]) -> Json:
    return {
        "row_idx": row_idx,
        "scene": str(scene),
        "seed": seed,
        "before_position": _rounded_vector(before_position),
        "after_position": _rounded_vector(after_position),
        "revisit_position": _rounded_vector(before_position),
        "before_rotation": _rounded_vector(before_rotation),
        "after_rotation": _rounded_vector(after_rotation),
        "revisit_rotation": _rounded_vector(revisit_rotation),
        "revisit_source": "set_agent_state_to_remembered_before_pose",
        "claim_boundary": CLAIM_BOUNDARY,
        "row_claim_boundary": ROW_CLAIM_BOUNDARY,
        "observation_boundary": OBSERVATION_BOUNDARY,
        "before_rgb_sha256": hashes["before_rgb"],
        "before_depth_sha256": hashes["before_depth"],
        "after_rgb_sha256": hashes["after_rgb"],
        "after_depth_sha256": hashes["after_depth"],
        "revisit_rgb_sha256": hashes["revisit_rgb"],
        "revisit_depth_sha256": hashes["revisit_depth"],
        "revisit_matches_before_rgb": hashes["revisit_rgb"] == hashes["before_rgb"],
        "revisit_matches_before_depth": hashes["revisit_depth"] == hashes["before_depth"],
        "after_differs_from_before_rgb": hashes["after_rgb"] != hashes["before_rgb"],
        "after_differs_from_before_depth": hashes["after_depth"] != hashes["before_depth"],
        "before_after_euclidean_distance": distances["before_after"],
        "before_revisit_euclidean_distance": distances["before_revisit"],
        "after_revisit_euclidean_distance": distances["after_revisit"],
        "before_after_geodesic_distance": geodesics["before_after"],
        "before_after_geodesic_status": geodesic_statuses["before_after"],
        "before_revisit_geodesic_distance": geodesics["before_revisit"],
        "before_revisit_geodesic_status": geodesic_statuses["before_revisit"],
        "after_revisit_geodesic_distance": geodesics["after_revisit"],
        "after_revisit_geodesic_status": geodesic_statuses["after_revisit"],
        "rgb_shape": _sensor_shape(observations, "rgb"),
        "depth_shape": _sensor_shape(observations, "depth"),
        **FALSE_BOUNDARY_FIELDS,
    }


def apply_perception_verify_update_trace(row: Mapping[str, object]) -> Json:
    """Add a platform-neutral verify/update trace to a Habitat geometry row."""
    traced = dict(row)
    before_position = cast(object, traced.get("before_position"))
    after_position = cast(object, traced.get("after_position"))
    revisit_position = cast(object, traced.get("revisit_position"))
    rgb_matches = traced.get("revisit_matches_before_rgb") is True
    depth_matches = traced.get("revisit_matches_before_depth") is True
    after_rgb_differs = traced.get("after_differs_from_before_rgb") is True
    after_depth_differs = traced.get("after_differs_from_before_depth") is True
    stale = not (rgb_matches and depth_matches)
    update_action = "update_memory" if stale else "keep_memory"
    memory_before = {
        "position": before_position,
        "rgb_sha256": traced.get("before_rgb_sha256"),
        "depth_sha256": traced.get("before_depth_sha256"),
    }
    memory_after_update = {
        "position": revisit_position if stale else before_position,
        "rgb_sha256": traced.get("revisit_rgb_sha256") if stale else traced.get("before_rgb_sha256"),
        "depth_sha256": traced.get("revisit_depth_sha256") if stale else traced.get("before_depth_sha256"),
    }
    traced.update(
        {
            "platform": "Habitat-Sim",
            "memory_anchor_id": f"habitat_pseudo_landmark_{traced.get('row_idx', 0)}",
            "memory_anchor_type": "pseudo_visual_landmark",
            "object_semantics_available": False,
            "memory_before": memory_before,
            "environment_after": {
                "position": after_position,
                "rgb_sha256": traced.get("after_rgb_sha256"),
                "depth_sha256": traced.get("after_depth_sha256"),
            },
            "revisit_observation": {
                "position": revisit_position,
                "rgb_sha256": traced.get("revisit_rgb_sha256"),
                "depth_sha256": traced.get("revisit_depth_sha256"),
            },
            "perception_evidence": {
                "evidence_type": "rgb_depth_hash_consistency_proxy",
                "rgb_revisit_matches_before": rgb_matches,
                "depth_revisit_matches_before": depth_matches,
                "after_differs_from_before_rgb": after_rgb_differs,
                "after_differs_from_before_depth": after_depth_differs,
                "before_after_euclidean_distance": traced.get("before_after_euclidean_distance"),
                "before_after_geodesic_distance": traced.get("before_after_geodesic_distance"),
            },
            "verifier_name": "rgb_depth_hash_consistency_proxy",
            "verifier_decision": "stale" if stale else "fresh",
            "verification_confidence": 1.0 if rgb_matches and depth_matches else 0.5,
            "update_action": update_action,
            "memory_after_update": memory_after_update,
            "path_or_pose_fields": {
                "before_position": before_position,
                "after_position": after_position,
                "revisit_position": revisit_position,
                "revisit_source": traced.get("revisit_source"),
                "before_after_euclidean_distance": traced.get("before_after_euclidean_distance"),
                "before_after_geodesic_distance": traced.get("before_after_geodesic_distance"),
            },
            "shortcut_setup_used": True,
            "task_action": "not_evaluated",
            "task_success": None,
            "failure_reason": None,
            "claim_boundary": VERIFY_UPDATE_BOUNDARY,
            "row_claim_boundary": VERIFY_UPDATE_BOUNDARY,
            "semantic_annotations_used": False,
            "object_labels_used": False,
            "gsam_used": False,
            "objectnav_used": False,
            "task_success_evaluated": False,
        }
    )
    return traced


def run_habitat_stale_memory_transfer(config: HabitatSmokeConfig) -> Json:
    try:
        _ = _habitat_sim_module()
    except Exception as exc:
        return blocked_result(config, "habitat_sim_import_error", str(exc))

    sim: SimulatorLike | None = None
    try:
        sim = _make_simulator(config)
        _seed_runtime(config, sim)
        capability = capability_report(config)
        agent = sim.initialize_agent(0)
        rows: list[Json] = []
        for row_idx in range(config.num_rows):
            before_position = _before_position(sim, row_idx, config)
            after_position = _after_position(sim, row_idx, before_position, config)
            before_rotation = _rotation(row_idx, config)
            after_rotation = before_rotation
            revisit_rotation = before_rotation

            before_observations, before_rgb, before_depth, actual_before_position, actual_before_rotation = _capture_hashes(
                agent, sim, before_position, before_rotation
            )
            _, after_rgb, after_depth, actual_after_position, actual_after_rotation = _capture_hashes(agent, sim, after_position, after_rotation)
            _, revisit_rgb, revisit_depth, actual_revisit_position, actual_revisit_rotation = _capture_hashes(
                agent, sim, actual_before_position, revisit_rotation
            )

            before_after_geo, before_after_status = _shortest_path_distance(sim, actual_before_position, actual_after_position)
            before_revisit_geo, before_revisit_status = _shortest_path_distance(sim, actual_before_position, actual_revisit_position)
            after_revisit_geo, after_revisit_status = _shortest_path_distance(sim, actual_after_position, actual_revisit_position)
            distances = {
                "before_after": round(euclidean_distance(actual_before_position, actual_after_position), 6),
                "before_revisit": round(euclidean_distance(actual_before_position, actual_revisit_position), 6),
                "after_revisit": round(euclidean_distance(actual_after_position, actual_revisit_position), 6),
            }
            geodesics = {
                "before_after": before_after_geo,
                "before_revisit": before_revisit_geo,
                "after_revisit": after_revisit_geo,
            }
            geodesic_statuses = {
                "before_after": before_after_status,
                "before_revisit": before_revisit_status,
                "after_revisit": after_revisit_status,
            }
            rows.append(
                _row(
                    row_idx,
                    config.scene,
                    config.seed,
                    actual_before_position,
                    actual_after_position,
                    actual_before_rotation,
                    actual_after_rotation,
                    actual_revisit_rotation,
                    before_observations,
                    {
                        "before_rgb": before_rgb,
                        "before_depth": before_depth,
                        "after_rgb": after_rgb,
                        "after_depth": after_depth,
                        "revisit_rgb": revisit_rgb,
                        "revisit_depth": revisit_depth,
                    },
                    distances,
                    geodesics,
                    geodesic_statuses,
                )
            )
        traced_rows = [apply_perception_verify_update_trace(row) for row in rows]
        return build_result(config.scene, config.seed, traced_rows, capability=capability)
    except Exception as exc:
        return blocked_result(config, "habitat_sim_runtime_error", str(exc))
    finally:
        if sim is not None:
            close = getattr(sim, "close", None)
            if callable(close):
                _ = close()


def render_readme(result: Mapping[str, object]) -> str:
    summary = cast(Mapping[str, object], result.get("summary", {}))
    capability = cast(Mapping[str, object], result.get("capability", {}))
    return "\n".join(
        [
            "# Habitat-Sim Stale-Memory Transfer Smoke",
            "",
            f"Status: `{result.get('status')}`",
            f"Scene: `{summary.get('scene') or capability.get('scene_path')}`",
            f"Seed: `{summary.get('seed')}`",
            f"Rows: `{summary.get('rows', 0)}`",
            f"Habitat-Sim available: `{capability.get('available', False)}`",
            f"Habitat-Sim version: `{capability.get('habitat_sim_version')}`",
            "",
            "This artifact is limited to Habitat-Sim perception-backed verify/update smoke over pseudo visual landmarks.",
            "Habitat test scenes lack semantic annotations, so verification uses RGB-D hash consistency only and does not use object labels, GSAM, ObjectNav, manipulation, or task-success evaluation.",
            f"Claim boundary: `{result.get('claim_boundary')}`",
            "",
        ]
    )


def write_outputs(result: Mapping[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = (out_dir / JSON_NAME).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    raw_rows = result.get("rows", [])
    rows = cast(Sequence[Mapping[str, object]], raw_rows)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with (out_dir / CSV_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _ = (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> HabitatSmokeConfig:
    parser = argparse.ArgumentParser(description="Run a minimal Habitat-Sim stale-memory transfer smoke.")
    _ = parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    _ = parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    _ = parser.add_argument("--seed", type=int, default=7)
    _ = parser.add_argument("--num-rows", type=int, default=5)
    _ = parser.add_argument("--meters-forward", type=float, default=1.0)
    _ = parser.add_argument("--width", type=int, default=160)
    _ = parser.add_argument("--height", type=int, default=120)
    _ = parser.add_argument("--sensor-height", type=float, default=1.5)
    args = parser.parse_args(argv)
    namespace = cast(Mapping[str, object], vars(args))
    return HabitatSmokeConfig(
        scene=cast(Path, namespace["scene"]),
        out_dir=cast(Path, namespace["out_dir"]),
        seed=int(cast(SupportsInt, namespace["seed"])),
        num_rows=int(cast(SupportsInt, namespace["num_rows"])),
        meters_forward=float(cast(SupportsFloat, namespace["meters_forward"])),
        width=int(cast(SupportsInt, namespace["width"])),
        height=int(cast(SupportsInt, namespace["height"])),
        sensor_height=float(cast(SupportsFloat, namespace["sensor_height"])),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    result = run_habitat_stale_memory_transfer(config)
    write_outputs(result, config.out_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
