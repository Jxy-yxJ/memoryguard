"""AI2-THOR-only screening for the pre-registered discriminative paired challenge.

Implements the frozen construction rule from
``results/0514_paired_hard_challenge_design_v1/paired_hard_challenge_design.json``:
a geometry-only screen over a pre-registered ``(scene, seed, target)`` universe,
followed by a deterministic freeze of the first ``K`` qualifying cases.

The screen uses simulator metadata for case construction only. Oracle metadata is
never a policy input, and no policy outcomes (GSAM, verification, task success)
are observed in this phase.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from embodied_memory_pilot.ai2thor_adapter import (
    DEFAULT_PROBE_ACTIONS,
    CapabilityReport,
    ai2thor_capability,
)
from embodied_memory_pilot.ai2thor_live_gsam_closed_loop import (
    ControllerFactory,
    ControllerLike,
    Json,
    _controller_factory,
    _find_after_object,
    _ground_distance,
    _make_controller_kwargs,
    _metadata,
    _object_id,
    _object_position,
    _object_type,
    _reachable_positions,
    _valid_position,
)
from embodied_memory_pilot.ai2thor_live_maintenance import _collect_memory, nearest_revisit_position


SCREENING_SCHEMA_VERSION = "0514_paired_hard_challenge_screen.v1"
CASE_LIST_SCHEMA_VERSION = "live_gsam_mixed_challenge_case_list.v2"
SCREENING_CLAIM_BOUNDARY = (
    "geometry_only_case_construction; uses AI2-THOR simulator metadata for screening and freezing only; "
    "no GSAM, no verification, no memory mutation, no task execution; "
    "not broad scale, not ObjectNav/SPL, not a manipulation benchmark, not passive-vs-active superiority"
)
DEFAULT_UNIVERSE: tuple[tuple[str, str], ...] = (
    ("FloorPlan1", "Apple"),
    ("FloorPlan1", "Book"),
    ("FloorPlan3", "Cup"),
    ("FloorPlan201", "Newspaper"),
    ("FloorPlan201", "Pencil"),
)
DEFAULT_SEEDS: tuple[int, ...] = (7, 11, 17, 23, 29, 37, 43, 53, 67, 73, 83, 97)
DEFAULT_D_PASSIVE_MIN_M = 2.0
DEFAULT_D_ACTIVE_MAX_M = 1.5
DEFAULT_K = 8


@dataclass(frozen=True)
class ScreeningConfig:
    universe: tuple[tuple[str, str], ...] = DEFAULT_UNIVERSE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    d_passive_min: float = DEFAULT_D_PASSIVE_MIN_M
    d_active_max: float = DEFAULT_D_ACTIVE_MAX_M
    k: int = DEFAULT_K
    width: int = 300
    height: int = 300
    platform_name: str = "CloudRendering"
    out_dir: Path = Path("results/0514_paired_hard_challenge_screen_v1")


def _candidate_id(scene: str, target: str, seed: int) -> str:
    return f"{scene}|{target}|{seed}"


def _pick_before_target(before_objects: Sequence[Mapping[str, object]], target_type: str) -> dict[str, object] | None:
    candidates = [
        dict(obj)
        for obj in before_objects
        if _object_type(obj) == target_type and _valid_position(obj) and bool(obj.get("pickupable"))
    ]
    candidates.sort(key=lambda obj: _object_id(obj))
    return candidates[0] if candidates else None


def _pick_after_target(
    before_target: Mapping[str, object],
    after_objects: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object] | None, str | None]:
    """Pair the before target with its after-spawn counterpart.

    Returns ``(after_target, pairing_method)`` where ``pairing_method`` is one of
    ``object_id``, ``unique_type``, ``ambiguous_type``, or ``missing``.
    """
    target_id = _object_id(before_target)
    for obj in after_objects:
        if target_id and _object_id(obj) == target_id:
            return dict(obj), "object_id"
    same_type = [dict(obj) for obj in after_objects if _object_type(obj) == _object_type(before_target)]
    if len(same_type) == 1:
        return same_type[0], "unique_type"
    if len(same_type) > 1:
        return None, "ambiguous_type"
    return None, "missing"


def screen_scene_seed(
    controller: ControllerLike,
    *,
    scene: str,
    seed: int,
    target_types: Sequence[str],
    config: ScreeningConfig,
) -> list[Json]:
    reach_event = controller.step("GetReachablePositions")
    reachable = _reachable_positions(cast(Sequence[object], _metadata(reach_event).get("actionReturn", [])))
    before_objects = _collect_memory(
        controller,
        DEFAULT_PROBE_ACTIONS,
        capture_images=False,
        scene_name=scene,
        phase="before",
    )
    spawn_event = controller.step(
        "InitialRandomSpawn",
        randomSeed=seed,
        forceVisible=True,
        numPlacementAttempts=5,
    )
    spawn_meta = _metadata(spawn_event)
    spawn_ok = bool(spawn_meta.get("lastActionSuccess", False))
    raw_after = spawn_meta.get("objects", [])
    after_objects = [dict(obj) for obj in raw_after if isinstance(obj, Mapping)] if isinstance(raw_after, Sequence) else []

    rows: list[Json] = []
    for target_type in target_types:
        row: Json = {
            "candidate_id": _candidate_id(scene, target_type, seed),
            "scene": scene,
            "seed": seed,
            "target": target_type,
            "spawn_ok": spawn_ok,
            "reachable_count": len(reachable),
            "before_object_id": None,
            "after_object_id": None,
            "after_pairing_method": None,
            "p_stale": None,
            "p_true": None,
            "g_passive": None,
            "g_active": None,
            "d_passive": None,
            "d_active": None,
            "qualifies": False,
            "status": "not_in_before_state",
        }
        before_target = _pick_before_target(before_objects, target_type)
        if before_target is None:
            rows.append(row)
            continue
        row["before_object_id"] = _object_id(before_target)
        after_target, pairing_method = _pick_after_target(before_target, after_objects)
        row["after_pairing_method"] = pairing_method
        if after_target is None:
            row["status"] = "ambiguous_after_pairing" if pairing_method == "ambiguous_type" else "not_spawned"
            rows.append(row)
            continue
        row["after_object_id"] = _object_id(after_target)
        if not bool(after_target.get("pickupable")):
            row["status"] = "not_pickupable"
            rows.append(row)
            continue

        p_stale = _object_position(before_target)
        p_true = _object_position(after_target)
        g_passive = nearest_revisit_position(p_stale, reachable) if reachable else dict(p_stale)
        g_active = nearest_revisit_position(p_true, reachable) if reachable else dict(p_true)
        d_passive = _ground_distance(g_passive, p_true)
        d_active = _ground_distance(g_active, p_true)
        qualifies = d_passive >= config.d_passive_min and d_active <= config.d_active_max
        row.update(
            {
                "p_stale": p_stale,
                "p_true": p_true,
                "g_passive": g_passive,
                "g_active": g_active,
                "d_passive": round(d_passive, 4),
                "d_active": round(d_active, 4),
                "qualifies": qualifies,
                "status": "qualifies" if qualifies else "below_threshold",
            }
        )
        rows.append(row)
    return rows


def freeze_case_list(rows: Sequence[Mapping[str, object]], k: int) -> list[Json]:
    qualifying = [dict(row) for row in rows if row.get("qualifies") is True]
    qualifying.sort(key=lambda row: (str(row.get("scene")), str(row.get("target")), int(cast(int, row.get("seed", 0)))))
    frozen: list[Json] = []
    for idx, row in enumerate(qualifying[:k]):
        frozen.append(
            {
                "row_idx": idx,
                "case_id": row.get("candidate_id"),
                "scene": row.get("scene"),
                "seed": row.get("seed"),
                "target": row.get("target"),
                "role": "paired_hard_challenge_frozen",
                "expected_risk": "geometry_discriminative_pre_registered",
                "d_passive": row.get("d_passive"),
                "d_active": row.get("d_active"),
            }
        )
    return frozen


def blocked_screening_result(capability: CapabilityReport, config: ScreeningConfig) -> Json:
    return {
        "schema": SCREENING_SCHEMA_VERSION,
        "status": "blocked",
        "capability": asdict(capability),
        "config": _config_dict(config),
        "summary": {"candidates": len(config.universe) * len(config.seeds), "frozen_cases": 0, "controller_started": False},
        "candidate_rows": [],
        "frozen_case_list": [],
        "claim_boundary": SCREENING_CLAIM_BOUNDARY,
    }


def _config_dict(config: ScreeningConfig) -> Json:
    payload = asdict(config)
    payload["out_dir"] = str(config.out_dir)
    return payload


def run_screening(
    config: ScreeningConfig,
    *,
    controller_factory: ControllerFactory | None = None,
    capability: CapabilityReport | None = None,
) -> Json:
    capability = capability or ai2thor_capability()
    if not capability.available:
        return blocked_screening_result(capability, config)

    make_controller = controller_factory or _controller_factory
    targets_by_scene: dict[str, list[str]] = {}
    for scene, target in config.universe:
        targets_by_scene.setdefault(scene, []).append(target)

    candidate_rows: list[Json] = []
    for seed in config.seeds:
        for scene, target_types in targets_by_scene.items():
            controller: ControllerLike | None = None
            try:
                kwargs = _make_controller_kwargs(scene, config.width, config.height, config.platform_name, capture_images=False)
                controller = make_controller(**kwargs)
                candidate_rows.extend(
                    screen_scene_seed(controller, scene=scene, seed=seed, target_types=tuple(target_types), config=config)
                )
            except Exception as exc:  # pragma: no cover - defensive boundary for simulator failures
                for target_type in target_types:
                    candidate_rows.append(
                        {
                            "candidate_id": _candidate_id(scene, target_type, seed),
                            "scene": scene,
                            "seed": seed,
                            "target": target_type,
                            "spawn_ok": False,
                            "reachable_count": 0,
                            "qualifies": False,
                            "status": f"controller_error:{type(exc).__name__}",
                        }
                    )
            finally:
                if controller is not None:
                    controller.stop()

    frozen = freeze_case_list(candidate_rows, config.k)
    status_counts: dict[str, int] = {}
    for row in candidate_rows:
        key = str(row.get("status"))
        status_counts[key] = status_counts.get(key, 0) + 1
    qualifying_count = status_counts.get("qualifies", 0)
    summary = {
        "candidates": len(candidate_rows),
        "spawned": sum(1 for row in candidate_rows if row.get("after_object_id") is not None),
        "qualifying": qualifying_count,
        "frozen_cases": len(frozen),
        "shortfall": len(frozen) < config.k,
        "status_counts": status_counts,
        "controller_started": True,
    }
    return {
        "schema": SCREENING_SCHEMA_VERSION,
        "status": "ok",
        "date": "2026-09-19",
        "capability": asdict(capability),
        "config": _config_dict(config),
        "summary": summary,
        "candidate_rows": candidate_rows,
        "frozen_case_list": frozen,
        "claim_boundary": SCREENING_CLAIM_BOUNDARY,
    }


def write_outputs(result: Json, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_hard_challenge_screen.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    rows = cast(Sequence[Mapping[str, object]], result.get("candidate_rows") or [])
    if rows:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(str(key))
        with (out_dir / "paired_hard_challenge_screen.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    frozen = cast(Sequence[Mapping[str, object]], result.get("frozen_case_list") or [])
    case_list = {
        "schema": CASE_LIST_SCHEMA_VERSION,
        "date": result.get("date"),
        "case_count": len(frozen),
        "cases": list(frozen),
        "screening_source": str(out_dir / "paired_hard_challenge_screen.json"),
        "claim_boundary": SCREENING_CLAIM_BOUNDARY,
    }
    (out_dir / "case_list_paired_hard_challenge_frozen_v1.json").write_text(
        json.dumps(case_list, indent=2, default=str), encoding="utf-8"
    )
    summary = cast(Mapping[str, object], result.get("summary") or {})
    readme = (
        "# Paired Hard Challenge Screening\n\n"
        f"Status: {result.get('status')}\n\n"
        f"Candidates: {summary.get('candidates')}\n\n"
        f"Qualifying: {summary.get('qualifying')}\n\n"
        f"Frozen cases: {summary.get('frozen_cases')}\n\n"
        f"Claim boundary: {result.get('claim_boundary')}\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> ScreeningConfig:
    parser = argparse.ArgumentParser(
        description="Screen the pre-registered discriminative paired passive-vs-active challenge universe"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--d-passive-min", type=float, default=DEFAULT_D_PASSIVE_MIN_M)
    parser.add_argument("--d-active-max", type=float, default=DEFAULT_D_ACTIVE_MAX_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--platform", default="CloudRendering", dest="platform_name")
    parser.add_argument("--out-dir", type=Path, default=Path("results/0514_paired_hard_challenge_screen_v1"))
    ns = parser.parse_args(argv)
    return ScreeningConfig(
        seeds=tuple(int(seed) for seed in ns.seeds),
        d_passive_min=float(ns.d_passive_min),
        d_active_max=float(ns.d_active_max),
        k=int(ns.k),
        width=int(ns.width),
        height=int(ns.height),
        platform_name=str(ns.platform_name),
        out_dir=ns.out_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    result = run_screening(config)
    write_outputs(result, config.out_dir)
    print(json.dumps(result.get("summary", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
