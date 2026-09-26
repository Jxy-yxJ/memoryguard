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

from embodied_memory_pilot.ai2thor_adapter import (
    DEFAULT_OBJECTS,
    DEFAULT_PROBE_ACTIONS,
    RICH_BEFORE_PROBE_ACTIONS,
    CapabilityReport,
    ai2thor_capability,
    configure_build_mirror,
)
from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import LocationDecision
from embodied_memory_pilot.ai2thor_navigation_smoke import _agent_position as _nav_agent_position
from embodied_memory_pilot.ai2thor_navigation_smoke import _navigation_step as _nav_navigation_step
from embodied_memory_pilot.ai2thor_live_maintenance import (
    _collect_memory,
    _look_at_angles,
    _make_controller_kwargs,
    _metadata,
    _object_position,
    _probe_visibility,
    _reachable_positions,
    nearest_revisit_position,
)
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import RearrangementTask, rearrangement_location


SCHEMA_VERSION = "ai2thor_live_gsam_closed_loop.v1"
CLAIM_BOUNDARY = (
    "controller-backed closed-loop GSAM stale-memory detection; InitialRandomSpawn occurs live; "
    "TeleportFull is a shortcut to remembered before-location and nearby positions for limited nav search; no TeleportObject; "
    "oracle metadata is offline labels only; nav search is limited position-hopping, not step-by-step navigation; "
    "not full navigation, manipulation, or task success"
)
MEMORY_UPDATE_BOUNDARY = (
    "controller-backed evidence-guided memory refresh from live revisit detector-verifier evidence; "
    "mutated in memory-store snapshot only; not persistent storage; not recovery search; not task success"
)
TASK_BRIDGE_BOUNDARY = (
    "task_bridge_teleport_shortcut_after_verified_memory_update; executes one downstream PickupObject/OpenObject "
    "attempt after detector-backed memory refresh; not full navigation, not a manipulation benchmark, "
    "not persistent memory writeback, not broad task success"
)
STEPWISE_TASK_BRIDGE_BOUNDARY = (
    "stepwise_task_bridge_after_verified_memory_update; uses bounded stepwise navigation before one downstream "
    "PickupObject/OpenObject attempt after detector-backed memory refresh; not ObjectNav/SPL, not full navigation, "
    "not a manipulation benchmark, not persistent memory writeback, not broad task success"
)
LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY = (
    "live_passive_task_bridge_from_stale_memory_old_position; no_verifier; no_memory_mutation; "
    "stepwise navigation only (no TeleportFull, no TeleportObject); "
    "no decision_stale check; no persistent writeback; "
    "passive downstream task evaluated from stale memory without detector evidence; not a manipulation benchmark; "
    "not ObjectNav/SPL; not broad passive-vs-active superiority claim"
)
NO_CONTROLLER_BOUNDARY = "AI2-THOR controller unavailable; no closed-loop GSAM evidence"
Json = dict[str, Any]


class ControllerLike(Protocol):
    def step(self, action: str, **kwargs: object) -> object: ...

    def stop(self) -> None: ...


class VerifierBackendLike(Protocol):
    name: str

    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float) -> LocationDecision: ...


ControllerFactory = Callable[..., ControllerLike]
VerifierFactory = Callable[[], VerifierBackendLike]


@dataclass(frozen=True)
class LiveGSAMClosedLoopConfig:
    scenes: tuple[str, ...] = ("FloorPlan1", "FloorPlan3", "FloorPlan201")
    seeds: tuple[int, ...] = (7, 11, 29, 31, 37)
    max_rows: int = 20
    targets_per_scene_seed: int = 2
    target_objects: tuple[str, ...] = DEFAULT_OBJECTS
    width: int = 300
    height: int = 300
    platform_name: str = "CloudRendering"
    build_base_url: str | None = None
    out_dir: Path = Path("results/ai2thor_live_gsam_closed_loop")
    case_list: Path | None = None
    verifier_threshold: float = 0.05
    verification_cost: float = 0.05
    stale_action_cost: float = 1.0
    active_verification_threshold: float = 0.0
    verification_budget: int | None = None
    selection_policy_name: str = "expected_verification_value_budgeted_v1"
    verifier_backend_name: str = "grounded_sam2"
    grounding_config: Path | None = None
    grounding_checkpoint: Path | None = None
    sam2_config: Path | None = None
    sam2_checkpoint: Path | None = None
    verifier_device: str = "cuda"
    backend_load_timeout_seconds: float | None = 600.0
    time_interval: float = 1.0
    nav_search_steps: int = 0
    ev_variant: str = "current"
    revisit_mode: str = "teleportfull"
    execute_task_bridge: bool = False
    honest_interaction: bool = False
    before_probe_actions: tuple[str, ...] | None = None
    max_alternate_poses: int = 0


@dataclass(frozen=True)
class StaleUncertaintyInputs:
    detector_confidence: float | None
    depth_consistency: float | None
    object_mobility_prior: float
    time_interval: float
    task_importance: float
    historical_failure: float
    scene_change_score: float
    target_visible_after_revisit: bool
    visibility_error: str | None
    distance_from_remembered: float | None
    teleport_success: bool
    same_type_density_score: float = 0.0
    same_type_nearby_count: int = 0
    same_type_scene_total: int = 0
    visual_instance_confusion_prior: float = 0.0


@dataclass(frozen=True)
class StaleUncertaintyEstimate:
    probability: float
    reason: str
    inputs: StaleUncertaintyInputs


@dataclass(frozen=True)
class TargetedCaseSpec:
    scene: str
    seed: int
    target: str
    source_row_idx: int | None


TARGETED_CASE_REASON = "prior_recorded_false_negative_slice"


def _targeted_case_key(scene: str, seed: int, target: str) -> tuple[str, int, str]:
    return scene, seed, target


def _case_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _load_targeted_case_specs(path: Path) -> list[TargetedCaseSpec]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    raw_cases: object
    if isinstance(payload, Mapping):
        payload_mapping = cast(Mapping[str, object], payload)
        raw_cases = payload_mapping.get("cases") or payload_mapping.get("rows") or []
    else:
        raw_cases = payload
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise ValueError("case-list must contain a sequence of case rows")

    specs: list[TargetedCaseSpec] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue
        case = cast(Mapping[str, object], raw_case)
        scene = case.get("scene")
        seed = _case_int(case.get("seed"))
        target = case.get("target") or case.get("object_type") or case.get("objectType")
        if not isinstance(scene, str) or seed is None or not isinstance(target, str):
            continue
        specs.append(
            TargetedCaseSpec(
                scene=scene,
                seed=seed,
                target=target,
                source_row_idx=_case_int(case.get("row_idx")),
            )
        )
    return specs



def _valid_position(obj: Mapping[str, object]) -> bool:
    pos = obj.get("position")
    return isinstance(pos, Mapping) and all(k in pos for k in ("x", "y", "z"))


def _object_id(obj: Mapping[str, object]) -> str:
    return str(obj.get("object_id") or obj.get("objectId") or obj.get("name") or obj.get("object_type") or "")


def _object_type(obj: Mapping[str, object]) -> str:
    return str(obj.get("object_type") or obj.get("objectType") or "Unknown")


def _visual_instance_confusion_prior(object_type: str) -> float:
    priors = {
        "Book": 0.6,
        "Newspaper": 0.75,
        "Pencil": 0.45,
    }
    return priors.get(object_type, 0.0)


def _ground_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["z"]) - float(b["z"]))


def _select_memory_targets(
    before_objects: Sequence[dict[str, object]],
    limit: int,
    *,
    preferred_objects: Sequence[str] = DEFAULT_OBJECTS,
) -> list[dict[str, object]]:
    preferred = {name: idx for idx, name in enumerate(preferred_objects)}
    candidates = [dict(o) for o in before_objects if _valid_position(o) and _object_type(o) != "Unknown"]
    candidates.sort(
        key=lambda o: (
            preferred.get(_object_type(o), len(preferred)),
            0 if bool(o.get("pickupable")) else 1,
            _object_id(o),
        )
    )
    seen: set[str] = set()
    selected: list[dict[str, object]] = []
    for obj in candidates:
        key = _object_id(obj)
        if key in seen:
            continue
        seen.add(key)
        selected.append(obj)
        if len(selected) >= limit:
            break
    return selected


def _find_after_object(target: Mapping[str, object], objects: Sequence[dict[str, object]]) -> dict[str, object] | None:
    target_id = _object_id(target)
    target_type = _object_type(target)
    for obj in objects:
        if target_id and _object_id(obj) == target_id:
            return dict(obj)
    for obj in objects:
        if target_type and _object_type(obj) == target_type:
            return dict(obj)
    return None


def _object_type_from_id(object_id: str) -> str:
    return object_id.split("|", 1)[0] if object_id else "Unknown"


def _resolve_current_object_id(
    previous_object_id: str,
    objects: Sequence[Mapping[str, object]],
    target_position: Mapping[str, float] | None,
) -> str:
    if not previous_object_id:
        return previous_object_id
    for obj in objects:
        if _object_id(obj) == previous_object_id:
            return previous_object_id
    target_type = _object_type_from_id(previous_object_id)
    candidates = [obj for obj in objects if _valid_position(obj) and _object_type(obj) == target_type and _object_id(obj)]
    if not candidates:
        return previous_object_id
    if target_position is None:
        return _object_id(candidates[0])
    closest = min(candidates, key=lambda obj: _ground_distance(target_position, cast(Mapping[str, float], obj["position"])))
    return _object_id(closest)


def _oracle_label(before_obj: Mapping[str, object], after_obj: Mapping[str, object] | None) -> tuple[bool | None, float | None, str]:
    if after_obj is None or not _valid_position(after_obj):
        return None, None, "target_not_visible_to_metadata_sweep"
    before_pos = _object_position(dict(before_obj))
    after_pos = _object_position(dict(after_obj))
    distance = _ground_distance(before_pos, after_pos)
    return distance > 0.05, round(distance, 4), "metadata_position_delta_offline_only"


def _object_mobility_prior(obj: Mapping[str, object]) -> float:
    pickupable = bool(obj.get("pickupable"))
    moveable = bool(obj.get("moveable") or obj.get("movable") or pickupable)
    return 1.0 if moveable else 0.0


def _build_uncertainty_inputs(
    before_obj: Mapping[str, object],
    remembered_pos: Mapping[str, float],
    *,
    target_visible_after_revisit: bool,
    visibility_error: str | None,
    found_pos: Mapping[str, float] | None,
    teleport_success: bool,
    time_interval: float = 1.0,
    gsam_detection_history: dict[str, list[bool]] | None = None,
    gsam_detection_counts: dict[str, list[int]] | None = None,
    after_metadata_objects: Sequence[Mapping[str, object]] | None = None,
) -> StaleUncertaintyInputs:
    distance_from_remembered = _ground_distance(remembered_pos, found_pos) if found_pos is not None else None
    predicted_detector_confidence = 0.0
    if found_pos is not None and target_visible_after_revisit:
        predicted_detector_confidence = 1.0
    elif found_pos is not None:
        predicted_detector_confidence = 0.5
    else:
        predicted_detector_confidence = 0.0
    obj_type = _object_type(before_obj)
    if gsam_detection_history is not None and obj_type in gsam_detection_history:
        past = gsam_detection_history[obj_type]
        if len(past) >= 3:
            empirical_rate = sum(1 for d in past if d) / len(past)
            predicted_detector_confidence = round(predicted_detector_confidence * 0.4 + empirical_rate * 0.6, 4)
    if gsam_detection_counts is not None and obj_type in gsam_detection_counts:
        counts = gsam_detection_counts[obj_type]
        if counts:
            avg_count = sum(counts) / len(counts)
            if avg_count > 50:
                predicted_detector_confidence = max(0.0, predicted_detector_confidence - 0.5)
            elif avg_count > 20:
                predicted_detector_confidence = max(0.0, predicted_detector_confidence - 0.3)
    detector_confidence = predicted_detector_confidence
    depth_consistency: float | None = None
    if distance_from_remembered is not None:
        depth_consistency = max(0.0, round(1.0 - distance_from_remembered / 0.5, 4))
    pickupable = bool(before_obj.get("pickupable"))
    moveable = bool(before_obj.get("moveable") or before_obj.get("movable") or pickupable)
    if pickupable:
        mobility_prior = 0.85
    elif moveable:
        mobility_prior = 0.5
    else:
        mobility_prior = 0.15
    task_importance = 1.0 if pickupable else (0.5 if moveable else 0.3)
    historical_failure = 0.0
    if gsam_detection_history is not None and obj_type in gsam_detection_history:
        past = gsam_detection_history[obj_type]
        if past:
            historical_failure = round(1.0 - sum(1 for d in past if d) / len(past), 4)
    same_type_density_score = 0.0
    same_type_nearby_count = 0
    same_type_scene_total = 0
    if after_metadata_objects is not None:
        same_type_objs = [o for o in after_metadata_objects if _valid_position(o) and _object_type(o) == obj_type]
        same_type_scene_total = len(same_type_objs)
        if same_type_scene_total > 0:
            same_type_nearby_count = sum(
                1 for o in same_type_objs
                if _ground_distance(remembered_pos, cast(Mapping[str, float], o["position"])) < 0.5
            )
            same_type_density_score = round(same_type_nearby_count / same_type_scene_total, 4)
    return StaleUncertaintyInputs(
        detector_confidence=detector_confidence,
        depth_consistency=depth_consistency,
        object_mobility_prior=mobility_prior,
        time_interval=time_interval,
        task_importance=task_importance,
        historical_failure=historical_failure,
        scene_change_score=1.0,
        target_visible_after_revisit=target_visible_after_revisit,
        visibility_error=visibility_error,
        distance_from_remembered=distance_from_remembered,
        teleport_success=teleport_success,
        same_type_density_score=same_type_density_score,
        same_type_nearby_count=same_type_nearby_count,
        same_type_scene_total=same_type_scene_total,
        visual_instance_confusion_prior=_visual_instance_confusion_prior(obj_type),
    )


def _apply_calibrated_ev_penalty(p_stale: float, inputs: StaleUncertaintyInputs, reason: str) -> tuple[float, str]:
    calibrated = p_stale
    calibrated_reason = reason
    if inputs.same_type_nearby_count >= 1:
        density_penalty = 0.25 * inputs.same_type_density_score
        calibrated = round(max(0.03, calibrated - density_penalty), 4)
    if inputs.visual_instance_confusion_prior > 0.0:
        visual_penalty = 0.35 * inputs.visual_instance_confusion_prior
        calibrated = round(max(0.03, calibrated - visual_penalty), 4)
    if calibrated != p_stale:
        calibrated_reason = f"{reason};visual_instance_confusion_calibrated"
        if inputs.same_type_nearby_count >= 1:
            calibrated_reason = f"{calibrated_reason};ambiguity_calibrated"
    return calibrated, calibrated_reason


def _estimate_stale_uncertainty(inputs: StaleUncertaintyInputs) -> StaleUncertaintyEstimate:
    mobility_bonus = 0.2 * inputs.object_mobility_prior
    time_factor = max(0.5, min(2.0, 0.8 + 0.2 * inputs.time_interval))
    if inputs.distance_from_remembered is not None:
        if inputs.distance_from_remembered > 0.05:
            probability = min(0.95, (0.65 + mobility_bonus) * time_factor)
            reason = "visible_target_displaced_from_remembered_location"
        else:
            depth_discount = (1.0 - inputs.depth_consistency) * 0.1 if inputs.depth_consistency is not None else 0.0
            probability = max(0.03, (0.2 - (0.1 if inputs.object_mobility_prior <= 0.15 else 0.0) + depth_discount) * time_factor)
            reason = "visible_target_near_remembered_location"
    elif inputs.visibility_error is not None or not inputs.target_visible_after_revisit:
        if not inputs.teleport_success:
            probability = 0.5 * time_factor
            reason = "revisit_failed_visibility_uninformative"
        else:
            detector_penalty = (1.0 - (inputs.detector_confidence or 0.0)) * 0.1
            history_penalty = inputs.historical_failure * 0.1
            probability = min(0.95, (0.55 + mobility_bonus + detector_penalty + history_penalty) * time_factor)
            reason = "target_not_visible_after_revisit_sweep"
    else:
        probability = (0.35 + 0.1 * inputs.object_mobility_prior) * time_factor
        reason = "visibility_context_ambiguous"
    if inputs.same_type_density_score > 0.5:
        density_penalty = 0.15 * inputs.same_type_density_score
        probability = max(0.03, probability - density_penalty)
        reason = f"{reason};same_type_instance_density_penalty"
    probability = round(min(0.95, max(0.03, probability)), 4)
    return StaleUncertaintyEstimate(probability=probability, reason=reason, inputs=inputs)


def _estimate_stale_probability(
    before_obj: Mapping[str, object],
    remembered_pos: Mapping[str, float],
    *,
    target_visible_after_revisit: bool,
    visibility_error: str | None,
    found_pos: Mapping[str, float] | None,
    teleport_success: bool,
) -> tuple[float, str]:
    inputs = _build_uncertainty_inputs(
        before_obj,
        remembered_pos,
        target_visible_after_revisit=target_visible_after_revisit,
        visibility_error=visibility_error,
        found_pos=found_pos,
        teleport_success=teleport_success,
    )
    estimate = _estimate_stale_uncertainty(inputs)
    return estimate.probability, estimate.reason


def _expected_verification_value(p_stale: float, stale_action_cost: float, verification_cost: float) -> float:
    return p_stale * stale_action_cost - verification_cost


def _active_verification_decision(expected_value: float, threshold: float) -> tuple[bool, str]:
    if expected_value > threshold + 1e-12:
        return True, "expected_verification_value_above_threshold"
    return False, "expected_verification_value_not_strictly_above_threshold"


def _verification_budget_limit(config: LiveGSAMClosedLoopConfig, candidate_count: int) -> int:
    if config.verification_budget is None:
        return candidate_count
    return max(0, min(int(config.verification_budget), candidate_count))


def _candidate_int(row: Mapping[str, object], key: str, default: int) -> int:
    value = row.get(key)
    return int(value) if isinstance(value, (int, float, str)) and not isinstance(value, bool) else default


def _rank_verification_candidates(candidates: Sequence[Json], *, budget: int) -> list[Json]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -float(item[1].get("expected_verification_value") or 0.0),
            _candidate_int(item[1], "target_idx", 0),
            str(item[1].get("target_object_id") or item[1].get("remembered_location") or item[1].get("target") or ""),
            _candidate_int(item[1], "row_idx", item[0]),
            item[0],
        ),
    )
    return [row for _, row in ranked[: max(0, budget)]]


def _apply_budget_selection(rows: Sequence[Json], config: LiveGSAMClosedLoopConfig) -> list[Json]:
    candidates = [row for row in rows if row.get("policy_should_verify") is True]
    budget = _verification_budget_limit(config, len(candidates))
    selected = _rank_verification_candidates(candidates, budget=budget)
    selected_ids = {_candidate_int(row, "row_idx", -1) for row in selected}
    ranked = _rank_verification_candidates(candidates, budget=len(candidates))
    rank_by_row_idx = {_candidate_int(row, "row_idx", -1): rank for rank, row in enumerate(ranked, start=1)}
    for row in rows:
        row["selection_policy_name"] = config.selection_policy_name
        row["verification_budget"] = config.verification_budget
        row_idx = _candidate_int(row, "row_idx", -1)
        rank = rank_by_row_idx.get(row_idx)
        row["candidate_rank_by_ev"] = rank
        selected_by_budget = row_idx in selected_ids
        row["candidate_selected_by_budget"] = selected_by_budget
        if row.get("policy_should_verify") is not True:
            row["budget_selection_reason"] = "not_ev_gate_candidate"
            row["budget_remaining_after_selection"] = budget
        elif selected_by_budget:
            row["budget_selection_reason"] = "selected_by_expected_verification_value_budget"
            row["budget_remaining_after_selection"] = max(0, budget - rank) if rank is not None else 0
        else:
            row["budget_selection_reason"] = "skipped_by_verification_budget"
            row["budget_remaining_after_selection"] = 0
        row["verifier_used"] = selected_by_budget
    return selected


def _mutate_memory_object(
    memory: list[Json],
    target_obj_id: str,
    new_position: dict[str, float] | None,
    confidence: float | None,
) -> Json:
    """Update an object's position in the memory store snapshot. Returns mutation evidence dict."""
    evidence: Json = {"memory_mutated": False, "memory_old_position": None, "memory_new_position": None}
    if new_position is None:
        return evidence
    for obj in memory:
        if _object_id(obj) == target_obj_id and _valid_position(obj):
            old_pos = cast(Mapping[str, object], obj.get("position") or {})
            evidence["memory_old_position"] = {k: float(cast(float, old_pos.get(k, 0.0))) for k in ("x", "y", "z")}
            pos = cast(dict[str, float], obj["position"])
            pos["x"] = float(new_position.get("x", pos.get("x", 0.0)))
            pos["y"] = float(new_position.get("y", pos.get("y", 0.0)))
            pos["z"] = float(new_position.get("z", pos.get("z", 0.0)))
            evidence["memory_new_position"] = dict(new_position)
            evidence["memory_mutated"] = True
            break
    return evidence


def _default_memory_update_fields() -> Json:
    return {
        "memory_update_instrumented": False,
        "memory_update_action": "not_applicable",
        "memory_update_source": None,
        "memory_update_old_location": None,
        "memory_update_candidate_location": None,
        "memory_update_confidence": None,
        "memory_update_boundary": MEMORY_UPDATE_BOUNDARY,
        "memory_mutated": False,
        "memory_old_position": None,
        "memory_new_position": None,
    }


def _memory_update_fields_from_verified_row(row: Mapping[str, object]) -> Json:
    fields = _default_memory_update_fields()
    fields["memory_update_instrumented"] = True
    fields["memory_update_old_location"] = row.get("remembered_location")
    fields["memory_update_confidence"] = row.get("decision_confidence")
    if row.get("decision_stale") is True:
        fields["memory_update_action"] = "propose_detector_evidence_refresh"
        fields["memory_update_source"] = "live_controller_revisit_detector_verifier_evidence"
        fields["memory_update_candidate_location"] = row.get("found_position_after_revisit")
    else:
        fields["memory_update_action"] = "none"
        fields["memory_update_source"] = "verified_fresh_or_no_stale_detector_evidence"
    return fields


def _task_bridge_action(row: Mapping[str, object]) -> str | None:
    if bool(row.get("target_pickupable")):
        return "PickupObject"
    if bool(row.get("target_openable")):
        return "OpenObject"
    return None


def _default_task_execution_fields() -> Json:
    return {
        "task_execution_evaluated": False,
        "downstream_task_success": None,
        "execute_action": "not_executed",
        "execution_boundary": "verification_only_no_downstream_action",
        "task_execution_failure_reason": None,
        "task_bridge_navigation_mode": "not_executed",
        "task_bridge_used_teleportfull": None,
        "task_bridge_path_steps": 0,
        "task_bridge_navigation_actions": [],
        "task_bridge_navigation_failure_reason": None,
        "interaction_attempt_mode": None,
        "interaction_used_force_action": None,
        "interaction_object_id_used": None,
        "interaction_rotation_actions": 0,
        "interaction_agent_object_distance": None,
        "interaction_target_visible": None,
        "interaction_horizon_deg": None,
        "interaction_visibility_sweep_used": None,
        "task_bridge_alternate_poses_tried": 0,
    }


def _default_revisit_fields() -> Json:
    return {
        "revisit_action": "TeleportFull",
        "used_teleportfull_for_revisit": True,
        "revisit_path_steps": 0,
        "revisit_navigation_actions": [],
        "revisit_navigation_failure_reason": None,
    }


def _normalize_nav_position(position: Mapping[str, float]) -> dict[str, float]:
    return {"x": float(position["x"]), "y": float(position.get("y", 0.9)), "z": float(position["z"])}


def _reachable_waypoint_route(
    start_position: Mapping[str, float],
    target_position: Mapping[str, float],
    reachable: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    if not reachable:
        return [_normalize_nav_position(target_position)]
    nodes = [_normalize_nav_position(pos) for pos in reachable]
    start_idx = min(range(len(nodes)), key=lambda idx: _ground_distance(_normalize_nav_position(start_position), nodes[idx]))
    goal_idx = min(range(len(nodes)), key=lambda idx: _ground_distance(_normalize_nav_position(target_position), nodes[idx]))
    if start_idx == goal_idx:
        return [nodes[goal_idx]]

    frontier: list[int] = [start_idx]
    came_from: dict[int, int | None] = {start_idx: None}
    while frontier:
        current_idx = frontier.pop(0)
        if current_idx == goal_idx:
            break
        current = nodes[current_idx]
        neighbors = sorted(
            (idx for idx, pos in enumerate(nodes) if idx not in came_from and _ground_distance(current, pos) <= 0.31),
            key=lambda idx: (_ground_distance(nodes[idx], nodes[goal_idx]), idx),
        )
        for neighbor_idx in neighbors:
            came_from[neighbor_idx] = current_idx
            frontier.append(neighbor_idx)

    if goal_idx not in came_from:
        return [nodes[goal_idx]]
    route_indices: list[int] = []
    cursor: int | None = goal_idx
    while cursor is not None:
        route_indices.append(cursor)
        cursor = came_from[cursor]
    route_indices.reverse()
    return [nodes[idx] for idx in route_indices[1:]] or [nodes[goal_idx]]


def _stepwise_revisit_to_position(
    controller: ControllerLike,
    *,
    start_position: Mapping[str, float],
    target_position: Mapping[str, float],
    max_steps: int = 12,
    waypoints: Sequence[Mapping[str, float]] | None = None,
) -> tuple[dict[str, float], list[str], int, bool, str | None]:
    current = _normalize_nav_position(start_position)
    route = [_normalize_nav_position(pos) for pos in (waypoints or [target_position])]
    action_budget = max_steps if waypoints is None else min(80, max(max_steps, len(route) * 4))
    current_heading = 0.0
    actions: list[str] = []
    path_steps = 0
    failure_reason: str | None = None

    for route_idx, goal in enumerate(route):
        waypoint_reached = False
        while len(actions) < action_budget:
            if _ground_distance(current, goal) <= 0.13:
                waypoint_reached = True
                break
            event, _moved, success, current_heading, action = _nav_navigation_step(controller, current, goal, current_heading)
            actions.append(action)
            if action == "MoveAhead":
                path_steps += 1
                if not success:
                    failure_reason = "collision_stuck"
                    break
                current = _nav_agent_position(_metadata(event))
            elif action in {"RotateLeft", "RotateRight"}:
                current = _nav_agent_position(_metadata(event))
        if failure_reason is not None:
            break
        if not waypoint_reached:
            waypoint_reached = _ground_distance(current, goal) <= 0.13
        if not waypoint_reached:
            failure_reason = "max_steps_exceeded"
            break
        if route_idx == len(route) - 1:
            return current, actions, path_steps, True, None

    if failure_reason is None:
        failure_reason = "max_steps_exceeded"
    return current, actions, path_steps, False, failure_reason


def _alternate_reachable_poses(
    target_position: Mapping[str, float],
    reachable: Sequence[Mapping[str, float]],
    k: int,
    exclude_position: Mapping[str, float] | None,
) -> list[dict[str, float]]:
    """Deterministic list of up to k reachable poses nearest to target_position, skipping exclude_position."""
    ordered = sorted(reachable, key=lambda pose: _ground_distance(target_position, pose))
    alternates: list[dict[str, float]] = []
    for pose in ordered:
        if exclude_position is not None and _ground_distance(pose, exclude_position) <= 1e-6:
            continue
        alternates.append({key: float(cast(float, pose.get(key, 0.0))) for key in ("x", "y", "z")})
        if len(alternates) >= k:
            break
    return alternates


def _attempt_pickup_honest(
    controller: ControllerLike,
    *,
    object_id: str,
    target_position: Mapping[str, float],
    action: str,
    max_rotations: int = 8,
) -> tuple[Json, bool, str | None]:
    """Face the target position, then attempt the action with forceAction=False.

    Returns ``(interaction_fields, success, failure_reason)``. The simulator enforces
    proximity and visibility preconditions, so a stale-location arm can fail when the
    object has moved away.
    """
    fields: Json = {
        "interaction_attempt_mode": "honest_face_then_pick",
        "interaction_used_force_action": False,
        "interaction_object_id_used": object_id,
        "interaction_rotation_actions": 0,
        "interaction_agent_object_distance": None,
        "interaction_target_visible": None,
        "interaction_horizon_deg": 0.0,
        "interaction_visibility_sweep_used": False,
    }
    target_pos = {
        "x": float(target_position.get("x", 0.0)),
        "y": float(target_position.get("y", 0.0)),
        "z": float(target_position.get("z", 0.0)),
    }
    rotations = 0
    probe = _metadata(controller.step("Pass"))
    agent = probe.get("agent")
    agent_map = agent if isinstance(agent, Mapping) else {}
    agent_pos = agent_map.get("position")
    rotation = agent_map.get("rotation")
    if isinstance(agent_pos, Mapping) and isinstance(rotation, Mapping):
        agent_x = float(cast(float, agent_pos.get("x", 0.0)))
        agent_z = float(cast(float, agent_pos.get("z", 0.0)))
        desired_yaw = math.degrees(math.atan2(target_pos["x"] - agent_x, target_pos["z"] - agent_z)) % 360.0
        diff = (desired_yaw - float(rotation.get("y", 0.0)) + 180.0) % 360.0 - 180.0
        steps = min(max_rotations, int(round(abs(diff) / 90.0)))
        for _ in range(steps):
            controller.step("RotateRight" if diff > 0 else "RotateLeft")
        rotations = steps
    fields["interaction_rotation_actions"] = rotations

    final = _metadata(controller.step("Pass"))
    raw_objects = final.get("objects", [])
    live_objects = [dict(obj) for obj in raw_objects if isinstance(obj, Mapping)] if isinstance(raw_objects, Sequence) else []
    target_obj = next((obj for obj in live_objects if _object_id(obj) == object_id), None)
    agent = final.get("agent")
    agent_map = agent if isinstance(agent, Mapping) else {}
    agent_position = agent_map.get("position")

    def _target_visible(meta: Mapping[str, object]) -> tuple[dict[str, object] | None, bool]:
        raw = meta.get("objects", [])
        objs = [dict(obj) for obj in raw if isinstance(obj, Mapping)] if isinstance(raw, Sequence) else []
        obj = next((candidate for candidate in objs if _object_id(candidate) == object_id), None)
        return obj, bool(obj.get("visible")) if obj is not None else False

    if target_obj is not None and not bool(target_obj.get("visible")):
        fields["interaction_visibility_sweep_used"] = True
        for look_action, degrees in (
            ("LookDown", 15.0),
            ("LookDown", 15.0),
            ("LookDown", 15.0),
            ("LookDown", 15.0),
            ("LookUp", 15.0),
            ("LookUp", 15.0),
            ("LookUp", 15.0),
            ("LookUp", 15.0),
            ("LookUp", 15.0),
            ("LookUp", 15.0),
        ):
            controller.step(look_action, degrees=degrees)
            probe = _metadata(controller.step("Pass"))
            probe_obj, visible = _target_visible(probe)
            agent_probe = probe.get("agent")
            agent_probe_map = agent_probe if isinstance(agent_probe, Mapping) else {}
            if isinstance(agent_probe_map.get("cameraHorizon"), (int, float)):
                fields["interaction_horizon_deg"] = round(float(agent_probe_map["cameraHorizon"]), 2)
            if probe_obj is not None:
                target_obj = probe_obj
            if visible:
                break
    if target_obj is not None and isinstance(agent_position, Mapping) and _valid_position(target_obj):
        fields["interaction_agent_object_distance"] = round(
            _ground_distance(cast(Mapping[str, float], agent_position), _object_position(target_obj)), 4
        )
        fields["interaction_target_visible"] = bool(target_obj.get("visible"))
    if target_obj is None:
        return fields, False, "target_object_id_missing"

    event = controller.step(action, objectId=object_id, forceAction=False)
    meta = _metadata(event)
    success = bool(meta.get("lastActionSuccess", False))
    return fields, success, (None if success else str(meta.get("errorMessage") or "action_failed"))


def _execute_pickup_attempt(
    controller: ControllerLike,
    *,
    action: str,
    object_id: str,
    target_position: Mapping[str, float] | None,
    honest_interaction: bool,
    boundary: str,
) -> Json:
    fields: Json = {}
    if honest_interaction and target_position is not None:
        interaction_fields, success, failure = _attempt_pickup_honest(
            controller, object_id=object_id, target_position=target_position, action=action
        )
        fields.update(interaction_fields)
    else:
        event = controller.step(action, objectId=object_id, forceAction=True)
        meta = _metadata(event)
        success = bool(meta.get("lastActionSuccess", False))
        failure = None if success else str(meta.get("errorMessage") or "action_failed")
        fields["interaction_attempt_mode"] = "forced_action"
        fields["interaction_used_force_action"] = True
        fields["interaction_object_id_used"] = object_id
    fields["task_execution_evaluated"] = True
    fields["execute_action"] = action
    fields["downstream_task_success"] = success
    fields["execution_boundary"] = boundary
    fields["task_execution_failure_reason"] = failure
    return fields


def _execute_task_bridge_for_row(
    row: Json,
    *,
    config: LiveGSAMClosedLoopConfig,
    make_controller: ControllerFactory,
) -> Json:
    fields = _default_task_execution_fields()
    if not config.execute_task_bridge:
        return fields
    if row.get("verifier_used") is not True or row.get("memory_update_source") != "live_controller_revisit_detector_verifier_evidence":
        return fields
    if row.get("memory_mutated") is not True or row.get("decision_stale") is not True:
        return fields
    action = _task_bridge_action(row)
    if action is None:
        fields["task_execution_evaluated"] = True
        fields["execute_action"] = "no_compatible_pickup_or_open_action"
        fields["downstream_task_success"] = False
        fields["execution_boundary"] = TASK_BRIDGE_BOUNDARY
        fields["task_execution_failure_reason"] = "target_not_pickupable_or_openable"
        fields["task_bridge_navigation_mode"] = "not_applicable_no_compatible_action"
        return fields

    controller: ControllerLike | None = None
    try:
        scene = str(row.get("scene") or "")
        seed = int(row.get("seed") or 0)
        kwargs = _make_controller_kwargs(scene, config.width, config.height, config.platform_name, capture_images=False)
        controller = make_controller(**kwargs)
        spawn_event = controller.step("InitialRandomSpawn", randomSeed=seed, forceVisible=True, numPlacementAttempts=5)
        current_objects_raw = _metadata(spawn_event).get("objects", [])
        spawn_meta = _metadata(spawn_event)
        current_objects = [cast(Mapping[str, object], obj) for obj in current_objects_raw] if isinstance(current_objects_raw, Sequence) else []
        target_position = row.get("memory_new_position") or row.get("memory_update_candidate_location") or row.get("found_position_after_revisit")
        position: dict[str, float] | None = None
        if isinstance(target_position, Mapping):
            position = {k: float(cast(float, target_position.get(k, 0.0))) for k in ("x", "y", "z")}
            if config.revisit_mode == "stepwise":
                navigation_goal = dict(position)
                reachable_event = controller.step("GetReachablePositions")
                reachable = _reachable_positions(cast(Sequence[object], _metadata(reachable_event).get("actionReturn", [])))
                waypoints: list[dict[str, float]] | None = None
                if reachable:
                    navigation_goal = nearest_revisit_position(position, reachable)
                    waypoints = _reachable_waypoint_route(_nav_agent_position(spawn_meta), navigation_goal, reachable)
                _, actions, path_steps, reached, failure_reason = _stepwise_revisit_to_position(
                    controller,
                    start_position=_nav_agent_position(spawn_meta),
                    target_position=navigation_goal,
                    waypoints=waypoints,
                )
                fields["task_bridge_navigation_mode"] = "stepwise_navigation"
                fields["task_bridge_used_teleportfull"] = False
                fields["task_bridge_path_steps"] = path_steps
                fields["task_bridge_navigation_actions"] = actions
                fields["task_bridge_navigation_failure_reason"] = failure_reason
                if not reached:
                    fields["task_execution_evaluated"] = True
                    fields["execute_action"] = action
                    fields["downstream_task_success"] = False
                    fields["execution_boundary"] = STEPWISE_TASK_BRIDGE_BOUNDARY
                    fields["task_execution_failure_reason"] = failure_reason or "stepwise_task_bridge_navigation_failed"
                    return fields
            else:
                teleport_event = controller.step(
                    "TeleportFull",
                    x=position["x"],
                    y=position.get("y", 0.9),
                    z=position["z"],
                    rotation={"x": 0.0, "y": 0.0, "z": 0.0},
                    horizon=0.0,
                    standing=True,
                )
                fields["task_bridge_navigation_mode"] = "teleportfull"
                fields["task_bridge_used_teleportfull"] = True
                teleport_objects_raw = _metadata(teleport_event).get("objects", [])
                if isinstance(teleport_objects_raw, Sequence):
                    current_objects = [cast(Mapping[str, object], obj) for obj in teleport_objects_raw]
        object_id = _resolve_current_object_id(str(row.get("target_object_id") or ""), current_objects, position)
        fields.update(
            _execute_pickup_attempt(
                controller,
                action=action,
                object_id=object_id,
                target_position=position,
                honest_interaction=config.honest_interaction,
                boundary=STEPWISE_TASK_BRIDGE_BOUNDARY if config.revisit_mode == "stepwise" else TASK_BRIDGE_BOUNDARY,
            )
        )
        if (
            config.honest_interaction
            and config.max_alternate_poses > 0
            and config.revisit_mode == "stepwise"
            and fields.get("downstream_task_success") is not True
            and position is not None
            and reachable
        ):
            first_goal = nearest_revisit_position(cast(dict[str, float], position), reachable)
            alternates = _alternate_reachable_poses(
                cast(Mapping[str, float], position), reachable, config.max_alternate_poses, first_goal
            )
            attempted_alternates = 0
            for candidate in alternates:
                attempted_alternates += 1
                current_agent = _nav_agent_position(_metadata(controller.step("Pass")))
                waypoints = _reachable_waypoint_route(current_agent, candidate, reachable)
                _, extra_actions, extra_steps, alt_reached, _alt_reason = _stepwise_revisit_to_position(
                    controller,
                    start_position=current_agent,
                    target_position=candidate,
                    waypoints=waypoints,
                )
                fields["task_bridge_path_steps"] = int(fields.get("task_bridge_path_steps") or 0) + extra_steps
                fields["task_bridge_navigation_actions"] = list(fields.get("task_bridge_navigation_actions") or []) + list(extra_actions)
                if not alt_reached:
                    continue
                retry_fields = _execute_pickup_attempt(
                    controller,
                    action=action,
                    object_id=object_id,
                    target_position=position,
                    honest_interaction=True,
                    boundary=STEPWISE_TASK_BRIDGE_BOUNDARY,
                )
                fields.update(retry_fields)
                if retry_fields.get("downstream_task_success") is True:
                    break
            fields["task_bridge_alternate_poses_tried"] = attempted_alternates
    except Exception as exc:  # pragma: no cover - defensive boundary for optional simulator failures
        fields["task_execution_evaluated"] = True
        fields["execute_action"] = action
        fields["downstream_task_success"] = False
        fields["execution_boundary"] = TASK_BRIDGE_BOUNDARY
        fields["task_execution_failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            controller.stop()
    return fields


def _execute_passive_task_bridge_for_row(
    row: Json,
    *,
    config: LiveGSAMClosedLoopConfig,
    make_controller: ControllerFactory,
) -> Json:
    """Execute a passive downstream task from stale memory_old_position without verifier evidence.

    This helper is the passive-arm counterpart of _execute_task_bridge_for_row.
    It uses only memory_old_position (stale, unverified) as the navigation target,
    never requires verifier_used/memory_mutated/decision_stale gates, and never
    mutates memory.  Navigation is always stepwise (no TeleportFull, no TeleportObject).
    """
    fields = _default_task_execution_fields()
    if not config.execute_task_bridge:
        return fields

    action = _task_bridge_action(row)
    if action is None:
        fields["task_execution_evaluated"] = True
        fields["execute_action"] = "no_compatible_pickup_or_open_action"
        fields["downstream_task_success"] = False
        fields["execution_boundary"] = LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY
        fields["task_execution_failure_reason"] = "target_not_pickupable_or_openable"
        fields["task_bridge_navigation_mode"] = "not_applicable_no_compatible_action"
        return fields

    # --- resolve stale memory position -------------------------------------------------
    target_position_source = row.get("memory_old_position")
    position: dict[str, float] | None = None
    if isinstance(target_position_source, Mapping):
        try:
            position = {
                k: float(cast(float, target_position_source.get(k, 0.0)))
                for k in ("x", "y", "z")
            }
        except (TypeError, ValueError):
            position = None

    if position is None:
        fields["task_execution_evaluated"] = True
        fields["execute_action"] = action
        fields["downstream_task_success"] = False
        fields["execution_boundary"] = LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY
        fields["task_execution_failure_reason"] = "no_memory_old_position_for_passive_task_bridge"
        fields["task_bridge_navigation_mode"] = "not_applicable_no_memory_position"
        return fields

    controller: ControllerLike | None = None
    try:
        scene = str(row.get("scene") or "")
        seed = int(row.get("seed") or 0)
        kwargs = _make_controller_kwargs(scene, config.width, config.height, config.platform_name, capture_images=False)
        controller = make_controller(**kwargs)
        spawn_event = controller.step("InitialRandomSpawn", randomSeed=seed, forceVisible=True, numPlacementAttempts=5)
        spawn_meta = _metadata(spawn_event)
        current_objects_raw = spawn_meta.get("objects", [])
        current_objects = [cast(Mapping[str, object], obj) for obj in current_objects_raw] if isinstance(current_objects_raw, Sequence) else []

        # Always stepwise for the passive arm — never TeleportFull, never TeleportObject.
        navigation_goal = dict(position)
        reachable_event = controller.step("GetReachablePositions")
        reachable = _reachable_positions(cast(Sequence[object], _metadata(reachable_event).get("actionReturn", [])))
        waypoints: list[dict[str, float]] | None = None
        if reachable:
            navigation_goal = nearest_revisit_position(position, reachable)
            waypoints = _reachable_waypoint_route(_nav_agent_position(spawn_meta), navigation_goal, reachable)
        _, actions, path_steps, reached, failure_reason = _stepwise_revisit_to_position(
            controller,
            start_position=_nav_agent_position(spawn_meta),
            target_position=navigation_goal,
            waypoints=waypoints,
        )
        fields["task_bridge_navigation_mode"] = "stepwise_navigation_live_passive"
        fields["task_bridge_used_teleportfull"] = False
        fields["task_bridge_path_steps"] = path_steps
        fields["task_bridge_navigation_actions"] = actions
        fields["task_bridge_navigation_failure_reason"] = failure_reason
        if not reached:
            fields["task_execution_evaluated"] = True
            fields["execute_action"] = action
            fields["downstream_task_success"] = False
            fields["execution_boundary"] = LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY
            fields["task_execution_failure_reason"] = failure_reason or "stepwise_passive_task_bridge_navigation_failed"
            return fields

        object_id = _resolve_current_object_id(str(row.get("target_object_id") or ""), current_objects, position)
        fields.update(
            _execute_pickup_attempt(
                controller,
                action=action,
                object_id=object_id,
                target_position=position,
                honest_interaction=config.honest_interaction,
                boundary=LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for optional simulator failures
        fields["task_execution_evaluated"] = True
        fields["execute_action"] = action
        fields["downstream_task_success"] = False
        fields["execution_boundary"] = LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY
        fields["task_execution_failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            controller.stop()
    return fields


def _controller_factory(**kwargs: object) -> ControllerLike:
    ct = importlib.import_module("ai2thor.controller")
    cls = cast(Callable[..., ControllerLike], ct.Controller)
    return cls(**kwargs)


def _build_backend(config: LiveGSAMClosedLoopConfig) -> VerifierBackendLike:
    if config.verifier_backend_name != "grounded_sam2":
        raise ValueError("closed-loop runner currently supports only grounded_sam2 verifier")
    if config.grounding_config is None or config.grounding_checkpoint is None:
        raise ValueError("grounded_sam2 requires --grounding-config and --grounding-checkpoint")
    from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import GroundedSAM2LocationBackend

    return GroundedSAM2LocationBackend(
        grounding_config=config.grounding_config,
        grounding_checkpoint=config.grounding_checkpoint,
        sam2_config=config.sam2_config,
        sam2_checkpoint=config.sam2_checkpoint,
        device=config.verifier_device,
        backend_load_timeout_seconds=config.backend_load_timeout_seconds,
    )


def blocked_result(capability: CapabilityReport) -> Json:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "capability": capability.as_dict(),
        "summary": {"rows": 0, "verifier_used_rows": 0, "controller_started": False},
        "rows": [],
        "claim_boundary": NO_CONTROLLER_BOUNDARY,
    }


def run_live_gsam_closed_loop(
    config: LiveGSAMClosedLoopConfig,
    *,
    controller_factory: ControllerFactory | None = None,
    verifier_factory: VerifierFactory | None = None,
    capability: CapabilityReport | None = None,
) -> Json:
    capability = capability or ai2thor_capability()
    if not capability.available:
        return blocked_result(capability)
    if config.build_base_url:
        configure_build_mirror(config.build_base_url)

    image_dir = config.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    make_controller = controller_factory or _controller_factory
    verifier: VerifierBackendLike | None = None
    rows: list[Json] = []
    pending_verifications: dict[int, tuple[RearrangementTask, str]] = {}
    before_memory_store: dict[tuple[str, int], list[Json]] = {}
    gsam_detection_history: dict[str, list[bool]] = {}
    gsam_detection_counts: dict[str, list[int]] = {}
    session_errors: list[Json] = []
    controller_started = False
    targeted_specs = _load_targeted_case_specs(config.case_list) if config.case_list is not None else []
    targeted_lookup = {_targeted_case_key(spec.scene, spec.seed, spec.target): spec for spec in targeted_specs}
    matched_targeted_keys: set[tuple[str, int, str]] = set()

    for seed in config.seeds:
        for scene in config.scenes:
            if len(rows) >= config.max_rows:
                break
            controller: ControllerLike | None = None
            try:
                kwargs = _make_controller_kwargs(scene, config.width, config.height, config.platform_name, capture_images=True)
                controller = make_controller(**kwargs)
                controller_started = True
                reach_event = controller.step("GetReachablePositions")
                reachable = _reachable_positions(cast(Sequence[object], _metadata(reach_event).get("actionReturn", [])))
                before_objects = _collect_memory(
                    controller,
                    config.before_probe_actions or DEFAULT_PROBE_ACTIONS,
                    capture_images=True,
                    image_dir=image_dir,
                    scene_name=scene,
                    phase="before",
                )
                selected = _select_memory_targets(before_objects, config.targets_per_scene_seed, preferred_objects=config.target_objects)
                before_memory_store[(scene, seed)] = [dict(o) for o in before_objects]
                spawn_event = controller.step(
                    "InitialRandomSpawn",
                    randomSeed=seed,
                    forceVisible=True,
                    numPlacementAttempts=5,
                )
                spawn_meta = _metadata(spawn_event)
                spawn_ok = bool(spawn_meta.get("lastActionSuccess", False))
                spawn_objects = spawn_meta.get("objects", [])
                after_metadata_objects = [dict(obj) for obj in spawn_objects if isinstance(obj, Mapping)] if isinstance(spawn_objects, Sequence) else []
                after_global = _collect_memory(controller, DEFAULT_PROBE_ACTIONS, capture_images=False, scene_name=scene, phase="after")

                for target_idx, target in enumerate(selected):
                    if len(rows) >= config.max_rows:
                        break
                    target_type = _object_type(target)
                    targeted_key = _targeted_case_key(scene, seed, target_type)
                    targeted_spec = targeted_lookup.get(targeted_key)
                    if config.case_list is not None and targeted_spec is None:
                        continue
                    if targeted_spec is not None:
                        matched_targeted_keys.add(targeted_key)
                    remembered = rearrangement_location(scene, dict(target), "before")
                    remembered_pos = _object_position(dict(target))
                    revisit_pos = nearest_revisit_position(remembered_pos, reachable) if reachable else dict(remembered_pos)
                    revisit_fields = _default_revisit_fields()
                    if config.revisit_mode == "stepwise":
                        start_position = _nav_agent_position(spawn_meta)
                        waypoints = _reachable_waypoint_route(start_position, revisit_pos, reachable) if reachable else None
                        _, actions, path_steps, reached, failure_reason = _stepwise_revisit_to_position(
                            controller,
                            start_position=start_position,
                            target_position=revisit_pos,
                            waypoints=waypoints,
                        )
                        revisit_fields.update(
                            {
                                "revisit_action": "stepwise_navigation",
                                "used_teleportfull_for_revisit": False,
                                "revisit_path_steps": path_steps,
                                "revisit_navigation_actions": actions,
                                "revisit_navigation_failure_reason": failure_reason,
                            }
                        )
                        teleport_ok = reached
                    else:
                        yaw, horizon = _look_at_angles(revisit_pos, remembered_pos)
                        teleport_event = controller.step(
                            "TeleportFull",
                            x=revisit_pos["x"],
                            y=revisit_pos.get("y", 0.9),
                            z=revisit_pos["z"],
                            rotation={"x": 0.0, "y": yaw, "z": 0.0},
                            horizon=horizon,
                            standing=True,
                        )
                        teleport_ok = bool(_metadata(teleport_event).get("lastActionSuccess", False))
                    after_objects, visible_after_revisit, visibility_error, found_pos = _probe_visibility(
                        controller,
                        target_type,
                        DEFAULT_PROBE_ACTIONS,
                        capture_images=True,
                        image_dir=image_dir,
                        scene_name=scene,
                        phase="after",
                        action_offset=200 + len(rows) * 10,
                    )
                    nav_search_used = False
                    nav_search_positions_probed = 0
                    if not visible_after_revisit and config.nav_search_steps > 0 and reachable:
                        search_positions = sorted(
                            reachable,
                            key=lambda p: _ground_distance(revisit_pos, p),
                        )[1:1 + config.nav_search_steps]
                        for i, search_pos in enumerate(search_positions):
                            nav_search_positions_probed += 1
                            controller.step(
                                "TeleportFull",
                                x=search_pos["x"],
                                y=search_pos.get("y", 0.9),
                                z=search_pos["z"],
                                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
                                horizon=0.0,
                                standing=True,
                            )
                            nav_after, nav_visible, nav_error, nav_found = _probe_visibility(
                                controller,
                                target_type,
                                DEFAULT_PROBE_ACTIONS,
                                capture_images=True,
                                image_dir=image_dir,
                                scene_name=scene,
                                phase="nav_search",
                                action_offset=300 + len(rows) * 10 + i * 10,
                            )
                            if nav_visible and nav_found is not None:
                                visible_after_revisit = True
                                visibility_error = None
                                found_pos = nav_found
                                nav_search_used = True
                                break
                    uncertainty_inputs = _build_uncertainty_inputs(
                        target,
                        remembered_pos,
                        target_visible_after_revisit=visible_after_revisit,
                        visibility_error=visibility_error,
                        found_pos=found_pos,
                        teleport_success=teleport_ok,
                        time_interval=config.time_interval,
                        gsam_detection_history=gsam_detection_history,
                        gsam_detection_counts=gsam_detection_counts,
                        after_metadata_objects=after_metadata_objects,
                    )
                    uncertainty_estimate = _estimate_stale_uncertainty(uncertainty_inputs)
                    p_stale = uncertainty_estimate.probability
                    policy_signal_reason = uncertainty_estimate.reason
                    if config.ev_variant == "calibrated":
                        p_stale, policy_signal_reason = _apply_calibrated_ev_penalty(
                            p_stale,
                            uncertainty_inputs,
                            policy_signal_reason,
                        )
                    expected_value = _expected_verification_value(p_stale, config.stale_action_cost, config.verification_cost)
                    should_verify, policy_decision_reason = _active_verification_decision(
                        expected_value,
                        config.active_verification_threshold,
                    )
                    after_obj = (
                        _find_after_object(target, after_objects)
                        or _find_after_object(target, after_global)
                        or _find_after_object(target, after_metadata_objects)
                    )
                    oracle_stale, oracle_distance, oracle_reason = _oracle_label(target, after_obj)
                    after_position_live = _object_position(after_obj) if after_obj and _valid_position(after_obj) else None
                    true_location = rearrangement_location(scene, after_obj, "after") if after_obj else remembered
                    task = RearrangementTask(
                        target=target_type,
                        true_location=true_location,
                        old_location=remembered,
                        scene=scene,
                        direct_action_cost=0.0,
                        old_action_cost=0.0,
                        scene_search_cost=0.0,
                        rearranged=bool(oracle_stale),
                    )
                    agreement = None
                    verifier_name: str | None = None
                    decision_stale: bool | None = None
                    decision_confidence: float | None = None
                    decision_reason: str | None = None
                    decision_matched_distance: float | None = None
                    decision_detections_considered: int | None = None
                    row_idx = len(rows)
                    rows.append(
                        {
                            "row_idx": row_idx,
                            "scene": scene,
                            "seed": seed,
                            "target_idx": target_idx,
                            "target": target_type,
                            "case_list_source": str(config.case_list) if config.case_list is not None else None,
                            "case_list_matched": targeted_spec is not None,
                            "case_list_source_row_idx": targeted_spec.source_row_idx if targeted_spec is not None else None,
                            "targeted_case_reason": TARGETED_CASE_REASON if targeted_spec is not None else None,
                            "target_object_id": _object_id(target),
                            "target_pickupable": bool(target.get("pickupable")),
                            "target_openable": bool(target.get("openable")),
                            "remembered_location": remembered,
                            "true_location_offline": true_location,
                            "spawn_action": "InitialRandomSpawn",
                            "spawn_success": spawn_ok,
                            "revisit_action": revisit_fields["revisit_action"],
                            "revisit_source": "nearest_reachable_to_remembered_before_location",
                            "teleport_success": teleport_ok,
                            "used_teleportfull_for_revisit": revisit_fields["used_teleportfull_for_revisit"],
                            "revisit_path_steps": revisit_fields["revisit_path_steps"],
                            "revisit_navigation_actions": revisit_fields["revisit_navigation_actions"],
                            "revisit_navigation_failure_reason": revisit_fields["revisit_navigation_failure_reason"],
                            "target_visible_after_revisit_sweep": visible_after_revisit,
                            "visibility_error": visibility_error,
                            "found_position_after_revisit": found_pos,
                            "nav_search_used": nav_search_used,
                            "nav_search_positions_probed": nav_search_positions_probed,
                            "uncertainty_model_name": "deterministic_live_revisit_uncertainty_v1",
                            "uncertainty_inputs": asdict(uncertainty_estimate.inputs),
                            "uncertainty_reason": uncertainty_estimate.reason,
                            "active_policy_name": "expected_verification_value_v1",
                            "p_stale_estimate": round(p_stale, 4),
                            "stale_action_cost": config.stale_action_cost,
                            "verification_cost": config.verification_cost,
                            "expected_verification_value": round(expected_value, 4),
                            "policy_should_verify": should_verify,
                            "policy_reason": f"{policy_signal_reason};{policy_decision_reason}",
                            "oracle_stale_label": oracle_stale,
                            "oracle_distance": oracle_distance,
                             "oracle_reason": oracle_reason,
                             "after_object_position_live": after_position_live,
                             "selection_policy_name": config.selection_policy_name,
                            "verification_budget": config.verification_budget,
                            "candidate_rank_by_ev": None,
                            "candidate_selected_by_budget": False,
                            "budget_selection_reason": None,
                            "budget_remaining_after_selection": None,
                            "verifier_used": False,
                            "verifier_name": verifier_name,
                            "decision_stale": decision_stale,
                            "decision_confidence": decision_confidence,
                            "decision_reason": decision_reason,
                            "decision_matched_distance": decision_matched_distance,
                            "decision_detections_considered": decision_detections_considered,
                            "decision_matches_oracle": agreement,
                            "ev_variant": config.ev_variant,
                            "same_type_density_score": uncertainty_inputs.same_type_density_score,
                            "same_type_nearby_count": uncertainty_inputs.same_type_nearby_count,
                            "same_type_scene_total": uncertainty_inputs.same_type_scene_total,
                            "visual_instance_confusion_prior": uncertainty_inputs.visual_instance_confusion_prior,
                            **_default_task_execution_fields(),
                            **_default_memory_update_fields(),
                            "row_claim_boundary": CLAIM_BOUNDARY,
                        }
                    )
                    if should_verify:
                        pending_verifications[row_idx] = (task, remembered)
            except Exception as exc:  # defensive boundary for flaky Unity/controller failures
                session_errors.append(
                    {
                        "scene": scene,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "rows_before_error": len(rows),
                    }
                )
            finally:
                if controller is not None:
                    controller.stop()
        if len(rows) >= config.max_rows:
            break

    selected_for_verification = _apply_budget_selection(rows, config)
    for row in selected_for_verification:
        row_idx = _candidate_int(row, "row_idx", -1)
        task, remembered = pending_verifications[row_idx]
        if verifier is None:
            verifier = verifier_factory() if verifier_factory is not None else _build_backend(config)
        decision = verifier.verify(task, remembered_location=remembered, image_dir=image_dir, threshold=config.verifier_threshold)
        oracle_stale = row.get("oracle_stale_label")
        row["verifier_name"] = verifier.name
        row["decision_stale"] = decision.stale
        row["decision_confidence"] = decision.confidence
        row["decision_reason"] = decision.reason
        row["decision_matched_distance"] = decision.matched_distance
        row["decision_detections_considered"] = decision.detections_considered
        row["decision_matches_oracle"] = decision.stale == oracle_stale if oracle_stale is not None else None
        detections = decision.detections_considered
        instance_ambiguity = detections is not None and detections >= 20 and decision.stale is False
        row["instance_ambiguity_detected"] = instance_ambiguity
        row["instance_ambiguity_reason"] = "high_same_type_detection_density_gsam_instance_confusion" if instance_ambiguity else None
        row.update(_memory_update_fields_from_verified_row(row))
        target_type_for_history = cast(str, row.get("target", ""))
        gsam_detection_history.setdefault(target_type_for_history, []).append(bool(decision.stale))
        gsam_detection_counts.setdefault(target_type_for_history, []).append(int(decision.detections_considered or 0))
        if decision.stale is True and before_memory_store:
            scene_str = cast(str, row.get("scene", ""))
            seed_int = cast(int, row.get("seed", 0))
            memory = before_memory_store.get((scene_str, seed_int))
            target_id = cast(str, row.get("target_object_id", ""))
            after_pos = row.get("after_object_position_live") or row.get("found_position_after_revisit")
            if memory is not None and target_id:
                pos_dict: dict[str, float] | None = None
                if isinstance(after_pos, Mapping):
                    pos_dict = {k: float(cast(float, after_pos.get(k, 0.0))) for k in ("x", "y", "z")}
                mutation_evidence = _mutate_memory_object(memory, target_id, pos_dict, decision.confidence)
                row["memory_mutated"] = mutation_evidence["memory_mutated"]
                row["memory_old_position"] = mutation_evidence["memory_old_position"]
                row["memory_new_position"] = mutation_evidence["memory_new_position"]
            else:
                row["memory_mutated"] = False
                row["memory_old_position"] = None
                row["memory_new_position"] = None
        else:
            row["memory_mutated"] = False
            row["memory_old_position"] = None
            row["memory_new_position"] = None
        row.update(_execute_task_bridge_for_row(row, config=config, make_controller=make_controller))

    labelled = [r for r in rows if r.get("oracle_stale_label") is not None]
    agreements = [r for r in labelled if r.get("decision_matches_oracle") is True]
    expected_values = [float(r["expected_verification_value"]) for r in rows if r.get("expected_verification_value") is not None]
    selected_expected_values = [float(r["expected_verification_value"]) for r in rows if r.get("candidate_selected_by_budget") is True]
    unmatched_targeted_cases = [
        {
            "scene": spec.scene,
            "seed": spec.seed,
            "target": spec.target,
            "source_row_idx": spec.source_row_idx,
        }
        for spec in targeted_specs
        if _targeted_case_key(spec.scene, spec.seed, spec.target) not in matched_targeted_keys
    ]
    summary = {
        "rows": len(rows),
        "labelled_rows": len(labelled),
        "agreement_rows": len(agreements),
        "agreement_rate": round(len(agreements) / max(len(labelled), 1), 4),
        "verifier_used_rows": sum(1 for r in rows if r.get("verifier_used")),
        "policy_verified_rows": sum(1 for r in rows if r.get("policy_should_verify") is True),
        "policy_skipped_rows": sum(1 for r in rows if r.get("policy_should_verify") is False),
        "mean_expected_verification_value": round(sum(expected_values) / max(len(expected_values), 1), 4),
        "verification_budget": config.verification_budget,
        "candidate_rows": sum(1 for r in rows if r.get("policy_should_verify") is True),
        "budget_selected_rows": sum(1 for r in rows if r.get("candidate_selected_by_budget") is True),
        "budget_skipped_rows": sum(1 for r in rows if r.get("policy_should_verify") is True and r.get("candidate_selected_by_budget") is not True),
        "mean_selected_expected_verification_value": round(sum(selected_expected_values) / max(len(selected_expected_values), 1), 4),
        "memory_update_candidate_rows": sum(1 for r in rows if r.get("memory_update_action") == "propose_detector_evidence_refresh"),
        "memory_updated_rows": sum(1 for r in rows if r.get("memory_mutated") is True),
        "task_execution_evaluated_rows": sum(1 for r in rows if r.get("task_execution_evaluated") is True),
        "task_execution_not_executed_rows": sum(1 for r in rows if r.get("execute_action") == "not_executed"),
        "downstream_task_success_rows": sum(1 for r in rows if r.get("downstream_task_success") is True),
        "downstream_task_failure_rows": sum(1 for r in rows if r.get("task_execution_evaluated") is True and r.get("downstream_task_success") is False),
        "case_list_source": str(config.case_list) if config.case_list is not None else None,
        "case_list_rows": len(targeted_specs),
        "case_list_matched_rows": sum(1 for r in rows if r.get("case_list_matched") is True),
        "case_list_unmatched_rows": len(unmatched_targeted_cases),
        "case_list_unmatched_cases": unmatched_targeted_cases,
        "session_error_count": len(session_errors),
        "session_errors": session_errors,
        "controller_started": controller_started,
        "scenes": list(config.scenes),
        "seeds": list(config.seeds),
        "threshold": config.verifier_threshold,
        "nav_search_steps": config.nav_search_steps,
        "nav_search_rows": sum(1 for r in rows if r.get("nav_search_used") is True),
        "instance_ambiguity_rows": sum(1 for r in rows if r.get("instance_ambiguity_detected") is True),
        "ev_variant": config.ev_variant,
        "same_type_density_rows": sum(1 for r in rows if float(r.get("same_type_density_score") or 0.0) > 0.0),
        "visual_instance_confusion_prior_rows": sum(1 for r in rows if float(r.get("visual_instance_confusion_prior") or 0.0) > 0.0),
        "teleport_object_used": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "capability": capability.as_dict(),
        "summary": summary,
        "rows": rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_outputs(result: Json, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_gsam_closed_loop.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    rows = cast(list[Json], result.get("rows", []))
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (out_dir / "live_gsam_closed_loop.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    readme = f"# Live GSAM Closed Loop\n\nRows: {len(rows)}\n\nClaim boundary: {result.get('claim_boundary')}\n"
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> LiveGSAMClosedLoopConfig:
    parser = argparse.ArgumentParser(description="AI2-THOR live GSAM closed-loop stale-memory validation")
    parser.add_argument("--scenes", nargs="+", default=["FloorPlan1", "FloorPlan3", "FloorPlan201"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 29, 31, 37])
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--targets-per-scene-seed", type=int, default=2)
    parser.add_argument("--target-objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--platform", default="CloudRendering", dest="platform_name")
    parser.add_argument("--build-base-url", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_live_gsam_closed_loop"))
    parser.add_argument("--case-list", type=Path, default=None)
    parser.add_argument("--verifier-threshold", type=float, default=0.05)
    parser.add_argument("--verification-cost", type=float, default=0.05)
    parser.add_argument("--stale-action-cost", type=float, default=1.0)
    parser.add_argument("--active-verification-threshold", type=float, default=0.0)
    parser.add_argument("--verification-budget", type=int, default=None)
    parser.add_argument("--grounding-config", type=Path, default=None)
    parser.add_argument("--grounding-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-config", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--verifier-device", default="cuda")
    parser.add_argument("--backend-load-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--time-interval", type=float, default=1.0)
    parser.add_argument("--nav-search-steps", type=int, default=0)
    parser.add_argument("--ev-variant", choices=["current", "calibrated"], default="current",
                        help="EV computation variant: current (default) or calibrated (applies ambiguity density penalty)")
    parser.add_argument("--revisit-mode", choices=["teleportfull", "stepwise"], default="teleportfull",
                        help="Revisit mechanism: default TeleportFull shortcut or opt-in bounded stepwise navigation instrumentation")
    parser.add_argument("--execute-task-bridge", action="store_true", help="Opt-in Wave 4 bridge: execute one PickupObject/OpenObject attempt after verified memory update")
    parser.add_argument("--honest-interaction", action="store_true", help="Use forceAction=False with a deterministic face-then-pick step so simulator proximity/visibility preconditions apply")
    parser.add_argument("--rich-before-probe", action="store_true", help="Collect the before-state memory with a horizon-sweep probe (LookDown passes) instead of the default five-action sweep")
    parser.add_argument("--max-alternate-poses", type=int, default=0, help="Bounded approach-pose fallback: after a failed honest pickup, try up to N further reachable poses nearest to the refreshed location")
    ns = parser.parse_args(argv)
    return LiveGSAMClosedLoopConfig(
        scenes=tuple(str(s) for s in ns.scenes),
        seeds=tuple(int(s) for s in ns.seeds),
        max_rows=int(ns.max_rows),
        targets_per_scene_seed=int(ns.targets_per_scene_seed),
        target_objects=tuple(str(s) for s in ns.target_objects),
        width=int(ns.width),
        height=int(ns.height),
        platform_name=str(ns.platform_name),
        build_base_url=ns.build_base_url if isinstance(ns.build_base_url, str) else None,
        out_dir=ns.out_dir,
        case_list=ns.case_list,
        verifier_threshold=float(ns.verifier_threshold),
        verification_cost=float(ns.verification_cost),
        stale_action_cost=float(ns.stale_action_cost),
        active_verification_threshold=float(ns.active_verification_threshold),
        verification_budget=ns.verification_budget if ns.verification_budget is None else int(ns.verification_budget),
        grounding_config=ns.grounding_config,
        grounding_checkpoint=ns.grounding_checkpoint,
        sam2_config=ns.sam2_config,
        sam2_checkpoint=ns.sam2_checkpoint,
        verifier_device=str(ns.verifier_device),
        backend_load_timeout_seconds=float(ns.backend_load_timeout_seconds),
        time_interval=float(ns.time_interval),
        nav_search_steps=int(ns.nav_search_steps),
        ev_variant=str(ns.ev_variant),
        revisit_mode=str(ns.revisit_mode),
        execute_task_bridge=bool(ns.execute_task_bridge),
        honest_interaction=bool(ns.honest_interaction),
        before_probe_actions=RICH_BEFORE_PROBE_ACTIONS if bool(ns.rich_before_probe) else None,
        max_alternate_poses=int(ns.max_alternate_poses),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    result = run_live_gsam_closed_loop(config)
    write_outputs(result, config.out_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
