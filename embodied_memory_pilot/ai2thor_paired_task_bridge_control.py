from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "ai2thor_paired_task_bridge_control.v1"
CLAIM_BOUNDARY = (
    "recorded/prepared passive-vs-active paired-control artifact derived from recorded active closed-loop "
    "controller-backed GSAM stale-memory detection artifact; passive rows are derived/synthetic and do NOT "
    "represent a live passive controller execution, only the recorded active evidence from the source "
    "artifact; active rows are recorded as-is from the source active closed-loop run; no new live experiment "
    "result; no AI2-THOR, GSAM, Habitat, navigation, manipulation, ObjectNav/SPL, or GPU workloads; "
    "not a manipulation benchmark; not cross-platform transfer (Habitat/Warehouse/ObjectNav); "
    "not persistent memory writeback or repair; not broad active-maintenance generalization; "
    "passive task-execution fields reflect whether downstream task was evaluated or only prepared/skipped"
)
LIVE_PASSIVE_CLAIM_BOUNDARY = (
    "live_passive paired-vs-active paired-control artifact; passive rows represent evaluable live controller "
    "execution derived from recorded active cases using stale memory_old_position without verifier evidence "
    "and without memory mutation; active rows are recorded as-is from source active closed-loop run; "
    "uses AI2-THOR controller only for the passive arm; no GSAM verifier, GPU verifier, Habitat, "
    "manipulation, ObjectNav/SPL, or persistent memory workload; "
    "not a manipulation benchmark; not cross-platform transfer (Habitat/Warehouse/ObjectNav); "
    "not persistent memory writeback or repair; not broad active-maintenance generalization; "
    "not passive-vs-active superiority claim; fixed-case evaluability only"
)
Json = dict[str, object]

def _stable_case_id(scene: str, seed: int, target: str) -> str:
    raw = f"{scene}:{seed}:{target}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]




def _json_default(value: object) -> str:
    return str(value)


def load_active_artifact(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as handle:
        data = cast(object, json.load(handle))
    if not isinstance(data, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return {str(key): value for key, value in cast(Mapping[str, object], data).items()}


def _rows_from_artifact(data: Mapping[str, object], source_path: str) -> list[Json]:
    raw_rows = data.get("rows")
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, list):
        raise ValueError(f"artifact rows must be a list: {source_path}")
    rows: list[Json] = []
    for idx, raw_row in enumerate(cast(list[object], raw_rows)):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"artifact row {idx} must be a JSON object: {source_path}")
        row: Json = {str(key): value for key, value in cast(Mapping[str, object], raw_row).items()}
        row["_source_file"] = source_path
        row["_source_row_offset"] = idx
        rows.append(row)
    return rows


def _safe_str(value: object) -> str:
    return str(value) if value is not None else ""


def _safe_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (str, float)):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    return 0


def _pick_field(row: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _build_active_pair_row(row: Mapping[str, object]) -> Json:
    scene = _safe_str(_pick_field(row, "scene"))
    seed = _safe_int(_pick_field(row, "seed"))
    target = _safe_str(_pick_field(row, "target"))
    case_id = _stable_case_id(scene, seed, target)

    return {
        "case_id": case_id,
        "arm": "active",
        "scene": scene,
        "seed": seed,
        "target": target,
        "target_object_id": _safe_str(_pick_field(row, "target_object_id")),
        "target_pickupable": bool(row.get("target_pickupable")),
        "target_openable": bool(row.get("target_openable")),
        "remembered_location": _safe_str(_pick_field(row, "remembered_location")),
        "verifier_used": bool(row.get("verifier_used", False)),
        "verifier_name": _safe_str(_pick_field(row, "verifier_name")),
        "decision_stale": row.get("decision_stale"),
        "decision_confidence": row.get("decision_confidence"),
        "decision_reason": _safe_str(_pick_field(row, "decision_reason")),
        "decision_matches_oracle": row.get("decision_matches_oracle"),
        "oracle_stale_label": row.get("oracle_stale_label"),
        "memory_mutated": bool(row.get("memory_mutated", False)),
        "memory_old_position": row.get("memory_old_position"),
        "memory_new_position": row.get("memory_new_position"),
        "memory_update_action": _safe_str(_pick_field(row, "memory_update_action")),
        "task_execution_evaluated": bool(row.get("task_execution_evaluated", False)),
        "downstream_task_success": row.get("downstream_task_success"),
        "execute_action": _safe_str(_pick_field(row, "execute_action")),
        "execution_boundary": _safe_str(_pick_field(row, "execution_boundary")),
        "bridge_mode": _safe_str(_pick_field(row, "task_bridge_navigation_mode", "revisit_action")),
        "used_teleportfull_for_revisit": bool(row.get("used_teleportfull_for_revisit", False)),
        "task_bridge_used_teleportfull": bool(row.get("task_bridge_used_teleportfull", False)),
        "revisit_path_steps": _safe_int(_pick_field(row, "revisit_path_steps")),
        "task_bridge_path_steps": _safe_int(_pick_field(row, "task_bridge_path_steps")),
        "interaction_attempt_mode": _safe_str(_pick_field(row, "interaction_attempt_mode")),
        "interaction_used_force_action": row.get("interaction_used_force_action"),
        "interaction_agent_object_distance": row.get("interaction_agent_object_distance"),
        "interaction_target_visible": row.get("interaction_target_visible"),
        "interaction_horizon_deg": row.get("interaction_horizon_deg"),
        "interaction_visibility_sweep_used": row.get("interaction_visibility_sweep_used"),
        "source_active_artifact": _safe_str(_pick_field(row, "_source_file")),
        "source_row_offset": _safe_int(_pick_field(row, "_source_row_offset")),
        "derived_from": "recorded_active_closed_loop_row",
        "row_type": "recorded_active",
    }


def _build_passive_pair_row(row: Mapping[str, object]) -> Json:
    scene = _safe_str(_pick_field(row, "scene"))
    seed = _safe_int(_pick_field(row, "seed"))
    target = _safe_str(_pick_field(row, "target"))
    case_id = _stable_case_id(scene, seed, target)

    active_task_evaluated = bool(row.get("task_execution_evaluated", False))
    if active_task_evaluated:
        passive_task_evaluated = False
        passive_downstream_success = None
        passive_execute_action = "not_executed"
        passive_execution_boundary = (
            "prepared_passive_control_reference_only; "
            "no passive controller execution; stale memory retained; "
            "downstream task not attempted on passive arm"
        )
    else:
        passive_task_evaluated = False
        passive_downstream_success = None
        passive_execute_action = _safe_str(_pick_field(row, "execute_action"))
        passive_execution_boundary = _safe_str(_pick_field(row, "execution_boundary"))

    return {
        "case_id": case_id,
        "arm": "passive",
        "scene": scene,
        "seed": seed,
        "target": target,
        "target_object_id": _safe_str(_pick_field(row, "target_object_id")),
        "target_pickupable": bool(row.get("target_pickupable")),
        "target_openable": bool(row.get("target_openable")),
        "remembered_location": _safe_str(_pick_field(row, "remembered_location")),
        "verifier_used": False,
        "verifier_name": "none_passive_arm",
        "decision_stale": None,
        "decision_confidence": None,
        "decision_reason": "passive_arm_no_verifier_executed",
        "decision_matches_oracle": None,
        "oracle_stale_label": row.get("oracle_stale_label"),
        "memory_mutated": False,
        "memory_old_position": row.get("memory_old_position") or row.get("memory_new_position"),
        "memory_new_position": row.get("memory_old_position"),
        "memory_update_action": "none_passive_arm",
        "task_execution_evaluated": passive_task_evaluated,
        "downstream_task_success": passive_downstream_success,
        "execute_action": passive_execute_action,
        "execution_boundary": passive_execution_boundary,
        "bridge_mode": "passive_no_bridge",
        "used_teleportfull_for_revisit": False,
        "task_bridge_used_teleportfull": False,
        "revisit_path_steps": 0,
        "task_bridge_path_steps": 0,
        "source_active_artifact": _safe_str(_pick_field(row, "_source_file")),
        "source_row_offset": _safe_int(_pick_field(row, "_source_row_offset")),
        "derived_from": "derived_from_recorded_active_row_no_live_passive_execution",
        "row_type": "derived_passive",
    }


def _build_live_passive_pair_row(
    active_row: Mapping[str, object],
    passive_task_fields: Mapping[str, object],
) -> Json:
    scene = _safe_str(_pick_field(active_row, "scene"))
    seed = _safe_int(_pick_field(active_row, "seed"))
    target = _safe_str(_pick_field(active_row, "target"))
    case_id = _stable_case_id(scene, seed, target)

    return {
        "case_id": case_id,
        "arm": "passive",
        "scene": scene,
        "seed": seed,
        "target": target,
        "target_object_id": _safe_str(_pick_field(active_row, "target_object_id")),
        "target_pickupable": bool(active_row.get("target_pickupable")),
        "target_openable": bool(active_row.get("target_openable")),
        "remembered_location": _safe_str(_pick_field(active_row, "remembered_location")),
        "verifier_used": False,
        "verifier_name": "none_passive_arm",
        "decision_stale": None,
        "decision_confidence": None,
        "decision_reason": "live_passive_arm_no_verifier_executed",
        "decision_matches_oracle": None,
        "oracle_stale_label": active_row.get("oracle_stale_label"),
        "memory_mutated": False,
        "memory_old_position": active_row.get("memory_old_position"),
        "memory_new_position": active_row.get("memory_old_position"),
        "memory_update_action": "none_live_passive_arm",
        "task_execution_evaluated": bool(passive_task_fields.get("task_execution_evaluated", False)),
        "downstream_task_success": passive_task_fields.get("downstream_task_success"),
        "execute_action": _safe_str(_pick_field(passive_task_fields, "execute_action")),
        "execution_boundary": _safe_str(_pick_field(passive_task_fields, "execution_boundary")),
        "task_execution_failure_reason": passive_task_fields.get("task_execution_failure_reason"),
        "bridge_mode": _safe_str(_pick_field(passive_task_fields, "task_bridge_navigation_mode")),
        "used_teleportfull_for_revisit": False,
        "task_bridge_used_teleportfull": bool(passive_task_fields.get("task_bridge_used_teleportfull", False)),
        "revisit_path_steps": 0,
        "task_bridge_path_steps": _safe_int(_pick_field(passive_task_fields, "task_bridge_path_steps")),
        "task_bridge_navigation_actions": passive_task_fields.get("task_bridge_navigation_actions", []),
        "task_bridge_navigation_failure_reason": passive_task_fields.get("task_bridge_navigation_failure_reason"),
        "interaction_attempt_mode": _safe_str(_pick_field(passive_task_fields, "interaction_attempt_mode")),
        "interaction_used_force_action": passive_task_fields.get("interaction_used_force_action"),
        "interaction_agent_object_distance": passive_task_fields.get("interaction_agent_object_distance"),
        "interaction_target_visible": passive_task_fields.get("interaction_target_visible"),
        "interaction_horizon_deg": passive_task_fields.get("interaction_horizon_deg"),
        "interaction_visibility_sweep_used": passive_task_fields.get("interaction_visibility_sweep_used"),
        "source_active_artifact": _safe_str(_pick_field(active_row, "_source_file")),
        "source_row_offset": _safe_int(_pick_field(active_row, "_source_row_offset")),
        "derived_from": "live_passive_controller_from_recorded_active_case",
        "row_type": "live_passive",
    }


def build_pairs(active_rows: Sequence[Mapping[str, object]]) -> list[Json]:
    paired: list[Json] = []
    for row in active_rows:
        paired.append(_build_active_pair_row(row))
        paired.append(_build_passive_pair_row(row))
    return paired


def _bool_count(rows: Sequence[Mapping[str, object]], key: str) -> int:
    return sum(1 for r in rows if r.get(key) is True)


def summarize_pairs(paired_rows: Sequence[Mapping[str, object]], source_active_artifact: str = "") -> Json:
    active_rows = [r for r in paired_rows if r.get("arm") == "active"]
    passive_rows = [r for r in paired_rows if r.get("arm") == "passive"]

    n_pairs = len(active_rows)
    active_successes = _bool_count(active_rows, "downstream_task_success")
    passive_successes = _bool_count(passive_rows, "downstream_task_success")

    passive_stale_failures = sum(
        1 for r in passive_rows
        if r.get("oracle_stale_label") is True
        and r.get("task_execution_evaluated") is True
        and r.get("downstream_task_success") is False
    )
    passive_prepared_stale_rows = sum(
        1 for r in passive_rows
        if r.get("oracle_stale_label") is True
        and r.get("task_execution_evaluated") is False
        and r.get("verifier_used") is False
        and r.get("memory_mutated") is False
    )

    active_task_evaluated = _bool_count(active_rows, "task_execution_evaluated")
    passive_task_evaluated = _bool_count(passive_rows, "task_execution_evaluated")
    passive_prepared_rows = sum(
        1 for r in passive_rows
        if r.get("task_execution_evaluated") is False and r.get("execute_action") == "not_executed"
    )

    passive_all_evaluated = n_pairs > 0 and passive_task_evaluated == n_pairs

    passive_live_rows = sum(1 for r in passive_rows if r.get("row_type") == "live_passive")
    passive_all_live = n_pairs > 0 and passive_live_rows == n_pairs and passive_all_evaluated
    paired_delta: int | None = active_successes - passive_successes if passive_all_evaluated else None

    passive_is_live_execution = passive_all_live
    passive_derivation_method = (
        "live_passive_controller_from_recorded_active_cases"
        if passive_all_live
        else "derived_from_recorded_active_rows_no_live_passive_controller"
    )
    claim_boundary_used = LIVE_PASSIVE_CLAIM_BOUNDARY if passive_all_live else CLAIM_BOUNDARY

    return {
        "schema_version": SCHEMA_VERSION,
        "n_pairs": n_pairs,
        "active_successes": active_successes,
        "passive_successes": passive_successes,
        "paired_delta": paired_delta,
        "paired_delta_evaluable": passive_all_evaluated,
        "paired_delta_not_evaluable_reason": (
            "passive_arm_not_live_executed_or_not_all_task_evaluated"
            if not passive_all_evaluated else ""
        ),
        "passive_stale_failures": passive_stale_failures,
        "passive_prepared_stale_rows": passive_prepared_stale_rows,
        "active_task_evaluated": active_task_evaluated,
        "passive_task_evaluated": passive_task_evaluated,
        "passive_prepared_rows": passive_prepared_rows,
        "passive_live_rows": passive_live_rows,
        "source_active_artifact": source_active_artifact,
        "passive_is_live_execution": passive_is_live_execution,
        "passive_derivation_method": passive_derivation_method,
        "claim_boundary": claim_boundary_used,
    }


REQUIRED_PAIR_FIELDS = [
    "case_id", "arm", "scene", "seed", "target",
    "target_object_id", "target_pickupable", "target_openable",
    "remembered_location",
    "verifier_used", "verifier_name",
    "decision_stale", "decision_confidence", "decision_reason",
    "decision_matches_oracle", "oracle_stale_label",
    "memory_mutated", "memory_old_position", "memory_new_position",
    "memory_update_action",
    "task_execution_evaluated", "downstream_task_success",
    "execute_action", "execution_boundary",
    "bridge_mode", "used_teleportfull_for_revisit",
    "task_bridge_used_teleportfull",
    "revisit_path_steps", "task_bridge_path_steps",
    "source_active_artifact", "source_row_offset",
    "derived_from", "row_type",
]

REQUIRED_SUMMARY_FIELDS = [
    "schema_version", "n_pairs", "active_successes",
    "passive_successes", "paired_delta", "paired_delta_evaluable",
    "paired_delta_not_evaluable_reason", "passive_stale_failures",
    "passive_prepared_stale_rows", "passive_live_rows",
    "active_task_evaluated", "passive_task_evaluated",
    "passive_prepared_rows", "source_active_artifact",
    "passive_is_live_execution", "passive_derivation_method",
    "claim_boundary",
]

FORBIDDEN_POSITIVE_CLAIM_TERMS = [
    "active-maintenance generalization",
    "ObjectNav",
    "SPL",
    "manipulation benchmark",
    "cross-platform transfer",
    "persistent writeback",
    "persistent memory repair",
    "Habitat transfer",
]


def _term_in_positive_context(term: str, boundary: str) -> bool:
    boundary_lower = boundary.lower()
    term_lower = term.lower()
    if term_lower not in boundary_lower:
        return False
    for clause in boundary_lower.split(";"):
        clause = clause.strip()
        if term_lower not in clause:
            continue
        before_term = clause.split(term_lower)[0].strip()
        ends_with_not = before_term.endswith("not ") or before_term.endswith("not")
        starts_negation = clause.startswith("no ") or clause.startswith("not ")
        if ends_with_not or starts_negation:
            continue
        return True
    return False


def _check_schema(pairs: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> Json:
    violations: list[str] = []

    for idx, row in enumerate(pairs):
        for field in REQUIRED_PAIR_FIELDS:
            if field not in row:
                violations.append(f"pair_row[{idx}]: missing required field '{field}'")
        if row.get("arm") == "active" and row.get("verifier_used") is not True:
            violations.append(f"pair_row[{idx}]: active arm must have verifier_used=true")
        if row.get("arm") == "passive":
            if row.get("verifier_used") is not False:
                violations.append(f"pair_row[{idx}]: passive arm must have verifier_used=false")
            if row.get("memory_mutated") is not False:
                violations.append(f"pair_row[{idx}]: passive arm must have memory_mutated=false")

    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            violations.append(f"summary: missing required field '{field}'")

    boundary = str(summary.get("claim_boundary", ""))
    for term in FORBIDDEN_POSITIVE_CLAIM_TERMS:
        if _term_in_positive_context(term, boundary):
            violations.append(f"claim_boundary positively asserts forbidden term: '{term}'")

    n_pairs = len([r for r in pairs if r.get("arm") == "active"])
    passive_count = len([r for r in pairs if r.get("arm") == "passive"])
    if n_pairs != passive_count:
        violations.append(f"unbalanced pairs: active={n_pairs} passive={passive_count}")
    if summary.get("n_pairs") != n_pairs:
        violations.append(f"summary n_pairs={summary.get('n_pairs')} != active rows={n_pairs}")

    for idx, row in enumerate(pairs):
        if row.get("arm") == "passive" and row.get("derived_from") == "recorded_active_closed_loop_row":
            violations.append(f"pair_row[{idx}]: passive row incorrectly claims recorded_active ancestry")

    ok = len(violations) == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_ok": ok,
        "n_violations": len(violations),
        "violations": violations,
        "checked_fields": len(REQUIRED_PAIR_FIELDS) + len(REQUIRED_SUMMARY_FIELDS),
    }


def build_paired_control(
    source_artifact_path: Path,
    *,
    source_label: str | None = None,
) -> Json:
    source_label = source_label or str(source_artifact_path)
    data = load_active_artifact(source_artifact_path)
    source_rows = _rows_from_artifact(data, source_label)

    if not source_rows:
        raise ValueError(f"no rows found in source artifact: {source_artifact_path}")

    paired_rows = build_pairs(source_rows)
    summary = summarize_pairs(paired_rows, source_active_artifact=source_label)
    schema_check = _check_schema(paired_rows, summary)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_active_artifact": source_label,
        "paired_rows": paired_rows,
        "summary": summary,
        "schema_check": schema_check,
    }


def build_live_passive_paired_control(
    source_artifact_path: Path,
    *,
    source_label: str | None = None,
    case_list_path: Path | None = None,
    revisit_mode: str = "stepwise",
    width: int = 300,
    height: int = 300,
    platform_name: str = "CloudRendering",
    controller_factory: object | None = None,
    honest_interaction: bool = False,
) -> Json:
    from embodied_memory_pilot.ai2thor_live_gsam_closed_loop import (
        ControllerFactory,
        LiveGSAMClosedLoopConfig,
        _controller_factory,
        _execute_passive_task_bridge_for_row,
    )

    if revisit_mode != "stepwise":
        raise ValueError("live-passive paired control currently supports only stepwise revisit_mode")

    source_label = source_label or str(source_artifact_path)
    data = load_active_artifact(source_artifact_path)
    source_rows = _rows_from_artifact(data, source_label)

    if not source_rows:
        raise ValueError(f"no rows found in source artifact: {source_artifact_path}")

    config = LiveGSAMClosedLoopConfig(
        width=width,
        height=height,
        platform_name=platform_name,
        case_list=case_list_path,
        revisit_mode=revisit_mode,
        execute_task_bridge=True,
        honest_interaction=honest_interaction,
    )
    make_controller = cast(ControllerFactory, controller_factory or _controller_factory)

    paired_rows: list[Json] = []
    for row in source_rows:
        active = _build_active_pair_row(row)
        paired_rows.append(active)

        passive_task_fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=make_controller,
        )
        passive = _build_live_passive_pair_row(row, passive_task_fields)
        paired_rows.append(passive)

    summary = summarize_pairs(paired_rows, source_active_artifact=source_label)
    schema_check = _check_schema(paired_rows, summary)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": LIVE_PASSIVE_CLAIM_BOUNDARY,
        "source_active_artifact": source_label,
        "case_list_path": str(case_list_path) if case_list_path is not None else None,
        "paired_rows": paired_rows,
        "summary": summary,
        "schema_check": schema_check,
    }


def _csv_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        return REQUIRED_PAIR_FIELDS
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    return fieldnames


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        _ = writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summary: Mapping[str, object]) -> None:
    fieldnames = [k for k in REQUIRED_SUMMARY_FIELDS if k in summary]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        _ = writer.writeheader()
        _ = writer.writerow(summary)


def _render_readme(result: Mapping[str, object]) -> str:
    summary = cast(Mapping[str, object], result.get("summary", {}))
    sc = cast(Mapping[str, object], result.get("schema_check", {}))
    lines = [
        "# Paired Passive-vs-Active Task Bridge Control",
        "",
        f"Schema: `{result.get('schema_version')}`",
        f"Status: `{result.get('status')}`",
        f"Source active artifact: `{result.get('source_active_artifact')}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| n_pairs | {summary.get('n_pairs')} |",
        f"| active_successes | {summary.get('active_successes')} |",
        f"| passive_successes | {summary.get('passive_successes')} |",
        f"| paired_delta | {summary.get('paired_delta')} |",
        f"| paired_delta_evaluable | {summary.get('paired_delta_evaluable')} |",
        f"| paired_delta_not_evaluable_reason | `{summary.get('paired_delta_not_evaluable_reason')}` |",
        f"| passive_stale_failures | {summary.get('passive_stale_failures')} |",
        f"| passive_prepared_stale_rows | {summary.get('passive_prepared_stale_rows')} |",
        f"| active_task_evaluated | {summary.get('active_task_evaluated')} |",
        f"| passive_task_evaluated | {summary.get('passive_task_evaluated')} |",
        f"| passive_prepared_rows | {summary.get('passive_prepared_rows')} |",
        f"| passive_live_rows | {summary.get('passive_live_rows', 0)} |",
        f"| passive_is_live_execution | {summary.get('passive_is_live_execution')} |",
        f"| passive_derivation_method | `{summary.get('passive_derivation_method')}` |",
        "",
        "## Claim Boundary",
        "",
        f"> {result.get('claim_boundary', '')}",
        "",
        "## Schema Check",
        "",
        f"Schema ok: `{sc.get('schema_ok')}`",
        f"Violations: {sc.get('n_violations')}",
        "",
    ]
    if str(result.get('claim_boundary', '')).startswith('live_passive'):
        lines += [
            "This is a live-passive paired-control artifact. Passive rows are evaluable live controller",
            "executions derived from recorded active cases using stale memory_old_position.",
            "The passive arm uses an AI2-THOR controller only; no verifier evidence, GSAM/GPU verifier,",
            "external services, or memory mutation are used for the passive arm.",
        ]
    else:
        lines += [
            "This is a prepared/recorded paired-control artifact. Passive rows are derived from recorded active rows.",
            "No live passive controller was executed. No AI2-THOR, GSAM, GPU, or external services were used.",
        ]
    return "\n".join(lines)


def write_outputs(result: Mapping[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = cast(list[Json], result.get("paired_rows", []))
    summary = cast(Json, result.get("summary", {}))
    schema_check = cast(Json, result.get("schema_check", {}))

    _ = (out_dir / "paired_task_bridge_control.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_csv(out_dir / "paired_task_bridge_control.csv", paired_rows)
    _write_summary_csv(out_dir / "paired_task_bridge_control_summary.csv", summary)
    _ = (out_dir / "README.md").write_text(_render_readme(result), encoding="utf-8")
    _ = (out_dir / "schema_check.json").write_text(
        json.dumps(schema_check, indent=2, default=_json_default), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired passive-vs-active task bridge control runner (recorded/prepared or live-passive evidence)"
    )
    parser.add_argument(
        "source_artifact",
        type=Path,
        default=Path("results/ai2thor_live_gsam_mixed_challenge_budgetfix_v1/live_gsam_closed_loop.json"),
        nargs="?",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/ai2thor_paired_task_bridge_control"),
    )
    parser.add_argument("--source-label", type=str, default=None)
    parser.add_argument(
        "--live-passive",
        action="store_true",
        help="Build live-passive paired control with evaluable passive rows (opt-in)",
    )
    parser.add_argument("--case-list", type=Path, default=None)
    parser.add_argument(
        "--revisit-mode",
        type=str,
        choices=["teleportfull", "stepwise"],
        default="stepwise",
        help="Revisit mode for live-passive passive arm (default: stepwise)",
    )
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--platform", type=str, default="CloudRendering", dest="platform_name")
    parser.add_argument(
        "--honest-interaction",
        action="store_true",
        help="Use forceAction=False with a deterministic face-then-pick step for the live passive arm",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_artifact = cast(Path, args.source_artifact)
    source_label = cast(str | None, args.source_label)
    out_dir = cast(Path, args.out_dir)
    if cast(bool, args.live_passive):
        result = build_live_passive_paired_control(
            source_artifact,
            source_label=source_label or str(source_artifact),
            case_list_path=cast(Path | None, args.case_list),
            revisit_mode=str(args.revisit_mode),
            width=int(args.width),
            height=int(args.height),
            platform_name=str(args.platform_name),
            honest_interaction=bool(args.honest_interaction),
        )
    else:
        result = build_paired_control(
            source_artifact,
            source_label=source_label or str(source_artifact),
        )
    write_outputs(result, out_dir)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
