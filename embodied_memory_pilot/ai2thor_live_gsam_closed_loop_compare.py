from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "ai2thor_live_gsam_closed_loop_compare.v1"
CLAIM_BOUNDARY = (
    "CPU-only recorded-row comparison of existing live_gsam_closed_loop artifacts; "
    "no new live experiment result; no AI2-THOR, GSAM, Habitat, navigation, manipulation, "
    "ObjectNav, task success, persistent memory repair/writeback, recovery-search, or "
    "Habitat object-level transfer; oracle labels are evaluation-only and never used for "
    "policy inclusion or ranking"
)
Json = dict[str, object]
Selector = Callable[[Mapping[str, object]], bool]

VARIANT_ORDER = (
    "passive_no_verification",
    "recorded_ev_gate",
    "recorded_budgeted_ev_gate",
)


def _json_default(value: object) -> str:
    return str(value)


def _load_artifact(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as handle:
        data = cast(object, json.load(handle))
    if not isinstance(data, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    data_items = cast(Mapping[object, object], data).items()
    return {str(key): value for key, value in data_items}


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
        row_items = cast(Mapping[object, object], raw_row).items()
        row: Json = {str(key): value for key, value in row_items}
        row["_source_file"] = source_path
        row["_source_row_offset"] = idx
        rows.append(row)
    return rows


def _is_true(value: object) -> bool:
    return value is True


def _select_passive_no_verification(_row: Mapping[str, object]) -> bool:
    return False


def _select_recorded_ev_gate(row: Mapping[str, object]) -> bool:
    if "policy_should_verify" in row:
        return _is_true(row.get("policy_should_verify"))
    return _is_true(row.get("verifier_used"))


def _select_recorded_budgeted_ev_gate(row: Mapping[str, object]) -> bool:
    if "candidate_selected_by_budget" in row:
        return _is_true(row.get("candidate_selected_by_budget"))
    return _is_true(row.get("verifier_used"))


def _variant_selectors() -> dict[str, Selector]:
    return {
        "passive_no_verification": _select_passive_no_verification,
        "recorded_ev_gate": _select_recorded_ev_gate,
        "recorded_budgeted_ev_gate": _select_recorded_budgeted_ev_gate,
    }


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _memory_update_is_proposed(row: Mapping[str, object]) -> bool:
    return row.get("memory_update_action") == "propose_detector_evidence_refresh"


def _detector_outcome_counts(rows: Sequence[Mapping[str, object]]) -> Json:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for row in rows:
        decision = row.get("decision_stale")
        label = row.get("oracle_stale_label")
        if not isinstance(decision, bool) or not isinstance(label, bool):
            continue
        if decision and label:
            true_positive += 1
        elif decision and not label:
            false_positive += 1
        elif not decision and not label:
            true_negative += 1
        else:
            false_negative += 1

    labelled_count = true_positive + false_positive + true_negative + false_negative
    false_negative_rate = false_negative / labelled_count if labelled_count else None
    false_positive_rate = false_positive / labelled_count if labelled_count else None
    return {
        "true_positive_rows_among_selected": true_positive,
        "false_positive_rows_among_selected": false_positive,
        "true_negative_rows_among_selected": true_negative,
        "false_negative_rows_among_selected": false_negative,
        "false_negative_rate_among_labelled_selected": _round_or_none(false_negative_rate),
        "false_positive_rate_among_labelled_selected": _round_or_none(false_positive_rate),
    }


def summarize_variant(variant: str, rows: Sequence[Mapping[str, object]], selector: Selector) -> Json:
    selected = [row for row in rows if selector(row)]
    labelled = [row for row in selected if row.get("oracle_stale_label") is not None]
    agreement = [row for row in selected if row.get("decision_matches_oracle") is True]
    expected_values = [value for row in selected if (value := _float_or_none(row.get("expected_verification_value"))) is not None]
    agreement_rate = len(agreement) / len(labelled) if labelled else None
    mean_expected_value = sum(expected_values) / len(expected_values) if expected_values else None
    summary: Json = {
        "variant": variant,
        "selected_rows": len(selected),
        "labelled_rows_among_selected": len(labelled),
        "agreement_rows_among_selected": len(agreement),
        "agreement_rows": len(agreement),
        "agreement_rate": _round_or_none(agreement_rate),
        "stale_decision_rows": sum(1 for row in selected if row.get("decision_stale") is True),
        "oracle_stale_rows_among_selected_labels": sum(1 for row in labelled if row.get("oracle_stale_label") is True),
        "proposed_update_rows": sum(1 for row in selected if _memory_update_is_proposed(row)),
        "memory_updated_rows": 0,
        "mean_expected_verification_value": _round_or_none(mean_expected_value),
    }
    summary.update(_detector_outcome_counts(selected))
    return summary


def compare_loaded_artifacts(artifacts: Sequence[tuple[str, Mapping[str, object]]]) -> Json:
    rows: list[Json] = []
    input_files: list[str] = []
    source_schemas: list[str | None] = []
    source_statuses: list[str | None] = []
    for source_path, data in artifacts:
        input_files.append(source_path)
        schema_version = data.get("schema_version")
        status = data.get("status")
        source_schemas.append(str(schema_version) if schema_version is not None else None)
        source_statuses.append(str(status) if status is not None else None)
        rows.extend(_rows_from_artifact(data, source_path))

    selectors = _variant_selectors()
    summary_variants = [summarize_variant(name, rows, selectors[name]) for name in VARIANT_ORDER]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "input_files": input_files,
        "input_rows": len(rows),
        "source_schema_versions": source_schemas,
        "source_statuses": source_statuses,
        "claim_boundary": CLAIM_BOUNDARY,
        "summary_variants": summary_variants,
    }


def compare_artifacts(paths: Sequence[Path]) -> Json:
    artifacts = [(str(path), _load_artifact(path)) for path in paths]
    return compare_loaded_artifacts(artifacts)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "variant",
            "selected_rows",
            "labelled_rows_among_selected",
            "agreement_rows_among_selected",
            "agreement_rows",
            "agreement_rate",
            "stale_decision_rows",
            "oracle_stale_rows_among_selected_labels",
            "true_positive_rows_among_selected",
            "false_positive_rows_among_selected",
            "true_negative_rows_among_selected",
            "false_negative_rows_among_selected",
            "false_negative_rate_among_labelled_selected",
            "false_positive_rate_among_labelled_selected",
            "proposed_update_rows",
            "memory_updated_rows",
            "mean_expected_verification_value",
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mapping_sequence(value: object) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[Mapping[str, object]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(cast(Mapping[str, object], item))
    return rows


def _object_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return value


def render_readme(result: Mapping[str, object]) -> str:
    variants = _mapping_sequence(result.get("summary_variants"))
    lines = [
        "# Live GSAM Closed Loop Compare",
        "",
        f"Schema: `{result.get('schema_version')}`",
        f"Status: `{result.get('status')}`",
        f"Input files: {len(_object_sequence(result.get('input_files')))}",
        f"Input rows: {result.get('input_rows')}",
        "",
        f"Claim boundary: {result.get('claim_boundary')}",
        "",
        "This is a CPU-only recorded-row comparison. It does not launch AI2-THOR, GSAM, Habitat, navigation, memory writeback, or GPU workloads.",
        "Oracle labels are used only for evaluation metrics, never for policy inclusion or ranking.",
        "",
        "## Variants",
    ]
    for row in variants:
        variant_line = (
            f"- `{row.get('variant')}`: selected={row.get('selected_rows')}, "
            f"agreement_rate={row.get('agreement_rate')}, "
            f"memory_updated_rows={row.get('memory_updated_rows')}"
        )
        lines.append(variant_line)
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: Mapping[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = (out_dir / "live_gsam_closed_loop_compare.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_csv(out_dir / "live_gsam_closed_loop_compare_summary.csv", _mapping_sequence(result.get("summary_variants")))
    _ = (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only recorded-row comparison for live GSAM closed-loop artifacts"
    )
    _ = parser.add_argument("json_paths", nargs="+", type=Path, help="Existing live_gsam_closed_loop.json-style artifacts")
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_live_gsam_closed_loop_compare"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = compare_artifacts(cast(Sequence[Path], args.json_paths))
    write_outputs(result, cast(Path, args.out_dir))
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
