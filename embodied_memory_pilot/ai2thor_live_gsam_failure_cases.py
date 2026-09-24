from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "ai2thor_live_gsam_failure_cases.v1"
CLAIM_BOUNDARY = (
    "CPU-only materialization of qualitative false-negative cases from a previously recorded "
    "AI2-THOR controller-backed GSAM detector artifact; oracle labels post-hoc only; "
    "not new live run/navigation/manipulation/ObjectNav/task success/recovery-search/"
    "policy superiority/Habitat transfer/memory writeback/memory repair"
)

Json = dict[str, object]
CASE_FIELDS = (
    "row_idx",
    "scene",
    "seed",
    "target",
    "visibility_error",
    "target_visible_after_revisit_sweep",
    "decision_reason",
    "decision_matched_distance",
    "decision_detections_considered",
    "expected_verification_value",
    "p_stale_estimate",
    "candidate_rank_by_ev",
    "candidate_selected_by_budget",
    "oracle_stale_label",
    "decision_stale",
)


def _json_default(value: object) -> str:
    return str(value)


def _load_artifact(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as handle:
        data = cast(object, json.load(handle))
    if not isinstance(data, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return {str(key): value for key, value in cast(Mapping[object, object], data).items()}


def load_live_rows(path: Path) -> list[Json]:
    data = _load_artifact(path)
    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"artifact rows must be a list: {path}")

    rows: list[Json] = []
    for idx, raw_row in enumerate(cast(list[object], raw_rows)):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"artifact row {idx} must be a JSON object: {path}")
        row = {str(key): value for key, value in cast(Mapping[object, object], raw_row).items()}
        row["_source_file"] = str(path)
        row["_source_row_offset"] = idx
        rows.append(row)
    return rows


def is_false_negative(row: Mapping[str, object]) -> bool:
    return row.get("decision_stale") is False and row.get("oracle_stale_label") is True


def _fallback_row_idx(row: Mapping[str, object], fallback: int) -> object:
    if row.get("row_idx") is not None:
        return row.get("row_idx")
    if row.get("_source_row_offset") is not None:
        return row.get("_source_row_offset")
    return fallback


def _sort_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=_json_default)


def _sort_number(value: object, fallback: int) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float, str)):
        return (0, value)
    return (1, fallback)


def _case_sort_key(item: tuple[int, Mapping[str, object]]) -> tuple[int, str, str, tuple[int, object]]:
    fallback_idx, row = item
    has_visibility_error = 0 if row.get("visibility_error") is not None else 1
    row_idx = _fallback_row_idx(row, fallback_idx)
    return (
        has_visibility_error,
        _sort_text(row.get("scene")),
        _sort_text(row.get("target")),
        _sort_number(row_idx, fallback_idx),
    )


def _compact_case(row: Mapping[str, object], fallback_idx: int) -> Json:
    case = {field: row.get(field) for field in CASE_FIELDS}
    case["row_idx"] = _fallback_row_idx(row, fallback_idx)
    return case


def materialize_failure_cases(
    rows: Sequence[Mapping[str, object]], *, max_cases: int | None = None
) -> Json:
    if max_cases is not None and max_cases < 0:
        raise ValueError("max_cases must be non-negative or None")

    row_list = list(rows)
    indexed_false_negatives = [(idx, row) for idx, row in enumerate(row_list) if is_false_negative(row)]
    ranked = sorted(indexed_false_negatives, key=_case_sort_key)
    selected = ranked if max_cases is None else ranked[:max_cases]
    cases = [_compact_case(row, idx) for idx, row in selected]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": CLAIM_BOUNDARY,
        "input_rows": len(row_list),
        "false_negative_rows": len(indexed_false_negatives),
        "case_rows": len(cases),
        "max_cases": max_cases,
        "memory_updated_rows": 0,
        "selection_rule": "decision_stale is False and oracle_stale_label is True",
        "ranking_rule": (
            "Within the false-negative subset only, rank by recorded visibility_error presence, "
            "scene, target, and row_idx/source offset; oracle labels are not used after subset selection."
        ),
        "case_fields": list(CASE_FIELDS),
        "cases": cases,
    }


def _mapping_sequence(value: object) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[Mapping[str, object]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(cast(Mapping[str, object], item))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CASE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CASE_FIELDS})


def _markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def render_readme(result: Mapping[str, object]) -> str:
    lines = [
        "# Live GSAM False-Negative Failure Cases",
        "",
        f"Schema: `{result.get('schema_version')}`",
        f"Status: `{result.get('status')}`",
        f"Input rows: {result.get('input_rows')}",
        f"False-negative rows: {result.get('false_negative_rows')}",
        f"Case rows: {result.get('case_rows')}",
        f"Memory updated rows: {result.get('memory_updated_rows')}",
        "",
        f"Claim boundary: {result.get('claim_boundary')}",
        "",
        "This table is CPU-only materialization from a previously recorded artifact and is not new live evidence.",
        "Oracle labels are used only to select recorded false negatives, then not used for within-subset ranking.",
        "The outputs do not launch AI2-THOR, GSAM, Habitat, GPU workloads, navigation, manipulation, recovery search, or memory writeback.",
        "",
        "## Cases",
    ]
    rows = _mapping_sequence(result.get("cases"))
    if not rows:
        lines.append("")
        lines.append("No false-negative cases were selected.")
        lines.append("")
        return "\n".join(lines)

    header = "| " + " | ".join(CASE_FIELDS) + " |"
    separator = "| " + " | ".join("---" for _ in CASE_FIELDS) + " |"
    lines.extend([header, separator])
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(field)) for field in CASE_FIELDS) + " |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: Mapping[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = (out_dir / "live_gsam_failure_cases.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_csv(out_dir / "live_gsam_failure_cases.csv", _mapping_sequence(result.get("cases")))
    _ = (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only qualitative false-negative case table for recorded live GSAM artifacts"
    )
    _ = parser.add_argument("artifact", type=Path, help="Existing live_gsam_closed_loop.json-style artifact")
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_live_gsam_failure_cases"))
    _ = parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_live_rows(cast(Path, args.artifact))
    result = materialize_failure_cases(rows, max_cases=cast(int | None, args.max_cases))
    write_outputs(result, cast(Path, args.out_dir))
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
