from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "ai2thor_live_gsam_failure_slices.v1"
CLAIM_BOUNDARY = (
    "CPU-only post-processing of a previously recorded AI2-THOR controller-backed "
    "GSAM detector artifact; oracle labels are evaluation-only detector outcome labels; "
    "no new live run; no AI2-THOR, GSAM, Habitat, GPU, navigation, manipulation, "
    "ObjectNav, task success, recovery-search success, live policy superiority, "
    "no memory writeback/repair, and no persistent memory update"
)

Json = dict[str, object]
OUTCOME_ORDER = ("tp", "fp", "tn", "fn", "unlabelled")
GROUP_FIELDS = (
    "scene",
    "target",
    "visibility_error",
    "target_visible_after_revisit_sweep",
    "decision_reason",
    "candidate_selected_by_budget",
)
CSV_FIELDS = (
    "outcome",
    "field",
    "value",
    "rows",
)


def _json_default(value: object) -> str:
    return str(value)


def detector_outcome(row: Mapping[str, object]) -> str:
    decision = row.get("decision_stale")
    label = row.get("oracle_stale_label")
    if not isinstance(decision, bool) or not isinstance(label, bool):
        return "unlabelled"
    if decision and label:
        return "tp"
    if decision and not label:
        return "fp"
    if not decision and not label:
        return "tn"
    return "fn"


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


def _safe_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=_json_default)


def _count_by_field(rows: Sequence[Mapping[str, object]], field: str) -> list[Json]:
    counts: Counter[str] = Counter(_safe_value(row.get(field)) for row in rows)
    return [
        {"field": field, "value": value, "rows": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _slice_rows_for_outcome(outcome: str, rows: Sequence[Mapping[str, object]]) -> list[Json]:
    selected = [row for row in rows if detector_outcome(row) == outcome]
    return [
        {"outcome": outcome, "field": field, "value": item["value"], "rows": item["rows"]}
        for field in GROUP_FIELDS
        for item in _count_by_field(selected, field)
    ]


def _outcome_counts(rows: Sequence[Mapping[str, object]]) -> Json:
    counts = Counter(detector_outcome(row) for row in rows)
    return {outcome: counts.get(outcome, 0) for outcome in OUTCOME_ORDER}


def summarize_failure_slices(rows: Sequence[Mapping[str, object]]) -> Json:
    row_list = list(rows)
    outcome_counts = _outcome_counts(row_list)
    false_negative_rows = [row for row in row_list if detector_outcome(row) == "fn"]
    false_positive_rows = [row for row in row_list if detector_outcome(row) == "fp"]
    labelled_rows = sum(cast(int, outcome_counts[outcome]) for outcome in ("tp", "fp", "tn", "fn"))
    slice_rows = [slice_row for outcome in OUTCOME_ORDER for slice_row in _slice_rows_for_outcome(outcome, row_list)]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": CLAIM_BOUNDARY,
        "input_rows": len(row_list),
        "labelled_rows": labelled_rows,
        "memory_updated_rows": 0,
        "group_fields": list(GROUP_FIELDS),
        "outcome_counts": outcome_counts,
        "false_negative_slices": {field: _count_by_field(false_negative_rows, field) for field in GROUP_FIELDS},
        "false_positive_slices": {field: _count_by_field(false_positive_rows, field) for field in GROUP_FIELDS},
        "outcome_slice_rows": slice_rows,
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
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def render_readme(result: Mapping[str, object]) -> str:
    outcome_counts = result.get("outcome_counts")
    counts: Mapping[str, object] = (
        cast(Mapping[str, object], outcome_counts) if isinstance(outcome_counts, Mapping) else {}
    )
    lines = [
        "# Live GSAM Failure Slices",
        "",
        f"Schema: `{result.get('schema_version')}`",
        f"Status: `{result.get('status')}`",
        f"Input rows: {result.get('input_rows')}",
        f"Labelled rows: {result.get('labelled_rows')}",
        f"Memory updated rows: {result.get('memory_updated_rows')}",
        "",
        f"Claim boundary: {result.get('claim_boundary')}",
        "",
        "This is CPU-only post-processing of recorded detector rows. It does not launch AI2-THOR, GSAM, Habitat, GPU workloads, navigation, manipulation, recovery search, or memory writeback.",
        "Oracle labels are used only as post-hoc evaluation labels, never as policy inputs or ranking features.",
        "",
        "## Detector Outcomes",
    ]
    for outcome in OUTCOME_ORDER:
        lines.append(f"- `{outcome}`: {counts.get(outcome, 0)}")
    lines.append("")
    lines.append("## Grouped Recorded Fields")
    for field in GROUP_FIELDS:
        lines.append(f"- `{field}`")
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: Mapping[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = (out_dir / "live_gsam_failure_slices.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_csv(out_dir / "live_gsam_failure_slices.csv", _mapping_sequence(result.get("outcome_slice_rows")))
    _ = (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only detector outcome slice summaries for recorded live GSAM artifacts"
    )
    _ = parser.add_argument("artifact", type=Path, help="Existing live_gsam_closed_loop.json-style artifact")
    _ = parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_live_gsam_failure_slices"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_live_rows(cast(Path, args.artifact))
    result = summarize_failure_slices(rows)
    write_outputs(result, cast(Path, args.out_dir))
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
