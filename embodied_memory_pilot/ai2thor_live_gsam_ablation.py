from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "ai2thor_live_gsam_ablation.v1"
CLAIM_BOUNDARY = (
    "CPU-only simulated-budget post-hoc ablation of recorded AI2-THOR controller-backed "
    "GSAM detector artifact; oracle labels evaluation-only and never used for budget "
    "selection; no new live experiment; no AI2-THOR/GSAM/Habitat/GPU; not task success/"
    "full navigation/manipulation/ObjectNav/recovery-search/memory writeback"
)
Json = dict[str, object]


def _json_default(value: object) -> str:
    return str(value)


def _load_artifact(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as handle:
        data = cast(object, json.load(handle))
    if not isinstance(data, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    data_items = cast(Mapping[object, object], data).items()
    return {str(key): value for key, value in data_items}


def load_live_rows(path: Path) -> list[Json]:
    data = _load_artifact(path)
    raw_rows = data.get("rows")
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, list):
        raise ValueError(f"artifact rows must be a list: {path}")
    rows: list[Json] = []
    for idx, raw_row in enumerate(cast(list[object], raw_rows)):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"artifact row {idx} must be a JSON object: {path}")
        row_items = cast(Mapping[object, object], raw_row).items()
        row: Json = {str(key): value for key, value in row_items}
        row["_source_file"] = str(path)
        row["_source_row_offset"] = idx
        rows.append(row)
    return rows


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


def _detector_outcome_counts(rows: Sequence[Mapping[str, object]]) -> Json:
    tp = fp = tn = fn = 0
    for row in rows:
        decision = row.get("decision_stale")
        label = row.get("oracle_stale_label")
        if not isinstance(decision, bool) or not isinstance(label, bool):
            continue
        if decision and label:
            tp += 1
        elif decision and not label:
            fp += 1
        elif not decision and not label:
            tn += 1
        else:
            fn += 1
    labelled = tp + fp + tn + fn
    return {
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "true_negative_rows": tn,
        "false_negative_rows": fn,
        "false_negative_rate": _round_or_none(fn / labelled if labelled else None),
        "false_positive_rate": _round_or_none(fp / labelled if labelled else None),
    }


def _strategy_metrics(selected: list[Json], strategy: str, budget: int | str) -> Json:
    labelled = [r for r in selected if r.get("oracle_stale_label") is not None]
    agreement = [r for r in selected if r.get("decision_matches_oracle") is True]
    ev_values = [
        v for r in selected
        if (v := _float_or_none(r.get("expected_verification_value"))) is not None
    ]
    agreement_rate = len(agreement) / len(labelled) if labelled else None
    mean_ev = sum(ev_values) / len(ev_values) if ev_values else None
    summary: Json = {
        "strategy": strategy,
        "budget": budget if isinstance(budget, str) else int(budget),
        "selected_rows": len(selected),
        "labelled_rows": len(labelled),
        "agreement_rows": len(agreement),
        "agreement_rate": _round_or_none(agreement_rate),
        "stale_decision_rows": sum(1 for r in selected if r.get("decision_stale") is True),
        "oracle_stale_rows": sum(1 for r in labelled if r.get("oracle_stale_label") is True),
        "mean_expected_verification_value": _round_or_none(mean_ev),
        "memory_updated_rows": 0,
    }
    summary.update(_detector_outcome_counts(selected))
    return summary


def simulate_ev_budgeted(rows: list[Json], budget: int) -> Json:
    sorted_rows = sorted(
        rows,
        key=lambda r: _float_or_none(r.get("expected_verification_value")) or float("-inf"),
        reverse=True,
    )
    selected = sorted_rows[:budget] if budget < len(sorted_rows) else sorted_rows
    return _strategy_metrics(selected, "ev_budgeted", budget)


def simulate_random_budgeted(rows: list[Json], budget: int, seed: int = 0) -> Json:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected = shuffled[:budget] if budget < len(shuffled) else shuffled
    return _strategy_metrics(selected, "random_budgeted", budget)


def simulate_all_verify(rows: list[Json]) -> Json:
    return _strategy_metrics(list(rows), "all_verify", "all")


def run_ablation(
    rows: list[Json],
    budgets: list[int],
    num_random_trials: int = 5,
) -> Json:
    all_verify_result = simulate_all_verify(rows)

    budget_results: list[Json] = []
    for budget in budgets:
        ev_result = simulate_ev_budgeted(rows, budget)

        random_trials: list[Json] = []
        for trial_seed in range(num_random_trials):
            random_trials.append(simulate_random_budgeted(rows, budget, seed=trial_seed))

        avg_random: Json = {}
        for key in ev_result:
            ev_val = ev_result[key]
            if key in ("strategy", "budget"):
                continue
            trial_vals = [t.get(key, 0) for t in random_trials]
            if all(isinstance(v, (int, float)) for v in trial_vals if v is not None):
                numeric = [cast(float, v) for v in trial_vals if v is not None]
                avg_random[key] = _round_or_none(sum(numeric) / len(numeric)) if numeric else None
            else:
                avg_random[key] = trial_vals[0]

        avg_random["strategy"] = "random_budgeted_avg"
        avg_random["budget"] = budget
        avg_random["random_trials"] = num_random_trials

        budget_results.append(ev_result)
        budget_results.append(avg_random)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "input_rows": len(rows),
        "budgets": budgets,
        "num_random_trials": num_random_trials,
        "claim_boundary": CLAIM_BOUNDARY,
        "all_verify": all_verify_result,
        "budget_comparisons": budget_results,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    union_fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                union_fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): v for k, v in row.items()})


def write_outputs(result: Json, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "live_gsam_ablation.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default, ensure_ascii=False)

    csv_rows: list[Json] = []
    all_verify = cast(Json, result.get("all_verify", {}))
    csv_rows.append(all_verify)
    for comp in cast(list[Json], result.get("budget_comparisons", [])):
        csv_rows.append(comp)
    _write_csv(out_dir / "live_gsam_ablation.csv", csv_rows)

    readme = (
        f"# Live GSAM Ablation Analysis\n\n"
        f"**Schema**: {SCHEMA_VERSION}\n"
        f"**Input rows**: {result.get('input_rows')}\n"
        f"**Budgets**: {result.get('budgets')}\n"
        f"**Random trials per budget**: {result.get('num_random_trials')}\n\n"
        f"## Claim Boundary\n\n{CLAIM_BOUNDARY}\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only live GSAM verification ablation")
    parser.add_argument("artifact", type=Path, help="Path to live_gsam_closed_loop.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                        help="Budget levels (default: 1 2 3 5 10)")
    parser.add_argument("--random-trials", type=int, default=5,
                        help="Number of random trials per budget (default: 5)")
    args = parser.parse_args()

    rows = load_live_rows(args.artifact)
    result = run_ablation(rows, args.budgets, args.random_trials)
    write_outputs(result, args.out_dir)
    print(f"Wrote ablation to {args.out_dir}")


if __name__ == "__main__":
    main()
