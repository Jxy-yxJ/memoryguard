"""Analyze the pre-registered discriminative paired passive-vs-active challenge.

Reads the frozen case list, the active closed-loop artifact, and the live-passive paired
artifact; produces exact paired counts, McNemar exact statistics, and the pre-registered
decision outcome. The decision-rule interpretation below was fixed before the passive-arm
outcomes were read.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

Json = dict[str, Any]


def _load(path: Path) -> Json:
    return cast(Json, json.loads(path.read_text(encoding="utf-8")))


def _case_key(scene: str, target: str, seed: int) -> str:
    return f"{scene}|{target}|{seed}"


def _mcnemar_exact_two_sided(active_only: int, passive_only: int) -> float:
    n = active_only + passive_only
    if n == 0:
        return 1.0
    k = min(active_only, passive_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def analyze(active_path: Path, paired_path: Path, case_list_path: Path) -> Json:
    active = _load(active_path)
    paired = _load(paired_path)
    case_list = _load(case_list_path)

    active_rows = {_case_key(str(r.get("scene")), str(r.get("target")), int(r.get("seed", 0))): r for r in active.get("rows", [])}
    frozen_meta = {
        _case_key(str(c.get("scene")), str(c.get("target")), int(c.get("seed", 0))): c for c in case_list.get("cases", [])
    }

    by_case: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in paired.get("paired_rows", []):
        key = str(row.get("case_id"))
        by_case.setdefault(key, {})[str(row.get("arm"))] = row

    pairs: list[Json] = []
    for key in sorted(by_case):
        arms = by_case[key]
        active_row = arms.get("active")
        passive_row = arms.get("passive")
        if active_row is None or passive_row is None:
            continue
        active_evaluated = bool(active_row.get("task_execution_evaluated"))
        passive_evaluated = bool(passive_row.get("task_execution_evaluated"))
        active_success = active_row.get("downstream_task_success")
        passive_success = passive_row.get("downstream_task_success")
        geo_key = _case_key(str(active_row.get("scene")), str(active_row.get("target")), int(cast(int, active_row.get("seed", 0))))
        active_source = active_rows.get(geo_key, {})
        meta = frozen_meta.get(geo_key, {})
        pairs.append(
            {
                "case_id": key,
                "scene": active_row.get("scene"),
                "target": active_row.get("target"),
                "seed": active_row.get("seed"),
                "d_passive": meta.get("d_passive"),
                "d_active": meta.get("d_active"),
                "active_evaluated": active_evaluated,
                "passive_evaluated": passive_evaluated,
                "active_success": active_success,
                "passive_success": passive_success,
                "active_failure_reason": active_source.get("task_execution_failure_reason"),
                "passive_failure_reason": passive_row.get("task_execution_failure_reason"),
                "active_interaction_distance": active_source.get("interaction_agent_object_distance"),
                "active_interaction_visible": active_source.get("interaction_target_visible"),
                "active_interaction_horizon": active_source.get("interaction_horizon_deg"),
                "active_memory_mutated": active_row.get("memory_mutated"),
                "active_decision_stale": active_row.get("decision_stale"),
            }
        )

    evaluable = [p for p in pairs if p["active_evaluated"] and p["passive_evaluated"]]
    n_evaluable = len(evaluable)
    both_success = sum(1 for p in evaluable if p["active_success"] is True and p["passive_success"] is True)
    both_fail = sum(1 for p in evaluable if p["active_success"] is False and p["passive_success"] is False)
    active_only = sum(1 for p in evaluable if p["active_success"] is True and p["passive_success"] is not True)
    passive_only = sum(1 for p in evaluable if p["active_success"] is not True and p["passive_success"] is True)
    active_success_count = sum(1 for p in evaluable if p["active_success"] is True)
    passive_success_count = sum(1 for p in evaluable if p["passive_success"] is True)
    active_not_evaluated = [p for p in pairs if not p["active_evaluated"]]

    support_max = max(1, n_evaluable // 8)
    non_discriminative_min = (n_evaluable + 1) // 2
    if n_evaluable == 0:
        decision = "not_evaluable"
    elif passive_success_count <= support_max and active_success_count > passive_success_count:
        decision = "discriminative_support"
    elif passive_success_count >= non_discriminative_min:
        decision = "non_discriminative"
    else:
        decision = "partial"

    amendment_path = case_list_path.parent / "freeze_amendment_all_36_v1.json"
    amendment = _load(amendment_path) if amendment_path.exists() else None

    return {
        "schema": "0514_paired_hard_challenge_result_to_claim.v1",
        "date": "2026-09-19",
        "source_active_artifact": str(active_path),
        "source_paired_artifact": str(paired_path),
        "source_case_list": str(case_list_path),
        "freeze_provenance": {
            "original_screen_frozen_cases": 8,
            "effective_frozen_cases": len(case_list.get("cases", [])),
            "amendment_record": str(amendment_path),
            "pre_outcome_amendment_validated": bool(
                amendment and amendment.get("pre_outcome_amendment_validated") is True
            ),
            "effective_case_list_sha256": amendment.get("effective_case_list_sha256") if amendment else None,
        },
        "frozen_cases": len(case_list.get("cases", [])),
        "paired_cases": len(pairs),
        "decision_rule": {
            "support": "active_success > passive_success AND passive_success <= max(1, n_evaluable // 8)",
            "non_discriminative": "passive_success >= ceil(n_evaluable / 2)",
            "partial": "otherwise",
            "note": "Proportional interpretation of the pre-registered 1/8 and 4/8 thresholds, fixed before passive-arm outcomes were read.",
        },
        "summary": {
            "n_evaluable_pairs": n_evaluable,
            "active_success": active_success_count,
            "passive_success": passive_success_count,
            "both_success": both_success,
            "both_fail": both_fail,
            "active_only_success": active_only,
            "passive_only_success": passive_only,
            "mcnemar_exact_two_sided_p": f"{_mcnemar_exact_two_sided(active_only, passive_only):.3e}",
            "active_not_evaluated": len(active_not_evaluated),
            "active_not_evaluated_cases": [p["case_id"] for p in active_not_evaluated],
            "decision": decision,
            "support_max_passive_success": support_max,
            "non_discriminative_min_passive_success": non_discriminative_min,
        },
        "failure_reason_counts": {
            "active": dict(Counter(str(p["active_failure_reason"]) for p in pairs if p["active_evaluated"])),
            "passive": dict(Counter(str(p["passive_failure_reason"]) for p in pairs if p["passive_evaluated"])),
        },
        "pairs": pairs,
        "claim_boundary": (
            "fixed-challenge paired mechanism evidence on a pre-registered geometry-screened case set; "
            "no broad scale, no statistical robustness beyond the exact paired test, no ObjectNav/SPL, "
            "no manipulation benchmark, no persistent writeback, no cross-platform transfer"
        ),
    }


def render_markdown(result: Json) -> str:
    s = cast(Mapping[str, object], result["summary"])
    lines = [
        "# Paired Hard Challenge Result-to-Claim",
        "",
        f"Date: {result['date']}",
        "",
        f"Frozen cases: {result['frozen_cases']} | Evaluable pairs: {s['n_evaluable_pairs']}",
        "",
        "## Primary endpoint (paired downstream PickupObject success)",
        "",
        f"- Active success: {s['active_success']}",
        f"- Passive success: {s['passive_success']}",
        f"- Both success: {s['both_success']} | Both fail: {s['both_fail']}",
        f"- Active-only success: {s['active_only_success']} | Passive-only success: {s['passive_only_success']}",
        f"- McNemar exact two-sided p: {s['mcnemar_exact_two_sided_p']}",
        f"- Active arm not evaluated (detector false negatives): {s['active_not_evaluated']} {s['active_not_evaluated_cases']}",
        "",
        f"## Pre-registered decision: **{s['decision']}**",
        "",
        f"Support threshold: passive success <= {s['support_max_passive_success']}; "
        f"non-discriminative threshold: passive success >= {s['non_discriminative_min_passive_success']}.",
        "",
        "## Boundary",
        "",
        str(result["claim_boundary"]),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze the pre-registered paired passive-vs-active challenge")
    parser.add_argument("--active", type=Path, default=Path("results/0514_paired_hard_challenge_v1/active/live_gsam_closed_loop.json"))
    parser.add_argument("--paired", type=Path, default=Path("results/0514_paired_hard_challenge_v1/paired/paired_task_bridge_control.json"))
    parser.add_argument("--case-list", type=Path, default=Path("results/0514_paired_hard_challenge_screen_v1/case_list_paired_hard_challenge_all_qualifying_v1.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/0514_paired_hard_challenge_v1"))
    args = parser.parse_args(argv)

    result = analyze(cast(Path, args.active), cast(Path, args.paired), cast(Path, args.case_list))
    out_dir = cast(Path, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = (out_dir / "paired_hard_challenge_result_to_claim.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    _ = (out_dir / "paired_hard_challenge_result_to_claim.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
