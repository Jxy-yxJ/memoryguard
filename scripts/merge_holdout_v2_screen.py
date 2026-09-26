"""Merge the held-out v2 screening artifacts and re-freeze the round-robin case list.

Outcome-free amendment step: combines the original v2 screen (15 scenes x 5 seeds) with the
pre-arm seed extension on the eight qualifying combinations (5 further seeds), then freezes the
round-robin K=24 list over the merged candidate rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodied_memory_pilot.ai2thor_paired_hard_challenge_screen import (  # noqa: E402
    CASE_LIST_SCHEMA_VERSION,
    SCREENING_CLAIM_BOUNDARY,
    freeze_case_list_round_robin,
)
BASE = ROOT / "results/0514_paired_holdout_v2_screen/paired_hard_challenge_screen.json"
EXTENSION = ROOT / "results/0514_paired_holdout_v2_screen_extension/paired_hard_challenge_screen.json"
MERGED = ROOT / "results/0514_paired_holdout_v2_screen/paired_hard_challenge_screen_merged.json"
CASE_LIST = ROOT / "results/0514_paired_holdout_v2_screen/case_list_paired_hard_challenge_frozen_v2.json"
K = 24


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = _load(BASE)
    extension = _load(EXTENSION)

    merged_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*base["candidate_rows"], *extension["candidate_rows"]]:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged_rows.append(row)

    frozen = freeze_case_list_round_robin(merged_rows, K)
    status_counts = Counter(str(row.get("status")) for row in merged_rows)
    summary = {
        "candidates": len(merged_rows),
        "spawned": sum(1 for row in merged_rows if row.get("after_object_id") is not None),
        "qualifying": status_counts.get("qualifies", 0),
        "frozen_cases": len(frozen),
        "shortfall": len(frozen) < K,
        "freeze_mode": "round_robin",
        "status_counts": dict(status_counts),
        "sources": {
            "base": str(BASE.relative_to(ROOT)),
            "extension": str(EXTENSION.relative_to(ROOT)),
            "base_sha256": _sha256(BASE),
            "extension_sha256": _sha256(EXTENSION),
        },
    }
    merged = {
        "schema": base.get("schema"),
        "status": "ok",
        "date": base.get("date"),
        "capability": base.get("capability"),
        "config": base.get("config"),
        "summary": summary,
        "candidate_rows": merged_rows,
        "frozen_case_list": frozen,
        "claim_boundary": SCREENING_CLAIM_BOUNDARY,
        "amendment": "pre-arm seed extension on the eight qualifying combinations; outcome-free",
    }
    MERGED.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")

    case_list = {
        "schema": CASE_LIST_SCHEMA_VERSION,
        "date": base.get("date"),
        "case_count": len(frozen),
        "cases": frozen,
        "screening_source": str(MERGED.relative_to(ROOT)),
        "amendment": "merged v2 screen plus pre-arm seed extension; frozen before any arm run",
        "claim_boundary": SCREENING_CLAIM_BOUNDARY,
    }
    CASE_LIST.write_text(json.dumps(case_list, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("case_list_sha256", _sha256(CASE_LIST))


if __name__ == "__main__":
    main()
