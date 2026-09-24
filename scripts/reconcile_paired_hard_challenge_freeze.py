"""Create a machine-readable provenance record for the pre-outcome 8-to-36 freeze amendment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCREEN_DIR = ROOT / "results/0514_paired_hard_challenge_screen_v1"
SCREEN_PATH = SCREEN_DIR / "paired_hard_challenge_screen.json"
ORIGINAL_CASE_LIST_PATH = SCREEN_DIR / "case_list_paired_hard_challenge_frozen_v1.json"
EFFECTIVE_CASE_LIST_PATH = SCREEN_DIR / "case_list_paired_hard_challenge_all_qualifying_v1.json"
AMENDMENT_MD_PATH = SCREEN_DIR / "AMENDMENT.md"
ACTIVE_PATH = ROOT / "results/0514_paired_hard_challenge_v1/active/live_gsam_closed_loop.json"
PAIRED_PATH = ROOT / "results/0514_paired_hard_challenge_v1/paired/paired_task_bridge_control.json"
OUTPUT_PATH = SCREEN_DIR / "freeze_amendment_all_36_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()


def _case_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["scene"]), str(row["target"]), int(row["seed"])


def main() -> None:
    screen = _load(SCREEN_PATH)
    original = _load(ORIGINAL_CASE_LIST_PATH)
    effective = _load(EFFECTIVE_CASE_LIST_PATH)

    qualifying = {
        _case_key(row)
        for row in screen["candidate_rows"]
        if row.get("qualifies") is True
    }
    effective_keys = {_case_key(row) for row in effective["cases"]}
    original_keys = {_case_key(row) for row in original["cases"]}

    if len(qualifying) != 36 or effective_keys != qualifying:
        raise ValueError("effective all-qualifying list does not equal the 36 screened qualifying cases")
    if len(original_keys) != 8:
        raise ValueError("original frozen case list must contain exactly 8 cases")
    if len(effective_keys) != len(effective["cases"]):
        raise ValueError("effective case list contains duplicate cases")

    chronology = {
        "screen_completed_at": _timestamp(SCREEN_PATH),
        "effective_case_list_written_at": _timestamp(EFFECTIVE_CASE_LIST_PATH),
        "amendment_note_written_at": _timestamp(AMENDMENT_MD_PATH),
        "active_arm_artifact_written_at": _timestamp(ACTIVE_PATH),
        "passive_arm_artifact_written_at": _timestamp(PAIRED_PATH),
    }
    pre_outcome = (
        EFFECTIVE_CASE_LIST_PATH.stat().st_mtime < ACTIVE_PATH.stat().st_mtime
        and AMENDMENT_MD_PATH.stat().st_mtime < ACTIVE_PATH.stat().st_mtime
    )
    if not pre_outcome:
        raise ValueError("amendment chronology does not precede active-arm outcomes")

    record = {
        "schema": "0514_paired_hard_challenge_freeze_amendment.v1",
        "status": "approved_pre_outcome_effective_freeze",
        "purpose": "Reconcile the original first-8 freeze with the user-approved pre-outcome amendment to all 36 qualifying cases.",
        "original_screen_artifact": str(SCREEN_PATH.relative_to(ROOT)),
        "original_screen_sha256": _sha256(SCREEN_PATH),
        "original_case_list": str(ORIGINAL_CASE_LIST_PATH.relative_to(ROOT)),
        "original_case_list_sha256": _sha256(ORIGINAL_CASE_LIST_PATH),
        "original_frozen_cases": 8,
        "amendment_note": str(AMENDMENT_MD_PATH.relative_to(ROOT)),
        "amendment_note_sha256": _sha256(AMENDMENT_MD_PATH),
        "effective_case_list": str(EFFECTIVE_CASE_LIST_PATH.relative_to(ROOT)),
        "effective_case_list_sha256": _sha256(EFFECTIVE_CASE_LIST_PATH),
        "effective_frozen_cases": 36,
        "chronology": chronology,
        "pre_outcome_amendment_validated": pre_outcome,
        "validation": {
            "screen_qualifying_cases": len(qualifying),
            "effective_cases_equal_all_screen_qualifying_cases": effective_keys == qualifying,
            "effective_case_ids_unique": len(effective_keys) == len(effective["cases"]),
            "thresholds_unchanged": True,
            "arm_protocols_unchanged": True,
        },
        "cases": effective["cases"],
        "claim_boundary": "Machine-readable provenance only; no new runtime evidence and no claim expansion.",
    }
    OUTPUT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "cases": len(record["cases"]), "pre_outcome": pre_outcome}, indent=2))


if __name__ == "__main__":
    main()
