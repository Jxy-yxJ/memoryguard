from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from embodied_memory_pilot.ai2thor_live_gsam_closed_loop_compare import (
    CLAIM_BOUNDARY,
    compare_artifacts,
    compare_loaded_artifacts,
    write_outputs,
)


class LiveGSAMClosedLoopCompareTest(unittest.TestCase):
    def _artifact(self, rows: list[dict[str, object]], *, schema_version: str = "ai2thor_live_gsam_closed_loop.v1") -> dict[str, object]:
        return {
            "schema_version": schema_version,
            "status": "ok",
            "claim_boundary": "source live boundary",
            "summary": {"rows": len(rows)},
            "rows": rows,
        }

    def test_old_artifact_fallback_uses_verifier_used_for_recorded_policies(self) -> None:
        artifact = self._artifact(
            [
                {
                    "row_idx": 0,
                    "verifier_used": True,
                    "decision_stale": True,
                    "decision_matches_oracle": True,
                    "oracle_stale_label": True,
                },
                {
                    "row_idx": 1,
                    "verifier_used": False,
                    "decision_stale": False,
                    "decision_matches_oracle": False,
                    "oracle_stale_label": True,
                },
            ]
        )

        result = compare_loaded_artifacts([("old.json", artifact)])
        variants = _variants_by_name(result)

        self.assertEqual(variants["passive_no_verification"]["selected_rows"], 0)
        self.assertEqual(variants["recorded_ev_gate"]["selected_rows"], 1)
        self.assertEqual(variants["recorded_budgeted_ev_gate"]["selected_rows"], 1)
        self.assertEqual(variants["recorded_ev_gate"]["agreement_rows"], 1)
        self.assertEqual(variants["recorded_ev_gate"]["agreement_rate"], 1.0)

    def test_budgeted_selection_metrics_and_expected_value_mean(self) -> None:
        artifact = self._artifact(
            [
                {
                    "row_idx": 0,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": False,
                    "expected_verification_value": 0.8,
                    "decision_stale": True,
                    "decision_matches_oracle": True,
                    "oracle_stale_label": True,
                    "memory_update_action": "propose_detector_evidence_refresh",
                },
                {
                    "row_idx": 1,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "expected_verification_value": 0.4,
                    "decision_stale": False,
                    "decision_matches_oracle": False,
                    "oracle_stale_label": True,
                    "memory_update_action": "none",
                },
                {
                    "row_idx": 2,
                    "policy_should_verify": False,
                    "candidate_selected_by_budget": True,
                    "expected_verification_value": 0.2,
                    "decision_stale": True,
                    "decision_matches_oracle": True,
                    "oracle_stale_label": False,
                    "memory_update_action": "propose_detector_evidence_refresh",
                },
            ]
        )

        result = compare_loaded_artifacts([("new.json", artifact)])
        variants = _variants_by_name(result)
        ev_gate = variants["recorded_ev_gate"]
        budgeted = variants["recorded_budgeted_ev_gate"]

        self.assertEqual(ev_gate["selected_rows"], 2)
        self.assertEqual(ev_gate["labelled_rows_among_selected"], 2)
        self.assertEqual(ev_gate["agreement_rows_among_selected"], 1)
        self.assertEqual(ev_gate["stale_decision_rows"], 1)
        self.assertEqual(ev_gate["oracle_stale_rows_among_selected_labels"], 2)
        self.assertEqual(ev_gate["proposed_update_rows"], 1)
        self.assertEqual(ev_gate["mean_expected_verification_value"], 0.6)
        self.assertEqual(budgeted["selected_rows"], 2)
        self.assertEqual(budgeted["agreement_rows_among_selected"], 1)
        self.assertEqual(budgeted["mean_expected_verification_value"], 0.3)

    def test_detector_outcome_counts_for_selected_labelled_rows(self) -> None:
        artifact = self._artifact(
            [
                {
                    "row_idx": 0,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "decision_stale": True,
                    "oracle_stale_label": True,
                },
                {
                    "row_idx": 1,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "decision_stale": True,
                    "oracle_stale_label": False,
                },
                {
                    "row_idx": 2,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "decision_stale": False,
                    "oracle_stale_label": False,
                },
                {
                    "row_idx": 3,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "decision_stale": False,
                    "oracle_stale_label": True,
                },
                {
                    "row_idx": 4,
                    "policy_should_verify": False,
                    "candidate_selected_by_budget": False,
                    "decision_stale": False,
                    "oracle_stale_label": True,
                },
            ]
        )

        result = compare_loaded_artifacts([("outcomes.json", artifact)])
        ev_gate = _variants_by_name(result)["recorded_ev_gate"]

        self.assertEqual(ev_gate["selected_rows"], 4)
        self.assertEqual(ev_gate["labelled_rows_among_selected"], 4)
        self.assertEqual(ev_gate["true_positive_rows_among_selected"], 1)
        self.assertEqual(ev_gate["false_positive_rows_among_selected"], 1)
        self.assertEqual(ev_gate["true_negative_rows_among_selected"], 1)
        self.assertEqual(ev_gate["false_negative_rows_among_selected"], 1)
        self.assertEqual(ev_gate["false_negative_rate_among_labelled_selected"], 0.25)
        self.assertEqual(ev_gate["false_positive_rate_among_labelled_selected"], 0.25)

    def test_oracle_labels_are_not_required_for_policy_inclusion(self) -> None:
        artifact = self._artifact(
            [
                {
                    "row_idx": 0,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "expected_verification_value": 0.5,
                    "decision_stale": True,
                    "decision_matches_oracle": None,
                    "oracle_stale_label": None,
                    "memory_update_action": "propose_detector_evidence_refresh",
                }
            ]
        )

        result = compare_loaded_artifacts([("nolabel.json", artifact)])
        variants = _variants_by_name(result)
        ev_gate = variants["recorded_ev_gate"]

        self.assertEqual(ev_gate["selected_rows"], 1)
        self.assertEqual(ev_gate["labelled_rows_among_selected"], 0)
        self.assertIsNone(ev_gate["agreement_rate"])
        self.assertEqual(ev_gate["false_negative_rows_among_selected"], 0)
        self.assertIsNone(ev_gate["false_negative_rate_among_labelled_selected"])
        self.assertEqual(ev_gate["oracle_stale_rows_among_selected_labels"], 0)

    def test_output_schema_claim_boundary_and_memory_updated_rows(self) -> None:
        artifact = self._artifact(
            [
                {
                    "row_idx": 0,
                    "policy_should_verify": True,
                    "candidate_selected_by_budget": True,
                    "expected_verification_value": 0.5,
                    "decision_stale": True,
                    "decision_matches_oracle": True,
                    "oracle_stale_label": True,
                    "memory_update_action": "propose_detector_evidence_refresh",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "live_gsam_closed_loop.json"
            _ = input_path.write_text(json.dumps(artifact), encoding="utf-8")
            result = compare_artifacts([input_path])
            out_dir = root / "out"
            write_outputs(result, out_dir)
            summary_csv = (out_dir / "live_gsam_closed_loop_compare_summary.csv").read_text(encoding="utf-8")
            readme = (out_dir / "README.md").read_text(encoding="utf-8")
            output_json = cast(
                dict[str, object],
                json.loads((out_dir / "live_gsam_closed_loop_compare.json").read_text(encoding="utf-8")),
            )

        self.assertEqual(result["schema_version"], "ai2thor_live_gsam_closed_loop_compare.v1")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["input_files"], [str(input_path)])
        self.assertEqual(result["input_rows"], 1)
        self.assertEqual(result["claim_boundary"], CLAIM_BOUNDARY)
        self.assertIn("CPU-only recorded-row comparison", str(result["claim_boundary"]))
        self.assertIn("memory_updated_rows", summary_csv)
        self.assertIn("false_negative_rows_among_selected", summary_csv)
        self.assertIn("recorded_budgeted_ev_gate", summary_csv)
        self.assertIn("CPU-only recorded-row comparison", readme)
        for row in cast(list[dict[str, object]], output_json["summary_variants"]):
            self.assertEqual(row["memory_updated_rows"], 0)


def _variants_by_name(result: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = cast(list[dict[str, object]], result["summary_variants"])
    return {str(row["variant"]): row for row in rows}


if __name__ == "__main__":
    _ = unittest.main()
