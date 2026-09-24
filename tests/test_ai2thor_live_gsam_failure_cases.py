from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

from embodied_memory_pilot.ai2thor_live_gsam_failure_cases import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    CASE_FIELDS,
    is_false_negative,
    load_live_rows,
    materialize_failure_cases,
    write_outputs,
)


class LiveGSAMFailureCasesTest(unittest.TestCase):
    def test_is_false_negative_requires_boolean_false_decision_and_true_label(self) -> None:
        self.assertTrue(is_false_negative({"decision_stale": False, "oracle_stale_label": True}))
        self.assertFalse(is_false_negative({"decision_stale": True, "oracle_stale_label": True}))
        self.assertFalse(is_false_negative({"decision_stale": False, "oracle_stale_label": False}))
        self.assertFalse(is_false_negative({"decision_stale": "false", "oracle_stale_label": True}))
        self.assertFalse(is_false_negative({"decision_stale": False, "oracle_stale_label": 1}))

    def test_load_live_rows_reads_closed_loop_artifact_and_preserves_source_offset(self) -> None:
        artifact = {"schema_version": "ai2thor_live_gsam_closed_loop.v1", "rows": [{"scene": "FloorPlan1"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_gsam_closed_loop.json"
            _ = path.write_text(json.dumps(artifact), encoding="utf-8")
            rows = load_live_rows(path)

        self.assertEqual(rows, [{"scene": "FloorPlan1", "_source_file": str(path), "_source_row_offset": 0}])

    def test_materialize_failure_cases_projects_compact_fields_and_ranks_without_oracle_labels(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "row_idx": 9,
                "scene": "FloorPlan3",
                "seed": 68,
                "target": "Spoon",
                "visibility_error": None,
                "target_visible_after_revisit_sweep": True,
                "decision_reason": "matched",
                "decision_matched_distance": 0.05,
                "decision_detections_considered": 4,
                "expected_verification_value": 0.2,
                "p_stale_estimate": 0.85,
                "candidate_rank_by_ev": 3,
                "candidate_selected_by_budget": True,
                "oracle_stale_label": True,
                "decision_stale": False,
                "extra_field_must_not_escape": "large diagnostic",
            },
            {
                "row_idx": 4,
                "scene": "FloorPlan1",
                "seed": 11,
                "target": "Book",
                "visibility_error": "target_not_visible_after_revisit_sweep",
                "target_visible_after_revisit_sweep": False,
                "decision_reason": "matched",
                "decision_matched_distance": 0.0377,
                "decision_detections_considered": 61,
                "expected_verification_value": 0.8,
                "p_stale_estimate": 0.85,
                "candidate_rank_by_ev": 1,
                "candidate_selected_by_budget": True,
                "oracle_stale_label": True,
                "decision_stale": False,
            },
            {
                "row_idx": 2,
                "scene": "FloorPlan1",
                "target": "Apple",
                "oracle_stale_label": False,
                "decision_stale": False,
            },
            {
                "row_idx": 1,
                "scene": "FloorPlan1",
                "target": "Mug",
                "oracle_stale_label": True,
                "decision_stale": True,
            },
        ]

        result = materialize_failure_cases(rows)
        cases = cast(list[dict[str, object]], result["cases"])

        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["input_rows"], 4)
        self.assertEqual(result["false_negative_rows"], 2)
        self.assertEqual(result["case_rows"], 2)
        self.assertEqual(result["selection_rule"], "decision_stale is False and oracle_stale_label is True")
        self.assertEqual([case["row_idx"] for case in cases], [4, 9])
        self.assertEqual(set(cases[0]), set(CASE_FIELDS))
        self.assertNotIn("extra_field_must_not_escape", cases[0])
        self.assertEqual(cases[0]["visibility_error"], "target_not_visible_after_revisit_sweep")
        self.assertEqual(cases[1]["visibility_error"], None)

    def test_max_cases_applies_after_deterministic_false_negative_ranking(self) -> None:
        rows = [
            {"row_idx": 3, "scene": "FloorPlan2", "target": "Cup", "visibility_error": None, "decision_stale": False, "oracle_stale_label": True},
            {"row_idx": 2, "scene": "FloorPlan1", "target": "Book", "visibility_error": "target_not_visible_after_revisit_sweep", "decision_stale": False, "oracle_stale_label": True},
            {"row_idx": 1, "scene": "FloorPlan1", "target": "Apple", "visibility_error": None, "decision_stale": False, "oracle_stale_label": True},
        ]

        result = materialize_failure_cases(rows, max_cases=1)
        cases = cast(list[dict[str, object]], result["cases"])

        self.assertEqual(result["false_negative_rows"], 3)
        self.assertEqual(result["case_rows"], 1)
        self.assertEqual(cases[0]["row_idx"], 2)

    def test_output_schema_claim_boundary_memory_updated_rows_and_write_outputs(self) -> None:
        result = materialize_failure_cases([
            {
                "row_idx": 4,
                "scene": "FloorPlan1",
                "seed": 11,
                "target": "Book",
                "visibility_error": "target_not_visible_after_revisit_sweep",
                "target_visible_after_revisit_sweep": False,
                "decision_reason": "matched",
                "decision_matched_distance": 0.0377,
                "decision_detections_considered": 61,
                "expected_verification_value": 0.8,
                "p_stale_estimate": 0.85,
                "candidate_rank_by_ev": 1,
                "candidate_selected_by_budget": True,
                "oracle_stale_label": True,
                "decision_stale": False,
            }
        ])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "cases"
            write_outputs(result, out_dir)
            output_json = cast(dict[str, object], json.loads((out_dir / "live_gsam_failure_cases.json").read_text(encoding="utf-8")))
            with (out_dir / "live_gsam_failure_cases.csv").open("r", encoding="utf-8") as handle:
                output_csv = list(csv.DictReader(handle))
            readme = (out_dir / "README.md").read_text(encoding="utf-8")

        self.assertEqual(output_json["schema_version"], SCHEMA_VERSION)
        self.assertEqual(output_json["claim_boundary"], CLAIM_BOUNDARY)
        self.assertEqual(output_json["memory_updated_rows"], 0)
        self.assertIn("CPU-only materialization of qualitative false-negative cases", str(output_json["claim_boundary"]))
        self.assertIn("oracle labels post-hoc only", str(output_json["claim_boundary"]))
        self.assertIn("not new live run", str(output_json["claim_boundary"]))
        self.assertIn("memory writeback", str(output_json["claim_boundary"]))
        self.assertEqual(output_csv[0]["row_idx"], "4")
        self.assertEqual(output_csv[0]["oracle_stale_label"], "True")
        self.assertIn("Memory updated rows: 0", readme)
        self.assertIn("not new live evidence", readme)

    def test_cli_writes_outputs_to_requested_directory(self) -> None:
        artifact = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "rows": [
                {
                    "row_idx": 1,
                    "scene": "FloorPlan1",
                    "seed": 7,
                    "target": "Book",
                    "visibility_error": "target_not_visible_after_revisit_sweep",
                    "target_visible_after_revisit_sweep": False,
                    "decision_stale": False,
                    "oracle_stale_label": True,
                },
                {"row_idx": 2, "scene": "FloorPlan1", "target": "Apple", "decision_stale": True, "oracle_stale_label": True},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "live_gsam_closed_loop.json"
            out_dir = root / "cases"
            _ = artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "embodied_memory_pilot.ai2thor_live_gsam_failure_cases",
                    str(artifact_path),
                    "--out-dir",
                    str(out_dir),
                    "--max-cases",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_json = cast(dict[str, object], json.loads((out_dir / "live_gsam_failure_cases.json").read_text(encoding="utf-8")))

        self.assertIn(SCHEMA_VERSION, completed.stdout)
        self.assertEqual(output_json["false_negative_rows"], 1)
        self.assertEqual(output_json["case_rows"], 1)
        self.assertEqual(output_json["memory_updated_rows"], 0)


if __name__ == "__main__":
    _ = unittest.main()
