from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

from embodied_memory_pilot.ai2thor_live_gsam_failure_slices import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    detector_outcome,
    load_live_rows,
    summarize_failure_slices,
    write_outputs,
)


class LiveGSAMFailureSlicesTest(unittest.TestCase):
    def test_detector_outcome_classifies_boolean_decision_and_label_pairs(self) -> None:
        self.assertEqual(detector_outcome({"decision_stale": True, "oracle_stale_label": True}), "tp")
        self.assertEqual(detector_outcome({"decision_stale": True, "oracle_stale_label": False}), "fp")
        self.assertEqual(detector_outcome({"decision_stale": False, "oracle_stale_label": False}), "tn")
        self.assertEqual(detector_outcome({"decision_stale": False, "oracle_stale_label": True}), "fn")

    def test_detector_outcome_requires_boolean_decision_and_label(self) -> None:
        self.assertEqual(detector_outcome({"decision_stale": None, "oracle_stale_label": True}), "unlabelled")
        self.assertEqual(detector_outcome({"decision_stale": False, "oracle_stale_label": None}), "unlabelled")
        self.assertEqual(detector_outcome({"decision_stale": "false", "oracle_stale_label": True}), "unlabelled")
        self.assertEqual(detector_outcome({"decision_stale": False, "oracle_stale_label": 1}), "unlabelled")

    def test_load_live_rows_reads_live_gsam_closed_loop_style_artifact(self) -> None:
        artifact = {"schema_version": "ai2thor_live_gsam_closed_loop.v1", "rows": [{"scene": "FloorPlan1"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_gsam_closed_loop.json"
            _ = path.write_text(json.dumps(artifact), encoding="utf-8")
            rows = load_live_rows(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scene"], "FloorPlan1")
        self.assertEqual(rows[0]["_source_row_offset"], 0)
        self.assertEqual(rows[0]["_source_file"], str(path))

    def test_summarize_failure_slices_groups_false_negatives_by_recorded_fields(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "scene": "FloorPlan1",
                "target": "Apple",
                "visibility_error": None,
                "target_visible_after_revisit_sweep": True,
                "decision_reason": "no_detection",
                "candidate_selected_by_budget": True,
                "decision_stale": False,
                "oracle_stale_label": True,
            },
            {
                "scene": "FloorPlan1",
                "target": "Book",
                "visibility_error": "no_visible_target",
                "target_visible_after_revisit_sweep": False,
                "decision_reason": "no_detection",
                "candidate_selected_by_budget": False,
                "decision_stale": False,
                "oracle_stale_label": True,
            },
            {
                "scene": "FloorPlan3",
                "target": "Apple",
                "visibility_error": None,
                "target_visible_after_revisit_sweep": True,
                "decision_reason": "matched",
                "candidate_selected_by_budget": True,
                "decision_stale": True,
                "oracle_stale_label": True,
            },
            {
                "scene": "FloorPlan3",
                "target": "Mug",
                "visibility_error": None,
                "target_visible_after_revisit_sweep": True,
                "decision_reason": "matched",
                "candidate_selected_by_budget": True,
                "decision_stale": True,
                "oracle_stale_label": False,
            },
            {
                "scene": "FloorPlan201",
                "target": "Pencil",
                "visibility_error": None,
                "target_visible_after_revisit_sweep": True,
                "decision_reason": None,
                "candidate_selected_by_budget": False,
                "decision_stale": None,
                "oracle_stale_label": True,
            },
        ]

        result = summarize_failure_slices(rows)
        fn_slices = cast(dict[str, list[dict[str, object]]], result["false_negative_slices"])

        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["input_rows"], 5)
        self.assertEqual(result["labelled_rows"], 4)
        self.assertEqual(result["memory_updated_rows"], 0)
        self.assertEqual(result["outcome_counts"], {"tp": 1, "fp": 1, "tn": 0, "fn": 2, "unlabelled": 1})
        self.assertEqual(fn_slices["scene"][0], {"field": "scene", "value": "FloorPlan1", "rows": 2})
        self.assertIn({"field": "target", "value": "Apple", "rows": 1}, fn_slices["target"])
        self.assertIn({"field": "target", "value": "Book", "rows": 1}, fn_slices["target"])
        self.assertIn(
            {"outcome": "fn", "field": "decision_reason", "value": "no_detection", "rows": 2},
            cast(list[dict[str, object]], result["outcome_slice_rows"]),
        )

    def test_output_schema_claim_boundary_and_write_outputs(self) -> None:
        result = summarize_failure_slices(
            [
                {
                    "scene": "FloorPlan1",
                    "target": "Apple",
                    "visibility_error": None,
                    "target_visible_after_revisit_sweep": True,
                    "decision_reason": "no_detection",
                    "candidate_selected_by_budget": True,
                    "decision_stale": False,
                    "oracle_stale_label": True,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            write_outputs(result, out_dir)
            output_json = cast(dict[str, object], json.loads((out_dir / "live_gsam_failure_slices.json").read_text(encoding="utf-8")))
            output_csv = (out_dir / "live_gsam_failure_slices.csv").read_text(encoding="utf-8")
            readme = (out_dir / "README.md").read_text(encoding="utf-8")

        self.assertEqual(output_json["schema_version"], SCHEMA_VERSION)
        self.assertEqual(output_json["claim_boundary"], CLAIM_BOUNDARY)
        self.assertEqual(output_json["memory_updated_rows"], 0)
        self.assertIn("CPU-only post-processing", str(output_json["claim_boundary"]))
        self.assertIn("oracle labels are evaluation-only", str(output_json["claim_boundary"]))
        self.assertIn("no new live run", str(output_json["claim_boundary"]))
        self.assertIn("no memory writeback", str(output_json["claim_boundary"]))
        self.assertIn("outcome,field,value,rows", output_csv)
        self.assertIn("fn,scene,FloorPlan1,1", output_csv)
        self.assertIn("Memory updated rows: 0", readme)
        self.assertIn("Oracle labels are used only as post-hoc evaluation labels", readme)

    def test_cli_writes_outputs_to_requested_directory(self) -> None:
        artifact = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "rows": [
                {
                    "scene": "FloorPlan1",
                    "target": "Apple",
                    "visibility_error": None,
                    "target_visible_after_revisit_sweep": True,
                    "decision_reason": "no_detection",
                    "candidate_selected_by_budget": True,
                    "decision_stale": False,
                    "oracle_stale_label": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "live_gsam_closed_loop.json"
            out_dir = root / "slices"
            _ = artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "embodied_memory_pilot.ai2thor_live_gsam_failure_slices",
                    str(artifact_path),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_json = cast(dict[str, object], json.loads((out_dir / "live_gsam_failure_slices.json").read_text(encoding="utf-8")))

        self.assertIn(SCHEMA_VERSION, completed.stdout)
        self.assertEqual(output_json["outcome_counts"], {"tp": 0, "fp": 0, "tn": 0, "fn": 1, "unlabelled": 0})
        self.assertEqual(output_json["memory_updated_rows"], 0)


if __name__ == "__main__":
    _ = unittest.main()
