from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_live_gsam_ablation import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    _detector_outcome_counts,
    load_live_rows,
    run_ablation,
    simulate_all_verify,
    simulate_ev_budgeted,
    simulate_random_budgeted,
    write_outputs,
)


class TestDetectorOutcomeCounts(unittest.TestCase):
    def test_counts_tp_fp_tn_fn(self) -> None:
        rows = [
            {"decision_stale": True, "oracle_stale_label": True},
            {"decision_stale": True, "oracle_stale_label": False},
            {"decision_stale": False, "oracle_stale_label": False},
            {"decision_stale": False, "oracle_stale_label": True},
        ]
        result = _detector_outcome_counts(rows)
        self.assertEqual(result["true_positive_rows"], 1)
        self.assertEqual(result["false_positive_rows"], 1)
        self.assertEqual(result["true_negative_rows"], 1)
        self.assertEqual(result["false_negative_rows"], 1)
        self.assertEqual(result["false_negative_rate"], 0.25)
        self.assertEqual(result["false_positive_rate"], 0.25)


class TestSimulationHelpers(unittest.TestCase):
    def _make_rows(self, count: int, values: list[float]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for i in range(count):
            rows.append({
                "expected_verification_value": values[i % len(values)],
                "decision_stale": i % 2 == 0,
                "oracle_stale_label": i % 3 == 0,
                "decision_matches_oracle": i % 4 == 0,
            })
        return rows

    def test_all_verify_selects_all_rows(self) -> None:
        rows = self._make_rows(5, [1.0, 0.5, 0.2, 0.8, 0.3])
        result = simulate_all_verify(rows)
        self.assertEqual(result["strategy"], "all_verify")
        self.assertEqual(result["selected_rows"], 5)

    def test_ev_budgeted_selects_top_k(self) -> None:
        rows = self._make_rows(5, [1.0, 0.5, 0.2, 0.8, 0.3])
        result = simulate_ev_budgeted(rows, 3)
        self.assertEqual(result["strategy"], "ev_budgeted")
        self.assertEqual(result["selected_rows"], 3)
        self.assertEqual(result["budget"], 3)

    def test_ev_budgeted_budget_gt_rows_selects_all(self) -> None:
        rows = self._make_rows(5, [1.0, 0.5, 0.2, 0.8, 0.3])
        result = simulate_ev_budgeted(rows, 10)
        self.assertEqual(result["selected_rows"], 5)

    def test_random_budgeted_is_stable_with_seed(self) -> None:
        rows = self._make_rows(10, list(range(10, 0, -1)))
        result_a = simulate_random_budgeted(rows, 3, seed=7)
        result_b = simulate_random_budgeted(rows, 3, seed=7)
        self.assertEqual(result_a["selected_rows"], result_b["selected_rows"])
        self.assertEqual(result_a["agreement_rows"], result_b["agreement_rows"])

    def test_random_budgeted_budget_gt_rows_selects_all(self) -> None:
        rows = self._make_rows(5, [1.0, 0.5, 0.2, 0.8, 0.3])
        result = simulate_random_budgeted(rows, 10, seed=0)
        self.assertEqual(result["selected_rows"], 5)


class TestRunAblation(unittest.TestCase):
    def test_run_ablation_structure(self) -> None:
        rows = [
            {
                "expected_verification_value": 0.77,
                "decision_stale": True,
                "oracle_stale_label": True,
                "decision_matches_oracle": True,
            },
            {
                "expected_verification_value": 0.77,
                "decision_stale": True,
                "oracle_stale_label": False,
                "decision_matches_oracle": False,
            },
            {
                "expected_verification_value": 0.77,
                "decision_stale": False,
                "oracle_stale_label": True,
                "decision_matches_oracle": False,
            },
        ]
        result = run_ablation(rows, [1, 2], num_random_trials=3)

        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["input_rows"], 3)
        self.assertEqual(result["budgets"], [1, 2])
        self.assertEqual(result["num_random_trials"], 3)
        self.assertEqual(result["claim_boundary"], CLAIM_BOUNDARY)

        all_verify = result["all_verify"]
        self.assertEqual(all_verify["strategy"], "all_verify")
        self.assertEqual(all_verify["selected_rows"], 3)

        comparisons = result["budget_comparisons"]
        self.assertEqual(len(comparisons), 4)

        ev_1 = comparisons[0]
        self.assertEqual(ev_1["strategy"], "ev_budgeted")
        self.assertEqual(ev_1["budget"], 1)

        random_avg_1 = comparisons[1]
        self.assertEqual(random_avg_1["strategy"], "random_budgeted_avg")
        self.assertEqual(random_avg_1["random_trials"], 3)

    def test_claim_boundary_and_memory(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "expected_verification_value": 0.77,
                "decision_stale": True,
                "oracle_stale_label": True,
                "decision_matches_oracle": True,
            },
        ]
        result = run_ablation(rows, [1], num_random_trials=1)
        self.assertEqual(result["claim_boundary"], CLAIM_BOUNDARY)
        self.assertEqual(result["all_verify"]["memory_updated_rows"], 0)
        for comp in result["budget_comparisons"]:
            self.assertEqual(comp.get("memory_updated_rows"), 0)


class TestLoadLiveRows(unittest.TestCase):
    def test_rows_have_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "input.json"
            artifact_path.write_text(json.dumps({
                "status": "ok",
                "rows": [
                    {"row_idx": 0, "scene": "FloorPlan1"},
                    {"row_idx": 1, "scene": "FloorPlan201"},
                ],
            }))
            rows = load_live_rows(artifact_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["_source_file"], str(artifact_path))
            self.assertEqual(rows[0]["_source_row_offset"], 0)
            self.assertEqual(rows[1]["_source_row_offset"], 1)


class TestWriteOutputs(unittest.TestCase):
    def test_writes_json_csv_readme(self) -> None:
        result = {
            "schema_version": SCHEMA_VERSION,
            "input_rows": 1,
            "budgets": [1],
            "num_random_trials": 1,
            "claim_boundary": CLAIM_BOUNDARY,
            "all_verify": {
                "strategy": "all_verify",
                "selected_rows": 1,
                "memory_updated_rows": 0,
            },
            "budget_comparisons": [
                {
                    "strategy": "ev_budgeted",
                    "budget": 1,
                    "selected_rows": 1,
                    "memory_updated_rows": 0,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "output"
            write_outputs(result, out_dir)
            self.assertTrue((out_dir / "live_gsam_ablation.json").is_file())
            self.assertTrue((out_dir / "live_gsam_ablation.csv").is_file())
            self.assertTrue((out_dir / "README.md").is_file())
            readme = (out_dir / "README.md").read_text()
            self.assertIn(CLAIM_BOUNDARY, readme)


if __name__ == "__main__":
    unittest.main()
