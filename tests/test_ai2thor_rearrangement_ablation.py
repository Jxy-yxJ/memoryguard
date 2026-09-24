import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_ablation import (
    DetectOnlyRearrangementRiskPolicy,
    ThresholdRearrangementRiskPolicy,
    VerifyAllRearrangementPolicy,
    aggregate_ablation_rows,
    delta_vs_baseline,
    evaluate_rearrangement_ablation,
    run_ablation_grid,
    write_outputs,
)


class AI2ThorRearrangementAblationTest(unittest.TestCase):
    def sample_probe(self):
        return {
            "status": "ok",
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                        {"x": 3, "y": 0, "z": 0},
                    ],
                    "before_visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        }
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        }
                    ],
                }
            ],
        }

    def test_verify_all_recovers_stale_hit(self) -> None:
        row = evaluate_rearrangement_ablation(
            self.sample_probe(),
            policy=VerifyAllRearrangementPolicy(),
            ablation="verify_all",
            budget=1,
            verification_cost=0.5,
        )

        self.assertEqual(row["completion_rate"], 1.0)
        self.assertEqual(row["stale_errors"], 0)
        self.assertEqual(row["verifications"], 1)
        self.assertEqual(row["verification_catches"], 1)

    def test_detect_only_catches_stale_hit_without_recovery_success(self) -> None:
        row = evaluate_rearrangement_ablation(
            self.sample_probe(),
            policy=DetectOnlyRearrangementRiskPolicy(threshold=0.1),
            ablation="detect_only_no_recovery",
            budget=1,
            verification_cost=0.5,
            recover_caught_stale=False,
        )

        self.assertEqual(row["completion_rate"], 0.0)
        self.assertEqual(row["stale_errors"], 0)
        self.assertEqual(row["verifications"], 1)
        self.assertEqual(row["verification_catches"], 1)

    def test_threshold_sensitivity_changes_stale_failure(self) -> None:
        low_threshold = evaluate_rearrangement_ablation(
            self.sample_probe(),
            policy=ThresholdRearrangementRiskPolicy(threshold=0.1),
            ablation="verify_threshold",
            budget=1,
            verification_cost=0.5,
        )
        high_threshold = evaluate_rearrangement_ablation(
            self.sample_probe(),
            policy=ThresholdRearrangementRiskPolicy(threshold=0.5),
            ablation="verify_threshold",
            budget=1,
            verification_cost=0.5,
        )

        self.assertEqual(low_threshold["completion_rate"], 1.0)
        self.assertEqual(low_threshold["stale_errors"], 0)
        self.assertEqual(high_threshold["completion_rate"], 0.0)
        self.assertEqual(high_threshold["stale_errors"], 1)

    def test_grid_aggregation_and_delta_use_salience_baseline(self) -> None:
        rows = run_ablation_grid(
            [self.sample_probe(), self.sample_probe()],
            budgets=[1],
            verification_costs=[0.5],
            thresholds=[0.1],
        )

        summary = aggregate_ablation_rows(rows)
        deltas = delta_vs_baseline(summary)

        self.assertTrue(any(row["ablation"] == "salience_no_verification" for row in summary))
        verify_delta = next(row for row in deltas if row["ablation"] == "verify_threshold_0p1")
        self.assertEqual(verify_delta["seeds"], 2)
        self.assertEqual(verify_delta["completion_rate_delta"], 1.0)
        self.assertEqual(verify_delta["stale_errors_delta"], -1.0)

    def test_write_outputs_emits_summary_and_delta_csv(self) -> None:
        rows = run_ablation_grid(
            [self.sample_probe()],
            budgets=[1],
            verification_costs=[0.5],
            thresholds=[0.1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            write_outputs(rows, out_dir)

            summary_text = (out_dir / "ai2thor_rearrangement_ablation_summary.csv").read_text(encoding="utf-8")
            delta_text = (out_dir / "ai2thor_rearrangement_ablation_delta.csv").read_text(encoding="utf-8")

        self.assertIn("verify_all", summary_text)
        self.assertIn("completion_rate_delta", delta_text)


if __name__ == "__main__":
    unittest.main()
