import unittest

from embodied_memory_pilot.ai2thor_stale_verification import (
    VerifyHighRiskSaliencePolicy,
    evaluate_policy_stale_verification,
    run_stale_verification,
)
from embodied_memory_pilot.pilot import FreshnessAwarePolicy, SaliencePolicy


class AI2ThorStaleVerificationTest(unittest.TestCase):
    def sample_probe(self):
        return {
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                    ],
                    "visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        }
                    ],
                }
            ]
        }

    def test_verification_catches_stale_hit_and_falls_back_to_search(self) -> None:
        verified = evaluate_policy_stale_verification(
            self.sample_probe(),
            policy=VerifyHighRiskSaliencePolicy(),
            budget=1,
            age_gap=50,
            stale_salience=1.6,
            stale_volatility=0.9,
            verification_cost=0.5,
        )

        self.assertEqual(verified["stale_errors"], 0)
        self.assertEqual(verified["verifications"], 1)
        self.assertEqual(verified["verification_catches"], 1)
        self.assertEqual(verified["completion_rate"], 1.0)

    def test_without_verification_stale_hit_counts_as_error(self) -> None:
        salience = evaluate_policy_stale_verification(
            self.sample_probe(),
            policy=SaliencePolicy(),
            budget=1,
            age_gap=50,
            stale_salience=1.6,
            stale_volatility=0.9,
            verification_cost=0.5,
        )

        self.assertEqual(salience["stale_errors"], 1)
        self.assertEqual(salience["verifications"], 0)
        self.assertEqual(salience["completion_rate"], 0.0)

    def test_run_stale_verification_reports_summary_ready_rows(self) -> None:
        rows = run_stale_verification(self.sample_probe(), budgets=[1], policies=[FreshnessAwarePolicy()])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "ai2thor_stale_verification")
        self.assertIn("verification_cost", rows[0])
        self.assertIn("stale_distractors", rows[0])


if __name__ == "__main__":
    unittest.main()
