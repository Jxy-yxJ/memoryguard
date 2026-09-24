import unittest

from embodied_memory_pilot.ai2thor_rearrangement_verification import (
    VerifyRearrangementRiskPolicy,
    evaluate_policy_rearrangement_verification,
    run_rearrangement_verification,
)
from embodied_memory_pilot.pilot import SaliencePolicy


class AI2ThorRearrangementVerificationTest(unittest.TestCase):
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

    def test_verification_catches_rearranged_stale_hit_and_recovers(self) -> None:
        verified = evaluate_policy_rearrangement_verification(
            self.sample_probe(),
            policy=VerifyRearrangementRiskPolicy(),
            budget=1,
            verification_cost=0.5,
        )

        self.assertEqual(verified["stale_errors"], 0)
        self.assertEqual(verified["verifications"], 1)
        self.assertEqual(verified["verification_catches"], 1)
        self.assertEqual(verified["completion_rate"], 1.0)

    def test_unverified_rearranged_stale_hit_remains_error(self) -> None:
        salience = evaluate_policy_rearrangement_verification(
            self.sample_probe(),
            policy=SaliencePolicy(),
            budget=1,
            verification_cost=0.5,
        )

        self.assertEqual(salience["stale_errors"], 1)
        self.assertEqual(salience["verifications"], 0)
        self.assertEqual(salience["completion_rate"], 0.0)

    def test_run_rearrangement_verification_reports_summary_ready_rows(self) -> None:
        rows = run_rearrangement_verification(
            self.sample_probe(),
            budgets=[1],
            policies=[VerifyRearrangementRiskPolicy()],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "ai2thor_rearrangement_verification")
        self.assertIn("moved_tasks", rows[0])
        self.assertIn("verification_cost", rows[0])


if __name__ == "__main__":
    unittest.main()
