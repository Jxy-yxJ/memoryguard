import unittest

from embodied_memory_pilot.ai2thor_interaction_memory import (
    VerifyInteractionRiskPolicy,
    build_interaction_tasks,
    evaluate_policy_interaction,
    run_interaction_benchmark,
)
from embodied_memory_pilot.pilot import SaliencePolicy


class AI2ThorInteractionMemoryTest(unittest.TestCase):
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
                        },
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "openable": True,
                            "receptacle": True,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "openable": True,
                            "receptacle": True,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                }
            ],
        }

    def test_build_interaction_tasks_labels_pickup_and_open_actions(self) -> None:
        tasks = build_interaction_tasks(self.sample_probe())

        self.assertEqual([task.action for task in tasks], ["PickupObject", "OpenObject"])
        self.assertTrue(tasks[0].rearranged)
        self.assertFalse(tasks[1].rearranged)

    def test_unverified_stale_interaction_memory_fails_before_action(self) -> None:
        row = evaluate_policy_interaction(self.sample_probe(), policy=SaliencePolicy(), budget=2)

        self.assertEqual(row["interaction_tasks"], 2)
        self.assertEqual(row["interaction_stale_errors"], 1)
        self.assertLess(row["completion_rate"], 1.0)

    def test_verification_recovers_stale_interaction_precondition(self) -> None:
        row = evaluate_policy_interaction(self.sample_probe(), policy=VerifyInteractionRiskPolicy(), budget=2)

        self.assertEqual(row["interaction_stale_errors"], 0)
        self.assertEqual(row["verification_catches"], 1)
        self.assertEqual(row["completion_rate"], 1.0)

    def test_run_interaction_benchmark_reports_summary_ready_rows(self) -> None:
        rows = run_interaction_benchmark(self.sample_probe(), budgets=[2], policies=[VerifyInteractionRiskPolicy()])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "ai2thor_interaction_memory")
        self.assertIn("pickup_tasks", rows[0])
        self.assertIn("open_tasks", rows[0])


if __name__ == "__main__":
    unittest.main()
