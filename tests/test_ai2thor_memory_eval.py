import unittest

from embodied_memory_pilot.ai2thor_memory_eval import (
    AI2ThorCalibratedPolicy,
    build_memory_items,
    build_query_tasks,
    evaluate_policy_on_queries,
    policy_suite,
    summarize_rows,
)
from embodied_memory_pilot.pilot import FifoPolicy, SaliencePolicy, TaskConditionedPolicy


class AI2ThorMemoryEvalTest(unittest.TestCase):
    def sample_probe(self):
        return {
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 1},
                        },
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "receptacle": True,
                            "position": {"x": 0, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Floor|1",
                            "object_type": "Floor",
                            "pickupable": False,
                            "receptacle": True,
                            "position": {"x": 0, "y": 0, "z": 0},
                        },
                    ],
                }
            ]
        }

    def test_build_memory_items_assigns_task_features(self) -> None:
        items = build_memory_items(self.sample_probe())

        apple = next(item for item in items if item.object_name == "Apple")
        cabinet = next(item for item in items if item.object_name == "Cabinet")

        self.assertGreater(apple.demand, cabinet.demand)
        self.assertGreater(apple.salience, 0)
        self.assertEqual(apple.location, "FloorPlan1:Apple|1")

    def test_build_query_tasks_targets_pickupable_objects(self) -> None:
        tasks = build_query_tasks(self.sample_probe())

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].target, "Apple")
        self.assertEqual(tasks[0].true_location, "FloorPlan1:Apple|1")

    def test_task_conditioned_can_retain_target_under_tight_budget(self) -> None:
        probe = self.sample_probe()
        rows = [
            evaluate_policy_on_queries(probe, policy=FifoPolicy(), budget=1),
            evaluate_policy_on_queries(probe, policy=SaliencePolicy(), budget=1),
            evaluate_policy_on_queries(probe, policy=TaskConditionedPolicy(), budget=1),
        ]

        summary = {row["policy"]: row for row in summarize_rows(rows)}

        self.assertEqual(summary["task_conditioned"]["query_hit_rate"], 1.0)
        self.assertLessEqual(summary["fifo"]["query_hit_rate"], summary["task_conditioned"]["query_hit_rate"])

    def test_ai2thor_calibrated_policy_prioritizes_pickupable_query_objects(self) -> None:
        probe = {
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "visible_objects": [
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "receptacle": True,
                        },
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                        },
                        {
                            "object_id": "CreditCard|1",
                            "object_type": "CreditCard",
                            "pickupable": True,
                            "receptacle": False,
                        },
                        {
                            "object_id": "Drawer|1",
                            "object_type": "Drawer",
                            "pickupable": False,
                            "receptacle": True,
                        },
                    ],
                }
            ]
        }

        row = evaluate_policy_on_queries(probe, policy=AI2ThorCalibratedPolicy(), budget=2)
        policy_names = [policy.name for policy in policy_suite()]

        self.assertIn("ai2thor_calibrated", policy_names)
        self.assertEqual(row["query_tasks"], 2)
        self.assertEqual(row["query_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
