import unittest

from embodied_memory_pilot.ai2thor_action_benchmark import (
    build_action_tasks,
    evaluate_policy_action_costs,
    navigation_distance,
    summarize_rows,
)
from embodied_memory_pilot.pilot import NoMemoryPolicy, TaskConditionedPolicy


class AI2ThorActionBenchmarkTest(unittest.TestCase):
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
                            "position": {"x": 3, "y": 1, "z": 4},
                        },
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "receptacle": True,
                            "position": {"x": 0, "y": 1, "z": 1},
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

    def test_navigation_distance_uses_ground_plane(self) -> None:
        distance = navigation_distance({"x": 0, "y": 10, "z": 0}, {"x": 3, "y": 1, "z": 4})

        self.assertEqual(distance, 5.0)

    def test_build_action_tasks_estimates_direct_and_search_costs(self) -> None:
        tasks = build_action_tasks(self.sample_probe())

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].target, "Apple")
        self.assertEqual(tasks[0].true_location, "FloorPlan1:Apple|1")
        self.assertEqual(tasks[0].direct_action_cost, 6.0)
        self.assertGreater(tasks[0].scene_search_cost, tasks[0].direct_action_cost)

    def test_task_conditioned_memory_reduces_action_cost_when_target_is_retained(self) -> None:
        probe = self.sample_probe()
        no_memory = evaluate_policy_action_costs(probe, policy=NoMemoryPolicy(), budget=1)
        task_conditioned = evaluate_policy_action_costs(probe, policy=TaskConditionedPolicy(), budget=1)

        self.assertEqual(task_conditioned["query_hit_rate"], 1.0)
        self.assertLess(task_conditioned["avg_action_cost"], no_memory["avg_action_cost"])
        self.assertGreater(task_conditioned["saved_action_cost"], 0)

        summary = {row["policy"]: row for row in summarize_rows([no_memory, task_conditioned])}
        self.assertEqual(summary["task_conditioned"]["completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
