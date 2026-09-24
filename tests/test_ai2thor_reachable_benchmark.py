import unittest

from embodied_memory_pilot.ai2thor_reachable_benchmark import (
    build_reachable_tasks,
    evaluate_policy_reachable_costs,
    nearest_position,
    reachable_path_distance,
)
from embodied_memory_pilot.pilot import NoMemoryPolicy, TaskConditionedPolicy


class AI2ThorReachableBenchmarkTest(unittest.TestCase):
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
                        {"x": 2, "y": 0, "z": 1},
                        {"x": 2, "y": 0, "z": 2},
                    ],
                    "visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 2},
                        },
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "receptacle": True,
                            "position": {"x": 0, "y": 1, "z": 1},
                        },
                    ],
                }
            ]
        }

    def test_nearest_position_uses_ground_plane(self) -> None:
        nearest = nearest_position(
            {"x": 2.1, "y": 99, "z": 1.9},
            [{"x": 0, "z": 0}, {"x": 2, "z": 2}],
        )

        self.assertEqual(nearest, {"x": 2, "z": 2})

    def test_reachable_path_distance_uses_grid_path_not_euclidean_shortcut(self) -> None:
        distance = reachable_path_distance(
            {"x": 0, "z": 0},
            {"x": 2, "z": 2},
            self.sample_probe()["scenes"][0]["reachable_positions"],
        )

        self.assertEqual(distance, 4.0)

    def test_build_reachable_tasks_uses_controller_reachable_positions(self) -> None:
        tasks = build_reachable_tasks(self.sample_probe())

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].target, "Apple")
        self.assertEqual(tasks[0].true_location, "FloorPlan1:Apple|1")
        self.assertEqual(tasks[0].direct_action_cost, 5.0)
        self.assertGreater(tasks[0].scene_search_cost, tasks[0].direct_action_cost)

    def test_task_conditioned_reduces_reachable_navigation_cost(self) -> None:
        probe = self.sample_probe()
        no_memory = evaluate_policy_reachable_costs(probe, policy=NoMemoryPolicy(), budget=1)
        task_conditioned = evaluate_policy_reachable_costs(probe, policy=TaskConditionedPolicy(), budget=1)

        self.assertEqual(task_conditioned["completion_rate"], 1.0)
        self.assertLess(task_conditioned["avg_reachable_cost"], no_memory["avg_reachable_cost"])
        self.assertGreater(task_conditioned["saved_reachable_cost"], 0)


if __name__ == "__main__":
    unittest.main()
