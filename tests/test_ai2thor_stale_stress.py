import unittest

from embodied_memory_pilot.ai2thor_stale_stress import (
    build_stale_stress_memory_items,
    evaluate_policy_stale_stress,
    run_stale_stress,
    select_probe_scenes,
)
from embodied_memory_pilot.ai2thor_memory_eval import build_memory_items
from embodied_memory_pilot.ai2thor_reachable_benchmark import build_reachable_tasks
from embodied_memory_pilot.pilot import FreshnessAwarePolicy, SaliencePolicy


class AI2ThorStaleStressTest(unittest.TestCase):
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

    def test_stress_items_add_older_same_target_distractors_without_adding_tasks(self) -> None:
        probe = self.sample_probe()

        stress_items = build_stale_stress_memory_items(probe, age_gap=50, stale_salience=1.6)
        base_items = build_memory_items(probe)
        tasks = build_reachable_tasks(probe)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(stress_items), len(base_items) + len(tasks))
        self.assertEqual(stress_items[0].object_name, "Apple")
        self.assertNotEqual(stress_items[0].location, tasks[0].true_location)
        self.assertLess(stress_items[0].observed_at, stress_items[-1].observed_at)

    def test_stale_stress_separates_salience_from_freshness_aware_policy(self) -> None:
        probe = self.sample_probe()

        salience = evaluate_policy_stale_stress(
            probe,
            policy=SaliencePolicy(),
            budget=1,
            age_gap=50,
            stale_salience=1.6,
            stale_volatility=0.9,
        )
        freshness = evaluate_policy_stale_stress(
            probe,
            policy=FreshnessAwarePolicy(),
            budget=1,
            age_gap=50,
            stale_salience=1.6,
            stale_volatility=0.9,
        )

        self.assertEqual(salience["completion_rate"], 0.0)
        self.assertEqual(salience["stale_errors"], 1)
        self.assertEqual(freshness["completion_rate"], 1.0)
        self.assertEqual(freshness["stale_errors"], 0)
        self.assertLess(freshness["avg_reachable_cost"], salience["avg_reachable_cost"])

    def test_run_stale_stress_reports_distractor_counts(self) -> None:
        rows = run_stale_stress(self.sample_probe(), budgets=[1], policies=[SaliencePolicy()])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "ai2thor_stale_reachable_stress")
        self.assertEqual(rows[0]["stale_distractors"], 1)
        self.assertEqual(rows[0]["reachable_tasks"], 1)

    def test_multiple_stale_distractors_are_observed_in_chronological_order(self) -> None:
        probe = self.sample_probe()
        probe["scenes"][0]["visible_objects"].append(
            {
                "object_id": "Book|1",
                "object_type": "Book",
                "pickupable": True,
                "receptacle": False,
                "position": {"x": 1, "y": 1, "z": 0},
            }
        )

        stress_items = build_stale_stress_memory_items(probe, age_gap=50, stale_salience=1.6)
        observed_at = [item.observed_at for item in stress_items]

        self.assertEqual(observed_at, sorted(observed_at))

    def test_select_probe_scenes_keeps_only_requested_scenes(self) -> None:
        probe = self.sample_probe()
        probe["scenes"].append({**probe["scenes"][0], "scene": "FloorPlan2"})

        selected = select_probe_scenes(probe, scenes=["FloorPlan2"])

        self.assertEqual([scene["scene"] for scene in selected["scenes"]], ["FloorPlan2"])
        self.assertEqual([scene["scene"] for scene in probe["scenes"]], ["FloorPlan1", "FloorPlan2"])
        self.assertEqual(selected["summary"]["scenes"], 1)


if __name__ == "__main__":
    unittest.main()
