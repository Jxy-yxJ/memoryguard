import unittest

from embodied_memory_pilot.ai2thor_reachable_benchmark import evaluate_policy_reachable_costs
from embodied_memory_pilot.ai2thor_weight_search import (
    WeightedAI2ThorPolicy,
    candidate_weight_grid,
    split_probe_by_scenes,
    train_weight_search,
)
from embodied_memory_pilot.pilot import SaliencePolicy


class AI2ThorWeightSearchTest(unittest.TestCase):
    def sample_probe(self):
        return {
            "scenes": [
                {
                    "scene": "TrainScene",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                        {"x": 3, "y": 0, "z": 0},
                    ],
                    "visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Book|1",
                            "object_type": "Book",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Cabinet|1",
                            "object_type": "Cabinet",
                            "pickupable": False,
                            "receptacle": True,
                            "position": {"x": 3, "y": 1, "z": 0},
                        },
                    ],
                },
                {
                    "scene": "HeldoutScene",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                    ],
                    "visible_objects": [
                        {
                            "object_id": "CreditCard|1",
                            "object_type": "CreditCard",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Drawer|1",
                            "object_type": "Drawer",
                            "pickupable": False,
                            "receptacle": True,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                },
            ]
        }

    def test_split_probe_by_scenes_preserves_requested_scene_order(self) -> None:
        train, heldout = split_probe_by_scenes(self.sample_probe(), train_scenes=["TrainScene"])

        self.assertEqual([scene["scene"] for scene in train["scenes"]], ["TrainScene"])
        self.assertEqual([scene["scene"] for scene in heldout["scenes"]], ["HeldoutScene"])

    def test_weighted_policy_can_score_pickupable_items_above_receptacles(self) -> None:
        policy = WeightedAI2ThorPolicy(
            name="weighted_test",
            weights={"salience": 1.0, "demand": 1.0, "target": 0.0, "freshness": 0.0, "stale": 0.0},
        )
        probe = self.sample_probe()

        row = evaluate_policy_reachable_costs(probe, policy=policy, budget=2)

        self.assertGreaterEqual(row["query_hit_rate"], 0.5)

    def test_train_weight_search_returns_best_weights_and_heldout_metrics(self) -> None:
        result = train_weight_search(
            self.sample_probe(),
            train_scenes=["TrainScene"],
            budgets=[1, 2],
            candidates=[
                {"salience": 0.1, "demand": 0.1, "target": 0.0, "freshness": 1.0, "stale": 0.0},
                {"salience": 1.0, "demand": 2.0, "target": 0.0, "freshness": 0.0, "stale": 0.0},
            ],
        )

        self.assertEqual(result["best_weights"]["demand"], 2.0)
        self.assertIn("heldout_summary", result)
        self.assertGreaterEqual(result["heldout_summary"][0]["reachable_tasks"], 1)

    def test_candidate_weight_grid_has_interpretable_weight_sets(self) -> None:
        candidates = candidate_weight_grid()

        self.assertGreaterEqual(len(candidates), 3)
        self.assertTrue(all("salience" in candidate and "demand" in candidate for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
