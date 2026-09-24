import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_maintenance import (
    HeuristicActiveMaintenanceSelection,
    PassiveMaintenanceSelection,
    RandomActiveMaintenanceSelection,
    evaluate_policy_rearrangement_maintenance,
    maintenance_tau,
    pareto_front,
    run_rearrangement_maintenance,
    summarize_rows,
    write_outputs,
)
from embodied_memory_pilot.pilot import MemoryItem

Probe = dict[str, object]


class AI2ThorRearrangementMaintenanceTest(unittest.TestCase):
    def sample_probe(self) -> Probe:
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
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        },
                    ],
                }
            ],
        }

    def test_passive_leaves_stale_error(self) -> None:
        rows = evaluate_policy_rearrangement_maintenance(
            self.sample_probe(),
            policy=PassiveMaintenanceSelection(),
            budget=1,
            maintenance_budget=0,
            maintenance_cost=0.5,
        )

        self.assertEqual(rows["stale_errors"], 1)
        self.assertEqual(rows["maintenance_checks"], 0)
        self.assertEqual(rows["completion_rate"], 0.0)

    def test_heuristic_active_recovers_stale_hit(self) -> None:
        row = evaluate_policy_rearrangement_maintenance(
            self.sample_probe(),
            policy=HeuristicActiveMaintenanceSelection(),
            budget=1,
            maintenance_budget=1,
            maintenance_cost=0.5,
        )

        self.assertEqual(row["maintenance_policy"], "heuristic_active")
        self.assertEqual(row["maintenance_checks"], 1)
        self.assertEqual(row["maintenance_catches"], 1)
        self.assertEqual(row["stale_errors"], 0)
        self.assertEqual(row["completion_rate"], 1.0)

    def test_random_active_is_reportable_and_tau_orders_items(self) -> None:
        row = evaluate_policy_rearrangement_maintenance(
            self.sample_probe(),
            policy=RandomActiveMaintenanceSelection(),
            budget=1,
            maintenance_budget=1,
            maintenance_cost=0.5,
            random_seed=7,
        )

        self.assertEqual(row["maintenance_policy"], "random_active")
        self.assertIn("saved_reachable_cost", row)
        self.assertLess(
            maintenance_tau(MemoryItem("Apple", "room_a", 1, 0.2, 0.9, 0.1), now=10, current_target="Apple", future_counts={"Apple": 0}),
            maintenance_tau(MemoryItem("Mug", "room_b", 9, 0.9, 0.1, 0.1), now=10, current_target="Apple", future_counts={"Apple": 0}),
        )

    def test_grid_summary_and_outputs(self) -> None:
        rows = run_rearrangement_maintenance(
            self.sample_probe(),
            budgets=[1],
            maintenance_budgets=[0, 1],
            policies=[PassiveMaintenanceSelection(), HeuristicActiveMaintenanceSelection()],
            maintenance_cost=0.5,
            random_seed=0,
        )

        summary = summarize_rows(rows)
        front = pareto_front(summary)

        self.assertEqual(len(summary), 4)
        self.assertGreaterEqual(len(front), 1)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_outputs(rows, self.sample_probe(), out_dir)

            summary_text = (out_dir / "ai2thor_rearrangement_maintenance_summary.csv").read_text(encoding="utf-8")
            pareto_text = (out_dir / "ai2thor_rearrangement_maintenance_pareto.csv").read_text(encoding="utf-8")

        self.assertIn("maintenance_policy", summary_text)
        self.assertIn("completion_rate", pareto_text)


if __name__ == "__main__":
    _ = unittest.main()
