import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_maintenance_multiseed import (
    aggregate_summaries,
    delta_vs_passive,
    pareto_front,
    write_outputs,
)


class AI2ThorRearrangementMaintenanceMultiseedTest(unittest.TestCase):
    def write_summary(self, root: Path, seed: str, active_completion: float, passive_completion: float = 0.0) -> Path:
        path = root / seed / "ai2thor_rearrangement_maintenance_summary.csv"
        path.parent.mkdir(parents=True)
        _ = path.write_text(
            "\n".join(
                [
                    "budget,maintenance_budget,maintenance_policy,runs,rearrangement_tasks,moved_tasks,memory_items,retained_items,completion_rate,avg_reachable_cost,query_hit_rate,stale_errors,maintenance_checks,maintenance_catches,maintenance_cost,saved_reachable_cost",
                    f"4,1,passive,1,2,2,33,4,{passive_completion},14.6,0.0,1,0,0,0.5,0.0",
                    f"4,1,heuristic_active,1,2,2,33,4,{active_completion},8.8,0.5,0,1,1,0.5,9.3",
                    f"4,1,random_active,1,2,2,33,4,{active_completion},8.8,0.5,0,1,1,0.5,9.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_aggregate_summaries_reports_seed_count_and_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                self.write_summary(root, "seed7", active_completion=0.5),
                self.write_summary(root, "seed11", active_completion=1.0),
            ]

            rows = aggregate_summaries(paths)

        heuristic = next(row for row in rows if row["maintenance_policy"] == "heuristic_active")
        passive = next(row for row in rows if row["maintenance_policy"] == "passive")
        self.assertEqual(heuristic["seeds"], 2)
        self.assertEqual(heuristic["completion_rate_mean"], 0.75)
        self.assertGreater(float(heuristic["completion_rate_std"]), 0.0)
        self.assertEqual(passive["stale_errors_mean"], 1.0)

    def test_delta_and_pareto_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = aggregate_summaries([self.write_summary(root, "seed7", active_completion=0.5)])
            deltas = delta_vs_passive(rows)
            front = pareto_front(rows)
            out_dir = root / "out"

            write_outputs(rows, out_dir)

            delta_text = (out_dir / "ai2thor_rearrangement_maintenance_multiseed_delta.csv").read_text(encoding="utf-8")
            pareto_text = (out_dir / "ai2thor_rearrangement_maintenance_multiseed_pareto.csv").read_text(encoding="utf-8")

        self.assertTrue(any(row["method_policy"] == "heuristic_active" for row in deltas))
        self.assertTrue(any(row["maintenance_policy"] == "heuristic_active" for row in front))
        self.assertIn("completion_rate_delta", delta_text)
        self.assertIn("maintenance_policy", pareto_text)


if __name__ == "__main__":
    _ = unittest.main()
