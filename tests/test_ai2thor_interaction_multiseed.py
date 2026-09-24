import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_interaction_multiseed import aggregate_summaries, delta_vs_baseline, write_outputs


class AI2ThorInteractionMultiseedTest(unittest.TestCase):
    def write_summary(self, root: Path, seed: str, salience_completion: float, verify_completion: float) -> Path:
        path = root / seed / "ai2thor_interaction_memory_summary.csv"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                [
                    "budget,policy,runs,interaction_tasks,pickup_tasks,open_tasks,moved_tasks,memory_items,retained_items,completion_rate,avg_interaction_cost,query_hit_rate,interaction_stale_errors,verifications,verification_catches,saved_reachable_cost",
                    f"16,salience,1,7,7,0,4,67,16,{salience_completion},9.1,0.4,3,0,0,19.8",
                    f"16,verify_interaction_risk,1,7,7,0,4,67,16,{verify_completion},8.6,0.4,0,6,3,18.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_aggregate_summaries_reports_interaction_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                self.write_summary(root, "seed7", salience_completion=0.4, verify_completion=0.8),
                self.write_summary(root, "seed11", salience_completion=0.2, verify_completion=0.6),
            ]

            rows = aggregate_summaries(paths)

        verify = next(row for row in rows if row["policy"] == "verify_interaction_risk")
        self.assertEqual(verify["seeds"], 2)
        self.assertEqual(verify["completion_rate_mean"], 0.7)
        self.assertEqual(verify["interaction_stale_errors_mean"], 0.0)
        self.assertEqual(verify["pickup_tasks_mean"], 7.0)

    def test_delta_vs_baseline_uses_interaction_stale_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = aggregate_summaries([self.write_summary(root, "seed7", 0.4, 0.8)])

            delta = delta_vs_baseline(rows)[0]

        self.assertEqual(delta["completion_rate_delta"], 0.4)
        self.assertEqual(delta["interaction_stale_errors_delta"], -3.0)

    def test_write_outputs_creates_delta_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = aggregate_summaries([self.write_summary(root, "seed7", 0.4, 0.8)])
            out_dir = root / "out"

            write_outputs(rows, out_dir)

            self.assertTrue((out_dir / "ai2thor_interaction_multiseed_delta.csv").exists())


if __name__ == "__main__":
    unittest.main()
