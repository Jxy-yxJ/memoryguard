import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_multiseed import aggregate_summaries, write_outputs


class AI2ThorRearrangementMultiseedTest(unittest.TestCase):
    def write_summary(self, root: Path, seed: str, salience_completion: float, verify_completion: float) -> Path:
        path = root / seed / "ai2thor_rearrangement_verification_summary.csv"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                [
                    "budget,policy,runs,rearrangement_tasks,moved_tasks,memory_items,retained_items,completion_rate,avg_reachable_cost,query_hit_rate,stale_errors,verifications,verification_catches,verification_cost,saved_reachable_cost",
                    f"16,salience,1,7,4,67,16,{salience_completion},8.6,0.4,3,0,0,0.5,19.8",
                    f"16,verify_rearrangement_risk,1,7,4,67,16,{verify_completion},8.1,0.4,0,6,3,0.5,18.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_aggregate_summaries_reports_mean_std_and_seed_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                self.write_summary(root, "seed7", salience_completion=0.4, verify_completion=0.8),
                self.write_summary(root, "seed11", salience_completion=0.2, verify_completion=0.6),
            ]

            rows = aggregate_summaries(paths)

        verify = next(row for row in rows if row["policy"] == "verify_rearrangement_risk")
        salience = next(row for row in rows if row["policy"] == "salience")
        self.assertEqual(verify["seeds"], 2)
        self.assertEqual(verify["completion_rate_mean"], 0.7)
        self.assertGreater(verify["completion_rate_std"], 0.0)
        self.assertEqual(salience["stale_errors_mean"], 3.0)

    def test_write_outputs_includes_verify_minus_salience_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = self.write_summary(root, "seed7", salience_completion=0.4, verify_completion=0.8)
            out_dir = root / "out"

            write_outputs(aggregate_summaries([input_path]), out_dir)

            text = (out_dir / "ai2thor_rearrangement_multiseed_delta.csv").read_text(encoding="utf-8")

        self.assertIn("completion_rate_delta", text)
        self.assertIn("0.4", text)


if __name__ == "__main__":
    unittest.main()
