import csv
import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_reliability import (
    budget_delta_rows,
    generate_reliability_outputs,
    main_budget_rows,
    threshold_sensitivity_rows,
    tradeoff_rows,
)


class AI2ThorRearrangementReliabilityTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames, rows) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def fixture_paths(self, root: Path):
        multiseed_summary = root / "multiseed_summary.csv"
        multiseed_delta = root / "multiseed_delta.csv"
        ablation_summary = root / "ablation_summary.csv"
        ablation_delta = root / "ablation_delta.csv"
        summary_fields = [
            "budget",
            "policy",
            "seeds",
            "completion_rate_mean",
            "completion_rate_std",
            "avg_reachable_cost_mean",
            "avg_reachable_cost_std",
            "stale_errors_mean",
            "stale_errors_std",
            "verifications_mean",
            "verification_catches_mean",
        ]
        self.write_csv(
            multiseed_summary,
            summary_fields,
            [
                {
                    "budget": 16,
                    "policy": "salience",
                    "seeds": 5,
                    "completion_rate_mean": 0.3467,
                    "completion_rate_std": 0.1204,
                    "avg_reachable_cost_mean": 8.9636,
                    "avg_reachable_cost_std": 1.4857,
                    "stale_errors_mean": 2.6,
                    "stale_errors_std": 1.5166,
                    "verifications_mean": 0.0,
                    "verification_catches_mean": 0.0,
                },
                {
                    "budget": 16,
                    "policy": "verify_rearrangement_risk",
                    "seeds": 5,
                    "completion_rate_mean": 0.8171,
                    "completion_rate_std": 0.156,
                    "avg_reachable_cost_mean": 8.2622,
                    "avg_reachable_cost_std": 1.2402,
                    "stale_errors_mean": 0.0,
                    "stale_errors_std": 0.0,
                    "verifications_mean": 4.6,
                    "verification_catches_mean": 2.6,
                },
                {
                    "budget": 32,
                    "policy": "verify_rearrangement_risk",
                    "seeds": 5,
                    "completion_rate_mean": 1.0,
                    "completion_rate_std": 0.0,
                    "avg_reachable_cost_mean": 7.7145,
                    "avg_reachable_cost_std": 2.0552,
                    "stale_errors_mean": 0.0,
                    "stale_errors_std": 0.0,
                    "verifications_mean": 5.8,
                    "verification_catches_mean": 3.4,
                },
            ],
        )
        self.write_csv(
            multiseed_delta,
            [
                "budget",
                "seeds",
                "completion_rate_delta",
                "avg_reachable_cost_delta",
                "stale_errors_delta",
                "verifications_delta",
                "verification_catches_delta",
            ],
            [
                {
                    "budget": 16,
                    "seeds": 5,
                    "completion_rate_delta": 0.4704,
                    "avg_reachable_cost_delta": -0.7014,
                    "stale_errors_delta": -2.6,
                    "verifications_delta": 4.6,
                    "verification_catches_delta": 2.6,
                },
                {
                    "budget": 32,
                    "seeds": 5,
                    "completion_rate_delta": 0.5848,
                    "avg_reachable_cost_delta": -0.8386,
                    "stale_errors_delta": -3.4,
                    "verifications_delta": 5.8,
                    "verification_catches_delta": 3.4,
                },
            ],
        )
        ablation_fields = [
            "budget",
            "verification_cost",
            "ablation",
            "policy",
            "seeds",
            "completion_rate_mean",
            "completion_rate_std",
            "avg_reachable_cost_mean",
            "avg_reachable_cost_std",
            "stale_errors_mean",
            "stale_errors_std",
            "verifications_mean",
            "verification_catches_mean",
        ]
        self.write_csv(
            ablation_summary,
            ablation_fields,
            [
                {
                    "budget": 16,
                    "verification_cost": 0.5,
                    "ablation": "verify_threshold_0p2",
                    "policy": "verify_threshold_0p2",
                    "seeds": 5,
                    "completion_rate_mean": 0.8171,
                    "completion_rate_std": 0.156,
                    "avg_reachable_cost_mean": 8.2622,
                    "avg_reachable_cost_std": 1.2402,
                    "stale_errors_mean": 0.0,
                    "stale_errors_std": 0.0,
                    "verifications_mean": 4.6,
                    "verification_catches_mean": 2.6,
                },
                {
                    "budget": 16,
                    "verification_cost": 0.5,
                    "ablation": "verify_threshold_0p5",
                    "policy": "verify_threshold_0p5",
                    "seeds": 5,
                    "completion_rate_mean": 0.6914,
                    "completion_rate_std": 0.1823,
                    "avg_reachable_cost_mean": 8.3146,
                    "avg_reachable_cost_std": 1.348,
                    "stale_errors_mean": 0.8,
                    "stale_errors_std": 0.8367,
                    "verifications_mean": 2.0,
                    "verification_catches_mean": 1.8,
                },
            ],
        )
        self.write_csv(
            ablation_delta,
            [
                "budget",
                "verification_cost",
                "ablation",
                "completion_rate_delta",
                "avg_reachable_cost_delta",
                "stale_errors_delta",
                "verification_catches_delta",
            ],
            [
                {
                    "budget": 16,
                    "verification_cost": 0.5,
                    "ablation": "verify_threshold_0p2",
                    "completion_rate_delta": 0.4704,
                    "avg_reachable_cost_delta": -0.7014,
                    "stale_errors_delta": -2.6,
                    "verification_catches_delta": 2.6,
                },
                {
                    "budget": 16,
                    "verification_cost": 0.5,
                    "ablation": "verify_threshold_0p5",
                    "completion_rate_delta": 0.3447,
                    "avg_reachable_cost_delta": -0.649,
                    "stale_errors_delta": -1.8,
                    "verification_catches_delta": 1.8,
                },
            ],
        )
        return multiseed_summary, multiseed_delta, ablation_summary, ablation_delta

    def test_extracts_budget_mean_std_and_delta_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture_paths(Path(tmp))

            budget_rows = main_budget_rows(paths[0])
            deltas = budget_delta_rows(paths[1])

        self.assertEqual(len(budget_rows), 3)
        self.assertEqual(budget_rows[1]["completion_rate_mean_std"], "0.8171 +/- 0.1560")
        self.assertEqual(deltas[-1]["budget"], 32)
        self.assertAlmostEqual(deltas[-1]["completion_rate_delta"], 0.5848)
        self.assertAlmostEqual(deltas[-1]["completion_rate_delta_trend"], 0.1144)

    def test_extracts_tradeoff_and_threshold_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture_paths(Path(tmp))

            tradeoffs = tradeoff_rows(paths[2], paths[3])
            thresholds = threshold_sensitivity_rows(paths[2])

        self.assertEqual(len(tradeoffs), 2)
        self.assertAlmostEqual(tradeoffs[0]["completion_rate_delta"], 0.4704)
        self.assertEqual(tradeoffs[0]["completion_rate_mean_std"], "0.8171 +/- 0.1560")
        self.assertEqual([row["threshold"] for row in thresholds], ["0.2", "0.5"])

    def test_generate_outputs_writes_reliability_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture_paths(root)
            out_dir = root / "out"

            files = generate_reliability_outputs(*paths, out_dir)

            self.assertIn("main_budget_mean_std.csv", files)
            self.assertIn("multi_budget_delta.csv", files)
            self.assertTrue(any(name.startswith("fig_rearrangement_budget_reliability.") for name in files))
            self.assertTrue(any(name.startswith("fig_rearrangement_cost_threshold_tradeoff.") for name in files))
            self.assertTrue((out_dir / "README.md").exists())
            self.assertIn("Budget 32 extension", (out_dir / "README.md").read_text(encoding="utf-8"))
            self.assertIn("Mean +/- std", (out_dir / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
