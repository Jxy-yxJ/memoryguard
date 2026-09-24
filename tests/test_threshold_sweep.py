import unittest

from embodied_memory_pilot.threshold_sweep import (
    TunableFreshnessPolicy,
    find_pareto_front,
    run_threshold_sweep,
)


class ThresholdSweepTest(unittest.TestCase):
    def test_tunable_policy_name_records_threshold(self) -> None:
        policy = TunableFreshnessPolicy(threshold=0.25)

        self.assertEqual(policy.name, "freshness_threshold_0.25")

    def test_threshold_sweep_reports_best_score(self) -> None:
        result = run_threshold_sweep(
            thresholds=[0.2, 0.4],
            seeds=range(2),
            budgets=[8],
            tasks=60,
            drift_scale=3.0,
            mode="symbolic",
        )

        self.assertEqual(len(result["summary"]), 2)
        self.assertIn("best", result)
        self.assertIn(result["best"]["threshold"], {0.2, 0.4})
        for row in result["summary"]:
            self.assertIn("score", row)
            self.assertIn("stale_errors", row)
            self.assertEqual(row["policy"], f"freshness_threshold_{row['threshold']:.2f}")

    def test_pareto_front_removes_dominated_rows(self) -> None:
        rows = [
            {"threshold": 0.2, "success_rate": 0.8, "avg_cost": 3.0, "stale_errors": 4.0},
            {"threshold": 0.4, "success_rate": 0.8, "avg_cost": 2.5, "stale_errors": 3.0},
            {"threshold": 0.6, "success_rate": 0.75, "avg_cost": 2.0, "stale_errors": 2.0},
        ]

        front = find_pareto_front(rows)

        self.assertEqual([row["threshold"] for row in front], [0.4, 0.6])


if __name__ == "__main__":
    unittest.main()
