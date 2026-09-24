import unittest

from embodied_memory_pilot.pilot import aggregate, run_suite


class MemoryPilotTest(unittest.TestCase):
    def test_task_conditioned_beats_fifo_on_search_cost(self) -> None:
        rows = run_suite(seeds=range(3), budgets=[3], tasks=80)
        summary = {(row["budget"], row["policy"]): row for row in aggregate(rows)}

        fifo = summary[(3, "fifo")]
        task_conditioned = summary[(3, "task_conditioned")]
        oracle = summary[(3, "oracle")]

        self.assertLess(task_conditioned["avg_cost"], fifo["avg_cost"])
        self.assertLessEqual(oracle["avg_cost"], task_conditioned["avg_cost"])

    def test_outputs_include_all_policies(self) -> None:
        rows = run_suite(seeds=range(1), budgets=[2], tasks=20)
        policies = {row["policy"] for row in rows}
        self.assertEqual(
            policies,
            {
                "no_memory",
                "fifo",
                "salience",
                "task_conditioned",
                "freshness_aware",
                "oracle",
                "fresh_oracle",
            },
        )

    def test_freshness_aware_reduces_stale_errors_under_drift(self) -> None:
        rows = run_suite(seeds=range(5), budgets=[8], tasks=120, drift_scale=3.0, task_budget=5.0)
        summary = {(row["budget"], row["policy"]): row for row in aggregate(rows)}

        task_conditioned = summary[(8, "task_conditioned")]
        freshness_aware = summary[(8, "freshness_aware")]

        self.assertLessEqual(freshness_aware["stale_errors"], task_conditioned["stale_errors"])

    def test_budgeted_success_penalizes_no_memory_search(self) -> None:
        rows = run_suite(seeds=range(3), budgets=[4], tasks=80, task_budget=5.0)
        summary = {(row["budget"], row["policy"]): row for row in aggregate(rows)}

        no_memory = summary[(4, "no_memory")]
        task_conditioned = summary[(4, "task_conditioned")]

        self.assertLess(no_memory["success_rate"], task_conditioned["success_rate"])


if __name__ == "__main__":
    unittest.main()
