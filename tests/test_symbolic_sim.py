import unittest

from embodied_memory_pilot.symbolic_sim import aggregate, run_symbolic_suite


class SymbolicSimTest(unittest.TestCase):
    def test_symbolic_memory_beats_no_memory_with_budget(self) -> None:
        rows = run_symbolic_suite(seeds=range(3), budgets=[8], tasks=80, task_budget=6.0)
        summary = {(row["budget"], row["policy"]): row for row in aggregate(rows)}

        self.assertGreater(summary[(8, "freshness_aware")]["success_rate"], summary[(8, "no_memory")]["success_rate"])

    def test_symbolic_stress_tracks_stale_errors(self) -> None:
        rows = run_symbolic_suite(seeds=range(3), budgets=[8], tasks=80, drift_scale=3.0, task_budget=6.0)
        summary = {(row["budget"], row["policy"]): row for row in aggregate(rows)}

        self.assertGreaterEqual(summary[(8, "task_conditioned")]["stale_errors"], 0)
        self.assertIn("verification_catches", summary[(8, "freshness_aware")])


if __name__ == "__main__":
    unittest.main()
