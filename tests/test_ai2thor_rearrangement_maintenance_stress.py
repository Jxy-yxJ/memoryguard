import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_benchmark import build_rearrangement_tasks
from embodied_memory_pilot.ai2thor_rearrangement_maintenance_stress import (
    build_selection_pressure_probe,
    run_selection_pressure_stress,
    selection_pressure_delta,
    summarize_stress,
    write_outputs,
)


class AI2ThorRearrangementMaintenanceStressTest(unittest.TestCase):
    def test_stress_probe_has_multiple_moved_tasks(self) -> None:
        probe = build_selection_pressure_probe()
        tasks = build_rearrangement_tasks(probe)

        self.assertGreaterEqual(len(tasks), 5)
        self.assertTrue(all(task.rearranged for task in tasks))

    def test_heuristic_beats_random_under_selection_pressure(self) -> None:
        _probe, rows = run_selection_pressure_stress(random_seeds=range(6))
        summary = summarize_stress(rows)
        by_policy = {str(row["maintenance_policy"]): row for row in summary}
        delta = selection_pressure_delta(summary)

        self.assertEqual(by_policy["heuristic_active"]["completion_rate_mean"], 1.0)
        self.assertGreater(float(delta["heuristic_vs_random_completion_delta"]), 0.0)
        self.assertLess(float(delta["heuristic_vs_random_stale_errors_delta"]), 0.0)
        self.assertGreater(float(by_policy["passive"]["stale_errors_mean"]), 0.0)

    def test_outputs_are_written(self) -> None:
        probe, rows = run_selection_pressure_stress(random_seeds=range(3))
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_outputs(probe, rows, out_dir)

            self.assertTrue((out_dir / "ai2thor_rearrangement_maintenance_stress.json").exists())
            summary_text = (out_dir / "ai2thor_rearrangement_maintenance_stress_summary.csv").read_text(encoding="utf-8")
            readme_text = (out_dir / "README.md").read_text(encoding="utf-8")

        self.assertIn("heuristic_active", summary_text)
        self.assertIn("controlled synthetic", readme_text)


if __name__ == "__main__":
    _ = unittest.main()
