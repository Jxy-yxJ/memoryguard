import csv
import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_paper_figures import (
    _format_location_for_box,
    extract_stale_case,
    generate_paper_figures,
    load_ablation_summary,
    pareto_rows,
    render_latex_includes,
)


_PAPER_FIGURE_PROBES = Path("results/ai2thor_rearrangement_6scene_seed7/ai2thor_rearrangement_probe.json")
_SKIP_REASON = "local rearrangement probe artifacts are not included in the repository"


class AI2ThorRearrangementPaperFiguresTest(unittest.TestCase):
    def test_load_ablation_summary_converts_numeric_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            path.write_text(
                "\n".join(
                    [
                        "budget,verification_cost,ablation,policy,seeds,rearrangement_tasks_mean,moved_tasks_mean,completion_rate_mean,completion_rate_std,avg_reachable_cost_mean,avg_reachable_cost_std,stale_errors_mean,stale_errors_std,verifications_mean,verification_catches_mean,recover_caught_stale",
                        "16,0.5,verify_threshold_0p2,verify_threshold_0p2,2,7.0,3.5,0.7143,0.202,7.9393,0.3182,0.0,0.0,5.0,2.0,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_ablation_summary(path)

        self.assertEqual(rows[0]["budget"], 16)
        self.assertEqual(rows[0]["seeds"], 2)
        self.assertAlmostEqual(float(rows[0]["verification_cost"]), 0.5)
        self.assertAlmostEqual(float(rows[0]["completion_rate_mean"]), 0.7143)

    def test_pareto_rows_filters_budget_and_paper_ablations(self) -> None:
        rows = [
            {"budget": 16, "verification_cost": 0.5, "ablation": "salience_no_verification"},
            {"budget": 16, "verification_cost": 0.5, "ablation": "verify_threshold_0p2"},
            {"budget": 8, "verification_cost": 0.5, "ablation": "verify_threshold_0p2"},
            {"budget": 16, "verification_cost": 0.5, "ablation": "detect_only_0p1"},
        ]

        selected = pareto_rows(rows)

        self.assertEqual([row["ablation"] for row in selected], ["salience_no_verification", "verify_threshold_0p2"])

    @unittest.skipUnless(_PAPER_FIGURE_PROBES.exists(), _SKIP_REASON)
    def test_extract_stale_case_finds_thresholded_verification_catch(self) -> None:
        case = extract_stale_case(
            [
                Path("results/ai2thor_rearrangement_6scene_seed7/ai2thor_rearrangement_probe.json"),
                Path("results/ai2thor_rearrangement_6scene_seed11/ai2thor_rearrangement_probe.json"),
            ]
        )

        self.assertEqual(case["salience_outcome"], "stale_error")
        self.assertEqual(case["verification_outcome"], "stale_caught_then_scene_search_recovery")
        self.assertGreater(float(case["risk_score"]), 0.2)
        self.assertIn("@before", str(case["selected_before_location"]))
        self.assertIn("@after", str(case["true_after_location"]))

    def test_latex_includes_reference_generated_paper_figures(self) -> None:
        includes = render_latex_includes(
            [
                "fig_rearrangement_verification_pareto.pdf",
                "fig_rearrangement_stale_case.pdf",
            ],
            seeds=5,
            total_tasks=29,
        )

        self.assertIn("fig_rearrangement_verification_pareto.pdf", includes)
        self.assertIn("fig:rearrangement-verification-pareto", includes)
        self.assertIn("fig_rearrangement_stale_case.pdf", includes)
        self.assertIn("5 saved AI2-THOR rearrangement probes and 29 total tasks", includes)

    def test_format_location_for_box_breaks_long_ai2thor_ids(self) -> None:
        label = _format_location_for_box("FloorPlan3:Pot|-01.58|+01.31|-01.58@before")
        lines = label.splitlines()

        self.assertEqual(lines[0], "Pot @ before")
        self.assertIn("x=-01.58", label)
        self.assertIn("z=-01.58", label)
        self.assertTrue(all(len(line) <= 24 for line in lines), label)

    @unittest.skipUnless(_PAPER_FIGURE_PROBES.exists(), _SKIP_REASON)
    def test_generate_paper_figures_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            out_dir = root / "figures"
            fieldnames = [
                "budget",
                "verification_cost",
                "ablation",
                "policy",
                "seeds",
                "rearrangement_tasks_mean",
                "moved_tasks_mean",
                "completion_rate_mean",
                "completion_rate_std",
                "avg_reachable_cost_mean",
                "avg_reachable_cost_std",
                "stale_errors_mean",
                "stale_errors_std",
                "verifications_mean",
                "verification_catches_mean",
                "recover_caught_stale",
            ]
            rows = [
                {
                    "budget": 16,
                    "verification_cost": cost,
                    "ablation": ablation,
                    "policy": ablation,
                    "seeds": 2,
                    "rearrangement_tasks_mean": 7.0,
                    "moved_tasks_mean": 3.5,
                    "completion_rate_mean": completion,
                    "completion_rate_std": 0.0,
                    "avg_reachable_cost_mean": avg_cost,
                    "avg_reachable_cost_std": 0.0,
                    "stale_errors_mean": stale,
                    "stale_errors_std": 0.0,
                    "verifications_mean": verifications,
                    "verification_catches_mean": catches,
                    "recover_caught_stale": 1,
                }
                for cost, ablation, completion, avg_cost, stale, verifications, catches in [
                    (0.0, "salience_no_verification", 0.4286, 8.2072, 2.0, 0.0, 0.0),
                    (0.5, "salience_no_verification", 0.4286, 8.2072, 2.0, 0.0, 0.0),
                    (0.0, "detect_only_0p2", 0.4286, 7.5822, 0.0, 5.0, 2.0),
                    (0.5, "detect_only_0p2", 0.4286, 7.9393, 0.0, 5.0, 2.0),
                    (0.0, "verify_threshold_0p2", 0.7143, 7.5822, 0.0, 5.0, 2.0),
                    (0.5, "verify_threshold_0p2", 0.7143, 7.9393, 0.0, 5.0, 2.0),
                    (0.0, "verify_threshold_0p5", 0.6429, 7.7, 0.5, 2.0, 0.5),
                    (0.5, "verify_threshold_0p5", 0.6429, 7.9, 0.5, 2.0, 0.5),
                    (0.0, "verify_all", 0.7143, 7.5822, 0.0, 5.0, 2.0),
                    (0.5, "verify_all", 0.7143, 7.9393, 0.0, 5.0, 2.0),
                ]
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            files = generate_paper_figures(
                summary,
                [Path("results/ai2thor_rearrangement_6scene_seed7/ai2thor_rearrangement_probe.json")],
                out_dir,
            )

            self.assertIn("fig_rearrangement_verification_pareto.pdf", files)
            self.assertIn("fig_rearrangement_stale_case.pdf", files)
            self.assertTrue((out_dir / "stale_memory_case.json").exists())
            self.assertTrue((out_dir / "rearrangement_paper_latex_includes.tex").exists())


if __name__ == "__main__":
    unittest.main()
