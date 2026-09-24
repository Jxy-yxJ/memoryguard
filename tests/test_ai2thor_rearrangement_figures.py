import tempfile
import unittest
from pathlib import Path

from embodied_memory_pilot.ai2thor_rearrangement_figures import (
    generate_figures,
    load_multiseed_rows,
    render_latex_includes,
    write_latex_includes,
)


class AI2ThorRearrangementFiguresTest(unittest.TestCase):
    def test_load_multiseed_rows_filters_paper_policies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            path.write_text(
                "\n".join(
                    [
                        "budget,policy,seeds,completion_rate_mean,completion_rate_std,avg_reachable_cost_mean,avg_reachable_cost_std,stale_errors_mean,stale_errors_std,verifications_mean,verifications_std,verification_catches_mean,verification_catches_std",
                        "8,salience,2,0.3,0.1,8.6,1.0,1.0,1.4,0,0,0,0",
                        "8,verify_rearrangement_risk,2,0.5,0.1,8.5,1.0,0.0,0.0,3.5,0.7,1.0,1.4",
                        "8,no_memory,2,0.0,0.0,10.1,0.6,0.0,0.0,0,0,0,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_multiseed_rows(path, policies=("salience", "verify_rearrangement_risk"))

        self.assertEqual([row["policy"] for row in rows], ["salience", "verify_rearrangement_risk"])
        self.assertEqual(rows[1]["completion_rate_mean"], 0.5)

    def test_latex_includes_reference_expected_figure_files(self) -> None:
        includes = render_latex_includes(
            [
                "fig_rearrangement_completion.pdf",
                "fig_rearrangement_stale_errors.pdf",
                "fig_rearrangement_delta.pdf",
            ]
        )

        self.assertIn("\\includegraphics", includes)
        self.assertIn("fig_rearrangement_completion.pdf", includes)
        self.assertIn("fig:rearrangement-completion", includes)

    def test_write_latex_includes_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_latex_includes(Path(tmp), ["fig_rearrangement_completion.pdf"])

            self.assertTrue(path.exists())
            self.assertIn("fig_rearrangement_completion.pdf", path.read_text(encoding="utf-8"))

    def test_generate_figures_creates_completion_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            delta = root / "delta.csv"
            out_dir = root / "figures"
            summary.write_text(
                "\n".join(
                    [
                        "budget,policy,seeds,completion_rate_mean,completion_rate_std,avg_reachable_cost_mean,avg_reachable_cost_std,query_hit_rate_mean,query_hit_rate_std,stale_errors_mean,stale_errors_std,verifications_mean,verifications_std,verification_catches_mean,verification_catches_std",
                        "8,salience,2,0.3,0.1,8.6,1.0,0.3,0.1,1.0,1.4,0,0,0,0",
                        "8,verify_rearrangement_risk,2,0.5,0.1,8.5,1.0,0.3,0.1,0.0,0.0,3.5,0.7,1.0,1.4",
                        "16,salience,2,0.4,0.0,8.2,0.6,0.4,0.0,2.0,1.4,0,0,0,0",
                        "16,verify_rearrangement_risk,2,0.7,0.2,7.9,0.3,0.4,0.0,0.0,0.0,5,1.4,2,1.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            delta.write_text(
                "\n".join(
                    [
                        "budget,method_policy,baseline_policy,seeds,completion_rate_delta,avg_reachable_cost_delta,stale_errors_delta,verifications_delta,verification_catches_delta",
                        "8,verify_rearrangement_risk,salience,2,0.2,-0.1,-1,3.5,1",
                        "16,verify_rearrangement_risk,salience,2,0.3,-0.3,-2,5,2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            files = generate_figures(summary, delta, out_dir)

            completion_files = [name for name in files if name.startswith("fig_rearrangement_completion.")]
            self.assertTrue(completion_files)
            self.assertTrue((out_dir / completion_files[0]).exists())
            self.assertTrue((out_dir / "rearrangement_latex_includes.tex").exists())


if __name__ == "__main__":
    unittest.main()
