"""Regenerate the 0514 component-ablation figure from the matched base counterfactual curve.

Source of truth: results/ai2thor_live_gsam_wave3_4_budget6_visual_prior_v1/component_ablation_current_base_curve.csv
(current 6, density_only 6, visual_only 0, combined 0 aggregate selected strict FNs over budgets 1-6).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SOURCE = Path(
    "results/ai2thor_live_gsam_wave3_4_budget6_visual_prior_v1/component_ablation_current_base_curve.csv"
)
OUTPUTS = (
    Path("figures/fig_0514_component_ablation.pdf"),
    Path("figures/fig_0514_component_ablation.png"),
    Path("paper/figures/fig_0514_component_ablation.pdf"),
)
VARIANTS = (
    ("current", "Current EV", "o", "-"),
    ("density_only", "Density only", "s", "--"),
    ("visual_only", "Visual prior only", "^", "-."),
    ("combined", "Combined", "D", ":"),
)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    budgets = [int(row["budget"]) for row in rows]

    figure, axis = plt.subplots(figsize=(4.4, 2.9))
    for column, label, marker, linestyle in VARIANTS:
        axis.plot(
            budgets,
            [int(row[column]) for row in rows],
            marker=marker,
            linestyle=linestyle,
            markersize=5,
            linewidth=1.6,
            label=label,
        )
    axis.set_xlabel("Verification budget")
    axis.set_ylabel("Selected strict FNs")
    axis.set_xticks(budgets)
    axis.set_ylim(-0.3, 3.5)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    print("wrote", ", ".join(str(path) for path in OUTPUTS))


if __name__ == "__main__":
    main()
