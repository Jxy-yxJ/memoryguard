from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


MAIN_POLICIES = ("salience", "verify_rearrangement_risk")
SENSITIVITY_ABLATIONS = (
    "salience_no_verification",
    "detect_only_0p2",
    "verify_threshold_0p2",
    "verify_threshold_0p5",
    "verify_all",
)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def main_budget_rows(multiseed_summary: Path) -> List[Dict[str, object]]:
    rows = [row for row in _read_rows(multiseed_summary) if row["policy"] in MAIN_POLICIES]
    out: List[Dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (int(item["budget"]), item["policy"])):
        completion_mean = _float(row, "completion_rate_mean")
        completion_std = _float(row, "completion_rate_std")
        cost_mean = _float(row, "avg_reachable_cost_mean")
        cost_std = _float(row, "avg_reachable_cost_std")
        stale_mean = _float(row, "stale_errors_mean")
        stale_std = _float(row, "stale_errors_std")
        out.append(
            {
                "budget": int(row["budget"]),
                "policy": row["policy"],
                "seeds": int(row["seeds"]),
                "completion_rate_mean": completion_mean,
                "completion_rate_std": completion_std,
                "completion_rate_mean_std": _mean_std(completion_mean, completion_std),
                "avg_reachable_cost_mean": cost_mean,
                "avg_reachable_cost_std": cost_std,
                "avg_reachable_cost_mean_std": _mean_std(cost_mean, cost_std),
                "stale_errors_mean": stale_mean,
                "stale_errors_std": stale_std,
                "stale_errors_mean_std": _mean_std(stale_mean, stale_std),
                "verifications_mean": _float(row, "verifications_mean"),
                "verification_catches_mean": _float(row, "verification_catches_mean"),
            }
        )
    return out


def budget_delta_rows(multiseed_delta: Path) -> List[Dict[str, object]]:
    rows = _read_rows(multiseed_delta)
    out: List[Dict[str, object]] = []
    previous: Dict[str, str] | None = None
    for row in sorted(rows, key=lambda item: int(item["budget"])):
        completion_delta = _float(row, "completion_rate_delta")
        stale_delta = _float(row, "stale_errors_delta")
        out.append(
            {
            "budget": int(row["budget"]),
            "seeds": int(row["seeds"]),
            "completion_rate_delta": completion_delta,
            "avg_reachable_cost_delta": _float(row, "avg_reachable_cost_delta"),
            "stale_errors_delta": stale_delta,
            "verifications_delta": _float(row, "verifications_delta"),
            "verification_catches_delta": _float(row, "verification_catches_delta"),
            "completion_rate_delta_trend": "" if previous is None else completion_delta - _float(previous, "completion_rate_delta"),
            "stale_errors_delta_trend": "" if previous is None else stale_delta - _float(previous, "stale_errors_delta"),
        }
        )
        previous = row
    return out


def tradeoff_rows(ablation_summary: Path, ablation_delta: Path) -> List[Dict[str, object]]:
    deltas = {
        (int(row["budget"]), float(row["verification_cost"]), row["ablation"]): row
        for row in _read_rows(ablation_delta)
    }
    rows = [
        row
        for row in _read_rows(ablation_summary)
        if row["ablation"] in SENSITIVITY_ABLATIONS
    ]
    out: List[Dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (int(item["budget"]), float(item["verification_cost"]), item["ablation"])):
        delta = deltas.get((int(row["budget"]), float(row["verification_cost"]), row["ablation"]), {})
        completion_mean = _float(row, "completion_rate_mean")
        completion_std = _float(row, "completion_rate_std")
        cost_mean = _float(row, "avg_reachable_cost_mean")
        cost_std = _float(row, "avg_reachable_cost_std")
        stale_mean = _float(row, "stale_errors_mean")
        stale_std = _float(row, "stale_errors_std")
        out.append(
            {
                "budget": int(row["budget"]),
                "verification_cost": _float(row, "verification_cost"),
                "ablation": row["ablation"],
                "seeds": int(row["seeds"]),
                "completion_rate_mean": completion_mean,
                "completion_rate_std": completion_std,
                "completion_rate_mean_std": _mean_std(completion_mean, completion_std),
                "completion_rate_delta": float(delta["completion_rate_delta"]) if delta else "",
                "avg_reachable_cost_mean": cost_mean,
                "avg_reachable_cost_std": cost_std,
                "avg_reachable_cost_mean_std": _mean_std(cost_mean, cost_std),
                "avg_reachable_cost_delta": float(delta["avg_reachable_cost_delta"]) if delta else "",
                "stale_errors_mean": stale_mean,
                "stale_errors_std": stale_std,
                "stale_errors_mean_std": _mean_std(stale_mean, stale_std),
                "stale_errors_delta": float(delta["stale_errors_delta"]) if delta else "",
                "verifications_mean": _float(row, "verifications_mean"),
                "verification_catches_mean": _float(row, "verification_catches_mean"),
                "verification_catches_delta": float(delta["verification_catches_delta"]) if delta else "",
            }
        )
    return out


def threshold_sensitivity_rows(ablation_summary: Path, *, verification_cost: float = 0.5) -> List[Dict[str, object]]:
    rows = []
    for row in _read_rows(ablation_summary):
        ablation = row["ablation"]
        if not ablation.startswith("verify_threshold_"):
            continue
        if float(row["verification_cost"]) != verification_cost:
            continue
        rows.append(row)
    return [
        {
            "budget": int(row["budget"]),
            "threshold": row["ablation"].replace("verify_threshold_", "").replace("p", "."),
            "verification_cost": _float(row, "verification_cost"),
            "seeds": int(row["seeds"]),
            "completion_rate_mean": _float(row, "completion_rate_mean"),
            "completion_rate_std": _float(row, "completion_rate_std"),
            "avg_reachable_cost_mean": _float(row, "avg_reachable_cost_mean"),
            "avg_reachable_cost_std": _float(row, "avg_reachable_cost_std"),
            "stale_errors_mean": _float(row, "stale_errors_mean"),
            "stale_errors_std": _float(row, "stale_errors_std"),
            "verifications_mean": _float(row, "verifications_mean"),
            "verification_catches_mean": _float(row, "verification_catches_mean"),
        }
        for row in sorted(rows, key=lambda item: (int(item["budget"]), item["ablation"]))
    ]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, out_dir: Path, stem: str) -> List[str]:
    files: List[str] = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path)
        files.append(path.name)
    return files


def _write_svg_placeholder(out_dir: Path, stem: str, title: str, lines: Sequence[str]) -> str:
    escaped_lines = [str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="36" y="52" font-family="serif" font-size="22">{title}</text>',
    ]
    for index, line in enumerate(escaped_lines[:12]):
        body.append(f'<text x="36" y="{92 + index * 24}" font-family="serif" font-size="15">{line}</text>')
    body.append("</svg>")
    path = out_dir / f"{stem}.svg"
    path.write_text("\n".join(body), encoding="utf-8")
    return path.name


def plot_budget_reliability(rows: Sequence[Dict[str, object]], out_dir: Path) -> List[str]:
    if not _matplotlib_available():
        lines = [
            f"budget={row['budget']} policy={row['policy']} completion={row['completion_rate_mean_std']} stale={row['stale_errors_mean_std']}"
            for row in rows
        ]
        return [_write_svg_placeholder(out_dir, "fig_rearrangement_budget_reliability", "Budget reliability", lines)]
    plt = _setup_matplotlib()
    by_policy: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    specs = [
        ("completion_rate", "Completion"),
        ("avg_reachable_cost", "Avg. reachable cost"),
        ("stale_errors", "Stale errors"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs):
        for policy, values in by_policy.items():
            values = sorted(values, key=lambda item: int(item["budget"]))
            budgets = [int(row["budget"]) for row in values]
            means = [float(row[f"{metric}_mean"]) for row in values]
            stds = [float(row[f"{metric}_std"]) for row in values]
            ax.errorbar(budgets, means, yerr=stds, marker="o", linewidth=1.6, capsize=3, label=policy)
        ax.set_xlabel("Memory budget")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    files = _save(fig, out_dir, "fig_rearrangement_budget_reliability")
    plt.close(fig)
    return files


def plot_tradeoff(rows: Sequence[Dict[str, object]], out_dir: Path, *, budget: int = 16) -> List[str]:
    if not _matplotlib_available():
        lines = [
            f"budget={row['budget']} cverify={row['verification_cost']} ablation={row['ablation']} completion={row['completion_rate_mean_std']}"
            for row in rows
        ]
        return [_write_svg_placeholder(out_dir, "fig_rearrangement_cost_threshold_tradeoff", "Threshold / cverify trade-off", lines)]
    plt = _setup_matplotlib()
    selected = [row for row in rows if int(row["budget"]) == budget]
    by_ablation: Dict[str, List[Dict[str, object]]] = {}
    for row in selected:
        by_ablation.setdefault(str(row["ablation"]), []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ablation, values in by_ablation.items():
        values = sorted(values, key=lambda item: float(item["verification_cost"]))
        costs = [float(row["verification_cost"]) for row in values]
        completion = [float(row["completion_rate_mean"]) for row in values]
        stale = [float(row["stale_errors_mean"]) for row in values]
        axes[0].plot(costs, completion, marker="o", linewidth=1.5, label=ablation)
        axes[1].plot(costs, stale, marker="o", linewidth=1.5, label=ablation)
    axes[0].set_xlabel("Verification cost")
    axes[0].set_ylabel("Completion")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_xlabel("Verification cost")
    axes[1].set_ylabel("Stale errors")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    files = _save(fig, out_dir, "fig_rearrangement_cost_threshold_tradeoff")
    plt.close(fig)
    return files


def render_readme(
    budget_rows: Sequence[Dict[str, object]],
    deltas: Sequence[Dict[str, object]],
    threshold_rows: Sequence[Dict[str, object]],
) -> str:
    budget16 = [row for row in budget_rows if int(row["budget"]) == 16]
    budget32 = [row for row in budget_rows if int(row["budget"]) == 32]
    verify16 = next((row for row in budget16 if row["policy"] == "verify_rearrangement_risk"), None)
    salience16 = next((row for row in budget16 if row["policy"] == "salience"), None)
    verify32 = next((row for row in budget32 if row["policy"] == "verify_rearrangement_risk"), None)
    delta_summary = ", ".join(
        f"{row['budget']}:{row['completion_rate_delta']:+.4f}" for row in deltas
    )
    lines = [
        "# AI2-THOR Rearrangement Reliability Summary",
        "",
        "This directory is generated from saved CSV summaries only; it does not start Unity.",
        "",
        "Mean +/- std text columns are included in `main_budget_mean_std.csv`, `verification_cost_threshold_tradeoff.csv`, and `threshold_sensitivity_cost0p5.csv`.",
        "",
    ]
    if salience16 and verify16:
        lines.extend(
            [
                "## Budget 16 mean +/- std",
                "",
                f"- Salience completion `{salience16['completion_rate_mean']:.4f} +/- {salience16['completion_rate_std']:.4f}`, stale errors `{salience16['stale_errors_mean']:.4f} +/- {salience16['stale_errors_std']:.4f}`.",
                f"- Verify-risk completion `{verify16['completion_rate_mean']:.4f} +/- {verify16['completion_rate_std']:.4f}`, stale errors `{verify16['stale_errors_mean']:.4f} +/- {verify16['stale_errors_std']:.4f}`.",
                "",
            ]
        )
    if verify32:
        lines.extend(
            [
                "## Budget 32 extension",
                "",
                f"- Verify-risk reaches completion `{verify32['completion_rate_mean']:.4f} +/- {verify32['completion_rate_std']:.4f}` with stale errors `{verify32['stale_errors_mean']:.4f} +/- {verify32['stale_errors_std']:.4f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Multi-budget completion delta vs salience",
            "",
            f"- {delta_summary}",
            "",
        ]
    )
    budget16_thresholds = [row for row in threshold_rows if int(row["budget"]) == 16]
    if budget16_thresholds:
        threshold_summary = ", ".join(
            f"tau={row['threshold']}:completion={row['completion_rate_mean']:.4f},stale={row['stale_errors_mean']:.4f}"
            for row in budget16_thresholds
        )
        lines.extend(
            [
                "## Threshold sensitivity at verification cost 0.5",
                "",
                f"- {threshold_summary}",
                "",
            ]
        )
    return "\n".join(lines)


def generate_reliability_outputs(
    multiseed_summary: Path,
    multiseed_delta: Path,
    ablation_summary: Path,
    ablation_delta: Path,
    out_dir: Path,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = main_budget_rows(multiseed_summary)
    deltas = budget_delta_rows(multiseed_delta)
    tradeoffs = tradeoff_rows(ablation_summary, ablation_delta)
    thresholds = threshold_sensitivity_rows(ablation_summary)

    _write_csv(out_dir / "main_budget_mean_std.csv", budgets)
    _write_csv(out_dir / "multi_budget_delta.csv", deltas)
    _write_csv(out_dir / "verification_cost_threshold_tradeoff.csv", tradeoffs)
    _write_csv(out_dir / "threshold_sensitivity_cost0p5.csv", thresholds)
    (out_dir / "ai2thor_rearrangement_reliability.json").write_text(
        json.dumps(
            {
                "main_budget_mean_std": budgets,
                "multi_budget_delta": deltas,
                "verification_cost_threshold_tradeoff": tradeoffs,
                "threshold_sensitivity_cost0p5": thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(render_readme(budgets, deltas, thresholds), encoding="utf-8")

    files = [
        "main_budget_mean_std.csv",
        "multi_budget_delta.csv",
        "verification_cost_threshold_tradeoff.csv",
        "threshold_sensitivity_cost0p5.csv",
        "ai2thor_rearrangement_reliability.json",
        "README.md",
    ]
    files.extend(plot_budget_reliability(budgets, out_dir))
    files.extend(plot_tradeoff(tradeoffs, out_dir))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reliability and sensitivity summaries for AI2-THOR rearrangement replay.")
    parser.add_argument(
        "--multiseed-summary",
        type=Path,
        default=Path("results/ai2thor_rearrangement_multiseed_5seed/ai2thor_rearrangement_multiseed_summary.csv"),
    )
    parser.add_argument(
        "--multiseed-delta",
        type=Path,
        default=Path("results/ai2thor_rearrangement_multiseed_5seed/ai2thor_rearrangement_multiseed_delta.csv"),
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=Path("results/ai2thor_rearrangement_ablation_5seed/ai2thor_rearrangement_ablation_summary.csv"),
    )
    parser.add_argument(
        "--ablation-delta",
        type=Path,
        default=Path("results/ai2thor_rearrangement_ablation_5seed/ai2thor_rearrangement_ablation_delta.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_rearrangement_reliability_5seed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = generate_reliability_outputs(
        args.multiseed_summary,
        args.multiseed_delta,
        args.ablation_summary,
        args.ablation_delta,
        args.out_dir,
    )
    print("\n".join(str(args.out_dir / name) for name in files))


if __name__ == "__main__":
    main()
