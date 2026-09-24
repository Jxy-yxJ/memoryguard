from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PAPER_POLICIES = ("salience", "freshness_aware", "verify_rearrangement_risk")
POLICY_LABELS = {
    "salience": "Salience",
    "freshness_aware": "Freshness-aware",
    "verify_rearrangement_risk": "Verify risk",
}


def load_multiseed_rows(path: Path, policies: Sequence[str] = PAPER_POLICIES) -> List[Dict[str, float | int | str]]:
    policy_set = set(policies)
    rows: List[Dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["policy"] not in policy_set:
                continue
            row: Dict[str, float | int | str] = {"policy": raw["policy"], "budget": int(raw["budget"]), "seeds": int(raw["seeds"])}
            for key, value in raw.items():
                if key in {"policy", "budget", "seeds"}:
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def load_delta_rows(path: Path) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, float | int | str] = {
                "budget": int(raw["budget"]),
                "method_policy": raw["method_policy"],
                "baseline_policy": raw["baseline_policy"],
                "seeds": int(raw["seeds"]),
            }
            for key, value in raw.items():
                if key in row:
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _by_policy(rows: Sequence[Dict[str, float | int | str]]) -> Dict[str, List[Dict[str, float | int | str]]]:
    grouped: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["budget"]))
    return grouped


def _save(fig, out_dir: Path, stem: str) -> List[str]:
    paths: List[str] = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path)
        paths.append(path.name)
    return paths


def _svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value == min_value:
        return (start + end) / 2.0
    return start + (value - min_value) / (max_value - min_value) * (end - start)


def _write_svg_line_plot(
    out_dir: Path,
    stem: str,
    series: Dict[str, List[tuple[int, float]]],
    *,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    width = 720
    height = 440
    left = 78
    right = 30
    top = 28
    bottom = 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    budgets = sorted({budget for values in series.values() for budget, _ in values})
    y_values = [value for values in series.values() for _, value in values]
    low = min(y_values) if y_min is None else y_min
    high = max(y_values) if y_max is None else y_max
    if low == high:
        high = low + 1.0
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    def x_pos(budget: int) -> float:
        return _scale(float(budget), float(min(budgets)), float(max(budgets)), left, left + plot_w)

    def y_pos(value: float) -> float:
        return _scale(value, low, high, top + plot_h, top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle" font-family="serif" font-size="16">Memory budget</text>',
        f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" font-family="serif" font-size="16" transform="rotate(-90 20 {top + plot_h / 2})">{_svg_escape(y_label)}</text>',
    ]
    for tick in budgets:
        x = x_pos(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="serif" font-size="13">{tick}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + (high - low) * frac
        y = y_pos(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="serif" font-size="12">{value:.2f}</text>')
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = " ".join(f"{x_pos(budget):.2f},{y_pos(value):.2f}" for budget, value in values)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for budget, value in values:
            lines.append(f'<circle cx="{x_pos(budget):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + index * 22
        lines.append(f'<line x1="{left + plot_w - 165}" y1="{legend_y}" x2="{left + plot_w - 135}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{left + plot_w - 128}" y="{legend_y + 4}" font-family="serif" font-size="13">{_svg_escape(name)}</text>')
    lines.append("</svg>")
    path = out_dir / f"{stem}.svg"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.name


def _plot_svg_completion(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> str:
    series: Dict[str, List[tuple[int, float]]] = {}
    for policy, values in _by_policy(rows).items():
        series[POLICY_LABELS.get(policy, policy)] = [(int(row["budget"]), float(row["completion_rate_mean"])) for row in values]
    return _write_svg_line_plot(out_dir, "fig_rearrangement_completion", series, y_label="Completion rate", y_min=0.0, y_max=1.0)


def _plot_svg_stale_errors(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> str:
    series: Dict[str, List[tuple[int, float]]] = {}
    for policy, values in _by_policy(rows).items():
        series[POLICY_LABELS.get(policy, policy)] = [(int(row["budget"]), float(row["stale_errors_mean"])) for row in values]
    return _write_svg_line_plot(out_dir, "fig_rearrangement_stale_errors", series, y_label="Stale errors", y_min=0.0)


def _plot_svg_delta(delta_rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> str:
    rows = sorted(delta_rows, key=lambda row: int(row["budget"]))
    series = {
        "Completion delta": [(int(row["budget"]), float(row["completion_rate_delta"])) for row in rows],
        "Stale-error delta": [(int(row["budget"]), float(row["stale_errors_delta"])) for row in rows],
    }
    values = [value for points in series.values() for _, value in points]
    bound = max(abs(min(values)), abs(max(values)), 0.1)
    return _write_svg_line_plot(out_dir, "fig_rearrangement_delta", series, y_label="Delta vs salience", y_min=-bound, y_max=bound)


def plot_completion(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> List[str]:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for policy, values in _by_policy(rows).items():
        budgets = [int(row["budget"]) for row in values]
        means = [float(row["completion_rate_mean"]) for row in values]
        stds = [float(row["completion_rate_std"]) for row in values]
        ax.errorbar(budgets, means, yerr=stds, marker="o", linewidth=1.8, capsize=3, label=POLICY_LABELS.get(policy, policy))
    ax.set_xlabel("Memory budget")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out_dir, "fig_rearrangement_completion")


def plot_stale_errors(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> List[str]:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for policy, values in _by_policy(rows).items():
        budgets = [int(row["budget"]) for row in values]
        means = [float(row["stale_errors_mean"]) for row in values]
        stds = [float(row["stale_errors_std"]) for row in values]
        ax.errorbar(budgets, means, yerr=stds, marker="o", linewidth=1.8, capsize=3, label=POLICY_LABELS.get(policy, policy))
    ax.set_xlabel("Memory budget")
    ax.set_ylabel("Stale errors")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out_dir, "fig_rearrangement_stale_errors")


def plot_delta(delta_rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> List[str]:
    plt = _setup_matplotlib()
    rows = sorted(delta_rows, key=lambda row: int(row["budget"]))
    budgets = [int(row["budget"]) for row in rows]
    completion_delta = [float(row["completion_rate_delta"]) for row in rows]
    stale_delta = [float(row["stale_errors_delta"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(4.8, 3.0))
    ax1.plot(budgets, completion_delta, marker="o", linewidth=1.8, color="tab:blue", label="Completion delta")
    ax1.axhline(0, color="0.4", linewidth=0.8)
    ax1.set_xlabel("Memory budget")
    ax1.set_ylabel("Completion delta", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(budgets, stale_delta, marker="s", linewidth=1.8, color="tab:red", label="Stale-error delta")
    ax2.set_ylabel("Stale-error delta", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(axis="y", alpha=0.25)
    return _save(fig, out_dir, "fig_rearrangement_delta")


def render_latex_includes(figure_files: Iterable[str]) -> str:
    captions = {
        "fig_rearrangement_completion.pdf": (
            "Two-seed AI2-THOR rearrangement completion rates under memory budgets.",
            "fig:rearrangement-completion",
        ),
        "fig_rearrangement_completion.svg": (
            "Two-seed AI2-THOR rearrangement completion rates under memory budgets.",
            "fig:rearrangement-completion",
        ),
        "fig_rearrangement_stale_errors.pdf": (
            "Stale-memory errors on true AI2-THOR rearrangement replay.",
            "fig:rearrangement-stale-errors",
        ),
        "fig_rearrangement_stale_errors.svg": (
            "Stale-memory errors on true AI2-THOR rearrangement replay.",
            "fig:rearrangement-stale-errors",
        ),
        "fig_rearrangement_delta.pdf": (
            "Verify-risk policy deltas against salience on two rearrangement seeds.",
            "fig:rearrangement-delta",
        ),
        "fig_rearrangement_delta.svg": (
            "Verify-risk policy deltas against salience on two rearrangement seeds.",
            "fig:rearrangement-delta",
        ),
    }
    blocks: List[str] = []
    for name in figure_files:
        if not (name.endswith(".pdf") or name.endswith(".svg")):
            continue
        caption, label = captions.get(name, (name, f"fig:{Path(name).stem.replace('_', '-')}"))
        blocks.append(
            "\n".join(
                [
                    "\\begin{figure}[t]",
                    "    \\centering",
                    f"    \\includegraphics[width=0.48\\textwidth]{{figures/{name}}}",
                    f"    \\caption{{{caption}}}",
                    f"    \\label{{{label}}}",
                    "\\end{figure}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_latex_includes(out_dir: Path, figure_files: Iterable[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rearrangement_latex_includes.tex"
    path.write_text(render_latex_includes(figure_files), encoding="utf-8")
    return path


def generate_figures(summary_path: Path, delta_path: Path, out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_multiseed_rows(summary_path)
    deltas = load_delta_rows(delta_path)
    files: List[str] = []
    if _matplotlib_available():
        files.extend(plot_completion(rows, out_dir))
        files.extend(plot_stale_errors(rows, out_dir))
        files.extend(plot_delta(deltas, out_dir))
    else:
        files.append(_plot_svg_completion(rows, out_dir))
        files.append(_plot_svg_stale_errors(rows, out_dir))
        files.append(_plot_svg_delta(deltas, out_dir))
    write_latex_includes(out_dir, files)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures for AI2-THOR rearrangement verification.")
    parser.add_argument("--summary", type=Path, default=Path("results/ai2thor_rearrangement_multiseed/ai2thor_rearrangement_multiseed_summary.csv"))
    parser.add_argument("--delta", type=Path, default=Path("results/ai2thor_rearrangement_multiseed/ai2thor_rearrangement_multiseed_delta.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = generate_figures(args.summary, args.delta, args.out_dir)
    print("\n".join(str(args.out_dir / name) for name in files))


if __name__ == "__main__":
    main()
