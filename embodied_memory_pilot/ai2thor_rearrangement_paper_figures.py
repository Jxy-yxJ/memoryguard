from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_rearrangement_ablation import ThresholdRearrangementRiskPolicy
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    RearrangementTask,
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    load_probe,
)
from embodied_memory_pilot.pilot import MemoryItem, MemoryStore, SaliencePolicy


PARETO_ABLATIONS = (
    "salience_no_verification",
    "detect_only_0p2",
    "verify_threshold_0p2",
    "verify_threshold_0p5",
    "verify_all",
)

ABLATION_LABELS = {
    "salience_no_verification": "Salience",
    "detect_only_0p2": "Detect only 0.2",
    "verify_threshold_0p2": "Verify 0.2",
    "verify_threshold_0p5": "Verify 0.5",
    "verify_all": "Verify all",
}

ABLATION_COLORS = {
    "salience_no_verification": "#4c4c4c",
    "detect_only_0p2": "#d55e00",
    "verify_threshold_0p2": "#0072b2",
    "verify_threshold_0p5": "#009e73",
    "verify_all": "#cc79a7",
}


def load_ablation_summary(path: Path) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, float | int | str] = {}
            for key, value in raw.items():
                if key in {"ablation", "policy"}:
                    row[key] = value
                elif key in {"budget", "seeds", "recover_caught_stale"}:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def pareto_rows(
    rows: Sequence[Dict[str, float | int | str]],
    *,
    budget: int = 16,
    ablations: Sequence[str] = PARETO_ABLATIONS,
) -> List[Dict[str, float | int | str]]:
    ablation_set = set(ablations)
    selected = [
        row
        for row in rows
        if int(row["budget"]) == budget and str(row["ablation"]) in ablation_set
    ]
    return sorted(
        selected,
        key=lambda row: (
            ablations.index(str(row["ablation"])),
            float(row["verification_cost"]),
        ),
    )


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
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def _save(fig: Any, out_dir: Path, stem: str) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path)
        paths.append(path.name)
    return paths


def _group_by_ablation(rows: Sequence[Dict[str, float | int | str]]) -> Dict[str, List[Dict[str, float | int | str]]]:
    grouped: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ablation"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row["verification_cost"]))
    return grouped


def _unique_pareto_points(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    seen: set[tuple[float, float, float]] = set()
    unique: List[Dict[str, float | int | str]] = []
    for row in rows:
        key = (
            float(row["avg_reachable_cost_mean"]),
            float(row["stale_errors_mean"]),
            float(row["completion_rate_mean"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _pareto_scope_text(rows: Sequence[Dict[str, float | int | str]], *, budget: int = 16) -> str:
    if not rows:
        return f"budget {budget}"
    seeds = int(rows[0]["seeds"])
    tasks_mean = float(rows[0].get("rearrangement_tasks_mean", 0.0))
    total_tasks = int(round(seeds * tasks_mean))
    if total_tasks > 0:
        return f"budget {budget}; {seeds} seeds; {total_tasks} total tasks"
    return f"budget {budget}; {seeds} seeds"


def plot_verification_pareto(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> List[str]:
    plt = _setup_matplotlib()
    budget = 16
    selected = pareto_rows(rows, budget=budget)
    grouped = _group_by_ablation(selected)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax_cost, ax_completion = axes

    for ablation in PARETO_ABLATIONS:
        values = grouped.get(ablation, [])
        if not values:
            continue
        color = ABLATION_COLORS[ablation]
        label = ABLATION_LABELS[ablation]
        points = _unique_pareto_points(values) if ablation == "salience_no_verification" else values
        costs = [float(row["avg_reachable_cost_mean"]) for row in points]
        stale = [float(row["stale_errors_mean"]) for row in points]
        sizes = [70 + 260 * float(row["completion_rate_mean"]) for row in points]
        marker = "X" if ablation == "salience_no_verification" else "o"
        ax_cost.plot(costs, stale, linewidth=1.2, color=color, alpha=0.65)
        ax_cost.scatter(
            costs,
            stale,
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            marker=marker,
            label=label,
            zorder=3,
        )
        if ablation == "verify_threshold_0p2":
            for row in values:
                ax_cost.annotate(
                    f"{float(row['verification_cost']):g}",
                    (float(row["avg_reachable_cost_mean"]), float(row["stale_errors_mean"])),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color=color,
                )

        x_values = [float(row["verification_cost"]) for row in values]
        completion = [float(row["completion_rate_mean"]) for row in values]
        ax_completion.plot(
            x_values,
            completion,
            marker=marker,
            linewidth=1.6,
            color=color,
            label=label,
        )

    ax_cost.set_xlabel("Average reachable cost")
    ax_cost.set_ylabel("Stale errors")
    ax_cost.grid(axis="both", alpha=0.25)
    ax_cost.set_ylim(bottom=-0.12)
    ax_cost.text(
        0.02,
        0.96,
        "marker size = completion",
        transform=ax_cost.transAxes,
        va="top",
        fontsize=8,
        color="0.25",
    )
    ax_cost.text(-0.14, 1.04, "a", transform=ax_cost.transAxes, fontsize=11, fontweight="bold")

    ax_completion.set_xlabel("Verification cost")
    ax_completion.set_ylabel("Completion rate")
    completion_values = [float(row["completion_rate_mean"]) for row in selected]
    completion_low = max(0.0, min(completion_values, default=0.0) - 0.08)
    completion_high = min(1.0, max(completion_values, default=1.0) + 0.08)
    if completion_high - completion_low < 0.25:
        midpoint = (completion_high + completion_low) / 2.0
        completion_low = max(0.0, midpoint - 0.125)
        completion_high = min(1.0, midpoint + 0.125)
    ax_completion.set_ylim(completion_low, completion_high)
    ax_completion.grid(axis="y", alpha=0.25)
    ax_completion.text(
        0.02,
        0.08,
        _pareto_scope_text(selected, budget=budget),
        transform=ax_completion.transAxes,
        va="bottom",
        fontsize=8,
        color="0.25",
    )
    ax_completion.text(-0.14, 1.04, "b", transform=ax_completion.transAxes, fontsize=11, fontweight="bold")

    handles, labels = ax_completion.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    files = _save(fig, out_dir, "fig_rearrangement_verification_pareto")
    plt.close(fig)
    return files


def _seed_from_path(path: Path) -> str:
    match = re.search(r"seed(\d+)", str(path))
    return match.group(1) if match else path.parent.name


def _verification_risk(item: MemoryItem, now: int) -> float:
    age = max(now - item.observed_at, 0)
    return item.volatility * (1.0 + age / 12.0)


def extract_stale_case(
    probe_paths: Sequence[Path],
    *,
    budget: int = 16,
    threshold: float = 0.2,
    verification_cost: float = 0.5,
) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    verifier = ThresholdRearrangementRiskPolicy(threshold)
    for path in probe_paths:
        probe = load_probe(path)
        items = build_rearrangement_memory_items(probe)
        tasks = build_rearrangement_tasks(probe)
        store = MemoryStore(policy=SaliencePolicy(), budget=budget)
        for item in items:
            store.observe(item, now=item.observed_at, current_target=item.object_name)

        query_now = max((item.observed_at for item in items), default=0) + 1
        task_by_old_location: Dict[str, RearrangementTask] = {task.old_location: task for task in tasks}
        for task in tasks:
            outcome = store.query(task.target, task.true_location)
            if not (outcome.found and outcome.stale and outcome.item is not None):
                continue
            if not verifier.should_verify(outcome.item, now=query_now):
                continue
            selected_old_task = task_by_old_location.get(outcome.item.location)
            stale_action_cost = selected_old_task.old_action_cost if selected_old_task else task.old_action_cost
            cases.append(
                {
                    "source_probe": str(path),
                    "seed": _seed_from_path(path),
                    "scene": task.scene,
                    "target": task.target,
                    "budget": budget,
                    "threshold": threshold,
                    "verification_cost": verification_cost,
                    "selected_before_location": outcome.item.location,
                    "true_after_location": task.true_location,
                    "task_old_location": task.old_location,
                    "direct_after_cost": task.direct_action_cost,
                    "stale_action_cost": stale_action_cost,
                    "fallback_search_cost": task.scene_search_cost,
                    "risk_score": round(_verification_risk(outcome.item, query_now), 4),
                    "retained_item": {
                        "object_name": outcome.item.object_name,
                        "observed_at": outcome.item.observed_at,
                        "salience": outcome.item.salience,
                        "volatility": outcome.item.volatility,
                        "demand": outcome.item.demand,
                    },
                    "salience_outcome": "stale_error",
                    "verification_outcome": "stale_caught_then_scene_search_recovery",
                    "rearranged": task.rearranged,
                }
            )
    if not cases:
        raise ValueError("No salience stale case caught by thresholded verification was found.")
    return max(
        cases,
        key=lambda case: (
            float(case["fallback_search_cost"]) - float(case["direct_after_cost"]),
            str(case["seed"]),
            str(case["scene"]),
            str(case["target"]),
        ),
    )


def _short_location(location: str, width: int = 48) -> str:
    if ":" in location:
        location = location.split(":", 1)[1]
    return textwrap.shorten(location, width=width, placeholder="...")


def _format_location_for_box(location: str) -> str:
    if ":" in location:
        location = location.split(":", 1)[1]
    state = ""
    if "@" in location:
        location, state = location.rsplit("@", 1)
    parts = location.split("|")
    if len(parts) >= 4:
        name, x_value, y_value, z_value = parts[:4]
        suffix = f" @ {state}" if state else ""
        return "\n".join(
            [
                f"{name}{suffix}",
                f"x={x_value}  y={y_value}",
                f"z={z_value}",
            ]
        )
    return "\n".join(textwrap.wrap(location, width=22))


def write_case_artifacts(case: Dict[str, Any], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stale_memory_case.json"
    md_path = out_dir / "stale_memory_case.md"
    json_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    md_lines = [
        "# Stale Memory Verification Case",
        "",
        f"- Source probe: `{case['source_probe']}`",
        f"- Seed: `{case['seed']}`",
        f"- Scene: `{case['scene']}`",
        f"- Target: `{case['target']}`",
        f"- Selected before-state memory: `{case['selected_before_location']}`",
        f"- True after-state target: `{case['true_after_location']}`",
        f"- Verification risk score: `{case['risk_score']}` at threshold `{case['threshold']}`",
        f"- Salience-only outcome: `{case['salience_outcome']}`",
        f"- Thresholded verification outcome: `{case['verification_outcome']}`",
        f"- Costs: stale action `{case['stale_action_cost']}`, direct after action `{case['direct_after_cost']}`, fallback scene search `{case['fallback_search_cost']}`, verification `{case['verification_cost']}`",
        "",
        "This is a replay/oracle-style verification case over saved before/after AI2-THOR metadata, not a live closed-loop sensing trace.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return [json_path.name, md_path.name]


def plot_stale_case(case: Dict[str, Any], out_dir: Path) -> List[str]:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axis("off")
    boxes = [
        (
            0.02,
            "Salience replay",
            "Retained before-state\n"
            f"{_format_location_for_box(str(case['selected_before_location']))}",
            "#e8e8e8",
        ),
        (
            0.36,
            "Verify risk",
            f"risk={float(case['risk_score']):.3f}\n"
            f"threshold={float(case['threshold']):.1f}\n"
            "saved after-state check",
            "#dbeaf7",
        ),
        (
            0.70,
            "Recovered search",
            "True after-state\n"
            f"{_format_location_for_box(str(case['true_after_location']))}",
            "#dff0e5",
        ),
    ]
    for x, heading, body, color in boxes:
        patch = plt.Rectangle((x, 0.34), 0.28, 0.50, facecolor=color, edgecolor="0.35", linewidth=0.8)
        ax.add_patch(patch)
        ax.text(x + 0.014, 0.77, heading, fontsize=8.8, fontweight="bold", va="top")
        ax.text(x + 0.014, 0.67, body, fontsize=7.2, va="top", linespacing=1.18)
    for start, end in ((0.30, 0.36), (0.64, 0.70)):
        ax.annotate("", xy=(end, 0.57), xytext=(start, 0.57), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "0.25"})
    ax.text(
        0.02,
        0.20,
        "\n".join(
            textwrap.wrap(
                f"Seed {case['seed']} / {case['scene']} / target {case['target']}: salience-only trusts the before-state memory and produces a stale error; thresholded verification catches the stale hit and falls back to scene search.",
                width=98,
            )
        ),
        fontsize=8.5,
        va="top",
    )
    ax.text(
        0.02,
        0.06,
        "\n".join(
            textwrap.wrap(
                f"Costs in this replay: stale action {float(case['stale_action_cost']):.2f}; direct after action {float(case['direct_after_cost']):.2f}; fallback search {float(case['fallback_search_cost']):.2f}; verification {float(case['verification_cost']):.2f}.",
                width=112,
            )
        ),
        fontsize=8,
        va="top",
        color="0.25",
    )
    files = _save(fig, out_dir, "fig_rearrangement_stale_case")
    plt.close(fig)
    return files


def render_latex_includes(figure_files: Iterable[str], *, seeds: int | None = None, total_tasks: int | None = None) -> str:
    if seeds is not None and total_tasks is not None:
        pareto_scope = f"{seeds} saved AI2-THOR rearrangement probes and {total_tasks} total tasks"
    else:
        pareto_scope = "the saved AI2-THOR rearrangement probes"
    captions = {
        "fig_rearrangement_verification_pareto.pdf": (
            f"Verification-cost and stale-error tradeoff on {pareto_scope}. "
            "All points use memory budget 16; marker size in panel (a) encodes completion.",
            "fig:rearrangement-verification-pareto",
            "0.95\\textwidth",
        ),
        "fig_rearrangement_stale_case.pdf": (
            "Qualitative stale-memory catch case from saved AI2-THOR before/after metadata. "
            "A salience-retained before-state object memory is stale after rearrangement; thresholded replay verification catches it and falls back to scene search.",
            "fig:rearrangement-stale-case",
            "0.95\\textwidth",
        ),
    }
    blocks: List[str] = []
    for name in figure_files:
        if not name.endswith(".pdf") or name not in captions:
            continue
        caption, label, width = captions[name]
        blocks.append(
            "\n".join(
                [
                    "\\begin{figure}[t]",
                    "    \\centering",
                    f"    \\includegraphics[width={width}]{{figures/{name}}}",
                    f"    \\caption{{{caption}}}",
                    f"    \\label{{{label}}}",
                    "\\end{figure}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_latex_includes(
    out_dir: Path,
    figure_files: Iterable[str],
    *,
    seeds: int | None = None,
    total_tasks: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rearrangement_paper_latex_includes.tex"
    path.write_text(render_latex_includes(figure_files, seeds=seeds, total_tasks=total_tasks), encoding="utf-8")
    return path


def generate_paper_figures(
    summary_path: Path,
    probe_paths: Sequence[Path],
    out_dir: Path,
    *,
    budget: int = 16,
    threshold: float = 0.2,
    verification_cost: float = 0.5,
) -> List[str]:
    rows = load_ablation_summary(summary_path)
    selected = pareto_rows(rows, budget=budget)
    if not selected:
        raise ValueError(f"No ablation rows found for budget {budget}.")
    seeds = int(selected[0]["seeds"])
    total_tasks = int(round(seeds * float(selected[0].get("rearrangement_tasks_mean", 0.0))))
    files: List[str] = []
    files.extend(plot_verification_pareto(rows, out_dir))
    case = extract_stale_case(probe_paths, budget=budget, threshold=threshold, verification_cost=verification_cost)
    files.extend(write_case_artifacts(case, out_dir))
    files.extend(plot_stale_case(case, out_dir))
    write_latex_includes(out_dir, files, seeds=seeds, total_tasks=total_tasks)
    files.append("rearrangement_paper_latex_includes.tex")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 26 paper figures for AI2-THOR rearrangement verification.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/ai2thor_rearrangement_ablation/ai2thor_rearrangement_ablation_summary.csv"),
    )
    parser.add_argument(
        "--probes",
        type=Path,
        nargs="+",
        default=[
            Path("results/ai2thor_rearrangement_6scene_seed7/ai2thor_rearrangement_probe.json"),
            Path("results/ai2thor_rearrangement_6scene_seed11/ai2thor_rearrangement_probe.json"),
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--verification-cost", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = generate_paper_figures(
        args.summary,
        args.probes,
        args.out_dir,
        budget=args.budget,
        threshold=args.threshold,
        verification_cost=args.verification_cost,
    )
    print("\n".join(str(args.out_dir / name) for name in files))


if __name__ == "__main__":
    main()
