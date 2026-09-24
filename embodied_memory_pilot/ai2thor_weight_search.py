from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_memory_eval import load_probe
from embodied_memory_pilot.ai2thor_reachable_benchmark import (
    evaluate_policy_reachable_costs,
    summarize_rows,
)
from embodied_memory_pilot.ai2thor_memory_eval import AI2ThorCalibratedPolicy
from embodied_memory_pilot.pilot import MemoryItem, MemoryPolicy, NoMemoryPolicy, SaliencePolicy, TaskConditionedPolicy


DEFAULT_TRAIN_SCENES = (
    "FloorPlan1",
    "FloorPlan2",
    "FloorPlan3",
    "FloorPlan201",
    "FloorPlan202",
    "FloorPlan301",
)


class WeightedAI2ThorPolicy(MemoryPolicy):
    def __init__(self, name: str, weights: Dict[str, float]) -> None:
        self.name = name
        self.weights = dict(weights)

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Dict[str, int] | None = None,
    ) -> float:
        age = now - item.observed_at
        freshness = 1.0 / (1.0 + age)
        target_match = 1.0 if item.object_name == current_target else 0.0
        stale_risk = item.volatility * min(age / 10.0, 2.0)
        pickupable_proxy = 1.0 if item.demand >= 0.45 else 0.0
        return (
            self.weights.get("salience", 0.0) * item.salience
            + self.weights.get("demand", 0.0) * item.demand
            + self.weights.get("target", 0.0) * target_match
            + self.weights.get("freshness", 0.0) * freshness
            + self.weights.get("pickupable", 0.0) * pickupable_proxy
            - self.weights.get("stale", 0.0) * stale_risk
            + self.weights.get("failure", 0.0) * item.failure_count
        )


def candidate_weight_grid() -> List[Dict[str, float]]:
    return [
        {"salience": 2.0, "demand": 0.5, "target": 0.2, "freshness": 0.1, "pickupable": 0.0, "stale": 0.2},
        {"salience": 1.5, "demand": 1.5, "target": 0.5, "freshness": 0.1, "pickupable": 0.5, "stale": 0.4},
        {"salience": 1.0, "demand": 2.5, "target": 0.8, "freshness": 0.1, "pickupable": 1.0, "stale": 0.4},
        {"salience": 0.8, "demand": 3.0, "target": 1.2, "freshness": 0.2, "pickupable": 1.2, "stale": 0.6},
        {"salience": 2.2, "demand": 2.0, "target": 1.4, "freshness": 0.4, "pickupable": 1.2, "stale": 1.0},
        {"salience": 1.2, "demand": 3.5, "target": 1.5, "freshness": 0.0, "pickupable": 2.0, "stale": 0.8},
        {"salience": 2.8, "demand": 1.0, "target": 0.5, "freshness": 0.0, "pickupable": 0.4, "stale": 0.2},
        {"salience": 0.5, "demand": 4.0, "target": 2.0, "freshness": 0.0, "pickupable": 2.0, "stale": 0.2},
    ]


def split_probe_by_scenes(probe: Dict[str, Any], train_scenes: Sequence[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    train_set = set(train_scenes)
    train_scenes_out: List[Dict[str, Any]] = []
    heldout_scenes_out: List[Dict[str, Any]] = []
    for scene in probe.get("scenes", []):
        if str(scene.get("scene")) in train_set:
            train_scenes_out.append(scene)
        else:
            heldout_scenes_out.append(scene)
    train = dict(probe)
    heldout = dict(probe)
    train["scenes"] = train_scenes_out
    heldout["scenes"] = heldout_scenes_out
    return train, heldout


def score_summary(summary: Sequence[Dict[str, Any]]) -> float:
    if not summary:
        return float("-inf")
    score = 0.0
    for row in summary:
        budget = int(row["budget"])
        budget_weight = 1.0 + budget / 16.0
        completion = float(row["completion_rate"])
        avg_cost = float(row["avg_reachable_cost"])
        stale = float(row["stale_errors"])
        score += budget_weight * (completion * 10.0 - avg_cost * 0.25 - stale * 0.4)
    return round(score, 6)


def evaluate_weight_set(
    probe: Dict[str, Any],
    weights: Dict[str, float],
    budgets: Sequence[int],
    name: str,
) -> Dict[str, Any]:
    policy = WeightedAI2ThorPolicy(name=name, weights=weights)
    rows = [evaluate_policy_reachable_costs(probe, policy=policy, budget=budget) for budget in budgets]
    summary = summarize_rows(rows)
    return {
        "name": name,
        "weights": dict(weights),
        "score": score_summary(summary),
        "rows": rows,
        "summary": summary,
    }


def train_weight_search(
    probe: Dict[str, Any],
    train_scenes: Sequence[str] = DEFAULT_TRAIN_SCENES,
    budgets: Sequence[int] = (2, 4, 8, 16),
    candidates: Iterable[Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    train_probe, heldout_probe = split_probe_by_scenes(probe, train_scenes=train_scenes)
    candidate_results: List[Dict[str, Any]] = []
    for index, weights in enumerate(candidates or candidate_weight_grid()):
        candidate_results.append(evaluate_weight_set(train_probe, weights, budgets, name=f"weighted_{index:02d}"))

    best = max(candidate_results, key=lambda result: result["score"])
    heldout_eval = evaluate_weight_set(heldout_probe, best["weights"], budgets, name="weighted_best_heldout")
    baseline_rows: List[Dict[str, Any]] = []
    baseline_policies: List[MemoryPolicy] = [
        NoMemoryPolicy(),
        SaliencePolicy(),
        TaskConditionedPolicy(),
        AI2ThorCalibratedPolicy(),
        WeightedAI2ThorPolicy("weighted_search", best["weights"]),
    ]
    for budget in budgets:
        for policy in baseline_policies:
            baseline_rows.append(evaluate_policy_reachable_costs(heldout_probe, policy=policy, budget=budget))
    return {
        "train_scenes": list(train_scenes),
        "heldout_scenes": [str(scene.get("scene")) for scene in heldout_probe.get("scenes", [])],
        "budgets": list(budgets),
        "best_name": best["name"],
        "best_weights": best["weights"],
        "train_score": best["score"],
        "train_summary": best["summary"],
        "heldout_score": heldout_eval["score"],
        "heldout_summary": heldout_eval["summary"],
        "heldout_baseline_summary": summarize_rows(baseline_rows),
        "candidate_results": [
            {"name": result["name"], "weights": result["weights"], "score": result["score"], "summary": result["summary"]}
            for result in candidate_results
        ],
    }


def write_outputs(result: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ai2thor_weight_search.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (out_dir / "ai2thor_weight_search_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("name", "score", "weights")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in result["candidate_results"]:
            writer.writerow(
                {
                    "name": candidate["name"],
                    "score": candidate["score"],
                    "weights": json.dumps(candidate["weights"], sort_keys=True),
                }
            )

    with (out_dir / "ai2thor_weight_search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "split",
            "budget",
            "policy",
            "reachable_tasks",
            "memory_items",
            "completion_rate",
            "avg_reachable_cost",
            "query_hit_rate",
            "stale_errors",
            "saved_reachable_cost",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, rows in (("train", result["train_summary"]), ("heldout", result["heldout_summary"])):
            for row in rows:
                writer.writerow({"split": split_name, **{field: row[field] for field in fieldnames if field != "split"}})

    with (out_dir / "ai2thor_weight_search_heldout_baselines.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "reachable_tasks",
            "memory_items",
            "completion_rate",
            "avg_reachable_cost",
            "query_hit_rate",
            "stale_errors",
            "saved_reachable_cost",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["heldout_baseline_summary"]:
            writer.writerow({field: row[field] for field in fieldnames})

    (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def render_readme(result: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# AI2-THOR Utility Weight Search",
            "",
            f"Best candidate: `{result['best_name']}`",
            f"Train scenes: `{', '.join(result['train_scenes'])}`",
            f"Held-out scenes: `{', '.join(result['heldout_scenes'])}`",
            f"Train score: `{result['train_score']}`",
            f"Held-out score: `{result['heldout_score']}`",
            f"Best weights: `{json.dumps(result['best_weights'], sort_keys=True)}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search AI2-THOR memory utility weights on train scenes.")
    parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_reachable_benchmark_9scene/ai2thor_reachable_probe.json"))
    parser.add_argument("--train-scenes", nargs="+", default=list(DEFAULT_TRAIN_SCENES))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_weight_search"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_weight_search(
        load_probe(args.probe),
        train_scenes=args.train_scenes,
        budgets=args.budgets,
    )
    write_outputs(result, args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "candidate_results"}, indent=2))


if __name__ == "__main__":
    main()
