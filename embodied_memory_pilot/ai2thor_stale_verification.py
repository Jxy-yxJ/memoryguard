from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS, load_probe
from embodied_memory_pilot.ai2thor_reachable_benchmark import ReachableTask, build_reachable_tasks
from embodied_memory_pilot.ai2thor_stale_stress import (
    build_stale_stress_memory_items,
    policy_suite,
    select_probe_scenes,
)
from embodied_memory_pilot.pilot import MemoryItem, MemoryPolicy, MemoryStore, NoMemoryPolicy, PolicyMetrics


class VerifyHighRiskSaliencePolicy(MemoryPolicy):
    name = "verify_high_risk_salience"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Dict[str, int] | None = None,
    ) -> float:
        freshness = 1.0 / (1.0 + now - item.observed_at)
        return 2.0 * item.salience + 0.2 * freshness + 0.1 * item.failure_count

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        age = now - item.observed_at
        return item.volatility * max(age, 0) > 1.0


def _evaluate_with_verification(
    tasks: Sequence[ReachableTask],
    items: Sequence[MemoryItem],
    policy: MemoryPolicy,
    budget: int,
    verification_cost: float,
) -> Dict[str, Any]:
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)
    saved_reachable_cost = 0.0

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    query_now = max((item.observed_at for item in items), default=0) + 1
    for task in tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)

        if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=query_now):
            metrics.verifications += 1
            metrics.total_cost += verification_cost
            if outcome.stale:
                metrics.verification_catches += 1
                metrics.successes += 1
                metrics.total_cost += task.scene_search_cost
            else:
                metrics.successes += 1
                metrics.query_hits += 1
                metrics.total_cost += task.direct_action_cost
                saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost - verification_cost, 0.0)
        elif outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost
            saved_reachable_cost += max(task.scene_search_cost - task.direct_action_cost, 0.0)
        elif outcome.found and outcome.stale:
            metrics.stale_errors += 1
            metrics.total_cost += task.direct_action_cost + task.scene_search_cost
        else:
            metrics.total_cost += task.scene_search_cost

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "ai2thor_stale_verification"
    row["memory_items"] = len(items)
    row["reachable_tasks"] = len(tasks)
    row["retained_items"] = len(store.items)
    row["avg_reachable_cost"] = round(metrics.avg_cost, 4)
    row["saved_reachable_cost"] = round(saved_reachable_cost, 4)
    row["completion_rate"] = row["success_rate"]
    row["verification_cost"] = verification_cost
    return row


def evaluate_policy_stale_verification(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    age_gap: int = 48,
    stale_salience: float = 1.4,
    stale_volatility: float = 0.9,
    verification_cost: float = 0.5,
) -> Dict[str, Any]:
    tasks = build_reachable_tasks(probe, target_objects=target_objects)
    items = build_stale_stress_memory_items(
        probe,
        target_objects=target_objects,
        age_gap=age_gap,
        stale_salience=stale_salience,
        stale_volatility=stale_volatility,
    )
    row = _evaluate_with_verification(
        tasks,
        items,
        policy=policy,
        budget=budget,
        verification_cost=verification_cost,
    )
    row["stale_distractors"] = len(tasks)
    return row


def run_stale_verification(
    probe: Dict[str, Any],
    budgets: Sequence[int],
    policies: Iterable[MemoryPolicy] | None = None,
    target_objects: Sequence[str] = TARGET_OBJECTS,
    age_gap: int = 48,
    stale_salience: float = 1.4,
    stale_volatility: float = 0.9,
    verification_cost: float = 0.5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected_policies = tuple(policies) if policies is not None else (*policy_suite(), VerifyHighRiskSaliencePolicy())
    for budget in budgets:
        for policy in selected_policies:
            rows.append(
                evaluate_policy_stale_verification(
                    probe,
                    policy=policy,
                    budget=budget,
                    target_objects=target_objects,
                    age_gap=age_gap,
                    stale_salience=stale_salience,
                    stale_volatility=stale_volatility,
                    verification_cost=verification_cost,
                )
            )
    return rows


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["budget"]), str(row["policy"])), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (budget, policy), group in sorted(grouped.items()):
        summary.append(
            {
                "budget": budget,
                "policy": policy,
                "runs": len(group),
                "reachable_tasks": int(group[0]["reachable_tasks"]) if group else 0,
                "memory_items": int(group[0]["memory_items"]) if group else 0,
                "stale_distractors": int(group[0]["stale_distractors"]) if group else 0,
                "retained_items": round(sum(float(r["retained_items"]) for r in group) / len(group), 2),
                "completion_rate": round(sum(float(r["completion_rate"]) for r in group) / len(group), 4),
                "avg_reachable_cost": round(sum(float(r["avg_reachable_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
                "verifications": round(sum(float(r["verifications"]) for r in group) / len(group), 2),
                "verification_catches": round(sum(float(r["verification_catches"]) for r in group) / len(group), 2),
                "verification_cost": round(sum(float(r["verification_cost"]) for r in group) / len(group), 4),
                "saved_reachable_cost": round(sum(float(r["saved_reachable_cost"]) for r in group) / len(group), 4),
            }
        )
    return summary


def write_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (out_dir / "ai2thor_stale_verification.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_stale_verification_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "reachable_tasks",
            "memory_items",
            "stale_distractors",
            "retained_items",
            "completion_rate",
            "avg_reachable_cost",
            "query_hit_rate",
            "stale_errors",
            "verifications",
            "verification_catches",
            "verification_cost",
            "saved_reachable_cost",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")


def render_readme(summary: Sequence[Dict[str, Any]]) -> str:
    best = max(
        summary,
        key=lambda row: (
            float(row["completion_rate"]),
            -float(row["stale_errors"]),
            -float(row["avg_reachable_cost"]),
            -float(row["verifications"]),
        ),
        default=None,
    )
    lines = [
        "# AI2-THOR Stale-Memory Verification Benchmark",
        "",
        "Mode: `ai2thor_stale_verification`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best policy: `{best['policy']}` at budget `{best['budget']}`",
                f"Completion rate: `{best['completion_rate']}`",
                f"Average reachable cost: `{best['avg_reachable_cost']}`",
                f"Stale errors: `{best['stale_errors']}`",
                f"Verifications: `{best['verifications']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This benchmark extends stale-memory stress with an explicit verification action.",
            "A policy may pay verification cost before trusting a suspicious memory; caught stale hits fall back to scene search instead of becoming stale errors.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR stale-memory verification benchmark.")
    parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_reachable_benchmark_9scene/ai2thor_reachable_probe.json"))
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--age-gap", type=int, default=48)
    parser.add_argument("--stale-salience", type=float, default=1.4)
    parser.add_argument("--stale-volatility", type=float, default=0.9)
    parser.add_argument("--verification-cost", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_stale_verification"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = select_probe_scenes(load_probe(args.probe), scenes=args.scenes)
    rows = run_stale_verification(
        probe,
        budgets=args.budgets,
        age_gap=args.age_gap,
        stale_salience=args.stale_salience,
        stale_volatility=args.stale_volatility,
        verification_cost=args.verification_cost,
    )
    write_outputs(rows, args.out_dir)
    print(json.dumps(summarize_rows(rows), indent=2))


if __name__ == "__main__":
    main()
