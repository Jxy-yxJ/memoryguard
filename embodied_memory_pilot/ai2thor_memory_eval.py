from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from embodied_memory_pilot.pilot import (
    FifoPolicy,
    FreshnessAwarePolicy,
    MemoryItem,
    MemoryPolicy,
    MemoryStore,
    NoMemoryPolicy,
    PolicyMetrics,
    SaliencePolicy,
    Task,
    TaskConditionedPolicy,
)


TARGET_OBJECTS = ("Mug", "Apple", "Knife", "RemoteControl", "Book")
HIGH_DEMAND_OBJECTS = set(TARGET_OBJECTS)


class AI2ThorCalibratedPolicy(MemoryPolicy):
    name = "ai2thor_calibrated"

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
        target_bonus = 1.4 if item.object_name == current_target else 0.0
        pickupable_or_target_bonus = 1.2 if item.demand >= 0.45 else 0.0
        stale_risk = item.volatility * min(age / 10.0, 2.0)
        return (
            2.2 * item.salience
            + 2.0 * item.demand
            + target_bonus
            + pickupable_or_target_bonus
            + 0.4 * freshness
            + 0.4 * item.failure_count
            - 1.0 * stale_risk
        )


def load_probe(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def object_features(obj: Dict[str, Any]) -> Dict[str, float]:
    object_type = str(obj.get("object_type") or "Unknown")
    pickupable = bool(obj.get("pickupable"))
    receptacle = bool(obj.get("receptacle"))
    is_target = object_type in HIGH_DEMAND_OBJECTS
    return {
        "demand": 1.0 if is_target else (0.45 if pickupable else 0.2),
        "salience": 0.9 if pickupable else (0.65 if receptacle else 0.35),
        "volatility": 0.25 if pickupable else 0.08,
    }


def build_memory_items(probe: Dict[str, Any]) -> List[MemoryItem]:
    items: List[MemoryItem] = []
    observed_at = 0
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        for obj in scene.get("visible_objects", []):
            object_type = str(obj.get("object_type") or "Unknown")
            object_id = str(obj.get("object_id") or obj.get("name") or object_type)
            features = object_features(obj)
            items.append(
                MemoryItem(
                    object_name=object_type,
                    location=f"{scene_name}:{object_id}",
                    observed_at=observed_at,
                    salience=features["salience"],
                    volatility=features["volatility"],
                    demand=features["demand"],
                )
            )
            observed_at += 1
    return items


def build_query_tasks(probe: Dict[str, Any], target_objects: Sequence[str] = TARGET_OBJECTS) -> List[Task]:
    target_set = set(target_objects)
    tasks: List[Task] = []
    seen_locations: set[str] = set()
    for scene in probe.get("scenes", []):
        if scene.get("status") != "ok":
            continue
        scene_name = str(scene.get("scene"))
        for obj in scene.get("visible_objects", []):
            object_type = str(obj.get("object_type") or "Unknown")
            if object_type not in target_set and not bool(obj.get("pickupable")):
                continue
            object_id = str(obj.get("object_id") or obj.get("name") or object_type)
            location = f"{scene_name}:{object_id}"
            if location in seen_locations:
                continue
            seen_locations.add(location)
            tasks.append(Task(target=object_type, true_location=location))
    return tasks


def policy_suite() -> tuple[MemoryPolicy, ...]:
    return (
        NoMemoryPolicy(),
        FifoPolicy(),
        SaliencePolicy(),
        TaskConditionedPolicy(),
        FreshnessAwarePolicy(),
        AI2ThorCalibratedPolicy(),
    )


def evaluate_policy_on_queries(
    probe: Dict[str, Any],
    policy: MemoryPolicy,
    budget: int,
    target_objects: Sequence[str] = TARGET_OBJECTS,
) -> Dict[str, Any]:
    items = build_memory_items(probe)
    tasks = build_query_tasks(probe, target_objects=target_objects)
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)

    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    for task in tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and not outcome.stale:
            metrics.successes += 1
            metrics.query_hits += 1
            metrics.total_cost += 1.0
        elif outcome.found and outcome.stale:
            metrics.stale_errors += 1
            metrics.total_cost += 4.0
        else:
            metrics.total_cost += 6.0

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "ai2thor_observation_replay"
    row["memory_items"] = len(items)
    row["query_tasks"] = len(tasks)
    row["retained_items"] = len(store.items)
    return row


def run_evaluation(
    probe: Dict[str, Any],
    budgets: Sequence[int],
    policies: Iterable[MemoryPolicy] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for policy in policies or policy_suite():
            rows.append(evaluate_policy_on_queries(probe, policy=policy, budget=budget))
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
                "query_tasks": int(group[0]["query_tasks"]) if group else 0,
                "memory_items": int(group[0]["memory_items"]) if group else 0,
                "retained_items": round(sum(float(r["retained_items"]) for r in group) / len(group), 2),
                "success_rate": round(sum(float(r["success_rate"]) for r in group) / len(group), 4),
                "avg_cost": round(sum(float(r["avg_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
            }
        )
    return summary


def write_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    (out_dir / "ai2thor_memory_eval.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "ai2thor_memory_eval_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "budget",
            "policy",
            "runs",
            "query_tasks",
            "memory_items",
            "retained_items",
            "success_rate",
            "avg_cost",
            "query_hit_rate",
            "stale_errors",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate memory policies on AI2-THOR scene-probe observations.")
    parser.add_argument("--probe", type=Path, default=Path("results/ai2thor_scene_probe_360/ai2thor_scene_probe.json"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_memory_eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = load_probe(args.probe)
    rows = run_evaluation(probe, budgets=args.budgets)
    write_outputs(rows, args.out_dir)
    print(json.dumps(summarize_rows(rows), indent=2))


if __name__ == "__main__":
    main()
