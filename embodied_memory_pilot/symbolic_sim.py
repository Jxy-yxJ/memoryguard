from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from embodied_memory_pilot.pilot import (
    FreshOraclePolicy,
    FreshnessAwarePolicy,
    FifoPolicy,
    HIGH_VALUE_OBJECTS,
    LOW_VALUE_OBJECTS,
    MemoryItem,
    MemoryPolicy,
    MemoryStore,
    NoMemoryPolicy,
    ObjectProfile,
    OraclePolicy,
    PolicyMetrics,
    ROOMS,
    SaliencePolicy,
    TaskConditionedPolicy,
)


ROOM_DISTANCE: Dict[Tuple[str, str], float] = {
    (src, dst): (0.0 if src == dst else 2.0 + abs(ROOMS.index(src) - ROOMS.index(dst)) * 0.8)
    for src in ROOMS
    for dst in ROOMS
}


@dataclass(frozen=True)
class SimTask:
    target: str
    start_room: str


@dataclass
class SymbolicHouse:
    profiles: Dict[str, ObjectProfile]
    locations: Dict[str, str]
    rng: random.Random
    drift_scale: float = 1.0

    @classmethod
    def create(cls, seed: int, drift_scale: float = 1.0) -> "SymbolicHouse":
        rng = random.Random(seed)
        profiles: Dict[str, ObjectProfile] = {}
        for name in HIGH_VALUE_OBJECTS:
            profiles[name] = ObjectProfile(
                name=name,
                demand=rng.uniform(0.65, 1.0),
                salience=rng.uniform(0.35, 0.75),
                volatility=rng.uniform(0.05, 0.22),
            )
        for name in LOW_VALUE_OBJECTS:
            profiles[name] = ObjectProfile(
                name=name,
                demand=rng.uniform(0.03, 0.22),
                salience=rng.uniform(0.45, 1.0),
                volatility=rng.uniform(0.05, 0.45),
            )
        locations = {name: rng.choice(ROOMS) for name in profiles}
        return cls(profiles=profiles, locations=locations, rng=rng, drift_scale=drift_scale)

    def sample_task(self) -> SimTask:
        objects = list(self.profiles)
        weights = [self.profiles[name].demand for name in objects]
        return SimTask(target=self.rng.choices(objects, weights=weights, k=1)[0], start_room=self.rng.choice(ROOMS))

    def drift(self) -> None:
        for name, profile in self.profiles.items():
            if self.rng.random() < min(profile.volatility * 0.05 * self.drift_scale, 0.8):
                self.locations[name] = self.rng.choice(ROOMS)

    def observe_room(self, room: str, now: int, failure_target: str | None = None) -> List[MemoryItem]:
        visible = [name for name, location in self.locations.items() if location == room]
        if len(visible) > 5:
            visible = self.rng.sample(visible, 5)
        observations: List[MemoryItem] = []
        for name in visible:
            profile = self.profiles[name]
            observations.append(
                MemoryItem(
                    object_name=name,
                    location=room,
                    observed_at=now,
                    salience=profile.salience,
                    volatility=profile.volatility,
                    demand=profile.demand,
                    failure_count=1 if failure_target == name else 0,
                )
            )
        return observations

    def search_cost(self, start_room: str, target: str) -> float:
        target_room = self.locations[target]
        return 3.0 + ROOM_DISTANCE[(start_room, target_room)]


def policy_suite() -> Tuple[MemoryPolicy, ...]:
    return (
        NoMemoryPolicy(),
        FifoPolicy(),
        SaliencePolicy(),
        TaskConditionedPolicy(),
        FreshnessAwarePolicy(),
        OraclePolicy(),
        FreshOraclePolicy(),
    )


def run_symbolic_policy(
    seed: int,
    budget: int,
    tasks_n: int,
    policy: MemoryPolicy,
    drift_scale: float = 1.0,
    task_budget: float = 6.0,
) -> PolicyMetrics:
    world = SymbolicHouse.create(seed=seed, drift_scale=drift_scale)
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)

    for now in range(tasks_n):
        world.drift()
        task = world.sample_task()
        target_room = world.locations[task.target]
        outcome = store.query(task.target, target_room)
        failure_target: str | None = None

        metrics.tasks += 1
        cost = 0.0

        if outcome.found and outcome.item is not None:
            if policy.should_verify(outcome.item, now=now):
                metrics.verifications += 1
                cost += 1.5
                if outcome.stale:
                    metrics.verification_catches += 1
                    cost += world.search_cost(task.start_room, task.target)
                else:
                    metrics.query_hits += 1
                    cost += ROOM_DISTANCE[(task.start_room, outcome.item.location)] + 0.5
            elif outcome.stale:
                metrics.stale_errors += 1
                failure_target = task.target
                cost += ROOM_DISTANCE[(task.start_room, outcome.item.location)] + 6.0
            else:
                metrics.query_hits += 1
                cost += ROOM_DISTANCE[(task.start_room, outcome.item.location)] + 0.5
        else:
            cost += world.search_cost(task.start_room, task.target)

        if cost <= task_budget:
            metrics.successes += 1
        metrics.total_cost += cost

        for item in world.observe_room(target_room, now=now, failure_target=failure_target):
            store.observe(item, now=now, current_target=task.target)

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    return metrics


def run_symbolic_suite(
    seeds: Iterable[int],
    budgets: Sequence[int],
    tasks: int,
    drift_scale: float = 1.0,
    task_budget: float = 6.0,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for seed in seeds:
        for budget in budgets:
            for policy in policy_suite():
                metrics = run_symbolic_policy(
                    seed=seed,
                    budget=budget,
                    tasks_n=tasks,
                    policy=policy,
                    drift_scale=drift_scale,
                    task_budget=task_budget,
                )
                row = metrics.as_dict(seed=seed, budget=budget)
                row["drift_scale"] = drift_scale
                row["task_budget"] = task_budget
                row["mode"] = "symbolic_closed_loop"
                rows.append(row)
    return rows


def aggregate(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((int(row["budget"]), str(row["policy"])), []).append(row)

    summary: List[Dict[str, float | int | str]] = []
    for (budget, policy), group in sorted(grouped.items()):
        summary.append(
            {
                "budget": budget,
                "policy": policy,
                "runs": len(group),
                "success_rate": round(sum(float(r["success_rate"]) for r in group) / len(group), 4),
                "avg_cost": round(sum(float(r["avg_cost"]) for r in group) / len(group), 4),
                "query_hit_rate": round(sum(float(r["query_hit_rate"]) for r in group) / len(group), 4),
                "stale_errors": round(sum(float(r["stale_errors"]) for r in group) / len(group), 2),
                "verifications": round(sum(float(r["verifications"]) for r in group) / len(group), 2),
                "verification_catches": round(
                    sum(float(r["verification_catches"]) for r in group) / len(group),
                    2,
                ),
            }
        )
    return summary


def write_outputs(rows: Sequence[Dict[str, float | int | str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    (out_dir / "sim_symbolic_pilot.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "sim_symbolic_pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "budget",
                "policy",
                "runs",
                "success_rate",
                "avg_cost",
                "query_hit_rate",
                "stale_errors",
                "verifications",
                "verification_catches",
            ),
        )
        writer.writeheader()
        writer.writerows(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a symbolic closed-loop embodied memory pilot.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--task-budget", type=float, default=6.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/sim_symbolic_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_symbolic_suite(
        seeds=range(args.seeds),
        budgets=args.budgets,
        tasks=args.tasks,
        drift_scale=args.drift_scale,
        task_budget=args.task_budget,
    )
    write_outputs(rows, args.out_dir)
    print(json.dumps(aggregate(rows), indent=2))


if __name__ == "__main__":
    main()
