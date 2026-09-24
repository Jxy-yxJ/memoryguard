from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOMS = ("kitchen", "bedroom", "bathroom", "living_room", "office")
HIGH_VALUE_OBJECTS = ("mug", "knife", "remote", "medicine", "keys")
LOW_VALUE_OBJECTS = (
    "book",
    "plate",
    "lamp",
    "towel",
    "soap",
    "pillow",
    "pen",
    "charger",
    "plant",
    "shoe",
    "bottle",
    "spoon",
)


@dataclass(frozen=True)
class ObjectProfile:
    name: str
    demand: float
    salience: float
    volatility: float


@dataclass(frozen=True)
class MemoryItem:
    object_name: str
    location: str
    observed_at: int
    salience: float
    volatility: float
    demand: float
    failure_count: int = 0


@dataclass(frozen=True)
class Task:
    target: str
    true_location: str


@dataclass(frozen=True)
class TraceStep:
    task: Task
    observations: Tuple[MemoryItem, ...]


@dataclass
class QueryOutcome:
    found: bool
    stale: bool
    item: Optional[MemoryItem]


@dataclass
class PolicyMetrics:
    policy: str
    tasks: int = 0
    successes: int = 0
    total_cost: float = 0.0
    stale_errors: int = 0
    query_hits: int = 0
    verifications: int = 0
    verification_catches: int = 0
    writes: int = 0
    evictions: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.tasks if self.tasks else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.tasks if self.tasks else 0.0

    @property
    def hit_rate(self) -> float:
        return self.query_hits / self.tasks if self.tasks else 0.0

    def as_dict(self, seed: int, budget: int) -> Dict[str, float | int | str]:
        return {
            "seed": seed,
            "budget": budget,
            "policy": self.policy,
            "tasks": self.tasks,
            "success_rate": round(self.success_rate, 4),
            "avg_cost": round(self.avg_cost, 4),
            "query_hit_rate": round(self.hit_rate, 4),
            "stale_errors": self.stale_errors,
            "verifications": self.verifications,
            "verification_catches": self.verification_catches,
            "writes": self.writes,
            "evictions": self.evictions,
        }


class MemoryPolicy:
    name = "base"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        raise NotImplementedError

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        return False


class NoMemoryPolicy(MemoryPolicy):
    name = "no_memory"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        return -math.inf


class FifoPolicy(MemoryPolicy):
    name = "fifo"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        return float(item.observed_at)


class SaliencePolicy(MemoryPolicy):
    name = "salience"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        freshness = 1.0 / (1.0 + now - item.observed_at)
        return 2.0 * item.salience + 0.2 * freshness + 0.1 * item.failure_count


class TaskConditionedPolicy(MemoryPolicy):
    name = "task_conditioned"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        age = now - item.observed_at
        freshness = 1.0 / (1.0 + age)
        target_bonus = 1.0 if item.object_name == current_target else 0.0
        stale_risk = item.volatility * min(age / 8.0, 2.0)
        return (
            3.0 * item.demand
            + 1.2 * target_bonus
            + 0.8 * freshness
            + 0.7 * item.failure_count
            - 1.8 * stale_risk
        )


class FreshnessAwarePolicy(MemoryPolicy):
    name = "freshness_aware"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        age = now - item.observed_at
        freshness = 1.0 / (1.0 + age)
        target_bonus = 1.0 if item.object_name == current_target else 0.0
        verification_need = item.volatility * math.log1p(age)
        utility = 3.0 * item.demand + 1.1 * target_bonus + 1.3 * freshness
        return utility - 2.8 * verification_need + 0.4 * item.failure_count

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        age = now - item.observed_at
        return item.volatility * math.log1p(age) > 0.32


class OraclePolicy(MemoryPolicy):
    name = "oracle"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        future = 0 if future_counts is None else future_counts.get(item.object_name, 0)
        age = now - item.observed_at
        return 10.0 * future + item.demand + 0.1 / (1.0 + age) - item.volatility


class FreshOraclePolicy(MemoryPolicy):
    name = "fresh_oracle"

    def score(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> float:
        future = 0 if future_counts is None else future_counts.get(item.object_name, 0)
        age = now - item.observed_at
        freshness_risk = item.volatility * math.log1p(age)
        return 10.0 * future + item.demand - 3.0 * freshness_risk

    def should_verify(self, item: MemoryItem, *, now: int) -> bool:
        age = now - item.observed_at
        return item.volatility * math.log1p(age) > 0.3


@dataclass
class MemoryStore:
    policy: MemoryPolicy
    budget: int
    items: List[MemoryItem] = field(default_factory=list)
    writes: int = 0
    evictions: int = 0

    def query(self, target: str, true_location: str) -> QueryOutcome:
        candidates = [item for item in self.items if item.object_name == target]
        if not candidates:
            return QueryOutcome(found=False, stale=False, item=None)
        best = max(candidates, key=lambda item: item.observed_at)
        return QueryOutcome(found=True, stale=best.location != true_location, item=best)

    def observe(
        self,
        item: MemoryItem,
        *,
        now: int,
        current_target: str,
        future_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        if self.budget <= 0 or isinstance(self.policy, NoMemoryPolicy):
            return

        self.writes += 1
        self.items = [
            existing
            for existing in self.items
            if not (
                existing.object_name == item.object_name
                and existing.location == item.location
                and existing.observed_at <= item.observed_at
            )
        ]
        self.items.append(item)

        while len(self.items) > self.budget:
            evict = min(
                self.items,
                key=lambda candidate: self.policy.score(
                    candidate,
                    now=now,
                    current_target=current_target,
                    future_counts=future_counts,
                ),
            )
            self.items.remove(evict)
            self.evictions += 1


@dataclass
class SyntheticWorld:
    profiles: Dict[str, ObjectProfile]
    locations: Dict[str, str]
    rng: random.Random
    drift_scale: float = 1.0

    @classmethod
    def create(cls, seed: int, drift_scale: float = 1.0) -> "SyntheticWorld":
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

    def sample_trace(self, n: int) -> List[TraceStep]:
        objects = list(self.profiles)
        weights = [self.profiles[name].demand for name in objects]
        trace: List[TraceStep] = []
        for now in range(n):
            target = self.rng.choices(objects, weights=weights, k=1)[0]
            self.maybe_drift()
            task = Task(target=target, true_location=self.locations[target])
            trace.append(TraceStep(task=task, observations=tuple(self.observe_search(target, now))))
        return trace

    def maybe_drift(self) -> None:
        for name, profile in self.profiles.items():
            if self.rng.random() < min(profile.volatility * 0.08 * self.drift_scale, 0.95):
                self.locations[name] = self.rng.choice(ROOMS)

    def observe_search(self, target: str, now: int, stale_failure: bool = False) -> List[MemoryItem]:
        observed = {target}
        distractor_count = self.rng.randint(2, 5)
        observed.update(self.rng.sample(list(self.profiles), k=distractor_count))
        items: List[MemoryItem] = []
        for name in observed:
            profile = self.profiles[name]
            items.append(
                MemoryItem(
                    object_name=name,
                    location=self.locations[name],
                    observed_at=now,
                    salience=profile.salience,
                    volatility=profile.volatility,
                    demand=profile.demand,
                    failure_count=1 if stale_failure and name == target else 0,
                )
            )
        return items


def future_counts(trace: Sequence[TraceStep], start: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for step in trace[start:]:
        counts[step.task.target] = counts.get(step.task.target, 0) + 1
    return counts


def run_policy(
    seed: int,
    budget: int,
    tasks_n: int,
    policy: MemoryPolicy,
    drift_scale: float = 1.0,
    task_budget: float = 7.0,
) -> PolicyMetrics:
    world = SyntheticWorld.create(seed, drift_scale=drift_scale)
    trace = world.sample_trace(tasks_n)
    store = MemoryStore(policy=policy, budget=0 if isinstance(policy, NoMemoryPolicy) else budget)
    metrics = PolicyMetrics(policy=policy.name)

    for idx, step in enumerate(trace):
        task = step.task
        now = idx
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        cost = 0.0
        stale_failure = False

        if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=now):
            metrics.verifications += 1
            cost += 2.0
            if outcome.stale:
                metrics.verification_catches += 1
                cost += 6.0
            else:
                metrics.query_hits += 1
                cost += 1.0
        elif outcome.found and not outcome.stale:
            metrics.query_hits += 1
            cost += 1.0
        elif outcome.found and outcome.stale:
            metrics.stale_errors += 1
            stale_failure = True
            cost += 8.0
        else:
            cost += 6.0

        if cost <= task_budget:
            metrics.successes += 1
        metrics.total_cost += cost

        remaining = future_counts(trace, idx + 1) if isinstance(policy, (OraclePolicy, FreshOraclePolicy)) else None
        for item in step.observations:
            if stale_failure and item.object_name == task.target:
                item = MemoryItem(
                    object_name=item.object_name,
                    location=item.location,
                    observed_at=item.observed_at,
                    salience=item.salience,
                    volatility=item.volatility,
                    demand=item.demand,
                    failure_count=1,
                )
            store.observe(item, now=now, current_target=task.target, future_counts=remaining)

    metrics.writes = store.writes
    metrics.evictions = store.evictions
    return metrics


def run_suite(
    seeds: Iterable[int],
    budgets: Sequence[int],
    tasks: int,
    drift_scale: float = 1.0,
    task_budget: float = 7.0,
) -> List[Dict[str, float | int | str]]:
    policies: Tuple[MemoryPolicy, ...] = (
        NoMemoryPolicy(),
        FifoPolicy(),
        SaliencePolicy(),
        TaskConditionedPolicy(),
        FreshnessAwarePolicy(),
        OraclePolicy(),
        FreshOraclePolicy(),
    )
    rows: List[Dict[str, float | int | str]] = []
    for seed in seeds:
        for budget in budgets:
            for policy in policies:
                metrics = run_policy(
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
    (out_dir / "offline_trace_pilot.json").write_text(
        json.dumps({"runs": list(rows), "summary": summary}, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "offline_trace_pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
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
    parser = argparse.ArgumentParser(description="Run a CPU-only embodied memory-policy pilot.")
    parser.add_argument("--seeds", type=int, default=5, help="number of deterministic seeds to run")
    parser.add_argument("--tasks", type=int, default=120, help="tasks per seed")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8], help="memory budgets")
    parser.add_argument("--drift-scale", type=float, default=1.0, help="multiplier for object relocation rate")
    parser.add_argument("--task-budget", type=float, default=5.0, help="maximum action/search cost for success")
    parser.add_argument("--out-dir", type=Path, default=Path("results/offline_trace_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_suite(
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
