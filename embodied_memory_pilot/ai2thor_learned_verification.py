"""Learned baseline: MLP stale-risk predictor for verification replay.

Trains a small MLP to predict whether a retained memory item will produce
a stale hit, using features from the item and the task context. Evaluates
via leave-one-seed-out cross-validation with the same recovery-by-search
fallback as the hand-set threshold policy.

No Unity needed — all data is in saved probes.
"""

from __future__ import annotations

import json
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    load_probe,
)
from embodied_memory_pilot.ai2thor_memory_eval import (
    MemoryItem,
    MemoryPolicy,
    MemoryStore,
    NoMemoryPolicy,
    PolicyMetrics,
    SaliencePolicy,
    TARGET_OBJECTS,
)
from embodied_memory_pilot.ai2thor_rearrangement_verification import (
    VerifyRearrangementRiskPolicy,
)


def _item_features(item: MemoryItem, query_now: float) -> List[float]:
    age = max(query_now - item.observed_at, 0.0)
    freshness = math.exp(-0.05 * age)
    return [
        freshness,
        float(item.salience),
        float(item.demand),
        float(item.volatility),
        min(age / 100.0, 1.0),
    ]


def _collect_stale_labels(probe_path: Path, budget: int = 16) -> List[Dict[str, Any]]:
    """Run salience policy on one seed, record which retained hits are stale."""
    probe = load_probe(probe_path)
    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe)
    if not items:
        return []

    query_now = max((it.observed_at for it in items), default=0) + 1
    store = MemoryStore(policy=SaliencePolicy(), budget=budget)
    for item in items:
        store.observe(item, now=item.observed_at, current_target=item.object_name)

    examples = []
    for task in tasks:
        outcome = store.query(task.target, task.true_location)
        if outcome.found and outcome.item is not None:
            feat = _item_features(outcome.item, query_now)
            examples.append({
                'features': feat,
                'stale': 1 if outcome.stale else 0,
                'object_name': outcome.item.object_name,
                'scene': task.scene,
            })
    return examples


# ── Simple MLP ─────────────────────────────────────────────────────

class SimpleMLP:
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        r = random.Random(42)
        self.hidden_dim = hidden_dim
        self.w1 = [[r.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [[r.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * hidden_dim
        self.w3 = [r.gauss(0, 0.1) for _ in range(hidden_dim)]
        self.b3 = 0.0
        self.input_dim = input_dim

    def _relu(self, x): return max(0.0, x)
    def _sigmoid(self, x):
        if x < -20: return 0.0
        if x > 20: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, features: List[float]) -> float:
        h1 = [self._relu(sum(self.w1[i][j] * features[j] for j in range(self.input_dim)) + self.b1[i])
              for i in range(self.hidden_dim)]
        h2 = [self._relu(sum(self.w2[i][j] * h1[j] for j in range(self.hidden_dim)) + self.b2[i])
              for i in range(self.hidden_dim)]
        z = sum(self.w3[i] * h2[i] for i in range(self.hidden_dim)) + self.b3
        return self._sigmoid(z)

    def train(self, examples: List[Dict], epochs: int = 150, lr: float = 0.01) -> List[float]:
        losses = []
        for _ in range(epochs):
            random.shuffle(examples)
            total_loss = 0.0
            for ex in examples:
                x, y_true = ex['features'], float(ex['stale'])
                # Forward
                h1_raw = [sum(self.w1[i][j]*x[j] for j in range(self.input_dim))+self.b1[i] for i in range(self.hidden_dim)]
                h1 = [self._relu(v) for v in h1_raw]
                h2_raw = [sum(self.w2[i][j]*h1[j] for j in range(self.hidden_dim))+self.b2[i] for i in range(self.hidden_dim)]
                h2 = [self._relu(v) for v in h2_raw]
                z = sum(self.w3[i]*h2[i] for i in range(self.hidden_dim))+self.b3
                y_pred = self._sigmoid(z)
                eps = 1e-7
                total_loss += -(y_true*math.log(y_pred+eps)+(1-y_true)*math.log(1-y_pred+eps))
                # Backward
                dz = y_pred - y_true
                for i in range(self.hidden_dim): self.w3[i] -= lr*dz*h2[i]
                self.b3 -= lr*dz
                dh2 = [dz*self.w3[i] for i in range(self.hidden_dim)]
                for i in range(self.hidden_dim):
                    if h2_raw[i] > 0:
                        for j in range(self.hidden_dim): self.w2[i][j] -= lr*dh2[i]*h1[j]
                        self.b2[i] -= lr*dh2[i]
                dh1 = [sum(dh2[i]*self.w2[i][j] for i in range(self.hidden_dim) if h2_raw[i]>0) for j in range(self.hidden_dim)]
                for i in range(self.hidden_dim):
                    if h1_raw[i] > 0:
                        for j in range(self.input_dim): self.w1[i][j] -= lr*dh1[i]*x[j]
                        self.b1[i] -= lr*dh1[i]
            losses.append(total_loss/max(len(examples),1))
        return losses


# ── MLP verification policy ────────────────────────────────────────

class MLPVerificationPolicy(MemoryPolicy):
    """Policy that uses MLP-predicted risk, with salience retention + recovery-by-search."""

    def __init__(self, mlp: SimpleMLP, threshold: float = 0.5):
        self.name = f"mlp_verify_t{threshold:.1f}"
        self.mlp = mlp
        self.threshold = threshold
        self._query_now: float = 0.0

    def set_query_now(self, now: float):
        self._query_now = now

    def should_verify(self, item: MemoryItem, now: float = 0) -> bool:
        feat = _item_features(item, max(now, self._query_now))
        return self.mlp.predict(feat) > self.threshold


def _evaluate_mlp_on_probe(
    probe_path: Path, policy: MLPVerificationPolicy, budget: int, verification_cost: float = 0.5
) -> Dict[str, Any]:
    probe = load_probe(probe_path)
    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe)
    if not items: return {}
    query_now = max((it.observed_at for it in items), default=0) + 1
    policy.set_query_now(query_now)
    store = MemoryStore(policy=SaliencePolicy(), budget=budget)
    for it in items:
        store.observe(it, now=it.observed_at, current_target=it.object_name)
    metrics = PolicyMetrics(policy=policy.name)
    for task in tasks:
        metrics.tasks += 1
        outcome = store.query(task.target, task.true_location)
        if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=query_now):
            metrics.verifications += 1; metrics.total_cost += verification_cost
            if outcome.stale:
                metrics.verification_catches += 1; metrics.successes += 1
                metrics.total_cost += task.scene_search_cost
            else:
                metrics.successes += 1; metrics.query_hits += 1
                metrics.total_cost += task.direct_action_cost
        elif outcome.found and not outcome.stale:
            metrics.successes += 1; metrics.query_hits += 1
            metrics.total_cost += task.direct_action_cost
        elif outcome.found and outcome.stale:
            metrics.stale_errors += 1; metrics.total_cost += task.scene_search_cost
        else:
            metrics.total_cost += task.scene_search_cost
    row = metrics.as_dict(seed=0, budget=budget)
    row["mode"] = "mlp_learned_verification"
    row["rearrangement_tasks"] = len(tasks)
    row["memory_items"] = len(items)
    row["avg_reachable_cost"] = round(metrics.avg_cost, 4)
    row["completion_rate"] = round(metrics.successes / max(metrics.tasks, 1), 4)
    return row


# ── Main CV ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, nargs="+",
                        default=[Path(f"results/ai2thor_rearrangement_6scene_seed{s}/ai2thor_rearrangement_probe.json")
                                 for s in [7,11,13,17,19,29,31,37,41,43]])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--verification-cost", type=float, default=0.5)
    parser.add_argument("--mlp-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_learned_verification"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    n_seeds = len(args.probes)

    for held_out_idx in range(n_seeds):
        held_out = args.probes[held_out_idx]
        train_paths = [p for i, p in enumerate(args.probes) if i != held_out_idx]

        # Collect training examples from train seeds
        train_examples = []
        for tp in train_paths:
            train_examples.extend(_collect_stale_labels(tp))
        if not train_examples:
            continue
        n_stale = sum(1 for e in train_examples if e['stale'])
        if n_stale == 0:
            continue

        mlp = SimpleMLP(input_dim=len(train_examples[0]['features']), hidden_dim=16)
        mlp.train(train_examples, epochs=args.epochs)

        for thresh in args.mlp_thresholds:
            policy = MLPVerificationPolicy(mlp, threshold=thresh)
            for budget in args.budgets:
                result = _evaluate_mlp_on_probe(held_out, policy, budget, args.verification_cost)
                if result:
                    result['held_out_idx'] = held_out_idx
                    all_runs.append(result)

    # Aggregate
    groups = defaultdict(list)
    for r in all_runs:
        key = (r.get('budget', 0), float(r.get('policy', '').replace('mlp_verify_t', '') or '0.5'))
        groups[key].append(r)

    summary = []
    for (budget, thresh), group in sorted(groups.items()):
        n = len(group)
        comps = [r.get('completion_rate', 0) for r in group]
        stales = [r.get('stale_errors', 0) for r in group]
        costs = [r.get('avg_reachable_cost', 0) for r in group]
        verifs = [r.get('verifications', 0) for r in group]
        catches = [r.get('verification_catches', 0) for r in group]
        summary.append({
            'budget': budget, 'mlp_threshold': thresh, 'seeds': n,
            'completion_rate_mean': round(mean(comps), 4) if n else 0,
            'stale_errors_mean': round(mean(stales), 4) if n else 0,
            'avg_reachable_cost_mean': round(mean(costs), 4) if n else 0,
            'verifications_mean': round(mean(verifs), 1) if n else 0,
            'verification_catches_mean': round(mean(catches), 1) if n else 0,
        })

    # Write
    with open(args.out_dir / "mlp_verification_cv.json", "w") as f:
        json.dump({'runs': all_runs, 'summary': summary}, f, indent=2)
    if summary:
        with open(args.out_dir / "mlp_verification_cv_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)

    print(f"\nMLP CV Results ({len(all_runs)} runs, {len(summary)} aggregate rows):")
    print(f"{'Budget':>6s} {'Thresh':>6s} {'Compl':>8s} {'Stale':>6s} {'Cost':>8s} {'Verif':>6s} {'Catch':>6s} {'Seeds':>5s}")
    for row in summary:
        print(f"{row['budget']:6d} {row['mlp_threshold']:6.2f} {row['completion_rate_mean']:8.4f} {row['stale_errors_mean']:6.1f} {row['avg_reachable_cost_mean']:8.4f} {row['verifications_mean']:6.1f} {row['verification_catches_mean']:6.1f} {row['seeds']:5d}")

    print(f"\n--- Reference: hand-set verify_all (b16, vc0.5) ---")
    print(f"Hand-set: compl=0.7707 stale=0.0 cost=8.56 verif=4.6 catch=2.7 (n=10)")
    for row in summary:
        if row['budget'] == 16:
            print(f"MLP(t={row['mlp_threshold']:.1f}): compl={row['completion_rate_mean']:.4f} stale={row['stale_errors_mean']:.1f} cost={row['avg_reachable_cost_mean']:.4f} verif={row['verifications_mean']:.1f} catch={row['verification_catches_mean']:.1f} (n={row['seeds']})")


if __name__ == "__main__":
    main()
