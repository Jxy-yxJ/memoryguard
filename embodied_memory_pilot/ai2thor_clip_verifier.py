"""CLIP-based perception verifier for AI2-THOR rearrangement replay.

Replaces oracle metadata checks with visual feature matching:
1. During "before" observation: capture per-object RGB crops, compute CLIP embeddings
2. During "after" verification: compare stored embedding with after-state crop
3. If cosine_sim < threshold -> object is stale -> fall back to scene search

This is a simulation proxy for the real-robot perception verifier.
No Unity needed for inference — only for probe collection with image capture.
"""

from __future__ import annotations

import math
import json
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ── CLIP model wrapper ────────────────────────────────────────────

class CLIPVerifier:
    """Lazy-loading CLIP model for image embedding and comparison."""

    _instance: Optional[CLIPVerifier] = None

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    @classmethod
    def get_instance(cls, model_name: str = "openai/clip-vit-base-patch32") -> CLIPVerifier:
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
        self._model.eval()

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Compute L2-normalized CLIP image embedding. Input: HWC RGB uint8."""
        self._load()
        pil_image = Image.fromarray(image)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            emb = self._model.get_image_features(**inputs)
        emb = emb.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))

    def is_stale(self, before_embedding: np.ndarray, after_embedding: np.ndarray,
                 threshold: float = 0.85) -> bool:
        return self.cosine_similarity(before_embedding, after_embedding) < threshold


# ── Image loading and cropping ─────────────────────────────────────

def load_image_crop(image_path: Path) -> np.ndarray:
    """Load a saved per-object crop as an RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def crop_from_mask(frame: np.ndarray, mask: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """Crop frame to mask bounding box with proportional padding."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return frame
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h, w = y2 - y1, x2 - x1
    pad_y, pad_x = int(h * padding), int(w * padding)
    y1 = max(0, y1 - pad_y)
    y2 = min(frame.shape[0], y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(frame.shape[1], x2 + pad_x)
    if y2 <= y1 or x2 <= x1:
        return frame
    return frame[y1:y2, x1:x2]


# ── Embedding cache ─────────────────────────────────────────────────

class CLIPEmbeddingCache:
    """Holds precomputed before/after CLIP embeddings for all objects in a probe."""

    def __init__(self, probe: Dict[str, Any], image_dir: Path,
                 verifier: Optional[CLIPVerifier] = None):
        self.verifier = verifier or CLIPVerifier.get_instance()
        self.before_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.after_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self._build(probe, image_dir)

    def _build(self, probe: Dict[str, Any], image_dir: Path) -> None:
        for scene_entry in probe.get("scenes", []):
            if scene_entry.get("status") != "ok":
                continue
            scene_name = str(scene_entry.get("scene", ""))
            for phase, store in [("before", self.before_embeddings), ("after", self.after_embeddings)]:
                phase_dir = image_dir / scene_name / phase
                if not phase_dir.is_dir():
                    continue
                store[scene_name] = {}
                for img_file in sorted(phase_dir.glob("*.jpg")):
                    if "_frame" in img_file.name:
                        continue  # skip full-frame fallback
                    # Filename: {action_idx}_{objectId}.jpg — extract object_type
                    stem = img_file.stem
                    parts = stem.split("_", 1)
                    obj_id = parts[1] if len(parts) > 1 else stem
                    object_type = obj_id.split("|")[0]  # "CounterTop|-01.81|..." -> "CounterTop"
                    try:
                        img = load_image_crop(img_file)
                        # Use first (earliest action) crop for each object_type
                        if object_type not in store[scene_name]:
                            store[scene_name][object_type] = self.verifier.embed_image(img)
                    except Exception:
                        continue

    def get_before_embedding(self, scene: str, object_type: str) -> Optional[np.ndarray]:
        return self.before_embeddings.get(scene, {}).get(object_type)

    def get_after_embedding(self, scene: str, object_type: str) -> Optional[np.ndarray]:
        return self.after_embeddings.get(scene, {}).get(object_type)

    def has_data(self) -> bool:
        return bool(self.before_embeddings) and bool(self.after_embeddings)


# ── CLIP-based evaluation (parallels the oracle evaluator) ──────────

def evaluate_clip_verification(
    probe: Dict[str, Any],
    image_dir: Path,
    policy_name: str = "verify_rearrangement_risk",
    budgets: Sequence[int] = (4, 8, 16),
    verification_cost: float = 0.5,
    clip_thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
) -> List[Dict[str, Any]]:
    """Run CLIP-based verification and compare against oracle on the same probe."""
    from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
        build_rearrangement_memory_items,
        build_rearrangement_tasks,
        load_probe,
    )
    from embodied_memory_pilot.ai2thor_memory_eval import (
        MemoryStore, PolicyMetrics, SaliencePolicy, TARGET_OBJECTS,
    )
    from embodied_memory_pilot.ai2thor_rearrangement_verification import (
        VerifyRearrangementRiskPolicy,
    )

    verifier = CLIPVerifier.get_instance()
    embedding_cache = CLIPEmbeddingCache(probe, image_dir, verifier)
    if not embedding_cache.has_data():
        return []  # No images to work with

    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe)
    if not items or not tasks:
        return []

    query_now = max((it.observed_at for it in items), default=0) + 1
    results = []

    for budget in budgets:
        for threshold in clip_thresholds:
            policy = VerifyRearrangementRiskPolicy()
            store = MemoryStore(policy=SaliencePolicy(), budget=budget)

            for item in items:
                store.observe(item, now=item.observed_at, current_target=item.object_name)

            # Oracle run
            oracle_metrics = PolicyMetrics(policy="oracle")
            # CLIP run
            clip_metrics = PolicyMetrics(policy=f"clip_t{threshold:.2f}")

            for task in tasks:
                outcome = store.query(task.target, task.true_location)
                oracle_metrics.tasks += 1
                clip_metrics.tasks += 1

                if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=query_now):
                    # Both would verify
                    oracle_metrics.verifications += 1
                    clip_metrics.verifications += 1

                    # Oracle: use ground-truth staleness
                    if outcome.stale:
                        oracle_metrics.verification_catches += 1
                        oracle_metrics.successes += 1
                        oracle_metrics.total_cost += verification_cost + task.scene_search_cost
                    else:
                        oracle_metrics.successes += 1
                        oracle_metrics.query_hits += 1
                        oracle_metrics.total_cost += verification_cost + task.direct_action_cost

                    # CLIP: compare per-object crop embeddings (before vs after)
                    scene_name = task.scene
                    object_type = task.target  # e.g. "Cup", "Fork"
                    before_emb = embedding_cache.get_before_embedding(scene_name, object_type)
                    after_emb = embedding_cache.get_after_embedding(scene_name, object_type)

                    clip_stale = True  # default: can't verify -> conservative
                    if before_emb is not None and after_emb is not None:
                        clip_stale = verifier.is_stale(before_emb, after_emb, threshold)

                    if clip_stale:
                        clip_metrics.verification_catches += 1
                        clip_metrics.successes += 1
                        clip_metrics.total_cost += verification_cost + task.scene_search_cost
                    else:
                        clip_metrics.successes += 1
                        clip_metrics.query_hits += 1
                        clip_metrics.total_cost += verification_cost + task.direct_action_cost

                elif outcome.found and not outcome.stale:
                    oracle_metrics.successes += 1
                    oracle_metrics.query_hits += 1
                    oracle_metrics.total_cost += task.direct_action_cost
                    clip_metrics.successes += 1
                    clip_metrics.query_hits += 1
                    clip_metrics.total_cost += task.direct_action_cost
                elif outcome.found and outcome.stale:
                    oracle_metrics.stale_errors += 1
                    oracle_metrics.total_cost += task.scene_search_cost
                    clip_metrics.stale_errors += 1
                    clip_metrics.total_cost += task.scene_search_cost
                else:
                    oracle_metrics.total_cost += task.scene_search_cost
                    clip_metrics.total_cost += task.scene_search_cost

            results.append({
                "budget": budget,
                "clip_threshold": threshold,
                "seeds": 1,
                "oracle_completion_rate": round(oracle_metrics.successes / max(oracle_metrics.tasks, 1), 4),
                "clip_completion_rate": round(clip_metrics.successes / max(clip_metrics.tasks, 1), 4),
                "oracle_stale_errors": oracle_metrics.stale_errors,
                "clip_stale_errors": clip_metrics.stale_errors,
                "oracle_avg_cost": round(oracle_metrics.avg_cost, 4),
                "clip_avg_cost": round(clip_metrics.avg_cost, 4),
                "oracle_verifications": oracle_metrics.verifications,
                "clip_verifications": clip_metrics.verifications,
                "oracle_catches": oracle_metrics.verification_catches,
                "clip_catches": clip_metrics.verification_catches,
            })

    return results


def compute_agreement(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate CLIP vs oracle comparison across runs."""
    if not results:
        return {}
    best = max(results, key=lambda r: r.get("clip_completion_rate", 0))
    return {
        "n_runs": len(results),
        "best_threshold": best.get("clip_threshold", 0),
        "best_clip_completion": best.get("clip_completion_rate", 0),
        "best_oracle_completion": best.get("oracle_completion_rate", 0),
        "completion_delta": round(best.get("clip_completion_rate", 0) - best.get("oracle_completion_rate", 0), 4),
        "best_clip_stale_errors": best.get("clip_stale_errors", 0),
    }


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-based verification vs oracle comparison")
    parser.add_argument("--probe", type=Path, required=True,
                        help="Path to probe JSON (must have associated image directory)")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Path to image directory ({probe_dir}/images/)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--clip-thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--verification-cost", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_clip_verification"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.probe) as f:
        probe = json.load(f)

    # Init verifier
    CLIPVerifier.get_instance(model_name=args.model)
    print(f"CLIP model: {args.model}")

    results = evaluate_clip_verification(
        probe=probe,
        image_dir=args.image_dir,
        budgets=args.budgets,
        verification_cost=args.verification_cost,
        clip_thresholds=args.clip_thresholds,
    )

    if not results:
        print("No results — image directory may be empty or probe has no images")
        return

    agreement = compute_agreement(results)

    # Write outputs
    with open(args.out_dir / "clip_vs_oracle.json", "w") as f:
        json.dump({"results": results, "agreement": agreement}, f, indent=2)

    with open(args.out_dir / "clip_vs_oracle_summary.csv", "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    # Print
    print(f"\nCLIP vs Oracle Comparison ({len(results)} runs):")
    print(f"{'Budget':>6s} {'Thr':>6s} {'OraCompl':>10s} {'ClipCompl':>10s} {'OraStale':>9s} {'ClipStale':>10s} {'OraCost':>8s} {'ClipCost':>8s}")
    for r in results:
        print(f"{r['budget']:6d} {r['clip_threshold']:6.2f} {r['oracle_completion_rate']:10.4f} {r['clip_completion_rate']:10.4f} {r['oracle_stale_errors']:9d} {r['clip_stale_errors']:10d} {r['oracle_avg_cost']:8.4f} {r['clip_avg_cost']:8.4f}")

    print(f"\nBest: threshold={agreement['best_threshold']} "
          f"clip_compl={agreement['best_clip_completion']:.4f} "
          f"delta={agreement['completion_delta']:+.4f}")

    print(f"\nWrote {args.out_dir}/clip_vs_oracle.json and _summary.csv")


if __name__ == "__main__":
    main()
