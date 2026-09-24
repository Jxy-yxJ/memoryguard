"""Object-crop CLIP verifier: crops manifest bbox region from before/after full frames,
computes CLIP image similarity to detect staleness.

Unlike `ai2thor_clip_verifier.py` which reads pre-saved per-object crop files from
`_save_object_images()`, this verifier crops full frames at the manifest bbox_xyxy position.
This enables a fairer comparison to GSAM: both use spatial priors from the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ── CLIP model wrapper ────────────────────────────────────────────

class CLIPVerifier:
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

    def is_stale(self, before_emb: np.ndarray, after_emb: np.ndarray, threshold: float = 0.85) -> bool:
        return self.cosine_similarity(before_emb, after_emb) < threshold


# ── Manifest-based frame cropping ─────────────────────────────────

def load_manifest(image_dir: Path) -> List[Dict[str, Any]]:
    """Load all image_manifest.jsonl entries from subdirectories."""
    entries: List[Dict[str, Any]] = []
    for mpath in sorted(image_dir.rglob("image_manifest.jsonl")):
        for line in mpath.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def crop_frame_region(frame_path: Path, bbox_xyxy: List[float]) -> np.ndarray:
    """Crop [x1, y1, x2, y2] region from image. Returns HWC RGB uint8."""
    img = Image.open(frame_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, x2 = max(0, x1), min(img.width, x2)
    y1, y2 = max(0, y1), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array(img)
    return np.array(img.crop((x1, y1, x2, y2)))


# ── Frame-crop embedding cache ────────────────────────────────────

class FrameCropEmbeddingCache:
    """Holds CLIP embeddings for manifest-bbox-cropped before/after frame regions."""

    def __init__(self, probe: Dict[str, Any], image_dir: Path,
                 verifier: Optional[CLIPVerifier] = None):
        self.verifier = verifier or CLIPVerifier.get_instance()
        self.before_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.after_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.similarities: Dict[str, float] = {}  # key = "scene|object_type"
        self._build(probe, image_dir)

    def _build(self, probe: Dict[str, Any], image_dir: Path) -> None:
        manifest_entries = load_manifest(image_dir)
        before_entries = [e for e in manifest_entries if e.get("phase") == "before"]
        after_entries = [e for e in manifest_entries if e.get("phase") == "after"]

        # Index after entries by (scene, action_idx, object_id)
        after_index: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        for e in after_entries:
            key = (str(e.get("scene", "")), int(e.get("action_idx", 0)), str(e.get("object_id", "")))
            after_index[key] = e

        for entry in before_entries:
            scene = str(entry.get("scene", ""))
            action_idx = int(entry.get("action_idx", 0))
            object_id = str(entry.get("object_id", ""))
            object_type = object_id.split("|")[0] if "|" in object_id else object_id
            bbox_xyxy = entry.get("bbox_xyxy")
            frame_path_str = entry.get("frame_path", "")

            if not bbox_xyxy or not frame_path_str:
                continue

            before_frame_path = Path(frame_path_str)
            if not before_frame_path.exists():
                continue

            # Crop before frame at bbox
            try:
                before_crop = crop_frame_region(before_frame_path, bbox_xyxy)
                before_emb = self.verifier.embed_image(before_crop)
            except Exception:
                continue

            # Find matching after entry
            after_key = (scene, action_idx, object_id)
            after_entry = after_index.get(after_key)

            if after_entry is None:
                # Try with same action_idx (unchanged object)
                after_entry = after_index.get(after_key)

            if after_entry is not None:
                after_frame_str = after_entry.get("frame_path", "")
                if after_frame_str:
                    after_frame_path = Path(after_frame_str)
                    if after_frame_path.exists():
                        try:
                            after_crop = crop_frame_region(after_frame_path, bbox_xyxy)
                            after_emb = self.verifier.embed_image(after_crop)
                            sim = self.verifier.cosine_similarity(before_emb, after_emb)
                            cache_key = f"{scene}|{object_type}"
                            self.before_embeddings.setdefault(scene, {})[object_type] = before_emb
                            self.after_embeddings.setdefault(scene, {})[object_type] = after_emb
                            self.similarities[cache_key] = sim
                        except Exception:
                            continue

    def get_before_embedding(self, scene: str, object_type: str) -> Optional[np.ndarray]:
        return self.before_embeddings.get(scene, {}).get(object_type)

    def get_after_embedding(self, scene: str, object_type: str) -> Optional[np.ndarray]:
        return self.after_embeddings.get(scene, {}).get(object_type)

    def get_similarity(self, scene: str, object_type: str) -> Optional[float]:
        return self.similarities.get(f"{scene}|{object_type}")

    def has_data(self) -> bool:
        return bool(self.similarities)


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_object_crop_clip_verification(
    probe: Dict[str, Any],
    image_dir: Path,
    budgets: Sequence[int] = (4, 8, 16),
    verification_cost: float = 0.5,
    clip_thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
) -> List[Dict[str, Any]]:
    """Run object-crop CLIP verification and compare against oracle."""
    from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
        build_rearrangement_memory_items,
        build_rearrangement_tasks,
    )
    from embodied_memory_pilot.ai2thor_memory_eval import (
        MemoryStore, PolicyMetrics, SaliencePolicy,
    )
    from embodied_memory_pilot.ai2thor_rearrangement_verification import (
        VerifyRearrangementRiskPolicy,
    )

    verifier = CLIPVerifier.get_instance()
    crop_cache = FrameCropEmbeddingCache(probe, image_dir, verifier)

    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe)
    if not items or not tasks:
        return []

    query_now = max((it.observed_at for it in items), default=0) + 1
    results: List[Dict[str, Any]] = []

    # Per-decision tracking for precision/recall
    all_decisions: List[Dict[str, Any]] = []

    for budget in budgets:
        for threshold in clip_thresholds:
            policy = VerifyRearrangementRiskPolicy()
            store = MemoryStore(policy=SaliencePolicy(), budget=budget)

            for item in items:
                store.observe(item, now=item.observed_at, current_target=item.object_name)

            oracle_metrics = PolicyMetrics(policy="oracle")
            clip_metrics = PolicyMetrics(policy=f"object_crop_clip_t{threshold:.2f}")

            for task in tasks:
                outcome = store.query(task.target, task.true_location)
                oracle_metrics.tasks += 1
                clip_metrics.tasks += 1

                if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=query_now):
                    oracle_metrics.verifications += 1
                    clip_metrics.verifications += 1

                    # Oracle
                    if outcome.stale:
                        oracle_metrics.verification_catches += 1
                        oracle_metrics.successes += 1
                        oracle_metrics.total_cost += verification_cost + task.scene_search_cost
                    else:
                        oracle_metrics.query_hits += 1
                        oracle_metrics.successes += 1
                        oracle_metrics.total_cost += verification_cost + task.direct_action_cost

                    # Object-crop CLIP
                    scene = str(task.scene)
                    obj_type = str(task.target)
                    clip_stale = crop_cache.get_similarity(scene, obj_type) is not None and \
                        verifier.is_stale(
                            crop_cache.get_before_embedding(scene, obj_type),
                            crop_cache.get_after_embedding(scene, obj_type),
                            threshold,
                        ) if crop_cache.get_before_embedding(scene, obj_type) is not None \
                           and crop_cache.get_after_embedding(scene, obj_type) is not None \
                           else True  # conservative: can't verify -> assume stale

                    all_decisions.append({
                        "scene": scene,
                        "target": obj_type,
                        "oracle_stale": outcome.stale,
                        "crop_clip_stale": clip_stale,
                        "threshold": threshold,
                        "budget": budget,
                        "crop_sim": crop_cache.get_similarity(scene, obj_type),
                    })

                    if clip_stale:
                        clip_metrics.verification_catches += 1
                        clip_metrics.successes += 1
                        clip_metrics.total_cost += verification_cost + task.scene_search_cost
                    else:
                        clip_metrics.query_hits += 1
                        clip_metrics.successes += 1
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

            n_tasks = max(oracle_metrics.tasks, 1)
            result = {
                "budget": budget,
                "clip_threshold": threshold,
                "seeds": 1,
                "oracle_completion_rate": round(oracle_metrics.successes / n_tasks, 4),
                "object_crop_clip_completion_rate": round(clip_metrics.successes / n_tasks, 4),
                "oracle_stale_errors": oracle_metrics.stale_errors,
                "object_crop_clip_stale_errors": clip_metrics.stale_errors,
                "oracle_avg_cost": round(oracle_metrics.total_cost / n_tasks, 4),
                "object_crop_clip_avg_cost": round(clip_metrics.total_cost / n_tasks, 4),
                "oracle_verifications": oracle_metrics.verifications,
                "object_crop_clip_verifications": clip_metrics.verifications,
                "oracle_catches": oracle_metrics.verification_catches,
                "object_crop_clip_catches": clip_metrics.verification_catches,
            }
            results.append(result)

    return results


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Object-crop CLIP verifier")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[16])
    parser.add_argument("--clip-thresholds", type=float, nargs="+",
                        default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--verification-cost", type=float, default=0.5)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/ai2thor_object_crop_clip_verification"))
    parser.add_argument("--seed", type=int, default=0, help="Seed label for output")
    args = parser.parse_args()

    probe = json.loads(args.probe.read_text())
    results = evaluate_object_crop_clip_verification(
        probe=probe,
        image_dir=args.image_dir,
        budgets=args.budgets,
        verification_cost=args.verification_cost,
        clip_thresholds=args.clip_thresholds,
    )

    # Setup output dir
    out_dir = args.out_dir
    if args.seed:
        out_dir = out_dir.parent / f"{out_dir.name}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = out_dir / "object_crop_clip_vs_oracle.json"
    json_path.write_text(json.dumps({"results": results, "config": {
        "model": args.model,
        "thresholds": args.clip_thresholds,
        "budgets": args.budgets,
        "seed": args.seed,
    }}, indent=2))

    # Write CSV summary
    csv_path = out_dir / "object_crop_clip_vs_oracle_summary.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    # Print summary
    print(f"\nResults saved to {out_dir}")
    best = None
    for r in results:
        oracle_cr = r["oracle_completion_rate"]
        clip_cr = r["object_crop_clip_completion_rate"]
        delta = round(clip_cr - oracle_cr, 4)
        print(f"  budget={r['budget']} threshold={r['clip_threshold']:.2f} "
              f"oracle={oracle_cr} crop_clip={clip_cr} delta={delta}")
        if best is None or abs(delta) < abs(best_delta := (
            (best["object_crop_clip_completion_rate"] - best["oracle_completion_rate"]) if best else 0)):
            best = r

    if best:
        print(f"\nBest: threshold={best['clip_threshold']} "
              f"oracle={best['oracle_completion_rate']} crop_clip={best['object_crop_clip_completion_rate']} "
              f"catches={best['oracle_catches']}/{best['object_crop_clip_catches']}")

    # Write README
    readme = out_dir / "README.md"
    readme.write_text(f"""# Object-Crop CLIP Verification

Model: `{args.model}`
Seed: `{args.seed}`
OK scenes: `{probe['summary']['ok_scenes']}`
Moved paired objects: `{probe['summary']['moved_objects']}`

Verification method: crop manifest bbox_xyxy region from before/after full frames,
compare CLIP image embeddings.
""")


if __name__ == "__main__":
    main()
