"""Grounded-SAM2-style location verifier for AI2-THOR rearrangement replay.

The CLIP verifier asks whether before/after crops look similar. This module
instead asks whether the target category is still detected at the remembered
location. GroundingDINO/SAM2 dependencies are optional so the replay evaluator
and tests can run without installing heavy vision packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, TypeVar

from embodied_memory_pilot.ai2thor_memory_eval import TARGET_OBJECTS
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    RearrangementTask,
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    load_probe,
    summarize_rearrangement_probe,
)
from embodied_memory_pilot.ai2thor_rearrangement_verification import VerifyRearrangementRiskPolicy
from embodied_memory_pilot.pilot import MemoryStore, PolicyMetrics, SaliencePolicy


LOCATION_RE = re.compile(
    r"^(?P<scene>[^:]+):(?P<object_id>.+)@(?P<phase>before|after)$"
)


@dataclass(frozen=True)
class ParsedLocation:
    scene: str
    object_id: str
    phase: str
    object_type: str
    world_x: Optional[float]
    world_z: Optional[float]


@dataclass(frozen=True)
class LocationDecision:
    stale: bool
    confidence: float
    reason: str
    matched_distance: Optional[float] = None
    detections_considered: int = 0


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    score: float
    center: Tuple[float, float]
    coordinate_space: str
    image_path: Optional[str] = None
    mask_area: Optional[float] = None
    xyxy: Optional[Tuple[float, float, float, float]] = None


class BackendLoadTimeoutError(TimeoutError):
    """Raised when optional perception backend initialization exceeds its budget."""


T = TypeVar("T")


@dataclass(frozen=True)
class ImageManifestEntry:
    scene: str
    phase: str
    action_idx: int
    object_id: str
    object_type: str
    bbox_xyxy: Tuple[float, float, float, float]
    frame_path: Path
    crop_path: Path


class LocationVerifierBackend(Protocol):
    name: str
    coordinate_space: str
    threshold_unit: str

    def verify(
        self,
        task: RearrangementTask,
        *,
        remembered_location: str,
        image_dir: Path,
        threshold: float,
    ) -> LocationDecision:
        ...


def _parse_float_token(token: str) -> Optional[float]:
    try:
        return float(token)
    except ValueError:
        return None


def parse_rearrangement_location(location: str) -> ParsedLocation:
    match = LOCATION_RE.match(location)
    if not match:
        raise ValueError(f"Unsupported rearrangement location: {location}")
    object_id = match.group("object_id")
    parts = object_id.split("|")
    object_type = parts[0] if parts else "Unknown"
    world_x = _parse_float_token(parts[1]) if len(parts) > 1 else None
    world_z = _parse_float_token(parts[3]) if len(parts) > 3 else None
    return ParsedLocation(
        scene=match.group("scene"),
        object_id=object_id,
        phase=match.group("phase"),
        object_type=object_type,
        world_x=world_x,
        world_z=world_z,
    )


def _world_distance(a: ParsedLocation | ObjectDetection, b: ObjectDetection) -> float:
    if isinstance(a, ParsedLocation):
        if a.world_x is None or a.world_z is None:
            return math.inf
        ax, az = a.world_x, a.world_z
    else:
        ax, az = a.center
    bx, bz = b.center
    return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)


def _parse_crop_filename(path: Path) -> Optional[ObjectDetection]:
    if "_frame" in path.stem:
        return None
    _, _, object_id = path.stem.partition("_")
    if not object_id:
        return None
    parts = object_id.split("|")
    if len(parts) < 4:
        return None
    world_x = _parse_float_token(parts[1])
    world_z = _parse_float_token(parts[3])
    if world_x is None or world_z is None:
        return None
    return ObjectDetection(
        label=parts[0],
        score=1.0,
        center=(world_x, world_z),
        coordinate_space="world_xz",
        image_path=str(path),
    )


def _stale_failure_cost(task_by_old_location: Dict[str, RearrangementTask], task: RearrangementTask, location: str) -> float:
    stale_task = task_by_old_location.get(location)
    old_cost = stale_task.old_action_cost if stale_task else task.old_action_cost
    return old_cost + task.scene_search_cost


class CropFilenameLocationBackend:
    """Metadata-proxy backend for offline evaluator validation.

    It reads object type and world position from AI2-THOR crop filenames. This
    is not a publishable perception baseline; it validates the location-verifier
    metrics before heavy GroundingDINO/SAM2 installation.
    """

    name = "crop_filename_proxy"
    coordinate_space = "world_xz"
    threshold_unit = "ai2thor_world_meters"

    def _after_detections(self, image_dir: Path, scene: str, label: str) -> List[ObjectDetection]:
        phase_dir = image_dir / scene / "after"
        detections: List[ObjectDetection] = []
        for path in sorted(phase_dir.glob("*.jpg")):
            detection = _parse_crop_filename(path)
            if detection is not None and detection.label == label:
                detections.append(detection)
        return detections

    def verify(
        self,
        task: RearrangementTask,
        *,
        remembered_location: str,
        image_dir: Path,
        threshold: float,
    ) -> LocationDecision:
        remembered = parse_rearrangement_location(remembered_location)
        detections = self._after_detections(image_dir, task.scene, task.target)
        if not detections:
            return LocationDecision(
                stale=True,
                confidence=0.0,
                reason="no_after_detection",
                detections_considered=0,
            )
        distances = [_world_distance(remembered, detection) for detection in detections]
        best_distance = min(distances)
        stale = best_distance > threshold
        return LocationDecision(
            stale=stale,
            confidence=1.0,
            reason="nearest_after_detection_distance",
            matched_distance=round(best_distance, 4),
            detections_considered=len(detections),
        )




def _load_case_list(case_list_path: Path) -> list[dict[str, Any]]:
    data = json.loads(case_list_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    return [dict(case) for case in cases if isinstance(case, dict)]






def _load_image_manifest_entries(image_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(image_dir.rglob('image_manifest.jsonl')):
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            raw['frame_path_abs'] = str(image_dir / str(raw.get('frame_path') or ''))
            entries.append(raw)
    return entries


def _match_case_to_manifest(case: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    scene = str(case.get('scene') or '')
    target = str(case.get('target') or '')
    target_object_id = str(case.get('target_object_id') or case.get('object_id') or '')
    candidates = [entry for entry in entries if str(entry.get('scene') or '') == scene and str(entry.get('phase') or '') == 'after']
    if target_object_id:
        for entry in candidates:
            if str(entry.get('object_id') or '') == target_object_id:
                return entry, 'exact_object_id_manifest_match'
    for entry in candidates:
        if str(entry.get('object_type') or '') == target:
            return entry, 'same_type_manifest_match'
    return None, 'no_manifest_match'






def make_scanrefer_style_localization_provenance(image_dir: Path, case_list_path: Path) -> dict[str, Any]:
    try:
        cases = _load_case_list(case_list_path)
        case_count = len(cases)
    except Exception:
        case_count = 0
    return {
        'schema': '0514_scanrefer_style_localization_provenance.v1',
        'status': 'blocked',
        'blocker_type': 'no_local_scanrefer_implementation',
        'blocker_message': 'No local ScanRefer-style localization implementation or data pipeline exists in this repository snapshot.',
        'case_list_path': str(case_list_path),
        'case_list_case_count': case_count,
        'image_dir': str(image_dir),
        'missing_components': ['scanrefer', '3d_language_grounding_pipeline', 'localized_box_or_point_labels'],
        'claim_boundary': 'AI2-THOR object grounding/localization provenance only; no detector superiority, recall, precision, broad benchmark, live AI2-THOR, navigation, manipulation, ObjectNav, recovery-search, memory-writeback, or policy-performance claim is supported.',
    }

def make_yolo_world_case_level_smoke(image_dir: Path, case_list_path: Path, run_model: bool = True) -> dict[str, Any]:
    cases = _load_case_list(case_list_path)
    entries = _load_image_manifest_entries(image_dir)
    blocker_type = None
    blocker_message = None
    try:
        import ultralytics  # type: ignore[import-not-found]
    except Exception as exc:
        blocker_type = 'missing_optional_dependencies'
        blocker_message = str(exc)
        ultralytics = None  # type: ignore[assignment]
    rows: list[dict[str, Any]] = []
    matched_rows = 0
    detections_total = 0
    for fallback_idx, case in enumerate(cases):
        entry, match_reason = _match_case_to_manifest(case, entries)
        row = {
            'source_row_idx': case.get('row_idx', fallback_idx),
            'scene': case.get('scene'),
            'seed': case.get('seed'),
            'target': case.get('target'),
            'target_object_id': case.get('target_object_id') or case.get('object_id'),
            'failure_type': case.get('failure_type'),
            'phase': entry.get('phase') if entry else None,
            'frame_path': entry.get('frame_path_abs') if entry else None,
            'manifest_object_id': entry.get('object_id') if entry else None,
            'match_reason': match_reason,
            'matched_image_status': 'matched' if entry else 'no_manifest_match',
            'detections_returned': 0,
            'scores': [],
            'boxes_xyxy': [],
            'labels': [],
        }
        if entry is not None:
            matched_rows += 1
        if run_model and blocker_type is None and entry is not None and ultralytics is not None:
            try:
                model = ultralytics.YOLO('yolov8n.pt')
                result = model.predict(source=Path(str(entry.get('frame_path_abs'))), verbose=False)
                row['detections_returned'] = len(result[0].boxes) if result else 0
                detections_total += row['detections_returned']
            except Exception as exc:
                blocker_type = 'network_or_model_unavailable'
                blocker_message = str(exc)
        rows.append(row)
    status = 'blocked' if blocker_type else 'ok'
    return {
        'schema': '0514_yolo_world_case_level_smoke.v1',
        'status': status,
        'blocker_type': blocker_type,
        'blocker_message': blocker_message,
        'case_list_path': str(case_list_path),
        'case_list_case_count': len(cases),
        'image_dir': str(image_dir),
        'model_name': 'yolo-world-via-ultralytics',
        'rows': rows,
        'matched_case_rows': matched_rows,
        'unmatched_case_rows': len(cases) - matched_rows,
        'detections_returned_total': detections_total,
        'claim_boundary': 'YOLO-World blocked dependency artifact over the existing saved Book/Newspaper case-list protocol only; no detector superiority, recall, precision, broad benchmark, live AI2-THOR, navigation, manipulation, ObjectNav, recovery-search, memory-writeback, or policy-performance claim is supported.',
    }

def make_owl_vit_case_level_smoke(image_dir: Path, case_list_path: Path, run_model: bool = True) -> dict[str, Any]:
    cases = _load_case_list(case_list_path)
    entries = _load_image_manifest_entries(image_dir)
    model_name = 'google/owlvit-base-patch32'
    processor = None
    model = None
    blocker_type = None
    blocker_message = None
    if run_model:
        try:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            processor = OwlViTProcessor.from_pretrained(model_name)
            model = OwlViTForObjectDetection.from_pretrained(model_name)
            import torch  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception as exc:
            blocker_type = 'network_or_model_unavailable'
            blocker_message = str(exc)
    rows: list[dict[str, Any]] = []
    matched_rows = 0
    detections_total = 0
    for fallback_idx, case in enumerate(cases):
        entry, match_reason = _match_case_to_manifest(case, entries)
        row = {
            'source_row_idx': case.get('row_idx', fallback_idx),
            'scene': case.get('scene'),
            'seed': case.get('seed'),
            'target': case.get('target'),
            'target_object_id': case.get('target_object_id') or case.get('object_id'),
            'failure_type': case.get('failure_type'),
            'visibility_slice': case.get('visibility_slice'),
            'nav_search_slice': case.get('nav_search_slice'),
            'phase': entry.get('phase') if entry else None,
            'frame_path': entry.get('frame_path_abs') if entry else None,
            'manifest_object_id': entry.get('object_id') if entry else None,
            'match_reason': match_reason,
            'matched_image_status': 'matched' if entry else 'no_manifest_match',
            'detections_returned': 0,
            'scores': [],
            'boxes_xyxy': [],
            'labels': [],
        }
        if entry is not None:
            matched_rows += 1
        if run_model and blocker_type is None and entry is not None and processor is not None and model is not None:
            try:
                import torch
                from PIL import Image
                image = Image.open(Path(str(entry.get('frame_path_abs')))).convert('RGB')
                label = str(case.get('target') or '')
                inputs = processor(text=[label], images=image, return_tensors='pt')
                outputs = model(**inputs)
                target_sizes = torch.tensor([[image.height, image.width]])
                processed = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
                det = processed[0] if processed else {}
                scores = [float(v) for v in det.get('scores', [])]
                boxes = [[float(x) for x in box] for box in det.get('boxes', [])]
                labels = [label for _ in scores]
                row['detections_returned'] = len(scores)
                row['scores'] = scores
                row['boxes_xyxy'] = boxes
                row['labels'] = labels
                detections_total += len(scores)
            except Exception as exc:
                row['runtime_error'] = str(exc)
        rows.append(row)
    status = 'blocked' if blocker_type else 'ok'
    return {
        'schema': '0514_owl_vit_case_level_smoke.v1',
        'status': status,
        'blocker_type': blocker_type,
        'blocker_message': blocker_message,
        'case_list_path': str(case_list_path),
        'case_list_case_count': len(cases),
        'image_dir': str(image_dir),
        'model_name': model_name,
        'threshold': 0.1,
        'rows': rows,
        'matched_case_rows': matched_rows,
        'unmatched_case_rows': len(cases) - matched_rows,
        'detections_returned_total': detections_total,
        'claim_boundary': 'OWL-ViT matched case-level smoke over saved dino_only_pass1 images only; no detector superiority, benchmark, navigation, manipulation, ObjectNav, recovery-search, memory-writeback, live AI2-THOR, or policy-performance claim is supported.',
    }

def make_owl_vit_detector_smoke(image_dir: Path, case_list_path: Path) -> dict[str, Any]:
    cases = _load_case_list(case_list_path)
    blocker_type = None
    try:
        import transformers  # type: ignore[import-not-found]
        from PIL import Image
    except Exception:
        blocker_type = 'missing_optional_dependencies'
        transformers = None  # type: ignore[assignment]
        Image = None  # type: ignore[assignment]
    if blocker_type is not None:
        return {
            'schema': '0514_owl_vit_detector_smoke.v1',
            'status': 'blocked',
            'blocker_type': blocker_type,
            'case_list_path': str(case_list_path),
            'case_list_case_count': len(cases),
            'image_dir': str(image_dir),
            'claim_boundary': 'OWL-ViT smoke only; no detector performance claims',
        }
    try:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except Exception:
        return {
            'schema': '0514_owl_vit_detector_smoke.v1',
            'status': 'blocked',
            'blocker_type': 'missing_optional_dependencies',
            'case_list_path': str(case_list_path),
            'case_list_case_count': len(cases),
            'image_dir': str(image_dir),
            'claim_boundary': 'OWL-ViT smoke only; no detector performance claims',
        }
    images = sorted(image_dir.rglob('*.jpg'))
    if not images:
        return {
            'schema': '0514_owl_vit_detector_smoke.v1',
            'status': 'blocked',
            'blocker_type': 'no_images_found',
            'case_list_path': str(case_list_path),
            'case_list_case_count': len(cases),
            'image_dir': str(image_dir),
            'claim_boundary': 'OWL-ViT smoke only; no detector performance claims',
        }
    model_name = 'google/owlvit-base-patch32'
    try:
        processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name)
    except Exception as exc:
        return {
            'schema': '0514_owl_vit_detector_smoke.v1',
            'status': 'blocked',
            'blocker_type': 'network_or_model_unavailable',
            'blocker_message': str(exc),
            'case_list_path': str(case_list_path),
            'case_list_case_count': len(cases),
            'image_dir': str(image_dir),
            'claim_boundary': 'OWL-ViT smoke only; no detector performance claims',
        }
    image_path = images[0]
    try:
        import torch
        image = Image.open(image_path).convert('RGB')
        texts = [case.get('target') for case in cases if isinstance(case, dict) and case.get('target')]
        texts = list(dict.fromkeys(texts))
        inputs = processor(text=texts, images=image, return_tensors='pt')
        outputs = model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]])
        detections = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
    except Exception as exc:
        return {
            'schema': '0514_owl_vit_detector_smoke.v1',
            'status': 'blocked',
            'blocker_type': 'runtime_error',
            'blocker_message': str(exc),
            'case_list_path': str(case_list_path),
            'case_list_case_count': len(cases),
            'image_dir': str(image_dir),
            'claim_boundary': 'OWL-ViT smoke only; no detector performance claims',
        }
    return {
        'schema': '0514_owl_vit_detector_smoke.v1',
        'status': 'ok',
        'blocker_type': None,
        'case_list_path': str(case_list_path),
        'case_list_case_count': len(cases),
        'image_dir': str(image_dir),
        'model_name': model_name,
        'smoke_image': str(image_path),
        'detections_returned': len(detections[0].get('scores', [])) if detections else 0,
        'claim_boundary': 'OWL-ViT smoke only; no detector performance claims',
    }

def make_detector_baseline_audit(image_dir: Path, case_list_path: Path) -> dict[str, Any]:
    cases = _load_case_list(case_list_path)
    covered_backends = ["GSAM/SAM2 live verifier", "GroundingDINO-box targeted reruns"]
    missing_backends = ["OWL-ViT", "YOLO-World", "ScanRefer-style localization"]
    blocked_reason = None
    try:
        import transformers  # type: ignore[import-not-found]
    except Exception:
        blocked_reason = "missing_optional_dependencies"
    try:
        import ultralytics  # type: ignore[import-not-found]
    except Exception:
        blocked_reason = blocked_reason or "missing_optional_dependencies"
    if blocked_reason is None:
        blocked_reason = "audit_only_no_new_smoke"
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append({
            "scene": case.get("scene"),
            "seed": case.get("seed"),
            "target": case.get("target"),
            "failure_type": case.get("failure_type"),
            "navigation_slice": case.get("nav_search_slice"),
            "covered_backends": covered_backends,
            "missing_backends": missing_backends,
            "image_dir": str(image_dir),
        })
    return {
        "schema": "0514_detector_baseline_audit_runtime.v1",
        "status": "blocked",
        "blocker_type": blocked_reason,
        "case_list_path": str(case_list_path),
        "case_list_case_count": len(cases),
        "covered_backends": covered_backends,
        "covered_backends_count": len(covered_backends),
        "missing_backends": missing_backends,
        "missing_backends_count": len(missing_backends),
        "rows": rows,
        "claim_boundary": "detector coverage audit only; no new detector performance claims",
    }


class GroundedSAM2LocationBackend:
    """Optional GroundingDINO + SAM2 image backend.

    This backend is intentionally dependency-light at import time. It uses
    GroundingDINO boxes as open-vocabulary detections and, when SAM2 is
    importable, refines the box center with the predicted mask centroid.
    """

    name = "grounded_sam2"
    coordinate_space = "normalized_frame_xy"
    threshold_unit = "frame_diagonal_fraction"

    def __init__(
        self,
        grounding_config: Path,
        grounding_checkpoint: Path,
        sam2_config: Path | None = None,
        sam2_checkpoint: Path | None = None,
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        backend_load_timeout_seconds: float | None = None,
    ) -> None:
        self.grounding_config = grounding_config
        self.grounding_checkpoint = grounding_checkpoint
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.backend_load_timeout_seconds = backend_load_timeout_seconds
        self._grounding_model = None
        self._sam2_predictor = None
        self.sam2_enabled = False
        self.sam2_status = "not_requested"

    @staticmethod
    def _prefix_env_path(env_var: str, prefix: str) -> None:
        current = os.environ.get(env_var, "")
        parts = [part for part in current.split(":") if part]
        if prefix in parts:
            parts = [part for part in parts if part != prefix]
        os.environ[env_var] = ":".join([prefix, *parts]) if parts else prefix

    @staticmethod
    def _normalize_sam2_config(config: Path | None) -> Path | None:
        if config is None:
            return None
        if not config.is_absolute():
            return config
        config_str = str(config)
        marker = "/sam2/"
        marker_idx = config_str.rfind(marker)
        if marker_idx >= 0:
            return Path(config_str[marker_idx + len(marker) :])
        return config

    def _prepare_backend_runtime(self) -> None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            self._prefix_env_path("LD_LIBRARY_PATH", f"{conda_prefix}/lib")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def _run_with_load_timeout(self, label: str, func: Callable[[], T]) -> T:
        timeout = self.backend_load_timeout_seconds
        if timeout is None or timeout <= 0:
            return func()

        previous_handler = signal.getsignal(signal.SIGALRM)

        def _handler(signum: int, frame: object) -> None:
            raise BackendLoadTimeoutError(f"{label} exceeded {timeout} seconds")

        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            return func()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _load_grounding(self) -> None:
        if self._grounding_model is not None:
            return
        self._prepare_backend_runtime()
        try:
            from groundingdino.util.inference import load_model
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "GroundingDINO is not installed. Install the official package "
                "from https://github.com/IDEA-Research/GroundingDINO."
            ) from exc
        self._grounding_model = self._run_with_load_timeout(
            "GroundingDINO load_model",
            lambda: load_model(
                str(self.grounding_config),
                str(self.grounding_checkpoint),
                device=self.device,
            ),
        )

    def _load_sam2(self) -> None:
        if self._sam2_predictor is not None:
            return
        if not self.sam2_config or not self.sam2_checkpoint:
            self.sam2_status = "not_configured"
            return
        self._prepare_backend_runtime()
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:  # pragma: no cover - optional dependency
            self.sam2_status = f"import_failed:{type(exc).__name__}"
            raise RuntimeError(
                "SAM2 config/checkpoint were provided, but SAM2 could not be imported. "
                "Install the official package from https://github.com/facebookresearch/sam2 "
                "or omit --sam2-config/--sam2-checkpoint to run GroundingDINO-box mode."
            ) from exc
        model = self._run_with_load_timeout(
            "SAM2 build_sam2",
            lambda: build_sam2(
                str(self._normalize_sam2_config(self.sam2_config)),
                str(self.sam2_checkpoint),
                device=self.device,
            ),
        )
        self._sam2_predictor = SAM2ImagePredictor(model)
        self.sam2_enabled = True
        self.sam2_status = "enabled"

    def _detect_image(self, image_path: Path, label: str) -> List[ObjectDetection]:
        self._load_grounding()
        try:
            import numpy as np
            from PIL import Image
            from groundingdino.util.inference import load_image, predict
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Grounded-SAM2 image dependencies are unavailable") from exc

        image_source, image_tensor = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=self._grounding_model,
            image=image_tensor,
            caption=f"{label}.",
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        width, height = Image.open(image_path).size
        detections: List[ObjectDetection] = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = [float(v) for v in box.tolist()]
            center = (cx * width, cy * height)
            score = float(logit)
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            det_xyxy = (x1, y1, x2, y2)
            if self._sam2_predictor is not None:
                try:
                    self._sam2_predictor.set_image(np.asarray(image_source))
                    masks, _, _ = self._sam2_predictor.predict(
                        box=np.asarray([x1, y1, x2, y2]),
                        multimask_output=False,
                    )
                    mask = masks[0]
                    ys, xs = np.where(mask)
                    if len(xs) and len(ys):
                        center = (float(xs.mean()), float(ys.mean()))
                except Exception:
                    self.sam2_status = "mask_refinement_failed"
                    pass
            detections.append(
                ObjectDetection(
                    label=str(phrase) or label,
                    score=score,
                    center=center,
                    coordinate_space="pixel_xy",
                    image_path=str(image_path),
                    xyxy=det_xyxy,
                )
            )
        return detections

    def _load_manifest(self, image_dir: Path) -> List[ImageManifestEntry]:
        # Always read fresh from disk — the manifest is appended to by
        # concurrent maintenance sweeps and caching can miss newly written
        # entries from earlier tasks.
        entries: List[ImageManifestEntry] = []
        for manifest_path in sorted(image_dir.rglob("image_manifest.jsonl")):
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                bbox = raw.get("bbox_xyxy") or []
                if len(bbox) != 4:
                    continue
                entries.append(
                    ImageManifestEntry(
                        scene=str(raw.get("scene") or ""),
                        phase=str(raw.get("phase") or ""),
                        action_idx=int(raw.get("action_idx", 0)),
                        object_id=str(raw.get("object_id") or ""),
                        object_type=str(raw.get("object_type") or ""),
                        bbox_xyxy=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                        frame_path=image_dir / str(raw.get("frame_path") or ""),
                        crop_path=image_dir / str(raw.get("crop_path") or ""),
                    )
                )
        return entries

    def _manifest_entry_for_location(
        self,
        image_dir: Path,
        remembered_location: str,
    ) -> Optional[ImageManifestEntry]:
        remembered = parse_rearrangement_location(remembered_location)
        for entry in self._load_manifest(image_dir):
            if (
                entry.scene == remembered.scene
                and entry.phase == remembered.phase
                and entry.object_id == remembered.object_id
            ):
                return entry
        return None

    def _after_entries_for_target(
        self,
        image_dir: Path,
        *,
        scene: str,
        object_type: str,
    ) -> List[ImageManifestEntry]:
        return [
            entry
            for entry in self._load_manifest(image_dir)
            if entry.scene == scene and entry.phase == "after" and entry.object_type == object_type
        ]

    @staticmethod
    def _bbox_center(entry: ImageManifestEntry) -> Tuple[float, float]:
        x1, y1, x2, y2 = entry.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def verify(
        self,
        task: RearrangementTask,
        *,
        remembered_location: str,
        image_dir: Path,
        threshold: float,
    ) -> LocationDecision:
        # Load GroundingDINO first so cv2 binds to the conda libstdc++ before
        # SAM2/torch extensions can expose older system C++ runtime paths.
        self._load_grounding()
        self._load_sam2()
        manifest_entry = self._manifest_entry_for_location(image_dir, remembered_location)
        if manifest_entry is None:
            return LocationDecision(stale=True, confidence=0.0, reason="no_before_manifest_entry")
        remembered_center = self._bbox_center(manifest_entry)

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PIL is required for Grounded-SAM2 frame-size checks") from exc

        after_entries = self._after_entries_for_target(
            image_dir,
            scene=task.scene,
            object_type=task.target,
        )
        # Fallback: when no manifest entries exist for this target (AI2-THOR
        # metadata didn't report it as visible from the revisit viewpoint),
        # directly scan all saved full-frame JPEGs with GroundingDINO.
        if not after_entries:
            after_frame_dir = image_dir / task.scene / "after"
            after_frame_paths: List[Path] = []
            if after_frame_dir.is_dir():
                after_frame_paths = sorted(after_frame_dir.glob("*_frame.jpg"))
            if not after_frame_paths:
                return LocationDecision(stale=True, confidence=0.0, reason="no_paired_after_frame")
            after_entries = [
                ImageManifestEntry(
                    scene=task.scene,
                    phase="after",
                    action_idx=i,
                    object_id="",
                    object_type=task.target,
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    frame_path=p,
                    crop_path=p,
                )
                for i, p in enumerate(after_frame_paths)
            ]

        best_distance = math.inf
        detections_considered = 0
        for after_entry in after_entries:
            after_frame = after_entry.frame_path
            if not after_frame.exists():
                continue
            with Image.open(after_frame) as after_img:
                width, height = after_img.size
            denominator = math.sqrt(float(width * width + height * height))
            after_detections = self._detect_image(after_frame, task.target)
            detections_considered += len(after_detections)
            for detection in after_detections:
                pixel_distance = math.sqrt(
                    (remembered_center[0] - detection.center[0]) ** 2
                    + (remembered_center[1] - detection.center[1]) ** 2
                )
                best_distance = min(best_distance, pixel_distance / denominator if denominator else math.inf)

        if math.isinf(best_distance):
            return LocationDecision(
                stale=True,
                confidence=0.0,
                reason="no_matching_detection",
                detections_considered=detections_considered,
            )
        stale = best_distance > threshold
        return LocationDecision(
            stale=stale,
            confidence=1.0,
            reason="nearest_after_detection_normalized_frame_distance",
            matched_distance=round(best_distance, 4),
            detections_considered=detections_considered,
        )

    def details(self) -> Dict[str, Any]:
        return {
            "sam2_enabled": self.sam2_enabled,
            "sam2_status": self.sam2_status,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
        }


def evaluate_location_verification(
    probe: Dict[str, Any],
    image_dir: Path,
    backend: LocationVerifierBackend,
    budgets: Sequence[int] = (4, 8, 16),
    location_thresholds: Sequence[float] = (0.1, 0.25, 0.5),
    verification_cost: float = 0.5,
) -> List[Dict[str, Any]]:
    items = build_rearrangement_memory_items(probe)
    tasks = build_rearrangement_tasks(probe, target_objects=TARGET_OBJECTS)
    if not items or not tasks:
        return []

    query_now = max((item.observed_at for item in items), default=0) + 1
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for threshold in location_thresholds:
            policy = VerifyRearrangementRiskPolicy()
            store = MemoryStore(policy=SaliencePolicy(), budget=budget)
            for item in items:
                store.observe(item, now=item.observed_at, current_target=item.object_name)

            oracle_metrics = PolicyMetrics(policy="oracle")
            location_metrics = PolicyMetrics(policy=backend.name)
            decision_rows: List[Dict[str, Any]] = []
            false_catches = 0
            missed_stale = 0
            unverified_stale_errors = 0
            task_by_old_location: Dict[str, RearrangementTask] = {task.old_location: task for task in tasks}

            for task in tasks:
                outcome = store.query(task.target, task.true_location)
                oracle_metrics.tasks += 1
                location_metrics.tasks += 1

                if outcome.found and outcome.item is not None and policy.should_verify(outcome.item, now=query_now):
                    oracle_metrics.verifications += 1
                    location_metrics.verifications += 1
                    oracle_stale = bool(outcome.stale)

                    if oracle_stale:
                        oracle_metrics.verification_catches += 1
                        oracle_metrics.successes += 1
                        oracle_metrics.total_cost += verification_cost + task.scene_search_cost
                    else:
                        oracle_metrics.successes += 1
                        oracle_metrics.query_hits += 1
                        oracle_metrics.total_cost += verification_cost + task.direct_action_cost

                    decision = backend.verify(
                        task,
                        remembered_location=outcome.item.location,
                        image_dir=image_dir,
                        threshold=threshold,
                    )
                    if decision.stale:
                        location_metrics.verification_catches += 1
                        location_metrics.successes += 1
                        location_metrics.total_cost += verification_cost + task.scene_search_cost
                        if not oracle_stale:
                            false_catches += 1
                    elif oracle_stale:
                        location_metrics.stale_errors += 1
                        location_metrics.total_cost += (
                            verification_cost
                            + _stale_failure_cost(task_by_old_location, task, outcome.item.location)
                        )
                        missed_stale += 1
                    else:
                        location_metrics.successes += 1
                        location_metrics.query_hits += 1
                        location_metrics.total_cost += verification_cost + task.direct_action_cost

                    decision_rows.append(
                        {
                            "scene": task.scene,
                            "target": task.target,
                            "remembered_location": outcome.item.location,
                            "oracle_stale": oracle_stale,
                            "location_stale": decision.stale,
                            "reason": decision.reason,
                            "matched_distance": decision.matched_distance,
                            "detections_considered": decision.detections_considered,
                        }
                    )
                elif outcome.found and not outcome.stale:
                    oracle_metrics.successes += 1
                    oracle_metrics.query_hits += 1
                    oracle_metrics.total_cost += task.direct_action_cost
                    location_metrics.successes += 1
                    location_metrics.query_hits += 1
                    location_metrics.total_cost += task.direct_action_cost
                elif outcome.found and outcome.stale:
                    oracle_metrics.stale_errors += 1
                    if outcome.item is not None:
                        oracle_metrics.total_cost += _stale_failure_cost(
                            task_by_old_location,
                            task,
                            outcome.item.location,
                        )
                    else:
                        oracle_metrics.total_cost += task.old_action_cost + task.scene_search_cost
                    location_metrics.stale_errors += 1
                    if outcome.item is not None:
                        location_metrics.total_cost += _stale_failure_cost(
                            task_by_old_location,
                            task,
                            outcome.item.location,
                        )
                    else:
                        location_metrics.total_cost += task.old_action_cost + task.scene_search_cost
                    unverified_stale_errors += 1
                else:
                    oracle_metrics.total_cost += task.scene_search_cost
                    location_metrics.total_cost += task.scene_search_cost

            comparable = sum(1 for row in decision_rows if row["oracle_stale"] == row["location_stale"])
            agreement = comparable / len(decision_rows) if decision_rows else 0.0
            stale_true = sum(1 for row in decision_rows if row["oracle_stale"])
            stale_pred = sum(1 for row in decision_rows if row["location_stale"])
            true_catches = sum(1 for row in decision_rows if row["oracle_stale"] and row["location_stale"])
            precision = true_catches / stale_pred if stale_pred else 0.0
            recall = true_catches / stale_true if stale_true else 0.0
            total_stale_failures = stale_true + unverified_stale_errors
            end_to_end_recall = true_catches / total_stale_failures if total_stale_failures else 0.0
            rows.append(
                {
                    "budget": budget,
                    "backend": backend.name,
                    "coordinate_space": backend.coordinate_space,
                    "threshold_unit": backend.threshold_unit,
                    "sam2_enabled": getattr(backend, "sam2_enabled", None),
                    "sam2_status": getattr(backend, "sam2_status", None),
                    "location_threshold": threshold,
                    "seeds": 1,
                    "oracle_completion_rate": round(oracle_metrics.success_rate, 4),
                    "location_completion_rate": round(location_metrics.success_rate, 4),
                    "oracle_stale_errors": oracle_metrics.stale_errors,
                    "location_stale_errors": location_metrics.stale_errors,
                    "oracle_avg_cost": round(oracle_metrics.avg_cost, 4),
                    "location_avg_cost": round(location_metrics.avg_cost, 4),
                    "oracle_verifications": oracle_metrics.verifications,
                    "location_verifications": location_metrics.verifications,
                    "oracle_catches": oracle_metrics.verification_catches,
                    "location_catches": location_metrics.verification_catches,
                    "verified_false_catches": false_catches,
                    "verified_missed_stale": missed_stale,
                    "verified_oracle_agreement": round(agreement, 4),
                    "verified_catch_precision": round(precision, 4),
                    "verified_catch_recall": round(recall, 4),
                    "verified_decision_count": len(decision_rows),
                    "verified_stale_count": stale_true,
                    "unverified_stale_errors": unverified_stale_errors,
                    "end_to_end_stale_recall": round(end_to_end_recall, 4),
                    "decisions": decision_rows,
                }
            )
    return rows


def summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    best = max(
        rows,
        key=lambda row: (
            float(row["location_completion_rate"]),
            float(row["verified_catch_precision"]),
            float(row["verified_catch_recall"]),
            -float(row["location_avg_cost"]),
        ),
    )
    return {
        "n_runs": len(rows),
        "best_backend": best["backend"],
        "best_threshold_unit": best["threshold_unit"],
        "best_threshold": best["location_threshold"],
        "best_location_completion": best["location_completion_rate"],
        "best_oracle_completion": best["oracle_completion_rate"],
        "completion_delta": round(float(best["location_completion_rate"]) - float(best["oracle_completion_rate"]), 4),
        "best_location_stale_errors": best["location_stale_errors"],
        "best_verified_catch_precision": best["verified_catch_precision"],
        "best_verified_catch_recall": best["verified_catch_recall"],
        "best_end_to_end_stale_recall": best["end_to_end_stale_recall"],
        "best_sam2_enabled": best.get("sam2_enabled"),
        "best_sam2_status": best.get("sam2_status"),
    }


def write_outputs(rows: Sequence[Dict[str, Any]], probe: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(rows)
    serializable_rows = [dict(row, decisions=row.get("decisions", [])) for row in rows]
    (out_dir / "grounded_sam2_location_vs_oracle.json").write_text(
        json.dumps(
            {
                "probe_summary": summarize_rearrangement_probe(probe),
                "results": serializable_rows,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "grounded_sam2_location_vs_oracle_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in rows[0].keys() if key != "decisions"] if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "decisions"})
    (out_dir / "README.md").write_text(render_readme(rows, probe), encoding="utf-8")


def render_readme(rows: Sequence[Dict[str, Any]], probe: Dict[str, Any]) -> str:
    probe_summary = summarize_rearrangement_probe(probe)
    summary = summarize_results(rows)
    lines = [
        "# Grounded-SAM2-Style Location Verification",
        "",
        "Mode: `ai2thor_grounded_sam2_location_verification`",
        f"OK scenes: `{probe_summary['ok_scenes']}`",
        f"Moved paired objects: `{probe_summary['moved_objects']}`",
        "",
    ]
    if summary:
        lines.extend(
            [
                f"Best backend: `{summary['best_backend']}`",
                f"Threshold unit: `{summary['best_threshold_unit']}`",
                f"Best threshold: `{summary['best_threshold']}`",
                f"Location completion: `{summary['best_location_completion']}`",
                f"Oracle completion: `{summary['best_oracle_completion']}`",
                f"Verified catch precision: `{summary['best_verified_catch_precision']}`",
                f"Verified catch recall: `{summary['best_verified_catch_recall']}`",
                f"End-to-end stale recall: `{summary['best_end_to_end_stale_recall']}`",
                f"SAM2 enabled: `{summary['best_sam2_enabled']}`",
                f"SAM2 status: `{summary['best_sam2_status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "`crop_filename_proxy` is a metadata-proxy backend for validating the evaluator only.",
            "Publishable perception claims require the `grounded_sam2` backend with official GroundingDINO and SAM2 weights.",
            "If SAM2 cannot be imported, the backend raises instead of silently downgrading the claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _make_backend(args: argparse.Namespace) -> LocationVerifierBackend:
    if args.backend == "crop-filename-proxy":
        return CropFilenameLocationBackend()
    return GroundedSAM2LocationBackend(
        grounding_config=args.grounding_config,
        grounding_checkpoint=args.grounding_checkpoint,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        backend_load_timeout_seconds=args.backend_load_timeout_seconds,
    )


def write_blocked_backend_output(
    probe: Dict[str, Any],
    out_dir: Path,
    exc: Exception,
    *,
    backend: str,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked",
        "backend": backend,
        "device": device,
        "blocker_type": type(exc).__name__,
        "error_message": str(exc),
        "probe_summary": summarize_rearrangement_probe(probe),
        "claim_boundary": "Grounded-SAM2 backend did not load; no real perception claim is supported.",
    }
    (out_dir / "grounded_sam2_location_vs_oracle_blocked.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Grounded-SAM2 Location Verification Blocked",
                "",
                f"Status: `{payload['status']}`",
                f"Backend: `{backend}`",
                f"Device: `{device}`",
                f"Blocker: `{payload['blocker_type']}`",
                f"Message: `{payload['error_message']}`",
                "",
                str(payload["claim_boundary"]),
                "Use `crop_filename_proxy` outputs only as metadata-proxy verifier evidence, not GSAM/DINO perception.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded-SAM2-style location verification vs oracle.")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--location-thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--verification-cost", type=float, default=0.5)
    parser.add_argument("--backend", choices=["crop-filename-proxy", "grounded-sam2"], default="crop-filename-proxy")
    parser.add_argument("--grounding-config", type=Path, default=Path("checkpoints/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--grounding-checkpoint", type=Path, default=Path("checkpoints/groundingdino_swint_ogc.pth"))
    parser.add_argument("--sam2-config", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--backend-load-timeout-seconds", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_grounded_sam2_location_verification"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = load_probe(args.probe)
    backend = _make_backend(args)
    try:
        rows = evaluate_location_verification(
            probe,
            args.image_dir,
            backend,
            budgets=args.budgets,
            location_thresholds=args.location_thresholds,
            verification_cost=args.verification_cost,
        )
    except BackendLoadTimeoutError as exc:
        write_blocked_backend_output(
            probe,
            args.out_dir,
            exc,
            backend=args.backend,
            device=args.device,
        )
        print(json.dumps({"status": "blocked", "blocker_type": type(exc).__name__, "error_message": str(exc)}, indent=2))
        return
    if not rows:
        print("No results; check probe, image directory, and backend dependencies.")
        return
    write_outputs(rows, probe, args.out_dir)
    print(json.dumps({"summary": summarize_results(rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
