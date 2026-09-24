"""Thin live bridge for Grounded-SAM2 location verification.

This module keeps the core replay verifier untouched. It only provides a small
adapter layer that turns live-style inputs into the same ``LocationDecision``
schema and writes replay-compatible audit rows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import LocationDecision, LocationVerifierBackend
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import RearrangementTask


@dataclass(frozen=True)
class LiveLocationQuery:
    """Live-style inputs for a single remembered location check."""

    scene_id: str
    object_type: str
    remembered_location: str
    frame_path: Path
    image_dir: Path
    threshold: float
    frame_id: str | None = None
    timestamp_ms: int | None = None
    camera_metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class LiveBridgeRecord:
    task: RearrangementTask
    query: LiveLocationQuery
    decision: LocationDecision
    policy_name: str = "live"
    policy_outcome: str = "unknown"
    backend_name: str = ""
    coordinate_space: str = ""
    threshold_unit: str = ""
    trace_id: str = ""
    source_mode: str = "live-style"
    policy_action: str = "verify"
    policy_reason: str = "replay-compatible verification"
    claim_boundary: str = "replay/oracle live-style adapter only"


class LiveLocationVerifier:
    """Wrap a replay verifier behind a live query object."""

    def __init__(self, backend: LocationVerifierBackend) -> None:
        self.backend: LocationVerifierBackend = backend

    def verify(self, task: RearrangementTask, query: LiveLocationQuery) -> LocationDecision:
        return self.backend.verify(
            task,
            remembered_location=query.remembered_location,
            image_dir=query.image_dir,
            threshold=query.threshold,
        )


class ReplayBridge:
    """Convert live verification calls back into replay-friendly audit rows."""

    def __init__(self, verifier: LiveLocationVerifier) -> None:
        self.verifier: LiveLocationVerifier = verifier

    def verify(self, task: RearrangementTask, query: LiveLocationQuery, policy_name: str = "live") -> LiveBridgeRecord:
        decision = self.verifier.verify(task, query)
        trace_id = f"{task.scene}:{task.target}:{query.threshold:.3f}:{query.frame_path.name}"
        policy_action = "accept_memory" if not decision.stale else "fallback_to_scene_search"
        return LiveBridgeRecord(
            task=task,
            query=query,
            decision=decision,
            policy_name=policy_name,
            backend_name=self.verifier.backend.name,
            coordinate_space=self.verifier.backend.coordinate_space,
            threshold_unit=self.verifier.backend.threshold_unit,
            trace_id=trace_id,
            source_mode="live-style" if query.camera_metadata is not None else "replay-derived",
            policy_action=policy_action,
            policy_reason=f"{decision.reason}:{'stale' if decision.stale else 'fresh'}",
            claim_boundary="replay/oracle live-style adapter only",
        )

    @staticmethod
    def row(record: LiveBridgeRecord) -> dict[str, object]:
        return {
            "scene": record.task.scene,
            "target": record.task.target,
            "true_location": record.task.true_location,
            "old_location": record.task.old_location,
            "frame_path": str(record.query.frame_path),
            "image_dir": str(record.query.image_dir),
            "threshold": record.query.threshold,
            "policy_name": record.policy_name,
            "policy_outcome": record.policy_outcome,
            "backend_name": record.backend_name,
            "coordinate_space": record.coordinate_space,
            "threshold_unit": record.threshold_unit,
            "frame_id": record.query.frame_id,
            "timestamp_ms": record.query.timestamp_ms,
            "trace_id": record.trace_id,
            "source_mode": record.source_mode,
            "policy_action": record.policy_action,
            "policy_reason": record.policy_reason,
            "claim_boundary": record.claim_boundary,
            "decision_stale": record.decision.stale,
            "decision_confidence": record.decision.confidence,
            "decision_reason": record.decision.reason,
            "decision_matched_distance": record.decision.matched_distance,
            "decision_detections_considered": record.decision.detections_considered,
        }

    @classmethod
    def summary(cls, records: Sequence[LiveBridgeRecord]) -> dict[str, object]:
        rows = [cls.row(record) for record in records]
        stale_count = sum(1 for row in rows if bool(row["decision_stale"]))
        return {
            "records": len(rows),
            "stale_count": stale_count,
            "fresh_count": len(rows) - stale_count,
            "stale_rate": round(stale_count / len(rows), 4) if rows else 0.0,
        }

    @classmethod
    def render_readme(cls, records: Sequence[LiveBridgeRecord]) -> str:
        summary = cls.summary(records)
        lines = [
            "# Grounded-SAM2 Live Bridge",
            "",
            f"Records: `{summary['records']}`",
            f"Stale count: `{summary['stale_count']}`",
            f"Fresh count: `{summary['fresh_count']}`",
            f"Stale rate: `{summary['stale_rate']}`",
            "",
            "This bridge only adapts live-style inputs to the same replay verifier schema.",
            "It does not alter the GroundingDINO/SAM2 backend or the replay benchmark.",
            "It now emits an explicit audit envelope with backend provenance, frame/timestamp hooks, trace_id, source_mode, policy_action, policy_reason, and claim_boundary.",
            "The boundary remains replay/oracle live-style adapter only, not live closed-loop evidence.",
            "",
        ]
        return "\n".join(lines)

    @classmethod
    def write_outputs(cls, records: Sequence[LiveBridgeRecord], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [cls.row(record) for record in records]
        _ = (out_dir / "live_bridge.json").write_text(
            json.dumps({"summary": cls.summary(records), "rows": rows}, indent=2),
            encoding="utf-8",
        )
        with (out_dir / "live_bridge.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        _ = (out_dir / "README.md").write_text(cls.render_readme(records), encoding="utf-8")
