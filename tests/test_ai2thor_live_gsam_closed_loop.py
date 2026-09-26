from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np

from embodied_memory_pilot.ai2thor_adapter import RICH_BEFORE_PROBE_ACTIONS, CapabilityReport
from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import LocationDecision
from embodied_memory_pilot.ai2thor_live_gsam_closed_loop import (
    LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY,
    LiveGSAMClosedLoopConfig,
    StaleUncertaintyInputs,
    _alternate_reachable_poses,
    _apply_calibrated_ev_penalty,
    _attempt_pickup_honest,
    _load_targeted_case_specs,
    _build_uncertainty_inputs,
    _estimate_stale_probability,
    _estimate_stale_uncertainty,
    _execute_passive_task_bridge_for_row,
    _execute_task_bridge_for_row,
    _object_mobility_prior,
    _select_memory_targets,
    _visual_instance_confusion_prior,
    _rank_verification_candidates,
    parse_args,
    run_live_gsam_closed_loop,
    write_outputs,
)
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import RearrangementTask


class _FakeEvent:
    def __init__(self, metadata: dict[str, object], events: list[dict[str, object]], frame: object | None = None, instance_masks: dict[str, object] | None = None):
        self.metadata = metadata
        self.events = events
        self.frame = frame if frame is not None else np.zeros((6, 6, 3), dtype=np.uint8)
        self.instance_masks = instance_masks or {}


class FakeClosedLoopController:
    def __init__(self) -> None:
        self.actions: list[tuple[str, dict[str, object]]] = []
        self.stopped = False
        self.spawned = False

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "GetReachablePositions":
            return _FakeEvent(
                metadata={
                    "agent": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "cameraHorizon": 0.0},
                    "actionReturn": [
                        {"x": 0.45, "y": 0.9, "z": 0.45},
                        {"x": 2.0, "y": 0.9, "z": 2.0},
                    ],
                },
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent(metadata=self._metadata([]) | {"lastActionSuccess": True}, events=[])
        if action == "TeleportFull":
            return _FakeEvent(metadata=self._metadata([]) | {"lastActionSuccess": True}, events=[])
        visible = [self._apple_after() if self.spawned else self._apple_before()]
        return _FakeEvent(metadata=self._metadata(visible), events=visible, instance_masks=self._masks(visible))

    def stop(self) -> None:
        self.stopped = True

    def _metadata(self, objects: list[dict[str, object]]) -> dict[str, object]:
        return {
            "agent": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "cameraHorizon": 0.0},
            "objects": objects,
        }

    def _apple_before(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "visible": True,
            "pickupable": True,
            "position": {"x": 0.50, "y": 1.10, "z": 0.50},
        }

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "visible": True,
            "pickupable": True,
            "position": {"x": 1.40, "y": 1.10, "z": 1.40},
        }

    def _masks(self, objects: list[dict[str, object]]) -> dict[str, object]:
        mask = np.array([
            [False, False, False, False, False, False],
            [False, True, True, False, False, False],
            [False, True, True, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ])
        return {str(obj["objectId"]): mask for obj in objects}


class FakeVerifier:
    name = "fake_gsam"

    def __init__(self) -> None:
        self.remembered_locations: list[str] = []
        self.image_dirs: list[Path] = []

    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float) -> LocationDecision:
        self.remembered_locations.append(remembered_location)
        self.image_dirs.append(image_dir)
        assert task.target == "Apple"
        assert remembered_location.endswith("@before")
        assert (image_dir / "FloorPlan1" / "before" / "image_manifest.jsonl").exists()
        assert (image_dir / "FloorPlan1" / "after" / "image_manifest.jsonl").exists()
        return LocationDecision(stale=False, confidence=0.9, reason="fake_detector_decision", matched_distance=0.01, detections_considered=1)


class FakeBudgetController(FakeClosedLoopController):
    def _metadata(self, objects: list[dict[str, object]]) -> dict[str, object]:
        return {
            "agent": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "cameraHorizon": 0.0},
            "objects": objects,
        }

    def _apple_before(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "visible": True,
            "pickupable": True,
            "position": {"x": 0.50, "y": 1.10, "z": 0.50},
        }

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "visible": True,
            "pickupable": True,
            "position": {"x": 1.40, "y": 1.10, "z": 1.40},
        }

    def _book_before(self) -> dict[str, object]:
        return {
            "objectType": "Book",
            "objectId": "Book|+00.20|+01.10|+00.20",
            "visible": True,
            "pickupable": False,
            "moveable": False,
            "position": {"x": 0.20, "y": 1.10, "z": 0.20},
        }

    def _book_after(self) -> dict[str, object]:
        return {
            "objectType": "Book",
            "objectId": "Book|+00.20|+01.10|+00.20",
            "visible": True,
            "pickupable": False,
            "moveable": False,
            "position": {"x": 1.20, "y": 1.10, "z": 1.20},
        }

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "GetReachablePositions":
            return _FakeEvent(
                metadata={
                    "agent": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "cameraHorizon": 0.0},
                    "actionReturn": [
                        {"x": 0.45, "y": 0.9, "z": 0.45},
                        {"x": 0.20, "y": 0.9, "z": 0.20},
                    ],
                },
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent(metadata=self._metadata([self._apple_after(), self._book_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "TeleportFull":
            return _FakeEvent(metadata=self._metadata([]) | {"lastActionSuccess": True}, events=[])
        visible = [self._apple_after(), self._book_after()] if self.spawned else [self._apple_before(), self._book_before()]
        return _FakeEvent(metadata=self._metadata(visible), events=visible, instance_masks=self._masks(visible))


class FakeFlexibleVerifier(FakeVerifier):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float) -> LocationDecision:
        self.targets.append(task.target)
        self.remembered_locations.append(remembered_location)
        self.image_dirs.append(image_dir)
        return LocationDecision(stale=False, confidence=0.9, reason="fake_detector_decision", matched_distance=0.01, detections_considered=1)


class FakeStaleVerifier(FakeVerifier):
    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float) -> LocationDecision:
        self.remembered_locations.append(remembered_location)
        self.image_dirs.append(image_dir)
        return LocationDecision(stale=True, confidence=0.77, reason="fake_stale_detector_decision", matched_distance=0.42, detections_considered=3)


class FakeTaskExecutionController(FakeClosedLoopController):
    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "PickupObject":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        return super().step(action, **kwargs)


class FakeStepwiseRevisitController(FakeClosedLoopController):
    def __init__(self) -> None:
        super().__init__()
        self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.agent_heading = 0.0

    def _metadata(self, objects: list[dict[str, object]]) -> dict[str, object]:
        return {
            "agent": {
                "position": dict(self.agent_position),
                "rotation": {"x": 0.0, "y": self.agent_heading, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "objects": objects,
        }

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "GetReachablePositions":
            return _FakeEvent(
                metadata=self._metadata([]) | {"actionReturn": [{"x": 0.0, "y": 0.9, "z": 0.25}]},
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
            self.agent_heading = 0.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "MoveAhead":
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.25}
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "RotateRight":
            self.agent_heading = (self.agent_heading + 90.0) % 360.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "RotateLeft":
            self.agent_heading = (self.agent_heading - 90.0) % 360.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        visible = [self._apple_after() if self.spawned else self._apple_before()]
        return _FakeEvent(metadata=self._metadata(visible), events=visible, instance_masks=self._masks(visible))


class FakeStepwiseTaskBridgeController(FakeStepwiseRevisitController):
    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.00|+01.10|+00.25",
            "visible": True,
            "pickupable": True,
            "position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "PickupObject":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "InitialRandomSpawn":
            self.spawned = True
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
            self.agent_heading = 0.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        return super().step(action, **kwargs)


class FakeWaypointTaskBridgeController(FakeStepwiseTaskBridgeController):
    def __init__(self) -> None:
        super().__init__()
        self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.agent_heading = 0.0

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "visible": True,
            "pickupable": True,
            "position": {"x": 0.50, "y": 1.10, "z": 0.50},
        }

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "GetReachablePositions":
            return _FakeEvent(
                metadata=self._metadata([])
                | {
                    "actionReturn": [
                        {"x": 0.0, "y": 0.9, "z": 0.0},
                        {"x": 0.25, "y": 0.9, "z": 0.0},
                        {"x": 0.50, "y": 0.9, "z": 0.0},
                        {"x": 0.50, "y": 0.9, "z": 0.25},
                        {"x": 0.50, "y": 0.9, "z": 0.50},
                    ]
                },
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
            self.agent_heading = 0.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "RotateRight":
            self.agent_heading = (self.agent_heading + 90.0) % 360.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "RotateLeft":
            self.agent_heading = (self.agent_heading - 90.0) % 360.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "MoveAhead":
            if self.agent_heading == 0.0:
                next_pos = {"x": self.agent_position["x"], "y": 0.9, "z": self.agent_position["z"] + 0.25}
            elif self.agent_heading == 90.0:
                next_pos = {"x": self.agent_position["x"] + 0.25, "y": 0.9, "z": self.agent_position["z"]}
            elif self.agent_heading == 180.0:
                next_pos = {"x": self.agent_position["x"], "y": 0.9, "z": self.agent_position["z"] - 0.25}
            else:
                next_pos = {"x": self.agent_position["x"] - 0.25, "y": 0.9, "z": self.agent_position["z"]}
            blocked_direct_diagonal = self.agent_position == {"x": 0.0, "y": 0.9, "z": 0.0} and self.agent_heading == 0.0
            if blocked_direct_diagonal:
                return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": False}, events=[])
            self.agent_position = next_pos
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "PickupObject":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        return super().step(action, **kwargs)


class FakeLongWaypointTaskBridgeController(FakeStepwiseTaskBridgeController):
    def __init__(self) -> None:
        super().__init__()
        self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.agent_heading = 0.0

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+03.50|+01.10|+00.00",
            "visible": True,
            "pickupable": True,
            "position": {"x": 3.50, "y": 1.10, "z": 0.00},
        }

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "GetReachablePositions":
            return _FakeEvent(
                metadata=self._metadata([])
                | {
                    "actionReturn": [
                        {"x": x / 4, "y": 0.9, "z": 0.0}
                        for x in range(15)
                    ]
                },
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
            self.agent_heading = 0.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "RotateRight":
            self.agent_heading = (self.agent_heading + 90.0) % 360.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "RotateLeft":
            self.agent_heading = (self.agent_heading - 90.0) % 360.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "MoveAhead":
            if self.agent_heading == 0.0:
                next_pos = {"x": self.agent_position["x"], "y": 0.9, "z": self.agent_position["z"] + 0.25}
            elif self.agent_heading == 90.0:
                next_pos = {"x": self.agent_position["x"] + 0.25, "y": 0.9, "z": self.agent_position["z"]}
            elif self.agent_heading == 180.0:
                next_pos = {"x": self.agent_position["x"], "y": 0.9, "z": self.agent_position["z"] - 0.25}
            else:
                next_pos = {"x": self.agent_position["x"] - 0.25, "y": 0.9, "z": self.agent_position["z"]}
            self.agent_position = next_pos
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "PickupObject":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        return super().step(action, **kwargs)


class FakePassiveClosedLoopController(FakeStepwiseTaskBridgeController):
    pass


class FakeTaskExecutionRenamedObjectController(FakeClosedLoopController):
    def _apple_after(self) -> dict[str, object]:
        obj = dict(super()._apple_after())
        obj["objectId"] = "Apple|+01.40|+01.10|+01.40"
        return obj

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "TeleportFull":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "PickupObject":
            success = kwargs.get("objectId") == "Apple|+01.40|+01.10|+01.40"
            return _FakeEvent(
                metadata=self._metadata([self._apple_after()]) | {
                    "lastActionSuccess": bool(success),
                    "errorMessage": "wrong_object_id" if not success else "",
                },
                events=[],
            )
        return super().step(action, **kwargs)


class FakeTaskExecutionOpenObjectController(FakeClosedLoopController):
    def _apple_before(self) -> dict[str, object]:
        return {
            "objectType": "Cabinet",
            "objectId": "Cabinet|+00.50|+00.00|+00.50",
            "visible": True,
            "pickupable": False,
            "openable": True,
            "position": {"x": 0.50, "y": 0.00, "z": 0.50},
        }

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Cabinet",
            "objectId": "Cabinet|+01.40|+00.00|+01.40",
            "visible": True,
            "pickupable": False,
            "openable": True,
            "position": {"x": 1.40, "y": 0.00, "z": 1.40},
        }

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "TeleportFull":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "OpenObject":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        return super().step(action, **kwargs)


class FakeTaskExecutionFailureController(FakeTaskExecutionController):
    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "TeleportFull":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "PickupObject":
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": False, "errorMessage": "forced_pickup_failure"}, events=[])
        return super().step(action, **kwargs)


class TestLiveGSAMClosedLoop(unittest.TestCase):
    def _assert_verification_only_execution_boundary(self, row: dict[str, object]) -> None:
        self.assertEqual(row["task_execution_evaluated"], False)
        self.assertIsNone(row["downstream_task_success"])
        self.assertEqual(row["execute_action"], "not_executed")
        self.assertEqual(row["execution_boundary"], "verification_only_no_downstream_action")

    def test_load_targeted_case_specs_from_failure_cases_artifact(self) -> None:
        payload = {
            "cases": [
                {"row_idx": 19, "scene": "FloorPlan1", "seed": 31, "target": "Book", "oracle_stale_label": True},
                {"row_idx": 4, "scene": "FloorPlan201", "seed": 7, "target": "Newspaper", "oracle_stale_label": True},
                {"scene": "FloorPlan201", "seed": "bad", "target": "Newspaper"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            specs = _load_targeted_case_specs(path)

        self.assertEqual([(spec.scene, spec.seed, spec.target, spec.source_row_idx) for spec in specs], [("FloorPlan1", 31, "Book", 19), ("FloorPlan201", 7, "Newspaper", 4)])

    def test_rank_verification_candidates_ignores_oracle_labels_and_uses_ev_tiebreaks(self) -> None:
        candidates = [
            {
                "row_idx": 0,
                "target_idx": 1,
                "target_object_id": "z-object",
                "expected_verification_value": 0.8,
                "oracle_stale_label": True,
                "oracle_distance": 99.0,
                "true_location_offline": "oracle-best",
            },
            {
                "row_idx": 1,
                "target_idx": 0,
                "target_object_id": "b-object",
                "expected_verification_value": 0.8,
                "oracle_stale_label": False,
                "oracle_distance": 0.0,
                "true_location_offline": "oracle-worst",
            },
            {
                "row_idx": 2,
                "target_idx": 0,
                "target_object_id": "a-object",
                "expected_verification_value": 0.8,
                "oracle_stale_label": True,
                "oracle_distance": 0.0,
                "true_location_offline": "oracle-middle",
            },
            {
                "row_idx": 3,
                "target_idx": 0,
                "target_object_id": "a-object",
                "expected_verification_value": 0.9,
                "oracle_stale_label": False,
                "oracle_distance": 100.0,
                "true_location_offline": "oracle-high",
            },
        ]

        ranked = _rank_verification_candidates(candidates, budget=4)

        self.assertEqual([row["row_idx"] for row in ranked], [3, 2, 1, 0])

    def test_uncertainty_estimator_uses_deterministic_live_revisit_inputs(self) -> None:
        before_obj = {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "pickupable": True,
            "position": {"x": 0.50, "y": 1.10, "z": 0.50},
        }
        remembered_pos = {"x": 0.50, "y": 1.10, "z": 0.50}
        found_pos = {"x": 1.40, "y": 1.10, "z": 1.40}

        self.assertEqual(_object_mobility_prior(before_obj), 1.0)
        inputs = _build_uncertainty_inputs(
            before_obj,
            remembered_pos,
            target_visible_after_revisit=True,
            visibility_error=None,
            found_pos=found_pos,
            teleport_success=True,
        )
        self.assertIsInstance(inputs, StaleUncertaintyInputs)
        self.assertEqual(inputs.detector_confidence, 1.0)
        self.assertEqual(inputs.depth_consistency, 0.0)
        self.assertAlmostEqual(inputs.object_mobility_prior, 0.85)
        self.assertEqual(inputs.time_interval, 1.0)
        self.assertEqual(inputs.task_importance, 1.0)
        self.assertEqual(inputs.historical_failure, 0.0)
        self.assertEqual(inputs.scene_change_score, 1.0)
        self.assertEqual(inputs.target_visible_after_revisit, True)
        self.assertIsNone(inputs.visibility_error)
        self.assertAlmostEqual(cast(float, inputs.distance_from_remembered), 1.2727922061)
        self.assertEqual(inputs.teleport_success, True)

        estimate = _estimate_stale_uncertainty(inputs)
        self.assertAlmostEqual(estimate.probability, 0.82)
        self.assertEqual(estimate.reason, "visible_target_displaced_from_remembered_location")
        self.assertEqual(estimate.inputs, inputs)
        wrapped_probability, wrapped_reason = _estimate_stale_probability(
            before_obj,
            remembered_pos,
            target_visible_after_revisit=True,
            visibility_error=None,
            found_pos=found_pos,
            teleport_success=True,
        )
        self.assertAlmostEqual(wrapped_probability, 0.82)
        self.assertEqual(wrapped_reason, "visible_target_displaced_from_remembered_location")

    def test_runner_uses_live_spawn_and_no_teleport_object(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeClosedLoopController()
        verifier = FakeVerifier()

        def controller_factory(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        action_names = [name for name, _ in controller.actions]
        self.assertIn("InitialRandomSpawn", action_names)
        self.assertIn("TeleportFull", action_names)
        self.assertNotIn("TeleportObject", action_names)
        self.assertLess(action_names.index("InitialRandomSpawn"), action_names.index("TeleportFull"))
        teleport_kwargs = [dict(kwargs) for name, kwargs in controller.actions if name == "TeleportFull"]
        self.assertTrue(any(kwargs.get("standing") is True for kwargs in teleport_kwargs))
        self.assertTrue(controller.stopped)

        rows = cast(list[dict[str, object]], result["rows"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["remembered_location"], "FloorPlan1:Apple|+00.50|+01.10|+00.50@before")
        self.assertEqual(row["oracle_stale_label"], True)
        self.assertEqual(row["policy_should_verify"], True)
        self.assertEqual(row["active_policy_name"], "expected_verification_value_v1")
        self.assertEqual(row["uncertainty_model_name"], "deterministic_live_revisit_uncertainty_v1")
        self.assertEqual(row["uncertainty_reason"], "visible_target_displaced_from_remembered_location")
        uncertainty_inputs = cast(dict[str, object], row["uncertainty_inputs"])
        self.assertEqual(
            set(uncertainty_inputs),
            {
                "detector_confidence",
                "depth_consistency",
                "object_mobility_prior",
                "time_interval",
                "task_importance",
                "historical_failure",
                "scene_change_score",
                "target_visible_after_revisit",
                "visibility_error",
                "distance_from_remembered",
                "teleport_success",
                "same_type_density_score",
                "same_type_nearby_count",
                "same_type_scene_total",
                "visual_instance_confusion_prior",
            },
        )
        self.assertEqual(uncertainty_inputs["detector_confidence"], 1.0)
        self.assertEqual(uncertainty_inputs["object_mobility_prior"], 0.85)
        self.assertEqual(uncertainty_inputs["task_importance"], 1.0)
        self.assertEqual(uncertainty_inputs["time_interval"], 1.0)
        self.assertEqual(uncertainty_inputs["target_visible_after_revisit"], True)
        self.assertEqual(uncertainty_inputs["teleport_success"], True)
        self.assertGreater(cast(float, uncertainty_inputs["distance_from_remembered"]), 0.05)
        self.assertAlmostEqual(cast(float, uncertainty_inputs["depth_consistency"]), 0.0)
        self.assertEqual(row["p_stale_estimate"], 0.82)
        self.assertGreater(cast(float, row["expected_verification_value"]), 0.0)
        self.assertEqual(row["decision_stale"], False)
        self.assertEqual(row["decision_matches_oracle"], False)
        self.assertEqual(row["memory_update_instrumented"], True)
        self.assertEqual(row["memory_update_action"], "none")
        self.assertEqual(row["memory_update_source"], "verified_fresh_or_no_stale_detector_evidence")
        self.assertEqual(row["memory_update_old_location"], row["remembered_location"])
        self.assertIsNone(row["memory_update_candidate_location"])
        self.assertEqual(row["memory_update_confidence"], 0.9)
        self._assert_verification_only_execution_boundary(row)
        self.assertIn("controller-backed evidence-guided memory refresh", str(row["memory_update_boundary"]))
        self.assertIn("not persistent storage", str(row["memory_update_boundary"]))
        self.assertIn("not recovery search", str(row["memory_update_boundary"]))
        self.assertIn("not task success", str(row["memory_update_boundary"]))
        self.assertEqual(verifier.remembered_locations, [row["remembered_location"]])
        self.assertIn("oracle metadata is offline labels only", str(row["row_claim_boundary"]))
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["policy_verified_rows"], 1)
        self.assertEqual(summary["policy_skipped_rows"], 0)
        self.assertEqual(summary["candidate_rows"], 1)
        self.assertEqual(summary["budget_selected_rows"], 1)
        self.assertEqual(summary["budget_skipped_rows"], 0)
        self.assertEqual(summary["memory_update_candidate_rows"], 0)
        self.assertEqual(summary["memory_updated_rows"], 0)
        self.assertEqual(summary["task_execution_evaluated_rows"], 0)
        self.assertEqual(summary["task_execution_not_executed_rows"], 1)
        self.assertEqual(summary["downstream_task_success_rows"], 0)

    def test_high_verification_cost_skips_verifier_but_keeps_offline_oracle_label(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeClosedLoopController()
        verifier = FakeVerifier()

        def controller_factory(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verification_cost=10.0,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        rows = cast(list[dict[str, object]], result["rows"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["oracle_stale_label"], True)
        self.assertEqual(row["policy_should_verify"], False)
        self.assertEqual(row["verifier_used"], False)
        self.assertIsNone(row["verifier_name"])
        self.assertIsNone(row["decision_stale"])
        self.assertIsNone(row["decision_matches_oracle"])
        self.assertEqual(row["memory_update_instrumented"], False)
        self.assertEqual(row["memory_update_action"], "not_applicable")
        self.assertIsNone(row["memory_update_source"])
        self.assertIsNone(row["memory_update_old_location"])
        self.assertIsNone(row["memory_update_candidate_location"])
        self.assertIsNone(row["memory_update_confidence"])
        self._assert_verification_only_execution_boundary(row)
        self.assertIn("controller-backed evidence-guided memory refresh", str(row["memory_update_boundary"]))
        self.assertEqual(verifier.remembered_locations, [])
        self.assertIn("oracle metadata is offline labels only", str(row["row_claim_boundary"]))
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["policy_verified_rows"], 0)
        self.assertEqual(summary["policy_skipped_rows"], 1)
        self.assertEqual(summary["memory_update_candidate_rows"], 0)
        self.assertEqual(summary["memory_updated_rows"], 0)
        self.assertEqual(summary["task_execution_evaluated_rows"], 0)
        self.assertEqual(summary["task_execution_not_executed_rows"], 1)
        self.assertEqual(summary["downstream_task_success_rows"], 0)

    def test_longer_time_interval_increases_stale_probability(self) -> None:
        before_obj: dict[str, object] = {
            "objectType": "Apple",
            "objectId": "Apple|+00.50|+01.10|+00.50",
            "pickupable": True,
            "position": {"x": 0.50, "y": 1.10, "z": 0.50},
        }
        remembered_pos: dict[str, float] = {"x": 0.50, "y": 1.10, "z": 0.50}
        found_pos: dict[str, float] = {"x": 1.40, "y": 1.10, "z": 1.40}

        inputs_default = _build_uncertainty_inputs(
            before_obj, remembered_pos,
            target_visible_after_revisit=True, visibility_error=None,
            found_pos=found_pos, teleport_success=True, time_interval=1.0,
        )
        inputs_long = _build_uncertainty_inputs(
            before_obj, remembered_pos,
            target_visible_after_revisit=True, visibility_error=None,
            found_pos=found_pos, teleport_success=True, time_interval=3.0,
        )
        self.assertEqual(inputs_default.time_interval, 1.0)
        self.assertEqual(inputs_long.time_interval, 3.0)
        est_default = _estimate_stale_uncertainty(inputs_default)
        est_long = _estimate_stale_uncertainty(inputs_long)
        self.assertAlmostEqual(est_default.probability, 0.82)
        self.assertGreater(est_long.probability, est_default.probability)
        self.assertLessEqual(est_long.probability, 0.95)

    def test_select_memory_targets_can_prioritize_openable_target_type(self) -> None:
        before_objects: list[dict[str, object]] = [
            {
                "objectType": "Apple",
                "objectId": "Apple|+00.50|+01.10|+00.50",
                "pickupable": True,
                "openable": False,
                "position": {"x": 0.50, "y": 1.10, "z": 0.50},
            },
            {
                "objectType": "Cabinet",
                "objectId": "Cabinet|+00.20|+00.00|+00.20",
                "pickupable": False,
                "openable": True,
                "position": {"x": 0.20, "y": 0.00, "z": 0.20},
            },
        ]

        selected = _select_memory_targets(before_objects, 1, preferred_objects=("Cabinet",))

        self.assertEqual(selected[0]["objectType"], "Cabinet")
        self.assertEqual(selected[0]["openable"], True)

    def test_policy_tie_at_threshold_skips_verifier(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeClosedLoopController()
        verifier = FakeVerifier()

        def controller_factory(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verification_cost=0.85,
                stale_action_cost=1.0,
                active_verification_threshold=0.0,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        rows = cast(list[dict[str, object]], result["rows"])
        row = rows[0]
        self.assertEqual(row["p_stale_estimate"], 0.82)
        self.assertEqual(row["expected_verification_value"], -0.03)
        self.assertEqual(row["policy_should_verify"], False)
        self.assertEqual(row["verifier_used"], False)
        self.assertEqual(row["memory_update_instrumented"], False)
        self.assertEqual(row["memory_update_action"], "not_applicable")
        self.assertEqual(verifier.remembered_locations, [])

    def test_budgeted_runner_verifies_only_top_expected_value_candidate(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeBudgetController()
        verifier = FakeFlexibleVerifier()

        def controller_factory(**kwargs: object) -> FakeBudgetController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeFlexibleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=2,
                targets_per_scene_seed=2,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                verification_budget=1,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        rows = cast(list[dict[str, object]], result["rows"])
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["target"] for row in rows], ["Apple", "Book"])
        self.assertGreater(cast(float, rows[0]["expected_verification_value"]), cast(float, rows[1]["expected_verification_value"]))
        self.assertEqual(rows[0]["candidate_rank_by_ev"], 1)
        self.assertEqual(rows[1]["candidate_rank_by_ev"], 2)
        self.assertEqual(rows[0]["candidate_selected_by_budget"], True)
        self.assertEqual(rows[1]["candidate_selected_by_budget"], False)
        self.assertEqual(rows[0]["verifier_used"], True)
        self.assertEqual(rows[1]["verifier_used"], False)
        self.assertEqual(rows[0]["memory_update_instrumented"], True)
        self.assertEqual(rows[0]["memory_update_action"], "none")
        self.assertEqual(rows[1]["memory_update_instrumented"], False)
        self.assertEqual(rows[1]["memory_update_action"], "not_applicable")
        for row in rows:
            self._assert_verification_only_execution_boundary(row)
        self.assertFalse(any(row["downstream_task_success"] is True and row["task_execution_evaluated"] is not True for row in rows))
        self.assertEqual(rows[0]["budget_remaining_after_selection"], 0)
        self.assertEqual(rows[1]["budget_selection_reason"], "skipped_by_verification_budget")
        self.assertEqual(verifier.targets, ["Apple"])
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["verification_budget"], 1)
        self.assertEqual(summary["candidate_rows"], 2)
        self.assertEqual(summary["budget_selected_rows"], 1)
        self.assertEqual(summary["budget_skipped_rows"], 1)
        self.assertEqual(summary["mean_selected_expected_verification_value"], rows[0]["expected_verification_value"])
        self.assertEqual(summary["memory_update_candidate_rows"], 0)
        self.assertEqual(summary["memory_updated_rows"], 0)
        self.assertEqual(summary["task_execution_evaluated_rows"], 0)
        self.assertEqual(summary["task_execution_not_executed_rows"], 2)
        self.assertEqual(summary["downstream_task_success_rows"], 0)

    def test_case_list_filters_targets_and_reports_unmatched_cases(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeBudgetController()
        verifier = FakeFlexibleVerifier()

        def controller_factory(**kwargs: object) -> FakeBudgetController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeFlexibleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            case_list = out_dir / "cases.json"
            case_list.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "row_idx": 19,
                                "scene": "FloorPlan1",
                                "seed": 7,
                                "target": "Book",
                                "oracle_stale_label": True,
                                "decision_stale": False,
                            },
                            {
                                "row_idx": 4,
                                "scene": "FloorPlan201",
                                "seed": 7,
                                "target": "Newspaper",
                                "oracle_stale_label": True,
                                "decision_stale": False,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=5,
                targets_per_scene_seed=2,
                out_dir=out_dir,
                case_list=case_list,
                verifier_threshold=0.05,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        rows = cast(list[dict[str, object]], result["rows"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["target"], "Book")
        self.assertEqual(row["case_list_source"], str(case_list))
        self.assertEqual(row["case_list_matched"], True)
        self.assertEqual(row["case_list_source_row_idx"], 19)
        self.assertEqual(row["targeted_case_reason"], "prior_recorded_false_negative_slice")
        self.assertNotEqual(row["targeted_case_reason"], "oracle_stale_label")
        self.assertEqual(verifier.targets, ["Book"])
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["case_list_source"], str(case_list))
        self.assertEqual(summary["case_list_rows"], 2)
        self.assertEqual(summary["case_list_matched_rows"], 1)
        self.assertEqual(summary["case_list_unmatched_rows"], 1)
        unmatched = cast(list[dict[str, object]], summary["case_list_unmatched_cases"])
        self.assertEqual(unmatched, [{"scene": "FloorPlan201", "seed": 7, "target": "Newspaper", "source_row_idx": 4}])
        self.assertEqual(summary["memory_updated_rows"], 0)

    def test_stepwise_revisit_mode_records_measured_path_without_teleportfull(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeStepwiseRevisitController()
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakeStepwiseRevisitController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                execute_task_bridge=True,
                revisit_mode="stepwise",
            )
            result = run_live_gsam_closed_loop(
                config,
                controller_factory=controller_factory,
                verifier_factory=verifier_factory,
                capability=capability,
            )

        row = cast(list[dict[str, object]], result["rows"])[0]
        self.assertEqual(row["revisit_action"], "stepwise_navigation")
        self.assertEqual(row["used_teleportfull_for_revisit"], False)
        self.assertGreater(cast(int, row["revisit_path_steps"]), 0)
        self.assertIn("MoveAhead", cast(list[str], row["revisit_navigation_actions"]))
        self.assertNotIn("TeleportFull", cast(list[str], row["revisit_navigation_actions"]))
        self.assertEqual(row["task_execution_evaluated"], True)
        self.assertEqual(row["execute_action"], "PickupObject")

    def test_stale_verified_row_mutates_memory_store_snapshot(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeClosedLoopController()
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        rows = cast(list[dict[str, object]], result["rows"])
        row = rows[0]
        self.assertEqual(row["decision_stale"], True)
        self.assertEqual(row["memory_update_instrumented"], True)
        self.assertEqual(row["memory_update_action"], "propose_detector_evidence_refresh")
        self.assertEqual(row["memory_update_source"], "live_controller_revisit_detector_verifier_evidence")
        self.assertEqual(row["memory_update_old_location"], row["remembered_location"])
        self.assertEqual(row["memory_update_candidate_location"], row["found_position_after_revisit"])
        self.assertEqual(row["memory_update_confidence"], 0.77)
        self.assertNotEqual(row["memory_update_source"], "oracle_stale_label")
        self.assertNotEqual(row["memory_update_source"], "oracle_distance")
        self.assertNotEqual(row["memory_update_source"], "true_location_offline")
        self.assertNotEqual(row["memory_update_candidate_location"], row["true_location_offline"])
        self.assertIn("controller-backed evidence-guided memory refresh", str(row["memory_update_boundary"]))
        self.assertIn("not persistent storage", str(row["memory_update_boundary"]))
        self.assertIn("not recovery search", str(row["memory_update_boundary"]))
        self.assertIn("not task success", str(row["memory_update_boundary"]))
        self.assertEqual(row["memory_mutated"], True)
        self.assertIsNotNone(row["memory_old_position"])
        self.assertIsNotNone(row["memory_new_position"])
        self.assertNotEqual(row["memory_old_position"], row["memory_new_position"])
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["memory_update_candidate_rows"], 1)
        self.assertEqual(summary["memory_updated_rows"], 1)

    def test_execute_task_bridge_records_pickup_success_for_mutated_stale_row(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controllers: list[FakeTaskExecutionController] = []
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakeTaskExecutionController:
            _ = kwargs
            controller = FakeTaskExecutionController()
            controllers.append(controller)
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                execute_task_bridge=True,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        rows = cast(list[dict[str, object]], result["rows"])
        row = rows[0]
        self.assertEqual(row["memory_mutated"], True)
        self.assertEqual(row["task_execution_evaluated"], True)
        self.assertEqual(row["execute_action"], "PickupObject")
        self.assertEqual(row["downstream_task_success"], True)
        self.assertIn("task_bridge", str(row["execution_boundary"]))
        self.assertTrue(any(action == "PickupObject" for controller in controllers for action, _ in controller.actions))
        bridge_controllers = [controller for controller in controllers if any(action == "PickupObject" for action, _ in controller.actions)]
        bridge_teleports = [kwargs for controller in bridge_controllers for action, kwargs in controller.actions if action == "TeleportFull"]
        self.assertTrue(any(kwargs.get("standing") is True for kwargs in bridge_teleports))
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["task_execution_evaluated_rows"], 1)
        self.assertEqual(summary["downstream_task_success_rows"], 1)

    def test_stepwise_revisit_mode_makes_task_bridge_avoid_teleportfull(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise")
        controller = FakeStepwiseTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.00|+01.10|+00.25",
            "target_pickupable": True,
            "target_openable": False,
            "decision_stale": True,
            "verifier_used": True,
            "memory_mutated": True,
            "memory_update_source": "live_controller_revisit_detector_verifier_evidence",
            "memory_new_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }

        fields = _execute_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        action_names = [action for action, _ in controller.actions]
        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "PickupObject")
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertIn("MoveAhead", action_names)
        self.assertNotIn("TeleportFull", action_names)
        self.assertIn("stepwise_task_bridge", str(fields["execution_boundary"]))

    def test_stepwise_task_bridge_uses_reachable_goal_and_succeeds(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise")
        controller = FakeStepwiseTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.00|+01.10|+00.25",
            "target_pickupable": True,
            "target_openable": False,
            "decision_stale": True,
            "verifier_used": True,
            "memory_mutated": True,
            "memory_update_source": "live_controller_revisit_detector_verifier_evidence",
            "memory_new_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }

        fields = _execute_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        action_names = [action for action, _ in controller.actions]
        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "PickupObject")
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertIn("GetReachablePositions", action_names)
        self.assertNotIn("TeleportFull", action_names)
        self.assertEqual(fields["task_bridge_navigation_mode"], "stepwise_navigation")
        self.assertEqual(fields["task_bridge_used_teleportfull"], False)
        self.assertGreater(cast(int, fields["task_bridge_path_steps"]), 0)
        self.assertEqual(fields["task_bridge_navigation_failure_reason"], None)


    def test_stepwise_task_bridge_routes_through_reachable_waypoints(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise")
        controller = FakeWaypointTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.50|+01.10|+00.50",
            "target_pickupable": True,
            "target_openable": False,
            "decision_stale": True,
            "verifier_used": True,
            "memory_mutated": True,
            "memory_update_source": "live_controller_revisit_detector_verifier_evidence",
            "memory_new_position": {"x": 0.50, "y": 1.10, "z": 0.50},
        }

        fields = _execute_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "PickupObject")
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertEqual(fields["task_bridge_navigation_failure_reason"], None)
        self.assertNotIn("TeleportFull", [action for action, _ in controller.actions])

    def test_stepwise_task_bridge_budget_scales_with_reachable_waypoints(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise")
        controller = FakeLongWaypointTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+03.50|+01.10|+00.00",
            "target_pickupable": True,
            "target_openable": False,
            "decision_stale": True,
            "verifier_used": True,
            "memory_mutated": True,
            "memory_update_source": "live_controller_revisit_detector_verifier_evidence",
            "memory_new_position": {"x": 3.5, "y": 1.10, "z": 0.0},
        }

        fields = _execute_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        action_names = [action for action, _ in controller.actions]
        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "PickupObject")
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertIsNone(fields["task_bridge_navigation_failure_reason"])
        self.assertGreater(len(cast(list[str], fields["task_bridge_navigation_actions"])), 12)
        self.assertNotIn("TeleportFull", action_names)

    def test_passive_closed_loop_skips_verifier_and_task_bridge(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakePassiveClosedLoopController()
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakePassiveClosedLoopController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(29,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                active_verification_threshold=999.0,
                execute_task_bridge=True,
                revisit_mode="stepwise",
            )
            result = run_live_gsam_closed_loop(
                config,
                controller_factory=controller_factory,
                verifier_factory=verifier_factory,
                capability=capability,
            )

        row = cast(list[dict[str, object]], result["rows"])[0]
        self.assertEqual(row["policy_should_verify"], False)
        self.assertEqual(row["verifier_used"], False)
        self.assertEqual(row["memory_mutated"], False)
        self.assertEqual(row["task_execution_evaluated"], False)
        self.assertEqual(row["execute_action"], "not_executed")
        self.assertEqual(row["downstream_task_success"], None)
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["verifier_used_rows"], 0)
        self.assertEqual(summary["memory_updated_rows"], 0)
        self.assertEqual(summary["task_execution_evaluated_rows"], 0)

    def test_execute_task_bridge_rejects_oracle_only_rows_without_detector_refresh(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True)
        controller = FakeTaskExecutionController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.50|+01.10|+00.50",
            "target_pickupable": True,
            "target_openable": False,
            "oracle_stale_label": True,
            "decision_stale": True,
            "verifier_used": False,
            "memory_mutated": True,
            "memory_update_source": "oracle_stale_label",
            "memory_new_position": {"x": 1.40, "y": 1.10, "z": 1.40},
        }

        fields = _execute_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self._assert_verification_only_execution_boundary(fields)
        self.assertFalse(any(action == "PickupObject" for action, _ in controller.actions))

    def test_execute_task_bridge_resolves_current_object_id_after_spawn(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controllers: list[FakeTaskExecutionRenamedObjectController] = []
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakeTaskExecutionRenamedObjectController:
            _ = kwargs
            controller = FakeTaskExecutionRenamedObjectController()
            controllers.append(controller)
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                execute_task_bridge=True,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        row = cast(list[dict[str, object]], result["rows"])[0]
        self.assertEqual(row["downstream_task_success"], True)
        bridge_controllers = [controller for controller in controllers if any(action == "PickupObject" for action, _ in controller.actions)]
        pickup_kwargs = [kwargs for controller in bridge_controllers for action, kwargs in controller.actions if action == "PickupObject"]
        self.assertTrue(any(kwargs.get("objectId") == "Apple|+01.40|+01.10|+01.40" for kwargs in pickup_kwargs))

    def test_execute_task_bridge_records_openobject_success_for_openable_row(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controllers: list[FakeTaskExecutionOpenObjectController] = []
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakeTaskExecutionOpenObjectController:
            _ = kwargs
            controller = FakeTaskExecutionOpenObjectController()
            controllers.append(controller)
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                execute_task_bridge=True,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        row = cast(list[dict[str, object]], result["rows"])[0]
        self.assertEqual(row["target_pickupable"], False)
        self.assertEqual(row["target_openable"], True)
        self.assertEqual(row["execute_action"], "OpenObject")
        self.assertEqual(row["downstream_task_success"], True)
        self.assertTrue(any(action == "OpenObject" for controller in controllers for action, _ in controller.actions))

    def test_execute_task_bridge_records_no_compatible_action_for_verified_stale_nonactionable_row(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True)
        controller = FakeTaskExecutionController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Toaster|-01.84|+00.90|+00.13",
            "target_pickupable": False,
            "target_openable": False,
            "decision_stale": True,
            "verifier_used": True,
            "memory_mutated": True,
            "memory_update_source": "live_controller_revisit_detector_verifier_evidence",
            "memory_new_position": {"x": -1.84, "y": 0.90, "z": 0.13},
        }

        fields = _execute_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "no_compatible_pickup_or_open_action")
        self.assertEqual(fields["downstream_task_success"], False)
        self.assertEqual(fields["task_execution_failure_reason"], "target_not_pickupable_or_openable")
        self.assertFalse(any(action in {"PickupObject", "OpenObject"} for action, _ in controller.actions))

    def test_execute_task_bridge_records_pickup_failure_summary(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controllers: list[FakeTaskExecutionFailureController] = []
        verifier = FakeStaleVerifier()

        def controller_factory(**kwargs: object) -> FakeTaskExecutionFailureController:
            _ = kwargs
            controller = FakeTaskExecutionFailureController()
            controllers.append(controller)
            return controller

        def verifier_factory() -> FakeStaleVerifier:
            return verifier

        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",),
                seeds=(7,),
                max_rows=1,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
                execute_task_bridge=True,
            )
            result = run_live_gsam_closed_loop(config, controller_factory=controller_factory, verifier_factory=verifier_factory, capability=capability)

        row = cast(list[dict[str, object]], result["rows"])[0]
        self.assertEqual(row["task_execution_evaluated"], True)
        self.assertEqual(row["execute_action"], "PickupObject")
        self.assertEqual(row["downstream_task_success"], False)
        self.assertEqual(row["task_execution_failure_reason"], "forced_pickup_failure")
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["task_execution_evaluated_rows"], 1)
        self.assertEqual(summary["downstream_task_success_rows"], 0)
        self.assertEqual(summary["downstream_task_failure_rows"], 1)

    def test_output_writes_json_csv_readme(self) -> None:
        result = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "status": "ok",
            "summary": {"rows": 1},
            "rows": [{"row_idx": 0, "target": "Apple"}],
            "claim_boundary": "boundary",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(result, out)
            self.assertTrue((out / "live_gsam_closed_loop.json").exists())
            self.assertTrue((out / "live_gsam_closed_loop.csv").exists())
            self.assertTrue((out / "README.md").exists())
            data = json.loads((out / "live_gsam_closed_loop.json").read_text())
            self.assertEqual(data["status"], "ok")
            self.assertIn("Apple", (out / "live_gsam_closed_loop.csv").read_text())

    def test_gsam_detection_history_produces_non_uniform_uncertainty(self) -> None:
        apple = {"objectType": "Apple", "pickupable": True, "position": {"x": 0.0, "y": 1.0, "z": 0.0}}
        newspaper = {"objectType": "Newspaper", "pickupable": True, "position": {"x": 0.0, "y": 1.0, "z": 0.0}}
        remembered = {"x": 0.0, "y": 1.0, "z": 0.0}
        history: dict[str, list[bool]] = {"Apple": [True, True, True, True], "Newspaper": [False, False, False, False]}
        apple_inputs = _build_uncertainty_inputs(apple, remembered, target_visible_after_revisit=False, visibility_error="target_not_visible", found_pos=None, teleport_success=True, time_interval=1.0, gsam_detection_history=history)
        newspaper_inputs = _build_uncertainty_inputs(newspaper, remembered, target_visible_after_revisit=False, visibility_error="target_not_visible", found_pos=None, teleport_success=True, time_interval=1.0, gsam_detection_history=history)
        self.assertNotEqual(apple_inputs.detector_confidence, newspaper_inputs.detector_confidence, "History should produce different predicted confidences")
        self.assertGreater(cast(float, apple_inputs.detector_confidence), cast(float, newspaper_inputs.detector_confidence))
        self.assertEqual(apple_inputs.historical_failure, 0.0)
        self.assertGreater(newspaper_inputs.historical_failure, 0.0)
        apple_p = _estimate_stale_uncertainty(apple_inputs).probability
        newspaper_p = _estimate_stale_uncertainty(newspaper_inputs).probability
        self.assertNotEqual(apple_p, newspaper_p, "P(stale) should differ when history differs")
        self.assertLess(apple_p, newspaper_p)

    def test_build_uncertainty_inputs_computes_same_type_density_from_metadata(self) -> None:
        apple = {"objectType": "Apple", "pickupable": True, "position": {"x": 0.0, "y": 1.0, "z": 0.0}}
        remembered = {"x": 0.0, "y": 1.0, "z": 0.0}
        after_objs: list[dict[str, object]] = [
            {"objectType": "Apple", "position": {"x": 0.1, "y": 1.0, "z": 0.1}},   # nearby same-type
            {"objectType": "Apple", "position": {"x": 0.2, "y": 1.0, "z": 0.0}},   # nearby same-type
            {"objectType": "Apple", "position": {"x": 2.0, "y": 1.0, "z": 2.0}},   # far same-type
            {"objectType": "Book", "position": {"x": 0.1, "y": 1.0, "z": 0.0}},    # different type
        ]
        inputs = _build_uncertainty_inputs(
            apple, remembered,
            target_visible_after_revisit=True, visibility_error=None,
            found_pos={"x": 0.1, "y": 1.0, "z": 0.1}, teleport_success=True,
            after_metadata_objects=after_objs,
        )
        self.assertEqual(inputs.same_type_scene_total, 3)
        self.assertEqual(inputs.same_type_nearby_count, 2)
        self.assertAlmostEqual(inputs.same_type_density_score, 2.0 / 3.0, places=3)

    def test_build_uncertainty_inputs_without_metadata_has_zero_density(self) -> None:
        apple = {"objectType": "Apple", "pickupable": True, "position": {"x": 0.0, "y": 1.0, "z": 0.0}}
        remembered = {"x": 0.0, "y": 1.0, "z": 0.0}
        inputs = _build_uncertainty_inputs(
            apple, remembered,
            target_visible_after_revisit=True, visibility_error=None,
            found_pos={"x": 0.0, "y": 1.0, "z": 0.0}, teleport_success=True,
        )
        self.assertEqual(inputs.same_type_scene_total, 0)
        self.assertEqual(inputs.same_type_nearby_count, 0)
        self.assertEqual(inputs.same_type_density_score, 0.0)

    def test_estimate_stale_uncertainty_penalizes_high_density(self) -> None:
        inputs_low = StaleUncertaintyInputs(
            detector_confidence=0.5, depth_consistency=0.8,
            object_mobility_prior=0.5, time_interval=1.0, task_importance=1.0,
            historical_failure=0.0, scene_change_score=1.0,
            target_visible_after_revisit=True, visibility_error=None,
            distance_from_remembered=0.03, teleport_success=True,
            same_type_density_score=0.0, same_type_nearby_count=0, same_type_scene_total=3,
        )
        inputs_high = StaleUncertaintyInputs(
            detector_confidence=0.5, depth_consistency=0.8,
            object_mobility_prior=0.5, time_interval=1.0, task_importance=1.0,
            historical_failure=0.0, scene_change_score=1.0,
            target_visible_after_revisit=True, visibility_error=None,
            distance_from_remembered=0.03, teleport_success=True,
            same_type_density_score=0.75, same_type_nearby_count=3, same_type_scene_total=4,
        )
        low_p = _estimate_stale_uncertainty(inputs_low).probability
        high_p = _estimate_stale_uncertainty(inputs_high).probability
        self.assertLess(high_p, low_p, "High same-type density should reduce p(stale)")

    def test_visual_instance_confusion_prior_flags_textured_duplicate_prone_types(self) -> None:
        self.assertEqual(_visual_instance_confusion_prior("Apple"), 0.0)
        self.assertGreater(_visual_instance_confusion_prior("Newspaper"), 0.0)
        self.assertGreater(_visual_instance_confusion_prior("Book"), 0.0)

    def test_calibrated_ev_penalty_uses_visual_prior_without_density(self) -> None:
        apple_inputs = StaleUncertaintyInputs(
            detector_confidence=0.5, depth_consistency=None,
            object_mobility_prior=0.85, time_interval=1.0, task_importance=1.0,
            historical_failure=0.0, scene_change_score=1.0,
            target_visible_after_revisit=False, visibility_error="target_not_visible",
            distance_from_remembered=None, teleport_success=True,
            same_type_density_score=0.0, same_type_nearby_count=0, same_type_scene_total=1,
            visual_instance_confusion_prior=0.0,
        )
        newspaper_inputs = StaleUncertaintyInputs(
            detector_confidence=0.5, depth_consistency=None,
            object_mobility_prior=0.85, time_interval=1.0, task_importance=1.0,
            historical_failure=0.0, scene_change_score=1.0,
            target_visible_after_revisit=False, visibility_error="target_not_visible",
            distance_from_remembered=None, teleport_success=True,
            same_type_density_score=0.0, same_type_nearby_count=0, same_type_scene_total=1,
            visual_instance_confusion_prior=0.75,
        )
        apple_p, apple_reason = _apply_calibrated_ev_penalty(0.82, apple_inputs, "base")
        newspaper_p, newspaper_reason = _apply_calibrated_ev_penalty(0.82, newspaper_inputs, "base")
        self.assertEqual(apple_p, 0.82)
        self.assertEqual(apple_reason, "base")
        self.assertLess(newspaper_p, apple_p)
        self.assertIn("visual_instance_confusion_calibrated", newspaper_reason)

    def test_runner_with_ev_variant_calibrated_adds_density_fields_to_rows(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeClosedLoopController()
        verifier = FakeVerifier()
        def cf(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            return controller
        def vf() -> FakeVerifier:
            return verifier
        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",), seeds=(7,), max_rows=3, targets_per_scene_seed=1,
                out_dir=Path(tmp), ev_variant="calibrated",
            )
            result = run_live_gsam_closed_loop(config, controller_factory=cf, verifier_factory=vf, capability=capability)
        rows = cast(list[dict[str, object]], result["rows"])
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["ev_variant"], "calibrated")
            self.assertIn("same_type_density_score", row)
            self.assertIn("same_type_nearby_count", row)
            self.assertIn("same_type_scene_total", row)
            self.assertIn("visual_instance_confusion_prior", row)
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["ev_variant"], "calibrated")
        self.assertIn("same_type_density_rows", summary)
        self.assertIn("visual_instance_confusion_prior_rows", summary)

    def test_runner_with_ev_variant_current_has_default_fields(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeClosedLoopController()
        verifier = FakeVerifier()
        def cf(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            return controller
        def vf() -> FakeVerifier:
            return verifier
        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1",), seeds=(7,), max_rows=1, targets_per_scene_seed=1,
                out_dir=Path(tmp), ev_variant="current",
            )
            result = run_live_gsam_closed_loop(config, controller_factory=cf, verifier_factory=vf, capability=capability)
        rows = cast(list[dict[str, object]], result["rows"])
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["ev_variant"], "current")
            self.assertIn("same_type_density_score", row)
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["ev_variant"], "current")


class FakePassiveTaskBridgeController(FakeStepwiseTaskBridgeController):
    """A stepwise passive controller that never issues TeleportFull."""

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "PickupObject":
            return _FakeEvent(
                metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True},
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
            self.agent_heading = 0.0
            return _FakeEvent(
                metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True},
                events=[],
            )
        return FakeStepwiseRevisitController.step(self, action, **kwargs)

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.00|+01.10|+00.25",
            "visible": True,
            "pickupable": True,
            "position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }


class FakePassiveNavigationFailureController(FakeStepwiseRevisitController):
    """A stepwise controller where MoveAhead always collides."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.agent_heading = 0.0

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append((action, cast(dict[str, object], kwargs)))
        if action == "GetReachablePositions":
            return _FakeEvent(
                metadata=self._metadata([]) | {"actionReturn": [{"x": 0.0, "y": 0.9, "z": 0.25}]},
                events=[],
            )
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent(
                metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True},
                events=[],
            )
        if action == "MoveAhead":
            return _FakeEvent(
                metadata=self._metadata([]) | {"lastActionSuccess": False},
                events=[],
            )
        return super().step(action, **kwargs)

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|+00.00|+01.10|+00.25",
            "visible": True,
            "pickupable": True,
            "position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }


class ExecutePassiveTaskBridgeTest(unittest.TestCase):
    def test_passive_executes_from_memory_old_position_with_stepwise_nav(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise")
        controller = FakePassiveTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.00|+01.10|+00.25",
            "target_pickupable": True,
            "target_openable": False,
            "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
            "verifier_used": False,
            "memory_mutated": False,
            "decision_stale": None,
        }

        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        action_names = [action for action, _ in controller.actions]
        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "PickupObject")
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertIn("live_passive", str(fields["execution_boundary"]))
        self.assertIn("no_verifier", str(fields["execution_boundary"]))
        self.assertIn("no_memory_mutation", str(fields["execution_boundary"]))
        self.assertIn("PickupObject", action_names)
        self.assertNotIn("TeleportFull", action_names)
        self.assertNotIn("TeleportObject", action_names)
        self.assertEqual(fields["task_bridge_navigation_mode"], "stepwise_navigation_live_passive")
        self.assertEqual(fields["task_bridge_used_teleportfull"], False)

    def test_passive_skipped_when_execute_task_bridge_disabled(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=False)
        controller = FakePassiveTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }

        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], False)
        self.assertEqual(fields["execute_action"], "not_executed")
        self.assertIsNone(fields["downstream_task_success"])

    def test_passive_no_memory_old_position_returns_evaluated_failure(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True)
        controller = FakePassiveTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.00|+01.10|+00.25",
            "target_pickupable": True,
            "target_openable": False,
        }

        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["downstream_task_success"], False)
        self.assertIn("no_memory_old_position", str(fields["task_execution_failure_reason"]))
        self.assertIn("live_passive", str(fields["execution_boundary"]))

    def test_passive_no_compatible_action_returns_evaluated_failure(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True)
        controller = FakePassiveTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Toaster|+00.00|+01.10|+00.25",
            "target_pickupable": False,
            "target_openable": False,
            "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }

        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "no_compatible_pickup_or_open_action")
        self.assertEqual(fields["downstream_task_success"], False)
        self.assertEqual(fields["task_execution_failure_reason"], "target_not_pickupable_or_openable")

    def test_passive_stepwise_navigation_failure_is_evaluated_failure(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True)
        controller = FakePassiveNavigationFailureController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.00|+01.10|+00.25",
            "target_pickupable": True,
            "target_openable": False,
            "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }

        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["downstream_task_success"], False)
        self.assertIn("live_passive", str(fields["execution_boundary"]))
        self.assertIsNotNone(fields["task_execution_failure_reason"])

    def test_passive_boundary_contains_live_passive_markers(self) -> None:
        self.assertIn("live_passive", LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY)
        self.assertIn("no_verifier", LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY)
        self.assertIn("no_memory_mutation", LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY)
        self.assertIn("no TeleportFull", LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY)
        self.assertIn("no TeleportObject", LIVE_PASSIVE_TASK_BRIDGE_BOUNDARY)

    def test_passive_does_not_require_verifier_or_mutation_gates(self) -> None:
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True)
        controller = FakePassiveTaskBridgeController()
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|+00.00|+01.10|+00.25",
            "target_pickupable": True,
            "target_openable": False,
            "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
            # Deliberately omit verifier_used, memory_mutated, decision_stale
        }

        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )

        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["execute_action"], "PickupObject")
        self.assertEqual(fields["downstream_task_success"], True)


class FakeHonestInteractionController(FakeStepwiseRevisitController):
    """Stepwise controller whose Apple sits at a configurable position."""

    def __init__(
        self,
        object_position: tuple[float, float, float] = (0.0, 1.10, 0.25),
        *,
        visible_from_horizon: float = 0.0,
    ) -> None:
        super().__init__()
        self.object_position = {"x": object_position[0], "y": object_position[1], "z": object_position[2]}
        self.visible_from_horizon = visible_from_horizon
        self.camera_horizon = 0.0
        self.pickup_calls: list[dict[str, object]] = []

    def _metadata(self, objects: list[dict[str, object]]) -> dict[str, object]:
        metadata = super()._metadata(objects)
        agent = cast(dict[str, object], metadata["agent"])
        agent["cameraHorizon"] = self.camera_horizon
        return metadata

    def _apple_after(self) -> dict[str, object]:
        return {
            "objectType": "Apple",
            "objectId": "Apple|honest",
            "visible": abs(self.camera_horizon) >= self.visible_from_horizon,
            "pickupable": True,
            "position": dict(self.object_position),
        }

    def step(self, action: str, **kwargs: object) -> object:
        if action in {"LookDown", "LookUp"}:
            self.actions.append((action, cast(dict[str, object], kwargs)))
            degrees = float(cast(float, kwargs.get("degrees", 0.0)))
            self.camera_horizon += degrees if action == "LookDown" else -degrees
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        if action == "PickupObject":
            self.actions.append((action, cast(dict[str, object], kwargs)))
            self.pickup_calls.append(cast(dict[str, object], kwargs))
            distance = math.hypot(
                float(self.agent_position["x"]) - float(self.object_position["x"]),
                float(self.agent_position["z"]) - float(self.object_position["z"]),
            )
            success = bool(kwargs.get("forceAction")) or distance <= 1.5
            metadata = self._metadata([self._apple_after()]) | {"lastActionSuccess": success}
            if not success:
                metadata["errorMessage"] = "Object is too far"
            return _FakeEvent(metadata=metadata, events=[])
        if action == "InitialRandomSpawn":
            self.spawned = True
            self.agent_position = {"x": 0.0, "y": 0.9, "z": 0.0}
            self.agent_heading = 0.0
            return _FakeEvent(metadata=self._metadata([self._apple_after()]) | {"lastActionSuccess": True}, events=[])
        return super().step(action, **kwargs)


def _honest_task_bridge_row() -> dict[str, object]:
    return {
        "scene": "FloorPlan1",
        "seed": 7,
        "target_object_id": "Apple|honest",
        "target_pickupable": True,
        "target_openable": False,
        "verifier_used": True,
        "memory_update_source": "live_controller_revisit_detector_verifier_evidence",
        "memory_mutated": True,
        "decision_stale": True,
        "memory_new_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
    }


class HonestInteractionTest(unittest.TestCase):
    def test_honest_interaction_uses_force_action_false_and_succeeds_near_target(self) -> None:
        controller = FakeHonestInteractionController()
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise", honest_interaction=True)
        fields = _execute_task_bridge_for_row(
            _honest_task_bridge_row(),
            config=config,
            make_controller=lambda **kwargs: controller,
        )
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertEqual(fields["interaction_attempt_mode"], "honest_face_then_pick")
        self.assertEqual(fields["interaction_used_force_action"], False)
        self.assertEqual(controller.pickup_calls[-1].get("forceAction"), False)

    def test_forced_default_keeps_force_action_true(self) -> None:
        controller = FakeHonestInteractionController()
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise")
        fields = _execute_task_bridge_for_row(
            _honest_task_bridge_row(),
            config=config,
            make_controller=lambda **kwargs: controller,
        )
        self.assertEqual(fields["downstream_task_success"], True)
        self.assertEqual(fields["interaction_attempt_mode"], "forced_action")
        self.assertEqual(controller.pickup_calls[-1].get("forceAction"), True)

    def test_facing_uses_ai2thor_yaw_convention(self) -> None:
        forward_controller = FakeHonestInteractionController(object_position=(0.0, 1.10, 1.0))
        forward_controller.spawned = True
        fields, _success, _failure = _attempt_pickup_honest(
            forward_controller,
            object_id="Apple|honest",
            target_position={"x": 0.0, "y": 1.10, "z": 1.0},
            action="PickupObject",
        )
        self.assertEqual(fields["interaction_rotation_actions"], 0)

        right_controller = FakeHonestInteractionController(object_position=(1.0, 1.10, 0.0))
        right_controller.spawned = True
        fields, _success, _failure = _attempt_pickup_honest(
            right_controller,
            object_id="Apple|honest",
            target_position={"x": 1.0, "y": 1.10, "z": 0.0},
            action="PickupObject",
        )
        self.assertEqual(fields["interaction_rotation_actions"], 1)

    def test_honest_pickup_sweeps_horizon_until_visible(self) -> None:
        controller = FakeHonestInteractionController(object_position=(0.0, 1.10, 0.25), visible_from_horizon=30.0)
        controller.spawned = True
        fields, success, failure = _attempt_pickup_honest(
            controller,
            object_id="Apple|honest",
            target_position={"x": 0.0, "y": 1.10, "z": 0.25},
            action="PickupObject",
        )
        self.assertTrue(success)
        self.assertIsNone(failure)
        self.assertTrue(fields["interaction_visibility_sweep_used"])
        self.assertGreaterEqual(float(fields["interaction_horizon_deg"]), 30.0)
        self.assertEqual(controller.pickup_calls[-1].get("forceAction"), False)

    def test_honest_pickup_fails_when_object_out_of_reach(self) -> None:
        controller = FakeHonestInteractionController(object_position=(5.0, 1.10, 5.0))
        controller.spawned = True
        fields, success, failure = _attempt_pickup_honest(
            controller,
            object_id="Apple|honest",
            target_position={"x": 5.0, "y": 1.10, "z": 5.0},
            action="PickupObject",
        )
        self.assertFalse(success)
        self.assertEqual(failure, "Object is too far")
        self.assertEqual(fields["interaction_used_force_action"], False)
        self.assertGreater(float(fields["interaction_agent_object_distance"]), 1.5)
        self.assertEqual(controller.pickup_calls[-1].get("forceAction"), False)

    def test_passive_honest_interaction_fails_from_stale_location_when_object_moved(self) -> None:
        controller = FakeHonestInteractionController(object_position=(5.0, 1.10, 5.0))
        config = LiveGSAMClosedLoopConfig(execute_task_bridge=True, revisit_mode="stepwise", honest_interaction=True)
        row: dict[str, object] = {
            "scene": "FloorPlan1",
            "seed": 7,
            "target_object_id": "Apple|honest",
            "target_pickupable": True,
            "target_openable": False,
            "memory_old_position": {"x": 0.0, "y": 1.10, "z": 0.25},
        }
        fields = _execute_passive_task_bridge_for_row(
            row,
            config=config,
            make_controller=lambda **kwargs: controller,
        )
        self.assertEqual(fields["task_execution_evaluated"], True)
        self.assertEqual(fields["downstream_task_success"], False)
        self.assertEqual(fields["interaction_attempt_mode"], "honest_face_then_pick")
        self.assertEqual(fields["interaction_used_force_action"], False)
        self.assertEqual(controller.pickup_calls[-1].get("forceAction"), False)


class RichBeforeProbeTest(unittest.TestCase):
    def test_parse_args_enables_rich_before_probe(self) -> None:
        config = parse_args(["--rich-before-probe"])
        self.assertEqual(config.before_probe_actions, RICH_BEFORE_PROBE_ACTIONS)

    def test_parse_args_defaults_to_standard_before_probe(self) -> None:
        config = parse_args([])
        self.assertIsNone(config.before_probe_actions)

    def test_parse_args_max_alternate_poses(self) -> None:
        config = parse_args(["--max-alternate-poses", "2"])
        self.assertEqual(config.max_alternate_poses, 2)

    def test_alternate_reachable_poses_orders_excludes_and_caps(self) -> None:
        target = {"x": 0.0, "y": 0.9, "z": 0.0}
        reachable = [
            {"x": 0.0, "y": 0.9, "z": 3.0},
            {"x": 0.0, "y": 0.9, "z": 0.5},
            {"x": 0.0, "y": 0.9, "z": 1.0},
        ]
        alternates = _alternate_reachable_poses(target, reachable, 2, {"x": 0.0, "y": 0.9, "z": 0.5})
        self.assertEqual([pose["z"] for pose in alternates], [1.0, 3.0])


class SessionResilienceTest(unittest.TestCase):
    def test_runner_records_session_errors_and_continues(self) -> None:
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        calls = {"count": 0}
        controllers: list[FakeClosedLoopController] = []

        def controller_factory(**kwargs: object) -> FakeClosedLoopController:
            _ = kwargs
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("Unity process has exited")
            controller = FakeClosedLoopController()
            controllers.append(controller)
            return controller

        verifier = FakeVerifier()
        with tempfile.TemporaryDirectory() as tmp:
            config = LiveGSAMClosedLoopConfig(
                scenes=("FloorPlan1", "FloorPlan3"),
                seeds=(7,),
                max_rows=5,
                targets_per_scene_seed=1,
                out_dir=Path(tmp),
                verifier_threshold=0.05,
            )
            result = run_live_gsam_closed_loop(
                config,
                controller_factory=controller_factory,
                verifier_factory=lambda: verifier,
                capability=capability,
            )

        rows = cast(list[dict[str, object]], result["rows"])
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["session_error_count"], 1)
        session_errors = cast(list[dict[str, object]], summary["session_errors"])
        self.assertEqual(session_errors[0]["scene"], "FloorPlan3")
        self.assertIn("Unity process has exited", str(session_errors[0]["error"]))


if __name__ == "__main__":
    unittest.main()
