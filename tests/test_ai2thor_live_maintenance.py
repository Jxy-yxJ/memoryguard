from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import cast

import numpy as np

from embodied_memory_pilot.ai2thor_adapter import CapabilityReport
from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import LocationDecision
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import RearrangementTask


class _FakeEvent:

    def __init__(self, metadata: dict[str, object], events: list[dict[str, object]], frame: object | None = None, instance_masks: dict[str, object] | None = None):
        self.metadata = metadata
        self.events = events
        self.frame = frame if frame is not None else np.zeros((6, 6, 3), dtype=np.uint8)
        self.instance_masks = instance_masks or {}


class FakeLiveMaintController:

    def __init__(
        self,
        *,
        target_event_visible: bool = True,
        _maint_visible: bool = False,
    ):
        self._actions: list[tuple[str, dict[str, object]]] = []
        self._stopped = False
        self.target_event_visible = target_event_visible
        self._maint_visible = _maint_visible

    def step(self, action: str, **kwargs: object) -> object:
        self._actions.append((action, cast(dict[str, object], kwargs)))

        visible_events: list[dict[str, object]] = []
        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        instance_masks: dict[str, object] = {}
        if self._maint_visible and action == "Pass":
            visible_events = [{
                "objectType": "Mug",
                "objectId": "Mug|-00.50|+01.10|+00.50",
                "visible": True, "isPickedUp": False, "pickupable": True,
                "position": {"x": -0.50, "y": 1.10, "z": 0.50},
            }]
            instance_masks = {
                "Mug|-00.50|+01.10|+00.50": np.array([[False, False, False, False, False, False],[False, True, True, False, False, False],[False, True, True, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False]])
            }
        elif self.target_event_visible and action == "Pass":
            visible_events = [{
                "objectType": "Apple",
                "objectId": "Apple|-00.47|+01.15|+00.48",
                "visible": True, "isPickedUp": False, "pickupable": True,
                "position": {"x": -0.47, "y": 1.15, "z": 0.48},
            }]
            instance_masks = {
                "Apple|-00.47|+01.15|+00.48": np.array([[False, False, False, False, False, False],[False, True, True, False, False, False],[False, True, True, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False]])
            }
        elif action not in ("GetReachablePositions", "InitialRandomSpawn", "TeleportFull", "RotateRight"):
            visible_events = [{
                "objectType": "Apple",
                "objectId": "Apple|-00.47|+01.15|+00.48",
                "visible": True, "isPickedUp": False, "pickupable": True,
                "position": {"x": -0.47, "y": 1.15, "z": 0.48},
            }]
            instance_masks = {
                "Apple|-00.47|+01.15|+00.48": np.array([[False, False, False, False, False, False],[False, True, True, False, False, False],[False, True, True, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False]])
            }

        if action == "GetReachablePositions":
            metadata: dict[str, object] = {
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
                "actionReturn": [
                    {"x": 1.0, "y": 0.9, "z": 1.0},
                    {"x": -1.0, "y": 0.9, "z": 0.5},
                ],
            }
        elif action == "InitialRandomSpawn":
            metadata = {
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
                "lastActionSuccess": True,
            }
        elif action == "TeleportFull":
            metadata = {
                "agent": {
                    "position": {"x": kwargs.get("x", 0.0), "y": kwargs.get("y", 0.9), "z": kwargs.get("z", 0.0)},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
                "lastActionSuccess": True,
            }
        else:
            metadata = {
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
            }
        if visible_events:
            metadata = dict(metadata)
            metadata["objects"] = visible_events
        return _FakeEvent(metadata=metadata, events=visible_events, frame=frame, instance_masks=instance_masks)

    def stop(self) -> None:
        self._stopped = True




class FakeGSAMBackend:
    name = "grounded_sam2_fake"

    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float):
        _ = task, remembered_location, threshold
        assert (image_dir / "FloorPlan1" / "before" / "image_manifest.jsonl").exists()
        assert (image_dir / "FloorPlan1" / "after" / "image_manifest.jsonl").exists()
        return LocationDecision(
            stale=True,
            confidence=1.0,
            reason="nearest_after_detection_normalized_frame_distance",
            matched_distance=0.25,
            detections_considered=3,
        )


class FallbackCaptureGSAMBackend:
    name = "grounded_sam2_fallback_capture_fake"

    def __init__(self) -> None:
        self.after_action_indices: list[int] = []

    def verify(self, task: RearrangementTask, *, remembered_location: str, image_dir: Path, threshold: float):
        _ = remembered_location, threshold
        after_manifest = image_dir / "FloorPlan1" / "after" / "image_manifest.jsonl"
        rows = cast(list[dict[str, object]], [json.loads(line) for line in after_manifest.read_text().splitlines()])
        self.after_action_indices = []
        for row in rows:
            action_idx = row.get("action_idx")
            if row.get("object_type") == "Apple" and isinstance(action_idx, (int, str)):
                self.after_action_indices.append(int(action_idx))
        assert task.target == "Apple"
        assert self.after_action_indices
        assert min(self.after_action_indices) >= 200
        return LocationDecision(
            stale=True,
            confidence=1.0,
            reason="nearest_after_detection_normalized_frame_distance",
            matched_distance=0.25,
            detections_considered=len(self.after_action_indices),
        )


class FallbackCaptureController(FakeLiveMaintController):
    def __init__(self) -> None:
        super().__init__(target_event_visible=False)
        self._teleport_full_count = 0
        self._pass_count = 0

    def step(self, action: str, **kwargs: object) -> object:
        self._actions.append((action, cast(dict[str, object], kwargs)))

        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        visible_events: list[dict[str, object]] = []
        instance_masks: dict[str, object] = {}
        if action == "Pass":
            self._pass_count += 1
            if self._teleport_full_count == 0 or self._teleport_full_count >= 2:
                visible_events = [{
                    "objectType": "Apple",
                    "object_id": "Apple|-00.47|+01.15|+00.48",
                    "object_type": "Apple",
                    "objectId": "Apple|-00.47|+01.15|+00.48",
                    "visible": True,
                    "isPickedUp": False,
                    "pickupable": True,
                    "position": {"x": -0.47, "y": 1.15, "z": 0.48},
                }]
                instance_masks = {
                    "Apple|-00.47|+01.15|+00.48": np.array([[False, False, False, False, False, False],[False, True, True, False, False, False],[False, True, True, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False]])
                }

        if action == "GetReachablePositions":
            metadata: dict[str, object] = {
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
                "actionReturn": [
                    {"x": 1.0, "y": 0.9, "z": 1.0},
                    {"x": -1.0, "y": 0.9, "z": 0.5},
                ],
            }
        elif action == "TeleportFull":
            metadata = {
                "agent": {
                    "position": {"x": kwargs.get("x", 0.0), "y": kwargs.get("y", 0.9), "z": kwargs.get("z", 0.0)},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
                "lastActionSuccess": True,
            }
            self._teleport_full_count += 1
        else:
            metadata = {
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraHorizon": 0.0,
                },
            }
        if visible_events:
            metadata = dict(metadata)
            metadata["objects"] = visible_events
        return _FakeEvent(metadata=metadata, events=visible_events, frame=frame, instance_masks=instance_masks)

def _scenes_probe() -> dict[str, object]:
    return cast(
        dict[str, object],
        {
            "scenes": [
                {
                    "status": "ok",
                    "scene": "FloorPlan1",
                    "random_seed": 0,
                    "before_visible_objects": [
                        {
                            "objectType": "Apple",
                            "object_id": "Apple|id",
                            "object_type": "Apple",
                            "objectId": "Apple|-00.47|+01.15|+00.48",
                            "visible": True,
                            "isPickedUp": False,
                            "pickupable": True,
                            "position": {"x": -0.47, "y": 1.15, "z": 0.48},
                        },
                        {
                            "objectType": "Mug",
                            "object_id": "Mug|id",
                            "object_type": "Mug",
                            "objectId": "Mug|-00.50|+01.10|+00.50",
                            "visible": True,
                            "isPickedUp": False,
                            "pickupable": True,
                            "position": {"x": -0.50, "y": 1.10, "z": 0.50},
                        },
                    ],
                    "after_visible_objects": [
                        {
                            "objectType": "Apple",
                            "object_id": "Apple|id",
                            "object_type": "Apple",
                            "objectId": "Apple|-00.47|+01.15|+00.48",
                            "visible": True,
                            "isPickedUp": False,
                            "pickupable": True,
                            "position": {"x": 1.0, "y": 1.15, "z": 1.0},
                        },
                    ],
                    "reachable_positions": [],
                },
            ],
        },
    )


class LiveMaintenanceSmokeTest(unittest.TestCase):
    def test_builds_tasks_from_scenes_probe(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import build_tasks_from_probe

        probe = _scenes_probe()
        tasks = build_tasks_from_probe(probe)
        self.assertGreaterEqual(len(tasks), 1)
        self.assertTrue(any(t.target == "Apple" for t in tasks))

    def test_passive_no_maintenance(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import run_live_maintenance_smoke, LiveMaintenanceConfig

        capability = CapabilityReport(
            available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint=""
        )
        controller = FakeLiveMaintController(target_event_visible=True)

        def factory(**kwargs: object) -> FakeLiveMaintController:
            _ = kwargs
            return controller

        probe = _scenes_probe()
        config = LiveMaintenanceConfig(
            maintenance_policy_name="passive",
            maintenance_budget_per_task=0,
            task_budget=3,
            random_seed=0,
        )
        result = run_live_maintenance_smoke(
            probe=probe,
            config=config,
            controller_factory=factory,
            capability=capability,
        )

        self.assertEqual(result["status"], "ok")
        summary = cast(dict[str, object], result["summary"])
        self.assertEqual(summary["maintenance_policy"], "passive")
        self.assertEqual(summary["maintenance_checks"], 0)
        self.assertTrue(controller._stopped)

    def test_maintenance_ignores_current_target_in_loop(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import run_live_maintenance_smoke, LiveMaintenanceConfig

        capability = CapabilityReport(
            available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint=""
        )
        controller = FakeLiveMaintController(target_event_visible=True, _maint_visible=True)

        def factory(**kwargs: object) -> FakeLiveMaintController:
            _ = kwargs
            return controller

        probe = _scenes_probe()
        config = LiveMaintenanceConfig(
            maintenance_policy_name="heuristic_active",
            maintenance_budget_per_task=1,
            task_budget=3,
            random_seed=0,
        )
        result = run_live_maintenance_smoke(
            probe=probe,
            config=config,
            controller_factory=factory,
            capability=capability,
        )
        rows = cast(list[dict[str, object]], result["rows"])
        if rows:
            maint_targets = [r.get("maintenance_target") for r in rows if r.get("maintenance_attempted")]
            for t in maint_targets:
                self.assertIsNotNone(t)

    def test_missing_ai2thor_returns_blocked(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import run_live_maintenance_smoke, LiveMaintenanceConfig

        capability = CapabilityReport(available=False, python="3.13", platform="linux", ai2thor_version="", blocker="", install_hint="")

        probe = _scenes_probe()
        config = LiveMaintenanceConfig()
        result = run_live_maintenance_smoke(
            probe=probe,
            config=config,
            controller_factory=None,
            capability=capability,
        )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("controller unavailable", str(result["claim_boundary"]))


    def test_grounded_sam_mode_writes_manifests_and_uses_backend(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import run_live_maintenance_smoke, LiveMaintenanceConfig, VerifierFactory
        import tempfile

        capability = CapabilityReport(
            available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint=""
        )
        controller = FakeLiveMaintController(target_event_visible=True, _maint_visible=True)

        def factory(**kwargs: object) -> FakeLiveMaintController:
            _ = kwargs
            return controller

        def verifier_factory() -> FakeGSAMBackend:
            return FakeGSAMBackend()

        probe = _scenes_probe()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            config = LiveMaintenanceConfig(
                maintenance_policy_name="heuristic_active",
                maintenance_budget_per_task=1,
                task_budget=3,
                random_seed=0,
                out_dir=out_dir,
            )
            result = run_live_maintenance_smoke(
                probe=probe,
                config=config,
                controller_factory=factory,
                capability=capability,
                verifier_backend_name="grounded_sam2",
                verifier_factory=cast(VerifierFactory, verifier_factory),
            )

            rows = cast(list[dict[str, object]], result["rows"])
            vr = cast(dict[str, object], rows[0]["verifier_result"])
            self.assertEqual(vr["verifier_name"], "grounded_sam2_fake")
            self.assertTrue(vr["verifier_used"])
            self.assertNotIn("crop_encoded_match", vr)
            self.assertTrue((out_dir / "images" / "FloorPlan1" / "before" / "image_manifest.jsonl").exists())
            self.assertTrue((out_dir / "images" / "FloorPlan1" / "after" / "image_manifest.jsonl").exists())

    def test_grounded_sam_verifies_after_multi_viewpoint_capture(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import run_live_maintenance_smoke, LiveMaintenanceConfig, VerifierFactory
        import tempfile

        capability = CapabilityReport(
            available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint=""
        )
        controller = FallbackCaptureController()
        backend = FallbackCaptureGSAMBackend()

        def factory(**kwargs: object) -> FallbackCaptureController:
            _ = kwargs
            return controller

        def verifier_factory() -> FallbackCaptureGSAMBackend:
            return backend

        probe = _scenes_probe()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            config = LiveMaintenanceConfig(
                maintenance_policy_name="heuristic_active",
                maintenance_budget_per_task=1,
                task_budget=1,
                random_seed=0,
                out_dir=out_dir,
            )
            result = run_live_maintenance_smoke(
                probe=probe,
                config=config,
                controller_factory=factory,
                capability=capability,
                verifier_backend_name="grounded_sam2",
                verifier_factory=cast(VerifierFactory, verifier_factory),
            )

            rows = cast(list[dict[str, object]], result["rows"])
            self.assertEqual(rows[0]["maintenance_perception_claim"], "metadata_found_maintenance_target_multi_viewpoint")
            vr = cast(dict[str, object], rows[0]["verifier_result"])
            self.assertTrue(vr["verifier_used"])
            self.assertEqual(vr["decision_reason"], "nearest_after_detection_normalized_frame_distance")
            self.assertTrue(backend.after_action_indices)
            self.assertGreaterEqual(min(backend.after_action_indices), 200)

    def test_output_writes_json_csv_readme(self) -> None:
        from embodied_memory_pilot.ai2thor_live_maintenance import run_live_maintenance_smoke, LiveMaintenanceConfig, write_outputs
        import tempfile

        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")
        controller = FakeLiveMaintController(target_event_visible=True)

        def factory(**kwargs: object) -> FakeLiveMaintController:
            _ = kwargs
            return controller

        probe = _scenes_probe()
        config = LiveMaintenanceConfig(task_budget=3, random_seed=0)
        result = run_live_maintenance_smoke(
            probe=probe,
            config=config,
            controller_factory=factory,
            capability=capability,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(result, out)
            self.assertTrue((out / "live_maintenance_smoke.json").exists())
            self.assertTrue((out / "live_maintenance_smoke.csv").exists())
            self.assertTrue((out / "README.md").exists())

            data = json.loads((out / "live_maintenance_smoke.json").read_text())
            self.assertEqual(data["status"], "ok")
            self.assertIn("rows", data)
            self.assertIn("summary", data)
            self.assertIn("claim_boundary", data)

            csv_text = (out / "live_maintenance_smoke.csv").read_text()
            self.assertIn("task_idx", csv_text)

            readme = (out / "README.md").read_text()
            self.assertIn("Live Maintenance Smoke", readme)
            self.assertIn("Claim Boundary", readme)


if __name__ == "__main__":
    _ = unittest.main()
