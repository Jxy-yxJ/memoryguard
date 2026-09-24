from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from embodied_memory_pilot.ai2thor_adapter import CapabilityReport
from embodied_memory_pilot.ai2thor_navigation_smoke import (
    NavigationSmokeConfig,
    nearest_reachable_point,
    run_navigation_smoke,
    write_outputs,
)


class FakeNavigationController:
    def __init__(self) -> None:
        self.actions: list[tuple[str, dict[str, object]]] = []
        self.position = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.rotation_y = 90.0
        self.stopped = False

    def step(self, action: str, **kwargs: object) -> SimpleNamespace:
        self.actions.append((action, dict(kwargs)))
        if action == "GetReachablePositions":
            return self._event(action_return=[{"x": 0.25, "y": 0.9, "z": 0.0}, {"x": 0.5, "y": 0.9, "z": 0.0}, {"x": 2.0, "y": 0.9, "z": 2.0}])
        if action == "InitialRandomSpawn":
            return self._event(objects=[self._apple(visible=False)], last_action_success=True)
        if action == "RotateRight":
            self.rotation_y = (self.rotation_y + 90.0) % 360.0
            return self._event(objects=[self._apple(visible=False)])
        if action == "RotateLeft":
            self.rotation_y = (self.rotation_y - 90.0) % 360.0
            return self._event(objects=[self._apple(visible=False)])
        if action == "MoveAhead":
            self.position = {"x": min(0.5, self.position["x"] + 0.25), "y": 0.9, "z": 0.0}
            visible = self.position["x"] >= 0.25
            return self._event(objects=[self._apple(visible=visible)], last_action_success=True)
        if action == "Pass":
            visible = self.position["x"] >= 0.25
            return self._event(objects=[self._apple(visible=visible)])
        raise AssertionError(f"unexpected action: {action}")

    def stop(self) -> None:
        self.stopped = True

    def _event(self, *, objects: list[dict[str, object]] | None = None, action_return: list[dict[str, float]] | None = None, last_action_success: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            metadata={
                "lastActionSuccess": last_action_success,
                "errorMessage": "",
                "actionReturn": action_return,
                "agent": {
                    "position": dict(self.position),
                    "rotation": {"x": 0.0, "y": self.rotation_y, "z": 0.0},
                    "cameraHorizon": 30.0,
                },
                "objects": objects or [],
            }
        )

    @staticmethod
    def _apple(*, visible: bool) -> dict[str, object]:
        return {
            "objectId": "Apple|+00.50|+00.90|+00.00",
            "objectType": "Apple",
            "name": "Apple",
            "visible": visible,
            "position": {"x": 0.25, "y": 0.9, "z": 0.0},
        }


class FakeHeadingAwareNavigationController:
    def __init__(self) -> None:
        self.actions: list[tuple[str, dict[str, object]]] = []
        self.position = {"x": -1.0, "y": 0.9, "z": 1.0}
        self.rotation_y = 270.0
        self.stopped = False

    def step(self, action: str, **kwargs: object) -> SimpleNamespace:
        self.actions.append((action, dict(kwargs)))
        if action == "GetReachablePositions":
            return self._event(action_return=[{"x": -1.0, "y": 0.9, "z": -0.25}, {"x": -1.0, "y": 0.9, "z": 0.0}, {"x": -1.0, "y": 0.9, "z": 0.25}])
        if action == "InitialRandomSpawn":
            return self._event(objects=[self._apple(visible=False)], last_action_success=True)
        if action == "RotateRight":
            self.rotation_y = (self.rotation_y + 90.0) % 360.0
            return self._event(objects=[self._apple(visible=False)])
        if action == "RotateLeft":
            self.rotation_y = (self.rotation_y - 90.0) % 360.0
            return self._event(objects=[self._apple(visible=False)])
        if action == "MoveAhead":
            if self.rotation_y % 360.0 == 0.0:
                self.position = {"x": self.position["x"], "y": 0.9, "z": self.position["z"] + 0.25}
            elif self.rotation_y % 360.0 == 90.0:
                self.position = {"x": self.position["x"] + 0.25, "y": 0.9, "z": self.position["z"]}
            elif self.rotation_y % 360.0 == 180.0:
                self.position = {"x": self.position["x"], "y": 0.9, "z": self.position["z"] - 0.25}
            else:
                self.position = {"x": self.position["x"] - 0.25, "y": 0.9, "z": self.position["z"]}
            visible = self.position["z"] <= -0.25
            return self._event(objects=[self._apple(visible=visible)], last_action_success=True)
        if action == "Pass":
            visible = self.position["z"] <= -0.25
            return self._event(objects=[self._apple(visible=visible)])
        raise AssertionError(f"unexpected action: {action}")

    def stop(self) -> None:
        self.stopped = True

    def _event(self, *, objects: list[dict[str, object]] | None = None, action_return: list[dict[str, float]] | None = None, last_action_success: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            metadata={
                "lastActionSuccess": last_action_success,
                "errorMessage": "",
                "actionReturn": action_return,
                "agent": {
                    "position": dict(self.position),
                    "rotation": {"x": 0.0, "y": self.rotation_y, "z": 0.0},
                    "cameraHorizon": 30.0,
                },
                "objects": objects or [],
            }
        )

    @staticmethod
    def _apple(*, visible: bool) -> dict[str, object]:
        return {
            "objectId": "Apple|-00.06|+01.16|-00.25",
            "objectType": "Apple",
            "name": "Apple",
            "visible": visible,
            "position": {"x": -0.0591, "y": 1.1578, "z": -0.2538},
        }


class TestAI2ThorNavigationSmoke(unittest.TestCase):
    def test_nearest_reachable_point_selects_closest_candidate(self) -> None:
        goal = {"x": 0.45, "y": 0.9, "z": 0.05}
        reachable = [{"x": 2.0, "y": 0.9, "z": 2.0}, {"x": 0.5, "y": 0.9, "z": 0.0}]

        selected, distance = nearest_reachable_point(goal, reachable)

        self.assertEqual(selected, {"x": 0.5, "y": 0.9, "z": 0.0})
        self.assertLess(distance, 0.08)

    def test_run_navigation_smoke_records_measured_path_without_teleportfull(self) -> None:
        controller = FakeNavigationController()
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")

        result = run_navigation_smoke(
            NavigationSmokeConfig(out_dir=Path("unused"), max_steps=4),
            controller_factory=lambda **kwargs: controller,
            capability=capability,
        )

        self.assertEqual(result["status"], "ok")
        row = result["row"]
        self.assertEqual(row["scene"], "FloorPlan1")
        self.assertEqual(row["seed"], 29)
        self.assertEqual(row["target"], "Apple")
        self.assertEqual(row["used_teleportfull_for_measured_path"], False)
        self.assertEqual(row["reached"], True)
        self.assertEqual(row["visible_at_destination"], True)
        self.assertGreater(row["path_steps"], 0)
        self.assertGreater(row["path_length_euclidean"], 0.0)
        self.assertNotIn("TeleportFull", [action for action, _ in controller.actions])
        self.assertTrue(controller.stopped)

    def test_heading_aware_navigation_reaches_goal_without_teleportfull(self) -> None:
        controller = FakeHeadingAwareNavigationController()
        capability = CapabilityReport(available=True, python="3.11", platform="linux", ai2thor_version="4.2.0", blocker=None, install_hint="")

        result = run_navigation_smoke(
            NavigationSmokeConfig(out_dir=Path("unused"), max_steps=8),
            controller_factory=lambda **kwargs: controller,
            capability=capability,
        )

        self.assertEqual(result["status"], "ok")
        row = result["row"]
        self.assertEqual(row["used_teleportfull_for_measured_path"], False)
        self.assertEqual(row["failure_reason"], None)
        self.assertEqual(row["reached"], True)
        self.assertEqual(row["visible_at_destination"], True)
        self.assertGreaterEqual(row["path_steps"], 1)
        self.assertGreater(row["path_length_euclidean"], 0.0)
        self.assertNotIn("TeleportFull", [action for action, _ in controller.actions])

    def test_write_outputs_creates_json_csv_and_readme(self) -> None:
        result = {
            "schema_version": "ai2thor_navigation_smoke.v1",
            "status": "ok",
            "row": {
                "scene": "FloorPlan1",
                "seed": 29,
                "target": "Apple",
                "used_teleportfull_for_measured_path": False,
                "path_steps": 2,
            },
            "claim_boundary": "mechanism instrumentation only",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(result, out)
            self.assertTrue((out / "navigation_smoke.json").exists())
            self.assertTrue((out / "navigation_smoke.csv").exists())
            self.assertTrue((out / "README.md").exists())
            loaded = json.loads((out / "navigation_smoke.json").read_text())
            self.assertEqual(loaded["row"]["used_teleportfull_for_measured_path"], False)


if __name__ == "__main__":
    unittest.main()
