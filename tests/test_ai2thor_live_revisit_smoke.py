import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

from embodied_memory_pilot.ai2thor_adapter import CapabilityReport
from embodied_memory_pilot.ai2thor_live_revisit_smoke import (
    ResultRow,
    select_revisit_target,
    run_live_revisit_smoke,
    write_outputs,
)


class FakeController:
    def __init__(
        self,
        teleport_success: bool = True,
        off_grid_target: bool = False,
        target_visible_after_sweep: bool = False,
        post_spawn_target_position: tuple[float, float, float] | None = None,
    ) -> None:
        self.teleport_success: bool = teleport_success
        self.off_grid_target: bool = off_grid_target
        self.target_visible_after_sweep: bool = target_visible_after_sweep
        self.post_spawn_target_position: tuple[float, float, float] | None = post_spawn_target_position
        self.teleported: bool = False
        self.actions: list[str] = []
        self.action_kwargs: list[dict[str, object]] = []
        self.stopped: bool = False

    def step(self, action: str, **kwargs: object) -> SimpleNamespace:
        self.actions.append(action)
        self.action_kwargs.append(dict(kwargs))
        if action == "GetReachablePositions":
            return SimpleNamespace(
                metadata={
                    "lastActionSuccess": True,
                    "actionReturn": [{"x": 1.0, "y": 0.9, "z": 1.0}, {"x": 3.0, "y": 0.9, "z": 3.0}],
                    "objects": [],
                    "agent": {
                        "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                        "cameraHorizon": 30.0,
                    },
                }
            )
        if action == "InitialRandomSpawn":
            if self.post_spawn_target_position is not None:
                px, py, pz = self.post_spawn_target_position
                return SimpleNamespace(
                    metadata={
                        "lastActionSuccess": True,
                        "objects": [self._apple_at(px, py, pz), self._cabinet()],
                        "agent": {
                            "position": {"x": 2.0, "y": 0.9, "z": 2.0},
                            "rotation": {"x": 0.0, "y": 180.0, "z": 0.0},
                            "cameraHorizon": 30.0,
                        },
                    }
                )
            return SimpleNamespace(
                metadata={
                    "lastActionSuccess": True,
                    "objects": [self._apple(visible=True, off_grid=self.off_grid_target), self._cabinet()],
                    "agent": {
                        "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                        "cameraHorizon": 30.0,
                    },
                }
            )
        if action == "TeleportFull":
            self.teleported = True
            return SimpleNamespace(
                metadata={
                    "lastActionSuccess": self.teleport_success,
                    "errorMessage": "blocked by obstacle" if not self.teleport_success else "",
                    "objects": [self._apple(visible=self.teleport_success, off_grid=self.off_grid_target)],
                    "agent": {
                        "position": {"x": kwargs.get("x", 1.0), "y": kwargs.get("y", 0.9), "z": kwargs.get("z", 1.0)},
                        "rotation": kwargs.get("rotation", {"x": 0.0, "y": 90.0, "z": 0.0}),
                        "cameraHorizon": kwargs.get("horizon", 30.0),
                    },
                }
            )
        if self.teleported and self.target_visible_after_sweep:
            target_visible = action == "RotateRight"
            return SimpleNamespace(
                metadata={
                    "lastActionSuccess": True,
                    "errorMessage": "",
                    "objects": [self._apple(visible=target_visible, off_grid=self.off_grid_target), self._cabinet()],
                    "agent": {
                        "position": {"x": 1.0, "y": 0.9, "z": 1.0},
                        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                        "cameraHorizon": 30.0,
                    },
                }
            )
        return SimpleNamespace(
            metadata={
                "lastActionSuccess": True,
                "errorMessage": "",
                "objects": [self._apple(visible=True, off_grid=self.off_grid_target), self._cabinet()],
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                    "cameraHorizon": 30.0,
                },
            }
        )

    def stop(self) -> None:
        self.stopped = True

    @staticmethod
    def _apple(visible: bool, off_grid: bool = False) -> dict[str, object]:
        x = 1.1 if off_grid else 1.0
        z = 0.9 if off_grid else 1.0
        return FakeController._apple_at(x, 0.9, z, visible)

    @staticmethod
    def _apple_at(x: float, y: float, z: float, visible: bool = True) -> dict[str, object]:
        return {
            "objectId": f"Apple|{x:+06.2f}|{y:+06.2f}|{z:+06.2f}",
            "objectType": "Apple",
            "name": "Apple",
            "visible": visible,
            "pickupable": True,
            "openable": False,
            "isOpen": False,
            "receptacle": False,
            "position": {"x": x, "y": y, "z": z},
        }

    @staticmethod
    def _cabinet() -> dict[str, object]:
        return {
            "objectId": "Cabinet|+00.00|+00.00|+00.00",
            "objectType": "Cabinet",
            "name": "Cabinet",
            "visible": True,
            "pickupable": False,
            "openable": True,
            "isOpen": False,
            "receptacle": True,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        }


class AI2ThorLiveRevisitSmokeTest(unittest.TestCase):
    def capability(self, available: bool = True) -> CapabilityReport:
        return CapabilityReport(
            available=available,
            python="3.11",
            platform="Linux",
            ai2thor_version="4.2.0" if available else None,
            blocker=None if available else "missing ai2thor",
            install_hint="install ai2thor",
        )

    def controller_factory(self, controller: FakeController) -> Callable[..., FakeController]:
        def make_controller(**kwargs: object) -> FakeController:
            _ = kwargs
            return controller

        return make_controller

    def test_select_revisit_target_prefers_pickupable_target_object(self) -> None:
        target = select_revisit_target(
            [
                {"object_id": "Cabinet|1", "object_type": "Cabinet", "pickupable": False, "position": {"x": 0, "y": 0, "z": 0}},
                {"object_id": "Apple|1", "object_type": "Apple", "pickupable": True, "position": {"x": 1, "y": 0, "z": 1}},
            ]
        )

        self.assertIsNotNone(target)
        self.assertEqual(cast(Mapping[str, object], target)["object_type"], "Apple")

    def test_blocked_result_schema_for_missing_ai2thor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(available=False),
            )

            summary = cast(Mapping[str, object], result["summary"])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(summary["live_controller_started"], False)
            self.assertEqual(summary["teleport_success"], False)
            self.assertEqual(result["claim_boundary"], "AI2-THOR controller unavailable; no live revisit evidence")

    def test_live_revisit_success_writes_controller_level_row(self) -> None:
        controller = FakeController(teleport_success=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(),
                controller_factory=self.controller_factory(controller),
            )

            summary = cast(Mapping[str, object], result["summary"])
            rows = cast(list[ResultRow], result["rows"])
            row = rows[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(summary["teleport_success"], True)
            self.assertEqual(row["policy_action"], "revisit_memory_location")
            self.assertEqual(row["perception_claim"], "metadata_found_target_at_revisit")
            self.assertIsNotNone(row["found_by_sweep_position"])
            self.assertIn("TeleportFull", controller.actions)
            self.assertTrue(controller.stopped)

    def test_live_revisit_uses_nearest_reachable_position_for_teleport(self) -> None:
        controller = FakeController(teleport_success=True, off_grid_target=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(),
                controller_factory=self.controller_factory(controller),
            )

            teleport_index = controller.actions.index("TeleportFull")
            teleport_kwargs = controller.action_kwargs[teleport_index]
            rows = cast(list[ResultRow], result["rows"])
            row = rows[0]
            remembered_position = cast(Mapping[str, object], row["remembered_position"])
            revisit_position = cast(Mapping[str, object], row["revisit_position"])
            self.assertEqual(teleport_kwargs["x"], 1.0)
            self.assertEqual(teleport_kwargs["z"], 1.0)
            self.assertEqual(remembered_position["x"], 1.1)
            self.assertEqual(remembered_position["z"], 0.9)
            self.assertEqual(revisit_position["x"], 1.0)
            self.assertEqual(revisit_position["z"], 1.0)
            self.assertEqual(row["revisit_position_source"], "nearest_reachable")
            self.assertEqual(row["reachable_positions_available"], True)

    def test_live_revisit_sweeps_until_target_visible(self) -> None:
        controller = FakeController(teleport_success=True, target_visible_after_sweep=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(),
                controller_factory=self.controller_factory(controller),
            )

            rows = cast(list[ResultRow], result["rows"])
            row = rows[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(row["visibility_probe_actions"], ["Pass", "RotateRight"])
            self.assertEqual(row["visibility_probe_count"], 2)
            self.assertEqual(row["target_visible_after_revisit"], True)
            self.assertEqual(row["post_revisit_target_visible_by_metadata"], True)
            self.assertEqual(row["visibility_failure_reason"], "")
            self.assertIsNotNone(row.get("found_by_sweep_position"))
            self.assertEqual(row["perception_claim"], "metadata_found_target_at_revisit")

    def test_teleport_failure_is_degraded_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(),
                controller_factory=self.controller_factory(FakeController(teleport_success=False)),
            )

            summary = cast(Mapping[str, object], result["summary"])
            rows = cast(list[ResultRow], result["rows"])
            row = rows[0]
            self.assertEqual(result["status"], "degraded")
            self.assertEqual(summary["teleport_attempted"], True)
            self.assertEqual(summary["teleport_success"], False)
            self.assertEqual(row["teleport_error_message"], "blocked by obstacle")
            self.assertEqual(row["claim_boundary"], "controller-level live revisit only unless verifier_used=true")
            self.assertEqual(row["perception_claim"], "metadata_found_target_at_revisit")
            self.assertIsNotNone(row["found_by_sweep_position"])

    def test_post_spawn_target_tracking(self) -> None:
        controller = FakeController(teleport_success=True, post_spawn_target_position=(3.0, 0.9, 3.0))
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(),
                controller_factory=self.controller_factory(controller),
            )

            rows = cast(list[ResultRow], result["rows"])
            row = rows[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(row["target_moved_by_spawn"], True)
            post_spawn = cast(Mapping[str, object], row["target_post_spawn_position"])
            self.assertEqual(post_spawn["x"], 3.0)
            self.assertEqual(post_spawn["z"], 3.0)
            self.assertGreater(cast(float, row["target_movement_distance"]), 0.0)
            self.assertEqual(row["target_post_spawn_visible"], True)

    def test_oracle_teleport_to_post_spawn_position(self) -> None:
        controller = FakeController(teleport_success=True, post_spawn_target_position=(3.0, 0.9, 3.0))
        with tempfile.TemporaryDirectory() as tmp:
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=Path(tmp),
                capability=self.capability(),
                controller_factory=self.controller_factory(controller),
            )

            rows = cast(list[ResultRow], result["rows"])
            row = rows[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(row["target_moved_by_spawn"], True)
            self.assertEqual(row["target_post_spawn_visible"], True)
            self.assertEqual(row["oracle_teleport_attempted"], True)
            self.assertEqual(row["oracle_teleport_success"], True)
            self.assertEqual(row["oracle_teleport_source"], "post_spawn_oracle")
            oracle_pos = cast(Mapping[str, object], row["oracle_teleport_position"])
            self.assertEqual(oracle_pos["x"], 3.0)
            self.assertEqual(oracle_pos["z"], 3.0)
            self.assertIn("TeleportFull", controller.actions)
            teleport_index = controller.actions.index("TeleportFull")
            teleport_kwargs = controller.action_kwargs[teleport_index]
            self.assertEqual(teleport_kwargs["x"], 3.0)
            self.assertEqual(teleport_kwargs["z"], 3.0)

    def test_output_schema_contains_trace_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = run_live_revisit_smoke(
                scene="FloorPlan1",
                out_dir=out_dir,
                capability=self.capability(),
                controller_factory=self.controller_factory(FakeController()),
            )
            write_outputs(result, out_dir)

            self.assertTrue((out_dir / "live_revisit_smoke.json").exists())
            self.assertTrue((out_dir / "live_revisit_smoke.csv").exists())
            self.assertTrue((out_dir / "README.md").exists())
            csv_text = (out_dir / "live_revisit_smoke.csv").read_text(encoding="utf-8")
            self.assertIn("trace_id", csv_text)
            self.assertIn("target_object_type", csv_text)
            self.assertIn("claim_boundary", csv_text)


if __name__ == "__main__":
    _ = unittest.main()
