from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from embodied_memory_pilot.ai2thor_adapter import CapabilityReport
from embodied_memory_pilot.ai2thor_paired_hard_challenge_screen import (
    SCREENING_SCHEMA_VERSION,
    ScreeningConfig,
    freeze_case_list,
    run_screening,
    screen_scene_seed,
    write_outputs,
)


class _FakeEvent:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata


def _obj(object_type: str, object_id: str, position: tuple[float, float, float], *, pickupable: bool = True) -> dict[str, object]:
    return {
        "objectType": object_type,
        "objectId": object_id,
        "visible": True,
        "pickupable": pickupable,
        "position": {"x": position[0], "y": position[1], "z": position[2]},
    }


class FakeScreenController:
    def __init__(
        self,
        *,
        before_objects: list[dict[str, object]],
        after_objects: list[dict[str, object]],
        reachable: list[dict[str, float]],
        spawn_ok: bool = True,
    ) -> None:
        self.before_objects = before_objects
        self.after_objects = after_objects
        self.reachable = reachable
        self.spawn_ok = spawn_ok
        self.spawned = False
        self.stopped = False
        self.actions: list[str] = []

    def step(self, action: str, **kwargs: object) -> object:
        self.actions.append(action)
        if action == "GetReachablePositions":
            return _FakeEvent({"actionReturn": self.reachable})
        if action == "InitialRandomSpawn":
            self.spawned = True
            return _FakeEvent({"lastActionSuccess": self.spawn_ok, "objects": self.after_objects})
        return _FakeEvent({"objects": self.before_objects})

    def stop(self) -> None:
        self.stopped = True


def _capability(available: bool = True) -> CapabilityReport:
    return CapabilityReport(
        available=available,
        python="3.11",
        platform="linux",
        ai2thor_version="4.2.0" if available else None,
        blocker=None if available else "missing",
        install_hint="",
    )


def _config(**overrides: object) -> ScreeningConfig:
    base: dict[str, object] = {
        "universe": (("FloorPlan1", "Apple"),),
        "seeds": (7,),
        "d_passive_min": 2.0,
        "d_active_max": 1.5,
        "k": 8,
    }
    base.update(overrides)
    return ScreeningConfig(**cast(dict, base))


class ScreenSceneSeedTest(unittest.TestCase):
    def test_qualifies_when_stale_location_far_and_active_location_near(self) -> None:
        controller = FakeScreenController(
            before_objects=[_obj("Apple", "Apple|0", (0.5, 1.1, 0.5))],
            after_objects=[_obj("Apple", "Apple|0", (5.0, 1.1, 5.0))],
            reachable=[
                {"x": 0.45, "y": 0.9, "z": 0.45},
                {"x": 5.05, "y": 0.9, "z": 5.05},
            ],
        )
        rows = screen_scene_seed(controller, scene="FloorPlan1", seed=7, target_types=("Apple",), config=_config())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["qualifies"])
        self.assertEqual(row["status"], "qualifies")
        self.assertEqual(row["after_pairing_method"], "object_id")
        self.assertGreaterEqual(float(row["d_passive"]), 2.0)
        self.assertLessEqual(float(row["d_active"]), 1.5)

    def test_below_threshold_when_displacement_small(self) -> None:
        controller = FakeScreenController(
            before_objects=[_obj("Apple", "Apple|0", (0.5, 1.1, 0.5))],
            after_objects=[_obj("Apple", "Apple|0", (0.8, 1.1, 0.8))],
            reachable=[{"x": 0.45, "y": 0.9, "z": 0.45}],
        )
        rows = screen_scene_seed(controller, scene="FloorPlan1", seed=7, target_types=("Apple",), config=_config())
        self.assertFalse(rows[0]["qualifies"])
        self.assertEqual(rows[0]["status"], "below_threshold")

    def test_ambiguous_same_type_pairing_is_excluded(self) -> None:
        controller = FakeScreenController(
            before_objects=[_obj("Book", "Book|0", (0.5, 1.1, 0.5))],
            after_objects=[
                _obj("Book", "Book|a", (5.0, 1.1, 5.0)),
                _obj("Book", "Book|b", (6.0, 1.1, 6.0)),
            ],
            reachable=[{"x": 0.45, "y": 0.9, "z": 0.45}],
        )
        rows = screen_scene_seed(controller, scene="FloorPlan1", seed=7, target_types=("Book",), config=_config())
        self.assertFalse(rows[0]["qualifies"])
        self.assertEqual(rows[0]["status"], "ambiguous_after_pairing")

    def test_not_spawned_when_target_missing_after_spawn(self) -> None:
        controller = FakeScreenController(
            before_objects=[_obj("Apple", "Apple|0", (0.5, 1.1, 0.5))],
            after_objects=[_obj("Mug", "Mug|0", (5.0, 1.1, 5.0))],
            reachable=[{"x": 0.45, "y": 0.9, "z": 0.45}],
        )
        rows = screen_scene_seed(controller, scene="FloorPlan1", seed=7, target_types=("Apple",), config=_config())
        self.assertEqual(rows[0]["status"], "not_spawned")

    def test_not_pickupable_is_excluded(self) -> None:
        controller = FakeScreenController(
            before_objects=[_obj("Apple", "Apple|0", (0.5, 1.1, 0.5))],
            after_objects=[_obj("Apple", "Apple|0", (5.0, 1.1, 5.0), pickupable=False)],
            reachable=[{"x": 0.45, "y": 0.9, "z": 0.45}],
        )
        rows = screen_scene_seed(controller, scene="FloorPlan1", seed=7, target_types=("Apple",), config=_config())
        self.assertEqual(rows[0]["status"], "not_pickupable")


class FreezeCaseListTest(unittest.TestCase):
    def _row(self, scene: str, target: str, seed: int, qualifies: bool) -> dict[str, object]:
        return {
            "candidate_id": f"{scene}|{target}|{seed}",
            "scene": scene,
            "target": target,
            "seed": seed,
            "qualifies": qualifies,
            "d_passive": 3.0,
            "d_active": 0.5,
        }

    def test_freeze_sorts_by_scene_target_seed_and_truncates(self) -> None:
        rows = [
            self._row("FloorPlan201", "Pencil", 7, True),
            self._row("FloorPlan1", "Apple", 29, True),
            self._row("FloorPlan1", "Apple", 7, True),
            self._row("FloorPlan1", "Apple", 11, False),
        ]
        frozen = freeze_case_list(rows, 2)
        self.assertEqual([row["case_id"] for row in frozen], ["FloorPlan1|Apple|7", "FloorPlan1|Apple|29"])
        self.assertEqual([row["row_idx"] for row in frozen], [0, 1])

    def test_freeze_shortfall_returns_all_qualifying(self) -> None:
        rows = [self._row("FloorPlan1", "Apple", 7, True), self._row("FloorPlan1", "Apple", 11, False)]
        frozen = freeze_case_list(rows, 8)
        self.assertEqual(len(frozen), 1)


class RunScreeningTest(unittest.TestCase):
    def test_run_screening_aggregates_and_writes_case_list(self) -> None:
        controllers: list[FakeScreenController] = []

        def factory(**kwargs: object) -> FakeScreenController:
            controller = FakeScreenController(
                before_objects=[_obj("Apple", "Apple|0", (0.5, 1.1, 0.5))],
                after_objects=[_obj("Apple", "Apple|0", (5.0, 1.1, 5.0))],
                reachable=[{"x": 0.45, "y": 0.9, "z": 0.45}, {"x": 5.05, "y": 0.9, "z": 5.05}],
            )
            controllers.append(controller)
            return controller

        config = _config(seeds=(7, 11), k=1, universe=(("FloorPlan1", "Apple"),))
        result = run_screening(config, controller_factory=factory, capability=_capability())
        self.assertEqual(result["schema"], SCREENING_SCHEMA_VERSION)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["candidates"], 2)
        self.assertEqual(result["summary"]["qualifying"], 2)
        self.assertEqual(result["summary"]["frozen_cases"], 1)
        self.assertTrue(result["summary"]["shortfall"] is False)
        self.assertTrue(all(controller.stopped for controller in controllers))

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_outputs(result, out_dir)
            case_list = json.loads((out_dir / "case_list_paired_hard_challenge_frozen_v1.json").read_text())
            self.assertEqual(case_list["case_count"], 1)
            self.assertEqual(case_list["cases"][0]["scene"], "FloorPlan1")
            self.assertTrue((out_dir / "paired_hard_challenge_screen.json").exists())
            self.assertTrue((out_dir / "paired_hard_challenge_screen.csv").exists())

    def test_blocked_when_capability_unavailable(self) -> None:
        def factory(**kwargs: object) -> FakeScreenController:
            raise AssertionError("controller must not be created when capability is unavailable")

        result = run_screening(_config(), controller_factory=factory, capability=_capability(available=False))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["summary"]["frozen_cases"], 0)


if __name__ == "__main__":
    unittest.main()
