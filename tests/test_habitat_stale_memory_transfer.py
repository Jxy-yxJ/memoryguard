from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np

from embodied_memory_pilot.habitat_stale_memory_transfer import (
    CLAIM_BOUNDARY,
    DEFAULT_OUT_DIR,
    DEFAULT_SCENE,
    FALSE_BOUNDARY_FIELDS,
    OBSERVATION_BOUNDARY,
    ROW_CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    HabitatSmokeConfig,
    blocked_result,
    build_result,
    euclidean_distance,
    observation_hash,
    parse_args,
    apply_perception_verify_update_trace,
    write_outputs,
)


def _sample_row(row_idx: int = 0) -> dict[str, object]:
    return {
        "row_idx": row_idx,
        "scene": "scene.glb",
        "seed": 7,
        "before_position": [0.0, 0.0, 0.0],
        "after_position": [1.0, 0.0, 0.0],
        "revisit_position": [0.0, 0.0, 0.0],
        "before_rotation": [0.0, 0.0, 0.0, 1.0],
        "after_rotation": [0.0, 0.0, 0.0, 1.0],
        "revisit_rotation": [0.0, 0.0, 0.0, 1.0],
        "revisit_source": "set_agent_state_to_remembered_before_pose",
        "claim_boundary": CLAIM_BOUNDARY,
        "row_claim_boundary": ROW_CLAIM_BOUNDARY,
        "observation_boundary": OBSERVATION_BOUNDARY,
        "before_rgb_sha256": "before-rgb",
        "before_depth_sha256": "before-depth",
        "after_rgb_sha256": "after-rgb",
        "after_depth_sha256": "after-depth",
        "revisit_rgb_sha256": "before-rgb",
        "revisit_depth_sha256": "before-depth",
        "revisit_matches_before_rgb": True,
        "revisit_matches_before_depth": True,
        "after_differs_from_before_rgb": True,
        "after_differs_from_before_depth": True,
        "before_after_euclidean_distance": 1.0,
        "before_revisit_euclidean_distance": 0.0,
        "after_revisit_euclidean_distance": 1.0,
        "before_after_geodesic_distance": None,
        "before_after_geodesic_status": "no_path",
        "before_revisit_geodesic_distance": 0.0,
        "before_revisit_geodesic_status": "ok",
        "after_revisit_geodesic_distance": None,
        "after_revisit_geodesic_status": "error:RuntimeError",
        **FALSE_BOUNDARY_FIELDS,
    }


class HabitatStaleMemoryTransferTest(unittest.TestCase):
    def test_constants_and_claim_boundary_are_narrow(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "habitat_stale_memory_transfer.v1")
        self.assertEqual(DEFAULT_SCENE, Path("/home/jxy/coding/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"))
        self.assertEqual(DEFAULT_OUT_DIR, Path("results/habitat_stale_memory_transfer_smoke"))
        self.assertEqual(OBSERVATION_BOUNDARY, "rgb_depth_hash_only_no_semantics")
        self.assertIn("perception-backed verify/update", CLAIM_BOUNDARY)
        self.assertIn("pseudo visual landmarks", CLAIM_BOUNDARY)
        self.assertIn("not object semantics", CLAIM_BOUNDARY)
        self.assertIn("not ObjectNav", CLAIM_BOUNDARY)
        self.assertIn("not manipulation", CLAIM_BOUNDARY)
        self.assertIn("not task success", CLAIM_BOUNDARY)
        self.assertTrue(all(value is False for value in FALSE_BOUNDARY_FIELDS.values()))

    def test_config_defaults_match_plan(self) -> None:
        config = HabitatSmokeConfig()
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.num_rows, 5)
        self.assertEqual(config.meters_forward, 1.0)
        self.assertEqual(config.scene, DEFAULT_SCENE)
        self.assertEqual(config.out_dir, DEFAULT_OUT_DIR)

    def test_observation_hash_is_deterministic_and_shape_sensitive(self) -> None:
        first = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        same = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        reshaped = np.array([1, 2, 3, 4], dtype=np.uint8)
        changed = np.array([[1, 2], [3, 5]], dtype=np.uint8)

        self.assertEqual(observation_hash(first), observation_hash(same))
        self.assertNotEqual(observation_hash(first), observation_hash(reshaped))
        self.assertNotEqual(observation_hash(first), observation_hash(changed))

    def test_euclidean_distance(self) -> None:
        self.assertEqual(euclidean_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), 5.0)
        self.assertAlmostEqual(euclidean_distance((1.0, 2.0), (4.0, 6.0)), 5.0)
        with self.assertRaises(ValueError):
            _ = euclidean_distance((0.0, 1.0), (0.0, 1.0, 2.0))

    def test_build_result_schema_summary_capability_and_row_fields(self) -> None:
        row = _sample_row()
        capability = {
            "available": True,
            "python": "3.11",
            "platform": "linux",
            "habitat_sim_version": "0.test",
            "habitat_version": None,
            "scene_path": "scene.glb",
            "blocker": None,
        }
        result = build_result("scene.glb", 7, [row], capability=capability)
        summary = cast(dict[str, object], result["summary"])
        result_rows = cast(list[dict[str, object]], result["rows"])

        self.assertEqual(set(result.keys()), {"schema_version", "status", "capability", "claim_boundary", "summary", "rows"})
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["claim_boundary"], CLAIM_BOUNDARY)
        self.assertEqual(result["capability"], capability)
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["scene"], "scene.glb")
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["before_after_rows"], 1)
        self.assertEqual(summary["revisit_rows"], 1)
        self.assertEqual(summary["rgb_revisit_match_rows"], 1)
        self.assertEqual(summary["depth_revisit_match_rows"], 1)
        self.assertEqual(summary["mean_euclidean_before_after"], 1.0)
        self.assertEqual(summary["mean_geodesic_before_after"], None)
        self.assertEqual(summary["claim_boundary"], CLAIM_BOUNDARY)
        for key in FALSE_BOUNDARY_FIELDS:
            self.assertEqual(summary[key], False)
            self.assertEqual(result_rows[0][key], False)

        expected_row_fields = {
            "row_idx",
            "scene",
            "seed",
            "before_position",
            "after_position",
            "revisit_position",
            "before_rotation",
            "after_rotation",
            "revisit_rotation",
            "revisit_source",
            "claim_boundary",
            "row_claim_boundary",
            "observation_boundary",
            "before_rgb_sha256",
            "before_depth_sha256",
            "after_rgb_sha256",
            "after_depth_sha256",
            "revisit_rgb_sha256",
            "revisit_depth_sha256",
            "before_after_euclidean_distance",
            "before_revisit_euclidean_distance",
            "after_revisit_euclidean_distance",
            "before_after_geodesic_distance",
            "before_after_geodesic_status",
            "before_revisit_geodesic_distance",
            "before_revisit_geodesic_status",
            "after_revisit_geodesic_distance",
            "after_revisit_geodesic_status",
        }
        self.assertTrue(expected_row_fields.issubset(result_rows[0].keys()))
        self.assertEqual(result_rows[0]["revisit_source"], "set_agent_state_to_remembered_before_pose")
        self.assertEqual(result_rows[0]["observation_boundary"], OBSERVATION_BOUNDARY)

    def test_blocked_result_uses_top_level_schema_and_capability(self) -> None:
        config = HabitatSmokeConfig(scene=Path("missing.glb"), out_dir=Path("out"), seed=13)
        result = blocked_result(config, "habitat_sim_import_error", "No module named habitat_sim")
        summary = cast(dict[str, object], result["summary"])
        capability = cast(dict[str, object], result["capability"])

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(set(["schema_version", "status", "capability", "claim_boundary", "summary", "rows"]).issubset(result.keys()), True)
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["claim_boundary"], "Habitat-Sim runtime unavailable or scene failed; no Habitat geometry/image-hash transfer evidence.")
        self.assertEqual(summary["rows"], 0)
        self.assertEqual(summary["scene"], "missing.glb")
        self.assertEqual(summary["seed"], 13)
        self.assertEqual(summary["before_after_rows"], 0)
        self.assertEqual(summary["revisit_rows"], 0)
        self.assertEqual(summary["claim_boundary"], result["claim_boundary"])
        for key in ("available", "python", "platform", "habitat_sim_version", "habitat_version", "scene_path", "blocker"):
            self.assertIn(key, capability)
        self.assertEqual(capability["available"], False)
        self.assertEqual(capability["scene_path"], "missing.glb")
        for key in FALSE_BOUNDARY_FIELDS:
            self.assertEqual(summary[key], False)

    def test_write_outputs_json_csv_readme(self) -> None:
        result = build_result(
            "scene.glb",
            7,
            [_sample_row()],
            capability={
                "available": True,
                "python": "3.11",
                "platform": "linux",
                "habitat_sim_version": "0.test",
                "habitat_version": None,
                "scene_path": "scene.glb",
                "blocker": None,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_outputs(result, out_dir)
            json_path = out_dir / "habitat_stale_memory_transfer.json"
            csv_path = out_dir / "habitat_stale_memory_transfer.csv"
            readme_path = out_dir / "README.md"

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(readme_path.exists())
            data = cast(dict[str, object], json.loads(json_path.read_text(encoding="utf-8")))
            summary = cast(dict[str, object], data["summary"])
            self.assertEqual(data["schema_version"], SCHEMA_VERSION)
            self.assertEqual(summary["seed"], 7)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["row_idx"], "0")
            self.assertIn("before_rgb_sha256", rows[0])
            self.assertIn("row_claim_boundary", rows[0])
            readme = readme_path.read_text(encoding="utf-8")
            self.assertIn("Habitat-Sim Stale-Memory Transfer Smoke", readme)
            self.assertIn("perception-backed verify/update smoke", readme)
            self.assertIn("does not use object labels, GSAM, ObjectNav, manipulation, or task-success evaluation", readme)

    def test_parse_args_defaults_and_overrides(self) -> None:
        defaults = parse_args([])
        self.assertEqual(defaults.scene, DEFAULT_SCENE)
        self.assertEqual(defaults.out_dir, DEFAULT_OUT_DIR)
        self.assertEqual(defaults.seed, 7)
        self.assertEqual(defaults.num_rows, 5)
        self.assertEqual(defaults.meters_forward, 1.0)

        parsed = parse_args([
            "--scene",
            "custom.glb",
            "--out-dir",
            "custom-out",
            "--seed",
            "29",
            "--num-rows",
            "3",
            "--meters-forward",
            "1.25",
            "--width",
            "64",
            "--height",
            "48",
            "--sensor-height",
            "1.1",
        ])
        self.assertEqual(parsed.scene, Path("custom.glb"))
        self.assertEqual(parsed.out_dir, Path("custom-out"))
        self.assertEqual(parsed.seed, 29)
        self.assertEqual(parsed.num_rows, 3)
        self.assertEqual(parsed.meters_forward, 1.25)
        self.assertEqual(parsed.width, 64)
        self.assertEqual(parsed.height, 48)
        self.assertEqual(parsed.sensor_height, 1.1)

    def test_apply_perception_verify_update_trace_adds_platform_neutral_verify_update_fields(self) -> None:
        row = _sample_row()
        traced = apply_perception_verify_update_trace(row)
        evidence = cast(dict[str, object], traced["perception_evidence"])

        self.assertEqual(traced["platform"], "Habitat-Sim")
        self.assertEqual(traced["memory_anchor_type"], "pseudo_visual_landmark")
        self.assertEqual(traced["object_semantics_available"], False)
        self.assertEqual(traced["verifier_name"], "rgb_depth_hash_consistency_proxy")
        self.assertEqual(traced["verifier_decision"], "fresh")
        self.assertEqual(traced["update_action"], "keep_memory")
        self.assertEqual(traced["memory_after_update"], traced["memory_before"])
        self.assertEqual(evidence["rgb_revisit_matches_before"], True)
        self.assertEqual(evidence["depth_revisit_matches_before"], True)
        self.assertEqual(evidence["after_differs_from_before_rgb"], True)
        self.assertEqual(traced["shortcut_setup_used"], True)
        self.assertEqual(traced["task_action"], "not_evaluated")
        self.assertEqual(traced["task_success"], None)
        self.assertEqual(traced["task_success_evaluated"], False)
        self.assertIn("perception-backed verify/update", cast(str, traced["claim_boundary"]))
        self.assertIn("not task success", cast(str, traced["claim_boundary"]))


if __name__ == "__main__":
    _ = unittest.main()
