import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch

from PIL import Image

from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import (
    BackendLoadTimeoutError,
    GroundedSAM2LocationBackend,
    CropFilenameLocationBackend,
    ObjectDetection,
    evaluate_location_verification,
    parse_rearrangement_location,
)


class GroundedSAM2LocationVerifierTest(unittest.TestCase):
    def sample_probe(self):
        return {
            "status": "ok",
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                        {"x": 3, "y": 0, "z": 0},
                    ],
                    "before_visible_objects": [
                        {
                            "object_id": "Apple|+01.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Book|+02.00|+01.00|+00.00",
                            "object_type": "Book",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|+03.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Book|+02.00|+01.00|+00.00",
                            "object_type": "Book",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                }
            ],
        }

    def _touch_jpeg(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(path)


    def test_backend_load_timeout_raises_clear_error(self) -> None:
        import time

        backend = GroundedSAM2LocationBackend(
            grounding_config=Path("missing_config.py"),
            grounding_checkpoint=Path("missing_checkpoint.pth"),
            backend_load_timeout_seconds=0.05,
        )

        with self.assertRaises(BackendLoadTimeoutError) as context:
            backend._run_with_load_timeout("test_load", lambda: time.sleep(1.0))

        self.assertIn("test_load", str(context.exception))
        self.assertIn("0.05", str(context.exception))

    def test_grounded_backend_uses_after_manifest_frame_not_before_action_idx(self) -> None:
        class StubGroundedBackend(GroundedSAM2LocationBackend):
            def _load_grounding(self) -> None:
                return None

            def _load_sam2(self) -> None:
                self.sam2_enabled = False
                self.sam2_status = "disabled_for_test"

            def _detect_image(self, image_path: Path, label: str):
                if image_path.name == "03_frame.jpg" and label == "Apple":
                    return [
                        ObjectDetection(
                            label="Apple",
                            score=1.0,
                            center=(30.0, 30.0),
                            coordinate_space="pixel_xy",
                            image_path=str(image_path),
                            xyxy=(20.0, 20.0, 40.0, 40.0),
                        )
                    ]
                return []

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            before_dir = image_dir / "FloorPlan1" / "before"
            after_dir = image_dir / "FloorPlan1" / "after"
            before_dir.mkdir(parents=True)
            after_dir.mkdir(parents=True)

            self._touch_jpeg(before_dir / "01_frame.jpg")
            self._touch_jpeg(after_dir / "01_frame.jpg")
            self._touch_jpeg(after_dir / "03_frame.jpg")

            before_entry = {
                "scene": "FloorPlan1",
                "phase": "before",
                "action_idx": 1,
                "object_id": "Apple|+01.00|+01.00|+00.00",
                "object_type": "Apple",
                "bbox_xyxy": [10, 10, 20, 20],
                "frame_path": "FloorPlan1/before/01_frame.jpg",
                "crop_path": "FloorPlan1/before/01_Apple.jpg",
            }
            after_entry = {
                "scene": "FloorPlan1",
                "phase": "after",
                "action_idx": 3,
                "object_id": "Apple|+01.00|+01.00|+00.00",
                "object_type": "Apple",
                "bbox_xyxy": [25, 25, 35, 35],
                "frame_path": "FloorPlan1/after/03_frame.jpg",
                "crop_path": "FloorPlan1/after/03_Apple.jpg",
            }
            (before_dir / "image_manifest.jsonl").write_text(
                json.dumps(before_entry) + "\n", encoding="utf-8"
            )
            (after_dir / "image_manifest.jsonl").write_text(
                json.dumps(after_entry) + "\n", encoding="utf-8"
            )

            backend = StubGroundedBackend(
                grounding_config=Path("missing_config.py"),
                grounding_checkpoint=Path("missing_checkpoint.pth"),
            )

            class Task:
                scene = "FloorPlan1"
                target = "Apple"

            decision = backend.verify(
                Task(),
                remembered_location="FloorPlan1:Apple|+01.00|+01.00|+00.00@before",
                image_dir=image_dir,
                threshold=0.25,
            )

        self.assertEqual(decision.reason, "nearest_after_detection_normalized_frame_distance")
        self.assertEqual(decision.detections_considered, 1)

    def test_grounded_backend_compares_against_remembered_before_center(self) -> None:
        class StubGroundedBackend(GroundedSAM2LocationBackend):
            def _load_grounding(self) -> None:
                return None

            def _load_sam2(self) -> None:
                self.sam2_enabled = False
                self.sam2_status = "disabled_for_test"

            def _detect_image(self, image_path: Path, label: str):
                if image_path.name == "03_frame.jpg" and label == "Apple":
                    return [
                        ObjectDetection(
                            label="Apple",
                            score=1.0,
                            center=(90.0, 90.0),
                            coordinate_space="pixel_xy",
                            image_path=str(image_path),
                            xyxy=(85.0, 85.0, 95.0, 95.0),
                        )
                    ]
                return []

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            before_dir = image_dir / "FloorPlan1" / "before"
            after_dir = image_dir / "FloorPlan1" / "after"
            before_dir.mkdir(parents=True)
            after_dir.mkdir(parents=True)

            self._touch_jpeg(before_dir / "01_frame.jpg")
            self._touch_jpeg(after_dir / "03_frame.jpg")

            before_entry = {
                "scene": "FloorPlan1",
                "phase": "before",
                "action_idx": 1,
                "object_id": "Apple|+01.00|+01.00|+00.00",
                "object_type": "Apple",
                "bbox_xyxy": [5, 5, 15, 15],
                "frame_path": "FloorPlan1/before/01_frame.jpg",
                "crop_path": "FloorPlan1/before/01_Apple.jpg",
            }
            after_entry = {
                "scene": "FloorPlan1",
                "phase": "after",
                "action_idx": 3,
                "object_id": "Apple|+02.00|+01.00|+02.00",
                "object_type": "Apple",
                "bbox_xyxy": [85, 85, 95, 95],
                "frame_path": "FloorPlan1/after/03_frame.jpg",
                "crop_path": "FloorPlan1/after/03_Apple.jpg",
            }
            (before_dir / "image_manifest.jsonl").write_text(
                json.dumps(before_entry) + "\n", encoding="utf-8"
            )
            (after_dir / "image_manifest.jsonl").write_text(
                json.dumps(after_entry) + "\n", encoding="utf-8"
            )

            backend = StubGroundedBackend(
                grounding_config=Path("missing_config.py"),
                grounding_checkpoint=Path("missing_checkpoint.pth"),
            )

            class Task:
                scene = "FloorPlan1"
                target = "Apple"

            decision = backend.verify(
                Task(),
                remembered_location="FloorPlan1:Apple|+01.00|+01.00|+00.00@before",
                image_dir=image_dir,
                threshold=0.1,
            )

        self.assertTrue(decision.stale)
        self.assertEqual(decision.reason, "nearest_after_detection_normalized_frame_distance")
        self.assertIsNotNone(decision.matched_distance)
        self.assertGreater(decision.matched_distance or 0.0, 0.1)

    def test_normalize_sam2_config_strips_absolute_prefix(self) -> None:
        backend = GroundedSAM2LocationBackend(
            grounding_config=Path("missing_config.py"),
            grounding_checkpoint=Path("missing_checkpoint.pth"),
            sam2_config=Path(
                "/home/jxy/.cache/memoryguard-grounded-sam2-src-unpacked2/sam2-main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
            ),
        )

        normalized = backend._normalize_sam2_config(backend.sam2_config)

        self.assertEqual(normalized, Path("configs/sam2.1/sam2.1_hiera_t.yaml"))

    def test_parse_rearrangement_location_extracts_world_xz(self) -> None:
        parsed = parse_rearrangement_location("FloorPlan1:Apple|+01.00|+01.00|-02.50@before")

        self.assertEqual(parsed.scene, "FloorPlan1")
        self.assertEqual(parsed.object_type, "Apple")
        self.assertEqual(parsed.world_x, 1.0)
        self.assertEqual(parsed.world_z, -2.5)

    def test_crop_filename_proxy_matches_oracle_for_moved_and_stable_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Apple|+03.00|+01.00|+00.00.jpg")
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Book|+02.00|+01.00|+00.00.jpg")

            rows = evaluate_location_verification(
                self.sample_probe(),
                image_dir,
                CropFilenameLocationBackend(),
                budgets=[2],
                location_thresholds=[0.25],
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["oracle_catches"], 1)
        self.assertEqual(row["location_catches"], 1)
        self.assertEqual(row["verified_false_catches"], 0)
        self.assertEqual(row["verified_missed_stale"], 0)
        self.assertEqual(row["verified_catch_precision"], 1.0)
        self.assertEqual(row["verified_catch_recall"], 1.0)
        self.assertEqual(row["verified_oracle_agreement"], 1.0)

    def test_missing_after_detection_is_conservative_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Book|+02.00|+01.00|+00.00.jpg")

            rows = evaluate_location_verification(
                self.sample_probe(),
                image_dir,
                CropFilenameLocationBackend(),
                budgets=[2],
                location_thresholds=[0.25],
            )

        row = rows[0]
        self.assertEqual(row["location_catches"], 1)
        self.assertEqual(row["verified_missed_stale"], 0)

    def test_unverified_stale_cost_matches_rearrangement_semantics(self) -> None:
        probe = {
            "status": "ok",
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                        {"x": 3, "y": 0, "z": 0},
                    ],
                    "before_visible_objects": [
                        {
                            "object_id": "Apple|+01.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        }
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|+03.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "embodied_memory_pilot.ai2thor_grounded_sam2_verifier.VerifyRearrangementRiskPolicy.should_verify",
            return_value=False,
        ):
            image_dir = Path(tmp)
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Apple|+03.00|+01.00|+00.00.jpg")
            rows = evaluate_location_verification(
                probe,
                image_dir,
                CropFilenameLocationBackend(),
                budgets=[1],
                location_thresholds=[0.25],
            )

        row = rows[0]
        self.assertEqual(row["unverified_stale_errors"], 1)
        self.assertGreater(row["oracle_avg_cost"], row["verified_decision_count"])
        self.assertGreater(row["location_avg_cost"], row["verified_decision_count"])

    def test_verifier_checks_retained_memory_location_not_task_default(self) -> None:
        probe = {
            "status": "ok",
            "scenes": [
                {
                    "scene": "FloorPlan1",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                        {"x": 3, "y": 0, "z": 0},
                    ],
                    "before_visible_objects": [
                        {
                            "object_id": "Apple|+01.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        }
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|+01.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        }
                    ],
                },
                {
                    "scene": "FloorPlan2",
                    "status": "ok",
                    "agent_position": {"x": 0, "y": 0, "z": 0},
                    "reachable_positions": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 2, "y": 0, "z": 0},
                        {"x": 3, "y": 0, "z": 0},
                    ],
                    "before_visible_objects": [
                        {
                            "object_id": "Apple|+02.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        }
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|+03.00|+01.00|+00.00",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        }
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Apple|+01.00|+01.00|+00.00.jpg")
            self._touch_jpeg(image_dir / "FloorPlan2" / "after" / "00_Apple|+03.00|+01.00|+00.00.jpg")

            rows = evaluate_location_verification(
                probe,
                image_dir,
                CropFilenameLocationBackend(),
                budgets=[2],
                location_thresholds=[0.25],
            )

        row = rows[0]
        floorplan1_decision = next(
            decision for decision in row["decisions"] if decision["scene"] == "FloorPlan1"
        )
        self.assertEqual(floorplan1_decision["remembered_location"], "FloorPlan2:Apple|+02.00|+01.00|+00.00@before")
        self.assertEqual(row["verified_decision_count"], 2)
        self.assertEqual(row["threshold_unit"], "ai2thor_world_meters")

    def test_manifest_supports_task_specific_location_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            manifest_dir = image_dir / "FloorPlan1" / "after"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "image_manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "scene": "FloorPlan1",
                        "phase": "after",
                        "action_idx": 0,
                        "object_id": "Apple|+03.00|+01.00|+00.00",
                        "object_type": "Apple",
                        "bbox_xyxy": [1, 1, 3, 3],
                        "frame_path": "FloorPlan1/after/00_frame.jpg",
                        "crop_path": "FloorPlan1/after/00_Apple|+03.00|+01.00|+00.00.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_frame.jpg")
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Apple|+03.00|+01.00|+00.00.jpg")

            backend = CropFilenameLocationBackend()
            rows = evaluate_location_verification(
                self.sample_probe(),
                image_dir,
                backend,
                budgets=[1],
                location_thresholds=[0.25],
            )

        self.assertEqual(rows[0]["verified_decision_count"], 1)
        self.assertEqual(rows[0]["threshold_unit"], "ai2thor_world_meters")


class TestOwlVitSmoke(unittest.TestCase):
    def test_owl_vit_case_level_smoke_records_case_rows_without_model_load(self) -> None:
        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import make_owl_vit_case_level_smoke

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / 'images'
            after_dir = image_dir / 'FloorPlan1' / 'after'
            after_dir.mkdir(parents=True)
            frame_path = after_dir / '003_frame.jpg'
            self_image = Image.new('RGB', (8, 8), color=(255, 255, 255))
            self_image.save(frame_path)
            manifest = {
                'scene': 'FloorPlan1',
                'phase': 'after',
                'action_idx': 3,
                'object_id': 'Book|+00.10|+01.00|+00.20',
                'object_type': 'Book',
                'bbox_xyxy': [1, 1, 5, 5],
                'frame_path': 'FloorPlan1/after/003_frame.jpg',
                'crop_path': 'FloorPlan1/after/003_Book.jpg',
            }
            (after_dir / 'image_manifest.jsonl').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
            case_list_path = root / 'cases.json'
            case_list_path.write_text(json.dumps({'cases': [
                {'row_idx': 4, 'scene': 'FloorPlan1', 'seed': 7, 'target': 'Book', 'target_object_id': 'Book|+00.10|+01.00|+00.20', 'failure_type': 'strict_fn', 'visibility_slice': 'visible_after_revisit', 'nav_search_slice': 'not_used'},
                {'row_idx': 5, 'scene': 'FloorPlan1', 'seed': 7, 'target': 'Newspaper', 'target_object_id': 'Newspaper|+00.00|+01.00|+00.00', 'failure_type': 'strict_fn'},
            ]}), encoding='utf-8')

            smoke = make_owl_vit_case_level_smoke(image_dir=image_dir, case_list_path=case_list_path, run_model=False)

        self.assertEqual(smoke['schema'], '0514_owl_vit_case_level_smoke.v1')
        self.assertEqual(smoke['status'], 'ok')
        self.assertEqual(smoke['case_list_case_count'], 2)
        self.assertEqual(smoke['matched_case_rows'], 1)
        self.assertEqual(smoke['unmatched_case_rows'], 1)
        matched = smoke['rows'][0]
        self.assertEqual(matched['source_row_idx'], 4)
        self.assertEqual(matched['target'], 'Book')
        self.assertEqual(matched['matched_image_status'], 'matched')
        self.assertEqual(matched['match_reason'], 'exact_object_id_manifest_match')
        self.assertEqual(matched['detections_returned'], 0)
        unmatched = smoke['rows'][1]
        self.assertEqual(unmatched['matched_image_status'], 'no_manifest_match')

    def test_owl_vit_smoke_returns_blocked_when_transformers_backend_is_unavailable(self) -> None:
        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import make_owl_vit_detector_smoke

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / 'images'
            case_list_path = Path(tmp) / 'case_list_false_negative_book_newspaper.json'
            case_list_path.write_text(json.dumps({'cases': [{'scene': 'FloorPlan1', 'seed': 7, 'target': 'Apple', 'failure_type': 'strict_fn', 'nav_search_slice': 'A'}]}), encoding='utf-8')
            smoke = make_owl_vit_detector_smoke(image_dir=image_dir, case_list_path=case_list_path)

        self.assertEqual(smoke['schema'], '0514_owl_vit_detector_smoke.v1')
        self.assertEqual(smoke['status'], 'blocked')
        self.assertIn(smoke['blocker_type'], {'missing_optional_dependencies', 'network_or_model_unavailable', 'audit_only_no_new_smoke'})
        self.assertEqual(smoke['case_list_case_count'], 1)

class TestScanReferStyleLocalization(unittest.TestCase):
    def test_scanrefer_style_localization_reports_blocked_provenance_only(self) -> None:
        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import make_scanrefer_style_localization_provenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = make_scanrefer_style_localization_provenance(
                image_dir=root / 'images',
                case_list_path=root / 'cases.json',
            )

        self.assertEqual(provenance['schema'], '0514_scanrefer_style_localization_provenance.v1')
        self.assertEqual(provenance['status'], 'blocked')
        self.assertEqual(provenance['blocker_type'], 'no_local_scanrefer_implementation')
        self.assertIn('scanrefer', provenance['missing_components'])
        self.assertIn('AI2-THOR object grounding/localization provenance only', provenance['claim_boundary'])

class TestYoloWorldSmoke(unittest.TestCase):
    def test_yolo_world_smoke_reports_blocked_when_ultralytics_is_missing(self) -> None:
        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import make_yolo_world_case_level_smoke

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / 'images'
            case_list_path = root / 'cases.json'
            case_list_path.write_text(json.dumps({'cases': [{'row_idx': 0, 'scene': 'FloorPlan1', 'seed': 7, 'target': 'Book', 'target_object_id': 'Book|+00.10|+01.00|+00.20', 'failure_type': 'strict_fn'}]}), encoding='utf-8')
            smoke = make_yolo_world_case_level_smoke(image_dir=image_dir, case_list_path=case_list_path, run_model=False)

        self.assertEqual(smoke['schema'], '0514_yolo_world_case_level_smoke.v1')
        self.assertEqual(smoke['status'], 'blocked')
        self.assertIn(smoke['blocker_type'], {'missing_optional_dependencies', 'audit_only_no_new_smoke'})
        self.assertEqual(smoke['case_list_case_count'], 1)
        self.assertEqual(smoke['claim_boundary'], 'YOLO-World blocked dependency artifact over the existing saved Book/Newspaper case-list protocol only; no detector superiority, recall, precision, broad benchmark, live AI2-THOR, navigation, manipulation, ObjectNav, recovery-search, memory-writeback, or policy-performance claim is supported.')

class TestDetectorBaselineAudit(unittest.TestCase):
    def test_make_detector_baseline_audit_reports_missing_owl_vit_and_yolo_world(self) -> None:
        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import make_detector_baseline_audit

        audit = make_detector_baseline_audit(Path('results/ai2thor_live_gsam_targeted_reproduction_post_05822d3/dino_only_pass1/images'), Path('results/ai2thor_live_gsam_targeted_reproduction_post_05822d3/case_list_false_negative_book_newspaper.json'))

        self.assertEqual(audit['status'], 'blocked')
        self.assertEqual(audit['blocker_type'], 'missing_optional_dependencies')
        self.assertIn('OWL-ViT', audit['missing_backends'])
        self.assertIn('YOLO-World', audit['missing_backends'])
        self.assertEqual(audit['claim_boundary'], 'detector coverage audit only; no new detector performance claims')

    def test_make_detector_baseline_audit_uses_existing_case_list(self) -> None:
        from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import make_detector_baseline_audit

        audit = make_detector_baseline_audit(Path('results/ai2thor_live_gsam_targeted_reproduction_post_05822d3/dino_only_pass1/images'), Path('results/ai2thor_live_gsam_targeted_reproduction_post_05822d3/case_list_false_negative_book_newspaper.json'))

        self.assertEqual(audit['case_list_case_count'], 8)
        self.assertGreaterEqual(audit['covered_backends_count'], 1)



if __name__ == "__main__":
    unittest.main()
