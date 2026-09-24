import tempfile
import unittest
from pathlib import Path

from PIL import Image

from embodied_memory_pilot.ai2thor_grounded_sam2_live_bridge import (
    LiveLocationQuery,
    LiveLocationVerifier,
    ReplayBridge,
)
from embodied_memory_pilot.ai2thor_grounded_sam2_verifier import CropFilenameLocationBackend
from embodied_memory_pilot.ai2thor_rearrangement_benchmark import RearrangementTask


class GroundedSAM2LiveBridgeTest(unittest.TestCase):
    def _touch_jpeg(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(path)

    def _write_manifest(self, image_dir: Path) -> None:
        manifest_dir = image_dir / "FloorPlan1" / "after"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        _ = (manifest_dir / "image_manifest.jsonl").write_text(
            (
                '{"scene": "FloorPlan1", "phase": "after", "action_idx": 0, '
                '"object_id": "Apple|+01.00|+01.00|+00.00", "object_type": "Apple", '
                '"bbox_xyxy": [1, 1, 3, 3], "frame_path": "FloorPlan1/after/00_frame.jpg", '
                '"crop_path": "FloorPlan1/after/00_Apple|+01.00|+01.00|+00.00.jpg"}\n'
            ),
            encoding="utf-8",
        )

    def test_live_bridge_wraps_replay_verifier(self) -> None:
        task = RearrangementTask(
            target="Apple",
            true_location="FloorPlan1:Apple|+01.00|+01.00|+00.00@after",
            old_location="FloorPlan1:Apple|+01.00|+01.00|+00.00@before",
            scene="FloorPlan1",
            direct_action_cost=1.0,
            old_action_cost=2.0,
            scene_search_cost=3.0,
            rearranged=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            self._write_manifest(image_dir)
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_frame.jpg")
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Apple|+01.00|+01.00|+00.00.jpg")

            query = LiveLocationQuery(
                scene_id="FloorPlan1",
                object_type="Apple",
                remembered_location=task.true_location,
                frame_path=image_dir / "FloorPlan1" / "after" / "00_frame.jpg",
                image_dir=image_dir,
                threshold=0.25,
                frame_id="frame-000",
                timestamp_ms=1234,
            )
            verifier = LiveLocationVerifier(CropFilenameLocationBackend())
            bridge = ReplayBridge(verifier)
            record = bridge.verify(task, query, policy_name="passive")

            self.assertFalse(record.decision.stale)
        self.assertEqual(record.policy_name, "passive")
        self.assertEqual(record.query.scene_id, "FloorPlan1")
        self.assertEqual(record.query.object_type, "Apple")
        self.assertEqual(record.backend_name, "crop_filename_proxy")
        self.assertEqual(record.policy_action, "accept_memory")
        self.assertEqual(record.claim_boundary, "replay/oracle live-style adapter only")

    def test_bridge_outputs_audit_rows(self) -> None:
        task = RearrangementTask(
            target="Apple",
            true_location="FloorPlan1:Apple|+01.00|+01.00|+00.00@after",
            old_location="FloorPlan1:Apple|+01.00|+01.00|+00.00@before",
            scene="FloorPlan1",
            direct_action_cost=1.0,
            old_action_cost=2.0,
            scene_search_cost=3.0,
            rearranged=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            self._write_manifest(image_dir)
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_frame.jpg")
            self._touch_jpeg(image_dir / "FloorPlan1" / "after" / "00_Apple|+01.00|+01.00|+00.00.jpg")

            query = LiveLocationQuery(
                scene_id="FloorPlan1",
                object_type="Apple",
                remembered_location=task.true_location,
                frame_path=image_dir / "FloorPlan1" / "after" / "00_frame.jpg",
                image_dir=image_dir,
                threshold=0.25,
                frame_id="frame-000",
                timestamp_ms=1234,
            )
            verifier = LiveLocationVerifier(CropFilenameLocationBackend())
            bridge = ReplayBridge(verifier)
            record = bridge.verify(task, query)
            out_dir = Path(tmp) / "out"
            ReplayBridge.write_outputs([record], out_dir)

            self.assertTrue((out_dir / "live_bridge.json").exists())
            self.assertTrue((out_dir / "live_bridge.csv").exists())
            self.assertTrue((out_dir / "README.md").exists())
            csv_text = (out_dir / "live_bridge.csv").read_text(encoding="utf-8")
            self.assertIn("trace_id", csv_text)
            self.assertIn("policy_action", csv_text)
            self.assertIn("backend_name", csv_text)
            self.assertIn("claim_boundary", csv_text)


if __name__ == "__main__":
    _ = unittest.main()
