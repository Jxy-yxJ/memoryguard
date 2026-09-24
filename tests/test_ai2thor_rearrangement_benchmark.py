import unittest
import tempfile
from pathlib import Path

import numpy as np

from embodied_memory_pilot.ai2thor_rearrangement_benchmark import (
    _save_object_images,
    build_rearrangement_memory_items,
    build_rearrangement_tasks,
    evaluate_policy_rearrangement,
    summarize_rearrangement_probe,
)
from embodied_memory_pilot.pilot import SaliencePolicy, TaskConditionedPolicy


class AI2ThorRearrangementBenchmarkTest(unittest.TestCase):
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
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 1, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Book|1",
                            "object_type": "Book",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                    "after_visible_objects": [
                        {
                            "object_id": "Apple|1",
                            "object_type": "Apple",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 3, "y": 1, "z": 0},
                        },
                        {
                            "object_id": "Book|1",
                            "object_type": "Book",
                            "pickupable": True,
                            "receptacle": False,
                            "position": {"x": 2, "y": 1, "z": 0},
                        },
                    ],
                }
            ],
        }

    def test_summary_counts_moved_objects(self) -> None:
        summary = summarize_rearrangement_probe(self.sample_probe())

        self.assertEqual(summary["ok_scenes"], 1)
        self.assertEqual(summary["moved_objects"], 1)
        self.assertEqual(summary["stable_objects"], 1)

    def test_pairs_moved_objects_when_ai2thor_object_id_changes_with_position(self) -> None:
        probe = self.sample_probe()
        probe["scenes"][0]["before_visible_objects"] = [
            {
                "object_id": "Book|+00.15|+01.10|+00.62",
                "object_type": "Book",
                "pickupable": True,
                "receptacle": False,
                "position": {"x": 0.15, "y": 1.1, "z": 0.62},
            }
        ]
        probe["scenes"][0]["after_visible_objects"] = [
            {
                "object_id": "Book|-00.34|+01.11|-00.51",
                "object_type": "Book",
                "pickupable": True,
                "receptacle": False,
                "position": {"x": -0.34, "y": 1.11, "z": -0.51},
            }
        ]

        summary = summarize_rearrangement_probe(probe)
        tasks = build_rearrangement_tasks(probe)

        self.assertEqual(summary["paired_objects"], 1)
        self.assertEqual(summary["moved_objects"], 1)
        self.assertEqual(tasks[0].target, "Book")

    def test_memory_comes_from_before_and_tasks_from_after(self) -> None:
        probe = self.sample_probe()

        memory_items = build_rearrangement_memory_items(probe)
        tasks = build_rearrangement_tasks(probe)

        self.assertEqual(memory_items[0].location, "FloorPlan1:Apple|1@before")
        self.assertEqual(tasks[0].true_location, "FloorPlan1:Apple|1@after")
        self.assertEqual(tasks[0].target, "Apple")
        self.assertGreater(tasks[0].direct_action_cost, 1.0)

    def test_rearranged_object_creates_stale_error_for_salience(self) -> None:
        row = evaluate_policy_rearrangement(self.sample_probe(), policy=SaliencePolicy(), budget=2)

        self.assertEqual(row["rearranged_tasks"], 1)
        self.assertGreaterEqual(row["stale_errors"], 1)
        self.assertLess(row["completion_rate"], 1.0)

    def test_task_conditioned_can_reduce_stale_error_under_tight_budget(self) -> None:
        probe = self.sample_probe()
        salience = evaluate_policy_rearrangement(probe, policy=SaliencePolicy(), budget=1)
        task_conditioned = evaluate_policy_rearrangement(probe, policy=TaskConditionedPolicy(), budget=1)

        self.assertLessEqual(task_conditioned["stale_errors"], salience["stale_errors"])

    def test_stable_object_memory_counts_as_valid_after_rearrangement(self) -> None:
        probe = self.sample_probe()
        probe["scenes"][0]["before_visible_objects"] = [
            {
                "object_id": "Book|1",
                "object_type": "Book",
                "pickupable": True,
                "receptacle": False,
                "position": {"x": 2, "y": 1, "z": 0},
            }
        ]
        probe["scenes"][0]["after_visible_objects"] = [
            {
                "object_id": "Book|1",
                "object_type": "Book",
                "pickupable": True,
                "receptacle": False,
                "position": {"x": 2, "y": 1, "z": 0},
            }
        ]

        row = evaluate_policy_rearrangement(probe, policy=SaliencePolicy(), budget=1, target_objects=["Book"])

        self.assertEqual(row["rearrangement_tasks"], 1)
        self.assertEqual(row["moved_tasks"], 0)
        self.assertEqual(row["stale_errors"], 0)
        self.assertEqual(row["completion_rate"], 1.0)

    def test_save_object_images_writes_manifest_for_location_verifier(self) -> None:
        class Event:
            frame = np.zeros((6, 6, 3), dtype=np.uint8)
            instance_masks = {
                "Apple|+01.00|+01.00|+00.00": np.array(
                    [
                        [False, False, False, False, False, False],
                        [False, True, True, False, False, False],
                        [False, True, True, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                    ]
                )
            }
            metadata = {
                "objects": [
                    {
                        "objectId": "Apple|+01.00|+01.00|+00.00",
                        "objectType": "Apple",
                        "visible": True,
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            _save_object_images(Event(), image_dir, "FloorPlan1", "before", 0)
            manifest = image_dir / "FloorPlan1" / "before" / "image_manifest.jsonl"

            self.assertTrue(manifest.exists())
            text = manifest.read_text(encoding="utf-8")
            self.assertIn('"object_id": "Apple|+01.00|+01.00|+00.00"', text)
            self.assertIn('"bbox_xyxy"', text)
            self.assertTrue((image_dir / "FloorPlan1" / "before" / "00_frame.jpg").exists())


if __name__ == "__main__":
    unittest.main()
