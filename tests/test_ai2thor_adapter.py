import unittest
from types import SimpleNamespace

from embodied_memory_pilot.ai2thor_adapter import (
    ai2thor_capability,
    merge_visible_objects,
    object_memory_summary,
    render_readme,
    summarize_scene_probe,
    visible_objects_from_event,
)


class AI2ThorAdapterTest(unittest.TestCase):
    def test_capability_report_has_install_hint(self) -> None:
        report = ai2thor_capability()
        self.assertIn("install", report.install_hint.lower())
        self.assertTrue(report.python)

    def test_visible_objects_from_event_filters_invisible(self) -> None:
        event = SimpleNamespace(
            metadata={
                "objects": [
                    {
                        "objectId": "Mug|1",
                        "objectType": "Mug",
                        "name": "mug",
                        "visible": True,
                        "pickupable": True,
                        "openable": False,
                        "isOpen": False,
                        "receptacle": False,
                    },
                    {"objectId": "Knife|1", "objectType": "Knife", "name": "knife", "visible": False},
                ]
            }
        )

        visible = visible_objects_from_event(event)

        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["object_type"], "Mug")
        self.assertTrue(visible[0]["pickupable"])
        self.assertFalse(visible[0]["openable"])
        self.assertFalse(visible[0]["isOpen"])
        self.assertFalse(visible[0]["receptacle"])

    def test_render_readme_includes_platform_override(self) -> None:
        result = {
            "status": "blocked",
            "scene": "FloorPlan1",
            "platform_name": "CloudRendering",
            "build_base_url": "https://mirror.example/ai2-thor-public/",
            "capability": {
                "python": "3.11",
                "platform": "Linux",
                "blocker": "missing build",
                "install_hint": "install ai2thor",
            },
        }

        readme = render_readme(result)

        self.assertIn("CloudRendering", readme)
        self.assertIn("https://mirror.example/ai2-thor-public/", readme)

    def test_object_memory_summary_counts_target_objects(self) -> None:
        visible = [
            {"object_type": "Mug", "pickupable": True, "receptacle": False},
            {"object_type": "Cabinet", "pickupable": False, "receptacle": True},
            {"object_type": "Mug", "pickupable": True, "receptacle": False},
        ]

        summary = object_memory_summary(visible, target_object_types=("Mug", "Knife"))

        self.assertEqual(summary["visible_count"], 3)
        self.assertEqual(summary["pickupable_count"], 2)
        self.assertEqual(summary["receptacle_count"], 1)
        self.assertEqual(summary["target_object_counts"], {"Mug": 2, "Knife": 0})

    def test_merge_visible_objects_deduplicates_by_object_id(self) -> None:
        frames = [
            [
                {"object_id": "Mug|1", "object_type": "Mug"},
                {"object_id": "Knife|1", "object_type": "Knife"},
            ],
            [
                {"object_id": "Mug|1", "object_type": "Mug"},
                {"object_id": "Plate|1", "object_type": "Plate"},
            ],
        ]

        merged = merge_visible_objects(frames)

        self.assertEqual([obj["object_id"] for obj in merged], ["Mug|1", "Knife|1", "Plate|1"])

    def test_summarize_scene_probe_aggregates_ok_scenes(self) -> None:
        scenes = [
            {
                "status": "ok",
                "scene": "FloorPlan1",
                "memory_summary": {
                    "visible_count": 3,
                    "pickupable_count": 2,
                    "receptacle_count": 1,
                    "target_object_counts": {"Mug": 2, "Knife": 0},
                },
            },
            {
                "status": "blocked",
                "scene": "FloorPlan2",
                "memory_summary": {
                    "visible_count": 0,
                    "pickupable_count": 0,
                    "receptacle_count": 0,
                    "target_object_counts": {"Mug": 0, "Knife": 0},
                },
            },
        ]

        summary = summarize_scene_probe(scenes)

        self.assertEqual(summary["scenes"], 2)
        self.assertEqual(summary["ok_scenes"], 1)
        self.assertEqual(summary["blocked_scenes"], 1)
        self.assertEqual(summary["total_visible"], 3)
        self.assertEqual(summary["target_object_totals"], {"Mug": 2, "Knife": 0})


if __name__ == "__main__":
    unittest.main()
