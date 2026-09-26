from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SCENE = "FloorPlan1"
DEFAULT_OBJECTS = ("Mug", "Apple", "Knife", "RemoteControl", "Book")
DEFAULT_PROBE_ACTIONS = ("Pass", "RotateRight", "RotateRight", "RotateRight", "RotateRight")
RICH_BEFORE_PROBE_ACTIONS = (
    "Pass",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "LookDown",
    "Pass",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "LookDown",
    "Pass",
    "RotateRight",
    "RotateRight",
    "RotateRight",
    "RotateRight",
)


@dataclass(frozen=True)
class CapabilityReport:
    available: bool
    python: str
    platform: str
    ai2thor_version: str | None
    blocker: str | None
    install_hint: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "python": self.python,
            "platform": self.platform,
            "ai2thor_version": self.ai2thor_version,
            "blocker": self.blocker,
            "install_hint": self.install_hint,
        }


def ai2thor_capability() -> CapabilityReport:
    spec = importlib.util.find_spec("ai2thor")
    python_version = sys.version.split()[0]
    install_hint = (
        "Create a Python 3.10/3.11 environment and install ai2thor there, e.g. "
        "`conda create -n memoryguard-ai2thor python=3.11 -y` then "
        "`conda activate memoryguard-ai2thor` and `pip install ai2thor`."
    )

    if spec is None:
        return CapabilityReport(
            available=False,
            python=python_version,
            platform=platform.platform(),
            ai2thor_version=None,
            blocker="ai2thor is not installed in the active Python environment.",
            install_hint=install_hint,
        )

    try:
        import ai2thor  # type: ignore

        version = getattr(ai2thor, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - depends on optional external package
        return CapabilityReport(
            available=False,
            python=python_version,
            platform=platform.platform(),
            ai2thor_version=None,
            blocker=f"ai2thor import failed: {exc}",
            install_hint=install_hint,
        )

    return CapabilityReport(
        available=True,
        python=python_version,
        platform=platform.platform(),
        ai2thor_version=str(version),
        blocker=None,
        install_hint=install_hint,
    )


def visible_objects_from_event(event: Any) -> List[Dict[str, Any]]:
    metadata = getattr(event, "metadata", {}) or {}
    objects = metadata.get("objects", [])
    visible: List[Dict[str, Any]] = []
    for obj in objects:
        if not obj.get("visible", False):
            continue
        visible.append(
            {
                "object_id": obj.get("objectId"),
                "object_type": obj.get("objectType"),
                "name": obj.get("name"),
                "position": obj.get("position"),
                "pickupable": obj.get("pickupable"),
                "openable": obj.get("openable"),
                "isOpen": obj.get("isOpen"),
                "receptacle": obj.get("receptacle"),
            }
        )
    return visible


def merge_visible_objects(frames: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        for obj in frame:
            object_id = str(obj.get("object_id") or obj.get("name") or obj.get("object_type"))
            if object_id not in merged:
                merged[object_id] = obj
    return list(merged.values())


def object_memory_summary(
    visible_objects: List[Dict[str, Any]],
    target_object_types: tuple[str, ...] = DEFAULT_OBJECTS,
) -> Dict[str, Any]:
    target_counts = {object_type: 0 for object_type in target_object_types}
    pickupable_count = 0
    receptacle_count = 0
    object_types: Dict[str, int] = {}

    for obj in visible_objects:
        object_type = str(obj.get("object_type") or "Unknown")
        object_types[object_type] = object_types.get(object_type, 0) + 1
        if bool(obj.get("pickupable")):
            pickupable_count += 1
        if bool(obj.get("receptacle")):
            receptacle_count += 1
        if object_type in target_counts:
            target_counts[object_type] += 1

    return {
        "visible_count": len(visible_objects),
        "pickupable_count": pickupable_count,
        "receptacle_count": receptacle_count,
        "unique_object_types": len(object_types),
        "object_type_counts": object_types,
        "target_object_counts": target_counts,
    }


def summarize_scene_probe(scene_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_totals: Dict[str, int] = {}
    total_visible = 0
    total_pickupable = 0
    total_receptacles = 0
    ok_scenes = 0

    for result in scene_results:
        if result.get("status") == "ok":
            ok_scenes += 1
        memory_summary = result.get("memory_summary", {})
        total_visible += int(memory_summary.get("visible_count", 0))
        total_pickupable += int(memory_summary.get("pickupable_count", 0))
        total_receptacles += int(memory_summary.get("receptacle_count", 0))
        for object_type, count in memory_summary.get("target_object_counts", {}).items():
            target_totals[object_type] = target_totals.get(object_type, 0) + int(count)

    scenes = len(scene_results)
    return {
        "scenes": scenes,
        "ok_scenes": ok_scenes,
        "blocked_scenes": scenes - ok_scenes,
        "total_visible": total_visible,
        "avg_visible_per_ok_scene": round(total_visible / ok_scenes, 4) if ok_scenes else 0.0,
        "total_pickupable": total_pickupable,
        "total_receptacles": total_receptacles,
        "target_object_totals": target_totals,
    }


def platform_class(platform_name: str | None) -> Any:
    if not platform_name:
        return None

    from ai2thor import platform as ai2thor_platform  # type: ignore  # pragma: no cover

    try:
        return getattr(ai2thor_platform, platform_name)
    except AttributeError as exc:  # pragma: no cover - depends on optional external package
        raise ValueError(f"Unknown AI2-THOR platform: {platform_name}") from exc


def configure_build_mirror(build_base_url: str | None) -> None:
    if not build_base_url:
        return

    import ai2thor.build  # type: ignore  # pragma: no cover

    normalized = build_base_url.rstrip("/") + "/"
    ai2thor.build.base_url = normalized


def run_smoke(
    scene: str,
    out_dir: Path,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = ("Pass",),
) -> Dict[str, Any]:
    capability = ai2thor_capability()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not capability.available:
        result = {
            "status": "blocked",
            "capability": capability.as_dict(),
            "scene": scene,
            "visible_objects": [],
        }
        write_result(result, out_dir)
        return result

    from ai2thor.controller import Controller  # type: ignore  # pragma: no cover

    try:
        configure_build_mirror(build_base_url)
        controller_kwargs: Dict[str, Any] = {"scene": scene, "width": width, "height": height}
        selected_platform = platform_class(platform_name)
        if selected_platform is not None:
            controller_kwargs["platform"] = selected_platform
        controller = Controller(**controller_kwargs)
    except Exception as exc:  # pragma: no cover - depends on Unity build availability
        result = {
            "status": "blocked",
            "capability": capability.as_dict(),
            "scene": scene,
            "platform_name": platform_name,
            "build_base_url": build_base_url,
            "blocker": f"AI2-THOR Controller failed to start: {exc}",
            "visible_objects": [],
        }
        write_result(result, out_dir)
        return result

    try:
        frames: List[List[Dict[str, Any]]] = []
        action_status: List[Dict[str, Any]] = []
        for action in actions:
            event = controller.step(action=action)
            frames.append(visible_objects_from_event(event))
            metadata = getattr(event, "metadata", {}) or {}
            action_status.append(
                {
                    "action": action,
                    "success": bool(metadata.get("lastActionSuccess", True)),
                    "error_message": metadata.get("errorMessage", ""),
                }
            )
        visible = merge_visible_objects(frames)
        memory_summary = object_memory_summary(visible)
        result = {
            "status": "ok",
            "capability": capability.as_dict(),
            "scene": scene,
            "platform_name": platform_name,
            "build_base_url": build_base_url,
            "visible_count": len(visible),
            "visible_objects": visible,
            "actions": list(actions),
            "action_status": action_status,
            "memory_summary": memory_summary,
            "target_object_types": list(DEFAULT_OBJECTS),
        }
    finally:
        controller.stop()

    write_result(result, out_dir)
    return result


def run_scene_probe(
    scenes: List[str],
    out_dir: Path,
    width: int = 300,
    height: int = 300,
    platform_name: str | None = None,
    build_base_url: str | None = None,
    actions: tuple[str, ...] = DEFAULT_PROBE_ACTIONS,
) -> Dict[str, Any]:
    scene_results = [
        run_smoke(
            scene=scene,
            out_dir=out_dir / "scenes" / scene,
            width=width,
            height=height,
            platform_name=platform_name,
            build_base_url=build_base_url,
            actions=actions,
        )
        for scene in scenes
    ]
    result = {
        "status": "ok" if all(scene.get("status") == "ok" for scene in scene_results) else "partial",
        "scenes": scene_results,
        "summary": summarize_scene_probe(scene_results),
    }
    write_scene_probe_result(result, out_dir)
    return result


def write_result(result: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ai2thor_capability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(result), encoding="utf-8")


def write_scene_probe_result(result: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ai2thor_scene_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "ai2thor_scene_probe_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "scene",
            "status",
            "visible_count",
            "pickupable_count",
            "receptacle_count",
            "unique_object_types",
            "target_object_counts",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scene_result in result["scenes"]:
            memory_summary = scene_result.get("memory_summary", {})
            writer.writerow(
                {
                    "scene": scene_result.get("scene"),
                    "status": scene_result.get("status"),
                    "visible_count": memory_summary.get("visible_count", 0),
                    "pickupable_count": memory_summary.get("pickupable_count", 0),
                    "receptacle_count": memory_summary.get("receptacle_count", 0),
                    "unique_object_types": memory_summary.get("unique_object_types", 0),
                    "target_object_counts": json.dumps(
                        memory_summary.get("target_object_counts", {}),
                        sort_keys=True,
                    ),
                }
            )
    (out_dir / "README.md").write_text(render_scene_probe_readme(result), encoding="utf-8")


def render_readme(result: Dict[str, Any]) -> str:
    capability = result["capability"]
    lines = [
        "# AI2-THOR Adapter Status",
        "",
        f"Status: `{result['status']}`",
        f"Scene: `{result['scene']}`",
        f"Platform override: `{result.get('platform_name') or 'default'}`",
        f"Build base URL: `{result.get('build_base_url') or 'default'}`",
        f"Python: `{capability['python']}`",
        f"Platform: `{capability['platform']}`",
        "",
    ]
    if result["status"] == "blocked":
        blocker = result.get("blocker") or capability["blocker"]
        lines.extend(
            [
                "## Blocker",
                "",
                blocker,
                "",
                "## Install Hint",
                "",
                capability["install_hint"],
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Visible objects: `{result.get('visible_count', 0)}`",
                "",
                "The adapter can now be extended into a full memory-policy pilot over AI2-THOR scenes.",
                "",
            ]
        )
    return "\n".join(lines)


def render_scene_probe_readme(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    return "\n".join(
        [
            "# AI2-THOR Scene Probe",
            "",
            f"Status: `{result['status']}`",
            f"Scenes: `{summary['scenes']}`",
            f"OK scenes: `{summary['ok_scenes']}`",
            f"Blocked scenes: `{summary['blocked_scenes']}`",
            f"Total visible objects: `{summary['total_visible']}`",
            f"Average visible objects per OK scene: `{summary['avg_visible_per_ok_scene']}`",
            f"Target object totals: `{json.dumps(summary['target_object_totals'], sort_keys=True)}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR adapter for embodied memory maintenance experiments.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ai2thor_smoke"))
    parser.add_argument("--check-only", action="store_true", help="only write dependency capability report")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument(
        "--probe-scenes",
        nargs="+",
        default=None,
        help="run a lightweight scene probe over one or more AI2-THOR scenes",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=None,
        help="AI2-THOR actions to run per scene before aggregating visible objects",
    )
    parser.add_argument(
        "--platform",
        dest="platform_name",
        default=None,
        help="optional AI2-THOR platform class, e.g. CloudRendering for Linux headless",
    )
    parser.add_argument(
        "--build-base-url",
        default=None,
        help=(
            "optional AI2-THOR build mirror root. It must contain a builds/ directory, "
            "e.g. https://mirror.example/ai2-thor-public/"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_scenes:
        result = run_scene_probe(
            scenes=args.probe_scenes,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            platform_name=args.platform_name,
            build_base_url=args.build_base_url,
            actions=tuple(args.actions or DEFAULT_PROBE_ACTIONS),
        )
    elif args.check_only:
        capability = ai2thor_capability()
        result = {
            "status": "ok" if capability.available else "blocked",
            "capability": capability.as_dict(),
            "scene": args.scene,
            "platform_name": args.platform_name,
            "build_base_url": args.build_base_url,
            "visible_objects": [],
        }
        write_result(result, args.out_dir)
    else:
        result = run_smoke(
            scene=args.scene,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            platform_name=args.platform_name,
            build_base_url=args.build_base_url,
            actions=tuple(args.actions or ("Pass",)),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
