"""Render watchable simulation demo videos for MemoryGuard cases.

Each video is a two-panel composition:
  left  - first-person RGB frame from AI2-THOR
  right - top-down map with the remembered (stale) location, the true object location,
          the agent position/heading, and the agent trail
plus a header caption strip, a footer status strip, an intro card, and an outcome end card.

Modes:
  active  - verify at the stale location, refresh memory, navigate to the refreshed location, honest pickup
  passive - navigate to the stale remembered location, honest pickup attempted there (stale-memory baseline)

Cases are read from a recorded active artifact (for the row's detected positions/decisions).

Usage:
    python scripts/make_demo_video.py --row-source <artifact.json> --case Scene:Target:Seed \
        --mode active --out-dir results/0514_demo_videos --panel 600 --fps 8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from embodied_memory_pilot.ai2thor_live_gsam_closed_loop import (  # noqa: E402
    _alternate_reachable_poses,
    _attempt_pickup_honest,
    _reachable_waypoint_route,
    _stepwise_revisit_to_position,
)
from embodied_memory_pilot.ai2thor_live_maintenance import (  # noqa: E402
    _make_controller_kwargs,
    _metadata,
    _reachable_positions,
    nearest_revisit_position,
)
from embodied_memory_pilot.ai2thor_navigation_smoke import _agent_position  # noqa: E402

Json = dict[str, Any]

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL_BG = (245, 245, 245)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)


class RecordingController:
    """Wraps an AI2-THOR controller and records frames plus agent pose per step."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller
        self.frames: list[Any] = []
        self.poses: list[tuple[float, float, float]] = []

    def step(self, action: str, **kwargs: object) -> object:
        event = self._controller.step(action, **kwargs)
        frame = getattr(event, "frame", None)
        metadata = getattr(event, "metadata", None)
        if frame is not None and isinstance(metadata, Mapping):
            agent = metadata.get("agent")
            agent_map = agent if isinstance(agent, Mapping) else {}
            position = agent_map.get("position")
            rotation = agent_map.get("rotation")
            if isinstance(position, Mapping) and isinstance(rotation, Mapping):
                self.poses.append(
                    (float(position.get("x", 0.0)), float(position.get("z", 0.0)), float(rotation.get("y", 0.0)))
                )
                self.frames.append(frame)
        return event

    def stop(self) -> None:
        self._controller.stop()


def _load_row(row_source: Path, scene: str, target: str, seed: int) -> Json:
    data = json.loads(row_source.read_text(encoding="utf-8"))
    for row in data.get("rows", []):
        if row.get("scene") == scene and row.get("target") == target and int(row.get("seed", -1)) == seed:
            return cast(Json, row)
    raise ValueError(f"case not found in artifact: {scene}/{target}/{seed}")


def _bounds(reachable: Sequence[Mapping[str, float]], points: Sequence[Mapping[str, float]]) -> tuple[float, float, float, float]:
    xs = [float(p["x"]) for p in [*reachable, *points]]
    zs = [float(p["z"]) for p in [*reachable, *points]]
    return min(xs), max(xs), min(zs), max(zs)


def _world_to_px(
    x: float, z: float, bounds: tuple[float, float, float, float], size: int, pad: int = 34
) -> tuple[int, int]:
    minx, maxx, minz, maxz = bounds
    span = max(maxx - minx, maxz - minz, 1e-6)
    scale = (size - 2 * pad) / span
    px = pad + (x - minx) * scale + (size - 2 * pad - (maxx - minx) * scale) / 2
    py = size - pad - (z - minz) * scale - (size - 2 * pad - (maxz - minz) * scale) / 2
    return int(px), int(py)


def _draw_map(
    size: int,
    *,
    reachable: Sequence[Mapping[str, float]],
    stale_pos: Mapping[str, float],
    true_pos: Mapping[str, float],
    trail: Sequence[tuple[float, float, float]],
    step_index: int,
) -> Image.Image:
    image = Image.new("RGB", (size, size), PANEL_BG)
    draw = ImageDraw.Draw(image)
    bounds = _bounds(reachable, [stale_pos, true_pos])
    for pose in reachable:
        px, py = _world_to_px(float(pose["x"]), float(pose["z"]), bounds, size)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(205, 205, 205))
    if len(trail) >= 2:
        points = [_world_to_px(x, z, bounds, size) for x, z, _yaw in trail[: step_index + 1]]
        draw.line(points, fill=(60, 110, 220), width=3)
    sx, sy = _world_to_px(float(stale_pos["x"]), float(stale_pos["z"]), bounds, size)
    tx, ty = _world_to_px(float(true_pos["x"]), float(true_pos["z"]), bounds, size)
    draw.line((sx - 9, sy - 9, sx + 9, sy + 9), fill=(200, 30, 30), width=5)
    draw.line((sx - 9, sy + 9, sx + 9, sy - 9), fill=(200, 30, 30), width=5)
    draw.ellipse((tx - 9, ty - 9, tx + 9, ty + 9), outline=(20, 140, 20), width=5)
    if trail:
        ax, az, yaw = trail[min(step_index, len(trail) - 1)]
        px, py = _world_to_px(ax, az, bounds, size)
        direction = (math.sin(math.radians(yaw)), -math.cos(math.radians(yaw)))
        tip = (px + direction[0] * 16, py + direction[1] * 16)
        left = (px - direction[1] * 9 - direction[0] * 6, py + direction[0] * 9 - direction[1] * 6)
        right = (px + direction[1] * 9 - direction[0] * 6, py - direction[0] * 9 - direction[1] * 6)
        draw.polygon([tip, left, right], fill=(20, 60, 200))
    label = _font(15)
    draw.text((sx + 12, sy - 8), "remembered (stale)", fill=(180, 20, 20), font=label)
    draw.text((tx + 12, ty - 8), "actual object", fill=(20, 120, 20), font=label)
    draw.text((10, 8), "top-down map", fill=(90, 90, 90), font=_font(14, bold=True))
    return image


def _strip(width: int, height: int, text: str, *, bg: tuple[int, int, int], fg: tuple[int, int, int], size: int = 22) -> Image.Image:
    strip = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(strip)
    draw.text((14, max(4, (height - size) // 2)), text, fill=fg, font=_font(size, bold=True))
    return strip


def _card(width: int, height: int, lines: Sequence[str], *, bg: tuple[int, int, int], fg: tuple[int, int, int]) -> Image.Image:
    card = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(card)
    y = height // 2 - len(lines) * 22
    for index, line in enumerate(lines):
        font = _font(30 if index == 0 else 20, bold=index == 0)
        draw.text((30, y), line, fill=fg, font=font)
        y += 52 if index == 0 else 34
    return card


def _compose(
    frame: Any,
    map_image: Image.Image,
    header: str,
    footer: str,
    panel: int,
    header_h: int = 44,
    footer_h: int = 34,
) -> Image.Image:
    first_person = Image.fromarray(frame).resize((panel, panel))
    width = panel * 2
    canvas = Image.new("RGB", (width, panel + header_h + footer_h), WHITE)
    canvas.paste(_strip(width, header_h, header, bg=(25, 25, 25), fg=WHITE), (0, 0))
    canvas.paste(first_person, (0, header_h))
    canvas.paste(map_image, (panel, header_h))
    canvas.paste(_strip(width, footer_h, footer, bg=(238, 238, 238), fg=(40, 40, 40), size=18), (0, header_h + panel))
    return canvas


def render_case(
    *,
    scene: str,
    target: str,
    seed: int,
    row: Json,
    mode: str,
    out_dir: Path,
    panel: int,
    fps: int,
) -> Path:
    frames_dir = out_dir / f"{scene}_{target}_{seed}_{mode}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    controller = RecordingController(
        __import__("ai2thor.controller", fromlist=["Controller"]).Controller(
            **_make_controller_kwargs(scene, panel, panel, "CloudRendering", capture_images=False)
        )
    )
    phases: list[Json] = []
    outcome = ""
    try:
        spawn_event = controller.step("InitialRandomSpawn", randomSeed=seed, forceVisible=True, numPlacementAttempts=5)
        spawn_meta = _metadata(spawn_event)
        after_objects = [dict(obj) for obj in spawn_meta.get("objects", []) if isinstance(obj, Mapping)]
        target_obj = next((obj for obj in after_objects if obj.get("objectType") == target and obj.get("pickupable")), None)
        if target_obj is None:
            raise ValueError(f"target not spawned: {scene}/{target}/{seed}")
        target_pos = {key: float(cast(float, target_obj["position"].get(key, 0.0))) for key in ("x", "y", "z")}
        reachable = _reachable_positions(
            cast(Sequence[object], _metadata(controller.step("GetReachablePositions")).get("actionReturn", []))
        )
        stale_pos = cast(Mapping[str, float], row.get("memory_old_position") or target_pos)
        refreshed_pos = cast(Mapping[str, float], row.get("memory_new_position") or target_pos)
        decision_stale = bool(row.get("decision_stale"))

        def open_phase(name: str, header: str) -> None:
            phases.append({"name": name, "header": header, "start": len(controller.frames), "end": len(controller.frames)})

        def close_phase() -> None:
            if phases:
                phases[-1]["end"] = max(int(phases[-1]["end"]), len(controller.frames) + fps)

        open_phase("spawn", f"1/5 Environment change: the {target} was moved")
        goal_stale = nearest_revisit_position(cast(dict[str, float], stale_pos), reachable)
        start = _agent_position(spawn_meta)
        close_phase()
        open_phase("stale", "2/5 Memory says the object was here (red X)")
        waypoints = _reachable_waypoint_route(start, goal_stale, reachable)
        _stepwise_revisit_to_position(controller, start_position=start, target_position=goal_stale, waypoints=waypoints)
        close_phase()

        current = _agent_position(_metadata(controller.step("Pass")))
        if mode == "active":
            open_phase("verify", "3/5 GSAM verifies memory: STALE" if decision_stale else "3/5 GSAM verifies memory: fresh")
            _open_marker = None  # no-op placeholder to keep the phase visible
            controller.step("Pass")
            close_phase()
            goal_refreshed = nearest_revisit_position(cast(dict[str, float], refreshed_pos), reachable)
            open_phase("refresh", "4/5 Navigate to the refreshed location (green circle)")
            waypoints = _reachable_waypoint_route(current, goal_refreshed, reachable)
            _stepwise_revisit_to_position(controller, start_position=current, target_position=goal_refreshed, waypoints=waypoints)
            close_phase()
            open_phase("pickup", "5/5 Honest pickup at the refreshed location")
            used_fallback = False
            _fields, success, _failure = _attempt_pickup_honest(
                controller, object_id=str(target_obj["objectId"]), target_position=target_pos, action="PickupObject"
            )
            if not success:
                for candidate in _alternate_reachable_poses(target_pos, reachable, 3, goal_refreshed):
                    current = _agent_position(_metadata(controller.step("Pass")))
                    waypoints = _reachable_waypoint_route(current, candidate, reachable)
                    _stepwise_revisit_to_position(
                        controller, start_position=current, target_position=candidate, waypoints=waypoints
                    )
                    _fields, success, _failure = _attempt_pickup_honest(
                        controller, object_id=str(target_obj["objectId"]), target_position=target_pos, action="PickupObject"
                    )
                    if success:
                        used_fallback = True
                        break
            phases[-1]["header"] = "5/5 Honest pickup" + (" (approach fallback)" if used_fallback else "")
            outcome = "SUCCESS: object picked up" if success else "FAILED"
            close_phase()
        else:
            open_phase("passive", "3/3 Passive baseline: act on stale memory only")
            _fields, success, _failure = _attempt_pickup_honest(
                controller, object_id=str(target_obj["objectId"]), target_position=stale_pos, action="PickupObject"
            )
            outcome = "FAILED: object is not at the remembered location" if not success else "SUCCESS"
            close_phase()
    finally:
        controller.stop()

    intro = _card(
        panel * 2,
        panel + 78,
        [
            f"{scene} / {target} / seed {seed}",
            "MemoryGuard demo: verify-update-act memory maintenance",
            "Red X = remembered (stale) location | Green circle = actual object",
        ],
        bg=(18, 22, 30),
        fg=WHITE,
    )
    end = _card(
        panel * 2,
        panel + 78,
        [
            "MemoryGuard: " + outcome if mode == "active" else "Passive baseline: " + outcome,
            "Active: verify -> detect stale -> refresh memory -> pick at new location"
            if mode == "active"
            else "Passive: acts from the remembered location and finds nothing",
        ],
        bg=(18, 22, 30) if mode == "active" else (40, 18, 18),
        fg=WHITE,
    )

    composed: list[Image.Image] = [intro] * max(1, int(fps * 2.2))
    for index in range(len(controller.frames)):
        header = next((str(p["header"]) for p in phases if int(p["start"]) <= index < int(p["end"])), "")
        map_image = _draw_map(
            panel,
            reachable=reachable,
            stale_pos=stale_pos,
            true_pos=target_pos,
            trail=controller.poses,
            step_index=index,
        )
        composed.append(_compose(controller.frames[index], map_image, header, f"frame {index + 1}/{len(controller.frames)}", panel))
    composed.extend([end] * max(1, int(fps * 2.8)))

    for index, image in enumerate(composed):
        image.save(frames_dir / f"{index:05d}.png")
    video_path = out_dir / f"{scene}_{target}_{seed}_{mode}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"), "-pix_fmt", "yuv420p", str(video_path)],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(frames_dir)
    return video_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render watchable two-panel simulation demos for MemoryGuard cases")
    parser.add_argument("--row-source", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="Scene:Target:Seed (repeatable)")
    parser.add_argument("--mode", choices=["active", "passive"], default="active")
    parser.add_argument("--out-dir", type=Path, default=Path("results/0514_demo_videos"))
    parser.add_argument("--panel", type=int, default=600)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args(argv)

    out_dir = cast(Path, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cast(list[str], args.case):
        scene, target, seed_str = case.split(":")
        seed = int(seed_str)
        row = _load_row(cast(Path, args.row_source), scene, target, seed)
        video = render_case(
            scene=scene,
            target=target,
            seed=seed,
            row=row,
            mode=str(args.mode),
            out_dir=out_dir,
            panel=int(args.panel),
            fps=int(args.fps),
        )
        print("wrote", video)


if __name__ == "__main__":
    main()
