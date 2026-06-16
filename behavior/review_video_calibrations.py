#!/usr/bin/env python
"""Per-video arena calibration review workflow.

For each selected video, this script loads a default cage-floor edge-points JSON
as the starting point, opens the arena edge editor from `arena_calibration.py`,
and saves a per-session calibration JSON plus diagnostics. It also writes a
manifest CSV that `dlc_locomotion_pipeline.py` can use to apply the matching
calibration for each session.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import arena_calibration as ac


def session_id_from_video(root: Path, video: Path) -> str:
    rel = video.parent.relative_to(root).as_posix()
    base = re.sub(r"_VIDEO\.avi$", "", video.name, flags=re.IGNORECASE)
    return f"{rel}/{base}"


def group_from_session(session_id: str) -> str:
    if session_id.startswith("Centimani_implanted/"):
        return "implanted"
    if session_id.startswith("Comparison_unimplanted/"):
        return "unimplanted"
    return "unknown"


def find_main_videos(root: Path, include_no_use: bool) -> list[Path]:
    videos = sorted(root.rglob("*_VIDEO.avi"))
    if not include_no_use:
        videos = [path for path in videos if "no_use_videos" not in path.parts]
    return videos


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "include",
        "session_id",
        "group",
        "video_path",
        "calibration_json",
        "edge_points_json",
        "arena_edge_length_cm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def manifest_row_for_calibration(
    video: Path,
    root: Path,
    output_json: Path,
    arena_edge_length_cm: float,
) -> dict[str, Any]:
    session_id = session_id_from_video(root, video)
    edge_json = output_json.with_name(f"{output_json.stem}_edge_points.json")
    return {
        "include": "true",
        "session_id": session_id,
        "group": group_from_session(session_id),
        "video_path": str(video.relative_to(root)),
        "calibration_json": str(output_json.relative_to(root)),
        "edge_points_json": str(edge_json.relative_to(root)),
        "arena_edge_length_cm": arena_edge_length_cm,
    }


def calibrate_one_video(
    video: Path,
    root: Path,
    output_root: Path,
    default_edge_points: Path,
    arena_edge_length_cm: float,
    no_gui: bool,
    output_pixels: int,
) -> dict[str, Any]:
    session_id = session_id_from_video(root, video)
    session_name = ac.safe_stem(video)
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    edge_points_path = default_edge_points
    output_json = session_dir / f"{session_name}_calibration.json"
    existing_edge_json = output_json.with_name(f"{output_json.stem}_edge_points.json")
    if output_json.exists() and existing_edge_json.exists():
        print(f"\nSession: {session_id}")
        print(f"Using existing calibration: {output_json}")
        return manifest_row_for_calibration(video, root, output_json, arena_edge_length_cm)
    print(f"\nSession: {session_id}")
    print(f"Video: {video}")
    print(f"Default edge points: {default_edge_points}")
    print(f"cage_floor_edge_length_cm = {arena_edge_length_cm:g}")
    if not no_gui:
        edge_text = input(f"Edge-points JSON [{edge_points_path}]: ").strip()
        if edge_text:
            edge_points_path = Path(edge_text)
            if not edge_points_path.is_absolute():
                edge_points_path = root / edge_points_path
        output_text = input(f"Output calibration JSON [{output_json}]: ").strip()
        if output_text:
            output_json = Path(output_text)
            if not output_json.is_absolute():
                output_json = root / output_json
            session_dir = output_json.parent
            session_dir.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        video=video,
        output=output_json,
        frame_index=0,
        points_per_edge=30,
        arena_side_length=1.0,
        target_units="normalized_arena_side",
        polynomial_order=3,
        output_pixels=output_pixels,
        output_dir=session_dir / "diagnostics",
        corners_json=None,
        edge_points_json=edge_points_path,
        search_radius_px=35.0,
        profile_half_width_px=3,
        gaussian_sigma=1.5,
        spline_smoothing=8.0,
        outlier_threshold_px=12.0,
        no_gui=no_gui,
    )
    ac.command_calibrate(ns)
    return manifest_row_for_calibration(video, root, output_json, arena_edge_length_cm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and save one calibration per video.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--default-edge-points-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/arena_calibration_per_video"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--arena-edge-length-cm", type=float, default=52.0)
    parser.add_argument("--session-filter", default=None, help="Regex filter applied to session_id.")
    parser.add_argument("--exclude-session-filter", default=None, help="Regex filter for sessions to skip.")
    parser.add_argument("--include-no-use", action="store_true")
    parser.add_argument("--no-gui", action="store_true", help="Use loaded edge points directly; useful for smoke/example runs.")
    parser.add_argument("--output-pixels", type=int, default=700)
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    default_edge_points = args.default_edge_points_json
    if not default_edge_points.is_absolute():
        default_edge_points = root / default_edge_points
    manifest_path = args.manifest or output_root / "calibration_manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    print(f"cage_floor_edge_length_cm = {args.arena_edge_length_cm:g}")
    videos = find_main_videos(root, args.include_no_use)
    include_re = re.compile(args.session_filter) if args.session_filter else None
    exclude_re = re.compile(args.exclude_session_filter) if args.exclude_session_filter else None
    selected = []
    for video in videos:
        session_id = session_id_from_video(root, video)
        if include_re and not include_re.search(session_id):
            continue
        if exclude_re and exclude_re.search(session_id):
            continue
        selected.append(video)
    rows = []
    for video in selected:
        row = calibrate_one_video(video, root, output_root, default_edge_points, args.arena_edge_length_cm, args.no_gui, args.output_pixels)
        rows.append(row)
        write_manifest(manifest_path, rows)
    write_manifest(manifest_path, rows)
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Reviewed calibrations: {len(rows)}")


if __name__ == "__main__":
    main()
