#!/usr/bin/env python
"""Summarize DLC keypoints, local motion, velocity, and arena coverage.

This is an exploratory analysis script for the freely roaming behavior folders.
It discovers recording sessions from the local naming convention, reads DLC
HDF5 pose tables and trusted local *_PROC pickle files, then writes CSV/JSON
summaries plus a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional visualization dependency
    plt = None


SESSION_SUFFIXES = {
    "VIDEO": re.compile(r"^(?P<base>.+)_VIDEO\.avi$", re.IGNORECASE),
    "TS": re.compile(r"^(?P<base>.+)_TS\.npy$", re.IGNORECASE),
    "DLC": re.compile(r"^(?P<base>.+)_DLC\.hdf5?$", re.IGNORECASE),
    "PROC": re.compile(r"^(?P<base>.+)_PROC$", re.IGNORECASE),
}


@dataclass(frozen=True)
class Session:
    session_id: str
    folder: str
    base: str
    files: dict[str, Path]


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def session_base_and_kind(path: Path) -> tuple[str, str] | None:
    for kind, pattern in SESSION_SUFFIXES.items():
        match = pattern.match(path.name)
        if match:
            return match.group("base"), kind
    return None


def discover_sessions(root: Path) -> list[Session]:
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parsed = session_base_and_kind(path)
        if parsed is None:
            continue
        base, kind = parsed
        folder = rel(path.parent, root)
        grouped.setdefault((folder, base), {})[kind] = path

    sessions = []
    for (folder, base), files in sorted(grouped.items()):
        sessions.append(Session(session_id=f"{folder}/{base}", folder=folder, base=base, files=files))
    return sessions


def read_dlc(path: Path) -> pd.DataFrame:
    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError("DLC HDF5 file has no pandas keys")
        key = "/df_with_missing" if "/df_with_missing" in keys else keys[0]
        return store[key]


def load_proc(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict in PROC pickle, found {type(value).__name__}")
    return value


def parse_avi_header(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12:
            return {"video_error": "file too small"}
        info["video_riff_header"] = header[:4].decode("ascii", errors="replace")
        info["video_riff_type"] = header[8:12].decode("ascii", errors="replace")
        scan = f.read(1024 * 1024)

    idx = scan.find(b"avih")
    if idx >= 0 and idx + 8 + 40 <= len(scan):
        size = struct.unpack_from("<I", scan, idx + 4)[0]
        chunk = scan[idx + 8 : idx + 8 + min(size, 56)]
        if len(chunk) >= 40:
            fields = struct.unpack_from("<10I", chunk, 0)
            microsec_per_frame = fields[0]
            total_frames = fields[4]
            width = fields[8]
            height = fields[9]
            fps = 1_000_000.0 / microsec_per_frame if microsec_per_frame else math.nan
            duration = total_frames / fps if fps else math.nan
            info.update(
                {
                    "video_width_px": int(width),
                    "video_height_px": int(height),
                    "video_fps_header": float(fps),
                    "video_total_frames_header": int(total_frames),
                    "video_duration_sec_header": float(duration),
                }
            )
    return info


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, q))


def numeric_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p05": math.nan,
            f"{prefix}_p25": math.nan,
            f"{prefix}_p75": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p05": float(np.percentile(finite, 5)),
        f"{prefix}_p25": float(np.percentile(finite, 25)),
        f"{prefix}_p75": float(np.percentile(finite, 75)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def bodyparts_from_dlc(df: pd.DataFrame) -> list[str]:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        return []
    bodyparts = []
    for bodypart in df.columns.get_level_values(0).unique():
        fields = {col[1] for col in df.columns if col[0] == bodypart and len(col) >= 2}
        if {"x", "y", "likelihood"}.issubset(fields):
            bodyparts.append(str(bodypart))
    return bodyparts


def summarize_dlc(session: Session, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = session.files["DLC"]
    df = read_dlc(path)
    bodyparts = bodyparts_from_dlc(df)
    rows: list[dict[str, Any]] = []
    for bodypart in bodyparts:
        part = df[bodypart]
        likelihood = part["likelihood"].to_numpy(dtype=float)
        x = part["x"].to_numpy(dtype=float)
        y = part["y"].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= threshold)
        row = {
            "session_id": session.session_id,
            "folder": session.folder,
            "bodypart": bodypart,
            "frames": int(len(df)),
            "valid_xy_fraction": float(np.mean(np.isfinite(x) & np.isfinite(y))) if len(df) else math.nan,
            "confidence_ge_threshold_fraction": float(np.mean(likelihood >= threshold)) if len(df) else math.nan,
            "usable_fraction": float(np.mean(good)) if len(df) else math.nan,
        }
        row.update(numeric_summary(likelihood, "likelihood"))
        if np.any(good):
            row.update(
                {
                    "x_good_min": float(np.min(x[good])),
                    "x_good_max": float(np.max(x[good])),
                    "y_good_min": float(np.min(y[good])),
                    "y_good_max": float(np.max(y[good])),
                }
            )
        rows.append(row)

    frame_time = None
    pose_time = None
    if isinstance(df.columns, pd.MultiIndex):
        if ("frame_time", "") in df.columns:
            frame_time = df[("frame_time", "")].to_numpy(dtype=float)
        if ("pose_time", "") in df.columns:
            pose_time = df[("pose_time", "")].to_numpy(dtype=float)

    session_row: dict[str, Any] = {
        "session_id": session.session_id,
        "folder": session.folder,
        "base": session.base,
        "dlc_frames": int(len(df)),
        "dlc_bodyparts": len(bodyparts),
        "dlc_bodypart_names": ";".join(bodyparts),
    }
    if frame_time is not None and frame_time.size:
        session_row["dlc_frame_time_start"] = float(np.nanmin(frame_time))
        session_row["dlc_frame_time_stop"] = float(np.nanmax(frame_time))
        session_row["dlc_frame_time_duration_sec"] = float(np.nanmax(frame_time) - np.nanmin(frame_time))
    if pose_time is not None and pose_time.size:
        session_row["dlc_pose_time_start"] = float(np.nanmin(pose_time))
        session_row["dlc_pose_time_stop"] = float(np.nanmax(pose_time))
    return session_row, rows


def summarize_timestamps(session: Session) -> dict[str, Any]:
    path = session.files.get("TS")
    if path is None:
        return {}
    ts = np.load(path, mmap_mode="r", allow_pickle=False)
    result: dict[str, Any] = {"ts_frames": int(ts.shape[0])}
    if ts.size:
        result.update(
            {
                "ts_start": float(ts[0]),
                "ts_stop": float(ts[-1]),
                "ts_duration_sec": float(ts[-1] - ts[0]),
                "ts_median_step_sec": float(np.median(np.diff(ts[: min(ts.size, 10000)]))) if ts.size > 1 else math.nan,
            }
        )
    return result


def summarize_video(session: Session) -> dict[str, Any]:
    path = session.files.get("VIDEO")
    if path is None:
        return {}
    return parse_avi_header(path)


def summarize_motion(
    session: Session,
    coverage_bin_size: float,
    min_dt: float,
    max_speed_px_per_sec: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = session.files["PROC"]
    proc = load_proc(path)
    x = np.asarray(proc.get("center_x", []), dtype=float)
    y = np.asarray(proc.get("center_y", []), dtype=float)
    t = np.asarray(proc.get("time_stamp", proc.get("frame_time", [])), dtype=float)
    heading = np.asarray(proc.get("heading_direction", []), dtype=float)
    head_angle = np.asarray(proc.get("head_angle", []), dtype=float)

    n = int(min(x.size, y.size, t.size))
    x = x[:n]
    y = y[:n]
    t = t[:n]
    valid_pos = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & ~((x == 0) & (y == 0))

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    step_valid = valid_pos[:-1] & valid_pos[1:] & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt >= min_dt)
    speed = np.full(max(n - 1, 0), np.nan, dtype=float)
    if speed.size:
        speed[step_valid] = np.sqrt(dx[step_valid] ** 2 + dy[step_valid] ** 2) / dt[step_valid]
        speed[speed > max_speed_px_per_sec] = np.nan

    motion_row: dict[str, Any] = {
        "session_id": session.session_id,
        "folder": session.folder,
        "proc_samples": n,
        "valid_position_samples": int(np.sum(valid_pos)),
        "valid_position_fraction": float(np.mean(valid_pos)) if n else math.nan,
        "valid_speed_samples": int(np.sum(np.isfinite(speed))),
    }
    if n:
        motion_row["proc_duration_sec"] = float(np.nanmax(t) - np.nanmin(t))
    motion_row.update(numeric_summary(speed, "speed_px_per_sec"))
    if heading.size:
        motion_row.update(numeric_summary(heading[np.isfinite(heading)], "heading_direction_deg"))
    if head_angle.size:
        motion_row.update(numeric_summary(head_angle[np.isfinite(head_angle)], "head_angle"))

    coverage = summarize_coverage(x[valid_pos], y[valid_pos], coverage_bin_size)
    coverage_row = {"session_id": session.session_id, "folder": session.folder}
    coverage_row.update(coverage)
    return motion_row, coverage_row


def proc_position_and_speed(
    proc_path: Path,
    min_dt: float,
    max_speed_px_per_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proc = load_proc(proc_path)
    x = np.asarray(proc.get("center_x", []), dtype=float)
    y = np.asarray(proc.get("center_y", []), dtype=float)
    t = np.asarray(proc.get("time_stamp", proc.get("frame_time", [])), dtype=float)
    n = int(min(x.size, y.size, t.size))
    x = x[:n]
    y = y[:n]
    t = t[:n]
    valid_pos = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & ~((x == 0) & (y == 0))
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    speed = np.full(max(n - 1, 0), np.nan, dtype=float)
    step_valid = valid_pos[:-1] & valid_pos[1:] & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt >= min_dt)
    if speed.size:
        speed[step_valid] = np.sqrt(dx[step_valid] ** 2 + dy[step_valid] ** 2) / dt[step_valid]
        speed[speed > max_speed_px_per_sec] = np.nan
    return x[valid_pos], y[valid_pos], speed[np.isfinite(speed)]


def summarize_coverage(x: np.ndarray, y: np.ndarray, bin_size: float) -> dict[str, Any]:
    if x.size == 0 or y.size == 0:
        return {
            "x_min": math.nan,
            "x_max": math.nan,
            "y_min": math.nan,
            "y_max": math.nan,
            "occupied_bins": 0,
            "total_bins_in_bbox": 0,
            "coverage_fraction_bbox": math.nan,
            "occupancy_entropy_norm": math.nan,
        }
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    if x_edges.size < 2:
        x_edges = np.array([x_min, x_min + bin_size])
    if y_edges.size < 2:
        y_edges = np.array([y_min, y_min + bin_size])
    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    occupied = int(np.sum(hist > 0))
    total = int(hist.size)
    probs = hist[hist > 0] / np.sum(hist)
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else math.nan
    entropy_norm = entropy / math.log2(total) if total > 1 and np.isfinite(entropy) else math.nan
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "bbox_width_px": x_max - x_min,
        "bbox_height_px": y_max - y_min,
        "coverage_bin_size_px": bin_size,
        "occupied_bins": occupied,
        "total_bins_in_bbox": total,
        "coverage_fraction_bbox": occupied / total if total else math.nan,
        "occupancy_entropy_norm": entropy_norm,
    }


def write_markdown_report(
    output_path: Path,
    root: Path,
    threshold: float,
    session_df: pd.DataFrame,
    keypoint_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    lines = [
        "# Tracking Data Understanding Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Root: `{root}`",
        f"DLC confidence threshold used for usable-frame summaries: `{threshold}`",
        "",
        "## Dataset Inventory",
        "",
        f"- Sessions discovered: {len(session_df)}",
        f"- Sessions with DLC: {int(session_df['has_dlc'].sum()) if 'has_dlc' in session_df else 0}",
        f"- Sessions with PROC local-motion data: {int(session_df['has_proc'].sum()) if 'has_proc' in session_df else 0}",
        f"- Sessions with timestamp arrays: {int(session_df['has_ts'].sum()) if 'has_ts' in session_df else 0}",
        f"- Sessions with video files: {int(session_df['has_video'].sum()) if 'has_video' in session_df else 0}",
        "",
        "## DLC Keypoints",
        "",
    ]

    if not keypoint_df.empty:
        bodyparts = sorted(keypoint_df["bodypart"].unique())
        lines.append(f"Detected bodyparts: {', '.join(bodyparts)}")
        lines.append("")
        grouped = keypoint_df.groupby("bodypart", as_index=False).agg(
            likelihood_median=("likelihood_median", "median"),
            likelihood_p05=("likelihood_p05", "median"),
            usable_fraction=("usable_fraction", "median"),
        )
        lines.append("| Bodypart | Median Likelihood | Median P05 Likelihood | Median Usable Fraction |")
        lines.append("|---|---:|---:|---:|")
        for row in grouped.sort_values("bodypart").itertuples(index=False):
            lines.append(
                f"| {row.bodypart} | {row.likelihood_median:.3f} | "
                f"{row.likelihood_p05:.3f} | {row.usable_fraction:.3f} |"
            )
    else:
        lines.append("No DLC keypoint summaries were produced.")

    lines.extend(["", "## Local Motion And Velocity", ""])
    if not motion_df.empty:
        cols = [
            "session_id",
            "proc_samples",
            "valid_position_fraction",
            "speed_px_per_sec_median",
            "speed_px_per_sec_p95",
            "speed_px_per_sec_max",
        ]
        lines.append("| Session | Samples | Valid Position Fraction | Median Speed px/s | P95 Speed px/s | Max Speed px/s |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in motion_df[cols].itertuples(index=False):
            lines.append(
                f"| {row.session_id} | {row.proc_samples} | {row.valid_position_fraction:.3f} | "
                f"{row.speed_px_per_sec_median:.2f} | {row.speed_px_per_sec_p95:.2f} | "
                f"{row.speed_px_per_sec_max:.2f} |"
            )
    else:
        lines.append("No PROC local-motion summaries were produced.")

    lines.extend(["", "## Arena Coverage", ""])
    if not coverage_df.empty:
        lines.append("| Session | BBox Width px | BBox Height px | Occupied Bins | Coverage Fraction | Occupancy Entropy |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in coverage_df.itertuples(index=False):
            lines.append(
                f"| {row.session_id} | {row.bbox_width_px:.1f} | {row.bbox_height_px:.1f} | "
                f"{row.occupied_bins} | {row.coverage_fraction_bbox:.3f} | "
                f"{row.occupancy_entropy_norm:.3f} |"
            )
    else:
        lines.append("No arena coverage summaries were produced.")

    lines.extend(
        [
            "",
            "## QC Figures",
            "",
        ]
    )
    if plot_paths:
        for path in plot_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- No plots were generated.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Units are pixels and seconds. Physical velocity needs calibration from pixels to distance.",
            "- `*_PROC` files are trusted local Python pickles; this script only loads them because they are local experiment outputs.",
            "- Coverage is estimated from processed center positions with `(0, 0)` removed as an invalid placeholder.",
            "- Coverage fraction is relative to each session's observed bounding box, not a fixed arena mask.",
            "- Very high instantaneous speeds are excluded above the configured cap to reduce timing or tracking artifacts.",
            "",
            "## Recommended Next Checks",
            "",
            "- Confirm which body point should define animal location: processed center, DLC body centroid, tail base, or a filtered body-axis midpoint.",
            "- Add arena calibration or a fixed arena mask so coverage is comparable across sessions.",
            "- Plot velocity histograms and occupancy heatmaps for visual QC.",
            "- Decide a confidence threshold and interpolation strategy for low-confidence DLC frames.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grid_axes(n_items: int) -> tuple[int, int]:
    cols = min(3, max(1, int(math.ceil(math.sqrt(n_items)))))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def make_plots(
    sessions: list[Session],
    keypoint_df: pd.DataFrame,
    output_dir: Path,
    min_dt: float,
    max_speed_px_per_sec: float,
    coverage_bin_size: float,
) -> list[Path]:
    if plt is None:
        return []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    if not keypoint_df.empty:
        for value_col, title, filename in [
            ("likelihood_median", "Median DLC Likelihood", "dlc_median_likelihood_heatmap.png"),
            ("usable_fraction", "DLC Usable Fraction", "dlc_usable_fraction_heatmap.png"),
        ]:
            pivot = keypoint_df.pivot(index="bodypart", columns="session_id", values=value_col)
            fig_width = max(10, 0.75 * len(pivot.columns))
            fig_height = max(5, 0.35 * len(pivot.index))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_title(title)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(x).split("/")[-1] for x in pivot.columns], rotation=60, ha="right")
            fig.colorbar(image, ax=ax, label=value_col)
            path = plot_dir / filename
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plot_paths.append(path)

    proc_sessions = [session for session in sessions if "PROC" in session.files]
    if proc_sessions:
        rows, cols = grid_axes(len(proc_sessions))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, session in zip(axes_arr, proc_sessions):
            _, _, speed = proc_position_and_speed(session.files["PROC"], min_dt, max_speed_px_per_sec)
            ax.hist(speed, bins=80, range=(0, min(max_speed_px_per_sec, percentile(speed, 99.5))), color="#4c78a8")
            ax.set_title(session.base, fontsize=9)
            ax.set_xlabel("Speed (px/s)")
            ax.set_ylabel("Frames")
        for ax in axes_arr[len(proc_sessions) :]:
            ax.axis("off")
        path = plot_dir / "velocity_histograms.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

        fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, session in zip(axes_arr, proc_sessions):
            x, y, _ = proc_position_and_speed(session.files["PROC"], min_dt, max_speed_px_per_sec)
            if x.size and y.size:
                x_edges = np.arange(float(np.min(x)), float(np.max(x)) + coverage_bin_size, coverage_bin_size)
                y_edges = np.arange(float(np.min(y)), float(np.max(y)) + coverage_bin_size, coverage_bin_size)
                if x_edges.size < 2:
                    x_edges = np.array([float(np.min(x)), float(np.min(x)) + coverage_bin_size])
                if y_edges.size < 2:
                    y_edges = np.array([float(np.min(y)), float(np.min(y)) + coverage_bin_size])
                hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
                ax.imshow(
                    np.log1p(hist.T),
                    origin="lower",
                    aspect="equal",
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    cmap="magma",
                )
            ax.set_title(session.base, fontsize=9)
            ax.set_xlabel("x px")
            ax.set_ylabel("y px")
        for ax in axes_arr[len(proc_sessions) :]:
            ax.axis("off")
        path = plot_dir / "arena_occupancy_maps.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DLC keypoint tracking and local motion.")
    parser.add_argument("root", nargs="?", default=".", type=Path, help="Data folder to analyze.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"), help="Directory for outputs.")
    parser.add_argument("--confidence-threshold", type=float, default=0.8, help="DLC likelihood threshold.")
    parser.add_argument("--coverage-bin-size", type=float, default=25.0, help="Occupancy bin size in pixels.")
    parser.add_argument("--min-dt", type=float, default=0.001, help="Minimum allowed time step for speed.")
    parser.add_argument(
        "--max-speed-px-per-sec",
        type=float,
        default=2000.0,
        help="Discard speeds above this value as likely artifacts.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip QC plot generation.")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(root)
    session_rows: list[dict[str, Any]] = []
    keypoint_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for session in sessions:
        row: dict[str, Any] = {
            "session_id": session.session_id,
            "folder": session.folder,
            "base": session.base,
            "has_dlc": "DLC" in session.files,
            "has_proc": "PROC" in session.files,
            "has_ts": "TS" in session.files,
            "has_video": "VIDEO" in session.files,
        }
        row.update(summarize_video(session))
        row.update(summarize_timestamps(session))

        if "DLC" in session.files:
            try:
                dlc_row, kp_rows = summarize_dlc(session, args.confidence_threshold)
                row.update(dlc_row)
                keypoint_rows.extend(kp_rows)
            except Exception as exc:
                errors.append({"session_id": session.session_id, "stage": "DLC", "error": repr(exc)})

        if "PROC" in session.files:
            try:
                motion_row, coverage_row = summarize_motion(
                    session,
                    args.coverage_bin_size,
                    args.min_dt,
                    args.max_speed_px_per_sec,
                )
                motion_rows.append(motion_row)
                coverage_rows.append(coverage_row)
            except Exception as exc:
                errors.append({"session_id": session.session_id, "stage": "PROC", "error": repr(exc)})
        session_rows.append(row)

    session_df = pd.DataFrame(session_rows)
    keypoint_df = pd.DataFrame(keypoint_rows)
    motion_df = pd.DataFrame(motion_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    errors_df = pd.DataFrame(errors)

    session_df.to_csv(output_dir / "session_summary.csv", index=False)
    keypoint_df.to_csv(output_dir / "dlc_keypoint_summary.csv", index=False)
    motion_df.to_csv(output_dir / "localmotion_velocity_summary.csv", index=False)
    coverage_df.to_csv(output_dir / "arena_coverage_summary.csv", index=False)
    errors_df.to_csv(output_dir / "analysis_errors.csv", index=False)

    overall = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "sessions": len(sessions),
        "confidence_threshold": args.confidence_threshold,
        "coverage_bin_size_px": args.coverage_bin_size,
        "errors": errors,
        "outputs": {
            "session_summary": str(output_dir / "session_summary.csv"),
            "dlc_keypoint_summary": str(output_dir / "dlc_keypoint_summary.csv"),
            "localmotion_velocity_summary": str(output_dir / "localmotion_velocity_summary.csv"),
            "arena_coverage_summary": str(output_dir / "arena_coverage_summary.csv"),
            "report": str(output_dir / "tracking_data_understanding_report.md"),
        },
    }
    plot_paths = []
    if not args.no_plots:
        plot_paths = make_plots(
            sessions,
            keypoint_df,
            output_dir,
            args.min_dt,
            args.max_speed_px_per_sec,
            args.coverage_bin_size,
        )
        overall["outputs"]["plots"] = [str(path) for path in plot_paths]
    (output_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    write_markdown_report(
        output_dir / "tracking_data_understanding_report.md",
        root,
        args.confidence_threshold,
        session_df,
        keypoint_df,
        motion_df,
        coverage_df,
        plot_paths,
    )

    print(f"Analyzed {len(sessions)} sessions")
    print(f"Wrote outputs to {output_dir}")
    if errors:
        print(f"Completed with {len(errors)} errors; see analysis_errors.csv")


if __name__ == "__main__":
    main()
