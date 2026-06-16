#!/usr/bin/env python
"""DLC-only locomotion pipeline from reliable trunk keypoints.

The centroid is always the non-weighted mean of the same five trunk keypoints:
tail_base, left_hip, right_hip, left_midside, and right_midside. Low-confidence
or implausible keypoints are masked first, each keypoint is interpolated
independently for short gaps only, and centroid/speed smoothing is applied only
within contiguous valid segments so missing samples cannot contaminate the
filter window.

Optional arena calibration can be applied from arena_calibration.py JSON files.
When calibration is supplied, raw-pixel QC is still performed first; cleaned
keypoints are then transformed into corrected square arena coordinates before
centroid, speed, and coverage are calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import traceback
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


RELIABLE_KEYPOINTS = ["tail_base", "left_hip", "right_hip", "left_midside", "right_midside"]
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


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    if len(cleaned) > 50:
        cleaned = cleaned[:50].rstrip("_")
    return f"{cleaned}__{digest}"


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
    return [
        Session(session_id=f"{folder}/{base}", folder=folder, base=base, files=files)
        for (folder, base), files in sorted(grouped.items())
    ]


def read_dlc(path: Path) -> pd.DataFrame:
    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"{path} has no pandas HDF5 keys")
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
            return {}
        scan = f.read(1024 * 1024)
    idx = scan.find(b"avih")
    if idx >= 0 and idx + 8 + 40 <= len(scan):
        size = struct.unpack_from("<I", scan, idx + 4)[0]
        chunk = scan[idx + 8 : idx + 8 + min(size, 56)]
        if len(chunk) >= 40:
            fields = struct.unpack_from("<10I", chunk, 0)
            microsec_per_frame = fields[0]
            fps = 1_000_000.0 / microsec_per_frame if microsec_per_frame else math.nan
            info.update(
                {
                    "video_total_frames_header": int(fields[4]),
                    "video_width_px": int(fields[8]),
                    "video_height_px": int(fields[9]),
                    "video_fps_header": float(fps),
                }
            )
    return info


def apply_spatial_calibration(
    x: np.ndarray,
    y: np.ndarray,
    calibration_params: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved arena calibration to x/y arrays.

    The supported calibration is the boundary polynomial JSON written by
    arena_calibration.py. If no calibration is supplied, coordinates remain raw
    camera pixels.
    """
    if calibration_params is None:
        return x, y
    transform = calibration_params.get("transform", {})
    if transform.get("type") != "boundary_polynomial_raw_pixel_to_square":
        raise ValueError(f"Unsupported calibration transform: {transform.get('type')}")
    terms = [(int(a), int(b)) for a, b in transform["terms"]]
    width = float(transform["source_width"])
    height = float(transform["source_height"])
    sx = x / max(1.0, width - 1)
    sy = y / max(1.0, height - 1)
    design = np.column_stack([(sx**px) * (sy**py) for px, py in terms])
    base = np.column_stack(
        [
            design @ np.asarray(transform["forward_coeff_x"], dtype=float),
            design @ np.asarray(transform["forward_coeff_y"], dtype=float),
        ]
    )
    h = np.asarray(transform.get("post_homography", np.eye(3).tolist()), dtype=float)
    homog = np.column_stack([base, np.ones(len(base))]) @ h.T
    mapped = homog[:, :2] / homog[:, 2:3]
    x_out = mapped[:, 0]
    y_out = mapped[:, 1]
    x_out[~np.isfinite(x) | ~np.isfinite(y)] = np.nan
    y_out[~np.isfinite(x) | ~np.isfinite(y)] = np.nan
    return x_out, y_out


def load_manifest(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    table = pd.read_csv(path)
    if "session_id" not in table.columns:
        raise ValueError("Calibration manifest must contain a session_id column.")
    if "include" in table.columns:
        include = table["include"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        table = table.loc[include].copy()
    manifest: dict[str, dict[str, Any]] = {}
    for row in table.to_dict(orient="records"):
        session_id = str(row["session_id"])
        calibration_json = row.get("calibration_json")
        if isinstance(calibration_json, str) and calibration_json:
            cal_path = Path(calibration_json)
            if not cal_path.is_absolute():
                cal_path = root / cal_path
            row["calibration_json"] = str(cal_path)
        manifest[session_id] = row
    return manifest


def read_calibration(path: Path | str | None) -> dict[str, Any] | None:
    if path is None or str(path) == "":
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_physical_scale(frame_table: pd.DataFrame, params: dict[str, Any], calibration_params: dict[str, Any] | None) -> None:
    edge_length_cm = params.get("arena_edge_length_cm")
    if edge_length_cm is None or not np.isfinite(float(edge_length_cm)):
        return
    if calibration_params is None:
        raise ValueError("Physical scaling requires calibrated arena coordinates.")
    target_size = float(calibration_params.get("target_size") or 1.0)
    if target_size <= 0:
        raise ValueError("Calibration target_size must be positive for physical scaling.")
    edge_length_m = float(edge_length_cm) / 100.0
    scale_m_per_coord = edge_length_m / target_size
    for src, dst in [
        ("centroid_x_clean", "centroid_x_m"),
        ("centroid_y_clean", "centroid_y_m"),
        ("centroid_x_smooth", "centroid_x_smooth_m"),
        ("centroid_y_smooth", "centroid_y_smooth_m"),
    ]:
        frame_table[dst] = frame_table[src].to_numpy(dtype=float) * scale_m_per_coord
    frame_table["speed_m_per_sec"] = frame_table["speed_px_per_sec"].to_numpy(dtype=float) * scale_m_per_coord
    params["arena_edge_length_m"] = edge_length_m
    params["scale_m_per_corrected_coord"] = scale_m_per_coord
    params["physical_units_status"] = "applied_from_calibrated_square_edge_length"


def get_time_vector(df: pd.DataFrame, fallback_fps: float) -> tuple[np.ndarray, str]:
    for col, name in [(("frame_time", ""), "frame_time"), (("pose_time", ""), "pose_time")]:
        if isinstance(df.columns, pd.MultiIndex) and col in df.columns:
            t = df[col].to_numpy(dtype=float)
            if np.sum(np.isfinite(t)) > max(2, 0.5 * len(t)):
                return t, name
    return np.arange(len(df), dtype=float) / fallback_fps, "frame_index_over_fps"


def map_dlc_times_to_video_frames(
    session: Session,
    frame_time: np.ndarray,
    tolerance_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each DLC row timestamp to the nearest raw video timestamp."""
    video_index = np.full(len(frame_time), -1, dtype=np.int64)
    time_error = np.full(len(frame_time), np.nan, dtype=float)
    if "TS" not in session.files or len(frame_time) == 0:
        return video_index, time_error
    ts = np.load(session.files["TS"], mmap_mode="r", allow_pickle=False)
    if ts.size == 0:
        return video_index, time_error
    idx = np.searchsorted(ts, frame_time)
    idx = np.clip(idx, 1, len(ts) - 1)
    left = ts[idx - 1]
    right = ts[idx]
    use_left = np.abs(frame_time - left) <= np.abs(right - frame_time)
    nearest = idx - use_left.astype(np.int64)
    error = np.abs(frame_time - ts[nearest])
    matched = np.isfinite(error) & (error <= tolerance_sec)
    video_index[matched] = nearest[matched]
    time_error[matched] = error[matched]
    return video_index, time_error


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def mark_jump_outliers(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    valid: np.ndarray,
    max_speed_px_per_sec: float,
    min_dt: float,
) -> np.ndarray:
    """Reject isolated keypoint spikes using adjacent valid-frame speeds."""
    cleaned = valid.copy()
    if len(x) < 3:
        return cleaned
    dx_prev = x[1:-1] - x[:-2]
    dy_prev = y[1:-1] - y[:-2]
    dt_prev = t[1:-1] - t[:-2]
    dx_next = x[2:] - x[1:-1]
    dy_next = y[2:] - y[1:-1]
    dt_next = t[2:] - t[1:-1]
    prev_valid = cleaned[:-2] & cleaned[1:-1] & np.isfinite(dt_prev) & (dt_prev >= min_dt)
    next_valid = cleaned[1:-1] & cleaned[2:] & np.isfinite(dt_next) & (dt_next >= min_dt)
    prev_speed = np.full(len(x) - 2, np.nan)
    next_speed = np.full(len(x) - 2, np.nan)
    prev_speed[prev_valid] = np.sqrt(dx_prev[prev_valid] ** 2 + dy_prev[prev_valid] ** 2) / dt_prev[prev_valid]
    next_speed[next_valid] = np.sqrt(dx_next[next_valid] ** 2 + dy_next[next_valid] ** 2) / dt_next[next_valid]
    spike = (prev_speed > max_speed_px_per_sec) & (next_speed > max_speed_px_per_sec)
    cleaned[1:-1][spike] = False
    return cleaned


def interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    masked = pd.Series(np.where(valid, values, np.nan), dtype="float64")
    interp = masked.interpolate(method="linear", limit=max_gap_frames, limit_area="inside")
    clean = interp.to_numpy(dtype=float)
    interpolated = (~valid) & np.isfinite(clean)
    return clean, interpolated


def smooth_segmentwise(values: np.ndarray, valid: np.ndarray, median_window: int, savgol_window: int) -> np.ndarray:
    """Smooth only within valid contiguous segments; NaNs are never filtered over."""
    out = np.full_like(values, np.nan, dtype=float)
    median_window = max(1, median_window if median_window % 2 == 1 else median_window + 1)
    savgol_window = max(3, savgol_window if savgol_window % 2 == 1 else savgol_window + 1)
    for start, stop in contiguous_true_runs(valid & np.isfinite(values)):
        segment = values[start:stop]
        if len(segment) < 3:
            out[start:stop] = segment
            continue
        med_size = min(median_window, len(segment) if len(segment) % 2 == 1 else len(segment) - 1)
        med_size = max(1, med_size)
        smoothed = median_filter(segment, size=med_size, mode="nearest") if med_size > 1 else segment.copy()
        sg_size = min(savgol_window, len(segment) if len(segment) % 2 == 1 else len(segment) - 1)
        if sg_size >= 5:
            smoothed = savgol_filter(smoothed, window_length=sg_size, polyorder=2, mode="interp")
        out[start:stop] = smoothed
    return out


def numeric_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_{k}": math.nan for k in ["mean", "median", "p05", "p25", "p75", "p95", "max"]}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p05": float(np.percentile(finite, 5)),
        f"{prefix}_p25": float(np.percentile(finite, 25)),
        f"{prefix}_p75": float(np.percentile(finite, 75)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_max": float(np.max(finite)),
    }


def add_proc_comparison(frame_df: pd.DataFrame, session: Session) -> None:
    if "PROC" not in session.files:
        return
    proc = load_proc(session.files["PROC"])
    frame_indices = frame_df["frame_index"].to_numpy(dtype=int) if "frame_index" in frame_df else np.arange(len(frame_df))
    for key in ["center_x", "center_y", "heading_direction", "head_angle", "time_stamp"]:
        values = np.full(len(frame_df), np.nan)
        if key in proc:
            arr = np.asarray(proc[key], dtype=float)
            in_bounds = (frame_indices >= 0) & (frame_indices < len(arr))
            values[in_bounds] = arr[frame_indices[in_bounds]]
        frame_df[f"proc_{key}"] = values
    frame_df["distance_to_proc_center_px"] = np.sqrt(
        (frame_df["centroid_x_smooth"] - frame_df["proc_center_x"]) ** 2
        + (frame_df["centroid_y_smooth"] - frame_df["proc_center_y"]) ** 2
    )


def subset_dlc_to_analysis_window(
    df: pd.DataFrame,
    frame_time: np.ndarray,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    duration = params.get("analysis_window_duration_sec")
    exact_start = params.get("analysis_window_start_sec")
    start_max = params.get("analysis_window_start_max_sec")
    if duration is None and exact_start is None and start_max is None:
        return df, frame_time, {"analysis_window_applied": False}

    finite = np.isfinite(frame_time)
    if np.sum(finite) < 2:
        raise ValueError("Cannot apply analysis window because frame_time has fewer than two finite values")
    if duration is None or duration <= 0:
        raise ValueError("--analysis-window-duration-sec must be positive when an analysis window is requested")

    recording_origin = float(np.nanmin(frame_time[finite]))
    rel_time = frame_time - recording_origin
    recording_duration = float(np.nanmax(rel_time[finite]))
    if recording_duration < duration:
        raise ValueError(
            f"Recording duration {recording_duration:.3f} sec is shorter than requested "
            f"analysis window {duration:.3f} sec"
        )

    if exact_start is not None:
        start = float(exact_start)
    else:
        max_allowed_start = 0.0 if start_max is None else float(start_max)
        latest_start_that_fits = recording_duration - float(duration)
        start = min(max_allowed_start, latest_start_that_fits)
        start = max(0.0, start)
    end = start + float(duration)
    if start_max is not None and start > float(start_max) + 1e-9:
        raise ValueError(f"Selected analysis window starts at {start:.3f} sec, later than allowed {float(start_max):.3f} sec")

    mask = finite & (rel_time >= start) & (rel_time < end)
    if np.sum(mask) < 2:
        raise ValueError(f"Analysis window {start:.3f}-{end:.3f} sec selected fewer than two DLC rows")

    info = {
        "analysis_window_applied": True,
        "analysis_window_requested_start_sec": exact_start,
        "analysis_window_requested_start_max_sec": start_max,
        "analysis_window_requested_duration_sec": duration,
        "analysis_window_start_sec": start,
        "analysis_window_end_sec": end,
        "analysis_window_duration_sec": float(duration),
        "analysis_window_recording_origin_time": recording_origin,
        "analysis_window_recording_duration_sec": recording_duration,
        "analysis_window_rows": int(np.sum(mask)),
    }
    return df.loc[mask].copy(), frame_time[mask], info


def build_frame_table(
    session: Session,
    df: pd.DataFrame,
    params: dict[str, Any],
    video_meta: dict[str, Any],
    calibration_params: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fps = float(video_meta.get("video_fps_header") or params["fallback_fps"])
    frame_time_full, time_source = get_time_vector(df, fps)
    df, frame_time, window_info = subset_dlc_to_analysis_window(df, frame_time_full, params)
    n = len(df)
    if np.issubdtype(df.index.dtype, np.integer):
        frame_index = df.index.to_numpy(dtype=np.int64)
    else:
        frame_index = np.arange(n, dtype=np.int64)
    frame_table = pd.DataFrame({"session_id": session.session_id, "frame_index": frame_index, "frame_time": frame_time})
    if window_info.get("analysis_window_applied"):
        frame_table["recording_time_sec"] = frame_time - float(window_info["analysis_window_recording_origin_time"])
    video_frame_index, video_time_error = map_dlc_times_to_video_frames(
        session, frame_time, params["timestamp_match_tolerance_sec"]
    )
    frame_table["video_frame_index"] = video_frame_index
    frame_table["video_time_error_sec"] = video_time_error
    if isinstance(df.columns, pd.MultiIndex) and ("pose_time", "") in df.columns:
        frame_table["pose_time"] = df[("pose_time", "")].to_numpy(dtype=float)
    else:
        frame_table["pose_time"] = np.nan

    frame_width = video_meta.get("video_width_px", params["frame_width_px"])
    frame_height = video_meta.get("video_height_px", params["frame_height_px"])
    keypoint_valid_cols = []
    keypoint_interp_cols = []
    clean_x_cols = []
    clean_y_cols = []

    for keypoint in params["keypoints"]:
        if keypoint not in df.columns.get_level_values(0):
            raise KeyError(f"Missing required DLC keypoint: {keypoint}")
        part = df[keypoint]
        x_raw = part["x"].to_numpy(dtype=float)
        y_raw = part["y"].to_numpy(dtype=float)
        likelihood = part["likelihood"].to_numpy(dtype=float)
        valid = (
            np.isfinite(x_raw)
            & np.isfinite(y_raw)
            & np.isfinite(likelihood)
            & (likelihood >= params["confidence_threshold"])
            & (x_raw >= 0)
            & (y_raw >= 0)
            & (x_raw < frame_width)
            & (y_raw < frame_height)
        )
        valid = mark_jump_outliers(
            x_raw,
            y_raw,
            frame_time,
            valid,
            params["max_keypoint_speed_px_per_sec"],
            params["min_dt_sec"],
        )
        x_clean, x_interp = interpolate_short_gaps(x_raw, valid, params["max_interpolation_gap_frames"])
        y_clean, y_interp = interpolate_short_gaps(y_raw, valid, params["max_interpolation_gap_frames"])
        x_analysis, y_analysis = apply_spatial_calibration(x_clean, y_clean, calibration_params)
        interpolated = x_interp | y_interp

        prefix = keypoint
        frame_table[f"{prefix}_x_raw"] = x_raw
        frame_table[f"{prefix}_y_raw"] = y_raw
        frame_table[f"{prefix}_likelihood"] = likelihood
        frame_table[f"{prefix}_valid"] = valid
        frame_table[f"{prefix}_interpolated"] = interpolated
        frame_table[f"{prefix}_x_clean_raw_px"] = x_clean
        frame_table[f"{prefix}_y_clean_raw_px"] = y_clean
        frame_table[f"{prefix}_x_clean"] = x_analysis
        frame_table[f"{prefix}_y_clean"] = y_analysis
        keypoint_valid_cols.append(f"{prefix}_valid")
        keypoint_interp_cols.append(f"{prefix}_interpolated")
        clean_x_cols.append(f"{prefix}_x_clean")
        clean_y_cols.append(f"{prefix}_y_clean")

    raw_all_observed = frame_table[keypoint_valid_cols].all(axis=1).to_numpy()
    cleaned_all_available = np.isfinite(frame_table[clean_x_cols]).all(axis=1).to_numpy() & np.isfinite(
        frame_table[clean_y_cols]
    ).all(axis=1).to_numpy()
    any_interpolated = frame_table[keypoint_interp_cols].any(axis=1).to_numpy()
    all_observed = raw_all_observed & cleaned_all_available

    raw_row_mask = pd.Series(raw_all_observed, index=frame_table.index)
    clean_row_mask = pd.Series(cleaned_all_available, index=frame_table.index)
    frame_table["centroid_x_raw"] = frame_table[clean_x_cols].where(raw_row_mask, axis=0).mean(axis=1)
    frame_table["centroid_y_raw"] = frame_table[clean_y_cols].where(raw_row_mask, axis=0).mean(axis=1)
    raw_px_x_cols = [f"{keypoint}_x_clean_raw_px" for keypoint in params["keypoints"]]
    raw_px_y_cols = [f"{keypoint}_y_clean_raw_px" for keypoint in params["keypoints"]]
    frame_table["centroid_x_overlay_px"] = frame_table[raw_px_x_cols].where(clean_row_mask, axis=0).mean(axis=1)
    frame_table["centroid_y_overlay_px"] = frame_table[raw_px_y_cols].where(clean_row_mask, axis=0).mean(axis=1)
    frame_table["centroid_available_raw"] = raw_all_observed
    frame_table["centroid_x_clean"] = frame_table[clean_x_cols].where(clean_row_mask, axis=0).mean(axis=1)
    frame_table["centroid_y_clean"] = frame_table[clean_y_cols].where(clean_row_mask, axis=0).mean(axis=1)
    frame_table["centroid_valid"] = cleaned_all_available
    frame_table["centroid_any_interpolated"] = any_interpolated & cleaned_all_available
    frame_table["centroid_all_observed"] = all_observed

    frame_table["centroid_x_smooth"] = smooth_segmentwise(
        frame_table["centroid_x_clean"].to_numpy(dtype=float),
        cleaned_all_available,
        params["median_filter_window_frames"],
        params["savgol_window_frames"],
    )
    frame_table["centroid_y_smooth"] = smooth_segmentwise(
        frame_table["centroid_y_clean"].to_numpy(dtype=float),
        cleaned_all_available,
        params["median_filter_window_frames"],
        params["savgol_window_frames"],
    )

    dx = np.diff(frame_table["centroid_x_smooth"].to_numpy(dtype=float))
    dy = np.diff(frame_table["centroid_y_smooth"].to_numpy(dtype=float))
    dt = np.diff(frame_time)
    speed = np.full(n, np.nan)
    speed_valid = (
        cleaned_all_available[:-1]
        & cleaned_all_available[1:]
        & np.isfinite(dx)
        & np.isfinite(dy)
        & np.isfinite(dt)
        & (dt >= params["min_dt_sec"])
    )
    step_speed = np.full(max(n - 1, 0), np.nan)
    step_speed[speed_valid] = np.sqrt(dx[speed_valid] ** 2 + dy[speed_valid] ** 2) / dt[speed_valid]
    step_speed[step_speed > params["max_centroid_speed_px_per_sec"]] = np.nan
    speed[1:] = step_speed
    frame_table["speed_px_per_sec"] = speed
    frame_table["speed_valid"] = np.isfinite(speed)
    apply_physical_scale(frame_table, params, calibration_params)

    add_proc_comparison(frame_table, session)

    summary = summarize_frame_table(frame_table, session, params, video_meta, time_source)
    summary.update(window_info)
    return frame_table, summary


def longest_false_gap_seconds(valid: np.ndarray, frame_time: np.ndarray, fps: float) -> float:
    if len(valid) == 0:
        return math.nan
    max_frames = 0
    for start, stop in contiguous_true_runs(~valid):
        max_frames = max(max_frames, stop - start)
    if np.sum(np.isfinite(frame_time)) > 1:
        median_dt = np.nanmedian(np.diff(frame_time[np.isfinite(frame_time)]))
        if np.isfinite(median_dt) and median_dt > 0:
            return float(max_frames * median_dt)
    return float(max_frames / fps)


def coverage_metrics(x: np.ndarray, y: np.ndarray, bin_size: float) -> dict[str, Any]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return {
            "bbox_width_px": math.nan,
            "bbox_height_px": math.nan,
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
        "bbox_width_px": x_max - x_min,
        "bbox_height_px": y_max - y_min,
        "occupied_bins": occupied,
        "total_bins_in_bbox": total,
        "coverage_fraction_bbox": occupied / total if total else math.nan,
        "occupancy_entropy_norm": entropy_norm,
    }


def summarize_frame_table(
    frame_table: pd.DataFrame,
    session: Session,
    params: dict[str, Any],
    video_meta: dict[str, Any],
    time_source: str,
) -> dict[str, Any]:
    n = len(frame_table)
    fps = float(video_meta.get("video_fps_header") or params["fallback_fps"])
    summary: dict[str, Any] = {
        "session_id": session.session_id,
        "folder": session.folder,
        "base": session.base,
        "frames": n,
        "time_source": time_source,
        "has_proc": "PROC" in session.files,
        "video_width_px": video_meta.get("video_width_px"),
        "video_height_px": video_meta.get("video_height_px"),
        "video_fps_header": video_meta.get("video_fps_header"),
        "confidence_threshold": params["confidence_threshold"],
        "max_interpolation_gap_frames": params["max_interpolation_gap_frames"],
        "median_filter_window_frames": params["median_filter_window_frames"],
        "savgol_window_frames": params["savgol_window_frames"],
    }
    t = frame_table["frame_time"].to_numpy(dtype=float)
    if np.sum(np.isfinite(t)) > 1:
        summary["duration_sec"] = float(np.nanmax(t) - np.nanmin(t))
    for keypoint in params["keypoints"]:
        summary[f"{keypoint}_valid_fraction"] = float(frame_table[f"{keypoint}_valid"].mean())
        summary[f"{keypoint}_interpolated_fraction"] = float(frame_table[f"{keypoint}_interpolated"].mean())
        summary[f"{keypoint}_likelihood_median"] = float(np.nanmedian(frame_table[f"{keypoint}_likelihood"]))
    centroid_valid = frame_table["centroid_valid"].to_numpy(dtype=bool)
    summary["centroid_valid_fraction"] = float(np.mean(centroid_valid)) if n else math.nan
    summary["centroid_all_observed_fraction"] = float(frame_table["centroid_all_observed"].mean()) if n else math.nan
    summary["centroid_any_interpolated_fraction"] = (
        float(frame_table["centroid_any_interpolated"].mean()) if n else math.nan
    )
    summary["longest_centroid_missing_gap_sec"] = longest_false_gap_seconds(centroid_valid, t, fps)
    summary.update(numeric_summary(frame_table["speed_px_per_sec"].to_numpy(dtype=float), "speed_px_per_sec"))
    if "speed_m_per_sec" in frame_table:
        summary.update(numeric_summary(frame_table["speed_m_per_sec"].to_numpy(dtype=float), "speed_m_per_sec"))
        summary["stationary_fraction_speed_lt_0p01_m_per_sec"] = float(
            np.nanmean(frame_table["speed_m_per_sec"].to_numpy(dtype=float) < 0.01)
        )
    summary["stationary_fraction_speed_lt_1_px_per_sec"] = float(
        np.nanmean(frame_table["speed_px_per_sec"].to_numpy(dtype=float) < 1)
    )
    if "centroid_x_smooth_m" in frame_table:
        bin_size = params.get("coverage_bin_size_m", params["coverage_bin_size_px"])
        summary.update(coverage_metrics(frame_table["centroid_x_smooth_m"].to_numpy(dtype=float), frame_table["centroid_y_smooth_m"].to_numpy(dtype=float), bin_size))
        summary["coverage_bin_size_m"] = bin_size
    else:
        summary.update(
            coverage_metrics(
                frame_table["centroid_x_smooth"].to_numpy(dtype=float),
                frame_table["centroid_y_smooth"].to_numpy(dtype=float),
                params["coverage_bin_size_px"],
            )
        )
    if "distance_to_proc_center_px" in frame_table:
        summary.update(numeric_summary(frame_table["distance_to_proc_center_px"].to_numpy(dtype=float), "proc_distance_px"))
    return summary


def write_frame_table(frame_table: pd.DataFrame, output_dir: Path, session_name: str) -> Path:
    path = output_dir / "dlc_locomotion.parquet"
    try:
        frame_table.to_parquet(path, index=False)
        return path
    except Exception:
        path = output_dir / "dlc_locomotion.csv.gz"
        frame_table.to_csv(path, index=False, compression="gzip")
        return path


def write_centroid_trajectory(frame_table: pd.DataFrame, output_dir: Path, session_name: str) -> Path:
    """Write a compact trajectory table for plot tuning and statistics."""
    cols = [
        "session_id",
        "frame_index",
        "frame_time",
        "video_frame_index",
        "video_time_error_sec",
        "centroid_x_clean",
        "centroid_y_clean",
        "centroid_x_smooth",
        "centroid_y_smooth",
        "speed_px_per_sec",
        "speed_valid",
        "centroid_valid",
        "centroid_all_observed",
        "centroid_any_interpolated",
        "centroid_x_m",
        "centroid_y_m",
        "centroid_x_smooth_m",
        "centroid_y_smooth_m",
        "speed_m_per_sec",
    ]
    available = [col for col in cols if col in frame_table.columns]
    path = output_dir / "centroid_trajectory.csv.gz"
    frame_table.loc[:, available].to_csv(path, index=False, compression="gzip")
    return path


def make_session_plots(frame_table: pd.DataFrame, session_dir: Path, session_name: str, bin_size: float) -> list[Path]:
    if plt is None:
        return []
    paths: list[Path] = []
    if {"centroid_x_smooth_m", "centroid_y_smooth_m"}.issubset(frame_table.columns):
        x = frame_table["centroid_x_smooth_m"].to_numpy(dtype=float)
        y = frame_table["centroid_y_smooth_m"].to_numpy(dtype=float)
        speed = frame_table["speed_m_per_sec"].to_numpy(dtype=float)
        x_label = "x (m)"
        y_label = "y (m)"
        speed_label = "speed (m/s)"
    else:
        x = frame_table["centroid_x_smooth"].to_numpy(dtype=float)
        y = frame_table["centroid_y_smooth"].to_numpy(dtype=float)
        speed = frame_table["speed_px_per_sec"].to_numpy(dtype=float)
        x_label = "x coordinate"
        y_label = "y coordinate"
        speed_label = "speed per second"
    valid = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(x[valid], y[valid], lw=0.6, color="#1f77b4")
    ax.scatter(x[valid][0:1], y[valid][0:1], s=40, color="green", label="start")
    ax.scatter(x[valid][-1:], y[valid][-1:], s=40, color="red", label="end")
    ax.set_title(f"{session_name} centroid trajectory")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(loc="best")
    path = session_dir / "centroid_trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if np.any(valid):
        x_edges = np.arange(float(np.nanmin(x)), float(np.nanmax(x)) + bin_size, bin_size)
        y_edges = np.arange(float(np.nanmin(y)), float(np.nanmax(y)) + bin_size, bin_size)
        if x_edges.size < 2:
            x_edges = np.array([float(np.nanmin(x)), float(np.nanmin(x)) + bin_size])
        if y_edges.size < 2:
            y_edges = np.array([float(np.nanmin(y)), float(np.nanmin(y)) + bin_size])
        hist, _, _ = np.histogram2d(x[valid], y[valid], bins=(x_edges, y_edges))
        image = ax.imshow(
            np.log1p(hist.T),
            origin="upper",
            aspect="equal",
            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
            cmap="magma",
        )
        fig.colorbar(image, ax=ax, label="log(occupancy + 1)")
    ax.set_title(f"{session_name} occupancy")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    path = session_dir / "occupancy_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size:
        upper = min(float(np.percentile(finite_speed, 99.5)), float(np.nanmax(finite_speed)))
        ax.hist(finite_speed, bins=80, range=(0, upper), color="#4c78a8")
    ax.set_title(f"{session_name} velocity")
    ax.set_xlabel(speed_label)
    ax.set_ylabel("frames")
    path = session_dir / "velocity_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int, outline: bool) -> None:
    if not np.isfinite(x) or not np.isfinite(y):
        return
    box = [x - radius, y - radius, x + radius, y + radius]
    if outline:
        draw.ellipse(box, outline=color, width=2)
    else:
        draw.ellipse(box, fill=color)


def make_overlay_video(
    session: Session,
    frame_table: pd.DataFrame,
    output_path: Path,
    params: dict[str, Any],
    mode: str,
    max_seconds: float,
    output_fps: float,
) -> str | None:
    if mode == "none" or "VIDEO" not in session.files:
        return None
    reader = imageio.get_reader(session.files["VIDEO"])
    meta = reader.get_meta_data()
    input_fps = float(meta.get("fps") or params["fallback_fps"])
    source_stride = max(1, int(round(input_fps / output_fps)))
    ts = None
    if "TS" in session.files:
        ts = np.load(session.files["TS"], mmap_mode="r", allow_pickle=False)
    video_frame_count = len(ts) if ts is not None else int(meta.get("nframes") or len(frame_table))
    max_source_frames = video_frame_count if mode == "full" else min(video_frame_count, int(max_seconds * input_fps))
    dlc_times = frame_table["frame_time"].to_numpy(dtype=float)
    colors = {
        "tail_base": (255, 222, 89),
        "left_hip": (80, 200, 120),
        "right_hip": (80, 160, 255),
        "left_midside": (255, 120, 90),
        "right_midside": (190, 120, 255),
    }
    font = ImageFont.load_default()
    trail: list[tuple[float, float]] = []
    try:
        with imageio.get_writer(output_path, fps=output_fps, codec="libx264", quality=7, macro_block_size=1) as writer:
            for frame_index, frame in enumerate(reader):
                if frame_index >= max_source_frames:
                    break
                if frame_index % source_stride != 0:
                    continue
                if ts is not None and frame_index < len(ts):
                    raw_time = float(ts[frame_index])
                    row_idx = nearest_dlc_row_for_time(
                        dlc_times,
                        raw_time,
                        params["overlay_time_tolerance_sec"],
                    )
                else:
                    row_idx = frame_index if frame_index < len(frame_table) else None
                if row_idx is None:
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([5, 5, 360, 25], fill=(0, 0, 0))
                    draw.text((9, 9), f"frame {frame_index}  no nearby DLC row", fill=(255, 255, 255), font=font)
                    writer.append_data(np.asarray(image))
                    continue
                if row_idx >= len(frame_table):
                    break
                row = frame_table.iloc[row_idx]
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                for keypoint in params["keypoints"]:
                    color = colors[keypoint]
                    x = float(row[f"{keypoint}_x_clean_raw_px"])
                    y = float(row[f"{keypoint}_y_clean_raw_px"])
                    outline = bool(row[f"{keypoint}_interpolated"])
                    draw_marker(draw, x, y, color, 4, outline)
                cx = float(row["centroid_x_overlay_px"])
                cy = float(row["centroid_y_overlay_px"])
                if np.isfinite(cx) and np.isfinite(cy):
                    trail.append((cx, cy))
                    trail = trail[-params["overlay_trail_frames"] :]
                    if len(trail) > 1:
                        draw.line(trail, fill=(255, 255, 255), width=2)
                    draw_marker(draw, cx, cy, (255, 255, 255), 7, False)
                    draw_marker(draw, cx, cy, (0, 0, 0), 8, True)
                speed = row["speed_px_per_sec"]
                dt_ms = ""
                if ts is not None and frame_index < len(ts):
                    dt_ms = f"  dt={(float(row['frame_time']) - float(ts[frame_index])) * 1000:.1f}ms"
                text = f"video {frame_index}  dlc {row_idx}{dt_ms}  speed={speed:.1f}px/s"
                draw.rectangle([5, 5, 430, 25], fill=(0, 0, 0))
                draw.text((9, 9), text, fill=(255, 255, 255), font=font)
                writer.append_data(np.asarray(image))
    finally:
        reader.close()
    return str(output_path)


def nearest_dlc_row_for_time(dlc_times: np.ndarray, raw_time: float, tolerance_sec: float) -> int | None:
    if dlc_times.size == 0 or not np.isfinite(raw_time):
        return None
    idx = int(np.searchsorted(dlc_times, raw_time))
    candidates = []
    if 0 <= idx < len(dlc_times):
        candidates.append(idx)
    if 0 <= idx - 1 < len(dlc_times):
        candidates.append(idx - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(float(dlc_times[item]) - raw_time))
    if abs(float(dlc_times[best]) - raw_time) <= tolerance_sec:
        return best
    return None


def write_report(
    output_dir: Path,
    params: dict[str, Any],
    session_summary: pd.DataFrame,
    outputs: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> Path:
    report = output_dir / "dlc_locomotion_report.md"
    calibration_line = (
        "- Spatial calibration was applied to cleaned keypoints before centroid calculation."
        if params.get("spatial_calibration_status") == "applied_to_cleaned_keypoints_before_centroid"
        else "- Current coordinates are raw camera pixels; no spatial calibration was applied."
    )
    lines = [
        "# DLC Locomotion Pipeline Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Core Assumptions",
        "",
        "- Locomotion center is the non-weighted centroid of the same five trunk keypoints on every frame.",
        f"- Keypoints: `{', '.join(params['keypoints'])}`.",
        "- The centroid is computed only when all five cleaned keypoints are present.",
        "- Low-confidence/missing keypoints are interpolated independently for short gaps only.",
        "- Long gaps remain missing; smoothing is segment-wise and never crosses missing samples.",
        calibration_line,
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "## Outputs",
        "",
        "- Per-session lightweight frame tables: `sessions/*/*_dlc_locomotion.parquet`",
        "- Per-session compact centroid trajectories: `sessions/*/*_centroid_trajectory.csv.gz`",
        "- Per-session trajectory, occupancy, and velocity plots: `sessions/*/*.png`",
        "- Optional per-session overlay videos: `sessions/*/*_overlay.mp4`",
        "- Session summary: `dlc_locomotion_session_summary.csv`",
        "- Error log: `dlc_locomotion_errors.csv`",
        "",
        "## Session Summary",
        "",
    ]
    if session_summary.empty:
        lines.append("No sessions were processed.")
    else:
        speed_m = "speed_m_per_sec_median" in session_summary.columns
        speed_median_col = "speed_m_per_sec_median" if speed_m else "speed_px_per_sec_median"
        speed_p95_col = "speed_m_per_sec_p95" if speed_m else "speed_px_per_sec_p95"
        cols = [
            "session_id",
            "frames",
            "centroid_valid_fraction",
            "centroid_any_interpolated_fraction",
            speed_median_col,
            speed_p95_col,
            "coverage_fraction_bbox",
        ]
        speed_label = "m/s" if speed_m else "coord/s"
        lines.append(f"| Session | Frames | Centroid Valid | Any Interpolated | Median Speed ({speed_label}) | P95 Speed ({speed_label}) | Coverage |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in session_summary[cols].itertuples(index=False):
            values = row._asdict()
            lines.append(
                f"| {values['session_id']} | {values['frames']} | {values['centroid_valid_fraction']:.3f} | "
                f"{values['centroid_any_interpolated_fraction']:.3f} | {values[speed_median_col]:.4f} | "
                f"{values[speed_p95_col]:.4f} | {values['coverage_fraction_bbox']:.3f} |"
            )
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            lines.append(f"- `{error['session_id']}` `{error['stage']}`: {error['error']}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def default_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "keypoints": RELIABLE_KEYPOINTS,
        "confidence_threshold": args.confidence_threshold,
        "max_interpolation_gap_frames": args.max_interpolation_gap_frames,
        "median_filter_window_frames": args.median_filter_window_frames,
        "savgol_window_frames": args.savgol_window_frames,
        "coverage_bin_size_px": args.coverage_bin_size,
        "coverage_bin_size_m": args.coverage_bin_size_m,
        "min_dt_sec": args.min_dt,
        "max_keypoint_speed_px_per_sec": args.max_keypoint_speed_px_per_sec,
        "max_centroid_speed_px_per_sec": args.max_centroid_speed_px_per_sec,
        "fallback_fps": args.fallback_fps,
        "frame_width_px": args.frame_width_px,
        "frame_height_px": args.frame_height_px,
        "overlay_trail_frames": args.overlay_trail_frames,
        "timestamp_match_tolerance_sec": args.timestamp_match_tolerance_sec,
        "overlay_time_tolerance_sec": args.overlay_time_tolerance_sec,
        "spatial_calibration_status": "not_applied_raw_camera_pixels",
        "arena_edge_length_cm": args.arena_edge_length_cm,
        "physical_units_status": "not_applied",
        "analysis_window_start_sec": args.analysis_window_start_sec,
        "analysis_window_start_max_sec": args.analysis_window_start_max_sec,
        "analysis_window_duration_sec": args.analysis_window_duration_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robust DLC-only locomotion tracking tables and QC outputs.")
    parser.add_argument("root", nargs="?", default=".", type=Path, help="Data root.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/dlc_locomotion"))
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-interpolation-gap-frames", type=int, default=50)
    parser.add_argument("--median-filter-window-frames", type=int, default=5)
    parser.add_argument("--savgol-window-frames", type=int, default=21)
    parser.add_argument("--coverage-bin-size", type=float, default=25.0)
    parser.add_argument("--coverage-bin-size-m", type=float, default=None)
    parser.add_argument("--min-dt", type=float, default=0.001)
    parser.add_argument("--max-keypoint-speed-px-per-sec", type=float, default=3000.0)
    parser.add_argument("--max-centroid-speed-px-per-sec", type=float, default=2000.0)
    parser.add_argument("--fallback-fps", type=float, default=100.0)
    parser.add_argument("--frame-width-px", type=float, default=530.0)
    parser.add_argument("--frame-height-px", type=float, default=510.0)
    parser.add_argument("--calibration-json", type=Path, default=None, help="Shared calibration JSON.")
    parser.add_argument("--calibration-manifest", type=Path, default=None, help="CSV mapping session_id to calibration_json and optional group/include.")
    parser.add_argument("--arena-edge-length-cm", type=float, default=None, help="Real square arena edge length in cm.")
    parser.add_argument("--include-no-use", action="store_true", help="Include sessions under no_use_videos.")
    parser.add_argument("--session-filter", default=None, help="Regex filter applied to session_id.")
    parser.add_argument("--overlay-mode", choices=["none", "preview", "full"], default="none")
    parser.add_argument("--overlay-seconds", type=float, default=60.0)
    parser.add_argument("--overlay-fps", type=float, default=20.0)
    parser.add_argument("--overlay-trail-frames", type=int, default=40)
    parser.add_argument("--timestamp-match-tolerance-sec", type=float, default=0.03)
    parser.add_argument("--overlay-time-tolerance-sec", type=float, default=0.03)
    parser.add_argument("--analysis-window-start-sec", type=float, default=None, help="Exact recording-relative start time in seconds for analysis cropping.")
    parser.add_argument("--analysis-window-start-max-sec", type=float, default=None, help="Latest allowed recording-relative start time in seconds; starts earlier only if needed to fit the requested duration.")
    parser.add_argument("--analysis-window-duration-sec", type=float, default=None, help="Recording-relative analysis window duration in seconds.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    params = default_params(args)
    if args.arena_edge_length_cm is not None:
        print(f"cage_floor_edge_length_cm = {args.arena_edge_length_cm:g}")
        params["arena_edge_length_m"] = args.arena_edge_length_cm / 100.0
        params["physical_units_status"] = "configured_from_cage_floor_edge_length"
        if args.coverage_bin_size_m is None and (args.calibration_json is not None or args.calibration_manifest is not None):
            params["coverage_bin_size_m"] = args.coverage_bin_size * params["arena_edge_length_m"]
    calibration_params = None
    if args.calibration_json is not None:
        calibration_params = read_calibration(args.calibration_json)
        params["spatial_calibration_status"] = "applied_to_cleaned_keypoints_before_centroid"
        params["calibration_json"] = str(args.calibration_json)
        params["target_units"] = calibration_params.get("target_units")
        params["target_size"] = calibration_params.get("target_size")
    manifest = load_manifest(args.calibration_manifest, root) if args.calibration_manifest is not None else {}

    session_regex = re.compile(args.session_filter) if args.session_filter else None
    sessions = [session for session in discover_sessions(root) if "DLC" in session.files]
    if not args.include_no_use:
        sessions = [session for session in sessions if "no_use_videos" not in session.session_id]
    if session_regex:
        sessions = [session for session in sessions if session_regex.search(session.session_id)]
    if manifest:
        sessions = [session for session in sessions if session.session_id in manifest]

    summaries: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for session in sessions:
        session_name = safe_name(session.session_id)
        session_dir = output_dir / "sessions" / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_params = params.copy()
            session_calibration = calibration_params
            manifest_row = manifest.get(session.session_id)
            if manifest_row is not None:
                session_calibration = read_calibration(manifest_row.get("calibration_json"))
                session_params["spatial_calibration_status"] = "applied_to_cleaned_keypoints_before_centroid"
                session_params["calibration_json"] = manifest_row.get("calibration_json")
                session_params["target_units"] = session_calibration.get("target_units") if session_calibration else None
                session_params["target_size"] = session_calibration.get("target_size") if session_calibration else None
                if pd.notna(manifest_row.get("arena_edge_length_cm", np.nan)):
                    session_params["arena_edge_length_cm"] = float(manifest_row["arena_edge_length_cm"])
                    session_params["arena_edge_length_m"] = session_params["arena_edge_length_cm"] / 100.0
                    if session_params.get("coverage_bin_size_m") is None:
                        session_params["coverage_bin_size_m"] = args.coverage_bin_size * session_params["arena_edge_length_m"]
                if "group" in manifest_row and pd.notna(manifest_row["group"]):
                    session_params["group"] = manifest_row["group"]
            video_meta = parse_avi_header(session.files["VIDEO"]) if "VIDEO" in session.files else {}
            df = read_dlc(session.files["DLC"])
            frame_table, summary = build_frame_table(session, df, session_params, video_meta, session_calibration)
            if manifest_row is not None and "group" in manifest_row and pd.notna(manifest_row["group"]):
                summary["group"] = manifest_row["group"]
            table_path = write_frame_table(frame_table, session_dir, session_name)
            trajectory_path = write_centroid_trajectory(frame_table, session_dir, session_name)
            plot_paths: list[Path] = []
            if not args.no_plots:
                plot_bin_size = session_params.get("coverage_bin_size_m") if "centroid_x_smooth_m" in frame_table else args.coverage_bin_size
                plot_paths = make_session_plots(frame_table, session_dir, session_name, float(plot_bin_size))
            overlay_path = None
            if args.overlay_mode != "none":
                overlay_path = make_overlay_video(
                    session,
                    frame_table,
                    session_dir / "overlay.mp4",
                    session_params,
                    args.overlay_mode,
                    args.overlay_seconds,
                    args.overlay_fps,
                )
            summary["frame_table_path"] = str(table_path)
            summary["centroid_trajectory_path"] = str(trajectory_path)
            summary["overlay_video_path"] = overlay_path
            summaries.append(summary)
            outputs.append(
                {
                    "session_id": session.session_id,
                    "frame_table": str(table_path),
                    "centroid_trajectory": str(trajectory_path),
                    "calibration_json": session_params.get("calibration_json"),
                    "arena_edge_length_cm": session_params.get("arena_edge_length_cm"),
                    "plots": [str(path) for path in plot_paths],
                    "overlay_video": overlay_path,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "session_id": session.session_id,
                    "stage": "session",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    summary_df = pd.DataFrame(summaries)
    error_df = pd.DataFrame(errors)
    summary_df.to_csv(output_dir / "dlc_locomotion_session_summary.csv", index=False)
    error_df.to_csv(output_dir / "dlc_locomotion_errors.csv", index=False)
    (output_dir / "dlc_locomotion_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (output_dir / "dlc_locomotion_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    processing_log = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(["python", Path(__file__).name]),
        "cage_floor_edge_length_cm": args.arena_edge_length_cm,
        "cage_floor_edge_length_m": args.arena_edge_length_cm / 100.0 if args.arena_edge_length_cm is not None else None,
        "calibration_manifest": str(args.calibration_manifest) if args.calibration_manifest is not None else None,
        "calibration_json": str(args.calibration_json) if args.calibration_json is not None else None,
        "analysis_window_start_sec": args.analysis_window_start_sec,
        "analysis_window_start_max_sec": args.analysis_window_start_max_sec,
        "analysis_window_duration_sec": args.analysis_window_duration_sec,
        "sessions_processed": int(len(summary_df)),
        "errors": int(len(error_df)),
    }
    (output_dir / "processing_log.json").write_text(json.dumps(processing_log, indent=2), encoding="utf-8")
    report_path = write_report(output_dir, params, summary_df, outputs, errors)

    print(f"Processed {len(summary_df)} DLC sessions")
    print(f"Wrote {output_dir}")
    print(f"Report: {report_path}")
    if errors:
        print(f"Completed with {len(errors)} errors; see dlc_locomotion_errors.csv")


if __name__ == "__main__":
    main()
