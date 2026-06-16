"""
Overlay a single matched unit across time (sessions) on shared axes.

Given one matched track id, this script collects every available (non-excluded)
passive grating tuning session for that track and overlays the waveform template
from each session on a single panel, colored by session date.

Companion to plot_matched_unit_tuning_tracks.py (column grid) and
plot_track_tuning_row.py (tuning curve per day); this view stacks every session's
waveform onto one panel so probe-drift / unit stability is visible at a glance.

Example:
    python rf_recon/FreelyMovingProcessing/Grating/plot_unit_overlay_across_time.py ^
        --track 12 ^
        --base-dir "\\10.129.151.108\\xieluanlabs\\xl_cl\\sortout\\CnL42SG" ^
        --match-dir "\\10.129.151.108\\xieluanlabs\\xl_cl\\sortout\\CnL42SG\\unit_match_all_pairs\\t0.60_w0.30_a0.60_ac0.60" ^
        --exclude-session 260226
"""

from __future__ import annotations

import argparse
import pickle
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ----------------------------------------------------------------------------
# Edit these to run without command-line args (just run the file directly).
# CLI flags, if given, override these.
# ----------------------------------------------------------------------------
TRACK = [157, 159, 209]                 # matched track_id(s) to plot
EXCLUDE_SESSIONS = ["260226", "20260313"]   # sessions to skip (260226 / 20260226 / CnL42SG_20260226)
NEIGHBOR_CHANNELS = 4                   # plot this many closest neighboring channels around the best channel
CHANNEL_MODE = "neighbor"               # "shank" = all channels, "neighbor" = best + nearby channels, "best" = best only
WAVEFORM_SOURCE = "analyzer"            # "analyzer" = read curated_analyzer directly, "pkl" = read tuning pickles
WAVEFORM_CACHE_DIR = None               # None -> <out-dir>/waveform_cache
TIME_SCALE_BAR_MS = 1.0                 # paper-style waveform scale bar
AMPLITUDE_SCALE_BAR_UV = 50.0           # paper-style waveform scale bar
# ----------------------------------------------------------------------------

DEFAULT_BASE_DIR = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG")
DEFAULT_MATCH_DIR = (
    DEFAULT_BASE_DIR
    / "unit_match_all_pairs"
    / "t0.60_w0.30_a0.60_ac0.60"
)
ANIMAL_PREFIX = "CnL42SG"


def normalize_session_token(token: str, animal_prefix: str = ANIMAL_PREFIX) -> str:
    """Convert 260226, 20260226, or CnL42SG_20260226 to CnL42SG_20260226."""
    token = str(token).strip()
    if not token:
        return token

    match = re.search(r"(\d{8}|\d{6})", token)
    if not match:
        return token

    digits = match.group(1)
    if len(digits) == 6:
        digits = "20" + digits
    return f"{animal_prefix}_{digits}"


def session_date(session: str) -> pd.Timestamp:
    return pd.to_datetime(session.split("_")[1], format="%Y%m%d")


def unit_label(shank: int, unit_id: int) -> str:
    return f"shank{int(shank)}_unit{int(unit_id)}"


def parse_track_tokens(track_values: list[str] | None) -> list[int]:
    """Parse --track values like 157, "157,159", or repeated --track flags."""
    if not track_values:
        default_tracks = TRACK if isinstance(TRACK, (list, tuple, set)) else [TRACK]
        return [int(track) for track in default_tracks]

    tracks: list[int] = []
    for value in track_values:
        for token in re.split(r"[,\s]+", str(value).strip()):
            if token:
                tracks.append(int(token))
    return tracks


def find_tuning_dirs(
    base_dir: Path,
    session_cols: list[str],
    excluded_sessions: set[str],
) -> dict[str, Path]:
    """Find the newest merged tuning folder for each non-excluded session."""
    tuning_dirs: dict[str, Path] = {}
    for session in session_cols:
        if session in excluded_sessions:
            continue

        analysis_dir = base_dir / session / "passive_embedding_analysis"
        if not analysis_dir.exists():
            continue

        dirs = [
            path
            for path in analysis_dir.iterdir()
            if path.is_dir() and path.name.endswith("tuning_curves")
        ]
        if not dirs:
            continue

        merged_dirs = [path for path in dirs if "merged" in path.name]
        tuning_dirs[session] = max(merged_dirs or dirs, key=lambda path: path.stat().st_mtime)

    return tuning_dirs


def collect_track_sessions(
    track_row: pd.Series,
    session_cols: list[str],
    tuning_dirs: dict[str, Path],
) -> list[dict]:
    """Load tuning pickles for every session this track appears in."""
    shank = int(track_row["shank"])
    sessions: list[dict] = []

    for session in session_cols:
        if session not in tuning_dirs:
            continue

        matched_unit = track_row.get(session)
        if pd.isna(matched_unit):
            continue

        unit_num = int(matched_unit)
        unit_id = unit_label(shank, unit_num)
        tuning_pkl = tuning_dirs[session] / f"{unit_id}_tuning.pkl"
        if not tuning_pkl.exists():
            continue

        try:
            with open(tuning_pkl, "rb") as handle:
                data = pickle.load(handle)
        except Exception as exc:  # Keep going if one file is corrupt.
            print(f"  warn: failed to load {tuning_pkl}: {exc}")
            continue

        sessions.append(
            {
                "session": session,
                "date": session_date(session),
                "unit_id": unit_id,
                "unit_num": unit_num,
                "pkl": str(tuning_pkl),
                "data": data,
            }
        )

    return sorted(sessions, key=lambda item: item["date"])


def find_unit_id(unit_ids, requested_unit: int):
    """Match CSV unit ids to analyzer unit ids without assuming int vs str dtype."""
    for unit_id in unit_ids:
        if str(unit_id) == str(requested_unit):
            return unit_id
    return None


def analyzer_path_for_session(base_dir: Path, session: str) -> Path:
    return base_dir / session / "curated_analyzer"


def waveform_cache_path(cache_dir: Path, session: str) -> Path:
    return cache_dir / f"{session}_waveforms.pkl"


def load_session_waveform_cache(
    session: str,
    cache_dir: Path,
    waveform_cache: dict[str, dict],
    refresh_cache: bool,
) -> dict:
    if session in waveform_cache:
        return waveform_cache[session]

    cache = {"session": session, "units": {}}
    cache_path = waveform_cache_path(cache_dir, session)
    if cache_path.exists() and not refresh_cache:
        try:
            with open(cache_path, "rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, dict) and isinstance(loaded.get("units"), dict):
                cache = loaded
        except Exception as exc:
            print(f"  warn: failed to load waveform cache {cache_path}: {exc}")

    waveform_cache[session] = cache
    return cache


def save_session_waveform_cache(session: str, cache_dir: Path, cache: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = waveform_cache_path(cache_dir, session)
    try:
        with open(cache_path, "wb") as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        print(f"  warn: failed to save waveform cache {cache_path}: {exc}")


def load_cached_analyzer(analyzer_path: Path, analyzer_cache: dict[Path, object]):
    from spikeinterface import load_sorting_analyzer

    analyzer_path = analyzer_path.resolve()
    if analyzer_path not in analyzer_cache:
        analyzer_cache[analyzer_path] = load_sorting_analyzer(analyzer_path)
    return analyzer_cache[analyzer_path]


def waveform_info_from_analyzer(
    analyzer_path: Path,
    unit_num: int,
    shank: int,
    analyzer_cache: dict[Path, object],
) -> dict:
    """Read a unit's waveform template directly from a SpikeInterface analyzer."""
    sorting_analyzer = load_cached_analyzer(analyzer_path, analyzer_cache)
    unit_ids = list(sorting_analyzer.unit_ids)
    analyzer_unit_id = find_unit_id(unit_ids, unit_num)
    if analyzer_unit_id is None:
        raise ValueError(f"unit {unit_num} not found in {analyzer_path}")

    templates_ext = sorting_analyzer.get_extension("templates")
    if templates_ext is None:
        raise ValueError(f"templates extension not found in {analyzer_path}")

    unit_index = unit_ids.index(analyzer_unit_id)
    template = np.asarray(templates_ext.get_data()[unit_index], dtype=float)
    if template.ndim != 2 or not template.size:
        raise ValueError(f"empty waveform template for unit {unit_num} in {analyzer_path}")

    best_channel = int(np.argmax(np.ptp(template, axis=0)))
    fs_wf = float(sorting_analyzer.sampling_frequency)
    n_samples = template.shape[0]
    time_ms = np.arange(n_samples) / fs_wf * 1000.0 - (n_samples / 2) / fs_wf * 1000.0

    locs = sorting_analyzer.get_channel_locations()
    channel_location = locs[best_channel].tolist() if locs is not None else None
    channel_locations = locs.tolist() if locs is not None else None

    channel_ids = list(getattr(sorting_analyzer, "channel_ids", [])) or None
    channel_groups = None
    recording = getattr(sorting_analyzer, "recording", None)
    if recording is not None:
        try:
            groups = recording.get_property("group")
            channel_groups = groups.tolist() if hasattr(groups, "tolist") else list(groups)
        except Exception:
            channel_groups = None

    return {
        "original_unit_id": int(unit_num),
        "shank": int(shank),
        "best_channel": best_channel,
        "channel_location_um": channel_location,
        "waveform_template": template[:, best_channel].tolist(),
        "waveform_template_all_channels": template.tolist(),
        "waveform_t_ms": time_ms.tolist(),
        "channel_locations_um": channel_locations,
        "waveform_channel_ids": channel_ids,
        "waveform_channel_groups": channel_groups,
        "sorting_folder": str(analyzer_path),
    }


def collect_track_sessions_from_analyzers(
    track_row: pd.Series,
    session_cols: list[str],
    base_dir: Path,
    excluded_sessions: set[str],
    analyzer_cache: dict[Path, object],
    waveform_cache_dir: Path,
    waveform_cache: dict[str, dict],
    refresh_waveform_cache: bool,
) -> list[dict]:
    """Read waveform templates directly from each session's curated_analyzer."""
    shank = int(track_row["shank"])
    sessions: list[dict] = []

    for session in session_cols:
        if session in excluded_sessions:
            continue

        matched_unit = track_row.get(session)
        if pd.isna(matched_unit):
            continue

        unit_num = int(matched_unit)
        unit_key = str(unit_num)
        analyzer_path = analyzer_path_for_session(base_dir, session)
        session_cache = load_session_waveform_cache(
            session,
            waveform_cache_dir,
            waveform_cache,
            refresh_waveform_cache,
        )

        if unit_key in session_cache["units"]:
            unit_info = session_cache["units"][unit_key]
        else:
            if not analyzer_path.exists():
                print(f"  warn: curated_analyzer not found for {session}: {analyzer_path}")
                continue

            try:
                unit_info = waveform_info_from_analyzer(analyzer_path, unit_num, shank, analyzer_cache)
            except Exception as exc:
                print(f"  warn: failed to load waveform for {session} unit {unit_num}: {exc}")
                continue

            session_cache["units"][unit_key] = unit_info
            save_session_waveform_cache(session, waveform_cache_dir, session_cache)

        sessions.append(
            {
                "session": session,
                "date": session_date(session),
                "unit_id": unit_label(shank, unit_num),
                "unit_num": unit_num,
                "analyzer": str(analyzer_path),
                "data": {"unit_info": unit_info},
            }
        )

    return sorted(sessions, key=lambda item: item["date"])


def _waveform_matrix(unit_info: dict) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Return waveform as samples x channels, time axis, and best channel index."""
    time_ms = np.asarray(unit_info.get("waveform_t_ms", []), dtype=float)
    waveforms = np.asarray(unit_info.get("waveform_template_all_channels", []), dtype=float)
    best_channel = unit_info.get("best_channel")
    best_channel = int(best_channel) if best_channel is not None else None

    if waveforms.ndim == 2 and waveforms.size:
        if time_ms.size and waveforms.shape[0] != time_ms.size and waveforms.shape[1] == time_ms.size:
            waveforms = waveforms.T
        if not time_ms.size or time_ms.size != waveforms.shape[0]:
            time_ms = np.arange(waveforms.shape[0])
        return waveforms, time_ms, best_channel

    waveform = np.asarray(unit_info.get("waveform_template", []), dtype=float)
    if waveform.size:
        if time_ms.size != waveform.size:
            time_ms = np.arange(waveform.size)
        return waveform[:, None], time_ms, 0

    return np.empty((0, 0)), np.empty(0), None


def _channel_order_from_locations(unit_info: dict, n_channels: int, channel_indices: list[int] | None = None) -> list[int]:
    if channel_indices is None:
        channel_indices = list(range(n_channels))

    locs = np.asarray(unit_info.get("channel_locations_um", []), dtype=float)
    if locs.ndim == 2 and locs.shape[0] == n_channels:
        channel_indices_arr = np.asarray(channel_indices, dtype=int)
        loc_subset = locs[channel_indices_arr]
        order = np.lexsort((loc_subset[:, 0], loc_subset[:, 1]))
        return [int(channel_indices_arr[idx]) for idx in order]
    return list(channel_indices)


def _shank_channel_indices(unit_info: dict, n_channels: int) -> list[int]:
    channel_groups = unit_info.get("waveform_channel_groups")
    shank = unit_info.get("shank")
    if channel_groups is None or shank is None:
        return _channel_order_from_locations(unit_info, n_channels)

    if len(channel_groups) != n_channels:
        return _channel_order_from_locations(unit_info, n_channels)

    shank_str = str(shank)
    matching = [idx for idx, group in enumerate(channel_groups) if str(group) == shank_str]
    if not matching:
        return _channel_order_from_locations(unit_info, n_channels)
    return _channel_order_from_locations(unit_info, n_channels, matching)


def _channel_indices(
    unit_info: dict,
    waveforms: np.ndarray,
    channel_mode: str,
    neighbor_channels: int,
) -> list[int]:
    n_channels = waveforms.shape[1]
    best_channel = unit_info.get("best_channel")
    best_channel = int(best_channel) if best_channel is not None else None
    if best_channel is None or not 0 <= best_channel < n_channels:
        return [0]

    if channel_mode == "shank":
        return _shank_channel_indices(unit_info, n_channels)

    if channel_mode == "best" or neighbor_channels <= 0:
        return [best_channel]

    locs = np.asarray(unit_info.get("channel_locations_um", []), dtype=float)
    if locs.ndim == 2 and locs.shape[0] == n_channels:
        distances = np.linalg.norm(locs - locs[best_channel], axis=1)
        keep = np.argsort(distances)[: min(n_channels, neighbor_channels + 1)]
        return sorted(int(idx) for idx in keep)

    before = neighbor_channels // 2
    after = neighbor_channels - before
    lo = max(0, best_channel - before)
    hi = min(n_channels, best_channel + after + 1)
    while hi - lo < min(n_channels, neighbor_channels + 1):
        if lo > 0:
            lo -= 1
        elif hi < n_channels:
            hi += 1
        else:
            break
    return list(range(lo, hi))


def _collect_waveform_plot_items(
    sessions: list[dict],
    colors: np.ndarray,
    channel_mode: str,
    neighbor_channels: int,
) -> list[dict]:
    plot_items: list[dict] = []
    for color, item in zip(colors, sessions):
        unit_info = item["data"].get("unit_info", {})
        waveforms, time_ms, best_channel = _waveform_matrix(unit_info)
        if not waveforms.size:
            continue
        channel_indices = _channel_indices(unit_info, waveforms, channel_mode, neighbor_channels)
        plot_items.append(
            {
                "session": item["session"],
                "date": item["date"],
                "unit_id": item["unit_id"],
                "color": color,
                "unit_info": unit_info,
                "waveforms": waveforms,
                "time_ms": time_ms,
                "best_channel": best_channel,
                "channel_indices": channel_indices,
            }
        )
    return plot_items


def _finite_ptp(values: np.ndarray) -> float:
    if not values.size or np.all(~np.isfinite(values)):
        return 0.0
    return float(np.nanmax(values) - np.nanmin(values))


def _median_site_spacing(locs: np.ndarray) -> float:
    if locs.ndim != 2 or locs.shape[0] < 2:
        return 20.0
    distances = []
    for idx, loc in enumerate(locs):
        delta = locs[np.arange(locs.shape[0]) != idx] - loc
        nearest = np.min(np.linalg.norm(delta, axis=1))
        if np.isfinite(nearest) and nearest > 0:
            distances.append(float(nearest))
    return float(np.median(distances)) if distances else 20.0


def _reference_channel_layout(plot_items: list[dict], channel_mode: str, neighbor_channels: int) -> tuple[np.ndarray, list[int]]:
    for item in plot_items:
        locs = np.asarray(item["unit_info"].get("channel_locations_um", []), dtype=float)
        if locs.ndim == 2 and locs.shape[0] == item["waveforms"].shape[1]:
            channels = _channel_indices(item["unit_info"], item["waveforms"], channel_mode, neighbor_channels)
            return locs, channels
    return np.empty((0, 2)), []


def _plot_scale_bar(
    ax: plt.Axes,
    x0: float,
    y0: float,
    time_scale_um_per_ms: float,
    amplitude_scale_um_per_uv: float,
    color: str = "0.05",
) -> None:
    x1 = x0 + TIME_SCALE_BAR_MS * time_scale_um_per_ms
    y1 = y0 + AMPLITUDE_SCALE_BAR_UV * amplitude_scale_um_per_uv
    ax.plot([x0, x1], [y0, y0], color=color, lw=1.8, solid_capstyle="butt")
    ax.plot([x0, x0], [y0, y1], color=color, lw=1.8, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y0 - 0.035 * abs(y0 if y0 else 1), f"{TIME_SCALE_BAR_MS:g} ms",
            ha="center", va="top", fontsize=8, color=color)
    ax.text(x0 - 0.02 * abs(x0 if x0 else 1), (y0 + y1) / 2, f"{AMPLITUDE_SCALE_BAR_UV:g} uV",
            ha="right", va="center", fontsize=8, color=color, rotation=90)


def _depth_ordered_channels(unit_info: dict, channel_indices: list[int]) -> list[int]:
    """Order channels top->bottom by physical y (dorsal up), then x.

    The neighbor/shank selection returns channels in channel-index order, which
    does NOT track probe geometry (e.g. index 190 sits between index 167 and 187
    physically). Stacking must follow real site positions, not index order.
    """
    locs = np.asarray(unit_info.get("channel_locations_um", []), dtype=float)
    if locs.ndim != 2 or not channel_indices or max(channel_indices) >= locs.shape[0]:
        return list(channel_indices)
    return sorted(channel_indices, key=lambda i: (-float(locs[i, 1]), float(locs[i, 0])))


def plot_waveform_stack_from_items(
    ax: plt.Axes,
    plot_items: list[dict],
    channel_mode: str,
) -> None:
    max_amplitude = 0.0
    n_rows = 1
    for plot_item in plot_items:
        waveforms = plot_item["waveforms"]
        channel_indices = plot_item["channel_indices"]
        if channel_indices:
            max_amplitude = max(max_amplitude, float(np.nanmax(np.ptp(waveforms[:, channel_indices], axis=0))))
            n_rows = max(n_rows, len(channel_indices))

    y_step = max(max_amplitude * 1.35, 1.0)
    for plot_item in plot_items:
        color = plot_item["color"]
        waveforms = plot_item["waveforms"]
        time_ms = plot_item["time_ms"]
        best_channel = plot_item["best_channel"]
        channel_indices = _depth_ordered_channels(plot_item["unit_info"], plot_item["channel_indices"])
        best_rank = channel_indices.index(best_channel) if best_channel in channel_indices else 0
        for channel_idx in channel_indices:
            offset_rank = channel_indices.index(channel_idx) - best_rank
            offset = -offset_rank * y_step
            lw = 1.6 if channel_idx == best_channel else 0.9
            alpha = 0.9 if channel_idx == best_channel else (0.32 if channel_mode == "shank" else 0.48)
            ax.plot(time_ms, waveforms[:, channel_idx] + offset, color=color, lw=lw, alpha=alpha)

    # Constrain the axes box so the waveform isn't stretched horizontally across
    # the wide panel. ~1 row -> square; more rows -> taller (each row stays ~1:1).
    ax.set_box_aspect(min(max(n_rows * 0.45 + 0.45, 1.0), 2.0))

    title_suffix = {
        "best": "best channel",
        "neighbor": "best channel + neighbors",
        "shank": "whole-shank channel stack",
    }.get(channel_mode, channel_mode)
    ax.set_title(f"Waveform overlay ({title_suffix})", fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_waveform_overlay(
    ax: plt.Axes,
    sessions: list[dict],
    colors: np.ndarray,
    channel_mode: str,
    neighbor_channels: int,
) -> None:
    plot_items = _collect_waveform_plot_items(sessions, colors, channel_mode, neighbor_channels)
    plot_waveform_stack_from_items(ax, plot_items, channel_mode)


def plot_probe_site_overlay(
    ax: plt.Axes,
    plot_items: list[dict],
    channel_mode: str,
    neighbor_channels: int,
) -> tuple[float, float]:
    locs, ref_channels = _reference_channel_layout(plot_items, channel_mode, neighbor_channels)
    if locs.size == 0 or not ref_channels:
        plot_waveform_stack_from_items(ax, plot_items, channel_mode)
        return 1.0, 1.0

    ref_channels = [idx for idx in ref_channels if 0 <= idx < locs.shape[0]]
    ref_locs = locs[ref_channels]
    spacing = _median_site_spacing(ref_locs)

    all_wf = []
    all_times = []
    for item in plot_items:
        valid = [idx for idx in ref_channels if idx < item["waveforms"].shape[1]]
        if valid:
            all_wf.append(item["waveforms"][:, valid])
            all_times.append(item["time_ms"])
    max_ptp = max((_finite_ptp(wf) for wf in all_wf), default=1.0)
    time_span = max((_finite_ptp(time) for time in all_times), default=TIME_SCALE_BAR_MS)

    amplitude_scale = min(max(spacing * 0.32 / max(max_ptp, 1e-9), 0.02), 0.6)
    time_scale = min(max(spacing * 0.75 / max(time_span, 1e-9), 2.0), 35.0)
    dot_size = max(26.0, min(95.0, spacing * 2.6))

    ax.scatter(ref_locs[:, 0], ref_locs[:, 1], s=dot_size, color="0.83", edgecolors="none", zorder=0)
    for item in plot_items:
        color = item["color"]
        waveforms = item["waveforms"]
        time_ms = item["time_ms"]
        best_channel = item["best_channel"]
        t_centered = time_ms - np.nanmean(time_ms)
        for channel_idx in ref_channels:
            if channel_idx >= waveforms.shape[1]:
                continue
            x0, y0 = locs[channel_idx]
            trace = waveforms[:, channel_idx]
            trace = trace - np.nanmean(trace)
            lw = 1.2 if channel_idx == best_channel else 0.75
            alpha = 0.92 if channel_idx == best_channel else 0.45
            ax.plot(
                x0 + t_centered * time_scale,
                y0 + trace * amplitude_scale,
                color=color,
                lw=lw,
                alpha=alpha,
                zorder=2 if channel_idx == best_channel else 1,
            )

    x_min = float(np.nanmin(ref_locs[:, 0]) - spacing * 2.2)
    x_max = float(np.nanmax(ref_locs[:, 0]) + spacing * 2.2)
    y_min = float(np.nanmin(ref_locs[:, 1]) - spacing * 2.4)
    y_max = float(np.nanmax(ref_locs[:, 1]) + spacing * 2.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Superimposed waveforms at probe sites", fontsize=11, fontweight="bold")
    _plot_scale_bar(ax, x_min + spacing * 0.35, y_min + spacing * 0.75, time_scale, amplitude_scale)
    return time_scale, amplitude_scale


def plot_best_channel_inset(ax: plt.Axes, plot_items: list[dict]) -> None:
    max_ptp = 0.0
    for item in plot_items:
        best_channel = item["best_channel"]
        if best_channel is None or best_channel >= item["waveforms"].shape[1]:
            continue
        max_ptp = max(max_ptp, _finite_ptp(item["waveforms"][:, best_channel]))

    for item in plot_items:
        best_channel = item["best_channel"]
        if best_channel is None or best_channel >= item["waveforms"].shape[1]:
            continue
        time_ms = item["time_ms"]
        trace = item["waveforms"][:, best_channel]
        ax.plot(time_ms - np.nanmean(time_ms), trace - np.nanmean(trace), color=item["color"], lw=1.25, alpha=0.9)

    ax.axhline(0, color="0.88", lw=0.6, zorder=0)
    ax.axvline(0, color="0.88", lw=0.6, zorder=0)
    ax.set_title(f"Superimposed Days 0-{len(plot_items) - 1}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("uV", fontsize=8)
    ax.tick_params(labelsize=7, length=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ylim = max(max_ptp * 0.75, AMPLITUDE_SCALE_BAR_UV)
    ax.set_ylim(-ylim, ylim)


def plot_session_color_key(ax: plt.Axes, plot_items: list[dict]) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, "Day", fontsize=7.5, fontweight="bold", va="top")
    start_date = plot_items[0]["date"]
    y = 0.88
    for item in plot_items:
        day = int((item["date"] - start_date).days)
        ax.text(0.02, y, str(day), color=item["color"], fontsize=7, va="center")
        ax.plot([0.42, 0.88], [y, y], color=item["color"], lw=1.0, transform=ax.transAxes, clip_on=False)
        y -= 0.8 / max(len(plot_items), 1)


def max_plotted_channels(sessions: list[dict], channel_mode: str, neighbor_channels: int) -> int:
    max_channels = 1
    for item in sessions:
        unit_info = item["data"].get("unit_info", {})
        waveforms, _, _ = _waveform_matrix(unit_info)
        if not waveforms.size:
            continue
        max_channels = max(max_channels, len(_channel_indices(unit_info, waveforms, channel_mode, neighbor_channels)))
    return max_channels


def plot_unit_overlay(
    track_row: pd.Series,
    sessions: list[dict],
    out_dir: Path,
    repro_stamp: str,
    channel_mode: str,
    neighbor_channels: int,
) -> Path:
    track_id = int(track_row["track_id"])
    shank = int(track_row["shank"])
    n_sessions = len(sessions)

    n_channels = max_plotted_channels(sessions, channel_mode, neighbor_channels)
    fig_height = max(6.4, min(15.0, 3.6 + 0.30 * n_channels))
    fig = plt.figure(figsize=(6.6, fig_height))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=(0.42, 2.6, 1.7),
        wspace=0.04,
    )
    key_ax = fig.add_subplot(gs[:, 0])
    probe_ax = fig.add_subplot(gs[:, 1])
    meta_ax = fig.add_subplot(gs[:, 2])
    fig.suptitle(
        f"{ANIMAL_PREFIX} matched track {track_id} | shank {shank}\n"
        f"{n_sessions} sessions "
        f"({sessions[0]['date'].strftime('%m-%d')} -> {sessions[-1]['date'].strftime('%m-%d')}) | "
        f"mean score {float(track_row.get('mean_score', np.nan)):.3f}",
        fontsize=10,
        fontweight="bold",
        y=0.997,
    )

    # Color each session by its date so drift over time is readable.
    dates = np.array([item["date"].value for item in sessions], dtype=float)
    norm = Normalize(vmin=dates.min(), vmax=dates.max() if dates.max() > dates.min() else dates.min() + 1)
    # Jet: dark blue -> cyan -> green -> yellow -> orange -> red across dates.
    cmap = plt.cm.jet
    colors = cmap(norm(dates))

    plot_items = _collect_waveform_plot_items(sessions, colors, channel_mode, neighbor_channels)
    if not plot_items:
        raise ValueError(
            "No waveform templates were available for the selected sessions. "
            "Try --waveform-source analyzer or refresh the tuning pickles."
        )
    plot_session_color_key(key_ax, plot_items)
    plot_waveform_stack_from_items(probe_ax, plot_items, channel_mode)

    meta_ax.axis("off")
    meta_ax.text(
        0.0,
        1.0,
        "Matched units\n"
        + "\n".join(
            f"Day {int((item['date'] - sessions[0]['date']).days):>2}: {item['unit_id']}"
            for item in sessions
        ),
        fontsize=6.2,
        va="top",
        family="monospace",
    )

    # Colorbar mapping color -> session date.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=probe_ax, shrink=0.46, pad=0.012, fraction=0.025)
    tick_dates = [sessions[0]["date"], sessions[-1]["date"]]
    if n_sessions >= 3:
        tick_dates.insert(1, sessions[n_sessions // 2]["date"])
    cbar.set_ticks([d.value for d in tick_dates])
    cbar.set_ticklabels([d.strftime("%m-%d") for d in tick_dates], fontsize=6.5)
    cbar.set_label("Date", fontsize=7, fontweight="bold")

    # Reproducibility stamp (per project convention: how to regenerate this figure).
    fig.text(0.005, 0.004, repro_stamp, fontsize=5.5, color="0.4", va="bottom", ha="left")

    fig.subplots_adjust(left=0.035, right=0.965, top=0.91, bottom=0.085)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_shank" if channel_mode == "shank" else f"_neighbor{neighbor_channels}" if channel_mode == "neighbor" else "_best"
    png_path = out_dir / f"track_{track_id:03d}_shank{shank}_overlay_across_time{suffix}.png"
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path


def build_repro_stamp(
    track: int,
    base_dir: Path,
    match_dir: Path,
    excluded: set[str],
    sessions: list[dict],
    waveform_source: str,
    channel_mode: str,
    neighbor_channels: int,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    excl = ",".join(sorted(excluded)) if excluded else "none"
    used = ",".join(item["session"].split("_")[1] for item in sessions)
    return (
        f"plot_unit_overlay_across_time.py --track {track} "
        f"--base-dir {base_dir} --match-dir {match_dir} "
        f"--exclude-session {excl} --waveform-source {waveform_source} "
        f"--channel-mode {channel_mode} --neighbor-channels {neighbor_channels} | "
        f"sessions_used: {used} | generated {timestamp}"
    )


def generate(
    track: int,
    base_dir: Path,
    match_dir: Path,
    out_dir: Path,
    exclude_sessions: list[str],
    waveform_source: str,
    channel_mode: str,
    neighbor_channels: int,
    analyzer_cache: dict[Path, object] | None = None,
    waveform_cache_dir: Path | None = None,
    waveform_cache: dict[str, dict] | None = None,
    refresh_waveform_cache: bool = False,
) -> Path:
    warnings.filterwarnings("ignore", category=UserWarning)

    tracks = pd.read_csv(match_dir / "unit_tracks.csv")
    if "track_id" not in tracks.columns:
        raise ValueError(f"unit_tracks.csv has no track_id column; columns: {list(tracks.columns)}")

    match = tracks[tracks["track_id"] == track]
    if match.empty:
        available = ", ".join(str(t) for t in tracks["track_id"].tolist()[:30])
        raise ValueError(f"track_id {track} not found. First available ids: {available} ...")
    track_row = match.iloc[0]

    session_cols = [col for col in tracks.columns if re.match(rf"{ANIMAL_PREFIX}_\d{{8}}$", col)]
    excluded = {normalize_session_token(session) for session in exclude_sessions}

    if waveform_source == "analyzer":
        if analyzer_cache is None:
            analyzer_cache = {}
        if waveform_cache is None:
            waveform_cache = {}
        if waveform_cache_dir is None:
            waveform_cache_dir = out_dir / "waveform_cache"
        sessions = collect_track_sessions_from_analyzers(
            track_row,
            session_cols,
            base_dir,
            excluded,
            analyzer_cache,
            waveform_cache_dir,
            waveform_cache,
            refresh_waveform_cache,
        )
    else:
        tuning_dirs = find_tuning_dirs(base_dir, session_cols, excluded)
        sessions = collect_track_sessions(track_row, session_cols, tuning_dirs)

    if not sessions:
        raise ValueError(
            f"track {track} (shank {int(track_row['shank'])}) has no loadable tuning "
            f"waveforms in any non-excluded session via {waveform_source}."
        )

    repro_stamp = build_repro_stamp(
        track,
        base_dir,
        match_dir,
        excluded,
        sessions,
        waveform_source,
        channel_mode,
        neighbor_channels,
    )
    png_path = plot_unit_overlay(track_row, sessions, out_dir, repro_stamp, channel_mode, neighbor_channels)

    print(f"track: {track} (shank {int(track_row['shank'])})")
    print(f"sessions_overlaid: {len(sessions)} -> "
          f"{', '.join(item['session'].split('_')[1] for item in sessions)}")
    print(f"matched_units: {', '.join(item['unit_id'] for item in sessions)}")
    print(f"waveform_source: {waveform_source}")
    print(f"excluded_sessions: {', '.join(sorted(excluded)) if excluded else 'none'}")
    print(f"figure: {png_path}")
    return png_path


def generate_many(
    tracks: list[int],
    base_dir: Path,
    match_dir: Path,
    out_dir: Path,
    exclude_sessions: list[str],
    waveform_source: str,
    channel_mode: str,
    neighbor_channels: int,
    waveform_cache_dir: Path | None,
    refresh_waveform_cache: bool,
) -> list[Path]:
    png_paths: list[Path] = []
    failures: list[tuple[int, str]] = []
    analyzer_cache: dict[Path, object] = {}
    waveform_cache: dict[str, dict] = {}
    if waveform_cache_dir is None:
        waveform_cache_dir = out_dir / "waveform_cache"

    for track in tracks:
        print("\n" + "=" * 72)
        print(f"Generating waveform overlay for track {track}")
        print("=" * 72)
        try:
            png_paths.append(
                generate(
                    track=track,
                    base_dir=base_dir,
                    match_dir=match_dir,
                    out_dir=out_dir,
                    exclude_sessions=exclude_sessions,
                    waveform_source=waveform_source,
                    channel_mode=channel_mode,
                    neighbor_channels=neighbor_channels,
                    analyzer_cache=analyzer_cache,
                    waveform_cache_dir=waveform_cache_dir,
                    waveform_cache=waveform_cache,
                    refresh_waveform_cache=refresh_waveform_cache,
                )
            )
        except Exception as exc:
            failures.append((track, str(exc)))
            print(f"  warn: track {track} failed: {exc}")

    if failures and not png_paths:
        details = "; ".join(f"track {track}: {err}" for track, err in failures)
        raise ValueError(f"No tracks plotted. Failures: {details}")

    if failures:
        print("\nTracks skipped:")
        for track, err in failures:
            print(f"  track {track}: {err}")

    if analyzer_cache:
        print(f"\ncurated_analyzers opened: {len(analyzer_cache)}")
    if waveform_source == "analyzer":
        print(f"waveform cache dir: {waveform_cache_dir}")

    return png_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        action="append",
        default=None,
        help=(
            "Matched track_id(s) to plot. Accepts --track 157, --track 157,159, "
            "or repeated flags. Defaults to TRACK in the script."
        ),
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <match-dir>/unit_overlay_across_time.",
    )
    parser.add_argument(
        "--exclude-session",
        action="append",
        default=list(EXCLUDE_SESSIONS),
        help=(
            "Session to exclude. Accepts forms like 260226, 20260226, or "
            "CnL42SG_20260226. Can be repeated."
        ),
    )
    parser.add_argument(
        "--channel-mode",
        choices=("shank", "neighbor", "best"),
        default=CHANNEL_MODE,
        help="Which channels to overlay: whole shank, nearby channels, or best channel only.",
    )
    parser.add_argument(
        "--waveform-source",
        choices=("analyzer", "pkl"),
        default=WAVEFORM_SOURCE,
        help="Read waveforms directly from curated_analyzer or from existing tuning pickles.",
    )
    parser.add_argument(
        "--waveform-cache-dir",
        type=Path,
        default=WAVEFORM_CACHE_DIR,
        help="Directory for cached analyzer waveform pickles. Defaults to <out-dir>/waveform_cache.",
    )
    parser.add_argument(
        "--refresh-waveform-cache",
        action="store_true",
        help="Re-read waveforms from curated_analyzer and overwrite waveform cache pickles.",
    )
    parser.add_argument(
        "--neighbor-channels",
        type=int,
        default=NEIGHBOR_CHANNELS,
        help="Total number of neighboring channels to include around the best channel when --channel-mode neighbor.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = args.out_dir or (args.match_dir / "unit_overlay_across_time")
    tracks = parse_track_tokens(args.track)
    png_paths = generate_many(
        tracks=tracks,
        base_dir=args.base_dir,
        match_dir=args.match_dir,
        out_dir=out_dir,
        exclude_sessions=args.exclude_session,
        waveform_source=args.waveform_source,
        channel_mode=args.channel_mode,
        neighbor_channels=max(0, args.neighbor_channels),
        waveform_cache_dir=args.waveform_cache_dir,
        refresh_waveform_cache=args.refresh_waveform_cache,
    )
    print("\nGenerated figures:")
    for png_path in png_paths:
        print(f"  {png_path}")


if __name__ == "__main__":
    main()
