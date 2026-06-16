"""
Plot one matched unit's tuning curve across days as a single row of panels.

One column per (non-excluded) passive grating session, ordered by date, with a
shared x and y axis so drift in gain/preference is directly comparable. Designed
to drop straight onto a slide: clean panels, no background grid, large fonts.

Tuning curves are computed directly from per-trial spike times in each day's
merged grating pickle (the same source the raster/PSTH plots use), via the
shared grating_utils.tuning_curve_from_spikes. This yields a window-matched SEM
(shaded band) since the mean and SEM use the same display window. The OSI /
preferred-orientation formulas match GratingTuningCurve.calculate_tuning_curves.

Example:
    python rf_recon/FreelyMovingProcessing/Grating/plot_track_tuning_row.py ^
        --track 12 ^
        --exclude-session 260226 --exclude-session 20260313 ^
        --window 0.05 0.5
"""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grating_utils import tuning_curve_from_spikes
from plot_matched_unit_raster_psth_by_day import (
    ANIMAL_PREFIX,
    DEFAULT_BASE_DIR,
    DEFAULT_MATCH_DIR,
    collect_track_sessions,
    find_merged_pkls,
    load_session_cache,
    normalize_session_token,
)

# ----------------------------------------------------------------------------
# Edit these to run without command-line args (just run the file directly).
# CLI flags, if given, override these.
# ----------------------------------------------------------------------------
TRACK = 157                                # matched track_id to plot
EXCLUDE_SESSIONS = ["260226", "20260313", "20260305"]    # sessions to skip
TUNING_WINDOW = (0.05, 0.5)                  # (start, end) sec post-stim averaged into the curve
# ----------------------------------------------------------------------------

CURVE_COLOR = "black"
ERR_COLOR = "0.6"
STAR_COLOR = "#D7263D"
OSI_COLOR = "#2E86AB"
PREF_COLOR = "#D7263D"


def compute_session_metrics(sessions: list[dict], window: tuple[float, float]) -> list[dict]:
    """Compute the tuning curve + scalar stats once per session, in date order.

    Curves are computed directly from per-trial spike times (from the per-day
    merged grating pickle), so the mean and its SEM share the display window.
    """
    metrics = []
    for item in sessions:
        curve = tuning_curve_from_spikes(item["ori_spikes"], window)
        metrics.append(
            {
                "date": item["date"],
                "session": item["session"],
                "curve": curve,
                "sem_rates": np.asarray(curve["sem_rates"], dtype=float),
                "osi": float(curve.get("osi", np.nan)),
                "preferred": float(curve.get("preferred_orientation_deg", np.nan)),
            }
        )
    return metrics


def plot_tuning_row(
    track_row: pd.Series,
    metrics: list[dict],
    window: tuple[float, float],
    out_dir: Path,
    repro_stamp: str,
) -> Path:
    track_id = int(track_row["track_id"])
    shank = int(track_row["shank"])
    n = len(metrics)

    # Slide-friendly proportions: wide-ish panels, generous height, big fonts.
    panel_w = 2.4
    fig_width = max(6.0, panel_w * n)
    fig, axes = plt.subplots(
        1, n, figsize=(fig_width, 3.8), squeeze=False, sharex=True, sharey=True
    )
    axes = axes[0]

    y_upper = 1.08  # shared y top, grown to fit the error band
    for ax, item in zip(axes, metrics):
        curve = item["curve"]
        orientations = curve["orientations"]
        mean_rates = curve["mean_rates"]
        sem_rates = item["sem_rates"]

        if orientations.size and mean_rates.size:
            # Normalize each day's curve to its own peak so shape / preferred
            # orientation are comparable across days regardless of gain.
            peak = float(np.nanmax(mean_rates))
            scale = peak if peak > 0 else 1.0
            norm_rates = mean_rates / scale
            norm_sem = sem_rates / scale

            # Shaded SEM band (gray), then the tuning curve as a black line.
            if np.isfinite(norm_sem).any():
                ax.fill_between(
                    orientations, norm_rates - norm_sem, norm_rates + norm_sem,
                    color=ERR_COLOR, alpha=0.35, lw=0, zorder=1,
                )
                y_upper = max(y_upper, float(np.nanmax(norm_rates + norm_sem)))
            ax.plot(orientations, norm_rates, marker="o", ms=5, lw=2.0,
                    color=CURVE_COLOR, zorder=2)

            preferred = item["preferred"]
            if np.isfinite(preferred):
                y_star = np.interp(
                    preferred, orientations, norm_rates,
                    left=norm_rates[0], right=norm_rates[-1],
                )
                ax.plot(preferred, y_star, marker="*", ms=14, color=STAR_COLOR, zorder=5)

        ax.set_title(item["date"].strftime("%y-%m-%d"), fontsize=12, fontweight="bold")
        ax.set_xlim(-5, 185)
        ax.set_xticks([0, 90, 180])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0, y_upper * 1.03)
    axes[0].set_ylabel("Neural response", fontsize=12, fontweight="bold")
    fig.supxlabel("Orientation (°)", fontsize=12, fontweight="bold", y=0.07)

    fig.suptitle(
        f"{ANIMAL_PREFIX} track {track_id} | shank {shank} | tuning across "
        f"{n} sessions | window {window[0]:g}–{window[1]:g}s",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Reproducibility stamp (per project convention: how to regenerate this figure).
    fig.text(0.005, 0.002, repro_stamp, fontsize=5.5, color="0.45", va="bottom", ha="left")

    fig.tight_layout(rect=(0, 0.10, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"track_{track_id:03d}_shank{shank}_tuning_row.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_metric_trend(
    track_row: pd.Series,
    metrics: list[dict],
    window: tuple[float, float],
    out_dir: Path,
    repro_stamp: str,
    *,
    metric_key: str,
    ylabel: str,
    fname_suffix: str,
    color: str,
    ylim: tuple[float, float] | None = None,
    yticks: list[float] | None = None,
) -> Path:
    """Plot one scalar tuning metric (OSI or preferred orientation) vs session."""
    track_id = int(track_row["track_id"])
    shank = int(track_row["shank"])
    n = len(metrics)

    x = np.arange(n)
    y = np.array([m[metric_key] for m in metrics], dtype=float)
    labels = [m["date"].strftime("%y-%m-%d") for m in metrics]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * n), 4.0))
    ax.plot(x, y, marker="o", ms=8, lw=2.0, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("Session", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(labelsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{ANIMAL_PREFIX} track {track_id} | shank {shank} | {ylabel} across "
        f"{n} sessions | window {window[0]:g}–{window[1]:g}s",
        fontsize=13, fontweight="bold", y=1.02,
    )

    fig.text(0.005, 0.002, repro_stamp, fontsize=5.5, color="0.45", va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"track_{track_id:03d}_shank{shank}_{fname_suffix}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png_path


def build_repro_stamp(
    track: int,
    base_dir: Path,
    match_dir: Path,
    window: tuple[float, float],
    excluded: set[str],
    sessions: list[dict],
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    excl = ",".join(sorted(excluded)) if excluded else "none"
    used = ",".join(item["session"].split("_")[1] for item in sessions)
    return (
        f"plot_track_tuning_row.py --track {track} "
        f"--base-dir {base_dir} --match-dir {match_dir} "
        f"--exclude-session {excl} --window {window[0]:g} {window[1]:g} | "
        f"sessions_used: {used} | generated {timestamp}"
    )


def generate(
    track: int,
    base_dir: Path,
    match_dir: Path,
    out_dir: Path,
    exclude_sessions: list[str],
    window: tuple[float, float],
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

    # Spike times come from each day's merged grating pickle, so the tuning curve
    # and its SEM are computed over the same display window (no regeneration).
    pkls = find_merged_pkls(base_dir, session_cols, excluded)
    if not pkls:
        raise FileNotFoundError("No merged grating pickles found for any session.")
    session_cache = {
        session: load_session_cache(path) for session, path in pkls.items()
    }

    sessions = collect_track_sessions(track_row, session_cols, session_cache)
    if not sessions:
        raise ValueError(
            f"track {track} (shank {int(track_row['shank'])}) has no spikes in any "
            f"non-excluded session's merged grating pickle."
        )

    metrics = compute_session_metrics(sessions, window)
    repro_stamp = build_repro_stamp(track, base_dir, match_dir, window, excluded, sessions)

    png_path = plot_tuning_row(track_row, metrics, window, out_dir, repro_stamp)
    osi_path = plot_metric_trend(
        track_row, metrics, window, out_dir, repro_stamp,
        metric_key="osi", ylabel="OSI", fname_suffix="osi_vs_session",
        color=OSI_COLOR, ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1.0],
    )
    pref_path = plot_metric_trend(
        track_row, metrics, window, out_dir, repro_stamp,
        metric_key="preferred", ylabel="Preferred orientation (°)",
        fname_suffix="pref_vs_session", color=PREF_COLOR,
        ylim=(-5, 185), yticks=[0, 45, 90, 135, 180],
    )

    print(f"track: {track} (shank {int(track_row['shank'])})")
    print(f"sessions: {len(sessions)} -> "
          f"{', '.join(item['session'].split('_')[1] for item in sessions)}")
    print(f"window: {window[0]:g}-{window[1]:g}s")
    print(f"excluded_sessions: {', '.join(sorted(excluded)) if excluded else 'none'}")
    print(f"figure (tuning row): {png_path}")
    print(f"figure (OSI vs session): {osi_path}")
    print(f"figure (preferred vs session): {pref_path}")
    return png_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=int, default=TRACK, help="Matched track_id to plot.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Defaults to <match-dir>/track_tuning_rows.",
    )
    parser.add_argument(
        "--exclude-session", action="append", default=list(EXCLUDE_SESSIONS),
        help="Session to exclude (260226 / 20260226 / CnL42SG_20260226). Repeatable.",
    )
    parser.add_argument(
        "--window", type=float, nargs=2, metavar=("START", "END"),
        default=list(TUNING_WINDOW),
        help="Post-stimulus window (s) averaged into the tuning curve. Default: 0.05 0.5.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = args.out_dir or (args.match_dir / "track_tuning_rows")
    generate(
        track=args.track,
        base_dir=args.base_dir,
        match_dir=args.match_dir,
        out_dir=out_dir,
        exclude_sessions=args.exclude_session,
        window=tuple(args.window),
    )


if __name__ == "__main__":
    main()
