"""
Generate 3-row summary figures for matched units across passive grating sessions.

For each matched unit track, this script plots all available tuning sessions as
columns with:
    1. waveform
    2. autocorrelogram
    3. orientation tuning curve

Example:
    python passive_visual/FreelyMovingProcessing/Grating/plot_matched_unit_tuning_tracks.py ^
        --base-dir "\\10.129.151.108\\xieluanlabs\\xl_cl\\sortout\\CnL42SG" ^
        --match-dir "\\10.129.151.108\\xieluanlabs\\xl_cl\\sortout\\CnL42SG\\unit_match_all_pairs\\t0.60_w0.30_a0.60_ac0.60" ^
        --exclude-session 260226
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from grating_utils import tuning_curve_from_psth
from server_fallback import resolve_output_folder


DEFAULT_BASE_DIR = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG")
DEFAULT_MATCH_DIR = (
    DEFAULT_BASE_DIR
    / "unit_match_all_pairs"
    / "t0.60_w0.30_a0.60_ac0.60"
)
DEFAULT_EXCLUDE_SESSIONS = ("260226", "20260313")
# Window (s post-stim) used to recompute tuning curves from each unit's saved
# PSTH. Kept identical to plot_unit_overlay_across_time.py so both scripts draw
# the same curve via grating_utils.tuning_curve_from_psth.
TUNING_WINDOW = (0.05, 0.5)


def normalize_session_token(token: str, animal_prefix: str = "CnL42SG") -> str:
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
        except Exception as exc:  # Keep the batch running if one file is corrupt.
            sessions.append(
                {
                    "session": session,
                    "unit_id": unit_id,
                    "pkl": str(tuning_pkl),
                    "error": str(exc),
                }
            )
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

    return sorted(sessions, key=lambda item: item.get("date", pd.Timestamp.max))


def plot_track(track_row: pd.Series, sessions: list[dict], out_dir: Path) -> tuple[plt.Figure, Path]:
    track_id = int(track_row["track_id"])
    shank = int(track_row["shank"])
    n_sessions = len(sessions)

    col_width = 1.7 if n_sessions >= 14 else 1.95
    fig_width = max(8.5, min(34, col_width * n_sessions))
    fig, axes = plt.subplots(3, n_sessions, figsize=(fig_width, 8.2), squeeze=False)

    fig.suptitle(
        f"CnL42SG matched track {track_id} | shank {shank} | "
        f"{n_sessions} tuning sessions | mean match score "
        f"{float(track_row.get('mean_score', np.nan)):.3f}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    for col, item in enumerate(sessions):
        data = item["data"]
        unit_info = data.get("unit_info", {})
        tuning = data.get("tuning", {})
        title = f"{item['date'].strftime('%m-%d')}\n{item['unit_id']}"

        plot_waveform(axes[0, col], unit_info, title, show_ylabel=(col == 0))
        plot_autocorrelogram(axes[1, col], unit_info, show_ylabel=(col == 0))
        plot_tuning_curve(axes[2, col], tuning, show_ylabel=(col == 0))

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.90,
        bottom=0.075,
        hspace=0.46,
        wspace=0.33,
    )
    png_path = out_dir / f"track_{track_id:03d}_shank{shank}_3row_tuning.png"
    fig.savefig(png_path, dpi=160)
    return fig, png_path


def plot_waveform(ax: plt.Axes, unit_info: dict, title: str, show_ylabel: bool) -> None:
    waveform = np.asarray(unit_info.get("waveform_template", []), dtype=float)
    time_ms = np.asarray(unit_info.get("waveform_t_ms", []), dtype=float)

    if waveform.size:
        if time_ms.size != waveform.size:
            time_ms = np.arange(waveform.size)
        ax.plot(time_ms, waveform, color="black", lw=1.8)
        ax.axhline(0, color="0.8", lw=0.6)

    ax.set_title(title, fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Waveform\n(uV)", fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)


def plot_autocorrelogram(ax: plt.Axes, unit_info: dict, show_ylabel: bool) -> None:
    counts = np.asarray(unit_info.get("acg_counts", []), dtype=float)
    lags = np.asarray(unit_info.get("acg_lags_ms", []), dtype=float)

    if counts.size:
        if lags.size != counts.size:
            lags = np.arange(counts.size) - counts.size // 2
        width = np.median(np.diff(lags)) * 0.85 if lags.size > 1 else 0.8
        ax.bar(lags, counts, width=width, color="#5B8DB8", edgecolor="white", linewidth=0.2)
        ax.axvline(0, color="red", ls="--", lw=1.0)

    if show_ylabel:
        ax.set_ylabel("ACG\nrate", fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.15)


def plot_tuning_curve(ax: plt.Axes, tuning: dict, show_ylabel: bool) -> None:
    # Recompute the curve + metrics from the saved PSTH over TUNING_WINDOW, via
    # the shared helper, so this matches plot_unit_overlay_across_time.py exactly.
    # (SEM is not reconstructable from a trial-averaged PSTH, so no error bars.)
    curve = tuning_curve_from_psth(tuning, TUNING_WINDOW)
    orientations = curve["orientations"]
    mean_rates = curve["mean_rates"]

    if orientations.size and mean_rates.size:
        ax.plot(
            orientations,
            mean_rates,
            marker="o",
            color="#2E86AB",
            lw=1.4,
        )

        baseline = curve.get("baseline_rate", np.nan)
        if np.isfinite(baseline):
            ax.axhline(float(baseline), color="0.55", ls="--", lw=0.8)

        preferred = curve.get("preferred_orientation_deg", np.nan)
        if np.isfinite(preferred):
            preferred = float(preferred)
            y_star = np.interp(
                preferred,
                orientations,
                mean_rates,
                left=mean_rates[0],
                right=mean_rates[-1],
            )
            ax.plot(preferred, y_star, marker="*", color="red", markersize=8, zorder=5)

        ax.set_xlim(-5, 185)
        ax.set_xticks([0, 45, 90, 135, 180])

    if show_ylabel:
        ax.set_ylabel("Tuning\nHz", fontweight="bold")
    ax.set_xlabel("Ori", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)

    osi = curve.get("osi", np.nan)
    modulation = curve.get("modulation_index", np.nan)
    max_rate = curve.get("max_rate", np.nan)
    stats_text = f"OSI {osi:.3f}\nMI {modulation:.3f}\nMax {max_rate:.1f}"
    ax.text(
        0.98,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.85,
        },
    )


def plot_response_change_page(track_row: pd.Series, sessions: list[dict]) -> plt.Figure:
    """Plot response changes for one matched track across sessions.

    Columns are orientations. The top row overlays PSTHs from all available
    sessions, and the bottom row shows session-by-time z-scored PSTH heatmaps.
    """
    track_id = int(track_row["track_id"])
    shank = int(track_row["shank"])

    orientations = sorted(
        {
            float(ori)
            for item in sessions
            for ori in item["data"].get("tuning", {}).get("psth_per_ori", {}).keys()
        }
    )
    if not orientations:
        raise ValueError(f"Track {track_id} has no PSTH orientations")

    n_orientations = len(orientations)
    fig_width = max(12, 1.75 * n_orientations)
    fig, axes = plt.subplots(
        2,
        n_orientations,
        figsize=(fig_width, 7.2),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    fig.suptitle(
        f"Response change | CnL42SG matched track {track_id} | shank {shank} | "
        f"{len(sessions)} sessions | mean match score "
        f"{float(track_row.get('mean_score', np.nan)):.3f}",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(sessions)))
    psth_lookup: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    all_values = []
    for session_index, item in enumerate(sessions):
        tuning = item["data"].get("tuning", {})
        time = np.asarray(tuning.get("psth_t", []), dtype=float)
        for orientation in orientations:
            psth = tuning.get("psth_per_ori", {}).get(orientation)
            if psth is None:
                psth = tuning.get("psth_per_ori", {}).get(str(orientation))
            if psth is None:
                continue
            values = np.asarray(psth, dtype=float)
            if time.size != values.size:
                continue
            psth_lookup[(session_index, orientation)] = (time, values)
            all_values.append(values)

    if all_values:
        all_concat = np.concatenate(all_values)
        z_mean = float(np.nanmean(all_concat))
        z_std = float(np.nanstd(all_concat))
        if not np.isfinite(z_std) or z_std == 0:
            z_std = 1.0
    else:
        z_mean = 0.0
        z_std = 1.0

    image_handle = None
    for col, orientation in enumerate(orientations):
        ax = axes[0, col]
        heat_rows = []
        heat_time = None
        for session_index, item in enumerate(sessions):
            found = psth_lookup.get((session_index, orientation))
            if found is None:
                heat_rows.append(None)
                continue
            time, values = found
            heat_time = time
            ax.plot(time, values, color=colors[session_index], lw=1.15, alpha=0.9)
            heat_rows.append((values - z_mean) / z_std)

        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.axvspan(0, 1, color="0.9", zorder=-10)
        ax.set_title(f"{orientation:g} deg", fontsize=9)
        if col == 0:
            ax.set_ylabel("PSTH\nHz", fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)

        ax = axes[1, col]
        if heat_time is not None:
            heat = np.full((len(sessions), heat_time.size), np.nan)
            for row_index, row_values in enumerate(heat_rows):
                if row_values is not None and row_values.size == heat_time.size:
                    heat[row_index, :] = row_values
            image_handle = ax.imshow(
                heat,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-2,
                vmax=2,
                extent=[heat_time[0], heat_time[-1], len(sessions) - 0.5, -0.5],
            )
            ax.axvline(0, color="black", ls="--", lw=0.7)
            ax.axvspan(0, 1, color="black", alpha=0.08)
        if col == 0:
            ax.set_ylabel("Session", fontweight="bold")
            tick_step = max(1, math.ceil(len(sessions) / 6))
            tick_positions = list(range(0, len(sessions), tick_step))
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(
                [sessions[index]["date"].strftime("%m-%d") for index in tick_positions],
                fontsize=7,
            )
        else:
            ax.set_yticks([])
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Add a compact color legend for session order on the last overlay panel.
    legend_ax = axes[0, -1]
    first_label = sessions[0]["date"].strftime("%m-%d")
    last_label = sessions[-1]["date"].strftime("%m-%d")
    legend_ax.text(
        0.98,
        0.96,
        f"color: {first_label} -> {last_label}",
        transform=legend_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )

    if image_handle is not None:
        cbar = fig.colorbar(image_handle, ax=axes[1, :].ravel().tolist(), shrink=0.75, pad=0.01)
        cbar.set_label("PSTH z-score", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.055,
        right=0.94,
        top=0.88,
        bottom=0.10,
        hspace=0.35,
        wspace=0.34,
    )
    return fig


def generate_figures(
    base_dir: Path,
    match_dir: Path,
    out_dir: Path,
    exclude_sessions: list[str],
    make_pdf: bool,
    make_response_change_pdf: bool,
    rank_by: str,
    ascending: bool,
) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)

    tracks = pd.read_csv(match_dir / "unit_tracks.csv")
    if rank_by not in tracks.columns:
        raise ValueError(f"Cannot rank by {rank_by!r}; available columns: {list(tracks.columns)}")
    tracks = tracks.sort_values(rank_by, ascending=ascending, na_position="last").reset_index(drop=True)

    session_cols = [col for col in tracks.columns if re.match(r"CnL42SG_\d{8}$", col)]
    excluded = {normalize_session_token(session) for session in exclude_sessions}
    tuning_dirs = find_tuning_dirs(base_dir, session_cols, excluded)

    out_dir = resolve_output_folder(out_dir)
    pdf_path = out_dir / "all_matched_units_3row_tuning.pdf"
    response_pdf_path = out_dir / "all_matched_units_response_change.pdf"

    index_rows = []
    pdf_context = PdfPages(pdf_path) if make_pdf else None
    response_pdf_context = PdfPages(response_pdf_path) if make_response_change_pdf else None
    try:
        for row_index, track_row in tracks.iterrows():
            track_id = int(track_row["track_id"])
            sessions = [
                item
                for item in collect_track_sessions(track_row, session_cols, tuning_dirs)
                if "data" in item
            ]

            if not sessions:
                index_rows.append(
                    {
                        "track_id": track_id,
                        "shank": int(track_row["shank"]),
                        "n_track_sessions": int(track_row["n_sessions"]),
                        "n_tuning_sessions": 0,
                        "first_session": "",
                        "last_session": "",
                        "png": "",
                        "status": "no_tuning_pickle",
                    }
                )
                continue

            fig, png_path = plot_track(track_row, sessions, out_dir)
            if pdf_context is not None:
                pdf_context.savefig(fig)
            plt.close(fig)

            if response_pdf_context is not None:
                try:
                    response_fig = plot_response_change_page(track_row, sessions)
                    response_pdf_context.savefig(response_fig)
                    plt.close(response_fig)
                except ValueError:
                    pass

            index_rows.append(
                {
                    "track_id": track_id,
                    "shank": int(track_row["shank"]),
                    "n_track_sessions": int(track_row["n_sessions"]),
                    "n_tuning_sessions": len(sessions),
                    "first_session": sessions[0]["session"],
                    "last_session": sessions[-1]["session"],
                    "png": str(png_path),
                    "status": "ok",
                }
            )

            if len(index_rows) % 25 == 0:
                print(f"processed {len(index_rows)} / {len(tracks)} tracks")
    finally:
        if pdf_context is not None:
            pdf_context.close()
        if response_pdf_context is not None:
            response_pdf_context.close()

    index = pd.DataFrame(index_rows)
    index_path = out_dir / "matched_unit_3row_tuning_index.csv"
    index.to_csv(index_path, index=False)

    print(f"output_dir: {out_dir}")
    print(f"index_csv: {index_path}")
    if make_pdf:
        print(f"combined_pdf: {pdf_path}")
    if make_response_change_pdf:
        print(f"response_change_pdf: {response_pdf_path}")
    print(f"excluded_sessions: {', '.join(sorted(excluded)) if excluded else 'none'}")
    print(f"ranked_by: {rank_by} ({'ascending' if ascending else 'descending'})")
    print(f"figures: {(index['status'] == 'ok').sum()}")
    print(f"skipped: {(index['status'] != 'ok').sum()}")
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <match-dir>/matched_unit_tuning_figures_3row_exclude_260226.",
    )
    parser.add_argument(
        "--exclude-session",
        action="append",
        default=list(DEFAULT_EXCLUDE_SESSIONS),
        help=(
            "Session to exclude. Accepts forms like 260226, 20260226, or "
            "CnL42SG_20260226. Can be repeated. Default: 260226."
        ),
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip the combined multi-page PDF and only write PNGs plus the index CSV.",
    )
    parser.add_argument(
        "--no-response-change-pdf",
        action="store_true",
        help="Skip the second PDF deck showing PSTH response changes across sessions.",
    )
    parser.add_argument(
        "--rank-by",
        default="mean_score",
        help="Column in unit_tracks.csv used to rank PDF pages and output generation order. Default: mean_score.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort rank column from low to high. Default is high to low.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.match_dir / "matched_unit_tuning_figures_3row_exclude_260226"

    generate_figures(
        base_dir=args.base_dir,
        match_dir=args.match_dir,
        out_dir=out_dir,
        exclude_sessions=args.exclude_session,
        make_pdf=not args.no_pdf,
        make_response_change_pdf=not args.no_response_change_pdf,
        rank_by=args.rank_by,
        ascending=args.ascending,
    )


if __name__ == "__main__":
    main()
