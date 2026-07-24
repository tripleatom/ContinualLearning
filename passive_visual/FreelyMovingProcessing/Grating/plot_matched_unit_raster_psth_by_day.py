"""
Per matched-unit raster + PSTH across passive grating days.

Each PDF page is one matched unit track (from unit_match's unit_tracks.csv).
Columns are grating orientations. The spikes for each day come from that day's
merged grating pickle in:

    <base_dir>/<session>/passive_embedding_analysis/*_grating_data_merged.pkl

Layout per page (2 rows x n_orientations columns):
    top    : spike raster, one orientation per column. Every recorded day is a
             separate, labelled block of trial rows (earliest day on top), so the
             response on each day is shown separately.
    bottom : PSTH (firing rate, Hz), one orientation per column, one smoothed line
             per day (viridis: earliest -> latest).

Example:
    python plot_matched_unit_raster_psth_by_day.py --exclude-session 260226 --max-tracks 50
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d

from server_fallback import resolve_output_folder


DEFAULT_BASE_DIR = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG")
DEFAULT_MATCH_DIR = (
    DEFAULT_BASE_DIR
    / "unit_match_all_pairs"
    / "t0.60_w0.30_a0.60_ac0.60"
)
DEFAULT_EXCLUDE_SESSIONS = ("260226", "260313", "260305")  # sessions to skip (e.g. no merged pickle, or bad data quality)
ANIMAL_PREFIX = "CnL42SG"

# PSTH binning / smoothing
BIN_WIDTH = 0.010          # seconds
SMOOTH_SIGMA_MS = 20.0     # gaussian smoothing width (ms)

# Orientations are intended to lie on a 22.5 deg grid. Some sessions record a
# slightly-off value (e.g. 157.0 instead of 157.5); snapping to this grid merges
# those near-duplicates into one stimulus column.
ORI_GRID_STEP = 22.5


def normalize_session_token(token: str, animal_prefix: str = ANIMAL_PREFIX) -> str:
    """Convert 260226, 20260226, or CnL42SG_20260226 to CnL42SG_20260226."""
    token = str(token).strip()
    match = re.search(r"(\d{8}|\d{6})", token)
    if not match:
        return token
    digits = match.group(1)
    if len(digits) == 6:
        digits = "20" + digits
    return f"{animal_prefix}_{digits}"


def session_date(session: str) -> pd.Timestamp:
    return pd.to_datetime(session.split("_")[1], format="%Y%m%d")


def unit_key(shank: int, unit_id: int) -> str:
    return f"shank{int(shank)}_unit{int(unit_id)}"


def find_merged_pkls(
    base_dir: Path,
    session_cols: list[str],
    excluded_sessions: set[str],
) -> dict[str, Path]:
    """Newest *_grating_data_merged.pkl for each non-excluded session."""
    pkls: dict[str, Path] = {}
    for session in session_cols:
        if session in excluded_sessions:
            continue
        analysis_dir = base_dir / session / "passive_embedding_analysis"
        if not analysis_dir.exists():
            continue
        candidates = sorted(analysis_dir.glob("*_grating_data_merged.pkl"))
        if not candidates:
            candidates = sorted(analysis_dir.glob("*_grating_data*.pkl"))
        if not candidates:
            continue
        pkls[session] = max(candidates, key=lambda p: p.stat().st_mtime)
    return pkls


def canonical_orientation(value: float, step: float = ORI_GRID_STEP) -> float:
    """Snap an orientation to the nearest multiple of `step` (merges 157.0 -> 157.5)."""
    return round(round(float(value) / step) * step, 1)


def group_trials_by_orientation(trials: list[dict]) -> dict[float, list[np.ndarray]]:
    """Spike-time arrays (seconds, relative to onset) grouped by canonical orientation."""
    grouped: dict[float, list[np.ndarray]] = {}
    for trial in trials:
        orientation = trial.get("orientation")
        if orientation is None:
            continue
        ori = canonical_orientation(orientation)
        spikes = np.asarray(trial.get("spike_times", []), dtype=float)
        grouped.setdefault(ori, []).append(spikes)
    return grouped


def collect_track_sessions(
    track_row: pd.Series,
    session_cols: list[str],
    session_cache: dict[str, dict],
) -> list[dict]:
    """For one track, pull its matched unit's spikes from each available day."""
    shank = int(track_row["shank"])
    sessions: list[dict] = []

    for session in session_cols:
        cache = session_cache.get(session)
        if cache is None:
            continue
        matched_unit = track_row.get(session)
        if pd.isna(matched_unit):
            continue

        key = unit_key(shank, int(matched_unit))
        trials = cache["spike_data"].get(key)
        if not trials:
            continue

        grouped = group_trials_by_orientation(trials)
        if not grouped:
            continue

        sessions.append(
            {
                "session": session,
                "date": session_date(session),
                "unit_key": key,
                "unit_num": int(matched_unit),
                "pkl": cache["pkl"],
                "window_pre": cache["window_pre"],
                "window_post": cache["window_post"],
                "ori_spikes": grouped,
            }
        )

    return sorted(sessions, key=lambda item: item["date"])


def subsample_indices(n: int, cap: int) -> np.ndarray:
    """Evenly spaced indices to show at most `cap` of `n` trials (cap<=0 -> all)."""
    if cap <= 0 or n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).round().astype(int)


def compute_psth(
    spike_arrays: list[np.ndarray],
    bin_edges: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    """Smoothed firing rate (Hz) averaged over trials."""
    n_trials = len(spike_arrays)
    if n_trials == 0:
        return np.zeros(len(bin_edges) - 1)
    all_spikes = np.concatenate(spike_arrays) if any(a.size for a in spike_arrays) else np.array([])
    counts, _ = np.histogram(all_spikes, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    rate = counts / (n_trials * bin_width)
    return gaussian_filter1d(rate, sigma=sigma_bins)


def plot_track_page(
    track_row: pd.Series,
    sessions: list[dict],
    stim_duration: float,
    max_trials_per_day: int,
) -> plt.Figure:
    track_id = int(track_row["track_id"])
    shank = int(track_row["shank"])
    n_days = len(sessions)

    orientations = sorted({ori for item in sessions for ori in item["ori_spikes"]})
    if not orientations:
        raise ValueError(f"Track {track_id} has no orientations")
    n_ori = len(orientations)

    win_pre = max(item["window_pre"] for item in sessions)
    win_post = max(item["window_post"] for item in sessions)
    bin_edges = np.arange(-win_pre, win_post + BIN_WIDTH, BIN_WIDTH)
    bin_centers = bin_edges[:-1] + BIN_WIDTH / 2
    sigma_bins = SMOOTH_SIGMA_MS / (BIN_WIDTH * 1000.0)

    # blue -> red day progression (matches the reference figure: early days blue, late red)
    day_colors = plt.cm.turbo(np.linspace(0.08, 0.92, n_days)) if n_days > 1 else np.array([plt.cm.turbo(0.5)])

    # Reserve equal vertical space per day (max trials shown across orientations),
    # so day blocks line up across orientation columns.
    shown_counts = np.zeros((n_days, n_ori), dtype=int)
    for di, item in enumerate(sessions):
        for oi, ori in enumerate(orientations):
            arrays = item["ori_spikes"].get(ori, [])
            shown_counts[di, oi] = len(subsample_indices(len(arrays), max_trials_per_day))
    rows_per_day = shown_counts.max(axis=1)
    rows_per_day[rows_per_day == 0] = 1
    day_offsets = np.concatenate([[0], np.cumsum(rows_per_day)])  # length n_days+1
    total_rows = int(day_offsets[-1])

    raster_in = float(np.clip(total_rows * 0.012, 3.5, 11.0))
    fig_width = max(14.0, 1.75 * n_ori)
    fig_height = raster_in + 3.2 + 1.4
    fig, axes = plt.subplots(
        2,
        n_ori,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={"height_ratios": [raster_in, 3.0]},
    )

    fig.suptitle(
        f"{ANIMAL_PREFIX} matched track {track_id} | shank {shank} | "
        f"{n_days} days | mean match score "
        f"{float(track_row.get('mean_score', np.nan)):.3f}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # ---- pre-compute PSTHs to get a shared y-limit ----
    psth_lookup: dict[tuple[int, float], np.ndarray] = {}
    psth_max = 0.0
    for di, item in enumerate(sessions):
        for ori in orientations:
            arrays = item["ori_spikes"].get(ori, [])
            rate = compute_psth(arrays, bin_edges, sigma_bins)
            psth_lookup[(di, ori)] = rate
            if rate.size:
                psth_max = max(psth_max, float(np.nanmax(rate)))
    psth_ylim = psth_max * 1.08 if psth_max > 0 else 1.0

    for oi, ori in enumerate(orientations):
        # ===== top: raster =====
        ax = axes[0, oi]
        positions: list[np.ndarray] = []
        offsets: list[float] = []
        colors: list = []
        for di, item in enumerate(sessions):
            arrays = item["ori_spikes"].get(ori, [])
            sel = subsample_indices(len(arrays), max_trials_per_day)
            base = day_offsets[di]
            for row_within, idx in enumerate(sel):
                spikes = arrays[idx]
                if spikes.size == 0:
                    continue
                positions.append(spikes)
                offsets.append(base + row_within)
                colors.append(day_colors[di])
            # day separator
            if di > 0:
                ax.axhline(base - 0.5, color="0.6", lw=0.5, alpha=0.6)

        if positions:
            ax.eventplot(
                positions,
                lineoffsets=offsets,
                colors=colors,
                linelengths=0.9,
                linewidths=0.5,
                orientation="horizontal",
            )
        ax.axvspan(0, stim_duration, color="0.92", zorder=-10)
        # black stimulus bar above the raster (reference-figure style)
        ax.plot([0, stim_duration], [1.012, 1.012], transform=ax.get_xaxis_transform(),
                color="black", lw=3.0, clip_on=False, solid_capstyle="butt")
        ax.set_xlim(-win_pre, win_post)
        ax.set_ylim(total_rows - 0.5, -0.5)  # earliest day on top
        ax.set_title(f"{ori:g}°", fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(labelsize=7)
        if oi == 0:
            ax.set_ylabel("Day (each block = one session)", fontweight="bold", fontsize=9)
            ticks = [day_offsets[di] + rows_per_day[di] / 2.0 for di in range(n_days)]
            ax.set_yticks(ticks)
            ax.set_yticklabels(
                [sessions[di]["date"].strftime("%m-%d") for di in range(n_days)],
                fontsize=6.5,
            )
            for di, lbl in enumerate(ax.get_yticklabels()):
                lbl.set_color(day_colors[di])
                lbl.set_fontweight("bold")
        else:
            ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ===== bottom: PSTH =====
        ax = axes[1, oi]
        for di in range(n_days):
            rate = psth_lookup[(di, ori)]
            ax.plot(bin_centers, rate, color=day_colors[di], lw=1.1, alpha=0.9)
        ax.axvspan(0, stim_duration, color="0.9", zorder=-10)
        ax.axvline(0, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlim(-win_pre, win_post)
        ax.set_ylim(0, psth_ylim)
        ax.set_xlabel("Time from onset (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)
        if oi == 0:
            ax.set_ylabel("Firing rate (Hz)", fontweight="bold", fontsize=9)
        else:
            ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.085, hspace=0.16, wspace=0.12)

    # day colorbar (date legend) in its own axis so it never overlaps a column
    cmap = ListedColormap(day_colors)
    norm = BoundaryNorm(np.arange(n_days + 1) - 0.5, n_days)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([0.935, 0.30, 0.012, 0.40])
    cbar = fig.colorbar(mappable, cax=cax)
    tick_idx = np.unique(np.linspace(0, n_days - 1, min(n_days, 10)).round().astype(int))
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([sessions[di]["date"].strftime("%m-%d") for di in tick_idx], fontsize=7)
    cbar.set_label("session (early → late)", fontsize=8)

    cap_note = (
        f"all trials/day" if max_trials_per_day <= 0
        else f"≤{max_trials_per_day} trials/day shown (subsampled)"
    )
    repro = (
        f"plot_matched_unit_raster_psth_by_day.py | track {track_id} shank {shank} | "
        f"bin {int(BIN_WIDTH * 1000)}ms, gauss {SMOOTH_SIGMA_MS:g}ms, "
        f"window [-{win_pre:g},{win_post:g}]s, stim {stim_duration:g}s, {cap_note} | "
        f"days: {', '.join(s['date'].strftime('%m%d') for s in sessions)} | "
        f"generated {datetime.now():%Y-%m-%d %H:%M}"
    )
    fig.text(0.005, 0.004, repro, fontsize=5.0, color="0.4", ha="left", va="bottom")
    return fig


def load_session_cache(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)
    params = data.get("extraction_params", {})
    return {
        "pkl": str(pkl_path),
        "spike_data": data.get("spike_data", {}),
        "window_pre": float(params.get("window_pre", 0.2)),
        "window_post": float(params.get("window_post", 2.0)),
        "stim_duration": float(
            data.get("experiment_parameters", {}).get("stimulus_duration", 0.0)
        ),
    }


def generate_pdf(
    base_dir: Path,
    match_dir: Path,
    out_dir: Path,
    exclude_sessions: list[str],
    rank_by: str,
    ascending: bool,
    max_tracks: int,
    max_trials_per_day: int,
    per_unit_format: str = "png",
    combined_pdf: bool = True,
) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)

    tracks = pd.read_csv(match_dir / "unit_tracks.csv")
    if rank_by not in tracks.columns:
        raise ValueError(f"Cannot rank by {rank_by!r}; columns: {list(tracks.columns)}")
    tracks = tracks.sort_values(rank_by, ascending=ascending, na_position="last").reset_index(drop=True)

    session_cols = [c for c in tracks.columns if re.match(rf"{ANIMAL_PREFIX}_\d{{8}}$", c)]
    excluded = {normalize_session_token(s) for s in exclude_sessions}
    pkls = find_merged_pkls(base_dir, session_cols, excluded)
    if not pkls:
        raise FileNotFoundError("No merged grating pickles found for any session.")

    print(f"Loading {len(pkls)} session pickles ...")
    session_cache: dict[str, dict] = {}
    stim_durations = []
    for session in session_cols:
        if session not in pkls:
            continue
        try:
            session_cache[session] = load_session_cache(pkls[session])
            sd = session_cache[session]["stim_duration"]
            if sd > 0:
                stim_durations.append(sd)
            print(f"  {session}: {len(session_cache[session]['spike_data'])} units")
        except Exception as exc:
            print(f"  {session}: failed to load ({exc})")
    stim_duration = float(np.median(stim_durations)) if stim_durations else 2.0

    out_dir = resolve_output_folder(out_dir)
    unit_dir = out_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "matched_unit_raster_psth_by_day.pdf"
    index_rows = []
    n_pages = 0

    pdf = PdfPages(pdf_path) if combined_pdf else None
    if pdf is not None:
        pdf.infodict()["Title"] = "Matched unit raster + PSTH by day"
        pdf.infodict()["Subject"] = (
            f"base_dir={base_dir}; match_dir={match_dir}; "
            f"excluded={sorted(excluded)}; ranked_by={rank_by}; "
            f"bin={BIN_WIDTH}s; sigma={SMOOTH_SIGMA_MS}ms; max_trials_per_day={max_trials_per_day}"
        )

    try:
        for rank, (_, track_row) in enumerate(tracks.iterrows()):
            if max_tracks and n_pages >= max_tracks:
                break
            track_id = int(track_row["track_id"])
            shank = int(track_row["shank"])
            sessions = collect_track_sessions(track_row, session_cols, session_cache)
            if not sessions:
                index_rows.append(
                    {"rank": rank, "track_id": track_id, "shank": shank,
                     "n_days": 0, "first_session": "", "last_session": "",
                     "figure": "", "status": "no_spikes"}
                )
                continue

            fig = plot_track_page(track_row, sessions, stim_duration, max_trials_per_day)

            # one figure file per unit
            fname = f"rank{rank:03d}_track{track_id:03d}_shank{shank}_{len(sessions)}days.{per_unit_format}"
            fig_path = unit_dir / fname
            fig.savefig(fig_path, dpi=200 if per_unit_format == "png" else None)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

            n_pages += 1
            index_rows.append(
                {"rank": rank, "track_id": track_id, "shank": shank,
                 "n_days": len(sessions), "first_session": sessions[0]["session"],
                 "last_session": sessions[-1]["session"],
                 "figure": str(fig_path), "status": "ok"}
            )
            if n_pages % 25 == 0:
                print(f"  rendered {n_pages} unit figures")
    finally:
        if pdf is not None:
            pdf.close()

    index = pd.DataFrame(index_rows)
    index_path = out_dir / "matched_unit_raster_psth_by_day_index.csv"
    index.to_csv(index_path, index=False)

    print(f"output_dir: {out_dir}")
    print(f"per_unit_figures: {unit_dir} ({n_pages} files, .{per_unit_format})")
    if pdf is not None:
        print(f"combined_pdf: {pdf_path}")
    print(f"index_csv: {index_path}")
    print(f"excluded_sessions: {', '.join(sorted(excluded)) if excluded else 'none'}")
    print(f"ranked_by: {rank_by} ({'ascending' if ascending else 'descending'})")
    print(f"unit_figures: {n_pages}")
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    parser.add_argument(
        "--out-dir", type=Path,
        default=DEFAULT_MATCH_DIR / "matched_unit_raster_psth_by_day",
    )
    parser.add_argument(
        "--exclude-session", action="append", default=list(DEFAULT_EXCLUDE_SESSIONS),
        help="Session to exclude (260226, 20260226, or CnL42SG_20260226). Repeatable.",
    )
    parser.add_argument("--rank-by", default="mean_score",
                        help="Column in unit_tracks.csv used to order pages.")
    parser.add_argument("--ascending", action="store_true",
                        help="Sort rank column low->high (default high->low).")
    parser.add_argument("--max-tracks", type=int, default=0,
                        help="Render at most this many pages (0 = all).")
    parser.add_argument("--max-trials-per-day", type=int, default=40,
                        help="Cap raster trials/day/orientation, evenly subsampled "
                             "(0 = show all). PSTH always uses all trials.")
    parser.add_argument("--per-unit-format", default="png", choices=["png", "pdf", "svg"],
                        help="File format for the one-figure-per-unit output (default png).")
    parser.add_argument("--no-combined-pdf", action="store_true",
                        help="Skip writing the combined multi-page PDF (per-unit files only).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_pdf(
        base_dir=args.base_dir,
        match_dir=args.match_dir,
        out_dir=args.out_dir,
        exclude_sessions=args.exclude_session,
        rank_by=args.rank_by,
        ascending=args.ascending,
        max_tracks=args.max_tracks,
        max_trials_per_day=args.max_trials_per_day,
        per_unit_format=args.per_unit_format,
        combined_pdf=not args.no_combined_pdf,
    )


if __name__ == "__main__":
    main()
