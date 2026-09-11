r"""Three-state sleep scoring from the thresholds the histograms produced.

Applies the per-channel thresholds written by plot_pc1_theta_velocity_histograms.py
(``*_bimodality_summary.csv``) to score every 1 s epoch, and draws the result over
the channel's own spectrogram. One figure per channel, so a channel whose
features separate badly is visible as such rather than averaged away.

The decision tree
-----------------
    PC1 > pc1_threshold                          -> NREM
    otherwise, theta > theta_threshold and
        velocity <  velocity_threshold           -> REM
        velocity >= velocity_threshold           -> WAKE   (theta/active wake)
    otherwise (theta low)
        velocity >= velocity_threshold           -> WAKE
        velocity <  velocity_threshold           -> QUIET   (immobile, no theta,
                                                             no slow waves)

The first three lines are the rule as specified. The last two are the case the
rule left open: low theta with the animal still is neither REM (no theta) nor
NREM (no slow waves), so it gets its own label instead of being forced into one
- typically quiet waking and drowsy transitions. Nothing is silently absorbed.

Two corrections this script has to make, both verified rather than assumed
-------------------------------------------------------------------------
PC1 sign
    compute_sleep_features.py takes PC1 straight from ``pca.fit_transform`` and
    PCA fixes the sign arbitrarily, so "high PC1" is NOT reliably NREM. On
    CnL46 260727 pre, channels 5 and 7 have PC1 anti-correlated with delta
    power (r = -0.78, -0.75) - there, ``PC1 > threshold`` would have selected
    WAKE. Orientation is settled per channel against log delta power, which is
    the physical anchor (NREM is high delta): a negative correlation flips both
    the trace and its threshold, after which "high is NREM" holds everywhere.

Missing / degenerate thresholds
    The GMM only yields a threshold where it found two separable components.
    theta_ratio is bimodal on about two thirds of channels, and some PC1 fits
    are degenerate (post channel 11's threshold is 19.0, outside the data
    entirely, so nothing would ever score as NREM). Any threshold falling
    outside the channel's own 1st-99th percentile is rejected and replaced -
    PC1 by that channel's Otsu split, theta by the median of the channels on
    which the fit did succeed - and every substitution is printed, stamped on
    the figure and recorded in the CSV, never applied quietly.

Velocity is session-level (one tracked animal), synced from camera time onto
the ephys clock through this session's DIO pulses, then averaged onto the same
1 s grid the spectrogram uses.

Usage
-----
    python score_states_from_thresholds.py                     # channels 1 5 7 11
    python score_states_from_thresholds.py --channels 0 3 --shank 0
    python score_states_from_thresholds.py --sessions post
"""
import argparse
import csv
import errno
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sleep_pipeline_config import (  # noqa: E402
    VELOCITY_MIN_COVERAGE, VELOCITY_SOURCE, VELOCITY_STEP_SEC,
    active_sleep_sessions, original_fs, plot_params, rec_folder, session_name,
    sleep_sessions, video_folder, mirror_on_backup_server,
    resolve_existing_file, resolve_output_folder,
)
from band_powers_io import band_time, load_spectrograms  # noqa: E402
from proc_func_velocity import aggregate_speed, velocity_output_name  # noqa: E402
from score_nrem_delta_velocity import otsu_threshold  # noqa: E402

DEFAULT_CHANNELS = [1, 5, 7, 11]
DEFAULT_SHANK = 0

# State codes, in the order they are drawn in the hypnogram and legend.
STATES = ["NREM", "REM", "WAKE", "QUIET", "unscored"]
STATE_COLORS = {
    "NREM": "#4C72B0",      # blue
    "REM": "#C44E52",       # red
    "WAKE": "#DD8452",      # orange
    "QUIET": "#8C8C8C",     # grey
    "unscored": "#FFFFFF",  # white - no camera coverage
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shank", type=int, default=DEFAULT_SHANK)
    p.add_argument("--channels", type=int, nargs="+", default=DEFAULT_CHANNELS,
                   help=f"channels to score (default: {' '.join(map(str, DEFAULT_CHANNELS))})")
    p.add_argument("--all-channels", action="store_true",
                   help="score every channel present on the selected shank; overrides --channels")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="sleep sessions to process (default: all active)")
    p.add_argument("--velocity-window-sec", type=float, default=VELOCITY_STEP_SEC,
                   help="window the speed is averaged over, matching the grid "
                        f"(default {VELOCITY_STEP_SEC:g}s, as the histograms used)")
    p.add_argument("--min-bout-sec", type=float, default=0.0,
                   help="drop bouts shorter than this before plotting (0 = raw "
                        "per-epoch scoring, which is what the thresholds define)")
    p.add_argument("--plot-max-points", type=int, default=0,
                   help="maximum time samples drawn in each figure (default 0 = full "
                        "resolution; scoring remains full resolution in all cases)")
    return p.parse_args()


def first_existing(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def with_backup_mirrors(paths):
    out = []
    for path in paths:
        mirrored = mirror_on_backup_server(path)
        if mirrored is not None:
            out.append(mirrored)
        out.append(path)
    return out


# =====================================================
# THRESHOLDS
# =====================================================

def load_thresholds(hist_dir, session_label, shank):
    """Per-channel thresholds from the bimodality summary CSV.

    Returns (rows_by_channel, theta_fallback) where theta_fallback is the
    median threshold over the channels whose theta fit did separate - used for
    the channels where it did not.
    """
    csv_path = resolve_existing_file(
        Path(hist_dir) / f"{session_label}_sh{shank}_bimodality_summary.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No threshold CSV: {csv_path}\n"
            f"Run plot_pc1_theta_velocity_histograms.py --shank {shank} first.")

    rows = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["channel"])] = row

    theta = [float(r["theta_ratio_threshold"]) for r in rows.values()
             if r["theta_ratio_bimodal"] == "True" and r["theta_ratio_threshold"]]
    theta_fallback = float(np.median(theta)) if theta else None
    return rows, theta_fallback, csv_path


def usable_threshold(value, values, name, channel, notes, fallback=None,
                     fallback_note=""):
    """A threshold only counts if it lands inside the data it will cut.

    A GMM can return a "threshold" beyond every sample it was fitted to - post
    channel 11's PC1 threshold of 19.0 against a range of about +/-10 is one -
    and applying it would put every epoch on one side without any warning. Such
    a value is rejected in favour of `fallback`, and the substitution is
    recorded in `notes` so it reaches the printout, the figure and the CSV.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = np.percentile(finite, [1, 99])

    if value is not None and np.isfinite(value) and lo <= value <= hi:
        return float(value), False

    if value is None or not np.isfinite(value):
        reason = "no threshold from the fit"
    else:
        reason = (f"fitted threshold {value:.3g} outside the data "
                  f"[{lo:.3g}, {hi:.3g}]")
    if fallback is None or not np.isfinite(fallback):
        notes.append(f"ch{channel} {name}: {reason}, and no fallback available")
        return None, True
    notes.append(f"ch{channel} {name}: {reason} -> using {fallback_note} "
                 f"{fallback:.4g}")
    return float(fallback), True


def pc1_orientation(pc1, delta):
    """+1 if high PC1 means high delta (so NREM), -1 if the PCA sign is flipped.

    PCA sign is arbitrary, so this has to be measured, not assumed. Delta power
    is the anchor: NREM is high delta by definition, so the sign of the
    correlation between PC1 and log delta is the sign of "high PC1 = NREM".
    """
    n = min(pc1.size, delta.size)
    good = np.isfinite(pc1[:n]) & np.isfinite(delta[:n]) & (delta[:n] > 0)
    if good.sum() < 10:
        return 1.0, np.nan
    r = float(np.corrcoef(pc1[:n][good], np.log10(delta[:n][good]))[0, 1])
    return (1.0 if r >= 0 else -1.0), r


# =====================================================
# VELOCITY ON THE EPHYS CLOCK
# =====================================================

def load_velocity_on_grid(session_cfg, grid, window_sec):
    """Session speed, synced to ephys time and averaged onto `grid`.

    Camera time is mapped onto ephys time by the linear fit between the first
    and last matched DIO pulse, exactly as score_nrem_delta_velocity.py and
    plot_sleep_spectrograms.py both do.
    """
    proc_file = session_cfg.get("proc_file")
    if not proc_file:
        print("    no proc_file registered - velocity unavailable")
        return None, None

    name = velocity_output_name(proc_file, VELOCITY_SOURCE)
    velocity_file = first_existing(with_backup_mirrors(
        [Path(proc_file).parent / name, Path(video_folder) / name]))
    sync_file = first_existing(with_backup_mirrors(
        [Path(rec_folder) / f"sync_times{session_cfg['suffix']}.pkl"]))
    if velocity_file is None:
        print(f"    velocity file not found ({name})")
        return None, None
    if sync_file is None:
        print(f"    sync_times{session_cfg['suffix']}.pkl not found")
        return None, None

    with open(velocity_file, "rb") as f:
        velocity_data = pickle.load(f)
    with open(sync_file, "rb") as f:
        sync = pickle.load(f)

    camera_time = np.asarray(velocity_data["time_stamp"], dtype=float)
    speed = np.asarray(velocity_data["velocity"], dtype=float)
    proc_pulses = np.asarray(sync["proc_rising_time"], dtype=float)
    ephys_pulses = np.asarray(sync["SG_rising_time"], dtype=float) / float(original_fs)

    inside = (camera_time >= proc_pulses[0]) & (camera_time <= proc_pulses[-1])
    camera_time, speed = camera_time[inside], speed[inside]
    ephys_time = np.interp(camera_time,
                           [proc_pulses[0], proc_pulses[-1]],
                           [ephys_pulses[0], ephys_pulses[-1]])

    windowed = aggregate_speed(ephys_time, speed, grid, window_sec,
                               min_coverage=VELOCITY_MIN_COVERAGE)
    covered = int(np.sum(np.isfinite(windowed["mean"])))
    print(f"    velocity: {velocity_file.name} -> {covered:,}/{grid.size:,} "
          f"epochs covered ({window_sec:g}s window), synced via {sync_file.name}")
    return windowed["mean"], velocity_file


# =====================================================
# SCORING
# =====================================================

def score_states(pc1, theta, speed, pc1_thr, theta_thr, velocity_thr):
    """The decision tree in the module docstring, one label per epoch.

    Epochs without camera coverage are 'unscored' rather than guessed: the two
    non-NREM branches both turn on velocity, so with no speed there is no way
    to tell REM from wake. NREM is decided on PC1 alone, so it is still
    assigned where the animal could not be seen.
    """
    states = np.full(pc1.size, "unscored", dtype=object)

    nrem = np.isfinite(pc1) & (pc1 > pc1_thr)
    states[nrem] = "NREM"

    rest = ~nrem & np.isfinite(pc1)
    have_speed = rest & np.isfinite(speed)
    high_theta = have_speed & np.isfinite(theta) & (theta > theta_thr)
    moving = speed >= velocity_thr

    states[high_theta & ~moving] = "REM"
    states[high_theta & moving] = "WAKE"
    states[have_speed & ~high_theta & moving] = "WAKE"
    states[have_speed & ~high_theta & ~moving] = "QUIET"
    return states


def enforce_min_bout(states, epoch_sec, min_bout_sec):
    """Absorb runs shorter than `min_bout_sec` into the preceding state."""
    if min_bout_sec <= 0:
        return states
    out = states.copy()
    min_len = max(1, int(round(min_bout_sec / epoch_sec)))
    start = 0
    for i in range(1, out.size + 1):
        if i == out.size or out[i] != out[start]:
            if (i - start) < min_len and start > 0:
                out[start:i] = out[start - 1]
            start = i
    return out


def state_summary(states, epoch_sec):
    total = states.size * epoch_sec
    rows = []
    for name in STATES:
        n = int(np.sum(states == name))
        if n:
            rows.append((name, n, n * epoch_sec / 60.0, 100.0 * n / states.size))
    return rows, total


# =====================================================
# FIGURE
# =====================================================

def color_scale(values):
    """vmin/vmax by plot_params['color_scale_method'], as in plot_sleep_spectrograms."""
    finite = values[np.isfinite(values)]
    method = plot_params["color_scale_method"]
    if method == "adaptive":
        median = np.median(finite)
        mad = np.median(np.abs(finite - median))
        lo = median - plot_params["adaptive_n_mad"] * mad
        hi = median + plot_params["adaptive_n_mad"] * mad
    elif method == "percentile":
        lo = np.percentile(finite, plot_params["vmin_percentile"])
        hi = np.percentile(finite, plot_params["vmax_percentile"])
    else:
        return plot_params["vmin_manual"], plot_params["vmax_manual"]
    span = hi - lo
    return (lo - plot_params["vmin_extension"] * span,
            hi + plot_params["vmax_extension"] * span)


def _display_indices(size, max_points):
    """Evenly spaced samples for plotting only; always retain the last bin."""
    if max_points <= 0 or size <= max_points:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, max_points, dtype=int))


def plot_scored(spec_z, freqs, spec_times, times, states, pc1, theta, speed,
                pc1_thr, theta_thr, velocity_thr, output_path, *,
                title, stamp_text, plot_max_points=0):
    """Spectrogram with the scored states beneath it, and the three features.

    Laid out like plot_sleep_spectrograms.py - same colormap, frequency range
    and colour-scale rule from plot_params - with one addition it needs and
    that one does not: the colorbar gets its OWN gridspec column rather than
    being attached to the spectrogram axes. `colorbar(ax=...)` takes its space
    out of the axes it is given, so the spectrogram would come out narrower
    than the panels below it and the shared x-axis would not line up. A
    dedicated column leaves every panel in column 0 exactly the same width.

    `spec_times` and `times` are kept separate rather than assumed equal: the
    features sit on band_time(), which is the spectrogram grid for a compact
    band-power file but the 500 Hz LFP grid for a legacy one. Both are seconds
    on the same clock, so the shared x-axis lines them up either way.
    """
    # The classifier has already run on every original 1-s sample.  Decimation
    # here affects only pixels/lines rendered into a PNG; it never changes a
    # state label, threshold, summary count or the data stored in the CSV.
    feature_idx = _display_indices(times.size, plot_max_points)
    spec_idx = _display_indices(spec_times.size, plot_max_points)
    plot_times = times[feature_idx]
    plot_states = states[feature_idx]
    plot_pc1, plot_theta, plot_speed = (pc1[feature_idx], theta[feature_idx],
                                         speed[feature_idx])
    plot_spec_times = spec_times[spec_idx]
    plot_spec_z = spec_z[:, spec_idx]
    codes = np.array([STATES.index(s) for s in plot_states])
    state_cmap = ListedColormap([STATE_COLORS[s] for s in STATES])
    vmin, vmax = color_scale(spec_z)

    figure = plt.figure(figsize=(20, 11))
    grid = figure.add_gridspec(
        5, 2, width_ratios=(60, 1), height_ratios=(3, 0.35, 1, 1, 1),
        hspace=0.16, wspace=0.012,
        left=0.07, right=0.93, top=0.90, bottom=0.07)

    ax_spec = figure.add_subplot(grid[0, 0])
    spec_cmap = plt.get_cmap(plot_params["cmap"]).copy()
    spec_cmap.set_bad("white")
    mesh = ax_spec.pcolormesh(plot_spec_times / 60.0, freqs, plot_spec_z,
                              shading="nearest", cmap=spec_cmap,
                              vmin=vmin, vmax=vmax)
    ax_spec.set_yscale("log")
    ax_spec.set_ylim(plot_params["freq_min"], plot_params["freq_max"])
    ax_spec.set_yticks([1, 4, 16, 64])
    ax_spec.set_yticklabels(["1", "4", "16", "64"])
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=12)
    ax_spec.tick_params(axis="y", labelsize=10, length=4)
    # labelbottom, not set_xticklabels([]): the panels share an x-axis, and
    # blanking the tick labels on one of them blanks them on the bottom panel
    # too, leaving the figure with no time axis at all.
    ax_spec.tick_params(labelbottom=False)
    for spine in ax_spec.spines.values():
        spine.set_visible(False)

    bar = figure.colorbar(mesh, cax=figure.add_subplot(grid[0, 1]))
    bar.set_label("Z-scored power", fontsize=10)
    bar.ax.tick_params(labelsize=9)
    bar.outline.set_visible(False)
    figure.suptitle(title, fontsize=13, y=0.965)

    # Hypnogram: one coloured column per epoch, on the spectrogram's x-axis.
    ax_state = figure.add_subplot(grid[1, 0], sharex=ax_spec)
    # shading='nearest' wants C to be (len(Y), len(X)); the row is duplicated so
    # the single band of colour fills the strip's height.
    ax_state.pcolormesh(plot_times / 60.0, [0, 1], np.vstack([codes, codes]),
                        shading="nearest", cmap=state_cmap,
                        vmin=-0.5, vmax=len(STATES) - 0.5)
    ax_state.set_yticks([])
    ax_state.set_ylabel("state", rotation=0, ha="right", va="center", fontsize=10)
    ax_state.tick_params(labelbottom=False, left=False)

    # In the free strip between the suptitle and the top of the gridspec, so it
    # cannot cover the spectrogram it describes.
    present = [s for s in STATES if np.any(states == s)]
    figure.legend(handles=[Patch(facecolor=STATE_COLORS[s], edgecolor="0.4",
                                 label=s) for s in present],
                  loc="upper center", bbox_to_anchor=(0.5, 0.935),
                  ncol=len(present), frameon=False, fontsize=11)

    def feature(row, values, threshold, label, color, log_y=False):
        ax = figure.add_subplot(grid[row, 0], sharex=ax_spec)
        ax.plot(plot_times / 60.0, values, color=color, lw=0.7)
        if threshold is not None:
            ax.axhline(threshold, color="black", ls="--", lw=1.2,
                       label=f"thr = {threshold:.3g}")
            ax.legend(fontsize=9, loc="upper right", frameon=False)
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return ax

    ax_pc1 = feature(2, plot_pc1, pc1_thr, "PC1\n(oriented)", "#4C72B0")
    ax_pc1.tick_params(labelbottom=False)
    ax_theta = feature(3, plot_theta, theta_thr, "Theta ratio\n5-10/2-15Hz", "#55A868")
    ax_theta.tick_params(labelbottom=False)
    ax_vel = feature(4, plot_speed, velocity_thr, "Speed\n(px/s)", "#C44E52", log_y=True)
    ax_vel.set_xlabel("Time (minutes)", fontsize=12)

    ax_spec.set_xlim(min(spec_times[0], times[0]) / 60.0,
                     max(spec_times[-1], times[-1]) / 60.0)

    figure.text(0.005, 0.004, stamp_text, fontsize=6, color="0.4",
                ha="left", va="bottom")
    # No bbox_inches='tight': it recrops around the artists and would undo the
    # explicit gridspec geometry that keeps the panels aligned.
    figure.savefig(output_path, dpi=plot_params["dpi"])
    plt.close(figure)


def save_with_fallback(save, out_dir, name):
    """Run `save(path)`, spilling to the backup server on ENOSPC."""
    path = Path(out_dir) / name
    try:
        save(path)
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        backup = mirror_on_backup_server(Path(out_dir))
        if backup is None:
            raise
        backup.mkdir(parents=True, exist_ok=True)
        path = backup / name
        print(f"    out of space - retrying on backup server: {path}")
        save(path)
    return path


# =====================================================
# MAIN
# =====================================================

def main():
    args = parse_args()
    low_freq = Path(rec_folder) / "low_freq"
    hist_dir = low_freq / "histograms"
    out_dir = resolve_output_folder(low_freq / "state_scoring")

    sessions = active_sleep_sessions(sleep_sessions)
    if args.sessions:
        sessions = {k: v for k, v in sessions.items() if k in args.sessions}
    if not sessions:
        raise SystemExit("No matching sleep sessions.")

    print("=" * 74)
    print(f"State scoring from histogram thresholds -- {session_name}")
    print(f"  shank    : {args.shank}")
    print(f"  channels : {args.channels}")
    print(f"  output   : {out_dir}")

    summary_rows = []
    for session_key, cfg in sessions.items():
        label = f"{session_name}{cfg['suffix']}"
        print("\n" + "#" * 74)
        print(f"SESSION {session_key}")

        band_file = resolve_existing_file(
            low_freq / f"{label}_all_shanks_band_powers.pkl")
        if not band_file.is_file():
            print(f"  band powers not found: {band_file}")
            continue
        with open(band_file, "rb") as f:
            all_data = pickle.load(f)
        if args.shank not in all_data["shanks_data"]:
            print(f"  shank {args.shank} not in {band_file.name}")
            continue
        shank_data = all_data["shanks_data"][args.shank]
        channel_ids = list(shank_data["channel_ids"])

        rows, theta_fallback, csv_path = load_thresholds(
            hist_dir, label, args.shank)
        print(f"  thresholds: {csv_path.name}")
        if theta_fallback is not None:
            n_fit = sum(1 for r in rows.values() if r["theta_ratio_bimodal"] == "True")
            print(f"  theta fallback (median of the {n_fit} channels whose fit "
                  f"separated): {theta_fallback:.4f}")

        grid = np.asarray(band_time(shank_data), dtype=float)
        epoch_sec = float(np.median(np.diff(grid)))
        speed, velocity_file = load_velocity_on_grid(
            cfg, grid, args.velocity_window_sec)
        if speed is None:
            speed = np.full(grid.size, np.nan)
            velocity_file = None
        velocity_thr = float(rows[channel_ids[0]]["velocity_threshold_px_s"])
        print(f"  velocity threshold (session-level): {velocity_thr:.4f} px/s")

        spectrograms = load_spectrograms(shank_data, band_file, args.shank)
        spec_freqs = np.asarray(shank_data["spectrogram_freqs"], dtype=float)
        spec_times = np.asarray(shank_data["spectrogram_times"], dtype=float)

        channels = channel_ids if args.all_channels else args.channels
        for channel in channels:
            if channel not in channel_ids:
                print(f"\n  ch{channel}: not on shank {args.shank} - skipping")
                continue
            ch_idx = channel_ids.index(channel)
            notes = []

            pc1 = np.asarray(shank_data["pc1_spectrogram"][ch_idx], dtype=float)
            theta = np.asarray(shank_data["band_powers"][channel]["theta_ratio"],
                               dtype=float)
            delta = np.asarray(shank_data["band_powers"][channel]["delta"],
                               dtype=float)

            n = min(pc1.size, theta.size, grid.size, speed.size)
            pc1, theta, speed_ch = pc1[:n], theta[:n], speed[:n]
            times = grid[:n]

            sign, r = pc1_orientation(pc1, delta)
            pc1 = sign * pc1
            row = rows[channel]
            pc1_thr_raw = (float(row["pc1_threshold"])
                           if row["pc1_threshold"] not in ("", "None") else None)
            if pc1_thr_raw is not None:
                pc1_thr_raw *= sign
            if sign < 0:
                notes.append(f"ch{channel} PC1: sign flipped (r={r:+.2f} vs log "
                             f"delta), trace and threshold negated so high = NREM")

            print(f"\n  ch{channel}: PC1 vs log delta r={r:+.3f} "
                  f"({'flipped' if sign < 0 else 'as stored'})")

            pc1_thr, pc1_sub = usable_threshold(
                pc1_thr_raw, pc1, "PC1", channel, notes,
                fallback=otsu_threshold(pc1[np.isfinite(pc1)]),
                fallback_note="this channel's Otsu split")
            theta_thr_raw = (float(row["theta_ratio_threshold"])
                             if row["theta_ratio_bimodal"] == "True"
                             and row["theta_ratio_threshold"] else None)
            theta_thr, theta_sub = usable_threshold(
                theta_thr_raw, theta, "theta", channel, notes,
                fallback=theta_fallback,
                fallback_note="the across-channel median")

            if pc1_thr is None or theta_thr is None:
                print(f"    cannot score ch{channel}: "
                      f"{'no PC1 threshold' if pc1_thr is None else ''}"
                      f"{' and ' if pc1_thr is None and theta_thr is None else ''}"
                      f"{'no theta threshold' if theta_thr is None else ''}")
                for note in notes:
                    print(f"      {note}")
                continue

            states = score_states(pc1, theta, speed_ch,
                                  pc1_thr, theta_thr, velocity_thr)
            states = enforce_min_bout(states, epoch_sec, args.min_bout_sec)

            print(f"    PC1 thr {pc1_thr:.3f}{' (substituted)' if pc1_sub else ''}, "
                  f"theta thr {theta_thr:.4f}{' (substituted)' if theta_sub else ''}, "
                  f"speed thr {velocity_thr:.3f} px/s")
            for note in notes:
                print(f"      {note}")
            rows_out, total = state_summary(states, epoch_sec)
            for name, count, minutes, percent in rows_out:
                print(f"      {name:<9} {count:>6} epochs  {minutes:>7.1f} min  "
                      f"{percent:>5.1f}%")

            # Same display transform the spectrogram figures use: dB, then a
            # robust z-score per frequency row, so state-dependent band changes
            # are visible against the 1/f background.
            spec_ch = spectrograms[ch_idx]
            log_spec = 10 * np.log10(spec_ch + 1e-12)
            median_f = np.median(log_spec, axis=1, keepdims=True)
            mad_f = np.median(np.abs(log_spec - median_f), axis=1, keepdims=True)
            spec_z = (log_spec - median_f) / (1.4826 * mad_f + 1e-10)
            keep = spec_times <= times[-1]
            spec_z, spec_t = spec_z[:, keep], spec_times[keep]
            if spec_t.size == 0:
                print(f"    no spectrogram bins within the scored window - skipping")
                continue

            stamp = (
                f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by "
                f"score_states_from_thresholds.py  |  {label} sh{args.shank} "
                f"ch{channel}  |  thresholds from {csv_path.name}: "
                f"PC1>{pc1_thr:.3f}{'*' if pc1_sub else ''} "
                f"theta>{theta_thr:.4f}{'*' if theta_sub else ''} "
                f"speed<{velocity_thr:.3f}px/s ({args.velocity_window_sec:g}s window)"
                f"{'  |  * = substituted, see notes' if (pc1_sub or theta_sub) else ''}"
                f"  |  PC1 orientation r={r:+.2f} vs log delta"
                f"{' (FLIPPED)' if sign < 0 else ''}"
                f"  |  min_bout={args.min_bout_sec:g}s  |  src={band_file.name}"
                + (f", {Path(velocity_file).name}" if velocity_file else ", no velocity")
            )
            title = (f"{label}  shank {args.shank}  channel {channel}  -  "
                     f"states from histogram thresholds")

            name = f"{label}_sh{args.shank}_ch{channel:03d}_state_scoring.png"
            path = save_with_fallback(
                lambda p: plot_scored(
                    spec_z, spec_freqs, spec_t, times, states, pc1, theta,
                    speed_ch, pc1_thr, theta_thr, velocity_thr, p,
                    title=title, stamp_text=stamp,
                    plot_max_points=args.plot_max_points),
                out_dir, name)
            print(f"    -> {path.name}")

            # The full per-epoch result -- until now this stage only ever
            # rendered a PNG and folded the labels into an aggregate CSV, so
            # nothing downstream could consume the actual per-epoch states.
            # `sleep_start_sample` is recorded the same way
            # score_nrem_delta_velocity.py records it, so a downstream script
            # can align either scorer's epoch grid onto the MUA clock through
            # detect_off_states.clock_offset_sec() without caring which one
            # produced it.
            pkl_name = f"{label}_sh{args.shank}_ch{channel:03d}_state_scoring.pkl"
            def _save_pkl(p, times=times, states=states, pc1=pc1, theta=theta,
                          speed_ch=speed_ch):
                with open(p, "wb") as f:
                    pickle.dump({
                        "session": session_name, "sleep_session": session_key,
                        "shank": args.shank, "channel": channel,
                        "epoch_sec": epoch_sec, "epoch_times": times,
                        "states": states, "pc1": pc1, "theta_ratio": theta,
                        "speed_px_s": speed_ch,
                        "pc1_threshold": pc1_thr, "pc1_threshold_substituted": pc1_sub,
                        "pc1_orientation_r": r, "pc1_sign_flipped": sign < 0,
                        "theta_threshold": theta_thr,
                        "theta_threshold_substituted": theta_sub,
                        "velocity_threshold_px_s": velocity_thr,
                        "min_bout_sec": args.min_bout_sec,
                        "sleep_start_sample": cfg.get("start_sample"),
                        "band_powers_file": str(band_file),
                        "velocity_file": str(velocity_file) if velocity_file else None,
                        "thresholds_csv": str(csv_path),
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            pkl_path = save_with_fallback(_save_pkl, out_dir, pkl_name)
            print(f"    -> {pkl_path.name}")

            record = {
                "session": session_key, "shank": args.shank, "channel": channel,
                "pc1_threshold": pc1_thr, "pc1_threshold_substituted": pc1_sub,
                "pc1_orientation_r": r, "pc1_sign_flipped": sign < 0,
                "theta_threshold": theta_thr,
                "theta_threshold_substituted": theta_sub,
                "velocity_threshold_px_s": velocity_thr,
                "epoch_sec": epoch_sec, "n_epochs": int(states.size),
                "figure": path.name,
            }
            for state in STATES:
                count = int(np.sum(states == state))
                record[f"{state}_epochs"] = count
                record[f"{state}_percent"] = 100.0 * count / states.size
            summary_rows.append(record)

    if summary_rows:
        name = f"{session_name}_sh{args.shank}_state_scoring_summary.csv"
        def write(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        path = save_with_fallback(write, out_dir, name)
        print(f"\nSummary: {path}")
    else:
        print("\nNothing scored.")


if __name__ == "__main__":
    main()
