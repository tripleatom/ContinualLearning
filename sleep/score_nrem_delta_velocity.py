r"""NREM scoring inputs: delta-band power and locomotion speed, epoch by epoch.

Stage 1 of "find NREM" -- this script only BUILDS and SHOWS the two distributions
you threshold on. It deliberately does not score anything: run it, read the
histograms, pick thresholds, then pass them back with ``--delta-thresh`` /
``--velocity-thresh`` to write the NREM windows.

What it computes
----------------
delta index
    ``band_powers[channel]['delta']`` from ``*_all_shanks_band_powers.pkl`` --
    0.5-4 Hz squared amplitude, already smoothed with a 10 s moving average at
    500 Hz by compute_sleep_features.py. Per channel it is epoch-averaged,
    log10'd, then robust-z-scored (median / 1.4826*MAD) over the whole session,
    and the z-scores are averaged across every good channel of every shank in
    ``SHANKS``.

    The per-channel normalisation matters: raw delta power varies several-fold
    with depth, so a plain average across channels is dominated by whichever
    channel happens to sit in the largest sink. Normalising first gives each
    channel one equal vote on "is this epoch high-delta relative to this
    channel's own baseline".

velocity
    Locomotion speed from the DLC body centroid (``*_velocity_body.pkl``,
    pixels/s), mapped from camera time onto ephys time through this session's
    DIO sync pulses, then reduced to one value per epoch. Both the window mean
    and median are kept -- mean punishes a brief movement inside an otherwise
    still window, median ignores it.

    The reduction uses a 10 s window centred on each epoch, not the epoch's own
    span, so the speed carries the same temporal support as the delta index it
    is thresholded against (10 s moving average, 10 s spectrogram window). It
    is the only averaging the speed receives: proc_func_velocity.py filters the
    POSITION before differentiating, precisely so that nothing has to smooth
    the speed afterwards. Changing ``--velocity-window-sec`` moves the scale of
    the trace, so re-read the histogram before reusing a threshold.

Bad channels
------------
``BAD_CHANNELS`` are excluded from every shank. These are channel IDs (the
``channel_ids`` in the LFP/band-power files), not depth positions. On CnL46
260727 shank 4 the listed channels have a median std of ~1500 uV against ~335 uV
for the rest, several saturating at 3200-4300 uV, and they are the only ones
anti-correlated with the shank median -- so the list separates cleanly by ID and
not at all by depth position.

Usage
-----
    python score_nrem_delta_velocity.py                  # both sessions
    python score_nrem_delta_velocity.py --sessions post
    python score_nrem_delta_velocity.py --epoch-sec 10
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server_fallback import resolve_existing_file, resolve_output_folder  # noqa: E402
from sleep_pipeline_config import (  # noqa: E402
    VELOCITY_MIN_COVERAGE, VELOCITY_SOURCE, VELOCITY_WINDOW_SEC,
    active_sleep_sessions, low_freq_folder, original_fs,
    rec_folder, session_name, sleep_sessions, video_folder,
)
from proc_func_velocity import aggregate_speed, velocity_output_name  # noqa: E402
from band_powers_io import band_time, epoch_average, epoch_times  # noqa: E402


# =====================================================
# WHAT TO SCORE
# =====================================================
#: Shanks whose channels are pooled into the delta index.
SHANKS = [4, 5]

#: Channel IDs excluded from every shank in SHANKS. See the module docstring for
#: the evidence that these are IDs rather than depth positions.
BAD_CHANNELS = [0, 1, 4, 5, 8, 12, 14, 18, 27, 29, 30, 31]

#: Scoring epoch, seconds. Matches sleep_detect_params['epoch_sec'].
DEFAULT_EPOCH_SEC = 4.0


# =====================================================
# HELPERS
# =====================================================

def robust_z(values):
    """(x - median) / (1.4826 * MAD), with a std fallback for degenerate MAD."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(values.shape, np.nan)
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = finite.std() or 1.0
    return (values - median) / scale


def find_modes(values, bins=120, smooth_bins=3.0):
    """Locate the two dominant modes and the trough between them.

    Non-parametric on purpose: the histogram is smoothed and its local maxima
    are read off directly, so what the figure marks is what the data shows
    rather than the fit of an assumed shape. Returns None when the smoothed
    histogram has fewer than two peaks -- an honest "this is not bimodal"
    rather than a trough invented between a peak and a shoulder.

    ``dip`` is how deep the trough is relative to the weaker of the two peaks:
    1.0 is complete separation, 0.0 is no dip at all. Below ~0.2 the two modes
    are not meaningfully distinct.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return None
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(hist.astype(np.float64), smooth_bins)

    peaks = [i for i in range(1, smooth.size - 1)
             if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    if len(peaks) < 2:
        return None

    low, high = sorted(sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2])
    trough = low + int(np.argmin(smooth[low:high + 1]))
    weaker = min(smooth[low], smooth[high])
    dip = float(1.0 - smooth[trough] / weaker) if weaker > 0 else 0.0
    return {
        "mode_low": float(centers[low]),
        "mode_high": float(centers[high]),
        "trough": float(centers[trough]),
        "dip": dip,
        "fraction_above_trough": float(np.mean(values > centers[trough])),
        "n_peaks": len(peaks),
    }


def annotate_modes(ax, modes, *, unit="", color="darkgreen"):
    """Draw the two modes and the trough onto a histogram axis."""
    if modes is None:
        ax.text(0.98, 0.94, "unimodal -- no trough found", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="crimson")
        return
    for key, style, label in (("mode_low", ":", "mode"),
                              ("mode_high", ":", "mode"),
                              ("trough", "-", "trough")):
        ax.axvline(modes[key], color=color, lw=1.4, ls=style)
    ax.text(0.98, 0.94,
            f"modes {modes['mode_low']:.2f} / {modes['mode_high']:.2f}{unit}\n"
            f"trough {modes['trough']:.2f}{unit}  (dip {modes['dip']:.2f})\n"
            f"{modes['fraction_above_trough'] * 100:.1f}% above trough",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=color)


def otsu_threshold(values, bins=256):
    """Otsu's between-class-variance split -- a starting guess, not a decision."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight0 = np.cumsum(hist)
    weight1 = weight0[-1] - weight0
    cumulative = np.cumsum(hist * centers)
    mean0 = cumulative / np.maximum(weight0, 1)
    mean1 = (cumulative[-1] - cumulative) / np.maximum(weight1, 1)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centers[int(np.argmax(between[:-1]))])


def first_existing(paths):
    for path in paths:
        path = resolve_existing_file(Path(path))
        if path.is_file():
            return path
    return None


# =====================================================
# LOADING
# =====================================================

def load_delta_index(band_powers_path, shanks, bad_channels, epoch_sec, verbose=True):
    """Per-epoch delta index pooled over the good channels of `shanks`."""
    with open(band_powers_path, "rb") as file:
        data = pickle.load(file)

    per_channel_z = []
    used = {}
    raw_mean = []
    fs = None
    for shank in shanks:
        if shank not in data["shanks_data"]:
            print(f"    shank {shank} absent from the band-power file, skipping")
            continue
        entry = data["shanks_data"][shank]
        # sampling_rate is the 500 Hz LFP rate, kept because the NREM mask this
        # module returns is sample-indexed at that rate. The delta envelope
        # itself is stored on band_time - the spectrogram grid in a compact
        # file, the LFP grid in a legacy one - so epochs are formed from those
        # timestamps rather than from a sample count, which is only exact when
        # the interval happens to divide the epoch.
        fs = float(entry["sampling_rate"])
        time = band_time(entry)
        good = [int(c) for c in sorted(entry["band_powers"]) if int(c) not in bad_channels]
        used[shank] = good
        for channel in good:
            delta = np.asarray(entry["band_powers"][channel]["delta"], dtype=np.float64)
            epoch_power = epoch_average(delta, time, epoch_sec)
            raw_mean.append(epoch_power)
            per_channel_z.append(robust_z(np.log10(epoch_power + 1e-12)))
        if verbose:
            print(f"    shank {shank}: {len(good)}/{len(entry['band_powers'])} channels "
                  f"kept -> {good}")

    if not per_channel_z:
        raise ValueError("No usable channels; check SHANKS / BAD_CHANNELS.")

    length = min(len(v) for v in per_channel_z)
    stack = np.vstack([v[:length] for v in per_channel_z])
    delta_index = np.nanmean(stack, axis=0)
    delta_raw = np.nanmean(np.vstack([v[:length] for v in raw_mean]), axis=0)
    epoch_time = epoch_times(epoch_sec, length)
    return {
        "delta_index": delta_index,
        "delta_raw_power": delta_raw,
        "epoch_time": epoch_time,
        "channels_used": used,
        "n_channels": stack.shape[0],
        "lfp_fs": fs,
    }


def load_velocity(session_cfg, epoch_time, epoch_sec, verbose=True,
                  window_sec=VELOCITY_WINDOW_SEC,
                  min_coverage=VELOCITY_MIN_COVERAGE):
    """Windowed speed on the ephys clock, or None when sync/tracking is missing.

    Camera time is mapped onto ephys time by the linear fit between the first and
    last matched DIO pulse pair, exactly as plot_sleep_spectrograms.py does.
    ``SG_rising_time`` is stored in acquisition samples, so it is divided by
    ``original_fs`` to land in the same seconds-from-epoch-start clock the LFP
    and band powers use.

    The reduction to one value per epoch uses a ``window_sec`` window centred on
    each epoch - 10 s, the spectrogram's own window - not the epoch's own
    ``epoch_sec`` span. That is deliberate: the delta index this speed is
    thresholded alongside carries 10 s of support (a 10 s moving average, then
    the 10 s spectrogram window), so gating it with a 4 s speed compared two
    signals of different bandwidth. Windows overlap between neighbouring epochs,
    exactly as the spectrogram's 10 s / 1 s grid does.
    """
    proc_file = session_cfg.get("proc_file")
    if proc_file is None:
        print("    no proc_file registered for this session -- skipping velocity")
        return None

    name = velocity_output_name(proc_file, VELOCITY_SOURCE)
    velocity_file = first_existing([Path(proc_file).parent / name, video_folder / name])
    if velocity_file is None:
        print(f"    velocity file not found ({name}); run proc_func_velocity.py "
              f"--source {VELOCITY_SOURCE}")
        return None

    sync_file = first_existing([Path(rec_folder) / f"sync_times{session_cfg['suffix']}.pkl"])
    if sync_file is None:
        print(f"    sync_times{session_cfg['suffix']}.pkl not found; run video_ephys_sync.py")
        return None

    with open(velocity_file, "rb") as file:
        velocity_data = pickle.load(file)
    with open(sync_file, "rb") as file:
        sync = pickle.load(file)

    camera_time = np.asarray(velocity_data["time_stamp"], dtype=np.float64)
    speed = np.asarray(velocity_data["velocity"], dtype=np.float64)
    proc_pulses = np.asarray(sync["proc_rising_time"], dtype=np.float64)
    ephys_pulses = np.asarray(sync["SG_rising_time"], dtype=np.float64) / float(original_fs)

    inside = (camera_time >= proc_pulses[0]) & (camera_time <= proc_pulses[-1])
    camera_time, speed = camera_time[inside], speed[inside]
    ephys_time = np.interp(camera_time,
                           [proc_pulses[0], proc_pulses[-1]],
                           [ephys_pulses[0], ephys_pulses[-1]])

    if verbose:
        n_nan = int(np.sum(~np.isfinite(speed)))
        print(f"    velocity: {velocity_file.name} ({speed.size} frames, "
              f"{n_nan:,} unusable), synced to "
              f"{ephys_time[0]:.1f}-{ephys_time[-1]:.1f} s via {sync_file.name}")

    windowed = aggregate_speed(ephys_time, speed, epoch_time, window_sec,
                               min_coverage=min_coverage, with_median=True)
    if verbose:
        scored = int(np.sum(np.isfinite(windowed["mean"])))
        print(f"    reduced to {epoch_time.size} epochs on a {window_sec:g}s window: "
              f"{scored} scorable, {epoch_time.size - scored} below "
              f"{min_coverage:.0%} camera coverage")

    return {
        "velocity_mean": windowed["mean"],
        "velocity_median": windowed["median"],
        "frames_per_epoch": windowed["n_frames"],
        "velocity_coverage": windowed["coverage"],
        "velocity_window_sec": float(window_sec),
        "velocity_min_coverage": float(min_coverage),
        "velocity_file": str(velocity_file),
        "sync_file": str(sync_file),
        "ephys_time_range": (float(ephys_time[0]), float(ephys_time[-1])),
    }


# =====================================================
# BOUT BUILDING
# =====================================================

def boxcar(values, width):
    """Centred moving average that ignores NaNs, preserving length."""
    values = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return values.copy()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones(int(width))
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out


def mask_to_bouts(mask, epoch_time, epoch_sec):
    """Contiguous True runs as (start_s, end_s), using epoch edges."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    half = epoch_sec / 2.0
    return [(float(epoch_time[s] - half), float(epoch_time[e - 1] + half))
            for s, e in zip(starts, ends)]


def merge_and_filter(bouts, merge_gap_sec, min_bout_sec):
    """Bridge short gaps, then drop bouts that are still too short."""
    if not bouts:
        return []
    merged = [list(bouts[0])]
    for start, end in bouts[1:]:
        if start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if (e - s) >= min_bout_sec]


def build_nrem(inputs, *, delta_thresh, velocity_thresh, smooth_epochs=3,
               merge_gap_sec=20.0, min_bout_sec=60.0, consolidated_sec=120.0,
               sleep_start_sample=None):
    """Turn the two epoch traces into NREM bouts and an LFP-rate mask.

    Epochs with no camera coverage (NaN speed) are rejected: "we could not see
    whether the animal moved" must not read as "the animal was still", or the
    uncovered ends of the recording would be scored as sleep by default.

    The speed is NOT smoothed again here. load_velocity already reduced it on a
    10 s window matched to the spectrogram, which is the one and only averaging
    it gets; a boxcar on top of that would stack a second, unrelated timescale
    onto a signal whose support was chosen deliberately. `smooth_epochs` still
    applies to the delta index, whose own support comes from the band-power
    file rather than from anything this module controls.
    """
    epoch_sec = float(inputs["epoch_sec"])
    epoch_time = np.asarray(inputs["epoch_time"], dtype=np.float64)
    delta = boxcar(inputs["delta_index"], smooth_epochs)

    speed = inputs.get("velocity_mean")
    if speed is None:
        raise ValueError("No velocity in this inputs pkl; cannot apply a speed gate.")
    speed = np.asarray(speed, dtype=np.float64)

    high_delta = delta > delta_thresh
    still = np.isfinite(speed) & (speed < velocity_thresh)
    raw_mask = high_delta & still

    bouts = merge_and_filter(mask_to_bouts(raw_mask, epoch_time, epoch_sec),
                             merge_gap_sec, min_bout_sec)
    consolidated = [b for b in bouts if (b[1] - b[0]) >= consolidated_sec]
    longest = max(consolidated, key=lambda b: b[1] - b[0], default=None)

    # Clean epoch mask, rebuilt from the surviving bouts so it matches them.
    clean = np.zeros(epoch_time.size, dtype=bool)
    for start, end in bouts:
        clean |= (epoch_time >= start) & (epoch_time <= end)

    fs = float(inputs.get("lfp_fs") or 500.0)
    n_lfp = int(round(epoch_time.size * epoch_sec * fs))
    nrem_mask_lfp = np.zeros(n_lfp, dtype=bool)
    for start, end in bouts:
        nrem_mask_lfp[max(0, int(start * fs)):min(n_lfp, int(end * fs))] = True

    return {
        "session": inputs.get("session"),
        "sleep_session": inputs.get("sleep_session"),
        "shanks": inputs.get("shanks"),
        "bad_channels": inputs.get("bad_channels"),
        "fs": fs,
        "epoch_sec": epoch_sec,
        "epoch_times": epoch_time,
        "lfp_time": np.arange(n_lfp) / fs,
        "sw_index": np.asarray(inputs["delta_index"]),
        "delta_index_smoothed": delta,
        "velocity_mean": np.asarray(inputs["velocity_mean"]),
        # Kept under its historical key for the plots and any saved pkl that
        # reads it; it is now the windowed speed itself, smoothed exactly once.
        "velocity_smoothed": speed,
        "velocity_coverage": inputs.get("velocity_coverage"),
        "nrem_epoch_mask": clean,
        "nrem_mask_lfp": nrem_mask_lfp,
        "bout_intervals_s": bouts,
        "consolidated_windows_s": consolidated,
        "fully_asleep_window_s": longest,
        # detect_off_states.py converts bout times onto the MUA epoch clock with
        # this, exactly as it does for score_nrem_epochs.py's output.
        "sleep_start_sample": sleep_start_sample,
        "params": {
            "delta_thresh": float(delta_thresh),
            "velocity_thresh": float(velocity_thresh),
            "velocity_units": "px/s (DLC body centroid)",
            "velocity_window_sec": inputs.get("velocity_window_sec"),
            "velocity_min_coverage": inputs.get("velocity_min_coverage"),
            "smooth_epochs": int(smooth_epochs),
            "merge_gap_sec": float(merge_gap_sec),
            "min_bout_sec": float(min_bout_sec),
            "consolidated_sec": float(consolidated_sec),
            "source": "score_nrem_delta_velocity.py",
        },
    }


def plot_nrem(nrem, output_path, *, session_label, source_note):
    """Verification figure: both traces with the accepted bouts shaded."""
    time_min = np.asarray(nrem["epoch_times"]) / 60.0
    delta = nrem["delta_index_smoothed"]
    speed = nrem["velocity_smoothed"]
    params = nrem["params"]

    figure, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                                gridspec_kw={"height_ratios": (1, 1, 0.3), "hspace": 0.12})

    def shade(ax):
        for start, end in nrem["bout_intervals_s"]:
            ax.axvspan(start / 60, end / 60, color="mediumseagreen", alpha=0.22, lw=0)

    ax = axes[0]
    shade(ax)
    ax.plot(time_min, delta, color="steelblue", lw=0.8)
    ax.axhline(params["delta_thresh"], color="crimson", lw=1.2, ls="--",
               label=f"delta > {params['delta_thresh']:.3f}")
    ax.set_ylabel("Delta index")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{session_label} -- NREM bouts (green)", fontsize=11)

    ax = axes[1]
    shade(ax)
    ax.plot(time_min, np.log10(speed + 1.0), color="darkorange", lw=0.8)
    ax.axhline(np.log10(params["velocity_thresh"] + 1.0), color="crimson", lw=1.2, ls="--",
               label=f"speed < {params['velocity_thresh']:g} px/s")
    ax.set_ylabel("log10(speed + 1)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    shade(ax)
    for start, end in nrem["consolidated_windows_s"]:
        ax.axvspan(start / 60, end / 60, color="darkgreen", alpha=0.55, lw=0)
    ax.set_yticks([])
    ax.set_ylabel("bouts", rotation=0, ha="right", va="center", fontsize=9)
    ax.set_xlabel("Time from epoch start (minutes)")
    ax.set_xlim(0, time_min[-1])

    total = sum(e - s for s, e in nrem["bout_intervals_s"])
    consolidated_total = sum(e - s for s, e in nrem["consolidated_windows_s"])
    longest = nrem["fully_asleep_window_s"]
    velocity_window = params.get("velocity_window_sec")
    window_note = f"{velocity_window:g}s window" if velocity_window else "epoch mean"
    stamp_figure(figure,
                 f"{source_note}  |  delta>{params['delta_thresh']:.3f} "
                 f"(smooth={params['smooth_epochs']}ep) "
                 f"speed<{params['velocity_thresh']:g}px/s ({window_note})  "
                 f"merge_gap={params['merge_gap_sec']:g}s min_bout={params['min_bout_sec']:g}s "
                 f"consolidated={params['consolidated_sec']:g}s  |  "
                 f"{len(nrem['bout_intervals_s'])} bouts {total / 60:.1f} min, "
                 f"{len(nrem['consolidated_windows_s'])} consolidated {consolidated_total / 60:.1f} min"
                 + (f", longest {longest[0]:.0f}-{longest[1]:.0f}s" if longest else "")
                 + "  |  reproduce: python score_nrem_delta_velocity.py --build")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# =====================================================
# FIGURE
# =====================================================

def stamp_figure(figure, text):
    """Embed a reproducibility line (what made this figure, from what, when)."""
    figure.text(0.005, 0.002,
                f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by "
                f"score_nrem_delta_velocity.py  |  {text}",
                fontsize=6, color="0.4", ha="left", va="bottom")


def plot_distributions(result, output_path, *, epoch_sec, session_label, source_note):
    delta = result["delta_index"]
    time = result["epoch_time"]
    speed = result.get("velocity_mean")
    speed_median = result.get("velocity_median")
    has_velocity = speed is not None and np.isfinite(speed).any()

    delta_otsu = otsu_threshold(delta)
    log_raw = np.log10(result["delta_raw_power"] + 1e-12)
    modes = {
        "delta_index": find_modes(delta),
        "delta_raw_log10": find_modes(log_raw),
        "log_speed": find_modes(np.log10(speed + 1.0)) if has_velocity else None,
    }

    figure = plt.figure(figsize=(14, 13))
    grid = figure.add_gridspec(4, 2, hspace=0.42, wspace=0.22,
                               height_ratios=(1, 1, 1.15, 0.95))

    # --- delta index -----------------------------------------------------
    ax = figure.add_subplot(grid[0, 0])
    ax.hist(delta, bins=120, color="steelblue", edgecolor="none")
    ax.axvline(delta_otsu, color="crimson", lw=1.6, ls="--",
               label=f"Otsu = {delta_otsu:.2f}")
    annotate_modes(ax, modes["delta_index"])
    ax.set_xlabel("Delta index  (mean robust-z of log10 delta power)")
    ax.set_ylabel(f"epochs ({epoch_sec:g} s)")
    ax.set_title(f"Delta index -- {result['n_channels']} channels pooled", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = figure.add_subplot(grid[0, 1])
    ax.hist(log_raw, bins=120, color="slateblue", edgecolor="none")
    annotate_modes(ax, modes["delta_raw_log10"])
    ax.set_xlabel("log10 mean delta power  (uV^2, unnormalised)")
    ax.set_ylabel(f"epochs ({epoch_sec:g} s)")
    ax.set_title("Raw pooled delta power, for reference", fontsize=10)

    # --- velocity --------------------------------------------------------
    ax = figure.add_subplot(grid[1, 0])
    if has_velocity:
        finite = speed[np.isfinite(speed)]
        ax.hist(np.log10(finite + 1.0), bins=120, color="darkorange", edgecolor="none")
        annotate_modes(ax, modes["log_speed"])
        ax.set_xlabel("log10(epoch mean speed + 1)   [px/s]")
        ax.set_ylabel(f"epochs ({epoch_sec:g} s)")
        ax.set_title("Speed (log axis reveals the still/moving split)", fontsize=10)
    else:
        ax.text(0.5, 0.5, "velocity unavailable", ha="center", va="center",
                transform=ax.transAxes, color="0.4")

    ax = figure.add_subplot(grid[1, 1])
    if has_velocity:
        finite = speed[np.isfinite(speed)]
        upper = np.percentile(finite, 90)
        ax.hist(finite[finite <= upper], bins=100, color="darkorange", edgecolor="none")
        ax.set_xlabel(f"epoch mean speed [px/s]  (lower 90%, <= {upper:.1f})")
        ax.set_ylabel(f"epochs ({epoch_sec:g} s)")
        ax.set_title("Speed, linear zoom on the still mode", fontsize=10)
    else:
        ax.text(0.5, 0.5, "velocity unavailable", ha="center", va="center",
                transform=ax.transAxes, color="0.4")

    # --- joint -----------------------------------------------------------
    ax = figure.add_subplot(grid[2, 0])
    if has_velocity:
        ok = np.isfinite(delta) & np.isfinite(speed)
        counts, xe, ye = np.histogram2d(delta[ok], np.log10(speed[ok] + 1.0), bins=(80, 80))
        ax.pcolormesh(xe, ye, np.log10(counts.T + 1), cmap="magma", shading="auto")
        ax.axvline(delta_otsu, color="cyan", lw=1.2, ls="--")
        ax.set_xlabel("Delta index")
        ax.set_ylabel("log10(mean speed + 1)  [px/s]")
        ax.set_title("Joint distribution -- NREM = high delta, low speed\n"
                     "(look for the bottom-right cloud)", fontsize=10)
    else:
        ax.text(0.5, 0.5, "velocity unavailable", ha="center", va="center",
                transform=ax.transAxes, color="0.4")

    ax = figure.add_subplot(grid[2, 1])
    if has_velocity:
        finite = np.sort(speed[np.isfinite(speed)])
        ax.plot(finite, np.arange(finite.size) / finite.size * 100,
                color="darkorange", lw=1.5)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("epoch mean speed [px/s]")
        ax.set_ylabel("% of epochs at or below")
        ax.set_title("Speed ECDF -- read a percentile off this", fontsize=10)
        ax.grid(alpha=0.3)
        for pct in (25, 50, 75):
            ax.axhline(pct, color="0.7", lw=0.7, ls=":")
    else:
        ax.text(0.5, 0.5, "velocity unavailable", ha="center", va="center",
                transform=ax.transAxes, color="0.4")

    # --- time course -----------------------------------------------------
    ax = figure.add_subplot(grid[3, :])
    ax.plot(time / 60, delta, color="steelblue", lw=0.7, label="delta index")
    ax.axhline(delta_otsu, color="crimson", lw=1.0, ls="--")
    ax.set_xlabel("Time from epoch start (minutes)")
    ax.set_ylabel("Delta index", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlim(0, time[-1] / 60)
    if has_velocity:
        ax_v = ax.twinx()
        ax_v.plot(time / 60, np.log10(speed + 1.0), color="darkorange", lw=0.6, alpha=0.75)
        ax_v.set_ylabel("log10(speed + 1)", color="darkorange")
        ax_v.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title("Time course -- NREM should be where blue is high and orange is low",
                 fontsize=10)

    figure.suptitle(f"{session_label}  |  NREM scoring inputs  |  "
                    f"{epoch_sec:g} s epochs, shanks {SHANKS}, "
                    f"{result['n_channels']} good channels", fontsize=12)

    speed_note = "velocity=none"
    if has_velocity:
        still = float(np.mean(speed[np.isfinite(speed)] < 5.0) * 100)
        speed_note = (f"speed px/s median={np.nanmedian(speed):.1f} "
                      f"p90={np.nanpercentile(speed, 90):.1f} <5px/s={still:.0f}%")
    dm = modes["delta_index"]
    mode_note = "delta_modes=none(unimodal)"
    if dm is not None:
        mode_note = (f"delta_modes={dm['mode_low']:.3f}/{dm['mode_high']:.3f} "
                     f"trough={dm['trough']:.3f} dip={dm['dip']:.2f}")
    sm = modes["log_speed"]
    speed_mode_note = ("log_speed_modes=none(unimodal)" if sm is None else
                       f"log_speed_modes={sm['mode_low']:.2f}/{sm['mode_high']:.2f} "
                       f"trough={sm['trough']:.2f} dip={sm['dip']:.2f}")

    stamp_figure(figure, f"{source_note}  |  epoch={epoch_sec:g}s  |  "
                         f"shanks={SHANKS} bad_channels={BAD_CHANNELS}  |  "
                         f"delta_otsu={delta_otsu:.3f}  |  {mode_note}  |  "
                         f"{speed_mode_note}  |  {speed_note}  |  "
                         f"reproduce: python score_nrem_delta_velocity.py "
                         f"--sessions {session_label.split()[-1]} --epoch-sec {epoch_sec:g}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return delta_otsu, modes


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="sleep sessions to process (default: all active)")
    parser.add_argument("--epoch-sec", type=float, default=DEFAULT_EPOCH_SEC)
    parser.add_argument("--shanks", type=int, nargs="+", default=None)
    parser.add_argument("--bad-channels", type=int, nargs="*", default=None,
                        help="override BAD_CHANNELS (channel IDs)")
    parser.add_argument("--build", action="store_true",
                        help="also threshold the traces and write *_sleep_periods.pkl")
    parser.add_argument("--delta-thresh", type=float, default=None,
                        help="delta-index threshold (default: this session's Otsu split)")
    parser.add_argument("--velocity-thresh", type=float, default=5.0,
                        help="windowed mean speed must stay below this, px/s "
                             "(default 5). Re-read the histograms after changing "
                             "--velocity-window-sec: the scale moves with it.")
    parser.add_argument("--velocity-window-sec", type=float,
                        default=VELOCITY_WINDOW_SEC,
                        help="window the speed is averaged over, centred on each "
                             f"epoch (default {VELOCITY_WINDOW_SEC:g}s, matching "
                             "the spectrogram window the delta index carries)")
    parser.add_argument("--velocity-min-coverage", type=float,
                        default=VELOCITY_MIN_COVERAGE,
                        help="fraction of expected camera frames a window needs "
                             "before it is scored at all")
    parser.add_argument("--smooth-epochs", type=int, default=3,
                        help="boxcar width for the DELTA index only; the speed "
                             "is smoothed once, by its own window")
    parser.add_argument("--merge-gap-sec", type=float, default=20.0)
    parser.add_argument("--min-bout-sec", type=float, default=60.0)
    parser.add_argument("--consolidated-sec", type=float, default=120.0)
    parser.add_argument("--recompute", action="store_true",
                        help="rebuild the epoch traces from the band-power file instead "
                             "of reusing the cached *_nrem_inputs_*.pkl")
    args = parser.parse_args()

    shanks = args.shanks if args.shanks is not None else SHANKS
    bad = args.bad_channels if args.bad_channels is not None else BAD_CHANNELS

    sessions = active_sleep_sessions(sleep_sessions)
    if args.sessions:
        sessions = {k: v for k, v in sessions.items() if k in args.sessions}
    if not sessions:
        raise SystemExit("No matching sleep sessions.")

    out_dir = Path(resolve_output_folder(Path(low_freq_folder) / "nrem_scoring"))

    print("=" * 74)
    print(f"NREM scoring inputs -- {session_name}")
    print(f"  shanks        : {shanks}")
    print(f"  bad channels  : {bad}  (channel IDs)")
    print(f"  epoch         : {args.epoch_sec:g} s")
    print(f"  speed window  : {args.velocity_window_sec:g} s centred on each epoch "
          f"(>={args.velocity_min_coverage:.0%} camera coverage)")
    print(f"  output        : {out_dir}")

    for key, cfg in sessions.items():
        suffix = cfg["suffix"].lstrip("_")
        label = f"{session_name} {suffix}"
        print("\n" + "#" * 74)
        print(f"SESSION {suffix}")

        # The cached inputs pkl is preferred: it is small, already carries the
        # epoch traces, and is immune to the band-power file being rewritten by
        # another pipeline stage mid-run.
        cache_path = out_dir / f"{session_name}_{suffix}_nrem_inputs_{args.epoch_sec:g}s.pkl"
        result = None
        if cache_path.is_file() and not args.recompute:
            with cache_path.open("rb") as file:
                result = pickle.load(file)
            print(f"  reusing cached traces: {cache_path.name} "
                  f"({result['delta_index'].size} epochs, {result['n_channels']} channels, "
                  f"shanks {result.get('shanks')})")
            if sorted(result.get("shanks", [])) != sorted(shanks) or \
                    sorted(result.get("bad_channels", [])) != sorted(bad):
                print("    WARNING: cached traces used different shanks/bad channels; "
                      "pass --recompute to rebuild")
            # The cache filename keys on epoch_sec only, so a pkl written before
            # the speed moved to a matched window - or with a different window -
            # would silently supply the old per-epoch means. Redo just the
            # velocity: it reads two small pkls, not the band powers.
            if result.get("velocity_window_sec") != args.velocity_window_sec:
                print(f"    velocity window in cache "
                      f"({result.get('velocity_window_sec', 'epoch mean')}) != "
                      f"{args.velocity_window_sec:g}s - recomputing velocity")
                velocity = load_velocity(
                    cfg, result["epoch_time"], args.epoch_sec,
                    window_sec=args.velocity_window_sec,
                    min_coverage=args.velocity_min_coverage)
                if velocity:
                    result.update(velocity)

        if result is None:
            band_file = resolve_existing_file(
                Path(low_freq_folder) / f"{session_name}_{suffix}_all_shanks_band_powers.pkl")
            if not band_file.is_file():
                print(f"  band powers not found: {band_file}")
                continue
            print(f"  band powers: {band_file}")
            result = load_delta_index(band_file, shanks, bad, args.epoch_sec)
            velocity = load_velocity(cfg, result["epoch_time"], args.epoch_sec,
                                     window_sec=args.velocity_window_sec,
                                     min_coverage=args.velocity_min_coverage)
            if velocity:
                result.update(velocity)
            result["band_powers_file"] = str(band_file)
        band_file = Path(result.get("band_powers_file", "cached"))

        figure_path = out_dir / f"{session_name}_{suffix}_nrem_inputs_{args.epoch_sec:g}s.png"
        otsu, modes = plot_distributions(
            result, figure_path, epoch_sec=args.epoch_sec, session_label=label,
            source_note=f"src={band_file.name}")

        result.update({
            "session": session_name, "sleep_session": suffix,
            "epoch_sec": args.epoch_sec, "shanks": shanks, "bad_channels": bad,
            "delta_otsu": otsu, "modes": modes, "band_powers_file": str(band_file),
        })
        pkl_path = out_dir / f"{session_name}_{suffix}_nrem_inputs_{args.epoch_sec:g}s.pkl"
        with pkl_path.open("wb") as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

        delta = result["delta_index"]
        print(f"\n  epochs                : {delta.size} x {args.epoch_sec:g} s "
              f"= {delta.size * args.epoch_sec / 60:.0f} min")
        print(f"  delta index           : min {delta.min():.2f}  median "
              f"{np.median(delta):.2f}  max {delta.max():.2f}")
        print(f"  delta Otsu split      : {otsu:.3f}  "
              f"({np.mean(delta > otsu) * 100:.1f}% of epochs above)")
        dm = modes["delta_index"]
        if dm is None:
            print("  delta modes           : UNIMODAL -- no trough; do not threshold on a dip")
        else:
            print(f"  delta modes           : {dm['mode_low']:.3f} and {dm['mode_high']:.3f}, "
                  f"trough {dm['trough']:.3f} (dip {dm['dip']:.2f}, "
                  f"{dm['fraction_above_trough'] * 100:.1f}% above)")
        sm = modes["log_speed"]
        if sm is None:
            print("  log-speed modes       : UNIMODAL -- pick a speed cut from the ECDF, not a dip")
        else:
            print(f"  log-speed modes       : {sm['mode_low']:.2f} and {sm['mode_high']:.2f}, "
                  f"trough {sm['trough']:.2f} (dip {sm['dip']:.2f}) "
                  f"-> {10 ** sm['trough'] - 1:.1f} px/s")
        speed = result.get("velocity_mean")
        if speed is not None and np.isfinite(speed).any():
            good = np.isfinite(speed)
            print(f"  speed [px/s]          : median {np.median(speed[good]):.1f}  "
                  f"p25 {np.percentile(speed[good], 25):.1f}  "
                  f"p75 {np.percentile(speed[good], 75):.1f}")
            for thresh in (1, 2, 5, 10, 20):
                print(f"    epochs < {thresh:>3} px/s     : "
                      f"{np.mean(speed[good] < thresh) * 100:5.1f}%")
            joint = np.mean((delta[good] > otsu) & (speed[good] < 5)) * 100
            print(f"  delta>Otsu AND <5 px/s: {joint:.1f}% of epochs "
                  f"({joint / 100 * delta.size * args.epoch_sec / 60:.0f} min)")
        else:
            print("  speed                 : unavailable")
        print(f"\n  figure: {figure_path}")
        print(f"  data  : {pkl_path}")

        if not args.build:
            continue

        delta_thresh = args.delta_thresh if args.delta_thresh is not None else otsu
        nrem = build_nrem(
            result,
            delta_thresh=delta_thresh,
            velocity_thresh=args.velocity_thresh,
            smooth_epochs=args.smooth_epochs,
            merge_gap_sec=args.merge_gap_sec,
            min_bout_sec=args.min_bout_sec,
            consolidated_sec=args.consolidated_sec,
            sleep_start_sample=cfg.get("start_sample"),
        )

        # Written into low_freq/ under the name detect_off_states.py looks for.
        periods_dir = Path(resolve_output_folder(Path(low_freq_folder)))
        periods_path = periods_dir / f"{session_name}_{suffix}_sleep_periods.pkl"
        with periods_path.open("wb") as file:
            pickle.dump(nrem, file, protocol=pickle.HIGHEST_PROTOCOL)
        nrem_figure = out_dir / f"{session_name}_{suffix}_nrem_bouts_{args.epoch_sec:g}s.png"
        plot_nrem(nrem, nrem_figure, session_label=label,
                  source_note=f"src={cache_path.name}")

        bouts = nrem["bout_intervals_s"]
        consolidated = nrem["consolidated_windows_s"]
        total = sum(e - s for s, e in bouts)
        session_sec = result["epoch_time"][-1] + args.epoch_sec / 2
        print(f"\n  --- NREM (delta > {delta_thresh:.3f}, speed < "
              f"{args.velocity_thresh:g} px/s) ---")
        print(f"  bouts >= {args.min_bout_sec:g}s   : {len(bouts)}, {total / 60:.1f} min "
              f"({total / session_sec * 100:.1f}% of the session)")
        print(f"  consolidated >= {args.consolidated_sec:g}s: {len(consolidated)}, "
              f"{sum(e - s for s, e in consolidated) / 60:.1f} min")
        longest = nrem["fully_asleep_window_s"]
        if longest:
            print(f"  longest window        : {longest[0]:.0f} - {longest[1]:.0f} s "
                  f"({longest[1] - longest[0]:.0f} s)")
        for index, (start, end) in enumerate(consolidated[:12]):
            print(f"    [{index}] {start:8.0f} - {end:8.0f} s   ({end - start:6.0f} s)")
        if len(consolidated) > 12:
            print(f"    ... and {len(consolidated) - 12} more")
        print(f"  periods pkl: {periods_path}")
        print(f"  bouts figure: {nrem_figure}")

    print("\n" + "=" * 74)
    print("Read the histograms, pick a delta and a speed threshold, then tell me\n"
          "and I'll add the bout-building step (merge gaps, drop short bouts).")
    print("=" * 74)


if __name__ == "__main__":
    main()
