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
    DIO sync pulses, then reduced to one value per epoch. Both the epoch mean
    and median are kept -- mean punishes a brief movement inside an otherwise
    still epoch, median ignores it.

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
    VELOCITY_SOURCE, active_sleep_sessions, low_freq_folder, original_fs,
    rec_folder, session_name, sleep_sessions, video_folder,
)
from proc_func_velocity import velocity_output_name  # noqa: E402


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


def epoch_reduce(signal, samples_per_epoch, func=np.mean):
    """Reduce a regularly sampled signal to one value per whole epoch."""
    signal = np.asarray(signal, dtype=np.float64)
    n_epochs = signal.size // samples_per_epoch
    if n_epochs == 0:
        return np.array([])
    trimmed = signal[:n_epochs * samples_per_epoch].reshape(n_epochs, samples_per_epoch)
    return func(trimmed, axis=1)


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
        fs = float(entry["sampling_rate"])
        samples_per_epoch = int(round(epoch_sec * fs))
        good = [int(c) for c in sorted(entry["band_powers"]) if int(c) not in bad_channels]
        used[shank] = good
        for channel in good:
            delta = np.asarray(entry["band_powers"][channel]["delta"], dtype=np.float64)
            epoch_power = epoch_reduce(delta, samples_per_epoch)
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
    epoch_time = (np.arange(length) + 0.5) * epoch_sec
    return {
        "delta_index": delta_index,
        "delta_raw_power": delta_raw,
        "epoch_time": epoch_time,
        "channels_used": used,
        "n_channels": stack.shape[0],
        "lfp_fs": fs,
    }


def load_velocity(session_cfg, epoch_time, epoch_sec, verbose=True):
    """Per-epoch speed on the ephys clock, or None when sync/tracking is missing.

    Camera time is mapped onto ephys time by the linear fit between the first and
    last matched DIO pulse pair, exactly as plot_sleep_spectrograms.py does.
    ``SG_rising_time`` is stored in acquisition samples, so it is divided by
    ``original_fs`` to land in the same seconds-from-epoch-start clock the LFP
    and band powers use.
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
        print(f"    velocity: {velocity_file.name} ({speed.size} frames), "
              f"synced to {ephys_time[0]:.1f}-{ephys_time[-1]:.1f} s "
              f"via {sync_file.name}")

    edges = np.concatenate([epoch_time - epoch_sec / 2, [epoch_time[-1] + epoch_sec / 2]])
    index = np.digitize(ephys_time, edges) - 1
    mean_speed = np.full(epoch_time.size, np.nan)
    median_speed = np.full(epoch_time.size, np.nan)
    coverage = np.zeros(epoch_time.size)
    finite = np.isfinite(speed)
    for epoch in range(epoch_time.size):
        selected = (index == epoch) & finite
        n = int(selected.sum())
        coverage[epoch] = n
        if n:
            mean_speed[epoch] = speed[selected].mean()
            median_speed[epoch] = np.median(speed[selected])

    return {
        "velocity_mean": mean_speed,
        "velocity_median": median_speed,
        "frames_per_epoch": coverage,
        "velocity_file": str(velocity_file),
        "sync_file": str(sync_file),
        "ephys_time_range": (float(ephys_time[0]), float(ephys_time[-1])),
    }


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
    print(f"  output        : {out_dir}")

    for key, cfg in sessions.items():
        suffix = cfg["suffix"].lstrip("_")
        label = f"{session_name} {suffix}"
        print("\n" + "#" * 74)
        print(f"SESSION {suffix}")

        band_file = resolve_existing_file(
            Path(low_freq_folder) / f"{session_name}_{suffix}_all_shanks_band_powers.pkl")
        if not band_file.is_file():
            print(f"  band powers not found: {band_file}")
            continue
        print(f"  band powers: {band_file}")

        result = load_delta_index(band_file, shanks, bad, args.epoch_sec)
        velocity = load_velocity(cfg, result["epoch_time"], args.epoch_sec)
        if velocity:
            result.update(velocity)

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
        if velocity:
            speed = result["velocity_mean"]
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

    print("\n" + "=" * 74)
    print("Read the histograms, pick a delta and a speed threshold, then tell me\n"
          "and I'll add the bout-building step (merge gaps, drop short bouts).")
    print("=" * 74)


if __name__ == "__main__":
    main()
