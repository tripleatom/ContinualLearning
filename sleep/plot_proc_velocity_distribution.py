from __future__ import annotations

import argparse
import errno
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleep.proc_func_velocity import (DEFAULT_CUTOFF_HZ, DEFAULT_FILTER_ORDER,
                                      DEFAULT_MAX_GAP_SEC, VELOCITY_SOURCES,
                                      aggregate_speed, compute_velocity_advanced,
                                      compute_velocity_from_keypoints,
                                      proc_session_name, velocity_output_name)
from server_fallback import (resolve_output_folder, resolve_existing_file,
                             mirror_on_backup_server)


DEFAULT_PROC_FILE = (
    r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260313"
    r"\video\front_camera_CnL42_2026-03-13_4_PROC"
)

# Defaults for the reduction onto a regular grid. One point per second, each
# averaging its own second, so consecutive points share no frames. Pass
# --window-sec 10 to match the NREM gate's window instead (that overlaps
# neighbouring points by 90%). Not imported from sleep_pipeline_config so this
# stays a standalone tool that can be pointed at any PROC file.
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_STEP_SEC = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot velocity distribution from a *_PROC tracking pickle."
    )
    parser.add_argument("--proc-file", type=Path, default=Path(DEFAULT_PROC_FILE))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source", choices=sorted(VELOCITY_SOURCES),
                        default="proc_center",
                        help="which tracked point to use: 'proc_center' (the PROC "
                             "file's own head centre) or 'dlc_body' (the DLC trunk "
                             "centroid). Each has its own pkl.")
    parser.add_argument("--velocity-threshold", type=float, default=530)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                        help="position low-pass cutoff, Hz (applied before "
                             "differentiating; see proc_func_velocity)")
    parser.add_argument("--filter-order", type=int, default=DEFAULT_FILTER_ORDER)
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                        help="window the speed is averaged over before its "
                             f"distribution is taken (default {DEFAULT_WINDOW_SEC:g}s, "
                             "the spectrogram window the NREM gate also uses)")
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC,
                        help="spacing of the regular grid the speed is placed on "
                             f"(default {DEFAULT_STEP_SEC:g}s)")
    parser.add_argument("--per-frame", action="store_true",
                        help="skip the reduction and show the raw camera frames. "
                             "The frame rate is not fixed, so this weights "
                             "densely-sampled stretches more heavily - useful for "
                             "judging the tracking itself, not for picking a "
                             "threshold anything downstream will apply.")
    parser.add_argument("--linear-bins", action="store_true",
                        help="use linear bins of --bin-width instead of log-spaced "
                             "ones. Speed spans three orders of magnitude, so linear "
                             "bins put the whole still mode in the first bar.")
    parser.add_argument("--bin-width", type=float, default=2.0,
                        help="bin width for --linear-bins")
    parser.add_argument("--max-plot-velocity", type=float, default=None)
    parser.add_argument("--recompute", action="store_true",
                        help="recompute from the PROC/DLC file even when a velocity "
                             "pkl already exists. Needed after the filtering changed: "
                             "an older pkl was built by the previous method.")
    return parser.parse_args()


def load_or_compute_velocity(args):
    proc_file = args.proc_file
    # Look for an existing velocity file on either server, but write a new one
    # to the new server (the old one is full).
    pkl_name = velocity_output_name(proc_file, args.source)
    velocity_file = resolve_existing_file(proc_file.parent / pkl_name)

    if velocity_file.exists() and not args.recompute:
        print(f"Loading existing velocity file: {velocity_file}")
        with open(velocity_file, "rb") as f:
            data = pickle.load(f)
        # 'cutoff_hz' is only written by the position-filtering method; without
        # it the pkl predates that change and its speed is not comparable.
        if "cutoff_hz" not in data:
            print("  WARNING: this pkl was written by the OLD velocity method "
                  "(no position low-pass). Pass --recompute to rebuild it.")
        return (
            np.asarray(data["time_stamp"], dtype=float),
            np.asarray(data["velocity"], dtype=float),
            velocity_file,
        )

    print(f"Computing velocity from PROC file: {proc_file}")
    compute = (compute_velocity_from_keypoints if args.source == "dlc_body"
               else compute_velocity_advanced)
    t, velocity, vx, vy, info = compute(
        proc_file,
        velocity_threshold=args.velocity_threshold,
        cutoff_hz=args.cutoff_hz,
        order=args.filter_order,
        max_gap_sec=args.max_gap_sec,
    )
    velocity_data = {
        "time_stamp": t,
        "velocity": velocity,
        "velocity_x": vx,
        "velocity_y": vy,
        "source_proc_file": str(proc_file),
        "source_proc_name": proc_file.name,
        **info,
    }
    velocity_file = resolve_output_folder(proc_file.parent) / pkl_name
    try:
        with open(velocity_file, "wb") as f:
            pickle.dump(velocity_data, f)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        backup_dir = mirror_on_backup_server(velocity_file.parent)
        if backup_dir is None:
            raise
        backup_dir.mkdir(parents=True, exist_ok=True)
        velocity_file = backup_dir / velocity_file.name
        print(f"Out of space while saving - retrying on backup server: {velocity_file}")
        with open(velocity_file, "wb") as f:
            pickle.dump(velocity_data, f)
    print(f"Saved velocity file: {velocity_file}")
    return np.asarray(t, dtype=float), np.asarray(velocity, dtype=float), velocity_file


def finite_velocity(velocity):
    values = np.asarray(velocity, dtype=float)
    values = values[np.isfinite(values)]
    return values[values >= 0]


def reduce_to_grid(time_stamp, velocity, window_sec, step_sec):
    """Average the speed onto a regular grid, and say what that changed.

    The camera's frame rate is not fixed, so a histogram of the raw frames
    weights each frame equally regardless of how much time it stands for, and
    stretches where frames happen to arrive closer together are over-
    represented. One value per `step_sec` of recording removes that, and taking
    it over `window_sec` makes the distribution the same one the NREM speed
    gate thresholds.
    """
    time_stamp = np.asarray(time_stamp, dtype=float)
    grid = np.arange(time_stamp[0], time_stamp[-1], float(step_sec))
    windowed = aggregate_speed(time_stamp, velocity, grid, window_sec,
                               min_coverage=0.5)
    return windowed["mean"], grid


def make_distribution_plot(time_stamp, velocity, proc_file, velocity_file,
                           output_path, bin_width, max_plot_velocity,
                           log_bins=True, note=""):
    values = finite_velocity(velocity)
    if values.size == 0:
        raise ValueError("No finite nonnegative velocity values found.")

    if max_plot_velocity is None:
        max_plot_velocity = float(np.nanpercentile(values, 99.5))

    mean = float(np.nanmean(values))
    median = float(np.nanmedian(values))
    p95 = float(np.nanpercentile(values, 95))
    p99 = float(np.nanpercentile(values, 99))
    stationary_frac = float(np.mean(values < 1.0))

    # Speed spans three orders of magnitude with most of its mass in the bottom
    # one, so linear bins of a couple of px/s put the entire still mode inside
    # the first bar and show nothing about it. Log bins resolve it; the linear
    # form is kept behind --linear-bins for comparison with older figures.
    positive = values[values > 0]
    if log_bins and positive.size >= 10:
        lo = float(np.percentile(positive, 0.1))
        bins = np.logspace(np.log10(lo), np.log10(max(positive.max(), lo * 10)), 61)
        x_scale, x_lim = "log", (bins[0], bins[-1])
    else:
        max_plot_velocity = max(max_plot_velocity, bin_width)
        bins = np.arange(0, max_plot_velocity + bin_width, bin_width)
        x_scale, x_lim = "linear", (0, max_plot_velocity)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].hist(values, bins=bins, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(median, color="black", linestyle="--", linewidth=1.2, label=f"median={median:.3g}")
    axes[0].axvline(p95, color="tab:orange", linestyle="--", linewidth=1.2, label=f"p95={p95:.3g}")
    axes[0].set_xscale(x_scale)
    axes[0].set_xlim(*x_lim)
    axes[0].set_xlabel("Velocity (px/s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Velocity Distribution{note}")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    sorted_values = np.sort(values)
    cdf = np.arange(1, sorted_values.size + 1) / sorted_values.size
    axes[1].plot(sorted_values, cdf, color="black", linewidth=1.5)
    axes[1].set_xscale(x_scale)
    axes[1].set_xlim(*x_lim)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Velocity (px/s)")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("Velocity CDF")
    axes[1].grid(True, alpha=0.25)

    # Reserve the bottom strip for the stamp, so it cannot land on the CDF's
    # x-axis label.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    summary = (
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by "
        f"plot_proc_velocity_distribution.py  |  "
        f"n={values.size:,} | mean={mean:.3g} | median={median:.3g} | "
        f"p95={p95:.3g} | p99={p99:.3g} | velocity<1={stationary_frac:.1%}"
        f" | bins={x_scale}{note}\n"
        f"PROC: {proc_file}\nVelocity: {velocity_file}"
    )
    fig.text(0.01, 0.01, summary, fontsize=8, color="0.25", ha="left", va="bottom")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n": int(values.size),
        "mean": mean,
        "median": median,
        "p95": p95,
        "p99": p99,
        "stationary_fraction_velocity_lt_1": stationary_frac,
        "output_path": str(output_path),
    }


def main():
    args = parse_args()
    proc_file = args.proc_file
    session_name = proc_session_name(proc_file)
    output_dir = resolve_output_folder(args.output_dir or (proc_file.parent / "figures"))
    suffix = "_per_frame" if args.per_frame else ""
    output_path = (output_dir /
                   f"{session_name}_{args.source}_velocity_distribution{suffix}.png")

    time_stamp, velocity, velocity_file = load_or_compute_velocity(args)

    if args.per_frame:
        frame_rate = 1.0 / float(np.median(np.diff(time_stamp)))
        note = f" (per camera frame, {frame_rate:.1f} fps)"
        print(f"Using raw camera frames: {velocity.size:,} samples at "
              f"{frame_rate:.1f} fps")
    else:
        n_before = velocity.size
        velocity, grid = reduce_to_grid(time_stamp, velocity,
                                        args.window_sec, args.step_sec)
        time_stamp = grid
        note = f" ({args.window_sec:g}s window, {args.step_sec:g}s grid)"
        print(f"Reduced {n_before:,} camera frames -> {velocity.size:,} points on a "
              f"{args.step_sec:g}s grid ({args.window_sec:g}s window, "
              f"{int(np.sum(~np.isfinite(velocity))):,} below 50% coverage)")

    stats = make_distribution_plot(
        time_stamp,
        velocity,
        proc_file,
        velocity_file,
        output_path,
        args.bin_width,
        args.max_plot_velocity,
        log_bins=not args.linear_bins,
        note=note,
    )

    print("Velocity distribution summary")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
