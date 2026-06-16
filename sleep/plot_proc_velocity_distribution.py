from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleep.proc_func_velocity import compute_velocity_advanced, proc_session_name


DEFAULT_PROC_FILE = (
    r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260313"
    r"\video\front_camera_CnL42_2026-03-13_4_PROC"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot velocity distribution from a *_PROC tracking pickle."
    )
    parser.add_argument("--proc-file", type=Path, default=Path(DEFAULT_PROC_FILE))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--velocity-threshold", type=float, default=530)
    parser.add_argument("--window-length", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--max-plot-velocity", type=float, default=None)
    return parser.parse_args()


def load_or_compute_velocity(args):
    proc_file = args.proc_file
    session_name = proc_session_name(proc_file)
    velocity_file = proc_file.parent / f"{session_name}_velocity_advanced.pkl"

    if velocity_file.exists():
        print(f"Loading existing velocity file: {velocity_file}")
        with open(velocity_file, "rb") as f:
            data = pickle.load(f)
        return (
            np.asarray(data["time_stamp"], dtype=float),
            np.asarray(data["velocity"], dtype=float),
            velocity_file,
        )

    print(f"Computing velocity from PROC file: {proc_file}")
    t, velocity, vx, vy = compute_velocity_advanced(
        proc_file,
        velocity_threshold=args.velocity_threshold,
        window_length=args.window_length,
        polyorder=args.polyorder,
    )
    velocity_data = {
        "time_stamp": t,
        "velocity": velocity,
        "velocity_x": vx,
        "velocity_y": vy,
        "source_proc_file": str(proc_file),
        "source_proc_name": proc_file.name,
    }
    with open(velocity_file, "wb") as f:
        pickle.dump(velocity_data, f)
    print(f"Saved velocity file: {velocity_file}")
    return np.asarray(t, dtype=float), np.asarray(velocity, dtype=float), velocity_file


def finite_velocity(velocity):
    values = np.asarray(velocity, dtype=float)
    values = values[np.isfinite(values)]
    return values[values >= 0]


def make_distribution_plot(time_stamp, velocity, proc_file, velocity_file, output_path, bin_width, max_plot_velocity):
    values = finite_velocity(velocity)
    if values.size == 0:
        raise ValueError("No finite nonnegative velocity values found.")

    if max_plot_velocity is None:
        max_plot_velocity = float(np.nanpercentile(values, 99.5))
    max_plot_velocity = max(max_plot_velocity, bin_width)
    bins = np.arange(0, max_plot_velocity + bin_width, bin_width)

    mean = float(np.nanmean(values))
    median = float(np.nanmedian(values))
    p95 = float(np.nanpercentile(values, 95))
    p99 = float(np.nanpercentile(values, 99))
    stationary_frac = float(np.mean(values < 1.0))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].hist(values, bins=bins, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(median, color="black", linestyle="--", linewidth=1.2, label=f"median={median:.2f}")
    axes[0].axvline(p95, color="tab:orange", linestyle="--", linewidth=1.2, label=f"p95={p95:.2f}")
    axes[0].set_xlim(0, max_plot_velocity)
    axes[0].set_xlabel("Velocity")
    axes[0].set_ylabel("Frame count")
    axes[0].set_title("Velocity Distribution")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    sorted_values = np.sort(values)
    cdf = np.arange(1, sorted_values.size + 1) / sorted_values.size
    axes[1].plot(sorted_values, cdf, color="black", linewidth=1.5)
    axes[1].set_xlim(0, max_plot_velocity)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Velocity")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("Velocity CDF")
    axes[1].grid(True, alpha=0.25)

    summary = (
        f"n={values.size:,} | mean={mean:.2f} | median={median:.2f} | "
        f"p95={p95:.2f} | p99={p99:.2f} | velocity<1={stationary_frac:.1%}\n"
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
    output_dir = args.output_dir or (proc_file.parent / "figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{session_name}_velocity_distribution.png"

    time_stamp, velocity, velocity_file = load_or_compute_velocity(args)
    stats = make_distribution_plot(
        time_stamp,
        velocity,
        proc_file,
        velocity_file,
        output_path,
        args.bin_width,
        args.max_plot_velocity,
    )

    print("Velocity distribution summary")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
