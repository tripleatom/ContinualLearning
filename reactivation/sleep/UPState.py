"""
Rule-based UP/DOWN state detection for sleep spike recordings.

The sleep decoder scripts in this repository usually operate on spike pickle
files rather than raw 30 kHz voltage traces. In that setting, population MUA is
estimated from the binned firing-rate matrix: DOWN states are low population
activity, UP states are high population activity, and intermediate bins are
left uncertain.

Usage
-----
Imported by apply_merged_decoder_to_sleep_original.py:
    detect_up_down_from_rates(...)
    plot_up_down_summary(...)

Standalone:
    python reactivation/sleep/UPState.py
    python reactivation/sleep/UPState.py --bin-ms 10 --sleep-label pre
"""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d


code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(code_dir / "reactivation" / "VStimOnDecoding"))


DOWN_LABEL = 0
UP_LABEL = 1
UNCERTAIN_LABEL = -1


@dataclass(frozen=True)
class UpDownParams:
    """Parameters for population-MUA UP/DOWN detection."""

    bin_size_sec: float
    smooth_sigma_sec: float = 0.05
    down_z_threshold: float = -0.5
    up_z_threshold: float = 0.0
    down_percentile: float | None = 20.0
    min_state_duration_sec: float = 0.05
    merge_gap_sec: float = 0.03


def _zscore(x):
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.nanmean(x))) / sd


def _min_bins(duration_sec, bin_size_sec):
    return max(1, int(round(float(duration_sec) / float(bin_size_sec))))


def _remove_short_true_runs(mask, min_bins):
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0:
        return mask

    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    starts = changes[0::2]
    stops = changes[1::2]
    for start, stop in zip(starts, stops):
        if stop - start < min_bins:
            mask[start:stop] = False
    return mask


def _clean_state_mask(mask, min_bins, gap_bins):
    if mask.size == 0:
        return mask
    structure = np.ones(max(1, gap_bins + 1), dtype=bool)
    mask = binary_closing(mask, structure=structure)
    mask = binary_opening(mask, structure=np.ones(max(1, min_bins), dtype=bool))
    return _remove_short_true_runs(mask, min_bins)


def detect_up_down_from_rates(
    X_sleep,
    centers,
    bin_size_sec,
    smooth_sigma_sec=0.05,
    down_z_threshold=-0.5,
    up_z_threshold=0.0,
    down_percentile=20.0,
    min_state_duration_sec=0.05,
    merge_gap_sec=0.03,
):
    """
    Detect UP/DOWN states from a binned sleep firing-rate matrix.

    Parameters
    ----------
    X_sleep : array, shape (n_bins, n_units)
        Binned firing rates in spikes/s, as returned by decode_utils.bin_spikes.
    centers : array, shape (n_bins,)
        Time at the center of each bin, in seconds.
    bin_size_sec : float
        Bin width in seconds.

    Returns
    -------
    result : dict
        Contains state labels, population MUA, z-scored MUA, masks, and event
        tables. Labels are: 1=UP, 0=DOWN, -1=uncertain/transition.
    """
    X_sleep = np.asarray(X_sleep, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if X_sleep.ndim != 2:
        raise ValueError("X_sleep must be a 2D array with shape (n_bins, n_units).")
    if X_sleep.shape[0] != centers.size:
        raise ValueError("X_sleep rows must match the number of bin centers.")
    if centers.size == 0:
        raise ValueError("Cannot detect UP/DOWN states from an empty sleep interval.")

    population_mua = np.nanmean(X_sleep, axis=1)
    sigma_bins = max(0.0, float(smooth_sigma_sec) / float(bin_size_sec))
    if sigma_bins > 0:
        population_mua_smooth = gaussian_filter1d(population_mua, sigma=sigma_bins, mode="nearest")
    else:
        population_mua_smooth = population_mua.astype(float, copy=True)
    population_mua_z = _zscore(population_mua_smooth)

    down_mask = population_mua_z < float(down_z_threshold)
    if down_percentile is not None:
        pct_cut = float(np.nanpercentile(population_mua_smooth, down_percentile))
        down_mask = down_mask | (population_mua_smooth <= pct_cut)

    up_mask = population_mua_z > float(up_z_threshold)
    up_mask = up_mask & ~down_mask

    min_bins = _min_bins(min_state_duration_sec, bin_size_sec)
    gap_bins = _min_bins(merge_gap_sec, bin_size_sec)
    down_mask = _clean_state_mask(down_mask, min_bins, gap_bins)
    up_mask = _clean_state_mask(up_mask & ~down_mask, min_bins, gap_bins)

    labels = np.full(centers.size, UNCERTAIN_LABEL, dtype=int)
    labels[down_mask] = DOWN_LABEL
    labels[up_mask] = UP_LABEL

    return {
        "centers_sec": centers,
        "state_label": labels,
        "population_mua": population_mua,
        "population_mua_smooth": population_mua_smooth,
        "population_mua_z": population_mua_z,
        "up_mask": up_mask,
        "down_mask": down_mask,
        "events": {
            "up": _mask_to_events(centers, up_mask, labels),
            "down": _mask_to_events(centers, down_mask, labels),
        },
        "params": {
            "bin_size_sec": float(bin_size_sec),
            "smooth_sigma_sec": float(smooth_sigma_sec),
            "down_z_threshold": float(down_z_threshold),
            "up_z_threshold": float(up_z_threshold),
            "down_percentile": None if down_percentile is None else float(down_percentile),
            "min_state_duration_sec": float(min_state_duration_sec),
            "merge_gap_sec": float(merge_gap_sec),
        },
    }


def _mask_to_events(centers, mask, labels):
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    starts = changes[0::2]
    stops = changes[1::2]
    events = []
    for start, stop in zip(starts, stops):
        events.append(
            {
                "start_sec": float(centers[start]),
                "end_sec": float(centers[stop - 1]),
                "center_sec": float(np.mean(centers[start:stop])),
                "duration_sec": float(centers[stop - 1] - centers[start]),
                "start_bin": int(start),
                "end_bin": int(stop - 1),
                "label": int(labels[start]),
            }
        )
    return events


def summarize_up_down(updown):
    """Return count and fraction summaries for an UP/DOWN result dict."""
    labels = np.asarray(updown["state_label"], dtype=int)
    n = max(labels.size, 1)
    return {
        "up_bins": int(np.sum(labels == UP_LABEL)),
        "down_bins": int(np.sum(labels == DOWN_LABEL)),
        "uncertain_bins": int(np.sum(labels == UNCERTAIN_LABEL)),
        "up_fraction": float(np.sum(labels == UP_LABEL) / n),
        "down_fraction": float(np.sum(labels == DOWN_LABEL) / n),
        "uncertain_fraction": float(np.sum(labels == UNCERTAIN_LABEL) / n),
        "up_events": int(len(updown["events"]["up"])),
        "down_events": int(len(updown["events"]["down"])),
        "n_bins": int(labels.size),
    }


def print_up_down_report(updown, figure_path=None, prefix=""):
    """Print UP/DOWN label counts, event counts, parameters, and figure path."""
    summary = summarize_up_down(updown)
    lead = f"{prefix} " if prefix else ""

    print(f"\n{lead}=== UP/DOWN state report ===")
    print(f"{lead}UP bins        : {summary['up_bins']} ({summary['up_fraction']:.2%})")
    print(f"{lead}DOWN bins      : {summary['down_bins']} ({summary['down_fraction']:.2%})")
    print(f"{lead}Uncertain bins : {summary['uncertain_bins']} ({summary['uncertain_fraction']:.2%})")
    print(f"{lead}UP events      : {summary['up_events']}")
    print(f"{lead}DOWN events    : {summary['down_events']}")
    print(f"{lead}Parameters     : {updown['params']}")
    if figure_path is not None:
        print(f"{lead}Figure saved   : {figure_path}")


def plot_up_down_summary(out_dir, centers, updown, decoder_pred=None, decoder_proba=None, classes=None):
    """Save a compact validation plot for detected UP/DOWN states."""
    out_dir.mkdir(parents=True, exist_ok=True)
    has_decoder = decoder_pred is not None
    n_rows = 3 if has_decoder else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.7 * n_rows), sharex=True)
    if n_rows == 2:
        axes = np.asarray(axes)

    axes[0].plot(centers, updown["population_mua_z"], color="black", linewidth=0.8)
    axes[0].axhline(updown["params"]["up_z_threshold"], color="#2c7fb8", linestyle=":", linewidth=1)
    axes[0].axhline(updown["params"]["down_z_threshold"], color="#d95f0e", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Population MUA z")
    axes[0].set_title("Sleep UP/DOWN state detection from binned population activity")

    state = updown["state_label"]
    axes[1].step(centers, state, where="mid", color="0.2", linewidth=0.9)
    axes[1].set_yticks([UNCERTAIN_LABEL, DOWN_LABEL, UP_LABEL])
    axes[1].set_yticklabels(["uncertain", "DOWN", "UP"])
    axes[1].set_ylabel("State")

    if has_decoder:
        y_plot = np.zeros_like(decoder_pred, dtype=float)
        y_plot[np.asarray(decoder_pred) == 1] = 1
        y_plot[np.asarray(decoder_pred) == -1] = -1
        axes[2].step(centers, y_plot, where="mid", color="0.25", linewidth=0.8)
        axes[2].set_yticks([-1, 0, 1])
        axes[2].set_yticklabels(["-1", "ITI/0", "+1"])
        axes[2].set_ylabel("Decoder")

    for ax in axes:
        _shade_states(ax, centers, state)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Sleep time in spike-time frame (s)")

    counts = {
        "up_bins": int(np.sum(state == UP_LABEL)),
        "down_bins": int(np.sum(state == DOWN_LABEL)),
        "uncertain_bins": int(np.sum(state == UNCERTAIN_LABEL)),
    }
    fig.text(
        0.01,
        0.01,
        "UP/DOWN detection from population MUA | "
        f"UP={counts['up_bins']} bins, DOWN={counts['down_bins']} bins, "
        f"uncertain={counts['uncertain_bins']} bins | params={updown['params']}",
        ha="left",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig_path = out_dir / "sleep_up_down_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UP/DOWN summary figure saved -> {fig_path}")
    return fig_path


def _shade_states(ax, centers, state):
    for label, color, alpha in [(DOWN_LABEL, "#d95f0e", 0.12), (UP_LABEL, "#2c7fb8", 0.08)]:
        mask = state == label
        for event in _mask_to_events(centers, mask, state):
            ax.axvspan(event["start_sec"], event["end_sec"], color=color, alpha=alpha, linewidth=0)


def load_sleep_rates_from_pkl(sleep_pkl_path, start_sec=None, end_sec=None, bin_size_sec=0.01):
    """Load a sleep spike pickle and return binned firing rates for all units."""
    from decode_utils import bin_spikes

    with open(sleep_pkl_path, "rb") as f:
        data = pickle.load(f)
    spike_data = data["spike_data"]

    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = float(data.get("window", {}).get("window_duration_sec", 0.0))
        if end_sec <= start_sec:
            max_spike = 0.0
            for unit in spike_data.values():
                spikes = np.asarray(unit.get("spike_times_sec", []), dtype=float)
                if spikes.size:
                    max_spike = max(max_spike, float(np.nanmax(spikes)))
            end_sec = max_spike
    if end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be greater than start_sec ({start_sec}).")

    n_bins = int(np.floor((float(end_sec) - float(start_sec)) / float(bin_size_sec)))
    if n_bins < 1:
        raise ValueError("Sleep interval is shorter than the selected UP/DOWN bin size.")
    edges = float(start_sec) + np.arange(n_bins + 1) * float(bin_size_sec)
    X_sleep, units = bin_spikes(spike_data, edges, bin_size_sec)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return X_sleep, centers, units, float(start_sec), float(end_sec)


def run_standalone(
    sleep_pkl_path,
    out_dir,
    label="sleep",
    start_sec=None,
    end_sec=None,
    bin_size_sec=0.01,
    smooth_sigma_sec=0.05,
    down_z_threshold=-0.5,
    up_z_threshold=0.0,
    down_percentile=20.0,
    min_state_duration_sec=0.05,
    merge_gap_sec=0.03,
):
    """Run UP/DOWN detection directly from one sleep spike pickle."""
    out_dir = Path(out_dir) / str(label)
    X_sleep, centers, units, start_sec_eff, end_sec_eff = load_sleep_rates_from_pkl(
        sleep_pkl_path,
        start_sec=start_sec,
        end_sec=end_sec,
        bin_size_sec=bin_size_sec,
    )
    updown = detect_up_down_from_rates(
        X_sleep,
        centers,
        bin_size_sec,
        smooth_sigma_sec=smooth_sigma_sec,
        down_z_threshold=down_z_threshold,
        up_z_threshold=up_z_threshold,
        down_percentile=down_percentile,
        min_state_duration_sec=min_state_duration_sec,
        merge_gap_sec=merge_gap_sec,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_up_down_summary(out_dir, centers, updown)
    print(f"\nSleep pickle   : {sleep_pkl_path}")
    print(f"Sleep interval : {start_sec_eff:.3f} to {end_sec_eff:.3f} s")
    print(f"Units          : {len(units)}")
    print(f"Bins           : {len(centers)}")
    print_up_down_report(updown, figure_path=fig_path)

    pkl_path = out_dir / "sleep_up_down_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "sleep_pkl": str(sleep_pkl_path),
                "sleep_label": label,
                "sleep_start_sec": start_sec_eff,
                "sleep_end_sec": end_sec_eff,
                "unit_labels": units,
                "updown": updown,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Results saved  : {pkl_path}")
    return updown


def _parse_args():
    parser = argparse.ArgumentParser(description="Detect sleep UP/DOWN states from sleep spike pickle files.")
    parser.add_argument("--sleep-pkl", default=None, help="Path to one sleep spike pickle. Defaults to params.sleep_blocks.")
    parser.add_argument("--sleep-label", default=None, help="Only run this label from params.sleep_blocks, e.g. pre or post.")
    parser.add_argument("--start-sec", type=float, default=None, help="Override sleep start time in seconds.")
    parser.add_argument("--end-sec", type=float, default=None, help="Override sleep end time in seconds.")
    parser.add_argument("--bin-ms", type=float, default=10.0, help="UP/DOWN detection bin size in ms.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults beside the task session reactivation folder.")
    parser.add_argument("--smooth-sigma-ms", type=float, default=50.0)
    parser.add_argument("--down-z", type=float, default=-0.5)
    parser.add_argument("--up-z", type=float, default=0.0)
    parser.add_argument("--down-percentile", type=float, default=20.0)
    parser.add_argument("--min-state-ms", type=float, default=50.0)
    parser.add_argument("--merge-gap-ms", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    bin_size_sec = args.bin_ms / 1000.0

    if args.sleep_pkl:
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.sleep_pkl).parent / "reactivation" / "up_down_states"
        run_standalone(
            args.sleep_pkl,
            out_dir,
            label=args.sleep_label or Path(args.sleep_pkl).stem,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            bin_size_sec=bin_size_sec,
            smooth_sigma_sec=args.smooth_sigma_ms / 1000.0,
            down_z_threshold=args.down_z,
            up_z_threshold=args.up_z,
            down_percentile=args.down_percentile,
            min_state_duration_sec=args.min_state_ms / 1000.0,
            merge_gap_sec=args.merge_gap_ms / 1000.0,
        )
        return

    from params import sleep_blocks, task_pkl

    session = Path(task_pkl).parent.name
    out_dir = Path(args.out_dir) if args.out_dir else Path(task_pkl).parent / "reactivation" / f"up_down_states_{session}"
    selected = []
    for label, pkl_path, start_sec, end_sec in sleep_blocks:
        if args.sleep_label is not None and label != args.sleep_label:
            continue
        selected.append((label, pkl_path, start_sec, end_sec))

    if not selected:
        raise ValueError(f"No sleep blocks matched sleep-label={args.sleep_label!r}.")

    for label, pkl_path, start_sec, end_sec in selected:
        if not pkl_path:
            print(f"\n[{label}] Skipped - no sleep_pkl path.")
            continue
        run_standalone(
            pkl_path,
            out_dir,
            label=label,
            start_sec=args.start_sec if args.start_sec is not None else start_sec,
            end_sec=args.end_sec if args.end_sec is not None else end_sec,
            bin_size_sec=bin_size_sec,
            smooth_sigma_sec=args.smooth_sigma_ms / 1000.0,
            down_z_threshold=args.down_z,
            up_z_threshold=args.up_z,
            down_percentile=args.down_percentile,
            min_state_duration_sec=args.min_state_ms / 1000.0,
            merge_gap_sec=args.merge_gap_ms / 1000.0,
        )


if __name__ == "__main__":
    main()
