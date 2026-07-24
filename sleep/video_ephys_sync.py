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

from trodes_io.DIO import concatenate_din_data, get_dio_folders


# Default paths for interactive/script use.
proc_file = r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL42\260324\video\front_camera_CnL42_2026-03-24_2_PROC"
rec_file = r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL42\260324\CnL42SG_20260324\CnL42_presleep_20260324_174238.rec"
DEFAULT_DIO_CHANNEL = 2
DEFAULT_FS = 30000


def load_proc_signal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load video-side sync signal from a trusted local *_PROC pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    missing = [key for key in ("signal_time", "signal") if key not in data]
    if missing:
        keys = ", ".join(sorted(map(str, data.keys())))
        raise KeyError(f"{path} is missing {missing}. Available keys: {keys}")

    signal_time = np.asarray(data["signal_time"], dtype=float).ravel()
    signal = np.asarray(data["signal"], dtype=float).ravel()
    if signal_time.size != signal.size:
        raise ValueError(
            f"PROC signal_time and signal lengths differ: "
            f"{signal_time.size} vs {signal.size}"
        )
    return signal_time, signal


def load_rec_dio(rec_path: Path, channel_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load SpikeGadgets/Trodes DIO event times and states from a .rec folder."""
    dio_folders = get_dio_folders(rec_path)
    if not dio_folders:
        raise FileNotFoundError(
            f"No *.DIO folder found under {rec_path}. "
            "Run Trodes export for DIO first, then retry."
        )

    time, state = concatenate_din_data(dio_folders, channel_id)
    if time is None or state is None:
        raise FileNotFoundError(
            f"No Din{channel_id}.dat file found in DIO folders under {rec_path}"
        )
    return np.asarray(time, dtype=float).ravel(), np.asarray(state, dtype=float).ravel()


def threshold_signal(signal: np.ndarray, threshold: float | None) -> tuple[np.ndarray, float]:
    """Convert an analog/digital square wave trace to 0/1."""
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError("Signal contains no finite values.")
    if threshold is None:
        threshold = 0.5 * (float(np.nanmin(finite)) + float(np.nanmax(finite)))
    return (signal > threshold).astype(np.int8), float(threshold)


def sampled_edges(time: np.ndarray, signal: np.ndarray, threshold: float | None):
    """Detect rising/falling edges from a sampled PROC square wave."""
    binary, threshold = threshold_signal(signal, threshold)
    transitions = np.diff(binary.astype(np.int8))
    rising_idx = np.flatnonzero(transitions == 1) + 1
    falling_idx = np.flatnonzero(transitions == -1) + 1
    return time[rising_idx], time[falling_idx], binary, threshold


def dio_edges(time: np.ndarray, state: np.ndarray):
    """Extract rising/falling edge times from Trodes DIO event state rows."""
    state = state.astype(np.int8)
    rising = time[state == 1]
    falling = time[state == 0]
    return rising, falling, state


def clean_short_dio_states(
    time: np.ndarray,
    state: np.ndarray,
    fs: float,
    min_duration_sec: float | None = None,
):
    """Delete very short DIO states caused by bounce/glitch transitions."""
    time = np.asarray(time, dtype=float).ravel()
    state = np.asarray(state, dtype=np.int8).ravel()
    if time.size != state.size or time.size < 3:
        return time, state, {"removed_transition_count": 0, "min_duration_sec": min_duration_sec}

    if min_duration_sec is None:
        durations = np.diff(time) / fs
        positive = durations[durations > 0]
        if positive.size == 0:
            min_duration_sec = 0.0
        else:
            min_duration_sec = 0.5 * float(np.nanmedian(positive))

    removed = []
    changed = True
    while changed and time.size >= 3:
        changed = False
        durations = np.diff(time) / fs
        short = np.flatnonzero(durations < min_duration_sec)
        # Ignore boundary truncation; only delete internal short states.
        short = short[(short > 0) & (short < time.size - 2)]
        if short.size == 0:
            break

        idx = int(short[0])
        # time[idx] enters a very short state; time[idx + 1] exits it.
        removed.append(
            {
                "enter_index": idx,
                "exit_index": idx + 1,
                "duration_sec": float(durations[idx]),
                "state": int(state[idx]),
            }
        )
        keep = np.ones(time.size, dtype=bool)
        keep[[idx, idx + 1]] = False
        time = time[keep]
        state = state[keep]
        changed = True

    return time, state, {
        "removed_transition_count": int(2 * len(removed)),
        "removed_short_states": removed,
        "min_duration_sec": float(min_duration_sec),
    }


def duty_cycle_diagnostics(time: np.ndarray, state: np.ndarray, fs: float) -> dict:
    """Summarize DIO high/low durations and duty-cycle outliers."""
    if time.size < 2:
        return {}
    durations = np.diff(time) / fs
    state_for_duration = state[:-1].astype(np.int8)
    result = {}
    for label, value in (("low", 0), ("high", 1)):
        vals = durations[state_for_duration == value]
        if vals.size:
            med = float(np.nanmedian(vals))
            mad = float(np.nanmedian(np.abs(vals - med)))
            tol = max(0.02, 8.0 * mad)
            bad = np.flatnonzero((state_for_duration == value) & (np.abs(durations - med) > tol))
            result[f"{label}_duration_median_sec"] = med
            result[f"{label}_duration_mad_sec"] = mad
            result[f"{label}_duration_outlier_count"] = int(bad.size)
        else:
            result[f"{label}_duration_median_sec"] = np.nan
            result[f"{label}_duration_mad_sec"] = np.nan
            result[f"{label}_duration_outlier_count"] = 0
    high = result.get("high_duration_median_sec", np.nan)
    low = result.get("low_duration_median_sec", np.nan)
    result["duty_cycle_median"] = float(high / (high + low)) if np.isfinite(high + low) and (high + low) else np.nan
    return result


def bad_interval_indices(rising_sec: np.ndarray, min_abs_error_sec: float = 0.02) -> tuple[np.ndarray, dict]:
    """Find intervals that break the square-wave period."""
    iti = np.diff(rising_sec)
    if iti.size == 0:
        return np.array([], dtype=int), {}
    med = float(np.nanmedian(iti))
    mad = float(np.nanmedian(np.abs(iti - med)))
    tol = max(min_abs_error_sec, 8.0 * mad)
    bad = np.flatnonzero(np.abs(iti - med) > tol)
    return bad, {
        "period_median_sec": med,
        "period_mad_sec": mad,
        "period_outlier_tolerance_sec": float(tol),
        "period_outlier_count": int(bad.size),
        "period_outlier_indices": bad.astype(int).tolist(),
    }


def segments_from_bad_intervals(n_edges: int, bad_iti: np.ndarray, min_edges: int = 3):
    """Split edge indices into clean contiguous segments."""
    segments = []
    start = 0
    for bad in bad_iti:
        end = int(bad) + 1
        if end - start >= min_edges:
            segments.append((start, end))
        start = end
    if n_edges - start >= min_edges:
        segments.append((start, n_edges))
    return segments


def match_clean_segments(
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    fs: float,
    search_radius_edges: int = 40,
    max_segment_median_error_sec: float = 0.01,
):
    """Match clean DIO rising-edge segments to PROC, skipping missing DIO pulses."""
    sg_sec = sg_rising / fs
    sg_bad, sg_period_diag = bad_interval_indices(sg_sec)
    proc_bad, proc_period_diag = bad_interval_indices(proc_rising)
    sg_segments = segments_from_bad_intervals(sg_sec.size, sg_bad)

    proc_period = float(np.nanmedian(np.diff(proc_rising)))
    proc_matches = []
    sg_matches = []
    segment_rows = []
    prev_proc_end = 0
    prev_sg_end_sec = None

    for seg_start, seg_end in sg_segments:
        n_edges = seg_end - seg_start
        if n_edges < 3:
            continue

        if prev_sg_end_sec is None:
            predicted = 0
            cand_start = 0
            cand_stop = min(search_radius_edges, proc_rising.size - n_edges)
        else:
            gap_edges = int(round((sg_sec[seg_start] - prev_sg_end_sec) / proc_period))
            predicted = prev_proc_end + max(gap_edges, 1)
            cand_start = max(prev_proc_end, predicted - search_radius_edges)
            cand_stop = min(proc_rising.size - n_edges, predicted + search_radius_edges)

        best = None
        sg_iti = np.diff(sg_sec[seg_start:seg_end])
        for proc_start in range(int(cand_start), int(cand_stop) + 1):
            proc_iti = np.diff(proc_rising[proc_start:proc_start + n_edges])
            err = np.abs(proc_iti - sg_iti)
            rank = (float(np.nanmedian(err)), float(np.nanmax(err)))
            if best is None or rank < best[0]:
                best = (rank, proc_start, err)

        if best is None:
            continue
        (median_err, max_err), proc_start, err = best
        if median_err > max_segment_median_error_sec:
            segment_rows.append(
                {
                    "sg_start": int(seg_start),
                    "sg_end": int(seg_end),
                    "accepted": False,
                    "median_iti_error_sec": median_err,
                    "max_iti_error_sec": max_err,
                }
            )
            continue

        proc_end = proc_start + n_edges
        proc_matches.append(proc_rising[proc_start:proc_end])
        sg_matches.append(sg_rising[seg_start:seg_end])
        segment_rows.append(
            {
                "sg_start": int(seg_start),
                "sg_end": int(seg_end),
                "proc_start": int(proc_start),
                "proc_end": int(proc_end),
                "accepted": True,
                "median_iti_error_sec": median_err,
                "max_iti_error_sec": max_err,
                "skipped_proc_edges_before_segment": int(proc_start - prev_proc_end),
            }
        )
        prev_proc_end = proc_end
        prev_sg_end_sec = sg_sec[seg_end - 1]

    if proc_matches:
        proc_matched = np.concatenate(proc_matches)
        sg_matched = np.concatenate(sg_matches)
    else:
        proc_matched = np.array([], dtype=float)
        sg_matched = np.array([], dtype=float)

    diagnostics = {
        "sg_period": sg_period_diag,
        "proc_period": proc_period_diag,
        "clean_segment_count": int(len(sg_segments)),
        "accepted_segment_count": int(sum(row.get("accepted", False) for row in segment_rows)),
        "segments": segment_rows,
        "matched_edge_count": int(proc_matched.size),
        "unmatched_proc_edge_count": int(proc_rising.size - proc_matched.size),
        "unmatched_sg_edge_count": int(sg_rising.size - sg_matched.size),
    }
    return proc_matched, sg_matched, diagnostics


def find_best_edge_match(
    proc_rising: np.ndarray,
    sg_rising_sec: np.ndarray,
    max_start_search: int = 50,
):
    """Find the best overlapping rising-edge segment by interval similarity."""
    if proc_rising.size < 2 or sg_rising_sec.size < 2:
        return 0, 0, min(proc_rising.size, sg_rising_sec.size), np.array([])

    best = None
    proc_start_max = min(max_start_search, proc_rising.size - 2)
    sg_start_max = min(max_start_search, sg_rising_sec.size - 2)

    for proc_start in range(proc_start_max + 1):
        for sg_start in range(sg_start_max + 1):
            n_edges = min(proc_rising.size - proc_start, sg_rising_sec.size - sg_start)
            if n_edges < 2:
                continue
            proc_iti = np.diff(proc_rising[proc_start:proc_start + n_edges])
            sg_iti = np.diff(sg_rising_sec[sg_start:sg_start + n_edges])
            iti_error = np.abs(proc_iti - sg_iti)
            median_error = float(np.nanmedian(iti_error))
            # Prefer lower error, then more shared edges.
            rank = (median_error, -n_edges)
            if best is None or rank < best[0]:
                best = (rank, proc_start, sg_start, n_edges, iti_error)

    if best is None:
        return 0, 0, min(proc_rising.size, sg_rising_sec.size), np.array([])
    _, proc_start, sg_start, n_edges, iti_error = best
    return proc_start, sg_start, n_edges, iti_error


def interval_match(proc_rising: np.ndarray, sg_rising_sec: np.ndarray) -> dict[str, float]:
    """Compare pulse interval patterns in seconds after best-start matching."""
    proc_start, sg_start, n_edges, iti_error = find_best_edge_match(proc_rising, sg_rising_sec)
    if n_edges < 2:
        return {
            "n_compared_edges": int(n_edges),
            "proc_start_edge": int(proc_start),
            "sg_start_edge": int(sg_start),
            "median_iti_error_sec": np.nan,
            "max_iti_error_sec": np.nan,
            "match_score": np.nan,
        }

    proc_iti = np.diff(proc_rising[proc_start:proc_start + n_edges])
    typical_iti = np.nanmedian(np.abs(proc_iti))
    score = 1.0 - np.nanmedian(iti_error) / (typical_iti + 1e-12)
    return {
        "n_compared_edges": int(n_edges),
        "proc_start_edge": int(proc_start),
        "sg_start_edge": int(sg_start),
        "median_iti_error_sec": float(np.nanmedian(iti_error)),
        "max_iti_error_sec": float(np.nanmax(iti_error)),
        "match_score": float(score),
    }


def map_sg_to_proc_time(sg_time_samples: np.ndarray, proc_rising: np.ndarray, sg_rising: np.ndarray):
    """Linearly map ephys sample time to video/PROC seconds using first/last rising edge."""
    if proc_rising.size < 2 or sg_rising.size < 2:
        raise ValueError("Need at least two rising edges in both signals to align.")
    return np.interp(
        sg_time_samples,
        [sg_rising[0], sg_rising[-1]],
        [proc_rising[0], proc_rising[-1]],
    )


def save_sync_pickle(
    output_path: Path,
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    diagnostics: dict,
    proc_path: Path,
    rec_path: Path,
    dio_channel: int,
    sg_time_zero_sample: float = 0.0,
):
    sg_rising_zero_based = sg_rising - sg_time_zero_sample
    sync_times = {
        "proc_rising_time": proc_rising,
        "SG_rising_time": sg_rising_zero_based,
        "diagnostics": diagnostics,
        "source_proc_file": str(proc_path),
        "source_rec_file": str(rec_path),
        "dio_channel": int(dio_channel),
        "SG_time_zero_sample": float(sg_time_zero_sample),
        "SG_rising_time_units": "samples_zero_based_from_rec_dio_start",
    }
    with open(output_path, "wb") as f:
        pickle.dump(sync_times, f)


def plot_alignment(
    output_path: Path,
    proc_time: np.ndarray,
    proc_binary: np.ndarray,
    sg_time_samples: np.ndarray,
    sg_state: np.ndarray,
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    fs: float,
):
    sg_time_proc = map_sg_to_proc_time(sg_time_samples, proc_rising, sg_rising)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(proc_time, proc_binary, color="black", linewidth=0.8)
    axes[0].set_title("PROC video-side square wave")
    axes[0].set_ylabel("state")

    axes[1].step(sg_time_samples / fs, sg_state, where="post", color="tab:blue", linewidth=0.8)
    axes[1].set_title("Ephys DIO square wave")
    axes[1].set_ylabel("state")
    axes[1].set_xlabel("Ephys time (s)")

    axes[2].plot(proc_time, proc_binary, color="black", linewidth=1.0, label="PROC")
    axes[2].step(sg_time_proc, sg_state, where="post", color="tab:blue", linewidth=0.8, alpha=0.75, label="DIO mapped")
    axes[2].set_title("Overlay after first/last-edge linear mapping")
    axes[2].set_ylabel("state")
    axes[2].set_xlabel("PROC/video time (s)")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sample_matches(
    output_path: Path,
    proc_time: np.ndarray,
    proc_binary: np.ndarray,
    sg_time_samples: np.ndarray,
    sg_state: np.ndarray,
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    fs: float,
    window_sec: float = 2.0,
    edge_window_sec: float = 10.0,
    sample_count: int = 15,
):
    """Plot zoomed alignment checks across the matched recording."""
    if proc_rising.size < 2 or sg_rising.size < 2:
        raise ValueError("Need at least two matched rising edges for sample plots.")

    sg_time_proc = map_sg_to_proc_time(sg_time_samples, proc_rising, sg_rising)
    sample_indices = np.unique(
        np.round(np.linspace(0, proc_rising.size - 1, sample_count)).astype(int)
    )

    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(14, 2.2 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]

    for ax, edge_idx in zip(axes, sample_indices):
        center = proc_rising[edge_idx]
        this_window = edge_window_sec if edge_idx in (sample_indices[0], sample_indices[-1]) else window_sec
        half_window = this_window / 2.0
        start = center - half_window
        stop = center + half_window

        proc_mask = (proc_time >= start) & (proc_time <= stop)
        sg_mask = (sg_time_proc >= start) & (sg_time_proc <= stop)

        ax.plot(proc_time[proc_mask], proc_binary[proc_mask], color="black", lw=1.1, label="PROC")
        ax.step(
            sg_time_proc[sg_mask],
            sg_state[sg_mask],
            where="post",
            color="tab:blue",
            lw=0.9,
            alpha=0.8,
            label="DIO mapped",
        )
        ax.axvline(proc_rising[edge_idx], color="black", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlim(start, stop)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("state")
        ax.set_title(
            f"Matched edge {edge_idx:,} / {proc_rising.size - 1:,} "
            f"({this_window:g}s window)"
        )

    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("PROC/video time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return sample_indices


def sample_match_errors(
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    fs: float,
    sample_count: int = 15,
):
    """Return timing errors at representative matched rising edges."""
    if proc_rising.size < 2 or sg_rising.size < 2:
        return []
    sg_rising_proc = map_sg_to_proc_time(sg_rising, proc_rising, sg_rising)
    sample_indices = np.unique(
        np.round(np.linspace(0, proc_rising.size - 1, sample_count)).astype(int)
    )
    rows = []
    for idx in sample_indices:
        rows.append(
            {
                "edge_index": int(idx),
                "proc_time": float(proc_rising[idx]),
                "dio_mapped_time": float(sg_rising_proc[idx]),
                "error_sec": float(sg_rising_proc[idx] - proc_rising[idx]),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare video PROC sync square wave with rec-folder DIO and save sync_times.pkl."
    )
    parser.add_argument("--proc-file", type=Path, default=Path(proc_file))
    parser.add_argument("--rec-file", type=Path, default=Path(rec_file))
    parser.add_argument("--dio-channel", type=int, default=DEFAULT_DIO_CHANNEL)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--proc-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-figure", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--sample-window-sec", type=float, default=12.0)
    parser.add_argument("--sample-edge-window-sec", type=float, default=12.0)
    parser.add_argument("--min-matched-edges", type=int, default=5)
    parser.add_argument("--max-median-iti-error-sec", type=float, default=0.02)
    parser.add_argument("--max-segment-median-error-sec", type=float, default=0.01)
    parser.add_argument("--min-dio-state-duration-sec", type=float, default=None)
    parser.add_argument("--disable-dio-cleaning", action="store_true")
    parser.add_argument("--force-save-sync", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    proc_path = args.proc_file
    rec_path = args.rec_file
    output_path = args.output or (rec_path.parent / "sync_times.pkl")
    sample_figure_path = args.sample_figure or (rec_path.parent / "sync_square_wave_match_samples.png")

    print(f"Loading PROC signal: {proc_path}")
    proc_time, proc_signal = load_proc_signal(proc_path)
    proc_rising, proc_falling, proc_binary, proc_threshold = sampled_edges(
        proc_time, proc_signal, args.proc_threshold
    )

    print(f"Loading rec DIO: {rec_path} Din{args.dio_channel}")
    sg_time, sg_state = load_rec_dio(rec_path, args.dio_channel)
    raw_sg_time = sg_time
    raw_sg_state = sg_state
    if args.disable_dio_cleaning:
        dio_cleaning = {"removed_transition_count": 0, "disabled": True}
    else:
        sg_time, sg_state, dio_cleaning = clean_short_dio_states(
            sg_time,
            sg_state,
            args.fs,
            args.min_dio_state_duration_sec,
        )

    sg_rising, sg_falling, sg_state = dio_edges(sg_time, sg_state)
    raw_sg_rising, raw_sg_falling, _ = dio_edges(raw_sg_time, raw_sg_state)

    proc_rising_matched, sg_rising_matched, segment_diagnostics = match_clean_segments(
        proc_rising,
        sg_rising,
        args.fs,
        max_segment_median_error_sec=args.max_segment_median_error_sec,
    )
    accepted_segments = [row for row in segment_diagnostics["segments"] if row.get("accepted")]
    if accepted_segments:
        segment_medians = np.asarray([row["median_iti_error_sec"] for row in accepted_segments])
        segment_maxes = np.asarray([row["max_iti_error_sec"] for row in accepted_segments])
        median_iti_error = float(np.nanmedian(segment_medians))
        max_iti_error = float(np.nanmax(segment_maxes))
    else:
        median_iti_error = np.nan
        max_iti_error = np.nan

    diagnostics = {}
    diagnostics.update(
        {
            "n_compared_edges": int(proc_rising_matched.size),
            "median_iti_error_sec": median_iti_error,
            "max_iti_error_sec": max_iti_error,
            "dio_cleaning": dio_cleaning,
            "dio_duty_cycle_raw": duty_cycle_diagnostics(raw_sg_time, raw_sg_state, args.fs),
            "dio_duty_cycle_cleaned": duty_cycle_diagnostics(sg_time, sg_state, args.fs),
            "segment_matching": segment_diagnostics,
            "sample_match_errors": sample_match_errors(
                proc_rising_matched,
                sg_rising_matched,
                args.fs,
                args.sample_count,
            ),
            "proc_threshold": proc_threshold,
            "n_proc_rising": int(proc_rising.size),
            "n_proc_falling": int(proc_falling.size),
            "n_sg_rising_raw": int(raw_sg_rising.size),
            "n_sg_falling_raw": int(raw_sg_falling.size),
            "n_sg_rising": int(sg_rising.size),
            "n_sg_falling": int(sg_falling.size),
            "fs": float(args.fs),
        }
    )

    print("\nSync comparison")
    print(f"  PROC rising edges: {proc_rising.size}")
    print(f"  Ephys rising edges raw: {raw_sg_rising.size}")
    print(f"  Ephys rising edges cleaned: {sg_rising.size}")
    print(f"  Removed DIO transitions: {dio_cleaning['removed_transition_count']}")
    print(f"  Compared edges: {diagnostics['n_compared_edges']}")
    print(f"  Accepted clean segments: {segment_diagnostics['accepted_segment_count']}")
    print(f"  Unmatched PROC edges: {segment_diagnostics['unmatched_proc_edge_count']}")
    print(f"  Unmatched ephys edges: {segment_diagnostics['unmatched_sg_edge_count']}")
    print(f"  Median ITI error: {diagnostics['median_iti_error_sec']:.6f} s")
    print(f"  Max ITI error: {diagnostics['max_iti_error_sec']:.6f} s")
    print("  Representative mapped-edge errors:")
    for row in diagnostics["sample_match_errors"]:
        print(
            f"    edge {row['edge_index']:>6}: "
            f"{row['error_sec'] * 1000:+.3f} ms"
        )

    good_match = (
        diagnostics["n_compared_edges"] >= args.min_matched_edges
        and np.isfinite(diagnostics["median_iti_error_sec"])
        and diagnostics["median_iti_error_sec"] <= args.max_median_iti_error_sec
        and segment_diagnostics["accepted_segment_count"] > 0
    )
    if good_match or args.force_save_sync:
        save_sync_pickle(
            output_path,
            proc_rising_matched,
            sg_rising_matched,
            diagnostics,
            proc_path,
            rec_path,
            args.dio_channel,
            raw_sg_time[0],
        )
        print(f"\nSaved sync pickle: {output_path}")
    else:
        print(
            "\nDid not save sync pickle: square waves do not match "
            f"(need >= {args.min_matched_edges} edges and median ITI error <= "
            f"{args.max_median_iti_error_sec} s)."
        )

    plot_sample_matches(
        sample_figure_path,
        proc_time,
        proc_binary,
        sg_time,
        sg_state,
        proc_rising_matched,
        sg_rising_matched,
        args.fs,
        args.sample_window_sec,
        args.sample_edge_window_sec,
        args.sample_count,
    )
    print(f"Saved sample match figure: {sample_figure_path}")


if __name__ == "__main__":
    main()
