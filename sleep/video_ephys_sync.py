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
from server_fallback import (resolve_existing_file, resolve_output_folder,
                             mirror_on_backup_server)


DEFAULT_DIO_CHANNEL = 2
DEFAULT_FS = 30000

# Which edge-matching algorithm to use (see resolve_match_algorithm):
#   'pulse'     - fixed-frequency square wave. Matches by cutting the train at
#                 period outliers and sliding each clean segment onto PROC.
#   'pulse_geo' - random ("geometric") inter-pulse intervals. Matches by using
#                 the interval sequence itself as a barcode.
#   'auto'      - pick from the spread of the PROC inter-pulse intervals.
MATCH_ALGORITHMS = ("auto", "pulse", "pulse_geo")
DEFAULT_MATCH_ALGORITHM = "auto"

# Robust ITI spread (MAD/median) above which a train is treated as pulse_geo.
# Measured: CnL42 fixed-period days ~0.02-0.04, CnL46 geo days ~0.3-0.4, so
# anywhere in the middle separates them with a wide margin.
GEO_ITI_CV_THRESHOLD = 0.15


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


def default_min_dio_state_duration(time: np.ndarray, fs: float, algorithm: str) -> float:
    """Pick the "shorter than this is a glitch, not a pulse" line for cleaning.

    A fixed-period wave has essentially ONE state duration, so half its median
    is safely below every real state. A pulse_geo train's real states span an
    order of magnitude (0.105-0.62 s on CnL46 260727), and half of THAT median
    lands at 0.125 s - above the genuine short pulses, which would delete ~5%
    of the train. Key off the short tail instead, which sits just under the
    shortest real state while still being far above contact-bounce glitches.
    """
    durations = np.diff(np.asarray(time, dtype=float).ravel()) / fs
    durations = durations[durations > 0]
    if durations.size == 0:
        return 0.0
    if algorithm == "pulse_geo":
        return 0.5 * float(np.percentile(durations, 1))
    return 0.5 * float(np.nanmedian(durations))


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


# =====================================================
# pulse_geo: random-interval pulse trains
# =====================================================
# The 'pulse' matcher above assumes one period, so it can only re-anchor at
# period outliers. When the generator emits RANDOM intervals instead, that
# assumption is gone - but the randomness is itself the solution: a handful of
# consecutive intervals is a near-unique barcode, so the two trains can be
# anchored to each other directly, anywhere in the recording, without counting
# pulses. Measured on CnL46 260727 pre: a 6-interval window matches its true
# location to <1 ms while the best decoy sits 40-120 ms away.
#
# After anchoring, SG->PROC is a straight line (the two clocks differ by a
# fixed rate offset - -15 ppm, i.e. 71 ms over that 77-minute session), so the
# matcher fits slope+intercept and assigns pulses by nearest neighbour. Both
# trains may drop pulses; nothing here assumes equal counts or contiguity.


def robust_iti_cv(rising_sec: np.ndarray) -> float:
    """Robust spread (MAD/median) of the inter-pulse intervals.

    MAD rather than SD so one long gap - a paused acquisition, an unplugged
    camera - cannot masquerade as jitter and flip the algorithm choice.
    """
    if rising_sec.size < 3:
        return float("nan")
    iti = np.diff(rising_sec)
    iti = iti[np.isfinite(iti) & (iti > 0)]
    if iti.size < 2:
        return float("nan")
    med = float(np.median(iti))
    if med <= 0:
        return float("nan")
    return float(1.4826 * float(np.median(np.abs(iti - med))) / med)


def resolve_match_algorithm(requested: str, proc_rising: np.ndarray) -> tuple[str, dict]:
    """Resolve 'auto' to 'pulse' or 'pulse_geo' from the PROC interval spread."""
    cv = robust_iti_cv(proc_rising)
    if requested != "auto":
        chosen = requested
    elif not np.isfinite(cv):
        # Too few edges to tell; the fixed-period matcher is the safer guess
        # because it degrades to "one clean segment" rather than misfiring.
        chosen = "pulse"
    else:
        chosen = "pulse_geo" if cv > GEO_ITI_CV_THRESHOLD else "pulse"
    return chosen, {
        "requested": requested,
        "resolved": chosen,
        "proc_iti_robust_cv": cv,
        "geo_cv_threshold": GEO_ITI_CV_THRESHOLD,
    }


def fingerprint_anchor(
    proc_iti: np.ndarray,
    sg_iti: np.ndarray,
    sg_start: int,
    window: int,
):
    """Locate one run of SG intervals inside the PROC interval sequence.

    Scores by worst-case (not mean) interval error, so a single badly placed
    edge disqualifies a candidate instead of being averaged away. Returns
    (proc_index, error_sec, runner_up_error_sec); the runner-up is what tells
    the caller whether the match is actually unique or just the least bad.
    """
    if window < 2 or proc_iti.size < window or sg_start + window > sg_iti.size:
        return None
    query = sg_iti[sg_start:sg_start + window]
    windows = np.lib.stride_tricks.sliding_window_view(proc_iti, window)
    err = np.max(np.abs(windows - query), axis=1)
    best = int(np.argmin(err))
    # Offsets within one window of the winner share most of their intervals
    # with it, so they are not independent alternatives - exclude them before
    # asking "how much better is the winner than anything else?".
    far = np.abs(np.arange(err.size) - best) > window
    runner_up = float(np.min(err[far])) if np.any(far) else float("inf")
    return best, float(err[best]), runner_up


def geo_anchors(
    proc_rising: np.ndarray,
    sg_sec: np.ndarray,
    window: int = 8,
    n_probes: int = 12,
    max_anchor_error_sec: float = 0.005,
    min_margin_ratio: float = 4.0,
    min_margin_sec: float = 0.01,
):
    """Tie the two trains together at several points spread across the session.

    Several probes rather than one: anchors near BOTH ends are what let the
    initial fit see the clock drift, and a probe that lands in a dropout-heavy
    stretch can then be outvoted instead of steering the whole match.
    """
    proc_iti = np.diff(proc_rising)
    sg_iti = np.diff(sg_sec)
    accepted: list[tuple[int, int]] = []
    rows: list[dict] = []
    if proc_iti.size < window or sg_iti.size < window:
        return accepted, rows

    probes = np.unique(np.linspace(0, sg_iti.size - window, n_probes).round().astype(int))
    for sg_start in probes:
        found = fingerprint_anchor(proc_iti, sg_iti, int(sg_start), window)
        if found is None:
            continue
        proc_start, err, runner_up = found
        unique = runner_up >= max(min_margin_sec, min_margin_ratio * err)
        ok = bool(err <= max_anchor_error_sec and unique)
        rows.append(
            {
                "sg_edge": int(sg_start),
                "proc_edge": int(proc_start),
                "max_iti_error_sec": err,
                "runner_up_error_sec": runner_up,
                "unique": bool(unique),
                "accepted": ok,
            }
        )
        if ok:
            accepted.append((int(sg_start), int(proc_start)))
    return accepted, rows


def _fit_time_map(sg_t: np.ndarray, proc_t: np.ndarray):
    """Least-squares SG->PROC line, returned as (slope, sg_ref, proc_ref).

    Kept in reference-point form and fitted on centred values because PROC
    times are unix epoch seconds (~1.79e9): a bare intercept would spend most
    of its float64 significand on the epoch rather than on the millisecond
    offsets this matcher cares about.
    """
    sg_ref = float(sg_t[0])
    proc_ref = float(proc_t[0])
    slope, offset = np.polyfit(sg_t - sg_ref, proc_t - proc_ref, 1)
    return float(slope), sg_ref, proc_ref + float(offset)


def _apply_time_map(time_map, sg_t: np.ndarray) -> np.ndarray:
    slope, sg_ref, proc_ref = time_map
    return proc_ref + slope * (sg_t - sg_ref)


def _initial_time_map(anchors, sg_sec, proc_rising, max_drift_ppm: float = 2000.0):
    """Seed the SG->PROC line from the anchors."""
    sg_t = np.asarray([sg_sec[a] for a, _ in anchors], dtype=float)
    proc_t = np.asarray([proc_rising[p] for _, p in anchors], dtype=float)
    if sg_t.size >= 2 and (sg_t[-1] - sg_t[0]) > 0:
        time_map = _fit_time_map(sg_t, proc_t)
        if abs(time_map[0] - 1.0) * 1e6 <= max_drift_ppm:
            return time_map
    # One anchor, no time spread, or a slope no real pair of clocks would
    # produce: fall back to a pure offset at unity rate and let the iterative
    # refit below recover the drift from the matches themselves.
    return 1.0, float(sg_t[0]), float(np.median(proc_t - sg_t) + sg_t[0])


def _nearest_pulse_matches(proc_rising: np.ndarray, mapped_sg: np.ndarray, tolerance: float):
    """One-to-one nearest-neighbour match of mapped SG pulses onto PROC pulses."""
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    if proc_rising.size < 2 or mapped_sg.size == 0:
        return empty
    idx = np.clip(np.searchsorted(proc_rising, mapped_sg), 1, proc_rising.size - 1)
    left_closer = np.abs(mapped_sg - proc_rising[idx - 1]) <= np.abs(proc_rising[idx] - mapped_sg)
    pick = np.where(left_closer, idx - 1, idx)
    resid = np.abs(proc_rising[pick] - mapped_sg)
    keep = np.flatnonzero(resid <= tolerance)
    if keep.size == 0:
        return empty
    # A dropped video frame leaves two SG pulses pointing at one PROC edge.
    # Sort by (PROC edge, distance) and take the first of each group so the
    # closer claimant wins and the mapping stays strictly one-to-one.
    order = keep[np.lexsort((resid[keep], pick[keep]))]
    first_of_group = np.concatenate(([True], np.diff(pick[order]) != 0))
    winners = np.sort(order[first_of_group])
    return winners, pick[winners]


def _matched_runs(sg_idx: np.ndarray, proc_idx: np.ndarray, min_pairs: int = 2):
    """Split matched pairs into runs where NEITHER train skipped a pulse."""
    if sg_idx.size == 0:
        return []
    breaks = np.flatnonzero((np.diff(sg_idx) != 1) | (np.diff(proc_idx) != 1)) + 1
    return [run for run in np.split(np.arange(sg_idx.size), breaks) if run.size >= min_pairs]


def match_geo_pulses(
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    fs: float,
    initial_tolerance_sec: float = 0.05,
    min_tolerance_sec: float = 0.004,
    max_iterations: int = 6,
    max_segment_median_error_sec: float = 0.01,
    fingerprint_edges: int = 8,
):
    """Match a random-interval PROC train to its DIO counterpart.

    Anchor by interval barcode, fit SG->PROC as a line, then alternate
    nearest-neighbour assignment and refitting so the fit tightens onto the
    pulses it just matched. Returns the same
    (proc_matched, sg_matched, diagnostics) contract as match_clean_segments,
    with sg_matched still in SAMPLES so the callers downstream are unchanged.
    """
    sg_sec = sg_rising / fs
    empty = np.array([], dtype=float)
    anchors, anchor_rows = geo_anchors(proc_rising, sg_sec, window=fingerprint_edges)
    diagnostics = {
        "algorithm": "pulse_geo",
        "fingerprint_edges": int(fingerprint_edges),
        "anchors": anchor_rows,
        "accepted_anchor_count": int(len(anchors)),
        "clean_segment_count": 0,
        "accepted_segment_count": 0,
        "segments": [],
        "matched_edge_count": 0,
        "unmatched_proc_edge_count": int(proc_rising.size),
        "unmatched_sg_edge_count": int(sg_rising.size),
    }
    if not anchors:
        diagnostics["failure"] = (
            "No unique interval fingerprint anchor found. The two trains may be "
            "from different sessions, or the intervals may not be random enough "
            "for pulse_geo (try --match-algorithm pulse)."
        )
        return empty, empty, diagnostics

    time_map = _initial_time_map(anchors, sg_sec, proc_rising)
    # A pulse must never be able to reach its neighbour, or the match can slip
    # by one for a whole stretch. The default suits the ~0.22 s shortest
    # interval seen so far; clamp it for any train that runs tighter.
    shortest_iti = float(np.min(np.diff(proc_rising))) if proc_rising.size > 1 else np.inf
    max_tolerance = float(min(initial_tolerance_sec, 0.4 * shortest_iti))
    tolerance = max_tolerance
    diagnostics["initial_tolerance_sec"] = max_tolerance
    diagnostics["shortest_proc_iti_sec"] = shortest_iti
    sg_idx = np.array([], dtype=int)
    proc_idx = np.array([], dtype=int)
    resid = np.array([], dtype=float)
    passes = []

    for _ in range(max_iterations):
        mapped = _apply_time_map(time_map, sg_sec)
        new_sg, new_proc = _nearest_pulse_matches(proc_rising, mapped, tolerance)
        if new_sg.size < 2:
            break
        new_resid = proc_rising[new_proc] - mapped[new_sg]
        passes.append(
            {
                "tolerance_sec": tolerance,
                "matched": int(new_sg.size),
                "median_abs_residual_sec": float(np.median(np.abs(new_resid))),
                "max_abs_residual_sec": float(np.max(np.abs(new_resid))),
                "slope": float(time_map[0]),
            }
        )
        settled = (
            new_sg.size == sg_idx.size
            and np.array_equal(new_sg, sg_idx)
            and np.array_equal(new_proc, proc_idx)
        )
        sg_idx, proc_idx, resid = new_sg, new_proc, new_resid
        if settled:
            break
        time_map = _fit_time_map(sg_sec[sg_idx], proc_rising[proc_idx])
        # Re-tighten around what the fit actually achieves, so later passes
        # reject pulses that only fitted under the loose seeding tolerance.
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        tolerance = float(np.clip(10.0 * 1.4826 * mad, min_tolerance_sec, max_tolerance))

    if sg_idx.size < 2:
        diagnostics["failure"] = (
            "Interval fingerprint anchored the trains, but fewer than two pulses "
            "survived nearest-neighbour matching."
        )
        return empty, empty, diagnostics

    # Report per-run interval agreement, mirroring the 'pulse' segment rows so
    # run_sync's accept/reject thresholds mean the same thing for both.
    segment_rows = []
    for run in _matched_runs(sg_idx, proc_idx):
        proc_run = proc_rising[proc_idx[run]]
        sg_run = sg_sec[sg_idx[run]]
        err = np.abs(np.diff(proc_run) - np.diff(sg_run))
        median_err = float(np.median(err))
        segment_rows.append(
            {
                "sg_start": int(sg_idx[run[0]]),
                "sg_end": int(sg_idx[run[-1]]) + 1,
                "proc_start": int(proc_idx[run[0]]),
                "proc_end": int(proc_idx[run[-1]]) + 1,
                "n_pairs": int(run.size),
                "accepted": bool(median_err <= max_segment_median_error_sec),
                "median_iti_error_sec": median_err,
                "max_iti_error_sec": float(np.max(err)),
            }
        )
    if not segment_rows:
        # Dropouts so dense that no two matched pulses are adjacent. Every pair
        # still passed the global fit, so report that instead of claiming the
        # match failed.
        abs_resid = np.abs(resid - np.median(resid))
        segment_rows.append(
            {
                "sg_start": int(sg_idx[0]),
                "sg_end": int(sg_idx[-1]) + 1,
                "proc_start": int(proc_idx[0]),
                "proc_end": int(proc_idx[-1]) + 1,
                "n_pairs": int(sg_idx.size),
                "accepted": bool(np.median(abs_resid) <= max_segment_median_error_sec),
                "median_iti_error_sec": float(np.median(abs_resid)),
                "max_iti_error_sec": float(np.max(abs_resid)),
                "note": "no adjacent matched pairs; errors are affine-fit residuals",
            }
        )

    centred = resid - np.median(resid)
    diagnostics.update(
        {
            "clean_segment_count": int(len(segment_rows)),
            "accepted_segment_count": int(sum(row["accepted"] for row in segment_rows)),
            "segments": segment_rows,
            "matched_edge_count": int(sg_idx.size),
            "unmatched_proc_edge_count": int(proc_rising.size - proc_idx.size),
            "unmatched_sg_edge_count": int(sg_rising.size - sg_idx.size),
            "iterations": passes,
            "final_tolerance_sec": tolerance,
            "time_map": {
                "slope": float(time_map[0]),
                "sg_ref_sec": float(time_map[1]),
                "proc_ref_sec": float(time_map[2]),
                "clock_drift_ppm": float((time_map[0] - 1.0) * 1e6),
            },
            "fit_residual_median_abs_sec": float(np.median(np.abs(centred))),
            "fit_residual_max_abs_sec": float(np.max(np.abs(centred))),
        }
    )
    return proc_rising[proc_idx], sg_rising[sg_idx], diagnostics


def match_pulse_trains(
    algorithm: str,
    proc_rising: np.ndarray,
    sg_rising: np.ndarray,
    fs: float,
    args,
):
    """Run whichever edge matcher `algorithm` names."""
    if algorithm == "pulse_geo":
        return match_geo_pulses(
            proc_rising,
            sg_rising,
            fs,
            initial_tolerance_sec=args.geo_match_tolerance_sec,
            fingerprint_edges=args.geo_fingerprint_edges,
            max_segment_median_error_sec=args.max_segment_median_error_sec,
        )
    if algorithm == "pulse":
        proc_matched, sg_matched, diagnostics = match_clean_segments(
            proc_rising,
            sg_rising,
            fs,
            max_segment_median_error_sec=args.max_segment_median_error_sec,
        )
        diagnostics["algorithm"] = "pulse"
        return proc_matched, sg_matched, diagnostics
    raise ValueError(f"Unknown match algorithm {algorithm!r}; expected one of {MATCH_ALGORITHMS}.")


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


def resolve_rec_folder(path: Path) -> Path:
    """Return the copy of a .rec folder that actually has exported DIO.

    A .rec folder often exists on BOTH servers while the Trodes DIO export
    only ran on one of them, so plain existence (resolve_existing_file) can
    hand back a copy with no *.DIO subfolder. Prefer a copy that has one.
    """
    candidates = [Path(path)]
    mirrored = mirror_on_backup_server(path)
    if mirrored is not None:
        candidates.append(mirrored)
    for candidate in candidates:
        try:
            if candidate.exists() and get_dio_folders(candidate):
                return candidate
        except OSError:
            continue
    # No DIO anywhere: fall back to whichever exists so load_rec_dio's own
    # "run Trodes export for DIO first" error names a real path.
    return resolve_existing_file(path)


def session_job(session_key, session_cfg, args, rec_folder):
    """Build one sync job (paths + label) from a registered sleep session.

    Pre-task and post-task sleep are separate videos recorded against separate
    .rec epochs, so each needs its OWN sync pickle. Paths come from
    sleep_pipeline_config.sleep_sessions (i.e. sleep_day_configs.json for
    ACTIVE_DATE) and outputs are suffixed "_pre" / "_post" - exactly what
    plot_sleep_spectrograms.py looks for.

    Returns None if that session has no video/rec registered, so a day that
    only has one of the two still processes the session it does have.
    """
    if session_cfg.get('proc_file') is None or session_cfg.get('rec_file_folder') is None:
        print(f"Skipping {session_key}: no proc_file / rec_file_folder registered "
              f"for this date (run set_sleep_day.py to add it).")
        return None

    suffix = session_cfg['suffix']
    rec_path = resolve_rec_folder(Path(session_cfg['rec_file_folder']))
    # Read the .rec from wherever its DIO export lives, but write beside the
    # NWBs in rec_folder - that is the ONLY place plot_sleep_spectrograms.py
    # looks (rec_folder / f"sync_times{suffix}.pkl"). The two are usually the
    # same directory, but a day whose .rec sits on a server while rec_folder
    # is a local disk (CnL46 260727: G:\) would otherwise write a perfectly
    # good pickle where nothing downstream can find it.
    out_dir = resolve_output_folder(Path(rec_folder))
    return {
        "label": session_key,
        "proc_path": resolve_existing_file(Path(session_cfg['proc_file'])),
        "rec_path": rec_path,
        "output_path": args.output or (out_dir / f"sync_times{suffix}.pkl"),
        "sample_figure_path": args.sample_figure or (
            out_dir / f"sync_square_wave_match_samples{suffix}.png"),
    }


def build_jobs(args):
    """Decide which sync run(s) this invocation performs.

    - --session pre|post : just that session
    - no --session       : BOTH active sessions for ACTIVE_DATE, writing
                           sync_times_pre.pkl and sync_times_post.pkl in one run

    All paths come from sleep_day_configs.json - there is no explicit-path or
    unsuffixed-output mode.
    """
    from sleep_pipeline_config import sleep_sessions, active_sleep_sessions, rec_folder

    if args.session is not None:
        sessions = {args.session: sleep_sessions[args.session]}
    else:
        # Same filter the rest of the pipeline uses: a session with both
        # sample bounds None means "not recorded that day" and is skipped.
        sessions = active_sleep_sessions(sleep_sessions)

    if len(sessions) > 1 and (args.output is not None or args.sample_figure is not None):
        raise SystemExit(
            "--output / --sample-figure apply to a single run, but this would "
            "process " + ", ".join(sessions) + ". Pass --session to pick one."
        )

    jobs = [session_job(key, cfg, args, rec_folder) for key, cfg in sessions.items()]
    return [job for job in jobs if job is not None]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare video PROC sync square wave with rec-folder DIO and save sync_times.pkl."
    )
    parser.add_argument(
        "--session",
        choices=("pre", "post"),
        default=None,
        help="Sync only this sleep session. Default: every active session for "
             "ACTIVE_DATE (writes sync_times_pre.pkl and sync_times_post.pkl).",
    )
    parser.add_argument(
        "--match-algorithm",
        choices=MATCH_ALGORITHMS,
        default=DEFAULT_MATCH_ALGORITHM,
        help="How PROC edges are matched to DIO edges. 'pulse' assumes a "
             "fixed-frequency square wave; 'pulse_geo' assumes random "
             "inter-pulse intervals and matches on the interval barcode. "
             "Default 'auto' picks per session from the PROC interval spread.",
    )
    parser.add_argument(
        "--geo-match-tolerance-sec",
        type=float,
        default=0.05,
        help="pulse_geo only: how far a mapped DIO pulse may sit from a PROC "
             "pulse on the first pass. Must stay below half the shortest "
             "inter-pulse interval or a pulse can claim its neighbour. Later "
             "passes tighten this automatically.",
    )
    parser.add_argument(
        "--geo-fingerprint-edges",
        type=int,
        default=8,
        help="pulse_geo only: how many consecutive intervals form the barcode "
             "used to anchor the trains. Raise it if anchors come back "
             "non-unique, lower it if pulses are dropped very densely.",
    )
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


def run_sync(args, proc_path, rec_path, output_path, sample_figure_path):
    """Run the PROC<->DIO comparison for one video/rec pair. Returns True if saved."""
    print(f"Loading PROC signal: {proc_path}")
    proc_time, proc_signal = load_proc_signal(proc_path)
    proc_rising, proc_falling, proc_binary, proc_threshold = sampled_edges(
        proc_time, proc_signal, args.proc_threshold
    )

    algorithm, algorithm_diag = resolve_match_algorithm(args.match_algorithm, proc_rising)
    print(
        f"Match algorithm: {algorithm}"
        + (f" (auto, PROC interval MAD/median = {algorithm_diag['proc_iti_robust_cv']:.3f})"
           if args.match_algorithm == "auto" else "")
    )

    print(f"Loading rec DIO: {rec_path} Din{args.dio_channel}")
    sg_time, sg_state = load_rec_dio(rec_path, args.dio_channel)
    raw_sg_time = sg_time
    raw_sg_state = sg_state
    if args.disable_dio_cleaning:
        dio_cleaning = {"removed_transition_count": 0, "disabled": True}
    else:
        min_state_duration = args.min_dio_state_duration_sec
        if min_state_duration is None:
            # Algorithm-aware, because a pulse_geo train's real states are far
            # shorter than half its median duration - see the helper's docstring.
            min_state_duration = default_min_dio_state_duration(sg_time, args.fs, algorithm)
        sg_time, sg_state, dio_cleaning = clean_short_dio_states(
            sg_time,
            sg_state,
            args.fs,
            min_state_duration,
        )

    sg_rising, sg_falling, sg_state = dio_edges(sg_time, sg_state)
    raw_sg_rising, raw_sg_falling, _ = dio_edges(raw_sg_time, raw_sg_state)

    proc_rising_matched, sg_rising_matched, segment_diagnostics = match_pulse_trains(
        algorithm,
        proc_rising,
        sg_rising,
        args.fs,
        args,
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
            "match_algorithm": algorithm_diag,
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
    if algorithm == "pulse_geo":
        print(f"  Fingerprint anchors accepted: "
              f"{segment_diagnostics['accepted_anchor_count']}/"
              f"{len(segment_diagnostics['anchors'])}")
        if "time_map" in segment_diagnostics:
            print(f"  Clock drift: {segment_diagnostics['time_map']['clock_drift_ppm']:+.2f} ppm")
            print(f"  Fit residual: median "
                  f"{segment_diagnostics['fit_residual_median_abs_sec'] * 1000:.3f} ms, max "
                  f"{segment_diagnostics['fit_residual_max_abs_sec'] * 1000:.3f} ms")
        if "failure" in segment_diagnostics:
            print(f"  pulse_geo failed: {segment_diagnostics['failure']}")
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

    if proc_rising_matched.size >= 2 and sg_rising_matched.size >= 2:
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
    else:
        # Nothing matched, so there is no mapping to draw the overlay with.
        print("Skipped sample match figure: fewer than two matched edges.")
    return bool(good_match or args.force_save_sync)


def main():
    args = parse_args()
    jobs = build_jobs(args)
    if not jobs:
        print("No sleep sessions to sync - nothing to do.")
        return

    saved = []
    for job in jobs:
        print("\n" + "#" * 70)
        print(f"SLEEP SESSION: {job['label']}")
        print("#" * 70)
        if run_sync(args, job["proc_path"], job["rec_path"],
                    job["output_path"], job["sample_figure_path"]):
            saved.append(job)

    if len(jobs) > 1:
        print("\n" + "=" * 70)
        print(f"SYNC COMPLETE - {len(saved)}/{len(jobs)} session(s) saved")
        for job in jobs:
            status = "saved  " if job in saved else "NOT SAVED"
            print(f"  {job['label']}: {status} {job['output_path']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
