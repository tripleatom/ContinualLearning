"""Shared kinematic feature helpers for VStimOnDecoding analyses."""

import pickle

import numpy as np

from plot_trial_traces import clean_position_p99, speed_to_cm_per_s


KINEMATIC_FEATURE_COLUMNS = [
    "speed_cms",
    "sin_heading",
    "cos_heading",
    "sin_head_angle",
    "cos_head_angle",
]
N_KINEMATIC_FEATURES = len(KINEMATIC_FEATURE_COLUMNS)

# Convenience subsets used by the task variants.
SPEED_ONLY_COLUMNS   = ["speed_cms"]
HEADING_ONLY_COLUMNS = ["sin_heading", "cos_heading", "sin_head_angle", "cos_head_angle"]


def load_task_kinematic_samples(task_pkl):
    """Load and clean task-session kinematics in the spike-window time frame.

    `position_time_sec` is already stored in window frame by
    extract_session_spikes.py (it computes `pos_t_window = step_time + delta`
    before saving), so no further offset arithmetic is needed here. The
    stored `session_to_window_offset_sec` is returned for reference only.

    Returns
    -------
    samples : dict
        Clean DLC sample arrays with keys t, speed_cms, sin_heading,
        cos_heading, sin_head_angle, cos_head_angle. All `t` values are
        in window frame, ready to bin against prepare_task_stim_type's
        bin_centers.
    clean_stats : dict
        Statistics from clean_position_p99.
    offset : float
        The session_to_window_offset_sec recorded in the pkl. Not applied
        to any returned array — kept for downstream audit / sanity prints.
    """
    with open(task_pkl, "rb") as f:
        data = pickle.load(f)

    sp = data.get("session_position")
    if sp is None:
        raise RuntimeError(
            "task_pkl has no session_position. Re-run extract_session_spikes.py "
            "so the full session DLC track is saved."
        )

    offset = float(np.asarray(sp["session_to_window_offset_sec"]))
    cleaned, clean_stats = clean_position_p99({
        "x": sp["position_x"],
        "y": sp["position_y"],
        "t": sp["position_time_sec"],
    })

    t_clean = np.asarray(cleaned["t"], dtype=float)
    if t_clean.size == 0:
        raise RuntimeError("No usable DLC samples after cleaning; cannot proceed.")

    t_orig = np.asarray(sp["position_time_sec"], dtype=float)
    heading_deg = np.asarray(sp["heading"], dtype=float)
    head_angle = np.asarray(sp["head_angle"], dtype=float)

    samples = {
        "t": t_clean,
        "speed_cms": np.asarray(speed_to_cm_per_s(cleaned["v"]), dtype=float),
        "sin_heading": np.interp(t_clean, t_orig, np.sin(np.deg2rad(heading_deg))),
        "cos_heading": np.interp(t_clean, t_orig, np.cos(np.deg2rad(heading_deg))),
        "sin_head_angle": np.interp(t_clean, t_orig, np.sin(head_angle)),
        "cos_head_angle": np.interp(t_clean, t_orig, np.cos(head_angle)),
    }
    return samples, clean_stats, offset


def print_kinematic_sample_report(samples, clean_stats, offset):
    """Print the standard DLC-cleaning summary used by analysis scripts."""
    print(f"session_to_window_offset_sec = {offset:.4f}  "
          f"(already applied by extract_session_spikes.py; reported for reference only)")
    print(
        f"Cleaned position: {clean_stats['n_kept']} samples kept "
        f"(zero-pad dropped={clean_stats['n_zero_pad_dropped']}, "
        f"flicker dropped={clean_stats['n_flicker_dropped']}, "
        f"speed-clip dropped={clean_stats['n_speed_dropped']} "
        f"@ {clean_stats['speed_threshold_cm_s']:.2f} cm/s)."
    )
    print(
        f"DLC time range (window frame): "
        f"{samples['t'].min():.2f} .. {samples['t'].max():.2f} s"
    )


def bin_mean(bin_centers, bin_size_sec, sample_times, values):
    """Mean of values per bin, with NaN where no sample fell inside."""
    n_bins = bin_centers.size
    edges_lo = bin_centers - 0.5 * bin_size_sec
    bin_idx = np.floor((sample_times - edges_lo[0]) / bin_size_sec).astype(int)
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)

    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    np.add.at(sums, bin_idx[in_range], values[in_range])
    np.add.at(counts, bin_idx[in_range], 1)

    out = np.full(n_bins, np.nan)
    has_samples = counts > 0
    out[has_samples] = sums[has_samples] / counts[has_samples]
    return out


def build_task_kinematics(bin_centers, bin_size_sec, samples, columns=None):
    """Return a (n_bins, len(columns)) feature matrix and mask of bins with valid tracking.

    Parameters
    ----------
    bin_centers, bin_size_sec : as returned by prepare_task_stim_type (window time frame).
    samples : dict from load_task_kinematic_samples — its 't' is ALREADY in window time
        (the offset has been subtracted), so per-bin assignment is correct.
    columns : list of column names to compute (subset of KINEMATIC_FEATURE_COLUMNS).
        Defaults to all five.
    """
    if columns is None:
        columns = KINEMATIC_FEATURE_COLUMNS
    binned = [
        bin_mean(bin_centers, bin_size_sec, samples["t"], samples[col])
        for col in columns
    ]
    stacked = np.column_stack(binned)
    keep = ~np.any(np.isnan(stacked), axis=1)
    return stacked, keep
