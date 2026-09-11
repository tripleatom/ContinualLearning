"""Velocity from the front-camera tracking of a sleep session.

Two sources, selected by `VELOCITY_SOURCE` in sleep_pipeline_config.py:

  'proc_center'  the single centre point stored in the *_PROC pickle
                 (center_x / center_y). That centre is a likelihood-weighted
                 mean of the six HEAD keypoints, and on frames where tracking
                 is poor the acquisition program repeats the previous frame's
                 value verbatim - so dropouts read as exactly zero speed.
  'dlc_body'     a centroid computed here from chosen keypoints in the
                 companion *_DLC.hdf5 (all 12 bodyparts, with likelihoods).
                 Defaults to the five BODY points, which track far better than
                 the head during sleep, and low-confidence frames become gaps
                 that are interpolated rather than silently frozen.

Both paths end in the same filtering (lowpass_velocity) and write the same pkl
keys (time_stamp / velocity / velocity_x / velocity_y) on the PROC time base,
so video<->ephys sync downstream is unchanged.

How the jitter is handled
-------------------------
Differentiation amplifies tracking jitter, and taking the speed magnitude then
RECTIFIES it: zero-mean position noise becomes a strictly positive speed
offset, so a motionless animal reads as a nonzero speed whose value depends on
lighting and tracking quality rather than on behaviour. Averaging afterwards
cannot undo that - it shrinks the variance of the bias, never the bias. So all
noise suppression happens in position space, before the derivative and before
the magnitude:

    clean position -> low-pass position -> differentiate once -> magnitude
    -> one windowed aggregation matched to the spectrogram (aggregate_speed)

and nothing smooths the speed again after that.
"""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, savgol_filter

from server_fallback import (mirror_on_backup_server, resolve_existing_file,
                             resolve_output_folder)

# Keypoints in the DLC companion file, grouped as the model names them.
HEAD_KEYPOINTS = ('nose', 'left_eye', 'right_eye',
                  'left_bar', 'right_bar', 'cable_base')
BODY_KEYPOINTS = ('left_midside', 'right_midside',
                  'left_hip', 'right_hip', 'tail_base')

# Per-frame likelihood a keypoint must reach to enter the centroid.
DEFAULT_LIKELIHOOD_THRESHOLD = 0.6

# Filtering defaults. Mirrored in sleep_pipeline_config.py (VELOCITY_CUTOFF_HZ,
# VELOCITY_FILTER_ORDER, VELOCITY_MAX_GAP_SEC) - repeated here so this module
# stays importable on its own, e.g. from plot_proc_velocity_distribution.py.
DEFAULT_CUTOFF_HZ = 0.5
DEFAULT_FILTER_ORDER = 2
DEFAULT_MAX_GAP_SEC = 0.5

# source -> output filename stem. 'velocity_advanced' is the historical name
# and must not change: existing pkls (and plot_sleep_spectrograms) use it.
VELOCITY_SOURCES = {
    'proc_center': 'velocity_advanced',
    'dlc_body': 'velocity_body',
}


def resolve_proc_file(proc_file):
    """The copy of a *_PROC file that actually exists, on either server.

    sleep_day_configs.json records whichever server a day was recorded on, but
    video folders get moved to the new server afterwards, so a registered path
    can be stale. Every entry point below resolves first (server_fallback
    checks the new server, then the recorded path) - the same thing
    video_ephys_sync.py does with its copy of this path.
    """
    return resolve_existing_file(Path(proc_file))


def compute_velocity(proc_file, velocity_threshold=530, method='savgol',
                     window_length=11, polyorder=3, median_window=5):
    """
    Compute velocity from position tracking data using smoothing methods.

    Parameters
    ----------
    proc_file : str
        Path to the _PROC pickle file containing tracking data
    velocity_threshold : float, optional
        Maximum velocity threshold. Values above this are set to 0 (default: 530)
    method : str, optional
        Smoothing method: 'savgol' (Savitzky-Golay), 'median', 'gaussian', or 'simple'
        (default: 'savgol')
    window_length : int, optional
        Length of the filter window (must be odd, for savgol/median). Default: 11
    polyorder : int, optional
        Order of polynomial for Savitzky-Golay filter. Default: 3
    median_window : int, optional
        Window size for median filtering of outliers. Default: 5

    Returns
    -------
    t : numpy.ndarray
        Time stamps aligned with velocity
    v : numpy.ndarray
        Velocity values (smoothed)
    v_raw : numpy.ndarray
        Raw velocity values (unsmoothed, for comparison)

    Examples
    --------
    >>> proc_file = r"\\server\path\to\Animal_date_session_PROC"
    >>> t, v, v_raw = compute_velocity(proc_file, method='savgol')
    """
    # Load data
    data = pickle.load(open(resolve_proc_file(proc_file), 'rb'))

    x = data['center_x']
    y = data['center_y']
    time_stamp = data['time_stamp']

    # Method 1: Smooth positions first, then compute velocity
    if method == 'savgol':
        # Savitzky-Golay filter - fits local polynomial to data
        # This preserves features while smoothing noise
        x_smooth = savgol_filter(
            x, window_length=window_length, polyorder=polyorder)
        y_smooth = savgol_filter(
            y, window_length=window_length, polyorder=polyorder)

        # Compute velocity using differences on smoothed positions
        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    elif method == 'median':
        # Median filter - robust to outliers
        x_smooth = median_filter(x, size=median_window)
        y_smooth = median_filter(y, size=median_window)

        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    elif method == 'gaussian':
        # Gaussian smoothing using convolution
        from scipy.ndimage import gaussian_filter1d
        sigma = window_length / 6  # rule of thumb
        x_smooth = gaussian_filter1d(x, sigma=sigma)
        y_smooth = gaussian_filter1d(y, sigma=sigma)

        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    else:  # 'simple' - original method
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    # Calculate distance
    d = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    epsilon = 1e-8
    dt = np.where(dt == 0, epsilon, dt)

    # Calculate raw velocity (before smoothing)
    dx_raw = np.diff(x)
    dy_raw = np.diff(y)
    d_raw = np.sqrt(dx_raw**2 + dy_raw**2)
    dt_raw = np.diff(time_stamp)
    dt_raw = np.where(dt_raw == 0, epsilon, dt_raw)
    v_raw = d_raw / dt_raw

    # Calculate velocity
    v = d / dt

    # Remove outliers: set velocities above threshold to 0
    v_raw_clean = np.where(v_raw > velocity_threshold, 0, v_raw)
    v = np.where(v > velocity_threshold, 0, v)

    # Deliberately NOT smoothed again here. Smoothing the speed after the
    # magnitude has been taken cannot remove the rectified jitter offset, only
    # its variance - the smoothing that matters already happened on x and y
    # above. See lowpass_velocity for the method these panels are compared to.
    return t, v, v_raw_clean


def compute_velocity_advanced(proc_file, velocity_threshold=530,
                              cutoff_hz=DEFAULT_CUTOFF_HZ,
                              order=DEFAULT_FILTER_ORDER,
                              max_gap_sec=DEFAULT_MAX_GAP_SEC):
    """Velocity of the PROC head centre, jitter filtered out of position.

    On frames where tracking fails the acquisition program repeats the previous
    frame's centre verbatim, so a dropout is a run of byte-identical positions
    rather than a gap. Those repeats are not observations - they are marked
    unobserved here, which lets `lowpass_velocity` interpolate short runs and
    refuse to invent a speed across long ones. (This is the failure mode the
    'dlc_body' source avoids entirely; see the module docstring.)

    Returns (time_stamp, v, vx, vy, info) - see `lowpass_velocity` for `info`.
    """
    data = pickle.load(open(resolve_proc_file(proc_file), 'rb'))

    x = np.asarray(data['center_x'], dtype=float)
    y = np.asarray(data['center_y'], dtype=float)
    time_stamp = np.asarray(data['time_stamp'], dtype=float)

    repeated = np.zeros(x.size, dtype=bool)
    repeated[1:] = (x[1:] == x[:-1]) & (y[1:] == y[:-1])

    v, vx, vy, info = lowpass_velocity(
        x, y, time_stamp, observed=~repeated, cutoff_hz=cutoff_hz, order=order,
        max_gap_sec=max_gap_sec, velocity_threshold=velocity_threshold)
    info['source'] = 'proc_center'
    info['n_frozen_frames'] = int(repeated.sum())
    info['frozen_fraction'] = float(np.mean(repeated))

    return time_stamp, v, vx, vy, info


# =====================================================
# FILTERING
# =====================================================

def uniform_grid(time_stamp, target_fs=None):
    """A uniform time grid spanning `time_stamp`, and the rate it samples at.

    The front camera does not run at a fixed frame rate, and every practical
    filter (Butterworth, Savitzky-Golay, any FIR) is defined on evenly spaced
    samples. The old code papered over this by handing savgol_filter a single
    `delta=dt_mean`, which silently assumes the very uniformity that is missing
    and makes the effective cutoff drift with the frame rate. Resampling once,
    explicitly, is both honest and exact enough: frame-time jitter is
    milliseconds against a cutoff of ~2 seconds.
    """
    time_stamp = np.asarray(time_stamp, dtype=float)
    dt = float(np.median(np.diff(time_stamp)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Frame timestamps are not increasing; cannot resample.")
    if target_fs is None:
        target_fs = 1.0 / dt
    span = time_stamp[-1] - time_stamp[0]
    grid = time_stamp[0] + np.arange(int(np.floor(span * target_fs)) + 1) / target_fs
    return grid, float(target_fs), dt


def reject_position_jumps(x, y, velocity_threshold=530, median_window=5,
                          jump_px=None, dt=None):
    """Mark samples that teleport away from a short median-filtered track.

    Outlier rejection belongs in POSITION, not in speed. A single bad detection
    makes two large steps - out and back - so a speed threshold flags both and
    then interpolates the speed across them, smearing one bad frame over its
    neighbours. Dropping the position sample instead leaves the trajectory
    continuous for the low-pass to see.

    `jump_px` defaults to the per-frame displacement that `velocity_threshold`
    allows (threshold x median frame interval), so the two stay consistent.

    Returns (mask of samples to reject, the jump_px actually used).
    """
    if jump_px is None:
        if dt is None:
            raise ValueError("Pass either jump_px or dt to derive it from.")
        jump_px = float(velocity_threshold) * float(dt)

    window = int(median_window) | 1  # median_filter wants an odd window
    residual = np.hypot(x - median_filter(x, size=window),
                        y - median_filter(y, size=window))
    return residual > jump_px, float(jump_px)


def covered_frames(time_stamp, observed, max_gap_sec):
    """Frames whose position rests on a real detection near enough in time.

    A frame is covered if it was observed itself, or if it sits inside a gap
    between observations no longer than `max_gap_sec`. Everything else - long
    dropouts, and the run-in/run-out before the first and after the last
    observation - is left uncovered, because interpolating position across a
    long gap draws a straight line, and a straight line differentiates to a
    low, entirely plausible speed. Reported as "the animal was still" that is
    the one error an immobility gate must never make.
    """
    observed = np.asarray(observed, dtype=bool)
    covered = observed.copy()
    index = np.flatnonzero(observed)
    if index.size < 2:
        return covered

    observed_time = time_stamp[index]
    left = np.searchsorted(observed_time, time_stamp, side='right') - 1
    inside = (left >= 0) & (left < observed_time.size - 1)
    gap = np.full(time_stamp.size, np.inf)
    gap[inside] = (observed_time[np.clip(left + 1, 0, observed_time.size - 1)]
                   - observed_time[np.clip(left, 0, observed_time.size - 1)])[inside]
    return covered | (inside & (gap <= max_gap_sec))


def lowpass_velocity(x, y, time_stamp, observed=None, *,
                     cutoff_hz=DEFAULT_CUTOFF_HZ, order=DEFAULT_FILTER_ORDER,
                     max_gap_sec=DEFAULT_MAX_GAP_SEC, velocity_threshold=530,
                     jump_px=None, median_window=5, target_fs=None):
    """Speed and components from an (x, y, t) track, jitter removed in position.

    Shared by both velocity sources so the two differ only in which point is
    tracked, never in the math. The steps, in the order that matters:

      1. interpolate the unobserved samples so the track is continuous - both
         the jump detector and the low-pass need an unbroken trajectory
      2. reject position outliers (`reject_position_jumps`), mark them
         unobserved too, and interpolate those as well
      3. resample onto a uniform grid (`uniform_grid`), because the camera's
         frame rate is not fixed
      4. zero-phase Butterworth low-pass at `cutoff_hz`, forward+backward
      5. differentiate ONCE with central differences. No smoothing is needed
         here: after step 4 the position is band-limited, so the derivative is
         too. A Savitzky-Golay derivative would smooth a second time, at a
         window length nobody chose in seconds.
      6. speed = hypot(vx, vy), taken only now that the jitter is gone
      7. interpolate back onto the caller's `time_stamp`, and NaN the frames
         that no observation supports (`covered_frames`)

    `observed` marks which samples are real detections (default: the finite
    ones). Speed near a long dropout is still influenced by the interpolated
    stretch beside it; `aggregate_speed`'s coverage fraction is what guards
    against that at the timescale the speed is actually read on.

    Returns (v, vx, vy, info) on the given `time_stamp`, NaN where uncovered.
    """
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    time_stamp = np.asarray(time_stamp, dtype=float)

    if observed is None:
        observed = np.isfinite(x) & np.isfinite(y)
    else:
        observed = np.asarray(observed, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if not observed.any():
        raise ValueError("No observed positions to compute velocity from.")

    # A continuous track first, so the jump detector sees no NaN holes.
    x[~observed] = np.nan
    y[~observed] = np.nan
    x = fill_gaps(x, time_stamp)
    y = fill_gaps(y, time_stamp)

    grid, grid_fs, dt = uniform_grid(time_stamp, target_fs)
    jumps, jump_px = reject_position_jumps(
        x, y, velocity_threshold, median_window, jump_px, dt)
    if jumps.any():
        observed = observed & ~jumps
        if not observed.any():
            raise ValueError("Every position sample was rejected as a jump; "
                             f"jump_px={jump_px:.1f} is too strict.")
        x[jumps] = np.nan
        y[jumps] = np.nan
        x = fill_gaps(x, time_stamp)
        y = fill_gaps(y, time_stamp)

    x_grid = np.interp(grid, time_stamp, x)
    y_grid = np.interp(grid, time_stamp, y)

    nyquist = grid_fs / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz={cutoff_hz} is at or above the Nyquist "
                         f"frequency of the {grid_fs:.1f} fps track ({nyquist:.1f} Hz).")
    b, a = butter(order, cutoff_hz / nyquist, btype='low')
    padlen = 3 * max(len(a), len(b))
    if grid.size <= padlen:
        raise ValueError(f"Track is only {grid.size} samples - too short to "
                         f"filter at {cutoff_hz} Hz.")
    x_grid = filtfilt(b, a, x_grid)
    y_grid = filtfilt(b, a, y_grid)

    # One differentiation of an already band-limited track.
    vx_grid = np.gradient(x_grid, 1.0 / grid_fs)
    vy_grid = np.gradient(y_grid, 1.0 / grid_fs)

    vx = np.interp(time_stamp, grid, vx_grid)
    vy = np.interp(time_stamp, grid, vy_grid)
    v = np.hypot(vx, vy)

    # After a 0.5 Hz low-pass this should never fire; kept as a guard, and
    # counted so a session where it does fire is visible rather than silent.
    over = v > velocity_threshold
    covered = covered_frames(time_stamp, observed, max_gap_sec) & ~over
    for values in (v, vx, vy):
        values[~covered] = np.nan

    info = {
        'filter': 'butterworth+filtfilt on resampled position',
        'cutoff_hz': float(cutoff_hz),
        'filter_order': int(order),
        'grid_fs': grid_fs,
        'frame_interval_median': dt,
        'frame_interval_iqr': float(np.subtract(*np.percentile(np.diff(time_stamp),
                                                               [75, 25]))),
        'velocity_threshold': float(velocity_threshold),
        'jump_px': jump_px,
        'median_window': int(median_window),
        'max_gap_sec': float(max_gap_sec),
        'n_jump_outliers': int(jumps.sum()),
        'n_over_threshold': int(over.sum()),
        'covered_mask': covered,
        'n_uncovered': int((~covered).sum()),
        'uncovered_fraction': float(np.mean(~covered)),
    }
    return v, vx, vy, info


# =====================================================
# AGGREGATION ONTO THE ANALYSIS GRID
# =====================================================

def aggregate_speed(time_stamp, speed, target_times, window_sec,
                    min_coverage=0.5, with_median=False):
    """Mean speed in a centred `window_sec` window at each of `target_times`.

    This is the ONLY averaging applied to the speed after `lowpass_velocity`:
    one reduction, whose support is chosen to match the spectrogram window the
    speed will be read against (10 s), so the two signals a bout is scored from
    have the same bandwidth. The pipeline used to stack a per-epoch mean and a
    multi-epoch boxcar on top of the in-filter smoothing, giving a total
    support that was never written down anywhere and that changed with
    epoch_sec.

    Windows overlap when the step is shorter than the window - that is the
    point, and it is exactly what the spectrogram's own 10 s / 1 s grid does.

    A window is NaN unless it holds at least `min_coverage` of the frames its
    duration implies, so "we could not see the animal" never reads as "the
    animal was still". Returns a dict with mean / median / n_frames / coverage.
    """
    time_stamp = np.asarray(time_stamp, dtype=float)
    speed = np.asarray(speed, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    finite = np.isfinite(speed)
    valid_time = time_stamp[finite]
    valid_speed = speed[finite]

    half = float(window_sec) / 2.0
    lo = np.searchsorted(valid_time, target_times - half, side='left')
    hi = np.searchsorted(valid_time, target_times + half, side='right')

    cumulative = np.concatenate([[0.0], np.cumsum(valid_speed)])
    n_frames = (hi - lo).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = (cumulative[hi] - cumulative[lo]) / n_frames
    mean[n_frames == 0] = np.nan

    dt = float(np.median(np.diff(time_stamp))) if time_stamp.size > 1 else np.nan
    expected = float(window_sec) / dt if dt > 0 else np.nan
    coverage = n_frames / expected if expected > 0 else np.full(n_frames.shape, np.nan)
    mean[coverage < min_coverage] = np.nan

    median = None
    if with_median:
        # O(window) per target, so only worth it on a coarse grid (epochs),
        # not on the frame grid the plots use.
        median = np.full(target_times.size, np.nan)
        for i, (start, stop) in enumerate(zip(lo, hi)):
            if stop > start and coverage[i] >= min_coverage:
                median[i] = np.median(valid_speed[start:stop])

    return {
        'mean': mean,
        'median': median,
        'n_frames': n_frames,
        'coverage': coverage,
        'window_sec': float(window_sec),
        'min_coverage': float(min_coverage),
        'expected_frames_per_window': expected,
    }


# =====================================================
# DLC KEYPOINT SOURCE
#   The *_DLC.hdf5 beside each *_PROC holds every bodypart the model tracks
#   (x, y, likelihood per frame) plus frame_time / pose_time. It is a pandas
#   HDFStore frame, but pandas needs pytables to read it and the analysis envs
#   do not all have pytables - so the column MultiIndex is rebuilt from the
#   raw HDF5 nodes with h5py, which every env has.
# =====================================================

def dlc_file_for_proc(proc_file):
    """Companion *_DLC.hdf5 for a *_PROC file (same stem), on either server."""
    proc_file = Path(proc_file)
    if not proc_file.name.endswith('_PROC'):
        raise ValueError(f"Not a *_PROC file: {proc_file}")
    return resolve_existing_file(
        proc_file.with_name(f"{proc_file.name[:-len('_PROC')]}_DLC.hdf5"))


def _column_keys(group, prefix):
    """Rebuild pandas MultiIndex column keys from their HDF5 level/label nodes.

    Returns [(bodypart, field), ...] - the last two index levels, which is what
    DLC uses whether or not the frame carries a leading 'scorer' level.
    """
    levels = [[name.decode() if isinstance(name, bytes) else str(name)
               for name in group[key][:]]
              for key in sorted(k for k in group if k.startswith(f"{prefix}_level"))]
    labels = [group[key][:]
              for key in sorted(k for k in group if k.startswith(f"{prefix}_label"))]
    if not levels or not labels:
        raise KeyError(f"{prefix} is not a MultiIndex in this DLC file")

    keys = []
    for column in range(len(labels[0])):
        names = tuple(levels[i][labels[i][column]] for i in range(len(levels)))
        keys.append(names[-2:] if len(names) >= 2 else (names[0], ''))
    return keys


def load_dlc_table(dlc_file):
    """Load a *_DLC.hdf5 as {(bodypart, field): array} without needing pytables."""
    import h5py  # imported lazily: only the DLC path needs it

    table = {}
    with h5py.File(dlc_file, 'r') as handle:
        frames = [key for key in handle
                  if isinstance(handle[key], h5py.Group)
                  and 'axis0_level0' in handle[key]]
        if not frames:
            raise KeyError(f"No pandas frame found in {dlc_file}")
        group = handle[frames[0]]
        for block in sorted(k for k in group if re.fullmatch(r"block\d+_values", k)):
            prefix = block[:-len('_values')]
            values = group[block][:]
            for column, key in enumerate(_column_keys(group, f"{prefix}_items")):
                table[key] = values[:, column]
    return table


def keypoint_centroid(table, keypoints, likelihood_threshold=DEFAULT_LIKELIHOOD_THRESHOLD,
                      weighted=True):
    """Per-frame centroid of `keypoints`, ignoring low-likelihood detections.

    Frames where NO requested keypoint clears the threshold come back as NaN
    rather than a stale or noisy position - the caller decides how to fill
    them, which is the whole point of not using the PROC centre (that one
    silently repeats the previous frame).

    Returns (x, y, n_used) where n_used counts the keypoints that contributed.
    """
    missing = [part for part in keypoints
               if (part, 'x') not in table or (part, 'likelihood') not in table]
    if missing:
        available = sorted({part for part, field in table if field == 'x'})
        raise KeyError(f"DLC file has no keypoint(s) {missing}. Available: {available}")

    xs = np.asarray([table[(part, 'x')] for part in keypoints], dtype=float)
    ys = np.asarray([table[(part, 'y')] for part in keypoints], dtype=float)
    ps = np.asarray([table[(part, 'likelihood')] for part in keypoints], dtype=float)

    keep = ps >= likelihood_threshold
    # Weighting by likelihood matches how the acquisition program builds its
    # own head centre; equal weights are available for a plain mean.
    weights = np.where(keep, ps if weighted else 1.0, 0.0)
    total = weights.sum(axis=0)
    good = total > 0

    x = np.full(total.shape, np.nan)
    y = np.full(total.shape, np.nan)
    x[good] = (xs * weights).sum(axis=0)[good] / total[good]
    y[good] = (ys * weights).sum(axis=0)[good] / total[good]
    return x, y, keep.sum(axis=0)


def align_dlc_to_proc(table, proc_data):
    """Match DLC rows to PROC frames, returning (dlc_index, time_stamp, how).

    The DLC rows and the PROC frames come from the same camera loop and carry
    the same `frame_time`, but the PROC file can hold a few extra frames for
    which no pose was written. Matching on frame_time (not position) keeps the
    velocity on the PROC `time_stamp` base that video_ephys_sync aligns to.
    """
    proc_frame_time = np.asarray(proc_data['frame_time'], dtype=float)
    proc_time_stamp = np.asarray(proc_data['time_stamp'], dtype=float)

    if ('frame_time', '') in table:
        dlc_frame_time = np.asarray(table[('frame_time', '')], dtype=float)
        common, idx_dlc, idx_proc = np.intersect1d(
            dlc_frame_time, proc_frame_time, return_indices=True)
        if common.size >= 0.5 * dlc_frame_time.size:
            return idx_dlc, proc_time_stamp[idx_proc], "exact frame_time match"

        # Clocks disagree (re-encoded video, edited PROC): fall back to nearest
        # frame within half a frame interval.
        tolerance = 0.5 * float(np.median(np.diff(proc_frame_time)))
        order = np.argsort(proc_frame_time)
        sorted_time = proc_frame_time[order]
        right = np.clip(np.searchsorted(sorted_time, dlc_frame_time), 1,
                        sorted_time.size - 1)
        left = right - 1
        pick = np.where(np.abs(sorted_time[right] - dlc_frame_time)
                        < np.abs(sorted_time[left] - dlc_frame_time), right, left)
        idx_proc = order[pick]
        close = np.abs(proc_frame_time[idx_proc] - dlc_frame_time) <= tolerance
        if close.sum() < 0.5 * dlc_frame_time.size:
            raise ValueError(
                f"Could not align DLC to PROC: only {close.sum()} of "
                f"{dlc_frame_time.size} rows fall within {tolerance:.4f}s of a frame.")
        return (np.flatnonzero(close), proc_time_stamp[idx_proc[close]],
                "nearest frame_time match")

    # No frame_time column: only a 1:1 recording can be matched safely.
    n_dlc = len(next(iter(table.values())))
    if n_dlc != proc_frame_time.size:
        raise ValueError(
            f"DLC file has no frame_time and its {n_dlc} rows do not match the "
            f"{proc_frame_time.size} PROC frames - cannot align.")
    return np.arange(n_dlc), proc_time_stamp, "positional (no frame_time)"


def fill_gaps(values, time_stamp):
    """Linearly interpolate NaN samples (low-confidence frames) over time."""
    values = np.asarray(values, dtype=float).copy()
    good = np.isfinite(values)
    if not good.any():
        raise ValueError("No finite samples to interpolate from.")
    if not good.all():
        values[~good] = np.interp(time_stamp[~good], time_stamp[good], values[good])
    return values


def compute_velocity_from_keypoints(proc_file, keypoints=BODY_KEYPOINTS,
                                    likelihood_threshold=DEFAULT_LIKELIHOOD_THRESHOLD,
                                    velocity_threshold=530,
                                    cutoff_hz=DEFAULT_CUTOFF_HZ,
                                    order=DEFAULT_FILTER_ORDER,
                                    max_gap_sec=DEFAULT_MAX_GAP_SEC,
                                    weighted=True):
    """Velocity of a DLC keypoint centroid, on the PROC time base.

    Parameters mirror compute_velocity_advanced; `keypoints` chooses which
    bodyparts form the tracked point (default: the five BODY_KEYPOINTS, which
    stay visible while the animal is curled up asleep).

    Returns (time_stamp, v, vx, vy, info) where `info` records the alignment,
    how many frames were low-confidence, and the settings used.
    """
    proc_file = resolve_proc_file(proc_file)
    dlc_file = dlc_file_for_proc(proc_file)
    if not dlc_file.is_file():
        raise FileNotFoundError(
            f"No DLC file for {proc_file.name}: expected {dlc_file}")

    with open(proc_file, 'rb') as f:
        proc_data = pickle.load(f)
    table = load_dlc_table(dlc_file)

    x_all, y_all, n_used_all = keypoint_centroid(
        table, keypoints, likelihood_threshold, weighted)
    idx_dlc, time_stamp, how = align_dlc_to_proc(table, proc_data)

    x = x_all[idx_dlc]
    y = y_all[idx_dlc]
    n_used = n_used_all[idx_dlc]
    dropped = ~np.isfinite(x)

    # Low-confidence frames are handed over as "not observed" rather than
    # pre-filled: lowpass_velocity interpolates the short runs itself and
    # refuses to invent a speed across the long ones.
    v, vx, vy, filter_info = lowpass_velocity(
        x, y, time_stamp, observed=~dropped, cutoff_hz=cutoff_hz, order=order,
        max_gap_sec=max_gap_sec, velocity_threshold=velocity_threshold)

    info = {
        'source': 'dlc_body',
        'keypoints': list(keypoints),
        'likelihood_threshold': float(likelihood_threshold),
        'likelihood_weighted': bool(weighted),
        'alignment': how,
        'n_frames': int(time_stamp.size),
        'n_proc_frames': int(np.asarray(proc_data['frame_time']).size),
        'n_low_confidence_frames': int(dropped.sum()),
        'low_confidence_fraction': float(np.mean(dropped)),
        'mean_keypoints_used': float(np.mean(n_used)),
        'centroid_x': fill_gaps(x, time_stamp),
        'centroid_y': fill_gaps(y, time_stamp),
        'n_keypoints_used': n_used,
        'low_confidence_mask': dropped,
        'source_dlc_file': str(dlc_file),
        **filter_info,
    }
    return time_stamp, v, vx, vy, info


def velocity_output_name(proc_file, source='proc_center'):
    """Build an informative velocity filename from a *_PROC file path."""
    if source not in VELOCITY_SOURCES:
        raise ValueError(f"Unknown velocity source {source!r}; "
                         f"expected one of {sorted(VELOCITY_SOURCES)}")
    return f'{proc_session_name(proc_file)}_{VELOCITY_SOURCES[source]}.pkl'


def proc_session_name(proc_file):
    """Extract the session name from a front-camera *_PROC file path."""
    proc_stem = Path(proc_file).name
    if proc_stem.endswith('_PROC'):
        proc_stem = proc_stem[:-len('_PROC')]
    if proc_stem.startswith('front_camera_'):
        proc_stem = proc_stem[len('front_camera_'):]
    return proc_stem


def stamp_figure(fig, text):
    """Embed a reproducibility line (what made this figure, from what, when)."""
    fig.text(0.005, 0.001, f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by "
                           f"proc_func_velocity.py  |  {text}",
             fontsize=6, color='0.4', ha='left', va='bottom')


def generate_velocity_outputs(proc_file, source='proc_center', overwrite=False,
                              show_plots=False, keypoints=BODY_KEYPOINTS,
                              likelihood_threshold=DEFAULT_LIKELIHOOD_THRESHOLD,
                              cutoff_hz=DEFAULT_CUTOFF_HZ,
                              order=DEFAULT_FILTER_ORDER,
                              max_gap_sec=DEFAULT_MAX_GAP_SEC):
    """Generate the velocity pickle and diagnostic plots for one PROC file.

    `source` selects which point is tracked - 'proc_center' (the PROC file's
    own head centre) or 'dlc_body' (a centroid of `keypoints` from the DLC
    companion file). Each source writes its own filename, so both can coexist
    for a session and neither overwrites the other.
    """
    registered = Path(proc_file)
    proc_file = resolve_proc_file(registered)
    if not proc_file.is_file():
        mirrored = mirror_on_backup_server(registered)
        tried = f"{registered}" + (f"\n  {mirrored}" if mirrored else "")
        raise FileNotFoundError(
            f"PROC file not found on either server. Tried:\n  {tried}\n"
            f"(the registered path may be stale - re-point it with set_sleep_day.py)")
    if proc_file != registered:
        print(f"PROC file found on the other server: {proc_file}")

    session_name = proc_session_name(proc_file)
    # Read the _PROC file in place, but write velocity + figures to the new
    # server under the same subpath (the old server is full).
    data_path = resolve_output_folder(proc_file.parent)
    figures_path = resolve_output_folder(data_path / 'figures')
    output_name = velocity_output_name(proc_file, source)
    velocity_output_file = data_path / output_name
    existing_output = resolve_existing_file(proc_file.parent / output_name)
    if existing_output.is_file() and not overwrite:
        print(f"Velocity output already exists - skipping: {existing_output}")
        return existing_output

    if source == 'dlc_body':
        return _generate_keypoint_velocity(
            proc_file, session_name, velocity_output_file, figures_path,
            keypoints, likelihood_threshold, show_plots,
            cutoff_hz, order, max_gap_sec)

    # Compare different methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    methods = ['simple', 'savgol', 'median', 'gaussian']

    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]

        t, v, v_raw = compute_velocity(
            proc_file, method=method, window_length=11)

        # Make sure arrays have the same length
        print(f"{method}: t={len(t)}, v={len(v)}, v_raw={len(v_raw)}")

        ax.plot(t, v_raw, alpha=0.3, label='Raw', linewidth=0.5)
        ax.plot(t, v, label='Smoothed', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity')
        ax.set_title(f'Method: {method}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_fig_file = figures_path / f'{session_name}_velocity_comparison.png'
    plt.savefig(comparison_fig_file, dpi=150)
    print(f"Saved comparison figure to: {comparison_fig_file}")
    if show_plots:
        plt.show()
    plt.close(fig)

    print("\nFiltering position and differentiating...")
    t_adv, v_adv, vx_adv, vy_adv, info = compute_velocity_advanced(
        proc_file, cutoff_hz=cutoff_hz, order=order, max_gap_sec=max_gap_sec)
    print(f"  Camera: {1 / info['frame_interval_median']:.1f} fps median "
          f"(frame interval IQR {info['frame_interval_iqr'] * 1000:.1f} ms), "
          f"resampled to {info['grid_fs']:.1f} Hz for filtering")
    print(f"  Frozen (repeated) frames: {info['n_frozen_frames']:,} "
          f"({info['frozen_fraction']:.1%})")
    print(f"  Position jumps rejected: {info['n_jump_outliers']:,} "
          f"(>{info['jump_px']:.1f} px from the local median)")
    print(f"  Frames left NaN (no observation within "
          f"{info['max_gap_sec']:g}s): {info['n_uncovered']:,} "
          f"({info['uncovered_fraction']:.1%})")
    # save data to pickle
    velocity_data = {
        'time_stamp': t_adv,
        'velocity': v_adv,
        'velocity_x': vx_adv,
        'velocity_y': vy_adv,
        'source_proc_file': str(proc_file),
        'source_proc_name': proc_file.name,
        **info,
    }
    with open(velocity_output_file, 'wb') as f:
        pickle.dump(velocity_data, f)
    print(f"Saved velocity data to: {velocity_output_file}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t_adv, v_adv, linewidth=1)
    axes[0].set_ylabel('Speed')
    axes[0].set_title(f"Position low-passed at {cutoff_hz:g} Hz, then differentiated")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_adv, vx_adv, label='Vx', alpha=0.7)
    axes[1].plot(t_adv, vy_adv, label='Vy', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    stamp_figure(fig, f"source=proc_center (PROC center_x/center_y)  |  "
                      f"butterworth order={order} cutoff={cutoff_hz:g}Hz on position "
                      f"resampled to {info['grid_fs']:.1f}Hz, then d/dt  |  "
                      f"jump>{info['jump_px']:.1f}px rejected, max_gap="
                      f"{max_gap_sec:g}s  |  {info['frozen_fraction']:.1%} frozen, "
                      f"{info['uncovered_fraction']:.1%} left NaN  |  "
                      f"proc={proc_file}")
    advanced_fig_file = figures_path / f'{session_name}_velocity_advanced.png'
    plt.savefig(advanced_fig_file, dpi=150)
    print(f"Saved advanced velocity figure to: {advanced_fig_file}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return velocity_output_file


def _generate_keypoint_velocity(proc_file, session_name, velocity_output_file,
                                figures_path, keypoints, likelihood_threshold,
                                show_plots, cutoff_hz=DEFAULT_CUTOFF_HZ,
                                order=DEFAULT_FILTER_ORDER,
                                max_gap_sec=DEFAULT_MAX_GAP_SEC):
    """The 'dlc_body' branch of generate_velocity_outputs."""
    print(f"\nComputing velocity from DLC keypoints: {', '.join(keypoints)}")
    t, v, vx, vy, info = compute_velocity_from_keypoints(
        proc_file, keypoints=keypoints, likelihood_threshold=likelihood_threshold,
        cutoff_hz=cutoff_hz, order=order, max_gap_sec=max_gap_sec)

    print(f"  DLC file: {info['source_dlc_file']}")
    print(f"  Alignment: {info['alignment']} "
          f"({info['n_frames']} of {info['n_proc_frames']} PROC frames)")
    print(f"  Keypoints above likelihood {likelihood_threshold}: "
          f"{info['mean_keypoints_used']:.2f} of {len(keypoints)} per frame on average")
    print(f"  Frames with no confident keypoint: "
          f"{info['n_low_confidence_frames']:,} ({info['low_confidence_fraction']:.1%})")
    print(f"  Camera: {1 / info['frame_interval_median']:.1f} fps median "
          f"(frame interval IQR {info['frame_interval_iqr'] * 1000:.1f} ms), "
          f"resampled to {info['grid_fs']:.1f} Hz for filtering")
    print(f"  Position jumps rejected: {info['n_jump_outliers']:,} "
          f"(>{info['jump_px']:.1f} px from the local median)")
    print(f"  Frames left NaN (no confident keypoint within "
          f"{info['max_gap_sec']:g}s): {info['n_uncovered']:,} "
          f"({info['uncovered_fraction']:.1%})")
    print(f"  Speed: median {np.nanmedian(v):.2f}, p95 {np.nanpercentile(v, 95):.2f}, "
          f"max {np.nanmax(v):.2f} (position units/s)")

    velocity_data = {
        'time_stamp': t,
        'velocity': v,
        'velocity_x': vx,
        'velocity_y': vy,
        'source_proc_file': str(proc_file),
        'source_proc_name': proc_file.name,
        **info,
    }
    with open(velocity_output_file, 'wb') as f:
        pickle.dump(velocity_data, f)
    print(f"Saved velocity data to: {velocity_output_file}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, v, linewidth=1)
    axes[0].set_ylabel('Speed')
    axes[0].set_title(f"Keypoint centroid ({', '.join(keypoints)}) - position "
                      f"low-passed at {cutoff_hz:g} Hz, then differentiated")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vx, label='Vx', alpha=0.7)
    axes[1].plot(t, vy, label='Vy', alpha=0.7)
    axes[1].set_ylabel('Velocity Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Tracking quality: how many keypoints actually carried each frame. Grey =
    # no confident keypoint (interpolated); red = long enough that the speed
    # was left NaN rather than interpolated across.
    axes[2].plot(t, info['n_keypoints_used'], linewidth=0.5, color='tab:green')
    for start, stop in _mask_spans(t, info['low_confidence_mask']):
        axes[2].axvspan(start, stop, color='lightgray', alpha=0.6, lw=0)
    for start, stop in _mask_spans(t, ~info['covered_mask']):
        axes[2].axvspan(start, stop, color='tab:red', alpha=0.35, lw=0)
    axes[2].set_ylabel(f'Keypoints used\n(likelihood >= {likelihood_threshold})')
    axes[2].set_ylim(-0.2, len(keypoints) + 0.2)
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    stamp_figure(fig, f"source=dlc_body keypoints={','.join(keypoints)} "
                      f"likelihood>={likelihood_threshold} weighted="
                      f"{info['likelihood_weighted']}  |  butterworth order={order} "
                      f"cutoff={cutoff_hz:g}Hz on position resampled to "
                      f"{info['grid_fs']:.1f}Hz, then d/dt  |  "
                      f"jump>{info['jump_px']:.1f}px rejected, max_gap={max_gap_sec:g}s  |  "
                      f"{info['low_confidence_fraction']:.1%} low-confidence, "
                      f"{info['uncovered_fraction']:.1%} left NaN  |  "
                      f"dlc={info['source_dlc_file']}")
    figure_file = figures_path / f'{session_name}_velocity_body.png'
    plt.savefig(figure_file, dpi=150)
    print(f"Saved keypoint velocity figure to: {figure_file}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return velocity_output_file


def _mask_spans(t, mask):
    """(start, stop) time spans for each run of True in `mask`."""
    if not np.any(mask):
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [mask.size - 1]
    return [(t[a], t[min(b, t.size - 1)]) for a, b in zip(starts, stops)]


if __name__ == "__main__":
    # ACTIVE_DATE selects the entry in sleep_day_configs.json. Each active
    # session's proc_file is processed automatically, so no path prompt is
    # needed and pre/post velocity files cannot be accidentally crossed.
    import argparse

    from sleep_pipeline_config import (
        ACTIVE_DATE,
        VELOCITY_CUTOFF_HZ,
        VELOCITY_FILTER_ORDER,
        VELOCITY_KEYPOINTS,
        VELOCITY_LIKELIHOOD_THRESHOLD,
        VELOCITY_MAX_GAP_SEC,
        VELOCITY_SOURCE,
        active_sleep_sessions,
        sleep_sessions,
    )

    parser = argparse.ArgumentParser(
        description="Compute velocity for every active sleep session of ACTIVE_DATE.")
    parser.add_argument(
        "--source", choices=sorted(VELOCITY_SOURCES), default=VELOCITY_SOURCE,
        help="Which tracked point to differentiate: 'proc_center' (the PROC "
             "file's own head centre) or 'dlc_body' (centroid of --keypoints "
             "from the DLC companion file). Default: VELOCITY_SOURCE in "
             "sleep_pipeline_config.py.")
    parser.add_argument(
        "--keypoints", nargs="+", default=list(VELOCITY_KEYPOINTS),
        help="Keypoints forming the centroid when --source dlc_body. "
             f"Default: {' '.join(VELOCITY_KEYPOINTS)}")
    parser.add_argument(
        "--likelihood-threshold", type=float, default=VELOCITY_LIKELIHOOD_THRESHOLD,
        help="Per-frame DLC likelihood a keypoint must reach to be averaged in.")
    parser.add_argument(
        "--cutoff-hz", type=float, default=VELOCITY_CUTOFF_HZ,
        help="Position low-pass cutoff in Hz, applied before differentiating. "
             f"Default: {VELOCITY_CUTOFF_HZ} Hz.")
    parser.add_argument(
        "--filter-order", type=int, default=VELOCITY_FILTER_ORDER,
        help="Butterworth order (run forward+backward, so zero phase).")
    parser.add_argument(
        "--max-gap-sec", type=float, default=VELOCITY_MAX_GAP_SEC,
        help="Frames further than this from a real detection are left NaN "
             "instead of interpolated.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even when the velocity pickle already exists "
             "(default: existing outputs are kept and the session is skipped).")
    args = parser.parse_args()

    sessions_to_run = active_sleep_sessions(sleep_sessions)
    print(f"Generating velocity outputs for sleep day {ACTIVE_DATE} "
          f"(source={args.source})")
    if not sessions_to_run:
        print("No active sleep sessions - nothing to do.")

    for session_key, session_cfg in sessions_to_run.items():
        proc_file = session_cfg.get('proc_file')
        print(f"\n{'=' * 60}\nSLEEP SESSION: {session_key}\n{'=' * 60}")
        if not proc_file:
            print("No proc_file registered - skipping.")
            continue
        try:
            generate_velocity_outputs(
                proc_file,
                source=args.source,
                overwrite=args.overwrite,
                keypoints=tuple(args.keypoints),
                likelihood_threshold=args.likelihood_threshold,
                cutoff_hz=args.cutoff_hz,
                order=args.filter_order,
                max_gap_sec=args.max_gap_sec,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"WARNING: {exc}")
