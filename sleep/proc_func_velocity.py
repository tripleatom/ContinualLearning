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

Both paths end in the same Savitzky-Golay differentiation and write the same
pkl keys (time_stamp / velocity / velocity_x / velocity_y) on the PROC
time base, so video<->ephys sync downstream is unchanged.
"""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from server_fallback import (mirror_on_backup_server, resolve_existing_file,
                             resolve_output_folder)

# Keypoints in the DLC companion file, grouped as the model names them.
HEAD_KEYPOINTS = ('nose', 'left_eye', 'right_eye',
                  'left_bar', 'right_bar', 'cable_base')
BODY_KEYPOINTS = ('left_midside', 'right_midside',
                  'left_hip', 'right_hip', 'tail_base')

# Per-frame likelihood a keypoint must reach to enter the centroid.
DEFAULT_LIKELIHOOD_THRESHOLD = 0.6

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

    # Optional: Apply additional smoothing to velocity itself
    if method in ['savgol', 'median', 'gaussian'] and len(v) > window_length:
        # Smooth velocity as well for extra smoothness
        v_smoothed = savgol_filter(v, window_length=min(window_length, len(v)//2*2+1),
                                   polyorder=min(polyorder, min(window_length, len(v)//2*2+1)-1))
        v = v_smoothed

    return t, v, v_raw_clean


def compute_velocity_advanced(proc_file, velocity_threshold=530,
                              window_length=11, polyorder=3):
    """
    Advanced velocity computation using Savitzky-Golay differentiation.
    This directly computes the derivative while smoothing, which is more
    accurate than smoothing then differencing.

    Parameters
    ----------
    proc_file : str
        Path to the _PROC pickle file containing tracking data
    velocity_threshold : float, optional
        Maximum velocity threshold. Values above this are set to NaN (default: 530)
    window_length : int, optional
        Length of the filter window (must be odd). Default: 11
    polyorder : int, optional
        Order of polynomial for Savitzky-Golay filter. Default: 3

    Returns
    -------
    t : numpy.ndarray
        Time stamps
    v : numpy.ndarray
        Velocity values (smoothed)
    vx : numpy.ndarray
        X component of velocity
    vy : numpy.ndarray
        Y component of velocity
    """
    # Load data
    data = pickle.load(open(resolve_proc_file(proc_file), 'rb'))

    x = data['center_x']
    y = data['center_y']
    time_stamp = data['time_stamp']

    v_interp, vx_interp, vy_interp = savgol_velocity(
        x, y, time_stamp, velocity_threshold, window_length, polyorder)

    return time_stamp, v_interp, vx_interp, vy_interp


def savgol_velocity(x, y, time_stamp, velocity_threshold=530,
                    window_length=11, polyorder=3):
    """Speed and components from an (x, y, t) track, by Savitzky-Golay derivative.

    Shared by both velocity sources (PROC centre and DLC keypoint centroid) so
    the two differ only in which point is tracked, never in the math.

    Returns (v, vx, vy), all on the given `time_stamp`, with values above
    `velocity_threshold` removed and interpolated over.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    time_stamp = np.asarray(time_stamp, dtype=float)

    # Calculate mean sampling rate
    dt_mean = np.mean(np.diff(time_stamp))

    # Use Savitzky-Golay filter with derivative mode
    # This computes the derivative while smoothing in one step
    vx = savgol_filter(x, window_length=window_length, polyorder=polyorder,
                       deriv=1, delta=dt_mean)
    vy = savgol_filter(y, window_length=window_length, polyorder=polyorder,
                       deriv=1, delta=dt_mean)

    # Calculate velocity magnitude
    v = np.sqrt(vx**2 + vy**2)

    # Remove outliers: set velocities above threshold to NaN (better than 0)
    v = np.where(v > velocity_threshold, np.nan, v)
    vx = np.where(np.abs(vx) > velocity_threshold, np.nan, vx)
    vy = np.where(np.abs(vy) > velocity_threshold, np.nan, vy)

    # Optionally interpolate over NaN values
    # This is better than setting to 0
    mask = ~np.isnan(v)
    if np.sum(mask) > 0:  # If we have valid values
        v_interp = np.interp(time_stamp, time_stamp[mask], v[mask])
        vx_interp = np.interp(time_stamp, time_stamp[mask], vx[mask])
        vy_interp = np.interp(time_stamp, time_stamp[mask], vy[mask])
    else:
        v_interp = v
        vx_interp = vx
        vy_interp = vy

    return v_interp, vx_interp, vy_interp


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
                                    velocity_threshold=530, window_length=11,
                                    polyorder=3, weighted=True):
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

    x = fill_gaps(x, time_stamp)
    y = fill_gaps(y, time_stamp)
    v, vx, vy = savgol_velocity(x, y, time_stamp, velocity_threshold,
                                window_length, polyorder)

    info = {
        'source': 'dlc_body',
        'keypoints': list(keypoints),
        'likelihood_threshold': float(likelihood_threshold),
        'likelihood_weighted': bool(weighted),
        'velocity_threshold': float(velocity_threshold),
        'window_length': int(window_length),
        'polyorder': int(polyorder),
        'alignment': how,
        'n_frames': int(time_stamp.size),
        'n_proc_frames': int(np.asarray(proc_data['frame_time']).size),
        'n_low_confidence_frames': int(dropped.sum()),
        'low_confidence_fraction': float(np.mean(dropped)),
        'mean_keypoints_used': float(np.mean(n_used)),
        'centroid_x': x,
        'centroid_y': y,
        'n_keypoints_used': n_used,
        'low_confidence_mask': dropped,
        'source_dlc_file': str(dlc_file),
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
                              likelihood_threshold=DEFAULT_LIKELIHOOD_THRESHOLD):
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
            keypoints, likelihood_threshold, show_plots)

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

    # Demonstrate advanced method
    print("\nTesting advanced method...")
    t_adv, v_adv, vx_adv, vy_adv = compute_velocity_advanced(
        proc_file, window_length=11)
    # save data to pickle
    velocity_data = {
        'time_stamp': t_adv,
        'velocity': v_adv,
        'velocity_x': vx_adv,
        'velocity_y': vy_adv,
        'source': 'proc_center',
        'source_proc_file': str(proc_file),
        'source_proc_name': proc_file.name,
    }
    with open(velocity_output_file, 'wb') as f:
        pickle.dump(velocity_data, f)
    print(f"Saved velocity data to: {velocity_output_file}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t_adv, v_adv, linewidth=1)
    axes[0].set_ylabel('Speed')
    axes[0].set_title('Advanced Method: Savitzky-Golay Derivative')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_adv, vx_adv, label='Vx', alpha=0.7)
    axes[1].plot(t_adv, vy_adv, label='Vy', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    stamp_figure(fig, f"source=proc_center (PROC center_x/center_y)  |  "
                      f"savgol deriv window=11 polyorder=3 threshold=530  |  "
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
                                show_plots):
    """The 'dlc_body' branch of generate_velocity_outputs."""
    print(f"\nComputing velocity from DLC keypoints: {', '.join(keypoints)}")
    t, v, vx, vy, info = compute_velocity_from_keypoints(
        proc_file, keypoints=keypoints, likelihood_threshold=likelihood_threshold)

    print(f"  DLC file: {info['source_dlc_file']}")
    print(f"  Alignment: {info['alignment']} "
          f"({info['n_frames']} of {info['n_proc_frames']} PROC frames)")
    print(f"  Keypoints above likelihood {likelihood_threshold}: "
          f"{info['mean_keypoints_used']:.2f} of {len(keypoints)} per frame on average")
    print(f"  Frames with no confident keypoint (interpolated): "
          f"{info['n_low_confidence_frames']:,} ({info['low_confidence_fraction']:.1%})")
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
    axes[0].set_title(f"Keypoint centroid ({', '.join(keypoints)}) - "
                      f"Savitzky-Golay derivative")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vx, label='Vx', alpha=0.7)
    axes[1].plot(t, vy, label='Vy', alpha=0.7)
    axes[1].set_ylabel('Velocity Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Tracking quality: how many keypoints actually carried each frame, with
    # the interpolated (no confident keypoint) frames shaded.
    axes[2].plot(t, info['n_keypoints_used'], linewidth=0.5, color='tab:green')
    for start, stop in _mask_spans(t, info['low_confidence_mask']):
        axes[2].axvspan(start, stop, color='lightgray', alpha=0.6, lw=0)
    axes[2].set_ylabel(f'Keypoints used\n(likelihood >= {likelihood_threshold})')
    axes[2].set_ylim(-0.2, len(keypoints) + 0.2)
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    stamp_figure(fig, f"source=dlc_body keypoints={','.join(keypoints)} "
                      f"likelihood>={likelihood_threshold} weighted="
                      f"{info['likelihood_weighted']}  |  savgol deriv "
                      f"window={info['window_length']} polyorder={info['polyorder']} "
                      f"threshold={info['velocity_threshold']:g}  |  "
                      f"{info['low_confidence_fraction']:.1%} frames interpolated  |  "
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
        VELOCITY_KEYPOINTS,
        VELOCITY_LIKELIHOOD_THRESHOLD,
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
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"WARNING: {exc}")
