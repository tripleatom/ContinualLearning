"""
Plot per-trial mouse traces from the pkl produced by readDIO_grating.py.
Each trial is a small subplot; dots are scatter-plotted at (x, y) and
color-coded by instantaneous speed (pixels/sec).

Output: <sortout_folder>/behavior_analysis/trial_traces_velocity.png
"""
import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter


# Position-unit → cm conversion. 1 dlc unit = 1 mm = 0.1 cm, so there are
# 10 position-units per cm. Set to None to bypass conversion entirely
# (speed reported in raw position-units/sec).
POSITION_UNITS_PER_CM = 10.0

# Drop any sample whose instantaneous speed exceeds this value (tracking glitches).
MAX_SPEED_CM_S = 600.0

# Hampel-filter parameters for flicker (jump-and-return) detection on x, y.
# Window is in samples (centered); a point is flagged if it deviates from the
# window's median by more than HAMPEL_K * MAD (in either x or y).
HAMPEL_WINDOW = 7
HAMPEL_K = 4.0


def _hampel_outlier_mask(values, window=7, k=4.0):
    """
    Return a boolean mask (True = outlier) flagging samples that deviate from
    the local rolling median by more than k * MAD. MAD is scaled by 1.4826 so
    k is interpreted in units of standard deviations under a Gaussian model.

    Centered window of `window` samples; ends use a one-sided window. A constant
    window (MAD == 0) yields no flags for that region.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    half = max(1, window // 2)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = values[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        if mad == 0:
            continue
        if abs(values[i] - med) > k * 1.4826 * mad:
            mask[i] = True
    return mask


def remove_flicker(x, y, t, window=HAMPEL_WINDOW, k=HAMPEL_K):
    """
    Drop samples flagged as flickers (isolated jump-and-return) by a Hampel
    filter on either x or y. Returns filtered (x, y, t) plus number dropped.
    """
    if x.size < 3:
        return x, y, t, 0
    bad = _hampel_outlier_mask(x, window=window, k=k) | \
          _hampel_outlier_mask(y, window=window, k=k)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return x, y, t, 0
    keep = ~bad
    return x[keep], y[keep], t[keep], n_bad


def _smooth_and_diff(values, t, window_s=0.2, polyorder=2):
    """
    Savitzky-Golay smoothing of a 1D signal followed by central-difference
    differentiation against the (possibly non-uniform) time vector.

    Stable against tiny dt between adjacent samples — the local polynomial fit
    suppresses high-frequency tracking jitter before differentiation.
    """
    if values.size < 5:
        return np.zeros_like(values)
    dt_med = np.median(np.diff(t))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return np.gradient(values, t)
    win = int(round(window_s / dt_med))
    win = max(polyorder + 2, win)
    if win % 2 == 0:
        win += 1
    win = min(win, values.size if values.size % 2 == 1 else values.size - 1)
    if win <= polyorder:
        smoothed = values
    else:
        smoothed = savgol_filter(values, window_length=win, polyorder=polyorder, mode='interp')
    return np.gradient(smoothed, t)


def compute_speed(x, y, t, smooth_window_s=0.2):
    """
    Per-sample speed (length-units / sec), computed in a stable way:
    Savitzky-Golay smoothing on x(t) and y(t), then central differences against
    the actual time vector via np.gradient. Output length matches x/y/t.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(x) < 2:
        return np.zeros_like(x)
    # Drop duplicate timestamps so np.gradient stays well-defined.
    keep = np.concatenate(([True], np.diff(t) > 0))
    if not np.all(keep):
        x, y, t = x[keep], y[keep], t[keep]
    if len(x) < 2:
        return np.zeros(len(keep))
    vx = _smooth_and_diff(x, t, window_s=smooth_window_s)
    vy = _smooth_and_diff(y, t, window_s=smooth_window_s)
    v = np.sqrt(vx * vx + vy * vy)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    # Re-expand to original length: pad dropped duplicate-timestamp slots with 0.
    if not np.all(keep):
        full = np.zeros(len(keep))
        full[keep] = v
        return full
    return v


def speed_threshold_in_position_units(max_speed_cm_s=None):
    """Return the speed cap converted to position-units/sec (px/s if calibrated).
    If `max_speed_cm_s` is None, falls back to the module-level MAX_SPEED_CM_S."""
    cap = MAX_SPEED_CM_S if max_speed_cm_s is None else float(max_speed_cm_s)
    if POSITION_UNITS_PER_CM is None:
        return cap
    return cap * POSITION_UNITS_PER_CM


def speed_to_cm_per_s(v):
    """Convert speed array from position-units/sec → cm/s (identity if no calibration)."""
    if POSITION_UNITS_PER_CM is None:
        return v
    return v / POSITION_UNITS_PER_CM


def clean_position(channels, *, confidence_threshold=None, max_speed_cm_s=None):
    """
    Standard DLC cleaning pipeline applied to a set of co-indexed 1-D arrays.

    Parameters
    ----------
    channels : dict[str, array_like]
        Must contain at least 'x', 'y', 't'. Any additional keys (e.g.
        'heading', 'head_angle', 'dlc_signal') are propagated through every
        mask so they stay aligned with the kept x/y/t samples.
    confidence_threshold : float or None
        If given and 'dlc_signal' is in `channels`, samples with
        `dlc_signal < confidence_threshold` (or non-finite) are dropped.

    Pipeline
    --------
    1. drop forward-fill zero pad ((x == 0) & (y == 0))
    2. drop Hampel-flagged flicker samples on x or y
    3. compute speed v from (x, y, t); drop samples with v > MAX_SPEED_CM_S
    4. optional: drop samples with dlc_signal below confidence_threshold

    Returns
    -------
    cleaned : dict
        Same keys as `channels`, plus 'v' (speed in position-units/s). All
        arrays have equal length.
    stats : dict
        Per-step drop counts: n_input, n_zero_pad_dropped, n_flicker_dropped,
        n_speed_dropped, n_confidence_dropped, n_kept.
    """
    if not {'x', 'y', 't'}.issubset(channels):
        raise KeyError("clean_position requires 'x', 'y', 't' in channels")

    out = {k: np.asarray(v, dtype=float).copy() for k, v in channels.items()}
    n_input = out['x'].size

    # Auto-pad mismatched aux channels with NaN so all arrays share length n_input.
    for k in list(out.keys()):
        if k in ('x', 'y', 't'):
            continue
        if out[k].size != n_input:
            out[k] = np.full(n_input, np.nan)

    # 1) zero-pad drop
    if out['x'].size:
        valid = ~((out['x'] == 0) & (out['y'] == 0))
        out = {k: v[valid] for k, v in out.items()}
    n_zero_pad_dropped = n_input - out['x'].size

    # 2) Hampel flicker drop on x or y
    n_before_flicker = out['x'].size
    if n_before_flicker >= 3:
        bad = (_hampel_outlier_mask(out['x'], window=HAMPEL_WINDOW, k=HAMPEL_K)
               | _hampel_outlier_mask(out['y'], window=HAMPEL_WINDOW, k=HAMPEL_K))
        if bad.any():
            keep = ~bad
            out = {k: v[keep] for k, v in out.items()}
    n_flicker_dropped = n_before_flicker - out['x'].size

    # 3) speed compute + clip
    out['v'] = compute_speed(out['x'], out['y'], out['t'])
    smax = speed_threshold_in_position_units(max_speed_cm_s)
    speed_keep = out['v'] <= smax
    n_speed_dropped = int((~speed_keep).sum())
    out = {k: v[speed_keep] for k, v in out.items()}

    # 4) optional confidence threshold (only if both threshold and channel present)
    n_confidence_dropped = 0
    if confidence_threshold is not None and 'dlc_signal' in out:
        sig = out['dlc_signal']
        conf_keep = np.isfinite(sig) & (sig >= confidence_threshold)
        n_confidence_dropped = int((~conf_keep).sum())
        out = {k: v[conf_keep] for k, v in out.items()}

    stats = {
        'n_input': n_input,
        'n_zero_pad_dropped': n_zero_pad_dropped,
        'n_flicker_dropped': n_flicker_dropped,
        'n_speed_dropped': n_speed_dropped,
        'n_confidence_dropped': n_confidence_dropped,
        'n_kept': out['x'].size,
    }
    return out, stats


def clean_position_p99(channels, hard_cm_s=60.0, *, confidence_threshold=None):
    """
    Like `clean_position`, but the speed cap is `min(hard_cm_s, p99(speed))`
    rather than the fixed module-level MAX_SPEED_CM_S. Useful when the 600 cm/s
    fallback is too lenient and a data-driven cap is preferred.

    Returns the same (cleaned, stats) shape as `clean_position`, with an extra
    'speed_threshold_cm_s' entry in `stats` recording the chosen cap.
    """
    pre, pre_stats = clean_position(
        channels,
        confidence_threshold=confidence_threshold,
        max_speed_cm_s=np.inf,
    )
    v_cm = speed_to_cm_per_s(pre['v'])
    if v_cm.size:
        p99_cm = float(np.percentile(v_cm, 99))
        threshold_cm = min(float(hard_cm_s), p99_cm)
    else:
        threshold_cm = float(hard_cm_s)
    smax = speed_threshold_in_position_units(threshold_cm)
    keep = pre['v'] <= smax
    cleaned = {k: v[keep] for k, v in pre.items()}
    stats = dict(pre_stats)
    stats['n_speed_dropped'] = int((~keep).sum())
    stats['n_kept'] = cleaned['x'].size
    stats['speed_threshold_cm_s'] = threshold_cm
    return cleaned, stats


def collect_traces(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    trials = data['trial_info']['all_trial_parameters']

    speed_max = speed_threshold_in_position_units()
    n_after_zero_pad = 0   # samples entering the flicker step (across all trials)
    n_after_flicker  = 0   # samples entering the speed-clip step
    n_flicker_dropped = 0
    n_speed_dropped = 0

    out = []
    for tp in trials:
        cleaned, stats = clean_position({
            'x': tp.get('position_x', []),
            'y': tp.get('position_y', []),
            't': tp.get('position_time', []),
        })
        n_after_zero_pad  += stats['n_input'] - stats['n_zero_pad_dropped']
        n_after_flicker   += (stats['n_input']
                              - stats['n_zero_pad_dropped']
                              - stats['n_flicker_dropped'])
        n_flicker_dropped += stats['n_flicker_dropped']
        n_speed_dropped   += stats['n_speed_dropped']
        out.append({
            'trial_index': tp.get('trial_index'),
            'choice': tp.get('choice'),
            'correct': tp.get('correct'),
            'rewarded_on_left': tp.get('rewarded_on_left'),
            'x': cleaned['x'], 'y': cleaned['y'],
            't': cleaned['t'], 'v': cleaned['v'],
        })

    unit = 'cm/s' if POSITION_UNITS_PER_CM is not None else 'position-units/s (no calibration set)'
    print(f"Speed threshold: {MAX_SPEED_CM_S} {unit} "
          f"(= {speed_max:.1f} position-units/sec)")
    if n_after_flicker:
        print(f"Dropped {n_speed_dropped}/{n_after_flicker} samples "
              f"({100 * n_speed_dropped / n_after_flicker:.2f}%) above threshold.")
    if n_after_zero_pad:
        print(f"Hampel flicker filter (window={HAMPEL_WINDOW}, k={HAMPEL_K}): "
              f"dropped {n_flicker_dropped}/{n_after_zero_pad} samples "
              f"({100 * n_flicker_dropped / max(n_after_zero_pad, 1):.2f}%).")
    return out, data


def plot_traces(traces, save_path, vmax_percentile=99):
    n = len(traces)
    if n == 0:
        print("No trials with position data; nothing to plot.")
        return

    # Shared axes limits across all trials so trajectories are comparable.
    all_x = np.concatenate([tr['x'] for tr in traces if tr['x'].size]) if any(tr['x'].size for tr in traces) else np.array([0, 1])
    all_y = np.concatenate([tr['y'] for tr in traces if tr['y'].size]) if any(tr['y'].size for tr in traces) else np.array([0, 1])
    xlim = (np.min(all_x), np.max(all_x))
    ylim = (np.min(all_y), np.max(all_y))

    # Shared color scale across trials, in cm/s when calibrated.
    # Clip top end so a few outliers don't wash out the rest.
    all_v_disp = np.concatenate([speed_to_cm_per_s(tr['v']) for tr in traces if tr['v'].size])
    vmax = float(np.percentile(all_v_disp, vmax_percentile)) if all_v_disp.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    speed_unit = 'cm/s' if POSITION_UNITS_PER_CM is not None else 'position-units/s'

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6),
                             squeeze=False, sharex=True, sharey=True)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n:
            ax.axis('off')
            continue
        tr = traces[idx]
        if tr['x'].size:
            # Connecting line for the trajectory (thin, behind the dots).
            ax.plot(tr['x'], tr['y'], color='0.6', linewidth=0.5,
                    alpha=0.7, zorder=1)
            ax.scatter(tr['x'], tr['y'], c=speed_to_cm_per_s(tr['v']),
                       cmap=cmap, norm=norm, s=2, linewidths=0, alpha=0.85,
                       zorder=2)
            # Start (green circle) and end (red square) markers.
            ax.scatter(tr['x'][0], tr['y'][0], marker='o', s=18,
                       facecolor='#2ecc71', edgecolor='black', linewidths=0.4,
                       zorder=3)
            ax.scatter(tr['x'][-1], tr['y'][-1], marker='s', s=18,
                       facecolor='#e74c3c', edgecolor='black', linewidths=0.4,
                       zorder=3)
        # Outline color: green for correct, red for incorrect, gray if unknown.
        correct = tr['correct']
        edge = 'gray' if correct is None else ('#2ecc71' if correct else '#e74c3c')
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()  # image coordinates: y increases downward
        ax.set_title(f"#{tr['trial_index']}", fontsize=6, pad=1)

    # Single shared colorbar.
    fig.subplots_adjust(left=0.03, right=0.92, top=0.96, bottom=0.04,
                        wspace=0.15, hspace=0.25)
    cbar_ax = fig.add_axes([0.94, 0.10, 0.012, 0.80])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'Speed ({speed_unit}, clipped at {vmax_percentile}th pct)', fontsize=9)

    fig.suptitle(f"Per-trial mouse trace, dots colored by speed  (n={n} trials)",
                 fontsize=11, y=0.995)

    # Figure-level legend for start/end markers (single instance, top-left).
    start_handle = plt.Line2D([], [], marker='o', linestyle='None',
                              markerfacecolor='#2ecc71', markeredgecolor='black',
                              markersize=6, label='start')
    end_handle = plt.Line2D([], [], marker='s', linestyle='None',
                            markerfacecolor='#e74c3c', markeredgecolor='black',
                            markersize=6, label='end')
    fig.legend(handles=[start_handle, end_handle], loc='upper left',
               bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_overlay(traces, save_path, vmax_percentile=99):
    """
    2x1 overlay: left panel colors trajectories by correctness
    (green = correct, red = incorrect, gray = unknown); right panel colors
    each segment by instantaneous speed using a shared colormap.
    """
    n = len(traces)
    if n == 0:
        print("No trials with position data; nothing to overlay.")
        return

    all_x = np.concatenate([tr['x'] for tr in traces if tr['x'].size]) if any(tr['x'].size for tr in traces) else np.array([0, 1])
    all_y = np.concatenate([tr['y'] for tr in traces if tr['y'].size]) if any(tr['y'].size for tr in traces) else np.array([0, 1])
    xlim = (np.min(all_x), np.max(all_x))
    ylim = (np.min(all_y), np.max(all_y))

    all_v_disp = np.concatenate([speed_to_cm_per_s(tr['v']) for tr in traces if tr['v'].size])
    vmax = float(np.percentile(all_v_disp, vmax_percentile)) if all_v_disp.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    speed_unit = 'cm/s' if POSITION_UNITS_PER_CM is not None else 'position-units/s'

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_corr, ax_vel = axes

    counts = {'correct': 0, 'incorrect': 0, 'unknown': 0}
    for tr in traces:
        if tr['x'].size == 0:
            continue
        correct = tr['correct']
        if correct is None:
            color, key = '0.5', 'unknown'
        elif correct:
            color, key = '#2ecc71', 'correct'
        else:
            color, key = '#e74c3c', 'incorrect'
        counts[key] += 1
        ax_corr.plot(tr['x'], tr['y'], color=color, linewidth=0.6,
                     alpha=0.35, zorder=1)

        if tr['x'].size >= 2:
            pts = np.column_stack([tr['x'], tr['y']]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_speed = 0.5 * (speed_to_cm_per_s(tr['v'][:-1])
                               + speed_to_cm_per_s(tr['v'][1:]))
            lc = LineCollection(segs, cmap=cmap, norm=norm,
                                linewidth=0.6, alpha=0.5, zorder=1)
            lc.set_array(seg_speed)
            ax_vel.add_collection(lc)

        for ax in (ax_corr, ax_vel):
            ax.scatter(tr['x'][0], tr['y'][0], marker='o', s=8,
                       facecolor='#2ecc71', edgecolor='black',
                       linewidths=0.2, alpha=0.6, zorder=3)
            ax.scatter(tr['x'][-1], tr['y'][-1], marker='s', s=8,
                       facecolor='#e74c3c', edgecolor='black',
                       linewidths=0.2, alpha=0.6, zorder=3)

    handles = [
        plt.Line2D([], [], color='#2ecc71', linewidth=2,
                   label=f"correct (n={counts['correct']})"),
        plt.Line2D([], [], color='#e74c3c', linewidth=2,
                   label=f"incorrect (n={counts['incorrect']})"),
    ]
    if counts['unknown']:
        handles.append(plt.Line2D([], [], color='0.5', linewidth=2,
                                  label=f"unknown (n={counts['unknown']})"))
    handles += [
        plt.Line2D([], [], marker='o', linestyle='None',
                   markerfacecolor='#2ecc71', markeredgecolor='black',
                   markersize=6, label='start'),
        plt.Line2D([], [], marker='s', linestyle='None',
                   markerfacecolor='#e74c3c', markeredgecolor='black',
                   markersize=6, label='end'),
    ]
    ax_corr.legend(handles=handles, loc='best', fontsize=8, frameon=False)
    ax_corr.set_title(f"Colored by correctness (n={n})", fontsize=11)
    ax_vel.set_title(f"Colored by speed (n={n})", fontsize=11)

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_vel, fraction=0.046, pad=0.04)
    cbar.set_label(f'Speed ({speed_unit}, clipped at {vmax_percentile}th pct)',
                   fontsize=9)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    from params import sortout_folder

    sortout = Path(sortout_folder)
    pkl_file = sortout / f'task_spikes_trial_{sortout.name}.pkl'
    if not pkl_file.exists():
        raise FileNotFoundError(
            f"Expected pkl not found: {pkl_file}\n"
            f"Run readDIO_grating.py first."
        )

    traces, _ = collect_traces(pkl_file)
    out_dir = sortout / 'behavior_analysis'
    plot_traces(traces, out_dir / 'trial_traces_velocity.png')
    plot_overlay(traces, out_dir / 'trial_traces_overlay.png')
