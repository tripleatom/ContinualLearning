"""
Plot per-trial duration distribution and in-task mouse velocity distribution
from the PKL produced by readDIO_grating.py.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from params import sortout_folder

# 1 dlc position-unit = 1 mm = 0.1 cm  →  10 units per cm
POSITION_UNITS_PER_CM = 10.0


def smooth_and_diff(values, t, window_s=0.2, polyorder=2):
    """
    Smooth a 1D signal with Savitzky-Golay, then differentiate against time
    using central differences. Handles non-uniform time by using np.gradient(values, t).

    window_s: smoothing window length in seconds (converted to odd sample count).
    """
    if values.size < 5:
        return np.zeros_like(values)
    # Median sampling rate from the time vector (robust to occasional gaps)
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


def load_neural_data(sortout_folder):
    session_folder = Path(sortout_folder)
    pkl_file = session_folder / f'task_spikes_trial_{session_folder.name}.pkl'
    if not pkl_file.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)


def compute_trial_velocities(trial_params, smooth_window_s=0.2, max_v_cm_s=60.0):
    """
    Stable per-trial speed in cm/s: Savitzky-Golay smoothing on x(t), y(t)
    followed by central-difference differentiation against the actual time
    vector. Output is converted from position-units/s (mm/s) to cm/s using
    POSITION_UNITS_PER_CM.

    Avoids the small-dt noise blowup of pointwise finite differences.

    smooth_window_s : smoothing window length (seconds).
    max_v_cm_s      : reject samples with speed above this (cm/s).
    """
    velocities = []
    per_trial_mean = []
    for tp in trial_params:
        x = np.asarray(tp.get('position_x', []), dtype=float)
        y = np.asarray(tp.get('position_y', []), dtype=float)
        t = np.asarray(tp.get('position_time', []), dtype=float)
        if x.size < 5 or t.size < 5 or t[-1] <= t[0]:
            per_trial_mean.append(np.nan)
            continue
        # Drop duplicate timestamps to keep gradient stable
        keep = np.concatenate(([True], np.diff(t) > 0))
        x, y, t = x[keep], y[keep], t[keep]
        if x.size < 5:
            per_trial_mean.append(np.nan)
            continue
        vx = smooth_and_diff(x, t, window_s=smooth_window_s)
        vy = smooth_and_diff(y, t, window_s=smooth_window_s)
        v = np.sqrt(vx ** 2 + vy ** 2) / POSITION_UNITS_PER_CM  # → cm/s
        v = v[v <= max_v_cm_s]
        if v.size == 0:
            per_trial_mean.append(np.nan)
            continue
        velocities.append(v)
        per_trial_mean.append(float(np.nanmean(v)))
    if velocities:
        all_v = np.concatenate(velocities)
    else:
        all_v = np.array([], dtype=float)
    return all_v, np.array(per_trial_mean, dtype=float)


def main():
    data = load_neural_data(sortout_folder)
    session_folder = Path(sortout_folder)

    trial_durations = np.asarray(data['trial_info']['trial_durations'], dtype=float)
    trial_params = data['trial_info']['all_trial_parameters']

    all_velocities, per_trial_mean_v = compute_trial_velocities(trial_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    parts = ax.violinplot(trial_durations, showmeans=False, showmedians=False,
                          showextrema=False, widths=0.8)
    for body in parts['bodies']:
        body.set_facecolor('steelblue')
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    # Strip plot of individual trials, jittered for visibility.
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.06, 0.06, size=trial_durations.size)
    ax.scatter(1 + jitter, trial_durations, s=8, color='black', alpha=0.4, zorder=3)
    # Mean and median markers.
    dur_mean = float(np.mean(trial_durations))
    dur_median = float(np.median(trial_durations))
    dur_min = float(np.min(trial_durations))
    dur_max = float(np.max(trial_durations))
    dur_std = float(np.std(trial_durations))
    dur_q1, dur_q3 = np.percentile(trial_durations, [25, 75])
    ax.hlines(dur_median, 0.7, 1.3, colors='red', linestyles='--',
              label=f'median = {dur_median:.2f}s')
    ax.hlines(dur_mean, 0.7, 1.3, colors='orange', linestyles='--',
              label=f'mean = {dur_mean:.2f}s')
    ax.set_xticks([1])
    ax.set_xticklabels(['trials'])
    ax.set_ylabel('Trial duration (s)')
    ax.set_title(f'Trial duration distribution (n={len(trial_durations)})')
    ax.legend(loc='upper left')

    stats_text = (f'n = {len(trial_durations)}\n'
                  f'min  = {dur_min:.2f} s\n'
                  f'max  = {dur_max:.2f} s\n'
                  f'mean = {dur_mean:.2f} s\n'
                  f'median = {dur_median:.2f} s\n'
                  f'std  = {dur_std:.2f} s\n'
                  f'IQR  = [{dur_q1:.2f}, {dur_q3:.2f}] s')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    ax = axes[1]
    if all_velocities.size > 0:
        # clip extreme tail for readability
        upper = np.nanpercentile(all_velocities, 99)
        v_plot = all_velocities[all_velocities <= upper]
        v_med = float(np.nanmedian(all_velocities))
        v_mean = float(np.nanmean(all_velocities))
        v_min = float(np.nanmin(all_velocities))
        v_max = float(np.nanmax(all_velocities))
        v_std = float(np.nanstd(all_velocities))
        v_q1, v_q3 = np.nanpercentile(all_velocities, [25, 75])
        ax.hist(v_plot, bins=60, color='seagreen', edgecolor='black', alpha=0.85)
        ax.axvline(v_med, color='red', linestyle='--',
                   label=f'median = {v_med:.2f} cm/s')
        ax.axvline(v_mean, color='orange', linestyle='--',
                   label=f'mean = {v_mean:.2f} cm/s')
        ax.set_xlabel('In-task velocity (cm/s)')
        ax.set_ylabel('Number of samples')
        ax.set_title(f'Mouse velocity distribution (clipped at 99th pct)')
        ax.legend(loc='upper left')

        v_stats = (f'n = {all_velocities.size}\n'
                   f'min  = {v_min:.2f} cm/s\n'
                   f'max  = {v_max:.2f} cm/s\n'
                   f'mean = {v_mean:.2f} cm/s\n'
                   f'median = {v_med:.2f} cm/s\n'
                   f'std  = {v_std:.2f} cm/s\n'
                   f'IQR  = [{v_q1:.2f}, {v_q3:.2f}] cm/s')
        ax.text(0.98, 0.98, v_stats, transform=ax.transAxes,
                ha='right', va='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.85))
    else:
        ax.text(0.5, 0.5, 'No position data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Mouse velocity distribution')

    fig.suptitle(f'Session: {session_folder.name}', y=1.02)
    fig.tight_layout()

    out_dir = session_folder / 'behavior_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f'trial_length_velocity_{session_folder.name}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved figure → {out_png}")

    print("\nTrial duration stats (s):")
    print(f"  n = {len(trial_durations)}")
    print(f"  mean   = {np.mean(trial_durations):.3f}")
    print(f"  median = {np.median(trial_durations):.3f}")
    print(f"  min    = {np.min(trial_durations):.3f}")
    print(f"  max    = {np.max(trial_durations):.3f}")

    if all_velocities.size > 0:
        print("\nVelocity stats (cm/s):")
        print(f"  n samples = {all_velocities.size}")
        print(f"  mean   = {np.nanmean(all_velocities):.3f}")
        print(f"  median = {np.nanmedian(all_velocities):.3f}")
        print(f"  per-trial mean velocity: mean={np.nanmean(per_trial_mean_v):.3f}, "
              f"median={np.nanmedian(per_trial_mean_v):.3f}")

    plt.show()


if __name__ == '__main__':
    main()
