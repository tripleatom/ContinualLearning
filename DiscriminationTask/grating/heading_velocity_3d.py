"""
3D scatter of (speed, heading, head_angle) per 500 ms bin, color-coded by
vstim_on / vstim_off, plus the three 2D projections.

Reuses the firing_velocity_corr.py cleaning pipeline:
  - drop forward-fill zero-pad (x==0 & y==0),
  - drop Hampel-flagged flicker samples,
  - compute Savitzky-Golay-smoothed speed,
  - clip samples whose speed exceeds MAX_SPEED_CM_S,
and additionally drops DLC samples whose confidence (dlc_signal) is < 0.5.

Bins are 500 ms; speed is averaged arithmetically and the two angle channels
are averaged circularly. A bin is kept only if it has at least one valid DLC
sample for all three channels.

Output: <sortout_folder>/behavior_analysis/velocity_fr/heading_velocity_3d.png
"""
import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import pickle
import numpy as np
import matplotlib.pyplot as plt

from plot_trial_traces import (
    POSITION_UNITS_PER_CM,
    clean_position_p99,
    speed_to_cm_per_s,
)
from firing_velocity_corr import BIN_SEC, compute_vstim_mask


CONFIDENCE_THRESHOLD = 0.5


def _circular_mean_per_bin(angles, bin_idx, n_bins, in_range):
    """Per-bin circular mean (radians); NaN where no samples land in the bin."""
    sin_sum = np.zeros(n_bins)
    cos_sum = np.zeros(n_bins)
    counts  = np.zeros(n_bins, dtype=int)
    a = angles[in_range]
    b = bin_idx[in_range]
    np.add.at(sin_sum, b, np.sin(a))
    np.add.at(cos_sum, b, np.cos(a))
    np.add.at(counts,  b, 1)
    out = np.full(n_bins, np.nan)
    has = counts > 0
    out[has] = np.arctan2(sin_sum[has], cos_sum[has])
    return out, has


def bin_kinematics(t, v, head, hangle, window_duration_sec, bin_sec=BIN_SEC):
    """
    Per-bin arrays at bin_sec resolution:
      bin_speed  : mean speed (position-units/s); NaN if empty
      bin_head   : circular-mean heading (rad); NaN if empty
      bin_hangle : circular-mean head_angle (rad); NaN if empty
      centers    : bin-center times (window-relative seconds)
    """
    n_bins = int(np.floor(window_duration_sec / bin_sec))
    edges = np.arange(n_bins + 1) * bin_sec
    centers = edges[:-1] + 0.5 * bin_sec

    bin_idx = np.floor(t / bin_sec).astype(int)
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)

    speed_sum = np.zeros(n_bins)
    speed_n   = np.zeros(n_bins, dtype=int)
    np.add.at(speed_sum, bin_idx[in_range], v[in_range])
    np.add.at(speed_n,   bin_idx[in_range], 1)
    bin_speed = np.full(n_bins, np.nan)
    has_speed = speed_n > 0
    bin_speed[has_speed] = speed_sum[has_speed] / speed_n[has_speed]

    bin_head,   _ = _circular_mean_per_bin(head,   bin_idx, n_bins, in_range)
    bin_hangle, _ = _circular_mean_per_bin(hangle, bin_idx, n_bins, in_range)

    return bin_speed, bin_head, bin_hangle, centers


def plot_3d_and_projections(speed, heading, hangle, vstim_mask, save_path):
    """
    Top-left: 3D scatter of (speed, heading, head_angle) colored by vstim state.
    Top-right + bottom row: the three 2D projections.
    """
    on  = vstim_mask
    off = ~vstim_mask
    speed_unit = 'cm/s' if POSITION_UNITS_PER_CM is not None else 'position-units/s'

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.scatter(speed[off], np.degrees(heading[off]), np.degrees(hangle[off]),
                 s=6, alpha=0.35, color='tab:blue',
                 label=f'vstim-off (n={int(off.sum())})', linewidths=0)
    ax3d.scatter(speed[on], np.degrees(heading[on]), np.degrees(hangle[on]),
                 s=6, alpha=0.45, color='tab:red',
                 label=f'vstim-on (n={int(on.sum())})', linewidths=0)
    ax3d.set_xlabel(f'Speed ({speed_unit})', fontsize=9)
    ax3d.set_ylabel('Heading (deg)', fontsize=9)
    ax3d.set_zlabel('Head angle (deg)', fontsize=9)
    ax3d.legend(fontsize=8, loc='upper left')
    ax3d.set_title('3D: speed × heading × head angle', fontsize=10)

    panels = [
        (gs[0, 1], speed,                np.degrees(heading), f'Speed ({speed_unit})', 'Heading (deg)'),
        (gs[1, 0], speed,                np.degrees(hangle),  f'Speed ({speed_unit})', 'Head angle (deg)'),
        (gs[1, 1], np.degrees(heading),  np.degrees(hangle),  'Heading (deg)',          'Head angle (deg)'),
    ]
    for spec, xv, yv, xl, yl in panels:
        ax = fig.add_subplot(spec)
        ax.scatter(xv[off], yv[off], s=6, alpha=0.35, color='tab:blue',
                   linewidths=0, label='vstim-off')
        ax.scatter(xv[on], yv[on], s=6, alpha=0.45, color='tab:red',
                   linewidths=0, label='vstim-on')
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'Per-bin kinematics  '
                 f'(bin={int(BIN_SEC * 1000)} ms, '
                 f'confidence ≥ {CONFIDENCE_THRESHOLD}, '
                 f'n_bins={int(off.sum() + on.sum())})',
                 fontsize=12)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    from DiscriminationTask.grating.task_params import sortout_folder

    sortout = Path(sortout_folder)
    pkl_file = sortout / f'task_spikes_{sortout.name}.pkl'
    if not pkl_file.exists():
        raise FileNotFoundError(
            f"Expected pkl not found: {pkl_file}\n"
            f"Run extract_session_spikes.py first."
        )

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    sp = data.get('session_position')
    if sp is None:
        raise RuntimeError(
            "Pkl has no session_position. Re-run extract_session_spikes.py "
            "with task_file=... so the full session DLC track is saved."
        )

    win = data['window']
    trial_onsets  = np.asarray(win['trial_onsets_sec'],  dtype=float)
    trial_offsets = np.asarray(win['trial_offsets_sec'], dtype=float)
    window_duration_sec = float(win['window_duration_sec'])

    cleaned, stats = clean_position_p99({
        'x':          sp['position_x'],
        'y':          sp['position_y'],
        't':          sp['position_time_sec'],
        'heading':    sp.get('heading',    []),
        'head_angle': sp.get('head_angle', []),
        'dlc_signal': sp.get('dlc_signal', []),
    }, confidence_threshold=CONFIDENCE_THRESHOLD)
    print(f"Cleaned position: {stats['n_kept']} samples kept "
          f"(flicker dropped={stats['n_flicker_dropped']}, "
          f"speed-clip dropped={stats['n_speed_dropped']} "
          f"@ {stats['speed_threshold_cm_s']:.2f} cm/s, "
          f"low-confidence dropped={stats['n_confidence_dropped']}).")
    t      = cleaned['t']
    v      = cleaned['v']
    head   = cleaned['heading']
    hangle = cleaned['head_angle']

    bin_speed, bin_head, bin_hangle, centers = bin_kinematics(
        t, v, head, hangle, window_duration_sec
    )

    keep = (np.isfinite(bin_speed) & np.isfinite(bin_head)
            & np.isfinite(bin_hangle))
    bin_speed_cm = speed_to_cm_per_s(bin_speed[keep])
    bin_head     = bin_head[keep]
    bin_hangle   = bin_hangle[keep]
    centers      = centers[keep]
    vstim_mask   = compute_vstim_mask(centers, trial_onsets, trial_offsets)

    print(f"Bins kept: {centers.size}  "
          f"(vstim-on={int(vstim_mask.sum())}, "
          f"vstim-off={int((~vstim_mask).sum())})")

    out_dir = sortout / 'behavior_analysis' / 'velocity_fr'
    plot_3d_and_projections(bin_speed_cm, bin_head, bin_hangle, vstim_mask,
                            out_dir / 'heading_velocity_3d.png')
