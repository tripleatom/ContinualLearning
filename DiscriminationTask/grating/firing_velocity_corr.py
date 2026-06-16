"""
Firing-rate vs running-speed correlation, bin = 500 ms.

Inputs: <sortout_folder>/task_spikes_<session>.pkl produced by
extract_session_spikes.py (continuous spike trains + session-wide DLC track,
all in window-relative seconds).

Three variants overlaid in the same figures:
  - vstim_off (blue)  : bins not overlapping any (trial_onset, trial_offset) window
  - vstim_on  (red)   : bins overlapping a vstim-on window
  - all       (black) : every kept bin (no vstim filtering)

Velocity is computed exactly the same way as plot_trial_traces.py: zero-pad
drop, Hampel flicker filter, Savitzky-Golay smoothed central differences,
MAX_SPEED_CM_S clip.

Outputs (under <sortout_folder>/behavior_analysis/velocity_fr/):
  - firing_vs_speed_per_unit.png   grid, each subplot overlays the three
                                   variants for one unit; per-unit Pearson r
                                   per variant shown above the subplot
  - population_speed_decode.png    1x3 grid, one panel per variant
"""
import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from plot_trial_traces import (
    POSITION_UNITS_PER_CM,
    clean_position_p99,
    speed_to_cm_per_s,
)


BIN_SEC = 0.5


def bin_firing_and_speed(spike_data, pos_t, pos_v,
                         window_duration_sec, bin_sec=BIN_SEC):
    """
    Build full-session arrays at bin_sec resolution:
      fr         : (n_bins, n_units) firing rate (Hz)
      bin_speed  : (n_bins,) mean speed (position-units/s); NaN where no DLC samples
      vstim_mask : (n_bins,) True for bins overlapping any (trial_onset, trial_offset)
      has_speed  : (n_bins,) True for bins that have at least one DLC sample
      unit_ids   : list of unit labels (column order of fr)
      centers    : (n_bins,) bin-center times (window-relative seconds)
    """
    n_bins = int(np.floor(window_duration_sec / bin_sec))
    edges = np.arange(n_bins + 1) * bin_sec
    centers = edges[:-1] + 0.5 * bin_sec

    # mean speed per bin
    bin_idx = np.floor(pos_t / bin_sec).astype(int)
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    speed_sum = np.zeros(n_bins)
    speed_n   = np.zeros(n_bins, dtype=int)
    np.add.at(speed_sum, bin_idx[in_range], pos_v[in_range])
    np.add.at(speed_n,   bin_idx[in_range], 1)
    has_speed = speed_n > 0
    bin_speed = np.full(n_bins, np.nan)
    bin_speed[has_speed] = speed_sum[has_speed] / speed_n[has_speed]

    # firing rate per bin per unit
    unit_ids = list(spike_data.keys())
    n_units = len(unit_ids)
    fr = np.zeros((n_bins, n_units))
    for j, uid in enumerate(unit_ids):
        st = np.asarray(spike_data[uid]['spike_times_sec'], dtype=float)
        st = st[(st >= 0) & (st < n_bins * bin_sec)]
        if st.size:
            counts, _ = np.histogram(st, bins=edges)
            fr[:, j] = counts / bin_sec

    return fr, bin_speed, unit_ids, centers


def select_bins(fr, bin_speed, centers, vstim_mask, has_speed, mode):
    """Apply a mode-specific bin filter on top of the always-required has_speed mask."""
    if mode == 'off':
        keep = (~vstim_mask) & has_speed
    elif mode == 'on':
        keep = vstim_mask & has_speed
    elif mode == 'all':
        keep = has_speed
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return fr[keep], bin_speed[keep], centers[keep], int(keep.sum())


def compute_vstim_mask(centers, trial_onsets, trial_offsets, bin_sec=BIN_SEC):
    edges_lo = centers - 0.5 * bin_sec
    edges_hi = centers + 0.5 * bin_sec
    mask = np.zeros(centers.size, dtype=bool)
    for ts, te in zip(trial_onsets, trial_offsets):
        mask |= (edges_hi > ts) & (edges_lo < te)
    return mask


VARIANT_COLORS = {
    'off': 'tab:blue',
    'on':  'tab:red',
    'all': 'black',
}
VARIANT_LABELS = {
    'off': 'vstim-off',
    'on':  'vstim-on',
    'all': 'all',
}


def _fit_one(speed_cm, y):
    """Pearson r and least-squares (slope, intercept) for one (speed, fr) series."""
    if y.size < 3 or np.std(y) == 0 or np.std(speed_cm) == 0:
        mean = float(np.mean(y)) if y.size else 0.0
        return float('nan'), 0.0, mean
    r, _ = pearsonr(speed_cm, y)
    slope, intercept = np.polyfit(speed_cm, y, 1)
    return r, slope, intercept


def plot_per_unit_grid_combined(per_variant, unit_ids, save_path):
    """
    per_variant: dict {mode: (fr_array, speed_cm_array)} for modes in
                 {'off', 'on', 'all'}; arrays may be empty (skipped).

    One subplot per unit, all three variants overlaid as colored scatters with
    matching fitted lines. Above each subplot: unit id and three colored r values.
    """
    n_units = len(unit_ids)
    if n_units == 0:
        print("No units to plot.")
        return

    # joint axis range (cm/s) over all variants present
    speeds = [s for (_, s) in per_variant.values() if s.size]
    if not speeds:
        print("No speed samples in any variant; skipping per-unit grid.")
        return
    s_min = float(min(s.min() for s in speeds))
    s_max = float(max(s.max() for s in speeds))
    xs = np.array([s_min, s_max])

    # joint firing-rate range across all variants and all units
    fr_arrays = [fr_v for (fr_v, sp_v) in per_variant.values()
                 if sp_v.size and fr_v.size]
    if fr_arrays:
        fr_min = float(min(a.min() for a in fr_arrays))
        fr_max = float(max(a.max() for a in fr_arrays))
        if fr_max == fr_min:
            fr_max = fr_min + 1.0
        pad = 0.03 * (fr_max - fr_min)
        y_lo, y_hi = fr_min - pad, fr_max + pad
    else:
        y_lo, y_hi = 0.0, 1.0

    pad_x = 0.03 * (s_max - s_min) if s_max > s_min else 1.0
    x_lo, x_hi = s_min - pad_x, s_max + pad_x

    ncols = int(math.ceil(math.sqrt(n_units)))
    nrows = int(math.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.1, nrows * 2.1),
                             squeeze=False, sharex=True, sharey=True)
    axes[0][0].set_xlim(x_lo, x_hi)
    axes[0][0].set_ylim(y_lo, y_hi)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n_units:
            ax.axis('off')
            continue

        rs = {}
        for mode in ('off', 'on', 'all'):
            fr_v, sp_v = per_variant.get(mode, (np.empty((0, 0)), np.empty(0)))
            if sp_v.size == 0 or fr_v.shape[1] == 0:
                rs[mode] = float('nan')
                continue
            y = fr_v[:, idx]
            color = VARIANT_COLORS[mode]
            ax.scatter(sp_v, y, s=2, alpha=0.25,
                       color=color, linewidths=0, zorder=1)
            r, slope, intercept = _fit_one(sp_v, y)
            ax.plot(xs, slope * xs + intercept,
                    color=color, linewidth=0.9, zorder=3)
            rs[mode] = r

        # unit id (centered, black) + colored r values below it
        ax.text(0.5, 1.13, unit_ids[idx],
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=5.5, color='black')
        for x_anchor, ha, mode in (
            (0.02, 'left',   'off'),
            (0.50, 'center', 'on'),
            (0.98, 'right',  'all'),
        ):
            r_val = rs[mode]
            txt = f"{mode}:{r_val:.2f}" if np.isfinite(r_val) else f"{mode}:nan"
            ax.text(x_anchor, 1.02, txt,
                    transform=ax.transAxes, ha=ha, va='bottom',
                    fontsize=5, color=VARIANT_COLORS[mode])
        ax.tick_params(labelsize=5)

    n_bins_str = ', '.join(
        f"{VARIANT_LABELS[m]}={per_variant[m][0].shape[0] if m in per_variant else 0}"
        for m in ('off', 'on', 'all')
    )
    fig.suptitle(f"Firing rate vs. speed per unit  "
                 f"(bin={int(BIN_SEC * 1000)} ms, n_units={n_units}; "
                 f"n_bins {n_bins_str})",
                 fontsize=11)
    fig.text(0.5, 0.02, 'Speed (cm/s)' if POSITION_UNITS_PER_CM is not None
                        else 'Speed (position-units/s)',
             ha='center', fontsize=10)
    fig.text(0.015, 0.5, 'Firing rate (Hz)', va='center',
             rotation=90, fontsize=10)
    fig.tight_layout(rect=[0.035, 0.035, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def _ridge_cv_decode(fr, speed_cm, n_splits=5, alpha=1.0):
    """Run K-fold ridge decoding; return (preds, r, mean_r2) or None if insufficient data."""
    if fr.shape[1] == 0 or fr.shape[0] < 2 * n_splits:
        return None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = np.full_like(speed_cm, np.nan, dtype=float)
    r2_per_fold = []
    for tr_idx, te_idx in kf.split(fr):
        model = Ridge(alpha=alpha)
        model.fit(fr[tr_idx], speed_cm[tr_idx])
        p = model.predict(fr[te_idx])
        preds[te_idx] = p
        ss_res = np.sum((speed_cm[te_idx] - p) ** 2)
        ss_tot = np.sum((speed_cm[te_idx] - speed_cm[te_idx].mean()) ** 2)
        r2_per_fold.append(1 - ss_res / ss_tot if ss_tot > 0 else float('nan'))
    if np.std(speed_cm) > 0 and np.std(preds) > 0:
        r, _ = pearsonr(speed_cm, preds)
    else:
        r = float('nan')
    return preds, r, float(np.nanmean(r2_per_fold))


def plot_population_decode_grid(per_variant, save_path, n_splits=5, alpha=1.0):
    """
    1x3 grid: one panel per variant, with the variant's color used for
    points + frame. Shared axis range across panels for easy comparison.
    """
    modes = ('off', 'on', 'all')
    results = {}
    for m in modes:
        fr_v, sp_v = per_variant.get(m, (np.empty((0, 0)), np.empty(0)))
        if sp_v.size == 0:
            results[m] = None
            continue
        results[m] = _ridge_cv_decode(fr_v, sp_v, n_splits=n_splits, alpha=alpha)

    if all(r is None for r in results.values()):
        print("Not enough data for population decoding in any variant; skipping.")
        return

    # joint range across all variants
    los, his = [], []
    for m in modes:
        if results[m] is None:
            continue
        preds, _, _ = results[m]
        sp_v = per_variant[m][1]
        los.append(float(min(sp_v.min(), np.nanmin(preds))))
        his.append(float(max(sp_v.max(), np.nanmax(preds))))
    lo = min(los)
    hi = max(his)

    unit = 'cm/s' if POSITION_UNITS_PER_CM is not None else 'position-units/s'
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)

    for ax, m in zip(axes, modes):
        color = VARIANT_COLORS[m]
        label = VARIANT_LABELS[m]
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)
        if results[m] is None:
            ax.set_title(f"{label}: insufficient data")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect('equal', adjustable='box')
            continue
        preds, r, r2_mean = results[m]
        sp_v = per_variant[m][1]
        n_units = per_variant[m][0].shape[1]
        n_bins = per_variant[m][0].shape[0]
        ax.scatter(sp_v, preds, s=4, alpha=0.3, color=color, linewidths=0)
        ax.plot([lo, hi], [lo, hi], '--', color='0.3', linewidth=0.8)
        ax.set_xlabel(f'Actual speed ({unit})')
        ax.set_ylabel(f'Predicted speed ({unit})')
        ax.set_title(f"{label}  (n_units={n_units}, n_bins={n_bins})\n"
                     f"r={r:.3f}, mean R²={r2_mean:.3f}",
                     color=color)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig.suptitle(f"Population firing → speed  "
                 f"(Ridge α={alpha}, {n_splits}-fold CV, "
                 f"bin={int(BIN_SEC * 1000)} ms)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
    spike_data = data['spike_data']
    trial_onsets  = np.asarray(win['trial_onsets_sec'],  dtype=float)
    trial_offsets = np.asarray(win['trial_offsets_sec'], dtype=float)
    window_duration_sec = float(win['window_duration_sec'])

    cleaned, stats = clean_position_p99({
        'x': sp['position_x'],
        'y': sp['position_y'],
        't': sp['position_time_sec'],
    })
    print(f"Cleaned position: {stats['n_kept']} samples kept "
          f"(flicker dropped={stats['n_flicker_dropped']}, "
          f"speed-clip dropped={stats['n_speed_dropped']} "
          f"@ {stats['speed_threshold_cm_s']:.2f} cm/s).")
    pt, pv = cleaned['t'], cleaned['v']
    fr_full, bin_speed_full, unit_ids, centers = bin_firing_and_speed(
        spike_data, pt, pv, window_duration_sec
    )
    has_speed = ~np.isnan(bin_speed_full)
    vstim_mask = compute_vstim_mask(centers, trial_onsets, trial_offsets)

    out_dir = sortout / 'behavior_analysis' / 'velocity_fr'
    per_variant = {}
    for mode in ('off', 'on', 'all'):
        fr, bs, _, n_kept = select_bins(
            fr_full, bin_speed_full, centers, vstim_mask, has_speed, mode
        )
        if n_kept == 0:
            print(f"[{VARIANT_LABELS[mode]}] no bins match — skipping.")
            per_variant[mode] = (np.empty((0, fr_full.shape[1])), np.empty(0))
            continue
        per_variant[mode] = (fr, speed_to_cm_per_s(bs))
        print(f"[{VARIANT_LABELS[mode]}] {fr.shape[0]} bins x {fr.shape[1]} units.")

    plot_per_unit_grid_combined(per_variant, unit_ids,
                                out_dir / 'firing_vs_speed_per_unit.png')
    plot_population_decode_grid(per_variant,
                                out_dir / 'population_speed_decode.png')
