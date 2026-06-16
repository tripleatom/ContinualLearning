"""
Firing-rate vs heading and head-angle correlation, bin = 500 ms.

Mirrors firing_velocity_corr.py but for circular covariates from the session
DLC track:
  - dlcHeading   : degrees in [0, 360)
  - dlcHeadAngle : radians in [-pi, pi]

Per-bin covariate aggregation uses circular mean (atan2 of mean cos/sin) so
wrap-around does not bias the average. Per-unit r is the circular-linear
correlation magnitude (Mardia/Zar); population decode is a ridge fit on
(cos, sin) recombined via arctan2.

Three variants overlaid in the same figures:
  - vstim_off (blue)  : bins not overlapping any (trial_onset, trial_offset)
  - vstim_on  (red)   : bins overlapping a vstim-on window
  - all       (black) : every kept bin (no vstim filtering)

Outputs (under <sortout_folder>/behavior_analysis/heading_fr/):
  - firing_vs_heading_per_unit.png
  - population_heading_decode.png
  - firing_vs_head_angle_per_unit.png
  - population_head_angle_decode.png
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

from plot_trial_traces import clean_position_p99
from firing_velocity_corr import (
    BIN_SEC,
    VARIANT_COLORS,
    VARIANT_LABELS,
    compute_vstim_mask,
)


def bin_firing_and_circular(spike_data, cov_t, cov_v, window_duration_sec,
                            radian=True, bin_sec=BIN_SEC):
    """
    Build full-session arrays at bin_sec resolution:
      fr        : (n_bins, n_units) firing rate (Hz)
      bin_cov   : (n_bins,) circular-mean of covariate per bin (in input units),
                  NaN where no DLC samples
      has_cov   : (n_bins,) True where at least one DLC sample landed
      unit_ids  : list of unit labels (column order of fr)
      centers   : (n_bins,) bin-center times (window-relative seconds)
    """
    n_bins = int(np.floor(window_duration_sec / bin_sec))
    edges = np.arange(n_bins + 1) * bin_sec
    centers = edges[:-1] + 0.5 * bin_sec

    cov_t = np.asarray(cov_t, dtype=float)
    cov_v = np.asarray(cov_v, dtype=float)
    finite = np.isfinite(cov_v) & np.isfinite(cov_t)
    cov_t = cov_t[finite]
    cov_v = cov_v[finite]

    theta = cov_v if radian else np.deg2rad(cov_v)
    bin_idx = np.floor(cov_t / bin_sec).astype(int)
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)

    cos_sum = np.zeros(n_bins)
    sin_sum = np.zeros(n_bins)
    n_per   = np.zeros(n_bins, dtype=int)
    np.add.at(cos_sum, bin_idx[in_range], np.cos(theta[in_range]))
    np.add.at(sin_sum, bin_idx[in_range], np.sin(theta[in_range]))
    np.add.at(n_per,   bin_idx[in_range], 1)
    has_cov = n_per > 0
    bin_cov = np.full(n_bins, np.nan)
    mean_rad = np.arctan2(sin_sum[has_cov], cos_sum[has_cov])  # in [-pi, pi]
    if radian:
        bin_cov[has_cov] = mean_rad
    else:
        deg = np.rad2deg(mean_rad) % 360.0   # back to [0, 360)
        bin_cov[has_cov] = deg

    unit_ids = list(spike_data.keys())
    n_units = len(unit_ids)
    fr = np.zeros((n_bins, n_units))
    for j, uid in enumerate(unit_ids):
        st = np.asarray(spike_data[uid]['spike_times_sec'], dtype=float)
        st = st[(st >= 0) & (st < n_bins * bin_sec)]
        if st.size:
            counts, _ = np.histogram(st, bins=edges)
            fr[:, j] = counts / bin_sec

    return fr, bin_cov, has_cov, unit_ids, centers


def select_bins(fr, bin_cov, centers, vstim_mask, has_cov, mode):
    if mode == 'off':
        keep = (~vstim_mask) & has_cov
    elif mode == 'on':
        keep = vstim_mask & has_cov
    elif mode == 'all':
        keep = has_cov
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return fr[keep], bin_cov[keep], centers[keep], int(keep.sum())


def _circ_lin_r(theta_rad, y):
    """Circular-linear correlation magnitude (Mardia/Zar). Returns r in [0, 1]."""
    if y.size < 3 or np.std(y) == 0:
        return float('nan')
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    if np.std(c) == 0 or np.std(s) == 0:
        return float('nan')
    rc,  _ = pearsonr(c, y)
    rs,  _ = pearsonr(s, y)
    rcs, _ = pearsonr(c, s)
    denom = 1.0 - rcs * rcs
    if denom <= 0:
        return float('nan')
    val = (rc * rc + rs * rs - 2 * rc * rs * rcs) / denom
    return float(np.sqrt(max(0.0, min(1.0, val))))


def _circ_tuning(theta_rad, y, n_bins=24):
    """Circular tuning curve: mean firing rate per angular bin (centers in [-pi, pi])."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    idx = np.digitize(theta_rad, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            means[b] = float(np.mean(y[sel]))
    return centers, means


def plot_per_unit_grid(per_variant, unit_ids, save_path, cov_label, cov_unit, radian):
    n_units = len(unit_ids)
    if n_units == 0:
        print("No units to plot.")
        return

    angles = [a for (_, a) in per_variant.values() if a.size]
    if not angles:
        print("No covariate samples in any variant; skipping per-unit grid.")
        return

    if radian:
        x_min, x_max = -np.pi, np.pi
    else:
        x_min, x_max = 0.0, 360.0
    x_curve_rad = np.linspace(-np.pi, np.pi, 24, endpoint=False) + (np.pi / 24)
    x_curve_disp = x_curve_rad if radian else (np.rad2deg(x_curve_rad) % 360.0)
    sort_idx = np.argsort(x_curve_disp)
    x_curve_disp_sorted = x_curve_disp[sort_idx]

    ncols = int(math.ceil(math.sqrt(n_units)))
    nrows = int(math.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.1, nrows * 2.1),
                             squeeze=False, sharex=True)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n_units:
            ax.axis('off')
            continue

        rs = {}
        for mode in ('off', 'on', 'all'):
            fr_v, ang_v = per_variant.get(mode, (np.empty((0, 0)), np.empty(0)))
            if ang_v.size == 0 or fr_v.shape[1] == 0:
                rs[mode] = float('nan')
                continue
            y = fr_v[:, idx]
            color = VARIANT_COLORS[mode]
            ax.scatter(ang_v, y, s=2, alpha=0.25,
                       color=color, linewidths=0, zorder=1)
            theta_rad = ang_v if radian else np.deg2rad(ang_v)
            rs[mode] = _circ_lin_r(theta_rad, y)
            # tuning curve overlay
            if y.size >= 12:
                _, means = _circ_tuning(theta_rad, y, n_bins=24)
                disp = means[sort_idx]
                if np.any(np.isfinite(disp)):
                    ax.plot(x_curve_disp_sorted, disp,
                            color=color, linewidth=0.9, zorder=3)

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
        ax.set_xlim(x_min, x_max)
        ax.tick_params(labelsize=5)

    n_bins_str = ', '.join(
        f"{VARIANT_LABELS[m]}={per_variant[m][0].shape[0] if m in per_variant else 0}"
        for m in ('off', 'on', 'all')
    )
    fig.suptitle(f"Firing rate vs. {cov_label} per unit  "
                 f"(bin={int(BIN_SEC * 1000)} ms, n_units={n_units}; "
                 f"n_bins {n_bins_str})",
                 fontsize=11)
    fig.text(0.5, 0.02, f'{cov_label} ({cov_unit})', ha='center', fontsize=10)
    fig.text(0.015, 0.5, 'Firing rate (Hz)', va='center',
             rotation=90, fontsize=10)
    fig.tight_layout(rect=[0.035, 0.035, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def _ridge_circ_decode(fr, theta_rad, n_splits=5, alpha=1.0):
    """K-fold ridge decode of (cos, sin), recombine via arctan2.
    Returns (pred_theta, circ_lin_r, mean_abs_err_rad) or None."""
    if fr.shape[1] == 0 or fr.shape[0] < 2 * n_splits:
        return None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    pred_cos = np.full_like(theta_rad, np.nan, dtype=float)
    pred_sin = np.full_like(theta_rad, np.nan, dtype=float)
    for tr_idx, te_idx in kf.split(fr):
        Y = np.column_stack([np.cos(theta_rad[tr_idx]), np.sin(theta_rad[tr_idx])])
        m = Ridge(alpha=alpha)
        m.fit(fr[tr_idx], Y)
        P = m.predict(fr[te_idx])
        pred_cos[te_idx] = P[:, 0]
        pred_sin[te_idx] = P[:, 1]
    pred_theta = np.arctan2(pred_sin, pred_cos)
    err = np.arctan2(np.sin(pred_theta - theta_rad),
                     np.cos(pred_theta - theta_rad))
    mae = float(np.mean(np.abs(err)))
    # circular-linear style summary: average of cos/sin Pearson r
    rc, _ = pearsonr(np.cos(pred_theta), np.cos(theta_rad))
    rs, _ = pearsonr(np.sin(pred_theta), np.sin(theta_rad))
    return pred_theta, float(0.5 * (rc + rs)), mae


def plot_population_decode_grid(per_variant, save_path, cov_label, cov_unit,
                                radian, n_splits=5, alpha=1.0):
    modes = ('off', 'on', 'all')
    results = {}
    for m in modes:
        fr_v, ang_v = per_variant.get(m, (np.empty((0, 0)), np.empty(0)))
        if ang_v.size == 0:
            results[m] = None
            continue
        theta_rad = ang_v if radian else np.deg2rad(ang_v)
        results[m] = _ridge_circ_decode(fr_v, theta_rad, n_splits=n_splits, alpha=alpha)

    if all(r is None for r in results.values()):
        print("Not enough data for population decoding in any variant; skipping.")
        return

    if radian:
        lo, hi = -np.pi, np.pi
    else:
        lo, hi = 0.0, 360.0

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
        pred_theta, rmean, mae = results[m]
        ang_v = per_variant[m][1]
        pred_disp = pred_theta if radian else (np.rad2deg(pred_theta) % 360.0)
        n_units = per_variant[m][0].shape[1]
        n_bins = per_variant[m][0].shape[0]
        ax.scatter(ang_v, pred_disp, s=4, alpha=0.3, color=color, linewidths=0)
        ax.plot([lo, hi], [lo, hi], '--', color='0.3', linewidth=0.8)
        ax.set_xlabel(f'Actual {cov_label} ({cov_unit})')
        ax.set_ylabel(f'Predicted {cov_label} ({cov_unit})')
        ax.set_title(f"{label}  (n_units={n_units}, n_bins={n_bins})\n"
                     f"r̄={rmean:.3f}, circ MAE={np.rad2deg(mae):.1f}°",
                     color=color)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig.suptitle(f"Population firing → {cov_label}  "
                 f"(Ridge α={alpha}, {n_splits}-fold CV on cos/sin, "
                 f"bin={int(BIN_SEC * 1000)} ms)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_one_covariate(spike_data, pos_t, cov_v, window_duration_sec,
                      trial_onsets, trial_offsets, out_dir,
                      cov_name, cov_label, cov_unit, radian):
    """Run the full per-unit + population-decode pipeline for one covariate."""
    fr_full, bin_cov_full, has_cov, unit_ids, centers = bin_firing_and_circular(
        spike_data, pos_t, cov_v, window_duration_sec, radian=radian
    )
    vstim_mask = compute_vstim_mask(centers, trial_onsets, trial_offsets)

    per_variant = {}
    for mode in ('off', 'on', 'all'):
        fr, bc, _, n_kept = select_bins(
            fr_full, bin_cov_full, centers, vstim_mask, has_cov, mode
        )
        if n_kept == 0:
            print(f"[{cov_name}/{VARIANT_LABELS[mode]}] no bins match — skipping.")
            per_variant[mode] = (np.empty((0, fr_full.shape[1])), np.empty(0))
            continue
        per_variant[mode] = (fr, bc)
        print(f"[{cov_name}/{VARIANT_LABELS[mode]}] {fr.shape[0]} bins x {fr.shape[1]} units.")

    plot_per_unit_grid(per_variant, unit_ids,
                       out_dir / f'firing_vs_{cov_name}_per_unit.png',
                       cov_label=cov_label, cov_unit=cov_unit, radian=radian)
    plot_population_decode_grid(per_variant,
                                out_dir / f'population_{cov_name}_decode.png',
                                cov_label=cov_label, cov_unit=cov_unit, radian=radian)


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
    if 'heading' not in sp or 'head_angle' not in sp:
        raise RuntimeError(
            "Pkl's session_position has no heading/head_angle. Re-run "
            "extract_session_spikes.py after the readDIO_grating.py update "
            "that exposes dlcHeading/dlcHeadAngle."
        )

    win = data['window']
    spike_data = data['spike_data']
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
    })
    print(f"Cleaned position: {stats['n_kept']} samples kept "
          f"(flicker dropped={stats['n_flicker_dropped']}, "
          f"speed-clip dropped={stats['n_speed_dropped']} "
          f"@ {stats['speed_threshold_cm_s']:.2f} cm/s).")
    pt = cleaned['t']
    heading_deg    = cleaned['heading']
    head_angle_rad = cleaned['head_angle']

    out_dir = sortout / 'behavior_analysis' / 'heading_fr'

    run_one_covariate(
        spike_data, pt, heading_deg, window_duration_sec,
        trial_onsets, trial_offsets, out_dir,
        cov_name='heading', cov_label='heading', cov_unit='deg',
        radian=False,
    )
    run_one_covariate(
        spike_data, pt, head_angle_rad, window_duration_sec,
        trial_onsets, trial_offsets, out_dir,
        cov_name='head_angle', cov_label='head angle', cov_unit='rad',
        radian=True,
    )
