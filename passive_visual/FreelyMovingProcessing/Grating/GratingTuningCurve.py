"""
Plot Orientation Tuning Curves for All Neurons

This script loads neural data and generates individual tuning curve plots
for each unit, saved to a dedicated folder.

Alongside the curve itself every unit is characterized by:
  * visual responsiveness   — R_evoked - R_baseline (baseline = pre-onset ITI),
                              paired Wilcoxon test overall and per orientation
  * split-half reliability  — median correlation between tuning curves built
                              from two random halves of the trials
  * bootstrap of the curve  — trials resampled within orientation, giving
                              R(theta) +/- 95% CI and CIs on OSI / preferred ori

All of these are printed on each unit's tuning-curve figure and written to
tuning_statistics.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

from grating_utils import format_grating_value, format_grating_values, resolve_data_path


# =============================================================================
# DATA LOADING (reusing from main script)
# =============================================================================

def load_neural_data(filepath):
    """Load neural data from pickle format."""
    filepath = Path(filepath)

    if filepath.suffix != '.pkl':
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Only .pkl files are supported.")

    print(f"Loading data from {filepath.suffix}: {filepath.name}")
    return _load_pickle(filepath)


def _load_pickle(filepath):
    """Load from pickle format."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# RESPONSIVENESS / RELIABILITY CHARACTERIZATION
# =============================================================================
#
# Three complementary characterizations are computed per unit:
#
#   1. Visual responsiveness — R_evoked - R_baseline, where R_baseline comes
#      from the pre-onset part of the inter-trial interval (paired with each
#      trial).  Tested with a paired Wilcoxon signed-rank test, over all trials
#      pooled and separately per orientation (Holm-Bonferroni corrected).
#   2. Split-half tuning-curve reliability — trials of each orientation are
#      randomly split in two, a tuning curve is built from each half, and the
#      two curves are correlated; repeated many times, reported as the median r.
#      Significance comes from an orientation-label shuffle null.
#   3. Bootstrap of the whole tuning curve — trials are resampled with
#      replacement within each orientation, giving R(theta) +/- 95% CI at every
#      orientation plus CIs on OSI and preferred orientation.
#
# All three read the same per-trial firing rates used for the tuning curve, so
# the numbers printed on the figure always describe the curve that is drawn.

DEFAULT_BASELINE_WINDOW = (-0.2, 0.0)   # ITI tail preceding stimulus onset
N_SPLITS = 1000       # split-half repetitions
N_BOOT = 1000         # bootstrap resamples
N_SHUFFLES = 1000     # orientation-label shuffles for the null distributions
ALPHA = 0.05


def _rate_in_window(spike_times, window):
    """Firing rate (Hz) of one trial inside [start, end)."""
    start, end = window
    spike_times = np.asarray(spike_times, dtype=float)
    return float(np.sum((spike_times >= start) & (spike_times < end)) / (end - start))


def _rowwise_pearson(A, B):
    """Pearson r between matching rows of two (n, k) arrays -> (n,) array."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = np.sum(A * B, axis=1)
    den = np.sqrt(np.sum(A ** 2, axis=1) * np.sum(B ** 2, axis=1))
    out = np.full(num.shape, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _osi_from_curves(curves, orientations):
    """Vector-sum OSI and preferred orientation for each row of (n, n_ori) curves."""
    theta = 2 * np.deg2rad(np.asarray(orientations, dtype=float))
    z = np.asarray(curves, dtype=float) @ np.exp(1j * theta)
    total = np.asarray(curves, dtype=float).sum(axis=1)
    osi = np.abs(z) / (total + 1e-12)
    pref = np.rad2deg((np.angle(z) / 2.0) % np.pi)
    return osi, pref


def _holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values (NaNs passed through)."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return adj
    m = finite.size
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _wilcoxon_p(diff):
    """Two-sided paired Wilcoxon signed-rank p for a difference vector."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 5 or np.all(diff == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(diff, zero_method='wilcox',
                                    alternative='two-sided').pvalue)
    except ValueError:
        return np.nan


def _circular_axial_stats(pref_deg, reference_deg):
    """
    Spread of a bootstrap distribution of preferred orientations (axial, period 180 deg).

    Returns (circular_sd_deg, ci_low_deg, ci_high_deg); the CI is a percentile
    interval on the deviation from `reference_deg`, wrapped to +/-90 deg and
    added back, so it stays interpretable even near the 0/180 deg wrap.
    """
    pref = np.asarray(pref_deg, dtype=float)
    pref = pref[np.isfinite(pref)]
    if pref.size < 2:
        return np.nan, np.nan, np.nan

    ang = 2 * np.deg2rad(pref)
    resultant = np.abs(np.mean(np.exp(1j * ang)))
    circ_sd = np.rad2deg(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))) / 2.0

    dev = (pref - reference_deg + 90.0) % 180.0 - 90.0
    lo, hi = np.percentile(dev, [2.5, 97.5])
    return (float(circ_sd),
            float((reference_deg + lo) % 180.0),
            float((reference_deg + hi) % 180.0))


def compute_visual_responsiveness(evoked_by_ori, baseline_by_ori, orientations,
                                  alpha=ALPHA):
    """
    Test whether a unit responds to the gratings at all.

    Per trial the evoked rate (analysis window) is paired with that same trial's
    baseline rate (pre-onset ITI window).  A paired Wilcoxon signed-rank test is
    run on the pooled trials and on each orientation separately; the
    per-orientation p-values are Holm-Bonferroni corrected across orientations.
    A Kruskal-Wallis test across orientations additionally asks whether the
    evoked rate depends on orientation at all.

    Args:
        evoked_by_ori:   dict orientation -> list/array of per-trial evoked rates (Hz)
        baseline_by_ori: dict orientation -> per-trial baseline rates (Hz), trial-matched
        orientations:    ordered list of orientations
        alpha:           significance level

    Returns:
        dict of responsiveness statistics (see keys below).
    """
    ev_all, bl_all = [], []
    per_ori_delta, per_ori_p, per_ori_n = [], [], []

    for ori in orientations:
        ev = np.asarray(evoked_by_ori.get(ori, []), dtype=float)
        bl = np.asarray(baseline_by_ori.get(ori, []), dtype=float)
        n = min(ev.size, bl.size)
        ev, bl = ev[:n], bl[:n]
        ev_all.append(ev)
        bl_all.append(bl)
        per_ori_n.append(int(n))
        per_ori_delta.append(float(np.mean(ev - bl)) if n else np.nan)
        per_ori_p.append(_wilcoxon_p(ev - bl) if n else np.nan)

    ev_all = np.concatenate(ev_all) if ev_all else np.array([])
    bl_all = np.concatenate(bl_all) if bl_all else np.array([])
    diff = ev_all - bl_all

    p_overall = _wilcoxon_p(diff)
    delta = float(np.mean(diff)) if diff.size else np.nan
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan
    cohens_dz = delta / sd if (sd and np.isfinite(sd) and sd > 0) else np.nan

    per_ori_p = np.asarray(per_ori_p, dtype=float)
    per_ori_p_holm = _holm_bonferroni(per_ori_p)
    per_ori_delta = np.asarray(per_ori_delta, dtype=float)

    # Strongest response: the orientation with the largest evoked-baseline change.
    if np.any(np.isfinite(per_ori_delta)):
        best_idx = int(np.nanargmax(np.abs(per_ori_delta)))
        best_ori = float(orientations[best_idx])
        best_delta = float(per_ori_delta[best_idx])
        best_p_holm = float(per_ori_p_holm[best_idx])
    else:
        best_idx, best_ori, best_delta, best_p_holm = -1, np.nan, np.nan, np.nan

    sig_mask = np.isfinite(per_ori_p_holm) & (per_ori_p_holm < alpha)
    n_sig_ori = int(np.sum(sig_mask))

    # Orientation-dependence of the evoked rate (ignores baseline).
    groups = [np.asarray(evoked_by_ori.get(ori, []), dtype=float) for ori in orientations]
    groups = [g for g in groups if g.size > 1 and np.ptp(g) > 0]
    if len(groups) > 1:
        try:
            p_kruskal = float(stats.kruskal(*groups).pvalue)
        except ValueError:
            p_kruskal = np.nan
    else:
        p_kruskal = np.nan

    baseline_rate = float(np.mean(bl_all)) if bl_all.size else np.nan
    evoked_rate = float(np.mean(ev_all)) if ev_all.size else np.nan
    resp_index = ((evoked_rate - baseline_rate) / (evoked_rate + baseline_rate + 1e-12)
                  if np.isfinite(evoked_rate) else np.nan)

    # A unit counts as visually responsive if the evoked rate differs from the
    # ITI baseline in either direction — suppression is a response too.
    responsive = bool(
        (np.isfinite(p_overall) and p_overall < alpha)
        or n_sig_ori > 0
    )
    if not responsive or not np.isfinite(delta):
        sign = 'none'
    else:
        sign = 'enhanced' if delta > 0 else 'suppressed'

    return {
        'baseline_rate_hz': baseline_rate,
        'evoked_rate_hz': evoked_rate,
        'delta_rate_hz': delta,
        'response_index': float(resp_index),
        'cohens_dz': float(cohens_dz) if np.isfinite(cohens_dz) else np.nan,
        'p_wilcoxon': p_overall,
        'p_kruskal_orientation': p_kruskal,
        'per_ori_delta_hz': per_ori_delta.tolist(),
        'per_ori_p': per_ori_p.tolist(),
        'per_ori_p_holm': per_ori_p_holm.tolist(),
        'per_ori_n_trials': per_ori_n,
        'sig_orientations': [float(orientations[i]) for i in np.flatnonzero(sig_mask)],
        'n_sig_orientations': n_sig_ori,
        'best_orientation_deg': best_ori,
        'best_delta_hz': best_delta,
        'best_p_holm': best_p_holm,
        'responsive': responsive,
        'response_sign': sign,
        'alpha': alpha,
        'n_trials': int(diff.size),
    }


def compute_tuning_reliability(trial_rates, orientations, n_splits=N_SPLITS,
                               n_shuffles=N_SHUFFLES, rng=None, alpha=ALPHA):
    """
    Split-half reliability of the tuning curve, with an orientation-shuffle null.

    For each of `n_splits` repetitions the trials of every orientation are split
    into two random halves; the two resulting tuning curves R_A and R_B are
    correlated (Pearson), and the median r over repetitions is reported.  The
    null is built by shuffling orientation labels across trials and repeating a
    single split per shuffle, which also yields a null distribution of OSI.

    Returns dict with median/CI of r, the Spearman-Brown corrected value
    (2r/(1+r), the reliability expected for the full trial count), the permutation
    p-values for reliability and for OSI, and the null OSI distribution summary.
    """
    rng = np.random.default_rng() if rng is None else rng

    rates_by_ori = [np.asarray(trial_rates.get(ori, []), dtype=float)
                    for ori in orientations]
    counts = np.array([r.size for r in rates_by_ori])
    usable = counts >= 2

    result = {
        'n_splits': int(n_splits),
        'n_shuffles': int(n_shuffles),
        'median_r': np.nan,
        'r_ci_low': np.nan,
        'r_ci_high': np.nan,
        'r_spearman_brown': np.nan,
        'p_perm_reliability': np.nan,
        'p_perm_osi': np.nan,
        'null_r_median': np.nan,
        'null_osi_mean': np.nan,
        'null_osi_p95': np.nan,
        'reliable': False,
        'alpha': alpha,
    }

    if usable.sum() < 3:
        # Fewer than three orientations with >= 2 trials: a correlation across
        # orientations is not meaningful.
        return result

    ori_used = [orientations[i] for i in np.flatnonzero(usable)]
    rates_used = [rates_by_ori[i] for i in np.flatnonzero(usable)]
    n_ori = len(ori_used)

    # ---- observed split-half distribution ----
    A = np.empty((n_splits, n_ori))
    B = np.empty((n_splits, n_ori))
    for k, rates in enumerate(rates_used):
        n = rates.size
        half = n // 2
        order = np.argsort(rng.random((n_splits, n)), axis=1)
        drawn = rates[order]
        A[:, k] = drawn[:, :half].mean(axis=1)
        B[:, k] = drawn[:, half:2 * half].mean(axis=1)

    r_split = _rowwise_pearson(A, B)
    r_split = r_split[np.isfinite(r_split)]
    if r_split.size == 0:
        return result

    median_r = float(np.median(r_split))
    ci_low, ci_high = (float(v) for v in np.percentile(r_split, [2.5, 97.5]))
    # Spearman-Brown: each half holds half the trials, so the split-half r
    # underestimates the reliability of the curve built from all trials.
    sb = 2 * median_r / (1 + median_r) if median_r > -1 else np.nan

    # ---- orientation-label shuffle null (reliability and OSI together) ----
    all_rates = np.concatenate(rates_used)
    n_total = all_rates.size
    bounds = np.concatenate([[0], np.cumsum([r.size for r in rates_used])])

    shuffled = all_rates[np.argsort(rng.random((n_shuffles, n_total)), axis=1)]
    A0 = np.empty((n_shuffles, n_ori))
    B0 = np.empty((n_shuffles, n_ori))
    curve0 = np.empty((n_shuffles, n_ori))
    for k in range(n_ori):
        block = shuffled[:, bounds[k]:bounds[k + 1]]
        half = block.shape[1] // 2
        A0[:, k] = block[:, :half].mean(axis=1)
        B0[:, k] = block[:, half:2 * half].mean(axis=1)
        curve0[:, k] = block.mean(axis=1)

    null_r = _rowwise_pearson(A0, B0)
    null_r = null_r[np.isfinite(null_r)]
    null_osi, _ = _osi_from_curves(curve0, ori_used)

    observed_curve = np.array([r.mean() for r in rates_used])[None, :]
    observed_osi = float(_osi_from_curves(observed_curve, ori_used)[0][0])

    # +1 corrections keep the permutation p-value strictly positive.
    p_rel = ((np.sum(null_r >= median_r) + 1) / (null_r.size + 1)
             if null_r.size else np.nan)
    p_osi = (np.sum(null_osi >= observed_osi) + 1) / (null_osi.size + 1)

    result.update({
        'median_r': median_r,
        'r_ci_low': ci_low,
        'r_ci_high': ci_high,
        'r_spearman_brown': float(sb),
        'p_perm_reliability': float(p_rel),
        'p_perm_osi': float(p_osi),
        'null_r_median': float(np.median(null_r)) if null_r.size else np.nan,
        'null_osi_mean': float(np.mean(null_osi)),
        'null_osi_p95': float(np.percentile(null_osi, 95)),
        'reliable': bool(np.isfinite(p_rel) and p_rel < alpha and median_r > 0),
        'n_orientations_used': n_ori,
    })
    return result


def bootstrap_tuning_curve(trial_rates, orientations, n_boot=N_BOOT, ci=95,
                           rng=None):
    """
    Bootstrap the whole tuning curve by resampling trials within each orientation.

    Each of `n_boot` iterations resamples the trials of every orientation with
    replacement and recomputes the curve, giving R(theta) +/- CI at every
    orientation together with bootstrap CIs on OSI, preferred orientation and
    modulation index.

    Returns dict with per-orientation mean/CI arrays and parameter CIs.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2

    rates_by_ori = [np.asarray(trial_rates.get(ori, []), dtype=float)
                    for ori in orientations]
    n_ori = len(orientations)

    if any(r.size == 0 for r in rates_by_ori):
        nan_curve = [np.nan] * n_ori
        return {
            'n_boot': int(n_boot), 'ci': ci,
            'mean_curve': nan_curve, 'ci_low': nan_curve, 'ci_high': nan_curve,
            'osi_mean': np.nan, 'osi_ci_low': np.nan, 'osi_ci_high': np.nan,
            'pref_ori_circ_sd_deg': np.nan, 'pref_ori_ci_low_deg': np.nan,
            'pref_ori_ci_high_deg': np.nan, 'modulation_ci_low': np.nan,
            'modulation_ci_high': np.nan,
        }

    boot = np.empty((n_boot, n_ori))
    for k, rates in enumerate(rates_by_ori):
        idx = rng.integers(0, rates.size, size=(n_boot, rates.size))
        boot[:, k] = rates[idx].mean(axis=1)

    boot_osi, boot_pref = _osi_from_curves(boot, orientations)
    boot_max = boot.max(axis=1)
    boot_min = boot.min(axis=1)
    boot_mod = (boot_max - boot_min) / (boot_max + boot_min + 1e-12)

    observed_curve = np.array([r.mean() for r in rates_by_ori])[None, :]
    observed_pref = float(_osi_from_curves(observed_curve, orientations)[1][0])
    pref_sd, pref_lo, pref_hi = _circular_axial_stats(boot_pref, observed_pref)

    return {
        'n_boot': int(n_boot),
        'ci': ci,
        'mean_curve': boot.mean(axis=0).tolist(),
        'ci_low': np.percentile(boot, lo_q, axis=0).tolist(),
        'ci_high': np.percentile(boot, hi_q, axis=0).tolist(),
        'osi_mean': float(boot_osi.mean()),
        'osi_ci_low': float(np.percentile(boot_osi, lo_q)),
        'osi_ci_high': float(np.percentile(boot_osi, hi_q)),
        'pref_ori_circ_sd_deg': pref_sd,
        'pref_ori_ci_low_deg': pref_lo,
        'pref_ori_ci_high_deg': pref_hi,
        'modulation_ci_low': float(np.percentile(boot_mod, lo_q)),
        'modulation_ci_high': float(np.percentile(boot_mod, hi_q)),
    }


def characterize_unit(trial_rates, trial_baseline_rates, orientations,
                      n_splits=N_SPLITS, n_boot=N_BOOT, n_shuffles=N_SHUFFLES,
                      rng=None, alpha=ALPHA):
    """Run all three characterizations for one unit and return them in a dict."""
    rng = np.random.default_rng() if rng is None else rng
    return {
        'responsiveness': compute_visual_responsiveness(
            trial_rates, trial_baseline_rates, orientations, alpha=alpha),
        'reliability': compute_tuning_reliability(
            trial_rates, orientations, n_splits=n_splits,
            n_shuffles=n_shuffles, rng=rng, alpha=alpha),
        'bootstrap': bootstrap_tuning_curve(
            trial_rates, orientations, n_boot=n_boot, rng=rng),
    }


def _fmt_p(p):
    """Compact p-value string for figure annotations."""
    if p is None or not np.isfinite(p):
        return 'n/a'
    if p < 1e-4:
        return '<1e-4'
    return f'{p:.4f}' if p < 0.01 else f'{p:.3f}'


# =============================================================================
# TUNING CURVE CALCULATION
# =============================================================================

def calculate_tuning_curves(neural_data, time_window=(0.07, 0.16),
                            baseline_window=DEFAULT_BASELINE_WINDOW,
                            characterize=True, n_splits=N_SPLITS,
                            n_boot=N_BOOT, n_shuffles=N_SHUFFLES, seed=0):
    """
    Calculate tuning curves for all units.

    Args:
        neural_data: dict as produced by GratingExport (spike_data / trial_info)
        time_window: (start, end) s re. onset — the evoked analysis window
        baseline_window: (start, end) s re. onset for the trial-matched baseline.
            Defaults to the pre-onset ITI tail (-0.2, 0) s.
        characterize: run responsiveness / split-half / bootstrap statistics
        n_splits, n_boot, n_shuffles: repetitions for the split-half, bootstrap
            and label-shuffle null respectively
        seed: base seed — each unit gets its own reproducible generator

    Returns:
        Dictionary containing:
        - unit_tuning_data: dict with unit_id as key, tuning data as value
        - unique_orientations: list of tested orientations
        - experiment_info: dict with experimental parameters
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start
    
    unit_ids = list(neural_data['spike_data'].keys())
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = sorted(neural_data['trial_info']['unique_orientations'])
    
    print(f"\nCalculating tuning curves:")
    print(f"  Units: {len(unit_ids)}")
    print(f"  Orientations: {format_grating_values(unique_orientations)}")
    print(f"  Time window: {window_start:.3f}-{window_end:.3f}s ({window_duration:.3f}s)")
    if characterize:
        print(f"  Baseline window: {baseline_window[0]:.3f}-{baseline_window[1]:.3f}s "
              f"(ITI, trial-matched)")
        print(f"  Statistics: {n_splits} split-halves, {n_boot} bootstraps, "
              f"{n_shuffles} label shuffles")

    unit_tuning_data = {}

    for unit_index, unit_id in enumerate(unit_ids):
        # Collect firing rates per trial
        unit_trials = neural_data['spike_data'][unit_id]
        trial_rates = {ori: [] for ori in unique_orientations}
        trial_baseline_rates = {ori: [] for ori in unique_orientations}

        for trial_data in unit_trials:
            orientation = trial_data['orientation']
            if orientation in unique_orientations:
                spike_times = np.array(trial_data['spike_times'])
                spikes_in_window = np.sum((spike_times >= window_start) & 
                                         (spike_times < window_end))
                firing_rate = spikes_in_window / window_duration
                trial_rates[orientation].append(firing_rate)
                trial_baseline_rates[orientation].append(
                    _rate_in_window(spike_times, baseline_window))
        
        # Calculate statistics per orientation
        mean_rates = []
        sem_rates = []
        std_rates = []
        trial_counts = []
        
        for ori in unique_orientations:
            rates = trial_rates[ori]
            if len(rates) > 0:
                mean_rates.append(np.mean(rates))
                sem_rates.append(stats.sem(rates))
                std_rates.append(np.std(rates))
                trial_counts.append(len(rates))
            else:
                mean_rates.append(0)
                sem_rates.append(0)
                std_rates.append(0)
                trial_counts.append(0)
        
        # Calculate tuning metrics
        mean_rates_arr = np.array(mean_rates)
        
        # Orientation selectivity index (OSI) - vector sum method
        theta_rad = 2 * np.deg2rad(unique_orientations)
        complex_sum = np.sum(mean_rates_arr * np.exp(1j * theta_rad))
        osi = np.abs(complex_sum) / (np.sum(mean_rates_arr) + 1e-12)
        preferred_ori = (np.angle(complex_sum) / 2.0) % np.pi
        preferred_ori_deg = np.rad2deg(preferred_ori)
        
        # Modulation index
        max_rate = np.max(mean_rates_arr)
        min_rate = np.min(mean_rates_arr)
        modulation_index = (max_rate - min_rate) / (max_rate + min_rate + 1e-12)
        
        # Baseline firing rate (mean across all orientations)
        baseline_rate = np.mean(mean_rates_arr)
        
        # PSTH: 20 ms bins, -0.2 to 1.5 s relative to stimulus onset
        psth_bin_s = 0.02
        psth_edges = np.arange(-0.2, 1.5 + psth_bin_s, psth_bin_s)
        psth_t = (psth_edges[:-1] + psth_edges[1:]) / 2
        psth_per_ori = {}
        for ori in unique_orientations:
            spikes_all = []
            n_ori_trials = 0
            for trial_data in unit_trials:
                if trial_data['orientation'] == ori:
                    spikes_all.append(np.array(trial_data['spike_times']))
                    n_ori_trials += 1
            if n_ori_trials > 0 and spikes_all:
                counts, _ = np.histogram(np.concatenate(spikes_all), bins=psth_edges)
                psth_per_ori[ori] = (counts / (n_ori_trials * psth_bin_s)).tolist()
            else:
                psth_per_ori[ori] = [0.0] * len(psth_t)

        unit_tuning_data[unit_id] = {
            'orientations': unique_orientations,
            'mean_rates': mean_rates,
            'sem_rates': sem_rates,
            'std_rates': std_rates,
            'trial_counts': trial_counts,
            'trial_rates': trial_rates,
            'trial_baseline_rates': trial_baseline_rates,
            'baseline_window': tuple(baseline_window),
            'time_window': tuple(time_window),
            'osi': osi,
            'preferred_orientation_deg': preferred_ori_deg,
            'modulation_index': modulation_index,
            'max_rate': max_rate,
            'min_rate': min_rate,
            'baseline_rate': baseline_rate,
            'psth_per_ori': psth_per_ori,
            'psth_t': psth_t.tolist(),
        }

        # Responsiveness, split-half reliability and bootstrap CIs — each unit
        # gets its own generator so results do not depend on unit ordering.
        if characterize:
            unit_tuning_data[unit_id].update(characterize_unit(
                trial_rates, trial_baseline_rates, unique_orientations,
                n_splits=n_splits, n_boot=n_boot, n_shuffles=n_shuffles,
                rng=np.random.default_rng(seed + unit_index),
            ))

    if characterize and unit_tuning_data:
        _print_characterization_summary(unit_tuning_data)

    experiment_info = {
        'time_window': time_window,
        'baseline_window': tuple(baseline_window),
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_units': len(unit_ids)
    }
    
    return {
        'unit_tuning_data': unit_tuning_data,
        'unique_orientations': unique_orientations,
        'experiment_info': experiment_info
    }


def _print_characterization_summary(unit_tuning_data):
    """Population-level recap of responsiveness and reliability."""
    n = len(unit_tuning_data)
    resp = [d['responsiveness'] for d in unit_tuning_data.values() if 'responsiveness' in d]
    rel = [d['reliability'] for d in unit_tuning_data.values() if 'reliability' in d]
    if not resp:
        return
    n_resp = sum(r['responsive'] for r in resp)
    n_rel = sum(r['reliable'] for r in rel)
    median_r = np.nanmedian([r['median_r'] for r in rel]) if rel else np.nan
    print(f"\n  Visually responsive: {n_resp}/{n} units "
          f"({100 * n_resp / max(n, 1):.1f}%)")
    print(f"  Reliably tuned:      {n_rel}/{n} units "
          f"({100 * n_rel / max(n, 1):.1f}%)")
    print(f"  Median split-half r across units: {median_r:.3f}")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def format_characterization_text(tuning_data, time_window=None):
    """
    Render the responsiveness / split-half / bootstrap statistics of one unit as
    a monospace block for the figure's text panel. Returns '' if the unit was
    analysed without characterization.
    """
    resp = tuning_data.get('responsiveness')
    rel = tuning_data.get('reliability')
    boot = tuning_data.get('bootstrap')
    if not resp and not rel and not boot:
        return ''

    bw = tuning_data.get('baseline_window', DEFAULT_BASELINE_WINDOW)
    tw = time_window or tuning_data.get('time_window', (np.nan, np.nan))
    orientations = tuning_data['orientations']
    lines = []

    if resp:
        n_ori = len(resp.get('per_ori_p_holm', []))
        sig_oris = resp.get('sig_orientations', [])
        sig_txt = ', '.join(format_grating_value(o) + '°' for o in sig_oris[:3])
        if len(sig_oris) > 3:
            sig_txt += ', …'
        lines += [
            '1. VISUAL RESPONSIVENESS',
            f"  baseline (ITI) [{bw[0]:.2f},{bw[1]:.2f}]s: {resp['baseline_rate_hz']:.2f} Hz",
            f"  evoked  [{tw[0]:.2f},{tw[1]:.2f}]s: {resp['evoked_rate_hz']:.2f} Hz",
            f"  Δ rate:   {resp['delta_rate_hz']:+.2f} Hz   "
            f"RI={resp['response_index']:+.3f}",
            f"  Wilcoxon paired: p={_fmt_p(resp['p_wilcoxon'])} "
            f"(n={resp['n_trials']}, dz={resp['cohens_dz']:+.2f})",
            f"  sig. oris (Holm): {resp['n_sig_orientations']}/{n_ori}",
        ]
        if sig_txt:
            lines += [f"    [{sig_txt}]"]
        lines += [
            f"  best ori {resp['best_orientation_deg']:.0f}°: "
            f"Δ={resp['best_delta_hz']:+.2f} Hz p={_fmt_p(resp['best_p_holm'])}",
            f"  Kruskal-Wallis (ori): p={_fmt_p(resp['p_kruskal_orientation'])}",
            f"  → {'RESPONSIVE' if resp['responsive'] else 'NOT responsive'}"
            + (f" ({resp['response_sign']})" if resp['responsive'] else '')
            + f"  α={resp['alpha']:g}",
            '',
        ]

    if rel:
        lines += ['2. SPLIT-HALF RELIABILITY']
        if np.isfinite(rel['median_r']):
            lines += [
                f"  median r ({rel['n_splits']} splits): {rel['median_r']:+.3f}",
                f"  95% range: [{rel['r_ci_low']:+.3f}, {rel['r_ci_high']:+.3f}]",
                f"  Spearman-Brown: {rel['r_spearman_brown']:+.3f}",
                f"  shuffle null r: {rel['null_r_median']:+.3f}  "
                f"p={_fmt_p(rel['p_perm_reliability'])}",
                f"  → {'RELIABLE' if rel['reliable'] else 'NOT reliable'} tuning",
            ]
        else:
            lines += ['  n/a (too few trials per orientation)']
        lines += ['']

    if boot:
        lines += ['3. BOOTSTRAP TUNING CURVE']
        if np.isfinite(boot.get('osi_mean', np.nan)):
            # Per-orientation R(θ) ± CI is drawn as the shaded band on the
            # tuning curve itself; only the summary parameters are printed here.
            lines += [
                f"  ({boot['n_boot']} resamples, {boot['ci']:.0f}% CI)",
                f"  OSI: {tuning_data['osi']:.3f} "
                f"[{boot['osi_ci_low']:.3f}, {boot['osi_ci_high']:.3f}]",
                f"  pref ori: {tuning_data['preferred_orientation_deg']:.1f}° "
                f"± {boot['pref_ori_circ_sd_deg']:.1f}° (circ SD)",
                f"      CI [{boot['pref_ori_ci_low_deg']:.1f}°, "
                f"{boot['pref_ori_ci_high_deg']:.1f}°]"
                + (' (wraps 180°)'
                   if boot['pref_ori_ci_low_deg'] > boot['pref_ori_ci_high_deg']
                   else ''),
                f"  mod. index: {tuning_data['modulation_index']:.3f} "
                f"[{boot['modulation_ci_low']:.3f}, {boot['modulation_ci_high']:.3f}]",
            ]
            if rel and np.isfinite(rel.get('p_perm_osi', np.nan)):
                lines += [f"  OSI vs null ({rel['null_osi_mean']:.3f}): "
                          f"p={_fmt_p(rel['p_perm_osi'])}"]
        else:
            lines += ['  n/a (no trials)']

    return '\n'.join(lines)


def plot_single_tuning_curve(unit_id, tuning_data, unit_info=None,
                             time_window=(0.07, 0.16), save_path=None):
    """
    Create a comprehensive tuning curve plot for a single unit.

    Layout (2 rows × 4 columns):
      Col 0: Cartesian tuning curve (row 0), PSTH (row 1)
      Col 1: Polar plot (row 0), autocorrelogram (row 1)
      Col 2: Waveform (row 0), tuning statistics + unit info (row 1)
      Col 3: Responsiveness / split-half reliability / bootstrap stats (both rows)

    Args:
        unit_info: dict from neural_data['unit_info'] — may contain
                   'waveform_template', 'waveform_t_ms', 'acg_counts',
                   'acg_lags_ms', 'best_channel', 'channel_location_um',
                   'shank', 'quality'.  Pass None to skip those panels.
    """
    if unit_info is None:
        unit_info = {}

    fig = plt.figure(figsize=(30, 13))
    gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.30,
                  left=0.05, right=0.98, top=0.84, bottom=0.07)

    orientations = tuning_data['orientations']
    mean_rates = np.array(tuning_data['mean_rates'])
    sem_rates = np.array(tuning_data['sem_rates'])

    # Build a short header with unit identity
    shank_str = f"shank{unit_info.get('shank', '?')}"
    ch_str = (f"ch{unit_info.get('best_channel', '?')}"
              if unit_info.get('best_channel') is not None else "")
    loc = unit_info.get('channel_location_um')
    loc_str = f"  [{loc[0]:.0f}, {loc[1]:.0f}] µm" if loc else ""
    quality = unit_info.get('quality', '')
    header = f"{unit_id}  |  {shank_str}  {ch_str}{loc_str}  {quality}"

    # Verdict banner: visually responsive? reliably tuned?
    resp_hdr = tuning_data.get('responsiveness', {})
    rel_hdr = tuning_data.get('reliability', {})
    if resp_hdr:
        verdict = ('RESPONSIVE' if resp_hdr.get('responsive') else 'not responsive')
        if resp_hdr.get('response_sign') in ('enhanced', 'suppressed'):
            verdict += f" [{resp_hdr['response_sign']}]"
        verdict += f" (Δ={resp_hdr.get('delta_rate_hz', np.nan):+.2f} Hz, "
        verdict += f"p={_fmt_p(resp_hdr.get('p_wilcoxon'))})"
        if rel_hdr and np.isfinite(rel_hdr.get('median_r', np.nan)):
            verdict += (f"   |   {'RELIABLE' if rel_hdr.get('reliable') else 'unreliable'} tuning "
                        f"(split-half r={rel_hdr['median_r']:.2f}, "
                        f"p={_fmt_p(rel_hdr.get('p_perm_reliability'))})")
        header = f"{header}\n{verdict}"
    fig.suptitle(header, fontsize=24, fontweight='bold', y=0.985)

    # ------------------------------------------------------------------ #
    # 1. Cartesian tuning curve  (row 0, col 0)
    # ------------------------------------------------------------------ #
    ax1 = fig.add_subplot(gs[0, 0])

    # Bootstrap CI band (trials resampled within each orientation)
    boot = tuning_data.get('bootstrap', {})
    ci_low = np.asarray(boot.get('ci_low', []), dtype=float)
    ci_high = np.asarray(boot.get('ci_high', []), dtype=float)
    has_ci = ci_low.size == len(orientations) and np.all(np.isfinite(ci_low))
    if has_ci:
        ax1.fill_between(orientations, ci_low, ci_high, color='#2E86AB',
                         alpha=0.18, zorder=1,
                         label=f"bootstrap {boot.get('ci', 95):.0f}% CI")

    ax1.plot(orientations, mean_rates, '-', color='#2E86AB',
             linewidth=6.5, zorder=2, label='mean ± SEM')
    ax1.errorbar(orientations, mean_rates, yerr=sem_rates,
                 fmt='none', ecolor='#A23B72', capsize=14, capthick=4.0,
                 elinewidth=4.0, zorder=4)
    ax1.plot(orientations, mean_rates, 'o', color='#2E86AB',
             markersize=18, markeredgecolor='white', markeredgewidth=2.0,
             zorder=3)
    pref_idx = np.argmax(mean_rates)
    ax1.plot(orientations[pref_idx], mean_rates[pref_idx],
             '*', color='red', markersize=34,
             markeredgecolor='white', markeredgewidth=1.5,
             zorder=6)

    # ITI baseline and per-orientation significance (Holm-corrected Wilcoxon)
    resp = tuning_data.get('responsiveness', {})
    baseline_hz = resp.get('baseline_rate_hz', np.nan)
    if np.isfinite(baseline_hz):
        bw = tuning_data.get('baseline_window', DEFAULT_BASELINE_WINDOW)
        ax1.axhline(baseline_hz, color='dimgray', linestyle='--', linewidth=3.0,
                    zorder=1,
                    label=f'ITI baseline ({bw[0]:.2f}–{bw[1]:.2f}s)')

    upper = np.maximum(mean_rates + sem_rates, ci_high) if has_ci else mean_rates + sem_rates
    p_holm = np.asarray(resp.get('per_ori_p_holm', []), dtype=float)
    if p_holm.size == len(orientations):
        span = float(np.nanmax(upper)) - min(0.0, float(np.nanmin(mean_rates)))
        offset = 0.05 * (span if span > 0 else 1.0)
        alpha_lvl = resp.get('alpha', ALPHA)
        for k, pv in enumerate(p_holm):
            if not np.isfinite(pv) or pv >= alpha_lvl:
                continue
            marker = '***' if pv < 0.001 else ('**' if pv < 0.01 else '*')
            ax1.text(orientations[k], upper[k] + offset, marker,
                     ha='center', va='bottom', fontsize=22, fontweight='bold',
                     color='#C0392B', zorder=7)
        ax1.margins(y=0.14)

    if has_ci or np.isfinite(baseline_hz):
        ax1.legend(fontsize=13, loc='best', frameon=True, framealpha=0.85,
                   edgecolor='none', handlelength=1.4, borderpad=0.3)

    ax1.set_xlabel('Orientation (degrees)', fontsize=22, fontweight='bold',
                   labelpad=10)
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=22, fontweight='bold')
    ax1.set_title('Tuning Curve', fontsize=24, fontweight='bold', pad=12)
    ax1.set_xticks(orientations)
    ax1.set_xticklabels([format_grating_value(o) for o in orientations],
                        rotation=45, ha='right')
    ax1.tick_params(axis='both', labelsize=20, width=2.5, length=9)
    for spine in ('left', 'bottom'):
        ax1.spines[spine].set_linewidth(2.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # 2. Polar tuning curve  (row 0, col 1)
    # ------------------------------------------------------------------ #
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    theta = 2 * np.deg2rad(orientations)
    theta_plot = np.concatenate([theta, [theta[0]]])
    rates_plot = np.concatenate([mean_rates, [mean_rates[0]]])
    ax2.plot(theta_plot, rates_plot, 'o-', linewidth=5.0, markersize=14, color='#2E86AB')
    ax2.fill(theta_plot, rates_plot, alpha=0.25, color='#2E86AB')
    pref_theta = 2 * np.deg2rad(tuning_data['preferred_orientation_deg'])
    ax2.plot(pref_theta, np.max(mean_rates), '*', color='red', markersize=30)
    ax2.set_title('Polar', fontsize=24, fontweight='bold', pad=12)
    ax2.set_thetagrids(np.arange(0, 360, 45),
                       [f'{a/2:g}°' for a in np.arange(0, 360, 45)],
                       fontsize=22)
    radial_max = float(np.nanmax(mean_rates)) if mean_rates.size else 0.0
    if radial_max > 0:
        ax2.set_rlim(0, radial_max * 1.18)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='x', pad=18)
    ax2.grid(True, linewidth=1.4, alpha=0.45)
    ax2.spines['polar'].set_linewidth(1.8)

    # ------------------------------------------------------------------ #
    # 3. PSTH  (row 1, col 0)
    # ------------------------------------------------------------------ #
    ax3 = fig.add_subplot(gs[1, 0])
    psth_t = np.array(tuning_data['psth_t'])
    psth_colors = plt.cm.hsv(np.linspace(0, 1, len(orientations) + 1)[:-1])
    for k, ori in enumerate(orientations):
        psth_rate = gaussian_filter1d(np.array(tuning_data['psth_per_ori'][ori]), sigma=1.5)
        ax3.plot(psth_t, psth_rate, color=psth_colors[k],
                 linewidth=3.5, label=f'{format_grating_value(ori)}°', alpha=0.9)
    ax3.axvline(0, color='black', linewidth=3.0, linestyle='--', label='onset')
    ax3.axvspan(time_window[0], time_window[1], alpha=0.15, color='gray',
                label='analysis\nwindow')
    ax3.set_xlabel('Time re. onset (s)', fontsize=22, fontweight='bold')
    ax3.set_ylabel('Firing Rate (Hz)', fontsize=22, fontweight='bold')
    ax3.set_title('PSTH', fontsize=24, fontweight='bold', pad=12)
    ax3.legend(fontsize=9, loc='upper right', ncol=2, frameon=True,
               framealpha=0.85, edgecolor='none',
               handlelength=1.2, handletextpad=0.4,
               columnspacing=0.8, labelspacing=0.25, borderpad=0.3)
    ax3.tick_params(axis='both', labelsize=22, width=2.8, length=10)
    for spine in ('left', 'bottom'):
        ax3.spines[spine].set_linewidth(2.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # 6. Waveform  (row 0, col 2)
    # ------------------------------------------------------------------ #
    ax6 = fig.add_subplot(gs[0, 2])
    wf = unit_info.get('waveform_template')
    wf_t = unit_info.get('waveform_t_ms')
    if wf is not None:
        wf_arr = np.array(wf)
        t_arr = np.array(wf_t) if wf_t is not None else np.arange(len(wf_arr))
        wf_arr = wf_arr / (np.max(np.abs(wf_arr)) + 1e-12)   # normalize to ±1
        ax6.plot(t_arr, wf_arr, color='black', linewidth=4.5)
    else:
        ax6.text(0.5, 0.5, 'no waveform', ha='center', va='center',
                 transform=ax6.transAxes, color='gray', fontsize=20)
    ax6.set_title('Waveform', fontsize=24, fontweight='bold', pad=12)
    ax6.set_axis_off()

    # ------------------------------------------------------------------ #
    # 7. ACG  (row 1, col 1)
    # ------------------------------------------------------------------ #
    ax7 = fig.add_subplot(gs[1, 1])
    acg = unit_info.get('acg_counts')
    acg_lags = unit_info.get('acg_lags_ms')
    if acg is not None:
        acg_arr = np.array(acg)
        lags_arr = np.array(acg_lags)
        # zero out the central bin (self-coincidence)
        center = len(acg_arr) // 2
        acg_arr[center] = 0
        ax7.bar(lags_arr, acg_arr, width=(lags_arr[1] - lags_arr[0]) * 0.9,
                color='#5B8DB8', edgecolor='none', alpha=0.9)
        ax7.axvline(0, color='red', linewidth=3.0, linestyle='--')
        ax7.set_xlim(-25, 25)
        ax7.set_xlabel('Lag (ms)', fontsize=22, fontweight='bold')
        ax7.set_ylabel('Rate (Hz)', fontsize=22, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'no ACG', ha='center', va='center',
                 transform=ax7.transAxes, color='gray', fontsize=20)
    ax7.set_title('Autocorrelogram', fontsize=24, fontweight='bold', pad=12)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    for spine in ('left', 'bottom'):
        ax7.spines[spine].set_linewidth(2.5)
    ax7.tick_params(labelsize=22, width=2.8, length=10)

    # ------------------------------------------------------------------ #
    # 8. Combined stats + unit info  (row 1, col 2)
    # ------------------------------------------------------------------ #
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.axis('off')
    loc = unit_info.get('channel_location_um')
    loc_txt = (f"[{loc[0]:.0f}, {loc[1]:.0f}] µm" if loc else 'N/A')
    stats_text = (
        f"TUNING STATISTICS\n"
        f"OSI:        {tuning_data['osi']:.3f}\n"
        f"Preferred:  {tuning_data['preferred_orientation_deg']:.2f}°\n"
        f"Mod. Index: {tuning_data['modulation_index']:.3f}\n"
        f"Max FR:     {tuning_data['max_rate']:.2f} Hz\n"
        f"Min FR:     {tuning_data['min_rate']:.2f} Hz\n"
        f"Mean FR:    {tuning_data['baseline_rate']:.2f} Hz\n"
        f"Trials:     {sum(tuning_data['trial_counts'])} "
        f"({min(tuning_data['trial_counts'])}"
        f"–{max(tuning_data['trial_counts'])}/ori)\n"
        f"\n"
        f"UNIT INFO\n"
        f"Shank:    {unit_info.get('shank', 'N/A')}\n"
        f"Channel:  {unit_info.get('best_channel', 'N/A')}\n"
        f"Position: {loc_txt}\n"
        f"Quality:  {unit_info.get('quality', 'N/A')}\n"
        f"N spikes: {unit_info.get('n_spikes_total', 'N/A')}"
    )
    ax8.text(0.02, 0.98, stats_text, transform=ax8.transAxes,
             fontsize=17, verticalalignment='top', fontfamily='monospace',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                       alpha=0.5))

    # ------------------------------------------------------------------ #
    # 9. Responsiveness / reliability / bootstrap  (col 3, both rows)
    # ------------------------------------------------------------------ #
    ax9 = fig.add_subplot(gs[:, 3])
    ax9.axis('off')
    char_text = format_characterization_text(tuning_data, time_window=time_window)
    if char_text:
        ax9.text(0.0, 1.0, char_text, transform=ax9.transAxes,
                 fontsize=15, verticalalignment='top', fontfamily='monospace',
                 fontweight='bold', linespacing=1.45,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#D6EAF8',
                           alpha=0.55))
    else:
        ax9.text(0.5, 0.5, 'no characterization statistics',
                 ha='center', va='center', transform=ax9.transAxes,
                 color='gray', fontsize=18)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_svg = save_path.suffix.lower() == '.svg'
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    transparent=is_svg,
                    facecolor='none' if is_svg else 'white')
        plt.close(fig)

    return fig


def plot_all_tuning_curves_summary(tuning_results, save_path=None, max_per_page=16):
    """
    Create summary plots showing all tuning curves in a grid.
    
    Args:
        tuning_results: Output from calculate_tuning_curves()
        save_path: Base path for saving (will add _page1.png, _page2.png, etc.)
        max_per_page: Maximum number of units per page (default 16 = 4x4 grid)
    """
    unit_tuning_data = tuning_results['unit_tuning_data']
    unique_orientations = tuning_results['unique_orientations']
    unit_ids = sorted(unit_tuning_data.keys())
    
    n_units = len(unit_ids)
    n_pages = int(np.ceil(n_units / max_per_page))
    
    figures = []
    
    for page in range(n_pages):
        start_idx = page * max_per_page
        end_idx = min((page + 1) * max_per_page, n_units)
        page_units = unit_ids[start_idx:end_idx]
        
        n_units_page = len(page_units)
        n_cols = 4
        n_rows = int(np.ceil(n_units_page / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 7 * n_rows))
        fig.suptitle(f'Tuning Curves Summary (Page {page + 1}/{n_pages})',
                     fontsize=28, fontweight='bold')

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, unit_id in enumerate(page_units):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            tuning_data = unit_tuning_data[unit_id]
            mean_rates = tuning_data['mean_rates']
            sem_rates = tuning_data['sem_rates']
            boot = tuning_data.get('bootstrap', {})
            resp = tuning_data.get('responsiveness', {})
            rel = tuning_data.get('reliability', {})

            ax.errorbar(unique_orientations, mean_rates, yerr=sem_rates,
                        marker='o', markersize=10, linewidth=3.0, capsize=6,
                        capthick=2.5, elinewidth=2.0, color='#2E86AB',
                        ecolor='#A23B72')
            ci_low = np.asarray(boot.get('ci_low', []), dtype=float)
            ci_high = np.asarray(boot.get('ci_high', []), dtype=float)
            if ci_low.size == len(unique_orientations) and np.all(np.isfinite(ci_low)):
                ax.fill_between(unique_orientations, ci_low, ci_high,
                                alpha=0.2, color='#2E86AB')
            else:
                ax.fill_between(unique_orientations,
                                np.array(mean_rates) - np.array(sem_rates),
                                np.array(mean_rates) + np.array(sem_rates),
                                alpha=0.2, color='#2E86AB')
            if np.isfinite(resp.get('baseline_rate_hz', np.nan)):
                ax.axhline(resp['baseline_rate_hz'], color='dimgray',
                           linestyle='--', linewidth=1.8)

            title = f'{unit_id}\nOSI: {tuning_data["osi"]:.2f}'
            if resp:
                title += (f' | Δ={resp["delta_rate_hz"]:+.1f}Hz '
                          f'p={_fmt_p(resp["p_wilcoxon"])}')
                if np.isfinite(rel.get('median_r', np.nan)):
                    title += f'\nsplit-half r={rel["median_r"]:.2f}'
                    title += f' ({"resp" if resp["responsive"] else "n.s."}'
                    title += f'/{"rel" if rel["reliable"] else "unrel"})'
            ax.set_title(title, fontsize=15, fontweight='bold')
            ax.set_xlabel('Orientation (°)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Rate (Hz)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=1.2)
            ax.tick_params(labelsize=14, width=1.8, length=6)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_linewidth(2.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide unused subplots
        for idx in range(n_units_page, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            suffix = Path(save_path).suffix or '.png'
            save_path_page = (Path(save_path).parent /
                              f"{Path(save_path).stem}_page{page + 1}{suffix}")
            is_svg = suffix.lower() == '.svg'
            fig.savefig(save_path_page, dpi=300, bbox_inches='tight',
                        transparent=is_svg,
                        facecolor='none' if is_svg else 'white')
            print(f"Saved summary page {page + 1} to: {save_path_page}")
        
        figures.append(fig)
    
    return figures


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def generate_tuning_curves(data_path, time_window=(0.07, 0.16),
                          output_folder=None, create_summary=True, plot_osi=True,
                          baseline_window=DEFAULT_BASELINE_WINDOW,
                          characterize=True, n_splits=N_SPLITS, n_boot=N_BOOT,
                          n_shuffles=N_SHUFFLES):
    """
    Complete pipeline: load data, calculate tuning curves, save all plots.

    Args:
        data_path: Path to neural data file
        time_window: Tuple of (start, end) time in seconds for analysis
        output_folder: Folder to save tuning curve plots (default: data_path_tuning_curves)
        create_summary: Whether to create summary plots with all units
        plot_osi: Whether to also plot the OSI distribution from the tuning
                  statistics CSV this run produces (default True)
        baseline_window: (start, end) s re. onset for the trial-matched ITI baseline
        characterize: run responsiveness / split-half / bootstrap statistics
        n_splits, n_boot, n_shuffles: repetition counts for those statistics

    Returns:
        Dictionary with tuning results
    """
    # Load data
    data = load_neural_data(data_path)
    all_unit_info = data.get('unit_info', {})

    # Calculate tuning curves (+ responsiveness / reliability / bootstrap)
    tuning_results = calculate_tuning_curves(
        data, time_window=time_window, baseline_window=baseline_window,
        characterize=characterize, n_splits=n_splits, n_boot=n_boot,
        n_shuffles=n_shuffles)
    unit_tuning_data = tuning_results['unit_tuning_data']

    # Set up output folder
    if output_folder is None:
        output_folder = Path(data_path).parent / f"{Path(data_path).stem}_tuning_curves"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving tuning curves to: {output_folder}")

    # Generate individual plots
    print(f"\nGenerating individual tuning curve plots...")
    unit_ids = sorted(unit_tuning_data.keys())

    for i, unit_id in enumerate(unit_ids, 1):
        clean_id = unit_id.replace('/', '_').replace('\\', '_')
        unit_info = all_unit_info.get(unit_id, {})

        # PNG figure
        save_path = output_folder / f"{clean_id}_tuning_curve.png"
        plot_single_tuning_curve(unit_id, unit_tuning_data[unit_id],
                                 unit_info=unit_info,
                                 time_window=time_window,
                                 save_path=save_path)

        if i % 10 == 0:
            print(f"  Processed {i}/{len(unit_ids)} units...")
    
    print(f"✓ Saved {len(unit_ids)} individual tuning curve plots")

    # Combined pkl: every unit's tuning + unit_info in a single file
    combined_path = output_folder / "all_units_tuning.pkl"
    combined = {
        unit_id: {
            'unit_id': unit_id,
            'tuning': unit_tuning_data[unit_id],
            'unit_info': all_unit_info.get(unit_id, {}),
        }
        for unit_id in unit_ids
    }
    with open(combined_path, 'wb') as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved combined tuning data for {len(unit_ids)} units to: {combined_path}")

    # Generate summary plots
    if create_summary:
        print(f"\nGenerating summary plots...")
        summary_path = output_folder / "tuning_curves_summary.png"
        plot_all_tuning_curves_summary(tuning_results, save_path=summary_path)
    
    # Save tuning statistics to CSV
    tuning_csv_path = output_folder / "tuning_statistics.csv"
    save_tuning_statistics(unit_tuning_data, tuning_csv_path)

    # OSI distribution summary, computed from the CSV just written
    if plot_osi:
        plot_osi_distribution(tuning_csv_path)

    print(f"\n✓ All plots saved to: {output_folder}")
    
    return tuning_results


def _find_unit_id(all_unit_info, unit_tuning_data, shank, unit_num):
    """Resolve (shank, unit_num) to a unit_id present in unit_tuning_data.

    Match strategy, in order:
      1. The trailing integer in the unit_id equals unit_num AND unit_info['shank'] == shank
      2. unit_num as a positional index into the sorted list of units on that shank
    """
    import re
    shank = int(shank)
    unit_num = int(unit_num)

    for uid, info in all_unit_info.items():
        if uid not in unit_tuning_data:
            continue
        try:
            if int(info.get('shank', -1)) != shank:
                continue
        except (TypeError, ValueError):
            continue
        nums = re.findall(r'\d+', str(uid))
        if nums and int(nums[-1]) == unit_num:
            return uid

    shank_units = sorted(
        uid for uid, info in all_unit_info.items()
        if uid in unit_tuning_data
        and str(info.get('shank', '')).isdigit()
        and int(info['shank']) == shank
    )
    if 0 <= unit_num < len(shank_units):
        return shank_units[unit_num]

    return None


def plot_selected_unit(data_path, shank, unit_num, time_window=(0.05, 1.0),
                       save_path=None, show=True,
                       baseline_window=DEFAULT_BASELINE_WINDOW,
                       characterize=True, n_splits=N_SPLITS, n_boot=N_BOOT,
                       n_shuffles=N_SHUFFLES):
    """
    Plot the tuning curve for one selected unit, addressed by shank# and unit#.

    Args:
        data_path: Path to neural data .pkl
        shank: Shank index (int).
        unit_num: Either the numeric suffix of the unit_id, or its 0-based
                  positional index among units on that shank.
        time_window: (start, end) seconds for firing-rate window.
        save_path: Optional path to save PNG. If None and show=True, displays.
        show: If True and not saving, call plt.show().

    Returns:
        (unit_id, fig)
    """
    data = load_neural_data(data_path)
    all_unit_info = data.get('unit_info', {})

    tuning_results = calculate_tuning_curves(
        data, time_window=time_window, baseline_window=baseline_window,
        characterize=characterize, n_splits=n_splits, n_boot=n_boot,
        n_shuffles=n_shuffles)
    unit_tuning_data = tuning_results['unit_tuning_data']

    matched = _find_unit_id(all_unit_info, unit_tuning_data, shank, unit_num)
    if matched is None:
        available = sorted(
            (int(info.get('shank', -1)), uid)
            for uid, info in all_unit_info.items()
            if uid in unit_tuning_data
        )
        msg = "\n".join(f"  shank{s}: {uid}" for s, uid in available[:25])
        raise ValueError(
            f"No unit found for shank={shank}, unit#={unit_num}.\n"
            f"First available units:\n{msg}"
        )

    print(f"Matched unit: {matched}  (shank={shank}, unit#={unit_num})")

    fig = plot_single_tuning_curve(
        matched, unit_tuning_data[matched],
        unit_info=all_unit_info.get(matched, {}),
        time_window=time_window,
        save_path=save_path,
    )
    if show and save_path is None:
        plt.show()
    return matched, fig


def save_tuning_statistics(unit_tuning_data, save_path):
    """Save tuning statistics to CSV file."""
    import csv
    
    save_path = Path(save_path)
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header — tuning metrics, then responsiveness / reliability / bootstrap
        writer.writerow([
            'unit_id', 'osi', 'preferred_orientation_deg', 'modulation_index',
            'max_rate_hz', 'min_rate_hz', 'baseline_rate_hz', 'range_hz',
            'total_trials',
            # responsiveness (evoked vs trial-matched ITI baseline)
            'iti_baseline_hz', 'evoked_hz', 'delta_rate_hz', 'response_index',
            'p_wilcoxon', 'cohens_dz', 'n_sig_orientations', 'best_ori_deg',
            'best_ori_delta_hz', 'best_ori_p_holm', 'p_kruskal_orientation',
            'responsive', 'response_sign',
            # split-half reliability
            'split_half_r_median', 'split_half_r_ci_low', 'split_half_r_ci_high',
            'split_half_r_spearman_brown', 'p_perm_reliability', 'reliable',
            # bootstrap
            'osi_ci_low', 'osi_ci_high', 'pref_ori_circ_sd_deg',
            'pref_ori_ci_low_deg', 'pref_ori_ci_high_deg',
            'modulation_ci_low', 'modulation_ci_high', 'p_perm_osi',
        ])

        def num(value, fmt='.4f'):
            """Format a possibly-missing/NaN statistic for the CSV."""
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return ''
            return format(value, fmt) if isinstance(value, float) else value

        # Data rows
        for unit_id in sorted(unit_tuning_data.keys()):
            data = unit_tuning_data[unit_id]
            resp = data.get('responsiveness', {})
            rel = data.get('reliability', {})
            boot = data.get('bootstrap', {})
            writer.writerow([
                unit_id,
                f"{data['osi']:.4f}",
                f"{data['preferred_orientation_deg']:.2f}",
                f"{data['modulation_index']:.4f}",
                f"{data['max_rate']:.2f}",
                f"{data['min_rate']:.2f}",
                f"{data['baseline_rate']:.2f}",
                f"{data['max_rate'] - data['min_rate']:.2f}",
                sum(data['trial_counts']),
                num(resp.get('baseline_rate_hz'), '.3f'),
                num(resp.get('evoked_rate_hz'), '.3f'),
                num(resp.get('delta_rate_hz'), '.3f'),
                num(resp.get('response_index'), '.4f'),
                num(resp.get('p_wilcoxon'), '.3e'),
                num(resp.get('cohens_dz'), '.3f'),
                resp.get('n_sig_orientations', ''),
                num(resp.get('best_orientation_deg'), '.2f'),
                num(resp.get('best_delta_hz'), '.3f'),
                num(resp.get('best_p_holm'), '.3e'),
                num(resp.get('p_kruskal_orientation'), '.3e'),
                int(resp['responsive']) if 'responsive' in resp else '',
                resp.get('response_sign', ''),
                num(rel.get('median_r')),
                num(rel.get('r_ci_low')),
                num(rel.get('r_ci_high')),
                num(rel.get('r_spearman_brown')),
                num(rel.get('p_perm_reliability'), '.3e'),
                int(rel['reliable']) if 'reliable' in rel else '',
                num(boot.get('osi_ci_low')),
                num(boot.get('osi_ci_high')),
                num(boot.get('pref_ori_circ_sd_deg'), '.2f'),
                num(boot.get('pref_ori_ci_low_deg'), '.2f'),
                num(boot.get('pref_ori_ci_high_deg'), '.2f'),
                num(boot.get('modulation_ci_low')),
                num(boot.get('modulation_ci_high')),
                num(rel.get('p_perm_osi'), '.3e'),
            ])

    print(f"✓ Saved tuning statistics to: {save_path}")


# =============================================================================
# OSI DISTRIBUTION
# =============================================================================

def plot_osi_distribution(csv_path, save_path=None):
    """
    Read a tuning_statistics.csv (as written by save_tuning_statistics) and plot
    an OSI distribution summary: histogram+KDE, cumulative distribution,
    violin+strip plot, and a text panel of summary statistics.

    Args:
        csv_path: Path to tuning_statistics.csv file
        save_path: Optional path to save the plot (default: same folder as CSV)
    """
    csv_path = Path(csv_path)

    print(f"Reading tuning statistics from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} units")
    print(f"OSI range: {df['osi'].min():.3f} - {df['osi'].max():.3f}")
    print(f"OSI mean: {df['osi'].mean():.3f} ± {df['osi'].std():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Orientation Selectivity Index (OSI) Distribution',
                 fontsize=16, fontweight='bold')

    # 1. Histogram with density curve
    ax1 = axes[0, 0]
    ax1.hist(df['osi'], bins=30, density=True,
             alpha=0.7, color='#2E86AB', edgecolor='black')

    kde = stats.gaussian_kde(df['osi'])
    x_range = np.linspace(df['osi'].min(), df['osi'].max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    mean_osi = df['osi'].mean()
    median_osi = df['osi'].median()
    ax1.axvline(mean_osi, color='green', linestyle='--', linewidth=2,
                label=f'Mean: {mean_osi:.3f}')
    ax1.axvline(median_osi, color='orange', linestyle='--', linewidth=2,
                label=f'Median: {median_osi:.3f}')

    ax1.set_xlabel('OSI', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('OSI Distribution (Histogram + KDE)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative distribution
    ax2 = axes[0, 1]
    sorted_osi = np.sort(df['osi'])
    cumulative = np.arange(1, len(sorted_osi) + 1) / len(sorted_osi)
    ax2.plot(sorted_osi, cumulative, linewidth=2, color='#2E86AB')
    ax2.axhline(0.5, color='orange', linestyle='--', linewidth=1.5,
                label=f'Median: {median_osi:.3f}')
    ax2.axvline(median_osi, color='orange', linestyle='--', linewidth=1.5)

    q25 = df['osi'].quantile(0.25)
    q75 = df['osi'].quantile(0.75)
    ax2.axvline(q25, color='gray', linestyle=':', linewidth=1.5,
                label=f'Q1: {q25:.3f}')
    ax2.axvline(q75, color='gray', linestyle=':', linewidth=1.5,
                label=f'Q3: {q75:.3f}')

    ax2.set_xlabel('OSI', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Violin + strip plot
    ax3 = axes[1, 0]
    parts = ax3.violinplot([df['osi']], positions=[1], widths=0.7,
                           showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.7)

    np.random.seed(42)
    y_jitter = np.random.normal(1, 0.04, size=len(df))
    ax3.scatter(y_jitter, df['osi'], alpha=0.3, s=20, color='darkblue')

    ax3.set_ylabel('OSI', fontsize=12, fontweight='bold')
    ax3.set_title('OSI Distribution (Violin + Strip)', fontsize=13, fontweight='bold')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['All Units'])
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    OSI DISTRIBUTION STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Sample Size:
    • Total units: {len(df)}

    Central Tendency:
    • Mean:       {df['osi'].mean():.4f}
    • Median:     {df['osi'].median():.4f}
    • Mode:       {df['osi'].mode().values[0] if len(df['osi'].mode()) > 0 else 'N/A'}

    Spread:
    • Std Dev:    {df['osi'].std():.4f}
    • Variance:   {df['osi'].var():.4f}
    • Range:      {df['osi'].max() - df['osi'].min():.4f}
    • IQR:        {df['osi'].quantile(0.75) - df['osi'].quantile(0.25):.4f}

    Distribution:
    • Min:        {df['osi'].min():.4f}
    • Q1 (25%):   {df['osi'].quantile(0.25):.4f}
    • Q2 (50%):   {df['osi'].quantile(0.50):.4f}
    • Q3 (75%):   {df['osi'].quantile(0.75):.4f}
    • Max:        {df['osi'].max():.4f}

    Shape:
    • Skewness:   {df['osi'].skew():.4f}
    • Kurtosis:   {df['osi'].kurtosis():.4f}

    Selectivity Categories:
    • High (OSI > 0.5):   {(df['osi'] > 0.5).sum()} ({(df['osi'] > 0.5).sum()/len(df)*100:.1f}%)
    • Medium (0.3-0.5):   {((df['osi'] >= 0.3) & (df['osi'] <= 0.5)).sum()} ({((df['osi'] >= 0.3) & (df['osi'] <= 0.5)).sum()/len(df)*100:.1f}%)
    • Low (OSI < 0.3):    {(df['osi'] < 0.3).sum()} ({(df['osi'] < 0.3).sum()/len(df)*100:.1f}%)
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    if save_path is None:
        save_path = csv_path.parent / 'OSI_distribution.png'
    else:
        save_path = Path(save_path)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved OSI distribution plot to: {save_path}")

    plt.close(fig)

    return df


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot grating tuning curves (all units or one selected unit)."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to neural data .pkl file")
    parser.add_argument("--shank", type=int, default=None,
                        help="Shank# of the unit to plot (use with --unit)")
    parser.add_argument("--unit", type=int, default=None,
                        help="Unit# on the given shank (numeric suffix of the "
                             "unit_id, or 0-based index among units on that shank)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (selected-unit mode) or folder (batch mode)")
    parser.add_argument("--t0", type=float, default=0.05, help="Window start (s)")
    parser.add_argument("--t1", type=float, default=1.0, help="Window end (s)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip summary grid in batch mode")
    parser.add_argument("--b0", type=float, default=DEFAULT_BASELINE_WINDOW[0],
                        help="Baseline (ITI) window start (s, re. onset)")
    parser.add_argument("--b1", type=float, default=DEFAULT_BASELINE_WINDOW[1],
                        help="Baseline (ITI) window end (s, re. onset)")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS,
                        help="Split-half repetitions")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="Bootstrap resamples of the tuning curve")
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES,
                        help="Orientation-label shuffles for the null distributions")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip responsiveness / reliability / bootstrap statistics")
    args = parser.parse_args()

    DATA_PATH = args.data
    if not DATA_PATH:
        DATA_PATH = resolve_data_path()

    time_window = (args.t0, args.t1)
    baseline_window = (args.b0, args.b1)
    stats_kwargs = dict(
        baseline_window=baseline_window,
        characterize=not args.no_stats,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        n_shuffles=args.n_shuffles,
    )

    try:
        if args.shank is not None and args.unit is not None:
            # ---- Selected-unit mode ----
            plot_selected_unit(
                data_path=DATA_PATH,
                shank=args.shank,
                unit_num=args.unit,
                time_window=time_window,
                save_path=args.out,
                show=(args.out is None),
                **stats_kwargs,
            )
            print("\n✓ Selected-unit plot complete.")
        else:
            # ---- Batch mode ----
            tuning_results = generate_tuning_curves(
                data_path=DATA_PATH,
                time_window=time_window,
                output_folder=args.out,
                create_summary=not args.no_summary,
                **stats_kwargs,
            )
            print("\n" + "="*60)
            print("Tuning curve analysis complete!")
            print("="*60)

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
