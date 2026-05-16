"""
Cortical Reactivation Detection — Step 1: Synchronous Activity Prior
=====================================================================
Adapted from Nguyen et al. 2023 (Nature) for spiking data.

This script implements the pipeline to detect moments of synchronous
population activity during sleep (the "prior" for reactivation classification).

Paper pipeline (ca2+ imaging) → Your adaptation (spikes at 30 kHz):
  - Deconvolved ΔF/F          → Binned spike counts (bin first from spike times)
  - Normalize by top 1%       → Z-score each neuron
  - Rolling max (4 frames)    → Rolling max (~380 ms window, rescaled to your bins)
  - DoG filter (1.5, 6, 25s)  → Same time constants, rescaled to bin size
  - Minimum across 3 filters  → Same
  - Top 5% of S1+S2 neurons   → Top 10% recommended (only 140 neurons)
  - Threshold > 5 SD          → Same

Data sources (project-specific):
  - preferred_neurons.pkl     → driven mask, selectivity index (from identify_preferred_neurons.py)
  - SORTOUT_FOLDER/shankX/    → Phy spike trains for sleep epochs (from reactivation_config.py)
  - EPOCHS dict               → sleep1 / sleep2 boundaries in raw 30 kHz samples

Usage:
    Configure reactivation_config.py, run identify_preferred_neurons.py first,
    then run this script.  Figures saved to REACTIVATION_DIR.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ranksums
from pathlib import Path


# ============================================================
# STEP 0: Bin spike times → firing rate matrix
# ============================================================

def bin_spikes(spike_times_list, total_duration_s, bin_size_s=0.05):
    """
    Convert a list of spike time arrays into a (n_neurons × n_bins) matrix.

    Parameters
    ----------
    spike_times_list : list of np.ndarray
        Each array holds spike times IN SECONDS for one neuron.
        If you have sample indices at 30 kHz, do: spikes_s = spikes_samples / 30000
    total_duration_s : float
        Total duration of the recording session in seconds.
    bin_size_s : float
        Bin width in seconds. Default = 50 ms.
        At 50 ms → 1 bin ≈ 1 imaging frame in the paper (paper: ~96 ms).
        If you want to match the paper exactly, use 0.096 s.

    Returns
    -------
    spike_matrix : np.ndarray, shape (n_neurons, n_bins)
        Spike counts per bin.
    time_axis : np.ndarray, shape (n_bins,)
        Center time of each bin in seconds.
    """
    n_bins = int(total_duration_s / bin_size_s)
    n_neurons = len(spike_times_list)
    spike_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)

    for i, spikes in enumerate(spike_times_list):
        if len(spikes) == 0:
            continue
        counts, _ = np.histogram(spikes, bins=n_bins,
                                  range=(0, total_duration_s))
        spike_matrix[i] = counts.astype(np.float32)

    time_axis = np.arange(n_bins) * bin_size_s + bin_size_s / 2.0
    return spike_matrix, time_axis


# ============================================================
# STEP 1A: Z-score normalize each neuron
# ============================================================

def zscore_normalize(spike_matrix):
    """
    Z-score each neuron's trace independently.

    This replaces the paper's "normalize by top 1% of values" which is
    specific to fluorescence scaling. Z-scoring is more natural for
    spike counts and makes neurons comparably scaled for the classifier.

    Parameters
    ----------
    spike_matrix : np.ndarray, shape (n_neurons, n_bins)

    Returns
    -------
    normalized : np.ndarray, shape (n_neurons, n_bins)
    """
    mean = spike_matrix.mean(axis=1, keepdims=True)
    std = spike_matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0   # silent neurons: prevent div-by-zero
    return (spike_matrix - mean) / std


# ============================================================
# STEP 1B: Rolling maximum
# ============================================================

def rolling_max(spike_matrix, window_s=0.38, bin_size_s=0.05):
    """
    Apply a rolling maximum across a time window for each neuron.

    Why rolling max?
    Reactivation events are fast (~350 ms bursts). The rolling max
    ensures that even brief spikes within the window get "expanded"
    so the DoG filter can detect them reliably.

    Paper used 4 frames at 10.42 Hz = ~380 ms window.
    We replicate this with window_s = 0.38 s (adjust if needed).

    Parameters
    ----------
    spike_matrix : np.ndarray, shape (n_neurons, n_bins)
    window_s : float
        Window duration in seconds. Default = 380 ms (paper equivalent).
    bin_size_s : float
        Bin size in seconds (must match binning step).

    Returns
    -------
    rolled : np.ndarray, shape (n_neurons, n_bins)
    """
    n_frames = max(1, round(window_s / bin_size_s))
    print(f"Rolling max window: {n_frames} bins = {n_frames * bin_size_s * 1000:.0f} ms")

    n_neurons, n_bins = spike_matrix.shape
    # Pad on the left so the output has the same length
    pad = n_frames - 1
    padded = np.pad(spike_matrix, ((0, 0), (pad, 0)), mode='edge')

    # Efficient sliding window max using stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, window_shape=n_frames, axis=1)
    return windows.max(axis=-1)


# ============================================================
# STEP 1C: Difference-of-Gaussian (DoG) filtering
# ============================================================

def apply_dog_filters(spike_matrix, bin_size_s=0.05,
                       sigma_broad_s=(1.5, 6.0, 25.0),
                       sigma_narrow_s=0.2):
    """
    High-pass filter using three Difference-of-Gaussian (DoG) filters.

    Why DoG?
    Sleep data has slow firing rate fluctuations tied to brain states
    (NREM/REM cycles, slow oscillations, UP/DOWN states). These slow
    co-fluctuations would be mistakenly classified as reactivations.
    The DoG filter removes slow components and preserves fast bursts.

    Each filter = Gaussian(narrow) - Gaussian(broad):
      - narrow sigma (~0.2s): keeps fast transient bursts
      - broad sigmas (1.5s, 6s, 25s): three scales of slow drift removed

    Parameters
    ----------
    spike_matrix : np.ndarray, shape (n_neurons, n_bins)
        Rolling-maxed, z-scored activity.
    bin_size_s : float
        Bin size in seconds.
    sigma_broad_s : tuple of float
        Broad Gaussian sigma values in seconds (slow drift timescales).
    sigma_narrow_s : float
        Narrow Gaussian sigma in seconds (preserve fast events).

    Returns
    -------
    filtered_traces : list of 3 np.ndarrays, each shape (n_neurons, n_bins)
    """
    sigma_narrow_bins = sigma_narrow_s / bin_size_s
    narrow = gaussian_filter1d(spike_matrix.astype(np.float64),
                                sigma=sigma_narrow_bins, axis=1)

    filtered_traces = []
    for sigma_s in sigma_broad_s:
        sigma_broad_bins = sigma_s / bin_size_s
        broad = gaussian_filter1d(spike_matrix.astype(np.float64),
                                   sigma=sigma_broad_bins, axis=1)
        dog = narrow - broad   # retains fast signal, removes slow drift
        filtered_traces.append(dog)

    return filtered_traces


# ============================================================
# STEP 1D: Take minimum across three filtered traces
# ============================================================

def take_minimum_across_filters(filtered_traces):
    """
    For each neuron and timepoint, take the minimum across the three
    DoG-filtered traces.

    Why the minimum?
    If a slow component is present in any of the three filtered traces,
    the minimum will capture that low value, ensuring slow fluctuations
    are suppressed even if only one filter catches them.

    Parameters
    ----------
    filtered_traces : list of 3 arrays, each shape (n_neurons, n_bins)

    Returns
    -------
    min_filtered : np.ndarray, shape (n_neurons, n_bins)
    """
    stacked = np.stack(filtered_traces, axis=0)  # (3, n_neurons, n_bins)
    return stacked.min(axis=0)


# ============================================================
# STEP 1E: Synchronous activity prior
# ============================================================

def compute_synchrony_prior(min_filtered, stimulus_driven_mask,
                              top_fraction=0.10, threshold_sd=5.0):
    """
    Detect timepoints with synchronous population activity.

    The paper uses top 5% of S1-driven + top 5% of S2-driven neurons.
    With 140 neurons, top 5% = only 7 neurons — too few for a reliable
    population signal. We recommend top 10–15% instead.

    Why this step?
    Not every ITI/sleep timepoint is a candidate reactivation.
    We only want to classify moments where the stimulus-driven
    population fires synchronously — a fast burst above baseline.

    Parameters
    ----------
    min_filtered : np.ndarray, shape (n_neurons, n_bins)
        Output of take_minimum_across_filters().
    stimulus_driven_mask : np.ndarray, shape (n_neurons,), dtype bool
        True for neurons identified as S1- or S2-driven in wake session.
    top_fraction : float
        Fraction of stimulus-driven neurons to use as reference.
        Paper: 0.05 (top 5%). Recommended for 140 neurons: 0.10–0.15.
    threshold_sd : float
        SD threshold. Paper uses 5.0.

    Returns
    -------
    synchrony_prior : np.ndarray, shape (n_bins,), dtype bool
        True at candidate reactivation timepoints.
    mean_activity : np.ndarray, shape (n_bins,)
        Mean filtered activity of reference neurons (for plotting).
    top_neuron_idx : np.ndarray
        Indices of neurons used as the reference population.
    """
    driven_activity = min_filtered[stimulus_driven_mask]  # (n_driven, n_bins)
    n_driven = driven_activity.shape[0]

    # Select top neurons by mean absolute activity (most reliably driven)
    n_top = max(5, int(n_driven * top_fraction))
    mean_response = driven_activity.mean(axis=1)
    local_top_idx = np.argsort(mean_response)[-n_top:]
    top_activity = driven_activity[local_top_idx]  # (n_top, n_bins)

    # Convert local indices back to full neuron indices
    global_driven_idx = np.where(stimulus_driven_mask)[0]
    top_neuron_idx = global_driven_idx[local_top_idx]

    print(f"Using {n_top} reference neurons out of {n_driven} stimulus-driven neurons.")

    # Average across top neurons
    mean_activity = top_activity.mean(axis=0)  # (n_bins,)

    # Threshold
    mu = mean_activity.mean()
    sd = mean_activity.std()
    threshold = mu + threshold_sd * sd
    synchrony_prior = mean_activity > threshold

    n_events = np.sum(np.diff(synchrony_prior.astype(int)) == 1)
    print(f"Detected {n_events} candidate synchronous events "
          f"(threshold = {threshold:.3f}, mean = {mu:.3f}, sd = {sd:.3f})")

    return synchrony_prior, mean_activity, top_neuron_idx


# ============================================================
# HELPER: Identify stimulus-driven neurons from wake data
# ============================================================

def identify_stimulus_driven_neurons(wake_spike_matrix, time_axis,
                                      stim_onsets_s1, stim_onsets_s2,
                                      stim_duration_s=2.0, baseline_s=2.0,
                                      p_threshold=0.01):
    """
    Wilcoxon rank-sum test: baseline vs. stimulus period activity.

    Replicates the paper's neuron identification step.
    For each neuron, compare activity in the 2s baseline window
    vs. the 2s stimulus window across all trials.

    Parameters
    ----------
    wake_spike_matrix : np.ndarray, shape (n_neurons, n_bins)
        Z-scored activity from the WAKE session.
    time_axis : np.ndarray, shape (n_bins,)
        Time axis for the wake session in seconds.
    stim_onsets_s1 : np.ndarray
        Onset times of S1 presentations in seconds.
    stim_onsets_s2 : np.ndarray
        Onset times of S2 presentations in seconds.
    stim_duration_s : float
        Duration of each stimulus (paper: 2s).
    baseline_s : float
        Duration of baseline window before each stimulus (paper: 2s).
    p_threshold : float
        Significance threshold. Paper: 0.01.

    Returns
    -------
    is_driven : np.ndarray, shape (n_neurons,), dtype bool
        True for neurons significantly driven by S1 or S2.
    selectivity_index : np.ndarray, shape (n_neurons,)
        (mean_S1_response - mean_S2_response) / (mean_S1_response + mean_S2_response)
        Positive = S1-preferring, negative = S2-preferring.
    mean_s1 : np.ndarray, shape (n_neurons,)
    mean_s2 : np.ndarray, shape (n_neurons,)
    """
    bin_size_s = time_axis[1] - time_axis[0]
    stim_bins = max(1, round(stim_duration_s / bin_size_s))
    base_bins = max(1, round(baseline_s / bin_size_s))
    n_neurons = wake_spike_matrix.shape[0]

    def get_epoch_activity(onsets, n_bins_window, offset_bins=0):
        """Collect binned activity from epochs."""
        epochs = []
        for t0 in onsets:
            start_bin = int(round(t0 / bin_size_s)) + offset_bins
            end_bin = start_bin + n_bins_window
            if start_bin >= 0 and end_bin <= wake_spike_matrix.shape[1]:
                epochs.append(wake_spike_matrix[:, start_bin:end_bin])
        if len(epochs) == 0:
            return np.zeros((n_neurons, 0))
        return np.concatenate(epochs, axis=1)  # (n_neurons, n_trials * n_bins)

    all_onsets = np.concatenate([stim_onsets_s1, stim_onsets_s2])

    # Baseline: 2s window immediately before each stimulus onset
    baseline_activity = get_epoch_activity(all_onsets, base_bins,
                                            offset_bins=-base_bins)
    s1_activity = get_epoch_activity(stim_onsets_s1, stim_bins)
    s2_activity = get_epoch_activity(stim_onsets_s2, stim_bins)

    is_driven_s1 = np.zeros(n_neurons, dtype=bool)
    is_driven_s2 = np.zeros(n_neurons, dtype=bool)
    mean_s1 = np.zeros(n_neurons)
    mean_s2 = np.zeros(n_neurons)

    for i in range(n_neurons):
        base = baseline_activity[i]
        if s1_activity.shape[1] > 0:
            stat, p = ranksums(s1_activity[i], base)
            if p < p_threshold and s1_activity[i].mean() > base.mean():
                is_driven_s1[i] = True
            mean_s1[i] = s1_activity[i].mean()
        if s2_activity.shape[1] > 0:
            stat, p = ranksums(s2_activity[i], base)
            if p < p_threshold and s2_activity[i].mean() > base.mean():
                is_driven_s2[i] = True
            mean_s2[i] = s2_activity[i].mean()

    is_driven = is_driven_s1 | is_driven_s2
    print(f"Stimulus-driven neurons: {is_driven.sum()} / {n_neurons} "
          f"(S1: {is_driven_s1.sum()}, S2: {is_driven_s2.sum()}, "
          f"both: {(is_driven_s1 & is_driven_s2).sum()})")

    # Selectivity index: positive = S1-preferring, negative = S2-preferring
    denom = mean_s1 + mean_s2
    denom[denom == 0] = 1e-9
    selectivity_index = (mean_s1 - mean_s2) / denom

    return is_driven, selectivity_index, mean_s1, mean_s2


# ============================================================
# FIGURE: Raster sorted by preference + synchrony prior on top
# ============================================================

def plot_prior_figure(sleep_spike_matrix, sleep_time_axis,
                       synchrony_prior, mean_activity,
                       selectivity_index, is_driven,
                       window_s=300.0, bin_size_s=0.05,
                       save_path=None):
    """
    Generate a figure analogous to Extended Data Fig. 1 (first panel):
      - Top panel: mean filtered activity of reference neurons + threshold events
      - Bottom panel: raster of all neurons sorted by S1/S2 preference

    Parameters
    ----------
    sleep_spike_matrix : np.ndarray, shape (n_neurons, n_bins)
        Z-scored spike matrix from the SLEEP session.
    sleep_time_axis : np.ndarray, shape (n_bins,)
        Time axis for sleep session in seconds.
    synchrony_prior : np.ndarray, shape (n_bins,), dtype bool
    mean_activity : np.ndarray, shape (n_bins,)
        Mean filtered reference neuron activity.
    selectivity_index : np.ndarray, shape (n_neurons,)
        From identify_stimulus_driven_neurons().
    is_driven : np.ndarray, shape (n_neurons,), dtype bool
    window_s : float
        How many seconds to display. Default = 300 s (5 minutes of sleep).
    save_path : str or None
        If provided, save figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    # Sort neurons: high SI (S1-preferring) → low SI (S2-preferring)
    sort_order = np.argsort(selectivity_index)[::-1]
    sorted_activity = sleep_spike_matrix[sort_order]
    sorted_si = selectivity_index[sort_order]
    sorted_driven = is_driven[sort_order]

    # Find S1/S2 boundary (where SI crosses 0)
    boundary = np.searchsorted(-sorted_si, 0)

    # Limit time window
    mask = sleep_time_axis <= window_s
    t_plot = sleep_time_axis[mask]

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 5], hspace=0.05)

    # ── Panel 1: mean reference neuron activity + events ──────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_plot, mean_activity[mask], color='steelblue',
              linewidth=0.8, label='Mean ref. activity')
    # Shade synchronous events in orange
    ax1.fill_between(t_plot,
                      mean_activity[mask] * synchrony_prior[mask],
                      0, color='orange', alpha=0.6, label='Synchronous event')
    ax1.set_ylabel('Filtered\nactivity\n(a.u.)', fontsize=9)
    ax1.set_xlim(t_plot[0], t_plot[-1])
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.5)
    ax1.set_title('Synchronous activity prior (Step 1)', fontsize=11)
    ax1.tick_params(labelbottom=False)

    # ── Panel 2: binary prior ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(t_plot, synchrony_prior[mask].astype(float),
                      color='orange', step='mid')
    ax2.set_ylabel('Prior\n(binary)', fontsize=9)
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([0, 1])
    ax2.tick_params(labelbottom=False)

    # ── Panel 3: raster sorted by selectivity ────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    n_neurons = sorted_activity.shape[0]

    im = ax3.imshow(
        sorted_activity[:, mask],
        aspect='auto',
        extent=[t_plot[0] - bin_size_s / 2, t_plot[-1] + bin_size_s / 2, n_neurons, 0],
        cmap='Greys',
        vmin=-0.5, vmax=2.0,
        interpolation='nearest'
    )

    # Highlight driven neurons with a thin colored bar on the right
    driven_colors = np.where(sorted_driven, 1.0, 0.0).reshape(-1, 1)
    ax3.imshow(driven_colors,
               aspect='auto',
               extent=[t_plot[-1] * 1.002, t_plot[-1] * 1.015, n_neurons, 0],
               cmap='Greens', vmin=0, vmax=1,
               interpolation='nearest', clip_on=False)

    # S1/S2 boundary line
    ax3.axhline(boundary, color='gray', linestyle='--',
                 linewidth=1.2, alpha=0.7)
    ax3.text(t_plot[-1] * 1.018, boundary / 2,
              'S1\npref.', color='green', fontsize=9,
              va='center', ha='left', clip_on=False)
    ax3.text(t_plot[-1] * 1.018,
              boundary + (n_neurons - boundary) / 2,
              'S2\npref.', color='red', fontsize=9,
              va='center', ha='left', clip_on=False)

    ax3.set_ylabel('Neuron # (sorted\nby S1/S2 pref.)', fontsize=9)
    ax3.set_xlabel('Time (s)', fontsize=10)

    # Mark synchronous events on the raster with vertical orange lines
    event_onsets = t_plot[np.where(np.diff(synchrony_prior[mask].astype(int)) == 1)[0] + 1]
    for t_ev in event_onsets:
        ax3.axvline(t_ev, color='orange', lw=0.8, alpha=0.8)

    plt.colorbar(im, ax=ax3, label='Z-scored activity', shrink=0.3,
                  location='right', pad=0.12)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


# ============================================================
# FIGURE: Pipeline stages — raw z-score vs processed signal
# ============================================================

def plot_pipeline_stages(sleep_z, sleep_rolled, filtered_traces, min_filtered,
                          mean_activity, synchrony_prior, sleep_time_axis,
                          top_neuron_idx, threshold_sd=2.0,
                          window_s=300.0, save_path=None):
    """
    Diagnostic figure showing how the reference-neuron signal evolves through
    each processing step:
      Row 1 — Raw z-scored mean FR of reference neurons
      Row 2 — After rolling maximum
      Row 3 — After each of the 3 DoG filters (overlaid)
      Row 4 — Minimum across filters (final signal) + threshold + events

    Parameters
    ----------
    sleep_z : (n_units, n_bins)  raw z-scored binned spikes
    sleep_rolled : (n_units, n_bins)  after rolling max
    filtered_traces : list of 3 (n_units, n_bins)  DoG outputs
    min_filtered : (n_units, n_bins)  min across filters
    mean_activity : (n_bins,)  mean of reference neurons in min_filtered space
    synchrony_prior : (n_bins,) bool
    sleep_time_axis : (n_bins,)
    top_neuron_idx : indices of reference neurons (into the n_units axis)
    threshold_sd : float  — used only to annotate the threshold line
    window_s : float  seconds to display
    save_path : str or None
    """
    mask   = sleep_time_axis <= window_s
    t_plot = sleep_time_axis[mask]

    ref = top_neuron_idx   # reference neuron indices

    raw_mean    = sleep_z[ref][:, mask].mean(axis=0)
    rolled_mean = sleep_rolled[ref][:, mask].mean(axis=0)
    dog_means   = [f[ref][:, mask].mean(axis=0) for f in filtered_traces]
    min_mean    = min_filtered[ref][:, mask].mean(axis=0)

    mu  = mean_activity[mask].mean()
    sd  = mean_activity[mask].std()
    thr = mu + threshold_sd * sd

    # Event onset times
    prior_masked = synchrony_prior[mask]
    event_onsets = t_plot[np.where(np.diff(prior_masked.astype(int)) == 1)[0] + 1]

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw=dict(hspace=0.08))
    fig.suptitle(
        f"Processing pipeline — reference neurons (n={len(ref)})\n"
        f"First {window_s:.0f} s of sleep",
        fontsize=11, fontweight="bold",
    )

    def mark_events(ax):
        for t_ev in event_onsets:
            ax.axvline(t_ev, color='orange', lw=0.8, alpha=0.7)

    # Row 1: raw z-score
    axes[0].plot(t_plot, raw_mean, color='steelblue', lw=0.7)
    axes[0].set_ylabel('Raw\nz-score', fontsize=9)
    axes[0].axhline(0, color='k', lw=0.5, ls='--')
    mark_events(axes[0])

    # Row 2: after rolling max
    axes[1].plot(t_plot, rolled_mean, color='teal', lw=0.7)
    axes[1].set_ylabel('Rolling\nmax', fontsize=9)
    axes[1].axhline(0, color='k', lw=0.5, ls='--')
    mark_events(axes[1])

    # Row 3: DoG filtered traces overlaid
    dog_colors = ['#e07b39', '#9b59b6', '#2ecc71']
    dog_sigmas = ['1.5 s', '6 s', '25 s']
    for dog, col, lbl in zip(dog_means, dog_colors, dog_sigmas):
        axes[2].plot(t_plot, dog, color=col, lw=0.7, alpha=0.85, label=f'broad σ={lbl}')
    axes[2].axhline(0, color='k', lw=0.5, ls='--')
    axes[2].set_ylabel('DoG\nfiltered', fontsize=9)
    axes[2].legend(fontsize=7, loc='upper right', framealpha=0.5)
    mark_events(axes[2])

    # Row 4: min across filters + threshold + shaded events
    axes[3].plot(t_plot, min_mean, color='#c0392b', lw=0.8, label='min across filters')
    axes[3].axhline(thr, color='orange', lw=1.2, ls='--',
                     label=f'threshold ({threshold_sd} SD = {thr:.3f})')
    axes[3].fill_between(t_plot, min_mean,
                          where=prior_masked, color='orange', alpha=0.4,
                          label='synchronous event')
    axes[3].axhline(0, color='k', lw=0.5, ls='--')
    axes[3].set_ylabel('Min\nfiltered', fontsize=9)
    axes[3].set_xlabel('Time (s)', fontsize=10)
    axes[3].legend(fontsize=7, loc='upper right', framealpha=0.5)
    mark_events(axes[3])

    for ax in axes:
        ax.set_xlim(t_plot[0], t_plot[-1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


# ============================================================
# MAIN PIPELINE — edit the section below for your data
# ============================================================

if __name__ == "__main__":

    # ── Add sleep/ dir to path so config imports work ─────────────────────
    _SLEEP_DIR = Path(__file__).resolve().parent
    if str(_SLEEP_DIR) not in sys.path:
        sys.path.insert(0, str(_SLEEP_DIR))

    from reactivation_config import (
        SESSION, SORTOUT_FOLDER, EPOCHS, FS_RAW, REACTIVATION_DIR,
    )
    from spikeinterface.extractors import read_phy
    from spikeinterface import load_sorting_analyzer

    REACTIVATION_DIR.mkdir(parents=True, exist_ok=True)

    BIN_SIZE = 0.05   # 50 ms bins

    print(f"\n{'='*65}")
    print(f"detect_reactivation.py — {SESSION}")

    # ── LOAD preferred_neurons.pkl (output of identify_preferred_neurons.py) ──
    pref_pkl = REACTIVATION_DIR / "preferred_neurons.pkl"
    if not pref_pkl.exists():
        raise FileNotFoundError(
            f"preferred_neurons.pkl not found: {pref_pkl}\n"
            "Run identify_preferred_neurons.py first."
        )
    with open(pref_pkl, "rb") as fh:
        pref = pickle.load(fh)

    unit_ids         = pref["unit_ids"]          # list[str], e.g. 'shank5_unit42'
    is_driven        = pref["driven"]             # (n_units,) bool
    selectivity_index = pref["SI"]               # (n_units,) float, NaN for silent
    n_units          = len(unit_ids)
    print(f"  Units loaded from preferred_neurons.pkl : {n_units}")
    print(f"  Stimulus-driven : {is_driven.sum()}  "
          f"(S1-pref: {pref['pref_s1'].sum()}, S2-pref: {pref['pref_s2'].sum()})")

    # ── LOAD spike trains from Phy sorting ────────────────────────────────
    def load_sorting_for_shank(sortout_folder: Path, shank_id: str):
        shank_folder = sortout_folder / f"shank{shank_id}"
        if not shank_folder.exists():
            raise FileNotFoundError(f"Shank folder not found: {shank_folder}")
        for root, dirs, _ in os.walk(shank_folder):
            for dname in sorted(dirs):
                if not dname.startswith("sorting_results_"):
                    continue
                sr = Path(root) / dname
                for sub in ("phy", "sorting_analyzer"):
                    p = sr / sub
                    if not p.exists():
                        continue
                    try:
                        s = (read_phy(p) if sub == "phy"
                             else load_sorting_analyzer(p).sorting)
                        print(f"  Loaded shank{shank_id} from: {p}")
                        return s
                    except Exception as exc:
                        print(f"  [WARN] {p}: {exc}")
        raise FileNotFoundError(f"No valid sorting under: {shank_folder}")

    # ── LOAD OR CACHE full spike trains ───────────────────────────────────
    # Cache lives next to the shank folders so it can be shared across scripts.
    spike_cache_pkl = SORTOUT_FOLDER / f"{SESSION}_spike_trains_cache.pkl"

    if spike_cache_pkl.exists():
        print(f"\n  Loading spike trains from cache: {spike_cache_pkl}")
        with open(spike_cache_pkl, "rb") as fh:
            cache = pickle.load(fh)
        full_trains = cache["full_trains"]
        fs_sort     = cache["fs_sort"]
        print(f"  Cached units: {len(full_trains)}  fs: {fs_sort:.0f} Hz")
    else:
        needed_shanks = sorted({uid.split("_unit")[0].replace("shank", "")
                                 for uid in unit_ids if "_unit" in uid})
        print(f"\n  Shanks needed: {needed_shanks}")

        sortings = {}
        fs_sort  = None
        for sh in needed_shanks:
            try:
                s = load_sorting_for_shank(SORTOUT_FOLDER, sh)
                sortings[sh] = s
                if fs_sort is None:
                    fs_sort = s.sampling_frequency
            except FileNotFoundError as e:
                print(f"  [WARN] {e}")

        if fs_sort is None:
            raise RuntimeError("Could not load any sorting. Check SORTOUT_FOLDER.")
        print(f"  Sorting fs: {fs_sort:.0f} Hz")

        full_trains = {}
        n_missing = 0
        for uid_str in unit_ids:
            parts = uid_str.split("_unit")
            if len(parts) != 2:
                full_trains[uid_str] = np.array([], dtype=np.int64)
                n_missing += 1
                continue
            shank_key = parts[0].replace("shank", "")
            raw_uid   = parts[1]
            s = sortings.get(shank_key)
            if s is None:
                full_trains[uid_str] = np.array([], dtype=np.int64)
                n_missing += 1
                continue
            matched = next((u for u in s.unit_ids if str(u) == raw_uid), None)
            if matched is None:
                print(f"  [WARN] Unit {raw_uid} not in shank{shank_key} sorting")
                full_trains[uid_str] = np.array([], dtype=np.int64)
                n_missing += 1
            else:
                full_trains[uid_str] = s.get_unit_spike_train(matched)
        if n_missing:
            print(f"  {n_missing} units not matched — will be silent.")

        with open(spike_cache_pkl, "wb") as fh:
            pickle.dump({"full_trains": full_trains, "fs_sort": float(fs_sort)},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = spike_cache_pkl.stat().st_size / 1024**2
        print(f"  Spike train cache saved → {spike_cache_pkl}  ({size_mb:.1f} MB)")

    # ── PROCESS EACH SLEEP EPOCH INDEPENDENTLY ────────────────────────────
    # Each epoch is z-scored and filtered on its own baseline so that
    # concatenation artifacts don't bias the thresholds.
    sleep_epoch_names = [n for n in ("sleep1", "sleep2") if n in EPOCHS]
    if not sleep_epoch_names:
        raise ValueError("No 'sleep1' or 'sleep2' keys found in EPOCHS. "
                         "Check reactivation_config.py.")

    # Replace NaN SI values with 0 (silent neurons treated as non-selective)
    si_safe = np.where(np.isnan(selectivity_index), 0.0, selectivity_index)

    for ep_name in sleep_epoch_names:
        print(f"\n{'='*65}")
        print(f"Processing epoch: {ep_name}")
        print(f"{'='*65}")

        ep_start, ep_end = EPOCHS[ep_name]
        ep_dur_s = (ep_end - ep_start) / fs_sort
        print(f"  Samples [{ep_start:,} – {ep_end:,}]  ({ep_dur_s:.1f} s)")

        # Extract spike times for this epoch (time starts at 0)
        spike_times_ep = []
        for uid in unit_ids:
            train = full_trains[uid]
            mask  = (train >= ep_start) & (train < ep_end)
            spike_times_ep.append((train[mask] - ep_start) / fs_sort)

        total_spk = sum(len(t) for t in spike_times_ep)
        print(f"  Spikes in epoch: {total_spk:,}")

        # STEP 0: Bin spikes
        print(f"\n  STEP 0: Binning spikes")
        ep_matrix, ep_time = bin_spikes(spike_times_ep, ep_dur_s, BIN_SIZE)
        print(f"  Matrix shape: {ep_matrix.shape}")

        # STEP 1A: Z-score
        print(f"  STEP 1A: Z-score normalization")
        ep_z = zscore_normalize(ep_matrix)

        # STEP 1B: Rolling max
        print(f"  STEP 1B: Rolling maximum")
        ep_rolled = rolling_max(ep_z, window_s=0.38, bin_size_s=BIN_SIZE)

        # STEP 1C: DoG filters
        print(f"  STEP 1C: DoG filtering")
        ep_filtered = apply_dog_filters(ep_rolled, bin_size_s=BIN_SIZE)

        # STEP 1D: Min across filters
        print(f"  STEP 1D: Minimum across filters")
        ep_min = take_minimum_across_filters(ep_filtered)

        # STEP 1E: Synchrony prior
        print(f"  STEP 1E: Computing synchrony prior")
        ep_prior, ep_mean_act, ep_top_idx = compute_synchrony_prior(
            ep_min,
            stimulus_driven_mask=is_driven,
            top_fraction=0.2,
            threshold_sd=2.0,
        )

        # SAVE per-epoch PKL
        out = {
            "synchrony_prior"   : ep_prior,
            "mean_activity"     : ep_mean_act,
            "sleep_time_axis"   : ep_time,
            "sleep_z"           : ep_z,
            "sleep_rolled"      : ep_rolled,
            "filtered_traces"   : ep_filtered,
            "min_filtered"      : ep_min,
            "top_neuron_idx"    : ep_top_idx,
            "is_driven"         : is_driven,
            "selectivity_index" : si_safe,
            "unit_ids"          : unit_ids,
            "bin_size_s"        : BIN_SIZE,
            "sleep_duration_s"  : ep_dur_s,
            "session"           : SESSION,
            "epoch"             : ep_name,
        }
        out_pkl = REACTIVATION_DIR / f"{SESSION}_{ep_name}_prior.pkl"
        with open(out_pkl, "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n  Saved → {out_pkl}")

        # Diagnostic pipeline-stages figure (one per epoch)
        fig_pipe = plot_pipeline_stages(
            ep_z, ep_rolled, ep_filtered, ep_min,
            ep_mean_act, ep_prior, ep_time,
            top_neuron_idx=ep_top_idx,
            threshold_sd=2.0,
            window_s=300.0,
            save_path=str(REACTIVATION_DIR / f"{SESSION}_{ep_name}_pipeline_stages.png"),
        )
        plt.close(fig_pipe)

    print(f"\n{'='*65}")
    print("detect_reactivation.py  DONE")
    print(f"  PKLs saved to: {REACTIVATION_DIR}")
    print(f"  Run plot_reactivation.py to generate raster figures and statistics.")
    print(f"{'='*65}")