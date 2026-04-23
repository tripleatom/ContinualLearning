# Reproducing Sugden et al. 2020 Figure 2 from Spike Data

## Protocol for Adapting Calcium Imaging Analyses to Extracellular Electrophysiology

**Reference**: Sugden et al., "Cortical reactivations of recent sensory experiences predict bidirectional network changes during learning", *Nature Neuroscience* 23, 981-991 (2020).

**Adaptation context**: Your data uses NET electrodes recording **~150 sorted single units in V1** during a 2AFC object discrimination task in freely moving mice, rather than two-photon calcium imaging of ~260 neurons in area LI (head-fixed). **You do not have CA1 LFP**, so SWR-gated reactivation detection is not available — this protocol uses the DoG temporal prior approach (Sugden's primary method) throughout. Panel 2g (SWR-reactivation coupling) is replaced with a cortical population-burst validation. Because you have fewer neurons in a lower visual area, this protocol includes specific adaptations for maximizing classifier performance with limited V1 populations (see "Adapting for 150 V1 Units" section).

---

## Overview of Figure 2 Panels

| Panel | Content | What it shows |
|-------|---------|---------------|
| **a** | Mean cue response heatmap | Average firing rate time courses for all neurons, sorted by preferred stimulus |
| **b** | Single-trial task responses | Individual trial activity during cue presentations, sorted by preferred stimulus |
| **c** | Reactivation events in sleep | Activity around classifier-detected reactivation events during SWS |
| **d** | Functional connectivity clusters | Network diagram of cue-driven neuron clusters based on noise correlations |
| **e** | Classifier accuracy on task data | Fraction of cue presentations correctly classified (cross-validated) |
| **f** | False positive rate | Fraction of classifier-detected events surviving in shuffled data (critical without SWR gating) |
| **g** | Population burst validation | Confirm reactivations are brief synchronous transients (replaces SWR panel — no CA1 LFP) |

---

## Prerequisites and Data Format

### Expected inputs

```python
# Your data should be organized as:

# 1. Spike times per unit (sorted, merged, clean)
spike_times: dict  # {unit_id: np.array of spike times in seconds}

# 2. Task event timestamps
stim_onsets: dict   # {trial_idx: onset_time_s}
stim_labels: dict   # {trial_idx: object_id}  (e.g., 0='A', 1='B', 2='C', 3='D')
trial_outcomes: dict  # {trial_idx: 'correct'/'incorrect'}

# 3. Sleep/rest period timestamps
sleep_start: float  # seconds
sleep_end: float    # seconds
# IMPORTANT: Identify immobility/sleep periods from behavioral tracking
# (e.g., accelerometer, video tracking). Without CA1 LFP you cannot
# use SWS-specific markers like delta power, so use behavioral quiescence
# as your sleep/rest proxy.

# 4. Unit metadata
unit_region: dict   # {unit_id: 'V1'}  (V1 only in your case)
unit_channel: dict  # {unit_id: channel_number}
```

### Required packages

```bash
pip install numpy scipy matplotlib seaborn scikit-learn networkx --break-system-packages
```

---

## Panel 2a: Mean Cue Response Heatmap

### What the original shows
Mean activity time courses (ΔF/F₀) for all simultaneously recorded neurons in response to each of the visual cues (food, neutral, aversive), with neurons sorted by their preferred cue. Three columns (one per cue type), ~260 rows (neurons), color = response amplitude.

### Adaptation for spike data
Replace ΔF/F₀ with z-scored firing rate. The rationale is that raw firing rates vary across neurons by orders of magnitude (some fire at 1 Hz, others at 40 Hz), so z-scoring normalizes each neuron to its own baseline variability, making them comparable on a shared color scale — analogous to ΔF/F₀ normalization.

### Step-by-step procedure

**Step 1: Compute peri-stimulus time histograms (PSTHs) for every neuron x stimulus combination.**

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

def compute_psth(spike_times_unit, event_times, window=(-0.5, 2.0), bin_size=0.025):
    """
    Compute PSTH for one unit around a set of events.
    
    Parameters
    ----------
    spike_times_unit : np.array
        Spike times for a single unit (seconds).
    event_times : np.array
        Stimulus onset times (seconds).
    window : tuple
        (pre, post) relative to event onset.
    bin_size : float
        Bin width in seconds. 25 ms matches typical visual response timescales
        and approximates the ~33 ms frame rate of 2-photon at 30 Hz.
    
    Returns
    -------
    psth : np.array
        Mean firing rate per bin (Hz).
    time_bins : np.array
        Bin centers (seconds relative to event).
    trial_rates : np.array
        Per-trial firing rates, shape (n_trials, n_bins).
    """
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)
    n_trials = len(event_times)
    
    trial_rates = np.zeros((n_trials, n_bins))
    
    for i, t0 in enumerate(event_times):
        relative_spikes = spike_times_unit - t0
        counts, _ = np.histogram(relative_spikes, bins=edges)
        trial_rates[i] = counts / bin_size  # convert to Hz
    
    psth = trial_rates.mean(axis=0)
    return psth, centers, trial_rates
```

**Step 2: Z-score each neuron's PSTH relative to its baseline period.**

```python
def zscore_psth(psth, time_bins, baseline_window=(-0.5, 0.0), trial_rates=None):
    """
    Z-score the PSTH using baseline period statistics.
    
    Why z-score: Sugden et al. used ΔF/F₀ which inherently normalizes
    each neuron's activity relative to its baseline fluorescence.
    Z-scoring achieves the same goal for firing rates — it puts all
    neurons on a common scale where 0 = baseline and units = SDs 
    above/below baseline.
    
    If trial_rates is provided, compute baseline mean and std across
    all baseline bins of all trials (more robust estimate).
    """
    baseline_mask = (time_bins >= baseline_window[0]) & (time_bins < baseline_window[1])
    
    if trial_rates is not None:
        baseline_vals = trial_rates[:, baseline_mask].ravel()
    else:
        baseline_vals = psth[baseline_mask]
    
    mu = baseline_vals.mean()
    sigma = baseline_vals.std()
    
    if sigma < 1e-10:  # avoid division by zero for silent neurons
        return np.zeros_like(psth)
    
    return (psth - mu) / sigma
```

**Step 3: Assign preferred stimulus and sort neurons.**

```python
def assign_preferred_stimulus(unit_ids, spike_times, stim_onsets, stim_labels, 
                                n_stimuli=4, response_window=(0.0, 0.5)):
    """
    For each unit, determine which stimulus evokes the strongest response.
    
    Uses mean firing rate in the response_window across all trials of each
    stimulus type. This mirrors Sugden's approach of sorting neurons by
    the cue that evoked the largest response.
    
    Parameters
    ----------
    response_window : tuple
        Time window (seconds post-stimulus) to measure response magnitude.
        0-500 ms captures the transient visual onset response.
        Adjust if your stimuli are longer or responses are delayed.
    """
    preferred = {}
    mean_responses = {}  # store for secondary sorting
    
    for uid in unit_ids:
        stim_means = []
        for s in range(n_stimuli):
            trials_this_stim = [t for t, lbl in stim_labels.items() if lbl == s]
            onset_times = np.array([stim_onsets[t] for t in trials_this_stim])
            
            # Count spikes in response window per trial
            rates = []
            for t0 in onset_times:
                n_spk = np.sum(
                    (spike_times[uid] >= t0 + response_window[0]) &
                    (spike_times[uid] < t0 + response_window[1])
                )
                rates.append(n_spk / (response_window[1] - response_window[0]))
            stim_means.append(np.mean(rates))
        
        preferred[uid] = np.argmax(stim_means)
        mean_responses[uid] = stim_means
    
    # Sort: primary by preferred stimulus, secondary by response strength
    sorted_units = sorted(unit_ids, 
                          key=lambda u: (preferred[u], -mean_responses[u][preferred[u]]))
    
    return preferred, mean_responses, sorted_units
```

**Step 4: Build and plot the heatmap.**

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_fig2a(sorted_units, spike_times, stim_onsets, stim_labels, preferred,
               n_stimuli=4, stim_names=None, window=(-0.5, 2.0), bin_size=0.025):
    """
    Plot mean cue response heatmap (Figure 2a equivalent).
    
    Layout: n_stimuli columns, n_neurons rows.
    Color: z-scored firing rate (blue-white-red diverging colormap).
    """
    if stim_names is None:
        stim_names = [f'Object {i}' for i in range(n_stimuli)]
    
    n_neurons = len(sorted_units)
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)
    
    # Pre-compute z-scored PSTHs: shape (n_neurons, n_stimuli, n_bins)
    Z = np.zeros((n_neurons, n_stimuli, n_bins))
    
    for i, uid in enumerate(sorted_units):
        for s in range(n_stimuli):
            trials = [t for t, lbl in stim_labels.items() if lbl == s]
            onsets = np.array([stim_onsets[t] for t in trials])
            psth, tbins, trial_rates = compute_psth(spike_times[uid], onsets,
                                                     window=window, bin_size=bin_size)
            Z[i, s, :] = zscore_psth(psth, tbins, trial_rates=trial_rates)
    
    # Smooth temporally for cleaner visualization (sigma = 2 bins = 50 ms)
    for i in range(n_neurons):
        for s in range(n_stimuli):
            Z[i, s, :] = gaussian_filter1d(Z[i, s, :], sigma=2)
    
    # Plot
    fig, axes = plt.subplots(1, n_stimuli, figsize=(3 * n_stimuli, 8),
                              sharey=True)
    vmax = np.percentile(np.abs(Z), 97)  # symmetric color limits
    
    for s in range(n_stimuli):
        ax = axes[s]
        im = ax.imshow(Z[:, s, :], aspect='auto', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax,
                        extent=[window[0], window[1], n_neurons, 0])
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_title(stim_names[s])
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
        
        if s == 0:
            ax.set_ylabel('Neurons (sorted by preferred stimulus)')
        
        # Add horizontal lines separating groups
        group_boundaries = []
        for g in range(n_stimuli - 1):
            count = sum(1 for u in sorted_units if preferred[u] <= g)
            group_boundaries.append(count)
        for gb in group_boundaries:
            ax.axhline(gb, color='k', linewidth=0.5)
    
    plt.colorbar(im, ax=axes[-1], label='Z-scored firing rate', shrink=0.6)
    plt.tight_layout()
    return fig
```

---

## Panel 2b: Single-Trial Activity During Task

### What the original shows
Example deconvolved activity (scaled by mean activity per cell) across individual cue presentations during task performance. Each column is a single trial, rows are neurons sorted by preferred cue. Color = scaled activity.

### Adaptation for spike data

Instead of deconvolved ΔF/F₀, use the firing rate in the response window (0 to 500 ms post-stimulus), normalized per neuron (divided by that neuron's mean firing rate across all conditions). This produces a unitless "relative activation" analogous to Sugden's scaling by mean activity.

```python
def compute_single_trial_matrix(sorted_units, spike_times, stim_onsets, stim_labels,
                                 target_stim, response_window=(0.0, 0.5)):
    """
    Build the neuron x trial matrix for a given stimulus type.
    
    Returns
    -------
    mat : np.array, shape (n_neurons, n_trials_of_this_stim)
        Normalized firing rate per neuron per trial.
    """
    trials = sorted(t for t, lbl in stim_labels.items() if lbl == target_stim)
    onsets = np.array([stim_onsets[t] for t in trials])
    n_neurons = len(sorted_units)
    n_trials = len(trials)
    
    mat = np.zeros((n_neurons, n_trials))
    
    for i, uid in enumerate(sorted_units):
        # Compute per-trial firing rate
        for j, t0 in enumerate(onsets):
            n_spk = np.sum(
                (spike_times[uid] >= t0 + response_window[0]) &
                (spike_times[uid] < t0 + response_window[1])
            )
            mat[i, j] = n_spk / (response_window[1] - response_window[0])
        
        # Normalize by this neuron's overall mean rate (across all stimuli)
        # to make neurons comparable
        overall_mean = mat[i].mean()
        if overall_mean > 0:
            mat[i] /= overall_mean
    
    return mat, trials


def plot_fig2b(sorted_units, spike_times, stim_onsets, stim_labels, preferred,
               n_stimuli=4, stim_names=None):
    """
    Plot single-trial response heatmap for each stimulus type.
    """
    if stim_names is None:
        stim_names = [f'Object {i}' for i in range(n_stimuli)]
    
    fig, axes = plt.subplots(1, n_stimuli, figsize=(3 * n_stimuli, 8), sharey=True)
    
    all_mats = []
    for s in range(n_stimuli):
        mat, _ = compute_single_trial_matrix(sorted_units, spike_times,
                                              stim_onsets, stim_labels, s)
        all_mats.append(mat)
    
    vmax = np.percentile(np.abs(np.concatenate([m.ravel() for m in all_mats])), 97)
    
    for s in range(n_stimuli):
        ax = axes[s]
        ax.imshow(all_mats[s], aspect='auto', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax)
        ax.set_xlabel(f'Trials ({stim_names[s]})')
        ax.set_title(stim_names[s])
        if s == 0:
            ax.set_ylabel('Neurons (sorted by preferred stimulus)')
        
        # Group boundaries
        for g in range(n_stimuli - 1):
            count = sum(1 for u in sorted_units if preferred[u] <= g)
            ax.axhline(count, color='k', linewidth=0.5)
    
    plt.tight_layout()
    return fig
```

---

## Panel 2c: Reactivation Events During Sleep/Rest

### What the original shows
Same layout as 2b, but each column is now a classifier-detected reactivation event during darkness. The classifier probability trace is shown on top; below is the deconvolved activity of all neurons around each event (±0.5 s window).

### This is the most complex panel. It requires three sub-steps:
1. Train a population decoder on task data
2. Apply the decoder to sleep data to detect reactivation events
3. Visualize peri-event activity around detected events

---

### Sub-step C1: Train the reactivation classifier

Sugden et al. used an Averaged One-Dependence Estimator (AODE), an extension of Naive Bayes that accounts for pairwise dependencies between neurons. The rationale for AODE over standard Naive Bayes is that neurons in visual cortex have correlated noise (noise correlations), so the conditional independence assumption of Naive Bayes is violated. AODE approximates pairwise dependencies without the full covariance matrix (which would be underdetermined with ~300 neurons and ~200 trials).

For spike data, you have two practical options:

**Option A: Gaussian Naive Bayes (simpler, good starting point)**

The reason this can work despite its simplicity is that you have many more neurons (500+) than Sugden, so the redundancy from ignoring correlations is partially offset by the richer population vector. Start here to verify the pipeline works, then upgrade.

**Option B: AODE (faithful to the paper)**

The AODE averages over N "super-parent" Naive Bayes models, where each model conditions on one particular neuron in addition to the class label. This captures the first-order pairwise structure.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import numpy as np

class AODEClassifier:
    """
    Averaged One-Dependence Estimator for reactivation detection.
    
    For each 'parent' neuron x_i, builds a Naive Bayes model conditioned
    on (class, x_i), then averages predictions across all parents.
    
    This captures pairwise correlations between x_i and every other neuron,
    approximating the joint distribution better than standard Naive Bayes
    without needing the full covariance matrix.
    
    Reference: Webb et al., Mach Learn 65:251-273 (2005)
    Sugden implementation: github.com/asugden/pool
    """
    
    def __init__(self, min_frequency=30):
        """
        Parameters
        ----------
        min_frequency : int
            Minimum number of training samples in which a parent neuron
            must be active to be included. Sugden used 30.
            Rationale: parents with too few active samples give unreliable
            conditional distributions.
        """
        self.min_frequency = min_frequency
        self.models = []
        self.parent_indices = []
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : np.array, shape (n_samples, n_neurons)
            Population firing rate vectors during stimulus presentations.
        y : np.array, shape (n_samples,)
            Stimulus labels.
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        
        self.models = []
        self.parent_indices = []
        
        for i in range(n_features):
            # Check if this neuron is active often enough
            # "Active" = above median rate for that neuron
            active = X[:, i] > np.median(X[:, i])
            if active.sum() >= self.min_frequency:
                # Build NB model on remaining features, 
                # concatenated with the parent feature
                other_idx = [j for j in range(n_features) if j != i]
                X_augmented = np.column_stack([X[:, i:i+1], X[:, other_idx]])
                
                model = GaussianNB()
                model.fit(X_augmented, y)
                
                self.models.append(model)
                self.parent_indices.append(i)
        
        return self
    
    def predict_proba(self, X):
        """
        Average predicted probabilities across all parent models.
        
        Returns
        -------
        proba : np.array, shape (n_samples, n_classes)
        """
        n_features = X.shape[1]
        all_proba = []
        
        for model, parent_i in zip(self.models, self.parent_indices):
            other_idx = [j for j in range(n_features) if j != parent_i]
            X_augmented = np.column_stack([X[:, parent_i:parent_i+1], X[:, other_idx]])
            all_proba.append(model.predict_proba(X_augmented))
        
        # Average across all parent models
        return np.mean(all_proba, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
```

**Building training data from task trials:**

```python
def build_classifier_training_data(sorted_units, spike_times, stim_onsets, 
                                    stim_labels, n_stimuli=4,
                                    response_window=(0.0, 0.5), bin_size=0.05):
    """
    Build population vectors for classifier training.
    
    Each sample is one time bin during one stimulus presentation.
    
    Why bin_size=50ms: This is a compromise between temporal resolution
    and having enough spikes per bin to estimate firing rates reliably.
    Sugden used individual imaging frames (~33 ms at 30 Hz). For spikes,
    50 ms bins typically give 0-3 spikes per bin per neuron, which is
    sufficient for rate estimation.
    
    Returns
    -------
    X : np.array, shape (n_samples, n_neurons)
    y : np.array, shape (n_samples,)
    """
    n_neurons = len(sorted_units)
    edges_in_window = np.arange(response_window[0], response_window[1], bin_size)
    n_bins_per_trial = len(edges_in_window) - 1
    
    X_list = []
    y_list = []
    
    for trial_idx, stim_id in stim_labels.items():
        t0 = stim_onsets[trial_idx]
        
        for b in range(n_bins_per_trial):
            t_start = t0 + edges_in_window[b]
            t_end = t0 + edges_in_window[b + 1]
            
            pop_vec = np.zeros(n_neurons)
            for i, uid in enumerate(sorted_units):
                n_spk = np.sum(
                    (spike_times[uid] >= t_start) & (spike_times[uid] < t_end)
                )
                pop_vec[i] = n_spk / bin_size  # firing rate in Hz
            
            X_list.append(pop_vec)
            y_list.append(stim_id)
    
    return np.array(X_list), np.array(y_list)
```

---

### Sub-step C2: Apply temporal prior (DoG filter) — CRITICAL without CA1 LFP

This is the most critical step in your pipeline. Raw classifier output on sleep data picks up slow drifts in population state (e.g., transitions between behavioral states, slow oscillations in overall firing rate). The temporal prior biases detection toward **brief, transient, synchronous** population bursts — which is the cortical signature of replay events.

Without CA1 LFP to gate detection to SWR epochs, the DoG filter is your **primary defense against false positives**. This is exactly what Sugden used as their main analysis method — the SWR coupling was a secondary validation, not a requirement.

Sugden used a high-pass difference-of-Gaussians (DoG) filter applied to each neuron's activity before classification:

```python
from scipy.ndimage import gaussian_filter1d

def apply_temporal_prior(rate_traces, fs, sigmas_s=(0.133, 4.0, 16.0)):
    """
    Apply Sugden's difference-of-Gaussians temporal filter.
    
    Purpose: Suppress slow fluctuations and enhance brief transients.
    
    Parameters
    ----------
    rate_traces : np.array, shape (n_neurons, n_timebins)
        Continuous firing rate traces during sleep.
    fs : float
        Sampling rate of the rate traces (1/bin_size).
    sigmas_s : tuple
        Gaussian sigmas in seconds. Sugden used:
        - 0.133 s (narrow, ~4 imaging frames at 30 Hz): captures the reactivation
        - 4.0 s: captures 1-second fluctuations  
        - 16.0 s: captures slow drift
        The filter = narrow - min(broad1, broad2), which passes only
        events faster than ~1 s.
    
    Returns
    -------
    filtered : np.array, same shape as rate_traces
        Temporally filtered firing rate traces.
    """
    sigmas_bins = [s * fs for s in sigmas_s]
    
    # Narrow Gaussian (captures brief events)
    narrow = gaussian_filter1d(rate_traces, sigma=sigmas_bins[0], axis=1)
    
    # Two broad Gaussians (capture slow fluctuations)
    broad1 = gaussian_filter1d(rate_traces, sigma=sigmas_bins[1], axis=1)
    broad2 = gaussian_filter1d(rate_traces, sigma=sigmas_bins[2], axis=1)
    
    # High-pass: subtract the minimum of the broad filters
    # Using minimum across the two broad scales makes the filter robust
    # to fluctuations at multiple slow timescales
    slow_component = np.minimum(broad1, broad2)
    filtered = narrow - slow_component
    
    # The temporal prior modulates each neuron's activity trace independently
    # Negative values are clipped to 0 (reactivation = excess activity)
    filtered = np.maximum(filtered, 0)
    
    return filtered
```

---

### Sub-step C3: Detect reactivation events (DoG-filtered sliding window)

```python
def detect_reactivations(sorted_units, spike_times, sleep_start, sleep_end,
                          classifier, n_stimuli=4,
                          bin_size=0.05, threshold=0.5):
    """
    Detect reactivation events during sleep/rest using DoG-filtered
    sliding-window classification.
    
    This is Sugden's primary analysis method. Without CA1 LFP, we rely
    entirely on the DoG temporal prior to select brief synchronous
    transients and reject slow state changes.
    
    Parameters
    ----------
    threshold : float
        Classifier confidence threshold. Sugden used 0.1 for their AODE.
        For GaussianNB on spike data, start with 0.3-0.5 and adjust
        based on the false-positive analysis (Panel 2f).
        
        IMPORTANT: The right threshold is the one where your identity
        shuffle (Panel 2f) gives < 5% false positive rate. This will
        likely be higher than Sugden's 0.1 because without SWR gating
        you need the classifier alone to reject non-replay events.
    
    Returns
    -------
    reactivation_events : dict
        {stim_id: list of (time, confidence)}
    continuous_proba : np.array, shape (n_timebins, n_stimuli)
        Classifier output for all time bins (for visualization).
    rate_matrix : np.array, shape (n_neurons, n_timebins)
        Continuous firing rate matrix (needed for Panel 2f shuffle analysis).
    """
    n_neurons = len(sorted_units)
    time_bins_sleep = np.arange(sleep_start, sleep_end, bin_size)
    n_timebins = len(time_bins_sleep)
    
    # Build continuous rate matrix
    # NOTE: For large datasets this is the bottleneck. Consider using
    # np.histogram for each unit instead of the inner loop:
    rate_matrix = np.zeros((n_neurons, n_timebins))
    edges = np.append(time_bins_sleep, time_bins_sleep[-1] + bin_size)
    
    for i, uid in enumerate(sorted_units):
        # Vectorized: histogram is much faster than per-bin counting
        mask = (spike_times[uid] >= sleep_start) & (spike_times[uid] < sleep_end)
        counts, _ = np.histogram(spike_times[uid][mask], bins=edges)
        rate_matrix[i] = counts / bin_size
    
    # Apply DoG temporal prior — this is CRITICAL without SWR gating
    # It rejects slow state changes and selects for brief transients
    rate_matrix_filtered = apply_temporal_prior(rate_matrix, fs=1/bin_size)
    
    # Classify each time bin using the filtered rates
    continuous_proba = classifier.predict_proba(rate_matrix_filtered.T)
    
    # Detect peaks in classifier output
    from scipy.signal import find_peaks
    reactivation_events = {s: [] for s in range(n_stimuli)}
    
    for s in range(n_stimuli):
        # Minimum distance between events: ~1 s (20 bins at 50 ms)
        # Sugden found inter-reactivation intervals peaked around ~1 s
        # with some bursting at shorter intervals
        peaks, properties = find_peaks(
            continuous_proba[:, s], 
            height=threshold,
            distance=int(1.0 / bin_size)
        )
        for p in peaks:
            reactivation_events[s].append(
                (time_bins_sleep[p], continuous_proba[p, s])
            )
    
    return reactivation_events, continuous_proba, rate_matrix
```

---

### Sub-step C4: Plot Figure 2c

```python
def plot_fig2c(sorted_units, spike_times, preferred, reactivation_events,
               stim_id, n_stimuli=4, peri_window=(-0.5, 0.5), bin_size=0.025,
               max_events=15, stim_names=None):
    """
    Plot reactivation event rasters (Figure 2c equivalent).
    
    Layout:
    - Top row: classifier output trace around each event
    - Bottom: heatmap of neuron activity (rows) x events (columns)
    
    max_events : int
        Show the top N events by classifier confidence.
        Sugden showed the top 15 per session.
    """
    if stim_names is None:
        stim_names = [f'Object {i}' for i in range(n_stimuli)]
    
    events = reactivation_events[stim_id]
    if len(events) == 0:
        print(f"No reactivation events detected for stimulus {stim_id}")
        return None
    
    # Sort by confidence, take top events
    events_sorted = sorted(events, key=lambda x: -x[1])[:max_events]
    n_events = len(events_sorted)
    n_neurons = len(sorted_units)
    
    edges = np.arange(peri_window[0], peri_window[1] + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)
    
    # Build peri-event activity matrix
    # Shape: (n_neurons, n_events * n_bins) for a "filmstrip" layout
    # Or: (n_neurons, n_events) using mean rate in a short window
    
    # Filmstrip approach (matching Sugden's layout):
    activity = np.zeros((n_neurons, n_events * n_bins))
    
    for ev_idx, (ev_time, ev_conf) in enumerate(events_sorted):
        for i, uid in enumerate(sorted_units):
            psth, _, _ = compute_psth(spike_times[uid], np.array([ev_time]),
                                       window=peri_window, bin_size=bin_size)
            # Z-score using session-wide baseline
            overall_rate = len(spike_times[uid]) / (spike_times[uid].max() - spike_times[uid].min())
            overall_std = max(np.sqrt(overall_rate * bin_size) / bin_size, 0.1)
            z = (psth - overall_rate) / overall_std
            
            activity[i, ev_idx * n_bins:(ev_idx + 1) * n_bins] = z
    
    # Plot
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(12, n_events * 2), 8),
                                          height_ratios=[1, 6], sharex=True)
    
    # Top: classifier confidence per event
    for ev_idx, (ev_time, ev_conf) in enumerate(events_sorted):
        x_center = (ev_idx + 0.5) * n_bins
        ax_top.bar(x_center, ev_conf, width=n_bins * 0.8, color='green', alpha=0.7)
    ax_top.set_ylabel('P(reactivation)')
    ax_top.set_ylim(0, 1)
    ax_top.set_title(f'{stim_names[stim_id]} reactivations (top {n_events} by confidence)')
    
    # Bottom: neuron activity heatmap
    vmax = np.percentile(np.abs(activity), 95)
    ax_bot.imshow(activity, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax_bot.set_ylabel('Neurons (sorted by preferred stimulus)')
    ax_bot.set_xlabel('Time around reactivation events')
    
    # Add vertical lines between events
    for ev_idx in range(1, n_events):
        ax_bot.axvline(ev_idx * n_bins, color='gray', linewidth=0.5, linestyle='--')
    
    # Add horizontal lines between neuron groups
    for g in range(n_stimuli - 1):
        count = sum(1 for u in sorted_units if preferred[u] <= g)
        ax_bot.axhline(count, color='k', linewidth=0.5)
    
    plt.tight_layout()
    return fig
```

---

## Panel 2d: Functional Connectivity Clusters

### What the original shows
A network diagram where each node is a cue-driven neuron, node size reflects response probability, edge thickness reflects pairwise joint response probability (functional connectivity from noise correlations), and nodes are colored/grouped into clusters found by trial-by-trial noise correlation structure.

### Adaptation for spike data

Noise correlations from spike data are actually more standard than from calcium data. Compute trial-by-trial response variability, subtract the stimulus-driven component, then correlate residuals across neuron pairs.

```python
from scipy.stats import pearsonr
import networkx as nx

def compute_noise_correlations(sorted_units, spike_times, stim_onsets, stim_labels,
                                 n_stimuli=4, response_window=(0.0, 0.5)):
    """
    Compute pairwise noise correlations between all neuron pairs.
    
    Noise correlations measure correlated trial-to-trial variability
    AFTER removing the stimulus-driven mean. High noise correlations
    between two neurons suggest they share common input or are 
    synaptically connected.
    
    Procedure:
    1. For each neuron, compute firing rate per trial.
    2. For each stimulus condition, subtract the condition mean (this
       removes the signal correlation component).
    3. Concatenate residuals across conditions.
    4. Compute Pearson correlation of residuals between all pairs.
    
    Returns
    -------
    noise_corr_matrix : np.array, shape (n_neurons, n_neurons)
    """
    n_neurons = len(sorted_units)
    
    # Step 1: Get per-trial rates
    all_residuals = {i: [] for i in range(n_neurons)}
    
    for s in range(n_stimuli):
        trials = [t for t, lbl in stim_labels.items() if lbl == s]
        onsets = np.array([stim_onsets[t] for t in trials])
        
        # Firing rates for each neuron on each trial of this stimulus
        rates = np.zeros((n_neurons, len(trials)))
        for i, uid in enumerate(sorted_units):
            for j, t0 in enumerate(onsets):
                n_spk = np.sum(
                    (spike_times[uid] >= t0 + response_window[0]) &
                    (spike_times[uid] < t0 + response_window[1])
                )
                rates[i, j] = n_spk / (response_window[1] - response_window[0])
        
        # Step 2: Subtract condition mean
        condition_mean = rates.mean(axis=1, keepdims=True)
        residuals = rates - condition_mean
        
        for i in range(n_neurons):
            all_residuals[i].extend(residuals[i].tolist())
    
    # Step 3-4: Concatenate and correlate
    residual_matrix = np.array([all_residuals[i] for i in range(n_neurons)])
    
    noise_corr_matrix = np.corrcoef(residual_matrix)
    np.fill_diagonal(noise_corr_matrix, 0)
    
    return noise_corr_matrix


def cluster_neurons(noise_corr_matrix, sorted_units, preferred, n_stimuli=4):
    """
    Cluster cue-driven neurons using noise correlation structure.
    
    Sugden used a community detection algorithm (Louvain) on the graph 
    of positive noise correlations, treating neurons as nodes and 
    positive noise correlations as edge weights.
    
    Reference: Python-Louvain (community module), Blondel et al. 2008.
    """
    import community as community_louvain  # pip install python-louvain
    
    # Build graph from positive noise correlations
    G = nx.Graph()
    n = len(sorted_units)
    
    for i in range(n):
        G.add_node(i, preferred=preferred[sorted_units[i]])
    
    for i in range(n):
        for j in range(i + 1, n):
            if noise_corr_matrix[i, j] > 0:
                G.add_edge(i, j, weight=noise_corr_matrix[i, j])
    
    # Louvain community detection
    partition = community_louvain.best_partition(G, weight='weight')
    
    return G, partition


def plot_fig2d(G, partition, sorted_units, preferred, n_stimuli=4,
               stim_names=None, stim_colors=None):
    """
    Plot network diagram of cue-driven neuron clusters (Figure 2d equivalent).
    """
    if stim_names is None:
        stim_names = [f'Obj {i}' for i in range(n_stimuli)]
    if stim_colors is None:
        stim_colors = plt.cm.Set2(np.linspace(0, 1, n_stimuli))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Node colors by preferred stimulus
    node_colors = [stim_colors[preferred[sorted_units[n]]] for n in G.nodes()]
    
    # Node sizes by degree (proxy for response probability)
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 * degrees[n] / max_deg + 50 for n in G.nodes()]
    
    # Edge widths by weight (noise correlation)
    edges = G.edges(data=True)
    edge_weights = [d['weight'] * 3 for _, _, d in edges]
    
    # Layout using community structure
    pos = nx.spring_layout(G, weight='weight', k=2, iterations=100, seed=42)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, alpha=0.2,
                            edge_color='gray')
    
    # Legend for stimulus colors
    for s in range(n_stimuli):
        ax.scatter([], [], c=[stim_colors[s]], s=100, label=stim_names[s])
    ax.legend(loc='upper right')
    ax.set_title('Functional connectivity clusters (noise correlations)')
    ax.axis('off')
    
    plt.tight_layout()
    return fig
```

---

## Panel 2e: Classifier Cross-Validation Accuracy

### What the original shows
Fraction of cue presentations correctly identified by the classifier, as a function of classifier output threshold, evaluated with 2/3 - 1/3 train-test splits.

### Implementation

```python
def plot_fig2e(X_train, y_train, n_stimuli=4, n_splits=3, stim_names=None,
               stim_colors=None):
    """
    Cross-validated classifier performance (Figure 2e equivalent).
    
    Sugden trained on 2/3 of trials and tested on the remaining 1/3.
    This function uses stratified K-fold to give the same split ratio.
    
    The x-axis is the classifier output threshold (minimum confidence
    to count as a "classification"). The y-axis is accuracy among
    trials that passed that threshold.
    """
    if stim_names is None:
        stim_names = [f'Object {i}' for i in range(n_stimuli)]
    if stim_colors is None:
        stim_colors = ['green', 'blue', 'red', 'orange'][:n_stimuli]
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store per-stimulus accuracy at each threshold
    accuracies = {s: np.zeros((n_splits, len(thresholds))) for s in range(n_stimuli)}
    overall_acc = np.zeros((n_splits, len(thresholds)))
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
        clf = AODEClassifier()  # or GaussianNB()
        clf.fit(X_train[train_idx], y_train[train_idx])
        proba = clf.predict_proba(X_train[test_idx])
        
        for ti, thresh in enumerate(thresholds):
            for s in range(n_stimuli):
                mask = (y_train[test_idx] == s)
                if mask.sum() == 0:
                    continue
                preds = proba[mask]
                max_proba = preds.max(axis=1)
                above_thresh = max_proba >= thresh
                if above_thresh.sum() > 0:
                    predicted_class = np.argmax(preds[above_thresh], axis=1)
                    accuracies[s][fold, ti] = (predicted_class == s).mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in range(n_stimuli):
        mean_acc = accuracies[s].mean(axis=0)
        sem_acc = accuracies[s].std(axis=0) / np.sqrt(n_splits)
        ax.plot(thresholds, mean_acc, color=stim_colors[s], label=stim_names[s])
        ax.fill_between(thresholds, mean_acc - sem_acc, mean_acc + sem_acc,
                         color=stim_colors[s], alpha=0.2)
    
    ax.set_xlabel('Classifier output threshold')
    ax.set_ylabel('Fraction correctly identified')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_title('Classifier cross-validation (Panel 2e)')
    plt.tight_layout()
    return fig
```

---

## Panel 2f: False Positive Rate (Shuffle Controls)

### What the original shows
The fraction of classifier-identified events that survive in shuffled data (cell identity randomized, or time randomized). This validates that detected reactivation events are not spurious.

### Implementation

Sugden performed two types of shuffle:
1. **Identity shuffle**: At detected reactivation times, randomize which neuron each activity value belongs to. This destroys the spatial pattern while preserving temporal structure.
2. **Time shuffle**: Circularly shift each neuron's time course by a random amount. This preserves each neuron's autocorrelation but destroys inter-neuron temporal alignment.

```python
def shuffle_identity(rate_matrix, n_shuffles=1000):
    """
    Shuffle neuron identities at each time bin independently.
    For each time point, randomly permute which neuron has which rate.
    """
    shuffled_matrices = []
    for _ in range(n_shuffles):
        m = rate_matrix.copy()
        for t in range(m.shape[1]):
            np.random.shuffle(m[:, t])
        shuffled_matrices.append(m)
    return shuffled_matrices


def shuffle_time_circular(rate_matrix, n_shuffles=1000):
    """
    Circularly shift each neuron's time course by a random offset.
    Preserves autocorrelation within each neuron but breaks 
    inter-neuron temporal alignment.
    """
    n_neurons, n_timebins = rate_matrix.shape
    shuffled_matrices = []
    for _ in range(n_shuffles):
        m = np.zeros_like(rate_matrix)
        for i in range(n_neurons):
            shift = np.random.randint(0, n_timebins)
            m[i] = np.roll(rate_matrix[i], shift)
        shuffled_matrices.append(m)
    return shuffled_matrices


def compute_false_positive_rate(classifier, rate_matrix_sleep, 
                                 reactivation_events, threshold,
                                 shuffle_type='identity', n_shuffles=100):
    """
    Compute false positive rate for reactivation detection.
    
    Procedure:
    1. For each shuffle, apply classifier to shuffled data at the same
       time points where real reactivations were detected.
    2. Count how many shuffled time points still exceed threshold.
    3. FP rate = (fraction of real events found in shuffled data).
    
    Returns
    -------
    fp_rates : dict
        {stim_id: fraction of real events surviving shuffle}
    """
    if shuffle_type == 'identity':
        shuffled = shuffle_identity(rate_matrix_sleep, n_shuffles)
    else:
        shuffled = shuffle_time_circular(rate_matrix_sleep, n_shuffles)
    
    n_stimuli = classifier.predict_proba(rate_matrix_sleep[:, :1].T).shape[1]
    
    fp_counts = {s: 0 for s in range(n_stimuli)}
    total_events = {s: len(evs) for s, evs in reactivation_events.items()}
    
    for shuf_mat in shuffled:
        proba = classifier.predict_proba(shuf_mat.T)
        for s in range(n_stimuli):
            fp_counts[s] += (proba[:, s].max() >= threshold)
    
    fp_rates = {s: fp_counts[s] / (n_shuffles * max(total_events[s], 1)) 
                for s in range(n_stimuli)}
    
    return fp_rates


def plot_fig2f(classifier, rate_matrix_sleep, reactivation_events,
               n_stimuli=4, stim_names=None, stim_colors=None):
    """
    Plot false positive rate across thresholds (Figure 2f equivalent).
    """
    if stim_names is None:
        stim_names = [f'Object {i}' for i in range(n_stimuli)]
    if stim_colors is None:
        stim_colors = ['green', 'blue', 'red', 'orange'][:n_stimuli]
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for s in range(n_stimuli):
        fp_at_thresh = []
        for thresh in thresholds:
            fp = compute_false_positive_rate(
                classifier, rate_matrix_sleep, reactivation_events,
                thresh, shuffle_type='identity', n_shuffles=50
            )
            fp_at_thresh.append(fp[s])
        ax.plot(thresholds, fp_at_thresh, color=stim_colors[s], label=stim_names[s])
    
    ax.set_xlabel('Classifier output threshold')
    ax.set_ylabel('Fraction of events found in shuffled data')
    ax.legend()
    ax.set_title('False positive rate (identity shuffle, Panel 2f)')
    ax.axhline(0.05, color='gray', linestyle='--', label='5% FP')
    plt.tight_layout()
    return fig
```

---

## Panel 2g: Population Burst Validation (Replaces SWR Coupling)

### Why this replaces the original Panel 2g
Sugden's Panel 2g showed that cortical reactivation events are temporally coupled to hippocampal sharp-wave ripples, validating that they are replay events rather than noise. **Without CA1 LFP, you cannot reproduce this directly.** Instead, you validate that your detected reactivations are genuine brief population synchrony events using cortical data alone.

### What to show instead
Three complementary validations, any of which strengthens your claim:

1. **Peri-reactivation population firing rate**: Mean population firing rate (summed across all neurons) aligned to reactivation event times. Real reactivations should show a sharp, brief peak at t=0 that rises well above baseline — confirming they are synchronous population transients, not classifier artifacts from slow drift.

2. **Inter-reactivation interval distribution**: Sugden found that inter-event intervals peaked around ~1 s with a long tail, consistent with SWR-associated bursting. If your events show a similar distribution (not uniform/random), this supports their biological origin.

3. **Reactivation-triggered multi-unit activity (MUA)**: If you have access to the raw broadband signal from your NETs, compute MUA (high-pass filtered > 300 Hz, rectified, smoothed) and show it peaks at reactivation times.

```python
def validate_reactivations_no_lfp(sorted_units, spike_times, 
                                    reactivation_events, n_stimuli=4,
                                    window=(-2.0, 2.0), bin_size=0.025):
    """
    Validate reactivation events using cortical population activity.
    Replaces Panel 2g (SWR coupling) when no CA1 LFP is available.
    
    Returns three validation metrics:
    1. Peri-event population rate (should show sharp peak at t=0)
    2. Inter-event interval distribution (should not be uniform)
    3. Per-stimulus event statistics
    """
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)
    
    results = {}
    
    for s in range(n_stimuli):
        events = reactivation_events[s]
        if len(events) < 5:
            continue
        
        event_times = np.array([ev[0] for ev in events])
        
        # 1. Peri-event population rate
        pop_rate_snippets = []
        for ev_time, _ in events:
            total_rate = np.zeros(n_bins)
            for uid in sorted_units:
                counts, _ = np.histogram(
                    spike_times[uid] - ev_time, bins=edges
                )
                total_rate += counts / bin_size
            pop_rate_snippets.append(total_rate)
        
        pop_rate_snippets = np.array(pop_rate_snippets)
        
        # Z-score relative to flanks (|t| > 1.5 s)
        baseline_mask = np.abs(centers) > 1.5
        bl_mean = pop_rate_snippets[:, baseline_mask].mean()
        bl_std = pop_rate_snippets[:, baseline_mask].std()
        pop_rate_z = (pop_rate_snippets - bl_mean) / bl_std
        
        # 2. Inter-event intervals
        sorted_times = np.sort(event_times)
        ieis = np.diff(sorted_times)
        
        results[s] = {
            'time_axis': centers,
            'pop_rate_mean': pop_rate_z.mean(axis=0),
            'pop_rate_sem': pop_rate_z.std(axis=0) / np.sqrt(len(events)),
            'inter_event_intervals': ieis,
            'n_events': len(events),
        }
    
    return results


def plot_fig2g_validation(validation_results, n_stimuli=4, 
                           stim_names=None, stim_colors=None):
    """
    Plot population burst validation (Panel 2g replacement).
    
    Left: Peri-reactivation population firing rate (z-scored).
          Expected: sharp transient peak at t=0.
    Right: Inter-reactivation interval distribution.
           Expected: peak around 1 s, long tail (not uniform).
    """
    if stim_names is None:
        stim_names = [f'Object {i}' for i in range(n_stimuli)]
    if stim_colors is None:
        stim_colors = ['green', 'blue', 'red', 'orange'][:n_stimuli]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Peri-event population rate
    for s in range(n_stimuli):
        if s not in validation_results:
            continue
        r = validation_results[s]
        ax1.plot(r['time_axis'], r['pop_rate_mean'], 
                 color=stim_colors[s], label=f"{stim_names[s]} (n={r['n_events']})")
        ax1.fill_between(r['time_axis'],
                          r['pop_rate_mean'] - r['pop_rate_sem'],
                          r['pop_rate_mean'] + r['pop_rate_sem'],
                          color=stim_colors[s], alpha=0.2)
    
    ax1.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Time relative to reactivation (s)')
    ax1.set_ylabel('Z-scored population rate')
    ax1.set_title('Peri-reactivation population activity')
    ax1.legend(fontsize=8)
    
    # Right: Inter-event interval distribution
    all_ieis = []
    for s in range(n_stimuli):
        if s not in validation_results:
            continue
        ieis = validation_results[s]['inter_event_intervals']
        ax2.hist(ieis, bins=np.arange(0, 30, 1), alpha=0.5,
                 color=stim_colors[s], label=stim_names[s], density=True)
        all_ieis.extend(ieis.tolist())
    
    ax2.set_xlabel('Inter-reactivation interval (s)')
    ax2.set_ylabel('Density')
    ax2.set_title('Inter-event interval distribution')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    return fig
```

---

## Adapting for 150 V1 Units

Sugden had ~260 neurons in area LI. You have ~150 in V1. This section addresses the concrete challenges this creates and provides alternative approaches ranked by what to try first.

### The fundamental problem

Classifier-based reactivation detection depends on the population vector being discriminative — meaning different stimuli produce sufficiently different patterns of activity across your neurons. With 150 V1 units, two compounding factors reduce discriminability: (1) fewer neurons means a lower-dimensional population vector, and (2) V1 neurons are tuned to low-level features (orientation, spatial frequency, contrast) rather than object identity, so many neurons may respond similarly to different objects if those objects share low-level statistics.

### Strategy 1: Reduce classification to 2-way (RECOMMENDED)

Your task is naturally a pair discrimination: objects A vs. B at place P1, objects C vs. D at place P2. Train separate binary classifiers for each pair rather than a single 4-way classifier.

**Why this helps**: A 2-way classifier needs to find ONE decision boundary in population space; a 4-way classifier needs to find boundaries separating all four classes simultaneously. With 150 V1 neurons, there may be enough information to separate A from B but not enough to simultaneously separate all four objects. Chance is 50% (vs. 25% for 4-way), so the signal-to-noise for detecting above-chance performance is better.

**For reactivation detection**: Run the A-vs-B classifier on sleep data separately from the C-vs-D classifier. A reactivation event for "object A" is detected when the A-vs-B classifier outputs high confidence for A during a transient population burst.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def build_pairwise_classifiers(sorted_units, spike_times, stim_onsets, 
                                 stim_labels, pairs=[(0,1), (2,3)],
                                 response_window=(0.0, 0.5), bin_size=0.05):
    """
    Train separate binary classifiers for each stimulus pair.
    
    Parameters
    ----------
    pairs : list of tuples
        Each tuple is (stim_id_A, stim_id_B) for one discrimination pair.
        Default: [(0,1), (2,3)] = objects A vs B, C vs D.
    
    Returns
    -------
    classifiers : dict
        {(stim_A, stim_B): fitted classifier}
    training_data : dict
        {(stim_A, stim_B): (X, y)} for cross-validation
    """
    classifiers = {}
    training_data = {}
    
    for pair in pairs:
        sA, sB = pair
        # Select trials for this pair only
        pair_labels = {t: lbl for t, lbl in stim_labels.items() if lbl in pair}
        
        X, y = build_classifier_training_data(
            sorted_units, spike_times, stim_onsets, pair_labels,
            n_stimuli=max(pair) + 1,  # keeps indexing consistent
            response_window=response_window, bin_size=bin_size
        )
        # Filter to only include this pair's labels
        mask = np.isin(y, pair)
        X, y = X[mask], y[mask]
        # Relabel to 0/1 for binary classification
        y_binary = (y == sB).astype(int)
        
        # LDA is better than NB for small populations — see explanation below
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(X, y_binary)
        
        classifiers[pair] = clf
        training_data[pair] = (X, y_binary)
    
    return classifiers, training_data


def detect_reactivations_pairwise(sorted_units, spike_times, sleep_start, 
                                    sleep_end, classifiers, pairs,
                                    bin_size=0.05, threshold=0.7):
    """
    Detect reactivations using pairwise classifiers.
    
    For each time bin, each pair classifier independently reports
    confidence for its two stimuli. A reactivation for stimulus A
    is detected when the A-vs-B classifier has high confidence for A
    AND the population is in a transient burst state.
    
    threshold : float
        For binary classifiers, use a higher threshold (0.7-0.8)
        than you would for multi-class. The reason: binary classifiers
        always assign some probability to each class, so even noise
        gets ~0.5. You need events well above 0.5 to be meaningful.
    """
    n_neurons = len(sorted_units)
    time_bins = np.arange(sleep_start, sleep_end, bin_size)
    n_timebins = len(time_bins)
    
    # Build and filter rate matrix
    rate_matrix = np.zeros((n_neurons, n_timebins))
    edges = np.append(time_bins, time_bins[-1] + bin_size)
    for i, uid in enumerate(sorted_units):
        mask = (spike_times[uid] >= sleep_start) & (spike_times[uid] < sleep_end)
        counts, _ = np.histogram(spike_times[uid][mask], bins=edges)
        rate_matrix[i] = counts / bin_size
    
    rate_matrix_filtered = apply_temporal_prior(rate_matrix, fs=1/bin_size)
    
    # Run each pairwise classifier
    from scipy.signal import find_peaks
    reactivation_events = {}
    
    for pair in pairs:
        clf = classifiers[pair]
        proba = clf.predict_proba(rate_matrix_filtered.T)
        
        for class_idx, stim_id in enumerate(pair):
            peaks, _ = find_peaks(
                proba[:, class_idx],
                height=threshold,
                distance=int(1.0 / bin_size)
            )
            reactivation_events[stim_id] = [
                (time_bins[p], proba[p, class_idx]) for p in peaks
            ]
    
    return reactivation_events, rate_matrix
```

### Strategy 2: Use temporal features (multiply effective dimensionality)

Instead of classifying a single 150-dimensional population vector per time bin, use the firing rate profile across multiple time bins as features. This effectively turns your 150 neurons into a 150 x T dimensional feature vector, where T is the number of time bins in the response window.

```python
def build_temporal_training_data(sorted_units, spike_times, stim_onsets,
                                   stim_labels, n_stimuli=4,
                                   response_window=(0.0, 0.5), 
                                   bin_size=0.05):
    """
    Build training data using temporal firing rate profiles.
    
    Instead of a single rate per neuron per trial, this uses 
    the firing rate in each time bin, concatenated across neurons.
    
    With 150 neurons and 10 bins (50ms bins over 500ms window),
    each trial becomes a 1500-dimensional vector. This captures
    response latency and dynamics, which differ between objects
    even if peak rates are similar.
    
    IMPORTANT: This larger feature space needs regularization.
    Use LDA with shrinkage or logistic regression with L2 penalty,
    NOT GaussianNB (which would treat each bin independently).
    """
    n_neurons = len(sorted_units)
    bin_edges = np.arange(response_window[0], response_window[1] + bin_size, bin_size)
    n_bins = len(bin_edges) - 1
    
    X_list = []
    y_list = []
    
    for trial_idx, stim_id in stim_labels.items():
        t0 = stim_onsets[trial_idx]
        
        # Build temporal profile: (n_neurons * n_bins,) vector
        trial_vec = np.zeros(n_neurons * n_bins)
        for i, uid in enumerate(sorted_units):
            for b in range(n_bins):
                t_start = t0 + bin_edges[b]
                t_end = t0 + bin_edges[b + 1]
                n_spk = np.sum(
                    (spike_times[uid] >= t_start) & (spike_times[uid] < t_end)
                )
                trial_vec[i * n_bins + b] = n_spk / bin_size
        
        X_list.append(trial_vec)
        y_list.append(stim_id)
    
    return np.array(X_list), np.array(y_list)
```

**Caveat for reactivation detection**: When applying a temporal-feature classifier to sleep data, you need to extract the same temporal profile around each candidate time point. This means your sliding window now extracts 500 ms of data per step, which reduces temporal resolution. For brief replay events (~50-100 ms), the temporal profile approach may actually be counterproductive because the replay is compressed in time and won't match the full 500 ms task-evoked profile. Use this for Panel 2e (to check if your population carries object information at all) but consider falling back to instantaneous population vectors for Panel 2c reactivation detection.

### Strategy 3: Use LDA instead of Naive Bayes

With 150 neurons, Linear Discriminant Analysis (LDA) with automatic shrinkage is likely your best classifier. Here's why:

GaussianNB assumes neurons are independent given the class label. This assumption becomes increasingly harmful with fewer neurons, because the classifier can't compensate for shared noise by using correlated neurons together.

AODE partially fixes this by modeling pairwise dependencies, but with only 150 neurons it trains 150 sub-models, each conditioned on one parent neuron. Many of these parent neurons will have weak object selectivity (they're V1 neurons), making the sub-models noisy.

LDA directly estimates the shared covariance matrix and finds the linear projections that maximize class separation. With shrinkage (regularization toward diagonal covariance), it handles the case where you have more features than samples gracefully. It's essentially the optimal linear classifier when the noise is approximately Gaussian — which is a reasonable approximation for firing rate vectors.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Option A: LDA with automatic shrinkage (RECOMMENDED for 150 units)
# solver='lsqr' allows shrinkage; 'auto' finds optimal regularization
clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# Option B: L2-regularized logistic regression (good alternative)
# C controls regularization; smaller C = more regularization
clf_lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', 
                             multi_class='multinomial', max_iter=1000)

# Both have .predict_proba() so they drop into the pipeline unchanged
```

### Strategy 4: Pre-select informative neurons

Not all 150 V1 neurons will carry object information. Some may respond identically to all objects (e.g., pure orientation-tuned neurons that happen to see similar orientations in all objects). Including non-informative neurons adds noise without signal, degrading classifier performance.

```python
from scipy.stats import f_oneway

def select_informative_units(sorted_units, spike_times, stim_onsets, 
                               stim_labels, n_stimuli=4,
                               response_window=(0.0, 0.5), 
                               alpha=0.05):
    """
    Select neurons with significant stimulus selectivity using one-way
    ANOVA across stimulus conditions.
    
    Returns the subset of units where firing rate differs significantly
    across at least some stimuli. With 150 V1 units, you might find
    that 50-80 are significantly selective — this is fine, the 
    classifier will work better with 80 informative neurons than 
    150 neurons where 70 are noise.
    
    alpha : float
        Significance threshold. Use 0.05 (uncorrected) rather than
        Bonferroni-corrected, because you'd rather include a few
        false positives than miss real signal with 150 neurons.
    """
    informative = []
    p_values = {}
    
    for uid in sorted_units:
        rates_by_stim = []
        for s in range(n_stimuli):
            trials = [t for t, lbl in stim_labels.items() if lbl == s]
            trial_rates = []
            for t_idx in trials:
                t0 = stim_onsets[t_idx]
                n_spk = np.sum(
                    (spike_times[uid] >= t0 + response_window[0]) &
                    (spike_times[uid] < t0 + response_window[1])
                )
                trial_rates.append(n_spk / (response_window[1] - response_window[0]))
            rates_by_stim.append(trial_rates)
        
        # One-way ANOVA across stimulus conditions
        F, p = f_oneway(*rates_by_stim)
        p_values[uid] = p
        
        if p < alpha:
            informative.append(uid)
    
    print(f"Selected {len(informative)}/{len(sorted_units)} informative units "
          f"(ANOVA p < {alpha})")
    
    return informative, p_values
```

### Decision tree: which strategy to try first

```
1. Run Panel 2e with ALL 150 units + GaussianNB (4-way)
   |
   |-- If accuracy >> chance (>40% for 4-way): 
   |   Great, proceed with full pipeline. 
   |   Consider upgrading to LDA for better sensitivity.
   |
   |-- If accuracy is marginal (30-40%):
   |   Try pre-selecting informative neurons (Strategy 4), 
   |   then re-run with LDA (Strategy 3).
   |
   |-- If accuracy is near chance (<30%):
   |   Switch to pairwise classifiers (Strategy 1).
   |   Also try temporal features (Strategy 2) to confirm 
   |   your V1 population carries object info at all.
   |   If pairwise accuracy is also near chance, your V1 
   |   population may genuinely not carry enough object 
   |   discriminative information for this analysis approach.
```

### What if nothing works? (honest assessment)

If 150 V1 neurons truly cannot discriminate your objects above chance, that itself is an informative result — it would suggest that object identity is represented downstream of V1 in your circuit (which is consistent with the visual hierarchy). In that case, the Sugden-style reactivation analysis isn't the right approach for V1 specifically, and you might instead look at:

- Whether V1 reactivations carry low-level feature information (orientation, spatial frequency) even if they don't carry object identity
- Cross-day stability of V1 population geometry (manifold analysis from your grant) without needing the classifier-based reactivation detection
- Multi-unit correlation structure changes across learning (Panel 2d-style noise correlation analysis still works regardless of classifier performance)

---

## Complete Pipeline: Putting It All Together

```python
def run_full_fig2_pipeline(spike_times, unit_ids, stim_onsets, stim_labels,
                            sleep_start, sleep_end,
                            n_stimuli=4, stim_names=None, output_dir='.'):
    """
    Master function to run the entire Figure 2 reproduction pipeline.
    
    No CA1 LFP required. Uses DoG temporal prior for reactivation
    detection and population burst validation in place of SWR coupling.
    
    Call this after you have loaded and preprocessed your data.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Panel 2a: Mean cue responses ---
    print("=== Panel 2a: Computing mean cue responses ===")
    preferred, mean_responses, sorted_units = assign_preferred_stimulus(
        unit_ids, spike_times, stim_onsets, stim_labels, n_stimuli
    )
    fig_2a = plot_fig2a(sorted_units, spike_times, stim_onsets, stim_labels,
                         preferred, n_stimuli, stim_names)
    fig_2a.savefig(os.path.join(output_dir, 'fig2a_mean_responses.pdf'), dpi=300)
    
    # --- Panel 2b: Single-trial task responses ---
    print("=== Panel 2b: Plotting single-trial responses ===")
    fig_2b = plot_fig2b(sorted_units, spike_times, stim_onsets, stim_labels,
                         preferred, n_stimuli, stim_names)
    fig_2b.savefig(os.path.join(output_dir, 'fig2b_single_trial.pdf'), dpi=300)
    
    # --- Train classifier ---
    print("=== Training reactivation classifier ===")
    X_train, y_train = build_classifier_training_data(
        sorted_units, spike_times, stim_onsets, stim_labels, n_stimuli
    )
    classifier = AODEClassifier(min_frequency=30)
    classifier.fit(X_train, y_train)
    
    # --- Panel 2c: Detect and plot reactivations ---
    print("=== Panel 2c: Detecting reactivation events (DoG-filtered) ===")
    reactivation_events, continuous_proba, rate_matrix_sleep = detect_reactivations(
        sorted_units, spike_times, sleep_start, sleep_end,
        classifier, n_stimuli
    )
    for s in range(n_stimuli):
        n_ev = len(reactivation_events[s])
        print(f"    Stimulus {s}: {n_ev} reactivation events")
        if n_ev > 0:
            fig_2c = plot_fig2c(sorted_units, spike_times, preferred,
                                reactivation_events, s, n_stimuli,
                                stim_names=stim_names)
            fig_2c.savefig(os.path.join(output_dir, f'fig2c_reactivations_stim{s}.pdf'), dpi=300)
    
    # --- Panel 2d: Functional connectivity ---
    print("=== Panel 2d: Computing noise correlations and clusters ===")
    noise_corr = compute_noise_correlations(
        sorted_units, spike_times, stim_onsets, stim_labels, n_stimuli
    )
    G, partition = cluster_neurons(noise_corr, sorted_units, preferred, n_stimuli)
    fig_2d = plot_fig2d(G, partition, sorted_units, preferred, n_stimuli, stim_names)
    fig_2d.savefig(os.path.join(output_dir, 'fig2d_connectivity.pdf'), dpi=300)
    
    # --- Panel 2e: Classifier cross-validation ---
    print("=== Panel 2e: Classifier cross-validation ===")
    fig_2e = plot_fig2e(X_train, y_train, n_stimuli, stim_names=stim_names)
    fig_2e.savefig(os.path.join(output_dir, 'fig2e_classifier_accuracy.pdf'), dpi=300)
    
    # --- Panel 2f: False positive rate (CRITICAL without SWR gating) ---
    print("=== Panel 2f: Computing false positive rates ===")
    print("    (This is your main validation without CA1 LFP)")
    fig_2f = plot_fig2f(classifier, rate_matrix_sleep, reactivation_events,
                         n_stimuli, stim_names)
    fig_2f.savefig(os.path.join(output_dir, 'fig2f_false_positive.pdf'), dpi=300)
    
    # --- Panel 2g: Population burst validation (replaces SWR coupling) ---
    print("=== Panel 2g: Population burst validation ===")
    print("    (Replaces SWR coupling — validating events are real transients)")
    validation = validate_reactivations_no_lfp(
        sorted_units, spike_times, reactivation_events, n_stimuli
    )
    fig_2g = plot_fig2g_validation(validation, n_stimuli, stim_names)
    fig_2g.savefig(os.path.join(output_dir, 'fig2g_population_validation.pdf'), dpi=300)
    
    print(f"\n=== Pipeline complete. All figures saved to {output_dir} ===")
    
    return {
        'sorted_units': sorted_units,
        'preferred': preferred,
        'classifier': classifier,
        'reactivation_events': reactivation_events,
        'noise_correlations': noise_corr,
        'rate_matrix_sleep': rate_matrix_sleep,
        'validation': validation,
    }
```

---

## Key Differences from Sugden to Keep in Mind

### 1. No CA1 LFP — implications throughout
Without hippocampal recording, you cannot: (a) confirm reactivations co-occur with SWRs, (b) use SWR-gated detection, or (c) do the closed-loop SWR disruption that your grant describes for later aims. For this figure, the DoG temporal prior + shuffle controls (Panel 2f) + population burst validation (Panel 2g replacement) collectively substitute for SWR coupling. When you present this, be upfront that you're detecting "cortical reactivation events" rather than "SWR-coupled reactivations" — the mechanistic link to hippocampal replay is inferred from the literature rather than directly demonstrated in your data.

### 2. Temporal resolution advantage
Your spike data has millisecond resolution vs. Sugden's ~33 ms calcium imaging frames. This means your reactivation events will be more precisely timed. Use this advantage by keeping bin sizes small (25-50 ms) for visualization.

### 3. No deconvolution needed
Sugden had to deconvolve calcium traces to estimate spike rates, introducing temporal smearing and false negatives. Your spike data skips this entirely. Where they reference "deconvolved activity", you use firing rates directly.

### 4. Four stimuli instead of three
Sugden had food/neutral/aversive (3 classes). You have 4 objects (A, B, C, D), so your classifier has a harder 4-way discrimination problem. Expect slightly lower per-class accuracy, but with 500+ neurons this should still work well. Your stimuli don't have inherent valence differences (all are rewarded via correct discrimination), so the reactivation rate bias toward "salient" cues that Sugden observed may manifest differently — perhaps biased toward more recently learned or more difficult discriminations instead.

### 5. Freely moving adds complexity
Movement artifacts and theta-state modulation of firing rates are present in your data but not in Sugden's head-fixed mice. For the sleep/rest reactivation analysis, **strictly restrict to periods of confirmed immobility** (accelerometer/EMG/video tracking). This is even more important without CA1 LFP, because theta-state population activity during locomotion can produce patterns the classifier might misinterpret. For the task analysis, consider adding speed as a covariate in your firing rate normalization if objects are encountered at different running speeds.

### 6. Place-specific stimuli
Your objects appear at specific locations (P1, P2), so some V1 neurons may show place-modulated responses (running speed, head direction, or spatial view). Since you only have V1 (no dedicated place cell recording), consider including place as a factor in your trial labels (8 conditions: 4 objects x 2 places) if you want to capture place-object conjunctions, or collapse across places if you want to focus on object identity.

### 7. V1 vs. LI
Sugden recorded from lateral visual association cortex (area LI), which is a higher visual area with more complex selectivity and stronger reward modulation. V1 neurons are more orientation/spatial-frequency tuned and may show weaker object selectivity. This could reduce classifier accuracy and reactivation detection sensitivity. If your objects are complex enough (not just oriented gratings), V1 populations should still carry discriminative information, but you may need more neurons or a more sensitive classifier. Consider that V1 reactivations might be more "stimulus-like" (recapitulating low-level features) while LI reactivations were more "association-like" (incorporating valence) — this is actually an interesting scientific distinction to discuss.

---

## Recommended Parameter Tuning Order

1. **Start with GaussianNB** instead of AODE to verify the basic pipeline works end-to-end.
2. **Verify classifier accuracy first** (Panel 2e). If cross-validated accuracy on task data is below ~60%, your V1 population may not carry enough object-discriminative information. Consider: increasing the response window, using more bins per trial, or checking that your objects evoke distinct V1 responses.
3. **Set classifier threshold to 0.3 initially**, then run the Panel 2f shuffle analysis. Adjust the threshold until your identity-shuffle false positive rate drops below 5%. Without SWR gating, you will likely need a threshold of 0.3-0.6 (higher than Sugden's 0.1).
4. **Check the DoG filter sigmas.** The default sigmas (0.133s, 4s, 16s) were tuned for 30 Hz calcium imaging. For 50 ms binned spike data (20 Hz effective rate), the narrow sigma may need adjustment. Try sigmas = (0.1, 2.0, 8.0) if the default misses events.
5. **Validate with Panel 2g replacement.** Confirm that detected events show a sharp population rate peak at t=0. If the peak is broad (>500 ms), your DoG filter is too permissive.
6. **Upgrade to AODE** once the basic pipeline works. AODE should improve detection specificity by capturing pairwise correlations between neurons.
7. **Iterate on sleep period definition.** Without CA1 LFP to identify SWS specifically, your results depend heavily on correctly identifying immobility periods. Try multiple immobility criteria (e.g., 5s vs. 30s of no movement) and check if reactivation rates are stable.

---

## Software Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.20 | Core computation |
| scipy | >= 1.7 | Signal processing, statistics |
| scikit-learn | >= 1.0 | GaussianNB, cross-validation |
| matplotlib | >= 3.5 | Plotting |
| seaborn | >= 0.12 | Optional enhanced plots |
| networkx | >= 2.6 | Graph construction (Panel 2d) |
| python-louvain | >= 0.16 | Community detection (Panel 2d) |

```bash
pip install numpy scipy scikit-learn matplotlib seaborn networkx python-louvain --break-system-packages
```
