"""
Shared utilities for Grating neural data analysis.

Provides data loading, feature extraction, orientation selectivity,
and common visualization helpers used by GratingLDA and GratingSVM.
"""
import csv
import platform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import h5py
import json
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SESSION PATH RESOLUTION
# =============================================================================

def load_session_paths(animal_id, experiment_date, log_dir=None):
    """
    Resolve rec_folder and task_file_paths for a session from the experiment log CSV.

    Parameters
    ----------
    animal_id : str
        Animal identifier, e.g. "CnL43".
    experiment_date : str
        Date string as used in the CSV, e.g. "260408".
    log_dir : Path or str, optional
        Directory containing the experiment log CSVs.
        Defaults to the experiment_log/ folder next to this file.

    Returns
    -------
    rec_folders : list of Path
        Paths to the .rec folder(s) for this session (one per passive recording).
    task_file_paths : list of Path
        Paths to all .txt task files for this session.
    """
    if platform.system() == "Darwin":
        parent_folder = Path(r"/Volumes/xieluanlabs/xl_cl/experiment_data")
    else:
        parent_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data")

    if log_dir is None:
        log_dir = Path(__file__).parent / "experiment_log"
    csv_path = Path(log_dir) / f"{animal_id}.csv"

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        reader.fieldnames = [k.strip() for k in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if row['date'] == experiment_date:
                break
        else:
            raise ValueError(f"No entry for date {experiment_date} in {csv_path}")

    session_base = parent_folder / animal_id / experiment_date
    ephys_folder = session_base / row['EphysFolder']
    rec_folders = [
        ephys_folder / f.strip()
        for f in row['PassiveFolder'].split(';')
        if f.strip()
    ]
    task_file_paths = [
        session_base / f.strip()
        for f in row['TaskFile'].split(';')
        if f.strip().endswith('.txt')
    ]
    return rec_folders, task_file_paths


# =============================================================================
# DATA LOADING
# =============================================================================

def load_neural_data(filepath):
    """Load neural data from pickle, HDF5, or NPZ format."""
    filepath = Path(filepath)
    loaders = {
        '.pkl': _load_pickle,
        '.h5': _load_hdf5,
        '.npz': _load_npz
    }

    loader = loaders.get(filepath.suffix)
    if not loader:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    print(f"Loading data from {filepath.suffix}: {filepath}")
    return loader(filepath)


def _load_pickle(filepath):
    """Load from pickle format."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _load_hdf5(filepath):
    """Load from HDF5 format."""
    data = {
        'metadata': {},
        'experiment_parameters': {},
        'trial_info': {},
        'spike_data': {},
        'unit_info': {},
        'extraction_params': {}
    }

    with h5py.File(filepath, 'r') as f:
        def load_group(group, target_dict):
            for key in group.attrs.keys():
                target_dict[key] = group.attrs[key]
            for key in group.keys():
                target_dict[key] = group[key][()]

        if 'metadata' in f:
            load_group(f['metadata'], data['metadata'])
        if 'experiment_parameters' in f:
            load_group(f['experiment_parameters'], data['experiment_parameters'])
        if 'extraction_params' in f:
            load_group(f['extraction_params'], data['extraction_params'])

        if 'trial_info' in f:
            trial_grp = f['trial_info']
            for key in ['orientations', 'unique_orientations', 'trial_windows']:
                if key in trial_grp:
                    data['trial_info'][key] = trial_grp[key][()].tolist()
            if 'all_trial_parameters' in trial_grp.attrs:
                data['trial_info']['all_trial_parameters'] = json.loads(
                    trial_grp.attrs['all_trial_parameters']
                )

        if 'spike_data' in f:
            for unit_id in f['spike_data'].keys():
                unit_grp = f['spike_data'][unit_id]
                trials_data = []
                for trial_key in sorted([k for k in unit_grp.keys() if k.startswith('trial_')]):
                    tgrp = unit_grp[trial_key]
                    trials_data.append({
                        'trial_index': tgrp.attrs['trial_index'],
                        'orientation': tgrp.attrs['orientation'] if tgrp.attrs['orientation'] != -999 else None,
                        'spike_count': tgrp.attrs['spike_count'],
                        'trial_start': tgrp.attrs['trial_start'],
                        'trial_end': tgrp.attrs['trial_end'],
                        'spike_times': tgrp['spike_times'][()].tolist()
                    })
                data['spike_data'][unit_id] = trials_data

        if 'unit_info' in f:
            for unit_id in f['unit_info'].keys():
                data['unit_info'][unit_id] = {}
                load_group(f['unit_info'][unit_id], data['unit_info'][unit_id])

    return data


def _load_npz(filepath):
    """Load from NPZ format (with companion pickle for complex data)."""
    data_npz = np.load(filepath, allow_pickle=True)
    pickle_path = filepath.with_suffix('.complex.pkl')
    complex_data = _load_pickle(pickle_path) if pickle_path.exists() else {}

    data = {
        'metadata': {},
        'experiment_parameters': {},
        'trial_info': {},
        'extraction_params': {},
        'spike_data': complex_data.get('spike_data', {}),
        'unit_info': complex_data.get('unit_info', {})
    }

    key_mapping = {
        'metadata_': ('metadata', 9),
        'exp_': ('experiment_parameters', 4),
        'params_': ('extraction_params', 7)
    }

    for key, value in data_npz.items():
        for prefix, (target, offset) in key_mapping.items():
            if key.startswith(prefix):
                data[target][key[offset:]] = value
                break

        if key == 'trial_orientations':
            data['trial_info']['orientations'] = value.tolist()
        elif key == 'trial_unique_orientations':
            data['trial_info']['unique_orientations'] = value.tolist()
        elif key == 'trial_windows':
            data['trial_info']['trial_windows'] = value.tolist()

    if 'trial_parameters' in complex_data:
        data['trial_info']['all_trial_parameters'] = complex_data['trial_parameters']

    return data


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def calculate_firing_rates(neural_data, time_window=(0.07, 0.16)):
    """
    Calculate firing rates for each unit in each trial.

    Returns:
        firing_rates: (n_trials, n_units) array
        orientation_labels: (n_trials,) array
        unit_ids: list of unit identifiers
        trial_info: dict with experiment metadata (includes spatial_freq_labels and unique_spatial_freqs)
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start

    unit_ids = list(neural_data['spike_data'].keys())
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = neural_data['trial_info']['unique_orientations']
    n_trials, n_units = len(orientations), len(unit_ids)

    spatial_freqs_all = [None] * n_trials
    if unit_ids:
        first_unit_trials = neural_data['spike_data'][unit_ids[0]]
        for t in first_unit_trials:
            idx = int(t['trial_index'])
            if 0 <= idx < n_trials:
                spatial_freqs_all[idx] = t.get('spatial_freq', None)

    has_sf = any(sf is not None for sf in spatial_freqs_all)

    print(f"\nCalculating firing rates:")
    print(f"  Units: {n_units} | Trials: {n_trials}")
    print(f"  Window: {window_start:.3f}-{window_end:.3f}s ({window_duration:.3f}s)")
    print(f"  Orientations: {unique_orientations}")
    for ori in unique_orientations:
        print(f"    {ori}°: {orientations.count(ori)} trials")
    if has_sf:
        unique_sfs = sorted(set(sf for sf in spatial_freqs_all if sf is not None))
        print(f"  Spatial frequencies: {unique_sfs}")

    firing_rates = np.full((n_trials, n_units), np.nan)
    for unit_idx, unit_id in enumerate(unit_ids):
        for trial_data in neural_data['spike_data'][unit_id]:
            trial_idx = int(trial_data['trial_index'])
            if 0 <= trial_idx < n_trials:
                spike_times = np.array(trial_data['spike_times'])
                spikes_in_window = np.sum((spike_times >= window_start) &
                                         (spike_times < window_end))
                firing_rates[trial_idx, unit_idx] = spikes_in_window / window_duration

    valid_mask = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_mask]
    orientation_labels = np.array(orientations)[valid_mask]
    sf_labels = np.array(spatial_freqs_all)[valid_mask] if has_sf else None

    print(f"  Valid trials: {np.sum(valid_mask)}/{n_trials}")
    print(f"  Mean firing rate: {np.mean(firing_rates_clean):.2f} Hz")

    unique_sfs = sorted(set(sf_labels.tolist())) if sf_labels is not None else [None]

    trial_info = {
        'valid_trials_mask': valid_mask,
        'unique_orientations': unique_orientations,
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_trials_per_orientation': {
            str(ori): int(np.sum(orientation_labels == ori))
            for ori in unique_orientations
        },
        'spatial_freq_labels': sf_labels,
        'unique_spatial_freqs': unique_sfs,
    }

    return firing_rates_clean, orientation_labels, unit_ids, trial_info


# =============================================================================
# ORIENTATION SELECTIVITY
# =============================================================================

def calculate_orientation_selectivity(unit_ids, orientation_labels, firing_rates):
    """
    Calculate orientation selectivity index (OSI) using vector sum method.
    OSI = |Σ r(θ)·exp(i·2θ)| / Σ r(θ)

    Returns:
        Dictionary with unit_ids, osi, and preferred_orientation_deg
    """
    orientations = np.unique(orientation_labels)
    theta_rad = 2 * np.deg2rad(np.array(orientations, dtype=float))

    mean_rates = np.array([
        firing_rates[orientation_labels == ori].mean(axis=0)
        for ori in orientations
    ])  # (n_orientations, n_units)

    complex_exp = np.exp(1j * theta_rad)[:, None]
    vector_sum = (mean_rates * complex_exp).sum(axis=0)

    osi = np.abs(vector_sum) / (mean_rates.sum(axis=0) + 1e-12)
    pref_orientation_deg = (np.angle(vector_sum) / 2.0) % np.pi
    pref_orientation_deg = np.rad2deg(pref_orientation_deg)

    print(f"\nOrientation Selectivity:")
    print(f"  Mean OSI: {osi.mean():.3f}")
    print(f"  Top units:")
    for idx in np.argsort(osi)[::-1][:min(10, len(unit_ids))]:
        print(f"    {unit_ids[idx]}: OSI={osi[idx]:.3f}, Pref={pref_orientation_deg[idx]:.1f}°")

    return {
        'unit_ids': unit_ids,
        'osi': osi,
        'preferred_orientation_deg': pref_orientation_deg
    }


def tuning_curve_from_psth(tuning, window=(0.05, 0.5)):
    """
    Recompute an orientation tuning curve (and its metrics) from a unit's stored
    PSTHs, averaging firing rate over a post-stimulus time window.

    This is the single shared definition used by the matched-unit plotting
    scripts (plot_matched_unit_tuning_tracks.py, plot_unit_overlay_across_time.py)
    so they draw identical curves. It operates on the per-unit ``tuning`` dict
    saved inside each ``*_tuning.pkl`` (keys ``psth_per_ori`` / ``psth_t``, with
    ``orientations`` / ``mean_rates`` as a fallback). The metric formulas (OSI,
    preferred orientation, modulation index, baseline) match
    GratingTuningCurve.calculate_tuning_curves, but evaluated over ``window``.

    Note: ``psth_per_ori`` is a trial-averaged trace, so per-trial SEM cannot be
    reconstructed here. For a window-matched SEM, compute the curve directly from
    per-trial spike times with ``tuning_curve_from_spikes`` (fed by the per-day
    merged grating pickle), which is what the matched-unit row plot uses.

    Parameters
    ----------
    tuning : dict
        Per-unit tuning dict from a ``*_tuning.pkl``.
    window : (float, float)
        (start, end) seconds relative to stimulus onset. Bins with
        ``start <= t < end`` are averaged.

    Returns
    -------
    dict with keys: orientations, mean_rates (np.ndarray), osi,
    preferred_orientation_deg, modulation_index, max_rate, min_rate,
    baseline_rate, window. Falls back to the pickle's precomputed values if no
    usable PSTHs are present.
    """
    time = np.asarray(tuning.get('psth_t', []), dtype=float)
    psth_per_ori = tuning.get('psth_per_ori', {})

    orientations, mean_rates = [], []
    if time.size and psth_per_ori:
        start, end = window
        mask = (time >= start) & (time < end)
        if mask.any():
            for ori, psth in psth_per_ori.items():
                values = np.asarray(psth, dtype=float)
                if values.size != time.size:
                    continue
                orientations.append(float(ori))
                mean_rates.append(float(np.nanmean(values[mask])))

    if not orientations:
        # Fallback: precomputed curve/metrics straight from the pickle.
        return {
            'orientations': np.asarray(tuning.get('orientations', []), dtype=float),
            'mean_rates': np.asarray(tuning.get('mean_rates', []), dtype=float),
            'osi': tuning.get('osi', np.nan),
            'preferred_orientation_deg': tuning.get('preferred_orientation_deg', np.nan),
            'modulation_index': tuning.get('modulation_index', np.nan),
            'max_rate': tuning.get('max_rate', np.nan),
            'min_rate': tuning.get('min_rate', np.nan),
            'baseline_rate': tuning.get('baseline_rate', np.nan),
            'window': tuple(window),
        }

    order = np.argsort(orientations)
    orientations = np.asarray(orientations)[order]
    mean_rates = np.asarray(mean_rates)[order]

    # Metrics — same formulas as GratingTuningCurve.calculate_tuning_curves.
    theta_rad = 2 * np.deg2rad(orientations)
    complex_sum = np.sum(mean_rates * np.exp(1j * theta_rad))
    osi = float(np.abs(complex_sum) / (np.sum(mean_rates) + 1e-12))
    preferred_ori_deg = float(np.rad2deg((np.angle(complex_sum) / 2.0) % np.pi))
    max_rate = float(np.max(mean_rates))
    min_rate = float(np.min(mean_rates))
    modulation_index = float((max_rate - min_rate) / (max_rate + min_rate + 1e-12))
    baseline_rate = float(np.mean(mean_rates))

    return {
        'orientations': orientations,
        'mean_rates': mean_rates,
        'osi': osi,
        'preferred_orientation_deg': preferred_ori_deg,
        'modulation_index': modulation_index,
        'max_rate': max_rate,
        'min_rate': min_rate,
        'baseline_rate': baseline_rate,
        'window': tuple(window),
    }


def tuning_curve_from_spikes(ori_spikes, window=(0.05, 0.5)):
    """
    Compute an orientation tuning curve (with window-matched SEM) directly from
    per-trial spike times.

    Unlike ``tuning_curve_from_psth`` (which works off the trial-averaged PSTH and
    therefore cannot recover per-trial variability), this takes the raw per-trial
    spike-time arrays — e.g. from the per-day merged grating pickle
    (``spike_data[unit_key]`` -> trials with ``spike_times``) — so the mean rate
    and its SEM are computed over the *same* window. The per-orientation firing
    rate per trial is ``(# spikes in [start, end)) / (end - start)``, matching
    GratingTuningCurve.calculate_tuning_curves; the metric formulas (OSI,
    preferred orientation, modulation index, baseline) are identical too.

    Parameters
    ----------
    ori_spikes : dict[float, list[np.ndarray]]
        Orientation (deg) -> list of spike-time arrays (s, relative to stim onset),
        one array per trial.
    window : (float, float)
        (start, end) seconds relative to stimulus onset.

    Returns
    -------
    dict with keys: orientations, mean_rates, sem_rates (np.ndarray), osi,
    preferred_orientation_deg, modulation_index, max_rate, min_rate,
    baseline_rate, n_trials (per orientation), window.
    """
    start, end = window
    duration = float(end - start)

    orientations, mean_rates, sem_rates, n_trials = [], [], [], []
    for ori in sorted(ori_spikes):
        trials = ori_spikes[ori]
        if not trials:
            continue
        rates = np.array(
            [np.sum((s >= start) & (s < end)) / duration for s in
             (np.asarray(t, dtype=float) for t in trials)],
            dtype=float,
        )
        n = rates.size
        orientations.append(float(ori))
        mean_rates.append(float(np.mean(rates)))
        sem_rates.append(float(np.std(rates, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        n_trials.append(int(n))

    orientations = np.asarray(orientations, dtype=float)
    mean_rates = np.asarray(mean_rates, dtype=float)
    sem_rates = np.asarray(sem_rates, dtype=float)

    if orientations.size == 0:
        nan = float('nan')
        return {
            'orientations': orientations, 'mean_rates': mean_rates,
            'sem_rates': sem_rates, 'osi': nan, 'preferred_orientation_deg': nan,
            'modulation_index': nan, 'max_rate': nan, 'min_rate': nan,
            'baseline_rate': nan, 'n_trials': np.asarray(n_trials, dtype=int),
            'window': tuple(window),
        }

    # Metrics — same formulas as GratingTuningCurve.calculate_tuning_curves.
    theta_rad = 2 * np.deg2rad(orientations)
    complex_sum = np.sum(mean_rates * np.exp(1j * theta_rad))
    osi = float(np.abs(complex_sum) / (np.sum(mean_rates) + 1e-12))
    preferred_ori_deg = float(np.rad2deg((np.angle(complex_sum) / 2.0) % np.pi))
    max_rate = float(np.max(mean_rates))
    min_rate = float(np.min(mean_rates))
    modulation_index = float((max_rate - min_rate) / (max_rate + min_rate + 1e-12))
    baseline_rate = float(np.mean(mean_rates))

    return {
        'orientations': orientations,
        'mean_rates': mean_rates,
        'sem_rates': sem_rates,
        'osi': osi,
        'preferred_orientation_deg': preferred_ori_deg,
        'modulation_index': modulation_index,
        'max_rate': max_rate,
        'min_rate': min_rate,
        'baseline_rate': baseline_rate,
        'n_trials': np.asarray(n_trials, dtype=int),
        'window': tuple(window),
    }


# =============================================================================
# SHARED VISUALIZATION HELPERS
# =============================================================================

def plot_confusion_matrix(fig, conf_matrix, orientations, label_suffix='°', subplot_pos=(3, 4, 3)):
    """Plot confusion matrix."""
    ax = fig.add_subplot(*subplot_pos)

    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=16)

    tick_labels = [f'{ori}{label_suffix}' for ori in orientations]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_yticklabels(tick_labels, fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=24, fontweight='bold', pad=12)
    ax.set_ylabel('True', fontsize=22, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=22, fontweight='bold')

    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = 'white' if conf_matrix[i, j] > thresh else 'black'
            ax.text(j, i, int(conf_matrix[i, j]), ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax


def plot_cv_scores(fig, results, subplot_pos=(3, 4, 4)):
    """
    Plot CV accuracy as two bars: real CV score vs. shuffled-label baseline.
    Error bars are the std across CV folds.

    Falls back to per-fold bars if 'cv_scores_shuffled' is absent.
    """
    ax = fig.add_subplot(*subplot_pos)

    scores = results['cv_scores']

    if 'cv_scores_shuffled' in results:
        shuffled = results['cv_scores_shuffled']
        means = np.array([scores.mean(), shuffled.mean()])
        stds = np.array([scores.std(), shuffled.std()])
        bar_labels = ['CV', 'Shuffled']
        bar_colors = ['#4C72B0', '#BBBBBB']

        bars = ax.bar(bar_labels, means, yerr=stds, color=bar_colors,
                      width=0.55, capsize=10, edgecolor='black', linewidth=1.8,
                      error_kw={'elinewidth': 2.5, 'ecolor': 'black'})
        ax.axhline(results['chance_accuracy'], color='black', linestyle='--',
                   linewidth=2.0,
                   label=f'Chance ({results["chance_accuracy"]:.3f})')

        y_top = float(min(1.15, max(1.0, (means + stds).max() + 0.18)))
        ax.set_ylim(0, y_top)
        ax.set_ylabel('Decoding accuracy', fontsize=22, fontweight='bold')
        ax.set_title('Cross-validation', fontsize=24, fontweight='bold', pad=12)
        ax.tick_params(axis='both', labelsize=18, length=7, width=2.0)
        ax.tick_params(axis='x', length=0)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(2.0)

        ax.legend(frameon=False, fontsize=16, loc='upper right')

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(m + s + 0.025, y_top - 0.02),
                    f'{m:.3f} ± {s:.3f}',
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
        return ax

    bars = ax.bar(range(len(scores)), scores, alpha=0.75, color='skyblue',
                  edgecolor='black', linewidth=1.5)
    ax.axhline(scores.mean(), color='red', linestyle='--', linewidth=2.0,
               label=f'Mean: {scores.mean():.3f}')
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':',
               linewidth=2.0,
               label=f'Chance: {results["chance_accuracy"]:.3f}')

    ax.set_xlabel('CV Fold', fontsize=22, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Cross-Validation Scores', fontsize=24, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=18, width=2.0, length=7)
    ax.legend(fontsize=16, frameon=False)
    ax.grid(True, alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    return ax


def plot_per_class_accuracy(fig, results, orientations, colors, label_suffix='°',
                            subplot_pos=(3, 4, 5)):
    """Plot per-class classification accuracy."""
    ax = fig.add_subplot(*subplot_pos)

    accuracies = results['orientation_accuracies']
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':',
               linewidth=2.0, label='Chance')

    ax.set_xlabel('Class', fontsize=22, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Per-Class Accuracy', fontsize=24, fontweight='bold', pad=12)
    ax.set_xticks(range(len(orientations)))
    ax.set_xticklabels([f'{ori}{label_suffix}' for ori in orientations],
                       fontsize=16)
    ax.tick_params(axis='both', labelsize=18, width=2.0, length=7)
    ax.legend(fontsize=16, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax


def plot_polar_accuracy(fig, results, orientations, subplot_pos=(3, 4, 6)):
    """Plot decoding accuracy in polar coordinates (orientation analysis only)."""
    ax = fig.add_subplot(*subplot_pos, projection='polar')

    theta = 2 * np.deg2rad(np.array(orientations, dtype=float))
    accuracies = results['orientation_accuracies']

    ax.plot(theta, accuracies, 'o-', linewidth=4.0, markersize=14,
            color='#2E86AB')
    ax.fill(theta, accuracies, alpha=0.25, color='#2E86AB')
    ax.set_ylim(0, 1)
    ax.set_title('Polar Decoding Accuracy', fontsize=24,
                 fontweight='bold', pad=14)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      [f'{a / 2:g}°' for a in np.arange(0, 360, 45)],
                      fontsize=18)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', pad=12)
    ax.grid(True, linewidth=1.6, alpha=0.5)
    return ax


def plot_sf_accuracy_bar(fig, results, sfs, colors, label_suffix=' cpd',
                         subplot_pos=(3, 4, 6)):
    """Bar chart of per-SF decoding accuracy."""
    ax = fig.add_subplot(*subplot_pos)

    accuracies = results['orientation_accuracies']
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':',
               linewidth=2.0,
               label=f'Chance: {results["chance_accuracy"]:.3f}')

    ax.set_xlabel('Spatial Frequency', fontsize=22, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Per-SF Decoding Accuracy', fontsize=24,
                 fontweight='bold', pad=12)
    ax.set_xticks(range(len(sfs)))
    ax.set_xticklabels([f'{sf}{label_suffix}' for sf in sfs], fontsize=16)
    ax.tick_params(axis='both', labelsize=18, width=2.0, length=7)
    ax.legend(fontsize=16, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    return ax


def plot_trial_distribution(fig, trial_info, orientations, colors, label_suffix='°',
                            subplot_pos=(3, 4, 9)):
    """Plot trial count distribution across classes."""
    ax = fig.add_subplot(*subplot_pos)

    counts = [trial_info['n_trials_per_orientation'][str(ori)] for ori in orientations]
    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Class', fontsize=22, fontweight='bold')
    ax.set_ylabel('Number of Trials', fontsize=22, fontweight='bold')
    ax.set_title('Trial Distribution', fontsize=24, fontweight='bold', pad=12)
    ax.set_xticks(range(len(orientations)))
    ax.set_xticklabels([f'{ori}{label_suffix}' for ori in orientations],
                       fontsize=16)
    ax.tick_params(axis='both', labelsize=18, width=2.0, length=7)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, int(h),
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax


def plot_summary_text(fig, results, labels, unit_ids, trial_info, label_suffix='°',
                      model_params=None, subplot_pos=(3, 4, 10)):
    """
    Plot summary statistics text.

    Args:
        model_params: dict of extra model-specific key-value pairs to display
                      (e.g. {'Kernel': 'rbf', 'C': 1.0} for SVM)
    """
    ax = fig.add_subplot(*subplot_pos)
    ax.axis('off')

    exp_params = trial_info.get('experiment_parameters', {})
    unique_classes = results['unique_orientations']

    extra_lines = ''
    if model_params:
        for k, v in model_params.items():
            extra_lines += f'\n    • {k}: {v}'

    # Training-set accuracy is leakage; only show it if predictions are provided
    # (legacy behavior). Prefer CV-based metrics.
    overall_line = ''
    if 'predictions' in results:
        overall_line = (
            f"Overall Accuracy (train): "
            f"{accuracy_score(labels, results['predictions']):.3f}\n    "
        )

    shuffled_line = ''
    if 'cv_scores_shuffled' in results:
        s = results['cv_scores_shuffled']
        shuffled_line = f"Shuffled CV: {s.mean():.3f} ± {s.std():.3f}\n    "

    summary = f"""
    Classification Summary

    {overall_line}CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    {shuffled_line}Chance Level: {results['chance_accuracy']:.3f}
{extra_lines}
    Experiment Info:
    • Total trials: {len(labels)}
    • Classes: {len(unique_classes)} ({min(unique_classes)}{label_suffix} - {max(unique_classes)}{label_suffix})
    • Units: {len(unit_ids)}
    • Stimulus duration: {exp_params.get('stimulus_duration', 'N/A')}s
    • ITI duration: {exp_params.get('iti_duration', 'N/A')}s
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', fontfamily='monospace',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgray',
                      alpha=0.8))
    return ax


def plot_prediction_confidence(fig, results, labels, subplot_pos=(3, 4, 12)):
    """Plot distribution of prediction confidence (max predicted probability)."""
    ax = fig.add_subplot(*subplot_pos)

    confidence = np.max(results['prediction_proba'], axis=1)
    correct = results['predictions'] == labels

    def _safe_hist(ax, data, **kwargs):
        if len(data) == 0:
            return
        ax.hist(data, bins=20, range=(0, 1), **kwargs)

    _safe_hist(ax, confidence[correct],  alpha=0.7, label='Correct',
               color='green', density=True, edgecolor='black', linewidth=1.0)
    _safe_hist(ax, confidence[~correct], alpha=0.7, label='Incorrect',
               color='red',   density=True, edgecolor='black', linewidth=1.0)

    ax.set_xlabel('Prediction Confidence', fontsize=22, fontweight='bold')
    ax.set_ylabel('Density', fontsize=22, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution', fontsize=24,
                 fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=18, width=2.0, length=7)
    ax.legend(fontsize=16, frameon=False)
    return ax
