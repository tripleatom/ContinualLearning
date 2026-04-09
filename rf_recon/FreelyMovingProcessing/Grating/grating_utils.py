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
    rec_folder : Path
        Path to the .rec folder for this session.
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
    rec_folder = session_base / row['EphysFolder'] / row['PassiveFolder']
    task_file_paths = [
        session_base / f.strip()
        for f in row['TaskFile'].split(';')
        if f.strip().endswith('.txt')
    ]
    return rec_folder, task_file_paths


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


# =============================================================================
# SHARED VISUALIZATION HELPERS
# =============================================================================

def plot_confusion_matrix(fig, conf_matrix, orientations, label_suffix='°', subplot_pos=(3, 4, 3)):
    """Plot confusion matrix."""
    ax = fig.add_subplot(*subplot_pos)

    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    tick_labels = [f'{ori}{label_suffix}' for ori in orientations]
    ax.set(xticks=np.arange(len(tick_labels)), yticks=np.arange(len(tick_labels)),
           xticklabels=tick_labels, yticklabels=tick_labels,
           title='Confusion Matrix', ylabel='True', xlabel='Predicted')

    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = 'white' if conf_matrix[i, j] > thresh else 'black'
            ax.text(j, i, int(conf_matrix[i, j]), ha='center', va='center',
                    color=color, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax


def plot_cv_scores(fig, results, subplot_pos=(3, 4, 4)):
    """Plot cross-validation accuracy scores."""
    ax = fig.add_subplot(*subplot_pos)

    scores = results['cv_scores']
    bars = ax.bar(range(len(scores)), scores, alpha=0.7, color='skyblue')
    ax.axhline(scores.mean(), color='red', linestyle='--',
               label=f'Mean: {scores.mean():.3f}')
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':',
               label=f'Chance: {results["chance_accuracy"]:.3f}')

    ax.set(xlabel='CV Fold', ylabel='Accuracy', ylim=[0, 1],
           title='Cross-Validation Scores')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)
    return ax


def plot_per_class_accuracy(fig, results, orientations, colors, label_suffix='°',
                            subplot_pos=(3, 4, 5)):
    """Plot per-class classification accuracy."""
    ax = fig.add_subplot(*subplot_pos)

    accuracies = results['orientation_accuracies']
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', label='Chance')

    ax.set(xlabel='Class', ylabel='Accuracy', ylim=[0, 1],
           title='Per-Class Accuracy',
           xticks=range(len(orientations)),
           xticklabels=[f'{ori}{label_suffix}' for ori in orientations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax


def plot_polar_accuracy(fig, results, orientations, subplot_pos=(3, 4, 6)):
    """Plot decoding accuracy in polar coordinates (orientation analysis only)."""
    ax = fig.add_subplot(*subplot_pos, projection='polar')

    theta = 2 * np.deg2rad(np.array(orientations, dtype=float))
    accuracies = results['orientation_accuracies']

    ax.plot(theta, accuracies, 'o-', linewidth=2, markersize=8)
    ax.fill(theta, accuracies, alpha=0.25)
    ax.set(ylim=[0, 1], title='Polar Decoding Accuracy')
    ax.set_thetagrids(np.arange(0, 360, 45),
                      [f'{int(a / 2)}°' for a in np.arange(0, 360, 45)])
    ax.grid(True)
    return ax


def plot_sf_accuracy_bar(fig, results, sfs, colors, label_suffix=' cpd',
                         subplot_pos=(3, 4, 6)):
    """Bar chart of per-SF decoding accuracy."""
    ax = fig.add_subplot(*subplot_pos)

    accuracies = results['orientation_accuracies']
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':',
               label=f'Chance: {results["chance_accuracy"]:.3f}')

    ax.set(xlabel='Spatial Frequency', ylabel='Accuracy', ylim=[0, 1],
           title='Per-SF Decoding Accuracy',
           xticks=range(len(sfs)),
           xticklabels=[f'{sf}{label_suffix}' for sf in sfs])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)
    return ax


def plot_trial_distribution(fig, trial_info, orientations, colors, label_suffix='°',
                            subplot_pos=(3, 4, 9)):
    """Plot trial count distribution across classes."""
    ax = fig.add_subplot(*subplot_pos)

    counts = [trial_info['n_trials_per_orientation'][str(ori)] for ori in orientations]
    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.7)

    ax.set(xlabel='Class', ylabel='Number of Trials',
           title='Trial Distribution',
           xticks=range(len(orientations)),
           xticklabels=[f'{ori}{label_suffix}' for ori in orientations])
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, int(h),
                ha='center', va='bottom', fontsize=10)

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

    summary = f"""
    Classification Summary

    Overall Accuracy: {accuracy_score(labels, results['predictions']):.3f}
    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}
{extra_lines}
    Experiment Info:
    • Total trials: {len(labels)}
    • Classes: {len(unique_classes)} ({min(unique_classes)}{label_suffix} - {max(unique_classes)}{label_suffix})
    • Units: {len(unit_ids)}
    • Stimulus duration: {exp_params.get('stimulus_duration', 'N/A')}s
    • ITI duration: {exp_params.get('iti_duration', 'N/A')}s
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    return ax


def plot_prediction_confidence(fig, results, labels, subplot_pos=(3, 4, 12)):
    """Plot distribution of prediction confidence (max predicted probability)."""
    ax = fig.add_subplot(*subplot_pos)

    confidence = np.max(results['prediction_proba'], axis=1)
    correct = results['predictions'] == labels

    ax.hist(confidence[correct], bins=20, alpha=0.7, label='Correct',
            color='green', density=True)
    ax.hist(confidence[~correct], bins=20, alpha=0.7, label='Incorrect',
            color='red', density=True)

    ax.set(xlabel='Prediction Confidence', ylabel='Density',
           title='Prediction Confidence Distribution')
    ax.legend()
    return ax
