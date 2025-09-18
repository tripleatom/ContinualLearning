import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import pickle
import h5py
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings
warnings.filterwarnings('ignore')


# ---------------------------
# Loaders
# ---------------------------

def load_neural_data(filepath):
    """Load neural data from saved file"""
    filepath = Path(filepath)
    if filepath.suffix == '.h5':
        return load_from_hdf5(filepath)
    elif filepath.suffix == '.pkl':
        return load_from_pickle(filepath)
    elif filepath.suffix == '.npz':
        return load_from_npz(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_from_pickle(filepath):
    """Load neural data from pickle format"""
    print(f"Loading data from pickle: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_from_hdf5(filepath):
    """Load neural data from HDF5 format"""
    print(f"Loading data from HDF5: {filepath}")
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Metadata
        data['metadata'] = {}
        if 'metadata' in f:
            metadata_grp = f['metadata']
            for key in metadata_grp.attrs.keys():
                data['metadata'][key] = metadata_grp.attrs[key]
            for key in metadata_grp.keys():
                data['metadata'][key] = metadata_grp[key][()]
        # Experiment parameters
        data['experiment_parameters'] = {}
        if 'experiment_parameters' in f:
            exp_grp = f['experiment_parameters']
            for key in exp_grp.attrs.keys():
                data['experiment_parameters'][key] = exp_grp.attrs[key]
        # Trial info
        data['trial_info'] = {}
        if 'trial_info' in f:
            trial_grp = f['trial_info']
            if 'orientations' in trial_grp:
                data['trial_info']['orientations'] = trial_grp['orientations'][()].tolist()
            if 'unique_orientations' in trial_grp:
                data['trial_info']['unique_orientations'] = trial_grp['unique_orientations'][()].tolist()
            if 'trial_windows' in trial_grp:
                data['trial_info']['trial_windows'] = trial_grp['trial_windows'][()].tolist()
            if 'all_trial_parameters' in trial_grp.attrs:
                data['trial_info']['all_trial_parameters'] = json.loads(trial_grp.attrs['all_trial_parameters'])
        # Spike data
        data['spike_data'] = {}
        if 'spike_data' in f:
            spike_grp = f['spike_data']
            for unit_id in spike_grp.keys():
                unit_grp = spike_grp[unit_id]
                trials_data = []
                trial_keys = sorted([k for k in unit_grp.keys() if k.startswith('trial_')])
                for trial_key in trial_keys:
                    tgrp = unit_grp[trial_key]
                    trial_data = {
                        'trial_index': tgrp.attrs['trial_index'],
                        'orientation': tgrp.attrs['orientation'] if tgrp.attrs['orientation'] != -999 else None,
                        'spike_count': tgrp.attrs['spike_count'],
                        'trial_start': tgrp.attrs['trial_start'],
                        'trial_end': tgrp.attrs['trial_end'],
                        'spike_times': tgrp['spike_times'][()].tolist()
                    }
                    trials_data.append(trial_data)
                data['spike_data'][unit_id] = trials_data
        # Unit info
        data['unit_info'] = {}
        if 'unit_info' in f:
            unit_grp = f['unit_info']
            for unit_id in unit_grp.keys():
                unit_info_grp = unit_grp[unit_id]
                info = {}
                for key in unit_info_grp.attrs.keys():
                    info[key] = unit_info_grp.attrs[key]
                for key in unit_info_grp.keys():
                    info[key] = unit_info_grp[key][()]
                data['unit_info'][unit_id] = info
        # Extraction params
        data['extraction_params'] = {}
        if 'extraction_params' in f:
            params_grp = f['extraction_params']
            for key in params_grp.attrs.keys():
                data['extraction_params'][key] = params_grp.attrs[key]
            for key in params_grp.keys():
                data['extraction_params'][key] = params_grp[key][()]
    return data


def load_from_npz(filepath):
    """Load neural data from npz format"""
    print(f"Loading data from NPZ: {filepath}")
    data_npz = np.load(filepath, allow_pickle=True)
    pickle_path = filepath.with_suffix('.complex.pkl')
    complex_data = load_from_pickle(pickle_path) if pickle_path.exists() else {}
    data = {
        'metadata': {},
        'experiment_parameters': {},
        'trial_info': {},
        'extraction_params': {},
        'spike_data': complex_data.get('spike_data', {}),
        'unit_info': complex_data.get('unit_info', {})
    }
    for key, value in data_npz.items():
        if key.startswith('metadata_'):
            data['metadata'][key[9:]] = value
        elif key.startswith('exp_'):
            data['experiment_parameters'][key[4:]] = value
        elif key.startswith('trial_'):
            if key == 'trial_orientations':
                data['trial_info']['orientations'] = value.tolist()
            elif key == 'trial_unique_orientations':
                data['trial_info']['unique_orientations'] = value.tolist()
            elif key == 'trial_windows':
                data['trial_info']['trial_windows'] = value.tolist()
        elif key.startswith('params_'):
            data['extraction_params'][key[7:]] = value
    if 'trial_parameters' in complex_data:
        data['trial_info']['all_trial_parameters'] = complex_data['trial_parameters']
    return data


# ---------------------------
# Analysis
# ---------------------------

def calculate_grating_firing_rates(neural_data, time_window=(0.07, 0.16)):
    """
    Return:
      firing_rates_clean: (n_valid_trials, n_units)
      orientation_labels_clean: (n_valid_trials,)
      unit_ids: list
      trial_info: dict
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start

    unit_ids = list(neural_data['spike_data'].keys())
    n_units = len(unit_ids)

    orientations = list(neural_data['trial_info']['orientations'])
    unique_orientations = list(neural_data['trial_info']['unique_orientations'])
    n_trials = len(orientations)

    print(f"Calculating firing rates for grating experiment:")
    print(f"- {n_units} units across {n_trials} trials")
    print(f"- Time window: {window_start:.3f}s to {window_end:.3f}s ({window_duration:.3f}s)")
    print(f"- Orientations tested: {unique_orientations}")

    for orientation in unique_orientations:
        count = orientations.count(orientation)
        print(f"- Orientation {orientation}°: {count} trials")

    firing_rates = np.full((n_trials, n_units), np.nan)

    for unit_idx, unit_id in enumerate(unit_ids):
        unit_trials = neural_data['spike_data'][unit_id]
        for trial_data in unit_trials:
            trial_idx = int(trial_data['trial_index'])
            spike_times = np.array(trial_data['spike_times'])
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            firing_rate = spikes_in_window / window_duration
            if 0 <= trial_idx < n_trials:
                firing_rates[trial_idx, unit_idx] = firing_rate

    valid_trials = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_trials]
    orientation_array = np.array(orientations)
    orientation_labels_clean = orientation_array[valid_trials]

    print(f"Valid trials: {np.sum(valid_trials)}/{n_trials}")
    if firing_rates_clean.size > 0:
        print(f"Mean firing rate: {np.nanmean(firing_rates_clean):.2f} Hz")

    trial_info = {
        'valid_trials_mask': valid_trials,
        'unique_orientations': unique_orientations,
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_trials_per_orientation': {ori: int(np.sum(orientation_labels_clean == ori))
                                     for ori in unique_orientations}
    }
    return firing_rates_clean, orientation_labels_clean, unit_ids, trial_info


def perform_grating_lda_analysis(firing_rates, orientation_labels, n_components=None):
    unique_orientations = np.unique(orientation_labels)
    n_orientations = len(unique_orientations)
    n_features = firing_rates.shape[1]

    print(f"\nGrating LDA Analysis:")
    print(f"Number of orientations: {n_orientations}")
    print(f"Orientations: {unique_orientations}°")
    print(f"Number of features (units): {n_features}")
    print(f"Number of trials: {len(orientation_labels)}")

    min_trials_per_class = np.min([np.sum(orientation_labels == ori) for ori in unique_orientations])
    print(f"Minimum trials per orientation: {int(min_trials_per_class)}")

    max_components = min(n_orientations - 1, n_features)
    if n_components is None:
        n_components = min(3, max_components)
    else:
        n_components = min(n_components, max_components)
    print(f"Using {n_components} LDA components")

    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_transformed = lda.fit_transform(firing_rates_scaled, orientation_labels)

    cv_folds = max(2, min(5, int(min_trials_per_class)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    lda_full = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda_full, firing_rates_scaled, orientation_labels,
                                cv=cv, scoring='accuracy')
    cv_results = cross_validate(lda_full, firing_rates_scaled, orientation_labels,
                                cv=cv, scoring=['accuracy', 'f1_macro'],
                                return_train_score=True, return_estimator=True)

    lda_full.fit(firing_rates_scaled, orientation_labels)
    predictions = lda_full.predict(firing_rates_scaled)
    prediction_proba = lda_full.predict_proba(firing_rates_scaled)

    conf_matrix = confusion_matrix(orientation_labels, predictions, labels=unique_orientations)
    orientation_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    chance_accuracy = 1.0 / n_orientations

    results = {
        'lda_model': lda,
        'lda_full': lda_full,
        'scaler': scaler,
        'transformed_data': lda_transformed,
        'original_data': firing_rates_scaled,
        'orientation_labels': orientation_labels,
        'predictions': predictions,
        'prediction_proba': prediction_proba,
        'cv_scores': cv_scores,
        'cv_results': cv_results,
        'confusion_matrix': conf_matrix,
        'orientation_accuracies': orientation_accuracies,
        'unique_orientations': unique_orientations,
        'n_components': n_components,
        'chance_accuracy': chance_accuracy,
        'explained_variance_ratio': getattr(lda, 'explained_variance_ratio_', None)
    }

    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Overall accuracy: {accuracy_score(orientation_labels, predictions):.3f}")
    print(f"Chance level: {chance_accuracy:.3f}")
    print(f"Improvement over chance: {accuracy_score(orientation_labels, predictions) - chance_accuracy:.3f}")
    return results


# ---------------------------
# Plots
# ---------------------------

def create_grating_lda_plots(results, unit_ids, trial_info, save_path=None):
    """
    Returns:
      fig (matplotlib.figure.Figure)
    """
    plt.style.use('default')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(20, 16))

    transformed_data = results['transformed_data']
    labels = results['orientation_labels']
    unique_orientations = results['unique_orientations']
    conf_matrix = results['confusion_matrix']
    cv_scores = results['cv_scores']
    orientation_accuracies = results['orientation_accuracies']

    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])

    # 1. 3D scatter
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    if results['n_components'] >= 3:
        for i, orientation in enumerate(unique_orientations):
            mask = labels == orientation
            ax1.scatter(transformed_data[mask, 0], transformed_data[mask, 1],
                        transformed_data[mask, 2], c=[colors[i]],
                        label=f'{orientation}°', alpha=0.7, s=30)
        ax1.set_xlabel('LD1'); ax1.set_ylabel('LD2'); ax1.set_zlabel('LD3')
        ax1.set_title('LDA 3D Scatter Plot\n(Grating Orientations)', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 0.5, 'Need ≥3 components\nfor 3D visualization',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('LDA 3D Scatter Plot\n(Not enough components)', fontsize=14)

    # 2. 2D scatter / 1D jitter
    ax2 = fig.add_subplot(3, 4, 2)
    if results['n_components'] >= 2:
        for i, orientation in enumerate(unique_orientations):
            mask = labels == orientation
            ax2.scatter(transformed_data[mask, 0], transformed_data[mask, 1],
                        c=[colors[i]], label=f'{orientation}°', alpha=0.7, s=30)
        ax2.set_xlabel('LD1'); ax2.set_ylabel('LD2')
        ax2.set_title('LDA 2D Scatter Plot\n(Grating Orientations)', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        for i, orientation in enumerate(unique_orientations):
            mask = labels == orientation
            y_jitter = np.random.normal(0, 0.1, np.sum(mask))
            ax2.scatter(transformed_data[mask, 0], y_jitter,
                        c=[colors[i]], label=f'{orientation}°', alpha=0.7, s=30)
        ax2.set_xlabel('LD1'); ax2.set_ylabel('Random jitter')
        ax2.set_title('LDA 1D Projection\n(Grating Orientations)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # 3. Confusion matrix
    ax3 = fig.add_subplot(3, 4, 3)
    im = ax3.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set(xticks=np.arange(conf_matrix.shape[1]),
            yticks=np.arange(conf_matrix.shape[0]),
            xticklabels=[f'{ori}°' for ori in unique_orientations],
            yticklabels=[f'{ori}°' for ori in unique_orientations],
            title='Confusion Matrix\n(Orientation Classification)',
            ylabel='True Orientation', xlabel='Predicted Orientation')
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax3.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black",
                     fontsize=10)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    # 4. CV scores
    ax4 = fig.add_subplot(3, 4, 4)
    bars = ax4.bar(range(len(cv_scores)), cv_scores, alpha=0.7, color='skyblue')
    ax4.axhline(y=cv_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {cv_scores.mean():.3f}')
    ax4.axhline(y=results['chance_accuracy'], color='gray', linestyle=':',
                label=f'Chance: {results["chance_accuracy"]:.3f}')
    ax4.set_xlabel('CV Fold'); ax4.set_ylabel('Accuracy')
    ax4.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10); ax4.grid(True, alpha=0.3); ax4.set_ylim([0, 1])
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    # 5. Per-orientation accuracy
    ax5 = fig.add_subplot(3, 4, 5)
    bars = ax5.bar(range(len(orientation_accuracies)), orientation_accuracies, color=colors, alpha=0.7)
    ax5.axhline(y=results['chance_accuracy'], color='gray', linestyle=':', label='Chance level')
    ax5.set_xlabel('Orientation'); ax5.set_ylabel('Accuracy')
    ax5.set_title('Per-Orientation Accuracy', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(len(unique_orientations)))
    ax5.set_xticklabels([f'{ori}°' for ori in unique_orientations])
    ax5.grid(True, alpha=0.3, axis='y'); ax5.set_ylim([0, 1]); ax5.legend()
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")

    # 6. Polar decoding accuracy
    ax6 = fig.add_subplot(3, 4, 6, projection='polar')
    orientation_rad = np.deg2rad(np.array(unique_orientations))
    orientation_rad_doubled = 2 * orientation_rad
    ax6.plot(orientation_rad_doubled, orientation_accuracies, 'o-', linewidth=2, markersize=8)
    ax6.fill(orientation_rad_doubled, orientation_accuracies, alpha=0.25)
    ax6.set_ylim(0, 1)
    ax6.set_title('Orientation Decoding\nAccuracy (Polar)', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True)
    ax6.set_thetagrids(np.arange(0, 360, 45), [f'{int(a/2)}°' for a in np.arange(0, 360, 45)])

    # 7. LDA coefficients
    ax7 = fig.add_subplot(3, 4, (7, 8))
    if hasattr(results['lda_full'], 'coef_'):
        coef_matrix = results['lda_full'].coef_
        im = ax7.imshow(coef_matrix, cmap='RdBu_r', aspect='auto')
        ax7.figure.colorbar(im, ax=ax7)
        ax7.set_xlabel('Units'); ax7.set_ylabel('Orientation Discriminant')
        ax7.set_title('LDA Coefficients Heatmap', fontsize=14, fontweight='bold')
        ax7.set_yticks(range(len(unique_orientations)))
        ax7.set_yticklabels([f'{ori}°' for ori in unique_orientations])
        if len(unit_ids) <= 20:
            ax7.set_xticks(range(len(unit_ids)))
            ax7.set_xticklabels([uid.split('_')[-1] for uid in unit_ids], rotation=45)
        else:
            ax7.set_xlabel(f'Units (n={len(unit_ids)})')

    # 8. Trial distribution
    ax8 = fig.add_subplot(3, 4, 9)
    trial_counts = [trial_info['n_trials_per_orientation'][ori] for ori in unique_orientations]
    bars = ax8.bar(range(len(unique_orientations)), trial_counts, color=colors, alpha=0.7)
    ax8.set_xlabel('Orientation'); ax8.set_ylabel('Number of Trials')
    ax8.set_title('Trial Distribution', fontsize=14, fontweight='bold')
    ax8.set_xticks(range(len(unique_orientations)))
    ax8.set_xticklabels([f'{ori}°' for ori in unique_orientations])
    ax8.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{int(h)}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax8.get_xticklabels(), rotation=45, ha="right")

    # 9. Summary
    ax9 = fig.add_subplot(3, 4, 10)
    ax9.axis('off')
    overall_acc = accuracy_score(labels, results['predictions'])
    mean_cv_acc = cv_scores.mean(); std_cv_acc = cv_scores.std()
    exp_params = trial_info.get('experiment_parameters', {})
    stim_duration = exp_params.get('stimulus_duration', 'Unknown')
    iti_duration = exp_params.get('iti_duration', 'Unknown')
    summary_text = f"""
    Grating Classification Summary:

    Overall Accuracy: {overall_acc:.3f}
    CV Accuracy: {mean_cv_acc:.3f} ± {std_cv_acc:.3f}
    Chance Level: {results['chance_accuracy']:.3f}
    Above Chance: {overall_acc - results['chance_accuracy']:.3f}

    Experiment Info:
    • Total trials: {len(labels)}
    • Orientations: {len(unique_orientations)}
    • Units: {len(unit_ids)}
    • LDA components: {results['n_components']}
    • Stimulus duration: {stim_duration}s
    • ITI duration: {iti_duration}s

    Orientation Range: {min(unique_orientations)}° - {max(unique_orientations)}°
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # 10. Feature importance
    ax10 = fig.add_subplot(3, 4, 11)
    if hasattr(results['lda_full'], 'coef_'):
        feature_importance = np.mean(np.abs(results['lda_full'].coef_), axis=0)
        sorted_idx = np.argsort(feature_importance)[::-1]
        n_show = min(15, len(feature_importance))
        y_pos = np.arange(n_show)
        ax10.barh(y_pos, feature_importance[sorted_idx[:n_show]], alpha=0.7, color='orange')
        ax10.set_yticks(y_pos)
        if len(unit_ids) <= 50:
            ax10.set_yticklabels([unit_ids[i].split('_')[-1] for i in sorted_idx[:n_show]])
        else:
            ax10.set_yticklabels([f'Unit_{i}' for i in sorted_idx[:n_show]])
        ax10.invert_yaxis()
        ax10.set_xlabel('Mean |Coefficient|')
        ax10.set_title(f'Top {n_show} Most Discriminative Units', fontsize=14, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='x')

    # 11. Prediction confidence
    ax11 = fig.add_subplot(3, 4, 12)
    prediction_confidence = np.max(results['prediction_proba'], axis=1)
    correct_predictions = (results['predictions'] == labels)
    ax11.hist(prediction_confidence[correct_predictions], bins=20, alpha=0.7,
              label='Correct', color='green', density=True)
    ax11.hist(prediction_confidence[~correct_predictions], bins=20, alpha=0.7,
              label='Incorrect', color='red', density=True)
    ax11.set_xlabel('Prediction Confidence'); ax11.set_ylabel('Density')
    ax11.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax11.legend()

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to: {save_path}")

    return fig


# ---------------------------
# Extra: simple OSI per unit
# ---------------------------

def analyze_orientation_selectivity(lda_results, unit_ids, orientation_labels, firing_rates):
    """
    Compute a simple vector-sum orientation selectivity index (OSI) per unit.
    OSI = |sum_r r(theta)*exp(i*2theta)| / sum_r r(theta)
    Returns dict with per-unit OSI and preferred orientation.
    """
    orientations = np.unique(orientation_labels)
    theta = np.deg2rad(orientations)
    theta2 = 2 * theta  # orientation (pi-periodic)

    # Mean rate per unit per orientation
    mean_rates = []
    for ori in orientations:
        mask = (orientation_labels == ori)
        if np.any(mask):
            mean_rates.append(firing_rates[mask].mean(axis=0))  # (n_units,)
        else:
            mean_rates.append(np.zeros(firing_rates.shape[1]))
    mean_rates = np.stack(mean_rates, axis=0)  # (n_oris, n_units)

    # Vector sum across orientations
    comp = np.exp(1j * theta2)[:, None]  # (n_oris, 1)
    vec = (mean_rates * comp).sum(axis=0)  # (n_units,)
    mag = np.abs(vec)
    denom = mean_rates.sum(axis=0) + 1e-12
    osi = mag / denom

    # Preferred orientation (half-angle)
    pref_angle = (np.angle(vec) / 2.0)  # in radians
    pref_deg = (np.rad2deg(pref_angle) % 180.0)

    results = {
        'unit_ids': unit_ids,
        'osi': osi,
        'preferred_orientation_deg': pref_deg
    }

    # Print a short summary
    print("\nOrientation Selectivity (vector-sum OSI):")
    print(f"- Mean OSI: {osi.mean():.3f}")
    top_k = min(10, len(unit_ids))
    top_idx = np.argsort(osi)[::-1][:top_k]
    for i in top_idx:
        print(f"  {unit_ids[i]}: OSI={osi[i]:.3f}, Pref≈{pref_deg[i]:.1f}°")
    return results


# ---------------------------
# Main driver
# ---------------------------

def main_grating_analysis(data_path, time_window=(0.07, 0.16), save_plots=True, fig_out=None):
    """
    End-to-end:
      1) load data
      2) compute firing rates
      3) LDA
      4) figures
    Returns:
      lda_results, firing_rates, orientation_labels, unit_ids
    """
    data = load_neural_data(data_path)
    firing_rates, orientation_labels, unit_ids, trial_info = calculate_grating_firing_rates(
        data, time_window=time_window
    )
    if len(orientation_labels) == 0 or firing_rates.size == 0:
        raise ValueError("No valid trials or firing rates computed. Check data/time_window.")

    lda_results = perform_grating_lda_analysis(firing_rates, orientation_labels)

    fig = create_grating_lda_plots(
        lda_results, unit_ids, trial_info,
        save_path=(Path(fig_out) if save_plots and fig_out else
                   (Path(data_path).with_suffix('.lda_overview.png') if save_plots else None))
    )
    return lda_results, firing_rates, orientation_labels, unit_ids


if __name__ == "__main__":
    # CHANGE THIS PATH
    data_file_path = "/Volumes/xieluanlabs/xl_cl/code/sortout/CnL22sg/CnL22SG_20250821_183152/embedding_analysis/CnL22SG_CnL22SG_20250821_183152_grating_data.pkl"

    try:
        lda_results, firing_rates, orientation_labels, unit_ids = main_grating_analysis(
            data_path=data_file_path,
            time_window=(0.07, 0.16),
            save_plots=True
        )

        # Optional: unit selectivity
        print("\nRunning additional orientation selectivity analysis...")
        _unit_selectivity = analyze_orientation_selectivity(
            lda_results, unit_ids, orientation_labels, firing_rates
        )

    except FileNotFoundError:
        print("Data file not found. Please update the data_file_path variable.")
    except Exception as e:
        print(f"Error: {e}")