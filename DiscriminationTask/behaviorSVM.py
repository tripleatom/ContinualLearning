"""
SVM Analysis for Behavior Trial Neural Data
Classifies trials based on white stimulus position (left vs right)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import h5py
import json
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_neural_data(filepath):
    """Load neural data from pickle format."""
    filepath = Path(filepath)

    if filepath.suffix != '.pkl':
        raise ValueError(f"Expected .pkl file, got {filepath.suffix}")

    print(f"Loading data from: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# ANALYSIS
# =============================================================================

def calculate_firing_rates(neural_data, time_window=(0.0, None)):
    """
    Calculate firing rates for each unit in each trial.

    Parameters:
    -----------
    neural_data : dict
        Neural data dictionary from PKL file
    time_window : tuple
        (start, end) time window relative to trial start in seconds.
        If end is None, uses trial duration

    Returns:
    --------
    firing_rates : ndarray
        (n_trials, n_units) array of firing rates
    condition_labels : ndarray
        (n_trials,) array of condition labels (0=right, 1=left)
    unit_ids : list
        List of unit identifiers
    trial_info : dict
        Experiment metadata
    """
    window_start = time_window[0]

    unit_info = neural_data.get('unit_info', {})
    unit_ids = [
        uid for uid in neural_data['spike_data'].keys()
        if unit_info.get(uid, {}).get('quality', 'unknown') != 'noise'
    ]
    n_noise = len(neural_data['spike_data']) - len(unit_ids)
    if n_noise:
        print(f"  Excluded {n_noise} noise unit(s)")

    trial_info_dict = neural_data['trial_info']
    if 'white_on_left' in trial_info_dict:
        condition_key = 'white_on_left'
        left_label, right_label = 'White on Left', 'White on Right'
    elif 'rewarded_on_left' in trial_info_dict:
        condition_key = 'rewarded_on_left'
        left_label, right_label = 'Rewarded on Left', 'Rewarded on Right'
    else:
        raise KeyError("trial_info must contain 'white_on_left' or 'rewarded_on_left'")

    white_on_left = np.array(trial_info_dict[condition_key])
    n_trials = len(white_on_left)
    n_units = len(unit_ids)

    print(f"\nCalculating firing rates:")
    print(f"  Units: {n_units} | Trials: {n_trials}")
    print(f"  Window start: {window_start:.3f}s")
    print(f"  Conditions:")
    print(f"    {left_label}:  {np.sum(white_on_left)} trials")
    print(f"    {right_label}: {np.sum(~white_on_left)} trials")

    firing_rates = np.full((n_trials, n_units), np.nan)

    for unit_idx, unit_id in enumerate(unit_ids):
        for trial_data in neural_data['spike_data'][unit_id]:
            trial_idx = int(trial_data['trial_index'])
            if 0 <= trial_idx < n_trials:
                spike_times = np.array(trial_data['spike_times'])
                trial_duration = trial_data['trial_duration']

                if time_window[1] is None:
                    window_end = trial_duration
                else:
                    window_end = min(time_window[1], trial_duration)

                window_duration = window_end - window_start

                spikes_in_window = np.sum((spike_times >= window_start) &
                                         (spike_times < window_end))
                firing_rates[trial_idx, unit_idx] = spikes_in_window / window_duration

    valid_mask = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_mask]
    condition_labels = white_on_left[valid_mask].astype(int)

    print(f"  Valid trials: {np.sum(valid_mask)}/{n_trials}")
    print(f"  Mean firing rate: {np.mean(firing_rates_clean):.2f} Hz")
    print(f"  Window duration: {window_duration:.3f}s")

    trial_info = {
        'valid_trials_mask': valid_mask,
        'unique_conditions': [0, 1],
        'condition_names': {0: right_label, 1: left_label},
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_trials_per_condition': {
            0: int(np.sum(condition_labels == 0)),
            1: int(np.sum(condition_labels == 1))
        },
        'window_duration': window_duration
    }

    return firing_rates_clean, condition_labels, unit_ids, trial_info


def perform_svm_analysis(firing_rates, condition_labels, C=1.0):
    """
    Perform linear SVM analysis with cross-validation for binary classification.

    Returns:
    --------
    dict : Dictionary with SVM results including:
        - transformed_data (decision scores), cv_scores, model coefficients, etc.
    """
    unique_conditions = np.unique(condition_labels)
    n_conditions = len(unique_conditions)
    n_features = firing_rates.shape[1]

    print(f"\nSVM Analysis (linear kernel, C={C}):")
    print(f"  Conditions: {n_conditions} (0=Right, 1=Left)")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(condition_labels)}")

    min_trials = min(np.sum(condition_labels == cond) for cond in unique_conditions)
    print(f"  Min trials per class: {int(min_trials)}")

    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)

    cv_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    clf_cv = SVC(kernel='linear', C=C)
    cv_scores = cross_val_score(clf_cv, firing_rates_scaled, condition_labels,
                                cv=cv, scoring='accuracy')
    cv_results = cross_validate(clf_cv, firing_rates_scaled, condition_labels,
                                cv=cv, scoring=['accuracy', 'f1_macro'],
                                return_train_score=True, return_estimator=True)

    # Fit full model for weights and decision scores (visualization only, not performance)
    model = SVC(kernel='linear', C=C)
    model.fit(firing_rates_scaled, condition_labels)
    decision_scores = model.decision_function(firing_rates_scaled).reshape(-1, 1)

    chance_accuracy = 1.0 / n_conditions

    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Chance level: {chance_accuracy:.3f}")
    print(f"  Above chance: {cv_scores.mean() - chance_accuracy:+.3f}")

    return {
        'model': model,
        'scaler': scaler,
        'transformed_data': decision_scores,
        'original_data': firing_rates_scaled,
        'condition_labels': condition_labels,
        'cv_scores': cv_scores,
        'cv_results': cv_results,
        'unique_conditions': unique_conditions,
        'chance_accuracy': chance_accuracy,
        'C': C,
    }


def calculate_stimulus_selectivity(unit_ids, condition_labels, firing_rates):
    """
    Calculate stimulus selectivity index for white position.
    Selectivity = (R_left - R_right) / (R_left + R_right)
    """
    rate_right = firing_rates[condition_labels == 0].mean(axis=0)
    rate_left = firing_rates[condition_labels == 1].mean(axis=0)

    selectivity = (rate_left - rate_right) / (rate_left + rate_right + 1e-12)
    preferred_condition = (selectivity > 0).astype(int)

    print(f"\nStimulus Selectivity:")
    print(f"  Mean |selectivity|: {np.abs(selectivity).mean():.3f}")
    print(f"  Units preferring left:  {np.sum(preferred_condition == 1)}")
    print(f"  Units preferring right: {np.sum(preferred_condition == 0)}")
    print(f"\n  Top selective units:")

    for idx in np.argsort(np.abs(selectivity))[::-1][:min(10, len(unit_ids))]:
        pref = "Left" if preferred_condition[idx] == 1 else "Right"
        print(f"    {unit_ids[idx]}: SI={selectivity[idx]:+.3f}, Pref={pref}")

    return {
        'unit_ids': unit_ids,
        'selectivity_index': selectivity,
        'preferred_condition': preferred_condition,
        'rate_left': rate_left,
        'rate_right': rate_right
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_figure(results, unit_ids, trial_info, save_path=None):
    """Create comprehensive SVM analysis visualization."""
    plt.style.use('default')

    fig = plt.figure(figsize=(18, 12))

    transformed = results['transformed_data']
    labels = results['condition_labels']
    condition_names = trial_info['condition_names']
    colors = ['#FF6B6B', '#4ECDC4']

    _plot_decision_score_projection(fig, transformed, labels, condition_names, colors)
    _plot_decision_score_histogram(fig, transformed, labels, condition_names, colors)
    _plot_cv_scores(fig, results)
    _plot_svm_weights(fig, results, unit_ids)
    _plot_trial_distribution(fig, trial_info, colors)
    _plot_summary_text(fig, results, labels, unit_ids, trial_info)
    _plot_feature_importance(fig, results, unit_ids)
    _plot_firing_rate_comparison(fig, results, unit_ids, labels)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved combined figure to: {save_path}")

        _save_individual_subfigures(results, unit_ids, trial_info, save_path)

    return fig


def _plot_decision_score_projection(fig, data, labels, condition_names, colors):
    ax = fig.add_subplot(3, 3, 1)

    for i, (cond, name) in enumerate(condition_names.items()):
        mask = labels == cond
        y_jitter = np.random.normal(i, 0.1, np.sum(mask))
        ax.scatter(data[mask, 0], y_jitter, c=[colors[i]], label=name, alpha=0.6, s=40)

    ax.set_xlabel('Decision Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(list(condition_names.values()))
    ax.set_title('SVM Decision Score Projection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_decision_score_histogram(fig, data, labels, condition_names, colors):
    ax = fig.add_subplot(3, 3, 2)

    for i, (cond, name) in enumerate(condition_names.items()):
        mask = labels == cond
        ax.hist(data[mask, 0], bins=30, alpha=0.6, color=colors[i],
                label=name, density=True)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2,
               label='Decision boundary')
    ax.set_xlabel('Decision Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('SVM Decision Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_cv_scores(fig, results):
    ax = fig.add_subplot(3, 3, 3)

    scores = results['cv_scores']
    bars = ax.bar(range(len(scores)), scores, alpha=0.7, color='skyblue')
    ax.axhline(scores.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {scores.mean():.3f}')
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', linewidth=2,
               label=f'Chance: {results["chance_accuracy"]:.3f}')

    ax.set(xlabel='CV Fold', ylabel='Accuracy', ylim=[0, 1],
           title='Cross-Validation Scores')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)


def _plot_svm_weights(fig, results, unit_ids):
    ax = fig.add_subplot(3, 3, (4, 5))

    if hasattr(results['model'], 'coef_'):
        coef = results['model'].coef_[0]
        sorted_idx = np.argsort(np.abs(coef))[::-1][:20]

        colors_bar = ['#4ECDC4' if c > 0 else '#FF6B6B' for c in coef[sorted_idx]]

        ax.barh(range(len(sorted_idx)), coef[sorted_idx], color=colors_bar, alpha=0.7)
        ax.set(yticks=range(len(sorted_idx)),
               yticklabels=[unit_ids[i] for i in sorted_idx],
               xlabel='SVM Weight',
               title='Top Discriminative Units (Positive=Left, Negative=Right)')
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')


def _plot_trial_distribution(fig, trial_info, colors):
    ax = fig.add_subplot(3, 3, 6)

    condition_names = trial_info['condition_names']
    counts = [trial_info['n_trials_per_condition'][i] for i in range(2)]
    labels = [condition_names[i] for i in range(2)]

    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.7)

    ax.set(xlabel='Condition', ylabel='Number of Trials',
           title='Trial Distribution',
           xticks=range(len(labels)),
           xticklabels=labels)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, int(h),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _plot_summary_text(fig, results, labels, unit_ids, trial_info):
    ax = fig.add_subplot(3, 3, 7)
    ax.axis('off')

    exp_params = trial_info.get('experiment_parameters', {})

    summary = f"""
    Classification Summary (SVM)

    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}
    Above Chance: {results['cv_scores'].mean() - results['chance_accuracy']:+.3f}

    Experiment Info:
    • Total trials: {len(labels)}
    • Conditions: 2 (Left/Right)
    • Units: {len(unit_ids)}
    • Kernel: linear
    • C (regularization): {results['C']:.1f}
    • Trial duration: {exp_params.get('trial_duration', 'N/A'):.3f}s
    • Left trials:  {trial_info['n_trials_per_condition'][1]}
    • Right trials: {trial_info['n_trials_per_condition'][0]}
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def _plot_feature_importance(fig, results, unit_ids):
    ax = fig.add_subplot(3, 3, 8)

    if hasattr(results['model'], 'coef_'):
        importance = np.abs(results['model'].coef_[0])
        top_idx = np.argsort(importance)[::-1][:15]

        ax.barh(range(len(top_idx)), importance[top_idx], alpha=0.7, color='orange')
        ax.set(yticks=range(len(top_idx)),
               yticklabels=[unit_ids[i] for i in top_idx],
               xlabel='|SVM Weight|',
               title='Top Discriminative Units')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')


def _plot_firing_rate_comparison(fig, results, unit_ids, labels):
    ax = fig.add_subplot(3, 3, 9)

    firing_rates = results['original_data']
    mean_right = firing_rates[labels == 0].mean(axis=0)
    mean_left = firing_rates[labels == 1].mean(axis=0)

    diff = mean_left - mean_right
    sorted_idx = np.argsort(diff)

    n_show = min(20, len(unit_ids))
    show_idx = np.concatenate([sorted_idx[:n_show//2], sorted_idx[-(n_show//2):]])

    x = np.arange(len(show_idx))
    width = 0.35

    ax.bar(x - width/2, mean_right[show_idx], width, label='Right',
           color='#FF6B6B', alpha=0.7)
    ax.bar(x + width/2, mean_left[show_idx], width, label='Left',
           color='#4ECDC4', alpha=0.7)

    ax.set(ylabel='Mean Firing Rate (Hz)',
           title='Unit Firing Rates by Condition',
           xticks=x,
           xticklabels=[unit_ids[i] for i in show_idx])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)


def _save_individual_subfigures(results, unit_ids, trial_info, base_save_path):
    """Save each subfigure as an individual file in the behavior_analysis folder."""
    base_path = Path(base_save_path)
    out_folder = base_path.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    transformed = results['transformed_data']
    labels = results['condition_labels']
    condition_names = trial_info['condition_names']
    colors = ['#FF6B6B', '#4ECDC4']

    print(f"\nSaving individual subfigures to: {out_folder}")

    # 1. Decision Score Projection
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cond, name) in enumerate(condition_names.items()):
            mask = labels == cond
            y_jitter = np.random.normal(i, 0.1, np.sum(mask))
            ax.scatter(transformed[mask, 0], y_jitter, c=[colors[i]], label=name, alpha=0.6, s=40)
        ax.set_xlabel('Decision Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(list(condition_names.values()))
        ax.set_title('SVM Decision Score Projection', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_folder / "svm_1_decision_projection.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: svm_1_decision_projection.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_1_decision_projection - {e}")

    # 2. Decision Score Histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cond, name) in enumerate(condition_names.items()):
            mask = labels == cond
            ax.hist(transformed[mask, 0], bins=30, alpha=0.6, color=colors[i], label=name, density=True)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Decision boundary')
        ax.set_xlabel('Decision Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('SVM Decision Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(out_folder / "svm_2_decision_histogram.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: svm_2_decision_histogram.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_2_decision_histogram - {e}")

    # 3. CV Scores
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        scores = results['cv_scores']
        bars = ax.bar(range(len(scores)), scores, alpha=0.7, color='skyblue')
        ax.axhline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
        ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', linewidth=2,
                   label=f'Chance: {results["chance_accuracy"]:.3f}')
        ax.set(xlabel='CV Fold', ylabel='Accuracy', ylim=[0, 1], title='Cross-Validation Scores')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        fig.savefig(out_folder / "svm_3_cv_scores.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: svm_3_cv_scores.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_3_cv_scores - {e}")

    # 4. SVM Weights
    try:
        if hasattr(results['model'], 'coef_'):
            fig, ax = plt.subplots(figsize=(10, 8))
            coef = results['model'].coef_[0]
            sorted_idx = np.argsort(np.abs(coef))[::-1][:20]
            colors_bar = ['#4ECDC4' if c > 0 else '#FF6B6B' for c in coef[sorted_idx]]
            ax.barh(range(len(sorted_idx)), coef[sorted_idx], color=colors_bar, alpha=0.7)
            ax.set(yticks=range(len(sorted_idx)), yticklabels=[unit_ids[i] for i in sorted_idx],
                   xlabel='SVM Weight', title='Top Discriminative Units (Positive=Left, Negative=Right)')
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            fig.savefig(out_folder / "svm_4_weights.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved: svm_4_weights.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_4_weights - {e}")

    # 5. Trial Distribution
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = [trial_info['n_trials_per_condition'][i] for i in range(2)]
        label_names = [condition_names[i] for i in range(2)]
        bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.7)
        ax.set(xlabel='Condition', ylabel='Number of Trials', title='Trial Distribution',
               xticks=range(len(label_names)), xticklabels=label_names)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, int(h), ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(out_folder / "svm_5_trial_distribution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: svm_5_trial_distribution.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_5_trial_distribution - {e}")

    # 6. Summary Text
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        exp_params = trial_info.get('experiment_parameters', {})
        summary = f"""
    Classification Summary (SVM)

    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}
    Above Chance: {results['cv_scores'].mean() - results['chance_accuracy']:+.3f}

    Experiment Info:
    • Total trials: {len(labels)}
    • Conditions: 2 (Left/Right)
    • Units: {len(unit_ids)}
    • Kernel: linear
    • C (regularization): {results['C']:.1f}
    • Trial duration: {exp_params.get('trial_duration', 'N/A'):.3f}s
    • Left trials:  {trial_info['n_trials_per_condition'][1]}
    • Right trials: {trial_info['n_trials_per_condition'][0]}
        """
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        fig.savefig(out_folder / "svm_6_summary_text.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: svm_6_summary_text.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_6_summary_text - {e}")

    # 7. Feature Importance
    try:
        if hasattr(results['model'], 'coef_'):
            fig, ax = plt.subplots(figsize=(8, 8))
            importance = np.abs(results['model'].coef_[0])
            top_idx = np.argsort(importance)[::-1][:15]
            ax.barh(range(len(top_idx)), importance[top_idx], alpha=0.7, color='orange')
            ax.set(yticks=range(len(top_idx)), yticklabels=[unit_ids[i] for i in top_idx],
                   xlabel='|SVM Weight|', title='Top Discriminative Units')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            fig.savefig(out_folder / "svm_7_feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved: svm_7_feature_importance.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_7_feature_importance - {e}")

    # 8. Firing Rate Comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        firing_rates = results['original_data']
        mean_right = firing_rates[labels == 0].mean(axis=0)
        mean_left = firing_rates[labels == 1].mean(axis=0)
        diff = mean_left - mean_right
        sorted_idx = np.argsort(diff)
        n_show = min(20, len(unit_ids))
        show_idx = np.concatenate([sorted_idx[:n_show//2], sorted_idx[-(n_show//2):]])
        x = np.arange(len(show_idx))
        width = 0.35
        ax.bar(x - width/2, mean_right[show_idx], width, label='Right', color='#FF6B6B', alpha=0.7)
        ax.bar(x + width/2, mean_left[show_idx], width, label='Left', color='#4ECDC4', alpha=0.7)
        ax.set(ylabel='Mean Firing Rate (Hz)', title='Unit Firing Rates by Condition',
               xticks=x, xticklabels=[unit_ids[i] for i in show_idx])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
        plt.tight_layout()
        fig.savefig(out_folder / "svm_8_firing_rate_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: svm_8_firing_rate_comparison.png")
    except Exception as e:
        print(f"  ✗ Failed: svm_8_firing_rate_comparison - {e}")

    print(f"\nAll individual subfigures saved to: {out_folder}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.0, None), C=1.0, save_plots=True,
                 output_path=None):
    """
    Complete analysis pipeline: load → analyze → visualize.

    Args:
        data_path: Path to neural data PKL file
        time_window: Tuple of (start, end) time in seconds relative to trial start.
                    If end is None, uses entire trial duration.
        C: SVM regularization parameter (smaller = more regularization)
        save_plots: Whether to save the figure
        output_path: Custom output path

    Returns:
        Tuple of (svm_results, firing_rates, condition_labels, unit_ids, selectivity)
    """
    data = load_neural_data(data_path)
    firing_rates, condition_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )

    if len(condition_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")

    svm_results = perform_svm_analysis(firing_rates, condition_labels, C=C)

    print("\nCalculating stimulus selectivity...")
    selectivity = calculate_stimulus_selectivity(unit_ids, condition_labels, firing_rates)

    if output_path is None and save_plots:
        session_dir = Path(data_path).parent
        output_path = session_dir / "behavior_analysis" / (Path(data_path).stem + ".svm_analysis.png")

    fig = create_analysis_figure(
        svm_results, unit_ids, trial_info,
        save_path=output_path if save_plots else None
    )

    return svm_results, firing_rates, condition_labels, unit_ids, selectivity


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313\behavior_trial_embedding_20260423_1631.pkl"

    try:
        results = run_analysis(
            data_path=DATA_PATH,
            time_window=(0.0, 1.0),
            C=1.0,
            save_plots=True
        )

        svm_results, firing_rates, condition_labels, unit_ids, selectivity = results

        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60)
        print(f"\nKey Results:")
        print(f"  Classification accuracy: {svm_results['cv_scores'].mean():.3f}")
        print(f"  Above chance: {svm_results['cv_scores'].mean() - 0.5:+.3f}")
        print(f"  Most selective unit: {selectivity['unit_ids'][np.argmax(np.abs(selectivity['selectivity_index']))]}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
