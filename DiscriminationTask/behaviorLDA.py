"""
LDA Analysis for Behavior Trial Neural Data
Classifies trials based on white stimulus position (left vs right)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import h5py
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
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

    # Support both white_on_left (old format) and rewarded_on_left (grating format)
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

    # Print summary
    print(f"\nCalculating firing rates:")
    print(f"  Units: {n_units} | Trials: {n_trials}")
    print(f"  Window start: {window_start:.3f}s")
    print(f"  Conditions:")
    print(f"    {left_label}:  {np.sum(white_on_left)} trials")
    print(f"    {right_label}: {np.sum(~white_on_left)} trials")
    
    # Calculate firing rates
    firing_rates = np.full((n_trials, n_units), np.nan)
    
    for unit_idx, unit_id in enumerate(unit_ids):
        for trial_data in neural_data['spike_data'][unit_id]:
            trial_idx = int(trial_data['trial_index'])
            if 0 <= trial_idx < n_trials:
                spike_times = np.array(trial_data['spike_times'])
                trial_duration = trial_data['trial_duration']
                
                # Determine window end
                if time_window[1] is None:
                    window_end = trial_duration
                else:
                    window_end = min(time_window[1], trial_duration)
                
                window_duration = window_end - window_start
                
                # Count spikes in window
                spikes_in_window = np.sum((spike_times >= window_start) & 
                                         (spike_times < window_end))
                firing_rates[trial_idx, unit_idx] = spikes_in_window / window_duration
    
    # Remove trials with missing data
    valid_mask = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_mask]
    condition_labels = white_on_left[valid_mask].astype(int)  # 0=right, 1=left
    
    print(f"  Valid trials: {np.sum(valid_mask)}/{n_trials}")
    print(f"  Mean firing rate: {np.mean(firing_rates_clean):.2f} Hz")
    print(f"  Window duration: {window_duration:.3f}s")
    
    trial_info = {
        'valid_trials_mask': valid_mask,
        'unique_conditions': [0, 1],  # Right, Left
        'condition_names': {0: right_label, 1: left_label},
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_trials_per_condition': {
            0: int(np.sum(condition_labels == 0)),
            1: int(np.sum(condition_labels == 1))
        },
        'window_duration': window_duration
    }
    
    return firing_rates_clean, condition_labels, unit_ids, trial_info


def perform_lda_analysis(firing_rates, condition_labels):
    """
    Perform LDA analysis with cross-validation for binary classification.
    
    Returns:
    --------
    dict : Dictionary with LDA results including:
        - transformed_data, predictions, cv_scores, confusion_matrix, etc.
    """
    unique_conditions = np.unique(condition_labels)
    n_conditions = len(unique_conditions)
    n_features = firing_rates.shape[1]
    
    print(f"\nLDA Analysis:")
    print(f"  Conditions: {n_conditions} (0=Right, 1=Left)")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(condition_labels)}")
    
    # For binary classification, LDA has only 1 component
    n_components = min(n_conditions - 1, n_features)
    
    # Count trials per condition
    min_trials = min(np.sum(condition_labels == cond) for cond in unique_conditions)
    print(f"  Min trials per class: {int(min_trials)}")
    print(f"  LDA components: {n_components}")
    
    # Standardize features
    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)
    
    # Fit LDA for dimensionality reduction (1D for binary)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_transformed = lda.fit_transform(firing_rates_scaled, condition_labels)
    
    # Cross-validation
    cv_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    lda_full = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda_full, firing_rates_scaled, condition_labels,
                                cv=cv, scoring='accuracy')
    cv_results = cross_validate(lda_full, firing_rates_scaled, condition_labels,
                                cv=cv, scoring=['accuracy', 'f1_macro'],
                                return_train_score=True, return_estimator=True)
    
    # Full model predictions
    lda_full.fit(firing_rates_scaled, condition_labels)
    predictions = lda_full.predict(firing_rates_scaled)
    prediction_proba = lda_full.predict_proba(firing_rates_scaled)
    
    # Performance metrics
    conf_matrix = confusion_matrix(condition_labels, predictions, 
                                   labels=unique_conditions)
    condition_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    chance_accuracy = 1.0 / n_conditions
    overall_accuracy = accuracy_score(condition_labels, predictions)
    
    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Overall accuracy: {overall_accuracy:.3f}")
    print(f"  Chance level: {chance_accuracy:.3f}")
    print(f"  Above chance: {overall_accuracy - chance_accuracy:+.3f}")
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)  # True positive rate (left trials)
    specificity = tn / (tn + fp)  # True negative rate (right trials)
    
    print(f"  Sensitivity (Left):  {sensitivity:.3f}")
    print(f"  Specificity (Right): {specificity:.3f}")
    
    return {
        'lda_model': lda,
        'lda_full': lda_full,
        'scaler': scaler,
        'transformed_data': lda_transformed,
        'original_data': firing_rates_scaled,
        'condition_labels': condition_labels,
        'predictions': predictions,
        'prediction_proba': prediction_proba,
        'cv_scores': cv_scores,
        'cv_results': cv_results,
        'confusion_matrix': conf_matrix,
        'condition_accuracies': condition_accuracies,
        'unique_conditions': unique_conditions,
        'n_components': n_components,
        'chance_accuracy': chance_accuracy,
        'explained_variance_ratio': getattr(lda, 'explained_variance_ratio_', None),
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def calculate_stimulus_selectivity(unit_ids, condition_labels, firing_rates):
    """
    Calculate stimulus selectivity index for white position.
    Selectivity = (R_left - R_right) / (R_left + R_right)
    
    Returns:
    --------
    dict : Dictionary with unit_ids, selectivity_index, and preferred_condition
    """
    # Calculate mean firing rate per condition
    rate_right = firing_rates[condition_labels == 0].mean(axis=0)
    rate_left = firing_rates[condition_labels == 1].mean(axis=0)
    
    # Selectivity index: positive = prefers left, negative = prefers right
    selectivity = (rate_left - rate_right) / (rate_left + rate_right + 1e-12)
    preferred_condition = (selectivity > 0).astype(int)  # 0=right, 1=left
    
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
    """Create comprehensive LDA analysis visualization."""
    plt.style.use('default')

    fig = plt.figure(figsize=(20, 16))

    # Extract commonly used data
    transformed = results['transformed_data']
    labels = results['condition_labels']
    condition_names = trial_info['condition_names']
    colors = ['#FF6B6B', '#4ECDC4']  # Red for right, Teal for left

    # Create subplots
    _plot_1d_lda_projection(fig, transformed, labels, condition_names, colors)
    _plot_1d_histogram(fig, transformed, labels, condition_names, colors)
    _plot_confusion_matrix(fig, results['confusion_matrix'], condition_names)
    _plot_cv_scores(fig, results)
    _plot_per_condition_accuracy(fig, results, condition_names, colors)
    _plot_roc_style(fig, results)
    _plot_lda_coefficients(fig, results, unit_ids)
    _plot_trial_distribution(fig, trial_info, colors)
    _plot_summary_text(fig, results, labels, unit_ids, trial_info)
    _plot_feature_importance(fig, results, unit_ids)
    _plot_prediction_confidence(fig, results, labels, condition_names, colors)
    _plot_firing_rate_comparison(fig, results, unit_ids, labels)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved combined figure to: {save_path}")

        # Save individual subfigures
        _save_individual_subfigures(results, unit_ids, trial_info, save_path)

    return fig


def _plot_1d_lda_projection(fig, data, labels, condition_names, colors):
    """Plot 1D LDA projection with jitter."""
    ax = fig.add_subplot(3, 4, 1)
    
    for i, (cond, name) in enumerate(condition_names.items()):
        mask = labels == cond
        y_jitter = np.random.normal(i, 0.1, np.sum(mask))
        ax.scatter(data[mask, 0], y_jitter,
                  c=[colors[i]], label=name, alpha=0.6, s=40)
    
    ax.set_xlabel('LD1 (Discriminant Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(list(condition_names.values()))
    ax.set_title('LDA 1D Projection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_1d_histogram(fig, data, labels, condition_names, colors):
    """Plot histogram of LDA scores."""
    ax = fig.add_subplot(3, 4, 2)
    
    for i, (cond, name) in enumerate(condition_names.items()):
        mask = labels == cond
        ax.hist(data[mask, 0], bins=30, alpha=0.6, color=colors[i],
               label=name, density=True)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2,
              label='Decision boundary')
    ax.set_xlabel('LD1 Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('LDA Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_confusion_matrix(fig, conf_matrix, condition_names):
    """Plot confusion matrix."""
    ax = fig.add_subplot(3, 4, 3)
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    
    labels = [condition_names[i] for i in range(len(condition_names))]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix', ylabel='True', xlabel='Predicted')
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(2):
        for j in range(2):
            color = 'white' if conf_matrix[i, j] > thresh else 'black'
            ax.text(j, i, int(conf_matrix[i, j]), ha='center', va='center',
                   color=color, fontsize=14, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _plot_cv_scores(fig, results):
    """Plot cross-validation scores."""
    ax = fig.add_subplot(3, 4, 4)
    
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
    
    # Add value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)


def _plot_per_condition_accuracy(fig, results, condition_names, colors):
    """Plot per-condition classification accuracy."""
    ax = fig.add_subplot(3, 4, 5)
    
    accuracies = results['condition_accuracies']
    labels = [condition_names[i] for i in range(len(condition_names))]
    
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', linewidth=2,
               label='Chance')
    
    ax.set(xlabel='Condition', ylabel='Accuracy', ylim=[0, 1],
           title='Per-Condition Accuracy',
           xticks=range(len(labels)),
           xticklabels=labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _plot_roc_style(fig, results):
    """Plot sensitivity vs specificity."""
    ax = fig.add_subplot(3, 4, 6)
    
    sensitivity = results['sensitivity']
    specificity = results['specificity']
    
    # Plot point
    ax.plot(1 - specificity, sensitivity, 'ro', markersize=15, 
           label=f'Model\n(Sens={sensitivity:.3f}, Spec={specificity:.3f})')
    
    # Plot chance line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance')
    
    ax.set(xlabel='False Positive Rate (1 - Specificity)',
           ylabel='True Positive Rate (Sensitivity)',
           xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title='Sensitivity vs Specificity', aspect='equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_lda_coefficients(fig, results, unit_ids):
    """Plot LDA coefficient bar chart."""
    ax = fig.add_subplot(3, 4, (7, 8))
    
    if hasattr(results['lda_full'], 'coef_'):
        coef = results['lda_full'].coef_[0]  # Binary classification: shape (1, n_features)
        sorted_idx = np.argsort(np.abs(coef))[::-1][:20]  # Top 20
        
        colors_bar = ['#4ECDC4' if c > 0 else '#FF6B6B' for c in coef[sorted_idx]]
        
        ax.barh(range(len(sorted_idx)), coef[sorted_idx], color=colors_bar, alpha=0.7)
        ax.set(yticks=range(len(sorted_idx)),
               yticklabels=[unit_ids[i] for i in sorted_idx],
               xlabel='LDA Coefficient',
               title='Top Discriminative Units (Positive=Left, Negative=Right)')
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')


def _plot_trial_distribution(fig, trial_info, colors):
    """Plot trial count distribution."""
    ax = fig.add_subplot(3, 4, 9)
    
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
    """Plot summary statistics text."""
    ax = fig.add_subplot(3, 4, 10)
    ax.axis('off')
    
    exp_params = trial_info.get('experiment_parameters', {})
    
    summary = f"""
    Classification Summary
    
    Overall Accuracy: {accuracy_score(labels, results['predictions']):.3f}
    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}
    
    Sensitivity (Left):  {results['sensitivity']:.3f}
    Specificity (Right): {results['specificity']:.3f}
    
    Experiment Info:
    • Total trials: {len(labels)}
    • Conditions: 2 (Left/Right)
    • Units: {len(unit_ids)}
    • LDA components: {results['n_components']}
    • Trial duration: {exp_params.get('trial_duration', 'N/A'):.3f}s
    • Left trials:  {trial_info['n_trials_per_condition'][1]}
    • Right trials: {trial_info['n_trials_per_condition'][0]}
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def _plot_feature_importance(fig, results, unit_ids):
    """Plot feature importance based on LDA coefficients."""
    ax = fig.add_subplot(3, 4, 11)
    
    if hasattr(results['lda_full'], 'coef_'):
        importance = np.abs(results['lda_full'].coef_[0])
        top_idx = np.argsort(importance)[::-1][:15]
        
        ax.barh(range(len(top_idx)), importance[top_idx], alpha=0.7, color='orange')
        ax.set(yticks=range(len(top_idx)),
               yticklabels=[unit_ids[i] for i in top_idx],
               xlabel='|LDA Coefficient|',
               title='Top Discriminative Units')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')


def _plot_prediction_confidence(fig, results, labels, condition_names, colors):
    """Plot distribution of prediction confidence."""
    ax = fig.add_subplot(3, 4, 12)
    
    confidence = np.max(results['prediction_proba'], axis=1)
    correct = results['predictions'] == labels
    
    ax.hist(confidence[correct], bins=20, alpha=0.7, label='Correct',
            color='green', density=True)
    ax.hist(confidence[~correct], bins=20, alpha=0.7, label='Incorrect',
            color='red', density=True)
    
    ax.set(xlabel='Prediction Confidence', ylabel='Density',
           title='Prediction Confidence Distribution',
           xlim=[0, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_firing_rate_comparison(fig, results, unit_ids, labels):
    """Plot mean firing rate comparison between conditions."""
    ax = fig.add_subplot(3, 4, (3, 6))  # Span multiple subplots
    
    # Calculate mean firing rates per condition
    firing_rates = results['original_data']
    mean_right = firing_rates[labels == 0].mean(axis=0)
    mean_left = firing_rates[labels == 1].mean(axis=0)
    
    # Sort by difference
    diff = mean_left - mean_right
    sorted_idx = np.argsort(diff)
    
    # Plot top units with biggest differences
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
    """Save each subfigure as an individual file in an LDA folder."""
    base_path = Path(base_save_path)
    lda_folder = base_path.parent / "LDA"
    lda_folder.mkdir(parents=True, exist_ok=True)

    # Extract commonly used data
    transformed = results['transformed_data']
    labels = results['condition_labels']
    condition_names = trial_info['condition_names']
    colors = ['#FF6B6B', '#4ECDC4']  # Red for right, Teal for left

    print(f"\nSaving individual subfigures to: {lda_folder}")

    # 1. LDA Projection
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cond, name) in enumerate(condition_names.items()):
            mask = labels == cond
            y_jitter = np.random.normal(i, 0.1, np.sum(mask))
            ax.scatter(transformed[mask, 0], y_jitter, c=[colors[i]], label=name, alpha=0.6, s=40)
        ax.set_xlabel('LD1 (Discriminant Score)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(list(condition_names.values()))
        ax.set_title('LDA 1D Projection', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(lda_folder / "1_lda_projection.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 1_lda_projection.png")
    except Exception as e:
        print(f"  ✗ Failed: 1_lda_projection - {e}")

    # 2. LDA Histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cond, name) in enumerate(condition_names.items()):
            mask = labels == cond
            ax.hist(transformed[mask, 0], bins=30, alpha=0.6, color=colors[i], label=name, density=True)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Decision boundary')
        ax.set_xlabel('LD1 Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('LDA Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(lda_folder / "2_lda_histogram.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 2_lda_histogram.png")
    except Exception as e:
        print(f"  ✗ Failed: 2_lda_histogram - {e}")

    # 3. Confusion Matrix
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        conf_matrix = results['confusion_matrix']
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax)
        label_names = [condition_names[i] for i in range(len(condition_names))]
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=label_names, yticklabels=label_names,
               title='Confusion Matrix', ylabel='True', xlabel='Predicted')
        thresh = conf_matrix.max() / 2
        for i in range(2):
            for j in range(2):
                color = 'white' if conf_matrix[i, j] > thresh else 'black'
                ax.text(j, i, int(conf_matrix[i, j]), ha='center', va='center',
                       color=color, fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(lda_folder / "3_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 3_confusion_matrix.png")
    except Exception as e:
        print(f"  ✗ Failed: 3_confusion_matrix - {e}")

    # 4. CV Scores
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
        fig.savefig(lda_folder / "4_cv_scores.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 4_cv_scores.png")
    except Exception as e:
        print(f"  ✗ Failed: 4_cv_scores - {e}")

    # 5. Per-Condition Accuracy
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        accuracies = results['condition_accuracies']
        label_names = [condition_names[i] for i in range(len(condition_names))]
        bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
        ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', linewidth=2, label='Chance')
        ax.set(xlabel='Condition', ylabel='Accuracy', ylim=[0, 1], title='Per-Condition Accuracy',
               xticks=range(len(label_names)), xticklabels=label_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(lda_folder / "5_per_condition_accuracy.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 5_per_condition_accuracy.png")
    except Exception as e:
        print(f"  ✗ Failed: 5_per_condition_accuracy - {e}")

    # 6. Sensitivity vs Specificity
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        sensitivity = results['sensitivity']
        specificity = results['specificity']
        ax.plot(1 - specificity, sensitivity, 'ro', markersize=15,
               label=f'Model\n(Sens={sensitivity:.3f}, Spec={specificity:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance')
        ax.set(xlabel='False Positive Rate (1 - Specificity)', ylabel='True Positive Rate (Sensitivity)',
               xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Sensitivity vs Specificity', aspect='equal')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(lda_folder / "6_roc_style.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 6_roc_style.png")
    except Exception as e:
        print(f"  ✗ Failed: 6_roc_style - {e}")

    # 7. LDA Coefficients
    try:
        if hasattr(results['lda_full'], 'coef_'):
            fig, ax = plt.subplots(figsize=(10, 8))
            coef = results['lda_full'].coef_[0]
            sorted_idx = np.argsort(np.abs(coef))[::-1][:20]
            colors_bar = ['#4ECDC4' if c > 0 else '#FF6B6B' for c in coef[sorted_idx]]
            ax.barh(range(len(sorted_idx)), coef[sorted_idx], color=colors_bar, alpha=0.7)
            ax.set(yticks=range(len(sorted_idx)), yticklabels=[unit_ids[i] for i in sorted_idx],
                   xlabel='LDA Coefficient', title='Top Discriminative Units (Positive=Left, Negative=Right)')
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            fig.savefig(lda_folder / "7_lda_coefficients.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved: 7_lda_coefficients.png")
    except Exception as e:
        print(f"  ✗ Failed: 7_lda_coefficients - {e}")

    # 8. Trial Distribution
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
        fig.savefig(lda_folder / "8_trial_distribution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 8_trial_distribution.png")
    except Exception as e:
        print(f"  ✗ Failed: 8_trial_distribution - {e}")

    # 9. Summary Text
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        exp_params = trial_info.get('experiment_parameters', {})
        summary = f"""
    Classification Summary

    Overall Accuracy: {accuracy_score(labels, results['predictions']):.3f}
    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}

    Sensitivity (Left):  {results['sensitivity']:.3f}
    Specificity (Right): {results['specificity']:.3f}

    Experiment Info:
    • Total trials: {len(labels)}
    • Conditions: 2 (Left/Right)
    • Units: {len(unit_ids)}
    • LDA components: {results['n_components']}
    • Trial duration: {exp_params.get('trial_duration', 'N/A'):.3f}s
    • Left trials:  {trial_info['n_trials_per_condition'][1]}
    • Right trials: {trial_info['n_trials_per_condition'][0]}
        """
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        fig.savefig(lda_folder / "9_summary_text.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 9_summary_text.png")
    except Exception as e:
        print(f"  ✗ Failed: 9_summary_text - {e}")

    # 10. Feature Importance
    try:
        if hasattr(results['lda_full'], 'coef_'):
            fig, ax = plt.subplots(figsize=(8, 8))
            importance = np.abs(results['lda_full'].coef_[0])
            top_idx = np.argsort(importance)[::-1][:15]
            ax.barh(range(len(top_idx)), importance[top_idx], alpha=0.7, color='orange')
            ax.set(yticks=range(len(top_idx)), yticklabels=[unit_ids[i] for i in top_idx],
                   xlabel='|LDA Coefficient|', title='Top Discriminative Units')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            fig.savefig(lda_folder / "10_feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved: 10_feature_importance.png")
    except Exception as e:
        print(f"  ✗ Failed: 10_feature_importance - {e}")

    # 11. Prediction Confidence
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        confidence = np.max(results['prediction_proba'], axis=1)
        correct = results['predictions'] == labels
        ax.hist(confidence[correct], bins=20, alpha=0.7, label='Correct', color='green', density=True)
        ax.hist(confidence[~correct], bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
        ax.set(xlabel='Prediction Confidence', ylabel='Density', title='Prediction Confidence Distribution', xlim=[0, 1])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(lda_folder / "11_prediction_confidence.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 11_prediction_confidence.png")
    except Exception as e:
        print(f"  ✗ Failed: 11_prediction_confidence - {e}")

    # 12. Firing Rate Comparison
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
        fig.savefig(lda_folder / "12_firing_rate_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 12_firing_rate_comparison.png")
    except Exception as e:
        print(f"  ✗ Failed: 12_firing_rate_comparison - {e}")

    print(f"\nAll individual subfigures saved to: {lda_folder}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.0, None), save_plots=True, 
                output_path=None):
    """
    Complete analysis pipeline: load → analyze → visualize.
    
    Args:
        data_path: Path to neural data PKL file
        time_window: Tuple of (start, end) time in seconds relative to trial start.
                    If end is None, uses entire trial duration.
        save_plots: Whether to save the figure
        output_path: Custom output path (defaults to data_path with .lda_analysis.png suffix)
    
    Returns:
        Tuple of (lda_results, firing_rates, condition_labels, unit_ids, selectivity)
    """
    # Load and process data
    data = load_neural_data(data_path)
    firing_rates, condition_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )
    
    if len(condition_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")
    
    # Perform LDA analysis
    lda_results = perform_lda_analysis(firing_rates, condition_labels)
    
    # Calculate stimulus selectivity
    print("\nCalculating stimulus selectivity...")
    selectivity = calculate_stimulus_selectivity(
        unit_ids, condition_labels, firing_rates
    )
    
    # Create visualization
    if output_path is None and save_plots:
        output_path = Path(data_path).with_suffix('.lda_analysis.png')
    
    fig = create_analysis_figure(
        lda_results, unit_ids, trial_info,
        save_path=output_path if save_plots else None
    )
    
    return lda_results, firing_rates, condition_labels, unit_ids, selectivity


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configure your data path here
    DATA_PATH = r"/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/behavior_trial_embedding_20260309_2000.pkl"
    
    # Or find the most recent PKL file
    # from pathlib import Path
    # session_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\cl\sortout\CnL42SG\CnL42SG_20251217_151103")
    # pkl_files = list(session_folder.glob('behavior_trial_embedding_*.pkl'))
    # if pkl_files:
    #     DATA_PATH = sorted(pkl_files)[-1]
    #     print(f"Using most recent file: {DATA_PATH}")
    
    try:
        results = run_analysis(
            data_path=DATA_PATH,
            time_window=(0.0, 1.0),  # Use entire trial duration
            save_plots=True
        )
        
        lda_results, firing_rates, condition_labels, unit_ids, selectivity = results
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60)
        print(f"\nKey Results:")
        print(f"  Classification accuracy: {lda_results['cv_scores'].mean():.3f}")
        print(f"  Above chance: {lda_results['cv_scores'].mean() - 0.5:+.3f}")
        print(f"  Most selective unit: {selectivity['unit_ids'][np.argmax(np.abs(selectivity['selectivity_index']))]}")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise