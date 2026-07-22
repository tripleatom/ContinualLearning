"""
Unified Classifier Analysis for Behavior Trial Neural Data

Runs LDA / Logistic Regression / linear SVM on per-trial firing rates and
classifies trials by white stimulus position (left vs right).

Select the classifier with `method` in {'lda', 'logreg', 'svm'}.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler

from server_fallback import resolve_output_folder

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# METHOD CONFIGURATION
# =============================================================================

METHOD_CONFIG = {
    'lda': {
        'display_name': 'LDA',
        'score_xlabel': 'LD1 (Discriminant Score)',
        'score_title_projection': 'LDA 1D Projection',
        'score_title_histogram': 'LDA Score Distribution',
        'weight_xlabel': 'LDA Coefficient',
        'weight_abs_xlabel': '|LDA Coefficient|',
        'summary_header': 'Classification Summary (LDA)',
        'model_detail_label': 'LDA components',
        'file_prefix': 'lda',
        'output_suffix': '.lda_analysis.png',
    },
    'logreg': {
        'display_name': 'LogReg',
        'score_xlabel': 'Decision Score (Log-Odds)',
        'score_title_projection': 'LogReg Decision Score Projection',
        'score_title_histogram': 'LogReg Decision Score Distribution',
        'weight_xlabel': 'LogReg Weight',
        'weight_abs_xlabel': '|LogReg Weight|',
        'summary_header': 'Classification Summary (LogReg)',
        'model_detail_label': 'Solver',
        'file_prefix': 'logreg',
        'output_suffix': '.logreg_analysis.png',
    },
    'svm': {
        'display_name': 'SVM',
        'score_xlabel': 'Decision Score',
        'score_title_projection': 'SVM Decision Score Projection',
        'score_title_histogram': 'SVM Decision Score Distribution',
        'weight_xlabel': 'SVM Weight',
        'weight_abs_xlabel': '|SVM Weight|',
        'summary_header': 'Classification Summary (SVM)',
        'model_detail_label': 'Kernel',
        'file_prefix': 'svm',
        'output_suffix': '.svm_analysis.png',
    },
}


def _build_estimator(method, C):
    if method == 'lda':
        return LinearDiscriminantAnalysis()
    if method == 'logreg':
        return LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
    if method == 'svm':
        return SVC(kernel='linear', C=C)
    raise ValueError(f"Unknown method '{method}'. Use 'lda', 'logreg', or 'svm'.")


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
    """Calculate firing rates for each unit in each trial."""
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
            1: int(np.sum(condition_labels == 1)),
        },
        'window_duration': window_duration,
    }
    return firing_rates_clean, condition_labels, unit_ids, trial_info


def perform_classification_analysis(firing_rates, condition_labels, method='lda', C=1.0):
    """Run cross-validated linear classification with the chosen method."""
    cfg = METHOD_CONFIG[method]
    unique_conditions = np.unique(condition_labels)
    n_conditions = len(unique_conditions)
    n_features = firing_rates.shape[1]

    print(f"\n{cfg['display_name']} Analysis (C={C}):")
    print(f"  Conditions: {n_conditions} (0=Right, 1=Left)")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(condition_labels)}")

    min_trials = min(np.sum(condition_labels == cond) for cond in unique_conditions)
    print(f"  Min trials per class: {int(min_trials)}")

    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)

    cv_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    clf_cv = _build_estimator(method, C)
    cv_scores = cross_val_score(clf_cv, firing_rates_scaled, condition_labels,
                                cv=cv, scoring='accuracy')
    cv_results = cross_validate(clf_cv, firing_rates_scaled, condition_labels,
                                cv=cv, scoring=['accuracy', 'f1_macro'],
                                return_train_score=True, return_estimator=True)

    # Fit full model for weights and decision scores (visualization only).
    model = _build_estimator(method, C)
    model.fit(firing_rates_scaled, condition_labels)

    if method == 'lda':
        decision_scores = model.transform(firing_rates_scaled)
        n_components = min(n_conditions - 1, n_features)
    else:
        decision_scores = model.decision_function(firing_rates_scaled).reshape(-1, 1)
        n_components = 1

    chance_accuracy = 1.0 / n_conditions

    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Chance level: {chance_accuracy:.3f}")
    print(f"  Above chance: {cv_scores.mean() - chance_accuracy:+.3f}")

    return {
        'method': method,
        'model': model,
        'scaler': scaler,
        'transformed_data': decision_scores,
        'original_data': firing_rates_scaled,
        'condition_labels': condition_labels,
        'cv_scores': cv_scores,
        'cv_results': cv_results,
        'unique_conditions': unique_conditions,
        'n_components': n_components,
        'chance_accuracy': chance_accuracy,
        'C': C,
    }


def calculate_stimulus_selectivity(unit_ids, condition_labels, firing_rates):
    """Selectivity = (R_left - R_right) / (R_left + R_right)."""
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
        'rate_right': rate_right,
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

COLORS = ['#FF6B6B', '#4ECDC4']  # Right, Left


def _model_detail_value(results):
    method = results['method']
    if method == 'lda':
        return str(results['n_components'])
    if method == 'logreg':
        return 'lbfgs (L2 penalty)'
    if method == 'svm':
        return 'linear'
    return ''


def _summary_text(results, labels, unit_ids, trial_info):
    cfg = METHOD_CONFIG[results['method']]
    exp_params = trial_info.get('experiment_parameters', {})
    td = exp_params.get('trial_duration', None)
    td_str = f"{td:.3f}s" if isinstance(td, (int, float)) else "N/A"

    return f"""
    {cfg['summary_header']}

    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}
    Above Chance: {results['cv_scores'].mean() - results['chance_accuracy']:+.3f}

    Experiment Info:
    • Total trials: {len(labels)}
    • Conditions: 2 (Left/Right)
    • Units: {len(unit_ids)}
    • {cfg['model_detail_label']}: {_model_detail_value(results)}
    • C (regularization): {results['C']:.1f}
    • Trial duration: {td_str}
    • Left trials:  {trial_info['n_trials_per_condition'][1]}
    • Right trials: {trial_info['n_trials_per_condition'][0]}
    """


def _draw_projection(ax, data, labels, condition_names, cfg):
    for i, (cond, name) in enumerate(condition_names.items()):
        mask = labels == cond
        y_jitter = np.random.normal(i, 0.1, np.sum(mask))
        ax.scatter(data[mask, 0], y_jitter, c=[COLORS[i]], label=name, alpha=0.6, s=40)
    ax.set_xlabel(cfg['score_xlabel'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(list(condition_names.values()))
    ax.set_title(cfg['score_title_projection'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def _draw_histogram(ax, data, labels, condition_names, cfg):
    for i, (cond, name) in enumerate(condition_names.items()):
        mask = labels == cond
        ax.hist(data[mask, 0], bins=30, alpha=0.6, color=COLORS[i],
                label=name, density=True)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2,
               label='Decision boundary')
    ax.set_xlabel(cfg['score_xlabel'].replace(' (Log-Odds)', '').replace(' (Discriminant Score)', ' Score'),
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(cfg['score_title_histogram'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')


def _draw_cv_scores(ax, results):
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


def _draw_weights(ax, results, unit_ids, cfg):
    if not hasattr(results['model'], 'coef_'):
        return
    coef = results['model'].coef_[0]
    sorted_idx = np.argsort(np.abs(coef))[::-1][:20]
    colors_bar = ['#4ECDC4' if c > 0 else '#FF6B6B' for c in coef[sorted_idx]]
    ax.barh(range(len(sorted_idx)), coef[sorted_idx], color=colors_bar, alpha=0.7)
    ax.set(yticks=range(len(sorted_idx)),
           yticklabels=[unit_ids[i] for i in sorted_idx],
           xlabel=cfg['weight_xlabel'],
           title='Top Discriminative Units (Positive=Left, Negative=Right)')
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')


def _draw_trial_distribution(ax, trial_info):
    condition_names = trial_info['condition_names']
    counts = [trial_info['n_trials_per_condition'][i] for i in range(2)]
    labels = [condition_names[i] for i in range(2)]
    bars = ax.bar(range(len(counts)), counts, color=COLORS, alpha=0.7)
    ax.set(xlabel='Condition', ylabel='Number of Trials',
           title='Trial Distribution',
           xticks=range(len(labels)), xticklabels=labels)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, int(h),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _draw_feature_importance(ax, results, unit_ids, cfg):
    if not hasattr(results['model'], 'coef_'):
        return
    importance = np.abs(results['model'].coef_[0])
    top_idx = np.argsort(importance)[::-1][:15]
    ax.barh(range(len(top_idx)), importance[top_idx], alpha=0.7, color='orange')
    ax.set(yticks=range(len(top_idx)),
           yticklabels=[unit_ids[i] for i in top_idx],
           xlabel=cfg['weight_abs_xlabel'],
           title='Top Discriminative Units')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')


def _draw_firing_rate_comparison(ax, results, unit_ids, labels):
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


# =============================================================================
# VISUALIZATION (combined + individual)
# =============================================================================

def create_analysis_figure(results, unit_ids, trial_info, save_path=None):
    """Create combined 3x3 figure and save individual subfigures."""
    cfg = METHOD_CONFIG[results['method']]
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 12))

    transformed = results['transformed_data']
    labels = results['condition_labels']
    condition_names = trial_info['condition_names']

    _draw_projection(fig.add_subplot(3, 3, 1), transformed, labels, condition_names, cfg)
    _draw_histogram(fig.add_subplot(3, 3, 2), transformed, labels, condition_names, cfg)
    _draw_cv_scores(fig.add_subplot(3, 3, 3), results)
    _draw_weights(fig.add_subplot(3, 3, (4, 5)), results, unit_ids, cfg)
    _draw_trial_distribution(fig.add_subplot(3, 3, 6), trial_info)

    ax_summary = fig.add_subplot(3, 3, 7)
    ax_summary.axis('off')
    ax_summary.text(0.05, 0.95, _summary_text(results, labels, unit_ids, trial_info),
                    transform=ax_summary.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    _draw_feature_importance(fig.add_subplot(3, 3, 8), results, unit_ids, cfg)
    _draw_firing_rate_comparison(fig.add_subplot(3, 3, 9), results, unit_ids, labels)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        out_dir = resolve_output_folder(save_path.parent)
        save_path = out_dir / save_path.name
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved combined figure to: {save_path}")
        _save_individual_subfigures(results, unit_ids, trial_info, save_path)

    return fig


def _save_individual_subfigures(results, unit_ids, trial_info, base_save_path):
    """Save each panel as a standalone PNG."""
    cfg = METHOD_CONFIG[results['method']]
    prefix = cfg['file_prefix']
    out_folder = resolve_output_folder(Path(base_save_path).parent)

    transformed = results['transformed_data']
    labels = results['condition_labels']
    condition_names = trial_info['condition_names']

    print(f"\nSaving individual subfigures to: {out_folder}")

    panels = [
        (f"{prefix}_1_decision_projection.png", (8, 6),
         lambda ax: _draw_projection(ax, transformed, labels, condition_names, cfg)),
        (f"{prefix}_2_decision_histogram.png", (8, 6),
         lambda ax: _draw_histogram(ax, transformed, labels, condition_names, cfg)),
        (f"{prefix}_3_cv_scores.png", (8, 6),
         lambda ax: _draw_cv_scores(ax, results)),
        (f"{prefix}_4_weights.png", (10, 8),
         lambda ax: _draw_weights(ax, results, unit_ids, cfg)),
        (f"{prefix}_5_trial_distribution.png", (6, 6),
         lambda ax: _draw_trial_distribution(ax, trial_info)),
        (f"{prefix}_7_feature_importance.png", (8, 8),
         lambda ax: _draw_feature_importance(ax, results, unit_ids, cfg)),
        (f"{prefix}_8_firing_rate_comparison.png", (12, 6),
         lambda ax: _draw_firing_rate_comparison(ax, results, unit_ids, labels)),
    ]

    for name, figsize, draw_fn in panels:
        try:
            fig, ax = plt.subplots(figsize=figsize)
            draw_fn(ax)
            plt.tight_layout()
            fig.savefig(out_folder / name, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved: {name}")
        except Exception as e:
            print(f"  ✗ Failed: {name} - {e}")

    # Summary text panel
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        ax.text(0.05, 0.95, _summary_text(results, labels, unit_ids, trial_info),
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        name = f"{prefix}_6_summary_text.png"
        fig.savefig(out_folder / name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {name}")
    except Exception as e:
        print(f"  ✗ Failed: {prefix}_6_summary_text - {e}")

    print(f"\nAll individual subfigures saved to: {out_folder}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, method='lda', time_window=(0.0, None), C=1.0,
                 save_plots=True, output_path=None,
                 _preloaded=None):
    """
    Complete analysis pipeline: load → analyze → visualize.

    Args:
        data_path: Path to neural data PKL file.
        method: One of 'lda', 'logreg', 'svm'.
        time_window: (start, end) seconds relative to trial start.
        C: Regularization parameter (ignored for LDA).
        save_plots: Whether to save figures.
        output_path: Custom output path. Defaults to
                     <session>/behavior_analysis/<stem><method_suffix>.png
        _preloaded: Optional precomputed (firing_rates, condition_labels, unit_ids,
                    trial_info) tuple to avoid reloading when running multiple methods.

    Returns:
        (results, firing_rates, condition_labels, unit_ids, selectivity)
    """
    if method not in METHOD_CONFIG:
        raise ValueError(f"method must be one of {list(METHOD_CONFIG)}; got {method!r}")

    if _preloaded is None:
        data = load_neural_data(data_path)
        firing_rates, condition_labels, unit_ids, trial_info = calculate_firing_rates(
            data, time_window=time_window
        )
    else:
        firing_rates, condition_labels, unit_ids, trial_info = _preloaded

    if len(condition_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")

    results = perform_classification_analysis(
        firing_rates, condition_labels, method=method, C=C
    )

    print("\nCalculating stimulus selectivity...")
    selectivity = calculate_stimulus_selectivity(unit_ids, condition_labels, firing_rates)

    if output_path is None and save_plots:
        session_dir = Path(data_path).parent
        suffix = METHOD_CONFIG[method]['output_suffix']
        output_path = session_dir / "behavior_analysis" / (Path(data_path).stem + suffix)

    create_analysis_figure(
        results, unit_ids, trial_info,
        save_path=output_path if save_plots else None,
    )

    return results, firing_rates, condition_labels, unit_ids, selectivity


def run_all_methods(data_path, methods=('lda', 'logreg', 'svm'),
                    time_window=(0.0, None), C=1.0, save_plots=True):
    """Run multiple classifiers on the same data without reloading it."""
    data = load_neural_data(data_path)
    preloaded = calculate_firing_rates(data, time_window=time_window)

    outputs = {}
    for m in methods:
        print("\n" + "=" * 60)
        print(f"Running {METHOD_CONFIG[m]['display_name']}")
        print("=" * 60)
        outputs[m] = run_analysis(
            data_path=data_path, method=m, time_window=time_window, C=C,
            save_plots=save_plots, _preloaded=preloaded,
        )
    return outputs


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313\behavior_trial_embedding_20260423_1631.pkl"

    try:
        all_results = run_all_methods(
            data_path=DATA_PATH,
            methods=('lda', 'logreg', 'svm'),
            time_window=(0.0, 1.0),
            C=1.0,
            save_plots=True,
        )

        print("\n" + "=" * 60)
        print("All analyses complete!")
        print("=" * 60)
        for method, res in all_results.items():
            results, _, _, _, selectivity = res
            print(f"\n{METHOD_CONFIG[method]['display_name']}:")
            print(f"  Classification accuracy: {results['cv_scores'].mean():.3f}")
            print(f"  Above chance: {results['cv_scores'].mean() - 0.5:+.3f}")
            top = selectivity['unit_ids'][np.argmax(np.abs(selectivity['selectivity_index']))]
            print(f"  Most selective unit: {top}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
