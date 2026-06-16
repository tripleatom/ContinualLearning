"""
SVM Decoding for Grating Orientation Neural Data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

from grating_utils import (
    load_neural_data,
    calculate_firing_rates,
    calculate_orientation_selectivity,
    plot_cv_scores,
    plot_trial_distribution,
    plot_summary_text,
)


# =============================================================================
# ANALYSIS
# =============================================================================

def perform_svm_analysis(firing_rates, orientation_labels, kernel='rbf', C=1.0, gamma='scale'):
    """
    Perform SVM decoding with cross-validation.

    Args:
        firing_rates: (n_trials, n_units) array of firing rates
        orientation_labels: (n_trials,) array of class labels
        kernel: SVM kernel ('rbf', 'linear', 'poly')
        C: Regularization parameter
        gamma: Kernel coefficient ('scale', 'auto', or float); used by rbf/poly/sigmoid

    Returns:
        Dictionary with SVM results including cross-validated scores,
        shuffled-label baseline, PCA projection, and optional linear weights.
    """
    unique_orientations = np.array([str(x) for x in np.unique(orientation_labels)])
    orientation_labels = np.array([str(x) for x in orientation_labels])

    n_orientations = len(unique_orientations)
    n_features = firing_rates.shape[1]

    print(f"\nSVM Analysis:")
    print(f"  Kernel: {kernel} | C: {C} | gamma: {gamma}")
    print(f"  Orientations: {n_orientations} ({unique_orientations}°)")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(orientation_labels)}")

    if n_features == 0:
        raise ValueError("firing_rates has 0 units (features). Check that the .pkl "
                         "was extracted with the correct passive_start/passive_end "
                         "and curated analyzer units.")

    min_trials = min(np.sum(orientation_labels == ori) for ori in unique_orientations)
    print(f"  Min trials per class: {int(min_trials)}")

    # Balance classes by subsampling each to min_trials
    rng = np.random.default_rng(42)
    balanced_idx = np.concatenate([
        rng.choice(np.where(orientation_labels == ori)[0], size=int(min_trials), replace=False)
        for ori in unique_orientations
    ])
    balanced_idx = np.sort(balanced_idx)
    firing_rates = firing_rates[balanced_idx]
    orientation_labels = orientation_labels[balanced_idx]
    print(f"  Balanced to {int(min_trials)} trials/class -> {len(orientation_labels)} total")

    # Cross-validation with scaling inside each fold to avoid leakage.
    cv_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    svm_cv = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))
    cv_scores = cross_val_score(svm_cv, firing_rates, orientation_labels,
                                cv=cv, scoring='accuracy')
    shuffled_labels = rng.permutation(orientation_labels)
    cv_scores_shuffled = cross_val_score(svm_cv, firing_rates, shuffled_labels,
                                         cv=cv, scoring='accuracy')

    # Full-data model is used only for visualization of PCA and linear weights.
    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)
    svm_full = SVC(kernel=kernel, C=C, gamma=gamma)
    svm_full.fit(firing_rates_scaled, orientation_labels)
    chance_accuracy = 1.0 / n_orientations

    print(f"  CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Shuffled-label CV: {cv_scores_shuffled.mean():.3f} +/- {cv_scores_shuffled.std():.3f}")
    print(f"  Chance level: {chance_accuracy:.3f}")

    # PCA projection for visualization (2D and 3D)
    n_pca = min(3, n_features, len(orientation_labels) - 1)
    pca = PCA(n_components=n_pca)
    pca_projection = pca.fit_transform(firing_rates_scaled)

    # Linear SVM weights (only available for linear kernel)
    linear_weights = getattr(svm_full, 'coef_', None)

    print(f"  PCA explained variance: {pca.explained_variance_ratio_[:n_pca].sum():.3f}")

    return {
        'svm_model': svm_full,
        'scaler': scaler,
        'pca': pca,
        'pca_projection': pca_projection,
        'original_data': firing_rates_scaled,
        'orientation_labels': orientation_labels,
        'cv_scores': cv_scores,
        'cv_scores_shuffled': cv_scores_shuffled,
        'unique_orientations': unique_orientations,
        'chance_accuracy': chance_accuracy,
        'linear_weights': linear_weights,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'kernel': kernel,
        'C': C,
        'gamma': gamma,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_figure(results, unit_ids, trial_info, save_path=None,
                           label_suffix='°', is_orientation=True):
    """Create SVM analysis visualization matching the LDA result layout."""
    plt.style.use('default')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(26, 14))

    labels = results['orientation_labels']
    unique_ori = results['unique_orientations']
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_ori) + 1)[:-1])

    _plot_pca_3d(fig, results, labels, unique_ori, colors, label_suffix)
    _plot_pca_2d(fig, results, labels, unique_ori, colors, label_suffix)
    plot_cv_scores(fig, results, subplot_pos=(2, 4, 3))
    plot_trial_distribution(fig, trial_info, unique_ori, colors, label_suffix,
                            subplot_pos=(2, 4, 4))
    _plot_svm_weights(fig, results, unit_ids, unique_ori, label_suffix)
    _plot_linear_feature_importance(fig, results, unit_ids)

    model_params = {
        'Kernel': results['kernel'],
        'C': results['C'],
        'gamma': results['gamma'],
    }
    plot_summary_text(fig, results, labels, unit_ids, trial_info, label_suffix,
                      model_params=model_params, subplot_pos=(2, 4, 8))

    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color=colors[i],
                   markerfacecolor=colors[i], markeredgecolor='none',
                   markersize=10, label=f'{ori}{label_suffix}')
        for i, ori in enumerate(unique_ori)
    ]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 8),
               frameon=False, fontsize=16, bbox_to_anchor=(0.5, 0.99))
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_svg = save_path.suffix.lower() == '.svg'
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    transparent=is_svg,
                    facecolor='none' if is_svg else 'white')
        print(f"Saved figure to: {save_path}")

    return fig


def _plot_pca_3d(fig, results, labels, orientations, colors, label_suffix='°'):
    """Plot 3D PCA projection colored by class (SVM has no built-in projection)."""
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    pca_proj = results['pca_projection']
    pca_var = results['pca_explained_variance']

    if pca_proj.shape[1] >= 3:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(pca_proj[mask, 0], pca_proj[mask, 1], pca_proj[mask, 2],
                       c=[colors[i]], label=f'{ori}{label_suffix}',
                       alpha=0.85, s=70, edgecolors='none')
        ax.set_xlabel(f'PC1 ({pca_var[0]:.1%})', fontsize=20,
                      fontweight='bold', labelpad=10)
        ax.set_ylabel(f'PC2 ({pca_var[1]:.1%})', fontsize=20,
                      fontweight='bold', labelpad=10)
        ax.set_zlabel(f'PC3 ({pca_var[2]:.1%})', fontsize=20,
                      fontweight='bold', labelpad=10)
    else:
        ax.text(0.5, 0.5, 0.5, 'Need ≥3 PCs\nfor 3D visualization',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)

    ax.set_title('SVM PCA 3D Projection', fontsize=24, fontweight='bold', pad=14)
    ax.tick_params(axis='both', labelsize=14)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor('lightgray')
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))


def _plot_pca_2d(fig, results, labels, orientations, colors, label_suffix='°'):
    """Plot 2D PCA projection colored by class."""
    ax = fig.add_subplot(2, 4, 2)
    pca_proj = results['pca_projection']
    pca_var = results['pca_explained_variance']

    if pca_proj.shape[1] >= 2:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(pca_proj[mask, 0], pca_proj[mask, 1],
                       c=[colors[i]], label=f'{ori}{label_suffix}',
                       alpha=0.85, s=70, edgecolors='none')
        ax.set_xlabel(f'PC1 ({pca_var[0]:.1%})', fontsize=22, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca_var[1]:.1%})', fontsize=22, fontweight='bold')
        ax.set_title('SVM PCA 2D Projection', fontsize=24, fontweight='bold', pad=12)
    else:
        y_jitter = np.random.normal(0, 0.1, len(labels))
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(pca_proj[mask, 0], y_jitter[mask],
                       c=[colors[i]], label=f'{ori}{label_suffix}',
                       alpha=0.85, s=70, edgecolors='none')
        ax.set_xlabel(f'PC1 ({pca_var[0]:.1%})', fontsize=22, fontweight='bold')
        ax.set_ylabel('Random jitter', fontsize=22, fontweight='bold')
        ax.set_title('SVM PCA 1D Projection', fontsize=24, fontweight='bold', pad=12)

    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(labelsize=18, width=2.0, length=7)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def _plot_svm_weights(fig, results, unit_ids, orientations, label_suffix='°'):
    """Plot SVM weight heatmap (only for linear kernel; shows a note otherwise)."""
    ax = fig.add_subplot(2, 4, (5, 6))

    weights = results['linear_weights']
    if weights is not None:
        im = ax.imshow(weights, cmap='RdBu_r', aspect='auto')
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=16, width=1.8, length=6)

        ax.set_xlabel('Units', fontsize=22, fontweight='bold')
        ax.set_ylabel('Class', fontsize=22, fontweight='bold')
        ax.set_title('SVM Weights (linear kernel)', fontsize=24,
                     fontweight='bold', pad=12)
        ax.set_yticks(range(len(orientations)))
        ax.set_yticklabels([f'{ori}{label_suffix}' for ori in orientations],
                           fontsize=18)

        if len(unit_ids) <= 20:
            ax.set_xticks(range(len(unit_ids)))
            ax.set_xticklabels([uid.split('_')[-1] for uid in unit_ids],
                               rotation=45, ha='right', fontsize=16)
        else:
            ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='both', width=2.0, length=7)
    else:
        ax.text(0.5, 0.5,
                f'SVM weights not available\nfor kernel="{results["kernel"]}".\n'
                'Use kernel="linear" to show weights.',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('SVM Weights', fontsize=24, fontweight='bold', pad=12)
        ax.axis('off')


def _plot_linear_feature_importance(fig, results, unit_ids):
    """Plot feature importance from mean absolute linear SVM weights."""
    ax = fig.add_subplot(2, 4, 7)

    weights = results['linear_weights']
    if weights is None:
        ax.text(0.5, 0.5,
                f'Feature weights not available\nfor kernel="{results["kernel"]}".\n'
                'Use kernel="linear" to show top units.',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Top Discriminative Units', fontsize=24,
                     fontweight='bold', pad=12)
        ax.axis('off')
        return ax

    importance = np.mean(np.abs(weights), axis=0)
    top_idx = np.argsort(importance)[::-1][:15]

    ax.barh(range(len(top_idx)), importance[top_idx],
            color='#E69F00', edgecolor='black', linewidth=1.0)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([unit_ids[i].split('_')[-1] for i in top_idx],
                       fontsize=16)
    ax.set_xlabel('Mean |Weight|', fontsize=22, fontweight='bold')
    ax.set_title('Top Discriminative Units', fontsize=24,
                 fontweight='bold', pad=12)
    ax.invert_yaxis()
    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(axis='x', labelsize=18, width=2.0, length=7)
    ax.tick_params(axis='y', width=2.0, length=7)
    return ax


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.07, 0.16), save_plots=True,
                 output_path=None, kernel='rbf', C=1.0, gamma='scale'):
    """
    Complete SVM decoding pipeline: load -> analyze -> visualize.

    If the data contains multiple spatial frequencies, each SF is analyzed
    and plotted separately.

    Args:
        data_path: Path to neural data file
        time_window: Tuple of (start, end) time in seconds
        save_plots: Whether to save figures
        output_path: Custom output path prefix
        kernel: SVM kernel type ('rbf', 'linear', 'poly')
        C: SVM regularization parameter
        gamma: Kernel coefficient

    Returns:
        List of (svm_results, firing_rates, orientation_labels, unit_ids) per SF
    """
    data = load_neural_data(data_path)
    firing_rates, orientation_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )

    if len(orientation_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")

    unique_sfs = trial_info['unique_spatial_freqs']
    sf_labels = trial_info['spatial_freq_labels']
    all_results = []

    # ── Analysis 1: same SF, decode orientation ────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# ORIENTATION DECODING PER SF ({len(unique_sfs)} SF(s))")
    print(f"{'#'*60}")

    for sf in unique_sfs:
        if sf is None:
            fr_sf, labels_sf, sf_tag, sf_display = firing_rates, orientation_labels, '', 'all SF'
        else:
            sf_mask = sf_labels == sf
            fr_sf = firing_rates[sf_mask]
            labels_sf = orientation_labels[sf_mask]
            sf_tag = f'_sf{sf}'
            sf_display = f'SF={sf} cpd'

        print(f"\n{'='*60}")
        print(f"Analyzing {sf_display}  ({len(labels_sf)} trials)")
        print(f"{'='*60}")

        if len(labels_sf) == 0:
            print(f"  No trials for {sf_display}, skipping.")
            continue

        unique_ori_sf = sorted(set(labels_sf.tolist()))
        if len(unique_ori_sf) < 2:
            print(f"  Only 1 orientation present for {sf_display} - skipping orientation decoding.")
            continue

        trial_info_sf = {
            'unique_orientations': unique_ori_sf,
            'experiment_parameters': trial_info['experiment_parameters'],
            'n_trials_per_orientation': {
                str(ori): int(np.sum(labels_sf == ori))
                for ori in unique_ori_sf
            },
        }

        svm_results = perform_svm_analysis(fr_sf, labels_sf,
                                           kernel=kernel, C=C, gamma=gamma)

        fig_path = None
        if save_plots:
            base = Path(output_path) if output_path else Path(data_path).with_suffix('')
            fig_path = Path(str(base) + sf_tag + '.svm_analysis.png')

        fig = create_analysis_figure(svm_results, unit_ids, trial_info_sf,
                                     save_path=fig_path)
        fig.suptitle(f"SVM Decoding - {sf_display} (kernel={kernel})",
                     fontsize=14, y=1.01)

        print(f"\nCalculating orientation selectivity for {sf_display}...")
        calculate_orientation_selectivity(unit_ids, labels_sf, fr_sf)

        all_results.append((svm_results, fr_sf, labels_sf, unit_ids))

    # ── Analysis 2: same orientation, decode SF ────────────────────────────────
    if sf_labels is not None and len(unique_sfs) > 1:
        unique_oris = sorted(set(orientation_labels.tolist()))
        print(f"\n{'#'*60}")
        print(f"# SF DECODING PER ORIENTATION ({len(unique_oris)} orientations)")
        print(f"{'#'*60}")

        for ori in unique_oris:
            ori_mask = orientation_labels == ori
            fr_ori = firing_rates[ori_mask]
            sf_ori = sf_labels[ori_mask]
            unique_sf_ori = sorted(set(sf_ori.tolist()))

            if len(unique_sf_ori) < 2:
                print(f"  Orientation {ori}°: only one SF present, skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"SF decoding - orientation={ori}°  ({len(sf_ori)} trials, SFs={unique_sf_ori})")
            print(f"{'='*60}")

            trial_info_ori = {
                'unique_orientations': unique_sf_ori,
                'experiment_parameters': trial_info['experiment_parameters'],
                'n_trials_per_orientation': {
                    str(sf): int(np.sum(sf_ori == sf))
                    for sf in unique_sf_ori
                },
            }

            svm_sf = perform_svm_analysis(fr_ori, sf_ori,
                                          kernel=kernel, C=C, gamma=gamma)

            fig_path = None
            if save_plots:
                base = Path(output_path) if output_path else Path(data_path).with_suffix('')
                fig_path = Path(str(base) + f'_ori{ori}.sf_svm_decoding.png')

            fig = create_analysis_figure(svm_sf, unit_ids, trial_info_ori,
                                         save_path=fig_path,
                                         label_suffix=' cpd',
                                         is_orientation=False)
            fig.suptitle(f"SF SVM Decoding - Orientation={ori}° (kernel={kernel})",
                         fontsize=14, y=1.01)

            all_results.append((svm_sf, fr_ori, sf_ori, unit_ids))

    return all_results


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = input("Enter path to neural data (.pkl file): ").strip().strip('"').strip("'")

    try:
        all_results = run_analysis(
            data_path=DATA_PATH,
            time_window=(0.05, 1.5),
            save_plots=True,
            kernel='rbf',
            C=1.0,
            gamma='scale',
        )
        print(f"\nAnalysis complete! ({len(all_results)} spatial frequency group(s))")

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
