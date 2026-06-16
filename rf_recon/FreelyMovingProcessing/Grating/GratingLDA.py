"""
LDA Analysis for Grating Orientation Neural Data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
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

def perform_lda_analysis(firing_rates, orientation_labels, n_components=None):
    """
    Perform LDA analysis with cross-validation.

    Returns:
        Dictionary with LDA results: transformed_data, cv_scores,
        cv_scores_shuffled (label-permuted baseline), and the fitted model
        (for projection/coefficient visualization only — never for metrics).
    """
    unique_orientations = np.array([str(x) for x in np.unique(orientation_labels)])
    orientation_labels = np.array([str(x) for x in orientation_labels])

    n_orientations = len(unique_orientations)
    n_features = firing_rates.shape[1]

    print(f"\nLDA Analysis:")
    print(f"  Orientations: {n_orientations} ({unique_orientations}°)")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(orientation_labels)}")

    min_trials = min(np.sum(orientation_labels == ori) for ori in unique_orientations)
    max_components = min(n_orientations - 1, n_features)
    n_components = min(n_components or 3, max_components)

    print(f"  Min trials per class: {int(min_trials)}")
    print(f"  LDA components: {n_components}")

    # Balance classes by subsampling each to min_trials
    rng = np.random.default_rng(42)
    balanced_idx = np.concatenate([
        rng.choice(np.where(orientation_labels == ori)[0], size=int(min_trials), replace=False)
        for ori in unique_orientations
    ])
    balanced_idx = np.sort(balanced_idx)
    firing_rates = firing_rates[balanced_idx]
    orientation_labels = orientation_labels[balanced_idx]
    print(f"  Balanced to {int(min_trials)} trials/class → {len(orientation_labels)} total")

    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)

    # Fit on all data — used only for projection/coefficient visualization,
    # NOT for any performance metric.
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_transformed = lda.fit_transform(firing_rates_scaled, orientation_labels)
    lda_full = LinearDiscriminantAnalysis()
    lda_full.fit(firing_rates_scaled, orientation_labels)

    cv_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_scores = cross_val_score(LinearDiscriminantAnalysis(),
                                firing_rates_scaled, orientation_labels,
                                cv=cv, scoring='accuracy')

    # Shuffled-label baseline (same CV folds, permuted labels)
    shuffled_labels = rng.permutation(orientation_labels)
    cv_scores_shuffled = cross_val_score(LinearDiscriminantAnalysis(),
                                         firing_rates_scaled, shuffled_labels,
                                         cv=cv, scoring='accuracy')

    chance_accuracy = 1.0 / n_orientations

    print(f"  CV accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Shuffled CV:    {cv_scores_shuffled.mean():.3f} ± {cv_scores_shuffled.std():.3f}")
    print(f"  Chance level:   {chance_accuracy:.3f}")

    return {
        'lda_model': lda,
        'lda_full': lda_full,
        'scaler': scaler,
        'transformed_data': lda_transformed,
        'original_data': firing_rates_scaled,
        'orientation_labels': orientation_labels,
        'cv_scores': cv_scores,
        'cv_scores_shuffled': cv_scores_shuffled,
        'unique_orientations': unique_orientations,
        'n_components': n_components,
        'chance_accuracy': chance_accuracy,
        'explained_variance_ratio': getattr(lda, 'explained_variance_ratio_', None)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_figure(results, unit_ids, trial_info, save_path=None,
                           label_suffix='°'):
    """Create LDA analysis visualization (leakage-free)."""
    plt.style.use('default')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(26, 14))

    transformed = results['transformed_data']
    labels = results['orientation_labels']
    unique_ori = results['unique_orientations']
    n_comp = results['n_components']
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_ori) + 1)[:-1])

    ax3d = _plot_3d_scatter(fig, transformed, labels, unique_ori, colors, n_comp, label_suffix)
    ax2d = _plot_2d_scatter(fig, transformed, labels, unique_ori, colors, n_comp, label_suffix)
    plot_cv_scores(fig, results, subplot_pos=(2, 4, 3))
    ax_td = plot_trial_distribution(fig, trial_info, unique_ori, colors, label_suffix,
                                    subplot_pos=(2, 4, 4))
    _plot_lda_coefficients(fig, results, unit_ids, unique_ori, label_suffix)
    _plot_feature_importance(fig, results, unit_ids)
    plot_summary_text(fig, results, labels, unit_ids, trial_info, label_suffix,
                      model_params={'LDA components': results['n_components']},
                      subplot_pos=(2, 4, 8))

    # Strip grids from shared util plots for consistent look
    if ax_td is not None:
        ax_td.grid(False)

    # Single shared legend for 3D + 2D scatters, placed between the two panels
    handles, leg_labels = ax2d.get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels,
                   loc='center', bbox_to_anchor=(0.30, 0.97),
                   ncol=min(len(handles), 8),
                   frameon=False, fontsize=16,
                   handletextpad=0.5, columnspacing=1.4)

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


def _plot_3d_scatter(fig, data, labels, orientations, colors, n_comp, label_suffix='°'):
    """Plot 3D LDA scatter."""
    ax = fig.add_subplot(2, 4, 1, projection='3d')

    if n_comp >= 3:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                       c=[colors[i]], label=f'{ori}{label_suffix}', alpha=0.85, s=70,
                       edgecolors='none')
        ax.set_xlabel('LD1', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_ylabel('LD2', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_zlabel('LD3', fontsize=20, fontweight='bold', labelpad=10)
    else:
        ax.text(0.5, 0.5, 0.5, 'Need ≥3 components\nfor 3D visualization',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)

    ax.set_title('LDA 3D Projection', fontsize=24, fontweight='bold', pad=14)
    ax.tick_params(axis='both', labelsize=14)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor('lightgray')
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    return ax


def _plot_2d_scatter(fig, data, labels, orientations, colors, n_comp, label_suffix='°'):
    """Plot 2D LDA scatter or 1D jitter."""
    ax = fig.add_subplot(2, 4, 2)

    if n_comp >= 2:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(data[mask, 0], data[mask, 1],
                       c=[colors[i]], label=f'{ori}{label_suffix}',
                       alpha=0.85, s=70, edgecolors='none')
        ax.set_xlabel('LD1', fontsize=22, fontweight='bold')
        ax.set_ylabel('LD2', fontsize=22, fontweight='bold')
        ax.set_title('LDA 2D Projection', fontsize=24, fontweight='bold', pad=12)
    else:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            y_jitter = np.random.normal(0, 0.1, np.sum(mask))
            ax.scatter(data[mask, 0], y_jitter,
                       c=[colors[i]], label=f'{ori}{label_suffix}',
                       alpha=0.85, s=70, edgecolors='none')
        ax.set_xlabel('LD1', fontsize=22, fontweight='bold')
        ax.set_ylabel('Random jitter', fontsize=22, fontweight='bold')
        ax.set_title('LDA 1D Projection', fontsize=24, fontweight='bold', pad=12)

    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(labelsize=18, width=2.0, length=7)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return ax


def _plot_lda_coefficients(fig, results, unit_ids, orientations, label_suffix='°'):
    """Plot LDA coefficient heatmap."""
    ax = fig.add_subplot(2, 4, (5, 6))

    if hasattr(results['lda_full'], 'coef_'):
        im = ax.imshow(results['lda_full'].coef_, cmap='RdBu_r', aspect='auto')
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=16, width=1.8, length=6)

        ax.set_xlabel('Units', fontsize=22, fontweight='bold')
        ax.set_ylabel('Discriminant', fontsize=22, fontweight='bold')
        ax.set_title('LDA Coefficients', fontsize=24, fontweight='bold', pad=12)
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


def _plot_feature_importance(fig, results, unit_ids):
    """Plot feature importance based on mean absolute LDA coefficients."""
    ax = fig.add_subplot(2, 4, 7)

    if hasattr(results['lda_full'], 'coef_'):
        importance = np.mean(np.abs(results['lda_full'].coef_), axis=0)
        top_idx = np.argsort(importance)[::-1][:15]

        ax.barh(range(len(top_idx)), importance[top_idx],
                color='#E69F00', edgecolor='black', linewidth=1.0)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([unit_ids[i].split('_')[-1] for i in top_idx],
                           fontsize=16)
        ax.set_xlabel('Mean |Coefficient|', fontsize=22, fontweight='bold')
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


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.07, 0.16), save_plots=True, output_path=None):
    """
    Complete LDA analysis pipeline: load → analyze → visualize.

    If the data contains multiple spatial frequencies, each SF is analyzed
    and plotted separately.

    Returns:
        List of (lda_results, firing_rates, orientation_labels, unit_ids) per SF
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
            print(f"  Only 1 orientation present for {sf_display} — skipping orientation decoding.")
            continue
        trial_info_sf = {
            'unique_orientations': unique_ori_sf,
            'experiment_parameters': trial_info['experiment_parameters'],
            'n_trials_per_orientation': {
                str(ori): int(np.sum(labels_sf == ori))
                for ori in unique_ori_sf
            },
        }

        lda_results = perform_lda_analysis(fr_sf, labels_sf)

        fig_path = None
        if save_plots:
            base = Path(output_path) if output_path else Path(data_path).with_suffix('')
            fig_path = Path(str(base) + sf_tag + '.lda_analysis.png')

        fig = create_analysis_figure(lda_results, unit_ids, trial_info_sf,
                                     save_path=fig_path)
        fig.suptitle(f"LDA Analysis — {sf_display}", fontsize=26, fontweight='bold', y=1.01)

        print(f"\nCalculating orientation selectivity for {sf_display}...")
        calculate_orientation_selectivity(unit_ids, labels_sf, fr_sf)

        all_results.append((lda_results, fr_sf, labels_sf, unit_ids))

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
            print(f"SF decoding — orientation={ori}°  ({len(sf_ori)} trials, SFs={unique_sf_ori})")
            print(f"{'='*60}")

            trial_info_ori = {
                'unique_orientations': unique_sf_ori,
                'experiment_parameters': trial_info['experiment_parameters'],
                'n_trials_per_orientation': {
                    str(sf): int(np.sum(sf_ori == sf))
                    for sf in unique_sf_ori
                },
            }

            lda_sf = perform_lda_analysis(fr_ori, sf_ori)

            fig_path = None
            if save_plots:
                base = Path(output_path) if output_path else Path(data_path).with_suffix('')
                fig_path = Path(str(base) + f'_ori{ori}.sf_decoding.png')

            fig = create_analysis_figure(lda_sf, unit_ids, trial_info_ori,
                                         save_path=fig_path,
                                         label_suffix=' cpd')
            fig.suptitle(f"SF Decoding — Orientation={ori}°", fontsize=26, fontweight='bold', y=1.01)

            all_results.append((lda_sf, fr_ori, sf_ori, unit_ids))

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
            save_plots=True
        )
        print(f"\nAnalysis complete! ({len(all_results)} spatial frequency group(s))")

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
