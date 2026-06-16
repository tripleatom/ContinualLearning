"""
Compare LDA decoding with raw firing rates vs. z-scored firing rates.

Note: GratingLDA.py already z-scores via StandardScaler before LDA, so
"z-scored" here matches that pipeline. "Raw" passes firing rates directly
to LDA with no per-feature centering/scaling.

For both conditions:
  - classes are balanced by subsampling to min trials/class
  - the same StratifiedKFold splits are used (seeded)
  - a shuffled-label baseline is computed with the same CV
  - z-scoring is fit inside each training fold (no leakage)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

from grating_utils import load_neural_data, calculate_firing_rates


# =============================================================================
# COMPARISON
# =============================================================================

def compare_raw_vs_zscore(firing_rates, labels, random_state=42):
    """
    Run LDA with raw and with z-scored features on the same balanced data
    and CV folds. Returns a dict with both score arrays and shuffled baselines.
    """
    labels = np.array([str(x) for x in labels])
    unique = np.unique(labels)
    n_classes = len(unique)
    min_trials = int(min(np.sum(labels == c) for c in unique))

    rng = np.random.default_rng(random_state)
    balanced_idx = np.sort(np.concatenate([
        rng.choice(np.where(labels == c)[0], size=min_trials, replace=False)
        for c in unique
    ]))
    X = firing_rates[balanced_idx]
    y = labels[balanced_idx]

    cv_folds = max(2, min(5, min_trials))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    raw_clf = LinearDiscriminantAnalysis()
    z_clf = Pipeline([('scaler', StandardScaler()),
                      ('lda', LinearDiscriminantAnalysis())])

    raw_scores = cross_val_score(raw_clf, X, y, cv=cv, scoring='accuracy')
    z_scores = cross_val_score(z_clf,  X, y, cv=cv, scoring='accuracy')

    y_shuf = rng.permutation(y)
    raw_shuf = cross_val_score(raw_clf, X, y_shuf, cv=cv, scoring='accuracy')
    z_shuf = cross_val_score(z_clf,  X, y_shuf, cv=cv, scoring='accuracy')

    chance = 1.0 / n_classes

    print(f"  Classes: {n_classes} | Min trials/class: {min_trials} | CV folds: {cv_folds}")
    print(f"  Raw     : {raw_scores.mean():.3f} ± {raw_scores.std():.3f}   "
          f"(shuffled {raw_shuf.mean():.3f} ± {raw_shuf.std():.3f})")
    print(f"  Z-score : {z_scores.mean():.3f} ± {z_scores.std():.3f}   "
          f"(shuffled {z_shuf.mean():.3f} ± {z_shuf.std():.3f})")
    print(f"  Chance  : {chance:.3f}")

    return {
        'raw': raw_scores, 'raw_shuffled': raw_shuf,
        'zscore': z_scores, 'zscore_shuffled': z_shuf,
        'chance': chance, 'n_classes': n_classes,
        'n_trials_total': len(y), 'cv_folds': cv_folds,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(results_by_group, title, save_path=None):
    """
    Bar chart: raw vs z-score CV accuracy across groups (e.g., per SF),
    with shuffled baselines and a chance line per group.
    """
    groups = list(results_by_group.keys())
    n_groups = len(groups)
    x = np.arange(n_groups)
    width = 0.35

    raw_mean = np.array([results_by_group[g]['raw'].mean() for g in groups])
    raw_std = np.array([results_by_group[g]['raw'].std() for g in groups])
    z_mean = np.array([results_by_group[g]['zscore'].mean() for g in groups])
    z_std = np.array([results_by_group[g]['zscore'].std() for g in groups])
    raw_shuf = np.array([results_by_group[g]['raw_shuffled'].mean() for g in groups])
    z_shuf = np.array([results_by_group[g]['zscore_shuffled'].mean() for g in groups])
    chance = np.array([results_by_group[g]['chance'] for g in groups])

    fig, ax = plt.subplots(figsize=(max(7, 2 + 2.0 * n_groups), 6))

    b1 = ax.bar(x - width / 2, raw_mean, width, yerr=raw_std, capsize=6,
                color='#BBBBBB', edgecolor='black', linewidth=1.6,
                error_kw={'elinewidth': 2.0}, label='Raw')
    b2 = ax.bar(x + width / 2, z_mean, width, yerr=z_std, capsize=6,
                color='#4C72B0', edgecolor='black', linewidth=1.6,
                error_kw={'elinewidth': 2.0}, label='Z-scored')

    ax.scatter(x - width / 2, raw_shuf, marker='_', s=320, color='black',
               linewidths=2.5, label='Shuffled (raw)', zorder=5)
    ax.scatter(x + width / 2, z_shuf, marker='_', s=320, color='black',
               linewidths=2.5, zorder=5)

    for xi, c in zip(x, chance):
        ax.hlines(c, xi - width, xi + width, linestyles='--',
                  colors='red', linewidth=1.8,
                  label='Chance' if xi == x[0] else None)

    for bars, means, stds in [(b1, raw_mean, raw_std), (b2, z_mean, z_std)]:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(m + s + 0.02, 1.05),
                    f'{m:.3f}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)
    ax.set_ylim(0, max(1.05, (np.maximum(raw_mean + raw_std, z_mean + z_std)).max() + 0.15))
    ax.set_ylabel('CV decoding accuracy', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=13, width=1.6, length=6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(1.8)
    ax.legend(frameon=False, fontsize=12, loc='upper right', ncol=2)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved figure to: {save_path}")

    return fig


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_comparison(data_path, time_window=(0.07, 0.16), save_plots=True,
                   output_path=None):
    """
    Compare raw vs z-scored LDA decoding for orientation, split by SF if present.
    """
    data = load_neural_data(data_path)
    firing_rates, ori_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )

    if len(ori_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")

    unique_sfs = trial_info['unique_spatial_freqs']
    sf_labels = trial_info['spatial_freq_labels']
    results_by_group = {}

    print(f"\n{'#' * 60}")
    print(f"# RAW vs Z-SCORE LDA — ORIENTATION DECODING")
    print(f"{'#' * 60}")

    for sf in unique_sfs:
        if sf is None:
            fr_sf, lbl_sf, tag = firing_rates, ori_labels, 'all SF'
        else:
            mask = sf_labels == sf
            fr_sf, lbl_sf, tag = firing_rates[mask], ori_labels[mask], f'SF={sf} cpd'

        if len(lbl_sf) == 0 or len(set(lbl_sf.tolist())) < 2:
            print(f"\n[{tag}] skipped (insufficient orientations).")
            continue

        print(f"\n[{tag}]  ({len(lbl_sf)} trials)")
        results_by_group[tag] = compare_raw_vs_zscore(fr_sf, lbl_sf)

    if not results_by_group:
        print("No groups with enough data to compare.")
        return results_by_group

    save_path = None
    if save_plots:
        base = Path(output_path) if output_path else Path(data_path).with_suffix('')
        save_path = Path(str(base) + '.lda_raw_vs_zscore.png')

    plot_comparison(results_by_group,
                    title='LDA Decoding: Raw vs Z-scored Firing Rates',
                    save_path=save_path)

    return results_by_group


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = input("Enter path to neural data (.pkl file): ").strip().strip('"').strip("'")
    try:
        run_comparison(
            data_path=DATA_PATH,
            time_window=(0.05, 1.5),
            save_plots=True,
        )
        print("\nDone.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
    except Exception as e:
        print(f"Error during comparison: {e}")
        raise
