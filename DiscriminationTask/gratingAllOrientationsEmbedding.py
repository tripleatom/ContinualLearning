"""
All-Orientations Grating + Behavior LDA Embedding
==================================================

Trains an LDA decoder on ALL grating orientations (not just 45°/135°),
then augments the 45° and 135° classes with behavior trials:
  behavior-left  → added to 45°  class
  behavior-right → added to 135° class

This increases the sample size of those two classes while keeping the
full multi-class structure that gives ≥3 LDA components (for 3D embedding).

Reports CV accuracy before and after mixing behavior data.
Visualization shows only 45°/135° + behavior points for clarity.

Label mapping:
  Grating 45°   → stays 45.0   (red circles,   grating)
  Grating 135°  → stays 135.0  (blue circles,  grating)
  Behavior Left  → mapped 45.0  (orange triangles)
  Behavior Right → mapped 135.0 (cyan triangles)
  Other orientations used for training but NOT plotted.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
_GRATING_DIR = _THIS_DIR.parent / 'rf_recon' / 'FreelyMovingProcessing' / 'Grating'
sys.path.insert(0, str(_GRATING_DIR))

import grating_utils
from gratingDecodeBehavior import load_behavior_data, align_units


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_grating_data(grating_pkl_path, time_window=(0.07, 0.16)):
    """
    Load grating data keeping ALL orientations (no orientation filter).

    Returns
    -------
    firing_rates      : (n_trials, n_units)
    orientation_labels: (n_trials,)  float degrees
    unit_ids          : list of str
    """
    data = grating_utils.load_neural_data(grating_pkl_path)

    # Filter noise units
    unit_info = data.get('unit_info', {})
    all_unit_ids = list(data['spike_data'].keys())
    good_units = [u for u in all_unit_ids
                  if unit_info.get(u, {}).get('quality', 'unknown') != 'noise']
    n_noise = len(all_unit_ids) - len(good_units)
    if n_noise:
        print(f"  [Grating] Excluded {n_noise} noise unit(s)")
    data_filtered = dict(data)
    data_filtered['spike_data'] = {u: data['spike_data'][u] for u in good_units}

    firing_rates, orientation_labels, unit_ids, trial_info = \
        grating_utils.calculate_firing_rates(data_filtered, time_window=time_window)

    orientation_labels = orientation_labels.astype(float)

    unique_oris, counts = np.unique(orientation_labels, return_counts=True)
    print(f"\n[Grating] All orientations loaded:")
    for ori, cnt in zip(unique_oris, counts):
        print(f"  {ori:6.1f}°: {cnt} trials")
    print(f"  Total: {len(orientation_labels)} trials  |  Units: {len(unit_ids)}")

    return firing_rates, orientation_labels, unit_ids


# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================

def build_datasets(grating_fr, orientation_labels, behavior_fr, behavior_labels):
    """
    Build pre-mix and post-mix datasets.

    Pre-mix  : grating only, all orientations
    Post-mix : grating + behavior (behavior merged into 45°/135° classes)

    Returns
    -------
    X_pre, y_pre       : grating-only dataset
    X_post, y_post     : mixed dataset
    source_post        : (n_post,)  0=grating, 1=behavior
    """
    # ---- pre-mix: grating only ----
    X_pre = grating_fr.copy()
    y_pre = orientation_labels.copy()

    # ---- post-mix: append behavior ----
    beh_mapped = np.where(behavior_labels == 1, 45.0, 135.0)
    X_post  = np.vstack([grating_fr, behavior_fr])
    y_post  = np.concatenate([orientation_labels, beh_mapped])
    source_post = np.concatenate([np.zeros(len(grating_fr), dtype=int),
                                   np.ones(len(behavior_fr),  dtype=int)])

    # Print class size comparison
    print("\n[Dataset] Class sizes before / after mixing behavior:")
    all_oris = np.unique(y_post)
    for ori in all_oris:
        n_pre  = np.sum(y_pre  == ori)
        n_post = np.sum(y_post == ori)
        tag = f"  (+{n_post - n_pre} behavior)" if n_post > n_pre else ""
        print(f"  {ori:6.1f}°:  pre={n_pre:3d}  post={n_post:3d}{tag}")

    return X_pre, y_pre, X_post, y_post, source_post


# =============================================================================
# ANALYSIS
# =============================================================================

def cv_accuracy(fr, labels, tag):
    """5-fold stratified LDA CV. Returns (scores, chance_level)."""
    labels = np.array(labels).astype(str)   # float labels → string class labels
    scaler = StandardScaler()
    X = scaler.fit_transform(fr)
    unique = np.unique(labels)
    min_n = min(np.sum(labels == c) for c in unique)
    n_folds = max(2, min(5, int(min_n)))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(LinearDiscriminantAnalysis(), X, labels,
                             cv=cv, scoring='accuracy')
    chance = 1.0 / len(unique)
    print(f"[CV — {tag}]  {n_folds}-fold: "
          f"{scores.mean():.3f} ± {scores.std():.3f}  "
          f"(chance={chance:.3f}, above={scores.mean()-chance:+.3f})")
    return scores, chance


def fit_lda_embedding(X, y, n_components=3):
    """Fit StandardScaler + LDA on (X, y). Returns scaler, lda, X_lda."""
    y = np.array(y).astype(str)             # float labels → string class labels
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    n_cls = len(np.unique(y))
    n_comp = min(n_components, n_cls - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_lda = lda.fit_transform(X_sc, y)
    print(f"\n[LDA Embedding] n_components={n_comp}  "
          f"(classes={n_cls}, features={X.shape[1]})")
    if hasattr(lda, 'explained_variance_ratio_'):
        for i, ev in enumerate(lda.explained_variance_ratio_):
            print(f"  LD{i+1} explained variance: {ev:.3f}")
    return scaler, lda, X_lda


# =============================================================================
# VISUALIZATION
# =============================================================================

_COLOR  = {(45.0,  0): '#E74C3C',   # grating 45°   → red
           (45.0,  1): '#FF8C69',   # behavior left → salmon
           (135.0, 0): '#3498DB',   # grating 135°  → blue
           (135.0, 1): '#1ABC9C'}   # behavior right → teal

_MARKER = {0: 'o', 1: '^'}
_SIZE   = {0: 25,  1: 40}

_LABEL  = {(45.0,  0): 'Grating 45°',
           (45.0,  1): 'Behavior Left',
           (135.0, 0): 'Grating 135°',
           (135.0, 1): 'Behavior Right'}


def _iter_plot_groups(y_plot, source_plot):
    for ori in [45.0, 135.0]:
        for src in [0, 1]:
            mask = (y_plot == ori) & (source_plot == src)
            if mask.any():
                yield ori, src, mask


def _scatter_3d(ax, X_plot, y_plot, source_plot, lda):
    n_comp = X_plot.shape[1]
    for ori, src, mask in _iter_plot_groups(y_plot, source_plot):
        x0 = X_plot[mask, 0]
        x1 = X_plot[mask, 1] if n_comp > 1 else np.zeros(mask.sum())
        x2 = X_plot[mask, 2] if n_comp > 2 else np.zeros(mask.sum())
        ax.scatter(x0, x1, x2,
                   c=_COLOR[(ori, src)], marker=_MARKER[src],
                   s=_SIZE[src], alpha=0.65, label=_LABEL[(ori, src)],
                   edgecolors='none')
    ev = getattr(lda, 'explained_variance_ratio_', [0, 0, 0])
    ax.set_xlabel(f'LD1 ({ev[0]:.2f})' if len(ev) > 0 else 'LD1', fontsize=8)
    ax.set_ylabel(f'LD2 ({ev[1]:.2f})' if len(ev) > 1 else 'LD2', fontsize=8)
    ax.set_zlabel(f'LD3 ({ev[2]:.2f})' if len(ev) > 2 else 'LD3', fontsize=8)
    ax.set_title('LDA 3D Embedding\n(45°/135° + behavior only)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.view_init(elev=20, azim=45)


def _scatter_2d(ax, X_plot, y_plot, source_plot, dim_x, dim_y, lda):
    n_comp = X_plot.shape[1]
    if n_comp <= dim_y:
        ax.axis('off')
        ax.text(0.5, 0.5, f'LD{dim_y+1} not available',
                ha='center', va='center', transform=ax.transAxes)
        return
    for ori, src, mask in _iter_plot_groups(y_plot, source_plot):
        ax.scatter(X_plot[mask, dim_x], X_plot[mask, dim_y],
                   c=_COLOR[(ori, src)], marker=_MARKER[src],
                   s=_SIZE[src], alpha=0.65, label=_LABEL[(ori, src)],
                   edgecolors='none')
    ev = getattr(lda, 'explained_variance_ratio_', [])
    def _ax_label(d):
        return f'LD{d+1} ({ev[d]:.2f})' if d < len(ev) else f'LD{d+1}'
    ax.set_xlabel(_ax_label(dim_x), fontsize=9)
    ax.set_ylabel(_ax_label(dim_y), fontsize=9)
    ax.set_title(f'LD{dim_x+1} vs LD{dim_y+1}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_cv_bars(ax, cv_pre, cv_post, chance_pre, chance_post):
    means = [cv_pre.mean(), cv_post.mean()]
    stds  = [cv_pre.std(),  cv_post.std()]
    bars = ax.bar([0, 1], means, yerr=stds, capsize=6,
                  color=['#2ECC71', '#9B59B6'], alpha=0.85,
                  error_kw={'linewidth': 2})
    # chance lines (pre and post may differ if n_orientations changes)
    ax.axhline(chance_pre,  color='#27AE60', linestyle='--', linewidth=1.5,
               label=f'Chance pre ({chance_pre:.2f})')
    if abs(chance_post - chance_pre) > 0.001:
        ax.axhline(chance_post, color='#8E44AD', linestyle=':',  linewidth=1.5,
                   label=f'Chance post ({chance_post:.2f})')
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Grating Only\n(all orient.)',
                        'Mixed\n(+behavior in 45°/135°)'], fontsize=9)
    ax.set_ylabel('CV Accuracy', fontsize=10)
    ax.set_title('CV Accuracy: Before vs After Mix',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')


def _plot_summary(ax, cv_pre, cv_post, chance_pre, chance_post,
                  orientation_labels, behavior_labels,
                  shared_unit_ids, lda, run_params):
    ax.axis('off')
    g_win = run_params.get('grating_time_window', ('?', '?'))
    b_win = run_params.get('behavior_time_window', ('?', '?'))
    sf_val = run_params.get('spatial_freq_filter')
    sf_str = f"{sf_val} cpd" if sf_val is not None else "all"
    ev = getattr(lda, 'explained_variance_ratio_', [])

    unique_oris, counts = np.unique(orientation_labels, return_counts=True)
    ori_lines = "".join(f"  {o:6.1f}°: {c}\n" for o, c in zip(unique_oris, counts))

    txt = (
        f"All-Orientations Embedding Summary\n"
        f"{'─'*36}\n"
        f"Shared units:  {len(shared_unit_ids)}\n"
        f"SF filter:     {sf_str}\n\n"
        f"[Grating — all orientations]\n"
        f"  Window:  {g_win[0]:.2f}–{g_win[1]:.2f} s\n"
        + ori_lines +
        f"\n[Behavior added to 45°/135°]\n"
        f"  Window:  {b_win[0]:.2f}–{b_win[1]:.2f} s\n"
        f"  Left → 45°:   {np.sum(behavior_labels==1)}\n"
        f"  Right → 135°: {np.sum(behavior_labels==0)}\n\n"
        f"[CV — {len(unique_oris)}-class LDA]\n"
        f"  Pre-mix:  {cv_pre.mean():.3f} ± {cv_pre.std():.3f}\n"
        f"  Post-mix: {cv_post.mean():.3f} ± {cv_post.std():.3f}\n"
        f"  Chance:   {chance_pre:.3f}\n\n"
        f"[LDA explained variance]\n"
        + "".join(f"  LD{i+1}: {v:.3f}\n" for i, v in enumerate(ev))
    )
    ax.text(0.04, 0.98, txt, transform=ax.transAxes, fontsize=8.2,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def create_figure(X_lda_plot, y_plot, source_plot,
                  cv_pre, cv_post, chance_pre, chance_post,
                  orientation_labels, behavior_labels,
                  shared_unit_ids, lda,
                  run_params,
                  save_path=None):
    """
    Layout (2 rows × 3 cols):
      [0,0] 3D embedding     [0,1] CV bars   [0,2] LD1 vs LD2
      [1,0] LD1 vs LD3       [1,1] LD2 vs LD3  [1,2] Summary
    """
    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax3d  = fig.add_subplot(gs[0, 0], projection='3d')
    ax_cv = fig.add_subplot(gs[0, 1])
    ax12  = fig.add_subplot(gs[0, 2])
    ax13  = fig.add_subplot(gs[1, 0])
    ax23  = fig.add_subplot(gs[1, 1])
    ax_tx = fig.add_subplot(gs[1, 2])

    _scatter_3d(ax3d, X_lda_plot, y_plot, source_plot, lda)
    _plot_cv_bars(ax_cv, cv_pre, cv_post, chance_pre, chance_post)
    _scatter_2d(ax12, X_lda_plot, y_plot, source_plot, 0, 1, lda)
    _scatter_2d(ax13, X_lda_plot, y_plot, source_plot, 0, 2, lda)
    _scatter_2d(ax23, X_lda_plot, y_plot, source_plot, 1, 2, lda)
    _plot_summary(ax_tx, cv_pre, cv_post, chance_pre, chance_post,
                  orientation_labels, behavior_labels,
                  shared_unit_ids, lda, run_params)

    sf_val = run_params.get('spatial_freq_filter')
    sf_str = f"{sf_val} cpd" if sf_val is not None else "all SF"
    n_oris = len(np.unique(orientation_labels))
    fig.suptitle(
        f"All-Orientations LDA Embedding  •  {n_oris} grating orientations  •  SF={sf_str}\n"
        f"LDA trained on all orientations  •  behavior merged into 45°/135°  "
        f"•  visualization: 45°/135° + behavior only  "
        f"(○=grating  ▲=behavior)",
        fontsize=11, fontweight='bold', y=1.01
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_embedding(grating_pkl, behavior_pkl,
                  grating_time_window=(0.07, 0.16),
                  behavior_time_window=(0.0, 1.0),
                  spatial_freq_filter=None,
                  save_plots=True,
                  output_path=None):
    """
    Full pipeline:
      1. Load ALL grating orientations + behavior
      2. Align to shared units
      3. Build pre/post-mix datasets
      4. CV accuracy before and after mixing
      5. Fit LDA on post-mix data → 3D embedding
      6. Plot only 45°/135° + behavior points

    Returns dict with all results.
    """
    print("=" * 60)
    print("All-Orientations Grating + Behavior LDA Embedding")
    print("=" * 60)

    # 1. Load
    grating_fr, orientation_labels, grating_unit_ids = \
        load_all_grating_data(grating_pkl, grating_time_window)

    behavior_fr, behavior_labels, behavior_unit_ids, _ = \
        load_behavior_data(behavior_pkl, behavior_time_window)

    # 2. Align
    shared_unit_ids, grating_fr_sh, behavior_fr_sh = align_units(
        grating_unit_ids, behavior_unit_ids, grating_fr, behavior_fr
    )

    # 3. Build datasets
    X_pre, y_pre, X_post, y_post, source_post = build_datasets(
        grating_fr_sh, orientation_labels,
        behavior_fr_sh, behavior_labels
    )

    # 4. CV accuracy
    print()
    cv_pre,  chance_pre  = cv_accuracy(X_pre,  y_pre,  'Grating Only (all orient.)')
    cv_post, chance_post = cv_accuracy(X_post, y_post, 'Mixed (+behavior in 45°/135°)')

    # 5. Fit LDA on post-mix for embedding
    scaler, lda, X_lda_post = fit_lda_embedding(X_post, y_post, n_components=3)

    # 6. Subset: only 45°/135° grating + behavior for plotting
    plot_mask = np.isin(y_post, [45.0, 135.0])
    X_lda_plot  = X_lda_post[plot_mask]
    y_plot       = y_post[plot_mask]
    source_plot  = source_post[plot_mask]

    n_g_plot = np.sum(plot_mask & (source_post == 0))
    n_b_plot = np.sum(plot_mask & (source_post == 1))
    print(f"\n[Plot subset] {n_g_plot} grating + {n_b_plot} behavior points shown")

    # 7. Figure
    run_params = {
        'grating_pkl':          str(grating_pkl),
        'behavior_pkl':         str(behavior_pkl),
        'grating_time_window':  grating_time_window,
        'behavior_time_window': behavior_time_window,
        'spatial_freq_filter':  spatial_freq_filter,
    }

    if output_path is None and save_plots:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        sf_tag = f'_sf{spatial_freq_filter}' if spatial_freq_filter else ''
        output_path = (
            Path(behavior_pkl).parent / 'passive-behavior' /
            (Path(behavior_pkl).stem +
             f'.all_orientations_embedding{sf_tag}_{ts}.png')
        )

    create_figure(
        X_lda_plot=X_lda_plot, y_plot=y_plot, source_plot=source_plot,
        cv_pre=cv_pre, cv_post=cv_post,
        chance_pre=chance_pre, chance_post=chance_post,
        orientation_labels=orientation_labels,
        behavior_labels=behavior_labels,
        shared_unit_ids=shared_unit_ids,
        lda=lda,
        run_params=run_params,
        save_path=output_path if save_plots else None,
    )

    plt.show()

    return {
        'lda':               lda,
        'scaler':            scaler,
        'X_lda_post':        X_lda_post,
        'X_lda_plot':        X_lda_plot,
        'y_post':            y_post,
        'y_plot':            y_plot,
        'source_post':       source_post,
        'cv_pre':            cv_pre,
        'cv_post':           cv_post,
        'chance_pre':        chance_pre,
        'chance_post':       chance_post,
        'shared_unit_ids':   shared_unit_ids,
        'orientation_labels': orientation_labels,
    }


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    GRATING_PKL = (
        "//Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/"
        "passive_embedding_analysis/"
        "CnL42SG_CnL42SG_passive_20260304_142720_grating_data.pkl"
    )
    BEHAVIOR_PKL = (
        "//Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/"
        "behavior_trial_embedding_20260309_2000.pkl"
    )

    try:
        results = run_embedding(
            grating_pkl=GRATING_PKL,
            behavior_pkl=BEHAVIOR_PKL,
            grating_time_window=(0.05, 1.5),
            behavior_time_window=(0.05, 1.5),
            spatial_freq_filter=None,
            save_plots=True,
        )

        print("\n" + "=" * 60)
        print("Embedding complete!")
        print("=" * 60)
        print(f"  Shared units:      {len(results['shared_unit_ids'])}")
        print(f"  LDA components:    {results['X_lda_post'].shape[1]}")
        print(f"  CV pre-mix:        {results['cv_pre'].mean():.3f} "
              f"± {results['cv_pre'].std():.3f}  "
              f"(chance={results['chance_pre']:.3f})")
        print(f"  CV post-mix:       {results['cv_post'].mean():.3f} "
              f"± {results['cv_post'].std():.3f}  "
              f"(chance={results['chance_post']:.3f})")

    except FileNotFoundError as e:
        print(f"Error: Data file not found — {e}")
        print("Update GRATING_PKL and BEHAVIOR_PKL paths.")
    except Exception as e:
        print(f"Error: {e}")
        raise
