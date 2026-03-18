"""
Grating + Behavior Mixed LDA / PCA Embedding
=============================================

Combines grating (45°/135°) and behavior (reward left/right) trials from
shared units using a 2-class label scheme:

  left-equiv  (class 1): grating 45°  +  behavior left
  right-equiv (class 0): grating 135° +  behavior right

Within each class, grating vs behavior trials are distinguished by marker.

Outputs
-------
  • 3D PCA scatter (color = class, marker = grating○ / behavior▲)
  • 2D PCA projections (PC1×PC2, PC1×PC3)
  • LDA 1D projection + histogram (grating vs behavior overlay)
  • CV accuracy: grating-only (2-class) vs mixed (2-class)
  • Summary statistics panel
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup so gratingDecodeBehavior and grating_utils are importable
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
_GRATING_DIR = _THIS_DIR.parent / 'rf_recon' / 'FreelyMovingProcessing' / 'Grating'
sys.path.insert(0, str(_GRATING_DIR))

from gratingDecodeBehavior import load_grating_data, load_behavior_data, align_units


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def build_mixed_dataset(grating_fr, grating_binary_labels,
                        behavior_fr, behavior_labels):
    """
    Stack grating and behavior firing rates with shared 2-class labels.

    Parameters
    ----------
    grating_binary_labels : (n_g,)  1=45°(left-equiv), 0=135°(right-equiv)
    behavior_labels       : (n_b,)  1=left,             0=right

    Returns
    -------
    X      : (n_g + n_b, n_units)  stacked firing rates
    y      : (n_g + n_b,)          2-class labels  (1=left-equiv, 0=right-equiv)
    source : (n_g + n_b,)          0=grating, 1=behavior
    """
    X = np.vstack([grating_fr, behavior_fr])
    y = np.concatenate([grating_binary_labels, behavior_labels]).astype(int)
    source = np.concatenate([np.zeros(len(grating_fr), dtype=int),
                              np.ones(len(behavior_fr),  dtype=int)])
    return X, y, source


def run_analysis(X, y):
    """
    Fit StandardScaler, LDA (1D), and PCA (3D) on the mixed dataset.

    Returns
    -------
    scaler, lda, pca, X_lda (n, 1), X_pca (n, 3)
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_sc, y)          # (n, 1)

    n_pc = min(3, X.shape[1])
    pca = PCA(n_components=n_pc)
    X_pca = pca.fit_transform(X_sc)             # (n, n_pc)

    print(f"\n[LDA] 1 discriminant axis (binary classification)")
    print(f"[PCA] Explained variance: "
          + "  ".join(f"PC{i+1}={v:.3f}" for i, v in
                      enumerate(pca.explained_variance_ratio_)))
    return scaler, lda, pca, X_lda, X_pca


def cv_accuracy(fr, labels, tag, chance):
    """5-fold stratified LDA CV. Returns score array."""
    scaler = StandardScaler()
    X = scaler.fit_transform(fr)
    min_n = min(np.sum(labels == c) for c in np.unique(labels))
    n_folds = max(2, min(5, int(min_n)))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(LinearDiscriminantAnalysis(), X, labels,
                             cv=cv, scoring='accuracy')
    print(f"[CV — {tag}]  {n_folds}-fold: "
          f"{scores.mean():.3f} ± {scores.std():.3f}  (chance={chance:.3f})")
    return scores


# =============================================================================
# VISUALIZATION
# =============================================================================

# colours: left-equiv=red family, right-equiv=blue family
# markers: grating=circle, behavior=triangle
_COLOR = {(1, 0): '#E74C3C',   # left,  grating  → red
          (1, 1): '#FF8C69',   # left,  behavior → salmon
          (0, 0): '#3498DB',   # right, grating  → blue
          (0, 1): '#1ABC9C'}   # right, behavior → teal

_MARKER = {0: 'o', 1: '^'}    # grating=circle, behavior=triangle

_LABEL = {(1, 0): 'Grating 45°',
          (1, 1): 'Behavior Left',
          (0, 0): 'Grating 135°',
          (0, 1): 'Behavior Right'}

_SIZE   = {0: 25,  1: 40}     # behavior points slightly larger


def create_figure(X_pca, X_lda, y, source,
                  cv_grating, cv_mixed,
                  grating_fr, grating_labels,
                  behavior_fr, behavior_labels,
                  shared_unit_ids, pca,
                  run_params,
                  save_path=None):
    """
    Layout (2 rows × 3 cols):
      [0,0] 3D PCA           [0,1] CV accuracy bars   [0,2] PC1 vs PC2
      [1,0] PC1 vs PC3       [1,1] LDA 1D projection  [1,2] Summary text
    """
    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax3d  = fig.add_subplot(gs[0, 0], projection='3d')
    ax_cv = fig.add_subplot(gs[0, 1])
    ax12  = fig.add_subplot(gs[0, 2])
    ax13  = fig.add_subplot(gs[1, 0])
    ax_ld = fig.add_subplot(gs[1, 1])
    ax_tx = fig.add_subplot(gs[1, 2])

    _plot_3d(ax3d, X_pca, y, source, pca)
    _plot_cv_bars(ax_cv, cv_grating, cv_mixed)
    _plot_2d(ax12, X_pca, y, source, dim_x=0, dim_y=1)
    _plot_2d(ax13, X_pca, y, source, dim_x=0, dim_y=2)
    _plot_lda_1d(ax_ld, X_lda, y, source)
    _plot_summary(ax_tx, cv_grating, cv_mixed,
                  grating_labels, behavior_labels,
                  shared_unit_ids, pca, run_params)

    sf_val = run_params.get('spatial_freq_filter')
    sf_str = f"{sf_val} cpd" if sf_val is not None else "all SF"
    fig.suptitle(
        f"Grating + Behavior Mixed Embedding  •  SF={sf_str}\n"
        f"shared units={len(shared_unit_ids)}  "
        f"grating={len(grating_fr)} trials  "
        f"behavior={len(behavior_fr)} trials  "
        f"(○=grating  ▲=behavior  red=left-equiv  blue=right-equiv)",
        fontsize=12, fontweight='bold', y=1.01
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Helper plot functions
# ---------------------------------------------------------------------------

def _iter_groups(y, source):
    """Yield (cls, src, mask) for the 4 display groups."""
    for cls in [1, 0]:
        for src in [0, 1]:
            mask = (y == cls) & (source == src)
            if mask.any():
                yield cls, src, mask


def _plot_3d(ax, X_pca, y, source, pca):
    n_pc = X_pca.shape[1]
    for cls, src, mask in _iter_groups(y, source):
        kw = dict(c=_COLOR[(cls, src)], marker=_MARKER[src],
                  alpha=0.65, s=_SIZE[src], label=_LABEL[(cls, src)],
                  edgecolors='none')
        x0 = X_pca[mask, 0]
        x1 = X_pca[mask, 1] if n_pc > 1 else np.zeros(mask.sum())
        x2 = X_pca[mask, 2] if n_pc > 2 else np.zeros(mask.sum())
        ax.scatter(x0, x1, x2, **kw)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=8)
    if n_pc > 1:
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=8)
    if n_pc > 2:
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2f})', fontsize=8)
    ax.set_title('PCA 3D Embedding\n(color=class  marker=source)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.view_init(elev=20, azim=45)


def _plot_2d(ax, X_pca, y, source, dim_x, dim_y):
    n_pc = X_pca.shape[1]
    if n_pc <= dim_y:
        ax.axis('off')
        ax.text(0.5, 0.5, f'PC{dim_y+1} not available',
                ha='center', va='center', transform=ax.transAxes)
        return
    for cls, src, mask in _iter_groups(y, source):
        ax.scatter(X_pca[mask, dim_x], X_pca[mask, dim_y],
                   c=_COLOR[(cls, src)], marker=_MARKER[src],
                   alpha=0.65, s=_SIZE[src], label=_LABEL[(cls, src)],
                   edgecolors='none')
    ax.set_xlabel(f'PC{dim_x+1}', fontsize=9)
    ax.set_ylabel(f'PC{dim_y+1}', fontsize=9)
    ax.set_title(f'PC{dim_x+1} vs PC{dim_y+1}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_lda_1d(ax, X_lda, y, source):
    """Overlay LDA 1D projections: grating (filled) vs behavior (outline)."""
    ld = X_lda[:, 0]
    colors_cls = {1: '#E74C3C', 0: '#3498DB'}   # red=left, blue=right
    cls_name   = {1: 'Left-equiv (45°/Left)', 0: 'Right-equiv (135°/Right)'}

    for cls in [1, 0]:
        # grating: filled histogram
        g_mask = (y == cls) & (source == 0)
        if g_mask.any():
            ax.hist(ld[g_mask], bins=20, alpha=0.55,
                    color=colors_cls[cls], density=True,
                    histtype='stepfilled',
                    label=f'Grating {cls_name[cls].split("(")[1].rstrip(")")}')
        # behavior: step outline
        b_mask = (y == cls) & (source == 1)
        if b_mask.any():
            ax.hist(ld[b_mask], bins=20, alpha=0.85,
                    color=colors_cls[cls], density=True,
                    histtype='step', linewidth=2, linestyle='--',
                    label=f'Behavior {cls_name[cls].split("(")[1].rstrip(")")}')

    ax.axvline(0, color='black', linestyle=':', linewidth=1.5,
               label='Decision boundary')
    ax.set_xlabel('LDA Score (LD1)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('LDA 1D Projection\n(solid=grating  dashed=behavior)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_cv_bars(ax, cv_grating, cv_mixed):
    means = [cv_grating.mean(), cv_mixed.mean()]
    stds  = [cv_grating.std(),  cv_mixed.std()]
    bars = ax.bar([0, 1], means, yerr=stds, capsize=6,
                  color=['#2ECC71', '#9B59B6'], alpha=0.85,
                  error_kw={'linewidth': 2})
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5,
               label='Chance (0.50)')
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Grating Only', 'Mixed\n(grating+behavior)'], fontsize=9)
    ax.set_ylabel('CV Accuracy', fontsize=10)
    ax.set_title('CV Accuracy Comparison\n(2-class: left-equiv vs right-equiv)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')


def _plot_summary(ax, cv_grating, cv_mixed,
                  grating_labels, behavior_labels,
                  shared_unit_ids, pca, run_params):
    ax.axis('off')
    g_win = run_params.get('grating_time_window', ('?', '?'))
    b_win = run_params.get('behavior_time_window', ('?', '?'))
    sf_val = run_params.get('spatial_freq_filter')
    sf_str = f"{sf_val} cpd" if sf_val is not None else "all"
    ev = pca.explained_variance_ratio_

    txt = (
        f"Mixed Embedding Summary\n"
        f"{'─'*36}\n"
        f"Label mapping:\n"
        f"  Grating 45°  + Behavior Left  = class 1\n"
        f"  Grating 135° + Behavior Right = class 0\n\n"
        f"Shared units:  {len(shared_unit_ids)}\n"
        f"SF filter:     {sf_str}\n\n"
        f"[Grating]\n"
        f"  Window:      {g_win[0]:.2f}–{g_win[1]:.2f} s\n"
        f"  45° trials:  {np.sum(grating_labels == 1)}\n"
        f"  135° trials: {np.sum(grating_labels == 0)}\n\n"
        f"[Behavior]\n"
        f"  Window:      {b_win[0]:.2f}–{b_win[1]:.2f} s\n"
        f"  Left trials: {np.sum(behavior_labels == 1)}\n"
        f"  Right trials:{np.sum(behavior_labels == 0)}\n\n"
        f"[CV — 2-class LDA]\n"
        f"  Grating only: {cv_grating.mean():.3f} ± {cv_grating.std():.3f}\n"
        f"  Mixed:        {cv_mixed.mean():.3f} ± {cv_mixed.std():.3f}\n"
        f"  Chance:       0.500\n\n"
        f"[PCA explained variance]\n"
        + "".join(f"  PC{i+1}: {v:.3f}\n" for i, v in enumerate(ev))
    )
    ax.text(0.04, 0.98, txt, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_embedding(grating_pkl, behavior_pkl,
                  grating_time_window=(0.05, 1.5),
                  behavior_time_window=(0.05, 1.5),
                  target_orientations=(45.0, 135.0),
                  spatial_freq_filter=None,
                  save_plots=True,
                  output_path=None):
    """
    Full pipeline:
      1. Load grating + behavior, align to shared units
      2. Merge with 2-class labels (45°/left=1, 135°/right=0)
      3. PCA 3D embedding + LDA 1D projection
      4. CV accuracy: grating-only vs mixed (both 2-class)
      5. Visualise

    Returns
    -------
    dict: scaler, lda, pca, X_lda, X_pca, y, source,
          cv_grating, cv_mixed, shared_unit_ids
    """
    print("=" * 60)
    print("Grating + Behavior Mixed LDA/PCA Embedding")
    print("=" * 60)

    # 1. Load
    grating_fr, grating_labels, grating_unit_ids, _ = \
        load_grating_data(grating_pkl, grating_time_window, target_orientations,
                          spatial_freq_filter=spatial_freq_filter)

    behavior_fr, behavior_labels, behavior_unit_ids, _ = \
        load_behavior_data(behavior_pkl, behavior_time_window)

    # 2. Align to shared units
    shared_unit_ids, grating_fr_sh, behavior_fr_sh = align_units(
        grating_unit_ids, behavior_unit_ids, grating_fr, behavior_fr
    )

    # 3. Build 2-class mixed dataset
    X, y, source = build_mixed_dataset(
        grating_fr_sh, grating_labels,
        behavior_fr_sh, behavior_labels
    )
    print(f"\n[Mixed dataset] {len(X)} total trials  "
          f"(left-equiv: {np.sum(y==1)}, right-equiv: {np.sum(y==0)})")
    print(f"  Grating trials:  {np.sum(source==0)}")
    print(f"  Behavior trials: {np.sum(source==1)}")

    # 4. Fit LDA + PCA
    scaler, lda, pca, X_lda, X_pca = run_analysis(X, y)

    # 5. CV accuracies (both 2-class)
    cv_grating = cv_accuracy(grating_fr_sh, grating_labels,
                             'Grating Only (2-class)', 0.5)
    cv_mixed   = cv_accuracy(X, y,
                             'Mixed (2-class)', 0.5)

    # 6. Figure
    run_params = {
        'grating_pkl':          str(grating_pkl),
        'behavior_pkl':         str(behavior_pkl),
        'grating_time_window':  grating_time_window,
        'behavior_time_window': behavior_time_window,
        'spatial_freq_filter':  spatial_freq_filter,
    }

    if output_path is None and save_plots:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        sf_tag = f'_sf{spatial_freq_filter}' if spatial_freq_filter else ''
        output_path = (
            Path(behavior_pkl).parent / 'passive-behavior' /
            (Path(behavior_pkl).stem + f'.grating_behavior_embedding{sf_tag}_{ts}.png')
        )

    create_figure(
        X_pca=X_pca, X_lda=X_lda, y=y, source=source,
        cv_grating=cv_grating, cv_mixed=cv_mixed,
        grating_fr=grating_fr_sh, grating_labels=grating_labels,
        behavior_fr=behavior_fr_sh, behavior_labels=behavior_labels,
        shared_unit_ids=shared_unit_ids,
        pca=pca,
        run_params=run_params,
        save_path=output_path if save_plots else None,
    )

    plt.show()

    return {
        'scaler':          scaler,
        'lda':             lda,
        'pca':             pca,
        'X_lda':           X_lda,
        'X_pca':           X_pca,
        'y':               y,
        'source':          source,
        'cv_grating':      cv_grating,
        'cv_mixed':        cv_mixed,
        'shared_unit_ids': shared_unit_ids,
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
            target_orientations=(45.0, 135.0),
            spatial_freq_filter=0.16,  # set to None to disable SF filtering
            save_plots=True,
        )

        print("\n" + "=" * 60)
        print("Embedding complete!")
        print("=" * 60)
        print(f"  Shared units:         {len(results['shared_unit_ids'])}")
        print(f"  Grating CV (2-class): {results['cv_grating'].mean():.3f} "
              f"± {results['cv_grating'].std():.3f}")
        print(f"  Mixed CV (2-class):   {results['cv_mixed'].mean():.3f} "
              f"± {results['cv_mixed'].std():.3f}")

    except FileNotFoundError as e:
        print(f"Error: Data file not found — {e}")
        print("Update GRATING_PKL and BEHAVIOR_PKL paths.")
    except Exception as e:
        print(f"Error: {e}")
        raise
