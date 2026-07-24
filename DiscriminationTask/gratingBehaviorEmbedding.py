"""
Grating + Behavior Mixed LDA Embedding
=======================================

Combines two grating (ori, SF) conditions and behavior (reward left/right) trials
from shared units using a 2-class label scheme:

  left-equiv  (class 1): behavior_left_stim  grating  +  behavior left
  right-equiv (class 0): behavior_right_stim grating  +  behavior right

Within each class, grating vs behavior trials are distinguished by marker.

Outputs
-------
  • LDA scatter: all trials projected onto LD1 (color=class, marker=grating○/behavior▲)
  • LDA scatter split by source (grating panel | behavior panel)
  • LDA 1D histogram (grating vs behavior overlay)
  • CV accuracy: grating-only / mixed / shuffled-behavior-mixed
  • Summary statistics panel
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup so gratingDecodeBehavior and grating_utils are importable
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
_GRATING_DIR = _THIS_DIR.parent / 'passive_visual' / 'FreelyMovingProcessing' / 'Grating'
sys.path.insert(0, str(_GRATING_DIR))

import grating_utils  # type: ignore  (path added dynamically above)
from gratingDecodeBehavior import load_behavior_data, align_units


# =============================================================================
# DATA LOADING
# =============================================================================

def load_grating_data_by_stim(grating_pkl_path, time_window, left_stim, right_stim):
    """
    Load grating data filtered to exactly the two (ori, SF) conditions
    specified by left_stim and right_stim.

    Parameters
    ----------
    left_stim  : dict with 'ori' and 'sf'  — mapped to class 1 (left-equiv)
    right_stim : dict with 'ori' and 'sf'  — mapped to class 0 (right-equiv)

    Returns
    -------
    firing_rates  : (n_trials, n_units)
    binary_labels : (n_trials,)  1=left_stim, 0=right_stim
    unit_ids      : list of str
    trial_info    : dict
    """
    data = grating_utils.load_neural_data(grating_pkl_path)

    unit_info = data.get('unit_info', {})
    all_unit_ids = list(data['spike_data'].keys())
    good_units = [u for u in all_unit_ids
                  if unit_info.get(u, {}).get('quality', 'unknown') != 'noise']
    n_noise = len(all_unit_ids) - len(good_units)
    if n_noise:
        print(f"  [Grating] Excluded {n_noise} noise unit(s)")
    data_filtered = dict(data)
    data_filtered['spike_data'] = {u: data['spike_data'][u] for u in good_units}

    firing_rates, ori_labels, unit_ids, trial_info = \
        grating_utils.calculate_firing_rates(data_filtered, time_window=time_window)

    ori_labels = ori_labels.astype(float)
    sf_labels  = trial_info['spatial_freq_labels'].astype(float)

    left_ori,  left_sf  = float(left_stim['ori']),  float(left_stim['sf'])
    right_ori, right_sf = float(right_stim['ori']), float(right_stim['sf'])

    left_mask  = (ori_labels == left_ori)  & (sf_labels == left_sf)
    right_mask = (ori_labels == right_ori) & (sf_labels == right_sf)
    keep_mask  = left_mask | right_mask

    if not keep_mask.any():
        raise ValueError(
            f"No trials found for left_stim={left_stim} or right_stim={right_stim}. "
            f"Available ori: {np.unique(ori_labels)}, SF: {np.unique(sf_labels)}"
        )

    firing_rates  = firing_rates[keep_mask]
    binary_labels = np.where(left_mask[keep_mask], 1, 0)

    n_left  = int(binary_labels.sum())
    n_right = int((binary_labels == 0).sum())
    print(f"\n[Grating] Filtered to target (ori, SF) conditions:")
    print(f"  Left  ori={left_ori}°  SF={left_sf}:  {n_left} trials")
    print(f"  Right ori={right_ori}°  SF={right_sf}: {n_right} trials")
    print(f"  Units: {len(unit_ids)}")

    trial_info['binary_labels'] = binary_labels
    trial_info['n_left']  = n_left
    trial_info['n_right'] = n_right

    return firing_rates, binary_labels, unit_ids, trial_info


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def build_mixed_dataset(grating_fr, grating_binary_labels,
                        behavior_fr, behavior_labels):
    """
    Stack grating and behavior firing rates with shared 2-class labels.

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
    Fit StandardScaler and binary 2-class LDA (1D) on the mixed dataset.

    Returns
    -------
    scaler, lda, X_lda (n, 1)
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_sc, y)   # (n, 1)

    print(f"\n[LDA] 2-class binary discriminant axis (left-equiv vs right-equiv)")
    return scaler, lda, X_lda


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
_SIZE   = {0: 25,  1: 40}     # behavior points slightly larger


def _build_label_map(left_stim, right_stim):
    """Return label dict keyed by (cls, src) based on stim dicts."""
    return {
        (1, 0): f"Grating ori={left_stim['ori']}°  SF={left_stim['sf']}",
        (1, 1): "Behavior Left",
        (0, 0): f"Grating ori={right_stim['ori']}°  SF={right_stim['sf']}",
        (0, 1): "Behavior Right",
    }


def _build_cls_name(left_stim, right_stim):
    """Return class name dict keyed by class int."""
    return {
        1: f"Left-equiv (ori={left_stim['ori']}° SF={left_stim['sf']})",
        0: f"Right-equiv (ori={right_stim['ori']}° SF={right_stim['sf']})",
    }


def _iter_groups(y, source):
    """Yield (cls, src, mask) for the 4 display groups."""
    for cls in [1, 0]:
        for src in [0, 1]:
            mask = (y == cls) & (source == src)
            if mask.any():
                yield cls, src, mask


def create_figure(X_lda, y, source,
                  cv_grating, cv_behavior, cv_mixed, cv_shuffle,
                  grating_fr, grating_labels,
                  behavior_fr, behavior_labels,
                  shared_unit_ids,
                  run_params,
                  behavior_left_stim, behavior_right_stim,
                  save_path=None):
    """
    Layout (2 rows × 3 cols):
      [0,0] LDA scatter (all trials)     [0,1] CV accuracy bars (3 bars)   [0,2] LDA scatter grating only
      [1,0] LDA scatter behavior only    [1,1] LDA 1D histogram             [1,2] Summary text
    """
    label_map = _build_label_map(behavior_left_stim, behavior_right_stim)
    cls_name  = _build_cls_name(behavior_left_stim, behavior_right_stim)

    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax_all  = fig.add_subplot(gs[0, 0])
    ax_cv   = fig.add_subplot(gs[0, 1])
    ax_grat = fig.add_subplot(gs[0, 2])
    ax_beh  = fig.add_subplot(gs[1, 0])
    ax_ld   = fig.add_subplot(gs[1, 1])
    ax_tx   = fig.add_subplot(gs[1, 2])

    _plot_lda_scatter(ax_all,  X_lda, y, source, label_map, title='LDA Embedding — All Trials')
    _plot_cv_bars(ax_cv, cv_grating, cv_behavior, cv_mixed, cv_shuffle)
    _plot_lda_scatter(ax_grat, X_lda, y, source, label_map,
                      title='LDA Embedding — Grating Only', show_source=0)
    _plot_lda_scatter(ax_beh,  X_lda, y, source, label_map,
                      title='LDA Embedding — Behavior Only', show_source=1)
    _plot_lda_1d(ax_ld, X_lda, y, source, cls_name)
    _plot_summary(ax_tx, cv_grating, cv_behavior, cv_mixed, cv_shuffle,
                  grating_labels, behavior_labels,
                  shared_unit_ids, run_params,
                  behavior_left_stim, behavior_right_stim)

    fig.suptitle(
        f"Grating + Behavior Mixed LDA Embedding\n"
        f"Left: ori={behavior_left_stim['ori']}° SF={behavior_left_stim['sf']}  "
        f"Right: ori={behavior_right_stim['ori']}° SF={behavior_right_stim['sf']}  "
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

def _plot_lda_scatter(ax, X_lda, y, source, label_map,
                      title='LDA Embedding', show_source=None):
    """
    Scatter plot in LD1 space.
    x = LD1 value, y = random jitter for visibility.
    color = class (left/right), marker = source (grating/behavior).
    show_source : if 0 or 1, only plot that source; None = plot all.
    """
    rng = np.random.default_rng(0)
    ld1 = X_lda[:, 0]

    legend_handles = []
    for cls, src, mask in _iter_groups(y, source):
        if show_source is not None and src != show_source:
            continue
        jitter = rng.uniform(-0.4, 0.4, mask.sum())
        h = ax.scatter(ld1[mask], jitter,
                       c=_COLOR[(cls, src)], marker=_MARKER[src],
                       alpha=0.65, s=_SIZE[src], edgecolors='none',
                       label=label_map[(cls, src)])
        legend_handles.append(h)

    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('LD1 (LDA score)', fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel('(jitter)', fontsize=8, color='gray')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')


def _plot_lda_1d(ax, X_lda, y, source, cls_name):
    """Overlay LDA 1D projections: grating (filled) vs behavior (outline)."""
    ld = X_lda[:, 0]
    colors_cls = {1: '#E74C3C', 0: '#3498DB'}

    for cls in [1, 0]:
        short = cls_name[cls].split('(')[1].rstrip(')')
        g_mask = (y == cls) & (source == 0)
        if g_mask.any():
            ax.hist(ld[g_mask], bins=20, alpha=0.55,
                    color=colors_cls[cls], density=True,
                    histtype='stepfilled', label=f'Grating {short}')
        b_mask = (y == cls) & (source == 1)
        if b_mask.any():
            ax.hist(ld[b_mask], bins=20, alpha=0.85,
                    color=colors_cls[cls], density=True,
                    histtype='step', linewidth=2, linestyle='--',
                    label=f'Behavior {short}')

    ax.axvline(0, color='black', linestyle=':', linewidth=1.5,
               label='Decision boundary')
    ax.set_xlabel('LDA Score (LD1)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('LDA 1D Projection\n(solid=grating  dashed=behavior)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_cv_bars(ax, cv_grating, cv_behavior, cv_mixed, cv_shuffle):
    """Four CV bars: grating-only, behavior-only, mixed, shuffled-behavior-mixed."""
    means = [cv_grating.mean(), cv_behavior.mean(), cv_mixed.mean(), cv_shuffle.mean()]
    stds  = [cv_grating.std(),  cv_behavior.std(),  cv_mixed.std(),  cv_shuffle.std()]
    bars = ax.bar([0, 1, 2, 3], means, yerr=stds, capsize=6,
                  color=['#2ECC71', '#F39C12', '#9B59B6', '#95A5A6'], alpha=0.85,
                  error_kw={'linewidth': 2})
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5,
               label='Chance (0.50)')
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Grating\nOnly', 'Behavior\nOnly',
                        'Mixed\n(grat+beh)', 'Mixed\n(shuffled beh FR)'], fontsize=9)
    ax.set_ylabel('CV Accuracy', fontsize=10)
    ax.set_title('CV Accuracy Comparison\n(2-class: left-equiv vs right-equiv)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')


def _plot_summary(ax, cv_grating, cv_behavior, cv_mixed, cv_shuffle,
                  grating_labels, behavior_labels,
                  shared_unit_ids, run_params,
                  behavior_left_stim, behavior_right_stim):
    ax.axis('off')
    g_win = run_params.get('grating_time_window', ('?', '?'))
    b_win = run_params.get('behavior_time_window', ('?', '?'))

    txt = (
        f"Mixed Embedding Summary\n"
        f"{'─'*36}\n"
        f"Label mapping:\n"
        f"  Grating ori={behavior_left_stim['ori']}° SF={behavior_left_stim['sf']}"
        f"  + Behavior Left  = class 1\n"
        f"  Grating ori={behavior_right_stim['ori']}° SF={behavior_right_stim['sf']}"
        f"  + Behavior Right = class 0\n\n"
        f"Shared units:  {len(shared_unit_ids)}\n\n"
        f"[Grating]\n"
        f"  Window:      {g_win[0]:.2f}–{g_win[1]:.2f} s\n"
        f"  Left trials:  {np.sum(grating_labels == 1)}\n"
        f"  Right trials: {np.sum(grating_labels == 0)}\n\n"
        f"[Behavior]\n"
        f"  Window:      {b_win[0]:.2f}–{b_win[1]:.2f} s\n"
        f"  Left trials: {np.sum(behavior_labels == 1)}\n"
        f"  Right trials:{np.sum(behavior_labels == 0)}\n\n"
        f"[CV — 2-class LDA]\n"
        f"  Grating only:     {cv_grating.mean():.3f} ± {cv_grating.std():.3f}\n"
        f"  Behavior only:    {cv_behavior.mean():.3f} ± {cv_behavior.std():.3f}\n"
        f"  Mixed:            {cv_mixed.mean():.3f} ± {cv_mixed.std():.3f}\n"
        f"  Mixed (shuffled): {cv_shuffle.mean():.3f} ± {cv_shuffle.std():.3f}\n"
        f"  Chance:           0.500\n"
    )
    ax.text(0.04, 0.98, txt, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_embedding(grating_pkl, behavior_pkl,
                  behavior_left_stim,
                  behavior_right_stim,
                  grating_time_window=(0.05, 1.5),
                  behavior_time_window=(0.05, 1.5),
                  save_plots=True,
                  output_path=None):
    """
    Full pipeline:
      1. Load grating filtered to the two (ori, SF) conditions + behavior
      2. Align to shared units
      3. Merge with 2-class labels
      4. Binary 2-class LDA embedding (LD1)
      5. CV accuracy: grating-only / mixed / shuffled-behavior-mixed
      6. Visualise

    Returns
    -------
    dict: scaler, lda, X_lda, y, source,
          cv_grating, cv_mixed, cv_shuffle, shared_unit_ids
    """
    print("=" * 60)
    print("Grating + Behavior Mixed LDA Embedding")
    print(f"  Left  → ori={behavior_left_stim['ori']}°  SF={behavior_left_stim['sf']}")
    print(f"  Right → ori={behavior_right_stim['ori']}°  SF={behavior_right_stim['sf']}")
    print("=" * 60)

    # 1. Load
    grating_fr, grating_labels, grating_unit_ids, _ = \
        load_grating_data_by_stim(grating_pkl, grating_time_window,
                                  behavior_left_stim, behavior_right_stim)

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

    # 4. Fit LDA (2-class binary)
    scaler, lda, X_lda = run_analysis(X, y)

    # 5. CV accuracies
    cv_grating  = cv_accuracy(grating_fr_sh, grating_labels,
                              'Grating Only (2-class)', 0.5)
    cv_behavior = cv_accuracy(behavior_fr_sh, behavior_labels,
                              'Behavior Only (2-class)', 0.5)
    cv_mixed    = cv_accuracy(X, y,
                              'Mixed (2-class)', 0.5)

    # Shuffled-behavior control: permute firing rates (not labels) so class
    # sizes match cv_mixed exactly, but neural identity is broken
    rng = np.random.default_rng(42)
    shuf_idx = rng.permutation(len(behavior_fr_sh))
    behavior_fr_shuf = behavior_fr_sh[shuf_idx]
    X_shuf, y_shuf, _ = build_mixed_dataset(
        grating_fr_sh, grating_labels,
        behavior_fr_shuf, behavior_labels
    )
    cv_shuffle = cv_accuracy(X_shuf, y_shuf,
                             'Mixed shuffled-behavior (2-class)', 0.5)

    # 6. Figure
    run_params = {
        'grating_pkl':          str(grating_pkl),
        'behavior_pkl':         str(behavior_pkl),
        'grating_time_window':  grating_time_window,
        'behavior_time_window': behavior_time_window,
    }

    if output_path is None and save_plots:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = (
            Path(behavior_pkl).parent / 'passive-behavior' /
            (Path(behavior_pkl).stem + f'.grating_behavior_embedding_{ts}.png')
        )

    create_figure(
        X_lda=X_lda, y=y, source=source,
        cv_grating=cv_grating, cv_behavior=cv_behavior, cv_mixed=cv_mixed, cv_shuffle=cv_shuffle,
        grating_fr=grating_fr_sh, grating_labels=grating_labels,
        behavior_fr=behavior_fr_sh, behavior_labels=behavior_labels,
        shared_unit_ids=shared_unit_ids,
        run_params=run_params,
        behavior_left_stim=behavior_left_stim,
        behavior_right_stim=behavior_right_stim,
        save_path=output_path if save_plots else None,
    )

    plt.show()

    return {
        'scaler':          scaler,
        'lda':             lda,
        'X_lda':           X_lda,
        'y':               y,
        'source':          source,
        'cv_grating':      cv_grating,
        'cv_behavior':     cv_behavior,
        'cv_mixed':        cv_mixed,
        'cv_shuffle':      cv_shuffle,
        'shared_unit_ids': shared_unit_ids,
    }


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import grating_config as cfg

    try:
        results = run_embedding(
            grating_pkl=cfg.GRATING_PKL,
            behavior_pkl=cfg.BEHAVIOR_PKL,
            behavior_left_stim=cfg.BEHAVIOR_LEFT_STIM,
            behavior_right_stim=cfg.BEHAVIOR_RIGHT_STIM,
            grating_time_window=cfg.GRATING_TIME_WINDOW,
            behavior_time_window=cfg.BEHAVIOR_TIME_WINDOW,
            save_plots=True,
        )

        print("\n" + "=" * 60)
        print("Embedding complete!")
        print("=" * 60)
        print(f"  Shared units:              {len(results['shared_unit_ids'])}")
        print(f"  Grating CV (2-class):      {results['cv_grating'].mean():.3f} "
              f"± {results['cv_grating'].std():.3f}")
        print(f"  Behavior CV (2-class):     {results['cv_behavior'].mean():.3f} "
              f"± {results['cv_behavior'].std():.3f}")
        print(f"  Mixed CV (2-class):        {results['cv_mixed'].mean():.3f} "
              f"± {results['cv_mixed'].std():.3f}")
        print(f"  Mixed CV (shuffled beh):   {results['cv_shuffle'].mean():.3f} "
              f"± {results['cv_shuffle'].std():.3f}")

    except FileNotFoundError as e:
        print(f"Error: Data file not found — {e}")
        print("Update paths in grating_config.py.")
    except Exception as e:
        print(f"Error: {e}")
        raise
