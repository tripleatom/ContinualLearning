"""
All-Orientations Grating + Behavior LDA Embedding
==================================================

Trains an LDA decoder on ALL grating (ori, SF) conditions,
then augments two user-specified grating conditions with behavior trials:
  behavior-left  → added to behavior_left_stim  class
  behavior-right → added to behavior_right_stim class

Label mapping (user-defined via behavior_left_stim / behavior_right_stim):
  Default example:
    behavior-left  → ori=0°, SF=0.04 cpd  (red/orange)
    behavior-right → ori=0°, SF=0.16 cpd  (blue/teal)
  Other (ori, SF) conditions used for training but NOT plotted.
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
_GRATING_DIR = _THIS_DIR.parent / 'passive_visual' / 'FreelyMovingProcessing' / 'Grating'
sys.path.insert(0, str(_GRATING_DIR))

import grating_utils
from gratingDecodeBehavior import load_behavior_data, align_units


# =============================================================================
# DATA LOADING
# =============================================================================

def _stim_key(ori, sf):
    """Canonical string label for a (ori, sf) grating condition."""
    return f"{float(ori):.1f}deg_sf{float(sf)}"


def load_all_grating_data(grating_pkl_path, time_window=(0.07, 0.16)):
    """
    Load grating data keeping ALL (ori, SF) conditions.

    Returns
    -------
    firing_rates : (n_trials, n_units)
    stim_labels  : (n_trials,) str  — composite 'Xdeg_sfY' per trial
    sf_labels    : (n_trials,) float
    ori_labels   : (n_trials,) float
    unit_ids     : list of str
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
    stim_labels = np.array([_stim_key(o, s) for o, s in zip(ori_labels, sf_labels)])

    unique_keys, counts = np.unique(stim_labels, return_counts=True)
    print(f"\n[Grating] All (ori, SF) conditions loaded:")
    for k, cnt in zip(unique_keys, counts):
        print(f"  {k}: {cnt} trials")
    print(f"  Total: {len(stim_labels)} trials  |  Units: {len(unit_ids)}")

    return firing_rates, stim_labels, sf_labels, ori_labels, unit_ids


# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================

def build_datasets(grating_fr, stim_labels, behavior_fr, behavior_labels,
                   behavior_left_stim, behavior_right_stim):
    """
    Build pre-mix and post-mix datasets.

    Pre-mix  : grating only, all (ori, SF) conditions
    Post-mix : grating + behavior merged into the two target stim classes

    Parameters
    ----------
    behavior_left_stim  : dict with 'ori' and 'sf' — grating class for behavior-left
    behavior_right_stim : dict with 'ori' and 'sf' — grating class for behavior-right

    Returns
    -------
    X_pre, y_pre        : grating-only dataset
    X_post, y_post      : mixed dataset (string labels)
    source_post         : (n_post,)  0=grating, 1=behavior
    left_key, right_key : str labels used for the two target classes
    """
    left_key  = _stim_key(behavior_left_stim['ori'],  behavior_left_stim['sf'])
    right_key = _stim_key(behavior_right_stim['ori'], behavior_right_stim['sf'])

    X_pre = grating_fr.copy()
    y_pre = stim_labels.copy()

    beh_mapped = np.where(behavior_labels == 1, left_key, right_key)
    X_post  = np.vstack([grating_fr, behavior_fr])
    y_post  = np.concatenate([stim_labels, beh_mapped])
    source_post = np.concatenate([np.zeros(len(grating_fr), dtype=int),
                                   np.ones(len(behavior_fr),  dtype=int)])

    print("\n[Dataset] Class sizes before / after mixing behavior:")
    for key in np.unique(y_post):
        n_pre  = np.sum(y_pre  == key)
        n_post = np.sum(y_post == key)
        tag = f"  (+{n_post - n_pre} behavior)" if n_post > n_pre else ""
        print(f"  {key}:  pre={n_pre:3d}  post={n_post:3d}{tag}")

    return X_pre, y_pre, X_post, y_post, source_post, left_key, right_key


# =============================================================================
# BALANCING
# =============================================================================

def balance_dataset(X, y, rng=None):
    """
    Subsample every class down to the size of the smallest class.

    Parameters
    ----------
    X   : (n_samples, n_features)
    y   : (n_samples,) class labels
    rng : np.random.Generator or None  (default: seed=0 for reproducibility)

    Returns
    -------
    X_bal : (n_balanced, n_features)
    y_bal : (n_balanced,)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    classes   = np.unique(y)
    min_n     = min(int(np.sum(y == c)) for c in classes)
    idx_keep  = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_keep.append(rng.choice(idx_c, size=min_n, replace=False))

    idx_keep = np.concatenate(idx_keep)
    idx_keep = np.sort(idx_keep)          # preserve original order within each class

    print(f"  [Balance] {len(y)} → {len(idx_keep)} samples  "
          f"({len(classes)} classes × {min_n} each)")
    return X[idx_keep], y[idx_keep]


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

_MARKER = {0: 'o', 1: '^'}
_SIZE   = {0: 25,  1: 40}

# Colors indexed by (stim_idx, src): stim_idx 0=left, 1=right; src 0=grating, 1=behavior
_COLOR_TABLE = {
    (0, 0): '#E74C3C',   # grating left-stim  → red
    (0, 1): '#FF8C69',   # behavior left       → salmon
    (1, 0): '#3498DB',   # grating right-stim  → blue
    (1, 1): '#1ABC9C',   # behavior right      → teal
}


def _build_plot_maps(left_key, right_key, behavior_left_stim, behavior_right_stim):
    """Return (color_map, label_map) keyed by (stim_key_str, src)."""
    color_map = {
        (left_key,  0): _COLOR_TABLE[(0, 0)],
        (left_key,  1): _COLOR_TABLE[(0, 1)],
        (right_key, 0): _COLOR_TABLE[(1, 0)],
        (right_key, 1): _COLOR_TABLE[(1, 1)],
    }
    label_map = {
        (left_key,  0): f"Grating ori={behavior_left_stim['ori']}° SF={behavior_left_stim['sf']}",
        (left_key,  1): f"Behavior Left",
        (right_key, 0): f"Grating ori={behavior_right_stim['ori']}° SF={behavior_right_stim['sf']}",
        (right_key, 1): f"Behavior Right",
    }
    return color_map, label_map


def _iter_plot_groups(y_plot, source_plot, left_key, right_key):
    for key in [left_key, right_key]:
        for src in [0, 1]:
            mask = (y_plot == key) & (source_plot == src)
            if mask.any():
                yield key, src, mask


def _scatter_3d(ax, X_plot, y_plot, source_plot, lda, color_map, label_map,
                left_key, right_key):
    n_comp = X_plot.shape[1]
    for key, src, mask in _iter_plot_groups(y_plot, source_plot, left_key, right_key):
        x0 = X_plot[mask, 0]
        x1 = X_plot[mask, 1] if n_comp > 1 else np.zeros(mask.sum())
        x2 = X_plot[mask, 2] if n_comp > 2 else np.zeros(mask.sum())
        ax.scatter(x0, x1, x2,
                   c=color_map[(key, src)], marker=_MARKER[src],
                   s=_SIZE[src], alpha=0.65, label=label_map[(key, src)],
                   edgecolors='none')
    ev = getattr(lda, 'explained_variance_ratio_', [0, 0, 0])
    ax.set_xlabel(f'LD1 ({ev[0]:.2f})' if len(ev) > 0 else 'LD1', fontsize=8)
    ax.set_ylabel(f'LD2 ({ev[1]:.2f})' if len(ev) > 1 else 'LD2', fontsize=8)
    ax.set_zlabel(f'LD3 ({ev[2]:.2f})' if len(ev) > 2 else 'LD3', fontsize=8)
    ax.set_title('LDA 3D Embedding\n(target stims + behavior only)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.view_init(elev=20, azim=45)


def _scatter_2d(ax, X_plot, y_plot, source_plot, dim_x, dim_y, lda,
                color_map, label_map, left_key, right_key):
    n_comp = X_plot.shape[1]
    if n_comp <= dim_y:
        ax.axis('off')
        ax.text(0.5, 0.5, f'LD{dim_y+1} not available',
                ha='center', va='center', transform=ax.transAxes)
        return
    for key, src, mask in _iter_plot_groups(y_plot, source_plot, left_key, right_key):
        ax.scatter(X_plot[mask, dim_x], X_plot[mask, dim_y],
                   c=color_map[(key, src)], marker=_MARKER[src],
                   s=_SIZE[src], alpha=0.65, label=label_map[(key, src)],
                   edgecolors='none')
    ev = getattr(lda, 'explained_variance_ratio_', [])
    def _ax_label(d):
        return f'LD{d+1} ({ev[d]:.2f})' if d < len(ev) else f'LD{d+1}'
    ax.set_xlabel(_ax_label(dim_x), fontsize=9)
    ax.set_ylabel(_ax_label(dim_y), fontsize=9)
    ax.set_title(f'LD{dim_x+1} vs LD{dim_y+1}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_cv_bars(ax, cv_pre, cv_post, cv_shuffle, chance_pre, chance_post):
    means = [cv_pre.mean(), cv_post.mean(), cv_shuffle.mean()]
    stds  = [cv_pre.std(),  cv_post.std(),  cv_shuffle.std()]
    bars = ax.bar([0, 1, 2], means, yerr=stds, capsize=6,
                  color=['#2ECC71', '#9B59B6', '#95A5A6'], alpha=0.85,
                  error_kw={'linewidth': 2})
    ax.axhline(chance_pre,  color='#27AE60', linestyle='--', linewidth=1.5,
               label=f'Chance pre ({chance_pre:.2f})')
    if abs(chance_post - chance_pre) > 0.001:
        ax.axhline(chance_post, color='#8E44AD', linestyle=':',  linewidth=1.5,
                   label=f'Chance post ({chance_post:.2f})')
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Grating Only\n(all conditions)',
                        f'Mixed\n(+behavior)',
                        f'Mixed\n(shuffled beh FR)'], fontsize=8)
    ax.set_ylabel('CV Accuracy', fontsize=10)
    ax.set_title('CV Accuracy: Before vs After Mix',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')


def _plot_summary(ax, cv_pre, cv_post, cv_shuffle, chance_pre,
                  stim_labels, behavior_labels,
                  shared_unit_ids, lda, run_params,
                  left_key, right_key):
    ax.axis('off')
    g_win = run_params.get('grating_time_window', ('?', '?'))
    b_win = run_params.get('behavior_time_window', ('?', '?'))
    ev = getattr(lda, 'explained_variance_ratio_', [])

    unique_stims, counts = np.unique(stim_labels, return_counts=True)
    stim_lines = "".join(f"  {k}: {c}\n" for k, c in zip(unique_stims, counts))

    txt = (
        f"All-Conditions Embedding Summary\n"
        f"{'─'*36}\n"
        f"Shared units:  {len(shared_unit_ids)}\n\n"
        f"[Grating — all (ori, SF) conditions]\n"
        f"  Window:  {g_win[0]:.2f}–{g_win[1]:.2f} s\n"
        + stim_lines +
        f"\n[Behavior mapping]\n"
        f"  Window:  {b_win[0]:.2f}–{b_win[1]:.2f} s\n"
        f"  Left  → {left_key}:  {np.sum(behavior_labels==1)}\n"
        f"  Right → {right_key}: {np.sum(behavior_labels==0)}\n\n"
        f"[CV — {len(unique_stims)}-class LDA]\n"
        f"  Pre-mix:        {cv_pre.mean():.3f} ± {cv_pre.std():.3f}\n"
        f"  Post-mix:       {cv_post.mean():.3f} ± {cv_post.std():.3f}\n"
        f"  Shuffled beh:   {cv_shuffle.mean():.3f} ± {cv_shuffle.std():.3f}\n"
        f"  Chance:         {chance_pre:.3f}\n\n"
        f"[LDA explained variance]\n"
        + "".join(f"  LD{i+1}: {v:.3f}\n" for i, v in enumerate(ev))
    )
    ax.text(0.04, 0.98, txt, transform=ax.transAxes, fontsize=8.2,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def create_figure(X_lda_plot, y_plot, source_plot,
                  cv_pre, cv_post, cv_shuffle, chance_pre, chance_post,
                  stim_labels, behavior_labels,
                  shared_unit_ids, lda,
                  run_params,
                  behavior_left_stim, behavior_right_stim,
                  left_key, right_key,
                  save_path=None):
    """
    Layout (2 rows × 3 cols):
      [0,0] 3D embedding     [0,1] CV bars   [0,2] LD1 vs LD2
      [1,0] LD1 vs LD3       [1,1] LD2 vs LD3  [1,2] Summary
    """
    color_map, label_map = _build_plot_maps(
        left_key, right_key, behavior_left_stim, behavior_right_stim
    )

    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax3d  = fig.add_subplot(gs[0, 0], projection='3d')
    ax_cv = fig.add_subplot(gs[0, 1])
    ax12  = fig.add_subplot(gs[0, 2])
    ax13  = fig.add_subplot(gs[1, 0])
    ax23  = fig.add_subplot(gs[1, 1])
    ax_tx = fig.add_subplot(gs[1, 2])

    _scatter_3d(ax3d, X_lda_plot, y_plot, source_plot, lda,
                color_map, label_map, left_key, right_key)
    _plot_cv_bars(ax_cv, cv_pre, cv_post, cv_shuffle, chance_pre, chance_post)
    _scatter_2d(ax12, X_lda_plot, y_plot, source_plot, 0, 1, lda,
                color_map, label_map, left_key, right_key)
    _scatter_2d(ax13, X_lda_plot, y_plot, source_plot, 0, 2, lda,
                color_map, label_map, left_key, right_key)
    _scatter_2d(ax23, X_lda_plot, y_plot, source_plot, 1, 2, lda,
                color_map, label_map, left_key, right_key)
    _plot_summary(ax_tx, cv_pre, cv_post, cv_shuffle, chance_pre,
                  stim_labels, behavior_labels,
                  shared_unit_ids, lda, run_params,
                  left_key, right_key)

    n_stims = len(np.unique(stim_labels))
    fig.suptitle(
        f"All-Conditions LDA Embedding  •  {n_stims} grating conditions\n"
        f"Behavior Left → {left_key}  •  Behavior Right → {right_key}  "
        f"•  (○=grating  ▲=behavior)",
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

def _apply_grating_filter(grating_fr, stim_labels, sf_labels, ori_labels, grating_filter):
    """
    Apply optional (ori, SF) filter to grating data.

    Parameters
    ----------
    grating_filter : dict or None
        {'ori': [0.0], 'sf': [0.04, 0.08, 0.16, 0.32]}
        Each key is optional; omit a key to keep all values of that dimension.
        Pass None to skip filtering entirely.

    Returns filtered (grating_fr, stim_labels, sf_labels, ori_labels).
    """
    if grating_filter is None:
        return grating_fr, stim_labels, sf_labels, ori_labels

    mask = np.ones(len(stim_labels), dtype=bool)
    if 'ori' in grating_filter:
        ori_keep = [float(o) for o in grating_filter['ori']]
        mask &= np.isin(ori_labels, ori_keep)
    if 'sf' in grating_filter:
        sf_keep = [float(s) for s in grating_filter['sf']]
        mask &= np.isin(sf_labels, sf_keep)

    n_before = len(stim_labels)
    n_after  = int(mask.sum())
    kept = np.unique(stim_labels[mask])
    print(f"\n[Grating filter] {n_before} → {n_after} trials  |  kept: {list(kept)}")

    return grating_fr[mask], stim_labels[mask], sf_labels[mask], ori_labels[mask]


def run_embedding(grating_pkl, behavior_pkl,
                  behavior_left_stim,
                  behavior_right_stim,
                  grating_time_window=(0.07, 0.16),
                  behavior_time_window=(0.0, 1.0),
                  grating_filter=None,
                  save_plots=True,
                  output_path=None):
    """
    Full pipeline:
      1. Load ALL grating (ori, SF) conditions + behavior
      2. Optionally filter grating to specific (ori, SF) subsets
      3. Align to shared units
      4. Build pre/post-mix datasets using user-defined stim mapping
      5. CV accuracy before and after mixing
      6. Fit LDA on post-mix data → 3D embedding
      7. Plot only the two target stim conditions + behavior

    Parameters
    ----------
    behavior_left_stim  : dict with 'ori' and 'sf'
        Grating condition that behavior-left trials map to.
    behavior_right_stim : dict with 'ori' and 'sf'
        Grating condition that behavior-right trials map to.
    grating_filter : dict or None
        Restrict grating training data, e.g.:
        {'ori': [0.0], 'sf': [0.04, 0.08, 0.16, 0.32]}
        Keys 'ori' and 'sf' are each optional.
    """
    print("=" * 60)
    print("All-Conditions Grating + Behavior LDA Embedding")
    print(f"  Left  → ori={behavior_left_stim['ori']}°  SF={behavior_left_stim['sf']}")
    print(f"  Right → ori={behavior_right_stim['ori']}°  SF={behavior_right_stim['sf']}")
    if grating_filter:
        print(f"  Grating filter: {grating_filter}")
    print("=" * 60)

    # 1. Load
    grating_fr, stim_labels, sf_labels, ori_labels, grating_unit_ids = \
        load_all_grating_data(grating_pkl, grating_time_window)

    # 2. Filter grating (optional)
    grating_fr, stim_labels, sf_labels, ori_labels = _apply_grating_filter(
        grating_fr, stim_labels, sf_labels, ori_labels, grating_filter
    )

    behavior_fr, behavior_labels, behavior_unit_ids, _ = \
        load_behavior_data(behavior_pkl, behavior_time_window)

    # 3. Align
    shared_unit_ids, grating_fr_sh, behavior_fr_sh = align_units(
        grating_unit_ids, behavior_unit_ids, grating_fr, behavior_fr
    )

    # 4. Build datasets
    X_pre, y_pre, X_post, y_post, source_post, left_key, right_key = build_datasets(
        grating_fr_sh, stim_labels,
        behavior_fr_sh, behavior_labels,
        behavior_left_stim, behavior_right_stim,
    )

    # 4b. Balance: subsample all classes to the smallest class size
    rng = np.random.default_rng(42)
    print("\n[Balance pre-mix]")
    X_pre_bal,  y_pre_bal  = balance_dataset(X_pre,  y_pre,  rng=np.random.default_rng(0))
    print("[Balance post-mix]")
    X_post_bal, y_post_bal = balance_dataset(X_post, y_post, rng=np.random.default_rng(0))

    # Keep source_post aligned with post-mix (needed for shuffled control + plotting)
    # Re-derive source for the balanced post-mix using the same label structure:
    # grating rows come before behavior rows in build_datasets, so source is recoverable
    # from y_post (behavior trials have labels from beh_mapped, grating from stim_labels)
    # However, we need the balanced indices to also filter source_post.
    # Simplest: re-run balance to get the kept indices.
    def _balance_indices(y, rng_seed=0):
        _rng   = np.random.default_rng(rng_seed)
        classes = np.unique(y)
        min_n   = min(int(np.sum(y == c)) for c in classes)
        idx_keep = np.concatenate([
            _rng.choice(np.where(y == c)[0], size=min_n, replace=False)
            for c in classes
        ])
        return np.sort(idx_keep)

    post_idx         = _balance_indices(y_post, rng_seed=0)
    source_post_bal  = source_post[post_idx]

    # 5. CV accuracy (on balanced data)
    print()
    cv_pre,  chance_pre  = cv_accuracy(X_pre_bal,  y_pre_bal,
                                       'Grating Only (filtered, balanced)')
    cv_post, chance_post = cv_accuracy(X_post_bal, y_post_bal,
                                       f'Mixed (+behavior in {left_key}/{right_key}, balanced)')

    # Shuffled-behavior control: permute firing rates (not labels) so class
    # sizes / label structure match cv_post exactly, but neural identity is broken
    shuf_idx = rng.permutation(len(behavior_fr_sh))
    behavior_fr_shuf = behavior_fr_sh[shuf_idx]
    _, _, X_shuf, y_shuf, _, _, _ = build_datasets(
        grating_fr_sh, stim_labels,
        behavior_fr_shuf, behavior_labels,
        behavior_left_stim, behavior_right_stim,
    )
    print("[Balance shuffled post-mix]")
    X_shuf_bal, y_shuf_bal = balance_dataset(X_shuf, y_shuf,
                                             rng=np.random.default_rng(0))
    cv_shuffle, _ = cv_accuracy(X_shuf_bal, y_shuf_bal,
                                f'Mixed (shuffled behavior FR in {left_key}/{right_key}, balanced)')

    # 6. Fit LDA on balanced post-mix for embedding
    scaler, lda, X_lda_post = fit_lda_embedding(X_post_bal, y_post_bal, n_components=3)

    # 7. Subset: only the two target stim conditions + behavior for plotting
    plot_mask = np.isin(y_post_bal, [left_key, right_key])
    X_lda_plot  = X_lda_post[plot_mask]
    y_plot      = y_post_bal[plot_mask]
    source_plot = source_post_bal[plot_mask]

    n_g_plot = np.sum(plot_mask & (source_post_bal == 0))
    n_b_plot = np.sum(plot_mask & (source_post_bal == 1))
    print(f"\n[Plot subset] {n_g_plot} grating + {n_b_plot} behavior points shown")

    # 8. Figure
    run_params = {
        'grating_pkl':          str(grating_pkl),
        'behavior_pkl':         str(behavior_pkl),
        'grating_time_window':  grating_time_window,
        'behavior_time_window': behavior_time_window,
    }

    if output_path is None and save_plots:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = (
            Path(behavior_pkl).parent / 'passive-behavior' /
            (Path(behavior_pkl).stem + f'.all_conditions_embedding_{ts}.png')
        )

    create_figure(
        X_lda_plot=X_lda_plot, y_plot=y_plot, source_plot=source_plot,
        cv_pre=cv_pre, cv_post=cv_post, cv_shuffle=cv_shuffle,
        chance_pre=chance_pre, chance_post=chance_post,
        stim_labels=y_pre_bal,
        behavior_labels=behavior_labels,
        shared_unit_ids=shared_unit_ids,
        lda=lda,
        run_params=run_params,
        behavior_left_stim=behavior_left_stim,
        behavior_right_stim=behavior_right_stim,
        left_key=left_key,
        right_key=right_key,
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
        'left_key':          left_key,
        'right_key':         right_key,
        'cv_pre':            cv_pre,
        'cv_post':           cv_post,
        'cv_shuffle':        cv_shuffle,
        'chance_pre':        chance_pre,
        'chance_post':       chance_post,
        'shared_unit_ids':   shared_unit_ids,
        'stim_labels':       stim_labels,
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
            grating_filter=cfg.GRATING_FILTER,
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
