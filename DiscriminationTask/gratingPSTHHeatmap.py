"""
Sorted PSTH Heatmaps: Behavior-Left+Stim vs Behavior-Right+Stim
================================================================

For each condition, trials are pooled from both sources:
  left  = behavior-left trials  +  grating (ori=left_stim_ori,  SF=left_stim_sf)
  right = behavior-right trials +  grating (ori=right_stim_ori, SF=right_stim_sf)

Produces side-by-side z-scored firing-rate heatmaps sorted by peak time
in the left condition (sort order applied to both panels).

Reads configuration from grating_config.py:
  GRATING_PKL, BEHAVIOR_PKL, BEHAVIOR_LEFT_STIM, BEHAVIOR_RIGHT_STIM,
  GRATING_TIME_WINDOW
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
_GRATING_DIR = _THIS_DIR.parent / 'rf_recon' / 'FreelyMovingProcessing' / 'Grating'
sys.path.insert(0, str(_GRATING_DIR))

import grating_utils  # type: ignore


# =============================================================================
# TRIAL INDEX HELPERS
# =============================================================================

def get_grating_trial_indices(grating_data, left_stim, right_stim):
    """
    Return (left_trial_indices, right_trial_indices) as int arrays.

    Reads orientations from trial_info and spatial_freq from each
    trial dict in spike_data (same approach as grating_utils).
    """
    ori_list = grating_data['trial_info']['orientations']
    n_trials = len(ori_list)

    first_uid = next(iter(grating_data['spike_data']))
    sf_per_trial = np.full(n_trials, np.nan)
    for t in grating_data['spike_data'][first_uid]:
        idx = int(t['trial_index'])
        sf  = t.get('spatial_freq', None)
        if sf is not None and 0 <= idx < n_trials:
            sf_per_trial[idx] = float(sf)

    ori_arr = np.array(ori_list, dtype=float)

    left_mask  = (ori_arr == float(left_stim['ori']))  & (sf_per_trial == float(left_stim['sf']))
    right_mask = (ori_arr == float(right_stim['ori'])) & (sf_per_trial == float(right_stim['sf']))
    return np.where(left_mask)[0], np.where(right_mask)[0]


def get_behavior_trial_indices(beh_data):
    """
    Return (left_trial_indices, right_trial_indices) as int arrays.

    Supports both 'white_on_left' and 'rewarded_on_left' condition keys.
    """
    tinfo = beh_data['trial_info']
    if 'white_on_left' in tinfo:
        flags = np.array(tinfo['white_on_left'], dtype=bool)
    elif 'rewarded_on_left' in tinfo:
        flags = np.array(tinfo['rewarded_on_left'], dtype=bool)
    else:
        raise KeyError("trial_info must contain 'white_on_left' or 'rewarded_on_left'")
    return np.where(flags)[0], np.where(~flags)[0]


# =============================================================================
# PSTH COMPUTATION
# =============================================================================

def compute_psth(spike_data, unit_ids, trial_indices, bin_edges):
    """
    Compute mean PSTH (Hz) for a set of trial indices.

    Parameters
    ----------
    spike_data    : dict  uid -> list[dict]  (each dict has 'trial_index', 'spike_times')
    unit_ids      : list[str]  — defines row order in output
    trial_indices : 1-D int array  — trial indices to include
    bin_edges     : (n_bins+1,) array

    Returns
    -------
    psth : (n_units, n_bins) float array, mean firing rate in Hz
    """
    n_bins    = len(bin_edges) - 1
    dt        = bin_edges[1] - bin_edges[0]
    psth      = np.zeros((len(unit_ids), n_bins))
    trial_set = set(trial_indices.tolist())

    for u_idx, uid in enumerate(unit_ids):
        counts = []
        for t in spike_data[uid]:
            if int(t['trial_index']) in trial_set:
                st   = np.array(t['spike_times'])
                hist, _ = np.histogram(st, bins=bin_edges)
                counts.append(hist)
        if counts:
            psth[u_idx] = np.mean(counts, axis=0) / dt  # convert to Hz

    return psth


# =============================================================================
# VISUALIZATION
# =============================================================================

def _zscore_and_sort(matrices, sort_ref_idx=0, sigma=1.0):
    """
    Shared normalisation used by both plot functions.

    Parameters
    ----------
    matrices    : list of (n_units, n_bins) arrays  — raw PSTHs
    sort_ref_idx: which matrix to use for determining sort order (peak bin)
    sigma       : Gaussian smoothing kernel (bins)

    Returns
    -------
    smoothed_sorted : list of z-scored, sorted arrays
    sort_idx        : argsort index (apply to unit_ids list if needed)
    vmax            : 97th-percentile absolute value (symmetric colour scale)
    """
    smoothed = [gaussian_filter1d(m, sigma=sigma, axis=1) for m in matrices]

    peak_bins = np.argmax(smoothed[sort_ref_idx], axis=1)
    sort_idx  = np.argsort(peak_bins)
    smoothed  = [m[sort_idx] for m in smoothed]

    # Z-score each unit relative to its mean/std across ALL conditions combined
    combined = np.hstack(smoothed)
    mu  = combined.mean(axis=1, keepdims=True)
    std = combined.std(axis=1,  keepdims=True)
    std[std == 0] = 1.0

    zscored = [(m - mu) / std for m in smoothed]
    vmax = np.percentile(np.abs(np.concatenate([z.ravel() for z in zscored])), 97)
    return zscored, sort_idx, vmax


def _draw_heatmap(ax, mat, bin_centers, cmap, title, vmax, n_units):
    """Draw one imshow heatmap panel and return the mappable."""
    extent = [bin_centers[0], bin_centers[-1], 0, n_units]
    im = ax.imshow(mat, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time from onset (s)', fontsize=10)
    ax.set_title(title, fontsize=10)
    return im


def plot_psth_heatmaps(psth_left, psth_right, bin_centers,
                       left_stim, right_stim,
                       n_grating_left, n_beh_left,
                       n_grating_right, n_beh_right,
                       sort_ref_idx=0,
                       sigma=1.0, save_path=None):
    """
    Pooled mode (1 × 2): grating + behavior trials are averaged into one
    heatmap per condition.

    Parameters
    ----------
    sort_ref_idx : int
        0 = sort by Left condition peak  (default)
        1 = sort by Right condition peak
    """
    _sort_labels = {0: 'Left', 1: 'Right'}

    (L_z, R_z), _, vmax = _zscore_and_sort(
        [psth_left, psth_right], sort_ref_idx=sort_ref_idx, sigma=sigma
    )
    n_units    = L_z.shape[0]
    sort_label = _sort_labels[sort_ref_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    panels = [
        (axes[0], L_z, 'Reds',
         f"Left-equiv:  Beh Left + Grating\n"
         f"ori={left_stim['ori']}°  SF={left_stim['sf']}\n"
         f"({n_grating_left} grating + {n_beh_left} behavior trials)"),
        (axes[1], R_z, 'Blues',
         f"Right-equiv: Beh Right + Grating\n"
         f"ori={right_stim['ori']}°  SF={right_stim['sf']}\n"
         f"({n_grating_right} grating + {n_beh_right} behavior trials)"),
    ]

    for ax, mat, cmap, title in panels:
        im = _draw_heatmap(ax, mat, bin_centers, cmap, title, vmax, n_units)
        plt.colorbar(im, ax=ax, label='Firing rate (z-score)', shrink=0.8)

    axes[0].set_ylabel(
        f'Neuron (sorted by {sort_label} peak time, n={n_units})', fontsize=10
    )
    fig.suptitle(
        f'Sorted PSTH Heatmaps — Left vs Right Conditions (pooled)  '
        f'[sort: {sort_label}]\n'
        f'Grating + behavior averaged; z-scored across both conditions',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    plt.show()
    return fig


def plot_psth_heatmaps_separate(psth_gl, psth_bl, psth_gr, psth_br,
                                bin_centers,
                                left_stim, right_stim,
                                n_grating_left, n_beh_left,
                                n_grating_right, n_beh_right,
                                sort_ref_idx=0,
                                sigma=1.0, save_path=None):
    """
    Separate mode (2 × 2): grating and behavior sessions shown independently.

    Layout
    ------
    [0,0] Grating  Left    [0,1] Grating  Right
    [1,0] Behavior Left    [1,1] Behavior Right

    Parameters
    ----------
    sort_ref_idx : int
        Which condition determines the neuron sort order (peak bin):
          0 = Grating Left   1 = Behavior Left
          2 = Grating Right  3 = Behavior Right
    """
    _sort_labels = {
        0: 'Grating Left',
        1: 'Behavior Left',
        2: 'Grating Right',
        3: 'Behavior Right',
    }

    (GL_z, BL_z, GR_z, BR_z), _, vmax = _zscore_and_sort(
        [psth_gl, psth_bl, psth_gr, psth_br], sort_ref_idx=sort_ref_idx, sigma=sigma
    )
    n_units = GL_z.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True, sharex=True)

    panels = [
        (axes[0, 0], GL_z, 'Reds',
         f"Grating — Left\n"
         f"ori={left_stim['ori']}°  SF={left_stim['sf']}\n"
         f"({n_grating_left} trials)"),
        (axes[0, 1], GR_z, 'Blues',
         f"Grating — Right\n"
         f"ori={right_stim['ori']}°  SF={right_stim['sf']}\n"
         f"({n_grating_right} trials)"),
        (axes[1, 0], BL_z, 'Reds',
         f"Behavior — Left\n({n_beh_left} trials)"),
        (axes[1, 1], BR_z, 'Blues',
         f"Behavior — Right\n({n_beh_right} trials)"),
    ]

    for ax, mat, cmap, title in panels:
        im = _draw_heatmap(ax, mat, bin_centers, cmap, title, vmax, n_units)
        plt.colorbar(im, ax=ax, label='z-score', shrink=0.8)

    sort_label = _sort_labels[sort_ref_idx]
    for ax in axes[:, 0]:
        ax.set_ylabel(
            f'Neuron (sorted by {sort_label} peak, n={n_units})', fontsize=9
        )
    for ax in axes[1, :]:
        ax.set_xlabel('Time from onset (s)', fontsize=10)
    for ax in axes[0, :]:
        ax.set_xlabel('')

    fig.suptitle(
        f'Sorted PSTH Heatmaps — Passive (Grating) vs Behavior  '
        f'[sort: {sort_label}]\n'
        f'Z-scored across all four conditions',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    plt.show()
    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_psth_heatmap(grating_pkl, behavior_pkl,
                     left_stim, right_stim,
                     time_window=(0.0, 1.5),
                     bin_size=0.05,
                     smooth_sigma=1.0,
                     separate=False,
                     save_plots=True,
                     output_path=None):
    """
    Full pipeline:
      1. Load raw spike data from grating and behavior pickles
      2. Filter noise units; find shared units
      3. Get trial indices for each condition from both sources
      4. Compute per-unit PSTHs per source (grating / behavior) and condition
      5. Plot heatmaps

    Parameters
    ----------
    separate : bool
        False (default) — pooled mode: grating + behavior averaged into one
                          heatmap per condition (1 × 2 layout).
        True            — separate mode: each source shown independently
                          (2 × 2 layout: rows = grating / behavior,
                                          cols = left / right condition).

    Returns
    -------
    dict with psth_gl, psth_bl, psth_gr, psth_br,
              psth_left, psth_right, bin_centers, shared_units
    """
    print("=" * 60)
    print("PSTH Heatmap: Behavior+Stim Left vs Right")
    print(f"  Left  → ori={left_stim['ori']}°  SF={left_stim['sf']}")
    print(f"  Right → ori={right_stim['ori']}°  SF={right_stim['sf']}")
    print("=" * 60)

    # --- Load raw pickle data ---
    grating_data = grating_utils.load_neural_data(grating_pkl)
    with open(behavior_pkl, 'rb') as f:
        beh_data = pickle.load(f)

    # --- Filter noise units ---
    def filter_noise(data):
        uinfo = data.get('unit_info', {})
        return [u for u in data['spike_data']
                if uinfo.get(u, {}).get('quality', 'unknown') != 'noise']

    grating_units = filter_noise(grating_data)
    beh_units     = filter_noise(beh_data)
    shared_units  = sorted(set(grating_units) & set(beh_units))

    print(f"\n[Alignment]  Grating units: {len(grating_units)} | "
          f"Behavior units: {len(beh_units)} | Shared: {len(shared_units)}")
    if not shared_units:
        raise ValueError("No shared units between grating and behavior data.")

    # --- Get trial indices per condition ---
    g_left, g_right = get_grating_trial_indices(grating_data, left_stim, right_stim)
    b_left, b_right = get_behavior_trial_indices(beh_data)

    print(f"\n[Trials]")
    print(f"  Grating  left  ({left_stim['ori']}° SF={left_stim['sf']}): "
          f"{len(g_left)} trials")
    print(f"  Grating  right ({right_stim['ori']}° SF={right_stim['sf']}): "
          f"{len(g_right)} trials")
    print(f"  Behavior left:  {len(b_left)} trials")
    print(f"  Behavior right: {len(b_right)} trials")

    # --- Time bins ---
    t_start, t_end = time_window
    bin_edges   = np.arange(t_start, t_end + bin_size, bin_size)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(f"\n[Bins]  {len(bin_edges)-1} bins × {bin_size*1000:.0f} ms  "
          f"({t_start:.2f} → {t_end:.2f} s)")

    # --- Compute PSTHs (shared units only) ---
    g_spk = {u: grating_data['spike_data'][u] for u in shared_units}
    b_spk = {u: beh_data['spike_data'][u]     for u in shared_units}

    print(f"\n[PSTH] Computing for {len(shared_units)} units ...")
    psth_gl = compute_psth(g_spk, shared_units, g_left,  bin_edges)
    psth_gr = compute_psth(g_spk, shared_units, g_right, bin_edges)
    psth_bl = compute_psth(b_spk, shared_units, b_left,  bin_edges)
    psth_br = compute_psth(b_spk, shared_units, b_right, bin_edges)

    # Weighted average by trial count
    n_gl, n_bl = len(g_left),  len(b_left)
    n_gr, n_br = len(g_right), len(b_right)
    denom_l = n_gl + n_bl or 1
    denom_r = n_gr + n_br or 1
    psth_left  = (psth_gl * n_gl + psth_bl * n_bl) / denom_l
    psth_right = (psth_gr * n_gr + psth_br * n_br) / denom_r

    # --- Base output path stem ---
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = Path(behavior_pkl).parent / 'passive-behavior'
    out_stem = Path(behavior_pkl).stem

    # --- Plot ---
    shared_kwargs = dict(
        bin_centers=bin_centers,
        left_stim=left_stim,   right_stim=right_stim,
        n_grating_left=n_gl,   n_beh_left=n_bl,
        n_grating_right=n_gr,  n_beh_right=n_br,
        sigma=smooth_sigma,
    )

    if separate:
        # 4 figures, each sorted by a different condition
        sort_names = {
            0: 'sort_grating_left',
            1: 'sort_behavior_left',
            2: 'sort_grating_right',
            3: 'sort_behavior_right',
        }
        for ref_idx, sort_tag in sort_names.items():
            sp = (out_dir / f'{out_stem}.psth_separate_{sort_tag}_{ts}.png'
                  if save_plots and output_path is None
                  else output_path)
            plot_psth_heatmaps_separate(
                psth_gl, psth_bl, psth_gr, psth_br,
                sort_ref_idx=ref_idx,
                save_path=sp,
                **shared_kwargs,
            )
    else:
        sort_names = {0: 'sort_left', 1: 'sort_right'}
        for ref_idx, sort_tag in sort_names.items():
            sp = (out_dir / f'{out_stem}.psth_pooled_{sort_tag}_{ts}.png'
                  if save_plots and output_path is None
                  else output_path)
            plot_psth_heatmaps(psth_left, psth_right,
                               sort_ref_idx=ref_idx,
                               save_path=sp, **shared_kwargs)

    return {
        'psth_gl':      psth_gl,
        'psth_bl':      psth_bl,
        'psth_gr':      psth_gr,
        'psth_br':      psth_br,
        'psth_left':    psth_left,
        'psth_right':   psth_right,
        'bin_centers':  bin_centers,
        'shared_units': shared_units,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import grating_config as cfg

    try:
        # Set separate=True for a 2×2 layout (grating vs behavior, side by side).
        # Set separate=False (default) for a 1×2 pooled layout.
        results = run_psth_heatmap(
            grating_pkl=cfg.GRATING_PKL,
            behavior_pkl=cfg.BEHAVIOR_PKL,
            left_stim=cfg.BEHAVIOR_LEFT_STIM,
            right_stim=cfg.BEHAVIOR_RIGHT_STIM,
            time_window=cfg.PSTH_TIME_WINDOW,   # independent from decoder window
            bin_size=cfg.PSTH_BIN_SIZE,
            smooth_sigma=1.0,
            separate=False,
            save_plots=True,
        )
        print("\n" + "=" * 60)
        print(f"Done — shared units: {len(results['shared_units'])}")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: Data file not found — {e}")
        print("Update paths in grating_config.py.")
    except Exception as e:
        print(f"Error: {e}")
        raise
