"""
Cue-preference analysis on a discrimination-task PKL produced by
DiscriminationTask/grating/readDIO_grating.py.

Pipeline:
  1. Build X[cell, trial, bin] with edges (-1, 0, 1, 2) s:
     bin 0 = baseline (-1..0), bins 1-2 = two 1-s response bins (0..1, 1..2).
     Counts are converted to firing rates (Hz) inside the test (here all bins
     are 1 s wide, so this is a no-op, but the code stays width-aware).
  2. Wilcoxon signed-rank (two-sided) per cell x cue x response-bin vs paired
     baseline; Bonferroni alpha = 0.01 / (n_cells * n_cues * 2).
  3. Preferred cue = (cue, bin) of largest mean delta-rate among significant.
  4. Select top-150 responsive cells by max delta-rate to preferred cue
     (max across the two response bins).
  5. Sort by (preferred_cue, -response).
  6. PSTH at fine bins (50 ms, sigma 75 ms) for plotting only; z-score per row.
  7. imshow with one column per cue; vertical lines at cue on/off.
"""

import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from params import task_spikes_file


def get_save_dir():
    """VisualResponse/ subfolder next to the sortout session pkl."""
    out = os.path.join(os.path.dirname(task_spikes_file), 'VisualResponse')
    os.makedirs(out, exist_ok=True)
    return out


# --- 1. Load + cue labels ----------------------------------------------------

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_cue_labels(data, mode='left_stim'):
    """
    Group trials into cues. Returns (cues, cue_names):
        cues[trial]    -- integer cue index
        cue_names[i]   -- readable label for cue i (e.g. "ori=45°, sf=0.04 cpd")

    Modes:
        'left_stim'             -> unique (left_ori, left_sf)        [DEFAULT]
        'right_stim'            -> unique (right_ori, right_sf)
        'stim_config'           -> unique full config: (L_ori, L_sf, R_ori, R_sf)
        'rewarded_stim'         -> unique (rewarded_ori, rewarded_sf)
        'non_rewarded_stim'     -> unique (non_rewarded_ori, non_rewarded_sf)
        'rewarded_orientation'  -> unique rewarded orientations
        'rewarded_on_left'      -> 2 cues: rewarded on right (0) / left (1)
        'orientation_pair'      -> unique (left_ori, right_ori) pairs
    """
    trials = data['trial_info']['all_trial_parameters']

    def _rew_sf(t):
        return t['left_spatial_freq'] if t['rewarded_on_left'] else t['right_spatial_freq']

    def _nonrew_sf(t):
        return t['right_spatial_freq'] if t['rewarded_on_left'] else t['left_spatial_freq']

    if mode == 'rewarded_on_left':
        cues = np.array([int(t['rewarded_on_left']) for t in trials])
        names = ['rewarded right', 'rewarded left']
        return cues, names

    if mode == 'left_stim':
        vals = [(t['left_orientation'], t['left_spatial_freq']) for t in trials]
        fmt = lambda v: f"ori={v[0]:g}°, sf={v[1]:g} cpd"
    elif mode == 'right_stim':
        vals = [(t['right_orientation'], t['right_spatial_freq']) for t in trials]
        fmt = lambda v: f"R: ori={v[0]:g}°, sf={v[1]:g} cpd"
    elif mode == 'stim_config':
        vals = [(t['left_orientation'], t['left_spatial_freq'],
                 t['right_orientation'], t['right_spatial_freq']) for t in trials]
        fmt = lambda v: (f"L: {v[0]:g}°/{v[1]:g}\n"
                         f"R: {v[2]:g}°/{v[3]:g}")
    elif mode == 'rewarded_stim':
        vals = [(t['rewarded_orientation'], _rew_sf(t)) for t in trials]
        fmt = lambda v: f"ori={v[0]:g}°, sf={v[1]:g} cpd"
    elif mode == 'non_rewarded_stim':
        vals = [(t['non_rewarded_orientation'], _nonrew_sf(t)) for t in trials]
        fmt = lambda v: f"ori={v[0]:g}°, sf={v[1]:g} cpd"
    elif mode == 'rewarded_orientation':
        vals = [(t['rewarded_orientation'],) for t in trials]
        fmt = lambda v: f"ori={v[0]:g}°"
    elif mode == 'orientation_pair':
        vals = [(t['left_orientation'], t['right_orientation']) for t in trials]
        fmt = lambda v: f"L={v[0]:g}°, R={v[1]:g}°"
    else:
        raise ValueError(f"Unknown cue mode: {mode}")

    # use tuple keys so sorting works for mixed numeric/None types
    keys = [tuple(v) for v in vals]
    uniq_keys = sorted(set(keys))
    key_to_idx = {k: i for i, k in enumerate(uniq_keys)}
    cues = np.array([key_to_idx[k] for k in keys], dtype=int)
    names = [fmt(k) for k in uniq_keys]
    return cues, names


def cue_is_rewarded(data, mode, n_cues, cues):
    """
    Boolean per cue: True if the cue's stimulus identity matches the rewarded
    grating. Returns None for modes where the concept doesn't apply.
    """
    trials = data['trial_info']['all_trial_parameters']

    def _rew_sf(t):
        return t['left_spatial_freq'] if t['rewarded_on_left'] else t['right_spatial_freq']

    # rewarded (ori, sf) — should be constant; take from first trial
    t0 = trials[0]
    rew_ori = t0['rewarded_orientation']
    rew_sf = _rew_sf(t0)

    out = np.zeros(n_cues, dtype=bool)
    if mode == 'left_stim':
        for ci in range(n_cues):
            tr = trials[int(np.where(cues == ci)[0][0])]
            out[ci] = (tr['left_orientation'] == rew_ori and
                       tr['left_spatial_freq'] == rew_sf)
    elif mode == 'right_stim':
        for ci in range(n_cues):
            tr = trials[int(np.where(cues == ci)[0][0])]
            out[ci] = (tr['right_orientation'] == rew_ori and
                       tr['right_spatial_freq'] == rew_sf)
    elif mode in ('rewarded_stim', 'rewarded_orientation'):
        out[:] = True
    elif mode == 'non_rewarded_stim':
        out[:] = False
    elif mode == 'rewarded_on_left':
        # both cues contain the rewarded grating, just on different sides
        return None
    elif mode == 'stim_config':
        return None  # both sides shown — can't color a single column
    elif mode == 'orientation_pair':
        return None
    return out


# --- 2. Spike-count tensor ---------------------------------------------------

def build_count_tensor(data, bin_edges=(-1.0, 0.0, 1.0, 2.0)):
    """X[cell, trial, bin] -- spike counts. spike_times are seconds rel trial start."""
    unit_ids = list(data['spike_data'].keys())
    n_cells = len(unit_ids)
    n_trials = data['metadata']['n_trials']
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(edges) - 1

    X = np.zeros((n_cells, n_trials, n_bins), dtype=np.int32)
    for c, uid in enumerate(unit_ids):
        for trial in data['spike_data'][uid]:
            t_idx = trial['trial_index']
            spikes = np.asarray(trial['spike_times'], dtype=float)
            X[c, t_idx, :], _ = np.histogram(spikes, bins=edges)
    return X, unit_ids, edges


# --- 3. Wilcoxon responsiveness test -----------------------------------------

def responsiveness_test(X, cues, bin_edges, baseline_bin=0, response_bins=(1, 2),
                        alpha=0.01):
    """
    Per (cell, cue, response_bin): paired two-sided Wilcoxon(resp vs baseline)
    over that cue's trials. Counts are converted to firing rates (Hz) so bins
    of different widths are comparable. Returns pvals, mean delta-rates,
    Bonferroni-significance mask, unique cues, and the corrected alpha.
    """
    n_cells = X.shape[0]
    unique_cues = np.unique(cues)
    n_cues = len(unique_cues)
    response_bins = tuple(response_bins)
    n_resp = len(response_bins)

    widths = np.diff(np.asarray(bin_edges, dtype=float))
    base_w = widths[baseline_bin]

    pvals = np.ones((n_cells, n_cues, n_resp))
    deltas = np.zeros((n_cells, n_cues, n_resp))

    for ci, cue in enumerate(unique_cues):
        mask = cues == cue
        if mask.sum() < 2:
            continue
        for c in range(n_cells):
            base = X[c, mask, baseline_bin].astype(float) / base_w  # Hz
            for bi, b in enumerate(response_bins):
                resp = X[c, mask, b].astype(float) / widths[b]      # Hz
                deltas[c, ci, bi] = resp.mean() - base.mean()
                diff = resp - base
                if not np.any(diff != 0):
                    continue  # leave p=1
                try:
                    _, p = stats.wilcoxon(resp, base, alternative='two-sided',
                                          zero_method='wilcox')
                    pvals[c, ci, bi] = p
                except ValueError:
                    pass

    bonf = alpha / (n_cells * n_cues * n_resp)
    sig = pvals < bonf
    return pvals, deltas, sig, unique_cues, bonf


# --- 4. Preferred cue --------------------------------------------------------

def assign_preference(deltas, sig):
    """
    For each cell, find the (cue, bin) with the largest |delta| among
    significant entries. Significance for inclusion, magnitude for ranking.
    Returns:
        pref[cell]   -- preferred cue index, or -1 if not responsive
        score[cell]  -- signed delta-rate at that (cue, bin); sign indicates
                        activation (+) vs suppression (-)
        sign[cell]   -- +1 / -1 / 0 (0 = not responsive)
    """
    n_cells, n_cues, n_bins = deltas.shape
    pref = np.full(n_cells, -1, dtype=int)
    score = np.zeros(n_cells)
    sign = np.zeros(n_cells, dtype=int)

    abs_masked = np.where(sig, np.abs(deltas), 0.0)   # [cell, cue, bin]
    cue_abs = abs_masked.max(axis=2)                  # [cell, cue]
    has_sig = sig.any(axis=(1, 2))

    for c in np.where(has_sig)[0]:
        best_cue = int(np.argmax(cue_abs[c]))
        bin_scores = np.where(sig[c, best_cue],
                              np.abs(deltas[c, best_cue]), -np.inf)
        best_bin = int(np.argmax(bin_scores))
        pref[c] = best_cue
        score[c] = deltas[c, best_cue, best_bin]      # signed
        sign[c] = 1 if score[c] > 0 else -1
    return pref, score, sign


# --- 5. Top-N + sort ---------------------------------------------------------

def select_and_sort(pref, score, sign, n_top=150):
    """
    Top-n_top responsive cells by |score|, then ordered:
      activated (sign=+1) first, then suppressed (sign=-1);
      within each block, by preferred cue, then by descending |score|.
    """
    responsive = np.where(pref >= 0)[0]
    if responsive.size == 0:
        return responsive
    top = responsive[np.argsort(-np.abs(score[responsive]))][:n_top]
    order = sorted(top, key=lambda c: (-sign[c], pref[c], -abs(score[c])))
    return np.asarray(order)


# --- 6. Fine-bin PSTH for plotting -------------------------------------------

def build_psth(data, cell_indices, unit_ids,
               t_start=-1.0, t_stop=2.0, bin_w=0.05, sigma_s=0.075):
    """
    psth[cell, trial, time_bin] in spikes/s, Gaussian-smoothed along time.
    Cells with no spike data in [t_start, t_stop) just stay zero.
    """
    edges = np.arange(t_start, t_stop + bin_w / 2, bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_trials = data['metadata']['n_trials']

    psth = np.zeros((len(cell_indices), n_trials, len(edges) - 1))
    for i, c in enumerate(cell_indices):
        uid = unit_ids[c]
        for trial in data['spike_data'][uid]:
            t_idx = trial['trial_index']
            spikes = np.asarray(trial['spike_times'], dtype=float)
            spikes = spikes[(spikes >= t_start) & (spikes < t_stop)]
            counts, _ = np.histogram(spikes, bins=edges)
            psth[i, t_idx, :] = counts / bin_w  # rate
    psth = gaussian_filter1d(psth, sigma=sigma_s / bin_w, axis=-1)
    return psth, centers


# --- 7. Heatmap --------------------------------------------------------------

def plot_heatmap(psth, cues, unique_cues, centers,
                 cue_on=0.0, cue_off=1.0, cue_names=None,
                 cue_is_rewarded=None,
                 row_pref=None, row_sign=None,
                 zscore='baseline', baseline_t=(-1.0, 0.0),
                 cmap='RdBu_r', save_path=None):
    """
    Stacked columns (one per cue) of mean PSTH across that cue's trials.

    zscore:
        'full'     -- per-cell mean/std across the concatenated time axis
                      (good for showing tuning shape)
        'baseline' -- per-cell mean/std computed only on baseline_t window
                      (preserves activation vs suppression sign in the image)

    row_pref / row_sign: 1-D arrays per row (in plot order). Cue-group dividers
        are drawn within each sign block; a thicker divider is drawn between
        the activated (+1) and suppressed (-1) blocks.
    """
    n_cells = psth.shape[0]
    n_cues = len(unique_cues)
    n_time = psth.shape[2]

    mean_psth = np.zeros((n_cells, n_cues, n_time))
    for ci, cue in enumerate(unique_cues):
        m = cues == cue
        if m.any():
            mean_psth[:, ci, :] = psth[:, m, :].mean(axis=1)

    if zscore == 'full':
        flat = mean_psth.reshape(n_cells, -1)
        mu = flat.mean(axis=1, keepdims=True)
        sd = flat.std(axis=1, keepdims=True) + 1e-9
        z = ((flat - mu) / sd).reshape(n_cells, n_cues, n_time)
    elif zscore == 'baseline':
        # baseline mean/std per cell across the baseline window, all cues pooled
        b_mask = (centers >= baseline_t[0]) & (centers < baseline_t[1])
        base = mean_psth[:, :, b_mask].reshape(n_cells, -1)  # [cell, n_cues*n_b]
        mu = base.mean(axis=1, keepdims=True)
        sd = base.std(axis=1, keepdims=True) + 1e-9
        z = (mean_psth - mu[:, None]) / sd[:, None]
    else:
        raise ValueError(f"Unknown zscore mode: {zscore}")

    vmax = np.nanpercentile(np.abs(z), 99)

    # use true bin edges for extent so the image fills [t_start, t_stop] cleanly
    bin_w = float(centers[1] - centers[0]) if len(centers) > 1 else 0.0
    t_lo = float(centers[0] - bin_w / 2)
    t_hi = float(centers[-1] + bin_w / 2)

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, axes = plt.subplots(1, n_cues, sharey=True,
                             figsize=(1.8 * n_cues + 1.6, 6.5),
                             gridspec_kw={'wspace': 0.06})
    if n_cues == 1:
        axes = [axes]

    # cue-group dividers (within each sign block); thicker sign-block divider
    pref_dividers = []
    if row_pref is not None:
        rp = np.asarray(row_pref)
        rs = np.asarray(row_sign) if row_sign is not None else np.ones_like(rp)
        change = np.where((rp[1:] != rp[:-1]) | (rs[1:] != rs[:-1]))[0] + 1
        pref_dividers = list(change)
    sign_divider = None
    if row_sign is not None:
        rs = np.asarray(row_sign)
        flip = np.where(rs[1:] != rs[:-1])[0]
        if flip.size:
            sign_divider = int(flip[0]) + 1

    # per-cue color palette (cue 0 = green, cue 1 = blue; black fallback)
    cue_palette = ['#1b7a3b', '#2166ac']
    def _cue_color(i):
        return cue_palette[i] if i < len(cue_palette) else 'black'

    extent = [t_lo, t_hi, n_cells, 0]
    for ci, ax in enumerate(axes):
        im = ax.imshow(z[:, ci, :], aspect='auto', cmap=cmap,
                       vmin=-vmax, vmax=vmax, extent=extent,
                       interpolation='nearest', rasterized=True)
        if cue_on is not None:
            ax.axvline(cue_on, color='k', lw=0.6, ls='--', alpha=0.7)
        if cue_off is not None:
            ax.axvline(cue_off, color='k', lw=0.6, ls='--', alpha=0.7)
        for d in pref_dividers:
            ax.axhline(d, color='k', lw=0.5, alpha=0.6)
        if sign_divider is not None:
            ax.axhline(sign_divider, color='k', lw=1.4)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(n_cells, 0)
        # integer-second ticks within data range
        tick_lo = int(np.ceil(t_lo))
        tick_hi = int(np.floor(t_hi))
        ax.set_xticks(np.arange(tick_lo, tick_hi + 1))
        ax.tick_params(length=3)
        # hide y-axis tick numbers; keep only the axis line
        ax.set_yticks([])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

        title = cue_names[ci] if cue_names is not None else f'cue={unique_cues[ci]}'
        title_color = _cue_color(ci)
        title_weight = 'bold' if (cue_is_rewarded is not None
                                  and cue_is_rewarded[ci]) else 'normal'
        ax.set_title(title, color=title_color, fontweight=title_weight,
                     fontsize=9, pad=4)
        if ci == 0:
            ax.set_ylabel('putative units grouped by preferred cue',
                          labelpad=45)

    # block labels on the left of the leftmost panel:
    #   outer column = "activated" / "suppressed"
    #   inner column = "prefer cue k" for each contiguous-pref sub-block
    if row_sign is not None and sign_divider is not None:
        ax0 = axes[0]
        y_act = sign_divider / 2.0
        y_supp = (sign_divider + n_cells) / 2.0
        ax0.text(-0.11, y_act, 'activated', transform=ax0.get_yaxis_transform(),
                 rotation=90, ha='right', va='center',
                 fontsize=11, fontweight='bold', color='#b2182b')
        ax0.text(-0.11, y_supp, 'suppressed', transform=ax0.get_yaxis_transform(),
                 rotation=90, ha='right', va='center',
                 fontsize=11, fontweight='bold', color='#2166ac')

        if row_pref is not None:
            rp = np.asarray(row_pref)
            # block boundaries = 0, every pref/sign change, n_cells
            bounds = [0] + list(pref_dividers) + [n_cells]
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                if hi <= lo:
                    continue
                cue_val = int(rp[lo])
                # positional cue index (0,1,...) matches column order
                cue_pos = int(np.where(np.asarray(unique_cues) == cue_val)[0][0])
                y_mid = (lo + hi) / 2.0
                ax0.text(-0.05, y_mid, f'prefer cue {cue_pos}',
                         transform=ax0.get_yaxis_transform(),
                         rotation=90, ha='right', va='center',
                         fontsize=9, color=_cue_color(cue_pos))

    # single shared x-axis label centered under the panel row
    fig.supxlabel('time from trial start (s)', fontsize=12, y=0.02)

    cbar_label = 'baseline-z firing rate' if zscore == 'baseline' else 'z-scored firing rate'
    cbar = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02,
                        fraction=0.025, aspect=25)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(length=3, width=0.8)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # also save a transparent vector svg alongside the png
        base, ext = os.path.splitext(save_path)
        if ext.lower() == '.png':
            fig.savefig(base + '.svg', bbox_inches='tight', transparent=True)
    return fig


# --- main --------------------------------------------------------------------

def main(cue_mode='left_stim', n_top=150, alpha=0.01, zscore='full'):
    data = load_data(task_spikes_file)
    cues, cue_names = get_cue_labels(data, mode=cue_mode)
    print(f"Loaded {data['metadata']['n_trials']} trials, "
          f"{len(data['spike_data'])} units")
    for ci, name in enumerate(cue_names):
        print(f"  cue {ci} ({name}): {(cues == ci).sum()} trials")

    # baseline: [-1, 0) s ; response bins: [0, 1) and [1, 2) s
    bin_edges = (-0.5, 0.0,0.5,1.0)
    X, unit_ids, _ = build_count_tensor(data, bin_edges=bin_edges)

    pvals, deltas, sig, unique_cues, bonf = responsiveness_test(
        X, cues, bin_edges=bin_edges,
        baseline_bin=0, response_bins=(1, 2), alpha=alpha,
    )

    pref, score, sign = assign_preference(deltas, sig)
    n_act = int(np.sum(sign > 0))
    n_supp = int(np.sum(sign < 0))
    # cells with both significantly-activated and significantly-suppressed entries
    has_pos = ((sig & (deltas > 0)).any(axis=(1, 2)))
    has_neg = ((sig & (deltas < 0)).any(axis=(1, 2)))
    n_mixed = int(np.sum(has_pos & has_neg))
    print(f"Bonferroni alpha = {bonf:.3e}")
    print(f"  activated:  {n_act} / {X.shape[0]}")
    print(f"  suppressed: {n_supp} / {X.shape[0]}")
    print(f"  mixed (sig + and -): {n_mixed} (assigned to whichever |delta| is larger)")

    order = select_and_sort(pref, score, sign, n_top=n_top)
    print(f"Selected top-{len(order)} responsive cells (by |delta|).")
    for s_label, s_val in [('activated', 1), ('suppressed', -1)]:
        sub = order[sign[order] == s_val]
        if sub.size == 0:
            continue
        print(f"  {s_label} ({sub.size}):")
        for cue_idx in unique_cues:
            n = int((pref[sub] == cue_idx).sum())
            if n:
                print(f"    preferred {cue_names[cue_idx]}: {n}")

    t_start, t_stop = -1.0, 2.0
    psth, centers = build_psth(
        data, order, unit_ids,
        t_start=t_start, t_stop=t_stop, bin_w=0.05, sigma_s=0.075,
    )

    is_rew_full = cue_is_rewarded(data, cue_mode, len(cue_names), cues)
    is_rew_plot = (None if is_rew_full is None
                   else [is_rew_full[c] for c in unique_cues])

    def _fmt_t(t):
        s = f"{t:g}".replace('-', 'm').replace('.', 'p')
        return s
    fname = f'cue_preference_psth_t{_fmt_t(t_start)}_to_{_fmt_t(t_stop)}s.png'

    titled_cue_names = [f'cue {i}: {cue_names[c]}'
                        for i, c in enumerate(unique_cues)]

    fig = plot_heatmap(
        psth, cues, unique_cues, centers,
        cue_on=0.0, cue_off=1.0,
        cue_names=titled_cue_names,
        cue_is_rewarded=is_rew_plot,
        row_pref=pref[order],
        row_sign=sign[order],
        zscore=zscore,
        save_path=os.path.join(get_save_dir(), fname),
    )
    plt.show()
    return order, pref, score, sign, psth, fig


if __name__ == '__main__':
    main()
