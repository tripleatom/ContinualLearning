"""
ITI speed matching for the stimulus decoders.

Stim-on and ITI epochs usually differ in running speed, so a "stim-on vs ITI"
decoder may be reading running-vs-stationary instead of the stimulus.  These helpers
build the control used by decoder_workbench.py --speed-match:

  match_iti_to_stimon     sub-sample ITI bins so their per-bin speed histogram
                          matches the combined stim-on histogram
  downsample_iti_uniform  the comparison arm: the same number of ITI bins, drawn
                          uniformly (speed ignored)
  iti_target_count        ITI target = size of the smaller stim-on class, so
                          balanced training trains on equal N per class

(Moved unchanged from the former compare_task_decoders_velocity_matched.py.)
"""

import numpy as np

import params as P


def _speed_edges_from_stim(s_stim, bin_width_cms, top_percentile):
    """Common bin edges used by both matched and unmatched paths."""
    top_edge = float(np.percentile(s_stim, top_percentile)) + bin_width_cms
    return np.arange(0.0, top_edge + bin_width_cms, bin_width_cms)


def match_iti_to_stimon(X, y, speeds, rng, n_target,
                        bin_width_cms=None, top_percentile=None):
    """
    Sub-sample ITI bins (y == 0) to exactly `n_target` samples whose per-bin
    speed histogram best matches the combined stim-on (y != 0) histogram.

    Water-fill allocation: each speed bin's quota is `n_target * p_stim[b]`,
    capped by the ITI samples available in that bin. Deficit from bins that
    run dry is redistributed across the still-supplied bins in proportion to
    the stim-on mass, iterated until n_target is reached or no ITI remains.
    If best-effort matching can't reach `n_target` (e.g. ITI is sparser than
    stim-on at some speeds), the remaining slots are filled uniformly from
    any leftover ITI samples so the training size matches the unmatched arm.

    Returns (X_matched, y_matched, kept_speeds, target_per_bin, edges).
    """
    bin_width_cms = P.speed_bin_cms if bin_width_cms is None else bin_width_cms
    top_percentile = P.speed_top_pct if top_percentile is None else top_percentile
    stim_mask = y != 0
    iti_mask = y == 0
    s_stim = speeds[stim_mask]
    s_iti = speeds[iti_mask]

    edges = _speed_edges_from_stim(s_stim, bin_width_cms, top_percentile)
    n_bins = len(edges) - 1

    iti_idx_all = np.where(iti_mask)[0]
    iti_bin_idx = np.digitize(s_iti, edges) - 1
    in_range = (iti_bin_idx >= 0) & (iti_bin_idx < n_bins)
    iti_idx_in = iti_idx_all[in_range]
    iti_bin_in = iti_bin_idx[in_range]

    n_stim, _ = np.histogram(s_stim, bins=edges)
    n_iti_in_bin = np.bincount(iti_bin_in, minlength=n_bins)

    target = np.zeros(n_bins, dtype=int)
    available = n_iti_in_bin.copy()
    weights = n_stim.astype(float)
    remaining = int(min(n_target, iti_idx_all.size))

    while remaining > 0 and available.sum() > 0:
        wsum = weights.sum()
        if wsum <= 0:
            break
        alloc = np.floor(remaining * weights / wsum).astype(int)
        alloc = np.minimum(alloc, available)
        if alloc.sum() == 0:
            # Round-up step so we don't stall on small remaining budgets.
            order = np.argsort(-weights * (available > 0))
            for b in order:
                if available[b] > 0:
                    alloc[b] = 1
                    break
        if alloc.sum() == 0:
            break
        target += alloc
        available -= alloc
        remaining -= int(alloc.sum())
        weights = np.where(available > 0, weights, 0.0)

    # Materialize the picks.
    keep_parts = []
    for b in range(n_bins):
        if target[b] == 0:
            continue
        candidates = iti_idx_in[iti_bin_in == b]
        keep_parts.append(rng.choice(candidates, size=int(target[b]), replace=False))
    keep_iti = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=int)

    # If we still have a deficit (in-range ITI exhausted or speed-out-of-range),
    # top up uniformly from any unused ITI so n_iti_kept == n_target.
    deficit = int(min(n_target, iti_idx_all.size)) - keep_iti.size
    if deficit > 0:
        leftover = np.setdiff1d(iti_idx_all, keep_iti, assume_unique=False)
        if leftover.size:
            take = min(deficit, leftover.size)
            keep_iti = np.concatenate([keep_iti,
                                       rng.choice(leftover, size=take, replace=False)])

    sel = np.sort(np.concatenate([np.where(stim_mask)[0], keep_iti]))
    return X[sel], y[sel], speeds[sel], target, edges


def downsample_iti_uniform(X, y, speeds, rng, n_target):
    """Uniformly random sub-sample of ITI to `n_target` (speed ignored)."""
    iti_idx = np.where(y == 0)[0]
    stim_idx = np.where(y != 0)[0]
    take = int(min(n_target, iti_idx.size))
    if take == iti_idx.size:
        kept = iti_idx
    else:
        kept = rng.choice(iti_idx, size=take, replace=False)
    sel = np.sort(np.concatenate([stim_idx, kept]))
    return X[sel], y[sel], speeds[sel]


def iti_target_count(y):
    """ITI target = min stim-on class count, so balanced training trains on equal N per class."""
    return int(min((y == 1).sum(), (y == -1).sum()))
