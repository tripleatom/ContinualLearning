r"""UP-state detection and EXCESS pairwise co-activation during NREM sleep.

The word "excess" is the whole point.  Inside an UP state every neuron fires
more, so raw co-firing rises for essentially every pair whether or not the two
neurons are functionally related.  Everything here is therefore built around
removing that shared drive before a pair's co-activation is reported.

Three coupling numbers are produced per pair, from the same bins:

``raw_corr``
    Pearson correlation of the two binned spike counts inside UP states, with no
    correction.  Kept deliberately, as the comparator that should NOT be used to
    test the hypothesis -- it is the quantity that rises for everyone.

``excess_z``  (primary)
    Observed coincidence count minus what a rate-preserving null predicts,
    expressed in null standard deviations.  The null shuffles each unit's binned
    counts WITHIN short blocks that never straddle an UP-state boundary, so each
    unit keeps its own firing rate, its per-UP-state excitability, and its slow
    within-UP time course; only co-fluctuation finer than the block is destroyed.
    This is the "shuffle spikes within each UP state" null, done at bin
    resolution so it can be run for every pair at once.

``resid_corr``
    Correlation of the residuals left after each unit's counts are predicted from
    UP-state identity x within-UP phase (a multiplicative marginal model).  This
    is the "model each neuron's firing with UP-state identity/phase and test
    residual pair coupling" route, and it is independent enough of the shuffle
    to serve as a robustness check rather than a restatement.

UP states
---------
Preferred source is the MUA OFF/ON detection (``sleep/MUA/detect_off_states.py``),
because it is derived from channel-level multi-unit activity and is therefore
independent of the sorted units whose co-firing is being measured -- using the
sorted population to both define UP states and measure coupling would build the
common drive into the definition.  When those files do not exist, the population
rate of the sorted units is used instead (mirroring reactivation/sleep/UPState.py)
and the circularity is reported rather than hidden.

Clocks
------
MUA ON windows are stored in whole-day acquisition samples; sleep spike pickles
are in seconds from their own block start.  The mapping is
``t_sleep = (daysample - sleep_start_sample) / fs`` and it is VERIFIED, not
assumed: ``check_up_alignment`` compares sorted-unit firing inside the mapped
windows against the gaps between them.  A correct mapping gives a large ratio
(UP states are where the spikes are); a ratio near 1 means the clocks disagree
and the run should stop rather than produce meaningless coupling.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from UPState import detect_up_down_from_rates  # noqa: E402  (repo-local module)


# --------------------------------------------------------------------------- #
# Loading spikes
# --------------------------------------------------------------------------- #

def load_sleep_spikes(sleep_pkl, unit_ids=None, verbose=True):
    """
    Read a ``sleep_spikes_*.pkl`` and return spike times per unit.

    Returns (spikes, meta) where spikes is {unit_id: np.ndarray of seconds from
    block start} and meta carries duration, sampling rate and the block's start
    sample in the sorter's space (needed to map MUA windows onto this clock).
    """
    sleep_pkl = Path(sleep_pkl)
    with sleep_pkl.open('rb') as f:
        data = pickle.load(f)

    spike_data = data['spike_data']
    window = data.get('window', {})
    meta = {
        'path': str(sleep_pkl),
        'sleep_id': data.get('metadata', {}).get('sleep_id'),
        'fs': float(data.get('metadata', {}).get('sampling_frequency', 30000.0)),
        'start_sample': window.get('sleep_start_sample'),
        'end_sample': window.get('sleep_end_sample'),
        'duration_sec': float(window.get('window_duration_sec', 0.0)),
    }

    wanted = set(spike_data) if unit_ids is None else set(unit_ids)
    spikes = {}
    for uid in sorted(spike_data):
        if uid not in wanted:
            continue
        times = np.asarray(spike_data[uid].get('spike_times_sec', []), dtype=float)
        spikes[uid] = np.sort(times[np.isfinite(times)])

    if meta['duration_sec'] <= 0 and spikes:
        meta['duration_sec'] = float(max(t[-1] if t.size else 0.0 for t in spikes.values()))

    if verbose:
        missing = wanted - set(spikes)
        print(f"Sleep spikes [{meta['sleep_id']}]: {len(spikes)} units, "
              f"{meta['duration_sec'] / 60:.1f} min")
        if missing:
            print(f"  {len(missing)} requested units absent from this sleep pkl")
    return spikes, meta


# --------------------------------------------------------------------------- #
# UP-state windows
# --------------------------------------------------------------------------- #

def _merge_windows(windows, min_gap_sec=0.0):
    """Sort and merge overlapping/adjacent [start, end) windows."""
    if len(windows) == 0:
        return np.zeros((0, 2))
    w = np.asarray(windows, dtype=float)
    w = w[np.argsort(w[:, 0])]
    merged = [w[0].copy()]
    for start, end in w[1:]:
        if start - merged[-1][1] <= min_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.asarray(merged)


def _intersect_windows(a, b):
    """Intersection of two sorted window sets."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 2))
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start = max(a[i, 0], b[j, 0])
        end = min(a[i, 1], b[j, 1])
        if end > start:
            out.append((start, end))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(out) if out else np.zeros((0, 2))


def up_windows_from_mua(windows_pkl, sleep_meta, min_shank_fraction=0.5,
                        min_duration_sec=0.05, max_duration_sec=2.0,
                        verbose=True):
    """
    Probe-wide UP windows from ``*_global_on_windows_*.pkl``.

    ``detect_off_states.py`` scores ON/OFF separately per shank.  A single set of
    windows is needed so that every pair is measured on the same time base, so
    the per-shank windows are combined: at each instant the fraction of scored
    shanks currently ON is computed, and an UP state is a stretch where that
    fraction reaches ``min_shank_fraction``.  With one shank this reduces to that
    shank's own ON windows.

    Window times are converted from whole-day samples to this sleep block's
    seconds; call ``check_up_alignment`` afterwards before trusting them.
    """
    windows_pkl = Path(windows_pkl)
    with windows_pkl.open('rb') as f:
        doc = pickle.load(f)

    rows = doc.get('rows', [])
    if not rows:
        raise ValueError(f"{windows_pkl.name} contains no ON windows.")
    fs = float(doc.get('sampling_frequency', sleep_meta.get('fs', 30000.0)))
    start_sample = sleep_meta.get('start_sample')
    if start_sample is None:
        raise ValueError("Sleep pkl has no sleep_start_sample; cannot map MUA windows.")

    per_shank = {}
    for row in rows:
        start = (float(row['on_start_daysample']) - float(start_sample)) / fs
        end = (float(row['on_end_daysample']) - float(start_sample)) / fs
        per_shank.setdefault(int(row['shank']), []).append((start, end))

    shanks = sorted(per_shank)
    n_shanks = len(shanks)
    need = max(1, int(np.ceil(min_shank_fraction * n_shanks)))

    # Sweep line over window edges: +1 when a shank turns ON, -1 when it ends.
    edges = []
    for shank in shanks:
        for start, end in _merge_windows(per_shank[shank]):
            edges.append((start, 1))
            edges.append((end, -1))
    edges.sort()

    combined, count, run_start = [], 0, None
    for time, delta in edges:
        was_enough = count >= need
        count += delta
        now_enough = count >= need
        if now_enough and not was_enough:
            run_start = time
        elif was_enough and not now_enough and run_start is not None:
            combined.append((run_start, time))
            run_start = None

    windows = np.asarray(combined) if combined else np.zeros((0, 2))
    windows = _clip_and_filter(windows, sleep_meta.get('duration_sec', np.inf),
                               min_duration_sec, max_duration_sec)

    if verbose:
        total = float(np.sum(windows[:, 1] - windows[:, 0])) if windows.size else 0.0
        print(f"UP windows from MUA ON states ({windows_pkl.name}):")
        print(f"  {n_shanks} shanks, threshold {need}/{n_shanks} simultaneously ON")
        print(f"  {len(windows)} windows, {total / 60:.1f} min "
              f"({100 * total / max(sleep_meta.get('duration_sec', 1), 1e-9):.1f}% of block), "
              f"median {1000 * np.median(windows[:, 1] - windows[:, 0]):.0f} ms"
              if len(windows) else "  no windows survived filtering")
    return windows


def _clip_and_filter(windows, duration_sec, min_duration_sec, max_duration_sec):
    if windows.size == 0:
        return windows
    windows = windows[(windows[:, 1] > 0) & (windows[:, 0] < duration_sec)]
    if windows.size == 0:
        return np.zeros((0, 2))
    windows[:, 0] = np.clip(windows[:, 0], 0, duration_sec)
    windows[:, 1] = np.clip(windows[:, 1], 0, duration_sec)
    lengths = windows[:, 1] - windows[:, 0]
    keep = (lengths >= min_duration_sec)
    if max_duration_sec is not None:
        keep &= (lengths <= max_duration_sec)
    return windows[keep]


def up_windows_from_population_rate(spikes, duration_sec, bin_sec=0.01,
                                    min_duration_sec=0.05, max_duration_sec=2.0,
                                    verbose=True, **detector_kwargs):
    """
    Fallback UP windows from the sorted population rate (mirrors UPState.py).

    Circularity warning: these windows are defined by the same spikes whose
    co-firing is later measured, so a pair's coupling can in principle be
    inflated by both units having helped define the UP state.  The block shuffle
    is applied inside the windows either way, which absorbs most of it, but the
    MUA-based windows remain the cleaner input.
    """
    n_bins = int(np.floor(duration_sec / bin_sec))
    if n_bins < 10:
        raise ValueError("Sleep block too short for UP/DOWN detection.")
    edges = np.arange(n_bins + 1) * bin_sec
    unit_ids = sorted(spikes)
    X = np.empty((n_bins, len(unit_ids)))
    for k, uid in enumerate(unit_ids):
        X[:, k] = np.histogram(spikes[uid], bins=edges)[0] / bin_sec
    centers = 0.5 * (edges[:-1] + edges[1:])

    updown = detect_up_down_from_rates(X, centers, bin_sec, **detector_kwargs)
    events = updown['events']['up']
    windows = np.array([[e['start_sec'] - 0.5 * bin_sec, e['end_sec'] + 0.5 * bin_sec]
                        for e in events]) if events else np.zeros((0, 2))
    windows = _clip_and_filter(windows, duration_sec, min_duration_sec, max_duration_sec)

    if verbose:
        total = float(np.sum(windows[:, 1] - windows[:, 0])) if windows.size else 0.0
        print(f"UP windows from sorted population rate (fallback, not MUA):")
        print(f"  {len(windows)} windows, {total / 60:.1f} min "
              f"({100 * total / max(duration_sec, 1e-9):.1f}% of block)")
        print("  NOTE: UP states defined by the same units whose coupling is measured.")
    return windows, updown


def load_nrem_windows(sleep_periods_pkl, epoch=None, verbose=True):
    """
    NREM windows (seconds from block start) from ``*_sleep_periods.pkl``.

    The scoring pickle stores bouts per epoch; the exact key layout has changed
    over time, so several shapes are accepted and anything unrecognized raises
    with what was actually found instead of guessing.
    """
    path = Path(sleep_periods_pkl)
    with path.open('rb') as f:
        doc = pickle.load(f)

    node = doc
    if epoch is not None and isinstance(doc, dict) and epoch in doc:
        node = doc[epoch]

    # 'bout_intervals_s' (score_nrem_epochs.py's actual key) is every scored NREM
    # bout after gap-bridging and the minimum-duration filter -- i.e. all of
    # scored NREM, which is what "restrict UP states to NREM" should mean.
    # 'consolidated_windows_s' is a stricter subset (a further, longer minimum
    # duration) meant for picking one solid nap, not for defining the NREM
    # universe, so it is tried only as a fallback when the full bout list is
    # unavailable. The non-'_s'-suffixed spellings are kept as a fallback too,
    # in case another scoring script uses them.
    for key in ('bout_intervals_s', 'nrem_windows', 'nrem_bouts', 'bouts',
                'nrem_periods', 'consolidated_windows_s', 'consolidated_windows'):
        if isinstance(node, dict) and key in node and node[key]:
            rows = node[key]
            break
    else:
        raise KeyError(f"No NREM window list in {path.name}; top-level keys: "
                       f"{list(node)[:12] if isinstance(node, dict) else type(node)}")

    if isinstance(rows, dict):
        rows = rows.get('nrem', rows)
    windows = np.asarray([[float(r[0]), float(r[1])] if not isinstance(r, dict)
                          else [float(r.get('start_sec', r.get('start'))),
                                float(r.get('end_sec', r.get('end')))]
                          for r in rows], dtype=float)
    windows = _merge_windows(windows)
    if verbose:
        total = float(np.sum(windows[:, 1] - windows[:, 0])) if windows.size else 0.0
        print(f"NREM windows ({path.name}): {len(windows)}, {total / 60:.1f} min")
    return windows


def check_up_alignment(spikes, windows, duration_sec, min_ratio=1.5, verbose=True,
                       universe_windows=None):
    """
    Verify that the UP windows really sit where the sorted spikes are.

    Returns (ratio, report).  ``ratio`` is the population firing rate inside the
    windows divided by the rate in the gaps between them.  UP states are by
    definition the high-activity phase, so a correct time base gives a clearly
    larger-than-1 ratio (typically 2-10x); a ratio near or below 1 means either
    the MUA windows were mapped onto the wrong clock, or "outside the windows"
    is dominated by something with its own high activity that the comparison
    should not include -- e.g. waking movement in an unrestricted presleep/
    postsleep block, where an animal is not asleep for most of the epoch.

    ``universe_windows`` restricts BOTH the "inside" and "outside" side of the
    ratio to a given time base (e.g. scored NREM) instead of the whole
    recording -- pass the NREM windows here (already used to restrict `windows`
    itself) so "outside the UP windows" means "DOWN time within NREM", not
    "any non-UP time including wake". Omit to use the whole ``duration_sec``.
    """
    if windows.size == 0:
        return np.nan, {'reason': 'no windows'}

    if universe_windows is not None and universe_windows.size:
        total_universe = float(np.sum(universe_windows[:, 1] - universe_windows[:, 0]))
        universe_label = 'scored NREM'
    else:
        universe_windows = np.array([[0.0, float(duration_sec)]])
        total_universe = float(duration_sec)
        universe_label = 'whole block'

    total_in = float(np.sum(windows[:, 1] - windows[:, 0]))
    total_out = total_universe - total_in
    if total_in <= 0 or total_out <= 0:
        return np.nan, {'reason': f'windows cover the whole {universe_label} span'}

    starts, ends = windows[:, 0], windows[:, 1]
    u_starts, u_ends = universe_windows[:, 0], universe_windows[:, 1]

    def _count_inside(times, w_starts, w_ends):
        if times.size == 0 or w_starts.size == 0:
            return 0
        idx = np.searchsorted(w_starts, times, side='right') - 1
        valid = idx >= 0
        return int(np.count_nonzero(valid & (times < w_ends[np.clip(idx, 0, None)])))

    n_in = 0
    n_universe = 0
    for times in spikes.values():
        n_universe += _count_inside(times, u_starts, u_ends)
        n_in += _count_inside(times, starts, ends)

    rate_in = n_in / total_in
    rate_out = (n_universe - n_in) / total_out
    ratio = rate_in / rate_out if rate_out > 0 else np.inf
    report = {'rate_in_up_hz': rate_in, 'rate_out_hz': rate_out, 'ratio': ratio,
              'spikes_in_up_fraction': n_in / max(n_universe, 1),
              'up_time_fraction': total_in / total_universe,
              'universe': universe_label, 'total_universe_sec': total_universe,
              'passed': bool(ratio >= min_ratio)}
    if verbose:
        print(f"UP-window alignment check (within {universe_label}): population rate "
              f"{rate_in:.2f} Hz inside vs {rate_out:.2f} Hz outside "
              f"(ratio {ratio:.2f}, need >= {min_ratio})")
        if not report['passed']:
            print("  FAILED -- either the windows do not sit on the spikes' clock "
                  "(check sleep_start_sample and the MUA epoch_start_sample), or "
                  "'outside' still includes non-NREM time with its own high activity.")
    return ratio, report


# --------------------------------------------------------------------------- #
# Binning inside UP states
# --------------------------------------------------------------------------- #

def bin_within_up(spikes, windows, bin_sec=0.025, block_bins=5, verbose=True):
    """
    Bin every unit inside the UP windows, keeping only whole shuffle blocks.

    Each UP window is tiled with ``bin_sec`` bins; the tail that cannot fill a
    complete block of ``block_bins`` bins is dropped so that the later shuffle
    can be done as a reshape and can never move a spike across an UP boundary.

    Returns dict with:
        counts     (n_units, n_bins) int32 spike counts
        unit_ids   ordered unit ids matching rows of counts
        up_id      (n_bins,) index of the UP window each bin belongs to
        phase      (n_bins,) position within the UP window, 0..1
        block_id   (n_bins,) shuffle-block index (contiguous, within one UP)
        n_blocks, block_bins, bin_sec, total_up_sec
    """
    unit_ids = sorted(spikes)
    if windows.size == 0:
        raise ValueError("No UP windows to bin.")

    edges_list, up_ids, phases, block_ids = [], [], [], []
    block_counter = 0
    for w, (start, end) in enumerate(windows):
        n_bins = int(np.floor((end - start) / bin_sec))
        n_bins -= n_bins % block_bins           # keep only complete blocks
        if n_bins < block_bins:
            continue
        edges = start + np.arange(n_bins + 1) * bin_sec
        edges_list.append(edges)
        up_ids.append(np.full(n_bins, w, dtype=np.int32))
        phases.append((np.arange(n_bins) + 0.5) / n_bins)
        n_blocks = n_bins // block_bins
        block_ids.append(np.repeat(np.arange(block_counter, block_counter + n_blocks),
                                   block_bins).astype(np.int32))
        block_counter += n_blocks

    if not edges_list:
        raise ValueError(f"No UP window is long enough for {block_bins} bins of "
                         f"{1000 * bin_sec:.0f} ms.")

    up_id = np.concatenate(up_ids)
    phase = np.concatenate(phases)
    block_id = np.concatenate(block_ids)
    n_bins_total = up_id.size

    counts = np.zeros((len(unit_ids), n_bins_total), dtype=np.int32)
    for k, uid in enumerate(unit_ids):
        times = spikes[uid]
        offset = 0
        for edges in edges_list:
            n = edges.size - 1
            lo = np.searchsorted(times, edges[0], side='left')
            hi = np.searchsorted(times, edges[-1], side='left')
            if hi > lo:
                counts[k, offset:offset + n] = np.histogram(times[lo:hi], bins=edges)[0]
            offset += n

    total_up_sec = n_bins_total * bin_sec
    if verbose:
        rates = counts.sum(axis=1) / total_up_sec
        print(f"Binned inside UP states: {len(unit_ids)} units x {n_bins_total} bins "
              f"of {1000 * bin_sec:.0f} ms = {total_up_sec / 60:.1f} min of UP time")
        print(f"  in-UP firing rate: median {np.median(rates):.2f} Hz "
              f"(range {rates.min():.2f}-{rates.max():.2f})")
    return {'counts': counts, 'unit_ids': unit_ids, 'up_id': up_id, 'phase': phase,
            'block_id': block_id, 'n_blocks': int(block_counter),
            'block_bins': int(block_bins), 'bin_sec': float(bin_sec),
            'total_up_sec': float(total_up_sec)}


# --------------------------------------------------------------------------- #
# Coupling estimators
# --------------------------------------------------------------------------- #

def _pair_correlation_matrix(X):
    """Full Pearson correlation matrix of rows of X, NaN-safe for flat rows."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum('ij,ij->i', Xc, Xc))
    with np.errstate(invalid='ignore', divide='ignore'):
        Xn = np.where(norm[:, None] > 0, Xc / norm[:, None], 0.0)
    C = Xn @ Xn.T
    flat = norm <= 0
    C[flat, :] = np.nan
    C[:, flat] = np.nan
    return C


def _residualize_up_phase(counts, up_id, phase, n_phase_bins=5):
    """
    Remove each unit's UP-state identity and within-UP phase profile.

    The prediction is the multiplicative marginal model
        pred(t) = mean_over_bins_of_that_UP * mean_over_bins_of_that_PHASE / grand_mean
    fitted per unit, which is what "UP-state identity/phase drive" means when the
    two factors are allowed to scale each other.  Residual = counts - pred.
    """
    n_units, n_bins = counts.shape
    phase_bin = np.clip((phase * n_phase_bins).astype(int), 0, n_phase_bins - 1)
    n_up = int(up_id.max()) + 1

    up_counts = np.bincount(up_id, minlength=n_up).astype(float)
    ph_counts = np.bincount(phase_bin, minlength=n_phase_bins).astype(float)

    X = counts.astype(np.float64)
    up_sum = np.zeros((n_units, n_up))
    ph_sum = np.zeros((n_units, n_phase_bins))
    for k in range(n_units):
        up_sum[k] = np.bincount(up_id, weights=X[k], minlength=n_up)
        ph_sum[k] = np.bincount(phase_bin, weights=X[k], minlength=n_phase_bins)

    grand = X.mean(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        up_mean = np.where(up_counts > 0, up_sum / np.maximum(up_counts, 1), 0.0)
        ph_mean = np.where(ph_counts > 0, ph_sum / np.maximum(ph_counts, 1), 0.0)
        pred = up_mean[:, up_id] * ph_mean[:, phase_bin] / np.where(grand > 0, grand, 1.0)
    return X - pred


def _block_shuffle(counts_blocks, rng):
    """Permute bins within each block, independently for every unit."""
    order = np.argsort(rng.random(counts_blocks.shape), axis=2)
    return np.take_along_axis(counts_blocks, order, axis=2)


def pair_coupling(binned, n_surrogates=200, n_phase_bins=5, seed=0,
                  dtype=np.float32, verbose=True):
    """
    Raw, excess (shuffle-corrected) and residual co-activation for every pair.

    The shuffle is applied to all units simultaneously and the resulting
    coincidence matrix is obtained with one matrix product per surrogate, so the
    cost is set by the number of surrogates rather than by the number of pairs.

    Returns dict of (n_units, n_units) matrices plus the pair-independent
    bookkeeping needed to interpret them.
    """
    counts = binned['counts']
    n_units, n_bins = counts.shape
    block_bins = binned['block_bins']
    n_blocks = binned['n_blocks']

    X = counts.astype(dtype)
    observed = X @ X.T

    blocks = counts.reshape(n_units, n_blocks, block_bins)
    rng = np.random.default_rng(seed)

    # Welford accumulation over surrogates: the full (S, n_units, n_units) stack
    # would be tens of GB for a few hundred units.
    mean = np.zeros((n_units, n_units), dtype=np.float64)
    m2 = np.zeros((n_units, n_units), dtype=np.float64)
    for s in range(1, int(n_surrogates) + 1):
        shuffled = _block_shuffle(blocks, rng)
        Xs = shuffled.reshape(n_units, n_bins).astype(dtype)
        Cs = (Xs @ Xs.T).astype(np.float64)
        delta = Cs - mean
        mean += delta / s
        m2 += delta * (Cs - mean)
        if verbose and (s % max(1, n_surrogates // 5) == 0):
            print(f"  surrogate {s}/{n_surrogates}")

    sd = np.sqrt(m2 / max(n_surrogates - 1, 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        excess_z = np.where(sd > 0, (observed - mean) / sd, np.nan)
    excess_rate = (observed - mean) / binned['total_up_sec']

    raw_corr = _pair_correlation_matrix(X.astype(np.float64))
    resid = _residualize_up_phase(counts, binned['up_id'], binned['phase'],
                                  n_phase_bins=n_phase_bins)
    resid_corr = _pair_correlation_matrix(resid)

    if verbose:
        iu = np.triu_indices(n_units, k=1)
        print(f"Coupling over {n_units * (n_units - 1) // 2} pairs "
              f"({n_surrogates} within-UP block shuffles, "
              f"block = {block_bins} x {1000 * binned['bin_sec']:.0f} ms):")
        print(f"  raw corr    median {np.nanmedian(raw_corr[iu]):+.4f}")
        print(f"  excess z    median {np.nanmedian(excess_z[iu]):+.3f}")
        print(f"  resid corr  median {np.nanmedian(resid_corr[iu]):+.4f}")

    return {'observed': observed, 'null_mean': mean, 'null_sd': sd,
            'excess_z': excess_z, 'excess_rate_hz': excess_rate,
            'raw_corr': raw_corr, 'resid_corr': resid_corr,
            'unit_ids': binned['unit_ids'], 'n_surrogates': int(n_surrogates),
            'total_up_sec': binned['total_up_sec'],
            'rate_hz': counts.sum(axis=1) / binned['total_up_sec']}
