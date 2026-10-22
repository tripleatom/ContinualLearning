r"""Jitter-corrected fine-timescale synchrony between unit pairs in NREM UP states.

The question each pair answers
------------------------------
"Are these two neurons' spikes more precisely aligned than expected from each
cell's own firing pattern during UP states?"

A raw cross-correlogram peak does not answer that: inside an UP state every
neuron fires more, and any shared change in firing probability (the UP onset,
a slow within-UP envelope) puts extra spikes near each other for every pair.
Interval jitter removes exactly that part.

The four steps
--------------
1. **Observed coincidences.**  Only spikes inside UP windows are kept.  For each
   pair, count spike pairs whose lag satisfies ``exclude <= |lag| <= w``
   (coincidence window ``w``, e.g. 5 or 10 ms).  Each UP window is placed on a
   private stretch of a concatenated "UP-time axis", separated from its
   neighbours by ``GAP_SEC`` of empty time, so no coincidence is ever counted
   across an UP-window boundary.

2. **Jitter surrogates.**  Each UP window is cut into consecutive jitter cells
   of width ``D`` starting at the UP onset (the last cell is shortened to end at
   the UP offset).  In a surrogate every spike is moved to a uniformly random
   time inside its own cell, independently for every spike of both neurons.
   Each neuron therefore keeps its exact spike count in every cell -- its
   rate, its per-UP-state excitability and its time course relative to UP
   onset at resolution ``D`` -- while alignment finer than ``D`` is destroyed.
   ``D`` should be much wider than ``w`` (e.g. w = 5 ms, D = 50 ms).

3. **Excess synchrony** = observed count - mean count over surrogates.
   Reported alongside it:

   ``excess_norm``  excess / sqrt(n_i * n_j): excess coincidences per
                    (geometric-mean) spike, comparable across pairs with
                    different rates and across blocks with different UP time.
                    This is the response used in the tuning test.
   ``excess_z``     excess / surrogate SD.
   ``p_jitter``     one-sided pair-level p: fraction of surrogates with a
                    count >= the observed one.

4. The tuning test itself (excess vs tuning similarity with rate and distance
   controls) lives in ``tc_stats.py`` and ``run_tuning_coupling.py``.

What D means
------------
Co-modulation slower than D is kept in the surrogates and so is NOT excess;
co-modulation faster than D (including genuinely precise spike alignment) IS
excess.  "Excess at D = 50 ms" therefore means "synchrony finer than ~50 ms",
which is why results are reported for several D rather than one.

Spike-sorting caveat
--------------------
Two units on nearby channels cannot both be detected within ~1 ms of each
other (template collisions), which carves a dip at 0 lag; a duplicated unit
does the opposite and gives a huge 0-lag peak.  ``exclude_ms`` drops the
central lags from every count, and ``near_zero`` (spike pairs within
``dup_ms``) lets the runner flag likely duplicates.
"""
from __future__ import annotations

import time

import numpy as np

#: Empty time inserted between consecutive UP windows on the UP-time axis.
#: Must exceed every lag that is ever counted (coincidence window and CCG span).
GAP_SEC = 1.0

#: Spike times sit on the acquisition sample grid, so many lags fall EXACTLY on
#: a bin edge or a window bound; float rounding would then put them on either
#: side at random, making alternate bins over- and under-full (a comb that the
#: continuous-time surrogates do not have).  Every lag comparison and binning
#: step adds this tolerance so exact-edge lags always land the same way.  It is
#: far below one sample (33 us at 30 kHz) and far above float error (~1e-12 s).
LAG_EPS = 1e-9

#: Acquisition rate; spike times are integer samples at this rate.
SAMPLE_RATE_HZ = 30000.0


# --------------------------------------------------------------------------- #
# UP-time axis and jitter cells
# --------------------------------------------------------------------------- #

def up_time_axis(spikes, windows, unit_ids=None, gap_sec=GAP_SEC):
    """
    Keep only spikes inside UP windows and place them on the UP-time axis.

    Window ``u`` occupies [offset_u, offset_u + duration_u) on the axis, with
    ``gap_sec`` of empty time before the next window, so spikes in different UP
    windows are always further apart than any lag that is counted.  Any sorted,
    non-overlapping window set works (e.g. NREM bouts, with a larger gap).

    Returns dict:
        t          (N,) sorted axis times, seconds
        unit       (N,) unit index (into unit_ids) of each spike
        win        (N,) UP-window index of each spike
        offset     (n_windows,) axis time at which each window starts
        duration   (n_windows,) window lengths, seconds
        unit_ids, n_spikes (per unit), total_up_sec
    """
    windows = np.asarray(windows, dtype=float)
    if windows.size == 0:
        raise ValueError("No UP windows.")
    starts, ends = windows[:, 0], windows[:, 1]
    if np.any(np.diff(starts) < 0) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("UP windows must be sorted and non-overlapping.")

    duration = ends - starts
    offset = np.concatenate([[0.0], np.cumsum(duration + gap_sec)[:-1]])

    unit_ids = sorted(spikes) if unit_ids is None else list(unit_ids)
    t_parts, u_parts, w_parts = [], [], []
    n_spikes = np.zeros(len(unit_ids), dtype=np.int64)
    for k, uid in enumerate(unit_ids):
        times = np.asarray(spikes.get(uid, []), dtype=float)
        idx = np.searchsorted(starts, times, side='right') - 1
        inside = (idx >= 0) & (times < ends[np.clip(idx, 0, None)])
        idx, times = idx[inside], times[inside]
        n_spikes[k] = times.size
        t_parts.append(times - starts[idx] + offset[idx])
        u_parts.append(np.full(times.size, k, dtype=np.int32))
        w_parts.append(idx.astype(np.int32))

    t = np.concatenate(t_parts) if t_parts else np.zeros(0)
    unit = np.concatenate(u_parts) if u_parts else np.zeros(0, dtype=np.int32)
    win = np.concatenate(w_parts) if w_parts else np.zeros(0, dtype=np.int32)
    order = np.argsort(t, kind='stable')
    return {'t': t[order], 'unit': unit[order], 'win': win[order],
            'offset': offset, 'duration': duration, 'unit_ids': unit_ids,
            'n_spikes': n_spikes, 'total_up_sec': float(duration.sum())}


def jitter_cells(axis, jitter_sec):
    """
    The jitter cell of every spike: consecutive cells of ``jitter_sec`` starting
    at each UP onset, the last one cut short at the UP offset.

    Returns (cell_start, cell_len, cell_id), each (N,), in axis time.
    ``cell_id`` is unique across the whole block (used by the self-test to
    confirm that surrogates keep every cell's count).
    """
    win = axis['win']
    rel = axis['t'] - axis['offset'][win]            # time since UP onset
    c = np.floor(rel / jitter_sec).astype(np.int64)
    cell_start = axis['offset'][win] + c * jitter_sec
    cell_end = axis['offset'][win] + np.minimum((c + 1) * jitter_sec,
                                                axis['duration'][win])
    n_cells = np.ceil(axis['duration'] / jitter_sec).astype(np.int64)
    first_cell = np.concatenate([[0], np.cumsum(n_cells)[:-1]])
    return cell_start, cell_end - cell_start, first_cell[win] + c


def jitter_times(t, cell_start, cell_len, rng, fs=SAMPLE_RATE_HZ):
    """Every spike redrawn uniformly in its cell, moved by a WHOLE number of
    samples, so surrogate lags sit on the same sample lattice as the observed
    ones.  (Continuous-time surrogates against sample-grid data give binned
    CCG ratios that are wrong wherever a bin holds a non-integer number of
    sample lags -- e.g. the first |lag| bin, where lag 0 is only half a
    sample wide.)  Returned in the input order, unsorted."""
    new = cell_start + rng.random(cell_start.size) * cell_len
    return t + np.round((new - t) * fs) / fs


def jitter_surrogate(t, unit, cell_start, cell_len, rng):
    """One interval-jitter surrogate of the spikes (t, unit).

    Returns (t, unit) sorted by the new times.
    """
    t = jitter_times(t, cell_start, cell_len, rng)
    order = np.argsort(t)
    return t[order], unit[order]


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #

def _forward_pairs(t, max_lag):
    """
    Every pair of spikes (a, b), a before b in the sorted array, with
    ``t[b] - t[a] <= max_lag``, yielded in batches of (a, b, lag).

    Batch d pairs each spike with the spike d places later.  A spike drops out
    as soon as its d-th successor is beyond ``max_lag`` (later successors are
    further still), so the total work is proportional to the number of pairs
    actually within range.
    """
    n = t.size
    active = np.arange(max(n - 1, 0))
    d = 1
    while active.size:
        active = active[active + d < n]
        partner = active + d
        lag = t[partner] - t[active]
        keep = lag <= max_lag + LAG_EPS
        active, partner, lag = active[keep], partner[keep], lag[keep]
        if active.size:
            yield active, partner, lag
        d += 1


def coincidence_counts(t, unit, n_units, lag_ranges):
    """
    Symmetric (n_units, n_units) coincidence-count matrices, one per lag range.

    ``lag_ranges`` is a list of (lo, hi) in seconds; a spike pair counts for a
    range when lo <= |lag| <= hi.  Same-unit pairs are zeroed.
    """
    lag_ranges = [(float(lo), float(hi)) for lo, hi in lag_ranges]
    max_lag = max(hi for _, hi in lag_ranges)
    acc = np.zeros((len(lag_ranges), n_units * n_units), dtype=np.int64)
    for a, b, lag in _forward_pairs(t, max_lag):
        key = unit[a].astype(np.int64) * n_units + unit[b]
        for r, (lo, hi) in enumerate(lag_ranges):
            m = (lag >= lo - LAG_EPS) & (lag <= hi + LAG_EPS)
            if m.any():
                acc[r] += np.bincount(key[m], minlength=n_units * n_units)
    out = acc.reshape(len(lag_ranges), n_units, n_units)
    out = out + out.transpose(0, 2, 1)
    idx = np.arange(n_units)
    out[:, idx, idx] = 0
    return out


def check_bin_width(bin_sec, fs=SAMPLE_RATE_HZ):
    """CCG bins must hold a whole number of sample lags; otherwise alternate
    bins hold n and n+1 lags and the CCG zig-zags by ~1/n."""
    samples = bin_sec * fs
    if abs(samples - round(samples)) > 1e-6:
        raise ValueError(f"CCG bin {bin_sec * 1e3:g} ms is {samples:g} samples at "
                         f"{fs:g} Hz; use a whole number of samples.")


def grouped_ccg(t, unit, group_matrix, n_groups, max_lag, bin_sec):
    """
    Folded cross-correlogram (counts vs |lag|) summed over the pairs of each
    group.  ``group_matrix[i, j]`` is the group of pair (i, j), or -1 to skip.

    Returns (n_groups, n_bins) counts; bin k covers [k, k+1) * bin_sec.
    """
    check_bin_width(bin_sec)
    n_bins = int(round(max_lag / bin_sec))
    acc = np.zeros(n_groups * n_bins, dtype=np.int64)
    for a, b, lag in _forward_pairs(t, max_lag):
        g = group_matrix[unit[a], unit[b]]
        k = ((lag + LAG_EPS) / bin_sec).astype(np.int64)
        m = (g >= 0) & (k < n_bins)          # bins are [k, k+1) * bin_sec
        if m.any():
            acc += np.bincount(g[m] * n_bins + k[m], minlength=n_groups * n_bins)
    return acc.reshape(n_groups, n_bins)


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #

def jitter_synchrony(spikes, windows, unit_ids=None, coinc_ms=(5.0, 10.0),
                     jitter_ms=(25.0, 50.0, 100.0), exclude_ms=1.0,
                     n_surrogates=500, dup_ms=0.5, seed=0, verbose=True):
    """
    Observed coincidences, jitter null and excess synchrony for every pair.

    Args:
        spikes: {unit_id: spike times (s)} on the same clock as ``windows``
        windows: (n, 2) sorted, non-overlapping UP windows (s)
        coinc_ms: coincidence windows w; lags exclude_ms <= |lag| <= w count
        jitter_ms: jitter cell widths D (each should be >> every w)
        exclude_ms: central lags excluded from every count (sorting collisions)
        n_surrogates: jitter surrogates per D
        dup_ms: pairs within this lag are counted separately (duplicate check)

    Returns dict with
        unit_ids, n_spikes, rate_hz, total_up_sec, n_up_windows
        observed   {w: (n, n) counts}
        near_zero  (n, n) spike pairs with |lag| <= dup_ms
        combos     {(w, D): {'jit_mean', 'jit_sd', 'excess', 'excess_norm',
                             'excess_z', 'p_jitter'}}  each (n, n)
        params
    """
    coinc_ms = tuple(float(w) for w in coinc_ms)
    jitter_ms = tuple(float(d) for d in jitter_ms)
    if max(coinc_ms) * 1e-3 >= GAP_SEC:
        raise ValueError("Coincidence window must be shorter than GAP_SEC.")
    for d in jitter_ms:
        if d < 2 * max(coinc_ms):
            print(f"  WARNING: jitter width {d:g} ms is not much wider than the "
                  f"coincidence window {max(coinc_ms):g} ms; the null will "
                  f"absorb part of the synchrony it is meant to expose.")

    axis = up_time_axis(spikes, windows, unit_ids)
    n = len(axis['unit_ids'])
    n_spikes = axis['n_spikes'].astype(float)
    ranges = [(exclude_ms * 1e-3, w * 1e-3) for w in coinc_ms]

    if verbose:
        print(f"Jitter synchrony: {n} units, {axis['t'].size} spikes in "
              f"{len(axis['duration'])} UP windows ({axis['total_up_sec']:.0f} s)")
        print(f"  coincidence |lag| in [{exclude_ms:g}, w] ms for w = "
              f"{', '.join(f'{w:g}' for w in coinc_ms)};  jitter D = "
              f"{', '.join(f'{d:g}' for d in jitter_ms)} ms;  "
              f"{n_surrogates} surrogates per D")

    counts = coincidence_counts(axis['t'], axis['unit'], n,
                                ranges + [(0.0, dup_ms * 1e-3)])
    observed = {w: counts[k].astype(float) for k, w in enumerate(coinc_ms)}
    near_zero = counts[-1].astype(float)

    geo = np.sqrt(np.outer(n_spikes, n_spikes))
    rng = np.random.default_rng(seed)
    combos = {}
    for d in jitter_ms:
        tic = time.time()
        cell_start, cell_len, _ = jitter_cells(axis, d * 1e-3)
        total = np.zeros((len(coinc_ms), n, n))
        total_sq = np.zeros((len(coinc_ms), n, n))
        at_least = np.zeros((len(coinc_ms), n, n))
        for s in range(int(n_surrogates)):
            ts, us = jitter_surrogate(axis['t'], axis['unit'], cell_start, cell_len, rng)
            c = coincidence_counts(ts, us, n, ranges).astype(float)
            total += c
            total_sq += c * c
            for k, w in enumerate(coinc_ms):
                at_least[k] += c[k] >= observed[w]
        for k, w in enumerate(coinc_ms):
            mean = total[k] / n_surrogates
            var = np.maximum(total_sq[k] / n_surrogates - mean ** 2, 0.0)
            sd = np.sqrt(var * n_surrogates / max(n_surrogates - 1, 1))
            excess = observed[w] - mean
            with np.errstate(invalid='ignore', divide='ignore'):
                combos[(w, d)] = {
                    'jit_mean': mean, 'jit_sd': sd, 'excess': excess,
                    'excess_norm': np.where(geo > 0, excess / geo, np.nan),
                    'excess_z': np.where(sd > 0, excess / sd, np.nan),
                    'p_jitter': (at_least[k] + 1) / (n_surrogates + 1),
                }
        if verbose:
            iu = np.triu_indices(n, k=1)
            parts = [f"w={w:g}: median excess/spike "
                     f"{np.nanmedian(combos[(w, d)]['excess_norm'][iu]):+.2e}, "
                     f"z {np.nanmedian(combos[(w, d)]['excess_z'][iu]):+.2f}"
                     for w in coinc_ms]
            print(f"  D={d:g} ms ({time.time() - tic:.0f} s): " + ";  ".join(parts))

    return {'unit_ids': axis['unit_ids'], 'n_spikes': axis['n_spikes'],
            'rate_hz': n_spikes / axis['total_up_sec'],
            'total_up_sec': axis['total_up_sec'],
            'n_up_windows': int(len(axis['duration'])),
            'observed': observed, 'near_zero': near_zero, 'combos': combos,
            'axis': axis,
            'params': {'coinc_ms': coinc_ms, 'jitter_ms': jitter_ms,
                       'exclude_ms': float(exclude_ms), 'dup_ms': float(dup_ms),
                       'n_surrogates': int(n_surrogates), 'seed': int(seed)}}


def jitter_corrected_ccg(axis, group_matrix, n_groups, jitter_ms, max_lag_ms=40.0,
                         bin_ms=1.0, n_surrogates=20, seed=0):
    """
    Group-summed folded CCG, observed and averaged over jitter surrogates.

    Returns (lag_centers_ms, observed, surrogate_mean), the last two
    (n_groups, n_bins).  observed - surrogate_mean is the jitter-corrected CCG.
    """
    max_lag, bin_sec = max_lag_ms * 1e-3, bin_ms * 1e-3
    obs = grouped_ccg(axis['t'], axis['unit'], group_matrix, n_groups, max_lag,
                      bin_sec).astype(float)
    cell_start, cell_len, _ = jitter_cells(axis, jitter_ms * 1e-3)
    rng = np.random.default_rng(seed + 1)
    sur = np.zeros_like(obs)
    for _ in range(int(n_surrogates)):
        ts, us = jitter_surrogate(axis['t'], axis['unit'], cell_start, cell_len, rng)
        sur += grouped_ccg(ts, us, group_matrix, n_groups, max_lag, bin_sec)
    centers = (np.arange(obs.shape[1]) + 0.5) * bin_ms
    return centers, obs, sur / max(int(n_surrogates), 1)
