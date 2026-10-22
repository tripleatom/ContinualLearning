r"""Descriptive co-firing measures for NREM sleep (no hypothesis tests).

Co-firing in NREM happens on three timescales with different causes:

    ~100 ms - 1 s   both units are in the same UP state
    ~10 - 100 ms    shared rises and falls of activity inside an UP state
    ~1 - 10 ms      precise spike alignment

The measures here are chosen to show which timescale carries the co-firing.

Time bases
----------
Spikes are placed on an "axis" (``tc_jitter.up_time_axis``): a set of sorted
windows (NREM bouts, or UP windows) laid end to end with empty gaps between
them, so a lag is never counted across two windows.

Baselines (surrogates) for the pooled cross-correlogram, from coarse to fine
----------------------------------------------------------------------------
Each spike is moved to a uniformly random time within a "cell":

    rate only       cell = the whole NREM bout        (independent firing)
    UP membership   cell = its own UP or DOWN period  (keeps how many spikes
                                                        each unit fires in each
                                                        UP state; nothing finer)
    50 ms jitter    cell = 50 ms pieces of its own    (also keeps the slow
                    UP or DOWN period, from its start  within-UP time course)

Differences between successive baselines split co-firing into
UP-membership, slow within-UP, and fine (< 50 ms) parts.

STTC (Cutts & Eglen 2014)
-------------------------
    STTC = 1/2 [ (P_A - T_B) / (1 - P_A T_B) + (P_B - T_A) / (1 - P_B T_A) ]

T_A is the fraction of analysed time lying within +-dt of an A spike, and
P_A the fraction of A's spikes that have a B spike within +-dt.  It is 1 for
identical trains, ~0 for independent ones, and depends little on firing rate.
Here it is computed inside UP windows only: T counts UP time only, and
partners in a different UP window never count.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse, signal

import tc_jitter as tj


# --------------------------------------------------------------------------- #
# Time bases and surrogates
# --------------------------------------------------------------------------- #

def nrem_segments(bouts, up_windows):
    """Partition every NREM bout into consecutive UP and non-UP segments.

    Returns (starts, ends, is_up), sorted.  ``up_windows`` must already lie
    inside the bouts (they are intersected with NREM upstream).
    """
    starts, ends, is_up = [], [], []
    up = np.asarray(up_windows, dtype=float).reshape(-1, 2)
    for b0, b1 in np.asarray(bouts, dtype=float):
        inside = up[(up[:, 0] >= b0) & (up[:, 1] <= b1)]
        t = b0
        for s, e in inside:
            if s > t:
                starts.append(t); ends.append(s); is_up.append(False)
            starts.append(s); ends.append(e); is_up.append(True)
            t = e
        if b1 > t:
            starts.append(t); ends.append(b1); is_up.append(False)
    return np.array(starts), np.array(ends), np.array(is_up, dtype=bool)


def real_times(axis, window_starts):
    """Axis times back to block-clock seconds."""
    return axis['t'] - axis['offset'][axis['win']] + np.asarray(window_starts)[axis['win']]


def segment_cells(axis, window_starts, seg_starts, seg_ends, width=None):
    """Jitter cell (axis start, length) of every spike: its segment, or
    ``width``-second pieces of it counted from the segment start."""
    real = real_times(axis, window_starts)
    k = np.clip(np.searchsorted(seg_starts, real, side='right') - 1, 0, len(seg_starts) - 1)
    s0, s1 = seg_starts[k], seg_ends[k]
    if width is None:
        cs, ce = s0, s1
    else:
        cs = s0 + np.floor((real - s0) / width) * width
        ce = np.minimum(cs + width, s1)
    shift = axis['offset'][axis['win']] - np.asarray(window_starts)[axis['win']]
    return cs + shift, ce - cs


def jitter_axis(axis, cell_start, cell_len, rng):
    """A surrogate axis: every spike uniform in its cell, whole-sample moves
    (see tc_jitter.jitter_times); window index kept."""
    t = tj.jitter_times(axis['t'], cell_start, cell_len, rng)
    order = np.argsort(t)
    return dict(axis, t=t[order], unit=axis['unit'][order], win=axis['win'][order])


def unit_times(axis, window_starts=None):
    """Per-unit sorted spike times: block clock if ``window_starts`` is given,
    otherwise axis time."""
    t = axis['t'] if window_starts is None else real_times(axis, window_starts)
    order = np.lexsort((t, axis['unit']))
    counts = np.bincount(axis['unit'], minlength=len(axis['unit_ids']))
    return np.split(t[order], np.cumsum(counts)[:-1])


# --------------------------------------------------------------------------- #
# Pooled cross-correlogram
# --------------------------------------------------------------------------- #

def pooled_ccg(axis, shank, fine_max=0.1, fine_bin=0.0002, broad_max=1.0,
               broad_bin=0.01):
    """
    Cross-correlogram summed over all pairs of DIFFERENT units, split into
    same-shank (row 0) and cross-shank (row 1) pairs.

    fine:  counts vs |lag|, bins of ``fine_bin`` up to ``fine_max`` (exact
           spike pairs)
    broad: counts vs lag, -broad_max..+broad_max in bins of ``broad_bin``
           (from binned trains via FFT; autocorrelations subtracted)
    """
    t, unit = axis['t'], axis['unit']
    shank = np.asarray(shank)
    n_units = len(axis['unit_ids'])

    tj.check_bin_width(fine_bin)
    tj.check_bin_width(broad_bin)
    nb = int(round(fine_max / fine_bin))
    fine = np.zeros(2 * nb, dtype=np.int64)
    for a, b, lag in tj._forward_pairs(t, fine_max):
        ua, ub = unit[a], unit[b]
        keep = ua != ub
        if not keep.any():
            continue
        kind = (shank[ua] != shank[ub]).astype(np.int64)
        k = ((lag + tj.LAG_EPS) / fine_bin).astype(np.int64)
        keep &= k < nb                       # bins are [k, k+1) * fine_bin
        fine += np.bincount((kind * nb + k)[keep], minlength=2 * nb)
    fine = fine.reshape(2, nb).astype(float)

    L = int(round(broad_max / broad_bin))
    n_bins = int(np.ceil(t.max() / broad_bin)) + 1 if t.size else 1
    nfft = 1 << int(np.ceil(np.log2(n_bins + L + 1)))
    idx = (t / broad_bin).astype(np.int64)
    order = np.argsort(unit, kind='stable')
    split = np.split(idx[order], np.cumsum(np.bincount(unit, minlength=n_units))[:-1])
    autos = np.zeros(nfft // 2 + 1)
    by_shank = {}
    for u, ix in enumerate(split):
        x = np.bincount(ix, minlength=n_bins).astype(float)
        autos += np.abs(np.fft.rfft(x, nfft)) ** 2
        by_shank[shank[u]] = by_shank.get(shank[u], 0) + x
    shank_power = sum(np.abs(np.fft.rfft(x, nfft)) ** 2 for x in by_shank.values())
    total_power = np.abs(np.fft.rfft(sum(by_shank.values()), nfft)) ** 2

    def lags(power):
        cc = np.fft.irfft(power, nfft)[:L + 1]
        return np.concatenate([cc[:0:-1], cc])

    broad = np.vstack([lags(shank_power - autos), lags(total_power - shank_power)])
    return {'fine_lag': (np.arange(nb) + 0.5) * fine_bin, 'fine': fine,
            'broad_lag': np.arange(-L, L + 1) * broad_bin, 'broad': broad}


# --------------------------------------------------------------------------- #
# Binned counts: spike-count correlation and population coupling
# --------------------------------------------------------------------------- #

def binned_counts(spike_lists, windows, width):
    """Sparse (n_units, n_bins) counts in ``width`` bins tiled inside each
    window from its start; incomplete bins at window ends are dropped."""
    windows = np.asarray(windows, dtype=float).reshape(-1, 2)
    ws, we = windows[:, 0], windows[:, 1]
    nb = np.floor((we - ws) / width + 1e-9).astype(np.int64)
    first = np.concatenate([[0], np.cumsum(nb)[:-1]])
    n_bins = int(nb.sum())
    rows, cols = [], []
    for i, t in enumerate(spike_lists):
        w = np.searchsorted(ws, t, side='right') - 1
        ok = w >= 0
        w = np.where(ok, w, 0)
        k = np.floor((t - ws[w]) / width).astype(np.int64)
        ok &= (k >= 0) & (k < nb[w])
        cols.append(first[w[ok]] + k[ok])
        rows.append(np.full(int(ok.sum()), i))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    X = sparse.csr_matrix((np.ones(rows.size), (rows, cols)),
                          shape=(len(spike_lists), max(n_bins, 1)))
    X.sum_duplicates()
    return X, n_bins


def count_covariance(X, n_bins):
    """Covariance matrix of binned counts (population, divides by n_bins)."""
    m = np.asarray(X.sum(axis=1)).ravel() / n_bins
    return (X @ X.T).toarray() / n_bins - np.outer(m, m)


def correlation_from_cov(C):
    sd = np.sqrt(np.clip(np.diag(C), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        R = C / np.outer(sd, sd)
    R[sd == 0, :] = np.nan
    R[:, sd == 0] = np.nan
    np.fill_diagonal(R, np.nan)
    return R


def population_coupling(C):
    """Correlation of each unit's counts with the summed counts of all others."""
    row, d = C.sum(axis=1), np.diag(C)
    rest_var = C.sum() - 2 * row + d
    with np.errstate(invalid='ignore', divide='ignore'):
        return (row - d) / np.sqrt(d * rest_var)


# --------------------------------------------------------------------------- #
# STTC
# --------------------------------------------------------------------------- #

def sttc(axis, dts):
    """
    STTC for every pair at each ``dt`` (seconds), inside the axis windows.

    Returns (len(dts), n, n) symmetric array (NaN on the diagonal and for
    units without spikes).  The axis gap must exceed max(dts).
    """
    dts = np.asarray(dts, dtype=float)
    n = len(axis['unit_ids'])
    total = float(axis['duration'].sum())
    w_start, w_end = axis['offset'], axis['offset'] + axis['duration']

    order = np.lexsort((axis['t'], axis['unit']))
    counts = np.bincount(axis['unit'], minlength=n)
    edges = np.cumsum(counts)[:-1]
    t_units = np.split(axis['t'][order], edges)
    w_units = np.split(axis['win'][order], edges)

    # T: fraction of analysed time within +-dt of the unit's spikes
    T = np.full((dts.size, n), np.nan)
    for i, (t, w) in enumerate(zip(t_units, w_units)):
        if t.size == 0:
            continue
        for k, dt in enumerate(dts):
            s = np.maximum(t - dt, w_start[w])
            e = np.minimum(t + dt, w_end[w])
            prev = np.concatenate([[-np.inf], np.maximum.accumulate(e)[:-1]])
            T[k, i] = np.sum(np.maximum(0.0, e - np.maximum(s, prev))) / total

    # P[k, a, b]: fraction of a's spikes with a b spike within +-dt
    P = np.full((dts.size, n, n), np.nan)
    for a, ta in enumerate(t_units):
        if ta.size == 0:
            continue
        lo = ta[:, None] - dts[None, :] - tj.LAG_EPS
        hi = ta[:, None] + dts[None, :] + tj.LAG_EPS
        for b, tb in enumerate(t_units):
            if b == a or tb.size == 0:
                continue
            has = np.searchsorted(tb, hi, side='right') > np.searchsorted(tb, lo, side='left')
            P[:, a, b] = has.mean(axis=0)

    Tb = T[:, None, :]          # T of the second index
    Ta = T[:, :, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        term_ab = (P - Tb) / (1 - P * Tb)            # A's spikes vs B's tiles
    out = 0.5 * (term_ab + np.transpose(term_ab, (0, 2, 1)))
    idx = np.arange(n)
    out[:, idx, idx] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Event-aligned rates and spectra
# --------------------------------------------------------------------------- #

def event_aligned_rate(spike_lists, events, pre, post, bin_sec):
    """Mean rate (Hz) of every unit around ``events``: (centers, (n_units, n_bins))."""
    events = np.asarray(events, dtype=float)
    edges = np.arange(-pre, post + bin_sec / 2, bin_sec)
    out = np.zeros((len(spike_lists), edges.size - 1))
    for i, t in enumerate(spike_lists):
        lo = np.searchsorted(t, events - pre)
        hi = np.searchsorted(t, events + post)
        lens = hi - lo
        if lens.sum() == 0:
            continue
        take = np.repeat(lo - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) \
            + np.arange(lens.sum())
        rel = t[take] - np.repeat(events, lens)
        out[i] = np.histogram(rel, edges)[0] / (events.size * bin_sec)
    return 0.5 * (edges[:-1] + edges[1:]), out


def population_spectrum(axis, fs=5000.0, nperseg=8192):
    """
    Power spectrum of the summed spike train of all units, and the sum of the
    single-unit spectra, over the axis windows joined without gaps.

    (pop - sum_single) / sum_single is the part of the spectrum that comes from
    spikes of DIFFERENT units being locked to each other; a periodic signal
    shared across units shows up there as a peak at its frequency.
    """
    starts_compact = np.concatenate([[0.0], np.cumsum(axis['duration'])[:-1]])
    compact = axis['t'] - axis['offset'][axis['win']] + starts_compact[axis['win']]
    n = int(np.ceil(axis['duration'].sum() * fs)) + 1
    idx = np.minimum((compact * fs).astype(np.int64), n - 1)
    f, p_pop = signal.welch(np.bincount(idx, minlength=n).astype(np.float32), fs=fs,
                            nperseg=nperseg)
    single = np.zeros_like(p_pop)
    order = np.argsort(axis['unit'], kind='stable')
    for ix in np.split(idx[order], np.cumsum(np.bincount(
            axis['unit'], minlength=len(axis['unit_ids'])))[:-1]):
        if ix.size:
            single += signal.welch(np.bincount(ix, minlength=n).astype(np.float32),
                                   fs=fs, nperseg=nperseg)[1]
    return f, p_pop, single
