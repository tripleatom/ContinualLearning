r"""Cross-correlograms (CCGs) of individual unit pairs during NREM sleep.

How to read one CCG (unit A -> unit B)
---------------------------------------
Take every spike of A as time 0 and ask how fast B fires at each moment before
and after it.  The y value at lag +5 ms is B's firing rate (Hz) 5 ms AFTER an
A spike; at -5 ms, 5 ms BEFORE.  Two reference lines:

    dashed gray   B's average rate -- what you would see if B ignored A
    orange        50 ms-jitter expectation -- what shared UP/DOWN states and
                  slow (> 50 ms) rate changes alone would produce.  With
                  --scope whole there are no UP states across the day, so the
                  jitter cells are plain 50 ms windows counted from the start of
                  each recording: orange = any shared rate change slower than
                  50 ms (behavioural state, arousal, UP/DOWN, ...)

Bars above the orange line are co-firing that shared slow drive does not
explain; a narrow peak there means spikes precisely aligned in time.

Pages (per sleep block, in <sortout>/cofiring/pair_ccgs/):

    example    one pair, annotated
    gallery    single pairs: strongest, strongest cross-shank, typical, random,
               plus a "check" row of pairs whose shape suggests an artifact
    grid       every pair among 8 units (autocorrelograms on the diagonal)
    all_pairs  every pair with enough spikes, one row each, sorted by excess

Usage
-----
    python plot_pair_ccgs.py                              # pre + post, NREM
    python plot_pair_ccgs.py --blocks post --scope up     # UP states only
    python plot_pair_ccgs.py --scope whole                # the whole day, every recording,
                                                          # spikes from the sorting analyzers
    python plot_pair_ccgs.py --scope whole --recordings presleep,postsleep
    python plot_pair_ccgs.py --blocks post --reuse --units shank0_unit1,shank2_unit7
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cofiring as cf                                    # noqa: E402
import tc_jitter as tj                                   # noqa: E402
import run_tuning_coupling as rt                         # noqa: E402
from describe_cofiring import unit_positions, MIN_UP_SPIKES, NREM_GAP_SEC  # noqa: E402
from run_tuning_coupling import COLORS, INK, INK_2, GRID, _style   # noqa: E402

EXCESS_WINDOW_MS = 10.0     # pairs are ranked by extra co-firing within +-this lag
MIN_EXPECTED = 50           # ...among pairs expecting at least this many coincidences there
N_GRID_UNITS = 8

REPRO_STAMP = ''


def set_stamp(argv, paths, found):
    global REPRO_STAMP
    inputs = '\n'.join(f'  {k}: {v}' for k, v in sorted(found.items()))
    REPRO_STAMP = (f"python plot_pair_ccgs.py {' '.join(argv)}\n"
                   f"session {paths['animal']} {paths['date']}  |  {paths['sortout']}\n"
                   f"generated {datetime.now():%Y-%m-%d %H:%M} by "
                   f"reactivation/tuning_coupling/plot_pair_ccgs.py\ninputs:\n{inputs}")


def save(fig, path):
    if REPRO_STAMP:
        fig.text(0.005, -0.01, REPRO_STAMP, fontsize=5.5, color=INK_2, ha='left',
                 va='top', family='monospace')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  figure: {path}")


# --------------------------------------------------------------------------- #
# Computing per-pair CCGs
# --------------------------------------------------------------------------- #

def pair_ccgs(axis, max_lag, bin_sec, chunk=30_000_000):
    """
    Counts for every ORDERED unit pair (i, j), autocorrelograms included:
    ccg[i, j, k] = number of j spikes at lag in bin k after (lag > 0) or
    before (lag < 0) an i spike.  Lag bins run from -max_lag to +max_lag.
    """
    tj.check_bin_width(bin_sec)
    n = len(axis['unit_ids'])
    nb = int(round(max_lag / bin_sec))
    acc = np.zeros(n * n * nb, dtype=np.int64)
    buf, size = [], 0
    for a, b, lag in tj._forward_pairs(axis['t'], max_lag):
        k = ((lag + tj.LAG_EPS) / bin_sec).astype(np.int64)
        keep = k < nb
        key = (axis['unit'][a].astype(np.int64) * n + axis['unit'][b]) * nb + k
        buf.append(key[keep])
        size += int(keep.sum())
        if size > chunk:
            acc += np.bincount(np.concatenate(buf), minlength=n * n * nb)
            buf, size = [], 0
    if buf:
        acc += np.bincount(np.concatenate(buf), minlength=n * n * nb)
    after = acc.reshape(n, n, nb)                       # j after i
    before = after.transpose(1, 0, 2)[:, :, ::-1]       # j before i
    return np.concatenate([before, after], axis=2).astype(np.float32)


def compute(spikes, up, bouts, ids, scope, max_lag, bin_sec, jitter_sec, n_surrogates,
            seed):
    """Observed and jitter-expected CCGs for all pairs in the chosen scope."""
    if scope == 'up':
        axis = tj.up_time_axis(spikes, up, ids)
        cs, cl, _ = tj.jitter_cells(axis, jitter_sec)
    else:
        axis = tj.up_time_axis(spikes, bouts, ids, gap_sec=NREM_GAP_SEC)
        seg_s, seg_e, _ = cf.nrem_segments(bouts, up)
        cs, cl = cf.segment_cells(axis, bouts[:, 0], seg_s, seg_e, width=jitter_sec)
    return compute_on_axis(axis, cs, cl, ids, max_lag, bin_sec, n_surrogates, seed)


def compute_on_axis(axis, cs, cl, ids, max_lag, bin_sec, n_surrogates, seed):
    """Observed CCGs and the mean over jitter surrogates (cells cs, cl)."""
    tic = time.time()
    obs = pair_ccgs(axis, max_lag, bin_sec)
    print(f"  observed CCGs: {obs.shape[0]} units, {obs.shape[2]} lag bins "
          f"({time.time() - tic:.0f} s)")
    rng = np.random.default_rng(seed)
    jit = np.zeros_like(obs)
    for s in range(n_surrogates):
        jit += pair_ccgs(cf.jitter_axis(axis, cs, cl, rng), max_lag, bin_sec)
    jit /= max(n_surrogates, 1)
    print(f"  jitter expectation from {n_surrogates} surrogates ({time.time() - tic:.0f} s)")
    nb = obs.shape[2] // 2
    lag_ms = (np.arange(-nb, nb) + 0.5) * bin_sec * 1e3
    return {'obs': obs, 'jit': jit, 'lag_ms': lag_ms,
            'n_spikes': np.bincount(axis['unit'], minlength=len(ids)),
            'total_sec': float(axis['duration'].sum()), 'bin_sec': bin_sec}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

MIN_WHOLE_SPIKES = 500       # whole-session units need at least this many spikes
RECORDING_GAP_SEC = 5.0      # empty time between recordings on the axis


def load_whole_session(paths, recordings=None, verbose=True):
    """
    Every unit's full spike train, straight from each shank's sorting analyzer.

    Spike trains are in the day's concatenated sorter space (one NWB per shank
    = that day's .rec files joined in ``conversion_list.txt`` order), so each
    recording is recovered as its own window from that file and the recordings
    are laid out with RECORDING_GAP_SEC between them: no lag and no jitter cell
    ever spans the join between two files.

    Returns (spikes {unit: seconds in sorter space}, shank {unit: int},
    windows (n_rec, 2) seconds, recording names, sampling rate).
    """
    from extract_sleep_session_spikes_multishank import (_resolve_sources,
                                                         day_concat_boundaries)
    day = rt.experiment_data_day_folder(paths['sortout'], paths['animal'], paths['date'])
    bounds = day_concat_boundaries(day, verbose=verbose)
    names = [k for k in bounds
             if recordings is None or any(r.lower() in k.lower() for r in recordings)]
    if not names:
        raise ValueError(f"No recording matches {recordings}; available: {list(bounds)}")
    sources, fs, curated = _resolve_sources(paths['sortout'])
    windows = np.array([[bounds[k][0] / fs, bounds[k][1] / fs] for k in names])
    spikes, shank = {}, {}
    for source in sources:
        sorting = source['sorting']
        for uid in source['unit_ids']:
            sh = source['shank_of'](uid)
            key = f"shank{sh}_unit{uid}" if sh is not None else f"unit{uid}"
            spikes[key] = np.sort(np.asarray(sorting.get_unit_spike_train(uid)) / fs)
            shank[key] = -1 if sh is None else int(sh)
    if verbose:
        total = float(np.sum(windows[:, 1] - windows[:, 0]))
        print(f"  {len(spikes)} units ({'curated' if curated else 'uncurated'}), "
              f"{len(names)} recordings, {total / 60:.1f} min: {', '.join(names)}")
    return spikes, shank, windows, names, fs


def pair_rate(data, i, j):
    """(observed, jitter) rate of j around i's spikes in Hz, and j's mean rate."""
    scale = 1.0 / (data['n_spikes'][i] * data['bin_sec'])
    return (data['obs'][i, j] * scale, data['jit'][i, j] * scale,
            data['n_spikes'][j] / data['total_sec'])


def draw_ccg(ax, data, i, j, ids, shank, small=False, auto=False):
    lag = data['lag_ms']
    obs, jit, mean = pair_rate(data, i, j)
    width = lag[1] - lag[0]
    ax.bar(lag, obs, width=width, color=INK_2 if auto else COLORS[0],
           alpha=0.55 if auto else 0.75, linewidth=0)
    if not auto:
        ax.plot(lag, jit, color=COLORS[1], linewidth=1.2 if small else 2)
        ax.axhline(mean, color=INK_2, linewidth=1, linestyle='--')
    ax.axvline(0, color=INK_2, linewidth=0.6)
    ax.set_xlim(lag[0] - width / 2, lag[-1] + width / 2)
    ax.set_ylim(bottom=0)
    if small:
        ax.tick_params(labelsize=7, colors=INK_2)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    else:
        _style(ax)


def pair_label(ids, shank, i, j):
    return f"{ids[i]} → {ids[j]}" + ("" if shank[i] == shank[j] else "  (cross-shank)")


def excess_stats(data):
    """Fractional extra co-firing within +-EXCESS_WINDOW_MS, and its expectation."""
    lag = data['lag_ms']
    w = np.abs(lag) <= EXCESS_WINDOW_MS
    o = data['obs'][:, :, w].sum(axis=2)
    e = data['jit'][:, :, w].sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = (o - e) / e
    return frac, e


def artifact_flags(data, shank):
    """
    Pairs whose CCG shape points to a recording/sorting problem rather than
    co-firing.  Both are symmetric (n, n) boolean matrices.

    zero_spike  : the +-1 ms bins exceed 3x the 2-6 ms flanks -- the same event
                  detected as a spike in both units (shared noise, crosstalk, or
                  a duplicated unit); implausible across shanks 300 um apart.
    split_like  : same shank, centre (+-1 ms) below half the flanks while the
                  flanks are > 2x the jitter expectation -- a refractory gap
                  with burst shoulders, i.e. the CCG looks like an
                  autocorrelogram: often ONE neuron split into two units (or
                  strong synchrony hidden by sorting collisions).
    """
    lag = data['lag_ms']
    centre = np.abs(lag) < 1
    flank = (np.abs(lag) > 2) & (np.abs(lag) < 6)
    o0 = data['obs'][:, :, centre].mean(axis=2)
    of = data['obs'][:, :, flank].mean(axis=2)
    j0 = data['jit'][:, :, centre].sum(axis=2)
    jf = data['jit'][:, :, flank].mean(axis=2)
    same = np.asarray(shank)[:, None] == np.asarray(shank)[None, :]
    zero_spike = (j0 >= 5) & (o0 > 3 * np.maximum(of, 1e-9))
    split_like = same & (jf * flank.sum() >= 5) & (o0 < 0.5 * of) & (of > 2 * jf)
    for m in (zero_spike, split_like):
        m |= m.T
        np.fill_diagonal(m, False)
    return zero_spike, split_like


def plot_example(data, i, j, ids, shank, title, path,
                 orange_txt='expected from shared UP/DOWN states and slow (>50 ms) rate changes'):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    draw_ccg(ax, data, i, j, ids, shank)
    obs, jit, mean = pair_rate(data, i, j)
    k = int(np.argmax(obs - jit))
    ax.set_ylim(0, 1.12 * max(obs.max(), jit.max()))
    ax.annotate(f"B fires at {obs[k]:.1f} Hz here,\nexpected {jit[k]:.1f} Hz from shared "
                f"slow drive\n→ {obs[k] / jit[k]:.1f}× at {data['lag_ms'][k]:+.1f} ms",
                (data['lag_ms'][k], obs[k]), xytext=(0.66, 0.72), textcoords='axes fraction',
                fontsize=10, color=INK, arrowprops=dict(arrowstyle='->', color=INK_2),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRID))
    ax.text(0.01, 0.98,
            "How to read this:\n"
            "• every spike of A is placed at time 0\n"
            "• bars = how fast B fires at each lag (right of 0: after A, left: before A)\n"
            f"• dashed gray = B's average rate ({mean:.1f} Hz): no relation to A\n"
            f"• orange = {orange_txt}\n"
            "• bars above orange = co-firing that slow drive does not explain\n"
            "• an empty bin at 0 on the same shank = sorting collisions, not biology",
            transform=ax.transAxes, va='top', fontsize=9.5, color=INK,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRID))
    ax.set_xlabel('lag of unit B relative to a unit-A spike (ms)', fontsize=12)
    ax.set_ylabel('rate of unit B (Hz)', fontsize=12)
    ax.set_title(f"{title}\nA = {ids[i]}  (shank {shank[i]}),  B = {ids[j]}  "
                 f"(shank {shank[j]})", fontsize=13, color=INK)
    save(fig, path)


def plot_gallery(data, rows, ids, shank, title, path, flags=None):
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.3 * ncol, 3.0 * len(rows)),
                             squeeze=False)
    fig.suptitle(f"{title}\nbars = rate of B around A's spikes (Hz);  orange = 50 ms-jitter "
                 f"expectation;  dashed = B's mean rate", fontsize=13, fontweight='bold',
                 color=INK)
    frac, _ = excess_stats(data)
    for r, (name, pairs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(pairs):
                ax.axis('off')
                continue
            i, j = pairs[c]
            draw_ccg(ax, data, i, j, ids, shank, small=True)
            note = ''
            if flags is not None:
                note = ('  [0-lag spike]' if flags[0][i, j] else
                        '  [split-like]' if flags[1][i, j] else '')
            ax.set_title(f"{ids[i]} → {ids[j]}{note}\n+{100 * frac[i, j]:.0f}% within "
                         f"±{EXCESS_WINDOW_MS:g} ms", fontsize=7.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{name}\n\nHz", fontsize=9, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel('lag (ms)', fontsize=8)
    fig.tight_layout()
    save(fig, path)


def plot_grid(data, units, ids, shank, title, path):
    m = len(units)
    fig, axes = plt.subplots(m, m, figsize=(2.2 * m, 1.9 * m), squeeze=False)
    fig.suptitle(f"{title}\nrow = reference unit A (spike at 0), column = unit B;  "
                 f"diagonal = autocorrelogram;  orange = 50 ms-jitter expectation",
                 fontsize=12, fontweight='bold', color=INK)
    for r, i in enumerate(units):
        for c, j in enumerate(units):
            ax = axes[r, c]
            draw_ccg(ax, data, i, j, ids, shank, small=True, auto=(i == j))
            if r == 0:
                ax.set_title(f"{ids[j]}\n(s{shank[j]})", fontsize=7.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{ids[i]}\n(s{shank[i]})", fontsize=7.5, color=INK)
            if r < m - 1:
                ax.set_xticklabels([])
    fig.tight_layout()
    save(fig, path)


def plot_all_pairs(data, ids, shank, title, path):
    n = len(ids)
    frac, expected = excess_stats(data)
    iu = np.triu_indices(n, k=1)
    ok = expected[iu] >= MIN_EXPECTED
    pi, pj = iu[0][ok], iu[1][ok]
    order = np.argsort(frac[pi, pj])[::-1]
    pi, pj = pi[order], pj[order]
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = data['obs'][pi, pj] / data['jit'][pi, pj]
    kernel = np.ones(3) / 3                      # edge-padded, so the ends are not pulled down
    ratio = np.apply_along_axis(
        lambda x: np.convolve(np.pad(x, 1, mode='edge'), kernel, 'valid'), 1, ratio)
    lag = data['lag_ms']

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.35, wspace=0.25)
    fig.suptitle(f"{title}\nevery pair expecting ≥ {MIN_EXPECTED} coincidences within "
                 f"±{EXCESS_WINDOW_MS:g} ms ({ok.sum()} of {len(iu[0])} pairs), one row each",
                 fontsize=13, fontweight='bold', color=INK)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(np.log2(np.clip(ratio, 0.25, 4)), aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest',
                   extent=[lag[0] - 0.5, lag[-1] + 0.5, len(pi), 0])
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_xlabel('lag (ms)', fontsize=11)
    ax.set_ylabel('pair (sorted: most extra co-firing at top)', fontsize=11)
    ax.set_title('observed / jitter expectation (red = more than expected)', fontsize=11,
                 color=INK)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(['0.5×', '0.7×', '1×', '1.4×', '2×'])

    ax = fig.add_subplot(gs[0, 1])
    same = shank[pi] == shank[pj]
    for k, (mask, name) in enumerate(((same, 'same shank'), (~same, 'cross shank'))):
        if mask.any():
            ax.plot(lag, np.nanmean(ratio[mask], axis=0), color=COLORS[k], linewidth=2,
                    label=f"{name} ({mask.sum()} pairs)")
    ax.axhline(1, color=INK_2, linewidth=1)
    ax.set_xlabel('lag (ms)', fontsize=11)
    ax.set_ylabel('mean observed / jitter', fontsize=11)
    ax.set_title('Average over the pairs at left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = fig.add_subplot(gs[1, 1])
    vals = 100 * frac[pi, pj]
    bins = np.linspace(np.percentile(vals, 0.5), np.percentile(vals, 99.5), 60)
    for k, (mask, name) in enumerate(((same, 'same shank'), (~same, 'cross shank'))):
        if mask.any():
            ax.hist(vals[mask], bins=bins, color=COLORS[k], alpha=0.6, density=True,
                    label=f"{name}: median {np.median(vals[mask]):+.1f}%")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel(f'extra co-firing within ±{EXCESS_WINDOW_MS:g} ms (% over jitter)',
                  fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title('How much extra co-firing pairs have', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    save(fig, path)


def choose_pairs(data, shank, seed, flags, per_row=6):
    """Rows of example pairs (unordered i < j), each an (A, B) tuple.  Flagged
    pairs are kept out of the ordinary rows and shown in their own row."""
    n = len(shank)
    frac, expected = excess_stats(data)
    iu = np.triu_indices(n, k=1)
    flagged = (flags[0] | flags[1])[iu]
    ok = (expected[iu] >= MIN_EXPECTED) & ~flagged
    pi, pj, f = iu[0][ok], iu[1][ok], frac[iu][ok]
    order = np.argsort(f)[::-1]
    cross = shank[pi] != shank[pj]
    rng = np.random.default_rng(seed)
    median_near = np.argsort(np.abs(f - np.median(f)))
    strongest = [(pi[k], pj[k]) for k in order[:per_row]]
    rows = [('strongest', strongest),
            ('strongest\ncross-shank', [(pi[k], pj[k]) for k in order
                                         if cross[k] and (pi[k], pj[k]) not in strongest]
             [:per_row]),
            ('typical\n(median)', [(pi[k], pj[k]) for k in median_near[:per_row]]),
            ('random', [(pi[k], pj[k]) for k in rng.choice(len(f), min(per_row, len(f)),
                                                            replace=False)])]
    fi, fj = iu[0][flagged], iu[1][flagged]
    if fi.size:
        rank = np.argsort(frac[fi, fj])[::-1]
        rows.append(('check:\nartifact?', [(fi[k], fj[k]) for k in rank[:per_row]]))
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sortout', default=None)
    parser.add_argument('--animal', default=None)
    parser.add_argument('--date', default=None)
    parser.add_argument('--blocks', default='pre,post')
    parser.add_argument('--scope', default='nrem', choices=('nrem', 'up', 'whole'),
                        help='nrem = all NREM incl. DOWN states (default); up = UP only; '
                             'whole = the entire day (every recording), spikes loaded '
                             'from each shank\'s sorting analyzer')
    parser.add_argument('--recordings', default=None,
                        help='with --scope whole: comma-separated substrings of the '
                             'recordings to include (default: all in conversion_list.txt)')
    parser.add_argument('--max-lag-ms', type=float, default=100.0)
    parser.add_argument('--bin-ms', type=float, default=1.0,
                        help='lag bin; must be a whole number of 30 kHz samples')
    parser.add_argument('--jitter-ms', type=float, default=50.0)
    parser.add_argument('--n-surrogates', type=int, default=10)
    parser.add_argument('--units', default=None,
                        help='comma-separated unit ids for the grid page '
                             f'(default: {N_GRID_UNITS} units from the strongest pairs)')
    parser.add_argument('--reuse', action='store_true',
                        help='load the newest saved CCGs for this block/scope instead '
                             'of recomputing (fast re-plotting)')
    parser.add_argument('--min-shank-fraction', type=float, default=0.5)
    parser.add_argument('--min-alignment-ratio', type=float, default=1.25)
    parser.add_argument('--allow-alignment-failure', action='store_true')
    parser.add_argument('--min-up-ms', type=float, default=50.0)
    parser.add_argument('--max-up-ms', type=float, default=2000.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    paths = rt.session_paths(args.animal, args.date, args.sortout)
    found, _ = rt.discover_inputs(paths, verbose=False)
    out_dir = paths['sortout'] / 'cofiring' / 'pair_ccgs'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    set_stamp(sys.argv[1:] if argv is None else list(argv), paths, found)
    label = f"{paths['animal']} {paths['date']}"

    if args.scope == 'whole':
        recs = ([r.strip() for r in args.recordings.split(',') if r.strip()]
                if args.recordings else None)
        blocks = ['session' if recs is None else 'session_' + '+'.join(recs)]
    else:
        blocks = [b.strip() for b in args.blocks.split(',') if b.strip()]

    for block in blocks:
        if args.scope != 'whole' and (f'sleep_{block}' not in found
                                      or f'mua_{block}' not in found):
            print(f"[{block}] skipped -- missing sleep spikes or MUA windows")
            continue
        print("\n" + "=" * 72 + f"\nPAIR CCGs, {block.upper()} ({args.scope})\n"
              + "=" * 72)
        rec_names = None
        cache = sorted(out_dir.glob(f"pair_ccgs_{block}_{args.scope}_*.npz"))
        if args.reuse and cache:
            print(f"  reusing {cache[-1].name}")
            z = np.load(cache[-1], allow_pickle=True)
            data = {k: z[k] for k in ('obs', 'jit', 'lag_ms', 'n_spikes')}
            data['total_sec'] = float(z['total_sec'])
            data['bin_sec'] = float(z['bin_sec'])
            ids, shank = list(z['unit_ids']), z['shank']
            rec_names = list(z['recordings']) if 'recordings' in z else None
        elif args.scope == 'whole':
            spikes, shank_of, windows, rec_names, fs = load_whole_session(paths, recs)
            probe = tj.up_time_axis(spikes, windows, gap_sec=RECORDING_GAP_SEC)
            ids = [u for u, c in zip(probe['unit_ids'], probe['n_spikes'])
                   if c >= MIN_WHOLE_SPIKES]
            print(f"  {len(ids)} units with >= {MIN_WHOLE_SPIKES} spikes")
            shank = np.array([shank_of[u] for u in ids])
            axis = tj.up_time_axis(spikes, windows, ids, gap_sec=RECORDING_GAP_SEC)
            cs, cl, _ = tj.jitter_cells(axis, args.jitter_ms / 1e3)
            print(f"  {axis['t'].size} spikes on the axis")
            data = compute_on_axis(axis, cs, cl, ids, args.max_lag_ms / 1e3,
                                   args.bin_ms / 1e3, args.n_surrogates, args.seed)
            npz = out_dir / f"pair_ccgs_{block}_{args.scope}_{stamp}.npz"
            np.savez_compressed(npz, unit_ids=np.array(ids, dtype=object), shank=shank,
                                recordings=np.array(rec_names, dtype=object),
                                **{k: data[k] for k in ('obs', 'jit', 'lag_ms', 'n_spikes')},
                                total_sec=data['total_sec'], bin_sec=data['bin_sec'])
            print(f"  saved {npz}")
        else:
            spikes, up, meta, report = rt.prepare_block(
                found[f'sleep_{block}'], None, found[f'mua_{block}'],
                nrem_pkl=found.get(f'nrem_{block}'),
                min_shank_fraction=args.min_shank_fraction, min_up_ms=args.min_up_ms,
                max_up_ms=args.max_up_ms, min_alignment_ratio=args.min_alignment_ratio,
                allow_alignment_failure=args.allow_alignment_failure, verbose=False)
            bouts = report['nrem_windows']
            if bouts is None:
                bouts = np.array([[0.0, meta['duration_sec']]])
            probe = tj.up_time_axis(spikes, up)
            ids = [u for u, c in zip(probe['unit_ids'], probe['n_spikes'])
                   if c >= MIN_UP_SPIKES]
            shank, _ = unit_positions(found.get('tuning'), ids, found[f'sleep_{block}'])
            data = compute(spikes, up, bouts, ids, args.scope, args.max_lag_ms / 1e3,
                           args.bin_ms / 1e3, args.jitter_ms / 1e3, args.n_surrogates,
                           args.seed)
            npz = out_dir / f"pair_ccgs_{block}_{args.scope}_{stamp}.npz"
            np.savez_compressed(npz, unit_ids=np.array(ids, dtype=object), shank=shank,
                                **{k: data[k] for k in ('obs', 'jit', 'lag_ms', 'n_spikes')},
                                total_sec=data['total_sec'], bin_sec=data['bin_sec'])
            print(f"  saved {npz}")

        if args.scope == 'whole':
            n_rec = len(rec_names) if rec_names is not None else '?'
            title = (f"{label} -- whole session ({n_rec} recordings, "
                     f"{data['total_sec'] / 60:.0f} min, wake + sleep)")
            orange_txt = 'expected from any shared rate change slower than 50 ms'
        else:
            scope_txt = 'all NREM (incl. DOWN)' if args.scope == 'nrem' else 'UP states only'
            title = f"{label} sleep {block.upper()} -- {scope_txt}"
            orange_txt = 'expected from shared UP/DOWN states and slow (>50 ms) rate changes'
        base = out_dir / f"pair_ccgs_{block}_{args.scope}"
        flags = artifact_flags(data, shank)
        iu = np.triu_indices(len(ids), k=1)
        print(f"  flagged pairs: {int(flags[0][iu].sum())} with a zero-lag spike, "
              f"{int(flags[1][iu].sum())} split-like (same shank) -- shown in the "
              f"gallery's 'check' row, excluded from the other rows")
        rows = choose_pairs(data, np.asarray(shank), args.seed, flags)
        i, j = rows[0][1][0]
        plot_example(data, i, j, ids, shank, f"{title}: how to read a cross-correlogram",
                     f"{base}_example_{stamp}.png", orange_txt=orange_txt)
        plot_gallery(data, rows, ids, shank, title, f"{base}_gallery_{stamp}.png",
                     flags=flags)
        if args.units:
            wanted = [u.strip() for u in args.units.split(',') if u.strip()]
            missing = [u for u in wanted if u not in ids]
            if missing:
                print(f"  not in this block (or too few spikes): {missing}")
            units = [ids.index(u) for u in wanted if u in ids]
        else:
            units = []
            for a, b in rows[0][1] + rows[1][1]:
                for u in (a, b):
                    if u not in units:
                        units.append(u)
            units = units[:N_GRID_UNITS]
        if len(units) >= 2:
            plot_grid(data, units, ids, shank, title, f"{base}_grid_{stamp}.png")
        plot_all_pairs(data, ids, np.asarray(shank), title, f"{base}_all_pairs_{stamp}.png")
    return 0


if __name__ == '__main__':
    sys.exit(main())
