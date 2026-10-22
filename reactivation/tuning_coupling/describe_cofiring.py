r"""What does co-firing look like in NREM sleep?  (Descriptive; nothing is tested.)

For each sleep block, all sorted units with enough spikes in UP states, five
figure pages:

  0  data check    fine cross-correlogram vs jitter (same vs cross shank) and
                   the cross-unit part of the population spike spectrum --
                   periodic noise or sorting artifacts show up here first
  A  UP states     every unit's rate aligned to UP onset and offset, onset
                   latency vs depth, population coupling
  B  CCG           pooled cross-correlogram at +-1 s, +-100 ms, +-15 ms against
                   three baselines (rate only / UP membership / 50 ms jitter),
                   and the split into UP-membership, slow and fine parts
  C  timescale     spike-count correlation vs bin width (1-100 ms): over all
                   NREM, inside UP states, and for the UP-membership baseline
  D  STTC          spike time tiling coefficient inside UP states vs dt,
                   pair matrix, rate dependence, same vs cross shank
  E  UP vs whole   the same metrics over three time scopes -- the whole sleep
                   block (incl. wake), all NREM (incl. DOWN), UP states only --
                   and whether the same pairs stand out in each

Usage
-----
    python describe_cofiring.py                    # grating_config session
    python describe_cofiring.py --date 260728 --blocks post
"""
from __future__ import annotations

import argparse
import pickle
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
from run_tuning_coupling import COLORS, INK, INK_2, GRID, _style   # noqa: E402

MIN_UP_SPIKES = 50          # units with fewer spikes inside UP windows are left out
NREM_GAP_SEC = 5.0          # empty time between NREM bouts on the NREM axis
PC_BIN_SEC = 0.025          # bin for population coupling
STTC_SHOW_MS = 10.0         # dt shown in the STTC matrix/histogram panels
RSC_SHOW_MS = 25.0          # bin shown in the r_sc histogram

REPRO_STAMP = ''


def set_stamp(argv, paths, found):
    global REPRO_STAMP
    inputs = '\n'.join(f'  {k}: {v}' for k, v in sorted(found.items()))
    REPRO_STAMP = (f"python describe_cofiring.py {' '.join(argv)}\n"
                   f"session {paths['animal']} {paths['date']}  |  {paths['sortout']}\n"
                   f"generated {datetime.now():%Y-%m-%d %H:%M} by "
                   f"reactivation/tuning_coupling/describe_cofiring.py\ninputs:\n{inputs}")


def save(fig, path):
    fig.tight_layout()
    if REPRO_STAMP:
        fig.text(0.005, -0.01, REPRO_STAMP, fontsize=5.5, color=INK_2, ha='left',
                 va='top', family='monospace')
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  figure: {path}")


def unit_positions(tuning_pkl, unit_ids, sleep_pkl):
    """Shank and channel y (um) for every unit, from the tuning book's unit_info
    (which holds ALL sorted units), falling back to the sleep pkl's shank."""
    book = {}
    if tuning_pkl is not None:
        with open(tuning_pkl, 'rb') as f:
            book = pickle.load(f)
    with open(sleep_pkl, 'rb') as f:
        sleep_units = pickle.load(f)['spike_data']
    shank, depth = [], []
    for uid in unit_ids:
        info = (book.get(uid) or {}).get('unit_info') or {}
        sh = info.get('shank', sleep_units.get(uid, {}).get('shank'))
        loc = info.get('channel_location_um') or (np.nan, np.nan)
        shank.append(-1 if sh is None else int(sh))
        depth.append(float(loc[1]))
    return np.array(shank), np.array(depth)


def spearman(x, y):
    """Rank correlation over the entries finite in both."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 else np.nan


def iqr_band(values, axis=0):
    return np.nanpercentile(values, 25, axis=axis), np.nanpercentile(values, 75, axis=axis)


# --------------------------------------------------------------------------- #
# One block
# --------------------------------------------------------------------------- #

def describe_block(block, found, args, out_dir, stamp, label):
    spikes, up, meta, report = rt.prepare_block(
        found[f'sleep_{block}'], None, found[f'mua_{block}'],
        nrem_pkl=None if args.no_nrem else found.get(f'nrem_{block}'),
        min_shank_fraction=args.min_shank_fraction, min_up_ms=args.min_up_ms,
        max_up_ms=args.max_up_ms, min_alignment_ratio=args.min_alignment_ratio,
        allow_alignment_failure=args.allow_alignment_failure)
    bouts = report['nrem_windows']
    whole = np.array([[0.0, meta['duration_sec']]])     # entire block, wake included
    if bouts is None:
        bouts = whole.copy()

    probe = tj.up_time_axis(spikes, up)
    ids = [u for u, n in zip(probe['unit_ids'], probe['n_spikes']) if n >= MIN_UP_SPIKES]
    print(f"  {len(ids)} of {len(spikes)} units have >= {MIN_UP_SPIKES} spikes in UP states")
    shank, depth = unit_positions(found.get('tuning'), ids, found[f'sleep_{block}'])
    n = len(ids)
    same = shank[:, None] == shank[None, :]
    iu = np.triu_indices(n, k=1)
    same_pair = same[iu]

    up_axis = tj.up_time_axis(spikes, up, ids)                      # gap 1 s
    nrem_axis = tj.up_time_axis(spikes, bouts, ids, gap_sec=NREM_GAP_SEC)
    all_times = [np.asarray(spikes[u]) for u in ids]
    rng = np.random.default_rng(args.seed)
    tag = f"{label} sleep {block.upper()}"
    base = out_dir / f"cofiring_{block}"
    info = (f"{len(ids)} units, {len(up)} NREM UP windows ({up_axis['total_up_sec']:.0f} s), "
            f"{np.sum(bouts[:, 1] - bouts[:, 0]) / 60:.1f} min NREM")
    result = {'unit_ids': ids, 'shank': shank, 'depth_um': depth,
              'up_windows': up, 'nrem_windows': bouts, 'alignment': report}

    # ---------------- 0: data check ---------------------------------------
    tic = time.time()
    groups = np.where(same, 0, 1)
    np.fill_diagonal(groups, -1)
    fine_bin, fine_max = 0.0001, 0.010
    obs0 = tj.grouped_ccg(up_axis['t'], up_axis['unit'], groups, 2, fine_max, fine_bin)
    cs, cl, _ = tj.jitter_cells(up_axis, 0.05)
    sur0 = np.zeros(obs0.shape)
    for _ in range(args.check_surrogates):
        ts, us = tj.jitter_surrogate(up_axis['t'], up_axis['unit'], cs, cl, rng)
        sur0 += tj.grouped_ccg(ts, us, groups, 2, fine_max, fine_bin)
    sur0 /= args.check_surrogates
    f, p_pop, p_single = cf.population_spectrum(up_axis)
    cross_rel = (p_pop - p_single) / p_single
    band = f > 80
    peak_idx = np.argsort(cross_rel[band])[::-1]
    peaks = []
    for k in peak_idx:                        # top 3 peaks at least 30 Hz apart
        fk = f[band][k]
        if all(abs(fk - p) > 30 for p in peaks):
            peaks.append(fk)
        if len(peaks) == 3:
            break
    print(f"  0 data check ({time.time() - tic:.0f} s): cross-unit spectral peaks "
          f"{', '.join(f'{p:.0f}' for p in peaks)} Hz")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(f"{tag} -- 0. data check (inside UP states)\n{info}",
                 fontsize=14, fontweight='bold', color=INK)
    ax = axes[0]
    lag_ms = (np.arange(obs0.shape[1]) + 0.5) * fine_bin * 1e3
    for g, name in enumerate(('same-shank pairs', 'cross-shank pairs')):
        ax.plot(lag_ms, obs0[g] / sur0[g], color=COLORS[g], linewidth=1.6, label=name)
    ax.axhline(1, color=INK_2, linewidth=1)
    ax.set_xlabel('|lag| (ms), 0.1 ms bins', fontsize=11)
    ax.set_ylabel('observed / 50 ms-jitter expectation', fontsize=11)
    ax.set_title('Fine cross-correlogram, all pairs pooled\n'
                 'dip near 0 on same shank = sorting collisions; '
                 'regular peaks = periodic artifact', fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax = axes[1]
    ax.semilogy(f, p_pop, color=COLORS[0], linewidth=1.3, label='summed population train')
    ax.semilogy(f, p_single, color=COLORS[1], linewidth=1.3, label='sum of single-unit spectra')
    ax.set_xlabel('frequency (Hz)', fontsize=11)
    ax.set_ylabel('power (spikes² / Hz)', fontsize=11)
    ax.set_title('Spike-train power spectra (UP time, 5 kHz)', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax = axes[2]
    ax.plot(f, cross_rel, color=COLORS[0], linewidth=1.2)
    ax.axhline(0, color=INK_2, linewidth=1)
    for p in peaks:
        ax.annotate(f"{p:.0f} Hz", (p, cross_rel[np.argmin(np.abs(f - p))]),
                    xytext=(4, 4), textcoords='offset points', fontsize=9, color=INK)
    ax.set_xlabel('frequency (Hz)', fontsize=11)
    ax.set_ylabel('(population − Σ single) / Σ single', fontsize=11)
    ax.set_title('Cross-unit locking by frequency (0 = independent)', fontsize=11, color=INK)
    _style(ax)
    save(fig, f"{base}_0_datacheck_{stamp}.png")
    result['datacheck'] = {'lag_ms': lag_ms, 'obs': obs0, 'jitter': sur0, 'freq': f,
                           'p_pop': p_pop, 'p_single': p_single, 'peaks_hz': peaks}

    # ---------------- A: UP-state structure -------------------------------
    tic = time.time()
    nrem_rate = np.array([np.sum(np.searchsorted(t, bouts[:, 1]) - np.searchsorted(t, bouts[:, 0]))
                          for t in all_times], float) / np.sum(bouts[:, 1] - bouts[:, 0])
    c_on, r_on = cf.event_aligned_rate(all_times, up[:, 0], 0.3, 0.5, 0.01)
    c_off, r_off = cf.event_aligned_rate(all_times, up[:, 1], 0.5, 0.3, 0.01)
    with np.errstate(invalid='ignore', divide='ignore'):
        n_on = r_on / nrem_rate[:, None]
        n_off = r_off / nrem_rate[:, None]
    base_on = np.nanmean(n_on[:, (c_on >= -0.3) & (c_on < -0.1)], axis=1)
    smooth = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, 'same'), 1, n_on)
    post_on = (c_on >= 0) & (c_on <= 0.3)
    peak_on = np.nanmax(smooth[:, post_on], axis=1)
    latency = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(peak_on[i]) or peak_on[i] < 1.2 * base_on[i]:
            continue
        half = base_on[i] + 0.5 * (peak_on[i] - base_on[i])
        after = np.flatnonzero((c_on >= -0.1) & (smooth[i] >= half))
        if after.size:
            latency[i] = c_on[after[0]] * 1e3
    order = np.argsort(np.where(np.isfinite(latency), latency, np.inf))

    X_nrem, B_nrem = cf.binned_counts(all_times, bouts, PC_BIN_SEC)
    X_up, B_up = cf.binned_counts(all_times, up, PC_BIN_SEC)
    X_whole, B_whole = cf.binned_counts(all_times, whole, PC_BIN_SEC)
    pc_nrem = cf.population_coupling(cf.count_covariance(X_nrem, B_nrem))
    pc_up = cf.population_coupling(cf.count_covariance(X_up, B_up))
    pc_whole = cf.population_coupling(cf.count_covariance(X_whole, B_whole))
    print(f"  A UP-state structure ({time.time() - tic:.0f} s): median onset latency "
          f"{np.nanmedian(latency):.0f} ms, {int(np.isfinite(latency).sum())}/{n} units "
          f"UP-modulated")

    fig, axes = plt.subplots(2, 3, figsize=(19, 11),
                             gridspec_kw={'height_ratios': [1.6, 1]})
    fig.suptitle(f"{tag} -- A. how units fire across an UP state\n{info}",
                 fontsize=14, fontweight='bold', color=INK)
    vmax = np.nanpercentile(n_on, 99)
    for col, (centers, mat, name) in enumerate(((c_on, n_on, 'UP onset'),
                                                (c_off, n_off, 'UP offset'))):
        ax = axes[0, col]
        im = ax.imshow(mat[order], aspect='auto', cmap='Blues', vmin=0, vmax=vmax,
                       extent=[centers[0] * 1e3 - 5, centers[-1] * 1e3 + 5, n, 0],
                       interpolation='nearest')
        ax.axvline(0, color=COLORS[1], linewidth=1.5)
        ax.set_xlabel(f'time from {name} (ms)', fontsize=11)
        ax.set_ylabel('unit (sorted by onset latency)', fontsize=11)
        ax.set_title(f'Rate around {name} / unit\'s NREM mean rate', fontsize=11,
                     color=INK)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='× NREM mean rate')
        ax2 = axes[1, col]
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0))
        ax2.fill_between(centers * 1e3, mean - sem, mean + sem, color=COLORS[0],
                         alpha=0.25, linewidth=0)
        ax2.plot(centers * 1e3, mean, color=COLORS[0], linewidth=2)
        ax2.axvline(0, color=COLORS[1], linewidth=1.5)
        ax2.set_xlabel(f'time from {name} (ms)', fontsize=11)
        ax2.set_ylabel('× NREM mean rate (mean ± SEM)', fontsize=11)
        ax2.set_title(f'Average unit around {name}', fontsize=11, color=INK)
        _style(ax2)
    ax = axes[0, 2]
    ok = np.isfinite(latency) & np.isfinite(depth)
    ax.scatter(latency[ok], depth[ok], s=22, color=COLORS[0], alpha=0.7, linewidths=0)
    ax.set_xlabel('onset latency (ms, half-rise)', fontsize=11)
    ax.set_ylabel('channel y on probe (µm)', fontsize=11)
    ax.set_title(f'Onset latency vs depth ({ok.sum()} UP-modulated units)', fontsize=11,
                 color=INK)
    _style(ax)
    ax = axes[1, 2]
    ax.scatter(pc_nrem, pc_up, s=22, color=COLORS[0], alpha=0.7, linewidths=0)
    lim = [np.nanmin([pc_nrem, pc_up]), np.nanmax([pc_nrem, pc_up])]
    ax.plot(lim, lim, color=INK_2, linewidth=1, label='equal')
    ax.set_xlabel('population coupling, all NREM (incl. DOWN)', fontsize=11)
    ax.set_ylabel('population coupling, inside UP only', fontsize=11)
    ax.set_title(f'Each unit vs the summed rest ({PC_BIN_SEC * 1e3:.0f} ms bins)',
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    save(fig, f"{base}_A_upstates_{stamp}.png")
    result['upstates'] = {'onset_t': c_on, 'onset_rate': r_on, 'offset_t': c_off,
                          'offset_rate': r_off, 'nrem_rate_hz': nrem_rate,
                          'latency_ms': latency, 'pop_coupling_nrem': pc_nrem,
                          'pop_coupling_up': pc_up}

    # ---------------- B: pooled CCG with baselines -------------------------
    tic = time.time()
    seg_s, seg_e, _ = cf.nrem_segments(bouts, up)
    b_start = bouts[:, 0]
    cells = {
        'rate': cf.segment_cells(nrem_axis, b_start, bouts[:, 0], bouts[:, 1]),
        'up': cf.segment_cells(nrem_axis, b_start, seg_s, seg_e),
        'jit': cf.segment_cells(nrem_axis, b_start, seg_s, seg_e, width=0.05),
    }
    obs = cf.pooled_ccg(nrem_axis, shank)
    base_ccg = {}
    for key, (c_s, c_l) in cells.items():
        acc = None
        for _ in range(args.ccg_surrogates):
            r = cf.pooled_ccg(cf.jitter_axis(nrem_axis, c_s, c_l, rng), shank)
            acc = r if acc is None else {k: acc[k] + r[k] if k in ('fine', 'broad') else acc[k]
                                         for k in acc}
        base_ccg[key] = {k: (v / args.ccg_surrogates if k in ('fine', 'broad') else v)
                         for k, v in acc.items()}
    print(f"  B pooled CCG ({time.time() - tic:.0f} s)")

    pairs = {'all': [0, 1], 'cross': [1], 'same': [0]}[args.ccg_pairs]
    def pooled(d, key):
        return d[key][pairs].sum(axis=0)
    zooms = [('broad', 1.0, '±1 s (10 ms bins)'), ('fine', 0.1, '±100 ms (1 ms bins)'),
             ('fine', 0.015, '±15 ms (0.2 ms bins)')]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f"{tag} -- B. pooled cross-correlogram, NREM ({args.ccg_pairs} pairs)\n"
                 f"{info};  y = coincidences relative to independent firing",
                 fontsize=14, fontweight='bold', color=INK)
    curves_out = {}
    for col, (kind, half, name) in enumerate(zooms):
        o = pooled(obs, kind)
        rb, ub, jb = (pooled(base_ccg[k], kind) for k in ('rate', 'up', 'jit'))
        if kind == 'broad':
            lag = obs['broad_lag']
        else:
            factor = 5 if half > 0.02 else 1          # 0.2 ms -> 1 ms for the middle zoom
            nb = int(round(half / 0.0002)) // factor * factor
            o, rb, ub, jb = (x[:nb].reshape(-1, factor).sum(1) for x in (o, rb, ub, jb))
            step = 0.0002 * factor
            pos = (np.arange(o.size) + 0.5) * step
            lag = np.concatenate([-pos[::-1], pos])
            o, rb, ub, jb = (np.concatenate([x[::-1], x]) for x in (o, rb, ub, jb))
        # per-bin reference: the rate-only baseline shares the data's sample
        # lattice, so bin-width effects (e.g. the half-sample folded lag-0 bin) cancel
        ref = np.where(rb > 0, rb, np.nan)
        o, ub, jb, rbn = o / ref, ub / ref, jb / ref, rb / ref
        curves_out[name] = {'lag_s': lag, 'observed': o, 'up_membership': ub,
                            'jitter50': jb, 'rate_only': rbn}
        ax = axes[0, col]
        ax.axhline(1, color=INK_2, linewidth=1, linestyle='--', label='independent (rate only)')
        ax.plot(lag * 1e3, ub, color=COLORS[0], linewidth=1.6, label='UP-membership baseline')
        ax.plot(lag * 1e3, jb, color=COLORS[1], linewidth=1.6, label='50 ms-jitter baseline')
        ax.plot(lag * 1e3, o, color=INK, linewidth=1.8, label='observed')
        ax.set_xlabel('lag (ms)', fontsize=11)
        ax.set_ylabel('× independent', fontsize=11)
        ax.set_title(name, fontsize=12, color=INK)
        if col == 0:
            ax.legend(frameon=False, fontsize=9)
        _style(ax)
        ax = axes[1, col]
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.plot(lag * 1e3, ub - 1, color=COLORS[0], linewidth=1.6,
                label='shared UP/DOWN membership')
        ax.plot(lag * 1e3, jb - ub, color=COLORS[1], linewidth=1.6,
                label='slow within-UP time course (> 50 ms)')
        ax.plot(lag * 1e3, o - jb, color=COLORS[2], linewidth=1.6,
                label='fine co-firing (< 50 ms)')
        ax.plot(lag * 1e3, o - 1, color=INK, linewidth=1.8, label='total above independent')
        ax.set_xlabel('lag (ms)', fontsize=11)
        ax.set_ylabel('excess, × independent', fontsize=11)
        ax.set_title('Where the co-firing comes from', fontsize=11, color=INK)
        if col == 0:
            ax.legend(frameon=False, fontsize=9)
        _style(ax)
    save(fig, f"{base}_B_ccg_{stamp}.png")
    result['ccg'] = curves_out

    # ---------------- C: correlation vs timescale ---------------------------
    tic = time.time()
    widths = np.array(rt._floats(args.widths_ms)) / 1e3
    sur_up = cf.jitter_axis(up_axis, *tj.jitter_cells(up_axis, 1e9)[:2], rng)
    sur_times = cf.unit_times(sur_up, up[:, 0])
    rsc = {'whole': [], 'nrem': [], 'up': [], 'up_sur': []}
    for w in widths:
        for key, times, wins in (('whole', all_times, whole), ('nrem', all_times, bouts),
                                 ('up', all_times, up), ('up_sur', sur_times, up)):
            X, B = cf.binned_counts(times, wins, w)
            rsc[key].append(cf.correlation_from_cov(cf.count_covariance(X, B))[iu])
    rsc = {k: np.array(v) for k, v in rsc.items()}       # (n_widths, n_pairs)
    print(f"  C correlation vs timescale ({time.time() - tic:.0f} s)")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    fig.suptitle(f"{tag} -- C. spike-count correlation vs timescale\n{info}",
                 fontsize=14, fontweight='bold', color=INK)
    ax = axes[0]
    for k, (key, name) in enumerate((('nrem', 'all NREM (incl. DOWN)'),
                                     ('up', 'inside UP states'),
                                     ('up_sur', 'inside UP, UP-membership baseline'))):
        ax.plot(widths * 1e3, np.nanmean(rsc[key], axis=1), color=COLORS[k], marker='o',
                markersize=8, linewidth=2, label=name)
    ax.set_xscale('log')
    ax.set_xticks(widths * 1e3)
    ax.set_xticklabels([f"{w * 1e3:g}" for w in widths])
    ax.minorticks_off()
    ax.set_xlabel('bin width (ms)', fontsize=11)
    ax.set_ylabel('mean r_sc over pairs', fontsize=11)
    ax.set_title('Mean correlation vs bin width', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax = axes[1]
    for k, (mask, name) in enumerate(((same_pair, 'same shank'), (~same_pair, 'cross shank'))):
        vals = rsc['up'][:, mask]
        lo, hi = iqr_band(vals, axis=1)
        ax.fill_between(widths * 1e3, lo, hi, color=COLORS[k], alpha=0.15, linewidth=0)
        ax.plot(widths * 1e3, np.nanmedian(vals, axis=1), color=COLORS[k], marker='o',
                markersize=8, linewidth=2, label=f"{name} ({mask.sum()} pairs)")
    ax.set_xscale('log')
    ax.set_xticks(widths * 1e3)
    ax.set_xticklabels([f"{w * 1e3:g}" for w in widths])
    ax.minorticks_off()
    ax.set_xlabel('bin width (ms)', fontsize=11)
    ax.set_ylabel('r_sc inside UP (median, IQR band)', fontsize=11)
    ax.set_title('Inside UP states: same vs cross shank', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax = axes[2]
    k_show = int(np.argmin(np.abs(widths - RSC_SHOW_MS / 1e3)))
    both = np.concatenate([rsc['up'][k_show], rsc['up_sur'][k_show]])
    bins = np.linspace(np.nanpercentile(both, 0.5), np.nanpercentile(both, 99.5), 70)
    ax.hist(rsc['up_sur'][k_show], bins=bins, color=COLORS[1], alpha=0.6,
            label='UP-membership baseline')
    ax.hist(rsc['up'][k_show], bins=bins, color=COLORS[0], alpha=0.6, label='observed')
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel(f'r_sc inside UP, {widths[k_show] * 1e3:g} ms bins', fontsize=11)
    ax.set_ylabel('pairs', fontsize=11)
    ax.set_title('Distribution over pairs', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    save(fig, f"{base}_C_timescale_{stamp}.png")
    result['rsc'] = {'widths_s': widths, **rsc}

    # ---------------- D: STTC ----------------------------------------------
    tic = time.time()
    dts = np.array(rt._floats(args.sttc_ms)) / 1e3
    sttc_obs = cf.sttc(up_axis, dts)
    sttc_sur = cf.sttc(sur_up, dts)
    sttc_nrem = cf.sttc(nrem_axis, dts)
    whole_axis = tj.up_time_axis(spikes, whole, ids)
    sttc_whole = cf.sttc(whole_axis, dts)
    s_obs = sttc_obs[:, iu[0], iu[1]]
    s_sur = sttc_sur[:, iu[0], iu[1]]
    s_nrem = sttc_nrem[:, iu[0], iu[1]]
    s_whole = sttc_whole[:, iu[0], iu[1]]
    print(f"  D STTC ({time.time() - tic:.0f} s): median at "
          + ", ".join(f"{d * 1e3:g} ms {np.nanmedian(s_obs[k]):+.3f}"
                      for k, d in enumerate(dts)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(f"{tag} -- D. spike time tiling coefficient (inside UP states)\n{info}",
                 fontsize=14, fontweight='bold', color=INK)
    ax = axes[0, 0]
    for k, (vals, name) in enumerate(((s_obs, 'observed'),
                                      (s_sur, 'UP-membership baseline'))):
        lo, hi = iqr_band(vals, axis=1)
        ax.fill_between(dts * 1e3, lo, hi, color=COLORS[k], alpha=0.15, linewidth=0)
        ax.plot(dts * 1e3, np.nanmedian(vals, axis=1), color=COLORS[k], marker='o',
                markersize=8, linewidth=2, label=name)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xscale('log')
    ax.set_xticks(dts * 1e3)
    ax.set_xticklabels([f"{d * 1e3:g}" for d in dts])
    ax.minorticks_off()
    ax.set_xlabel('Δt (ms)', fontsize=11)
    ax.set_ylabel('STTC (median, IQR band)', fontsize=11)
    ax.set_title('STTC vs Δt', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    k_show = int(np.argmin(np.abs(dts - STTC_SHOW_MS / 1e3)))
    d_show = dts[k_show] * 1e3
    ax = axes[0, 1]
    pos_order = np.lexsort((depth, shank))
    M = sttc_obs[k_show][np.ix_(pos_order, pos_order)]
    lim = np.nanpercentile(np.abs(M), 99)
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    edges = np.flatnonzero(np.diff(shank[pos_order])) + 0.5
    for e in edges:
        ax.axhline(e, color=INK, linewidth=0.6)
        ax.axvline(e, color=INK, linewidth=0.6)
    ticks = [np.mean(np.flatnonzero(shank[pos_order] == s)) for s in np.unique(shank)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"s{s}" for s in np.unique(shank)])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"s{s}" for s in np.unique(shank)])
    ax.set_title(f'STTC at Δt = {d_show:g} ms, units by shank then depth', fontsize=11,
                 color=INK)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='STTC')

    ax = axes[1, 0]
    rate_up = np.bincount(up_axis['unit'], minlength=n) / up_axis['total_up_sec']
    geo = np.sqrt(rate_up[iu[0]] * rate_up[iu[1]])
    ax.scatter(geo, s_obs[k_show], s=5, color=COLORS[0], alpha=0.25, linewidths=0)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('geometric-mean rate inside UP (Hz)', fontsize=11)
    ax.set_ylabel(f'STTC at Δt = {d_show:g} ms', fontsize=11)
    ax.set_title('STTC vs firing rate (flat = rate-independent)', fontsize=11, color=INK)
    _style(ax)

    ax = axes[1, 1]
    vals = s_obs[k_show]
    bins = np.linspace(np.nanpercentile(vals, 0.5), np.nanpercentile(vals, 99.5), 70)
    for k, (mask, name) in enumerate(((same_pair, 'same shank'), (~same_pair, 'cross shank'))):
        ax.hist(vals[mask], bins=bins, color=COLORS[k], alpha=0.6, density=True,
                label=f"{name}: median {np.nanmedian(vals[mask]):+.3f}")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel(f'STTC at Δt = {d_show:g} ms', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title('Same vs cross shank', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    save(fig, f"{base}_D_sttc_{stamp}.png")
    result['sttc'] = {'dt_s': dts, 'observed': sttc_obs, 'up_membership': sttc_sur,
                      'nrem': sttc_nrem, 'whole': sttc_whole}

    # ---------------- E: UP states vs the whole period -----------------------
    scopes = (('whole', 'whole block (incl. wake)'), ('nrem', 'all NREM (incl. DOWN)'),
              ('up', 'UP states only'))
    sttc_scope = {'whole': s_whole, 'nrem': s_nrem, 'up': s_obs}
    pc_scope = {'whole': pc_whole, 'nrem': pc_nrem, 'up': pc_up}
    rank = {}                                  # Spearman rho, UP vs another scope
    for other in ('whole', 'nrem'):
        rank[('rsc', other)] = np.array([spearman(rsc['up'][k], rsc[other][k])
                                         for k in range(len(widths))])
        rank[('sttc', other)] = np.array([spearman(s_obs[k], sttc_scope[other][k])
                                          for k in range(len(dts))])
    k_w = int(np.argmin(np.abs(widths - RSC_SHOW_MS / 1e3)))
    k_d = int(np.argmin(np.abs(dts - STTC_SHOW_MS / 1e3)))
    print(f"  E UP vs whole: mean r_sc at {widths[k_w] * 1e3:g} ms "
          + ", ".join(f"{name} {np.nanmean(rsc[key][k_w]):+.4f}" for key, name in scopes)
          + f"; rank agreement UP vs whole rho={rank[('rsc', 'whole')][k_w]:.2f}")

    whole_min = meta['duration_sec'] / 60
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    fig.suptitle(f"{tag} -- E. co-firing inside UP states vs over the whole period\n"
                 f"{info}; whole block {whole_min:.1f} min", fontsize=14,
                 fontweight='bold', color=INK)

    def log_ms_axis(ax, values_s, label):
        ax.set_xscale('log')
        ax.set_xticks(values_s * 1e3)
        ax.set_xticklabels([f"{v * 1e3:g}" for v in values_s])
        ax.minorticks_off()
        ax.set_xlabel(label, fontsize=11)

    ax = axes[0, 0]
    for k, (key, name) in enumerate(scopes):
        ax.plot(widths * 1e3, np.nanmean(rsc[key], axis=1), color=COLORS[k], marker='o',
                markersize=8, linewidth=2, label=name)
    log_ms_axis(ax, widths, 'bin width (ms)')
    ax.set_ylabel('mean r_sc over pairs', fontsize=11)
    ax.set_title('Spike-count correlation by scope', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[0, 1]
    for k, (key, name) in enumerate(scopes):
        ax.plot(dts * 1e3, np.nanmedian(sttc_scope[key], axis=1), color=COLORS[k],
                marker='o', markersize=8, linewidth=2, label=name)
    ax.axhline(0, color=INK_2, linewidth=1)
    log_ms_axis(ax, dts, 'Δt (ms)')
    ax.set_ylabel('median STTC over pairs', fontsize=11)
    ax.set_title('STTC by scope', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[0, 2]
    ax.scatter(pc_whole, pc_up, s=22, color=COLORS[0], alpha=0.7, linewidths=0)
    lim = [np.nanmin([pc_whole, pc_up]), np.nanmax([pc_whole, pc_up])]
    ax.plot(lim, lim, color=INK_2, linewidth=1, label='equal')
    ax.set_xlabel('population coupling, whole block', fontsize=11)
    ax.set_ylabel('population coupling, UP states only', fontsize=11)
    ax.set_title(f'Per unit ({PC_BIN_SEC * 1e3:.0f} ms bins), '
                 f'ρ = {spearman(pc_whole, pc_up):.2f}', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    for col, (x, y, name) in enumerate((
            (rsc['whole'][k_w], rsc['up'][k_w], f'r_sc, {widths[k_w] * 1e3:g} ms bins'),
            (s_whole[k_d], s_obs[k_d], f'STTC, Δt = {dts[k_d] * 1e3:g} ms'))):
        ax = axes[1, col]
        ok = np.isfinite(x) & np.isfinite(y)
        lo = np.percentile(np.r_[x[ok], y[ok]], 0.5)
        hi = np.percentile(np.r_[x[ok], y[ok]], 99.5)
        hb = ax.hexbin(x[ok], y[ok], gridsize=70, extent=(lo, hi, lo, hi), bins='log',
                       cmap='Blues', mincnt=1, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, label='equal')
        ax.set_xlabel(f'{name} -- whole block', fontsize=11)
        ax.set_ylabel(f'{name} -- UP states only', fontsize=11)
        ax.set_title(f'Per pair ({ok.sum()} pairs), ρ = {spearman(x, y):.2f}',
                     fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
        plt.colorbar(hb, ax=ax, fraction=0.045, pad=0.02, label='pairs (log)')
        _style(ax)

    ax = axes[1, 2]
    for c, (metric, xs, name) in enumerate((('rsc', widths, 'r_sc (x = bin width)'),
                                            ('sttc', dts, 'STTC (x = Δt)'))):
        for other, style in (('whole', '-'), ('nrem', '--')):
            ax.plot(xs * 1e3, rank[(metric, other)], color=COLORS[c], linestyle=style,
                    marker='o', markersize=7, linewidth=2,
                    label=f"{name}: UP vs {'whole block' if other == 'whole' else 'all NREM'}")
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks([1, 2.5, 5, 10, 25, 50, 100])
    ax.set_xticklabels(['1', '2.5', '5', '10', '25', '50', '100'])
    ax.set_xlabel('timescale (ms)', fontsize=11)
    ax.set_ylabel('Spearman ρ across pairs', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Do the same pairs co-fire in UP states\nas over the whole period?',
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    save(fig, f"{base}_E_up_vs_whole_{stamp}.png")
    result['scopes'] = {'pop_coupling': pc_scope, 'rank_rho': rank}

    # ---------------- tables ------------------------------------------------
    pair_table = {'unit_i': np.array(ids, dtype=object)[iu[0]],
                  'unit_j': np.array(ids, dtype=object)[iu[1]],
                  'shank_i': shank[iu[0]], 'shank_j': shank[iu[1]],
                  'same_shank': same_pair.astype(float),
                  'abs_dy_um': np.abs(depth[iu[0]] - depth[iu[1]]),
                  'rate_up_i_hz': rate_up[iu[0]], 'rate_up_j_hz': rate_up[iu[1]]}
    for k, w in enumerate(widths):
        pair_table[f"rsc_up_{w * 1e3:g}ms"] = rsc['up'][k]
        pair_table[f"rsc_nrem_{w * 1e3:g}ms"] = rsc['nrem'][k]
        pair_table[f"rsc_upbaseline_{w * 1e3:g}ms"] = rsc['up_sur'][k]
        pair_table[f"rsc_whole_{w * 1e3:g}ms"] = rsc['whole'][k]
    for k, d in enumerate(dts):
        pair_table[f"sttc_{d * 1e3:g}ms"] = s_obs[k]
        pair_table[f"sttc_upbaseline_{d * 1e3:g}ms"] = s_sur[k]
        pair_table[f"sttc_nrem_{d * 1e3:g}ms"] = s_nrem[k]
        pair_table[f"sttc_whole_{d * 1e3:g}ms"] = s_whole[k]
    unit_table = {'unit': np.array(ids, dtype=object), 'shank': shank, 'depth_um': depth,
                  'n_spikes_up': np.bincount(up_axis['unit'], minlength=n),
                  'rate_up_hz': rate_up, 'rate_nrem_hz': nrem_rate,
                  'onset_latency_ms': latency, 'pop_coupling_nrem': pc_nrem,
                  'pop_coupling_up': pc_up, 'pop_coupling_whole': pc_whole}
    rt.write_csv(pair_table, f"{base}_pairs_{stamp}.csv")
    rt.write_csv(unit_table, f"{base}_units_{stamp}.csv")
    result['pair_table'], result['unit_table'] = pair_table, unit_table
    return result


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
    parser.add_argument('--out', default=None, help='default <sortout>/cofiring')
    parser.add_argument('--widths-ms', default='1,2,5,10,25,50,100',
                        help='bin widths for spike-count correlation')
    parser.add_argument('--sttc-ms', default='2.5,5,10,25,50,100', help='STTC Δt values')
    parser.add_argument('--ccg-pairs', default='all', choices=('all', 'cross', 'same'),
                        help='pairs pooled in the B figure (default all)')
    parser.add_argument('--ccg-surrogates', type=int, default=3,
                        help='surrogates per CCG baseline (default 3)')
    parser.add_argument('--check-surrogates', type=int, default=10)
    parser.add_argument('--min-shank-fraction', type=float, default=0.5)
    parser.add_argument('--min-alignment-ratio', type=float, default=1.25)
    parser.add_argument('--allow-alignment-failure', action='store_true')
    parser.add_argument('--min-up-ms', type=float, default=50.0)
    parser.add_argument('--max-up-ms', type=float, default=2000.0)
    parser.add_argument('--no-nrem', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    paths = rt.session_paths(args.animal, args.date, args.sortout)
    found, missing = rt.discover_inputs(paths)
    out_dir = Path(args.out) if args.out else paths['sortout'] / 'cofiring'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    set_stamp(sys.argv[1:] if argv is None else list(argv), paths, found)
    label = f"{paths['animal']} {paths['date']}"

    results = {}
    for block in [b.strip() for b in args.blocks.split(',') if b.strip()]:
        need = [k for k in (f'sleep_{block}', f'mua_{block}') if k not in found]
        if need:
            print(f"\n[{block}] skipped -- missing {', '.join(need)}")
            continue
        print("\n" + "=" * 72 + f"\nCO-FIRING, sleep {block.upper()}\n" + "=" * 72)
        try:
            results[block] = describe_block(block, found, args, out_dir, stamp, label)
        except ValueError as exc:
            print(f"[{block}] skipped -- {exc}")

    if not results:
        return 1
    out_pkl = out_dir / f"cofiring_results_{stamp}.pkl"
    with out_pkl.open('wb') as f:
        pickle.dump({'session': paths, 'inputs': {k: str(v) for k, v in found.items()},
                     'params': vars(args), 'blocks': results}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResults: {out_pkl}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
