r"""Sleep reactivation bias: NREM population vectors scored by the two awake KDEs.

    python pls_sleep_bias.py CnL46 260729                       # pre and post sleep
    python pls_sleep_bias.py CnL46 260729 --blocks post --view-min 5
    python pls_sleep_bias.py CnL46 260729 --view-start 1800     # zoom starts 1800 s into the block
    python pls_sleep_bias.py CnL46 260729 --awake-bin-ms 20     # awake model on 20 ms bins too

Awake model (pls_task_trials.py, in-sample on every trial): the stim-on bins of
the day's task runs (--awake-bin-ms, first --max-sec of each stim period), every
unit z-scored over those bins, PLSRegression against the left grating, and one
3-D Gaussian KDE per left grating on the scores of LV1-3.

Sleep: the sleep block's spikes (sleep_spikes_*sleep_<block>.pkl) are counted in
--bin-ms windows moved in --step-ms steps (default 20 ms / 10 ms, so neighbouring
windows share half their spikes) inside the scored NREM bouts
(<block>_sleep_periods.pkl, bout_intervals_s). Windows never straddle two bouts.
Every unit is z-scored with its own mean / sd over all NREM windows of the
block - not with the awake mean / sd, which belong to another bin width and
another state. Units with no spike in NREM get z = 0. Each window is projected
into the awake LV space and scored by both KDEs:

    R(t) = ln p(z_sleep(t) | 45 left) - ln p(z_sleep(t) | 315 left)

R > 0 = 45-left-like, R < 0 = 315-left-like (same sign as B in pls_task_trials).

Absolute likelihood: R says which awake distribution a window is closer to, not
whether it is close to either. That is

    ln D(t) = ln [ p(z | 45 left) + p(z | 315 left) ]

(and ln of the larger of the two, the density under the best awake distribution,
which is within ln 2 of ln D). The reference is the awake stim bins' own ln D,
leave-one-out: a bin's kernel is taken out of its own condition's KDE, otherwise
every awake bin sits on a density peak it made itself.

Support: windows whose ln D is below the SUPPORT_PCT percentile of the awake
leave-one-out ln D are flagged (in_support = False); far out there both
densities are ~0 and |R| is set by the tails.

Outputs, in <sortout>/pls_task/:
  pls_sleep_<sortout>_<block>_bin<ms>ms_step<ms>ms.png
  pls_sleep_<sortout>_<block>_bin<ms>ms_step<ms>ms_lv.png  the windows in the awake LV space, those with
                                                         |R| >= --large-r marked
  pls_sleep_<sortout>_<block>_bin<ms>ms_step<ms>ms.npz   time_sec (window centre, s from block start),
                                                         bout, bias, logp_45, logp_315, log_d, logp_best,
                                                         in_support, scores, awake_log_d (leave-one-out)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

GRATING_DIR = Path(__file__).resolve().parents[3] / 'DiscriminationTask' / 'grating'  # task_day_discovery lives here
REPO_ROOT = GRATING_DIR.parents[1]
for _p in (str(GRATING_DIR), str(REPO_ROOT / 'reactivation' / 'tuning_coupling'),
           str(REPO_ROOT / 'reactivation' / 'sleep_apply')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pls_task_trials import (COLORS, DEFAULT_MAX_SEC, KDE_DIMS, KDE_GRID, KDE_MASS_LEVELS, _mass_levels, fit_kdes,
                             fit_pls, marginal_kde, stack_runs, zscore)
from task_day_discovery import REGISTRY_FILE, load_registry, short_date
from sleep_states import load_nrem_intervals
from tc_upstates import load_sleep_spikes

SUPPORT_PCT = 1.0          # in_support: ln D above this percentile of the awake bins' leave-one-out ln D
VIEW_PCT = 99.5            # y range of the traces = this percentile of |R| over the in-support windows
FINE_SEC = 10.0            # length of the second, finer zoom
N_SCATTER = 20000          # sleep windows drawn in the LV scatter
DEFAULT_LARGE_R = 10.0     # |R| from which a window is marked in the LV-space figure
LV_VIEW_PCT = (0.1, 99.9)  # axis range of the LV-space figure = these percentiles of the scores


# ============================================================ awake model

def awake_model(pkl_files, bin_sec, max_sec, n_components):
    """PLS + the two KDEs fit on every stim bin of the day (no held-out fold:
    the sleep data is the test set)."""
    counts, unit_ids, rows = stack_runs(pkl_files, bin_sec, max_sec)
    stim = rows['fit']
    if set(np.unique(rows['left_ori']).tolist()) != set(COLORS):
        raise ValueError(f"left orientations are {np.unique(rows['left_ori']).tolist()}, expected {sorted(COLORS)}")
    x, kept = zscore(counts, stim)
    y = np.where(rows['left_ori'] == 45, 1.0, -1.0)[stim]
    pls, scores = fit_pls(x[stim], y, n_components)
    sign = np.sign((pls.transform(x[stim]) * scores).sum(axis=0))     # fit_pls flips LVs; sleep must follow
    kdes = fit_kdes(scores[:, :KDE_DIMS], y)
    return {'pls': pls, 'sign': sign, 'kdes': kdes, 'scores': scores, 'y': y,
            'unit_ids': [u for u, k in zip(unit_ids, kept) if k], 'n_units_all': len(unit_ids),
            'n_trials': np.unique(rows['trial'][stim]).size}


def score_points(model, z):
    """(scores, logp_45, logp_315) of z-scored population vectors (rows of z)."""
    scores = model['pls'].transform(z) * model['sign']
    pts = scores[:, :KDE_DIMS].T
    return scores, model['kdes'][1].logpdf(pts), model['kdes'][-1].logpdf(pts)


def awake_loo_logp(model):
    """(logp_45, logp_315) of the awake stim bins, each bin's own kernel removed
    from the KDE of its own condition."""
    pts = model['scores'][:, :KDE_DIMS].T
    out = {}
    for c, kde in model['kdes'].items():
        logp = kde.logpdf(pts)
        own = model['y'] == c
        peak = 1.0 / np.sqrt(np.linalg.det(2 * np.pi * kde.covariance))     # a kernel's value at its centre
        rest = (kde.n * np.exp(logp[own]) - peak) / (kde.n - 1)
        logp[own] = np.log(np.maximum(rest, 1e-300))
        out[c] = logp
    return out[1], out[-1]


# ============================================================ sleep windows

def sliding_counts(spikes, unit_ids, windows, bin_sec, step_sec):
    """Spike counts in bin_sec windows moved by step_sec inside each [start, end)
    of windows. Returns (counts (T x N), centre times, index of the window's bout)."""
    n_sub = int(round(bin_sec / step_sec))
    if n_sub < 1 or abs(n_sub * step_sec - bin_sec) > 1e-9:
        raise ValueError(f"bin {bin_sec * 1000:g} ms must be a whole multiple of step {step_sec * 1000:g} ms")
    all_counts, all_t, all_bout = [], [], []
    for b, (t0, t1) in enumerate(windows):
        n_step = int(np.floor((t1 - t0) / step_sec + 1e-9))
        if n_step < n_sub:
            continue
        fine = np.zeros((n_step + 1, len(unit_ids)), dtype=np.int32)     # row 0 stays 0 for the cumsum
        for u, uid in enumerate(unit_ids):
            train = spikes[uid]
            lo, hi = np.searchsorted(train, [t0, t0 + n_step * step_sec])
            idx = np.floor((train[lo:hi] - t0) / step_sec).astype(int)
            fine[1:, u] = np.bincount(idx[idx < n_step], minlength=n_step)
        cum = np.cumsum(fine, axis=0)
        counts = cum[n_sub:] - cum[:-n_sub]
        all_counts.append(counts)
        all_t.append(t0 + np.arange(counts.shape[0]) * step_sec + bin_sec / 2)
        all_bout.append(np.full(counts.shape[0], b))
    if not all_counts:
        raise ValueError("no NREM bout is long enough for one window")
    return np.concatenate(all_counts), np.concatenate(all_t), np.concatenate(all_bout)


def zscore_units(counts):
    """Every unit z-scored over the rows; units that never vary are set to 0."""
    mu, sd = counts.mean(axis=0), counts.std(axis=0)
    silent = sd == 0
    return (counts - mu) / np.where(silent, 1.0, sd), silent


def find_block_inputs(config, block):
    """(sleep spikes pkl, sleep periods pkl) of one block, or raises."""
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    real = lambda hits: sorted(p for p in hits if not p.name.startswith('.'))
    spikes = real(sortout.glob(f'sleep_spikes_*sleep_{block}.pkl'))
    if not spikes:
        raise FileNotFoundError(f"no sleep_spikes_*sleep_{block}.pkl in {sortout} - run "
                                f"extract_sleep_session_spikes_multishank.py")
    nrem = []
    for root in (Path(config['ephys_folder'].replace('\\', '/')), sortout):
        nrem += real(root.glob(f'**/*{block}_sleep_periods.pkl'))
    if not nrem:
        raise FileNotFoundError(f"no *{block}_sleep_periods.pkl under {config['ephys_folder']} - run "
                                f"sleep/staging/score_sleep_states.py")
    return spikes[0], nrem[0]


# ============================================================ figure

def _trace(ax, t, r, bout, ok, lim, t0, t1, unit=1.0):
    """R(t) in [t0, t1): red above 0, blue below, broken between bouts; windows
    outside the awake support get a grey tick under the trace."""
    m = (t >= t0) & (t < t1)
    tt, rr, bb = t[m] / unit, np.clip(r[m], -lim, lim), bout[m]
    for b in np.unique(bb):
        s = bb == b
        ax.fill_between(tt[s], 0, rr[s], where=rr[s] > 0, color=COLORS[45], alpha=0.6, lw=0, interpolate=True)
        ax.fill_between(tt[s], 0, rr[s], where=rr[s] < 0, color=COLORS[315], alpha=0.6, lw=0, interpolate=True)
        ax.plot(tt[s], rr[s], color='k', lw=0.4)
    out = ~ok[m]
    ax.plot(tt[out], np.full(out.sum(), -lim * 1.06), '|', color='0.5', ms=3, mew=0.5)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlim(t0 / unit, t1 / unit)
    ax.set_ylim(-lim * 1.12, lim * 1.12)
    ax.set_ylabel('R(t)')


def _density_trace(ax, t, log_d, bout, awake_log_d, ylim, t0, t1):
    """ln D(t) in [t0, t1), broken between bouts, against the awake bins' own."""
    m = (t >= t0) & (t < t1)
    for b in np.unique(bout[m]):
        s = m & (bout == b)
        ax.plot(t[s], np.clip(log_d[s], *ylim), color='k', lw=0.4)
    lo, med = np.percentile(awake_log_d, [SUPPORT_PCT, 50])
    ax.axhline(med, color='tab:green', lw=0.9, label='awake stim bins (leave-one-out): median')
    ax.axhline(lo, color='tab:green', lw=0.9, ls='--', label=f'their {SUPPORT_PCT:g} percentile = support limit')
    ax.set_xlim(t0, t1)
    ax.set_ylim(*ylim)
    ax.set_ylabel('ln D(t)')


def plot_sleep_bias(model, res, nrem, view, title, stamp, out):
    t, r, bout, ok = res['time_sec'], res['bias'], res['bout'], res['in_support']
    lim = float(np.percentile(np.abs(r[ok]), VIEW_PCT)) if ok.any() else float(np.percentile(np.abs(r), VIEW_PCT))
    v0, v1 = view
    f0, f1 = v0, min(v0 + FINE_SEC, v1)
    log_d, awake_log_d = res['log_d'], res['awake_log_d']
    d_lim = (float(np.percentile(log_d, 0.5)), float(max(log_d.max(), awake_log_d.max())) + 0.5)
    fig = plt.figure(figsize=(18, 19))
    grid = fig.add_gridspec(6, 3, height_ratios=[1, 1.3, 1, 1.3, 1, 1.6], hspace=0.5, wspace=0.2,
                            left=0.05, right=0.98, top=0.945, bottom=0.07)

    ax = fig.add_subplot(grid[0, :2])
    for a, b in nrem:
        ax.axvspan(a / 60, b / 60, color='0.92', lw=0, zorder=0)
    _trace(ax, t, r, bout, ok, lim, 0.0, res['duration_sec'], unit=60.0)
    ax.axvspan(v0 / 60, v1 / 60, facecolor='none', edgecolor='k', lw=1.0)
    ax.set_xlabel('time in the sleep block (min)')
    ax.set_title('whole block: R(t) inside the scored NREM bouts (grey); box = the zoom below', fontsize=9)

    ax = fig.add_subplot(grid[1, :])
    _trace(ax, t, r, bout, ok, lim, v0, v1)
    ax.axvspan(f0, f1, color='0.9', lw=0, zorder=0)
    ax.text(0.003, 0.96, '45\N{DEGREE SIGN} left \N{UPWARDS ARROW}', transform=ax.transAxes, va='top',
            fontsize=8, color=COLORS[45])
    ax.text(0.003, 0.04, '315\N{DEGREE SIGN} left \N{DOWNWARDS ARROW}', transform=ax.transAxes, va='bottom',
            fontsize=8, color=COLORS[315])
    ax.set_xlabel('time in the sleep block (s)')
    ax.set_title(f'{(v1 - v0) / 60:.1f} min of NREM (grey band = the {f1 - f0:g} s below; grey ticks = window '
                 f'outside the awake support)', fontsize=9)

    ax = fig.add_subplot(grid[2, :])
    _density_trace(ax, t, log_d, bout, awake_log_d, d_lim, v0, v1)
    ax.legend(fontsize=7, frameon=False, loc='lower right', ncol=2)
    ax.set_xlabel('time in the sleep block (s)')
    ax.set_title('same stretch: absolute likelihood  ln D(t) = ln [ p(z | 45\N{DEGREE SIGN} left) + '
                 'p(z | 315\N{DEGREE SIGN} left) ]', fontsize=9)

    ax = fig.add_subplot(grid[3, :])
    _trace(ax, t, r, bout, ok, lim, f0, f1)
    ax.set_xlabel('time in the sleep block (s)')
    ax.set_title(f'{f1 - f0:g} s at full resolution: R(t)', fontsize=9)

    ax = fig.add_subplot(grid[4, :])
    _density_trace(ax, t, log_d, bout, awake_log_d, d_lim, f0, f1)
    ax.set_xlabel('time in the sleep block (s)')
    ax.set_title(f'{f1 - f0:g} s at full resolution: ln D(t)', fontsize=9)

    ax = fig.add_subplot(grid[5, 1])
    edges = np.linspace(d_lim[0], d_lim[1], 81)
    ax.hist(np.clip(log_d, *d_lim), bins=edges, density=True, color='0.6', label=f'NREM windows ({log_d.size})')
    ax.hist(np.clip(awake_log_d, *d_lim), bins=edges, density=True, histtype='step', color='tab:green', lw=1.3,
            label=f'awake stim bins, leave-one-out ({awake_log_d.size})')
    ax.axvline(res['log_d_silent'], color='k', lw=1.0, ls='--',
               label=f"window with no spike at all: {res['log_d_silent']:.2f}")
    ax.set_xlabel(f'ln D  (clipped below at {d_lim[0]:.1f})')
    ax.set_ylabel('density')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title(f"absolute likelihood: NREM median {np.median(log_d):.2f}, awake median "
                 f"{np.median(awake_log_d):.2f}\n{1 - ok.mean():.1%} of NREM below the awake "
                 f"{SUPPORT_PCT:g} percentile", fontsize=9)

    ax = fig.add_subplot(grid[5, 2])
    hb = ax.hexbin(np.clip(log_d, *d_lim), np.clip(r, -lim, lim), gridsize=60, bins='log', cmap='Greys',
                   extent=(d_lim[0], d_lim[1], -lim, lim), mincnt=1)
    fig.colorbar(hb, ax=ax, label='windows')
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(np.percentile(awake_log_d, SUPPORT_PCT), color='tab:green', lw=0.9, ls='--')
    ax.set_xlabel('ln D  (green: support limit)')
    ax.set_ylabel('R')
    ax.set_title(f"R against absolute likelihood: median |R| {res['abs_r_low']:.2f} in the least likely\n"
                 f"tenth of windows, {res['abs_r_high']:.2f} in the most likely tenth", fontsize=9)

    ax = fig.add_subplot(grid[5, 0])
    edges = np.linspace(-lim, lim, 81)
    ax.hist(np.clip(r, -lim, lim), bins=edges, color='0.75', label=f'all NREM windows ({r.size})')
    ax.hist(np.clip(r[ok], -lim, lim), bins=edges, color='0.3', label=f'inside the awake support ({ok.sum()})')
    ax.axvline(0, color='k', lw=0.6)
    ax.axvline(res['bias_silent'], color='tab:green', lw=1.0, ls='--',
               label=f"window with no spike at all: R = {res['bias_silent']:+.2f}")
    ax.set_yscale('log')
    ax.set_xlabel(f'R  (clipped at \N{PLUS-MINUS SIGN}{lim:.1f})')
    ax.set_ylabel('windows')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"R over NREM: median {np.median(r):+.2f}, {np.mean(r > 0):.0%} of windows > 0\n"
                 f"in support: median {np.median(r[ok]) if ok.any() else np.nan:+.2f}, "
                 f"{np.mean(r[ok] > 0) if ok.any() else np.nan:.0%} > 0", fontsize=9)

    ax = fig.add_subplot(grid[0, 2])
    pick = np.random.default_rng(0).choice(r.size, min(N_SCATTER, r.size), replace=False)
    ax.scatter(res['scores'][pick, 0], res['scores'][pick, 1], s=2, color='0.6', alpha=0.3, lw=0,
               label=f'NREM windows ({pick.size} drawn)', rasterized=True)
    for c, ori in ((1, 45), (-1, 315)):
        m = model['y'] == c
        ax.scatter(model['scores'][m, 0], model['scores'][m, 1], s=4, color=COLORS[ori], alpha=0.5, lw=0,
                   label=f'awake stim bins, left {ori}\N{DEGREE SIGN}')
    both = np.vstack([res['scores'][:, :2], model['scores'][:, :2]])     # a few windows land tens of units away
    lo, hi = np.percentile(both, [0.1, 99.9], axis=0)
    ax.set_xlim(lo[0] - 0.1 * (hi[0] - lo[0]), hi[0] + 0.1 * (hi[0] - lo[0]))
    ax.set_ylim(lo[1] - 0.1 * (hi[1] - lo[1]), hi[1] + 0.1 * (hi[1] - lo[1]))
    ax.set_xlabel('LV1')
    ax.set_ylabel('LV2  (view: 0.1-99.9 percentiles)')
    ax.legend(fontsize=7, frameon=False, markerscale=3)
    ax.set_title(f'sleep windows in the awake LV space  ({ok.mean():.0%} inside the awake support)', fontsize=9)

    fig.suptitle(title, fontsize=11, x=0.01, ha='left', y=0.99)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_lv_space(model, res, large_r, title, stamp, out):
    """NREM windows in the awake LV space with the large-|R| windows marked.

    Top row: every pair of LVs - all windows grey, the awake KDEs as contours,
    the windows with |R| >= large_r coloured by the sign of R (hollow = outside
    the awake support). Bottom row: the same planes coloured by R itself."""
    scores, r, ok = res['scores'][:, :KDE_DIMS], res['bias'], res['in_support']
    big = np.abs(r) >= large_r
    names = {1: 45, -1: 315}
    lo, hi = np.percentile(np.vstack([scores, model['scores'][:, :KDE_DIMS]]), LV_VIEW_PCT, axis=0)
    pad = 0.15 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    pick = np.random.default_rng(0).choice(r.size, min(N_SCATTER, r.size), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        ga, gb = np.linspace(lo[a], hi[a], KDE_GRID), np.linspace(lo[b], hi[b], KDE_GRID)
        mesh_a, mesh_b = np.meshgrid(ga, gb)
        grid = np.vstack([mesh_a.ravel(), mesh_b.ravel()])
        seen = ((scores[:, a] >= lo[a]) & (scores[:, a] <= hi[a]) & (scores[:, b] >= lo[b]) & (scores[:, b] <= hi[b]))

        ax = axes[0, col]
        ax.scatter(scores[pick, a], scores[pick, b], s=2, color='0.75', alpha=0.4, lw=0, rasterized=True)
        for c, ori in names.items():
            dens = marginal_kde(model['kdes'][c], (a, b))(grid).reshape(mesh_a.shape)
            ax.contour(mesh_a, mesh_b, dens, levels=_mass_levels(dens, KDE_MASS_LEVELS), colors=[COLORS[ori]],
                       linewidths=[0.6, 1.0, 1.4])
        for sign, ori in ((1, 45), (-1, 315)):
            m = big & (np.sign(r) == sign)
            ax.scatter(scores[m & ok, a], scores[m & ok, b], s=7, color=COLORS[ori], alpha=0.7, lw=0, rasterized=True)
            ax.scatter(scores[m & ~ok, a], scores[m & ~ok, b], s=9, facecolor='none', edgecolor=COLORS[ori],
                       lw=0.5, alpha=0.8, rasterized=True)
        ax.set_title(f'LV{a + 1}-LV{b + 1}:  {(big & seen).sum()} of the {big.sum()} large-|R| windows in view',
                     fontsize=9)

        ax = axes[1, col]
        order = pick[np.argsort(np.abs(r[pick]))]              # large |R| drawn last, on top
        sc = ax.scatter(scores[order, a], scores[order, b], s=3, c=r[order], cmap='RdBu_r', vmin=-large_r,
                        vmax=large_r, lw=0, rasterized=True)
        fig.colorbar(sc, ax=ax, extend='both', label='R')
        ax.set_title(f'same plane, {pick.size} windows coloured by R', fontsize=9)
        for ax in axes[:, col]:
            ax.set_xlim(lo[a], hi[a])
            ax.set_ylim(lo[b], hi[b])
            ax.set_xlabel(f'LV{a + 1}')
            ax.set_ylabel(f'LV{b + 1}')
    ax = axes[0, 0]
    ax.scatter([], [], s=8, color='0.75', label=f'NREM windows ({pick.size} of {r.size} drawn)')
    for sign, ori in ((1, 45), (-1, 315)):
        m = big & (np.sign(r) == sign)
        bound = '\N{GREATER-THAN OR EQUAL TO} +' if sign > 0 else '\N{LESS-THAN OR EQUAL TO} \N{MINUS SIGN}'
        ax.scatter([], [], s=14, color=COLORS[ori],
                   label=f"R {bound}{large_r:g}  ({m.sum()} windows, {(m & ~ok).sum()} outside the awake support = hollow)")
        ax.plot([], [], color=COLORS[ori], lw=1.2, label=f'awake KDE, left {ori}\N{DEGREE SIGN}')
    ax.legend(fontsize=7, frameon=False, loc='upper left', title='contours hold ' + ' / '.join(
        f'{m:.0%}' for m in sorted(KDE_MASS_LEVELS)) + ' of the awake mass', title_fontsize=7)
    fig.suptitle(f"{title}\nlarge |R|: {big.sum()} windows ({big.mean():.1%}) with |R| \N{GREATER-THAN OR EQUAL TO} "
                 f"{large_r:g};  {np.mean(~ok[big]) if big.any() else np.nan:.0%} of them outside the awake support, "
                 f"against {np.mean(~ok[~big]):.1%} of the other windows", fontsize=11, x=0.01, ha='left', y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.11, wspace=0.2, hspace=0.25)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ============================================================ driver

def run_block(model, config, block, bin_sec, step_sec, view_min, view_start, large_r, title, stamp, out_dir,
              log=print):
    sleep_pkl, nrem_pkl = find_block_inputs(config, block)
    spikes, meta = load_sleep_spikes(sleep_pkl, verbose=False)
    missing = [u for u in model['unit_ids'] if u not in spikes]
    if missing:
        raise ValueError(f"{sleep_pkl.name}: {len(missing)} awake units are absent (e.g. {missing[:3]}) - the "
                         f"sleep and task pkls come from different sorts")
    nrem = load_nrem_intervals(nrem_pkl, require_bouts=True)
    nrem = np.clip(nrem, 0.0, meta['duration_sec'])
    nrem = nrem[nrem[:, 1] > nrem[:, 0]]
    counts, t, bout = sliding_counts(spikes, model['unit_ids'], nrem, bin_sec, step_sec)
    z, silent = zscore_units(counts)
    scores, lp45, lp315 = score_points(model, z)
    bias = lp45 - lp315

    log_d = np.logaddexp(lp45, lp315)
    awake_log_d = np.logaddexp(*awake_loo_logp(model))
    in_support = log_d >= np.percentile(awake_log_d, SUPPORT_PCT)
    d_lo, d_hi = np.percentile(log_d, [10, 90])
    z_silent = ((0 - counts.mean(axis=0)) / np.where(silent, 1.0, counts.std(axis=0)))[None]
    _, s45, s315 = score_points(model, z_silent)
    res = {'time_sec': t, 'bout': bout, 'bias': bias, 'logp_45': lp45, 'logp_315': lp315,
           'log_d': log_d, 'logp_best': np.maximum(lp45, lp315), 'awake_log_d': awake_log_d,
           'in_support': in_support, 'scores': scores, 'bias_silent': float(s45[0] - s315[0]),
           'log_d_silent': float(np.logaddexp(s45[0], s315[0])),
           'abs_r_low': float(np.median(np.abs(bias[log_d <= d_lo]))),
           'abs_r_high': float(np.median(np.abs(bias[log_d >= d_hi]))),
           'duration_sec': meta['duration_sec']}

    nrem_sec = float(np.sum(nrem[:, 1] - nrem[:, 0]))
    log(f"  [{block}] {len(nrem)} NREM bouts, {nrem_sec / 60:.1f} min of a {meta['duration_sec'] / 60:.1f} min "
        f"block -> {t.size} windows x {z.shape[1]} units ({silent.sum()} silent in NREM, z = 0); "
        f"mean spikes per window {counts.sum(axis=1).mean():.1f}")
    log(f"    R: median {np.median(bias):+.2f}, 1-99 percentiles {np.percentile(bias, 1):+.1f} .. "
        f"{np.percentile(bias, 99):+.1f}, {np.mean(bias > 0):.1%} of windows > 0; a window with no spike "
        f"gives R = {res['bias_silent']:+.2f}")
    log(f"    ln D = ln(p45 + p315): NREM median {np.median(log_d):.2f} (1-99 percentiles "
        f"{np.percentile(log_d, 1):.1f} .. {np.percentile(log_d, 99):.1f}), awake leave-one-out median "
        f"{np.median(awake_log_d):.2f} ({np.percentile(awake_log_d, 1):.1f} .. {np.percentile(awake_log_d, 99):.1f}); "
        f"no-spike window {res['log_d_silent']:.2f}; {np.mean(log_d > np.median(awake_log_d)):.1%} of NREM windows "
        f"above the awake median")
    log(f"    median |R|: {res['abs_r_low']:.2f} in the least likely tenth of windows, {res['abs_r_high']:.2f} in "
        f"the most likely tenth")
    log(f"    inside the awake support: {in_support.mean():.1%} of windows"
        + (f"; there R median {np.median(bias[in_support]):+.2f}, {np.mean(bias[in_support] > 0):.1%} > 0"
           if in_support.any() else ''))

    longest = nrem[np.argmax(nrem[:, 1] - nrem[:, 0])]
    v0 = longest[0] if view_start is None else float(view_start)
    in_bout = nrem[(nrem[:, 0] <= v0) & (nrem[:, 1] > v0)]
    v1 = min(v0 + view_min * 60.0, in_bout[0, 1] if len(in_bout) else v0 + view_min * 60.0)

    tag = f"{block}_bin{bin_sec * 1000:g}ms_step{step_sec * 1000:g}ms"
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    out = out_dir / f"pls_sleep_{sortout.name}_{tag}.png"
    plot_sleep_bias(model, res, nrem, (v0, v1),
                    f"{title}\nsleep {block.upper()}: R(t) = ln p(z | 45\N{DEGREE SIGN} left) / p(z | "
                    f"315\N{DEGREE SIGN} left),  {bin_sec * 1000:g} ms windows every {step_sec * 1000:g} ms in "
                    f"{nrem_sec / 60:.1f} min of scored NREM ({t.size} windows)",
                    stamp + f"\nsleep inputs: {sleep_pkl} ; {nrem_pkl} (bout_intervals_s)\n"
                            f"sleep: counts z-scored per unit over all NREM windows of the block ({silent.sum()} units "
                            f"silent in NREM set to 0); in support = ln D >= {SUPPORT_PCT:g} percentile of "
                            f"the awake bins' leave-one-out ln D; traces clipped at the {VIEW_PCT:g} percentile of |R| in support; "
                            f"zoom {v0:.0f}-{v1:.0f} s", out)
    big = np.abs(bias) >= large_r
    lv_out = out.with_name(out.stem + '_lv.png')
    plot_lv_space(model, res, large_r,
                  f"{title}\nsleep {block.upper()}: {t.size} NREM windows ({bin_sec * 1000:g} ms every "
                  f"{step_sec * 1000:g} ms) projected into the awake LV space",
                  stamp + f"\nsleep inputs: {sleep_pkl} ; {nrem_pkl} (bout_intervals_s)\n"
                          f"sleep: counts z-scored per unit over all NREM windows of the block; large = |R| >= {large_r:g}; "
                          f"hollow = ln D below the {SUPPORT_PCT:g} percentile of the awake bins' leave-one-out ln D; view = "
                          f"{LV_VIEW_PCT} percentiles of the sleep + awake scores (+ 15 %); bottom row colour clipped at "
                          f"+-{large_r:g}", lv_out)
    log(f"    large |R| (>= {large_r:g}): {big.sum()} windows ({big.mean():.1%}), {np.mean(bias[big] > 0):.0%} of them "
        f"R > 0, {np.mean(~in_support[big]):.0%} outside the awake support; median distance from the origin in "
        f"LV1-{KDE_DIMS}: {np.median(np.linalg.norm(scores[big, :KDE_DIMS], axis=1)):.2f} against "
        f"{np.median(np.linalg.norm(scores[~big, :KDE_DIMS], axis=1)):.2f} for the other windows")
    np.savez_compressed(out.with_suffix('.npz'), unit_ids=np.array(model['unit_ids']), nrem_windows_sec=nrem,
                        bin_sec=bin_sec, step_sec=step_sec, large_r=large_r, **res)
    log(f"    saved {out}\n    saved {lv_out}\n    saved {out.with_suffix('.npz')}")
    return out


def run_day(animal, date, blocks=('pre', 'post'), bin_ms=20.0, step_ms=10.0, awake_bin_ms=100.0,
            n_components=3, max_sec=DEFAULT_MAX_SEC, view_min=3.0, view_start=None, large_r=DEFAULT_LARGE_R,
            out_dir=None, log=print):
    key = f"{animal}_{short_date(date)}"
    config = load_registry().get(key)
    if config is None:
        raise KeyError(f"{key} is not in {REGISTRY_FILE} - register the day with task_day_gui.py")
    if n_components < KDE_DIMS:
        raise ValueError(f"the KDE needs --n-components >= {KDE_DIMS}")
    pkl_files = [t['trial_pkl'].replace('\\', '/') for t in config.get('tasks', []) if t.get('trial_pkl')]
    pkl_files = [p for p in pkl_files if Path(p).is_file()]
    if not pkl_files:
        raise FileNotFoundError(f"{key}: no trial pkl on disk - run extract_task_day.py first")

    model = awake_model(pkl_files, awake_bin_ms / 1000.0, max_sec, n_components)
    log(f"{key}: awake model from {len(pkl_files)} run(s), {model['n_trials']} trials, {model['y'].size} stim bins "
        f"of {awake_bin_ms:g} ms x {len(model['unit_ids'])} units; {n_components} LVs, KDE on LV1-{KDE_DIMS}")

    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    out_dir = Path(out_dir) if out_dir else sortout / 'pls_task'
    out_dir.mkdir(parents=True, exist_ok=True)
    window = 'whole stim period' if max_sec is None else f'first {max_sec:g} s of each stim period'
    title = (f"{key}  awake PLS ({awake_bin_ms:g} ms bins, {window}, {model['n_trials']} trials, "
             f"{len(model['unit_ids'])} units, {n_components} LVs) + one 3-D KDE per left grating")
    stamp = (f"Repro: python {Path(__file__).name} {animal} {short_date(date)} --blocks {','.join(blocks)} "
             f"--bin-ms {bin_ms:g} --step-ms {step_ms:g} --awake-bin-ms {awake_bin_ms:g} --n-components "
             f"{n_components} --max-sec {max_sec or 0:g} --view-min {view_min:g} --large-r {large_r:g}"
             + (f" --view-start {view_start:g}" if view_start is not None else '')
             + f" | {datetime.now().isoformat(timespec='seconds')}\n"
             f"awake inputs: {' ; '.join(pkl_files)}\n"
             f"awake: all {model['n_units_all']} units of the pkl (uncurated), silent ones dropped; z-score per unit "
             f"over the stim bins; PLSRegression(scale=False), y = +1 (45 left) / -1 (315 left), fit on every trial; "
             f"KDE = scipy gaussian_kde, Scott bandwidth, on LV1-{KDE_DIMS}; natural log")
    outs = []
    for block in blocks:
        try:
            outs.append(run_block(model, config, block, bin_ms / 1000.0, step_ms / 1000.0, view_min, view_start,
                                  large_r, title, stamp, out_dir, log))
        except FileNotFoundError as exc:
            log(f"  [{block}] skipped - {exc}")
    return outs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('animal')
    parser.add_argument('date', help='YYMMDD')
    parser.add_argument('--blocks', default='pre,post', help='sleep blocks (default: %(default)s)')
    parser.add_argument('--bin-ms', type=float, default=20.0, help='sleep window (default: %(default)s)')
    parser.add_argument('--step-ms', type=float, default=10.0, help='sleep window step (default: %(default)s)')
    parser.add_argument('--awake-bin-ms', type=float, default=100.0,
                        help='bin of the awake PLS / KDE model (default: %(default)s, as pls_task_trials.py)')
    parser.add_argument('--n-components', type=int, default=3)
    parser.add_argument('--max-sec', type=float, default=DEFAULT_MAX_SEC,
                        help='awake model: use at most this much of each stim period; 0 = all of it '
                             '(default: %(default)s)')
    parser.add_argument('--view-min', type=float, default=3.0, help='length of the zoomed trace, min')
    parser.add_argument('--view-start', type=float, default=None,
                        help='start of the zoomed trace, s from block start (default: start of the longest bout)')
    parser.add_argument('--large-r', type=float, default=DEFAULT_LARGE_R,
                        help='windows with |R| >= this are marked in the LV-space figure (default: %(default)s)')
    parser.add_argument('--out-dir', default=None, help='default: <sortout>/pls_task')
    args = parser.parse_args(argv)
    blocks = tuple(b.strip() for b in args.blocks.split(',') if b.strip())
    outs = run_day(args.animal, args.date, blocks, args.bin_ms, args.step_ms, args.awake_bin_ms,
                   args.n_components, args.max_sec or None, args.view_min, args.view_start, args.large_r, args.out_dir)
    return 0 if outs else 1


if __name__ == '__main__':
    sys.exit(main())
