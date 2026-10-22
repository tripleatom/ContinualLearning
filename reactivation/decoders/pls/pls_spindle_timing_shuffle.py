r"""Independent-neuron timing-shuffle control for spindle-aligned PLS replay.

Each unit's spikes are circularly shifted by an independent random offset
inside every NREM bout.  The awake PLS transform and KDEs are never refit.
This preserves each unit's NREM spike count and within-bout temporal pattern,
but destroys precise cross-neuron timing and hence coordinated population
patterns.  Spindle events, peak times, and their across-shank merge are fixed.

Outputs per sleep block:
  *_pls.png       observed and shuffled peak-bin activity in the awake LV space
  *_heatmap.png   spindle-peak heatmaps, traces, ranked values, permutation nulls
  *.npz           observed, representative-shuffle, and permutation results
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

GRATING_DIR = Path(__file__).resolve().parents[3] / 'DiscriminationTask' / 'grating'  # task_day_discovery lives here
if str(GRATING_DIR) not in sys.path:
    sys.path.insert(0, str(GRATING_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from pls_sleep_bias import (SUPPORT_PCT, awake_loo_logp, awake_model, load_nrem_intervals,
                            load_sleep_spikes, score_points, sliding_counts)
from pls_spindle_bias import load_spindles
from pls_task_trials import BIAS_CLIP, COLORS, DEFAULT_MAX_SEC, KDE_DIMS
from task_day_discovery import REGISTRY_FILE, load_registry, short_date


def shift_units_within_nrem(spikes, unit_ids, nrem, rng, min_shift_sec):
    """Circularly shift each unit independently in each NREM bout."""
    shifted = {}
    for uid in unit_ids:
        source = np.asarray(spikes[uid], dtype=float)
        target = source.copy()
        for start, end in nrem:
            lo, hi = np.searchsorted(source, [start, end])
            if hi <= lo:
                continue
            duration = end - start
            if duration <= 2 * min_shift_sec:
                delta = duration / 2
            else:
                delta = rng.uniform(min_shift_sec, duration - min_shift_sec)
            target[lo:hi] = start + np.mod(source[lo:hi] - start + delta, duration)
        shifted[uid] = np.sort(target)
    return shifted


def decode_events(model, spikes, anchors, mu, sd, half_sec, bin_sec, step_sec):
    windows = np.column_stack([anchors - half_sec - bin_sec / 2,
                               anchors + half_sec + bin_sec / 2])
    counts, _, event = sliding_counts(spikes, model['unit_ids'], windows, bin_sec, step_sec)
    n_lag = counts.shape[0] // len(anchors)
    if counts.shape[0] != n_lag * len(anchors):
        raise RuntimeError('event windows produced unequal lag counts')
    z = (counts - mu) / sd
    scores, lp45, lp315 = score_points(model, z)
    shape = (len(anchors), n_lag)
    return {
        'bias': (lp45 - lp315).reshape(shape),
        'log_best': np.maximum(lp45, lp315).reshape(shape),
        'log_d': np.logaddexp(lp45, lp315).reshape(shape),
        'scores': scores.reshape(len(anchors), n_lag, -1),
    }


def shuffle_statistics(decoded, near, support_cut):
    event_strength = np.median(np.abs(decoded['bias'][:, near]), axis=1)
    event_log_best = np.median(decoded['log_best'][:, near], axis=1)
    return (float(np.median(event_strength)), float(np.median(event_log_best)),
            float(np.mean(decoded['log_d'][:, near] >= support_cut)))


def smooth_bias(bias, smooth_ms, step_sec):
    if smooth_ms <= 0:
        return bias
    sigma = (smooth_ms / 1000) / (2 * np.sqrt(2 * np.log(2))) / step_sec
    return gaussian_filter1d(bias, sigma, axis=1, mode='nearest')


def p_upper(observed, null):
    return (1 + np.count_nonzero(np.asarray(null) >= observed)) / (len(null) + 1)


def plot_pls(model, observed, shuffled, zero, title, stamp, out):
    awake = model['scores'][:, :KDE_DIMS]
    obs = observed['scores'][:, zero, :KDE_DIMS]
    shuf = shuffled['scores'][:, zero, :KDE_DIMS]
    obs_r = observed['bias'][:, zero]
    shuf_r = shuffled['bias'][:, zero]
    all_scores = np.vstack([awake, obs, shuf])
    lo, hi = np.percentile(all_scores, [0.2, 99.8], axis=0)
    pad = np.maximum(0.1 * (hi - lo), 1e-6)
    lo, hi = lo - pad, hi + pad
    pairs = ((0, 1), (0, 2), (1, 2))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for row, (data, r, name) in enumerate(((obs, obs_r, 'observed spindle peaks'),
                                           (shuf, shuf_r, 'one independent timing shuffle'))):
        for col, (a, b) in enumerate(pairs):
            ax = axes[row, col]
            for sign, ori in ((1, 45), (-1, 315)):
                m = model['y'] == sign
                ax.scatter(awake[m, a], awake[m, b], s=6, color=COLORS[ori], alpha=0.16,
                           lw=0, rasterized=True, label=f'awake {ori}\N{DEGREE SIGN}' if col == 0 else None)
            sc = ax.scatter(data[:, a], data[:, b], c=np.clip(r, -BIAS_CLIP, BIAS_CLIP),
                            cmap='RdBu_r', vmin=-BIAS_CLIP, vmax=BIAS_CLIP,
                            s=13, alpha=0.8, lw=0, rasterized=True)
            ax.set(xlim=(lo[a], hi[a]), ylim=(lo[b], hi[b]),
                   xlabel=f'LV{a + 1}', ylabel=f'LV{b + 1}',
                   title=f'{name}: LV{a + 1} vs LV{b + 1}')
            if col == 0:
                ax.legend(fontsize=7, frameon=False)
    cax = fig.add_axes([0.925, 0.18, 0.014, 0.64])
    fig.colorbar(sc, cax=cax, extend='both', label='R at spindle peak')
    fig.suptitle(title, x=0.01, ha='left', y=0.99, fontsize=11)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.subplots_adjust(left=0.06, right=0.9, top=0.91, bottom=0.07, wspace=0.24, hspace=0.3)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_heatmap(observed, representative, lag, order, null_trace, null_rank,
                 null_strength, null_log_best, null_support, observed_stats,
                 smooth_ms, step_sec, title, stamp, out, shuffle_order=None):
    independently_ranked = shuffle_order is not None
    if shuffle_order is None:
        shuffle_order = order
    obs = smooth_bias(observed['bias'], smooth_ms, step_sec)[order]
    shuf = smooth_bias(representative['bias'], smooth_ms, step_sec)[shuffle_order]
    n = len(order)
    step = float(np.median(np.diff(lag)))
    extent = (lag[0] - step / 2, lag[-1] + step / 2, n - 0.5, -0.5)
    fig = plt.figure(figsize=(18, 11.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(2.2, 1), hspace=0.28, wspace=0.24,
                          left=0.055, right=0.98, top=0.91, bottom=0.1)
    shuffle_name = ('representative timing shuffle, independently zero-lag ranked'
                    if independently_ranked else 'representative timing shuffle')
    for col, (matrix, name) in enumerate(((obs, 'observed'), (shuf, shuffle_name))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-BIAS_CLIP, vmax=BIAS_CLIP,
                       interpolation='nearest', extent=extent)
        ax.axvline(0, color='k', lw=0.8)
        ylabel = ('observed zero-lag rank' if col == 0 else
                  ('shuffled zero-lag rank' if independently_ranked else 'observed zero-lag rank'))
        ax.set(xlabel='time from spindle peak (s)', ylabel=ylabel,
               title=f'{name}: {n} spindle events')
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.11,
                     extend='both', label='R (100 ms display smoothing)' if smooth_ms == 100 else 'R')

    ax = fig.add_subplot(gs[0, 2])
    shown_obs = np.median(smooth_bias(observed['bias'], smooth_ms, step_sec), axis=0)
    shown_null = np.array([smooth_bias(x[None], smooth_ms, step_sec)[0] for x in null_trace])
    ax.fill_between(lag, *np.percentile(shown_null, [2.5, 97.5], axis=0), color='0.8',
                    label='timing shuffle: 95% envelope')
    ax.plot(lag, np.median(shown_null, axis=0), color='0.4', lw=1, label='shuffle median')
    ax.plot(lag, shown_obs, color='tab:purple', lw=1.5, label='observed median')
    ax.axvline(0, color='k', lw=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='time from spindle peak (s)', ylabel='median R', title='spindle-triggered population trace')
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    rank = np.arange(1, n + 1)
    ax.fill_between(rank, *np.percentile(null_rank, [2.5, 97.5], axis=0), color='0.8',
                    label='shuffle: 95% of ranked values')
    ax.plot(rank, np.median(null_rank, axis=0), color='0.45', lw=1)
    values = observed['bias'][:, np.argmin(np.abs(lag))][order]
    ax.scatter(rank, values, s=9, c=np.where(values >= 0, COLORS[45], COLORS[315]), label='observed')
    ax.axhline(0, color='k', lw=0.6)
    lim = np.percentile(np.abs(np.r_[values, null_rank.ravel()]), 99.5)
    ax.set_ylim(-lim, lim)
    ax.set(xlabel='rank', ylabel='raw R in zero-lag 20 ms bin', title='ranked spindle-peak values')
    ax.legend(fontsize=7, frameon=False)

    labels = ('median event |R| within +/-50 ms', 'median awake log-density within +/-50 ms')
    nulls = (null_strength, null_log_best)
    for col, index in ((1, 0), (2, 1)):
        ax = fig.add_subplot(gs[1, col])
        values_null = np.asarray(nulls[index])
        ax.hist(values_null, bins=min(25, max(5, len(values_null) // 4)), color='0.75', edgecolor='none')
        ax.axvline(observed_stats[index], color='tab:purple', lw=2, label='observed')
        extra = ''
        if col == 2:
            extra = (f"awake-support fraction\nobserved {observed_stats[2]:.1%}; "
                     f"shuffle median {np.median(null_support):.1%}; "
                     f"p={p_upper(observed_stats[2], null_support):.3g}")
        ax.set(xlabel=labels[index], ylabel='shuffle count',
               title=f'one-sided p={p_upper(observed_stats[index], values_null):.3g}')
        if extra:
            ax.text(0.02, 0.97, extra, transform=ax.transAxes, ha='left', va='top', fontsize=8)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(title, x=0.01, ha='left', y=0.985, fontsize=11)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_saved_shuffled_ranked(npz_path, title=None, out=None):
    """Create the third figure from an existing timing-shuffle NPZ."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as z:
        observed = {'bias': z['observed_bias']}
        representative = {'bias': z['representative_shuffle_bias']}
        lag = z['lag_sec']
        order = z['order']
        zero = int(np.argmin(np.abs(lag)))
        shuffle_order = np.argsort(-representative['bias'][:, zero])
        observed_stats = (float(z['observed_strength']), float(z['observed_log_best']),
                          float(z['observed_support_fraction']))
        smooth_ms = float(z['display_smooth_ms'])
        step_sec = float(z['step_sec'])
        if title is None:
            title = npz_path.stem + '\nrepresentative shuffle independently ranked by its own zero-lag R'
        if out is None:
            out = npz_path.with_name(npz_path.stem + '_heatmap_shuffled_ranked.png')
        plot_heatmap(observed, representative, lag, order, z['null_median_trace'],
                     z['null_rank_zero'], z['null_strength'], z['null_log_best'],
                     z['null_support_fraction'], observed_stats, smooth_ms, step_sec,
                     title, f'Replotted from {npz_path}', out, shuffle_order=shuffle_order)
    return Path(out)


def run_block(model, config, block, n_shuffle, n_trace_shuffle, half_sec, bin_sec, step_sec,
              near_sec, min_shift_sec, smooth_ms, seed, title, stamp, out_dir, spindle_source='legacy', log=print):
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    spindle_dir = Path(config['ephys_folder'].replace('\\', '/')) / 'low_freq' / 'v1_spindles'
    events, n_shanks, csv_files, n_raw = load_spindles(spindle_dir, block, None, spindle_source)
    nrem_files = sorted(p for p in spindle_dir.glob(f'*_{block}_sleep_periods.pkl') if not p.name.startswith('.'))
    sleep_files = sorted(p for p in sortout.glob(f'sleep_spikes_*sleep_{block}.pkl') if not p.name.startswith('.'))
    if not nrem_files or not sleep_files:
        raise FileNotFoundError(f'{block}: missing spindle NREM or sleep-spike input')
    spikes, meta = load_sleep_spikes(sleep_files[0], verbose=False)
    missing = [u for u in model['unit_ids'] if u not in spikes]
    if missing:
        raise ValueError(f'{len(missing)} awake units missing from sleep spikes')
    nrem = load_nrem_intervals(nrem_files[0], require_bouts=True)
    nrem = np.asarray(nrem, float)

    margin = half_sec + bin_sec
    anchors = events[:, 1]
    containing = np.array([np.flatnonzero((nrem[:, 0] + margin <= t) & (t <= nrem[:, 1] - margin))[0]
                           if np.any((nrem[:, 0] + margin <= t) & (t <= nrem[:, 1] - margin)) else -1
                           for t in anchors])
    keep = containing >= 0
    events, n_shanks, anchors, containing = events[keep], n_shanks[keep], anchors[keep], containing[keep]
    if not len(events):
        raise ValueError(f'{block}: no spindle peak has a full peri-event window inside NREM')

    counts, _, _ = sliding_counts(spikes, model['unit_ids'], nrem, bin_sec, step_sec)
    mu, sd = counts.mean(axis=0), counts.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    del counts
    observed = decode_events(model, spikes, anchors, mu, sd, half_sec, bin_sec, step_sec)
    lag = (np.arange(observed['bias'].shape[1]) - (observed['bias'].shape[1] - 1) / 2) * step_sec
    zero = int(np.argmin(np.abs(lag)))
    near = np.abs(lag) <= near_sec + 1e-12
    awake_log_d = np.logaddexp(*awake_loo_logp(model))
    support_cut = float(np.percentile(awake_log_d, SUPPORT_PCT))
    observed_stats = shuffle_statistics(observed, near, support_cut)
    order = np.argsort(-observed['bias'][:, zero])

    rng = np.random.default_rng(seed)
    null_strength, null_log_best, null_support = [], [], []
    null_trace, null_rank = [], []
    representative = None
    for index in range(n_shuffle):
        shifted = shift_units_within_nrem(spikes, model['unit_ids'], nrem, rng, min_shift_sec)
        full_trace = index < n_trace_shuffle
        decoded = decode_events(model, shifted, anchors, mu, sd,
                                half_sec if full_trace else near_sec, bin_sec, step_sec)
        if full_trace:
            if representative is None:
                representative = decoded
            stats = shuffle_statistics(decoded, near, support_cut)
            null_trace.append(np.median(decoded['bias'], axis=0))
            zero_shuffle = zero
        else:
            local_lag = ((np.arange(decoded['bias'].shape[1]) - (decoded['bias'].shape[1] - 1) / 2)
                         * step_sec)
            local_near = np.abs(local_lag) <= near_sec + 1e-12
            stats = shuffle_statistics(decoded, local_near, support_cut)
            zero_shuffle = int(np.argmin(np.abs(local_lag)))
        null_strength.append(stats[0])
        null_log_best.append(stats[1])
        null_support.append(stats[2])
        null_rank.append(-np.sort(-decoded['bias'][:, zero_shuffle]))
        if (index + 1) % max(1, n_shuffle // 5) == 0:
            log(f'    [{block}] shuffle {index + 1}/{n_shuffle}')

    null_strength = np.asarray(null_strength)
    null_log_best = np.asarray(null_log_best)
    null_support = np.asarray(null_support)
    null_trace = np.asarray(null_trace)
    null_rank = np.asarray(null_rank)
    tag = (f'pls_spindle_timing_shuffle_{sortout.name}_{block}_allshanks_'
           f'bin{bin_sec * 1000:g}ms_step{step_sec * 1000:g}ms_peak_n{n_shuffle}'
           + ('' if spindle_source == 'legacy' else '_' + spindle_source))
    base = out_dir / tag
    block_title = (f'{title}\nsleep {block.upper()}: independent per-neuron circular shifts within NREM; '
                   f'{len(events)} merged spindle peaks')
    plot_pls(model, observed, representative, zero, block_title, stamp, base.with_name(tag + '_pls.png'))
    plot_heatmap(observed, representative, lag, order, null_trace, null_rank,
                 null_strength, null_log_best, null_support, observed_stats,
                 smooth_ms, step_sec, block_title, stamp, base.with_name(tag + '_heatmap.png'))
    shuffle_order = np.argsort(-representative['bias'][:, zero])
    plot_heatmap(observed, representative, lag, order, null_trace, null_rank,
                 null_strength, null_log_best, null_support, observed_stats,
                 smooth_ms, step_sec, block_title, stamp,
                 base.with_name(tag + '_heatmap_shuffled_ranked.png'), shuffle_order=shuffle_order)
    np.savez_compressed(base.with_suffix('.npz'), lag_sec=lag, onset_sec=events[:, 0],
                        peak_sec=events[:, 1], offset_sec=events[:, 2], n_shanks=n_shanks,
                        observed_bias=observed['bias'], observed_scores=observed['scores'],
                        representative_shuffle_bias=representative['bias'],
                        representative_shuffle_scores=representative['scores'], order=order,
                        null_median_trace=null_trace, null_rank_zero=null_rank,
                        observed_strength=observed_stats[0], null_strength=null_strength,
                        observed_log_best=observed_stats[1], null_log_best=null_log_best,
                        observed_support_fraction=observed_stats[2], null_support_fraction=null_support,
                        support_log_d_cut=support_cut, near_sec=near_sec, min_shift_sec=min_shift_sec,
                        n_shuffle=n_shuffle, n_trace_shuffle=n_trace_shuffle, seed=seed,
                        bin_sec=bin_sec, step_sec=step_sec,
                        display_smooth_ms=smooth_ms)
    log(f'  [{block}] {n_raw} shank detections -> {len(events)} merged peaks; '
        f'median |R| near peak {observed_stats[0]:.3g}, shuffle median {np.median(null_strength):.3g}, '
        f'p={p_upper(observed_stats[0], null_strength):.3g}; awake log-density p='
        f'{p_upper(observed_stats[1], null_log_best):.3g}; support-fraction p='
        f'{p_upper(observed_stats[2], null_support):.3g}')
    log(f'    saved {base.with_name(tag + "_pls.png")}\n'
        f'    saved {base.with_name(tag + "_heatmap.png")}\n'
        f'    saved {base.with_name(tag + "_heatmap_shuffled_ranked.png")}\n'
        f'    saved {base.with_suffix(".npz")}')
    return [base.with_name(tag + '_pls.png'), base.with_name(tag + '_heatmap.png'),
            base.with_name(tag + '_heatmap_shuffled_ranked.png'), base.with_suffix('.npz')]


def run_day(animal, date, blocks=('pre', 'post'), n_shuffle=100, n_trace_shuffle=20, half_sec=0.5,
            bin_ms=20.0, step_ms=10.0, near_ms=50.0, min_shift_sec=1.0,
            smooth_ms=100.0, awake_bin_ms=100.0, n_components=3,
            max_sec=DEFAULT_MAX_SEC, seed=0, out_dir=None, spindle_source='legacy', log=print):
    key = f'{animal}_{short_date(date)}'
    config = load_registry().get(key)
    if config is None:
        raise KeyError(f'{key} is not in {REGISTRY_FILE}')
    pkl_files = [t['trial_pkl'].replace('\\', '/') for t in config.get('tasks', []) if t.get('trial_pkl')]
    pkl_files = [p for p in pkl_files if Path(p).is_file()]
    model = awake_model(pkl_files, awake_bin_ms / 1000, max_sec, n_components)
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    out_dir = Path(out_dir) if out_dir else sortout / 'pls_task'
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (f'{key} awake PLS fixed from {len(pkl_files)} run(s), {model["n_trials"]} trials, '
             f'{len(model["unit_ids"])} units, {n_components} LVs')
    stamp = (f'Repro: python {Path(__file__).name} {animal} {short_date(date)} --blocks {",".join(blocks)} '
             f'--n-shuffle {n_shuffle} --n-trace-shuffle {n_trace_shuffle} --half-sec {half_sec:g} '
             f'--bin-ms {bin_ms:g} --step-ms {step_ms:g} '
             f'--near-ms {near_ms:g} --min-shift-sec {min_shift_sec:g} --smooth-ms {smooth_ms:g} '
             f'--seed {seed}' + (f' --spindle-source {spindle_source}' if spindle_source != 'legacy' else '')
             + f' | {datetime.now().isoformat(timespec="seconds")}\n'
             'Null: each awake-model unit independently circular-shifted inside every NREM bout; '
             'awake PLS/KDE, spindle peaks, NREM normalization, and event detection fixed.')
    outs = []
    for block in blocks:
        outs.extend(run_block(model, config, block, n_shuffle, n_trace_shuffle, half_sec, bin_ms / 1000,
                              step_ms / 1000, near_ms / 1000, min_shift_sec, smooth_ms,
                              seed, title, stamp, out_dir, spindle_source, log))
    return outs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('animal')
    p.add_argument('date', help='YYMMDD')
    p.add_argument('--blocks', default='pre,post')
    p.add_argument('--n-shuffle', type=int, default=100)
    p.add_argument('--n-trace-shuffle', type=int, default=20,
                   help='shuffles decoded across the full peri-event window; remaining shuffles use +/-near-ms')
    p.add_argument('--half-sec', type=float, default=0.5)
    p.add_argument('--bin-ms', type=float, default=20.0)
    p.add_argument('--step-ms', type=float, default=10.0)
    p.add_argument('--near-ms', type=float, default=50.0,
                   help='robust event-statistic neighborhood around peak (default: %(default)s)')
    p.add_argument('--min-shift-sec', type=float, default=1.0)
    p.add_argument('--smooth-ms', type=float, default=100.0)
    p.add_argument('--awake-bin-ms', type=float, default=100.0)
    p.add_argument('--n-components', type=int, default=3)
    p.add_argument('--max-sec', type=float, default=DEFAULT_MAX_SEC)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--spindle-source', choices=('legacy', 'bosc'), default='legacy')
    args = p.parse_args(argv)
    if (args.n_shuffle < 1 or not 1 <= args.n_trace_shuffle <= args.n_shuffle
            or args.bin_ms <= 0 or args.step_ms <= 0 or args.near_ms < 0):
        p.error('n-shuffle >= n-trace-shuffle >= 1, bin/step > 0, and near-ms >= 0 are required')
    blocks = tuple(x.strip() for x in args.blocks.split(',') if x.strip())
    run_day(args.animal, args.date, blocks, args.n_shuffle, args.n_trace_shuffle, args.half_sec,
            args.bin_ms, args.step_ms, args.near_ms, args.min_shift_sec,
            args.smooth_ms, args.awake_bin_ms, args.n_components,
            args.max_sec or None, args.seed, args.out_dir, args.spindle_source)
    return 0


if __name__ == '__main__':
    sys.exit(main())
