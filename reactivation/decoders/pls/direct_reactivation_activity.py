r"""Direct spike-pattern and time-resolved awake-template analysis of sleep events."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRATING_DIR = HERE.parents[2] / 'DiscriminationTask' / 'grating'  # task_day_discovery lives here
for _p in (str(HERE), str(GRATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pls_sleep_bias import load_nrem_intervals, load_sleep_spikes, sliding_counts
from pls_spindle_bias import load_spindles
from pls_spindle_timing_shuffle import shift_units_within_nrem
from pls_task_trials import COLORS, DEFAULT_MAX_SEC, stack_runs
from task_day_discovery import REGISTRY_FILE, load_registry, short_date


def p_upper(observed, null):
    null = np.asarray(null)
    return (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)


def row_corr(a, templates):
    """Pearson correlation of every row of a with every template row."""
    a = np.asarray(a, np.float32)
    templates = np.asarray(templates, np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    templates = templates - templates.mean(axis=1, keepdims=True)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
    return a @ templates.T


def make_templates(counts, rows, kept, max_bins):
    fit = rows['fit'] & (rows['bin'] < max_bins)
    mu = counts[fit][:, kept].mean(axis=0)
    sd = counts[fit][:, kept].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    z = (counts[:, kept] - mu) / sd
    # Compare like with like: retain only stimulus-time bins represented for
    # both orientations.  A few sessions end one class one bin earlier.
    common_bins = [b for b in range(max_bins)
                   if np.any(fit & (rows['left_ori'] == 45) & (rows['bin'] == b))
                   and np.any(fit & (rows['left_ori'] == 315) & (rows['bin'] == b))]
    templates, labels, bins = [], [], []
    for ori in (45, 315):
        for b in common_bins:
            m = fit & (rows['left_ori'] == ori) & (rows['bin'] == b)
            templates.append(z[m].mean(axis=0))
            labels.append(ori)
            bins.append(b)
    return np.asarray(templates), np.asarray(labels), np.asarray(bins), mu, sd


def trial_folds(rows, n_folds=5):
    trials = np.unique(rows['trial'][rows['fit']])
    fold = {}
    for ori in (45, 315):
        ids = [t for t in trials if rows['left_ori'][np.flatnonzero(rows['trial'] == t)[0]] == ori]
        for i, trial in enumerate(ids):
            fold[trial] = i % n_folds
    return np.array([fold.get(t, -1) for t in rows['trial']])


def validate_templates(counts, rows, kept, max_bins, n_folds=5):
    folds = trial_folds(rows, n_folds)
    correct, total = 0, 0
    for f in range(n_folds):
        train = rows['fit'] & (folds != f) & (rows['bin'] < max_bins)
        test = rows['fit'] & (folds == f) & (rows['bin'] < max_bins)
        common_bins = [b for b in range(max_bins)
                       if np.any(train & (rows['left_ori'] == 45) & (rows['bin'] == b))
                       and np.any(train & (rows['left_ori'] == 315) & (rows['bin'] == b))]
        test &= np.isin(rows['bin'], common_bins)
        if not np.any(test):
            continue
        mu = counts[train][:, kept].mean(axis=0)
        sd = counts[train][:, kept].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        z = (counts[:, kept] - mu) / sd
        templates, labels = [], []
        for ori in (45, 315):
            for b in common_bins:
                m = train & (rows['left_ori'] == ori) & (rows['bin'] == b)
                templates.append(z[m].mean(axis=0))
                labels.append(ori)
        corr = row_corr(z[test], np.asarray(templates))
        labels = np.asarray(labels)
        s45 = corr[:, labels == 45].max(axis=1)
        s315 = corr[:, labels == 315].max(axis=1)
        pred = np.where(s45 >= s315, 45, 315)
        correct += np.count_nonzero(pred == rows['left_ori'][test])
        total += test.sum()
    return correct / total if total else np.nan


def event_counts(spikes, unit_ids, anchors, half_sec, bin_sec, step_sec):
    windows = np.column_stack([anchors - half_sec - bin_sec / 2,
                               anchors + half_sec + bin_sec / 2])
    counts, _, event = sliding_counts(spikes, unit_ids, windows, bin_sec, step_sec)
    n_lag = counts.shape[0] // len(anchors)
    if counts.shape[0] != len(anchors) * n_lag:
        raise RuntimeError('unequal event window sizes')
    lag = (np.arange(n_lag) - (n_lag - 1) / 2) * step_sec
    return counts.reshape(len(anchors), n_lag, len(unit_ids)), lag


def similarities(z, templates, labels):
    shape = z.shape[:2]
    corr = row_corr(z.reshape(-1, z.shape[-1]), templates)
    s45 = corr[:, labels == 45].max(axis=1).reshape(shape)
    s315 = corr[:, labels == 315].max(axis=1).reshape(shape)
    return s45, s315, s45 - s315, np.maximum(s45, s315)


def event_excluded_nrem_stats(spikes, unit_ids, nrem, anchors, bin_sec, step_sec, exclude_sec):
    counts, time, _ = sliding_counts(spikes, unit_ids, nrem, bin_sec, step_sec)
    background = np.ones(len(time), bool)
    for anchor in anchors:
        lo, hi = np.searchsorted(time, [anchor - exclude_sec, anchor + exclude_sec])
        background[lo:hi] = False
    if background.sum() < 100:
        raise ValueError('too few event-excluded NREM windows for normalization')
    mu = counts[background].mean(axis=0)
    sd = counts[background].std(axis=0)
    return mu, sd, background.sum()


def plot_activity(templates, labels, bins, selectivity, order, z, raw_counts, lag,
                  event_class, bin_sec, title, stamp, out):
    awake_diff = np.stack([
        templates[(labels == 45) & (bins == b)].mean(axis=0)
        - templates[(labels == 315) & (bins == b)].mean(axis=0)
        for b in sorted(np.unique(bins))
    ], axis=1)
    positive = event_class == 45
    negative = event_class == 315
    mean45 = np.mean(z[positive], axis=0).T if positive.any() else np.full((z.shape[2], z.shape[1]), np.nan)
    mean315 = np.mean(z[negative], axis=0).T if negative.any() else np.full((z.shape[2], z.shape[1]), np.nan)
    lim = np.nanpercentile(np.abs(np.c_[awake_diff, mean45, mean315]), 99)
    lim = max(float(lim), 0.5)
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(2.3, 1), hspace=0.32, wspace=0.25,
                          left=0.055, right=0.98, top=0.9, bottom=0.1)
    panels = ((awake_diff[order], 'awake 45 - 315 template', np.arange(awake_diff.shape[1]) * 0.1),
              (mean45[order], f'45-like sleep events (n={positive.sum()})', lag),
              (mean315[order], f'315-like sleep events (n={negative.sum()})', lag))
    for col, (matrix, name, x) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                       interpolation='nearest', extent=(x[0], x[-1] + (x[1] - x[0]), len(order), 0))
        if col:
            ax.axvline(0, color='k', lw=0.7)
        ax.set(xlabel='awake time from stimulus onset (s)' if col == 0 else 'time from spindle peak (s)',
               ylabel='neurons sorted 45-preferring to 315-preferring' if col == 0 else '', title=name)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.11,
                     label='standardized activity')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(selectivity[order], np.arange(len(order)), color='k', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.6)
    ax.set(ylim=(len(order), 0), xlabel='awake selectivity (45 - 315)', ylabel='sorted neuron',
           title='awake selectivity used for row order')
    q = max(1, len(order) // 4)
    pref45, pref315 = order[:q], order[-q:]
    for col, (mask, name) in enumerate(((positive, '45-like events'), (negative, '315-like events')), start=1):
        ax = fig.add_subplot(gs[1, col])
        if mask.any():
            hz45 = raw_counts[mask][:, :, pref45].mean(axis=(0, 2)) / bin_sec
            hz315 = raw_counts[mask][:, :, pref315].mean(axis=(0, 2)) / bin_sec
            ax.plot(lag, hz45, color=COLORS[45], label='awake 45-preferring quartile')
            ax.plot(lag, hz315, color=COLORS[315], label='awake 315-preferring quartile')
            ax.plot(lag, hz45 - hz315, color='k', lw=0.9, label='difference')
        ax.axvline(0, color='0.4', lw=0.7)
        ax.set(xlabel='time from spindle peak (s)', ylabel='raw firing rate (Hz)', title=name)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, x=0.01, ha='left', y=0.985, fontsize=11)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_similarity(s45, s315, diff, best, rep_s45, rep_s315, rep_diff,
                    lag, order, null_best, null_absdiff, null_trace, title, stamp, out):
    zero = int(np.argmin(np.abs(lag)))
    rep_ordered = rep_diff[order]
    vmax = max(0.2, np.percentile(np.abs(np.r_[diff.ravel(), rep_diff.ravel()]), 99))
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(2.1, 1), hspace=0.3, wspace=0.26,
                          left=0.055, right=0.98, top=0.9, bottom=0.1)
    for col, (matrix, name) in enumerate(((diff[order], 'observed template preference'),
                                          (rep_ordered, 'representative timing shuffle, observed row order'))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', extent=(lag[0], lag[-1], len(order), 0))
        ax.axvline(0, color='k', lw=0.7)
        ax.set(xlabel='time from spindle peak (s)', ylabel='observed zero-lag template rank' if col == 0 else '',
               title=name)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.11,
                     label='S45 - S315')
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(s315[:, zero], s45[:, zero], s=12, color='tab:purple', alpha=0.65, label='observed')
    ax.scatter(rep_s315[:, zero], rep_s45[:, zero], s=10, color='0.55', alpha=0.45, label='shuffle')
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7)
    ax.set(xlabel='max similarity to 315 templates', ylabel='max similarity to 45 templates',
           title='100 ms population vectors at spindle peak')
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    null_trace = np.asarray(null_trace)
    ax.fill_between(lag, *np.percentile(null_trace, [2.5, 97.5], axis=0), color='0.8',
                    label='timing shuffle 95% envelope')
    ax.plot(lag, np.median(null_trace, axis=0), color='0.45', lw=1, label='shuffle median')
    ax.plot(lag, np.median(best, axis=0), color='tab:purple', lw=1.4, label='observed')
    ax.axvline(0, color='k', lw=0.7)
    ax.set(xlabel='time from spindle peak (s)', ylabel='median best awake-template similarity',
           title='awake-pattern similarity')
    ax.legend(fontsize=7, frameon=False)
    near = np.abs(lag) <= 0.05 + 1e-12
    observed_best = float(np.median(best[:, near]))
    observed_absdiff = float(np.median(np.abs(diff[:, near])))
    for col, values, observed, label in ((1, null_best, observed_best, 'median best-template similarity'),
                                          (2, null_absdiff, observed_absdiff, 'median |S45 - S315|')):
        ax = fig.add_subplot(gs[1, col])
        ax.hist(values, bins=20, color='0.75')
        ax.axvline(observed, color='tab:purple', lw=2, label='observed')
        ax.set(xlabel=label, ylabel='shuffle count', title=f'one-sided p={p_upper(observed, values):.3g}')
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha='left', y=0.985, fontsize=11)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def run_block(awake_counts, awake_rows, awake_unit_ids, awake_kept, templates_full, labels,
              bins, selectivity_full, cv_accuracy, config, block, n_shuffle, n_trace_shuffle,
              half_sec, bin_sec, step_sec, min_shift_sec, seed, title, stamp, out_dir, log=print):
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    spindle_dir = Path(config['ephys_folder'].replace('\\', '/')) / 'low_freq' / 'v1_spindles'
    events, n_shanks, _, n_raw = load_spindles(spindle_dir, block, None)
    nrem_path = sorted(spindle_dir.glob(f'*_{block}_sleep_periods.pkl'))[0]
    sleep_path = sorted(sortout.glob(f'sleep_spikes_*sleep_{block}.pkl'))[0]
    spikes, _ = load_sleep_spikes(sleep_path, verbose=False)
    nrem = load_nrem_intervals(nrem_path, require_bouts=True)
    unit_ids = [u for u, k in zip(awake_unit_ids, awake_kept) if k and u in spikes]
    awake_positions = {u: i for i, u in enumerate([u for u, k in zip(awake_unit_ids, awake_kept) if k])}
    cols = np.array([awake_positions[u] for u in unit_ids])
    templates = templates_full[:, cols]
    selectivity = selectivity_full[cols]

    margin = half_sec + bin_sec
    anchors = events[:, 1]
    inside = np.array([np.any((nrem[:, 0] + margin <= t) & (t <= nrem[:, 1] - margin)) for t in anchors])
    events, n_shanks, anchors = events[inside], n_shanks[inside], anchors[inside]
    mu, sd, n_background = event_excluded_nrem_stats(
        spikes, unit_ids, nrem, anchors, bin_sec, step_sec, exclude_sec=1.0)
    sleep_kept = sd > 0
    unit_ids = [u for u, k in zip(unit_ids, sleep_kept) if k]
    mu, sd = mu[sleep_kept], sd[sleep_kept]
    templates, selectivity = templates[:, sleep_kept], selectivity[sleep_kept]
    raw, lag = event_counts(spikes, unit_ids, anchors, half_sec, bin_sec, step_sec)
    z = (raw - mu) / sd
    s45, s315, diff, best = similarities(z, templates, labels)
    zero = int(np.argmin(np.abs(lag)))
    event_class = np.where(diff[:, zero] >= 0, 45, 315)
    order = np.argsort(-diff[:, zero])
    neuron_order = np.argsort(-selectivity)
    near = np.abs(lag) <= 0.05 + 1e-12
    observed_best = float(np.median(best[:, near]))
    observed_absdiff = float(np.median(np.abs(diff[:, near])))

    rng = np.random.default_rng(seed)
    null_best, null_absdiff, null_trace = [], [], []
    representative = None
    for index in range(n_shuffle):
        shifted = shift_units_within_nrem(spikes, unit_ids, nrem, rng, min_shift_sec)
        full = index < n_trace_shuffle
        shuffled_raw, shuffled_lag = event_counts(
            shifted, unit_ids, anchors, half_sec if full else 0.05, bin_sec, step_sec)
        shuffled_z = (shuffled_raw - mu) / sd
        a, b, d, best_shuffle = similarities(shuffled_z, templates, labels)
        local_near = np.abs(shuffled_lag) <= 0.05 + 1e-12
        null_best.append(float(np.median(best_shuffle[:, local_near])))
        null_absdiff.append(float(np.median(np.abs(d[:, local_near]))))
        if full:
            null_trace.append(np.median(best_shuffle, axis=0))
            if representative is None:
                representative = (a, b, d)
        if (index + 1) % max(1, n_shuffle // 5) == 0:
            log(f'    [{block}] direct-template shuffle {index + 1}/{n_shuffle}')
    null_best, null_absdiff = np.asarray(null_best), np.asarray(null_absdiff)
    rep_s45, rep_s315, rep_diff = representative
    tag = (f'direct_reactivation_{sortout.name}_{block}_allshanks_'
           f'bin{bin_sec * 1000:g}ms_step{step_sec * 1000:g}ms_peak_n{n_shuffle}')
    block_title = (f'{title}\nsleep {block.upper()}: {len(events)} merged spindle peaks, '
                   f'{len(unit_ids)} nonsilent units; awake template CV accuracy {cv_accuracy:.1%}')
    plot_activity(templates, labels, bins, selectivity, neuron_order, z, raw, lag,
                  event_class, bin_sec, block_title, stamp, out_dir / f'{tag}_activity.png')
    plot_similarity(s45, s315, diff, best, rep_s45, rep_s315, rep_diff,
                    lag, order, null_best, null_absdiff, null_trace,
                    block_title, stamp, out_dir / f'{tag}_template.png')
    np.savez_compressed(out_dir / f'{tag}.npz', unit_ids=np.asarray(unit_ids), lag_sec=lag,
                        onset_sec=events[:, 0], peak_sec=events[:, 1], offset_sec=events[:, 2],
                        event_class=event_class, neuron_order=neuron_order, selectivity=selectivity,
                        observed_s45=s45, observed_s315=s315, observed_diff=diff, observed_best=best,
                        representative_s45=rep_s45, representative_s315=rep_s315,
                        representative_diff=rep_diff, null_best=null_best, null_absdiff=null_absdiff,
                        null_best_trace=np.asarray(null_trace), observed_best_stat=observed_best,
                        observed_absdiff_stat=observed_absdiff, awake_template_cv_accuracy=cv_accuracy,
                        n_background_windows=n_background, bin_sec=bin_sec, step_sec=step_sec,
                        n_shuffle=n_shuffle, seed=seed)
    log(f'  [{block}] {n_raw} shank detections -> {len(events)} events; best-template similarity '
        f'{observed_best:.3f} vs shuffle {np.median(null_best):.3f}, p={p_upper(observed_best, null_best):.3g}; '
        f'|template difference| {observed_absdiff:.3f} vs {np.median(null_absdiff):.3f}, '
        f'p={p_upper(observed_absdiff, null_absdiff):.3g}')
    log(f'    saved {out_dir / (tag + "_activity.png")}\n'
        f'    saved {out_dir / (tag + "_template.png")}\n    saved {out_dir / (tag + ".npz")}')


def run_day(animal, date, blocks=('pre', 'post'), n_shuffle=100, n_trace_shuffle=20,
            half_sec=0.5, bin_ms=100.0, step_ms=10.0, min_shift_sec=1.0,
            max_sec=DEFAULT_MAX_SEC, seed=0, out_dir=None, log=print):
    key = f'{animal}_{short_date(date)}'
    config = load_registry().get(key)
    if config is None:
        raise KeyError(f'{key} is not in {REGISTRY_FILE}')
    pkl_files = [t['trial_pkl'].replace('\\', '/') for t in config.get('tasks', []) if t.get('trial_pkl')]
    pkl_files = [p for p in pkl_files if Path(p).is_file()]
    awake_counts, awake_unit_ids, rows = stack_runs(pkl_files, bin_ms / 1000, max_sec)
    fit = rows['fit']
    awake_kept = awake_counts[fit].std(axis=0) > 0
    max_bins = int(round(max_sec / (bin_ms / 1000)))
    templates, labels, bins, _, _ = make_templates(awake_counts, rows, awake_kept, max_bins)
    cv_accuracy = validate_templates(awake_counts, rows, awake_kept, max_bins)
    selectivity = (templates[labels == 45].mean(axis=0) - templates[labels == 315].mean(axis=0))
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    out_dir = Path(out_dir) if out_dir else sortout / 'pls_task'
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (f'{key}: direct 100 ms population activity; {len(pkl_files)} awake run(s), '
             f'{np.unique(rows["trial"][fit]).size} trials, {awake_kept.sum()} awake-variable units')
    stamp = (f'Repro: python {Path(__file__).name} {animal} {short_date(date)} --blocks {",".join(blocks)} '
             f'--n-shuffle {n_shuffle} --n-trace-shuffle {n_trace_shuffle} --half-sec {half_sec:g} '
             f'--bin-ms {bin_ms:g} --step-ms {step_ms:g} --min-shift-sec {min_shift_sec:g} --seed {seed} | '
             f'{datetime.now().isoformat(timespec="seconds")}\n'
             'Awake templates: every 100 ms position in first 3 s, unit z-score from awake stimulus bins; '
             'validation split by trial. Sleep: unit z-score from event-excluded NREM 100 ms windows; '
             'normalization fixed for independent-unit circular timing shuffles.')
    for block in blocks:
        run_block(awake_counts, rows, awake_unit_ids, awake_kept, templates, labels, bins,
                  selectivity, cv_accuracy, config, block, n_shuffle, n_trace_shuffle,
                  half_sec, bin_ms / 1000, step_ms / 1000, min_shift_sec,
                  seed, title, stamp, out_dir, log)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('animal'); p.add_argument('date')
    p.add_argument('--blocks', default='pre,post')
    p.add_argument('--n-shuffle', type=int, default=100)
    p.add_argument('--n-trace-shuffle', type=int, default=20)
    p.add_argument('--half-sec', type=float, default=0.5)
    p.add_argument('--bin-ms', type=float, default=100.0)
    p.add_argument('--step-ms', type=float, default=10.0)
    p.add_argument('--min-shift-sec', type=float, default=1.0)
    p.add_argument('--max-sec', type=float, default=DEFAULT_MAX_SEC)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args(argv)
    if not 1 <= args.n_trace_shuffle <= args.n_shuffle:
        p.error('require n-shuffle >= n-trace-shuffle >= 1')
    run_day(args.animal, args.date, tuple(x.strip() for x in args.blocks.split(',') if x.strip()),
            args.n_shuffle, args.n_trace_shuffle, args.half_sec, args.bin_ms,
            args.step_ms, args.min_shift_sec, args.max_sec or None,
            args.seed, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
