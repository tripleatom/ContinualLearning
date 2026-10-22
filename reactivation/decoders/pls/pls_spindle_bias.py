r"""Sleep reactivation bias R(t) around V1 spindle onset or peak, one row per spindle, ranked.

    python pls_spindle_bias.py CnL46 260729                  # pre and post, spindles merged across shanks
    python pls_spindle_bias.py CnL46 260729 --shank 0        # only shank 0's event list
    python pls_spindle_bias.py CnL46 260729 --rank-window -0.5 0.5
    python pls_spindle_bias.py CnL46 260729 --smooth-ms 50   # display-only temporal smoothing
    python pls_spindle_bias.py CnL46 260729 --align peak --smooth-ms 100

Bias: exactly pls_sleep_bias.py - the awake PLS + one 3-D KDE per left grating,
sleep spikes in --bin-ms windows every --step-ms, each unit z-scored with its
mean / sd over the block's NREM windows, R(t) = ln p(z | 45 left) / p(z | 315
left). The NREM bouts are the ones the spindles were detected in
(<ephys>/low_freq/v1_spindles/*_<block>_sleep_periods.pkl), not the older
sleep_segmentation scoring.

Events: <ephys>/low_freq/v1_spindles/*_<block>_sh<N>_v1_spindles.csv
(sleep/spindles/detect_v1_spindles.py), onset_s = seconds from the sleep block's start,
the clock of the sleep spike pkl. The detector runs per shank and the same
spindle shows up on several shanks, so by default events whose [onset, offset]
overlap across shanks are merged into one (onset = the earliest, offset = the
latest, peak = the peak_s of the constituent with the largest peak_z, n_shanks
kept). --shank N uses one shank's list as it is.

Every event gives R on windows centred from -half to +half (--half-sec, 0.5 s)
around its --align anchor (onset by default, or peak). Its value is the median
of R over --rank-window (default anchor -> +half): the median, because single
windows reach |R| of 30 and would own a mean. Events are ranked by that value.

Reference: --n-random onsets drawn uniformly inside the NREM bouts and treated
the same way. The bands in the figure are the 2.5-97.5 percentiles over random
sets of as many onsets as there are spindles.

Outputs, in <sortout>/pls_task/:
  pls_spindle_<sortout>_<block>_<shanks>_bin<ms>ms_step<ms>ms.png
  pls_spindle_<sortout>_<block>_<shanks>_bin<ms>ms_step<ms>ms.npz   lag_sec, bias (events x lags, ranked),
        value, value_pre, onset_sec, offset_sec, n_shanks, log_d, random_bias, random_value
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

GRATING_DIR = Path(__file__).resolve().parents[3] / 'DiscriminationTask' / 'grating'  # task_day_discovery lives here
if str(GRATING_DIR) not in sys.path:
    sys.path.insert(0, str(GRATING_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.ndimage import gaussian_filter1d

from pls_sleep_bias import awake_model, load_nrem_intervals, load_sleep_spikes, score_points, sliding_counts
from pls_task_trials import BIAS_CLIP, COLORS, DEFAULT_MAX_SEC, KDE_DIMS
from task_day_discovery import REGISTRY_FILE, load_registry, short_date

N_BOOT = 2000              # random sets of n_events onsets behind the reference bands
BAND_SETS = 500            # of those, the sets used for the band of the triggered trace


# ============================================================ events

SPINDLE_SOURCES = {'legacy': ('v1_spindles', '_v1_spindles.csv', 'peak_z', 'detect_v1_spindles.py'),
                   'bosc': ('spindle_bosc', '_spindle_bosc.csv', 'peak_power_ratio', 'spindle_bosc.py')}


def load_spindles(spindle_dir, block, shank=None, source='legacy'):
    """(events (n x 3: onset, peak, offset, s from block start), n_shanks, csv files).
    shank None = every shank's list, overlapping events merged. spindle_dir is always the
    v1_spindles folder (it holds the NREM scoring); source='bosc' reads the sibling
    spindle_bosc folder, and its peak_power_ratio plays the role of peak_z when merging."""
    folder, suffix, strength, script = SPINDLE_SOURCES[source]
    spindle_dir = Path(spindle_dir).parent / folder
    pattern = f"*_{block}_sh{'*' if shank is None else shank}{suffix}"
    files = sorted(p for p in spindle_dir.glob(pattern) if not p.name.startswith('.'))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {spindle_dir} - run sleep/spindles/{script}")
    rows = []
    for f in files:
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                rows.append((float(r['onset_s']), float(r['peak_s']), float(r['offset_s']),
                             float(r.get(strength, 'nan')), int(r['shank'])))
    if not rows:
        raise ValueError(f"{pattern}: no event in {len(files)} file(s)")
    rows.sort()
    merged = []                       # [onset, peak, offset, peak_z, {shanks}]
    for on, peak, off, peak_z, sh in rows:
        if shank is None and merged and on <= merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], off)
            if not np.isfinite(merged[-1][3]) or (np.isfinite(peak_z) and peak_z > merged[-1][3]):
                merged[-1][1], merged[-1][3] = peak, peak_z
            merged[-1][4].add(sh)
        else:
            merged.append([on, peak, off, peak_z, {sh}])
    return (np.array([[m[0], m[1], m[2]] for m in merged]), np.array([len(m[4]) for m in merged]),
            files, len(rows))


def peri_event(model, spikes, onsets, mu, sd, half_sec, bin_sec, step_sec):
    """(R, ln D), each events x lags, on windows centred -half .. +half around every onset."""
    windows = np.column_stack([onsets - half_sec - bin_sec / 2, onsets + half_sec + bin_sec / 2])
    counts, _, event = sliding_counts(spikes, model['unit_ids'], windows, bin_sec, step_sec)
    n_lag = counts.shape[0] // len(onsets)
    assert counts.shape[0] == n_lag * len(onsets) and np.all(event == np.repeat(np.arange(len(onsets)), n_lag))
    _, lp45, lp315 = score_points(model, (counts - mu) / sd)
    return (lp45 - lp315).reshape(len(onsets), n_lag), np.logaddexp(lp45, lp315).reshape(len(onsets), n_lag)


# ============================================================ figure

def plot_spindle_bias(res, half_sec, rank_window, title, stamp, out, smooth_ms=0.0, align='onset'):
    lag, bias, value = res['lag_sec'], res['bias'], res['value']          # rows already ranked, most 45-like first
    n = len(value)
    step = float(np.median(np.diff(lag)))
    if smooth_ms > 0:
        # Interpret the requested width as FWHM, then convert to Gaussian sigma.
        # This is display-only: ranking, values, tests, and saved bias arrays
        # remain based on the original decoding windows.
        sigma_samples = (smooth_ms / 1000.0) / (2 * np.sqrt(2 * np.log(2))) / step
        shown_bias = gaussian_filter1d(bias, sigma_samples, axis=1, mode='nearest')
        shown_random_bias = gaussian_filter1d(res['random_bias'], sigma_samples, axis=1, mode='nearest')
        smooth_label = f'; Gaussian smoothing {smooth_ms:g} ms FWHM (display only)'
    else:
        shown_bias = bias
        shown_random_bias = res['random_bias']
        smooth_label = '; unsmoothed'
    rng = np.random.default_rng(0)
    sets = rng.integers(0, len(res['random_value']), size=(N_BOOT, n))
    fig = plt.figure(figsize=(18, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 2.6], width_ratios=[3, 1, 1.9], hspace=0.22, wspace=0.2,
                            left=0.05, right=0.98, top=0.9, bottom=0.11)

    ax = fig.add_subplot(grid[0, 0])
    # median over events, not the mean: one window at R = -300 would own the mean of its lag
    null_median = np.stack([np.median(shown_random_bias[sets[i]], axis=0) for i in range(BAND_SETS)])
    ax.fill_between(lag, *np.percentile(null_median, [2.5, 97.5], axis=0), color='0.8', lw=0,
                    label=f'random NREM onsets: 95% of the medians of {n}')
    q25, q50, q75 = np.percentile(shown_bias, [25, 50, 75], axis=0)
    ax.fill_between(lag, q25, q75, color='tab:purple', alpha=0.2, lw=0, label='spindles: quartiles over events')
    ax.plot(lag, q50, color='tab:purple', lw=1.3, label='spindles: median over events')
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='tab:green', lw=0.8)
    ax.axvspan(*rank_window, color='tab:green', alpha=0.07, lw=0)
    ax.set_xlim(-half_sec, half_sec)
    ax.set_ylabel('R')
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper left')
    ax.set_title('spindle-triggered R(t)  (green band = the window each event is valued on)' + smooth_label,
                 fontsize=9)

    ax = fig.add_subplot(grid[1, 0])
    im = ax.imshow(shown_bias, aspect='auto', cmap='RdBu_r', vmin=-BIAS_CLIP, vmax=BIAS_CLIP,
                   interpolation='nearest',
                   extent=(lag[0] - step / 2, lag[-1] + step / 2, n - 0.5, -0.5))
    onset_rel = res['onset_sec'] - res['align_sec']
    offset_rel = res['offset_sec'] - res['align_sec']
    shown = (offset_rel >= -half_sec) & (offset_rel <= half_sec)
    ax.plot(offset_rel[shown], np.arange(n)[shown], 'k|', ms=3, mew=0.8,
            label='spindle offset (if within the view)')
    if align == 'peak':
        shown = (onset_rel >= -half_sec) & (onset_rel <= half_sec)
        ax.plot(onset_rel[shown], np.arange(n)[shown], '|', color='0.45', ms=3, mew=0.8,
                label='spindle onset (if within the view)')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlim(-half_sec, half_sec)
    ax.set_xlabel(f'time from spindle {align} (s)')
    ax.set_ylabel(f'spindle, ranked by its value (rank 1 = most 45\N{DEGREE SIGN}-left-like)')
    ax.legend(fontsize=7, loc='lower left')
    fig.colorbar(im, ax=ax, extend='both', orientation='horizontal', fraction=0.035, pad=0.07,
                 label='R = ln p(z | 45\N{DEGREE SIGN} left) / p(z | 315\N{DEGREE SIGN} left)')
    ax.set_title(f'every spindle, one row each ({n} events)', fontsize=9)

    ax = fig.add_subplot(grid[1, 1])
    rows = np.arange(n)
    ax.barh(rows, value, height=0.9, color=np.where(value > 0, COLORS[45], COLORS[315]))
    ax.plot(res['value_pre'], rows, '.', color='0.35', ms=3, label=f'same event, {-half_sec:g} -> 0 s')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel(f'event value = median R, {rank_window[0]:g} -> {rank_window[1]:g} s')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title(f'{np.mean(value > 0):.0%} of events > 0', fontsize=9)

    ax = fig.add_subplot(grid[0, 2])
    lim = max(np.abs(value).max(), np.percentile(np.abs(res['random_value']), 99.5)) * 1.05
    edges = np.linspace(-lim, lim, 41)
    ax.hist(res['random_value'], bins=edges, density=True, color='0.8',
            label=f"random NREM onsets ({len(res['random_value'])})")
    ax.hist(value, bins=edges, density=True, histtype='step', color='k', lw=1.3, label=f'spindles ({n})')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('event value')
    ax.set_ylabel('density')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"median: spindles {np.median(value):+.2f}, random {np.median(res['random_value']):+.2f}\n"
                 f"(Mann-Whitney p = {res['p_shift']:.3g};  on |value| p = {res['p_abs']:.3g})", fontsize=9)

    ax = fig.add_subplot(grid[1, 2])
    null_sorted = -np.sort(-res['random_value'][sets], axis=1)             # each random set ranked like the events
    rank = np.arange(1, n + 1)
    ax.fill_between(rank, *np.percentile(null_sorted, [2.5, 97.5], axis=0), color='0.8', lw=0,
                    label=f'random NREM onsets: 95% of ranked sets of {n}')
    ax.plot(rank, np.median(null_sorted, axis=0), color='0.5', lw=0.8)
    ax.scatter(rank, value, s=10, c=np.where(value > 0, COLORS[45], COLORS[315]), zorder=3, label='spindles')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('rank')
    ax.set_ylabel('event value')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('ranked event values against ranked random onsets', fontsize=9)

    fig.suptitle(title, fontsize=11, x=0.01, ha='left', y=0.985)
    fig.text(0.005, 0.004, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ============================================================ driver

def run_block(model, config, block, shank, half_sec, rank_window, bin_sec, step_sec, smooth_ms, n_random, seed,
              title, stamp, out_dir, align='onset', spindle_source='legacy', log=print):
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    spindle_dir = Path(config['ephys_folder'].replace('\\', '/')) / 'low_freq' / 'v1_spindles'
    events, n_shanks, csv_files, n_raw = load_spindles(spindle_dir, block, shank, spindle_source)
    nrem_pkl = sorted(p for p in spindle_dir.glob(f'*_{block}_sleep_periods.pkl') if not p.name.startswith('.'))
    sleep_pkl = sorted(p for p in sortout.glob(f'sleep_spikes_*sleep_{block}.pkl') if not p.name.startswith('.'))
    if not nrem_pkl or not sleep_pkl:
        raise FileNotFoundError(f"missing {'the NREM scoring in ' + str(spindle_dir) if not nrem_pkl else ''} "
                                f"{'sleep_spikes_*sleep_' + block + '.pkl in ' + str(sortout) if not sleep_pkl else ''}")
    spikes, meta = load_sleep_spikes(sleep_pkl[0], verbose=False)
    nrem = load_nrem_intervals(nrem_pkl[0], require_bouts=True)

    counts, _, _ = sliding_counts(spikes, model['unit_ids'], nrem, bin_sec, step_sec)
    mu, sd = counts.mean(axis=0), counts.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)        # silent in NREM: z = 0 there, raw count around the events
    del counts

    margin = half_sec + bin_sec
    align_col = 0 if align == 'onset' else 1
    anchors = events[:, align_col]
    inside = (anchors >= margin) & (anchors <= meta['duration_sec'] - margin)
    events, n_shanks = events[inside], n_shanks[inside]
    anchors = events[:, align_col]
    bias, log_d = peri_event(model, spikes, anchors, mu, sd, half_sec, bin_sec, step_sec)
    lag = (np.arange(bias.shape[1]) - (bias.shape[1] - 1) / 2) * step_sec
    in_rank = (lag >= rank_window[0] - 1e-9) & (lag <= rank_window[1] + 1e-9)
    value = np.median(bias[:, in_rank], axis=1)
    value_pre = np.median(bias[:, lag < -1e-9], axis=1)

    rng = np.random.default_rng(seed)
    room = np.clip(nrem[:, 1] - nrem[:, 0] - 2 * margin, 0, None)
    which = rng.choice(len(nrem), size=n_random, p=room / room.sum())
    random_onsets = np.sort(nrem[which, 0] + margin + rng.random(n_random) * room[which])
    random_bias, _ = peri_event(model, spikes, random_onsets, mu, sd, half_sec, bin_sec, step_sec)
    random_value = np.median(random_bias[:, in_rank], axis=1)

    order = np.argsort(-value)
    res = {'lag_sec': lag, 'bias': bias[order], 'log_d': log_d[order], 'value': value[order],
           'value_pre': value_pre[order], 'onset_sec': events[order, 0], 'peak_sec': events[order, 1],
           'offset_sec': events[order, 2], 'align_sec': anchors[order],
           'n_shanks': n_shanks[order], 'random_bias': random_bias, 'random_value': random_value,
           'random_onset_sec': random_onsets,
           'p_shift': float(mannwhitneyu(value, random_value).pvalue),
           'p_abs': float(mannwhitneyu(np.abs(value), np.abs(random_value)).pvalue)}
    p_prepost = float(wilcoxon(value, value_pre).pvalue)

    which_shanks = 'allshanks' if shank is None else f'sh{shank}'
    in_nrem = np.mean([np.any((nrem[:, 0] <= t) & (t < nrem[:, 1])) for t in events[:, 0]])
    log(f"  [{block}] {n_raw} event(s) in {len(csv_files)} csv -> {len(events)} spindles"
        + (f" after merging across shanks (median {np.median(n_shanks):.0f} shanks per spindle)" if shank is None
           else f" on shank {shank}")
        + f"; {in_nrem:.0%} of onsets inside the NREM bouts; median duration "
          f"{np.median(events[:, 2] - events[:, 0]):.2f} s; aligned to {align}")
    log(f"    event value (median R, {rank_window[0]:g} -> {rank_window[1]:g} s): median {np.median(value):+.2f}, "
        f"range {value.min():+.2f} .. {value.max():+.2f}, {np.mean(value > 0):.0%} > 0;  random NREM onsets: "
        f"median {np.median(random_value):+.2f}, {np.mean(random_value > 0):.0%} > 0")
    log(f"    spindles vs random: Mann-Whitney p = {res['p_shift']:.3g} on the value, {res['p_abs']:.3g} on |value| "
        f"(median |value| {np.median(np.abs(value)):.2f} vs {np.median(np.abs(random_value)):.2f});  after vs before "
        f"{align} within event: median {np.median(value):+.2f} vs {np.median(value_pre):+.2f}, Wilcoxon p = "
        f"{p_prepost:.3g}")

    smooth_values = np.atleast_1d(smooth_ms).astype(float)
    source = (f"{len(events)} spindles, merged across {len(csv_files)} shanks" if shank is None
              else f"{len(events)} spindles of shank {shank}")
    outs = []
    for smooth in smooth_values:
        src_suffix = '' if spindle_source == 'legacy' else '_' + spindle_source
        smooth_suffix = f"_smooth{smooth:g}ms" if smooth > 0 else ''
        align_suffix = '_alignpeak' if align == 'peak' else ''
        rank_suffix = '_rank0ms' if rank_window == (0.0, 0.0) else ''
        out = out_dir / (f"pls_spindle_{sortout.name}_{block}_{which_shanks}_bin{bin_sec * 1000:g}ms_"
                         f"step{step_sec * 1000:g}ms{align_suffix}{rank_suffix}{smooth_suffix}{src_suffix}.png")
        plot_spindle_bias(res, half_sec, rank_window,
                          f"{title}\nsleep {block.upper()}: R(t) around V1 spindle {align},  {source},  "
                          f"{bin_sec * 1000:g} ms windows every {step_sec * 1000:g} ms",
                          stamp + f"\nsleep inputs: {sleep_pkl[0]} ; {nrem_pkl[0]} (bout_intervals_s) ; "
                                  f"{csv_files[0].parent}/{csv_files[0].name.replace('sh0', 'sh*') if shank is None else csv_files[0].name} ({spindle_source})\n"
                                  f"sleep: counts z-scored per unit with mean / sd over all NREM windows of the block; "
                                  f"event value = median R over {rank_window[0]:g} -> {rank_window[1]:g} s; overlapping "
                                  f"events of different shanks merged (peak from largest peak_z); reference = {n_random} uniform "
                                  f"random onsets inside the NREM bouts, bands = 2.5-97.5 percentiles over {N_BOOT} sets "
                                  f"of {len(events)}; colour clipped at +-{BIAS_CLIP:g}", out, smooth, align)
        np.savez_compressed(out.with_suffix('.npz'), unit_ids=np.array(model['unit_ids']), nrem_windows_sec=nrem,
                            bin_sec=bin_sec, step_sec=step_sec, display_smooth_ms=smooth,
                            rank_window=np.array(rank_window), align_to=np.array(align), **res)
        log(f"    saved {out}\n    saved {out.with_suffix('.npz')}")
        outs.append(out)
    return outs


def run_day(animal, date, blocks=('pre', 'post'), shank=None, half_sec=0.5, rank_window=None, bin_ms=20.0,
            step_ms=10.0, smooth_ms=0.0, awake_bin_ms=100.0, n_components=3, max_sec=DEFAULT_MAX_SEC,
            n_random=2000, seed=0, out_dir=None, align='onset', spindle_source='legacy', log=print):
    key = f"{animal}_{short_date(date)}"
    config = load_registry().get(key)
    if config is None:
        raise KeyError(f"{key} is not in {REGISTRY_FILE} - register the day with task_day_gui.py")
    if n_components < KDE_DIMS:
        raise ValueError(f"the KDE needs --n-components >= {KDE_DIMS}")
    rank_window = (0.0, half_sec) if rank_window is None else tuple(rank_window)
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
    smooth_values = np.atleast_1d(smooth_ms).astype(float)
    smooth_arg = ' '.join(f'{x:g}' for x in smooth_values)
    stamp = (f"Repro: python {Path(__file__).name} {animal} {short_date(date)} --blocks {','.join(blocks)}"
             + (f" --shank {shank}" if shank is not None else '')
             + f" --align {align} --half-sec {half_sec:g} --rank-window {rank_window[0]:g} {rank_window[1]:g} --bin-ms {bin_ms:g} "
               f"--step-ms {step_ms:g} --smooth-ms {smooth_arg} --awake-bin-ms {awake_bin_ms:g} "
               f"--n-components {n_components} --max-sec "
               f"{max_sec or 0:g} --n-random {n_random} --seed {seed}"
             + (f" --spindle-source {spindle_source}" if spindle_source != 'legacy' else '')
             + f" | {datetime.now().isoformat(timespec='seconds')}\n"
               f"awake inputs: {' ; '.join(pkl_files)}\n"
               f"awake: z-score per unit over the stim bins; PLSRegression(scale=False), y = +1 (45 left) / -1 (315 "
               f"left), fit on every trial; KDE = scipy gaussian_kde, Scott bandwidth, on LV1-{KDE_DIMS}; natural log")
    outs = []
    for block in blocks:
        try:
            outs.extend(run_block(model, config, block, shank, half_sec, rank_window, bin_ms / 1000.0,
                                  step_ms / 1000.0, smooth_values, n_random, seed, title, stamp, out_dir,
                                  align=align, spindle_source=spindle_source, log=log))
        except FileNotFoundError as exc:
            log(f"  [{block}] skipped - {exc}")
    return outs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('animal')
    parser.add_argument('date', help='YYMMDD')
    parser.add_argument('--blocks', default='pre,post', help='sleep blocks (default: %(default)s)')
    parser.add_argument('--shank', type=int, default=None,
                        help="use this shank's spindle list only (default: all shanks, overlapping events merged)")
    parser.add_argument('--align', choices=('onset', 'peak'), default='onset',
                        help='event alignment anchor (default: %(default)s)')
    parser.add_argument('--half-sec', type=float, default=0.5, help='view = anchor +- this (default: %(default)s)')
    parser.add_argument('--rank-window', type=float, nargs=2, default=None, metavar=('START', 'END'),
                        help='event value = median R over this window, s from alignment anchor (default: 0 HALF_SEC)')
    parser.add_argument('--bin-ms', type=float, default=20.0, help='sleep window (default: %(default)s)')
    parser.add_argument('--step-ms', type=float, default=10.0, help='sleep window step (default: %(default)s)')
    parser.add_argument('--smooth-ms', type=float, nargs='+', default=[0.0],
                        help='one or more Gaussian FWHM values for display-only temporal smoothing; 0 disables')
    parser.add_argument('--awake-bin-ms', type=float, default=100.0)
    parser.add_argument('--n-components', type=int, default=3)
    parser.add_argument('--max-sec', type=float, default=DEFAULT_MAX_SEC,
                        help='awake model: use at most this much of each stim period; 0 = all of it')
    parser.add_argument('--n-random', type=int, default=2000, help='random NREM onsets of the reference')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--spindle-source', choices=('legacy', 'bosc'), default='legacy',
                        help='spindle event list: legacy v1_spindles or spindle_bosc (default: %(default)s)')
    parser.add_argument('--out-dir', default=None, help='default: <sortout>/pls_task')
    args = parser.parse_args(argv)
    if any(x < 0 for x in args.smooth_ms):
        parser.error('--smooth-ms must be >= 0')
    blocks = tuple(b.strip() for b in args.blocks.split(',') if b.strip())
    outs = run_day(args.animal, args.date, blocks, args.shank, args.half_sec, args.rank_window, args.bin_ms,
                   args.step_ms, args.smooth_ms, args.awake_bin_ms, args.n_components, args.max_sec or None,
                   args.n_random, args.seed, args.out_dir, args.align, args.spindle_source)
    return 0 if outs else 1


if __name__ == '__main__':
    sys.exit(main())
