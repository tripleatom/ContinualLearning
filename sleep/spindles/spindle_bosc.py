# SPDX-License-Identifier: GPL-3.0-or-later
"""BOSC spindle comparison beside the legacy spindle detector.

Python adaptation of Turnbull et al. (2026), Mendeley V3, and BOSC2 by
Jeremy B. Caplan, Adam M. Hughes, Tara A. Whitten and Clayton T. Dickson
(copyright 2010, GPL-3.0-or-later). See spindle_bosc.md for attribution,
verified parameters, and explicit differences from the MATLAB scripts.

python sleep/spindles/spindle_bosc.py --animal CnL46 --dates 260729 260730
Uses the legacy run's saved channel and state file for a paired comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import argparse
import csv
import gc
import json
import pickle
from fractions import Fraction

import numpy as np
from scipy.signal import fftconvolve, resample_poly
from scipy.special import digamma
from scipy.stats import chi2

from detect_v1_spindles import nrem_samples, runs

from server_fallback import resolve_existing_file, resolve_output_folder

FREQUENCIES = 2. ** (np.arange(21) / 4.)
SPINDLE_IDX = np.arange(11, 18)  # MATLAB F(12:18), 6.727...19.027 Hz
CHI2_GEOMEAN = np.exp(digamma(1) + np.log(2))
SOURCE = 'https://data.mendeley.com/datasets/hzw6zpzp9j/3'


def morlet_power(x, fs, frequency, width=6.):
    """BOSC_tf normalization and full-convolution crop; no band-pass first."""
    sigma = width / (2 * np.pi * frequency)
    t = -3.6 * sigma + np.arange(int(np.floor(7.2 * sigma * fs)) + 1) / fs
    wave = np.exp(-t*t / (2 * sigma*sigma)) * np.exp(2j * np.pi * frequency * t)
    wave /= np.sqrt(sigma * np.sqrt(np.pi))
    y = fftconvolve(x, wave, mode='full')
    start = int(np.ceil(len(wave) / 2)) - 1
    return np.abs(y[start:start + len(x)]) ** 2


def bisquare_fit(x, y):
    """Leverage-adjusted Tukey IRLS (4.685); not MATLAB bitwise parity."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    leverage = np.sum(design * (design @ np.linalg.pinv(design.T @ design)), axis=1)
    for _ in range(100):
        residual = (y - design @ beta) / np.sqrt(np.maximum(1 - leverage, 1e-8))
        scale = np.median(np.abs(residual - np.median(residual))) / .67448975
        if scale < 1e-12:
            break
        u = residual / (4.685 * scale)
        weights = np.where(np.abs(u) < 1, (1 - u*u)**2, 0.)
        if np.count_nonzero(weights) < 3:
            raise ValueError('Background robust fit has too few weighted frequencies')
        new = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    return beta[::-1]  # slope, intercept


def optimized_background(power, frequencies=FREQUENCIES):
    """Median -> geometric mean; remove >99.9% power; robust refit."""
    power = np.asarray(power, float)
    if power.ndim != 2 or power.shape[0] != len(frequencies) or power.shape[1] < 10:
        raise ValueError('Background needs frequency x time power samples')
    logp = np.log(np.maximum(power, np.finfo(float).tiny))
    correction = digamma(1) + np.log(2) - np.log(chi2.ppf(.5, 2))
    med = (np.median(logp, axis=1) + correction) / np.log(10)
    initial = np.polyfit(np.log10(frequencies), med, 1)
    initial_bg = 10 ** np.polyval(initial, np.log10(frequencies))
    cutoff = chi2.ppf(.999, 2) * initial_bg / CHI2_GEOMEAN
    logp[power >= cutoff[:, None]] = np.nan
    refined = (np.nanmedian(logp, axis=1) + correction) / np.log(10)
    if not np.isfinite(refined).all():
        raise ValueError('Nonfinite background spectrum')
    coefficients = bisquare_fit(np.log10(frequencies), refined)
    background = 10 ** np.polyval(coefficients, np.log10(frequencies))
    return background, coefficients, med, refined


def sustained_mask(power, threshold, minimum_samples, valid):
    result = np.zeros(len(power), bool)
    for a, b in runs((power > threshold) & valid):
        if b - a >= minimum_samples:
            result[a:b] = True
    return result


def combine_events(detected, ratio, valid, fs, *, merge_gap=.1, min_duration=.35):
    merged = []
    for a, b in runs(np.any(detected, axis=0) & valid):
        if merged and a - merged[-1][1] <= merge_gap * fs and valid[merged[-1][1]:a].all():
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    events = []
    for a, b in merged:
        if ((b - a) / fs < min_duration or a == 0 or b == len(valid)
                or not valid[a - 1] or not valid[b]):
            continue
        fidx, offset = np.unravel_index(np.argmax(ratio[:, a:b]), (len(SPINDLE_IDX), b-a))
        events.append(dict(onset_s=a/fs, peak_s=(a+offset)/fs, offset_s=b/fs,
                           duration_s=(b-a)/fs, peak_frequency_hz=float(FREQUENCIES[SPINDLE_IDX[fidx]]),
                           peak_power_ratio=float(ratio[fidx, a+offset]),
                           n_detected_frequencies=int(np.any(detected[:, a:b], axis=1).sum())))
    return events


def detect_bosc(x, fs, nrem, *, confidence=.99, cycles=5., width=6.,
                min_duration=.35, merge_gap=.1):
    x = np.asarray(x, float)
    nrem = np.asarray(nrem, bool)
    if x.ndim != 1 or nrem.shape != x.shape or not np.isfinite(fs) or fs <= 2 * FREQUENCIES[-1]:
        raise ValueError('Expected one LFP trace, matching mask and fs >64 Hz')
    if not (0 < confidence < 1 and cycles > 0 and width > 0 and min_duration > 0 and merge_gap >= 0):
        raise ValueError('Invalid BOSC parameters')
    nrem = nrem & np.isfinite(x)
    if nrem.sum() < 20 * fs or np.std(x[nrem]) <= 0:
        raise ValueError('Need at least 20 seconds of nonflat clean NREM')
    center, scale = np.mean(x[nrem]), np.std(x[nrem], ddof=1)
    x = (x - center) / scale
    power = np.full((len(FREQUENCIES), len(x)), np.nan, dtype=np.float32)
    background_valid = np.zeros(len(x), bool)
    valid = np.zeros(len(x), bool)
    bg_guard = int(np.ceil(width * fs / FREQUENCIES[0]))
    det_guard = int(np.ceil((width + cycles) * fs / FREQUENCIES[SPINDLE_IDX[0]]))
    for a, b in runs(nrem):
        if b - a <= 2 * det_guard + int(min_duration * fs):
            continue
        for i, frequency in enumerate(FREQUENCIES):
            power[i, a:b] = morlet_power(x[a:b], fs, frequency, width)
        valid[a + det_guard:b - det_guard] = True
        if b - a > 2 * bg_guard:
            background_valid[a + bg_guard:b - bg_guard] = True
    if background_valid.sum() < 10 * fs:
        raise ValueError('Insufficient NREM for background after 6-second edge guards')
    bg, coefficients, observed, refined = optimized_background(power[:, background_valid])
    threshold = chi2.ppf(confidence, 2) * bg / CHI2_GEOMEAN
    ratio = power[SPINDLE_IDX] / threshold[SPINDLE_IDX, None]
    detected = np.stack([sustained_mask(power[i], threshold[i], cycles * fs / FREQUENCIES[i], valid)
                         for i in SPINDLE_IDX])
    events = combine_events(detected, ratio, valid, fs, merge_gap=merge_gap, min_duration=min_duration)
    return dict(events=events, power_ratio=ratio, detected=detected, valid=valid,
                background=bg, threshold=threshold, fit_coefficients=coefficients,
                log_geomean_observed=observed, log_geomean_refined=refined,
                background_exposure_s=float(background_valid.sum()/fs),
                clean_nrem_s=float(valid.sum()/fs), center=center, scale=scale,
                pepisode=detected[:, valid].mean(axis=1))


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_reference(reference, args):
    legacy = json.loads(reference.read_text())
    source = Path(resolve_existing_file(Path(legacy['lfp'])))
    states_path = Path(resolve_existing_file(Path(legacy['states'])))
    out = Path(resolve_output_folder(source.parent / 'spindle_bosc'))
    out.mkdir(parents=True, exist_ok=True)
    stem = source.stem.replace('_lfp_traces', '_spindle_bosc')
    result_path = out / (stem + '.json')
    if result_path.exists() and not args.overwrite:
        saved = json.loads(result_path.read_text())
        if saved['parameters'] != dict(fs=args.fs, confidence=args.confidence, cycles=args.cycles):
            raise ValueError(f'{result_path}: parameters differ; use --overwrite')
        return saved
    with np.load(source, allow_pickle=False) as data:
        ids = data['channel_ids'].astype(str)
        matches = np.flatnonzero(ids == str(legacy['best_channel']))
        if len(matches) != 1:
            raise ValueError('Legacy representative channel missing or ambiguous')
        traces = data['traces']
        x = traces[:, matches[0]].astype(float)
        del traces
        source_fs = float(data['sampling_rate'])
        start_sample = int(data['sleep_start_sample'])
        recording_offset = start_sample / float(data['original_fs'])
        label = str(data['session_name']) + '_' + str(data['sleep_session'])
    with states_path.open('rb') as stream:
        states = pickle.load(stream)
    if states['session'] != label or int(states.get('sleep_start_sample') or 0) != start_sample:
        raise ValueError('LFP/state timebase mismatch')
    native_mask = nrem_samples(states, len(x), source_fs)
    # Resample each finite NREM run independently: artifacts cannot bleed across gaps.
    fraction = Fraction(args.fs / source_fs).limit_denominator(10000)
    fs = source_fs * fraction.numerator / fraction.denominator
    if not np.isclose(fs, args.fs):
        raise ValueError('Cannot represent requested resampling ratio')
    resampled = np.full(int(np.ceil(len(x) * fs / source_fs)), np.nan)
    for a, b in runs(native_mask & np.isfinite(x)):
        y = resample_poly(x[a:b], fraction.numerator, fraction.denominator)
        left = int(np.ceil(a * fs / source_fs))
        right = min(len(resampled), int(np.floor(b * fs / source_fs)))
        # Native epoch edges lie on integer seconds in the existing scorer.
        if abs(a * fs / source_fs - round(a * fs / source_fs)) > 1e-6:
            raise ValueError('NREM run start not on target grid; use native --fs')
        resampled[left:right] = y[:right-left]
    del x
    nrem = nrem_samples(states, len(resampled), fs) & np.isfinite(resampled)
    result = detect_bosc(resampled, fs, nrem, confidence=args.confidence, cycles=args.cycles)
    with reference.with_suffix('.csv').open() as stream:
        old = list(csv.DictReader(stream))
    def covered(e):
        a, b = int(np.ceil(float(e['onset_s'])*fs)), int(np.ceil(float(e['offset_s'])*fs))
        return a >= 0 and b <= len(result['valid']) and b > a and result['valid'][a:b].all()
    comparable_old = [e for e in old if covered(e)]
    for event in result['events']:
        event.update(channel=legacy['best_channel'], depth_um=legacy['best_depth_um'],
                     shank=legacy['parameters']['shank'])
        for field in ('onset', 'peak', 'offset'):
            event[field + '_recording_s'] = event[field + '_s'] + recording_offset
        event['overlaps_legacy'] = any(event['onset_s'] < float(e['offset_s']) and event['offset_s'] > float(e['onset_s']) for e in comparable_old)
    old_with_overlap = sum(any(float(e['onset_s']) < b['offset_s'] and float(e['offset_s']) > b['onset_s'] for b in result['events']) for e in comparable_old)
    fields = ['onset_s', 'peak_s', 'offset_s', 'duration_s', 'peak_frequency_hz', 'peak_power_ratio',
              'n_detected_frequencies', 'channel', 'depth_um', 'shank', 'onset_recording_s',
              'peak_recording_s', 'offset_recording_s', 'overlaps_legacy']
    write_csv(out / (stem + '.csv'), result['events'], fields)
    step = max(1, int(round(fs / 50)))
    np.savez_compressed(out / (stem + '.npz'), frequencies=FREQUENCIES,
                        spindle_frequencies=FREQUENCIES[SPINDLE_IDX],
                        time_s=np.arange(0, len(resampled), step)/fs,
                        power_ratio=result['power_ratio'][:, ::step],
                        detected=result['detected'][:, ::step], valid=result['valid'][::step],
                        background=result['background'], threshold=result['threshold'],
                        pepisode=result['pepisode'], fit_coefficients=result['fit_coefficients'])
    summary = dict(session=label, shank=legacy['parameters']['shank'], channel=legacy['best_channel'],
                   depth_um=legacy['best_depth_um'], event_count=len(result['events']),
                   legacy_event_count=len(old), legacy_comparable_count=len(comparable_old),
                   legacy_events_overlapping_bosc=old_with_overlap,
                   bosc_events_overlapping_legacy=sum(e['overlaps_legacy'] for e in result['events']),
                   clean_nrem_s=result['clean_nrem_s'], background_exposure_s=result['background_exposure_s'],
                   rate_per_nrem_min=len(result['events'])*60/result['clean_nrem_s'],
                   median_duration_s=float(np.median([e['duration_s'] for e in result['events']])) if result['events'] else None,
                   lfp=str(source), states=str(states_path), legacy_reference=str(reference),
                   source_code=SOURCE, source_fs=source_fs,
                   parameters=dict(fs=args.fs, confidence=args.confidence, cycles=args.cycles),
                   fixed_parameters=dict(width=6, merge_gap_s=.1, min_duration_s=.35,
                                         frequencies_hz=FREQUENCIES.tolist(), spindle_bins_hz=FREQUENCIES[SPINDLE_IDX].tolist()),
                   status='BOSC candidates; physiological validity not established',
                   adaptations=['Python bisquare IRLS, not numerically validated against MATLAB robustfit',
                                '250 Hz default; each finite clean NREM run resampled and transformed separately',
                                'Background fitted to clean NREM; existing artifact mask replaces unspecified squared-amplitude threshold',
                                'Half-open sample intervals and explicit edge guards replace MATLAB indexing/truncation quirks',
                                'Legacy representative channel retained for paired comparison'])
    plot_result(resampled, fs, result, old, out / stem, summary)
    result_path.write_text(json.dumps(summary, indent=2))
    print(f'{stem}: BOSC={len(result["events"])} old={len(old)} overlap-old={old_with_overlap}/{len(comparable_old)}', flush=True)
    return summary


def plot_result(x, fs, result, old, prefix, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].loglog(FREQUENCIES, 10**result['log_geomean_observed'], '.-', label='Median-derived background estimate')
    axes[0, 0].loglog(FREQUENCIES, result['background'], label='Optimized background fit')
    axes[0, 0].loglog(FREQUENCIES, result['threshold'], '--', label=f'{100*summary["parameters"]["confidence"]:g}% power threshold')
    axes[0, 0].set(xlabel='Frequency (Hz)', ylabel='Normalized wavelet power')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(FREQUENCIES[SPINDLE_IDX], result['pepisode'], width=.8)
    axes[0, 1].set(xlabel='Frequency (Hz)', ylabel=f'Fraction of valid NREM meeting {summary["parameters"]["cycles"]:g}-cycle rule')
    axes[1, 0].hist([e['duration_s'] for e in result['events']], bins=20, alpha=.6, label='BOSC')
    axes[1, 0].hist([float(e['duration_s']) for e in old], bins=20, alpha=.4, label='Legacy')
    axes[1, 0].set(xlabel='Duration (s)', ylabel='Candidates'); axes[1, 0].legend()
    axes[1, 1].eventplot([[e['peak_s']/60 for e in result['events']], [float(e['peak_s'])/60 for e in old]], lineoffsets=[1, 0])
    axes[1, 1].set(yticks=[0, 1], yticklabels=['Legacy', 'BOSC'], xlabel='Sleep-local time (min)')
    fig.suptitle(f'{summary["session"]} sh{summary["shank"]} ch{summary["channel"]}: BOSC {len(result["events"])} / legacy {len(old)}')
    fig.savefig(str(prefix) + '_summary.png', dpi=140); plt.close(fig)
    if not result['events']:
        return
    order = np.argsort([e['peak_power_ratio'] for e in result['events']])
    chosen = [order[int(round(q*(len(order)-1)))] for q in (.1, .5, .9)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), layout='constrained')
    for col, index in enumerate(chosen):
        e = result['events'][index]
        a, b = max(0, int((e['onset_s']-1)*fs)), min(len(x), int((e['offset_s']+1)*fs))
        t = np.arange(a, b)/fs
        axes[0, col].plot(t, x[a:b], color='0.2', lw=.7)
        axes[0, col].set_title(f'{[10,50,90][col]}th percentile power ratio; peak {e["peak_s"]:.3f}s\n'
                              f'{e["duration_s"]:.2f}s; {e["peak_frequency_hz"]:.2f}Hz')
        mesh = axes[1, col].pcolormesh(t, FREQUENCIES[SPINDLE_IDX],
                                      np.log2(np.maximum(result['power_ratio'][:, a:b], 1e-6)),
                                      shading='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        axes[2, col].pcolormesh(t, FREQUENCIES[SPINDLE_IDX], result['detected'][:, a:b], shading='auto', cmap='Greys', vmin=0, vmax=1)
        for row in range(3):
            axes[row, col].axvline(e['onset_s'], color='green', lw=.8)
            axes[row, col].axvline(e['offset_s'], color='green', lw=.8)
        axes[2, col].set_xlabel('Sleep-local seconds')
    axes[0, 0].set_ylabel('Saved broadband LFP')
    axes[1, 0].set_ylabel('Wavelet frequency (Hz)')
    axes[2, 0].set_ylabel('Cycle-qualified detection\nfrequency (Hz)')
    fig.colorbar(mesh, ax=list(axes[1]), label='log2(power / threshold); 0 = threshold')
    fig.suptitle(f'{prefix.name}: reproducibly selected examples, not confirmed spindles')
    fig.savefig(str(prefix) + '_examples.png', dpi=140); plt.close(fig)


def write_report(out, rows):
    lines = ['# BOSC versus legacy spindle screening', '',
             'Source-informed Python adaptation of the authors\' Mendeley V3 MATLAB code: ' + SOURCE, '',
             'Same representative channels and saved NREM labels as the old screening. '
             'BOSC uses 6.73–19.03 Hz wavelet bins, 99% power, five cycles, 0.1 s merging and 0.35 s minimum duration. '
             'Legacy detection used 9–17 Hz amplitude thresholds. Counts are per channel, not unique cross-shank events.', '',
             '**These are candidate detections, not an accuracy assessment.** '
             'The frequency band and decision rules both differ. Some raw examples remain dominated by sharp transients. '
             'The background model is a straight line in log–log power; inspect its fit to the observed spectrum, '
             'especially near preprocessing cutoffs and broad spectral peaks. More events or overlap does not prove validity.', '',
             'Python robust regression has not been validated numerically against MATLAB robustfit. '
             'The default resampling to 250 Hz, clean-NREM background, segmented artifact handling and corrected interval indexing '
             'are documented adaptations. July 30 retains the old provisional shank-0 NREM labels, including spectral-only post-sleep.', '',
             'All event CSVs contain sleep-local and whole-recording onset, peak and offset. '
             'Example columns are selected by the 10th, 50th and 90th percentiles of event power ratio. '
             'Peak frequency denotes the maximum wavelet power/threshold ratio within the event, '
             'which need not be the frequency that passed the duration criterion.', '',
             '| Session | Shank | Channel | BOSC | Legacy | Legacy overlapping BOSC / comparable | BOSC / NREM min | Plots |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        stem = f'{row["session"]}_sh{row["shank"]}_spindle_bosc'
        lines.append(f'| {row["session"]} | {row["shank"]} | {row["channel"]} | {row["event_count"]} | '
                     f'{row["legacy_event_count"]} | {row["legacy_events_overlapping_bosc"]}/{row["legacy_comparable_count"]} | '
                     f'{row["rate_per_nrem_min"]:.2f} | [Summary]({stem}_summary.png) · [Examples]({stem}_examples.png) |')
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--animal', default='CnL46')
    p.add_argument('--dates', nargs='+', default=['260729', '260730'])
    p.add_argument('--shanks', nargs='+', type=int)
    p.add_argument('--fs', type=float, default=250.)
    p.add_argument('--confidence', type=float, default=.99)
    p.add_argument('--cycles', type=float, default=5.)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    registry = json.loads((sleep_paths.SLEEP_DIR / 'sleep_day_configs.json').read_text())
    for day in args.dates:
        low = Path(resolve_existing_file(Path(registry[args.animal+'_'+day]['rec_folder']) / 'low_freq'))
        references = sorted((low/'v1_spindles').glob('*_v1_spindles.json'))
        if not references:
            raise FileNotFoundError(f'No legacy channel/state references under {low}')
        rows = []
        for reference in references:
            meta = json.loads(reference.read_text())
            if args.shanks and meta['parameters']['shank'] not in args.shanks:
                continue
            if meta['best_channel'] is None:
                print('SKIP no legacy representative:', reference.name, flush=True)
                continue
            print('BOSC', reference.name, flush=True)
            rows.append(process_reference(reference, args))
            gc.collect()
        out = Path(resolve_output_folder(low / 'spindle_bosc'))
        out.mkdir(parents=True, exist_ok=True)
        (out/'comparison.json').write_text(json.dumps(rows, indent=2))
        keys = ['session', 'shank', 'channel', 'depth_um', 'event_count', 'legacy_event_count',
                'legacy_comparable_count', 'legacy_events_overlapping_bosc', 'bosc_events_overlapping_legacy',
                'rate_per_nrem_min', 'median_duration_s']
        write_csv(out/'comparison.csv', [{k:r[k] for k in keys} for r in rows], keys)
        write_report(out, rows)


if __name__ == '__main__':
    main()
