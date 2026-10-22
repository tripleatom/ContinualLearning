"""Audit saved spindle candidates against unfiltered LFP; write beside results.

Examples are selected by peak-amplitude rank (10th, 50th, 90th percentiles),
not by visual appearance. Raw spectra provide a check independent of filtering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import csv
import gc
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram, welch

from detect_v1_spindles import nrem_samples, runs


def spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs)), nfft=int(fs * 4), detrend='linear')
    return f, p


def audit(path):
    meta = json.loads(path.read_text())
    out = path.parent / 'validation'
    out.mkdir(exist_ok=True)
    stem = path.stem
    with path.with_suffix('.csv').open() as stream:
        events = list(csv.DictReader(stream))
    if not events:
        return None
    with path.with_suffix('.npz').open('rb') as stream, np.load(stream) as data:
        best = int(data['best_index'])
    with np.load(meta['lfp']) as data:
        fs = float(data['sampling_rate'])
        traces = data['traces']
        x = traces[:, best].astype(float)
        del traces
    with Path(meta['states']).open('rb') as stream:
        state = pickle.load(stream)
    nrem = nrem_samples(state, len(x), fs)
    valid = np.zeros(len(x), bool)
    filtered = np.full(len(x), np.nan)
    amp = np.full(len(x), np.nan, dtype=np.float32)
    sos = butter(4, [9, 17], fs=fs, btype='bandpass', output='sos')
    guard = int(np.ceil(fs))
    for a, b in runs(nrem):
        if b - a <= 2 * guard + 32:
            continue
        y = sosfiltfilt(sos, x[a:b])
        env = uniform_filter1d(np.abs(hilbert(y)), size=max(1, int(.1 * fs)))
        filtered[a + guard:b - guard] = y[guard:-guard]
        amp[a + guard:b - guard] = env[guard:-guard]
        valid[a + guard:b - guard] = True
    z = (amp - np.mean(amp[valid])) / np.std(amp[valid])
    occupied = np.zeros(len(x), bool)
    for e in events:
        a, b = [int(round(float(e[k]) * fs)) for k in ('onset_s', 'offset_s')]
        occupied[max(0, a - guard):min(len(x), b + guard)] = True
    rows, spectra, controls = [], [], []
    for i, e in enumerate(events):
        a, peak, b = [int(round(float(e[k]) * fs)) for k in ('onset_s', 'peak_s', 'offset_s')]
        if not valid[a:b].all():
            raise ValueError(f'Event outside valid NREM: {path}, {i}')
        if not np.isclose(z[peak], float(e['peak_z']), atol=1e-4):
            raise ValueError(f'Reconstructed envelope differs: {path}, {i}')
        f, p = spectrum(x[a:b], fs)
        band = (f >= 9) & (f <= 17)
        broad = (f >= 4) & (f <= 40)
        search = (f >= 6) & (f <= 30)
        peak_hz = float(f[search][np.argmax(p[search])])
        frac = float(p[band].sum() / p[broad].sum())
        spectra.append(p / p[broad].sum())
        control = None
        # Nearest complete non-event clean-NREM window, same duration, <=30 s away.
        for shift in range(3, 31):
            for sign in (-1, 1):
                ca = a + sign * int(shift * fs)
                cb = ca + b - a
                if ca >= 0 and cb <= len(x) and valid[ca:cb].all() and not occupied[ca:cb].any():
                    control = (ca, cb)
                    break
            if control:
                break
        ratio = np.nan
        if control:
            _, cp = spectrum(x[control[0]:control[1]], fs)
            controls.append(cp / cp[broad].sum())
            ratio = float(p[band].sum() / cp[band].sum())
        rows.append(dict(event_index=i, onset_s=float(e['onset_s']), peak_s=float(e['peak_s']),
                         duration_s=float(e['duration_s']), peak_z=float(e['peak_z']),
                         raw_peak_6_30_hz=peak_hz, raw_sigma_fraction_4_40=frac,
                         sigma_power_vs_control=ratio,
                         control_onset_s=control[0] / fs if control else np.nan,
                         control_offset_s=control[1] / fs if control else np.nan))
    with (out / (stem + '_event_metrics.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    order = np.argsort([r['peak_z'] for r in rows])
    choices = [int(order[int(round(q * (len(order) - 1)))]) for q in (.1, .5, .9)]
    fig, axes = plt.subplots(5, 3, figsize=(16, 12), layout='constrained')
    for col, index in enumerate(choices):
        e, row = events[index], rows[index]
        peak = int(round(row['peak_s'] * fs))
        a, b = max(0, peak - int(3 * fs)), min(len(x), peak + int(3 * fs))
        t = np.arange(a, b) / fs - row['peak_s']
        lo = float(e['onset_s']) - row['peak_s']
        hi = float(e['offset_s']) - row['peak_s']
        stride = max(1, int(fs / 400))
        axes[0, col].plot(t[::stride], (x[a:b] - np.median(x[a:b]))[::stride], lw=.65, color='0.2')
        axes[0, col].set_title(f'{["10th", "50th", "90th"][col]} percentile peak amplitude\n'
                               f'Event {index}; sleep t={row["peak_s"]:.3f} s; {row["duration_s"]:.2f} s')
        axes[1, col].plot(t[::stride], filtered[a:b:stride], lw=.7)
        axes[1, col].plot(t[::stride], amp[a:b:stride], lw=1, color='orange')
        axes[2, col].plot(t[::stride], z[a:b:stride], lw=1)
        axes[2, col].axhline(1.5, ls='--', color='gray', label='Boundary 1.5 SD')
        axes[2, col].axhline(3, ls=':', color='red', label='Peak 3 SD')
        sf, st, sp = spectrogram(x[a:b], fs=fs, nperseg=int(.5 * fs),
                                 noverlap=int(.45 * fs), nfft=int(2 * fs), detrend='linear')
        keep = (sf >= 4) & (sf <= 40)
        db = 10 * np.log10(np.maximum(sp[keep], 1e-20))
        axes[3, col].pcolormesh(st + t[0], sf[keep], db, shading='auto', cmap='magma',
                               vmin=np.percentile(db, 10), vmax=np.percentile(db, 99))
        axes[3, col].axhline(9, color='cyan', lw=.7, ls='--')
        axes[3, col].axhline(17, color='cyan', lw=.7, ls='--')
        ea, eb = [int(round(float(e[k]) * fs)) for k in ('onset_s', 'offset_s')]
        f, p = spectrum(x[ea:eb], fs)
        axes[4, col].semilogy(f, p, label='Event raw LFP')
        if np.isfinite(row['control_onset_s']):
            ca, cb = int(round(row['control_onset_s'] * fs)), int(round(row['control_offset_s'] * fs))
            _, cp = spectrum(x[ca:cb], fs)
            axes[4, col].semilogy(f, cp, color='gray', label='Matched NREM control')
        axes[4, col].axvspan(9, 17, color='orange', alpha=.15)
        axes[4, col].set(xlim=(4, 40), xlabel='Frequency (Hz)')
        axes[4, col].legend(fontsize=7)
        for r in range(4):
            axes[r, col].axvline(lo, color='green', lw=.8)
            axes[r, col].axvline(hi, color='green', lw=.8)
            axes[r, col].set_xlim(-3, 3)
        axes[3, col].set_xlabel('Seconds from saved peak')
    for ax, label in zip(axes[:, 0], ['Raw LFP\nmedian removed', '9–17 Hz LFP\nand amplitude', 'NREM amplitude z', 'Raw spectrogram\nfrequency (Hz)', 'Raw PSD\ninput units²/Hz']):
        ax.set_ylabel(label)
    fig.suptitle(f'{stem}: channel {meta["best_channel"]}, probe y={meta["best_depth_um"]:g} µm\n'
                 'Green lines = saved onset/offset; spectrogram colors = local dB scale; candidates require review', fontsize=12)
    fig.savefig(out / (stem + '_examples.png'), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].hist([r['duration_s'] for r in rows], bins=np.linspace(.3, 3, 19))
    axes[0, 0].set(xlabel='Detected duration (s)', ylabel='Candidates')
    axes[0, 1].hist([r['raw_peak_6_30_hz'] for r in rows], bins=np.arange(6, 32, 1))
    axes[0, 1].axvspan(9, 17, color='orange', alpha=.2)
    axes[0, 1].set(xlabel='Strongest raw spectral peak in 6–30 Hz', ylabel='Candidates')
    axes[1, 0].plot(f, np.mean(spectra, axis=0), label='Events')
    if controls:
        axes[1, 0].plot(f, np.mean(controls, axis=0), label=f'NREM controls (n={len(controls)})')
    axes[1, 0].axvspan(9, 17, color='orange', alpha=.2)
    axes[1, 0].set(xlim=(4, 40), xlabel='Frequency (Hz)', ylabel='Mean normalized raw PSD')
    axes[1, 0].legend()
    edges = np.arange(0, len(x) / fs + 120, 120.)
    counts = np.histogram([r['peak_s'] for r in rows], edges)[0]
    exposure = np.array([valid[int(a * fs):int(b * fs)].sum() / fs / 60 for a, b in zip(edges[:-1], edges[1:])])
    rates = np.divide(counts, exposure, out=np.full(len(counts), np.nan), where=exposure >= .25)
    axes[1, 1].plot((edges[:-1] + 60) / 60, rates, '.-')
    axes[1, 1].set(xlabel='Sleep-local time (min)', ylabel='Candidates / clean NREM minute', title='2-min bins; ≥15 s clean NREM required')
    fig.suptitle(stem + ': characteristics and incidence (screening candidates)', fontsize=12)
    fig.savefig(out / (stem + '_characteristics.png'), dpi=150)
    plt.close(fig)
    summary = dict(file=stem, n=len(rows), examples=choices,
                   median_duration_s=float(np.median([r['duration_s'] for r in rows])),
                   raw_peak_in_9_17_fraction=float(np.mean([9 <= r['raw_peak_6_30_hz'] <= 17 for r in rows])),
                   median_sigma_vs_control=float(np.nanmedian([r['sigma_power_vs_control'] for r in rows])),
                   matched_controls=len(controls), reconstruction_checked=True)
    (out / (stem + '_audit.json')).write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)
    return summary


def main():
    registry = json.loads((sleep_paths.SLEEP_DIR / 'sleep_day_configs.json').read_text())
    for day in ('260729', '260730'):
        folder = Path(registry['CnL46_' + day]['rec_folder']) / 'low_freq/v1_spindles'
        summaries = []
        for path in sorted(folder.glob('*_v1_spindles.json')):
            print('AUDIT', path.name, flush=True)
            result = audit(path)
            if result:
                summaries.append(result)
            gc.collect()
        (folder / 'validation' / 'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
