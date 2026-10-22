"""Compare three LFP filters on July 27, with fixed trials, contacts and scales."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import interp1d
from plot_layers import (DEFAULT_DAY, VIEW_SEC, LFP_DISPLAY_GAIN, plt, process_shank,
                         read_timing, passive_bounds, passive_psd, smooth_laminar_csd)

FILTERS = ((.5, 100), (.5, 300), (None, 300))
LABELS = ('0.5–100 Hz', '0.5–300 Hz', 'Low-pass 300 Hz\n(no additional high-pass)')


def compare(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    x, y = a[valid], b[valid]
    return dict(correlation=float(np.corrcoef(x, y)[0, 1]),
                rmse=float(np.sqrt(np.mean((x - y) ** 2))),
                relative_rmse=float(np.sqrt(np.mean((x - y) ** 2) / np.mean(y ** 2))))


def plot(shank, reference, means, csds, spectrum, output):
    time, depths = reference['time_sec'], reference['depth_um']
    rejected, filled = reference['rejected'], reference['filled']
    shown = (time >= VIEW_SEC[0]) & (time <= VIEW_SEC[1])
    gain = LFP_DISPLAY_GAIN * 25 * .42 / max(np.max(abs(means[:, shown][:, :, ~rejected])), 1e-12)
    limit = float(np.nanpercentile(abs(csds[:, shown]), 99)) or 1
    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True,
                             width_ratios=(.85, 1.7, 1.7, 1.7), layout='constrained')
    grid = np.arange(0, 776, 25.)
    power = np.full(len(grid), np.nan)
    power[np.searchsorted(grid, depths)] = np.where(rejected, np.nan, spectrum['band_power_uv2'][:, 0])
    power /= np.nanmax(power)
    axes[0].plot(power, grid, 'k', lw=1.5)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel('Normalized power\n(500 Hz–5 kHz)')
    axes[0].set_ylabel('Probe depth (µm; 0 nearer surface)')
    axes[0].set_title('Whole passive recording\nSame profile for all filters')
    axes[0].grid(alpha=.2)
    dense = np.arange(0, 776, 5.)
    for k, ax in enumerate(axes[1:]):
        csd = csds[k]
        display = np.full((len(dense), len(time)), np.nan)
        good = np.flatnonzero(np.isfinite(csd).all(axis=0))
        breaks = np.flatnonzero(np.diff(depths[good]) > 37.5) + 1
        for idx in np.split(good, breaks):
            if not len(idx):
                continue
            if len(idx) == 1:
                display[abs(dense - depths[idx[0]]) <= 2.5] = csd[:, idx[0]]
            else:
                mask = (dense >= depths[idx[0]]) & (dense <= depths[idx[-1]])
                display[mask] = interp1d(depths[idx], csd[:, idx], axis=1)(dense[mask]).T
        # Keep 0.5-ms samples when comparing filters with a 300-Hz cutoff.
        im = ax.pcolormesh(time[shown], dense, display[:, shown], cmap='jet',
                           shading='nearest', vmin=-limit, vmax=limit, rasterized=True)
        for ch, depth in enumerate(depths):
            label = str(reference['channel_labels'][ch]).rsplit(':', 1)[-1]
            if not rejected[ch]:
                ax.plot(time[shown], depth - gain * means[k, shown, ch], 'k', lw=.6)
            if k == 2:
                ax.text(1.01, depth, label + (' ×' if rejected[ch] else ''),
                        transform=ax.get_yaxis_transform(), fontsize=6, va='center',
                        color='r' if rejected[ch] else 'k')
        if filled.any():
            ax.plot(np.full(filled.sum(), VIEW_SEC[1]), depths[filled], ls='', marker='<',
                    color='k', ms=5, clip_on=False)
        bar = max(v for v in (.1, .2, .5, 1, 2, 5, 10, 20, 50, 100) if v * gain <= 45)
        ax.plot([.19] * 2, [depths[-1] + 55, depths[-1] + 55 - gain * bar], 'k', lw=2)
        ax.text(.185, depths[-1] + 55, f'{bar:g} µV', ha='right', va='bottom', fontsize=8)
        ax.set_xlim(*VIEW_SEC)
        ax.set_xlabel('Time from stimulus onset (s)')
        ax.set_title(f'{LABELS[k]}\nLFP over CSD')
    axes[0].set_ylim(depths[-1] + 70, -75)
    fig.colorbar(im, ax=list(axes[1:]), shrink=.6, pad=.03,
                 label='CSD (µV/mm²); shared scale\nblue = sink; red = source')
    fig.suptitle(f'CnL46 2026-07-27 — Shank {shank} — {len(reference["onset_sec"])} trials\n'
                 f'Same LFP gain (×{LFP_DISPLAY_GAIN:g}) and CSD limits across filters; fixed rejected contacts', fontsize=14)
    fig.get_layout_engine().set(rect=(0, .045, 1, .955))
    fig.text(.005, .006, f'Generated {datetime.now():%Y-%m-%d %H:%M:%S} | '
             'Baseline −0.5..0 s; third-order Butterworth forward/backward; resampled to 2 kHz; '
             'three-contact Hamming CSD; reference contact rejection from 0.5–100 Hz.\n'
             'Power: native-rate Welch, 2 s Hann, 50% overlap, integrated 500–5000 Hz over entire passive recording. '
             'No additional high-pass means existing acquisition/anti-alias filters remain.', fontsize=7)
    fig.savefig(output / f'shank{shank}_filter_comparison.png', dpi=160)
    plt.close(fig)
    return dict(lfp_depth_um_per_uv=gain, csd_limit_uv_per_mm2=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse', action='store_true')
    cli = parser.parse_args()
    nwb_dir = DEFAULT_DAY / 'CnL46_20260727'
    root = nwb_dir / 'layer_identification'
    output = root / 'filter_comparison'
    output.mkdir(exist_ok=True)
    metadata = json.loads((root / 'metadata.json').read_text())
    args = SimpleNamespace(**{key: Path(value) if key in ('dio', 'bad_channels', 'nwb_dir') else value
                             for key, value in metadata.items()})
    assert args.pre_sec == .5
    on, off, duration = read_timing(args)
    bounds = passive_bounds(args)
    rows, checks = [], []
    for shank in range(8):
        path = nwb_dir / f'CnL46_20260727sh{shank}.nwb'
        with np.load(root / f'shank{shank}_averages.npz') as saved:
            reference = {key: saved[key] for key in ('time_sec', 'depth_um', 'channel_labels',
                         'lfp_uv', 'rejected', 'filled', 'onset_sec')}
        cache = output / f'shank{shank}_filter_averages.npz'
        if cli.reuse and cache.exists():
            with np.load(cache) as stored:
                assert np.array_equal(stored['onset_sec'], on)
                assert np.array_equal(stored['channel_labels'], reference['channel_labels'])
                means = stored['lfp_uv']
        else:
            result = process_shank(path, args, on, off, duration, filter_bands=FILTERS)
            assert np.array_equal(result['channel_labels'], reference['channel_labels'])
            assert np.array_equal(result['time_sec'], reference['time_sec'])
            means = result['filter_lfp_uv']
        error = float(np.max(abs(means[0] - reference['lfp_uv'])))
        if error > 1e-8:
            raise ValueError(f'Baseline filter differs from saved mean: {error} µV')
        assert np.isfinite(means).all()
        assert np.allclose(means[:, reference['time_sec'] < 0].mean(axis=1), 0, atol=1e-8)
        csds = np.array([smooth_laminar_csd(v, reference['depth_um'], reference['rejected']) for v in means])
        np.savez_compressed(cache, lfp_uv=means, csd_uv_per_mm2=csds, filter_bands_json=json.dumps(FILTERS),
                            time_sec=reference['time_sec'], depth_um=reference['depth_um'],
                            channel_labels=reference['channel_labels'], rejected=reference['rejected'],
                            filled=reference['filled'], onset_sec=on, offset_sec=off)
        power_cache = output / f'shank{shank}_power_500_5000.npz'
        if cli.reuse and power_cache.exists():
            with np.load(power_cache) as saved:
                assert np.array_equal(saved['channel_labels'], reference['channel_labels'])
                assert np.array_equal(saved['passive_sample_bounds'], bounds)
                spectrum = {key: saved[key] for key in saved.files if key != 'channel_labels'}
        else:
            spectrum = passive_psd(path, reference['channel_labels'], bounds)
            np.savez_compressed(power_cache, **spectrum, channel_labels=reference['channel_labels'])
        assert np.isfinite(spectrum['psd_uv2_per_hz']).all()
        assert np.all(spectrum['band_power_uv2'] >= 0)
        for k, ref in ((1, 0), (2, 0), (2, 1)):
            for name, lo, hi in [('display', 0, .2), ('extended', -.1, .5)]:
                mask = (reference['time_sec'] >= lo) & (reference['time_sec'] <= hi)
                for kind, array in [('LFP', means[:, :, ~reference['rejected']]), ('CSD', csds)]:
                    rows.append(dict(shank=shank, filter=LABELS[k].replace('\n', ' '),
                                     reference_filter=LABELS[ref].replace('\n', ' '), signal=kind,
                                     window=name, **compare(array[k, mask], array[ref, mask])))
        scales = plot(shank, reference, means, csds, spectrum, output)
        checks.append(dict(shank=shank, baseline_filter_max_error_uv=error,
                           psd_windows=int(spectrum['psd_window_count']), **scales))
        (output / 'validation.json').write_text(json.dumps(checks, indent=2))
        print(f'Shank {shank}: filter comparison and 500–5000 Hz power saved', flush=True)
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for k, ref in ((1, 0), (2, 0), (2, 1)):
        for kind in ('LFP', 'CSD'):
            group = [r for r in rows if r['filter'] == LABELS[k].replace('\n', ' ')
                     and r['reference_filter'] == LABELS[ref].replace('\n', ' ')
                     and r['signal'] == kind and r['window'] == 'display']
            entry = dict(filter=LABELS[k], reference_filter=LABELS[ref], signal=kind)
            for key in ('correlation', 'rmse', 'relative_rmse'):
                values = [r[key] for r in group]
                entry[key] = dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values)))
            summary.append(entry)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'metadata.json').write_text(json.dumps(dict(source_metadata=metadata,
        filters=FILTERS, trials=len(on), baseline_sec=[-.5, 0], display_sec=VIEW_SEC,
        fixed_rejection='from original 0.5–100 Hz mean', shared_scales='per shank across all three filters',
        power_band_hz=[500, 5000], power_method='native-rate Welch, 2 s Hann, 50% overlap, integrated density',
        note='Not an exact reproduction of Senzai methods; temporal filters are sensitivity variants.'), indent=2))
    print(f'Completed filter comparison: {output}', flush=True)


if __name__ == '__main__':
    main()
