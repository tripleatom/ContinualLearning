"""Repeated nested random trial subsets against the full-session LFP/CSD."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from plot_layers import DEFAULT_DAY, VIEW_SEC, process_shank, read_timing, smooth_laminar_csd


def metrics(value, reference):
    valid = np.isfinite(value) & np.isfinite(reference)
    a, b = value[valid], reference[valid]
    return dict(correlation=float(np.corrcoef(a, b)[0, 1]),
                rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                relative_rmse=float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2))))


def plot_comparison(shank, result, averages, output, session, pre_sec, stamp):
    time, depths = result['time_sec'], result['depth_um']
    window = (time >= max(VIEW_SEC[0], -pre_sec)) & (time <= VIEW_SEC[1])
    rejected = result['rejected']
    traces = [averages[0, 0], averages[0, 1], result['lfp_uv']]
    csds = [smooth_laminar_csd(v, depths, rejected) for v in traces]
    gain = 25 * .42 / max(np.max(abs(v[window][:, ~rejected])) for v in traces)
    limit = np.nanpercentile(abs(csds[-1][window]), 99)
    bar_x = time[window][0] + .93 * np.ptp(time[window])
    fig, axes = plt.subplots(2, 3, figsize=(17, 14), sharex=True, sharey=True,
                             layout='constrained')
    for col, (count, lfp, csd) in enumerate(zip((100, 400, len(result['onset_sec'])), traces, csds)):
        ax = axes[0, col]
        for ch in np.flatnonzero(~rejected):
            ax.plot(time[window], depths[ch] - gain * lfp[window, ch], color='k', lw=.65)
        ax.set_title(f'{count} trials — LFP (same voltage gain)')
        ax.plot([bar_x, bar_x], [820, 820 - 100 * gain], color='k', lw=2)
        ax.text(bar_x - .03 * np.ptp(time[window]), 820, '100 µV', ha='right', fontsize=8)
        # Native 25-µm contact grid preserves all unsupported depths as blanks.
        grid = np.arange(0, 776, 25)
        display = np.full((len(grid), window.sum()), np.nan)
        display[np.searchsorted(grid, depths)] = csd[window].T
        im = axes[1, col].pcolormesh(time[window][::4], grid, display[:, ::4],
                                    shading='nearest', cmap='jet', vmin=-limit, vmax=limit)
        axes[1, col].set_title(f'{count} trials — CSD; shared ±{limit:.0f} µV/mm²')
        axes[1, col].set_xlabel('Time from stimulus onset (s)')
        for row in range(2):
            axes[row, col].axvline(0, color='0.5', ls='--', lw=.8)
            axes[row, col].set_xlim(time[window][0], time[window][-1])
    axes[0, 0].set_ylim(835, -25)
    for ax in axes[:, 0]:
        ax.set_ylabel('Depth (µm; 0 nearer surface)')
    fig.colorbar(im, ax=list(axes[1]), shrink=.6, label='CSD (µV/mm²); blue = sink (−), green = 0, red = source (+)')
    fig.suptitle(f'{session} — Shank {shank}: trial-count comparison\n'
                 f'Random repeat 1: 100 trials nested in 400; baseline −{pre_sec:g}..0 s; unchanged filtering'
                 + (f"\nAuto-rejected channels (no LFP trace; CSD filled from neighbors where isolated): "
                    f"{', '.join(str(v).rsplit(':', 1)[-1] for v in result['channel_labels'][rejected])}"
                    if rejected.any() else ''), fontsize=14)
    # Reproducibility stamp: inputs and parameters behind this figure.
    fig.get_layout_engine().set(rect=(0, .03, 1, .97))
    fig.text(.005, .003, stamp, fontsize=6, color='0.35', va='bottom')
    fig.savefig(output / f'shank{shank}_trial_count_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260727)
    parser.add_argument('--reuse', action='store_true')
    # Session inputs must match the plot_layers.py run that saved the reference.
    parser.add_argument('--nwb-dir', type=Path, default=DEFAULT_DAY / 'CnL46_20260727')
    parser.add_argument('--dio', type=Path, default=DEFAULT_DAY / 'CnL46_drifting_grating_exp_20260727_131003_DIO.npz')
    parser.add_argument('--bad-channels', type=Path)
    parser.add_argument('--pre-sec', type=float, default=.5)
    parser.add_argument('--shanks', type=int, nargs='+', default=list(range(8)))
    cli = parser.parse_args()
    if cli.repeats < 1:
        parser.error('repeats must be positive')
    nwb_dir = cli.nwb_dir
    output = nwb_dir / 'layer_identification' / 'trial_count_comparison'
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(pre_sec=cli.pre_sec, dio_fs=30000, offset_sec=0, batch_trials=32,
                           bad_channels=cli.bad_channels or nwb_dir / 'bad_channels.txt',
                           dio=cli.dio)
    on, off, duration = read_timing(args)
    stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} | script={Path(__file__).name} | nwb_dir={nwb_dir}\n"
             f"dio={args.dio} | bad_channels={args.bad_channels}\n"
             f"seed={cli.seed}, repeats={cli.repeats} (repeat 1 shown), {len(on)} trials in full average, "
             f"baseline=-{args.pre_sec:g}..0 s, shown {VIEW_SEC[0]:g}..{VIEW_SEC[1]:g} s, LFP 0.5-100 Hz at 2 kHz, CSD = -d2V/dz2 after 3-contact Hamming smoothing")
    rng = np.random.default_rng(cli.seed)
    selected = np.array([rng.permutation(len(on))[:400] for _ in range(cli.repeats)])
    membership = np.zeros((len(on), cli.repeats, 2), bool)
    for repeat in range(cli.repeats):
        for k, count in enumerate((100, 400)):
            membership[selected[repeat, :count], repeat, k] = True
    rows, validations = [], []
    for shank in cli.shanks:
        cache = output / f'shank{shank}_subsets.npz'
        with np.load(output.parent / f'shank{shank}_averages.npz') as saved:
            reference = {key: saved[key] for key in ('lfp_uv', 'time_sec', 'depth_um', 'channel_labels', 'onset_sec')}
            # Contacts auto-rejected by plot_layers.py; none if its outputs predate that step.
            rejected = saved['rejected'] if 'rejected' in saved.files else np.zeros(len(saved['depth_um']), bool)
            reference['rejected'] = rejected
        if cli.reuse and cache.exists():
            with np.load(cache) as stored:
                assert np.array_equal(stored['selected_trial_indices'], selected)
                averages = stored['lfp_uv']
        else:
            sums = np.zeros((cli.repeats, 2) + reference['lfp_uv'].shape)
            def collect(index, epoch):
                sums[membership[index]] += epoch
            (path,) = nwb_dir.glob(f'*sh{shank}.nwb')
            result = process_shank(path, args,
                                   on, off, duration, epoch_callback=collect)
            assert np.array_equal(result['channel_labels'], reference['channel_labels'])
            assert np.array_equal(on, reference['onset_sec'])
            error = float(np.max(abs(result['lfp_uv'] - reference['lfp_uv'])))
            if error > 1e-8:
                raise ValueError(f'Full average differs from saved reference: {error}')
            validations.append(dict(shank=shank, full_average_max_error_uv=error))
            averages = sums / np.array([100, 400])[None, :, None, None]
            np.savez_compressed(cache, lfp_uv=averages, selected_trial_indices=selected,
                                time_sec=reference['time_sec'], depth_um=reference['depth_um'],
                                channel_labels=reference['channel_labels'], seed=cli.seed)
        full_csd = smooth_laminar_csd(reference['lfp_uv'], reference['depth_um'], rejected)
        for repeat in range(cli.repeats):
            for k, count in enumerate((100, 400)):
                lfp = averages[repeat, k]
                assert np.isfinite(lfp).all()
                assert np.allclose(lfp[reference['time_sec'] < 0].mean(axis=0), 0, atol=1e-8)
                csd = smooth_laminar_csd(lfp, reference['depth_um'], rejected)
                for name, lo, hi in [('display', -.1, .5), ('evoked', 0, .3)]:
                    mask = (reference['time_sec'] >= lo) & (reference['time_sec'] <= hi)
                    for kind, value, full in [('LFP', lfp[:, ~rejected], reference['lfp_uv'][:, ~rejected]),
                                              ('CSD', csd, full_csd)]:
                        rows.append(dict(shank=shank, repeat=repeat, trials=count, window=name,
                                         signal=kind, **metrics(value[mask], full[mask])))
        plot_comparison(shank, reference, averages, output, nwb_dir.name, args.pre_sec, stamp)
        print(f'Shank {shank}: comparison saved', flush=True)
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for shank in cli.shanks:
        for count in (100, 400):
            for kind in ('LFP', 'CSD'):
                group = [r for r in rows if r['shank'] == shank and r['trials'] == count
                         and r['signal'] == kind and r['window'] == 'display']
                item = dict(shank=shank, trials=count, signal=kind)
                for key in ('correlation', 'rmse', 'relative_rmse'):
                    values = [r[key] for r in group]
                    item[key] = dict(median=float(np.median(values)), min=float(min(values)), max=float(max(values)))
                summary.append(item)
    if cli.reuse and not validations and (output / 'summary.json').exists():
        # A redraw recomputes nothing; keep the validation of the run that built the cache.
        validations = json.loads((output / 'summary.json').read_text()).get('validation', [])
    (output / 'summary.json').write_text(json.dumps(dict(seed=cli.seed, repeats=cli.repeats,
        method='Uniform random without replacement; nested 100 within 400; identical subsets across shanks. '
               'Reference includes subset trials, so agreement is optimistic versus independent validation. '
               'Metrics flatten time and valid depth, without amplitude normalization. Power unchanged.',
        validation=validations, results=summary), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
