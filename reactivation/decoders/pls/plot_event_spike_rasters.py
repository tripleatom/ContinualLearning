r"""Write one raw all-neuron spike raster for every analyzed spindle event."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRATING_DIR = HERE.parents[2] / 'DiscriminationTask' / 'grating'  # task_day_discovery lives here
for _p in (str(HERE), str(GRATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from pls_sleep_bias import load_sleep_spikes
from task_day_discovery import REGISTRY_FILE, load_registry, short_date


def raster_segments(spikes, unit_ids, order, peak, half_sec):
    """Return vertical line segments at every unbinned spike time."""
    pieces = []
    for rank, position in enumerate(order):
        times = spikes[unit_ids[position]]
        lo = np.searchsorted(times, peak - half_sec, side='left')
        hi = np.searchsorted(times, peak + half_sec, side='right')
        relative = times[lo:hi] - peak
        if relative.size:
            y0 = np.full(relative.size, rank - 0.38)
            y1 = np.full(relative.size, rank + 0.38)
            pieces.append(np.stack((np.column_stack((relative, y0)),
                                    np.column_stack((relative, y1))), axis=1))
    return np.concatenate(pieces) if pieces else np.empty((0, 2, 2))


def plot_event(spikes, unit_ids, order, selectivity, event, event_index, n_events,
               event_class, s45, s315, best, half_sec, title, out_path, dpi):
    onset, peak, offset = event
    segments = raster_segments(spikes, unit_ids, order, peak, half_sec)
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    if len(segments):
        ax.add_collection(LineCollection(segments, colors='black', linewidths=0.42,
                                         rasterized=True))
    ax.axvspan(onset - peak, offset - peak, color='gold', alpha=0.12,
               label='detected spindle interval')
    ax.axvline(onset - peak, color='darkorange', lw=0.7, ls='--')
    ax.axvline(0, color='tab:red', lw=1.0, label='spindle peak')
    ax.axvline(offset - peak, color='darkorange', lw=0.7, ls='--')
    q = max(1, len(order) // 4)
    ax.axhline(q - 0.5, color='tab:red', lw=0.55, alpha=0.6)
    ax.axhline(len(order) - q - 0.5, color='tab:blue', lw=0.55, alpha=0.6)
    ax.set(xlim=(-half_sec, half_sec), ylim=(len(order) - 0.5, -0.5),
           xlabel='time from spindle peak (s)',
           ylabel='neuron rank (awake 45-preferring to 315-preferring)')
    duration_ms = (offset - onset) * 1000
    d = s45 - s315
    ax.set_title(
        f'{title} | event {event_index + 1:04d}/{n_events} | {event_class}-like\n'
        f'peak={peak:.3f} s; spindle duration={duration_ms:.0f} ms; '
        f'S45={s45:.3f}, S315={s315:.3f}, D={d:+.3f}, best={best:.3f}',
        fontsize=10)
    ax.text(0.003, 0.997,
            f'top red boundary: 45-preferring quartile\n'
            f'bottom blue boundary: 315-preferring quartile\n'
            f'{len(segments):,} raw spikes; no binning or smoothing',
            transform=ax.transAxes, va='top', ha='left', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.82))
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return len(segments)


def run_block(animal, date, block, half_sec=0.5, dpi=120, out_dir=None, log=print):
    date = short_date(date)
    key = f'{animal}_{date}'
    config = load_registry().get(key)
    if config is None:
        raise KeyError(f'{key} is not in {REGISTRY_FILE}')
    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    result_pattern = (f'direct_reactivation_{sortout.name}_{block}_allshanks_'
                      f'bin100ms_step10ms_peak_n100.npz')
    result_path = sortout / 'pls_task' / result_pattern
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = np.load(result_path)
    unit_ids = result['unit_ids'].tolist()
    order = result['neuron_order'].astype(int)
    selectivity = result['selectivity']
    events = np.column_stack((result['onset_sec'], result['peak_sec'], result['offset_sec']))
    classes = result['event_class'].astype(int)
    zero = int(np.argmin(np.abs(result['lag_sec'])))
    s45 = result['observed_s45'][:, zero]
    s315 = result['observed_s315'][:, zero]
    best = result['observed_best'][:, zero]

    sleep_files = sorted(sortout.glob(f'sleep_spikes_*sleep_{block}.pkl'))
    if not sleep_files:
        raise FileNotFoundError(f'no sleep spike file for {block} in {sortout}')
    spikes, _ = load_sleep_spikes(sleep_files[0], unit_ids=unit_ids, verbose=False)
    missing = set(unit_ids) - set(spikes)
    if missing:
        raise ValueError(f'{len(missing)} analyzed units missing from sleep spike file')

    root = (Path(out_dir) if out_dir else sortout / 'pls_task' /
            'direct_raw_spike_rasters_100ms_template_peak')
    block_dir = root / block
    for label in (45, 315):
        (block_dir / f'{label}_like').mkdir(parents=True, exist_ok=True)
    rows = []
    title = f'{key} sleep {block.upper()}'
    for i, event in enumerate(events):
        label = int(classes[i])
        filename = f'event_{i + 1:04d}_peak_{event[1]:010.3f}s_{label}like.png'
        relative = Path(f'{label}_like') / filename
        n_spikes = plot_event(
            spikes, unit_ids, order, selectivity, event, i, len(events), label,
            float(s45[i]), float(s315[i]), float(best[i]), half_sec, title,
            block_dir / relative, dpi)
        rows.append(dict(event_number=i + 1, onset_sec=f'{event[0]:.6f}',
                         peak_sec=f'{event[1]:.6f}', offset_sec=f'{event[2]:.6f}',
                         duration_ms=f'{(event[2] - event[0]) * 1000:.3f}',
                         event_class=label, s45=f'{s45[i]:.8f}',
                         s315=f'{s315[i]:.8f}', difference=f'{s45[i] - s315[i]:.8f}',
                         best_similarity=f'{best[i]:.8f}', n_raw_spikes=n_spikes,
                         figure=str(relative)))
        if (i + 1) % 50 == 0 or i + 1 == len(events):
            log(f'  [{block}] saved {i + 1}/{len(events)} event rasters')
    index_path = block_dir / 'event_index.csv'
    with index_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log(f'  [{block}] folder: {block_dir}')
    log(f'  [{block}] index:  {index_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('animal')
    parser.add_argument('date')
    parser.add_argument('--blocks', default='pre,post')
    parser.add_argument('--half-sec', type=float, default=0.5)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--out-dir')
    args = parser.parse_args(argv)
    for block in [x.strip() for x in args.blocks.split(',') if x.strip()]:
        run_block(args.animal, args.date, block, args.half_sec, args.dpi, args.out_dir)


if __name__ == '__main__':
    main()
