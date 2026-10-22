"""Build a compact review table and cross-shank depth profiles for this run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from server_fallback import resolve_output_folder


def main():
    registry = json.loads((sleep_paths.SLEEP_DIR / 'sleep_day_configs.json').read_text())
    folders = {day: Path(resolve_output_folder(
        Path(registry['CnL46_' + day]['rec_folder']) / 'low_freq' / 'v1_spindles'))
        for day in ('260729', '260730')}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for ax, (day, session) in zip(axes.flat, [('260729', 'pre'), ('260729', 'post'),
                                           ('260730', 'pre'), ('260730', 'post')]):
        for path in sorted(folders[day].glob(f'*_{session}_sh*_v1_spindles.json')):
            meta = json.loads(path.read_text())
            with np.load(path.with_suffix('.npz')) as data:
                power = data['mean_9_17_power']
                depth = data['depth_um']
                region = data['spindle_region']
                maximum = np.nanmax(power)
                shank = meta['parameters']['shank']
                if np.isfinite(maximum) and maximum > 0:
                    ax.plot(power / maximum, depth, label=f'sh{shank}')
                rows.append(dict(day=day, session=session, shank=shank,
                                 channel=meta['best_channel'], probe_y_um=meta['best_depth_um'],
                                 region_min_um=float(depth[region].min()) if region.any() else '',
                                 region_max_um=float(depth[region].max()) if region.any() else '',
                                 events=meta['event_count'], clean_nrem_min=round(meta['clean_nrem_sec'] / 60, 3),
                                 events_per_min=round(meta['event_count'] * 60 / meta['clean_nrem_sec'], 3),
                                 status=meta['status']))
        ax.set(title=f'CnL46 {day} {session}', xlabel='Power / maximum within shank', ylabel='Probe y (µm)')
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle('Spindle screening: NREM depth profiles (each shank normalized separately)')
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / 'depth_profiles.png', dpi=160)
    plt.close(fig)
    if rows:
        for folder in folders.values():
            with (folder / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ['# CnL46 spindle screening: July 29 and 30, 2026', '',
             'All available LFP shanks were screened with the default 9–17 Hz detector.',
             'These are candidate events; V1 identity and anatomical layer were not inferred.', '',
             '- July 29: pre/post shanks 0–7; fresh multi-shank NREM scoring (0,1,2,3,5,6).',
             '- July 30: pre shanks 0–7, post shanks 0–2 available at run time. Provisional NREM from shank 0 because only its spectral features exist. Post shanks 3–7 were unavailable.',
             '- July 30 post: no tracking; NREM is spectral-only.',
             '- Shanks 4 and 7 were screened as requested, but the existing sleep configuration flags weak spectral-tilt signal on July 29. Review these especially carefully.',
             '- Counts are per representative shank contact. Shared events across shanks are not deduplicated; do not sum them as unique spindle counts.',
             '- Coordinates are probe y, not cortical depth from pia. Endpoint maxima may indicate the peak lies beyond the sampled extent.',
             '- Fresh state files and per-shank plots are in each session recording\'s low_freq/v1_spindles folder. This combined report is saved in both locations; original labels and active configuration were preserved.', '',
             '[Depth profiles](depth_profiles.png) | [Machine-readable table](summary.csv)', '',
             '| Day | Sleep | Shank | Channel | Probe y (µm) | Region (µm) | Events | Clean NREM (min) |',
             '|---|---|---:|---:|---:|---|---:|---:|']
    for row in rows:
        lines.append(f'| {row["day"]} | {row["session"]} | {row["shank"]} | {row["channel"]} | {row["probe_y_um"]} | {row["region_min_um"]}–{row["region_max_um"]} | {row["events"]} | {row["clean_nrem_min"]} |')
    for folder in folders.values():
        (folder / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
