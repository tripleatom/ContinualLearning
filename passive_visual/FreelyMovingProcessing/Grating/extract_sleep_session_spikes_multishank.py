r"""Extract sorted-unit sleep spikes for sessions sorted per-shank (no single
combined curated_analyzer) — the layout used by CnL46's 8shank32 probe.

extract_sleep_session_spikes.py assumes a single `sortout_folder/curated_analyzer`
holding every shank (via its 'group' property). That layout does not exist for
CnL46: each shank was sorted independently into
`shank{N}/sorting_results_*/sorting_analyzer`, exactly the layout
GratingExport.py's raw-shank fallback already handles. This script reuses that
same fallback (`GratingExport._find_shank_raw_analyzers`) so unit ids come out
as identical "shankX_unitY" strings — required for the sleep spikes to line up
with all_units_tuning.pkl in the tuning-coupling pipeline.

Locating the sleep window
--------------------------
Per-shank sorting here runs on one NWB per shank, itself a concatenation of that
day's .rec files in the order recorded in the day's `conversion_list.txt`
(written by rec2nwb). `day_concat_boundaries()` reads that file and returns each
recording's [start, end) sample range in the shared concatenated-sorter space —
the same space `sorting.get_unit_spike_train()` returns spikes in — so the sleep
window for e.g. "presleep" is exact, not inferred.

Output matches extract_sleep_session_spikes.py's schema exactly (metadata/window/
spike_data with 'spike_times_sec' per unit), so every downstream consumer
(UPState.py, tc_coupling.py, apply_merged_decoder_to_sleep.py) works unchanged.

Usage
-----
    python extract_sleep_session_spikes_multishank.py --list         # show boundaries, don't extract
    python extract_sleep_session_spikes_multishank.py                # extract pre + post
    python extract_sleep_session_spikes_multishank.py --blocks post
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from spikeinterface import load_sorting_analyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from GratingExport import _find_shank_raw_analyzers, _has_sort_data  # noqa: E402


def day_concat_boundaries(experiment_data_day_folder, verbose=True):
    """
    Read `conversion_list.txt` and return {recording_label: (start, end)} sample
    ranges in the shared per-shank sorter space, in file order.

    `recording_label` is the file's own prefix up to (not including) the
    "_YYYYMMDD_HHMMSS" timestamp -- e.g. "CnL46_presleep" from
    "CnL46_presleep_20260729_170047.rec" -- since that is the stable part callers
    match against ("presleep", "postsleep", "passive", "task", "task2", ...).
    """
    folder = Path(experiment_data_day_folder)
    path = folder / 'conversion_list.txt'
    if not path.is_file():
        raise FileNotFoundError(
            f"No conversion_list.txt in {folder} -- cannot recover the sample "
            f"offsets used when this day's per-shank NWBs were built.")

    pattern = re.compile(r'^\s*\d+\.\s+(\S+?)\.rec:\s+(\d+)\s+timestamps\s*$')
    rows = []
    for line in path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            rows.append((m.group(1), int(m.group(2))))
    if not rows:
        raise ValueError(f"Could not parse any recording rows from {path}")

    boundaries, cursor = {}, 0
    for name, n_samples in rows:
        boundaries[name] = (cursor, cursor + n_samples)
        cursor += n_samples

    if verbose:
        print(f"Day concatenation order ({path}):")
        for name, (start, end) in boundaries.items():
            print(f"  {name:40s} [{start:>12d}, {end:>12d})  "
                  f"{(end - start) / 30000.0:8.1f} s")
        print(f"  total: {cursor} samples")
    return boundaries


def find_block_boundary(boundaries, keyword):
    """Look up a recording's (start, end) by a case-insensitive substring match."""
    hits = [v for k, v in boundaries.items() if keyword.lower() in k.lower()]
    if not hits:
        raise KeyError(f"No recording in conversion_list.txt matches '{keyword}'. "
                       f"Available: {list(boundaries)}")
    if len(hits) > 1:
        raise KeyError(f"'{keyword}' matches more than one recording: "
                       f"{[k for k in boundaries if keyword.lower() in k.lower()]}")
    return hits[0]


def _resolve_sources(sortout_folder):
    """
    Build the same per-unit source list GratingExport.py builds: curated_analyzer
    if present, else one raw sorting_analyzer per shank. Returns (sources, fs,
    use_curated) with each source providing 'sorting', 'shank_of', 'quality_of'.
    """
    sortout_path = Path(sortout_folder)
    curated_path = sortout_path / 'curated_analyzer'
    use_curated = curated_path.exists()
    sources, fs = [], None

    if use_curated:
        sorting_analyzer = load_sorting_analyzer(curated_path)
        sorting = sorting_analyzer.sorting
        fs = sorting.sampling_frequency
        group_prop = sorting.get_property('group')
        group_map = ({uid: int(g) for uid, g in zip(sorting.unit_ids, group_prop)}
                     if group_prop is not None else {})
        label_prop = sorting.get_property('unit_label')
        if label_prop is None:
            raise ValueError(f"curated_analyzer missing 'unit_label': {curated_path}")
        label_map = {uid: str(l) for uid, l in zip(sorting.unit_ids, label_prop)}
        sources.append({
            'sorting': sorting, 'unit_ids': sorting.unit_ids,
            'shank_of': lambda uid: group_map.get(uid, None),
            'quality_of': lambda uid: label_map[uid],
        })
        print(f"Loaded curated_analyzer: {curated_path}")
    else:
        if not _has_sort_data(sortout_path):
            raise FileNotFoundError(
                f"Neither curated_analyzer nor any shank*/sorting_results_*/"
                f"sorting_analyzer found under {sortout_path}")
        for shank_id, analyzer_path in _find_shank_raw_analyzers(sortout_path):
            sa = load_sorting_analyzer(analyzer_path)
            s = sa.sorting
            if fs is None:
                fs = s.sampling_frequency
            print(f"Loaded raw (uncurated) sorting_analyzer for shank {shank_id}: "
                 f"{analyzer_path} ({len(s.unit_ids)} units, quality='unsorted')")
            sources.append({
                'sorting': s, 'unit_ids': s.unit_ids,
                'shank_of': (lambda uid, _shank=shank_id: _shank),
                'quality_of': (lambda uid: 'unsorted'),
            })
    return sources, fs, use_curated


def extract_sleep_spikes_multishank(sortout_folder, sleep_start, sleep_end,
                                    sleep_id, overwrite=True):
    """
    Per-shank equivalent of extract_sleep_session_spikes.extract_sleep_session_spikes.

    Same output schema (metadata/window/spike_data), same unit-id convention
    ("shankX_unitY") as GratingExport, so a sleep spikes pkl written here matches
    the units in all_units_tuning.pkl one-to-one.
    """
    sortout_path = Path(sortout_folder)
    sleep_start, sleep_end = int(sleep_start), int(sleep_end)
    if sleep_end <= sleep_start:
        raise ValueError("sleep_end must be greater than sleep_start.")

    sources, fs, use_curated = _resolve_sources(sortout_path)
    window_duration_sec = (sleep_end - sleep_start) / fs
    print(f"Sampling frequency : {fs} Hz")
    print(f"Sleep window       : samples {sleep_start} - {sleep_end} "
          f"({window_duration_sec:.1f} s)")

    pkl_file = sortout_path / f"sleep_spikes_{sleep_id}.pkl"
    if pkl_file.exists() and not overwrite:
        print(f"{pkl_file} exists and overwrite=False - skipping.")
        return pkl_file

    spike_data = {}
    for source in sources:
        sorting = source['sorting']
        for unit_id in source['unit_ids']:
            spike_train = sorting.get_unit_spike_train(unit_id)
            mask = (spike_train >= sleep_start) & (spike_train < sleep_end)
            spike_times_sec = (spike_train[mask] - sleep_start) / fs

            shank = source['shank_of'](unit_id)
            quality = source['quality_of'](unit_id)
            uid_str = f"shank{shank}_unit{unit_id}" if shank is not None else f"unit{unit_id}"

            spike_data[uid_str] = {
                'spike_times_sec': spike_times_sec,
                'n_spikes': len(spike_times_sec),
                'unit_id': int(unit_id),
                'shank': shank,
                'quality': quality,
            }

    print(f"Extracted {len(spike_data)} units "
         f"({'curated' if use_curated else 'raw/unsorted'})")

    output = {
        'metadata': {
            'sleep_id': sleep_id,
            'extraction_date': datetime.now().isoformat(),
            'sampling_frequency': fs,
            'n_units': len(spike_data),
            'sortout_folder': str(sortout_path),
            'curation_status': 'curated' if use_curated else 'unsorted',
        },
        'window': {
            'sleep_start_sample': sleep_start,
            'sleep_end_sample': sleep_end,
            'window_duration_sec': window_duration_sec,
        },
        'spike_data': spike_data,
    }

    print(f"Saving -> {pkl_file}")
    with open(pkl_file, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--blocks', default='presleep,postsleep',
                        help='Comma-separated recording-name keywords to extract '
                             '(matched against conversion_list.txt), default '
                             'presleep,postsleep')
    parser.add_argument('--list', action='store_true',
                        help='Print the day concatenation boundaries and exit')
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--animal', default=None,
                        help='Override grating_config.ANIMAL_ID (does not edit the file)')
    parser.add_argument('--date', default=None,
                        help='Override grating_config.EXPERIMENT_DATE (6-digit, e.g. 260729)')
    parser.add_argument('--sortout', default=None,
                        help='Override grating_config.SORTOUT_FOLDER')
    args = parser.parse_args(argv)

    from grating_config import ANIMAL_ID, EXPERIMENT_DATE, SORTOUT_FOLDER
    animal_id = args.animal or ANIMAL_ID
    experiment_date = args.date or EXPERIMENT_DATE
    sortout_folder = Path(args.sortout) if args.sortout else Path(SORTOUT_FOLDER)
    if args.animal or args.date:
        # SORTOUT_FOLDER is derived from ANIMAL_ID/EXPERIMENT_DATE in
        # grating_config.py; rebuild it the same way rather than reusing the
        # config's (now stale) value when either override is given.
        sortout_folder = (sortout_folder.parent.parent / animal_id
                         / f"{animal_id}_20{experiment_date}")

    # sortout layout:       sortout/<ANIMAL>/<ANIMAL>_20<DATE>
    # experiment_data layout: experiment_data/<ANIMAL>/<DATE>/<ANIMAL>_20<DATE>
    # -- one more path segment (the bare DATE folder) than sortout has.
    experiment_data_animal_folder = Path(
        str(sortout_folder).replace('sortout', 'experiment_data')).parent
    day_folder = (experiment_data_animal_folder / experiment_date
                 / f"{animal_id}_20{experiment_date}")
    boundaries = day_concat_boundaries(day_folder)
    if args.list:
        return 0

    for keyword in [b.strip() for b in args.blocks.split(',') if b.strip()]:
        label = 'pre' if 'presleep' in keyword.lower() else (
            'post' if 'postsleep' in keyword.lower() else keyword)
        start, end = find_block_boundary(boundaries, keyword)
        sleep_id = f"{experiment_date}_sleep_{label}"
        print(f"\n=== Extracting {sleep_id} ({keyword}: samples {start}-{end}) ===")
        pkl_path = extract_sleep_spikes_multishank(
            sortout_folder=sortout_folder, sleep_start=start, sleep_end=end,
            sleep_id=sleep_id, overwrite=args.overwrite)
        print(f"  -> {pkl_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
