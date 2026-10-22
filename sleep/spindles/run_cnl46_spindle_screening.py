"""Reproducible July 29/30 screening using available LFP and spectral features.

Writes beside the band-power files; preserves active-day config and older labels.
July 30 uses provisional single-shank sleep scoring because only shank 0 has
precomputed features. Missing tracking is explicitly allowed and recorded.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import argparse
import gc
import json
import pickle
import traceback

import score_sleep_states as scorer
from detect_v1_spindles import process_file
from server_fallback import resolve_output_folder


def main():
    registry = json.loads((sleep_paths.SLEEP_DIR / 'sleep_day_configs.json').read_text())
    records = []
    for day in ('260729', '260730'):
        cfg = registry['CnL46_' + day]
        folder = Path(cfg['rec_folder'])
        scorer.rec_folder = str(folder)
        scorer.session_name = folder.name
        scorer.video_folder = folder.parent / 'video'
        day_out = Path(resolve_output_folder(folder / 'low_freq' / 'v1_spindles'))
        day_out.mkdir(parents=True, exist_ok=True)
        for session in ('pre', 'post'):
            label = folder.name + '_' + session
            state_path = day_out / (label + '_sleep_periods.pkl')
            band_path = folder / 'low_freq' / (label + '_all_shanks_band_powers.pkl')
            try:
                if not state_path.exists():
                    args = scorer.parse_args(['--allow-missing-velocity'])
                    if day == '260730':
                        args.shanks = [0]
                        args.min_valid_shanks = 1
                    state = scorer.score_block(band_path, session,
                                               dict(cfg[session], suffix='_' + session), args)
                    state['screening_note'] = ('Provisional single-shank (0) NREM labels'
                                               if day == '260730' else 'Fresh canonical multi-shank scoring')
                    scorer.print_report(state)
                    with state_path.open('wb') as stream:
                        pickle.dump(state, stream)
                    scorer.plot_result(state, day_out / (label + '_sleep_periods.png'))
                    del state
                files = sorted((folder / 'low_freq').glob(label + '_sh*_lfp_traces.npz'))
                for path in files:
                    shank = int(path.name.split('_sh')[1].split('_')[0])
                    print(f'RUN {label} shank {shank}', flush=True)
                    args = argparse.Namespace(shank=shank, exclude_channels=[], depth_range=None,
                                              peak_threshold=3., boundary_threshold=1.5,
                                              min_duration=.3, max_duration=3., merge_gap=.2,
                                              max_depth_gap_um=50., min_region_channels=3,
                                              out_dir=day_out)
                    result_path = day_out / (path.stem.replace('_lfp_traces', '_v1_spindles') + '.json')
                    try:
                        if not result_path.exists():
                            process_file(path, state_path, args)
                        result = json.loads(result_path.read_text())
                        records.append(dict(day=day, session=session, shank=shank,
                                            **{key: result[key] for key in ('status', 'best_channel', 'best_depth_um', 'event_count', 'clean_nrem_sec')}))
                    except Exception as error:
                        traceback.print_exc()
                        records.append(dict(day=day, session=session, shank=shank, error=str(error)))
                    gc.collect()
                    (day_out / 'summary.json').write_text(json.dumps(
                        [row for row in records if row['day'] == day], indent=2))
            except Exception as error:
                traceback.print_exc()
                records.append(dict(day=day, session=session, error=str(error)))
                (day_out / 'summary.json').write_text(json.dumps(
                    [row for row in records if row['day'] == day], indent=2))
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
