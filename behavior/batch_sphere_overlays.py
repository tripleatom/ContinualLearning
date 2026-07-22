"""
Batch-render plot_overlay (correctness + speed 2-panel) for every
behavior-only session JSON in \\\\10.129.151.108\\xieluanlabs\\xl_cl\\behavior\\sphere\\.

Outputs land at <sphere>/<Animal>/trial_overlays/<session-stem>_overlay.png.
Existing PNGs are skipped unless --overwrite is passed.
"""
import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

import plot_trial_traces as ptt
from server_fallback import resolve_output_folder


SPHERE_DEFAULT = r"\\10.129.151.108\xieluanlabs\xl_cl\behavior\sphere"
SESSION_RE = re.compile(r"^(?P<animal>[A-Za-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})_Session(?P<sess>\d+)_Data\.json$")


def parse_session_name(path):
    m = SESSION_RE.match(path.name)
    if not m:
        return None
    return m.group('animal'), path.name[:-len('_Data.json')]


def build_traces_from_json(json_path):
    """Mirror readDIO_grating's per-trial slicing on a sphere _Data.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    dlc_x = np.asarray(data.get('dlcX', []), dtype=float)
    dlc_y = np.asarray(data.get('dlcY', []), dtype=float)
    step_time = np.asarray(data.get('stepTime', []), dtype=float)
    session_t0 = data.get('startTime')
    trials = data.get('trials', []) or []

    if (dlc_x.size == 0 or step_time.size == 0 or session_t0 is None
            or not trials):
        return []

    n = min(dlc_x.size, dlc_y.size, step_time.size)
    dlc_x, dlc_y, step_time = dlc_x[:n], dlc_y[:n], step_time[:n]

    out = []
    for tr in trials:
        # Mirror readDIO_grating.py: slice [stimulusOnsetTime, choiceTime] using
        # searchsorted on stepTime. stimulusOnsetTime/choiceTime are Unix epoch;
        # stepTime is session-relative seconds; convert via session_t0.
        stim_on = tr.get('stimulusOnsetTime')
        stim_off = tr.get('choiceTime')
        if stim_on is None or stim_off is None or stim_off < stim_on:
            continue
        i0 = int(np.searchsorted(step_time, stim_on - session_t0, side='left'))
        i1 = int(np.searchsorted(step_time, stim_off - session_t0, side='right'))
        i0 = max(0, i0)
        i1 = min(n, i1)
        if i1 <= i0:
            continue
        cleaned, _stats = ptt.clean_position({
            'x': dlc_x[i0:i1],
            'y': dlc_y[i0:i1],
            't': step_time[i0:i1],
        })
        if cleaned['x'].size == 0:
            continue
        out.append({
            'trial_index': tr.get('trialNumber'),
            'choice': tr.get('choice'),
            'correct': tr.get('correct'),
            'rewarded_on_left': tr.get('rewardedOnLeft'),
            'x': cleaned['x'], 'y': cleaned['y'],
            't': cleaned['t'], 'v': cleaned['v'],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sphere-root', default=SPHERE_DEFAULT,
                    help='Root folder containing *_Data.json sessions and per-animal subfolders.')
    ap.add_argument('--position-units-per-cm', type=float, default=10.0,
                    help='dlcX/dlcY units per cm. Sphere rig is calibrated 1 px = 1 mm, '
                         'so 10 px = 1 cm (default).')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-render even when the output PNG already exists.')
    ap.add_argument('--limit', type=int, default=0,
                    help='Process at most N sessions (0 = all). Useful for smoke tests.')
    args = ap.parse_args()

    sphere = Path(args.sphere_root)
    if not sphere.exists():
        print(f"ERROR: sphere root not found: {sphere}", file=sys.stderr)
        sys.exit(1)

    ptt.POSITION_UNITS_PER_CM = float(args.position_units_per_cm)

    json_files = sorted(sphere.glob('*_Data.json'))
    if args.limit > 0:
        json_files = json_files[:args.limit]

    n_written = n_skipped = n_failed = 0
    animals_touched = set()

    for jp in json_files:
        parsed = parse_session_name(jp)
        if parsed is None:
            print(f"SKIP non-matching name: {jp.name}")
            n_skipped += 1
            continue
        animal, stem = parsed
        out_dir = resolve_output_folder(sphere / animal / 'trial_overlays')
        out_path = out_dir / f"{stem}_overlay.png"

        if out_path.exists() and not args.overwrite:
            print(f"SKIP exists: {out_path}")
            n_skipped += 1
            animals_touched.add(animal)
            continue

        try:
            traces = build_traces_from_json(jp)
        except Exception as e:
            print(f"FAIL load {jp.name}: {e}")
            traceback.print_exc()
            n_failed += 1
            continue

        if not traces:
            print(f"SKIP no usable trials: {jp.name}")
            n_skipped += 1
            continue

        try:
            ptt.plot_overlay(traces, out_path)
            print(f"OK {len(traces)} trials -> {out_path}")
            n_written += 1
            animals_touched.add(animal)
        except Exception as e:
            print(f"FAIL render {jp.name}: {e}")
            traceback.print_exc()
            n_failed += 1

    print(f"\nProcessed {len(json_files)} sessions across {len(animals_touched)} animals: "
          f"{n_written} written, {n_skipped} skipped, {n_failed} failed.")


if __name__ == '__main__':
    main()
