"""Detect MUA in the presleep and postsleep epochs of a concatenated session.

Unlike ``sleep/compute_sleep_mua.py``, which bins spike times from an existing
Phy/MountainSort curation, this script needs no sorting at all: it runs
MountainSort5's threshold detector directly (via ``mua_detect.detect_mua``) on
each sleep epoch and saves the raw threshold events plus a binned rate.

Epoch boundaries are read from ``conversion_list.txt`` rather than hardcoded.
Each per-shank NWB is the concatenation, in listed order, of that day's ``.rec``
files, so the cumulative timestamp counts give exact sample bounds for every
epoch. The script asserts the total against the NWB frame count before using it.

Artifact timestamps are reused from the shank's sorting run when one is present
under ``SORTOUT_ROOT``, which skips the detection pass and puts MUA events and
sorted spikes on identical exclusions. Shanks with no matching cache fall back
to detecting within the epoch. See ``REUSE_SORTING_ARTIFACTS``.

"presleep" and "postsleep" name recording blocks, not scored sleep: the animal
wakes and walks around inside both. Events are detected across the whole epoch
and stored as seconds from its start, so scored sleep or NREM windows are a
downstream mask on ``channel_spike_times``, not something applied here.

Output is two pickles in ``<session>/MUA/`` — one per sleep epoch, each holding
every shank, with per-channel event times inside. See the OUTPUT section below
for the exact layout.

Usage
-----
    python find_sleep_mua.py                      # all shanks, pre + post
    python find_sleep_mua.py --shanks 4 5 6       # selected shanks
    python find_sleep_mua.py --epochs pre         # one epoch
    python find_sleep_mua.py --limit-sec 120      # smoke test on 2 min/epoch
    python find_sleep_mua.py --overwrite          # redo finished outputs
"""

import argparse
import errno
import json
import pickle
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mua_detect import (  # noqa: E402
    detect_mua,
    has_artifact_cache,
    load_or_compute_artifacts,
    rescue_units,
)


# =====================================================
# SESSION / PATHS
# =====================================================
SESSION_FOLDER = Path(
    r'\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\260727\CnL46_20260727'
)
# Base name of the per-shank NWBs: "<NWB_BASE>sh<N>.nwb"
NWB_BASE = 'CnL46_20260727'
SHANKS = [0, 1, 2, 3, 4, 5, 6, 7]

# Epochs to analyze, mapped to the output suffix used elsewhere in the sleep
# pipeline (low_freq/*_pre_*, *_post_*). Keys are matched against the .rec
# filenames in conversion_list.txt.
EPOCHS = {'presleep': 'pre', 'postsleep': 'post'}

#: Results land in <SESSION_FOLDER>/MUA/, alongside the session's raw data.
OUTPUT_SUBFOLDER = 'MUA'

#: Root of the sorting output tree. Per-shank folders sit at
#: <SORTOUT_ROOT>/<animal>/<NWB_BASE>/shank<N>, matching MsSorting.py's own
#: layout, and each holds the ``_artifact_cache/`` that run wrote.
SORTOUT_ROOT = Path(r'\\10.129.151.88\xieluanlabs2\xl_cl\sortout')

# Reuse the sorting run's artifact timestamps instead of detecting per epoch.
# That cache is keyed on the whole-day frame count, so the repair is applied to
# the full concatenated recording and the epoch is sliced out afterwards — MUA
# events and sorted spikes then rest on identical artifact exclusions, and the
# detection pass (the slow part, several minutes per shank) is skipped entirely.
#
# Day-wide statistics are also what makes pre and post comparable. Neither sleep
# epoch is uniformly asleep — the animal wakes and walks around within them — so
# an epoch-local threshold is set partly by however much movement that epoch
# happened to contain, and the post/pre ratio this script reports would then
# rest on two differently-calibrated detectors. One threshold for the day
# removes that. Set False to detect within each epoch instead, which is also
# what happens automatically for any shank with no matching cache.
REUSE_SORTING_ARTIFACTS = True


# =====================================================
# DETECTION PARAMETERS
# =====================================================
# Deliberately looser than the 5.5 used for sorting: MUA is meant to be
# inclusive, capturing small spikes that would never be clusterable.
DETECT_THRESHOLD = 4.0

# -1 = negative-going only. The sorting config uses 0 (both signs) so isosplit
# sees positive-going units too; for a MUA rate the positive crossings are
# mostly noise and the rising phase of spikes already counted.
DETECT_SIGN = -1

# Refractory / peak-isolation window, also MS5's default.
DETECT_TIME_RADIUS_MSEC = 0.5

# Sites on these probes sit 25 um apart. 50 um keeps only the largest copy of a
# spike within +/- 2 sites, so one unit near the boundary between two channels
# is not counted twice. Set 0.0 for fully independent per-channel MUA, or None
# to suppress across the whole shank (MS5's own default).
DETECT_CHANNEL_RADIUS = 50.0

# Per-channel median/MAD rather than whitening. Whitening decorrelates channels,
# which is what the sorter wants but which can smear one large spike onto its
# neighbours and distort a per-channel MUA count.
SCALE_MODE = 'zscore'

FREQ_MIN, FREQ_MAX = 300, 6000
REMOVE_ARTIFACTS = True
CHUNK_DURATION_SEC = 1200.0

# Binning for the saved rate trace, and Gaussian smoothing width in bins.
# Same values compute_sleep_mua.py used, so rate traces stay comparable.
RATE_BIN_SEC = 0.020
SMOOTH_SIGMA_BINS = 2


# =====================================================
# EPOCH BOUNDARIES FROM conversion_list.txt
# =====================================================
_CONV_LINE = re.compile(r'^\s*(\d+)\.\s*(\S+\.rec)\s*:\s*(\d+)\s+timestamps', re.I)


def parse_conversion_list(session_folder: Path):
    """Return [(rec_name, n_timestamps), ...] in concatenation order."""
    path = Path(session_folder) / 'conversion_list.txt'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found — it defines the epoch order and lengths, so the '
            f'sleep windows cannot be derived without it.'
        )
    entries = []
    for line in path.read_text().splitlines():
        m = _CONV_LINE.match(line)
        if m:
            entries.append((m.group(2), int(m.group(3))))
    if not entries:
        raise ValueError(f'No "<file>.rec: <n> timestamps" lines parsed from {path}')
    return entries


def epoch_label(rec_name: str):
    """Map a .rec filename to an epoch key, or None if it is not one we want."""
    low = rec_name.lower()
    # Check postsleep before presleep so neither substring shadows the other.
    for key in sorted(EPOCHS, key=len, reverse=True):
        if key in low:
            return key
    return None


def epoch_windows(session_folder: Path, total_frames: int):
    """Return {epoch_key: (start_sample, end_sample)} for the wanted epochs.

    Bounds are cumulative offsets into the concatenated per-shank NWB.
    """
    entries = parse_conversion_list(session_folder)
    total_listed = sum(n for _, n in entries)
    if total_listed != total_frames:
        raise ValueError(
            f'conversion_list.txt totals {total_listed} timestamps but the NWB has '
            f'{total_frames} frames. The epoch bounds would be wrong — check that '
            f'the NWB was built from exactly these .rec files in this order.'
        )
    windows, offset = {}, 0
    for rec_name, n in entries:
        key = epoch_label(rec_name)
        if key is not None:
            windows[key] = (offset, offset + n)
        offset += n
    missing = set(EPOCHS) - set(windows)
    if missing:
        print(f'  WARNING: no .rec matched {sorted(missing)} — those epochs are skipped')
    return windows




# =====================================================
# OUTPUT
# =====================================================
# One pkl per epoch, holding every shank:
#
#   {
#     'session', 'epoch', 'suffix', 'sampling_frequency', 'duration_sec',
#     'epoch_start_sample', 'epoch_end_sample', 'limit_sec', 'partial',
#     'mua_bin_size', 'smooth_sigma_bins', 'mua_time', 'params',
#     'shanks': {
#        4: {'channel_ids', 'channel_locations', 'ycoord',
#            'channel_spike_times':  {channel_id: seconds from epoch start},
#            'channel_amplitudes':   {channel_id: sigma units, negative},
#            'channel_n_events', 'channel_rate_hz',
#            'n_events', 'population_rate_hz',
#            'artifact_source': 'sorting' | 'epoch' | 'none',
#            'mua_rate', 'mua_rate_smooth'},
#        ...
#     },
#   }
#
# Per-channel spike times are the primary product; pool them when you want one
# MUA train for a shank:
#
#   sh = data['shanks'][4]
#   pooled = np.sort(np.concatenate(list(sh['channel_spike_times'].values())))
#
# Per-channel binned rates are deliberately not stored — rebin from the times,
# which keeps the file a few hundred MB instead of several GB.


def epoch_pkl_path(out_dir: Path, epoch_key: str) -> Path:
    return out_dir / f'{NWB_BASE}_{EPOCHS[epoch_key]}_mua_events.pkl'


def load_epoch_pkl(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as e:
        print(f'  WARNING: could not read {path.name} ({e}) — starting fresh')
        return None


def load_mua(epoch: str, shank: int, session_folder=None, nwb_base=None):
    """Load one shank's MUA from an epoch pkl, flattened into a single dict.

    The per-shank entry is returned with the epoch-level metadata merged in, so
    everything (``mua_time``, ``channel_spike_times``, ``sampling_frequency``,
    ...) sits at one level instead of needing ``data['shanks'][shank]``.

        from find_sleep_mua import load_mua
        mua = load_mua('pre', shank=5)
        pooled = np.sort(np.concatenate(list(mua['channel_spike_times'].values())))
    """
    session_folder = Path(session_folder or SESSION_FOLDER)
    base = nwb_base or NWB_BASE
    alias = {v: k for k, v in EPOCHS.items()}
    epoch_key = alias.get(epoch, epoch)
    if epoch_key not in EPOCHS:
        raise ValueError(f'Unknown epoch {epoch!r}; expected one of '
                         f'{sorted(set(EPOCHS) | set(EPOCHS.values()))}')
    path = session_folder / OUTPUT_SUBFOLDER / f'{base}_{EPOCHS[epoch_key]}_mua_events.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if shank not in data.get('shanks', {}):
        raise KeyError(f'Shank {shank} not in {path.name}; present: '
                       f'{sorted(data.get("shanks", {}))}')
    merged = {k: v for k, v in data.items() if k != 'shanks'}
    merged.update(data['shanks'][shank])
    return merged


def dump_pkl(path: Path, obj):
    """pickle.dump with the pipeline's out-of-space fallback to the backup server."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        try:
            from server_fallback import mirror_on_backup_server
        except ImportError:
            raise
        backup = mirror_on_backup_server(path.parent)
        if backup is None:
            raise
        backup.mkdir(parents=True, exist_ok=True)
        alt = backup / path.name
        print(f'  Out of space — retrying on backup server: {alt}')
        with open(alt, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return alt


# =====================================================
# ARTIFACTS
# =====================================================

def sorting_shank_folder(shank: int) -> Path:
    """Where MsSorting.py put this shank's outputs, including _artifact_cache/."""
    animal = NWB_BASE.split('_')[0]
    return SORTOUT_ROOT / animal / NWB_BASE / f'shank{shank}'


def resolve_artifacts(rec, shank: int, out_dir: Path, suffix: str, args):
    """Decide how this shank's artifacts are handled.

    Returns ``(recording, detect_mua_kwargs, source)``. ``recording`` is the
    full-day recording to slice the epoch out of — already artifact-repaired
    when the sorting cache was reused, untouched otherwise.

    The cache key covers the recording's frame count, so it only matches on the
    whole concatenated day. That is why the repair is applied here, before the
    epoch is sliced, rather than being left to ``detect_mua``: handing it a
    frame-sliced epoch could never hit the sort's cache.
    """
    if not REMOVE_ARTIFACTS:
        return rec, {'remove_artifacts': False, 'artifact_cache_folder': None}, 'none'

    if REUSE_SORTING_ARTIFACTS:
        sort_dir = sorting_shank_folder(shank)
        if has_artifact_cache(rec, sort_dir):
            repaired, timestamps, _ = load_or_compute_artifacts(
                rescue_units(rec, verbose=args.verbose),
                cache_folder=sort_dir,
                verbose=args.verbose,
            )
            n_flagged = int(sum(len(t) for t in timestamps))
            print(f'  artifacts reused from sorting ({n_flagged} flagged samples): {sort_dir}')
            return repaired, {'remove_artifacts': False, 'artifact_cache_folder': None}, 'sorting'
        print(f'  no matching artifact cache under {sort_dir}')
        print('  -> detecting artifacts within the epoch instead')

    # Epoch-local detection: a full detection pass, and a threshold calibrated
    # only on this epoch, so it is not directly comparable across pre and post.
    return rec, {
        'remove_artifacts': True,
        'artifact_cache_folder': out_dir / '_artifact_cache' / f'{suffix}_sh{shank}',
    }, 'epoch'


# =====================================================
# PER-SHANK DETECTION
# =====================================================

def detect_shank(shank: int, epoch_key: str, window, out_dir: Path, args):
    """Run detection for one shank of one epoch. Returns the per-shank dict."""
    import spikeinterface.extractors as se

    suffix = EPOCHS[epoch_key]
    nwb_path = SESSION_FOLDER / f'{NWB_BASE}sh{shank}.nwb'
    if not nwb_path.exists():
        print(f'  NWB not found: {nwb_path} — skipping')
        return None

    rec = se.read_nwb_recording(str(nwb_path))
    fs = rec.get_sampling_frequency()
    total_frames = rec.get_num_frames()

    start, end = window
    if args.limit_sec is not None:
        end = min(end, start + int(args.limit_sec * fs))
    if not (0 <= start < end <= total_frames):
        raise ValueError(f'Bad window [{start}, {end}) for {total_frames} frames')

    rec_src, artifact_kwargs, artifact_source = resolve_artifacts(
        rec, shank, out_dir, suffix, args)

    # Sliced after the repair, so the epoch inherits the whole-day artifact
    # treatment when that is where the timestamps came from.
    rec_epoch = rec_src.frame_slice(start_frame=start, end_frame=end)
    n_ch = rec_epoch.get_num_channels()
    dur = (end - start) / fs
    print(f'  window samples [{start}, {end}) = {start/fs:.1f}-{end/fs:.1f} s  '
          f'({dur:.1f} s, {n_ch} ch)')

    t0 = time.time()
    ev = detect_mua(
        rec_epoch,
        detect_threshold=DETECT_THRESHOLD,
        detect_sign=DETECT_SIGN,
        detect_time_radius_msec=DETECT_TIME_RADIUS_MSEC,
        detect_channel_radius=DETECT_CHANNEL_RADIUS,
        scale_mode=SCALE_MODE,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        chunk_duration_sec=CHUNK_DURATION_SEC,
        verbose=args.verbose,
        **artifact_kwargs,
    )
    elapsed = time.time() - t0

    # Split events by channel. Times are seconds from the start of the epoch.
    locs = np.asarray(rec.get_channel_locations())
    times_sec = ev.times / ev.sampling_frequency
    channel_spike_times, channel_amplitudes = {}, {}
    for m, cid in enumerate(ev.channel_ids):
        sel = ev.channel_indices == m
        key = cid.item() if hasattr(cid, 'item') else cid
        channel_spike_times[key] = times_sec[sel]
        channel_amplitudes[key] = ev.amplitudes[sel]

    _, rate = ev.rate_histogram(RATE_BIN_SEC)
    pop_rate = len(ev.times) / dur

    print(f'  {len(ev.times)} events, {pop_rate:.1f} Hz population rate '
          f'({pop_rate / n_ch:.2f} Hz/channel) in {elapsed:.1f} s')

    return {
        'shank': shank,
        'sampling_frequency': float(fs),
        'epoch_start_sample': int(start),
        'epoch_end_sample': int(end),
        'channel_ids': np.asarray(ev.channel_ids),
        'channel_locations': locs,
        'ycoord': locs[:, 1] if locs.ndim == 2 and locs.shape[1] > 1 else None,
        'channel_spike_times': channel_spike_times,
        'channel_amplitudes': channel_amplitudes,
        'channel_n_events': {k: int(len(v)) for k, v in channel_spike_times.items()},
        'channel_rate_hz': {k: float(len(v) / dur) for k, v in channel_spike_times.items()},
        'n_events': int(len(ev.times)),
        'n_channels': int(n_ch),
        'population_rate_hz': float(pop_rate),
        'mean_amplitude': float(np.mean(ev.amplitudes)) if len(ev.times) else float('nan'),
        'mua_rate': rate.astype(np.float32),
        'mua_rate_smooth': gaussian_filter1d(rate, SMOOTH_SIGMA_BINS).astype(np.float32),
        'duration_sec': float(dur),
        'elapsed_sec': float(elapsed),
        # 'sorting' | 'epoch' | 'none'. Worth reading before params['remove_artifacts'],
        # which records False for reused artifacts because detect_mua's own stage
        # was skipped — the repair had already been applied to the full day.
        'artifact_source': artifact_source,
        'params': ev.params,
    }


def process_epoch(epoch_key, window, shanks, out_dir: Path, args):
    """Detect every requested shank for one epoch and write that epoch's pkl."""
    suffix = EPOCHS[epoch_key]
    pkl_path = epoch_pkl_path(out_dir, epoch_key)
    existing = load_epoch_pkl(pkl_path)

    # A --limit-sec run writes the same filename from a truncated window, so
    # existence alone must not count as done — otherwise a smoke-test file gets
    # passed off as a finished epoch.
    prev_partial = bool(existing.get('partial', True)) if existing else False
    if existing and prev_partial and args.limit_sec is None:
        print(f'  {pkl_path.name} exists but is PARTIAL — recomputing from scratch')
        existing = None

    now_partial = args.limit_sec is not None
    if existing and existing.get('epoch') == epoch_key:
        data = existing
        data['shanks'] = dict(data.get('shanks', {}))
    else:
        data = {'session': NWB_BASE, 'epoch': epoch_key, 'suffix': suffix, 'shanks': {}}

    for shank in shanks:
        print(f'\n--- {epoch_key} | shank {shank} ---')
        if shank in data['shanks'] and not args.overwrite and prev_partial == now_partial:
            print(f'  already in {pkl_path.name} — skipping (use --overwrite to redo)')
            continue
        try:
            res = detect_shank(shank, epoch_key, window, out_dir, args)
        except Exception as e:
            print(f'  FAILED: {e}')
            traceback.print_exc()
            continue
        if res is None:
            continue

        data['shanks'][shank] = res
        # Epoch-level metadata. Every shank of an epoch spans the same window,
        # so these are shank-independent; they are taken from the shank that
        # just ran rather than assumed.
        data.update({
            'sampling_frequency': res['sampling_frequency'],
            'epoch_start_sample': res['epoch_start_sample'],
            'epoch_end_sample': res['epoch_end_sample'],
            'duration_sec': res['duration_sec'],
            'limit_sec': float('nan') if args.limit_sec is None else float(args.limit_sec),
            'partial': now_partial,
            'mua_bin_size': RATE_BIN_SEC,
            'smooth_sigma_bins': SMOOTH_SIGMA_BINS,
            'mua_time': (np.arange(len(res['mua_rate'])) + 0.5) * RATE_BIN_SEC,
            'params': res['params'],
        })

        # Write after every shank so a crash late in the run costs one shank,
        # not the whole epoch.
        saved = dump_pkl(pkl_path, data)
        size_mb = saved.stat().st_size / 1024 ** 2
        print(f'  saved -> {saved.name} ({len(data["shanks"])} shank(s), {size_mb:.1f} MB)')

    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shanks', type=int, nargs='+', default=None,
                    help=f'shanks to process (default {SHANKS})')
    ap.add_argument('--epochs', nargs='+', default=None,
                    choices=sorted(EPOCHS) + [v for v in EPOCHS.values()],
                    help='epochs to process (default: all)')
    ap.add_argument('--limit-sec', type=float, default=None,
                    help='only analyze this many seconds from each epoch start '
                         '(smoke test)')
    ap.add_argument('--overwrite', action='store_true',
                    help='recompute shanks already present in the epoch pkl')
    ap.add_argument('--quiet', dest='verbose', action='store_false',
                    help='suppress per-chunk detection progress')
    args = ap.parse_args()

    shanks = args.shanks if args.shanks is not None else SHANKS
    if args.epochs is None:
        epochs = list(EPOCHS)
    else:
        alias = {v: k for k, v in EPOCHS.items()}
        epochs = [alias.get(e, e) for e in args.epochs]

    import spikeinterface.extractors as se
    probe_nwb = SESSION_FOLDER / f'{NWB_BASE}sh{shanks[0]}.nwb'
    if not probe_nwb.exists():
        raise FileNotFoundError(f'{probe_nwb} not found — check SESSION_FOLDER/NWB_BASE')
    total_frames = se.read_nwb_recording(str(probe_nwb)).get_num_frames()

    windows = epoch_windows(SESSION_FOLDER, total_frames)
    out_dir = SESSION_FOLDER / OUTPUT_SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'MUA detection (no sorting) — {NWB_BASE}')
    print(f'  session : {SESSION_FOLDER}')
    print(f'  output  : {out_dir}')
    print(f'  shanks  : {shanks}')
    print(f'  epochs  : {epochs}')
    print(f'  detect  : threshold={DETECT_THRESHOLD} sign={DETECT_SIGN} '
          f'channel_radius={DETECT_CHANNEL_RADIUS} scale={SCALE_MODE}')
    for k in epochs:
        if k in windows:
            s, e = windows[k]
            print(f'  {k:10s}: samples [{s}, {e})  = {(e - s) / 30000:.1f} s '
                  f'-> {epoch_pkl_path(out_dir, k).name}')
    print('=' * 70)

    results = {}
    for epoch_key in epochs:
        if epoch_key not in windows:
            continue
        print('\n' + '#' * 70)
        print(f'EPOCH: {epoch_key}')
        print('#' * 70)
        results[epoch_key] = process_epoch(
            epoch_key, windows[epoch_key], shanks, out_dir, args)

    # -------- summary --------
    rows = []
    for epoch_key, data in results.items():
        for shank, res in sorted(data.get('shanks', {}).items()):
            rows.append({'epoch': epoch_key, 'shank': shank,
                         'population_rate_hz': res['population_rate_hz'],
                         'n_events': res['n_events'],
                         'partial': bool(data.get('partial', False))})
    if not rows:
        print('\nNothing processed.')
        return

    summary_path = out_dir / f'{NWB_BASE}_mua_summary.json'
    summary_path.write_text(json.dumps(rows, indent=2))

    print('\n' + '=' * 70)
    print('SUMMARY (population rate, Hz)')
    if any(r['partial'] for r in rows):
        print('  * = truncated by --limit-sec, not a full epoch')
    print(f'{"shank":>6}  {"pre":>12}  {"post":>12}  {"post/pre":>9}')
    by = {(r['epoch'], r['shank']): r for r in rows}

    def cell(r):
        return '-' if r is None else f'{r["population_rate_hz"]:.1f}' + ('*' if r['partial'] else '')

    for sh in sorted({r['shank'] for r in rows}):
        pre, post = by.get(('presleep', sh)), by.get(('postsleep', sh))
        ratio = ('-' if pre is None or post is None else
                 f'{post["population_rate_hz"] / pre["population_rate_hz"]:.3f}')
        print(f'{sh:>6}  {cell(pre):>12}  {cell(post):>12}  {ratio:>9}')
    print(f'\nSummary written to {summary_path}')
    for epoch_key in results:
        print(f'  {epoch_pkl_path(out_dir, epoch_key)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
