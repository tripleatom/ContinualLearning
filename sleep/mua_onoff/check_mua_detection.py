"""Validate MUA detection by drawing detected events on the traces the detector saw.

This rebuilds the *exact* preprocessing chain ``find_sleep_mua.py`` runs —
artifact repair (reused from the sorting run), CMR, 300-6000 Hz bandpass, then
per-channel median/MAD z-scoring — and reads short windows out of it. Traces are
therefore in the same noise-sigma units the threshold is expressed in, so a
detection at -4.0 can be read straight off the y axis.

Per shank it writes one trace figure per time window, plus one waveform figure:

``*_sh<N>_w<i>_traces.png``
    Every channel of the shank for one window, stacked in depth order, with the
    detected events marked.
``*_sh<N>_waveforms.png``
    Snippets around detected events on the busiest channels, individual events
    faint and the mean bold. Real spikes give a sharp biphasic mean; noise
    crossings give a flat smear.

Kept vs suppressed events
-------------------------
MountainSort5's detector (``mountainsort5.core.detect_spikes``) keeps a
threshold crossing only if it is the most extreme sample within
+/- ``time_radius`` on its own channel *and* on every channel within
``channel_radius``. On these probes the sites form one column 25 um apart, so
``DETECT_CHANNEL_RADIUS = 50`` um means each channel competes with itself
plus/minus two sites: one spike straddling several sites is counted once, on the
site where it is largest.

Both are drawn, so the rule can be checked rather than taken on trust:

    filled red    kept — what actually lands in the pkl
    open grey     suppressed — a crossing on this channel that lost to a larger
                  copy within 50 um

Set ``--marks all`` to see every per-channel crossing with nothing suppressed
(equivalent to ``detect_channel_radius=0.0``), or ``--marks saved`` for only
what was stored.

Numeric checks are printed alongside the figures, since a plot alone can hide a
systematic error:

1. every event is at or beyond threshold, with the right sign
2. every event is the most extreme sample within +/- ``detect_time_radius_msec``
   on its own channel (the peak-isolation rule)
3. no two events on one channel are closer than that radius (the refractory)
4. when events come from a saved pkl, the stored amplitude matches the trace
   value re-read here — an end-to-end check that this script reconstructed the
   same preprocessing the saved run used

Windows default to 30 ms and are centred on randomly chosen detected events, so
every figure resolves individual spike waveforms and is guaranteed to contain
something to check. Widen with ``--window-sec`` for context, but past ~50 ms a
1 ms spike falls below one pixel and there is nothing left to eyeball.

Usage
-----
    python check_mua_detection.py                        # every shank, presleep
    python check_mua_detection.py --shanks 0 1 --epoch post
    python check_mua_detection.py --n-windows 8 --window-sec 0.05
    python check_mua_detection.py --marks all            # nothing suppressed
    python check_mua_detection.py --window-mode spread   # unbiased sampling
    python check_mua_detection.py --events detect        # ignore the pkl
"""

import argparse
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import find_sleep_mua as fsm  # noqa: E402
from mua_detect import detect_mua_from_traces, preprocess_for_mua  # noqa: E402


#: Context read on each side of a window and then trimmed off. Matches
#: ``detect_mua``'s ``chunk_pad_sec``, which is what makes a windowed read
#: bit-identical to the full-epoch read detection actually used.
PAD_SEC = 0.1

#: Vertical gap between stacked channels, in noise sigma.
DEFAULT_SPACING = 15.0

#: Half-width of the waveform snippets, in milliseconds.
SNIPPET_MS = 1.5


# =====================================================
# STAMP
# =====================================================

def _stamp_figure(fig, info, y=0.004):
    """Embed a small reproducibility stamp (params/paths/timestamp) at the bottom."""
    fig.text(0.01, y, info, ha='left', va='bottom', fontsize=6, color='0.35')


def repro_info(ctx, extra=''):
    info = (
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"script={Path(__file__).name}\n"
        f"session={fsm.NWB_BASE} shank={ctx.shank} epoch={ctx.epoch_key} "
        f"samples=[{ctx.start}, {ctx.end}) fs={ctx.fs:.0f} Hz | "
        f"artifacts={ctx.artifact_source} events={ctx.event_source}\n"
        f"detect: threshold={fsm.DETECT_THRESHOLD} sign={fsm.DETECT_SIGN} "
        f"time_radius={fsm.DETECT_TIME_RADIUS_MSEC} ms "
        f"channel_radius={fsm.DETECT_CHANNEL_RADIUS} um scale={fsm.SCALE_MODE} "
        f"band={fsm.FREQ_MIN}-{fsm.FREQ_MAX} Hz\n"
        f"nwb={ctx.nwb_path}"
    )
    if extra:
        info += f"\n{extra}"
    return info


# =====================================================
# RECORDING / EVENTS
# =====================================================

def build_preprocessed(shank: int, epoch_key: str, args):
    """Rebuild the preprocessed epoch recording find_sleep_mua would detect on."""
    import spikeinterface.extractors as se

    nwb_path = fsm.SESSION_FOLDER / f'{fsm.NWB_BASE}sh{shank}.nwb'
    if not nwb_path.exists():
        raise FileNotFoundError(f'{nwb_path} not found — check SESSION_FOLDER/NWB_BASE')

    rec = se.read_nwb_recording(str(nwb_path))
    windows = fsm.epoch_windows(fsm.SESSION_FOLDER, rec.get_num_frames())
    if epoch_key not in windows:
        raise KeyError(f'{epoch_key} not among the epochs in conversion_list.txt: '
                       f'{sorted(windows)}')
    start, end = windows[epoch_key]

    out_dir = fsm.SESSION_FOLDER / fsm.OUTPUT_SUBFOLDER
    suffix = fsm.EPOCHS[epoch_key]
    ns = SimpleNamespace(verbose=args.verbose)

    if args.skip_artifacts:
        print('  artifacts: SKIPPED (--skip-artifacts) — traces differ slightly '
              'from what detection saw')
        rec_src, artifact_kwargs, artifact_source = rec, {
            'remove_artifacts': False, 'artifact_cache_folder': None}, 'skipped'
    else:
        rec_src, artifact_kwargs, artifact_source = fsm.resolve_artifacts(
            rec, shank, out_dir, suffix, ns)
        if artifact_source == 'epoch':
            print('  WARNING: no sorting cache for this shank, so rebuilding the '
                  'traces needs a full artifact detection pass (minutes).')
            print('           Use --skip-artifacts for a quick approximate look.')

    rec_epoch = rec_src.frame_slice(start_frame=start, end_frame=end)
    rec_pre = preprocess_for_mua(
        rec_epoch,
        remove_artifacts=artifact_kwargs['remove_artifacts'],
        artifact_cache_folder=artifact_kwargs['artifact_cache_folder'],
        scale_mode=fsm.SCALE_MODE,
        freq_min=fsm.FREQ_MIN,
        freq_max=fsm.FREQ_MAX,
        verbose=args.verbose,
    )
    return rec_pre, start, end, artifact_source, nwb_path


def load_saved_events(shank: int, epoch_key: str):
    """Per-channel event times from the saved epoch pkl, or None if unavailable."""
    path = fsm.epoch_pkl_path(fsm.SESSION_FOLDER / fsm.OUTPUT_SUBFOLDER, epoch_key)
    if not path.exists():
        return None, f'{path.name} not written yet'
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as e:
        # A run writing this file right now is the likely cause; fall back rather
        # than fail, since re-detecting the window gives the same answer.
        return None, f'could not read {path.name} ({e})'
    if shank not in data.get('shanks', {}):
        return None, f'shank {shank} not in {path.name} yet'
    return data['shanks'][shank], f'{path.name}'


def saved_events_in_window(saved, ch_ids, fs, s0, n_win):
    """Stored events falling in a window, as (sample_in_window, row, amplitude)."""
    idx_of = {cid: i for i, cid in enumerate(ch_ids)}
    times, rows, amps = [], [], []
    for cid, t_sec in saved['channel_spike_times'].items():
        if cid not in idx_of or len(t_sec) == 0:
            continue
        samp = np.rint(np.asarray(t_sec) * fs).astype(np.int64)
        sel = (samp >= s0) & (samp < s0 + n_win)
        if not sel.any():
            continue
        times.append(samp[sel] - s0)
        rows.append(np.full(int(sel.sum()), idx_of[cid], dtype=np.int32))
        amps.append(np.asarray(saved['channel_amplitudes'][cid])[sel])
    if not times:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int32),
                np.array([], dtype=np.float32))
    return np.concatenate(times), np.concatenate(rows), np.concatenate(amps)


def detect_events_in_window(traces, locs, fs, pad, n_win, channel_radius):
    """Fresh detection on a padded window, trimmed back to the window itself."""
    t, c, a = detect_mua_from_traces(
        traces,
        channel_locations=locs,
        sampling_frequency=fs,
        detect_threshold=fsm.DETECT_THRESHOLD,
        detect_sign=fsm.DETECT_SIGN,
        detect_time_radius_msec=fsm.DETECT_TIME_RADIUS_MSEC,
        detect_channel_radius=channel_radius,
    )
    t = np.asarray(t, dtype=np.int64) - pad
    keep = (t >= 0) & (t < n_win)
    return t[keep], np.asarray(c, dtype=np.int32)[keep], np.asarray(a)[keep]


def choose_windows(saved, ctx, args, n_win, pad, n_windows, mode):
    """Window start samples, either centred on real events or spread evenly.

    Centring on events guarantees every figure has something to check, which
    matters once the windows are short enough to resolve a spike. It needs event
    times up front, so it is only available from a saved pkl — a fresh detection
    would have to scan the whole epoch first, and falls back to even spread.
    """
    total = ctx.end - ctx.start
    lo, hi = pad, total - pad - n_win
    if hi <= lo:
        raise ValueError(f'epoch is too short for a {n_win}-sample window')

    if mode == 'events' and saved is not None:
        pooled = [np.rint(np.asarray(t) * ctx.fs).astype(np.int64)
                  for t in saved['channel_spike_times'].values() if len(t)]
        if pooled:
            all_t = np.concatenate(pooled)
            rng = np.random.default_rng(args.seed)
            picks = rng.choice(all_t, size=min(n_windows, len(all_t)), replace=False)
            return np.clip(np.sort(picks) - n_win // 2, lo, hi).astype(np.int64)
        print('  pkl holds no events — falling back to evenly spread windows')

    return (lo + np.linspace(0, hi - lo, n_windows + 2)[1:-1]).astype(np.int64)


def split_kept_suppressed(kept_t, kept_c, all_t, all_c, n_ch):
    """Events in the radius-0 superset that the channel radius threw away.

    A larger ``channel_radius`` only ever adds neighbours to compare against, so
    the kept set is always a subset of the radius-0 set and a plain difference
    is well defined.
    """
    if len(all_t) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
    kept_key = set((kept_t.astype(np.int64) * n_ch + kept_c).tolist())
    all_key = all_t.astype(np.int64) * n_ch + all_c
    mask = np.array([k not in kept_key for k in all_key.tolist()], dtype=bool)
    return all_t[mask], all_c[mask]


# =====================================================
# NUMERIC CHECKS
# =====================================================

def check_window(traces, times, rows, amps, fs, from_pkl):
    """Verify the detection rules hold for one window's events."""
    r = int(np.ceil(fsm.DETECT_TIME_RADIUS_MSEC / 1000 * fs))
    out = {'n': len(times), 'below_threshold': 0, 'not_local_peak': 0,
           'refractory_violations': 0, 'max_amp_mismatch': 0.0,
           'min_isi_samples': np.inf}
    if len(times) == 0:
        return out

    vals = traces[times, rows]

    # 1. sign and magnitude
    if fsm.DETECT_SIGN < 0:
        out['below_threshold'] = int(np.sum(vals > -fsm.DETECT_THRESHOLD))
    elif fsm.DETECT_SIGN > 0:
        out['below_threshold'] = int(np.sum(vals < fsm.DETECT_THRESHOLD))
    else:
        out['below_threshold'] = int(np.sum(np.abs(vals) < fsm.DETECT_THRESHOLD))

    # 2. peak isolation on its own channel
    n = traces.shape[0]
    bad_peak = 0
    for t, c, v in zip(times, rows, vals):
        lo, hi = max(0, t - r), min(n, t + r + 1)
        seg = traces[lo:hi, c]
        if fsm.DETECT_SIGN < 0:
            if v > seg.min():
                bad_peak += 1
        elif fsm.DETECT_SIGN > 0:
            if v < seg.max():
                bad_peak += 1
        elif np.abs(v) < np.abs(seg).max():
            bad_peak += 1
    out['not_local_peak'] = bad_peak

    # 3. refractory, per channel
    viol, min_isi = 0, np.inf
    for c in np.unique(rows):
        t_c = np.sort(times[rows == c])
        if len(t_c) < 2:
            continue
        d = np.diff(t_c)
        min_isi = min(min_isi, int(d.min()))
        viol += int(np.sum(d < r))
    out['refractory_violations'] = viol
    out['min_isi_samples'] = min_isi

    # 4. stored amplitude vs the trace value re-read here
    if from_pkl and len(amps):
        out['max_amp_mismatch'] = float(np.abs(np.asarray(amps) - vals).max())

    return out


def aggregate(stats):
    agg = {k: sum(s[k] for s in stats) for k in
           ('n', 'below_threshold', 'not_local_peak', 'refractory_violations')}
    agg['max_amp_mismatch'] = max((s['max_amp_mismatch'] for s in stats), default=0.0)
    agg['min_isi_samples'] = min((s['min_isi_samples'] for s in stats), default=np.inf)
    return agg


def summary_line(agg, n_kept, n_suppressed):
    return (f"checks over {n_kept} kept events ({n_suppressed} suppressed by the "
            f"{fsm.DETECT_CHANNEL_RADIUS} um radius): "
            f"{agg['below_threshold']} under threshold, "
            f"{agg['not_local_peak']} not a local peak, "
            f"{agg['refractory_violations']} refractory violations, "
            f"max |stored-retraced| amplitude = {agg['max_amp_mismatch']:.2e}")


def print_checks(agg, fs, from_pkl, label=''):
    r = int(np.ceil(fsm.DETECT_TIME_RADIUS_MSEC / 1000 * fs))
    print(f'\n  CHECKS over {agg["n"]} events{label}')

    def line(text, bad, detail=''):
        print(f'    [{"PASS" if bad == 0 else "FAIL"}] {text:<44s} {bad:>5d} bad {detail}')

    line(f'beyond {fsm.DETECT_THRESHOLD} sigma, sign {fsm.DETECT_SIGN}',
         agg['below_threshold'])
    line(f'most extreme within +/-{r} samples on channel', agg['not_local_peak'])
    line(f'refractory >= {r} samples on a channel', agg['refractory_violations'],
         '' if agg['min_isi_samples'] == np.inf
         else f'(min ISI {int(agg["min_isi_samples"])} samples)')
    if from_pkl:
        ok = agg['max_amp_mismatch'] < 1e-3
        print(f'    [{"PASS" if ok else "FAIL"}] stored amplitude matches retraced '
              f'value{"":<7s} max diff {agg["max_amp_mismatch"]:.3e}')
        if not ok:
            print('           -> the saved events came from a different '
                  'preprocessing than this script rebuilt.')
            print("              Check the shank's artifact_source and params "
                  'against the current')
            print('              REUSE_SORTING_ARTIFACTS / ARTIFACT_PARAM_DEFAULTS, '
                  'then re-run with --overwrite.')
            print('              The first three checks stay valid — they are '
                  'geometric, not absolute.')


# =====================================================
# PLOTS
# =====================================================

def plot_window(rec_pre, saved, ctx, args, s0, index, out_path):
    """One window, every channel of the shank, stacked in depth order."""
    fs = ctx.fs
    pad = int(PAD_SEC * fs)
    n_win = int(args.window_sec * fs)
    ch_ids = list(rec_pre.get_channel_ids())
    locs = np.asarray(rec_pre.get_channel_locations(), dtype=np.float32)
    n_ch = len(ch_ids)

    has_y = locs.ndim == 2 and locs.shape[1] > 1
    depth_order = np.argsort(locs[:, 1]) if has_y else np.arange(n_ch)
    if args.channels:
        wanted = {ch_ids.index(c) if c in ch_ids else int(c) for c in args.channels}
        show_rows = [i for i in depth_order if i in wanted]
    else:
        show_rows = list(depth_order)

    traces = np.asarray(
        rec_pre.get_traces(start_frame=s0 - pad, end_frame=s0 + n_win + pad),
        dtype=np.float32)
    win = traces[pad:pad + n_win]
    t_ms = np.arange(n_win) / fs * 1000.0

    # Kept events: what the pkl holds, or a faithful re-detection.
    if saved is not None:
        kept_t, kept_c, kept_a = saved_events_in_window(saved, ch_ids, fs, s0, n_win)
    else:
        kept_t, kept_c, kept_a = detect_events_in_window(
            traces, locs, fs, pad, n_win, fsm.DETECT_CHANNEL_RADIUS)

    # Superset with no cross-channel suppression at all.
    sup_t = sup_c = np.array([], dtype=np.int64)
    if args.marks in ('both', 'all'):
        all_t, all_c, _ = detect_events_in_window(traces, locs, fs, pad, n_win, 0.0)
        sup_t, sup_c = split_kept_suppressed(kept_t, kept_c, all_t, all_c, n_ch)

    stats = check_window(win, kept_t, kept_c, kept_a, fs, saved is not None)

    fig, ax = plt.subplots(figsize=(16, max(6.0, 0.42 * len(show_rows) + 2.2)))
    offsets = -np.arange(len(show_rows)) * args.spacing

    for k, ch in enumerate(show_rows):
        ax.plot(t_ms, win[:, ch] + offsets[k], lw=0.5, color='0.25')
        ax.axhline(offsets[k] - fsm.DETECT_THRESHOLD, lw=0.5, ls='--',
                   color='tab:blue', alpha=0.22)
        if args.marks in ('both', 'all') and len(sup_t):
            sel = sup_c == ch
            if sel.any():
                ax.plot(sup_t[sel] / fs * 1000.0, win[sup_t[sel], ch] + offsets[k],
                        'o', ms=5.0, mfc='none', mec='0.45', mew=0.9, alpha=0.85)
        if args.marks in ('both', 'saved') and len(kept_t):
            sel = kept_c == ch
            if sel.any():
                ax.plot(kept_t[sel] / fs * 1000.0, win[kept_t[sel], ch] + offsets[k],
                        'o', ms=3.4, color='tab:red', mec='none', alpha=0.95)

    ax.set_yticks(offsets)
    ax.set_yticklabels(
        [f'{ch_ids[i]}  ({locs[i, 1]:.0f})' if has_y else str(ch_ids[i])
         for i in show_rows], fontsize=6)
    ax.set_ylabel('channel id  (depth, um) — shallow at top', fontsize=9)
    ax.set_xlabel('time within window (ms)', fontsize=9)
    ax.set_xlim(0, args.window_sec * 1000.0)
    ax.tick_params(labelsize=7)

    marks_txt = {
        'both': f'filled red = kept, open grey = suppressed by the '
                f'{fsm.DETECT_CHANNEL_RADIUS} um channel radius',
        'saved': 'filled red = kept (stored in the pkl)',
        'all': f'open grey = every per-channel crossing (channel_radius=0), '
               f'filled red = kept at {fsm.DETECT_CHANNEL_RADIUS} um',
    }[args.marks]
    ax.set_title(
        f'MUA detection check — {fsm.NWB_BASE} shank {ctx.shank}, {ctx.epoch_key}, '
        f'window {index + 1}\n'
        f't = {(ctx.start + s0) / fs:.2f} s in session '
        f'({s0 / fs:.2f} s into {ctx.epoch_key})  —  '
        f'{len(kept_t)} kept, {len(sup_t)} suppressed, {len(show_rows)} channels\n'
        f'{marks_txt}; dashed = {fsm.DETECT_THRESHOLD} sigma threshold',
        fontsize=10)

    _stamp_figure(fig, repro_info(
        ctx, extra=summary_line(aggregate([stats]), len(kept_t), len(sup_t))))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return stats, len(kept_t), len(sup_t)


def plot_waveforms(rec_pre, saved, ctx, args, out_path):
    """Snippets around detected events on the busiest channels.

    Uses its own, longer windows rather than the trace figures': those are cut
    short enough to resolve a single spike, which would pool far too few events
    for a stable mean. These are spread evenly rather than centred on events, so
    the average is not biased toward whatever the seed happened to pick.
    """
    fs = ctx.fs
    pad = int(PAD_SEC * fs)
    n_win = int(args.waveform_window_sec * fs)
    half = int(SNIPPET_MS / 1000 * fs)
    ch_ids = list(rec_pre.get_channel_ids())
    locs = np.asarray(rec_pre.get_channel_locations(), dtype=np.float32)

    starts = choose_windows(saved, ctx, args, n_win, pad,
                            args.waveform_windows, 'spread')

    snippets = {}
    for s0 in starts:
        traces = np.asarray(
            rec_pre.get_traces(start_frame=s0 - pad, end_frame=s0 + n_win + pad),
            dtype=np.float32)
        win = traces[pad:pad + n_win]
        if saved is not None:
            t, c, _ = saved_events_in_window(saved, ch_ids, fs, s0, n_win)
        else:
            t, c, _ = detect_events_in_window(
                traces, locs, fs, pad, n_win, fsm.DETECT_CHANNEL_RADIUS)
        for ti, ci in zip(t, c):
            if ti - half < 0 or ti + half + 1 > win.shape[0]:
                continue
            snippets.setdefault(int(ci), []).append(win[ti - half:ti + half + 1, ci])

    if not snippets:
        print('    no events available for the waveform figure — skipped')
        return

    busiest = sorted(snippets, key=lambda c: len(snippets[c]), reverse=True)[:6]
    n_col = min(3, len(busiest))
    n_row = int(np.ceil(len(busiest) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 3.1 * n_row + 1.0),
                             squeeze=False)
    t_ms = (np.arange(2 * half + 1) - half) / fs * 1000.0
    rng = np.random.default_rng(args.seed)

    for ax, c in zip(axes.ravel(), busiest):
        arr = np.asarray(snippets[c])
        show = arr if len(arr) <= args.max_snippets else \
            arr[rng.choice(len(arr), args.max_snippets, replace=False)]
        ax.plot(t_ms, show.T, lw=0.3, color='0.6', alpha=0.25)
        ax.plot(t_ms, arr.mean(0), lw=1.8, color='tab:red')
        ax.axhline(-fsm.DETECT_THRESHOLD, lw=0.7, ls='--', color='tab:blue', alpha=0.6)
        ax.axvline(0, lw=0.5, color='0.4', alpha=0.5)
        ax.set_title(f'channel {ch_ids[c]} — {len(arr)} events', fontsize=9)
        ax.set_xlabel('ms from event', fontsize=8)
        ax.set_ylabel('sigma', fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(busiest):]:
        ax.axis('off')

    fig.suptitle(
        f'MUA waveform check — {fsm.NWB_BASE} shank {ctx.shank}, {ctx.epoch_key}\n'
        f'grey = individual events (up to {args.max_snippets}), red = mean',
        fontsize=10)
    _stamp_figure(fig, repro_info(
        ctx, extra=f'snippet half-width={SNIPPET_MS} ms, '
                   f'{sum(len(v) for v in snippets.values())} events pooled over '
                   f'{args.waveform_windows} x {args.waveform_window_sec} s '
                   f'evenly spread windows'))
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# =====================================================
# PER-SHANK DRIVER
# =====================================================

def run_shank(shank: int, epoch_key: str, out_dir: Path, args):
    """Every figure for one shank. Returns a row for the cross-shank summary."""
    print(f'\n--- shank {shank} ---')
    rec_pre, start, end, artifact_source, nwb_path = build_preprocessed(
        shank, epoch_key, args)
    fs = rec_pre.get_sampling_frequency()

    saved, note = (None, 'forced fresh detection') if args.events == 'detect' \
        else load_saved_events(shank, epoch_key)
    if args.events == 'pkl' and saved is None:
        raise SystemExit(f'--events pkl requested but unavailable: {note}')
    event_source = 'fresh detection' if saved is None else note
    print(f'  events: {"detecting per window (" + note + ")" if saved is None else "from " + note}')

    ctx = SimpleNamespace(shank=shank, epoch_key=epoch_key, start=start, end=end,
                          fs=fs, artifact_source=artifact_source,
                          event_source=event_source, nwb_path=nwb_path)

    pad = int(PAD_SEC * fs)
    n_win = int(args.window_sec * fs)
    starts = choose_windows(saved, ctx, args, n_win, pad,
                            args.n_windows, args.window_mode)
    mode = 'centred on events' if (args.window_mode == 'events' and saved is not None) \
        else 'spread over the epoch'
    print(f'  {len(starts)} x {args.window_sec * 1000:.0f} ms windows, {mode}')

    base = f'{fsm.NWB_BASE}_{fsm.EPOCHS[epoch_key]}_sh{shank}'
    stats, n_kept, n_sup = [], 0, 0
    for i, s0 in enumerate(starts):
        png = out_dir / f'{base}_w{i + 1}_traces.png'
        st, nk, ns_ = plot_window(rec_pre, saved, ctx, args, s0, i, png)
        stats.append(st)
        n_kept += nk
        n_sup += ns_
        print(f'    w{i + 1}: {nk} kept, {ns_} suppressed -> {png.name}')

    wave_png = out_dir / f'{base}_waveforms.png'
    plot_waveforms(rec_pre, saved, ctx, args, wave_png)
    print(f'    waveforms -> {wave_png.name}')

    agg = aggregate(stats)
    print_checks(agg, fs, saved is not None)
    return {'shank': shank, 'agg': agg, 'kept': n_kept, 'suppressed': n_sup,
            'from_pkl': saved is not None, 'artifact_source': artifact_source}


# =====================================================
# MAIN
# =====================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shanks', type=int, nargs='+', default=None,
                    help=f'shanks to check (default {fsm.SHANKS})')
    ap.add_argument('--epoch', default='pre',
                    choices=sorted(fsm.EPOCHS) + list(fsm.EPOCHS.values()))
    ap.add_argument('--n-windows', type=int, default=5,
                    help='time windows per shank, one figure each (default 5)')
    ap.add_argument('--window-sec', type=float, default=0.03,
                    help='length of each window in seconds (default 0.03). Short '
                         'on purpose: a spike is ~1 ms, so anything past ~50 ms '
                         'squeezes the waveform below one pixel')
    ap.add_argument('--window-mode', choices=['events', 'spread'], default='events',
                    help="'events' centres each window on a randomly chosen "
                         "detected event (needs the pkl), 'spread' spaces them "
                         'evenly over the epoch (default events)')
    ap.add_argument('--marks', choices=['both', 'saved', 'all'], default='both',
                    help="'both' draws kept events filled red and radius-suppressed "
                         "ones open grey, 'saved' only the kept ones, 'all' "
                         'emphasises every per-channel crossing (default both)')
    ap.add_argument('--waveform-windows', type=int, default=5,
                    help='windows pooled for the waveform figure (default 5)')
    ap.add_argument('--waveform-window-sec', type=float, default=2.0,
                    help='length of each waveform window in seconds (default 2.0)')
    ap.add_argument('--channels', nargs='+', default=None,
                    help='restrict to these channel ids (default: all on the shank)')
    ap.add_argument('--spacing', type=float, default=DEFAULT_SPACING,
                    help=f'vertical gap between channels, in sigma (default {DEFAULT_SPACING})')
    ap.add_argument('--events', choices=['auto', 'pkl', 'detect'], default='auto',
                    help='where events come from: the saved pkl, a fresh detection '
                         'on the window, or auto (pkl when present)')
    ap.add_argument('--max-snippets', type=int, default=150,
                    help='individual waveforms drawn per channel (default 150)')
    ap.add_argument('--skip-artifacts', action='store_true',
                    help='skip artifact repair — fast, but the traces then differ '
                         'slightly from what detection saw')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None,
                    help='output folder (default <session>/MUA/validation)')
    ap.add_argument('--quiet', dest='verbose', action='store_false')
    args = ap.parse_args()

    shanks = args.shanks if args.shanks is not None else fsm.SHANKS
    alias = {v: k for k, v in fsm.EPOCHS.items()}
    epoch_key = alias.get(args.epoch, args.epoch)

    out_dir = Path(args.out) if args.out else \
        fsm.SESSION_FOLDER / fsm.OUTPUT_SUBFOLDER / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'MUA detection check — {fsm.NWB_BASE}, {epoch_key}')
    print(f'  shanks : {shanks}')
    print(f'  output : {out_dir}')
    print(f'  figures: {args.n_windows} trace + 1 waveform per shank')
    print('=' * 70)

    rows = []
    for shank in shanks:
        try:
            rows.append(run_shank(shank, epoch_key, out_dir, args))
        except Exception as e:
            print(f'  FAILED on shank {shank}: {e}')
            traceback.print_exc()

    if not rows:
        print('\nNothing checked.')
        return

    print('\n' + '=' * 70)
    print(f'SUMMARY — {epoch_key}')
    print(f'{"shank":>5} {"kept":>7} {"suppr":>7} {"suppr%":>7} {"bad":>5} '
          f'{"max amp diff":>13}  result')
    for r in rows:
        a = r['agg']
        bad = a['below_threshold'] + a['not_local_peak'] + a['refractory_violations']
        amp_bad = r['from_pkl'] and a['max_amp_mismatch'] >= 1e-3
        pct = 100.0 * r['suppressed'] / max(1, r['kept'] + r['suppressed'])
        print(f'{r["shank"]:>5} {r["kept"]:>7} {r["suppressed"]:>7} {pct:>6.1f}% '
              f'{bad:>5} {a["max_amp_mismatch"]:>13.3e}  '
              f'{"PASS" if bad == 0 and not amp_bad else "FAIL"}')
    print('=' * 70)


if __name__ == '__main__':
    main()
