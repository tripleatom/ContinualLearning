"""Per-contact NREM spindle rate across every shank of one registered day.

detect_v1_spindles.py scores one representative contact per shank; this runs
the same 9-17 Hz envelope detector on every screened contact, so rate can be
compared across depth. Each contact is thresholded twice:
  own    - against its own NREM envelope mean/SD (burstiness of that contact)
  shared - against the shank's median contact mean/SD (tracks amplitude)
Adjacent contacts see largely the same field potential; events are counted per
contact and are not unique spindles.

    python sleep/spindles/spindles_per_channel.py --animal CnL46 --date 260727
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import argparse
import csv
import json
import pickle
from datetime import datetime

import numpy as np
from scipy.signal import butter, sosfiltfilt

from detect_v1_spindles import detect_events, nrem_envelope, nrem_samples
from server_fallback import resolve_existing_file, resolve_output_folder

BAND = (9., 17.)
MODES = ('own', 'shared')

# Approximate layer 5 depth (um, probe y; y=0 is near the brain surface), as
# estimated by the experimenter - not histology.
LAYER5_UM = {'CnL46': {0: 450, 1: 450, 2: 450, 3: 450, 4: 300, 5: 300, 6: 300, 7: 300}}
LAYER5_HALF_WIDTH_UM = 50

# Contacts verified bad by hand (original channel IDs), added to the screen.
KNOWN_BAD = {'CnL46': {4: (0, 1, 4, 5, 8, 12, 14, 18, 27, 29, 30, 31),
                       5: (0, 1, 4, 5, 8, 12, 14, 18, 27, 29, 30, 31)}}


def load_states(low_freq, label):
    """Canonical four-state file if present, else the legacy delta+velocity mask."""
    for path in (low_freq / 'sleep_segmentation' / f'{label}_sleep_periods.pkl',
                 low_freq / f'{label}_sleep_periods.pkl'):
        try:
            path = Path(resolve_existing_file(path))
        except FileNotFoundError:
            continue
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            states = pickle.load(stream)
        if 'states' not in states:
            # score_nrem_delta_velocity.py output: NREM mask only. Epochs with no
            # camera coverage are already non-NREM there; it has no artifact mask.
            nrem = np.asarray(states['nrem_epoch_mask'], bool)
            states = dict(states, states=np.where(nrem, 'NREM', 'OTHER'),
                          valid_mask=np.ones(len(nrem), bool),
                          artifact_mask=np.zeros(len(nrem), bool))
        return states, path
    raise FileNotFoundError(f'No sleep_periods.pkl for {label}')


def screen_channels(traces, nrem, ids, known_bad, *, rms_ratio, corr_diff, step):
    """Flag contacts that break from their depth neighbours (columns are depth-ordered).

    Broadband NREM RMS and correlation with the shank median both vary smoothly
    with depth in median-referenced LFP (the correlation changes sign), so
    shank-wide limits reject healthy contacts. Each contact is compared with its
    nearest two accepted contacts on either side; the worst outlier is removed
    and the comparison repeated, so a bad contact cannot condemn its neighbours.
    """
    x = traces[nrem][::step].astype(np.float32)
    finite = np.isfinite(x).all(0)
    x[:, ~finite] = 0
    std = x.std(0)
    usable = finite & (std > 0)
    median_trace = np.median(x[:, usable], axis=1) if usable.any() else np.zeros(len(x))
    corr = np.full(x.shape[1], np.nan)
    for ch in np.flatnonzero(usable):
        corr[ch] = np.corrcoef(x[:, ch], median_trace)[0, 1]
    reasons = [[] if usable[ch] else ['flat/nonfinite'] for ch in range(x.shape[1])]
    for ch in range(x.shape[1]):
        if int(ids[ch]) in known_bad:
            reasons[ch].append('known_bad')
    while True:
        good = np.flatnonzero([not r for r in reasons])
        worst, worst_score, worst_why = None, 1., None
        for ch in good:
            near = np.r_[good[good < ch][-2:], good[good > ch][:2]]
            if len(near) < 2:
                continue
            ratio = std[ch] / np.median(std[near])
            shift = abs(corr[ch] - np.median(corr[near]))
            score = max(abs(np.log(ratio)) / np.log(rms_ratio), shift / corr_diff)
            if score > worst_score:
                why = ([f'rms_x{ratio:.2f}_of_neighbours'] if abs(np.log(ratio)) > np.log(rms_ratio) else []) \
                    + ([f'corr_shift_{shift:.2f}'] if shift > corr_diff else [])
                worst, worst_score, worst_why = ch, score, why
        if worst is None:
            break
        reasons[worst] += worst_why
    return std, corr, ['+'.join(r) for r in reasons]


def summarize(events, clean_min):
    if not events:
        return dict(n=0, rate=0., duration=np.nan, amplitude=np.nan)
    return dict(n=len(events), rate=len(events) / clean_min,
                duration=float(np.mean([e['duration_s'] for e in events])),
                amplitude=float(np.mean([np.sqrt(e['peak_power']) for e in events])))


def process_shank(lfp_path, states, args, animal, rng):
    with np.load(lfp_path, allow_pickle=False) as data:
        traces = data['traces']
        fs = float(data['sampling_rate'])
        ids = data['channel_ids']
        depths = data['ycoord'].astype(float)
        shank = int(data['shank'])
        offset = float(data['sleep_start_sample']) / float(data['original_fs'])
        if int(states.get('sleep_start_sample') or 0) != int(data['sleep_start_sample']):
            raise ValueError('State/LFP sleep-start mismatch')
    nrem = nrem_samples(states, len(traces), fs)
    std, corr, reasons = screen_channels(
        traces, nrem, ids, KNOWN_BAD.get(animal, {}).get(shank, ()),
        rms_ratio=args.neighbour_rms_ratio, corr_diff=args.neighbour_corr_diff,
        step=max(1, int(fs / 100)))
    good = np.array([not r for r in reasons])
    rows = [dict(shank=shank, channel_id=int(ids[ch]), depth_um=depths[ch], fs=fs,
                 good=bool(good[ch]), bad_reason=reasons[ch], broadband_std=float(std[ch]),
                 corr_with_shank_median=float(corr[ch])) for ch in range(len(ids))]
    if args.screen_only or not good.any():
        return rows, [], None
    amplitude, valid = nrem_envelope(traces, fs, nrem, np.flatnonzero(good), band=BAND)
    clean_min = valid.sum() / fs / 60
    if clean_min * 60 < 10:
        raise ValueError('Insufficient NREM after filter-edge guards')
    options = dict(boundary=args.boundary_threshold, peak=args.peak_threshold,
                   min_duration=args.min_duration, max_duration=args.max_duration,
                   merge_gap=args.merge_gap)
    means = np.array([amplitude[valid, ch].mean(dtype=float) if good[ch] else np.nan
                      for ch in range(len(ids))])
    sds = np.array([amplitude[valid, ch].std(dtype=float) if good[ch] else np.nan
                    for ch in range(len(ids))])
    shared = dict(center=float(np.nanmedian(means)), scale=float(np.nanmedian(sds)))
    events_out, by_channel = [], {}
    for ch in np.flatnonzero(good):
        rows[ch].update(power_9_17=float(np.mean(amplitude[valid, ch].astype(float) ** 2)),
                        clean_nrem_min=clean_min)
        for mode in MODES:
            events, _ = detect_events(amplitude[:, ch], valid, fs, **options,
                                      **(shared if mode == 'shared' else {}))
            for key, value in summarize(events, clean_min).items():
                rows[ch][f'{key}_{mode}'] = value
            for event in events:
                events_out.append(dict(event, threshold=mode, shank=shank,
                                       channel_id=int(ids[ch]), depth_um=depths[ch],
                                       peak_recording_s=event['peak_s'] + offset))
            if mode == 'own':
                by_channel[ch] = events
    # Broadband snippets around events of the highest-rate contact, plus the
    # lowest-rate contact at the same times, for visual review.
    order = sorted(by_channel, key=lambda ch: rows[ch]['rate_own'])
    low, high = order[0], order[-1]
    half = int(1.5 * fs)
    peaks = [int(round(e['peak_s'] * fs)) for e in by_channel[high]]
    peaks = [p for p in peaks if p - half >= 0 and p + half < len(traces)
             and valid[p - half:p + half].all()]
    picked = sorted(rng.choice(peaks, size=min(args.n_examples, len(peaks)), replace=False)) if peaks else []
    sos = butter(4, BAND, fs=fs, btype='bandpass', output='sos')
    raw = np.array([[traces[p - half:p + half, ch].astype(float) for p in picked]
                    for ch in (high, low)])
    examples = dict(shank=shank, fs=fs, raw=raw,
                    band=sosfiltfilt(sos, raw, axis=-1) if len(picked) else raw,
                    peak_s=np.array(picked) / fs,
                    channels=[int(ids[high]), int(ids[low])],
                    depths=[depths[high], depths[low]],
                    rates=[rows[high]['rate_own'], rows[low]['rate_own']])
    return rows, events_out, examples


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stamp(fig, args, extra):
    fig.get_layout_engine().set(rect=(0, .05, 1, .95))  # keep the stamp clear of axis labels
    fig.text(.005, .003, f'{datetime.now():%Y-%m-%d %H:%M} | sleep/spindles/spindles_per_channel.py --animal {args.animal} '
             f'--date {args.date} | band {BAND[0]:g}-{BAND[1]:g} Hz, 100 ms envelope, 1 s edge guard | '
             f'boundary {args.boundary_threshold} SD, peak {args.peak_threshold} SD, duration '
             f'{args.min_duration}-{args.max_duration} s, merge {args.merge_gap} s | contact screen vs depth '
             f'neighbours: RMS x{args.neighbour_rms_ratio}, corr shift {args.neighbour_corr_diff} | {extra}',
             fontsize=6, va='bottom', wrap=True)


def grid(rows, session, key, shanks, depths):
    out = np.full((len(depths), len(shanks)), np.nan)
    for row in rows:
        if row['session'] == session and row['good'] and key in row:
            out[depths.index(row['depth_um']), shanks.index(row['shank'])] = row[key]
    return out


def plot_all(rows, examples, sessions, args, out, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shanks = sorted({row['shank'] for row in rows})
    depths = sorted({row['depth_um'] for row in rows})
    layer5 = LAYER5_UM.get(args.animal, {})
    panels = [('rate_own', 'Rate, own threshold (events/min)'),
              ('rate_shared', 'Rate, shared threshold (events/min)'),
              ('power_9_17', '9-17 Hz power (input units²)')]
    pitch = np.median(np.diff(depths)) if len(depths) > 1 else 25.

    fig, axes = plt.subplots(len(sessions), 3, figsize=(15, 5.2 * len(sessions)),
                             squeeze=False, layout='constrained')
    for r, session in enumerate(sessions):
        for c, (key, title) in enumerate(panels):
            ax = axes[r, c]
            values = grid(rows, session, key, shanks, depths)
            vmax = max(np.nanmax(grid(rows, s, key, shanks, depths)) for s in sessions)
            mesh = ax.imshow(values, aspect='auto', cmap='viridis', vmin=0, vmax=vmax,
                             extent=(-.5, len(shanks) - .5, depths[-1] + pitch / 2, depths[0] - pitch / 2))
            ax.set_facecolor('lightgray')  # excluded contacts
            for i, shank in enumerate(shanks):
                if shank in layer5:
                    for edge in (-LAYER5_HALF_WIDTH_UM, LAYER5_HALF_WIDTH_UM):
                        ax.plot([i - .5, i + .5], [layer5[shank] + edge] * 2, 'w--', lw=1)
            ax.set(xticks=range(len(shanks)), xticklabels=shanks, xlabel='Shank',
                   ylabel='Probe y (µm; 0 near surface)', title=f'{session}: {title}')
            fig.colorbar(mesh, ax=ax)
    fig.suptitle(f'{info["session_name"]} per-contact NREM spindles '
                 f'(gray = excluded contact; white dashes = approx. layer 5 ± {LAYER5_HALF_WIDTH_UM} µm)')
    stamp(fig, args, info['stamp'])
    fig.savefig(out / f'{info["session_name"]}_per_channel_spindle_heatmap.png', dpi=160)
    plt.close(fig)

    colors = dict(pre='tab:blue', post='tab:red')
    fig, axes = plt.subplots(2, len(shanks), figsize=(2.6 * len(shanks), 10),
                             sharey=True, squeeze=False, layout='constrained')
    for i, shank in enumerate(shanks):
        for session in sessions:
            sel = sorted((row for row in rows if row['session'] == session and row['shank'] == shank
                          and row['good'] and 'rate_own' in row), key=lambda row: row['depth_um'])
            y = [row['depth_um'] for row in sel]
            color = colors.get(session)
            axes[0, i].plot([row['rate_own'] for row in sel], y, '.-', c=color, label=f'{session} own')
            axes[0, i].plot([row['rate_shared'] for row in sel], y, '.--', c=color, alpha=.6,
                            label=f'{session} shared')
            axes[1, i].plot([row['power_9_17'] for row in sel], y, '.-', c=color, label=session)
        for ax in axes[:, i]:
            if shank in layer5:
                ax.axhspan(layer5[shank] - LAYER5_HALF_WIDTH_UM, layer5[shank] + LAYER5_HALF_WIDTH_UM,
                           color='gold', alpha=.3)
            ax.set_ylim(depths[-1] + pitch, depths[0] - pitch)
        axes[0, i].set(title=f'Shank {shank}', xlabel='Events/min')
        axes[1, i].set(xlabel='9-17 Hz power')
    axes[0, 0].set_ylabel('Probe y (µm; 0 near surface)')
    axes[1, 0].set_ylabel('Probe y (µm; 0 near surface)')
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f'{info["session_name"]} spindle rate and power by depth '
                 f'(gold = approx. layer 5 ± {LAYER5_HALF_WIDTH_UM} µm; excluded contacts omitted)')
    stamp(fig, args, info['stamp'])
    fig.savefig(out / f'{info["session_name"]}_per_channel_spindle_depth_profiles.png', dpi=160)
    plt.close(fig)

    for session in sessions:
        found = [e for e in examples.get(session, []) if e is not None and len(e['peak_s'])]
        if not found:
            continue
        ex = max(found, key=lambda e: e['rates'][0])
        n = len(ex['peak_s'])
        t = (np.arange(ex['raw'].shape[-1]) - ex['raw'].shape[-1] // 2) / ex['fs']
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), sharex=True, sharey=True,
                                 squeeze=False, layout='constrained')
        for row in range(2):
            for k in range(n):
                axes[row, k].plot(t, ex['raw'][row, k], c='0.6', lw=.6, label='saved LFP')
                axes[row, k].plot(t, ex['band'][row, k], c='tab:red', lw=.8, label='9-17 Hz')
                axes[row, k].set_title(f't = {ex["peak_s"][k]:.1f} s', fontsize=8)
            axes[row, 0].set_ylabel(f'{"highest" if row == 0 else "lowest"}-rate contact\n'
                                    f'ch {ex["channels"][row]}, y {ex["depths"][row]:g} µm, '
                                    f'{ex["rates"][row]:.2f}/min')
        for ax in axes[1]:
            ax.set_xlabel('Time from peak on top contact (s)')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{info["session_name"]} {session}, shank {ex["shank"]}: random events of the '
                     f'highest-rate contact (own threshold), same times on the lowest-rate contact')
        stamp(fig, args, info['stamp'] + f' | example seed {args.seed}')
        fig.savefig(out / f'{info["session_name"]}_{session}_per_channel_spindle_examples.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--animal', required=True)
    parser.add_argument('--date', required=True, help='Registry date, e.g. 260727')
    parser.add_argument('--sessions', nargs='+', default=['pre', 'post'])
    parser.add_argument('--shanks', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--peak-threshold', type=float, default=3.)
    parser.add_argument('--boundary-threshold', type=float, default=1.5)
    parser.add_argument('--min-duration', type=float, default=.3)
    parser.add_argument('--max-duration', type=float, default=3.)
    parser.add_argument('--merge-gap', type=float, default=.2)
    parser.add_argument('--neighbour-rms-ratio', type=float, default=1.4,
                        help='Exclude contacts whose NREM RMS differs from their depth neighbours by this factor')
    parser.add_argument('--neighbour-corr-diff', type=float, default=.1,
                        help='Exclude contacts whose correlation with the shank median departs this far from neighbours')
    parser.add_argument('--n-examples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--screen-only', action='store_true', help='Only write the contact screen table')
    args = parser.parse_args()

    registry = json.loads((sleep_paths.SLEEP_DIR / 'sleep_day_configs.json').read_text())
    folder = Path(registry[f'{args.animal}_{args.date}']['rec_folder'])
    low_freq = folder / 'low_freq'
    out = args.out_dir or Path(resolve_output_folder(low_freq / 'v1_spindles' / 'per_channel'))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, events, examples, sources = [], [], {}, []
    for session in args.sessions:
        label = f'{folder.name}_{session}'
        states, state_path = load_states(low_freq, label)
        sources.append(f'{session} NREM: {state_path.name} ({states.get("params", {}).get("source", "score_sleep_states.py")})')
        for shank in args.shanks:
            lfp_path = Path(resolve_existing_file(low_freq / f'{label}_sh{shank}_lfp_traces.npz'))
            print(f'RUN {label} shank {shank}', flush=True)
            shank_rows, shank_events, shank_examples = process_shank(lfp_path, states, args, args.animal, rng)
            rows += [dict(session=session, **row) for row in shank_rows]
            events += [dict(session=session, **event) for event in shank_events]
            examples.setdefault(session, []).append(shank_examples)
            bad = [f'{row["channel_id"]}({row["bad_reason"]})' for row in shank_rows if not row['good']]
            best = max((row for row in shank_rows if 'rate_own' in row), key=lambda row: row['rate_own'], default=None)
            print(f'   excluded {len(bad)}: {" ".join(bad)}', flush=True)
            if best:
                print(f'   {best["clean_nrem_min"]:.1f} min clean NREM; top contact ch {best["channel_id"]} '
                      f'y={best["depth_um"]:g}: {best["rate_own"]:.2f}/min', flush=True)
    stem = f'{folder.name}_per_channel_spindle'
    write_csv(out / f'{stem}_{"screen" if args.screen_only else "summary"}.csv', rows)
    if not args.screen_only:
        write_csv(out / f'{stem}_events.csv', events)
        info = dict(session_name=folder.name, stamp=' ; '.join(sources) + f' | LFP: {low_freq}')
        (out / f'{stem}_parameters.json').write_text(json.dumps(dict(
            vars(args), out_dir=str(out), band_hz=BAND, sources=sources, low_freq=str(low_freq),
            layer5_um=LAYER5_UM.get(args.animal), known_bad=KNOWN_BAD.get(args.animal),
            run=f'{datetime.now():%Y-%m-%d %H:%M}'), indent=2, default=str), encoding='utf-8')
        plot_all(rows, examples, args.sessions, args, out, info)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
