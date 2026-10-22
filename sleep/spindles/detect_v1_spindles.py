"""Locate spindle-rich V1 contacts and detect NREM events from extracted LFP.

Defaults implement the proposed 9--17 Hz heuristic, not a published detector.
Input traces are time x channels, as written by extract_sleep_lfp.py.
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

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def nrem_samples(states, n, fs):
    """Map centered scorer epochs to samples without extrapolating across gaps."""
    times = np.asarray(states['epoch_times'], float)
    labels = np.asarray(states['states'])
    valid = np.asarray(states['valid_mask'], bool)
    artifact = np.asarray(states['artifact_mask'], bool)
    width = float(states['epoch_sec'])
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or not np.isfinite(width) or width <= 0
            or any(a.shape != times.shape for a in (labels, valid, artifact))):
        raise ValueError('Invalid sleep scorer epoch arrays')
    sample_times = np.arange(n) / fs
    # Nearest center; explicit half-open epoch support prevents extrapolation.
    right = np.searchsorted(times, sample_times).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    idx = np.where(abs(sample_times - times[left]) < abs(sample_times - times[right]), left, right)
    supported = ((sample_times >= times[idx] - width / 2)
                 & (sample_times < times[idx] + width / 2))
    return supported & (labels[idx] == 'NREM') & valid[idx] & ~artifact[idx]


def detect_events(amplitude, valid, fs, *, boundary=1.5, peak=3.,
                  min_duration=.3, max_duration=3., merge_gap=.2,
                  center=None, scale=None):
    """`center`/`scale` replace this trace's own NREM mean/SD (shared threshold)."""
    values = amplitude[valid]
    sd = scale if scale is not None else (np.std(values) if len(values) else 0.)
    if not np.isfinite(sd) or sd <= 0:
        return [], np.full(len(amplitude), np.nan)
    z = (amplitude - (np.mean(values) if center is None else center)) / sd
    spans = runs(valid & (z >= boundary))
    merged = []
    for start, end in spans:
        if (merged and start - merged[-1][1] <= merge_gap * fs
                and valid[merged[-1][1]:start].all()):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    events = []
    for start, end in merged:
        # A truncated event touching excluded data has unknown duration.
        if start == 0 or end == len(valid) or not valid[start - 1] or not valid[end]:
            continue
        p = start + int(np.argmax(z[start:end]))
        if min_duration <= (end - start) / fs <= max_duration and z[p] >= peak:
            events.append(dict(onset_s=start / fs, peak_s=p / fs, offset_s=end / fs,
                               duration_s=(end - start) / fs, peak_z=float(z[p]),
                               peak_power=float(amplitude[p] ** 2)))
    return events, z


def nrem_envelope(traces, fs, nrem, channels, *, band=(9., 17.), edge_sec=1.):
    """Smoothed band amplitude of `channels`, filtered per contiguous NREM run.

    Returns (amplitude, valid): amplitude is NaN outside the edge-guarded runs
    and on channels not requested; valid marks the guarded NREM samples.
    """
    n, c = traces.shape
    sos = butter(4, band, fs=fs, btype='bandpass', output='sos')
    guard = int(np.ceil(edge_sec * fs))
    valid = np.zeros(n, bool)
    segments = [(a, b) for a, b in runs(nrem) if b - a > 2 * guard + 32]
    for a, b in segments:
        valid[a + guard:b - guard] = True
    amplitude = np.full((n, c), np.nan, dtype=np.float32)
    for ch in channels:
        for a, b in segments:
            filtered = sosfiltfilt(sos, traces[a:b, ch].astype(float))
            env = np.abs(hilbert(filtered))
            env = uniform_filter1d(env, size=max(1, int(.1 * fs)))
            amplitude[a + guard:b - guard, ch] = env[guard:-guard]
    return amplitude, valid


def analyze(traces, fs, depths, nrem, *, excluded=(), band=(9., 17.),
            edge_sec=1., min_clean_sec=10., region_fraction=.6,
            min_region_channels=3, max_depth_gap_um=50., **event_options):
    """Process each clean NREM run separately; never concatenate discontinuities.

    Channel QC rejects nonfinite/flat channels and extreme broadband RMS
    outliers. Explicit exclusions remain essential for known bad contacts.
    """
    traces = np.asarray(traces)
    depths = np.asarray(depths, float)
    if (traces.ndim != 2 or depths.shape != (traces.shape[1],)
            or np.shape(nrem) != (traces.shape[0],)
            or not np.all(np.isfinite(depths)) or np.any(np.diff(depths) < 0)):
        raise ValueError('Expected time x channels, sorted finite depths, and sample NREM mask')
    if not np.isfinite(fs) or not 0 < band[0] < band[1] < fs / 2:
        raise ValueError('Spindle band must lie below Nyquist')
    if (edge_sec <= 0 or min_clean_sec <= 0 or not 0 < region_fraction <= 1
            or min_region_channels < 2 or max_depth_gap_um <= 0):
        raise ValueError('Invalid guard, exposure, or region parameters')
    n, c = traces.shape
    if np.sum(nrem) < min_clean_sec * fs:
        raise ValueError('Insufficient valid NREM')
    rms = np.full(c, np.nan)
    good = np.ones(c, bool)
    for ch in range(c):
        x = traces[nrem, ch].astype(float)
        good[ch] = np.isfinite(x).all() and np.std(x) > 0
        if good[ch]:
            rms[ch] = np.std(x)
    good[list(excluded)] = False
    median_rms = np.nanmedian(rms[good]) if good.any() else np.nan
    good &= (rms >= median_rms / 10) & (rms <= median_rms * 10)
    amplitude, valid = nrem_envelope(traces, fs, nrem, np.flatnonzero(good),
                                     band=band, edge_sec=edge_sec)
    if valid.sum() < min_clean_sec * fs:
        raise ValueError('Insufficient NREM after filter-edge guards')
    power = np.full(c, np.nan)
    for ch in np.flatnonzero(good):
        power[ch] = np.mean(amplitude[valid, ch].astype(float) ** 2)
    # Smooth only inside runs of good contacts and across small physical gaps.
    smooth = np.full(c, np.nan)
    blocks = []
    for a, b in runs(good):
        cuts = np.r_[a, a + 1 + np.flatnonzero(np.diff(depths[a:b]) > max_depth_gap_um), b]
        for left, right in zip(cuts[:-1], cuts[1:]):
            blocks.append((left, right))
            smooth[left:right] = uniform_filter1d(power[left:right], size=3, mode='nearest')
    regions = []
    if good.any():
        for a, b in blocks:
            for left, right in runs(smooth[a:b] >= region_fraction * np.nanmax(smooth)):
                if right - left >= min_region_channels:
                    regions.append((a + left, a + right))
    best = -1
    region = np.zeros(c, bool)
    events = []
    if regions:
        a, b = max(regions, key=lambda span: np.mean(smooth[span[0]:span[1]]))
        region[a:b] = True
        best = a + int(np.argmax(power[a:b]))
        events, _ = detect_events(amplitude[:, best], valid, fs, **event_options)
    # +/- 1.5 s event-triggered profile, 50 Hz output; only complete clean windows.
    offsets = np.arange(-int(1.5 * fs), int(1.5 * fs) + 1, max(1, int(fs / 50)))
    triggered = np.zeros((c, len(offsets)))
    count = 0
    for event in events:
        idx = int(round(event['peak_s'] * fs)) + offsets
        if idx[0] >= 0 and idx[-1] < n and valid[idx[0]:idx[-1] + 1].all():
            triggered += amplitude[idx].T.astype(float) ** 2
            count += 1
    triggered = triggered / count if count else np.full_like(triggered, np.nan)
    return dict(mean_9_17_power=power, smoothed_power=smooth, good_channel=good,
                broadband_rms=rms, spindle_region=region, best_index=best,
                clean_nrem_sec=float(valid.sum() / fs), events=events,
                triggered_power=triggered, triggered_time_s=offsets / fs,
                triggered_event_count=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shank', required=True, type=int, help='Known V1 shank (no anatomical inference)')
    parser.add_argument('--lfp', type=Path, help='Explicit extracted LFP NPZ; requires --states')
    parser.add_argument('--states', type=Path, help='Canonical sleep_periods.pkl from score_sleep_states.py')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--exclude-channels', nargs='*', default=[], help='Original channel IDs')
    parser.add_argument('--depth-range', nargs=2, type=float, metavar=('MIN', 'MAX'))
    parser.add_argument('--peak-threshold', type=float, default=3.)
    parser.add_argument('--boundary-threshold', type=float, default=1.5)
    parser.add_argument('--min-duration', type=float, default=.3)
    parser.add_argument('--max-duration', type=float, default=3.)
    parser.add_argument('--merge-gap', type=float, default=.2)
    parser.add_argument('--max-depth-gap-um', type=float, default=50.)
    parser.add_argument('--min-region-channels', type=int, default=3)
    args = parser.parse_args()
    if bool(args.lfp) != bool(args.states):
        parser.error('--lfp and --states must be supplied together')
    if not (0 < args.boundary_threshold <= args.peak_threshold
            and 0 < args.min_duration <= args.max_duration and args.merge_gap >= 0):
        parser.error('Invalid event thresholds/durations')
    if args.depth_range and args.depth_range[0] > args.depth_range[1]:
        parser.error('Depth range must be increasing')
    if args.lfp:
        jobs = [(args.lfp, args.states)]
    else:
        from sleep_pipeline_config import (low_freq_folder, session_name, sleep_sessions,
                                           active_sleep_sessions, resolve_existing_file)
        jobs = []
        for cfg in active_sleep_sessions(sleep_sessions).values():
            label = session_name + cfg['suffix']
            jobs.append((Path(resolve_existing_file(low_freq_folder / f'{label}_sh{args.shank}_lfp_traces.npz')),
                         Path(resolve_existing_file(low_freq_folder / 'sleep_segmentation' / f'{label}_sleep_periods.pkl'))))
    for lfp_path, state_path in jobs:
        process_file(lfp_path, state_path, args)


def process_file(lfp_path, state_path, args):
    with np.load(lfp_path, allow_pickle=False) as data:
        traces = data['traces']
        fs = float(data['sampling_rate'])
        ids = data['channel_ids']
        depths = data['ycoord']
        offset = float(data['sleep_start_sample']) / float(data['original_fs'])
        if int(data['shank']) != args.shank:
            raise ValueError('Requested V1 shank does not match LFP metadata')
        with state_path.open('rb') as stream:
            states = pickle.load(stream)
        expected = str(data['session_name']) + '_' + str(data['sleep_session'])
        if states['session'] != expected:
            raise ValueError(f'State/LFP session mismatch: expected {expected}')
        if int(states.get('sleep_start_sample') or 0) != int(data['sleep_start_sample']):
            raise ValueError('State/LFP sleep-start mismatch')
    unknown = set(args.exclude_channels) - set(ids.astype(str))
    if unknown:
        raise ValueError(f'Unknown excluded channel IDs: {sorted(unknown)}')
    excluded = np.isin(ids.astype(str), args.exclude_channels)
    if args.depth_range:
        excluded |= (depths < args.depth_range[0]) | (depths > args.depth_range[1])
    result = analyze(traces, fs, depths, nrem_samples(states, len(traces), fs),
                     excluded=np.flatnonzero(excluded), peak=args.peak_threshold,
                     boundary=args.boundary_threshold, min_duration=args.min_duration,
                     max_duration=args.max_duration, merge_gap=args.merge_gap,
                     max_depth_gap_um=args.max_depth_gap_um,
                     min_region_channels=args.min_region_channels)
    best = result['best_index']
    for event in result['events']:
        event.update(best_channel=str(ids[best]), best_depth_um=float(depths[best]), shank=args.shank)
        for name in ('onset', 'peak', 'offset'):
            event[name + '_recording_s'] = event[name + '_s'] + offset
    out = args.out_dir or lfp_path.parent / 'v1_spindles'
    out.mkdir(parents=True, exist_ok=True)
    stem = lfp_path.stem.replace('_lfp_traces', '_v1_spindles')
    arrays = {k: v for k, v in result.items() if k != 'events'}
    np.savez_compressed(out / (stem + '.npz'), **arrays, channel_ids=ids, depth_um=depths,
                        recording_start_s=offset)
    fields = ['onset_s', 'peak_s', 'offset_s', 'duration_s', 'peak_z', 'peak_power',
              'best_channel', 'best_depth_um', 'shank', 'onset_recording_s',
              'peak_recording_s', 'offset_recording_s']
    with (out / (stem + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result['events'])
    metadata = dict(lfp=str(lfp_path), states=str(state_path),
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    band_hz=[9, 17], edge_guard_s=1., envelope_smoothing_s=.1,
                    region_fraction=.6, min_clean_nrem_s=10., rms_outlier_factor=10.,
                    best_channel=str(ids[best]) if best >= 0 else None,
                    best_depth_um=float(depths[best]) if best >= 0 else None,
                    clean_nrem_sec=result['clean_nrem_sec'], event_count=len(result['events']),
                    status='candidate_region' if best >= 0 else 'no_contiguous_region',
                    note='Probe y-coordinate is not anatomical layer or spindle generator; power uses input LFP units squared.')
    (out / (stem + '.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plot_result(result, depths, out / (stem + '.png'))
    print(f'{stem}: channel={metadata["best_channel"]}, depth={metadata["best_depth_um"]}, '
          f'events={len(result["events"])}; saved to {out}')


def plot_result(result, depths, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout='constrained')
    axes[0].plot(result['mean_9_17_power'], depths, '.-', label='NREM power')
    axes[0].plot(result['smoothed_power'], depths, label='Depth-smoothed')
    region = result['spindle_region']
    axes[0].scatter(result['mean_9_17_power'][region], depths[region], c='orange', label='Selected region')
    best = result['best_index']
    if best >= 0:
        axes[0].axhline(depths[best], c='orange', ls='--')
    axes[0].set(xlabel='9–17 Hz power (input units²)', ylabel='Probe y-coordinate (µm)')
    axes[0].legend()
    if result['triggered_event_count']:
        # Normalize within contact so the heatmap also exposes spatial extent.
        relative = result['triggered_power'] / result['mean_9_17_power'][:, None]
        mesh = axes[1].pcolormesh(result['triggered_time_s'], np.arange(len(depths)), relative, shading='auto')
        fig.colorbar(mesh, ax=axes[1], label='Power / channel NREM mean')
        axes[1].set(xlabel='Time from spindle peak (s)', ylabel='Depth-ordered contact index')
    else:
        axes[1].text(.5, .5, 'No complete event-triggered windows', ha='center', transform=axes[1].transAxes)
    fig.suptitle(f'V1 spindle candidates: {len(result["events"])} events; '
                 f'{result["triggered_event_count"]} complete windows')
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
