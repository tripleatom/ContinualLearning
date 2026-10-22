"""Whole-passive PSD and trial-averaged LFP/CSD from NWB and passive DIO.

Run from any directory; see README.md for methods and clock assumptions.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import re

import h5py
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, spectrogram
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ((500, 5000),)
# Displayed time window (s from onset). Saved arrays keep the full epoch.
VIEW_SEC = (0, .2)
LFP_DISPLAY_GAIN = 6.0  # Visual amplification only; voltage arrays are unchanged.
DEFAULT_DAY = Path(r'\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\260727')


def find_outlier_channels(lfp, time, depth_um, threshold=.25, reach_um=50):
    """Contacts whose evoked mean LFP departs from their neighbors' interpolation.

    Score = RMS(channel - linear interpolation of the nearest accepted contact
    on each side, within reach_um) / mean RMS of those two contacts, over
    0..0.3 s. The worst contact above threshold is rejected and all scores are
    recomputed, so a broken contact does not condemn its neighbors. Contacts
    without an accepted neighbor on both sides cannot be scored and are kept.
    """
    v = lfp[(time >= 0) & (time <= .3)]
    rms = np.sqrt(np.mean(v ** 2, axis=0))
    rejected = np.zeros(len(depth_um), bool)
    while threshold > 0:
        good = np.flatnonzero(~rejected)
        worst, worst_score = None, threshold
        for i in good:
            offset = depth_um[good] - depth_um[i]
            above, below = good[(offset < 0) & (offset >= -reach_um)], good[(offset > 0) & (offset <= reach_um)]
            if not len(above) or not len(below):
                continue
            a, b = above[-1], below[0]
            w = (depth_um[i] - depth_um[a]) / (depth_um[b] - depth_um[a])
            score = np.sqrt(np.mean((v[:, i] - (1 - w) * v[:, a] - w * v[:, b]) ** 2)) / np.mean(rms[[a, b]])
            if score > worst_score:
                worst, worst_score = i, score
        if worst is None:
            break
        rejected[worst] = True
    return rejected


def find_power_outliers(power, depth_um, exclude=None, ratio=2., reach_um=50, noise=None):
    """Attenuated or noisy contacts, judged against their neighbors' band power.

    A contact is rejected when its power is more than `ratio` times below the
    median of the accepted contacts within reach_um (at least three, on either
    side, so one hot neighbor cannot move the median). Power that far ABOVE the neighbors is rejected only if `noise`
    (density well above the spike band) is raised by the same factor: a contact
    beside a large unit has excess power confined to the spike band and must
    be kept. Worst first, rescoring after each rejection. Contacts in
    `exclude` are never used as neighbors and are not returned.
    """
    exclude = np.zeros(len(depth_um), bool) if exclude is None else exclude
    rejected = np.zeros(len(depth_um), bool)
    while ratio > 1:
        good = np.flatnonzero(~exclude & ~rejected)
        worst, worst_score = None, np.log(ratio)
        for i in good:
            near = good[(good != i) & (abs(depth_um[good] - depth_um[i]) <= reach_um)]
            if len(near) < 3 or power[i] <= 0:
                continue
            score = np.log(power[i] / np.median(power[near]))
            if score > 0:
                broadband = noise is not None and noise[i] > ratio * np.median(noise[near])
                score = score if broadband else 0
            score = abs(score)
            if score > worst_score:
                worst, worst_score = i, score
        if worst is None:
            break
        rejected[worst] = True
    return rejected


def floor_scaled_power(power, rejected, percentiles=(5, 95)):
    """Power above the shank's floor, in units of its floor-to-top range.

    Floor and top are the given percentiles over accepted contacts, so neither
    a shared pedestal nor a single hot contact sets the scale. Returns the
    profile (NaN at rejected contacts), floor and top in the input units.
    """
    power = np.where(rejected, np.nan, power)
    if not np.isfinite(power).any():
        return power, np.nan, np.nan
    floor, top = np.nanpercentile(power, percentiles)
    return (power - floor) / (top - floor) if top > floor else power * np.nan, floor, top


def fill_rejected(lfp, depth_um, rejected):
    """Replace isolated rejected contacts by the mean of their two neighbors.

    Returns the filled LFP and the contacts usable for CSD. A rejected contact
    is filled only if both adjacent contacts are accepted and at the regular
    spacing; otherwise it is dropped and CSD stays masked around it.
    """
    lfp, used = lfp.copy(), ~rejected
    gaps = np.diff(depth_um)
    for i in np.flatnonzero(rejected):
        if (0 < i < len(depth_um) - 1 and not rejected[i - 1] and not rejected[i + 1]
                and max(gaps[i - 1], gaps[i]) <= gaps.min() * 1.5):
            lfp[:, i] = (lfp[:, i - 1] + lfp[:, i + 1]) / 2
            used[i] = True
    return lfp, used


def smooth_laminar_csd(lfp, depth_um, rejected=None):
    """Three-contact Hamming smoothing of LFP, then spatial curvature.

    Weights follow Schaefer et al. 2015 (0.23, 0.54, 0.23). No smoothing
    across missing contacts; no extrapolated endpoint CSD. Neighboring
    endpoints use replicated values for smoothing only, not differentiation.
    Rejected contacts are filled or dropped first (see fill_rejected).
    """
    if rejected is not None and np.any(rejected):
        lfp, used = fill_rejected(lfp, depth_um, rejected)
        out = np.full(lfp.shape, np.nan)
        out[:, used] = smooth_laminar_csd(lfp[:, used], depth_um[used])
        return out
    gaps = np.diff(depth_um)
    segments = np.split(np.arange(len(depth_um)), np.flatnonzero(gaps > gaps.min() * 1.5) + 1)
    smoothed = lfp.copy()
    for idx in segments:
        if len(idx) < 3:
            continue
        pad = np.pad(lfp[:, idx], ((0, 0), (1, 1)), mode='edge')
        smoothed[:, idx] = .23 * pad[:, :-2] + .54 * pad[:, 1:-1] + .23 * pad[:, 2:]
    return csd_curvature(smoothed, depth_um)


def passive_bounds(args):
    """Resolve the entire passive recording, including ITIs and idle periods."""
    if args.passive_start_sample is not None and args.passive_end_sample is not None:
        return args.passive_start_sample, args.passive_end_sample
    if args.passive_start_sample is not None or args.passive_end_sample is not None:
        raise ValueError('Provide both passive sample bounds')
    offset, matches = 0, []
    for line in (args.nwb_dir / 'conversion_list.txt').read_text().splitlines():
        # Older lists say "timestamps", newer ones "samples" (packets + interpolated fills).
        match = re.match(r'\s*\d+\.\s+(.+\.rec):\s+(\d+) (?:timestamps|samples)', line)
        if match:
            name, count = match[1], int(match[2])
            if 'passive' in name.lower():
                matches.append((offset, offset + count))
            offset += count
    if len(matches) != 1:
        raise ValueError('Cannot select one passive recording; supply explicit sample bounds')
    return matches[0]


def passive_psd(path, labels, bounds, batch_windows=8):
    """Welch density over a continuous passive recording, not trial epochs.

    Global 2-second Hann windows, 50% overlap, 0.5-Hz bins. Chunk boundaries
    neither duplicate nor drop Welch windows. Short final remainder follows
    conventional Welch behavior (only complete windows).

    The spectrum and band density are means over windows. Band power is the
    median over windows of each window's integrated power, so brief large
    events (movement, chewing, cable knocks) do not dominate it; the mean and
    the per-window values are returned alongside.
    """
    with h5py.File(path, 'r') as f:
        es = f['acquisition/ElectricalSeries']
        data = es['data']
        fs = float(es['starting_time'].attrs['rate'])
        # Native-rate Welch preserves the entire 500–5000 Hz band. The LFP
        # resampling/filtering path must never be used for this calculation.
        target_fs = int(round(fs))
        if abs(fs - target_fs) > 1e-9 or fs / 2 <= max(hi for _, hi in BANDS):
            raise ValueError('PSD needs integer sampling rate and Nyquist above 5000 Hz')
        factor, nperseg, step = 1, 2 * target_fs, target_fs
        start, end = bounds
        if not 0 <= start < end <= len(data):
            raise ValueError('Passive recording bounds outside NWB')
        table = f[es['electrodes'].attrs['table']]
        all_labels = [v.decode() if isinstance(v, bytes) else str(v) for v in table['label'][:][es['electrodes'][:]]]
        keep = [all_labels.index(str(label)) for label in labels]
        gain = float(data.attrs.get('conversion', 1)) * 1e6
        offset = float(data.attrs.get('offset', 0)) * 1e6
        cg = es['channel_conversion'][:] if 'channel_conversion' in es else np.ones(data.shape[1])
        n_samples = (end - start) // factor
        n_windows = (n_samples - nperseg) // step + 1
        if n_windows < 1:
            raise ValueError('Passive recording must contain at least two seconds')
        total = np.zeros((nperseg // 2 + 1, len(keep)))
        window_power = np.zeros((n_windows, len(keep), len(BANDS)), np.float32)
        for first in range(0, n_windows, batch_windows):
            count = min(batch_windows, n_windows - first)
            first_sample = first * step
            final_sample = first_sample + (count - 1) * step + nperseg
            left_sample = max(0, first_sample - target_fs)
            right_sample = min(n_samples, final_sample + target_fs)
            raw = data[start + left_sample * factor:start + right_sample * factor, :].astype(np.float32)
            raw = raw[:, keep] * (gain * cg[keep]).astype(np.float32) + offset
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite passive data')
            trace = raw
            del raw
            trace = trace[first_sample - left_sample:final_sample - left_sample]
            # Per-window periodograms; their mean over windows is the Welch estimate.
            freq, _, windows = spectrogram(trace, fs=target_fs, window='hann', nperseg=nperseg,
                                           noverlap=nperseg - step, detrend='constant', scaling='density', axis=0)
            total += windows.sum(axis=-1)
            for b, (lo, hi) in enumerate(BANDS):
                select = (freq >= lo) & (freq <= hi)
                window_power[first:first + count, :, b] = np.trapz(windows[select], freq[select], axis=0).T
            del windows
            print(f'{path.stem}: PSD {first + count}/{n_windows} windows', flush=True)
        density = total / n_windows
        mean_power = np.column_stack([np.trapz(density[(freq >= lo) & (freq <= hi)],
                                                     freq[(freq >= lo) & (freq <= hi)], axis=0)
                                        for lo, hi in BANDS])
        band_density = mean_power / np.array([hi - lo for lo, hi in BANDS])
        band_power = np.median(window_power, axis=0).astype(float)
        maximum = band_power.max(axis=0)
        # Density well above the spike band: amplifier/contact noise, not units.
        above = (freq >= 1.8 * max(hi for _, hi in BANDS)) & (freq <= .93 * fs / 2)
        noise = density[above].mean(axis=0) if above.any() else np.full(len(keep), np.nan)
        return dict(psd_frequency_hz=freq, psd_uv2_per_hz=density, noise_floor_uv2_per_hz=noise,
                    band_density_uv2_per_hz=band_density,
                    band_power_uv2=band_power, band_power_mean_uv2=mean_power,
                    window_band_power_uv2=window_power, psd_sample_rate_hz=fs,
                    normalized_power=np.divide(band_power, maximum, out=np.zeros_like(band_power), where=maximum > 0),
                    passive_sample_bounds=np.array(bounds), passive_duration_sec=(end - start) / fs,
                    psd_window_count=n_windows)


def csd_curvature(lfp, depth_um):
    """Negative second derivative, uV/mm^2; omit unsupported endpoints.

    Three-point finite differences support unequal spacing, but gaps larger
    than 1.5 times the smallest contact spacing are masked, not bridged.
    """
    z = np.asarray(depth_um, float) / 1000
    gaps = np.diff(z)
    if len(z) < 3 or np.any(gaps <= 0):
        raise ValueError('CSD needs at least three distinct, increasing depths')
    out = np.full_like(lfp, np.nan, dtype=float)
    out[:, 1:-1] = -2 * (np.diff(lfp, axis=1)[:, 1:] / gaps[1:]
                            - np.diff(lfp, axis=1)[:, :-1] / gaps[:-1]) / (gaps[1:] + gaps[:-1])
    valid = (gaps[:-1] <= gaps.min() * 1.5) & (gaps[1:] <= gaps.min() * 1.5)
    out[:, np.flatnonzero(~valid) + 1] = np.nan
    return out


def read_timing(args):
    with np.load(args.dio) as dio:
        on = np.asarray(dio['rising_times'], float).ravel() / args.dio_fs
        off = np.asarray(dio['falling_times'], float).ravel() / args.dio_fs
    if (on.shape != off.shape or not len(on) or not np.all(np.isfinite(on))
            or not np.all(np.isfinite(off)) or np.any(np.diff(on) <= 0)
            or np.any(off <= on) or np.any(off[:-1] > on[1:])):
        raise ValueError('DIO on/off pairs must be finite, ordered, nonoverlapping trials')
    duration = float(np.median(off - on))
    if np.max(abs(off - on - duration)) > 1 / args.dio_fs + 1e-9:
        raise ValueError('Variable stimulus durations: group trials by duration before averaging')
    return on + args.offset_sec, off + args.offset_sec, duration


def process_shank(path, args, on, off, duration, epoch_callback=None, filter_bands=None):
    fs_out = 2000
    pre_n = round(args.pre_sec * fs_out)
    n_time = pre_n + round(duration * fs_out) + 1
    time = (np.arange(n_time) - pre_n) / fs_out
    with h5py.File(path, 'r') as f:
        es = f['acquisition/ElectricalSeries']
        data = es['data']
        fs = float(es['starting_time'].attrs['rate'])
        factor = round(fs / fs_out)
        if abs(fs / fs_out - factor) > 1e-9:
            raise ValueError('Raw sampling rate must be an integer multiple of 2000 Hz')
        table = f[es['electrodes'].attrs['table']]
        rows = es['electrodes'][:]
        labels = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in table['label'][:][rows]])
        depths = table['rel_y'][:][rows].astype(float)
        x = table['rel_x'][:][rows]
        bad = set(map(int, args.bad_channels.read_text().split())) if args.bad_channels.exists() else set()
        channel_numbers = np.array([int(label.rsplit(':', 1)[1]) for label in labels])
        keep = np.flatnonzero(~np.isin(channel_numbers, list(bad)))
        keep = keep[np.argsort(depths[keep])]
        depths = depths[keep]
        if len(np.unique(x[keep])) != 1 or len(np.unique(depths)) != len(depths):
            raise ValueError('Expected one linear contact column per shank')
        if np.any(on - args.pre_sec < 0) or off[-1] >= len(data) / fs:
            raise ValueError('Trial window outside NWB recording')
        gain = float(data.attrs.get('conversion', 1)) * 1e6
        offset = float(data.attrs.get('offset', 0)) * 1e6
        if data.attrs.get('unit') not in ('volts', 'V'):
            raise ValueError('Expected NWB data in volts')
        channel_gain = es['channel_conversion'][:] if 'channel_conversion' in es else np.ones(data.shape[1])
        filter_bands = filter_bands or ((.5, 100),)
        lfp_sums = np.zeros((len(filter_bands), n_time, len(keep)))
        filters = [butter(3, (lo, hi) if lo is not None else hi, fs=fs_out,
                          btype='bandpass' if lo is not None else 'lowpass', output='sos')
                   for lo, hi in filter_bands]
        for first in range(0, len(on), args.batch_trials):
            stop = min(first + args.batch_trials, len(on))
            # Read contiguous batches, with real data padding for zero-phase filters.
            left = max(0, int((on[first] - args.pre_sec - 3) * fs) // factor * factor)
            right = min(len(data), int((off[stop - 1] + 3) * fs) + 1)
            raw = data[left:right, :].astype(np.float32)
            raw = raw[:, keep] * (gain * channel_gain[keep]).astype(np.float32) + offset
            if not np.all(np.isfinite(raw)):
                raise ValueError('Nonfinite raw data')
            traces = resample_poly(raw, 1, factor, axis=0).astype(np.float32)
            del raw
            starts = np.rint((on[first:stop] - left / fs) * fs_out).astype(int)
            for k, sos in enumerate(filters):
                lfp = sosfiltfilt(sos, traces, axis=0)
                for j, center in enumerate(starts):
                    epoch = lfp[center - pre_n:center - pre_n + n_time].copy()
                    epoch -= epoch[:pre_n].mean(axis=0)
                    lfp_sums[k] += epoch
                    if epoch_callback is not None and k == 0:
                        epoch_callback(first + j, epoch)
                del lfp
            print(f'{path.stem}: {stop}/{len(on)} trials', flush=True)
        means = lfp_sums / len(on)
        mean_lfp = means[0]
        result = dict(time_sec=time, depth_um=depths, channel_labels=labels[keep],
                    lfp_uv=mean_lfp, csd_uv_per_mm2=csd_curvature(mean_lfp, depths),
                    onset_sec=on, offset_sec=off, excluded_labels=labels[np.isin(channel_numbers, list(bad))])
        if len(filter_bands) > 1:
            result['filter_lfp_uv'] = means
        return result


def plot_results(results, output, pre_sec, duration, stamp=''):
    plot_depths = np.unique(np.concatenate([r['depth_um'] for _, r in results]))
    session_label = output.parent.name if output.name == 'layer_identification' else output.name
    view_start, view_end = max(VIEW_SEC[0], -pre_sec), min(VIEW_SEC[1], duration)
    for shank, r in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 10), width_ratios=[1, 2.2],
                                 sharey=True, squeeze=False, layout='constrained')
        for kind in ('power', 'lfp', 'csd'):
            # LFP and CSD share the same time/depth coordinates in panel B.
            ax = axes[0, 0 if kind == 'power' else 1]
            indices = np.searchsorted(plot_depths, r['depth_um'])
            if kind == 'power':
                for b, band in enumerate(BANDS):
                    profile = np.full(len(plot_depths), np.nan)
                    profile[indices], floor, top = floor_scaled_power(r['band_power_uv2'][:, b], r['rejected'])
                    ax.plot(profile, plot_depths, color='k', lw=1.4)
                for edge in (0, 1):
                    ax.axvline(edge, color='0.6', lw=.8, ls=':')
                # Hot contacts may exceed 1; the limits follow them up to 3.
                ax.set_xlim(-.25, min(max(1.25, np.nanmax(profile, initial=1) * 1.05), 3))
                ax.set_xlabel('Power above floor / (95th pct − floor)\n(500 Hz–5 kHz)')
                ax.grid(alpha=.2)
                # Absolute values, so a flat shank is recognisable despite the rescaling.
                ax.set_title(f'A. High-frequency power\nWhole passive recording: {float(r["passive_duration_sec"]):.1f} s\n'
                             f'floor (5th pct) {np.sqrt(floor):.1f}, 95th pct {np.sqrt(top):.1f} µV rms', fontsize=10)
            elif kind == 'lfp':
                # Same gain for all contacts within a shank; display a scale bar
                # per shank so weak responses remain visible across probes.
                shown = (r['time_sec'] >= view_start) & (r['time_sec'] <= view_end)
                trace_gain = LFP_DISPLAY_GAIN * np.min(np.diff(plot_depths)) * .42 / max(np.max(abs(r['lfp_uv'][shown][:, ~r['rejected']])), 1e-12)
                for ch, depth in enumerate(r['depth_um']):
                    label = str(r['channel_labels'][ch]).rsplit(':', 1)[-1]
                    if r['rejected'][ch]:
                        # Auto-rejected contact: no trace, ID kept in red.
                        ax.text(1.01, depth, f'{label} ×', transform=ax.get_yaxis_transform(), fontsize=6, va='center', color='r')
                        continue
                    # With depth increasing downward, positive voltage goes up.
                    ax.plot(r['time_sec'], depth - trace_gain * r['lfp_uv'][:, ch], color='k', lw=.7, zorder=3)
                    ax.text(1.01, depth, label, transform=ax.get_yaxis_transform(), fontsize=6, va='center')
                ax.axvline(0, color='0.45', lw=.8, ls='--')
                ax.axvline(duration, color='0.45', lw=.8, ls='--')
                ax.set_xlim(view_start, view_end)
                ax.set_xlabel('Time from onset (s)')
                ax.set_title(f'B. Trial-averaged LFP (0.5–100 Hz)\n{len(r["onset_sec"])} trials; channel IDs at right'
                             + (f'; {r["rejected"].sum()} auto-rejected (red ×)' if r['rejected'].any() else ''), fontsize=10)
                bar_uv = max(v for v in (.1, .2, .5, 1, 2, 5, 10, 20, 50, 100)
                             if v * trace_gain <= 45)
                bar_x = view_end - .05 * (view_end - view_start)
                ax.plot([bar_x] * 2, [plot_depths[-1] + 45, plot_depths[-1] + 45 - trace_gain * bar_uv], color='k', lw=2)
                ax.text(bar_x - .03, plot_depths[-1] + 45, f'{bar_uv:g} µV', ha='right', va='bottom', fontsize=7)
                ax.set_ylim(plot_depths[0] - 75, plot_depths[-1] + 70)
            else:
                csd = r['csd_uv_per_mm2']
                evoked = (r['time_sec'] >= max(view_start, 0)) & (r['time_sec'] <= view_end)
                finite_values = abs(csd[evoked][np.isfinite(csd[evoked])])
                limit = (float(np.percentile(finite_values, 99)) or 1) if finite_values.size else 1
                # Linear interpolation for display only, confined to contiguous
                # supported CSD depths. Never draw across missing-contact gaps.
                dense_depth = np.arange(plot_depths.min(), plot_depths.max() + 1, 5.)
                display = np.full((len(dense_depth), len(r['time_sec'])), np.nan)
                valid_idx = np.flatnonzero(np.isfinite(csd).all(axis=0))
                breaks = np.flatnonzero(np.diff(r['depth_um'][valid_idx]) > np.min(np.diff(plot_depths)) * 1.5) + 1
                for idx in np.split(valid_idx, breaks):
                    if len(idx) == 0:
                        continue
                    if len(idx) == 1:
                        display[abs(dense_depth - r['depth_um'][idx[0]]) <= 2.5] = csd[:, idx[0]] / limit
                        continue
                    select = (dense_depth >= r['depth_um'][idx[0]]) & (dense_depth <= r['depth_um'][idx[-1]])
                    display[select] = interp1d(r['depth_um'][idx], csd[:, idx], axis=1)(dense_depth[select]).T / limit
                for row, limits in enumerate([(view_start, view_end)]):
                    axis = axes[0, 1]
                    im = axis.pcolormesh(r['time_sec'][::4], dense_depth, display[:, ::4], shading='nearest', cmap='jet', vmin=-1, vmax=1, rasterized=True, zorder=0)
                    axis.axvline(0, color='k', lw=.8, ls='--')
                    axis.set_xlim(*limits)
                    axis.set_xlabel('Time from onset (s)')
                    axis.set_title(f'B. LFP traces over CSD — {len(r["onset_sec"])} trials\n'
                                   f'LFP: 0.5–100 Hz, display gain ×{LFP_DISPLAY_GAIN:g}; CSD scale: ±{limit:.0f} µV/mm²\n'
                                   'Channel IDs at right; red × = rejected'
                                   + ('; ◂ = filled for CSD' if r['filled'].any() else ''), fontsize=10)
                    if r['filled'].any():
                        # An empty unclipped artist collapses the constrained layout.
                        axis.plot(np.full(r['filled'].sum(), limits[1]), r['depth_um'][r['filled']],
                                  ls='', marker='<', color='k', ms=6, clip_on=False)
        axes[0, 0].set_ylabel('Probe depth (µm; 0 nearer surface)')
        # All panels share y: invert exactly once.
        axes[0, 0].invert_yaxis()
        fig.colorbar(im, ax=axes[0, 1], shrink=.6, pad=.12,
                     label='CSD / displayed-window scale (99th percentile)\nnegative / blue = sink; 0 = green; positive / red = source')
        fig.suptitle(f'{session_label} — Shank {shank}\n'
                     '500 Hz–5 kHz power normalized across depth; LFP baseline subtracted; CSD from smoothed mean LFP', fontsize=12)
        # Reproducibility stamp: inputs and parameters behind this figure.
        fig.get_layout_engine().set(rect=(0, .035, 1, .965))
        fig.text(.005, .003, stamp, fontsize=6, color='0.35', va='bottom')
        fig.savefig(output / f'shank{shank}_layer_profiles.png', dpi=180, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb-dir', type=Path, default=DEFAULT_DAY / 'CnL46_20260727')
    parser.add_argument('--dio', type=Path, default=DEFAULT_DAY / 'CnL46_drifting_grating_exp_20260727_131003_DIO.npz')
    parser.add_argument('--bad-channels', type=Path)
    parser.add_argument('--output', type=Path, help='Defaults to <nwb-dir>/layer_identification')
    parser.add_argument('--pre-sec', type=float, default=.5)
    parser.add_argument('--dio-fs', type=float, default=30000)
    parser.add_argument('--offset-sec', type=float, default=0, help='Passive origin in concatenated NWB seconds')
    parser.add_argument('--shanks', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--batch-trials', type=int, default=32)
    parser.add_argument('--passive-start-sample', type=int, help='Otherwise inferred from conversion_list.txt')
    parser.add_argument('--passive-end-sample', type=int, help='Exclusive end of the complete passive recording')
    parser.add_argument('--reuse', action='store_true', help='Use saved arrays only if parameters and source paths match')
    parser.add_argument('--outlier-threshold', type=float, default=.25,
                        help='Reject contacts deviating from neighbor interpolation by this RMS fraction; 0 disables')
    parser.add_argument('--power-outlier-ratio', type=float, default=2.,
                        help='Reject contacts whose band power is this many times above or below their neighbors; 0 disables')
    args = parser.parse_args()
    if args.pre_sec <= 0 or args.batch_trials <= 0:
        parser.error('pre-sec and batch-trials must be positive')
    args.bad_channels = args.bad_channels or args.nwb_dir / 'bad_channels.txt'
    args.output = args.output or args.nwb_dir / 'layer_identification'
    args.output.mkdir(parents=True, exist_ok=True)
    on, off, duration = read_timing(args)
    bounds = passive_bounds(args)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'reuse'}
    metadata.update(analysis_version=2, stimulus_duration_sec=duration, n_trials=len(on), bands_hz=BANDS,
                    lfp_band_hz=[.5, 100], sample_rate_hz=2000, reference='as recorded',
                    passive_sample_bounds=list(bounds),
                    psd='whole passive recording at native sampling rate; 2 s Hann windows, 50% overlap, constant detrend; '
                        'integrated 500–5000 Hz power per window, median over windows (mean and per-window values also saved)',
                    power_normalization='(power - 5th percentile) / (95th - 5th percentile) over accepted contacts, per shank',
                    power_rejection='band power more than power_outlier_ratio times below the median of accepted contacts '
                                    'within 50 um (at least three), or that far above it with 9-14 kHz noise raised alike '
                                    '(excess confined to the spike band is a unit and is kept); worst first, after the LFP-based rejection',
                    outlier_rejection='evoked mean LFP (0..0.3 s) vs linear interpolation of nearest accepted contacts within 50 um, '
                                      'RMS residual / mean neighbor RMS > outlier_threshold, worst first; isolated rejects filled '
                                      'by the two-neighbor mean for CSD only, others dropped',
                    csd='three-contact Hamming LFP smoothing (0.23, 0.54, 0.23), negative second derivative; uV/mm^2; no conductivity assumption',
                    csd_display='linear interpolation within supported depth runs; per-shank 99th percentile of |CSD| over the displayed window; jet colormap: blue sinks (negative), green zero, red sources (positive)')
    fingerprint = json.dumps(metadata, sort_keys=True)
    results = []
    for shank in args.shanks:
        paths = list(args.nwb_dir.glob(f'*sh{shank}.nwb'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one NWB for shank {shank}: {paths}')
        save = args.output / f'shank{shank}_averages.npz'
        have_psd = False
        if args.reuse and save.exists():
            with np.load(save) as stored:
                cached_parameters = json.loads(stored['parameters'].item())
                current_parameters = json.loads(fingerprint)
                # Relocating results does not change the computed arrays.
                cached_parameters.pop('output', None)
                current_parameters.pop('output', None)
                extraction_keys = ('nwb_dir', 'dio', 'bad_channels', 'pre_sec', 'dio_fs', 'offset_sec', 'lfp_band_hz', 'sample_rate_hz', 'reference')
                if any(cached_parameters.get(key) != current_parameters.get(key) for key in extraction_keys):
                    raise ValueError('Cached LFP parameters differ; rerun without --reuse')
                result = {k: stored[k] for k in stored.files if k != 'parameters'}
                if not (np.array_equal(result['onset_sec'], on) and np.array_equal(result['offset_sec'], off)):
                    raise ValueError('Cached trial timing differs')
                psd_keys = ('nwb_dir', 'bad_channels', 'passive_sample_bounds', 'sample_rate_hz',
                            'reference', 'psd', 'bands_hz', 'power_normalization')
                have_psd = (all(cached_parameters.get(key) == current_parameters.get(key) for key in psd_keys)
                            and 'psd_uv2_per_hz' in result and 'noise_floor_uv2_per_hz' in result)
        else:
            result = process_shank(paths[0], args, on, off, duration)
        comparison_power = args.output / 'filter_comparison' / f'shank{shank}_power_500_5000.npz'
        if args.reuse and not have_psd and comparison_power.exists():
            # Reuse the native-rate power already computed by the filter
            # comparison in this same session/output directory.
            with np.load(comparison_power) as cached:
                if (np.array_equal(cached['channel_labels'], result['channel_labels'])
                        and np.array_equal(cached['passive_sample_bounds'], bounds)
                        and cached['band_power_uv2'].shape == (len(result['channel_labels']), 1)
                        and 'window_band_power_uv2' in cached.files and 'noise_floor_uv2_per_hz' in cached.files
                        and float(cached['psd_sample_rate_hz']) > 10000):
                    result.update({k: cached[k] for k in cached.files if k != 'channel_labels'})
                    have_psd = True
        if not have_psd:
            result.update(passive_psd(paths[0], result['channel_labels'], bounds))
        result.pop('trial_power_uv2', None)
        result.pop('power_uv2', None)
        result['rejected_lfp'] = find_outlier_channels(result['lfp_uv'], result['time_sec'], result['depth_um'],
                                                       args.outlier_threshold)
        result['rejected_power'] = find_power_outliers(result['band_power_uv2'][:, 0], result['depth_um'],
                                                       result['rejected_lfp'], args.power_outlier_ratio,
                                                       noise=result['noise_floor_uv2_per_hz'])
        result['rejected'] = result['rejected_lfp'] | result['rejected_power']
        result['filled'] = fill_rejected(result['lfp_uv'], result['depth_um'], result['rejected'])[1] & result['rejected']
        # The unsmoothed CSD keeps every contact; only the smoothed one applies rejection.
        result['csd_raw_uv_per_mm2'] = csd_curvature(result['lfp_uv'], result['depth_um'])
        result['csd_uv_per_mm2'] = smooth_laminar_csd(result['lfp_uv'], result['depth_um'], result['rejected'])
        np.savez_compressed(save, **result, parameters=fingerprint)
        results.append((shank, result))
    stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} | script={Path(__file__).name} | nwb_dir={args.nwb_dir}\n"
             f"dio={args.dio} | bad_channels={args.bad_channels}\n"
             f"n_trials={len(on)}, stimulus={duration:g} s, baseline=-{args.pre_sec:g}..0 s, offset_sec={args.offset_sec:g}, "
             f"outlier_threshold={args.outlier_threshold:g}, power_outlier_ratio={args.power_outlier_ratio:g}, "
             f"power = median over 2 s windows, shown {VIEW_SEC[0]:g}..{VIEW_SEC[1]:g} s, LFP display gain={LFP_DISPLAY_GAIN:g}, "
             f"LFP 0.5-100 Hz at 2 kHz, CSD = -d2V/dz2 after 3-contact Hamming smoothing, "
             f"PSD over passive samples [{bounds[0]}, {bounds[1]})")
    plot_results(results, args.output, args.pre_sec, duration, stamp)
    checks = []
    for shank, result in results:
        if not (np.isfinite(result['psd_uv2_per_hz']).all()
                and np.all(result['psd_uv2_per_hz'] >= 0)
                and np.isfinite(result['lfp_uv']).all()
                and np.allclose(result['lfp_uv'][result['time_sec'] < 0].mean(axis=0), 0, atol=1e-8)):
            raise ValueError(f'Numerical validation failed for shank {shank}')
        checks.append(dict(shank=shank, contacts=len(result['depth_um']),
                           csd_depths=int(np.isfinite(result['csd_uv_per_mm2'][0]).sum()),
                           trials=len(result['onset_sec']), psd_windows=int(result['psd_window_count']),
                           passive_duration_sec=float(result['passive_duration_sec']),
                           excluded=result['excluded_labels'].tolist(),
                           auto_rejected=result['channel_labels'][result['rejected']].tolist(),
                           auto_rejected_by_power=result['channel_labels'][result['rejected_power']].tolist(),
                           auto_rejected_filled=result['channel_labels'][result['filled']].tolist()))
    (args.output / 'validation.json').write_text(json.dumps(dict(analysis_version=2, checks=checks), indent=2))
    # Remove only the superseded aggregate figures produced by this script.
    for filename in ('figure1_power.pdf', 'figure2_lfp.pdf', 'figure3_csd.pdf',
                     'figure1_power.png', 'figure2_lfp.png', 'figure3_csd.png'):
        (args.output / filename).unlink(missing_ok=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {len(results)} combined shank figures and numerical results to {args.output}', flush=True)


if __name__ == '__main__':
    main()
