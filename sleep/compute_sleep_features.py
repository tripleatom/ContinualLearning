"""Spectrograms -> one band-power pickle per sleep session.

Per channel this computes delta / theta-ratio / sigma / gamma envelopes and a
per-channel PC1, all derived from the saved spectrogram (10 s window, 1 s
step, log-spaced 1-100 Hz), and stores them ON THE SPECTROGRAM TIME BASE
(1 s bins) in float32.

The band envelopes used to be computed by bandpass-filtering the LFP traces,
squaring, and smoothing with a `band_params['smoothing_window']`-second moving
average. The spectrogram's own 10 s window already provides that smoothing,
so each envelope is now simply the PSD integrated over the band's frequency
rows - same information, and this stage no longer reads the LFP traces at all
(which was its slowest part: loading the full-rate traces plus six
forward-backward filters per channel). Every consumer takes log + robust-z of
these envelopes, so the change in absolute calibration (filter passband vs
Welch integral) cancels downstream. One edge moves: delta (0.5-4 Hz in
band_params) is integrated from 1 Hz, the bottom of the saved grid - the LFP
is highpassed at 1 Hz (preproc_params['lfp_min']) anyway, so the lost
half-octave carried little power to begin with.

The output keeps the exact 'compact' schema compute_sleep_features has written
since band powers moved onto the spectrogram grid, so score_nrem_epochs.py,
score_nrem_delta_velocity.py, plot_sleep_spectrograms.py and band_powers_io.py
all read it unchanged. The spectrogram itself is still not copied in - it sits
in `*_sh<N>_spectrograms.npz` beside this file, and
band_powers_io.load_spectrograms() reads it from there.
"""
import argparse
import errno
import numpy as np
import pickle
from sklearn.decomposition import PCA

from sleep_pipeline_config import (
    rec_folder,
    session_name,
    shanks,
    low_freq_folder,
    band_params,
    sleep_sessions,
    active_sleep_sessions,
    resolve_output_folder,
    resolve_existing_file,
    mirror_on_backup_server,
)


def log_bin_widths(freqs):
    """Bandwidth each log-spaced bin stands for (geometric-midpoint edges).

    Mirrors the edge definition of logbin_rows in
    compute_sleep_spectrograms.py, so integrating PSD * width over a band
    recovers the power the linear grid held there.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    inner = np.sqrt(freqs[:-1] * freqs[1:])
    edges = np.concatenate(([freqs[0] ** 2 / inner[0]], inner,
                            [freqs[-1] ** 2 / inner[-1]]))
    return np.diff(edges)


def band_power_from_spec(spec_ch, freqs, widths, low, high):
    """PSD integrated over the rows whose center lies in [low, high] Hz.

    spec_ch : (n_freqs, n_times) linear PSD (density scaling) for one channel.
    Returns the band-power envelope, one value per spectrogram bin. Band edges
    outside the saved grid are silently clipped to it (delta's 0.5 Hz edge).
    """
    sel = (freqs >= low) & (freqs <= high)
    if not sel.any():
        raise ValueError(
            f"band {low}-{high} Hz has no rows in the saved grid "
            f"({freqs[0]:.2f}-{freqs[-1]:.1f} Hz)")
    return (spec_ch[sel] * widths[sel, None]).sum(axis=0)


parser = argparse.ArgumentParser(
    description="Compute per-channel band-power / PC1 features from spectrograms.")
parser.add_argument(
    "--overwrite", action="store_true",
    help="Recompute even when the band-powers pickle already exists "
         "(default: existing outputs are kept and the session is skipped).")
args = parser.parse_args()

# === PROCESS ALL SLEEP SESSIONS / SHANKS ===
print(f"Processing data from: {low_freq_folder}")

sessions_to_run = active_sleep_sessions(sleep_sessions)
if not sessions_to_run:
    print("No active sleep sessions (pre/post both start=end=None) - nothing to do.")

for session_key, session_cfg in sessions_to_run.items():
    session_label = f"{session_name}{session_cfg['suffix']}"

    # One pickle covers every shank for this session, so the skip check reads
    # whatever's already there (if anything) and works out which REQUESTED
    # shanks are actually missing from it - e.g. `shanks` in
    # sleep_pipeline_config.py grew since the last run, so shank 0 is done but
    # shanks 1-7 are not. Previously this was file-level only: if the pickle
    # existed at all, the whole session was skipped even when it was missing
    # shanks the current run asked for, silently leaving them uncomputed.
    output_check = resolve_existing_file(
        low_freq_folder / f'{session_label}_all_shanks_band_powers.pkl')
    all_shanks_data = {}
    if output_check.exists() and not args.overwrite:
        try:
            with open(output_check, 'rb') as f:
                all_shanks_data = pickle.load(f).get('shanks_data', {})
        except (OSError, EOFError, pickle.PickleError) as e:
            print(f"WARNING: could not read existing {output_check.name} ({e}) - "
                  f"recomputing every requested shank for this session")
            all_shanks_data = {}

    shanks_to_process = [s for s in shanks if s not in all_shanks_data]
    if output_check.exists() and not args.overwrite and not shanks_to_process:
        print(f"\n{session_key}: {output_check.name} already has every requested "
              f"shank {shanks} - skipping ({output_check})")
        continue
    if all_shanks_data and shanks_to_process:
        print(f"\n{session_key}: {output_check.name} already has shank(s) "
              f"{sorted(all_shanks_data)} - computing missing shank(s) "
              f"{shanks_to_process} only")

    print(f"\n{'#'*60}")
    print(f"SLEEP SESSION: {session_key}")
    print(f"{'#'*60}")

    for ish in shanks_to_process:
        print(f"\n{'='*60}")
        print(f"PROCESSING SHANK {ish}")
        print(f"{'='*60}")

        # === LOAD DATA ===
        # Spectrograms only (falls back to the backup server if saved there);
        # the LFP traces are not needed - every feature comes from the PSD.
        spectrogram_file = resolve_existing_file(
            low_freq_folder / f'{session_label}_sh{ish}_spectrograms.npz')

        if not spectrogram_file.exists():
            print(f"WARNING: Spectrogram file not found: {spectrogram_file}")
            continue

        print(f"Loading spectrograms from: {spectrogram_file}")
        spec_data = np.load(spectrogram_file)

        # Shape: (n_channels, n_freqs, n_times)
        spectrograms = spec_data['spectrograms']
        freqs = spec_data['freqs'].astype(np.float64)
        times = spec_data['times']
        channel_ids = spec_data['channel_ids']
        # The LFP rate the spectrogram was computed from; kept in the pkl
        # because sample-indexed masks downstream are built against it.
        sampling_rate = float(spec_data['sampling_rate'])
        start_time = spec_data['start_time']

        n_channels, n_freqs, n_times = spectrograms.shape
        print(f"\nLoaded data:")
        print(f"  Spectrograms shape: {spectrograms.shape}")
        print(f"  Frequency axis: {n_freqs} bins, {freqs[0]:.2f}-{freqs[-1]:.1f} Hz")
        print(f"  Time bins: {n_times} ({times[0]:.1f}-{times[-1]:.1f} s)")
        print(f"  Source LFP rate: {sampling_rate:g} Hz")

        widths = log_bin_widths(freqs)
        bands = band_params['bands']
        # (num_low, num_high, den_low, den_high)
        tr_lo, tr_hi, tr_den_lo, tr_den_hi = bands['theta_ratio']

        # === PC1 + BAND POWERS, PER CHANNEL, ALL FROM THE SPECTROGRAM ===
        print(f"\nComputing PC1 and band powers for {n_channels} channels...")

        pc1_channels = np.zeros((n_channels, n_times))
        all_bands_data = {}

        for ch_idx, ch_id in enumerate(channel_ids):
            if (ch_idx + 1) % 5 == 0 or ch_idx == 0:
                print(f"  Channel {ch_id} ({ch_idx + 1}/{n_channels})")

            # (n_freqs, n_times) linear PSD; float64 for the PCA arithmetic
            spec_ch = spectrograms[ch_idx].astype(np.float64)

            # --- PC1: log-transform, z-score each frequency row, PCA over time
            log_spec = np.log10(spec_ch + 1e-10)
            log_spec = (log_spec - log_spec.mean(axis=1, keepdims=True)) / \
                (log_spec.std(axis=1, keepdims=True) + 1e-12)
            pca = PCA(n_components=1)
            pc1_channels[ch_idx] = pca.fit_transform(log_spec.T).flatten()

            # --- band envelopes: PSD integral per bin. The 10 s spectrogram
            # window is the temporal smoothing here; no extra smoothing on top.
            delta = band_power_from_spec(spec_ch, freqs, widths,
                                         bands['delta'][0], bands['delta'][1])
            theta_num = band_power_from_spec(spec_ch, freqs, widths, tr_lo, tr_hi)
            theta_den = band_power_from_spec(spec_ch, freqs, widths,
                                             tr_den_lo, tr_den_hi)
            theta_ratio = theta_num / (theta_den + 1e-10)
            sigma = band_power_from_spec(spec_ch, freqs, widths,
                                         bands['sigma'][0], bands['sigma'][1])
            gamma = band_power_from_spec(spec_ch, freqs, widths,
                                         bands['gamma'][0], bands['gamma'][1])

            all_bands_data[ch_id] = {
                'delta': delta.astype(np.float32),
                'theta_ratio': theta_ratio.astype(np.float32),
                'sigma': sigma.astype(np.float32),
                'gamma': gamma.astype(np.float32),
            }

        # === STORE DATA FOR THIS SHANK ===
        # `band_time` is the same array object as `spectrogram_times`, so
        # pickle stores it once; it is named separately because it is what the
        # envelopes are indexed by, which readers should not have to infer.
        # `sampling_rate` still means the LFP rate the features derive from -
        # anything building a sample-indexed mask reads that, not band_fs.
        # The first spectrogram center sits half a window after the LFP start
        # and the last half a window before its end, so the LFP duration is
        # times[0] + times[-1]; n_lfp_samples/duration_sec are kept only so
        # the pkl schema is unchanged.
        duration_sec = float(times[0]) + float(times[-1])
        shank_data = {
            'channel_ids': channel_ids,
            'sampling_rate': sampling_rate,
            'band_time': times,
            'band_fs': float(1.0 / np.median(np.diff(times))),
            'n_lfp_samples': int(round(duration_sec * sampling_rate)),
            'duration_sec': duration_sec,
            'pc1_spectrogram': pc1_channels.astype(np.float32),
            'start_time': start_time,
            'band_powers': all_bands_data,  # Dictionary of {ch_id: {band: array}}
            # Spectrogram timing information for synchronization
            'spectrogram_times': times,  # Original spectrogram time points
            'spectrogram_freqs': freqs.astype(np.float32),
            # The spectrogram itself stays in <session>_sh<N>_spectrograms.npz;
            # band_powers_io.load_spectrograms() reads it from there.
        }

        all_shanks_data[ish] = shank_data

        print(f"\n✓ Shank {ish} processing complete")
        print(f"  Envelopes + PC1 on the spectrogram grid: {n_times} samples, "
              f"range {times[0]:.2f} - {times[-1]:.2f} s "
              f"({shank_data['band_fs']:.3f} Hz)")

    # === SAVE ALL DATA TO PICKLE (this sleep session) ===
    out_dir = resolve_output_folder(low_freq_folder)
    output_file = out_dir / f'{session_label}_all_shanks_band_powers.pkl'
    print(f"\n{'='*60}")
    print(f"Saving all shanks data to: {output_file}")
    print(f"{'='*60}")

    save_data = {
        'session_name': session_name,
        'sleep_session': session_key,
        # The shanks actually IN the file (old + newly computed), not just
        # this run's request - accurate even after a partial/merged re-run.
        'shanks': sorted(all_shanks_data),
        'band_params': band_params,
        'rec_folder': str(rec_folder),
        # 'compact' = envelopes on the spectrogram grid, no spectrogram copy.
        # Files without this key are the legacy 500 Hz layout; band_powers_io
        # reads either.
        'layout': 'compact',
        'shanks_data': all_shanks_data,
    }

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        backup_dir = mirror_on_backup_server(out_dir)
        if backup_dir is None:
            raise
        backup_dir.mkdir(parents=True, exist_ok=True)
        output_file = backup_dir / output_file.name
        print(f"Out of space while saving - retrying on backup server: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)

    print(f"\n✓ All data saved successfully!")
    print(f"\nSummary ({session_key}):")
    for ish in shanks:
        if ish in all_shanks_data:
            n_channels = len(all_shanks_data[ish]['channel_ids'])
            duration = all_shanks_data[ish]['duration_sec']
            print(f"  Shank {ish}: {n_channels} channels, {duration:.1f} s")
        else:
            print(f"  Shank {ish}: NOT PROCESSED")

print(f"\n{'='*60}")
print("HOW TO LOAD THE DATA")
print(f"{'='*60}")
print("""
import pickle
import numpy as np
from band_powers_io import band_time, band_fs, load_spectrograms, epoch_average

# Load all data (pick the pre/post file you want, e.g. "..._post_all_shanks_band_powers.pkl")
path = '..._all_shanks_band_powers.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

# Access data for a specific shank
shank_id = 0
shank_data = data['shanks_data'][shank_id]

# Get timing arrays. band_time() is the grid the envelopes (band powers, PC1)
# are stored on - the spectrogram grid, 1 s bins. Use it instead of reading a
# key directly and it works for files written before the compact layout too.
t = band_time(shank_data)                       # seconds from the sleep epoch start
start_time = shank_data['start_time']           # Absolute start time
fs_lfp = shank_data['sampling_rate']            # LFP rate the features derive from

# Get spectrograms (read from the sibling npz; not duplicated in this pickle)
spectrograms = load_spectrograms(shank_data, path, shank_id)   # (n_ch, n_freqs, n_times)
freqs = shank_data['spectrogram_freqs']

# Get PC1, on the same grid as the band powers
pc1 = shank_data['pc1_spectrogram']             # (n_channels, len(t))

# Get band powers for a specific channel
channel_ids = shank_data['channel_ids']
ch_id = channel_ids[0]
delta = shank_data['band_powers'][ch_id]['delta']
theta_ratio = shank_data['band_powers'][ch_id]['theta_ratio']
sigma = shank_data['band_powers'][ch_id]['sigma']
gamma = shank_data['band_powers'][ch_id]['gamma']

# Reduce to fixed epochs (exact at any sample rate, unlike reshaping by count)
delta_4s = epoch_average(delta, t, 4.0)

# SYNCHRONIZE WITH VELOCITY DATA
# If you have velocity with timestamps, interpolate onto the same grid:
velocity_time = np.array([...])  # Your velocity timestamps
velocity_data = np.array([...])  # Your velocity values
velocity_interp = np.interp(t, velocity_time, velocity_data)

# Or use start_time to align with absolute timestamps
absolute_time = start_time + t
""")
