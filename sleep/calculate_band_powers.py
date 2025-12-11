from pathlib import Path
import numpy as np
import pickle
from scipy import signal
from sklearn.decomposition import PCA

# === CONFIGURATION ===
rec_folder = r"D:\cl\ephys\sleep\CnL42SG_20251112_170949.rec"
session_name = Path(rec_folder).stem.split('.')[0]
shanks = [0, 1, 2, 3, 4, 5, 6, 7]  # Loop through multiple shanks

# === BAND DEFINITIONS ===
band_params = {
    'bands': {
        'delta': (0.5, 4),
        'theta': (5, 10),
        'sigma': (9, 25),
        'gamma': (40, 100),
        # (num_low, num_high, den_low, den_high)
        'theta_ratio': (5, 10, 2, 15),
    },
    'smoothing_window': 10,  # seconds for band power smoothing
}

# === DEFINE FILTERING FUNCTIONS ===


def bandpass_filter_trace(trace, fs, low, high):
    """Apply bandpass filter to trace using SOS for numerical stability"""
    nyq = fs / 2
    sos = signal.butter(4, [low/nyq, high/nyq], btype='band', output='sos')
    return signal.sosfiltfilt(sos, trace)


def compute_band_power(trace, fs, low, high, window_sec=10):
    """Compute smoothed band power"""
    filtered = bandpass_filter_trace(trace, fs, low, high)
    power = filtered ** 2
    # Smooth with moving average
    window_samples = int(window_sec * fs)
    kernel = np.ones(window_samples) / window_samples
    smoothed = np.convolve(power, kernel, mode='same')
    return smoothed


# === PROCESS ALL SHANKS ===
low_freq_folder = Path(rec_folder) / "low_freq"
print(f"Processing data from: {low_freq_folder}")

all_shanks_data = {}

for ish in shanks:
    print(f"\n{'='*60}")
    print(f"PROCESSING SHANK {ish}")
    print(f"{'='*60}")

    # === LOAD DATA ===
    # Load spectrograms
    spectrogram_file = low_freq_folder / \
        f'{session_name}_sh{ish}_spectrograms.npz'

    if not spectrogram_file.exists():
        print(f"WARNING: Spectrogram file not found: {spectrogram_file}")
        continue

    print(f"Loading spectrograms from: {spectrogram_file}")
    spec_data = np.load(spectrogram_file)
    # Load LFP traces
    lfp_file = low_freq_folder / f'{session_name}_sh{ish}_lfp_traces.npz'
    if not lfp_file.exists():
        print(f"WARNING: LFP file not found: {lfp_file}")
        continue

    print(f"Loading LFP traces from: {lfp_file}")
    lfp_data = np.load(lfp_file)
    lfp_traces = lfp_data['traces']  # Shape: (n_samples, n_channels)

    # Shape: (n_channels, n_freqs, n_times)
    spectrograms = spec_data['spectrograms']
    freqs = spec_data['freqs']
    times = spec_data['times']
    channel_ids = spec_data['channel_ids']
    sampling_rate_spec = float(spec_data['sampling_rate'])
    sampling_rate_lfp = float(lfp_data['sampling_rate'])
    assert np.isclose(sampling_rate_spec, sampling_rate_lfp), "FS mismatch!"
    sampling_rate = sampling_rate_lfp

    start_time = spec_data['start_time']

    print(f"\nLoaded data:")
    print(f"  Spectrograms shape: {spectrograms.shape}")
    print(f"  LFP traces shape: {lfp_traces.shape}")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Total duration: {lfp_traces.shape[0] / sampling_rate:.1f} s")
    print(f"  Number of channels: {len(channel_ids)}")


    # Reshape for PCA: (n_times, n_channels * n_freqs)# === COMPUTE PC1 FOR EACH CHANNEL INDEPENDENTLY ===
    print("\nComputing PC1 (per-channel)...")

    n_channels, n_freqs, n_times = spectrograms.shape
    pc1_channels = np.zeros((n_channels, n_times))

    for ch in range(n_channels):

        if (ch + 1) % 5 == 0 or ch == 0:
            print(f"  Channel {channel_ids[ch]} ({ch+1}/{n_channels})")

        # Extract spectrogram for ONE channel: shape (n_freqs, n_times)
        spec_ch = spectrograms[ch]

        # Log transform for stability
        log_spec = np.log10(spec_ch + 1e-10)

        # Standardize features before PCA (important!)
        log_spec = (log_spec - log_spec.mean(axis=1, keepdims=True)) / \
                (log_spec.std(axis=1, keepdims=True) + 1e-12)

        # PCA input must be (n_times, n_freqs)
        X = log_spec.T

        # PCA → PC1 time course
        pca = PCA(n_components=1)
        pc1_ch = pca.fit_transform(X).flatten()

        pc1_channels[ch] = pc1_ch

        # Optional debug
        # print(f"    PC1 var explained={pca.explained_variance_ratio_[0]*100:.2f}%")

    # === INTERPOLATE EACH PC1 TO MATCH LFP SAMPLING RATE ===
    print("\nInterpolating each PC1 to LFP time base...")

    lfp_time = np.arange(lfp_traces.shape[0]) / sampling_rate
    pc1_interp_channels = np.zeros((n_channels, len(lfp_time)))

    for ch in range(n_channels):
        pc1_interp_channels[ch] = np.interp(lfp_time, times, pc1_channels[ch])


    # === COMPUTE BAND POWERS FOR ALL CHANNELS ===
    print(f"\nComputing band powers for {len(channel_ids)} channels...")

    # Storage for all channels' band powers
    all_bands_data = {}

    for ch_idx, ch_id in enumerate(channel_ids):
        if (ch_idx + 1) % 5 == 0 or ch_idx == 0:
            print(
                f"  Processing channel {ch_id} ({ch_idx + 1}/{len(channel_ids)})...")

        analysis_trace = lfp_traces[:, ch_idx]

        # Compute band powers for this channel
        bands_data = {}

        bands_data['delta'] = compute_band_power(
            analysis_trace, sampling_rate,
            band_params['bands']['delta'][0],
            band_params['bands']['delta'][1],
            band_params['smoothing_window']
        )

        bands_data['theta_ratio_num'] = compute_band_power(
            analysis_trace, sampling_rate,
            band_params['bands']['theta_ratio'][0],
            band_params['bands']['theta_ratio'][1],
            band_params['smoothing_window']
        )

        bands_data['theta_ratio_den'] = compute_band_power(
            analysis_trace, sampling_rate,
            band_params['bands']['theta_ratio'][2],
            band_params['bands']['theta_ratio'][3],
            band_params['smoothing_window']
        )

        bands_data['theta_ratio'] = bands_data['theta_ratio_num'] / \
            (bands_data['theta_ratio_den'] + 1e-10)

        bands_data['sigma'] = compute_band_power(
            analysis_trace, sampling_rate,
            band_params['bands']['sigma'][0],
            band_params['bands']['sigma'][1],
            band_params['smoothing_window']
        )

        bands_data['gamma'] = compute_band_power(
            analysis_trace, sampling_rate,
            band_params['bands']['gamma'][0],
            band_params['bands']['gamma'][1],
            band_params['smoothing_window']
        )

        # Store for this channel
        all_bands_data[ch_id] = bands_data

    # === STORE DATA FOR THIS SHANK ===
    shank_data = {
        'channel_ids': channel_ids,
        'sampling_rate': sampling_rate,
        'lfp_time': lfp_time,
        'pc1_spectrogram': pc1_interp_channels,
        'start_time': start_time,
        'band_powers': all_bands_data,  # Dictionary of {ch_id: {band: array}}
        # Spectrogram timing information for synchronization
        'spectrogram_times': times,  # Original spectrogram time points
        'spectrogram_freqs': freqs,  # Frequency array for spectrograms
        # Full spectrograms (n_channels, n_freqs, n_times)
        'spectrograms': spectrograms,
    }

    all_shanks_data[ish] = shank_data

    print(f"\n✓ Shank {ish} processing complete")
    print(
        f"  LFP time: {len(lfp_time)} samples, range {lfp_time[0]:.2f} - {lfp_time[-1]:.2f} s")
    print(
        f"  Spectrogram time: {len(times)} samples, range {times[0]:.2f} - {times[-1]:.2f} s")

# === SAVE ALL DATA TO PICKLE ===
output_file = low_freq_folder / f'{session_name}_all_shanks_band_powers.pkl'
print(f"\n{'='*60}")
print(f"Saving all shanks data to: {output_file}")
print(f"{'='*60}")

save_data = {
    'session_name': session_name,
    'shanks': shanks,
    'band_params': band_params,
    'rec_folder': str(rec_folder),
    'shanks_data': all_shanks_data,
}

with open(output_file, 'wb') as f:
    pickle.dump(save_data, f)

print(f"\n✓ All data saved successfully!")
print(f"\nSummary:")
for ish in shanks:
    if ish in all_shanks_data:
        n_channels = len(all_shanks_data[ish]['channel_ids'])
        duration = all_shanks_data[ish]['lfp_time'][-1]
        print(f"  Shank {ish}: {n_channels} channels, {duration:.1f} s")
    else:
        print(f"  Shank {ish}: NOT PROCESSED")

print(f"\n{'='*60}")
print("HOW TO LOAD THE DATA")
print(f"{'='*60}")
print("""
import pickle
import numpy as np

# Load all data
with open('..._all_shanks_band_powers.pkl', 'rb') as f:
    data = pickle.load(f)

# Access data for a specific shank
shank_id = 0
shank_data = data['shanks_data'][shank_id]

# Get timing arrays
lfp_time = shank_data['lfp_time']              # High-resolution time for LFP/bands
spectrogram_times = shank_data['spectrogram_times']  # Lower-resolution time for spectrograms
start_time = shank_data['start_time']           # Absolute start time

# Get spectrograms
spectrograms = shank_data['spectrograms']       # (n_channels, n_freqs, n_times)
freqs = shank_data['spectrogram_freqs']         # Frequency array

# Get PC1 (already interpolated to LFP time)
pc1 = shank_data['pc1_spectrogram']

# Get band powers for a specific channel
channel_ids = shank_data['channel_ids']
ch_id = channel_ids[0]
delta = shank_data['band_powers'][ch_id]['delta']
theta_ratio = shank_data['band_powers'][ch_id]['theta_ratio']
sigma = shank_data['band_powers'][ch_id]['sigma']
gamma = shank_data['band_powers'][ch_id]['gamma']

# SYNCHRONIZE WITH VELOCITY DATA
# If you have velocity with timestamps, interpolate to common time base:
velocity_time = np.array([...])  # Your velocity timestamps
velocity_data = np.array([...])  # Your velocity values

# Option 1: Interpolate velocity to LFP time (for band powers)
velocity_interp = np.interp(lfp_time, velocity_time, velocity_data)

# Option 2: Find nearest spectrogram times for velocity
# (if you want to correlate velocity with raw spectrograms)
for i, t in enumerate(spectrogram_times):
    idx = np.argmin(np.abs(velocity_time - t))
    vel_at_spec_time = velocity_data[idx]
    
# Option 3: Use start_time to align with absolute timestamps
absolute_lfp_time = start_time + lfp_time
absolute_spec_time = start_time + spectrogram_times
""")
