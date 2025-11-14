from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA

rec_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\sleep\CnL39SG\CnL39SG_20251102_210043.rec"
session_name = Path(rec_folder).stem.split('.')[0]
ish = 0

# === PLOTTING PARAMETERS ===
plot_params = {
    # Window size for each figure
    'window_size': 3000,  # seconds
    
    # Color scale options: 'adaptive', 'percentile', 'manual'
    'color_scale_method': 'adaptive',  # 'adaptive', 'percentile', or 'manual'
    
    # For 'adaptive' method (median ± N * MAD)
    'adaptive_n_mad': 3,
    
    # For 'percentile' method
    'vmin_percentile': 2,
    'vmax_percentile': 98,
    
    # For 'manual' method
    'vmin_manual': -40,
    'vmax_manual': 20,
    
    # Frequency display range for spectrogram
    'freq_min': 0.5,
    'freq_max': 100,
    
    # Band definitions (in Hz)
    'bands': {
        'delta': (0.5, 4),
        'theta': (5, 10),
        'sigma': (9, 25),
        'gamma': (40, 100),
        'theta_ratio': (5, 10, 2, 15),  # (num_low, num_high, den_low, den_high)
    },
    
    # Y-axis limits for normalized band power plots (in standard deviations)
    'band_ylim': (-4, 4),  # Show ±4 standard deviations
    
    # Colormap
    'cmap': 'jet',
    
    # Figure size
    'figsize': (16, 10),
    
    # DPI
    'dpi': 300,
}

# Load data
low_freq_folder = Path(rec_folder) / "low_freq"

print(f"Loading data from: {low_freq_folder}")

# Load spectrograms
spectrogram_file = low_freq_folder / f'{session_name}_sh{ish}_spectrograms.npz'
print(f"Loading spectrograms from: {spectrogram_file}")
spec_data = np.load(spectrogram_file)

spectrograms = spec_data['spectrograms']  # Shape: (n_channels, n_freqs, n_times)
freqs = spec_data['freqs']
times = spec_data['times']
channel_ids = spec_data['channel_ids']
sampling_rate = spec_data['sampling_rate']
start_time = spec_data['start_time']

# Load LFP traces
lfp_file = low_freq_folder / f'{session_name}_sh{ish}_lfp_traces.npz'
print(f"Loading LFP traces from: {lfp_file}")
lfp_data = np.load(lfp_file)
lfp_traces = lfp_data['traces']  # Shape: (n_samples, n_channels)

print(f"\nLoaded data:")
print(f"  Spectrograms shape: {spectrograms.shape}")
print(f"  LFP traces shape: {lfp_traces.shape}")
print(f"  Sampling rate: {sampling_rate} Hz")
print(f"  Total duration: {lfp_traces.shape[0] / sampling_rate:.1f} s")
print(f"  Number of channels: {len(channel_ids)}")

# Compute PCA on spectrograms to get PC1
print("\nComputing PCA on spectrograms...")
n_channels, n_freqs, n_times = spectrograms.shape

# Reshape for PCA: (n_times, n_channels * n_freqs)
spec_for_pca = spectrograms.reshape(n_channels * n_freqs, n_times).T

# Apply PCA
pca = PCA(n_components=1)
pc1_spectrogram = pca.fit_transform(spec_for_pca).flatten()

print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance")

# Interpolate PC1 to match LFP sampling rate
lfp_time_full = np.arange(lfp_traces.shape[0]) / sampling_rate
pc1_interp = np.interp(lfp_time_full, times, pc1_spectrogram)

# Determine color scale for spectrogram
all_values = spectrograms.flatten()
if plot_params['color_scale_method'] == 'adaptive':
    median_val = np.median(all_values)
    mad = np.median(np.abs(all_values - median_val))
    vmin = median_val - plot_params['adaptive_n_mad'] * mad
    vmax = median_val + plot_params['adaptive_n_mad'] * mad
elif plot_params['color_scale_method'] == 'percentile':
    vmin = np.percentile(all_values, plot_params['vmin_percentile'])
    vmax = np.percentile(all_values, plot_params['vmax_percentile'])
else:  # manual
    vmin = plot_params['vmin_manual']
    vmax = plot_params['vmax_manual']

print(f"\nColor scale: vmin={vmin:.2f}, vmax={vmax:.2f} dB")

# Compute band powers
def bandpass_filter_trace(trace, fs, low, high):
    """Apply bandpass filter to trace using SOS for numerical stability"""
    nyq = fs / 2
    # Use second-order sections (SOS) for better numerical stability
    # especially important for very low frequencies like 0.5 Hz
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

# Create output folder
output_folder = low_freq_folder / "spectrogram"
output_folder.mkdir(exist_ok=True)
print(f"\nSaving plots to: {output_folder}")

# Time arrays
spec_time = times

# Determine number of windows
total_duration = lfp_traces.shape[0] / sampling_rate
n_windows = int(np.ceil(total_duration / plot_params['window_size']))

print(f"\nProcessing {len(channel_ids)} channels with {n_windows} windows each ({plot_params['window_size']}s per window)...")

# Loop through each channel
for ch_idx, ch_id in enumerate(channel_ids):
    print(f"\n=== Processing Channel {ch_id} ({ch_idx + 1}/{len(channel_ids)}) ===")
    
    # Get data for this channel
    analysis_trace = lfp_traces[:, ch_idx]
    channel_spectrogram = spectrograms[ch_idx, :, :]
    
    # Compute band powers for this channel
    print(f"  Computing band powers for channel {ch_id}...")
    bands_data = {}
    bands_data['delta'] = compute_band_power(analysis_trace, sampling_rate, 
                                             plot_params['bands']['delta'][0], 
                                             plot_params['bands']['delta'][1])

    bands_data['theta_ratio_num'] = compute_band_power(analysis_trace, sampling_rate,
                                                        plot_params['bands']['theta_ratio'][0],
                                                        plot_params['bands']['theta_ratio'][1])

    bands_data['theta_ratio_den'] = compute_band_power(analysis_trace, sampling_rate,
                                                        plot_params['bands']['theta_ratio'][2],
                                                        plot_params['bands']['theta_ratio'][3])

    bands_data['theta_ratio'] = bands_data['theta_ratio_num'] / (bands_data['theta_ratio_den'] + 1e-10)

    bands_data['sigma'] = compute_band_power(analysis_trace, sampling_rate,
                                             plot_params['bands']['sigma'][0],
                                             plot_params['bands']['sigma'][1])

    bands_data['gamma'] = compute_band_power(analysis_trace, sampling_rate,
                                            plot_params['bands']['gamma'][0],
                                            plot_params['bands']['gamma'][1])
    
    # Time array for LFP
    lfp_time = np.arange(len(analysis_trace)) / sampling_rate
    
    # Create plots for each window
    for win_idx in range(n_windows):
        print(f"  Plotting window {win_idx + 1}/{n_windows}...")
        
        # Time range for this window
        t_start = win_idx * plot_params['window_size']
        t_end = min((win_idx + 1) * plot_params['window_size'], total_duration)
        
        # Create figure with subplots
        fig = plt.figure(figsize=plot_params['figsize'], constrained_layout=True)
        gs = fig.add_gridspec(6, 1, height_ratios=[2, 1, 1, 1, 1, 1], hspace=0.3)
        
        # 1. Spectrogram
        ax1 = fig.add_subplot(gs[0])
        
        # Find time indices for spectrogram
        spec_mask = (spec_time >= t_start) & (spec_time <= t_end)
        
        im = ax1.pcolormesh(spec_time[spec_mask], freqs, channel_spectrogram[:, spec_mask],
                           shading='gouraud', cmap=plot_params['cmap'],
                           vmin=vmin, vmax=vmax)
        
        ax1.set_ylabel('Frequency (Hz)', fontsize=10)
        ax1.set_ylim([plot_params['freq_min'], plot_params['freq_max']])
        ax1.set_yscale('log')
        ax1.set_yticks([1, 4, 16, 64])
        ax1.set_yticklabels(['1', '4', '16', '64'])
        ax1.set_xlim([t_start, t_end])
        ax1.set_title(f'Spectrogram - Ch{ch_id} (Shank {ish})', fontsize=12)
        ax1.set_xticklabels([])
        
        cbar = plt.colorbar(im, ax=ax1, label='Power (dB)')
        
        # 2. PC1 / Spectrogram trace (from PCA)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        lfp_mask = (lfp_time >= t_start) & (lfp_time <= t_end)
        
        # Use the actual PC1 from spectrogram PCA
        pc1_window = pc1_interp[lfp_mask]
        pc1_norm = (pc1_window - np.mean(pc1_window)) / (np.std(pc1_window) + 1e-10)
        
        ax2.plot(lfp_time[lfp_mask], pc1_norm, 'k-', linewidth=0.5)
        ax2.set_ylabel('PC1\nSpectrogram', fontsize=9)
        ax2.set_xlim([t_start, t_end])
        ax2.set_ylim(plot_params['band_ylim'])
        ax2.set_xticklabels([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Theta ratio
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        theta_ratio_norm = bands_data['theta_ratio'][lfp_mask]
        theta_ratio_norm = (theta_ratio_norm - np.mean(theta_ratio_norm)) / (np.std(theta_ratio_norm) + 1e-10)
        ax3.plot(lfp_time[lfp_mask], theta_ratio_norm, 'k-', linewidth=0.5)
        ax3.set_ylabel('Theta ratio\n5-10Hz/2-15Hz', fontsize=9)
        ax3.set_xlim([t_start, t_end])
        ax3.set_ylim(plot_params['band_ylim'])
        ax3.set_xticklabels([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Delta (0.5-4 Hz)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        delta_norm = bands_data['delta'][lfp_mask]
        delta_norm = (delta_norm - np.mean(delta_norm)) / (np.std(delta_norm) + 1e-10)
        ax4.plot(lfp_time[lfp_mask], delta_norm, 'k-', linewidth=0.5)
        ax4.set_ylabel('0.5-4 Hz\n(Delta)', fontsize=9)
        ax4.set_xlim([t_start, t_end])
        ax4.set_ylim(plot_params['band_ylim'])
        ax4.set_xticklabels([])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Sigma (9-25 Hz)
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        sigma_norm = bands_data['sigma'][lfp_mask]
        sigma_norm = (sigma_norm - np.mean(sigma_norm)) / (np.std(sigma_norm) + 1e-10)
        ax5.plot(lfp_time[lfp_mask], sigma_norm, 'k-', linewidth=0.5)
        ax5.set_ylabel('9-25Hz\n(Sigma)', fontsize=9)
        ax5.set_xlim([t_start, t_end])
        ax5.set_ylim(plot_params['band_ylim'])
        ax5.set_xticklabels([])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # 6. Gamma (40-100 Hz)
        ax6 = fig.add_subplot(gs[5], sharex=ax1)
        gamma_norm = bands_data['gamma'][lfp_mask]
        gamma_norm = (gamma_norm - np.mean(gamma_norm)) / (np.std(gamma_norm) + 1e-10)
        ax6.plot(lfp_time[lfp_mask], gamma_norm, 'k-', linewidth=0.5)
        ax6.set_ylabel('40-100Hz\n(Gamma)', fontsize=9)
        ax6.set_xlabel('Time (s)', fontsize=10)
        ax6.set_xlim([t_start, t_end])
        ax6.set_ylim(plot_params['band_ylim'])
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        
        # Add scale bar
        ax6.text(0.98, 0.95, '500s', transform=ax6.transAxes,
                ha='right', va='top', fontsize=9)
        
        # Save figure with channel ID in filename
        output_file = output_folder / f'{session_name}_sh{ish}_ch{ch_id:03d}_summary_{t_start:06d}s-{int(t_end):06d}s.png'
        plt.savefig(output_file, dpi=plot_params['dpi'], bbox_inches='tight')
        plt.close()

print(f"\n=== All channel plots saved successfully! ===")
print(f"Output directory: {output_folder}")
print(f"Total files created: {len(channel_ids) * n_windows}")