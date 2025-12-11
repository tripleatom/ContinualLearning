from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

# === CONFIGURATION ===
rec_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\sleep\CnL39SG\CnL39SG_20251102_210043.rec"
rec_folder = Path(rec_folder)  # Convert to Path object
session_name = rec_folder.stem.split('.')[0]
shanks = [0,1,2,3]
fs = 30000  # Original sampling rate

# === PLOTTING PARAMETERS ===
plot_params = {
    # Color scale options: 'adaptive', 'percentile', 'manual'
    'color_scale_method': 'percentile',  # Use percentile for better range
    
    # For 'adaptive' method (median ± N * MAD)
    'adaptive_n_mad': 3,
    
    # For 'percentile' method
    'vmin_percentile': 0,   # Adjusted to show more blue
    'vmax_percentile': 95,  # Adjusted to show more dynamic range
    
    # For 'manual' method
    'vmin_manual': -3,
    'vmax_manual': 3,
    
    # Color scale extension (as fraction of range)
    'vmin_extension': 0.2,  # Extend minimum by 20% of range
    'vmax_extension': 0.2,  # Extend maximum by 20% of range
    
    # Frequency display range for spectrogram
    'freq_min': 0.5,
    'freq_max': 100,
    
    # Y-axis limits for normalized band power plots (in standard deviations)
    'band_ylim': (-4, 4),
    
    # Colormap
    'cmap': 'jet',
    
    # Figure size (much wider for full recording)
    'figsize': (30, 12),
    
    # DPI
    'dpi': 150,  # Lower DPI for large figures
}

# === LOAD SYNCHRONIZATION AND VELOCITY DATA ===
print("Loading velocity and synchronization data...")

# Velocity file
velocity_file = rec_folder.parent / "velocity_advanced.pkl"
if velocity_file.exists():  
    with open(velocity_file, 'rb') as f:
        velocity_data = pickle.load(f)
    velocity_time_raw = velocity_data['time_stamp']
    velocity_raw = velocity_data['velocity']
    print(f"✓ Loaded velocity data: {len(velocity_raw)} samples")
else:
    velocity_time_raw = None
    velocity_raw = None
    print("⚠ Velocity file not found")

# Sync times file
sync_times_file = rec_folder.parent / "sync_times.pkl"
if sync_times_file.exists():
    with open(sync_times_file, 'rb') as f:
        sync_times = pickle.load(f)
    proc_rising_time = sync_times['proc_rising_time']
    SG_rising_time = sync_times['SG_rising_time'] / fs
    print(f"✓ Loaded sync times")
    print(f"  Proc rising time range: {proc_rising_time[0]:.2f} - {proc_rising_time[-1]:.2f} s")
    print(f"  SG rising time range: {SG_rising_time[0]:.2f} - {SG_rising_time[-1]:.2f} s")
else:
    proc_rising_time = None
    SG_rising_time = None
    print("⚠ Sync times file not found")

# === SYNCHRONIZE VELOCITY WITH SPECTROGRAM ===
velocity_synced = None
velocity_time_synced = None

if velocity_time_raw is not None and proc_rising_time is not None and SG_rising_time is not None:
    print("\nSynchronizing velocity with spectrogram...")
    
    # Pick up velocity in the range of proc_rising_time[0:-1]
    vel_mask = (velocity_time_raw >= proc_rising_time[0]) & (velocity_time_raw <= proc_rising_time[-1])
    velocity_synced = velocity_raw[vel_mask]
    velocity_time_synced = velocity_time_raw[vel_mask]
    
    # Remap velocity time to match spectrogram time (SG_rising_time[0] to SG_rising_time[-1])
    # Linear mapping from proc_rising_time to SG_rising_time
    velocity_time_synced = np.interp(velocity_time_synced, 
                                     [proc_rising_time[0], proc_rising_time[-1]],
                                     [SG_rising_time[0], SG_rising_time[-1]])
    
    print(f"✓ Synchronized velocity")
    print(f"  Velocity samples: {len(velocity_synced)}")
    print(f"  Velocity time range: {velocity_time_synced[0]:.2f} - {velocity_time_synced[-1]:.2f} s")
    print(f"  Spectrogram time range: {SG_rising_time[0]:.2f} - {SG_rising_time[-1]:.2f} s")
    
    plot_velocity = True
else:
    plot_velocity = False
    print("\n⚠ Cannot synchronize velocity - missing data or sync times")

# === LOAD DATA ===
low_freq_folder = rec_folder / "low_freq"
print(f"\nLoading data from: {low_freq_folder}")

# Load computed band powers and spectrograms (all in one pickle file)
band_powers_file = low_freq_folder / f'{session_name}_all_shanks_band_powers.pkl'
print(f"Loading data from: {band_powers_file}")

with open(band_powers_file, 'rb') as f:
    all_data = pickle.load(f)

# === CREATE OUTPUT FOLDER ===
output_folder = low_freq_folder / "spectrogram"
output_folder.mkdir(exist_ok=True)
print(f"\nSaving plots to: {output_folder}")

# === LOOP THROUGH ALL SHANKS ===
total_files_created = 0
for shank_id in shanks:
    print(f"\n{'='*60}")
    print(f"PROCESSING SHANK {shank_id}")
    print(f"{'='*60}")
    
    # Check if shank data exists
    if shank_id not in all_data['shanks_data']:
        print(f"⚠ Shank {shank_id} not found in data, skipping...")
        continue
    
    shank_data = all_data['shanks_data'][shank_id]

    # Extract all needed data
    lfp_time = shank_data['lfp_time']
    pc1_spectrogram = shank_data['pc1_spectrogram']
    channel_ids = shank_data['channel_ids']
    times = shank_data['spectrogram_times']
    freqs = shank_data['spectrogram_freqs']
    spectrograms = shank_data['spectrograms']  # (n_channels, n_freqs, n_times)
    sampling_rate = shank_data['sampling_rate']

    print(f"\nLoaded data for Shank {shank_id}:")
    print(f"  Spectrograms shape: {spectrograms.shape}")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Total duration: {lfp_time[-1]:.1f} s")
    print(f"  Number of channels: {len(channel_ids)}")

    # === CROP DATA TO SYNCHRONIZED TIME RANGE ===
    if plot_velocity and SG_rising_time is not None:
        print(f"\nCropping data to synchronized time range...")
        
        # Crop spectrogram and times to SG_rising_time range
        spec_mask = (times >= SG_rising_time[0]) & (times <= SG_rising_time[-1])
        times_cropped = times[spec_mask]
        spectrograms_cropped = spectrograms[:, :, spec_mask]
        
        # Crop LFP-based data (band powers, PC1) to same range
        lfp_mask = (lfp_time >= SG_rising_time[0]) & (lfp_time <= SG_rising_time[-1])
        lfp_time_cropped = lfp_time[lfp_mask]
        pc1_cropped = pc1_spectrogram[:,lfp_mask]
        
        print(f"  Cropped spectrogram time: {times_cropped[0]:.2f} - {times_cropped[-1]:.2f} s")
        print(f"  Cropped LFP time: {lfp_time_cropped[0]:.2f} - {lfp_time_cropped[-1]:.2f} s")
        print(f"  Velocity time: {velocity_time_synced[0]:.2f} - {velocity_time_synced[-1]:.2f} s")
    else:
        # Use full data if no synchronization
        times_cropped = times
        spectrograms_cropped = spectrograms
        lfp_time_cropped = lfp_time
        pc1_cropped = pc1_spectrogram

        # === COLOR SCALE WILL BE DETERMINED PER CHANNEL AFTER Z-SCORING ===

    # === PLOTTING ===
    total_duration = lfp_time_cropped[-1] - lfp_time_cropped[0]
    print(f"\nProcessing {len(channel_ids)} channels, full recording ({total_duration:.1f}s each)...")

    # Time range for full recording
    t_start = lfp_time_cropped[0]
    t_end = lfp_time_cropped[-1]

    # Determine subplot layout
    if plot_velocity:
        n_subplots = 7
        height_ratios = [2, 1, 1, 1, 1, 1, 1]
    else:
        n_subplots = 6
        height_ratios = [2, 1, 1, 1, 1, 1]

    # Loop through each channel
    for ch_idx, ch_id in enumerate(channel_ids):
        print(f"\n=== Processing Channel {ch_id} ({ch_idx + 1}/{len(channel_ids)}) ===")
        
        # Get data for this channel (already cropped)
        channel_spectrogram = spectrograms_cropped[ch_idx, :, :]
        
        # Z-score the spectrogram
        channel_spectrogram_mean = np.mean(channel_spectrogram)
        channel_spectrogram_std = np.std(channel_spectrogram)
        channel_spectrogram_zscored = (channel_spectrogram - channel_spectrogram_mean) / (channel_spectrogram_std + 1e-10)
        
        # Print z-scored data range
        print(f"  Z-scored spectrogram range: [{np.min(channel_spectrogram_zscored):.3f}, {np.max(channel_spectrogram_zscored):.3f}]")
        
        
        # Determine color scale for z-scored spectrogram
        zscored_values = channel_spectrogram_zscored.flatten()

        if plot_params['color_scale_method'] == 'adaptive':
            median_val = np.median(zscored_values)
            mad = np.median(np.abs(zscored_values - median_val))
            vmin_base = median_val - plot_params['adaptive_n_mad'] * mad
            vmax_base = median_val + plot_params['adaptive_n_mad'] * mad
            
            # Extend by configured percentages
            range_val = vmax_base - vmin_base
            vmin = vmin_base - plot_params['vmin_extension'] * range_val
            vmax = vmax_base + plot_params['vmax_extension'] * range_val

        elif plot_params['color_scale_method'] == 'percentile':
            vmin_base = np.percentile(zscored_values, plot_params['vmin_percentile'])
            vmax_base = np.percentile(zscored_values, plot_params['vmax_percentile'])
            
            # Extend by configured percentages
            range_val = vmax_base - vmin_base
            vmin = vmin_base - plot_params['vmin_extension'] * range_val
            vmax = vmax_base + plot_params['vmax_extension'] * range_val

        else:  # manual
            vmin = plot_params['vmin_manual']
            vmax = plot_params['vmax_manual']

        print(f"  Color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")
        
        # Load band powers for this channel from the nested dictionary
        bands_data_full = {
            'delta': shank_data['band_powers'][ch_id]['delta'],
            'theta_ratio': shank_data['band_powers'][ch_id]['theta_ratio'],
            'sigma': shank_data['band_powers'][ch_id]['sigma'],
            'gamma': shank_data['band_powers'][ch_id]['gamma'],
        }
        
        # Crop band powers to synchronized time range
        bands_data = {}
        for band_name, band_values in bands_data_full.items():
            bands_data[band_name] = band_values[lfp_mask] if plot_velocity else band_values
        
        # Create figure
        print(f"  Creating full recording plot...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=plot_params['figsize'], constrained_layout=True)
        gs = fig.add_gridspec(n_subplots, 1, height_ratios=height_ratios, hspace=0.3)
        
        subplot_idx = 0
        
        # 1. Spectrogram
        ax1 = fig.add_subplot(gs[subplot_idx])
        subplot_idx += 1
        
        im = ax1.pcolormesh(
            times_cropped,
            freqs,
            channel_spectrogram_zscored,
            shading='gouraud',
            cmap=plot_params['cmap'],
            vmin=vmin,
            vmax=vmax
        )

        ax1.set_ylabel('Frequency (Hz)', fontsize=10)
        ax1.set_ylim([plot_params['freq_min'], plot_params['freq_max']])
        ax1.set_yscale('log')
        ax1.set_yticks([1, 4, 16, 64])
        ax1.set_yticklabels(['1', '4', '16', '64'])
        ax1.set_xlim([t_start, t_end])
        ax1.set_title(f'Spectrogram (Z-scored) - Ch{ch_id} (Shank {shank_id})', fontsize=12)
        ax1.set_xticklabels([])

        cbar = plt.colorbar(im, ax=ax1, label='Z-scored Power')


        
        # 2. PC1 Spectrogram
        ax2 = fig.add_subplot(gs[subplot_idx], sharex=ax1)
        subplot_idx += 1
        # Extract PC1 for this specific channel
        pc1_channel = pc1_cropped[ch_idx, :]
        pc1_norm = (pc1_channel - np.mean(pc1_channel)) / (np.std(pc1_channel) + 1e-10)
        ax2.plot(lfp_time_cropped, pc1_norm, 'k-', linewidth=0.5)
        ax2.set_ylabel('PC1\nSpectrogram', fontsize=9)
        ax2.set_xlim([t_start, t_end])
        ax2.set_ylim(plot_params['band_ylim'])
        ax2.set_xticklabels([])
        # Remove all spines
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.tick_params(left=False, bottom=False)
        
        # 3. Theta ratio
        ax3 = fig.add_subplot(gs[subplot_idx], sharex=ax1)
        subplot_idx += 1
        theta_ratio_norm = (bands_data['theta_ratio'] - np.mean(bands_data['theta_ratio'])) / (np.std(bands_data['theta_ratio']) + 1e-10)
        ax3.plot(lfp_time_cropped, theta_ratio_norm, 'k-', linewidth=0.5)
        ax3.set_ylabel('Theta ratio\n5-10Hz/2-15Hz', fontsize=9)
        ax3.set_xlim([t_start, t_end])
        ax3.set_ylim(plot_params['band_ylim'])
        ax3.set_xticklabels([])
        # Remove all spines
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.tick_params(left=False, bottom=False)
        
        # 4. Delta
        ax4 = fig.add_subplot(gs[subplot_idx], sharex=ax1)
        subplot_idx += 1
        delta_norm = (bands_data['delta'] - np.mean(bands_data['delta'])) / (np.std(bands_data['delta']) + 1e-10)
        ax4.plot(lfp_time_cropped, delta_norm, 'k-', linewidth=0.5)
        ax4.set_ylabel('0.5-4 Hz\n(Delta)', fontsize=9)
        ax4.set_xlim([t_start, t_end])
        ax4.set_ylim(plot_params['band_ylim'])
        ax4.set_xticklabels([])
        # Remove all spines
        for spine in ax4.spines.values():
            spine.set_visible(False)
        ax4.tick_params(left=False, bottom=False)
        
        # 5. Sigma
        ax5 = fig.add_subplot(gs[subplot_idx], sharex=ax1)
        subplot_idx += 1
        sigma_norm = (bands_data['sigma'] - np.mean(bands_data['sigma'])) / (np.std(bands_data['sigma']) + 1e-10)
        ax5.plot(lfp_time_cropped, sigma_norm, 'k-', linewidth=0.5)
        ax5.set_ylabel('9-25Hz\n(Sigma)', fontsize=9)
        ax5.set_xlim([t_start, t_end])
        ax5.set_ylim(plot_params['band_ylim'])
        ax5.set_xticklabels([])
        # Remove all spines
        for spine in ax5.spines.values():
            spine.set_visible(False)
        ax5.tick_params(left=False, bottom=False)
        
        # 6. Gamma
        ax6 = fig.add_subplot(gs[subplot_idx], sharex=ax1)
        subplot_idx += 1
        gamma_norm = (bands_data['gamma'] - np.mean(bands_data['gamma'])) / (np.std(bands_data['gamma']) + 1e-10)
        ax6.plot(lfp_time_cropped, gamma_norm, 'k-', linewidth=0.5)
        ax6.set_ylabel('40-100Hz\n(Gamma)', fontsize=9)
        ax6.set_xlim([t_start, t_end])
        ax6.set_ylim(plot_params['band_ylim'])
        # Remove all spines
        for spine in ax6.spines.values():
            spine.set_visible(False)
        ax6.tick_params(left=False, bottom=False)
        
        # 7. Velocity (if available)
        if plot_velocity:
            ax7 = fig.add_subplot(gs[subplot_idx], sharex=ax1)
            subplot_idx += 1
            
            ax7.plot(velocity_time_synced, velocity_synced, 'b-', linewidth=0.5)
            ax7.set_ylabel('Velocity\n(cm/s)', fontsize=9)
            ax7.set_xlim([t_start, t_end])
            # Remove all spines
            for spine in ax7.spines.values():
                spine.set_visible(False)
            ax7.tick_params(left=False, bottom=False)
            ax7.set_xlabel('Time (s)', fontsize=10)
        else:
            ax6.set_xlabel('Time (s)', fontsize=10)
        
        # Add visible scale bar for 500s
        last_ax = fig.get_axes()[-1]
        
        # Calculate scale bar position and length
        # Draw the scale bar in data coordinates
        scale_length = 500  # seconds
        x_end = t_end - 100  # 100s from the right edge
        x_start = x_end - scale_length

        # Get the y-axis limits
        y_limits = last_ax.get_ylim()
        y_range = y_limits[1] - y_limits[0]
        y_pos = y_limits[0] - 0.15 * y_range  # Position below the plot

        # Draw the scale bar
        last_ax.plot([x_start, x_end], [y_pos, y_pos], 
                    color='black', linewidth=8, solid_capstyle='butt',
                    clip_on=False)  # Important: don't clip the line

        # Add text label below the bar
        last_ax.text((x_start + x_end) / 2, y_pos - 0.1 * y_range, '500s', 
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    clip_on=False)  # Important: don't clip the text
        
        # Save figure
        output_file = output_folder / f'{session_name}_sh{shank_id}_ch{ch_id:03d}_full_recording.png'
        print(f"  Saving to: {output_file.name}")
        plt.savefig(output_file, dpi=plot_params['dpi'], bbox_inches='tight')
        plt.close()
        total_files_created += 1
        
        print(f"  ✓ Completed Shank {shank_id}, Channel {ch_id}")

print(f"\n{'='*60}")
print("PLOTTING COMPLETE")
print(f"{'='*60}")
print(f"Output directory: {output_folder}")
print(f"Total files created: {total_files_created} (one full recording per channel per shank)")
print(f"Processed shanks: {shanks}")
if plot_velocity:
    print(f"\n✓ Velocity data included and synchronized")
else:
    print(f"\n⚠ Velocity data not included")