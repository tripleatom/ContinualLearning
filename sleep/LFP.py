import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from pathlib import Path
import pickle

rec_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\sleep\CnL39SG\CnL39SG_20251102_210043.rec"
session_name = Path(rec_folder).stem.split('.')[0]
shanks = [0, 1, 2, 3]

# === PARAMETERS ===
start_time = 600  # Start time in seconds
end_time = 4500  # End time in seconds (None = end of recording)
window_duration = 10  # Duration of each plot window in seconds

# Preprocessing parameters
preproc_params = {
    'car_reference': 'global',
    'car_operator': 'median',
    'initial_filter_min': 0.1,
    'initial_filter_max': 100,
    'target_fs': 1000,
    'lfp_filter_min': 0.1,
    'lfp_filter_max': 300,
    'dtype': 'float32'
}

# Process each shank
for ish in shanks:
    print(f"\n{'='*70}")
    print(f"PROCESSING SHANK {ish}")
    print(f"{'='*70}\n")
    
    # Load recording for this shank
    rec_file = f"{rec_folder}\\{session_name}sh{ish}.nwb"
    rec = se.NwbRecordingExtractor(rec_file)
    folder = Path(rec_file).parent
    
    # Store original recording info
    original_fs = rec.get_sampling_frequency()
    original_duration = rec.get_total_duration()
    
    # === PREPROCESSING ===
    # Common reference FIRST (on raw data)
    rec_car = spre.common_reference(rec, reference=preproc_params['car_reference'], 
                                     operator=preproc_params['car_operator'])

    # Bandpass filter for LFP band
    rec_filtered = spre.bandpass_filter(rec_car, 
                                        freq_min=preproc_params['initial_filter_min'], 
                                        freq_max=preproc_params['initial_filter_max'], 
                                        dtype=preproc_params['dtype'])

    # Downsample
    print("Downsampling...")
    target_fs = preproc_params['target_fs']
    rec_downsampled = spre.resample(rec_filtered, target_fs)

    # Apply LFP filter
    print("Applying LFP filter...")
    rec_lfp = spre.bandpass_filter(rec_downsampled, 
                                    freq_min=preproc_params['lfp_filter_min'], 
                                    freq_max=preproc_params['lfp_filter_max'], 
                                    dtype=preproc_params['dtype'])

    # Get channel locations and sort by depth
    channel_locations = rec.get_channel_locations()
    xcoord = channel_locations[:, 0]
    ycoord = channel_locations[:, 1]  # Second column is depth
    channel_ids = rec.get_channel_ids()

    # Sort channels by depth - ASCENDING ORDER (0μm, 20μm, 40μm, ... = shallow to deep)
    depth_order = np.argsort(ycoord)
    sorted_channel_ids = channel_ids[depth_order]
    ycoord_sorted = ycoord[depth_order]
    xcoord_sorted = xcoord[depth_order]

    # Get basic info
    fs = rec_lfp.get_sampling_frequency()
    n_channels = rec_lfp.get_num_channels()
    duration = rec_lfp.get_total_duration()

    print(f"Original sampling rate: {original_fs} Hz")
    print(f"Downsampled to: {fs} Hz")
    print(f"Number of channels: {n_channels}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Depth range: {ycoord_sorted.min():.1f} to {ycoord_sorted.max():.1f} μm (shallow to deep)")

    # Set end time if not specified
    current_end_time = end_time
    if current_end_time is None:
        current_end_time = duration

    # Validate time range
    current_start_time = max(0, start_time)
    if current_end_time > duration:
        current_end_time = duration

    print(f"\nExtracting LFP data from {current_start_time:.2f}s to {current_end_time:.2f}s")
    
    # === EXTRACT AND SAVE LFP DATA ===
    print("Extracting LFP traces...")
    # Get all LFP traces for the specified time range, using sorted channel order
    lfp_traces = rec_lfp.get_traces(
        start_frame=int(current_start_time * fs), 
        end_frame=int(current_end_time * fs),
        channel_ids=sorted_channel_ids.tolist()
    )
    
    # Create data dictionary to save
    lfp_data = {
        'traces': lfp_traces,  # Shape: (n_timepoints, n_channels) - already depth-sorted
        'sampling_rate': fs,
        'channel_ids': sorted_channel_ids,
        'channel_locations': channel_locations[depth_order],  # Sorted by depth
        'xcoord': xcoord_sorted,
        'ycoord': ycoord_sorted,  # Depth coordinates (sorted)
        'depth_order': depth_order,  # Original indices in sorted order
        'time_range': (current_start_time, current_end_time),
        'duration': current_end_time - current_start_time,
        'n_channels': n_channels,
        'n_timepoints': lfp_traces.shape[0],
        'preprocessing': preproc_params.copy(),
        'original_fs': original_fs,
        'original_duration': original_duration,
        'session_name': session_name,
        'shank': ish,
        'rec_file': rec_file
    }
    
    # Create output directory for this shank
    output_dir = folder / "lfp_plots" / f"shank{ish}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LFP data with informative filename
    pkl_filename = (f"{session_name}_sh{ish}_LFP_"
                   f"{preproc_params['lfp_filter_min']:.1f}-{preproc_params['lfp_filter_max']:.0f}Hz_"
                   f"{fs:.0f}Hz_"
                   f"CAR-{preproc_params['car_operator']}_"
                   f"{current_start_time:.0f}-{current_end_time:.0f}s.pkl")
    pkl_path = output_dir / pkl_filename
    
    print(f"Saving LFP data to: {pkl_filename}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(lfp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  LFP data saved: {lfp_traces.shape[0]} timepoints x {n_channels} channels")
    print(f"  File size: {pkl_path.stat().st_size / 1024**2:.2f} MB")
    
    print(f"\nPlotting from {current_start_time:.2f}s to {current_end_time:.2f}s")
    print(f"Window duration: {window_duration}s")

    # Calculate number of windows
    n_windows = int(np.ceil((current_end_time - current_start_time) / window_duration))
    print(f"Total windows to plot: {n_windows}")
    print(f"Output directory: {output_dir}")

    # === PLOT ALL WINDOWS ===
    for win_idx in range(n_windows):
        win_start = current_start_time + win_idx * window_duration
        win_end = min(win_start + window_duration, current_end_time)
        actual_duration = win_end - win_start
        
        if (win_idx + 1) % 10 == 0 or win_idx == 0:
            print(f"Processing window {win_idx+1}/{n_windows}: {win_start:.2f}s - {win_end:.2f}s")
        
        # Get traces from already extracted LFP data (more efficient than re-extracting)
        start_idx = int((win_start - current_start_time) * fs)
        end_idx = int((win_end - current_start_time) * fs)
        traces = lfp_traces[start_idx:end_idx, :]
        time_vector = np.arange(traces.shape[0]) / fs + win_start
        
        # Create figure with 2 subplots: traces on left, heatmap on right
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
        
        # === LEFT: LFP TRACES ===
        ax1 = fig.add_subplot(gs[0])
        
        # Calculate offset based on trace standard deviation
        seg_stds = np.std(traces, axis=0)
        offset_multiplier = np.median(seg_stds) * 15
        offsets = -np.arange(n_channels) * offset_multiplier
        
        # Plot all channels ordered by depth (shallow at top, deep at bottom)
        for i in range(n_channels):
            ax1.plot(time_vector, traces[:, i] + offsets[i], linewidth=0.5, alpha=0.8, color='black')
            
            # Add depth labels on the right
            ax1.text(time_vector[-1] + 0.1, offsets[i], f'{ycoord_sorted[i]:.0f}μm', 
                    fontsize=8, va='center')
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Depth (0μm → deep)', fontsize=12)
        ax1.set_title(f'LFP Traces by Depth ({win_start:.1f}s - {win_end:.1f}s)', fontsize=13, fontweight='bold')
        ax1.set_xlim([win_start, win_end])
        ax1.set_ylim([offsets.min() - offset_multiplier, offsets.max() + offset_multiplier])
        
        # Remove y-axis ticks and spines for cleaner look
        ax1.set_yticks([])
        ax1.spines['left'].set_visible(False)
        
        # Add depth indicators on left
        ax1.text(win_start - 0.2, offsets[0], f'0μm', 
                fontsize=10, va='center', ha='right', weight='bold')
        ax1.text(win_start - 0.2, offsets[-1], f'{ycoord_sorted[-1]:.0f}μm', 
                fontsize=10, va='center', ha='right', weight='bold')
        
        ax1.grid(True, alpha=0.3, axis='x')
        
        # === RIGHT: HEATMAP ===
        ax2 = fig.add_subplot(gs[1])
        
        # Compute z-score for better visualization
        traces_zscore = (traces - np.mean(traces, axis=0)) / (np.std(traces, axis=0) + 1e-10)
        
        im = ax2.imshow(traces_zscore.T, aspect='auto', cmap='RdBu_r',
                        extent=[time_vector[0], time_vector[-1], n_channels-0.5, -0.5],
                        interpolation='bilinear', vmin=-3, vmax=3)
        
        # Set y-axis to show depths
        depth_ticks = np.linspace(0, n_channels-1, min(10, n_channels), dtype=int)
        ax2.set_yticks(depth_ticks)
        ax2.set_yticklabels([f'{ycoord_sorted[i]:.0f}μm' for i in depth_ticks])
        
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Depth', fontsize=12)
        ax2.set_title(f'LFP Heatmap (z-scored)', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Z-score', fontsize=11)
        
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add overall title with shank info
        fig.suptitle(f'Shank {ish} | LFP Analysis: Window {win_idx+1}/{n_windows} ({win_start:.1f}s - {win_end:.1f}s) | 0.1-300 Hz, {fs:.0f} Hz', 
                     fontsize=14, fontweight='bold')
        
        # Save figure
        output_file = output_dir / f'sh{ish}_lfp_window_{win_idx+1:04d}_{win_start:.1f}s-{win_end:.1f}s.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n{'='*70}")
    print(f"Shank {ish}: All {n_windows} windows plotted successfully!")
    print(f"LFP data saved: {pkl_filename}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")

print(f"\n{'#'*70}")
print(f"ALL SHANKS PROCESSING COMPLETE!")
print(f"{'#'*70}")