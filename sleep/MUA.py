import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.extractors import PhySortingExtractor
from pathlib import Path
import pickle
import os
from scipy.ndimage import gaussian_filter1d

# === PARAMETERS ===
animal_id = 'CnL39SG'
session_id = 'CnL39SG_20251102_210043'
rec_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\sleep\CnL39SG\CnL39SG_20251102_210043.rec"
sortoutfolder = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout"

shanks = [0, 1, 2, 3]
start_time = 0  # Start time in seconds
end_time = 4500  # End time in seconds (None = end of recording)
window_duration = 10  # Duration of each plot window in seconds

# MUA parameters
mua_bin_size = 0.020  # 20 ms bins
mua_smooth_sigma = 2  # Gaussian smoothing sigma (in bins)

# Preprocessing parameters for LFP
preproc_params = {
    'car_reference': 'global',
    'car_operator': 'median',
    'initial_filter_min': 0.1,
    'initial_filter_max': 100,
    'target_fs': 1000,
    'lfp_filter_min': 0.3,
    'lfp_filter_max': 4,
    'dtype': 'float32'
}

# Process each shank
for ish in shanks:
    print(f"\n{'='*70}")
    print(f"PROCESSING SHANK {ish}")
    print(f"{'='*70}\n")
    
    # === LOAD SPIKE DATA ===
    print(f'Loading spike data for {animal_id}/{session_id}/shank{ish}')
    session_folder = Path(sortoutfolder) / animal_id / session_id
    shank_folder = session_folder / f'shank{ish}'
    
    # Find sorting results folders
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))
    
    if len(sorting_results_folders) == 0:
        print(f"  No sorting results found for shank {ish}, skipping...")
        continue
    
    # Use the first (or most recent) sorting results folder
    sorting_results_folder = Path(sorting_results_folders[0])
    phy_folder = sorting_results_folder / 'phy'
    
    if not phy_folder.exists():
        print(f"  Phy folder not found: {phy_folder}, skipping...")
        continue
    
    print(f"  Loading sorting from: {phy_folder}")
    
    # Load sorting
    sorting = PhySortingExtractor(phy_folder)
    unit_ids = sorting.unit_ids
    unit_qualities = sorting.get_property('quality')
    fs_spikes = sorting.sampling_frequency
    
    print(f"  Total units: {len(unit_ids)}")
    
    # Filter out noise units
    good_unit_mask = unit_qualities != 'noise'
    good_unit_ids = unit_ids[good_unit_mask]
    good_unit_qualities = unit_qualities[good_unit_mask]
    
    print(f"  Good units (non-noise): {len(good_unit_ids)}")
    
    if len(good_unit_ids) == 0:
        print(f"  No good units found for shank {ish}, skipping...")
        continue
    
    # Load all spike times for good units
    all_spike_times = []
    unit_spike_times = {}
    
    for unit_id in good_unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id)
        spike_times_sec = spike_train / fs_spikes
        unit_spike_times[unit_id] = spike_times_sec
        all_spike_times.extend(spike_times_sec)
    
    all_spike_times = np.sort(all_spike_times)
    
    print(f"  Total spikes from good units: {len(all_spike_times)}")
    
    # === LOAD LFP DATA ===
    print("\nLoading LFP recording...")
    rec_file = f"{rec_folder}\\{session_id}sh{ish}.nwb"
    rec = se.NwbRecordingExtractor(rec_file)
    folder = Path(rec_file).parent
    
    # Store original recording info
    original_fs = rec.get_sampling_frequency()
    original_duration = rec.get_total_duration()
    
    # Preprocessing
    rec_car = spre.common_reference(rec, reference=preproc_params['car_reference'], 
                                     operator=preproc_params['car_operator'])
    rec_filtered = spre.bandpass_filter(rec_car, 
                                        freq_min=preproc_params['initial_filter_min'], 
                                        freq_max=preproc_params['initial_filter_max'], 
                                        dtype=preproc_params['dtype'])
    
    print("Downsampling LFP...")
    target_fs = preproc_params['target_fs']
    rec_downsampled = spre.resample(rec_filtered, target_fs)
    
    print("Applying LFP filter...")
    rec_lfp = spre.bandpass_filter(rec_downsampled, 
                                    freq_min=preproc_params['lfp_filter_min'], 
                                    freq_max=preproc_params['lfp_filter_max'], 
                                    dtype=preproc_params['dtype'])
    
    # Get channel info
    channel_locations = rec.get_channel_locations()
    ycoord = channel_locations[:, 1]
    channel_ids = rec.get_channel_ids()
    
    # Sort by depth
    depth_order = np.argsort(ycoord)
    sorted_channel_ids = channel_ids[depth_order]
    ycoord_sorted = ycoord[depth_order]
    
    fs_lfp = rec_lfp.get_sampling_frequency()
    n_channels = rec_lfp.get_num_channels()
    duration = rec_lfp.get_total_duration()
    
    print(f"  LFP: {fs_lfp} Hz, {n_channels} channels, {duration:.2f}s")
    
    # Set time range
    current_end_time = end_time if end_time is not None else duration
    current_start_time = max(0, start_time)
    if current_end_time > duration:
        current_end_time = duration
    
    # === COMPUTE MUA RATE ===
    print("\nComputing MUA rate...")
    
    # Create time bins for MUA
    mua_time_bins = np.arange(current_start_time, current_end_time, mua_bin_size)
    
    # Filter spikes to time range
    spike_mask = (all_spike_times >= current_start_time) & (all_spike_times < current_end_time)
    spikes_in_range = all_spike_times[spike_mask]
    
    # Compute spike counts in bins
    spike_counts, _ = np.histogram(spikes_in_range, bins=mua_time_bins)
    
    # Convert to rate (spikes/sec)
    mua_rate = spike_counts / mua_bin_size
    
    # Smooth MUA rate
    mua_rate_smooth = gaussian_filter1d(mua_rate, sigma=mua_smooth_sigma)
    
    # Time vector for MUA (center of bins)
    mua_time = mua_time_bins[:-1] + mua_bin_size / 2
    
    print(f"  MUA computed: {len(mua_time)} time bins")
    print(f"  Mean MUA rate: {np.mean(mua_rate_smooth):.2f} spikes/s")
    print(f"  Max MUA rate: {np.max(mua_rate_smooth):.2f} spikes/s")
    
    # === EXTRACT LFP DATA ===
    print("Extracting LFP traces...")
    lfp_traces = rec_lfp.get_traces(
        start_frame=int(current_start_time * fs_lfp), 
        end_frame=int(current_end_time * fs_lfp),
        channel_ids=sorted_channel_ids.tolist()
    )
    
    # === SAVE MUA AND SPIKE DATA ===
    output_dir = folder / "mua_analysis" / f"shank{ish}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    mua_data = {
        'mua_rate': mua_rate,
        'mua_rate_smooth': mua_rate_smooth,
        'mua_time': mua_time,
        'mua_bin_size': mua_bin_size,
        'mua_smooth_sigma': mua_smooth_sigma,
        'all_spike_times': spikes_in_range,
        'unit_spike_times': {str(k): v[(v >= current_start_time) & (v < current_end_time)] 
                            for k, v in unit_spike_times.items()},
        'unit_ids': good_unit_ids,
        'unit_qualities': good_unit_qualities,
        'n_units': len(good_unit_ids),
        'n_spikes': len(spikes_in_range),
        'fs_spikes': fs_spikes,
        'time_range': (current_start_time, current_end_time),
        'duration': current_end_time - current_start_time,
        'session_name': session_id,
        'shank': ish,
        'sorting_folder': str(sorting_results_folder)
    }
    
    # Save MUA data
    mua_pkl_filename = (f"{session_id}_sh{ish}_MUA_"
                       f"{mua_bin_size*1000:.0f}ms-bins_"
                       f"sigma{mua_smooth_sigma}_"
                       f"{current_start_time:.0f}-{current_end_time:.0f}s.pkl")
    mua_pkl_path = output_dir / mua_pkl_filename
    
    print(f"\nSaving MUA data to: {mua_pkl_filename}")
    with open(mua_pkl_path, 'wb') as f:
        pickle.dump(mua_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  File size: {mua_pkl_path.stat().st_size / 1024**2:.2f} MB")
    
    # === CALCULATE GLOBAL MUA RANGE FOR CONSISTENT Y-AXIS ===
    mua_min = 0  # Always start at 0 for rate
    mua_max = np.max(mua_rate_smooth)
    mua_mean = np.mean(mua_rate_smooth)
    mua_std = np.std(mua_rate_smooth)
    
    # Add some padding to the max
    mua_ylim = [mua_min, mua_max * 1.1]
    
    print(f"\nMUA range for all plots: {mua_min:.2f} - {mua_max:.2f} spikes/s")
    print(f"MUA mean ± std: {mua_mean:.2f} ± {mua_std:.2f} spikes/s")
    
    # === PLOT ALL WINDOWS ===
    n_windows = int(np.ceil((current_end_time - current_start_time) / window_duration))
    print(f"Plotting {n_windows} windows...")
    
    for win_idx in range(n_windows):
        win_start = current_start_time + win_idx * window_duration
        win_end = min(win_start + window_duration, current_end_time)
        
        if (win_idx + 1) % 10 == 0 or win_idx == 0:
            print(f"  Processing window {win_idx+1}/{n_windows}: {win_start:.2f}s - {win_end:.2f}s")
        
        # Get LFP for this window
        start_idx_lfp = int((win_start - current_start_time) * fs_lfp)
        end_idx_lfp = int((win_end - current_start_time) * fs_lfp)
        traces = lfp_traces[start_idx_lfp:end_idx_lfp, :]
        time_vector_lfp = np.arange(traces.shape[0]) / fs_lfp + win_start
        
        # Get MUA for this window
        mua_mask = (mua_time >= win_start) & (mua_time < win_end)
        mua_time_win = mua_time[mua_mask]
        mua_rate_win = mua_rate_smooth[mua_mask]
        
        # Get spikes for this window
        spike_times_win = {}
        for unit_id in good_unit_ids:
            unit_spikes = unit_spike_times[unit_id]
            unit_spikes_win = unit_spikes[(unit_spikes >= win_start) & (unit_spikes < win_end)]
            if len(unit_spikes_win) > 0:
                spike_times_win[unit_id] = unit_spikes_win
        
        # === CREATE FIGURE ===
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.3)
        
        # === TOP: RASTER PLOT ===
        ax1 = fig.add_subplot(gs[0])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(spike_times_win)))
        for i, (unit_id, spikes) in enumerate(spike_times_win.items()):
            ax1.scatter(spikes, np.ones_like(spikes) * i, 
                       marker='|', s=50, alpha=0.8, color=colors[i])
        
        ax1.set_ylabel('Unit ID', fontsize=12)
        ax1.set_title(f'Spike Raster ({len(spike_times_win)} units, {sum(len(s) for s in spike_times_win.values())} spikes)', 
                     fontsize=13, fontweight='bold')
        ax1.set_xlim([win_start, win_end])
        ax1.set_ylim([-0.5, len(spike_times_win) + 0.5])
        ax1.grid(True, alpha=0.3, axis='x')
        
        # === MIDDLE: MUA RATE ===
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        ax2.fill_between(mua_time_win, 0, mua_rate_win, alpha=0.3, color='C0')
        ax2.plot(mua_time_win, mua_rate_win, linewidth=1.5, color='C0')
        ax2.set_ylabel('MUA Rate (spikes/s)', fontsize=12)
        ax2.set_title(f'Multi-Unit Activity (bin={mua_bin_size*1000:.0f}ms, σ={mua_smooth_sigma})', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlim([win_start, win_end])
        ax2.set_ylim(mua_ylim)  # Fixed y-limits for all plots
        ax2.grid(True, alpha=0.3)
        
        # Mark UP states (high MUA) - use global threshold
        mua_threshold = mua_mean + 0.5 * mua_std
        ax2.axhline(mua_threshold, color='red', linestyle='--', alpha=0.5, 
                   linewidth=1, label=f'UP threshold ({mua_threshold:.1f} sp/s)')
        
        # Also show mean
        ax2.axhline(mua_mean, color='gray', linestyle=':', alpha=0.5, 
                   linewidth=1, label=f'Mean ({mua_mean:.1f} sp/s)')
        ax2.legend(loc='upper right', fontsize=9)
        
        # === BOTTOM: LFP (selected channels) ===
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Show a subset of channels for clarity (e.g., every 4th channel)
        channel_step = max(1, n_channels // 16)
        channels_to_plot = np.arange(0, n_channels, channel_step)
        
        seg_stds = np.std(traces[:, channels_to_plot], axis=0)
        offset_multiplier = np.median(seg_stds) * 8
        offsets = -np.arange(len(channels_to_plot)) * offset_multiplier
        
        for i, ch_idx in enumerate(channels_to_plot):
            ax3.plot(time_vector_lfp, traces[:, ch_idx] + offsets[i], 
                    linewidth=0.5, alpha=0.7, color='black')
            
            # Add depth labels
            ax3.text(win_end + 0.05, offsets[i], f'{ycoord_sorted[ch_idx]:.0f}μm', 
                    fontsize=7, va='center')
        
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('LFP (depth)', fontsize=12)
        ax3.set_title(f'LFP Traces ({len(channels_to_plot)} channels, {preproc_params["lfp_filter_min"]}-{preproc_params["lfp_filter_max"]} Hz)', 
                     fontsize=13, fontweight='bold')
        ax3.set_xlim([win_start, win_end])
        ax3.set_ylim([offsets.min() - offset_multiplier, offsets.max() + offset_multiplier])
        ax3.set_yticks([])
        ax3.spines['left'].set_visible(False)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add overall title
        fig.suptitle(f'Shank {ish} | Window {win_idx+1}/{n_windows} ({win_start:.1f}s - {win_end:.1f}s) | {len(good_unit_ids)} units', 
                     fontsize=14, fontweight='bold')
        
        # Save figure
        output_file = output_dir / f'sh{ish}_raster_mua_lfp_window_{win_idx+1:04d}_{win_start:.1f}s-{win_end:.1f}s.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n{'='*70}")
    print(f"Shank {ish}: Complete!")
    print(f"  MUA data saved: {mua_pkl_filename}")
    print(f"  Plots saved to: {output_dir}")
    print(f"{'='*70}")

print(f"\n{'#'*70}")
print(f"ALL SHANKS PROCESSING COMPLETE!")
print(f"{'#'*70}")