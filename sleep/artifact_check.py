import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from process_func.recording_proc import rm_artifacts, compute_norms_numba
from pathlib import Path

rec_file = "/Volumes/xieluanlabs/xl_cl/ephys/sleep/CnL39SG/CnL39SG_20251102_210043.rec/CnL39SG_20251102_210043sh0.nwb"
rec = se.NwbRecordingExtractor(rec_file)

# Extract shank number and recording name
ish = int(rec_file.split('sh')[-1].split('.')[0])
rec_name = Path(rec_file).stem  # e.g., "CnL39SG_20251102_210043sh0"
folder = Path(rec_file).parent

print(f"Processing recording: {rec_file}")

# Set chunk time
chunk_time = 0.05

# Create output directory structure
output_base = folder / "artifact_test" / rec_name / f"chunk_{chunk_time}s"
output_base.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_base}")

# Load or compute artifacts
rec_clean = rm_artifacts(rec, folder, ish, threshold=6, chunk_time=chunk_time, overwrite=False)

# Load artifact indices
artifact_file = folder / f'artifact_indices_sh{ish}_{chunk_time}.npy'
artifact_indices = np.load(artifact_file)

print(f"Total artifact windows detected: {len(artifact_indices)}")

# ============ VISUALIZATION ============

fs = rec.get_sampling_frequency()
chunk_size = int(chunk_time * fs)
n_channels = rec.get_num_channels()
n_timepoints = rec.get_num_frames()

# 1. OVERVIEW: Artifact distribution over time
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Plot artifact locations
artifact_times = artifact_indices / fs
axes[0].scatter(artifact_times, np.ones_like(artifact_times), 
                marker='|', s=100, c='red', alpha=0.6)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Artifacts')
axes[0].set_title(f'Artifact Distribution (n={len(artifact_indices)}, chunk={chunk_time}s)')
axes[0].set_ylim([0.5, 1.5])
axes[0].set_xlim([0, n_timepoints / fs])
axes[0].grid(True, alpha=0.3)

# 2. HISTOGRAM: Artifact rate over time
time_bins = np.arange(0, n_timepoints / fs, 10)  # 10-second bins
artifact_hist, _ = np.histogram(artifact_times, bins=time_bins)
axes[1].bar(time_bins[:-1], artifact_hist, width=10, alpha=0.7, color='red')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Artifact count per 10s')
axes[1].set_title('Artifact Rate Over Time')
axes[1].grid(True, alpha=0.3)

# 3. CHUNK NORMS: Recreate detection statistics
num_chunks = int(np.ceil(n_timepoints / chunk_size))

# Load high-pass filtered data for visualization (same as detection)
rec_detect = spre.bandpass_filter(rec, freq_min=300, freq_max=6000)

# Compute norms for a subset of channels (for speed)
n_channels_plot = min(4, n_channels)
norms = np.zeros((num_chunks, n_channels_plot))

print("Computing chunk norms for visualization...")
print("Computing chunk norms with Numba...")
traces = rec_detect.get_traces(return_scaled=True)
norms = compute_norms_numba(traces, chunk_size)

# Plot chunk norms with threshold
chunk_times = np.arange(num_chunks) * chunk_size / fs

for ch in range(n_channels_plot):
    vals = norms[:, ch]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    threshold_val = mean_val + 6 * std_val
    
    axes[2].plot(chunk_times, vals, alpha=0.5, label=f'Ch {ch}')
    axes[2].axhline(threshold_val, color='red', linestyle='--', alpha=0.3)

# Mark detected artifacts
artifact_chunks = artifact_indices / chunk_size
axes[2].scatter(artifact_chunks * chunk_size / fs, 
                np.zeros_like(artifact_chunks), 
                marker='x', s=50, c='red', alpha=0.6, label='Artifacts')

axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Chunk Norm')
axes[2].set_title(f'Chunk Norms and Detection Threshold (first 4 channels, chunk={chunk_time}s)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_base / '1_artifact_overview.png', dpi=150)
plt.close()
print(f"Saved: 1_artifact_overview.png")

# ============ DETAILED VIEW: Example artifact windows ============

# Select a few artifact examples to visualize
n_examples = min(5, len(artifact_indices))
if n_examples > 0:
    # Sample evenly spaced artifacts
    example_indices = np.linspace(0, len(artifact_indices)-1, n_examples, dtype=int)
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, artifact_idx in enumerate(example_indices):
        artifact_frame = artifact_indices[artifact_idx]
        
        # Load a window around the artifact
        window_ms = 200  # 200ms window for better context
        window_samples = int(window_ms * fs / 1000)
        start_frame = max(0, artifact_frame - window_samples // 2)
        end_frame = min(n_timepoints, artifact_frame + window_samples // 2)
        
        # Get raw and cleaned traces
        raw_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                     return_scaled=True)
        clean_traces = rec_clean.get_traces(start_frame=start_frame, end_frame=end_frame,
                                             return_scaled=True)
        
        time_vec = (np.arange(raw_traces.shape[0]) + start_frame) / fs * 1000  # in ms
        artifact_time = artifact_frame / fs * 1000
        
        # Plot raw traces (all channels)
        for ch in range(n_channels):
            offset = ch * 300
            axes[i, 0].plot(time_vec, raw_traces[:, ch] + offset, 
                           alpha=0.6, linewidth=0.5, color='C0')
        
        axes[i, 0].axvspan(artifact_time, artifact_time + chunk_size/fs*1000, 
                          alpha=0.3, color='red', label='Artifact window')
        axes[i, 0].set_xlabel('Time (ms)')
        axes[i, 0].set_ylabel('Channel (offset)')
        axes[i, 0].set_title(f'Example {i+1}: RAW traces (t={artifact_frame/fs:.2f}s, all {n_channels} channels)')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot cleaned traces (all channels)
        for ch in range(n_channels):
            offset = ch * 300
            axes[i, 1].plot(time_vec, clean_traces[:, ch] + offset, 
                           alpha=0.6, linewidth=0.5, color='C2')
        
        axes[i, 1].axvspan(artifact_time, artifact_time + chunk_size/fs*1000, 
                          alpha=0.3, color='green', label='Interpolated')
        axes[i, 1].set_xlabel('Time (ms)')
        axes[i, 1].set_ylabel('Channel (offset)')
        axes[i, 1].set_title(f'Example {i+1}: CLEANED traces (interpolation)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_base / '2_artifact_examples.png', dpi=150)
    plt.close()
    print(f"Saved: 2_artifact_examples.png")

# ============ PLOT ALL DETECTED ARTIFACTS ============

print("Creating individual plots for all artifacts...")

# Create subfolder for individual artifacts
artifact_plots_dir = output_base / "all_artifacts"
artifact_plots_dir.mkdir(exist_ok=True)

# Process in batches if there are many artifacts
max_individual_plots = 50  # Limit to prevent too many files
n_artifacts_to_plot = min(len(artifact_indices), max_individual_plots)

if len(artifact_indices) > max_individual_plots:
    print(f"Warning: {len(artifact_indices)} artifacts detected. Only plotting first {max_individual_plots}.")
    artifact_indices_to_plot = artifact_indices[:max_individual_plots]
else:
    artifact_indices_to_plot = artifact_indices

for idx, artifact_frame in enumerate(artifact_indices_to_plot):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Load a window around the artifact
    window_ms = 200
    window_samples = int(window_ms * fs / 1000)
    start_frame = max(0, artifact_frame - window_samples // 2)
    end_frame = min(n_timepoints, artifact_frame + window_samples // 2)
    
    # Get raw and cleaned traces
    raw_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                 return_scaled=True)
    clean_traces = rec_clean.get_traces(start_frame=start_frame, end_frame=end_frame,
                                         return_scaled=True)
    
    time_vec = (np.arange(raw_traces.shape[0]) + start_frame) / fs * 1000
    artifact_time = artifact_frame / fs * 1000
    
    # Plot raw traces
    for ch in range(n_channels):
        offset = ch * 300
        axes[0].plot(time_vec, raw_traces[:, ch] + offset, 
                   alpha=0.6, linewidth=0.5, color='C0')
    
    axes[0].axvspan(artifact_time, artifact_time + chunk_size/fs*1000, 
                  alpha=0.3, color='red', label='Artifact window')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Channel (offset)')
    axes[0].set_title(f'RAW: Artifact #{idx+1} at t={artifact_frame/fs:.2f}s')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot cleaned traces
    for ch in range(n_channels):
        offset = ch * 300
        axes[1].plot(time_vec, clean_traces[:, ch] + offset, 
                   alpha=0.6, linewidth=0.5, color='C2')
    
    axes[1].axvspan(artifact_time, artifact_time + chunk_size/fs*1000, 
                  alpha=0.3, color='green', label='Interpolated')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Channel (offset)')
    axes[1].set_title(f'CLEANED: Artifact #{idx+1} (interpolated)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(artifact_plots_dir / f'artifact_{idx+1:04d}_t{artifact_frame/fs:.2f}s.png', dpi=100)
    plt.close()
    
    if (idx + 1) % 10 == 0:
        print(f"  Plotted {idx+1}/{n_artifacts_to_plot} artifacts")

print(f"Saved all individual artifact plots to: {artifact_plots_dir}")

# ============ LFP COMPARISON ============

# Common reference and filter
rec_car = spre.common_reference(rec_clean, reference='global', operator='median')
rec_lfp = spre.bandpass_filter(rec_car, freq_min=0.1, freq_max=100, dtype="float32")

# Pick a time window that contains artifacts
if len(artifact_indices) > 0:
    # Find a window with artifacts
    artifact_frame = artifact_indices[len(artifact_indices)//2]
    
    # Load 5 seconds around an artifact
    window_sec = 5
    window_samples = int(window_sec * fs)
    start_frame = max(0, artifact_frame - window_samples // 2)
    end_frame = min(n_timepoints, artifact_frame + window_samples // 2)
    
    # Get raw LFP (without artifact removal)
    rec_car_raw = spre.common_reference(rec, reference='global', operator='median')
    rec_lfp_raw = spre.bandpass_filter(rec_car_raw, freq_min=0.1, freq_max=100, 
                                        dtype="float32")
    
    lfp_raw = rec_lfp_raw.get_traces(start_frame=start_frame, end_frame=end_frame,
                                      return_scaled=True)
    lfp_clean = rec_lfp.get_traces(start_frame=start_frame, end_frame=end_frame,
                                    return_scaled=True)
    
    time_vec = (np.arange(lfp_raw.shape[0]) + start_frame) / fs
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot all channels with smaller offset
    for ch in range(n_channels):
        axes[0].plot(time_vec, lfp_raw[:, ch] + ch * 150, 
                    alpha=0.6, linewidth=0.8)
    
    # Mark artifact windows
    for art_frame in artifact_indices:
        if start_frame <= art_frame <= end_frame:
            art_time = art_frame / fs
            axes[0].axvspan(art_time, art_time + chunk_size/fs, 
                           alpha=0.2, color='red')
    
    axes[0].set_ylabel('LFP amplitude (µV, offset)')
    axes[0].set_title(f'LFP WITHOUT artifact removal (all {n_channels} channels)')
    axes[0].grid(True, alpha=0.3)
    
    # Cleaned LFP
    for ch in range(n_channels):
        axes[1].plot(time_vec, lfp_clean[:, ch] + ch * 150, 
                    alpha=0.6, linewidth=0.8)
    
    for art_frame in artifact_indices:
        if start_frame <= art_frame <= end_frame:
            art_time = art_frame / fs
            axes[1].axvspan(art_time, art_time + chunk_size/fs, 
                           alpha=0.2, color='green')
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('LFP amplitude (µV, offset)')
    axes[1].set_title(f'LFP WITH artifact removal (chunk={chunk_time}s)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_base / '3_lfp_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: 3_lfp_comparison.png")

# ============ SUMMARY STATISTICS ============

# Create summary text file
summary_file = output_base / 'artifact_summary.txt'
with open(summary_file, 'w') as f:
    f.write(f"Artifact Detection Summary\n")
    f.write(f"=" * 50 + "\n\n")
    f.write(f"Recording: {rec_name}\n")
    f.write(f"Shank: {ish}\n")
    f.write(f"Chunk time: {chunk_time} s\n")
    f.write(f"Chunk size: {chunk_size} samples\n")
    f.write(f"Sampling rate: {fs} Hz\n")
    f.write(f"Total duration: {n_timepoints/fs:.2f} s\n")
    f.write(f"Number of channels: {n_channels}\n\n")
    f.write(f"Detection threshold: 6 std\n")
    f.write(f"Detection frequency range: 300-6000 Hz\n\n")
    f.write(f"Results:\n")
    f.write(f"-" * 50 + "\n")
    f.write(f"Total artifacts detected: {len(artifact_indices)}\n")
    f.write(f"Total time removed: {len(artifact_indices)*chunk_size/fs:.2f} s\n")
    f.write(f"Percentage of data removed: {len(artifact_indices)*chunk_size/n_timepoints*100:.2f}%\n")
    f.write(f"Artifact rate: {len(artifact_indices)/(n_timepoints/fs)*60:.2f} artifacts/min\n")

print(f"\nVisualization complete!")
print(f"Results saved to: {output_base}")
print(f"\nSummary:")
print(f"  Total artifacts: {len(artifact_indices)}")
print(f"  Time removed: {len(artifact_indices)*chunk_size/fs:.2f}s")
print(f"  Percentage removed: {len(artifact_indices)*chunk_size/n_timepoints*100:.2f}%")
print(f"  Artifact rate: {len(artifact_indices)/(n_timepoints/fs)*60:.2f} artifacts/min")