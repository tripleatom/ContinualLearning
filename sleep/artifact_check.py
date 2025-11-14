import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from process_func.recording_proc import rm_artifacts, compute_norms_numba
from pathlib import Path

rec_file = r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\sleep\CnL39SG\CnL39SG_20251102_210043.rec\CnL39SG_20251102_210043sh0.nwb"
rec = se.NwbRecordingExtractor(rec_file)

# Extract shank number and recording name
ish = int(rec_file.split('sh')[-1].split('.')[0])
rec_name = Path(rec_file).stem  # e.g., "CnL39SG_20251102_210043sh0"
folder = Path(rec_file).parent

print(f"Processing recording: {rec_file}")

# Set chunk time
chunk_time = 0.02  # in seconds
threshold = 8  # threshold in std deviations

# Create output directory structure
output_base = folder / "artifact_test" / rec_name / f"chunk_{chunk_time}s_thresh_{threshold}"
output_base.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_base}")

# Load or compute artifacts
rec_clean = rm_artifacts(rec, folder, ish, threshold=threshold, chunk_time=chunk_time, overwrite=False)

# Load artifact indices
artifact_file = folder / f'artifact_indices_sh{ish}_{chunk_time}_{threshold}.npy'
artifact_indices = np.load(artifact_file)

print(f"Total artifact windows detected: {len(artifact_indices)}")

# Get recording parameters
fs = rec.get_sampling_frequency()
chunk_size = int(chunk_time * fs)
n_channels = rec.get_num_channels()
n_timepoints = rec.get_num_frames()

# ============ VISUALIZATION ============

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
# Load high-pass filtered data for visualization (same as detection)
rec_detect = spre.bandpass_filter(rec, freq_min=300, freq_max=6000)

# Compute norms for a subset of channels (for speed)
n_channels_plot = min(4, n_channels)

print("Computing chunk norms for visualization...")
traces = rec_detect.get_traces(return_scaled=True)
norms = compute_norms_numba(traces, chunk_size)

# Use actual number of chunks from norms output
num_chunks = norms.shape[0]

# Plot chunk norms with threshold
chunk_times = np.arange(num_chunks) * chunk_size / fs

for ch in range(n_channels_plot):
    vals = norms[:, ch]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    threshold_val = mean_val + threshold * std_val
    
    axes[2].plot(chunk_times, vals, alpha=0.5, label=f'Ch {ch}')
    axes[2].axhline(threshold_val, color='red', linestyle='--', alpha=0.3)

# Mark detected artifacts
artifact_chunks = artifact_indices / chunk_size
axes[2].scatter(artifact_chunks * chunk_size / fs, 
                np.zeros_like(artifact_chunks), 
                marker='x', s=50, c='red', alpha=0.6, label='Artifacts')

axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Chunk Norm')
axes[2].set_title(f'Chunk Norms and Detection Threshold (first {n_channels_plot} channels, chunk={chunk_time}s, thresh={threshold}σ)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_base / '1_artifact_overview.png', dpi=150)
plt.close()
print(f"Saved: 1_artifact_overview.png")

# ============ PREPARE LFP DATA ============
print("\nPreparing LFP data for comparison...")

# Raw LFP (without artifact removal)
rec_car_raw = spre.common_reference(rec, reference='global', operator='median')
rec_lfp_raw = spre.bandpass_filter(rec_car_raw, freq_min=0.1, freq_max=100, dtype="float32")

# Cleaned LFP (with artifact removal)
rec_car_clean = spre.common_reference(rec_clean, reference='global', operator='median')
rec_lfp_clean = spre.bandpass_filter(rec_car_clean, freq_min=0.1, freq_max=100, dtype="float32")

print("LFP data ready.")

# ============ PLOT ALL ARTIFACTS: RAW DATA COMPARISON ============
print(f"\nCreating raw data comparison for all {len(artifact_indices)} artifacts...")

# Create subfolder for raw data comparisons
raw_plots_dir = output_base / "raw_data_comparison"
raw_plots_dir.mkdir(exist_ok=True)

window_ms = 200  # 200ms window for context
window_samples = int(window_ms * fs / 1000)

for idx, artifact_frame in enumerate(artifact_indices):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate window
    start_frame = max(0, artifact_frame - window_samples // 2)
    end_frame = min(n_timepoints, artifact_frame + window_samples // 2)
    
    # Get raw and cleaned traces (broadband)
    raw_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                 return_scaled=True)
    clean_traces = rec_clean.get_traces(start_frame=start_frame, end_frame=end_frame,
                                         return_scaled=True)
    
    time_vec_ms = (np.arange(raw_traces.shape[0]) + start_frame) / fs * 1000
    artifact_time_ms = artifact_frame / fs * 1000
    
    # Plot RAW
    for ch in range(n_channels):
        offset = ch * 300
        axes[0].plot(time_vec_ms, raw_traces[:, ch] + offset, 
                    alpha=0.6, linewidth=0.5, color='C0')
    
    axes[0].axvspan(artifact_time_ms, artifact_time_ms + chunk_size/fs*1000, 
                   alpha=0.3, color='red', label='Artifact window')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Channel (offset, µV)')
    axes[0].set_title(f'RAW DATA - Artifact #{idx+1} at t={artifact_frame/fs:.2f}s')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot CLEANED
    for ch in range(n_channels):
        offset = ch * 300
        axes[1].plot(time_vec_ms, clean_traces[:, ch] + offset, 
                    alpha=0.6, linewidth=0.5, color='C2')
    
    axes[1].axvspan(artifact_time_ms, artifact_time_ms + chunk_size/fs*1000, 
                   alpha=0.3, color='green', label='Interpolated')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Channel (offset, µV)')
    axes[1].set_title(f'AFTER REMOVAL - Artifact #{idx+1} (interpolated)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(raw_plots_dir / f'raw_artifact_{idx+1:04d}_t{artifact_frame/fs:.2f}s.png', 
               dpi=100, bbox_inches='tight')
    plt.close()
    
    # Progress update
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"  Processed {idx+1}/{len(artifact_indices)} artifacts")

print(f"Saved raw data comparisons to: {raw_plots_dir}")

# ============ PLOT ALL ARTIFACTS: LFP COMPARISON ============
print(f"\nCreating LFP comparison for all {len(artifact_indices)} artifacts...")

# Create subfolder for LFP comparisons
lfp_plots_dir = output_base / "lfp_comparison"
lfp_plots_dir.mkdir(exist_ok=True)

for idx, artifact_frame in enumerate(artifact_indices):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate window
    start_frame = max(0, artifact_frame - window_samples // 2)
    end_frame = min(n_timepoints, artifact_frame + window_samples // 2)
    
    # Get LFP traces
    lfp_raw = rec_lfp_raw.get_traces(start_frame=start_frame, end_frame=end_frame,
                                      return_scaled=True)
    lfp_clean = rec_lfp_clean.get_traces(start_frame=start_frame, end_frame=end_frame,
                                          return_scaled=True)
    
    time_vec_ms = (np.arange(lfp_raw.shape[0]) + start_frame) / fs * 1000
    artifact_time_ms = artifact_frame / fs * 1000
    
    # Plot RAW LFP
    for ch in range(n_channels):
        offset = ch * 150  # Smaller offset for LFP
        axes[0].plot(time_vec_ms, lfp_raw[:, ch] + offset, 
                    alpha=0.6, linewidth=0.8, color='C0')
    
    axes[0].axvspan(artifact_time_ms, artifact_time_ms + chunk_size/fs*1000, 
                   alpha=0.3, color='red', label='Artifact window')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Channel (offset, µV)')
    axes[0].set_title(f'RAW LFP (0.1-100 Hz) - Artifact #{idx+1} at t={artifact_frame/fs:.2f}s')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot CLEANED LFP
    for ch in range(n_channels):
        offset = ch * 150
        axes[1].plot(time_vec_ms, lfp_clean[:, ch] + offset, 
                    alpha=0.6, linewidth=0.8, color='C2')
    
    axes[1].axvspan(artifact_time_ms, artifact_time_ms + chunk_size/fs*1000, 
                   alpha=0.3, color='green', label='Interpolated')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Channel (offset, µV)')
    axes[1].set_title(f'CLEANED LFP (0.1-100 Hz) - Artifact #{idx+1} (interpolated)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(lfp_plots_dir / f'lfp_artifact_{idx+1:04d}_t{artifact_frame/fs:.2f}s.png', 
               dpi=100, bbox_inches='tight')
    plt.close()
    
    # Progress update
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"  Processed {idx+1}/{len(artifact_indices)} artifacts")

print(f"Saved LFP comparisons to: {lfp_plots_dir}")

# ============ STATISTICAL COMPARISON ============
print("\nComputing statistical comparisons...")

# Compute RMS before and after for each artifact
rms_before_raw = []
rms_after_raw = []
rms_before_lfp = []
rms_after_lfp = []

for artifact_frame in artifact_indices:
    # Define artifact window
    start_artifact = artifact_frame
    end_artifact = min(n_timepoints, artifact_frame + chunk_size)
    
    # Get traces in artifact window
    raw_window = rec.get_traces(start_frame=start_artifact, end_frame=end_artifact, 
                                 return_scaled=True)
    clean_window = rec_clean.get_traces(start_frame=start_artifact, end_frame=end_artifact,
                                         return_scaled=True)
    lfp_raw_window = rec_lfp_raw.get_traces(start_frame=start_artifact, end_frame=end_artifact,
                                             return_scaled=True)
    lfp_clean_window = rec_lfp_clean.get_traces(start_frame=start_artifact, end_frame=end_artifact,
                                                 return_scaled=True)
    
    # Compute RMS across channels
    rms_before_raw.append(np.sqrt(np.mean(raw_window**2)))
    rms_after_raw.append(np.sqrt(np.mean(clean_window**2)))
    rms_before_lfp.append(np.sqrt(np.mean(lfp_raw_window**2)))
    rms_after_lfp.append(np.sqrt(np.mean(lfp_clean_window**2)))

rms_before_raw = np.array(rms_before_raw)
rms_after_raw = np.array(rms_after_raw)
rms_before_lfp = np.array(rms_before_lfp)
rms_after_lfp = np.array(rms_after_lfp)

# Plot statistical comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Broadband RMS comparison
axes[0, 0].scatter(rms_before_raw, rms_after_raw, alpha=0.5, s=30)
max_val = max(rms_before_raw.max(), rms_after_raw.max())
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Identity line')
axes[0, 0].set_xlabel('RMS Before (µV)')
axes[0, 0].set_ylabel('RMS After (µV)')
axes[0, 0].set_title('Broadband RMS: Before vs After')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Broadband RMS reduction
rms_reduction_raw = (rms_before_raw - rms_after_raw) / rms_before_raw * 100
axes[0, 1].hist(rms_reduction_raw, bins=30, alpha=0.7, color='C0')
axes[0, 1].axvline(np.median(rms_reduction_raw), color='red', linestyle='--', 
                   label=f'Median: {np.median(rms_reduction_raw):.1f}%')
axes[0, 1].set_xlabel('RMS Reduction (%)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title(f'Broadband RMS Reduction Distribution (n={len(artifact_indices)})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. LFP RMS comparison
axes[1, 0].scatter(rms_before_lfp, rms_after_lfp, alpha=0.5, s=30, color='C1')
max_val_lfp = max(rms_before_lfp.max(), rms_after_lfp.max())
axes[1, 0].plot([0, max_val_lfp], [0, max_val_lfp], 'r--', alpha=0.5, label='Identity line')
axes[1, 0].set_xlabel('RMS Before (µV)')
axes[1, 0].set_ylabel('RMS After (µV)')
axes[1, 0].set_title('LFP RMS: Before vs After')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. LFP RMS reduction
rms_reduction_lfp = (rms_before_lfp - rms_after_lfp) / rms_before_lfp * 100
axes[1, 1].hist(rms_reduction_lfp, bins=30, alpha=0.7, color='C1')
axes[1, 1].axvline(np.median(rms_reduction_lfp), color='red', linestyle='--',
                   label=f'Median: {np.median(rms_reduction_lfp):.1f}%')
axes[1, 1].set_xlabel('RMS Reduction (%)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'LFP RMS Reduction Distribution (n={len(artifact_indices)})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_base / '2_statistical_comparison.png', dpi=150)
plt.close()
print(f"Saved: 2_statistical_comparison.png")

# ============ SUMMARY STATISTICS ============

# Create detailed summary text file
summary_file = output_base / 'artifact_summary.txt'
with open(summary_file, 'w') as f:
    f.write(f"Artifact Detection and Removal Summary\n")
    f.write(f"=" * 70 + "\n\n")
    
    f.write(f"RECORDING INFORMATION\n")
    f.write(f"-" * 70 + "\n")
    f.write(f"Recording: {rec_name}\n")
    f.write(f"Shank: {ish}\n")
    f.write(f"Sampling rate: {fs} Hz\n")
    f.write(f"Total duration: {n_timepoints/fs:.2f} s ({n_timepoints/fs/60:.2f} min)\n")
    f.write(f"Number of channels: {n_channels}\n\n")
    
    f.write(f"DETECTION PARAMETERS\n")
    f.write(f"-" * 70 + "\n")
    f.write(f"Chunk time: {chunk_time} s\n")
    f.write(f"Chunk size: {chunk_size} samples\n")
    f.write(f"Detection threshold: {threshold} std deviations\n")
    f.write(f"Detection frequency range: 300-6000 Hz\n\n")
    
    f.write(f"DETECTION RESULTS\n")
    f.write(f"-" * 70 + "\n")
    f.write(f"Total artifacts detected: {len(artifact_indices)}\n")
    f.write(f"Total time removed: {len(artifact_indices)*chunk_size/fs:.2f} s\n")
    f.write(f"Percentage of data removed: {len(artifact_indices)*chunk_size/n_timepoints*100:.3f}%\n")
    f.write(f"Artifact rate: {len(artifact_indices)/(n_timepoints/fs)*60:.2f} artifacts/min\n\n")
    
    f.write(f"RMS STATISTICS (BROADBAND)\n")
    f.write(f"-" * 70 + "\n")
    f.write(f"Mean RMS before: {np.mean(rms_before_raw):.2f} µV\n")
    f.write(f"Mean RMS after: {np.mean(rms_after_raw):.2f} µV\n")
    f.write(f"Median RMS reduction: {np.median(rms_reduction_raw):.2f}%\n")
    f.write(f"Mean RMS reduction: {np.mean(rms_reduction_raw):.2f}%\n")
    f.write(f"RMS reduction range: [{np.min(rms_reduction_raw):.2f}%, {np.max(rms_reduction_raw):.2f}%]\n\n")
    
    f.write(f"RMS STATISTICS (LFP 0.1-100 Hz)\n")
    f.write(f"-" * 70 + "\n")
    f.write(f"Mean RMS before: {np.mean(rms_before_lfp):.2f} µV\n")
    f.write(f"Mean RMS after: {np.mean(rms_after_lfp):.2f} µV\n")
    f.write(f"Median RMS reduction: {np.median(rms_reduction_lfp):.2f}%\n")
    f.write(f"Mean RMS reduction: {np.mean(rms_reduction_lfp):.2f}%\n")
    f.write(f"RMS reduction range: [{np.min(rms_reduction_lfp):.2f}%, {np.max(rms_reduction_lfp):.2f}%]\n\n")
    
    f.write(f"OUTPUT FILES\n")
    f.write(f"-" * 70 + "\n")
    f.write(f"1. Overview plot: 1_artifact_overview.png\n")
    f.write(f"2. Statistical comparison: 2_statistical_comparison.png\n")
    f.write(f"3. Raw data comparisons: raw_data_comparison/ directory\n")
    f.write(f"4. LFP comparisons: lfp_comparison/ directory\n")

print(f"\nSaved summary: {summary_file}")

# Print summary to console
print(f"\n" + "="*70)
print(f"ARTIFACT REMOVAL SUMMARY")
print(f"="*70)
print(f"Total artifacts detected: {len(artifact_indices)}")
print(f"Time removed: {len(artifact_indices)*chunk_size/fs:.2f}s ({len(artifact_indices)*chunk_size/n_timepoints*100:.3f}%)")
print(f"Artifact rate: {len(artifact_indices)/(n_timepoints/fs)*60:.2f} artifacts/min")
print(f"\nBroadband RMS reduction: {np.median(rms_reduction_raw):.1f}% (median)")
print(f"LFP RMS reduction: {np.median(rms_reduction_lfp):.1f}% (median)")
print(f"\nAll results saved to: {output_base}")
print(f"="*70)