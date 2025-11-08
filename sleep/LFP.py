import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from process_func.recording_proc import rm_artifacts
from pathlib import Path

rec_file = "/Volumes/xieluanlabs/xl_cl/ephys/sleep/CnL39SG/CnL39SG_20251102_210043.rec/CnL39SG_20251102_210043sh0.nwb"
rec = se.NwbRecordingExtractor(rec_file)

# Extract shank number from filename
ish = int(rec_file.split('sh')[-1].split('.')[0])  # Extracts 0 from 'sh0.nwb'
folder = Path(rec_file).parent  # Get parent folder for saving artifacts

rec_clean = rm_artifacts(rec, folder, ish, threshold=6, chunk_time=0.02, overwrite=True)
# Common reference FIRST (on raw data)
rec_car = spre.common_reference(rec_clean, reference='global', operator='median')

# Bandpass filter for spike band (to detect artifacts)
rec_filtered = spre.bandpass_filter(rec_car, freq_min=0.1, freq_max=100, dtype="float32")

# Remove artifacts (operates on spike-band filtered data)
print("Removing artifacts...")
# rec_clean = rm_artifacts(rec_filtered, folder, ish, threshold=6, chunk_time=0.02, overwrite=False)
# rec_clean = rec_filtered
# Now downsample the CLEAN data
print("Downsampling...")
target_fs = 1000  # 1kHz is good for LFP
rec_downsampled = spre.resample(rec_filtered, target_fs)

# Then bandpass filter for LFP range
print("Applying LFP filter...")
rec_lfp = spre.bandpass_filter(rec_downsampled, freq_min=0.1, freq_max=300, dtype="float32")

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

print(f"Original sampling rate: {rec.get_sampling_frequency()} Hz")
print(f"Downsampled to: {fs} Hz")
print(f"Number of channels: {n_channels}")
print(f"Total duration: {duration:.2f} seconds")
print(f"Depth range: {ycoord_sorted.min():.1f} to {ycoord_sorted.max():.1f} μm (shallow to deep)")

# === PLOT LFP TRACES ORDERED BY DEPTH ===
start_time = 10  # Start at 10 seconds to avoid artifacts
window_duration = 10

# Get traces using sorted channel IDs
traces = rec_lfp.get_traces(start_frame=int(start_time * fs), 
                             end_frame=int((start_time + window_duration) * fs),
                             channel_ids=sorted_channel_ids.tolist())
time_vector = np.arange(traces.shape[0]) / fs + start_time

plt.figure(figsize=(15, 10))

# Calculate offset based on trace standard deviation
seg_stds = np.std(traces, axis=0)
offset_multiplier = np.median(seg_stds) * 15
offsets = -np.arange(n_channels) * offset_multiplier

# Plot all channels ordered by depth (shallow at top, deep at bottom)
for i in range(n_channels):
    plt.plot(time_vector, traces[:, i] + offsets[i], linewidth=0.5, alpha=0.8, color='black')
    
    # Add depth labels on the right
    plt.text(time_vector[-1] + 0.2, offsets[i], f'{ycoord_sorted[i]:.0f}μm', 
            fontsize=8, va='center')

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Depth (0μm → deep)', fontsize=12)
plt.title(f'LFP by Depth: {window_duration}s window (0.1-100 Hz, {fs} Hz, median CAR, artifacts removed)', fontsize=14)
plt.xlim([start_time, start_time + window_duration])
plt.ylim([offsets.min() - offset_multiplier, offsets.max() + offset_multiplier])

# Remove y-axis ticks and spines for cleaner look
plt.yticks([])
ax = plt.gca()
ax.spines['left'].set_visible(False)

# Add depth indicators on left
plt.text(start_time - 0.5, offsets[0], f'0μm (shallow)', 
        fontsize=10, va='center', ha='right', weight='bold')
plt.text(start_time - 0.5, offsets[-1], f'{ycoord_sorted[-1]:.0f}μm (deep)', 
        fontsize=10, va='center', ha='right', weight='bold')

plt.tight_layout()
plt.savefig('lfp_by_depth_clean.png', dpi=150)
plt.show()

# === HEATMAP VIEW ===
plt.figure(figsize=(15, 8))
im = plt.imshow(traces.T, aspect='auto', cmap='RdBu_r',
                extent=[time_vector[0], time_vector[-1], n_channels-0.5, -0.5],
                interpolation='bilinear')

# Set y-axis to show depths
depth_ticks = np.linspace(0, n_channels-1, min(10, n_channels), dtype=int)
plt.yticks(depth_ticks, [f'{ycoord_sorted[i]:.0f}μm' for i in depth_ticks])

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Depth (0μm → deep)', fontsize=12)
plt.title(f'LFP Heatmap by Depth: {window_duration}s window (artifacts removed)', fontsize=14)

cbar = plt.colorbar(im)
cbar.set_label('Amplitude (μV)', fontsize=11)

plt.tight_layout()
plt.savefig('lfp_heatmap_by_depth_clean.png', dpi=150)
plt.show()