import numpy as np
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from tqdm import tqdm

# =====================================================
# SETTINGS
# =====================================================
rec_folder = r"D:\cl\ephys\sleep\CnL42SG_20251112_170949.rec"
session_name = Path(rec_folder).stem.split('.')[0]
shanks = [4,5,6,7]
preproc_params = {
    'reference': 'global',
    'operator': 'median',
    'target_fs': 500,      # Downsample target FS
    'lfp_min': 1,           # LFP band (safer than 0.1 Hz)
    'lfp_max': 200,
}

# Chunking parameters
CHUNK_DURATION = 1200  # seconds - process 60s at a time
# Adjust this based on your RAM - smaller = less memory but slower

# =====================================================
# PROCESS EACH SHANK
# =====================================================
for ish in shanks:
    print("\n" + "=" * 75)
    print(f"PROCESSING SHANK {ish}")
    print("=" * 75 + "\n")
    
    # Load NWB
    rec_path = f"{rec_folder}\\{session_name}sh{ish}.nwb"
    rec = se.NwbRecordingExtractor(rec_path)
    
    # Basic info
    orig_fs = rec.get_sampling_frequency()
    orig_dur = rec.get_total_duration()
    n_channels = rec.get_num_channels()
    print(f"Original FS: {orig_fs} Hz")
    print(f"Duration:    {orig_dur:.2f} sec")
    print(f"Channels:    {n_channels}")
    
    # =====================================================
    # PREPROCESSING PIPELINE
    # =====================================================
    # 1. CAR
    print("\n1. Applying CAR...")
    rec_car = spre.common_reference(
        rec,
        reference=preproc_params['reference'],
        operator=preproc_params['operator']
    )
    
    # 2. RESAMPLE (SpikeInterface includes anti-alias lowpass)
    print("2. Downsampling...")
    rec_ds = spre.resample(rec_car, preproc_params['target_fs'])
    
    # 3. LFP BANDPASS (ONLY ONCE)
    print("3. Bandpass filtering (LFP band)...")
    rec_lfp = spre.bandpass_filter(
        rec_ds,
        freq_min=preproc_params['lfp_min'],
        freq_max=preproc_params['lfp_max']
    )
    
    # =====================================================
    # CHANNEL ORDERING BY DEPTH
    # =====================================================
    channel_ids = rec_lfp.get_channel_ids()
    chan_locs = rec_lfp.get_channel_locations()
    xcoord = chan_locs[:, 0]
    ycoord = chan_locs[:, 1]
    depth_order = np.argsort(ycoord)
    sorted_channels = channel_ids[depth_order]
    print(f"\nSorted channels by depth ({len(sorted_channels)} channels)")
    
    # =====================================================
    # EXTRACT LFP TRACES IN CHUNKS
    # =====================================================
    print("\nExtracting LFP traces in chunks...")
    fs = rec_lfp.get_sampling_frequency()
    duration = rec_lfp.get_total_duration()
    total_samples = rec_lfp.get_num_frames()
    n_channels_sorted = len(sorted_channels)
    
    # Calculate chunk parameters
    chunk_samples = int(CHUNK_DURATION * fs)
    n_chunks = int(np.ceil(total_samples / chunk_samples))
    
    print(f"  Total duration: {duration:.2f} sec")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Total samples: {total_samples}")
    print(f"  Processing in {n_chunks} chunks of {CHUNK_DURATION}s each")
    
    # Pre-allocate output array
    traces = np.zeros((total_samples, n_channels_sorted), dtype='float32')
    
    # Process each chunk with progress bar
    for i_chunk in tqdm(range(n_chunks), desc="Processing chunks"):
        start_sample = i_chunk * chunk_samples
        end_sample = min((i_chunk + 1) * chunk_samples, total_samples)
        
        # Extract chunk
        chunk = rec_lfp.get_traces(
            start_frame=start_sample,
            end_frame=end_sample,
            channel_ids=sorted_channels.tolist()
        )
        
        # Store in pre-allocated array
        traces[start_sample:end_sample, :] = chunk.astype('float32')
    
    print(f"\n  Final LFP shape: {traces.shape} (time x channels)")
    
    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    out_dir = Path(rec_path).parent / "low_freq"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"{session_name}_sh{ish}_lfp_traces.npz"
    
    print(f"\nSaving → {out_file}")
    np.savez(
        out_file,
        traces=traces,
        sampling_rate=fs,
        channel_ids=sorted_channels,
        channel_locations=chan_locs[depth_order],
        xcoord=xcoord[depth_order],
        ycoord=ycoord[depth_order],
        depth_order=depth_order,
        duration=duration,
        n_channels=len(sorted_channels),
        n_timepoints=traces.shape[0],
        original_fs=orig_fs,
        original_duration=orig_dur,
        session_name=session_name,
        shank=ish,
    )
    print("Done!")

print("\n" + "#" * 75)
print("ALL SHANKS PROCESSED")
print("#" * 75)