from pathlib import Path
import numpy as np
import pickle
from scipy import signal
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from tqdm import tqdm

# === CONFIGURATION ===
base_folder = r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\3cms_2\data_251103_165302"
session_name = base_folder.split('\\')[-1]
output_folder = base_folder

# Shank assignments
brain_shanks = [0, 1, 2, 3]
spinal_shanks = [4, 5, 6, 7]

# Preprocessing parameters (matching your pipeline)
preproc_params = {
    'car_reference': 'global',  # Common average reference
    'car_operator': 'median',
    'lfp_filter_min': 0.1,  # Hz
    'lfp_filter_max': 300,  # Hz
    'target_fs': 1000,  # Target sampling rate for LFP
}

# Coherence parameters
coherence_params = {
    'fs': None,  # Will be set to target_fs after preprocessing
    'nperseg': 2048,  # Window length for coherence calculation
    'noverlap': None,  # Will be set to nperseg // 2
    'nfft': None,  # Will be set to nperseg
    'freq_range': (0.5, 100),  # Frequency range to keep
}

# === LOAD AND PREPROCESS NWB FILES ===
print(f"Loading and preprocessing NWB files from: {base_folder}")

# Storage for all preprocessed LFP data
all_lfp_data = {}
all_channel_info = {}
sampling_rate = None
total_channels = 0

# Load and preprocess each shank's NWB file
for shank_id in brain_shanks + spinal_shanks:
    nwb_file = Path(base_folder) / f"{session_name}sh{shank_id}.nwb"
    
    if not nwb_file.exists():
        print(f"WARNING: File not found: {nwb_file}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing Shank {shank_id}: {nwb_file.name}")
    print(f"{'='*60}")
    
    # Load recording
    rec = se.NwbRecordingExtractor(str(nwb_file))
    pos = rec.get_channel_locations()
    
    # Store original recording info
    original_fs = rec.get_sampling_frequency()
    original_duration = rec.get_total_duration()
    
    print(f"Original sampling rate: {original_fs} Hz")
    print(f"Original duration: {original_duration} s")
    print(f"Number of channels: {rec.get_num_channels()}")
    
    # Apply preprocessing pipeline efficiently
    print("\n=== Preprocessing ===")
    # 1. Common average reference
    print("1. Applying common average reference...")
    rec_car = spre.common_reference(rec, reference=preproc_params['car_reference'], 
                                    operator=preproc_params['car_operator'])
    
    # 2. Bandpass filter for LFP range
    print("2. Applying bandpass filter...")
    rec_filtered = spre.bandpass_filter(rec_car, 
                                        freq_min=preproc_params['lfp_filter_min'],
                                        freq_max=preproc_params['lfp_filter_max'])
    
    # 3. Downsample to target frequency
    print("3. Downsampling...")
    target_fs = preproc_params['target_fs']
    rec_downsampled = spre.resample(rec_filtered, resample_rate=target_fs)
    
    print(f"Downsampled sampling rate: {rec_downsampled.get_sampling_frequency()} Hz")
    
    # Get traces as numpy array
    # Check SpikeInterface version - newer versions return (n_samples, n_channels)
    print("4. Loading preprocessed data into memory...")
    traces = rec_downsampled.get_traces(return_scaled=True)
    
    print(f"   Raw traces shape from get_traces(): {traces.shape}")
    
    # Ensure we have (n_samples, n_channels) format
    n_channels = rec_downsampled.get_num_channels()
    if traces.shape[0] == n_channels:
        # Got (n_channels, n_samples), need to transpose
        lfp_data = traces.T
        print(f"   Transposed to: {lfp_data.shape}")
    else:
        # Already (n_samples, n_channels)
        lfp_data = traces
        print(f"   Already in correct format: {lfp_data.shape}")
    
    # Get sampling rate (should be same for all shanks after preprocessing)
    current_fs = rec_downsampled.get_sampling_frequency()
    if sampling_rate is None:
        sampling_rate = current_fs
    else:
        assert np.isclose(sampling_rate, current_fs), \
               f"Sampling rate mismatch for shank {shank_id}"
    
    # Store data
    all_lfp_data[shank_id] = lfp_data
    
    # --- Save physical electrode positions ---
    # pos: (n_channels, 2) containing (x, y)
    pos = rec.get_channel_locations()   
    all_channel_info[shank_id] = {
        'pos_xy': pos,                                     # <--- Save (x, y)
        'n_channels': lfp_data.shape[1],
        'channel_ids': rec_downsampled.get_channel_ids().tolist(),
    }
    
    total_channels += lfp_data.shape[1]
    
    print(f"\n✓ Preprocessed LFP shape: {lfp_data.shape} (n_samples, n_channels)")
    print(f"✓ Duration: {lfp_data.shape[0] / sampling_rate:.1f} s")
    print(f"✓ Number of channels: {lfp_data.shape[1]}")

print(f"\n{'='*60}")
print(f"Total channels loaded: {total_channels}")
print(f"Final sampling rate: {sampling_rate} Hz")
print(f"{'='*60}")

# Update coherence parameters
coherence_params['fs'] = sampling_rate
coherence_params['noverlap'] = coherence_params['nperseg'] // 2
coherence_params['nfft'] = coherence_params['nperseg']

# === ORGANIZE CHANNELS BY SHANK ===
print("\nOrganizing channels by shank...")

# Channel info already organized by shank from loading
shank_channels = {}
for shank_id in brain_shanks + spinal_shanks:
    if shank_id not in all_lfp_data:
        print(f"  Shank {shank_id}: NOT LOADED")
        shank_channels[shank_id] = {
            'indices': [],
            'channel_ids': [],
        }
        continue
    
    n_channels = all_channel_info[shank_id]['n_channels']
    shank_channels[shank_id] = {
        'indices': list(range(n_channels)),  # Local indices within shank
        'channel_ids': all_channel_info[shank_id]['channel_ids'],
    }
    print(f"  Shank {shank_id}: {n_channels} channels")

# === CALCULATE COHERENCE ===
print("\nCalculating coherence between brain and spinal channels...")

# Prepare results storage
coherence_results = {
    'brain_shank': [],
    'brain_channel_idx': [],
    'brain_channel_id': [],
    'brain_channel_position': [],  # Store position info if available
    'spinal_shank': [],
    'spinal_channel_idx': [],
    'spinal_channel_id': [],
    'spinal_channel_position': [],  # Store position info if available
    'frequencies': None,
    'coherence': [],
    'phase': [],
    'parameters': coherence_params.copy(),
    'preprocessing': preproc_params.copy(),
    'metadata': {
        'base_folder': str(base_folder),
        'session_name': session_name,
        'sampling_rate': sampling_rate,
        'brain_shanks': brain_shanks,
        'spinal_shanks': spinal_shanks,
        'channel_info': all_channel_info,  # Store all channel metadata
    }
}

# Count total pairs for progress bar
total_pairs = sum(len(shank_channels[bs]['indices']) for bs in brain_shanks if bs in all_lfp_data) * \
              sum(len(shank_channels[ss]['indices']) for ss in spinal_shanks if ss in all_lfp_data)

print(f"Total channel pairs to compute: {total_pairs}")

# Calculate coherence for all brain-spinal pairs
with tqdm(total=total_pairs, desc="Computing coherence") as pbar:
    for brain_shank in brain_shanks:
        if brain_shank not in all_lfp_data:
            continue
            
        brain_ch_indices = shank_channels[brain_shank]['indices']
        brain_ch_ids = shank_channels[brain_shank]['channel_ids']
        brain_lfp = all_lfp_data[brain_shank]
        
        for spinal_shank in spinal_shanks:
            if spinal_shank not in all_lfp_data:
                continue
                
            spinal_ch_indices = shank_channels[spinal_shank]['indices']
            spinal_ch_ids = shank_channels[spinal_shank]['channel_ids']
            spinal_lfp = all_lfp_data[spinal_shank]
            
            # For each brain channel
            for brain_idx, brain_ch_id in zip(brain_ch_indices, brain_ch_ids):
                brain_signal = brain_lfp[:, brain_idx]
                
                # For each spinal channel
                for spinal_idx, spinal_ch_id in zip(spinal_ch_indices, spinal_ch_ids):
                    spinal_signal = spinal_lfp[:, spinal_idx]
                    
                    # Calculate coherence
                    freqs, Cxy = signal.coherence(
                        brain_signal,
                        spinal_signal,
                        fs=coherence_params['fs'],
                        nperseg=coherence_params['nperseg'],
                        noverlap=coherence_params['noverlap'],
                        nfft=coherence_params['nfft']
                    )
                    
                    # Calculate cross-spectral phase
                    _, Pxy = signal.csd(
                        brain_signal,
                        spinal_signal,
                        fs=coherence_params['fs'],
                        nperseg=coherence_params['nperseg'],
                        noverlap=coherence_params['noverlap'],
                        nfft=coherence_params['nfft']
                    )
                    phase = np.angle(Pxy)
                    
                    # Filter to desired frequency range
                    if coherence_results['frequencies'] is None:
                        freq_mask = (freqs >= coherence_params['freq_range'][0]) & \
                                   (freqs <= coherence_params['freq_range'][1])
                        coherence_results['frequencies'] = freqs[freq_mask]
                    
                    # Store results
                    coherence_results['brain_shank'].append(brain_shank)
                    coherence_results['brain_channel_idx'].append(brain_idx)
                    coherence_results['brain_channel_id'].append(brain_ch_id)
                    coherence_results['spinal_shank'].append(spinal_shank)
                    coherence_results['spinal_channel_idx'].append(spinal_idx)
                    coherence_results['spinal_channel_id'].append(spinal_ch_id)
                    coherence_results['coherence'].append(Cxy[freq_mask])
                    coherence_results['phase'].append(phase[freq_mask])

                    # Store physical channel position
                    brain_xy = all_channel_info[brain_shank]['pos_xy'][brain_idx]
                    spinal_xy = all_channel_info[spinal_shank]['pos_xy'][spinal_idx]

                    coherence_results['brain_channel_position'].append(brain_xy)
                    coherence_results['spinal_channel_position'].append(spinal_xy)

                    
                    pbar.update(1)

# Convert lists to arrays for easier manipulation
coherence_results['brain_shank'] = np.array(coherence_results['brain_shank'])
coherence_results['brain_channel_idx'] = np.array(coherence_results['brain_channel_idx'])
coherence_results['brain_channel_id'] = np.array(coherence_results['brain_channel_id'])
coherence_results['brain_channel_position'] = np.array(coherence_results['brain_channel_position'], dtype=object)
coherence_results['spinal_shank'] = np.array(coherence_results['spinal_shank'])
coherence_results['spinal_channel_idx'] = np.array(coherence_results['spinal_channel_idx'])
coherence_results['spinal_channel_id'] = np.array(coherence_results['spinal_channel_id'])
coherence_results['spinal_channel_position'] = np.array(coherence_results['spinal_channel_position'], dtype=object)
coherence_results['coherence'] = np.array(coherence_results['coherence'])  # Shape: (n_pairs, n_freqs)
coherence_results['phase'] = np.array(coherence_results['phase'])  # Shape: (n_pairs, n_freqs)

print(f"\nCoherence calculation complete!")
print(f"  Total pairs computed: {len(coherence_results['coherence'])}")
print(f"  Frequency points: {len(coherence_results['frequencies'])}")
print(f"  Coherence array shape: {coherence_results['coherence'].shape}")

# === SAVE RESULTS ===
output_path = Path(output_folder) / f"{session_name}_brain_spinal_coherence.pkl"
print(f"\nSaving results to: {output_path}")

with open(output_path, 'wb') as f:
    pickle.dump(coherence_results, f)

print(f"Results saved successfully!")

# === PRINT SUMMARY ===
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Brain shanks: {brain_shanks}")
print(f"Spinal shanks: {spinal_shanks}")
print(f"\nChannels per shank:")
for shank_id in brain_shanks + spinal_shanks:
    n_channels = len(shank_channels[shank_id]['indices'])
    print(f"  Shank {shank_id}: {n_channels} channels")
print(f"\nTotal brain-spinal pairs: {len(coherence_results['coherence'])}")
print(f"Frequency range: {coherence_params['freq_range'][0]}-{coherence_params['freq_range'][1]} Hz")
print(f"Frequency resolution: {coherence_results['frequencies'][1] - coherence_results['frequencies'][0]:.3f} Hz")

# Show example of how to access data
print("\n" + "="*60)
print("HOW TO USE THE SAVED DATA")
print("="*60)
print("""
# Load the data:
import pickle
with open('..._brain_spinal_coherence.pkl', 'rb') as f:
    results = pickle.load(f)

# Access coherence for a specific pair:
pair_idx = 0
brain_shank = results['brain_shank'][pair_idx]
brain_ch = results['brain_channel_id'][pair_idx]
spinal_shank = results['spinal_shank'][pair_idx]
spinal_ch = results['spinal_channel_id'][pair_idx]
coherence = results['coherence'][pair_idx]  # Coherence vs frequency
phase = results['phase'][pair_idx]  # Phase vs frequency
freqs = results['frequencies']

# Filter for specific shanks or channels:
import numpy as np
mask = (results['brain_shank'] == 0) & (results['spinal_shank'] == 4)
pairs_0_4 = np.where(mask)[0]

# Average coherence across all pairs:
mean_coherence = np.mean(results['coherence'], axis=0)
""")

print("\nDone!")