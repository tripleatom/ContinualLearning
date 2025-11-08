import os
import numpy as np
import spikeinterface.preprocessing as spre
from numba import jit


@jit(nopython=True, parallel=True)
def compute_norms_numba(data, chunk_size):
    """
    Fast chunk norm computation with Numba.
    
    Args:
        data: 2D array (n_samples, n_channels)
        chunk_size: Size of chunks
    
    Returns:
        norms: 2D array (n_chunks, n_channels)
    """
    n_samples, n_channels = data.shape
    n_chunks = n_samples // chunk_size
    norms = np.zeros((n_chunks, n_channels))
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        for ch in range(n_channels):
            norm_val = 0.0
            for i in range(start, end):
                norm_val += data[i, ch] ** 2
            norms[chunk_idx, ch] = np.sqrt(norm_val)
    
    return norms

def rm_artifacts(rec_raw, folder, ish, threshold=6, chunk_time=0.02, 
                      detect_freq_range=(300, 6000), overwrite=False, 
                      block_size_sec=10):
    """
    Fast artifact removal using vectorized operations.
    """
    fs = rec_raw.get_sampling_frequency()
    chunk_size = int(chunk_time * fs)
    block_size = int(block_size_sec * fs)
    
    # Highpass filter for detection
    rec_detect = spre.bandpass_filter(rec_raw, freq_min=detect_freq_range[0], 
                                      freq_max=detect_freq_range[1])
    
    n_timepoints = rec_detect.get_num_frames()
    n_channels = rec_detect.get_num_channels()
    num_chunks = int(np.ceil(n_timepoints / chunk_size))

    artifact_file = folder / f'artifact_indices_sh{ish}_{chunk_time}.npy'

    if not overwrite and os.path.exists(artifact_file):
        print(f"Loading existing artifact indices from {artifact_file}")
        artifact_indices = np.load(artifact_file)
    else:
        print("Computing artifact indices with vectorized approach...")
        norms = np.zeros((num_chunks, n_channels))
        
        # Process in larger blocks to reduce I/O overhead
        num_blocks = int(np.ceil(n_timepoints / block_size))
        
        for block_idx in range(num_blocks):
            start_frame = block_idx * block_size
            end_frame = min((block_idx + 1) * block_size, n_timepoints)
            
            # Load one large block
            block_data = rec_detect.get_traces(start_frame=start_frame, 
                                               end_frame=end_frame, 
                                               return_scaled=True)
            
            # Compute start and end chunk indices for this block
            start_chunk = start_frame // chunk_size
            end_chunk = min((end_frame + chunk_size - 1) // chunk_size, num_chunks)
            
            # Vectorized chunk norm computation
            for chunk_idx in range(start_chunk, end_chunk):
                chunk_start = chunk_idx * chunk_size - start_frame
                chunk_end = min((chunk_idx + 1) * chunk_size - start_frame, 
                               block_data.shape[0])
                
                if chunk_start >= 0 and chunk_start < block_data.shape[0]:
                    chunk_data = block_data[chunk_start:chunk_end, :]
                    norms[chunk_idx] = np.linalg.norm(chunk_data, axis=0)
            
            print(f"Processed block {block_idx+1}/{num_blocks}")
        
        # Vectorized artifact detection across all channels
        use_chunk = np.ones(num_chunks, dtype=bool)
        
        for ch in range(n_channels):
            vals = norms[:, ch]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            
            # Find artifacts
            artifact_mask = vals > mean_val + threshold * std_val
            artifact_chunks = np.where(artifact_mask)[0]
            
            if artifact_chunks.size > 0:
                # Vectorized neighbor marking
                all_bad = np.concatenate([
                    artifact_chunks - 1,
                    artifact_chunks,
                    artifact_chunks + 1
                ])
                all_bad = all_bad[(all_bad >= 0) & (all_bad < num_chunks)]
                use_chunk[all_bad] = False
                
                print(f"Ch {ch}: n_artifacts={len(artifact_chunks)}")
        
        artifact_indices = np.where(~use_chunk)[0] * chunk_size
        np.save(artifact_file, artifact_indices)
        print(f"Total chunks removed: {(~use_chunk).sum()}/{num_chunks}")
    
    chunk_time_ms = chunk_size / fs * 1000
    if artifact_indices.size > 0:
        rec_clean = spre.remove_artifacts(
            rec_raw, list_triggers=artifact_indices, 
            ms_before=0, ms_after=chunk_time_ms,
            mode='cubic'
        )
    else:
        rec_clean = rec_raw
    
    return rec_clean