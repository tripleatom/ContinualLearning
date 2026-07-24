import os
import errno
import numpy as np
import spikeinterface.preprocessing as spre

from server_fallback import mirror_on_backup_server


def compute_chunk_norms(data, chunk_size):
    """
    L2 norm of each (chunk, channel).

    Args:
        data: 2D array (n_samples, n_channels)
        chunk_size: number of samples per chunk

    Returns:
        norms: 2D array (n_chunks, n_channels). Trailing samples that do not
        fill a complete chunk are dropped.
    """
    n_samples, n_channels = data.shape
    n_chunks = n_samples // chunk_size
    trimmed = data[:n_chunks * chunk_size]
    return np.linalg.norm(trimmed.reshape(n_chunks, chunk_size, n_channels), axis=1)


def rm_artifacts(rec_raw, folder, ish, threshold=6, chunk_time=0.02,
                 detect_freq_range=(300, 6000), overwrite=False,
                 block_size_sec=10):
    """
    Fast artifact removal using vectorized operations.
    """
    fs = rec_raw.get_sampling_frequency()
    chunk_size = int(chunk_time * fs)
    # Snap the block size to a whole number of chunks so every block starts on a
    # chunk boundary; this keeps each block's norms aligned to the global chunk
    # grid (and avoids partially-computed norms for chunks that would otherwise
    # straddle a block boundary).
    block_size = max(chunk_size, (int(block_size_sec * fs) // chunk_size) * chunk_size)

    # Highpass filter for detection
    rec_detect = spre.bandpass_filter(rec_raw, freq_min=detect_freq_range[0],
                                      freq_max=detect_freq_range[1])

    n_timepoints = rec_detect.get_num_frames()
    n_channels = rec_detect.get_num_channels()
    num_chunks = int(np.ceil(n_timepoints / chunk_size))

    artifact_file = folder / f'artifact_indices_sh{ish}_{chunk_time}_{threshold}.npy'

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

            # Load one large block and compute all its chunk norms at once
            block_data = rec_detect.get_traces(start_frame=start_frame,
                                               end_frame=end_frame,
                                               return_scaled=True)
            block_norms = compute_chunk_norms(block_data, chunk_size)

            start_chunk = start_frame // chunk_size
            norms[start_chunk:start_chunk + block_norms.shape[0]] = block_norms

            print(f"Processed block {block_idx + 1}/{num_blocks}")

        # Vectorized artifact detection across all channels at once: a chunk is
        # discarded if any channel exceeds mean + threshold * std for that channel.
        chunk_thresholds = norms.mean(axis=0) + threshold * norms.std(axis=0)
        use_chunk = ~np.any(norms > chunk_thresholds, axis=1)

        artifact_indices = np.where(~use_chunk)[0] * chunk_size
        try:
            np.save(artifact_file, artifact_indices)
        except OSError as e:
            if e.errno != errno.ENOSPC:
                raise
            backup_folder = mirror_on_backup_server(folder)
            if backup_folder is None:
                raise
            backup_folder.mkdir(parents=True, exist_ok=True)
            artifact_file = backup_folder / artifact_file.name
            print(f"Out of space while saving - retrying on backup server: {artifact_file}")
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
