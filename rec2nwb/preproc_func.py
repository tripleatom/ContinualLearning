import os
import re
import numpy as np
import spikeinterface.preprocessing as spre

def parse_session_info(rec_folder: str) -> tuple:
    r"""
    Extract animal ID, session ID, and folder name from a recording folder path.

    Supports folder names such as:
      1. \\10.129.151.108\xieluanlabs\xl_cl\ephys\CnL14_20240915_161250.rec
      2. \\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CNL35\CNL35_250305_191757

    Args:
        rec_folder (str): Path to the recording folder.
    
    Returns:
        tuple: (animal_id, session_id, folder_name)
    """
    rec_folder = str(rec_folder)
    basename = os.path.basename(rec_folder.rstrip("\\/"))
    pattern = r'([A-Za-z]+\d+)_(\d{6,8}_\d{6})(?:\.rec)?$'
    match = re.search(pattern, basename)
    if match:
        animal_id, session_id = match.groups()
        return animal_id, session_id, f"{animal_id}_{session_id}"

    # Fallback: remove '.rec' if present and split by underscore
    parts = basename.replace('.rec', '').split('_')
    if len(parts) >= 2:
        animal_id = parts[0]
        session_id = '_'.join(parts[1:])
        return animal_id, session_id, f"{animal_id}_{session_id}"

    raise ValueError("Recording folder name doesn't match the expected format.")


def get_bad_ch_id(rec, folder, ish, load_if_exists=True):
    r"""
    Retrieve or detect bad channel IDs.

    Args:
        rec: Recording object.
        folder: Parent folder for the NWB file.
        ish: Shank identifier.
        load_if_exists (bool): If True, load from file if available.

    Returns:
        np.ndarray: Array of bad channel IDs.
    """
    bad_ch_file = folder / f'bad_ch_id_sh{ish}.npy'
    if load_if_exists and os.path.exists(bad_ch_file):
        bad_ch_id = np.load(bad_ch_file)
    else:
        bad_ch_id, _ = spre.detect_bad_channels(
            rec, num_random_chunks=400, n_neighbors=5, dead_channel_threshold=-0.2
        )
        np.save(bad_ch_file, bad_ch_id)

    print('Bad channel IDs:', bad_ch_id)
    return bad_ch_id


def rm_artifacts(rec_filtered, folder, ish, threshold=6, chunk_time=0.05, overwrite=False):
    r"""
    Remove artifacts from the filtered recording.

    Args:
        rec_filtered: The filtered recording object.
        folder: Parent folder for saving results.
        ish: Shank identifier.
        bad_ch_id: List of bad channel IDs (optional).
        threshold: Threshold for artifact detection.
        chunk_time: Chunk size in seconds.
        overwrite: If True, recompute artifact indices even if they already exist.

    Returns:
        Recording object with artifacts removed.
    """
    fs = rec_filtered.get_sampling_frequency()
    chunk_size = int(chunk_time * fs)
    n_timepoints = rec_filtered.get_num_frames()
    n_channels = rec_filtered.get_num_channels()
    num_chunks = int(np.ceil(n_timepoints / chunk_size))

    artifact_file = folder / f'artifact_indices_sh{ish}.npy'
    if not overwrite and os.path.exists(artifact_file):
        artifact_indices = np.load(artifact_file)
    else:
        # Compute norm of traces per chunk and channel.
        norms = np.zeros((num_chunks, n_channels))
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_timepoints)
            chunk = rec_filtered.get_traces(start_frame=start, end_frame=end, return_scaled=True)
            norms[i] = np.linalg.norm(chunk, axis=0)

        # Determine which chunks to discard based on threshold.
        use_chunk = np.ones(num_chunks, dtype=bool)
        for ch in range(n_channels):
            vals = norms[:, ch]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            # Identify chunks with high norm (artifacts)
            artifact_chunks = np.where(vals > mean_val + threshold * std_val)[0]

            # Avoid using artifact chunk and its neighbors.
            if artifact_chunks.size > 0:
                use_chunk[artifact_chunks] = False
                use_chunk[artifact_chunks[artifact_chunks > 0] - 1] = False
                use_chunk[artifact_chunks[artifact_chunks < num_chunks - 1] + 1] = False

            print(f"For channel {ch}: mean={mean_val:.2f}, stdev={std_val:.2f}, "
                  f"chunk size = {chunk_size}, n_artifacts = {len(artifact_chunks)}")

        # Convert chunk indices to timepoints.
        artifact_indices = np.where(~use_chunk)[0] * chunk_size
        np.save(artifact_file, artifact_indices)

    # Convert chunk size to milliseconds.
    chunk_time_ms = chunk_size / fs * 1000
    if artifact_indices.size > 0:
        #FIXME: how this handles the connection point. will this set all channels to 0?
        # mode“zeros”, “linear”, “cubic”, “average”, “median”, default: “zeros”
        rec_rm_artifacts = spre.remove_artifacts(
            rec_filtered, list_triggers=artifact_indices, ms_before=0, ms_after=chunk_time_ms,
            mode='linear'
        )
    else:
        rec_rm_artifacts = rec_filtered

    return rec_rm_artifacts