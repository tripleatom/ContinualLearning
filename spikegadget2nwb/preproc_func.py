import os
import numpy as np
import spikeinterface.preprocessing as spre
import re

def parse_session_info(rec_folder):
    """
    Extract animal ID and session ID from recording folder path.
    
    Args:
        rec_folder (str): Path to recording folder
        
    Returns:
        tuple: (animal_id, session_id)
    """
    # Method 1: Using regex to match specific pattern
    pattern = r'([A-Za-z]+\d+)_(\d{8}_\d{6})\.rec$'
    match = re.search(pattern, rec_folder)
    if match:
        return match.group(1), match.group(2)
    
    # Method 2: Extract from basename using regex
    basename = rec_folder.split('\\')[-1].replace('.rec', '')
    parts = basename.split('_')
    if len(parts) >= 3:
        return parts[0], '_'.join(parts[1:])
    
    return None, None

def get_bad_ch_id(rec, folder, ish,  load_if_exists=True):
    # folder: parent folder for nwb file
    if load_if_exists and os.path.exists(folder / f'bad_ch_id_sh{ish}.npy'):
        bad_ch_id = np.load(folder / f'bad_ch_id_sh{ish}.npy')
    else:
        bad_ch_id, _ = spre.detect_bad_channels(
            rec, num_random_chunks=400, n_neighbors=5, dead_channel_threshold=-0.2)

        np.save(folder / f'bad_ch_id_sh{ish}.npy', bad_ch_id)

    print('Bad channel IDs:', bad_ch_id)
    return bad_ch_id


def rm_artifacts(rec_filtered, folder, ish, bad_ch_id=[], threshold=7, chunk_size=900):
    # folder: parent folder for nwb file
    n_timepoints = rec_filtered.get_num_frames()
    n_channels = rec_filtered.get_num_channels()
    num_chunks = int(np.ceil(n_timepoints / chunk_size))

    # load artifact indices if exists
    if os.path.exists(folder / f'artifact_indices_sh{ish}.npy'):
        artifact_indices = np.load(folder / f'artifact_indices_sh{ish}.npy')
    else:
    # mask artifacts
        norms = np.zeros((num_chunks, n_channels))
        for i in range(num_chunks):
            start = int(i * chunk_size)
            end = int(np.minimum((i + 1) * chunk_size, n_timepoints))
            chunk = rec_filtered.get_traces(start_frame=start, end_frame=end, return_scaled=True)

            norms[i] = np.linalg.norm(chunk, axis=0)

        
        use_it = np.ones(num_chunks, dtype=bool)
        
    # if detect artifacts in a chunk, don't use it and the two neighboring chunks

        for m in range(n_channels):
            # if m in bad_ch_id:
            #     continue
            vals = norms[:, m]

            sigma0 = np.std(vals)
            mean0 = np.mean(vals)

            artifact_indices = np.where(vals > mean0 + threshold * sigma0)[0]

            # check if the first chunk is above threshold, ensure that we don't use negative indices later
            negIndBool = np.where(artifact_indices > 0)[0]

            # check if the last chunk is above threshold to avoid a IndexError
            maxIndBool = np.where(artifact_indices < num_chunks - 1)[0]

            use_it[artifact_indices] = 0
            use_it[artifact_indices[negIndBool] - 1] = 0  # don't use the neighbor chunks either
            use_it[artifact_indices[maxIndBool] + 1] = 0  # don't use the neighbor chunks either

            print("For channel %d: mean=%.2f, stdev=%.2f, chunk size = %d, n_artifacts = %d" % (m, mean0, sigma0, chunk_size, len(artifact_indices)))


        artifact_indices = np.where(use_it == 0)[0]
        artifact_indices = artifact_indices * chunk_size
        # save artifact indices
        np.save(folder / f'artifact_indices_sh{ish}.npy', artifact_indices)

    chunk_time = chunk_size / rec_filtered.get_sampling_frequency()*1000

    if artifact_indices.size > 0:
        rec_rm_artifacts = spre.remove_artifacts(rec_filtered, list_triggers=artifact_indices, ms_before=0, ms_after=chunk_time)

    else:
        rec_rm_artifacts = rec_filtered


    return rec_rm_artifacts
