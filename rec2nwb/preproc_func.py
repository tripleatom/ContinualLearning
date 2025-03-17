import os
import numpy as np
import spikeinterface.preprocessing as spre
import re

import os
import re

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
    # Get the basename (folder name) and remove any trailing path separators
    rec_folder = str(rec_folder)
    basename = os.path.basename(rec_folder.rstrip("\\/"))
    
    
    # Regex pattern:
    # - ([A-Za-z]+\d+): captures animal ID (e.g., CnL14 or CNL35)
    # - _(\d{6,8}_\d{6}): captures session ID (date_time, e.g., 250305_191757)
    # - (?:\.rec)?$ : optionally matches a trailing '.rec'
    pattern = r'([A-Za-z]+\d+)_(\d{6,8}_\d{6})(?:\.rec)?$'
    match = re.search(pattern, basename)
    if match:
        animal_id = match.group(1)
        session_id = match.group(2)
        folder_name = f"{animal_id}_{session_id}"
        return animal_id, session_id, folder_name
    
    # Fallback: remove '.rec' if present, then split by underscore
    cleaned = basename.replace('.rec', '')
    parts = cleaned.split('_')
    if len(parts) >= 2:
        animal_id = parts[0]
        session_id = '_'.join(parts[1:])
        folder_name = f"{animal_id}_{session_id}"
        return animal_id, session_id, folder_name

    raise ValueError("Recording folder name doesn't match the expected format.")



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
