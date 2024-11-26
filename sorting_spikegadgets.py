from spikegadget2nwb.read_spikegadget import get_ephys_folder
import spikeinterface.extractors as se
import os
import numpy as np
import spikeinterface.preprocessing as spre
import time
import spikeinterface as si
from pathlib import Path
from Timer import Timer


def get_bad_ch_id(rec, folder, load_if_exists=True):
    if load_if_exists and os.path.exists(folder / f'bad_ch_id_sh{ish}.npy'):
        bad_ch_id = np.load(folder / f'bad_ch_id_sh{ish}.npy')
    else:
        bad_ch_id, _ = spre.detect_bad_channels(
            rec, num_random_chunks=400, n_neighbors=5, dead_channel_threshold=-0.2)

        np.save(folder / f'bad_ch_id_sh{ish}.npy', bad_ch_id)

    print('Bad channel IDs:', bad_ch_id)
    return bad_ch_id


subject_id = "CnL14"
exp_date = "20241004"
exp_time = "153555"
session_description = subject_id + '_' + exp_date + '_' + exp_time + '.rec'
ephys_folder = Path(r"D:\cl\rf_reconstruction\head_fixed")
folder = ephys_folder / session_description
# folder = rec_folder / Path(subject_id + '_' +
#                             exp_date + '_' + exp_time + '.mountainsort')


ishs = [0, 1, 2, 3]

for ish in ishs:
    tt = Timer(f'shank {ish}')
    print(f'Processing shank {ish}...')
    nwb_file = folder / (session_description + f'sh{ish}.nwb')
    print(nwb_file)
    rec = se.NwbRecordingExtractor(nwb_file)

    rec_filtered = spre.bandpass_filter(rec, freq_min=300, freq_max=6000)
    bad_ch_id = get_bad_ch_id(rec_filtered, folder)
    remaining_ch = np.array(
        [ch for ch in rec.get_channel_ids() if ch not in bad_ch_id])

    # export remaining channel ids to a npy file
    np.save(os.path.join(folder, f'remaining_ch_sh{ish}.npy'), remaining_ch)

    threshold = 7
    chunk_size = 900
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
            chunk = rec_filtered.get_traces(
                start_frame=start, end_frame=end, return_scaled=True)

            norms[i] = np.linalg.norm(chunk, axis=0)

        use_it = np.ones(num_chunks, dtype=bool)
    # if detect artifacts in a chunk, don't use it and the two neighboring chunks

        for m in range(n_channels):
            if m in bad_ch_id:
                continue
            vals = norms[:, m]

            sigma0 = np.std(vals)
            mean0 = np.mean(vals)

            artifact_indices = np.where(vals > mean0 + threshold * sigma0)[0]

            # check if the first chunk is above threshold, ensure that we don't use negative indices later
            negIndBool = np.where(artifact_indices > 0)[0]

            # check if the last chunk is above threshold to avoid a IndexError
            maxIndBool = np.where(artifact_indices < num_chunks - 1)[0]

            use_it[artifact_indices] = 0
            # don't use the neighbor chunks either
            use_it[artifact_indices[negIndBool] - 1] = 0
            # don't use the neighbor chunks either
            use_it[artifact_indices[maxIndBool] + 1] = 0

            print("For channel %d: mean=%.2f, stdev=%.2f, chunk size = %d, n_artifacts = %d" % (
                m, mean0, sigma0, chunk_size, len(artifact_indices)))

        artifact_indices = np.where(use_it == 0)[0]
        artifact_indices = artifact_indices * chunk_size
        # save artifact indices
        np.save(folder / f'artifact_indices_sh{ish}.npy', artifact_indices)

    chunk_time = chunk_size / rec.get_sampling_frequency()*1000

    if artifact_indices.size > 0:
        rec_rm_artifacts = spre.remove_artifacts(
            rec_filtered, list_triggers=artifact_indices, ms_before=0, ms_after=chunk_time)

    else:
        rec_rm_artifacts = rec_filtered

    # rec_clean = rec_rm_artifacts.channel_slice(remaining_ch)
    rec_ref = spre.common_reference(rec_rm_artifacts, reference='global', operator='average')
    recording_whitened = spre.whiten(rec_ref, dtype='float32')

    import mountainsort5 as ms5
    import json

    experiment_length = rec_ref.get_duration() / 60  # in minutes

    threshold = 5.5
    phase1_detect_time_radius_msec = .4

    if experiment_length < 25:
        sorting_params = ms5.Scheme1SortingParameters(
            detect_time_radius_msec=phase1_detect_time_radius_msec, detect_threshold=threshold)
        sorting = ms5.sorting_scheme1(
            recording_whitened, sorting_parameters=sorting_params)
        
        assert isinstance(sorting, si.BaseSorting)
    else:
        sorting_params = ms5.Scheme2SortingParameters(
            phase1_detect_threshold=threshold, detect_threshold=threshold,
            phase1_detect_channel_radius=100, detect_channel_radius=100, phase1_detect_time_radius_msec=phase1_detect_time_radius_msec, training_duration_sec=25*60,
            training_recording_sampling_mode='uniform')
        sorting = ms5.sorting_scheme2(
            recording=recording_whitened, sorting_parameters=sorting_params)

    current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    folder_name = f'sorting_results_sh{ish}_' + current_time
    sort_out_folder = folder / folder_name
    if not os.path.exists(sort_out_folder):
        os.makedirs(sort_out_folder)

    # write a into json file: sorting_params.json
    with open(sort_out_folder / 'sorting_params.json', 'w') as f:
        json.dump(sorting_params.__dict__, f)

    print(f'unit number:{len(sorting.get_unit_ids())}')

    sorting.register_recording(rec_ref)
    sorting.save(folder=os.path.join(sort_out_folder, 'sorting'))

    from spikeinterface import create_sorting_analyzer
    from spikeinterface.exporters import export_to_phy

    sorting_analyzer_folder = sort_out_folder / 'sorting_analyzer'

    if not os.path.exists(sorting_analyzer_folder):
        sorting_analyzer = create_sorting_analyzer(
            sorting=sorting, recording=rec_ref, format='memory',)

    # print(sorting_analyzer)
    # sorting_analyzer.compute("random_spikes")
    # sorting_analyzer.compute("waveforms", ms_before=2.0, ms_after=2.0)
    # sorting_analyzer.compute(["templates"])

    # phy_folder = sort_out_folder / 'phy'
    # if not phy_folder.exists():
    #     phy_folder.mkdir()
    # export_to_phy(
    #     sorting_analyzer,
    #     phy_folder,
    #     verbose=True,
    #     remove_if_exists=True,
    # )

    tt.report()
