import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
import spikeinterface as si
from pathlib import Path
import time
import matplotlib.pyplot as plt
from Timer import Timer
from spikegadget2nwb.preproc_func import get_bad_ch_id, rm_artifacts, parse_session_info
import numpy as np
import os
import mountainsort5 as ms5
import spikeinterface.exporters as sexp


rec_folder = r"D:\cl\ephys\CnL22_20250216_212206.rec"
animal_id, session_id = parse_session_info(rec_folder)

ishs = ['0', '1', '2', '3']


rec_folder = Path(rec_folder)

for ish in ishs:
    nwb_folder = Path(rec_folder) / f'{animal_id}_{session_id}.recsh{ish}.nwb'
    out_folder = Path('sortout') / animal_id / session_id / ish
    if not out_folder.exists():
        out_folder.mkdir(parents=True)


    rec = se.NwbRecordingExtractor(nwb_folder)
    print(rec)


    #%%
    rec_filt = sp.bandpass_filter(rec, freq_min=300, freq_max=6000, dtype='int32')

    bad_ch_id = get_bad_ch_id(rec, rec_folder, ish)
    remaining_ch = np.array([ch for ch in rec.get_channel_ids() if ch not in bad_ch_id])
    np.save(os.path.join(rec_folder, f'remaining_ch_sh{ish}.npy'), remaining_ch)
    print('Remaining channel IDs:', remaining_ch)
    chunk_size = 900
    rec_rm_artifacts = rm_artifacts(rec_filt, rec_folder, ish, bad_ch_id=bad_ch_id, chunk_size=chunk_size)
    # rec_clean = rec_rm_artifacts.channel_slice(remaining_ch)

    rec_cr = sp.common_reference(rec_rm_artifacts, reference='global', operator='median')

    rec_whiten = sp.whiten(rec_cr, dtype='float32')
    rec_preprocessed = rec_whiten


    #%%

    threshold = 3.5
    phase1_detect_time_radius_msec = .4
    npca_ch=3
    npca_sub=10

    timer = Timer('ms5')
    print('starting ms5 sorting')
    sorting_params = ms5.Scheme1SortingParameters(
        detect_time_radius_msec=phase1_detect_time_radius_msec, detect_threshold=threshold,
        npca_per_channel=npca_ch, npca_per_subdivision=npca_sub)
    sorting = ms5.sorting_scheme1(
        recording=rec_preprocessed, sorting_parameters=sorting_params)
    t_end = time.time()
    timer.report()

    import os
    import json
    current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    folder_name = 'sorting_results_' + current_time
    sort_out_folder = out_folder / folder_name
    if not os.path.exists(sort_out_folder):
        os.makedirs(sort_out_folder)
    with open(sort_out_folder / 'sorting_params.json', 'w') as f:
        json.dump(sorting_params.__dict__, f)



    print(sorting.unit_ids)

    print(sorting.count_num_spikes_per_unit())

    sorting.register_recording(rec_cr)
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec_rm_artifacts ,format='binary_folder', folder=sort_out_folder/'sorting_analyzer')

    print(sorting_analyzer)

    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    try:
        sorting_analyzer.compute(["templates", "quality_metrics","noise_levels", "amplitude_scalings", "template_metrics", "spike_amplitudes"])

        try:
            sexp.export_to_phy(sorting_analyzer, output_folder=folder_name / 'phy')
        except:
            print(f'Shank {ish} Failed to export to phy')

        for unit_id in sorting.get_unit_ids():
            sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
            plt.savefig(sort_out_folder / f'unit_summary_{unit_id}.png')
            plt.close()

        
    except Exception as e:
        print("Amplitude scaling computation failed, continuing with other metrics...")
        # Compute the remaining metrics individually
        metrics_list = ["templates", "quality_metrics","noise_levels", "template_metrics", "spike_amplitudes"]
        for metric in metrics_list:
            try:
                sorting_analyzer.compute([metric])
            except Exception as e:
                print(f"Failed to compute {metric}: {e}")



