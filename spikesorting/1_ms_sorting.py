import os
import time
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
import spikeinterface.exporters as sexp
import mountainsort5 as ms5

from Timer import Timer
from rec2nwb.preproc_func import rm_artifacts, parse_session_info


def main(rec_folder, threshold=5.5):
    # Define recording folder and parse session info
    rec_folder = Path(rec_folder)
    animal_id, session_id, folder_name = parse_session_info(str(rec_folder))
    shanks = ['0', '1', '2', '3']
    # shanks = ['2']

    for shank in shanks:
        # Construct paths for NWB file and output folder
        nwb_folder = rec_folder / f"{folder_name}sh{shank}.nwb"
        out_folder = Path("sortout") / animal_id / session_id / shank
        out_folder.mkdir(parents=True, exist_ok=True)

        # Load recording from NWB file
        rec = se.NwbRecordingExtractor(str(nwb_folder))
        print("Recording:", rec)

        # Preprocessing: bandpass filter
        rec_filt = sp.bandpass_filter(
            rec, freq_min=300, freq_max=6000, dtype=np.float32)

        # Remove artifacts using a chunk-based approach
        chunk_time = 0.02
        artifacts_thres = 6.0
        # FIXME: the artifact problem is severe, don't directly set to 0, try to interpolate...
        rec_rm_artifacts = rm_artifacts(
            rec_filt, rec_folder, shank,
            chunk_time=chunk_time, threshold=artifacts_thres,
            overwrite=True)

        # Apply common reference and whitening
        rec_cr = sp.common_reference(
            rec_rm_artifacts, reference="global", operator="median")
        # rec_whiten = sp.whiten(rec_cr, dtype="float32")
        recording_preprocessed: si.BaseRecording = sp.whiten(rec_cr)

        # Define sorting parameters
        detect_time_radius_msec = 0.4
        npca_per_channel = 3
        npca_per_subdivision = 10

        timer = Timer("ms5")
        print("Starting ms5 sorting...")
        sorting_params = ms5.Scheme1SortingParameters(
            detect_sign=0,
            detect_time_radius_msec=detect_time_radius_msec,
            detect_threshold=threshold,
            npca_per_channel=npca_per_channel,
            npca_per_subdivision=npca_per_subdivision
        )
        sorting = ms5.sorting_scheme1(
            recording=recording_preprocessed, sorting_parameters=sorting_params)
        timer.report()

        # Create sorting results folder
        current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        results_folder_name = f"sorting_results_{current_time}"
        sort_out_folder = out_folder / results_folder_name
        sort_out_folder.mkdir(parents=True, exist_ok=True)
        with open(sort_out_folder / "sorting_params.json", "w") as f:
            json.dump(sorting_params.__dict__, f)

        print("Unit IDs:", sorting.unit_ids)
        print("Spike counts per unit:", sorting.count_num_spikes_per_unit())

        # Register recording and create a sorting analyzer
        sorting.register_recording(rec_cr)
        analyzer_folder = sort_out_folder / "sorting_analyzer"
        sorting_analyzer = si.create_sorting_analyzer(sorting=sorting,
                                                      recording=rec_rm_artifacts,
                                                      format="binary_folder",
                                                      folder=str(analyzer_folder))
        print("Sorting analyzer:", sorting_analyzer)

        # Compute metrics

        try:
            sorting_analyzer.compute(['random_spikes', 'waveforms', 'noise_levels'])
            sorting_analyzer.compute('templates')
            _ = sorting_analyzer.compute('template_similarity')
            _ = sorting_analyzer.compute('spike_amplitudes')
            _ = sorting_analyzer.compute('correlograms')
            _ = sorting_analyzer.compute('unit_locations')

            out_fig_folder = sort_out_folder / 'raw_units'
            out_fig_folder.mkdir(parents=True, exist_ok=True)

            for unit_id in sorting.get_unit_ids():
                sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
                plt.savefig(out_fig_folder / f'unit_summary_{unit_id}.png')
                plt.close()

        except Exception as e:
            print(f"Error during metrics computation: {e}")


if __name__ == "__main__":
    threshold = 5.5
    rec_folder = Path(r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/250411/CnL34/CnL34_250411_154730")
    main(threshold=threshold, rec_folder=rec_folder)
