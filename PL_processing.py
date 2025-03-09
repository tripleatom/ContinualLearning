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
from spikegadget2nwb.preproc_func import get_bad_ch_id, rm_artifacts, parse_session_info


def main(rec_folder, threshold=3.5):
    # Define recording folder and parse session info
    rec_folder = Path(rec_folder)
    animal_id, session_id, folder_name = parse_session_info(str(rec_folder))
    shanks = ['0', '1', '2', '3']

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
            rec, freq_min=300, freq_max=6000, dtype="int32")

        # Get bad channels and determine remaining channels
        bad_ch_id = get_bad_ch_id(rec, rec_folder, shank)
        remaining_ch = np.array(
            [ch for ch in rec.get_channel_ids() if ch not in bad_ch_id])
        np.save(rec_folder / f"remaining_ch_sh{shank}.npy", remaining_ch)
        print("Remaining channel IDs:", remaining_ch)

        # Remove artifacts using a chunk-based approach
        chunk_size = 900
        rec_rm_artifacts = rm_artifacts(rec_filt, rec_folder, shank, bad_ch_id=bad_ch_id, chunk_size=chunk_size)

        # Apply common reference and whitening
        rec_cr = sp.common_reference(
            rec_rm_artifacts, reference="global", operator="median")
        rec_whiten = sp.whiten(rec_cr, dtype="float32")
        rec_preprocessed = rec_whiten

        # Define sorting parameters
        detect_time_radius_msec = 0.4
        npca_per_channel = 3
        npca_per_subdivision = 10

        timer = Timer("ms5")
        print("Starting ms5 sorting...")
        sorting_params = ms5.Scheme1SortingParameters(
            detect_time_radius_msec=detect_time_radius_msec,
            detect_threshold=threshold,
            npca_per_channel=npca_per_channel,
            npca_per_subdivision=npca_per_subdivision
        )
        sorting = ms5.sorting_scheme1(
            recording=rec_preprocessed, sorting_parameters=sorting_params)
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
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        try:
            metrics = ["templates", "quality_metrics", "noise_levels",
                       "amplitude_scalings", "template_metrics", "spike_amplitudes"]
            sorting_analyzer.compute(metrics)

            # Export to phy (if possible)
            try:
                phy_output_folder = sort_out_folder / "phy"
                sexp.export_to_phy(
                    sorting_analyzer, output_folder=str(phy_output_folder))
            except Exception as e:
                print(f"Shank {shank} failed to export to phy: {e}")

            # Plot and save unit summaries
            for unit_id in sorting.get_unit_ids():
                sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
                plt.savefig(sort_out_folder / f"unit_summary_{unit_id}.png")
                plt.close()

        except Exception as e:
            print(
                "Amplitude scaling computation failed, continuing with other metrics...")
            for metric in ["templates", "quality_metrics", "noise_levels", "template_metrics", "spike_amplitudes"]:
                try:
                    sorting_analyzer.compute([metric])
                except Exception as metric_err:
                    print(f"Failed to compute {metric}: {metric_err}")


if __name__ == "__main__":
    threshold = 3.5
    rec_folder = Path(
        r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CNL36\CNL36_250305_194558")
    main(threshold=threshold, rec_folder=rec_folder)
