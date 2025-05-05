import os
from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.exporters as sexp
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import apply_sortingview_curation
from spikeinterface.widgets import plot_sorting_summary
import numpy as np
import spikeinterface.preprocessing as sp

from rec2nwb.preproc_func import parse_session_info

# Constants
BASE_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed"
# BASE_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI"
DATES = ['250505']
# ANIMAL_IDS = ['CoI06', 'CoI07', 'CoI08', 'CoI09', 'CoI10']
ANIMAL_IDS = ['CnL38']
ISHS = ['0', '1', '2', '3']
SORTOUT_FOLDER = Path(__file__).parents[1] / 'sortout'

for date in DATES:
    for animal_id in ANIMAL_IDS:
        # Construct experiment folder path
        experiment_folder = Path(BASE_FOLDER) / f"{date}/{animal_id}"
        if not experiment_folder.exists():
            print(f"Experiment folder not found: {experiment_folder}")
            continue
        # Select the first subdirectory found (if any)
        rec_folder = next((p for p in experiment_folder.iterdir() if p.is_dir()), None)
        print(f"Recording folder: {rec_folder}")
        if rec_folder is None:
            continue

        # Parse session info (returns animal_id, session_id, folder_name)
        animal_id, session_id, folder_name = parse_session_info(rec_folder)
        session_folder = SORTOUT_FOLDER / f"{animal_id}/{session_id}"

        for ish in ISHS:

            print(f"Processing {animal_id} {session_id} shank {ish}...")
            # Build recording file path
            recording_file = rec_folder / f"{folder_name}sh{ish}.nwb"
            
            # Create the NWB recording extractor
            recording = se.NwbRecordingExtractor(str(recording_file))
            rec_filt = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
            
            shank_folder = session_folder / ish
            # Find folders starting with 'sorting_results_'
            sorting_results_folders = [
                os.path.join(root, d)
                for root, dirs, _ in os.walk(shank_folder)
                for d in dirs
                if d.startswith('sorting_results_')
            ]
            if not sorting_results_folders:
                print(f"No sorting results folder found in {shank_folder}")
                continue

            for sorting_results_folder in sorting_results_folders:
                sorting_results_folder = Path(sorting_results_folder)
                analyzer_folder = sorting_results_folder / 'sorting_analyzer'
                
                # Load the sorting analyzer
                sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
                sorting = sorting_analyzer.sorting
                sorting_analyzer = create_sorting_analyzer(sorting, rec_filt)

                sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
                _ = sorting_analyzer.compute('spike_amplitudes')
                _ = sorting_analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")
                
                # Export sorting results to Phy format
                output_folder = sorting_results_folder / 'phy'
                if output_folder.exists():
                    print(f"Phy folder already exists: {output_folder}")
                    continue
                sexp.export_to_phy(sorting_analyzer, output_folder=output_folder)
