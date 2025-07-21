import os
import numpy as np
import pickle
import pandas as pd
from pathlib import Path

from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor

from process_func.DIO import get_dio_folders, concatenate_din_data


def extract_mean_firing_rates(spike_train, trial_windows, fs):
    """
    Compute mean firing rate per trial.
    Returns numpy array of rates.
    """
    rates = np.zeros(len(trial_windows))
    for idx, (start, end) in enumerate(trial_windows):
        spikes = spike_train[(spike_train >= start) & (spike_train < end)]
        duration = (end - start) / fs
        rates[idx] = len(spikes) / duration if duration > 0 else 0
    return rates


def process_new_experiment(rec_folder, task_file,
                           overwrite=True):
    """
    Process experiment data:
      - Extract trial info from task["trials"][param_key]
      - Use DIO edges to define trial windows
      - Compute mean firing rates for each neural unit across trials
    Parameters:
      rec_folder: path to recording folder
      task_file: path to pickle with 'trials' list and optional 'summary'
    """
    rec_folder = Path(rec_folder)
    task_file = Path(task_file)

    # Get animal and session IDs
    animal_id = rec_folder.name.split('.')[0].split('_')[0]
    session_id = rec_folder.name.split('.')[0]
    
    print(f"Processing {animal_id}/{session_id}")

    # Load task and trials
    with open(task_file, 'rb') as f:
        task = pickle.load(f)
    left_params = task['trial_left_params']

    left_params = left_params[1:-1] # remove the first and last trial
    # extract the orientation of the left grating
    left_orientation = [param['orientation'] for param in left_params]
 

    # Read digital input to define windows
    dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
    pd_time, pd_state = concatenate_din_data(dio_folders, 3)
    rising = np.where(pd_state == 1)[0]
    falling = np.where(pd_state == 0)[0][1:]
    rising_times = pd_time[rising]
    falling_times = pd_time[falling]

    # Determine number of trials
    n_trials = len(left_orientation)
    trial_windows = [(rising_times[i], falling_times[i]) for i in range(n_trials)]

    # Construct session folder (similar to static_disc)
    code_folder = Path(__file__).parent.parent.parent
    session_folder = code_folder / f"sortout/{animal_id}/{session_id}"
    
    # Prepare output folder
    out_folder = session_folder / 'freelymovingRF'
    out_folder.mkdir(parents=True, exist_ok=True)

    # Compute neural responses
    all_units_responses = []
    unit_info = []
    all_unit_qualities = []
    fs = None

    # Iterate through shanks (similar to static_disc)
    ishs = ['0', '1', '2', '3']
    
    for ish in ishs:
        print(f'Processing shank {ish}')
        shank_folder = session_folder / f'shank{ish}'
        
        if not shank_folder.exists():
            print(f"Shank folder {shank_folder} does not exist, skipping...")
            continue
            
        # Find sorting results folders
        sorting_results_folders = []
        for root, dirs, files in os.walk(shank_folder):
            for dir_name in dirs:
                if dir_name.startswith('sorting_results_'):
                    sorting_results_folders.append(os.path.join(root, dir_name))
        
        for sorting_results_folder in sorting_results_folders:
            phy_folder = Path(sorting_results_folder) / 'phy'
            
            try:
                # Load sorting analyzer first, then fall back to phy
                sorting_analyzer = load_sorting_analyzer(Path(sorting_results_folder) / 'sorting_analyzer')
                sorting = sorting_analyzer.sorting
                
                # Alternative: load from Phy if needed
                if phy_folder.exists():
                    sorting = PhySortingExtractor(phy_folder)

                # Set sampling freq
                if fs is None:
                    fs = sorting.sampling_frequency
                    print(f"Sampling frequency: {fs} Hz")

                print(f"Processing {sorting_results_folder}")
                print(f"unit number: {len(sorting.unit_ids)}")
                
                # Get unit qualities for this sorting
                unit_ids = sorting.unit_ids
                unit_qualities_this_sort = sorting.get_property('quality') if hasattr(sorting, 'get_property') else ['good'] * len(unit_ids)
                
                # Loop units
                for i, unit_id in enumerate(unit_ids):
                    spike_train = sorting.get_unit_spike_train(unit_id)
                    rates = extract_mean_firing_rates(spike_train, trial_windows, fs)
                    all_units_responses.append({
                        'unit_id': unit_id,
                        'mean_firing_rates': rates.tolist(),
                        'total_spikes': int(len(spike_train))
                    })
                    
                    # Track unit info and quality
                    unit_info.append((ish, unit_id))
                    all_unit_qualities.append(unit_qualities_this_sort[i] if isinstance(unit_qualities_this_sort, (list, np.ndarray)) else 'good')
                    
            except Exception as e:
                print(f"Error processing {sorting_results_folder}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save output
    out_file = out_folder / 'freelymovingRF_data.npz'
    if out_file.exists() and not overwrite:
        print(f"Skipping save, file exists: {out_file}")
        return out_file

    np.savez(out_file,
             left_orientation=left_orientation,
             all_units_responses=all_units_responses,
             unit_info=unit_info,
             unit_qualities=all_unit_qualities,
             session=str(rec_folder.name),
             task_summary=task.get('summary', {}))

    print(f"Saved new experiment data to {out_file}")
    print(f"Processed {len(all_units_responses)} units with quality information")
    return out_file


if __name__ == '__main__':
    rec_folder = r"G:\freelymovingRF\250712\CnL39SG\CnL39SG_20250712_184715.rec"
    task_file = r"D:\cl\grating_disk_representation\task_data\CnL39_1_20250712_190810.pkl"
    process_new_experiment(rec_folder, task_file)
