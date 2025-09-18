import os
import numpy as np
import scipy.io
import h5py
from datetime import datetime
from pathlib import Path
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
from rf_recon.rf_func import dereference

def process_static_grating_responses(rec_folder, stimdata_file, peaks_file, overwrite=True):
    """
    Process static grating responses from an experiment folder.
    
    The function:
      - Loads stimulus and timing information from MAT/HDF5 files,
      - Stores spike times for each trial with pre-stimulus and post-stimulus periods
      - Organizes trial information in a structured format
      - Saves detailed trial-by-trial data into an NPZ file
    
    Parameters:
      experiment_folder (str or Path): Path to the experiment folder.
      overwrite (bool): If False and the NPZ file already exists, skip writing. 
                        If True, overwrite any existing NPZ file.
    
    Returns:
      npz_file (Path): Path to the saved (or existing) NPZ file.
    """
    stim_str = str(stimdata_file)
    timestamp_str = stim_str.split('_')[-2] + '_' + stim_str.split('_')[-1].split('.')[0]
    date_format = "%Y-%m-%d_%H-%M-%S"
    dt_object = datetime.strptime(timestamp_str, date_format)
    
    # Load peaks data (to get rising edges/trial start times)
    peaks_data = scipy.io.loadmat(peaks_file, struct_as_record=False, squeeze_me=True)
    rising_edges = peaks_data['locs']
    
    
    # Parse session info (animal_id, session_id, folder_name)
    animal_id, session_id, folder_name = parse_session_info(rec_folder)
    ishs = ['0', '1', '2', '3']
    
    # Open the Stimdata file to get stimulus parameters

    with h5py.File(stimdata_file, 'r') as f:
        patternParams_group = f['Stimdata']['movieParams']
        
        # Process orientation, phase, spatialFreq
        orientation_data = patternParams_group['orientation'][()]
        stim_orientation = np.array([dereference(ref, f) for ref in orientation_data]).flatten().astype(float)
        
        phase_data = patternParams_group['phase'][()]
        stim_phase = np.array([dereference(ref, f) for ref in phase_data]).flatten().astype(float)
        
        spatialFreq_data = patternParams_group['spatialFreq'][()]
        stim_spatialFreq = np.array([dereference(ref, f) for ref in spatialFreq_data]).flatten().astype(float)

        temporalFreq_data = patternParams_group['temporalFreq'][()]
        stim_temporalFreq = np.array([dereference(ref, f) for ref in temporalFreq_data]).flatten().astype(float)

        t_trial = f['Stimdata']['movieDuration'][()][0,0]
    
    print("Orientation:", stim_orientation)
    print("Phase:", stim_phase)
    print("Spatial Frequency:", stim_spatialFreq)
    print("Temporal Frequency:", stim_temporalFreq)


    # Determine the number of drifting grating stimuli and extract the corresponding rising edges
    n_drifting_grating = stim_orientation.shape[0]
    print(f"Number of drifting grating stimuli: {n_drifting_grating}")
    print(f"Number of rising edges: {len(rising_edges)}")
    
    # Unique stimulus parameters
    unique_orientation = np.unique(stim_orientation)
    unique_phase = np.unique(stim_phase)
    unique_spatialFreq = np.unique(stim_spatialFreq)
    unique_temporalFreq = np.unique(stim_temporalFreq)
    
    n_orientation = len(unique_orientation)
    n_phase = len(unique_phase)
    n_spatialFreq = len(unique_spatialFreq)
    n_temporalFreq = len(unique_temporalFreq)

    # Compute the number of repeats/trials per condition
    n_repeats = n_drifting_grating // (n_orientation * n_phase * n_spatialFreq * n_temporalFreq)

    # Define time windows (in seconds)
    pre_stim_window = 0.05    # 50ms before stimulus
    post_stim_window = t_trial   # duration of the trial
    
    all_units_data = []
    unit_info = []
    all_unit_qualities = []
    
    # Construct session folder for sorting results
    code_folder = Path(__file__).parent.parent.parent
    session_folder = code_folder / rf"sortout/{animal_id}/{animal_id}_{session_id}"
    
    # Check if the output file already exists
    npz_file = session_folder / f'drifting_grating_responses_{dt_object.strftime("%Y%m%d_%H%M")}.npz'
    if npz_file.exists() and not overwrite:
        print(f"File {npz_file} already exists and overwrite=False. Skipping computation and returning existing file.")
        return npz_file
    
    for ish in ishs:
        print(f'Processing {animal_id}/{session_id}/{ish}')
        shank_folder = session_folder / f'shank{ish}'
        sorting_results_folders = []
        for root, dirs, files in os.walk(shank_folder):
            for dir_name in dirs:
                if dir_name.startswith('sorting_results_'):
                    sorting_results_folders.append(os.path.join(root, dir_name))
        
        for sorting_results_folder in sorting_results_folders:
            phy_folder = Path(sorting_results_folder) / 'phy'
            out_fig_folder = Path(sorting_results_folder) / 'drifting_grating'
            if not out_fig_folder.exists():
                out_fig_folder.mkdir(parents=True)
            
            # Load sorting analyzer (optionally use curated data)
            # sorting_anaylzer = load_sorting_analyzer(Path(sorting_results_folder) / 'sorting_analyzer')
            # sorting = sorting_anaylzer.sorting
            sorting = PhySortingExtractor(phy_folder)
            unit_ids = sorting.unit_ids
            unit_qualities_this_sort = sorting.get_property('quality')
            fs = sorting.sampling_frequency
            
            for i, unit_id in enumerate(unit_ids):
                spike_train = sorting.get_unit_spike_train(unit_id)
                
                # Create structured data for this unit
                unit_data = {
                    'unit_id': unit_id,
                    'shank': ish,
                    'sampling_rate': fs,
                    'trials': []
                }
                
                # Process each trial
                for trial_idx, edge in enumerate(rising_edges):
                    # Define time windows relative to stimulus onset
                    pre_start_time = edge - pre_stim_window * fs
                    post_end_time = edge + post_stim_window * fs
                    
                    # Extract spikes in the trial window
                    trial_spike_mask = (spike_train >= pre_start_time) & (spike_train < post_end_time)
                    trial_spikes = spike_train[trial_spike_mask]
                    
                    # Convert spike times relative to stimulus onset (in seconds)
                    relative_spike_times = (trial_spikes - edge) / fs
                    
                    # Get stimulus parameters for this trial
                    trial_orientation = stim_orientation[trial_idx]
                    trial_phase = stim_phase[trial_idx]
                    trial_spatialFreq = stim_spatialFreq[trial_idx]
                    trial_temporalFreq = stim_temporalFreq[trial_idx]

                    # Find condition indices
                    ori_idx = np.where(unique_orientation == trial_orientation)[0][0]
                    phase_idx = np.where(unique_phase == trial_phase)[0][0]
                    sf_idx = np.where(unique_spatialFreq == trial_spatialFreq)[0][0]
                    tf_idx = np.where(unique_temporalFreq == trial_temporalFreq)[0][0]

                    # Calculate repeat number for this condition
                    condition_trials = []
                    for prev_trial in range(trial_idx):
                        if (stim_orientation[prev_trial] == trial_orientation and
                            stim_phase[prev_trial] == trial_phase and
                            stim_spatialFreq[prev_trial] == trial_spatialFreq and
                            stim_temporalFreq[prev_trial] == trial_temporalFreq):
                            condition_trials.append(prev_trial)
                    repeat_idx = len(condition_trials)
                    
                    # Store trial information
                    trial_info = {
                        'trial_number': trial_idx,
                        'stimulus_onset_time': edge,  # in samples
                        'stimulus_onset_time_sec': edge / fs,  # in seconds
                        'orientation': trial_orientation,
                        'phase': trial_phase,
                        'spatial_frequency': trial_spatialFreq,
                        'orientation_idx': ori_idx,
                        'phase_idx': phase_idx,
                        'spatial_freq_idx': sf_idx,
                        'temporal_freq_idx': tf_idx,
                        'repeat_idx': repeat_idx,
                        'spike_times': relative_spike_times,  # relative to stimulus onset
                        'pre_stim_spikes': relative_spike_times[relative_spike_times < 0],
                        'post_stim_spikes': relative_spike_times[relative_spike_times >= 0],
                        'pre_stim_count': np.sum(relative_spike_times < 0),
                        'post_stim_count': np.sum(relative_spike_times >= 0),
                        'firing_rate_pre': np.sum(relative_spike_times < 0) / pre_stim_window,
                        'firing_rate_post': np.sum(relative_spike_times >= 0) / post_stim_window,
                    }
                    
                    unit_data['trials'].append(trial_info)
                
                all_units_data.append(unit_data)
                unit_info.append((ish, unit_id))
                all_unit_qualities.append(unit_qualities_this_sort[i])
    
    print(f"Processed {len(all_units_data)} units across {len(rising_edges)} trials")
    
    # Create summary statistics
    summary_stats = {
        'n_units': len(all_units_data),
        'n_trials': len(rising_edges),
        'n_orientation': n_orientation,
        'n_phase': n_phase,
        'n_spatialFreq': n_spatialFreq,
        'n_repeats': n_repeats,
        'pre_stim_window': pre_stim_window,
        'post_stim_window': post_stim_window,
    }
    
    # Save the data to an NPZ file in the session folder
    print(f"Saving data to {npz_file} (overwrite={overwrite})")
    np.savez(
        npz_file,
        # Original data structure (for backwards compatibility)
        unit_info=unit_info,
        unit_qualities=all_unit_qualities,
        stim_orientation=stim_orientation,
        stim_phase=stim_phase,
        stim_spatialFreq=stim_spatialFreq,
        unique_orientation=unique_orientation,
        unique_phase=unique_phase,
        unique_spatialFreq=unique_spatialFreq,
        rising_edges=rising_edges,
        
        # New structured data
        units_data=all_units_data,
        summary_stats=summary_stats,
        
        # Timing parameters
        pre_stim_window=pre_stim_window,
        post_stim_window=post_stim_window,
    )
    print("Data saved with detailed trial structure")
    return npz_file


def load_and_analyze_static_grating_data(npz_file):
    """
    Helper function to load and explore the saved data structure.
    """
    data = np.load(npz_file, allow_pickle=True)
    
    units_data = data['units_data']
    summary_stats = data['summary_stats'].item()
    
    print("Summary Statistics:")
    for key, value in summary_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nFirst unit example:")
    first_unit = units_data[0]
    print(f"  Unit ID: {first_unit['unit_id']}")
    print(f"  Shank: {first_unit['shank']}")
    print(f"  Number of trials: {len(first_unit['trials'])}")
    
    first_trial = first_unit['trials'][0]
    print(f"\nFirst trial example:")
    for key, value in first_trial.items():
        if key in ['spike_times', 'pre_stim_spikes', 'stim_spikes']:
            print(f"  {key}: {len(value)} spikes")
        else:
            print(f"  {key}: {value}")
    
    return data


# Example usage:
if __name__ == '__main__':
    rec_folder = Path(input("Please enter the full path to the recording folder: ").strip().strip('"'))
    stimdata_file = Path(input("Please enter the full path to the stimulus data .mat/.h5 file: ").strip().strip('"'))
    peaks_file = Path(input("Please enter the full path to the peaks_xx.mat file: ").strip().strip('"'))

    # Process the data
    npz_path = process_static_grating_responses(rec_folder, stimdata_file, peaks_file, overwrite=True)

    print(f"Data saved to: {npz_path}")