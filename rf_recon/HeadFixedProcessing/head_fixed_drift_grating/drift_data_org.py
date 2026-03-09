import os
import numpy as np
import scipy.io
import h5py
from datetime import datetime
from pathlib import Path
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
from rf_recon.rf_func import dereference
import pickle


def process_drifting_grating_responses(rec_folder, stimdata_file, peaks_file, overwrite=True):
    """
    Process drifting grating responses from an experiment folder.
    
    The function:
      - Loads stimulus and timing information from MAT/HDF5 files,
      - Stores spike times for each trial with pre-stimulus and post-stimulus periods
      - Organizes trial information in a structured format matching the embedding extractor schema
      - Saves detailed trial-by-trial data into a PKL file
    
    Parameters:
      rec_folder (str or Path): Path to the experiment folder.
      stimdata_file (str or Path): Path to the stimulus data MAT/HDF5 file.
      peaks_file (str or Path): Path to the peaks MAT file.
      overwrite (bool): If False and the PKL file already exists, skip writing. 
                        If True, overwrite any existing PKL file.
    
    Returns:
      pkl_file (Path): Path to the saved (or existing) PKL file.
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
    ishs = ['0', '1', '2', '3', '4', '5', '6', '7']  # Assuming 8 shanks
    
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
    pre_stim_window = 0.2    # 50ms before stimulus
    post_stim_window = t_trial   # duration of the trial
    
    # Construct session folder for sorting results
    project_folder = Path(__file__).parent.parent.parent.parent
    session_folder = project_folder / rf"sortout/{animal_id}/{animal_id}_{session_id}"
    
    # Check if the output file already exists
    pkl_file = session_folder / f'drifting_grating_embedding_{dt_object.strftime("%Y%m%d_%H%M")}.pkl'
    if pkl_file.exists() and not overwrite:
        print(f"File {pkl_file} already exists and overwrite=False. Skipping computation and returning existing file.")
        return pkl_file
    
    all_units_data = []
    fs = None  # Will be set from first valid sorting
    
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
            
            # Load sorting
            sorting = PhySortingExtractor(phy_folder)
            unit_ids = sorting.unit_ids
            unit_qualities_this_sort = sorting.get_property('quality')
            
            if fs is None:
                fs = sorting.sampling_frequency
            
            for i, unit_id in enumerate(unit_ids):
                spike_train = sorting.get_unit_spike_train(unit_id)
                
                # Create structured data for this unit
                unit_data = {
                    'unit_id': unit_id,
                    'shank': ish,
                    'quality': unit_qualities_this_sort[i],
                    'sorting_folder': str(sorting_results_folder),
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
                        'temporal_frequency': trial_temporalFreq,
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
    
    print(f"Processed {len(all_units_data)} units across {len(rising_edges)} trials")
    
    # Save the data to a PKL file using the unified schema
    save_drifting_grating_to_pkl(
        animal_id=animal_id,
        session_id=session_id,
        rec_folder=rec_folder,
        stimdata_file=stimdata_file,
        fs=fs,
        rising_edges=rising_edges,
        stim_orientation=stim_orientation,
        stim_phase=stim_phase,
        stim_spatialFreq=stim_spatialFreq,
        stim_temporalFreq=stim_temporalFreq,
        unique_orientation=unique_orientation,
        unique_phase=unique_phase,
        unique_spatialFreq=unique_spatialFreq,
        unique_temporalFreq=unique_temporalFreq,
        n_repeats=n_repeats,
        pre_stim_window=pre_stim_window,
        post_stim_window=post_stim_window,
        all_units_data=all_units_data,
        output_path=pkl_file,
    )

    return pkl_file


def save_drifting_grating_to_pkl(
    *,
    animal_id,
    session_id,
    rec_folder,
    stimdata_file,
    fs,
    rising_edges,
    stim_orientation,
    stim_phase,
    stim_spatialFreq,
    stim_temporalFreq,
    unique_orientation,
    unique_phase,
    unique_spatialFreq,
    unique_temporalFreq,
    n_repeats,
    pre_stim_window,
    post_stim_window,
    all_units_data,
    output_path,
):
    """
    Save drifting grating responses in the SAME schema as the embedding extractor.
    This ensures compatibility with downstream analysis code.
    """
    
    # Create trial windows (using rising edges as both start and end for compatibility)
    trial_windows = [(int(edge), int(edge)) for edge in rising_edges]
    
    # Build all trial parameters list
    all_trial_parameters = []
    for trial_idx in range(len(rising_edges)):
        trial_params = {
            'trial_index': trial_idx,
            'orientation': float(stim_orientation[trial_idx]),
            'phase': float(stim_phase[trial_idx]),
            'spatial_frequency': float(stim_spatialFreq[trial_idx]),
            'temporal_frequency': float(stim_temporalFreq[trial_idx]),
        }
        all_trial_parameters.append(trial_params)
    
    # Initialize the unified data structure
    neural_data = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'task_file': str(stimdata_file),
            'extraction_date': datetime.now().isoformat(),
            'n_trials': len(rising_edges),
            'experiment_type': 'drifting_grating',
            'sampling_frequency': fs,
        },
        'experiment_parameters': {
            'stimulus_duration': float(post_stim_window),
            'iti_duration': None,  # Not applicable for this experiment type
            'trial_duration': float(post_stim_window),
            'total_trials': len(rising_edges),
            'n_orientations': len(unique_orientation),
            'n_phases': len(unique_phase),
            'n_spatial_frequencies': len(unique_spatialFreq),
            'n_temporal_frequencies': len(unique_temporalFreq),
            'n_repeats': n_repeats,
        },
        'trial_info': {
            'orientations': stim_orientation.tolist(),
            'unique_orientations': unique_orientation.tolist(),
            'phases': stim_phase.tolist(),
            'unique_phases': unique_phase.tolist(),
            'spatial_frequencies': stim_spatialFreq.tolist(),
            'unique_spatial_frequencies': unique_spatialFreq.tolist(),
            'temporal_frequencies': stim_temporalFreq.tolist(),
            'unique_temporal_frequencies': unique_temporalFreq.tolist(),
            'trial_windows': trial_windows,
            'all_trial_parameters': all_trial_parameters,
        },
        'spike_data': {},
        'unit_info': {},
        'extraction_params': {
            'window_pre': pre_stim_window,
            'window_post': post_stim_window,
            'total_units': len(all_units_data),
        }
    }
    
    # Populate spike_data and unit_info in the unified format
    unit_counter = 0
    
    for unit in all_units_data:
        # Create unique unit identifier matching the embedding extractor format
        unique_unit_id = f"shank{unit['shank']}_unit{unit['unit_id']}"
        
        # Store unit metadata
        neural_data['unit_info'][unique_unit_id] = {
            'original_unit_id': int(unit['unit_id']),
            'shank': unit['shank'],
            'quality': unit.get('quality', 'unknown'),
            'sorting_folder': unit.get('sorting_folder', ''),
            'n_spikes_total': sum(len(t['spike_times']) for t in unit['trials']),
            'unit_index': unit_counter,
        }
        
        # Store spike data for each trial
        trials_out = []
        for t in unit['trials']:
            trial_data = {
                'trial_index': t['trial_number'],
                'orientation': t['orientation'],
                'phase': t.get('phase'),
                'spatial_frequency': t.get('spatial_frequency'),
                'temporal_frequency': t.get('temporal_frequency'),
                'spike_times': t['spike_times'].tolist(),
                'spike_count': len(t['spike_times']),
                'trial_start': t['stimulus_onset_time'],
                'trial_end': t['stimulus_onset_time'],  # Same as start for compatibility
                # Additional metrics preserved for analysis
                'pre_stim_count': t.get('pre_stim_count'),
                'post_stim_count': t.get('post_stim_count'),
                'firing_rate_pre': t.get('firing_rate_pre'),
                'firing_rate_post': t.get('firing_rate_post'),
            }
            trials_out.append(trial_data)
        
        neural_data['spike_data'][unique_unit_id] = trials_out
        unit_counter += 1
    
    print(f"Saving unified PKL format → {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(neural_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return output_path


# Example usage:
if __name__ == '__main__':
    rec_folder = Path(input("Please enter the full path to the recording folder: ").strip().strip('"'))
    stimdata_file = Path(input("Please enter the full path to the stimulus data .mat/.h5 file: ").strip().strip('"'))
    peaks_file = Path(input("Please enter the full path to the peaks_xx.mat file: ").strip().strip('"'))

    # Process the data
    pkl_path = process_drifting_grating_responses(rec_folder, stimdata_file, peaks_file, overwrite=True)

    print(f"Data saved to: {pkl_path}")
    
    # Load and verify the structure
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nData structure verification:")
    print(f"- Metadata keys: {list(data['metadata'].keys())}")
    print(f"- Experiment type: {data['metadata']['experiment_type']}")
    print(f"- Total units: {len(data['spike_data'])}")
    print(f"- Total trials: {data['metadata']['n_trials']}")
    print(f"- Unique orientations: {data['trial_info']['unique_orientations']}")