import sys
from pathlib import Path

# Add the parent 'code' directory to Python path
# Use __file__ to get the script location, not cwd()
code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import os
import numpy as np
from datetime import datetime
from pathlib import Path
from spikeinterface.extractors import PhySortingExtractor
from spikeinterface import load_sorting_analyzer
from process_func import DIO
from rec2nwb.preproc_func import parse_session_info
import pickle
import json

def get_rewarded_on_left(filepath):
    """
    Extract rewarded on left values for all trials in a session

    Parameters:
    -----------
    filepath : str
        Path to JSON session file

    Returns:
    --------
    list of bool
        Array of True/False values indicating if white was on left for each trial
        Length of array = number of trials

    Example:
    --------
    >>> rewarded_positions = get_rewarded_on_left('session_data.json')
    >>> print(f"Total trials: {len(rewarded_positions)}")
    >>> print(f"First 10 trials: {rewarded_positions[:10]}")
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    trials = data.get('trials', [])
    rewarded_on_left = [trial.get('rewardedOnLeft', None) for trial in trials]

    return rewarded_on_left

def process_behavior_trial_responses(rec_folder, trial_start, trial_end, rewarded_on_left, sortout_folder, task_start=0, task_end=None, overwrite=True):
    """
    Process behavior trial responses from an experiment folder.

    The function:
      - Loads spike times for each trial based on trial start/end timestamps
      - Stores trial information (rewarded_on_left condition)
      - Organizes trial information in a structured format matching the embedding extractor schema
      - Saves detailed trial-by-trial data into a PKL file

    Parameters:
      rec_folder (str or Path): Path to the experiment folder.
      trial_start (np.ndarray): Array of trial start indices (in samples, 0-based relative to experiment start).
      trial_end (np.ndarray): Array of trial end indices (in samples, 0-based relative to experiment start).
      rewarded_on_left (np.ndarray): Boolean array indicating rewarded stimulus on left side.
      sortout_folder (str or Path): Path to the session sortout folder (parent of shank0, shank1, ... folders).
      task_start (int): Sample index in the concatenated recording where this experiment starts.
                        Spikes are filtered to [task_start, task_end) and shifted by -task_start
                        to align with 0-based DIO trial timestamps. Default: 0.
      task_end (int or None): Sample index where this experiment ends. If None, uses the last spike index + 1.
      overwrite (bool): If False and the PKL file already exists, skip writing.
                        If True, overwrite any existing PKL file.

    Returns:
      pkl_file (Path): Path to the saved (or existing) PKL file.
    """
    
    # Parse session info (animal_id, session_id, folder_name)
    animal_id, session_id, folder_name = parse_session_info(rec_folder)
    ishs = ['0', '1', '2', '3', '4', '5', '6', '7']  # Assuming 8 shanks
    
    # Convert to numpy arrays if not already
    trial_start = np.array(trial_start)
    trial_end = np.array(trial_end)
    rewarded_on_left = np.array(rewarded_on_left, dtype=bool)

    # Get number of trials
    n_trials = len(trial_start)
    print(f"Number of trials: {n_trials}")
    print(f"Trial start shape: {trial_start.shape}")
    print(f"Trial end shape: {trial_end.shape}")
    print(f"Rewarded on left shape: {rewarded_on_left.shape}")

    # Verify array lengths match
    assert len(trial_start) == len(trial_end) == len(rewarded_on_left), \
        "trial_start, trial_end, and rewarded_on_left must have the same length"

    # Unique stimulus conditions
    unique_conditions = np.unique(rewarded_on_left)
    n_conditions = len(unique_conditions)
    print(f"Unique conditions (rewarded_on_left): {unique_conditions}")

    # Compute the number of repeats per condition
    n_left = np.sum(rewarded_on_left)
    n_right = np.sum(~rewarded_on_left)
    print(f"Trials with rewarded on left: {n_left}")
    print(f"Trials with rewarded on right: {n_right}")
    
    # Define time windows (relative to trial start/end)
    pre_trial_window = 0.2    # 200ms before trial start
    post_trial_window = 0.2   # 200ms after trial end
    
    # Construct session folder for sorting results
    session_folder = Path(sortout_folder)
    
    # Check if the output file already exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pkl_file = session_folder / f'behavior_trial_embedding_{timestamp}.pkl'
    if pkl_file.exists() and not overwrite:
        print(f"File {pkl_file} already exists and overwrite=False. Skipping computation and returning existing file.")
        return pkl_file
    
    all_units_data = []
    fs = None  # Will be set from first valid sorting

    # Load unit quality labels from session folder
    labels_file = session_folder / 'unit_labels.json'
    unit_labels = {}
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            unit_labels = json.load(f)
        print(f"Loaded unit labels from {labels_file}")
    else:
        print(f"No unit_labels.json found in {session_folder}, quality will be 'unknown'")

    for ish in ishs:
        print(f'Processing {animal_id}/{session_id}/{ish}')
        shank_folder = session_folder / f'shank{ish}'
        
        if not shank_folder.exists():
            print(f"Shank folder {shank_folder} does not exist, skipping...")
            continue
            
        sorting_results_folders = []
        for root, dirs, files in os.walk(shank_folder):
            for dir_name in dirs:
                if dir_name.startswith('sorting_results_'):
                    sorting_results_folders.append(os.path.join(root, dir_name))
        
        for sorting_results_folder in sorting_results_folders:
            # Look for phy and sorting_analyzer folders
            phy_folder = Path(sorting_results_folder) / 'phy'
            sorting_analyzer_folder = Path(sorting_results_folder) / 'sorting_analyzer'

            # Prioritize phy folder, then fall back to sorting_analyzer
            if phy_folder.exists():
                try:
                    sorting = PhySortingExtractor(phy_folder)
                    print(f"Loaded from phy: {phy_folder}")
                except Exception as e:
                    print(f"Failed to load from phy: {e}")
                    if sorting_analyzer_folder.exists():
                        try:
                            sorting_analyzer = load_sorting_analyzer(sorting_analyzer_folder)
                            sorting = sorting_analyzer.sorting
                            print(f"Loaded from sorting_analyzer: {sorting_analyzer_folder}")
                        except Exception as e2:
                            print(f"Failed to load from sorting_analyzer: {e2}")
                            continue
                    else:
                        print(f"Could not load sorting from {sorting_results_folder}")
                        continue
            elif sorting_analyzer_folder.exists():
                try:
                    sorting_analyzer = load_sorting_analyzer(sorting_analyzer_folder)
                    sorting = sorting_analyzer.sorting
                    print(f"Loaded from sorting_analyzer: {sorting_analyzer_folder}")
                except Exception as e:
                    print(f"Failed to load from sorting_analyzer: {e}")
                    continue
            else:
                print(f"No valid sorting folder found in {sorting_results_folder}")
                continue
            
            unit_ids = sorting.unit_ids
            unit_qualities_this_sort = sorting.get_property('quality')
            
            if fs is None:
                fs = sorting.sampling_frequency
                print(f"Sampling frequency: {fs} Hz")
            
            for unit_id in unit_ids:
                spike_train = sorting.get_unit_spike_train(unit_id)

                # Align spike train from concatenated space to experiment-local (0-based) space
                task_end_eff = task_end if task_end is not None else int(spike_train[-1]) + 1
                task_mask = (spike_train >= task_start) & (spike_train < task_end_eff)
                spike_train = spike_train[task_mask] - task_start
                
                # Create structured data for this unit
                quality = unit_labels.get(f'shank{ish}', {}).get(str(unit_id), 'unknown')
                unit_data = {
                    'unit_id': unit_id,
                    'shank': ish,
                    'quality': quality,
                    'sorting_folder': str(sorting_results_folder),
                    'sampling_rate': fs,
                    'trials': []
                }
                
                # Process each trial
                for trial_idx in range(n_trials):
                    t_start = trial_start[trial_idx]
                    t_end = trial_end[trial_idx]
                    
                    # Define extended time windows (including pre/post periods)
                    pre_start_idx = t_start - int(pre_trial_window * fs)
                    post_end_idx = t_end + int(post_trial_window * fs)
                    
                    # Extract spikes in the extended trial window
                    trial_spike_mask = (spike_train >= pre_start_idx) & (spike_train < post_end_idx)
                    trial_spikes = spike_train[trial_spike_mask]
                    
                    # Convert spike times relative to trial start (in seconds)
                    relative_spike_times = (trial_spikes - t_start) / fs
                    
                    # Get stimulus condition for this trial
                    trial_rewarded_left = rewarded_on_left[trial_idx]

                    # Find condition index
                    condition_idx = 1 if trial_rewarded_left else 0

                    # Calculate repeat number for this condition
                    condition_trials = []
                    for prev_trial in range(trial_idx):
                        if rewarded_on_left[prev_trial] == trial_rewarded_left:
                            condition_trials.append(prev_trial)
                    repeat_idx = len(condition_trials)

                    # Calculate trial duration
                    trial_duration = (t_end - t_start) / fs

                    # Store trial information
                    trial_info = {
                        'trial_number': trial_idx,
                        'trial_start_time': t_start,  # in samples
                        'trial_end_time': t_end,      # in samples
                        'trial_start_time_sec': t_start / fs,  # in seconds
                        'trial_end_time_sec': t_end / fs,      # in seconds
                        'trial_duration': trial_duration,
                        'rewarded_on_left': bool(trial_rewarded_left),
                        'condition_idx': condition_idx,
                        'repeat_idx': repeat_idx,
                        'spike_times': relative_spike_times,  # relative to trial start
                        'pre_trial_spikes': relative_spike_times[relative_spike_times < 0],
                        'during_trial_spikes': relative_spike_times[(relative_spike_times >= 0) & 
                                                                    (relative_spike_times < trial_duration)],
                        'post_trial_spikes': relative_spike_times[relative_spike_times >= trial_duration],
                        'pre_trial_count': np.sum(relative_spike_times < 0),
                        'during_trial_count': np.sum((relative_spike_times >= 0) & 
                                                     (relative_spike_times < trial_duration)),
                        'post_trial_count': np.sum(relative_spike_times >= trial_duration),
                        'firing_rate_pre': np.sum(relative_spike_times < 0) / pre_trial_window,
                        'firing_rate_during': np.sum((relative_spike_times >= 0) & 
                                                     (relative_spike_times < trial_duration)) / trial_duration,
                        'firing_rate_post': np.sum(relative_spike_times >= trial_duration) / post_trial_window,
                    }
                    
                    unit_data['trials'].append(trial_info)
                
                all_units_data.append(unit_data)
    
    print(f"Processed {len(all_units_data)} units across {n_trials} trials")
    
    # Save the data to a PKL file using the unified schema
    save_behavior_trial_to_pkl(
        animal_id=animal_id,
        session_id=session_id,
        rec_folder=rec_folder,
        fs=fs,
        trial_start=trial_start,
        trial_end=trial_end,
        rewarded_on_left=rewarded_on_left,
        unique_conditions=unique_conditions,
        pre_trial_window=pre_trial_window,
        post_trial_window=post_trial_window,
        all_units_data=all_units_data,
        output_path=pkl_file,
    )

    return pkl_file


def save_behavior_trial_to_pkl(
    *,
    animal_id,
    session_id,
    rec_folder,
    fs,
    trial_start,
    trial_end,
    rewarded_on_left,
    unique_conditions,
    pre_trial_window,
    post_trial_window,
    all_units_data,
    output_path,
):
    """
    Save behavior trial responses in the SAME schema as the embedding extractor.
    This ensures compatibility with downstream analysis code.
    """

    # Create trial windows
    trial_windows = [(int(start), int(end)) for start, end in zip(trial_start, trial_end)]

    # Build all trial parameters list
    all_trial_parameters = []
    for trial_idx in range(len(trial_start)):
        trial_params = {
            'trial_index': trial_idx,
            'rewarded_on_left': bool(rewarded_on_left[trial_idx]),
            'trial_start': int(trial_start[trial_idx]),
            'trial_end': int(trial_end[trial_idx]),
            'trial_duration': float((trial_end[trial_idx] - trial_start[trial_idx]) / fs),
        }
        all_trial_parameters.append(trial_params)

    # Calculate average trial duration
    avg_trial_duration = np.mean((trial_end - trial_start) / fs)

    # Initialize the unified data structure
    neural_data = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'task_file': None,  # Not applicable for this experiment
            'extraction_date': datetime.now().isoformat(),
            'n_trials': len(trial_start),
            'experiment_type': 'behavior_trial',
            'sampling_frequency': fs,
        },
        'experiment_parameters': {
            'stimulus_duration': None,  # Variable per trial
            'iti_duration': None,  # Not extracted
            'trial_duration': float(avg_trial_duration),
            'total_trials': len(trial_start),
            'n_conditions': len(unique_conditions),
            'n_left_trials': int(np.sum(rewarded_on_left)),
            'n_right_trials': int(np.sum(~rewarded_on_left)),
        },
        'trial_info': {
            'rewarded_on_left': rewarded_on_left.tolist(),
            'unique_conditions': unique_conditions.tolist(),
            'trial_windows': trial_windows,
            'all_trial_parameters': all_trial_parameters,
            'trial_durations': [(trial_end[i] - trial_start[i]) / fs 
                               for i in range(len(trial_start))],
        },
        'spike_data': {},
        'unit_info': {},
        'extraction_params': {
            'window_pre': pre_trial_window,
            'window_post': post_trial_window,
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
                'rewarded_on_left': t['rewarded_on_left'],
                'condition_idx': t['condition_idx'],
                'spike_times': t['spike_times'].tolist(),
                'spike_count': len(t['spike_times']),
                'trial_start': t['trial_start_time'],
                'trial_end': t['trial_end_time'],
                'trial_duration': t['trial_duration'],
                # Additional metrics preserved for analysis
                'pre_trial_count': t.get('pre_trial_count'),
                'during_trial_count': t.get('during_trial_count'),
                'post_trial_count': t.get('post_trial_count'),
                'firing_rate_pre': t.get('firing_rate_pre'),
                'firing_rate_during': t.get('firing_rate_during'),
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
    # Example 1: Manual input
    # rec_folder = r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/251002/CnL42SG/CnL42SG_20251002_200839.rec"
    rec_folder  = r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260304/CnL42_20260304/CnL42_task_20260304_171711.rec"
    dio_folders = DIO.get_dio_folders(rec_folder)
    dio_folders = sorted(dio_folders, key=lambda x:x.name)
    trial_pd_time, trial_pd_state = DIO.concatenate_din_data(dio_folders, 5)
    trial_pd_time = trial_pd_time - trial_pd_time[0] # reset the time to start from 0

    trial_start = trial_pd_time[np.where(trial_pd_state==1)[0]]
    trial_end = trial_pd_time[np.where(trial_pd_state==0)[0][1:]]
    # rec_folder = Path(input("Please enter the full path to the recording folder: ").strip().strip('"'))
    task_file = r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260304/CnL42_2026-03-04_Session001_Data.json"

    rewarded_on_left = get_rewarded_on_left(task_file)
    # Load your trial data (modify this based on how you store your data)
    # Option A: If you have a .mat file with trial info
    # import scipy.io
    # trial_data = scipy.io.loadmat('trial_info.mat')
    # trial_start = trial_data['trial_start'].flatten()
    # trial_end = trial_data['trial_end'].flatten()
    # rewarded_on_left = trial_data['rewarded_on_left'].flatten().astype(bool)
    
    # Option B: If you have numpy arrays
    # trial_start = np.load('trial_start.npy')
    # trial_end = np.load('trial_end.npy')
    # rewarded_on_left = np.load('rewarded_on_left.npy')
    
    # Option C: Example dummy data for testing
    # n_trials = 160
    # trial_start = np.array([i * 30000 for i in range(n_trials)])  # Example: trials every 30000 samples
    # trial_end = np.array([i * 30000 + 15000 for i in range(n_trials)])  # Example: 15000 samples duration
    # rewarded_on_left = np.random.rand(n_trials) > 0.5  # Random left/right
    n_DIO = len(trial_start)
    n_trials = len(rewarded_on_left)

    if n_DIO != n_trials:
        raise ValueError(f"DIO trials ({n_DIO}) != behavior trials ({n_trials}). Please check your data.")
    else:
        print(f"DIO and behavior trial counts match: {n_trials} trials.")
    print(f"Processing {n_trials} trials...")
    print(f"Trial start range: {trial_start[0]} to {trial_start[-1]}")
    print(f"Trial end range: {trial_end[0]} to {trial_end[-1]}")

    sortout_folder = input("Please enter the path to the session sortout folder (parent of shank0, shank1, ... folders): ").strip().strip('"')

    # Offset of this experiment in the concatenated recording (in sample points)
    task_start = 259297964      # sample point where this experiment starts in the concatenated recording
    task_end   = 323137986   # sample point where it ends; None = use last spike as upper bound

    # Process the data
    pkl_path = process_behavior_trial_responses(
        rec_folder=rec_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        rewarded_on_left=rewarded_on_left,
        sortout_folder=sortout_folder,
        task_start=task_start,
        task_end=task_end,
        overwrite=True
    )

    print(f"\nData saved to: {pkl_path}")
    
    # Load and verify the structure
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nData structure verification:")
    print(f"- Metadata keys: {list(data['metadata'].keys())}")
    print(f"- Experiment type: {data['metadata']['experiment_type']}")
    print(f"- Total units: {len(data['spike_data'])}")
    print(f"- Total trials: {data['metadata']['n_trials']}")
    print(f"- Conditions: {data['trial_info']['unique_conditions']}")
    print(f"- Left trials: {data['experiment_parameters']['n_left_trials']}")
    print(f"- Right trials: {data['experiment_parameters']['n_right_trials']}")