import os
import numpy as np
import scipy.io
import h5py
from pathlib import Path
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
import pandas as pd
import pickle

def get_left_right_objects(data):
    """
    Extract left and right object indices for each trial.
    
    Parameters:
    data: DataFrame or list of dictionaries with 'target_index', 'distractor_index', 'target_position'
    
    Returns:
    DataFrame with trial info and left/right object indices
    """
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Initialize columns for left and right objects
    df['left_object_index'] = None
    df['right_object_index'] = None
    
    # Assign objects based on target position
    for idx, row in df.iterrows():
        if row['target_position'] == 'LEFT':
            df.at[idx, 'left_object_index'] = row['target_index']
            df.at[idx, 'right_object_index'] = row['distractor_index']
        else:  # target_position == 'RIGHT'
            df.at[idx, 'left_object_index'] = row['distractor_index']
            df.at[idx, 'right_object_index'] = row['target_index']
    
    return df

def extract_temporal_spike_features(spike_train, trial_windows, fs, n_time_bins=20):
    """
    Extract temporal features from spike trains for each trial
    
    Parameters:
    spike_train: Array of spike times (in sample indices)
    trial_windows: List of (start_sample, end_sample) for each trial
    fs: Sampling frequency
    n_time_bins: Number of time bins to divide each trial into
    
    Returns:
    Dictionary with various temporal features
    """
    
    n_trials = len(trial_windows)
    
    # Initialize feature arrays
    binned_spike_counts = np.zeros((n_trials, n_time_bins))
    binned_firing_rates = np.zeros((n_trials, n_time_bins))
    first_spike_latencies = np.zeros(n_trials)
    last_spike_times = np.zeros(n_trials)
    total_spike_counts = np.zeros(n_trials)
    mean_firing_rates = np.zeros(n_trials)
    spike_time_variances = np.zeros(n_trials)
    inter_spike_intervals = []
    
    for trial_idx, (start_sample, end_sample) in enumerate(trial_windows):
        # Get spikes in this trial
        trial_spikes = spike_train[(spike_train >= start_sample) & (spike_train < end_sample)]
        trial_duration_samples = end_sample - start_sample
        trial_duration_sec = trial_duration_samples / fs
        
        # Basic metrics
        total_spike_counts[trial_idx] = len(trial_spikes)
        mean_firing_rates[trial_idx] = len(trial_spikes) / trial_duration_sec if trial_duration_sec > 0 else 0
        
        if len(trial_spikes) > 0:
            # Convert to relative times within trial (in seconds)
            trial_spikes_relative = (trial_spikes - start_sample) / fs
            
            # First and last spike timing
            first_spike_latencies[trial_idx] = trial_spikes_relative[0]
            last_spike_times[trial_idx] = trial_spikes_relative[-1]
            
            # Spike time variance (measure of temporal spread)
            spike_time_variances[trial_idx] = np.var(trial_spikes_relative)
            
            # Inter-spike intervals for this trial
            if len(trial_spikes) > 1:
                isis = np.diff(trial_spikes_relative)
                inter_spike_intervals.extend(isis)
            
            # Binned spike counts and rates
            bin_edges = np.linspace(0, trial_duration_sec, n_time_bins + 1)
            for bin_idx in range(n_time_bins):
                bin_start_time = bin_edges[bin_idx]
                bin_end_time = bin_edges[bin_idx + 1]
                bin_duration = bin_end_time - bin_start_time
                
                spikes_in_bin = np.sum((trial_spikes_relative >= bin_start_time) & 
                                     (trial_spikes_relative < bin_end_time))
                
                binned_spike_counts[trial_idx, bin_idx] = spikes_in_bin
                binned_firing_rates[trial_idx, bin_idx] = spikes_in_bin / bin_duration if bin_duration > 0 else 0
        else:
            # No spikes in trial
            first_spike_latencies[trial_idx] = trial_duration_sec  # Max possible latency
            last_spike_times[trial_idx] = 0
            spike_time_variances[trial_idx] = 0
    
    return {
        'binned_spike_counts': binned_spike_counts,
        'binned_firing_rates': binned_firing_rates,
        'first_spike_latencies': first_spike_latencies,
        'last_spike_times': last_spike_times,
        'total_spike_counts': total_spike_counts,
        'mean_firing_rates': mean_firing_rates,
        'spike_time_variances': spike_time_variances,
        'all_inter_spike_intervals': np.array(inter_spike_intervals) if inter_spike_intervals else np.array([]),
        'n_time_bins': n_time_bins
    }

def process_object_discrimination_with_temporal_features(rec_folder, task_file, overwrite=True, n_time_bins=20):
    """
    Process object discrimination responses WITH temporal spike information.
    
    Enhanced version that extracts detailed temporal features from spike trains
    
    Parameters:
      rec_folder (str or Path): Path to the recording folder.
      task_file (str or Path): Path to the task pickle file.
      overwrite (bool): If False and the NPZ file already exists, skip writing. 
                        If True, overwrite any existing NPZ file.
      n_time_bins (int): Number of time bins for temporal analysis
    
    Returns:
      npz_file (Path): Path to the saved (or existing) NPZ file.
    """
    
    rec_folder = Path(rec_folder)
    task_file = Path(task_file)
    
    # Import DIO processing
    import process_func.DIO as DIO
    
    # Get animal and session IDs
    animal_id = rec_folder.name.split('.')[0].split('_')[0]
    session_id = rec_folder.name.split('.')[0]
    
    print(f"Processing {animal_id}/{session_id} with temporal features")
    print(f"Using {n_time_bins} time bins per trial")
    
    # Load DIO data
    dio_folders = DIO.get_dio_folders(rec_folder)
    dio_folders = sorted(dio_folders, key=lambda x: x.name)
    
    pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 3)
    
    # Load task data
    with open(task_file, 'rb') as f:
        task = pickle.load(f)
    
    summary = task['summary']
    trials = task['trials']
    
    # Find edges
    rising_edge = np.where(pd_state == 1)[0]
    falling_edge = np.where(pd_state == 0)[0][1:]
    
    print(f"Number of rising edges: {len(rising_edge)}")
    print(f"Number of falling edges: {len(falling_edge)}")
    
    # Get edge times in sample indices
    rising_edge_samples = pd_time[rising_edge]
    falling_edge_samples = pd_time[falling_edge]
    
    # Process trial data
    df = pd.DataFrame(trials)
    trial_data = get_left_right_objects(df)
    
    n_trials = len(trial_data)
    max_processable_trials = min(len(rising_edge_samples), len(falling_edge_samples), n_trials)
    print(f"Processing {max_processable_trials} trials")
    
    # Construct session folder
    code_folder = Path(__file__).parent.parent.parent
    session_folder = code_folder / f"sortout/{animal_id}/{session_id}"
    
    # Check if output file exists
    npz_file = session_folder / 'object_discrimination_temporal_responses.npz'
    if npz_file.exists() and not overwrite:
        print(f"File {npz_file} already exists and overwrite=False. Skipping computation.")
        return npz_file
    
    all_units_responses = []
    unit_info = []
    all_unit_qualities = []
    
    ishs = ['0', '1', '2', '3']
    fs = None
    
    for ish in ishs:
        print(f'Processing shank {ish}')
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
            phy_folder = Path(sorting_results_folder) / 'phy'
            out_fig_folder = Path(sorting_results_folder) / 'temporal_analysis'
            if not out_fig_folder.exists():
                out_fig_folder.mkdir(parents=True)
            
            try:
                # Load sorting analyzer
                sorting_analyzer = load_sorting_analyzer(Path(sorting_results_folder) / 'sorting_analyzer')
                sorting = sorting_analyzer.sorting
                
                # Alternative: load from Phy if needed
                if phy_folder.exists():
                    sorting = PhySortingExtractor(phy_folder)
                
                unit_ids = sorting.unit_ids
                unit_qualities_this_sort = sorting.get_property('quality') if hasattr(sorting, 'get_property') else ['good'] * len(unit_ids)
                
                # Get sampling frequency
                if fs is None:
                    fs = sorting.sampling_frequency
                    print(f"Using sampling frequency: {fs} Hz")
                
                print(f"Processing {len(unit_ids)} units from {sorting_results_folder}")
                
                for i, unit_id in enumerate(unit_ids):
                    # GET THE ACTUAL SPIKE TIMES HERE!
                    spike_train = sorting.get_unit_spike_train(unit_id)  # Raw spike sample indices
                    
                    print(f"Unit {unit_id}: {len(spike_train)} spikes, range {spike_train[0] if len(spike_train) > 0 else 'N/A'} to {spike_train[-1] if len(spike_train) > 0 else 'N/A'}")
                    
                    # Time windows for analysis
                    visual_transmission_delay = 0.05  # seconds
                    baseline_window = 0.2  # seconds
                    iti_window = 0.5  # seconds
                    
                    # Create trial windows for different phases
                    display_windows = []
                    baseline_windows = []
                    iti_windows = []
                    
                    for trial_idx in range(max_processable_trials):
                        # Display period
                        display_start = rising_edge_samples[trial_idx] + int(visual_transmission_delay * fs)
                        display_end = falling_edge_samples[trial_idx]
                        display_windows.append((display_start, display_end))
                        
                        # Baseline period
                        baseline_start = rising_edge_samples[trial_idx] - int(baseline_window * fs)
                        baseline_end = rising_edge_samples[trial_idx]
                        baseline_windows.append((baseline_start, baseline_end))
                        
                        # ITI period
                        if trial_idx < max_processable_trials - 1:
                            iti_start = falling_edge_samples[trial_idx]
                            iti_end = min(iti_start + int(iti_window * fs), 
                                        rising_edge_samples[trial_idx + 1])
                        else:
                            iti_start = falling_edge_samples[trial_idx]
                            iti_end = iti_start + int(iti_window * fs)
                        iti_windows.append((iti_start, iti_end))
                    
                    # Extract temporal features for each phase
                    print(f"  Extracting temporal features for unit {unit_id}...")
                    
                    display_temporal = extract_temporal_spike_features(spike_train, display_windows, fs, n_time_bins)
                    baseline_temporal = extract_temporal_spike_features(spike_train, baseline_windows, fs, n_time_bins)
                    iti_temporal = extract_temporal_spike_features(spike_train, iti_windows, fs, n_time_bins)
                    
                    # Store comprehensive response data
                    unit_responses = {
                        # Original firing rates (for backward compatibility)
                        'display_responses': display_temporal['mean_firing_rates'],
                        'baseline_responses': baseline_temporal['mean_firing_rates'],
                        'iti_responses': iti_temporal['mean_firing_rates'],
                        
                        # NEW: Temporal features
                        'display_temporal': display_temporal,
                        'baseline_temporal': baseline_temporal,
                        'iti_temporal': iti_temporal,
                        
                        # Trial information
                        'trial_data': trial_data.iloc[:max_processable_trials].copy(),
                        
                        # Spike train metadata
                        'unit_id': unit_id,
                        'total_spikes': len(spike_train),
                        'spike_train_range': [spike_train[0], spike_train[-1]] if len(spike_train) > 0 else [0, 0]
                    }
                    
                    all_units_responses.append(unit_responses)
                    unit_info.append((ish, unit_id))
                    all_unit_qualities.append(unit_qualities_this_sort[i] if isinstance(unit_qualities_this_sort, (list, np.ndarray)) else 'good')
                    
                    # Print debug info for first unit
                    if len(all_units_responses) == 1:
                        print(f"Debug info for first unit:")
                        print(f"  Display firing rates: mean={np.mean(display_temporal['mean_firing_rates']):.3f}, max={np.max(display_temporal['mean_firing_rates']):.3f}")
                        print(f"  Temporal bins shape: {display_temporal['binned_firing_rates'].shape}")
                        print(f"  First spike latencies: mean={np.mean(display_temporal['first_spike_latencies']):.3f}s")
                        print(f"  Spike time variance: mean={np.mean(display_temporal['spike_time_variances']):.3f}")
                    
            except Exception as e:
                print(f"Error processing {sorting_results_folder}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"Processed {len(all_units_responses)} units total")
    
    if len(all_units_responses) == 0:
        print("WARNING: No units were processed successfully!")
        return None
    
    # Print overall statistics
    all_display_rates = [np.mean(unit['display_responses']) for unit in all_units_responses]
    all_baseline_rates = [np.mean(unit['baseline_responses']) for unit in all_units_responses]
    all_iti_rates = [np.mean(unit['iti_responses']) for unit in all_units_responses]
    
    print(f"Overall firing rate statistics:")
    print(f"  Display: mean={np.mean(all_display_rates):.3f} Hz, max={np.max(all_display_rates):.3f} Hz")
    print(f"  Baseline: mean={np.mean(all_baseline_rates):.3f} Hz, max={np.max(all_baseline_rates):.3f} Hz")
    print(f"  ITI: mean={np.mean(all_iti_rates):.3f} Hz, max={np.max(all_iti_rates):.3f} Hz")
    
    # Print temporal statistics
    all_display_variance = [np.mean(unit['display_temporal']['spike_time_variances']) for unit in all_units_responses]
    all_first_latencies = [np.mean(unit['display_temporal']['first_spike_latencies']) for unit in all_units_responses]
    
    print(f"Temporal statistics:")
    print(f"  Mean spike time variance: {np.mean(all_display_variance):.6f}")
    print(f"  Mean first spike latency: {np.mean(all_first_latencies):.3f}s")
    
    # Prepare summary statistics
    summary_stats = {
        'n_units': len(all_units_responses),
        'n_trials_processed': max_processable_trials,
        'n_time_bins': n_time_bins,
        'unique_left_objects': sorted(trial_data['left_object_index'].unique()),
        'unique_right_objects': sorted(trial_data['right_object_index'].unique()),
        'unique_positions': sorted(trial_data['target_position'].unique()),
        'sampling_frequency': fs,
        'has_temporal_features': True
    }
    
    # Save the data
    session_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving temporal data to {npz_file} (overwrite={overwrite})")
    
    np.savez(
        npz_file,
        all_units_responses=all_units_responses,
        unit_info=unit_info,
        unit_qualities=all_unit_qualities,
        trial_data=trial_data.iloc[:max_processable_trials],
        rising_edge_time=rising_edge_samples[:max_processable_trials],
        falling_edge_time=falling_edge_samples[:max_processable_trials],
        summary_stats=summary_stats,
        task_summary=summary
    )
    
    print("Temporal data saved successfully!")
    print(f"Each unit now has:")
    print(f"  - Binned firing rates: {n_time_bins} bins per trial")
    print(f"  - Spike timing features: latencies, variances, ISIs")
    print(f"  - Traditional firing rates: for compatibility")
    
    return npz_file

# Example usage
if __name__ == '__main__':
    # Your existing paths
    rec_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\250609\CnL22SG\CnL22SG_20250609_164650.rec"
    task_file = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\250609\CnL22SG\CnL22_20250609_1.pkl"
    
    print(f"Recording folder: {rec_folder}")
    print(f"Task file: {task_file}")
    
    # Process with temporal features
    npz_path = process_object_discrimination_with_temporal_features(
        rec_folder, 
        task_file, 
        overwrite=True,
        n_time_bins=20  # Divide each trial into 20 time bins
    )
    print(f"Results saved to: {npz_path}")