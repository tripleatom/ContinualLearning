import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import os
import h5py
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rf_recon.rf_grid.parse_grating_experiment import parse_grating_experiment


def extract_grating_neural_data_for_embedding(rec_folder, task_file_path, output_format='pickle'):
    """
    Extract and save neural spike data and grating trial parameters for embedding analysis.
    
    Parameters:
    -----------
    rec_folder : Path or str
        Path to the recording folder
    task_file_path : Path or str  
        Path to the grating task file
    output_format : str
        Format to save data ('hdf5', 'pickle', or 'npz')
        
    Returns:
    --------
    dict : Dictionary containing all extracted grating data
    """
    
    # Setup paths
    rec_folder = Path(rec_folder)
    task_file_path = Path(task_file_path)
    
    # More robust parsing of animal and session IDs
    rec_name = rec_folder.name
    if rec_name.endswith('.rec'):
        rec_name = rec_name[:-4]  # Remove .rec extension
    
    # Split by underscore and take first part as animal_id
    name_parts = rec_name.split('_')
    animal_id = name_parts[0]
    session_id = rec_name
    
    print(f"Extracting grating data for {animal_id}/{session_id}")
    
    # Load grating task data with error handling
    try:
        task_file = parse_grating_experiment(task_file_path)
        df = task_file['trial_data']
    except Exception as e:
        print(f"Error parsing grating experiment: {e}")
        return None
    
    # Extract timing parameters with better error handling
    try:
        stimulus_duration = task_file['parameters']['stimulus_duration']
        ITI_duration = task_file['parameters']['iti_duration']
        
        # Handle string values with 's' suffix
        if isinstance(stimulus_duration, str):
            stimulus_duration = float(stimulus_duration.rstrip('s'))
        else:
            stimulus_duration = float(stimulus_duration)
            
        if isinstance(ITI_duration, str):
            ITI_duration = float(ITI_duration.rstrip('s'))
        else:
            ITI_duration = float(ITI_duration)
            
        n_repeats = task_file['parameters']['total_trials']
        trial_duration = stimulus_duration + ITI_duration
    except Exception as e:
        print(f"Error extracting timing parameters: {e}")
        # Set default values if extraction fails
        stimulus_duration = 2.0
        ITI_duration = 1.0
        n_repeats = len(df) if df is not None else 0
        trial_duration = stimulus_duration + ITI_duration
    
    print(f"Stimulus duration: {stimulus_duration}s")
    print(f"ITI duration: {ITI_duration}s")
    print(f"Total repeats: {n_repeats}")
    print(f"Trial duration: {trial_duration}s")
    
    # Load DIO data with error handling
    task_id = task_file_path.stem
    task_folder = task_file_path.parent
    processed_dio_folder = task_folder / f"{task_id}_DIO.npz"
    
    try:
        dio_data = np.load(processed_dio_folder)
        rising_times = dio_data['rising_times']
        falling_times = dio_data['falling_times']
    except Exception as e:
        print(f"Error loading DIO data from {processed_dio_folder}: {e}")
        return None
    
    # Validate trial data
    if df is None or len(df) == 0:
        print("No trial data found")
        return None
    
    # Trial windows and orientations
    n_trials = len(df)
    
    # Check if L_Orient column exists
    if 'L_Orient' not in df.columns:
        print(f"L_Orient column not found in trial data. Available columns: {list(df.columns)}")
        # Look for alternative orientation columns
        orientation_cols = [col for col in df.columns if 'orient' in col.lower()]
        if orientation_cols:
            print(f"Using {orientation_cols[0]} as orientation column")
            L_Orient = df[orientation_cols[0]].values
        else:
            print("No orientation column found, creating dummy orientations")
            L_Orient = np.zeros(n_trials)  # Default to 0 degrees
    else:
        L_Orient = df['L_Orient'].values
    
    orientations = L_Orient
    unique_orientations = np.unique(orientations)

    print(f"Unique orientations: {unique_orientations}")
    print(f"Number of trials: {n_trials}")

    # Validate DIO timing data
    if len(rising_times) < n_trials or len(falling_times) < n_trials:
        print(f"Warning: DIO timing mismatch. Trials: {n_trials}, Rising: {len(rising_times)}, Falling: {len(falling_times)}")
        # Use the minimum number of trials available
        n_trials = min(n_trials, len(rising_times), len(falling_times))
        orientations = orientations[:n_trials]
        df = df.iloc[:n_trials]
        print(f"Adjusted to {n_trials} trials")

    trial_windows = [(rising_times[i], falling_times[i]) for i in range(n_trials)]
    
    # Setup session folder structure with better path handling
    code_folder = Path(__file__).parent.parent.parent.parent
    session_folder = code_folder / f"sortout/{animal_id}/{session_id}"
    
    # Create output directory if it doesn't exist
    output_dir = session_folder / 'embedding_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data structure to store everything
    neural_data = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'task_file': str(task_file_path),
            'extraction_date': datetime.now().isoformat(),
            'n_trials': n_trials,
            'experiment_type': 'grating'
        },
        'experiment_parameters': {
            'stimulus_duration': stimulus_duration,
            'iti_duration': ITI_duration,
            'trial_duration': trial_duration,
            'total_trials': n_repeats
        },
        'trial_info': {
            'orientations': orientations.tolist(),
            'unique_orientations': unique_orientations.tolist(),
            'trial_windows': trial_windows,
            'all_trial_parameters': df.to_dict('records')
        },
        'spike_data': {},
        'unit_info': {}
    }
    
    # Print trial distribution
    for orientation in unique_orientations:
        n_trials_orient = np.sum(orientations == orientation)
        print(f"Orientation {orientation}°: {n_trials_orient} trials")
    
    print(f"Trial parameters extracted: {list(df.columns)}")
    
    # Window parameters for spike extraction
    window_pre = 0.2   # seconds before stimulus onset
    window_post = 2.0  # seconds after stimulus onset
    
    # Initialize spike data storage
    unit_counter = 0
    ishs = ['0', '1', '2', '3']
    
    # Check if session folder exists
    if not session_folder.exists():
        print(f"Warning: Session folder {session_folder} does not exist")
        print("Creating basic folder structure...")
        session_folder.mkdir(parents=True, exist_ok=True)
    
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
        
        if not sorting_results_folders:
            print(f"No sorting results found in {shank_folder}")
            continue
        
        for sorting_results_folder in sorting_results_folders:
            phy_folder = Path(sorting_results_folder) / 'phy'
            
            try:
                # Load sorting data
                sorting_analyzer_path = Path(sorting_results_folder) / 'sorting_analyzer'
                sorting = None
                fs = None
                
                if phy_folder.exists():
                    print(f"Loading Phy sorting from {phy_folder}")
                    sorting = PhySortingExtractor(phy_folder)
                    fs = sorting.sampling_frequency
                elif sorting_analyzer_path.exists():
                    print(f"Loading sorting analyzer from {sorting_analyzer_path}")
                    sorting_analyzer = load_sorting_analyzer(sorting_analyzer_path)
                    sorting = sorting_analyzer.sorting
                    fs = sorting.sampling_frequency
                else:
                    print(f"No valid sorting data found in {sorting_results_folder}")
                    continue
                
                if sorting is None:
                    print(f"Failed to load sorting from {sorting_results_folder}")
                    continue
                
                neural_data['metadata']['sampling_frequency'] = fs
                
                print(f"Processing {sorting_results_folder} with {len(sorting.unit_ids)} units at {fs} Hz")
                
                # Get unit qualities with proper error handling
                unit_ids = sorting.unit_ids
                unit_qualities = []
                
                try:
                    if hasattr(sorting, 'get_property'):
                        unit_qualities = sorting.get_property('quality')
                    else:
                        unit_qualities = ['good'] * len(unit_ids)
                except:
                    unit_qualities = ['good'] * len(unit_ids)
                
                # Ensure unit_qualities has the right length
                if len(unit_qualities) != len(unit_ids):
                    unit_qualities = ['good'] * len(unit_ids)
                
                for unit_idx, unit_id in enumerate(unit_ids):
                    quality = unit_qualities[unit_idx] if unit_idx < len(unit_qualities) else 'unknown'
                    
                    # Skip noise units
                    if quality == 'noise':
                        continue
                    
                    print(f"Processing unit {unit_id} (quality: {quality})")
                    
                    # Create unique unit identifier
                    unique_unit_id = f"shank{ish}_sorting{Path(sorting_results_folder).name}_unit{unit_id}"
                    
                    # Get spike train with error handling
                    try:
                        spike_train = sorting.get_unit_spike_train(unit_id)
                    except Exception as e:
                        print(f"Error getting spike train for unit {unit_id}: {e}")
                        continue
                    
                    if len(spike_train) == 0:
                        print(f"Unit {unit_id} has no spikes, skipping...")
                        continue
                    
                    # Store unit metadata
                    neural_data['unit_info'][unique_unit_id] = {
                        'original_unit_id': int(unit_id),
                        'shank': ish,
                        'quality': quality,
                        'sorting_folder': sorting_results_folder,
                        'n_spikes_total': len(spike_train),
                        'unit_index': unit_counter
                    }
                    
                    # Extract spikes for each trial
                    trial_spike_data = []
                    
                    for i_trial, (start, end) in enumerate(trial_windows):
                        # Convert to samples for indexing
                        start_samples = int(start)
                        window_pre_samples = int(window_pre * fs)
                        window_post_samples = int(window_post * fs)
                        
                        # Extract spikes in window around stimulus onset
                        trial_spikes = spike_train[
                            (spike_train >= start_samples - window_pre_samples) & 
                            (spike_train < start_samples + window_post_samples)
                        ]
                        
                        # Convert to relative time (seconds from stimulus onset)
                        if len(trial_spikes) > 0:
                            trial_spikes_relative = (trial_spikes - start_samples) / fs
                        else:
                            trial_spikes_relative = np.array([])
                        
                        # Store trial data
                        trial_data = {
                            'trial_index': i_trial,
                            'orientation': orientations[i_trial] if i_trial < len(orientations) else None,
                            'spike_times': trial_spikes_relative.tolist(),
                            'spike_count': len(trial_spikes_relative),
                            'trial_start': start,
                            'trial_end': end
                        }
                        
                        trial_spike_data.append(trial_data)
                    
                    neural_data['spike_data'][unique_unit_id] = trial_spike_data
                    unit_counter += 1
                    
            except Exception as e:
                print(f"Error processing {sorting_results_folder}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Add final parameters
    neural_data['extraction_params'] = {
        'window_pre': window_pre,
        'window_post': window_post,
        'total_units': unit_counter,
        'shanks_processed': ishs
    }
    
    print(f"\nGrating data extraction complete!")
    print(f"Total units processed: {unit_counter}")
    print(f"Total trials: {n_trials}")
    print(f"Orientations tested: {unique_orientations}")
    
    # Save data
    base_filename = f"{animal_id}_{session_id}_grating_data"
    
    try:
        if output_format == 'hdf5':
            save_to_hdf5(neural_data, output_dir / f"{base_filename}.h5")
        elif output_format == 'pickle':
            save_to_pickle(neural_data, output_dir / f"{base_filename}.pkl")
        elif output_format == 'npz':
            save_to_npz(neural_data, output_dir / f"{base_filename}.npz")
        else:
            raise ValueError("output_format must be 'hdf5', 'pickle', or 'npz'")
    except Exception as e:
        print(f"Error saving data: {e}")
    
    return neural_data


def save_to_pickle(data, filepath):
    """Save neural data to pickle format"""
    print(f"Saving data to pickle: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_to_hdf5(data, filepath):
    """Save neural data to HDF5 format"""
    print(f"Saving data to HDF5: {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata
        metadata_grp = f.create_group('metadata')
        for key, value in data['metadata'].items():
            if isinstance(value, str):
                metadata_grp.attrs[key] = value
            else:
                metadata_grp.create_dataset(key, data=value)
        
        # Save experiment parameters
        exp_grp = f.create_group('experiment_parameters')
        for key, value in data['experiment_parameters'].items():
            exp_grp.attrs[key] = value
        
        # Save trial info
        trial_grp = f.create_group('trial_info')
        trial_grp.create_dataset('orientations', data=data['trial_info']['orientations'])
        trial_grp.create_dataset('unique_orientations', data=data['trial_info']['unique_orientations'])
        
        # Save trial windows
        trial_windows_data = np.array(data['trial_info']['trial_windows'])
        trial_grp.create_dataset('trial_windows', data=trial_windows_data)
        
        # Save trial parameters as JSON string (HDF5 doesn't handle nested dicts well)
        trial_grp.attrs['all_trial_parameters'] = json.dumps(data['trial_info']['all_trial_parameters'])
        
        # Save spike data
        spike_grp = f.create_group('spike_data')
        for unit_id, trials_data in data['spike_data'].items():
            unit_grp = spike_grp.create_group(unit_id)
            
            # Create datasets for each trial
            for i, trial_data in enumerate(trials_data):
                trial_grp = unit_grp.create_group(f'trial_{i:04d}')
                trial_grp.attrs['trial_index'] = trial_data['trial_index']
                trial_grp.attrs['orientation'] = trial_data['orientation'] if trial_data['orientation'] is not None else -999
                trial_grp.attrs['spike_count'] = trial_data['spike_count']
                trial_grp.attrs['trial_start'] = trial_data['trial_start']
                trial_grp.attrs['trial_end'] = trial_data['trial_end']
                trial_grp.create_dataset('spike_times', data=trial_data['spike_times'])
        
        # Save unit info
        unit_grp = f.create_group('unit_info')
        for unit_id, info in data['unit_info'].items():
            unit_info_grp = unit_grp.create_group(unit_id)
            for key, value in info.items():
                if isinstance(value, str):
                    unit_info_grp.attrs[key] = value
                else:
                    unit_info_grp.create_dataset(key, data=value)
        
        # Save extraction params
        params_grp = f.create_group('extraction_params')
        for key, value in data['extraction_params'].items():
            if isinstance(value, (list, np.ndarray)):
                params_grp.create_dataset(key, data=value)
            else:
                params_grp.attrs[key] = value


def save_to_npz(data, filepath):
    """Save neural data to NPZ format"""
    print(f"Saving data to NPZ: {filepath}")
    
    # Flatten the structure for npz
    save_dict = {}
    
    # Metadata
    for key, value in data['metadata'].items():
        save_dict[f'metadata_{key}'] = value
    
    # Experiment parameters
    for key, value in data['experiment_parameters'].items():
        save_dict[f'exp_{key}'] = value
    
    # Trial info (basic arrays)
    save_dict['trial_orientations'] = data['trial_info']['orientations']
    save_dict['trial_unique_orientations'] = data['trial_info']['unique_orientations']
    save_dict['trial_windows'] = np.array(data['trial_info']['trial_windows'])
    
    # Extraction params
    for key, value in data['extraction_params'].items():
        save_dict[f'params_{key}'] = value
    
    # Save basic arrays
    np.savez_compressed(filepath, **save_dict)
    
    # Save complex nested data as pickle alongside
    complex_data = {
        'spike_data': data['spike_data'],
        'unit_info': data['unit_info'],
        'trial_parameters': data['trial_info']['all_trial_parameters']
    }
    pickle_path = filepath.with_suffix('.complex.pkl')
    save_to_pickle(complex_data, pickle_path)


def load_neural_data(filepath):
    """Load neural data from saved file"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        return load_from_hdf5(filepath)
    elif filepath.suffix == '.pkl':
        return load_from_pickle(filepath)
    elif filepath.suffix == '.npz':
        return load_from_npz(filepath)
    else:
        raise ValueError("Unsupported file format")


def load_from_pickle(filepath):
    """Load neural data from pickle format"""
    print(f"Loading data from pickle: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_from_hdf5(filepath):
    """Load neural data from HDF5 format"""
    print(f"Loading data from HDF5: {filepath}")
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        data['metadata'] = {}
        metadata_grp = f['metadata']
        for key in metadata_grp.attrs.keys():
            data['metadata'][key] = metadata_grp.attrs[key]
        for key in metadata_grp.keys():
            data['metadata'][key] = metadata_grp[key][()]
        
        # Load experiment parameters
        data['experiment_parameters'] = {}
        exp_grp = f['experiment_parameters']
        for key in exp_grp.attrs.keys():
            data['experiment_parameters'][key] = exp_grp.attrs[key]
        
        # Load trial info
        data['trial_info'] = {}
        trial_grp = f['trial_info']
        data['trial_info']['orientations'] = trial_grp['orientations'][()].tolist()
        data['trial_info']['unique_orientations'] = trial_grp['unique_orientations'][()].tolist()
        data['trial_info']['trial_windows'] = trial_grp['trial_windows'][()].tolist()
        data['trial_info']['all_trial_parameters'] = json.loads(trial_grp.attrs['all_trial_parameters'])
        
        # Load spike data
        data['spike_data'] = {}
        spike_grp = f['spike_data']
        for unit_id in spike_grp.keys():
            unit_grp = spike_grp[unit_id]
            trials_data = []
            
            # Sort trial keys to maintain order
            trial_keys = sorted([k for k in unit_grp.keys() if k.startswith('trial_')])
            
            for trial_key in trial_keys:
                trial_grp = unit_grp[trial_key]
                trial_data = {
                    'trial_index': trial_grp.attrs['trial_index'],
                    'orientation': trial_grp.attrs['orientation'] if trial_grp.attrs['orientation'] != -999 else None,
                    'spike_count': trial_grp.attrs['spike_count'],
                    'trial_start': trial_grp.attrs['trial_start'],
                    'trial_end': trial_grp.attrs['trial_end'],
                    'spike_times': trial_grp['spike_times'][()].tolist()
                }
                trials_data.append(trial_data)
            
            data['spike_data'][unit_id] = trials_data
        
        # Load unit info
        data['unit_info'] = {}
        unit_grp = f['unit_info']
        for unit_id in unit_grp.keys():
            unit_info_grp = unit_grp[unit_id]
            info = {}
            for key in unit_info_grp.attrs.keys():
                info[key] = unit_info_grp.attrs[key]
            for key in unit_info_grp.keys():
                info[key] = unit_info_grp[key][()]
            data['unit_info'][unit_id] = info
        
        # Load extraction params
        data['extraction_params'] = {}
        params_grp = f['extraction_params']
        for key in params_grp.attrs.keys():
            data['extraction_params'][key] = params_grp.attrs[key]
        for key in params_grp.keys():
            data['extraction_params'][key] = params_grp[key][()]
    
    return data


def load_from_npz(filepath):
    """Load neural data from npz format"""
    print(f"Loading data from NPZ: {filepath}")
    
    # Load basic arrays
    data_npz = np.load(filepath, allow_pickle=True)
    
    # Load complex data from pickle
    pickle_path = filepath.with_suffix('.complex.pkl')
    if pickle_path.exists():
        complex_data = load_from_pickle(pickle_path)
    else:
        complex_data = {}
    
    # Reconstruct data structure
    data = {
        'metadata': {},
        'experiment_parameters': {},
        'trial_info': {},
        'extraction_params': {},
        'spike_data': complex_data.get('spike_data', {}),
        'unit_info': complex_data.get('unit_info', {})
    }
    
    # Populate from npz
    for key, value in data_npz.items():
        if key.startswith('metadata_'):
            data['metadata'][key[9:]] = value
        elif key.startswith('exp_'):
            data['experiment_parameters'][key[4:]] = value
        elif key.startswith('trial_'):
            if key == 'trial_orientations':
                data['trial_info']['orientations'] = value.tolist()
            elif key == 'trial_unique_orientations':
                data['trial_info']['unique_orientations'] = value.tolist()
            elif key == 'trial_windows':
                data['trial_info']['trial_windows'] = value.tolist()
        elif key.startswith('params_'):
            data['extraction_params'][key[7:]] = value
    
    # Add trial parameters from complex data
    if 'trial_parameters' in complex_data:
        data['trial_info']['all_trial_parameters'] = complex_data['trial_parameters']
    
    return data


if __name__ == "__main__":
    # Example usage with better error handling
    try:
        rec_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert_860\20251031\CnL39SG_20251031_085159.rec")
        task_file_path = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert_860\20251031\CnL39_drifting_grating_exp_20251031_085247.txt")

        # Check if paths exist
        if not rec_folder.exists():
            print(f"Recording folder does not exist: {rec_folder}")
            exit(1)
        
        if not task_file_path.exists():
            print(f"Task file does not exist: {task_file_path}")
            exit(1)
        
        # Extract and save grating data
        neural_data = extract_grating_neural_data_for_embedding(
            rec_folder=rec_folder,
            task_file_path=task_file_path,
            output_format='pickle'
        )
        
        if neural_data is None:
            print("Failed to extract neural data")
            exit(1)
        
        print("\nGrating data structure:")
        print(f"- {len(neural_data['spike_data'])} units")
        print(f"- {neural_data['metadata']['n_trials']} trials") 
        print(f"- {len(neural_data['trial_info']['unique_orientations'])} orientations tested")
        print(f"- Orientations: {neural_data['trial_info']['unique_orientations']}")
        print(f"- Stimulus duration: {neural_data['experiment_parameters']['stimulus_duration']}s")
        
        # Show example data structure
        if neural_data['spike_data']:
            example_unit = list(neural_data['spike_data'].keys())[0]
            print(f"\nExample unit data structure for {example_unit}:")
            print(f"- Number of trials: {len(neural_data['spike_data'][example_unit])}")
            example_trial = neural_data['spike_data'][example_unit][0]
            print(f"- Example trial data keys: {list(example_trial.keys())}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()