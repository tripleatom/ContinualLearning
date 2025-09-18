import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import os
from collections import defaultdict
import h5py
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rf_recon.rf_grid.task_file_reader import load_task_file


def extract_neural_data_for_embedding(rec_folder, task_file_path, output_format='hdf5'):
    """
    Extract and save neural spike data and trial parameters for embedding analysis.
    
    Parameters:
    -----------
    rec_folder : Path or str
        Path to the recording folder
    task_file_path : Path or str  
        Path to the task file
    output_format : str
        Format to save data ('hdf5', 'pickle', or 'npz')
        
    Returns:
    --------
    dict : Dictionary containing all extracted data
    """
    
    # Setup paths
    rec_folder = Path(rec_folder)
    task_file_path = Path(task_file_path)
    
    animal_id = rec_folder.name.split('.')[0].split('_')[0]
    session_id = rec_folder.name.split('.')[0]
    
    print(f"Extracting data for {animal_id}/{session_id}")
    
    # Load task data
    task_file = load_task_file(task_file_path)
    df = task_file.get_trial_data()
    n_stim_types = len(np.unique(df['StimulusIndex'].values))
    
    # Load DIO data
    task_id = task_file_path.stem
    task_folder = task_file_path.parent
    processed_dio_folder = task_folder / f"{task_id}_DIO.npz"
    dio_data = np.load(processed_dio_folder)
    rising_times = dio_data['rising_times']
    falling_times = dio_data['falling_times']
    
    # Trial windows
    n_trials = len(df)
    trial_windows = [(rising_times[i], falling_times[i]) for i in range(n_trials)]
    
    # Setup session folder structure
    code_folder = Path(__file__).parent.parent.parent.parent
    session_folder = code_folder / f"sortout/{animal_id}/{session_id}"
    
    # Data structure to store everything
    neural_data = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'task_file': str(task_file_path),
            'extraction_date': datetime.now().isoformat(),
            'n_trials': n_trials
        },
        'trial_info': {},
        'spike_data': {},
        'unit_info': {},
        'timing': {
            'rising_times': rising_times,
            'falling_times': falling_times,
            'trial_windows': trial_windows
        }
    }
    
    # Extract trial parameters
    trial_info = df.to_dict('records')
    neural_data['trial_info'] = {
        'stimulus_index': df['StimulusIndex'].values,
        'all_parameters': trial_info,
        'parameter_columns': list(df.columns),
        'unique_stimuli': np.unique(df['StimulusIndex'].values).tolist()
    }
    
    print(f"Found {len(neural_data['trial_info']['unique_stimuli'])} unique stimuli")
    print(f"Trial parameters: {list(df.columns)}")
    
    # Window parameters
    window_pre = 0.2   # seconds before stimulus onset
    window_post = 1.0  # seconds after stimulus onset
    
    # Initialize spike data storage
    unit_counter = 0
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
                # Load sorting data
                sorting_analyzer_path = Path(sorting_results_folder) / 'sorting_analyzer'
                
                if phy_folder.exists():
                    sorting = PhySortingExtractor(phy_folder)
                elif sorting_analyzer_path.exists():
                    sorting_analyzer = load_sorting_analyzer(sorting_analyzer_path)
                    sorting = sorting_analyzer.sorting
                else:
                    print(f"No valid sorting data found in {sorting_results_folder}")
                    continue
                
                fs = sorting.sampling_frequency
                neural_data['metadata']['sampling_frequency'] = fs
                
                print(f"Processing {sorting_results_folder} with {len(sorting.unit_ids)} units")
                
                # Get unit qualities
                unit_ids = sorting.unit_ids
                unit_qualities = sorting.get_property('quality') if hasattr(sorting, 'get_property') else ['good'] * len(unit_ids)
                
                for unit_idx, unit_id in enumerate(unit_ids):
                    quality = unit_qualities[unit_idx] if unit_idx < len(unit_qualities) else 'unknown'
                    
                    # Skip noise units
                    if quality == 'noise':
                        continue
                    
                    print(f"Processing unit {unit_id} (quality: {quality})")
                    
                    # Create unique unit identifier
                    unique_unit_id = f"shank{ish}_sorting{Path(sorting_results_folder).name}_unit{unit_id}"
                    
                    # Get spike train
                    spike_train = sorting.get_unit_spike_train(unit_id)
                    
                    # Store unit metadata
                    neural_data['unit_info'][unique_unit_id] = {
                        'original_unit_id': unit_id,
                        'shank': ish,
                        'quality': quality,
                        'sorting_folder': sorting_results_folder,
                        'n_spikes_total': len(spike_train),
                        'unit_index': unit_counter
                    }
                    
                    # Extract spikes for each trial
                    unit_trial_data = {
                        'spike_times': [],      # List of arrays, one per trial
                        'spike_counts': [],     # Total spikes per trial
                        'trial_indices': [],    # Which trials have data
                        'stimulus_indices': []  # Stimulus index for each trial
                    }
                    
                    for i_trial, (start, end) in enumerate(trial_windows):
                        # Extract spikes in window around stimulus onset
                        trial_spikes = spike_train[
                            (spike_train >= start - window_pre * fs) & 
                            (spike_train < start + window_post * fs)
                        ]
                        
                        if len(trial_spikes) > 0:
                            # Convert to relative time (seconds from stimulus onset)
                            trial_spikes_relative = (trial_spikes - start) / fs
                            unit_trial_data['spike_times'].append(trial_spikes_relative)
                        else:
                            unit_trial_data['spike_times'].append(np.array([]))
                        
                        unit_trial_data['spike_counts'].append(len(trial_spikes))
                        unit_trial_data['trial_indices'].append(i_trial)
                        
                        # Add stimulus index for this trial
                        if i_trial < len(neural_data['trial_info']['stimulus_index']):
                            unit_trial_data['stimulus_indices'].append(
                                neural_data['trial_info']['stimulus_index'][i_trial]
                            )
                        else:
                            unit_trial_data['stimulus_indices'].append(None)
                    
                    # Convert lists to arrays where appropriate
                    unit_trial_data['spike_counts'] = np.array(unit_trial_data['spike_counts'])
                    unit_trial_data['trial_indices'] = np.array(unit_trial_data['trial_indices'])
                    unit_trial_data['stimulus_indices'] = np.array(unit_trial_data['stimulus_indices'])
                    
                    neural_data['spike_data'][unique_unit_id] = unit_trial_data
                    unit_counter += 1
                    
            except Exception as e:
                print(f"Error processing {sorting_results_folder}: {e}")
                continue
    
    # Add analysis parameters
    neural_data['analysis_params'] = {
        'window_pre': window_pre,
        'window_post': window_post,
        'total_units': unit_counter,
        'shanks_processed': ishs
    }
    
    print(f"\nExtraction complete!")
    print(f"Total units processed: {unit_counter}")
    print(f"Total trials: {n_trials}")
    
    # Save data
    output_dir = session_folder / 'embedding_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"{session_id}_neural_data_RFGRid_{n_stim_types}"
    
    if output_format == 'hdf5':
        save_to_hdf5(neural_data, output_dir / f"{base_filename}.h5")
    elif output_format == 'pickle':
        save_to_pickle(neural_data, output_dir / f"{base_filename}.pkl")
    elif output_format == 'npz':
        save_to_npz(neural_data, output_dir / f"{base_filename}.npz")
    else:
        raise ValueError("output_format must be 'hdf5', 'pickle', or 'npz'")
    
    return neural_data


def save_to_hdf5(data, filepath):
    """Save neural data to HDF5 format (recommended for large datasets)"""
    print(f"Saving to HDF5: {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata
        metadata_grp = f.create_group('metadata')
        for key, value in data['metadata'].items():
            if isinstance(value, str):
                metadata_grp.attrs[key] = value
            else:
                metadata_grp.create_dataset(key, data=value)
        
        # Save trial info
        trial_grp = f.create_group('trial_info')
        trial_grp.create_dataset('stimulus_index', data=data['trial_info']['stimulus_index'])
        trial_grp.create_dataset('unique_stimuli', data=data['trial_info']['unique_stimuli'])
        trial_grp.attrs['parameter_columns'] = json.dumps(data['trial_info']['parameter_columns'])
        
        # Save trial parameters as a structured array or individual datasets
        params_grp = trial_grp.create_group('parameters')
        for param in data['trial_info']['parameter_columns']:
            param_data = [trial[param] for trial in data['trial_info']['all_parameters']]
            params_grp.create_dataset(param, data=param_data)
        
        # Save timing data
        timing_grp = f.create_group('timing')
        timing_grp.create_dataset('rising_times', data=data['timing']['rising_times'])
        timing_grp.create_dataset('falling_times', data=data['timing']['falling_times'])
        
        # Save spike data
        spike_grp = f.create_group('spike_data')
        for unit_id, unit_data in data['spike_data'].items():
            unit_grp = spike_grp.create_group(unit_id)
            unit_grp.create_dataset('spike_counts', data=unit_data['spike_counts'])
            unit_grp.create_dataset('trial_indices', data=unit_data['trial_indices'])
            unit_grp.create_dataset('stimulus_indices', data=unit_data['stimulus_indices'])
            
            # Save spike times as variable-length arrays
            spike_times_grp = unit_grp.create_group('spike_times')
            for i, spikes in enumerate(unit_data['spike_times']):
                spike_times_grp.create_dataset(f'trial_{i:04d}', data=spikes)
        
        # Save unit info
        unit_grp = f.create_group('unit_info')
        for unit_id, info in data['unit_info'].items():
            unit_info_grp = unit_grp.create_group(unit_id)
            for key, value in info.items():
                if isinstance(value, str):
                    unit_info_grp.attrs[key] = value
                else:
                    unit_info_grp.create_dataset(key, data=value)
        
        # Save analysis params
        analysis_grp = f.create_group('analysis_params')
        for key, value in data['analysis_params'].items():
            if isinstance(value, (list, np.ndarray)):
                analysis_grp.create_dataset(key, data=value)
            else:
                analysis_grp.attrs[key] = value


def save_to_pickle(data, filepath):
    """Save neural data to pickle format"""
    print(f"Saving to pickle: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_to_npz(data, filepath):
    """Save neural data to compressed numpy format"""
    print(f"Saving to NPZ: {filepath}")
    
    # Flatten the nested structure for npz
    save_dict = {}
    
    # Metadata
    for key, value in data['metadata'].items():
        save_dict[f'metadata_{key}'] = value
    
    # Trial info
    save_dict['trial_stimulus_index'] = data['trial_info']['stimulus_index']
    save_dict['trial_unique_stimuli'] = data['trial_info']['unique_stimuli']
    save_dict['trial_parameter_columns'] = data['trial_info']['parameter_columns']
    
    # Timing
    save_dict['timing_rising'] = data['timing']['rising_times']
    save_dict['timing_falling'] = data['timing']['falling_times']
    
    # Analysis params
    for key, value in data['analysis_params'].items():
        save_dict[f'analysis_{key}'] = value
    
    # Save spike data and unit info separately due to nested structure
    np.savez_compressed(filepath, **save_dict)
    
    # Save the complex nested data as pickle alongside
    complex_data = {
        'spike_data': data['spike_data'],
        'unit_info': data['unit_info'],
        'trial_parameters': data['trial_info']['all_parameters']
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


def load_from_hdf5(filepath):
    """Load neural data from HDF5 format"""
    print(f"Loading from HDF5: {filepath}")
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        data['metadata'] = dict(f['metadata'].attrs)
        for key in f['metadata'].keys():
            data['metadata'][key] = f['metadata'][key][()]
        
        # Load trial info
        data['trial_info'] = {
            'stimulus_index': f['trial_info/stimulus_index'][()],
            'unique_stimuli': f['trial_info/unique_stimuli'][()].tolist(),
            'parameter_columns': json.loads(f['trial_info'].attrs['parameter_columns'])
        }
        
        # Load trial parameters
        params = {}
        for param in data['trial_info']['parameter_columns']:
            params[param] = f[f'trial_info/parameters/{param}'][()].tolist()
        data['trial_info']['all_parameters'] = [
            {param: params[param][i] for param in data['trial_info']['parameter_columns']}
            for i in range(len(params[data['trial_info']['parameter_columns'][0]]))
        ]
        
        # Load timing
        data['timing'] = {
            'rising_times': f['timing/rising_times'][()],
            'falling_times': f['timing/falling_times'][()]
        }
        
        # Load spike data
        data['spike_data'] = {}
        for unit_id in f['spike_data'].keys():
            unit_data = {
                'spike_counts': f[f'spike_data/{unit_id}/spike_counts'][()],
                'trial_indices': f[f'spike_data/{unit_id}/trial_indices'][()],
                'stimulus_indices': f[f'spike_data/{unit_id}/stimulus_indices'][()],
                'spike_times': []
            }
            
            # Load spike times
            spike_times_grp = f[f'spike_data/{unit_id}/spike_times']
            for i in range(len(unit_data['trial_indices'])):
                trial_key = f'trial_{i:04d}'
                if trial_key in spike_times_grp:
                    unit_data['spike_times'].append(spike_times_grp[trial_key][()])
                else:
                    unit_data['spike_times'].append(np.array([]))
            
            data['spike_data'][unit_id] = unit_data
        
        # Load unit info
        data['unit_info'] = {}
        for unit_id in f['unit_info'].keys():
            unit_info = dict(f[f'unit_info/{unit_id}'].attrs)
            for key in f[f'unit_info/{unit_id}'].keys():
                unit_info[key] = f[f'unit_info/{unit_id}/{key}'][()]
            data['unit_info'][unit_id] = unit_info
        
        # Load analysis params
        data['analysis_params'] = dict(f['analysis_params'].attrs)
        for key in f['analysis_params'].keys():
            data['analysis_params'][key] = f['analysis_params'][key][()]
    
    return data


def load_from_pickle(filepath):
    """Load neural data from pickle format"""
    print(f"Loading from pickle: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_from_npz(filepath):
    """Load neural data from npz format"""
    print(f"Loading from NPZ: {filepath}")
    
    # Load the main npz file
    npz_data = np.load(filepath, allow_pickle=True)
    
    # Load the complex data from pickle
    complex_path = filepath.with_suffix('.complex.pkl')
    if complex_path.exists():
        complex_data = load_from_pickle(complex_path)
    else:
        complex_data = {'spike_data': {}, 'unit_info': {}, 'trial_parameters': []}
    
    # Reconstruct the full data structure
    data = {
        'metadata': {},
        'trial_info': {},
        'timing': {},
        'analysis_params': {},
        'spike_data': complex_data['spike_data'],
        'unit_info': complex_data['unit_info']
    }
    
    # Parse the flattened keys
    for key, value in npz_data.items():
        if key.startswith('metadata_'):
            data['metadata'][key[9:]] = value
        elif key.startswith('trial_'):
            data['trial_info'][key[6:]] = value
        elif key.startswith('timing_'):
            data['timing'][key[7:]] = value
        elif key.startswith('analysis_'):
            data['analysis_params'][key[9:]] = value
    
    # Add trial parameters
    data['trial_info']['all_parameters'] = complex_data['trial_parameters']
    
    return data


if __name__ == "__main__":
    # Example usage
    rec_folder = Path(r"/Volumes/xieluanlabs/xl_cl/RF_GRID/250821/CnL38SG/CnL38SG_20250821_141733.rec")
    task_file_path = Path(r"/Volumes/xieluanlabs/xl_cl/RF_GRID/250821/CnL38_20250821_1.txt")

    # Extract and save data
    neural_data = extract_neural_data_for_embedding(
        rec_folder=rec_folder,
        task_file_path=task_file_path,
        output_format='pickle'  # or 'pickle' or 'npz'
    )
    
    # Example of loading data back
    # data_file = Path("path/to/your/saved/file.h5")
    # loaded_data = load_neural_data(data_file)
    
    print("\nData structure:")
    print(f"- {len(neural_data['spike_data'])} units")
    print(f"- {neural_data['metadata']['n_trials']} trials") 
    print(f"- {len(neural_data['trial_info']['unique_stimuli'])} unique stimuli")
    print(f"- Trial parameters: {neural_data['trial_info']['parameter_columns']}")