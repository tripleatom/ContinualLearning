import pickle
from pathlib import Path
from datetime import datetime

def combine_drifting_grating_files(pkl_file1, pkl_file2, output_file=None, overwrite=True):
    """
    Combine two drifting grating response PKL files using the unified schema.

    This function merges trials from two separate recordings, combining data
    for the same units across both files and adding new units as needed.

    Parameters:
        pkl_file1 (str or Path): Path to the first PKL file
        pkl_file2 (str or Path): Path to the second PKL file
        output_file (str or Path, optional): Path for the combined output file.
                                            If None, creates a file in the same
                                            directory as pkl_file1.
        overwrite (bool): Whether to overwrite existing output file

    Returns:
        Path: Path to the combined PKL file
    """
    # Load both files
    print(f"Loading {pkl_file1}...")
    with open(pkl_file1, 'rb') as f:
        data1 = pickle.load(f)

    print(f"Loading {pkl_file2}...")
    with open(pkl_file2, 'rb') as f:
        data2 = pickle.load(f)

    # Verify both files are drifting_grating experiments
    if data1['metadata']['experiment_type'] != 'drifting_grating' or \
       data2['metadata']['experiment_type'] != 'drifting_grating':
        raise ValueError("Both files must be drifting_grating experiment types")

    # Extract spike data and unit info
    spike_data1 = data1['spike_data']
    spike_data2 = data2['spike_data']
    unit_info1 = data1['unit_info']
    unit_info2 = data2['unit_info']

    # Create dictionaries to organize units by unique_unit_id
    combined_spike_data = {}
    combined_unit_info = {}

    # Track trial offset for second file
    n_trials_file1 = data1['metadata']['n_trials']

    # Add units from first file
    print(f"Processing {len(spike_data1)} units from file 1...")
    for unit_id, trials in spike_data1.items():
        combined_spike_data[unit_id] = trials.copy()
        combined_unit_info[unit_id] = unit_info1[unit_id].copy()

    # Add units from second file
    print(f"Processing {len(spike_data2)} units from file 2...")
    for unit_id, trials in spike_data2.items():
        # Adjust trial indices for the second file
        adjusted_trials = []
        for trial in trials:
            adjusted_trial = trial.copy()
            adjusted_trial['trial_index'] = trial['trial_index'] + n_trials_file1
            adjusted_trials.append(adjusted_trial)

        # If unit exists in first file, merge trials
        if unit_id in combined_spike_data:
            combined_spike_data[unit_id].extend(adjusted_trials)
        else:
            # New unit from second file
            combined_spike_data[unit_id] = adjusted_trials
            combined_unit_info[unit_id] = unit_info2[unit_id].copy()

    # Update unit indices and spike counts
    for idx, unit_id in enumerate(sorted(combined_unit_info.keys())):
        combined_unit_info[unit_id]['unit_index'] = idx
        combined_unit_info[unit_id]['n_spikes_total'] = sum(
            trial['spike_count'] for trial in combined_spike_data[unit_id]
        )

    # Combine trial information
    combined_orientations = data1['trial_info']['orientations'] + data2['trial_info']['orientations']
    combined_phases = data1['trial_info']['phases'] + data2['trial_info']['phases']
    combined_spatial_freqs = data1['trial_info']['spatial_frequencies'] + data2['trial_info']['spatial_frequencies']
    combined_temporal_freqs = data1['trial_info']['temporal_frequencies'] + data2['trial_info']['temporal_frequencies']
    combined_trial_windows = data1['trial_info']['trial_windows'] + data2['trial_info']['trial_windows']

    # Combine all_trial_parameters with adjusted indices
    combined_trial_params = data1['trial_info']['all_trial_parameters'].copy()
    for trial_param in data2['trial_info']['all_trial_parameters']:
        adjusted_param = trial_param.copy()
        adjusted_param['trial_index'] = trial_param['trial_index'] + n_trials_file1
        combined_trial_params.append(adjusted_param)

    # Get unique values
    unique_orientation = sorted(set(combined_orientations))
    unique_phase = sorted(set(combined_phases))
    unique_spatial_freq = sorted(set(combined_spatial_freqs))
    unique_temporal_freq = sorted(set(combined_temporal_freqs))

    # Calculate combined statistics
    total_trials = len(combined_orientations)
    n_repeats = total_trials // (len(unique_orientation) * len(unique_phase) *
                                  len(unique_spatial_freq) * len(unique_temporal_freq))

    # Determine output file path
    if output_file is None:
        file1_path = Path(pkl_file1)
        output_file = file1_path.parent / f'combined_{file1_path.name}'
    else:
        output_file = Path(output_file)

    # Check if output exists
    if output_file.exists() and not overwrite:
        print(f"File {output_file} already exists and overwrite=False. Skipping.")
        return output_file

    # Build combined neural_data structure
    combined_data = {
        'metadata': {
            'animal_id': data1['metadata']['animal_id'],
            'session_id': f"{data1['metadata']['session_id']}_combined",
            'recording_folder': f"{data1['metadata']['recording_folder']} + {data2['metadata']['recording_folder']}",
            'task_file': f"{data1['metadata']['task_file']} + {data2['metadata']['task_file']}",
            'extraction_date': datetime.now().isoformat(),
            'n_trials': total_trials,
            'experiment_type': 'drifting_grating',
            'sampling_frequency': data1['metadata']['sampling_frequency'],
            'combined_from': [str(pkl_file1), str(pkl_file2)],
        },
        'experiment_parameters': {
            'stimulus_duration': data1['experiment_parameters']['stimulus_duration'],
            'iti_duration': None,
            'trial_duration': data1['experiment_parameters']['trial_duration'],
            'total_trials': total_trials,
            'n_orientations': len(unique_orientation),
            'n_phases': len(unique_phase),
            'n_spatial_frequencies': len(unique_spatial_freq),
            'n_temporal_frequencies': len(unique_temporal_freq),
            'n_repeats': n_repeats,
        },
        'trial_info': {
            'orientations': combined_orientations,
            'unique_orientations': unique_orientation,
            'phases': combined_phases,
            'unique_phases': unique_phase,
            'spatial_frequencies': combined_spatial_freqs,
            'unique_spatial_frequencies': unique_spatial_freq,
            'temporal_frequencies': combined_temporal_freqs,
            'unique_temporal_frequencies': unique_temporal_freq,
            'trial_windows': combined_trial_windows,
            'all_trial_parameters': combined_trial_params,
        },
        'spike_data': combined_spike_data,
        'unit_info': combined_unit_info,
        'extraction_params': {
            'window_pre': data1['extraction_params']['window_pre'],
            'window_post': data1['extraction_params']['window_post'],
            'total_units': len(combined_unit_info),
        }
    }

    # Save combined data
    print(f"Saving combined data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nCombination complete!")
    print(f"  Total units: {len(combined_unit_info)}")
    print(f"  Total trials: {total_trials}")
    print(f"  File 1 had {data1['metadata']['n_trials']} trials")
    print(f"  File 2 had {data2['metadata']['n_trials']} trials")
    print(f"  Output saved to: {output_file}")

    return output_file


# Example usage
if __name__ == '__main__':
    pkl_file1 = Path(input("Enter path to first PKL file: ").strip().strip('"'))
    pkl_file2 = Path(input("Enter path to second PKL file: ").strip().strip('"'))

    # Optional: specify output file
    use_custom_output = input("Specify custom output path? (y/n, default=n): ").strip().lower()
    if use_custom_output == 'y':
        output_file = Path(input("Enter output file path: ").strip().strip('"'))
    else:
        output_file = None

    # Combine the files
    combined_file = combine_drifting_grating_files(
        pkl_file1,
        pkl_file2,
        output_file=output_file,
        overwrite=True
    )

    print(f"\nCombined file created: {combined_file}")

    # Load and verify the combined structure
    with open(combined_file, 'rb') as f:
        combined_data = pickle.load(f)

    print("\nCombined data structure verification:")
    print(f"- Animal ID: {combined_data['metadata']['animal_id']}")
    print(f"- Session ID: {combined_data['metadata']['session_id']}")
    print(f"- Experiment type: {combined_data['metadata']['experiment_type']}")
    print(f"- Total units: {len(combined_data['spike_data'])}")
    print(f"- Total trials: {combined_data['metadata']['n_trials']}")
    print(f"- Unique orientations: {combined_data['trial_info']['unique_orientations']}")