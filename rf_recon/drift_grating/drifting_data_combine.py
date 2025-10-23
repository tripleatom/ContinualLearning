import numpy as np
from pathlib import Path
from collections import defaultdict

def combine_drifting_grating_files(npz_file1, npz_file2, output_file=None, overwrite=True):
    """
    Combine two drifting grating response NPZ files.
    
    This function merges trials from two separate recordings, combining data
    for the same units across both files and adding new units as needed.
    
    Parameters:
        npz_file1 (str or Path): Path to the first NPZ file
        npz_file2 (str or Path): Path to the second NPZ file
        output_file (str or Path, optional): Path for the combined output file.
                                            If None, creates a file in the same
                                            directory as npz_file1.
        overwrite (bool): Whether to overwrite existing output file
    
    Returns:
        Path: Path to the combined NPZ file
    """
    # Load both files
    print(f"Loading {npz_file1}...")
    data1 = np.load(npz_file1, allow_pickle=True)
    print(f"Loading {npz_file2}...")
    data2 = np.load(npz_file2, allow_pickle=True)
    
    # Extract units data
    units_data1 = data1['units_data']
    units_data2 = data2['units_data']
    
    # Create a dictionary to organize units by (shank, unit_id)
    units_dict = defaultdict(lambda: {'trials': [], 'metadata': None})
    
    # Add units from first file
    print(f"Processing {len(units_data1)} units from file 1...")
    for unit in units_data1:
        key = (unit['shank'], unit['unit_id'])
        units_dict[key]['trials'].extend(unit['trials'])
        units_dict[key]['metadata'] = {
            'unit_id': unit['unit_id'],
            'shank': unit['shank'],
            'sampling_rate': unit['sampling_rate']
        }
    
    # Add units from second file
    print(f"Processing {len(units_data2)} units from file 2...")
    for unit in units_data2:
        key = (unit['shank'], unit['unit_id'])
        units_dict[key]['trials'].extend(unit['trials'])
        if units_dict[key]['metadata'] is None:
            units_dict[key]['metadata'] = {
                'unit_id': unit['unit_id'],
                'shank': unit['shank'],
                'sampling_rate': unit['sampling_rate']
            }
    
    # Rebuild combined units data
    combined_units_data = []
    combined_unit_info = []
    combined_unit_qualities = []
    
    for (shank, unit_id), unit_dict in sorted(units_dict.items()):
        combined_unit = {
            'unit_id': unit_dict['metadata']['unit_id'],
            'shank': unit_dict['metadata']['shank'],
            'sampling_rate': unit_dict['metadata']['sampling_rate'],
            'trials': sorted(unit_dict['trials'], key=lambda x: x['trial_number'])
        }
        combined_units_data.append(combined_unit)
        combined_unit_info.append((shank, unit_id))
    
    # Combine stimulus arrays
    stim_orientation = np.concatenate([data1['stim_orientation'], data2['stim_orientation']])
    stim_phase = np.concatenate([data1['stim_phase'], data2['stim_phase']])
    stim_spatialFreq = np.concatenate([data1['stim_spatialFreq'], data2['stim_spatialFreq']])
    
    # Combine rising edges
    rising_edges = np.concatenate([data1['rising_edges'], data2['rising_edges']])
    
    # Get unique values (these should be the same across files, but just in case)
    unique_orientation = np.unique(stim_orientation)
    unique_phase = np.unique(stim_phase)
    unique_spatialFreq = np.unique(stim_spatialFreq)
    
    # Update summary statistics
    n_orientation = len(unique_orientation)
    n_phase = len(unique_phase)
    n_spatialFreq = len(unique_spatialFreq)
    total_trials = len(rising_edges)
    n_repeats = total_trials // (n_orientation * n_phase * n_spatialFreq)
    
    summary_stats = {
        'n_units': len(combined_units_data),
        'n_trials': total_trials,
        'n_orientation': n_orientation,
        'n_phase': n_phase,
        'n_spatialFreq': n_spatialFreq,
        'n_repeats': n_repeats,
        'pre_stim_window': float(data1['pre_stim_window']),
        'post_stim_window': float(data1['post_stim_window']),
    }
    
    # Determine output file path
    if output_file is None:
        file1_path = Path(npz_file1)
        output_file = file1_path.parent / f'combined_{file1_path.name}'
    else:
        output_file = Path(output_file)
    
    # Check if output exists
    if output_file.exists() and not overwrite:
        print(f"File {output_file} already exists and overwrite=False. Skipping.")
        return output_file
    
    # Handle unit qualities - try to combine if available
    try:
        combined_unit_qualities = np.concatenate([
            data1['unit_qualities'],
            data2['unit_qualities']
        ])
    except (KeyError, ValueError):
        print("Warning: Could not combine unit_qualities arrays. Skipping this field.")
        combined_unit_qualities = np.array([])
    
    # Save combined data
    print(f"Saving combined data to {output_file}...")
    np.savez(
        output_file,
        # Original data structure
        unit_info=combined_unit_info,
        unit_qualities=combined_unit_qualities,
        stim_orientation=stim_orientation,
        stim_phase=stim_phase,
        stim_spatialFreq=stim_spatialFreq,
        unique_orientation=unique_orientation,
        unique_phase=unique_phase,
        unique_spatialFreq=unique_spatialFreq,
        rising_edges=rising_edges,
        
        # New structured data
        units_data=combined_units_data,
        summary_stats=summary_stats,
        
        # Timing parameters
        pre_stim_window=float(data1['pre_stim_window']),
        post_stim_window=float(data1['post_stim_window']),
    )
    
    print(f"\nCombination complete!")
    print(f"  Total units: {len(combined_units_data)}")
    print(f"  Total trials: {total_trials}")
    print(f"  File 1 had {len(data1['rising_edges'])} trials")
    print(f"  File 2 had {len(data2['rising_edges'])} trials")
    print(f"  Output saved to: {output_file}")
    
    return output_file


# Example usage
if __name__ == '__main__':
    npz_file1 = Path(input("Enter path to first NPZ file: ").strip().strip('"'))
    npz_file2 = Path(input("Enter path to second NPZ file: ").strip().strip('"'))
    
    # Optional: specify output file
    use_custom_output = input("Specify custom output path? (y/n, default=n): ").strip().lower()
    if use_custom_output == 'y':
        output_file = Path(input("Enter output file path: ").strip().strip('"'))
    else:
        output_file = None
    
    # Combine the files
    combined_file = combine_drifting_grating_files(
        npz_file1, 
        npz_file2, 
        output_file=output_file,
        overwrite=True
    )
    
    print(f"\nCombined file created: {combined_file}")