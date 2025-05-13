import os
import numpy as np
import scipy.io
import h5py
from pathlib import Path
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info

def dereference(item, f):
    """Recursively dereference an h5py item."""
    if isinstance(item, h5py.Reference):
        data = f[item][()]
        if isinstance(data, np.ndarray) and data.size == 1:
            return data.item()
        return data
    elif isinstance(item, np.ndarray):
        return np.array([dereference(elem, f) for elem in item])
    else:
        return item

def process_static_grating_responses(rec_folder, stimdata_file, overwrite=True):
    """
    Process static grating responses from an experiment folder.
    
    The function:
      - Loads stimulus and timing information from MAT/HDF5 files,
      - Computes per-unit responses (organizes responses in a 5D array of shape 
        (n_units, n_orientation, n_phase, n_spatialFreq, n_repeats)),
      - Saves the responses along with stimulus parameters, unit information, and additional metadata 
        into an NPZ file in the corresponding session folder.
    
    Parameters:
      experiment_folder (str or Path): Path to the experiment folder.
      overwrite (bool): If False and the NPZ file already exists, skip writing. 
                        If True, overwrite any existing NPZ file.
    
    Returns:
      npz_file (Path): Path to the saved (or existing) NPZ file.
    """
    DIN_file = rec_folder / "DIN.mat"
    peaks_file = rec_folder / "peaks.mat"
    
    # Load peaks data (to get rising edges)
    peaks_data = scipy.io.loadmat(peaks_file, struct_as_record=False, squeeze_me=True)
    rising_edges = peaks_data['locs']
    
    # Open DIN file and extract digital input frequency
    with h5py.File(DIN_file, 'r') as f:
        print("Top-level keys in DIN file:", list(f.keys()))
        freq_params = f["frequency_parameters"]
        data = freq_params['board_dig_in_sample_rate'][:]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        digInFreq = data[0][0]
    
    # Parse session info (animal_id, session_id, folder_name)
    animal_id, session_id, folder_name = parse_session_info(rec_folder)
    ishs = ['0', '1', '2', '3']
    
    # Open the Stimdata file to get stimulus parameters
    with h5py.File(stimdata_file, 'r') as f:
        patternParams_group = f['Stimdata']['patternParams']
        
        # Process orientation, phase, spatialFreq
        orientation_data = patternParams_group['orientation'][()]
        stim_orientation = np.array([dereference(ref, f) for ref in orientation_data]).flatten().astype(float)
        
        phase_data = patternParams_group['phase'][()]
        stim_phase = np.array([dereference(ref, f) for ref in phase_data]).flatten().astype(float)
        
        spatialFreq_data = patternParams_group['spatialFreq'][()]
        stim_spatialFreq = np.array([dereference(ref, f) for ref in spatialFreq_data]).flatten().astype(float)
    
    print("Orientation:", stim_orientation)
    print("Phase:", stim_phase)
    print("Spatial Frequency:", stim_spatialFreq)
    
    # Determine the number of static grating stimuli and extract the corresponding rising edges
    n_static_grating = stim_orientation.shape[0]
    print(f"Number of static grating stimuli: {n_static_grating}")
    print(f"Number of rising edges: {len(rising_edges)}")
    static_grating_rising_edges = rising_edges[-n_static_grating:]
    
    # Unique stimulus parameters
    unique_orientation = np.unique(stim_orientation)
    unique_phase = np.unique(stim_phase)
    unique_spatialFreq = np.unique(stim_spatialFreq)
    
    n_orientation = len(unique_orientation)
    n_phase = len(unique_phase)
    n_spatialFreq = len(unique_spatialFreq)
    
    # Compute the number of repeats/trials per condition
    n_repeats = n_static_grating // (n_orientation * n_phase * n_spatialFreq)
    
    all_units_responses = []
    unit_info = []
    all_unit_qualities = []  # List to store unit qualities from all shanks
    
    # Construct session folder for sorting results (hard-coded base path)
    code_folder = Path(__file__).parent.parent.parent
    session_folder = code_folder / rf"sortout/{animal_id}/{session_id}"
    
    # Check if the output file already exists
    npz_file = session_folder / 'static_grating_responses.npz'
    if npz_file.exists() and not overwrite:
        print(f"File {npz_file} already exists and overwrite=False. Skipping computation and returning existing file.")
        return npz_file
    
    for ish in ishs:
        print(f'Processing {animal_id}/{session_id}/{ish}')
        shank_folder = session_folder / f'{ish}'
        sorting_results_folders = []
        for root, dirs, files in os.walk(shank_folder):
            for dir_name in dirs:
                if dir_name.startswith('sorting_results_'):
                    sorting_results_folders.append(os.path.join(root, dir_name))
        
        for sorting_results_folder in sorting_results_folders:
            phy_folder = Path(sorting_results_folder) / 'phy'
            out_fig_folder = Path(sorting_results_folder) / 'static_grating'
            if not out_fig_folder.exists():
                out_fig_folder.mkdir(parents=True)
            
            # Load sorting analyzer (optionally use curated data)
            sorting_anaylzer = load_sorting_analyzer(Path(sorting_results_folder) / 'sorting_analyzer')
            sorting = sorting_anaylzer.sorting
            sorting = PhySortingExtractor(phy_folder)
            unit_ids = sorting.unit_ids
            # Get the quality for all units in this sorting result
            unit_qualities_this_sort = sorting.get_property('quality')
            fs = sorting.sampling_frequency
            
            for i, unit_id in enumerate(unit_ids):
                spike_train = sorting.get_unit_spike_train(unit_id)
                
                # Initialize array to store responses for each stimulus
                responses = np.zeros((n_static_grating,))
                
                # Calculate responses for each static grating stimulus
                visual_transimission_delay = 0.05  # seconds
                average_time = 0.2  # seconds
                for j, edge in enumerate(static_grating_rising_edges):
                    start_time = edge + visual_transimission_delay * fs
                    end_time = start_time + average_time * fs
                    responses[j] = np.sum((spike_train >= start_time) & (spike_train < end_time)) / average_time
                
                # Build a 4D response array for this unit: (orientation, phase, spatialFreq, repeats)
                response_array = np.zeros((n_orientation, n_phase, n_spatialFreq, n_repeats))
                for i_ori, ori in enumerate(unique_orientation):
                    for i_phase, ph in enumerate(unique_phase):
                        for i_sf, sf in enumerate(unique_spatialFreq):
                            mask = ((stim_orientation == ori) &
                                    (stim_phase == ph) &
                                    (stim_spatialFreq == sf))
                            idxs = np.where(mask)[0]
                            response_array[i_ori, i_phase, i_sf, :] = responses[idxs]
                
                all_units_responses.append(response_array)
                unit_info.append((ish, unit_id))
                # Append unit quality from this shank's sorting result
                all_unit_qualities.append(unit_qualities_this_sort[i])
    
    # Stack all unit responses into a single 5D array: (n_units, n_orientation, n_phase, n_spatialFreq, n_repeats)
    all_units_responses = np.stack(all_units_responses, axis=0)
    print("Final all_units_responses shape:", all_units_responses.shape)
    
    # Save the data to an NPZ file in the session folder, including the combined unit qualities
    print(f"Saving data to {npz_file} (overwrite={overwrite})")
    np.savez(
        npz_file,
        all_units_responses=all_units_responses,
        unit_info=unit_info,
        unit_qualities=all_unit_qualities,  # Combined unit qualities from all shanks
        stim_orientation=stim_orientation,
        stim_phase=stim_phase,
        stim_spatialFreq=stim_spatialFreq,
        unique_orientation=unique_orientation,
        unique_phase=unique_phase,
        unique_spatialFreq=unique_spatialFreq,
        static_grating_rising_edges=static_grating_rising_edges,
        digInFreq=digInFreq,
    )
    print("data saved")
    return npz_file

# Example call:
if __name__ == '__main__':
    rec_folder = Path(input("Please enter the full path to the recording folder: ").strip())
    stimdata_file = Path(input("Please enter the full path to the .mat file: ").strip())

    print(f"Recording folder: {rec_folder}")
    print(f"Stimulus data file: {stimdata_file}")
    # Pass overwrite=True if you want to overwrite an existing file:
    npz_path = process_static_grating_responses(rec_folder, stimdata_file, overwrite=True)