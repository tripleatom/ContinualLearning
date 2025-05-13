import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py
from rf_recon.rf_func import dereference
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
import scipy
import os

rec_folder = Path(input("Please enter the full path to the recording folder: ").strip().strip('"').strip("'"))
stimdata_file = Path(input("Please enter the full path to the .mat file: ").strip().strip('"').strip("'"))

print(f"Recording folder: {rec_folder}")
print(f"Stimulus data file: {stimdata_file}")

DIN_file = rec_folder / "DIN.mat"
peaks_file = rec_folder / "peaks.mat"

# Load the peaks data
peaks_data = scipy.io.loadmat(peaks_file, struct_as_record=False, squeeze_me=True)
rising_edges = peaks_data['locs']

# Open the HDF5-based MAT file to load digital input frequency
with h5py.File(DIN_file, 'r') as f:
    print("Top-level keys in DIN file:", list(f.keys()))
    freq_params = f["frequency_parameters"]
    data = freq_params['board_dig_in_sample_rate'][:]
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    digInFreq = data[0][0]

# Parse session information
animal_id, session_id, folder_name = parse_session_info(rec_folder)
ishs = ['0', '1', '2', '3']  # list of shanks

# Load stimulus parameters from MAT file
with h5py.File(stimdata_file, 'r') as f:
    patternParams_group = f['Stimdata']['patternParams']
    # Orientation
    orientation_data = patternParams_group['orientation'][()]
    stim_orientation = np.array([dereference(ref, f) for ref in orientation_data]).flatten().astype(float)

print("Orientation:", stim_orientation)

# Calculate the number of static grating stimuli and extract corresponding rising edges
n_static_grating = np.shape(stim_orientation)[0]
print(f"Number of static grating stimuli: {n_static_grating}")
print(f"Number of rising edges: {len(rising_edges)}")
static_grating_rising_edges = rising_edges[-n_static_grating:]

# Prepare a container to hold info from all shanks.
all_shank_info = {}

# Process each shank
for ish in ishs:
    print(f'\nProcessing shank {ish} for {animal_id}/{session_id}')
    # Define the shank folder (adjust path as needed)
    code_folder = Path(__file__).parent.parent.parent
    shank_folder = code_folder / rf'sortout/{animal_id}/{session_id}/{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))
    
    # Dictionary to hold unit info for the current shank
    unit_info_dict = {}
    
    # Loop over each sorting result folder for this shank
    for sorting_results_folder in sorting_results_folders:
        print(f"Processing sorting results folder: {sorting_results_folder}")
        phy_folder = Path(sorting_results_folder) / 'phy'
        out_fig_folder = Path(sorting_results_folder) / 'static_grating'
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)
        
        # Load the sorting extractor from the phy folder
        sorting = PhySortingExtractor(phy_folder)
        unit_ids = sorting.unit_ids
        fs = sorting.sampling_frequency
        qualities = sorting.get_property('quality')
        
        # Loop over each unit
        for idx, unit_id in enumerate(unit_ids):
            quality = qualities[idx]
            print(f"Processing unit {unit_id}: {quality}")
           
            spike_train = sorting.get_unit_spike_train(unit_id)
            
            # Initialize responses array for each static grating stimulus
            responses = np.zeros((n_static_grating,))
            blank_average_time = 10.0  # seconds for blank response calculation
            blank_start_time = static_grating_rising_edges[0] - blank_average_time * fs
            blank_end_time = static_grating_rising_edges[0]
            blank_spikes = np.sum((spike_train >= blank_start_time) & (spike_train < blank_end_time))
            blank_mean = blank_spikes / blank_average_time
            
            # Compute responses for each stimulus
            visual_transimission_delay = 0.05  # 50ms delay
            average_time = 0.2  # 200ms window for average firing rate
            for j, edge in enumerate(static_grating_rising_edges):
                start_time = edge + visual_transimission_delay * fs
                end_time = start_time + average_time * fs
                responses[j] = np.sum((spike_train >= start_time) & (spike_train < end_time)) / average_time
            
            # Determine unique stimulus parameters and number of conditions
            unique_orientation = np.unique(stim_orientation)
            n_orientation = len(unique_orientation)
            
            # Calculate the number of repeats per condition
            n_repeats = n_static_grating // n_orientation
            
            # Create a 2D response array: orientation x repeats
            response_array = np.zeros((n_orientation, n_repeats))
            for i_ori, ori in enumerate(unique_orientation):
                mask = (stim_orientation == ori)
                idxs = np.where(mask)[0]
                response_array[i_ori, :] = responses[idxs]
            
            # Calculate mean response and standard error for each orientation
            mean_response = np.mean(response_array, axis=1)
            sem_response = np.std(response_array, axis=1) / np.sqrt(n_repeats)  # Standard Error of the Mean
            
            # Calculate gOSI (global OSI)
            R_theta = mean_response
            theta_radians = np.deg2rad(unique_orientation)
            numerator = np.sum(R_theta * np.exp(1j * 2 * theta_radians))
            denominator = np.sum(R_theta)
            gOSI = np.abs(numerator / denominator)

            # Calculate preferred orientation index and degree
            pref_ori_index = np.argmax(mean_response)
            pref_ori_deg = unique_orientation[pref_ori_index]
            
            # Plot the tuning curve for each unit
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(unique_orientation, mean_response, yerr=sem_response,
                       marker='o', linestyle='-', color='blue',
                       capsize=5, capthick=1, elinewidth=1)
            ax.set_xlabel('Orientation (degrees)', fontsize=12)
            ax.set_ylabel('Mean Response (Hz)', fontsize=12)
            ax.set_title(f"Unit {unit_id}: {quality}", fontsize=14)
            ax.tick_params(labelsize=10)
            ax.grid(True)

            # Place annotation inside the plot area
            text_str = f"gOSI: {gOSI:.2f}\nPreferred Angle: {pref_ori_deg:.1f}°"
            ax.text(
                0.02, 0.98, text_str,
                transform=ax.transAxes,
                fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

            # Tighten layout with custom margins
            plt.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10)

            out_file = out_fig_folder / f'unit_{unit_id}_tuning_curve.png'
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved tuning curve for unit {unit_id} to {out_file}")
            
            # Save unit indices and metrics in the current shank's info dictionary
            unit_info_dict[str(unit_id)] = {
                'mean_response': mean_response.tolist(),
                'unique_orientation': unique_orientation.tolist(),
                'gOSI': float(gOSI),
                'pref_ori_index': int(pref_ori_index),
                'pref_ori_deg': float(pref_ori_deg)
            }
    
    # After processing all sorting result folders for this shank, store its unit info
    all_shank_info[ish] = unit_info_dict

# Save all shank info into one NPZ file in the parent folder of the experiment folder.
npz_out_path = code_folder / "sortout" / animal_id / session_id / "single_static_grating_tuning_metrics.npz"
npz_out_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
np.savez_compressed(npz_out_path, all_shank_info=all_shank_info)
print(f"\nSaved all shank info to {npz_out_path}")