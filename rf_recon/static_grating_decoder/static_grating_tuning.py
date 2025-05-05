import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os
import h5py
from rf_recon.rf_func import find_stim_index, h5py_to_dict, hex_offsets
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.colors as mcolors

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

# Define experiment folder (adjust to your environment)
experiment_folder = r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/250412/CnL34"  # for mac
experiment_folder = Path(experiment_folder)
rec_folder = next((p for p in experiment_folder.iterdir() if p.is_dir()), None)
print("Recording folder:", rec_folder)
Stimdata_file = next(experiment_folder.glob("static_grating*.mat"), None)
print("Stimdata file:", Stimdata_file)
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
with h5py.File(Stimdata_file, 'r') as f:
    patternParams_group = f['Stimdata']['patternParams']
    # Orientation
    orientation_data = patternParams_group['orientation'][()]
    stim_orientation = np.array([dereference(ref, f) for ref in orientation_data]).flatten().astype(float)
    # Phase
    phase_data = patternParams_group['phase'][()]
    stim_phase = np.array([dereference(ref, f) for ref in phase_data]).flatten().astype(float)
    # Spatial frequency
    spatialFreq_data = patternParams_group['spatialFreq'][()]
    stim_spatialFreq = np.array([dereference(ref, f) for ref in spatialFreq_data]).flatten().astype(float)

print("Orientation:", stim_orientation)
print("Phase:", stim_phase)
print("Spatial Frequency:", stim_spatialFreq)

# Calculate the number of static grating stimuli and extract corresponding rising edges
n_static_grating = np.shape(stim_orientation)[0]
print(f"Number of static grating stimuli: {n_static_grating}")
print(f"Number of rising edges: {len(rising_edges)}")
static_grating_rising_edges = rising_edges[-n_static_grating:]

# Define custom colormap
pink_reds = mcolors.LinearSegmentedColormap.from_list(
    'PinkReds',
    [(1, 0.9, 0.9), (1, 0.6, 0.6), (0.8, 0, 0)]
)

phase_offset_r = 0.2
phase_offset_theta = np.pi / 60
phase_quad_offsets = [
    (phase_offset_r,  phase_offset_theta),   # top-right
    (-phase_offset_r, phase_offset_theta),   # top-left
    (-phase_offset_r, -phase_offset_theta),  # bottom-left
    (phase_offset_r,  -phase_offset_theta)    # bottom-right
]

# Prepare a container to hold info from all shanks.
all_shank_info = {}

# Process each shank
for ish in ishs:
    print(f'\nProcessing shank {ish} for {animal_id}/{session_id}')
    # Define the shank folder (adjust path as needed)
    shank_folder = rf'/Volumes/xieluanlabs/xl_cl/code/sortout/{animal_id}/{session_id}/{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))
    
    # Dictionary to hold unit info for the current shank
    unit_info_dict = {}
    
    # Loop over each sorting result folder for this shank
    for sorting_results_folder in sorting_results_folders:
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
            # if quality == 'noise':
            #     continue
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
            unique_phase = np.unique(stim_phase)
            unique_spatialFreq = np.unique(stim_spatialFreq)
            n_orientation = len(unique_orientation)
            n_phase = len(unique_phase)
            n_spatialFreq = len(unique_spatialFreq)
            
            # Calculate number of repeats per condition
            n_repeats = n_static_grating // (n_orientation * n_phase * n_spatialFreq)
            
            # Create a 4D response array: orientation x phase x spatialFreq x repeats
            response_array = np.zeros((n_orientation, n_phase, n_spatialFreq, n_repeats))
            for i_ori, ori in enumerate(unique_orientation):
                for i_ph, ph in enumerate(unique_phase):
                    for i_sf, sf in enumerate(unique_spatialFreq):
                        mask = ((stim_orientation == ori) &
                                (stim_phase == ph) &
                                (stim_spatialFreq == sf))
                        idxs = np.where(mask)[0]
                        response_array[i_ori, i_ph, i_sf, :] = responses[idxs]
            
            # 1. OSI (Orientation Selectivity Index)
            mean_over_repeats = np.mean(response_array, axis=3)
            i_ori_sel, i_phase_sel, i_sf_sel = np.unravel_index(
                np.argmax(mean_over_repeats), mean_over_repeats.shape
            )
            R_pref = mean_over_repeats[i_ori_sel, i_phase_sel, i_sf_sel]
            i_orth = (i_ori_sel + (n_orientation // 2)) % n_orientation
            R_orth = mean_over_repeats[i_orth, i_phase_sel, i_sf_sel]
            OSI = (R_pref - R_orth) / (R_pref + R_orth)
            pref_ori_deg = unique_orientation[i_ori_sel]
            
            # 2. gOSI (global OSI)
            sf_means = np.mean(response_array, axis=(0, 1, 3))
            i_best_sf = np.argmax(sf_means)
            best_sf_value = unique_spatialFreq[i_best_sf]
            R_theta = np.mean(response_array[:, :, i_best_sf, :], axis=(1,2))
            theta_radians = np.deg2rad(unique_orientation)
            numerator = np.sum(R_theta * np.exp(1j * 2 * theta_radians))
            denominator = np.sum(R_theta)
            gOSI = np.abs(numerator / denominator)
            
            # 3. SFDI (Spatial Frequency Discrimination Index)
            mean_by_ori = np.mean(response_array, axis=(1,2,3))
            i_ori_pref = np.argmax(mean_by_ori)
            mean_ori_sf = np.mean(response_array[i_ori_pref, :, :, :], axis=(0,2))
            i_sf_pref = np.argmax(mean_ori_sf)
            R_max = mean_ori_sf[i_sf_pref]
            R_min = np.min(mean_ori_sf)
            SSE = 0.0
            for sf_idx in range(len(unique_spatialFreq)):
                sf_mean = mean_ori_sf[sf_idx]
                all_trials_sf = response_array[i_ori_pref, :, sf_idx, :].flatten()
                SSE += np.sum((all_trials_sf - sf_mean) ** 2)
            N = response_array.shape[1] * response_array.shape[3] * response_array.shape[2]
            M = len(unique_spatialFreq)
            SFDI = (R_max - R_min) / (R_max + R_min + SSE / (N - M))
            
            print(f"Unit {unit_id}: OSI = {OSI:.2f}, gOSI = {gOSI:.2f}, SFDI = {SFDI:.2f}")
            print(f"Preferred orientation: {pref_ori_deg:.1f}°, best SF = {best_sf_value:.2f} cpd")
            
            # Save unit indices and metrics in the current shank's info dictionary
            unit_info_dict[str(unit_id)] = {
                'OSI': float(OSI),
                'gOSI': float(gOSI),
                'SFDI': float(SFDI),
                'pref_ori_index': int(i_ori_sel),
                'pref_ori_deg': float(pref_ori_deg),
                'pref_phase_index': int(i_phase_sel),
                'pref_sf_index': int(i_sf_sel),
                'i_ori_pref': int(i_ori_pref),
                'i_sf_pref': int(i_sf_pref),
                'R_max': float(R_max),
                'R_min': float(R_min)
            }
            
            # 4. Generate polar (fan) plot for response visualization
            overall_std = np.std(response_array)
            vmin = blank_mean
            vmax = blank_mean + 3.0 * overall_std
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            even_r = np.linspace(1, 5, n_spatialFreq)
            theta_grid, r_grid = np.meshgrid(np.deg2rad(unique_orientation), even_r, indexing='ij')
            theta_flat = theta_grid.flatten()
            r_flat = r_grid.flatten()
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 4), dpi=300)
            for idx_val in range(len(theta_flat)):
                theta_val = theta_flat[idx_val]
                r_val = r_flat[idx_val]
                o_idx = idx_val // n_spatialFreq
                sf_idx = idx_val % n_spatialFreq
                for p_idx, ph in enumerate(unique_phase):
                    data_here = response_array[o_idx, p_idx, sf_idx, :].flatten()
                    data_here_sorted = np.sort(data_here)[::-1]
                    local_uv = hex_offsets(len(data_here_sorted), radius=0.05)
                    dr_phase, dtheta_phase = phase_quad_offsets[p_idx]
                    dtheta_local = local_uv[:, 0] / r_val if r_val != 0 else local_uv[:, 0]
                    new_r = r_val + local_uv[:, 1] + dr_phase
                    new_theta = theta_val + dtheta_local + dtheta_phase
                    ax.scatter(new_theta, new_r,
                               c=data_here_sorted,
                               cmap=pink_reds,
                               s=1.2,
                               norm=norm,
                               clip_on=False,
                               alpha=1,
                               marker='h',
                               edgecolors='none')
            
            plt.colorbar(ax.collections[0], ax=ax, label='Firing Rate (Hz)')
            ax.set_thetamin(0)
            ax.set_thetamax(150)
            ax.set_thetagrids([0, 30, 60, 90, 120, 150],
                              labels=[f"{d}°" for d in [0, 30, 60, 90, 120, 150]])
            ax.set_rticks(even_r)
            ax.set_yticklabels([f"{sf:.2f}" for sf in unique_spatialFreq])
            ax.spines['polar'].set_visible(False)
            ax.set_frame_on(False)
            ax.set_title(f"Unit {unit_id}: {quality}")
            plt.tight_layout()
            text_str = (
                f"Preferred orientation: {pref_ori_deg:.1f}°\n"
                f"SFDI = {SFDI:.2f}\n"
                f"OSI = {OSI:.2f},  gOSI = {gOSI:.2f},  best SF = {best_sf_value:.2f} cpd"
            )
            fig.text(0.5, 0.01, text_str, ha='center', va='bottom', fontsize=10)
            out_file = out_fig_folder / f'unit_{unit_id}_fan_plot.png'
            plt.savefig(out_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved fan plot for unit {unit_id} to {out_file}")
    
    # After processing all sorting result folders for this shank, store its unit info
    all_shank_info[ish] = unit_info_dict

# Save all shank info into one NPZ file in the parent folder of the experiment folder.
npz_out_path = experiment_folder / rf"/Volumes/xieluanlabs/xl_cl/code/sortout/{animal_id}/{session_id}/static_grating_tuning_metrics.npz"
np.savez_compressed(npz_out_path, all_shank_info=all_shank_info)
print(f"\nSaved all shank info to {npz_out_path}")
