import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os
import h5py
from rf_func import find_stim_index, h5py_to_dict, hex_offsets
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.colors as mcolors

def dereference(item, f):
    """Recursively dereference an h5py item."""
    if isinstance(item, h5py.Reference):
        # Get the referenced data
        data = f[item][()]
        # If data is a single-element array, extract its value
        if isinstance(data, np.ndarray) and data.size == 1:
            return data.item()
        return data
    elif isinstance(item, np.ndarray):
        # Recursively dereference elements if item is an array
        return np.array([dereference(elem, f) for elem in item])
    else:
        return item
    
# animal_ids = ['CnL34', 'CnL35', 'CnL36']

# base_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CnL22\250307"
experiment_folder = r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/250314/CnL22"  # for mac
experiment_folder = Path(experiment_folder)
rec_folder = next((p for p in experiment_folder.iterdir() if p.is_dir()), None)
print(rec_folder)
Stimdata_file = next(experiment_folder.glob("static_grating*.mat"), None)
print(Stimdata_file)
DIN_file = rec_folder / "DIN.mat"
peaks_file = rec_folder / "peaks.mat"

# Load the peaks data
peaks_data = scipy.io.loadmat(
    peaks_file, struct_as_record=False, squeeze_me=True)
rising_edges = peaks_data['locs']

# Open the HDF5-based MAT file and load data
with h5py.File(DIN_file, 'r') as f:
    # Show the top-level keys in the file
    print("Top-level keys:", list(f.keys()))
    # Access the frequency parameters struct
    freq_params = f["frequency_parameters"]
    # Load the dataset for 'board_dig_in_sample_rate'
    data = freq_params['board_dig_in_sample_rate'][:]
    # If the data is stored as bytes, decode it (if necessary)
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    # Extract the digital input frequency from the nested array structure
    digInFreq = data[0][0]

animal_id, session_id, folder_name = parse_session_info(rec_folder)
ishs = ['0', '1', '2', '3']
# ishs = ['0']

with h5py.File(Stimdata_file, 'r') as f:
    patternParams_group = f['Stimdata']['patternParams']

    # Process 'orientation'
    orientation_data = patternParams_group['orientation'][()]
    stim_orientation = np.array([dereference(ref, f)
                                for ref in orientation_data])
    stim_orientation = stim_orientation.flatten().astype(float)

    # Process 'phase'
    phase_data = patternParams_group['phase'][()]
    stim_phase = np.array([dereference(ref, f) for ref in phase_data])
    stim_phase = stim_phase.flatten().astype(float)

    # Process 'spatialFreq'
    spatialFreq_data = patternParams_group['spatialFreq'][()]
    stim_spatialFreq = np.array([dereference(ref, f)
                                for ref in spatialFreq_data])
    stim_spatialFreq = stim_spatialFreq.flatten().astype(float)

print("Orientation:", stim_orientation)
print("Phase:", stim_phase)
print("Spatial Frequency:", stim_spatialFreq)

# Calculate the number of static gratings stimuli
n_static_grating = np.shape(stim_orientation)[0]
print(f"Number of static grating stimuli: {n_static_grating}")
print(f"Number of rising edges: {len(rising_edges)}")
static_grating_rising_edges = rising_edges[-n_static_grating:]


pink_reds = mcolors.LinearSegmentedColormap.from_list(
    'PinkReds',
    # Light pink ➜ medium pink ➜ dark red
    [(1, 0.9, 0.9), (1, 0.6, 0.6), (0.8, 0, 0)]
)

phase_offset_r = 0.2
phase_offset_theta = np.pi / 60
phase_quad_offsets = [
    (phase_offset_r,  phase_offset_theta),  # top-right
    (-phase_offset_r, phase_offset_theta),  # top-left
    (-phase_offset_r, -phase_offset_theta),  # bottom-left
    (phase_offset_r,  -phase_offset_theta)  # bottom-right
]

for ish in ishs:
    print(f'Processing {animal_id}/{session_id}/{ish}')
    # rec_folder = rf'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\{animal_id}\{session_id}\{ish}'
    # for mac
    shank_folder = rf'/Volumes/xieluanlabs/xl_cl/code/sortout/{animal_id}/{session_id}/{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            # Check if the folder name matches the pattern
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))

    for sorting_results_folder in sorting_results_folders:
        phy_folder = Path(sorting_results_folder) / 'phy'
        out_fig_folder = Path(sorting_results_folder) / 'static_grating'
        out_fig_folder = Path(out_fig_folder)
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)

            
        #TODO: use curated data
        sorting_anaylzer = load_sorting_analyzer(
            Path(sorting_results_folder) / 'sorting_analyzer')

        # sorting = PhySortingExtractor(phy_folder)
        sorting = sorting_anaylzer.sorting

        unit_ids = sorting.unit_ids
        fs = sorting.sampling_frequency
        n_unit = len(unit_ids)

        for i, unit_id in enumerate(unit_ids):
            spike_train = sorting.get_unit_spike_train(unit_id)

            # Initialize arrays to store responses
            responses = np.zeros((n_static_grating,))

            # FIXME: blank_mean should calculate from the real blank trials
            blank_average_time = 10.0  # 10s
            blank_start_time = static_grating_rising_edges[0] - blank_average_time * fs
            blank_end_time = static_grating_rising_edges[0]
            blank_spikes = np.sum((spike_train >= blank_start_time) & (spike_train < blank_end_time))
            blank_mean = blank_spikes / blank_average_time

            # Calculate responses for each static grating stimulus
            visual_transimission_delay = 0.05  # 50ms
            average_time = 0.2  # 200ms
            for j, edge in enumerate(static_grating_rising_edges):
                start_time = edge + visual_transimission_delay * fs
                end_time = start_time + average_time * fs
                responses[j] = np.sum((spike_train >= start_time) & (spike_train < end_time)) / average_time

            # After computing 'responses' for each unit_id:
            # print(f"Unit {unit_id} responses: {responses}")

            unique_orientation = np.unique(stim_orientation)
            unique_phase = np.unique(stim_phase)
            unique_spatialFreq = np.unique(stim_spatialFreq)

            n_orientation = len(unique_orientation)
            n_phase = len(unique_phase)
            n_spatialFreq = len(unique_spatialFreq)

            # Create a 3D array to store responses
            n_repeats = n_static_grating // (n_orientation * n_phase * n_spatialFreq)
            response_array = np.zeros((n_orientation, n_phase, n_spatialFreq, n_repeats))

            for i, ori in enumerate(unique_orientation):
                for j, ph in enumerate(unique_phase):
                    for k, sf in enumerate(unique_spatialFreq):
                        # Build a boolean mask for this combination (ori, ph, sf)
                        mask = (
                            (stim_orientation == ori) &
                            (stim_phase == ph) &
                            (stim_spatialFreq == sf)
                        )
                        # Find all indices that match
                        idxs = np.where(mask)[0]

                        # Now assign these responses to our 4D array
                        response_array[i, j, k, :] = responses[idxs]

            #TODO: calculate the metrics
            #1. OSI
            # 1) Average only over trials (axis=3)
            mean_over_repeats = np.mean(response_array, axis=3)
            # Now mean_over_repeats.shape = (n_orientation, n_phase, n_spatialFreq)

            # 2) Find the (orientation, phase, spatialFreq) that gives the maximum mean response
            #    We use unravel_index because mean_over_repeats is now 3D.
            i_ori, i_phase, i_sf = np.unravel_index(
                np.argmax(mean_over_repeats), 
                mean_over_repeats.shape
            )

            # Preferred response
            R_pref = mean_over_repeats[i_ori, i_phase, i_sf]

            # 3) Identify the orthogonal orientation index
            #    Assuming your orientations are evenly spaced and you want a 90° shift:
            i_orth = (i_ori + (n_orientation // 2)) % n_orientation

            # Response at orthogonal orientation (same phase & SF)
            R_orth = mean_over_repeats[i_orth, i_phase, i_sf]

            # 4) Compute OSI
            OSI = (R_pref - R_orth) / (R_pref + R_orth)

            # If you want the actual orientation in degrees:
            pref_ori_deg = unique_orientation[i_ori]



            #2. gOSI, global OSI
            # --- 1) Find the "preferred" spatial frequency across orientation & phase ---
            #     (You can change this logic if your experiment defines "preferred SF" differently.)
            sf_means = np.mean(response_array, axis=(0,1,3))  # shape = (n_spatialFreq,)
            i_best_sf = np.argmax(sf_means)
            best_sf_value = unique_spatialFreq[i_best_sf]

            # --- 2) For each orientation θ, compute Rθ = mean response at best SF (averaging repeats, phases) ---
            #     (If you prefer to pick the best phase for each orientation, adjust accordingly.)
            R_theta = np.mean(response_array[:, :, i_best_sf, :], axis=(1,2))  # shape = (n_orientation,)

            # --- 3) Convert orientation from degrees to radians.  Typically, gOSI uses 2θ for orientation periodicity of 180°.
            theta_radians = np.deg2rad(unique_orientation)  # shape = (n_orientation,)

            # --- 4) Compute the complex vector sum for 2θ and take magnitude ---
            #     numerator = Σ ( Rθ * e^{i2θ} )
            #     denominator = Σ Rθ
            numerator = np.sum(R_theta * np.exp(1j * 2 * theta_radians))
            denominator = np.sum(R_theta)
            gOSI = np.abs(numerator / denominator)

            print(f"Best SF = {best_sf_value:.2f} cpd")
            print(f"gOSI = {gOSI:.3f}")

            #3. SFDI, spatial frequency discrimination index
            # 1) Find the "preferred orientation" by averaging over phase, SF, repeats
            mean_by_ori = np.mean(response_array, axis=(1,2,3))  # shape: (n_ori,)
            i_ori_pref = np.argmax(mean_by_ori)

            # 2) At that orientation, compute mean response for each SF (averaging over phase & repeats)
            mean_ori_sf = np.mean(response_array[i_ori_pref, :, :, :], axis=(0,2))  # shape: (n_sf,)

            # Identify the preferred SF index and Rmax, also find Rmin
            i_sf_pref = np.argmax(mean_ori_sf)
            R_max = mean_ori_sf[i_sf_pref]         # max across SF at that orientation
            R_min = np.min(mean_ori_sf)            # min across SF at that orientation

            # 3) Compute SSE: sum of squared errors at the preferred orientation across *all* SF
            #    i.e., for each SF, compare each trial to that SF's mean
            SSE = 0.0
            for sf_idx in range(len(unique_spatialFreq)):
                sf_mean = mean_ori_sf[sf_idx]  # average response for this SF
                # all trials for this orientation & SF (flatten across phase & repeats)
                all_trials_sf = response_array[i_ori_pref, :, sf_idx, :].flatten()
                SSE += np.sum((all_trials_sf - sf_mean)**2)

            # 4) Define N and M
            N = response_array.shape[1] * response_array.shape[3] * response_array.shape[2]  
            #         = n_phase * n_repeats * n_sf  (all trials at this orientation)
            M = len(unique_spatialFreq)           # number of SF tested

            # 5) Compute SFDI
            SFDI = (R_max - R_min) / (R_max + R_min + SSE / (N - M))

            print(f"Preferred orientation index = {i_ori_pref}")
            print(f"Preferred SF index = {i_sf_pref}")
            print(f"R_max = {R_max:.2f},  R_min = {R_min:.2f}")
            print(f"SFDI = {SFDI:.3f}")

            #4. response reliability


            overall_std = np.std(response_array)

            vmin = blank_mean
            vmax = blank_mean + 3.0 * overall_std
            norm = Normalize(vmin=vmin, vmax=vmax)

            # TODO: seperate different phases

            # Create evenly spaced radial positions for plotting (e.g., 1 to 5)
            even_r = np.linspace(1, 5, n_spatialFreq)

            # Create a meshgrid so that each response value is associated with an orientation and a radial position.
            # We use the evenly spaced radial positions for plotting.
            theta_grid, r_grid = np.meshgrid(np.deg2rad(
                unique_orientation), even_r, indexing='ij')

            # Flatten the arrays for scatter plotting.
            theta_flat = theta_grid.flatten()
            r_flat = r_grid.flatten()
            response_flat = response_array.flatten()

            # Define normalization for the color scale.
            # Here, blank_mean maps to white and blank_mean + 3*overall_std to dark red.
            # Replace with your blank sweep mean
            blank_mean = np.mean(response_array)
            # Replace with your overall standard deviation
            overall_std = np.std(response_array)
            vmin = blank_mean
            vmax = blank_mean + 3.0 * overall_std
            norm = Normalize(vmin=vmin, vmax=vmax)

            # Create the polar scatter plot.
            fig, ax = plt.subplots(
                subplot_kw={'projection': 'polar'}, figsize=(8, 4), dpi=300)

            for i in range(len(theta_flat)):
                theta = theta_flat[i]
                r = r_flat[i]

                o_idx = i // n_spatialFreq
                sf_idx = i % n_spatialFreq

                for p_idx, ph in enumerate(unique_phase):
                    # Extract only this phase's data
                    data_here = response_array[o_idx, p_idx, sf_idx, :].flatten()

                    # Sort largest to smallest if desired
                    data_here_sorted = np.sort(data_here)[::-1]

                    # Generate a local beehive for these repeats
                    local_uv = hex_offsets(len(data_here_sorted), radius=0.05)

                    dr_phase, dtheta_phase = phase_quad_offsets[p_idx]
        
                    # Convert the local (u, v) offsets to polar increments:
                    # For the tangential part, convert arc-length to angular offset.

                    if r != 0:
                        dtheta_local = local_uv[:, 0] / r
                    else:
                        dtheta_local = local_uv[:, 0] 


                    new_r = r + local_uv[:, 1] + dr_phase
                    new_theta = theta + dtheta_local + dtheta_phase

                    sc = ax.scatter(new_theta, new_r,
                                    c=data_here_sorted,
                                    cmap=pink_reds,
                                    s=1.2,
                                    norm=norm,
                                    clip_on=False,
                                    alpha=1,
                                    marker='h',
                                    edgecolors='none')

            plt.colorbar(ax.collections[0], ax=ax, label='Firing Rate (Hz)')

            # Confine the angular range to 0-150 degrees.
            ax.set_thetamin(0)
            ax.set_thetamax(150)
            ax.set_thetagrids([0, 30, 60, 90, 120, 150], labels=[
                              f"{d}°" for d in [0, 30, 60, 90, 120, 150]])

            # Set radial ticks at the evenly spaced positions.
            ax.set_rticks(even_r)
            # Label each tick with the corresponding real spatial frequency (in cpd).
            ax.set_yticklabels([f"{sf:.2f}" for sf in unique_spatialFreq])

            # Remove the outermost axis label.
            ax.spines['polar'].set_visible(False)

            # remove the edge of the plot
            ax.set_frame_on(False)

            ax.set_title(f"Unit {unit_id}")
            plt.tight_layout()
            text_str = (
                f"Preferred orientation: {pref_ori_deg:.1f}°\n"
                f"SFDI = {SFDI:.2f}\n"
                f"OSI = {OSI:.2f},  gOSI = {gOSI:.2f},  best SF = {best_sf_value:.2f} cpd"

            )
            fig.text(
                0.5,    # x-position (fraction of figure width)
                0.01,   # y-position (fraction of figure height)
                text_str,
                ha='center', va='bottom', fontsize=10
            )
            out_file = out_fig_folder / f'unit_{unit_id}_fan_plot.png'
            plt.savefig(out_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved fan plot for unit {unit_id} to {out_file}")
