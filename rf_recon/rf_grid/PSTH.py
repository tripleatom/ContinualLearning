import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

from process_func.DIO import get_dio_folders, concatenate_din_data
from task_file_reader import load_task_file
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor

rec_folder = Path(r"C:\Users\Windows\freelymovingRF\250817\CnL22SG\CnL22SG_20250817_145143.rec")
task_file_Path = Path(r"D:\cl\grating_disk_representation\MouseGratingSphere\ExperimentData\CnL22_20250817_1.txt")

animal_id = rec_folder.name.split('.')[0].split('_')[0]
session_id = rec_folder.name.split('.')[0]

print(f"Processing {animal_id}/{session_id}")

task_file = load_task_file(task_file_Path)

# Get all trial data
df = task_file.get_trial_data()

# # Now you can work with standard pandas operations
# print(df.head())
# print(df.describe())
# print(df['StimulusIndex'].value_counts())

stimulus_index = df['StimulusIndex'].values
# remove the first trial
stimulus_index = stimulus_index[1:]


# read din data
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
pd_time, pd_state = concatenate_din_data(dio_folders, 3)
pd_time = pd_time - pd_time[0]

rising = np.where(pd_state == 1)[0]
falling = np.where(pd_state == 0)[0]
rising_times = pd_time[rising]
falling_times = pd_time[falling]


fs = 30000
print(np.diff(rising_times)/fs)
print(np.diff(falling_times)/fs)

# throw the first trial, and the last is not a trial
rising_times = rising_times[1:]
falling_times = falling_times[1:]

n_trials = len(stimulus_index)
trial_windows = [(rising_times[i], falling_times[i]) for i in range(n_trials)]

code_folder = Path(__file__).parent.parent.parent
session_folder = code_folder / f"sortout/{animal_id}/{session_id}"

out_folder = session_folder / 'RFBlock'
out_folder.mkdir(parents=True, exist_ok=True)


# Compute neural responses
all_units_responses = []
unit_info = []
all_unit_qualities = []
fs = None

# Iterate through shanks (similar to static_disc)
ishs = ['0', '1', '2', '3']

# Setup for visualization - similar to first file
unique_stim = np.unique(stimulus_index)  # Exclude last trial
n_stim_types = len(unique_stim)
colors = plt.cm.viridis(np.linspace(0, 1, n_stim_types))
stim2color = dict(zip(unique_stim, colors))

# Sort trials by stimulus index for better visualization
sorted_idx = np.argsort(stimulus_index)
sorted_stimulus_index = stimulus_index[sorted_idx]

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
            # Load sorting analyzer first, then fall back to phy
            sorting_analyzer_path = Path(sorting_results_folder) / 'sorting_analyzer'

            if phy_folder.exists():
                sorting = PhySortingExtractor(phy_folder)

            elif sorting_analyzer_path.exists():
                sorting_analyzer = load_sorting_analyzer(sorting_analyzer_path)
                sorting = sorting_analyzer.sorting
            else:
                print(f"No valid sorting data found in {sorting_results_folder}")
                continue

            # Set sampling freq
            if fs is None:
                fs = sorting.sampling_frequency
                print(f"Sampling frequency: {fs} Hz")

            print(f"Processing {sorting_results_folder}")
            print(f"unit number: {len(sorting.unit_ids)}")
            
            # Get unit qualities for this sorting
            unit_ids = sorting.unit_ids
            unit_qualities = sorting.get_property('quality') if hasattr(sorting, 'get_property') else ['good'] * len(unit_ids)
            
            # Window parameters (matching first file's style)
            window_pre = 0.2  # seconds before stimulus onset
            window_post = 1.0  # seconds after stimulus onset
            
            for unit_idx, unit_id in enumerate(unit_ids):
                print(f"Processing {animal_id}/{session_id}/shank{ish}/unit{unit_id}")
                spike_train = sorting.get_unit_spike_train(unit_id)
                quality = unit_qualities[unit_idx] if unit_idx < len(unit_qualities) else 'unknown'
                
                if quality == 'noise':
                    print(f"Skipping unit {unit_id} due to low quality")
                    continue

                unit_trial_spikes = []
                # store each trial spikes
                for i_trial, (start, end) in enumerate(trial_windows):
                    trial_spikes = spike_train[(spike_train >= start - window_pre * fs) & 
                                              (spike_train < start + window_post * fs)]
                    if len(trial_spikes) > 0:
                        trial_spikes_relative = (trial_spikes - start) / fs
                        unit_trial_spikes.append(trial_spikes_relative)
                    else:
                        unit_trial_spikes.append([])

                # Group trials by stimulus index
                groups = {stim: np.where(stimulus_index[:-1] == stim)[0] for stim in unique_stim}
                
                # --- Create combined figure (matching first file's style) ---
                plt.style.use('default')
                fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                fig.patch.set_facecolor('white')

                # --- Raster plot ---
                y_base = 0
                yticks = []
                ylabels = []
                
                for stim in unique_stim:
                    idxs = groups[stim]
                    n = len(idxs)
                    for i, tidx in enumerate(idxs):
                        if tidx < len(unit_trial_spikes):
                            spikes = unit_trial_spikes[tidx]
                            y = y_base + i + 0.5
                            if len(spikes) > 0:
                                ax_raster.scatter(np.array(spikes)*1000, np.full_like(spikes, y),
                                                s=8, color=stim2color[stim], marker='|', 
                                                alpha=0.8, linewidth=1.5)
                    yticks.append(y_base + n/2)
                    ylabels.append(f"Stim {stim}")
                    y_base += n

                ax_raster.set_ylim(0, y_base)
                ax_raster.set_yticks(yticks)
                ax_raster.set_yticklabels(ylabels, fontsize=11)
                ax_raster.set_ylabel('Trial Block (by stimulus)', fontsize=12, fontweight='bold')
                ax_raster.set_title(f"Unit {unit_id} — Quality: {quality}", fontsize=14, fontweight='bold', pad=20)
                ax_raster.grid(True, alpha=0.3, linestyle='--')
                ax_raster.spines['top'].set_visible(False)
                ax_raster.spines['right'].set_visible(False)
                ax_raster.set_xlim(-window_pre*1000, window_post*1000)
                
                # Add stimulus onset line
                ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stimulus onset')

                # --- PSTH with smoothing ---
                bin_width = 0.010  # 10ms bins
                bin_edges = np.arange(-window_pre, window_post + bin_width, bin_width)
                bin_centers = bin_edges[:-1] + bin_width/2
                
                # Gaussian smoothing parameters
                sigma_ms = 20  # smoothing width in ms
                sigma_bins = sigma_ms / (bin_width * 1000)  # convert to bins

                for stim in unique_stim:
                    idxs = groups[stim]
                    # Collect all spikes for this stimulus type
                    allspikes = []
                    for idx in idxs:
                        if idx < len(unit_trial_spikes) and len(unit_trial_spikes[idx]) > 0:
                            allspikes.extend(unit_trial_spikes[idx])
                    
                    if len(allspikes) > 0:
                        allspikes = np.array(allspikes)
                        counts, _ = np.histogram(allspikes, bins=bin_edges)
                        rate = counts / (len(idxs) * bin_width)  # in Hz
                        
                        # Apply Gaussian smoothing
                        rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
                        
                        ax_psth.plot(bin_centers*1000, rate_smooth,
                                   label=f"Stim {stim}", color=stim2color[stim], 
                                   linewidth=2.5, alpha=0.9)

                ax_psth.set_xlabel('Time from stimulus onset (ms)', fontsize=12, fontweight='bold')
                ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
                ax_psth.set_title('Peri-Stimulus Time Histogram (smoothed)', fontsize=12, fontweight='bold')
                ax_psth.legend(title='Stimulus', title_fontsize=9, fontsize=8, 
                              ncol=min(3, n_stim_types), loc='upper right', 
                              frameon=True, fancybox=True, shadow=True, 
                              bbox_to_anchor=(0.98, 0.98))
                ax_psth.grid(True, alpha=0.3, linestyle='--')
                ax_psth.spines['top'].set_visible(False)
                ax_psth.spines['right'].set_visible(False)
                ax_psth.set_xlim(-window_pre*1000, window_post*1000)
                
                # Add stimulus onset line
                ax_psth.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)

                # Adjust layout and styling
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.3)
                
                # Save with higher DPI and better format
                fig.savefig(out_folder / f"shank{ish}_unit{unit_id:03d}_{quality}.png", 
                           dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close(fig)
                
                print(f"Saved figure for shank{ish}_unit{unit_id}")
                
        except Exception as e:
            print(f"Error processing {sorting_results_folder}: {e}")
            continue

print("Processing complete!")