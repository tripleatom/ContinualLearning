import scipy.io
import h5py
from pathlib import Path
from rec2nwb.preproc_func import parse_session_info
import os
from spikeinterface.extractors import PhySortingExtractor
from spikeinterface.core import load_sorting_analyzer
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# rec_folder = r"F:\flicker\250717\CnL22\CnL22_250717_110531"
rec_folder = input("Enter the path to the recording folder: ")
ishs = [0,1,2,3]

rec_folder = Path(rec_folder)

# load stimulus sequence
# matfile = r"F:\flicker\featureSequence_experiment_06042025.mat"
matfile = "/Volumes/xieluanlabs/xl_cl/flicker/featureSequence_experiment_06042025.mat"
data = scipy.io.loadmat(matfile, struct_as_record=False, squeeze_me=True)
# Access the 'features' struct
features = data['features']
# Access section1
section1 = features.section1
# Get the trialFrequencies array
trial_frequencies = section1.trialFrequencies
n_trials = len(trial_frequencies)
freqs = np.unique(trial_frequencies)
colors = plt.cm.viridis(np.linspace(0, 1, len(freqs)))
freq2color = dict(zip(freqs, colors))

sorted_idx = np.argsort(trial_frequencies)
sorted_trial_frequencies = trial_frequencies[sorted_idx]



# load peaks file
DIN_file = rec_folder / "DIN.mat"
peaks_file = rec_folder / "peaks.mat"
print(DIN_file)
print(peaks_file)

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

recorded_trials = len(rising_edges)

print(f"Recorded trials: {recorded_trials}")
print(f"Expected trials: {n_trials}")

# store each unit's response for each trial separately
animal_id, session_id, folder_name = parse_session_info(rec_folder)
code_folder = Path(__file__).parent.parent.parent
session_folder = code_folder / rf"sortout/{animal_id}/{animal_id}_{session_id}"

# Create session-level flicker folder
session_flicker_folder = session_folder / 'flicker'
if not session_flicker_folder.exists():
    session_flicker_folder.mkdir(parents=True)

for ish in ishs:
    print(f"Processing {animal_id}/{session_id}/shank{ish}")
    shank_folder = session_folder / f'shank{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))
    
    for sorting_results_folder in sorting_results_folders:
        phy_folder = Path(sorting_results_folder) / 'phy'
        
        # Load sorting analyzer (optionally use curated data)
        if phy_folder.exists():
            sorting = PhySortingExtractor(phy_folder)
        else:
            sorting_anaylzer = load_sorting_analyzer(Path(sorting_results_folder) / 'sorting_analyzer')
            sorting = sorting_anaylzer.sorting

        unit_ids = sorting.unit_ids
        fs = sorting.sampling_frequency
        unit_qualities = sorting.get_property('quality')
        window_pre = 0.5  # seconds before peak
        window_post = 4.0  # seconds after peak
        
        for i, unit_id in enumerate(unit_ids):
            print(f"Processing {animal_id}/{session_id}/shank{ish}/unit{unit_id}")
            spike_train = sorting.get_unit_spike_train(unit_id)
            quality = unit_qualities[i]
            # skip units with quality 'noise'
            if quality == 'noise':
                print(f"Skipping unit {unit_id} due to low quality")
                continue

            # Convert rising_edges to seconds if not already
            # If rising_edges are in samples, convert to seconds: rising_edges_sec = rising_edges / fs

            unit_trial_spikes = []  # list to hold spike times for each trial for this unit

            for peak_time in rising_edges:
                start_idx = peak_time - window_pre * fs
                end_idx = peak_time + window_post * fs
                # Extract spikes in this window (spike_train is in sample indices)
                spikes_in_window = spike_train[(spike_train >= start_idx) & (spike_train < end_idx)]
                if len(spikes_in_window) > 0:
                    # Optionally, convert to relative time from window start (in seconds)
                    spikes_relative = (spikes_in_window - peak_time) / fs
                    unit_trial_spikes.append(spikes_relative)
                else:
                    unit_trial_spikes.append([])

            # Now unit_trial_spikes is a list of arrays, one per trial, for this unit
            # create combined figure
            # --- plotting (corrected x/y dims) ---

            # 1) prepare grouping
            groups = {f: np.where(trial_frequencies==f)[0] for f in freqs}

            # Set up professional styling
            plt.style.use('default')
            fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.patch.set_facecolor('white')

            # --- Raster: scatter spikes, grouped by freq ---
            y_base = 0
            yticks = []
            ylabels = []
            for f in freqs:
                idxs = groups[f]
                n = len(idxs)
                for i, tidx in enumerate(idxs):
                    spikes = unit_trial_spikes[tidx]
                    y = y_base + i + 0.5
                    ax_raster.scatter(spikes*1000, np.full_like(spikes, y),
                                    s=8, color=freq2color[f], marker='|', alpha=0.8, linewidth=1.5)
                yticks.append(y_base + n/2)
                ylabels.append(f"{f} Hz")
                y_base += n

            ax_raster.set_ylim(0, y_base)
            ax_raster.set_yticks(yticks)
            ax_raster.set_yticklabels(ylabels, fontsize=11)
            ax_raster.set_ylabel('Trial Block (by frequency)', fontsize=12, fontweight='bold')
            ax_raster.set_title(f"Unit {unit_id} — Quality: {quality}", fontsize=14, fontweight='bold', pad=20)
            ax_raster.grid(True, alpha=0.3, linestyle='--')
            ax_raster.spines['top'].set_visible(False)
            ax_raster.spines['right'].set_visible(False)
            ax_raster.set_xlim(-window_pre*1000, window_post*1000)
            
            # Add stimulus onset line
            ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stimulus onset')

            # --- PSTH on bottom axis (use bin centers) with smoothing ---
            bin_width = 0.010  # seconds
            bin_edges = np.arange(-window_pre, window_post + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width/2
            
            # Gaussian smoothing parameters
            sigma_ms = 20  # smoothing width in ms
            sigma_bins = sigma_ms / (bin_width * 1000)  # convert to bins

            for f in freqs:
                idxs = groups[f]
                allspikes = np.hstack([unit_trial_spikes[i] for i in idxs if len(unit_trial_spikes[i]) > 0])
                counts, _ = np.histogram(allspikes, bins=bin_edges)
                rate = counts / (len(idxs) * bin_width)  # in Hz
                
                # Apply Gaussian smoothing
                rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
                
                ax_psth.plot(bin_centers*1000, rate_smooth,
                            label=f"{f} Hz", color=freq2color[f], linewidth=2.5, alpha=0.9)

            ax_psth.set_xlabel('Time from stimulus onset (ms)', fontsize=12, fontweight='bold')
            ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
            ax_psth.set_title('Peri-Stimulus Time Histogram (smoothed)', fontsize=12, fontweight='bold')
            ax_psth.legend(title='Frequency', title_fontsize=9, fontsize=8, 
                          ncol=min(3, len(freqs)), loc='upper right', 
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
            fig.savefig(session_flicker_folder / f"shank{ish}_unit{unit_id:03d}_{quality}.png", 
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)