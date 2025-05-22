import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os
import h5py
from rf_func import find_stim_index, moving_average, schmitt_trigger
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info

rec_folder = Path(input("Please enter the full path to the recording folder: ").strip().strip('"'))
stimdata_file = Path(input("Please enter the full path to the .mat file: ").strip().strip('"'))

print(f"Recording folder: {rec_folder}")
print(f"Stimulus data file: {stimdata_file}")

print(stimdata_file)
DIN_file = rec_folder / "DIN.mat"
peaks_file = rec_folder / "peaks.mat"

# Load the peaks data
peaks_data = scipy.io.loadmat(peaks_file, struct_as_record=False, squeeze_me=True)
rising_edges = peaks_data['locs']

# read digInFreq
with h5py.File(DIN_file, 'r') as f:
    data = f["frequency_parameters"]['board_dig_in_sample_rate'][:]
digInFreq = (data.decode('utf-8') if isinstance(data, bytes) else data)[0][0]


animal_id, session_id, folder_name = parse_session_info(rec_folder)
ishs = ['0', '1', '2', '3']
# ishs = ['0']

# get stim data from mat file
# mat_data = scipy.io.loadmat(
#     Stimdata_file, struct_as_record=False, squeeze_me=True)
# stimdata = mat_data['Stimdata']
# black_on = stimdata.black_on
# black_off = stimdata.black_off
# white_on = stimdata.white_on
# white_off = stimdata.white_off

# n_col = stimdata.n_col
# n_row = stimdata.n_row
# n_trial = stimdata.n_trial  # trials for each color
# t_trial = stimdata.t_trial  # time for each trial, s

with h5py.File(stimdata_file, 'r') as f:
    # Access the dataset for 'Stimdata'
    stimdata = f["Stimdata"]
    # orientations = stimdata['orientations'][:]
    # spatialFreqs = stimdata['spatialFreqs'][:]
    # phases = stimdata['phases'][:]
    # Extract the relevant fields from the nested array structure
    black_on = stimdata['black_on'][0]
    black_off = stimdata['black_off'][0]
    white_on = stimdata['white_on'][0]
    white_off = stimdata['white_off'][0]
    n_col = stimdata['n_col'][0][0].astype(int)
    n_row = stimdata['n_row'][0][0].astype(int)
    t_trial = stimdata['t_trial'][0][0]
    n_trial = stimdata['n_trial'][0][0]
    white_order = stimdata['white_order'][:].astype(int)
    black_order = stimdata['black_order'][:].astype(int)

n_rising_edges = len(rising_edges)
# n_rising_edges = 8960
n_trial = (n_rising_edges//(n_col * n_row)//2).astype(int)
print('repeats: ', n_trial)

rising_edges = rising_edges[:int(n_rising_edges)]

n_dots = n_col * n_row
dot_time = t_trial
trial_dur = n_col * n_row * n_trial * t_trial

white_order = white_order - 1 # matlab index starts from 1
black_order = black_order - 1 

# generate the stimuli pattern in array
black_dots_stimuli = np.ones((len(black_order), n_row, n_col))
white_dots_stimuli = np.zeros((len(white_order), n_row, n_col))

for i, dot in enumerate(black_order):
    row = dot // n_col
    col = dot % n_col
    black_dots_stimuli[i, row, col] = 0

for i, dot in enumerate(white_order):
    row = dot // n_col
    col = dot % n_col
    white_dots_stimuli[i, row, col] = 1


black_rising = rising_edges[0:n_dots * n_trial]
white_rising = rising_edges[n_dots * n_trial:]
black_rising = np.append(black_rising, black_rising[-1] + t_trial * digInFreq)
white_rising = np.append(white_rising, white_rising[-1] + t_trial * digInFreq)
black_start = black_rising[0]
black_end = black_rising[-1]
white_start = white_rising[0]
white_end = white_rising[-1]

for ish in ishs:
    print(f'Processing {animal_id}/{session_id}/{ish}')
    # rec_folder = rf'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\{animal_id}\{session_id}\{ish}'
    # for mac
    shank_folder = rf'//10.129.151.108/xieluanlabs/xl_cl/code/sortout/{animal_id}/{session_id}/{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            # Check if the folder name matches the pattern
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))

    for sorting_results_folder in sorting_results_folders:
        phy_folder = Path(sorting_results_folder) / 'phy'
        out_fig_folder = Path(sorting_results_folder) / 'STA'
        out_fig_folder = Path(out_fig_folder)
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)

        # sorting_anaylzer = load_sorting_analyzer(
        #     Path(sorting_results_folder) / 'sorting_analyzer')

        sorting = PhySortingExtractor(phy_folder)
        qualities = sorting.get_property('quality')
        # sorting = sorting_anaylzer.sorting

        unit_ids = sorting.unit_ids
        fs = sorting.sampling_frequency
        n_unit = len(unit_ids)

        # %% calculate STA
        STA = np.zeros((n_unit, n_row, n_col))
        STA_black = np.zeros((n_unit, n_row, n_col))
        STA_white = np.zeros((n_unit, n_row, n_col))

        for i, unit_id in enumerate(unit_ids):
            quality = qualities[i]
            # calculate STA for each unit
            spikes = sorting.get_unit_spike_train(unit_id)
            white_dot_spikes = spikes[(spikes > white_start) & (spikes < white_end)]
            black_dot_spikes = spikes[(spikes > black_start) & (spikes < black_end)]

            ST = []
            ST_white = []
            ST_black = []
            prior_time = 0.05  # s
            for spike in white_dot_spikes:
                i_trial = find_stim_index(spike-prior_time*fs, white_rising)
                if i_trial is None:
                    continue
                i_stimuli = i_trial % n_dots
                ST_white.append(white_dots_stimuli[i_stimuli])
                ST.append(white_dots_stimuli[i_stimuli])

            for spike in black_dot_spikes:
                i_trial = find_stim_index(spike-prior_time*fs, black_rising)
                if i_trial is None:
                    continue
                i_stimuli = i_trial % n_dots
                ST.append(black_dots_stimuli[i_stimuli])
                ST_black.append(black_dots_stimuli[i_stimuli])

            # average of ST
            ST = np.array(ST)
            ST_white = np.array(ST_white)
            ST_black = np.array(ST_black)
            STA[i, :, :] = np.mean(ST, axis=0)
            STA_white[i, :, :] = np.mean(ST_white, axis=0)
            STA_black[i, :, :] = np.mean(ST_black, axis=0)

        # %% calculate firing rate per pixel

        # parameter to tune
        delay = 0.05  # s
        average_length = 0.2  # s

        firing_all = np.zeros((len(unit_ids), n_row, n_col))
        firing_black_all = np.zeros((len(unit_ids), n_row, n_col))
        firing_white_all = np.zeros((len(unit_ids), n_row, n_col))
        for i_unit, unit_id in enumerate(unit_ids):
            spikes = sorting.get_unit_spike_train(unit_id)
            white_firing = np.zeros((n_dots, n_trial))
            black_firing = np.zeros((n_dots, n_trial))

            for i in range(n_dots):
                white_indexes = np.arange(0, n_trial * n_dots, n_dots) + np.where(white_order == i)[0] # indexes of the dot in the stimuli in each trial
                white_start_times = white_rising[white_indexes] + delay * fs
                white_end_times = white_start_times + average_length * fs
                for j in range(len(white_start_times)):
                    white_dot_spike = spikes[(spikes > white_start_times[j]) & (
                        spikes < white_end_times[j])]
                    white_firing[i, j] = len(white_dot_spike) / average_length

                black_indexes = np.arange(0, n_trial * n_dots, n_dots) + np.where(black_order == i)[0] # indexes of the dot in the stimuli in each trial
                black_start_times = black_rising[black_indexes] + delay * fs
                black_end_times = black_start_times + average_length * fs
                for j in range(len(black_start_times)):
                    black_dot_spike = spikes[(spikes > black_start_times[j]) & (
                        spikes < black_end_times[j])]
                    black_firing[i, j] = len(black_dot_spike) / average_length

            white_firing_ave = np.mean(white_firing, axis=1)
            black_firing_ave = np.mean(black_firing, axis=1)

            # rearange to n_row x n_col
            white_firing_ave = white_firing_ave.reshape(n_row, n_col)
            black_firing_ave = black_firing_ave.reshape(n_row, n_col)

            firing = white_firing_ave - black_firing_ave
            firing_all[i_unit, :, :] = firing
            firing_black_all[i_unit, :, :] = black_firing_ave
            firing_white_all[i_unit, :, :] = white_firing_ave

        # %% plot
        # Calculate angular extents (in degrees)
        display_width_mm = 707
        display_height_mm = 393
        distance_mm = 570

        x_min_deg = np.degrees(np.arctan((-display_width_mm/2) / distance_mm))
        x_max_deg = np.degrees(np.arctan((display_width_mm/2) / distance_mm))
        y_min_deg = np.degrees(np.arctan((-display_height_mm/2) / distance_mm))
        y_max_deg = np.degrees(np.arctan((display_height_mm/2) / distance_mm))

        for i, unit_id in enumerate(unit_ids):
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
            
            # Plot Firing data using visual angle extents
            im1 = axes[0, 0].imshow(firing_all[i], cmap='bwr', 
                                    extent=[x_min_deg, x_max_deg, y_min_deg, y_max_deg],
                                    origin='lower')
            axes[0, 0].set_title('Firing white - Firing black')
            axes[0, 0].set_xlabel('Horizontal angle (deg)')
            axes[0, 0].set_ylabel('Vertical angle (deg)')
            fig.colorbar(im1, ax=axes[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
            
            im2 = axes[0, 1].imshow(-firing_black_all[i], cmap='bwr', 
                                    extent=[x_min_deg, x_max_deg, y_min_deg, y_max_deg],
                                    origin='lower')
            axes[0, 1].set_title('- Firing Black')
            axes[0, 1].set_xlabel('Horizontal angle (deg)')
            axes[0, 1].set_ylabel('Vertical angle (deg)')
            fig.colorbar(im2, ax=axes[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
            
            im3 = axes[0, 2].imshow(firing_white_all[i], cmap='bwr', 
                                    extent=[x_min_deg, x_max_deg, y_min_deg, y_max_deg],
                                    origin='lower')
            axes[0, 2].set_title('Firing White')
            axes[0, 2].set_xlabel('Horizontal angle (deg)')
            axes[0, 2].set_ylabel('Vertical angle (deg)')
            fig.colorbar(im3, ax=axes[0, 2], orientation='vertical', fraction=0.046, pad=0.04,
                        label='averaged firing rate for each pixel')
            
            # Plot STA data with angular extents
            im4 = axes[1, 0].imshow(STA[i], cmap='bwr', 
                                    extent=[x_min_deg, x_max_deg, y_min_deg, y_max_deg],
                                    origin='lower')
            axes[1, 0].set_title('STA')
            axes[1, 0].set_xlabel('Horizontal angle (deg)')
            axes[1, 0].set_ylabel('Vertical angle (deg)')
            fig.colorbar(im4, ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
            
            im5 = axes[1, 1].imshow(STA_black[i], cmap='bwr', 
                                    extent=[x_min_deg, x_max_deg, y_min_deg, y_max_deg],
                                    origin='lower')
            axes[1, 1].set_title('STA Black')
            axes[1, 1].set_xlabel('Horizontal angle (deg)')
            axes[1, 1].set_ylabel('Vertical angle (deg)')
            fig.colorbar(im5, ax=axes[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
            
            im6 = axes[1, 2].imshow(STA_white[i], cmap='bwr', 
                                    extent=[x_min_deg, x_max_deg, y_min_deg, y_max_deg],
                                    origin='lower')
            axes[1, 2].set_title('STA White')
            axes[1, 2].set_xlabel('Horizontal angle (deg)')
            axes[1, 2].set_ylabel('Vertical angle (deg)')
            fig.colorbar(im6, ax=axes[1, 2], orientation='vertical', fraction=0.046, pad=0.04,
                        label='averaged stimuli gray scale')
            
            # Set overall figure title and save the plot
            fig.suptitle(f'Unit {unit_id}: {quality}', fontsize=16)
            plt.savefig(out_fig_folder / f'unit_{unit_id}_prior_time_{prior_time}_fr_{average_length}.png')
            plt.close()
