import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os
import process_func.DIO as DIO
from scipy.signal import find_peaks
from rf_func import find_stim_index
from spikeinterface.extractors import PhySortingExtractor

animal_id = 'LGN01'
session_id = '20241204_164306'
ishs = ['0']
# ishs = ['0', '1', '2', '3']

dot_time = 0.2 # each stimulus last for 0.2s
trial_dur = 480 # each trial last for 480 s. 5*8 pixels, 60 repeats.

dots_order = scipy.io.loadmat(r'\\10.129.151.108\xieluanlabs\xl_cl\code\rf_recon\dots_order.mat')
dots_order = dots_order['dots_order'][0] - 1 # matlab index starts from 1

black_dots_stimuli = np.ones((len(dots_order), 5, 8))
white_dots_stimuli = np.zeros((len(dots_order), 5, 8))

for i, dot in enumerate(dots_order):
    row = dot // 8
    col = dot % 8
    black_dots_stimuli[i, row, col] = 0
    white_dots_stimuli[i, row, col] = 1


start_time = 5365388 # ephys recording start time stamp
end_time = 36528408

rec_folder = rf"D:\cl\rf_reconstruction\head_fixed\{animal_id}_{session_id}.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)

pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 1)

time_diff = np.diff(pd_time)/30000
freq = 1./time_diff / 1000 # kHz

minima_indices, _ = find_peaks(-freq, distance=500, height=-1)
black_start = 6044083 # black dots start time stamp
black_end = 20456037

# get the local minima numbers between black_start and black_end
black_minima_indices = minima_indices[(pd_time[minima_indices] > black_start) & (pd_time[minima_indices] < black_end)]
black_minima = pd_time[black_minima_indices]
black_minima = np.insert(black_minima, 0, black_start)
black_minima = np.append(black_minima, black_end)

print('black minima number: ', len(black_minima))

white_start = 21071020 # white dots start time stamp
white_end = 35482975

# get the local minima numbers between white_start and white_end
white_minima_indices = minima_indices[(pd_time[minima_indices] > white_start) & (pd_time[minima_indices] < white_end)]
white_minima = pd_time[white_minima_indices]
white_minima = np.insert(white_minima, 0, white_start)
white_minima = np.append(white_minima, white_end)
print('white minima number: ', len(white_minima))


for ish in ishs:
    rec_folder = rf'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\{animal_id}\{session_id}\{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(rec_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):  # Check if the folder name matches the pattern
                sorting_results_folders.append(os.path.join(root, dir_name))

    for sorting_results_folder in sorting_results_folders:
        phy_folder = Path(sorting_results_folder) / 'phy'
        out_fig_folder = Path(sorting_results_folder) / 'STA'
        out_fig_folder = Path(out_fig_folder)
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)

        sorting = PhySortingExtractor(phy_folder)

        unit_ids = sorting.unit_ids
        fs = sorting.sampling_frequency


        # %% calculate STA
        n_unit = len(unit_ids)
        STA = np.zeros((n_unit, 5, 8))
        STA_black = np.zeros((n_unit, 5, 8))
        STA_white = np.zeros((n_unit, 5, 8))

        for i, unit_id in enumerate(unit_ids):
            spikes = sorting.get_unit_spike_train(unit_id) + start_time  # spike train in sorting is start from 0
            white_dot_spikes = spikes[(spikes > white_start) & (spikes < white_end)]
            black_dot_spikes = spikes[(spikes > black_start) & (spikes < black_end)]

            ST  = []
            ST_white = []
            ST_black = []
            prior_time = 0.2 # s
            for spike in white_dot_spikes:
                i_stimuli = find_stim_index(spike-prior_time*fs, white_minima)
                if i_stimuli is None:
                    continue
                ST_white.append(white_dots_stimuli[i_stimuli])
                ST.append(white_dots_stimuli[i_stimuli])

            for spike in black_dot_spikes:
                i_stimuli = find_stim_index(spike-prior_time*fs, black_minima)
                if i_stimuli is None:
                    continue
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

        #parameter to tune
        delay = 0. #s
        average_length = 0.5 #s

        firing_all = np.zeros((len(unit_ids), 5, 8))
        firing_black_all = np.zeros((len(unit_ids), 5, 8))
        firing_white_all = np.zeros((len(unit_ids), 5, 8))
        for i_unit, unit_id in enumerate(unit_ids):
            spikes = sorting.get_unit_spike_train(unit_id) + start_time  # spike train in sorting is start from 0

            white_firing = np.zeros((40, 60))
            black_firing = np.zeros((40, 60))

            for i in range(40):
                index = np.where(dots_order == i)[0]
                white_start_times = white_minima[index] + delay * fs
                white_end_times = white_start_times + average_length * fs
                for j in range(len(white_start_times)):
                    white_dot_spike = spikes[(spikes > white_start_times[j]) & (spikes < white_end_times[j])]
                    white_firing[i, j] = len(white_dot_spike) / average_length


                black_start_times = black_minima[index] + delay * fs
                black_end_times = black_start_times + average_length * fs
                for j in range(len(black_start_times)):
                    black_dot_spike = spikes[(spikes > black_start_times[j]) & (spikes < black_end_times[j])]
                    black_firing[i, j] = len(black_dot_spike) / average_length

            white_firing_ave = np.mean(white_firing, axis=1)
            black_firing_ave = np.mean(black_firing, axis=1)

            # rearange to 5x8
            white_firing_ave = white_firing_ave.reshape(5, 8)
            black_firing_ave = black_firing_ave.reshape(5, 8)

            firing = white_firing_ave - black_firing_ave
            firing_all[i_unit, :, :] = firing
            firing_black_all[i_unit, :, :] = black_firing_ave
            firing_white_all[i_unit, :, :] = white_firing_ave

        # %% plot
        for i, unit_id in enumerate(unit_ids):

            # Create a 2x3 grid of subplots
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

            # First row (Firing data)
            im1 = axes[0, 0].imshow(firing_all[i], cmap='bwr')
            axes[0, 0].set_title('Firing white - Firing black')
            fig.colorbar(im1, ax=axes[0, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im2 = axes[0, 1].imshow(-firing_black_all[i], cmap='bwr')
            axes[0, 1].set_title('- Firing Black')
            fig.colorbar(im2, ax=axes[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im3 = axes[0, 2].imshow(firing_white_all[i], cmap='bwr')
            axes[0, 2].set_title('Firing White')
            fig.colorbar(im3, ax=axes[0, 2], orientation='vertical', fraction=0.046, pad=0.04, label='averaged firing rate for each pixel')

            # Second row (STA data)
            im4 = axes[1, 0].imshow(STA[i], cmap='bwr')
            axes[1, 0].set_title('STA')
            fig.colorbar(im4, ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im5 = axes[1, 1].imshow(STA_black[i], cmap='bwr')
            axes[1, 1].set_title('STA Black')
            fig.colorbar(im5, ax=axes[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im6 = axes[1, 2].imshow(STA_white[i], cmap='bwr')
            axes[1, 2].set_title('STA White')
            fig.colorbar(im6, ax=axes[1, 2], orientation='vertical', fraction=0.046, pad=0.04, label='averaged stimuli gray scale')

            # Set the overall title
            fig.suptitle(f'Unit {unit_id}', fontsize=16)
            plt.savefig(out_fig_folder / f'unit_{unit_id}_prior_time_{prior_time}.png')
            plt.close()
