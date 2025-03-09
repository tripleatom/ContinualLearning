import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os


animal_id = 'CnL22'
session_id = '20241113_155342'
ishs = ['0', '1', '2', '3']

dot_time = 0.2 # seconds
trial_dur = 480 #seconds

# parameter to tune
delay = 0. #s
average_length = 0.5 #s

dots_order = scipy.io.loadmat(r'\\10.129.151.108\xieluanlabs\xl_cl\code\rf_recon\dots_order.mat')
dots_order = dots_order['dots_order'][0] - 1 # matlab index starts from 1


start_time = 97322875
end_time = 127991645
black_start = 97830201
white_start = 112858628

black_start = black_start - start_time
white_start = white_start - start_time


for ish in ishs:
    rec_folder = rf'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\{animal_id}\{session_id}\{ish}'

    sorting_results_folders = []
    for root, dirs, files in os.walk(rec_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):  # Check if the folder name matches the pattern
                sorting_results_folders.append(os.path.join(root, dir_name))


    for sorting_results_folder in sorting_results_folders:


        sorting_analyzer_folder = Path(sorting_results_folder) / 'sorting_analyzer'
        out_fig_folder = Path(sorting_results_folder) / 'pixel_ave'
        out_fig_folder = Path(out_fig_folder)
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)


        sorting_analyzer = load_sorting_analyzer(sorting_analyzer_folder)
        print(sorting_analyzer)
        sorting = sorting_analyzer.sorting

        unit_ids = sorting.unit_ids

        firing_all = np.zeros((len(unit_ids), 5, 8))
        firing_black_all = np.zeros((len(unit_ids), 5, 8))
        firing_white_all = np.zeros((len(unit_ids), 5, 8))
        for i_unit, unit_id in enumerate(unit_ids):
            spikes = sorting.get_unit_spike_train(unit_id)

            white_firing = np.zeros((40, 60))
            black_firing = np.zeros((40, 60))

            for i in range(40):
                index = np.where(dots_order == i)[0]
                start_time = index * dot_time + delay
                end_time = start_time + average_length
                for j in range(len(start_time)):
                    white_dot_spike = spikes[(spikes > white_start + start_time[j] * 30000) & (spikes < white_start + end_time[j] * 30000)]
                    black_dot_spike = spikes[(spikes > black_start + start_time[j] * 30000) & (spikes < black_start + end_time[j] * 30000)]
                    white_firing[i, j] = len(white_dot_spike) / average_length
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

        # normalize all firing to 0-1
        # firing_all = (firing_all - np.min(firing_all)) / (np.max(firing_all) - np.min(firing_all))
        # firing_black_all = (firing_black_all - np.min(firing_black_all)) / (np.max(firing_black_all) - np.min(firing_black_all))
        # firing_white_all = (firing_white_all - np.min(firing_white_all)) / (np.max(firing_white_all) - np.min(firing_white_all))

        for i, unit_id in enumerate(unit_ids):
            firing = firing_all[i]
            firing_black = firing_black_all[i]
            firing_white = firing_white_all[i]

            fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)

            # Plot the subplots
            im1 = axes[0].imshow(firing, cmap='bwr')
            axes[0].set_title('Firing')

            im2 = axes[1].imshow(firing_black, cmap='bwr')
            axes[1].set_title('Firing Black')

            im3 = axes[2].imshow(firing_white, cmap='bwr')
            axes[2].set_title('Firing White')

            fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
            fig.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
           

            # Add a single colorbar to the whole figure
            # cbar.set_label('Activation')

            # Add the main title
            fig.suptitle(f'Unit {unit_id}', fontsize=16)

            plt.savefig(out_fig_folder / f'unit_{unit_id}_delay_{delay}_ave_{average_length}.png')
            plt.close()