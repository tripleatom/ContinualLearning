import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os


animal_id = 'CnL22'
session_id = '20241113_155342'
ishs = ['0', '1', '2', '3']

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


start_time = 97322875 # ephys recording start time stamp
end_time = 127991645
black_start = 97830201 # black dots start time stamp
white_start = 112858628 # white dots start time stamp

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
        out_fig_folder = Path(sorting_results_folder) / 'STA'
        out_fig_folder = Path(out_fig_folder)
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)


        sorting_analyzer = load_sorting_analyzer(sorting_analyzer_folder)
        print(sorting_analyzer)
        sorting = sorting_analyzer.sorting

        unit_ids = sorting.unit_ids

        for i, unit_id in enumerate(unit_ids):
            spikes = sorting.get_unit_spike_train(unit_id)
            white_dot_spike = spikes[(spikes > white_start) & (spikes < white_start + 480 * 30000)]
            black_dot_spike = spikes[(spikes > black_start) & (spikes < black_start + 480 * 30000)]

            # convert to s
            white_dot_spike = (white_dot_spike - white_start) / 30000
            black_dot_spike = (black_dot_spike - black_start) / 30000

            ST  = []
            ST_white = []
            ST_black = []
            prior_time = 0.3
            for spikes in white_dot_spike:
                i_stimuli = int((spikes - prior_time) / dot_time) # 200ms before the spike, devided by dot_time (200ms) to get the index of the stimuli
                ST_white.append(white_dots_stimuli[i_stimuli])
                ST.append(white_dots_stimuli[i_stimuli])

            for spikes in black_dot_spike:
                i_stimuli = int((spikes - prior_time) / dot_time)
                ST.append(black_dots_stimuli[i_stimuli])
                ST_black.append(black_dots_stimuli[i_stimuli])

            # average of ST
            ST = np.array(ST)
            ST_white = np.array(ST_white)
            ST_black = np.array(ST_black)
            STA = np.mean(ST, axis=0)
            STA_white = np.mean(ST_white, axis=0)
            STA_black = np.mean(ST_black, axis=0)
            # plot STA, colorbar range from 0 to 1, cmap is blue to red
            # plt.imshow(STA, cmap='bwr', vmin=0, vmax=1)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)

            # Plot the subplots
            im0 = axes[0].imshow(STA, cmap='bwr')
            im1 = axes[1].imshow(STA_black, cmap='bwr')
            im2 = axes[2].imshow(STA_white, cmap='bwr')

            # Add colorbars for each subplot
            fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
            
            axes[0].set_title("STA")
            axes[1].set_title("STA_black")
            axes[2].set_title("STA_white")
            plt.savefig(out_fig_folder / f'unit_{unit_id}_prior_time_{prior_time}.png')
            plt.close()
