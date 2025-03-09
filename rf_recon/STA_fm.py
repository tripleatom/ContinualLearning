import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
import scipy.io
import numpy as np
from pathlib import Path
import os
import process_func.DIO as DIO
from scipy.signal import find_peaks
import pickle
from rf_func import average_array


animal_id = 'CnL22'
session_id = '20241203_123534'
ishs = ['0', '1', '2', '3']

rec_folder = rf"D:\cl\rf_reconstruction\freelymoving\{animal_id}_{session_id}.rec"
rec_folder = Path(rec_folder)

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)
pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 2)


dlc_proc_file = r"D:\cl\video\Imaging_source_CnL22_2024-12-03_3_PROC"
with open(dlc_proc_file, 'rb') as f:
    dlc_proc = pickle.load(f)

ts_file = r"D:\cl\video\Imaging_source_CnL22_2024-12-03_3_TS.npy"
ts = np.load(ts_file)

averaged_ts = average_array(ts, 10)

unity_frame_folder = rec_folder / 'unity'
# load all images in unity folder in grayscale, and store them in a array
unity_frames = []
for unity_frame_file in unity_frame_folder.glob('*.png'):
    unity_frame = plt.imread(unity_frame_file)
    gray = np.dot(unity_frame[..., :3], [0.299, 0.587, 0.114])
    unity_frames.append(gray)
unity_frames = np.array(unity_frames)


files = rec_folder.glob("head_conf_eye_conf*.npy")
full_paths = [os.path.abspath(f) for f in files]
conf_file = Path(full_paths[0])
conf = np.load(conf_file)

conf = average_array(conf, 10)

# alignment
ephys_start_time = pd_time[0]
video_start = ts[0]

rising_edge, _ = find_peaks(pd_state)
ephys_peak_time = pd_time[rising_edge]
dlc_rising_edge, _ = find_peaks(dlc_proc['signal'])
dlc_peak_time = dlc_proc['signal_time'][dlc_rising_edge]

ephys_first_rising = ephys_peak_time[0]
dlc_first_rising = dlc_peak_time[0]


for ish in ishs:
    rec_folder = rf'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\{animal_id}\{session_id}\{ish}'
    sorting_results_folders = []
    for root, dirs, files in os.walk(rec_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):  # Check if the folder name matches the pattern
                sorting_results_folders.append(os.path.join(root, dir_name))

    for sorting_results_folder in sorting_results_folders:

        sorting_analyzer_folder = Path(sorting_results_folder) / 'sorting_analyzer'
        out_fig_folder = Path(sorting_results_folder) / 'STA_fm'
        out_fig_folder = Path(out_fig_folder)
        if not out_fig_folder.exists():
            out_fig_folder.mkdir(parents=True)

        sorting_analyzer = load_sorting_analyzer(sorting_analyzer_folder)
        print(sorting_analyzer)
        sorting = sorting_analyzer.sorting

        unit_ids = sorting.unit_ids
        fs = sorting.sampling_frequency


        n_unit = len(unit_ids)
        STA = np.zeros((n_unit, 27, 48))

        for i, unit_id in enumerate(unit_ids):
            spikes = sorting.get_unit_spike_train(unit_id) + ephys_start_time
            ts_aligned = averaged_ts - dlc_first_rising
            spikes_aligned = (spikes - ephys_first_rising)/fs  # convert to seconds


            prior_time = 0.3
            ST = []

            for spike in spikes_aligned:
                if spike <= 0 or spike >= ts_aligned[-1]:
                    continue
                idx = np.searchsorted(ts_aligned, spike-prior_time) - 1
                if idx < 0:
                    continue

                if np.any(conf[idx] < 0.7):
                    continue
                frame = unity_frames[idx]
                ST.append(frame)

            ST = np.array(ST)
            STA[i] = np.mean(ST, axis=0)

            plt.imshow(STA[i], cmap='gray')
            plt.title(f'unit {unit_id}')
            plt.savefig(out_fig_folder / f'unit_{unit_id}_priot_time_{prior_time}.png')
            plt.close()