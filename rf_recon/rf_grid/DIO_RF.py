import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import pickle

from process_func.DIO import get_dio_folders, concatenate_din_data
from task_file_reader import load_task_file
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor

rec_folder = Path(r"/Volumes/xieluanlabs/xl_cl/RF_GRID/250821/CnL39SG/CnL39SG_20250821_163039.rec")
task_file_Path = Path(r"/Volumes/xieluanlabs/xl_cl/RF_GRID/250821/CnL39_20250821_3.txt")
task_id = task_file_Path.stem
folder_path = task_file_Path.parent

animal_id = rec_folder.name.split('.')[0].split('_')[0]
session_id = rec_folder.name.split('.')[0]

print(f"Processing {animal_id}/{session_id}")

task_file = load_task_file(task_file_Path)
df = task_file.get_trial_data()
stimulus_duration = np.unique(df['StimulusDuration'].values)[0]
ITI_duration = np.unique(df['ITIDuration'].values)[0]
print(stimulus_duration, ITI_duration)
trial_duration = stimulus_duration + ITI_duration
n_repeats = task_file.get_experiment_params()['total_trials']
print("total_trials", n_repeats)

fs = 30000
# read din data
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
pd_time, pd_state = concatenate_din_data(dio_folders, 3)
pd_time = pd_time - pd_time[0]

rising = np.where(pd_state == 1)[0]
falling = np.where(pd_state == 0)[0]
rising_times = pd_time[rising]
falling_times = pd_time[falling]

rising_diff = np.diff(rising_times)/fs
print(np.where(rising_diff > 5)[0])

rising_rf_start = 1199
rising_rf_end = 2999

rising_times_rf = rising_times[rising_rf_start:rising_rf_end]
print(np.shape(rising_times_rf))
rising_diff = np.diff(rising_times_rf)/fs
plt.plot(rising_diff)
plt.show()
# print(rising_diff)
# print(np.where(rising_diff > 1.5)[0])

falling_times_rf = falling_times[rising_rf_start:rising_rf_end]

trial_times_rf = falling_times_rf - rising_times_rf
plt.plot(trial_times_rf/fs)
plt.show()

# adding the middle missed trial

# index = np.where(rising_diff > trial_duration+1)[0][0]
# print(index)

# rising_times_post = np.delete(rising_times_rf, index+1)

# print("post")
# print(np.shape(rising_times_post))
# plt.plot(np.diff(rising_times_post)/fs)
# plt.show()

# inserting_index = index
# a = rising_times_rf[inserting_index]
# b  = a + trial_duration * fs
# c = b + trial_duration * fs
# d = c + trial_duration * fs

# tmp = np.delete(rising_times_rf, inserting_index + 1)          # delete original idx 
# rising_times_post = np.insert(tmp, inserting_index + 1, [b,c,d]) 
# print("DIO shape", np.shape(rising_times_post))
if np.shape(rising_times_rf)[0] == n_repeats:
    print("DIO shape is correct")
else:
    print("DIO shape is incorrect")

# # a0 = rising_times_rf[0]
# # b0 = a0 - trial_duration * fs

# # rising_times_post = np.insert(rising_times_post, 0, [b0])


# falling_times_post = rising_times_post + stimulus_duration * fs

# save the rising and falling times
save_path = folder_path / f"{task_id}_DIO.npz"

np.savez_compressed(save_path, rising_times=rising_times_rf, falling_times=falling_times_rf)
print(f"Saved to {save_path}")