from pathlib import Path
from parse_grating_experiment import parse_grating_experiment
import numpy as np
from process_func.DIO import get_dio_folders, concatenate_din_data
import matplotlib.pyplot as plt

rec_folder = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39SG_20251031_085159.rec")
task_file_Path = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39_drifting_grating_exp_20251031_085247.txt")
task_id = task_file_Path.stem
folder_path = task_file_Path.parent

animal_id = rec_folder.name.split('.')[0].split('_')[0]
session_id = rec_folder.name.split('.')[0]

print(f"Processing {animal_id}/{session_id}")

task_file = parse_grating_experiment(task_file_Path)

# Access specific data
print(f"Animal: {task_file['metadata']['animal_id']}")
print(f"Total trials: {task_file['parameters']['total_trials']}")

# Work with trial data
df = task_file['trial_data']
# mean_heading = df['Heading'].mean()
# trials_90deg = df[df['L_Orient'] == 90.0]

# print(mean_heading)
# print(trials_90deg)

stimulus_duration = task_file['parameters']['stimulus_duration']
ITI_duration = task_file['parameters']['iti_duration']
stimulus_duration = float(stimulus_duration.rstrip('s'))
ITI_duration = float(ITI_duration.rstrip('s'))
n_repeats = task_file['parameters']['total_trials']
trial_duration = stimulus_duration + ITI_duration

print("stimulus_duration", stimulus_duration, "s")
print("ITI_duration", ITI_duration, "s")
print("n_repeats", n_repeats)
print("trial_duration", trial_duration, "s")

fs = 30000
# read din data
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
pd_time, pd_state = concatenate_din_data(dio_folders, 3)
pd_time = pd_time - pd_time[0]

rising = np.where(pd_state == 1)[0]
falling = np.where(pd_state == 0)[0]
rising_times = pd_time[rising]
falling_times = pd_time[falling]

# rising_times = np.delete(rising_times, [52])
rising_times = np.delete(rising_times, [56])  # remove known glitch
falling_times = np.delete(falling_times, [55])  # remove known glitch

# rising_times = np.insert(rising_times, 51, rising_times[50] + 30000 * 3)
# rising_times = np.insert(rising_times, 51, rising_times[50]+30000*3)


rising_diff = np.diff(rising_times)/fs
print(np.where(rising_diff > 5)[0])




rising_rf_start = 0
# rising_rf_end = 

rising_times_rf = rising_times[rising_rf_start:]

print("rising times shape", np.shape(rising_times_rf))

falling_times_rf = rising_times_rf + int(stimulus_duration*fs)
# falling_times_rf = falling_times[rising_rf_start:]
trial_times_rf = falling_times_rf - rising_times_rf
plt.plot(rising_diff/fs)
plt.show()

# print(np.diff(rising_times_rf)/fs)

save_path = folder_path / f"{task_id}_DIO.npz"
np.savez_compressed(save_path, rising_times=rising_times_rf, falling_times=falling_times_rf)
print(f"Saved to {save_path}")