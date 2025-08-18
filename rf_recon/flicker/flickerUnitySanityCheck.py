import process_func.DIO as DIO
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.signal import find_peaks
import pickle

rec_folder = r"/Volumes/xieluanlabs/xl_cl/flicker/250723/CnL22SG/CnL22SG_20250723_165112.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)

pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 3)

ephys_freq = 30000
rising_edges = np.where(pd_state==1)[0]
pd_time_rising = pd_time[rising_edges]

t_interval = 4

# Keep the first rising edge, then only those at least 3s after the previous kept one
filtered_times = [pd_time_rising[0]]
for t in pd_time_rising[1:]:
    if t - filtered_times[-1] >= t_interval * ephys_freq:
        filtered_times.append(t)
filtered_times = np.array(filtered_times)

print(len(filtered_times))

t_trial_interval = np.diff(filtered_times) / ephys_freq
# plot the histogram of t_trial_interval
plt.hist(t_trial_interval,)

plt.figure(figsize=(10, 5))
plt.vlines(pd_time_rising, 0, 1, colors='red', linewidth=2, label='Rising Edges')
plt.ylim(-0.1, 1.1)
# plt.xlim(0, 10)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Method 1: Vertical Lines')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()