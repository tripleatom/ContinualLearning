import process_func.DIO as DIO

import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.signal import find_peaks


rec_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CnL22_20241113_155342.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)

pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 1)

time_diff = np.diff(pd_time)/30000
freq = 1./time_diff / 1000 # kHz

minima_indices, _ = find_peaks(-freq, distance=500, height=-1)

print("Indices of local minima:", minima_indices)
print("Values of local minima:", freq[minima_indices])

plt.figure(figsize=(10, 5))
plt.plot(pd_time[1:], freq)
plt.scatter(pd_time[minima_indices], freq[minima_indices], color='red', label="Local Minima")
plt.ylabel("Frequency (kHz)")

mplcursors.cursor(hover=True)

plt.show()

print('start time: ', pd_time[0])
print('end time: ', pd_time[-1])

# use interactive mode to choose the start time of 2 trials
# 1st: black dot
# 2nd: white dot

black_start = 97830201 # black dots start time stamp
black_end = 112242129
white_start = 112858619 # white dots start time stamp
white_end = 127270571
black_duration = (black_end - black_start)/30000
black_duration

# get the local minima numbers between black_start and black_end
black_minima_indices = minima_indices[(pd_time[minima_indices] > black_start) & (pd_time[minima_indices] < black_end)]
black_minima_indices

print('black minima number: ', len(black_minima_indices))


white_duration = (white_end - white_start)/30000
white_duration

# get the local minima numbers between white_start and white_end
white_minima_indices = minima_indices[(pd_time[minima_indices] > white_start) & (pd_time[minima_indices] < white_end)]
white_minima_indices

print('white minima number: ', len(white_minima_indices))
