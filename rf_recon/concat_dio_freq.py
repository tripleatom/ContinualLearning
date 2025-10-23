#%%
import process_func.DIO as DIO

import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


rec_folder = r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/251002/CnL42SG/CnL42SG_20251002_200839.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)
#%%
pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 3)
pd_time = pd_time - pd_time[0] # reset the time to start from 0

time_diff = np.diff(pd_time)/30000
freq = 1./time_diff / 1000 # kHz

pd_time = pd_time[1:] # align the time with freq
# f = interp1d(pd_time, freq, kind='cubic')
# pd_time_regular = np.linspace(pd_time.min(), pd_time.max(), len(pd_time)*10)
# freq_regular = f(pd_time_regular)

#%%
maximum_indices, _ = find_peaks(freq, distance=500, height=9.5, prominence=0.5)
print("Indices of local maxima:", maximum_indices)
print("Values of local maxima:", freq[maximum_indices])

plt.figure(figsize=(10, 5))
plt.plot(pd_time[1:], freq)
T1 = 0
T2 = 50000000

valid_indices = maximum_indices[(pd_time[maximum_indices] > T1) & (pd_time[maximum_indices] < T2)]

plt.scatter(pd_time[valid_indices], freq[valid_indices], color='red', label="Local Maxima")
plt.ylabel("Frequency (kHz)")
plt.xlim(T1, T2)

mplcursors.cursor(hover=True)

plt.show()
print('valid maxima number: ', len(valid_indices))
# use interactive mode to choose the start time of 2 trials
# 1st: black dot
# 2nd: white dot

#%%
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
