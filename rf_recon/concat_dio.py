import process_func.DIO as DIO

import numpy as np
import matplotlib.pyplot as plt
import mplcursors


rec_folder = r"D:\cl\rf_reconstruction\head_fixed\CnL22_20241113_155342.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)

pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 1)

time_diff = np.diff(pd_time)/30000
freq = 1./time_diff / 1000 # kHz

plt.figure(figsize=(10, 5))
# plt.plot(pd_time[30000:48000]/30000, freq[30000:48000])
plt.plot(pd_time[1:], freq)
plt.ylabel("Frequency (kHz)")

mplcursors.cursor(hover=True)

plt.show()

print('start time: ', pd_time[0])
print('end time: ', pd_time[-1])

# use interactive mode to choose the start time of 2 trials
# 1st: black dot
# 2nd: white dot