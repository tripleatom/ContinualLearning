import process_func.DIO as DIO

import numpy as np
import matplotlib.pyplot as plt
import mplcursors


rec_folder = r"D:\cl\rf_reconstruction\head_fixed\CnL14_20241004_153555.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)

pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 2)

time_diff = np.diff(pd_time)/30000
freq = 1./time_diff / 1000

plt.figure(figsize=(10, 5))
# plt.plot(pd_time[30000:48000]/30000, freq[30000:48000])
plt.plot(pd_time[1:]/30000, freq)
plt.ylabel("Frequency (kHz)")

mplcursors.cursor(hover=True)

plt.show()

# use interactive mode to choose the start time of 2 trials
# 1st: black dot
# 2nd: white dot