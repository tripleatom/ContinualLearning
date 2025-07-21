import process_func.DIO as DIO

import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.signal import find_peaks


rec_folder = r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/250609/CnL22SG/CnL22SG_20250609_164650.rec"

dio_folders = DIO.get_dio_folders(rec_folder)
dio_folders = sorted(dio_folders, key=lambda x:x.name)

pd_time, pd_state = DIO.concatenate_din_data(dio_folders, 3)

print('start time: ', pd_time[0])
print('end time: ', pd_time[-1])

mplcursors.cursor(hover=True)

plt.show()
# plt.close()

# calculate how many rising edges in the dio signal, the dio signal change from 0 to 1
rising_edge, _ = find_peaks(pd_state)
print('rising edge numbers:', len(rising_edge))

plt.figure(figsize=(10, 5))
plt.plot(pd_time, pd_state)
plt.scatter(pd_time[rising_edge], pd_state[rising_edge], color='red')
plt.ylabel("DIO")
plt.show()
plt.close()

#%% load corresbonding dlc processor file
dlc_proc_file = r"D:\cl\video\Imaging_source_CnL22_2024-12-03_3_PROC"
import pickle
with open(dlc_proc_file, 'rb') as f:
    dlc_proc = pickle.load(f)



# calculate the rising edge numbers in the dlc signal, the dlc signal change from 0 to 1
dlc_rising_edge, _ = find_peaks(dlc_proc['signal'])
print('rising edge numbers:', len(dlc_rising_edge))

plt.plot(dlc_proc['signal_time'], dlc_proc['signal'])

# mark the rising edge with red dots
plt.scatter(dlc_proc['signal_time'][dlc_rising_edge], dlc_proc['signal'][dlc_rising_edge], color='red')

plt.show()



# confirmed the sg dio and teensy recorded the same numbers of rising edges
