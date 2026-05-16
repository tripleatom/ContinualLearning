import matplotlib.pyplot as plt
npz_file = r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260320/CnL42_drifting_grating_exp_20260320_130447_DIO.npz"
import numpy as np
data = np.load(npz_file, allow_pickle=True)
rising_times = data['rising_times']
ITI = (rising_times[1:] - rising_times[:-1])/30000
plt.hist(ITI, bins=50)
print(data.files)