import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.signal import find_peaks
import scipy.io
from rf_func import moving_average, schmitt_trigger
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



# Path to the .mat file
# base_folder = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CnL22\250307"
experiment_folder = r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/CnL22/250307"  # for mac
experiment_folder = Path(experiment_folder)
rec_folder = next((p for p in experiment_folder.iterdir() if p.is_dir()), None)
Stimdata_file = next(experiment_folder.glob("*.mat"), None)
DIN_file = rec_folder / "DIN.mat"
mat_data =scipy.io.loadmat(Stimdata_file, struct_as_record=False, squeeze_me=True)

# Extract the struct object
stimdata = mat_data['Stimdata']

# Now, stimdata should be an object with attributes corresponding to each field
# For instance, you can access them using dot notation:
black_on = stimdata.black_on
black_off = stimdata.black_off
white_on = stimdata.white_on
white_off = stimdata.white_off

n_col = stimdata.n_col
n_row = stimdata.n_row
n_trial = stimdata.n_trial
t_trial = stimdata.t_trial

# Open the HDF5-based MAT file and load data
with h5py.File(DIN_file, 'r') as f:
    # Show the top-level keys in the file
    print("Top-level keys:", list(f.keys()))
    
    # Access the dataset or group named "recFile"
    DIN = f["recFile"]
    # Load the entire dataset into a NumPy array
    DIN_data = DIN[:]  
    print("DIN_data shape:", DIN_data.shape)
    
    # Access the frequency parameters struct
    freq_params = f["frequency_parameters"]
    # Load the dataset for 'board_dig_in_sample_rate'
    data = freq_params['board_dig_in_sample_rate'][:]
    # If the data is stored as bytes, decode it (if necessary)
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    # Extract the digital input frequency from the nested array structure
    digInFreq = data[0][0]

# Compute indices of rising edges in the DIN signal
# (Assumes DIN_data[0, :] contains the digital signal)
t_rising_edge_diode = np.where(np.diff(DIN_data[:, 2]) > 0)[0]
# Compute inter-pulse intervals (in samples)
inter_pulse_interval_diode = np.diff(t_rising_edge_diode)

pd_t = t_rising_edge_diode[0:-1]
pd = -inter_pulse_interval_diode

# 1) Smooth the original data (optional)
pd_smoothed = moving_average(pd, window_size=5)

# 2) Define Schmitt trigger thresholds based on the smoothed signal's min/max
min_val = np.min(pd_smoothed)
max_val = np.max(pd_smoothed)
low_threshold = min_val + 0.25 * (max_val - min_val)
high_threshold = min_val + 0.9 * (max_val - min_val)

# 3) Apply Schmitt trigger
pd_schmitt = schmitt_trigger(pd_smoothed, low_threshold, high_threshold)

# 4) Find rising edges in the Schmitt-triggered output
#    A rising edge occurs where the signal changes from 0 to 1
rising_edges = np.where(np.diff(pd_schmitt) > 0)[0]

# 5) Plot original (top) and smoothed (bottom) signals in a 2×1 figure,
#    marking the rising edges from the Schmitt output on both.
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

# Top subplot: original data
axes[0].plot(pd, label="Original Data", color="C0")
axes[0].plot(rising_edges, pd[rising_edges], "ro", label="Rising Edges (Schmitt)")
axes[0].set_title("Original Data with Rising Edges (detected on Smoothed + Schmitt)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()

# Bottom subplot: smoothed data
axes[1].plot(pd_smoothed, label="Smoothed Data", color="C1")
axes[1].plot(rising_edges, pd_smoothed[rising_edges], "ro", label="Rising Edges")
axes[1].plot(pd_schmitt * max_val, "g--", label="Schmitt Output (scaled)")  # Optional: show Schmitt
axes[1].set_title("Smoothed Data + Schmitt Output + Rising Edges")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

plt.tight_layout()
plt.show()

black_start = 1677280
black_end = 27347800
black_rising = rising_edges[(pd_t[rising_edges] > black_start) & (pd_t[rising_edges] < black_end)]
black_rising

print('black rising number: ', len(black_rising))
print('real black rising:' ,  n_col * n_row * n_trial)

white_start = 27630400
white_end = 53253600

white_rising = rising_edges[(pd_t[rising_edges] > white_start) & (pd_t[rising_edges] < white_end)]
white_rising

print('white rising number: ', len(white_rising))
print('real white rising:' ,  n_col * n_row * n_trial)