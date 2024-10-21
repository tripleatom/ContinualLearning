import numpy as np
import matplotlib.pyplot as plt
import pickle

Animal='CnL13'
date = '2024-09-15'
session = 1

proc_file = rf"\\10.129.151.108\xieluanlabs\xl_cl\batch1_video\Imaging_source_{Animal}_{date}_{session}_PROC"

data = pickle.load(open(proc_file, 'rb'))
# data.keys()

# Assuming 'data' is a NumPy structured array or a dictionary with NumPy arrays
# Replace 'data' with your actual data variable

# Calculate differences between consecutive positions
dx = np.diff(data['x_pos'])
dy = np.diff(data['y_pos'])

# Calculate the distance between consecutive points
d = np.sqrt(dx**2 + dy**2)

# Calculate the time differences between consecutive time stamps
dt = np.diff(data['time_stamp'])

# Avoid division by zero by replacing zeros in dt with a small number
epsilon = 1e-8
dt = np.where(dt == 0, epsilon, dt)

# Calculate the velocity (distance/time)
v = d / dt

# remove outliers v>500, set to 0
v = np.where(v>530, 0, v)

# Use the midpoints of time stamps for plotting or the time stamps starting from index 1
time_stamps = data['time_stamp'][1:]  # Aligns with the length of 'v'

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First subplot: Velocity over Time
axs[0].plot(time_stamps, v, color='blue')
axs[0].set_xlabel('Time Stamp')
axs[0].set_ylabel('Velocity')
axs[0].set_title('Velocity over Time')

# Second subplot: Histogram of Velocity
axs[1].hist(v, bins=30, color='green', alpha=0.7)
axs[1].set_xlabel('Velocity')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of Velocity')

scatter = axs[2].scatter(data['x_pos'][1:], data['y_pos'][1:], c=v, alpha=0.3, s=5)
cbar = fig.colorbar(scatter, ax=axs[2])
cbar.set_label('Velocity')
axs[2].set_title('Velocity over Position')

fig.suptitle(f"{Animal}_{date}_{session}", fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig(rf"\\10.129.151.108\xieluanlabs\xl_cl\code\behavior_analysis\{Animal}_{date}_{session}_velocity.png")
