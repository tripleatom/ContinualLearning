import numpy as np


# the stim intervals are from the end of 1st stim to the start of the last stim
def find_stim_index(time, stim_intervals):
    for i in range(len(stim_intervals)-1):
        if time >= stim_intervals[i] and time < stim_intervals[i+1]:
            return i



def average_array(arr, block_size, axis=0):
    # Move the target axis to the front for easier indexing
    arr = np.moveaxis(arr, axis, 0)
    n = arr.shape[0]
    
    full_blocks = (n // block_size) * block_size
    # Average the full blocks
    averaged_data = arr[:full_blocks].reshape(-1, block_size, *arr.shape[1:]).mean(axis=1)

    # Handle the remainder
    remainder = n % block_size
    if remainder > 0:
        remainder_mean = arr[full_blocks:].mean(axis=0, keepdims=True)
        averaged_data = np.concatenate([averaged_data, remainder_mean], axis=0)
    
    # Move the axis back to its original position
    averaged_data = np.moveaxis(averaged_data, 0, axis)
    return averaged_data


def moving_average(x, window_size=5):
    """Simple moving average filter."""
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

def schmitt_trigger(signal, low_threshold, high_threshold):
    """
    Convert 'signal' into a binary 0/1 waveform using hysteresis:
    - Goes HIGH if signal > high_threshold.
    - Goes LOW if signal < low_threshold.
    """
    is_high = False
    output = np.zeros_like(signal, dtype=float)
    for i in range(len(signal)):
        if not is_high and signal[i] > high_threshold:
            is_high = True
        elif is_high and signal[i] < low_threshold:
            is_high = False
        output[i] = 1.0 if is_high else 0.0
    return output