import numpy as np
import h5py
import matplotlib.pyplot as plt

def dereference(item, f):
    """Recursively dereference an h5py item."""
    if isinstance(item, h5py.Reference):
        data = f[item][()]
        if isinstance(data, np.ndarray) and data.size == 1:
            return data.item()
        return data
    elif isinstance(item, np.ndarray):
        return np.array([dereference(elem, f) for elem in item])
    else:
        return item

def hex_offsets(n_points, radius=0.1):
    # Generate local offsets in hex layout
    coords = [(0, 0)]
    if n_points == 1:
        return np.array(coords)

    directions = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2],
    ])

    ring = 1
    while len(coords) < n_points:
        # Start at bottom-left
        x = ring * radius * directions[4][0]
        y = ring * radius * directions[4][1]

        for side in range(6):
            for step in range(ring):
                coords.append((x, y))
                if len(coords) >= n_points:
                    break
                dx, dy = directions[side]
                x += dx * radius
                y += dy * radius
            if len(coords) >= n_points:
                break

        ring += 1

    return np.array(coords[:n_points])


def h5py_to_dict(h5_group):
    """Recursively convert an h5py Group into a nested dictionary."""
    out_dict = {}
    for key in h5_group.keys():
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            # Recursively get subgroup
            out_dict[key] = h5py_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Get dataset data
            data = item[()]
            # Optional: Convert bytes to string if needed
            if isinstance(data, bytes):
                data = data.decode()
            # Optional: Convert MATLAB char arrays
            if isinstance(data, np.ndarray) and data.dtype.kind in {'S', 'O'}:
                try:
                    data = data.astype(str)
                except:
                    pass
            out_dict[key] = data
    return out_dict


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
    averaged_data = arr[:full_blocks].reshape(-1,
                                              block_size, *arr.shape[1:]).mean(axis=1)

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
