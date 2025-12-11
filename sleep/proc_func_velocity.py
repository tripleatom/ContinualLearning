import numpy as np
import pickle
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from pathlib import Path
import matplotlib.pyplot as plt


def compute_velocity(proc_file, velocity_threshold=530, method='savgol',
                     window_length=11, polyorder=3, median_window=5):
    """
    Compute velocity from position tracking data using smoothing methods.

    Parameters
    ----------
    proc_file : str
        Path to the _PROC pickle file containing tracking data
    velocity_threshold : float, optional
        Maximum velocity threshold. Values above this are set to 0 (default: 530)
    method : str, optional
        Smoothing method: 'savgol' (Savitzky-Golay), 'median', 'gaussian', or 'simple'
        (default: 'savgol')
    window_length : int, optional
        Length of the filter window (must be odd, for savgol/median). Default: 11
    polyorder : int, optional
        Order of polynomial for Savitzky-Golay filter. Default: 3
    median_window : int, optional
        Window size for median filtering of outliers. Default: 5

    Returns
    -------
    t : numpy.ndarray
        Time stamps aligned with velocity
    v : numpy.ndarray
        Velocity values (smoothed)
    v_raw : numpy.ndarray
        Raw velocity values (unsmoothed, for comparison)

    Examples
    --------
    >>> proc_file = r"\\server\path\to\Animal_date_session_PROC"
    >>> t, v, v_raw = compute_velocity(proc_file, method='savgol')
    """
    # Load data
    data = pickle.load(open(proc_file, 'rb'))

    x = data['center_x']
    y = data['center_y']
    time_stamp = data['time_stamp']

    # Method 1: Smooth positions first, then compute velocity
    if method == 'savgol':
        # Savitzky-Golay filter - fits local polynomial to data
        # This preserves features while smoothing noise
        x_smooth = savgol_filter(
            x, window_length=window_length, polyorder=polyorder)
        y_smooth = savgol_filter(
            y, window_length=window_length, polyorder=polyorder)

        # Compute velocity using differences on smoothed positions
        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    elif method == 'median':
        # Median filter - robust to outliers
        x_smooth = median_filter(x, size=median_window)
        y_smooth = median_filter(y, size=median_window)

        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    elif method == 'gaussian':
        # Gaussian smoothing using convolution
        from scipy.ndimage import gaussian_filter1d
        sigma = window_length / 6  # rule of thumb
        x_smooth = gaussian_filter1d(x, sigma=sigma)
        y_smooth = gaussian_filter1d(y, sigma=sigma)

        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    else:  # 'simple' - original method
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(time_stamp)
        t = time_stamp[1:]

    # Calculate distance
    d = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    epsilon = 1e-8
    dt = np.where(dt == 0, epsilon, dt)

    # Calculate raw velocity (before smoothing)
    dx_raw = np.diff(x)
    dy_raw = np.diff(y)
    d_raw = np.sqrt(dx_raw**2 + dy_raw**2)
    dt_raw = np.diff(time_stamp)
    dt_raw = np.where(dt_raw == 0, epsilon, dt_raw)
    v_raw = d_raw / dt_raw

    # Calculate velocity
    v = d / dt

    # Remove outliers: set velocities above threshold to 0
    v_raw_clean = np.where(v_raw > velocity_threshold, 0, v_raw)
    v = np.where(v > velocity_threshold, 0, v)

    # Optional: Apply additional smoothing to velocity itself
    if method in ['savgol', 'median', 'gaussian'] and len(v) > window_length:
        # Smooth velocity as well for extra smoothness
        v_smoothed = savgol_filter(v, window_length=min(window_length, len(v)//2*2+1),
                                   polyorder=min(polyorder, min(window_length, len(v)//2*2+1)-1))
        v = v_smoothed

    return t, v, v_raw_clean


def compute_velocity_advanced(proc_file, velocity_threshold=530,
                              window_length=11, polyorder=3):
    """
    Advanced velocity computation using Savitzky-Golay differentiation.
    This directly computes the derivative while smoothing, which is more
    accurate than smoothing then differencing.

    Parameters
    ----------
    proc_file : str
        Path to the _PROC pickle file containing tracking data
    velocity_threshold : float, optional
        Maximum velocity threshold. Values above this are set to NaN (default: 530)
    window_length : int, optional
        Length of the filter window (must be odd). Default: 11
    polyorder : int, optional
        Order of polynomial for Savitzky-Golay filter. Default: 3

    Returns
    -------
    t : numpy.ndarray
        Time stamps
    v : numpy.ndarray
        Velocity values (smoothed)
    vx : numpy.ndarray
        X component of velocity
    vy : numpy.ndarray
        Y component of velocity
    """
    # Load data
    data = pickle.load(open(proc_file, 'rb'))

    x = data['center_x']
    y = data['center_y']
    time_stamp = data['time_stamp']

    # Calculate mean sampling rate
    dt_mean = np.mean(np.diff(time_stamp))

    # Use Savitzky-Golay filter with derivative mode
    # This computes the derivative while smoothing in one step
    vx = savgol_filter(x, window_length=window_length, polyorder=polyorder,
                       deriv=1, delta=dt_mean)
    vy = savgol_filter(y, window_length=window_length, polyorder=polyorder,
                       deriv=1, delta=dt_mean)

    # Calculate velocity magnitude
    v = np.sqrt(vx**2 + vy**2)

    # Remove outliers: set velocities above threshold to NaN (better than 0)
    v = np.where(v > velocity_threshold, np.nan, v)
    vx = np.where(np.abs(vx) > velocity_threshold, np.nan, vx)
    vy = np.where(np.abs(vy) > velocity_threshold, np.nan, vy)

    # Optionally interpolate over NaN values
    # This is better than setting to 0
    mask = ~np.isnan(v)
    if np.sum(mask) > 0:  # If we have valid values
        v_interp = np.interp(time_stamp, time_stamp[mask], v[mask])
        vx_interp = np.interp(time_stamp, time_stamp[mask], vx[mask])
        vy_interp = np.interp(time_stamp, time_stamp[mask], vy[mask])
    else:
        v_interp = v
        vx_interp = vx
        vy_interp = vy

    return time_stamp, v_interp, vx_interp, vy_interp


# Example usage showing comparison of methods
if __name__ == "__main__":

    proc_file = Path(input("Enter the path to the _PROC pickle file: ").strip().strip("'").strip('"'))
    data_path = proc_file.parent
    figures_path = data_path / 'figures'
    figures_path.mkdir(exist_ok=True)
    # Compare different methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    methods = ['simple', 'savgol', 'median', 'gaussian']

    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]

        t, v, v_raw = compute_velocity(
            proc_file, method=method, window_length=11)

        # Make sure arrays have the same length
        print(f"{method}: t={len(t)}, v={len(v)}, v_raw={len(v_raw)}")

        ax.plot(t, v_raw, alpha=0.3, label='Raw', linewidth=0.5)
        ax.plot(t, v, label='Smoothed', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity')
        ax.set_title(f'Method: {method}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_path / 'velocity_comparison.png', dpi=150)
    plt.show()

    # Demonstrate advanced method
    print("\nTesting advanced method...")
    t_adv, v_adv, vx_adv, vy_adv = compute_velocity_advanced(
        proc_file, window_length=11)
    # save data to pickle
    velocity_data = {
        'time_stamp': t_adv,
        'velocity': v_adv,
        'velocity_x': vx_adv,
        'velocity_y': vy_adv
    }
    with open(data_path / 'velocity_advanced.pkl', 'wb') as f:
        pickle.dump(velocity_data, f)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t_adv, v_adv, linewidth=1)
    axes[0].set_ylabel('Speed')
    axes[0].set_title('Advanced Method: Savitzky-Golay Derivative')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_adv, vx_adv, label='Vx', alpha=0.7)
    axes[1].plot(t_adv, vy_adv, label='Vy', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_path / 'velocity_advanced.png', dpi=150)
    plt.show()
