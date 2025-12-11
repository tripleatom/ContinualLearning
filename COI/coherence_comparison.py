import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import signal

# === CONFIGURATION ===
# List of pickle files to compare
pkl_files = [
    r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\3cms\data_251103_163841\data_251103_163841_brain_spinal_coherence.pkl",
    r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\3cms_2\data_251103_165302\data_251103_165302_brain_spinal_coherence.pkl",
    r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\data_251103_165957\data_251103_165957_brain_spinal_coherence.pkl",
]

# Labels for each session
session_labels = ['3cm', '3cm_2', 'static']

# Output folder for comparison plots
output_folder = r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\coherence_comparison"

# Smoothing parameters
smoothing_params = {
    'method': 'frequency_bin',  # Options: 'frequency_bin', 'moving_average', 'savgol', 'none'
    'freq_bin_size': 2.0,  # Hz - bin frequencies together (larger = smoother)
    'moving_avg_window': 5,  # Number of frequency points for moving average
    'savgol_window': 11,  # Window length for Savitzky-Golay filter (must be odd)
    'savgol_polyorder': 3,  # Polynomial order for Savitzky-Golay filter
}

# Plotting parameters - matching the example style
plot_params = {
    'figsize': (4, 3),  # Smaller, compact figure
    'dpi': 150,
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # Blue, Orange, Green
    'linewidth': 2,
    'freq_range': (0, 50),  # 0-50 Hz range
}

# === SMOOTHING FUNCTIONS ===
def smooth_coherence_frequency_bin(freqs, coherence, bin_size):
    """
    Smooth coherence by binning frequencies together and averaging.
    
    Parameters:
    -----------
    freqs : array
        Frequency array
    coherence : array
        Coherence values
    bin_size : float
        Size of frequency bins in Hz
    
    Returns:
    --------
    freqs_binned : array
        Binned frequency centers
    coherence_binned : array
        Averaged coherence in each bin
    """
    # Create bins
    freq_min, freq_max = freqs.min(), freqs.max()
    bins = np.arange(freq_min, freq_max + bin_size, bin_size)
    
    # Digitize frequencies into bins
    bin_indices = np.digitize(freqs, bins)
    
    # Average coherence within each bin
    freqs_binned = []
    coherence_binned = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            freqs_binned.append(np.mean(freqs[mask]))
            coherence_binned.append(np.mean(coherence[mask]))
    
    return np.array(freqs_binned), np.array(coherence_binned)

def smooth_coherence_moving_average(coherence, window):
    """
    Smooth coherence using a moving average filter.
    
    Parameters:
    -----------
    coherence : array
        Coherence values
    window : int
        Window size for moving average
    
    Returns:
    --------
    coherence_smoothed : array
        Smoothed coherence
    """
    if window < 1:
        return coherence
    
    # Use uniform convolution for moving average
    kernel = np.ones(window) / window
    coherence_smoothed = np.convolve(coherence, kernel, mode='same')
    
    return coherence_smoothed

def smooth_coherence_savgol(coherence, window, polyorder):
    """
    Smooth coherence using Savitzky-Golay filter.
    
    Parameters:
    -----------
    coherence : array
        Coherence values
    window : int
        Window length (must be odd)
    polyorder : int
        Polynomial order
    
    Returns:
    --------
    coherence_smoothed : array
        Smoothed coherence
    """
    if window < polyorder + 2 or len(coherence) < window:
        return coherence
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    coherence_smoothed = signal.savgol_filter(coherence, window, polyorder)
    
    return coherence_smoothed

def apply_smoothing(freqs, coherence, params):
    """
    Apply smoothing based on method specified in params.
    
    Returns:
    --------
    freqs_out : array
        Output frequencies (may be binned)
    coherence_out : array
        Smoothed coherence
    """
    method = params['method']
    
    if method == 'frequency_bin':
        return smooth_coherence_frequency_bin(freqs, coherence, params['freq_bin_size'])
    elif method == 'moving_average':
        coherence_smoothed = smooth_coherence_moving_average(coherence, params['moving_avg_window'])
        return freqs, coherence_smoothed
    elif method == 'savgol':
        coherence_smoothed = smooth_coherence_savgol(coherence, params['savgol_window'], 
                                                      params['savgol_polyorder'])
        return freqs, coherence_smoothed
    elif method == 'none':
        return freqs, coherence
    else:
        print(f"WARNING: Unknown smoothing method '{method}', using no smoothing")
        return freqs, coherence

# === LOAD ALL PICKLE FILES ===
print("Loading pickle files...")
sessions_data = []
session_names = []

for i, pkl_file in enumerate(pkl_files):
    pkl_path = Path(pkl_file)
    
    if not pkl_path.exists():
        print(f"WARNING: File not found: {pkl_file}")
        continue
    
    print(f"\nLoading: {pkl_path.name}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    sessions_data.append(data)
    session_names.append(session_labels[i])
    
    print(f"  Session: {session_labels[i]}")
    print(f"  Number of pairs: {len(data['coherence'])}")
    print(f"  Frequency points: {len(data['frequencies'])}")

n_sessions = len(sessions_data)
print(f"\nTotal sessions loaded: {n_sessions}")

if n_sessions == 0:
    print("ERROR: No valid pickle files found!")
    exit(1)

# === VERIFY COMPATIBILITY ===
print("\nVerifying data compatibility...")
ref_freqs = sessions_data[0]['frequencies']
for i, data in enumerate(sessions_data[1:], 1):
    if not np.allclose(ref_freqs, data['frequencies']):
        print(f"WARNING: Session {i} has different frequency array!")

print(f"\nSmoothing method: {smoothing_params['method']}")
if smoothing_params['method'] == 'frequency_bin':
    print(f"  Frequency bin size: {smoothing_params['freq_bin_size']} Hz")
elif smoothing_params['method'] == 'moving_average':
    print(f"  Moving average window: {smoothing_params['moving_avg_window']} points")
elif smoothing_params['method'] == 'savgol':
    print(f"  Savitzky-Golay window: {smoothing_params['savgol_window']}, polyorder: {smoothing_params['savgol_polyorder']}")

# === BUILD PAIR INDEX ===
print("\nBuilding pair index across sessions...")
pair_to_sessions = {}

for session_idx, data in enumerate(sessions_data):
    for pair_idx in range(len(data['coherence'])):
        pair_key = (
            data['brain_shank'][pair_idx],
            data['brain_channel_id'][pair_idx],
            data['spinal_shank'][pair_idx],
            data['spinal_channel_id'][pair_idx]
        )
        
        if pair_key not in pair_to_sessions:
            pair_to_sessions[pair_key] = {}
        
        pair_to_sessions[pair_key][session_idx] = pair_idx

print(f"Total unique pairs: {len(pair_to_sessions)}")

# Filter to only pairs that exist in ALL sessions
complete_pairs = {k: v for k, v in pair_to_sessions.items() if len(v) == n_sessions}
print(f"Pairs present in all {n_sessions} sessions: {len(complete_pairs)}")

if len(complete_pairs) == 0:
    print("ERROR: No pairs found in all sessions!")
    exit(1)

# === CREATE OUTPUT FOLDER ===
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)
print(f"\nSaving comparison plots to: {output_path}")

# === PLOT EACH PAIR ===
print(f"\nGenerating {len(complete_pairs)} comparison plots...")

for pair_key, session_pair_indices in tqdm(complete_pairs.items(), desc="Plotting pairs"):
    brain_shank, brain_ch, spinal_shank, spinal_ch = pair_key
    
    # Get position information from the first session (should be same across sessions)
    first_session_idx = 0
    first_pair_idx = session_pair_indices[first_session_idx]
    data = sessions_data[first_session_idx]
    
    # Extract positions (x, y) for brain and spinal channels
    brain_pos = data['brain_channel_position'][first_pair_idx]
    spinal_pos = data['spinal_channel_position'][first_pair_idx]
    
    # Format positions for display
    brain_x, brain_y = brain_pos[0], brain_pos[1]
    spinal_x, spinal_y = spinal_pos[0], spinal_pos[1]
    
    # Create figure with style matching the example
    fig, ax = plt.subplots(figsize=plot_params['figsize'])
    
    # Plot coherence for each session
    for session_idx in range(n_sessions):
        pair_idx = session_pair_indices[session_idx]
        data = sessions_data[session_idx]
        
        freqs = data['frequencies']
        coherence = data['coherence'][pair_idx]
        
        # Apply frequency range filter (0-50 Hz)
        freq_mask = (freqs >= plot_params['freq_range'][0]) & \
                    (freqs <= plot_params['freq_range'][1])
        freqs_filtered = freqs[freq_mask]
        coherence_filtered = coherence[freq_mask]
        
        # Apply smoothing
        freqs_plot, coherence_plot = apply_smoothing(freqs_filtered, coherence_filtered, 
                                                      smoothing_params)
        
        # Plot with clean style
        ax.plot(freqs_plot, coherence_plot, 
               color=plot_params['colors'][session_idx],
               linewidth=plot_params['linewidth'],
               label=session_names[session_idx])
    
    # Formatting - clean style like the example, with position info
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Coherence', fontsize=10)
    ax.set_title(f'Brain Sh{brain_shank} Ch{brain_ch} ({brain_x:.0f}, {brain_y:.0f}) - '
                f'Spinal Sh{spinal_shank} Ch{spinal_ch} ({spinal_x:.0f}, {spinal_y:.0f})',
                fontsize=9, fontweight='normal')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, None])  # Auto-scale y-axis, starting from 0
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure with position info in filename
    filename = (f"coherence_brain_sh{brain_shank}_ch{brain_ch:03d}_x{brain_x:.0f}_y{brain_y:.0f}_"
                f"spinal_sh{spinal_shank}_ch{spinal_ch:03d}_x{spinal_x:.0f}_y{spinal_y:.0f}.png")
    save_path = output_path / filename
    plt.savefig(save_path, dpi=plot_params['dpi'], bbox_inches='tight')
    plt.close()

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Sessions compared: {n_sessions}")
for i, name in enumerate(session_names):
    print(f"  {i+1}. {name}")
print(f"\nTotal pairs plotted: {len(complete_pairs)}")
print(f"Output folder: {output_path}")
print(f"\n{'='*60}")

# === CREATE SUMMARY STATISTICS ===
print("\nCalculating summary statistics...")

summary_data = {
    'session_names': session_names,
    'frequencies': ref_freqs,
    'mean_coherence': [],
    'std_coherence': [],
    'sem_coherence': [],
}

for session_idx in range(n_sessions):
    data = sessions_data[session_idx]
    
    # Get coherence for all pairs in this session
    all_coherence = []
    for pair_key in complete_pairs.keys():
        pair_idx = pair_to_sessions[pair_key][session_idx]
        all_coherence.append(data['coherence'][pair_idx])
    
    all_coherence = np.array(all_coherence)
    
    summary_data['mean_coherence'].append(np.mean(all_coherence, axis=0))
    summary_data['std_coherence'].append(np.std(all_coherence, axis=0))
    summary_data['sem_coherence'].append(np.std(all_coherence, axis=0) / np.sqrt(len(all_coherence)))

# Plot summary comparison - matching example style
fig, ax = plt.subplots(figsize=(5, 4))

for session_idx in range(n_sessions):
    freqs = summary_data['frequencies']
    mean_coh = summary_data['mean_coherence'][session_idx]
    sem_coh = summary_data['sem_coherence'][session_idx]
    
    # Apply frequency range filter (0-50 Hz)
    freq_mask = (freqs >= plot_params['freq_range'][0]) & \
                (freqs <= plot_params['freq_range'][1])
    freqs_filtered = freqs[freq_mask]
    mean_filtered = mean_coh[freq_mask]
    sem_filtered = sem_coh[freq_mask]
    
    # Apply smoothing
    freqs_plot, mean_plot = apply_smoothing(freqs_filtered, mean_filtered, smoothing_params)
    _, sem_plot = apply_smoothing(freqs_filtered, sem_filtered, smoothing_params)
    
    color = plot_params['colors'][session_idx]
    
    # Plot mean with clean style
    ax.plot(freqs_plot, mean_plot, 
           color=color,
           linewidth=plot_params['linewidth'],
           label=session_names[session_idx])
    
    # Plot SEM as shaded region
    ax.fill_between(freqs_plot, 
                    mean_plot - sem_plot,
                    mean_plot + sem_plot,
                    color=color,
                    alpha=0.2)

# Clean formatting
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('Coherence', fontsize=11)
ax.set_title('Average Brain-Spinal Coherence', fontsize=12, fontweight='normal')
ax.set_xlim([0, 50])
ax.set_ylim([0, None])
ax.legend(loc='upper right', frameon=True, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
summary_path = output_path / "summary_average_coherence.png"
plt.savefig(summary_path, dpi=plot_params['dpi'], bbox_inches='tight')
plt.close()

print(f"Summary plot saved: {summary_path}")

# Save summary statistics
summary_pkl_path = output_path / "summary_statistics.pkl"
with open(summary_pkl_path, 'wb') as f:
    pickle.dump(summary_data, f)

print(f"Summary statistics saved: {summary_pkl_path}")
print("\nDone!")