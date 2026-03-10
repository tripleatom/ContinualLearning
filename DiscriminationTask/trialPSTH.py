import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle

def plot_behavior_trial_psth(pkl_file, out_folder=None):
    """
    Plot PSTH for behavior trial data, grouped by white stimulus position.
    
    Parameters:
    -----------
    pkl_file : str or Path
        Path to the PKL file containing neural data
    out_folder : str or Path, optional
        Output folder for figures. If None, creates folder next to PKL file
    """
    
    # Load the data
    pkl_file = Path(pkl_file)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded data from {pkl_file}")
    print(f"Animal: {data['metadata']['animal_id']}")
    print(f"Session: {data['metadata']['session_id']}")
    print(f"Total units: {data['extraction_params']['total_units']}")
    print(f"Total trials: {data['metadata']['n_trials']}")
    
    # Set up output folder
    if out_folder is None:
        out_folder = pkl_file.parent / 'behavior_trial_psth'
    else:
        out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {out_folder}")
    
    # Extract metadata
    animal_id = data['metadata']['animal_id']
    session_id = data['metadata']['session_id']
    fs = data['metadata']['sampling_frequency']
    
    # Get trial information — support both white_on_left and rewarded_on_left
    trial_info_dict = data['trial_info']
    if 'white_on_left' in trial_info_dict:
        condition_key = 'white_on_left'
        left_label, right_label = 'White on Left', 'White on Right'
    elif 'rewarded_on_left' in trial_info_dict:
        condition_key = 'rewarded_on_left'
        left_label, right_label = 'Rewarded on Left', 'Rewarded on Right'
    else:
        raise KeyError("trial_info must contain 'white_on_left' or 'rewarded_on_left'")

    white_on_left = np.array(trial_info_dict[condition_key])

    # Window parameters
    window_pre = data['extraction_params']['window_pre']
    window_post = data['extraction_params']['window_post']

    # Get unique conditions
    condition_labels = {False: right_label, True: left_label}
    colors = {False: '#FF6B6B', True: '#4ECDC4'}  # Red for right, teal for left

    print(f"\nConditions:")
    print(f"  {left_label}: {np.sum(white_on_left)} trials")
    print(f"  {right_label}: {np.sum(~white_on_left)} trials")
    
    # Process each unit
    for unit_id, unit_data in data['spike_data'].items():
        print(f"\nProcessing {unit_id}")
        
        # Get unit info
        unit_info = data['unit_info'][unit_id]
        shank = unit_info['shank']
        original_id = unit_info['original_unit_id']
        quality = unit_info.get('quality', 'unknown')
        
        if quality == 'noise':
            print(f"  Skipping {unit_id} due to low quality")
            continue
        
        # Collect spike times for all trials
        unit_trial_spikes = []
        for trial in unit_data:
            spike_times = np.array(trial['spike_times'])
            unit_trial_spikes.append(spike_times)
        
        # Group trials by condition
        groups = {
            False: np.where(~white_on_left)[0],  # White on right
            True: np.where(white_on_left)[0]      # White on left
        }
        
        # Get average trial duration for window calculation
        avg_trial_duration = data['experiment_parameters']['trial_duration']
        window_post_trial = avg_trial_duration + window_post
        
        # --- Create combined figure ---
        plt.style.use('default')
        fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_facecolor('white')
        
        # --- Raster plot ---
        y_base = 0
        yticks = []
        ylabels = []
        
        for condition in [False, True]:  # Plot right first, then left
            idxs = groups[condition]
            n = len(idxs)
            
            for i, tidx in enumerate(idxs):
                if tidx < len(unit_trial_spikes):
                    spikes = unit_trial_spikes[tidx]
                    y = y_base + i + 0.5
                    if len(spikes) > 0:
                        ax_raster.scatter(np.array(spikes)*1000, np.full_like(spikes, y),
                                        s=8, color=colors[condition], marker='|', 
                                        alpha=0.8, linewidth=1.5)
            
            yticks.append(y_base + n/2)
            ylabels.append(condition_labels[condition])
            y_base += n
        
        ax_raster.set_ylim(0, y_base)
        ax_raster.set_yticks(yticks)
        ax_raster.set_yticklabels(ylabels, fontsize=11)
        ax_raster.set_ylabel('Trial Block', fontsize=12, fontweight='bold')
        ax_raster.set_title(f"{unit_id} — Quality: {quality}", fontsize=14, fontweight='bold', pad=20)
        ax_raster.grid(True, alpha=0.3, linestyle='--')
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)
        ax_raster.set_xlim(-window_pre*1000, window_post_trial*1000)
        
        # Add trial start/end lines
        ax_raster.axvline(x=0, color='green', linestyle='--', alpha=0.7, 
                         linewidth=2, label='Trial start')
        ax_raster.axvline(x=avg_trial_duration*1000, color='red', linestyle='--', 
                         alpha=0.7, linewidth=2, label='Trial end')
        ax_raster.legend(loc='upper right', fontsize=9)
        
        # --- PSTH with smoothing ---
        bin_width = 0.010  # 10ms bins
        bin_edges = np.arange(-window_pre, window_post_trial + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + bin_width/2
        
        # Gaussian smoothing parameters
        sigma_ms = 20  # smoothing width in ms
        sigma_bins = sigma_ms / (bin_width * 1000)  # convert to bins
        
        for condition in [False, True]:
            idxs = groups[condition]
            
            # Collect all spikes for this condition
            allspikes = []
            for idx in idxs:
                if idx < len(unit_trial_spikes) and len(unit_trial_spikes[idx]) > 0:
                    allspikes.extend(unit_trial_spikes[idx])
            
            if len(allspikes) > 0:
                allspikes = np.array(allspikes)
                counts, _ = np.histogram(allspikes, bins=bin_edges)
                rate = counts / (len(idxs) * bin_width)  # in Hz
                
                # Apply Gaussian smoothing
                rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
                
                ax_psth.plot(bin_centers*1000, rate_smooth,
                           label=condition_labels[condition], 
                           color=colors[condition], 
                           linewidth=2.5, alpha=0.9)
        
        ax_psth.set_xlabel('Time from trial start (ms)', fontsize=12, fontweight='bold')
        ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
        ax_psth.set_title('Peri-Stimulus Time Histogram (smoothed)', fontsize=12, fontweight='bold')
        ax_psth.legend(title='Condition', title_fontsize=10, fontsize=10, 
                      loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax_psth.grid(True, alpha=0.3, linestyle='--')
        ax_psth.spines['top'].set_visible(False)
        ax_psth.spines['right'].set_visible(False)
        ax_psth.set_xlim(-window_pre*1000, window_post_trial*1000)
        
        # Add trial start/end lines
        ax_psth.axvline(x=0, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax_psth.axvline(x=avg_trial_duration*1000, color='red', linestyle='--', 
                       alpha=0.7, linewidth=2)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save figure
        fig_name = f"{unit_id}_{quality}.png"
        fig.savefig(out_folder / fig_name, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"  Saved figure: {fig_name}")
    
    print(f"\nProcessing complete! All figures saved to {out_folder}")


# Example usage
if __name__ == '__main__':
    # Option 1: Specify the PKL file path directly
    pkl_file = r"/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/behavior_trial_embedding_20260309_1932.pkl"
    
    # Option 2: Or find the most recent PKL file in a session folder
    # from pathlib import Path
    # session_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\ephys\cl\sortout\CnL42SG\CnL42SG_20251217_151103")
    # pkl_files = list(session_folder.glob('behavior_trial_embedding_*.pkl'))
    # if pkl_files:
    #     pkl_file = sorted(pkl_files)[-1]  # Get most recent
    #     print(f"Found PKL file: {pkl_file}")
    # else:
    #     print("No PKL files found!")
    #     exit()
    
    # Plot the data
    plot_behavior_trial_psth(pkl_file)