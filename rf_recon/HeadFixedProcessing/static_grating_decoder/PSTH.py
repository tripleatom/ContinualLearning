import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from itertools import product

def plot_static_grating_psth(npz_file, output_folder=None, units_to_plot=None):
    """
    Plot PSTH for static grating responses using the style from the first code.
    
    Parameters:
    -----------
    npz_file : Path or str
        Path to the NPZ file containing static grating responses
    output_folder : Path or str, optional
        Folder to save figures. If None, saves to npz_file's parent directory
    units_to_plot : list, optional
        List of unit indices to plot. If None, plots all units
    """
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    units_data = data['units_data']
    
    # Get stimulus parameters
    unique_orientation = data['unique_orientation']
    unique_phase = data['unique_phase']
    unique_spatialFreq = data['unique_spatialFreq']
    
    # Get unit qualities if available
    unit_qualities = data.get('unit_qualities', None)
    
    # Time windows
    pre_stim_window = float(data['pre_stim_window'])
    post_stim_window = float(data['post_stim_window'])
    
    # Setup output folder
    if output_folder is None:
        output_folder = Path(npz_file).parent / 'static_grating_psth'
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine which units to plot
    if units_to_plot is None:
        units_to_plot = range(len(units_data))
    
    # Create all possible stimulus combinations
    stim_combinations = list(product(unique_orientation, unique_phase, unique_spatialFreq))
    n_stim_types = len(stim_combinations)
    
    # Create color map for different stimulus conditions
    colors = plt.cm.viridis(np.linspace(0, 1, n_stim_types))
    stim2color = {combo: colors[i] for i, combo in enumerate(stim_combinations)}
    
    print(f"Processing {len(units_to_plot)} units with {n_stim_types} stimulus conditions")
    
    # Process each unit
    for unit_idx in units_to_plot:
        unit_data = units_data[unit_idx]
        unit_id = unit_data['unit_id']
        shank = unit_data['shank']
        trials = unit_data['trials']
        
        # Get quality for this unit if available
        quality = unit_qualities[unit_idx] if unit_qualities is not None and unit_idx < len(unit_qualities) else 'unknown'
        
        print(f"Processing shank{shank}_unit{unit_id} (quality: {quality})")
        
        # Organize trials by stimulus condition
        trials_by_condition = {combo: [] for combo in stim_combinations}
        
        for trial in trials:
            stim_key = (trial['orientation'], trial['phase'], trial['spatial_frequency'])
            trials_by_condition[stim_key].append(trial)
        
        # Create figure with two subplots (matching first code's style)
        plt.style.use('default')
        fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor('white')
        
        # --- Raster plot ---
        y_base = 0
        yticks = []
        ylabels = []
        
        for stim_idx, (stim_combo, stim_trials) in enumerate(trials_by_condition.items()):
            if len(stim_trials) == 0:
                continue
                
            ori, phase, sf = stim_combo
            color = stim2color[stim_combo]
            
            # Plot spikes for each trial of this condition
            for trial_i, trial in enumerate(stim_trials):
                y_pos = y_base + trial_i + 0.5
                spike_times = trial['spike_times'] * 1000  # Convert to ms
                
                if len(spike_times) > 0:
                    ax_raster.scatter(spike_times, np.full_like(spike_times, y_pos),
                                    s=8, color=color, marker='|', 
                                    alpha=0.8, linewidth=1.5)
            
            # Add tick for this stimulus condition
            if len(stim_trials) > 0:
                yticks.append(y_base + len(stim_trials)/2)
                ylabels.append(f"O:{ori:.0f}° P:{phase:.0f}° SF:{sf:.2f}")
                y_base += len(stim_trials)
        
        # Configure raster plot
        ax_raster.set_ylim(0, y_base)
        ax_raster.set_yticks(yticks)
        ax_raster.set_yticklabels(ylabels, fontsize=9)
        ax_raster.set_ylabel('Stimulus Conditions', fontsize=12, fontweight='bold')
        ax_raster.set_title(f"Unit {unit_id} (Shank {shank}) — Quality: {quality} - Static Grating Responses", 
                           fontsize=14, fontweight='bold', pad=20)
        ax_raster.grid(True, alpha=0.3, linestyle='--')
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)
        ax_raster.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
        
        # Add stimulus onset line
        ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stimulus onset')
        
        # --- PSTH with smoothing (matching first code's style) ---
        bin_width = 0.010  # 10ms bins
        bin_edges = np.arange(-pre_stim_window, post_stim_window + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + bin_width/2
        
        # Gaussian smoothing parameters (matching first code)
        sigma_ms = 20  # smoothing width in ms
        sigma_bins = sigma_ms / (bin_width * 1000)  # convert to bins
        
        # Plot PSTH for each stimulus condition
        legend_labels = []
        for stim_idx, (stim_combo, stim_trials) in enumerate(trials_by_condition.items()):
            if len(stim_trials) == 0:
                continue
                
            ori, phase, sf = stim_combo
            color = stim2color[stim_combo]
            
            # Collect all spikes for this stimulus condition
            all_spikes = []
            for trial in stim_trials:
                all_spikes.extend(trial['spike_times'])
            
            if len(all_spikes) > 0:
                all_spikes = np.array(all_spikes)
                counts, _ = np.histogram(all_spikes, bins=bin_edges)
                
                # Calculate firing rate in Hz
                n_trials = len(stim_trials)
                rate = counts / (n_trials * bin_width)
                
                # Apply Gaussian smoothing
                rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
                
                # Plot with abbreviated label for legend
                label = f"O:{ori:.0f}° SF:{sf:.2f}"
                ax_psth.plot(bin_centers*1000, rate_smooth,
                           label=label, color=color, 
                           linewidth=2.5, alpha=0.9)
                legend_labels.append(label)
        
        # Configure PSTH plot
        ax_psth.set_xlabel('Time from stimulus onset (ms)', fontsize=12, fontweight='bold')
        ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
        ax_psth.set_title('Peri-Stimulus Time Histogram (smoothed)', fontsize=12, fontweight='bold')
        
        # Add legend with adaptive columns
        n_cols = min(3, (n_stim_types + 2) // 3)  # Max 3 columns
        if len(legend_labels) > 0:
            ax_psth.legend(title='Stimulus', title_fontsize=9, fontsize=8, 
                          ncol=n_cols, loc='upper right', 
                          frameon=True, fancybox=True, shadow=True, 
                          bbox_to_anchor=(0.98, 0.98))
        
        ax_psth.grid(True, alpha=0.3, linestyle='--')
        ax_psth.spines['top'].set_visible(False)
        ax_psth.spines['right'].set_visible(False)
        ax_psth.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
        
        # Add stimulus onset line
        ax_psth.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save figure with high quality
        output_file = output_folder / f"shank{shank}_unit{unit_id:03d}_{quality}_static_grating.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"  Saved: {output_file}")
    
    print(f"\nAll figures saved to: {output_folder}")
    return output_folder


def plot_orientation_tuning_summary(npz_file, output_folder=None, units_to_plot=None):
    """
    Create summary plots showing orientation tuning across all phases and spatial frequencies.
    This creates a simplified view focusing on orientation selectivity.
    """
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    units_data = data['units_data']
    
    # Get unique orientations
    unique_orientation = data['unique_orientation']
    n_orientations = len(unique_orientation)
    
    # Get unit qualities if available
    unit_qualities = data.get('unit_qualities', None)
    
    # Time windows
    pre_stim_window = float(data['pre_stim_window'])
    post_stim_window = float(data['post_stim_window'])
    
    # Setup output folder
    if output_folder is None:
        output_folder = Path(npz_file).parent / 'orientation_tuning'
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine which units to plot
    if units_to_plot is None:
        units_to_plot = range(len(units_data))
    
    # Create color map for orientations
    colors = plt.cm.hsv(np.linspace(0, 1, n_orientations + 1))[:-1]  # Skip last color to avoid wrap
    ori2color = {ori: colors[i] for i, ori in enumerate(unique_orientation)}
    
    print(f"Creating orientation tuning plots for {len(units_to_plot)} units")
    
    for unit_idx in units_to_plot:
        unit_data = units_data[unit_idx]
        unit_id = unit_data['unit_id']
        shank = unit_data['shank']
        trials = unit_data['trials']

        # Get quality for this unit if available
        quality = unit_qualities[unit_idx] if unit_qualities is not None and unit_idx < len(unit_qualities) else 'unknown'
        
        # Group trials by orientation (averaging across phases and spatial frequencies)
        trials_by_orientation = {ori: [] for ori in unique_orientation}
        
        for trial in trials:
            ori = trial['orientation']
            trials_by_orientation[ori].append(trial)
        
        # Create figure
        plt.style.use('default')
        fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_facecolor('white')
        
        # --- Simplified raster plot by orientation ---
        y_base = 0
        yticks = []
        ylabels = []
        
        for ori in unique_orientation:
            ori_trials = trials_by_orientation[ori]
            color = ori2color[ori]
            
            for trial_i, trial in enumerate(ori_trials):
                y_pos = y_base + trial_i + 0.5
                spike_times = trial['spike_times'] * 1000
                
                if len(spike_times) > 0:
                    ax_raster.scatter(spike_times, np.full_like(spike_times, y_pos),
                                    s=8, color=color, marker='|', 
                                    alpha=0.8, linewidth=1.5)
            
            if len(ori_trials) > 0:
                yticks.append(y_base + len(ori_trials)/2)
                ylabels.append(f"{ori:.0f}°")
                y_base += len(ori_trials)
        
        ax_raster.set_ylim(0, y_base)
        ax_raster.set_yticks(yticks)
        ax_raster.set_yticklabels(ylabels, fontsize=11)
        ax_raster.set_ylabel('Orientation', fontsize=12, fontweight='bold')
        ax_raster.set_title(f"Unit {unit_id} (Shank {shank}) — Quality: {quality} - Orientation Tuning", 
                           fontsize=14, fontweight='bold', pad=20)
        ax_raster.grid(True, alpha=0.3, linestyle='--')
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)
        ax_raster.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
        ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # --- PSTH by orientation ---
        bin_width = 0.010
        bin_edges = np.arange(-pre_stim_window, post_stim_window + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + bin_width/2
        sigma_bins = 20 / (bin_width * 1000)
        
        for ori in unique_orientation:
            ori_trials = trials_by_orientation[ori]
            color = ori2color[ori]
            
            all_spikes = []
            for trial in ori_trials:
                all_spikes.extend(trial['spike_times'])
            
            if len(all_spikes) > 0:
                all_spikes = np.array(all_spikes)
                counts, _ = np.histogram(all_spikes, bins=bin_edges)
                rate = counts / (len(ori_trials) * bin_width)
                rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
                
                ax_psth.plot(bin_centers*1000, rate_smooth,
                           label=f"{ori:.0f}°", color=color, 
                           linewidth=2.5, alpha=0.9)
        
        ax_psth.set_xlabel('Time from stimulus onset (ms)', fontsize=12, fontweight='bold')
        ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
        ax_psth.set_title('Orientation Tuning Curves', fontsize=12, fontweight='bold')
        ax_psth.legend(title='Orientation', title_fontsize=9, fontsize=8, 
                      ncol=min(4, n_orientations), loc='upper right')
        ax_psth.grid(True, alpha=0.3, linestyle='--')
        ax_psth.spines['top'].set_visible(False)
        ax_psth.spines['right'].set_visible(False)
        ax_psth.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
        ax_psth.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        output_file = output_folder / f"shank{shank}_unit{unit_id:03d}_{quality}_orientation_tuning.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"  Saved: {output_file}")
    
    print(f"\nOrientation tuning plots saved to: {output_folder}")
    return output_folder


# Example usage
if __name__ == '__main__':
    # Load NPZ file path
    npz_file = Path(input("Enter the path to the static_grating_responses.npz file: ").strip().strip('"'))
    
    if not npz_file.exists():
        print(f"Error: File {npz_file} does not exist!")
    else:
        # Plot full PSTH with all stimulus conditions
        print("\n1. Creating full PSTH plots with all stimulus conditions...")
        plot_static_grating_psth(npz_file)
        
        # Plot simplified orientation tuning
        print("\n2. Creating orientation tuning summary plots...")
        plot_orientation_tuning_summary(npz_file)
        
        print("\nAll plots completed!")