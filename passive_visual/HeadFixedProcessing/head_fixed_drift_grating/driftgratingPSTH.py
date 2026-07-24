import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from itertools import product

def plot_drifting_grating_psth(npz_file, output_folder=None, units_to_plot=None, 
                                group_by='orientation_tf', separate_plots=False):
    """
    Plot PSTH for drifting grating responses with temporal frequency.
    
    Parameters:
    -----------
    npz_file : Path or str
        Path to the NPZ file containing drifting grating responses
    output_folder : Path or str, optional
        Folder to save figures. If None, saves to npz_file's parent directory
    units_to_plot : list, optional
        List of unit indices to plot. If None, plots all units
    group_by : str
        How to group stimuli in visualization:
        - 'orientation_tf': Group by orientation and temporal frequency (default)
        - 'all': Show all stimulus combinations
        - 'orientation': Group by orientation only
        - 'temporal_freq': Group by temporal frequency only
    separate_plots : bool
        If True, create separate plots for each temporal frequency
    """
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    units_data = data['units_data']
    
    # Get stimulus parameters
    unique_orientation = data['unique_orientation']
    unique_phase = data['unique_phase']
    unique_spatialFreq = data['unique_spatialFreq']
    unique_temporalFreq = data.get('unique_temporalFreq', np.array([0]))  # Default if missing
    
    # Get unit qualities if available
    unit_qualities = data.get('unit_qualities', None)
    
    # Time windows
    pre_stim_window = float(data['pre_stim_window'])
    post_stim_window = float(data['post_stim_window'])
    post_stim_window = min(post_stim_window, 1.0)  # Cap at 1s for visualization
    
    # Setup output folder
    if output_folder is None:
        output_folder = Path(npz_file).parent / 'drifting_grating_psth'
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine which units to plot
    if units_to_plot is None:
        units_to_plot = range(len(units_data))
    
    print(f"Processing {len(units_to_plot)} units")
    print(f"Stimulus parameters:")
    print(f"  Orientations: {unique_orientation}")
    print(f"  Temporal frequencies: {unique_temporalFreq}")
    print(f"  Spatial frequencies: {unique_spatialFreq}")
    print(f"  Phases: {unique_phase}")
    
    # Process each unit
    for unit_idx in units_to_plot:
        unit_data = units_data[unit_idx]
        unit_id = unit_data['unit_id']
        shank = unit_data['shank']
        trials = unit_data['trials']
        
        # Get quality for this unit if available
        quality = unit_qualities[unit_idx] if unit_qualities is not None and unit_idx < len(unit_qualities) else 'unknown'
        
        print(f"\nProcessing shank{shank}_unit{unit_id} (quality: {quality})")
        
        if separate_plots:
            # Create separate plot for each temporal frequency
            for tf in unique_temporalFreq:
                create_tf_specific_plot(trials, unit_id, shank, quality, tf,
                                       unique_orientation, unique_phase, unique_spatialFreq,
                                       pre_stim_window, post_stim_window, output_folder)
        else:
            # Create combined plot based on grouping strategy
            create_combined_plot(trials, unit_id, shank, quality,
                               unique_orientation, unique_phase, unique_spatialFreq, unique_temporalFreq,
                               pre_stim_window, post_stim_window, output_folder, group_by)
    
    print(f"\nAll figures saved to: {output_folder}")
    return output_folder


def create_combined_plot(trials, unit_id, shank, quality,
                        unique_orientation, unique_phase, unique_spatialFreq, unique_temporalFreq,
                        pre_stim_window, post_stim_window, output_folder, group_by):
    """Create a combined plot based on grouping strategy."""
    
    # Determine grouping and create color maps
    if group_by == 'orientation_tf':
        # Group by orientation and temporal frequency combinations
        stim_combinations = list(product(unique_orientation, unique_temporalFreq))
        n_groups = len(stim_combinations)
        colors = plt.cm.viridis(np.linspace(0, 1, n_groups))
        group2color = {combo: colors[i] for i, combo in enumerate(stim_combinations)}
        
    elif group_by == 'all':
        # Show all possible combinations
        stim_combinations = list(product(unique_orientation, unique_phase, 
                                       unique_spatialFreq, unique_temporalFreq))
        n_groups = len(stim_combinations)
        colors = plt.cm.viridis(np.linspace(0, 1, n_groups))
        group2color = {combo: colors[i] for i, combo in enumerate(stim_combinations)}
        
    elif group_by == 'orientation':
        # Group by orientation only
        stim_combinations = unique_orientation
        n_groups = len(stim_combinations)
        colors = plt.cm.hsv(np.linspace(0, 1, n_groups + 1))[:-1]
        group2color = {ori: colors[i] for i, ori in enumerate(stim_combinations)}
        
    elif group_by == 'temporal_freq':
        # Group by temporal frequency only
        stim_combinations = unique_temporalFreq
        n_groups = len(stim_combinations)
        colors = plt.cm.plasma(np.linspace(0, 1, n_groups))
        group2color = {tf: colors[i] for i, tf in enumerate(stim_combinations)}
    
    # Organize trials by grouping
    trials_by_group = {}
    for trial in trials:
        if group_by == 'orientation_tf':
            key = (trial['orientation'], trial.get('temporal_frequency', 0))
        elif group_by == 'all':
            key = (trial['orientation'], trial['phase'], 
                  trial['spatial_frequency'], trial.get('temporal_frequency', 0))
        elif group_by == 'orientation':
            key = trial['orientation']
        elif group_by == 'temporal_freq':
            key = trial.get('temporal_frequency', 0)
        
        if key not in trials_by_group:
            trials_by_group[key] = []
        trials_by_group[key].append(trial)
    
    # Create figure
    plt.style.use('default')
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('white')
    
    # --- Raster plot ---
    y_base = 0
    yticks = []
    ylabels = []
    
    for group_key, group_trials in sorted(trials_by_group.items()):
        if len(group_trials) == 0:
            continue
        
        color = group2color[group_key]
        
        # Create label based on grouping
        if group_by == 'orientation_tf':
            ori, tf = group_key
            label = f"O:{ori:.0f}° TF:{tf:.1f}Hz"
        elif group_by == 'all':
            ori, phase, sf, tf = group_key
            label = f"O:{ori:.0f}° P:{phase:.0f}° SF:{sf:.2f} TF:{tf:.1f}"
        elif group_by == 'orientation':
            label = f"{group_key:.0f}°"
        elif group_by == 'temporal_freq':
            label = f"{group_key:.1f}Hz"
        
        # Plot spikes for each trial
        for trial_i, trial in enumerate(group_trials):
            y_pos = y_base + trial_i + 0.5
            spike_times = trial['spike_times'] * 1000  # Convert to ms
            
            if len(spike_times) > 0:
                ax_raster.scatter(spike_times, np.full_like(spike_times, y_pos),
                                s=8, color=color, marker='|', 
                                alpha=0.8, linewidth=1.5)
        
        yticks.append(y_base + len(group_trials)/2)
        ylabels.append(label)
        y_base += len(group_trials)
    
    # Configure raster plot
    ax_raster.set_ylim(0, y_base)
    ax_raster.set_yticks(yticks)
    ax_raster.set_yticklabels(ylabels, fontsize=9)
    ax_raster.set_ylabel('Stimulus Conditions', fontsize=12, fontweight='bold')
    ax_raster.set_title(f"Unit {unit_id} (Shank {shank}) — Quality: {quality} - Drifting Grating Responses", 
                       fontsize=14, fontweight='bold', pad=20)
    ax_raster.grid(True, alpha=0.3, linestyle='--')
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
    ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # --- PSTH with smoothing ---
    bin_width = 0.010  # 10ms bins
    bin_edges = np.arange(-pre_stim_window, post_stim_window + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width/2
    
    # Gaussian smoothing parameters
    sigma_ms = 20  # smoothing width in ms
    sigma_bins = sigma_ms / (bin_width * 1000)
    
    # Plot PSTH for each group
    for group_key, group_trials in sorted(trials_by_group.items()):
        if len(group_trials) == 0:
            continue
        
        color = group2color[group_key]
        
        # Create label for legend
        if group_by == 'orientation_tf':
            ori, tf = group_key
            label = f"O:{ori:.0f}° TF:{tf:.1f}Hz"
        elif group_by == 'all':
            ori, phase, sf, tf = group_key
            label = f"O:{ori:.0f}° TF:{tf:.1f}"
        elif group_by == 'orientation':
            label = f"{group_key:.0f}°"
        elif group_by == 'temporal_freq':
            label = f"{group_key:.1f}Hz"
        
        # Collect all spikes for this group
        all_spikes = []
        for trial in group_trials:
            all_spikes.extend(trial['spike_times'])
        
        if len(all_spikes) > 0:
            all_spikes = np.array(all_spikes)
            counts, _ = np.histogram(all_spikes, bins=bin_edges)
            
            # Calculate firing rate in Hz
            n_trials = len(group_trials)
            rate = counts / (n_trials * bin_width)
            
            # Apply Gaussian smoothing
            rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
            
            ax_psth.plot(bin_centers*1000, rate_smooth,
                       label=label, color=color, 
                       linewidth=2.5, alpha=0.9)
    
    # Configure PSTH plot
    ax_psth.set_xlabel('Time from stimulus onset (ms)', fontsize=12, fontweight='bold')
    ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
    ax_psth.set_title('Peri-Stimulus Time Histogram (smoothed)', fontsize=12, fontweight='bold')
    
    # Add legend with adaptive columns
    n_cols = min(3, (n_groups + 2) // 3)
    if len(ax_psth.get_lines()) > 0:
        ax_psth.legend(title='Stimulus', title_fontsize=9, fontsize=8, 
                      ncol=n_cols, loc='upper right', 
                      frameon=True, fancybox=True, shadow=True, 
                      bbox_to_anchor=(0.98, 0.98))
    
    ax_psth.grid(True, alpha=0.3, linestyle='--')
    ax_psth.spines['top'].set_visible(False)
    ax_psth.spines['right'].set_visible(False)
    ax_psth.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
    ax_psth.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save figure
    output_file = output_folder / f"shank{shank}_unit{unit_id:03d}_{quality}_drifting_{group_by}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: {output_file}")


def create_tf_specific_plot(trials, unit_id, shank, quality, target_tf,
                           unique_orientation, unique_phase, unique_spatialFreq,
                           pre_stim_window, post_stim_window, output_folder):
    """Create a plot for a specific temporal frequency."""
    
    # Filter trials for this temporal frequency
    tf_trials = [t for t in trials if t.get('temporal_frequency', 0) == target_tf]
    
    if len(tf_trials) == 0:
        print(f"  No trials for TF={target_tf:.1f}Hz, skipping")
        return
    
    # Group by orientation for this TF
    trials_by_ori = {}
    for trial in tf_trials:
        ori = trial['orientation']
        if ori not in trials_by_ori:
            trials_by_ori[ori] = []
        trials_by_ori[ori].append(trial)
    
    # Create color map for orientations
    n_orientations = len(unique_orientation)
    colors = plt.cm.hsv(np.linspace(0, 1, n_orientations + 1))[:-1]
    ori2color = {ori: colors[i] for i, ori in enumerate(unique_orientation)}
    
    # Create figure
    plt.style.use('default')
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('white')
    
    # --- Raster plot ---
    y_base = 0
    yticks = []
    ylabels = []
    
    for ori in unique_orientation:
        if ori not in trials_by_ori:
            continue
        
        ori_trials = trials_by_ori[ori]
        color = ori2color[ori]
        
        for trial_i, trial in enumerate(ori_trials):
            y_pos = y_base + trial_i + 0.5
            spike_times = trial['spike_times'] * 1000
            
            if len(spike_times) > 0:
                ax_raster.scatter(spike_times, np.full_like(spike_times, y_pos),
                                s=8, color=color, marker='|', 
                                alpha=0.8, linewidth=1.5)
        
        yticks.append(y_base + len(ori_trials)/2)
        ylabels.append(f"{ori:.0f}°")
        y_base += len(ori_trials)
    
    ax_raster.set_ylim(0, y_base)
    ax_raster.set_yticks(yticks)
    ax_raster.set_yticklabels(ylabels, fontsize=11)
    ax_raster.set_ylabel('Orientation', fontsize=12, fontweight='bold')
    ax_raster.set_title(f"Unit {unit_id} (Shank {shank}) — Quality: {quality} - TF: {target_tf:.1f}Hz", 
                       fontsize=14, fontweight='bold', pad=20)
    ax_raster.grid(True, alpha=0.3, linestyle='--')
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
    ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # --- PSTH ---
    bin_width = 0.010
    bin_edges = np.arange(-pre_stim_window, post_stim_window + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width/2
    sigma_bins = 20 / (bin_width * 1000)
    
    for ori in unique_orientation:
        if ori not in trials_by_ori:
            continue
        
        ori_trials = trials_by_ori[ori]
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
    ax_psth.set_title(f'Orientation Tuning at TF={target_tf:.1f}Hz', fontsize=12, fontweight='bold')
    ax_psth.legend(title='Orientation', title_fontsize=9, fontsize=8, 
                  ncol=min(4, n_orientations), loc='upper right')
    ax_psth.grid(True, alpha=0.3, linestyle='--')
    ax_psth.spines['top'].set_visible(False)
    ax_psth.spines['right'].set_visible(False)
    ax_psth.set_xlim(-pre_stim_window*1000, post_stim_window*1000)
    ax_psth.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    output_file = output_folder / f"shank{shank}_unit{unit_id:03d}_{quality}_TF{target_tf:.1f}Hz.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: {output_file}")


def plot_temporal_frequency_comparison(npz_file, output_folder=None, units_to_plot=None):
    """
    Create a comparison plot showing how orientation tuning changes with temporal frequency.
    Creates a multi-panel figure with one panel per temporal frequency.
    """
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    units_data = data['units_data']
    
    # Get parameters
    unique_orientation = data['unique_orientation']
    unique_temporalFreq = data.get('unique_temporalFreq', np.array([0]))
    unit_qualities = data.get('unit_qualities', None)
    
    pre_stim_window = float(data['pre_stim_window'])
    post_stim_window = float(data['post_stim_window'])
    
    # Setup output folder
    if output_folder is None:
        output_folder = Path(npz_file).parent / 'tf_comparison'
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine which units to plot
    if units_to_plot is None:
        units_to_plot = range(len(units_data))
    
    n_tfs = len(unique_temporalFreq)
    
    for unit_idx in units_to_plot:
        unit_data = units_data[unit_idx]
        unit_id = unit_data['unit_id']
        shank = unit_data['shank']
        trials = unit_data['trials']
        
        quality = unit_qualities[unit_idx] if unit_qualities is not None and unit_idx < len(unit_qualities) else 'unknown'
        
        # Create multi-panel figure
        fig, axes = plt.subplots(1, n_tfs, figsize=(6*n_tfs, 5), sharey=True)
        if n_tfs == 1:
            axes = [axes]
        
        fig.suptitle(f"Unit {unit_id} (Shank {shank}) — Quality: {quality} - TF Comparison", 
                    fontsize=14, fontweight='bold')
        
        # Color map for orientations
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_orientation) + 1))[:-1]
        ori2color = {ori: colors[i] for i, ori in enumerate(unique_orientation)}
        
        # Process each temporal frequency
        for tf_idx, tf in enumerate(unique_temporalFreq):
            ax = axes[tf_idx]
            
            # Filter trials for this TF
            tf_trials = [t for t in trials if t.get('temporal_frequency', 0) == tf]
            
            # Group by orientation
            trials_by_ori = {}
            for trial in tf_trials:
                ori = trial['orientation']
                if ori not in trials_by_ori:
                    trials_by_ori[ori] = []
                trials_by_ori[ori].append(trial)
            
            # Calculate mean firing rates for each orientation
            ori_rates = []
            ori_sems = []
            
            for ori in unique_orientation:
                if ori in trials_by_ori:
                    ori_trials = trials_by_ori[ori]
                    trial_rates = [t['firing_rate_post'] for t in ori_trials]
                    ori_rates.append(np.mean(trial_rates))
                    ori_sems.append(np.std(trial_rates) / np.sqrt(len(trial_rates)))
                else:
                    ori_rates.append(0)
                    ori_sems.append(0)
            
            # Plot tuning curve
            ax.errorbar(unique_orientation, ori_rates, yerr=ori_sems,
                       marker='o', markersize=8, linewidth=2.5,
                       capsize=5, capthick=2)
            
            ax.set_xlabel('Orientation (°)', fontsize=10, fontweight='bold')
            if tf_idx == 0:
                ax.set_ylabel('Firing Rate (Hz)', fontsize=10, fontweight='bold')
            ax.set_title(f'TF = {tf:.1f} Hz', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        output_file = output_folder / f"shank{shank}_unit{unit_id:03d}_{quality}_tf_comparison.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"  Saved: {output_file}")
    
    print(f"\nTF comparison plots saved to: {output_folder}")
    return output_folder


# Example usage
if __name__ == '__main__':
    # Load NPZ file path
    npz_file = Path(input("Enter the path to the drifting_grating_responses.npz file: ").strip().strip('"'))
    
    if not npz_file.exists():
        print(f"Error: File {npz_file} does not exist!")
    else:
        print("\n1. Creating combined PSTH plots (orientation + TF)...")
        plot_drifting_grating_psth(npz_file, group_by='orientation_tf')
        
        # print("\n2. Creating separate plots for each temporal frequency...")
        # plot_drifting_grating_psth(npz_file, separate_plots=True)
        
        print("\n3. Creating temporal frequency comparison plots...")
        plot_temporal_frequency_comparison(npz_file)
        
        print("\n4. Creating orientation-only grouped plots...")
        plot_drifting_grating_psth(npz_file, group_by='orientation')
        
        print("\nAll plots completed!")