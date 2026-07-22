"""
Single-Trial Population Rasters Sorted by Preferred Orientation

Generate one figure per trial, showing the entire population's response
to that single stimulus presentation. Neurons sorted by preferred orientation.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from grating_utils import load_neural_data, resolve_data_path


# =============================================================================
# PREFERRED ORIENTATION CALCULATION
# =============================================================================

def calculate_preferred_orientations(neural_data, time_window=(0.07, 0.16)):
    """
    Calculate preferred orientation and OSI for each unit.
    
    Returns:
        Dictionary with unit preferences
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start
    
    unit_ids = list(neural_data['spike_data'].keys())
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = sorted(neural_data['trial_info']['unique_orientations'])
    
    print(f"\nCalculating preferred orientations:")
    print(f"  Units: {len(unit_ids)}")
    print(f"  Orientations: {unique_orientations}")
    
    unit_data = {}
    
    for unit_id in unit_ids:
        unit_trials = neural_data['spike_data'][unit_id]
        
        # Calculate mean firing rate per orientation
        mean_rates = []
        for ori in unique_orientations:
            rates = []
            for trial_data in unit_trials:
                if trial_data['orientation'] == ori:
                    spike_times = np.array(trial_data['spike_times'])
                    spikes = np.sum((spike_times >= window_start) & (spike_times < window_end))
                    rates.append(spikes / window_duration)
            mean_rates.append(np.mean(rates) if rates else 0)
        
        mean_rates = np.array(mean_rates)
        
        # Vector sum method for preferred orientation
        theta_rad = 2 * np.deg2rad(unique_orientations)
        complex_sum = np.sum(mean_rates * np.exp(1j * theta_rad))
        osi = np.abs(complex_sum) / (np.sum(mean_rates) + 1e-12)
        preferred_ori = (np.angle(complex_sum) / 2.0) % np.pi
        preferred_ori_deg = np.rad2deg(preferred_ori)
        
        # Find closest actual orientation tested
        closest_ori = unique_orientations[np.argmin(np.abs(np.array(unique_orientations) - preferred_ori_deg))]
        
        unit_data[unit_id] = {
            'preferred_orientation': closest_ori,
            'preferred_orientation_continuous': preferred_ori_deg,
            'osi': osi,
            'mean_rates': mean_rates,
            'max_rate': np.max(mean_rates)
        }
    
    # Sort units by preferred orientation, then by OSI within each group
    units_sorted = sorted(unit_ids, 
                         key=lambda uid: (unit_data[uid]['preferred_orientation'], 
                                         -unit_data[uid]['osi']))
    
    # Count neurons per preference
    print("\nNeurons by preferred orientation:")
    for ori in unique_orientations:
        n = sum(1 for uid in unit_ids if unit_data[uid]['preferred_orientation'] == ori)
        print(f"  {ori}°: {n} units")
    
    return {
        'unit_data': unit_data,
        'units_sorted_by_preference': units_sorted,
        'unique_orientations': unique_orientations
    }


# =============================================================================
# ORGANIZE TRIALS
# =============================================================================

def organize_trials_by_orientation(neural_data):
    """
    Get trial counts for each orientation.
    
    Returns:
        Dict mapping orientation -> number of trials
    """
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = sorted(neural_data['trial_info']['unique_orientations'])
    
    trials_by_orientation = {ori: orientations.count(ori) for ori in unique_orientations}
    
    return trials_by_orientation


# =============================================================================
# SINGLE TRIAL RASTER PLOTTING
# =============================================================================

def plot_single_trial_raster(neural_data, pref_data, stimulus_orientation,
                             trial_number, time_window=(-0.1, 0.3), save_path=None,
                             stamp_info=None):
    """
    Plot population raster for a SINGLE trial.
    Neurons sorted by preferred orientation (grouped), then by OSI within groups.

    Args:
        neural_data: Loaded neural data
        pref_data: Output from calculate_preferred_orientations()
        stimulus_orientation: Orientation being presented
        trial_number: Which trial number (0-indexed within this orientation)
        time_window: Time window for display
        save_path: Path to save figure
        stamp_info: Optional reproducibility stamp text embedded at the figure bottom
    """
    unit_data = pref_data['unit_data']
    units_sorted = pref_data['units_sorted_by_preference']
    unique_orientations = pref_data['unique_orientations']
    
    # Collect spike data for this specific trial
    spike_data_by_neuron = []
    neurons_with_data = []
    
    for unit_id in units_sorted:
        unit_trials = neural_data['spike_data'][unit_id]
        
        # Find the Nth trial of this orientation for this unit
        this_ori_trials = [t for t in unit_trials if t['orientation'] == stimulus_orientation]
        
        if trial_number < len(this_ori_trials):
            trial_data = this_ori_trials[trial_number]
            spike_times = np.array(trial_data['spike_times'])
            
            # Filter to time window (spike times already trial-aligned)
            mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
            spike_times_windowed = spike_times[mask]
            
            spike_data_by_neuron.append({
                'unit_id': unit_id,
                'spike_times': spike_times_windowed,
                'osi': unit_data[unit_id]['osi'],
                'preferred_orientation': unit_data[unit_id]['preferred_orientation']
            })
            neurons_with_data.append(unit_id)
    
    if len(neurons_with_data) == 0:
        print(f"  Warning: No data for trial {trial_number} of {stimulus_orientation}°")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for preferred orientation
    pref_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_orientations)))
    pref_color_map = {ori: pref_colors[i] for i, ori in enumerate(unique_orientations)}
    
    # Track group boundaries
    group_boundaries = []
    current_pref = None
    
    # Plot rasters
    for neuron_idx, data in enumerate(spike_data_by_neuron):
        spike_times = data['spike_times']
        pref_ori = data['preferred_orientation']
        
        # Mark group boundaries
        if pref_ori != current_pref:
            group_boundaries.append(neuron_idx)
            current_pref = pref_ori
        
        if len(spike_times) > 0:
            color = pref_color_map[pref_ori]
            ax.scatter(spike_times, [neuron_idx] * len(spike_times),
                      s=15, c=[color], marker='|', linewidths=1.5, alpha=0.9)
    
    group_boundaries.append(len(neurons_with_data))  # Final boundary
    
    # Add colored background for each preference group
    from matplotlib.patches import Rectangle
    for i in range(len(group_boundaries) - 1):
        y_start = group_boundaries[i]
        y_end = group_boundaries[i + 1]
        
        # Get preference for this group
        pref_ori = spike_data_by_neuron[y_start]['preferred_orientation']
        color = pref_color_map[pref_ori]
        
        # Add background
        ax.add_patch(Rectangle(
            (time_window[0], y_start - 0.5),
            time_window[1] - time_window[0],
            y_end - y_start,
            facecolor=color,
            alpha=0.15,
            zorder=0
        ))
        
        # Add group separator (except last)
        if i < len(group_boundaries) - 2:
            ax.axhline(y_end - 0.5, color='black', linewidth=2, 
                      linestyle='--', alpha=0.6)
        
        # Add group label on right
        n_neurons = y_end - y_start
        ax.text(time_window[1] + 0.01, (y_start + y_end) / 2 - 0.5,
               f'Prefer {pref_ori}°\n({n_neurons} units)',
               va='center', ha='left', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.4))
    
    # Add stimulus onset line
    ax.axvline(0, color='red', linewidth=3, linestyle='-', alpha=0.8, 
              label='Stimulus Onset', zorder=10)
    
    # Styling
    ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Neurons (grouped by preferred orientation)', fontsize=14, fontweight='bold')
    ax.set_title(f'{stimulus_orientation}° Grating - Trial #{trial_number + 1}\n' +
                 f'Population Response (n={len(neurons_with_data)} neurons)',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(time_window)
    ax.set_ylim(-1, len(neurons_with_data))
    ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=1)
    
    # Add summary text
    total_spikes = sum(len(d['spike_times']) for d in spike_data_by_neuron)
    
    summary_text = f"Stimulus: {stimulus_orientation}°\n"
    summary_text += f"Trial: {trial_number + 1}\n"
    summary_text += f"Neurons: {len(neurons_with_data)}\n"
    summary_text += f"Total spikes: {total_spikes}"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=11, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    if stamp_info:
        fig.text(0.01, 0.003, stamp_info, ha='left', va='bottom', fontsize=6, color='0.4')

    plt.tight_layout(rect=(0, 0.02, 1, 1))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def generate_single_trial_rasters(data_path, time_window=(-0.1, 0.3),
                                  preference_window=(0.07, 0.16), 
                                  output_folder=None,
                                  max_trials_per_orientation=None):
    """
    Generate one figure per trial for each orientation.
    
    Args:
        data_path: Path to neural data pickle file
        time_window: Time window for raster display
        preference_window: Time window for calculating preferred orientation
        output_folder: Folder to save plots
        max_trials_per_orientation: Limit number of trials to plot (None = all)
    
    Returns:
        Dictionary with preference data
    """
    # Load data
    data = load_neural_data(data_path)

    # Calculate preferred orientations
    pref_data = calculate_preferred_orientations(data, time_window=preference_window)
    unique_orientations = pref_data['unique_orientations']

    # Get trial counts
    trials_by_ori = organize_trials_by_orientation(data)

    # One shared reproducibility stamp for every figure from this run
    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stamp_info = (
        f"Generated {run_timestamp} | script=singleTrialRaster.py | data={Path(data_path).name}\n"
        f"display_window=[{time_window[0]}, {time_window[1]}]s, "
        f"preference_window=[{preference_window[0]}, {preference_window[1]}]s"
    )

    # Set up output folder
    if output_folder is None:
        output_folder = Path(data_path).parent / f"{Path(data_path).stem}_single_trial_rasters"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving single-trial rasters to: {output_folder}")
    
    # Generate figures for each orientation
    total_figures = 0
    
    for ori in unique_orientations:
        print(f"\n{'='*60}")
        print(f"Processing {ori}° orientation...")
        print(f"{'='*60}")
        
        # Create subfolder for this orientation
        ori_folder = output_folder / f"orientation_{int(ori)}deg"
        ori_folder.mkdir(parents=True, exist_ok=True)
        
        n_trials = trials_by_ori[ori]
        
        # Limit trials if requested
        if max_trials_per_orientation is not None:
            n_trials = min(n_trials, max_trials_per_orientation)
            print(f"  Plotting {n_trials}/{trials_by_ori[ori]} trials (limited)")
        else:
            print(f"  Plotting all {n_trials} trials")
        
        # Generate one figure per trial
        for trial_num in range(n_trials):
            save_path = ori_folder / f"trial_{trial_num + 1:03d}.png"
            
            plot_single_trial_raster(
                data, pref_data, ori, trial_num,
                time_window=time_window,
                save_path=save_path,
                stamp_info=stamp_info
            )
            
            total_figures += 1
            
            if (trial_num + 1) % 10 == 0:
                print(f"    Completed {trial_num + 1}/{n_trials} trials...")
        
        print(f"  ✓ Saved {n_trials} trial figures to: {ori_folder}")
    
    print(f"\n{'='*60}")
    print(f"✓ Generated {total_figures} total figures")
    print(f"✓ All rasters saved to: {output_folder}")
    print(f"{'='*60}")
    
    # Save summary
    save_trial_summary(trials_by_ori, output_folder / "trial_summary.txt", 
                      max_trials_per_orientation)
    
    return pref_data


def save_trial_summary(trials_by_ori, save_path, max_trials=None):
    """Save summary of trials processed."""
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SINGLE-TRIAL RASTER SUMMARY\n")
        f.write("Neurons sorted by: Preferred Orientation\n")
        f.write("="*70 + "\n\n")
        
        total_trials = 0
        total_plotted = 0
        
        for ori in sorted(trials_by_ori.keys()):
            n_trials = trials_by_ori[ori]
            n_plotted = n_trials if max_trials is None else min(n_trials, max_trials)
            
            total_trials += n_trials
            total_plotted += n_plotted
            
            f.write(f"Orientation {ori}°:\n")
            f.write(f"  Total trials available: {n_trials}\n")
            f.write(f"  Trials plotted: {n_plotted}\n")
            f.write(f"  Folder: orientation_{int(ori)}deg/\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Total trials available: {total_trials}\n")
        f.write(f"Total figures generated: {total_plotted}\n")
        
        if max_trials is not None:
            f.write(f"\nNote: Limited to {max_trials} trials per orientation\n")
    
    print(f"✓ Saved trial summary to: {save_path}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = resolve_data_path()

    # Optional: customize output folder
    OUTPUT_FOLDER = None  # Set to None to use default
    
    # Optional: limit number of trials per orientation (None = plot all)
    # Useful if you have many trials - set to a number like 20
    MAX_TRIALS_PER_ORI = None
    
    try:
        pref_data = generate_single_trial_rasters(
            data_path=DATA_PATH,
            time_window=(-0.5, 2),  # Display window for raster
            preference_window=(0.07, 0.16),  # Window for calculating preference
            output_folder=OUTPUT_FOLDER,
            max_trials_per_orientation=MAX_TRIALS_PER_ORI
        )
        
        print("\n" + "="*60)
        print("Single-trial raster generation complete!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise