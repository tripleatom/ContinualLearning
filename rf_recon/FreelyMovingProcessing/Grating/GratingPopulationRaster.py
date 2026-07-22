"""Population Raster Plots Grouped by Preferred Orientation"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from grating_utils import resolve_data_path


def load_neural_data(filepath):
    """Load neural data from pickle format."""
    filepath = Path(filepath)
    print(f"Loading data: {filepath.name}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_preferred_orientations(neural_data, time_window=(0.07, 0.16), osi_threshold=None):
    """Calculate preferred orientation for each unit using vector sum method."""
    window_start, window_end = time_window
    window_duration = window_end - window_start

    unit_ids = list(neural_data['spike_data'].keys())
    unique_orientations = sorted(neural_data['trial_info']['unique_orientations'])

    print(f"Calculating orientations: {len(unit_ids)} units, window {window_start:.2f}-{window_end:.2f}s")

    unit_preferences = {}
    
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

        # Find closest actual orientation tested (with circular distance for 0/180)
        # Normalize test orientations to [0, 180) range
        test_orientations = np.array(unique_orientations) % 180

        # Calculate circular distance for orientation (180° periodic)
        angular_diffs = np.abs(test_orientations - preferred_ori_deg)
        # Handle wraparound: if difference > 90°, use 180 - difference
        angular_diffs = np.minimum(angular_diffs, 180 - angular_diffs)

        # Find orientation with minimum circular distance
        closest_idx = np.argmin(angular_diffs)
        closest_ori = unique_orientations[closest_idx]
        
        unit_preferences[unit_id] = {
            'preferred_orientation': closest_ori,
            'preferred_orientation_continuous': preferred_ori_deg,
            'osi': osi,
            'mean_rates': mean_rates,
            'max_rate': np.max(mean_rates),
            'baseline_rate': np.mean(mean_rates)
        }
    
    # Filter units by OSI threshold
    if osi_threshold is not None:
        n_before = len(unit_preferences)
        unit_preferences = {uid: pref for uid, pref in unit_preferences.items()
                           if pref['osi'] >= osi_threshold}
        print(f"OSI filter ({osi_threshold}): {n_before} → {len(unit_preferences)} units")

    # Group and sort units by preferred orientation
    orientation_groups = {ori: [] for ori in unique_orientations}
    for unit_id, pref_data in unit_preferences.items():
        orientation_groups[pref_data['preferred_orientation']].append(unit_id)

    for ori in orientation_groups:
        orientation_groups[ori].sort(key=lambda uid: unit_preferences[uid]['osi'], reverse=True)

    print(f"Distribution: {', '.join(f'{ori}°:{len(orientation_groups[ori])}' for ori in unique_orientations)}")
    
    return {
        'unit_preferences': unit_preferences,
        'orientation_groups': orientation_groups,
        'unique_orientations': unique_orientations
    }


# =============================================================================
# POPULATION RASTER PLOTTING
# =============================================================================

def plot_population_raster(neural_data, preference_data, stimulus_orientation,
                          time_window=(-0.1, 0.3), save_path=None):
    """
    Plot population raster for a specific stimulus orientation,
    with neurons grouped by their preferred orientation.
    
    Args:
        neural_data: Loaded neural data
        preference_data: Output from calculate_preferred_orientations()
        stimulus_orientation: Which orientation stimulus to plot
        time_window: Time window relative to stimulus onset (start, end) in seconds
        save_path: Path to save figure
    """
    unit_preferences = preference_data['unit_preferences']
    orientation_groups = preference_data['orientation_groups']
    unique_orientations = preference_data['unique_orientations']
    
    print(f"Plotting {stimulus_orientation}°...")

    # Create ordered list of units
    ordered_units = []
    for pref_ori in unique_orientations:
        ordered_units.extend(orientation_groups[pref_ori])

    # Collect spike times
    all_spike_data = []
    neuron_trial_counts = []
    
    for neuron_idx, unit_id in enumerate(ordered_units):
        unit_trials = neural_data['spike_data'][unit_id]
        trial_count = 0

        for trial_data in unit_trials:
            if trial_data['orientation'] == stimulus_orientation:
                spike_times = np.array(trial_data['spike_times'])
                mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                spike_times_windowed = spike_times[mask]

                all_spike_data.append({
                    'neuron_idx': neuron_idx,
                    'trial_idx': trial_count,
                    'spike_times': spike_times_windowed
                })
                trial_count += 1

        neuron_trial_counts.append(trial_count)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(12, len(ordered_units) * 0.3)))

    # Color map for preference groups
    group_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_orientations)))
    pref_color_map = {ori: group_colors[i] for i, ori in enumerate(unique_orientations)}

    # Plot rasters
    for spike_data in all_spike_data:
        y_pos = sum(neuron_trial_counts[:spike_data['neuron_idx']]) + spike_data['trial_idx']
        if len(spike_data['spike_times']) > 0:
            ax.scatter(spike_data['spike_times'], [y_pos] * len(spike_data['spike_times']),
                      s=2, c='black', marker='|', linewidths=0.5)
    
    # Add background colors and labels for preference groups
    neuron_y_positions = [0]
    for tc in neuron_trial_counts:
        neuron_y_positions.append(neuron_y_positions[-1] + tc)

    current_neuron_idx = 0
    for pref_ori in unique_orientations:
        units = orientation_groups[pref_ori]
        if len(units) == 0:
            continue

        y_start = neuron_y_positions[current_neuron_idx]
        y_end = neuron_y_positions[current_neuron_idx + len(units)]

        ax.add_patch(Rectangle((time_window[0], y_start),
                               time_window[1] - time_window[0], y_end - y_start,
                               facecolor=pref_color_map[pref_ori], alpha=0.15, zorder=0))

        ax.text(time_window[1] + 0.01, (y_start + y_end) / 2,
               f'Prefer {pref_ori}°\n({len(units)} units)',
               va='center', ha='left', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=pref_color_map[pref_ori], alpha=0.3))

        current_neuron_idx += len(units)
    
    # Styling
    ax.axvline(0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=12)
    ax.set_ylabel('Trials (grouped by neuron)', fontsize=12)
    ax.set_title(f'{stimulus_orientation}° Grating - Neurons Grouped by Preferred Orientation',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(time_window)
    ax.set_ylim(-0.5, neuron_y_positions[-1] - 0.5)
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {Path(save_path).name}")
        plt.close(fig)

    return fig


def plot_population_raster_with_psth(neural_data, preference_data, stimulus_orientation,
                                    time_window=(-0.1, 0.3), bin_size=0.01, save_path=None):
    """Plot raster with PSTH for each preference group."""
    unit_preferences = preference_data['unit_preferences']
    orientation_groups = preference_data['orientation_groups']
    unique_orientations = preference_data['unique_orientations']

    print(f"Plotting {stimulus_orientation}° with PSTH...")

    ordered_units = []
    for pref_ori in unique_orientations:
        ordered_units.extend(orientation_groups[pref_ori])

    time_bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    psth_by_group = {ori: np.zeros(len(time_bins) - 1) for ori in unique_orientations}
    trial_counts_by_group = {ori: 0 for ori in unique_orientations}

    all_spike_data = []
    neuron_trial_counts = []

    for neuron_idx, unit_id in enumerate(ordered_units):
        unit_trials = neural_data['spike_data'][unit_id]
        pref_ori = unit_preferences[unit_id]['preferred_orientation']
        trial_count = 0

        for trial_data in unit_trials:
            if trial_data['orientation'] == stimulus_orientation:
                spike_times = np.array(trial_data['spike_times'])
                mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                spike_times_windowed = spike_times[mask]

                hist, _ = np.histogram(spike_times_windowed, bins=time_bins)
                psth_by_group[pref_ori] += hist
                trial_counts_by_group[pref_ori] += 1

                all_spike_data.append({
                    'neuron_idx': neuron_idx,
                    'trial_idx': trial_count,
                    'spike_times': spike_times_windowed
                })
                trial_count += 1

        neuron_trial_counts.append(trial_count)
    
    # Normalize PSTHs
    for ori in unique_orientations:
        if trial_counts_by_group[ori] > 0:
            psth_by_group[ori] /= (trial_counts_by_group[ori] * bin_size)

    # Create figure
    fig = plt.figure(figsize=(16, 16))
    ax_psth = plt.subplot2grid((4, 1), (0, 0))

    group_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_orientations)))
    pref_color_map = {ori: group_colors[i] for i, ori in enumerate(unique_orientations)}
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2

    for ori in unique_orientations:
        if len(orientation_groups[ori]) > 0:
            ax_psth.plot(time_centers, psth_by_group[ori], linewidth=2,
                        label=f'Prefer {ori}°', color=pref_color_map[ori])
            ax_psth.fill_between(time_centers, 0, psth_by_group[ori],
                                alpha=0.3, color=pref_color_map[ori])

    ax_psth.axvline(0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax_psth.set_xlabel('Time from Stimulus Onset (s)', fontsize=12)
    ax_psth.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax_psth.set_title(f'Population PSTH - {stimulus_orientation}° Stimulus', fontsize=14, fontweight='bold')
    ax_psth.legend(loc='upper right', fontsize=10)
    ax_psth.grid(True, alpha=0.3)
    ax_psth.set_xlim(time_window)
    
    # Raster panel
    ax_raster = plt.subplot2grid((4, 1), (1, 0), rowspan=3)

    neuron_y_positions = [0]
    for tc in neuron_trial_counts:
        neuron_y_positions.append(neuron_y_positions[-1] + tc)

    for spike_data in all_spike_data:
        y_pos = sum(neuron_trial_counts[:spike_data['neuron_idx']]) + spike_data['trial_idx']
        if len(spike_data['spike_times']) > 0:
            ax_raster.scatter(spike_data['spike_times'], [y_pos] * len(spike_data['spike_times']),
                            s=2, c='black', marker='|', linewidths=0.5)

    current_neuron_idx = 0
    for pref_ori in unique_orientations:
        units = orientation_groups[pref_ori]
        if len(units) == 0:
            continue

        y_start = neuron_y_positions[current_neuron_idx]
        y_end = neuron_y_positions[current_neuron_idx + len(units)]

        ax_raster.add_patch(Rectangle((time_window[0], y_start),
                                     time_window[1] - time_window[0], y_end - y_start,
                                     facecolor=pref_color_map[pref_ori], alpha=0.15, zorder=0))

        ax_raster.text(time_window[1] + 0.01, (y_start + y_end) / 2,
                      f'Prefer {pref_ori}°\n({len(units)} units)',
                      va='center', ha='left', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor=pref_color_map[pref_ori], alpha=0.3))

        current_neuron_idx += len(units)

    ax_raster.axvline(0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax_raster.set_xlabel('Time from Stimulus Onset (s)', fontsize=12)
    ax_raster.set_ylabel('Trials (grouped by neuron)', fontsize=12)
    ax_raster.set_xlim(time_window)
    ax_raster.set_ylim(-0.5, neuron_y_positions[-1] - 0.5)
    ax_raster.grid(True, axis='x', alpha=0.3, linestyle=':')

    plt.suptitle(f'{stimulus_orientation}° Grating', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {Path(save_path).name}")
        plt.close(fig)

    return fig


def plot_all_orientations_summary(neural_data, preference_data, time_window=(-0.1, 0.3), save_path=None):
    """Plot all orientations horizontally for comparison."""
    orientation_groups = preference_data['orientation_groups']
    unique_orientations = preference_data['unique_orientations']

    print("Creating summary figure with all orientations...")

    # Create ordered list of units
    ordered_units = []
    for pref_ori in unique_orientations:
        ordered_units.extend(orientation_groups[pref_ori])

    n_orientations = len(unique_orientations)
    fig, axes = plt.subplots(1, n_orientations, figsize=(5 * n_orientations, max(12, len(ordered_units) * 0.3)),
                            sharey=True)

    if n_orientations == 1:
        axes = [axes]

    # Color map
    group_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_orientations)))
    pref_color_map = {ori: group_colors[i] for i, ori in enumerate(unique_orientations)}

    for ori_idx, stimulus_ori in enumerate(unique_orientations):
        ax = axes[ori_idx]

        # Collect spike data
        all_spike_data = []
        neuron_trial_counts = []

        for neuron_idx, unit_id in enumerate(ordered_units):
            unit_trials = neural_data['spike_data'][unit_id]
            trial_count = 0

            for trial_data in unit_trials:
                if trial_data['orientation'] == stimulus_ori:
                    spike_times = np.array(trial_data['spike_times'])
                    mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                    spike_times_windowed = spike_times[mask]

                    all_spike_data.append({
                        'neuron_idx': neuron_idx,
                        'trial_idx': trial_count,
                        'spike_times': spike_times_windowed
                    })
                    trial_count += 1

            neuron_trial_counts.append(trial_count)

        # Plot rasters
        for spike_data in all_spike_data:
            y_pos = sum(neuron_trial_counts[:spike_data['neuron_idx']]) + spike_data['trial_idx']
            if len(spike_data['spike_times']) > 0:
                ax.scatter(spike_data['spike_times'], [y_pos] * len(spike_data['spike_times']),
                          s=1, c='black', marker='|', linewidths=0.3)

        # Add background colors
        neuron_y_positions = [0]
        for tc in neuron_trial_counts:
            neuron_y_positions.append(neuron_y_positions[-1] + tc)

        current_neuron_idx = 0
        for pref_ori in unique_orientations:
            units = orientation_groups[pref_ori]
            if len(units) == 0:
                continue

            y_start = neuron_y_positions[current_neuron_idx]
            y_end = neuron_y_positions[current_neuron_idx + len(units)]

            ax.add_patch(Rectangle((time_window[0], y_start),
                                  time_window[1] - time_window[0], y_end - y_start,
                                  facecolor=pref_color_map[pref_ori], alpha=0.15, zorder=0))

            current_neuron_idx += len(units)

        # Styling
        ax.axvline(0, color='red', linewidth=1.5, linestyle='-', alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{stimulus_ori}°', fontsize=12, fontweight='bold')
        ax.set_xlim(time_window)
        ax.set_ylim(-0.5, neuron_y_positions[-1] - 0.5)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')

        if ori_idx == 0:
            ax.set_ylabel('Trials (grouped by neuron)', fontsize=10)

    plt.suptitle('Population Response Across All Orientations', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {Path(save_path).name}")
        plt.close(fig)

    return fig


def generate_population_rasters(data_path, time_window=(-0.1, 0.3),
                                preference_window=(0.07, 0.16),
                                output_folder=None, include_psth=True,
                                osi_threshold=None, include_summary=True):
    """Generate population raster plots for all orientations."""
    data = load_neural_data(data_path)
    preference_data = calculate_preferred_orientations(data, time_window=preference_window,
                                                      osi_threshold=osi_threshold)
    unique_orientations = preference_data['unique_orientations']

    if output_folder is None:
        output_folder = Path(data_path).parent / f"{Path(data_path).stem}_population_rasters"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_folder}")

    for ori in unique_orientations:
        save_path = output_folder / f"population_raster_{ori}deg.png"
        if include_psth:
            plot_population_raster_with_psth(data, preference_data, ori,
                                            time_window=time_window, save_path=save_path)
        else:
            plot_population_raster(data, preference_data, ori,
                                  time_window=time_window, save_path=save_path)

    # Create summary figure with all orientations
    if include_summary:
        summary_path = output_folder / "population_raster_all_orientations.png"
        plot_all_orientations_summary(data, preference_data, time_window=time_window,
                                     save_path=summary_path)

    print(f"✓ Done: {output_folder}")
    save_preference_summary(preference_data, output_folder / "preference_summary.txt")

    return preference_data


def save_preference_summary(preference_data, save_path):
    """Save neuron preference summary."""
    unit_preferences = preference_data['unit_preferences']
    orientation_groups = preference_data['orientation_groups']
    unique_orientations = preference_data['unique_orientations']

    with open(save_path, 'w') as f:
        f.write("NEURON ORIENTATION PREFERENCES\n" + "="*70 + "\n\n")

        for ori in unique_orientations:
            units = orientation_groups[ori]
            f.write(f"Prefer {ori}° ({len(units)} units):\n")
            for unit_id in units:
                pref = unit_preferences[unit_id]
                f.write(f"  {unit_id:30s} | OSI: {pref['osi']:.3f} | Max: {pref['max_rate']:.1f}Hz\n")
            f.write("\n")

        all_osis = [pref['osi'] for pref in unit_preferences.values()]
        f.write(f"\nSTATISTICS\n{'='*70}\n")
        f.write(f"Total units: {len(unit_preferences)}\n")
        for ori in unique_orientations:
            n = len(orientation_groups[ori])
            pct = 100 * n / len(unit_preferences) if len(unit_preferences) > 0 else 0
            f.write(f"  {ori}°: {n} units ({pct:.1f}%)\n")
        f.write(f"\nOSI: mean={np.mean(all_osis):.3f}, median={np.median(all_osis):.3f}, "
               f"range=[{np.min(all_osis):.3f}, {np.max(all_osis):.3f}]\n")

    print(f"✓ Summary: {Path(save_path).name}")


if __name__ == "__main__":
    DATA_PATH = resolve_data_path()

    generate_population_rasters(
        data_path=DATA_PATH,
        time_window=(-0.2, 2.0),
        preference_window=(0.07, 1.0),
        output_folder=None,
        include_psth=True,
        osi_threshold=0.2,
        include_summary=True
    )