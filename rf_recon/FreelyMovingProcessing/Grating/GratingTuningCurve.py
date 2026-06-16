"""
Plot Orientation Tuning Curves for All Neurons

This script loads neural data and generates individual tuning curve plots
for each unit, saved to a dedicated folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING (reusing from main script)
# =============================================================================

def load_neural_data(filepath):
    """Load neural data from pickle format."""
    filepath = Path(filepath)

    if filepath.suffix != '.pkl':
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Only .pkl files are supported.")

    print(f"Loading data from {filepath.suffix}: {filepath.name}")
    return _load_pickle(filepath)


def _load_pickle(filepath):
    """Load from pickle format."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# TUNING CURVE CALCULATION
# =============================================================================

def calculate_tuning_curves(neural_data, time_window=(0.07, 0.16)):
    """
    Calculate tuning curves for all units.
    
    Returns:
        Dictionary containing:
        - unit_tuning_data: dict with unit_id as key, tuning data as value
        - unique_orientations: list of tested orientations
        - experiment_info: dict with experimental parameters
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start
    
    unit_ids = list(neural_data['spike_data'].keys())
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = sorted(neural_data['trial_info']['unique_orientations'])
    
    print(f"\nCalculating tuning curves:")
    print(f"  Units: {len(unit_ids)}")
    print(f"  Orientations: {unique_orientations}")
    print(f"  Time window: {window_start:.3f}-{window_end:.3f}s ({window_duration:.3f}s)")
    
    unit_tuning_data = {}
    
    for unit_id in unit_ids:
        # Collect firing rates per trial
        unit_trials = neural_data['spike_data'][unit_id]
        trial_rates = {ori: [] for ori in unique_orientations}
        
        for trial_data in unit_trials:
            orientation = trial_data['orientation']
            if orientation in unique_orientations:
                spike_times = np.array(trial_data['spike_times'])
                spikes_in_window = np.sum((spike_times >= window_start) & 
                                         (spike_times < window_end))
                firing_rate = spikes_in_window / window_duration
                trial_rates[orientation].append(firing_rate)
        
        # Calculate statistics per orientation
        mean_rates = []
        sem_rates = []
        std_rates = []
        trial_counts = []
        
        for ori in unique_orientations:
            rates = trial_rates[ori]
            if len(rates) > 0:
                mean_rates.append(np.mean(rates))
                sem_rates.append(stats.sem(rates))
                std_rates.append(np.std(rates))
                trial_counts.append(len(rates))
            else:
                mean_rates.append(0)
                sem_rates.append(0)
                std_rates.append(0)
                trial_counts.append(0)
        
        # Calculate tuning metrics
        mean_rates_arr = np.array(mean_rates)
        
        # Orientation selectivity index (OSI) - vector sum method
        theta_rad = 2 * np.deg2rad(unique_orientations)
        complex_sum = np.sum(mean_rates_arr * np.exp(1j * theta_rad))
        osi = np.abs(complex_sum) / (np.sum(mean_rates_arr) + 1e-12)
        preferred_ori = (np.angle(complex_sum) / 2.0) % np.pi
        preferred_ori_deg = np.rad2deg(preferred_ori)
        
        # Modulation index
        max_rate = np.max(mean_rates_arr)
        min_rate = np.min(mean_rates_arr)
        modulation_index = (max_rate - min_rate) / (max_rate + min_rate + 1e-12)
        
        # Baseline firing rate (mean across all orientations)
        baseline_rate = np.mean(mean_rates_arr)
        
        # PSTH: 20 ms bins, -0.2 to 1.5 s relative to stimulus onset
        psth_bin_s = 0.02
        psth_edges = np.arange(-0.2, 1.5 + psth_bin_s, psth_bin_s)
        psth_t = (psth_edges[:-1] + psth_edges[1:]) / 2
        psth_per_ori = {}
        for ori in unique_orientations:
            spikes_all = []
            n_ori_trials = 0
            for trial_data in unit_trials:
                if trial_data['orientation'] == ori:
                    spikes_all.append(np.array(trial_data['spike_times']))
                    n_ori_trials += 1
            if n_ori_trials > 0 and spikes_all:
                counts, _ = np.histogram(np.concatenate(spikes_all), bins=psth_edges)
                psth_per_ori[ori] = (counts / (n_ori_trials * psth_bin_s)).tolist()
            else:
                psth_per_ori[ori] = [0.0] * len(psth_t)

        unit_tuning_data[unit_id] = {
            'orientations': unique_orientations,
            'mean_rates': mean_rates,
            'sem_rates': sem_rates,
            'std_rates': std_rates,
            'trial_counts': trial_counts,
            'trial_rates': trial_rates,
            'osi': osi,
            'preferred_orientation_deg': preferred_ori_deg,
            'modulation_index': modulation_index,
            'max_rate': max_rate,
            'min_rate': min_rate,
            'baseline_rate': baseline_rate,
            'psth_per_ori': psth_per_ori,
            'psth_t': psth_t.tolist(),
        }
    
    experiment_info = {
        'time_window': time_window,
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_units': len(unit_ids)
    }
    
    return {
        'unit_tuning_data': unit_tuning_data,
        'unique_orientations': unique_orientations,
        'experiment_info': experiment_info
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_single_tuning_curve(unit_id, tuning_data, unit_info=None,
                             time_window=(0.07, 0.16), save_path=None):
    """
    Create a comprehensive tuning curve plot for a single unit.

    Layout (3 rows × 4 columns):
      Col 0-1: Cartesian tuning curve (rows 0-2)
      Col 2:   Polar plot (row 0), Boxplot (row 1), Tuning statistics (row 2)
      Col 3:   Waveform (row 0), ACG (row 1), Unit info (row 2)

    Args:
        unit_info: dict from neural_data['unit_info'] — may contain
                   'waveform_template', 'waveform_t_ms', 'acg_counts',
                   'acg_lags_ms', 'best_channel', 'channel_location_um',
                   'shank', 'quality'.  Pass None to skip those panels.
    """
    if unit_info is None:
        unit_info = {}

    fig = plt.figure(figsize=(24, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)

    orientations = tuning_data['orientations']
    mean_rates = np.array(tuning_data['mean_rates'])
    sem_rates = np.array(tuning_data['sem_rates'])

    # Build a short header with unit identity
    shank_str = f"shank{unit_info.get('shank', '?')}"
    ch_str = (f"ch{unit_info.get('best_channel', '?')}"
              if unit_info.get('best_channel') is not None else "")
    loc = unit_info.get('channel_location_um')
    loc_str = f"  [{loc[0]:.0f}, {loc[1]:.0f}] µm" if loc else ""
    quality = unit_info.get('quality', '')
    header = f"{unit_id}  |  {shank_str}  {ch_str}{loc_str}  {quality}"
    fig.suptitle(header, fontsize=26, fontweight='bold', y=0.995)

    # ------------------------------------------------------------------ #
    # 1. Cartesian tuning curve  (row 0, col 0)
    # ------------------------------------------------------------------ #
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(orientations, mean_rates, '-', color='#2E86AB',
             linewidth=6.5, zorder=2)
    ax1.errorbar(orientations, mean_rates, yerr=sem_rates,
                 fmt='none', ecolor='#A23B72', capsize=14, capthick=4.0,
                 elinewidth=4.0, zorder=4)
    ax1.plot(orientations, mean_rates, 'o', color='#2E86AB',
             markersize=18, markeredgecolor='white', markeredgewidth=2.0,
             zorder=3)
    pref_idx = np.argmax(mean_rates)
    ax1.plot(orientations[pref_idx], mean_rates[pref_idx],
             '*', color='red', markersize=34,
             markeredgecolor='white', markeredgewidth=1.5,
             zorder=6)
    ax1.set_xlabel('Orientation (degrees)', fontsize=22, fontweight='bold',
                   labelpad=10)
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=22, fontweight='bold')
    ax1.set_title('Tuning Curve', fontsize=24, fontweight='bold', pad=12)
    ax1.set_xticks(orientations)
    ax1.set_xticklabels([f'{o:g}' for o in orientations],
                        rotation=45, ha='right')
    ax1.tick_params(axis='both', labelsize=20, width=2.5, length=9)
    for spine in ('left', 'bottom'):
        ax1.spines[spine].set_linewidth(2.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # 2. Polar tuning curve  (row 0, col 1)
    # ------------------------------------------------------------------ #
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    theta = 2 * np.deg2rad(orientations)
    theta_plot = np.concatenate([theta, [theta[0]]])
    rates_plot = np.concatenate([mean_rates, [mean_rates[0]]])
    ax2.plot(theta_plot, rates_plot, 'o-', linewidth=5.0, markersize=14, color='#2E86AB')
    ax2.fill(theta_plot, rates_plot, alpha=0.25, color='#2E86AB')
    pref_theta = 2 * np.deg2rad(tuning_data['preferred_orientation_deg'])
    ax2.plot(pref_theta, np.max(mean_rates), '*', color='red', markersize=30)
    ax2.set_title('Polar', fontsize=24, fontweight='bold', pad=12)
    ax2.set_thetagrids(np.arange(0, 360, 45),
                       [f'{a/2:g}°' for a in np.arange(0, 360, 45)],
                       fontsize=22)
    radial_max = float(np.nanmax(mean_rates)) if mean_rates.size else 0.0
    if radial_max > 0:
        ax2.set_rlim(0, radial_max * 1.18)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='x', pad=18)
    ax2.grid(True, linewidth=1.4, alpha=0.45)
    ax2.spines['polar'].set_linewidth(1.8)

    # ------------------------------------------------------------------ #
    # 3. PSTH  (row 1, col 0)
    # ------------------------------------------------------------------ #
    ax3 = fig.add_subplot(gs[1, 0])
    psth_t = np.array(tuning_data['psth_t'])
    psth_colors = plt.cm.hsv(np.linspace(0, 1, len(orientations) + 1)[:-1])
    for k, ori in enumerate(orientations):
        psth_rate = gaussian_filter1d(np.array(tuning_data['psth_per_ori'][ori]), sigma=1.5)
        ax3.plot(psth_t, psth_rate, color=psth_colors[k],
                 linewidth=3.5, label=f'{ori}°', alpha=0.9)
    ax3.axvline(0, color='black', linewidth=3.0, linestyle='--', label='onset')
    ax3.axvspan(time_window[0], time_window[1], alpha=0.15, color='gray',
                label='analysis\nwindow')
    ax3.set_xlabel('Time re. onset (s)', fontsize=22, fontweight='bold')
    ax3.set_ylabel('Firing Rate (Hz)', fontsize=22, fontweight='bold')
    ax3.set_title('PSTH', fontsize=24, fontweight='bold', pad=12)
    ax3.legend(fontsize=9, loc='upper right', ncol=2, frameon=True,
               framealpha=0.85, edgecolor='none',
               handlelength=1.2, handletextpad=0.4,
               columnspacing=0.8, labelspacing=0.25, borderpad=0.3)
    ax3.tick_params(axis='both', labelsize=22, width=2.8, length=10)
    for spine in ('left', 'bottom'):
        ax3.spines[spine].set_linewidth(2.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # 6. Waveform  (row 0, col 2)
    # ------------------------------------------------------------------ #
    ax6 = fig.add_subplot(gs[0, 2])
    wf = unit_info.get('waveform_template')
    wf_t = unit_info.get('waveform_t_ms')
    if wf is not None:
        wf_arr = np.array(wf)
        t_arr = np.array(wf_t) if wf_t is not None else np.arange(len(wf_arr))
        wf_arr = wf_arr / (np.max(np.abs(wf_arr)) + 1e-12)   # normalize to ±1
        ax6.plot(t_arr, wf_arr, color='black', linewidth=4.5)
    else:
        ax6.text(0.5, 0.5, 'no waveform', ha='center', va='center',
                 transform=ax6.transAxes, color='gray', fontsize=20)
    ax6.set_title('Waveform', fontsize=24, fontweight='bold', pad=12)
    ax6.set_axis_off()

    # ------------------------------------------------------------------ #
    # 7. ACG  (row 1, col 1)
    # ------------------------------------------------------------------ #
    ax7 = fig.add_subplot(gs[1, 1])
    acg = unit_info.get('acg_counts')
    acg_lags = unit_info.get('acg_lags_ms')
    if acg is not None:
        acg_arr = np.array(acg)
        lags_arr = np.array(acg_lags)
        # zero out the central bin (self-coincidence)
        center = len(acg_arr) // 2
        acg_arr[center] = 0
        ax7.bar(lags_arr, acg_arr, width=(lags_arr[1] - lags_arr[0]) * 0.9,
                color='#5B8DB8', edgecolor='none', alpha=0.9)
        ax7.axvline(0, color='red', linewidth=3.0, linestyle='--')
        ax7.set_xlim(-25, 25)
        ax7.set_xlabel('Lag (ms)', fontsize=22, fontweight='bold')
        ax7.set_ylabel('Rate (Hz)', fontsize=22, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'no ACG', ha='center', va='center',
                 transform=ax7.transAxes, color='gray', fontsize=20)
    ax7.set_title('Autocorrelogram', fontsize=24, fontweight='bold', pad=12)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    for spine in ('left', 'bottom'):
        ax7.spines[spine].set_linewidth(2.5)
    ax7.tick_params(labelsize=22, width=2.8, length=10)

    # ------------------------------------------------------------------ #
    # 8. Combined stats + unit info  (row 1, col 2)
    # ------------------------------------------------------------------ #
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.axis('off')
    loc = unit_info.get('channel_location_um')
    loc_txt = (f"[{loc[0]:.0f}, {loc[1]:.0f}] µm" if loc else 'N/A')
    stats_text = (
        f"TUNING STATISTICS\n"
        f"OSI:        {tuning_data['osi']:.3f}\n"
        f"Preferred:  {tuning_data['preferred_orientation_deg']:.1f}°\n"
        f"Mod. Index: {tuning_data['modulation_index']:.3f}\n"
        f"Max FR:     {tuning_data['max_rate']:.2f} Hz\n"
        f"Min FR:     {tuning_data['min_rate']:.2f} Hz\n"
        f"Baseline:   {tuning_data['baseline_rate']:.2f} Hz\n"
        f"Trials:     {sum(tuning_data['trial_counts'])} "
        f"({min(tuning_data['trial_counts'])}"
        f"–{max(tuning_data['trial_counts'])}/ori)\n"
        f"\n"
        f"UNIT INFO\n"
        f"Shank:    {unit_info.get('shank', 'N/A')}\n"
        f"Channel:  {unit_info.get('best_channel', 'N/A')}\n"
        f"Position: {loc_txt}\n"
        f"Quality:  {unit_info.get('quality', 'N/A')}\n"
        f"N spikes: {unit_info.get('n_spikes_total', 'N/A')}"
    )
    ax8.text(0.02, 0.98, stats_text, transform=ax8.transAxes,
             fontsize=16, verticalalignment='top', fontfamily='monospace',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                       alpha=0.5))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_svg = save_path.suffix.lower() == '.svg'
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    transparent=is_svg,
                    facecolor='none' if is_svg else 'white')
        plt.close(fig)

    return fig


def plot_all_tuning_curves_summary(tuning_results, save_path=None, max_per_page=16):
    """
    Create summary plots showing all tuning curves in a grid.
    
    Args:
        tuning_results: Output from calculate_tuning_curves()
        save_path: Base path for saving (will add _page1.png, _page2.png, etc.)
        max_per_page: Maximum number of units per page (default 16 = 4x4 grid)
    """
    unit_tuning_data = tuning_results['unit_tuning_data']
    unique_orientations = tuning_results['unique_orientations']
    unit_ids = sorted(unit_tuning_data.keys())
    
    n_units = len(unit_ids)
    n_pages = int(np.ceil(n_units / max_per_page))
    
    figures = []
    
    for page in range(n_pages):
        start_idx = page * max_per_page
        end_idx = min((page + 1) * max_per_page, n_units)
        page_units = unit_ids[start_idx:end_idx]
        
        n_units_page = len(page_units)
        n_cols = 4
        n_rows = int(np.ceil(n_units_page / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 7 * n_rows))
        fig.suptitle(f'Tuning Curves Summary (Page {page + 1}/{n_pages})',
                     fontsize=28, fontweight='bold')

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, unit_id in enumerate(page_units):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            tuning_data = unit_tuning_data[unit_id]
            mean_rates = tuning_data['mean_rates']
            sem_rates = tuning_data['sem_rates']

            ax.errorbar(unique_orientations, mean_rates, yerr=sem_rates,
                        marker='o', markersize=10, linewidth=3.0, capsize=6,
                        capthick=2.5, elinewidth=2.0, color='#2E86AB',
                        ecolor='#A23B72')
            ax.fill_between(unique_orientations,
                            np.array(mean_rates) - np.array(sem_rates),
                            np.array(mean_rates) + np.array(sem_rates),
                            alpha=0.2, color='#2E86AB')

            ax.set_title(f'{unit_id}\nOSI: {tuning_data["osi"]:.2f}',
                         fontsize=18, fontweight='bold')
            ax.set_xlabel('Orientation (°)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Rate (Hz)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=1.2)
            ax.tick_params(labelsize=14, width=1.8, length=6)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_linewidth(2.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide unused subplots
        for idx in range(n_units_page, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            suffix = Path(save_path).suffix or '.png'
            save_path_page = (Path(save_path).parent /
                              f"{Path(save_path).stem}_page{page + 1}{suffix}")
            is_svg = suffix.lower() == '.svg'
            fig.savefig(save_path_page, dpi=300, bbox_inches='tight',
                        transparent=is_svg,
                        facecolor='none' if is_svg else 'white')
            print(f"Saved summary page {page + 1} to: {save_path_page}")
        
        figures.append(fig)
    
    return figures


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def generate_tuning_curves(data_path, time_window=(0.07, 0.16), 
                          output_folder=None, create_summary=True):
    """
    Complete pipeline: load data, calculate tuning curves, save all plots.
    
    Args:
        data_path: Path to neural data file
        time_window: Tuple of (start, end) time in seconds for analysis
        output_folder: Folder to save tuning curve plots (default: data_path_tuning_curves)
        create_summary: Whether to create summary plots with all units
    
    Returns:
        Dictionary with tuning results
    """
    # Load data
    data = load_neural_data(data_path)
    all_unit_info = data.get('unit_info', {})

    # Calculate tuning curves
    tuning_results = calculate_tuning_curves(data, time_window=time_window)
    unit_tuning_data = tuning_results['unit_tuning_data']

    # Set up output folder
    if output_folder is None:
        output_folder = Path(data_path).parent / f"{Path(data_path).stem}_tuning_curves"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving tuning curves to: {output_folder}")

    # Generate individual plots + per-unit pkl
    print(f"\nGenerating individual tuning curve plots...")
    unit_ids = sorted(unit_tuning_data.keys())

    for i, unit_id in enumerate(unit_ids, 1):
        clean_id = unit_id.replace('/', '_').replace('\\', '_')
        unit_info = all_unit_info.get(unit_id, {})

        # PNG figure
        save_path = output_folder / f"{clean_id}_tuning_curve.png"
        plot_single_tuning_curve(unit_id, unit_tuning_data[unit_id],
                                 unit_info=unit_info,
                                 time_window=time_window,
                                 save_path=save_path)

        # Per-unit pkl: tuning data + unit identity/waveform/ACG
        pkl_path = output_folder / f"{clean_id}_tuning.pkl"
        unit_pkg = {
            'unit_id': unit_id,
            'tuning': unit_tuning_data[unit_id],
            'unit_info': unit_info,
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(unit_pkg, f, protocol=pickle.HIGHEST_PROTOCOL)

        if i % 10 == 0:
            print(f"  Processed {i}/{len(unit_ids)} units...")
    
    print(f"✓ Saved {len(unit_ids)} individual tuning curve plots")
    
    # Generate summary plots
    if create_summary:
        print(f"\nGenerating summary plots...")
        summary_path = output_folder / "tuning_curves_summary.png"
        plot_all_tuning_curves_summary(tuning_results, save_path=summary_path)
    
    # Save tuning statistics to CSV
    save_tuning_statistics(unit_tuning_data, output_folder / "tuning_statistics.csv")
    
    print(f"\n✓ All plots saved to: {output_folder}")
    
    return tuning_results


def _find_unit_id(all_unit_info, unit_tuning_data, shank, unit_num):
    """Resolve (shank, unit_num) to a unit_id present in unit_tuning_data.

    Match strategy, in order:
      1. The trailing integer in the unit_id equals unit_num AND unit_info['shank'] == shank
      2. unit_num as a positional index into the sorted list of units on that shank
    """
    import re
    shank = int(shank)
    unit_num = int(unit_num)

    for uid, info in all_unit_info.items():
        if uid not in unit_tuning_data:
            continue
        try:
            if int(info.get('shank', -1)) != shank:
                continue
        except (TypeError, ValueError):
            continue
        nums = re.findall(r'\d+', str(uid))
        if nums and int(nums[-1]) == unit_num:
            return uid

    shank_units = sorted(
        uid for uid, info in all_unit_info.items()
        if uid in unit_tuning_data
        and str(info.get('shank', '')).isdigit()
        and int(info['shank']) == shank
    )
    if 0 <= unit_num < len(shank_units):
        return shank_units[unit_num]

    return None


def plot_selected_unit(data_path, shank, unit_num, time_window=(0.05, 1.0),
                       save_path=None, show=True):
    """
    Plot the tuning curve for one selected unit, addressed by shank# and unit#.

    Args:
        data_path: Path to neural data .pkl
        shank: Shank index (int).
        unit_num: Either the numeric suffix of the unit_id, or its 0-based
                  positional index among units on that shank.
        time_window: (start, end) seconds for firing-rate window.
        save_path: Optional path to save PNG. If None and show=True, displays.
        show: If True and not saving, call plt.show().

    Returns:
        (unit_id, fig)
    """
    data = load_neural_data(data_path)
    all_unit_info = data.get('unit_info', {})

    tuning_results = calculate_tuning_curves(data, time_window=time_window)
    unit_tuning_data = tuning_results['unit_tuning_data']

    matched = _find_unit_id(all_unit_info, unit_tuning_data, shank, unit_num)
    if matched is None:
        available = sorted(
            (int(info.get('shank', -1)), uid)
            for uid, info in all_unit_info.items()
            if uid in unit_tuning_data
        )
        msg = "\n".join(f"  shank{s}: {uid}" for s, uid in available[:25])
        raise ValueError(
            f"No unit found for shank={shank}, unit#={unit_num}.\n"
            f"First available units:\n{msg}"
        )

    print(f"Matched unit: {matched}  (shank={shank}, unit#={unit_num})")

    fig = plot_single_tuning_curve(
        matched, unit_tuning_data[matched],
        unit_info=all_unit_info.get(matched, {}),
        time_window=time_window,
        save_path=save_path,
    )
    if show and save_path is None:
        plt.show()
    return matched, fig


def save_tuning_statistics(unit_tuning_data, save_path):
    """Save tuning statistics to CSV file."""
    import csv
    
    save_path = Path(save_path)
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'unit_id', 'osi', 'preferred_orientation_deg', 'modulation_index',
            'max_rate_hz', 'min_rate_hz', 'baseline_rate_hz', 'range_hz',
            'total_trials'
        ])
        
        # Data rows
        for unit_id in sorted(unit_tuning_data.keys()):
            data = unit_tuning_data[unit_id]
            writer.writerow([
                unit_id,
                f"{data['osi']:.4f}",
                f"{data['preferred_orientation_deg']:.2f}",
                f"{data['modulation_index']:.4f}",
                f"{data['max_rate']:.2f}",
                f"{data['min_rate']:.2f}",
                f"{data['baseline_rate']:.2f}",
                f"{data['max_rate'] - data['min_rate']:.2f}",
                sum(data['trial_counts'])
            ])
    
    print(f"✓ Saved tuning statistics to: {save_path}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot grating tuning curves (all units or one selected unit)."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to neural data .pkl file")
    parser.add_argument("--shank", type=int, default=None,
                        help="Shank# of the unit to plot (use with --unit)")
    parser.add_argument("--unit", type=int, default=None,
                        help="Unit# on the given shank (numeric suffix of the "
                             "unit_id, or 0-based index among units on that shank)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (selected-unit mode) or folder (batch mode)")
    parser.add_argument("--t0", type=float, default=0.05, help="Window start (s)")
    parser.add_argument("--t1", type=float, default=1.0, help="Window end (s)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip summary grid in batch mode")
    args = parser.parse_args()

    DATA_PATH = args.data
    if not DATA_PATH:
        DATA_PATH = input("Enter path to neural data (.pkl file): ").strip().strip('"').strip("'")

    time_window = (args.t0, args.t1)

    try:
        if args.shank is not None and args.unit is not None:
            # ---- Selected-unit mode ----
            plot_selected_unit(
                data_path=DATA_PATH,
                shank=args.shank,
                unit_num=args.unit,
                time_window=time_window,
                save_path=args.out,
                show=(args.out is None),
            )
            print("\n✓ Selected-unit plot complete.")
        else:
            # ---- Batch mode ----
            tuning_results = generate_tuning_curves(
                data_path=DATA_PATH,
                time_window=time_window,
                output_folder=args.out,
                create_summary=not args.no_summary,
            )
            print("\n" + "="*60)
            print("Tuning curve analysis complete!")
            print("="*60)

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
