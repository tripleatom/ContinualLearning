import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import warnings


def load_halfvisualfield_data(npz_file):
    """
    Load halfVisualField data from npz file and filter out noise units.
    
    Returns:
    firing_rates: Array of shape (n_units, n_trials) with firing rates
    orientations: Array of orientations for each trial
    unit_info: List of (shank, unit_id) tuples
    unit_qualities: List of quality strings
    metadata: Dictionary with additional information
    """
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract data
    left_orientations = data['left_orientation']
    all_units_responses = data['all_units_responses']
    unit_info = data['unit_info']
    unit_qualities = data['unit_qualities']
    
    # Extract metadata
    metadata = {
        'session': str(data.get('session', 'unknown')),
        'n_trials_original': int(data.get('n_trials_original', len(left_orientations))),
        'n_trials_filtered': int(data.get('n_trials_filtered', len(left_orientations))),
        'bad_trials': data.get('bad_trials', np.array([])),
        'sampling_frequency': data.get('sampling_frequency', None)
    }
    
    # Filter out noise units
    noise_mask = np.array(unit_qualities) != 'noise'
    good_units_responses = [resp for i, resp in enumerate(all_units_responses) if noise_mask[i]]
    good_unit_info = [info for i, info in enumerate(unit_info) if noise_mask[i]]
    good_unit_qualities = [qual for i, qual in enumerate(unit_qualities) if noise_mask[i]]
    
    # Convert to numpy array
    firing_rates = np.array([resp['mean_firing_rates'] for resp in good_units_responses])
    
    print(f"Loaded {len(good_units_responses)} non-noise units from {len(all_units_responses)} total units")
    print(f"Number of trials: {len(left_orientations)} (after excluding {len(metadata['bad_trials'])} bad trials)")
    print(f"Unique orientations: {np.unique(left_orientations)}")
    
    return firing_rates, np.array(left_orientations), good_unit_info, good_unit_qualities, metadata


def gaussian_tuning_curve(orientation, amplitude, preferred_orientation, bandwidth, baseline):
    """
    Gaussian tuning curve model.
    
    Parameters:
    orientation: array of orientation values
    amplitude: peak response amplitude
    preferred_orientation: preferred orientation (degrees)
    bandwidth: tuning width parameter
    baseline: baseline firing rate
    """
    # Handle circular nature of orientation (0° = 180°)
    angular_diff = np.abs(orientation - preferred_orientation)
    angular_diff = np.minimum(angular_diff, 180 - angular_diff)
    
    return baseline + amplitude * np.exp(-(angular_diff**2) / (2 * bandwidth**2))


def compute_tuning_metrics(orientations, firing_rates):
    """
    Compute various tuning curve metrics including gOSI.
    
    Returns:
    metrics: Dictionary with tuning metrics
    """
    unique_orientations = np.unique(orientations)
    mean_rates = []
    sem_rates = []
    
    for ori in unique_orientations:
        ori_trials = firing_rates[orientations == ori]
        mean_rates.append(np.mean(ori_trials))
        sem_rates.append(stats.sem(ori_trials))
    
    mean_rates = np.array(mean_rates)
    sem_rates = np.array(sem_rates)
    
    # Preferred orientation (orientation with maximum response)
    preferred_idx = np.argmax(mean_rates)
    preferred_orientation = unique_orientations[preferred_idx]
    max_response = mean_rates[preferred_idx]
    
    # Baseline (minimum response)
    baseline = np.min(mean_rates)
    
    # Modulation depth
    modulation_depth = (max_response - baseline) / (max_response + baseline) if (max_response + baseline) > 0 else 0
    
    # Standard Orientation selectivity index (OSI)
    osi = (max_response - baseline) / max_response if max_response > 0 else 0
    
    # Global Orientation Selectivity Index (gOSI)
    # gOSI = (Rpref - Rorth) / (Rpref + Rorth)
    # where Rpref is response to preferred orientation and Rorth is response to orthogonal orientation
    orthogonal_orientation = (preferred_orientation + 90) % 180
    
    # Find closest orthogonal orientation in data
    orth_diffs = np.abs(unique_orientations - orthogonal_orientation)
    orth_diffs = np.minimum(orth_diffs, 180 - orth_diffs)  # circular distance
    orth_idx = np.argmin(orth_diffs)
    orthogonal_response = mean_rates[orth_idx]
    
    if (max_response + orthogonal_response) > 0:
        gosi = (max_response - orthogonal_response) / (max_response + orthogonal_response)
    else:
        gosi = 0
    
    # Circular variance (1 - |vector sum|/sum of responses)
    # Convert orientations to radians and double for circular statistics
    ori_rad = np.deg2rad(unique_orientations * 2)  # Double for orientation (not direction)
    weighted_sum_cos = np.sum(mean_rates * np.cos(ori_rad))
    weighted_sum_sin = np.sum(mean_rates * np.sin(ori_rad))
    total_response = np.sum(mean_rates)
    
    if total_response > 0:
        circular_variance = 1 - np.sqrt(weighted_sum_cos**2 + weighted_sum_sin**2) / total_response
    else:
        circular_variance = 1
    
    # ANOVA test for orientation selectivity
    orientation_groups = [firing_rates[orientations == ori] for ori in unique_orientations]
    
    # Check if we have enough data for ANOVA
    valid_groups = [group for group in orientation_groups if len(group) > 0]
    if len(valid_groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*valid_groups)
        except:
            f_stat, p_value = np.nan, 1.0
    else:
        f_stat, p_value = np.nan, 1.0
    
    metrics = {
        'preferred_orientation': preferred_orientation,
        'max_response': max_response,
        'baseline': baseline,
        'modulation_depth': modulation_depth,
        'osi': osi,
        'gosi': gosi,
        'orthogonal_response': orthogonal_response,
        'circular_variance': circular_variance,
        'f_statistic': f_stat,
        'p_value': p_value,
        'is_tuned': p_value < 0.05 if not np.isnan(p_value) else False,
        'unique_orientations': unique_orientations,
        'mean_rates': mean_rates,
        'sem_rates': sem_rates
    }
    
    return metrics


def fit_tuning_curve(orientations, firing_rates, model='gaussian'):
    """
    Fit a tuning curve model to the data with improved error handling.
    
    Parameters:
    orientations: array of orientations
    firing_rates: array of firing rates
    model: 'gaussian' for Gaussian model
    
    Returns:
    fitted_params: fitted parameters
    fitted_curve: fitted curve values
    r_squared: goodness of fit
    """
    unique_orientations = np.unique(orientations)
    mean_rates = []
    
    for ori in unique_orientations:
        mean_rates.append(np.mean(firing_rates[orientations == ori]))
    
    mean_rates = np.array(mean_rates)
    
    # Check if we have enough data points
    if len(unique_orientations) < 4:
        return None, None, None, 0
    
    # Initial parameter guesses
    amplitude_guess = np.max(mean_rates) - np.min(mean_rates)
    preferred_guess = unique_orientations[np.argmax(mean_rates)]
    baseline_guess = np.min(mean_rates)
    
    # Only fit Gaussian model (simpler and more robust)
    try:
        bandwidth_guess = 30  # degrees
        initial_guess = [amplitude_guess, preferred_guess, bandwidth_guess, baseline_guess]
        
        # Bounds: amplitude > 0, orientation 0-180, bandwidth > 0, baseline >= 0
        bounds = ([0, 0, 1, 0], [np.inf, 180, 90, np.inf])
        
        # Suppress optimization warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_params, _ = curve_fit(gaussian_tuning_curve, unique_orientations, mean_rates,
                                       p0=initial_guess, bounds=bounds, maxfev=2000)
        
        # Generate fitted curve
        orientation_fine = np.linspace(0, 180, 181)
        fitted_curve = gaussian_tuning_curve(orientation_fine, *fitted_params)
        
        # Calculate R-squared
        predicted = gaussian_tuning_curve(unique_orientations, *fitted_params)
        ss_res = np.sum((mean_rates - predicted) ** 2)
        ss_tot = np.sum((mean_rates - np.mean(mean_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return fitted_params, fitted_curve, orientation_fine, r_squared
        
    except Exception as e:
        return None, None, None, 0


def plot_individual_tuning_curve(orientations, firing_rates, unit_idx, unit_info, unit_quality, 
                                metrics, fit_results=None):
    """
    Plot tuning curve for an individual unit and return the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    unique_orientations = metrics['unique_orientations']
    mean_rates = metrics['mean_rates']
    sem_rates = metrics['sem_rates']
    
    # Plot raw data points
    for ori in unique_orientations:
        ori_rates = firing_rates[orientations == ori]
        ax.scatter([ori] * len(ori_rates), ori_rates, alpha=0.3, s=20, color='gray')
    
    # Plot mean ± SEM
    ax.errorbar(unique_orientations, mean_rates, yerr=sem_rates, 
                fmt='o-', color='blue', markersize=8, linewidth=2, capsize=5)
    
    # Plot fitted curve if available
    if fit_results is not None:
        fitted_params, fitted_curve, orientation_fine, r_squared = fit_results
        if fitted_curve is not None:
            ax.plot(orientation_fine, fitted_curve, '--', color='red', linewidth=2, 
                   label=f'Fitted curve (R² = {r_squared:.3f})')
    
    # Formatting
    ax.set_xlabel('Grating Orientation (degrees)', fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_xlim(-10, 190)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.grid(True, alpha=0.3)
    
    # Title with unit info and metrics
    shank, unit_id = unit_info
    title = (f"Unit {unit_idx}: Shank {shank}, ID {unit_id} ({unit_quality})\n"
             f"Preferred: {metrics['preferred_orientation']:.0f}°, "
             f"OSI: {metrics['osi']:.3f}, "
             f"gOSI: {metrics['gosi']:.3f}, "
             f"p = {metrics['p_value']:.3e}")
    ax.set_title(title, fontsize=14)
    
    if fit_results is not None and fit_results[1] is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_tuning_summary(all_metrics, unit_info, unit_qualities, metadata, save_dir):
    """
    Create summary plots of tuning properties across all units.
    """
    # Convert metrics to arrays for plotting
    preferred_orientations = [m['preferred_orientation'] for m in all_metrics]
    osis = [m['osi'] for m in all_metrics]
    gosis = [m['gosi'] for m in all_metrics]
    modulation_depths = [m['modulation_depth'] for m in all_metrics]
    circular_variances = [m['circular_variance'] for m in all_metrics]
    p_values = [m['p_value'] for m in all_metrics if not np.isnan(m['p_value'])]
    max_responses = [m['max_response'] for m in all_metrics]
    
    # Create summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Preferred orientation distribution
    ax = axes[0, 0]
    ax.hist(preferred_orientations, bins=18, range=(0, 180), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Preferred Orientation (degrees)')
    ax.set_ylabel('Number of Units')
    ax.set_title('Distribution of Preferred Orientations')
    ax.set_xlim(0, 180)
    
    # 2. OSI vs gOSI comparison
    ax = axes[0, 1]
    ax.scatter(osis, gosis, alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Unity line')
    ax.set_xlabel('OSI')
    ax.set_ylabel('gOSI')
    ax.set_title('OSI vs gOSI')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. gOSI distribution
    ax = axes[0, 2]
    ax.hist(gosis, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(gosis), color='red', linestyle='--', label=f'Median: {np.median(gosis):.3f}')
    ax.set_xlabel('Global Orientation Selectivity Index (gOSI)')
    ax.set_ylabel('Number of Units')
    ax.set_title('Distribution of gOSI')
    ax.legend()
    
    # 4. Tuning significance
    ax = axes[1, 0]
    if len(p_values) > 0:
        tuned_units = np.sum([p < 0.05 for p in p_values])
        total_units = len(p_values)
        labels = ['Tuned', 'Not Tuned']
        sizes = [tuned_units, total_units - tuned_units]
        colors = ['lightgreen', 'lightcoral']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Orientation Tuning Significance\n({tuned_units}/{total_units} units tuned)')
    else:
        ax.text(0.5, 0.5, 'No valid p-values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Tuning Significance: No Data')
    
    # 5. Quality breakdown
    ax = axes[1, 1]
    quality_counts = {}
    for quality in unit_qualities:
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    qualities = list(quality_counts.keys())
    counts = list(quality_counts.values())
    colors_qual = ['green' if q == 'good' else 'orange' if q == 'mua' else 'gray' for q in qualities]
    
    bars = ax.bar(qualities, counts, color=colors_qual, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Unit Quality')
    ax.set_ylabel('Number of Units')
    ax.set_title('Unit Quality Distribution')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               str(count), ha='center', va='bottom', fontsize=12)
    
    # 6. gOSI vs max response
    ax = axes[1, 2]
    scatter = ax.scatter(gosis, max_responses, c=preferred_orientations, cmap='hsv', alpha=0.7)
    ax.set_xlabel('gOSI')
    ax.set_ylabel('Max Response (Hz)')
    ax.set_title('gOSI vs Max Response')
    plt.colorbar(scatter, ax=ax, label='Preferred Orientation (°)')
    
    # Main title
    session_name = metadata['session']
    tuned_count = len(p_values) if len(p_values) > 0 else 0
    fig.suptitle(f"Tuning Curve Summary - {session_name}\n"
                f"Total Units: {len(all_metrics)}, "
                f"Median gOSI: {np.median(gosis):.3f}", 
                fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tuning_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_tuning_curves(npz_file, save_individual=True, max_individual_plots=50):
    """
    Main function to analyze tuning curves for all units.
    
    Parameters:
    npz_file: Path to the halfVisualField npz file
    save_individual: Whether to save individual unit plots
    max_individual_plots: Maximum number of individual plots to save
    
    Returns:
    all_metrics: List of tuning metrics for each unit
    metrics_df: DataFrame with all metrics
    """
    print(f"Analyzing tuning curves from {npz_file}")
    
    # Load data
    firing_rates, orientations, unit_info, unit_qualities, metadata = load_halfvisualfield_data(npz_file)
    
    if len(unit_info) == 0:
        print("No units found!")
        return None, None
    
    # Setup save directory
    save_dir = Path(npz_file).parent / 'tuning_analysis'
    save_dir.mkdir(exist_ok=True)
    
    # Analyze each unit
    all_metrics = []
    all_fit_results = []
    
    print(f"Analyzing {len(unit_info)} units...")
    
    # Create PDF for individual plots if saving
    pdf_pages = None
    if save_individual:
        pdf_path = save_dir / 'individual_tuning_curves.pdf'
        pdf_pages = PdfPages(pdf_path)
    
    try:
        for unit_idx in range(len(unit_info)):
            unit_rates = firing_rates[unit_idx]
            
            # Compute tuning metrics
            metrics = compute_tuning_metrics(orientations, unit_rates)
            all_metrics.append(metrics)
            
            # Fit tuning curve
            fit_results = fit_tuning_curve(orientations, unit_rates, model='gaussian')
            all_fit_results.append(fit_results)
            
            # Plot individual tuning curve
            if save_individual and unit_idx < max_individual_plots:
                fig = plot_individual_tuning_curve(
                    orientations, unit_rates, unit_idx, unit_info[unit_idx], 
                    unit_qualities[unit_idx], metrics, fit_results
                )
                if pdf_pages is not None:
                    pdf_pages.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            if (unit_idx + 1) % 10 == 0:
                print(f"Processed {unit_idx + 1}/{len(unit_info)} units")
                
    finally:
        if pdf_pages is not None:
            pdf_pages.close()
            print(f"Saved individual tuning curves to {pdf_path}")
    
    # Create summary plots
    plot_tuning_summary(all_metrics, unit_info, unit_qualities, metadata, save_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'unit_idx': range(len(unit_info)),
        'shank': [info[0] for info in unit_info],
        'unit_id': [info[1] for info in unit_info],
        'quality': unit_qualities,
        'preferred_orientation': [m['preferred_orientation'] for m in all_metrics],
        'max_response': [m['max_response'] for m in all_metrics],
        'baseline': [m['baseline'] for m in all_metrics],
        'osi': [m['osi'] for m in all_metrics],
        'gosi': [m['gosi'] for m in all_metrics],
        'orthogonal_response': [m['orthogonal_response'] for m in all_metrics],
        'modulation_depth': [m['modulation_depth'] for m in all_metrics],
        'circular_variance': [m['circular_variance'] for m in all_metrics],
        'f_statistic': [m['f_statistic'] for m in all_metrics],
        'p_value': [m['p_value'] for m in all_metrics],
        'is_tuned': [m['is_tuned'] for m in all_metrics]
    })
    
    # Add fitting results if available
    valid_fits = [fit for fit in all_fit_results if fit[0] is not None]
    if len(valid_fits) > 0:
        param_names = ['amplitude', 'preferred_ori_fit', 'bandwidth', 'baseline_fit']
        
        for i, param_name in enumerate(param_names):
            metrics_df[param_name] = [fit[0][i] if fit[0] is not None else np.nan for fit in all_fit_results]
        
        metrics_df['r_squared'] = [fit[3] if fit[0] is not None else np.nan for fit in all_fit_results]
    
    csv_path = save_dir / 'tuning_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved tuning metrics to {csv_path}")
    
    # Print summary statistics
    tuned_units = sum(m['is_tuned'] for m in all_metrics)
    gosi_values = [m['gosi'] for m in all_metrics]
    
    print(f"\nTuning Analysis Summary:")
    print(f"Total units analyzed: {len(all_metrics)}")
    print(f"Significantly tuned units: {tuned_units} ({100*tuned_units/len(all_metrics):.1f}%)")
    print(f"Median OSI: {np.median([m['osi'] for m in all_metrics]):.3f}")
    print(f"Median gOSI: {np.median(gosi_values):.3f}")
    print(f"Mean gOSI: {np.mean(gosi_values):.3f} ± {np.std(gosi_values):.3f}")
    print(f"gOSI range: {np.min(gosi_values):.3f} to {np.max(gosi_values):.3f}")
    print(f"Results saved to: {save_dir}")
    
    return all_metrics, metrics_df


if __name__ == '__main__':
    # Example usage
    npz_file = r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22SG\CnL22SG_20250727_172100\HalfGrating\HalfGrating_data.npz"
    
    if Path(npz_file).exists():
        # Analyze tuning curves
        all_metrics, metrics_df = analyze_tuning_curves(
            npz_file, 
            save_individual=True,
            max_individual_plots=50
        )
        
        if all_metrics is not None:
            print("Tuning curve analysis completed successfully!")
        else:
            print("Analysis failed - no units found.")
    else:
        print(f"File not found: {npz_file}")
        print("Please update the path to your data file.")