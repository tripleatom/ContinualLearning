import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Object to orientation mapping
OBJECT_TO_ORIENTATION = {14: 0, 15: 45, 16: 90, 17: 135}

# Define pairwise comparisons
PAIRWISE_COMPARISONS = {
    'orthogonal': {
        'pair': (0, 90),
        'description': 'Orthogonal orientations (horizontal vs vertical)',
        'type': 'cardinal'
    },
    'oblique': {
        'pair': (45, 135), 
        'description': 'Oblique orientations (diagonal pair)',
        'type': 'oblique'
    },
    'cardinal_vs_oblique_1': {
        'pair': (0, 45),
        'description': 'Cardinal vs oblique (horizontal vs diagonal)',
        'type': 'mixed'
    },
    'cardinal_vs_oblique_2': {
        'pair': (90, 135),
        'description': 'Cardinal vs oblique (vertical vs diagonal)',
        'type': 'mixed'
    },
    'adjacent_1': {
        'pair': (0, 45),
        'description': 'Adjacent orientations (0° vs 45°)',
        'type': 'adjacent'
    },
    'adjacent_2': {
        'pair': (45, 90),
        'description': 'Adjacent orientations (45° vs 90°)',
        'type': 'adjacent'
    },
    'adjacent_3': {
        'pair': (90, 135),
        'description': 'Adjacent orientations (90° vs 135°)',
        'type': 'adjacent'
    }
}

def load_temporal_data_for_pairwise(npz_file):
    """
    Load temporal data for pairwise analysis
    """
    
    print(f"Loading data from {npz_file}")
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    unit_responses = data['all_units_responses']
    unit_qualities = data['unit_qualities']
    summary_stats = data['summary_stats'].item()
    
    # Filter good units
    good_mask = np.array([qual != 'noise' for qual in unit_qualities])
    unit_responses_good = unit_responses[good_mask]
    n_good_units = good_mask.sum()
    
    print(f"Using {n_good_units} good quality units")
    
    if len(unit_responses_good) == 0:
        raise ValueError("No good quality units found")
    
    # Check for temporal features
    first_unit = unit_responses_good[0]
    has_temporal = 'display_temporal' in first_unit
    
    if not has_temporal:
        raise ValueError("No temporal features found. Please run temporal_spike_processing.py first")
    
    # Get trial information
    trial_data = first_unit['trial_data']
    n_trials = len(trial_data)
    n_time_bins = summary_stats.get('n_time_bins', 20)
    
    # Extract orientations
    left_objects = trial_data['left_object_index'].values
    right_objects = trial_data['right_object_index'].values
    
    left_orientations = np.array([OBJECT_TO_ORIENTATION.get(obj, np.nan) for obj in left_objects])
    right_orientations = np.array([OBJECT_TO_ORIENTATION.get(obj, np.nan) for obj in right_objects])
    
    return {
        'unit_responses': unit_responses_good,
        'n_good_units': n_good_units,
        'n_trials': n_trials,
        'n_time_bins': n_time_bins,
        'left_orientations': left_orientations,
        'right_orientations': right_orientations,
        'trial_data': trial_data
    }

def extract_pairwise_features(unit_responses, orientations, ori1, ori2, response_phase='display', 
                            feature_type='combined'):
    """
    Extract features for pairwise orientation comparison
    
    Parameters:
    unit_responses: List of unit response dictionaries
    orientations: Array of orientation labels for each trial
    ori1, ori2: The two orientations to compare
    response_phase: 'display', 'baseline', 'iti'
    feature_type: 'binned_rates', 'spike_timing', 'combined'
    
    Returns:
    X: Feature matrix for trials with ori1 or ori2
    y: Binary labels (0 for ori1, 1 for ori2)
    trial_indices: Original trial indices
    """
    
    # Find trials with these orientations
    mask_ori1 = orientations == ori1
    mask_ori2 = orientations == ori2
    combined_mask = mask_ori1 | mask_ori2
    
    if np.sum(combined_mask) == 0:
        raise ValueError(f"No trials found for orientations {ori1}° and {ori2}°")
    
    n_trials_selected = np.sum(combined_mask)
    n_units = len(unit_responses)
    
    print(f"  Pairwise {ori1}° vs {ori2}°: {n_trials_selected} trials")
    print(f"    {ori1}°: {np.sum(mask_ori1)} trials")
    print(f"    {ori2}°: {np.sum(mask_ori2)} trials")
    
    # Extract trial indices
    trial_indices = np.where(combined_mask)[0]
    
    # Create binary labels
    y = np.zeros(n_trials_selected)
    y[orientations[combined_mask] == ori2] = 1  # ori2 = class 1, ori1 = class 0
    
    # Extract features
    phase_key = f"{response_phase}_temporal"
    n_time_bins = unit_responses[0][phase_key]['binned_firing_rates'].shape[1]
    
    all_features = []
    
    for unit_idx, unit_data in enumerate(unit_responses):
        temporal_data = unit_data[phase_key]
        
        if feature_type in ['binned_rates', 'combined']:
            # Time-binned firing rates
            binned_rates = temporal_data['binned_firing_rates'][combined_mask]  # Shape: (n_selected_trials, n_time_bins)
            
            # Add each time bin as a feature
            for bin_idx in range(n_time_bins):
                all_features.append(binned_rates[:, bin_idx])
        
        if feature_type in ['spike_timing', 'combined']:
            # Spike timing features
            all_features.append(temporal_data['first_spike_latencies'][combined_mask])
            all_features.append(temporal_data['last_spike_times'][combined_mask])
            all_features.append(temporal_data['spike_time_variances'][combined_mask])
            all_features.append(temporal_data['total_spike_counts'][combined_mask])
            all_features.append(temporal_data['mean_firing_rates'][combined_mask])
    
    # Stack features into matrix
    X = np.column_stack(all_features)
    
    print(f"    Feature matrix: {X.shape}")
    
    return X, y, trial_indices

def perform_pairwise_classification(X, y, ori1, ori2, cv_folds=5, random_state=42):
    """
    Perform binary classification for pairwise orientation comparison
    """
    
    # Remove invalid trials
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    if len(X_clean) == 0:
        print(f"    No valid trials for {ori1}° vs {ori2}°")
        return {}
    
    print(f"    Valid trials: {len(X_clean)}")
    print(f"    Class distribution: {ori1}°={np.sum(y_clean==0)}, {ori2}°={np.sum(y_clean==1)}")
    
    # Check if we have both classes
    if len(np.unique(y_clean)) < 2:
        print(f"    Only one class present, skipping...")
        return {}
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Adjust CV folds
    min_samples = min(np.sum(y_clean == 0), np.sum(y_clean == 1))
    cv_folds = min(cv_folds, min_samples)
    
    print(f"    Using {cv_folds} CV folds")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Test multiple classifiers
    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(clf, X_scaled, y_clean, cv=skf, scoring='accuracy')
            
            # Fit for additional metrics
            clf.fit(X_scaled, y_clean)
            
            # Get predictions and probabilities
            y_pred = cross_val_predict(clf, X_scaled, y_clean, cv=skf)
            
            if hasattr(clf, 'predict_proba'):
                y_proba = cross_val_predict(clf, X_scaled, y_clean, cv=skf, method='predict_proba')[:, 1]
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_clean, y_proba)
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr, roc_auc = None, None, None
                y_proba = None
            
            # Statistical test vs chance (50%)
            chance_level = 0.5
            t_stat, p_value = ttest_1samp(cv_scores, chance_level)
            
            results[clf_name] = {
                'cv_scores': cv_scores,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'y_true': y_clean,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'chance_level': chance_level,
                't_stat': t_stat,
                'p_value': p_value,
                'above_chance': p_value < 0.05 and cv_scores.mean() > chance_level,
                'classifier': clf
            }
            
            # Fixed print statement - separate the AUC formatting
            auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "0.000"
            print(f"      {clf_name}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} "
                  f"(p={p_value:.3f}, AUC={auc_str})")
            
        except Exception as e:
            print(f"      Error with {clf_name}: {e}")
    
    return results

def create_pairwise_summary_figure(all_pairwise_results, session_info):
    """
    Create comprehensive summary figure for all pairwise comparisons
    """
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Collect all results for visualization
    comparison_names = []
    comparison_types = []
    left_vf_accs = []
    right_vf_accs = []
    left_vf_aucs = []
    right_vf_aucs = []
    left_vf_sig = []
    right_vf_sig = []
    
    for comp_name, comp_data in all_pairwise_results.items():
        if 'left_vf' in comp_data and 'right_vf' in comp_data:
            comparison_names.append(comp_name)
            comparison_types.append(PAIRWISE_COMPARISONS[comp_name]['type'])
            
            # Get best accuracy for each VF
            left_best_acc = 0
            right_best_acc = 0
            left_best_auc = 0
            right_best_auc = 0
            left_sig = False
            right_sig = False
            
            for method in ['SVM', 'Random Forest', 'Logistic Regression']:
                if method in comp_data['left_vf']:
                    acc = comp_data['left_vf'][method]['mean_accuracy']
                    auc_val = comp_data['left_vf'][method].get('roc_auc', 0) or 0
                    sig = comp_data['left_vf'][method]['above_chance']
                    if acc > left_best_acc:
                        left_best_acc = acc
                        left_best_auc = auc_val
                        left_sig = sig
                
                if method in comp_data['right_vf']:
                    acc = comp_data['right_vf'][method]['mean_accuracy']
                    auc_val = comp_data['right_vf'][method].get('roc_auc', 0) or 0
                    sig = comp_data['right_vf'][method]['above_chance']
                    if acc > right_best_acc:
                        right_best_acc = acc
                        right_best_auc = auc_val
                        right_sig = sig
            
            left_vf_accs.append(left_best_acc)
            right_vf_accs.append(right_best_acc)
            left_vf_aucs.append(left_best_auc)
            right_vf_aucs.append(right_best_auc)
            left_vf_sig.append(left_sig)
            right_vf_sig.append(right_sig)
    
    # Plot 1: Accuracy comparison across all pairs
    ax1 = fig.add_subplot(gs[0, :2])
    
    if comparison_names:
        x = np.arange(len(comparison_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, left_vf_accs, width, label='Left VF', alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, right_vf_accs, width, label='Right VF', alpha=0.8, color='red')
        
        # Add significance markers
        for i, (bar1, bar2, sig_left, sig_right) in enumerate(zip(bars1, bars2, left_vf_sig, right_vf_sig)):
            if sig_left:
                ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                        '*', ha='center', va='bottom', fontsize=14, color='blue', fontweight='bold')
            if sig_right:
                ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                        '*', ha='center', va='bottom', fontsize=14, color='red', fontweight='bold')
        
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (50%)')
        ax1.set_xlabel('Pairwise Comparison')
        ax1.set_ylabel('Best Accuracy')
        ax1.set_title('Pairwise Orientation Decoding Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.replace('_', '\n') for name in comparison_names], rotation=0, ha='center')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bars, accs in [(bars1, left_vf_accs), (bars2, right_vf_accs)]:
            for bar, acc in zip(bars, accs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: ROC AUC comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if comparison_names and any(auc > 0 for auc in left_vf_aucs + right_vf_aucs):
        bars1 = ax2.bar(x - width/2, left_vf_aucs, width, label='Left VF', alpha=0.8, color='lightblue')
        bars2 = ax2.bar(x + width/2, right_vf_aucs, width, label='Right VF', alpha=0.8, color='lightcoral')
        
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (50%)')
        ax2.set_xlabel('Pairwise Comparison')
        ax2.set_ylabel('Best ROC AUC')
        ax2.set_title('ROC AUC Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name.replace('_', '\n') for name in comparison_names], rotation=0, ha='center')
        ax2.legend()
        ax2.set_ylim(0, 1)
    
    # Plot 3: Comparison type analysis
    ax3 = fig.add_subplot(gs[1, :2])
    
    if comparison_types:
        # Group by comparison type
        type_groups = {}
        for i, comp_type in enumerate(comparison_types):
            if comp_type not in type_groups:
                type_groups[comp_type] = {'left': [], 'right': []}
            type_groups[comp_type]['left'].append(left_vf_accs[i])
            type_groups[comp_type]['right'].append(right_vf_accs[i])
        
        types = list(type_groups.keys())
        left_means = [np.mean(type_groups[t]['left']) for t in types]
        right_means = [np.mean(type_groups[t]['right']) for t in types]
        left_stds = [np.std(type_groups[t]['left']) for t in types]
        right_stds = [np.std(type_groups[t]['right']) for t in types]
        
        x_types = np.arange(len(types))
        bars1 = ax3.bar(x_types - width/2, left_means, width, yerr=left_stds, 
                       label='Left VF', alpha=0.8, color='blue', capsize=5)
        bars2 = ax3.bar(x_types + width/2, right_means, width, yerr=right_stds,
                       label='Right VF', alpha=0.8, color='red', capsize=5)
        
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Comparison Type')
        ax3.set_ylabel('Mean Accuracy')
        ax3.set_title('Performance by Comparison Type')
        ax3.set_xticks(x_types)
        ax3.set_xticklabels(types)
        ax3.legend()
        ax3.set_ylim(0, 1)
    
    # Plot 4: Best vs worst pairs
    ax4 = fig.add_subplot(gs[1, 2:])
    
    if comparison_names:
        all_accs = left_vf_accs + right_vf_accs
        all_names = [f"Left {name}" for name in comparison_names] + [f"Right {name}" for name in comparison_names]
        
        best_idx = np.argmax(all_accs)
        worst_idx = np.argmin(all_accs)
        
        bars = ax4.bar(['Best Pair', 'Worst Pair'], 
                      [all_accs[best_idx], all_accs[worst_idx]], 
                      color=['green', 'orange'], alpha=0.7)
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Accuracy')
        ax4.set_title(f'Best: {all_names[best_idx]}\nWorst: {all_names[worst_idx]}')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, [all_accs[best_idx], all_accs[worst_idx]]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: ROC curves for best performing pair
    ax5 = fig.add_subplot(gs[2, :2])
    
    if comparison_names:
        # Find best performing pair overall
        best_overall_idx = np.argmax([max(left_vf_accs[i], right_vf_accs[i]) for i in range(len(comparison_names))])
        best_comp_name = comparison_names[best_overall_idx]
        
        if best_comp_name in all_pairwise_results:
            best_results = all_pairwise_results[best_comp_name]
            
            # Plot ROC curves for best method in each VF
            for vf_name, vf_results in [('Left VF', best_results.get('left_vf', {})), 
                                       ('Right VF', best_results.get('right_vf', {}))]:
                best_method = None
                best_acc = 0
                
                for method, method_results in vf_results.items():
                    if method_results.get('mean_accuracy', 0) > best_acc:
                        best_acc = method_results['mean_accuracy']
                        best_method = method
                
                if best_method and 'fpr' in vf_results[best_method] and vf_results[best_method]['fpr'] is not None:
                    fpr = vf_results[best_method]['fpr']
                    tpr = vf_results[best_method]['tpr']
                    auc_val = vf_results[best_method]['roc_auc']
                    
                    color = 'blue' if 'Left' in vf_name else 'red'
                    ax5.plot(fpr, tpr, color=color, linewidth=2, 
                            label=f'{vf_name} ({best_method}) AUC={auc_val:.3f}')
            
            ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
            ax5.set_xlabel('False Positive Rate')
            ax5.set_ylabel('True Positive Rate')
            ax5.set_title(f'ROC Curves - Best Pair: {best_comp_name}')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
    
    # Plot 6: Statistical summary
    ax6 = fig.add_subplot(gs[2, 2:])
    
    # Create statistical summary
    if comparison_names:
        summary_lines = []
        summary_lines.append("PAIRWISE DECODING SUMMARY:")
        summary_lines.append("")
        
        for i, comp_name in enumerate(comparison_names):
            pair_info = PAIRWISE_COMPARISONS[comp_name]
            ori1, ori2 = pair_info['pair']
            
            summary_lines.append(f"{comp_name.upper()}: {ori1}° vs {ori2}°")
            summary_lines.append(f"  Left VF:  {left_vf_accs[i]:.3f} {'*' if left_vf_sig[i] else ''}")
            summary_lines.append(f"  Right VF: {right_vf_accs[i]:.3f} {'*' if right_vf_sig[i] else ''}")
            summary_lines.append("")
        
        summary_lines.append("* = Significantly above chance (p < 0.05)")
        
        summary_text = "\n".join(summary_lines)
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    # Plot 7: Overall interpretation
    ax7 = fig.add_subplot(gs[3, :])
    
    if comparison_names:
        # Overall analysis
        interpretation_lines = []
        interpretation_lines.append("INTERPRETATION:")
        interpretation_lines.append("")
        
        # Best performing pair
        best_overall = max([max(left_vf_accs[i], right_vf_accs[i]) for i in range(len(comparison_names))])
        best_pair_idx = np.argmax([max(left_vf_accs[i], right_vf_accs[i]) for i in range(len(comparison_names))])
        best_pair_name = comparison_names[best_pair_idx]
        best_pair_info = PAIRWISE_COMPARISONS[best_pair_name]
        
        interpretation_lines.append(f"BEST DISCRIMINATION: {best_pair_name} ({best_pair_info['pair'][0]}° vs {best_pair_info['pair'][1]}°)")
        interpretation_lines.append(f"  Performance: {best_overall:.3f}")
        interpretation_lines.append(f"  Type: {best_pair_info['type']}")
        interpretation_lines.append("")
        
        # Count significant pairs
        n_sig_left = sum(left_vf_sig)
        n_sig_right = sum(right_vf_sig)
        
        interpretation_lines.append(f"SIGNIFICANT PAIRS:")
        interpretation_lines.append(f"  Left VF: {n_sig_left}/{len(comparison_names)} pairs")
        interpretation_lines.append(f"  Right VF: {n_sig_right}/{len(comparison_names)} pairs")
        interpretation_lines.append("")
        
        # Type analysis
        if 'cardinal' in comparison_types and 'oblique' in comparison_types:
            cardinal_idx = [i for i, t in enumerate(comparison_types) if t == 'cardinal']
            oblique_idx = [i for i, t in enumerate(comparison_types) if t == 'oblique']
            
            if cardinal_idx and oblique_idx:
                cardinal_mean = np.mean([max(left_vf_accs[i], right_vf_accs[i]) for i in cardinal_idx])
                oblique_mean = np.mean([max(left_vf_accs[i], right_vf_accs[i]) for i in oblique_idx])
                
                interpretation_lines.append(f"CARDINAL vs OBLIQUE:")
                interpretation_lines.append(f"  Cardinal (0°/90°): {cardinal_mean:.3f}")
                interpretation_lines.append(f"  Oblique (45°/135°): {oblique_mean:.3f}")
                interpretation_lines.append(f"  Advantage: {'Cardinal' if cardinal_mean > oblique_mean else 'Oblique'}")
        
        interpretation_text = "\n".join(interpretation_lines)
        
        ax7.text(0.05, 0.95, interpretation_text, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    
    plt.suptitle(f"Pairwise Orientation Decoding Analysis\n"
                f"Animal: {session_info['animal_id']}, Session: {session_info['session_id']}", 
                fontsize=16, y=0.98)
    
    return fig

def pairwise_orientation_decoding_analysis(npz_file, response_phases=['display'], 
                                         feature_types=['combined'], cv_folds=5, random_state=42):
    """
    Main function for pairwise orientation decoding analysis
    """
    
    npz_file = Path(npz_file)
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name
    
    # Create output directory
    decode_folder = session_folder / "pairwise_decoding"
    decode_folder.mkdir(exist_ok=True)
    
    print(f"PAIRWISE ORIENTATION DECODING ANALYSIS")
    print("=" * 60)
    print("Testing all meaningful orientation pairs:")
    for comp_name, comp_info in PAIRWISE_COMPARISONS.items():
        ori1, ori2 = comp_info['pair']
        print(f"  {comp_name}: {ori1}° vs {ori2}° ({comp_info['description']})")
    print("=" * 60)
    
    # Load data
    try:
        data_dict = load_temporal_data_for_pairwise(npz_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    session_info = {
        'animal_id': animal_id,
        'session_id': session_id,
        'n_units': data_dict['n_good_units']
    }
    
    all_pairwise_results = {}
    
    # Test each pairwise comparison
    for response_phase in response_phases:
        for feature_type in feature_types:
            
            print(f"\n{'='*50}")
            print(f"Testing: {feature_type} features, {response_phase} phase")
            print('='*50)
            
            phase_results = {}
            
            for comp_name, comp_info in PAIRWISE_COMPARISONS.items():
                ori1, ori2 = comp_info['pair']
                
                print(f"\nTesting {comp_name}: {ori1}° vs {ori2}°")
                print(f"Description: {comp_info['description']}")
                
                try:
                    # Test left visual field
                    print(f"  Left Visual Field:")
                    X_left, y_left, _ = extract_pairwise_features(
                        data_dict['unit_responses'], 
                        data_dict['left_orientations'], 
                        ori1, ori2, response_phase, feature_type
                    )
                    
                    results_left = perform_pairwise_classification(
                        X_left, y_left, ori1, ori2, cv_folds, random_state
                    )
                    
                    # Test right visual field
                    print(f"  Right Visual Field:")
                    X_right, y_right, _ = extract_pairwise_features(
                        data_dict['unit_responses'], 
                        data_dict['right_orientations'], 
                        ori1, ori2, response_phase, feature_type
                    )
                    
                    results_right = perform_pairwise_classification(
                        X_right, y_right, ori1, ori2, cv_folds, random_state
                    )
                    
                    # Store results
                    phase_results[comp_name] = {
                        'left_vf': results_left,
                        'right_vf': results_right,
                        'comparison_info': comp_info,
                        'feature_type': feature_type,
                        'response_phase': response_phase
                    }
                    
                except Exception as e:
                    print(f"  Error processing {comp_name}: {e}")
                    continue
            
            # Store phase results
            config_key = f"{feature_type}_{response_phase}"
            all_pairwise_results[config_key] = phase_results
    
    # Create summary figure for each configuration
    for config_key, config_results in all_pairwise_results.items():
        if config_results:  # Only if we have results
            
            print(f"\nCreating summary figure for {config_key}...")
            
            try:
                fig = create_pairwise_summary_figure(config_results, session_info)
                
                # Save figure
                output_file = decode_folder / f"pairwise_summary_{config_key}.png"
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Figure saved: {output_file}")
                
            except Exception as e:
                print(f"Error creating figure for {config_key}: {e}")
    
    # Create detailed individual comparison figures
    print(f"\nCreating detailed individual comparison figures...")
    
    for config_key, config_results in all_pairwise_results.items():
        for comp_name, comp_data in config_results.items():
            
            try:
                # Create detailed figure for this specific comparison
                fig = create_detailed_pairwise_figure(comp_name, comp_data, session_info)
                
                output_file = decode_folder / f"detailed_{comp_name}_{config_key}.png"
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Error creating detailed figure for {comp_name}: {e}")
    
    # Save comprehensive results
    results_file = decode_folder / "pairwise_results.npz"
    np.savez(results_file, all_results=all_pairwise_results, session_info=session_info, allow_pickle=True)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PAIRWISE DECODING SUMMARY")
    print("=" * 60)
    
    for config_key, config_results in all_pairwise_results.items():
        print(f"\nConfiguration: {config_key}")
        
        best_performance = 0
        best_pair = ""
        
        for comp_name, comp_data in config_results.items():
            # Find best performance for this pair
            pair_best = 0
            
            for vf_name in ['left_vf', 'right_vf']:
                if vf_name in comp_data:
                    for method, method_results in comp_data[vf_name].items():
                        acc = method_results.get('mean_accuracy', 0)
                        if acc > pair_best:
                            pair_best = acc
            
            print(f"  {comp_name}: {pair_best:.3f}")
            
            if pair_best > best_performance:
                best_performance = pair_best
                best_pair = comp_name
        
        print(f"  BEST: {best_pair} ({best_performance:.3f})")
    
    return all_pairwise_results


def create_detailed_pairwise_figure(comp_name, comp_data, session_info):
    """
    Create detailed figure for a specific pairwise comparison
    """
    
    comp_info = comp_data['comparison_info']
    ori1, ori2 = comp_info['pair']
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    methods = []
    left_accs = []
    right_accs = []
    left_stds = []
    right_stds = []
    
    for method in ['SVM', 'Random Forest', 'Logistic Regression']:
        if (method in comp_data.get('left_vf', {}) and 
            method in comp_data.get('right_vf', {})):
            methods.append(method)
            left_accs.append(comp_data['left_vf'][method]['mean_accuracy'])
            right_accs.append(comp_data['right_vf'][method]['mean_accuracy'])
            left_stds.append(comp_data['left_vf'][method]['std_accuracy'])
            right_stds.append(comp_data['right_vf'][method]['std_accuracy'])
    
    if methods:
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, left_accs, width, yerr=left_stds, 
                       label='Left VF', alpha=0.8, color='blue', capsize=5)
        bars2 = ax1.bar(x + width/2, right_accs, width, yerr=right_stds,
                       label='Right VF', alpha=0.8, color='red', capsize=5)
        
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (50%)')
        ax1.set_xlabel('Classification Method')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{comp_name}: {ori1}° vs {ori2}°\n{comp_info["description"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bars, accs in [(bars1, left_accs), (bars2, right_accs)]:
            for bar, acc in zip(bars, accs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # ROC curves
    ax2 = fig.add_subplot(gs[0, 2:])
    
    for vf_name, vf_data in [('Left VF', comp_data.get('left_vf', {})), 
                           ('Right VF', comp_data.get('right_vf', {}))]:
        for method, method_results in vf_data.items():
            if 'fpr' in method_results and method_results['fpr'] is not None:
                fpr = method_results['fpr']
                tpr = method_results['tpr']
                auc_val = method_results['roc_auc']
                
                color = 'blue' if 'Left' in vf_name else 'red'
                linestyle = '-' if method == 'SVM' else '--' if method == 'Random Forest' else ':'
                
                ax2.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2, 
                        label=f'{vf_name} {method} (AUC={auc_val:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrices
    if 'SVM' in comp_data.get('left_vf', {}):
        ax3 = fig.add_subplot(gs[1, 0])
        plot_pairwise_confusion_matrix(comp_data['left_vf']['SVM'], ax3, f'Left VF - SVM\n{ori1}° vs {ori2}°', ori1, ori2)
    
    if 'SVM' in comp_data.get('right_vf', {}):
        ax4 = fig.add_subplot(gs[1, 1])
        plot_pairwise_confusion_matrix(comp_data['right_vf']['SVM'], ax4, f'Right VF - SVM\n{ori1}° vs {ori2}°', ori1, ori2)
    
    # CV score comparison
    ax5 = fig.add_subplot(gs[1, 2:])
    
    if methods:
        # Find best method for each VF
        best_left_method = methods[np.argmax(left_accs)] if left_accs else None
        best_right_method = methods[np.argmax(right_accs)] if right_accs else None
        
        if best_left_method and best_left_method in comp_data.get('left_vf', {}):
            left_cv_scores = comp_data['left_vf'][best_left_method]['cv_scores']
            ax5.plot(left_cv_scores, 'bo-', label=f'Left VF ({best_left_method})', markersize=8)
        
        if best_right_method and best_right_method in comp_data.get('right_vf', {}):
            right_cv_scores = comp_data['right_vf'][best_right_method]['cv_scores']
            ax5.plot(right_cv_scores, 'ro-', label=f'Right VF ({best_right_method})', markersize=8)
        
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
        ax5.set_xlabel('CV Fold')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Best Method CV Performance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
    
    # Statistical summary
    ax6 = fig.add_subplot(gs[2, :])
    
    summary_lines = []
    summary_lines.append(f"STATISTICAL SUMMARY: {ori1}° vs {ori2}°")
    summary_lines.append(f"Comparison Type: {comp_info['type']}")
    summary_lines.append(f"Description: {comp_info['description']}")
    summary_lines.append("")
    
    for method in methods:
        if (method in comp_data.get('left_vf', {}) and 
            method in comp_data.get('right_vf', {})):
            
            left_result = comp_data['left_vf'][method]
            right_result = comp_data['right_vf'][method]
            
            summary_lines.append(f"{method}:")
            summary_lines.append(f"  Left VF:  {left_result['mean_accuracy']:.3f} ± {left_result['std_accuracy']:.3f} "
                                f"(p={left_result['p_value']:.3f}) {'*' if left_result['above_chance'] else ''}")
            summary_lines.append(f"  Right VF: {right_result['mean_accuracy']:.3f} ± {right_result['std_accuracy']:.3f} "
                                f"(p={right_result['p_value']:.3f}) {'*' if right_result['above_chance'] else ''}")
            
            if left_result.get('roc_auc') and right_result.get('roc_auc'):
                summary_lines.append(f"  ROC AUC:  Left={left_result['roc_auc']:.3f}, Right={right_result['roc_auc']:.3f}")
            summary_lines.append("")
    
    summary_lines.append("* = Significantly above chance (p < 0.05)")
    summary_lines.append("")
    
    # Interpretation
    if methods:
        best_overall_acc = max(max(left_accs) if left_accs else 0, max(right_accs) if right_accs else 0)
        
        summary_lines.append("INTERPRETATION:")
        if best_overall_acc > 0.75:
            summary_lines.append(f"  Excellent discrimination between {ori1}° and {ori2}° (>75%)")
        elif best_overall_acc > 0.65:
            summary_lines.append(f"  Good discrimination between {ori1}° and {ori2}° (>65%)")
        elif best_overall_acc > 0.55:
            summary_lines.append(f"  Moderate discrimination between {ori1}° and {ori2}° (>55%)")
        else:
            summary_lines.append(f"  Poor discrimination between {ori1}° and {ori2}° (<55%)")
        
        if abs(max(left_accs) - max(right_accs)) > 0.1:
            better_vf = "Left" if max(left_accs) > max(right_accs) else "Right"
            summary_lines.append(f"  {better_vf} visual field shows better discrimination")
        else:
            summary_lines.append("  Similar performance in both visual fields")
    
    summary_text = "\n".join(summary_lines)
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.suptitle(f"Detailed Pairwise Analysis: {ori1}° vs {ori2}°\n"
                f"Animal: {session_info['animal_id']}, Session: {session_info['session_id']}", 
                fontsize=14, y=0.98)
    
    return fig


def plot_pairwise_confusion_matrix(results, ax, title, ori1, ori2):
    """Plot confusion matrix for pairwise comparison"""
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Labels: 0 = ori1, 1 = ori2
    labels = [f'{ori1}°', f'{ori2}°']
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)


if __name__ == '__main__':
    # NPZ file with temporal features
    npz_file = r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22sg\CnL22SG_20250609_164650\object_discrimination_temporal_responses.npz"
    
    print("PAIRWISE ORIENTATION DECODING ANALYSIS")
    print("=" * 60)
    print("Testing meaningful orientation pairs:")
    print("• Orthogonal: 0° vs 90° (horizontal vs vertical)")
    print("• Oblique: 45° vs 135° (diagonal orientations)")
    print("• Adjacent: consecutive 45° steps")
    print("• Cardinal vs Oblique: mixed pairs")
    print("=" * 60)
    
    # Run comprehensive pairwise analysis
    try:
        all_results = pairwise_orientation_decoding_analysis(
            npz_file,
            response_phases=['display', 'baseline'],  # Test display and baseline
            feature_types=['combined'],  # Use combined temporal features
            cv_folds=5,
            random_state=42
        )
        
        if all_results:
            print(f"\n{'='*60}")
            print("PAIRWISE ANALYSIS COMPLETE!")
            print("=" * 60)
            
            # Print key findings
            best_overall = 0
            best_config = ""
            best_pair = ""
            
            for config_key, config_results in all_results.items():
                for comp_name, comp_data in config_results.items():
                    for vf_name in ['left_vf', 'right_vf']:
                        if vf_name in comp_data:
                            for method, method_results in comp_data[vf_name].items():
                                acc = method_results.get('mean_accuracy', 0)
                                if acc > best_overall:
                                    best_overall = acc
                                    best_config = config_key
                                    best_pair = comp_name
                                    best_vf = vf_name
                                    best_method = method
            
            print(f"BEST PAIRWISE DISCRIMINATION:")
            print(f"  Pair: {best_pair}")
            print(f"  Config: {best_config}")
            print(f"  VF: {best_vf}, Method: {best_method}")
            print(f"  Accuracy: {best_overall:.3f}")
            
            decode_folder = Path(npz_file).parent / "pairwise_decoding"
            print(f"\nAll results saved in: {decode_folder}")
            
        else:
            print("No results generated. Check your data and try again.")
    
    except Exception as e:
        print(f"Error in pairwise analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("PAIRWISE ANALYSIS COMPLETE!")
    print("Check the 'pairwise_decoding' folder for detailed results!")
    print("=" * 60)