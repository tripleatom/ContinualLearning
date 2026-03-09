import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.stats import ttest_rel, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

# Object to orientation mapping
OBJECT_TO_ORIENTATION = {14: 0, 15: 45, 16: 90, 17: 135}

def load_temporal_data(npz_file):
    """
    Load temporal spike data from NPZ file
    
    Returns:
    dict with temporal features and trial information
    """
    
    print(f"Loading temporal data from {npz_file}")
    
    # Load data with robust trial data handling
    data = np.load(npz_file, allow_pickle=True)
    unit_responses = data['all_units_responses']
    unit_qualities = data['unit_qualities']
    summary_stats = data['summary_stats'].item()
    
    print(f"Loaded {len(unit_responses)} units")
    print(f"Has temporal features: {summary_stats.get('has_temporal_features', False)}")
    
    # Filter good units
    good_mask = np.array([qual != 'noise' for qual in unit_qualities])
    unit_responses_good = unit_responses[good_mask]
    n_good_units = good_mask.sum()
    
    print(f"Using {n_good_units} good quality units")
    
    # Check if we have temporal features
    if len(unit_responses_good) == 0:
        raise ValueError("No good quality units found")
    
    first_unit = unit_responses_good[0]
    has_temporal = 'display_temporal' in first_unit
    
    if not has_temporal:
        raise ValueError("No temporal features found. Please run temporal_spike_processing.py first")
    
    # Get trial information
    trial_data = first_unit['trial_data']
    n_trials = len(trial_data)
    n_time_bins = summary_stats.get('n_time_bins', 20)
    
    print(f"Number of trials: {n_trials}")
    print(f"Time bins per trial: {n_time_bins}")
    
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
        'trial_data': trial_data,
        'summary_stats': summary_stats
    }

def extract_temporal_feature_matrix(unit_responses, feature_type='binned_rates', response_phase='display'):
    """
    Extract temporal feature matrix from unit responses
    
    Parameters:
    unit_responses: List of unit response dictionaries
    feature_type: 'binned_rates', 'spike_timing', 'combined', 'all'
    response_phase: 'display', 'baseline', 'iti'
    
    Returns:
    feature_matrix: (n_trials, n_features)
    feature_names: List of feature names
    """
    
    n_units = len(unit_responses)
    
    # Get dimensions from first unit
    phase_key = f"{response_phase}_temporal"
    first_unit_temporal = unit_responses[0][phase_key]
    n_trials = len(first_unit_temporal['mean_firing_rates'])
    n_time_bins = first_unit_temporal['binned_firing_rates'].shape[1]
    
    print(f"Extracting {feature_type} features for {response_phase} phase")
    print(f"  {n_units} units, {n_trials} trials, {n_time_bins} time bins")
    
    all_features = []
    feature_names = []
    
    for unit_idx, unit_data in enumerate(unit_responses):
        temporal_data = unit_data[phase_key]
        
        if feature_type in ['binned_rates', 'combined', 'all']:
            # Time-binned firing rates
            binned_rates = temporal_data['binned_firing_rates']  # Shape: (n_trials, n_time_bins)
            
            # Add each time bin as a separate feature
            for bin_idx in range(n_time_bins):
                all_features.append(binned_rates[:, bin_idx])
                feature_names.append(f'unit_{unit_idx}_bin_{bin_idx}_rate')
        
        if feature_type in ['spike_timing', 'combined', 'all']:
            # Spike timing features
            all_features.append(temporal_data['first_spike_latencies'])
            feature_names.append(f'unit_{unit_idx}_first_latency')
            
            all_features.append(temporal_data['last_spike_times'])
            feature_names.append(f'unit_{unit_idx}_last_time')
            
            all_features.append(temporal_data['spike_time_variances'])
            feature_names.append(f'unit_{unit_idx}_time_variance')
            
            all_features.append(temporal_data['total_spike_counts'])
            feature_names.append(f'unit_{unit_idx}_spike_count')
        
        if feature_type in ['all']:
            # Additional derived features
            
            # Peak firing time (bin with maximum rate)
            binned_rates = temporal_data['binned_firing_rates']
            peak_times = np.argmax(binned_rates, axis=1) / n_time_bins  # Normalized peak time
            all_features.append(peak_times)
            feature_names.append(f'unit_{unit_idx}_peak_time')
            
            # Temporal modulation (max - min rate)
            temporal_modulation = np.max(binned_rates, axis=1) - np.min(binned_rates, axis=1)
            all_features.append(temporal_modulation)
            feature_names.append(f'unit_{unit_idx}_temporal_mod')
            
            # Firing rate slope (early vs late)
            early_rate = np.mean(binned_rates[:, :n_time_bins//3], axis=1)  # First third
            late_rate = np.mean(binned_rates[:, -n_time_bins//3:], axis=1)  # Last third
            rate_slope = late_rate - early_rate
            all_features.append(rate_slope)
            feature_names.append(f'unit_{unit_idx}_rate_slope')
    
    # Stack features into matrix
    feature_matrix = np.column_stack(all_features)
    
    print(f"  Created feature matrix: {feature_matrix.shape}")
    print(f"  Feature types: {len(feature_names)} features")
    
    return feature_matrix, feature_names

def perform_temporal_classification(X, y, method='all', cv_folds=5, random_state=42):
    """
    Perform classification using multiple methods on temporal features
    """
    
    results = {}
    
    # Remove invalid trials
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    if len(np.unique(y_clean)) < 2:
        print("Insufficient classes for classification")
        return {}
    
    print(f"Classification data: {X_clean.shape[0]} trials, {X_clean.shape[1]} features")
    print(f"Orientations: {np.unique(y_clean)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Encode labels
    unique_orientations = np.unique(y_clean)
    orientation_to_idx = {ori: idx for idx, ori in enumerate(unique_orientations)}
    y_encoded = np.array([orientation_to_idx[ori] for ori in y_clean])
    
    # Adjust CV folds
    min_samples = np.min(np.bincount(y_encoded))
    cv_folds = min(cv_folds, min_samples)
    
    print(f"Using {cv_folds} CV folds, min samples per class: {min_samples}")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Define classifiers
    classifiers = {}
    if method in ['svm', 'all']:
        classifiers['SVM'] = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state)
    if method in ['rf', 'all']:
        classifiers['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=random_state)
    if method in ['mlp', 'all']:
        classifiers['Neural Network'] = MLPClassifier(hidden_layer_sizes=(100, 50), 
                                                     max_iter=1000, random_state=random_state)
    if method in ['lr', 'all']:
        classifiers['Logistic Regression'] = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Test each classifier
    for clf_name, clf in classifiers.items():
        print(f"\nTesting {clf_name}...")
        
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=skf, scoring='accuracy')
            
            # Fit for additional metrics
            clf.fit(X_scaled, y_encoded)
            
            # Get predictions for confusion matrix
            y_pred = cross_val_predict(clf, X_scaled, y_encoded, cv=skf)
            
            # Statistical test vs chance
            chance_level = 1.0 / len(unique_orientations)
            t_stat, p_value = ttest_1samp(cv_scores, chance_level)
            
            results[clf_name] = {
                'cv_scores': cv_scores,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'y_true': y_encoded,
                'y_pred': y_pred,
                'classifier': clf,
                'orientations': unique_orientations,
                'chance_level': chance_level,
                't_stat': t_stat,
                'p_value': p_value,
                'above_chance': p_value < 0.05 and cv_scores.mean() > chance_level
            }
            
            print(f"  {clf_name} Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  vs Chance ({chance_level:.3f}): t={t_stat:.2f}, p={p_value:.3f}")
            print(f"  Significantly above chance: {results[clf_name]['above_chance']}")
            
            # Feature importance for tree-based methods
            if hasattr(clf, 'feature_importances_'):
                results[clf_name]['feature_importance'] = clf.feature_importances_
            
        except Exception as e:
            print(f"  Error with {clf_name}: {e}")
    
    return results

def create_temporal_decoding_figure(results_left, results_right, session_info, response_type, feature_type):
    """
    Create comprehensive figure for temporal decoding results
    """
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Top row: Performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Collect results for plotting
    methods = []
    left_accs = []
    right_accs = []
    left_stds = []
    right_stds = []
    left_significant = []
    right_significant = []
    
    all_methods = ['SVM', 'Random Forest', 'Neural Network', 'Logistic Regression']
    for method in all_methods:
        if method in results_left and method in results_right:
            methods.append(method)
            left_accs.append(results_left[method]['mean_accuracy'])
            right_accs.append(results_right[method]['mean_accuracy'])
            left_stds.append(results_left[method]['std_accuracy'])
            right_stds.append(results_right[method]['std_accuracy'])
            left_significant.append(results_left[method]['above_chance'])
            right_significant.append(results_right[method]['above_chance'])
    
    if methods:
        x = np.arange(len(methods))
        width = 0.35
        
        # Create bars with significance markers
        bars1 = ax1.bar(x - width/2, left_accs, width, yerr=left_stds, 
                       label='Left Visual Field', alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, right_accs, width, yerr=right_stds,
                       label='Right Visual Field', alpha=0.8, color='red')
        
        # Add significance markers
        for i, (bar1, bar2, sig_left, sig_right) in enumerate(zip(bars1, bars2, left_significant, right_significant)):
            if sig_left:
                ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + left_stds[i] + 0.02,
                        '*', ha='center', va='bottom', fontsize=16, fontweight='bold', color='blue')
            if sig_right:
                ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + right_stds[i] + 0.02,
                        '*', ha='center', va='bottom', fontsize=16, fontweight='bold', color='red')
        
        # Add chance level and labels
        chance_level = results_left[methods[0]]['chance_level'] if methods else 0.25
        ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Chance ({chance_level:.3f})')
        
        ax1.set_xlabel('Classification Method')
        ax1.set_ylabel('Decoding Accuracy')
        ax1.set_title(f'Temporal Decoding Performance\n{feature_type} features, {response_type}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bars, accs in [(bars1, left_accs), (bars2, right_accs)]:
            for bar, acc in zip(bars, accs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Performance vs chance comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if methods:
        # Find best method for each visual field
        best_left_idx = np.argmax(left_accs) if left_accs else 0
        best_right_idx = np.argmax(right_accs) if right_accs else 0
        
        if best_left_idx < len(methods) and best_right_idx < len(methods):
            best_left_method = methods[best_left_idx]
            best_right_method = methods[best_right_idx]
            
            # Plot CV scores for best methods
            if best_left_method in results_left:
                left_cv_scores = results_left[best_left_method]['cv_scores']
                ax2.plot(left_cv_scores, 'bo-', label=f'Left VF ({best_left_method})', markersize=8)
            
            if best_right_method in results_right:
                right_cv_scores = results_right[best_right_method]['cv_scores']
                ax2.plot(right_cv_scores, 'ro-', label=f'Right VF ({best_right_method})', markersize=8)
            
            ax2.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7, label='Chance')
            ax2.set_xlabel('CV Fold')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Best Method CV Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
    
    # Second row: Confusion matrices
    if methods and 'SVM' in results_left:
        ax3 = fig.add_subplot(gs[1, 0])
        plot_temporal_confusion_matrix(results_left['SVM'], ax3, 'Left VF - SVM')
        
        ax4 = fig.add_subplot(gs[1, 1])
        plot_temporal_confusion_matrix(results_right['SVM'], ax4, 'Right VF - SVM')
    
    # Feature importance (if available)
    if methods and 'Random Forest' in results_left and 'feature_importance' in results_left['Random Forest']:
        ax5 = fig.add_subplot(gs[1, 2:])
        plot_temporal_feature_importance(results_left['Random Forest'], results_right['Random Forest'], ax5)
    
    # Third row: Statistical analysis
    ax6 = fig.add_subplot(gs[2, :2])
    
    if methods:
        # Create performance summary table
        summary_text = "STATISTICAL SUMMARY:\n\n"
        for i, method in enumerate(methods):
            if method in results_left and method in results_right:
                left_result = results_left[method]
                right_result = results_right[method]
                
                summary_text += f"{method}:\n"
                summary_text += f"  Left VF:  {left_result['mean_accuracy']:.3f} ± {left_result['std_accuracy']:.3f}"
                summary_text += f" {'*' if left_result['above_chance'] else ''}\n"
                summary_text += f"  Right VF: {right_result['mean_accuracy']:.3f} ± {right_result['std_accuracy']:.3f}"
                summary_text += f" {'*' if right_result['above_chance'] else ''}\n"
                summary_text += f"  p-values: L={left_result['p_value']:.3f}, R={right_result['p_value']:.3f}\n\n"
        
        summary_text += f"* = Significantly above chance (p < 0.05)\n"
        summary_text += f"Chance level: {chance_level:.3f}"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Statistical Results')
    
    # Best vs worst comparison
    ax7 = fig.add_subplot(gs[2, 2:])
    
    if methods and len(left_accs) > 1:
        # Compare best and worst methods
        all_accs = left_accs + right_accs
        all_labels = [f"Left {m}" for m in methods] + [f"Right {m}" for m in methods]
        
        best_idx = np.argmax(all_accs)
        worst_idx = np.argmin(all_accs)
        
        ax7.bar(['Best Method', 'Worst Method'], 
               [all_accs[best_idx], all_accs[worst_idx]], 
               color=['green', 'orange'], alpha=0.7)
        ax7.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7)
        ax7.set_ylabel('Accuracy')
        ax7.set_title(f'Best: {all_labels[best_idx]}\nWorst: {all_labels[worst_idx]}')
        ax7.set_ylim(0, 1)
        
        # Add value labels
        ax7.text(0, all_accs[best_idx] + 0.02, f'{all_accs[best_idx]:.3f}', 
                ha='center', va='bottom', fontweight='bold')
        ax7.text(1, all_accs[worst_idx] + 0.02, f'{all_accs[worst_idx]:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Bottom row: Overall summary
    ax8 = fig.add_subplot(gs[3, :])
    
    # Create comprehensive summary
    summary_title = f"TEMPORAL DECODING SUMMARY - {response_type.upper()}\n"
    summary_title += f"Animal: {session_info['animal_id']}, Session: {session_info['session_id']}\n"
    summary_title += f"Units: {session_info['n_units']}, Features: {feature_type}\n\n"
    
    if methods:
        best_overall_acc = max(max(left_accs) if left_accs else 0, max(right_accs) if right_accs else 0)
        best_left_acc = max(left_accs) if left_accs else 0
        best_right_acc = max(right_accs) if right_accs else 0
        
        summary_title += f"BEST PERFORMANCE:\n"
        summary_title += f"Overall: {best_overall_acc:.3f}\n"
        summary_title += f"Left VF: {best_left_acc:.3f}, Right VF: {best_right_acc:.3f}\n\n"
        
        # Count significant results
        n_sig_left = sum(left_significant)
        n_sig_right = sum(right_significant)
        
        summary_title += f"SIGNIFICANCE:\n"
        summary_title += f"Left VF: {n_sig_left}/{len(methods)} methods above chance\n"
        summary_title += f"Right VF: {n_sig_right}/{len(methods)} methods above chance\n\n"
        
        summary_title += f"INTERPRETATION:\n"
        if best_overall_acc > chance_level + 0.1:
            summary_title += "Strong temporal orientation encoding detected!\n"
        elif best_overall_acc > chance_level + 0.05:
            summary_title += "Moderate temporal orientation encoding detected.\n"
        else:
            summary_title += "Weak temporal orientation encoding.\n"
        
        if abs(best_left_acc - best_right_acc) > 0.05:
            summary_title += f"Visual field difference detected ({abs(best_left_acc - best_right_acc):.3f}).\n"
        else:
            summary_title += "No major visual field differences.\n"
    
    ax8.text(0.05, 0.95, summary_title, transform=ax8.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    
    plt.suptitle(f"Temporal Orientation Decoding Analysis\n{feature_type} Features - {response_type}", 
                fontsize=16, y=0.98)
    
    return fig

def plot_temporal_confusion_matrix(results, ax, title):
    """Plot confusion matrix for temporal decoding"""
    y_true = results['y_true']
    y_pred = results['y_pred']
    orientations = results['orientations']
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create orientation labels
    ori_labels = [f"{int(ori)}°" for ori in orientations]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=ori_labels, yticklabels=ori_labels)
    ax.set_xlabel('Predicted Orientation')
    ax.set_ylabel('Actual Orientation')
    ax.set_title(title)

def plot_temporal_feature_importance(results_left, results_right, ax):
    """Plot feature importance from Random Forest"""
    if 'feature_importance' in results_left and 'feature_importance' in results_right:
        fi_left = results_left['feature_importance']
        fi_right = results_right['feature_importance']
        
        # Show top features
        n_features = min(15, len(fi_left))
        combined_importance = fi_left + fi_right
        top_indices = np.argsort(combined_importance)[-n_features:]
        
        x = np.arange(n_features)
        width = 0.35
        
        ax.barh(x - width/2, fi_left[top_indices], width, label='Left VF', alpha=0.8)
        ax.barh(x + width/2, fi_right[top_indices], width, label='Right VF', alpha=0.8)
        
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top Temporal Features (Random Forest)')
        ax.set_yticks(x)
        ax.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax.legend()

def temporal_orientation_decoding_analysis(npz_file, feature_types=['all'], response_phases=['display'], 
                                         cv_folds=5, random_state=42):
    """
    Main function for temporal orientation decoding analysis
    
    Parameters:
    npz_file: Path to NPZ file with temporal features
    feature_types: List of feature types to test ['binned_rates', 'spike_timing', 'combined', 'all']
    response_phases: List of response phases ['display', 'baseline', 'iti']
    """
    
    npz_file = Path(npz_file)
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name
    
    # Create output directory
    decode_folder = session_folder / "temporal_decoding"
    decode_folder.mkdir(exist_ok=True)
    
    print(f"TEMPORAL ORIENTATION DECODING ANALYSIS")
    print("=" * 60)
    
    # Load temporal data
    try:
        data_dict = load_temporal_data(npz_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you've run temporal_spike_processing.py first!")
        return None
    
    session_info = {
        'animal_id': animal_id,
        'session_id': session_id,
        'n_units': data_dict['n_good_units'],
        'n_trials': data_dict['n_trials'],
        'n_time_bins': data_dict['n_time_bins']
    }
    
    all_results = {}
    
    # Test different feature types and response phases
    for response_phase in response_phases:
        for feature_type in feature_types:
            
            print(f"\n{'='*50}")
            print(f"Testing: {feature_type} features, {response_phase} phase")
            print('='*50)
            
            try:
                # Extract feature matrix
                feature_matrix, feature_names = extract_temporal_feature_matrix(
                    data_dict['unit_responses'], feature_type, response_phase
                )
                
                # Remove invalid trials
                left_orientations = data_dict['left_orientations']
                right_orientations = data_dict['right_orientations']
                
                valid_mask = (~np.isnan(feature_matrix).any(axis=1) & 
                             ~np.isnan(left_orientations) & 
                             ~np.isnan(right_orientations))
                
                feature_matrix_valid = feature_matrix[valid_mask]
                left_orientations_valid = left_orientations[valid_mask]
                right_orientations_valid = right_orientations[valid_mask]
                
                print(f"Valid trials: {feature_matrix_valid.shape[0]}")
                
                # Classify left and right visual fields
                print(f"\nClassifying Left Visual Field...")
                results_left = perform_temporal_classification(
                    feature_matrix_valid, left_orientations_valid, 
                    method='all', cv_folds=cv_folds, random_state=random_state
                )
                
                print(f"\nClassifying Right Visual Field...")
                results_right = perform_temporal_classification(
                    feature_matrix_valid, right_orientations_valid,
                    method='all', cv_folds=cv_folds, random_state=random_state
                )
                
                # Create comprehensive figure
                if results_left or results_right:
                    fig = create_temporal_decoding_figure(
                        results_left, results_right, session_info, response_phase, feature_type
                    )
                    
                    # Save figure
                    output_file = decode_folder / f"temporal_decoding_{feature_type}_{response_phase}.png"
                    fig.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"Figure saved: {output_file}")
                
                # Store results
                all_results[f"{feature_type}_{response_phase}"] = {
                    'left_results': results_left,
                    'right_results': results_right,
                    'feature_matrix_shape': feature_matrix_valid.shape,
                    'n_features': len(feature_names),
                    'feature_names': feature_names
                }
                
            except Exception as e:
                print(f"Error processing {feature_type}_{response_phase}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save comprehensive results
    results_file = decode_folder / "temporal_decoding_results.npz"
    np.savez(results_file, all_results=all_results, session_info=session_info, allow_pickle=True)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    
    best_performance = 0
    best_config = ""
    
    for config_name, config_results in all_results.items():
        left_results = config_results['left_results']
        right_results = config_results['right_results']
        
        print(f"\n{config_name}:")
        
        # Find best method for this configuration
        best_left_acc = 0
        best_right_acc = 0
        
        for method_name in ['SVM', 'Random Forest', 'Neural Network', 'Logistic Regression']:
            if method_name in left_results:
                acc = left_results[method_name]['mean_accuracy']
                if acc > best_left_acc:
                    best_left_acc = acc
            if method_name in right_results:
                acc = right_results[method_name]['mean_accuracy']
                if acc > best_right_acc:
                    best_right_acc = acc
        
        overall_best = max(best_left_acc, best_right_acc)
        print(f"  Best Left VF: {best_left_acc:.3f}")
        print(f"  Best Right VF: {best_right_acc:.3f}")
        print(f"  Overall Best: {overall_best:.3f}")
        
        if overall_best > best_performance:
            best_performance = overall_best
            best_config = config_name
    
    print(f"\nBEST OVERALL CONFIGURATION:")
    print(f"  {best_config}: {best_performance:.3f}")
    print(f"  Chance level: 0.25")
    print(f"  Above chance: {'Yes' if best_performance > 0.3 else 'Maybe' if best_performance > 0.27 else 'No'}")
    
    return all_results


def create_feature_comparison_figure(all_results, session_info):
    """
    Create figure comparing all feature types and response phases
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect all results for comparison
    configs = []
    left_accs = []
    right_accs = []
    
    for config_name, config_results in all_results.items():
        left_results = config_results['left_results']
        right_results = config_results['right_results']
        
        # Get best accuracy for each VF
        best_left = 0
        best_right = 0
        
        for method_name in ['SVM', 'Random Forest', 'Neural Network', 'Logistic Regression']:
            if method_name in left_results:
                best_left = max(best_left, left_results[method_name]['mean_accuracy'])
            if method_name in right_results:
                best_right = max(best_right, right_results[method_name]['mean_accuracy'])
        
        configs.append(config_name)
        left_accs.append(best_left)
        right_accs.append(best_right)
    
    # Plot 1: Overall comparison
    ax1 = axes[0, 0]
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, left_accs, width, label='Left VF', alpha=0.8)
    bars2 = ax1.bar(x + width/2, right_accs, width, label='Right VF', alpha=0.8)
    
    ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7, label='Chance')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Best Accuracy')
    ax1.set_title('Feature Type & Phase Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot 2: Feature type comparison (aggregate across phases)
    ax2 = axes[0, 1]
    feature_types = ['binned_rates', 'spike_timing', 'combined', 'all']
    feature_accs = {ft: [] for ft in feature_types}
    
    for config_name in configs:
        for ft in feature_types:
            if ft in config_name:
                overall_acc = max(left_accs[configs.index(config_name)], 
                                right_accs[configs.index(config_name)])
                feature_accs[ft].append(overall_acc)
    
    # Average across phases
    feature_means = [np.mean(feature_accs[ft]) if feature_accs[ft] else 0 for ft in feature_types]
    feature_stds = [np.std(feature_accs[ft]) if len(feature_accs[ft]) > 1 else 0 for ft in feature_types]
    
    bars = ax2.bar(feature_types, feature_means, yerr=feature_stds, capsize=5, alpha=0.8)
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Feature Type')
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Feature Type Performance')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Response phase comparison
    ax3 = axes[1, 0]
    phases = ['display', 'baseline', 'iti']
    phase_accs = {phase: [] for phase in phases}
    
    for config_name in configs:
        for phase in phases:
            if phase in config_name:
                overall_acc = max(left_accs[configs.index(config_name)], 
                                right_accs[configs.index(config_name)])
                phase_accs[phase].append(overall_acc)
    
    phase_means = [np.mean(phase_accs[phase]) if phase_accs[phase] else 0 for phase in phases]
    phase_stds = [np.std(phase_accs[phase]) if len(phase_accs[phase]) > 1 else 0 for phase in phases]
    
    bars = ax3.bar(phases, phase_means, yerr=phase_stds, capsize=5, alpha=0.8, 
                   color=['red', 'blue', 'green'])
    ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Response Phase')
    ax3.set_ylabel('Mean Accuracy')
    ax3.set_title('Response Phase Performance')
    
    # Plot 4: Best configuration details
    ax4 = axes[1, 1]
    
    if configs:
        best_idx = np.argmax([max(left_accs[i], right_accs[i]) for i in range(len(configs))])
        best_config = configs[best_idx]
        
        # Show method comparison for best config
        if best_config in all_results:
            best_results = all_results[best_config]
            methods = []
            method_accs = []
            
            for method in ['SVM', 'Random Forest', 'Neural Network', 'Logistic Regression']:
                if method in best_results['left_results'] or method in best_results['right_results']:
                    methods.append(method)
                    left_acc = best_results['left_results'].get(method, {}).get('mean_accuracy', 0)
                    right_acc = best_results['right_results'].get(method, {}).get('mean_accuracy', 0)
                    method_accs.append(max(left_acc, right_acc))
            
            if methods:
                bars = ax4.bar(methods, method_accs, alpha=0.8)
                ax4.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7)
                ax4.set_xlabel('Method')
                ax4.set_ylabel('Best Accuracy')
                ax4.set_title(f'Best Config: {best_config}')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, acc in zip(bars, method_accs):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle(f"Temporal Decoding Feature Comparison\n"
                f"Animal: {session_info['animal_id']}, Session: {session_info['session_id']}", 
                fontsize=14, y=0.98)
    
    return fig


if __name__ == '__main__':
    # NPZ file with temporal features
    npz_file = r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22sg\CnL22SG_20250609_164650\object_discrimination_temporal_responses.npz"
    
    print("REAL TEMPORAL DECODING ANALYSIS")
    print("=" * 60)
    print("Using actual spike timing data from your recordings!")
    print("=" * 60)
    
    # Run comprehensive analysis
    try:
        # Test different combinations of features and response phases
        all_results = temporal_orientation_decoding_analysis(
            npz_file,
            feature_types=['binned_rates', 'spike_timing', 'combined', 'all'],
            response_phases=['display', 'baseline', 'iti'],
            cv_folds=5,
            random_state=42
        )
        
        if all_results:
            # Create comparison figure
            session_info = {
                'animal_id': Path(npz_file).parent.parent.name,
                'session_id': Path(npz_file).parent.name
            }
            
            comparison_fig = create_feature_comparison_figure(all_results, session_info)
            
            # Save comparison figure
            decode_folder = Path(npz_file).parent / "temporal_decoding"
            comparison_file = decode_folder / "feature_comparison_summary.png"
            comparison_fig.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close(comparison_fig)
            
            print(f"\nComparison figure saved: {comparison_file}")
            print(f"\nAll results saved in: {decode_folder}")
            
        else:
            print("No results generated. Check your data and try again.")
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nTroubleshooting:")
        print(f"1. Make sure you've run temporal_spike_processing.py first")
        print(f"2. Check that the NPZ file exists and contains temporal features")
        print(f"3. Verify your data has good quality units")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("Check the 'temporal_decoding' folder for all results!")
    print("=" * 60)