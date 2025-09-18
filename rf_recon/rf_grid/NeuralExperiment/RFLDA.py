import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import pickle
import h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Import the loading functions from your extraction script
# Assuming the extraction script is named 'neural_data_extraction.py'
try:
    from neural_data_extraction import load_neural_data
except ImportError:
    # If import fails, define the loading functions here
    def load_neural_data(filepath):
        """Load neural data from saved file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.h5':
            return load_from_hdf5(filepath)
        elif filepath.suffix == '.pkl':
            return load_from_pickle(filepath)
        elif filepath.suffix == '.npz':
            return load_from_npz(filepath)
        else:
            raise ValueError("Unsupported file format")
    
    def load_from_pickle(filepath):
        """Load neural data from pickle format"""
        print(f"Loading from pickle: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def calculate_firing_rates(neural_data, time_window=(0.07, 0.16)):
    """
    Calculate firing rates for each unit and trial in a specific time window.
    
    Parameters:
    -----------
    neural_data : dict
        Neural data loaded from file
    time_window : tuple
        Time window in seconds (start, end) relative to stimulus onset
        
    Returns:
    --------
    firing_rates : np.ndarray
        Array of shape (n_trials, n_units) containing firing rates
    stimulus_labels : np.ndarray
        Array of stimulus indices for each trial
    unit_ids : list
        List of unit identifiers
    """
    
    window_start, window_end = time_window
    window_duration = window_end - window_start
    
    # Get all unit IDs
    unit_ids = list(neural_data['spike_data'].keys())
    n_units = len(unit_ids)
    n_trials = neural_data['metadata']['n_trials']
    
    print(f"Calculating firing rates for {n_units} units across {n_trials} trials")
    print(f"Time window: {window_start:.3f}s to {window_end:.3f}s ({window_duration:.3f}s duration)")
    
    # Initialize arrays
    firing_rates = np.full((n_trials, n_units), np.nan)
    stimulus_labels = neural_data['trial_info']['stimulus_index']
    
    # Calculate firing rates for each unit
    for unit_idx, unit_id in enumerate(unit_ids):
        spike_data = neural_data['spike_data'][unit_id]
        
        for trial_idx in range(n_trials):
            if trial_idx < len(spike_data['spike_times']):
                spike_times = spike_data['spike_times'][trial_idx]
                
                # Count spikes in the time window
                spikes_in_window = np.sum((spike_times >= window_start) & 
                                        (spike_times < window_end))
                
                # Convert to firing rate (spikes/second)
                firing_rate = spikes_in_window / window_duration
                firing_rates[trial_idx, unit_idx] = firing_rate
    
    # Remove trials with any NaN values
    valid_trials = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_trials]
    stimulus_labels_clean = stimulus_labels[valid_trials]
    
    print(f"Valid trials: {np.sum(valid_trials)}/{n_trials}")
    print(f"Mean firing rate across all units/trials: {np.nanmean(firing_rates_clean):.2f} Hz")
    
    return firing_rates_clean, stimulus_labels_clean, unit_ids


def perform_lda_analysis(firing_rates, stimulus_labels, n_components=None):
    """
    Perform LDA analysis with cross-validation.
    
    Parameters:
    -----------
    firing_rates : np.ndarray
        Firing rates array (n_trials, n_units)
    stimulus_labels : np.ndarray
        Stimulus labels for each trial
    n_components : int or None
        Number of LDA components (default: min(n_classes-1, n_features))
        
    Returns:
    --------
    dict : Dictionary containing LDA results
    """
    
    # Get unique stimuli
    unique_stimuli = np.unique(stimulus_labels)
    n_stimuli = len(unique_stimuli)
    n_features = firing_rates.shape[1]
    
    print(f"\nLDA Analysis:")
    print(f"Number of stimuli: {n_stimuli}")
    print(f"Number of features (units): {n_features}")
    print(f"Number of trials: {len(stimulus_labels)}")
    
    # Determine number of components
    max_components = min(n_stimuli - 1, n_features)
    if n_components is None:
        n_components = min(3, max_components)  # Use 3 for visualization
    else:
        n_components = min(n_components, max_components)
    
    print(f"Using {n_components} LDA components")
    
    # Standardize features
    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_transformed = lda.fit_transform(firing_rates_scaled, stimulus_labels)
    
    # Cross-validation
    cv_folds = min(5, np.min(np.bincount(stimulus_labels)))  # Ensure each fold has all classes
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Full LDA for cross-validation (all components)
    lda_full = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda_full, firing_rates_scaled, stimulus_labels, 
                               cv=cv, scoring='accuracy')
    
    # Detailed cross-validation with confusion matrices
    cv_results = cross_validate(lda_full, firing_rates_scaled, stimulus_labels, 
                               cv=cv, scoring=['accuracy', 'f1_macro'], 
                               return_train_score=True, return_estimator=True)
    
    # Fit full model for final predictions
    lda_full.fit(firing_rates_scaled, stimulus_labels)
    predictions = lda_full.predict(firing_rates_scaled)
    prediction_proba = lda_full.predict_proba(firing_rates_scaled)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(stimulus_labels, predictions)
    
    # Calculate per-class accuracy
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    results = {
        'lda_model': lda,  # For visualization (reduced components)
        'lda_full': lda_full,  # Full model for classification
        'scaler': scaler,
        'transformed_data': lda_transformed,
        'original_data': firing_rates_scaled,
        'stimulus_labels': stimulus_labels,
        'predictions': predictions,
        'prediction_proba': prediction_proba,
        'cv_scores': cv_scores,
        'cv_results': cv_results,
        'confusion_matrix': conf_matrix,
        'class_accuracies': class_accuracies,
        'unique_stimuli': unique_stimuli,
        'n_components': n_components,
        'explained_variance_ratio': lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else None
    }
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Overall accuracy: {accuracy_score(stimulus_labels, predictions):.3f}")
    
    return results


def create_comprehensive_lda_plots(results, unit_ids, save_path=None):
    """
    Create comprehensive visualization of LDA results.
    
    Parameters:
    -----------
    results : dict
        Results from LDA analysis
    unit_ids : list
        List of unit identifiers
    save_path : str or Path, optional
        Path to save the figure
    """
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Get data
    transformed_data = results['transformed_data']
    labels = results['stimulus_labels']
    unique_stimuli = results['unique_stimuli']
    conf_matrix = results['confusion_matrix']
    cv_scores = results['cv_scores']
    class_accuracies = results['class_accuracies']
    
    # Color map for stimuli
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_stimuli)))
    
    # 1. 3D LDA scatter plot (top left)
    ax1 = fig.add_subplot(3, 4, (1, 5))  # Spans 2 rows
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    
    if results['n_components'] >= 3:
        for i, stimulus in enumerate(unique_stimuli):
            mask = labels == stimulus
            ax1.scatter(transformed_data[mask, 0], transformed_data[mask, 1], 
                       transformed_data[mask, 2], c=[colors[i]], 
                       label=f'Stimulus {stimulus}', alpha=0.7, s=30)
        ax1.set_xlabel('LD1')
        ax1.set_ylabel('LD2')
        ax1.set_zlabel('LD3')
        ax1.set_title('LDA 3D Scatter Plot', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax1.text(0.5, 0.5, 0.5, 'Need ≥3 components\nfor 3D visualization', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('LDA 3D Scatter Plot\n(Not enough components)', fontsize=14)
    
    # 2. 2D LDA scatter plot (top middle)
    ax2 = fig.add_subplot(3, 4, 2)
    if results['n_components'] >= 2:
        for i, stimulus in enumerate(unique_stimuli):
            mask = labels == stimulus
            ax2.scatter(transformed_data[mask, 0], transformed_data[mask, 1], 
                       c=[colors[i]], label=f'Stimulus {stimulus}', alpha=0.7, s=30)
        ax2.set_xlabel('LD1')
        ax2.set_ylabel('LD2')
        ax2.set_title('LDA 2D Scatter Plot', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        for i, stimulus in enumerate(unique_stimuli):
            mask = labels == stimulus
            y_jitter = np.random.normal(0, 0.1, np.sum(mask))
            ax2.scatter(transformed_data[mask, 0], y_jitter, 
                       c=[colors[i]], label=f'Stimulus {stimulus}', alpha=0.7, s=30)
        ax2.set_xlabel('LD1')
        ax2.set_ylabel('Random jitter')
        ax2.set_title('LDA 1D Projection', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix (top right)
    ax3 = fig.add_subplot(3, 4, 3)
    im = ax3.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set(xticks=np.arange(conf_matrix.shape[1]),
            yticks=np.arange(conf_matrix.shape[0]),
            xticklabels=unique_stimuli, yticklabels=unique_stimuli,
            title='Confusion Matrix', ylabel='True Label', xlabel='Predicted Label')
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax3.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 4. Cross-validation scores (top far right)
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.bar(range(len(cv_scores)), cv_scores, alpha=0.7, color='skyblue')
    ax4.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {cv_scores.mean():.3f}')
    ax4.set_xlabel('CV Fold')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Per-class accuracy (middle left)
    ax5 = fig.add_subplot(3, 4, 5)
    bars = ax5.bar(range(len(class_accuracies)), class_accuracies, 
                   color=colors[:len(class_accuracies)], alpha=0.7)
    ax5.set_xlabel('Stimulus')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(len(unique_stimuli)))
    ax5.set_xticklabels(unique_stimuli)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 6. LDA weights/coefficients heatmap (middle center-right)
    ax6 = fig.add_subplot(3, 4, (6, 7))
    if hasattr(results['lda_full'], 'coef_'):
        coef_matrix = results['lda_full'].coef_
        im = ax6.imshow(coef_matrix, cmap='RdBu_r', aspect='auto')
        ax6.figure.colorbar(im, ax=ax6)
        ax6.set_xlabel('Units')
        ax6.set_ylabel('LDA Components')
        ax6.set_title('LDA Coefficients Heatmap', fontsize=14, fontweight='bold')
        
        # Set ticks
        if len(unit_ids) <= 20:
            ax6.set_xticks(range(len(unit_ids)))
            ax6.set_xticklabels([uid.split('_')[-1] for uid in unit_ids], rotation=45)
        else:
            ax6.set_xlabel(f'Units (n={len(unit_ids)})')
    
    # 7. Explained variance (if available) (middle right)
    ax7 = fig.add_subplot(3, 4, 8)
    if results['explained_variance_ratio'] is not None:
        ax7.bar(range(len(results['explained_variance_ratio'])), 
                results['explained_variance_ratio'], alpha=0.7, color='green')
        ax7.set_xlabel('Component')
        ax7.set_ylabel('Explained Variance Ratio')
        ax7.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Explained variance\nnot available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Explained Variance', fontsize=14, fontweight='bold')
    
    # 8. Classification summary statistics (bottom left)
    ax8 = fig.add_subplot(3, 4, 9)
    ax8.axis('off')
    
    # Summary statistics
    overall_acc = accuracy_score(labels, results['predictions'])
    mean_cv_acc = cv_scores.mean()
    std_cv_acc = cv_scores.std()
    n_trials_per_class = [np.sum(labels == s) for s in unique_stimuli]
    
    summary_text = f"""
    Classification Summary:
    
    Overall Accuracy: {overall_acc:.3f}
    CV Accuracy: {mean_cv_acc:.3f} ± {std_cv_acc:.3f}
    
    Dataset Info:
    • Total trials: {len(labels)}
    • Number of stimuli: {len(unique_stimuli)}
    • Number of units: {len(unit_ids)}
    • LDA components: {results['n_components']}
    
    Trials per stimulus:
    """
    
    for i, (stim, count) in enumerate(zip(unique_stimuli, n_trials_per_class)):
        summary_text += f"• Stimulus {stim}: {count}\n    "
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 9. Feature importance (if available) (bottom center)
    ax9 = fig.add_subplot(3, 4, 10)
    if hasattr(results['lda_full'], 'coef_'):
        # Calculate feature importance as mean absolute coefficient
        feature_importance = np.mean(np.abs(results['lda_full'].coef_), axis=0)
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        # Show top 15 features
        n_show = min(15, len(feature_importance))
        y_pos = np.arange(n_show)
        
        ax9.barh(y_pos, feature_importance[sorted_idx[:n_show]], alpha=0.7, color='orange')
        ax9.set_yticks(y_pos)
        if len(unit_ids) <= 50:
            ax9.set_yticklabels([unit_ids[i].split('_')[-1] for i in sorted_idx[:n_show]])
        else:
            ax9.set_yticklabels([f'Unit_{i}' for i in sorted_idx[:n_show]])
        ax9.invert_yaxis()
        ax9.set_xlabel('Mean |Coefficient|')
        ax9.set_title(f'Top {n_show} Most Important Units', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='x')
    
    # 10. Prediction confidence (bottom right)
    ax10 = fig.add_subplot(3, 4, 11)
    prediction_confidence = np.max(results['prediction_proba'], axis=1)
    correct_predictions = (results['predictions'] == labels)
    
    ax10.hist(prediction_confidence[correct_predictions], bins=20, alpha=0.7, 
              label='Correct', color='green', density=True)
    ax10.hist(prediction_confidence[~correct_predictions], bins=20, alpha=0.7, 
              label='Incorrect', color='red', density=True)
    ax10.set_xlabel('Prediction Confidence')
    ax10.set_ylabel('Density')
    ax10.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Learning curve placeholder (bottom far right)
    ax11 = fig.add_subplot(3, 4, 12)
    
    # Simple visualization of training vs validation performance
    train_scores = results['cv_results']['train_accuracy']
    val_scores = results['cv_results']['test_accuracy']
    
    folds = range(1, len(train_scores) + 1)
    ax11.plot(folds, train_scores, 'o-', label='Training', color='blue', alpha=0.7)
    ax11.plot(folds, val_scores, 's-', label='Validation', color='red', alpha=0.7)
    ax11.fill_between(folds, train_scores, alpha=0.2, color='blue')
    ax11.fill_between(folds, val_scores, alpha=0.2, color='red')
    
    ax11.set_xlabel('CV Fold')
    ax11.set_ylabel('Accuracy')
    ax11.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, 1])
    
    # Adjust layout
    plt.tight_layout()
    
    # Add main title
    fig.suptitle('Neural Data LDA Classification Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.94)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def main(data_path, time_window=(0.07, 0.16), save_plots=True):
    """
    Main function to run the complete LDA analysis pipeline.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the neural data file
    time_window : tuple
        Time window for firing rate calculation (start, end) in seconds
    save_plots : bool
        Whether to save plots
    """
    
    print("="*60)
    print("NEURAL DATA LDA DECODING ANALYSIS")
    print("="*60)
    
    # Load data
    print(f"Loading data from: {data_path}")
    neural_data = load_neural_data(data_path)
    
    # Calculate firing rates
    firing_rates, stimulus_labels, unit_ids = calculate_firing_rates(
        neural_data, time_window=time_window
    )
    
    # Perform LDA analysis
    lda_results = perform_lda_analysis(firing_rates, stimulus_labels)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Determine save path
    if save_plots:
        data_path = Path(data_path)
        save_path = data_path.parent / f"{data_path.stem}_lda_analysis.png"
    else:
        save_path = None
    
    fig = create_comprehensive_lda_plots(lda_results, unit_ids, save_path)
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Overall Classification Accuracy: {accuracy_score(stimulus_labels, lda_results['predictions']):.3f}")
    print(f"Cross-Validation Accuracy: {lda_results['cv_scores'].mean():.3f} ± {lda_results['cv_scores'].std():.3f}")
    print(f"Number of units used: {len(unit_ids)}")
    print(f"Time window: {time_window[0]:.3f}s to {time_window[1]:.3f}s")
    print(f"Number of trials: {len(stimulus_labels)}")
    print(f"Number of stimulus types: {len(lda_results['unique_stimuli'])}")
    
    plt.show()
    
    return lda_results, firing_rates, stimulus_labels


if __name__ == "__main__":
    # Example usage - update this path to your actual data file
    data_file_path = "/Volumes/xieluanlabs/xl_cl/code/sortout/CnL39SG/CnL39SG_20250821_163039/embedding_analysis/CnL39SG_20250821_163039_neural_data_RFGRid_9.pkl"  # or .h5, .npz
    
    # Run analysis
    try:
        lda_results, firing_rates, stimulus_labels = main(
            data_path=data_file_path,
            time_window=(0.07, 0.16),  # 70ms to 160ms after stimulus onset
            save_plots=True
        )
        
        # Additional analysis can be performed here with the results
        
    except FileNotFoundError:
        print(f"Data file not found. Please update the data_file_path variable.")
        print("Make sure to run the neural data extraction script first.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your data file and ensure it was created with the extraction script.")