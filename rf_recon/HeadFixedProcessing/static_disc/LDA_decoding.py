import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

# Define the object-to-orientation mapping
OBJECT_TO_ORIENTATION = {
    14: 0,    # 0 degrees
    15: 45,   # 45 degrees  
    16: 90,   # 90 degrees
    17: 135   # 135 degrees
}

ORIENTATION_TO_OBJECT = {v: k for k, v in OBJECT_TO_ORIENTATION.items()}

def convert_objects_to_orientations(object_indices):
    """Convert object indices to orientation values"""
    orientations = np.array([OBJECT_TO_ORIENTATION.get(obj, np.nan) for obj in object_indices])
    return orientations

def extract_firing_rates_for_orientation_lda(unit_responses_data, trial_data_df, response_type='display_responses'):
    """
    Extract firing rates and convert object indices to orientations
    
    Returns:
    firing_rates: Matrix of shape (n_trials, n_units) with firing rates
    left_orientations: Array of left visual field orientations for each trial
    right_orientations: Array of right visual field orientations for each trial  
    valid_trials: Boolean mask for trials with valid data
    """
    
    print(f"Extracting {response_type} firing rates for orientation decoding...")
    
    # Get trial information from unit responses (they have the proper column names)
    if len(unit_responses_data) > 0 and 'trial_data' in unit_responses_data[0]:
        trial_data = unit_responses_data[0]['trial_data']
        print("Using trial data from unit responses")
    else:
        trial_data = trial_data_df
        print("Using provided trial data")
    
    print(f"Trial data columns: {trial_data.columns.tolist()}")
    
    # Get left and right object indices
    if 'left_object_index' in trial_data.columns and 'right_object_index' in trial_data.columns:
        left_objects = trial_data['left_object_index'].values
        right_objects = trial_data['right_object_index'].values
    else:
        raise ValueError("Cannot find left/right object columns in trial data")
    
    # Convert object indices to orientations
    left_orientations = convert_objects_to_orientations(left_objects)
    right_orientations = convert_objects_to_orientations(right_objects)
    
    print(f"Left orientations: {np.unique(left_orientations[~np.isnan(left_orientations)])}")
    print(f"Right orientations: {np.unique(right_orientations[~np.isnan(right_orientations)])}")
    
    # Extract firing rates
    n_units = len(unit_responses_data)
    n_trials = len(trial_data)
    
    print(f"Processing {n_units} units across {n_trials} trials")
    
    # Initialize firing rate matrix
    firing_rates = np.zeros((n_trials, n_units))
    
    for unit_idx, unit_data in enumerate(unit_responses_data):
        if isinstance(unit_data, dict) and response_type in unit_data:
            unit_firing_rates = unit_data[response_type]
            n_available = min(len(unit_firing_rates), n_trials)
            firing_rates[:n_available, unit_idx] = unit_firing_rates[:n_available]
        else:
            print(f"Warning: Unit {unit_idx} missing {response_type}")
    
    # Identify valid trials
    valid_trials = (~np.isnan(firing_rates).any(axis=1) & 
                   ~np.isnan(left_orientations) & 
                   ~np.isnan(right_orientations))
    
    print(f"Found {valid_trials.sum()} valid trials out of {n_trials} total")
    print(f"Firing rate range: {np.nanmin(firing_rates):.3f} to {np.nanmax(firing_rates):.3f} Hz")
    
    return firing_rates, left_orientations, right_orientations, valid_trials


def perform_orientation_lda(firing_rates, orientations, visual_field, cv_folds=5, random_state=42, normalize=True):
    """
    Perform LDA on firing rate data for orientation decoding
    
    Parameters:
    firing_rates: Matrix of firing rates (n_trials, n_units)
    orientations: Orientation values for each trial
    visual_field: 'left' or 'right' for labeling
    cv_folds: Number of cross-validation folds
    random_state: Random seed
    normalize: Whether to standardize firing rates
    
    Returns:
    Results tuple with classifier, LDA projection, scores, etc.
    """
    
    # Remove invalid trials
    valid_mask = (~np.isnan(firing_rates).any(axis=1) & ~np.isnan(orientations))
    
    X_clean = firing_rates[valid_mask]
    y_clean = orientations[valid_mask]
    
    print(f"\n{visual_field.capitalize()} visual field orientation LDA:")
    print(f"  Clean data shape: {X_clean.shape}")
    print(f"  Unique orientations: {np.unique(y_clean)}")
    
    if len(np.unique(y_clean)) < 2:
        print(f"  Warning: Only {len(np.unique(y_clean))} unique orientations found, skipping LDA")
        return None, None, np.array([]), None, None, None, None
    
    # Normalize firing rates if requested
    if normalize:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_clean)
        print(f"  Normalized firing rates: mean={np.mean(X_final):.3f}, std={np.std(X_final):.3f}")
    else:
        X_final = X_clean
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_clean.astype(int))
    
    # Adjust CV folds if necessary
    min_samples = np.min(np.bincount(y_encoded))
    cv_folds = min(cv_folds, min_samples)
    
    print(f"  Using {cv_folds} CV folds")
    print(f"  Min samples per orientation: {min_samples}")
    
    # Perform LDA with cross-validation
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(clf, X_final, y_encoded, cv=skf, scoring='accuracy')
    
    # Fit on full data for visualization
    clf.fit(X_final, y_encoded)
    X_lda = clf.transform(X_final)
    
    print(f"  LDA accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    return clf, X_lda, scores, skf, X_final, y_encoded, le


def create_orientation_comparison_figure(results_left, results_right, session_info, response_type):
    """
    Create comprehensive figure comparing left and right visual field orientation decoding
    """
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Unpack results
    clf_left, X_lda_left, scores_left, skf_left, X_left, y_left, le_left = results_left
    clf_right, X_lda_right, scores_right, skf_right, X_right, y_right, le_right = results_right
    
    # Top row: 3D LDA projections and accuracy comparison
    if X_lda_left is not None and X_lda_left.shape[1] >= 3:
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        plot_orientation_lda_3d(X_lda_left, y_left, le_left, ax1, "Left Visual Field\nOrientation LDA")
    else:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, 'Insufficient data\nfor left visual field', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Left Visual Field")
    
    if X_lda_right is not None and X_lda_right.shape[1] >= 3:
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        plot_orientation_lda_3d(X_lda_right, y_right, le_right, ax2, "Right Visual Field\nOrientation LDA")
    else:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, 'Insufficient data\nfor right visual field', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Right Visual Field")
    
    # Accuracy comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if len(scores_left) > 0 and len(scores_right) > 0:
        categories = ['Left Visual Field', 'Right Visual Field']
        accuracies = [scores_left.mean(), scores_right.mean()]
        errors = [scores_left.std(), scores_right.std()]
        
        bars = ax3.bar(categories, accuracies, yerr=errors, capsize=5, alpha=0.7, 
                      color=['blue', 'red'])
        
        for bar, acc, err in zip(bars, accuracies, errors):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel("Decoding Accuracy", fontsize=12)
    ax3.set_title("Left vs Right Visual Field\nOrientation Decoding", fontsize=14)
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=15)
    
    # CV score distribution
    ax4 = fig.add_subplot(gs[0, 3])
    if len(scores_left) > 0 and len(scores_right) > 0:
        ax4.boxplot([scores_left, scores_right], labels=['Left VF', 'Right VF'])
        ax4.set_ylabel("CV Accuracy", fontsize=12)
        ax4.set_title("CV Score Distribution", fontsize=14)
        ax4.set_ylim(0, 1)
    
    # Middle row: Confusion matrices
    ax5 = fig.add_subplot(gs[1, 0])
    if clf_left is not None:
        plot_orientation_confusion_matrix(X_left, y_left, clf_left, skf_left, le_left, ax5, 
                                         "Left VF Confusion Matrix")
    
    ax6 = fig.add_subplot(gs[1, 1])
    if clf_right is not None:
        plot_orientation_confusion_matrix(X_right, y_right, clf_right, skf_right, le_right, ax6, 
                                         "Right VF Confusion Matrix")
    
    # Per-orientation accuracy
    ax7 = fig.add_subplot(gs[1, 2])
    if clf_left is not None:
        plot_per_orientation_accuracy(X_left, y_left, clf_left, skf_left, le_left, ax7,
                                     "Left VF Per-Orientation Accuracy")
    
    ax8 = fig.add_subplot(gs[1, 3])
    if clf_right is not None:
        plot_per_orientation_accuracy(X_right, y_right, clf_right, skf_right, le_right, ax8,
                                     "Right VF Per-Orientation Accuracy")
    
    # Bottom row: Additional analyses
    
    # Orientation distribution
    ax9 = fig.add_subplot(gs[2, 0])
    if clf_left is not None:
        orientations_left = [le_left.inverse_transform([y])[0] for y in np.unique(y_left)]
        counts_left = [np.sum(y_left == y_val) for y_val in np.unique(y_left)]
        ax9.bar([f"{int(ori)}°" for ori in orientations_left], counts_left, alpha=0.7, color='blue')
        ax9.set_title("Left VF Orientation Trials", fontsize=14)
        ax9.set_ylabel("Number of Trials", fontsize=12)
        ax9.tick_params(axis='x', rotation=45)
    
    ax10 = fig.add_subplot(gs[2, 1])
    if clf_right is not None:
        orientations_right = [le_right.inverse_transform([y])[0] for y in np.unique(y_right)]
        counts_right = [np.sum(y_right == y_val) for y_val in np.unique(y_right)]
        ax10.bar([f"{int(ori)}°" for ori in orientations_right], counts_right, alpha=0.7, color='red')
        ax10.set_title("Right VF Orientation Trials", fontsize=14)
        ax10.set_ylabel("Number of Trials", fontsize=12)
        ax10.tick_params(axis='x', rotation=45)
    
    # Statistical comparison
    ax11 = fig.add_subplot(gs[2, 2:])
    if len(scores_left) > 0 and len(scores_right) > 0 and len(scores_left) == len(scores_right):
        t_stat, p_value = ttest_rel(scores_left, scores_right)
        
        ax11.plot(scores_left, 'bo-', label='Left Visual Field', alpha=0.7, markersize=8)
        ax11.plot(scores_right, 'ro-', label='Right Visual Field', alpha=0.7, markersize=8)
        ax11.set_xlabel("CV Fold", fontsize=12)
        ax11.set_ylabel("Accuracy", fontsize=12)
        ax11.set_title(f"Left vs Right Visual Field Comparison\n"
                      f"t-test: t={t_stat:.3f}, p={p_value:.3f}\n"
                      f"{'Left > Right' if np.mean(scores_left) > np.mean(scores_right) else 'Right > Left'}", 
                      fontsize=14)
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        ax11.set_ylim(0, 1)
        
        # Add chance level line
        ax11.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    
    # Overall title
    fig.suptitle(f"Orientation Decoding in Left vs Right Visual Fields\n"
                f"Animal: {session_info['animal_id']}, Session: {session_info['session_id']}\n"
                f"Units: {session_info['n_units']}, Response: {response_type}\n"
                f"Orientations: 0°, 45°, 90°, 135° (Objects 14, 15, 16, 17)", 
                fontsize=16, y=0.98)
    
    return fig


def plot_orientation_lda_3d(X_lda, y, le, ax, title):
    """Plot 3D LDA scatter with orientation labels"""
    unique_orientations = np.unique(y)
    colors = ['red', 'orange', 'green', 'blue']  # Colors for 0°, 45°, 90°, 135°
    
    for i, ori_encoded in enumerate(unique_orientations):
        mask = y == ori_encoded
        ori_value = le.inverse_transform([ori_encoded])[0]
        color = colors[i % len(colors)]
        
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], X_lda[mask, 2],
                   color=color, label=f"{int(ori_value)}°", alpha=0.6, s=30)
    
    ax.set_xlabel("LDA1")
    ax.set_ylabel("LDA2")
    ax.set_zlabel("LDA3")
    ax.set_title(title)
    ax.legend()


def plot_orientation_confusion_matrix(X, y, clf, skf, le, ax, title):
    """Plot confusion matrix with orientation labels"""
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    unique_labels = np.unique(y)
    ori_labels = [f"{int(le.inverse_transform([label])[0])}°" for label in unique_labels]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=ori_labels, yticklabels=ori_labels)
    ax.set_xlabel("Predicted Orientation")
    ax.set_ylabel("Actual Orientation")
    ax.set_title(title)


def plot_per_orientation_accuracy(X, y, clf, skf, le, ax, title):
    """Plot per-orientation accuracy"""
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    
    unique_orientations = np.unique(y)
    ori_accuracies = []
    ori_labels = []
    
    for ori_encoded in unique_orientations:
        ori_value = le.inverse_transform([ori_encoded])[0]
        ori_labels.append(f"{int(ori_value)}°")
        
        ori_mask = y == ori_encoded
        ori_acc = accuracy_score(y[ori_mask], y_pred[ori_mask])
        ori_accuracies.append(ori_acc)
    
    colors = ['red', 'orange', 'green', 'blue']
    bars = ax.bar(range(len(ori_labels)), ori_accuracies, alpha=0.7, 
                  color=[colors[i % len(colors)] for i in range(len(ori_labels))])
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(range(len(ori_labels)))
    ax.set_xticklabels(ori_labels)
    ax.set_ylim(0, 1)
    
    # Add chance level line
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    
    for bar, acc in zip(bars, ori_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)


def decode_orientations_by_visual_field(npz_file, cv_folds=5, random_state=42, 
                                       response_type='display_responses', normalize=True):
    """
    Main function to decode orientations in left vs right visual fields
    
    Parameters:
    npz_file: Path to NPZ file with object discrimination responses
    cv_folds: Number of cross-validation folds
    random_state: Random seed
    response_type: Type of responses to analyze
    normalize: Whether to normalize firing rates
    """
    
    npz_file = Path(npz_file)
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name
    
    # Create output directory
    decode_folder = session_folder / "orientation_decoding"
    decode_folder.mkdir(exist_ok=True)
    
    print(f"Loading data from {npz_file}")
    print(f"Orientation mapping: {OBJECT_TO_ORIENTATION}")
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    unit_responses = data['all_units_responses']
    unit_qualities = data['unit_qualities']
    trial_data_raw = data['trial_data']
    
    print(f"Trial data type: {type(trial_data_raw)}")
    print(f"Trial data shape: {trial_data_raw.shape if hasattr(trial_data_raw, 'shape') else 'No shape'}")
    
    # Handle different trial_data formats - same robust approach as before
    if isinstance(trial_data_raw, np.ndarray):
        if trial_data_raw.ndim == 0:
            # 0-dimensional array containing an object
            try:
                trial_data = trial_data_raw.item()
                print(f"Extracted trial_data type: {type(trial_data)}")
            except ValueError:
                print("Cannot extract with .item(), trying alternative approaches...")
                # Try to access the data differently
                if hasattr(trial_data_raw, 'tolist'):
                    trial_data = trial_data_raw.tolist()
                else:
                    # Check if it's stored in unit responses instead
                    if len(unit_responses) > 0 and 'trial_data' in unit_responses[0]:
                        trial_data = unit_responses[0]['trial_data']
                        print("Using trial_data from unit_responses[0]")
                    else:
                        raise ValueError("Cannot access trial_data from any source")
        elif trial_data_raw.ndim == 1:
            # 1-dimensional array - might be a list of records
            trial_data = trial_data_raw
        else:
            # Multi-dimensional array
            trial_data = trial_data_raw
    else:
        trial_data = trial_data_raw
    
    # Convert to DataFrame if not already
    if not isinstance(trial_data, pd.DataFrame):
        if isinstance(trial_data, dict):
            trial_data = pd.DataFrame(trial_data)
        elif isinstance(trial_data, (list, np.ndarray)):
            trial_data = pd.DataFrame(trial_data)
        else:
            print(f"Unexpected trial_data type: {type(trial_data)}")
            # Last resort: try to get trial data from unit responses
            if len(unit_responses) > 0 and 'trial_data' in unit_responses[0]:
                trial_data = unit_responses[0]['trial_data']
                print("Using trial_data from unit_responses[0] as fallback")
                if not isinstance(trial_data, pd.DataFrame):
                    trial_data = pd.DataFrame(trial_data)
            else:
                raise ValueError(f"Cannot convert trial_data to DataFrame. Type: {type(trial_data)}")
    
    print(f"Final trial_data shape: {trial_data.shape}")
    print(f"Trial_data columns: {trial_data.columns.tolist()}")
    
    # Filter good units
    good_mask = np.array([qual != 'noise' for qual in unit_qualities])
    unit_responses_good = unit_responses[good_mask]
    n_good_units = good_mask.sum()
    
    print(f"Using {n_good_units} good quality units")
    
    # Extract firing rates and orientation labels
    firing_rates, left_orientations, right_orientations, valid_trials = extract_firing_rates_for_orientation_lda(
        unit_responses_good, trial_data, response_type
    )
    
    # Use only valid trials
    firing_rates_valid = firing_rates[valid_trials]
    left_orientations_valid = left_orientations[valid_trials]
    right_orientations_valid = right_orientations[valid_trials]
    
    print(f"Using {firing_rates_valid.shape[0]} valid trials")
    
    # Perform LDA for left and right visual fields
    results_left = perform_orientation_lda(
        firing_rates_valid, left_orientations_valid, 'left', cv_folds, random_state, normalize
    )
    
    results_right = perform_orientation_lda(
        firing_rates_valid, right_orientations_valid, 'right', cv_folds, random_state, normalize
    )
    
    # Create comprehensive figure
    session_info = {
        'animal_id': animal_id,
        'session_id': session_id,
        'n_units': n_good_units
    }
    
    fig = create_orientation_comparison_figure(results_left, results_right, session_info, response_type)
    
    # Save figure
    output_file = decode_folder / f"orientation_decoding_visual_fields_{response_type}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save results
    results_dict = {
        'left_vf_scores': results_left[2] if results_left[0] is not None else np.array([]),
        'right_vf_scores': results_right[2] if results_right[0] is not None else np.array([]),
        'response_type': response_type,
        'normalize': normalize,
        'n_good_units': n_good_units,
        'n_valid_trials': firing_rates_valid.shape[0],
        'animal_id': animal_id,
        'session_id': session_id,
        'object_to_orientation': OBJECT_TO_ORIENTATION
    }
    
    results_file = decode_folder / f"orientation_lda_results_{response_type}.npz"
    np.savez(results_file, **results_dict)
    
    print(f"\nResults saved to {decode_folder}")
    print(f"Figure: {output_file}")
    
    if results_left[0] is not None:
        print(f"Left visual field accuracy: {results_left[2].mean():.3f} ± {results_left[2].std():.3f}")
    if results_right[0] is not None:
        print(f"Right visual field accuracy: {results_right[2].mean():.3f} ± {results_right[2].std():.3f}")
    
    return results_dict


if __name__ == '__main__':
    # Example usage
    npz_file = r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22sg\CnL22SG_20250609_164650\object_discrimination_responses.npz"
    
    # Analyze different response types
    for response_type in ['display_responses', 'baseline_responses', 'iti_responses']:
        print(f"\n{'='*60}")
        print(f"Analyzing {response_type} for orientation decoding")
        print('='*60)
        
        try:
            results = decode_orientations_by_visual_field(
                npz_file, 
                cv_folds=5, 
                random_state=42, 
                response_type=response_type,
                normalize=True
            )
            
        except Exception as e:
            print(f"Error processing {response_type}: {e}")
            import traceback
            traceback.print_exc()
            continue