import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
import seaborn as sns
from scipy.stats import ttest_ind


def load_halfvisualfield_data(npz_file):
    """
    Load halfVisualField data from npz file and filter out noise units.
    
    Parameters:
    npz_file: Path to the npz file containing halfVisualField data
    
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
    
    # Extract metadata if available
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
    print(f"Original trials: {metadata['n_trials_original']}")
    print(f"Unique orientations: {np.unique(left_orientations)}")
    
    # Quality breakdown
    quality_counts = {}
    for q in good_unit_qualities:
        quality_counts[q] = quality_counts.get(q, 0) + 1
    print(f"Unit quality breakdown: {quality_counts}")
    
    return firing_rates, np.array(left_orientations), good_unit_info, good_unit_qualities, metadata


def compute_lda_decoding(firing_rates, orientations, cv_folds=5, random_state=42):
    """
    Perform LDA decoding of orientations from firing rates.
    
    Parameters:
    firing_rates: Array of shape (n_units, n_trials)
    orientations: Array of orientations for each trial
    cv_folds: Number of cross-validation folds
    random_state: Random seed for reproducibility
    
    Returns:
    clf: Fitted LDA classifier
    X_lda: LDA-transformed data
    scores: Cross-validation scores
    y_pred_cv: Cross-validation predictions
    skf: StratifiedKFold object
    le: LabelEncoder object
    """
    # Transpose to get (n_trials, n_units) format for sklearn
    X = firing_rates.T
    
    # Encode orientations as integers
    le = LabelEncoder()
    y_cls = le.fit_transform(orientations)
    
    # Check if we have enough trials per class for CV
    unique_classes, class_counts = np.unique(y_cls, return_counts=True)
    min_class_count = np.min(class_counts)
    
    if min_class_count < cv_folds:
        print(f"Warning: Minimum class has only {min_class_count} trials, reducing CV folds to {min_class_count}")
        cv_folds = min(cv_folds, min_class_count)
    
    # Perform LDA
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y_cls, cv=skf)
    
    # Get cross-validation predictions for per-class analysis
    y_pred_cv = cross_val_predict(clf, X, y_cls, cv=skf)
    
    # Fit on all data and transform
    clf.fit(X, y_cls)
    X_lda = clf.transform(X)
    
    print(f"LDA components: {X_lda.shape[1]}")
    print(f"Explained variance ratio: {clf.explained_variance_ratio_}")
    
    return clf, X_lda, scores, y_pred_cv, skf, le


def plot_lda_projection_3d(X_lda, y_cls, le, ax_3d):
    """
    Plot LDA projections in 3D
    """
    unique_oris = np.unique(y_cls)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_oris)))
    
    # 3D plot
    if X_lda.shape[1] >= 3:
        for i, ori_enc in enumerate(unique_oris):
            ori_val = le.inverse_transform([ori_enc])[0]
            idx = np.where(y_cls == ori_enc)[0]
            ax_3d.scatter(X_lda[idx, 0], X_lda[idx, 1], X_lda[idx, 2],
                         color=colors[i], label=f"{ori_val}°", alpha=0.7, s=50)
        
        ax_3d.set_xlabel("LDA Component 1", fontsize=12)
        ax_3d.set_ylabel("LDA Component 2", fontsize=12)
        ax_3d.set_zlabel("LDA Component 3", fontsize=12)
        ax_3d.set_title("LDA Projection (3D)", fontsize=14)
        ax_3d.legend(title="Orientation", fontsize=10)
    else:
        # If less than 3 components, show text message
        ax_3d.text(0.5, 0.5, 0.5, f'Only {X_lda.shape[1]} LDA component(s)\navailable.\nNeed 3+ for 3D plot.', 
                  ha='center', va='center', transform=ax_3d.transAxes, fontsize=12)
        ax_3d.set_title("3D LDA Not Available", fontsize=14)


def plot_decoding_results(X_lda, orientations, scores, y_pred_cv, le, unit_info, unit_qualities, 
                         metadata, npz_file, save_dir=None):
    """
    Create comprehensive plots of LDA decoding results with 3D visualization.
    
    Parameters:
    X_lda: LDA-transformed data
    orientations: Original orientation values
    scores: Cross-validation scores
    y_pred_cv: Cross-validation predictions
    le: LabelEncoder object
    unit_info: List of (shank, unit_id) tuples
    unit_qualities: List of quality strings
    metadata: Dictionary with session metadata
    npz_file: Path to original data file (for metadata)
    save_dir: Directory to save plots (optional)
    """
    if save_dir is None:
        save_dir = Path(npz_file).parent / 'lda_results'
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get metadata
    session_name = metadata['session']
    n_units = len(unit_info)
    n_trials = len(orientations)
    n_bad_trials = len(metadata['bad_trials'])
    
    # Encode orientations for plotting
    y_cls = le.transform(orientations)
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(20, 16))
    
    # 3D LDA plot
    ax1_3d = plt.subplot(2, 3, 1, projection='3d')
    plot_lda_projection_3d(X_lda, y_cls, le, ax1_3d)
    
    # Subplot positions for remaining plots
    subplot_positions = [(2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]
    
    # 2. Cross-validation accuracy
    ax2 = plt.subplot(*subplot_positions[0])
    fold_numbers = range(1, len(scores) + 1)
    bars = ax2.bar(fold_numbers, scores, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("CV Fold", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.set_title(f"Cross-Validation Accuracy\nMean: {scores.mean():.3f} ± {scores.std():.3f}", fontsize=16)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Per-orientation accuracy (CORRECTED)
    ax3 = plt.subplot(*subplot_positions[1])
    unique_oris = np.unique(y_cls)
    
    # Calculate per-orientation accuracy using the same CV predictions
    ori_accuracies = []
    ori_labels = []
    ori_sample_counts = []
    
    for ori_enc in unique_oris:
        ori_val = le.inverse_transform([ori_enc])[0]
        ori_labels.append(f"{ori_val}°")
        ori_mask = y_cls == ori_enc
        ori_acc = accuracy_score(y_cls[ori_mask], y_pred_cv[ori_mask])
        ori_accuracies.append(ori_acc)
        ori_sample_counts.append(np.sum(ori_mask))
    
    # Weighted average (should match CV accuracy)
    weighted_avg = np.average(ori_accuracies, weights=ori_sample_counts)
    unweighted_avg = np.mean(ori_accuracies)
    
    bars = ax3.bar(ori_labels, ori_accuracies, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Orientation", fontsize=14)
    ax3.set_ylabel("Accuracy", fontsize=14)
    ax3.set_title(f"Per-Orientation Accuracy\nWeighted Avg: {weighted_avg:.3f}, Unweighted Avg: {unweighted_avg:.3f}", fontsize=14)
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars with sample counts
    for bar, acc, count in zip(bars, ori_accuracies, ori_sample_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    # 4. Confusion matrix
    ax4 = plt.subplot(*subplot_positions[2])
    cm = confusion_matrix(y_cls, y_pred_cv)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ori_labels, yticklabels=ori_labels, ax=ax4)
    ax4.set_xlabel("Predicted Orientation", fontsize=14)
    ax4.set_ylabel("True Orientation", fontsize=14)
    ax4.set_title("Confusion Matrix", fontsize=16)
    
    # 5. Real vs shuffled comparison
    ax5 = plt.subplot(*subplot_positions[3])
    
    # Shuffle the data
    np.random.seed(42)  # For reproducibility
    X_shuffled = X_lda.copy()
    np.random.shuffle(X_shuffled)
    scores_shuffled = cross_val_score(LinearDiscriminantAnalysis(), X_shuffled, y_cls, 
                                    cv=min(5, len(scores)))
    
    means = [scores.mean(), scores_shuffled.mean()]
    stds = [scores.std(), scores_shuffled.std()]
    
    bars = ax5.bar(['Real', 'Shuffled'], means, yerr=stds,
                   color=['dodgerblue', 'salmon'], edgecolor='black', capsize=8, alpha=0.7)
    ax5.set_ylabel("Accuracy", fontsize=14)
    ax5.set_ylim(0, 1)
    ax5.set_title("Real vs Shuffled Accuracy", fontsize=16)
    ax5.grid(True, alpha=0.3)
    
    # Significance testing
    t_stat, p_val = ttest_ind(scores, scores_shuffled)
    sig_label = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    # Add significance bar
    max_height = max(m + s for m, s in zip(means, stds))
    ax5.plot([0, 0, 1, 1], [max_height + 0.02, max_height + 0.03, max_height + 0.03, max_height + 0.02], 
             'k-', linewidth=1.5)
    ax5.text(0.5, max_height + 0.04, f"{sig_label} (p={p_val:.3f})", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 6. Unit quality and shank distribution
    ax6 = plt.subplot(*subplot_positions[4])
    
    # Create combined quality and shank information
    shank_quality_info = {}
    for (shank, unit_id), quality in zip(unit_info, unit_qualities):
        key = f"Shank {shank}\n({quality})"
        shank_quality_info[key] = shank_quality_info.get(key, 0) + 1
    
    if shank_quality_info:
        keys = list(shank_quality_info.keys())
        counts = list(shank_quality_info.values())
        
        # Color by quality
        colors = []
        for key in keys:
            if 'good' in key:
                colors.append('green')
            elif 'mua' in key:
                colors.append('orange')
            else:
                colors.append('gray')
        
        bars = ax6.bar(range(len(keys)), counts, color=colors, edgecolor='black', alpha=0.7)
        ax6.set_xlabel("Shank (Quality)", fontsize=14)
        ax6.set_ylabel("Number of Units", fontsize=14)
        ax6.set_title("Units by Shank and Quality", fontsize=16)
        ax6.set_xticks(range(len(keys)))
        ax6.set_xticklabels(keys, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=12)
    
    # Main title
    fig.suptitle(f"HalfVisualField LDA Decoding Results\n"
                f"Session: {session_name} | Units: {n_units} | Trials: {n_trials} "
                f"(excluded {n_bad_trials} bad trials)\n"
                f"Overall CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f} | "
                f"LDA Components: {X_lda.shape[1]}", 
                fontsize=18, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    fig.savefig(save_dir / 'lda_decoding_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved decoding results to {save_dir}")
    
    # Print detailed per-orientation analysis
    print(f"\nPer-Orientation Analysis:")
    print(f"CV Accuracy (overall): {scores.mean():.3f}")
    print(f"Weighted average of per-orientation accuracies: {weighted_avg:.3f}")
    print(f"Unweighted average of per-orientation accuracies: {unweighted_avg:.3f}")
    print(f"Difference (weighted - CV): {weighted_avg - scores.mean():.3f}")
    
    return scores


def run_halfvisualfield_lda_decoding(npz_file, cv_folds=5, random_state=42, save_dir=None):
    """
    Main function to run LDA decoding on halfVisualField data.
    
    Parameters:
    npz_file: Path to the halfVisualField npz file
    cv_folds: Number of cross-validation folds
    random_state: Random seed for reproducibility
    save_dir: Directory to save results (optional)
    
    Returns:
    scores: Cross-validation scores
    """
    print(f"Running LDA decoding on {npz_file}")
    
    # Load and filter data
    firing_rates, orientations, unit_info, unit_qualities, metadata = load_halfvisualfield_data(npz_file)
    
    # Check if we have enough data
    if len(unit_info) == 0:
        print("No non-noise units found!")
        return None
    
    unique_orientations = np.unique(orientations)
    if len(unique_orientations) < 2:
        print("Need at least 2 different orientations for decoding!")
        return None
    
    print(f"Found {len(unique_orientations)} unique orientations: {unique_orientations}")
    
    # Perform LDA decoding
    clf, X_lda, scores, y_pred_cv, skf, le = compute_lda_decoding(
        firing_rates, orientations, cv_folds, random_state
    )
    
    # Create plots
    plot_decoding_results(X_lda, orientations, scores, y_pred_cv, le, unit_info, unit_qualities, 
                         metadata, npz_file, save_dir)
    
    # Print summary statistics
    print(f"\nDecoding Summary:")
    print(f"Session: {metadata['session']}")
    print(f"Number of units: {len(unit_info)}")
    print(f"Number of trials: {len(orientations)} (filtered from {metadata['n_trials_original']})")
    print(f"Bad trials excluded: {len(metadata['bad_trials'])}")
    print(f"Unique orientations: {unique_orientations}")
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Individual fold accuracies: {scores}")
    print(f"LDA components: {X_lda.shape[1]}")
    
    # Chance level calculation
    chance_level = 1.0 / len(unique_orientations)
    print(f"Chance level: {chance_level:.3f}")
    print(f"Above chance: {'Yes' if scores.mean() > chance_level else 'No'}")
    
    return scores


if __name__ == '__main__':
    # Example usage
    npz_file = r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22sg\CnL22SG_20250727_172100\HalfGrating\HalfGrating_data.npz"
    if Path(npz_file).exists():
        scores = run_halfvisualfield_lda_decoding(npz_file, cv_folds=5, random_state=42)
        if scores is not None:
            print(f"\nDecoding completed successfully!")
            print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        print(f"File not found: {npz_file}")
        print("Please update the path to your halfVisualField data file.")