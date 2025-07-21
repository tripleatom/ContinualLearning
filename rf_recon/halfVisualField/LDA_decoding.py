import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
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
    """
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract data
    left_orientations = data['left_orientation']
    all_units_responses = data['all_units_responses']
    unit_info = data['unit_info']
    unit_qualities = data['unit_qualities']
    
    # Filter out noise units
    noise_mask = np.array(unit_qualities) != 'noise'
    good_units_responses = [resp for i, resp in enumerate(all_units_responses) if noise_mask[i]]
    good_unit_info = [info for i, info in enumerate(unit_info) if noise_mask[i]]
    good_unit_qualities = [qual for i, qual in enumerate(unit_qualities) if noise_mask[i]]
    
    # Convert to numpy array
    firing_rates = np.array([resp['mean_firing_rates'] for resp in good_units_responses])
    
    print(f"Loaded {len(good_units_responses)} non-noise units from {len(all_units_responses)} total units")
    print(f"Number of trials: {len(left_orientations)}")
    print(f"Unique orientations: {np.unique(left_orientations)}")
    
    return firing_rates, np.array(left_orientations), good_unit_info, good_unit_qualities


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
    skf: StratifiedKFold object
    le: LabelEncoder object
    """
    # Transpose to get (n_trials, n_units) format for sklearn
    X = firing_rates.T
    
    # Encode orientations as integers
    le = LabelEncoder()
    y_cls = le.fit_transform(orientations)
    
    # Perform LDA
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y_cls, cv=skf)
    
    # Fit on all data and transform
    clf.fit(X, y_cls)
    X_lda = clf.transform(X)
    
    return clf, X_lda, scores, skf, le


def plot_decoding_results(X_lda, orientations, scores, le, unit_info, unit_qualities, 
                         npz_file, save_dir=None):
    """
    Create comprehensive plots of LDA decoding results.
    
    Parameters:
    X_lda: LDA-transformed data
    orientations: Original orientation values
    scores: Cross-validation scores
    le: LabelEncoder object
    unit_info: List of (shank, unit_id) tuples
    unit_qualities: List of quality strings
    npz_file: Path to original data file (for metadata)
    save_dir: Directory to save plots (optional)
    """
    if save_dir is None:
        save_dir = Path(npz_file).parent / 'lda_results'
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get metadata
    session_name = Path(npz_file).parent.name
    n_units = len(unit_info)
    n_trials = len(orientations)
    
    # Encode orientations for plotting
    y_cls = le.transform(orientations)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. LDA scatter plot (top left)
    ax1 = plt.subplot(2, 3, 1)
    unique_oris = np.unique(y_cls)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_oris)))
    
    for i, ori_enc in enumerate(unique_oris):
        ori_val = le.inverse_transform([ori_enc])[0]
        idx = np.where(y_cls == ori_enc)[0]
        ax1.scatter(X_lda[idx, 0], X_lda[idx, 1], 
                   color=colors[i], label=f"{ori_val}°", alpha=0.7, s=50)
    
    ax1.set_xlabel("LDA Component 1", fontsize=14)
    ax1.set_ylabel("LDA Component 2", fontsize=14)
    ax1.set_title("LDA Projection (2D)", fontsize=16)
    ax1.legend(title="Orientation", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-validation accuracy (top middle)
    ax2 = plt.subplot(2, 3, 2)
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
    
    # 3. Per-orientation accuracy (top right)
    ax3 = plt.subplot(2, 3, 3)
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_lda, y_cls, cv=5)
    
    ori_accuracies = []
    ori_labels = []
    for ori_enc in unique_oris:
        ori_val = le.inverse_transform([ori_enc])[0]
        ori_labels.append(f"{ori_val}°")
        ori_mask = y_cls == ori_enc
        ori_acc = accuracy_score(y_cls[ori_mask], y_pred[ori_mask])
        ori_accuracies.append(ori_acc)
    
    bars = ax3.bar(ori_labels, ori_accuracies, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Orientation", fontsize=14)
    ax3.set_ylabel("Accuracy", fontsize=14)
    ax3.set_title("Per-Orientation Accuracy", fontsize=16)
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, ori_accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Confusion matrix (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_cls, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ori_labels, yticklabels=ori_labels, ax=ax4)
    ax4.set_xlabel("Predicted Orientation", fontsize=14)
    ax4.set_ylabel("True Orientation", fontsize=14)
    ax4.set_title("Confusion Matrix", fontsize=16)
    
    # 5. Real vs shuffled comparison (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    
    # Shuffle the data
    X_shuffled = X_lda[np.random.permutation(len(X_lda))]
    scores_shuffled = cross_val_score(LinearDiscriminantAnalysis(), X_shuffled, y_cls, cv=5)
    
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
    
    # 6. Unit quality distribution (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    quality_counts = {}
    for quality in unit_qualities:
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    if quality_counts:
        qualities = list(quality_counts.keys())
        counts = list(quality_counts.values())
        colors = ['green' if q == 'good' else 'orange' if q == 'mua' else 'red' for q in qualities]
        
        bars = ax6.bar(qualities, counts, color=colors, edgecolor='black', alpha=0.7)
        ax6.set_xlabel("Unit Quality", fontsize=14)
        ax6.set_ylabel("Number of Units", fontsize=14)
        ax6.set_title("Unit Quality Distribution", fontsize=16)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=12)
    
    # Main title
    fig.suptitle(f"HalfVisualField LDA Decoding Results\n"
                f"Session: {session_name} | Units: {n_units} | Trials: {n_trials}\n"
                f"Overall CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", 
                fontsize=18, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    fig.savefig(save_dir / 'lda_decoding_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved decoding results to {save_dir}")
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
    firing_rates, orientations, unit_info, unit_qualities = load_halfvisualfield_data(npz_file)
    
    # Check if we have enough data
    if len(unit_info) == 0:
        print("No non-noise units found!")
        return None
    
    if len(np.unique(orientations)) < 2:
        print("Need at least 2 different orientations for decoding!")
        return None
    
    # Perform LDA decoding
    clf, X_lda, scores, skf, le = compute_lda_decoding(
        firing_rates, orientations, cv_folds, random_state
    )
    
    # Create plots
    plot_decoding_results(X_lda, orientations, scores, le, unit_info, unit_qualities, 
                         npz_file, save_dir)
    
    # Print summary statistics
    print(f"\nDecoding Summary:")
    print(f"Number of units: {len(unit_info)}")
    print(f"Number of trials: {len(orientations)}")
    print(f"Unique orientations: {np.unique(orientations)}")
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Individual fold accuracies: {scores}")
    
    return scores


if __name__ == '__main__':
    # Example usage
    npz_file = r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL38SG\CnL38SG_20250712_181216\freelymovingRF\freelymovingRF_data.npz"
    if Path(npz_file).exists():
        scores = run_halfvisualfield_lda_decoding(npz_file, cv_folds=5, random_state=42)
        if scores is not None:
            print(f"\nDecoding completed successfully!")
            print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        print(f"File not found: {npz_file}")
        print("Please update the path to your halfVisualField data file.") 