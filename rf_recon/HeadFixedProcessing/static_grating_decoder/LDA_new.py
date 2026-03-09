import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def extract_features_from_npz(npz_file, time_window=(0, 0.5)):
    """
    Extract firing rate features from NPZ file for LDA analysis.
    
    Parameters:
        npz_file: Path to NPZ file
        time_window: Tuple (start, end) in seconds relative to stimulus onset
    
    Returns:
        X: Feature matrix (n_trials, n_units)
        y: Labels (orientation indices)
        trial_info: Dictionary with trial metadata
    """
    data = np.load(npz_file, allow_pickle=True)
    units_data = data['units_data']
    
    n_units = len(units_data)
    
    # Get number of trials from first unit
    n_trials = len(units_data[0]['trials'])
    
    # Feature matrix: each row is a trial, columns are firing rates per unit
    X = np.zeros((n_trials, n_units))
    y = np.zeros(n_trials, dtype=int)
    
    trial_metadata = {
        'orientation': np.zeros(n_trials),
        'phase': np.zeros(n_trials),
        'spatial_freq': np.zeros(n_trials),
        'temporal_freq': np.zeros(n_trials),
        'repeat_idx': np.zeros(n_trials, dtype=int)
    }
    
    # Calculate window duration
    window_duration = time_window[1] - time_window[0]
    
    # Extract features for each trial
    for trial_idx in range(n_trials):
        # For each unit, compute firing rate in the time window
        for unit_idx, unit in enumerate(units_data):
            trial_data = unit['trials'][trial_idx]
            spike_times = trial_data['spike_times']
            
            # Count spikes in the time window
            spikes_in_window = np.sum((spike_times >= time_window[0]) & 
                                     (spike_times < time_window[1]))
            
            # Calculate firing rate (spikes/sec)
            firing_rate = spikes_in_window / window_duration
            
            X[trial_idx, unit_idx] = firing_rate
        
        # Get labels from first unit (same for all units)
        trial_data = units_data[0]['trials'][trial_idx]
        y[trial_idx] = trial_data['orientation_idx']
        
        trial_metadata['orientation'][trial_idx] = trial_data['orientation']
        trial_metadata['phase'][trial_idx] = trial_data['phase']
        trial_metadata['spatial_freq'][trial_idx] = trial_data['spatial_frequency']
        trial_metadata['temporal_freq'][trial_idx] = trial_data.get('temporal_frequency', 0)
        trial_metadata['repeat_idx'][trial_idx] = trial_data['repeat_idx']
    
    return X, y, trial_metadata


def perform_lda_analysis(npz_file, output_folder=None, time_window=(0, 0.5), 
                        n_cv_folds=5):
    """
    Perform LDA analysis on drifting grating responses.
    
    Parameters:
        npz_file: Path to NPZ file
        output_folder: Where to save figures (default: folder with same name as npz file)
        time_window: Time window for analysis (start, end) in seconds
        n_cv_folds: Number of cross-validation folds
    
    Returns:
        results: Dictionary containing LDA results and cross-validation scores
    """
    npz_file = Path(npz_file)
    if output_folder is None:
        # Create folder with same name as npz file (without extension)
        output_folder = npz_file.parent / npz_file.stem
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("Extracting features from NPZ file...")
    X, y, trial_metadata = extract_features_from_npz(
        npz_file, time_window=time_window
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of trials: {len(y)}")
    print(f"Number of orientations: {len(np.unique(y))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform LDA
    print("\nPerforming LDA...")
    n_components = min(len(np.unique(y)) - 1, 3)  # Max 3 for 3D visualization
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)
    
    print(f"LDA components: {n_components}")
    print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
    
    # Cross-validation
    print(f"\nPerforming {n_cv_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    lda_cv = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda_cv, X_scaled, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Plot 3D scatter if we have 3 components
    if n_components >= 3:
        plot_3d_lda_projection(X_lda, y, trial_metadata, output_folder)
    elif n_components == 2:
        plot_2d_lda_projection(X_lda, y, trial_metadata, output_folder)
    
    # Plot cross-validation results
    plot_cv_scores(cv_scores, len(np.unique(y)), output_folder)
    
    # Additional analysis: confusion matrix from cross-validation
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    
    y_pred = cross_val_predict(lda_cv, X_scaled, y, cv=cv)
    conf_matrix = confusion_matrix(y, y_pred)
    plot_confusion_matrix(conf_matrix, trial_metadata['orientation'], output_folder)
    
    results = {
        'lda': lda,
        'scaler': scaler,
        'X_lda': X_lda,
        'y': y,
        'cv_scores': cv_scores,
        'confusion_matrix': conf_matrix,
        'trial_metadata': trial_metadata,
        'explained_variance_ratio': lda.explained_variance_ratio_
    }
    
    return results


def plot_3d_lda_projection(X_lda, y, trial_metadata, output_folder):
    """Plot 3D scatter of LDA projection colored by orientation."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_orientations = np.unique(trial_metadata['orientation'])
    colors = cm.hsv(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
    
    for idx, ori in enumerate(unique_orientations):
        mask = trial_metadata['orientation'] == ori
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], X_lda[mask, 2],
                  c=[colors[idx]], label=f'{int(ori)}°', 
                  s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('LDA Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontsize=12, fontweight='bold')
    ax.set_zlabel('LDA Component 3', fontsize=12, fontweight='bold')
    ax.set_title('LDA 3D Scatter (full-data fit)', fontsize=14, fontweight='bold')
    
    # Place legend outside the plot
    ax.legend(title='Orientation', bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_folder / 'lda_3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'lda_3d_scatter.pdf', bbox_inches='tight')
    print(f"Saved 3D scatter plot to {output_folder / 'lda_3d_scatter.png'}")
    plt.close()


def plot_2d_lda_projection(X_lda, y, trial_metadata, output_folder):
    """Plot 2D scatter of LDA projection colored by orientation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_orientations = np.unique(trial_metadata['orientation'])
    colors = cm.hsv(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
    
    for idx, ori in enumerate(unique_orientations):
        mask = trial_metadata['orientation'] == ori
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                  c=[colors[idx]], label=f'{int(ori)}°', 
                  s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('LDA Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontsize=12, fontweight='bold')
    ax.set_title('LDA 2D Scatter (full-data fit)', fontsize=14, fontweight='bold')
    ax.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_folder / 'lda_2d_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'lda_2d_scatter.pdf', bbox_inches='tight')
    print(f"Saved 2D scatter plot to {output_folder / 'lda_2d_scatter.png'}")
    plt.close()


def plot_cv_scores(cv_scores, n_orientations, output_folder):
    """Plot cross-validation scores."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_folds = len(cv_scores)
    x_pos = np.arange(n_folds)
    
    ax.bar(x_pos, cv_scores, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {cv_scores.mean():.3f}')
    
    # Chance level is 1/n_orientations
    chance_level = 1.0 / n_orientations
    ax.axhline(y=chance_level, color='gray', linestyle=':', 
               linewidth=1, label=f'Chance level: {chance_level:.3f}')
    
    ax.set_xlabel('CV Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in range(n_folds)])
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_folder / 'cv_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'cv_scores.pdf', bbox_inches='tight')
    print(f"Saved CV scores plot to {output_folder / 'cv_scores.png'}")
    plt.close()


def plot_confusion_matrix(conf_matrix, orientations, output_folder):
    """Plot confusion matrix."""
    unique_ori = np.unique(orientations)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(unique_ori)))
    ax.set_yticks(np.arange(len(unique_ori)))
    ax.set_xticklabels([f'{int(o)}°' for o in unique_ori])
    ax.set_yticklabels([f'{int(o)}°' for o in unique_ori])
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Orientation', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Orientation', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Cross-Validation)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_folder / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'confusion_matrix.pdf', bbox_inches='tight')
    print(f"Saved confusion matrix to {output_folder / 'confusion_matrix.png'}")
    plt.close()


# Example usage
if __name__ == '__main__':
    npz_file = Path(input("Enter path to NPZ file: ").strip().strip('"'))
    
    # Optional parameters
    use_custom_params = input("Use custom parameters? (y/n, default=n): ").strip().lower()
    
    if use_custom_params == 'y':
        time_start = float(input("Time window start (sec, default=0): ") or "0")
        time_end = float(input("Time window end (sec, default=0.5): ") or "0.5")
        n_folds = int(input("Number of CV folds (default=5): ") or "5")
    else:
        time_start, time_end = 0.07, 0.2
        n_folds = 5
    
    # Perform analysis
    results = perform_lda_analysis(
        npz_file,
        time_window=(time_start, time_end),
        n_cv_folds=n_folds
    )
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)
    print(f"Mean CV accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")
    print(f"Explained variance: {results['explained_variance_ratio']}")
    print(f"Output saved to: {npz_file.parent / npz_file.stem}")