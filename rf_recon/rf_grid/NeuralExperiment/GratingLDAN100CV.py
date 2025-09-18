import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from GratingLDA100Trials import load_neural_data, calculate_grating_firing_rates
import warnings
warnings.filterwarnings('ignore')


def create_train_test_split(firing_rates, orientation_labels, train_per_ori=40, test_per_ori=20, seed=42):
    """Create balanced train/test split."""
    np.random.seed(seed)
    unique_orientations = np.unique(orientation_labels)
    
    print(f"Creating split: {train_per_ori} train, {test_per_ori} test per orientation")
    
    train_indices = []
    test_indices = []
    
    for ori in unique_orientations:
        ori_indices = np.where(orientation_labels == ori)[0]
        available = len(ori_indices)
        needed = train_per_ori + test_per_ori
        
        if needed > available:
            print(f"Warning: {ori}° needs {needed} but has {available} trials")
            # Use what's available
            train_per_ori_actual = min(train_per_ori, available // 2)
            test_per_ori_actual = min(test_per_ori, available - train_per_ori_actual)
        else:
            train_per_ori_actual = train_per_ori
            test_per_ori_actual = test_per_ori
        
        # Randomly select trials
        shuffled = np.random.permutation(ori_indices)
        train_indices.extend(shuffled[:train_per_ori_actual])
        test_indices.extend(shuffled[train_per_ori_actual:train_per_ori_actual + test_per_ori_actual])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    return (firing_rates[train_indices], orientation_labels[train_indices],
            firing_rates[test_indices], orientation_labels[test_indices])


def create_pairwise_data(firing_rates, orientation_labels, ori1, ori2):
    """Extract data for specific pair of orientations."""
    mask = (orientation_labels == ori1) | (orientation_labels == ori2)
    pairwise_data = firing_rates[mask]
    pairwise_labels = orientation_labels[mask]
    return pairwise_data, pairwise_labels


def run_lda_analysis(train_data, train_labels, test_data, test_labels, scale=True):
    """Run LDA analysis and return results.

    Parameters:
    - scale: if True, apply `StandardScaler` fit on the train set.
             Set to False when data were already pre z-scored upstream.
    """
    # Optionally standardize data (fit on train only)
    if scale:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
    else:
        scaler = None
        train_scaled = train_data
        test_scaled = test_data
    
    # Train LDA classifier
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_scaled, train_labels)
    
    # Make predictions
    train_pred = lda.predict(train_scaled)
    test_pred = lda.predict(test_scaled)
    
    # Calculate accuracies
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    chance_acc = 1.0 / len(np.unique(train_labels))
    
    # Confusion matrix
    orientations = np.unique(test_labels)
    conf_matrix = confusion_matrix(test_labels, test_pred, labels=orientations)
    
    results = {
        'lda': lda,
        'scaler': scaler,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'chance_acc': chance_acc,
        'conf_matrix': conf_matrix,
        'orientations': orientations,
        'test_labels': test_labels,
        'test_pred': test_pred
    }
    
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Chance level: {chance_acc:.3f}")
    print(f"Improvement: {test_acc - chance_acc:.3f}")
    
    return results


def run_multiple_splits(firing_rates, orientation_labels, n_splits=5, save_dir=None, scale=True):
    """Run multiple train/test splits for robust evaluation."""
    print(f"\nRunning {n_splits} splits for robust evaluation")
    print("=" * 50)
    
    all_test_acc = []
    all_train_acc = []
    all_results = []
    
    for i in range(n_splits):
        print(f"\nSplit {i+1}/{n_splits}")
        
        # Create split
        train_data, train_labels, test_data, test_labels = create_train_test_split(
            firing_rates, orientation_labels, seed=42+i
        )
        
        # Run analysis
        results = run_lda_analysis(train_data, train_labels, test_data, test_labels, scale=scale)
        results['split_num'] = i + 1
        
        all_results.append(results)
        all_test_acc.append(results['test_acc'])
        all_train_acc.append(results['train_acc'])
    
    # Summary statistics
    test_mean = np.mean(all_test_acc)
    test_std = np.std(all_test_acc)
    train_mean = np.mean(all_train_acc)
    train_std = np.std(all_train_acc)
    
    print(f"\n{'='*50}")
    print("SUMMARY ACROSS ALL SPLITS:")
    print(f"Test Accuracy: {test_mean:.3f} ± {test_std:.3f}")
    print(f"Train Accuracy: {train_mean:.3f} ± {train_std:.3f}")
    print(f"Chance Level: {all_results[0]['chance_acc']:.3f}")
    print(f"Average improvement: {test_mean - all_results[0]['chance_acc']:.3f}")
    print(f"{'='*50}")
    
    # Plot results
    if save_dir:
        plot_multiple_splits_summary(all_test_acc, all_train_acc, 
                                   all_results[0]['chance_acc'], save_dir)
    
    return all_results, {'test_mean': test_mean, 'test_std': test_std,
                        'train_mean': train_mean, 'train_std': train_std}


def compare_training_sizes(firing_rates, orientation_labels, save_dir=None, scale=True):
    """Compare performance with different training set sizes."""
    print(f"\nComparing different training set sizes")
    print("=" * 50)
    
    train_sizes = np.arange(30, 171, 10)
    test_size = 30
    n_splits = 3
    
    results = {}
    
    for train_size in train_sizes:
        print(f"\nTesting with {train_size} training trials per orientation...")
        
        test_accs = []
        train_accs = []  # Added to track train accuracies
        for split in range(n_splits):
            try:
                train_data, train_labels, test_data, test_labels = create_train_test_split(
                    firing_rates, orientation_labels, 
                    train_per_ori=train_size, test_per_ori=test_size, seed=42+split
                )
                
                result = run_lda_analysis(train_data, train_labels, test_data, test_labels, scale=scale)
                test_accs.append(result['test_acc'])
                train_accs.append(result['train_acc'])  # Added
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if test_accs:
            test_mean = np.mean(test_accs)
            test_std = np.std(test_accs)
            train_mean = np.mean(train_accs)  # Added
            train_std = np.std(train_accs)    # Added
            results[train_size] = {
                'test_mean': test_mean, 'test_std': test_std,
                'train_mean': train_mean, 'train_std': train_std  # Added
            }
            print(f"  Test accuracy: {test_mean:.3f} ± {test_std:.3f}")
            print(f"  Train accuracy: {train_mean:.3f} ± {train_std:.3f}")  # Added
    
    # Plot training size comparison
    if save_dir and len(results) > 1:
        plot_training_size_comparison(results, save_dir)
    
    return results


def run_pairwise_analysis(firing_rates, orientation_labels, save_dir=None, selected_pairs=None, scale=True):
    """Run pairwise decoding analysis for all or selected orientation pairs."""
    print(f"\nPAIRWISE DECODING ANALYSIS")
    print("=" * 50)
    
    unique_orientations = np.unique(orientation_labels)
    
    # Generate all possible pairs or use selected pairs
    if selected_pairs is None:
        all_pairs = list(combinations(unique_orientations, 2))
        print(f"Running analysis for all {len(all_pairs)} possible pairs")
    else:
        all_pairs = selected_pairs
        print(f"Running analysis for {len(all_pairs)} selected pairs: {selected_pairs}")
    
    pairwise_results = {}
    
    for ori1, ori2 in all_pairs:
        print(f"\nAnalyzing pair: {ori1}° vs {ori2}°")
        
        # Extract pairwise data
        pair_data, pair_labels = create_pairwise_data(firing_rates, orientation_labels, ori1, ori2)
        
        # Check if we have enough data
        ori1_count = np.sum(pair_labels == ori1)
        ori2_count = np.sum(pair_labels == ori2)
        
        print(f"  {ori1}°: {ori1_count} trials, {ori2}°: {ori2_count} trials")
        
        if ori1_count < 20 or ori2_count < 20:
            print(f"  Skipping - insufficient trials")
            continue
        
        # Run multiple splits for this pair
        n_splits = 5
        test_accs = []
        train_accs = []
        
        for split in range(n_splits):
            try:
                train_data, train_labels, test_data, test_labels = create_train_test_split(
                    pair_data, pair_labels, train_per_ori=50, test_per_ori=30, seed=42+split
                )
                
                result = run_lda_analysis(train_data, train_labels, test_data, test_labels, scale=scale)
                test_accs.append(result['test_acc'])
                train_accs.append(result['train_acc'])
                
            except Exception as e:
                print(f"    Split {split+1} error: {e}")
                continue
        
        if test_accs:
            pair_key = f"{ori1}v{ori2}"
            pairwise_results[pair_key] = {
                'ori1': ori1,
                'ori2': ori2,
                'test_mean': np.mean(test_accs),
                'test_std': np.std(test_accs),
                'train_mean': np.mean(train_accs),
                'train_std': np.std(train_accs),
                'n_splits': len(test_accs)
            }
            
            print(f"  Results: Test {np.mean(test_accs):.3f}±{np.std(test_accs):.3f}, "
                  f"Train {np.mean(train_accs):.3f}±{np.std(train_accs):.3f}")
    
    # Plot pairwise results
    if save_dir and pairwise_results:
        plot_pairwise_results(pairwise_results, save_dir)
        plot_pairwise_matrix(pairwise_results, unique_orientations, save_dir)
    
    return pairwise_results


def plot_multiple_splits_summary(test_accs, train_accs, chance_acc, save_dir):
    """Plot summary of multiple splits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Individual split accuracies
    splits = range(1, len(test_accs) + 1)
    ax1.plot(splits, test_accs, 'o-', label='Test Accuracy', markersize=8, linewidth=2)
    ax1.plot(splits, train_accs, 's-', label='Train Accuracy', markersize=8, linewidth=2)
    ax1.axhline(chance_acc, color='red', linestyle='--', label='Chance Level')
    ax1.set_xlabel('Split Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Across Different Splits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Summary statistics
    test_mean, test_std = np.mean(test_accs), np.std(test_accs)
    train_mean, train_std = np.mean(train_accs), np.std(train_accs)
    
    categories = ['Test', 'Train', 'Chance']
    means = [test_mean, train_mean, chance_acc]
    stds = [test_std, train_std, 0]
    colors = ['blue', 'green', 'red']
    
    bars = ax2.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Average Performance')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        if std > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'multiple_splits_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_training_size_comparison(results, save_dir):
    """Plot how performance changes with training set size - now includes both train and test accuracy."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    train_sizes = sorted(results.keys())
    test_means = [results[size]['test_mean'] for size in train_sizes]
    test_stds = [results[size]['test_std'] for size in train_sizes]
    train_means = [results[size]['train_mean'] for size in train_sizes]  # Added
    train_stds = [results[size]['train_std'] for size in train_sizes]    # Added
    
    # Plot both test and train accuracies
    ax.errorbar(train_sizes, test_means, yerr=test_stds, 
               marker='o', linewidth=2, markersize=8, capsize=5, 
               label='Test Accuracy', color='blue')
    ax.errorbar(train_sizes, train_means, yerr=train_stds, 
               marker='s', linewidth=2, markersize=8, capsize=5, 
               label='Train Accuracy', color='green')  # Added
    
    ax.set_xlabel('Training Trials per Orientation')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Training Set Size')
    ax.legend()  # Added to show both lines
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels for test accuracy
    for size, test_mean, test_std in zip(train_sizes, test_means, test_stds):
        ax.text(size, test_mean + test_std + 0.02, f'{test_mean:.3f}', 
               ha='center', va='bottom', fontsize=9, color='blue')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_size_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_pairwise_results(pairwise_results, save_dir):
    """Plot pairwise decoding results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    pairs = list(pairwise_results.keys())
    test_means = [pairwise_results[pair]['test_mean'] for pair in pairs]
    test_stds = [pairwise_results[pair]['test_std'] for pair in pairs]
    train_means = [pairwise_results[pair]['train_mean'] for pair in pairs]
    train_stds = [pairwise_results[pair]['train_std'] for pair in pairs]
    
    x_pos = np.arange(len(pairs))
    
    # Plot 1: Test accuracy for all pairs
    bars1 = ax1.bar(x_pos, test_means, yerr=test_stds, capsize=3, alpha=0.7, color='blue')
    ax1.axhline(0.5, color='red', linestyle='--', label='Chance Level')
    ax1.set_xlabel('Orientation Pairs')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Pairwise Decoding Performance')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{pairwise_results[pair]['ori1']}°v{pairwise_results[pair]['ori2']}°" 
                        for pair in pairs], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.4, 1.0])
    
    # Add value labels
    for bar, mean, std in zip(bars1, test_means, test_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Train vs Test comparison
    width = 0.35
    ax2.bar(x_pos - width/2, test_means, width, yerr=test_stds, 
           label='Test Accuracy', alpha=0.7, capsize=3, color='blue')
    ax2.bar(x_pos + width/2, train_means, width, yerr=train_stds, 
           label='Train Accuracy', alpha=0.7, capsize=3, color='green')
    ax2.axhline(0.5, color='red', linestyle='--', label='Chance Level')
    ax2.set_xlabel('Orientation Pairs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs Test Accuracy Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{pairwise_results[pair]['ori1']}°v{pairwise_results[pair]['ori2']}°" 
                        for pair in pairs], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'pairwise_decoding_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_pairwise_matrix(pairwise_results, unique_orientations, save_dir):
    """Plot pairwise decoding results as a matrix."""
    n_ori = len(unique_orientations)
    accuracy_matrix = np.full((n_ori, n_ori), np.nan)
    
    # Fill the matrix
    for pair_key, result in pairwise_results.items():
        ori1, ori2 = result['ori1'], result['ori2']
        ori1_idx = np.where(unique_orientations == ori1)[0][0]
        ori2_idx = np.where(unique_orientations == ori2)[0][0]
        
        # Fill both symmetric positions
        accuracy_matrix[ori1_idx, ori2_idx] = result['test_mean']
        accuracy_matrix[ori2_idx, ori1_idx] = result['test_mean']
    
    # Set diagonal to 1.0 (perfect discrimination within same orientation)
    np.fill_diagonal(accuracy_matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create masked array to handle NaN values
    masked_matrix = np.ma.masked_where(np.isnan(accuracy_matrix), accuracy_matrix)
    
    im = ax.imshow(masked_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy')
    
    # Set ticks and labels
    ax.set_xticks(range(n_ori))
    ax.set_yticks(range(n_ori))
    ax.set_xticklabels([f'{ori}°' for ori in unique_orientations])
    ax.set_yticklabels([f'{ori}°' for ori in unique_orientations])
    ax.set_xlabel('Orientation')
    ax.set_ylabel('Orientation')
    ax.set_title('Pairwise Decoding Accuracy Matrix')
    
    # Add text annotations
    for i in range(n_ori):
        for j in range(n_ori):
            if not np.isnan(accuracy_matrix[i, j]):
                text = ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                             ha="center", va="center", 
                             color="white" if accuracy_matrix[i, j] < 0.75 else "black",
                             fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'pairwise_accuracy_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(conf_matrix, orientations, save_dir, split_num=None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize confusion matrix
    conf_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(conf_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=[f'{ori}°' for ori in orientations],
           yticklabels=[f'{ori}°' for ori in orientations],
           title='Confusion Matrix (Normalized)' + (f' - Split {split_num}' if split_num else ''),
           ylabel='True Orientation',
           xlabel='Predicted Orientation')
    
    # Add text annotations
    thresh = conf_norm.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, f'{conf_norm[i, j]:.2f}',
                   ha="center", va="center",
                   color="white" if conf_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Save plot
    filename = f'confusion_matrix{"_split" + str(split_num) if split_num else ""}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def main_analysis(data_file_path, selected_pairs=None, pre_zscore=True):
    """Main analysis function."""
    print("ENHANCED TRAIN/TEST LDA ANALYSIS WITH PAIRWISE DECODING")
    print("=" * 60)

    # Create save directory
    data_dir = os.path.dirname(data_file_path)
    save_dir_suffix = "prez" if pre_zscore else "raw"
    save_dir = os.path.join(data_dir, f"lda_analysis_results_{save_dir_suffix}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")
    
    try:
        # Load data
        print("Loading neural data...")
        data = load_neural_data(data_file_path)
        
        # Calculate firing rates
        firing_rates, orientation_labels, unit_ids, trial_info = calculate_grating_firing_rates(
            data, 
            time_window=(0.07, 0.16),
            subset_config=None
        )

        # Optional pre z-score across trials for each unit
        # Note: This uses all trials. For strict no-leakage setups, rely on `scale=True` below instead.
        if pre_zscore:
            mu = np.mean(firing_rates, axis=0)
            sigma = np.std(firing_rates, axis=0)
            sigma[sigma == 0] = 1.0
            firing_rates = (firing_rates - mu) / sigma
            print("Applied pre z-score normalization to firing rates (per unit, across all trials).")
        
        print(f"Loaded: {firing_rates.shape[0]} trials, {firing_rates.shape[1]} units")
        print(f"Orientations: {np.unique(orientation_labels)}")
        
        # Check trial counts
        for ori in np.unique(orientation_labels):
            count = np.sum(orientation_labels == ori)
            print(f"  {ori}°: {count} trials")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Analysis 1: Single split example
    print(f"\n{'='*60}")
    print("SINGLE SPLIT EXAMPLE")
    print("=" * 60)
    
    train_data, train_labels, test_data, test_labels = create_train_test_split(
        firing_rates, orientation_labels, train_per_ori=50, test_per_ori=30
    )

    # If we pre z-scored, skip StandardScaler here; otherwise use it
    use_scaler = not pre_zscore
    print(f"Scaler inside LDA: {'ON' if use_scaler else 'OFF'}")
    single_result = run_lda_analysis(train_data, train_labels, test_data, test_labels, scale=use_scaler)
    
    # Plot confusion matrix for single split
    plot_confusion_matrix(single_result['conf_matrix'], single_result['orientations'], 
                         save_dir, split_num=1)
    
    # Analysis 2: Multiple splits
    print(f"\n{'='*60}")
    print("MULTIPLE SPLITS ANALYSIS")
    print("=" * 60)
    
    all_results, summary_stats = run_multiple_splits(
        firing_rates, orientation_labels, n_splits=5, save_dir=save_dir, scale=use_scaler
    )
    
    # Analysis 3: Training size comparison
    print(f"\n{'='*60}")
    print("TRAINING SIZE COMPARISON")
    print("=" * 60)
    
    size_results = compare_training_sizes(
        firing_rates, orientation_labels, save_dir=save_dir, scale=use_scaler
    )
    
    # Analysis 4: Pairwise decoding
    print(f"\n{'='*60}")
    print("PAIRWISE DECODING ANALYSIS")
    print("=" * 60)
    
    pairwise_results = run_pairwise_analysis(
        firing_rates, orientation_labels, save_dir=save_dir, selected_pairs=selected_pairs, scale=use_scaler
    )
    
    print(f"\nAnalysis complete! Results saved to: {save_dir}")
    return all_results, summary_stats, size_results, pairwise_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run grating LDA analysis with optional pre z-scoring.")
    parser.add_argument("--data", dest="data_file_path", type=str, required=False,
                        default="/Volumes/xieluanlabs/xl_cl/code/sortout/CnL39SG/CnL39SG_20250821_163039/embedding_analysis/CnL39SG_CnL39SG_20250821_163039_grating_data.pkl",
                        help="Path to data file (.pkl/.h5/.npz)")
    parser.add_argument("--pre_zscore", dest="pre_zscore", action="store_true",
                        help="Apply pre z-scoring across all trials (per unit)")
    parser.add_argument("--no-pre_zscore", dest="pre_zscore", action="store_false",
                        help="Do not pre z-score (scaler in LDA will be used)")
    parser.set_defaults(pre_zscore=True)

    # Optional: selected pairs as comma-separated pairs like "0:45,90:135"
    parser.add_argument("--pairs", dest="pairs", type=str, default=None,
                        help="Selected orientation pairs, e.g., '0:45,0:90,90:135'. If omitted, runs all pairs.")

    args = parser.parse_args()

    if args.pairs:
        try:
            selected_pairs = []
            for token in args.pairs.split(','):
                a, b = token.split(':')
                selected_pairs.append((int(a), int(b)))
        except Exception:
            print("Failed to parse --pairs. Expected format like '0:45,90:135'. Falling back to all pairs.")
            selected_pairs = None
    else:
        selected_pairs = None

    results = main_analysis(args.data_file_path, selected_pairs=selected_pairs, pre_zscore=args.pre_zscore)
