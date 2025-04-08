import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def decoding_accuracy(npz_file, threshold, cv_folds=5, random_state=42):
    """
    Runs the decoding pipeline for a single gOSI threshold.
    
    Returns:
      cv_scores (np.ndarray): cross-validated decoding accuracy scores.
    """
    npz_file = Path(npz_file)
    data = np.load(npz_file, allow_pickle=True)
    # Expected keys: 'all_units_responses', 'unique_orientation', 'unit_qualities', 'all_shank_info'
    all_units_responses = data['all_units_responses']  # shape: (n_units, n_ori, n_phase, n_sf, n_repeats)
    unique_orientation = data['unique_orientation']
    unit_qualities = np.array(data['unit_qualities'])


    metrics_file = npz_file.parent /"static_grating_tuning_metrics.npz"
    data = np.load(metrics_file, allow_pickle=True)
    
    # Load the dictionary with shank info to extract gOSI for each unit.
    # It is assumed the NPZ file has a key "all_shank_info" that is a dictionary.
    all_shank_info = data["all_shank_info"].item()
    all_gOSI = []
    for shank_id, units_dict in all_shank_info.items():
        for unit_id, metrics_dict in units_dict.items():
            # NOTE: In your code, gOSI was extracted as metrics_dict['OSI'].
            gOSI_value = metrics_dict['gOSI']
            all_gOSI.append(gOSI_value)
    all_gOSI = np.array(all_gOSI)
    
    # Filter: remove units labeled as 'noise' and units with gOSI <= threshold.
    valid_units_mask = (unit_qualities != 'noise') & (all_gOSI > threshold)
    filtered_responses = all_units_responses[valid_units_mask, ...]
    print(f"Threshold {threshold:.2f}: {all_units_responses.shape[0]} units originally, "
          f"{filtered_responses.shape[0]} units after filtering.")
    
    # Build the feature matrix X and labels y.
    n_units, n_ori, n_phase, n_sf, n_repeats = filtered_responses.shape
    n_trials = n_ori * n_phase * n_sf * n_repeats
    # Reshape: each trial is a row, each unit is a feature.
    responses_transposed = np.transpose(filtered_responses, (1, 2, 3, 4, 0))
    X = responses_transposed.reshape(n_trials, n_units)
    # Label vector: each trial's orientation.
    y = np.repeat(unique_orientation, n_phase * n_sf * n_repeats)
    y = np.array(y)
    le = LabelEncoder()
    y_class = le.fit_transform(y)
    
    # Run LDA decoding with cross-validation.
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X, y_class, cv=skf)
    return cv_scores

def compare_thresholds(npz_file, thresholds, cv_folds=5, random_state=42):
    """
    Loops over an array of thresholds, computes decoding accuracy for each,
    and visualizes the result.
    """
    mean_accuracies = []
    std_accuracies = []
    
    # Loop over each threshold value.
    for thr in thresholds:
        cv_scores = decoding_accuracy(npz_file, thr, cv_folds=cv_folds, random_state=random_state)
        mean_accuracies.append(np.mean(cv_scores))
        std_accuracies.append(np.std(cv_scores))
    
    # Visualize the result: decoding accuracy versus threshold.
    plt.figure(figsize=(8,6))
    plt.errorbar(thresholds, mean_accuracies, yerr=std_accuracies, fmt='-o', capsize=5, color='navy')
    plt.xlabel('gOSI Threshold')
    plt.ylabel('Decoding Accuracy')
    plt.title('Decoding Accuracy vs gOSI Threshold')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()
    
    # Optionally, print the results.
    for thr, mean_acc, std_acc in zip(thresholds, mean_accuracies, std_accuracies):
        print(f"Threshold {thr:.2f}: Accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

if __name__ == '__main__':
    # Path to your NPZ file.
    npz_file_path = '/Volumes/xieluanlabs/xl_cl/code/sortout/CnL22/250314_174049/static_grating_responses.npz'
    # Define a set of thresholds to compare.
    thresholds = np.linspace(0.05, 0.3, 6)  # e.g., [0.1, 0.2, 0.3, 0.4, 0.5]
    compare_thresholds(npz_file_path, thresholds)