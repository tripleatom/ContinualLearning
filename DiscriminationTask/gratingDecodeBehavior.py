"""
Cross-Session Grating-to-Behavior Decoder

Train a decoder on grating orientation data (45° vs 135°), then apply the same
decoder to behavior task data (reward left vs reward right).

Hypothesis: the same neural code underlies both tasks —
  45° grating  ↔  reward on left  (condition 1)
  135° grating ↔  reward on right (condition 0)

Supported decoders  (set DECODER in __main__)
  'lda'        – Linear Discriminant Analysis  (default)
  'svm_linear' – Support Vector Machine, linear kernel
  'svm_rbf'    – Support Vector Machine, RBF kernel
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add grating utilities to path
_GRATING_DIR = Path(__file__).parent.parent / 'passive_visual' / 'FreelyMovingProcessing' / 'Grating'
sys.path.insert(0, str(_GRATING_DIR))
import grating_utils


# =============================================================================
# DATA LOADING
# =============================================================================

def load_grating_data(grating_pkl_path, time_window=(0.07, 0.16),
                      target_orientations=(45.0, 135.0),
                      spatial_freq_filter=None):
    """
    Load grating data and filter to target orientations only.

    Parameters
    ----------
    spatial_freq_filter : float or None
        If set, keep only trials with this spatial frequency (e.g. 0.16).
        If None, all spatial frequencies are included.

    Returns:
        firing_rates : (n_trials, n_units)
        labels       : (n_trials,) — 1 for 45° (left-equiv), 0 for 135° (right-equiv)
        unit_ids     : list of unit ID strings
        trial_info   : metadata dict
    """
    data = grating_utils.load_neural_data(grating_pkl_path)

    # Filter out noise units (grating_utils doesn't do this by default)
    unit_info = data.get('unit_info', {})
    all_unit_ids = list(data['spike_data'].keys())
    good_units = [u for u in all_unit_ids
                  if unit_info.get(u, {}).get('quality', 'unknown') != 'noise']
    n_noise = len(all_unit_ids) - len(good_units)
    if n_noise:
        print(f"  [Grating] Excluded {n_noise} noise unit(s)")
        data_filtered = dict(data)
        data_filtered['spike_data'] = {u: data['spike_data'][u] for u in good_units}
    else:
        data_filtered = data

    firing_rates, orientation_labels, unit_ids, trial_info = \
        grating_utils.calculate_firing_rates(data_filtered, time_window=time_window)

    orientation_labels = orientation_labels.astype(float)

    # Filter to target orientations
    target_set = set(float(o) for o in target_orientations)
    mask = np.array([o in target_set for o in orientation_labels])

    if not np.any(mask):
        raise ValueError(
            f"No trials found for orientations {target_orientations}. "
            f"Available: {np.unique(orientation_labels)}"
        )

    sf_labels = trial_info.get('spatial_freq_labels')
    firing_rates = firing_rates[mask]
    orientation_labels = orientation_labels[mask]
    if sf_labels is not None:
        sf_labels = sf_labels[mask]

    # Optional: filter by spatial frequency
    if spatial_freq_filter is not None:
        if sf_labels is None:
            print("  [Grating] WARNING: no spatial frequency info found; skipping SF filter.")
        else:
            sf_mask = sf_labels == float(spatial_freq_filter)
            if not np.any(sf_mask):
                available = np.unique(sf_labels[~np.isnan(sf_labels.astype(float))])
                raise ValueError(
                    f"No trials found for spatial_freq={spatial_freq_filter}. "
                    f"Available SFs: {available}"
                )
            firing_rates = firing_rates[sf_mask]
            orientation_labels = orientation_labels[sf_mask]
            sf_labels = sf_labels[sf_mask]
            print(f"  [Grating] SF filter = {spatial_freq_filter} cpd → {sf_mask.sum()} trials kept")

    # Map orientations to binary labels: first target → 1 (left-equiv), second → 0
    left_ori = float(target_orientations[0])
    binary_labels = (orientation_labels == left_ori).astype(int)

    n_left  = np.sum(binary_labels == 1)
    n_right = np.sum(binary_labels == 0)
    sf_info = f", SF={spatial_freq_filter} cpd" if spatial_freq_filter is not None else ""
    print(f"\n[Grating] Filtered to {target_orientations[0]}°/{target_orientations[1]}° only{sf_info}:")
    print(f"  {target_orientations[0]}° (left-equiv):  {n_left} trials")
    print(f"  {target_orientations[1]}° (right-equiv): {n_right} trials")
    print(f"  Units: {len(unit_ids)}")

    trial_info['orientation_labels_deg'] = orientation_labels
    trial_info['binary_labels'] = binary_labels
    trial_info['n_left']  = int(n_left)
    trial_info['n_right'] = int(n_right)
    trial_info['left_ori']  = target_orientations[0]
    trial_info['right_ori'] = target_orientations[1]

    return firing_rates, binary_labels, unit_ids, trial_info


def load_grating_data_by_stim(grating_pkl_path, time_window, left_stim, right_stim):
    """
    Load grating data filtered to exactly the two (ori, SF) conditions
    specified by left_stim and right_stim.

    Parameters
    ----------
    left_stim  : dict with 'ori' and 'sf'  — mapped to class 1 (left-equiv)
    right_stim : dict with 'ori' and 'sf'  — mapped to class 0 (right-equiv)

    Returns
    -------
    firing_rates  : (n_trials, n_units)
    binary_labels : (n_trials,)  1=left_stim, 0=right_stim
    unit_ids      : list of str
    trial_info    : dict
    """
    data = grating_utils.load_neural_data(grating_pkl_path)

    unit_info = data.get('unit_info', {})
    all_unit_ids = list(data['spike_data'].keys())
    good_units = [u for u in all_unit_ids
                  if unit_info.get(u, {}).get('quality', 'unknown') != 'noise']
    n_noise = len(all_unit_ids) - len(good_units)
    if n_noise:
        print(f"  [Grating] Excluded {n_noise} noise unit(s)")
        data_filtered = dict(data)
        data_filtered['spike_data'] = {u: data['spike_data'][u] for u in good_units}
    else:
        data_filtered = data

    firing_rates, ori_labels, unit_ids, trial_info = \
        grating_utils.calculate_firing_rates(data_filtered, time_window=time_window)

    ori_labels = ori_labels.astype(float)
    sf_labels  = trial_info['spatial_freq_labels'].astype(float)

    left_ori,  left_sf  = float(left_stim['ori']),  float(left_stim['sf'])
    right_ori, right_sf = float(right_stim['ori']), float(right_stim['sf'])

    left_mask  = (ori_labels == left_ori)  & (sf_labels == left_sf)
    right_mask = (ori_labels == right_ori) & (sf_labels == right_sf)
    keep_mask  = left_mask | right_mask

    if not keep_mask.any():
        raise ValueError(
            f"No trials found for left_stim={left_stim} or right_stim={right_stim}. "
            f"Available ori: {np.unique(ori_labels)}, SF: {np.unique(sf_labels)}"
        )

    firing_rates  = firing_rates[keep_mask]
    binary_labels = np.where(left_mask[keep_mask], 1, 0)

    n_left  = int(binary_labels.sum())
    n_right = int((binary_labels == 0).sum())
    print(f"\n[Grating] Filtered to target (ori, SF) conditions:")
    print(f"  Left  ori={left_ori}°  SF={left_sf}:  {n_left} trials")
    print(f"  Right ori={right_ori}°  SF={right_sf}: {n_right} trials")
    print(f"  Units: {len(unit_ids)}")

    trial_info['binary_labels'] = binary_labels
    trial_info['n_left']  = n_left
    trial_info['n_right'] = n_right
    trial_info['left_stim']  = left_stim
    trial_info['right_stim'] = right_stim

    return firing_rates, binary_labels, unit_ids, trial_info


def load_behavior_data(behavior_pkl_path, time_window=(0.0, 1.0)):
    """
    Load behavior trial data.

    Returns:
        firing_rates      : (n_trials, n_units)
        condition_labels  : (n_trials,) — 1=left, 0=right
        unit_ids          : list of unit ID strings
        trial_info        : metadata dict
    """
    filepath = Path(behavior_pkl_path)
    print(f"\nLoading behavior data from: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    unit_info = data.get('unit_info', {})
    unit_ids = [
        uid for uid in data['spike_data'].keys()
        if unit_info.get(uid, {}).get('quality', 'unknown') != 'noise'
    ]
    n_noise = len(data['spike_data']) - len(unit_ids)
    if n_noise:
        print(f"  [Behavior] Excluded {n_noise} noise unit(s)")

    trial_info_dict = data['trial_info']
    if 'white_on_left' in trial_info_dict:
        condition_key = 'white_on_left'
        left_label, right_label = 'White on Left', 'White on Right'
    elif 'rewarded_on_left' in trial_info_dict:
        condition_key = 'rewarded_on_left'
        left_label, right_label = 'Reward on Left', 'Reward on Right'
    else:
        raise KeyError("trial_info must contain 'white_on_left' or 'rewarded_on_left'")

    condition_flags = np.array(trial_info_dict[condition_key])
    n_trials = len(condition_flags)
    n_units = len(unit_ids)
    window_start = time_window[0]

    print(f"\n[Behavior] Calculating firing rates:")
    print(f"  Units: {n_units} | Trials: {n_trials}")
    print(f"  {left_label}: {np.sum(condition_flags)} trials")
    print(f"  {right_label}: {np.sum(~condition_flags)} trials")

    firing_rates = np.full((n_trials, n_units), np.nan)
    window_duration = None

    for unit_idx, unit_id in enumerate(unit_ids):
        for trial_data in data['spike_data'][unit_id]:
            trial_idx = int(trial_data['trial_index'])
            if 0 <= trial_idx < n_trials:
                spike_times = np.array(trial_data['spike_times'])
                trial_duration = trial_data['trial_duration']
                window_end = trial_duration if time_window[1] is None \
                    else min(time_window[1], trial_duration)
                wd = window_end - window_start
                if window_duration is None:
                    window_duration = wd
                n_spikes = np.sum((spike_times >= window_start) & (spike_times < window_end))
                firing_rates[trial_idx, unit_idx] = n_spikes / wd

    valid_mask = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_mask]
    condition_labels = condition_flags[valid_mask].astype(int)

    print(f"  Valid trials: {np.sum(valid_mask)}/{n_trials}")

    trial_info = {
        'valid_trials_mask': valid_mask,
        'condition_names': {0: right_label, 1: left_label},
        'n_trials_per_condition': {
            0: int(np.sum(condition_labels == 0)),
            1: int(np.sum(condition_labels == 1))
        },
        'experiment_parameters': data.get('experiment_parameters', {}),
        'window_duration': window_duration
    }

    return firing_rates_clean, condition_labels, unit_ids, trial_info


# =============================================================================
# UNIT ALIGNMENT
# =============================================================================

def align_units(grating_unit_ids, behavior_unit_ids, grating_fr, behavior_fr):
    """
    Find shared units and reindex firing-rate matrices to the shared subset.

    Returns:
        shared_unit_ids    : list of shared unit ID strings
        grating_fr_shared  : (n_grating_trials, n_shared)
        behavior_fr_shared : (n_behavior_trials, n_shared)
    """
    shared = sorted(set(grating_unit_ids) & set(behavior_unit_ids))
    n_shared = len(shared)
    n_grating_only = len(set(grating_unit_ids) - set(behavior_unit_ids))
    n_behavior_only = len(set(behavior_unit_ids) - set(grating_unit_ids))

    print(f"\n[Alignment] Unit overlap:")
    print(f"  Grating units:  {len(grating_unit_ids)}")
    print(f"  Behavior units: {len(behavior_unit_ids)}")
    print(f"  Shared units:   {n_shared}")
    print(f"  Grating-only:   {n_grating_only}")
    print(f"  Behavior-only:  {n_behavior_only}")

    if n_shared == 0:
        raise ValueError(
            "No shared units found between grating and behavior datasets. "
            "Unit IDs must match (e.g. 'shank0_unit5'). Check that both PKL files "
            "are from the same recording session."
        )
    if n_shared < 5:
        print(f"  WARNING: Only {n_shared} shared units — decoder may be unreliable.")

    g_idx = [grating_unit_ids.index(u) for u in shared]
    b_idx = [behavior_unit_ids.index(u) for u in shared]

    return shared, grating_fr[:, g_idx], behavior_fr[:, b_idx]


# =============================================================================
# DECODER: BUILD, TRAIN, APPLY
# =============================================================================

def _build_model(decoder_type):
    """Return an unfitted sklearn classifier for the given decoder type."""
    if decoder_type == 'lda':
        return LinearDiscriminantAnalysis()
    elif decoder_type == 'svm_linear':
        return SVC(kernel='linear', probability=True, C=1.0)
    elif decoder_type == 'svm_rbf':
        return SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    else:
        raise ValueError(f"Unknown decoder_type '{decoder_type}'. "
                         f"Choose from: 'lda', 'svm_linear', 'svm_rbf'")


def _project(model, decoder_type, fr_scaled):
    """
    Return a 1D projection score array for scatter/histogram plots.
    LDA → transform (LD1); SVM → decision_function (signed distance to boundary).
    """
    if decoder_type == 'lda':
        return model.transform(fr_scaled)          # (n, 1)
    else:
        scores = model.decision_function(fr_scaled)  # (n,) for binary SVC
        return scores.reshape(-1, 1)


def train_decoder(grating_fr, grating_labels, decoder_type='lda'):
    """
    Fit StandardScaler + decoder on grating (45° vs 135°) data.

    Parameters
    ----------
    decoder_type : str
        'lda', 'svm_linear', or 'svm_rbf'

    Returns
    -------
    scaler, model, cv_scores, grating_proj (n_trials, 1)
    """
    scaler = StandardScaler()
    fr_scaled = scaler.fit_transform(grating_fr)

    model = _build_model(decoder_type)

    min_trials = min(np.sum(grating_labels == c) for c in np.unique(grating_labels))
    n_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, fr_scaled, grating_labels, cv=cv, scoring='accuracy')

    model.fit(fr_scaled, grating_labels)
    grating_proj = _project(model, decoder_type, fr_scaled)

    train_acc = accuracy_score(grating_labels, model.predict(fr_scaled))
    print(f"\n[Grating Decoder — {decoder_type.upper()}] Training:")
    print(f"  CV ({n_folds}-fold) accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Chance level: 0.500")

    return scaler, model, cv_scores, grating_proj


def decode_behavior(decoder_type, model, scaler, behavior_fr, behavior_labels):
    """
    Apply grating-trained decoder to behavior data.
    Uses scaler.transform() (not fit_transform) to preserve grating statistics.

    Returns
    -------
    predictions, probas, behavior_proj, accuracy, conf_matrix, sensitivity, specificity
    """
    fr_scaled = scaler.transform(behavior_fr)
    predictions = model.predict(fr_scaled)
    probas = model.predict_proba(fr_scaled)
    behavior_proj = _project(model, decoder_type, fr_scaled)

    acc = accuracy_score(behavior_labels, predictions)
    conf = confusion_matrix(behavior_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n[Behavior Decoding — {decoder_type.upper()}] Transfer results:")
    print(f"  Accuracy:      {acc:.3f}  (chance = 0.500)")
    print(f"  Above chance:  {acc - 0.5:+.3f}")
    print(f"  Sensitivity (Left):  {sensitivity:.3f}")
    print(f"  Specificity (Right): {specificity:.3f}")
    print(f"  Confusion matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

    return predictions, probas, behavior_proj, acc, conf, sensitivity, specificity


# =============================================================================
# VISUALIZATION
# =============================================================================

def _proj_label(decoder_type):
    return 'LD1 Score' if decoder_type == 'lda' else 'Decision Score'


def create_figure(grating_fr, grating_labels, grating_proj, grating_cv_scores,
                  behavior_labels, behavior_proj, behavior_predictions,
                  behavior_probas, behavior_acc, behavior_conf,
                  sensitivity, specificity,
                  shared_unit_ids, model, scaler, decoder_type,
                  grating_trial_info, behavior_trial_info,
                  run_params,
                  behavior_left_stim=None, behavior_right_stim=None,
                  save_path=None):
    """
    Create comprehensive cross-decoder visualization.

    run_params : dict with keys grating_pkl, behavior_pkl,
                 grating_time_window, behavior_time_window, spatial_freq_filter
    """
    # Build dynamic labels from stim dicts (fall back to trial_info if not provided)
    if behavior_left_stim is not None:
        _left_lbl  = f"ori={behavior_left_stim['ori']}° SF={behavior_left_stim['sf']} (left)"
        _right_lbl = f"ori={behavior_right_stim['ori']}° SF={behavior_right_stim['sf']} (right)"
        _left_ytick  = f"ori={behavior_right_stim['ori']}° SF={behavior_right_stim['sf']}"
        _right_ytick = f"ori={behavior_left_stim['ori']}° SF={behavior_left_stim['sf']}"
    else:
        _left_lbl    = f"{grating_trial_info.get('left_ori', '?')}° (left)"
        _right_lbl   = f"{grating_trial_info.get('right_ori', '?')}° (right)"
        _left_ytick  = f"{grating_trial_info.get('right_ori', '?')}°"
        _right_ytick = f"{grating_trial_info.get('left_ori', '?')}°"

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    colors_g = {1: '#E74C3C', 0: '#3498DB'}
    colors_b = {1: '#E74C3C', 0: '#3498DB'}
    cond_names = behavior_trial_info['condition_names']
    proj_lbl = _proj_label(decoder_type)

    fr_scaled_g = scaler.transform(grating_fr)
    grating_train_acc = accuracy_score(grating_labels, model.predict(fr_scaled_g))

    # --- 1. Grating projection scatter ---
    ax1 = fig.add_subplot(gs[0, 0])
    for lbl, name, color in [(1, _left_lbl, colors_g[1]),
                              (0, _right_lbl, colors_g[0])]:
        mask = grating_labels == lbl
        y = np.random.normal(lbl, 0.08, mask.sum())
        ax1.scatter(grating_proj[mask, 0], y, c=color, alpha=0.6, s=25, label=name)
    ax1.set_xlabel(proj_lbl, fontsize=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([_left_ytick, _right_ytick])
    ax1.set_title(f'Grating: {decoder_type.upper()} Projection\n(train)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- 2. Behavior projection scatter ---
    ax2 = fig.add_subplot(gs[0, 1])
    for lbl in [1, 0]:
        mask = behavior_labels == lbl
        y = np.random.normal(lbl, 0.08, mask.sum())
        ax2.scatter(behavior_proj[mask, 0], y, c=colors_b[lbl], alpha=0.6, s=25,
                    label=cond_names[lbl])
    ax2.set_xlabel(proj_lbl, fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([cond_names[0], cond_names[1]])
    ax2.set_title(f'Behavior: {decoder_type.upper()} Projection\n(transfer)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 3. Score distribution overlay ---
    ax3 = fig.add_subplot(gs[0, 2])
    for lbl, name, color in [(1, 'Left', colors_g[1]),
                              (0, 'Right', colors_g[0])]:
        ax3.hist(grating_proj[grating_labels == lbl, 0], bins=20, alpha=0.5, color=color,
                 density=True, histtype='stepfilled', label=f'Grating {name}')
        ax3.hist(behavior_proj[behavior_labels == lbl, 0], bins=20, alpha=0.5, color=color,
                 density=True, histtype='step', linewidth=2, linestyle='--',
                 label=f'Behavior {name}')
    ax3.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
                label='Decision boundary')
    ax3.set_xlabel(proj_lbl, fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Score Distribution\n(solid=grating, dashed=behavior)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- 4. Accuracy comparison bar ---
    ax4 = fig.add_subplot(gs[0, 3])
    bar_labels = ['Grating\n(CV)', 'Grating\n(train)', 'Behavior\n(transfer)']
    bar_values = [grating_cv_scores.mean(), grating_train_acc, behavior_acc]
    bar_errors = [grating_cv_scores.std(), 0, 0]
    bars4 = ax4.bar(range(3), bar_values, color=['#2ECC71', '#27AE60', '#E67E22'],
                    alpha=0.8, yerr=bar_errors, capsize=5, error_kw={'linewidth': 2})
    ax4.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Chance (0.50)')
    ax4.set_ylim([0, 1.05])
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(bar_labels, fontsize=9)
    ax4.set_ylabel('Accuracy', fontsize=10)
    ax4.set_title('Accuracy Comparison', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars4, bar_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- 5. Grating confusion matrix ---
    ax5 = fig.add_subplot(gs[1, 0])
    g_pred = model.predict(fr_scaled_g)
    g_conf = confusion_matrix(grating_labels, g_pred, labels=[0, 1])
    im5 = ax5.imshow(g_conf, cmap='Blues', interpolation='nearest')
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=[_left_ytick + '\n(right)', _right_ytick + '\n(left)'],
            yticklabels=[_left_ytick + '\n(right)', _right_ytick + '\n(left)'],
            xlabel='Predicted', ylabel='True',
            title='Grating Confusion\n(train)')
    thresh5 = g_conf.max() / 2
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, int(g_conf[i, j]), ha='center', va='center', fontsize=14,
                     color='white' if g_conf[i, j] > thresh5 else 'black', fontweight='bold')

    # --- 6. Behavior confusion matrix ---
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(behavior_conf, cmap='Oranges', interpolation='nearest')
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=[cond_names[0], cond_names[1]],
            yticklabels=[cond_names[0], cond_names[1]],
            xlabel='Predicted', ylabel='True',
            title='Behavior Confusion\n(transfer)')
    thresh6 = behavior_conf.max() / 2
    for i in range(2):
        for j in range(2):
            ax6.text(j, i, int(behavior_conf[i, j]), ha='center', va='center', fontsize=14,
                     color='white' if behavior_conf[i, j] > thresh6 else 'black', fontweight='bold')
    plt.setp(ax6.get_xticklabels(), rotation=20, ha='right')

    # --- 7. CV fold scores ---
    ax7 = fig.add_subplot(gs[1, 2])
    cv_bars = ax7.bar(range(len(grating_cv_scores)), grating_cv_scores,
                      alpha=0.7, color='#2ECC71')
    ax7.axhline(grating_cv_scores.mean(), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean: {grating_cv_scores.mean():.3f}')
    ax7.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Chance')
    ax7.set(xlabel='CV Fold', ylabel='Accuracy', ylim=[0, 1],
            title='Grating CV Scores\n(decoder quality)')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    for bar in cv_bars:
        h = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=8)

    # --- 8. Prediction confidence (behavior) ---
    ax8 = fig.add_subplot(gs[1, 3])
    confidence = np.max(behavior_probas, axis=1)
    correct_mask = behavior_predictions == behavior_labels
    ax8.hist(confidence[correct_mask], bins=20, alpha=0.7, color='green',
             label=f'Correct ({correct_mask.sum()})', density=True)
    ax8.hist(confidence[~correct_mask], bins=20, alpha=0.7, color='red',
             label=f'Incorrect ({(~correct_mask).sum()})', density=True)
    ax8.set(xlabel='Prediction Confidence', ylabel='Density',
            title='Behavior Prediction\nConfidence', xlim=[0, 1])
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # --- 9. Decoder weights (coefficients) ---
    ax9 = fig.add_subplot(gs[2, :2])
    coef = None
    if decoder_type == 'lda' and hasattr(model, 'coef_'):
        coef = model.coef_[0]
        coef_label = f'LDA Coefficient\n(positive = left, negative = right)'
    elif decoder_type == 'svm_linear' and hasattr(model, 'coef_'):
        coef = model.coef_[0]
        coef_label = f'SVM Weight\n(positive = left, negative = right)'

    if coef is not None:
        top_n = min(20, len(shared_unit_ids))
        top_idx = np.argsort(np.abs(coef))[::-1][:top_n]
        bar_colors_w = ['#E74C3C' if c > 0 else '#3498DB' for c in coef[top_idx]]
        ax9.barh(range(len(top_idx)), coef[top_idx], color=bar_colors_w, alpha=0.8)
        ax9.set(yticks=range(len(top_idx)),
                yticklabels=[shared_unit_ids[i] for i in top_idx],
                xlabel=coef_label,
                title=f'Top Discriminative Units ({decoder_type.upper()}, n={len(shared_unit_ids)} shared)')
        ax9.invert_yaxis()
        ax9.axvline(0, color='black', linewidth=1)
        ax9.grid(True, alpha=0.3, axis='x')
    else:
        # RBF SVM has no linear coefficients — show feature importance via |mean FR diff|
        imp_g = np.abs(
            fr_scaled_g[grating_labels == 1].mean(0) - fr_scaled_g[grating_labels == 0].mean(0)
        )
        top_n = min(20, len(shared_unit_ids))
        top_idx = np.argsort(imp_g)[::-1][:top_n]
        ax9.barh(range(len(top_idx)), imp_g[top_idx], alpha=0.8, color='purple')
        ax9.set(yticks=range(len(top_idx)),
                yticklabels=[shared_unit_ids[i] for i in top_idx],
                xlabel='Mean |FR difference| (left vs right)',
                title=f'Top Discriminative Units ({decoder_type.upper()}, n={len(shared_unit_ids)} shared)')
        ax9.invert_yaxis()
        ax9.grid(True, alpha=0.3, axis='x')

    # --- 10. Full meta-information panel ---
    ax10 = fig.add_subplot(gs[2, 2])
    ax10.axis('off')

    g_win = run_params['grating_time_window']
    b_win = run_params['behavior_time_window']
    b_win_str = (f"{b_win[0]:.2f}–{b_win[1]:.2f} s" if b_win[1] is not None
                 else f"{b_win[0]:.2f} s – trial end")
    n_left_b  = behavior_trial_info['n_trials_per_condition'][1]
    n_right_b = behavior_trial_info['n_trials_per_condition'][0]

    meta = (
        f"Run Parameters\n"
        f"{'─'*36}\n"
        f"Decoder:       {decoder_type.upper()}\n"
        f"Left stim:     {_left_lbl}\n"
        f"Right stim:    {_right_lbl}\n\n"
        f"[Grating — train]\n"
        f"  Window:      {g_win[0]:.2f}–{g_win[1]:.2f} s\n"
        f"  Left trials:  {grating_trial_info['n_left']}\n"
        f"  Right trials: {grating_trial_info['n_right']}\n"
        f"  CV acc:      {grating_cv_scores.mean():.3f} ± {grating_cv_scores.std():.3f}\n"
        f"  Train acc:   {grating_train_acc:.3f}\n\n"
        f"[Behavior — transfer]\n"
        f"  Window:      {b_win_str}\n"
        f"  Left trials: {n_left_b}\n"
        f"  Right trials:{n_right_b}\n"
        f"  Accuracy:    {behavior_acc:.3f}\n"
        f"  Above chance:{behavior_acc - 0.5:+.3f}\n"
        f"  Sensitivity: {sensitivity:.3f}\n"
        f"  Specificity: {specificity:.3f}\n\n"
        f"Shared units:  {len(shared_unit_ids)}\n"
        f"Grating units: {len(grating_fr[0]) if hasattr(grating_fr, '__len__') else '?'}"
    )
    ax10.text(0.04, 0.98, meta, transform=ax10.transAxes, fontsize=8.5,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # --- 11. Per-condition accuracy + file paths ---
    ax11 = fig.add_subplot(gs[2, 3])
    b_per = behavior_conf.diagonal() / behavior_conf.sum(axis=1)
    bars11 = ax11.bar(range(2), b_per, color=['#3498DB', '#E74C3C'], alpha=0.8)
    ax11.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Chance')
    ax11.set(xticks=[0, 1], xticklabels=[cond_names[0], cond_names[1]],
             ylabel='Accuracy', ylim=[0, 1],
             title='Behavior Per-Condition\nAccuracy (transfer)')
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars11, b_per):
        ax11.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}',
                  ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.setp(ax11.get_xticklabels(), rotation=20, ha='right')

    # File paths in figure footer
    grating_name = Path(run_params['grating_pkl']).name
    behavior_name = Path(run_params['behavior_pkl']).name
    fig.text(0.01, 0.005,
             f"Grating: {grating_name}   |   Behavior: {behavior_name}",
             fontsize=7, color='gray', ha='left', va='bottom', fontfamily='monospace')

    fig.suptitle(
        f"Grating → Behavior Cross-Session Decoder  [{decoder_type.upper()}]\n"
        f"Left: {_left_lbl}  •  Right: {_right_lbl}  •  tested on reward left vs right",
        fontsize=13, fontweight='bold', y=1.01
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_cross_decode(grating_pkl, behavior_pkl,
                     behavior_left_stim,
                     behavior_right_stim,
                     grating_time_window=(0.07, 0.16),
                     behavior_time_window=(0.0, 1.0),
                     decoder='lda',
                     save_plots=True,
                     output_path=None):
    """
    Full cross-session decode pipeline:
      1. Load grating data → filter to the two (ori, SF) conditions → train decoder
      2. Load behavior data → align units → apply decoder
      3. Evaluate and visualize

    Parameters
    ----------
    grating_pkl : str or Path
    behavior_pkl : str or Path
    behavior_left_stim  : dict with 'ori' and 'sf' — grating class for left (class 1)
    behavior_right_stim : dict with 'ori' and 'sf' — grating class for right (class 0)
    grating_time_window : tuple  (start, end) in seconds
    behavior_time_window : tuple  (start, end) in seconds; end=None uses trial duration
    decoder : str  'lda', 'svm_linear', or 'svm_rbf'
    save_plots : bool
    output_path : str or Path, optional

    Returns
    -------
    dict with keys: model, scaler, decoder_type, shared_unit_ids, grating_cv_scores,
                    behavior_accuracy, behavior_conf_matrix, sensitivity, specificity,
                    grating_proj, behavior_proj, grating_labels, behavior_labels,
                    behavior_predictions
    """
    print("=" * 60)
    print(f"Cross-Session Grating → Behavior Decoder  [{decoder.upper()}]")
    print(f"  Left  → ori={behavior_left_stim['ori']}°  SF={behavior_left_stim['sf']}")
    print(f"  Right → ori={behavior_right_stim['ori']}°  SF={behavior_right_stim['sf']}")
    print("=" * 60)

    # 1. Load data
    grating_fr, grating_labels, grating_unit_ids, grating_trial_info = \
        load_grating_data_by_stim(grating_pkl, grating_time_window,
                                  behavior_left_stim, behavior_right_stim)

    behavior_fr, behavior_labels, behavior_unit_ids, behavior_trial_info = \
        load_behavior_data(behavior_pkl, behavior_time_window)

    # 2. Align units
    shared_unit_ids, grating_fr_shared, behavior_fr_shared = align_units(
        grating_unit_ids, behavior_unit_ids, grating_fr, behavior_fr
    )

    # 3. Train decoder on grating data
    scaler, model, grating_cv_scores, grating_proj = \
        train_decoder(grating_fr_shared, grating_labels, decoder_type=decoder)

    # 4. Decode behavior data
    (behavior_predictions, behavior_probas, behavior_proj,
     behavior_acc, behavior_conf, sensitivity, specificity) = \
        decode_behavior(decoder, model, scaler, behavior_fr_shared, behavior_labels)

    # 5. Visualize
    run_params = {
        'grating_pkl':          str(grating_pkl),
        'behavior_pkl':         str(behavior_pkl),
        'grating_time_window':  grating_time_window,
        'behavior_time_window': behavior_time_window,
        'decoder':              decoder,
    }

    if output_path is None and save_plots:
        dec_tag = f'_{decoder}'
        output_path = (Path(behavior_pkl).parent / 'passive-behavior' /
                       (Path(behavior_pkl).stem + f'.grating_cross_decode{dec_tag}.png'))

    create_figure(
        grating_fr=grating_fr_shared,
        grating_labels=grating_labels,
        grating_proj=grating_proj,
        grating_cv_scores=grating_cv_scores,
        behavior_labels=behavior_labels,
        behavior_proj=behavior_proj,
        behavior_predictions=behavior_predictions,
        behavior_probas=behavior_probas,
        behavior_acc=behavior_acc,
        behavior_conf=behavior_conf,
        sensitivity=sensitivity,
        specificity=specificity,
        shared_unit_ids=shared_unit_ids,
        model=model,
        scaler=scaler,
        decoder_type=decoder,
        grating_trial_info=grating_trial_info,
        behavior_trial_info=behavior_trial_info,
        run_params=run_params,
        behavior_left_stim=behavior_left_stim,
        behavior_right_stim=behavior_right_stim,
        save_path=output_path if save_plots else None,
    )

    plt.show()

    return {
        'model': model,
        'scaler': scaler,
        'decoder_type': decoder,
        'shared_unit_ids': shared_unit_ids,
        'grating_cv_scores': grating_cv_scores,
        'behavior_accuracy': behavior_acc,
        'behavior_conf_matrix': behavior_conf,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'grating_proj': grating_proj,
        'behavior_proj': behavior_proj,
        'grating_labels': grating_labels,
        'behavior_labels': behavior_labels,
        'behavior_predictions': behavior_predictions,
    }


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import grating_config as cfg

    try:
        results = run_cross_decode(
            grating_pkl=cfg.GRATING_PKL,
            behavior_pkl=cfg.BEHAVIOR_PKL,
            behavior_left_stim=cfg.BEHAVIOR_LEFT_STIM,
            behavior_right_stim=cfg.BEHAVIOR_RIGHT_STIM,
            grating_time_window=cfg.GRATING_TIME_WINDOW,
            behavior_time_window=cfg.BEHAVIOR_TIME_WINDOW,
            decoder=cfg.DECODER,
            save_plots=True,
        )

        print("\n" + "=" * 60)
        print("Cross-decode complete!")
        print("=" * 60)
        print(f"  Decoder:               {results['decoder_type'].upper()}")
        print(f"  Shared units:          {len(results['shared_unit_ids'])}")
        print(f"  Grating CV accuracy:   {results['grating_cv_scores'].mean():.3f}")
        print(f"  Behavior accuracy:     {results['behavior_accuracy']:.3f}")
        print(f"  Above chance:          {results['behavior_accuracy'] - 0.5:+.3f}")

    except FileNotFoundError as e:
        print(f"Error: Data file not found — {e}")
        print("Update GRATING_PKL and BEHAVIOR_PKL paths.")
    except Exception as e:
        print(f"Error: {e}")
        raise
