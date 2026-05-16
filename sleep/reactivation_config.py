"""
reactivation_config.py  (v2 — whole-recording, combined-classifier)
=====================================================================
Central configuration for the sleep reactivation pipeline.

Key changes from v1
--------------------
• Classifier now trained on BOTH passive grating (45°/135°) AND behavior
  task (left-reward/right-reward) trials combined — same 2-class label:
      class 1  =  45° grating  +  reward-on-left
      class 0  =  135° grating +  reward-on-right
  This mirrors gratingBehaviorEmbedding.py and maximises training data.

• Scoring runs over the ENTIRE concatenated recording (no task_start/end
  restriction), with each 350 ms window annotated by epoch type:
      'passive'   passive grating viewing
      'task'      discrimination behaviour task
      'sleep1'    first sleep period
      'sleep2'    second sleep period
      'other'     everything else (transitions, ITIs, etc.)

Pipeline scripts (run in order)
---------------------------------
  build_reactivation_classifier.py
  compute_reactivation_score_fullrec.py
  detect_reactivation_events_fullrec.py
  reactivation_drift_analysis.py        (optional — needs pre/post pkls)

Data format assumption
-----------------------
Both PKL files come from the SAME concatenated recording sorted together,
so unit IDs (e.g. 'shank5_unit42') match directly without cross-session
matching.  The passive grating pkl uses the same unified schema as the
behavior pkl (produced by drift_data_org.py / readDIO_grating.py).
"""

from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  ← UPDATE THESE FOR YOUR MACHINE
# ─────────────────────────────────────────────────────────────────────────────

# Passive grating pkl  (drifting_grating_embedding_*.pkl from drift_data_org.py
# OR grating_data.pkl from the freely-moving grating pipeline)
GRATING_PKL = Path(
    "//Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/"
    "passive_embedding_analysis/"
    "CnL42SG_CnL42SG_passive_20260304_142720_grating_data.pkl"
)

# Behavior trial pkl  (behavior_trial_embedding_*.pkl from readDIO_grating.py)
BEHAVIOR_PKL = Path(
    "//Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/"
    "behavior_trial_embedding_20260309_2000.pkl"
)

# Phy/sorting_analyzer folder (parent of shank5/, shank0/, etc.)
# This is the sortout folder for the SAME recording as above.
SORTOUT_FOLDER = Path(
    r"/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304"
)

# Session label (used in output filenames)
SESSION = "CnL42_20260304"

# Output root for reactivation results
REACTIVATION_DIR = Path(
    r"/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/sleep_analysis/reactivation"
)

# ─────────────────────────────────────────────────────────────────────────────
# EPOCH SAMPLE BOUNDARIES  (in raw 30 kHz samples in the concatenated sorting)
#
# Fill in the sample-index start and end for each recording epoch.
# These are used to annotate every scoring window by its epoch type.
# Overlapping windows are assigned to the FIRST matching epoch in the list.
# ─────────────────────────────────────────────────────────────────────────────
FS_RAW = 30_000   # Hz

EPOCHS = {
    # name → (start_sample, end_sample)  in the concatenated recording
    "passive"  : (0, 149938305),   # ← UPDATE: passive grating viewing epoch
    "sleep1"   : (149938306, 259297964),   # ← UPDATE: first sleep period
    "task"     : (259297965, 323137986),   # ← UPDATE: discrimination behaviour task epoch
    "sleep2"   : (323137987, 481204357),   # ← UPDATE: second sleep period
    # Everything else is labelled "other" automatically
}

# Example (comment out the dict above and uncomment these):
# EPOCHS = {
#     "passive" : (  50_000_000, 130_000_000),
#     "task"    : (259_297_964, 323_137_986),
#     "sleep1"  : (330_000_000, 450_000_000),
#     "sleep2"  : (460_000_000, 580_000_000),
# }

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFIER = {
    # Orientations that define class 1 (left-equiv) in the grating data.
    # Trials with these orientations are labelled class 1; others class 0.
    "grating_target_ori"  : (45.0,),    # class 1 → left-equiv
    "grating_other_ori"   : (135.0,),   # class 0 → right-equiv

    # Firing-rate window (seconds, relative to stimulus / trial onset)
    # Applied to both grating and behavior trials.
    "fr_window"           : (0.05, 1.5),  # (start_sec, end_sec)

    # Optional: restrict grating trials to one spatial frequency (cpd).
    # Set to None to use all SFs.
    "spatial_freq_filter" : None,    # e.g. 0.16

    # Unit quality filter: include units with these Phy labels.
    "quality_keep"        : ["good", "mua"],

    # Minimum mean firing rate across all trials for a unit to be included.
    "min_fr_hz"           : 0.5,

    # scikit-learn LogisticRegression hyperparameters
    "C"                   : 1.0,
    "max_iter"            : 2000,
    "class_weight"        : "balanced",
    "solver"              : "lbfgs",

    # Cross-validation
    "cv_folds"            : 5,
    "random_state"        : 42,
}

# ─────────────────────────────────────────────────────────────────────────────
# WHOLE-RECORDING SCORING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SCORING = {
    # Sliding window (Nguyen et al. 2023: 350 ms)
    "window_ms"           : 350,
    "step_ms"             : 50,

    # Minimum number of spikes in a window to produce a valid score
    "min_spikes_in_win"   : 2,

    # Score only windows that do NOT overlap with known task/stimulus epochs
    # (i.e. exclude 'passive' and 'task' epochs from replay scoring).
    # Set to False to score all windows.
    "exclude_active_epochs" : True,

    # Epochs to exclude when exclude_active_epochs=True
    "active_epoch_names"  : ["passive", "task"],
}

# ─────────────────────────────────────────────────────────────────────────────
# EVENT DETECTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
DETECTION = {
    # Posterior probability threshold (Nguyen et al. 2023: 0.75)
    "react_thresh"        : 0.75,

    # Circular-shift shuffle
    "n_shuffles"          : 1000,
    "min_shift_sec"       : 30.0,   # minimum shift to avoid autocorrelation
    "random_state"        : 42,

    # Z-score threshold for labelling epochs as significantly reactivating
    "z_thresh"            : 2.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT ANALYSIS PARAMETERS (optional — Step 4)
# ─────────────────────────────────────────────────────────────────────────────
DRIFT = {
    # Task pkl recorded BEFORE the sleep periods
    "pre_sleep_task_pkl"  : Path(r"/path/to/pre_sleep_behavior_trial_embedding.pkl"),
    # Task pkl recorded AFTER the sleep periods
    "post_sleep_task_pkl" : Path(r"/path/to/post_sleep_behavior_trial_embedding.pkl"),
    "fr_window"           : (0.05, 1.5),
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_epochs():
    """Raise if any epoch has None boundaries."""
    bad = [name for name, (s, e) in EPOCHS.items() if s is None or e is None]
    if bad:
        raise ValueError(
            f"EPOCHS not fully set: {bad}.\n"
            "Update EPOCHS in reactivation_config.py with actual sample ranges."
        )
    print("✓ Epoch boundaries validated.")


def get_epoch_for_sample(sample: int) -> str:
    """Return the epoch label for a given sample index (first match)."""
    for name, (s, e) in EPOCHS.items():
        if s is not None and e is not None and s <= sample < e:
            return name
    return "other"


def get_epoch_array(starts: np.ndarray, win_samp: int) -> np.ndarray:
    """
    Return epoch label for each window centre.
    starts : (n_windows,) sample index of window start
    """
    centres = starts + win_samp // 2
    labels  = np.array(["other"] * len(centres), dtype=object)
    for name, (s, e) in EPOCHS.items():
        if s is None or e is None:
            continue
        mask = (centres >= s) & (centres < e)
        labels[mask] = name
    return labels


if __name__ == "__main__":
    print(f"Session          : {SESSION}")
    print(f"Grating pkl      : {GRATING_PKL}")
    print(f"Behavior pkl     : {BEHAVIOR_PKL}")
    print(f"Sortout folder   : {SORTOUT_FOLDER}")
    print(f"Reactivation dir : {REACTIVATION_DIR}")
    print(f"Epochs:")
    for name, (s, e) in EPOCHS.items():
        if s is not None and e is not None:
            dur = (e - s) / FS_RAW
            print(f"  {name:10s}: {s:12,} – {e:12,}  ({dur:.1f} s)")
        else:
            print(f"  {name:10s}: NOT SET")
    print(f"Window           : {SCORING['window_ms']} ms  step {SCORING['step_ms']} ms")
    print(f"React threshold  : P > {DETECTION['react_thresh']}")
    print(f"N shuffles       : {DETECTION['n_shuffles']}")
