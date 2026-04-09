"""
Shared configuration for grating-behavior analysis scripts
===========================================================

Edit this file to change paths, time windows, and stimulus mappings.
All three analysis scripts read from here:
  - gratingDecodeBehavior.py        (cross-session decoder)
  - gratingBehaviorEmbedding.py     (PCA/LDA mixed embedding)
  - gratingAllOrientationsEmbedding.py  (all-orientations LDA embedding)
"""

# =============================================================================
# FILE PATHS  ← edit these for each session
# =============================================================================

GRATING_PKL = (
    "/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260320/passive_embedding_analysis/CnL42_CnL42_passive_20260320_130227_grating_data_merged.pkl"
)

BEHAVIOR_PKL = (
    "/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260320/behavior_trial_embedding_20260323_0039.pkl"
)

# =============================================================================
# TIME WINDOWS  (seconds relative to stimulus onset)
# =============================================================================

GRATING_TIME_WINDOW  = (0.05, 1.5)   # (start, end)
BEHAVIOR_TIME_WINDOW = (0.05, 1.5)   # (start, end); set end=None to use trial duration

# PSTH heatmap — independent window and bin size
# Starts before stimulus onset to show pre-stimulus baseline
PSTH_TIME_WINDOW = (-0.2, 1.5)   # (start, end) in seconds; negative = pre-stimulus
PSTH_BIN_SIZE    = 0.02           # bin width in seconds (50 ms)

# =============================================================================
# GRATING STIMULUS FILTER
# Used by: gratingAllOrientationsEmbedding
#
# GRATING_FILTER – dict or None
#   Keys 'ori' and 'sf' are each optional; omit a key to keep all values.
#   Set to None to use every (ori, SF) combination.
#   Example: {'ori': [0.0], 'sf': [0.04, 0.08, 0.16, 0.32]}
# =============================================================================

GRATING_FILTER = {
    'ori': [0.0],  # Keep all orientations
    'sf':  [0.04, 0.08, 0.16, 0.32]  # Keep only these spatial frequencies,
}

# =============================================================================
# BEHAVIOR ↔ GRATING MAPPING
# Used by: gratingAllOrientationsEmbedding
#
# Defines which grating (ori, SF) condition each behavior choice maps to.
#   behavior-left  trials → BEHAVIOR_LEFT_STIM  grating class
#   behavior-right trials → BEHAVIOR_RIGHT_STIM grating class
# =============================================================================

BEHAVIOR_LEFT_STIM  = {'ori': 0.0, 'sf': 0.04}
BEHAVIOR_RIGHT_STIM = {'ori': 0.0, 'sf': 0.16}

# =============================================================================
# DECODER TYPE
# Used by: gratingDecodeBehavior
#
# Options: 'lda'  |  'svm_linear'  |  'svm_rbf'
# =============================================================================

DECODER = 'lda'
