# =============================================================================
# SHARED PARAMETERS — edit this file only
# =============================================================================
PKL_PATH    = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313\behavior_trial_embedding_20260419_2331.pkl"
OUTPUT_DIR  = None        # None → figures/ folder next to the pkl

PSTH_WINDOW = (-0.2, 1.5) # seconds relative to stimulus onset  (used by fig2a)
BIN_SIZE    = 0.025       # PSTH bin width in seconds            (used by fig2a)
RESP_WINDOW = (0.05, 0.7)  # firing rate measurement window       (used by fig2b, fig2d)
TOP_N_SELECTIVE = 50       # fig2d: keep top-N most SF-selective units
# =============================================================================
