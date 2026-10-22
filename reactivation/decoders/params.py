"""
Shared parameters for all decoding scripts in reactivation/decoders/.

Edit values here; the apply_/compare_/prepare_*.py scripts import from this
module so the same data paths, label schemes, and CV settings are used
everywhere.
"""

# ---------------------------------------------------------------- #
#  Data files                                                       #
# ---------------------------------------------------------------- #
SESSION_FOLDER = r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout\CnL42SG\CnL42SG_20260324"

task_pkl    = rf"{SESSION_FOLDER}\task_spikes_CnL42SG_20260324.pkl"
passive_pkl = rf"{SESSION_FOLDER}\passive_spikes_260324.pkl"

# Sleep blocks consumed by apply_merged_decoder_to_sleep.py.
# Each entry: (label, pkl_path, start_sec, end_sec) in the pkl's own
# spike_times_sec frame. end_sec=None ⇒ use window_duration_sec from the pkl.
sleep_blocks = [
    ("pre",  rf"{SESSION_FOLDER}\sleep_spikes_260324_sleep_pre.pkl",  None, None),
    ("post", rf"{SESSION_FOLDER}\sleep_spikes_260324_sleep_post.pkl", None, None),
]

# ``spike_times_sec`` in each extracted sleep pickle is already relative to
# the start of that block.  Consequently the local decoder range above is
# None/None (the full block); whole-day sample indices belong only in the
# SleepBlock metadata used by ``sleep_states``.

# Identity used to resolve completed NREM and channel-MUA ON/OFF artifacts
# through sleep/sleep_day_configs.json.  Set either to None to defer annotation
# until the artifacts are accessible (then rerun).
sleep_registry_animal = "CnL42"
sleep_registry_date = "260324"
sleep_on_min_shank_fraction = 0.5

# ---------------------------------------------------------------- #
#  CV / binning                                                     #
# ---------------------------------------------------------------- #
bin_sizes_ms = [ 50, 75, 100, 150, 200, 300, 400, 500]
n_splits     = 5
random_state = 42

# Default bin size used when prepare_*.py scripts are run as __main__ helpers.
prep_default_bin_ms = 50.0

# ---------------------------------------------------------------- #
#  Joint stim-identity label scheme                                 #
#  Used by every prepare_/compare_/apply_ script. Labels are        #
#  defined on the LEFT-side grating identity (orientation, SF)      #
#  jointly, so task and passive can share an identical label set.   #
#                                                                   #
#  A bin is labelled                                                #
#    +1  iff its trial's left grating matches ALL keys of class_pos #
#    -1  iff its trial's left grating matches ALL keys of class_neg #
#     0  iff the bin centre falls outside every stimulus epoch (ITI)#
#    dropped  for any other (orientation, SF) combination           #
# ---------------------------------------------------------------- #
class_pos = {"orientation": 45.0, "spatial_freq": 0.08}   # → +1
class_neg = {"orientation": 135.0, "spatial_freq": 0.08}   # → -1

# Translation from canonical keys/values above to the trial_params column
# names actually stored in the task vs passive pkls.
#
# Each entry is either
#   "column_name"                              — canonical value used as-is
#   ("column_name", {canonical_val: raw_val})   — canonical value is looked
#                                                  up to get the value this
#                                                  pkl actually stores
#
# This session's task grating identities don't share the passive session's
# raw column values even though they're the "same" class: task stores the
# negative-class orientation as 315° (not 135°), and both task and passive
# store spatial_freq as 0.16 (not the 0.08 used as the canonical key here).
# The value maps below let class_pos/class_neg stay defined once, while
# TASK_COL_MAP/PASSIVE_COL_MAP translate each canonical value to what
# CnL42SG_20260324's pkls actually store:
#   class_pos (45.0, 0.08)  → task (leftOrientation=45.0,  leftSpatialFreq=0.16)
#                           → passive (L_Orient=45.0,      L_SF=0.16)
#   class_neg (135.0, 0.08) → task (leftOrientation=315.0, leftSpatialFreq=0.16)
#                           → passive (L_Orient=135.0,     L_SF=0.16)
TASK_COL_MAP = {
    "orientation":  ("leftOrientation",  {45.0: 45.0, 135.0: 315.0}),
    "spatial_freq": ("leftSpatialFreq", {0.08: 0.16}),
}
PASSIVE_COL_MAP = {
    "orientation":  "L_Orient",
    "spatial_freq": ("L_SF", {0.08: 0.16}),
}

# ---------------------------------------------------------------- #
#  Sleep event detection                                            #
# ---------------------------------------------------------------- #
event_threshold        = 0.55
event_min_distance_sec = 0.50
top_n_events_per_class = 25
plot_window_sec        = 1.0

# ---------------------------------------------------------------- #
#  Velocity-matched compare                                         #
# ---------------------------------------------------------------- #
speed_bin_cms = 2.0
speed_top_pct = 99
hist_bin_ms   = 100
n_repeats     = 5
n_bootstrap   = 2000
