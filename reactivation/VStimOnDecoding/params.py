"""
Shared parameters for all decoding scripts in reactivation/VStimOnDecoding/.

Edit values here; the apply_/compare_/prepare_*.py scripts import from this
module so the same data paths, label schemes, and CV settings are used
everywhere.
"""

# ---------------------------------------------------------------- #
#  Data files                                                       #
# ---------------------------------------------------------------- #
SESSION_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313"

task_pkl    = rf"{SESSION_FOLDER}\task_spikes_CnL42SG_20260313.pkl"
passive_pkl = rf"{SESSION_FOLDER}\passive_spikes_260313.pkl"

# Sleep blocks consumed by apply_merged_decoder_to_sleep.py.
# Each entry: (label, pkl_path, start_sec, end_sec) in the pkl's own
# spike_times_sec frame. end_sec=None ⇒ use window_duration_sec from the pkl.
sleep_blocks = [
    ("pre",  rf"{SESSION_FOLDER}\sleep_spikes_260313_sleep_pre.pkl",  0.0, None),
    ("post", rf"{SESSION_FOLDER}\sleep_spikes_260313_sleep_post.pkl", 0.0, None),
]

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
class_pos = {"orientation": 0.0, "spatial_freq": 0.04}   # → +1
class_neg = {"orientation": 0.0, "spatial_freq": 0.16}   # → -1

# Translation from canonical keys above to the trial_params column names
# actually stored in the task vs passive pkls.
TASK_COL_MAP    = {"orientation": "leftOrientation", "spatial_freq": "leftSpatialFreq"}
PASSIVE_COL_MAP = {"orientation": "L_Orient",        "spatial_freq": "L_SF"}

# ---------------------------------------------------------------- #
#  Sleep event detection                                            #
# ---------------------------------------------------------------- #
event_threshold        = 0.60
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
