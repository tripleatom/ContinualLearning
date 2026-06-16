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
event_threshold        = 0.55
event_min_distance_sec = 0.50
top_n_events_per_class = 25
plot_window_sec        = 1.0

# ---------------------------------------------------------------- #
#  Sleep UP/DOWN state detection + optional LFP validation           #
# ---------------------------------------------------------------- #
# UPState.py can already detect UP/DOWN states from binned sleep
# spike activity. Fill these NWB paths in later if you also want to
# extract raw/LFP slow-wave features from the original recording and
# validate the spike-derived state labels.
#
# Each entry should match sleep_blocks by label:
#     "pre":  r"path\to\pre_sleep_raw.nwb"
#     "post": r"path\to\post_sleep_raw.nwb"
sleep_nwb_paths = {
    "pre":  "",
    "post": "",
}

# Optional: leave empty to use all valid channels found in the NWB file.
# Otherwise set a list of V1 channel/electrode ids, e.g. [0, 1, 2, 3].
sleep_lfp_channels = {
    "pre":  [],
    "post": [],
}

# Raw/LFP extraction settings for future LFP-assisted validation.
raw_fs_hz = 30000.0
lfp_target_fs_hz = 1000.0
lfp_lowpass_hz = 300.0
slow_lfp_band_hz = (0.5, 4.0)

# Spike-derived population-MUA UP/DOWN detection settings.
updown_bin_ms = 10.0
updown_smooth_sigma_ms = 50.0
updown_down_z_threshold = -0.5
updown_up_z_threshold = 0.0
updown_down_percentile = 20.0
updown_min_state_ms = 50.0
updown_merge_gap_ms = 30.0

# ---------------------------------------------------------------- #
#  Velocity-matched compare                                         #
# ---------------------------------------------------------------- #
speed_bin_cms = 2.0
speed_top_pct = 99
hist_bin_ms   = 100
n_repeats     = 5
n_bootstrap   = 2000
