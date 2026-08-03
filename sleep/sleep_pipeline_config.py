"""Shared configuration for the sleep LFP / spectrogram pipeline.

Single source of truth imported by:
  - extract_sleep_lfp.py          (NWB -> low_freq/*_lfp_traces.npz)
  - compute_sleep_spectrograms.py (LFP traces -> *_spectrograms.npz)
  - compute_sleep_features.py     (LFP + spectrograms -> feature pkl)
  - plot_sleep_spectrograms.py    (feature pkl -> per-channel figures)

Edit values here; the scripts pull from this module so they stay in sync.

`sleep_sessions` (pre-task / post-task) lets every stage - extract_sleep_lfp.py,
compute_sleep_spectrograms.py, compute_sleep_features.py, video_ephys_sync.py
and plot_sleep_spectrograms.py - process both sleep windows in one run, or skip
whichever one has start_sample=end_sample=None. Outputs are suffixed "_pre" /
"_post" throughout so the two never collide.

Per-day paths (rec_folder, nwb_session_name, and each session's
start/end sample, .rec epoch folder, video _PROC file) come from
sleep_day_configs.json via sleep_day_registry.py, keyed by ACTIVE_ANIMAL +
ACTIVE_DATE below. To switch session: edit those two. If that animal-day
hasn't been registered yet, run `python set_sleep_day.py` first (it prompts
once and saves the answer) - this module itself never prompts, since
extract_sleep_lfp.py re-imports it in spawned worker processes on Windows.

`python sleep_pipeline_gui.py` drives the whole sequence and writes
ACTIVE_ANIMAL / ACTIVE_DATE / SESSION_FILTER / shanks here before each run, so
running the stage scripts by hand afterwards uses exactly the same settings.
"""
from pathlib import Path

from server_fallback import (
    mirror_on_backup_server,
    resolve_output_folder,
    resolve_existing_file,
)
from sleep_day_registry import entry_animal, load_day_config

# =====================================================
# SESSION / PATHS  (shared)
# =====================================================
# Which registered animal-day to run. Edit these per session; see module
# docstring. The registry is keyed by BOTH, so the same date can exist for
# several animals without them overwriting each other.
ACTIVE_ANIMAL = "CnL46"
ACTIVE_DATE = "260727"

# Optional per-run restriction on which sleep session(s) the stages process:
#   None (or "both") -> every active session for ACTIVE_DATE
#   "pre" / "post"   -> only that one, even if the other is registered
# Applied inside active_sleep_sessions(), so it reaches every stage at once
# (extract / spectrograms / features / plots / sync). Kept separate from the
# registry: the registry says what EXISTS for a day, this says what to run now.
SESSION_FILTER = None

_day_cfg = load_day_config(ACTIVE_DATE, ACTIVE_ANIMAL)

# Animal actually registered for this day (identical to ACTIVE_ANIMAL, except
# for pre-animal registry entries, where it is read off the rec_folder path).
animal = entry_animal(_day_cfg)

# Recording folder. low_freq outputs are written to / read from
# rec_folder / "low_freq".
rec_folder = _day_cfg['rec_folder']

# Used for OUTPUT filenames: {session_name}_sh{ish}_lfp_traces.npz, etc.
session_name = Path(rec_folder).stem.split('.')[0]

# NWB files are usually a different (6-digit date) naming, e.g. "CnL42SG_260324sh0.nwb",
# but this session's NWBs were exported with the full 8-digit date instead.
# Base name used for the input .nwb files (everything before "sh{ish}.nwb").
nwb_session_name = _day_cfg['nwb_session_name']

# Shanks to process (shared by both scripts).
shanks = [0, 1, 2, 3, 7]

# Folder holding the per-shank LFP / spectrogram .npz files.
low_freq_folder = Path(rec_folder) / "low_freq"

# Original (pre-downsample) acquisition sampling rate, Hz. Used by plotting to
# convert SpikeGadgets sync sample indices to seconds.
original_fs = 30000


# =====================================================
# SLEEP SESSIONS: pre-task / post-task  (extract_sleep_lfp.py,
# compute_sleep_spectrograms.py, compute_sleep_features.py)
# =====================================================
# Most days have a pre-task AND a post-task sleep period within the SAME
# continuous NWB recording. Each entry below gives the window to analyze, in
# ORIGINAL-FS sample indices, plus an output-filename suffix so pre/post
# results don't collide (e.g. "{session_name}_post_sh5_lfp_traces.npz").
#   - Leave one bound as None to mean "use the recording start/end" for that
#     side (same as before).
#   - Leave BOTH start_sample and end_sample as None to SKIP that session
#     entirely (e.g. a day with no pre-task sleep recorded).
#
# 'proc_file' is the matching front-camera *_PROC tracking file for that
# sleep session (used by plot_sleep_spectrograms.py to load/derive the
# correct velocity_advanced.pkl). 'rec_file_folder' is that session's own
# .rec epoch folder (e.g. CnL42_presleep_..._.rec vs CnL42_postsleep_..._.rec)
# - the DIO source for computing that session's own sync_times pkl via
# video_ephys_sync.py. Every day now records a SEPARATE video for pre-task
# vs post-task sleep, so both fields must be set per session - do NOT point
# both sessions at the same PROC file or .rec folder. Either may be None if
# that session has no video (velocity overlay / sync are then skipped for
# it). All of the above is sourced from sleep_day_configs.json - see
# set_sleep_day.py to add/edit a day.
sleep_sessions = {
    'pre': {
        'start_sample': _day_cfg['pre']['start_sample'],
        'end_sample': _day_cfg['pre']['end_sample'],
        'suffix': '_pre',
        'proc_file': _day_cfg['pre']['proc_file'],
        'rec_file_folder': _day_cfg['pre']['rec_file_folder'],
    },
    'post': {
        'start_sample': _day_cfg['post']['start_sample'],
        'end_sample': _day_cfg['post']['end_sample'],
        'suffix': '_post',
        'proc_file': _day_cfg['post']['proc_file'],
        'rec_file_folder': _day_cfg['post']['rec_file_folder'],
    },
}

# Folder holding this day's camera recordings + tracking outputs
# (velocity_advanced.pkl files are written here by proc_func_velocity.py).
video_folder = Path(rec_folder).parent / "video"


# =====================================================
# VELOCITY  (proc_func_velocity.py -> plot_sleep_spectrograms.py)
# =====================================================
# Which tracked point the speed trace is differentiated from:
#   'proc_center' -> center_x / center_y stored in the *_PROC file. That is a
#                    likelihood-weighted mean of the SIX HEAD keypoints, and on
#                    frames where tracking fails the acquisition program repeats
#                    the previous frame's position, so dropouts read as exactly
#                    zero speed (~19% of frames on CnL42 260320 pre).
#   'dlc_body'    -> centroid of VELOCITY_KEYPOINTS, read from the *_DLC.hdf5
#                    companion file. Low-confidence frames become gaps that are
#                    interpolated instead of frozen.
# Each source writes its own file (..._velocity_advanced.pkl vs
# ..._velocity_body.pkl), so switching never overwrites the other one, and
# plot_sleep_spectrograms.py loads whichever is selected here.
VELOCITY_SOURCE = "dlc_body"

# Keypoints averaged for 'dlc_body'. The trunk points stay visible while the
# animal is curled up asleep, unlike nose/eyes/bars.
VELOCITY_KEYPOINTS = ("left_midside", "right_midside",
                      "left_hip", "right_hip", "tail_base")

# A keypoint only enters the centroid on frames where DLC is at least this
# confident about it.
VELOCITY_LIKELIHOOD_THRESHOLD = 0.6


# =====================================================
# POPULATION FIRING RATE  (sleep_population_rate.py -> plot_sleep_spectrograms.py)
# =====================================================
# Adds a population-rate trace panel under the spectrogram, built from the
# shank's own spike sorting so the rate shown belongs to the same electrodes
# as the spectrogram above it. Sorting runs on the whole day, so spike samples
# share the coordinate system of each session's start_sample/end_sample.
# Skipped silently for a shank/day with no sorting.
SORTOUT_ROOTS = (
    r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout",
    r"\\10.129.151.108\xieluanlabs\xl_cl\sortout",
)

population_rate_params = {
    'enabled': True,
    # Bin width (s). The panel spans the whole session (thousands of seconds),
    # so 1 s bins keep it readable; drop to ~0.1 for a shorter window.
    'bin_size_sec': 1.0,
    # Gaussian smoothing, in bins. 0 disables.
    'smooth_sigma_bins': 2.0,
}


def active_sleep_sessions(sessions=None):
    """Return the sleep_sessions entries that aren't (start=None, end=None).

    A session with both bounds set to None is treated as "not recorded /
    don't analyze" for that day and is skipped by every calculation script.
    SESSION_FILTER narrows the result further to a single session when set.
    """
    sessions = sleep_sessions if sessions is None else sessions
    active = {
        name: cfg for name, cfg in sessions.items()
        if not (cfg['start_sample'] is None and cfg['end_sample'] is None)
    }
    if SESSION_FILTER in (None, 'both'):
        return active
    if SESSION_FILTER not in sessions:
        raise ValueError(
            f"SESSION_FILTER={SESSION_FILTER!r} is not a sleep session; "
            f"use None/'both' or one of {sorted(sessions)}."
        )
    return {name: cfg for name, cfg in active.items() if name == SESSION_FILTER}


# =====================================================
# LFP PREPROCESSING  (extract_sleep_lfp.py)
# =====================================================
preproc_params = {
    'reference': 'global',
    'operator': 'median',
    'target_fs': 500,      # Downsample target FS (Hz)
    'lfp_min': 1,          # LFP band low edge (Hz; safer than 0.1 Hz)
    'lfp_max': 200,        # LFP band high edge (Hz)
}

# Downsampling method:
#   "resample"  -> FFT-based (scipy), includes anti-alias. Robust, slower.
#   "decimate"  -> integer-factor decimation. Cheaper, but requires
#                  orig_fs / target_fs to be (near) integer. Anti-aliasing is
#                  provided by bandpass filtering BEFORE decimation, which is
#                  valid because lfp_max (200 Hz) < new Nyquist (target_fs/2).
DOWNSAMPLE_METHOD = "decimate"   # "decimate" or "resample"

# Parallelization (the heavy step is reading 30 kHz + filtering).
# n_jobs=-1 uses all cores; lower it if the network share is the bottleneck.
# chunk_duration controls the work unit handed to each worker.
N_JOBS = -1
CHUNK_DURATION = "30s"


# =====================================================
# SPECTROGRAM  (compute_sleep_spectrograms.py)
# =====================================================
# Computed on the 500 Hz LFP. These give:
#   • ~2 s window  • ~0.24 Hz freq resolution  • ~0.5 s time steps
spec_params = {
    "nperseg": 1024,       # 1024 samples / 500 Hz ≈ 2.048 s
    "noverlap": 768,       # 75% overlap = 1.5 s overlap
    "nfft": 2048,          # gives ~0.24 Hz freq resolution
    "scaling": "density",
    "mode": "psd",
}


# =====================================================
# SLEEP FEATURES  (compute_sleep_features.py)
# =====================================================
band_params = {
    'bands': {
        'delta': (0.5, 4),
        'theta': (5, 10),
        'sigma': (9, 25),
        'gamma': (40, 100),
        # (num_low, num_high, den_low, den_high)
        'theta_ratio': (5, 10, 2, 15),
    },
    'smoothing_window': 10,  # seconds for band power smoothing
}


# =====================================================
# PLOTTING  (plot_sleep_spectrograms.py)
# =====================================================
plot_params = {
    # Color scale options: 'adaptive', 'percentile', 'manual'
    # NOTE: the spectrogram is now a PER-FREQUENCY robust z-score (MAD units),
    # which has extreme negative outliers (near-zero power bins -> ~-120 dB,
    # and low-MAD frequency rows). 'percentile' with vmin_percentile=0 picks up
    # those outliers, blowing up the lower bound so the real signal collapses
    # into the red end of the colormap. Use a symmetric 'manual' scale instead.
    # 'adaptive' = median ± N*MAD of the cleaned z-scores. Median ~0 -> the
    # scale is symmetric about zero, so z=0 sits at the colormap center (green
    # in jet); blue = below-baseline, red = above-baseline power. Adapts per
    # channel and saturates, without the warm bias of the asymmetric percentile.
    'color_scale_method': 'adaptive',

    # For 'adaptive' method (median ± N * MAD)
    'adaptive_n_mad': 3,

    # For 'percentile' method. Computed per channel on the cleaned (finite,
    # non-artifact) z-score bins, so each channel uses its full color range.
    'vmin_percentile': 2,
    'vmax_percentile': 98,

    # For 'manual' method (data already in MAD units -> symmetric ±N is natural)
    'vmin_manual': -3,
    'vmax_manual': 3,

    # Color scale extension (as fraction of range). 0 = the percentile bounds
    # ARE the color limits, so ~2% of pixels saturate at each end (crisp
    # contrast). Raise toward 0.2 to pad the scale and reduce saturation.
    'vmin_extension': 0.0,
    'vmax_extension': 0.0,

    # Frequency display range for spectrogram
    'freq_min': 0.5,
    'freq_max': 100,

    # Which trace panels to draw below the spectrogram, top-to-bottom. Options:
    #   'pc1', 'theta_ratio', 'delta', 'sigma', 'gamma'
    # (velocity is appended separately when available.)
    'trace_panels': ['delta', '4_25', 'gamma'],

    # Y-axis limits for normalized band power plots (in standard deviations)
    'band_ylim': (-4, 4),

    # Colormap
    'cmap': 'jet',

    # Figure size (much wider for full recording)
    'figsize': (30, 12),

    # DPI
    'dpi': 150,  # Lower DPI for large figures
}


# =====================================================
# BROADBAND ARTIFACT MASKING  (plot_sleep_spectrograms.py, Option B)
# =====================================================
# Motion / EMG / cable artifacts lift power across ALL frequencies at once
# (unlike real brain states, which have a spectral tilt). We flag spectrogram
# time bins whose mean-over-frequency power is a robust outlier, dilate the
# mask to cover the band-power smoothing smear, then shade those spans and
# exclude them from the per-panel z-normalization.
artifact_params = {
    'enabled': True,
    # How artifact periods are handled in the figure:
    #   'blank'       -> keep the real timeline, draw artifact bins as white gaps
    #   'concatenate' -> excise artifact periods and stitch the clean parts into
    #                    a continuous "artifact-free time" axis (seams marked).
    'remove_mode': 'concatenate',
    # Robust z-score (median / MAD) threshold on broadband power. Lower =
    # more aggressive. ~5 is a good start; drop toward 4 to catch more.
    'n_mad': 5.0,
    # Only average frequencies up to this value when measuring broadband
    # power (None = all freqs up to Nyquist). Keeping it at the display max
    # focuses on the bands you actually plot.
    'fmax': 100,
    # Dilate the mask by this many seconds on EACH side. A brief artifact
    # bleeds into the smoothed band traces and the per-frequency z-score
    # baseline, so we pad generously (10 s each side) to remove its influence.
    'dilate_sec': 10.0,
    # Optional velocity gate: also mark bins where synced velocity exceeds
    # this (cm/s). Set to None to disable. Union'd with the LFP detector.
    'velocity_threshold': None,
    # Appearance of the shaded artifact spans.
    'shade_color': 'lightgray',
    'shade_alpha': 0.55,
}


# =====================================================
# NREM SCORING AND CONSOLIDATED-WINDOW SELECTION  (score_nrem_epochs.py)
# =====================================================
# Finds the period(s) where the animal is "fully asleep" (consolidated NREM),
# from the 500 Hz band-powers pkl. NREM = sustained high slow-wave activity
# (PC1 of the spectrogram, oriented by delta) with low movement (broadband
# power proxy). Output feeds the UP/DOWN stage.
sleep_detect_params = {
    # Which shanks to aggregate for the sleep index (None = use `shanks`).
    'shanks': None,
    # Epoch length (s) for scoring; signals are averaged within each epoch.
    'epoch_sec': 4.0,
    # NREM is declared where the spectral-tilt index  z(log delta) - z(log gamma)
    # exceeds this (SD units; higher = more selective). The tilt index already
    # suppresses movement, so the gamma gate below is optional.
    'nrem_sw_z_thresh': 1.0,
    # Optional extra gate: reject epochs whose gamma (z) exceeds this. Set to
    # None to rely on the tilt index alone (recommended starting point).
    'move_z_thresh': None,
    # Smooth the per-epoch scores with this many epochs (boxcar) before
    # thresholding, to avoid flicker.
    'smooth_epochs': 3,
    # Bridge NREM gaps shorter than this (s) and drop NREM bouts shorter than
    # `min_bout_sec`.
    'merge_gap_sec': 20.0,
    'min_bout_sec': 60.0,
    # A bout must be at least this long (s) to count as "consolidated / fully
    # asleep" (the window handed to the UP/DOWN stage).
    'consolidated_sec': 120.0,
    # Reuse the artifact detector to also exclude saturated bins from scoring.
    'use_artifact_mask': True,
}


# =====================================================
# CORTICAL UP / DOWN STATE SCORING  (score_cortical_up_down_states.py)
# =====================================================
# Runs inside a consolidated-NREM window found by Stage 2. Re-extracts that
# window from the NWB at a higher LFP rate (so broadband gamma reaches ~200 Hz),
# builds a population-activity proxy, and segments DOWN (population silence) vs
# UP states. Hybrid: the proxy is LFP broadband-gamma now, but the detector is
# written so sorted-MUA spike rate can be dropped in unchanged (`mua_pkl`).
up_down_params = {
    # Which shank to analyze, and which NREM window: 'longest' uses Stage 2's
    # fully_asleep_window_s; or give an explicit (start_s, end_s) in lfp_time.
    'shank': 5,
    'window': 'longest',
    # Higher-rate LFP just for this window. 30000/1250 = 24 (integer decimate).
    'target_fs': 1250,
    'extract_filter': (0.1, 300.0),     # bandpass before decimation (anti-alias)
    # Deep-layer channels (L5/L6) carry the clearest UP/DOWN; range in microns.
    'deep_layer_um': (300, 775),
    # Population-activity proxy = power in this band (MUA-like at 1250 Hz).
    'gamma_broad': (30.0, 200.0),
    'env_smooth_ms': 25,                # envelope smoothing
    # DOWN = proxy below this percentile (within the NREM window) for a run of
    # [min_down_ms, max_down_ms]; runs closer than merge_gap_ms are merged.
    'down_percentile': 40,
    'min_down_ms': 50,
    'max_down_ms': 600,
    'merge_gap_ms': 30,
    'min_up_ms': 50,
    'delta_band': (0.5, 4.0),           # slow-wave trace for the figure
    # Figure zoom: a representative slice of the window (seconds).
    'zoom_sec': 12,
    'zoom_offset_sec': 30,
    # Optional: path to a MUA pkl (from compute_sleep_mua.py) to use spike rate instead of the
    # LFP proxy. None = LFP-only (hybrid stage 1).
    'mua_pkl': None,
}
