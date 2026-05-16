"""
sleep_config.py
===============
Central configuration for the UP/DOWN state analysis pipeline.
All other scripts import from here — change paths and timestamps once.

NWB file:  CnL42_20260304sh5.nwb
Shank:     5
Region:    V1
Channels:  32 (0–775 µm, 25 µm spacing)
Fs raw:    30 kHz
"""

from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (update NWB_PATH to the location on your machine)
# ─────────────────────────────────────────────────────────────────────────────
NWB_PATH    = r"/path/to/CnL42_20260304sh5.nwb"   # ← UPDATE THIS
SESSION     = "CnL42_20260304sh5"
SHANK       = 5
OUTPUT_DIR  = Path(NWB_PATH).parent / "sleep_analysis"   # all outputs land here

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP WINDOWS  (seconds from recording start)
# Two NREM-rich sleep epochs identified by the experimenter.
# Replace the placeholder values with your actual timestamps.
# ─────────────────────────────────────────────────────────────────────────────
SLEEP_WINDOWS = [
    {"name": "sleep1", "start": None, "end": None},   # ← UPDATE: e.g. 1200, 4800
    {"name": "sleep2", "start": None, "end": None},   # ← UPDATE: e.g. 6000, 9600
]

# ─────────────────────────────────────────────────────────────────────────────
# RECORDING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
FS_RAW      = 30_000          # Hz — raw acquisition rate
FS_LFP      = 1_250           # Hz — target LFP rate after downsampling
CONVERSION  = 1.95e-7         # int16 → volts (from NWB metadata)
N_CHANNELS  = 32

# Channel depth map (µm), ascending order matches channel index after depth sort
CHANNEL_DEPTHS = np.arange(0, N_CHANNELS * 25, 25)   # 0, 25, 50 … 775 µm

# Channels considered "deep layer" (L5/L6) for UP/DOWN detection
DEEP_LAYER_RANGE_UM = (300, 775)

# ─────────────────────────────────────────────────────────────────────────────
# LFP PREPROCESSING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LFP_PREPROC = {
    "car_reference"    : "global",
    "car_operator"     : "median",
    "initial_filter_min": 0.1,     # Hz
    "initial_filter_max": 600.0,   # Hz  (wide enough for broadband gamma)
    "target_fs"        : FS_LFP,
    "lfp_filter_min"   : 0.1,
    "lfp_filter_max"   : 600.0,
    "dtype"            : "float32",
}

# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY BANDS
# ─────────────────────────────────────────────────────────────────────────────
BANDS = {
    "delta"       : (0.5,  4.0),
    "theta"       : (5.0, 10.0),
    "sigma"       : (9.0, 25.0),
    "gamma_low"   : (30.0, 100.0),
    "gamma_broad" : (30.0, 200.0),   # added for UP/DOWN detection
    "theta_ratio" : {                 # numerator / denominator bands
        "num": (5.0, 10.0),
        "den": (2.0, 15.0),
    },
    # EMG proxy: high-frequency LFP coherence (200–600 Hz)
    "emg_proxy"   : (200.0, 600.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP STATE SCORING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SCORING = {
    "epoch_sec"         : 4.0,        # scoring epoch length (seconds)
    "smooth_sec"        : 10.0,       # band-power smoothing window (seconds)
    "min_bout_sec"      : 30.0,       # minimum state-bout length (seconds)
    # Thresholds (set as multiples of SD above/below mean during recording)
    # These are initial values; visually inspect and adjust per session.
    "nrem_pc1_thresh_sd": 0.0,        # PC1 > mean + N*SD  → NREM candidate
    "rem_theta_thresh_sd": 0.5,       # theta ratio > mean + N*SD → REM candidate
    "wake_emg_thresh_sd": 0.5,        # EMG proxy > mean + N*SD  → Awake
}

# ─────────────────────────────────────────────────────────────────────────────
# UP / DOWN STATE DETECTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
UPDOWN = {
    # Delta envelope (DOWN state proxy)
    "delta_smooth_ms"      : 100,     # ms — smoothing for delta envelope
    "down_thresh_sd"       : 0.5,     # delta env > mean + N*SD → DOWN candidate peak
    "down_min_dur_ms"      : 50,      # ms — minimum DOWN state duration
    "down_max_dur_ms"      : 400,     # ms — maximum DOWN state duration

    # Gamma envelope (UP state proxy)
    "gamma_smooth_ms"      : 25,      # ms — Gaussian sigma for gamma envelope
    "gamma_down_percentile": 50,      # gamma must be BELOW this percentile for DOWN

    # MUA cross-validation
    "mua_bin_ms"           : 20,      # ms — MUA bin size
    "mua_smooth_sigma"     : 2.5,     # bins — Gaussian sigma for MUA smoothing
    "mua_down_max_frac"    : 0.20,    # DOWN state MUA must be < this fraction of NREM mean

    # UP state bounds
    "up_min_dur_ms"        : 100,     # ms
    "up_max_dur_ms"        : 2000,    # ms

    # Validation window around each DOWN state for MUA check
    "mua_check_window_ms"  : 200,     # ± this many ms around DOWN state center
}

# ─────────────────────────────────────────────────────────────────────────────
# SPECTROGRAM PARAMETERS  (matching existing calculate_spectrogram.py)
# ─────────────────────────────────────────────────────────────────────────────
SPEC_PARAMS = {
    "nperseg"  : 1024,    # ~0.82 s window at 1250 Hz
    "noverlap" : 768,     # 75% overlap → ~0.2 s time resolution
    "nfft"     : 2048,    # ~0.6 Hz frequency resolution
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — create output sub-directory per sleep window
# ─────────────────────────────────────────────────────────────────────────────
def get_window_dir(window_name: str) -> Path:
    d = OUTPUT_DIR / window_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def validate_windows():
    """Call this at the start of any script to catch un-set timestamps early."""
    for w in SLEEP_WINDOWS:
        if w["start"] is None or w["end"] is None:
            raise ValueError(
                f"Sleep window '{w['name']}' has no timestamps set. "
                f"Please edit SLEEP_WINDOWS in sleep_config.py."
            )
        if w["end"] <= w["start"]:
            raise ValueError(
                f"Sleep window '{w['name']}': end ({w['end']}) must be > start ({w['start']})."
            )
    print("✓ Sleep window timestamps validated.")


if __name__ == "__main__":
    print(f"Session   : {SESSION}")
    print(f"NWB path  : {NWB_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Fs raw    : {FS_RAW} Hz  |  Fs LFP: {FS_LFP} Hz")
    print(f"Channels  : {N_CHANNELS}  (0–{CHANNEL_DEPTHS[-1]} µm)")
    print(f"Sleep windows:")
    for w in SLEEP_WINDOWS:
        dur = None if (w['start'] is None or w['end'] is None) else f"{w['end']-w['start']:.0f} s"
        print(f"  {w['name']}: {w['start']} – {w['end']}  ({dur})")
