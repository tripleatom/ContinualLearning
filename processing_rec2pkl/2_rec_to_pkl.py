# ================================================================
# Converts ephys .rec + PsychoPy CSV (+ optional taskmeta.npz)
# into a unified pickle (.pkl) for downstream analysis.
#
# Required:
#   • <session>.rec folder (TDT/OpenEphys DIO source)
#   • PsychoPy CSV with stim_on_s / stim_off_s
#
# Optional:
#   • <csv_stem>_DIO_cleaned.npz (from DIO verification script)
#   • <csv_stem>_taskmeta.npz (from static or drifting generator)
#
# Output:
#   • "<animal_id>_<session_id>_grating_data.pkl"
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# =========================
# 1) User-defined paths
# =========================
rec_folder       = Path(r"L:\xl_cl\Albert_passive_grating\CnL42SG_20251022_160759.rec")
csv_path         = Path(r"L:\xl_cl\Albert\20251022_logs\CnL42_0_90_two_grating_passive_static_20251022_160842.csv")
dio_npz_path     = Path(r"L:\xl_cl\Albert\20251022_dio\CnL42_0_90_two_grating_passive_static_20251022_160842_DIO_cleaned.npz")
taskmeta_npz_path= None   # optional: set to Path(...) if known
output_dir       = Path(r"L:\xl_cl\Albert\20251022_pkl")

# ================================================================
# 2) Load and combine data
# ================================================================

rec_name   = rec_folder.name.replace(".rec", "")
parts      = rec_name.split("_")
animal_id  = parts[0]
session_id = rec_name

print(f"\n[Info] Processing session: {animal_id}/{session_id}")

# -----------------------
# Load CSV timing
# -----------------------
df = pd.read_csv(csv_path)
if not {"stim_on_s", "stim_off_s"}.issubset(df.columns):
    raise ValueError("CSV missing required columns 'stim_on_s' and 'stim_off_s'")

csv_on  = df["stim_on_s"].to_numpy(float)
csv_off = df["stim_off_s"].to_numpy(float)

csv_off_prev = np.r_[np.nan, csv_off[:-1]]
csv_iti = csv_on - csv_off_prev
mean_iti = np.nanmean(csv_iti)
print(f"[Info] Loaded {len(csv_on)} CSV trials | mean ITI = {mean_iti:.3f}s")

# -----------------------
# Load DIO-cleaned edges
# -----------------------
if dio_npz_path is None:
    dio_guess = csv_path.with_name(f"{csv_path.stem}_DIO_cleaned.npz")
else:
    dio_guess = Path(dio_npz_path)

if dio_guess.exists():
    dio = np.load(dio_guess, allow_pickle=True)
    rising_times  = dio["rising_times"]
    falling_times = dio["falling_times"]
    fs = dio["fs"] if "fs" in dio.files else 30000
    print(f"[Info] Loaded DIO edges: {len(rising_times)} rising | fs={fs} Hz")
else:
    print("[Warning] DIO NPZ not found — proceeding without it.")
    rising_times = falling_times = np.array([])
    fs = None

# -----------------------
# Load optional task metadata NPZ
# -----------------------
if taskmeta_npz_path is None:
    taskmeta_guess = csv_path.with_name(f"{csv_path.stem}_taskmeta.npz")
else:
    taskmeta_guess = Path(taskmeta_npz_path)

meta = {}
if taskmeta_guess.exists():
    task_npz = np.load(taskmeta_guess, allow_pickle=True)
    print(f"[Info] Loaded task metadata: {taskmeta_guess.name}")
    meta = {k: task_npz[k].item() if task_npz[k].ndim == 0 else task_npz[k] for k in task_npz.files}
else:
    print("[Info] No taskmeta NPZ found — using defaults.")

stimulus_duration = float(meta.get("stimulus_duration_s", 1.0))
iti_duration      = float(meta.get("iti_duration_s", 0.5))
n_trials          = int(meta.get("n_trials", len(csv_on)))
orientation_pairs = meta.get("orientation_pairs", np.array([[0, 90]]))
left_seq  = meta.get("left_ori_seq", np.zeros(len(csv_on)))
right_seq = meta.get("right_ori_seq", np.zeros(len(csv_on)))

# -----------------------
# Build trial windows
# -----------------------
if len(rising_times) and len(falling_times):
    n = min(len(csv_on), len(rising_times), len(falling_times))
    trial_windows = [(float(rising_times[i]), float(falling_times[i])) for i in range(n)]
else:
    trial_windows = [(float(csv_on[i]), float(csv_off[i])) for i in range(len(csv_on))]

# -----------------------
# Combine into unified dict
# -----------------------
data = {
    "metadata": {
        "animal_id": animal_id,
        "session_id": session_id,
        "recording_folder": str(rec_folder),
        "csv_path": str(csv_path),
        "dio_npz_path": str(dio_guess) if dio_guess.exists() else None,
        "taskmeta_npz_path": str(taskmeta_guess) if taskmeta_guess.exists() else None,
        "extraction_date": datetime.now().isoformat(),
        "fs": fs,
        "n_trials": len(trial_windows),
        "source": "CSV+DIO",
    },
    "experiment_parameters": {
        "stimulus_duration_s": stimulus_duration,
        "iti_duration_s": iti_duration,
        "orientation_pairs": orientation_pairs.tolist() if isinstance(orientation_pairs, np.ndarray) else orientation_pairs,
    },
    "trial_info": {
        "trial_windows": trial_windows,
        "csv_iti": csv_iti.tolist(),
        "left_ori_seq": left_seq.tolist(),
        "right_ori_seq": right_seq.tolist(),
        "csv_df": df.to_dict("records"),
    },
    "dio_edges": {
        "rising_times": rising_times.tolist(),
        "falling_times": falling_times.tolist(),
    },
    "spike_data": {},  # Placeholder for later neural data
}

# -----------------------
# Save to pickle
# -----------------------
output_dir.mkdir(parents=True, exist_ok=True)
pkl_path = output_dir / f"{animal_id}_{session_id}_grating_data.pkl"

with open(pkl_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[Success] Saved combined data to {pkl_path}")
