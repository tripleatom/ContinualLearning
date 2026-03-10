from pathlib import Path
from parse_grating_experiment import parse_grating_experiment
import numpy as np
from process_func.DIO import get_dio_folders, concatenate_din_data
import matplotlib.pyplot as plt


def fix_rising_edges(rising_times, n_trials, trial_duration_samples, tolerance=0.3):
    """
    Automatically detect and fix abnormal rising edges.

    Handles two classes of anomalies:
      - Glitch (extra edge): interval between consecutive risings is much shorter
        than expected (< (1-tolerance) * trial_duration). The edge that leaves the
        worse next-gap is removed.
      - Missing trial: interval is much longer than expected
        (> (1+tolerance) * trial_duration). Missing edges are interpolated at
        regular multiples of trial_duration_samples.

    Parameters
    ----------
    rising_times : np.ndarray
        Raw rising edge sample indices.
    n_trials : int
        Expected number of trials from the task file.
    trial_duration_samples : int
        Expected inter-trial interval in samples (stimulus + ITI).
    tolerance : float
        Fractional tolerance around the expected interval (default 0.3 = ±30%).

    Returns
    -------
    fixed : np.ndarray
        Corrected rising edge sample indices.
    log : list of str
        Description of each fix applied.
    """
    expected = trial_duration_samples
    min_interval = (1 - tolerance) * expected
    max_interval = (1 + tolerance) * expected

    fixed = list(rising_times.copy())
    log = []
    i = 0

    while i < len(fixed) - 1:
        interval = fixed[i + 1] - fixed[i]

        if interval < min_interval:
            # Glitch: two edges too close together.
            # Keep the one whose next gap is closer to expected.
            next_if_remove_i1 = fixed[i + 2] - fixed[i] if i + 2 < len(fixed) else np.inf
            next_if_remove_i0 = fixed[i + 2] - fixed[i + 1] if i + 2 < len(fixed) else np.inf
            if abs(next_if_remove_i1 - expected) <= abs(next_if_remove_i0 - expected):
                log.append(f"Removed glitch at index {i+1} (sample {fixed[i+1]}, "
                           f"interval {interval/expected:.2f}x expected)")
                fixed.pop(i + 1)
            else:
                log.append(f"Removed glitch at index {i} (sample {fixed[i]}, "
                           f"interval {interval/expected:.2f}x expected)")
                fixed.pop(i)
            # Re-check same position after removal

        elif interval > max_interval:
            # Missing trial(s): gap is ~N times the expected interval.
            n_missing = round(interval / expected) - 1
            for k in range(1, n_missing + 1):
                inserted = int(fixed[i] + k * expected)
                fixed.insert(i + k, inserted)
                log.append(f"Inserted missing edge at sample {inserted} "
                           f"(gap was {interval/expected:.2f}x expected)")
            i += n_missing + 1

        else:
            i += 1

    if len(fixed) != n_trials:
        log.append(f"WARNING: {len(fixed)} edges after fix, but {n_trials} trials expected")
    else:
        log.append(f"OK: {len(fixed)} edges match {n_trials} expected trials")

    return np.array(fixed), log


# ── Paths ──────────────────────────────────────────────────────────────────────
rec_folder = Path(r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260304/CnL42_20260304/CnL42SG_passive_20260304_142720.rec")
task_file_Path = Path(r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260304/CnL42_drifting_grating_exp_20260304_142748.txt")
task_id = task_file_Path.stem
folder_path = task_file_Path.parent

animal_id = rec_folder.name.split('.')[0].split('_')[0]
session_id = rec_folder.name.split('.')[0]
print(f"Processing {animal_id}/{session_id}")

# ── Task parameters ────────────────────────────────────────────────────────────
task_file = parse_grating_experiment(task_file_Path)
print(f"Animal: {task_file['metadata']['animal_id']}")
print(f"Total trials: {task_file['parameters']['total_trials']}")

df = task_file['trial_data']
stimulus_duration = float(task_file['parameters']['stimulus_duration'].rstrip('s'))
ITI_duration = float(task_file['parameters']['iti_duration'].rstrip('s'))
n_repeats = task_file['parameters']['total_trials']
trial_duration = stimulus_duration + ITI_duration

print(f"stimulus_duration={stimulus_duration}s  ITI={ITI_duration}s  "
      f"trial_duration={trial_duration}s  n_trials={n_repeats}")

# ── Raw DIO signal ─────────────────────────────────────────────────────────────
fs = 30000
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
pd_time, pd_state = concatenate_din_data(dio_folders, 3)
pd_time = pd_time - pd_time[0]

rising_times = pd_time[np.where(pd_state == 1)[0]]
falling_times = pd_time[np.where(pd_state == 0)[0]]

# Ensure pairs start with a rising edge: discard leading falling edges
if len(falling_times) > 0 and len(rising_times) > 0 and falling_times[0] < rising_times[0]:
    falling_times = falling_times[1:]
    print("Discarded leading falling edge (no matching rising edge)")

print(f"Raw edges: {len(rising_times)} rising, {len(falling_times)} falling (expected {n_repeats})")

# ── Auto-fix rising edges ──────────────────────────────────────────────────────
trial_duration_samples = int(trial_duration * fs)
rising_times_fixed, fix_log = fix_rising_edges(rising_times, n_repeats, trial_duration_samples)

print("\nFix log:")
for entry in fix_log:
    print(f"  {entry}")

# Falling times derived from fixed rising times (more reliable than raw falling edges)
falling_times_fixed = rising_times_fixed + int(stimulus_duration * fs)

# ── Diagnostics plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

raw_diff = np.diff(rising_times) / fs
fixed_diff = np.diff(rising_times_fixed) / fs

axes[0].plot(raw_diff, marker='o', ms=3)
axes[0].axhline(trial_duration, color='r', linestyle='--', label=f'expected ({trial_duration}s)')
axes[0].set_title('Inter-trial intervals — RAW')
axes[0].set_ylabel('Interval (s)')
axes[0].legend()

axes[1].plot(fixed_diff, marker='o', ms=3, color='green')
axes[1].axhline(trial_duration, color='r', linestyle='--', label=f'expected ({trial_duration}s)')
axes[1].set_title('Inter-trial intervals — FIXED')
axes[1].set_ylabel('Interval (s)')
axes[1].legend()

plt.tight_layout()
fig.savefig(rec_folder / f"{task_id}_DIO_fix.png", dpi=150)
plt.show()

# ── Save ───────────────────────────────────────────────────────────────────────
save_path = folder_path / f"{task_id}_DIO.npz"
np.savez_compressed(save_path, rising_times=rising_times_fixed, falling_times=falling_times_fixed)
print(f"Saved to {save_path}")
