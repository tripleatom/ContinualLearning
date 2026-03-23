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


def segment_rising_edges_by_task(rising_times, task_n_trials_list, task_trial_durations_samples,
                                  gap_factor=3.0):
    """
    Split a continuous array of rising edges into per-task segments.

    Between tasks there is typically a large gap (experimenter switches tasks).
    This function first finds candidate split points (intervals >> expected trial
    duration), then assigns each segment to the corresponding task file in order.

    Parameters
    ----------
    rising_times : np.ndarray
        All rising edge sample indices from the full recording.
    task_n_trials_list : list of int
        Expected number of trials for each task, in time order.
    task_trial_durations_samples : list of int
        Expected trial duration (samples) for each task, in time order.
    gap_factor : float
        An inter-trial interval larger than gap_factor * expected_duration is
        treated as a between-task gap (default 3.0).

    Returns
    -------
    segments : list of np.ndarray
        One rising-edge array per task.
    """
    n_tasks = len(task_n_trials_list)
    if n_tasks == 1:
        return [rising_times]

    intervals = np.diff(rising_times)

    # Use the average expected duration across tasks as a conservative threshold
    avg_expected = np.mean(task_trial_durations_samples)
    gap_threshold = gap_factor * avg_expected

    # Indices where a between-task gap occurs (index i → gap between edge i and i+1)
    gap_indices = np.where(intervals > gap_threshold)[0]

    if len(gap_indices) >= n_tasks - 1:
        # Use the n_tasks-1 largest gaps as split points
        largest_gaps = np.argsort(intervals[gap_indices])[::-1][:n_tasks - 1]
        split_after = sorted(gap_indices[largest_gaps])
    else:
        # Fewer gaps detected than expected — fall back to cumulative trial counts
        print(f"  WARNING: found {len(gap_indices)} inter-task gaps, expected {n_tasks-1}. "
              f"Falling back to cumulative trial-count split.")
        split_after = []
        cursor = 0
        for n in task_n_trials_list[:-1]:
            cursor += n
            split_after.append(cursor - 1)  # split after edge index cursor-1

    # Build segments
    boundaries = [-1] + list(split_after) + [len(rising_times) - 1]
    segments = []
    for k in range(n_tasks):
        start = boundaries[k] + 1
        end = boundaries[k + 1] + 1
        segments.append(rising_times[start:end])

    return segments


def process_task(task_file_path, rising_segment, fs=30000):
    """
    Fix DIO edges, plot diagnostics, and save results for a single task.

    Parameters
    ----------
    task_file_path : Path
        Path to the .txt task file.
    rising_segment : np.ndarray
        Rising edge sample indices that belong to this task.
    fs : int
        Sampling rate (default 30000).
    """
    task_file = parse_grating_experiment(task_file_path)
    task_id = task_file_path.stem
    folder_path = task_file_path.parent

    stimulus_duration = float(task_file['parameters']['stimulus_duration'].rstrip('s'))
    ITI_duration = float(task_file['parameters']['iti_duration'].rstrip('s'))
    n_repeats = task_file['parameters']['total_trials']
    trial_duration = stimulus_duration + ITI_duration

    print(f"\n  Task: {task_id}")
    print(f"  stimulus={stimulus_duration}s  ITI={ITI_duration}s  "
          f"trial_duration={trial_duration}s  n_trials={n_repeats}")
    print(f"  Segment edges: {len(rising_segment)} (expected {n_repeats})")

    trial_duration_samples = int(trial_duration * fs)
    rising_fixed, fix_log = fix_rising_edges(rising_segment, n_repeats, trial_duration_samples)

    print("  Fix log:")
    for entry in fix_log:
        print(f"    {entry}")

    falling_fixed = rising_fixed + int(stimulus_duration * fs)

    # Diagnostics plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(task_id)

    raw_diff = np.diff(rising_segment) / fs
    fixed_diff = np.diff(rising_fixed) / fs

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
    fig.savefig(folder_path / f"{task_id}_DIO_fix.png", dpi=150)
    plt.show()

    save_path = folder_path / f"{task_id}_DIO.npz"
    np.savez_compressed(save_path, rising_times=rising_fixed, falling_times=falling_fixed)
    print(f"  Saved to {save_path}")

    return {
        'task_id': task_id,
        'rising_times': rising_fixed,
        'falling_times': falling_fixed,
        'trial_duration_samples': trial_duration_samples,
        'stimulus_duration': stimulus_duration,
    }


# ── Paths ──────────────────────────────────────────────────────────────────────
rec_folder = Path(r"F:\CnL42SG\Cnl42SG_20260319\CnL42_passive_20260319_124530.rec")

# For a single txt file use a list with one element.
# For multiple tasks recorded in the same rec, list all txt files in time order.
task_file_paths = [
    Path(r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260319\CnL42_drifting_grating_exp_20260319_124602.txt"),
    Path(r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260319\CnL42_drifting_grating_exp_20260319_140702.txt"),
]

animal_id = rec_folder.name.split('.')[0].split('_')[0]
session_id = rec_folder.name.split('.')[0]
print(f"Processing {animal_id}/{session_id}  —  {len(task_file_paths)} task(s)")

# ── Parse all task files ────────────────────────────────────────────────────────
task_metas = []
for p in task_file_paths:
    tf = parse_grating_experiment(p)
    stim = float(tf['parameters']['stimulus_duration'].rstrip('s'))
    iti = float(tf['parameters']['iti_duration'].rstrip('s'))
    n = tf['parameters']['total_trials']
    task_metas.append({'path': p, 'n_trials': n, 'trial_duration': stim + iti})
    print(f"  {p.stem}: {n} trials, {stim+iti}s/trial")

# ── Raw DIO signal ─────────────────────────────────────────────────────────────
fs = 30000
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
pd_time, pd_state = concatenate_din_data(dio_folders, 3)
pd_time = pd_time - pd_time[0]

rising_times = pd_time[np.where(pd_state == 1)[0]]
falling_times = pd_time[np.where(pd_state == 0)[0]]

if len(falling_times) > 0 and len(rising_times) > 0 and falling_times[0] < rising_times[0]:
    falling_times = falling_times[1:]
    print("Discarded leading falling edge (no matching rising edge)")

total_expected = sum(m['n_trials'] for m in task_metas)
print(f"Raw edges: {len(rising_times)} rising (total expected {total_expected})")

# ── Segment DIO by task ────────────────────────────────────────────────────────
task_n_trials = [m['n_trials'] for m in task_metas]
task_durations_samples = [int(m['trial_duration'] * fs) for m in task_metas]

segments = segment_rising_edges_by_task(rising_times, task_n_trials, task_durations_samples)

for k, (meta, seg) in enumerate(zip(task_metas, segments)):
    print(f"\nTask {k+1}/{len(task_metas)}: {meta['path'].stem}  —  {len(seg)} edges in segment")

# ── Per-task processing ────────────────────────────────────────────────────────
for meta, seg in zip(task_metas, segments):
    process_task(meta['path'], seg, fs=fs)
