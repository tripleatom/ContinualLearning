import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import numpy as np
import pickle
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from process_func import DIO
from rec2nwb.preproc_func import parse_session_info
from readDIO_grating import get_trial_params


def extract_session_spikes(
    rec_folder,
    sortout_folder,
    trial_start,
    trial_end,
    trial_params,
    task_start=0,
    task_end=None,
    pre_sec=30.0,
    post_sec=30.0,
    overwrite=True,
):
    """
    Extract a continuous spike train per unit covering [first_stim - pre_sec, last_stim_end + post_sec].

    Parameters
    ----------
    rec_folder     : str or Path  – recording folder (.rec)
    sortout_folder : str or Path  – session sortout folder containing curated_analyzer/
    trial_start    : array-like   – trial start sample indices in experiment-local space
    trial_end      : array-like   – trial end sample indices in experiment-local space
    trial_params   : list of dict – per-trial behavioral parameters (from get_trial_params)
    task_start     : int          – sample index in concatenated recording where this task starts
    task_end       : int or None  – sample index where this task ends (None → last spike + 1)
    pre_sec        : float        – seconds before first stimulus to include (default 30)
    post_sec       : float        – seconds after last stimulus to include (default 30)
    overwrite      : bool         – overwrite existing output file

    Returns
    -------
    pkl_file : Path – path to saved pickle file
    """
    animal_id, session_id, _ = parse_session_info(rec_folder)

    trial_start = np.array(trial_start).ravel()
    trial_end   = np.array(trial_end).ravel()

    session_folder = Path(sortout_folder)
    curated_analyzer_folder = session_folder / 'curated_analyzer'
    if not curated_analyzer_folder.exists():
        raise FileNotFoundError(f"No curated_analyzer found in {session_folder}")

    sorting_analyzer = load_sorting_analyzer(curated_analyzer_folder)
    sorting = sorting_analyzer.sorting
    fs = sorting.sampling_frequency
    print(f"Sampling frequency: {fs} Hz")

    # --- define the extraction window in experiment-local samples ---
    pre_samples  = int(pre_sec  * fs)
    post_samples = int(post_sec * fs)
    window_start = trial_start[0] - pre_samples   # may be negative if task_start < first stim
    window_end   = trial_end[-1]  + post_samples
    window_start = max(window_start, 0)            # clamp to recording start

    first_stim_onset_sec  = (trial_start[0]  - window_start) / fs
    last_stim_offset_sec  = (trial_end[-1]   - window_start) / fs
    trial_onsets_sec      = (trial_start     - window_start) / fs
    trial_offsets_sec     = (trial_end       - window_start) / fs
    window_duration_sec   = (window_end - window_start) / fs

    print(f"Extraction window: {window_start} – {window_end} samples "
          f"({window_duration_sec:.1f} s)")
    print(f"First stimulus onset at t={first_stim_onset_sec:.2f} s within window")
    print(f"Last stimulus offset at t={last_stim_offset_sec:.2f} s within window")

    # --- output file ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pkl_file = session_folder / f'session_spikes_{timestamp}.pkl'
    if pkl_file.exists() and not overwrite:
        print(f"{pkl_file} exists and overwrite=False – skipping.")
        return pkl_file

    # --- collect spike data for every unit ---
    unit_ids    = sorting.unit_ids
    group_prop  = sorting.get_property('group')
    label_prop  = sorting.get_property('unit_label')
    group_map   = {uid: int(g)  for uid, g in zip(unit_ids, group_prop)} if group_prop is not None else {}
    label_map   = {uid: str(l)  for uid, l in zip(unit_ids, label_prop)} if label_prop is not None else {}

    spike_data = {}
    n_units = 0

    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id)

        # shift from concatenated space to experiment-local space
        task_end_eff = task_end if task_end is not None else int(spike_train[-1]) + 1
        task_mask   = (spike_train >= task_start) & (spike_train < task_end_eff)
        spike_train = spike_train[task_mask] - task_start

        # keep only spikes inside the extraction window
        win_mask    = (spike_train >= window_start) & (spike_train < window_end)
        win_spikes  = spike_train[win_mask]

        # express in seconds relative to window_start
        spike_times_sec = (win_spikes - window_start) / fs

        shank   = group_map.get(unit_id, None)
        quality = label_map.get(unit_id, 'unknown')
        uid_str = f"shank{shank}_unit{unit_id}" if shank is not None else f"unit{unit_id}"

        spike_data[uid_str] = {
            'spike_times_sec': spike_times_sec,   # 1-D array, t=0 → window_start
            'n_spikes': len(spike_times_sec),
            'unit_id': int(unit_id),
            'shank': shank,
            'quality': quality,
        }
        n_units += 1

    print(f"Extracted {n_units} units")

    # --- assemble output ---
    output = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'extraction_date': datetime.now().isoformat(),
            'sampling_frequency': fs,
            'n_units': n_units,
            'n_trials': len(trial_start),
        },
        'window': {
            # all times in seconds; t=0 is window_start (= first_stim - pre_sec)
            'window_start_sample': int(window_start),
            'window_end_sample': int(window_end),
            'window_duration_sec': window_duration_sec,
            'pre_stim_sec': pre_sec,
            'post_stim_sec': post_sec,
            'first_stim_onset_sec': first_stim_onset_sec,
            'last_stim_offset_sec': last_stim_offset_sec,
            'trial_onsets_sec': trial_onsets_sec,    # shape (n_trials,)
            'trial_offsets_sec': trial_offsets_sec,  # shape (n_trials,)
        },
        'trial_params': trial_params,
        'spike_data': spike_data,  # dict keyed by unit label
    }

    print(f"Saving → {pkl_file}")
    with open(pkl_file, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    return pkl_file


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from params import rec_folder, task_file, sortout_folder, task_start, task_end, din_channel

    # Load DIO trial timestamps (same logic as readDIO_grating.py)
    dio_folders = DIO.get_dio_folders(rec_folder)
    dio_folders = sorted(dio_folders, key=lambda x: x.name)
    trial_pd_time, trial_pd_state = DIO.concatenate_din_data(dio_folders, din_channel)
    trial_pd_time  = trial_pd_time.ravel()  - trial_pd_time.ravel()[0]
    trial_pd_state = trial_pd_state.ravel()
    trial_start = trial_pd_time[np.where(trial_pd_state == 1)[0]]
    trial_end   = trial_pd_time[np.where(trial_pd_state == 0)[0][1:]]

    trial_params = get_trial_params(task_file)

    n_DIO    = len(trial_start)
    n_trials = len(trial_params)
    if n_DIO != n_trials:
        raise ValueError(f"DIO trials ({n_DIO}) != behavior trials ({n_trials}). Check your data.")
    print(f"DIO and behavior trial counts match: {n_trials} trials.")

    pkl_path = extract_session_spikes(
        rec_folder=rec_folder,
        sortout_folder=sortout_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        trial_params=trial_params,
        task_start=task_start,
        task_end=task_end,
        pre_sec=30.0,
        post_sec=30.0,
        overwrite=True,
    )

    print(f"\nData saved to: {pkl_path}")

    # Quick verification
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    w = data['window']
    print(f"\nWindow: {w['window_duration_sec']:.1f} s total")
    print(f"  t=0             → {w['pre_stim_sec']:.0f} s before first stimulus")
    print(f"  t={w['first_stim_onset_sec']:.2f} s  → first stimulus onset")
    print(f"  t={w['last_stim_offset_sec']:.2f} s → last stimulus offset")
    print(f"  t={w['window_duration_sec']:.2f} s  → {w['post_stim_sec']:.0f} s after last stimulus")
    print(f"Units extracted: {data['metadata']['n_units']}")
    print(f"Example unit spike count: "
          f"{list(data['spike_data'].values())[0]['n_spikes']}")
