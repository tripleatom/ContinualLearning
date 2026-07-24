import sys
import shutil
import errno
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(code_dir))

import numpy as np
import pickle
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from recording_preproc.preproc_func import parse_session_info
from grating_utils import parse_grating_experiment


# --- fallback storage location when the primary server runs low on space ---
# Only the server name/share changes; the rest of the path (xl_cl/...) stays identical.
PRIMARY_SERVER_ROOT = r"\\10.129.151.108\xieluanlabs"
BACKUP_SERVER_ROOT  = r"\\10.129.151.88\xieluanlabs2"
MIN_FREE_BYTES      = 5 * 1024 ** 3  # switch to backup server once less than this remains free


def _mirror_on_backup_server(path):
    """Map a path under PRIMARY_SERVER_ROOT to the equivalent path under BACKUP_SERVER_ROOT."""
    path_str = str(path)
    if path_str.lower().startswith(PRIMARY_SERVER_ROOT.lower()):
        return Path(BACKUP_SERVER_ROOT + path_str[len(PRIMARY_SERVER_ROOT):])
    return None


def _resolve_output_folder(session_folder, min_free_bytes=MIN_FREE_BYTES):
    """Return session_folder, or its mirror on the backup server if the primary is low on space."""
    try:
        free = shutil.disk_usage(session_folder).free
    except OSError:
        free = None

    if free is not None and free >= min_free_bytes:
        return session_folder

    backup_folder = _mirror_on_backup_server(session_folder)
    free_str = "unknown" if free is None else f"{free / 1e9:.1f} GB"
    if backup_folder is None:
        print(f"Warning: low space on {session_folder} ({free_str} free), "
              f"but no backup mapping exists for this path - saving in place.")
        return session_folder

    backup_folder.mkdir(parents=True, exist_ok=True)
    print(f"Low space on {session_folder} ({free_str} free) - "
          f"saving to backup server instead: {backup_folder}")
    return backup_folder


def extract_passive_session_spikes(
    rec_folders,
    sortout_folder,
    trial_start,
    trial_end,
    trial_params,
    passive_id,
    session_ids,
    session_trial_counts,
    passive_start=0,
    passive_end=None,
    pre_sec=30.0,
    post_sec=30.0,
    overwrite=True,
):
    """
    Extract a continuous spike train per unit covering [first_stim - pre_sec, last_stim_end + post_sec].

    All passive sessions from a single day are combined into one output file.
    The window is continuous and includes all stimuli, ITIs, and inter-session gaps.
    Spike times are expressed in seconds relative to window_start (t=0).

    Parameters
    ----------
    rec_folders          : list of str or Path – recording folder(s) (.rec) for this day
    sortout_folder       : str or Path  – sortout folder containing curated_analyzer/
    trial_start          : array-like   – stimulus onset sample indices concatenated across
                                          all sessions, in passive-session-local space
                                          (zero-aligned to start of passive DIO data)
    trial_end            : array-like   – stimulus offset sample indices (same space)
    trial_params         : list of dict – per-trial behavioral parameters for all sessions
    passive_id           : str          – identifier for the combined output file (e.g. date)
    session_ids          : list of str  – identifier for each individual passive session
    session_trial_counts : list of int  – number of trials per session, in order
    passive_start        : int          – sample index in concatenated sorter space where the
                                          passive session starts (0 if sorter ran on passive only)
    passive_end          : int or None  – sample index where the passive session ends
                                          (None → last spike + 1)
    pre_sec              : float        – seconds before first stimulus to include (default 30)
    post_sec             : float        – seconds after last stimulus to include (default 30)
    overwrite            : bool         – overwrite existing output file

    Returns
    -------
    pkl_file : Path – path to saved pickle file
    """
    rec_folders = [Path(r) for r in rec_folders]
    animal_id, session_id, _ = parse_session_info(rec_folders[0])

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

    # --- define the extraction window in passive-session-local samples ---
    pre_samples  = int(pre_sec  * fs)
    post_samples = int(post_sec * fs)
    window_start = trial_start[0] - pre_samples
    window_end   = trial_end[-1]  + post_samples
    window_start = max(window_start, 0)

    first_stim_onset_sec = (trial_start[0] - window_start) / fs
    last_stim_offset_sec = (trial_end[-1]  - window_start) / fs
    trial_onsets_sec     = (trial_start    - window_start) / fs
    trial_offsets_sec    = (trial_end      - window_start) / fs
    window_duration_sec  = (window_end - window_start) / fs

    # --- ITI timing: gap between consecutive stimuli (includes inter-session gaps) ---
    iti_onsets_sec    = trial_offsets_sec[:-1]
    iti_offsets_sec   = trial_onsets_sec[1:]
    iti_durations_sec = iti_offsets_sec - iti_onsets_sec

    # --- session boundary trial indices: [0, n0, n0+n1, ...] ---
    session_boundaries = np.concatenate([[0], np.cumsum(session_trial_counts)])

    print(f"Sessions combined  : {len(session_ids)}")
    for k, sid in enumerate(session_ids):
        s, e = int(session_boundaries[k]), int(session_boundaries[k + 1])
        print(f"  [{k+1}] {sid}: trials {s}–{e-1} "
              f"(onset {trial_onsets_sec[s]:.1f} s – {trial_onsets_sec[e-1]:.1f} s)")
    print(f"Extraction window  : {window_start} – {window_end} samples "
          f"({window_duration_sec:.1f} s)")
    print(f"First stim onset   : t={first_stim_onset_sec:.2f} s")
    print(f"Last stim offset   : t={last_stim_offset_sec:.2f} s")
    print(f"ITIs (all)         : n={len(iti_durations_sec)}  "
          f"mean={iti_durations_sec.mean():.2f} s  std={iti_durations_sec.std():.3f} s")

    # --- output file ---
    output_folder = _resolve_output_folder(session_folder)
    pkl_file = output_folder / f'passive_spikes_{passive_id}.pkl'
    if pkl_file.exists() and not overwrite:
        print(f"{pkl_file} exists and overwrite=False – skipping.")
        return pkl_file

    # --- collect spike data for every unit ---
    unit_ids   = sorting.unit_ids
    group_prop = sorting.get_property('group')
    label_prop = sorting.get_property('unit_label')
    group_map  = {uid: int(g) for uid, g in zip(unit_ids, group_prop)} if group_prop is not None else {}
    label_map  = {uid: str(l) for uid, l in zip(unit_ids, label_prop)} if label_prop is not None else {}

    spike_data = {}
    n_units = 0

    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id)

        # shift from concatenated sorter space to passive-session-local space
        passive_end_eff = passive_end if passive_end is not None else int(spike_train[-1]) + 1
        passive_mask    = (spike_train >= passive_start) & (spike_train < passive_end_eff)
        spike_train     = spike_train[passive_mask] - passive_start

        # keep only spikes inside the extraction window
        win_mask        = (spike_train >= window_start) & (spike_train < window_end)
        win_spikes      = spike_train[win_mask]

        # express in seconds relative to window_start (t=0)
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
            'passive_id': passive_id,
            'session_ids': session_ids,
            'session_trial_counts': list(session_trial_counts),
            'recording_folders': [str(r) for r in rec_folders],
            'extraction_date': datetime.now().isoformat(),
            'sampling_frequency': fs,
            'n_units': n_units,
            'n_sessions': len(session_ids),
            'n_trials': len(trial_start),
        },
        'window': {
            # all times in seconds; t=0 is window_start (= first_stim_of_day - pre_sec)
            'window_start_sample': int(window_start),
            'window_end_sample': int(window_end),
            'window_duration_sec': window_duration_sec,
            'pre_stim_sec': pre_sec,
            'post_stim_sec': post_sec,
            'first_stim_onset_sec': first_stim_onset_sec,
            'last_stim_offset_sec': last_stim_offset_sec,
            'trial_onsets_sec': trial_onsets_sec,       # shape (n_trials_total,)
            'trial_offsets_sec': trial_offsets_sec,     # shape (n_trials_total,)
            'iti_onsets_sec': iti_onsets_sec,           # shape (n_trials_total - 1,)
            'iti_offsets_sec': iti_offsets_sec,         # shape (n_trials_total - 1,)
            'iti_durations_sec': iti_durations_sec,     # shape (n_trials_total - 1,)
            'session_boundaries': session_boundaries,   # shape (n_sessions + 1,)
        },
        'trial_params': trial_params,
        'spike_data': spike_data,   # dict keyed by "shankX_unitY"
    }

    print(f"Saving → {pkl_file}")
    try:
        with open(pkl_file, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        backup_folder = _mirror_on_backup_server(session_folder)
        if backup_folder is None:
            raise
        backup_folder.mkdir(parents=True, exist_ok=True)
        pkl_file = backup_folder / pkl_file.name
        print(f"Out of space while saving - retrying on backup server: {pkl_file}")
        with open(pkl_file, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = output['metadata']
    win  = output['window']
    print("\n--- Metadata ---")
    print(f"  animal_id        : {meta['animal_id']}")
    print(f"  passive_id       : {meta['passive_id']}")
    print(f"  n_sessions       : {meta['n_sessions']}")
    print(f"  session_ids      : {meta['session_ids']}")
    print(f"  n_trials         : {meta['n_trials']}")
    print(f"  n_units          : {meta['n_units']}")
    print(f"  sampling_freq    : {meta['sampling_frequency']} Hz")
    print("--- Window ---")
    print(f"  duration         : {win['window_duration_sec']:.2f} s")
    print(f"  first_stim_onset : {win['first_stim_onset_sec']:.3f} s")
    print(f"  last_stim_offset : {win['last_stim_offset_sec']:.3f} s")
    print(f"  mean ITI         : {win['iti_durations_sec'].mean():.3f} s")

    return pkl_file


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from grating_config import ANIMAL_ID, EXPERIMENT_DATE, SORTOUT_FOLDER, PASSIVE_START, PASSIVE_END
    from grating_utils import load_session_paths

    rec_folders, passive_file_paths = load_session_paths(ANIMAL_ID, EXPERIMENT_DATE)
    print(f"Animal: {ANIMAL_ID}  Date: {EXPERIMENT_DATE}")
    print(f"  {len(rec_folders)} rec folder(s), {len(passive_file_paths)} passive file(s)")
    print(f"  passive_start={PASSIVE_START}  passive_end={PASSIVE_END}  (sorter concatenated space)")

    # --- load and concatenate all passive sessions from this day ---
    all_trial_start      = []
    all_trial_end        = []
    all_trial_params     = []
    session_ids          = []
    session_trial_counts = []

    for passive_file_path in passive_file_paths:
        passive_id = passive_file_path.stem
        print(f"\n  Loading: {passive_id}")

        # load pre-processed DIO timing (output of DIO_grating.py)
        dio_npz_path = passive_file_path.parent / f"{passive_id}_DIO.npz"
        if not dio_npz_path.exists():
            raise FileNotFoundError(
                f"DIO npz not found: {dio_npz_path}\n"
                f"Run DIO_grating.py first to generate trial timing."
            )
        dio_data = np.load(dio_npz_path)
        ts = dio_data['rising_times'].ravel()
        te = dio_data['falling_times'].ravel()
        print(f"    DIO trials: {len(ts)}")

        # parse passive file for per-trial behavioral parameters
        passive_file = parse_grating_experiment(passive_file_path)
        if passive_file['trial_data'] is not None:
            params = passive_file['trial_data'].to_dict(orient='records')
        else:
            params = []
            print("    WARNING: no trial_data found — trial_params will be empty for this session")

        if len(params) > 0 and len(ts) != len(params):
            raise ValueError(
                f"DIO trials ({len(ts)}) != behavior trials ({len(params)}) "
                f"in {passive_id}. Check your data."
            )

        all_trial_start.append(ts)
        all_trial_end.append(te)
        all_trial_params.extend(params)
        session_ids.append(passive_id)
        session_trial_counts.append(len(ts))

    all_trial_start = np.concatenate(all_trial_start)
    all_trial_end   = np.concatenate(all_trial_end)

    print(f"\nCombined: {len(session_ids)} sessions, {len(all_trial_start)} trials total")

    # use the experiment date as the combined output identifier
    combined_passive_id = EXPERIMENT_DATE

    pkl_path = extract_passive_session_spikes(
        rec_folders=rec_folders,
        sortout_folder=SORTOUT_FOLDER,
        trial_start=all_trial_start,
        trial_end=all_trial_end,
        trial_params=all_trial_params,
        passive_id=combined_passive_id,
        session_ids=session_ids,
        session_trial_counts=session_trial_counts,
        passive_start=PASSIVE_START,
        passive_end=PASSIVE_END,
        pre_sec=30.0,
        post_sec=30.0,
        overwrite=True,
    )

    print(f"\nData saved to: {pkl_path}")

    # Quick verification
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    w = data['window']
    m = data['metadata']
    print(f"\nWindow : {w['window_duration_sec']:.1f} s total")
    print(f"  t=0  → {w['pre_stim_sec']:.0f} s before first stimulus")
    print(f"  t={w['first_stim_onset_sec']:.2f} s → first stimulus onset")
    print(f"  t={w['last_stim_offset_sec']:.2f} s → last stimulus offset")
    print(f"  mean ITI : {w['iti_durations_sec'].mean():.3f} s ± {w['iti_durations_sec'].std():.3f} s")
    print(f"Sessions : {m['n_sessions']}  Trials : {m['n_trials']}  Units : {m['n_units']}")
    print(f"Session boundaries (trial idx): {w['session_boundaries'].tolist()}")
    print(f"Example unit spike count: {list(data['spike_data'].values())[0]['n_spikes']}")
