import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from rf_recon.FreelyMovingProcessing.Grating.parse_grating_experiment import parse_grating_experiment


def _compute_acg(spike_train_samples, fs, max_lag_ms=25.0, bin_ms=1.0):
    """
    Compute autocorrelogram from a spike train (in samples).

    Returns:
        acg_counts: normalized counts (Hz units, firing rate in each lag bin)
        lags_ms: lag axis in milliseconds
    """
    n_bins = int(max_lag_ms / bin_ms)
    bins_edges = np.arange(-n_bins - 0.5, n_bins + 1.5) * bin_ms

    if len(spike_train_samples) == 0:
        lags_ms = np.arange(-n_bins, n_bins + 1) * bin_ms
        return np.zeros(2 * n_bins + 1), lags_ms

    t_ms = np.sort(spike_train_samples.astype(float)) / fs * 1000.0

    # Subsample for speed on very dense recordings
    if len(t_ms) > 5000:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(t_ms), 5000, replace=False))
        t_ms = t_ms[idx]

    all_diffs = []
    for i in range(len(t_ms)):
        lo = np.searchsorted(t_ms, t_ms[i] - max_lag_ms, 'left')
        hi = np.searchsorted(t_ms, t_ms[i] + max_lag_ms + 1e-6, 'right')
        diffs = t_ms[lo:hi] - t_ms[i]
        diffs = diffs[diffs != 0]
        all_diffs.append(diffs)

    all_diffs = np.concatenate(all_diffs) if all_diffs else np.array([])
    counts, _ = np.histogram(all_diffs, bins=bins_edges)
    lags_ms = np.arange(-n_bins, n_bins + 1) * bin_ms

    # Normalize: counts / (n_spikes * bin_width_s) → Hz
    counts_norm = counts.astype(float) / (len(t_ms) * bin_ms / 1000.0)
    return counts_norm, lags_ms


def _extract_waveform_info(unit_id, sorting_analyzer):
    """
    Extract mean waveform on the best channel plus full-channel templates.

    Uses the curated sorting_analyzer templates extension.

    Returns:
        waveform: list[float] — mean waveform on best channel, or None
        waveform_t_ms: list[float] — time axis in ms, or None
        best_channel: int or None
        channel_location_um: [x, y] in µm, or None
        waveform_template_all_channels: list[list[float]] — samples x channels, or None
        channel_locations_um: list[list[float]], or None
        waveform_channel_ids: list, or None
        waveform_channel_groups: list, or None
    """
    try:
        templates_ext = sorting_analyzer.get_extension('templates')
        if templates_ext is not None:
            unit_ids_list = list(sorting_analyzer.unit_ids)
            if unit_id in unit_ids_list:
                idx = unit_ids_list.index(unit_id)
                templates_data = templates_ext.get_data()   # (n_units, n_samples, n_channels)
                template = templates_data[idx]               # (n_samples, n_channels)
                best_ch = int(np.argmax(np.ptp(template, axis=0)))
                wf = template[:, best_ch].tolist()
                fs_wf = sorting_analyzer.sampling_frequency
                n_samples = template.shape[0]
                t_ms = (np.arange(n_samples) / fs_wf * 1000.0 - (n_samples / 2) / fs_wf * 1000.0).tolist()
                locs = sorting_analyzer.get_channel_locations()
                loc = locs[best_ch].tolist() if locs is not None else None
                all_locs = locs.tolist() if locs is not None else None
                channel_ids = list(getattr(sorting_analyzer, 'channel_ids', [])) or None
                channel_groups = None
                recording = getattr(sorting_analyzer, 'recording', None)
                if recording is not None:
                    try:
                        groups = recording.get_property('group')
                        channel_groups = groups.tolist() if hasattr(groups, 'tolist') else list(groups)
                    except Exception:
                        channel_groups = None
                return wf, t_ms, best_ch, loc, template.tolist(), all_locs, channel_ids, channel_groups
    except Exception as e:
        print(f"    [waveform] sorting_analyzer failed for unit {unit_id}: {e}")

    return None, None, None, None, None, None, None, None


def extract_grating_neural_data_for_embedding(
    rec_folder,
    task_file_path,
    sortout_folder,
    passive_start=0,
    passive_end=None,
):
    rec_folder = Path(rec_folder)
    task_file_path = Path(task_file_path)
    sortout_path = Path(sortout_folder)
    curated_analyzer_path = (
        sortout_path if sortout_path.name == 'curated_analyzer'
        else sortout_path / 'curated_analyzer'
    )
    if not curated_analyzer_path.exists():
        raise FileNotFoundError(f"curated_analyzer path does not exist: {curated_analyzer_path}")

    rec_name = rec_folder.name
    if rec_name.endswith('.rec'):
        rec_name = rec_name[:-4]
    animal_id = rec_name.split('_')[0]
    session_id = rec_name

    print(f"Extracting grating data for {animal_id}/{session_id}")

    try:
        task_file = parse_grating_experiment(task_file_path)
        df = task_file['trial_data']
    except Exception as e:
        print(f"Error parsing grating experiment: {e}")
        return None

    try:
        stimulus_duration = task_file['parameters']['stimulus_duration']
        ITI_duration = task_file['parameters']['iti_duration']
        if isinstance(stimulus_duration, str):
            stimulus_duration = float(stimulus_duration.rstrip('s'))
        else:
            stimulus_duration = float(stimulus_duration)
        if isinstance(ITI_duration, str):
            ITI_duration = float(ITI_duration.rstrip('s'))
        else:
            ITI_duration = float(ITI_duration)
        n_repeats = task_file['parameters']['total_trials']
        trial_duration = stimulus_duration + ITI_duration
    except Exception as e:
        print(f"Error extracting timing parameters: {e}")
        stimulus_duration = 2.0
        ITI_duration = 1.0
        n_repeats = len(df) if df is not None else 0
        trial_duration = stimulus_duration + ITI_duration

    print(f"Stimulus duration: {stimulus_duration}s, ITI: {ITI_duration}s, Total trials: {n_repeats}")

    task_id = task_file_path.stem
    processed_dio_folder = task_file_path.parent / f"{task_id}_DIO.npz"
    try:
        dio_data = np.load(processed_dio_folder)
        rising_times = dio_data['rising_times']
        falling_times = dio_data['falling_times']
    except Exception as e:
        print(f"Error loading DIO data from {processed_dio_folder}: {e}")
        return None

    if df is None or len(df) == 0:
        print("No trial data found")
        return None

    n_trials = len(df)

    if 'L_Orient' in df.columns:
        orientations = df['L_Orient'].values
    else:
        orientation_cols = [col for col in df.columns if 'orient' in col.lower()]
        if orientation_cols:
            print(f"Using {orientation_cols[0]} as orientation column")
            orientations = df[orientation_cols[0]].values
        else:
            print("No orientation column found, using dummy orientations")
            orientations = np.zeros(n_trials)

    spatial_freqs = df['L_SF'].values if 'L_SF' in df.columns else np.full(n_trials, None)
    contrasts = df['L_Contrast'].values if 'L_Contrast' in df.columns else np.full(n_trials, None)
    phases = df['L_Phase'].values if 'L_Phase' in df.columns else np.full(n_trials, None)

    unique_orientations = np.unique(orientations)
    unique_spatial_freqs = np.unique(spatial_freqs) if 'L_SF' in df.columns else 'N/A'
    unique_phases = np.unique(phases) if 'L_Phase' in df.columns else 'N/A'
    print(f"Unique orientations: {unique_orientations}")
    print(f"Unique spatial frequencies: {unique_spatial_freqs}")
    print(f"Unique phases: {unique_phases}")
    print(f"Number of trials: {n_trials}")

    if len(rising_times) < n_trials or len(falling_times) < n_trials:
        n_trials = min(n_trials, len(rising_times), len(falling_times))
        orientations = orientations[:n_trials]
        df = df.iloc[:n_trials]
        print(f"DIO mismatch, adjusted to {n_trials} trials")

    trial_windows = [(rising_times[i], falling_times[i]) for i in range(n_trials)]

    neural_data = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'task_file': str(task_file_path),
            'extraction_date': datetime.now().isoformat(),
            'n_trials': n_trials,
            'experiment_type': 'grating'
        },
        'experiment_parameters': {
            'stimulus_duration': stimulus_duration,
            'iti_duration': ITI_duration,
            'trial_duration': trial_duration,
            'total_trials': n_repeats
        },
        'trial_info': {
            'orientations': orientations.tolist(),
            'unique_orientations': unique_orientations.tolist(),
            'trial_windows': trial_windows,
            'all_trial_parameters': df.to_dict('records')
        },
        'spike_data': {},
        'unit_info': {}
    }

    window_pre = 0.2   # seconds before stimulus onset
    window_post = 2.0  # seconds after stimulus onset
    unit_counter = 0
    shanks_processed = []

    sorting_analyzer = load_sorting_analyzer(curated_analyzer_path)
    sorting = sorting_analyzer.sorting
    sorting_folder = curated_analyzer_path
    print(f"Loaded curated_analyzer: {curated_analyzer_path}")

    fs = sorting.sampling_frequency
    neural_data['metadata']['sampling_frequency'] = fs
    unit_ids = sorting.unit_ids

    group_prop = sorting.get_property('group')
    group_map = (
        {uid: int(g) for uid, g in zip(unit_ids, group_prop)}
        if group_prop is not None else {}
    )
    label_prop = sorting.get_property('unit_label')
    if label_prop is None:
        raise ValueError(
            f"curated_analyzer is missing required sorting property 'unit_label': "
            f"{curated_analyzer_path}"
        )
    label_map = {uid: str(l) for uid, l in zip(unit_ids, label_prop)}

    window_pre_samples = int(window_pre * fs)
    window_post_samples = int(window_post * fs)

    for unit_id in unit_ids:
        shank = group_map.get(unit_id, None)
        shank_key = f'shank{shank}' if shank is not None else 'unknown'
        quality = label_map[unit_id]

        if shank_key not in shanks_processed:
            shanks_processed.append(shank_key)

        try:
            spike_train = sorting.get_unit_spike_train(unit_id)
        except Exception as e:
            print(f"Error getting spike train for unit {unit_id}: {e}")
            continue

        if len(spike_train) == 0:
            continue

        # Align spike train from concatenated sorting space to passive-local space
        passive_end_eff = passive_end if passive_end is not None else int(spike_train[-1]) + 1
        passive_mask = (spike_train >= passive_start) & (spike_train < passive_end_eff)
        spike_train = spike_train[passive_mask] - passive_start

        unique_unit_id = f"{shank_key}_unit{unit_id}"

        # Waveform + channel location
        wf, wf_t_ms, best_ch, ch_loc, wf_all_ch, ch_locs, wf_ch_ids, wf_ch_groups = _extract_waveform_info(
            unit_id, sorting_analyzer
        )

        # ACG from full (task-masked) spike train
        acg_counts, acg_lags_ms = _compute_acg(spike_train, fs)

        neural_data['unit_info'][unique_unit_id] = {
            'original_unit_id': int(unit_id),
            'shank': shank,
            'quality': quality,
            'sorting_folder': str(sorting_folder),
            'n_spikes_total': len(spike_train),
            'unit_index': unit_counter,
            # waveform / electrode
            'best_channel': best_ch,
            'channel_location_um': ch_loc,
            'waveform_template': wf,
            'waveform_template_all_channels': wf_all_ch,
            'waveform_t_ms': wf_t_ms,
            'channel_locations_um': ch_locs,
            'waveform_channel_ids': wf_ch_ids,
            'waveform_channel_groups': wf_ch_groups,
            # autocorrelogram
            'acg_counts': acg_counts.tolist(),
            'acg_lags_ms': acg_lags_ms.tolist(),
        }

        trial_spike_data = []
        for i_trial, (start, end) in enumerate(trial_windows):
            start_samples = int(start)
            trial_spikes = spike_train[
                (spike_train >= start_samples - window_pre_samples) &
                (spike_train < start_samples + window_post_samples)
            ]
            trial_spikes_relative = (trial_spikes - start_samples) / fs if len(trial_spikes) > 0 else np.array([])
            trial_spike_data.append({
                'trial_index': i_trial,
                'orientation': orientations[i_trial] if i_trial < len(orientations) else None,
                'spatial_freq': spatial_freqs[i_trial] if i_trial < len(spatial_freqs) else None,
                'contrast': contrasts[i_trial] if i_trial < len(contrasts) else None,
                'phase': phases[i_trial] if i_trial < len(phases) else None,
                'spike_times': trial_spikes_relative.tolist(),
                'spike_count': len(trial_spikes_relative),
                'trial_start': start,
                'trial_end': end
            })

        neural_data['spike_data'][unique_unit_id] = trial_spike_data
        unit_counter += 1

    neural_data['extraction_params'] = {
        'window_pre': window_pre,
        'window_post': window_post,
        'total_units': unit_counter,
        'shanks_processed': shanks_processed
    }

    print(f"\nExtraction complete: {unit_counter} units, {n_trials} trials, orientations: {unique_orientations}, spatial_freqs: {unique_spatial_freqs}, phases: {unique_phases}")

    return neural_data


def merge_grating_neural_data(data_list):
    """
    Merge neural data dicts from multiple experiments (same session, same sorting).

    Trial info is concatenated in order. Spike data for each unit is concatenated
    with trial indices re-numbered sequentially. Unit info is taken from the first
    experiment that contains the unit (waveform/ACG are session-level, not trial-level).

    Parameters
    ----------
    data_list : list of dict
        Output of extract_grating_neural_data_for_embedding, one per experiment.

    Returns
    -------
    merged : dict
        Single merged neural_data dict.
    """
    if len(data_list) == 1:
        return data_list[0]

    base = data_list[0]

    # Build merged trial_info and spike_data
    orientations = []
    trial_windows = []
    all_trial_parameters = []
    spike_data = {}
    unit_info = {}
    trial_offset = 0

    for d in data_list:
        ti = d['trial_info']
        orientations.extend(ti['orientations'])
        trial_windows.extend(ti['trial_windows'])
        all_trial_parameters.extend(ti['all_trial_parameters'])

        for uid, trials in d['spike_data'].items():
            if uid not in spike_data:
                spike_data[uid] = []
            for t in trials:
                t_copy = dict(t)
                t_copy['trial_index'] = t['trial_index'] + trial_offset
                spike_data[uid].append(t_copy)

        for uid, info in d['unit_info'].items():
            if uid not in unit_info:
                unit_info[uid] = dict(info)
            unit_info[uid]['n_spikes_total'] = (
                unit_info[uid].get('n_spikes_total', 0) + info['n_spikes_total']
            )

        trial_offset += d['metadata']['n_trials']

    total_trials = trial_offset
    unique_orientations = np.unique(orientations).tolist()

    merged = {
        'metadata': {
            **base['metadata'],
            'task_file': [str(d['metadata']['task_file']) for d in data_list],
            'n_trials': total_trials,
        },
        'experiment_parameters': {
            **base['experiment_parameters'],
            'total_trials': total_trials,
        },
        'trial_info': {
            'orientations': orientations,
            'unique_orientations': unique_orientations,
            'trial_windows': trial_windows,
            'all_trial_parameters': all_trial_parameters,
        },
        'spike_data': spike_data,
        'unit_info': unit_info,
        'extraction_params': {
            **base['extraction_params'],
            'total_units': len(spike_data),
        },
    }

    return merged


def load_neural_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _rec_folders_for_task_files(rec_folders, task_file_paths):
    """
    Match rec folders to task files.

    If the CSV has one passive recording and multiple task files, reuse that
    recording for every task. If it has one passive recording per task, keep
    them paired in CSV order.
    """
    if not rec_folders:
        raise ValueError("No passive recording folders were found in the CSV.")
    if not task_file_paths:
        raise ValueError("No task files were found in the CSV.")
    if len(rec_folders) == 1:
        return [rec_folders[0] for _ in task_file_paths]
    if len(rec_folders) == len(task_file_paths):
        return list(rec_folders)
    raise ValueError(
        f"Found {len(rec_folders)} passive recording folder(s) and "
        f"{len(task_file_paths)} task file(s). Use either one passive folder "
        "for all tasks or one passive folder per task."
    )


def _normalize_passive_window(window):
    """Accept {'passive_start': x, 'passive_end': y} or (x, y)."""
    if isinstance(window, dict):
        return {
            'passive_start': window.get('passive_start', window.get('start')),
            'passive_end': window.get('passive_end', window.get('end')),
        }
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return {'passive_start': window[0], 'passive_end': window[1]}
    raise ValueError(
        "Each PASSIVE_WINDOWS entry must be a dict with passive_start/passive_end "
        "or a two-item tuple/list."
    )


if __name__ == "__main__":
    import traceback
    from rf_recon.FreelyMovingProcessing.Grating.grating_utils import load_session_paths
    try:
        # ── Animal / session config ────────────────────────────────────────────
        from rf_recon.FreelyMovingProcessing.Grating.grating_config import (
            ANIMAL_ID as Animal_id,
            EXPERIMENT_DATE as experiment_date,
            SORTOUT_FOLDER,
            PASSIVE_START,
            PASSIVE_END,
            PASSIVE_WINDOWS,
        )

        rec_folders, passive_log_paths = load_session_paths(Animal_id, experiment_date)
        rec_folders_for_tasks = _rec_folders_for_task_files(rec_folders, passive_log_paths)
        rec_folder = rec_folders_for_tasks[0]

        # passive_start / passive_end: sample offsets in the concatenated sorting space.
        # Set passive_start=0, passive_end=None if sorting was done on passive only.
        if PASSIVE_WINDOWS is None:
            passive_windows = [
                {'passive_start': PASSIVE_START, 'passive_end': PASSIVE_END}
                for _ in passive_log_paths
            ]
        else:
            if len(PASSIVE_WINDOWS) != len(passive_log_paths):
                raise ValueError(
                    f"PASSIVE_WINDOWS has {len(PASSIVE_WINDOWS)} entries, but "
                    f"{len(passive_log_paths)} task file(s) were found in the CSV."
                )
            passive_windows = [_normalize_passive_window(w) for w in PASSIVE_WINDOWS]

        for p in passive_log_paths:
            if not p.exists():
                print(f"Task file does not exist: {p}")
                exit(1)

        sortout_folder = Path(SORTOUT_FOLDER)
        print(f"Using sortout folder: {sortout_folder}")

        all_data = []
        print(f"Task files found: {len(passive_log_paths)}")
        for task_file_path, passive_window, rec_folder_for_task in zip(
            passive_log_paths, passive_windows, rec_folders_for_tasks
        ):
            print(f"\n{'='*60}")
            print(f"Extracting: {task_file_path.stem}")
            print(
                f"Passive window: start={passive_window['passive_start']} "
                f"end={passive_window['passive_end']}"
            )
            nd = extract_grating_neural_data_for_embedding(
                rec_folder_for_task, task_file_path, sortout_folder,
                passive_start=passive_window['passive_start'],
                passive_end=passive_window['passive_end'],
            )
            if nd is None:
                print(f"Failed to extract {task_file_path.stem}, skipping.")
                continue
            all_data.append(nd)

        if not all_data:
            print("No data extracted.")
            exit(1)

        rec_name = rec_folder.name.replace('.rec', '')
        animal_id = rec_name.split('_')[0]
        output_base = sortout_folder.parent if sortout_folder.name == 'curated_analyzer' else sortout_folder
        output_dir = output_base / 'passive_embedding_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Save options ────────────────────────────────────────────────────────
        # merge_output=True  → one merged pkl
        # merge_output=False → one pkl per experiment
        merge_output = True

        if merge_output and len(all_data) > 1:
            print(f"\nMerging {len(all_data)} experiments...")
            neural_data = merge_grating_neural_data(all_data)
            filepath = output_dir / f"{rec_name}_grating_data_merged.pkl"
            print(f"Saving merged data to {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(neural_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n- {len(neural_data['spike_data'])} units")
            print(f"- {neural_data['metadata']['n_trials']} trials total")
            print(f"- Orientations: {neural_data['trial_info']['unique_orientations']}")
        else:
            for nd, log_path in zip(all_data, passive_log_paths):
                task_time = log_path.stem.rsplit('_', 1)[-1]  # just the HHmmss timestamp
                filepath = output_dir / f"{rec_name}_{task_time}_grating_data.pkl"
                print(f"Saving {filepath}")
                with open(filepath, 'wb') as f:
                    pickle.dump(nd, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"  - {len(nd['spike_data'])} units, {nd['metadata']['n_trials']} trials")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
