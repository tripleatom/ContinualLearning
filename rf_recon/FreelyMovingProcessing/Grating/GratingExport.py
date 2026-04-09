import numpy as np
import pickle
import os
import json
from pathlib import Path
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
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


def _extract_waveform_info(unit_id, phy_folder, sorting_analyzer_path):
    """
    Extract mean waveform on the best channel and that channel's probe location.

    Tries (in order):
      1. sorting_analyzer templates extension
      2. phy templates.npy + channel_positions.npy

    Returns:
        waveform: list[float] — mean waveform on best channel, or None
        waveform_t_ms: list[float] — time axis in ms, or None
        best_channel: int or None
        channel_location_um: [x, y] in µm, or None
    """
    # --- try sorting_analyzer first ---
    if sorting_analyzer_path and Path(sorting_analyzer_path).exists():
        try:
            analyzer = load_sorting_analyzer(sorting_analyzer_path)
            templates_ext = analyzer.get_extension('templates')
            if templates_ext is not None:
                unit_ids_list = list(analyzer.unit_ids)
                if unit_id in unit_ids_list:
                    idx = unit_ids_list.index(unit_id)
                    templates_data = templates_ext.get_data()   # (n_units, n_samples, n_channels)
                    template = templates_data[idx]               # (n_samples, n_channels)
                    best_ch = int(np.argmax(np.ptp(template, axis=0)))
                    wf = template[:, best_ch].tolist()
                    fs_wf = analyzer.sampling_frequency
                    n_samples = template.shape[0]
                    t_ms = (np.arange(n_samples) / fs_wf * 1000.0 - (n_samples / 2) / fs_wf * 1000.0).tolist()
                    locs = analyzer.get_channel_locations()
                    loc = locs[best_ch].tolist() if locs is not None else None
                    return wf, t_ms, best_ch, loc
        except Exception as e:
            print(f"    [waveform] sorting_analyzer failed for unit {unit_id}: {e}")

    # --- fall back to phy ---
    if phy_folder and Path(phy_folder).exists():
        try:
            templates_file = Path(phy_folder) / 'templates.npy'
            positions_file = Path(phy_folder) / 'channel_positions.npy'
            if templates_file.exists():
                templates = np.load(templates_file)   # (n_templates, n_samples, n_channels)
                uid = int(unit_id)
                if uid < templates.shape[0]:
                    template = templates[uid]           # (n_samples, n_channels)
                    best_ch = int(np.argmax(np.ptp(template, axis=0)))
                    wf = template[:, best_ch].tolist()
                    # phy templates have no guaranteed sample rate stored here;
                    # use index as proxy and let the caller supply fs if needed
                    n_samples = template.shape[0]
                    t_ms = (np.arange(n_samples) - n_samples // 2).tolist()  # in samples, caller converts
                    loc = None
                    if positions_file.exists():
                        positions = np.load(positions_file)
                        if best_ch < len(positions):
                            loc = positions[best_ch].tolist()
                    return wf, t_ms, best_ch, loc
        except Exception as e:
            print(f"    [waveform] phy fallback failed for unit {unit_id}: {e}")

    return None, None, None, None


def extract_grating_neural_data_for_embedding(rec_folder, task_file_path, sortout_folder, task_start=0, task_end=None):
    rec_folder = Path(rec_folder)
    task_file_path = Path(task_file_path)
    session_folder = Path(sortout_folder)

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

    output_dir = session_folder / 'passive_embedding_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load unit quality labels from session folder
    labels_file = session_folder / 'unit_labels.json'
    unit_labels = {}
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            unit_labels = json.load(f)
        print(f"Loaded unit labels from {labels_file}")
    else:
        print(f"No unit_labels.json found in {session_folder}, quality will be 'unknown'")

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
    ishs = ['0', '1', '2', '3', '4', '5', '6', '7']

    for ish in ishs:
        shank_folder = session_folder / f'shank{ish}'
        if not shank_folder.exists():
            continue

        sorting_results_folders = [
            os.path.join(root, d)
            for root, dirs, _ in os.walk(shank_folder)
            for d in dirs if d.startswith('sorting_results_')
        ]

        for sorting_results_folder in sorting_results_folders:
            phy_folder = Path(sorting_results_folder) / 'phy'
            sorting_analyzer_path = Path(sorting_results_folder) / 'sorting_analyzer'

            sorting = None
            loaded_from_phy = False
            if phy_folder.exists():
                try:
                    sorting = PhySortingExtractor(phy_folder)
                    loaded_from_phy = True
                    print(f"Loaded from phy: {phy_folder}")
                except Exception as e:
                    print(f"Failed to load from phy: {e}")
                    if sorting_analyzer_path.exists():
                        try:
                            sorting = load_sorting_analyzer(sorting_analyzer_path).sorting
                            # print(f"Loaded from sorting_analyzer: {sorting_analyzer_path}")
                        except Exception as e2:
                            print(f"Failed to load from sorting_analyzer: {e2}")
                    else:
                        print(f"Could not load sorting from {sorting_results_folder}")
            elif sorting_analyzer_path.exists():
                try:
                    sorting = load_sorting_analyzer(sorting_analyzer_path).sorting
                    # print(f"Loaded from sorting_analyzer: {sorting_analyzer_path}")
                except Exception as e:
                    print(f"Failed to load from sorting_analyzer: {e}")
            else:
                print(f"No valid sorting folder found in {sorting_results_folder}")

            if sorting is None:
                continue

            fs = sorting.sampling_frequency
            neural_data['metadata']['sampling_frequency'] = fs
            unit_ids = sorting.unit_ids

            if loaded_from_phy:
                phy_qualities = sorting.get_property('quality')
                phy_quality_map = {uid: q for uid, q in zip(unit_ids, phy_qualities)} if phy_qualities is not None else {}

            for unit_id in unit_ids:
                # Use phy quality if loaded from phy, otherwise use unit_labels.json
                if loaded_from_phy:
                    quality = phy_quality_map.get(unit_id, 'unknown')
                else:
                    quality = unit_labels.get(f'shank{ish}', {}).get(str(unit_id), 'unknown')
                if quality == 'noise':
                    continue

                try:
                    spike_train = sorting.get_unit_spike_train(unit_id)
                except Exception as e:
                    print(f"Error getting spike train for unit {unit_id}: {e}")
                    continue

                if len(spike_train) == 0:
                    continue

                # Align spike train from concatenated space to experiment-local space
                task_end_eff = task_end if task_end is not None else int(spike_train[-1]) + 1
                task_mask = (spike_train >= task_start) & (spike_train < task_end_eff)
                spike_train = spike_train[task_mask] - task_start

                unique_unit_id = f"shank{ish}_unit{unit_id}"

                # Waveform + channel location
                wf, wf_t_ms, best_ch, ch_loc = _extract_waveform_info(
                    unit_id, phy_folder, sorting_analyzer_path
                )

                # ACG from full (task-masked) spike train
                acg_counts, acg_lags_ms = _compute_acg(spike_train, fs)

                neural_data['unit_info'][unique_unit_id] = {
                    'original_unit_id': int(unit_id),
                    'shank': ish,
                    'quality': quality,
                    'sorting_folder': sorting_results_folder,
                    'n_spikes_total': len(spike_train),
                    'unit_index': unit_counter,
                    # waveform / electrode
                    'best_channel': best_ch,
                    'channel_location_um': ch_loc,
                    'waveform_template': wf,
                    'waveform_t_ms': wf_t_ms,
                    # autocorrelogram
                    'acg_counts': acg_counts.tolist(),
                    'acg_lags_ms': acg_lags_ms.tolist(),
                }

                window_pre_samples = int(window_pre * fs)
                window_post_samples = int(window_post * fs)
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
        'shanks_processed': ishs
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


if __name__ == "__main__":
    import traceback
    from rf_recon.FreelyMovingProcessing.Grating.grating_utils import load_session_paths
    try:
        # ── Animal / session config ────────────────────────────────────────────
        from rf_recon.FreelyMovingProcessing.Grating.grating_config import ANIMAL_ID as Animal_id, EXPERIMENT_DATE as experiment_date

        rec_folder, passive_log_paths = load_session_paths(Animal_id, experiment_date)

        # task_start / task_end: sample offsets in the concatenated sorting space.
        # Set task_start=0, task_end=None if sorting was done on this session alone.
        passive_offsets = [
            {'task_start': 0, 'task_end': None}
            for _ in passive_log_paths
        ]

        for p in passive_log_paths:
            if not p.exists():
                print(f"Task file does not exist: {p}")
                exit(1)

        sortout_folder = input("Please enter the path to the session sortout folder (parent of shank0, shank1, ... folders): ").strip().strip('"')

        all_data = []
        for task_file_path, offsets in zip(passive_log_paths, passive_offsets):
            print(f"\n{'='*60}")
            print(f"Extracting: {task_file_path.stem}")
            nd = extract_grating_neural_data_for_embedding(
                rec_folder, task_file_path, sortout_folder,
                task_start=offsets['task_start'], task_end=offsets['task_end']
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
        output_dir = Path(sortout_folder) / 'passive_embedding_analysis'
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
