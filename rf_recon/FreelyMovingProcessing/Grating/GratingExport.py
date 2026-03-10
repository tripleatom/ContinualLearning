import numpy as np
import pickle
import os
import json
from pathlib import Path
from datetime import datetime
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rf_recon.FreelyMovingProcessing.Grating.parse_grating_experiment import parse_grating_experiment


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
                neural_data['unit_info'][unique_unit_id] = {
                    'original_unit_id': int(unit_id),
                    'shank': ish,
                    'quality': quality,
                    'sorting_folder': sorting_results_folder,
                    'n_spikes_total': len(spike_train),
                    'unit_index': unit_counter
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

    base_filename = f"{animal_id}_{session_id}_grating_data"
    filepath = output_dir / f"{base_filename}.pkl"
    print(f"Saving to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(neural_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return neural_data


def load_neural_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import traceback
    try:
        rec_folder = Path(r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260304/CnL42_20260304/CnL42SG_passive_20260304_142720.rec")
        task_file_path = Path(r"/Volumes/xieluanlabs/xl_cl/experiment_data/CnL42/260304/CnL42_drifting_grating_exp_20260304_142748.txt")

        if not rec_folder.exists():
            print(f"Recording folder does not exist: {rec_folder}")
            exit(1)
        if not task_file_path.exists():
            print(f"Task file does not exist: {task_file_path}")
            exit(1)

        sortout_folder = input("Please enter the path to the session sortout folder (parent of shank0, shank1, ... folders): ").strip().strip('"')

        # Offset of this experiment in the concatenated recording (in sample points)
        # Set task_start/task_end if sorting was done on a concatenated recording;
        # use task_start=0, task_end=None if sorting was done on this session alone.
        task_start = 0       # sample point where this experiment starts in the concatenated recording
        task_end   = 149938305    # sample point where it ends; None = use last spike as upper bound

        neural_data = extract_grating_neural_data_for_embedding(rec_folder, task_file_path, sortout_folder, task_start=task_start, task_end=task_end)

        if neural_data is None:
            print("Failed to extract neural data")
            exit(1)

        print(f"\n- {len(neural_data['spike_data'])} units")
        print(f"- {neural_data['metadata']['n_trials']} trials")
        print(f"- Orientations: {neural_data['trial_info']['unique_orientations']}")
        print(f"- Stimulus duration: {neural_data['experiment_parameters']['stimulus_duration']}s")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
