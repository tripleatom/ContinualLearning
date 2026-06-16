import sys
from pathlib import Path

# Add the parent 'code' directory to Python path
# Use __file__ to get the script location, not cwd()
code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import numpy as np
from datetime import datetime
from pathlib import Path
from spikeinterface import load_sorting_analyzer
from process_func import DIO
from rec2nwb.preproc_func import parse_session_info
import pickle
import json

def get_session_position(filepath):
    """
    Return the full session DLC tracking (not sliced to stim windows).

    Output dict keys:
        dlc_x, dlc_y      : 1-D arrays, position samples
        dlc_heading       : 1-D array, heading direction
        dlc_head_angle    : 1-D array, head angle
        dlc_signal        : 1-D array, tracking quality/confidence
        step_time         : 1-D array, session-relative seconds (same frame as JSON's stepTime)
        session_t0        : float, Unix-epoch session start (matches stimulusOnsetTime/choiceTime)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {
        'dlc_x':          np.asarray(data.get('dlcX', []),         dtype=float),
        'dlc_y':          np.asarray(data.get('dlcY', []),         dtype=float),
        'dlc_heading':    np.asarray(data.get('dlcHeading', []),   dtype=float),
        'dlc_head_angle': np.asarray(data.get('dlcHeadAngle', []), dtype=float),
        'dlc_signal':     np.asarray(data.get('dlcSignal', []),    dtype=float),
        'step_time':      np.asarray(data.get('stepTime', []),     dtype=float),
        'session_t0':     data.get('startTime'),
    }


def get_trial_params(filepath):
    """
    Extract per-trial behavioral and grating parameters from a session JSON file.

    Returns a list of dicts, one per trial, with keys:
        rewardedOnLeft, leftOrientation, rightOrientation,
        leftSpatialFreq, rightSpatialFreq, leftTemporalFreq, rightTemporalFreq,
        leftContrast, rightContrast, rewardedOrientation, nonRewardedOrientation,
        choice, correct, rewarded, stimulusOnsetTime, choiceTime, choiceLatency,
        position_x, position_y, position_time

    position_x/position_y/position_time cover only the visual-stim ON window:
    samples whose stepTime falls in [stimulusOnsetTime, choiceTime]. The JSON
    stores stepTime in session-relative seconds while stimulusOnsetTime and
    choiceTime are Unix epoch seconds, so we subtract the session-level
    startTime before searchsorted'ing. If either boundary is missing the
    position arrays are returned empty.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    keys = [
        'rewardedOnLeft',
        'leftOrientation', 'rightOrientation',
        'leftSpatialFreq', 'rightSpatialFreq',
        'leftTemporalFreq', 'rightTemporalFreq',
        'leftContrast', 'rightContrast',
        'rewardedOrientation', 'nonRewardedOrientation',
        'choice', 'correct', 'rewarded',
        'stimulusOnsetTime', 'choiceTime', 'choiceLatency',
    ]

    dlc_x = np.asarray(data.get('dlcX', []), dtype=float)
    dlc_y = np.asarray(data.get('dlcY', []), dtype=float)
    dlc_heading = np.asarray(data.get('dlcHeading', []), dtype=float)
    dlc_head_angle = np.asarray(data.get('dlcHeadAngle', []), dtype=float)
    dlc_signal = np.asarray(data.get('dlcSignal', []), dtype=float)
    step_time = np.asarray(data.get('stepTime', []), dtype=float)
    session_t0 = data.get('startTime')  # Unix-epoch session start, matches stim/choice times

    def _slice(arr, sl):
        return arr[sl] if arr.size == step_time.size else np.array([], dtype=float)

    out = []
    for trial in data.get('trials', []):
        td = {k: trial.get(k, None) for k in keys}
        stim_on = trial.get('stimulusOnsetTime')
        stim_off = trial.get('choiceTime')  # photodiode falls = stim off
        if (stim_on is not None and stim_off is not None
                and step_time.size > 0 and stim_off >= stim_on
                and session_t0 is not None):
            stim_on_rel = stim_on - session_t0
            stim_off_rel = stim_off - session_t0
            i0 = int(np.searchsorted(step_time, stim_on_rel, side='left'))
            i1 = int(np.searchsorted(step_time, stim_off_rel, side='right'))
            sl = slice(i0, i1)
            td['position_x'] = dlc_x[sl]
            td['position_y'] = dlc_y[sl]
            td['position_time'] = step_time[sl]
            td['heading'] = _slice(dlc_heading, sl)
            td['head_angle'] = _slice(dlc_head_angle, sl)
            td['dlc_signal'] = _slice(dlc_signal, sl)
        else:
            td['position_x'] = np.array([], dtype=float)
            td['position_y'] = np.array([], dtype=float)
            td['position_time'] = np.array([], dtype=float)
            td['heading'] = np.array([], dtype=float)
            td['head_angle'] = np.array([], dtype=float)
            td['dlc_signal'] = np.array([], dtype=float)
        out.append(td)
    return out

def process_behavior_trial_responses(rec_folder, trial_start, trial_end, trial_params, sortout_folder, task_start=0, task_end=None, overwrite=True):
    """
    Process behavior trial responses from an experiment folder.

    The function:
      - Loads spike times for each trial based on trial start/end timestamps
      - Stores trial information (rewarded_on_left condition)
      - Organizes trial information in a structured format matching the embedding extractor schema
      - Saves detailed trial-by-trial data into a PKL file

    Parameters:
      rec_folder (str or Path): Path to the experiment folder.
      trial_start (np.ndarray): Array of trial start indices (in samples, 0-based relative to experiment start).
      trial_end (np.ndarray): Array of trial end indices (in samples, 0-based relative to experiment start).
      rewarded_on_left (np.ndarray): Boolean array indicating rewarded stimulus on left side.
      sortout_folder (str or Path): Path to the session sortout folder (parent of shank0, shank1, ... folders).
      task_start (int): Sample index in the concatenated recording where this experiment starts.
                        Spikes are filtered to [task_start, task_end) and shifted by -task_start
                        to align with 0-based DIO trial timestamps. Default: 0.
      task_end (int or None): Sample index where this experiment ends. If None, uses the last spike index + 1.
      overwrite (bool): If False and the PKL file already exists, skip writing.
                        If True, overwrite any existing PKL file.

    Returns:
      pkl_file (Path): Path to the saved (or existing) PKL file.
    """
    
    # Parse session info (animal_id, session_id, folder_name)
    animal_id, session_id, folder_name = parse_session_info(rec_folder)

    # Convert to numpy arrays if not already
    trial_start = np.array(trial_start).ravel()
    trial_end = np.array(trial_end).ravel()
    rewarded_on_left = np.array([t['rewardedOnLeft'] for t in trial_params], dtype=bool)

    # Get number of trials
    n_trials = len(trial_start)
    print(f"Number of trials: {n_trials}")
    print(f"Trial start shape: {trial_start.shape}")
    print(f"Trial end shape: {trial_end.shape}")
    print(f"Rewarded on left shape: {rewarded_on_left.shape}")

    # Verify array lengths match
    assert len(trial_start) == len(trial_end) == len(rewarded_on_left), \
        "trial_start, trial_end, and rewarded_on_left must have the same length"

    # Unique stimulus conditions
    unique_conditions = np.unique(rewarded_on_left)
    n_conditions = len(unique_conditions)
    print(f"Unique conditions (rewarded_on_left): {unique_conditions}")

    # Compute the number of repeats per condition
    n_left = np.sum(rewarded_on_left)
    n_right = np.sum(~rewarded_on_left)
    print(f"Trials with rewarded on left: {n_left}")
    print(f"Trials with rewarded on right: {n_right}")
    
    # Define time windows (relative to trial start/end)
    pre_trial_window = 2.0    # 2s before trial start
    post_trial_window = 2.0   # 2s after trial end
    
    # Construct session folder for sorting results
    session_folder = Path(sortout_folder)
    
    # Check if the output file already exists
    pkl_file = session_folder / f'task_spikes_trial_{session_folder.name}.pkl'
    if pkl_file.exists() and not overwrite:
        print(f"File {pkl_file} already exists and overwrite=False. Skipping computation and returning existing file.")
        return pkl_file
    
    all_units_data = []

    curated_analyzer_folder = session_folder / 'curated_analyzer'
    if not curated_analyzer_folder.exists():
        raise FileNotFoundError(f"No curated_analyzer found in {session_folder}")

    sorting_analyzer = load_sorting_analyzer(curated_analyzer_folder)
    sorting = sorting_analyzer.sorting
    print(f"Loaded from curated_analyzer: {curated_analyzer_folder}")

    fs = sorting.sampling_frequency
    print(f"Sampling frequency: {fs} Hz")

    unit_ids = sorting.unit_ids

    # Read shank (group) and quality (unit_label) directly from sorting properties
    group_prop = sorting.get_property('group')
    label_prop = sorting.get_property('unit_label')
    group_map = {uid: int(g) for uid, g in zip(unit_ids, group_prop)} if group_prop is not None else {}
    label_map = {uid: str(l) for uid, l in zip(unit_ids, label_prop)} if label_prop is not None else {}

    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id)

        # Align spike train from concatenated space to experiment-local (0-based) space
        task_end_eff = task_end if task_end is not None else int(spike_train[-1]) + 1
        task_mask = (spike_train >= task_start) & (spike_train < task_end_eff)
        spike_train = spike_train[task_mask] - task_start

        shank = group_map.get(unit_id, None)
        quality = label_map.get(unit_id, 'unknown')
        unit_data = {
            'unit_id': unit_id,
            'shank': shank,
            'quality': quality,
            'sorting_folder': str(curated_analyzer_folder),
            'sampling_rate': fs,
            'trials': []
        }

        for trial_idx in range(n_trials):
            t_start = trial_start[trial_idx]
            t_end = trial_end[trial_idx]

            pre_start_idx = t_start - int(pre_trial_window * fs)
            post_end_idx = t_end + int(post_trial_window * fs)

            trial_spike_mask = (spike_train >= pre_start_idx) & (spike_train < post_end_idx)
            trial_spikes = spike_train[trial_spike_mask]

            relative_spike_times = (trial_spikes - t_start) / fs

            trial_rewarded_left = rewarded_on_left[trial_idx]
            condition_idx = 1 if trial_rewarded_left else 0
            repeat_idx = sum(1 for prev in range(trial_idx) if rewarded_on_left[prev] == trial_rewarded_left)
            trial_duration = (t_end - t_start) / fs
            tp = trial_params[trial_idx]

            trial_info = {
                'trial_number': trial_idx,
                'trial_start_time': t_start,
                'trial_end_time': t_end,
                'trial_start_time_sec': t_start / fs,
                'trial_end_time_sec': t_end / fs,
                'trial_duration': trial_duration,
                'rewarded_on_left': bool(trial_rewarded_left),
                'condition_idx': condition_idx,
                'repeat_idx': repeat_idx,
                'left_orientation': tp['leftOrientation'],
                'right_orientation': tp['rightOrientation'],
                'left_spatial_freq': tp['leftSpatialFreq'],
                'right_spatial_freq': tp['rightSpatialFreq'],
                'left_temporal_freq': tp['leftTemporalFreq'],
                'right_temporal_freq': tp['rightTemporalFreq'],
                'left_contrast': tp['leftContrast'],
                'right_contrast': tp['rightContrast'],
                'rewarded_orientation': tp['rewardedOrientation'],
                'non_rewarded_orientation': tp['nonRewardedOrientation'],
                'choice': tp['choice'],
                'correct': tp['correct'],
                'rewarded': tp['rewarded'],
                'stimulus_onset_time': tp['stimulusOnsetTime'],
                'choice_time': tp['choiceTime'],
                'choice_latency': tp['choiceLatency'],
                'spike_times': relative_spike_times,
                'pre_trial_spikes': relative_spike_times[relative_spike_times < 0],
                'during_trial_spikes': relative_spike_times[(relative_spike_times >= 0) &
                                                            (relative_spike_times < trial_duration)],
                'post_trial_spikes': relative_spike_times[relative_spike_times >= trial_duration],
                'pre_trial_count': np.sum(relative_spike_times < 0),
                'during_trial_count': np.sum((relative_spike_times >= 0) &
                                             (relative_spike_times < trial_duration)),
                'post_trial_count': np.sum(relative_spike_times >= trial_duration),
                'firing_rate_pre': np.sum(relative_spike_times < 0) / pre_trial_window,
                'firing_rate_during': np.sum((relative_spike_times >= 0) &
                                             (relative_spike_times < trial_duration)) / trial_duration,
                'firing_rate_post': np.sum(relative_spike_times >= trial_duration) / post_trial_window,
            }

            unit_data['trials'].append(trial_info)

        all_units_data.append(unit_data)
    
    print(f"Processed {len(all_units_data)} units across {n_trials} trials")
    
    # Save the data to a PKL file using the unified schema
    save_behavior_trial_to_pkl(
        animal_id=animal_id,
        session_id=session_id,
        rec_folder=rec_folder,
        fs=fs,
        trial_start=trial_start,
        trial_end=trial_end,
        rewarded_on_left=rewarded_on_left,
        trial_params=trial_params,
        unique_conditions=unique_conditions,
        pre_trial_window=pre_trial_window,
        post_trial_window=post_trial_window,
        all_units_data=all_units_data,
        output_path=pkl_file,
    )

    return pkl_file


def save_behavior_trial_to_pkl(
    *,
    animal_id,
    session_id,
    rec_folder,
    fs,
    trial_start,
    trial_end,
    rewarded_on_left,
    trial_params,
    unique_conditions,
    pre_trial_window,
    post_trial_window,
    all_units_data,
    output_path,
):
    """
    Save behavior trial responses in the SAME schema as the embedding extractor.
    This ensures compatibility with downstream analysis code.
    """

    # Create trial windows
    trial_windows = [(int(start), int(end)) for start, end in zip(trial_start, trial_end)]

    # Build all trial parameters list
    all_trial_parameters = []
    for trial_idx in range(len(trial_start)):
        tp = trial_params[trial_idx]
        all_trial_parameters.append({
            'trial_index': trial_idx,
            'rewarded_on_left': bool(rewarded_on_left[trial_idx]),
            'trial_start': int(trial_start[trial_idx]),
            'trial_end': int(trial_end[trial_idx]),
            'trial_duration': float((trial_end[trial_idx] - trial_start[trial_idx]) / fs),
            'left_orientation': tp['leftOrientation'],
            'right_orientation': tp['rightOrientation'],
            'left_spatial_freq': tp['leftSpatialFreq'],
            'right_spatial_freq': tp['rightSpatialFreq'],
            'left_temporal_freq': tp['leftTemporalFreq'],
            'right_temporal_freq': tp['rightTemporalFreq'],
            'left_contrast': tp['leftContrast'],
            'right_contrast': tp['rightContrast'],
            'rewarded_orientation': tp['rewardedOrientation'],
            'non_rewarded_orientation': tp['nonRewardedOrientation'],
            'choice': tp['choice'],
            'correct': tp['correct'],
            'rewarded': tp['rewarded'],
            'stimulus_onset_time': tp['stimulusOnsetTime'],
            'choice_time': tp['choiceTime'],
            'choice_latency': tp['choiceLatency'],
            'position_x': np.asarray(tp['position_x']).tolist(),
            'position_y': np.asarray(tp['position_y']).tolist(),
            'position_time': np.asarray(tp['position_time']).tolist(),
            'heading': np.asarray(tp.get('heading', [])).tolist(),
            'head_angle': np.asarray(tp.get('head_angle', [])).tolist(),
            'dlc_signal': np.asarray(tp.get('dlc_signal', [])).tolist(),
        })

    # Calculate average trial duration
    avg_trial_duration = np.mean((trial_end - trial_start) / fs)

    # Initialize the unified data structure
    neural_data = {
        'metadata': {
            'animal_id': animal_id,
            'session_id': session_id,
            'recording_folder': str(rec_folder),
            'task_file': None,  # Not applicable for this experiment
            'extraction_date': datetime.now().isoformat(),
            'n_trials': len(trial_start),
            'experiment_type': 'behavior_trial',
            'sampling_frequency': fs,
        },
        'experiment_parameters': {
            'stimulus_duration': None,  # Variable per trial
            'iti_duration': None,  # Not extracted
            'trial_duration': float(avg_trial_duration),
            'total_trials': len(trial_start),
            'n_conditions': len(unique_conditions),
            'n_left_trials': int(np.sum(rewarded_on_left)),
            'n_right_trials': int(np.sum(~rewarded_on_left)),
        },
        'trial_info': {
            'rewarded_on_left': rewarded_on_left.tolist(),
            'unique_conditions': unique_conditions.tolist(),
            'trial_windows': trial_windows,
            'all_trial_parameters': all_trial_parameters,
            'trial_durations': [(trial_end[i] - trial_start[i]) / fs 
                               for i in range(len(trial_start))],
        },
        'spike_data': {},
        'unit_info': {},
        'extraction_params': {
            'window_pre': pre_trial_window,
            'window_post': post_trial_window,
            'total_units': len(all_units_data),
        }
    }
    
    # Populate spike_data and unit_info in the unified format
    unit_counter = 0
    
    for unit in all_units_data:
        # Create unique unit identifier matching the embedding extractor format
        shank = unit.get('shank')
        unique_unit_id = f"shank{shank}_unit{unit['unit_id']}" if shank is not None else f"unit{unit['unit_id']}"

        # Store unit metadata
        neural_data['unit_info'][unique_unit_id] = {
            'original_unit_id': int(unit['unit_id']),
            'shank': shank,
            'quality': unit.get('quality', 'unknown'),
            'sorting_folder': unit.get('sorting_folder', ''),
            'n_spikes_total': sum(len(t['spike_times']) for t in unit['trials']),
            'unit_index': unit_counter,
        }
        
        # Store spike data for each trial
        trials_out = []
        for t in unit['trials']:
            trial_data = {
                'trial_index': t['trial_number'],
                'rewarded_on_left': t['rewarded_on_left'],
                'condition_idx': t['condition_idx'],
                'spike_times': t['spike_times'].tolist(),
                'spike_count': len(t['spike_times']),
                'trial_start': t['trial_start_time'],
                'trial_end': t['trial_end_time'],
                'trial_duration': t['trial_duration'],
                'left_orientation': t.get('left_orientation'),
                'right_orientation': t.get('right_orientation'),
                'left_spatial_freq': t.get('left_spatial_freq'),
                'right_spatial_freq': t.get('right_spatial_freq'),
                'left_temporal_freq': t.get('left_temporal_freq'),
                'right_temporal_freq': t.get('right_temporal_freq'),
                'left_contrast': t.get('left_contrast'),
                'right_contrast': t.get('right_contrast'),
                'rewarded_orientation': t.get('rewarded_orientation'),
                'non_rewarded_orientation': t.get('non_rewarded_orientation'),
                'choice': t.get('choice'),
                'correct': t.get('correct'),
                'rewarded': t.get('rewarded'),
                'stimulus_onset_time': t.get('stimulus_onset_time'),
                'choice_time': t.get('choice_time'),
                'choice_latency': t.get('choice_latency'),
                'pre_trial_count': t.get('pre_trial_count'),
                'during_trial_count': t.get('during_trial_count'),
                'post_trial_count': t.get('post_trial_count'),
                'firing_rate_pre': t.get('firing_rate_pre'),
                'firing_rate_during': t.get('firing_rate_during'),
                'firing_rate_post': t.get('firing_rate_post'),
            }
            trials_out.append(trial_data)
        
        neural_data['spike_data'][unique_unit_id] = trials_out
        unit_counter += 1
    
    print(f"Saving unified PKL format → {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(neural_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return output_path


if __name__ == '__main__':
    from DiscriminationTask.grating.task_params import rec_folder, task_file, sortout_folder, task_start, task_end, din_channel

    # Load DIO trial timestamps
    dio_folders = DIO.get_dio_folders(rec_folder)
    dio_folders = sorted(dio_folders, key=lambda x: x.name)
    trial_pd_time, trial_pd_state = DIO.concatenate_din_data(dio_folders, din_channel)
    trial_pd_time = trial_pd_time.ravel() - trial_pd_time.ravel()[0]
    trial_pd_state = trial_pd_state.ravel()
    trial_start = trial_pd_time[np.where(trial_pd_state == 1)[0]]
    trial_end   = trial_pd_time[np.where(trial_pd_state == 0)[0][1:]]

    # Load behavioral trial parameters from task JSON
    trial_params = get_trial_params(task_file)

    n_DIO    = len(trial_start)
    n_trials = len(trial_params)
    if n_DIO != n_trials:
        raise ValueError(f"DIO trials ({n_DIO}) != behavior trials ({n_trials}). Please check your data.")
    print(f"DIO and behavior trial counts match: {n_trials} trials.")

    # Process and save
    pkl_path = process_behavior_trial_responses(
        rec_folder=rec_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        trial_params=trial_params,
        sortout_folder=sortout_folder,
        task_start=task_start,
        task_end=task_end,
        overwrite=True,
    )

    print(f"\nData saved to: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print("\nData structure verification:")
    print(f"- Total units: {len(data['spike_data'])}")
    print(f"- Total trials: {data['metadata']['n_trials']}")
    print(f"- Left trials: {data['experiment_parameters']['n_left_trials']}")
    print(f"- Right trials: {data['experiment_parameters']['n_right_trials']}")