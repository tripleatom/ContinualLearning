import os
import numpy as np
import scipy.io
import h5py
import pandas as pd
from pathlib import Path
from spikeinterface.extractors import PhySortingExtractor
from rec2nwb.preproc_func import parse_session_info
from rf_recon.rf_func import dereference

def load_rising_edges_any(peaks_path):
    """
    Returns (rising_edges_array, units) where units is 'samples' or 'seconds'.
    Supports:
      - .mat with variable 'locs' (samples)
      - .npz with 'rising_times' (preferred) or 'locs'
    """
    p = Path(peaks_path)
    ext = p.suffix.lower()

    if ext == ".mat":
        md = scipy.io.loadmat(peaks_path, struct_as_record=False, squeeze_me=True)
        if "locs" not in md:
            raise KeyError("MAT file missing variable 'locs'")
        arr = np.asarray(md["locs"], dtype=float)
        return arr, "samples"

    if ext == ".npz":
        d = np.load(peaks_path)
        key = "rising_times" if "rising_times" in d.files else ("locs" if "locs" in d.files else None)
        if key is None:
            raise KeyError("NPZ must contain 'rising_times' or 'locs'")
        arr = np.asarray(d[key], dtype=float)
        # Heuristic: if spacing is small (<<10s), assume seconds; otherwise samples.
        diffs = np.diff(arr) if arr.size > 1 else np.array([np.inf])
        units = "seconds" if np.nanmedian(diffs) < 10 else "samples"
        return arr, units

    raise ValueError(f"Unsupported peaks file type: {ext}")

def process_static_grating_responses(rec_folder, stimdata_file, peaks_file, overwrite=True):
    """
    Process static grating responses from an experiment folder.
    - Loads stimulus metadata from CSV or MAT/HDF5
    - Loads photodiode rising edges from .mat or .npz
    - Extracts per-trial spike times (pre/post) for each unit
    - Saves detailed trial-by-trial data into an NPZ
    """
    # --- peaks (photodiode) ---
    rising_edges_raw, peaks_units = load_rising_edges_any(peaks_file)

    # --- session info ---
    rec_folder = Path(rec_folder)
    stimdata_file = Path(stimdata_file)
    animal_id, session_id, folder_name = parse_session_info(rec_folder)
    ishs = ['0', '1', '2', '3']

    # --- stimulus metadata: CSV or HDF5/MAT ---
    if stimdata_file.suffix.lower() == ".csv":
        print(f"Loading stimulus metadata from CSV: {stimdata_file}")
        df = pd.read_csv(stimdata_file)

        if "left_ori" in df.columns:
            stim_orientation = df["left_ori"].to_numpy(float)
        elif "orientation" in df.columns:
            stim_orientation = df["orientation"].to_numpy(float)
        else:
            raise ValueError("CSV must contain 'left_ori' or 'orientation' for static gratings.")

        if "left_phase_deg" in df.columns:
            stim_phase = df["left_phase_deg"].to_numpy(float)
        elif "phase" in df.columns:
            stim_phase = df["phase"].to_numpy(float)
        else:
            stim_phase = np.zeros_like(stim_orientation, float)

        if "left_sf_cpd" in df.columns:
            stim_spatialFreq = df["left_sf_cpd"].to_numpy(float)
        elif "spatialFreq" in df.columns:
            stim_spatialFreq = df["spatialFreq"].to_numpy(float)
        else:
            stim_spatialFreq = np.zeros_like(stim_orientation, float)

        if "t_trial_s" in df.columns:
            t_trial = float(df["t_trial_s"].iloc[0])
        elif {"stim_on_s", "stim_off_s"}.issubset(df.columns):
            t_trial = float(np.median(df["stim_off_s"].to_numpy() - df["stim_on_s"].to_numpy()))
        else:
            t_trial = 1.0
            print("[WARN] CSV missing t_trial_s and stim_on/off_s; using t_trial=1.0s")
    else:
        print(f"Loading stimulus metadata from MAT/HDF5: {stimdata_file}")
        with h5py.File(stimdata_file, 'r') as f:
            patternParams_group = f['Stimdata']['patternParams']
            orientation_data = patternParams_group['orientation'][()]
            stim_orientation = np.array([dereference(ref, f) for ref in orientation_data]).flatten().astype(float)

            phase_data = patternParams_group['phase'][()]
            stim_phase = np.array([dereference(ref, f) for ref in phase_data]).flatten().astype(float)

            spatialFreq_data = patternParams_group['spatialFreq'][()]
            stim_spatialFreq = np.array([dereference(ref, f) for ref in spatialFreq_data]).flatten().astype(float)

            t_trial = f['Stimdata']['t_trial'][()][0, 0]

    print("Orientation:", stim_orientation)
    print("Phase:", stim_phase)
    print("Spatial Frequency:", stim_spatialFreq)

    n_static_grating = int(stim_orientation.shape[0])
    print(f"Number of static grating stimuli: {n_static_grating}")
    print(f"Number of rising edges (raw): {len(rising_edges_raw)}")
    if len(rising_edges_raw) < n_static_grating:
        raise ValueError("Not enough rising edges to cover all static gratings.")

    # --- uniques & repeats ---
    unique_orientation = np.unique(stim_orientation)
    unique_phase = np.unique(stim_phase)
    unique_spatialFreq = np.unique(stim_spatialFreq)
    n_orientation = len(unique_orientation)
    n_phase = len(unique_phase)
    n_spatialFreq = len(unique_spatialFreq)
    denom = max(1, n_orientation * n_phase * n_spatialFreq)
    n_repeats = n_static_grating // denom

    # --- windows (sec) ---
    pre_stim_window = 0.05
    post_stim_window = float(t_trial)

    all_units_data = []
    unit_info = []
    all_unit_qualities = []

    code_folder = Path(__file__).parent.parent.parent
    session_folder = code_folder / rf"sortout/{animal_id}/{animal_id}_{session_id}"
    npz_file = session_folder / 'static_grating_responses.npz'
    if npz_file.exists() and not overwrite:
        print(f"File {npz_file} exists and overwrite=False. Returning existing file.")
        return npz_file

    # We’ll convert rising edges to samples once we know fs (do it once).
    rising_edges_samples = None
    static_grating_rising_edges = None

    for ish in ishs:
        print(f'Processing {animal_id}/{session_id}/{ish}')
        shank_folder = session_folder / f'shank{ish}'
        sorting_results_folders = []
        for root, dirs, files in os.walk(shank_folder):
            for dir_name in dirs:
                if dir_name.startswith('sorting_results_'):
                    sorting_results_folders.append(os.path.join(root, dir_name))

        for sorting_results_folder in sorting_results_folders:
            phy_folder = Path(sorting_results_folder) / 'phy'
            out_fig_folder = Path(sorting_results_folder) / 'static_grating'
            out_fig_folder.mkdir(parents=True, exist_ok=True)

            sorting = PhySortingExtractor(phy_folder)
            unit_ids = sorting.unit_ids
            unit_qualities_this_sort = sorting.get_property('quality')
            fs = float(sorting.sampling_frequency)

            # Convert rising edges to samples once per session (after fs is known)
            if rising_edges_samples is None:
                if peaks_units == "seconds":
                    rising_edges_samples = (np.asarray(rising_edges_raw, float) * fs).astype(np.int64)
                else:
                    rising_edges_samples = np.asarray(rising_edges_raw, float).astype(np.int64)

                if len(rising_edges_samples) < n_static_grating:
                    raise ValueError("Not enough rising edges (after unit conversion) for all stimuli.")
                static_grating_rising_edges = rising_edges_samples[-n_static_grating:]

            for i, unit_id in enumerate(unit_ids):
                spike_train = sorting.get_unit_spike_train(unit_id)  # samples (int64)

                unit_data = {
                    'unit_id': unit_id,
                    'shank': ish,
                    'sampling_rate': fs,
                    'trials': []
                }

                for trial_idx, edge in enumerate(static_grating_rising_edges):
                    pre_start_time = edge - int(round(pre_stim_window * fs))
                    post_end_time  = edge + int(round(post_stim_window * fs))

                    mask = (spike_train >= pre_start_time) & (spike_train < post_end_time)
                    trial_spikes = spike_train[mask]
                    rel_spike_times = (trial_spikes - edge) / fs

                    trial_orientation = float(stim_orientation[trial_idx])
                    trial_phase = float(stim_phase[trial_idx])
                    trial_spatialFreq = float(stim_spatialFreq[trial_idx])

                    ori_idx = int(np.where(unique_orientation == trial_orientation)[0][0])
                    phase_idx = int(np.where(unique_phase == trial_phase)[0][0])
                    sf_idx = int(np.where(unique_spatialFreq == trial_spatialFreq)[0][0])

                    # repeat index for this condition
                    repeat_idx = int(np.sum(
                        (stim_orientation[:trial_idx] == trial_orientation) &
                        (stim_phase[:trial_idx] == trial_phase) &
                        (stim_spatialFreq[:trial_idx] == trial_spatialFreq)
                    ))

                    trial_info = {
                        'trial_number': trial_idx,
                        'stimulus_onset_time': int(edge),
                        'stimulus_onset_time_sec': float(edge / fs),
                        'orientation': trial_orientation,
                        'phase': trial_phase,
                        'spatial_frequency': trial_spatialFreq,
                        'orientation_idx': ori_idx,
                        'phase_idx': phase_idx,
                        'spatial_freq_idx': sf_idx,
                        'repeat_idx': repeat_idx,
                        'spike_times': rel_spike_times,  # seconds, relative to onset
                        'pre_stim_spikes': rel_spike_times[rel_spike_times < 0],
                        'post_stim_spikes': rel_spike_times[rel_spike_times >= 0],
                        'pre_stim_count': int(np.sum(rel_spike_times < 0)),
                        'post_stim_count': int(np.sum(rel_spike_times >= 0)),
                        'firing_rate_pre': float(np.sum(rel_spike_times < 0) / pre_stim_window),
                        'firing_rate_post': float(np.sum(rel_spike_times >= 0) / post_stim_window),
                    }

                    unit_data['trials'].append(trial_info)

                all_units_data.append(unit_data)
                unit_info.append((ish, unit_id))
                all_unit_qualities.append(unit_qualities_this_sort[i] if unit_qualities_this_sort is not None else None)

    print(f"Processed {len(all_units_data)} units across {len(static_grating_rising_edges)} trials")

    summary_stats = {
        'n_units': len(all_units_data),
        'n_trials': len(static_grating_rising_edges),
        'n_orientation': n_orientation,
        'n_phase': n_phase,
        'n_spatialFreq': n_spatialFreq,
        'n_repeats': n_repeats,
        'pre_stim_window': pre_stim_window,
        'post_stim_window': post_stim_window,
    }

    session_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {npz_file} (overwrite={overwrite})")
    np.savez(
        npz_file,
        unit_info=unit_info,
        unit_qualities=all_unit_qualities,
        stim_orientation=stim_orientation,
        stim_phase=stim_phase,
        stim_spatialFreq=stim_spatialFreq,
        unique_orientation=unique_orientation,
        unique_phase=unique_phase,
        unique_spatialFreq=unique_spatialFreq,
        static_grating_rising_edges=static_grating_rising_edges,
        units_data=all_units_data,
        summary_stats=summary_stats,
        pre_stim_window=pre_stim_window,
        post_stim_window=post_stim_window,
    )
    print("Data saved with detailed trial structure")
    return npz_file

def load_and_analyze_static_grating_data(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    units_data = data['units_data']
    summary_stats = data['summary_stats'].item()

    print("Summary Statistics:")
    for k, v in summary_stats.items():
        print(f"  {k}: {v}")

    print("\nFirst unit example:")
    first_unit = units_data[0]
    print(f"  Unit ID: {first_unit['unit_id']}")
    print(f"  Shank: {first_unit['shank']}")
    print(f"  Number of trials: {len(first_unit['trials'])}")

    first_trial = first_unit['trials'][0]
    print("\nFirst trial example:")
    for key, value in first_trial.items():
        if key in ['spike_times', 'pre_stim_spikes', 'post_stim_spikes']:
            print(f"  {key}: {len(value)} spikes")
        else:
            print(f"  {key}: {value}")
    return data

if __name__ == '__main__':
    rec_folder = Path(input("Please enter the full path to the recording folder: ").strip().strip('"'))
    stimdata_file = Path(input("Please enter the full path to the stimulus data CSV/.mat/.h5 file: ").strip().strip('"'))
    peaks_file = Path(input("Please enter the full path to the peaks_xx.mat or *_DIO_cleaned.npz file: ").strip().strip('"'))
    npz_path = process_static_grating_responses(rec_folder, stimdata_file, peaks_file, overwrite=True)
    print(f"Data saved to: {npz_path}")
