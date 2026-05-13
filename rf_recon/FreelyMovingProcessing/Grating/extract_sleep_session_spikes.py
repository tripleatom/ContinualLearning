import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(code_dir))

import numpy as np
import pickle
from datetime import datetime
from spikeinterface import load_sorting_analyzer


def extract_sleep_session_spikes(
    sortout_folder,
    sleep_start,
    sleep_end,
    sleep_id,
    overwrite=True,
):
    """
    Extract a continuous spike train per unit covering a sleep / rest window.

    Spike times are expressed in seconds relative to sleep_start (t=0), matching
    the zero-aligned convention used by extract_passive_session_spikes.py and
    extract_session_spikes.py. The resulting pickle is consumable by
    apply_merged_decoder_to_sleep.py with sleep_start_sec=0 and
    sleep_end_sec=window_duration_sec.

    Parameters
    ----------
    sortout_folder : str or Path
        Folder containing curated_analyzer/ (same one used for task/passive).
    sleep_start    : int
        Sample index in the sorter's concatenated space where the sleep window starts.
    sleep_end      : int
        Sample index in the sorter's concatenated space where the sleep window ends.
    sleep_id       : str
        Identifier used in the output filename (e.g. "260313" or "260313_sleep1").
    overwrite      : bool
        Overwrite existing output file.

    Returns
    -------
    pkl_file : Path
    """
    session_folder = Path(sortout_folder)
    curated_analyzer_folder = session_folder / "curated_analyzer"
    if not curated_analyzer_folder.exists():
        raise FileNotFoundError(f"No curated_analyzer found in {session_folder}")

    sleep_start = int(sleep_start)
    sleep_end = int(sleep_end)
    if sleep_end <= sleep_start:
        raise ValueError("sleep_end must be greater than sleep_start.")

    sorting_analyzer = load_sorting_analyzer(curated_analyzer_folder)
    sorting = sorting_analyzer.sorting
    fs = sorting.sampling_frequency
    window_duration_sec = (sleep_end - sleep_start) / fs
    print(f"Sampling frequency : {fs} Hz")
    print(f"Sleep window       : samples {sleep_start} – {sleep_end} "
          f"({window_duration_sec:.1f} s)")

    pkl_file = session_folder / f"sleep_spikes_{sleep_id}.pkl"
    if pkl_file.exists() and not overwrite:
        print(f"{pkl_file} exists and overwrite=False – skipping.")
        return pkl_file

    unit_ids = sorting.unit_ids
    group_prop = sorting.get_property("group")
    label_prop = sorting.get_property("unit_label")
    group_map = {uid: int(g) for uid, g in zip(unit_ids, group_prop)} if group_prop is not None else {}
    label_map = {uid: str(l) for uid, l in zip(unit_ids, label_prop)} if label_prop is not None else {}

    spike_data = {}
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id)
        mask = (spike_train >= sleep_start) & (spike_train < sleep_end)
        win_spikes = spike_train[mask]
        spike_times_sec = (win_spikes - sleep_start) / fs

        shank = group_map.get(unit_id, None)
        quality = label_map.get(unit_id, "unknown")
        uid_str = f"shank{shank}_unit{unit_id}" if shank is not None else f"unit{unit_id}"

        spike_data[uid_str] = {
            "spike_times_sec": spike_times_sec,
            "n_spikes": len(spike_times_sec),
            "unit_id": int(unit_id),
            "shank": shank,
            "quality": quality,
        }

    print(f"Extracted {len(spike_data)} units")

    output = {
        "metadata": {
            "sleep_id": sleep_id,
            "extraction_date": datetime.now().isoformat(),
            "sampling_frequency": fs,
            "n_units": len(spike_data),
            "sortout_folder": str(session_folder),
        },
        "window": {
            "sleep_start_sample": sleep_start,
            "sleep_end_sample": sleep_end,
            "window_duration_sec": window_duration_sec,
        },
        "spike_data": spike_data,
    }

    print(f"Saving → {pkl_file}")
    with open(pkl_file, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    return pkl_file


if __name__ == "__main__":
    from grating_config import EXPERIMENT_DATE, SORTOUT_FOLDER

    # =========================================================
    # Sample indices in the sorter's concatenated space for each sleep block.
    # Inspect curated_analyzer to find where each sleep period lives.
    SLEEP_BLOCKS = [
        # (sleep_id_suffix, start_sample, end_sample)
        ("pre",  187761701, 243144525),
        ("post", 334584075, 545044198),
    ]
    # =========================================================

    for suffix, start, end in SLEEP_BLOCKS:
        if end <= start:
            print(f"\n[{suffix}] Skipped — start/end not set (start={start}, end={end}).")
            continue

        sleep_id = f"{EXPERIMENT_DATE}_sleep_{suffix}"
        print(f"\n=== Extracting {sleep_id} ===")
        pkl_path = extract_sleep_session_spikes(
            sortout_folder=SORTOUT_FOLDER,
            sleep_start=start,
            sleep_end=end,
            sleep_id=sleep_id,
            overwrite=True,
        )

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        w = data["window"]
        m = data["metadata"]
        print(f"Window   : {w['window_duration_sec']:.1f} s "
              f"(samples {w['sleep_start_sample']} – {w['sleep_end_sample']})")
        print(f"Units    : {m['n_units']}")
        print(f"In apply_merged_decoder_to_sleep.py, set:")
        print(f"  sleep_pkl       = r\"{pkl_path}\"")
        print(f"  sleep_start_sec = 0.0")
        print(f"  sleep_end_sec   = {w['window_duration_sec']:.3f}")
