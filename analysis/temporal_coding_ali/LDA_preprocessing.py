import re
import numpy as np


def get_good_units(unit_info):
    """
    Filter units by quality. Returns only units labeled 'good' or 'mua'.
    
    Parameters
    ----------
    unit_info : dict
        Dictionary of unit metadata, e.g. data["unit_info"].
        Keys = unit IDs, values = dict with fields like 'quality', 'shank', etc.
    
    Returns
    -------
    good_units : list of tuples
        List of (unit_id, metadata) for good or MUA units.
    good_units_by_shank : dict
        Dictionary grouping unit IDs by shank, e.g. {"0": [unit1, unit2], "1": [...] }.
    """
    good_units = []
    good_units_by_shank = {}

    for unit_id, meta in unit_info.items():
        q = str(meta.get("quality", "")).lower()
        if q in ("good", "mua", "unsorted"):
            good_units.append((unit_id, meta))
            shank = str(meta.get("shank", "unknown"))
            if shank not in good_units_by_shank:
                good_units_by_shank[shank] = []
            good_units_by_shank[shank].append(unit_id)

    return good_units, good_units_by_shank



def extract_orientation_spikes(data, good_units, clean_ids=False):
    """
    From `data` keep only trials for units in `good_units`, and within each trial
    keep only 'orientation' and 'spike_times'.

    Parameters
    ----------
    data : dict
        Must contain data["spike_data"] : { unit_id : [ {orientation, spike_times, ...}, ... ] }
    good_units : list of (unit_id, meta)
        Output of your get_good_units(unit_info).
    clean_ids : bool (default False)
        If True, unit keys become 'shankX_unitY' when pattern is present.

    Returns
    -------
    out : dict
        { unit_id_or_shankX_unitY : [ { "orientation": ..., "spike_times": [...] }, ... ] }
    """
    spike_data = data["spike_data"]
    out = {}

    for unit_id, _ in good_units:
        if unit_id not in spike_data:
            continue

        key = unit_id
        if clean_ids:
            m = re.search(r"(shank\d+).*?(unit\d+)", unit_id)
            if m:
                key = f"{m.group(1)}_{m.group(2)}"

        trials = spike_data[unit_id]
        out[key] = [
            {"orientation": t.get("orientation", None),
             "spike_times": t.get("spike_times", [])}
            for t in trials
        ]

    return out


def add_firing_rate(cleaned_spike_data, window_start, window_end, key_name="firing_rate"):
    """
    Add firing rates to each trial in cleaned_spike_data for a given time window.

    Parameters
    ----------
    cleaned_spike_data : dict
        { unit_id : [ { "orientation": ..., "spike_times": [...] }, ... ] }
    window_start : float
        Start time of window (seconds).
    window_end : float
        End time of window (seconds).
    key_name : str, default "firing_rate"
        Key under which to store the computed rate in each trial dict.

    Returns
    -------
    cleaned_spike_data : dict
        Updated dictionary with firing rates added to each trial.
    """
    window_dur = window_end - window_start

    for uid, trials in cleaned_spike_data.items():
        for trial in trials:
            spikes = np.asarray(trial.get("spike_times", []), float)
            count = np.sum((spikes >= window_start) & (spikes <= window_end))
            rate = count / window_dur
            trial[key_name] = rate

    return cleaned_spike_data

def remove_zero_firing_trials(cleaned_spike_data, key_name="firing_rate", verbose=True):
    """
    Remove trials with zero firing rate from each unit in cleaned_spike_data.

    Parameters
    ----------
    cleaned_spike_data : dict
        { unit_id : [ { "orientation": ..., "spike_times": [...], "firing_rate": ... }, ... ] }
    key_name : str, default "firing_rate"
        The key in each trial dict that stores the firing rate.
    verbose : bool, default True
        If True, print how many trials were removed per unit and in total.

    Returns
    -------
    cleaned_spike_data : dict
        Updated dictionary with zero-firing trials removed.
    """
    total_removed = 0
    removed_per_unit = {}

    for uid, trials in cleaned_spike_data.items():
        before = len(trials)
        # keep only trials with nonzero firing rate
        trials_filtered = [t for t in trials if t.get(key_name, 0) > 0]
        after = len(trials_filtered)
        removed = before - after
        if removed > 0:
            removed_per_unit[uid] = removed
            total_removed += removed
        cleaned_spike_data[uid] = trials_filtered

    if verbose:
        print("Trials removed per unit:")
        for uid, removed in removed_per_unit.items():
            print(f"  {uid}: {removed} removed")
        print(f"Total trials removed: {total_removed}")

    return cleaned_spike_data

def report_zero_firing_trials(cleaned_spike_data, key_name="firing_rate"):
    """
    Count how many trials per unit have zero firing rate, without removing them.

    Parameters
    ----------
    cleaned_spike_data : dict
        { unit_id : [ { "orientation": ..., "spike_times": [...], "firing_rate": ... }, ... ] }
    key_name : str, default "firing_rate"
        The key in each trial dict that stores the firing rate.

    Returns
    -------
    zero_counts : dict
        { unit_id : number of zero-rate trials }
    total_zeros : int
        Total number of zero-rate trials across all units.
    """
    zero_counts = {}
    total_zeros = 0

    for uid, trials in cleaned_spike_data.items():
        zero_count = sum(1 for t in trials if t.get(key_name, 0) == 0)
        if zero_count > 0:
            zero_counts[uid] = zero_count
            total_zeros += zero_count

    print("Zero firing-rate trials per unit:")
    for uid, count in zero_counts.items():
        print(f"  {uid}: {count} zeros out of {len(cleaned_spike_data[uid])} trials")
    print(f"Total zero firing-rate trials: {total_zeros}")

    return zero_counts, total_zeros
