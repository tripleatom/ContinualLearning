r"""Sleep spikes and NREM UP-state windows for the tuning-coupling analysis.

Everything here answers one question: *when* was the cortex in an NREM UP
state, on the same clock as the sorted spikes?

``load_sleep_spikes``
    Spike times per unit from a ``sleep_spikes_*.pkl`` (seconds from the sleep
    block's own start).

UP windows
    From the channel-level MUA ON/OFF detection (``sleep/mua_onoff/detect_off_states.py``)
    via ``sleep_states.load_mua_on_intervals`` (reactivation/sleep_apply).  MUA is used,
    not the sorted units, because the sorted units are the ones whose synchrony is being
    measured -- defining UP states from them would build their co-firing into the windows.

NREM bouts
    Loaded with ``sleep_states.load_nrem_intervals`` (reactivation/sleep_apply); UP
    windows are intersected with them so only NREM UP states are analysed.

``check_up_alignment``
    Verifies the clock mapping.  MUA windows are stored in whole-day samples and
    mapped as ``t_sleep = (daysample - sleep_start_sample) / fs``.  If that is
    right, sorted units fire much more inside the UP windows than in the DOWN
    time between them (within NREM); a ratio near 1 means the clocks disagree
    and the run must stop.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Loading spikes
# --------------------------------------------------------------------------- #

def load_sleep_spikes(sleep_pkl, unit_ids=None, verbose=True):
    """
    Read a ``sleep_spikes_*.pkl`` and return spike times per unit.

    Returns (spikes, meta) where spikes is {unit_id: np.ndarray of seconds from
    block start} and meta carries duration, sampling rate and the block's start
    sample in the sorter's space (needed to map MUA windows onto this clock).
    """
    sleep_pkl = Path(sleep_pkl)
    with sleep_pkl.open('rb') as f:
        data = pickle.load(f)

    spike_data = data['spike_data']
    window = data.get('window', {})
    meta = {
        'path': str(sleep_pkl),
        'sleep_id': data.get('metadata', {}).get('sleep_id'),
        'fs': float(data.get('metadata', {}).get('sampling_frequency', 30000.0)),
        'start_sample': window.get('sleep_start_sample'),
        'end_sample': window.get('sleep_end_sample'),
        'duration_sec': float(window.get('window_duration_sec', 0.0)),
    }

    wanted = set(spike_data) if unit_ids is None else set(unit_ids)
    spikes = {}
    for uid in sorted(spike_data):
        if uid not in wanted:
            continue
        times = np.asarray(spike_data[uid].get('spike_times_sec', []), dtype=float)
        spikes[uid] = np.sort(times[np.isfinite(times)])

    if meta['duration_sec'] <= 0 and spikes:
        meta['duration_sec'] = float(max(t[-1] if t.size else 0.0 for t in spikes.values()))

    if verbose:
        missing = wanted - set(spikes)
        print(f"Sleep spikes [{meta['sleep_id']}]: {len(spikes)} units, "
              f"{meta['duration_sec'] / 60:.1f} min")
        if missing:
            print(f"  {len(missing)} requested units absent from this sleep pkl")
    return spikes, meta


# --------------------------------------------------------------------------- #
# UP-state windows
# --------------------------------------------------------------------------- #

def _intersect_windows(a, b):
    """Intersection of two sorted window sets."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 2))
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start = max(a[i, 0], b[j, 0])
        end = min(a[i, 1], b[j, 1])
        if end > start:
            out.append((start, end))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(out) if out else np.zeros((0, 2))


def _clip_and_filter(windows, duration_sec, min_duration_sec, max_duration_sec):
    if windows.size == 0:
        return windows
    windows = windows[(windows[:, 1] > 0) & (windows[:, 0] < duration_sec)]
    if windows.size == 0:
        return np.zeros((0, 2))
    windows[:, 0] = np.clip(windows[:, 0], 0, duration_sec)
    windows[:, 1] = np.clip(windows[:, 1], 0, duration_sec)
    lengths = windows[:, 1] - windows[:, 0]
    keep = (lengths >= min_duration_sec)
    if max_duration_sec is not None:
        keep &= (lengths <= max_duration_sec)
    return windows[keep]


def check_up_alignment(spikes, windows, duration_sec, min_ratio=1.5, verbose=True,
                       universe_windows=None):
    """
    Verify that the UP windows really sit where the sorted spikes are.

    Returns (ratio, report).  ``ratio`` is the population firing rate inside the
    windows divided by the rate in the gaps between them.  UP states are by
    definition the high-activity phase, so a correct time base gives a clearly
    larger-than-1 ratio (typically 2-10x); a ratio near or below 1 means either
    the MUA windows were mapped onto the wrong clock, or "outside the windows"
    is dominated by something with its own high activity that the comparison
    should not include -- e.g. waking movement in an unrestricted presleep/
    postsleep block, where an animal is not asleep for most of the epoch.

    ``universe_windows`` restricts BOTH the "inside" and "outside" side of the
    ratio to a given time base (e.g. scored NREM) instead of the whole
    recording -- pass the NREM windows here (already used to restrict `windows`
    itself) so "outside the UP windows" means "DOWN time within NREM", not
    "any non-UP time including wake". Omit to use the whole ``duration_sec``.
    """
    if windows.size == 0:
        return np.nan, {'reason': 'no windows'}

    if universe_windows is not None and universe_windows.size:
        total_universe = float(np.sum(universe_windows[:, 1] - universe_windows[:, 0]))
        universe_label = 'scored NREM'
    else:
        universe_windows = np.array([[0.0, float(duration_sec)]])
        total_universe = float(duration_sec)
        universe_label = 'whole block'

    total_in = float(np.sum(windows[:, 1] - windows[:, 0]))
    total_out = total_universe - total_in
    if total_in <= 0 or total_out <= 0:
        return np.nan, {'reason': f'windows cover the whole {universe_label} span'}

    starts, ends = windows[:, 0], windows[:, 1]
    u_starts, u_ends = universe_windows[:, 0], universe_windows[:, 1]

    def _count_inside(times, w_starts, w_ends):
        if times.size == 0 or w_starts.size == 0:
            return 0
        idx = np.searchsorted(w_starts, times, side='right') - 1
        valid = idx >= 0
        return int(np.count_nonzero(valid & (times < w_ends[np.clip(idx, 0, None)])))

    n_in = 0
    n_universe = 0
    for times in spikes.values():
        n_universe += _count_inside(times, u_starts, u_ends)
        n_in += _count_inside(times, starts, ends)

    rate_in = n_in / total_in
    rate_out = (n_universe - n_in) / total_out
    ratio = rate_in / rate_out if rate_out > 0 else np.inf
    report = {'rate_in_up_hz': rate_in, 'rate_out_hz': rate_out, 'ratio': ratio,
              'spikes_in_up_fraction': n_in / max(n_universe, 1),
              'up_time_fraction': total_in / total_universe,
              'universe': universe_label, 'total_universe_sec': total_universe,
              'passed': bool(ratio >= min_ratio)}
    if verbose:
        print(f"UP-window alignment check (within {universe_label}): population rate "
              f"{rate_in:.2f} Hz inside vs {rate_out:.2f} Hz outside "
              f"(ratio {ratio:.2f}, need >= {min_ratio})")
        if not report['passed']:
            print("  FAILED -- either the windows do not sit on the spikes' clock "
                  "(check sleep_start_sample and the MUA epoch_start_sample), or "
                  "'outside' still includes non-NREM time with its own high activity.")
    return ratio, report


