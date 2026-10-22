r"""Extract one day's discrimination-task spikes into two pickles per task run.

    python extract_task_day.py CnL46 260728                 # every task of the day
    python extract_task_day.py CnL46 260812 --tasks 1       # only the 2nd task
    python extract_task_day.py CnL46 260728 --check-only    # DIO/JSON matching report, no spikes
    python extract_task_day.py CnL46 260728 --overwrite

Paths, sample windows and the DIO channel come from task_day_configs.json when
task_day_gui.py saved the day there, otherwise straight from disk via
task_day_discovery.discover_task_day (conversion_list.txt, behavior JSON,
.rec DIO, sortout folder) - there is no task_params.py to edit.

Per task (= one behavior JSON) this writes, in the sortout folder (new-server
mirror if the sortout lives on the read-only old server):

  task_spikes_trial_<sortout>[_SessionNNN_Kind].pkl
      Same schema as readDIO_task.py: per unit, per trial, spike times relative
      to stimulus onset in [onset - pre_sec, offset + post_sec).
  task_spikes_<sortout>[_SessionNNN_Kind].pkl
      Same schema as extract_session_spikes.py, but the window is the WHOLE
      task .rec: t = 0 is the first sample of the task recording and
      window_duration_sec is its full length.

Trial timing
------------
Trials are photodiode pulses on the task .rec's DIO (rising = stim on,
falling = stim off), in samples from the .rec's first DIO timestamp - the same
zero the existing readDIO_task.py uses. They are NOT assumed to be one-to-one
with the JSON: the task can leave a trial that started but was never logged
(CnL46 260728: 71 pulses, 70 JSON trials). Each JSON trial is matched to the
pulse whose onset agrees with its stimulusOnsetTime after fitting
dio_time = a * json_time + b over all trials (offset from the histogram of
all onset differences, then a linear fit to absorb clock drift). Only matched
trials are written; unmatched JSON trials and DIO pulses are listed in the
metadata.

That same fit puts the JSON's DLC tracking (Unix-time dlcReadTime_hr, or the
older session-relative stepTime) on the ephys clock, so position and spikes
share one time axis in both pickles.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRATING_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(GRATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from task_day_discovery import (discover_task_day, fill_output_paths, get_day_config,
                                save_day_config)

DIN_CHANNELS = range(1, 9)
PREFERRED_DIN = 5          # photodiode channel on every day so far
COARSE_TOL_SEC = 0.5       # first-pass onset match window around the histogram offset
FINE_TOL_SEC = 0.1         # final onset match window after the linear clock fit
DURATION_WARN_SEC = 0.1    # flag trials whose DIO and JSON stim durations disagree by more
INTERRUPT_GAP_SEC = 0.05   # a photodiode dropout longer than this inside a trial = stim interrupted


# ============================================================ behavior JSON

JSON_TRIAL_KEYS = [
    'rewardedOnLeft',
    'leftOrientation', 'rightOrientation',
    'leftSpatialFreq', 'rightSpatialFreq',
    'leftTemporalFreq', 'rightTemporalFreq',
    'leftContrast', 'rightContrast',
    'rewardedOrientation', 'nonRewardedOrientation',
    'choice', 'correct', 'rewarded',
    'stimulusOnsetTime', 'choiceTime', 'choiceLatency',
]


def load_behavior(task_file):
    """(trials, tracking) from a behavior JSON.

    tracking holds Unix-time samples, whichever of the two JSON layouts the
    file uses: the current dlc*_hr arrays stamped by dlcReadTime_hr (Unix), or
    the older dlcX/dlcY/... stamped by session-relative stepTime.
    """
    with open(task_file, 'r') as f:
        data = json.load(f)
    trials = [{k: t.get(k) for k in JSON_TRIAL_KEYS} for t in data.get('trials', [])]

    tracking = None
    if data.get('dlcReadTime_hr'):
        t = np.asarray(data['dlcReadTime_hr'], dtype=float)
        get = lambda k: np.asarray(data.get(k + '_hr', []), dtype=float)
        tracking = {'unix_time': t, 'x': get('dlcX'), 'y': get('dlcY'),
                    'heading': get('dlcHeading'), 'head_angle': get('dlcHeadAngle'),
                    'signal': get('dlcSignal'), 'source': 'dlc*_hr / dlcReadTime_hr'}
    elif data.get('stepTime') and data.get('startTime') is not None:
        t = np.asarray(data['stepTime'], dtype=float) + float(data['startTime'])
        get = lambda k: np.asarray(data.get(k, []), dtype=float)
        tracking = {'unix_time': t, 'x': get('dlcX'), 'y': get('dlcY'),
                    'heading': get('dlcHeading'), 'head_angle': get('dlcHeadAngle'),
                    'signal': get('dlcSignal'), 'source': 'dlc* / startTime + stepTime'}
    if tracking is not None:
        n = tracking['unix_time'].size
        for k in ('x', 'y', 'heading', 'head_angle', 'signal'):
            if tracking[k].size != n:
                tracking[k] = np.full(n, np.nan)
    return trials, tracking, data.get('startTime')


# ============================================================ photodiode DIO

def read_photodiode(rec_folder, channel):
    """Stim pulses on one DIN channel: (onsets, offsets) in samples from the
    .rec's first DIO timestamp, or None if the channel has no file.

    A trailing rising edge with no falling edge (recording stopped mid-stim)
    is dropped.
    """
    from trodes_io import DIO
    folders = DIO.get_dio_folders(rec_folder)
    if not folders or DIO.extract_DIN(folders[0], channel) is None:
        return None
    t, s = DIO.concatenate_din_data(folders, channel)
    t = np.asarray(t).ravel().astype(np.int64)
    s = np.asarray(s).ravel()
    t = t - t[0]
    rising = t[s == 1]
    falling = t[s == 0]
    onsets, offsets = [], []
    for i, r in enumerate(rising):
        j = np.searchsorted(falling, r, side='right')
        if j >= len(falling):
            continue
        if i + 1 < len(rising) and falling[j] > rising[i + 1]:
            continue
        onsets.append(r)
        offsets.append(falling[j])
    return np.asarray(onsets, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def _nearest(sorted_vals, targets):
    """Index of the nearest element of sorted_vals for each target."""
    idx = np.clip(np.searchsorted(sorted_vals, targets), 1, len(sorted_vals) - 1)
    left, right = sorted_vals[idx - 1], sorted_vals[idx]
    return np.where(np.abs(targets - left) <= np.abs(targets - right), idx - 1, idx)


def match_trials(dio_on, dio_off, fs, trials):
    """Match JSON trials to DIO pulses by onset time.

    Returns a dict with 'dio_index' (per JSON trial, -1 = unmatched), the clock
    fit dio_sec = a * (json_unix - ref) + b, and QC numbers.
    """
    n_json = len(trials)
    j_on = np.array([t['stimulusOnsetTime'] if t['stimulusOnsetTime'] is not None else np.nan
                     for t in trials], dtype=float)
    j_off = np.array([t['choiceTime'] if t['choiceTime'] is not None else np.nan
                      for t in trials], dtype=float)
    result = {'dio_index': np.full(n_json, -1), 'n_matched': 0, 'a': 1.0, 'b': np.nan,
              'ref': np.nan, 'onset_resid_ms': np.array([]), 'duration_diff_ms': np.full(n_json, np.nan)}
    valid = np.isfinite(j_on)
    if len(dio_on) < 2 or valid.sum() < 2:
        return result

    d_on = dio_on / fs
    ref = float(np.nanmin(j_on))
    j_rel = j_on - ref

    # 1) offset = mode of all pairwise onset differences (the true offset collects
    #    one vote per trial; wrong pairings spread out).
    diffs = (d_on[:, None] - j_rel[None, valid]).ravel()
    bin_w = 0.05
    edges = np.arange(diffs.min(), diffs.max() + 2 * bin_w, bin_w)
    counts, _ = np.histogram(diffs, bins=edges)
    counts = np.convolve(counts, np.ones(5), mode='same')
    center = edges[np.argmax(counts)] + bin_w / 2
    near = diffs[np.abs(diffs - center) < 0.25]
    b0 = float(np.median(near)) if near.size else float(center)

    def assign(pred, tol):
        idx = np.full(n_json, -1)
        err = np.full(n_json, np.inf)
        ok = np.isfinite(pred)
        nearest = _nearest(d_on, pred[ok])
        e = np.abs(d_on[nearest] - pred[ok])
        idx[ok] = np.where(e < tol, nearest, -1)
        err[ok] = e
        # one pulse per trial: keep the closer claim
        for i in np.unique(idx[idx >= 0]):
            claimants = np.where(idx == i)[0]
            if len(claimants) > 1:
                keep = claimants[np.argmin(err[claimants])]
                idx[claimants[claimants != keep]] = -1
        return idx

    idx = assign(j_rel + b0, COARSE_TOL_SEC)
    a, b = 1.0, b0
    m = idx >= 0
    if m.sum() >= 3:
        a, b = np.polyfit(j_rel[m], d_on[idx[m]], 1)
    idx = assign(a * j_rel + b, FINE_TOL_SEC)
    m = idx >= 0
    if m.sum() >= 3:
        a, b = np.polyfit(j_rel[m], d_on[idx[m]], 1)

    resid = (d_on[idx[m]] - (a * j_rel[m] + b)) * 1000
    dur_diff = np.full(n_json, np.nan)
    dio_dur = (dio_off - dio_on) / fs
    both = m & np.isfinite(j_off)
    dur_diff[both] = (dio_dur[idx[both]] - (j_off[both] - j_on[both])) * 1000
    result.update(dio_index=idx, n_matched=int(m.sum()), a=float(a), b=float(b), ref=ref,
                  onset_resid_ms=resid, duration_diff_ms=dur_diff)
    return result


def merge_split_pulses(on, off, fs, trials, match):
    """Stim-off sample per JSON trial, with photodiode dropouts bridged.

    The photodiode can fall and rise again inside one logged trial (CnL46
    260812 trial 16: a 6 ms flicker; 260731 trials 19 and 81: 0.9 s and 4.5 s
    dark). The trial is matched to the first pulse, so its falling edge is not
    the stim offset. Any following pulse that has no JSON trial of its own and
    ends before the trial's choiceTime (+ FINE_TOL_SEC) is merged into it.

    Returns (trial_off, merged): trial_off[j] is -1 for unmatched trials;
    merged maps JSON trial -> {'pulses': [...], 'gap_sec': [...]} and
    match['duration_diff_ms'] is updated for those trials.
    """
    idx = match['dio_index']
    trial_off = np.full(len(trials), -1, dtype=np.int64)
    trial_off[idx >= 0] = off[idx[idx >= 0]]
    taken = set(idx[idx >= 0].tolist())
    merged = {}
    for j in np.where(idx >= 0)[0]:
        j_on, j_off = trials[j]['stimulusOnsetTime'], trials[j]['choiceTime']
        if j_off is None:
            continue
        limit = (match['a'] * (j_off - match['ref']) + match['b'] + FINE_TOL_SEC) * fs
        pulses, gaps = [], []
        q = idx[j] + 1
        while q < len(on) and q not in taken and off[q] <= limit:
            gaps.append(float((on[q] - trial_off[j]) / fs))
            pulses.append(int(q))
            trial_off[j] = off[q]
            q += 1
        if pulses:
            merged[int(j)] = {'pulses': pulses, 'gap_sec': gaps}
            match['duration_diff_ms'][j] = ((trial_off[j] - on[idx[j]]) / fs - (j_off - j_on)) * 1000
    return trial_off, merged


def choose_channel(rec_folder, fs, trials, requested='auto', log=print):
    """(channel, pulses, match) - the requested channel, or with 'auto' the DIN
    channel whose pulses match the most JSON trials (ties -> PREFERRED_DIN)."""
    if requested not in (None, '', 'auto'):
        ch = int(requested)
        pulses = read_photodiode(rec_folder, ch)
        if pulses is None:
            raise FileNotFoundError(f"No Din{ch} file in {rec_folder}")
        return ch, pulses, match_trials(*pulses, fs, trials)

    best = None
    n_json = len(trials)
    counts = {}
    for ch in DIN_CHANNELS:
        pulses = read_photodiode(rec_folder, ch)
        if pulses is None:
            continue
        n_p = counts[ch] = len(pulses[0])
        if n_p < max(2, 0.5 * n_json) or n_p > 5 * n_json + 10:
            continue            # constant lines and the camera/sync square wave
        match = match_trials(*pulses, fs, trials)
        score = (match['n_matched'], ch == PREFERRED_DIN)
        log(f"    Din{ch}: {n_p} pulses, {match['n_matched']}/{n_json} JSON trials matched")
        if best is None or score > best[0]:
            best = (score, ch, pulses, match)
    if best is None or best[3]['n_matched'] == 0:
        raise RuntimeError(f"no DIN channel's pulses match the {n_json} JSON trials "
                           f"(pulses per channel: {counts}) - the trials may predate the "
                           f".rec, or the task had no photodiode stimulus. Set the DIN "
                           f"channel by hand to force a match.")
    return best[1], best[2], best[3]


def report_match(match, n_pulses, n_json, merged=None, log=print):
    idx = match['dio_index']
    merged = merged or {}
    merged_pulses = [q for m in merged.values() for q in m['pulses']]
    unmatched_json = np.where(idx < 0)[0]
    unmatched_dio = np.setdiff1d(np.arange(n_pulses), np.r_[idx[idx >= 0], merged_pulses].astype(int))
    for j, m in merged.items():
        kind = 'stim interrupted' if max(m['gap_sec']) > INTERRUPT_GAP_SEC else 'flicker'
        log(f"  note: JSON trial {j}: photodiode pulse(s) {m['pulses']} merged into its stim window "
            f"(dark for {', '.join(f'{g * 1000:.0f} ms' for g in m['gap_sec'])} - {kind})")
    resid = match['onset_resid_ms']
    dd = match['duration_diff_ms'][np.isfinite(match['duration_diff_ms'])]
    log(f"  matched {match['n_matched']}/{n_json} JSON trials to {n_pulses} DIO pulses; "
        f"clock fit: dio = {match['a']:.7f} * json + {match['b']:.3f} s")
    if resid.size:
        log(f"  onset residual: median |r| = {np.median(np.abs(resid)):.1f} ms, "
            f"max = {np.max(np.abs(resid)):.1f} ms")
    if dd.size:
        log(f"  stim duration DIO - JSON: median = {np.median(dd):+.1f} ms, "
            f"max |d| = {np.max(np.abs(dd)):.1f} ms")
        bad = np.where(np.abs(match['duration_diff_ms']) > DURATION_WARN_SEC * 1000)[0]
        if bad.size:
            log(f"  WARNING: {bad.size} trial(s) with stim duration off by > "
                f"{DURATION_WARN_SEC * 1000:.0f} ms: JSON trials {bad.tolist()}")
    if unmatched_json.size:
        log(f"  WARNING: JSON trials with no DIO pulse (dropped): {unmatched_json.tolist()}")
    if unmatched_dio.size:
        log(f"  note: DIO pulses with no JSON trial (ignored): {unmatched_dio.tolist()}")
    return unmatched_json, unmatched_dio


# ============================================================ units

def load_units(sortout_folder, task_start, task_end, log=print):
    """Spike trains of every unit inside [task_start, task_end), shifted to
    task-local samples. Uses curated_analyzer when present, else each shank's
    latest raw sorting_analyzer (same fallback GratingExport / the sleep
    extraction use, so unit ids match: 'shankX_unitY'). Raw units take their
    quality from sortout/unit_labels.json when it exists."""
    from passive_visual.FreelyMovingProcessing.Grating.extract_sleep_session_spikes_multishank import \
        _resolve_sources
    sources, fs, use_curated = _resolve_sources(sortout_folder)

    labels = {}
    labels_file = Path(sortout_folder) / 'unit_labels.json'
    if not use_curated and labels_file.is_file():
        with open(labels_file) as f:
            labels = json.load(f)
        log(f"  unit quality from {labels_file.name}")

    units = {}
    for source in sources:
        sorting = source['sorting']
        for unit_id in source['unit_ids']:
            train = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=np.int64)
            lo, hi = np.searchsorted(train, [task_start, task_end])
            shank = source['shank_of'](unit_id)
            quality = source['quality_of'](unit_id)
            if labels:
                quality = labels.get(f"shank{shank}", {}).get(str(unit_id), quality)
            uid = f"shank{shank}_unit{unit_id}" if shank is not None else f"unit{unit_id}"
            units[uid] = {'unit_id': int(unit_id), 'shank': shank, 'quality': quality,
                          'train': train[lo:hi] - task_start}
    status = 'curated_analyzer' if use_curated else 'per-shank sorting_analyzer (uncurated)'
    log(f"  {len(units)} units from {status}, fs = {fs} Hz")
    return units, float(fs), status


def read_merge_map(sortout_folder):
    path = Path(sortout_folder) / 'unit_merge_map.json'
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ============================================================ tracking

def tracking_on_ephys_clock(tracking, match):
    """Tracking samples with 'time_sec' on the task-local ephys clock, sorted
    and with repeated timestamps removed."""
    if tracking is None or not np.isfinite(match['b']):
        return None
    t = match['a'] * (tracking['unix_time'] - match['ref']) + match['b']
    order = np.argsort(t, kind='stable')
    t = t[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    out = {k: tracking[k][order][keep] for k in ('x', 'y', 'heading', 'head_angle', 'signal')}
    out['time_sec'] = t[keep]
    out['source'] = tracking['source']
    return out


def slice_tracking(trk, t0, t1):
    if trk is None:
        return {k: [] for k in ('position_x', 'position_y', 'position_time', 'heading',
                                'head_angle', 'dlc_signal')}
    i0, i1 = np.searchsorted(trk['time_sec'], [t0, t1], side='left')
    return {'position_x': trk['x'][i0:i1].tolist(), 'position_y': trk['y'][i0:i1].tolist(),
            'position_time': trk['time_sec'][i0:i1].tolist(),
            'heading': trk['heading'][i0:i1].tolist(), 'head_angle': trk['head_angle'][i0:i1].tolist(),
            'dlc_signal': trk['signal'][i0:i1].tolist()}


# ============================================================ pickles

def _trial_param_dict(tp):
    return {
        'left_orientation': tp['leftOrientation'], 'right_orientation': tp['rightOrientation'],
        'left_spatial_freq': tp['leftSpatialFreq'], 'right_spatial_freq': tp['rightSpatialFreq'],
        'left_temporal_freq': tp['leftTemporalFreq'], 'right_temporal_freq': tp['rightTemporalFreq'],
        'left_contrast': tp['leftContrast'], 'right_contrast': tp['rightContrast'],
        'rewarded_orientation': tp['rewardedOrientation'],
        'non_rewarded_orientation': tp['nonRewardedOrientation'],
        'choice': tp['choice'], 'correct': tp['correct'], 'rewarded': tp['rewarded'],
        'stimulus_onset_time': tp['stimulusOnsetTime'], 'choice_time': tp['choiceTime'],
        'choice_latency': tp['choiceLatency'],
    }


def build_outputs(config, task, units, fs, sort_status, trials, json_idx, on, off,
                  trk, match, channel, unmatched_json, unmatched_dio, json_start, merged=None):
    """(trial_pkl_dict, task_pkl_dict) in readDIO_task.py / extract_session_spikes.py schemas."""
    merged = merged or {}

    def photodiode_fields(j):
        gaps = merged.get(int(j), {}).get('gap_sec', [])
        return {'photodiode_n_pulses': 1 + len(gaps), 'photodiode_gap_sec': list(gaps),
                'stim_interrupted': bool(gaps and max(gaps) > INTERRUPT_GAP_SEC)}

    pre_sec, post_sec = float(config['pre_sec']), float(config['post_sec'])
    n = len(json_idx)
    task_len = int(task['task_end']) - int(task['task_start'])
    rewarded_on_left = np.array([bool(trials[j]['rewardedOnLeft']) for j in json_idx], dtype=bool)
    unique_conditions = np.unique(rewarded_on_left)
    durations = (off - on) / fs
    onsets_sec, offsets_sec = on / fs, off / fs

    metadata = {
        'animal_id': config['animal'],
        'session_id': f"20{config['date']}" if len(config['date']) == 6 else config['date'],
        'recording_folder': task['rec_folder'],
        'task_file': task['task_file'],
        'sortout_folder': config['sortout_folder'],
        'unit_source': sort_status,
        'extraction_date': datetime.now().isoformat(),
        'sampling_frequency': fs,
        'n_trials': n,
        'task_start_sample': int(task['task_start']),   # in the sorter's concatenated sample space
        'task_end_sample': int(task['task_end']),
        'din_channel': int(channel),
        'dio_json_clock_fit': {'a': match['a'], 'b_sec': match['b'], 'json_ref_unix': match['ref'],
                               'note': 'task_local_sec = a * (json_unix - json_ref_unix) + b_sec'},
        'json_trial_index': json_idx.tolist(),
        'unmatched_json_trials': unmatched_json.tolist(),
        'unmatched_dio_pulses': unmatched_dio.tolist(),
        'merged_dio_pulses': merged,     # JSON trial -> photodiode pulses bridged into its stim window
        'tracking_source': trk['source'] if trk else None,
        'position_time_frame': 'seconds from the first sample of the task .rec (ephys clock)',
        'unit_merge_map': read_merge_map(config['sortout_folder']),
        'extracted_by': str(Path(__file__).resolve()),
    }
    experiment_parameters = {
        'stimulus_duration': None, 'iti_duration': None,
        'trial_duration': float(np.mean(durations)) if n else 0.0,
        'total_trials': n, 'n_conditions': len(unique_conditions),
        'n_left_trials': int(rewarded_on_left.sum()), 'n_right_trials': int((~rewarded_on_left).sum()),
    }
    all_trial_parameters = []
    for k, j in enumerate(json_idx):
        d = {'trial_index': k, 'json_trial_index': int(j),
             'rewarded_on_left': bool(rewarded_on_left[k]),
             'trial_start': int(on[k]), 'trial_end': int(off[k]),
             'trial_duration': float(durations[k]),
             'trial_onset_sec': float(onsets_sec[k]), 'trial_offset_sec': float(offsets_sec[k])}
        d.update(_trial_param_dict(trials[j]))
        d.update(photodiode_fields(j))
        d.update(slice_tracking(trk, onsets_sec[k], offsets_sec[k]))
        all_trial_parameters.append(d)
    trial_info = {
        'rewarded_on_left': rewarded_on_left.tolist(),
        'unique_conditions': unique_conditions.tolist(),
        'trial_windows': [(int(a), int(b)) for a, b in zip(on, off)],
        'all_trial_parameters': all_trial_parameters,
        'trial_durations': durations.tolist(),
    }

    # ---- per-trial spikes (readDIO_task.py schema)
    pre_n, post_n = int(pre_sec * fs), int(post_sec * fs)
    spike_data, unit_info = {}, {}
    repeat_idx = np.zeros(n, dtype=int)
    for cond in unique_conditions:
        sel = np.where(rewarded_on_left == cond)[0]
        repeat_idx[sel] = np.arange(sel.size)
    for u_i, (uid, u) in enumerate(units.items()):
        train = u['train']
        lo = np.searchsorted(train, on - pre_n)
        hi = np.searchsorted(train, off + post_n)
        rows, total = [], 0
        for k in range(n):
            rel = (train[lo[k]:hi[k]] - on[k]) / fs
            dur = durations[k]
            pre_c = int(np.sum(rel < 0))
            dur_c = int(np.sum((rel >= 0) & (rel < dur)))
            post_c = int(np.sum(rel >= dur))
            total += rel.size
            row = {'trial_index': k, 'json_trial_index': int(json_idx[k]),
                   'rewarded_on_left': bool(rewarded_on_left[k]),
                   'condition_idx': int(rewarded_on_left[k]), 'repeat_idx': int(repeat_idx[k]),
                   'spike_times': rel.tolist(), 'spike_count': int(rel.size),
                   'trial_start': int(on[k]), 'trial_end': int(off[k]), 'trial_duration': float(dur)}
            row.update(_trial_param_dict(trials[json_idx[k]]))
            row.update(photodiode_fields(json_idx[k]))
            row.update({'pre_trial_count': pre_c, 'during_trial_count': dur_c, 'post_trial_count': post_c,
                        'firing_rate_pre': pre_c / pre_sec if pre_sec else np.nan,
                        'firing_rate_during': dur_c / dur if dur > 0 else np.nan,
                        'firing_rate_post': post_c / post_sec if post_sec else np.nan})
            rows.append(row)
        spike_data[uid] = rows
        unit_info[uid] = {'original_unit_id': u['unit_id'], 'shank': u['shank'], 'quality': u['quality'],
                          'sorting_folder': config['sortout_folder'], 'n_spikes_total': total,
                          'unit_index': u_i}
    trial_pkl = {
        'metadata': dict(metadata, experiment_type='behavior_trial'),
        'experiment_parameters': experiment_parameters,
        'trial_info': trial_info,
        'spike_data': spike_data,
        'unit_info': unit_info,
        'extraction_params': {'window_pre': pre_sec, 'window_post': post_sec,
                              'total_units': len(units)},
    }

    # ---- whole task period (extract_session_spikes.py schema, window = whole .rec)
    duration_sec = task_len / fs
    session_position = None
    if trk is not None:
        in_win = (trk['time_sec'] >= 0) & (trk['time_sec'] <= duration_sec)
        pt = trk['time_sec'][in_win]
        px, py = trk['x'][in_win], trk['y'][in_win]
        if pt.size >= 2:
            vx, vy = np.gradient(px, pt), np.gradient(py, pt)
        else:
            vx = vy = np.zeros_like(pt)
        offset = (match['a'] * (float(json_start) - match['ref']) + match['b']
                  if json_start is not None else np.nan)
        session_position = {
            'position_x': px, 'position_y': py, 'position_time_sec': pt,
            'velocity_x': vx, 'velocity_y': vy, 'speed': np.hypot(vx, vy),
            'heading': trk['heading'][in_win], 'head_angle': trk['head_angle'][in_win],
            'dlc_signal': trk['signal'][in_win],
            'session_to_window_offset_sec': offset,
        }
    task_spike_data = {uid: {'spike_times_sec': u['train'] / fs, 'n_spikes': int(u['train'].size),
                             'unit_id': u['unit_id'], 'shank': u['shank'], 'quality': u['quality']}
                       for uid, u in units.items()}
    task_pkl = {
        'metadata': dict(metadata, experiment_type='behavior_task_whole_period', n_units=len(units)),
        'experiment_parameters': experiment_parameters,
        'trial_info': trial_info,
        'window': {
            'window_start_sample': 0, 'window_end_sample': task_len,
            'window_duration_sec': duration_sec,
            'pre_stim_sec': float(onsets_sec[0]) if n else None,
            'post_stim_sec': float(duration_sec - offsets_sec[-1]) if n else None,
            'first_stim_onset_sec': float(onsets_sec[0]) if n else None,
            'last_stim_offset_sec': float(offsets_sec[-1]) if n else None,
            'trial_onsets_sec': onsets_sec, 'trial_offsets_sec': offsets_sec,
            'note': 't = 0 is the first sample of the task .rec; the window is the whole task recording',
        },
        'trial_params': [trials[j] for j in json_idx],
        'session_position': session_position,
        'spike_data': task_spike_data,
    }
    return trial_pkl, task_pkl


def _save(obj, path, log):
    from server_fallback import resolve_output_folder
    folder = resolve_output_folder(Path(path).parent)
    out = folder / Path(path).name
    tmp = out.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    log(f"  saved {out}")
    return out


# ============================================================ driver

def run_task(config, task, overwrite=False, check_only=False, log=print):
    """Match, extract and save one task. Returns a summary dict (also stored
    in the registry as task['last_run'])."""
    name = Path(task['task_file']).name
    log(f"\n=== {name}  <->  {task.get('rec_name')} ===")
    for key in ('rec_folder', 'task_start', 'task_end'):
        if task.get(key) in (None, ''):
            raise ValueError(f"{name}: '{key}' is not set - fill it in the GUI")
    if not config.get('sortout_folder') and not check_only:
        raise ValueError("sortout_folder is not set")

    if not check_only:
        trial_out, task_out = Path(task['trial_pkl']), Path(task['task_pkl'])
        if not overwrite and trial_out.exists() and task_out.exists():
            log(f"  both outputs exist - skipping (use overwrite):\n    {trial_out}\n    {task_out}")
            return {'status': 'skipped (exists)'}

    trials, tracking, json_start = load_behavior(task['task_file'])
    log(f"  JSON: {len(trials)} trials; tracking: {tracking['source'] if tracking else 'none'}")

    fs_guess = 30000.0
    params_json = Path(task['rec_folder']) / 'params.json'
    if params_json.is_file():
        try:
            fs_guess = float(json.loads(params_json.read_text()).get('samplerate', fs_guess))
        except (OSError, ValueError):
            pass
    channel, (on, off), match = choose_channel(task['rec_folder'], fs_guess, trials,
                                               task.get('din_channel', 'auto'), log=log)
    log(f"  photodiode: Din{channel}")
    trial_off, merged = merge_split_pulses(on, off, fs_guess, trials, match)
    unmatched_json, unmatched_dio = report_match(match, len(on), len(trials), merged, log=log)
    summary = {'din_channel_used': channel, 'n_json_trials': len(trials), 'n_dio_pulses': int(len(on)),
               'n_matched': match['n_matched'], 'unmatched_json_trials': unmatched_json.tolist(),
               'unmatched_dio_pulses': unmatched_dio.tolist(), 'merged_dio_pulses': merged,
               'run_at': datetime.now().isoformat(timespec='seconds')}
    task_len = int(task['task_end']) - int(task['task_start'])
    if len(off) and off.max() > task_len:
        raise ValueError(f"DIO pulses run to sample {off.max()} but the task window is only "
                         f"{task_len} samples - task_start/task_end do not belong to this .rec")
    if check_only:
        summary['status'] = 'checked'
        return summary
    if match['n_matched'] == 0:
        raise RuntimeError("no trial matched - nothing to extract")

    units, fs, sort_status = load_units(config['sortout_folder'], int(task['task_start']),
                                        int(task['task_end']), log=log)
    if abs(fs - fs_guess) > 1:
        log(f"  WARNING: sorter fs {fs} != .rec params.json samplerate {fs_guess}; "
            f"re-matching with the sorter fs")
        match = match_trials(on, off, fs, trials)
        trial_off, merged = merge_split_pulses(on, off, fs, trials, match)
    json_idx = np.where(match['dio_index'] >= 0)[0]
    order = np.argsort(match['dio_index'][json_idx])
    json_idx = json_idx[order]
    pulse_idx = match['dio_index'][json_idx]
    trk = tracking_on_ephys_clock(tracking, match)

    trial_pkl, task_pkl = build_outputs(config, task, units, fs, sort_status, trials, json_idx,
                                        on[pulse_idx], trial_off[json_idx], trk, match, channel,
                                        unmatched_json, unmatched_dio, json_start, merged)
    summary['trial_pkl'] = str(_save(trial_pkl, trial_out, log))
    summary['task_pkl'] = str(_save(task_pkl, task_out, log))
    summary['n_units'] = len(units)
    summary['status'] = 'done'
    return summary


def run_day(animal, date, task_indices=None, overwrite=False, check_only=False,
            din=None, rediscover=False, log=print):
    config = discover_task_day(animal, date) if rediscover else get_day_config(animal, date)
    fill_output_paths(config)
    log(f"{animal} {date}: sortout = {config.get('sortout_folder')}")
    for note in config.get('notes', []):
        log(f"  note: {note}")
    tasks = config.get('tasks', [])
    if not tasks:
        log("  no behavior JSON with trials - nothing to do")
        return 0
    failures = 0
    for i, task in enumerate(tasks):
        if task_indices is not None and i not in task_indices:
            continue
        if din is not None:
            task['din_channel'] = din
        if not task.get('rec_folder'):
            log(f"\n=== {Path(task['task_file']).name}: no paired task .rec - skipped ===")
            task['last_run'] = {'status': 'skipped (no .rec)',
                                'run_at': datetime.now().isoformat(timespec='seconds')}
            continue
        try:
            task['last_run'] = run_task(config, task, overwrite=overwrite,
                                        check_only=check_only, log=log)
        except Exception as exc:          # keep going with the day's other tasks
            failures += 1
            task['last_run'] = {'status': f'failed: {exc}',
                                'run_at': datetime.now().isoformat(timespec='seconds')}
            log(f"  FAILED: {type(exc).__name__}: {exc}")
    save_day_config(config)      # remembers the config used + each task's last_run
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('animal')
    parser.add_argument('date', help='YYMMDD')
    parser.add_argument('--tasks', default=None,
                        help='comma-separated task indices (0-based, in config order); default all')
    parser.add_argument('--din', default=None, help="photodiode DIN channel (default: config / auto)")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--check-only', action='store_true',
                        help='only match DIO pulses to JSON trials and report; no spikes, no files')
    parser.add_argument('--rediscover', action='store_true',
                        help='ignore task_day_configs.json and rebuild the config from disk')
    args = parser.parse_args(argv)
    task_indices = ({int(x) for x in args.tasks.split(',') if x.strip()}
                    if args.tasks else None)
    failures = run_day(args.animal, args.date, task_indices, args.overwrite,
                       args.check_only, args.din, args.rediscover)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
