r"""One figure per task run: every trial's photodiode pulse against its JSON stim window.

    python plot_trial_alignment.py CnL46                  # every registered day
    python plot_trial_alignment.py CnL46 260731 260812    # only these days
    python plot_trial_alignment.py CnL46 --out-dir D:/qc  # all in one folder instead

Each figure is saved next to the task's pickles in the sortout folder, as
task_alignment_<sortout>[_SessionNNN_Kind].png.

Visual check of the DIO <-> JSON matching extract_task_day.py does. Nothing is
read from the spike pickles: the photodiode edges and the JSON are read again
and run through the same match_trials / merge_split_pulses, so a row shows
exactly the stim window an extraction would write.

Left panel, one row per JSON trial, t = 0 at the JSON stimulusOnsetTime put on
the ephys clock: photodiode-high on top (black = the trial's pulse, orange = a
pulse merged into it, red = no JSON trial, grey = another trial's), JSON
stimulusOnsetTime -> choiceTime below in blue. Photodiode pulses with no JSON
trial get their own rows at the bottom. Middle / right panels: photodiode minus
JSON onset and offset per trial, in ms.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRATING_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(GRATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract_task_day as etd
from task_day_discovery import REGISTRY_FILE, load_registry, short_date

PRE_SEC = 1.0              # shown before stim onset
DIFF_LIM_MS = 120.0        # x range of the onset/offset difference panels; beyond = triangle at the edge
ROW_INCH = 0.11
COLORS = {'own': 'k', 'merged': 'tab:orange', 'none': 'tab:red', 'other': '0.7'}


def _rec_fs(rec_folder):
    try:
        return float(json.loads((Path(rec_folder) / 'params.json').read_text()).get('samplerate', 30000.0))
    except (OSError, ValueError):
        return 30000.0


def plot_task(key, task, out_dir, log=print):
    stem = Path(task['task_file'].replace('\\', '/')).stem
    tag = stem.split('_', 2)[-1]
    trials, _, _ = etd.load_behavior(task['task_file'])
    fs = _rec_fs(task['rec_folder'])
    channel, (on, off), match = etd.choose_channel(task['rec_folder'], fs, trials,
                                                   task.get('din_channel', 'auto'), log=lambda *a: None)
    trial_off, merged = etd.merge_split_pulses(on, off, fs, trials, match)
    idx = match['dio_index']
    on_s, off_s = on / fs, off / fs
    to_dio = lambda u: match['a'] * (u - match['ref']) + match['b']

    owner = {int(p): int(j) for j, p in enumerate(idx) if p >= 0}
    merged_owner = {q: j for j, m in merged.items() for q in m['pulses']}
    extra = [q for q in range(len(on)) if q not in owner and q not in merged_owner]

    j_on = np.array([to_dio(t['stimulusOnsetTime']) if t['stimulusOnsetTime'] is not None else np.nan
                     for t in trials])
    j_off = np.array([to_dio(t['choiceTime']) if t['choiceTime'] is not None else np.nan for t in trials])
    matched = idx >= 0
    d_on = np.where(matched, on_s[np.clip(idx, 0, None)], np.nan)
    d_off = np.where(matched, trial_off / fs, np.nan)
    dur = np.fmax(j_off - j_on, d_off - d_on)
    x_max = max(3.0, float(np.nanpercentile(dur, 95)) + 1.5) if np.isfinite(dur).any() else 3.0

    n_rows = len(trials) + len(extra)
    fig, (ax, ax_on, ax_off) = plt.subplots(
        1, 3, sharey=True, figsize=(15, max(5.0, ROW_INCH * n_rows + 2.2)),
        gridspec_kw={'width_ratios': [5, 1, 1], 'wspace': 0.05})

    def draw_pulses(row, t0, lo, hi, own):
        for q in range(np.searchsorted(off_s, lo), np.searchsorted(on_s, hi)):
            kind = ('none' if q in extra else 'own' if owner.get(q) == own
                    else 'merged' if merged_owner.get(q) == own else 'other')
            if own is None and kind == 'other':
                continue
            ax.broken_barh([(on_s[q] - t0, off_s[q] - on_s[q])], (row - 0.42, 0.42), color=COLORS[kind], lw=0)

    for j in range(len(trials)):
        if not np.isfinite(j_on[j]):
            continue
        t0 = j_on[j]
        draw_pulses(j, t0, t0 - PRE_SEC, t0 + x_max, j)
        if np.isfinite(j_off[j]):
            ax.broken_barh([(0, j_off[j] - t0)], (j + 0.04, 0.34), color='tab:blue', lw=0)
        if not matched[j]:
            ax.text(x_max, j, ' no photodiode pulse', color='tab:red', fontsize=6, va='center', ha='right')
        elif dur[j] > x_max:
            ax.text(x_max, j, f'JSON {j_off[j] - j_on[j]:.1f} s / photodiode {d_off[j] - d_on[j]:.1f} s > ',
                    color='w', fontsize=6, va='center', ha='right', fontweight='bold')
        if j in merged:
            gaps = ', '.join(f'{g * 1000:.0f} ms' for g in merged[j]['gap_sec'])
            ax.text(-PRE_SEC, j, f' merged, dark {gaps}', color='tab:orange', fontsize=6, va='center')
    for k, q in enumerate(extra):
        row = len(trials) + k
        draw_pulses(row, on_s[q], on_s[q] - PRE_SEC, on_s[q] + x_max, None)
        ax.text(-PRE_SEC, row, f' pulse {q} at {on_s[q]:.1f} s, {off_s[q] - on_s[q]:.2f} s long: no JSON trial',
                color='tab:red', fontsize=6, va='center')

    ax.axvline(0, color='tab:green', lw=0.6)
    ax.set_xlim(-PRE_SEC, x_max)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.set_yticks(np.arange(0, len(trials), 10 if len(trials) > 30 else 5))
    ax.set_ylabel('JSON trial')
    ax.set_xlabel('time from JSON stimulus onset on the ephys clock (s)')
    for kind, label in (('own', 'photodiode high (matched pulse)'), ('merged', 'pulse merged into the trial'),
                        ('none', 'pulse with no JSON trial'), ('other', "another trial's pulse")):
        ax.plot([], [], color=COLORS[kind], lw=5, label=label)
    ax.plot([], [], color='tab:blue', lw=5, label='JSON stimulusOnset -> choiceTime')
    ax.legend(fontsize=7, ncol=5, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)

    rows = np.arange(len(trials))
    for a, diff, name in ((ax_on, (d_on - j_on) * 1000, 'onset'), (ax_off, (d_off - j_off) * 1000, 'offset')):
        ok = np.isfinite(diff)
        inside = ok & (np.abs(diff) <= DIFF_LIM_MS)
        a.plot(diff[inside], rows[inside], '.', color='k', ms=3)
        for j in rows[ok & ~inside]:
            a.plot(np.sign(diff[j]) * DIFF_LIM_MS * 0.97, j, '>' if diff[j] > 0 else '<', color='tab:red', ms=4)
            a.text(0, j, f'{diff[j]:+.0f} ms', color='tab:red', fontsize=6, va='center', ha='center')
        a.axvline(0, color='tab:green', lw=0.6)
        for v in (-etd.DURATION_WARN_SEC * 1000, etd.DURATION_WARN_SEC * 1000):
            a.axvline(v, color='0.7', lw=0.6, ls='--')
        a.set_xlim(-DIFF_LIM_MS, DIFF_LIM_MS)
        med = np.nanmedian(diff) if ok.any() else np.nan
        a.set_xlabel(f'photodiode - JSON\n{name} (ms)')
        a.set_title(f'median {med:+.1f} ms\nmax |d| {np.nanmax(np.abs(diff)) if ok.any() else np.nan:.1f} ms',
                    fontsize=8)

    fig.suptitle(f"{key}  {tag}  ({task.get('rec_name')}, Din{channel}):  {match['n_matched']}/{len(trials)} JSON "
                 f"trials matched to {len(on)} photodiode pulses,  {len(merged_owner)} merged,  {len(extra)} with no "
                 f"JSON trial", fontsize=10, x=0.01, ha='left', y=0.995)
    top = 1 - 0.75 / fig.get_figheight()
    fig.subplots_adjust(left=0.05, right=0.99, top=top - 0.35 / fig.get_figheight(),
                        bottom=1.0 / fig.get_figheight())
    fig.text(0.005, 0.1 / fig.get_figheight(),
             f"Repro: python {Path(__file__).name} {key.split('_')[0]} {key.split('_')[1]} | registry {REGISTRY_FILE} | "
             f"JSON {task['task_file']}\nDIO {task['rec_folder']} | clock fit dio = {match['a']:.7f} * json + "
             f"{match['b']:.3f} s | merge tol {etd.FINE_TOL_SEC} s | {datetime.now().isoformat(timespec='seconds')}",
             fontsize=5, color='0.35', va='bottom')
    if out_dir is None:      # next to the task's pickles: task_spikes_trial_<x>.pkl -> task_alignment_<x>.png
        trial_pkl = Path(task['trial_pkl'])
        out = trial_pkl.with_name(trial_pkl.stem.replace('task_spikes_trial_', 'task_alignment_') + '.png')
    else:
        out = Path(out_dir) / f"{key}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log(f"{key} {tag}: {match['n_matched']}/{len(trials)} matched, merged {sorted(merged)}, "
        f"no-JSON pulses {extra} -> {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('animal')
    parser.add_argument('dates', nargs='*', help='YYMMDD; default every day of the animal in the registry')
    parser.add_argument('--out-dir', default=None,
                        help='one folder for every figure; default: next to each task\'s pickles in the sortout folder')
    args = parser.parse_args(argv)
    registry = load_registry()
    wanted = {short_date(d) for d in args.dates}
    out_dir = args.out_dir
    failures = 0
    for key in sorted(registry):
        config = registry[key]
        if config['animal'] != args.animal or (wanted and config['date'] not in wanted):
            continue
        for task in config.get('tasks', []):
            name = Path(task['task_file'].replace('\\', '/')).name
            if not task.get('rec_folder'):
                print(f"{key} {name}: no paired task .rec - skipped")
                continue
            try:
                plot_task(key, task, out_dir)
            except Exception as exc:
                failures += 1
                print(f"{key} {name}: FAILED: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
