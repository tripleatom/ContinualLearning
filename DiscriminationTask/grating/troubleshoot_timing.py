import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import numpy as np
import matplotlib.pyplot as plt
from spikeinterface import load_sorting_analyzer
from trodes_io import DIO
from readDIO_grating import get_trial_params


def compare_dio_json_timing(trial_start, trial_end, trial_params, fs, tol=0.05):
    """
    Cross-validate DIO hardware timestamps against behavioral JSON timing.

    Two clock-offset-independent checks:
      1. Trial duration: (trial_end - trial_start) / fs  vs  choiceLatency
      2. Inter-trial interval: diff(trial_start) / fs   vs  diff(stimulusOnsetTime)

    Parameters
    ----------
    trial_start  : np.ndarray  - rising-edge sample indices (0-based)
    trial_end    : np.ndarray  - falling-edge sample indices (0-based)
    trial_params : list of dict - from get_trial_params()
    fs           : float        - sampling frequency in Hz
    tol          : float        - mismatch threshold in seconds (default 50 ms)
    """
    n = len(trial_start)

    # --- DIO trial durations (seconds) ---
    dio_dur = (trial_end - trial_start) / fs

    # --- JSON trial duration from choiceTime - stimulusOnsetTime (both Unix seconds) ---
    # choiceLatency in the JSON is always 0.0 (not populated by the task software)
    json_dur = np.array(
        [tp['choiceTime'] - tp['stimulusOnsetTime']
         if tp['choiceTime'] is not None and tp['stimulusOnsetTime'] is not None
         else np.nan
         for tp in trial_params], dtype=float)

    valid = ~np.isnan(json_dur)
    dur_diff = dio_dur[valid] - json_dur[valid]
    bad_dur = np.where(valid)[0][np.abs(dur_diff) > tol]

    # --- Inter-trial intervals ---
    dio_iti = np.diff(trial_start) / fs

    # stimulusOnsetTime is a Unix timestamp in seconds; diff gives ITI in seconds directly
    json_onset = np.array(
        [tp['stimulusOnsetTime'] if tp['stimulusOnsetTime'] is not None else np.nan
         for tp in trial_params], dtype=float)

    iti_valid = ~(np.isnan(json_onset[:-1]) | np.isnan(json_onset[1:]))
    json_iti = np.diff(json_onset)
    iti_diff = dio_iti[iti_valid] - json_iti[iti_valid]
    bad_iti = np.where(iti_valid)[0][np.abs(iti_diff) > tol]

    print("\n=== DIO vs JSON Timing Comparison ===")
    print(f"  Trials: {n}  |  JSON duration = choiceTime − stimulusOnsetTime (Unix seconds)")
    print(f"\n  Trial duration  (DIO − JSON) over {valid.sum()} valid trials:")
    print(f"    mean={np.mean(dur_diff)*1000:+.1f} ms  std={np.std(dur_diff)*1000:.1f} ms  "
          f"max|diff|={np.max(np.abs(dur_diff))*1000:.1f} ms")
    if len(bad_dur):
        print(f"    WARNING: {len(bad_dur)} trial(s) exceed {tol*1000:.0f} ms tolerance → indices {bad_dur.tolist()}")
    else:
        print(f"    OK: all within {tol*1000:.0f} ms tolerance")

    print(f"\n  Inter-trial interval (DIO − JSON) over {iti_valid.sum()} valid gaps:")
    print(f"    mean={np.mean(iti_diff)*1000:+.1f} ms  std={np.std(iti_diff)*1000:.1f} ms  "
          f"max|diff|={np.max(np.abs(iti_diff))*1000:.1f} ms")
    if len(bad_iti):
        print(f"    WARNING: {len(bad_iti)} gap(s) exceed {tol*1000:.0f} ms tolerance → after trial indices {bad_iti.tolist()}")
    else:
        print(f"    OK: all within {tol*1000:.0f} ms tolerance")

    return {
        'dio_durations':       dio_dur,
        'json_durations':      json_dur,
        'duration_diff_ms':    dur_diff * 1000,
        'bad_duration_trials': bad_dur,
        'dio_iti':             dio_iti,
        'json_iti':            json_iti,
        'iti_diff_ms':         iti_diff * 1000,
        'bad_iti_trials':      bad_iti,
    }


def plot_dio_json_comparison(timing_result, save_dir=None):
    """
    Four-panel diagnostic plot for DIO vs JSON timing comparison.

      Row 1: trial durations side-by-side + their difference
      Row 2: inter-trial intervals side-by-side + their difference
    """
    dio_dur  = timing_result['dio_durations']
    json_dur = timing_result['json_durations']
    dur_diff = timing_result['duration_diff_ms']
    dio_iti  = timing_result['dio_iti']
    json_iti = timing_result['json_iti']
    iti_diff = timing_result['iti_diff_ms']

    trial_idx = np.arange(len(dio_dur))
    iti_idx   = np.arange(len(dio_iti))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('DIO vs JSON Timing Comparison', fontsize=13)

    ax = axes[0, 0]
    ax.plot(trial_idx, dio_dur,  label='DIO',  lw=1)
    ax.plot(trial_idx, json_dur, label='JSON', lw=1, alpha=0.8)
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Trial duration')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(trial_idx, dur_diff, color='C2', lw=1)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Trial index')
    ax.set_ylabel('DIO − JSON (ms)')
    ax.set_title('Trial duration difference')

    ax = axes[1, 0]
    ax.plot(iti_idx, dio_iti,  label='DIO',  lw=1)
    ax.plot(iti_idx, json_iti, label='JSON', lw=1, alpha=0.8)
    ax.set_xlabel('Gap index (after trial N)')
    ax.set_ylabel('ITI (s)')
    ax.set_title('Inter-trial interval')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(iti_idx, iti_diff, color='C3', lw=1)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Gap index (after trial N)')
    ax.set_ylabel('DIO − JSON (ms)')
    ax.set_title('ITI difference')

    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'dio_json_comparison.png', dpi=150)
    plt.show()


def show_vstim_fraction(trial_start, trial_end, fs, task_start_sample, task_end_sample, save_dir=None):
    """
    Print and plot the fraction of the recording occupied by visual stimulation.

    Parameters
    ----------
    trial_start       : np.ndarray  rising-edge sample indices (0-based, zeroed)
    trial_end         : np.ndarray  falling-edge sample indices (0-based, zeroed)
    fs                : float       sampling frequency in Hz
    task_start_sample : int         absolute sample index of recording start (from params)
    task_end_sample   : int         absolute sample index of recording end   (from params)
    """
    total_samples  = task_end_sample - task_start_sample
    total_sec      = total_samples / fs

    vstim_on_samples = np.sum(trial_end - trial_start)
    vstim_on_sec     = vstim_on_samples / fs
    vstim_off_sec    = total_sec - vstim_on_sec

    frac_on  = vstim_on_sec  / total_sec
    frac_off = vstim_off_sec / total_sec

    print("\n=== VStim Fraction of Whole Recording ===")
    print(f"  Total recording : {total_sec:.1f} s  ({total_samples} samples @ {fs:.0f} Hz)")
    print(f"  VStim ON        : {vstim_on_sec:.1f} s  ({frac_on*100:.1f}%)")
    print(f"  VStim OFF       : {vstim_off_sec:.1f} s  ({frac_off*100:.1f}%)")
    print(f"  Number of trials: {len(trial_start)}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [frac_on, frac_off],
        labels=[f'VStim ON\n{frac_on*100:.1f}%', f'VStim OFF\n{frac_off*100:.1f}%'],
        colors=['#4C72B0', '#C7C7C7'],
        startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
    )
    ax.set_title(f'VStim fraction  (total {total_sec:.0f} s, {len(trial_start)} trials)')
    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'vstim_fraction.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    from DiscriminationTask.grating.task_params import rec_folder, task_file, sortout_folder, din_channel, task_start, task_end

    # Load DIO trial timestamps
    dio_folders = DIO.get_dio_folders(rec_folder)
    trial_pd_time, trial_pd_state = DIO.concatenate_din_data(dio_folders, din_channel)
    trial_pd_time = trial_pd_time.ravel() - trial_pd_time.ravel()[0]
    trial_pd_state = trial_pd_state.ravel()
    trial_start = trial_pd_time[trial_pd_state == 1]
    trial_end   = trial_pd_time[trial_pd_state == 0][1:]

    # Load JSON trial parameters
    trial_params = get_trial_params(task_file)

    if len(trial_start) != len(trial_params):
        print(f"WARNING: DIO has {len(trial_start)} trials, JSON has {len(trial_params)} trials")
        n = min(len(trial_start), len(trial_params))
        trial_start  = trial_start[:n]
        trial_end    = trial_end[:n]
        trial_params = trial_params[:n]
        print(f"Comparing first {n} trials only.")

    # Get sampling frequency
    fs = load_sorting_analyzer(Path(sortout_folder) / 'curated_analyzer').sorting.sampling_frequency

    save_dir = Path(sortout_folder) / 'behavior_analysis'
    save_dir.mkdir(exist_ok=True)

    timing_result = compare_dio_json_timing(trial_start, trial_end, trial_params, fs)
    plot_dio_json_comparison(timing_result, save_dir=save_dir)

    show_vstim_fraction(trial_start, trial_end, fs, task_start, task_end, save_dir=save_dir)
