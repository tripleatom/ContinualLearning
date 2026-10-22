# Sleep LFP Processing Pipeline

This directory extracts sleep LFP signals, computes time-frequency features,
scores NREM epochs, and optionally scores cortical UP/DOWN states.

## Layout

| Folder | Contents |
|---|---|
| `sleep/` | Per-day setup and shared config: `set_sleep_day.py`, `sleep_pipeline_gui.py`, `sleep_pipeline_config.py`, `sleep_day_registry.py`, `sleep_day_configs.json`, `sleep_paths.py` |
| `sync/` | Video-ephys sync and tracking velocity |
| `lfp/` | LFP extraction, spectrograms, band powers, artifacts, plots |
| `staging/` | `score_sleep_states.py`: NREM / REM / WAKE / QUIET scoring |
| `mua_onoff/` | Channel MUA detection and ON/OFF (UP/DOWN) states |
| `spindles/` | V1 spindle detection, BOSC comparison, validation |

Scripts import each other by bare module name. Each script in a subfolder starts
with `import sleep_paths`, which puts `sleep/`, its subfolders and the repo root
on `sys.path`, so run them as `python sleep/<folder>/<script>.py` from anywhere.

## Pipeline

Run the active pipeline in this order:

1. `sync/video_ephys_sync.py`
   Match the video PROC sync pulse train to the .rec DIO, writing
   `sync_times_{pre,post}.pkl`. `--match-algorithm` selects how:
   `pulse` for a fixed-frequency square wave (CnL42 days), `pulse_geo` for a
   random-interval train (CnL46 days onward), which anchors the two trains by
   treating a run of consecutive intervals as a barcode and then fits
   SG->PROC as a line so the ~15 ppm clock drift between the camera and the
   ephys box is absorbed. The default `auto` picks per session from the
   spread of the PROC intervals, so no day needs the flag set by hand.
2. `sync/proc_func_velocity.py`
   Compute tracking velocity for each session. `VELOCITY_SOURCE` selects the
   tracked point: `proc_center` (the `_PROC` file's own head centre) or
   `dlc_body` (centroid of `VELOCITY_KEYPOINTS` from the `_DLC.hdf5`
   companion file, default the five trunk points). Each source writes its own
   pkl, and `lfp/plot_sleep_spectrograms.py` loads whichever is selected.

   Jitter is removed from **position**, not from speed: the track is resampled
   onto a uniform grid (the camera's frame rate is not fixed), low-passed at
   `VELOCITY_CUTOFF_HZ` with a zero-phase Butterworth, differentiated once, and
   only then turned into a speed magnitude. Order matters - `sqrt(vx^2+vy^2)`
   rectifies zero-mean jitter into a positive speed offset that no later
   averaging can remove. Frames further than `VELOCITY_MAX_GAP_SEC` from a real
   detection stay NaN rather than being interpolated across, because a long
   dropout interpolates to a straight line and a straight line reads as "still".

   Downstream, the speed is reduced exactly once more, on a
   `VELOCITY_WINDOW_SEC` window (10 s, derived from `spec_params` so it matches
   the spectrogram window) - in `staging/score_sleep_states.py` for WAKE/stationary
   separation and in `lfp/plot_sleep_spectrograms.py` for the plotted panel.
   Nothing smooths the speed after that.
3. `lfp/extract_sleep_lfp.py`
   Extract and preprocess 1250 Hz LFP traces from NWB recordings.
4. `lfp/compute_sleep_spectrograms.py`
   Compute per-channel spectrograms from the extracted LFP, reduced to a
   log-spaced 1-100 Hz frequency axis before saving.
5. `lfp/compute_sleep_features.py`
   Compute PC1 and delta, sigma, gamma, and theta-ratio features, all
   derived from the saved spectrogram (the LFP traces are not re-read).
6. `lfp/plot_sleep_spectrograms.py`
   Create per-channel spectrogram figures and trace-data exports.

Then, optionally:

- `staging/score_sleep_states.py`
  Canonical four-state scorer. Per-channel delta/gamma scores vote within each
  shank, shanks vote at probe level, velocity separates WAKE from stationary
  periods, and theta separates REM from QUIET. It also selects consolidated
  NREM windows. The 60 s minimum NREM bout is explicitly a reviewable heuristic:
  every run reports the retained bouts and duration at 0, 30, and 60 s.
- `mua_onoff/find_sleep_mua.py` followed by `mua_onoff/detect_off_states.py`
  Detect channel-level MUA and score cortical ON/OFF states. These ON windows
  are the canonical state input for reactivation analyses.

## Running it

`python sleep_pipeline_gui.py` runs steps 1-6 from one panel: pick the animal
and recording day, the sessions (both / pre / post) and the shanks, see a
preflight of what is already on disk, then run any subset of the stages with
their output streamed into the log. It writes `ACTIVE_ANIMAL`, `ACTIVE_DATE`,
`SESSION_FILTER`, `shanks` and `VELOCITY_SOURCE` into
`sleep_pipeline_config.py` before each run, so running the stage scripts by
hand afterwards uses the same settings.

Per-day paths live in `sleep_day_configs.json`, keyed `<animal>_<date>` (e.g.
`CnL42_260324`) so two animals recorded on the same date stay separate. An
animal-day that isn't registered there has no paths for any stage; register it
with `python set_sleep_day.py [--animal CnL42] [--date YYMMDD]`, which the GUI
offers to launch whenever the selected pair is unknown.

Shared settings live in `sleep_pipeline_config.py`. Broadband artifact
detection is implemented in `lfp/sleep_artifact_detection.py`.

The former `reactivation_config.py` was removed because it described a separate
four-script pipeline that was not present in the repository. Reactivation now
uses the registered day metadata and the `SleepBlock` contract in `reactivation/sleep_apply/sleep_states.py`.

## Spectrogram features

- Window: 12,500 samples = 10 seconds at 1250 Hz
- Step: 1 second (saved spectrogram sampled at 1 Hz)
- FFT size: 12,500, 0.1 Hz native frequency resolution
- Saved frequency axis: 100 log-spaced bins spanning 1-100 Hz (linear rows
  averaged into each bin; the lowest few bins interpolated)
- Per-channel processing across each configured shank
- Compressed NumPy output for downstream feature computation

Each shank's spectrogram file contains:

```text
spectrograms: (n_channels, n_frequencies, n_times)
freqs: frequency axis
times: time axis
channel_ids: per-channel identifiers
sampling_rate
start_time
spec_params
```

## Supporting and review scripts

- `lfp/plot_sleep_spectrograms.py`: create spectrogram and feature summaries
- `lfp/plot_processed_sleep_lfp.py`: regenerate plots from exported trace-data files
- `lfp/review_sleep_artifacts.py`: inspect artifact removal results
- `mua_onoff/find_sleep_mua.py`: compute sleep multi-unit activity
- `mua_onoff/detect_off_states.py`: detect per-shank and probe-wide ON/OFF windows

`reactivation/sleep_apply/sleep_states.py` joins these completed outputs to decoder
results without re-estimating state from the sorted units being tested.

Generated data filenames have intentionally not been renamed, preserving
compatibility with existing recordings and downstream analyses.

## V1 spindle localization and events

After extracting LFP and running `staging/score_sleep_states.py`, run from the repository
root (replace `5` with a shank independently identified as V1):

```powershell
python sleep/spindles/detect_v1_spindles.py --shank 5
```

This uses the active day and pre/post selection in `sleep_pipeline_config.py`.
To use explicit files instead:

```powershell
python sleep/spindles/detect_v1_spindles.py --shank 5 --lfp PATH_TO_lfp_traces.npz --states PATH_TO_sleep_periods.pkl --out-dir spindle_results
```

Use `--depth-range MIN MAX` to restrict the cortical portion of the shank and
`--exclude-channels ID ID` for known bad contacts (original IDs, not column indices).
Probe y-coordinate is preserved; it does not establish cortical layer, depth from
pia, or the spindle generator. Confirm V1 and cortical extent independently.

The detector filters each valid, artifact-free NREM run at 9–17 Hz, trims one
second at both edges, and smooths Hilbert amplitude over 100 ms. It requires at
least 10 seconds of retained NREM. Channel power is the mean squared smoothed
amplitude in that NREM. Flat/nonfinite contacts and broadband RMS outliers beyond
10 times or below one tenth of the median are excluded. This basic QC cannot
identify every noisy contact; inspect the output and exclude known bad channels.

Power is smoothed over three depth-ordered contacts without crossing excluded
channels or physical gaps above `--max-depth-gap-um` (default 50 µm). Candidate
regions need at least `--min-region-channels` (default 3) contacts above 60% of
the maximum smoothed power. The highest-mean region supplies its strongest
contact for event detection. If none qualifies, the output reports no region.
These relative criteria identify a candidate region, not evidence of a spindle
by themselves, and can select a region even when no events pass.

Events use NREM amplitude z-scores: 1.5 SD boundaries, a 3 SD peak, gaps up to
0.2 seconds merged, and total duration 0.3–3 seconds. These options are exposed
as `--boundary-threshold`, `--peak-threshold`, `--merge-gap`, `--min-duration`,
and `--max-duration`. Events touching excluded data are rejected. Defaults are
implementation heuristics, not a reproduction of Takigawa et al.'s detector.

Outputs under `low_freq/v1_spindles/` (or `--out-dir`) are:

- CSV: event onset/peak/offset in sleep-local and whole-recording seconds,
  duration, peak z-score/power, representative contact ID and probe coordinate.
- NPZ: all contact IDs/depths, QC, NREM power, selected region, representative
  column (`best_index=-1` if none), and peak-triggered depth × time power.
- PNG: depth power profile and event-triggered power relative to each contact's
  NREM mean. Only fully valid ±1.5-second windows enter the triggered average.
- JSON: source files, selection result, parameters and retained NREM exposure.

Power retains the extracted LFP's units squared; it is not assumed to be µV².
Matching output filenames are replaced on rerun. Review the profile and events
before using this heuristic detector as a reactivation covariate.
