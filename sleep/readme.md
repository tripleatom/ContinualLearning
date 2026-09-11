# Sleep LFP Processing Pipeline

This directory extracts sleep LFP signals, computes time-frequency features,
scores NREM epochs, and optionally scores cortical UP/DOWN states.

## Pipeline

Run the active pipeline in this order:

1. `video_ephys_sync.py`
   Match the video PROC sync pulse train to the .rec DIO, writing
   `sync_times_{pre,post}.pkl`. `--match-algorithm` selects how:
   `pulse` for a fixed-frequency square wave (CnL42 days), `pulse_geo` for a
   random-interval train (CnL46 days onward), which anchors the two trains by
   treating a run of consecutive intervals as a barcode and then fits
   SG->PROC as a line so the ~15 ppm clock drift between the camera and the
   ephys box is absorbed. The default `auto` picks per session from the
   spread of the PROC intervals, so no day needs the flag set by hand.
2. `proc_func_velocity.py`
   Compute tracking velocity for each session. `VELOCITY_SOURCE` selects the
   tracked point: `proc_center` (the `_PROC` file's own head centre) or
   `dlc_body` (centroid of `VELOCITY_KEYPOINTS` from the `_DLC.hdf5`
   companion file, default the five trunk points). Each source writes its own
   pkl, and `plot_sleep_spectrograms.py` loads whichever is selected.

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
   the spectrogram window) - in `score_nrem_delta_velocity.py` for the NREM
   speed gate and in `plot_sleep_spectrograms.py` for the plotted panel. Nothing
   smooths the speed after that.
3. `extract_sleep_lfp.py`
   Extract and preprocess 1250 Hz LFP traces from NWB recordings.
4. `compute_sleep_spectrograms.py`
   Compute per-channel spectrograms from the extracted LFP, reduced to a
   log-spaced 1-100 Hz frequency axis before saving.
5. `compute_sleep_features.py`
   Compute PC1 and delta, sigma, gamma, and theta-ratio features, all
   derived from the saved spectrogram (the LFP traces are not re-read).
6. `plot_sleep_spectrograms.py`
   Create per-channel spectrogram figures and trace-data exports.

Then, optionally:

- `score_nrem_epochs.py`
  Score NREM epochs and select consolidated NREM windows.
- `score_cortical_up_down_states.py`
  Score cortical UP/DOWN states inside a selected NREM window.

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
detection is implemented in `sleep_artifact_detection.py`.

`legacy_score_sleep_stages.py` contains the older, disconnected
NREM/REM/Wake implementation. `legacy_sleep_lfp.py` contains the older
standalone LFP extraction and plotting workflow.

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

- `plot_sleep_spectrograms.py`: create spectrogram and feature summaries
- `plot_processed_sleep_lfp.py`: regenerate plots from exported trace-data files
- `review_sleep_artifacts.py`: inspect artifact removal results
- `compute_sleep_mua.py`: compute and plot sleep multi-unit activity

Generated data filenames have intentionally not been renamed, preserving
compatibility with existing recordings and downstream analyses.
