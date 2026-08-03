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
3. `extract_sleep_lfp.py`
   Extract and preprocess 500 Hz LFP traces from NWB recordings.
4. `compute_sleep_spectrograms.py`
   Compute per-channel spectrograms from the extracted LFP.
5. `compute_sleep_features.py`
   Compute PC1 and delta, theta, sigma, gamma, and theta-ratio features.
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

- Window: 1,024 samples, approximately 2.0 seconds at 500 Hz
- Overlap: 75%
- FFT size: 2,048, approximately 0.24 Hz frequency resolution
- Time resolution: approximately 0.5 seconds
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
