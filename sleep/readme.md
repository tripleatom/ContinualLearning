# Sleep LFP Processing Pipeline

This directory extracts sleep LFP signals, computes time-frequency features,
scores NREM epochs, and optionally scores cortical UP/DOWN states.

## Pipeline

Run the active pipeline in this order:

1. `extract_sleep_lfp.py`
   Extract and preprocess 500 Hz LFP traces from NWB recordings.
2. `compute_sleep_spectrograms.py`
   Compute per-channel spectrograms from the extracted LFP.
3. `compute_sleep_features.py`
   Compute PC1 and delta, theta, sigma, gamma, and theta-ratio features.
4. `score_nrem_epochs.py`
   Score NREM epochs and select consolidated NREM windows.
5. `score_cortical_up_down_states.py`
   Score cortical UP/DOWN states inside a selected NREM window.

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
