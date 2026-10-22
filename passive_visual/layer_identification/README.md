# Passive visual layer profiles

`plot_layers.py` creates one large PNG per shank, combining these analyses
as two aligned panels: power, and black LFP traces over the CSD color map.
The three analyses are:

1. Whole-passive-period 500–5000 Hz power, matching the frequency band and
   single black depth-profile presentation in Senzai et al. (2019), Fig. 1C.
   Native-rate Welch estimation uses 2-second Hann windows, 50% overlap,
   per-window constant detrending, and 0.5-Hz bins. Integrate PSD over the
   band within each window, then take the median over windows as the band
   power (µV²), so brief large events do not dominate; the mean and the
   per-window values are saved too. The displayed profile is
   (power − floor) / (top − floor), with floor and top the 5th and 95th
   percentiles over accepted contacts of the shank, so neither the pedestal
   shared by all contacts nor one hot contact sets the scale. The panel title
   gives floor and top in µV rms; check them, because a flat shank is
   stretched to the same 0..1 range. A contact beside a large unit runs past
   1 and is kept. Contacts are also rejected on this power: more than
   `--power-outlier-ratio` (default 2) times below the median of at least
   three accepted contacts within 50 µm, or that far above it with the
   9–14 kHz noise floor raised alike. This is our explicit
   spectral estimator, not a claim to reproduce their unspecified estimator.
   All passive recording periods are included: stimuli, ITIs and idle time.
   Welch uses complete windows; a final remainder shorter than one second
   is not used. Windows remain on the same global grid across read batches.
2. Trial-averaged 0.5–100 Hz LFP, displayed from 0 to +0.2 seconds
   (`VIEW_SEC` in `plot_layers.py`).
   Saved averages retain the full −0.5-second-to-stimulus-offset window.
   Subtract each trial/contact's pre-stimulus mean before averaging. Plot
   every retained contact as a depth-offset trace with its original channel
   number. All contacts within a shank share one voltage gain; a labeled
   voltage bar is shown per shank. Display gain is 6×; added depth margins
   keep the edge traces visible. Excluded channels have no trace.
3. CSD proxy: negative second spatial derivative of the averaged LFP,
   in µV/mm². No tissue conductivity is assumed. Negative values are sinks.
   Average aligned trials separately for every channel first; retain the
   channel/depth dimension. The resulting mean LFP traces are the CSD input.
   First apply three-contact Hamming smoothing of the LFP with weights
   [0.23, 0.54, 0.23], separately within each contiguous contact run. Then
   take the negative second derivative. Smoothing replicates the values
   at each run's endpoints; derivative endpoints remain masked (no Vaknin
   extrapolation). Contacts adjacent to missing-contact gaps are masked.
   Both unsmoothed and smoothed CSD arrays are saved.

   The combined LFP/CSD panel shows 0..+0.2 seconds, with both signals sharing
   the same time and depth coordinates. Black LFP traces appear above the colors.
   Colors use the `jet` map: blue for negative sinks, green for zero,
   and red for positive sources. Each shank uses a symmetric limit equal to the 99th percentile
   of absolute CSD over the displayed window; the physical limit is printed
   over its panel. Colors express fractions of that limit, so colors
   across shanks do NOT imply equal physical amplitudes. Linear depth
   interpolation is only for display, within contiguous supported runs.
   Numerical estimates are neither interpolated nor clipped.

The voltage reference is preserved as recorded. Raw voltage is converted to
µV using NWB conversion/offset metadata, resampled to 2 kHz with an anti-alias
polyphase filter. LFP uses a third-order Butterworth filter applied forward
and backward; PSD uses native-rate voltage (30 kHz here), without that LFP
bandpass or 2-kHz resampling, so the full 5-kHz upper limit is preserved.
LFP batches include three seconds of real-data padding on
each side where available. Trial alignment is rounded to the nearest 0.5 ms.
Known bad channels are matched to the original channel number after the colon
in NWB electrode labels, not to the local electrode table index.

Probe depth is plotted with 0 µm at the top and increasing depth downward,
following the confirmed orientation that 0 is closer to the brain surface.
Coordinates retain their NWB values; zero is not assumed to be exactly at pia.
LFP traces retain positive voltage upward. These are exploratory profiles, not automatic
L1–L6 assignments. Trials are pooled across stimulus conditions, with no
additional trial artifact rejection. The CSD calculation assumes a single contact column
per shank and does not interpolate across excluded contacts.

## Verified first dataset: CnL46, July 27, 2026

Day directory:
`\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\260727`

- NWBs: `CnL46_20260727/CnL46_20260727sh{0..7}.nwb`.
- Raw passive DIO: `CnL46_20260727/CnL46_passive_20260727_130929.rec/`
  `CnL46_passive_20260727_130929.DIO/`
  `CnL46_passive_20260727_130929.dio_Controller_Din3.dat`.
- Processed timing: `CnL46_drifting_grating_exp_20260727_131003_DIO.npz`.
- Stimulus log: `CnL46_drifting_grating_exp_20260727_131003.txt`.
- Bad-channel list: `CnL46_20260727/bad_channels.txt`.

The stimulus log and processed DIO contain 1,600 trials with 1-second stimuli.
The raw DIO clock is 30 kHz. Its first timestamp is 1,356,414; the first raw
rising edge is 1,492,345. Their difference is 135,931 samples, matching the
first processed onset. `conversion_list.txt` places passive first in the
NWB concatenation, with 79,782,737 samples, so `--offset-sec 0` applies.
Processed offsets are exactly onset + 30,000 samples; therefore the plots use
the existing cleaned timing and nominal 1-second stimulus duration, rather
than reconstructing each raw photodiode falling edge. The repository's DIO
workflow can repair edges; that provenance is retained rather than treating
all processed times as independently measured raw edges.

## Run

From the repository root, in an environment with NumPy, SciPy, h5py, Matplotlib:

```powershell
python passive_visual/layer_identification/plot_layers.py
python passive_visual/layer_identification/test_plot_layers.py
```

Results default to `<nwb-dir>/layer_identification/`, alongside the session's
NWB files (not inside this repository). For this session that is
`\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\260727\CnL46_20260727\layer_identification`.
The folder contains `shank{N}_layer_profiles.png` for each shank (no PDFs),
one NPZ per shank containing
mean LFP, raw/smoothed CSD, full PSD, mean/normalized band densities, recording
bounds, depths, channel labels and trial timing, plus JSON parameters.
`--reuse` reuses saved shank arrays if the analysis parameters match, allowing
a changed output location. Version-1 LFP averages can be reused, but their
trial-power results are discarded and the continuous PSD is recomputed.
Rerun without `--reuse`
after changing any source data. `--shanks 0 1`, `--pre-sec`, `--nwb-dir`,
`--dio`, `--dio-fs`, `--offset-sec`, `--bad-channels`, and `--output` permit
other sessions. Verify digital and recording clock origins before changing
sessions. The full passive recording bounds are inferred from the unique
passive entry in `conversion_list.txt`, or can be set explicitly with
`--passive-start-sample` and `--passive-end-sample` (exclusive). This session
uses samples [0, 79782737) at 30 kHz: 2659.4246 seconds.
Variable-duration trials are rejected and must be grouped first.

## Other sessions: CnL46, July 28–30, 2026

Passive is first in each day's `conversion_list.txt`, so `--offset-sec 0`
applies. All three days have 3,200 logged trials (1-second stimuli, 32
conditions) with a 0.2-second ITI, against 0.5 seconds on July 27. They were
run with `--pre-sec 0.2`, so the baseline is the ITI alone; it still follows
the previous stimulus offset by only 0.2 seconds. None of these days has a
`bad_channels.txt`; the July 27 list was passed with `--bad-channels`.

- July 28 and 29: the reviewed `CnL46_drifting_grating_exp_*_DIO.npz`.
- July 30: no reviewed timing file exists. The stimulus program stalled for
  logged trials 2575–2758 (stimuli stayed on for about 6 seconds), and one
  later photodiode pulse is split in two. `dio_from_raw.py` reads the raw
  `Din3` photodiode file and keeps the 3,014 pulses whose width is within
  0.03 seconds of the median; the result is saved in that session's
  `layer_identification/CnL46_passive_20260730_DIO_clean_pulses.npz`.

## CSD references and interpretation

[Schaefer et al., 2015, Frontiers in Neural Circuits](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00052/pdf)
describes spatial Hamming smoothing and linear interpolation for laminar
CSD visualization. Our weights follow that method; our 25-µm probe spacing
differs from their 100-µm spacing, so the physical smoothing scale differs.
We retain missing-data gaps instead of reconstructing those regions.

[Allen SDK CSD guidance](https://allensdk.readthedocs.io/en/latest/allensdk.brain_observatory.ecephys.ecephys_session.html)
also describes second-derivative CSD and recommends spatial smoothing for
display. Color polarity is a plotting convention, explicitly labeled here.
Our normalization and display window are visualization choices, not a universal
literature standard. The second-derivative formula in the initial version
was standard; its unsmoothed estimate and shared color scale made local
channel discontinuities dominate the display. Smoothing does not establish
that such discontinuities are physiological. In particular, inspect shank 1
channel 4 at y=325 µm before assigning a cortical layer.

## Trial-count sensitivity

Run `python -B passive_visual/layer_identification/compare_trial_counts.py`
to compare 100 and 400 trials with the saved 1,600-trial averages. Ten seeded
random subsets are drawn without replacement, with each 100-trial subset
nested inside its corresponding 400-trial subset. The same trial selections
are used for every shank. Filtering, baseline, bad channels, and spatial
smoothing are unchanged. A single pass through each NWB accumulates all
subset means and verifies that the recomputed full mean matches the saved
reference within 1e-8 µV.

Outputs go to the session's `layer_identification/trial_count_comparison/`:
one six-panel PNG per shank (LFP and CSD for each trial count, showing repeat
1), subset LFP arrays and zero-based selected trial indices in NPZ files,
`metrics.csv`, and `summary.json`. Each shank uses a shared voltage gain and
shared CSD color scale across counts. Power is unchanged because it uses the
whole passive recording. No PDFs are produced. `--reuse` redraws/recomputes
metrics from the subset cache; rerun without it if source data change.

For another session pass the same `--nwb-dir`, `--dio`, `--bad-channels`
and `--pre-sec` given to `plot_layers.py`, which must run first: its saved
full average is the reference. Every figure carries a stamp at the bottom
with its inputs, parameters and generation time.

Metrics include flattened time-by-depth correlation, RMSE, and RMSE divided
by the RMS of the full average, over −0.1..0.5 s and 0..0.3 s (fixed metric windows,
independent of the displayed 0..0.2 s). Missing CSD
depths are excluded. The full reference includes the selected trials, so
this measures convergence toward that session mean, not independent
replication. Random subsets pool stimulus conditions, as does the original
analysis; this is not a condition-stratified comparison or a test of drift.

## Temporal-filter sensitivity (July 27)

`python -B passive_visual/layer_identification/compare_lfp_filters.py` compares
0.5–100 Hz, 0.5–300 Hz, and 300-Hz low-pass without an additional high-pass.
All use third-order Butterworth forward/backward filtering after anti-alias
resampling to 2 kHz, with identical batches and real-data padding. Every
trial/channel is baseline-subtracted over −0.5..0 s and all 1,600 trials
are averaged. The original 0.5–100 Hz mean must reproduce the saved result
within 1e-8 µV. The original rejection/filling mask stays fixed across
filters, as do spatial smoothing and the trial set. Acquisition filters
remain present in the low-pass-only condition.

Results are saved only to the session's `layer_identification/filter_comparison/`.
Each shank has one PNG with power plus three LFP/CSD overlays, using the
same LFP gain and CSD color limits across filters. Display gain is 6× and
the displayed window is 0..0.2 s. The shared color limit is the pooled 99th
percentile of absolute CSD across filters over this window. Numerical arrays
are not clipped. NPZs retain full −0.5..1-s LFP/CSD, masks and trial timing;
`metrics.csv` compares each variant to 0.5–100 Hz over 0..0.2 s and −0.1..0.5 s.
RMSE is in µV for LFP and µV/mm² for CSD; relative RMSE divides by reference
RMS. `validation.json` records reference agreement and shared scales.

The comparison also computes native-rate 500–5000 Hz power. Afterward,
`plot_layers.py --reuse` can reuse these same-session power caches to update
the ordinary two-panel figures, retaining the original 0.5–100 Hz LFP until
a filter is chosen. `compare_lfp_filters.py --reuse` reuses both caches;
rerun without reuse after changing inputs or extraction parameters.
