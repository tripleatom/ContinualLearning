# BOSC spindle comparison

`spindle_bosc.py` lives beside `detect_v1_spindles.py`; outputs live beside the
band-power files in `low_freq/spindle_bosc/`. The legacy outputs are untouched.

```powershell
python sleep/spindles/spindle_bosc.py --animal CnL46 --dates 260729 260730
```

The paired comparison reads each `low_freq/v1_spindles/*_v1_spindles.json` to
reuse its exact representative channel, LFP file and saved NREM state file.
Thus it tests a different detector, not a different channel-selection method.
`--shanks 0 1 2` limits shanks. `--overwrite` explicitly replaces existing BOSC
results. `--fs` defaults to 250 Hz; use the saved LFP rate for native-rate work.

## Verified sources and attribution

Turnbull C, Prostebby M, Babale I, Pagliardini S, Dickson C (2026).
[Improved sleep spindle detection in rats using the BOSC method: A comparison
with the traditional automated approach](https://doi.org/10.1016/j.jneumeth.2026.110812).
The publisher full text was inaccessible; the implementation was checked against
the authors' public MATLAB source preview, **Mendeley version 3**, the version
linked by the paper: <https://data.mendeley.com/datasets/hzw6zpzp9j/3>.
Later dataset versions exist; this port does not silently substitute them.

Verified files (dataset file UUIDs):

- `BOSC_Spindle_Detector.m`: `9f6f81cb-78cc-4560-b9ac-c01173270f41`, dated May 15, 2026.
- `BOSC_bgfit.m`: `4a25ef64-8b94-4521-81cd-867088371748`.
- `BOSC_tf.m`: `424a1cc0-626f-4a9d-af30-c89b3a3b4462`.
- `BOSC_detect.m`: `a4c3f8c3-48c6-4ad1-9c32-b60c86f021f7`.
- `BOSC_thresholds.m`: `74da156f-1868-49d8-99fc-70e60a75515f`.

The dataset lists CC BY 4.0; the BOSC2 source headers specifically carry
GPL-3.0-or-later and copyright 2010 Jeremy B. Caplan, Adam M. Hughes,
Tara A. Whitten and Clayton T. Dickson. This Python adaptation is designated
GPL-3.0-or-later; see <https://www.gnu.org/licenses/gpl-3.0.html>.
Changes include Python/NumPy/SciPy implementation, segmented NREM processing,
robust-fit implementation, explicit timestamps, export, and comparison plots.
No warranty is provided. No MATLAB source was executed.

## Parameters taken from released code

- Six-cycle Morlet transform, with Gaussian support ±3.6 temporal SD.
- Background frequency grid: `2**(arange(21)/4)`, 1–32 Hz.
- Detection bins: MATLAB `F(12:18)`, approximately
  6.73, 8, 9.51, 11.31, 13.45, 16, 19.03 Hz. Comments in the source mentioning
  8–16 or 12–16 are inconsistent with its executable frequency indices.
- Optimized background: convert temporal log-medians to geometric-mean power,
  initial log–log fit, remove power above the estimated 99.9th percentile,
  repeat the log-median estimate and robustly refit across frequency.
- Threshold: `chi2.ppf(0.99, 2) * geometric_background / exp(digamma(1)+log(2))`.
  The geometric-mean correction is essential; using background/2 is incorrect.
- Power must remain above threshold for **five cycles at a given frequency**;
  only then are detections combined across the seven bins.
- Merge gaps ≤0.1 seconds; retain events ≥0.35 seconds. The released code has no
  three-second maximum duration, so this implementation does not add one.

## Explicit adaptations and limits

- Saved LFP is resampled to 250 Hz with anti-aliasing, independently within each
  finite NREM run. No wavelet spans NREM/artifact gaps. A nonaligned run onset
  raises an error rather than silently shifting timestamps.
- Background and normalization use clean NREM only, using the same state files
  as the legacy comparison. The source script uses the signal supplied to it.
- Existing artifact masks replace the source's unfilled, data-specific
  `artifact_threshold_squared` parameter. No new amplitude threshold is guessed.
- Background excludes six seconds at either end of each run; detection excludes
  `(6+5)/6.73` seconds. Events truncated at invalid boundaries are rejected.
  Runs too short to support the analysis do not contribute background.
- Python uses leverage-adjusted Tukey bisquare IRLS (tuning constant 4.685).
  It has not been numerically compared with MATLAB's proprietary `robustfit`;
  this is a source-informed adaptation, not a claim of exact reproduction.
- Half-open intervals avoid BOSC_detect's extra endpoint sample, and absolute
  timestamps avoid the example script's extra record truncation and hard-coded
  11000-sample offset. Peak time/frequency is the largest power/threshold ratio
  within the event, a descriptive addition (not the source's cycle estimate).
- This broader-band BOSC default and the legacy 9–17 Hz detector differ in both
  frequency coverage and criteria. Overlap is a comparison, not accuracy.
- Existing provisional NREM labels, noisy contacts, and unknown V1 anatomy
  remain limitations. BOSC detections can still include non-spindle activity.

## Outputs

Each analyzed shank/session has event CSV (sleep-local and recording times),
metadata JSON, compact 50-Hz diagnostic NPZ, background/duration/timing summary
PNG and three amplitude-ranked example panels (10th/50th/90th percentiles).
Detection itself uses the analysis sampling rate, not the 50-Hz plot export.
Comparison tables give counts and overlap; legacy comparable counts include
only events fully inside the BOSC clean-NREM exposure. Cross-shank events are
not deduplicated. Tests use known synthetic bursts, stationary exponential
power, scale invariance, direct wavelet convolution and invalid-gap cases.
