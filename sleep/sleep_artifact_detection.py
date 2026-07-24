"""
sleep_artifact_detection.py
===========================
Shared helpers for broadband-artifact detection and artifact-aware
normalization, used by plot_sleep_spectrograms.py (Option B masking) and
score_nrem_epochs.py (consolidated-NREM segmentation).

A motion / EMG / cable artifact lifts power across ALL frequencies at once,
unlike a real brain state which has a spectral tilt. We therefore measure a
"broadband level" (mean power across frequency, in dB) per spectrogram bin and
flag robust outliers. The same broadband level doubles as a movement/EMG-like
proxy for sleep scoring (high broadband -> moving/awake).
"""
import numpy as np
from scipy.ndimage import binary_dilation


def broadband_level(spec, freqs, fmax=None):
    """Mean power across frequency, in dB, per time bin.

    spec  : (n_freqs, n_times) linear power for one channel.
    fmax  : if given, only average frequencies <= fmax.
    Returns (n_times,) array. High values => broadband power (movement/artifact).
    """
    fmask = np.ones_like(freqs, bool) if fmax is None else (freqs <= fmax)
    return np.mean(10 * np.log10(spec[fmask] + 1e-12), axis=0)


def robust_z(x):
    """Median/MAD robust z-score of a 1D array."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)


def detect_broadband_artifacts(spec_ch, freqs, times, n_mad=5.0,
                               dilate_sec=5.0, fmax=None):
    """Flag spectrogram time bins saturated across ALL frequencies.

    Returns (mask, z) on the spectrogram time base: mask True for artifact bins,
    z the robust z-score of broadband power.
    """
    bb = broadband_level(spec_ch, freqs, fmax=fmax)
    z = robust_z(bb)
    mask = z > n_mad
    if dilate_sec and dilate_sec > 0 and len(times) > 1:
        dt = float(np.median(np.diff(times)))
        k = max(1, int(round(dilate_sec / dt)))
        mask = binary_dilation(mask, np.ones(2 * k + 1, bool))
    return mask, z


def mask_to_spans(times, mask):
    """Convert a boolean mask to a list of (t_start, t_end) span tuples."""
    spans = []
    if not np.any(mask):
        return spans
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    for s, e in zip(starts, ends):
        spans.append((times[s], times[min(e, len(times) - 1)]))
    return spans


def znorm_masked(x, bad):
    """Z-normalize x using only non-artifact samples, then NaN the bad ones."""
    x = np.asarray(x, dtype=float).copy()
    good = ~bad
    if np.any(good):
        mu = np.nanmean(x[good])
        sd = np.nanstd(x[good])
    else:
        mu, sd = np.nanmean(x), np.nanstd(x)
    z = (x - mu) / (sd + 1e-10)
    z[bad] = np.nan
    return z
