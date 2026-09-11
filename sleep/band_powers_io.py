"""Reading ``*_all_shanks_band_powers.pkl``, in either of its two layouts.

compute_sleep_features.py originally stored every envelope - delta, theta
ratio, sigma, gamma and PC1 - at the full 500 Hz LFP rate in float64, plus a
verbatim copy of the spectrogram that already lives in the sibling
``*_sh<N>_spectrograms.npz``. Each envelope is a moving average over
``band_params['smoothing_window']`` seconds, so it carries nothing above
~1/window Hz and 500 Hz was roughly 5000x more samples than the signal
supports; the files reached ~31 GB per sleep epoch, two thirds of it those
envelopes. They are now written on the spectrogram time base (``band_time``,
~2 Hz) in float32, with the spectrogram left in its npz - about 900x smaller
for the same information.

Both layouts are readable through the helpers here, so days computed before
the change keep working without being recomputed:

    LEGACY   'lfp_time' (500 Hz), 'spectrograms' in the pickle,
             band_powers[ch] holds delta/theta_ratio/theta_ratio_num/
             theta_ratio_den/sigma/gamma as float64
    COMPACT  'band_time' (~2 Hz) + 'band_fs', no 'spectrograms',
             band_powers[ch] holds delta/theta_ratio/sigma/gamma as float32

``sampling_rate`` means the same thing in both - the 500 Hz LFP rate the
envelopes were COMPUTED at - so anything that needs the LFP rate itself
(sample-indexed masks, for one) keeps reading that field. What changed is the
rate the envelopes are STORED at, which is what ``band_time`` / ``band_fs``
describe.
"""
import numpy as np
from pathlib import Path


def is_compact(shank_data):
    """True when this shank was written in the compact layout."""
    return "band_time" in shank_data


def band_time(shank_data):
    """Seconds-from-epoch-start for each stored envelope sample."""
    key = "band_time" if is_compact(shank_data) else "lfp_time"
    return np.asarray(shank_data[key], dtype=np.float64)


def band_fs(shank_data):
    """Sample rate of the stored envelopes, Hz.

    NOT ``shank_data['sampling_rate']`` - that is the 500 Hz LFP rate the
    envelopes were computed at, which is also the stored rate only in the
    legacy layout.
    """
    if "band_fs" in shank_data:
        return float(shank_data["band_fs"])
    time = band_time(shank_data)
    if time.size < 2:
        return float(shank_data.get("sampling_rate", 0.0))
    return 1.0 / float(np.median(np.diff(time)))


def spectrogram_npz_for(band_powers_path, shank):
    """The ``*_sh<N>_spectrograms.npz`` beside a band-powers pickle."""
    path = Path(band_powers_path)
    stem = path.name.replace("_all_shanks_band_powers.pkl", "")
    return path.parent / f"{stem}_sh{shank}_spectrograms.npz"


def load_spectrograms(shank_data, band_powers_path, shank):
    """The ``(n_channels, n_freqs, n_times)`` spectrogram for one shank.

    Taken from the pickle when it is there (legacy), otherwise read from the
    npz that compute_sleep_spectrograms.py wrote - the same array either way,
    it is simply no longer duplicated into the pickle.
    """
    if "spectrograms" in shank_data:
        return np.asarray(shank_data["spectrograms"])
    path = spectrogram_npz_for(band_powers_path, shank)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The band-power pickle no longer carries a copy "
            f"of the spectrogram, so this file is required - re-run "
            f"compute_sleep_spectrograms.py for shank {shank}.")
    with np.load(path, allow_pickle=False) as data:
        return data["spectrograms"]


def n_epochs(time, epoch_sec):
    """How many whole `epoch_sec` windows, counted from t=0, `time` covers.

    The last sample stands for the interval it starts, so one sample interval
    is added before counting: a 500 Hz grid ending at 1199.998 s covers 300
    whole 4 s epochs, not 299, which keeps legacy files reducing to exactly
    the same number of epochs as the old reshape-by-count did.
    """
    time = np.asarray(time, dtype=np.float64)
    if time.size == 0 or epoch_sec <= 0:
        return 0
    step = float(np.median(np.diff(time))) if time.size > 1 else 0.0
    return int(np.floor((float(time[-1]) + step) / float(epoch_sec)))


def epoch_times(epoch_sec, count):
    """Centre time of each epoch, on the same clock as `band_time`."""
    return (np.arange(count) + 0.5) * float(epoch_sec)


def epoch_average(values, time, epoch_sec):
    """Mean of `values` within consecutive `epoch_sec` windows.

    Epochs are anchored at t=0 - the start of the sleep epoch, the clock the
    velocity, sync and MUA stages all use - and assigned from the timestamps
    rather than from a fixed sample count. A fixed count is exact only when
    the sample interval divides the epoch: at 0.512 s bins in a 4 s epoch it
    would round 7.8125 samples to 8 and drift 2.4%, minutes of slip across a
    long session. Working from `time` is exact at any rate, and reduces to the
    old behaviour for a legacy 500 Hz file.

    Epochs with no samples come back NaN; the trailing partial epoch is dropped.
    """
    values = np.asarray(values, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    length = min(values.size, time.size)
    values, time = values[:length], time[:length]

    count = n_epochs(time, epoch_sec)
    if count == 0:
        return np.array([])

    index = np.floor(time / float(epoch_sec)).astype(np.int64)
    keep = (index >= 0) & (index < count)
    totals = np.bincount(index[keep], weights=values[keep], minlength=count)
    counts = np.bincount(index[keep], minlength=count).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)
