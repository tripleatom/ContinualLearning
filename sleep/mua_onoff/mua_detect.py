"""MUA detection by reusing MountainSort5's spike detector.

This wraps ``mountainsort5.core.detect_spikes.detect_spikes`` — the exact
function that produces the "Detected N spikes" line in an MS5 run — so that
threshold events can be obtained without any of the downstream clustering.

The preprocessing mirrors ``_sort_shank`` in the SpikeSorting repo's
``spikesorting/MsSorting.py``: unit rescue -> artifact repair -> CMR ->
bandpass -> whitening. The artifact stage imports ``spikesorting.artifact_utils``
from that repo and uses the same on-disk cache format, so if a shank has already
been sorted, MUA detection reuses the cached artifact timestamps instead of
re-running detection.

The MS5 detector is a pure-python loop over threshold crossings, so it is run
here chunk-by-chunk with overlap instead of on the whole recording at once.
That keeps memory flat and lets long sessions run without loading all traces.
"""

import contextlib
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import spikeinterface as si
import spikeinterface.preprocessing as sp

from mountainsort5.core.detect_spikes import detect_spikes


# --------------------------------------------------------------------------
# Locating the SpikeSorting repo (same code, different path per machine)
# --------------------------------------------------------------------------

#: Checked in order, after ``$SPIKESORTING_REPO`` and any explicit argument.
SPIKESORTING_REPO_CANDIDATES = (
    Path('/Users/xiaorongzhang/Codes/SpikeSorting'),          # macOS
    Path(r'C:\Users\Windows\SpikeSorting'),                    # Windows
    Path.home() / 'Codes' / 'SpikeSorting',
    Path.home() / 'SpikeSorting',
)


def find_spikesorting_repo(repo_path: Union[str, Path, None] = None) -> Optional[Path]:
    """Return the SpikeSorting repo root, or None if it cannot be found.

    Resolution order: explicit argument, ``$SPIKESORTING_REPO``, then
    :data:`SPIKESORTING_REPO_CANDIDATES`. A directory counts as the repo only if
    it actually contains ``spikesorting/artifact_utils.py``.
    """
    candidates = []
    if repo_path is not None:
        candidates.append(Path(repo_path))
    env = os.environ.get('SPIKESORTING_REPO')
    if env:
        candidates.append(Path(env))
    candidates.extend(SPIKESORTING_REPO_CANDIDATES)

    for cand in candidates:
        try:
            if (cand / 'spikesorting' / 'artifact_utils.py').is_file():
                return cand.resolve()
        except OSError:  # unreadable / bad drive letter on the other platform
            continue
    return None


def _import_artifact_utils(repo_path: Union[str, Path, None] = None):
    """Import ``spikesorting.artifact_utils``, putting the repo on sys.path."""
    try:
        import spikesorting.artifact_utils as au  # already importable
        return au
    except ImportError:
        pass

    repo = find_spikesorting_repo(repo_path)
    if repo is None:
        raise ImportError(
            'Could not locate the SpikeSorting repo, which provides the artifact '
            'removal step. Pass repo_path=..., set the SPIKESORTING_REPO '
            'environment variable, or call detect_mua(..., remove_artifacts=False).'
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import spikesorting.artifact_utils as au
    return au


# --------------------------------------------------------------------------
# Artifact detection with the same cache format MsSorting.py uses
# --------------------------------------------------------------------------

#: The cache key is built from these, so sharing a cache with a sorting run
#: means matching the parameters that run actually used — not MsSorting.py's
#: code defaults, which are only what ``sorter_params.get(...)`` falls back to.
#: These mirror the ``sorter_params`` in the SpikeSorting repo's
#: ``pipeline_gui_settings.json``; ``global_stats_sample_batches`` in particular
#: is 3 there and ``None`` in the code default, and ``None`` means the global
#: statistics pass visits every batch instead of 3, which is both a different
#: key and several times slower.
ARTIFACT_PARAM_DEFAULTS = {
    'artifact_detection_method': 'rolling_std',
    'artifact_slope_threshold': 500,
    'artifact_rolling_window_size': 100,
    'artifact_rolling_z_threshold': 30,
    'artifact_time_batch_sec': 600,
    'artifact_use_global_stats': True,
    'artifact_global_stats_sample_batches': 3,
}


def _artifact_meta(rec: si.BaseRecording, artifact_params: dict) -> dict:
    """Build the cache-key dict exactly as MsSorting._load_or_compute_artifacts does."""
    p = {**ARTIFACT_PARAM_DEFAULTS, **(artifact_params or {})}
    return {
        'detection_method': p['artifact_detection_method'],
        'slope_threshold': p['artifact_slope_threshold'],
        'rolling_window_size': p['artifact_rolling_window_size'],
        'rolling_z_threshold': p['artifact_rolling_z_threshold'],
        'time_batch_sec': p['artifact_time_batch_sec'],
        'use_global_stats': p['artifact_use_global_stats'],
        'global_stats_sample_batches': p['artifact_global_stats_sample_batches'],
        'n_samples': int(rec.get_num_frames()),
        'n_channels': int(rec.get_num_channels()),
        'sampling_rate': float(rec.get_sampling_frequency()),
    }


def _key_from_meta(meta: dict) -> str:
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]


def artifact_cache_key(rec: si.BaseRecording,
                       artifact_params: Optional[dict] = None) -> str:
    """The 12-char hash MsSorting.py names its artifact cache files with."""
    return _key_from_meta(_artifact_meta(rec, artifact_params))


def has_artifact_cache(rec: si.BaseRecording,
                       cache_folder: Union[str, Path, None],
                       artifact_params: Optional[dict] = None) -> bool:
    """True if ``cache_folder`` already holds artifacts for exactly this recording.

    Lets a caller test a previous sorting run's folder before committing to it,
    so a miss costs a pair of ``stat`` calls rather than a full detection pass.
    Note the key covers the recording's frame count, so it only matches when
    ``rec`` spans the same samples the sorter saw — a frame-sliced epoch of a
    concatenated day will not match that day's cache.
    """
    if cache_folder is None:
        return False
    key = artifact_cache_key(rec, artifact_params)
    cache_dir = Path(cache_folder) / '_artifact_cache'
    return ((cache_dir / f'cache_meta_{key}.json').is_file()
            and (cache_dir / f'artifact_timestamps_{key}.npz').is_file())


def load_or_compute_artifacts(
    rec: si.BaseRecording, *,
    cache_folder: Union[str, Path, None] = None,
    artifact_params: Optional[dict] = None,
    repo_path: Union[str, Path, None] = None,
    dither: bool = True,
    dither_seed: Optional[int] = 0,
    correct_dc_offset: bool = False,
    verbose: bool = True,
):
    """Return (rec_repaired, artifact_timestamps, meta).

    ``cache_folder`` is the per-shank folder that MsSorting.py calls
    ``out_folder`` — i.e. ``<sortout>/<animal>/<animal>_<session>/shank<N>``.
    The ``_artifact_cache`` subfolder and the ``cache_meta_<hash>.json`` /
    ``artifact_timestamps_<hash>.npz`` names match, so pointing at a folder from
    a previous sorting run skips detection entirely. Pass None to always detect
    and write nothing.

    ``dither_seed`` is a deliberate departure from MsSorting.py, which leaves the
    repair RNG unseeded. The dither is drawn lazily inside ``get_traces``, so an
    unseeded generator makes event times shift slightly between otherwise
    identical runs. See :func:`preprocess_for_mua` for the full caveat.
    """
    au = _import_artifact_utils(repo_path)
    meta = _artifact_meta(rec, artifact_params)
    key = _key_from_meta(meta)

    meta_path = ts_path = None
    artifact_timestamps = None
    if cache_folder is not None:
        cache_dir = Path(cache_folder) / '_artifact_cache'
        meta_path = cache_dir / f'cache_meta_{key}.json'
        ts_path = cache_dir / f'artifact_timestamps_{key}.npz'
        if meta_path.exists() and ts_path.exists():
            with open(meta_path) as f:
                cached_meta = json.load(f)
            if cached_meta == meta:
                if verbose:
                    print(f'  artifact cache hit ({key}) — skipping detection')
                data = np.load(str(ts_path), allow_pickle=True)
                artifact_timestamps = [data[f'ch_{i:03d}'] for i in range(meta['n_channels'])]

    if artifact_timestamps is None:
        if verbose:
            print('  running artifact detection...')
        artifact_timestamps = au.detect_artifacts_recording(
            rec,
            detection_method=meta['detection_method'],
            slope_threshold=meta['slope_threshold'],
            rolling_window_size=meta['rolling_window_size'],
            rolling_z_threshold=meta['rolling_z_threshold'],
            time_batch_sec=meta['time_batch_sec'],
            use_global_stats=meta['use_global_stats'],
            global_stats_sample_batches=meta['global_stats_sample_batches'],
        )
        if ts_path is not None:
            ts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            np.savez(str(ts_path),
                     **{f'ch_{i:03d}': ts for i, ts in enumerate(artifact_timestamps)})
            if verbose:
                print(f'  saved artifact cache -> {ts_path.name}')

    rec_repaired = au.LazyArtifactRepairRecording(
        rec, artifact_timestamps, dither=dither, rng=dither_seed,
        correct_dc_offset=correct_dc_offset)
    return rec_repaired, artifact_timestamps, meta


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

@dataclass
class MuaEvents:
    """Threshold events found across a recording.

    times, channel_indices and amplitudes are parallel arrays sorted by time.
    ``amplitudes`` are read off the same (scaled) traces the detector saw, so
    with ``scale_mode='whiten'`` or ``'zscore'`` they are in noise-sigma units.
    """
    times: npt.NDArray[np.int64]
    channel_indices: npt.NDArray[np.int32]
    amplitudes: npt.NDArray[np.float32]
    channel_ids: npt.NDArray = field(default_factory=lambda: np.array([]))
    sampling_frequency: float = 0.0
    num_frames: int = 0
    params: dict = field(default_factory=dict)

    @property
    def times_sec(self) -> npt.NDArray[np.float64]:
        return self.times / self.sampling_frequency

    def per_channel_times(self):
        """Dict of channel_id -> event times in seconds."""
        return {
            cid: self.times[self.channel_indices == m] / self.sampling_frequency
            for m, cid in enumerate(self.channel_ids)
        }

    def rate_histogram(self, bin_size_sec: float = 0.010, per_channel: bool = False):
        """Binned MUA rate in Hz.

        Returns (bin_edges_sec, rate). ``rate`` is shape (n_bins,) when
        ``per_channel`` is False, else (n_channels, n_bins).
        """
        duration = self.num_frames / self.sampling_frequency
        edges = np.arange(0, duration + bin_size_sec, bin_size_sec)
        t = self.times_sec
        if not per_channel:
            counts, _ = np.histogram(t, bins=edges)
            return edges, counts / bin_size_sec
        n_ch = len(self.channel_ids)
        counts = np.zeros((n_ch, len(edges) - 1), dtype=np.float64)
        for m in range(n_ch):
            counts[m], _ = np.histogram(t[self.channel_indices == m], bins=edges)
        return edges, counts / bin_size_sec

    def save(self, path: Union[str, Path]):
        """Write events to a .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 times=self.times,
                 channel_indices=self.channel_indices,
                 amplitudes=self.amplitudes,
                 channel_ids=self.channel_ids,
                 sampling_frequency=self.sampling_frequency,
                 num_frames=self.num_frames,
                 params=json.dumps(self.params, default=str))
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> 'MuaEvents':
        d = np.load(str(path), allow_pickle=True)
        return MuaEvents(
            times=d['times'],
            channel_indices=d['channel_indices'],
            amplitudes=d['amplitudes'],
            channel_ids=d['channel_ids'],
            sampling_frequency=float(d['sampling_frequency']),
            num_frames=int(d['num_frames']),
            params=json.loads(str(d['params'])) if 'params' in d else {},
        )


# --------------------------------------------------------------------------
# Preprocessing — mirrors MsSorting._sort_shank
# --------------------------------------------------------------------------

def rescue_units(recording: si.BaseRecording, verbose: bool = True) -> si.BaseRecording:
    """Scale volts to uV when the traces peak below 1e-6 (MsSorting.py:159-164).

    Idempotent: a recording already in uV is returned unchanged, so it is safe
    to apply before artifact repair and again inside :func:`preprocess_for_mua`.
    """
    probe_n = min(int(recording.get_sampling_frequency()), recording.get_num_frames())
    traces_sample = recording.get_traces(start_frame=0, end_frame=probe_n)
    if np.abs(traces_sample).max() < 1e-6:
        if verbose:
            print('0. Data appears to be in volts (peak < 1e-6). Rescaling by 1e6...')
        return sp.scale(recording, gain=1e6)
    return recording


def preprocess_for_mua(
    recording: si.BaseRecording, *,
    remove_artifacts: bool = True,
    artifact_cache_folder: Union[str, Path, None] = None,
    artifact_params: Optional[dict] = None,
    artifact_dither: bool = False,
    artifact_dither_seed: Optional[int] = 0,
    artifact_correct_dc_offset: bool = False,
    spikesorting_repo: Union[str, Path, None] = None,
    common_reference: bool = True,
    freq_min: float = 300,
    freq_max: float = 6000,
    scale_mode: str = 'whiten',
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> si.BaseRecording:
    """Run the MsSorting.py preprocessing chain and return the recording.

    Stages, in the same order as ``MsSorting._sort_shank``:

    0. Unit rescue — if the raw traces peak below 1e-6 the recording is in volts
       rather than uV, so it is scaled by 1e6.
    1. Artifact detection and lazy PCHIP repair
       (``spikesorting.artifact_utils``), with cached timestamps.
    2. ``unsigned_to_signed`` when the dtype is unsigned, then global median CMR.
    3. Bandpass 300-6000 Hz, float32.
    4. Whitening.

    ``scale_mode`` controls the last stage:
      - ``'whiten'`` (default) — spikeinterface whitening, identical to the
        sorting pipeline, so ``detect_threshold=5.5`` means what it does there.
      - ``'zscore'`` — per-channel median/MAD scaling instead. No cross-channel
        mixing, which is often preferable for per-channel MUA because whitening
        can spread a large spike onto its neighbours.
      - ``'none'`` — stop after the bandpass; the threshold is then in uV.

    Deviations from MsSorting.py
    ----------------------------
    MS5 reads the whole recording in one ``get_traces()`` call, so it never has
    to care whether the chain gives the same answer under chunked reads. This
    module reads in chunks, which exposed two stages that are not chunk-invariant
    as configured there. Both defaults are changed here; pass the MsSorting.py
    value to restore its exact behaviour.

    ``artifact_correct_dc_offset=False`` (MsSorting.py: True)
        The repair shifts post-artifact samples to match the pre-artifact
        baseline *within the requested window*, so the same sample gets a
        different offset depending on where the read started. Measured on
        synthetic data this moved ~1% of detected events. It is safe to drop
        because the 300 Hz highpass two stages later removes the DC step the
        correction exists to suppress.

    ``artifact_dither=False`` (MsSorting.py: True)
        The repair fills gaps with Gaussian noise matched to the channel's noise
        floor, drawn lazily inside ``get_traces`` from a generator whose state
        advances per call — so the noise in a gap depends on how reads were
        chunked. Dither is there to stop the sorter seeing unnaturally flat
        stretches that would bias whitening and clustering. Threshold counting
        has no such concern, and synthetic noise in a repaired gap can only
        manufacture events, never recover real ones, so the smooth PCHIP
        interpolation is left alone.

    ``seed=0`` (MsSorting.py: unseeded)
        The whitening (or zscore) matrix is estimated from randomly drawn
        slices. Left unseeded, repeated runs on identical data give different
        event counts. ``artifact_dither_seed`` likewise seeds the dither if you
        turn it back on.

    With these defaults, chunked reads are bit-identical to a single-shot read
    provided ``chunk_pad_sec`` covers the bandpass filter's own margin — the
    default 0.1 s covers spikeinterface's 5 ms comfortably. Verified across
    chunk sizes of 3, 5, 7 and 11 s against ``chunk_duration_sec=None``.
    """
    if scale_mode not in ('whiten', 'zscore', 'none'):
        raise ValueError(f"scale_mode must be 'whiten', 'zscore' or 'none', got {scale_mode!r}")

    # 0. Unit rescue (MsSorting.py:159-164)
    rec = rescue_units(recording, verbose=verbose)

    # 1. Artifact detection + lazy repair
    if remove_artifacts:
        if verbose:
            print('1. Artifact detection / repair...')
        rec, _, _ = load_or_compute_artifacts(
            rec,
            cache_folder=artifact_cache_folder,
            artifact_params=artifact_params,
            repo_path=spikesorting_repo,
            dither=artifact_dither,
            dither_seed=artifact_dither_seed,
            correct_dc_offset=artifact_correct_dc_offset,
            verbose=verbose,
        )
    elif verbose:
        print('1. Skipping artifact removal (remove_artifacts=False) — any repair '
              'must already be applied to the recording passed in')

    # 2. CMR
    if rec.get_dtype().kind == 'u':
        rec = sp.unsigned_to_signed(rec)
    if common_reference:
        if verbose:
            print('2. Common median reference...')
        rec = sp.common_reference(rec, reference='global', operator='median')

    # 3. Bandpass
    if verbose:
        print(f'3. Bandpass filter ({freq_min}-{freq_max} Hz)...')
    rec = sp.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max, dtype=np.float32)

    # 4. Scaling to noise-sigma units
    if scale_mode == 'whiten':
        if verbose:
            print('4. Whitening...')
        rec = sp.whiten(rec, seed=seed)
    elif scale_mode == 'zscore':
        if verbose:
            print('4. Per-channel z-scoring (median+mad)...')
        rec = sp.zscore(rec, mode='median+mad', seed=seed)
    elif verbose:
        print('4. No scaling — detect_threshold is in raw units')

    return rec


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def detect_mua(
    recording: si.BaseRecording, *,
    detect_threshold: float = 5.0,
    detect_sign: int = -1,
    detect_time_radius_msec: float = 0.5,
    detect_channel_radius: Optional[float] = 0.0,
    preprocess: bool = True,
    scale_mode: str = 'whiten',
    remove_artifacts: bool = True,
    artifact_cache_folder: Union[str, Path, None] = None,
    artifact_params: Optional[dict] = None,
    artifact_dither: bool = False,
    artifact_dither_seed: Optional[int] = 0,
    artifact_correct_dc_offset: bool = False,
    spikesorting_repo: Union[str, Path, None] = None,
    common_reference: bool = True,
    freq_min: float = 300,
    freq_max: float = 6000,
    seed: Optional[int] = 0,
    chunk_duration_sec: Optional[float] = 60.0,
    chunk_pad_sec: float = 0.1,
    collapse_simultaneous: bool = False,
    verbose: bool = True,
) -> MuaEvents:
    """Detect MUA threshold events using MountainSort5's detector.

    Detection parameters mirror ``ms5.Scheme1SortingParameters``; preprocessing
    parameters are passed through to :func:`preprocess_for_mua`, which
    reproduces the MsSorting.py chain.

    detect_sign
        -1 negative-going peaks (the usual choice for extracellular MUA),
        +1 positive, 0 both (detector works on -|trace|).
    detect_time_radius_msec
        An event must be the most extreme sample within +/- this window on its
        own channel and on every channel inside ``detect_channel_radius``.
        Doubles as the refractory period, so 0.5 ms caps one channel at 2 kHz.
    detect_channel_radius
        Radius in channel-location units for cross-channel peak suppression.
        ``0.0`` (default) compares each channel only against itself, giving
        independent per-channel MUA — one physical spike can then be counted on
        several channels. Set it to your inter-site spacing (or ``None`` for
        all channels, the MS5 default) to keep only the event on the channel
        where it is largest.
    collapse_simultaneous
        MS5's scheme 1 drops all but one event per identical frame after
        detection. That is there to keep isosplit happy, and it throws away
        genuinely simultaneous spikes on distant channels, so it is off here.
    chunk_duration_sec
        Detection is run on chunks of this length so memory stays flat. ``None``
        loads the whole recording at once, the way MS5 itself does — exact, but
        needs num_samples * num_channels * 4 bytes of RAM.
    chunk_pad_sec
        Context loaded on each side of a chunk and then discarded. Must cover
        both the detector's ``detect_time_radius_msec`` neighbourhood and the
        bandpass filter's internal margin (5 ms in spikeinterface); the 0.1 s
        default clears both, making chunked output bit-identical to a full read.
    """
    if recording.get_num_segments() > 1:
        recording = si.concatenate_recordings(recording_list=[recording])

    if preprocess:
        rec = preprocess_for_mua(
            recording,
            remove_artifacts=remove_artifacts,
            artifact_cache_folder=artifact_cache_folder,
            artifact_params=artifact_params,
            artifact_dither=artifact_dither,
            artifact_dither_seed=artifact_dither_seed,
            artifact_correct_dc_offset=artifact_correct_dc_offset,
            spikesorting_repo=spikesorting_repo,
            common_reference=common_reference,
            freq_min=freq_min,
            freq_max=freq_max,
            scale_mode=scale_mode,
            seed=seed,
            verbose=verbose,
        )
    else:
        rec = recording

    fs = rec.get_sampling_frequency()
    N = rec.get_num_frames()
    M = rec.get_num_channels()
    channel_locations = np.asarray(rec.get_channel_locations(), dtype=np.float32)

    time_radius = int(math.ceil(detect_time_radius_msec / 1000 * fs))
    # Context discarded after each chunk. It has to satisfy two consumers: the
    # detector needs time_radius + 1 samples to see the same neighbourhood a
    # single-shot run would, and the lazy bandpass needs its own filter margin
    # (5 ms in spikeinterface) or the filtered values near a chunk edge drift.
    pad = max(time_radius + 1, int(chunk_pad_sec * fs))
    if chunk_duration_sec is None:
        chunk_size = N
    else:
        chunk_size = max(int(chunk_duration_sec * fs), 10 * pad)

    all_times, all_channels, all_amps = [], [], []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        load_start = max(0, start - pad)
        load_end = min(N, end + pad)
        traces = np.asarray(rec.get_traces(start_frame=load_start, end_frame=load_end),
                            dtype=np.float32)

        # detect_spikes drops events within margin_left/right of the array
        # edges. Use those margins to drop the overlap region instead: only
        # events belonging to [start, end) survive, so chunks never double-count.
        margin_left = start - load_start
        margin_right = load_end - end
        if traces.shape[0] <= margin_left + margin_right:
            continue

        # detect_spikes prints its adjacency table on every call.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            times, channel_indices = detect_spikes(
                traces=traces,
                channel_locations=channel_locations,
                time_radius=time_radius,
                channel_radius=detect_channel_radius,
                detect_threshold=detect_threshold,
                detect_sign=detect_sign,
                margin_left=margin_left,
                margin_right=margin_right,
                verbose=False,
            )

        if len(times):
            amps = traces[times, channel_indices]
            all_times.append(times.astype(np.int64) + load_start)
            all_channels.append(channel_indices)
            all_amps.append(amps)

        if verbose:
            n = int(sum(len(t) for t in all_times))
            print(f'  MUA detect: {end / fs:8.1f} s / {N / fs:.1f} s  ({n} events)')

    if all_times:
        times = np.concatenate(all_times)
        channel_indices = np.concatenate(all_channels)
        amplitudes = np.concatenate(all_amps).astype(np.float32)
        order = np.argsort(times, kind='stable')
        times, channel_indices, amplitudes = times[order], channel_indices[order], amplitudes[order]
    else:
        times = np.array([], dtype=np.int64)
        channel_indices = np.array([], dtype=np.int32)
        amplitudes = np.array([], dtype=np.float32)

    if collapse_simultaneous and len(times):
        keep = np.concatenate([[0], np.nonzero(np.diff(times) > 0)[0] + 1])
        times, channel_indices, amplitudes = times[keep], channel_indices[keep], amplitudes[keep]

    if verbose:
        duration = N / fs
        print(f'Detected {len(times)} MUA events over {duration:.1f} s '
              f'({len(times) / duration:.1f} Hz across {M} channels)')

    return MuaEvents(
        times=times,
        channel_indices=channel_indices.astype(np.int32),
        amplitudes=amplitudes,
        channel_ids=np.asarray(rec.get_channel_ids()),
        sampling_frequency=fs,
        num_frames=N,
        params={
            'detect_threshold': detect_threshold,
            'detect_sign': detect_sign,
            'detect_time_radius_msec': detect_time_radius_msec,
            'detect_channel_radius': detect_channel_radius,
            'collapse_simultaneous': collapse_simultaneous,
            'preprocess': preprocess,
            'scale_mode': scale_mode if preprocess else None,
            'remove_artifacts': remove_artifacts if preprocess else False,
            'common_reference': common_reference if preprocess else False,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'seed': seed,
            'artifact_dither': artifact_dither if preprocess and remove_artifacts else False,
            'artifact_dither_seed': artifact_dither_seed,
            'artifact_correct_dc_offset': artifact_correct_dc_offset,
            'chunk_duration_sec': chunk_duration_sec,
            'chunk_pad_sec': chunk_pad_sec,
        },
    )


def detect_mua_from_traces(
    traces: npt.NDArray[np.float32], *,
    channel_locations: npt.NDArray[np.float32],
    sampling_frequency: float,
    detect_threshold: float = 5.0,
    detect_sign: int = -1,
    detect_time_radius_msec: float = 0.5,
    detect_channel_radius: Optional[float] = 0.0,
):
    """Single-shot detection on an in-memory (num_samples, num_channels) array.

    Traces are used as-is — filter and scale them yourself first. Returns
    (times, channel_indices, amplitudes).
    """
    traces = np.asarray(traces, dtype=np.float32)
    time_radius = int(math.ceil(detect_time_radius_msec / 1000 * sampling_frequency))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        times, channel_indices = detect_spikes(
            traces=traces,
            channel_locations=np.asarray(channel_locations, dtype=np.float32),
            time_radius=time_radius,
            channel_radius=detect_channel_radius,
            detect_threshold=detect_threshold,
            detect_sign=detect_sign,
            margin_left=time_radius,
            margin_right=time_radius,
            verbose=False,
        )
    amps = traces[times, channel_indices] if len(times) else np.array([], dtype=np.float32)
    return times, channel_indices, amps
