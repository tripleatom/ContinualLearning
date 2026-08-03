"""Population firing rate for a sleep session, from the shank's spike sorting.

Used by plot_sleep_spectrograms.py to draw a "population rate" trace panel
underneath the spectrogram, so slow-wave / UP-DOWN structure can be read
against actual spiking rather than only against LFP band power.

Time base
---------
Sorting runs on the WHOLE day's concatenated recording, so spike sample
indices share the coordinate system of `start_sample` / `end_sample` in
sleep_day_configs.json. The sleep pipeline's `lfp_time`, by contrast, starts
at 0 at the sleep window (compute_sleep_features.py builds it as
`arange(n) / sampling_rate` over the already-sliced window). Everything
returned here is therefore shifted by `start_sample / fs` so it can be
plotted directly against `lfp_time`.

Curation
--------
Prefers a curated `phy/` export when one exists (dropping units labelled
'noise', like compute_sleep_mua.py). Falls back to the raw sorting analyzer,
which is uncurated - the returned info dict says which was used and how many
units contributed, so the figure can label it honestly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def animal_folder_name(session_name: str) -> str:
    """Sortout's per-animal folder for a session name.

    Sortout is keyed by the session's own prefix, which is not always the
    registry's `animal`: CnL42 days sort under "CnL42SG". Splitting the date
    off the session name gets both right ("CnL46_20260727" -> "CnL46",
    "CnL42SG_20260324" -> "CnL42SG").
    """
    return session_name.rsplit("_", 1)[0]


def find_sorting_folder(session_name: str, shank: int, sortout_roots) -> Path | None:
    """Newest sorting_results_* folder for one shank, across the sortout roots."""
    animal = animal_folder_name(session_name)
    for root in sortout_roots:
        shank_dir = Path(root) / animal / session_name / f"shank{shank}"
        try:
            if not shank_dir.is_dir():
                continue
            # Names carry a sort-friendly timestamp (sorting_results_YYYYMMDD_HHMM_*),
            # so the last one is the most recent run.
            candidates = sorted(p for p in shank_dir.glob("sorting_results_*") if p.is_dir())
        except OSError:
            continue
        if candidates:
            return candidates[-1]
    return None


def _load_spike_trains(sorting_folder: Path):
    """Spike trains (in samples) per unit, preferring a curated phy export.

    Imported lazily: spikeinterface is a heavy import and only this code path
    needs it, so plotting still works in an environment without it.
    """
    phy_folder = sorting_folder / "phy"
    if phy_folder.is_dir():
        from spikeinterface.extractors import read_phy

        sorting = read_phy(phy_folder)
        quality = sorting.get_property("quality")
        unit_ids = sorting.unit_ids
        if quality is not None:
            unit_ids = unit_ids[np.asarray(quality) != "noise"]
        return sorting, unit_ids, "phy (curated, non-noise units)"

    analyzer_sorting = sorting_folder / "sorting_analyzer" / "sorting"
    if analyzer_sorting.is_dir():
        import spikeinterface as si

        sorting = si.load(analyzer_sorting)
        return sorting, sorting.unit_ids, "sorting_analyzer (UNCURATED, all units)"

    raise FileNotFoundError(
        f"No phy export or sorting_analyzer under {sorting_folder}"
    )


def population_rate(
    session_name: str,
    shank: int,
    start_sample,
    end_sample,
    fs: float,
    sortout_roots,
    bin_size_sec: float = 1.0,
    smooth_sigma_bins: float = 2.0,
):
    """Population firing rate over one sleep window, on the lfp_time base.

    Returns (time_sec, rate_spikes_per_sec, info) or (None, None, info) when
    no sorting exists for this shank - a missing sorting is a normal state
    (not every day is sorted), so the caller can skip the panel and carry on.
    """
    info = {"shank": int(shank), "session_name": session_name}

    sorting_folder = find_sorting_folder(session_name, shank, sortout_roots)
    if sorting_folder is None:
        info["status"] = "no sorting_results_* folder found"
        return None, None, info
    info["sorting_folder"] = str(sorting_folder)

    try:
        sorting, unit_ids, source = _load_spike_trains(sorting_folder)
    except Exception as exc:                      # unreadable/partial sorting
        info["status"] = f"could not load sorting: {exc}"
        return None, None, info
    info["source"] = source
    info["n_units"] = int(len(unit_ids))
    if len(unit_ids) == 0:
        info["status"] = "sorting has no usable units"
        return None, None, info

    spike_fs = float(sorting.sampling_frequency)
    all_spikes = [np.asarray(sorting.get_unit_spike_train(u), dtype=np.int64)
                  for u in unit_ids]
    spikes = np.sort(np.concatenate(all_spikes)) if all_spikes else np.array([], np.int64)
    info["n_spikes_total"] = int(spikes.size)

    # Window bounds in the sorting's own (whole-day) sample space.
    lo = 0 if start_sample is None else int(start_sample)
    hi = int(end_sample) if end_sample is not None else int(spikes[-1]) if spikes.size else lo
    if hi <= lo:
        info["status"] = f"empty sleep window [{lo}, {hi})"
        return None, None, info

    in_window = spikes[(spikes >= lo) & (spikes < hi)]
    info["n_spikes_in_window"] = int(in_window.size)
    if in_window.size == 0:
        info["status"] = "no spikes inside the sleep window"
        return None, None, info

    # Shift to lfp_time (0 at the sleep window start) and bin.
    rel_sec = (in_window - lo) / spike_fs
    duration = (hi - lo) / fs
    edges = np.arange(0.0, duration + bin_size_sec, bin_size_sec)
    counts, _ = np.histogram(rel_sec, bins=edges)
    rate = counts / bin_size_sec

    if smooth_sigma_bins and smooth_sigma_bins > 0:
        from scipy.ndimage import gaussian_filter1d

        rate = gaussian_filter1d(rate, sigma=float(smooth_sigma_bins))

    time_sec = edges[:-1] + bin_size_sec / 2.0
    info.update(
        {
            "status": "ok",
            "bin_size_sec": float(bin_size_sec),
            "smooth_sigma_bins": float(smooth_sigma_bins),
            "mean_rate": float(np.mean(rate)),
            "max_rate": float(np.max(rate)),
            "spike_fs": spike_fs,
        }
    )
    return time_sec, rate, info
