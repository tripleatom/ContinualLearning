r"""Pairwise visual-tuning similarity and per-unit covariates.

This is the "passive" half of the tuning-coupling analysis: everything that is
measured while the animal is awake and looking at gratings, reduced to

    * one similarity value per unit PAIR   (how alike is their tuning?)
    * a handful of nuisance covariates     (rate, position, cell type)

so that the sleep half (``tc_coupling.py``) only has to supply the coupling
value for the same pairs.

Two similarity measures, as specified
-------------------------------------
``delta_pref_deg``
    Absolute difference of preferred orientation, wrapped onto the orientation
    circle where 0 deg == 180 deg, so it lives in [0, 90].  0 = identical
    preference, 90 = orthogonal.  Preferred orientation is the vector-sum
    estimate already stored by GratingTuningCurve.

``signal_corr``
    Correlation of the two full tuning curves across orientations.  By default
    this is the CROSS-VALIDATED version: unit A's curve is built from one random
    half of the trials and unit B's from the *other* half, averaged over many
    splits (and over both assignments).  Using disjoint trials for the two units
    matters -- a correlation computed from the same trials is inflated by any
    trial-to-trial noise the pair shares, which is a co-fluctuation measure, i.e.
    exactly the thing the sleep half is supposed to predict.  The uncorrected
    version is also returned as ``signal_corr_raw``.

Which units enter
-----------------
Tuning similarity is only meaningful for units whose tuning curve is itself
trustworthy, so by default a unit must be visually responsive AND have a
significant split-half reliability (the statistics written by
GratingTuningCurve.calculate_tuning_curves).  Relax with ``min_*`` arguments.

Input
-----
``all_units_tuning.pkl`` as written by GratingTuningCurve.generate_tuning_curves:
``{unit_id: {'tuning': {...}, 'unit_info': {...}}}``.  The per-trial rates stored
there (``tuning['trial_rates']``) are what the cross-validated correlation
resamples, so no re-reading of spike data is needed.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


# Waveform trough-to-peak below this is called narrow-spiking (putative FS/PV).
# 0.45 ms is the usual mouse-cortex split; check the printed histogram before
# trusting it on a new probe, and override if the dip sits elsewhere.
FS_TROUGH_PEAK_MS = 0.45


@dataclass(frozen=True)
class UnitTuning:
    """Everything one unit contributes to the pair table."""

    unit_id: str
    shank: int | None
    x_um: float
    y_um: float
    pref_ori_deg: float
    osi: float
    mean_rates: np.ndarray          # tuning curve, one entry per orientation
    trial_rates: dict               # orientation -> per-trial evoked rates (Hz)
    responsive: bool
    reliable: bool
    split_half_r: float
    trough_peak_ms: float
    cell_type: str                  # 'narrow' | 'wide' | 'unknown'


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _waveform_trough_peak_ms(unit_info):
    """Trough-to-peak duration (ms) of the template, or NaN if unavailable.

    Measured as the time from the global minimum (trough) to the maximum that
    FOLLOWS it, which is the standard definition; a template whose peak precedes
    its trough (positive-going unit) returns NaN rather than a meaningless value.
    """
    wf = unit_info.get('waveform_template')
    t_ms = unit_info.get('waveform_t_ms')
    if wf is None or t_ms is None:
        return np.nan
    wf = np.asarray(wf, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    if wf.size < 3 or wf.size != t_ms.size or not np.all(np.isfinite(wf)):
        return np.nan
    trough = int(np.argmin(wf))
    if trough >= wf.size - 1:
        return np.nan
    peak = trough + int(np.argmax(wf[trough:]))
    if peak <= trough:
        return np.nan
    return float(t_ms[peak] - t_ms[trough])


def load_unit_tuning(tuning_pkl, require_responsive=True, require_reliable=True,
                     min_split_half_r=None, min_osi=None,
                     fs_threshold_ms=FS_TROUGH_PEAK_MS, verbose=True):
    """
    Read ``all_units_tuning.pkl`` and return ``{unit_id: UnitTuning}`` for the
    units that pass the tuning-quality gate.

    Args:
        tuning_pkl: path to all_units_tuning.pkl
        require_responsive / require_reliable: keep only units flagged as
            visually responsive / reliably tuned by the characterization step
        min_split_half_r: additional floor on the median split-half correlation
        min_osi: additional floor on OSI (off by default -- the hypothesis is
            about tuning SIMILARITY, which is defined for weakly tuned units too)
        fs_threshold_ms: trough-to-peak cut for narrow- vs wide-spiking
    """
    tuning_pkl = Path(tuning_pkl)
    with tuning_pkl.open('rb') as f:
        book = pickle.load(f)

    kept, dropped = {}, {'no_stats': 0, 'not_responsive': 0, 'not_reliable': 0,
                         'low_r': 0, 'low_osi': 0, 'no_curve': 0}

    for unit_id, entry in book.items():
        tuning = entry.get('tuning', entry)
        info = entry.get('unit_info', {}) or {}

        resp = tuning.get('responsiveness')
        rel = tuning.get('reliability')
        if (require_responsive or require_reliable) and (resp is None or rel is None):
            dropped['no_stats'] += 1
            continue
        if require_responsive and not resp.get('responsive', False):
            dropped['not_responsive'] += 1
            continue
        if require_reliable and not rel.get('reliable', False):
            dropped['not_reliable'] += 1
            continue
        split_r = float(rel.get('median_r', np.nan)) if rel else np.nan
        if min_split_half_r is not None and not (split_r >= min_split_half_r):
            dropped['low_r'] += 1
            continue
        if min_osi is not None and not (float(tuning.get('osi', np.nan)) >= min_osi):
            dropped['low_osi'] += 1
            continue

        mean_rates = np.asarray(tuning.get('mean_rates', []), dtype=float)
        if mean_rates.size < 3 or not np.all(np.isfinite(mean_rates)):
            dropped['no_curve'] += 1
            continue

        loc = info.get('channel_location_um') or (np.nan, np.nan)
        ttp = _waveform_trough_peak_ms(info)
        cell_type = ('unknown' if not np.isfinite(ttp)
                     else ('narrow' if ttp < fs_threshold_ms else 'wide'))

        kept[unit_id] = UnitTuning(
            unit_id=unit_id,
            shank=info.get('shank'),
            x_um=float(loc[0]) if np.isfinite(loc[0]) else np.nan,
            y_um=float(loc[1]) if np.isfinite(loc[1]) else np.nan,
            pref_ori_deg=float(tuning.get('preferred_orientation_deg', np.nan)),
            osi=float(tuning.get('osi', np.nan)),
            mean_rates=mean_rates,
            trial_rates={float(k): np.asarray(v, dtype=float)
                         for k, v in (tuning.get('trial_rates') or {}).items()},
            responsive=bool(resp.get('responsive', False)) if resp else False,
            reliable=bool(rel.get('reliable', False)) if rel else False,
            split_half_r=split_r,
            trough_peak_ms=ttp,
            cell_type=cell_type,
        )

    if verbose:
        print(f"Tuning units: kept {len(kept)} of {len(book)}")
        for reason, n in dropped.items():
            if n:
                print(f"  dropped ({reason}): {n}")
        types = {t: sum(1 for u in kept.values() if u.cell_type == t)
                 for t in ('narrow', 'wide', 'unknown')}
        print(f"  cell types (trough-to-peak < {fs_threshold_ms} ms = narrow): {types}")
    return kept


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def circular_orientation_difference(a_deg, b_deg):
    """|a - b| on the orientation circle (period 180 deg) -> [0, 90]."""
    diff = np.abs(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float)) % 180.0
    return np.minimum(diff, 180.0 - diff)


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else np.nan


def _half_curves(unit, orientations, n_splits, rng):
    """
    Draw ``n_splits`` pairs of half-trial tuning curves for one unit.

    Returns (A, B), each (n_splits, n_orientations); A and B use disjoint trials
    of the same orientation, so any pair of curves drawn from DIFFERENT units
    shares no trial and therefore no trial-to-trial noise.
    """
    n_ori = len(orientations)
    A = np.full((n_splits, n_ori), np.nan)
    B = np.full((n_splits, n_ori), np.nan)
    for k, ori in enumerate(orientations):
        rates = unit.trial_rates.get(float(ori))
        if rates is None or rates.size < 2:
            continue
        n = rates.size
        half = n // 2
        order = np.argsort(rng.random((n_splits, n)), axis=1)
        drawn = rates[order]
        A[:, k] = drawn[:, :half].mean(axis=1)
        B[:, k] = drawn[:, half:2 * half].mean(axis=1)
    return A, B


def pair_tuning_similarity(units, n_splits=200, cross_validated=True, seed=0,
                           verbose=True):
    """
    Tuning similarity for every pair of units.

    Args:
        units: {unit_id: UnitTuning} from load_unit_tuning
        n_splits: random half-splits used for the cross-validated correlation
        cross_validated: if False, only the plain curve correlation is computed
            (``signal_corr`` then equals ``signal_corr_raw``)

    Returns:
        dict with 'unit_ids' (ordered list) and, for each pair (i<j) in that
        order, arrays 'delta_pref_deg', 'signal_corr', 'signal_corr_raw',
        plus 'pair_index' as (n_pairs, 2) int array into unit_ids.
    """
    unit_ids = sorted(units)
    n_units = len(unit_ids)
    if n_units < 2:
        raise ValueError(f"Need at least 2 units with usable tuning, got {n_units}.")

    curves = np.array([units[u].mean_rates for u in unit_ids], dtype=float)
    n_ori = curves.shape[1]
    if any(units[u].mean_rates.size != n_ori for u in unit_ids):
        raise ValueError("Units disagree on the number of orientations.")

    prefs = np.array([units[u].pref_ori_deg for u in unit_ids], dtype=float)
    pairs = np.array(list(combinations(range(n_units), 2)), dtype=int)
    i, j = pairs[:, 0], pairs[:, 1]

    delta_pref = circular_orientation_difference(prefs[i], prefs[j])

    # Plain (noise-attenuated, noise-correlation-contaminated) curve correlation.
    centered = curves - curves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        unit_normed = centered / norms[:, None]
    raw_corr = np.einsum('ij,ij->i', unit_normed[i], unit_normed[j])
    raw_corr[(norms[i] == 0) | (norms[j] == 0)] = np.nan

    if not cross_validated:
        return {'unit_ids': unit_ids, 'pair_index': pairs,
                'delta_pref_deg': delta_pref,
                'signal_corr': raw_corr.copy(), 'signal_corr_raw': raw_corr,
                'n_splits': 0}

    # Cross-validated: correlate unit i's half-A curve with unit j's half-B
    # curve (and the mirror image), so the two curves never share a trial.
    rng = np.random.default_rng(seed)
    orientations = sorted(units[unit_ids[0]].trial_rates) or list(range(n_ori))
    if len(orientations) != n_ori:
        if verbose:
            print(f"  note: {len(orientations)} orientations in trial_rates vs "
                  f"{n_ori} in mean_rates -- falling back to raw signal correlation")
        return {'unit_ids': unit_ids, 'pair_index': pairs,
                'delta_pref_deg': delta_pref,
                'signal_corr': raw_corr.copy(), 'signal_corr_raw': raw_corr,
                'n_splits': 0}

    halves_A = np.empty((n_units, n_splits, n_ori))
    halves_B = np.empty((n_units, n_splits, n_ori))
    for k, uid in enumerate(unit_ids):
        halves_A[k], halves_B[k] = _half_curves(units[uid], orientations, n_splits, rng)

    def _normalize(block):
        block = block - np.nanmean(block, axis=2, keepdims=True)
        norm = np.sqrt(np.nansum(block ** 2, axis=2, keepdims=True))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(norm > 0, block / norm, np.nan)

    nA, nB = _normalize(halves_A), _normalize(halves_B)
    # mean over splits of 0.5 * (corr(A_i, B_j) + corr(B_i, A_j))
    cross = 0.5 * (np.nansum(nA[i] * nB[j], axis=2) + np.nansum(nB[i] * nA[j], axis=2))
    with np.errstate(invalid='ignore'):
        signal_corr = np.nanmean(cross, axis=1)

    if verbose:
        print(f"Pair tuning similarity: {pairs.shape[0]} pairs from {n_units} units")
        print(f"  delta preferred ori: median {np.nanmedian(delta_pref):.1f} deg")
        print(f"  signal corr (cross-validated, {n_splits} splits): "
              f"median {np.nanmedian(signal_corr):+.3f}  "
              f"[raw median {np.nanmedian(raw_corr):+.3f}]")

    return {'unit_ids': unit_ids, 'pair_index': pairs,
            'delta_pref_deg': delta_pref,
            'signal_corr': signal_corr, 'signal_corr_raw': raw_corr,
            'n_splits': int(n_splits)}


def pair_covariates(units, unit_ids, pair_index):
    """
    Nuisance covariates for each pair, in the same order as ``pair_index``.

    Geometry note: this probe's channel map gives x in {0, 300, 600, 900} um for
    EIGHT shanks, so shanks 0/4, 1/5, 2/6 and 3/7 share an x coordinate and a
    true inter-shank distance cannot be recovered from the map alone.  Rather
    than invent one, cross-shank separation enters as the pair of components that
    ARE defined -- ``abs_dy_um`` (depth) and ``abs_dx_um`` (map x) -- plus the
    categorical ``same_shank``; ``dist_um`` is filled in only for same-shank
    pairs, where it is simply the depth difference.
    """
    i, j = pair_index[:, 0], pair_index[:, 1]
    shank = np.array([units[u].shank if units[u].shank is not None else -1
                      for u in unit_ids], dtype=float)
    x = np.array([units[u].x_um for u in unit_ids], dtype=float)
    y = np.array([units[u].y_um for u in unit_ids], dtype=float)
    ctype = np.array([units[u].cell_type for u in unit_ids], dtype=object)

    same_shank = (shank[i] == shank[j]) & (shank[i] >= 0)
    abs_dy = np.abs(y[i] - y[j])
    abs_dx = np.abs(x[i] - x[j])
    dist = np.where(same_shank, abs_dy, np.nan)

    def _pair_type(a, b):
        if a == 'unknown' or b == 'unknown':
            return 'unknown'
        return {('wide', 'wide'): 'WW', ('narrow', 'narrow'): 'NN'}.get(
            (a, b), 'NW') if a == b else 'NW'

    pair_type = np.array([_pair_type(ctype[a], ctype[b]) for a, b in zip(i, j)],
                         dtype=object)

    return {
        'same_shank': same_shank.astype(float),
        'abs_dy_um': abs_dy,
        'abs_dx_um': abs_dx,
        'dist_um': dist,
        'pair_type': pair_type,
        'osi_geomean': np.sqrt(
            np.clip([units[u].osi for u in unit_ids], 0, None)[i] *
            np.clip([units[u].osi for u in unit_ids], 0, None)[j]),
    }
