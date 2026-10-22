r"""Is fine-timescale synchrony in NREM UP states stronger for similarly tuned pairs?

Runs the whole test for ONE session (the one in grating_config.py, or --animal/--date):

  1. visual tuning     -- per pair: preferred-orientation difference on the
                          orientation circle (0 deg == 180 deg, so 0-90) and the
                          cross-validated tuning-curve correlation
                          (tc_similarity.py)
  2. NREM UP states    -- MUA ON windows (detect_off_states.py) combined across
                          shanks, intersected with scored NREM, clock-checked
                          (tc_upstates.py)
  3. excess synchrony  -- spike pairs within +-w ms inside the same UP window,
                          minus the mean over interval-jitter surrogates that
                          keep every spike inside its own D-ms cell of UP time
                          (tc_jitter.py)
  4. the test          -- excess_norm ~ tuning similarity + firing rates +
                          recording distance + cell-type pair, with a p-value
                          from shuffling tuning across units (tc_stats.py)
  5. pre vs post       -- the same model on the within-pair change

Every (w, D) combination is run; one ("--headline") gets the detailed figure.

Usage
-----
    python run_tuning_coupling.py --check        # which inputs exist
    python run_tuning_coupling.py --self-test    # synthetic data, known answers
    python run_tuning_coupling.py                # full run, pre and post
    python run_tuning_coupling.py --date 260728 --coinc-ms 5 --jitter-ms 50
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'passive_visual' / 'FreelyMovingProcessing' / 'Grating'))
sys.path.insert(0, str(REPO / 'reactivation' / 'sleep_apply'))

import tc_similarity as sim          # noqa: E402
import tc_upstates as ups            # noqa: E402
import tc_jitter as tj               # noqa: E402
import tc_stats as st                # noqa: E402
from sleep_states import SleepBlock, load_mua_on_intervals, load_nrem_intervals   # noqa: E402


CONTROLS = ('log_rate_geomean', 'abs_log_rate_ratio', 'same_shank',
            'abs_dy_um', 'abs_dx_um', 'pair_type')
PREDICTORS = ('delta_pref_deg', 'signal_corr')
PRETTY = {'delta_pref_deg': 'Δ preferred orientation (deg)',
          'signal_corr': 'tuning-curve correlation (cross-validated)'}
#: A pair is flagged as a possible duplicate unit when this fraction of the
#: sparser unit's spikes has a partner within dup_ms; flagged pairs are kept in
#: the CSV but left out of every test.
DUP_FRACTION = 0.2
#: Δ-preferred-orientation groups for the jitter-corrected CCG figure.
PREF_GROUPS = ((0.0, 30.0), (30.0, 60.0), (60.0, 90.01))
# Categorical slots 1-3 of the reference palette (validated for all pairs).
COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def col(kind, w, d=None):
    """Column name of a per-combination measure, e.g. excess_norm_w5_D50."""
    return f"{kind}_w{w:g}" if d is None else f"{kind}_w{w:g}_D{d:g}"


# --------------------------------------------------------------------------- #
# Input discovery
# --------------------------------------------------------------------------- #

def session_paths(animal=None, date=None, sortout=None):
    """Resolve the session's folders (grating_config is the default)."""
    if sortout is None or animal is None or date is None:
        from grating_config import ANIMAL_ID, EXPERIMENT_DATE, SORTOUT_FOLDER
        animal = animal or ANIMAL_ID
        date = date or EXPERIMENT_DATE
        sortout = Path(sortout) if sortout else Path(SORTOUT_FOLDER)
    return {'animal': animal, 'date': date, 'sortout': Path(sortout)}


def experiment_data_day_folder(sortout, animal, date):
    """
    The experiment_data mirror of a sortout session folder.

        sortout:         sortout/<ANIMAL>/<ANIMAL>_20<DATE>
        experiment_data: experiment_data/<ANIMAL>/<DATE>/<ANIMAL>_20<DATE>

    MUA off_states output and *_sleep_periods.pkl live under the second tree.
    """
    sortout = Path(sortout)
    if 'sortout' not in sortout.parts:
        return None
    animal_folder = Path(str(sortout).replace('sortout', 'experiment_data', 1)).parent
    return animal_folder / date / f"{animal}_20{date}"


def _glob_real(folder, pattern):
    """glob() without macOS AppleDouble '._name' sidecars, which would otherwise
    sort first and be picked instead of the real pickle."""
    return sorted(p for p in Path(folder).glob(pattern) if not p.name.startswith('.'))


def discover_inputs(paths, verbose=True):
    """Locate every input; for a missing one, say what would create it."""
    sortout, date = paths['sortout'], paths['date']
    found, missing = {}, {}
    day_folder = experiment_data_day_folder(sortout, paths['animal'], date)
    roots = [sortout] + ([day_folder] if day_folder is not None else [])

    tuning = _glob_real(sortout, '**/all_units_tuning.pkl')
    if tuning:
        found['tuning'] = max(tuning, key=lambda p: p.stat().st_mtime)
    else:
        missing['tuning'] = "python GratingTuningCurve.py"

    for block in ('pre', 'post'):
        hits = _glob_real(sortout, f'sleep_spikes_*sleep_{block}.pkl')
        if hits:
            found[f'sleep_{block}'] = hits[0]
        else:
            missing[f'sleep_{block}'] = "python extract_sleep_session_spikes_multishank.py"

        mua = []
        for root in roots:
            mua += _glob_real(root, f'MUA/off_states/*global_on_windows*_{block}*.pkl')
            mua += _glob_real(root, f'MUA/off_states/*{block}*global_on_windows*.pkl')
        if mua:
            found[f'mua_{block}'] = mua[0]
        else:
            missing[f'mua_{block}'] = (f"python sleep/mua_onoff/find_sleep_mua.py && python "
                                       f"sleep/mua_onoff/detect_off_states.py --epochs {block}")

        # NREM scoring is one file per epoch, each on its own clock.
        nrem = []
        for root in roots:
            nrem += _glob_real(root, f'**/*{block}_sleep_periods.pkl')
        if nrem:
            found[f'nrem_{block}'] = nrem[0]
        else:
            missing[f'nrem_{block}'] = "python sleep/staging/score_sleep_states.py"

    if verbose:
        print(f"\nSession {paths['animal']} {date}   ({sortout}, exists: {sortout.exists()})")
        print("Found:")
        for key, path in found.items():
            print(f"  {key:12s} {path}")
        print("Missing:" + ("  (nothing)" if not missing else ""))
        for key, how in missing.items():
            tag = '[optional]' if key.startswith('nrem') else '[REQUIRED]'
            print(f"  {key:12s} {tag} {how}")
    return found, missing


# --------------------------------------------------------------------------- #
# One sleep block
# --------------------------------------------------------------------------- #

def prepare_block(sleep_pkl, unit_ids, mua_windows_pkl, nrem_pkl=None,
                  min_shank_fraction=0.5, min_up_ms=50.0, max_up_ms=2000.0,
                  min_alignment_ratio=1.25, allow_alignment_failure=False,
                  verbose=True):
    """
    Spikes of the tuned units plus the NREM UP windows they are analysed in.

    Returns (spikes, windows, meta, alignment_report).  Raises when the UP
    windows fail the clock check, rather than measuring synchrony in the wrong
    place.
    """
    spikes, meta = ups.load_sleep_spikes(sleep_pkl, unit_ids=unit_ids, verbose=verbose)
    if len(spikes) < 2:
        raise ValueError(f"{Path(sleep_pkl).name}: fewer than 2 tuned units present.")

    # NREM first: presleep/postsleep blocks are mostly wake, so the alignment
    # check has to compare UP against DOWN *within NREM*, not against wake.
    nrem = None
    if nrem_pkl is not None:
        nrem = load_nrem_intervals(nrem_pkl, require_bouts=True)
        if verbose:
            print(f"NREM windows ({Path(nrem_pkl).name}): {len(nrem)}, "
                  f"{float(np.sum(nrem[:, 1] - nrem[:, 0])) / 60:.1f} min")

    if meta['start_sample'] is None or meta['end_sample'] is None:
        raise ValueError(f"{Path(sleep_pkl).name} has no sleep sample window; cannot map MUA windows.")
    block = SleepBlock(label=str(meta['sleep_id'] or Path(sleep_pkl).stem),
                       sampling_frequency_hz=meta['fs'],
                       day_start_sample=int(meta['start_sample']),
                       day_end_sample=int(meta['end_sample']),
                       duration_sec=meta['duration_sec'])
    windows, on_info = load_mua_on_intervals(
        mua_windows_pkl, block, min_shank_fraction=min_shank_fraction,
        min_duration_sec=min_up_ms / 1000.0, max_duration_sec=max_up_ms / 1000.0)
    if verbose:
        total = float(np.sum(windows[:, 1] - windows[:, 0])) if len(windows) else 0.0
        print(f"UP windows from MUA ON states ({Path(mua_windows_pkl).name}):")
        print(f"  {on_info['n_shanks']} shanks, threshold {on_info['required_shanks_on']}/"
              f"{on_info['n_shanks']} simultaneously ON")
        if len(windows):
            print(f"  {len(windows)} windows, {total / 60:.1f} min "
                  f"({100 * total / max(meta['duration_sec'], 1e-9):.1f}% of block), "
                  f"median {1000 * np.median(windows[:, 1] - windows[:, 0]):.0f} ms")
        else:
            print("  no windows survived filtering")
    if nrem is not None:
        before = len(windows)
        windows = ups._intersect_windows(windows, nrem)
        windows = ups._clip_and_filter(windows, meta['duration_sec'],
                                       min_up_ms / 1000.0, max_up_ms / 1000.0)
        if verbose:
            print(f"  restricted to scored NREM: {before} -> {len(windows)} UP windows")
    if len(windows) == 0:
        raise ValueError("No UP windows survived filtering.")

    ratio, report = ups.check_up_alignment(
        spikes, windows, meta['duration_sec'], min_ratio=min_alignment_ratio,
        verbose=verbose, universe_windows=nrem)
    if not report.get('passed', False) and not allow_alignment_failure:
        raise ValueError(
            f"UP windows from {Path(mua_windows_pkl).name} do not line up with the "
            f"sorted spikes (in/out rate ratio {ratio:.2f} < {min_alignment_ratio}, "
            f"universe = {report.get('universe')}).")
    report['nrem_windows'] = nrem        # None when no NREM scoring was used
    return spikes, windows, meta, report


def build_pair_table(similarity, covariates, sync):
    """
    One row per tuned pair: tuning similarity, covariates, and every synchrony
    measure for every (w, D).  Pairs with a unit absent from this block are NaN.
    """
    unit_ids = similarity['unit_ids']
    pair_index = similarity['pair_index']
    row_of = {uid: k for k, uid in enumerate(sync['unit_ids'])}
    rows = np.array([row_of.get(uid, -1) for uid in unit_ids])
    i, j = pair_index[:, 0], pair_index[:, 1]
    ri, rj = rows[i], rows[j]
    usable = (ri >= 0) & (rj >= 0)

    def pull(matrix):
        out = np.full(len(i), np.nan)
        out[usable] = np.asarray(matrix, dtype=float)[ri[usable], rj[usable]]
        return out

    def per_unit(values):
        out = np.full(len(unit_ids), np.nan)
        out[rows >= 0] = np.asarray(values, dtype=float)[rows[rows >= 0]]
        return out

    n = per_unit(sync['n_spikes'])
    rate = per_unit(sync['rate_hz'])
    with np.errstate(divide='ignore', invalid='ignore'):
        log_geo = 0.5 * (np.log10(rate[i]) + np.log10(rate[j]))
        log_ratio = np.abs(np.log10(rate[i]) - np.log10(rate[j]))
        near_zero = pull(sync['near_zero'])
        dup_frac = near_zero / np.minimum(n[i], n[j])

    table = {
        'unit_i': np.array([unit_ids[k] for k in i], dtype=object),
        'unit_j': np.array([unit_ids[k] for k in j], dtype=object),
        'delta_pref_deg': similarity['delta_pref_deg'],
        'signal_corr': similarity['signal_corr'],
        'signal_corr_raw': similarity['signal_corr_raw'],
        'n_spikes_i': n[i], 'n_spikes_j': n[j],
        'rate_i_hz': rate[i], 'rate_j_hz': rate[j],
        'log_rate_geomean': log_geo, 'abs_log_rate_ratio': log_ratio,
        **covariates,
        'near_zero_frac': dup_frac,
        'possible_duplicate': (dup_frac > DUP_FRACTION).astype(float),
    }
    for w, observed in sync['observed'].items():
        table[col('observed', w)] = pull(observed)
    for (w, d), measures in sync['combos'].items():
        for kind, matrix in measures.items():
            table[col(kind, w, d)] = pull(matrix)
    return table


def testable(table, response):
    """The table with flagged duplicate pairs removed from ``response``."""
    work = dict(table)
    work[response] = np.where(np.asarray(table['possible_duplicate']) > 0, np.nan,
                              np.asarray(table[response], dtype=float))
    return work


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

#: Filled in by main() so every figure carries the command that produced it.
REPRO_STAMP = ''


def set_repro_stamp(argv, paths, found):
    global REPRO_STAMP
    inputs = '\n'.join(f'  {k}: {v}' for k, v in sorted(found.items()))
    REPRO_STAMP = (f"python run_tuning_coupling.py {' '.join(argv)}\n"
                   f"session {paths['animal']} {paths['date']}  |  {paths['sortout']}\n"
                   f"generated {datetime.now():%Y-%m-%d %H:%M} by "
                   f"reactivation/tuning_coupling/run_tuning_coupling.py\n"
                   f"inputs:\n{inputs}")


def stamp_figure(fig):
    if REPRO_STAMP:
        fig.text(0.005, -0.01, REPRO_STAMP, fontsize=5.5, color=INK_2,
                 ha='left', va='top', family='monospace')


def _style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=10)


def _profile(ax, x, y, edges, color, result, xlabel):
    xc, m, sem, n = st.binned_profile(x, y, edges)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.errorbar(xc, m, yerr=sem, color=color, marker='o', markersize=8,
                linewidth=2, capsize=4)
    for xi, mi, ni in zip(xc, m, n):
        if np.isfinite(mi):
            ax.annotate(f"n={ni}", (xi, mi), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=8, color=INK_2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('excess coincidences per spike', fontsize=11)
    if result is not None:
        ax.set_title(f"β = {result['beta_predictor']:+.2e} (per SD of predictor), "
                     f"p = {result['p_permutation_two_sided']:.4f}\n"
                     f"{result['n_pairs']} pairs, {result['n_permutations']} unit shuffles",
                     fontsize=10, color=INK)
    _style(ax)


def plot_block_detail(table, results, ccg, sync, w, d, label, out_path):
    """Headline (w, D) for one block: CCG, tuning profiles, sanity checks, null."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(f"{label}: synchrony within ±{w:g} ms, jitter-corrected "
                 f"(D = {d:g} ms)\n{sync['n_up_windows']} NREM UP windows, "
                 f"{sync['total_up_sec']:.0f} s of UP time, "
                 f"{len(sync['unit_ids'])} tuned units",
                 fontsize=15, fontweight='bold', color=INK)
    ok = np.asarray(table['possible_duplicate']) == 0
    excess_norm = np.where(ok, table[col('excess_norm', w, d)], np.nan)

    # (a) jitter-corrected CCG by Δpref group
    ax = axes[0, 0]
    lags, obs, sur, geo_sum, n_pairs = ccg
    ax.axvspan(sync['params']['exclude_ms'], w, color=GRID, alpha=0.6, lw=0,
               label=f"counted window ({sync['params']['exclude_ms']:g}–{w:g} ms)")
    ax.axhline(0, color=INK_2, linewidth=1)
    for g, (lo, hi) in enumerate(PREF_GROUPS):
        if geo_sum[g] <= 0:
            continue
        per_bin = (obs[g] - sur[g]) / geo_sum[g]
        ax.plot(lags, per_bin, color=COLORS[g], linewidth=2,
                label=f"Δpref {lo:.0f}–{min(hi, 90):.0f}° ({n_pairs[g]} pairs)")
    ax.set_xlabel('|lag| (ms)', fontsize=11)
    ax.set_ylabel('observed − jitter mean\n(coincidences per spike per 1 ms bin)',
                  fontsize=11)
    ax.set_title('Jitter-corrected cross-correlogram (folded)', fontsize=12, color=INK)
    ax.legend(fontsize=9, frameon=False)
    _style(ax)

    # (b, c) excess vs tuning similarity
    _profile(axes[0, 1], table['delta_pref_deg'], excess_norm, np.linspace(0, 90, 7),
             COLORS[0], results.get(('delta_pref_deg', w, d)), PRETTY['delta_pref_deg'])
    _profile(axes[0, 2], table['signal_corr'], excess_norm, np.linspace(-1, 1, 9),
             COLORS[0], results.get(('signal_corr', w, d)), PRETTY['signal_corr'])

    # (d) observed vs jitter expectation: how much of raw co-firing is expected
    ax = axes[1, 0]
    o = np.asarray(table[col('observed', w)], dtype=float)[ok]
    e = np.asarray(table[col('jit_mean', w, d)], dtype=float)[ok]
    good = np.isfinite(o) & np.isfinite(e) & (e > 0) & (o > 0)
    ax.scatter(e[good], o[good], s=9, color=COLORS[0], alpha=0.35, linewidths=0)
    if good.any():
        lim = [min(e[good].min(), o[good].min()), max(e[good].max(), o[good].max())]
        ax.plot(lim, lim, color=INK_2, linewidth=1.2, label='observed = jitter mean')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel('jitter-expected coincidences', fontsize=11)
    ax.set_ylabel('observed coincidences', fontsize=11)
    ax.set_title('Per pair: most raw co-firing is expected from rates', fontsize=12,
                 color=INK)
    _style(ax)

    # (e) pair-level excess z
    ax = axes[1, 1]
    z = np.asarray(table[col('excess_z', w, d)], dtype=float)[ok]
    z = z[np.isfinite(z)]
    p = np.asarray(table[col('p_jitter', w, d)], dtype=float)[ok]
    p = p[np.isfinite(p)]
    ax.hist(z, bins=60, color=COLORS[0], edgecolor='white', linewidth=0.5)
    ax.axvline(0, color=INK_2, linewidth=1.2)
    ax.set_xlabel('excess z (in jitter-null SDs)', fontsize=11)
    ax.set_ylabel('pairs', fontsize=11)
    ax.set_title(f"median z = {np.median(z) if z.size else np.nan:+.2f};  "
                 f"{100 * np.mean(p < 0.01) if p.size else np.nan:.1f}% of pairs "
                 f"p_jitter < 0.01", fontsize=11, color=INK)
    _style(ax)

    # (f) permutation null for the orientation predictor
    ax = axes[1, 2]
    r = results.get(('delta_pref_deg', w, d))
    if r is not None:
        ax.hist(r['null_distribution'], bins=40, color=GRID, edgecolor='white')
        ax.axvline(r['beta_predictor'], color=COLORS[1], linewidth=2.5,
                   label=f"observed β = {r['beta_predictor']:+.2e}")
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel('β(Δ pref) with tuning shuffled across units', fontsize=11)
    ax.set_ylabel('shuffles', fontsize=11)
    ax.set_title('Unit-shuffle null (hypothesis: β < 0)', fontsize=12, color=INK)
    _style(ax)
    n_dup = int((~ok).sum())
    if n_dup:
        fig.text(0.995, 0.005, f"{n_dup} possible duplicate pairs excluded",
                 ha='right', fontsize=9, color=INK_2)

    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(out_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  figure: {out_path}")


def plot_beta_vs_jitter(results_by_block, coinc_ms, jitter_ms, out_path, title):
    """β(tuning) against jitter width D, one line per coincidence window."""
    blocks = [b for b in ('pre', 'post', 'pre_vs_post') if b in results_by_block]
    fig, axes = plt.subplots(len(PREDICTORS), len(blocks),
                             figsize=(5.5 * len(blocks), 8.5), squeeze=False)
    fig.suptitle(f"{title}\nβ(tuning) on excess synchrony vs jitter width "
                 f"(filled = p < 0.05; band = unit-shuffle null mean ± 2 SD)",
                 fontsize=14, fontweight='bold', color=INK)
    for row, predictor in enumerate(PREDICTORS):
        for c, block in enumerate(blocks):
            ax = axes[row, c]
            ax.axhline(0, color=INK_2, linewidth=1)
            for k, w in enumerate(coinc_ms[:3]):
                rs = [results_by_block[block].get((predictor, w, d)) for d in jitter_ms]
                x = np.array([d for d, r in zip(jitter_ms, rs) if r is not None])
                if not x.size:
                    continue
                rs = [r for r in rs if r is not None]
                beta = np.array([r['beta_predictor'] for r in rs])
                mu = np.array([r['beta_null_mean'] for r in rs])
                sd = np.array([r['beta_null_sd'] for r in rs])
                p = np.array([r['p_permutation_two_sided'] for r in rs])
                ax.fill_between(x, mu - 2 * sd, mu + 2 * sd, color=COLORS[k],
                                alpha=0.12, linewidth=0)
                ax.plot(x, beta, color=COLORS[k], linewidth=2, label=f"±{w:g} ms")
                ax.scatter(x, beta, s=70, zorder=3, edgecolors=COLORS[k],
                           facecolors=[COLORS[k] if pi < 0.05 else 'white' for pi in p],
                           linewidths=2)
            ax.set_xscale('log')
            ax.set_xticks(jitter_ms)
            ax.set_xticklabels([f"{d:g}" for d in jitter_ms])
            ax.minorticks_off()
            ax.set_xlabel('jitter width D (ms)', fontsize=11)
            ax.set_ylabel(f"β on {PRETTY[predictor].split(' (')[0]}", fontsize=11)
            name = {'pre': 'sleep PRE', 'post': 'sleep POST',
                    'pre_vs_post': 'POST − PRE (within pair)'}[block]
            ax.set_title(name, fontsize=12, color=INK)
            ax.legend(title='coincidence window', fontsize=9, title_fontsize=9,
                      frameon=False)
            _style(ax)
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(out_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  figure: {out_path}")


def headline_ccg(sync, similarity, table, w, d, n_surrogates, seed):
    """Jitter-corrected CCG summed within Δpref groups (for the detail figure)."""
    ids = sync['unit_ids']
    row_of = {uid: k for k, uid in enumerate(ids)}
    n = len(ids)
    groups = np.full((n, n), -1, dtype=np.int64)
    geo_sum = np.zeros(len(PREF_GROUPS))
    n_pairs = np.zeros(len(PREF_GROUPS), dtype=int)
    dup = np.asarray(table['possible_duplicate']) > 0
    for (a, b), dp, is_dup in zip(similarity['pair_index'], similarity['delta_pref_deg'],
                                  dup):
        ra, rb = row_of.get(similarity['unit_ids'][a]), row_of.get(similarity['unit_ids'][b])
        if ra is None or rb is None or is_dup or not np.isfinite(dp):
            continue
        for g, (lo, hi) in enumerate(PREF_GROUPS):
            if lo <= dp < hi:
                groups[ra, rb] = groups[rb, ra] = g
                geo_sum[g] += np.sqrt(float(sync['n_spikes'][ra]) * sync['n_spikes'][rb])
                n_pairs[g] += 1
    lags, obs, sur = tj.jitter_corrected_ccg(sync['axis'], groups, len(PREF_GROUPS),
                                             jitter_ms=d, max_lag_ms=min(40.0, d),
                                             n_surrogates=n_surrogates, seed=seed)
    return lags, obs, sur, geo_sum, n_pairs


def write_csv(table, path):
    keys = [k for k, v in table.items() if np.asarray(v).ndim == 1]
    cols = [np.asarray(table[k]) for k in keys]
    with Path(path).open('w') as f:
        f.write(','.join(keys) + '\n')
        for row in range(len(cols[0])):
            f.write(','.join((f"{c[row]:.6g}" if np.issubdtype(c.dtype, np.number)
                              else str(c[row])) for c in cols) + '\n')
    print(f"  pair table: {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _floats(text):
    return tuple(float(x) for x in str(text).split(',') if x.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='List inputs and exit')
    parser.add_argument('--self-test', action='store_true',
                        help='Run on synthetic data with known answers')
    parser.add_argument('--sortout', default=None)
    parser.add_argument('--animal', default=None)
    parser.add_argument('--date', default=None)
    parser.add_argument('--blocks', default='pre,post')
    parser.add_argument('--out', default=None,
                        help='Output folder (default <sortout>/tuning_coupling_jitter)')
    parser.add_argument('--coinc-ms', default='5,10',
                        help='Coincidence windows w, ms (default 5,10)')
    parser.add_argument('--jitter-ms', default='25,50,100',
                        help='Jitter cell widths D, ms (default 25,50,100)')
    parser.add_argument('--headline', default='5,50',
                        help='w,D shown in the detailed figure (default 5,50)')
    parser.add_argument('--exclude-ms', type=float, default=1.0,
                        help='Central lags |lag| < this are not counted -- sorting '
                             'collisions (default 1.0; 0 counts every lag)')
    parser.add_argument('--n-surrogates', type=int, default=500)
    parser.add_argument('--ccg-surrogates', type=int, default=20,
                        help='Surrogates for the CCG figure only (default 20)')
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-splits', type=int, default=200,
                        help='Half-splits for the cross-validated tuning correlation')
    parser.add_argument('--min-shank-fraction', type=float, default=0.5,
                        help='Fraction of shanks simultaneously ON for an UP window')
    parser.add_argument('--min-alignment-ratio', type=float, default=1.25)
    parser.add_argument('--allow-alignment-failure', action='store_true')
    parser.add_argument('--min-up-ms', type=float, default=50.0)
    parser.add_argument('--max-up-ms', type=float, default=2000.0)
    parser.add_argument('--no-nrem', action='store_true',
                        help='Do not restrict UP windows to scored NREM')
    parser.add_argument('--keep-unreliable', action='store_true',
                        help='Keep units whose tuning is not split-half reliable')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    if args.self_test:
        from tc_selftest import run_self_test
        return run_self_test(seed=args.seed)

    coinc_ms, jitter_ms = _floats(args.coinc_ms), _floats(args.jitter_ms)
    head_w, head_d = _floats(args.headline)
    if head_w not in coinc_ms or head_d not in jitter_ms:
        parser.error('--headline must be one of the --coinc-ms / --jitter-ms values')

    paths = session_paths(args.animal, args.date, args.sortout)
    found, missing = discover_inputs(paths)
    if args.check:
        return 0
    if 'tuning' not in found:
        print("\nCannot run: no all_units_tuning.pkl for this session.")
        return 1

    out_dir = Path(args.out) if args.out else paths['sortout'] / 'tuning_coupling_jitter'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    set_repro_stamp(sys.argv[1:] if argv is None else list(argv), paths, found)
    label = f"{paths['animal']} {paths['date']}"

    print("\n" + "=" * 72 + "\n1. VISUAL TUNING\n" + "=" * 72)
    units = sim.load_unit_tuning(found['tuning'], require_reliable=not args.keep_unreliable)
    similarity = sim.pair_tuning_similarity(units, n_splits=args.n_splits, seed=args.seed)
    covariates = sim.pair_covariates(units, similarity['unit_ids'], similarity['pair_index'])
    unit_ids, pair_index = similarity['unit_ids'], similarity['pair_index']
    n_units = len(unit_ids)
    matrices = {p: st.pairs_to_matrix(similarity[p], pair_index, n_units) for p in PREDICTORS}

    tables, results_by_block, syncs, up_summary = {}, {}, {}, {}
    for block in [b.strip() for b in args.blocks.split(',') if b.strip()]:
        if f'sleep_{block}' not in found or f'mua_{block}' not in found:
            need = [k for k in (f'sleep_{block}', f'mua_{block}') if k not in found]
            print(f"\n[{block}] skipped -- missing {', '.join(need)}")
            continue
        print("\n" + "=" * 72 + f"\n2. NREM UP STATES [{block}]\n" + "=" * 72)
        try:
            spikes, windows, meta, report = prepare_block(
                found[f'sleep_{block}'], unit_ids, found[f'mua_{block}'],
                nrem_pkl=None if args.no_nrem else found.get(f'nrem_{block}'),
                min_shank_fraction=args.min_shank_fraction,
                min_up_ms=args.min_up_ms, max_up_ms=args.max_up_ms,
                min_alignment_ratio=args.min_alignment_ratio,
                allow_alignment_failure=args.allow_alignment_failure)
        except ValueError as exc:
            print(f"[{block}] skipped -- {exc}")
            continue

        print("\n" + "=" * 72 + f"\n3. JITTER-CORRECTED SYNCHRONY [{block}]\n" + "=" * 72)
        sync = tj.jitter_synchrony(
            spikes, windows, unit_ids=[u for u in unit_ids if u in spikes],
            coinc_ms=coinc_ms, jitter_ms=jitter_ms, exclude_ms=args.exclude_ms,
            n_surrogates=args.n_surrogates, seed=args.seed)
        table = build_pair_table(similarity, covariates, sync)
        n_dup = int(np.nansum(table['possible_duplicate']))
        print(f"  {n_dup} pairs flagged as possible duplicates "
              f"(> {100 * DUP_FRACTION:.0f}% of spikes within "
              f"{sync['params']['dup_ms']:g} ms) -- excluded from the tests")
        tables[block], syncs[block] = table, sync
        up_summary[block] = {'n_windows': sync['n_up_windows'],
                             'total_up_sec': sync['total_up_sec'],
                             'alignment': report, 'n_duplicate_pairs': n_dup}

        print("\n" + "=" * 72 + f"\n4. TEST -- excess_norm ~ tuning + controls [{block}]\n"
              + "=" * 72)
        results = {}
        for w in coinc_ms:
            for d in jitter_ms:
                response = col('excess_norm', w, d)
                work = testable(table, response)
                for predictor in PREDICTORS:
                    try:
                        r = st.unit_permutation_test(
                            work, matrices[predictor], pair_index, n_units,
                            response=response, controls=CONTROLS,
                            n_permutations=args.n_permutations, seed=args.seed,
                            verbose=False)
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        print(f"  {response} ~ {predictor}: no fit ({exc})")
                        continue
                    results[(predictor, w, d)] = r
                    print(f"  w={w:>4g} D={d:>5g}  {predictor:15s} "
                          f"β={r['beta_predictor']:+.2e}  z={r['beta_z_vs_null']:+5.2f}  "
                          f"p={r['p_permutation_two_sided']:.4f}  ({r['n_pairs']} pairs)")
        results_by_block[block] = results

        tic = time.time()
        ccg = headline_ccg(sync, similarity, table, head_w, head_d,
                           args.ccg_surrogates, args.seed)
        print(f"  headline CCG ({time.time() - tic:.0f} s)")
        plot_block_detail(table, results, ccg, sync, head_w, head_d,
                          f"{label} sleep {block.upper()}",
                          out_dir / f"jitter_sync_{block}_w{head_w:g}_D{head_d:g}_{stamp}.png")
        up_summary[block]['headline_ccg'] = dict(zip(
            ('lag_ms', 'observed', 'jitter_mean', 'geo_spikes', 'n_pairs'), ccg))

    if 'pre' in tables and 'post' in tables:
        print("\n" + "=" * 72 + "\n5. PRE vs POST -- within-pair change\n" + "=" * 72)
        prepost = {}
        dup = np.maximum(tables['pre']['possible_duplicate'],
                         tables['post']['possible_duplicate'])
        for w in coinc_ms:
            for d in jitter_ms:
                response = col('excess_norm', w, d)
                t_pre = testable(dict(tables['pre'], possible_duplicate=dup), response)
                t_post = testable(dict(tables['post'], possible_duplicate=dup), response)
                for predictor in PREDICTORS:
                    try:
                        r = st.compare_pre_post(
                            t_pre, t_post, matrices[predictor], pair_index, n_units,
                            response=response, controls=CONTROLS,
                            n_permutations=args.n_permutations, seed=args.seed,
                            verbose=False)
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        print(f"  Δ{response} ~ {predictor}: no fit ({exc})")
                        continue
                    prepost[(predictor, w, d)] = r
                    print(f"  w={w:>4g} D={d:>5g}  {predictor:15s} "
                          f"β={r['beta_predictor']:+.2e}  z={r['beta_z_vs_null']:+5.2f}  "
                          f"p={r['p_permutation_two_sided']:.4f}  ({r['n_pairs']} pairs)")
        results_by_block['pre_vs_post'] = prepost

    if not tables:
        print("\nNo block could be analysed.")
        return 1

    plot_beta_vs_jitter(results_by_block, coinc_ms, jitter_ms,
                        out_dir / f"jitter_sync_beta_vs_D_{stamp}.png", label)

    out_pkl = out_dir / f"jitter_sync_results_{stamp}.pkl"
    with out_pkl.open('wb') as f:
        pickle.dump({
            'session': paths, 'inputs': {k: str(v) for k, v in found.items()},
            'params': vars(args), 'unit_ids': unit_ids, 'pair_index': pair_index,
            'tables': tables,
            'results': {block: {f"{p}|w{w:g}|D{d:g}": {k: v for k, v in r.items()
                                                       if k != 'mask'}
                                for (p, w, d), r in res.items()}
                        for block, res in results_by_block.items()},
            'up_summary': up_summary,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResults: {out_pkl}")
    for block, table in tables.items():
        write_csv(table, out_dir / f"jitter_sync_pairs_{block}_{stamp}.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
