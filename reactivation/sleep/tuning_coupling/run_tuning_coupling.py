r"""Do NREM UP-state co-activation partners share visual tuning?

Runs the whole test for ONE session (the one in grating_config.py):

  1. visual tuning        -- preferred-orientation difference (0 deg == 180 deg)
                             and cross-validated tuning-curve correlation, from
                             all_units_tuning.pkl, restricted to units whose
                             tuning is responsive and split-half reliable
  2. NREM UP states       -- MUA global ON windows from sleep/MUA/detect_off_states.py,
                             REQUIRED by default (a block without them is skipped,
                             not silently replaced); --population-up opts into the
                             sorted-population-rate fallback instead. Optionally
                             intersected with scored NREM.
  3. excess co-activation -- co-firing in short bins inside UP states, minus a
                             within-UP block-shuffle null that preserves each
                             unit's rate and UP-state drive
  4. the test             -- coupling ~ tuning similarity + rate + distance +
                             cell type, with a unit-label permutation p-value
  5. pre vs post          -- the same model on the within-pair change, which is
                             the experience-dependent comparison

Usage
-----
    python run_tuning_coupling.py --check          # what exists, what is missing
    python run_tuning_coupling.py                  # full run, both sleep blocks
    python run_tuning_coupling.py --blocks post --n-surrogates 500
    python run_tuning_coupling.py --self-test      # synthetic data, known answer

Every stage prints what it did and every output is written next to the session.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'passive_visual' / 'FreelyMovingProcessing' / 'Grating'))

import tc_similarity as sim          # noqa: E402
import tc_coupling as cpl            # noqa: E402
import tc_stats as st                # noqa: E402


CONTROLS = ('log_rate_geomean', 'abs_log_rate_ratio', 'same_shank',
            'abs_dy_um', 'abs_dx_um', 'pair_type')
COUPLING_MEASURES = ('excess_z', 'resid_corr_z', 'raw_corr_z')
PREDICTORS = ('delta_pref_deg', 'signal_corr')


# --------------------------------------------------------------------------- #
# Input discovery
# --------------------------------------------------------------------------- #

def session_paths(animal=None, date=None, sortout=None):
    """Resolve the configured session's folders (grating_config is the default)."""
    if sortout is None or animal is None or date is None:
        from grating_config import ANIMAL_ID, EXPERIMENT_DATE, SORTOUT_FOLDER
        animal = animal or ANIMAL_ID
        date = date or EXPERIMENT_DATE
        sortout = Path(sortout) if sortout else Path(SORTOUT_FOLDER)
    return {'animal': animal, 'date': date, 'sortout': Path(sortout)}


def experiment_data_day_folder(sortout, animal, date):
    """
    The experiment_data mirror of a sortout session folder.

    Layout differs by one path segment, not just the leading directory name:
        sortout:         sortout/<ANIMAL>/<ANIMAL>_20<DATE>
        experiment_data: experiment_data/<ANIMAL>/<DATE>/<ANIMAL>_20<DATE>
    A bare string-replace of 'sortout' -> 'experiment_data' therefore lands one
    level short (drops the bare-DATE folder) and silently finds nothing -- this
    reconstructs the path explicitly instead. MUA/off_states output and
    *_sleep_periods.pkl (NREM scoring) both live under this tree, not sortout.
    """
    sortout = Path(sortout)
    if 'sortout' not in sortout.parts:
        return None
    experiment_data_animal_folder = Path(
        str(sortout).replace('sortout', 'experiment_data', 1)).parent
    return experiment_data_animal_folder / date / f"{animal}_20{date}"


def _glob_real(folder, pattern):
    """glob() that drops macOS AppleDouble sidecar files (._name, 4KB metadata
    stubs SMB/AFP shares create alongside a real file whenever its extended
    attributes get touched from macOS) -- Path.glob does NOT exclude these on
    its own, and since '._x' sorts before 'x' the sidecar wins every
    sorted(...)[0] pick here, silently handing downstream code a 4KB stub
    instead of the real pickle.
    """
    return sorted(p for p in Path(folder).glob(pattern) if not p.name.startswith('.'))


def discover_inputs(paths, verbose=True):
    """
    Locate every file the analysis needs and say what to run for the missing ones.

    Nothing is inferred silently: each entry is either a concrete existing path
    or None with the command that would create it.
    """
    sortout = paths['sortout']
    date = paths['date']
    found, missing = {}, {}
    day_folder = experiment_data_day_folder(sortout, paths['animal'], date)

    tuning = _glob_real(sortout, '**/all_units_tuning.pkl')
    if tuning:
        found['tuning'] = max(tuning, key=lambda p: p.stat().st_mtime)
    else:
        missing['tuning'] = ("python GratingTuningCurve.py   "
                             "(writes <session>_tuning_curves/all_units_tuning.pkl)")

    for block in ('pre', 'post'):
        hits = _glob_real(sortout, f'sleep_spikes_*sleep_{block}.pkl')
        if hits:
            found[f'sleep_{block}'] = hits[0]
        else:
            missing[f'sleep_{block}'] = (
                f"python extract_sleep_session_spikes.py   "
                f"(needs SLEEP_BLOCKS sample range for '{block}' in grating_config.py)")

        mua = _glob_real(sortout, f'MUA/off_states/*global_on_windows*_{block}*.pkl')
        mua += _glob_real(sortout, f'MUA/off_states/*{block}*global_on_windows*.pkl')
        if day_folder is not None:
            mua += _glob_real(day_folder, f'MUA/off_states/*global_on_windows*_{block}*.pkl')
            mua += _glob_real(day_folder, f'MUA/off_states/*{block}*global_on_windows*.pkl')
        if mua:
            found[f'mua_{block}'] = mua[0]
        else:
            missing[f'mua_{block}'] = (
                f"python sleep/MUA/find_sleep_mua.py && "
                f"python sleep/MUA/detect_off_states.py --epochs {block}   "
                f"(REQUIRED by default -- this block will be skipped without it; "
                f"--population-up overrides, using the less independent sorted "
                f"population rate instead)")

    # NREM scoring is written one file PER EPOCH (*_pre_sleep_periods.pkl,
    # *_post_sleep_periods.pkl -- each in that epoch's own relative clock), not
    # one shared file, so pre and post must be looked up separately. A run
    # handed the wrong epoch's file would silently apply the wrong clock's
    # windows -- look for the per-block name first, and only fall back to a
    # single non-block-tagged file (old-style output, or a single combined
    # scoring pass) when no per-block file exists.
    for block in ('pre', 'post'):
        nrem = _glob_real(sortout, f'**/*{block}_sleep_periods.pkl')
        if day_folder is not None:
            nrem += _glob_real(day_folder, f'**/*{block}_sleep_periods.pkl')
        if nrem:
            found[f'nrem_{block}'] = nrem[0]
        else:
            missing[f'nrem_{block}'] = (
                f"python sleep/score_nrem_epochs.py   "
                f"(optional: without it, '{block}' UP states are not restricted "
                f"to scored NREM)")

    if verbose:
        print(f"\nSession {paths['animal']} {date}")
        print(f"  sortout: {sortout}")
        print(f"  exists : {sortout.exists()}")
        print("\nFound:")
        for key, path in found.items():
            print(f"  {key:12s} {path}")
        print("\nMissing:")
        if not missing:
            print("  (nothing)")
        for key, how in missing.items():
            optional = key.startswith('nrem')  # MUA is required by default; see note above
            print(f"  {key:12s} {'[optional] ' if optional else '[REQUIRED]  '}{how}")
    return found, missing


# --------------------------------------------------------------------------- #
# One sleep block
# --------------------------------------------------------------------------- #

def analyse_block(sleep_pkl, unit_ids, mua_windows_pkl=None, nrem_pkl=None,
                  nrem_epoch=None, bin_ms=25.0, block_bins=5, n_surrogates=200,
                  min_shank_fraction=0.5, min_up_ms=50.0, max_up_ms=2000.0,
                  n_phase_bins=5, seed=0, min_alignment_ratio=1.5,
                  allow_alignment_failure=False, verbose=True):
    """Load one sleep block, build UP states, and return the coupling matrices."""
    spikes, meta = cpl.load_sleep_spikes(sleep_pkl, unit_ids=unit_ids, verbose=verbose)
    if len(spikes) < 2:
        raise ValueError(f"{Path(sleep_pkl).name}: fewer than 2 of the tuned units "
                         f"are present in this sleep block.")

    # NREM restriction happens BEFORE the alignment check, not after: a
    # presleep/postsleep block is mostly wake (the animal walks around in both
    # -- see find_sleep_mua.py's own docstring), and waking movement can easily
    # out-fire a brief, quiet NREM UP burst. Comparing "inside UP windows" to
    # "everywhere else in the block" then fails even a correctly-mapped clock,
    # because "everywhere else" is dominated by wake, not DOWN states. Checking
    # alignment against the scored-NREM universe instead asks the right
    # question: within NREM, are MUA UP windows where the sorted units fire?
    nrem = (cpl.load_nrem_windows(nrem_pkl, epoch=nrem_epoch, verbose=verbose)
           if nrem_pkl is not None else None)

    source = 'mua'
    updown = None
    if mua_windows_pkl is not None:
        windows = cpl.up_windows_from_mua(
            mua_windows_pkl, meta, min_shank_fraction=min_shank_fraction,
            min_duration_sec=min_up_ms / 1000.0,
            max_duration_sec=max_up_ms / 1000.0, verbose=verbose)
    else:
        windows, updown = cpl.up_windows_from_population_rate(
            spikes, meta['duration_sec'],
            min_duration_sec=min_up_ms / 1000.0,
            max_duration_sec=max_up_ms / 1000.0, verbose=verbose)
        source = 'population_rate'

    if nrem is not None:
        before = len(windows)
        windows = cpl._intersect_windows(windows, nrem)
        windows = cpl._clip_and_filter(windows, meta['duration_sec'],
                                       min_up_ms / 1000.0, max_up_ms / 1000.0)
        if verbose:
            print(f"  restricted to scored NREM: {before} -> {len(windows)} UP windows")

    if len(windows) == 0:
        raise ValueError("No UP windows survived filtering"
                         + (" (intersected with scored NREM)" if nrem is not None else "") + ".")

    min_ratio = min_alignment_ratio if mua_windows_pkl is not None else 1.0
    ratio, report = cpl.check_up_alignment(
        spikes, windows, meta['duration_sec'], min_ratio=min_ratio, verbose=verbose,
        universe_windows=nrem)
    if mua_windows_pkl is not None and not report.get('passed', False) and not allow_alignment_failure:
        raise ValueError(
            f"UP windows from {Path(mua_windows_pkl).name} do not line up with "
            f"the sorted spikes (in/out rate ratio {ratio:.2f}, universe="
            f"{report.get('universe', 'whole block')}). Either the MUA files and "
            f"the sleep pkl are on different clocks, or (if no NREM scoring was "
            f"available to restrict the comparison) 'outside the UP windows' is "
            f"dominated by wake rather than DOWN states; fix the mapping, supply "
            f"NREM scoring, or pass --population-up to use population-rate UP "
            f"states instead.")

    binned = cpl.bin_within_up(spikes, windows, bin_sec=bin_ms / 1000.0,
                               block_bins=block_bins, verbose=verbose)
    coupling = cpl.pair_coupling(binned, n_surrogates=n_surrogates,
                                 n_phase_bins=n_phase_bins, seed=seed,
                                 verbose=verbose)
    coupling.update({'up_windows': windows, 'up_source': source,
                     'alignment': report, 'sleep_meta': meta,
                     'updown': updown, 'bin_sec': bin_ms / 1000.0,
                     'block_bins': block_bins})
    return coupling


def build_pair_table(units, similarity, covariates, coupling):
    """
    Merge tuning similarity, covariates and coupling into one column dict.

    The coupling matrices are indexed by the units present in the sleep block,
    which can be a subset of the tuned units, so every column is built through
    an explicit index map rather than by position.
    """
    unit_ids = similarity['unit_ids']
    pair_index = similarity['pair_index']
    row_of = {uid: k for k, uid in enumerate(coupling['unit_ids'])}
    keep_unit = np.array([uid in row_of for uid in unit_ids])

    i, j = pair_index[:, 0], pair_index[:, 1]
    usable = keep_unit[i] & keep_unit[j]

    rows = np.array([row_of.get(uid, -1) for uid in unit_ids])
    ri, rj = rows[i], rows[j]

    def pull(matrix):
        out = np.full(len(i), np.nan)
        out[usable] = np.asarray(matrix)[ri[usable], rj[usable]]
        return out

    rate = np.full(len(unit_ids), np.nan)
    rate[keep_unit] = coupling['rate_hz'][rows[keep_unit]]
    rate_i, rate_j = rate[i], rate[j]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_geo = 0.5 * (np.log10(rate_i) + np.log10(rate_j))
        log_ratio = np.abs(np.log10(rate_i) - np.log10(rate_j))

    table = {
        'unit_i': np.array([unit_ids[k] for k in i], dtype=object),
        'unit_j': np.array([unit_ids[k] for k in j], dtype=object),
        'delta_pref_deg': similarity['delta_pref_deg'],
        'signal_corr': similarity['signal_corr'],
        'signal_corr_raw': similarity['signal_corr_raw'],
        'log_rate_geomean': log_geo,
        'abs_log_rate_ratio': log_ratio,
        'rate_i_hz': rate_i, 'rate_j_hz': rate_j,
        'excess_z': pull(coupling['excess_z']),
        'excess_rate_hz': pull(coupling['excess_rate_hz']),
        'raw_corr': pull(coupling['raw_corr']),
        'resid_corr': pull(coupling['resid_corr']),
        **covariates,
    }
    table['raw_corr_z'] = st.fisher_z(table['raw_corr'])
    table['resid_corr_z'] = st.fisher_z(table['resid_corr'])
    return table


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_block(table, results, label, out_path, up_windows=None, up_source=''):
    """Coupling-vs-similarity profiles, permutation nulls and UP diagnostics."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f"Tuning similarity vs NREM UP-state co-activation — {label}"
                 f"   (UP states: {up_source})", fontsize=17, fontweight='bold')

    pref_edges = np.linspace(0, 90, 7)
    corr_edges = np.linspace(-1, 1, 9)

    for col, (measure, pretty) in enumerate(
            [('raw_corr_z', 'RAW co-firing (comparator)'),
             ('excess_z', 'EXCESS co-activation (primary)'),
             ('resid_corr_z', 'RESIDUAL correlation')]):
        ax = axes[0, col]
        x, m, s, n = st.binned_profile(table['delta_pref_deg'], table[measure], pref_edges)
        ax.errorbar(x, m, yerr=s, marker='o', markersize=9, linewidth=2.5,
                    capsize=5, color='#2E86AB')
        ax.set_xlabel('Δ preferred orientation (deg)', fontsize=13, fontweight='bold')
        ax.set_ylabel(pretty, fontsize=12, fontweight='bold')
        ax.set_title(pretty, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        for xi, ni in zip(x, n):
            ax.annotate(f'{ni}', (xi, ax.get_ylim()[0]), fontsize=8,
                        ha='center', va='bottom', color='gray')

        ax2 = axes[1, col]
        x2, m2, s2, _ = st.binned_profile(table['signal_corr'], table[measure], corr_edges)
        ax2.errorbar(x2, m2, yerr=s2, marker='s', markersize=9, linewidth=2.5,
                     capsize=5, color='#A23B72')
        ax2.set_xlabel('tuning-curve correlation (cross-validated)',
                       fontsize=13, fontweight='bold')
        ax2.set_ylabel(pretty, fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        key = ('signal_corr', measure)
        if key in results:
            r = results[key]
            ax2.set_title(f"β={r['beta_predictor']:+.3f}, "
                          f"p={r['p_permutation_two_sided']:.4f}",
                          fontsize=12, fontweight='bold')
        key = ('delta_pref_deg', measure)
        if key in results:
            r = results[key]
            ax.set_title(f"{pretty}\nβ={r['beta_predictor']:+.3f}, "
                         f"p={r['p_permutation_two_sided']:.4f}", fontsize=12,
                         fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  figure: {out_path}")


def plot_permutation_nulls(results, out_path, title):
    """Observed tuning coefficient against its unit-shuffle null."""
    keys = [k for k in results if k[1] in COUPLING_MEASURES]
    if not keys:
        return
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        r = results[key]
        null = r['null_distribution']
        ax.hist(null, bins=40, color='0.75', edgecolor='none')
        ax.axvline(r['beta_predictor'], color='#C0392B', linewidth=3,
                   label=f"observed {r['beta_predictor']:+.3f}")
        ax.set_title(f"{key[1]} ~ {key[0]}\np={r['p_permutation_two_sided']:.4f}",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('β under shuffled tuning', fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  figure: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true',
                        help='List the inputs this session has and what is missing')
    parser.add_argument('--self-test', action='store_true',
                        help='Run on synthetic data with a known planted effect')
    parser.add_argument('--sortout', default=None, help='Override session folder')
    parser.add_argument('--animal', default=None)
    parser.add_argument('--date', default=None)
    parser.add_argument('--blocks', default='pre,post',
                        help='Sleep blocks to analyse (default pre,post)')
    parser.add_argument('--out', default=None, help='Output folder')
    parser.add_argument('--bin-ms', type=float, default=25.0)
    parser.add_argument('--block-ms', type=float, default=125.0,
                        help='Shuffle block; counts are permuted within it (default 125 ms)')
    parser.add_argument('--n-surrogates', type=int, default=200)
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-splits', type=int, default=200,
                        help='Half-splits for the cross-validated tuning correlation')
    parser.add_argument('--population-up', action='store_true',
                        help='REQUIRED to analyse a block without detect_off_states.py '
                             'MUA windows (falls back to sorted population-rate UP '
                             'states); also overrides MUA windows when both exist. '
                             'Off by default: a block with no MUA output is skipped, '
                             'not silently substituted.')
    parser.add_argument('--min-shank-fraction', type=float, default=0.5,
                        help='Fraction of scored shanks that must be simultaneously '
                             'ON for a probe-wide UP window (default 0.5). Raising it '
                             'demands tighter cross-shank synchrony -- fewer, purer UP '
                             'windows and usually a better alignment ratio, at the '
                             'cost of less UP time to bin.')
    parser.add_argument('--no-nrem', action='store_true',
                        help='Do not restrict UP states to scored NREM')
    parser.add_argument('--keep-unreliable', action='store_true',
                        help='Keep units whose tuning is not split-half reliable')
    parser.add_argument('--allow-alignment-failure', action='store_true',
                        help='Proceed even if MUA windows fail the clock check')
    parser.add_argument('--min-alignment-ratio', type=float, default=1.5,
                        help='In/out firing-rate ratio the UP windows must clear '
                             '(within scored NREM, or the whole block with no '
                             'NREM) before coupling is computed on them. Lower '
                             'this explicitly for a session with a genuinely '
                             'weaker UP/DOWN contrast rather than reaching for '
                             '--allow-alignment-failure, which skips the check '
                             'altogether -- this way the chosen bar is visible '
                             'in the run instead of hidden behind a bypass flag.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    if args.self_test:
        from tc_selftest import run_self_test
        return run_self_test(seed=args.seed)

    paths = session_paths(args.animal, args.date, args.sortout)
    found, missing = discover_inputs(paths)
    if args.check:
        return 0

    if 'tuning' not in found:
        print("\nCannot run: no all_units_tuning.pkl for this session.")
        return 1

    out_dir = Path(args.out) if args.out else paths['sortout'] / 'tuning_coupling'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"

    print("\n" + "=" * 72)
    print("1. VISUAL TUNING")
    print("=" * 72)
    units = sim.load_unit_tuning(found['tuning'],
                                 require_reliable=not args.keep_unreliable)
    similarity = sim.pair_tuning_similarity(units, n_splits=args.n_splits,
                                            seed=args.seed)
    covariates = sim.pair_covariates(units, similarity['unit_ids'],
                                     similarity['pair_index'])
    n_units = len(similarity['unit_ids'])
    matrices = {
        'delta_pref_deg': st.pairs_to_matrix(similarity['delta_pref_deg'],
                                             similarity['pair_index'], n_units),
        'signal_corr': st.pairs_to_matrix(similarity['signal_corr'],
                                          similarity['pair_index'], n_units),
    }

    blocks = [b.strip() for b in args.blocks.split(',') if b.strip()]
    tables, results_by_block, couplings = {}, {}, {}

    for block in blocks:
        key = f'sleep_{block}'
        if key not in found:
            print(f"\n[{block}] skipped — {missing.get(key, 'no sleep pkl')}")
            continue
        mua = found.get(f'mua_{block}')
        if mua is None and not args.population_up:
            print(f"\n[{block}] skipped — no MUA off-state windows from "
                  f"detect_off_states.py ({missing.get(f'mua_{block}', 'not found')}). "
                  f"UP states must come from detect_off_states.py by default; "
                  f"pass --population-up to fall back to the sorted population "
                  f"rate instead (less independent from the units being tested).")
            continue
        print("\n" + "=" * 72)
        print(f"2-3. SLEEP BLOCK '{block.upper()}': UP STATES AND EXCESS CO-ACTIVATION")
        print("=" * 72)
        mua = None if args.population_up else mua
        nrem = None if args.no_nrem else found.get(f'nrem_{block}')
        coupling = analyse_block(
            found[key], unit_ids=similarity['unit_ids'], mua_windows_pkl=mua,
            nrem_pkl=nrem, nrem_epoch=block, bin_ms=args.bin_ms,
            block_bins=max(2, int(round(args.block_ms / args.bin_ms))),
            n_surrogates=args.n_surrogates, seed=args.seed,
            min_alignment_ratio=args.min_alignment_ratio,
            min_shank_fraction=args.min_shank_fraction,
            allow_alignment_failure=args.allow_alignment_failure)
        table = build_pair_table(units, similarity, covariates, coupling)
        tables[block] = table
        couplings[block] = coupling

        print("\n" + "=" * 72)
        print(f"4. TEST — does tuning similarity predict coupling? [{block}]")
        print("=" * 72)
        results = {}
        for predictor in PREDICTORS:
            for measure in COUPLING_MEASURES:
                print(f"\n  {measure} ~ {predictor}")
                results[(predictor, measure)] = st.unit_permutation_test(
                    table, matrices[predictor], similarity['pair_index'], n_units,
                    response=measure, controls=CONTROLS,
                    n_permutations=args.n_permutations, seed=args.seed,
                    verbose=False)
                print(st.format_result(results[(predictor, measure)], label='    '))
        results_by_block[block] = results

        plot_block(table, results, f"{paths['animal']} {paths['date']} — sleep {block}",
                   out_dir / f"tuning_coupling_{block}_{stamp}.png",
                   up_source=coupling['up_source'])
        plot_permutation_nulls(
            {k: v for k, v in results.items() if k[0] == 'signal_corr'},
            out_dir / f"tuning_coupling_{block}_nulls_{stamp}.png",
            f"Unit-shuffle null — sleep {block}")

    if len(tables) == 2 and 'pre' in tables and 'post' in tables:
        print("\n" + "=" * 72)
        print("5. PRE vs POST — is the relationship experience-dependent?")
        print("=" * 72)
        prepost = {}
        for predictor in PREDICTORS:
            for measure in ('excess_z', 'resid_corr_z'):
                print(f"\n  Δ{measure} (post − pre) ~ {predictor}")
                prepost[(predictor, measure)] = st.compare_pre_post(
                    tables['pre'], tables['post'], matrices[predictor],
                    similarity['pair_index'], n_units, response=measure,
                    controls=CONTROLS, n_permutations=args.n_permutations,
                    seed=args.seed, verbose=True)
                print(st.format_result(prepost[(predictor, measure)], label='    '))
        results_by_block['pre_vs_post'] = prepost
        plot_permutation_nulls(
            {k: v for k, v in prepost.items() if k[0] == 'signal_corr'},
            out_dir / f"tuning_coupling_prepost_nulls_{stamp}.png",
            "Unit-shuffle null — post minus pre")

    out_pkl = out_dir / f"tuning_coupling_results_{stamp}.pkl"
    with out_pkl.open('wb') as f:
        pickle.dump({
            'session': paths, 'inputs': {k: str(v) for k, v in found.items()},
            'params': vars(args), 'unit_ids': similarity['unit_ids'],
            'pair_index': similarity['pair_index'],
            'tables': tables,
            'results': {block: {f"{p}|{m}": {kk: vv for kk, vv in r.items()
                                             if kk not in ('mask', 'null_distribution')}
                                for (p, m), r in res.items()}
                        for block, res in results_by_block.items()},
            'up_summary': {b: {'source': c['up_source'],
                               'n_windows': len(c['up_windows']),
                               'total_up_sec': c['total_up_sec'],
                               'alignment': c['alignment']}
                           for b, c in couplings.items()},
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResults: {out_pkl}")

    for block, table in tables.items():
        csv_path = out_dir / f"pair_table_{block}_{stamp}.csv"
        keys = [k for k, v in table.items() if np.asarray(v).ndim == 1]
        with csv_path.open('w') as f:
            f.write(','.join(keys) + '\n')
            cols = [np.asarray(table[k]) for k in keys]
            for row in range(len(cols[0])):
                f.write(','.join(
                    (f"{c[row]:.6g}" if np.issubdtype(c.dtype, np.number) else str(c[row]))
                    for c in cols) + '\n')
        print(f"Pair table: {csv_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
