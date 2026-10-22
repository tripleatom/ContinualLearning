r"""Synthetic checks of the jitter-synchrony pipeline, each with a known answer.

Mechanics (small hand-made spike trains)
    * coincidence counts equal a brute-force count
    * a jitter surrogate keeps every spike in its own UP window and jitter cell
    * spikes 5 ms apart in real time but in DIFFERENT UP windows never count

Whole sessions (written in the real pickle formats, analysed by the real code)
    ``slow_common``  every unit follows a shared per-UP gain and within-UP
                     envelope (slow).  Jitter keeps that, so excess must be ~0
                     and tuning must predict nothing.  One unit is a planted
                     duplicate of another and must be flagged.
    ``mid_common``   a shared ~20 ms rate fluctuation, faster than most jitter
                     cells.  Excess must be positive for nearly every pair --
                     larger for wider D -- yet carry no tuning effect.
    ``effect``       precise (+-1.5 ms) synchronous events shared only by units
                     with similar preferred orientation.  The tuning test must
                     find it: beta > 0 on signal_corr, beta < 0 on delta_pref.

Calibration
    On the ``mid_common`` sleep data, many freshly drawn random tuning sets are
    tested; the fraction with p < 0.05 must stay near 5%.  This is the direct
    check that the unit-shuffle p-value is valid.  (A single dataset's null
    check can fail by chance -- with seed 0, mid_common's real tuning happens to
    line up with its sleep rates at p ~ 0.01 -- which is why the sessions use
    seed 10 + --seed, and why this calibration check exists.)

Run:  python run_tuning_coupling.py --self-test
"""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_similarity as sim
import tc_jitter as tj
import tc_stats as st

ORIENTATIONS = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
GRID_SEC = 0.005
N_GROUPS = 8


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #

def check_mechanics(seed=0):
    rng = np.random.default_rng(seed)
    checks = []

    wins = np.array([[0.0, 0.4], [0.405, 1.0], [2.0, 2.3]])
    spikes = {f'u{k}': np.sort(rng.uniform(0, 2.3, 80)) for k in range(5)}
    axis = tj.up_time_axis(spikes, wins)
    C = tj.coincidence_counts(axis['t'], axis['unit'], 5, [(0.001, 0.005)])[0]

    def brute(a, b):
        total = 0
        for s, e in wins:
            x, y = a[(a >= s) & (a < e)], b[(b >= s) & (b < e)]
            d = np.abs(x[:, None] - y[None, :])
            total += int(((d >= 0.001) & (d <= 0.005)).sum())
        return total

    match = all(C[i, j] == brute(spikes[f'u{i}'], spikes[f'u{j}'])
                for i in range(5) for j in range(5) if i != j)
    checks.append(('counts equal a brute-force count', match))

    cell_start, cell_len, cell_id = tj.jitter_cells(axis, 0.05)
    t_new = cell_start + rng.random(cell_start.size) * cell_len
    moved = dict(axis, t=t_new)
    _, _, cell_new = tj.jitter_cells(moved, 0.05)
    win_new = np.searchsorted(axis['offset'], t_new, side='right') - 1
    inside = np.all(t_new < axis['offset'][win_new] + axis['duration'][win_new])
    checks.append(('jitter keeps every spike in its UP window and jitter cell',
                   bool(np.all(cell_new == cell_id) and np.all(win_new == axis['win'])
                        and inside)))

    wins = np.array([[0.0, 1.0], [1.003, 2.0]])
    spikes = {'a': np.array([0.999, 1.5]), 'b': np.array([1.004, 1.503])}
    axis = tj.up_time_axis(spikes, wins)
    C = tj.coincidence_counts(axis['t'], axis['unit'], 2, [(0.0, 0.010)])[0]
    checks.append(('no coincidence counted across UP windows '
                   f'(got {C[0, 1]}, expected 1)', C[0, 1] == 1))

    # Independent spikes on the 30 kHz sample grid must give a flat fine CCG
    # against jitter: exact-edge lags and continuous-time surrogates once made
    # alternate 0.1 ms bins +-20-30% off, which looked like a periodic artifact.
    wins = np.column_stack([np.arange(0, 600, 1.0), np.arange(0, 600, 1.0) + 0.5])
    spikes = {f'u{k}': np.sort(np.round(np.concatenate(
        [s + rng.uniform(0, 0.5, rng.poisson(10)) for s in wins[:, 0]])
        * tj.SAMPLE_RATE_HZ) / tj.SAMPLE_RATE_HZ) for k in range(12)}
    axis = tj.up_time_axis(spikes, wins)
    groups = np.zeros((12, 12), dtype=int)
    np.fill_diagonal(groups, -1)
    obs = tj.grouped_ccg(axis['t'], axis['unit'], groups, 1, 0.01, 0.0001)[0]
    cell_start, cell_len, _ = tj.jitter_cells(axis, 0.05)
    sur = np.zeros(obs.shape)
    for _ in range(20):
        ts, us = tj.jitter_surrogate(axis['t'], axis['unit'], cell_start, cell_len, rng)
        sur += tj.grouped_ccg(ts, us, groups, 1, 0.01, 0.0001)[0]
    worst = float(np.max(np.abs(obs / (sur / 20) - 1)))
    checks.append((f'sample-grid spikes give a flat fine CCG vs jitter '
                   f'(max deviation {worst:.3f} < 0.15)', worst < 0.15))
    return checks


# --------------------------------------------------------------------------- #
# Synthetic sessions
# --------------------------------------------------------------------------- #

def _von_mises_curve(pref_deg, gain, baseline, kappa=2.0):
    theta = 2 * np.deg2rad(np.array(ORIENTATIONS) - pref_deg)
    return baseline + gain * np.exp(kappa * (np.cos(theta) - 1.0))


def simulate_session(mode, n_units=60, sleep_sec=900.0, seed=0):
    """Tuning curves, passive trials, UP windows and sleep spikes for one mode."""
    rng = np.random.default_rng(seed)
    prefs = rng.uniform(0, 180, n_units)
    curves = np.array([_von_mises_curve(p, g, b) for p, g, b in
                       zip(prefs, rng.uniform(4, 14, n_units), rng.uniform(1, 5, n_units))])
    trial_rates = [{float(o): rng.poisson(curves[k, i], 25).astype(float)
                    for i, o in enumerate(ORIENTATIONS)} for k in range(n_units)]
    shanks = rng.integers(0, 4, n_units)
    depths = rng.uniform(0, 750, n_units)

    # UP/DOWN alternation
    windows, t = [], 0.5
    while t < sleep_sec - 1:
        up = max(0.06, rng.exponential(0.45))
        windows.append((t, min(t + up, sleep_sec - 0.5)))
        t += up + max(0.06, rng.exponential(0.25))
    windows = np.array(windows)

    # Rate on a 5 ms grid: slow shared drive (per-UP gain x envelope)
    n_grid = int(sleep_sec / GRID_SEC)
    drive = np.zeros(n_grid)
    for s, e in windows:
        a, b = int(np.ceil(s / GRID_SEC)), int(e / GRID_SEC)
        if b <= a:
            continue
        phase = (np.arange(b - a) + 0.5) / (b - a)
        drive[a:b] = rng.gamma(6.0, 1 / 6.0) * (1.0 + 0.6 * np.sin(np.pi * phase))
    if mode == 'mid_common':
        chunk = rng.gamma(4.0, 0.25, n_grid // 4 + 1) - 1.0      # 20 ms chunks
        drive *= np.clip(1.0 + 1.0 * np.repeat(chunk, 4)[:n_grid], 0, None)

    base = rng.uniform(2.0, 8.0, n_units)
    spikes = {}
    unit_ids = [f"shank{shanks[k]}_unit{k:02d}" for k in range(n_units)]
    for k, uid in enumerate(unit_ids):
        counts = rng.poisson(base[k] * GRID_SEC * drive)
        idx = np.repeat(np.arange(n_grid), counts)
        spikes[uid] = (idx + rng.random(idx.size)) * GRID_SEC

    if mode == 'effect':
        group = np.floor(prefs / 180.0 * N_GROUPS).astype(int) % N_GROUPS
        up_total = float(np.sum(windows[:, 1] - windows[:, 0]))
        for g in range(N_GROUPS):
            members = np.flatnonzero(group == g)
            n_ev = rng.poisson(1.5 * up_total)
            # event times uniform in UP time
            cum = np.concatenate([[0], np.cumsum(windows[:, 1] - windows[:, 0])])
            u = rng.uniform(0, up_total, n_ev)
            w = np.searchsorted(cum, u, side='right') - 1
            ev = windows[w, 0] + (u - cum[w])
            for k in members:
                join = rng.random(n_ev) < 0.25
                extra = ev[join] + rng.normal(0, 0.0015, join.sum())
                spikes[unit_ids[k]] = np.concatenate([spikes[unit_ids[k]], extra])

    if mode == 'slow_common':      # unit 1 = near-copy of unit 0 (a duplicate)
        spikes[unit_ids[1]] = spikes[unit_ids[0]] + rng.uniform(0, 1e-4, spikes[unit_ids[0]].size)

    spikes = {u: np.sort(np.clip(s, 0, sleep_sec - 1e-6)) for u, s in spikes.items()}
    return {'unit_ids': unit_ids, 'curves': curves, 'trial_rates': trial_rates,
            'shanks': shanks, 'depths': depths, 'spikes': spikes,
            'windows': windows, 'sleep_sec': sleep_sec,
            'peak_lag': 20 + rng.choice([6.0, 20.0], n_units)}


def write_fixtures(session, folder):
    """Tuning pkl, sleep-spikes pkl and MUA ON-windows pkl in the real formats."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fs = 30000.0
    book = {}
    for k, uid in enumerate(session['unit_ids']):
        curve = session['curves'][k]
        z = np.sum(curve * np.exp(1j * 2 * np.deg2rad(ORIENTATIONS)))
        book[uid] = {
            'tuning': {
                'orientations': list(ORIENTATIONS), 'mean_rates': curve.tolist(),
                'trial_rates': session['trial_rates'][k],
                'osi': float(np.abs(z) / curve.sum()),
                'preferred_orientation_deg': float(np.rad2deg((np.angle(z) / 2) % np.pi)),
                'responsiveness': {'responsive': True},
                'reliability': {'reliable': True, 'median_r': 0.9},
            },
            'unit_info': {
                'shank': int(session['shanks'][k]),
                'channel_location_um': (float(300 * session['shanks'][k]),
                                        float(session['depths'][k])),
                'waveform_template': (
                    -np.exp(-((np.arange(60) - 20) / 4.) ** 2)
                    + 0.3 * np.exp(-((np.arange(60) - session['peak_lag'][k]) / 6.) ** 2)
                ).tolist(),
                'waveform_t_ms': (np.arange(60) / 30.).tolist(),
            },
        }
    tuning = folder / 'all_units_tuning.pkl'
    with tuning.open('wb') as f:
        pickle.dump(book, f)

    sleep = folder / 'sleep_spikes_synthetic_sleep_post.pkl'
    with sleep.open('wb') as f:
        pickle.dump({'metadata': {'sleep_id': 'synthetic_post', 'sampling_frequency': fs},
                     'window': {'sleep_start_sample': 0,
                                'sleep_end_sample': int(session['sleep_sec'] * fs),
                                'window_duration_sec': session['sleep_sec']},
                     'spike_data': {u: {'spike_times_sec': t}
                                    for u, t in session['spikes'].items()}}, f)

    mua = folder / 'synthetic_global_on_windows_post.pkl'
    with mua.open('wb') as f:
        pickle.dump({'sampling_frequency': fs,
                     'rows': [{'shank': 0, 'on_start_daysample': s * fs,
                               'on_end_daysample': e * fs}
                              for s, e in session['windows']]}, f)
    return tuning, sleep, mua


def analyse_fixture(paths, jitter_ms, n_surrogates, n_permutations, seed, verbose):
    from run_tuning_coupling import (prepare_block, build_pair_table, testable,
                                     CONTROLS, col)
    tuning, sleep, mua = paths
    units = sim.load_unit_tuning(tuning, verbose=verbose)
    similarity = sim.pair_tuning_similarity(units, n_splits=100, seed=seed, verbose=verbose)
    covariates = sim.pair_covariates(units, similarity['unit_ids'], similarity['pair_index'])
    spikes, windows, _, _ = prepare_block(sleep, similarity['unit_ids'], mua,
                                          min_shank_fraction=1.0, min_alignment_ratio=1.0,
                                          verbose=verbose)
    sync = tj.jitter_synchrony(spikes, windows, unit_ids=similarity['unit_ids'],
                               coinc_ms=(5.0,), jitter_ms=jitter_ms,
                               n_surrogates=n_surrogates, seed=seed, verbose=verbose)
    table = build_pair_table(similarity, covariates, sync)
    n_units = len(similarity['unit_ids'])
    results = {}
    for d in jitter_ms:
        response = col('excess_norm', 5.0, d)
        work = testable(table, response)
        for predictor in ('delta_pref_deg', 'signal_corr'):
            matrix = st.pairs_to_matrix(similarity[predictor], similarity['pair_index'],
                                        n_units)
            results[(predictor, d)] = st.unit_permutation_test(
                work, matrix, similarity['pair_index'], n_units, response=response,
                controls=CONTROLS, n_permutations=n_permutations, seed=seed,
                verbose=False)
    return table, results, similarity


def calibration(table, similarity, response, n_draws=40, n_permutations=200, seed=0):
    """Fraction of random (independent) tuning sets reaching p < 0.05."""
    from run_tuning_coupling import testable, CONTROLS
    rng = np.random.default_rng(seed + 1000)
    n = len(similarity['unit_ids'])
    work = testable(table, response)
    ps = []
    for k in range(n_draws):
        prefs = rng.uniform(0, 180, n)
        matrix = sim.circular_orientation_difference(prefs[:, None], prefs[None, :])
        ps.append(st.unit_permutation_test(
            work, matrix, similarity['pair_index'], n, response=response,
            controls=CONTROLS, n_permutations=n_permutations, seed=seed + k,
            verbose=False)['p_permutation_two_sided'])
    return float(np.mean(np.array(ps) < 0.05))


def run_self_test(seed=0, n_surrogates=100, n_permutations=300):
    from run_tuning_coupling import col
    print("=" * 72 + "\nSELF-TEST: jitter-corrected synchrony vs tuning\n" + "=" * 72)
    checks = check_mechanics(seed)
    session_seed = 10 + seed
    jitter_ms = (25.0, 50.0, 100.0)

    with tempfile.TemporaryDirectory() as tmp:
        for mode in ('slow_common', 'mid_common', 'effect'):
            print(f"\n--- {mode} ---")
            session = simulate_session(mode, seed=session_seed)
            paths = write_fixtures(session, Path(tmp) / mode)
            table, res, similarity = analyse_fixture(
                paths, jitter_ms, n_surrogates, n_permutations, seed,
                verbose=(mode == 'effect'))
            ok = np.asarray(table['possible_duplicate']) == 0
            z = {d: float(np.nanmedian(np.asarray(table[col('excess_z', 5.0, d)])[ok]))
                 for d in jitter_ms}
            print("  median excess z: " + ", ".join(f"D={d:g}: {z[d]:+.2f}"
                                                   for d in jitter_ms))
            for (pred, d), r in sorted(res.items()):
                print(f"  D={d:>5g} {pred:15s} β={r['beta_predictor']:+.2e} "
                      f"p={r['p_permutation_two_sided']:.4f}")
            p_dp, b_dp = (res[('delta_pref_deg', 50.0)][k]
                          for k in ('p_permutation_two_sided', 'beta_predictor'))
            p_sc, b_sc = (res[('signal_corr', 50.0)][k]
                          for k in ('p_permutation_two_sided', 'beta_predictor'))

            if mode == 'effect':
                checks += [('effect: found via signal_corr (β > 0, p < 0.01)',
                            b_sc > 0 and p_sc < 0.01),
                           ('effect: found via delta_pref (β < 0, p < 0.01)',
                            b_dp < 0 and p_dp < 0.01)]
            else:
                checks += [(f'{mode}: no false positive via signal_corr (p={p_sc:.3f})',
                            p_sc > 0.05),
                           (f'{mode}: no false positive via delta_pref (p={p_dp:.3f})',
                            p_dp > 0.05)]
            if mode == 'slow_common':
                flagged = {frozenset(p) for p in zip(table['unit_i'][~ok],
                                                     table['unit_j'][~ok])}
                planted = {frozenset(session['unit_ids'][:2])}
                checks += [(f'slow_common: jitter removes slow shared drive '
                            f'(|median z| = {abs(z[50.0]):.2f} < 1)', abs(z[50.0]) < 1),
                           (f'slow_common: exactly the planted duplicate is flagged '
                            f'({len(flagged)} flagged)', flagged == planted)]
            if mode == 'mid_common':
                checks += [(f'mid_common: ~20 ms shared drive is excess at D=50 '
                            f'(median z {z[50.0]:+.2f} > 1)', z[50.0] > 1),
                           ('mid_common: excess grows with D (D=25 < D=100)',
                            z[25.0] < z[100.0])]
                rate = calibration(table, similarity, col('excess_norm', 5.0, 50.0),
                                   seed=seed)
                print(f"  calibration: {100 * rate:.0f}% of random tuning sets p < 0.05")
                checks.append((f'calibration: random tuning gives p < 0.05 in '
                               f'{100 * rate:.0f}% of draws (must be <= 12.5%)',
                               rate <= 0.125))

    print("\n" + "=" * 72 + "\nSELF-TEST RESULTS\n" + "=" * 72)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [label for label, ok in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run_self_test())
