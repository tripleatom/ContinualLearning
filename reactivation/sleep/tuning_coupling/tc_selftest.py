r"""Synthetic end-to-end check of the tuning-coupling pipeline.

Three datasets are simulated with the same machinery and analysed by exactly the
code the real run uses (same loaders, same pickle formats, same estimators).
They separate the two things that can make two neurons fire together:

``slow_common`` (null)
    UP states differ in overall excitability and each has a within-UP envelope,
    shared by every unit -- this IS "UP-state-wide drive".  The block shuffle
    preserves it, so excess co-activation must come out near zero for every pair
    and tuning must predict nothing.

``fast_common`` (null)
    A global drive shared by all units that fluctuates FASTER than the shuffle
    block.  Genuine fine-timescale co-firing, so excess z must be strongly
    positive for essentially every pair -- yet it carries no tuning structure, so
    the tuning coefficient must still be null.  This is the case the whole
    "excess" framing exists for: cortex-wide co-activation must not be mistaken
    for tuning-specific co-activation.

``effect``
    Slow drive as in the null, plus orientation-specific fast latents shared only
    by similarly tuned units.  The test must find it, with a positive slope on
    tuning-curve correlation and a negative slope on preferred-orientation
    difference.

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
import tc_coupling as cpl
import tc_stats as st


ORIENTATIONS = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]


def _von_mises_curve(pref_deg, gain, baseline, kappa=2.0):
    theta = 2 * np.deg2rad(np.array(ORIENTATIONS) - pref_deg)
    return baseline + gain * np.exp(kappa * (np.cos(theta) - 1.0))


def simulate_session(n_units=60, n_trials_per_ori=25, sleep_sec=900.0,
                     bin_sec=0.025, tuning_latent_gain=0.0, fast_common_gain=0.0,
                     slow_gain=0.8, up_mean_sec=0.45, down_mean_sec=0.25, seed=0):
    """
    Build one synthetic session: tuning curves, passive trials, sleep spikes.

    The sleep drive of unit k in bin t is

        rate_k(t) = base_k * g_up(t) * env(phase(t))              <- slow, shared
                    * (1 + fast_common_gain * C(t))               <- fast, shared
                    * (1 + tuning_latent_gain * L_{k}(t))         <- fast, tuning-specific

    where g_up is a per-UP-state excitability, env a within-UP envelope, C a
    global fast fluctuation and L a fast fluctuation shared only by units whose
    preferred orientations fall in the same bin of the 180 deg circle.  Setting
    the two fast gains chooses which dataset is being made.
    """
    rng = np.random.default_rng(seed)

    prefs = rng.uniform(0, 180, n_units)
    gains = rng.uniform(4.0, 14.0, n_units)
    baselines = rng.uniform(1.0, 5.0, n_units)
    shanks = rng.integers(0, 4, n_units)
    depths = rng.uniform(0, 750, n_units)
    curves = np.array([_von_mises_curve(p, g, b)
                       for p, g, b in zip(prefs, gains, baselines)])

    # ---- passive trials: Poisson counts in a 1 s window around each curve ----
    trial_rates = []
    for k in range(n_units):
        per_ori = {}
        for o, ori in enumerate(ORIENTATIONS):
            per_ori[float(ori)] = rng.poisson(curves[k, o], n_trials_per_ori).astype(float)
        trial_rates.append(per_ori)

    # ---- sleep: alternating UP/DOWN with per-UP excitability and an envelope --
    n_bins = int(sleep_sec / bin_sec)
    up_mask = np.zeros(n_bins, dtype=bool)
    up_index = np.full(n_bins, -1, dtype=int)
    phase = np.zeros(n_bins)
    t, u = 0, 0
    while t < n_bins:
        up_len = max(2, int(rng.exponential(up_mean_sec / bin_sec)))
        down_len = max(2, int(rng.exponential(down_mean_sec / bin_sec)))
        stop = min(t + up_len, n_bins)
        if stop > t:
            up_mask[t:stop] = True
            up_index[t:stop] = u
            phase[t:stop] = (np.arange(stop - t) + 0.5) / (stop - t)
            u += 1
        t += up_len + down_len
    n_up = max(u, 1)

    # Slow, shared drive: UP-state identity x within-UP envelope. This is the
    # confound the block shuffle is supposed to preserve (and therefore remove
    # from the excess measure).
    up_gain = rng.gamma(6.0, 1 / 6.0, n_up)
    slow = np.ones(n_bins)
    inside = up_index >= 0
    envelope = 1.0 + 0.6 * np.sin(np.pi * phase[inside])
    slow[inside] = (1.0 + slow_gain * (up_gain[up_index[inside]] - 1.0)) * envelope

    # Fast fluctuations, on the bin timescale.
    common_fast = rng.gamma(4.0, 0.25, n_bins) - 1.0
    n_latent = 8
    latents = rng.gamma(4.0, 0.25, (n_latent, n_bins)) - 1.0
    latent_of_unit = np.floor(prefs / 180.0 * n_latent).astype(int) % n_latent

    sleep_rate = rng.uniform(1.5, 9.0, n_units)      # mean in-UP rate, Hz
    spikes = {}
    unit_ids = [f"shank{shanks[k]}_unit{k}" for k in range(n_units)]
    for k, uid in enumerate(unit_ids):
        drive = slow.copy()
        if fast_common_gain > 0:
            drive = drive * (1.0 + fast_common_gain * common_fast)
        if tuning_latent_gain > 0:
            drive = drive * (1.0 + tuning_latent_gain * latents[latent_of_unit[k]])
        lam = np.clip(sleep_rate[k] * bin_sec * drive, 0, None) * up_mask
        counts = rng.poisson(lam)
        idx = np.repeat(np.arange(n_bins), counts)
        spikes[uid] = np.sort((idx + rng.random(idx.size)) * bin_sec)

    peak_lag = 20 + rng.choice([6.0, 20.0], n_units)   # narrow vs wide spiking

    return {'unit_ids': unit_ids, 'prefs': prefs, 'curves': curves, 'peak_lag': peak_lag,
            'trial_rates': trial_rates, 'shanks': shanks, 'depths': depths,
            'spikes': spikes, 'sleep_sec': sleep_sec, 'up_mask': up_mask,
            'bin_sec': bin_sec, 'latent_of_unit': latent_of_unit}


def write_fixtures(session, folder):
    """Write the synthetic session in the real pickle formats."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    book = {}
    for k, uid in enumerate(session['unit_ids']):
        curve = session['curves'][k]
        theta = 2 * np.deg2rad(ORIENTATIONS)
        z = np.sum(curve * np.exp(1j * theta))
        book[uid] = {
            'unit_id': uid,
            'tuning': {
                'orientations': list(ORIENTATIONS),
                'mean_rates': curve.tolist(),
                'trial_rates': session['trial_rates'][k],
                'osi': float(np.abs(z) / (curve.sum() + 1e-12)),
                'preferred_orientation_deg': float(np.rad2deg((np.angle(z) / 2) % np.pi)),
                'modulation_index': float((curve.max() - curve.min()) /
                                          (curve.max() + curve.min())),
                'responsiveness': {'responsive': True, 'delta_rate_hz': 5.0,
                                   'p_wilcoxon': 1e-9},
                'reliability': {'reliable': True, 'median_r': 0.9,
                                'p_perm_reliability': 0.001},
            },
            'unit_info': {
                'shank': int(session['shanks'][k]),
                'channel_location_um': (float(300 * session['shanks'][k]),
                                        float(session['depths'][k])),
                # Peak latency varies per unit so both cell types appear and the
                # pair_type control is actually exercised.
                'waveform_template': (
                    -np.exp(-((np.arange(60) - 20) / 4.) ** 2)
                    + 0.3 * np.exp(-((np.arange(60) - session['peak_lag'][k]) / 6.) ** 2)
                ).tolist(),
                'waveform_t_ms': (np.arange(60) / 30.).tolist(),
                'n_spikes_total': int(session['spikes'][uid].size),
            },
        }
    tuning_path = folder / 'all_units_tuning.pkl'
    with tuning_path.open('wb') as f:
        pickle.dump(book, f)

    sleep_path = folder / 'sleep_spikes_synthetic_sleep_post.pkl'
    with sleep_path.open('wb') as f:
        pickle.dump({
            'metadata': {'sleep_id': 'synthetic_post', 'sampling_frequency': 30000.0,
                         'n_units': len(session['unit_ids'])},
            'window': {'sleep_start_sample': 0,
                       'sleep_end_sample': int(session['sleep_sec'] * 30000),
                       'window_duration_sec': session['sleep_sec']},
            'spike_data': {uid: {'spike_times_sec': times, 'n_spikes': times.size,
                                 'unit_id': k, 'shank': int(session['shanks'][k]),
                                 'quality': 'good'}
                           for k, (uid, times) in enumerate(session['spikes'].items())},
        }, f)
    return tuning_path, sleep_path


def analyse_fixture(tuning_path, sleep_path, n_surrogates=100, n_permutations=500,
                    seed=0, verbose=True):
    """Run the real pipeline on synthetic files and return the fitted results."""
    units = sim.load_unit_tuning(tuning_path, verbose=verbose)
    similarity = sim.pair_tuning_similarity(units, n_splits=100, seed=seed,
                                            verbose=verbose)
    covariates = sim.pair_covariates(units, similarity['unit_ids'],
                                     similarity['pair_index'])

    spikes, meta = cpl.load_sleep_spikes(sleep_path, unit_ids=similarity['unit_ids'],
                                         verbose=verbose)
    windows, _ = cpl.up_windows_from_population_rate(spikes, meta['duration_sec'],
                                                     verbose=verbose)
    cpl.check_up_alignment(spikes, windows, meta['duration_sec'], min_ratio=1.0,
                           verbose=verbose)
    binned = cpl.bin_within_up(spikes, windows, verbose=verbose)
    coupling = cpl.pair_coupling(binned, n_surrogates=n_surrogates, seed=seed,
                                 verbose=verbose)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_tuning_coupling import build_pair_table, CONTROLS
    table = build_pair_table(units, similarity, covariates, coupling)

    n_units = len(similarity['unit_ids'])
    out = {}
    for predictor in ('delta_pref_deg', 'signal_corr'):
        matrix = st.pairs_to_matrix(similarity[predictor], similarity['pair_index'],
                                    n_units)
        for measure in ('excess_z', 'raw_corr_z', 'resid_corr_z'):
            out[(predictor, measure)] = st.unit_permutation_test(
                table, matrix, similarity['pair_index'], n_units, response=measure,
                controls=CONTROLS, n_permutations=n_permutations, seed=seed,
                verbose=False)
    return table, out, coupling


def run_self_test(seed=0, n_units=60, sleep_sec=900.0, n_permutations=500):
    """Simulate, analyse, and check the answers against what was planted."""
    print("=" * 72)
    print("SELF-TEST: synthetic sessions with a known answer")
    print("=" * 72)

    datasets = (
        # name,          tuning latent, fast common,  what must come out
        ('slow_common',  0.0,           0.0),
        ('fast_common',  0.0,           1.1),
        ('effect',       1.1,           0.0),
    )

    checks, summary = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, latent_gain, fast_gain in datasets:
            print(f"\n{'-' * 72}\n{name.upper()}: tuning latent gain {latent_gain}, "
                  f"global fast gain {fast_gain}\n{'-' * 72}")
            session = simulate_session(n_units=n_units, sleep_sec=sleep_sec,
                                       tuning_latent_gain=latent_gain,
                                       fast_common_gain=fast_gain, seed=seed)
            folder = Path(tmp) / name
            tuning_path, sleep_path = write_fixtures(session, folder)
            table, results, _ = analyse_fixture(
                tuning_path, sleep_path, seed=seed, n_permutations=n_permutations,
                verbose=(name == 'effect'))

            raw_med = float(np.nanmedian(table['raw_corr']))
            exc_med = float(np.nanmedian(table['excess_z']))
            beta_sc = results[('signal_corr', 'excess_z')]['beta_predictor']
            p_sc = results[('signal_corr', 'excess_z')]['p_permutation_two_sided']
            beta_dp = results[('delta_pref_deg', 'excess_z')]['beta_predictor']
            p_dp = results[('delta_pref_deg', 'excess_z')]['p_permutation_two_sided']
            beta_raw = results[('signal_corr', 'raw_corr_z')]['beta_predictor']
            p_raw = results[('signal_corr', 'raw_corr_z')]['p_permutation_two_sided']
            summary[name] = dict(raw_med=raw_med, exc_med=exc_med, beta_sc=beta_sc,
                                 p_sc=p_sc, beta_dp=beta_dp, p_dp=p_dp)

            print(f"\n  median RAW co-firing correlation : {raw_med:+.4f}")
            print(f"  median EXCESS z                  : {exc_med:+.3f}")
            print(f"  excess_z ~ signal_corr    : beta={beta_sc:+.4f}  p={p_sc:.4f}")
            print(f"  excess_z ~ delta_pref_deg : beta={beta_dp:+.4f}  p={p_dp:.4f}")
            print(f"  raw_corr_z ~ signal_corr  : beta={beta_raw:+.4f}  p={p_raw:.4f}")

            if name == 'effect':
                checks += [
                    ('effect: found via tuning-curve correlation (beta > 0)',
                     p_sc < 0.01 and beta_sc > 0),
                    ('effect: found via preferred-orientation difference (beta < 0)',
                     p_dp < 0.01 and beta_dp < 0),
                ]
            else:
                checks += [
                    (f'{name}: no false positive via tuning-curve correlation',
                     p_sc > 0.05),
                    (f'{name}: no false positive via preferred-orientation difference',
                     p_dp > 0.05),
                ]
            if name == 'slow_common':
                checks.append(
                    ('slow_common: shuffle removes UP-state-wide drive '
                     f'(|median excess z| = {abs(exc_med):.2f} < 1)', abs(exc_med) < 1.0))
            if name == 'fast_common':
                checks.append(
                    ('fast_common: fine-timescale global drive DOES raise excess z '
                     f'(median {exc_med:+.2f} > 1)', exc_med > 1.0))
            checks.append((f'{name}: raw co-firing is positive for a typical pair',
                           raw_med > 0.0))

    print("\n" + "=" * 72)
    print("SELF-TEST RESULTS")
    print("=" * 72)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [label for label, ok in checks if not ok]
    print("=" * 72)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if not failed:
        print("\nThe estimator separates the three cases it has to separate:\n"
              "  UP-state-wide drive  -> no excess, no tuning effect\n"
              "  global fast drive    -> excess for everyone, still no tuning effect\n"
              "  tuning-specific drive-> excess that tracks tuning similarity")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run_self_test())
