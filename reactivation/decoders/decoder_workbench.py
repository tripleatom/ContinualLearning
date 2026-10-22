"""
Decoder workbench: one configurable replacement for the compare_* scripts.

Firing rate (every unit, binned) is ALWAYS an input.  You choose, on top of it:

  context     task | passive | merged (task + passive stacked)
  features    any of  speed        -> speed_cms
                      heading      -> sin_heading, cos_heading
                      head_angle   -> sin_head_angle, cos_head_angle
                      position     -> pos_x, pos_y (arena position)
  classifiers any of  Gaussian NB, Poisson NB, LDA, Logistic Reg, SVM,
                      Random Forest, AODE
  bin sizes   ms, swept

Labels, CV and balancing are exactly those of the compare_* scripts: +1 / -1 =
left grating matches params.class_pos / class_neg during stimulus-on, 0 = ITI;
stratified k-fold, training folds undersampled to balance (class x context for
merged), test folds at natural ratio.

With features selected, a firing-rate-only run on the SAME bins is done for a
paired comparison (disable with --no-baseline), and the difference is
bootstrapped over CV repeats.  --shuffles N adds a label-permutation null.

--speed-match (task context) is the running-speed control: stim-on and ITI bins
usually differ in speed, so the ITI is sub-sampled until its speed histogram matches
the stim-on one and compared with a uniformly sub-sampled ITI of the same size.  The
two arms then replace the with/without-features arms (features may still be added to
both).  The speed histograms and a KS check are saved as a figure.

Conventions inherited from the existing scripts
  * merged context: passive rows (head-fixed) get zeros in the kinematic columns.
  * bins with no clean DLC tracking are dropped (from both arms) when features
    are selected.
Deliberate difference
  * Poisson NB and AODE model spike counts / binarised spikes; they always see
    the firing-rate columns only, so for them "with features" == baseline.

Examples
    python decoder_workbench.py                                   # rate only, task
    python decoder_workbench.py --context merged --features speed heading
    python decoder_workbench.py --features speed --classifiers "Random Forest" LDA --bins 100 200 --repeats 3 --shuffles 10
    python decoder_gui.py                                         # same thing, with checkboxes
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'DiscriminationTask' / 'grating'))

import params as P

CONTEXTS = ('task', 'passive', 'merged')
# How training folds are balanced for the merged context:
#   class_group  every (class x task/passive) cell equalised (train_task_test_passive.py);
#                also reports test accuracy split by context.
#   class        classes equalised only (compare_merged_*.py).
MERGED_BALANCE = ('class_group', 'class')
CLASSIFIER_NAMES = ('Gaussian NB', 'Poisson NB', 'LDA', 'Logistic Reg', 'SVM',
                    'Random Forest', 'AODE')
RATE_ONLY_CLASSIFIERS = ('Poisson NB', 'AODE')
FEATURE_GROUPS = {
    'speed':      ['speed_cms'],
    'heading':    ['sin_heading', 'cos_heading'],
    'head_angle': ['sin_head_angle', 'cos_head_angle'],
    'position':   ['pos_x', 'pos_y'],
}
FEATURE_HELP = {
    'speed':      'running speed, cm/s',
    'heading':    'movement direction, as sin and cos',
    'head_angle': 'head angle, as sin and cos',
    'position':   'arena x, y',
}


def feature_columns(features):
    return [col for group in FEATURE_GROUPS if group in features
            for col in FEATURE_GROUPS[group]]


def default_config():
    return {
        'context': 'task',
        'features': [],
        'classifiers': list(CLASSIFIER_NAMES),
        'bin_sizes_ms': list(P.bin_sizes_ms),
        'n_splits': int(P.n_splits),
        'n_repeats': int(P.n_repeats),
        'n_shuffles': 0,
        'random_state': int(P.random_state),
        'baseline': True,
        'merged_balance': 'class_group',
        'count_models_see_features': False,
        'speed_match': False,
        'task_pkl': P.task_pkl,
        'passive_pkl': P.passive_pkl,
        'out_dir': None,
    }


def validate(cfg):
    if cfg['context'] not in CONTEXTS:
        raise ValueError(f"context must be one of {CONTEXTS}")
    unknown = [f for f in cfg['features'] if f not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown features {unknown}; choose from {list(FEATURE_GROUPS)}")
    if cfg.get('merged_balance', 'class_group') not in MERGED_BALANCE:
        raise ValueError(f"merged_balance must be one of {MERGED_BALANCE}")
    if cfg.get('speed_match') and cfg['context'] != 'task':
        raise ValueError("speed matching uses the task session's tracking: choose the task context")
    if cfg['features'] and cfg['context'] == 'passive':
        raise ValueError("passive recordings are head-fixed: there is no kinematics to add. "
                         "Use context 'task' or 'merged', or clear the features.")
    unknown = [c for c in cfg['classifiers'] if c not in CLASSIFIER_NAMES]
    if unknown or not cfg['classifiers']:
        raise ValueError(f"classifiers must be a non-empty subset of {list(CLASSIFIER_NAMES)}")
    if not cfg['bin_sizes_ms']:
        raise ValueError("need at least one bin size")
    if cfg['n_repeats'] < 1 or cfg['n_splits'] < 2:
        raise ValueError("need n_repeats >= 1 and n_splits >= 2")


# ------------------------------------------------------------------ #
#  Data                                                               #
# ------------------------------------------------------------------ #

def build_dataset(cfg, bin_ms, kin_samples):
    """Return (X_rate, kin, y, groups, speeds) for one bin size.

    X_rate : (n, n_units) firing rates;  kin : (n, n_kin) or None;
    groups : None, or 0 = task / 1 = passive rows for the merged context;
    speeds : per-bin mean speed (cm/s) when cfg['speed_match'], else None.
    """
    import numpy as np
    from prepare_task_stimtype import prepare_task_stim_type
    from prepare_passive_stimtype import prepare_passive_stim_type
    from kinematics_utils import build_task_kinematics

    sec = bin_ms / 1000.0
    cols = feature_columns(cfg['features'])
    task = passive = None

    if cfg['context'] in ('task', 'merged'):
        X, y, bc, units = prepare_task_stim_type(
            cfg['task_pkl'], P.class_pos, P.class_neg, P.TASK_COL_MAP,
            bin_size_sec=sec, balance_classes=False, random_state=cfg['random_state'])
        kin = speeds = keep = None
        if cfg.get('speed_match'):
            sp, keep = build_task_kinematics(bc, sec, kin_samples, columns=['speed_cms'])
            speeds = sp[:, 0]
        if cols:
            kin, keep_k = build_task_kinematics(bc, sec, kin_samples, columns=cols)
            keep = keep_k if keep is None else (keep & keep_k)
        if keep is not None:
            print(f"  [task] {int((~keep).sum())} bins dropped (no clean tracking)")
            X, y = X[keep], y[keep]
            if kin is not None:
                kin = kin[keep].astype(X.dtype)
            if speeds is not None:
                speeds = speeds[keep]
        task = (X, kin, y, units, speeds)

    if cfg['context'] in ('passive', 'merged'):
        X, y, _, units = prepare_passive_stim_type(
            cfg['passive_pkl'], P.class_pos, P.class_neg, P.PASSIVE_COL_MAP,
            bin_size_sec=sec, balance_classes=False, random_state=cfg['random_state'])
        passive = (X, None, y, units)

    if cfg['context'] == 'task':
        return task[0], task[1], task[2], None, task[4]
    if cfg['context'] == 'passive':
        return passive[0], None, passive[2], None, None

    common = sorted(set(task[3]) & set(passive[3]))
    if not common:
        raise RuntimeError("No common units between task and passive recordings.")
    print(f"  [merged] common units: {len(common)}  (task={len(task[3])}, passive={len(passive[3])})")
    X_t = task[0][:, [task[3].index(u) for u in common]]
    X_p = passive[0][:, [passive[3].index(u) for u in common]]
    kin = None
    if cols:
        kin = np.vstack([task[1], np.zeros((X_p.shape[0], len(cols)), dtype=X_t.dtype)])
    y = np.concatenate([task[2], passive[2]])
    groups = np.concatenate([np.zeros(len(task[2]), dtype=int), np.ones(len(passive[2]), dtype=int)])
    return np.vstack([X_t, X_p]), kin, y, groups, None


# ------------------------------------------------------------------ #
#  One CV evaluation                                                  #
# ------------------------------------------------------------------ #

def cv_once(name, proto, X, y, groups, n_splits, seed):
    """-> (overall, {class: recall}, {group: acc} or None, {group: {class: recall}} or None)."""
    from decode_utils import run_cv_balanced_train, run_cv_balanced_train_grouped
    if groups is None:
        _, mean, _, pcm, _ = run_cv_balanced_train(name, proto, X, y, n_splits=n_splits,
                                                   random_state=seed)
        return mean, pcm, None, None
    return run_cv_balanced_train_grouped(name, proto, X, y, groups, n_splits=n_splits,
                                         random_state=seed)


def null_once(name, proto, X, y, groups, n_splits, seed, n_shuffles):
    from decode_utils import run_cv_balanced_train_shuffle, run_cv_balanced_train_shuffle_grouped
    if groups is None:
        mean, std, _, _ = run_cv_balanced_train_shuffle(name, proto, X, y, n_splits, seed,
                                                        n_shuffles=n_shuffles)
    else:
        mean, std, _, _ = run_cv_balanced_train_shuffle_grouped(name, proto, X, y, groups,
                                                                n_splits, seed,
                                                                n_shuffles=n_shuffles)
    return mean, std


# ------------------------------------------------------------------ #
#  Speed-matching diagnostics                                         #
# ------------------------------------------------------------------ #

def speed_diagnostics(cfg, kin_samples):
    """Speed histograms of stim-on vs matched / uniform ITI at params.hist_bin_ms, with KS tests."""
    import numpy as np
    from scipy.stats import ks_2samp
    import speed_matching as sm

    X, _, y, _, sp = build_dataset(cfg, P.hist_bin_ms, kin_samples)
    n_target = sm.iti_target_count(y)
    _, y_m, sp_m, _, edges = sm.match_iti_to_stimon(
        X, y, sp, np.random.default_rng(cfg['random_state']), n_target)
    _, y_u, sp_u = sm.downsample_iti_uniform(
        X, y, sp, np.random.default_rng(cfg['random_state'] + 1), n_target)
    ks_m = ks_2samp(sp[y != 0], sp_m[y_m == 0])
    ks_u = ks_2samp(sp[y != 0], sp_u[y_u == 0])
    print(f"  speed check @ {P.hist_bin_ms} ms: n target ITI (= min stim-on class) = {n_target}")
    print(f"  KS(stim-on vs matched   ITI): stat={ks_m.statistic:.3f}  p={ks_m.pvalue:.2e}")
    print(f"  KS(stim-on vs unmatched ITI): stat={ks_u.statistic:.3f}  p={ks_u.pvalue:.2e}")
    return {'hist_bin_ms': P.hist_bin_ms, 'n_target': int(n_target), 'edges': edges,
            'speeds': sp, 'y': y, 'speeds_matched': sp_m, 'y_matched': y_m,
            'speeds_unmatched': sp_u, 'y_unmatched': y_u,
            'ks_matched': (float(ks_m.statistic), float(ks_m.pvalue)),
            'ks_unmatched': (float(ks_u.statistic), float(ks_u.pvalue))}


# ------------------------------------------------------------------ #
#  Sweep                                                              #
# ------------------------------------------------------------------ #

def _blank_arm(names):
    return {n: {'per_repeat': [], 'per_class': {}, 'per_group': {}} for n in names}


def run_sweep(cfg):
    import numpy as np
    from decode_utils import make_classifiers
    from kinematics_utils import load_task_kinematic_samples, print_kinematic_sample_report

    validate(cfg)
    cols = feature_columns(cfg['features'])
    names = list(cfg['classifiers'])
    protos = make_classifiers(cfg['random_state'])
    speed_mode = bool(cfg.get('speed_match'))
    arms = ['selected'] + (['baseline'] if speed_mode or (cols and cfg['baseline']) else [])
    if speed_mode:
        import speed_matching as sm
        arm_labels = {'selected': 'ITI speed-matched', 'baseline': 'ITI uniform (unmatched)'}
        arm_short = {'selected': 'matched', 'baseline': 'unmatched'}
        delta_label = 'matched - unmatched'
    else:
        arm_labels = {'selected': 'with features' if cols else 'rate only', 'baseline': 'rate only'}
        arm_short = {'selected': 'selected', 'baseline': 'baseline'}
        delta_label = 'with - rate only'
    rate_only = () if cfg.get('count_models_see_features') else RATE_ONLY_CLASSIFIERS
    bins = list(cfg['bin_sizes_ms'])

    kin_samples = speed_diag = None
    if cols or speed_mode:
        kin_samples, stats, offset = load_task_kinematic_samples(cfg['task_pkl'])
        print_kinematic_sample_report(kin_samples, stats, offset)
    if speed_mode:
        speed_diag = speed_diagnostics(cfg, kin_samples)

    res = {a: _blank_arm(names) for a in arms}
    null = {n: {'means': [], 'stds': []} for n in names}
    chance, sweep_stats = [], []
    work_per_bin = len(names) * cfg['n_repeats'] * len(arms) + (len(names) if cfg['n_shuffles'] else 0)
    total, done = work_per_bin * len(bins), 0
    t0 = time.time()

    for bi, bms in enumerate(bins):
        print(f"\n{'=' * 58}\n  Bin size: {bms} ms   ({bi + 1}/{len(bins)})\n{'=' * 58}", flush=True)
        X_rate, kin, y, groups, speeds = build_dataset(cfg, bms, kin_samples)
        n_rate = X_rate.shape[1]
        cv_groups = groups if cfg.get('merged_balance', 'class_group') == 'class_group' else None
        n_target = sm.iti_target_count(y) if speed_mode else None
        X_sel = np.hstack([X_rate, kin]) if kin is not None else X_rate
        counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
        chance.append(max(counts.values()) / len(y))
        print(f"  n_bins={len(y)}  classes={counts}  units={n_rate}  "
              f"features={'rate' if kin is None else 'rate + ' + ', '.join(feature_columns(cfg['features']))}"
              f"  (dim {X_sel.shape[1]})", flush=True)
        sweep_stats.append({'bin_ms': bms, 'n_bins': int(len(y)), 'class_counts': counts,
                            'n_units': int(n_rate), 'dim': int(X_sel.shape[1])})

        for arm in arms:
            for n in names:
                res[arm][n]['per_repeat'].append([])
        for r in range(cfg['n_repeats']):
            if speed_mode:
                # Same scheme as the former velocity-matched script: the CV seed is fixed and
                # the repeats differ in which ITI bins are drawn.
                seed = cfg['random_state']
                rng_m = np.random.default_rng(cfg['random_state'] + 1000 * bms + 2 * r)
                rng_u = np.random.default_rng(cfg['random_state'] + 1000 * bms + 2 * r + 1)
                X_m, y_m, _, _, _ = sm.match_iti_to_stimon(X_sel, y, speeds, rng_m, n_target)
                X_u, y_u, _ = sm.downsample_iti_uniform(X_sel, y, speeds, rng_u, n_target)
                arm_data = {'selected': (X_m, y_m), 'baseline': (X_u, y_u)}
            else:
                seed = cfg['random_state'] + 1000 * bms + r
            for name in names:
                same_as_baseline = name in rate_only and not speed_mode
                cache = None
                for arm in arms:
                    if speed_mode:
                        X, y_arm = arm_data[arm]
                        if name in rate_only:
                            X = X[:, :n_rate]
                    else:
                        X = X_rate if (arm == 'baseline' or same_as_baseline) else X_sel
                        y_arm = y
                    if same_as_baseline and arm == 'baseline' and cache is not None:
                        out = cache
                    else:
                        out = cv_once(name, protos[name], X, y_arm, cv_groups, cfg['n_splits'], seed)
                        if same_as_baseline:
                            cache = out
                    mean, pcm, gm, gpcm = out
                    slot = res[arm][name]
                    slot['per_repeat'][bi].append(mean)
                    for c, v in pcm.items():
                        slot['per_class'].setdefault(c, [[] for _ in bins])[bi].append(v)
                    if gm is not None:
                        for g, v in gm.items():
                            slot['per_group'].setdefault(g, [[] for _ in bins])[bi].append(v)
                    done += 1
                    print(f"[progress] {done}/{total}", flush=True)
            print(f"  repeat {r + 1}/{cfg['n_repeats']} done", flush=True)
        if speed_mode:
            counts_m = np.unique(y_m, return_counts=True)[1]
            chance[-1] = float(counts_m.max() / counts_m.sum())
            sweep_stats[-1]['n_iti_target'] = int(n_target)

        if cfg['n_shuffles']:
            if speed_mode:
                X_n, y_n, _, _, _ = sm.match_iti_to_stimon(
                    X_sel, y, speeds, np.random.default_rng(cfg['random_state'] + bms), n_target)
            for name in names:
                if speed_mode:
                    X = X_n[:, :n_rate] if name in rate_only else X_n
                    y_null, null_seed = y_n, cfg['random_state']
                else:
                    X = X_rate if name in rate_only else X_sel
                    y_null, null_seed = y, cfg['random_state'] + 1000 * bms + 7777
                m, s = null_once(name, protos[name], X, y_null, cv_groups, cfg['n_splits'],
                                 null_seed, cfg['n_shuffles'])
                null[name]['means'].append(m)
                null[name]['stds'].append(s)
                done += 1
                print(f"[progress] {done}/{total}", flush=True)

        for name in names:
            line = "  " + f"{name:<14}"
            for arm in arms:
                v = np.array(res[arm][name]['per_repeat'][bi])
                line += f" {arm_short[arm]}={v.mean():.3f}±{v.std():.3f}"
            if len(arms) == 2:
                d = (np.array(res['selected'][name]['per_repeat'][bi])
                     - np.array(res['baseline'][name]['per_repeat'][bi])).mean()
                line += f"  Δ={d:+.3f}"
            print(line, flush=True)
        print(f"  >> {time.time() - t0:.0f}s elapsed", flush=True)

    return {'config': cfg, 'bin_sizes_ms': bins, 'feature_columns': cols,
            'arms': res, 'null': null if cfg['n_shuffles'] else None,
            'chance': chance, 'sweep_stats': sweep_stats,
            'rate_only_classifiers': list(rate_only),
            'speed_match': speed_mode, 'speed_diag': speed_diag,
            'arm_labels': arm_labels, 'delta_label': delta_label,
            'elapsed_sec': time.time() - t0}


# ------------------------------------------------------------------ #
#  Summaries                                                          #
# ------------------------------------------------------------------ #

def summarize(results, n_bootstrap=2000):
    """Per arm/classifier mean+std over repeats, per-class, per-group, and paired deltas."""
    import numpy as np
    rng = np.random.default_rng(results['config']['random_state'] + 7777)

    def ms(lists):
        return ([float(np.nanmean(v)) for v in lists], [float(np.nanstd(v)) for v in lists])

    out = {'arms': {}, 'delta': {}}
    for arm, per_clf in results['arms'].items():
        out['arms'][arm] = {}
        for name, d in per_clf.items():
            m, s = ms(d['per_repeat'])
            out['arms'][arm][name] = {
                'means': m, 'stds': s,
                'per_class': {c: dict(zip(('means', 'stds'), ms(v))) for c, v in d['per_class'].items()},
                'per_group': {g: dict(zip(('means', 'stds'), ms(v))) for g, v in d['per_group'].items()},
            }
    if 'baseline' in results['arms']:
        for name in results['arms']['selected']:
            sel = results['arms']['selected'][name]['per_repeat']
            base = results['arms']['baseline'][name]['per_repeat']
            rows = {'mean': [], 'ci_lo': [], 'ci_hi': [], 'p_two_sided': []}
            for a, b in zip(sel, base):
                d = np.array(a) - np.array(b)
                rows['mean'].append(float(d.mean()))
                if d.size > 1:
                    bs = d[rng.integers(0, d.size, size=(n_bootstrap, d.size))].mean(axis=1)
                    rows['ci_lo'].append(float(np.percentile(bs, 2.5)))
                    rows['ci_hi'].append(float(np.percentile(bs, 97.5)))
                    rows['p_two_sided'].append(float(np.mean(np.abs(bs - d.mean()) >= abs(d.mean()))))
                else:
                    rows['ci_lo'].append(float('nan')); rows['ci_hi'].append(float('nan'))
                    rows['p_two_sided'].append(float('nan'))
            out['delta'][name] = rows
    return out


# ------------------------------------------------------------------ #
#  Figure                                                             #
# ------------------------------------------------------------------ #

def repro_command(cfg):
    parts = ['python', 'reactivation/decoders/decoder_workbench.py', '--context', cfg['context']]
    if cfg['features']:
        parts += ['--features', *cfg['features']]
    parts += ['--classifiers', *[f'"{c}"' for c in cfg['classifiers']],
              '--bins', *map(str, cfg['bin_sizes_ms']),
              '--splits', str(cfg['n_splits']), '--repeats', str(cfg['n_repeats']),
              '--shuffles', str(cfg['n_shuffles']), '--seed', str(cfg['random_state'])]
    if not cfg['baseline']:
        parts.append('--no-baseline')
    if cfg.get('count_models_see_features'):
        parts.append('--count-models-see-features')
    if cfg.get('speed_match'):
        parts.append('--speed-match')
    if cfg['context'] == 'merged' and cfg.get('merged_balance', 'class_group') != 'class_group':
        parts += ['--merged-balance', cfg['merged_balance']]
    return ' '.join(parts)


def make_figure(results, summary, path):
    import numpy as np
    import matplotlib.pyplot as plt

    cfg = results['config']
    bins = results['bin_sizes_ms']
    names = list(cfg['classifiers'])
    arms = list(results['arms'])
    has_delta = bool(summary['delta'])
    L = results['arm_labels']
    merged = cfg['context'] == 'merged' and bool(next(iter(summary['arms']['selected'].values()))['per_group'])
    classes = sorted(next(iter(summary['arms']['selected'].values()))['per_class'])
    class_names = {1: f"+1  {P.class_pos}", -1: f"-1  {P.class_neg}", 0: "0  ITI"}
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    colors = {n: plt.get_cmap('tab10')(i) for i, n in enumerate(names)}

    n_rows = 2 + int(has_delta) + int(merged)
    fig = plt.figure(figsize=(5.2 * max(len(classes), 2), 4.2 * n_rows))
    gs = fig.add_gridspec(n_rows, len(classes))

    def style(ax, title, ylabel='Accuracy (recall)'):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Bin size (ms)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(bins)
        ax.set_xlim(min(bins) - 0.08 * max(bins), max(bins) * 1.08)
        ax.spines[['top', 'right']].set_visible(False)

    def lines(ax, getter, with_null=False):
        for i, name in enumerate(names):
            for arm in arms:
                d = getter(summary['arms'][arm][name])
                if d is None:
                    continue
                m, s = np.array(d['means']), np.array(d['stds'])
                kw = dict(linestyle='-' if arm == 'selected' else '--',
                          linewidth=1.6 if arm == 'selected' else 1.0)
                ax.plot(bins, m, marker=markers[i % 7], color=colors[name],
                        label=name if arm == 'selected' else None, **kw)
                if arm == 'selected':
                    ax.fill_between(bins, m - s, m + s, color=colors[name], alpha=0.12, linewidth=0)
            if with_null and results['null']:
                ax.plot(bins, results['null'][name]['means'], color=colors[name],
                        linestyle=':', linewidth=0.9, marker='x', markersize=4)
        ax.set_ylim(0, 1.02)

    ax = fig.add_subplot(gs[0, :])
    lines(ax, lambda d: d, with_null=True)
    ax.plot(bins, results['chance'], color='k', linestyle=':', linewidth=1.2, label='chance (majority)')
    feat = ', '.join(results['feature_columns']) or 'none'
    style(ax, f"Overall accuracy - {cfg['context']} - rate + [{feat}]"
              + (f"   (solid: {L['selected']}, dashed: {L['baseline']}, dotted: label-shuffle null)"
                 if results['null'] or len(arms) == 2 else ""), 'Accuracy')
    handles, labels = ax.get_legend_handles_labels()
    key = [plt.Line2D([], [], color='k', linestyle='-', label=L['selected'])]
    if len(arms) == 2:
        key.append(plt.Line2D([], [], color='k', linestyle='--', label=L['baseline']))
    if results['null']:
        key.append(plt.Line2D([], [], color='k', linestyle=':', marker='x', markersize=4, label='label-shuffle null'))
    ax.legend(handles + key, labels + [k.get_label() for k in key], fontsize=8, ncol=4, loc='lower right')

    for j, c in enumerate(classes):
        ax = fig.add_subplot(gs[1, j])
        lines(ax, lambda d, c=c: d['per_class'].get(c))
        style(ax, f"Recall: {class_names.get(c, c)}")

    row = 2
    if has_delta:
        ax = fig.add_subplot(gs[row, :])
        for i, name in enumerate(names):
            d = summary['delta'][name]
            m = np.array(d['mean'])
            ax.plot(bins, m, marker=markers[i % 7], color=colors[name], linewidth=1.3, label=name)
            lo, hi = np.array(d['ci_lo']), np.array(d['ci_hi'])
            if np.isfinite(lo).any():
                ax.fill_between(bins, lo, hi, color=colors[name], alpha=0.10, linewidth=0)
        ax.axhline(0, color='k', linewidth=0.8)
        style(ax, f"Paired difference ({results['delta_label']}), 95% bootstrap CI over CV repeats",
              'Δ accuracy')
        ax.legend(fontsize=8, ncol=4)
        row += 1
    if merged:
        for j, (g, label) in enumerate(((0, 'task rows'), (1, 'passive rows'))):
            ax = fig.add_subplot(gs[row, j])
            lines(ax, lambda d, g=g: d['per_group'].get(g))
            style(ax, f"Merged decoder, test accuracy on {label}")

    stamp = (f"Repro: {repro_command(cfg)}\n"
             f"Data: {Path(cfg['task_pkl']).name if cfg['context'] != 'passive' else ''} "
             f"{Path(cfg['passive_pkl']).name if cfg['context'] != 'task' else ''} | "
             f"{cfg['n_splits']}-fold CV, train balanced / test natural | "
             f"{'Poisson NB & AODE see the added features too' if cfg.get('count_models_see_features') else 'Poisson NB & AODE use rate columns only'} | "
             f"Generated {datetime.now():%Y-%m-%d %H:%M} | "
             f"folder: {Path(cfg['task_pkl']).parent}")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.text(0.5, 0.004, stamp, ha='center', va='bottom', fontsize=6.5, color='gray')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def best_bin(results, summary):
    """Bin size with the highest accuracy averaged over the selected classifiers."""
    import numpy as np
    per_clf = [summary['arms']['selected'][n]['means'] for n in results['config']['classifiers']]
    return results['bin_sizes_ms'][int(np.argmax(np.mean(per_clf, axis=0)))]


def z_at_bin(results, summary, bin_ms):
    """z of each arm's accuracy against the with-features label-shuffle null at one bin."""
    import numpy as np
    if not results['null']:
        return None
    i = results['bin_sizes_ms'].index(bin_ms)
    out = {}
    for name in results['config']['classifiers']:
        mu = float(results['null'][name]['means'][i])
        sd = float(max(results['null'][name]['stds'][i], 1e-9))
        row = {'shuffle_mean': mu, 'shuffle_std': sd,
               'acc_selected': summary['arms']['selected'][name]['means'][i]}
        row['z_selected'] = (row['acc_selected'] - mu) / sd
        if 'baseline' in summary['arms']:
            row['acc_baseline'] = summary['arms']['baseline'][name]['means'][i]
            row['z_baseline'] = (row['acc_baseline'] - mu) / sd
        out[name] = row
    return out


def print_z_table(z, bin_ms, labels):
    if not z:
        return
    print("\nz relative to the shuffle null at bin = %d ms" % bin_ms)
    for name, r in z.items():
        base = f"  {labels['baseline']} {r['acc_baseline']:.3f} (z={r['z_baseline']:+.1f})" if 'acc_baseline' in r else ''
        print(f"  {name:<14} shuffle {r['shuffle_mean']:.3f}±{r['shuffle_std']:.3f}{base}"
              f"  {labels['selected']} {r['acc_selected']:.3f} (z={r['z_selected']:+.1f})")


def print_delta_table(results, summary):
    if not summary['delta']:
        return
    print(f"\nPaired difference ({results['delta_label']}): mean [95% CI] p (two-sided)")
    for name, d in summary['delta'].items():
        for i, bms in enumerate(results['bin_sizes_ms']):
            print(f"  {name:<14} {bms:>4} ms  {d['mean'][i]:+.3f} [{d['ci_lo'][i]:+.3f}, "
                  f"{d['ci_hi'][i]:+.3f}]  p={d['p_two_sided'][i]:.3f}")


def make_bar_figure(results, summary, path, bin_ms):
    """Grouped bars at one bin size: rate only / with features / label-shuffle null."""
    import numpy as np
    import matplotlib.pyplot as plt

    cfg = results['config']
    names = list(cfg['classifiers'])
    i = results['bin_sizes_ms'].index(bin_ms)
    classes = sorted(next(iter(summary['arms']['selected'].values()))['per_class'])
    class_names = {1: f"+1  {P.class_pos}", -1: f"-1  {P.class_neg}", 0: "0  ITI"}
    series = []
    if 'baseline' in summary['arms']:
        series.append((results['arm_labels']['baseline'], '#9aa5b1', 'baseline'))
    series.append((results['arm_labels']['selected'], '#4C72B0', 'selected'))
    panels = [('Overall', None)] + [(class_names.get(c, c), c) for c in classes]

    n_bars = len(series) + (1 if results['null'] else 0)
    width = 0.8 / n_bars
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6), squeeze=False)
    for ax, (title, c) in zip(axes[0], panels):
        for k, (label, color, arm) in enumerate(series):
            vals, errs = [], []
            for n in names:
                d = summary['arms'][arm][n]
                d = d if c is None else d['per_class'][c]
                vals.append(d['means'][i]); errs.append(d['stds'][i])
            ax.bar(x + (k - (n_bars - 1) / 2) * width, vals, width, yerr=errs, color=color,
                   label=label, capsize=2)
        if results['null'] and c is None:
            ax.bar(x + (len(series) - (n_bars - 1) / 2) * width,
                   [results['null'][n]['means'][i] for n in names], width,
                   yerr=[results['null'][n]['stds'][i] for n in names], color='#d9d9d9',
                   edgecolor='k', linewidth=0.5, label='label-shuffle null', capsize=2)
        if c is None:
            ax.axhline(results['chance'][i], color='k', linestyle=':', linewidth=1, label='chance (majority)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('Accuracy / recall')
    axes[0][0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f"{cfg['context']} decoder at {bin_ms} ms - rate + [{', '.join(results['feature_columns']) or 'none'}]" + ("  - ITI speed control" if results['speed_match'] else ""),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.5, 0.004, f"Repro: {repro_command(cfg)}  | bar bin: {bin_ms} ms | "
                         f"generated {datetime.now():%Y-%m-%d %H:%M}",
             ha='center', va='bottom', fontsize=6.5, color='gray')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_speed_figure(results, path):
    """Per-bin speed: stim-on classes, uniformly sub-sampled ITI, speed-matched ITI."""
    import numpy as np
    import matplotlib.pyplot as plt

    d = results['speed_diag']
    cfg = results['config']
    edges = d['edges']
    mids = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    sp, y = d['speeds'], d['y']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharex=True)

    ax = axes[0]
    n_pos, _ = np.histogram(sp[y == 1], bins=edges)
    n_neg, _ = np.histogram(sp[y == -1], bins=edges)
    ax.bar(mids, n_pos, width=width * 0.45, align='edge', color='#2c7fb8', label=f'+1 (n={int((y == 1).sum())})')
    ax.bar(mids - width * 0.45, n_neg, width=width * 0.45, align='edge', color='#d95f0e',
           label=f'-1 (n={int((y == -1).sum())})')
    ax.set_title(f"Stim-on speed  (bin={d['hist_bin_ms']} ms)")

    n_stim, _ = np.histogram(sp[y != 0], bins=edges)
    for ax, key, color, label, ks in ((axes[1], 'unmatched', '#7f7f7f', 'ITI unmatched, uniform DS', d['ks_unmatched']),
                                      (axes[2], 'matched', '#41ab5d', 'ITI velocity-matched', d['ks_matched'])):
        s_k, y_k = d[f'speeds_{key}'], d[f'y_{key}']
        n_iti, _ = np.histogram(s_k[y_k == 0], bins=edges)
        ax.bar(mids, n_iti, width=width * 0.9, color=color, label=f'{label} (n={int(n_iti.sum())})')
        ax.step(edges[:-1], n_stim * (n_iti.sum() / max(n_stim.sum(), 1)), where='post', color='black',
                linewidth=1.4, label='stim-on (rescaled)')
        ax.set_title(f"ITI speed - {key}  (KS={ks[0]:.3f}, p={ks[1]:.1e})")
    for ax in axes:
        ax.set_xlabel('Speed (cm/s)')
        ax.set_ylabel('Bin count')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Per-bin running speed: stim-on vs ITI  |  speed bin {P.speed_bin_cms} cm/s up to "
                 f"p{P.speed_top_pct}(stim-on)  |  ITI target = smaller stim-on class = {d['n_target']}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.text(0.5, 0.004, f"Repro: {repro_command(cfg)}  | generated {datetime.now():%Y-%m-%d %H:%M}",
             ha='center', va='bottom', fontsize=6.5, color='gray')
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def parse_args(argv=None):
    d = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--context', choices=CONTEXTS, default=d['context'])
    ap.add_argument('--features', nargs='*', default=d['features'], choices=list(FEATURE_GROUPS),
                    help='kinematic properties added to the firing rates (default: none)')
    ap.add_argument('--classifiers', nargs='+', default=d['classifiers'], choices=CLASSIFIER_NAMES)
    ap.add_argument('--bins', nargs='+', type=int, default=d['bin_sizes_ms'], help='bin sizes, ms')
    ap.add_argument('--splits', type=int, default=d['n_splits'])
    ap.add_argument('--repeats', type=int, default=d['n_repeats'], help='CV repeats per cell')
    ap.add_argument('--shuffles', type=int, default=d['n_shuffles'],
                    help='label permutations for the null (0 = off)')
    ap.add_argument('--seed', type=int, default=d['random_state'])
    ap.add_argument('--merged-balance', choices=MERGED_BALANCE, default=d['merged_balance'],
                    help='merged context: equalise class x context cells (default) or class only')
    ap.add_argument('--count-models-see-features', action='store_true',
                    help='also feed the kinematic columns to Poisson NB and AODE (what the old '
                         'compare_* scripts did; default: they see firing rates only)')
    ap.add_argument('--speed-match', action='store_true',
                    help='task context: control for running speed by matching the ITI speed '
                         'histogram to the stim-on one (vs a uniformly sub-sampled ITI)')
    ap.add_argument('--no-baseline', action='store_true',
                    help='skip the rate-only comparison run')
    ap.add_argument('--bar-bin', type=int, default=None,
                    help='bin size (ms) for the bar chart (default: best mean accuracy)')
    ap.add_argument('--task-pkl', default=d['task_pkl'])
    ap.add_argument('--passive-pkl', default=d['passive_pkl'])
    ap.add_argument('--out-dir', default=None, help='default: <session folder>/reactivation')
    return ap.parse_args(argv)


def main(argv=None):
    import pickle
    args = parse_args(argv)
    cfg = {'context': args.context, 'features': list(args.features),
           'classifiers': list(args.classifiers), 'bin_sizes_ms': list(args.bins),
           'n_splits': args.splits, 'n_repeats': args.repeats, 'n_shuffles': args.shuffles,
           'random_state': args.seed, 'baseline': not args.no_baseline,
           'merged_balance': args.merged_balance,
           'count_models_see_features': args.count_models_see_features,
           'speed_match': args.speed_match,
           'task_pkl': args.task_pkl, 'passive_pkl': args.passive_pkl, 'out_dir': args.out_dir}
    validate(cfg)
    ref = cfg['passive_pkl'] if cfg['context'] == 'passive' else cfg['task_pkl']
    out_dir = Path(cfg['out_dir']) if cfg['out_dir'] else Path(ref).parent / 'reactivation'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config: {json.dumps({k: v for k, v in cfg.items() if k not in ('task_pkl', 'passive_pkl')})}")
    print(f"Output folder: {out_dir}", flush=True)

    results = run_sweep(cfg)
    summary = summarize(results)

    tag = '+'.join(['rate'] + cfg['features']) + ('_speedmatched' if cfg['speed_match'] else '')
    stem = (f"decoder_{cfg['context']}_{Path(ref).parent.name}_{tag}_{cfg['n_splits']}fold_"
            f"{datetime.now():%Y%m%d_%H%M%S}")
    pkl_path, png_path = out_dir / f"{stem}.pkl", out_dir / f"{stem}.png"
    bar_bin = args.bar_bin if args.bar_bin in results['bin_sizes_ms'] else best_bin(results, summary)
    summary['bar_bin_ms'] = bar_bin
    summary['z_at_bar_bin'] = z_at_bin(results, summary, bar_bin)
    with open(pkl_path, 'wb') as f:
        pickle.dump({**results, 'summary': summary, 'repro': repro_command(cfg)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    make_figure(results, summary, png_path)
    bar_path = out_dir / f"{stem}_bar{bar_bin}ms.png"
    make_bar_figure(results, summary, bar_path, bar_bin)
    print_delta_table(results, summary)
    print_z_table(summary['z_at_bar_bin'], bar_bin, results['arm_labels'])
    if results['speed_diag']:
        speed_path = out_dir / f"{stem}_speed_distributions.png"
        make_speed_figure(results, speed_path)
        print(f"[saved] {speed_path}")
    print(f"\nTotal {results['elapsed_sec'] / 60:.1f} min")
    print(f"[saved] {pkl_path}")
    print(f"[saved] {png_path}")
    print(f"[saved] {bar_path}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
