"""
Within-context decoding across task / passive / pooled, with the merged
decoder additionally evaluated separately on its task and passive halves.

For each classifier x bin size, computes these accuracies
(all stratified k-fold CV with balanced-training undersampling):

   1) passive_cv              : passive (neural only).
   2) merged_cv               : task (neural+kin) stacked with passive
                                (neural + zeros for kin).
   3) merged_neural_cv        : task (neural) stacked with passive (neural),
                                no kinematic columns.
   4) task_kin_cv             : task (neural + kin).
   5) task_neural_cv          : task (neural only).

Plus, for (2) and (3), the test-set accuracy is split by source context:
   merged_cv_task, merged_cv_passive,
   merged_neural_cv_task, merged_neural_cv_passive
(same fits as the parent merged_cv / merged_neural_cv — just sliced).

Useful comparisons:
  - (3)/(4) vs (1): is the stimulus more decodable in task or in passive
    at the within-context ceiling?
  - (2) vs (1)/(3): does pooling task and passive bins help either one?
  - merged_cv_task vs merged_cv_passive: where does the merged decoder
    actually pick up its accuracy?

Conditions (3), (4) use the SAME task bins (DLC keep_t mask applied to
all), so kin vs neural comparisons isolate the kinematic-column
contribution. Training-set balancing:
  - passive_cv / task_*_cv (single-context): undersample majority class
    each fold; evaluate on the natural-ratio test fold.
  - merged_cv / merged_neural_cv: undersample every (class x group)
    cell to the smallest non-empty cell. This decouples the class signal
    from the task-vs-passive base-rate confound (without it, the decoder
    can exploit the fact that, e.g., ITI bins are ~91% from the task
    context and +1 / -1 bins are mostly from passive). Evaluation is
    still on the natural-ratio test fold, and the test fold is also
    sliced by source context to produce the *_task / *_passive views.
n_repeats different RNG seeds give an estimate of mean / std under
different undersampling draws / CV splits.
"""

import sys
import pickle
import time
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'DiscriminationTask' / 'grating'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from decode_utils import (
    make_classifiers,
    run_cv_balanced_train, run_cv_balanced_train_grouped,
    run_cv_balanced_train_shuffle, run_cv_balanced_train_shuffle_grouped,
)
from kinematics_utils import (
    KINEMATIC_FEATURE_COLUMNS,
    N_KINEMATIC_FEATURES,
    build_task_kinematics,
    load_task_kinematic_samples,
    print_kinematic_sample_report,
)
from prepare_task_stimtype import prepare_task_stim_type, infer_rewarded_combination
from prepare_passive_stimtype import prepare_passive_stim_type

from params import (
    task_pkl, passive_pkl,
    bin_sizes_ms, n_splits, random_state, n_repeats,
    class_pos, class_neg, TASK_COL_MAP, PASSIVE_COL_MAP,
)

out_dir = Path(task_pkl).parent / 'reactivation'
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {out_dir}")

session = Path(task_pkl).parent.name
N_KIN = N_KINEMATIC_FEATURES

with open(task_pkl, 'rb') as _f:
    rewarded_combination = infer_rewarded_combination(pickle.load(_f)['trial_params'])
print(f"Class +1 (left grating matches): {class_pos}")
print(f"Class -1 (left grating matches): {class_neg}")
print(f"Rewarded grating in this session: {rewarded_combination}")


# ------------------------------------------------------------------ #
#  Load TASK DLC samples                                              #
# ------------------------------------------------------------------ #
kin_samples, clean_stats, offset = load_task_kinematic_samples(task_pkl)
print_kinematic_sample_report(kin_samples, clean_stats, offset)


# ------------------------------------------------------------------ #
#  Sweep bin sizes                                                    #
# ------------------------------------------------------------------ #
CONDITIONS = [
    'passive_cv', 'merged_cv', 'merged_neural_cv',
    'task_kin_cv', 'task_neural_cv',
]
# Extra views derived from the merged CV calls (no extra fit work, just splitting
# the held-out test set by source context). Stored/plotted but not iterated over.
EXTRA_CONDITIONS = [
    'merged_cv_task', 'merged_cv_passive',
    'merged_neural_cv_task', 'merged_neural_cv_passive',
]
ALL_CONDITIONS = CONDITIONS + EXTRA_CONDITIONS
CLASSIFIERS = make_classifiers(random_state)

results = {
    cond: {name: {'means': [], 'stds': [], 'per_class': {}} for name in CLASSIFIERS}
    for cond in ALL_CONDITIONS
}
# Permutation-null per (condition, classifier, bin). Same CV pipeline as the
# real run, but with y permuted N_SHUFFLES_NULL times — gives an empirical
# chance line that respects the (class+group)-balanced training scheme.
N_SHUFFLES_NULL = 10
null_results = {
    cond: {name: {'means': [], 'stds': []} for name in CLASSIFIERS}
    for cond in ALL_CONDITIONS
}
n_units_common = None
sweep_stats = []
n_bins_sweep = len(bin_sizes_ms)

t_sweep_start = time.time()
for bms_idx, bms in enumerate(bin_sizes_ms):
    print(f"\n{'=' * 60}")
    print(f"  Bin size: {bms} ms")
    print(f"{'=' * 60}")
    t_bin_start = time.time()
    bin_size_sec = bms / 1000.0

    # ---- Task ----
    print("[TASK]")
    X_t, y_t, bc_t, units_t = prepare_task_stim_type(
        task_pkl, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False,
        random_state=random_state,
    )
    kin_t, keep_t = build_task_kinematics(bc_t, bin_size_sec, kin_samples)
    n_dropped_t = int((~keep_t).sum())
    X_t, y_t, kin_t = X_t[keep_t], y_t[keep_t], kin_t[keep_t]

    # ---- Passive (head-fixed) ----
    print("[PASSIVE]")
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False,
        random_state=random_state,
    )
    kin_p_zero = np.zeros((X_p.shape[0], N_KIN), dtype=np.float64)

    # ---- Align units across contexts ----
    common_units = sorted(set(units_t) & set(units_p))
    if not common_units:
        raise RuntimeError("No common units between task and passive recordings.")
    idx_t = [units_t.index(u) for u in common_units]
    idx_p = [units_p.index(u) for u in common_units]
    X_t_a = X_t[:, idx_t]
    X_p_a = X_p[:, idx_p]
    if n_units_common is None:
        n_units_common = len(common_units)
        print(f"  Common units: {n_units_common}  "
              f"(task={len(units_t)}, passive={len(units_p)})")

    # ---- Feature matrices for the five within-context conditions ----
    X_t_kin    = np.hstack([X_t_a, kin_t.astype(X_t_a.dtype)])
    X_p_kin0   = np.hstack([X_p_a, kin_p_zero.astype(X_p_a.dtype)])
    X_t_neural = X_t_a
    X_p_neural = X_p_a
    X_merged          = np.vstack([X_t_kin, X_p_kin0])
    X_merged_neural   = np.vstack([X_t_neural, X_p_neural])
    y_merged          = np.concatenate([y_t, y_p])
    # 0 = task half, 1 = passive half — used to split merged-CV test accuracy.
    groups_merged = np.concatenate([
        np.zeros(X_t_kin.shape[0], dtype=int),
        np.ones(X_p_kin0.shape[0], dtype=int),
    ])
    GROUP_TO_COND = {
        'merged_cv':        {0: 'merged_cv_task',        1: 'merged_cv_passive'},
        'merged_neural_cv': {0: 'merged_neural_cv_task', 1: 'merged_neural_cv_passive'},
    }


    print(f"  [passive_cv]    X={X_p_neural.shape}  "
          f"+1={np.sum(y_p==1)}  -1={np.sum(y_p==-1)}  0={np.sum(y_p==0)}")
    print(f"  [merged_cv]     X={X_merged.shape}    "
          f"+1={np.sum(y_merged==1)}  -1={np.sum(y_merged==-1)}  0={np.sum(y_merged==0)}")
    print(f"  [merged_neural_cv] X={X_merged_neural.shape}  "
          f"+1={np.sum(y_merged==1)}  -1={np.sum(y_merged==-1)}  0={np.sum(y_merged==0)}")
    # Show (class x group) cell sizes for the merged training pool — these
    # are what the (class+group)-balanced undersampler equalises each fold.
    _cell_counts = {
        (int(c), int(g)): int(np.sum((y_merged == c) & (groups_merged == g)))
        for c in np.unique(y_merged) for g in (0, 1)
    }
    _nonempty = [n for n in _cell_counts.values() if n > 0]
    _min_cell = min(_nonempty) if _nonempty else 0
    print("  [merged class x group]  "
          + "  ".join(
              f"c={k[0]:+d},{'t' if k[1] == 0 else 'p'}={v}"
              for k, v in _cell_counts.items()
          )
          + f"  -> balanced per-cell ~{int(_min_cell * (n_splits - 1) / n_splits)}"
          + f"  (merged train size ~{int(_min_cell * (n_splits - 1) / n_splits) * len(_nonempty)})")
    print(f"  [task_kin_cv]   X={X_t_kin.shape}     "
          f"+1={np.sum(y_t==1)}  -1={np.sum(y_t==-1)}  0={np.sum(y_t==0)}")
    print(f"  [task_neural_cv] X={X_t_neural.shape} "
          f"+1={np.sum(y_t==1)}  -1={np.sum(y_t==-1)}  0={np.sum(y_t==0)}")

    # ---- Run all classifiers across n_repeats for all conditions ----
    total_iters = len(CLASSIFIERS) * n_repeats * len(CONDITIONS)
    pbar = tqdm(total=total_iters, desc=f"  bin {bms} ms", ncols=90, leave=True)

    rep_means = {cond: {n: [] for n in CLASSIFIERS} for cond in ALL_CONDITIONS}
    rep_pc    = {cond: {n: {} for n in CLASSIFIERS} for cond in ALL_CONDITIONS}

    for r in range(n_repeats):
        rep_seed = random_state + 1000 * bms + r
        for name, clf_proto in CLASSIFIERS.items():
            # (1) passive CV
            _, mean_r, _, pcm_r, _ = run_cv_balanced_train(
                name, clf_proto, X_p_neural, y_p,
                n_splits=n_splits, random_state=rep_seed,
            )
            rep_means['passive_cv'][name].append(mean_r)
            for c, m in pcm_r.items():
                rep_pc['passive_cv'][name].setdefault(c, []).append(m)
            pbar.set_postfix_str(f"r={r+1} passive_cv {name}")
            pbar.update(1)

            # (2) merged CV  (task+kin stacked with passive+kin=0)
            mean_r, pcm_r, gmeans_r, gpcm_r = run_cv_balanced_train_grouped(
                name, clf_proto, X_merged, y_merged, groups_merged,
                n_splits=n_splits, random_state=rep_seed,
            )
            rep_means['merged_cv'][name].append(mean_r)
            for c, m in pcm_r.items():
                rep_pc['merged_cv'][name].setdefault(c, []).append(m)
            for g, gcond in GROUP_TO_COND['merged_cv'].items():
                rep_means[gcond][name].append(gmeans_r[g])
                for c, m in gpcm_r[g].items():
                    rep_pc[gcond][name].setdefault(c, []).append(m)
            pbar.set_postfix_str(f"r={r+1} merged_cv {name}")
            pbar.update(1)

            # (3) merged CV, neural only (no kin columns at all)
            mean_r, pcm_r, gmeans_r, gpcm_r = run_cv_balanced_train_grouped(
                name, clf_proto, X_merged_neural, y_merged, groups_merged,
                n_splits=n_splits, random_state=rep_seed,
            )
            rep_means['merged_neural_cv'][name].append(mean_r)
            for c, m in pcm_r.items():
                rep_pc['merged_neural_cv'][name].setdefault(c, []).append(m)
            for g, gcond in GROUP_TO_COND['merged_neural_cv'].items():
                rep_means[gcond][name].append(gmeans_r[g])
                for c, m in gpcm_r[g].items():
                    rep_pc[gcond][name].setdefault(c, []).append(m)
            pbar.set_postfix_str(f"r={r+1} merged_neural_cv {name}")
            pbar.update(1)

            # (4) task+kin CV
            _, mean_r, _, pcm_r, _ = run_cv_balanced_train(
                name, clf_proto, X_t_kin, y_t,
                n_splits=n_splits, random_state=rep_seed,
            )
            rep_means['task_kin_cv'][name].append(mean_r)
            for c, m in pcm_r.items():
                rep_pc['task_kin_cv'][name].setdefault(c, []).append(m)
            pbar.set_postfix_str(f"r={r+1} task_kin_cv {name}")
            pbar.update(1)

            # (5) task neural-only CV
            _, mean_r, _, pcm_r, _ = run_cv_balanced_train(
                name, clf_proto, X_t_neural, y_t,
                n_splits=n_splits, random_state=rep_seed,
            )
            rep_means['task_neural_cv'][name].append(mean_r)
            for c, m in pcm_r.items():
                rep_pc['task_neural_cv'][name].setdefault(c, []).append(m)
            pbar.set_postfix_str(f"r={r+1} task_neural_cv {name}")
            pbar.update(1)
    pbar.close()

    # ---- Permutation null per (classifier x condition) for this bin ----
    null_pbar = tqdm(
        total=len(CLASSIFIERS) * len(CONDITIONS),
        desc=f"  null  {bms} ms ({N_SHUFFLES_NULL} shuffles)",
        ncols=90, leave=True,
    )
    null_seed = random_state + 1000 * bms + 7777
    for name, clf_proto in CLASSIFIERS.items():
        # passive_cv null
        nm, ns, _, _ = run_cv_balanced_train_shuffle(
            name, clf_proto, X_p_neural, y_p,
            n_splits=n_splits, random_state=null_seed, n_shuffles=N_SHUFFLES_NULL,
        )
        null_results['passive_cv'][name]['means'].append(nm)
        null_results['passive_cv'][name]['stds' ].append(ns)
        null_pbar.set_postfix_str(f"passive_cv {name}");  null_pbar.update(1)

        # merged_cv null (grouped — also fills merged_cv_task / _passive)
        nm, ns, gm, gs = run_cv_balanced_train_shuffle_grouped(
            name, clf_proto, X_merged, y_merged, groups_merged,
            n_splits=n_splits, random_state=null_seed, n_shuffles=N_SHUFFLES_NULL,
        )
        null_results['merged_cv'][name]['means'].append(nm)
        null_results['merged_cv'][name]['stds' ].append(ns)
        for g, gcond in GROUP_TO_COND['merged_cv'].items():
            null_results[gcond][name]['means'].append(gm[g])
            null_results[gcond][name]['stds' ].append(gs[g])
        null_pbar.set_postfix_str(f"merged_cv {name}");  null_pbar.update(1)

        # merged_neural_cv null
        nm, ns, gm, gs = run_cv_balanced_train_shuffle_grouped(
            name, clf_proto, X_merged_neural, y_merged, groups_merged,
            n_splits=n_splits, random_state=null_seed, n_shuffles=N_SHUFFLES_NULL,
        )
        null_results['merged_neural_cv'][name]['means'].append(nm)
        null_results['merged_neural_cv'][name]['stds' ].append(ns)
        for g, gcond in GROUP_TO_COND['merged_neural_cv'].items():
            null_results[gcond][name]['means'].append(gm[g])
            null_results[gcond][name]['stds' ].append(gs[g])
        null_pbar.set_postfix_str(f"merged_neural_cv {name}");  null_pbar.update(1)

        # task_kin_cv null
        nm, ns, _, _ = run_cv_balanced_train_shuffle(
            name, clf_proto, X_t_kin, y_t,
            n_splits=n_splits, random_state=null_seed, n_shuffles=N_SHUFFLES_NULL,
        )
        null_results['task_kin_cv'][name]['means'].append(nm)
        null_results['task_kin_cv'][name]['stds' ].append(ns)
        null_pbar.set_postfix_str(f"task_kin_cv {name}");  null_pbar.update(1)

        # task_neural_cv null
        nm, ns, _, _ = run_cv_balanced_train_shuffle(
            name, clf_proto, X_t_neural, y_t,
            n_splits=n_splits, random_state=null_seed, n_shuffles=N_SHUFFLES_NULL,
        )
        null_results['task_neural_cv'][name]['means'].append(nm)
        null_results['task_neural_cv'][name]['stds' ].append(ns)
        null_pbar.set_postfix_str(f"task_neural_cv {name}");  null_pbar.update(1)
    null_pbar.close()

    for cond in ALL_CONDITIONS:
        for name in CLASSIFIERS:
            results[cond][name]['means'].append(float(np.nanmean(rep_means[cond][name])))
            results[cond][name]['stds'].append(float(np.nanstd(rep_means[cond][name])))
            for c, vals in rep_pc[cond][name].items():
                if c not in results[cond][name]['per_class']:
                    results[cond][name]['per_class'][c] = {'means': [], 'stds': []}
                results[cond][name]['per_class'][c]['means'].append(float(np.nanmean(vals)))
                results[cond][name]['per_class'][c]['stds'].append(float(np.nanstd(vals)))

    bin_elapsed = time.time() - t_bin_start
    sweep_elapsed = time.time() - t_sweep_start
    bins_done = bms_idx + 1
    eta_str = ""
    if bins_done < n_bins_sweep:
        remaining_weight = sum(1.0 / b for b in bin_sizes_ms[bins_done:])
        done_weight      = sum(1.0 / b for b in bin_sizes_ms[:bins_done])
        eta_sec = sweep_elapsed * remaining_weight / max(done_weight, 1e-9)
        eta_str = f"  | ETA remaining sweep ~{eta_sec/60:.1f} min"
    print(f"  >> bin {bms} ms done in {bin_elapsed:.1f}s  "
          f"(sweep so far {sweep_elapsed/60:.1f} min, {bins_done}/{n_bins_sweep} bins){eta_str}")
    sweep_stats.append({
        'bin_ms': bms,
        'n_task_train': int(X_t_kin.shape[0]),
        'n_passive_test': int(X_p_kin0.shape[0]),
        'n_merged': int(X_merged.shape[0]),
        'n_merged_neural': int(X_merged_neural.shape[0]),
        'n_dropped_task_no_tracking': n_dropped_t,
        'bin_elapsed_sec': float(bin_elapsed),
    })

print(f"\nTotal CV sweep wall time: {(time.time() - t_sweep_start)/60:.2f} min")

clf_names   = list(CLASSIFIERS.keys())
all_classes = sorted(results['passive_cv'][clf_names[0]]['per_class'].keys())


# ------------------------------------------------------------------ #
#  Best-decoder summary                                               #
# ------------------------------------------------------------------ #
def _best_in_condition(cond_results):
    """Return (classifier, bin_idx, mean, std) of best mean acc across (clf, bin)."""
    best = None
    for name, res in cond_results.items():
        for i, m in enumerate(res['means']):
            if best is None or m > best[2]:
                best = (name, i, float(m), float(res['stds'][i]))
    return best


best_per_condition = {}
print("\n" + "=" * 78)
print("BEST DECODER per condition  (argmax of mean overall accuracy across bins)")
print("=" * 78)
print(f"{'condition':<24}  {'classifier':<14}  {'bin_ms':>6}  {'mean':>6}  {'std':>6}")
print("-" * 78)
for cond in ALL_CONDITIONS:
    name, idx, m, s = _best_in_condition(results[cond])
    bms = bin_sizes_ms[idx]
    best_per_condition[cond] = {
        'classifier': name, 'bin_ms': bms,
        'mean_acc': m, 'std_acc': s,
    }
    print(f"{cond:<24}  {name:<14}  {bms:>6d}  {m:>6.3f}  {s:>6.3f}")

# Global ranking across (condition, classifier, bin).
all_combos = []
for cond in ALL_CONDITIONS:
    for name, res in results[cond].items():
        for i, m in enumerate(res['means']):
            all_combos.append((float(m), float(res['stds'][i]),
                               cond, name, bin_sizes_ms[i]))
all_combos.sort(key=lambda r: r[0], reverse=True)

print("\nTop 10 overall  (condition x classifier x bin)")
print("-" * 78)
print(f"{'rank':>4}  {'mean':>6}  {'std':>6}  {'condition':<24}  {'classifier':<14}  {'bin_ms':>6}")
for rank, (m, s, cond, name, bms) in enumerate(all_combos[:10], 1):
    print(f"{rank:>4}  {m:>6.3f}  {s:>6.3f}  {cond:<24}  {name:<14}  {bms:>6d}")

m, s, cond, name, bms = all_combos[0]
best_overall = {
    'condition': cond, 'classifier': name, 'bin_ms': bms,
    'mean_acc': m, 'std_acc': s,
}
print(f"\n>>> BEST OVERALL: {cond} / {name} @ {bms} ms  "
      f"= {m:.3f} +/- {s:.3f}")


# ------------------------------------------------------------------ #
#  Figure: n_classifiers rows x (1 + n_classes) cols                  #
# ------------------------------------------------------------------ #
def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())

CLASS_LABELS = {
    -1: f'stim-on  left={_fmt_class(class_neg)} (-1)',
     0: 'ITI (0)',
     1: f'stim-on  left={_fmt_class(class_pos)} (+1)',
}
COND_STYLE = {
    'passive_cv':     {'color': '#0072B2', 'marker': 's', 'ls': '-',
                       'label': 'passive CV (within)'},
    'merged_cv':      {'color': '#009E73', 'marker': '^', 'ls': '-',
                       'label': 'merged CV (task+kin + passive+kin=0)'},
    'merged_neural_cv':{'color': '#117733', 'marker': '<', 'ls': '-',
                       'label': 'merged CV (task neural + passive neural)'},
    'task_kin_cv':    {'color': '#E69F00', 'marker': 'v', 'ls': '-',
                       'label': 'task+kin CV (within)'},
    'task_neural_cv': {'color': '#56B4E9', 'marker': 'P', 'ls': '-',
                       'label': 'task neural CV (within)'},
    'merged_cv_task':          {'color': '#009E73', 'marker': '^', 'ls': ':',
                                'label': 'merged CV  test=task half'},
    'merged_cv_passive':       {'color': '#009E73', 'marker': '^', 'ls': '-.',
                                'label': 'merged CV  test=passive half'},
    'merged_neural_cv_task':   {'color': '#117733', 'marker': '<', 'ls': ':',
                                'label': 'merged-neural CV  test=task half'},
    'merged_neural_cv_passive':{'color': '#117733', 'marker': '<', 'ls': '-.',
                                'label': 'merged-neural CV  test=passive half'},
}

n_rows = len(clf_names)
n_cols = 1 + len(all_classes)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.2 * n_rows),
                         sharex=True)
if n_rows == 1:
    axes = axes[np.newaxis, :]


def _plot_three(ax, dataset_results, clf_name, y_key, title):
    for cond in ALL_CONDITIONS:
        res = dataset_results[cond][clf_name]
        if y_key == 'overall':
            means = np.array(res['means']); stds = np.array(res['stds'])
        else:
            means = np.array(res['per_class'][y_key]['means'])
            stds  = np.array(res['per_class'][y_key]['stds'])
        style = COND_STYLE[cond]
        ax.errorbar(bin_sizes_ms, means, yerr=stds,
                    marker=style['marker'], color=style['color'],
                    linestyle=style['ls'],
                    linewidth=1.6, capsize=3, label=style['label'])
    ax.set_title(title, fontsize=10)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms], fontsize=8)
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.set_ylim(0.0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)


for row, clf_name in enumerate(clf_names):
    _plot_three(axes[row, 0], results, clf_name, 'overall',
                f'{clf_name} - overall acc')
    axes[row, 0].set_ylabel(f'{clf_name}\nAccuracy', fontsize=10)
    if row == 0:
        axes[row, 0].legend(fontsize=8, loc='lower right')
    for col_i, c in enumerate(all_classes, start=1):
        _plot_three(axes[row, col_i], results, clf_name, c,
                    f'{clf_name} - recall  {CLASS_LABELS.get(c, c)}')

for ax in axes[-1, :]:
    ax.set_xlabel('Bin size (ms)', fontsize=10)

def _stem_part(d):
    return '_'.join(f'{k}{v:g}' for k, v in d.items())

stem = (f"train_task_test_passive_{session}_pos-{_stem_part(class_pos)}_"
        f"neg-{_stem_part(class_neg)}_{n_splits}fold")

fig.suptitle(
    f'Within-context decoding (task / passive / merged)  |  {session}  |  '
    f'+1: {_fmt_class(class_pos)}  vs  -1: {_fmt_class(class_neg)}  |  '
    f'common units={n_units_common}  |  '
    f'n_repeats={n_repeats}, balanced training / natural-ratio test  |  '
    f'kin = [speed, sin/cos heading, sin/cos head_angle]\n'
    f'BEST: {best_overall["condition"]} / {best_overall["classifier"]} '
    f'@ {best_overall["bin_ms"]} ms = '
    f'{best_overall["mean_acc"]:.3f} +/- {best_overall["std_acc"]:.3f}',
    fontsize=11, y=1.00,
)
plt.tight_layout(rect=[0, 0, 1, 0.985])

fig_path = out_dir / f"{stem}.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved -> {fig_path}")

pkl_out = out_dir / f"{stem}.pkl"
with open(pkl_out, 'wb') as f:
    pickle.dump({
        'bin_sizes_ms':         bin_sizes_ms,
        'results':              results,
        'null_results':         null_results,
        'n_shuffles_null':      N_SHUFFLES_NULL,
        'conditions':           ALL_CONDITIONS,
        'conditions_core':      CONDITIONS,
        'conditions_extra':     EXTRA_CONDITIONS,
        'n_splits':             n_splits,
        'n_repeats':            n_repeats,
        'class_pos':            class_pos,
        'class_neg':            class_neg,
        'task_col_map':         TASK_COL_MAP,
        'passive_col_map':      PASSIVE_COL_MAP,
        'rewarded_combination': rewarded_combination,
        'n_units_common':       n_units_common,
        'sweep_stats':          sweep_stats,
        'feature_columns_added': KINEMATIC_FEATURE_COLUMNS,
        'kinematics_zero_fill_for_passive': True,
        'clean_position_stats': clean_stats,
        'session_to_window_offset_sec': offset,
        'best_per_condition':   best_per_condition,
        'best_overall':         best_overall,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved -> {pkl_out}")


# ------------------------------------------------------------------ #
#  Bar-plot figure: rows = classifiers, cols = bin sizes              #
#  (same data as the line plot, but easier to compare conditions)     #
# ------------------------------------------------------------------ #
def _bar_panel(ax, dataset_results, dataset_null, clf_name, bin_idx, conds, show_xticks):
    means = np.array([dataset_results[c][clf_name]['means'][bin_idx] for c in conds])
    stds  = np.array([dataset_results[c][clf_name]['stds'][bin_idx]  for c in conds])
    null_m = np.array([dataset_null[c][clf_name]['means'][bin_idx] for c in conds])
    null_s = np.array([dataset_null[c][clf_name]['stds' ][bin_idx] for c in conds])
    colors = [COND_STYLE[c]['color'] for c in conds]
    # Hatch the EXTRA conditions (merged-split halves) so they're visually distinct
    # from the parent merged_cv / merged_neural_cv bars that share their color.
    hatches = ['//' if c in EXTRA_CONDITIONS else '' for c in conds]
    xs = np.arange(len(conds))
    bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=2,
                  edgecolor='black', linewidth=0.5)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    # Per-bar permutation-null reference: dotted line at the null mean,
    # gray band for +/- 1 std across shuffles.
    half_w = 0.4
    for x, nm, ns in zip(xs, null_m, null_s):
        ax.fill_between([x - half_w, x + half_w], nm - ns, nm + ns,
                        color='gray', alpha=0.20, linewidth=0)
        ax.hlines(nm, x - half_w, x + half_w,
                  colors='dimgray', linestyles=':', linewidth=1.0)
    ax.set_ylim(0.0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(xs)
    if show_xticks:
        ax.set_xticklabels(conds, rotation=60, ha='right', fontsize=7)
    else:
        ax.set_xticklabels([])


n_rows_bar = len(clf_names)
n_cols_bar = len(bin_sizes_ms)
fig2, axes2 = plt.subplots(
    n_rows_bar, n_cols_bar,
    figsize=(max(3.2 * n_cols_bar, 6.0), 2.6 * n_rows_bar),
    sharey=True,
)
if n_rows_bar == 1:
    axes2 = np.array([axes2])
if n_cols_bar == 1:
    axes2 = axes2.reshape(n_rows_bar, 1)

for row, clf_name in enumerate(clf_names):
    axes2[row, 0].set_ylabel(f'{clf_name}\nAccuracy', fontsize=10)
    for col, bms in enumerate(bin_sizes_ms):
        _bar_panel(
            axes2[row, col], results, null_results, clf_name, col, ALL_CONDITIONS,
            show_xticks=(row == n_rows_bar - 1),
        )
        if row == 0:
            axes2[row, col].set_title(f'{bms} ms', fontsize=10)

fig2.suptitle(
    f'Within-context decoding (overall accuracy, bar view)  |  {session}  |  '
    f'+1: {_fmt_class(class_pos)}  vs  -1: {_fmt_class(class_neg)}  |  '
    f'common units={n_units_common}  |  n_repeats={n_repeats}\n'
    f'hatched bars = merged-CV halves (test=task or test=passive); '
    f'dotted line + gray band over each bar = permutation-null mean +/- 1 std '
    f'({N_SHUFFLES_NULL} shuffles).  '
    f'BEST: {best_overall["condition"]} / {best_overall["classifier"]} '
    f'@ {best_overall["bin_ms"]} ms = '
    f'{best_overall["mean_acc"]:.3f} +/- {best_overall["std_acc"]:.3f}',
    fontsize=11, y=1.00,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])

fig_path2 = out_dir / f"{stem}_bars.png"
fig2.savefig(fig_path2, dpi=150, bbox_inches='tight')
print(f"Bar figure saved -> {fig_path2}")

plt.show()
