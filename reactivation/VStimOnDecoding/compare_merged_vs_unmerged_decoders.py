"""
Compare classifiers trained on TASK-only, PASSIVE-only, and MERGED (task+passive)
bin data across bin sizes (30–500 ms).

Both contexts use the SAME joint (orientation, SF) label scheme defined in
params.py — class_pos and class_neg are dicts of canonical keys, translated
to task vs passive column names via TASK_COL_MAP / PASSIVE_COL_MAP:
    +1   stim-on  &  left grating matches every key of class_pos
    -1   stim-on  &  left grating matches every key of class_neg
     0   ITI

For each bin size and each of the 7 classifiers, run-cv-balanced-train is
executed three times (task / passive / merged) and accuracies + per-class
recalls are stored.

Figure layout
-------------
7 rows (one per classifier) × 4 columns (overall, recall +1, recall 0, recall -1).
Each panel: three lines (task / passive / merged) vs bin size.
"""

import sys
import pickle
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

from decode_utils import make_classifiers, run_cv_balanced_train
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

with open(task_pkl, 'rb') as _f:
    rewarded_combination = infer_rewarded_combination(pickle.load(_f)['trial_params'])
print(f"Class +1 (left grating matches): {class_pos}")
print(f"Class -1 (left grating matches): {class_neg}")
print(f"Rewarded grating in this session: {rewarded_combination}")

DATASETS = ['task', 'passive', 'merged']
CLASSIFIERS = make_classifiers(random_state)

# results[dataset][clf_name] = {'means': [...], 'stds': [...], 'per_class': {c: {'means':[], 'stds':[]}}}
results = {
    ds: {name: {'means': [], 'stds': [], 'per_class': {}} for name in CLASSIFIERS}
    for ds in DATASETS
}
n_units_common = None

# ------------------------------------------------------------------ #
#  Sweep over bin sizes                                               #
# ------------------------------------------------------------------ #
for bms in bin_sizes_ms:
    print(f"\n{'=' * 60}")
    print(f"  Bin size: {bms} ms")
    print(f"{'=' * 60}")

    # ---- Task data ----
    print("[TASK]")
    X_t, y_t, _, units_t = prepare_task_stim_type(
        task_pkl, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bms / 1000.0,
        balance_classes=False,
        random_state=random_state,
    )

    # ---- Passive data ----
    print("[PASSIVE]")
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bms / 1000.0,
        balance_classes=False,
        random_state=random_state,
    )

    # ---- Align units (intersection, sorted) ----
    common_units = sorted(set(units_t) & set(units_p))
    if len(common_units) == 0:
        raise RuntimeError(
            "No common units between task and passive recordings — "
            "cannot merge feature matrices."
        )
    idx_t = [units_t.index(u) for u in common_units]
    idx_p = [units_p.index(u) for u in common_units]
    X_t_a = X_t[:, idx_t]
    X_p_a = X_p[:, idx_p]
    if n_units_common is None:
        n_units_common = len(common_units)
        print(f"  Common units: {n_units_common}  "
              f"(task={len(units_t)}, passive={len(units_p)})")

    # ---- Build merged ----
    X_m = np.vstack([X_t_a, X_p_a])
    y_m = np.concatenate([y_t, y_p])

    datasets_xy = {
        'task':    (X_t_a, y_t),
        'passive': (X_p_a, y_p),
        'merged':  (X_m,   y_m),
    }

    for ds_name, (X, y) in datasets_xy.items():
        print(f"  [{ds_name}] X={X.shape}  "
              f"+1={np.sum(y==1)}  -1={np.sum(y==-1)}  0={np.sum(y==0)}")
        for name, clf_proto in CLASSIFIERS.items():
            print(f"    [{name}]", end='', flush=True)
            # Average over n_repeats CV passes with distinct seeds (paired
            # within a repeat across datasets — task/passive/merged use the
            # SAME cv_seed). Reported mean = mean over repeats; std =
            # std over repeats (not over folds).
            rep_means = []
            rep_pc    = {}
            for r in range(n_repeats):
                cv_seed = random_state + 1000 * bms + r
                _, mean_r, _, pcm_r, _ = run_cv_balanced_train(
                    name, clf_proto, X, y,
                    n_splits=n_splits, random_state=cv_seed,
                )
                rep_means.append(mean_r)
                for c, m in pcm_r.items():
                    rep_pc.setdefault(c, []).append(m)
            results[ds_name][name]['means'].append(float(np.mean(rep_means)))
            results[ds_name][name]['stds'].append(float(np.std(rep_means)))
            for c, vals in rep_pc.items():
                if c not in results[ds_name][name]['per_class']:
                    results[ds_name][name]['per_class'][c] = {'means': [], 'stds': []}
                results[ds_name][name]['per_class'][c]['means'].append(float(np.mean(vals)))
                results[ds_name][name]['per_class'][c]['stds'].append(float(np.std(vals)))
            print(f"  mean={np.mean(rep_means):.3f} (n_repeats={n_repeats})")

# Collect class labels actually observed (use task's record as canonical;
# all three datasets share {-1, 0, +1}).
clf_names   = list(CLASSIFIERS.keys())
all_classes = sorted(results['task'][clf_names[0]]['per_class'].keys())

# ------------------------------------------------------------------ #
#  Figure: 7 rows (classifiers) × 4 cols (overall, +1, 0, -1)          #
# ------------------------------------------------------------------ #
def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())

CLASS_LABELS = {
    -1: f'stim-on  left={_fmt_class(class_neg)} (-1)',
     0: 'ITI (0)',
     1: f'stim-on  left={_fmt_class(class_pos)} (+1)',
}
DATASET_STYLE = {
    'task':    {'color': '#D55E00', 'marker': 'o', 'label': 'Task only'},
    'passive': {'color': '#0072B2', 'marker': 's', 'label': 'Passive only'},
    'merged':  {'color': '#009E73', 'marker': '^', 'label': 'Merged (task+passive)'},
}

n_rows = len(clf_names)
n_cols = 1 + len(all_classes)  # overall + one per class
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.2 * n_rows),
                         sharex=True)
if n_rows == 1:
    axes = axes[np.newaxis, :]


def _plot_three(ax, dataset_results, clf_name, y_key, title):
    for ds in DATASETS:
        res = dataset_results[ds][clf_name]
        if y_key == 'overall':
            means = np.array(res['means'])
            stds  = np.array(res['stds'])
        else:
            means = np.array(res['per_class'][y_key]['means'])
            stds  = np.array(res['per_class'][y_key]['stds'])
        style = DATASET_STYLE[ds]
        ax.errorbar(bin_sizes_ms, means, yerr=stds,
                    marker=style['marker'], color=style['color'],
                    linewidth=1.6, capsize=3, label=style['label'])
    ax.set_title(title, fontsize=10)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms], fontsize=8)
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.set_ylim(0.0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)


for row, clf_name in enumerate(clf_names):
    # column 0: overall accuracy
    _plot_three(axes[row, 0], results, clf_name, 'overall',
                f'{clf_name} — overall acc')
    axes[row, 0].set_ylabel(f'{clf_name}\nAccuracy', fontsize=10)
    if row == 0:
        axes[row, 0].legend(fontsize=8, loc='lower right')

    # columns 1..: per-class recall
    for col_i, c in enumerate(all_classes, start=1):
        _plot_three(axes[row, col_i], results, clf_name, c,
                    f'{clf_name} — recall  {CLASS_LABELS.get(c, c)}')
        if col_i == 1 and row == 0:
            axes[row, col_i].legend(fontsize=7, loc='lower right')

for ax in axes[-1, :]:
    ax.set_xlabel('Bin size (ms)', fontsize=10)

session = Path(task_pkl).parent.name


def _stem_part(d):
    return '_'.join(f'{k}{v:g}' for k, v in d.items())


stem = (f"merged_vs_unmerged_{session}_pos-{_stem_part(class_pos)}_"
        f"neg-{_stem_part(class_neg)}_{n_splits}fold")

fig.suptitle(
    f'Merged vs unmerged decoders  |  {session}  |  '
    f'+1: {_fmt_class(class_pos)}  vs  -1: {_fmt_class(class_neg)}  |  '
    f'common units={n_units_common}  |  {n_splits}-fold CV  |  '
    f'balanced train / natural-ratio test',
    fontsize=12, y=1.00,
)
plt.tight_layout(rect=[0, 0, 1, 0.985])

fig_path = out_dir / f"{stem}.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {fig_path}")

pkl_out = out_dir / f"{stem}.pkl"
with open(pkl_out, 'wb') as f:
    pickle.dump({
        'bin_sizes_ms':         bin_sizes_ms,
        'results':              results,
        'n_splits':             n_splits,
        'n_repeats':            n_repeats,
        'class_pos':            class_pos,
        'class_neg':            class_neg,
        'task_col_map':         TASK_COL_MAP,
        'passive_col_map':      PASSIVE_COL_MAP,
        'n_units_common':       n_units_common,
        'rewarded_combination': rewarded_combination,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved → {pkl_out}")

plt.show()
