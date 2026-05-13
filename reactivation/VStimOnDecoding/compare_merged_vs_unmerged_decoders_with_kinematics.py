"""
Kinematics-augmented merged-vs-unmerged decoder comparison.

Mirrors compare_merged_vs_unmerged_decoders.py (TASK / PASSIVE / MERGED, 7
classifiers across bin sizes 30–500 ms) but appends five behavioral columns
to the input features:

  features added = [
      speed_cms,
      sin(heading),    cos(heading),       # circular, encoded as (sin, cos)
      sin(head_angle), cos(head_angle),    # circular, encoded as (sin, cos)
  ]

For TASK (freely-moving in 2-D arena) the columns come from session_position
DLC, computed per task bin as the mean of clean DLC samples falling inside.
For PASSIVE (head-fixed) the animal cannot translate or turn, so the kinematic
columns are zero-filled — exactly what the model "would see" for a stationary
animal.

Goal: does the merged decoder (task + passive) outperform either context alone
when both have access to kinematic context?

Caveat: zero-filled passive kinematics make those five columns a perfect
TASK-vs-PASSIVE indicator inside the merged feature matrix. A linear or
tree-based classifier can use them to identify *context* rather than *stimulus*
identity. The stimulus labels (+1 / -1 / 0) are shared across contexts, so
context-driven splits don't directly inflate stim recall — but compare to
compare_merged_vs_unmerged_decoders.py to see whether any apparent gain is
genuine.
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

from decode_utils import make_classifiers, run_cv_balanced_train
from kinematics_utils import (
    KINEMATIC_FEATURE_COLUMNS,
    N_KINEMATIC_FEATURES,
    build_task_kinematics,
    load_task_kinematic_samples,
    print_kinematic_sample_report,
)
from prepare_task_stimtype import prepare_task_stim_type
from prepare_passive_stimtype import prepare_passive_stim_type
from prepare_task_stimtype import infer_rewarded_combination

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
#  Step 1 — load TASK DLC, build kinematic sample arrays              #
# ------------------------------------------------------------------ #
kin_samples, clean_stats, offset = load_task_kinematic_samples(task_pkl)
print_kinematic_sample_report(kin_samples, clean_stats, offset)


# ------------------------------------------------------------------ #
#  Step 3 — sweep bin sizes: task / passive / merged on augmented X   #
# ------------------------------------------------------------------ #
DATASETS = ['task', 'passive', 'merged']
CLASSIFIERS = make_classifiers(random_state)

results = {
    ds: {name: {'means': [], 'stds': [], 'per_class': {}} for name in CLASSIFIERS}
    for ds in DATASETS
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

    # ---- Passive (head-fixed: kinematics = 0) ----
    print("[PASSIVE]")
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False,
        random_state=random_state,
    )
    kin_p = np.zeros((X_p.shape[0], N_KIN), dtype=np.float64)

    # ---- Align units across contexts (intersection, sorted) ----
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

    # ---- Augment ----
    X_t_aug = np.hstack([X_t_a, kin_t.astype(X_t_a.dtype)])
    X_p_aug = np.hstack([X_p_a, kin_p.astype(X_p_a.dtype)])
    X_m_aug = np.vstack([X_t_aug, X_p_aug])
    y_m     = np.concatenate([y_t, y_p])

    datasets_xy = {
        'task':    (X_t_aug, y_t),
        'passive': (X_p_aug, y_p),
        'merged':  (X_m_aug, y_m),
    }
    for ds_name, (X, y) in datasets_xy.items():
        print(f"  [{ds_name}] X={X.shape}  "
              f"+1={np.sum(y==1)}  -1={np.sum(y==-1)}  0={np.sum(y==0)}")

    # Average over n_repeats CV passes per (dataset, classifier). Same cv_seed
    # is used across datasets within a repeat so paired comparisons stay valid.
    pbar = tqdm(total=len(datasets_xy) * len(CLASSIFIERS) * n_repeats,
                desc=f"  bin {bms} ms CV", ncols=90, leave=True)
    rep_means = {ds: {n: [] for n in CLASSIFIERS} for ds in datasets_xy}
    rep_pc    = {ds: {n: {} for n in CLASSIFIERS} for ds in datasets_xy}
    for r in range(n_repeats):
        cv_seed = random_state + 1000 * bms + r
        for ds_name, (X, y) in datasets_xy.items():
            for name, clf_proto in CLASSIFIERS.items():
                _, mean_r, _, pcm_r, _ = run_cv_balanced_train(
                    name, clf_proto, X, y,
                    n_splits=n_splits, random_state=cv_seed,
                )
                rep_means[ds_name][name].append(mean_r)
                for c, m in pcm_r.items():
                    rep_pc[ds_name][name].setdefault(c, []).append(m)
                pbar.set_postfix_str(f"r={r+1}/{n_repeats} {ds_name} {name}")
                pbar.update(1)
    pbar.close()
    for ds_name in datasets_xy:
        for name in CLASSIFIERS:
            results[ds_name][name]['means'].append(float(np.mean(rep_means[ds_name][name])))
            results[ds_name][name]['stds'].append(float(np.std(rep_means[ds_name][name])))
            for c, vals in rep_pc[ds_name][name].items():
                if c not in results[ds_name][name]['per_class']:
                    results[ds_name][name]['per_class'][c] = {'means': [], 'stds': []}
                results[ds_name][name]['per_class'][c]['means'].append(float(np.mean(vals)))
                results[ds_name][name]['per_class'][c]['stds'].append(float(np.std(vals)))

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
        'n_task': int(X_t_aug.shape[0]),
        'n_passive': int(X_p_aug.shape[0]),
        'n_merged': int(X_m_aug.shape[0]),
        'n_dropped_task_no_tracking': n_dropped_t,
        'bin_elapsed_sec': float(bin_elapsed),
    })

print(f"\nTotal CV sweep wall time: {(time.time() - t_sweep_start)/60:.2f} min")

clf_names   = list(CLASSIFIERS.keys())
all_classes = sorted(results['task'][clf_names[0]]['per_class'].keys())


# ------------------------------------------------------------------ #
#  Step 4 — figure: 7 rows × (1 + n_classes) cols                     #
# ------------------------------------------------------------------ #
def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())

CLASS_LABELS = {
    -1: f'stim-on  left={_fmt_class(class_neg)} (-1)',
     0: 'ITI (0)',
     1: f'stim-on  left={_fmt_class(class_pos)} (+1)',
}
DATASET_STYLE = {
    'task':    {'color': '#D55E00', 'marker': 'o', 'label': 'Task only + kin'},
    'passive': {'color': '#0072B2', 'marker': 's', 'label': 'Passive only + kin=0'},
    'merged':  {'color': '#009E73', 'marker': '^', 'label': 'Merged + kin'},
}

n_rows = len(clf_names)
n_cols = 1 + len(all_classes)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.2 * n_rows),
                         sharex=True)
if n_rows == 1:
    axes = axes[np.newaxis, :]


def _plot_three(ax, dataset_results, clf_name, y_key, title):
    for ds in DATASETS:
        res = dataset_results[ds][clf_name]
        if y_key == 'overall':
            means = np.array(res['means']); stds = np.array(res['stds'])
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
    _plot_three(axes[row, 0], results, clf_name, 'overall',
                f'{clf_name} — overall acc')
    axes[row, 0].set_ylabel(f'{clf_name}\nAccuracy', fontsize=10)
    if row == 0:
        axes[row, 0].legend(fontsize=8, loc='lower right')
    for col_i, c in enumerate(all_classes, start=1):
        _plot_three(axes[row, col_i], results, clf_name, c,
                    f'{clf_name} — recall  {CLASS_LABELS.get(c, c)}')
        if col_i == 1 and row == 0:
            axes[row, col_i].legend(fontsize=7, loc='lower right')

for ax in axes[-1, :]:
    ax.set_xlabel('Bin size (ms)', fontsize=10)

def _stem_part(d):
    return '_'.join(f'{k}{v:g}' for k, v in d.items())

stem = (f"merged_vs_unmerged_with_kinematics_{session}_pos-{_stem_part(class_pos)}_"
        f"neg-{_stem_part(class_neg)}_{n_splits}fold")

fig.suptitle(
    f'Merged vs unmerged decoders + kinematics  |  {session}  |  '
    f'+1: {_fmt_class(class_pos)}  vs  -1: {_fmt_class(class_neg)}  |  '
    f'common units={n_units_common}  |  {n_splits}-fold CV  |  '
    f'balanced train / natural-ratio test  |  '
    f'kin = [speed, sin/cos heading, sin/cos head_angle], zero-filled for passive',
    fontsize=11, y=1.00,
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
        'rewarded_combination': rewarded_combination,
        'n_units_common':       n_units_common,
        'sweep_stats':          sweep_stats,
        'feature_columns_added': KINEMATIC_FEATURE_COLUMNS,
        'kinematics_zero_fill_for_passive': True,
        'clean_position_stats': clean_stats,
        'session_to_window_offset_sec': offset,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved → {pkl_out}")

plt.show()
