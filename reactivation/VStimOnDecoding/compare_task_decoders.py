"""
Compare multiple decoders for task data across bin sizes (30–500 ms).
Data source : prepare_task_stimtype.py  (task_spikes_*.pkl)
Decoding the LEFT-side grating identity using the joint (orientation, SF) label:
  +1  stim-on, left grating matches class_pos
  -1  stim-on, left grating matches class_neg
   0  ITI
Classifiers : Gaussian NB, Poisson NB, LDA, Logistic Regression, SVM, Random Forest, AODE

Figure layout
-------------
Top    : overall accuracy vs bin size
Bottom : per-class recall (+1 | -1 | 0), one panel each
"""

import sys
import pickle
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

from decode_utils import make_classifiers, run_cv_balanced_train, run_cv_balanced_train_shuffle, plot_panel
from prepare_task_stimtype import prepare_task_stim_type, infer_rewarded_combination

from params import (
    task_pkl as pkl_file,
    bin_sizes_ms, n_splits, random_state,
    class_pos, class_neg, TASK_COL_MAP,
)

out_dir = Path(pkl_file).parent / 'reactivation'
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {out_dir}")

# Infer the rewarded (orientation, SF) combination from trial_params so it can be
# saved as a side channel — the rewarded grating may appear on either side.
with open(pkl_file, 'rb') as _f:
    _trial_params = pickle.load(_f)['trial_params']
rewarded_combination = infer_rewarded_combination(_trial_params)
print(f"Class +1 (left grating matches): {class_pos}")
print(f"Class -1 (left grating matches): {class_neg}")
print(f"Rewarded grating in this session: {rewarded_combination}")

CLASSIFIERS = make_classifiers(random_state)
results = {name: {'means': [], 'stds': [], 'per_class': {}} for name in CLASSIFIERS}
chance_vals = []

# ------------------------------------------------------------------ #
#  Sweep over bin sizes                                               #
# ------------------------------------------------------------------ #
for bms in bin_sizes_ms:
    print(f"\n{'=' * 52}")
    print(f"  Bin size: {bms} ms")
    print(f"{'=' * 52}")

    X, y, _, _ = prepare_task_stim_type(
        pkl_file, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bms / 1000.0, random_state=random_state,
    )

    print(f"  X shape: {X.shape}  "
          f"+1={np.sum(y==1)}  -1={np.sum(y==-1)}  0={np.sum(y==0)}")
    chance_recorded = False

    for name, clf_proto in CLASSIFIERS.items():
        print(f"  [{name}]", end='', flush=True)
        folds, mean, chance, pc_means, pc_stds = run_cv_balanced_train(
            name, clf_proto, X, y,
            n_splits=n_splits, random_state=random_state,
        )

        results[name]['means'].append(mean)
        results[name]['stds'].append(float(np.std(folds)))

        for c, m in pc_means.items():
            if c not in results[name]['per_class']:
                results[name]['per_class'][c] = {'means': [], 'stds': []}
            results[name]['per_class'][c]['means'].append(m)
            results[name]['per_class'][c]['stds'].append(pc_stds[c])

        if not chance_recorded:
            chance_vals.append(chance)
            chance_recorded = True
        print(f"  mean={mean:.3f}")

mean_chance = float(np.mean(chance_vals))
all_classes = sorted(results[next(iter(results))]['per_class'].keys())

# ------------------------------------------------------------------ #
#  Plot                                                               #
# ------------------------------------------------------------------ #
def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())

CLASS_LABELS = {
    -1: f'Stim on — left = {_fmt_class(class_neg)}  (-1)',
     0: 'ITI (0)',
     1: f'Stim on — left = {_fmt_class(class_pos)}  (+1)',
}

n_class_panels = len(all_classes)
fig = plt.figure(figsize=(6 * (n_class_panels + 1), 9))
ax_top   = fig.add_subplot(2, 1, 1)
axes_bot = [fig.add_subplot(2, n_class_panels, n_class_panels + i + 1)
            for i in range(n_class_panels)]

plot_panel(ax_top, results, bin_sizes_ms, 'overall',
           'Overall accuracy — task decoding (+1 / -1 / 0)', 0.0)
ax_top.legend(fontsize=9, loc='lower right', ncol=2)

for ax, c in zip(axes_bot, all_classes):
    plot_panel(ax, results, bin_sizes_ms, c,
               CLASS_LABELS.get(c, f'Class {c}'), 0.0)
    ax.legend(fontsize=8, loc='lower right', ncol=1)

plt.tight_layout()

info = (
    f"Data: {Path(pkl_file).name}  |  "
    f"Bins: {bin_sizes_ms[0]}–{bin_sizes_ms[-1]} ms  |  "
    f"CV: {n_splits}-fold  |  "
    f"Train: balanced (undersampled), Test: natural ratio  |  "
    f"Chance: {mean_chance:.2f}  |  "
    f"Seed: {random_state}"
)
fig.text(0.5, -0.01, info, ha='center', va='top', fontsize=7.5, color='gray')

session = Path(pkl_file).parent.name
stem    = f"task_{session}_balanced_train_natural_test_{n_splits}fold"

fig_path = out_dir / f"{stem}.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {fig_path}")

pkl_out = out_dir / f"{stem}.pkl"
with open(pkl_out, 'wb') as f:
    pickle.dump({
        'bin_sizes_ms':         bin_sizes_ms,
        'results':              results,
        'chance':               mean_chance,
        'n_splits':             n_splits,
        'class_pos':            class_pos,
        'class_neg':            class_neg,
        'col_map':              TASK_COL_MAP,
        'rewarded_combination': rewarded_combination,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved → {pkl_out}")

# ------------------------------------------------------------------ #
#  Pick best bin size (argmax mean accuracy across classifiers)       #
# ------------------------------------------------------------------ #
clf_names = list(CLASSIFIERS.keys())
mean_acc_per_bin = np.mean(
    [[results[n]['means'][i] for n in clf_names] for i in range(len(bin_sizes_ms))],
    axis=1,
)
bar_idx    = int(np.argmax(mean_acc_per_bin))
bar_bin_ms = bin_sizes_ms[bar_idx]
print(f"\nBest bin size (argmax): {bar_bin_ms} ms  (mean acc = {mean_acc_per_bin[bar_idx]:.3f})")

X_bar, y_bar, _, _ = prepare_task_stim_type(
    pkl_file, class_pos, class_neg, TASK_COL_MAP,
    bin_size_sec=bar_bin_ms / 1000.0, random_state=random_state,
)

# ------------------------------------------------------------------ #
#  Shuffle null distribution at best bin size                         #
# ------------------------------------------------------------------ #
print(f"Computing shuffle null distribution at {bar_bin_ms} ms ...")
shuffle_results = {}
for name, clf_proto in CLASSIFIERS.items():
    print(f"  [shuffle {name}]", end='', flush=True)
    sm, ss, spc_m, spc_s = run_cv_balanced_train_shuffle(
        name, clf_proto, X_bar, y_bar, n_splits=n_splits, random_state=random_state
    )
    shuffle_results[name] = {'mean': sm, 'std': ss, 'per_class': {
        c: {'mean': spc_m[c], 'std': spc_s[c]} for c in spc_m
    }}
    print(f"  shuf_mean={sm:.3f}")

# ------------------------------------------------------------------ #
#  Bar plot: per-classifier at best bin size                          #
# ------------------------------------------------------------------ #
n_clf     = len(clf_names)
n_panels  = len(all_classes)
x         = np.arange(n_clf)
width     = 0.55
color_bar = '#4C72B0'

fig2    = plt.figure(figsize=(max(8, n_clf * 1.6) * max(n_panels, 1), 10))
ax2_top   = fig2.add_subplot(2, 1, 1)
axes2_bot = [fig2.add_subplot(2, n_panels, n_panels + i + 1) for i in range(n_panels)]


bar_w  = width * 0.45
offset = width * 0.25
color_shuf = '#AAAAAA'


def _bar_single(ax, means, stds, shuf_means, shuf_stds, title, ylabel):
    bars = ax.bar(x - offset, means, bar_w, color=color_bar,
                  yerr=stds, capsize=3, error_kw=dict(linewidth=1), label='Actual')
    ax.bar(x + offset, shuf_means, bar_w, color=color_shuf,
           yerr=shuf_stds, capsize=3, error_kw=dict(linewidth=1), label='Shuffle')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.2f}', ha='center', va='bottom', fontsize=8, color=color_bar)
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper right')


_bar_single(
    ax2_top,
    [results[n]['means'][bar_idx]        for n in clf_names],
    [results[n]['stds'][bar_idx]         for n in clf_names],
    [shuffle_results[n]['mean']          for n in clf_names],
    [shuffle_results[n]['std']           for n in clf_names],
    f'Overall accuracy  (bin = {bar_bin_ms} ms)',
    'Accuracy',
)

for ax, c in zip(axes2_bot, all_classes):
    _bar_single(
        ax,
        [results[n]['per_class'][c]['means'][bar_idx]  for n in clf_names],
        [results[n]['per_class'][c]['stds'][bar_idx]   for n in clf_names],
        [shuffle_results[n]['per_class'][c]['mean']    for n in clf_names],
        [shuffle_results[n]['per_class'][c]['std']     for n in clf_names],
        f'{CLASS_LABELS.get(c, f"Class {c}")}  (bin = {bar_bin_ms} ms)',
        'Recall',
    )

fig2.suptitle(
    f'Per-classifier decoding  |  bin = {bar_bin_ms} ms, {n_splits}-fold CV\n'
    f'Balanced train / natural-ratio test',
    fontsize=11, y=1.01,
)
plt.tight_layout()

fig2_path = out_dir / f"{stem}_bar{bar_bin_ms}ms.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Bar-plot figure saved → {fig2_path}")

plt.show()
