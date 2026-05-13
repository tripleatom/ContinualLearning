"""
Heading-as-feature twin of compare_task_reward_decoders_with_velocity.py.

Adds four circular-feature columns to the classifier input and compares
against the firing-rate-only baseline:

  features added = [sin(heading), cos(heading), sin(head_angle), cos(head_angle)]

Why sin/cos?  Heading is on a circle (0–360°) and head_angle is on a circle
(±π rad). Feeding the raw degree value to a linear or tree-based classifier
creates a false discontinuity at the 0°/360° wrap — two samples that are 1°
apart on the circle would be 359 apart numerically. (sin, cos) is the
standard "property engineering" trick: it embeds the circle into ℝ² so that
nearby angles are also nearby in feature space.

Per-bin aggregation: mean(sin) and mean(cos) across clean DLC samples inside
the bin. This is the magnitude-and-direction of the circular mean and works
without unwrapping.

For each bin size:
  baseline  : X = firing rates                                   (n_bins, n_units)
  augmented : X = [firing rates | sin_h | cos_h | sin_ha | cos_ha] (n_bins, n_units + 4)
Same (y, bin selection, CV seed) is used for both arms within a repeat, so
Δ = augmented − baseline is a paired statistic per classifier × bin × repeat.

Outputs (under <pkl_folder>/reactivation/):
  task_<session>_heading_feature_vs_baseline_<n>fold.png
  task_<session>_heading_feature_delta_<n>fold.png
  task_<session>_heading_feature_bar<bms>ms.png
  task_<session>_heading_feature_<n>fold.pkl
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

from decode_utils import make_classifiers, run_cv_balanced_train, run_cv_balanced_train_shuffle
from kinematics_utils import (
    HEADING_ONLY_COLUMNS,
    build_task_kinematics,
    load_task_kinematic_samples,
    print_kinematic_sample_report,
)
from prepare_task_stimtype import prepare_task_stim_type, infer_rewarded_combination

from params import (
    task_pkl as pkl_file,
    bin_sizes_ms, n_splits, random_state,
    n_repeats, n_bootstrap,
    class_pos, class_neg, TASK_COL_MAP,
)

out_dir = Path(pkl_file).parent / 'reactivation'
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {out_dir}")

session = Path(pkl_file).parent.name


# ------------------------------------------------------------------ #
#  Step 1 — load DLC, build circular sample arrays (window frame)     #
# ------------------------------------------------------------------ #
# Interpolation in sin/cos space (done inside load_task_kinematic_samples)
# avoids wraparound: a 359°→1° step is continuous in (sin, cos) but a 358°
# jump in raw degrees. We use only the four sin/cos columns here.
kin_samples, clean_stats, offset = load_task_kinematic_samples(pkl_file)
print_kinematic_sample_report(kin_samples, clean_stats, offset)

with open(pkl_file, 'rb') as _f:
    _trial_params = pickle.load(_f)['trial_params']


# ------------------------------------------------------------------ #
#  Class definitions and rewarded-combination side channel            #
# ------------------------------------------------------------------ #
rewarded_combination = infer_rewarded_combination(_trial_params)
print(f"Class +1 (left grating matches): {class_pos}")
print(f"Class -1 (left grating matches): {class_neg}")
print(f"Rewarded grating in this session: {rewarded_combination}")


def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())


CLASS_LABELS = {
    -1: f'Stim on — left = {_fmt_class(class_neg)}  (-1)',
     0: 'ITI (0)',
     1: f'Stim on — left = {_fmt_class(class_pos)}  (+1)',
}


# ------------------------------------------------------------------ #
#  Step 2 — sweep bin sizes: baseline (X) vs augmented ([X | circ])   #
# ------------------------------------------------------------------ #
CLASSIFIERS = make_classifiers(random_state)
clf_names_seed = list(CLASSIFIERS.keys())
n_bins_sweep   = len(bin_sizes_ms)

per_repeat_baseline = {n: [[] for _ in range(n_bins_sweep)] for n in clf_names_seed}
per_repeat_augmented = {n: [[] for _ in range(n_bins_sweep)] for n in clf_names_seed}
per_repeat_pc_baseline  = {n: {} for n in clf_names_seed}
per_repeat_pc_augmented = {n: {} for n in clf_names_seed}

results_baseline  = {n: {'means': [], 'stds': [], 'per_class': {}} for n in clf_names_seed}
results_augmented = {n: {'means': [], 'stds': [], 'per_class': {}} for n in clf_names_seed}
chance_baseline, chance_augmented = [], []
sweep_stats = []

t_sweep_start = time.time()
for bms_idx, bms in enumerate(bin_sizes_ms):
    print(f"\n{'=' * 58}")
    print(f"  Bin size: {bms} ms  (n_repeats={n_repeats})")
    print(f"{'=' * 58}")
    t_bin_start = time.time()

    X, y, bc, _ = prepare_task_stim_type(pkl_file, class_pos, class_neg, TASK_COL_MAP, random_state=random_state, bin_size_sec=bms / 1000.0)
    extras, keep = build_task_kinematics(bc, bms / 1000.0, kin_samples, columns=HEADING_ONLY_COLUMNS)
    n_dropped = int((~keep).sum())
    X = X[keep]; y = y[keep]
    X_aug = np.hstack([X, extras[keep].astype(X.dtype)])

    n_stim = int((y != 0).sum())
    n_iti  = int((y == 0).sum())
    print(f"  n_total_bins={len(y)}  n_dropped_no_tracking={n_dropped}  "
          f"n_stim={n_stim}  n_iti={n_iti}  n_units={X.shape[1]}  "
          f"X_aug feature dim={X_aug.shape[1]} "
          f"(added: sin/cos heading + sin/cos head_angle)")

    pbar = tqdm(total=n_repeats * len(CLASSIFIERS),
                desc=f"  bin {bms} ms CV", ncols=90, leave=True)
    for r in range(n_repeats):
        cv_seed = random_state + 1000 * bms + r
        for name, clf_proto in CLASSIFIERS.items():
            _, mean_b, ch_b, pcm_b, _ = run_cv_balanced_train(
                name, clf_proto, X, y,
                n_splits=n_splits, random_state=cv_seed,
            )
            _, mean_a, ch_a, pcm_a, _ = run_cv_balanced_train(
                name, clf_proto, X_aug, y,
                n_splits=n_splits, random_state=cv_seed,
            )
            per_repeat_baseline[name][bms_idx].append(mean_b)
            per_repeat_augmented[name][bms_idx].append(mean_a)
            for c, m in pcm_b.items():
                per_repeat_pc_baseline[name].setdefault(
                    c, [[] for _ in range(n_bins_sweep)])[bms_idx].append(m)
            for c, m in pcm_a.items():
                per_repeat_pc_augmented[name].setdefault(
                    c, [[] for _ in range(n_bins_sweep)])[bms_idx].append(m)
            pbar.set_postfix_str(f"r={r+1}/{n_repeats} {name}")
            pbar.update(1)
    pbar.close()

    for name in clf_names_seed:
        b_mu = float(np.mean(per_repeat_baseline[name][bms_idx]))
        b_sd = float(np.std(per_repeat_baseline[name][bms_idx]))
        a_mu = float(np.mean(per_repeat_augmented[name][bms_idx]))
        a_sd = float(np.std(per_repeat_augmented[name][bms_idx]))
        print(f"  [{name}]  baseline={b_mu:.3f}±{b_sd:.3f}  "
              f"augmented={a_mu:.3f}±{a_sd:.3f}  "
              f"Δ={a_mu - b_mu:+.3f}")

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

    chance_baseline.append(ch_b)
    chance_augmented.append(ch_a)
    sweep_stats.append({
        'bin_ms': bms, 'n_total': len(y), 'n_dropped': n_dropped,
        'n_stim': n_stim, 'n_iti': n_iti,
        'n_units': int(X.shape[1]),
        'bin_elapsed_sec': float(bin_elapsed),
    })

print(f"\nTotal CV sweep wall time: {(time.time() - t_sweep_start)/60:.2f} min")

for name in clf_names_seed:
    for bms_idx in range(n_bins_sweep):
        accs_b = np.array(per_repeat_baseline[name][bms_idx])
        accs_a = np.array(per_repeat_augmented[name][bms_idx])
        results_baseline[name]['means'].append(float(np.mean(accs_b)))
        results_baseline[name]['stds'].append(float(np.std(accs_b)))
        results_augmented[name]['means'].append(float(np.mean(accs_a)))
        results_augmented[name]['stds'].append(float(np.std(accs_a)))
    for c, per_bin_lists in per_repeat_pc_baseline[name].items():
        results_baseline[name]['per_class'].setdefault(c, {'means': [], 'stds': []})
        for vals in per_bin_lists:
            results_baseline[name]['per_class'][c]['means'].append(float(np.nanmean(vals)))
            results_baseline[name]['per_class'][c]['stds'].append(float(np.nanstd(vals)))
    for c, per_bin_lists in per_repeat_pc_augmented[name].items():
        results_augmented[name]['per_class'].setdefault(c, {'means': [], 'stds': []})
        for vals in per_bin_lists:
            results_augmented[name]['per_class'][c]['means'].append(float(np.nanmean(vals)))
            results_augmented[name]['per_class'][c]['stds'].append(float(np.nanstd(vals)))

mean_chance_baseline  = float(np.mean(chance_baseline))
mean_chance_augmented = float(np.mean(chance_augmented))
all_classes = sorted(results_baseline[next(iter(results_baseline))]['per_class'].keys())


# ------------------------------------------------------------------ #
#  Step 4 — accuracy comparison figure                                #
# ------------------------------------------------------------------ #
def _plot_overlay(ax, results_a, results_b, bin_sizes_ms, key, title):
    """Solid = augmented (firing rates + circ features), dashed = baseline (firing rates only)."""
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for (name, res_a), marker in zip(results_a.items(), markers):
        res_b = results_b[name]
        if key == 'overall':
            ma, sa = np.array(res_a['means']), np.array(res_a['stds'])
            mb     = np.array(res_b['means'])
        else:
            ma = np.array(res_a['per_class'][key]['means'])
            sa = np.array(res_a['per_class'][key]['stds'])
            mb = np.array(res_b['per_class'][key]['means'])
        line, = ax.plot(bin_sizes_ms, ma, marker=marker, linewidth=1.6, label=name)
        ax.fill_between(bin_sizes_ms, ma - sa, ma + sa,
                        color=line.get_color(), alpha=0.12, linewidth=0)
        ax.plot(bin_sizes_ms, mb, marker=marker, linestyle='--',
                linewidth=1.1, color=line.get_color())
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Bin size (ms)', fontsize=10)
    ax.set_ylabel('Accuracy (recall)', fontsize=10)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms])
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.set_ylim(0.0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)


n_class_panels = len(all_classes)
fig = plt.figure(figsize=(6 * (n_class_panels + 1), 9))
ax_top   = fig.add_subplot(2, 1, 1)
axes_bot = [fig.add_subplot(2, n_class_panels, n_class_panels + i + 1)
            for i in range(n_class_panels)]

_plot_overlay(ax_top, results_augmented, results_baseline, bin_sizes_ms,
              'overall',
              'Overall accuracy — solid: firing rates + sin/cos(heading + head_angle), dashed: firing rates only')
_style_handles = [
    plt.Line2D([], [], color='black', linestyle='-',  linewidth=1.6, label='+ heading features'),
    plt.Line2D([], [], color='black', linestyle='--', linewidth=1.1, label='baseline'),
]
clf_handles = ax_top.get_legend_handles_labels()[0]
clf_labels  = ax_top.get_legend_handles_labels()[1]
ax_top.legend(clf_handles + _style_handles,
              clf_labels  + ['+ heading features', 'baseline'],
              fontsize=8, loc='lower right', ncol=3)

for ax, c in zip(axes_bot, all_classes):
    _plot_overlay(ax, results_augmented, results_baseline, bin_sizes_ms, c,
                  CLASS_LABELS.get(c, f'Class {c}'))
    ax.legend(fontsize=8, loc='lower right', ncol=2)

plt.tight_layout()

info = (
    f"Data: {Path(pkl_file).name}  |  "
    f"Bins: {bin_sizes_ms[0]}–{bin_sizes_ms[-1]} ms  |  "
    f"CV: {n_splits}-fold  |  "
    f"Train: balanced (undersampled), Test: natural ratio  |  "
    f"Features added: sin/cos(heading), sin/cos(head_angle)  |  "
    f"Chance baseline={mean_chance_baseline:.2f}  augmented={mean_chance_augmented:.2f}  |  "
    f"Seed: {random_state}  |  n_repeats={n_repeats}"
)
fig.text(0.5, -0.01, info, ha='center', va='top', fontsize=7.5, color='gray')

stem = f"task_{session}_heading_feature_vs_baseline_{n_splits}fold"
fig_path = out_dir / f"{stem}.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nComparison figure saved → {fig_path}")


# ------------------------------------------------------------------ #
#  Step 5 — paired bootstrap of Δ = augmented − baseline              #
# ------------------------------------------------------------------ #
print(f"\nBootstrapping paired Δ = augmented − baseline "
      f"(n_repeats={n_repeats}, n_bootstrap={n_bootstrap}) ...")
rng_boot = np.random.default_rng(random_state + 7777)
delta_summary = {n: {'mean': [], 'ci_lo': [], 'ci_hi': [], 'p_two_sided': []}
                 for n in clf_names_seed}

for name in clf_names_seed:
    for i in range(n_bins_sweep):
        accs_b = np.array(per_repeat_baseline[name][i])
        accs_a = np.array(per_repeat_augmented[name][i])
        deltas = accs_a - accs_b
        idx = rng_boot.integers(0, deltas.size, size=(n_bootstrap, deltas.size))
        bs_means = deltas[idx].mean(axis=1)
        observed = float(np.mean(deltas))
        centered = bs_means - observed
        p_two = float(np.mean(np.abs(centered) >= abs(observed)))
        delta_summary[name]['mean'].append(observed)
        delta_summary[name]['ci_lo'].append(float(np.percentile(bs_means, 2.5)))
        delta_summary[name]['ci_hi'].append(float(np.percentile(bs_means, 97.5)))
        delta_summary[name]['p_two_sided'].append(p_two)

delta_summary_pc = {c: {n: {'mean': [], 'ci_lo': [], 'ci_hi': [], 'p_two_sided': []}
                        for n in clf_names_seed} for c in all_classes}
for c in all_classes:
    for name in clf_names_seed:
        for i in range(n_bins_sweep):
            ab = np.array(per_repeat_pc_baseline[name].get(c, [[]] * n_bins_sweep)[i])
            aa = np.array(per_repeat_pc_augmented[name].get(c, [[]] * n_bins_sweep)[i])
            if ab.size == 0 or aa.size == 0:
                delta_summary_pc[c][name]['mean'].append(np.nan)
                delta_summary_pc[c][name]['ci_lo'].append(np.nan)
                delta_summary_pc[c][name]['ci_hi'].append(np.nan)
                delta_summary_pc[c][name]['p_two_sided'].append(np.nan)
                continue
            d = aa - ab
            idx = rng_boot.integers(0, d.size, size=(n_bootstrap, d.size))
            bs = d[idx].mean(axis=1)
            observed = float(np.nanmean(d))
            centered = bs - observed
            p_two = float(np.mean(np.abs(centered) >= abs(observed)))
            delta_summary_pc[c][name]['mean'].append(observed)
            delta_summary_pc[c][name]['ci_lo'].append(float(np.nanpercentile(bs, 2.5)))
            delta_summary_pc[c][name]['ci_hi'].append(float(np.nanpercentile(bs, 97.5)))
            delta_summary_pc[c][name]['p_two_sided'].append(p_two)


# ------------------------------------------------------------------ #
#  Figure: heading-feature gain (Δ ± 95% CI) per classifier           #
# ------------------------------------------------------------------ #
def _plot_delta(ax, summary_dict, title):
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for (name, marker) in zip(clf_names_seed, markers):
        means = np.array(summary_dict[name]['mean'])
        los   = np.array(summary_dict[name]['ci_lo'])
        his   = np.array(summary_dict[name]['ci_hi'])
        line, = ax.plot(bin_sizes_ms, means, marker=marker, linewidth=1.6, label=name)
        ax.fill_between(bin_sizes_ms, los, his,
                        color=line.get_color(), alpha=0.18, linewidth=0)
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Bin size (ms)', fontsize=10)
    ax.set_ylabel('Δ accuracy  (augmented − baseline)', fontsize=10)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms])
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.spines[['top', 'right']].set_visible(False)


fig_d = plt.figure(figsize=(6 * (n_class_panels + 1), 9))
ax_d_top   = fig_d.add_subplot(2, 1, 1)
axes_d_bot = [fig_d.add_subplot(2, n_class_panels, n_class_panels + i + 1)
              for i in range(n_class_panels)]

_plot_delta(ax_d_top, delta_summary,
            'Overall Δ accuracy: gain from adding sin/cos heading + head_angle (paired, 95% CI bootstrap)')
ax_d_top.legend(fontsize=8, loc='upper right', ncol=2)
for ax, c in zip(axes_d_bot, all_classes):
    _plot_delta(ax, delta_summary_pc[c],
                f"{CLASS_LABELS.get(c, f'Class {c}')}  —  Δ recall")
    ax.legend(fontsize=7, loc='upper right', ncol=2)

info_d = (
    f"n_repeats={n_repeats}  |  n_bootstrap={n_bootstrap}  |  "
    f"Bands = 95% paired-bootstrap CI of mean Δ  |  "
    f"Δ > 0 ⇒ heading features helped; CI excluding 0 ⇒ effect is significant"
)
fig_d.text(0.5, -0.01, info_d, ha='center', va='top', fontsize=8, color='gray')
plt.tight_layout()
fig_d_path = out_dir / f"task_{session}_heading_feature_delta_{n_splits}fold.png"
fig_d.savefig(fig_d_path, dpi=150, bbox_inches='tight')
print(f"Δ figure saved → {fig_d_path}")


# ------------------------------------------------------------------ #
#  Step 6 — best bin (augmented), shuffle null, bar plot              #
# ------------------------------------------------------------------ #
clf_names = list(CLASSIFIERS.keys())
mean_acc_per_bin = np.mean(
    [[results_augmented[n]['means'][i] for n in clf_names] for i in range(len(bin_sizes_ms))],
    axis=1,
)
bar_idx    = int(np.argmax(mean_acc_per_bin))
bar_bin_ms = bin_sizes_ms[bar_idx]
print(f"\nBest augmented bin size (argmax mean acc): {bar_bin_ms} ms  "
      f"(mean acc = {mean_acc_per_bin[bar_idx]:.3f})")

X_bar, y_bar, bc_bar, _ = prepare_task_stim_type(pkl_file, class_pos, class_neg, TASK_COL_MAP, random_state=random_state, bin_size_sec=bar_bin_ms / 1000.0)
extras_bar, keep_bar = build_task_kinematics(bc_bar, bar_bin_ms / 1000.0, kin_samples, columns=HEADING_ONLY_COLUMNS)
X_bar, y_bar = X_bar[keep_bar], y_bar[keep_bar]
X_bar_aug = np.hstack([X_bar, extras_bar[keep_bar].astype(X_bar.dtype)])

print(f"Computing shuffle null on augmented data at {bar_bin_ms} ms ...")
shuffle_results = {}
for name, clf_proto in CLASSIFIERS.items():
    sm, ss, spc_m, spc_s = run_cv_balanced_train_shuffle(
        name, clf_proto, X_bar_aug, y_bar,
        n_splits=n_splits, random_state=random_state,
    )
    shuffle_results[name] = {
        'mean': sm, 'std': ss,
        'per_class': {c: {'mean': spc_m[c], 'std': spc_s[c]} for c in spc_m},
    }
    print(f"  [shuffle {name}]  shuf_mean={sm:.3f}")

n_clf    = len(clf_names)
n_panels = len(all_classes)
x        = np.arange(n_clf)
width    = 0.55
bar_w    = width * 0.3
offset   = width * 0.35
color_base = '#7f7f7f'
color_aug  = '#4C72B0'
color_shuf = '#AAAAAA'

fig2      = plt.figure(figsize=(max(8, n_clf * 1.6) * max(n_panels, 1), 10))
ax2_top   = fig2.add_subplot(2, 1, 1)
axes2_bot = [fig2.add_subplot(2, n_panels, n_panels + i + 1) for i in range(n_panels)]


def _bar_three(ax, base_means, base_stds, aug_means, aug_stds,
               shuf_means, shuf_stds, title, ylabel):
    ax.bar(x - offset, base_means, bar_w, color=color_base,
           yerr=base_stds, capsize=3, error_kw=dict(linewidth=1), label='Baseline')
    bars_a = ax.bar(x, aug_means, bar_w, color=color_aug,
                    yerr=aug_stds, capsize=3, error_kw=dict(linewidth=1), label='+ Heading feats')
    ax.bar(x + offset, shuf_means, bar_w, color=color_shuf,
           yerr=shuf_stds, capsize=3, error_kw=dict(linewidth=1), label='Shuffle (aug)')
    for bar, m in zip(bars_a, aug_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.2f}', ha='center', va='bottom', fontsize=8, color=color_aug)
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper right')


_bar_three(
    ax2_top,
    [results_baseline[n]['means'][bar_idx]  for n in clf_names],
    [results_baseline[n]['stds'][bar_idx]   for n in clf_names],
    [results_augmented[n]['means'][bar_idx] for n in clf_names],
    [results_augmented[n]['stds'][bar_idx]  for n in clf_names],
    [shuffle_results[n]['mean']             for n in clf_names],
    [shuffle_results[n]['std']              for n in clf_names],
    f'Overall accuracy — baseline vs + heading features vs shuffle  (bin = {bar_bin_ms} ms)',
    'Accuracy',
)

for ax, c in zip(axes2_bot, all_classes):
    _bar_three(
        ax,
        [results_baseline[n]['per_class'][c]['means'][bar_idx]  for n in clf_names],
        [results_baseline[n]['per_class'][c]['stds'][bar_idx]   for n in clf_names],
        [results_augmented[n]['per_class'][c]['means'][bar_idx] for n in clf_names],
        [results_augmented[n]['per_class'][c]['stds'][bar_idx]  for n in clf_names],
        [shuffle_results[n]['per_class'][c]['mean']             for n in clf_names],
        [shuffle_results[n]['per_class'][c]['std']              for n in clf_names],
        f'{CLASS_LABELS.get(c, f"Class {c}")}  (bin = {bar_bin_ms} ms)',
        'Recall',
    )

fig2.suptitle(
    f'Per-classifier decoding — heading features vs firing-rate baseline  |  '
    f'bin = {bar_bin_ms} ms, {n_splits}-fold CV\n'
    f'Balanced train / natural-ratio test',
    fontsize=11, y=1.01,
)
plt.tight_layout()
fig2_path = out_dir / f"task_{session}_heading_feature_bar{bar_bin_ms}ms.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Bar-plot figure saved → {fig2_path}")


# ------------------------------------------------------------------ #
#  Step 6b — z relative to shuffle null at the best bin               #
# ------------------------------------------------------------------ #
print(f"\n--- z relative to shuffle null at bin = {bar_bin_ms} ms ---")
print(f"{'Classifier':>14}  {'acc_base':>8}  {'acc_aug':>8}  "
      f"{'shuf_mean':>9}  {'z_base':>7}  {'z_aug':>7}  {'Δ_aug−base':>11}  {'p_Δ':>7}")
z_summary = {}
for name in clf_names:
    sm_mu = float(shuffle_results[name]['mean'])
    sm_sd = float(max(shuffle_results[name]['std'], 1e-9))
    ab = float(results_baseline[name]['means'][bar_idx])
    aa = float(results_augmented[name]['means'][bar_idx])
    zb = (ab - sm_mu) / sm_sd
    za = (aa - sm_mu) / sm_sd
    d_mu = float(delta_summary[name]['mean'][bar_idx])
    p_d  = float(delta_summary[name]['p_two_sided'][bar_idx])
    z_summary[name] = {
        'acc_baseline':  ab, 'acc_augmented': aa,
        'shuffle_mean':  sm_mu, 'shuffle_std':  sm_sd,
        'z_baseline':    zb, 'z_augmented':   za,
        'delta_mean':    d_mu, 'delta_p':      p_d,
    }
    print(f"{name:>14}  {ab:>8.3f}  {aa:>8.3f}  {sm_mu:>9.3f}  "
          f"{zb:>+7.2f}  {za:>+7.2f}  {d_mu:>+11.3f}  {p_d:>7.4f}")
print("Interpretation: Δ > 0 and CI excluding 0 ⇒ adding heading + head_angle as features "
      "carries information beyond firing rates alone.")


# ------------------------------------------------------------------ #
#  Step 7 — save results pkl                                          #
# ------------------------------------------------------------------ #
pkl_out = out_dir / f"task_{session}_heading_feature_{n_splits}fold.pkl"
with open(pkl_out, 'wb') as f:
    pickle.dump({
        'bin_sizes_ms':            bin_sizes_ms,
        'results_baseline':        results_baseline,
        'results_augmented':       results_augmented,
        'shuffle_results':         shuffle_results,
        'chance_baseline':         mean_chance_baseline,
        'chance_augmented':        mean_chance_augmented,
        'n_splits':                n_splits,
        'sweep_stats':             sweep_stats,
        'best_bin_ms':             bar_bin_ms,
        'clean_position_stats':    clean_stats,
        'n_repeats':               n_repeats,
        'n_bootstrap':             n_bootstrap,
        'feature_columns_added':   HEADING_ONLY_COLUMNS,
        'session_to_window_offset_sec': offset,
        'class_pos':               class_pos,
        'class_neg':               class_neg,
        'col_map':                 TASK_COL_MAP,
        'rewarded_combination':    rewarded_combination,
        'per_repeat_baseline':     per_repeat_baseline,
        'per_repeat_augmented':    per_repeat_augmented,
        'per_repeat_pc_baseline':  per_repeat_pc_baseline,
        'per_repeat_pc_augmented': per_repeat_pc_augmented,
        'delta_summary':           delta_summary,
        'delta_summary_per_class': delta_summary_pc,
        'z_summary_at_best_bin':   z_summary,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved → {pkl_out}")

plt.show()
