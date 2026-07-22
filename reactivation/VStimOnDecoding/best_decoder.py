"""
Re-plot the figures from compare_task_decoders_with_kinematics.py using ONLY
Random Forest, from the already-saved results pickle. No CV is re-run.

Reproduces:
  * accuracy vs bin size  (overall + per-class panels)
  * Δ accuracy vs bin size  (paired bootstrap, 95% CI bands)
  * best-bin bar plot  (Baseline / + Kinematics / Shuffle, overall + per-class)
  * z-vs-shuffle summary table at the chosen bin

Outputs are written next to the input pickle with a `_rf` suffix so the
original figures are not overwritten.
"""

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from server_fallback import resolve_output_folder


# ------------------------------------------------------------------ #
#  Inputs                                                             #
# ------------------------------------------------------------------ #
pkl = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313\reactivation\task_CnL42SG_20260313_kinematics_feature_5fold.pkl"
CLF = 'Random Forest'

pkl_path = Path(pkl)
out_dir  = resolve_output_folder(pkl_path.parent)
stem_in  = pkl_path.stem                              # e.g. task_<session>_kinematics_feature_5fold
session_tag = stem_in.replace('_kinematics_feature_', '_kinematics_feature_').replace('5fold', '')  # purely cosmetic; we just reuse stem_in for filenames

with open(pkl_path, 'rb') as f:
    R = pickle.load(f)

bin_sizes_ms       = list(R['bin_sizes_ms'])
n_splits           = R['n_splits']
n_repeats          = R['n_repeats']
n_bootstrap        = R['n_bootstrap']
chance_b           = float(R['chance_baseline'])
chance_a           = float(R['chance_augmented'])
class_pos          = R['class_pos']
class_neg          = R['class_neg']
rewarded           = R.get('rewarded_combination')

if CLF not in R['results_baseline']:
    raise KeyError(f"{CLF!r} not in results_baseline; available: {list(R['results_baseline'])}")

res_b   = R['results_baseline'][CLF]
res_a   = R['results_augmented'][CLF]
shuf    = R['shuffle_results'][CLF]
dsum    = R['delta_summary'][CLF]
dsum_pc = {c: R['delta_summary_per_class'][c][CLF]
           for c in R['delta_summary_per_class']}

all_classes = sorted(res_b['per_class'].keys())
n_bins_sweep = len(bin_sizes_ms)


def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())


CLASS_LABELS = {
    -1: f'Stim on — left = {_fmt_class(class_neg)}  (-1)',
     0: 'ITI (0)',
     1: f'Stim on — left = {_fmt_class(class_pos)}  (+1)',
}

RF_COLOR   = '#4C72B0'
COLOR_BASE = '#7f7f7f'
COLOR_AUG  = '#4C72B0'
COLOR_SHUF = '#AAAAAA'


# ------------------------------------------------------------------ #
#  Figure 1 — accuracy vs bin size (single classifier overlay)        #
# ------------------------------------------------------------------ #
def _plot_acc(ax, key, title):
    if key == 'overall':
        ma = np.asarray(res_a['means']);  sa = np.asarray(res_a['stds'])
        mb = np.asarray(res_b['means']);  sb = np.asarray(res_b['stds'])
    else:
        ma = np.asarray(res_a['per_class'][key]['means'])
        sa = np.asarray(res_a['per_class'][key]['stds'])
        mb = np.asarray(res_b['per_class'][key]['means'])
        sb = np.asarray(res_b['per_class'][key]['stds'])
    ax.plot(bin_sizes_ms, ma, marker='o', linewidth=3.5, markersize=10,
            color=RF_COLOR, label='+ kinematics')
    ax.fill_between(bin_sizes_ms, ma - sa, ma + sa,
                    color=RF_COLOR, alpha=0.18, linewidth=0)
    ax.plot(bin_sizes_ms, mb, marker='o', linestyle='--', markersize=10,
            linewidth=3.0, color=RF_COLOR, label='baseline')
    ax.fill_between(bin_sizes_ms, mb - sb, mb + sb,
                    color=RF_COLOR, alpha=0.08, linewidth=0)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=12)
    ax.set_xlabel('Bin size (ms)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Accuracy (recall)', fontsize=20, fontweight='bold')
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms],
                       rotation=45, ha='right')
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.set_ylim(0.0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(axis='both', labelsize=16, width=2.0, length=7)
    ax.legend(fontsize=16, loc='lower right', frameon=False)


n_class_panels = len(all_classes)
fig1 = plt.figure(figsize=(6.0 * max(n_class_panels, 1) + 3.0, 12))
ax_top = fig1.add_subplot(2, 1, 1)
axes_bot = [fig1.add_subplot(2, n_class_panels, n_class_panels + i + 1)
            for i in range(n_class_panels)]

_plot_acc(ax_top, 'overall',
          f'{CLF} — overall accuracy  '
          f'(solid: firing rates + speed + sin/cos(heading + head_angle); '
          f'dashed: firing rates only)')
ax_top.set_box_aspect(1 / 1.5)
for ax, c in zip(axes_bot, all_classes):
    _plot_acc(ax, c, CLASS_LABELS.get(c, f'Class {c}'))

plt.tight_layout()
info = (
    f"Data: {pkl_path.name}  |  Classifier: {CLF}  |  "
    f"Bins: {bin_sizes_ms[0]}–{bin_sizes_ms[-1]} ms  |  "
    f"CV: {n_splits}-fold  |  Balanced train / natural-ratio test  |  "
    f"Chance baseline={chance_b:.2f}  augmented={chance_a:.2f}  |  "
    f"n_repeats={n_repeats}"
)
fig1.text(0.5, -0.01, info, ha='center', va='top', fontsize=12, color='gray')

fig1_path = out_dir / f"{stem_in}_rf_vs_baseline.png"
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
fig1.savefig(fig1_path.with_suffix('.svg'), bbox_inches='tight', transparent=True)
print(f"Accuracy figure  -> {fig1_path}  (+ .svg)")


# ------------------------------------------------------------------ #
#  Figure 2 — Δ accuracy vs bin size with paired-bootstrap CI band    #
# ------------------------------------------------------------------ #
def _plot_delta(ax, summary, title):
    means = np.asarray(summary['mean'])
    los   = np.asarray(summary['ci_lo'])
    his   = np.asarray(summary['ci_hi'])
    ax.plot(bin_sizes_ms, means, marker='o', linewidth=3.5, markersize=10,
            color=RF_COLOR, label=CLF)
    ax.fill_between(bin_sizes_ms, los, his,
                    color=RF_COLOR, alpha=0.22, linewidth=0)
    ax.axhline(0.0, color='black', linewidth=2.0, linestyle=':')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=12)
    ax.set_xlabel('Bin size (ms)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Δ accuracy  (augmented − baseline)',
                  fontsize=14, fontweight='bold')
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms],
                       rotation=45, ha='right')
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(axis='both', labelsize=16, width=2.0, length=7)
    ax.legend(fontsize=16, loc='upper right', frameon=False)


fig2 = plt.figure(figsize=(6.0 * max(n_class_panels, 1) + 3.0, 12))
ax_d_top = fig2.add_subplot(2, 1, 1)
axes_d_bot = [fig2.add_subplot(2, n_class_panels, n_class_panels + i + 1)
              for i in range(n_class_panels)]

_plot_delta(ax_d_top, dsum,
            f'{CLF} — Δ accuracy: gain from adding speed + sin/cos(heading + head_angle) '
            f'(paired, 95% CI bootstrap)')
ax_d_top.set_box_aspect(1 / 1.5)
for ax, c in zip(axes_d_bot, all_classes):
    _plot_delta(ax, dsum_pc[c],
                f"{CLASS_LABELS.get(c, f'Class {c}')}  —  Δ recall")

info_d = (
    f"Classifier: {CLF}  |  n_repeats={n_repeats}  |  n_bootstrap={n_bootstrap}  |  "
    f"Band = 95% paired-bootstrap CI of mean Δ  |  "
    f"Δ > 0 ⇒ kinematic features helped; CI excluding 0 ⇒ effect is significant"
)
fig2.text(0.5, -0.01, info_d, ha='center', va='top', fontsize=12, color='gray')
plt.tight_layout()
fig2_path = out_dir / f"{stem_in}_rf_delta.png"
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
fig2.savefig(fig2_path.with_suffix('.svg'), bbox_inches='tight', transparent=True)
print(f"Delta figure     -> {fig2_path}  (+ .svg)")


# ------------------------------------------------------------------ #
#  Figure 3 — best-bin per-class bar plots                            #
#    Fig 3a: decoding accuracy (+ kinematics)                         #
#    Fig 3b: shuffled accuracy                                        #
#    Layout: Cue 0 (sf=0.04, green) | Cue 1 (sf=0.16, blue) | gap |   #
#            No cue (ITI, gray)                                       #
# ------------------------------------------------------------------ #
rf_means = np.asarray(res_a['means'])
bar_idx  = int(np.argmax(rf_means))
bar_bin  = bin_sizes_ms[bar_idx]
print(f"\nBest {CLF} bin (argmax of augmented mean acc): {bar_bin} ms "
      f"(mean = {rf_means[bar_idx]:.3f})")
print(f"  [pkl's recorded best_bin_ms (across classifiers) = {R.get('best_bin_ms')}]")

# Resolve which stim class label (-1 / +1) corresponds to which spatial freq.
sf_pos = class_pos.get('spatial_freq')
sf_neg = class_neg.get('spatial_freq')
sf_to_class = {round(float(sf_pos), 3): +1, round(float(sf_neg), 3): -1}
try:
    cue0_class = sf_to_class[0.04]   # sf = 0.04  -> "cue 0" (green)
    cue1_class = sf_to_class[0.16]   # sf = 0.16  -> "cue 1" (blue)
except KeyError as e:
    raise KeyError(f"Could not map sf to class. class_pos={class_pos}, "
                   f"class_neg={class_neg}") from e
iti_class = 0

CUE_GREEN = '#2ca02c'
CUE_BLUE  = '#1f77b4'
NOCUE_GRAY = '#7f7f7f'

BAR_W = 0.45
bar_specs = [
    ('cue 0\n(sf=0.04)',  cue0_class, CUE_GREEN,  0.0),
    ('cue 1\n(sf=0.16)',  cue1_class, CUE_BLUE,   BAR_W),       # touching cue 0
    ('No cue\n(ITI)',     iti_class,  NOCUE_GRAY, BAR_W + 1.0), # gap before No cue
]
positions = [s[3] for s in bar_specs]
labels    = [s[0] for s in bar_specs]
colors    = [s[2] for s in bar_specs]
classes   = [s[1] for s in bar_specs]


def _per_class_pair(source):
    """Return (means, stds) lists in the bar order. `source['per_class'][c]` holds
    either {'means': [..bins..], 'stds': [..bins..]} (decoding case) or
    {'mean': float, 'std': float} (shuffle case)."""
    means, stds = [], []
    for c in classes:
        pc = source['per_class'][c]
        if 'means' in pc:
            means.append(float(pc['means'][bar_idx]))
            stds.append(float(pc['stds'][bar_idx]))
        else:
            means.append(float(pc['mean']))
            stds.append(float(pc['std']))
    return means, stds


def _draw_bars(ax, means, stds, title, ylabel):
    ax.bar(positions, means, width=BAR_W,
           color=colors, yerr=stds, capsize=10,
           edgecolor='black', linewidth=1.5,
           error_kw=dict(linewidth=2.5))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=22, fontweight='bold',
                       rotation=45, ha='right')
    for tick, col in zip(ax.get_xticklabels(), colors):
        tick.set_color(col)
    ax.set_xlim(positions[0] - 0.35, positions[-1] + 0.35)
    top = float(max(1.0, max(m + s for m, s in zip(means, stds)) + 0.04))
    ax.set_ylim(0, top)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'], fontsize=22)
    ax.set_title(title, fontsize=26, fontweight='bold', pad=16)
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', width=2.5, length=9)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)


# --- Combined figure: decoding (left) + shuffle (right) --- #
aug_means,  aug_stds  = _per_class_pair(res_a)
shuf_means, shuf_stds = _per_class_pair(shuf)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 11), sharey=True)
_draw_bars(
    ax3a, aug_means, aug_stds,
    f'{CLF} decoding (+ kinematics)\nbin = {bar_bin} ms',
    'Fraction of cue presentations\ncorrectly identified',
)
_draw_bars(
    ax3b, shuf_means, shuf_stds,
    f'{CLF} shuffled labels\nbin = {bar_bin} ms',
    'Fraction of classified events\nin randomized data',
)
ax3a.set_box_aspect(10 / 6)
ax3b.set_box_aspect(10 / 6)
plt.tight_layout()
fig3_path = out_dir / f"{stem_in}_rf_bar{bar_bin}ms.png"
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
fig3.savefig(fig3_path.with_suffix('.svg'), bbox_inches='tight', transparent=True)
print(f"Bar-plot figure  -> {fig3_path}  (+ .svg)")


# ------------------------------------------------------------------ #
#  Console — z relative to shuffle null at the best (RF) bin          #
# ------------------------------------------------------------------ #
sm_mu = float(shuf['mean'])
sm_sd = float(max(shuf['std'], 1e-9))
ab    = float(res_b['means'][bar_idx])
aa    = float(res_a['means'][bar_idx])
zb    = (ab - sm_mu) / sm_sd
za    = (aa - sm_mu) / sm_sd
d_mu  = float(dsum['mean'][bar_idx])
p_d   = float(dsum['p_two_sided'][bar_idx])

print(f"\n--- {CLF} at bin = {bar_bin} ms ---")
print(f"{'acc_base':>9}  {'acc_aug':>8}  {'shuf_mean':>9}  "
      f"{'z_base':>7}  {'z_aug':>7}  {'delta':>11}  {'p_delta':>8}")
print(f"{ab:>9.3f}  {aa:>8.3f}  {sm_mu:>9.3f}  "
      f"{zb:>+7.2f}  {za:>+7.2f}  {d_mu:>+11.3f}  {p_d:>8.4f}")
print("Note: chance value (majority-class prevalence) is "
      f"{chance_b:.3f} — compare against it when reading the bars.")

plt.show()
