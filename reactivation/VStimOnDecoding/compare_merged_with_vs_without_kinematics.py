"""
Does adding kinematics to the merged (task+passive) decoder help?

Side-by-side comparison of the MERGED decoder trained on:
  * spikes only                            (no kinematics)
  * spikes + [speed, sin/cos heading, sin/cos head_angle]
        - TASK rows: per-bin DLC means (window time)
        - PASSIVE rows: zero-filled (head-fixed animal)

For each bin size and each of the 7 classifiers we run two CV evaluations
(no-kin vs +kin) on the SAME merged feature matrix shape (only the appended
5 kinematic columns differ). Both training pipelines mirror
compare_merged_vs_unmerged_decoders.py / _with_kinematics.py exactly so the
delta is attributable to the kinematic columns alone.

Figure: 7 rows (classifiers) x 4 cols (overall + recall per class).
Each panel overlays the two merged curves; a thin Δ-recall annotation marks
where +kin > no-kin at each bin.
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

with open(task_pkl, 'rb') as _f:
    rewarded_combination = infer_rewarded_combination(pickle.load(_f)['trial_params'])
print(f"Class +1 (left grating matches): {class_pos}")
print(f"Class -1 (left grating matches): {class_neg}")
print(f"Rewarded grating in this session: {rewarded_combination}")
N_KIN = N_KINEMATIC_FEATURES


# ------------------------------------------------------------------ #
#  Step 1 - load TASK DLC, build kinematic sample arrays              #
# ------------------------------------------------------------------ #
kin_samples, clean_stats, offset = load_task_kinematic_samples(task_pkl)
print_kinematic_sample_report(kin_samples, clean_stats, offset)


# ------------------------------------------------------------------ #
#  Step 3 - sweep bin sizes: merged no-kin vs merged +kin             #
# ------------------------------------------------------------------ #
VARIANTS = ['no_kin', 'with_kin']
CLASSIFIERS = make_classifiers(random_state)

results = {
    v: {name: {'means': [], 'stds': [], 'per_class': {}} for name in CLASSIFIERS}
    for v in VARIANTS
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
    # Drop bins without tracking so the two variants use the SAME rows.
    X_t, y_t, kin_t = X_t[keep_t], y_t[keep_t], kin_t[keep_t]

    # ---- Passive ----
    print("[PASSIVE]")
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False,
        random_state=random_state,
    )
    kin_p = np.zeros((X_p.shape[0], N_KIN), dtype=np.float64)

    # ---- Align units (intersection, sorted) ----
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

    # ---- Build merged: spikes-only and spikes+kin (same rows, same labels) ----
    X_m_nokin = np.vstack([X_t_a, X_p_a])
    X_t_aug   = np.hstack([X_t_a, kin_t.astype(X_t_a.dtype)])
    X_p_aug   = np.hstack([X_p_a, kin_p.astype(X_p_a.dtype)])
    X_m_kin   = np.vstack([X_t_aug, X_p_aug])
    y_m       = np.concatenate([y_t, y_p])

    variants_xy = {
        'no_kin':   (X_m_nokin, y_m),
        'with_kin': (X_m_kin,   y_m),
    }
    for v_name, (X, y) in variants_xy.items():
        print(f"  [merged {v_name}] X={X.shape}  "
              f"+1={np.sum(y==1)}  -1={np.sum(y==-1)}  0={np.sum(y==0)}")

    # Average over n_repeats CV passes per (variant, classifier). Same cv_seed
    # is reused across variants within a repeat so the paired Δ across no_kin
    # vs with_kin stays valid.
    pbar = tqdm(total=len(variants_xy) * len(CLASSIFIERS) * n_repeats,
                desc=f"  bin {bms} ms CV", ncols=90, leave=True)
    rep_means = {v: {n: [] for n in CLASSIFIERS} for v in variants_xy}
    rep_pc    = {v: {n: {} for n in CLASSIFIERS} for v in variants_xy}
    for r in range(n_repeats):
        cv_seed = random_state + 1000 * bms + r
        for v_name, (X, y) in variants_xy.items():
            for name, clf_proto in CLASSIFIERS.items():
                _, mean_r, _, pcm_r, _ = run_cv_balanced_train(
                    name, clf_proto, X, y,
                    n_splits=n_splits, random_state=cv_seed,
                )
                rep_means[v_name][name].append(mean_r)
                for c, m in pcm_r.items():
                    rep_pc[v_name][name].setdefault(c, []).append(m)
                pbar.set_postfix_str(f"r={r+1}/{n_repeats} {v_name} {name}")
                pbar.update(1)
    pbar.close()
    for v_name in variants_xy:
        for name in CLASSIFIERS:
            results[v_name][name]['means'].append(float(np.mean(rep_means[v_name][name])))
            results[v_name][name]['stds'].append(float(np.std(rep_means[v_name][name])))
            for c, vals in rep_pc[v_name][name].items():
                if c not in results[v_name][name]['per_class']:
                    results[v_name][name]['per_class'][c] = {'means': [], 'stds': []}
                results[v_name][name]['per_class'][c]['means'].append(float(np.mean(vals)))
                results[v_name][name]['per_class'][c]['stds'].append(float(np.std(vals)))

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
        'n_merged': int(X_m_nokin.shape[0]),
        'n_dropped_task_no_tracking': n_dropped_t,
        'bin_elapsed_sec': float(bin_elapsed),
    })

print(f"\nTotal CV sweep wall time: {(time.time() - t_sweep_start)/60:.2f} min")

clf_names   = list(CLASSIFIERS.keys())
all_classes = sorted(results['no_kin'][clf_names[0]]['per_class'].keys())


# ------------------------------------------------------------------ #
#  Step 4 - kinematics-helped summary (Δ = with_kin - no_kin)         #
# ------------------------------------------------------------------ #
def _delta(clf_name, key):
    if key == 'overall':
        a = np.array(results['with_kin'][clf_name]['means'])
        b = np.array(results['no_kin'][clf_name]['means'])
    else:
        a = np.array(results['with_kin'][clf_name]['per_class'][key]['means'])
        b = np.array(results['no_kin'][clf_name]['per_class'][key]['means'])
    return a - b


print("\n" + "=" * 72)
print("Δ accuracy (with_kin − no_kin), averaged across bin sizes:")
print("=" * 72)
print(f"{'classifier':<22}{'overall':>10}" +
      "".join(f"{'rec ' + str(c):>10}" for c in all_classes))
for clf_name in clf_names:
    row = [f"{_delta(clf_name, 'overall').mean():+.4f}"]
    row += [f"{_delta(clf_name, c).mean():+.4f}" for c in all_classes]
    print(f"{clf_name:<22}" + "".join(f"{v:>10}" for v in row))


# ------------------------------------------------------------------ #
#  Step 5 - figure: 7 rows × (1 + n_classes) cols                     #
# ------------------------------------------------------------------ #
def _fmt_class(d):
    return ', '.join(f'{k}={v:g}' for k, v in d.items())

CLASS_LABELS = {
    -1: f'stim-on  left={_fmt_class(class_neg)} (-1)',
     0: 'ITI (0)',
     1: f'stim-on  left={_fmt_class(class_pos)} (+1)',
}
VARIANT_STYLE = {
    'no_kin':   {'color': '#009E73', 'marker': '^', 'label': 'Merged (spikes only)'},
    'with_kin': {'color': '#CC79A7', 'marker': 'D', 'label': 'Merged + kinematics'},
}

n_rows = len(clf_names)
n_cols = 1 + len(all_classes)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.2 * n_rows),
                         sharex=True)
if n_rows == 1:
    axes = axes[np.newaxis, :]


def _plot_two(ax, clf_name, y_key, title):
    for v in VARIANTS:
        res = results[v][clf_name]
        if y_key == 'overall':
            means = np.array(res['means']); stds = np.array(res['stds'])
        else:
            means = np.array(res['per_class'][y_key]['means'])
            stds  = np.array(res['per_class'][y_key]['stds'])
        style = VARIANT_STYLE[v]
        ax.errorbar(bin_sizes_ms, means, yerr=stds,
                    marker=style['marker'], color=style['color'],
                    linewidth=1.6, capsize=3, label=style['label'])
    # annotate Δ at each bin
    d = _delta(clf_name, y_key)
    for x, dv in zip(bin_sizes_ms, d):
        ax.annotate(f"{dv:+.02f}", xy=(x, 0.03), ha='center', fontsize=6,
                    color=('#444' if dv >= 0 else '#aa0000'))
    ax.axhline(0.5, color='0.7', linestyle=':', linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms], fontsize=8)
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.set_ylim(0.0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)


for row, clf_name in enumerate(clf_names):
    _plot_two(axes[row, 0], clf_name, 'overall', f'{clf_name} — overall acc')
    axes[row, 0].set_ylabel(f'{clf_name}\nAccuracy', fontsize=10)
    if row == 0:
        axes[row, 0].legend(fontsize=8, loc='lower right')
    for col_i, c in enumerate(all_classes, start=1):
        _plot_two(axes[row, col_i], clf_name, c,
                  f'{clf_name} — recall  {CLASS_LABELS.get(c, c)}')
        if col_i == 1 and row == 0:
            axes[row, col_i].legend(fontsize=7, loc='lower right')

for ax in axes[-1, :]:
    ax.set_xlabel('Bin size (ms)', fontsize=10)

def _stem_part(d):
    return '_'.join(f'{k}{v:g}' for k, v in d.items())

stem = (f"merged_with_vs_without_kinematics_{session}_pos-{_stem_part(class_pos)}_"
        f"neg-{_stem_part(class_neg)}_{n_splits}fold")

fig.suptitle(
    f'Merged decoder: spikes-only vs spikes+kinematics  |  {session}  |  '
    f'+1: {_fmt_class(class_pos)}  vs  -1: {_fmt_class(class_neg)}  |  '
    f'common units={n_units_common}  |  {n_splits}-fold CV  |  '
    f'balanced train / natural-ratio test  |  '
    f'kin = [speed, sin/cos heading, sin/cos head_angle], zero-filled for passive  |  '
    f'Δ shown at the bottom of each panel',
    fontsize=11, y=1.00,
)
plt.tight_layout(rect=[0, 0, 1, 0.985])

fig_path = out_dir / f"{stem}.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {fig_path}")

pkl_out = out_dir / f"{stem}.pkl"
with open(pkl_out, 'wb') as f:
    pickle.dump({
        'bin_sizes_ms':        bin_sizes_ms,
        'results':             results,
        'n_splits':            n_splits,
        'n_repeats':           n_repeats,
        'class_pos':            class_pos,
        'class_neg':            class_neg,
        'task_col_map':         TASK_COL_MAP,
        'passive_col_map':      PASSIVE_COL_MAP,
        'rewarded_combination': rewarded_combination,
        'n_units_common':      n_units_common,
        'sweep_stats':         sweep_stats,
        'feature_columns_added': KINEMATIC_FEATURE_COLUMNS,
        'kinematics_zero_fill_for_passive': True,
        'clean_position_stats': clean_stats,
        'session_to_window_offset_sec': offset,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved → {pkl_out}")

plt.show()
