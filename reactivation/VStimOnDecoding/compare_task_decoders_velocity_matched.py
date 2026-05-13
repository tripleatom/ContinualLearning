"""
Velocity-matched twin of compare_task_reward_decoders.py.

Stim-on epochs and ITI epochs typically differ in running speed, so a "stim-on
vs ITI" decoder may really be picking up a running-vs-stationary signal. This
script controls for that: ITI bins are sub-sampled so their per-bin speed
histogram matches the stim-on speed histogram, and the full classifier x
bin-size sweep is then run on the matched data. The unmatched baseline is
re-run in the same loop (same seed, same usable bins) for an apples-to-apples
overlay.

Velocity is computed from the session DLC track stored in
data['session_position'] of task_spikes_*.pkl, using the same cleaning pipeline
as DiscriminationTask/grating/plot_trial_traces.py (zero-pad drop,
Hampel flicker filter, Savitzky-Golay smoothed central differences, p99 cap).

Outputs (under <pkl_folder>/reactivation/):
  task_<session>_velocity_distributions.png
  task_<session>_velocity_matched_vs_unmatched_<n>fold.png
  task_<session>_velocity_matched_bar<bms>ms.png
  task_<session>_velocity_matched_<n>fold.pkl
"""

import sys
import pickle
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'DiscriminationTask' / 'grating'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from decode_utils import make_classifiers, run_cv_balanced_train, run_cv_balanced_train_shuffle
from kinematics_utils import (
    SPEED_ONLY_COLUMNS,
    build_task_kinematics,
    load_task_kinematic_samples,
    print_kinematic_sample_report,
)
from prepare_task_stimtype import prepare_task_stim_type, infer_rewarded_combination

from params import (
    task_pkl as pkl_file,
    bin_sizes_ms, n_splits, random_state,
    speed_bin_cms, speed_top_pct, hist_bin_ms, n_repeats, n_bootstrap,
    class_pos, class_neg, TASK_COL_MAP,
)

out_dir = Path(pkl_file).parent / 'reactivation'
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {out_dir}")

session = Path(pkl_file).parent.name


# ------------------------------------------------------------------ #
#  Step 1 — load DLC, build per-sample speed in window frame          #
# ------------------------------------------------------------------ #
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
#  Step 2 — per-bin mean speed (window-frame aligned)                 #
# ------------------------------------------------------------------ #
def bin_speeds(bin_centers, bin_size_sec):
    """Mean cleaned speed (cm/s) per bin; NaN for bins with no DLC sample.

    Uses kin_samples loaded above (window-frame DLC times), so per-bin
    assignment is correct with respect to bin_centers from prepare_task_stim_type.
    """
    extras, _ = build_task_kinematics(bin_centers, bin_size_sec, kin_samples,
                                       columns=SPEED_ONLY_COLUMNS)
    return extras[:, 0]


# ------------------------------------------------------------------ #
#  Step 3 — ITI subsampling (matched / unmatched, equal training N)   #
# ------------------------------------------------------------------ #
def _speed_edges_from_stim(s_stim, bin_width_cms, top_percentile):
    """Common bin edges used by both matched and unmatched paths."""
    top_edge = float(np.percentile(s_stim, top_percentile)) + bin_width_cms
    return np.arange(0.0, top_edge + bin_width_cms, bin_width_cms)


def match_iti_to_stimon(X, y, speeds, rng, n_target,
                        bin_width_cms=speed_bin_cms,
                        top_percentile=speed_top_pct):
    """
    Sub-sample ITI bins (y == 0) to exactly `n_target` samples whose per-bin
    speed histogram best matches the combined stim-on (y != 0) histogram.

    Water-fill allocation: each speed bin's quota is `n_target * p_stim[b]`,
    capped by the ITI samples available in that bin. Deficit from bins that
    run dry is redistributed across the still-supplied bins in proportion to
    the stim-on mass, iterated until n_target is reached or no ITI remains.
    If best-effort matching can't reach `n_target` (e.g. ITI is sparser than
    stim-on at some speeds), the remaining slots are filled uniformly from
    any leftover ITI samples so the training size matches the unmatched arm.

    Returns (X_matched, y_matched, kept_speeds, target_per_bin, edges).
    """
    stim_mask = y != 0
    iti_mask  = y == 0
    s_stim = speeds[stim_mask]
    s_iti  = speeds[iti_mask]

    edges = _speed_edges_from_stim(s_stim, bin_width_cms, top_percentile)
    n_bins = len(edges) - 1

    iti_idx_all = np.where(iti_mask)[0]
    iti_bin_idx = np.digitize(s_iti, edges) - 1
    in_range = (iti_bin_idx >= 0) & (iti_bin_idx < n_bins)
    iti_idx_in = iti_idx_all[in_range]
    iti_bin_in = iti_bin_idx[in_range]

    n_stim, _    = np.histogram(s_stim, bins=edges)
    n_iti_in_bin = np.bincount(iti_bin_in, minlength=n_bins)

    target    = np.zeros(n_bins, dtype=int)
    available = n_iti_in_bin.copy()
    weights   = n_stim.astype(float)
    remaining = int(min(n_target, iti_idx_all.size))

    while remaining > 0 and available.sum() > 0:
        wsum = weights.sum()
        if wsum <= 0:
            break
        alloc = np.floor(remaining * weights / wsum).astype(int)
        alloc = np.minimum(alloc, available)
        if alloc.sum() == 0:
            # Round-up step so we don't stall on small remaining budgets.
            order = np.argsort(-weights * (available > 0))
            for b in order:
                if available[b] > 0:
                    alloc[b] = 1
                    break
        if alloc.sum() == 0:
            break
        target    += alloc
        available -= alloc
        remaining -= int(alloc.sum())
        weights = np.where(available > 0, weights, 0.0)

    # Materialize the picks.
    keep_parts = []
    for b in range(n_bins):
        if target[b] == 0:
            continue
        candidates = iti_idx_in[iti_bin_in == b]
        keep_parts.append(rng.choice(candidates, size=int(target[b]), replace=False))
    keep_iti = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=int)

    # If we still have a deficit (in-range ITI exhausted or speed-out-of-range),
    # top up uniformly from any unused ITI so n_iti_kept == n_target.
    deficit = int(min(n_target, iti_idx_all.size)) - keep_iti.size
    if deficit > 0:
        leftover = np.setdiff1d(iti_idx_all, keep_iti, assume_unique=False)
        if leftover.size:
            take = min(deficit, leftover.size)
            keep_iti = np.concatenate([keep_iti,
                                       rng.choice(leftover, size=take, replace=False)])

    sel = np.sort(np.concatenate([np.where(stim_mask)[0], keep_iti]))
    return X[sel], y[sel], speeds[sel], target, edges


def downsample_iti_uniform(X, y, speeds, rng, n_target):
    """Uniformly random sub-sample of ITI to `n_target`. Returns same shape as match_iti_to_stimon (no per-bin target)."""
    iti_idx  = np.where(y == 0)[0]
    stim_idx = np.where(y != 0)[0]
    take = int(min(n_target, iti_idx.size))
    if take == iti_idx.size:
        kept = iti_idx
    else:
        kept = rng.choice(iti_idx, size=take, replace=False)
    sel = np.sort(np.concatenate([stim_idx, kept]))
    return X[sel], y[sel], speeds[sel]


def iti_target_count(y):
    """ITI target = min stim-on class count, so balanced training trains on equal N per class."""
    return int(min((y == 1).sum(), (y == -1).sum()))


# ------------------------------------------------------------------ #
#  Step 4 — velocity-distribution figure (at hist_bin_ms)             #
# ------------------------------------------------------------------ #
print(f"\nBuilding velocity-distribution figure at {hist_bin_ms} ms bins ...")
X_h, y_h, bc_h, _ = prepare_task_stim_type(pkl_file, class_pos, class_neg, TASK_COL_MAP, random_state=random_state, bin_size_sec=hist_bin_ms / 1000.0)
sp_h = bin_speeds(bc_h, hist_bin_ms / 1000.0)
keep_h = ~np.isnan(sp_h)
X_h, y_h, sp_h = X_h[keep_h], y_h[keep_h], sp_h[keep_h]

n_target_h = iti_target_count(y_h)
rng_hist_m = np.random.default_rng(random_state)
rng_hist_u = np.random.default_rng(random_state + 1)
X_h_m, y_h_m, sp_h_m, target_h, edges_h = match_iti_to_stimon(
    X_h, y_h, sp_h, rng_hist_m, n_target_h,
)
X_h_u, y_h_u, sp_h_u = downsample_iti_uniform(X_h, y_h, sp_h, rng_hist_u, n_target_h)

ks_stat_m, ks_p_m = ks_2samp(sp_h[(y_h != 0)], sp_h_m[y_h_m == 0])
ks_stat_u, ks_p_u = ks_2samp(sp_h[(y_h != 0)], sp_h_u[y_h_u == 0])
print(f"  n target ITI (= min stim-on class) = {n_target_h}")
print(f"  KS(stim-on vs matched   ITI) @ {hist_bin_ms} ms: stat={ks_stat_m:.3f}  p={ks_p_m:.2e}")
print(f"  KS(stim-on vs unmatched ITI) @ {hist_bin_ms} ms: stat={ks_stat_u:.3f}  p={ks_p_u:.2e}")
print(f"  n stim-on={int((y_h != 0).sum())}  n raw ITI={int((y_h == 0).sum())}  "
      f"n matched ITI={int((y_h_m == 0).sum())}  n unmatched ITI={int((y_h_u == 0).sum())}")

fig0, axes0 = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True, sharey=False)
mids = 0.5 * (edges_h[:-1] + edges_h[1:])
width = edges_h[1] - edges_h[0]

# Panel A: stim-on +1 / -1
ax = axes0[0]
n_pos, _ = np.histogram(sp_h[y_h == 1],  bins=edges_h)
n_neg, _ = np.histogram(sp_h[y_h == -1], bins=edges_h)
ax.bar(mids, n_pos, width=width * 0.45, align='edge', color='#2c7fb8',
       label=f'+1 (n={int((y_h == 1).sum())})')
ax.bar(mids - width * 0.45, n_neg, width=width * 0.45, align='edge', color='#d95f0e',
       label=f'-1 (n={int((y_h == -1).sum())})')
ax.set_title(f'Stim-on speed  (bin={hist_bin_ms} ms)')
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Bin count')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# Panel B: unmatched ITI (uniform downsample to n_target) + stim-on overlay
ax = axes0[1]
n_iti_raw, _ = np.histogram(sp_h[y_h == 0],       bins=edges_h)
n_iti_u,   _ = np.histogram(sp_h_u[y_h_u == 0],   bins=edges_h)
n_stim_h,  _ = np.histogram(sp_h[y_h != 0],       bins=edges_h)
ax.bar(mids, n_iti_u, width=width * 0.9, color='#7f7f7f',
       label=f'ITI unmatched, uniform DS (n={int(n_iti_u.sum())})')
stim_scaled_u = n_stim_h * (n_iti_u.sum() / max(n_stim_h.sum(), 1))
ax.step(edges_h[:-1], stim_scaled_u, where='post', color='black', linewidth=1.4,
        label='stim-on (rescaled)')
ax.set_title(f'ITI speed — unmatched  (KS={ks_stat_u:.3f})')
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Bin count')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# Panel C: matched ITI (same size as B) + stim-on overlay
ax = axes0[2]
n_iti_m, _ = np.histogram(sp_h_m[y_h_m == 0], bins=edges_h)
ax.bar(mids, n_iti_m, width=width * 0.9, color='#41ab5d',
       label=f'ITI velocity-matched (n={int(n_iti_m.sum())})')
stim_scaled_m = n_stim_h * (n_iti_m.sum() / max(n_stim_h.sum(), 1))
ax.step(edges_h[:-1], stim_scaled_m, where='post', color='black', linewidth=1.4,
        label='stim-on (rescaled)')
ax.set_title(f'ITI speed — matched  (KS={ks_stat_m:.3f})')
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Bin count')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

fig0.suptitle(
    f"Per-bin running speed: stim-on vs ITI (unmatched, uniform DS) vs ITI (velocity-matched)\n"
    f"Session: {session}  |  Speed bin = {speed_bin_cms} cm/s  |  Upper edge = p{speed_top_pct}(stim-on)  |  "
    f"n_target ITI = min stim-on class = {n_target_h}",
    fontsize=11,
)
fig0.tight_layout(rect=[0, 0, 1, 0.93])
fig0_path = out_dir / f"task_{session}_velocity_distributions.png"
fig0.savefig(fig0_path, dpi=150, bbox_inches='tight')
print(f"Velocity-distribution figure saved → {fig0_path}")


# ------------------------------------------------------------------ #
#  Step 5 — sweep bin sizes: matched + unmatched, same seed/folds     #
# ------------------------------------------------------------------ #
CLASSIFIERS = make_classifiers(random_state)
clf_names_seed = list(CLASSIFIERS.keys())
n_bins_sweep   = len(bin_sizes_ms)

# Per-repeat storage: [bin_idx] -> list of n_repeats overall accuracies.
per_repeat_matched   = {n: [[] for _ in range(n_bins_sweep)] for n in clf_names_seed}
per_repeat_unmatched = {n: [[] for _ in range(n_bins_sweep)] for n in clf_names_seed}
per_repeat_pc_matched   = {n: {} for n in clf_names_seed}   # name -> class -> [bin_idx] -> list
per_repeat_pc_unmatched = {n: {} for n in clf_names_seed}

results_matched   = {n: {'means': [], 'stds': [], 'per_class': {}} for n in clf_names_seed}
results_unmatched = {n: {'means': [], 'stds': [], 'per_class': {}} for n in clf_names_seed}
chance_matched, chance_unmatched = [], []
sweep_stats = []

for bms_idx, bms in enumerate(bin_sizes_ms):
    print(f"\n{'=' * 58}")
    print(f"  Bin size: {bms} ms  (n_repeats={n_repeats})")
    print(f"{'=' * 58}")

    X, y, bc, _ = prepare_task_stim_type(pkl_file, class_pos, class_neg, TASK_COL_MAP, random_state=random_state, bin_size_sec=bms / 1000.0)
    sp_b = bin_speeds(bc, bms / 1000.0)
    keep = ~np.isnan(sp_b)
    n_dropped = int((~keep).sum())
    X, y, sp_b = X[keep], y[keep], sp_b[keep]

    n_target = iti_target_count(y)
    n_stim  = int((y != 0).sum())
    n_iti_r = int((y == 0).sum())
    print(f"  n_total_bins={len(y)}  n_dropped_no_tracking={n_dropped}  "
          f"n_stim={n_stim}  n_iti_raw={n_iti_r}  n_target={n_target}")

    for r in range(n_repeats):
        rng_m = np.random.default_rng(random_state + 1000 * bms + 2 * r)
        rng_u = np.random.default_rng(random_state + 1000 * bms + 2 * r + 1)
        X_m, y_m, _, _, _ = match_iti_to_stimon(X, y, sp_b, rng_m, n_target)
        X_u, y_u, _      = downsample_iti_uniform(X, y, sp_b, rng_u, n_target)

        for name, clf_proto in CLASSIFIERS.items():
            _, mean_m, ch_m, pcm_m, _ = run_cv_balanced_train(
                name, clf_proto, X_m, y_m,
                n_splits=n_splits, random_state=random_state,
            )
            _, mean_u, ch_u, pcm_u, _ = run_cv_balanced_train(
                name, clf_proto, X_u, y_u,
                n_splits=n_splits, random_state=random_state,
            )
            per_repeat_matched[name][bms_idx].append(mean_m)
            per_repeat_unmatched[name][bms_idx].append(mean_u)
            for c, m in pcm_m.items():
                per_repeat_pc_matched[name].setdefault(
                    c, [[] for _ in range(n_bins_sweep)])[bms_idx].append(m)
            for c, m in pcm_u.items():
                per_repeat_pc_unmatched[name].setdefault(
                    c, [[] for _ in range(n_bins_sweep)])[bms_idx].append(m)

    # Per-bin summary print after all repeats are done.
    for name in clf_names_seed:
        m_mu = float(np.mean(per_repeat_matched[name][bms_idx]))
        m_sd = float(np.std(per_repeat_matched[name][bms_idx]))
        u_mu = float(np.mean(per_repeat_unmatched[name][bms_idx]))
        u_sd = float(np.std(per_repeat_unmatched[name][bms_idx]))
        print(f"  [{name}]  matched={m_mu:.3f}±{m_sd:.3f}  "
              f"unmatched={u_mu:.3f}±{u_sd:.3f}  "
              f"Δ={u_mu - m_mu:+.3f}")

    chance_matched.append(ch_m)
    chance_unmatched.append(ch_u)
    sweep_stats.append({
        'bin_ms': bms, 'n_total': len(y), 'n_dropped': n_dropped,
        'n_stim': n_stim, 'n_iti_raw': n_iti_r, 'n_target': n_target,
    })

# Build results_* (mean and std across repeats) from per_repeat_*
for name in clf_names_seed:
    for bms_idx in range(n_bins_sweep):
        accs_m = np.array(per_repeat_matched[name][bms_idx])
        accs_u = np.array(per_repeat_unmatched[name][bms_idx])
        results_matched[name]['means'].append(float(np.mean(accs_m)))
        results_matched[name]['stds'].append(float(np.std(accs_m)))
        results_unmatched[name]['means'].append(float(np.mean(accs_u)))
        results_unmatched[name]['stds'].append(float(np.std(accs_u)))
    for c, per_bin_lists in per_repeat_pc_matched[name].items():
        results_matched[name]['per_class'].setdefault(c, {'means': [], 'stds': []})
        for vals in per_bin_lists:
            results_matched[name]['per_class'][c]['means'].append(float(np.nanmean(vals)))
            results_matched[name]['per_class'][c]['stds'].append(float(np.nanstd(vals)))
    for c, per_bin_lists in per_repeat_pc_unmatched[name].items():
        results_unmatched[name]['per_class'].setdefault(c, {'means': [], 'stds': []})
        for vals in per_bin_lists:
            results_unmatched[name]['per_class'][c]['means'].append(float(np.nanmean(vals)))
            results_unmatched[name]['per_class'][c]['stds'].append(float(np.nanstd(vals)))

mean_chance_matched   = float(np.mean(chance_matched))
mean_chance_unmatched = float(np.mean(chance_unmatched))
all_classes = sorted(results_matched[next(iter(results_matched))]['per_class'].keys())


# ------------------------------------------------------------------ #
#  Step 6 — accuracy comparison figure                                #
# ------------------------------------------------------------------ #
def _plot_overlay(ax, results_m, results_u, bin_sizes_ms, key, title):
    """Solid = velocity-matched, dashed = unmatched. One color per classifier."""
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for (name, res_m), marker in zip(results_m.items(), markers):
        res_u = results_u[name]
        if key == 'overall':
            mm, sm = np.array(res_m['means']), np.array(res_m['stds'])
            mu     = np.array(res_u['means'])
        else:
            mm = np.array(res_m['per_class'][key]['means'])
            sm = np.array(res_m['per_class'][key]['stds'])
            mu = np.array(res_u['per_class'][key]['means'])
        line, = ax.plot(bin_sizes_ms, mm, marker=marker, linewidth=1.6, label=name)
        ax.fill_between(bin_sizes_ms, mm - sm, mm + sm,
                        color=line.get_color(), alpha=0.12, linewidth=0)
        ax.plot(bin_sizes_ms, mu, marker=marker, linestyle='--',
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

_plot_overlay(ax_top, results_matched, results_unmatched, bin_sizes_ms,
              'overall',
              'Overall accuracy — solid: ITI velocity-matched, dashed: uniform-DS (same N)')
_style_handles = [
    plt.Line2D([], [], color='black', linestyle='-',  linewidth=1.6, label='matched'),
    plt.Line2D([], [], color='black', linestyle='--', linewidth=1.1, label='unmatched'),
]
clf_handles = ax_top.get_legend_handles_labels()[0]
clf_labels  = ax_top.get_legend_handles_labels()[1]
ax_top.legend(clf_handles + _style_handles,
              clf_labels  + ['matched', 'unmatched'],
              fontsize=8, loc='lower right', ncol=3)

for ax, c in zip(axes_bot, all_classes):
    _plot_overlay(ax, results_matched, results_unmatched, bin_sizes_ms, c,
                  CLASS_LABELS.get(c, f'Class {c}'))
    ax.legend(fontsize=8, loc='lower right', ncol=2)

plt.tight_layout()

info = (
    f"Data: {Path(pkl_file).name}  |  "
    f"Bins: {bin_sizes_ms[0]}–{bin_sizes_ms[-1]} ms  |  "
    f"CV: {n_splits}-fold  |  "
    f"Train: balanced (undersampled), Test: natural ratio  |  "
    f"Speed bin: {speed_bin_cms} cm/s up to p{speed_top_pct}(stim-on)  |  "
    f"Chance matched={mean_chance_matched:.2f}  unmatched={mean_chance_unmatched:.2f}  |  "
    f"Seed: {random_state}"
)
fig.text(0.5, -0.01, info, ha='center', va='top', fontsize=7.5, color='gray')

stem = f"task_{session}_velocity_matched_vs_unmatched_{n_splits}fold"
fig_path = out_dir / f"{stem}.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nComparison figure saved → {fig_path}")


# ------------------------------------------------------------------ #
#  Step 6b — paired bootstrap of Δ = unmatched − matched              #
# ------------------------------------------------------------------ #
print(f"\nBootstrapping paired Δ = unmatched − matched "
      f"(n_repeats={n_repeats}, n_bootstrap={n_bootstrap}) ...")
rng_boot = np.random.default_rng(random_state + 7777)
delta_summary = {n: {'mean': [], 'ci_lo': [], 'ci_hi': [], 'p_two_sided': []}
                 for n in clf_names_seed}

for name in clf_names_seed:
    for i in range(n_bins_sweep):
        accs_m = np.array(per_repeat_matched[name][i])
        accs_u = np.array(per_repeat_unmatched[name][i])
        deltas = accs_u - accs_m
        # Bootstrap mean Δ
        idx = rng_boot.integers(0, deltas.size, size=(n_bootstrap, deltas.size))
        bs_means = deltas[idx].mean(axis=1)
        # Two-sided p: prob that a re-centered bootstrap mean is at least as
        # extreme as the observed mean (achieved significance level).
        observed = float(np.mean(deltas))
        centered = bs_means - observed
        p_two = float(np.mean(np.abs(centered) >= abs(observed)))
        delta_summary[name]['mean'].append(observed)
        delta_summary[name]['ci_lo'].append(float(np.percentile(bs_means, 2.5)))
        delta_summary[name]['ci_hi'].append(float(np.percentile(bs_means, 97.5)))
        delta_summary[name]['p_two_sided'].append(p_two)


# ------------------------------------------------------------------ #
#  Figure: velocity-driven accuracy gap (Δ ± 95% CI) per classifier   #
# ------------------------------------------------------------------ #
n_class_panels = len(all_classes)
fig_d = plt.figure(figsize=(6 * (n_class_panels + 1), 9))
ax_d_top   = fig_d.add_subplot(2, 1, 1)
axes_d_bot = [fig_d.add_subplot(2, n_class_panels, n_class_panels + i + 1)
              for i in range(n_class_panels)]


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
    ax.set_ylabel('Δ accuracy  (unmatched − matched)', fontsize=10)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms])
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.spines[['top', 'right']].set_visible(False)


# Per-class deltas (built on the fly from per_repeat_pc_*)
delta_summary_pc = {c: {n: {'mean': [], 'ci_lo': [], 'ci_hi': [], 'p_two_sided': []}
                        for n in clf_names_seed} for c in all_classes}
for c in all_classes:
    for name in clf_names_seed:
        for i in range(n_bins_sweep):
            am = np.array(per_repeat_pc_matched[name].get(c, [[]] * n_bins_sweep)[i])
            au = np.array(per_repeat_pc_unmatched[name].get(c, [[]] * n_bins_sweep)[i])
            if am.size == 0 or au.size == 0:
                delta_summary_pc[c][name]['mean'].append(np.nan)
                delta_summary_pc[c][name]['ci_lo'].append(np.nan)
                delta_summary_pc[c][name]['ci_hi'].append(np.nan)
                delta_summary_pc[c][name]['p_two_sided'].append(np.nan)
                continue
            d = au - am
            idx = rng_boot.integers(0, d.size, size=(n_bootstrap, d.size))
            bs = d[idx].mean(axis=1)
            observed = float(np.nanmean(d))
            centered = bs - observed
            p_two = float(np.mean(np.abs(centered) >= abs(observed)))
            delta_summary_pc[c][name]['mean'].append(observed)
            delta_summary_pc[c][name]['ci_lo'].append(float(np.nanpercentile(bs, 2.5)))
            delta_summary_pc[c][name]['ci_hi'].append(float(np.nanpercentile(bs, 97.5)))
            delta_summary_pc[c][name]['p_two_sided'].append(p_two)


_plot_delta(ax_d_top, delta_summary,
            'Overall Δ accuracy: velocity-driven gap (paired, 95% CI bootstrap)')
ax_d_top.legend(fontsize=8, loc='upper right', ncol=2)
for ax, c in zip(axes_d_bot, all_classes):
    _plot_delta(ax, delta_summary_pc[c],
                f"{CLASS_LABELS.get(c, f'Class {c}')}  —  Δ recall")
    ax.legend(fontsize=7, loc='upper right', ncol=2)

info_d = (
    f"n_repeats={n_repeats}  |  n_bootstrap={n_bootstrap}  |  "
    f"Bands = 95% paired-bootstrap CI of mean Δ  |  "
    f"Δ > 0 ⇒ velocity contributed; CI excluding 0 ⇒ effect is significant"
)
fig_d.text(0.5, -0.01, info_d, ha='center', va='top', fontsize=8, color='gray')
plt.tight_layout()
fig_d_path = out_dir / f"task_{session}_velocity_effect_delta_{n_splits}fold.png"
fig_d.savefig(fig_d_path, dpi=150, bbox_inches='tight')
print(f"Δ figure saved → {fig_d_path}")


# ------------------------------------------------------------------ #
#  Step 7 — best bin (matched), shuffle null, bar plot                #
# ------------------------------------------------------------------ #
clf_names = list(CLASSIFIERS.keys())
mean_acc_per_bin = np.mean(
    [[results_matched[n]['means'][i] for n in clf_names] for i in range(len(bin_sizes_ms))],
    axis=1,
)
bar_idx    = int(np.argmax(mean_acc_per_bin))
bar_bin_ms = bin_sizes_ms[bar_idx]
print(f"\nBest matched bin size (argmax mean acc): {bar_bin_ms} ms  "
      f"(mean acc = {mean_acc_per_bin[bar_idx]:.3f})")

X_bar, y_bar, bc_bar, _ = prepare_task_stim_type(pkl_file, class_pos, class_neg, TASK_COL_MAP, random_state=random_state, bin_size_sec=bar_bin_ms / 1000.0)
sp_bar = bin_speeds(bc_bar, bar_bin_ms / 1000.0)
keep_bar = ~np.isnan(sp_bar)
X_bar, y_bar, sp_bar = X_bar[keep_bar], y_bar[keep_bar], sp_bar[keep_bar]
n_target_bar = iti_target_count(y_bar)
X_bar_m, y_bar_m, _, _, _ = match_iti_to_stimon(
    X_bar, y_bar, sp_bar,
    np.random.default_rng(random_state + bar_bin_ms), n_target_bar,
)

print(f"Computing shuffle null on matched data at {bar_bin_ms} ms ...")
shuffle_results = {}
for name, clf_proto in CLASSIFIERS.items():
    sm, ss, spc_m, spc_s = run_cv_balanced_train_shuffle(
        name, clf_proto, X_bar_m, y_bar_m,
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
bar_w    = width * 0.45
offset   = width * 0.25
color_bar = '#4C72B0'
color_shuf = '#AAAAAA'

fig2      = plt.figure(figsize=(max(8, n_clf * 1.6) * max(n_panels, 1), 10))
ax2_top   = fig2.add_subplot(2, 1, 1)
axes2_bot = [fig2.add_subplot(2, n_panels, n_panels + i + 1) for i in range(n_panels)]


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
    [results_matched[n]['means'][bar_idx] for n in clf_names],
    [results_matched[n]['stds'][bar_idx]  for n in clf_names],
    [shuffle_results[n]['mean']           for n in clf_names],
    [shuffle_results[n]['std']            for n in clf_names],
    f'Overall accuracy — velocity-matched  (bin = {bar_bin_ms} ms)',
    'Accuracy',
)

for ax, c in zip(axes2_bot, all_classes):
    _bar_single(
        ax,
        [results_matched[n]['per_class'][c]['means'][bar_idx] for n in clf_names],
        [results_matched[n]['per_class'][c]['stds'][bar_idx]  for n in clf_names],
        [shuffle_results[n]['per_class'][c]['mean']           for n in clf_names],
        [shuffle_results[n]['per_class'][c]['std']            for n in clf_names],
        f'{CLASS_LABELS.get(c, f"Class {c}")}  (bin = {bar_bin_ms} ms)',
        'Recall',
    )

fig2.suptitle(
    f'Per-classifier decoding (velocity-matched ITI)  |  bin = {bar_bin_ms} ms, {n_splits}-fold CV\n'
    f'Balanced train / natural-ratio test',
    fontsize=11, y=1.01,
)
plt.tight_layout()
fig2_path = out_dir / f"{stem}_bar{bar_bin_ms}ms.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Bar-plot figure saved → {fig2_path}")


# ------------------------------------------------------------------ #
#  Step 7b — z relative to shuffle null at the best bin               #
# ------------------------------------------------------------------ #
print(f"\n--- z relative to shuffle null at bin = {bar_bin_ms} ms ---")
print(f"{'Classifier':>14}  {'acc_match':>9}  {'acc_unmatch':>11}  "
      f"{'shuf_mean':>9}  {'z_match':>8}  {'z_unmatch':>9}  {'Δ_match−unmatch':>16}  {'p_Δ':>7}")
z_summary = {}
for name in clf_names:
    sm_mu = float(shuffle_results[name]['mean'])
    sm_sd = float(max(shuffle_results[name]['std'], 1e-9))
    am = float(results_matched[name]['means'][bar_idx])
    au = float(results_unmatched[name]['means'][bar_idx])
    zm = (am - sm_mu) / sm_sd
    zu = (au - sm_mu) / sm_sd
    d_mu = float(delta_summary[name]['mean'][bar_idx])
    p_d  = float(delta_summary[name]['p_two_sided'][bar_idx])
    z_summary[name] = {
        'acc_matched':   am, 'acc_unmatched': au,
        'shuffle_mean':  sm_mu, 'shuffle_std':  sm_sd,
        'z_matched':     zm, 'z_unmatched':   zu,
        'delta_mean':    d_mu, 'delta_p':      p_d,
    }
    print(f"{name:>14}  {am:>9.3f}  {au:>11.3f}  {sm_mu:>9.3f}  "
          f"{zm:>+8.2f}  {zu:>+9.2f}  {d_mu:>+16.3f}  {p_d:>7.4f}")
print("Interpretation: z_matched ≈ 0 ⇒ no signal beyond velocity. "
      "Δ CI excluding 0 (p_Δ small) ⇒ velocity matters.")


# ------------------------------------------------------------------ #
#  Step 8 — save results pkl                                          #
# ------------------------------------------------------------------ #
pkl_out = out_dir / f"task_{session}_velocity_matched_{n_splits}fold.pkl"
with open(pkl_out, 'wb') as f:
    pickle.dump({
        'bin_sizes_ms':           bin_sizes_ms,
        'results_matched':        results_matched,
        'results_unmatched':      results_unmatched,
        'shuffle_results':        shuffle_results,
        'chance_matched':         mean_chance_matched,
        'chance_unmatched':       mean_chance_unmatched,
        'n_splits':               n_splits,
        'speed_bin_cms':          speed_bin_cms,
        'speed_top_percentile':   speed_top_pct,
        'hist_bin_ms':            hist_bin_ms,
        'speed_edges_cms_hist':     edges_h,
        'stimon_speed_hist':        n_stim_h,
        'iti_raw_speed_hist':       n_iti_raw,
        'iti_unmatched_speed_hist': n_iti_u,
        'iti_matched_speed_hist':   n_iti_m,
        'matched_target_per_bin':   target_h,
        'n_target_hist':            n_target_h,
        'ks_matched':               ks_stat_m,
        'ks_unmatched':             ks_stat_u,
        'sweep_stats':            sweep_stats,
        'best_bin_ms':              bar_bin_ms,
        'clean_position_stats':     clean_stats,
        'session_to_window_offset_sec': offset,
        'n_repeats':                n_repeats,
        'n_bootstrap':              n_bootstrap,
        'class_pos':                class_pos,
        'class_neg':                class_neg,
        'col_map':                  TASK_COL_MAP,
        'rewarded_combination':     rewarded_combination,
        'per_repeat_matched':       per_repeat_matched,
        'per_repeat_unmatched':     per_repeat_unmatched,
        'per_repeat_pc_matched':    per_repeat_pc_matched,
        'per_repeat_pc_unmatched':  per_repeat_pc_unmatched,
        'delta_summary':            delta_summary,
        'delta_summary_per_class':  delta_summary_pc,
        'z_summary_at_best_bin':    z_summary,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Results saved → {pkl_out}")

plt.show()
