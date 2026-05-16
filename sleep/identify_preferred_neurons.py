"""
identify_preferred_neurons.py
==============================
Identify S1-preferring (45°) and S2-preferring (135°) neurons using the
method from Nguyen et al. (cortical reactivation paper):

  Step 1 — Stimulus-driven neurons
    Wilcoxon rank-sum test comparing baseline FR vs. stimulus FR (p < 0.01)
    for S1 or S2 trials.  Baseline = spike count in pre-stimulus window
    (-2 s to 0 s); stimulus = spike count in stimulus window (0 to 2 s).

  Step 2 — Selectivity index
    SI = (R_S1 - R_S2) / (R_S1 + R_S2)
    SI > 0 → S1-preferring (45°)   SI < 0 → S2-preferring (135°)

Visualisations
--------------
  Figure 1  — Heatmap (sorted by SI, like paper Fig 1c) + SI distribution
  Figure 2  — Scatter R_S1 vs R_S2 with stimulus-driven colour coding
  Figure 3  — Mean PSTH for top-20 S1-pref and top-20 S2-pref neurons

Output saved to REACTIVATION_DIR / preferred_neurons.pkl
"""

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ranksums

# ── imports ──────────────────────────────────────────────────────────────────
_GRATING_DIR = (
    Path(__file__).resolve().parent.parent
    / "rf_recon" / "FreelyMovingProcessing" / "Grating"
)
if str(_GRATING_DIR) not in sys.path:
    sys.path.insert(0, str(_GRATING_DIR))
import grating_utils

from reactivation_config import GRATING_PKL, SESSION, REACTIVATION_DIR

REACTIVATION_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
S1_ORI         = 45.0      # "grating 1" orientation
S2_ORI         = 135.0     # "grating 2" orientation
STIM_WIN       = (0.0, 2.0)   # seconds relative to trial onset
# Pre-stimulus baseline window (negative = before trial onset)
# If spike_times don't extend to negative values, we approximate with
# a shuffled-trial baseline (see fallback below).
PRE_WIN        = (-2.0, 0.0)
P_THRESH       = 0.01
TOP_FRAC       = 0.05      # top 5% for reactivation classifier (shown in annotations)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"identify_preferred_neurons.py — {SESSION}")
print(f"  Grating pkl: {GRATING_PKL}")

if not GRATING_PKL.exists():
    raise FileNotFoundError(f"GRATING_PKL not found: {GRATING_PKL}")

neural_data = grating_utils.load_neural_data(GRATING_PKL)

# Remove noise units
unit_info = neural_data.get('unit_info', {})
all_unit_ids = list(neural_data['spike_data'].keys())
unit_ids = [u for u in all_unit_ids
            if str(unit_info.get(u, {}).get('quality', 'unknown')).lower() != 'noise']
n_noise = len(all_unit_ids) - len(unit_ids)
if n_noise:
    print(f"  Excluded {n_noise} noise unit(s)")

orientations = np.array(neural_data['trial_info']['orientations'], dtype=float)
n_units      = len(unit_ids)
n_trials     = len(orientations)

print(f"  Units: {n_units}  (noise excluded: {n_noise})   Trials: {n_trials}")
print(f"  Unique orientations: {np.unique(orientations)}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD SPIKE-COUNT MATRICES
#   stim_counts[trial, unit]  — spikes during stimulus window
#   base_counts[trial, unit]  — spikes during baseline window
# ─────────────────────────────────────────────────────────────────────────────
stim_dur = STIM_WIN[1] - STIM_WIN[0]
pre_dur  = PRE_WIN[1]  - PRE_WIN[0]   # positive duration

stim_counts = np.full((n_trials, n_units), np.nan)
base_counts = np.full((n_trials, n_units), np.nan)
has_baseline = False   # will be set True if any pre-stim spikes found

for ui, uid in enumerate(unit_ids):
    for td in neural_data['spike_data'][uid]:
        ti = int(td['trial_index'])
        if not (0 <= ti < n_trials):
            continue
        spkt = np.array(td['spike_times'], dtype=float)

        stim_counts[ti, ui] = np.sum(
            (spkt >= STIM_WIN[0]) & (spkt < STIM_WIN[1])
        )
        n_pre = np.sum((spkt >= PRE_WIN[0]) & (spkt < PRE_WIN[1]))
        base_counts[ti, ui] = n_pre
        if n_pre > 0 or np.any((spkt >= PRE_WIN[0]) & (spkt < PRE_WIN[1])):
            has_baseline = True

# Remove trials with missing data
valid_mask = ~(np.isnan(stim_counts).any(axis=1) |
               np.isnan(base_counts).any(axis=1))
stim_counts = stim_counts[valid_mask]
base_counts = base_counts[valid_mask]
oris        = orientations[valid_mask]

s1_mask = oris == S1_ORI
s2_mask = oris == S2_ORI

print(f"  Valid trials (all orientations): {np.sum(valid_mask)} / {n_trials}")
print(f"    S1 (45°)  trials used for Wilcoxon / SI: {s1_mask.sum()}")
print(f"    S2 (135°) trials used for Wilcoxon / SI: {s2_mask.sum()}")

# Check whether pre-stim window really contains spikes
has_baseline = np.any(base_counts > 0)
print(f"  Pre-stimulus baseline {'found' if has_baseline else 'NOT found — using shuffled baseline'}")

# ─────────────────────────────────────────────────────────────────────────────
# MEAN FIRING RATES  (Hz)
# ─────────────────────────────────────────────────────────────────────────────
# Use only S1 / S2 trials
r_s1 = stim_counts[s1_mask].mean(axis=0) / stim_dur   # (n_units,)
r_s2 = stim_counts[s2_mask].mean(axis=0) / stim_dur

# ─────────────────────────────────────────────────────────────────────────────
# WILCOXON RANK-SUM TEST — stimulus-driven
# ─────────────────────────────────────────────────────────────────────────────
p_s1 = np.ones(n_units)
p_s2 = np.ones(n_units)

for ui in range(n_units):
    s1_resp = stim_counts[s1_mask, ui]
    s2_resp = stim_counts[s2_mask, ui]

    if has_baseline:
        baseline_s1 = base_counts[s1_mask, ui]
        baseline_s2 = base_counts[s2_mask, ui]
    else:
        # Fallback: cross-orientation comparison
        # "baseline for S1" = S2 stim responses (and vice versa)
        baseline_s1 = s2_resp
        baseline_s2 = s1_resp

    # Only test if there's non-trivial variance
    if len(np.unique(np.concatenate([s1_resp, baseline_s1]))) > 1:
        _, p_s1[ui] = ranksums(s1_resp, baseline_s1, alternative='greater')
    if len(np.unique(np.concatenate([s2_resp, baseline_s2]))) > 1:
        _, p_s2[ui] = ranksums(s2_resp, baseline_s2, alternative='greater')

# Stimulus-driven: significant for S1 OR S2
driven_s1 = p_s1 < P_THRESH
driven_s2 = p_s2 < P_THRESH
driven    = driven_s1 | driven_s2

print(f"\n  Stimulus-driven neurons (p<{P_THRESH}):")
print(f"    S1-driven : {driven_s1.sum()}")
print(f"    S2-driven : {driven_s2.sum()}")
print(f"    Any-driven: {driven.sum()} / {n_units}")

# ─────────────────────────────────────────────────────────────────────────────
# SELECTIVITY INDEX
# ─────────────────────────────────────────────────────────────────────────────
denom = r_s1 + r_s2
# Avoid division by zero for silent neurons
safe_denom = np.where(denom > 0, denom, np.nan)
SI = (r_s1 - r_s2) / safe_denom   # range (-1, +1); NaN for silent units

# Among driven neurons, label preference
pref_s1 = driven & (SI > 0)    # S1-preferring
pref_s2 = driven & (SI < 0)    # S2-preferring

print(f"\n  Preferred neurons (driven + SI sign):")
print(f"    S1-preferring (45°) : {pref_s1.sum()}")
print(f"    S2-preferring (135°): {pref_s2.sum()}")

# Top 5% by drive strength (for classifier)
n_top = max(1, int(np.ceil(TOP_FRAC * n_units)))
top_s1_idx = np.argsort(r_s1)[::-1][:n_top]   # highest S1 FR
top_s2_idx = np.argsort(r_s2)[::-1][:n_top]

top_s1_ids = [unit_ids[i] for i in top_s1_idx]
top_s2_ids = [unit_ids[i] for i in top_s2_idx]
print(f"\n  Top {TOP_FRAC*100:.0f}% by drive strength ({n_top} each):")
print(f"    Top S1: {top_s1_ids[:5]} ...")
print(f"    Top S2: {top_s2_ids[:5]} ...")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PSTH  (for visualisation)
# ─────────────────────────────────────────────────────────────────────────────
# Bins span exactly the extraction window (-0.2 to 2.0 s), 50 ms bins.
ep = neural_data.get('extraction_params', {})
t_pre  = float(ep.get('window_pre',  0.2))   # seconds before stim onset
t_post = float(ep.get('window_post', 2.0))   # seconds after stim onset
psth_bins = np.arange(-t_pre, t_post + 1e-9, 0.05)
bin_ctrs  = 0.5 * (psth_bins[:-1] + psth_bins[1:])
bin_dur   = psth_bins[1] - psth_bins[0]

# Baseline bins: all bins before stimulus onset (t < 0)
bl_mask = bin_ctrs < 0

# psth_s1[unit, bin] and psth_s2[unit, bin] — raw spike counts summed over trials
psth_s1 = np.zeros((n_units, len(bin_ctrs)))
psth_s2 = np.zeros((n_units, len(bin_ctrs)))
cnt_s1  = np.zeros(n_units, dtype=int)
cnt_s2  = np.zeros(n_units, dtype=int)

ori_all = orientations   # full (n_trials,) before valid_mask trim
for ui, uid in enumerate(unit_ids):
    for td in neural_data['spike_data'][uid]:
        ti  = int(td['trial_index'])
        if not (0 <= ti < len(ori_all)):
            continue
        ori = float(ori_all[ti])
        spkt = np.array(td['spike_times'], dtype=float)
        hist, _ = np.histogram(spkt, bins=psth_bins)
        if ori == S1_ORI:
            psth_s1[ui] += hist
            cnt_s1[ui] += 1
        elif ori == S2_ORI:
            psth_s2[ui] += hist
            cnt_s2[ui] += 1

# Convert to Hz (avoid /0)
with np.errstate(invalid='ignore'):
    psth_s1_hz = np.where(cnt_s1[:, None] > 0,
                          psth_s1 / (cnt_s1[:, None] * bin_dur), 0.0)
    psth_s2_hz = np.where(cnt_s2[:, None] > 0,
                          psth_s2 / (cnt_s2[:, None] * bin_dur), 0.0)

# ── Z-score: baseline = pre-stimulus bins (<0) pooled across S1 and S2 ───────
# mean_bl[unit] and std_bl[unit] computed from the baseline portion of each
# neuron's combined (S1 + S2) PSTH, so S1 and S2 share the same z-score scale.
total_trials = np.maximum(cnt_s1 + cnt_s2, 1)[:, None]
psth_all_hz  = (psth_s1 + psth_s2) / (total_trials * bin_dur)

bl_mean = psth_all_hz[:, bl_mask].mean(axis=1, keepdims=True)   # (n_units, 1)
bl_std  = psth_all_hz[:, bl_mask].std(axis=1, keepdims=True)
# Add a small floor to avoid divide-by-zero for silent neurons (< 0.01 Hz std)
bl_std  = np.where(bl_std < 0.01, 0.01, bl_std)

psth_s1_z = (psth_s1_hz - bl_mean) / bl_std
psth_s2_z = (psth_s2_hz - bl_mean) / bl_std

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out = {
    "unit_ids"    : unit_ids,
    "SI"          : SI,
    "r_s1"        : r_s1,
    "r_s2"        : r_s2,
    "p_s1"        : p_s1,
    "p_s2"        : p_s2,
    "driven"      : driven,
    "pref_s1"     : pref_s1,
    "pref_s2"     : pref_s2,
    "top_s1_ids"  : top_s1_ids,
    "top_s2_ids"  : top_s2_ids,
    "psth_s1_hz"  : psth_s1_hz,
    "psth_s2_hz"  : psth_s2_hz,
    "psth_s1_z"   : psth_s1_z,
    "psth_s2_z"   : psth_s2_z,
    "psth_bin_ctrs": bin_ctrs,
    "has_baseline" : has_baseline,
    "session"     : SESSION,
}
out_pkl = REACTIVATION_DIR / "preferred_neurons.pkl"
with open(out_pkl, "wb") as fh:
    pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f"\n  Saved → {out_pkl}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Heatmap + SI histogram
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 6))
fig1.suptitle(
    f"{SESSION} — Grating-Preferred Neurons\n"
    f"S1=45° (n={pref_s1.sum()}), S2=135° (n={pref_s2.sum()}),  "
    f"driven={driven.sum()}/{n_units}",
    fontsize=11, fontweight="bold"
)

# Panel A — SI distribution
ax = axes[0]
si_vals  = SI[driven] if driven.sum() > 0 else SI[~np.isnan(SI)]
bins_si  = np.linspace(-1, 1, 31)
c_s1     = np.array([0.2, 0.5, 0.9])    # blue
c_s2     = np.array([0.9, 0.3, 0.2])    # red
ax.hist(si_vals[si_vals > 0], bins=bins_si, color=c_s1,
        alpha=0.7, label=f"S1-pref (n={pref_s1.sum()})")
ax.hist(si_vals[si_vals < 0], bins=bins_si, color=c_s2,
        alpha=0.7, label=f"S2-pref (n={pref_s2.sum()})")
ax.axvline(0, color='k', lw=1.2, ls='--')
ax.set_xlabel("Selectivity Index  (S1−S2)/(S1+S2)", fontsize=9)
ax.set_ylabel("# driven neurons")
ax.set_title("Selectivity Index Distribution\n(driven neurons only)")
ax.legend(fontsize=9)

# Panel B — bar chart of all SI values sorted
ax = axes[1]
si_all_sorted = np.sort(SI[~np.isnan(SI)])
colors_bar = ['steelblue' if s > 0 else 'tomato' for s in si_all_sorted]
ax.bar(range(len(si_all_sorted)), si_all_sorted, color=colors_bar,
       width=1.0, linewidth=0)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel("Neuron rank (sorted by SI)")
ax.set_ylabel("Selectivity Index")
ax.set_title("All neurons — SI ranked\n(blue=S1-pref, red=S2-pref)")
ax.set_ylim(-1.05, 1.05)

plt.tight_layout()
fig1_path = REACTIVATION_DIR / f"{SESSION}_preferred_neurons_heatmap.png"
plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Fig 1 → {fig1_path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Scatter R_S1 vs R_S2
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 6))
ax.scatter(r_s1[~driven], r_s2[~driven],
           c='gray', alpha=0.4, s=25, label='not driven', zorder=2)
ax.scatter(r_s1[pref_s1], r_s2[pref_s1],
           c='steelblue', alpha=0.8, s=50, label=f'S1-pref (n={pref_s1.sum()})', zorder=4)
ax.scatter(r_s1[pref_s2], r_s2[pref_s2],
           c='tomato', alpha=0.8, s=50, label=f'S2-pref (n={pref_s2.sum()})', zorder=4)
# Mark top-5%
ax.scatter(r_s1[top_s1_idx], r_s2[top_s1_idx],
           facecolors='none', edgecolors='blue', linewidths=1.5, s=80,
           label=f'Top {TOP_FRAC*100:.0f}% S1', zorder=5)
ax.scatter(r_s1[top_s2_idx], r_s2[top_s2_idx],
           facecolors='none', edgecolors='red', linewidths=1.5, s=80,
           label=f'Top {TOP_FRAC*100:.0f}% S2', zorder=5)

lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='diagonal')
ax.set_xlabel("Mean FR — 45° trials (Hz)", fontsize=10)
ax.set_ylabel("Mean FR — 135° trials (Hz)", fontsize=10)
ax.set_title(f"{SESSION}\nFiring Rate: 45° vs 135°  (p<{P_THRESH} Wilcoxon)",
             fontsize=10)
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
fig2_path = REACTIVATION_DIR / f"{SESSION}_preferred_neurons_scatter.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Fig 2 → {fig2_path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — PSTH for top S1 and top S2 neurons
# ─────────────────────────────────────────────────────────────────────────────
n_show = min(20, n_units)
# Pick top neurons: among SI>0 and SI<0, ranked by |SI|
s1_rank = np.argsort(SI)[::-1]   # highest SI first
s2_rank = np.argsort(SI)          # lowest SI first (most negative)
top_s1_show = s1_rank[:n_show]
top_s2_show = s2_rank[:n_show]

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
fig3.suptitle(f"{SESSION} — PSTHs for top S1-pref & S2-pref neurons\n"
              f"(showing top {n_show} each by |SI|)", fontsize=11)

stim_off = t_post   # seconds, for the vertical line marking stim offset

for row, (idxs, label, col45, col135) in enumerate([
    (top_s1_show, "S1-pref (45° > 135°)", 'steelblue', 'tomato'),
    (top_s2_show, "S2-pref (135° > 45°)", 'tomato', 'steelblue'),
]):
    # Left: z-scored preferred-stimulus PSTH heatmap, neurons sorted by peak time
    ax = axes3[row, 0]
    # row 0 = S1-pref → show S1 (45°); row 1 = S2-pref → show S2 (135°)
    mat_z = psth_s1_z[idxs] if row == 0 else psth_s2_z[idxs]
    pref_label = "45°" if row == 0 else "135°"
    # Sort neurons by peak bin during stimulus (t >= 0)
    stim_bins = bin_ctrs >= 0
    peak_bins = np.argmax(mat_z[:, stim_bins], axis=1)
    sort_rows = np.argsort(peak_bins)
    mat_sorted = mat_z[sort_rows]
    vabs = np.nanpercentile(np.abs(mat_sorted), 98)
    vabs = max(vabs, 1.0)
    im = ax.imshow(mat_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs,
                   extent=[bin_ctrs[0], bin_ctrs[-1], len(idxs), 0],
                   interpolation="nearest")
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axvline(stim_off, color='k', lw=0.8, ls=':')
    ax.set_xlabel("Time from stim onset (s)")
    ax.set_ylabel("Neuron rank (sorted by peak)")
    ax.set_title(f"{label}\nZ-scored FR — {pref_label} trials")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z-score")

    # Right: mean z-scored PSTH ± SEM for S1 and S2
    ax = axes3[row, 1]
    mean_s1 = psth_s1_z[idxs].mean(axis=0)
    mean_s2 = psth_s2_z[idxs].mean(axis=0)
    sem_s1  = psth_s1_z[idxs].std(axis=0) / np.sqrt(len(idxs))
    sem_s2  = psth_s2_z[idxs].std(axis=0) / np.sqrt(len(idxs))
    ax.fill_between(bin_ctrs, mean_s1 - sem_s1, mean_s1 + sem_s1,
                    alpha=0.25, color=col45)
    ax.fill_between(bin_ctrs, mean_s2 - sem_s2, mean_s2 + sem_s2,
                    alpha=0.25, color=col135)
    ax.plot(bin_ctrs, mean_s1, color=col45,  lw=1.8, label='45°')
    ax.plot(bin_ctrs, mean_s2, color=col135, lw=1.8, label='135°')
    ax.axhline(0, color='gray', lw=0.7, ls='-')
    ax.axvline(0, color='k', lw=0.8, ls='--', label='stim on')
    ax.axvline(stim_off, color='k', lw=0.8, ls=':', label='stim off')
    ax.set_xlabel("Time from stim onset (s)")
    ax.set_ylabel("Z-scored FR (a.u.)")
    ax.set_title(f"Mean z-scored PSTH ± SEM   {label}")
    ax.legend(fontsize=8)

plt.tight_layout()
fig3_path = REACTIVATION_DIR / f"{SESSION}_preferred_neurons_psth.png"
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Fig 3 → {fig3_path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — All neurons ranked by SI, z-scored PSTH for both stimuli
# ─────────────────────────────────────────────────────────────────────────────
# Sort all neurons by SI descending (S1-pref on top, S2-pref at bottom).
# Neurons with NaN SI (no spikes at all) go to the bottom.
si_for_sort = np.where(np.isnan(SI), -np.inf, SI)
rank_order  = np.argsort(si_for_sort)[::-1]   # (n_units,)

mat_s1_ranked = psth_s1_z[rank_order]   # (n_units, n_bins)
mat_s2_ranked = psth_s2_z[rank_order]

# Shared colour scale: symmetric around 0, clipped at 98th percentile of |z|
vabs = np.nanpercentile(np.abs(np.concatenate([mat_s1_ranked, mat_s2_ranked])), 98)
vabs = max(vabs, 1.0)

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 8),
                            gridspec_kw=dict(wspace=0.08))
fig4.suptitle(
    f"{SESSION} — All neurons ranked by Selectivity Index\n"
    f"S1-pref (45°, n={pref_s1.sum()}) → S2-pref (135°, n={pref_s2.sum()})",
    fontsize=11, fontweight="bold",
)

extent = [bin_ctrs[0], bin_ctrs[-1], n_units, 0]

for ax, mat, stim_label in [
    (axes4[0], mat_s1_ranked, "45° (S1)"),
    (axes4[1], mat_s2_ranked, "135° (S2)"),
]:
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs,
                   extent=extent, interpolation="nearest")
    ax.axvline(0,      color='k', lw=1.0, ls='--')
    ax.axvline(t_post, color='k', lw=1.0, ls=':')
    ax.set_xlabel("Time from stim onset (s)", fontsize=10)
    ax.set_title(f"Z-scored PSTH — {stim_label}", fontsize=11)

    # Annotate the SI=0 boundary
    n_s1_pref = int(np.sum(si_for_sort[rank_order] > 0))
    ax.axhline(n_s1_pref, color='white', lw=1.2, ls='--', alpha=0.8)
    ax.text(bin_ctrs[-1] * 0.98, n_s1_pref + n_units * 0.01,
            "SI=0", color='white', fontsize=7, ha='right', va='top')

axes4[0].set_ylabel("Neuron rank (S1-pref → S2-pref)", fontsize=10)
axes4[1].set_yticks([])

# Single shared colorbar
cbar = fig4.colorbar(im, ax=axes4.tolist(), fraction=0.02, pad=0.02)
cbar.set_label("Z-score", fontsize=10)

fig4_path = REACTIVATION_DIR / f"{SESSION}_all_neurons_ranked_psth.png"
plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Fig 4 → {fig4_path.name}")

print(f"\n{'='*65}")
print("identify_preferred_neurons.py  DONE")
print(f"{'='*65}")
