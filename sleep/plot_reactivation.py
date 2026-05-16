"""
Cortical Reactivation — Step 2: Figures & Sleep1/Sleep2 Comparison
===================================================================
Loads the per-epoch PKLs produced by detect_reactivation.py and generates:

  1. Raster figures (2 variants × 4 windows × 2 epochs = 16 figures)
       - 'all'          : every neuron sorted by S1/S2 selectivity index
       - 'reactivation' : only the top reference neurons used for the prior

  2. Sleep1 vs Sleep2 statistical comparison figure
       - Event rate per 5-min segment (Mann-Whitney U test)
       - Mean activity amplitude at detected events

Usage:
    Run detect_reactivation.py first, then:
        python plot_reactivation.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import mannwhitneyu
from scipy.ndimage import label as scipy_label


# ============================================================
# FIGURE: Raster + synchrony prior (colorbar-free, flexible subset)
# ============================================================

def plot_prior_figure_v2(sleep_z, sleep_time_axis,
                          synchrony_prior, mean_activity,
                          selectivity_index, is_driven,
                          top_neuron_idx,
                          neuron_subset='all',
                          window_start_s=0.0,
                          window_s=300.0,
                          bin_size_s=0.05,
                          epoch_label='',
                          save_path=None):
    """
    Raster figure with synchronized top panels.  No colorbar — all three
    panels share the full subplot width so their x-axes are aligned.

    Parameters
    ----------
    sleep_z : (n_neurons, n_bins)
        Z-scored spike matrix for this epoch.
    sleep_time_axis : (n_bins,)
        Time axis in seconds (bin centers, starting near 0).
    synchrony_prior : (n_bins,) bool
    mean_activity : (n_bins,)
        Mean filtered activity of reference neurons.
    selectivity_index : (n_neurons,)
        SI values (NaN-safe, floats).
    is_driven : (n_neurons,) bool
    top_neuron_idx : (n_top,) int
        Global indices of the reference neurons used to compute the prior.
    neuron_subset : {'all', 'reactivation'}
        'all'          – show every neuron sorted by SI.
        'reactivation' – show only top_neuron_idx neurons, sorted by SI.
    window_start_s : float
        Start of the displayed time window (seconds).
    window_s : float
        Duration of the displayed time window (seconds).
    bin_size_s : float
    epoch_label : str
        Label added to the figure title (e.g. 'sleep1').
    save_path : str or None
    """
    # ── select neurons ────────────────────────────────────────────────────
    if neuron_subset == 'reactivation':
        neuron_idx = top_neuron_idx
        si_sub     = selectivity_index[neuron_idx]
        driven_sub = is_driven[neuron_idx]
        act_sub    = sleep_z[neuron_idx]
    else:
        neuron_idx = np.arange(sleep_z.shape[0])
        si_sub     = selectivity_index
        driven_sub = is_driven
        act_sub    = sleep_z

    # Sort by SI descending (S1-preferring on top)
    sort_order    = np.argsort(si_sub)[::-1]
    sorted_act    = act_sub[sort_order]
    sorted_si     = si_sub[sort_order]
    sorted_driven = driven_sub[sort_order]
    n_neurons     = sorted_act.shape[0]

    # S1/S2 boundary
    boundary = int(np.searchsorted(-sorted_si, 0))

    # ── time mask ─────────────────────────────────────────────────────────
    mask   = (sleep_time_axis >= window_start_s) & \
             (sleep_time_axis <  window_start_s + window_s)
    t_plot = sleep_time_axis[mask]

    if t_plot.size == 0:
        print(f"  [WARN] No bins in window [{window_start_s:.0f}, "
              f"{window_start_s + window_s:.0f}) s — skipping figure.")
        return None

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 5], hspace=0.05)

    subset_label = ('Reference neurons only'
                    if neuron_subset == 'reactivation' else 'All neurons')
    fig.suptitle(
        f"Synchronous activity prior  |  {epoch_label}  |  {subset_label}\n"
        f"Window: {window_start_s:.0f}–{window_start_s + window_s:.0f} s",
        fontsize=11, fontweight='bold',
    )

    # ── Panel 1: mean reference activity ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_plot, mean_activity[mask], color='steelblue',
             linewidth=0.8, label='Mean ref. activity')
    ax1.fill_between(t_plot,
                     mean_activity[mask] * synchrony_prior[mask],
                     0, color='orange', alpha=0.6, label='Synchronous event')
    ax1.set_ylabel('Filtered\nactivity\n(a.u.)', fontsize=9)
    ax1.set_xlim(t_plot[0], t_plot[-1])
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.5)
    ax1.tick_params(labelbottom=False)

    # ── Panel 2: binary prior ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(t_plot, synchrony_prior[mask].astype(float),
                     color='orange', step='mid')
    ax2.set_ylabel('Prior\n(binary)', fontsize=9)
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([0, 1])
    ax2.tick_params(labelbottom=False)

    # ── Panel 3: raster (no colorbar) ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    ax3.imshow(
        sorted_act[:, mask],
        aspect='auto',
        extent=[t_plot[0] - bin_size_s / 2,
                t_plot[-1] + bin_size_s / 2,
                n_neurons, 0],
        cmap='Greys',
        vmin=-0.5, vmax=2.0,
        interpolation='nearest',
    )

    # Thin colored bar on the right showing driven neurons
    driven_colors = np.where(sorted_driven, 1.0, 0.0).reshape(-1, 1)
    right_edge    = t_plot[-1] + bin_size_s / 2
    bar_width     = (t_plot[-1] - t_plot[0]) * 0.013
    ax3.imshow(driven_colors,
               aspect='auto',
               extent=[right_edge + bar_width * 0.2,
                       right_edge + bar_width * 1.2,
                       n_neurons, 0],
               cmap='Greens', vmin=0, vmax=1,
               interpolation='nearest', clip_on=False)

    # S1/S2 boundary
    ax3.axhline(boundary, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    bar_x = right_edge + bar_width * 1.5
    ax3.text(bar_x, boundary / 2,
             'S1\npref.', color='green', fontsize=9,
             va='center', ha='left', clip_on=False)
    ax3.text(bar_x, boundary + (n_neurons - boundary) / 2,
             'S2\npref.', color='red', fontsize=9,
             va='center', ha='left', clip_on=False)

    ax3.set_ylabel(f'Neuron # (n={n_neurons},\nsorted by SI)', fontsize=9)
    ax3.set_xlabel('Time (s)', fontsize=10)

    # Vertical orange lines at event onsets
    prior_masked  = synchrony_prior[mask]
    event_onsets  = t_plot[np.where(np.diff(prior_masked.astype(int)) == 1)[0] + 1]
    for t_ev in event_onsets:
        ax3.axvline(t_ev, color='orange', lw=0.8, alpha=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    return fig


# ============================================================
# STATISTICS: Compare event rates between sleep1 and sleep2
# ============================================================

def compare_sleep_epochs(prior1, time1, mean_act1,
                          prior2, time2, mean_act2,
                          segment_s=300.0):
    """
    Compare synchronous event rates between two sleep epochs.

    Divides each epoch into non-overlapping segments of `segment_s` seconds,
    counts events/min per segment, then runs a Mann-Whitney U test.

    Parameters
    ----------
    prior1, prior2 : (n_bins,) bool
    time1, time2   : (n_bins,) float  — time axis in seconds
    mean_act1, mean_act2 : (n_bins,) float
    segment_s : float
        Segment length in seconds (default 300 s = 5 min).

    Returns
    -------
    stats : dict with keys:
        rates_s1, rates_s2     — event rates per segment (events/min)
        mean_s1, mean_s2       — mean rate
        sem_s1, sem_s2         — SEM of rate
        u_stat, p_value        — Mann-Whitney result
        amp_s1, amp_s2         — mean activity amplitude at event peaks
        n_segments_s1          — number of segments used for sleep1
        n_segments_s2          — number of segments used for sleep2
    """
    def _segment_rates(prior, time, mean_act):
        bin_size = float(time[1] - time[0])
        seg_bins = max(1, int(round(segment_s / bin_size)))
        total_bins = len(prior)
        n_segs = total_bins // seg_bins

        rates = []
        for i in range(n_segs):
            seg = prior[i * seg_bins: (i + 1) * seg_bins]
            n_events = int(np.sum(np.diff(seg.astype(int)) == 1))
            seg_dur_min = (seg_bins * bin_size) / 60.0
            rates.append(n_events / seg_dur_min)

        # Event peak amplitudes (global across epoch)
        event_mask = prior.astype(bool)
        amp = float(mean_act[event_mask].mean()) if event_mask.any() else 0.0

        return np.array(rates), amp

    rates1, amp1 = _segment_rates(prior1, time1, mean_act1)
    rates2, amp2 = _segment_rates(prior2, time2, mean_act2)

    u_stat, p_value = mannwhitneyu(rates1, rates2, alternative='two-sided')

    return {
        "rates_s1"      : rates1,
        "rates_s2"      : rates2,
        "mean_s1"       : float(rates1.mean()),
        "mean_s2"       : float(rates2.mean()),
        "sem_s1"        : float(rates1.std() / np.sqrt(len(rates1))),
        "sem_s2"        : float(rates2.std() / np.sqrt(len(rates2))),
        "u_stat"        : float(u_stat),
        "p_value"       : float(p_value),
        "amp_s1"        : amp1,
        "amp_s2"        : amp2,
        "n_segments_s1" : len(rates1),
        "n_segments_s2" : len(rates2),
    }


def plot_comparison_figure(stats, session='', save_path=None):
    """
    Bar + strip plot comparing sleep1 vs sleep2 event rates and amplitudes.

    Parameters
    ----------
    stats : dict returned by compare_sleep_epochs()
    session : str  — used in the title
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Sleep1 vs Sleep2 synchronous event comparison  |  {session}",
        fontsize=12, fontweight='bold',
    )

    labels   = ['Sleep 1', 'Sleep 2']
    colors   = ['#4878CF', '#D65F5F']
    x_pos    = [0, 1]

    # ── Left panel: event rate ────────────────────────────────────────────
    ax = axes[0]
    means = [stats['mean_s1'], stats['mean_s2']]
    sems  = [stats['sem_s1'],  stats['sem_s2']]
    all_rates = [stats['rates_s1'], stats['rates_s2']]

    bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7,
                  width=0.4, capsize=6, error_kw=dict(lw=1.5))

    # Individual segment dots
    rng = np.random.default_rng(42)
    for xi, rates, col in zip(x_pos, all_rates, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(rates))
        ax.scatter(xi + jitter, rates, color=col, alpha=0.5,
                   s=20, zorder=3, linewidths=0)

    # p-value annotation
    p = stats['p_value']
    p_str = (f"p = {p:.3f}" if p >= 0.001
             else f"p = {p:.2e}")
    sig_str = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'n.s.')
    y_max = max(r.max() for r in all_rates) * 1.15
    ax.plot([0, 1], [y_max, y_max], 'k-', lw=1)
    ax.text(0.5, y_max * 1.02, f"{sig_str}\n({p_str}, Mann-Whitney U)",
            ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Event rate (events / min)', fontsize=10)
    ax.set_title(f'n segs: Sleep1={stats["n_segments_s1"]}, '
                 f'Sleep2={stats["n_segments_s2"]}', fontsize=9)
    ax.set_ylim(0, y_max * 1.18)

    # ── Right panel: mean amplitude at events ────────────────────────────
    ax2 = axes[1]
    amps = [stats['amp_s1'], stats['amp_s2']]
    ax2.bar(x_pos, amps, color=colors, alpha=0.7, width=0.4)
    for xi, val, col in zip(x_pos, amps, colors):
        ax2.text(xi, val + max(amps) * 0.02, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Mean activity at events (a.u.)', fontsize=10)
    ax2.set_title('Mean amplitude at synchronous events', fontsize=9)
    ax2.set_ylim(0, max(amps) * 1.2 if max(amps) > 0 else 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    _SLEEP_DIR = Path(__file__).resolve().parent
    if str(_SLEEP_DIR) not in sys.path:
        sys.path.insert(0, str(_SLEEP_DIR))

    from reactivation_config import SESSION, REACTIVATION_DIR

    REACTIVATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load per-epoch PKLs ───────────────────────────────────────────────
    epochs_data = {}
    for ep_name in ("sleep1", "sleep2"):
        pkl_path = REACTIVATION_DIR / f"{SESSION}_{ep_name}_prior.pkl"
        if not pkl_path.exists():
            print(f"[WARN] PKL not found: {pkl_path}  — skipping {ep_name}")
            continue
        with open(pkl_path, "rb") as fh:
            epochs_data[ep_name] = pickle.load(fh)
        print(f"Loaded {ep_name}  ({epochs_data[ep_name]['sleep_duration_s']:.1f} s)")

    if not epochs_data:
        raise FileNotFoundError(
            "No epoch PKLs found. Run detect_reactivation.py first."
        )

    # ── Per-epoch raster figures ──────────────────────────────────────────
    N_WINDOWS  = 10     # evenly spaced windows per epoch
    WINDOW_S   = 300.0  # 5 min per window

    print(f"\nGenerating raster figures ({N_WINDOWS} windows × "
          f"2 subsets × {len(epochs_data)} epochs) ...")

    for ep_name, d in epochs_data.items():
        dur_s     = d['sleep_duration_s']
        time_axis = d['sleep_time_axis']
        sleep_z   = d['sleep_z']
        prior     = d['synchrony_prior']
        mean_act  = d['mean_activity']
        si        = d['selectivity_index']
        is_driv   = d['is_driven']
        top_idx   = d['top_neuron_idx']
        bin_s     = d['bin_size_s']

        # Evenly spaced window starts; clamp so window fits within epoch
        max_start  = max(0.0, dur_s - WINDOW_S)
        win_starts = np.linspace(0, max_start, N_WINDOWS)

        for wi, w_start in enumerate(win_starts, start=1):
            for subset in ('all', 'reactivation'):
                fname = (f"{SESSION}_{ep_name}_{subset}_window{wi}.png")
                fig = plot_prior_figure_v2(
                    sleep_z, time_axis,
                    prior, mean_act,
                    si, is_driv, top_idx,
                    neuron_subset=subset,
                    window_start_s=w_start,
                    window_s=WINDOW_S,
                    bin_size_s=bin_s,
                    epoch_label=ep_name,
                    save_path=str(REACTIVATION_DIR / fname),
                )
                if fig is not None:
                    plt.close(fig)

    # ── Statistical comparison ────────────────────────────────────────────
    if 'sleep1' in epochs_data and 'sleep2' in epochs_data:
        print("\nComputing sleep1 vs sleep2 statistics ...")
        d1 = epochs_data['sleep1']
        d2 = epochs_data['sleep2']

        stats = compare_sleep_epochs(
            prior1    = d1['synchrony_prior'],
            time1     = d1['sleep_time_axis'],
            mean_act1 = d1['mean_activity'],
            prior2    = d2['synchrony_prior'],
            time2     = d2['sleep_time_axis'],
            mean_act2 = d2['mean_activity'],
            segment_s = 300.0,
        )

        print(f"  Sleep1 event rate: {stats['mean_s1']:.3f} ± {stats['sem_s1']:.3f} events/min"
              f"  (n={stats['n_segments_s1']} segments)")
        print(f"  Sleep2 event rate: {stats['mean_s2']:.3f} ± {stats['sem_s2']:.3f} events/min"
              f"  (n={stats['n_segments_s2']} segments)")
        print(f"  Mann-Whitney U = {stats['u_stat']:.1f},  p = {stats['p_value']:.4f}")

        fig_cmp = plot_comparison_figure(
            stats,
            session=SESSION,
            save_path=str(REACTIVATION_DIR / f"{SESSION}_sleep1_vs_sleep2_comparison.png"),
        )
        plt.close(fig_cmp)
    else:
        print("\n[INFO] Only one epoch available — skipping comparison figure.")

    print(f"\n{'='*65}")
    print("plot_reactivation.py  DONE")
    print(f"  Figures saved to: {REACTIVATION_DIR}")
    print(f"{'='*65}")
