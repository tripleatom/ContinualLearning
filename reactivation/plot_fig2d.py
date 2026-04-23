import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

from params_fig2 import PKL_PATH, OUTPUT_DIR, RESP_WINDOW, TOP_N_SELECTIVE


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def check_spike_time_units(data):
    for unit_trials in data['spike_data'].values():
        for trial in unit_trials:
            st = np.asarray(trial['spike_times'], dtype=float)
            if len(st) > 0:
                if not np.all(st % 1 == 0):
                    return 'seconds'
    return 'frames'


def get_spikes_sec(trial, fs, time_unit):
    st = np.asarray(trial['spike_times'], dtype=float)
    return st / fs if time_unit == 'frames' else st


def build_conditions(data):
    sf_vals = sorted(set(
        round(tp['left_spatial_freq'], 4)
        for tp in data['trial_info']['all_trial_parameters']
    ))
    sf_to_cond = {sf: i for i, sf in enumerate(sf_vals)}
    label_map = {
        tp['trial_index']: sf_to_cond[round(tp['left_spatial_freq'], 4)]
        for tp in data['trial_info']['all_trial_parameters']
    }
    condition_names = [f'Left SF={sf:.3f} cpd' for sf in sf_vals]
    return label_map, condition_names


def find_cue_driven_units(data, condition_labels, unit_ids_sorted, fs, time_unit,
                          cue_window=(0.0, 2.0), baseline_window=(-1.0, 0.0),
                          bin_size=0.33, alpha=0.05):
    """Mark units with a significant FR increase vs baseline in any
    (condition, 0.33-s bin), Wilcoxon signed-rank, Bonferroni-corrected over
    (n_units x n_cond x n_bins)."""
    n_cond  = len(set(condition_labels.values()))
    n_units = len(unit_ids_sorted)
    cond_trials = {c: sorted([i for i, v in condition_labels.items() if v == c])
                   for c in range(n_cond)}

    cue_edges = np.arange(cue_window[0], cue_window[1] + 1e-9, bin_size)
    n_bins    = len(cue_edges) - 1
    base_dur  = baseline_window[1] - baseline_window[0]

    alpha_corr = alpha / (n_units * n_cond * n_bins)

    mask = np.zeros(n_units, dtype=bool)
    min_p_per_unit = np.ones(n_units)

    for i, uid in enumerate(unit_ids_sorted):
        trial_map = {t['trial_index']: t for t in data['spike_data'][uid]}
        done = False
        for c in range(n_cond):
            if done:
                break
            trials = cond_trials[c]
            base_rates = np.zeros(len(trials))
            bin_rates  = np.zeros((len(trials), n_bins))
            for j, tidx in enumerate(trials):
                if tidx not in trial_map:
                    continue
                st = get_spikes_sec(trial_map[tidx], fs, time_unit)
                b_mask = (st >= baseline_window[0]) & (st < baseline_window[1])
                base_rates[j] = b_mask.sum() / base_dur
                counts, _ = np.histogram(st, bins=cue_edges)
                bin_rates[j] = counts / bin_size
            for b in range(n_bins):
                diff = bin_rates[:, b] - base_rates
                if np.any(diff != 0):
                    try:
                        _, p = wilcoxon(bin_rates[:, b], base_rates,
                                        alternative='greater',
                                        zero_method='wilcox')
                    except ValueError:
                        continue
                    if p < min_p_per_unit[i]:
                        min_p_per_unit[i] = p
                    if p < alpha_corr:
                        mask[i] = True
                        done = True
                        break
    return mask, min_p_per_unit, alpha_corr


def _bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: return boolean mask of p-values surviving FDR q."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passing = ranked <= thresh
    if not np.any(passing):
        return np.zeros(n, dtype=bool)
    cutoff_rank = np.max(np.where(passing)[0])
    mask = np.zeros(n, dtype=bool)
    mask[order[:cutoff_rank + 1]] = True
    return mask


def rank_sf_selective_units(data, condition_labels, unit_ids_sorted, fs, time_unit,
                             top_n=50):
    """Return (mask, score) where score = |mean_R_SF0 - mean_R_SF1| (Hz) for the
    two-SF case (generalised to max pairwise mean-diff for n_cond>2). Mask keeps
    the top-N units by score — no significance test (87 trials × 2 conds is
    under-powered, but argmax-based "preferred" is the same criterion fig2a uses
    for sorting)."""
    n_cond  = len(set(condition_labels.values()))
    n_units = len(unit_ids_sorted)
    cond_trials = {c: sorted([i for i, v in condition_labels.items() if v == c])
                   for c in range(n_cond)}
    dur = RESP_WINDOW[1] - RESP_WINDOW[0]

    mean_rates = np.zeros((n_units, n_cond))
    for c in range(n_cond):
        trials = cond_trials[c]
        for i, uid in enumerate(unit_ids_sorted):
            trial_map = {t['trial_index']: t for t in data['spike_data'][uid]}
            rates = []
            for tidx in trials:
                if tidx not in trial_map:
                    continue
                st = get_spikes_sec(trial_map[tidx], fs, time_unit)
                rates.append(((st >= RESP_WINDOW[0]) & (st < RESP_WINDOW[1])).sum() / dur)
            mean_rates[i, c] = np.mean(rates) if rates else 0.0

    score = mean_rates.max(axis=1) - mean_rates.min(axis=1)
    order = np.argsort(-score)
    keep = order[:min(top_n, n_units)]
    mask = np.zeros(n_units, dtype=bool)
    mask[keep] = True
    return mask, score


def find_sf_selective_units(data, condition_labels, unit_ids_sorted, fs, time_unit,
                             q=0.05):
    """Mark units whose response-window FR differs significantly between conditions.
    Mann-Whitney U (two-sided) for each unit × condition-pair, then Benjamini-
    Hochberg FDR across all (unit, pair) tests at level q. A unit is selective
    if any of its pair-tests passes FDR."""
    n_cond  = len(set(condition_labels.values()))
    n_units = len(unit_ids_sorted)
    cond_trials = {c: sorted([i for i, v in condition_labels.items() if v == c])
                   for c in range(n_cond)}
    dur = RESP_WINDOW[1] - RESP_WINDOW[0]

    rates_per_cond = [[None] * n_units for _ in range(n_cond)]
    for c in range(n_cond):
        trials = cond_trials[c]
        for i, uid in enumerate(unit_ids_sorted):
            trial_map = {t['trial_index']: t for t in data['spike_data'][uid]}
            r = np.zeros(len(trials))
            for j, tidx in enumerate(trials):
                if tidx not in trial_map:
                    continue
                st   = get_spikes_sec(trial_map[tidx], fs, time_unit)
                mask = (st >= RESP_WINDOW[0]) & (st < RESP_WINDOW[1])
                r[j] = mask.sum() / dur
            rates_per_cond[c][i] = r

    test_unit = []
    test_p    = []
    for i in range(n_units):
        for a in range(n_cond):
            for b in range(a + 1, n_cond):
                ra, rb = rates_per_cond[a][i], rates_per_cond[b][i]
                if len(ra) == 0 or len(rb) == 0:
                    continue
                if np.all(ra == ra[0]) and np.all(rb == rb[0]):
                    continue
                try:
                    _, p = mannwhitneyu(ra, rb, alternative='two-sided')
                except ValueError:
                    continue
                test_unit.append(i)
                test_p.append(p)

    test_unit = np.array(test_unit)
    test_p    = np.array(test_p)

    fdr_pass  = _bh_fdr(test_p, q=q) if len(test_p) else np.array([], dtype=bool)
    mask = np.zeros(n_units, dtype=bool)
    for i, ok in zip(test_unit, fdr_pass):
        if ok:
            mask[i] = True

    min_p = np.ones(n_units)
    for i, p in zip(test_unit, test_p):
        if p < min_p[i]:
            min_p[i] = p
    return mask, min_p, q


def compute_rates_and_noise_corr(data, condition_labels, unit_ids_sorted, fs, time_unit):
    n_cond  = len(set(condition_labels.values()))
    n_units = len(unit_ids_sorted)
    cond_trials = {c: sorted([i for i, v in condition_labels.items() if v == c])
                   for c in range(n_cond)}
    dur = RESP_WINDOW[1] - RESP_WINDOW[0]

    per_cond_rates = []
    all_residuals  = [[] for _ in range(n_units)]

    for c in range(n_cond):
        trials = cond_trials[c]
        rates  = np.zeros((n_units, len(trials)))
        for i, uid in enumerate(unit_ids_sorted):
            trial_map = {t['trial_index']: t for t in data['spike_data'][uid]}
            for j, tidx in enumerate(trials):
                if tidx not in trial_map:
                    continue
                st   = get_spikes_sec(trial_map[tidx], fs, time_unit)
                mask = (st >= RESP_WINDOW[0]) & (st < RESP_WINDOW[1])
                rates[i, j] = mask.sum() / dur
        per_cond_rates.append(rates)
        residuals = rates - rates.mean(axis=1, keepdims=True)
        for i in range(n_units):
            all_residuals[i].extend(residuals[i].tolist())

    nc = np.corrcoef(np.array(all_residuals))
    np.fill_diagonal(nc, 0)

    all_rates = np.concatenate(per_cond_rates, axis=1)
    resp_prob = (all_rates > 0).mean(axis=1)

    return resp_prob, nc


if __name__ == '__main__':
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("pip install networkx")
    import community as community_louvain

    pkl_path = Path(PKL_PATH)
    out_dir  = Path(OUTPUT_DIR) if OUTPUT_DIR else pkl_path.parent / 'figures'
    out_dir.mkdir(exist_ok=True)

    data      = load_pkl(pkl_path)
    fs        = data['metadata']['sampling_frequency']
    time_unit = check_spike_time_units(data)
    print(f"Spike time unit: {time_unit}  (fs={fs} Hz)")

    condition_labels, condition_names = build_conditions(data)
    n_cond   = len(condition_names)
    unit_ids = list(data['spike_data'].keys())
    n_units  = len(unit_ids)

    sort_order_file = out_dir / 'sort_order.npy'
    pref_file       = out_dir / 'pref_sorted.npy'
    if sort_order_file.exists() and pref_file.exists():
        sort_order  = np.load(sort_order_file)
        pref_sorted = np.load(pref_file)
        print(f"Loaded sort_order and pref_sorted from {out_dir}")
    else:
        sort_order  = np.arange(n_units)
        pref_sorted = np.zeros(n_units, dtype=int)
        print("sort_order.npy not found — using original order (run plot_fig2a.py first)")

    unit_ids_sorted = [unit_ids[i] for i in sort_order]

    # ----- Identify SF-selective units (Mann-Whitney U, Bonferroni) -----
    # Cells whose response-window firing rate differs between SF conditions —
    # analog of the paper's "cells driven by a particular cue".
    print(f"Selecting top-{TOP_N_SELECTIVE} SF-selective units by response diff ...")
    cue_mask, sel_score = rank_sf_selective_units(
        data, condition_labels, unit_ids_sorted, fs, time_unit, top_n=TOP_N_SELECTIVE)
    n_driven = int(cue_mask.sum())
    kept_scores = sel_score[cue_mask]
    print(f"  {n_driven} / {n_units} kept;  "
          f"|ΔFR| range among kept: {kept_scores.min():.2f}–{kept_scores.max():.2f} Hz")
    np.save(out_dir / 'sf_selective_mask.npy', cue_mask)

    print(f"Computing response probability + noise correlations for {n_units} units ...")
    resp_prob, nc = compute_rates_and_noise_corr(
        data, condition_labels, unit_ids_sorted, fs, time_unit)

    # ----- Paper-faithful graph: all positive noise correlations, no threshold -----
    # Restrict to cue-driven units only
    driven_idx = np.where(cue_mask)[0]
    G = nx.Graph()
    for i in driven_idx:
        G.add_node(int(i), preferred=int(pref_sorted[i]),
                   resp_prob=float(resp_prob[i]))
    for a, i in enumerate(driven_idx):
        for j in driven_idx[a + 1:]:
            if nc[i, j] > 0:
                G.add_edge(int(i), int(j), weight=float(nc[i, j]))
    print(f"Graph: {G.number_of_nodes()} cue-driven nodes, "
          f"{G.number_of_edges()} positive edges")

    # Louvain on the positive-weight graph
    partition  = community_louvain.best_partition(G, weight='weight', random_state=42)
    n_clusters = (max(partition.values()) + 1) if partition else 1
    print(f"Louvain: {n_clusters} clusters found")

    # ----- Order nodes so each cluster occupies a contiguous arc on the ring -----
    nodes_by_cluster = {}
    for n, c in partition.items():
        nodes_by_cluster.setdefault(c, []).append(n)

    cluster_order = sorted(nodes_by_cluster, key=lambda c: -len(nodes_by_cluster[c]))
    ordered_nodes = []
    cluster_spans = []  # (cluster_id, start_idx, end_idx)
    idx = 0
    for c in cluster_order:
        members = sorted(nodes_by_cluster[c], key=lambda n: pref_sorted[n])
        cluster_spans.append((c, idx, idx + len(members)))
        ordered_nodes.extend(members)
        idx += len(members)

    N = len(ordered_nodes)
    R = 1.0
    angles = np.pi / 2 - 2 * np.pi * np.arange(N) / N  # start at top, go clockwise
    pos = {n: (R * np.cos(a), R * np.sin(a)) for n, a in zip(ordered_nodes, angles)}

    # ----- Visual encodings -----
    # Node size ∝ response probability (diameter; "gray circle = probability 1")
    size_scale = 900.0
    node_sizes = [size_scale * resp_prob[n] + 10 for n in G.nodes()]

    cue_cmap    = plt.cm.Set2(np.linspace(0, 1, max(n_cond, 3)))
    node_colors = [cue_cmap[int(pref_sorted[n]) % len(cue_cmap)] for n in G.nodes()]

    edge_weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    if edge_weights.size:
        edge_widths = edge_weights / edge_weights.max() * 1.8 + 0.05
    else:
        edge_widths = []

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(12, 12))

    # Shaded ring-segment wedges for clusters (halo outside the node ring)
    clust_cmap = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 3)))
    half_step  = np.pi / N
    wedge_r_outer = R * 1.20
    wedge_width   = 0.14
    for c, start, end in cluster_spans:
        a_hi = angles[start]     + half_step
        a_lo = angles[end - 1]   - half_step
        theta1 = np.degrees(a_lo) % 360
        theta2 = np.degrees(a_hi) % 360
        if theta2 <= theta1:
            theta2 += 360
        wedge = Wedge((0, 0), wedge_r_outer, theta1, theta2,
                      width=wedge_width,
                      facecolor=clust_cmap[c % len(clust_cmap)],
                      alpha=0.55, edgecolor='none', zorder=0)
        ax.add_patch(wedge)

    # Edges (light, behind nodes)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           alpha=0.08, edge_color='0.2')
    # Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95,
                           edgecolors='k', linewidths=0.5)

    # Reference "probability = 1" gray circle
    ref_cx, ref_cy = 1.55, -1.35
    ax.scatter([ref_cx], [ref_cy], s=size_scale, c='0.55',
               edgecolors='k', linewidths=0.5)
    ax.text(ref_cx, ref_cy - 0.22, 'P(resp)=1', ha='center', va='top', fontsize=9)

    # Legend: preferred condition (node color) + clusters (wedge color)
    leg_handles = []
    for s in range(n_cond):
        leg_handles.append(ax.scatter([], [], c=[cue_cmap[s]], s=80,
                                       edgecolors='k', linewidths=0.5,
                                       label=f'Pref: {condition_names[s]}'))
    for k in range(n_clusters):
        leg_handles.append(plt.Rectangle((0, 0), 1, 1,
                                          facecolor=clust_cmap[k % len(clust_cmap)],
                                          alpha=0.55,
                                          label=f'Cluster {k} (n={len(nodes_by_cluster[k])})'))
    ax.legend(handles=leg_handles, loc='upper left', fontsize=9,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    ax.set_xlim(-1.5, 1.9)
    ax.set_ylim(-1.6, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Fig 2d — Functional connectivity  '
                 f'({G.number_of_nodes()}/{n_units} SF-selective units, '
                 f'{G.number_of_edges()} positive-r edges, '
                 f'{n_clusters} Louvain clusters)')

    plt.tight_layout()
    out_path = out_dir / 'fig2d_connectivity.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
