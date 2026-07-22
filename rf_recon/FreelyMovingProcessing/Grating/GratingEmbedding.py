"""
Unsupervised embedding analysis for Grating orientation neural data.

Mirrors the GratingLDA pipeline but uses PCA, UMAP, and t-SNE. Labels are
never used to fit the embeddings, only to color the projections and to
evaluate label structure post-hoc against null baselines:
  - label shuffle:   permute trial labels (random label-population assignment)
  - column shuffle:  permute trials independently per neuron (destroys
                     cross-neuron covariance while preserving marginals;
                     a stricter null asking whether structure requires
                     *coordinated* activity).

Beyond projection, the script also computes:
  - silhouette and k-NN label purity (real vs both nulls)
  - trustworthiness of each low-D embedding vs the high-D representation
  - HDBSCAN clustering on UMAP with ARI/NMI vs orientation labels
  - circular fit to PC1/PC2 class centroids (orientation is axial: period 180°)
  - polar population-vector plot
  - cross-SF projection (fit PCA on one SF, project the others)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    from sklearn.cluster import HDBSCAN as _SkHDBSCAN
    _HAS_HDBSCAN = True
except ImportError:
    try:
        import hdbscan as _hdbscan_pkg
        _HAS_HDBSCAN = True
        _SkHDBSCAN = None
    except ImportError:
        _HAS_HDBSCAN = False
        _SkHDBSCAN = None

from grating_utils import (
    load_neural_data,
    calculate_firing_rates,
    calculate_orientation_selectivity,
    plot_trial_distribution,
    resolve_data_path,
)


# =============================================================================
# UNSUPERVISED QUALITY METRICS
# =============================================================================

def _knn_label_purity(embedding, labels, k=5):
    """Fraction of each point's k nearest neighbors sharing its label."""
    n = len(labels)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    _, idx = nn.kneighbors(embedding)
    idx = idx[:, 1:]
    neighbor_labels = labels[idx]
    return float(np.mean(neighbor_labels == labels[:, None]))


def _shuffled_label_metrics(embedding, labels, n_shuffles=50, k=5, rng=None):
    """Silhouette and k-NN purity under random label permutations."""
    rng = rng or np.random.default_rng(42)
    sil = np.empty(n_shuffles)
    knn = np.empty(n_shuffles)
    for i in range(n_shuffles):
        permuted = rng.permutation(labels)
        sil[i] = (silhouette_score(embedding, permuted)
                  if len(np.unique(permuted)) > 1 else np.nan)
        knn[i] = _knn_label_purity(embedding, permuted, k=k)
    return sil, knn


def _column_shuffled_pca_metrics(X, labels, n_components, n_shuffles=20,
                                 k=5, rng=None):
    """
    Stricter null: shuffle each neuron's trials independently, re-fit PCA,
    score with the original labels. Tests whether label structure requires
    coordinated (across-neuron) activity rather than only the marginals.
    """
    rng = rng or np.random.default_rng(123)
    sil = np.empty(n_shuffles)
    knn = np.empty(n_shuffles)
    for i in range(n_shuffles):
        X_perm = np.empty_like(X)
        for j in range(X.shape[1]):
            X_perm[:, j] = X[rng.permutation(X.shape[0]), j]
        emb = PCA(n_components=n_components, random_state=42).fit_transform(X_perm)
        sil[i] = (silhouette_score(emb, labels)
                  if len(np.unique(labels)) > 1 else np.nan)
        knn[i] = _knn_label_purity(emb, labels, k=k)
    return sil, knn


def _label_structure_metrics(embedding, labels, X_highd, n_shuffles=50, k=5, rng=None):
    """Real metrics + trustworthiness + shuffled-label null."""
    sil_real = (float(silhouette_score(embedding, labels))
                if len(np.unique(labels)) > 1 else np.nan)
    knn_real = _knn_label_purity(embedding, labels, k=k)

    trust_k = min(k, len(labels) - 1)
    trust = float(trustworthiness(X_highd, embedding, n_neighbors=trust_k))

    sil_shuf, knn_shuf = _shuffled_label_metrics(embedding, labels,
                                                 n_shuffles=n_shuffles, k=k, rng=rng)
    return {
        'silhouette_real': sil_real,
        'silhouette_shuffled': sil_shuf,
        'knn_purity_real': knn_real,
        'knn_purity_shuffled': knn_shuf,
        'trustworthiness': trust,
        'k': k,
    }


def _hdbscan_cluster(embedding, min_cluster_size=5):
    """HDBSCAN clustering; returns (labels, ok). Falls back gracefully."""
    if not _HAS_HDBSCAN:
        return None, False
    mcs = max(2, min(min_cluster_size, len(embedding) // 4))
    try:
        if _SkHDBSCAN is not None:
            labels = _SkHDBSCAN(min_cluster_size=mcs).fit_predict(embedding)
        else:
            labels = _hdbscan_pkg.HDBSCAN(min_cluster_size=mcs).fit_predict(embedding)
    except Exception as e:
        print(f"  HDBSCAN failed: {e}")
        return None, False
    return labels, True


def _circular_fit(centroids_2d, label_angles_deg):
    """
    Fit a circle to PC1/PC2 class centroids and compare the angular
    ordering of centroids to the axial-angle ordering of labels (period 180°).

    Returns:
        center (a, b), radius r, radial RMS residual, circular correlation,
        and arrays of expected/observed angles (in radians, axial doubled).
    """
    x = centroids_2d[:, 0]
    y = centroids_2d[:, 1]
    # Algebraic circle fit: x² + y² = 2ax + 2by + c
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b_vec = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radial_rms = float(np.sqrt(np.mean((radii - r) ** 2)))

    observed = np.arctan2(y - cy, x - cx)         # in (-π, π]
    expected = np.deg2rad(2.0 * np.asarray(label_angles_deg, dtype=float))

    # Circular correlation (Fisher–Lee)
    o = observed - np.angle(np.mean(np.exp(1j * observed)))
    e = expected - np.angle(np.mean(np.exp(1j * expected)))
    num = np.sum(np.sin(o) * np.sin(e))
    den = np.sqrt(np.sum(np.sin(o) ** 2) * np.sum(np.sin(e) ** 2))
    circ_corr = float(num / den) if den > 0 else np.nan

    return {
        'center': (float(cx), float(cy)),
        'radius': r,
        'radial_rms': radial_rms,
        'circ_corr': circ_corr,
        'observed_angles': observed,
        'expected_angles': expected,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def perform_embedding_analysis(firing_rates, orientation_labels,
                               n_pca_components=3, umap_n_neighbors=15,
                               umap_min_dist=0.1, tsne_perplexity=30,
                               is_circular=True, random_state=42):
    """
    Fit PCA, UMAP, t-SNE on z-scored firing rates and compute unsupervised
    quality metrics + null baselines.
    """
    labels = np.array([str(x) for x in orientation_labels])
    unique = np.array(sorted(set(labels.tolist())))
    n_classes = len(unique)
    n_features = firing_rates.shape[1]

    print(f"\nUnsupervised Embedding Analysis:")
    print(f"  Classes: {n_classes} ({unique})")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(labels)}")

    min_trials = min(np.sum(labels == ori) for ori in unique)

    rng = np.random.default_rng(random_state)
    balanced_idx = np.concatenate([
        rng.choice(np.where(labels == ori)[0], size=int(min_trials), replace=False)
        for ori in unique
    ])
    balanced_idx = np.sort(balanced_idx)
    firing_rates = firing_rates[balanced_idx]
    labels = labels[balanced_idx]
    print(f"  Balanced to {int(min_trials)} trials/class → {len(labels)} total")

    scaler = StandardScaler()
    X = scaler.fit_transform(firing_rates)

    # PCA
    n_pca = min(n_pca_components, min(X.shape) - 1)
    pca_full_n = max(n_pca, min(10, min(X.shape) - 1))
    pca = PCA(n_components=pca_full_n, random_state=random_state)
    pca_full = pca.fit_transform(X)
    pca_embed = pca_full[:, :n_pca]
    print(f"  PCA components: {n_pca} (cum. var = {pca.explained_variance_ratio_[:n_pca].sum():.3f})")

    # UMAP
    umap_embed = None
    if _HAS_UMAP and len(labels) >= 4:
        n_neighbors = min(umap_n_neighbors, len(labels) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=umap_min_dist, random_state=random_state)
        umap_embed = reducer.fit_transform(X)
        print(f"  UMAP: n_neighbors={n_neighbors}, min_dist={umap_min_dist}")
    elif not _HAS_UMAP:
        print("  UMAP not installed (pip install umap-learn) — skipping.")

    # t-SNE
    perplexity = min(tsne_perplexity, max(2, (len(labels) - 1) // 3))
    tsne_embed = TSNE(n_components=2, perplexity=perplexity,
                      random_state=random_state, init='pca').fit_transform(X)
    print(f"  t-SNE: perplexity={perplexity}")

    # Metrics for each embedding
    pca_metrics = _label_structure_metrics(pca_embed, labels, X, rng=rng)
    print(f"  PCA  sil={pca_metrics['silhouette_real']:.3f}  "
          f"kNN={pca_metrics['knn_purity_real']:.3f}  "
          f"trust={pca_metrics['trustworthiness']:.3f}")

    umap_metrics = None
    if umap_embed is not None:
        umap_metrics = _label_structure_metrics(umap_embed, labels, X, rng=rng)
        print(f"  UMAP sil={umap_metrics['silhouette_real']:.3f}  "
              f"kNN={umap_metrics['knn_purity_real']:.3f}  "
              f"trust={umap_metrics['trustworthiness']:.3f}")

    tsne_metrics = _label_structure_metrics(tsne_embed, labels, X, rng=rng)
    print(f"  tSNE sil={tsne_metrics['silhouette_real']:.3f}  "
          f"kNN={tsne_metrics['knn_purity_real']:.3f}  "
          f"trust={tsne_metrics['trustworthiness']:.3f}")

    # Column-shuffle null (slower; fewer iterations)
    print("  Column-shuffle null (PCA, 20 iters)…")
    col_sil, col_knn = _column_shuffled_pca_metrics(X, labels, n_pca, rng=rng)
    print(f"    sil  shuf cols: {col_sil.mean():.3f} ± {col_sil.std():.3f}")
    print(f"    kNN  shuf cols: {col_knn.mean():.3f} ± {col_knn.std():.3f}")

    # HDBSCAN on UMAP (or PCA-2D fallback)
    cluster_target = umap_embed if umap_embed is not None else pca_embed[:, :2]
    cluster_target_name = 'UMAP' if umap_embed is not None else 'PCA-2D'
    hdb_labels, hdb_ok = _hdbscan_cluster(cluster_target)
    hdb_info = None
    if hdb_ok:
        valid = hdb_labels >= 0
        n_clusters = int(len(set(hdb_labels[valid])))
        n_noise = int(np.sum(~valid))
        ari = float(adjusted_rand_score(labels, hdb_labels))
        nmi = float(normalized_mutual_info_score(labels, hdb_labels))
        hdb_info = {
            'labels': hdb_labels,
            'target': cluster_target_name,
            'embedding': cluster_target,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'ari': ari,
            'nmi': nmi,
        }
        print(f"  HDBSCAN on {cluster_target_name}: "
              f"{n_clusters} clusters, {n_noise} noise, ARI={ari:.3f}, NMI={nmi:.3f}")

    # Circular structure (axial: orientations period 180°)
    circ_info = None
    pv_polar = None
    if is_circular:
        try:
            label_angles = np.array([float(lbl) for lbl in unique])
        except ValueError:
            label_angles = None
        if label_angles is not None and len(unique) >= 3:
            centroids = np.array([
                pca_embed[labels == ori][:, :2].mean(axis=0) for ori in unique
            ])
            circ_info = _circular_fit(centroids, label_angles)
            print(f"  Circular fit: r={circ_info['radius']:.3f}, "
                  f"radial RMS={circ_info['radial_rms']:.3f}, "
                  f"circ_corr={circ_info['circ_corr']:.3f}")

            # Population vector at axial angle 2θ
            pv = np.array([
                pca_embed[labels == ori][:, :2].mean(axis=0) for ori in unique
            ])
            pv_polar = {
                'angles_rad': np.deg2rad(2.0 * label_angles),
                'magnitudes': np.linalg.norm(pv - pv.mean(axis=0), axis=1),
                'label_angles_deg': label_angles,
            }

    chance_purity = 1.0 / n_classes

    return {
        'pca_model': pca,
        'scaler': scaler,
        'pca_embedding': pca_embed,
        'umap_embedding': umap_embed,
        'tsne_embedding': tsne_embed,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'orientation_labels': labels,
        'unique_orientations': unique,
        'n_components': n_pca,
        'pca_metrics': pca_metrics,
        'umap_metrics': umap_metrics,
        'tsne_metrics': tsne_metrics,
        'column_shuffle_sil': col_sil,
        'column_shuffle_knn': col_knn,
        'hdbscan': hdb_info,
        'circular_fit': circ_info,
        'pv_polar': pv_polar,
        'chance_purity': chance_purity,
        'X_scaled': X,
        'is_circular': is_circular,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_figure(results, unit_ids, trial_info, save_path=None,
                           label_suffix='°'):
    """Create unsupervised embedding visualization (3×4 layout)."""
    plt.style.use('default')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(26, 20))

    labels = results['orientation_labels']
    unique_ori = results['unique_orientations']
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_ori) + 1)[:-1])

    _plot_3d_scatter(fig, results['pca_embedding'], labels, unique_ori, colors,
                     title='PCA 3D', axis_labels=('PC1', 'PC2', 'PC3'),
                     label_suffix=label_suffix, subplot_pos=(3, 4, 1))
    ax2d = _plot_2d_scatter(fig, results['pca_embedding'], labels, unique_ori, colors,
                            title='PCA 2D', axis_labels=('PC1', 'PC2'),
                            label_suffix=label_suffix, subplot_pos=(3, 4, 2))

    if results['umap_embedding'] is not None:
        _plot_2d_scatter(fig, results['umap_embedding'], labels, unique_ori, colors,
                        title='UMAP 2D', axis_labels=('UMAP1', 'UMAP2'),
                        label_suffix=label_suffix, subplot_pos=(3, 4, 3))
    else:
        ax = fig.add_subplot(3, 4, 3)
        ax.text(0.5, 0.5, 'UMAP unavailable',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')

    _plot_2d_scatter(fig, results['tsne_embedding'], labels, unique_ori, colors,
                    title=f't-SNE 2D', axis_labels=('tSNE1', 'tSNE2'),
                    label_suffix=label_suffix, subplot_pos=(3, 4, 4))

    _plot_explained_variance(fig, results, subplot_pos=(3, 4, 5))
    _plot_top_loadings(fig, results, unit_ids, subplot_pos=(3, 4, 6))
    _plot_hdbscan(fig, results, subplot_pos=(3, 4, 7))

    if results['is_circular'] and results['circular_fit'] is not None:
        _plot_circular_fit(fig, results, unique_ori, colors,
                           label_suffix=label_suffix, subplot_pos=(3, 4, 8))
    else:
        ax = fig.add_subplot(3, 4, 8)
        ax.text(0.5, 0.5, 'Circular fit\nN/A (non-circular labels)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')

    if results['is_circular'] and results['pv_polar'] is not None:
        _plot_pv_polar(fig, results, unique_ori, colors,
                       label_suffix=label_suffix, subplot_pos=(3, 4, 9))
    else:
        ax = fig.add_subplot(3, 4, 9)
        ax.text(0.5, 0.5, 'Polar PV\nN/A (non-circular labels)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')

    _plot_structure_metrics(fig, results, subplot_pos=(3, 4, 10))

    ax_td = plot_trial_distribution(fig, trial_info, unique_ori, colors,
                                    label_suffix, subplot_pos=(3, 4, 11))
    if ax_td is not None:
        ax_td.grid(False)

    _plot_summary_text(fig, results, labels, unit_ids, trial_info,
                       label_suffix, subplot_pos=(3, 4, 12))

    handles, leg_labels = ax2d.get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels,
                   loc='center', bbox_to_anchor=(0.5, 0.97),
                   ncol=min(len(handles), 10),
                   frameon=False, fontsize=15,
                   handletextpad=0.5, columnspacing=1.4)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_svg = save_path.suffix.lower() == '.svg'
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    transparent=is_svg,
                    facecolor='none' if is_svg else 'white')
        print(f"Saved figure to: {save_path}")

    return fig


def _plot_3d_scatter(fig, data, labels, orientations, colors,
                     title, axis_labels, label_suffix='°', subplot_pos=(3, 4, 1)):
    ax = fig.add_subplot(*subplot_pos, projection='3d')

    if data.shape[1] >= 3:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                       c=[colors[i]], label=f'{ori}{label_suffix}',
                       alpha=0.85, s=60, edgecolors='none')
        ax.set_xlabel(axis_labels[0], fontsize=18, fontweight='bold', labelpad=8)
        ax.set_ylabel(axis_labels[1], fontsize=18, fontweight='bold', labelpad=8)
        ax.set_zlabel(axis_labels[2], fontsize=18, fontweight='bold', labelpad=8)
    else:
        ax.text(0.5, 0.5, 0.5, 'Need ≥3 components',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

    ax.set_title(title, fontsize=22, fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=12)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor('lightgray')
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    return ax


def _plot_2d_scatter(fig, data, labels, orientations, colors,
                     title, axis_labels, label_suffix='°', subplot_pos=(3, 4, 2)):
    ax = fig.add_subplot(*subplot_pos)

    for i, ori in enumerate(orientations):
        mask = labels == ori
        ax.scatter(data[mask, 0], data[mask, 1],
                   c=[colors[i]], label=f'{ori}{label_suffix}',
                   alpha=0.85, s=60, edgecolors='none')
    ax.set_xlabel(axis_labels[0], fontsize=20, fontweight='bold')
    ax.set_ylabel(axis_labels[1], fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='bold', pad=10)

    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(labelsize=16, width=2.0, length=6)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return ax


def _plot_explained_variance(fig, results, subplot_pos=(3, 4, 5)):
    ax = fig.add_subplot(*subplot_pos)
    evr = results['explained_variance_ratio']
    n_show = min(10, len(evr))
    idx = np.arange(1, n_show + 1)

    ax.bar(idx, evr[:n_show], color='#4C72B0', alpha=0.85,
           edgecolor='black', linewidth=1.2, label='Per PC')
    ax2 = ax.twinx()
    ax2.plot(idx, np.cumsum(evr[:n_show]), 'o-', color='#E69F00',
             linewidth=2.5, markersize=8, label='Cumulative')
    ax2.set_ylabel('Cumulative', fontsize=16, fontweight='bold', color='#E69F00')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelsize=12, colors='#E69F00')

    ax.set_xlabel('Principal Component', fontsize=20, fontweight='bold')
    ax.set_ylabel('Var. Explained', fontsize=20, fontweight='bold')
    ax.set_title('PCA Explained Variance', fontsize=22, fontweight='bold', pad=10)
    ax.set_xticks(idx)
    ax.tick_params(axis='both', labelsize=14, width=2.0, length=6)
    for spine in ('top',):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom', 'right'):
        ax.spines[spine].set_linewidth(2.0)


def _plot_top_loadings(fig, results, unit_ids, subplot_pos=(3, 4, 6)):
    ax = fig.add_subplot(*subplot_pos)
    pca = results['pca_model']
    n_comp = results['n_components']
    loadings = np.abs(pca.components_[:n_comp])
    importance = loadings.mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:15]

    ax.barh(range(len(top_idx)), importance[top_idx],
            color='#E69F00', edgecolor='black', linewidth=1.0)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([unit_ids[i].split('_')[-1] for i in top_idx], fontsize=13)
    ax.set_xlabel('Mean |PC loading|', fontsize=20, fontweight='bold')
    ax.set_title('Top Units (PC loadings)', fontsize=22, fontweight='bold', pad=10)
    ax.invert_yaxis()
    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(axis='x', labelsize=14, width=2.0, length=6)


def _plot_hdbscan(fig, results, subplot_pos=(3, 4, 7)):
    ax = fig.add_subplot(*subplot_pos)
    hdb = results['hdbscan']
    if hdb is None:
        ax.text(0.5, 0.5, 'HDBSCAN unavailable',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        return

    emb = hdb['embedding']
    clust = hdb['labels']
    unique_c = sorted(set(clust))
    cmap = plt.cm.tab10
    for i, c in enumerate(unique_c):
        mask = clust == c
        if c == -1:
            ax.scatter(emb[mask, 0], emb[mask, 1], c='lightgray', s=35,
                       alpha=0.7, label='noise', edgecolors='none')
        else:
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(i % 10)], s=55,
                       alpha=0.85, label=f'C{c}', edgecolors='none')

    ax.set_title(
        f"HDBSCAN on {hdb['target']}\n"
        f"k={hdb['n_clusters']}, noise={hdb['n_noise']}, "
        f"ARI={hdb['ari']:.2f}, NMI={hdb['nmi']:.2f}",
        fontsize=18, fontweight='bold', pad=10)
    ax.set_xlabel(f"{hdb['target']}1", fontsize=18, fontweight='bold')
    ax.set_ylabel(f"{hdb['target']}2", fontsize=18, fontweight='bold')
    ax.legend(fontsize=11, frameon=False, ncol=2, loc='best')
    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(labelsize=13, width=2.0, length=6)


def _plot_circular_fit(fig, results, orientations, colors,
                       label_suffix='°', subplot_pos=(3, 4, 8)):
    ax = fig.add_subplot(*subplot_pos)
    cf = results['circular_fit']
    pca_embed = results['pca_embedding']
    labels = results['orientation_labels']

    ax.scatter(pca_embed[:, 0], pca_embed[:, 1], c='lightgray',
               s=20, alpha=0.4, edgecolors='none', zorder=1)

    centroids = np.array([
        pca_embed[labels == ori][:, :2].mean(axis=0) for ori in orientations
    ])
    for i, ori in enumerate(orientations):
        ax.scatter(centroids[i, 0], centroids[i, 1], c=[colors[i]],
                   s=220, edgecolors='black', linewidth=1.8, zorder=3,
                   label=f'{ori}{label_suffix}')

    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = cf['center']
    r = cf['radius']
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
            'k--', linewidth=1.8, alpha=0.7, zorder=2)
    ax.scatter([cx], [cy], marker='+', c='black', s=200, linewidth=2.5, zorder=4)

    ax.set_title(f"PC1/PC2 circle fit\n"
                 f"r={r:.2f}, radial RMS={cf['radial_rms']:.2f}, "
                 f"circ_corr={cf['circ_corr']:.2f}",
                 fontsize=17, fontweight='bold', pad=10)
    ax.set_xlabel('PC1', fontsize=18, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=18, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(labelsize=13, width=2.0, length=6)


def _plot_pv_polar(fig, results, orientations, colors,
                   label_suffix='°', subplot_pos=(3, 4, 9)):
    ax = fig.add_subplot(*subplot_pos, projection='polar')
    pv = results['pv_polar']
    angles = pv['angles_rad']
    mags = pv['magnitudes']

    for i, (theta, m, ori) in enumerate(zip(angles, mags, orientations)):
        ax.plot([0, theta], [0, m], '-', color=colors[i], linewidth=2.8)
        ax.scatter([theta], [m], c=[colors[i]], s=130,
                   edgecolors='black', linewidth=1.2, zorder=3,
                   label=f'{ori}{label_suffix}')

    order = np.argsort(angles)
    closed_theta = np.append(angles[order], angles[order][0])
    closed_mag = np.append(mags[order], mags[order][0])
    ax.plot(closed_theta, closed_mag, 'k-', linewidth=1.2, alpha=0.5)

    ax.set_title('Population vector\n(PC1/PC2 centroids, axial 2θ)',
                 fontsize=17, fontweight='bold', pad=14)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      [f'{a/2:g}°' for a in np.arange(0, 360, 45)],
                      fontsize=13)
    ax.set_yticklabels([])
    ax.grid(True, linewidth=1.2, alpha=0.5)


def _plot_structure_metrics(fig, results, subplot_pos=(3, 4, 10)):
    """Real vs shuffled silhouette / kNN purity, plus trustworthiness."""
    ax = fig.add_subplot(*subplot_pos)

    embeds = [('PCA', results['pca_metrics'])]
    if results['umap_metrics'] is not None:
        embeds.append(('UMAP', results['umap_metrics']))
    embeds.append(('tSNE', results['tsne_metrics']))

    rows = []
    for emb_name, m in embeds:
        rows.append((f'{emb_name}\nSil', m['silhouette_real'],
                     m['silhouette_shuffled'].mean(),
                     m['silhouette_shuffled'].std()))
        rows.append((f'{emb_name}\nkNN', m['knn_purity_real'],
                     m['knn_purity_shuffled'].mean(),
                     m['knn_purity_shuffled'].std()))

    # PCA column-shuffle null (stricter)
    rows.append(('PCA-Sil\ncol-shuf', results['pca_metrics']['silhouette_real'],
                 results['column_shuffle_sil'].mean(),
                 results['column_shuffle_sil'].std()))
    rows.append(('PCA-kNN\ncol-shuf', results['pca_metrics']['knn_purity_real'],
                 results['column_shuffle_knn'].mean(),
                 results['column_shuffle_knn'].std()))

    x = np.arange(len(rows))
    width = 0.4
    real_vals = [r[1] for r in rows]
    shuf_means = [r[2] for r in rows]
    shuf_stds = [r[3] for r in rows]
    tick_labels = [r[0] for r in rows]

    ax.bar(x - width / 2, real_vals, width, color='#4C72B0',
           edgecolor='black', linewidth=1.2, label='Real')
    ax.bar(x + width / 2, shuf_means, width, yerr=shuf_stds,
           color='#BBBBBB', edgecolor='black', linewidth=1.2, capsize=4,
           error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
           label='Null')

    ax.axhline(results['chance_purity'], color='black', linestyle='--',
               linewidth=1.2, alpha=0.6,
               label=f'kNN chance ({results["chance_purity"]:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax.set_title('Label-structure vs nulls', fontsize=20, fontweight='bold', pad=10)
    ax.tick_params(axis='y', labelsize=13, width=2.0, length=6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(2.0)
    ax.legend(fontsize=11, frameon=False, loc='best')


def _plot_summary_text(fig, results, labels, unit_ids, trial_info,
                       label_suffix='°', subplot_pos=(3, 4, 12)):
    ax = fig.add_subplot(*subplot_pos)
    ax.axis('off')

    exp_params = trial_info.get('experiment_parameters', {})
    unique_classes = results['unique_orientations']
    cum_var = float(np.cumsum(results['explained_variance_ratio'])
                    [results['n_components'] - 1])

    pca_m = results['pca_metrics']
    tsne_m = results['tsne_metrics']

    def _fmt(m):
        return (f"sil={m['silhouette_real']:.2f} (shuf {m['silhouette_shuffled'].mean():.2f}), "
                f"kNN={m['knn_purity_real']:.2f} (shuf {m['knn_purity_shuffled'].mean():.2f}), "
                f"trust={m['trustworthiness']:.2f}")

    umap_line = ''
    if results['umap_metrics'] is not None:
        umap_line = f"    UMAP: {_fmt(results['umap_metrics'])}\n"

    hdb_line = ''
    if results['hdbscan'] is not None:
        h = results['hdbscan']
        hdb_line = (f"    HDBSCAN on {h['target']}: {h['n_clusters']} clusters, "
                    f"{h['n_noise']} noise, ARI={h['ari']:.2f}, NMI={h['nmi']:.2f}\n")

    circ_line = ''
    if results['circular_fit'] is not None:
        cf = results['circular_fit']
        circ_line = (f"    Circle fit (PC1/PC2): r={cf['radius']:.2f}, "
                     f"radial RMS={cf['radial_rms']:.2f}, circ_corr={cf['circ_corr']:.2f}\n")

    summary = f"""
    Unsupervised Summary

    PCA cum. var ({results['n_components']} PCs): {cum_var:.3f}
    PCA:  {_fmt(pca_m)}
{umap_line}    tSNE: {_fmt(tsne_m)}
    PCA col-shuf: sil={results['column_shuffle_sil'].mean():.2f}±{results['column_shuffle_sil'].std():.2f},
                  kNN={results['column_shuffle_knn'].mean():.2f}±{results['column_shuffle_knn'].std():.2f}
{hdb_line}{circ_line}    kNN chance: {results['chance_purity']:.3f}

    Experiment Info:
    • Trials: {len(labels)}    • Classes: {len(unique_classes)}    • Units: {len(unit_ids)}
    • Stim: {exp_params.get('stimulus_duration', 'N/A')}s | ITI: {exp_params.get('iti_duration', 'N/A')}s
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                      alpha=0.85))
    return ax


# =============================================================================
# CROSS-SF PROJECTION
# =============================================================================

def cross_sf_projection_analysis(firing_rates, orientation_labels, sf_labels,
                                 unit_ids, save_path=None, random_state=42):
    """
    Fit PCA on each SF independently, project all SFs into that same PC space.
    Asks whether the orientation manifold generalizes across spatial frequencies.

    Produces a (n_sf × n_sf) grid: row i = PCA fit on SF i, columns = data
    projected from each SF. The diagonal is the within-SF embedding.
    """
    unique_sfs = sorted(set(sf_labels.tolist()))
    if len(unique_sfs) < 2:
        return None

    print(f"\n{'#'*60}\n# CROSS-SF PROJECTION ({len(unique_sfs)} SFs)\n{'#'*60}")

    unique_ori = np.array(sorted(set(str(x) for x in orientation_labels)))
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_ori) + 1)[:-1])

    n_sf = len(unique_sfs)
    fig, axes = plt.subplots(n_sf, n_sf, figsize=(5 * n_sf, 5 * n_sf),
                             squeeze=False)

    scaler_global = StandardScaler().fit(firing_rates)
    X_all = scaler_global.transform(firing_rates)
    labels_str = np.array([str(x) for x in orientation_labels])

    cross_metrics = np.full((n_sf, n_sf), np.nan)

    for i, sf_fit in enumerate(unique_sfs):
        fit_mask = sf_labels == sf_fit
        X_fit = X_all[fit_mask]
        labels_fit = labels_str[fit_mask]
        pca = PCA(n_components=min(3, min(X_fit.shape) - 1),
                  random_state=random_state).fit(X_fit)

        for j, sf_proj in enumerate(unique_sfs):
            ax = axes[i, j]
            proj_mask = sf_labels == sf_proj
            X_proj = X_all[proj_mask]
            labels_proj = labels_str[proj_mask]
            Z = pca.transform(X_proj)

            for k, ori in enumerate(unique_ori):
                m = labels_proj == ori
                if not m.any():
                    continue
                ax.scatter(Z[m, 0], Z[m, 1], c=[colors[k]],
                           s=50, alpha=0.8, edgecolors='none',
                           label=f'{ori}°' if (i == 0 and j == 0) else None)

            if len(np.unique(labels_proj)) > 1 and Z.shape[1] >= 2:
                sil = silhouette_score(Z[:, :2], labels_proj)
                knn = _knn_label_purity(Z[:, :2], labels_proj, k=5)
                cross_metrics[i, j] = knn
                ax.set_title(f"fit={sf_fit}→proj={sf_proj}\n"
                             f"sil={sil:.2f}, kNN={knn:.2f}",
                             fontsize=13, fontweight='bold')
            else:
                ax.set_title(f"fit={sf_fit}→proj={sf_proj}",
                             fontsize=13, fontweight='bold')

            ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
            ax.tick_params(labelsize=10)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)

    # Build a single shared legend at the top
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.98), ncol=min(len(handles), 10),
                   frameon=False, fontsize=13)

    fig.suptitle("Cross-SF PCA projection (rows: fit SF, cols: projected SF)",
                 fontsize=20, fontweight='bold', y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    print("Cross-SF kNN purity (rows=fit, cols=proj):")
    header = "        " + "  ".join([f"{sf:>6}" for sf in unique_sfs])
    print(header)
    for i, sf_fit in enumerate(unique_sfs):
        row = "  ".join([f"{cross_metrics[i, j]:6.3f}"
                        if not np.isnan(cross_metrics[i, j]) else "   nan"
                        for j in range(n_sf)])
        print(f"  {sf_fit:>5}: {row}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_svg = save_path.suffix.lower() == '.svg'
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    transparent=is_svg,
                    facecolor='none' if is_svg else 'white')
        print(f"Saved cross-SF figure to: {save_path}")

    return {'cross_knn': cross_metrics, 'sfs': unique_sfs, 'figure': fig}


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.07, 0.16), save_plots=True, output_path=None):
    """Complete unsupervised embedding pipeline."""
    data = load_neural_data(data_path)
    firing_rates, orientation_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )

    if len(orientation_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")

    unique_sfs = trial_info['unique_spatial_freqs']
    sf_labels = trial_info['spatial_freq_labels']
    all_results = []

    print(f"\n{'#'*60}")
    print(f"# ORIENTATION EMBEDDING PER SF ({len(unique_sfs)} SF(s))")
    print(f"{'#'*60}")

    for sf in unique_sfs:
        if sf is None:
            fr_sf, labels_sf, sf_tag, sf_display = firing_rates, orientation_labels, '', 'all SF'
        else:
            sf_mask = sf_labels == sf
            fr_sf = firing_rates[sf_mask]
            labels_sf = orientation_labels[sf_mask]
            sf_tag = f'_sf{sf}'
            sf_display = f'SF={sf} cpd'

        print(f"\n{'='*60}\nEmbedding {sf_display}  ({len(labels_sf)} trials)\n{'='*60}")

        if len(labels_sf) == 0:
            continue
        unique_ori_sf = sorted(set(labels_sf.tolist()))
        if len(unique_ori_sf) < 2:
            print(f"  Only 1 orientation for {sf_display} — skipping.")
            continue

        trial_info_sf = {
            'unique_orientations': unique_ori_sf,
            'experiment_parameters': trial_info['experiment_parameters'],
            'n_trials_per_orientation': {
                str(ori): int(np.sum(labels_sf == ori)) for ori in unique_ori_sf
            },
        }

        results = perform_embedding_analysis(fr_sf, labels_sf, is_circular=True)

        fig_path = None
        if save_plots:
            base = Path(output_path) if output_path else Path(data_path).with_suffix('')
            fig_path = Path(str(base) + sf_tag + '.embedding_analysis.png')
        fig = create_analysis_figure(results, unit_ids, trial_info_sf,
                                     save_path=fig_path)
        fig.suptitle(f"Unsupervised Embedding — {sf_display}",
                     fontsize=24, fontweight='bold', y=1.0)

        print(f"\nCalculating orientation selectivity for {sf_display}...")
        calculate_orientation_selectivity(unit_ids, labels_sf, fr_sf)

        all_results.append((results, fr_sf, labels_sf, unit_ids))

    if sf_labels is not None and len(unique_sfs) > 1:
        unique_oris = sorted(set(orientation_labels.tolist()))
        print(f"\n{'#'*60}")
        print(f"# SF EMBEDDING PER ORIENTATION ({len(unique_oris)} orientations)")
        print(f"{'#'*60}")

        for ori in unique_oris:
            ori_mask = orientation_labels == ori
            fr_ori = firing_rates[ori_mask]
            sf_ori = sf_labels[ori_mask]
            unique_sf_ori = sorted(set(sf_ori.tolist()))
            if len(unique_sf_ori) < 2:
                continue

            print(f"\n{'='*60}\nSF embedding — orientation={ori}°"
                  f"  ({len(sf_ori)} trials)\n{'='*60}")

            trial_info_ori = {
                'unique_orientations': unique_sf_ori,
                'experiment_parameters': trial_info['experiment_parameters'],
                'n_trials_per_orientation': {
                    str(sf): int(np.sum(sf_ori == sf)) for sf in unique_sf_ori
                },
            }
            results = perform_embedding_analysis(fr_ori, sf_ori, is_circular=False)

            fig_path = None
            if save_plots:
                base = Path(output_path) if output_path else Path(data_path).with_suffix('')
                fig_path = Path(str(base) + f'_ori{ori}.sf_embedding.png')
            fig = create_analysis_figure(results, unit_ids, trial_info_ori,
                                         save_path=fig_path,
                                         label_suffix=' cpd')
            fig.suptitle(f"Unsupervised Embedding — Orientation={ori}°",
                         fontsize=24, fontweight='bold', y=1.0)

            all_results.append((results, fr_ori, sf_ori, unit_ids))

        # Cross-SF projection (orientation manifold generalization)
        fig_path = None
        if save_plots:
            base = Path(output_path) if output_path else Path(data_path).with_suffix('')
            fig_path = Path(str(base) + '.cross_sf_projection.png')
        cross_sf_projection_analysis(firing_rates, orientation_labels, sf_labels,
                                     unit_ids, save_path=fig_path)

    return all_results


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = resolve_data_path()

    try:
        all_results = run_analysis(
            data_path=DATA_PATH,
            time_window=(0.05, 1.5),
            save_plots=True
        )
        print(f"\nAnalysis complete! ({len(all_results)} group(s))")

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
