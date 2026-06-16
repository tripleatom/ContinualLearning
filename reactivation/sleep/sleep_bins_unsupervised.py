"""
Plot every bin of each sleep block in an unsupervised low-dimensional
embedding, colored by population firing rate.

For each sleep block in params.sleep_blocks:
  1. Bin all units in the sleep pkl over [start_sec, end_sec] at BIN_SIZE_MS.
  2. Standardize features (per-unit z-score using the sleep period itself).
  3. Project bins to 2D with PCA, t-SNE, UMAP, Isomap, PHATE, NMF
     (UMAP/PHATE require optional installs).
  4. Color each point by mean firing rate across units (Hz).
  5. Save a 2x3 multi-method PNG per sleep block.
"""

import sys
import pickle
from datetime import datetime
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'reactivation' / 'VStimOnDecoding'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, Isomap

try:
    import umap  # umap-learn
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

try:
    import phate  # phate
    _PHATE_AVAILABLE = True
except ImportError:
    _PHATE_AVAILABLE = False

from decode_utils import bin_spikes

from params import sleep_blocks, random_state


# -------- configuration --------
BIN_SIZE_MS         = 200      # sleep-bin size for the embedding
TSNE_PERPLEXITY     = 30
TSNE_MAX_SAMPLES    = 5000     # downsample for t-SNE if too many bins
UMAP_N_NEIGHBORS    = 30
UMAP_MIN_DIST       = 0.1
UMAP_MAX_SAMPLES    = 20000    # UMAP is faster than t-SNE; allow more points
ISOMAP_N_NEIGHBORS  = 15
ISOMAP_MAX_SAMPLES  = 5000     # Isomap O(n^2) memory
PHATE_KNN           = 5
PHATE_MAX_SAMPLES   = 10000
NMF_MAX_ITER        = 500
FIRING_RATE_LOG     = True     # log-color scale (pop rate spans many decades)
OUT_SUBDIR          = "sleep_unsupervised_bins"


def _load_sleep_bins(sleep_pkl_path, start_sec, end_sec, bin_size_sec):
    with open(sleep_pkl_path, "rb") as f:
        data = pickle.load(f)
    spike_data = data["spike_data"]

    if end_sec is None:
        end_sec = float(data.get("window", {}).get("window_duration_sec", 0.0))
    if start_sec is None:
        start_sec = 0.0
    if end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be > start_sec ({start_sec}).")

    n_bins = int(np.floor((end_sec - start_sec) / bin_size_sec))
    if n_bins < 2:
        raise ValueError("Sleep interval shorter than two bins of the chosen size.")
    edges = start_sec + np.arange(n_bins + 1) * bin_size_sec
    X, units = bin_spikes(spike_data, edges, bin_size_sec)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return X, centers, units, float(start_sec), float(end_sec)


def _zscore_features(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = np.maximum(X.std(axis=0, keepdims=True), 1e-9)
    return (X - mu) / sd


def _project_pca(Xz):
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(Xz), pca.explained_variance_ratio_


def _project_tsne(Xz, rng):
    n = Xz.shape[0]
    if n > TSNE_MAX_SAMPLES:
        sub_idx = rng.choice(n, size=TSNE_MAX_SAMPLES, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n)
    perplexity = max(5.0, min(TSNE_PERPLEXITY, (len(sub_idx) - 1) / 3.0))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(Xz[sub_idx])
    return coords, sub_idx, perplexity


def _project_umap(Xz, rng):
    if not _UMAP_AVAILABLE:
        return None, None, None
    n = Xz.shape[0]
    if n > UMAP_MAX_SAMPLES:
        sub_idx = rng.choice(n, size=UMAP_MAX_SAMPLES, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n)
    n_neighbors = max(2, min(UMAP_N_NEIGHBORS, len(sub_idx) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        random_state=random_state,
    )
    coords = reducer.fit_transform(Xz[sub_idx])
    return coords, sub_idx, n_neighbors


def _project_isomap(Xz, rng):
    n = Xz.shape[0]
    if n > ISOMAP_MAX_SAMPLES:
        sub_idx = rng.choice(n, size=ISOMAP_MAX_SAMPLES, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n)
    n_neighbors = max(2, min(ISOMAP_N_NEIGHBORS, len(sub_idx) - 1))
    coords = Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(Xz[sub_idx])
    return coords, sub_idx, n_neighbors


def _project_phate(X_input, rng):
    """PHATE works directly on the (non-negative) firing-rate matrix."""
    if not _PHATE_AVAILABLE:
        return None, None, None
    n = X_input.shape[0]
    if n > PHATE_MAX_SAMPLES:
        sub_idx = rng.choice(n, size=PHATE_MAX_SAMPLES, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n)
    knn = max(2, min(PHATE_KNN, len(sub_idx) - 1))
    op = phate.PHATE(n_components=2, knn=knn, random_state=random_state, verbose=0)
    coords = op.fit_transform(X_input[sub_idx])
    return coords, sub_idx, knn


def _project_nmf(X_rate):
    """NMF on the raw non-negative firing-rate matrix."""
    if X_rate.min() < 0:
        X_rate = np.clip(X_rate, 0, None)
    model = NMF(n_components=2, init="nndsvda", max_iter=NMF_MAX_ITER,
                random_state=random_state)
    coords = model.fit_transform(X_rate)
    return coords, model.reconstruction_err_


def _firing_rate_scatter(ax, coords, rate, title, axis_label):
    norm = LogNorm(vmin=max(rate.min(), 1e-3), vmax=max(rate.max(), 1e-3)) \
        if FIRING_RATE_LOG and rate.max() > 0 else None
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=rate, cmap="viridis", norm=norm,
        s=6, alpha=0.7, linewidths=0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.spines[["top", "right"]].set_visible(False)
    return sc


def _stamp_figure(fig, label, pkl_path, start_sec, end_sec, X,
                  tsne_n, perplexity, umap_n, umap_neighbors,
                  isomap_n, isomap_neighbors,
                  phate_n, phate_knn, nmf_err):
    umap_str = (
        f"umap_n={umap_n} | umap_neighbors={umap_neighbors} | "
        f"umap_min_dist={UMAP_MIN_DIST} | umap_max={UMAP_MAX_SAMPLES}"
        if _UMAP_AVAILABLE else "umap=unavailable (pip install umap-learn)"
    )
    phate_str = (
        f"phate_n={phate_n} | phate_knn={phate_knn} | phate_max={PHATE_MAX_SAMPLES}"
        if _PHATE_AVAILABLE else "phate=unavailable (pip install phate)"
    )
    info = (
        f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} | "
        f"script={Path(__file__).name} | block={label}\n"
        f"sleep_pkl={Path(pkl_path).name} | interval={start_sec:.3f}-{end_sec:.3f}s | "
        f"bin_ms={BIN_SIZE_MS} | n_bins={X.shape[0]} | n_units={X.shape[1]}\n"
        f"tsne_perplexity={perplexity:.2f} | tsne_n={tsne_n} | tsne_max={TSNE_MAX_SAMPLES} | "
        f"{umap_str}\n"
        f"isomap_n={isomap_n} | isomap_neighbors={isomap_neighbors} | "
        f"isomap_max={ISOMAP_MAX_SAMPLES} | {phate_str} | "
        f"nmf_recon_err={nmf_err:.3f} | nmf_max_iter={NMF_MAX_ITER} | "
        f"firing_rate_log={FIRING_RATE_LOG} | random_state={random_state}"
    )
    fig.text(0.01, 0.005, info, ha="left", va="bottom", fontsize=7)


def _scatter_or_placeholder(ax, fig, coords, pop_rate, sub_idx, title, axis_label, missing_msg):
    if coords is None:
        ax.text(0.5, 0.5, missing_msg, ha="center", va="center")
        ax.set_axis_off()
        return
    sc = _firing_rate_scatter(ax, coords, pop_rate[sub_idx], title, axis_label)
    fig.colorbar(sc, ax=ax, label="Mean firing rate (Hz)")


def plot_sleep_block(label, pkl_path, X, centers, units, start_sec, end_sec, out_dir):
    pop_rate = X.mean(axis=1)
    Xz = _zscore_features(X)

    pca_coords, evr = _project_pca(Xz)
    rng = np.random.default_rng(random_state)
    tsne_coords, tsne_idx, perplexity = _project_tsne(Xz, rng)
    umap_coords, umap_idx, umap_neighbors = _project_umap(Xz, rng)
    isomap_coords, isomap_idx, isomap_neighbors = _project_isomap(Xz, rng)
    phate_coords, phate_idx, phate_knn = _project_phate(X, rng)
    nmf_coords, nmf_err = _project_nmf(X)
    full_idx = np.arange(X.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))

    sc0 = _firing_rate_scatter(
        axes[0, 0], pca_coords, pop_rate,
        f"PCA (EVR {evr[0]:.2f}/{evr[1]:.2f})",
        "PC",
    )
    fig.colorbar(sc0, ax=axes[0, 0], label="Mean firing rate (Hz)")

    sc1 = _firing_rate_scatter(
        axes[0, 1], tsne_coords, pop_rate[tsne_idx],
        f"t-SNE (n={len(tsne_idx)}, perp={perplexity:.1f})",
        "t-SNE",
    )
    fig.colorbar(sc1, ax=axes[0, 1], label="Mean firing rate (Hz)")

    _scatter_or_placeholder(
        axes[0, 2], fig, umap_coords, pop_rate,
        umap_idx if umap_idx is not None else full_idx,
        f"UMAP (n={len(umap_idx) if umap_idx is not None else 0}, "
        f"neighbors={umap_neighbors}, min_dist={UMAP_MIN_DIST})",
        "UMAP",
        "UMAP unavailable\n(pip install umap-learn)",
    )

    sc3 = _firing_rate_scatter(
        axes[1, 0], isomap_coords, pop_rate[isomap_idx],
        f"Isomap (n={len(isomap_idx)}, neighbors={isomap_neighbors})",
        "Iso",
    )
    fig.colorbar(sc3, ax=axes[1, 0], label="Mean firing rate (Hz)")

    _scatter_or_placeholder(
        axes[1, 1], fig, phate_coords, pop_rate,
        phate_idx if phate_idx is not None else full_idx,
        f"PHATE (n={len(phate_idx) if phate_idx is not None else 0}, "
        f"knn={phate_knn})",
        "PHATE",
        "PHATE unavailable\n(pip install phate)",
    )

    sc5 = _firing_rate_scatter(
        axes[1, 2], nmf_coords, pop_rate,
        f"NMF (recon_err={nmf_err:.2f})",
        "NMF",
    )
    fig.colorbar(sc5, ax=axes[1, 2], label="Mean firing rate (Hz)")

    fig.suptitle(
        f"Sleep block '{label}' - unsupervised bin embedding\n"
        f"interval {start_sec:.1f}-{end_sec:.1f}s, bin {BIN_SIZE_MS} ms, "
        f"n_bins={X.shape[0]}, n_units={X.shape[1]}",
        fontsize=12,
    )
    _stamp_figure(
        fig, label, pkl_path, start_sec, end_sec, X,
        len(tsne_idx), perplexity,
        len(umap_idx) if umap_idx is not None else 0,
        umap_neighbors if umap_neighbors is not None else 0,
        len(isomap_idx), isomap_neighbors,
        len(phate_idx) if phate_idx is not None else 0,
        phate_knn if phate_knn is not None else 0,
        nmf_err,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"sleep_unsupervised_{label}_{BIN_SIZE_MS}ms_{stamp}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {fig_path}")
    return fig_path


def main():
    from params import task_pkl
    base_out_dir = Path(task_pkl).parent / "reactivation" / OUT_SUBDIR

    bin_size_sec = BIN_SIZE_MS / 1000.0
    for label, pkl_path, start_sec, end_sec in sleep_blocks:
        if not pkl_path:
            print(f"[{label}] skipped - no pkl path.")
            continue
        print(f"\n========== Sleep block '{label}' ==========")
        X, centers, units, s, e = _load_sleep_bins(pkl_path, start_sec, end_sec, bin_size_sec)
        print(f"  X={X.shape}  units={len(units)}  bins={len(centers)}")
        plot_sleep_block(label, pkl_path, X, centers, units, s, e, base_out_dir)


if __name__ == "__main__":
    main()
