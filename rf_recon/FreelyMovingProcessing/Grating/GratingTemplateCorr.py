"""
Per-class neural population similarity for grating data.

For each class (orientation, optionally per spatial frequency), build a
population-vector "template" = mean firing-rate vector across units, then
report two class-by-class Pearson correlation matrices, mirroring the
diagnostic used in reactivation/sleep/template_loo_correlation.py:

  1. sample-vs-template
       rows = true class of the trial,
       cols = template class.
       Diagonal entries are leave-one-out bin-vs-own-class-template r
       (so the trial being correlated is excluded from its own template).
       Off-diagonals use the full template of the other class.

  2. sample-vs-sample
       rows = first trial's class, cols = randomly sampled comparison
       trial's class. Diagonal entries are random within-class pairs and
       exclude self-pairs.

Data loading and feature extraction reuse grating_utils so the pipeline
is identical to GratingLDA.py.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from grating_utils import load_neural_data, calculate_firing_rates


# =============================================================================
# CORE CORRELATION OPS
# =============================================================================

def _pearson_rows_to_vec(X, t):
    """Pearson r between each row of X (n,k) and a single vector t (k,).
    Returns NaN for rows with zero variance."""
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    tc = t - t.mean()
    num = Xc @ tc
    denom = np.linalg.norm(Xc, axis=1) * np.linalg.norm(tc)
    out = np.full(X.shape[0], np.nan, dtype=float)
    ok = denom > 1e-12
    out[ok] = num[ok] / denom[ok]
    return out


def _pearson_row_pairs(A, B):
    """Pearson r between corresponding rows of A and B (same shape)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", Ac, Bc)
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    out = np.full(A.shape[0], np.nan, dtype=float)
    ok = denom > 1e-12
    out[ok] = num[ok] / denom[ok]
    return out


def loo_template_corr(X_c):
    """LOO template correlations within one class.

    Uses (n*mean - x_i)/(n-1) so the leave-one-out template is computed
    in one shot, never recomputing the mean per trial.
    """
    X_c = np.asarray(X_c, dtype=float)
    n = X_c.shape[0]
    if n < 2:
        return np.array([])
    full_sum = X_c.sum(axis=0)
    loo_mean = (full_sum[None, :] - X_c) / (n - 1)
    Xc = X_c - X_c.mean(axis=1, keepdims=True)
    Tc = loo_mean - loo_mean.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", Xc, Tc)
    denom = np.linalg.norm(Xc, axis=1) * np.linalg.norm(Tc, axis=1)
    out = np.full(n, np.nan, dtype=float)
    ok = denom > 1e-12
    out[ok] = num[ok] / denom[ok]
    return out


def random_bin_corr(X_a, X_b, rng, n_pairs=None, exclude_self=False):
    """Pearson r between random rows of X_a and random rows of X_b.

    If exclude_self, X_a and X_b are assumed to be the same matrix and
    random within-class pairs never compare a trial with itself.
    """
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    if X_a.size == 0 or X_b.size == 0:
        return np.array([])
    if exclude_self and X_a.shape[0] < 2:
        return np.array([])
    if n_pairs is None:
        n_pairs = X_a.shape[0]
        idx_a = np.arange(X_a.shape[0])
    else:
        idx_a = rng.integers(0, X_a.shape[0], size=n_pairs)
    if exclude_self:
        idx_b = rng.integers(0, X_b.shape[0] - 1, size=n_pairs)
        idx_b = idx_b + (idx_b >= idx_a)
    else:
        idx_b = rng.integers(0, X_b.shape[0], size=n_pairs)
    return _pearson_row_pairs(X_a[idx_a], X_b[idx_b])


# =============================================================================
# MATRIX BUILDERS
# =============================================================================

def template_corr_matrix(X, y, classes):
    """K x K matrix: rows=true class, cols=template class.
    Diagonal is LOO; off-diagonals use the full mean template."""
    full_templates = {c: X[y == c].mean(axis=0) for c in classes if np.any(y == c)}
    K = len(classes)
    mat = np.full((K, K), np.nan, dtype=float)
    for i, c_row in enumerate(classes):
        Xi = X[y == c_row]
        if Xi.size == 0:
            continue
        for j, c_col in enumerate(classes):
            if c_col not in full_templates:
                continue
            r = (loo_template_corr(Xi) if c_row == c_col
                 else _pearson_rows_to_vec(Xi, full_templates[c_col]))
            mat[i, j] = np.nanmean(r) if r.size else np.nan
    return mat


def sample_pair_corr_matrix(X, y, classes, rng):
    """K x K matrix of mean random trial-to-trial correlations.
    Diagonal entries are within-class random pairs (self-pairs excluded)."""
    K = len(classes)
    mat = np.full((K, K), np.nan, dtype=float)
    for i, c_row in enumerate(classes):
        Xi = X[y == c_row]
        if Xi.size == 0:
            continue
        for j, c_col in enumerate(classes):
            Xj = X[y == c_col]
            r = random_bin_corr(Xi, Xj, rng, exclude_self=(c_row == c_col))
            mat[i, j] = np.nanmean(r) if r.size else np.nan
    return mat


# =============================================================================
# PIPELINE PER GROUP
# =============================================================================

def compute_for_group(X, y, classes, random_state=42):
    """Both correlation matrices + LOO distributions for one group of trials."""
    counts = {c: int(np.sum(y == c)) for c in classes}
    loo = {c: loo_template_corr(X[y == c]) for c in classes if counts[c] > 1}
    template_mat = template_corr_matrix(X, y, classes)
    rng = np.random.default_rng(random_state)
    pair_mat = sample_pair_corr_matrix(X, y, classes, rng)
    return {
        "classes": list(classes),
        "counts": counts,
        "loo": loo,
        "template_mat": template_mat,
        "pair_mat": pair_mat,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _draw_heatmap(ax, mat, classes, title, label_suffix='°', vmin=-1, vmax=1):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ticks = [f"{c}{label_suffix}" for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(ticks, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(ticks, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black",
                        fontsize=10, fontweight='bold')
    return im


def plot_group(result, group_label, label_suffix='°', save_path=None):
    """Side-by-side heatmaps for the two correlation matrices in one group."""
    classes = result["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    im1 = _draw_heatmap(axes[0], result["template_mat"], classes,
                        "sample-vs-template\n(diag = LOO)",
                        label_suffix=label_suffix)
    axes[0].set_xlabel("Template class", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("True class", fontsize=12, fontweight='bold')

    im2 = _draw_heatmap(axes[1], result["pair_mat"], classes,
                        "sample-vs-sample\n(diag excludes self-pairs)",
                        label_suffix=label_suffix)
    axes[1].set_xlabel("Comparison class", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("True class", fontsize=12, fontweight='bold')

    for im, ax in zip((im1, im2), axes):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean r")

    count_str = "  ".join(f"{c}{label_suffix}: n={result['counts'][c]}"
                          for c in classes)
    fig.suptitle(f"Population-vector similarity — {group_label}\n{count_str}",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  figure -> {save_path}")
    return fig


def save_pkl(result, group_label, label_suffix, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "group": group_label,
            "label_suffix": label_suffix,
            "classes": result["classes"],
            "counts": result["counts"],
            "loo_correlations": result["loo"],
            "sample_vs_template_mean_corr": result["template_mat"],
            "sample_vs_template_mean_corr_note": (
                "Rows = true class of trial; cols = template class. "
                "Diagonal = leave-one-out (trial excluded from its own template); "
                "off-diagonals = full-mean templates."
            ),
            "sample_vs_sample_mean_corr": result["pair_mat"],
            "sample_vs_sample_mean_corr_note": (
                "Rows = trial class; cols = random comparison class. "
                "Diagonal = within-class random pairs, self-pairs excluded."
            ),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  data   -> {save_path}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.07, 0.16), save_plots=True,
                 output_path=None, random_state=42):
    """
    Build similarity matrices for orientation classes per SF, and (if there
    are multiple SFs) for SF classes per orientation, mirroring GratingLDA.
    """
    data = load_neural_data(data_path)
    firing_rates, ori_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )
    if len(ori_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")

    sf_labels = trial_info['spatial_freq_labels']
    unique_sfs = trial_info['unique_spatial_freqs']

    base = Path(output_path) if output_path else Path(data_path).with_suffix('')
    all_results = {}

    # --- Orientation similarity, per SF ---------------------------------------
    print(f"\n{'#'*60}\n# ORIENTATION SIMILARITY PER SF\n{'#'*60}")
    for sf in unique_sfs:
        if sf is None:
            X_sf, y_sf, tag, display = firing_rates, ori_labels, '', 'all SF'
        else:
            mask = sf_labels == sf
            X_sf, y_sf = firing_rates[mask], ori_labels[mask]
            tag = f'_sf{sf}'
            display = f'SF={sf} cpd'

        classes = sorted(set(y_sf.tolist()))
        if len(classes) < 2:
            print(f"\n[{display}] skipped (need >=2 orientations).")
            continue

        print(f"\n[{display}]  trials={len(y_sf)}  classes={classes}")
        res = compute_for_group(X_sf, y_sf, classes, random_state=random_state)
        _print_matrices(res, label_suffix='°')

        if save_plots:
            fig_path = Path(str(base) + tag + '.pop_similarity.png')
            pkl_path = Path(str(base) + tag + '.pop_similarity.pkl')
            plot_group(res, display, label_suffix='°', save_path=fig_path)
            save_pkl(res, display, '°', pkl_path)
        all_results[('orientation', sf)] = res

    # --- SF similarity, per orientation ---------------------------------------
    if sf_labels is not None and len(unique_sfs) > 1:
        print(f"\n{'#'*60}\n# SF SIMILARITY PER ORIENTATION\n{'#'*60}")
        for ori in sorted(set(ori_labels.tolist())):
            mask = ori_labels == ori
            X_ori, sf_ori = firing_rates[mask], sf_labels[mask]
            classes = sorted(set(sf_ori.tolist()))
            if len(classes) < 2:
                print(f"\n[ori={ori}°] skipped (only one SF).")
                continue

            display = f'orientation={ori}°'
            print(f"\n[{display}]  trials={len(sf_ori)}  SFs={classes}")
            res = compute_for_group(X_ori, sf_ori, classes,
                                    random_state=random_state)
            _print_matrices(res, label_suffix=' cpd')

            if save_plots:
                fig_path = Path(str(base) + f'_ori{ori}.sf_similarity.png')
                pkl_path = Path(str(base) + f'_ori{ori}.sf_similarity.pkl')
                plot_group(res, display, label_suffix=' cpd',
                           save_path=fig_path)
                save_pkl(res, display, ' cpd', pkl_path)
            all_results[('spatial_freq', ori)] = res

    return all_results


def _print_matrices(res, label_suffix=''):
    classes = res["classes"]
    header = "        " + "  ".join(f"{c}{label_suffix}" for c in classes)
    for name, mat in [("sample-vs-template (diag=LOO)", res["template_mat"]),
                      ("sample-vs-sample  (diag excl self)", res["pair_mat"])]:
        print(f"  {name}:")
        print(header)
        for i, c in enumerate(classes):
            row = "  ".join(f"{v:+.3f}" if not np.isnan(v) else "  nan "
                            for v in mat[i])
            print(f"    {c}{label_suffix}: {row}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = input("Enter path to neural data (.pkl file): ").strip().strip('"').strip("'")
    try:
        run_analysis(
            data_path=DATA_PATH,
            time_window=(0.05, 1.5),
            save_plots=True,
        )
        print("\nDone.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
