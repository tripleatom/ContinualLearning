"""
Leave-one-out template-correlation diagnostic.

For each class label used to build the templates in
`plot_event_patterns()` (see apply_merged_decoder_to_sleep.py), this script
reports how self-consistent the per-class activity vectors are:

    For every training bin x_i of class c:
        template_LOO = mean of all class-c bins EXCEPT x_i
        r_i = pearson_corr(x_i, template_LOO)

A class with high mean LOO correlation has bins that look like each other (a
tight template); a class with low LOO correlation has heterogeneous activity
patterns and the template is a weak summary.

We also report two class-by-class correlation matrices:
    1. sample-vs-template: bins compared with class templates
    2. sample-vs-sample: bins compared with random bins from each class

Outputs (saved next to the sleep-decoding output dir):
    template_loo_correlation_<bin_ms>ms.png
    template_loo_correlation_<bin_ms>ms.pkl
"""

import sys
import pickle
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
# Fall back to the sibling VStimOnDecoding folder for shared modules
# (params.py, decode_utils, prepare_*).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "VStimOnDecoding"))

import numpy as np
import matplotlib.pyplot as plt

from decode_utils import balance_by_undersampling
from prepare_passive_stimtype import prepare_passive_stim_type
from prepare_task_stimtype import prepare_task_stim_type

from params import (
    task_pkl, passive_pkl,
    bin_sizes_ms, random_state,
    class_pos, class_neg,
    TASK_COL_MAP, PASSIVE_COL_MAP,
)


# ---- which bin size(s) to analyze ----------------------------------- #
# Set to None to run every bin size in params.bin_sizes_ms.
# Set to an int (e.g. 100) to run only that one.
BIN_MS_OVERRIDE = None
USE_BALANCED = True  # match what plot_event_patterns() actually sees


def _align_columns(X, units, common_units):
    idx = [units.index(u) for u in common_units]
    return X[:, idx]


def _prepare_merged_training(bin_size_sec):
    X_t, y_t, _, units_t = prepare_task_stim_type(
        task_pkl, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False, random_state=random_state,
    )
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False, random_state=random_state,
    )
    common_units = sorted(set(units_t) & set(units_p))
    if not common_units:
        raise RuntimeError("No common units between task and passive data.")
    X_t = _align_columns(X_t, units_t, common_units)
    X_p = _align_columns(X_p, units_p, common_units)
    return np.vstack([X_t, X_p]), np.concatenate([y_t, y_p]), common_units


def _pearson_rows(X, t):
    """Pearson r between each row of X (n,k) and a single vector t (k,).

    If a row or t has zero variance, returns NaN for that pair."""
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
    """Pearson r between corresponding rows of A and B.

    If a row in either matrix has zero variance, returns NaN for that pair.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, got {A.shape} and {B.shape}")
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", Ac, Bc)
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    out = np.full(A.shape[0], np.nan, dtype=float)
    ok = denom > 1e-12
    out[ok] = num[ok] / denom[ok]
    return out


def loo_template_corr(X_c):
    """Leave-one-out template correlations within one class.

    For each row x_i of X_c, correlate x_i with the mean of the other rows.
    Uses the identity (n*mean - x_i) / (n-1) so we never recompute the mean.
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
    """Correlate bins in X_a with random bins from X_b.

    By default, one random X_b bin is sampled for each X_a bin, with
    replacement, so the average is a sample-to-sample correlation summary.
    If exclude_self is True, X_a and X_b are assumed to be the same class
    matrix and random within-class pairs never compare a bin with itself.
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


def template_corr_matrix(X, y, labels):
    """3x3 matrix of mean correlation: rows = bin's true class,
    cols = template class. Diagonal is LOO; off-diagonals use full templates."""
    full_templates = {lab: X[y == lab].mean(axis=0) for lab in labels if np.any(y == lab)}
    mat = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for i, lab_row in enumerate(labels):
        Xi = X[y == lab_row]
        if Xi.size == 0:
            continue
        for j, lab_col in enumerate(labels):
            if lab_col not in full_templates:
                continue
            if lab_row == lab_col:
                r = loo_template_corr(Xi)
            else:
                r = _pearson_rows(Xi, full_templates[lab_col])
            mat[i, j] = np.nanmean(r) if r.size else np.nan
    return mat


def sample_pair_corr_matrix(X, y, labels, rng):
    """3x3 matrix of mean random bin-to-bin correlations.

    Rows = first bin's class, cols = randomly sampled comparison bin's class.
    Diagonal entries are random within-class pairs and exclude self-pairs.
    """
    mat = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for i, lab_row in enumerate(labels):
        Xi = X[y == lab_row]
        if Xi.size == 0:
            continue
        for j, lab_col in enumerate(labels):
            Xj = X[y == lab_col]
            r = random_bin_corr(Xi, Xj, rng, exclude_self=(lab_row == lab_col))
            mat[i, j] = np.nanmean(r) if r.size else np.nan
    return mat


LABELS = [-1, 0, 1]
CLASS_COLORS = {-1: "#d95f0e", 0: "0.45", 1: "#2c7fb8"}


def compute_for_bin(bin_ms):
    print(f"\n=== Template LOO @ {bin_ms} ms ===")
    X, y, common_units = _prepare_merged_training(bin_ms / 1000.0)
    if USE_BALANCED:
        rng = np.random.default_rng(random_state)
        X, y = balance_by_undersampling(X, y, rng)
    counts = {lab: int(np.sum(y == lab)) for lab in LABELS}
    print(f"  bins per class: {counts}   n_units={len(common_units)}")

    loo = {lab: loo_template_corr(X[y == lab]) for lab in LABELS if counts[lab] > 1}
    for lab in LABELS:
        r = loo.get(lab, np.array([]))
        if r.size:
            print(f"  class {lab:+d}: mean LOO r = {np.nanmean(r):+.3f}  "
                  f"median = {np.nanmedian(r):+.3f}  n = {r.size}")
        else:
            print(f"  class {lab:+d}: insufficient bins")

    template_mat = template_corr_matrix(X, y, LABELS)
    rng_pairs = np.random.default_rng(random_state)
    pair_mat = sample_pair_corr_matrix(X, y, LABELS, rng_pairs)

    print("  sample-vs-template mean correlation (rows=true class, cols=template):")
    print("        " + "  ".join(f"{l:+d}" for l in LABELS))
    for i, lab in enumerate(LABELS):
        row = "  ".join(f"{v:+.3f}" if not np.isnan(v) else "  nan " for v in template_mat[i])
        print(f"    {lab:+d}: {row}")

    print("  sample-vs-sample mean correlation (random bin pairs; diagonal excludes self-pairs):")
    print("        " + "  ".join(f"{l:+d}" for l in LABELS))
    for i, lab in enumerate(LABELS):
        row = "  ".join(f"{v:+.3f}" if not np.isnan(v) else "  nan " for v in pair_mat[i])
        print(f"    {lab:+d}: {row}")

    return {
        "bin_ms": bin_ms,
        "counts": counts,
        "loo": loo,
        "template_mat": template_mat,
        "pair_mat": pair_mat,
        "common_units": common_units,
    }


def _draw_violin(ax, loo, title):
    positions, data, tick_labels = [], [], []
    for i, lab in enumerate(LABELS):
        r = loo.get(lab, np.array([]))
        r = r[~np.isnan(r)] if r.size else r
        if r.size == 0:
            continue
        positions.append(i)
        data.append(r)
        tick_labels.append(f"{lab:+d}\n(n={r.size})")
    if data:
        parts = ax.violinplot(data, positions=positions,
                              showmeans=False, showextrema=False, widths=0.7)
        for body, x in zip(parts["bodies"], positions):
            body.set_facecolor(CLASS_COLORS[LABELS[x]])
            body.set_alpha(0.45)
        ax.boxplot(data, positions=positions, widths=0.18,
                   showfliers=False, patch_artist=False)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylim(-0.2, 1.0)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_heatmap(ax, cmat, title, vmin=-1, vmax=1):
    im = ax.imshow(cmat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([f"{l:+d}" for l in LABELS], fontsize=8)
    ax.set_yticklabels([f"{l:+d}" for l in LABELS], fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = cmat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=8)
    return im


def plot_combined(results, out_dir):
    """One figure with all bin sizes:
        row 0  – LOO violin per class, one column per bin size
        row 1  – cross-class correlation heatmap, one column per bin size
        row 2  – summary line plot: mean LOO r vs bin size, per class
    """
    n = len(results)
    if n == 0:
        return
    fig = plt.figure(figsize=(2.6 * n + 1.2, 12))
    gs = fig.add_gridspec(
        4, n + 1,
        height_ratios=[1.1, 1.0, 1.0, 1.0],
        width_ratios=[1] * n + [0.06],
        wspace=0.35, hspace=0.45,
    )

    # row 0: violins
    for k, res in enumerate(results):
        ax = fig.add_subplot(gs[0, k])
        _draw_violin(ax, res["loo"], f"{res['bin_ms']} ms")
        if k == 0:
            ax.set_ylabel("LOO Pearson r\n(bin vs. own-class template)")

    # row 1: sample-vs-template heatmaps
    last_im = None
    for k, res in enumerate(results):
        ax = fig.add_subplot(gs[1, k])
        last_im = _draw_heatmap(ax, res["template_mat"], f"{res['bin_ms']} ms")
        if k == 0:
            ax.set_ylabel("True class\nsample-vs-template")
        ax.set_xlabel("Template class")

    # row 2: sample-vs-sample heatmaps
    for k, res in enumerate(results):
        ax = fig.add_subplot(gs[2, k])
        last_im = _draw_heatmap(ax, res["pair_mat"], f"{res['bin_ms']} ms")
        if k == 0:
            ax.set_ylabel("True class\nsample-vs-sample")
        ax.set_xlabel("Comparison class")
    if last_im is not None:
        cax = fig.add_subplot(gs[1:3, -1])
        fig.colorbar(last_im, cax=cax, label="mean r")

    # row 3: summary line plot
    ax_sum = fig.add_subplot(gs[3, :n])
    bin_axis = [res["bin_ms"] for res in results]
    for lab in LABELS:
        means = []
        for res in results:
            r = res["loo"].get(lab, np.array([]))
            means.append(np.nanmean(r) if r.size else np.nan)
        ax_sum.plot(bin_axis, means, marker="o",
                    color=CLASS_COLORS[lab], label=f"class {lab:+d}")
    ax_sum.set_xlabel("Bin size (ms)")
    ax_sum.set_ylabel("Mean LOO r")
    ax_sum.set_title("Template self-consistency vs. bin size")
    ax_sum.set_xscale("log")
    ax_sum.set_xticks(bin_axis)
    ax_sum.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax_sum.legend(loc="lower right", frameon=False)
    ax_sum.spines[["top", "right"]].set_visible(False)
    ax_sum.grid(alpha=0.3)

    fig.suptitle(
        f"Template LOO diagnostics across bin sizes "
        f"({'balanced' if USE_BALANCED else 'unbalanced'})",
        fontsize=12, y=0.995,
    )
    fig_path = out_dir / "template_loo_correlation_all_bins.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nCombined figure -> {fig_path}")


def save_per_bin_pkl(res, out_dir):
    pkl_path = out_dir / f"template_loo_correlation_{res['bin_ms']}ms.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "bin_ms": res["bin_ms"],
            "balanced": USE_BALANCED,
            "labels": LABELS,
            "counts": res["counts"],
            "loo_correlations": {lab: res["loo"].get(lab, np.array([])) for lab in LABELS},
            "sample_vs_template_mean_corr": res["template_mat"],
            "sample_vs_template_mean_corr_note": (
                "Rows are true sample class; columns are template class. "
                "Diagonal entries are leave-one-out bin-vs-own-template correlations; "
                "off-diagonal entries are bin-vs-other-class-template correlations."
            ),
            "sample_vs_sample_mean_corr": res["pair_mat"],
            "sample_vs_sample_mean_corr_note": (
                "Rows are true sample class; columns are random comparison sample class. "
                "All entries are mean bin-to-bin correlations. Diagonal entries use "
                "random within-class pairs and exclude self-pairs."
            ),
            "common_units": res["common_units"],
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  data   -> {pkl_path}")


def main():
    session = Path(task_pkl).parent.name
    out_dir = Path(task_pkl).parent / "reactivation" / f"sleep_merged_decoder_{session}" / "template_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if BIN_MS_OVERRIDE is None:
        bins_to_run = list(bin_sizes_ms)
    else:
        bins_to_run = [BIN_MS_OVERRIDE]

    results = []
    for bms in bins_to_run:
        res = compute_for_bin(bms)
        save_per_bin_pkl(res, out_dir)
        results.append(res)

    plot_combined(results, out_dir)
    plt.show()


if __name__ == "__main__":
    main()
