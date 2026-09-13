r"""Pair-level statistics for the tuning-similarity / sleep-coupling hypothesis.

The model
---------
For every pair of units,

    coupling ~ b0 + b1 * tuning_similarity + controls + e

with controls for what else could make two neurons co-fire: their firing rates
(geometric mean, and how unequal they are), their physical separation (same
shank, depth difference, lateral map offset) and their putative cell types
(wide/wide, narrow/narrow, mixed).  The hypothesis is a claim about b1.

Why the p-value is a permutation p-value
----------------------------------------
Pairs are not independent observations: with n units each unit appears in n-1
pairs, so an OLS standard error computed as if the ~n^2/2 pairs were independent
is far too small and would call almost anything significant.  The test used here
instead permutes the assignment of TUNING to units and refits:

    * each unit keeps its own rate, position and cell type (so the controls and
      the dependence structure among pairs are untouched),
    * only which tuning curve belongs to which unit is scrambled,
    * because similarity depends on the unordered pair alone, a permutation is
      just a relabelled lookup into the similarity matrix, which makes an exact
      recomputation cheap enough to do thousands of times.

The null being tested is therefore precisely "coupling is unrelated to which
tuning curve a unit happens to have", which is the scientific null.  OLS
standard errors are still reported, flagged as anticonservative.

Pre vs post
-----------
``compare_pre_post`` regresses the WITHIN-PAIR change in coupling on tuning
similarity, restricted to pairs measured in both blocks.  Differencing removes
any stable functional architecture shared by the two sleep blocks, so a positive
slope there is the experience-dependent claim; a positive slope in each block
separately with no difference between them is the "stable architecture" result.
"""
from __future__ import annotations

import numpy as np


def fisher_z(r):
    """arctanh with the +-1 endpoints kept finite."""
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    return np.arctanh(r)


def pairs_to_matrix(values, pair_index, n_units, fill=np.nan):
    """Scatter a per-pair vector into a symmetric (n_units, n_units) matrix."""
    M = np.full((n_units, n_units), fill, dtype=float)
    i, j = pair_index[:, 0], pair_index[:, 1]
    M[i, j] = values
    M[j, i] = values
    return M


def matrix_to_pairs(matrix, pair_index):
    """Read a per-pair vector out of a symmetric matrix."""
    return np.asarray(matrix)[pair_index[:, 0], pair_index[:, 1]]


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #

def build_design(table, predictor, controls=('log_rate_geomean', 'abs_log_rate_ratio',
                                             'same_shank', 'abs_dy_um', 'abs_dx_um',
                                             'pair_type'), standardize=True):
    """
    Assemble (X, y_names) from a column dict.

    ``pair_type`` is expanded into dummies against the most common level, so the
    reported intercept refers to the majority cell-type combination.  Columns
    that are constant or entirely missing are dropped and reported, rather than
    silently producing a rank-deficient fit.
    """
    columns, names = [], []
    for name in (predictor,) + tuple(controls):
        if name not in table:
            continue
        values = table[name]
        if values.dtype == object:          # categorical
            levels, counts = np.unique(values.astype(str), return_counts=True)
            if levels.size < 2:
                continue
            base = levels[np.argmax(counts)]
            for level in levels:
                if level == base:
                    continue
                columns.append((values.astype(str) == level).astype(float))
                names.append(f"{name}[{level}]")
        else:
            values = np.asarray(values, dtype=float)
            columns.append(values)
            names.append(name)

    X = np.column_stack(columns) if columns else np.zeros((len(table[predictor]), 0))
    keep = []
    dropped = []
    for k, name in enumerate(names):
        col = X[:, k]
        if not np.any(np.isfinite(col)) or np.nanstd(col) == 0:
            dropped.append(name)
        else:
            keep.append(k)
    X, names = X[:, keep], [names[k] for k in keep]

    scale = np.ones(X.shape[1])
    if standardize and X.size:
        scale = np.nanstd(X, axis=0)
        scale[scale == 0] = 1.0
        X = (X - np.nanmean(X, axis=0)) / scale

    return X, names, scale, dropped


def ols(X, y):
    """Least squares with intercept. Returns beta, se, t, r2, dof."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        t = beta / se
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, r2, dof


def fit_pair_model(table, predictor, response, controls=None, standardize=True):
    """
    Fit one pair-level model and return coefficients plus the rows actually used.

    Rows with a missing response, predictor or control are dropped listwise; the
    count is reported so a control that silently removes half the pairs (e.g. a
    distance defined only within shank) cannot pass unnoticed.
    """
    kwargs = {} if controls is None else {'controls': tuple(controls)}
    X, names, scale, dropped = build_design(table, predictor, standardize=standardize,
                                            **kwargs)
    y = np.asarray(table[response], dtype=float)

    finite = np.isfinite(y)
    if X.size:
        finite &= np.all(np.isfinite(X), axis=1)
    n_used = int(finite.sum())
    if n_used < X.shape[1] + 10:
        raise ValueError(f"Only {n_used} usable pairs for {response} ~ {predictor}.")

    beta, se, t, r2, dof = ols(X[finite], y[finite])
    return {
        'predictor': predictor, 'response': response,
        'terms': ['intercept'] + names,
        'beta': beta, 'se_ols': se, 't_ols': t, 'r2': r2, 'dof': dof,
        'beta_predictor': float(beta[1]) if len(beta) > 1 else np.nan,
        'se_predictor_ols': float(se[1]) if len(se) > 1 else np.nan,
        'n_pairs': n_used, 'n_dropped': int((~finite).sum()),
        'dropped_terms': dropped, 'standardized': bool(standardize),
        'scale': scale, 'mask': finite,
    }


# --------------------------------------------------------------------------- #
# Permutation inference
# --------------------------------------------------------------------------- #

def unit_permutation_test(table, predictor_matrix, pair_index, n_units, response,
                          controls=None, n_permutations=1000, seed=0,
                          standardize=True, verbose=True):
    """
    Permute tuning across units and refit, giving an exact-by-construction null
    for the tuning coefficient.

    Args:
        predictor_matrix: (n_units, n_units) similarity matrix -- permuting unit
            labels is then a relabelled lookup, so each permutation costs one
            gather plus one least-squares solve.
        response: column name of the coupling measure in ``table``

    Returns dict with observed beta, the null distribution, and a two-sided p.
    """
    rng = np.random.default_rng(seed)
    predictor_name = 'tuning_similarity'
    work = dict(table)
    work[predictor_name] = matrix_to_pairs(predictor_matrix, pair_index)

    observed = fit_pair_model(work, predictor_name, response, controls=controls,
                              standardize=standardize)
    beta_obs = observed['beta_predictor']

    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(n_units)
        work[predictor_name] = predictor_matrix[perm[pair_index[:, 0]],
                                                perm[pair_index[:, 1]]]
        try:
            null[k] = fit_pair_model(work, predictor_name, response,
                                     controls=controls,
                                     standardize=standardize)['beta_predictor']
        except (ValueError, np.linalg.LinAlgError):
            null[k] = np.nan
        if verbose and n_permutations >= 5 and (k + 1) % max(1, n_permutations // 4) == 0:
            print(f"    permutation {k + 1}/{n_permutations}")

    null = null[np.isfinite(null)]
    p_two = ((np.sum(np.abs(null) >= abs(beta_obs)) + 1) / (null.size + 1)
             if null.size else np.nan)
    p_greater = (np.sum(null >= beta_obs) + 1) / (null.size + 1) if null.size else np.nan
    p_less = (np.sum(null <= beta_obs) + 1) / (null.size + 1) if null.size else np.nan

    z = ((beta_obs - np.mean(null)) / np.std(null)
         if null.size and np.std(null) > 0 else np.nan)

    return {**observed,
            'beta_null_mean': float(np.mean(null)) if null.size else np.nan,
            'beta_null_sd': float(np.std(null)) if null.size else np.nan,
            'beta_z_vs_null': float(z),
            'p_permutation_two_sided': float(p_two),
            'p_permutation_greater': float(p_greater),
            'p_permutation_less': float(p_less),
            'n_permutations': int(null.size),
            'null_distribution': null}


# --------------------------------------------------------------------------- #
# Descriptive profiles and pre/post
# --------------------------------------------------------------------------- #

def binned_profile(x, y, bin_edges):
    """Mean +- SEM of y in bins of x (for the figure, not for inference)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    idx = np.digitize(x, bin_edges) - 1
    centers, means, sems, counts = [], [], [], []
    for b in range(len(bin_edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
        counts.append(n)
        means.append(float(np.mean(y[sel])) if n else np.nan)
        sems.append(float(np.std(y[sel], ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
    return (np.array(centers), np.array(means), np.array(sems), np.array(counts))


def compare_pre_post(table_pre, table_post, predictor_matrix, pair_index, n_units,
                     response, controls=None, n_permutations=1000, seed=0,
                     verbose=True):
    """
    Test whether the tuning-coupling relationship strengthens from pre to post.

    Both tables must be indexed by the same ``pair_index``.  The response is the
    within-pair difference (post - pre), so anything stable about a pair --
    including a stable functional architecture that has nothing to do with the
    session -- cancels before the tuning term is estimated.
    """
    y_pre = np.asarray(table_pre[response], dtype=float)
    y_post = np.asarray(table_post[response], dtype=float)
    if y_pre.shape != y_post.shape:
        raise ValueError("pre and post tables describe different pair sets.")

    delta_table = dict(table_post)
    delta_table[f'delta_{response}'] = y_post - y_pre

    if verbose:
        both = np.isfinite(y_pre) & np.isfinite(y_post)
        print(f"Pre/post comparison on {int(both.sum())} pairs measured in both blocks")
        print(f"  {response}: pre median {np.nanmedian(y_pre[both]):+.4f}, "
              f"post median {np.nanmedian(y_post[both]):+.4f}")

    return unit_permutation_test(delta_table, predictor_matrix, pair_index, n_units,
                                 response=f'delta_{response}', controls=controls,
                                 n_permutations=n_permutations, seed=seed,
                                 verbose=verbose)


def format_result(result, label=''):
    """One-block text summary of a permutation-tested model."""
    lines = [f"{label}{result['response']} ~ {result['predictor']} + controls",
             f"  pairs used        : {result['n_pairs']} "
             f"({result['n_dropped']} dropped for missing values)",
             f"  beta (tuning)     : {result['beta_predictor']:+.4f}"
             + ("  [standardized]" if result['standardized'] else ""),
             f"  OLS se (anticons.): {result['se_predictor_ols']:.4f}",
             f"  shuffle null      : {result['beta_null_mean']:+.4f} "
             f"+- {result['beta_null_sd']:.4f}  (n={result['n_permutations']})",
             f"  z vs null         : {result['beta_z_vs_null']:+.2f}",
             f"  p (permutation)   : {result['p_permutation_two_sided']:.4f} two-sided, "
             f"{result['p_permutation_greater']:.4f} greater",
             f"  model R^2         : {result['r2']:.4f}"]
    if result['dropped_terms']:
        lines.append(f"  dropped terms     : {', '.join(result['dropped_terms'])}")
    return "\n".join(lines)
