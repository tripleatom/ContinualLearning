"""
Shared utilities for decoding analyses:
  bin_spikes, balance_by_undersampling, PoissonNB, AODEClassifier,
  make_classifiers, run_cv, plot_panel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scipy.special import logsumexp


# ------------------------------------------------------------------ #
#  Data helpers                                                        #
# ------------------------------------------------------------------ #

def bin_spikes(spike_data, bin_edges, bin_size_sec):
    """Convert spike_data dict → firing-rate matrix (n_bins, n_units).

    Returns
    -------
    X           : (n_bins, n_units) float32  spikes/s
    unit_labels : list of str
    """
    unit_labels = sorted(spike_data.keys())
    n_bins = len(bin_edges) - 1
    X = np.zeros((n_bins, len(unit_labels)), dtype=np.float32)
    for col, uid in enumerate(unit_labels):
        spikes = np.asarray(spike_data[uid]['spike_times_sec'])
        counts, _ = np.histogram(spikes, bins=bin_edges)
        X[:, col] = counts / bin_size_sec
    return X, unit_labels


def balance_by_undersampling(X, y, rng, bin_centers=None):
    """Undersample every class to the size of the smallest class.

    Parameters
    ----------
    bin_centers : array or None – if provided, also subset and return it

    Returns
    -------
    (X, y) or (X, y, bin_centers) depending on whether bin_centers is given
    """
    classes = np.unique(y)
    min_count = min(int(np.sum(y == c)) for c in classes)
    sel = np.sort(np.concatenate([
        rng.choice(np.where(y == c)[0], size=min_count, replace=False)
        for c in classes
    ]))
    if bin_centers is not None:
        return X[sel], y[sel], bin_centers[sel]
    return X[sel], y[sel]


def balance_by_class_group_undersampling(X, y, groups, rng):
    """Undersample every (class, group) cell to the size of the smallest
    non-empty cell. Decouples the class signal from the group signal in
    the training set (used by the merged-CV runner to prevent the decoder
    from cheating via task-vs-passive base-rate differences).

    Empty cells stay empty. Returns (X_balanced, y_balanced, groups_balanced).
    """
    classes = np.unique(y)
    gs      = np.unique(groups)
    cell_counts = []
    for c in classes:
        for g in gs:
            n = int(np.sum((y == c) & (groups == g)))
            if n > 0:
                cell_counts.append(n)
    if not cell_counts:
        raise ValueError("All (class, group) cells are empty.")
    min_cell = min(cell_counts)
    picks = []
    for c in classes:
        for g in gs:
            idx = np.where((y == c) & (groups == g))[0]
            if len(idx) == 0:
                continue
            picks.append(rng.choice(idx, size=min_cell, replace=False))
    sel = np.sort(np.concatenate(picks))
    return X[sel], y[sel], groups[sel]


# ------------------------------------------------------------------ #
#  Classifiers                                                         #
# ------------------------------------------------------------------ #

class PoissonNB(BaseEstimator, ClassifierMixin):
    """Naive Bayes with Poisson likelihood — suited for spike-rate data."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        eps = 1e-10
        self._log_prior = np.array([np.log(np.mean(y == c)) for c in self.classes_])
        self._log_lam = np.array([
            np.log(np.clip(X[y == c].mean(axis=0), eps, None))
            for c in self.classes_
        ])
        self._lam_sum = np.exp(self._log_lam).sum(axis=1)
        return self

    def predict(self, X):
        scores = X @ self._log_lam.T - self._lam_sum + self._log_prior
        return self.classes_[np.argmax(scores, axis=1)]


# ------------------------------------------------------------------ #
#  AODE                                                                #
# ------------------------------------------------------------------ #

def binarize(X, threshold=0.0):
    """Return 1 where firing rate > threshold, else 0."""
    return (X > threshold).astype(np.float32)


class AODEClassifier(BaseEstimator, ClassifierMixin):
    """Averaged One-Dependency Estimator on binarised spike data (scikit-learn API).

    Training logic ported from
      https://github.com/asugden/flow/blob/master/flow/classifier/aode.py
    and inference re-implemented in pure NumPy (no C extension).  X is a firing-rate
    matrix; it is binarised (> 0) inside fit / predict, so this behaves like any other
    estimator in make_classifiers().
    """

    def __init__(self, pseudocount=0.1):
        self.pseudocount = pseudocount

    def fit(self, X, y):
        X_bin = binarize(X).astype(np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X_bin.shape[1]
        self.class_prior_ = np.array([float(np.mean(y == c)) for c in self.classes_])
        self._cond, self._marg = {}, {}
        pc = self.pseudocount

        for c in self.classes_:
            stims = X_bin[y == c].T                    # (n_cells, n_onsets)
            stiminv = 1.0 - stims
            nonsets = stims.shape[1]

            # Joint counts by matrix multiply: c=1,i=1 / c=1,i=0 / c=0,i=1 / c=0,i=0
            cond = np.stack([stims @ stims.T, stims @ stiminv.T,
                             stiminv @ stims.T, stiminv @ stiminv.T], axis=2)
            for q in range(4):                          # same cell is excluded from conditioning
                np.fill_diagonal(cond[:, :, q], 0.0)
            marg = np.stack([np.sum(stims, axis=1), np.sum(stiminv, axis=1)], axis=1)

            denom = float(nonsets + 4 * pc)
            cond = (cond + pc) / denom
            marg = (marg + pc * 2) / denom

            # Divide the joint by the parent marginal to get P(x_i | x_c, class);
            # cols 0,1 are conditioned on c=1, cols 2,3 on c=0.
            p1, p0 = marg[:, 0:1], marg[:, 1:2]
            cond[:, :, 0] /= p1
            cond[:, :, 1] /= p1
            cond[:, :, 2] /= p0
            cond[:, :, 3] /= p0
            self._cond[c], self._marg[c] = cond, marg
        return self

    def predict_log_proba(self, X):
        """Log of the (unnormalised) AODE score per class: (n_samples, n_classes)."""
        X_bin = binarize(X).astype(np.float64)
        n_samples, n_cells = X_bin.shape
        eps = 1e-10
        log_cond = {c: np.log(np.clip(self._cond[c], eps, None)) for c in self.classes_}
        log_marg = {c: np.log(np.clip(self._marg[c], eps, None)) for c in self.classes_}
        log_prior = [np.log(max(p, eps)) for p in self.class_prior_]

        j_idx = np.arange(n_cells)
        log_proba = np.empty((n_samples, len(self.classes_)))
        for s in range(n_samples):
            x = X_bin[s].astype(int)
            # cond_idx[j, i] picks the conditional-probability column for (x_j, x_i)
            cond_idx = (1 - x[:, np.newaxis]) * 2 + (1 - x[np.newaxis, :])
            for ci, c in enumerate(self.classes_):
                log_parent = log_marg[c][j_idx, 1 - x]            # log P(x_j | class)
                lc = log_cond[c][j_idx[:, None], j_idx[None, :], cond_idx]
                lc[j_idx, j_idx] = 0.0                            # exclude self
                log_terms = log_parent + lc.sum(axis=1)
                # average over parents in log space, plus the prior
                log_proba[s, ci] = logsumexp(log_terms) - np.log(n_cells) + log_prior[ci]
        return log_proba

    def predict_proba(self, X):
        lp = self.predict_log_proba(X)
        return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def _scaled(clf):
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def make_classifiers(random_state=42):
    """Return a fresh classifier dict; call once per analysis to avoid shared mutable state."""
    return {
        'Gaussian NB':   GaussianNB(),
        'Poisson NB':    PoissonNB(),
        'LDA':           LinearDiscriminantAnalysis(),
        'Logistic Reg':  _scaled(LogisticRegression(max_iter=1000, random_state=random_state)),
        'SVM':           _scaled(LinearSVC(max_iter=2000, random_state=random_state)),
        'Random Forest': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
        'AODE':          AODEClassifier(),
    }


# ------------------------------------------------------------------ #
#  Cross-validated runner                                              #
# ------------------------------------------------------------------ #

def run_cv(name, clf_proto, X, y, n_splits, random_state):
    """
    Stratified k-fold CV for any scikit-learn-style classifier (AODE included).

    `name` is only a label for the caller; the model comes from `clf_proto`.

    Returns
    -------
    fold_accs       : list[float]
    mean_acc        : float
    chance          : float
    per_class_means : dict {class_label: float}  mean per-fold recall
    per_class_stds  : dict {class_label: float}  std  per-fold recall
    """
    classes = sorted(int(c) for c in np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accs = []
    per_class_fold = {c: [] for c in classes}

    for train_idx, test_idx in skf.split(X, y):
        clf = clone(clf_proto)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        y_te  = y[test_idx]

        fold_accs.append(float(np.mean(preds == y_te)))
        for c in classes:
            mask = y_te == c
            per_class_fold[c].append(
                float(np.mean(preds[mask] == c)) if mask.any() else np.nan)

    _, counts = np.unique(y, return_counts=True)
    chance = float(counts.max() / counts.sum())
    per_class_means = {c: float(np.nanmean(per_class_fold[c])) for c in classes}
    per_class_stds  = {c: float(np.nanstd(per_class_fold[c]))  for c in classes}

    return fold_accs, float(np.mean(fold_accs)), chance, per_class_means, per_class_stds


def run_cv_balanced_train(name, clf_proto, X, y, n_splits, random_state):
    """
    Stratified k-fold CV: undersample majority classes in the training split
    each fold, but test on the held-out split at the natural (imbalanced) ratio.

    Returns same signature as run_cv.
    """
    classes = sorted(int(c) for c in np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)

    fold_accs = []
    per_class_fold = {c: [] for c in classes}

    for train_idx, test_idx in skf.split(X, y):
        X_tr_full, y_tr_full = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        X_tr, y_tr = balance_by_undersampling(X_tr_full, y_tr_full, rng)

        clf = clone(clf_proto)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        fold_accs.append(float(np.mean(preds == y_te)))
        for c in classes:
            mask = y_te == c
            per_class_fold[c].append(
                float(np.mean(preds[mask] == c)) if mask.any() else np.nan)

    _, counts = np.unique(y, return_counts=True)
    chance = float(counts.max() / counts.sum())
    per_class_means = {c: float(np.nanmean(per_class_fold[c])) for c in classes}
    per_class_stds  = {c: float(np.nanstd(per_class_fold[c]))  for c in classes}

    return fold_accs, float(np.mean(fold_accs)), chance, per_class_means, per_class_stds


def run_cv_balanced_train_grouped(name, clf_proto, X, y, groups, n_splits, random_state):
    """Stratified k-fold CV for a multi-group dataset (e.g. task + passive
    merged). Each training fold is undersampled to balance every
    (class x group) cell to the smallest non-empty cell, which keeps the
    decoder from cheating via class/group correlation. The held-out test
    fold stays at natural ratio, and test-set accuracy is also split by
    `groups` (e.g., 0=task, 1=passive).

    Returns
    -------
    mean_acc        : float                            overall mean across folds
    per_class       : dict {class: float}              mean per-fold recall
    group_means     : dict {group: float}              mean per-fold acc, per group
    group_per_class : dict {group: {class: float}}     per-fold recall, per group
    """
    classes = sorted(int(c) for c in np.unique(y))
    group_ids = sorted(int(g) for g in np.unique(groups))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)

    fold_accs = []
    per_class_fold = {c: [] for c in classes}
    g_fold_accs = {g: [] for g in group_ids}
    g_per_class_fold = {g: {c: [] for c in classes} for g in group_ids}

    for train_idx, test_idx in skf.split(X, y):
        X_tr_full, y_tr_full = X[train_idx], y[train_idx]
        g_tr_full            = groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        g_te = groups[test_idx]

        X_tr, y_tr, _ = balance_by_class_group_undersampling(
            X_tr_full, y_tr_full, g_tr_full, rng,
        )

        clf = clone(clf_proto)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        fold_accs.append(float(np.mean(preds == y_te)))
        for c in classes:
            mask = y_te == c
            per_class_fold[c].append(
                float(np.mean(preds[mask] == c)) if mask.any() else np.nan)

        for g in group_ids:
            gmask = g_te == g
            g_fold_accs[g].append(
                float(np.mean(preds[gmask] == y_te[gmask])) if gmask.any() else np.nan)
            for c in classes:
                gcmask = gmask & (y_te == c)
                g_per_class_fold[g][c].append(
                    float(np.mean(preds[gcmask] == c)) if gcmask.any() else np.nan)

    return (
        float(np.mean(fold_accs)),
        {c: float(np.nanmean(per_class_fold[c])) for c in classes},
        {g: float(np.nanmean(g_fold_accs[g])) for g in group_ids},
        {g: {c: float(np.nanmean(g_per_class_fold[g][c])) for c in classes}
         for g in group_ids},
    )


def run_cv_balanced_train_shuffle_grouped(name, clf_proto, X, y, groups, n_splits,
                                           random_state, n_shuffles=10):
    """Permutation null for run_cv_balanced_train_grouped.

    Permutes y globally each shuffle (groups stay fixed since they encode the
    real recording context). Runs the full grouped CV pipeline n_shuffles
    times and returns null distribution statistics for both the overall
    accuracy and per-group accuracy.

    Returns
    -------
    shuf_mean        : float           mean overall null accuracy
    shuf_std         : float           std overall null accuracy
    group_shuf_means : dict {g: float}
    group_shuf_stds  : dict {g: float}
    """
    group_ids = sorted(int(g) for g in np.unique(groups))
    rng = np.random.default_rng(random_state + 99999)

    all_means = []
    g_all_means = {g: [] for g in group_ids}

    for i in range(n_shuffles):
        y_shuf = rng.permutation(y)
        mean, _, g_means, _ = run_cv_balanced_train_grouped(
            name, clf_proto, X, y_shuf, groups, n_splits, random_state + i,
        )
        all_means.append(mean)
        for g in group_ids:
            g_all_means[g].append(g_means[g])

    return (
        float(np.mean(all_means)),
        float(np.std(all_means)),
        {g: float(np.mean(g_all_means[g])) for g in group_ids},
        {g: float(np.std(g_all_means[g])) for g in group_ids},
    )


def run_cv_balanced_train_shuffle(name, clf_proto, X, y, n_splits, random_state, n_shuffles=10):
    """Run run_cv_balanced_train with permuted labels to estimate null distribution.

    Returns
    -------
    shuf_mean    : float   mean accuracy across shuffles
    shuf_std     : float   std across shuffles
    shuf_pc_means: dict {class: float}
    shuf_pc_stds : dict {class: float}
    """
    classes = sorted(int(c) for c in np.unique(y))
    rng = np.random.default_rng(random_state + 99999)
    all_means = []
    all_pc = {c: [] for c in classes}

    for i in range(n_shuffles):
        y_shuf = rng.permutation(y)
        _, mean, _, pc_means, _ = run_cv_balanced_train(
            name, clf_proto, X, y_shuf, n_splits, random_state + i
        )
        all_means.append(mean)
        for c in classes:
            all_pc[c].append(pc_means.get(c, np.nan))

    return (
        float(np.mean(all_means)),
        float(np.std(all_means)),
        {c: float(np.nanmean(all_pc[c])) for c in classes},
        {c: float(np.nanstd(all_pc[c]))  for c in classes},
    )


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

def plot_panel(ax, results, bin_sizes_ms, y_key, title, ylim_bottom):
    """Plot one accuracy panel (overall or per-class) with error bars."""
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for (name, res), marker in zip(results.items(), markers):
        if y_key == 'overall':
            means = np.array(res['means'])
            stds  = np.array(res['stds'])
        else:
            means = np.array(res['per_class'][y_key]['means'])
            stds  = np.array(res['per_class'][y_key]['stds'])
        ax.errorbar(bin_sizes_ms, means, yerr=stds,
                    marker=marker, linewidth=1.5, capsize=4, label=name)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Bin size (ms)', fontsize=11)
    ax.set_ylabel('Accuracy (recall)', fontsize=11)
    ax.set_xticks(bin_sizes_ms)
    ax.set_xticklabels([str(b) for b in bin_sizes_ms])
    ax.set_xlim(bin_sizes_ms[0] - 15, bin_sizes_ms[-1] + 15)
    ax.set_ylim(ylim_bottom, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
