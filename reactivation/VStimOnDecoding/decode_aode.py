"""
AODE (Averaged One-Dependency Estimator) cross-validated decoding.

Training logic ported from:
  https://github.com/asugden/flow/blob/master/flow/classifier/aode.py

Inference is re-implemented in pure NumPy (no C extension required).
Input firing-rate matrices are binarized before training.
"""

import numpy as np
from scipy.special import logsumexp


# ------------------------------------------------------------------ #
#  Core AODE classifier                                               #
# ------------------------------------------------------------------ #

class AODEClassifier:
    """Averaged One-Dependency Estimator trained on binarized spike data."""

    def __init__(self, pseudocount=0.1):
        self._pseudocount = pseudocount
        self._cond = {}        # {class: (n_cells, n_cells, 4)}
        self._marg = {}        # {class: (n_cells, 2)}
        self._classnames = []
        self.ncells = None

    def train(self, tevents):
        """Train on (optionally soft) binary spike data.

        Parameters
        ----------
        tevents : dict {class_name: ndarray (n_onsets, n_cells, n_frames)}
            Per-class spike indicators in [0, 1].  The max across frames is
            used, matching the original implementation.
        """
        self._classnames = list(tevents.keys())
        self.ncells = tevents[self._classnames[0]].shape[1]

        for condition in self._classnames:
            stims = np.max(tevents[condition], axis=2).T  # (n_cells, n_onsets)
            stiminv = 1.0 - stims
            nonsets = stims.shape[1]

            # Vectorised joint counts via matrix multiply (replaces inner loop)
            self._cond[condition] = np.stack([
                stims @ stims.T,    # TT: c=1, i=1
                stims @ stiminv.T,  # TF: c=1, i=0
                stiminv @ stims.T,  # FT: c=0, i=1
                stiminv @ stiminv.T,# FF: c=0, i=0
            ], axis=2)              # (n_cells, n_cells, 4)

            # Zero diagonal — same cell is excluded from conditioning
            for q in range(4):
                np.fill_diagonal(self._cond[condition][:, :, q], 0.0)

            self._marg[condition] = np.stack([
                np.sum(stims, axis=1),   # P(c=1 | class)
                np.sum(stiminv, axis=1), # P(c=0 | class)
            ], axis=1)                   # (n_cells, 2)

            pc = self._pseudocount
            denom_cond = float(nonsets + 4 * pc)
            denom_marg = float(nonsets + 4 * pc)

            self._cond[condition] = (self._cond[condition] + pc) / denom_cond
            self._marg[condition] = (self._marg[condition] + pc * 2) / denom_marg

            # Divide joint by parent marginal to get P(x_i | x_c, class)
            # cols 0,1 conditioned on c=1; cols 2,3 conditioned on c=0
            p1 = self._marg[condition][:, 0:1]  # (n_cells, 1)
            p0 = self._marg[condition][:, 1:2]
            self._cond[condition][:, :, 0] /= p1
            self._cond[condition][:, :, 1] /= p1
            self._cond[condition][:, :, 2] /= p0
            self._cond[condition][:, :, 3] /= p0

        return self

    def predict_log_proba(self, X_bin, prior_probs=None):
        """Compute AODE log-scores for each class.

        Parameters
        ----------
        X_bin  : (n_samples, n_cells) binary {0, 1} array
        prior_probs : dict {class: float} or None (uniform)

        Returns
        -------
        log_proba : (n_samples, n_classes) array
        """
        n_samples, n_cells = X_bin.shape
        n_classes = len(self._classnames)

        if prior_probs is None:
            prior_probs = {k: 1.0 / n_classes for k in self._classnames}

        eps = 1e-10
        log_cond = {k: np.log(np.clip(self._cond[k], eps, None))
                    for k in self._classnames}
        log_marg = {k: np.log(np.clip(self._marg[k], eps, None))
                    for k in self._classnames}
        log_prior = {k: np.log(max(prior_probs[k], eps)) for k in self._classnames}

        j_idx = np.arange(n_cells)
        log_proba = np.empty((n_samples, n_classes))

        for s in range(n_samples):
            x = X_bin[s].astype(int)   # (n_cells,)

            # cond_idx[j, i] picks the right conditional probability column
            # col 0: c=1,i=1  col 1: c=1,i=0  col 2: c=0,i=1  col 3: c=0,i=0
            x_j = x[:, np.newaxis]     # (n_cells, 1)
            x_i = x[np.newaxis, :]     # (1, n_cells)
            cond_idx = (1 - x_j) * 2 + (1 - x_i)  # (n_cells, n_cells)

            for ci, k in enumerate(self._classnames):
                # log P(x_j | class) for every parent j
                log_parent = log_marg[k][j_idx, 1 - x]  # (n_cells,)
                # 1-x: x=1 → col 0 = P(1|class), x=0 → col 1 = P(0|class)

                # log P(x_i | x_j, class) for all (j, i) pairs
                lc = log_cond[k][j_idx[:, None],
                                 j_idx[None, :],
                                 cond_idx]             # (n_cells, n_cells)
                lc[j_idx, j_idx] = 0.0                 # exclude self (diagonal)

                log_terms = log_parent + lc.sum(axis=1)  # (n_cells,)

                # Average over parents in log-space + prior
                log_proba[s, ci] = (logsumexp(log_terms) - np.log(n_cells)
                                    + log_prior[k])

        return log_proba

    def predict(self, X_bin, prior_probs=None):
        lp = self.predict_log_proba(X_bin, prior_probs)
        idx = np.argmax(lp, axis=1)
        return np.array([self._classnames[i] for i in idx])


# ------------------------------------------------------------------ #
#  Convenience helpers                                                #
# ------------------------------------------------------------------ #

def binarize(X, threshold=0.0):
    """Return 1 where firing rate > threshold, else 0."""
    return (X > threshold).astype(np.float32)
