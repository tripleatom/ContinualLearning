r"""PLS of binned task population activity against the left-side grating identity.

    python pls_task_trials.py CnL46 260729                  # every task run of the day, pooled
    python pls_task_trials.py CnL46 260729 --max-sec 0      # whole stim period instead of its first 3 s
    python pls_task_trials.py CnL46 260729 --pre-sec 0      # without the pre-stimulus control
    python pls_task_trials.py CnL46 260729 --bin-ms 50 --n-components 4

Input: the day's task_spikes_trial_<...>.pkl files (extract_task_day.py), found
through task_day_configs.json.

Response matrix: each trial's stimulus-on period [onset, offset), capped at
--max-sec (default 3 s: a handful of trials last tens of seconds and would
otherwise hold most of the bins), is cut into --bin-ms bins (a trailing partial
bin is dropped) and every unit's spikes are counted per bin. Trials and task runs are stacked along time into X (T x N),
each row labelled with its trial's left grating orientation. Every unit is
z-scored over the T rows, then PLSRegression is fit on X against the label.

Because N is in the hundreds, PLS separates the two classes in-sample whatever
the data, so the figure pairs the in-sample latent space (top row) with
held-out predictions from a trial-grouped cross-validation and a trial-label
shuffle (bottom row). All bins of one trial always stay in the same fold.

Pre-stimulus control (--pre-sec, default 2 s): the bins before stimulus onset
are never used for the z-score or the fit; they are only projected through the
held-out models. The left grating is not on yet, so predictions there should
sit at chance - if they do not, the axis is picking up something other than
the stimulus (position, the previous trial, slow drift).

KDE figure: a Gaussian KDE (scipy gaussian_kde, Scott bandwidth) is fit to the
in-sample scores of the first three LVs, separately for the bins of each left
grating. The two 3-D densities are shown as their 2-D marginals on every LV
pair (top row) and as the log ratio of those marginals (bottom row).

Bias figure: B = ln p(z | 45 left) / p(z | 315 left) for every bin of every
trial, strictly held out. Per fold, the units' mean / sd, the PLS projection
and the two 3-D KDEs all come from the stim bins of the training trials only;
a test bin's counts are normalised with that mean / sd, projected and scored
by both KDEs. B > 0 = 45-left-like, B < 0 = 315-left-like. Test bins cover the
whole trial: --pre-sec before onset, the full stim period (also past
--max-sec) and BIAS_POST_SEC after stim offset.

Outputs, in <sortout>/pls_task/:
  pls_task_<sortout>_bin<ms>ms[_max<sec>s].png
  pls_task_<sortout>_bin<ms>ms[_max<sec>s]_kde.png
  pls_task_<sortout>_bin<ms>ms[_max<sec>s]_bias.png
  pls_task_<sortout>_bin<ms>ms[_max<sec>s].npz    counts, labels, scores, held-out predictions; rows with
                                                  fit = False were never fitted (pre-stimulus, past
                                                  --max-sec, after stim offset)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

GRATING_DIR = Path(__file__).resolve().parents[3] / 'DiscriminationTask' / 'grating'  # task_day_discovery lives here
if str(GRATING_DIR) not in sys.path:
    sys.path.insert(0, str(GRATING_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

from task_day_discovery import REGISTRY_FILE, load_registry, short_date

COLORS = {45: 'tab:red', 315: 'tab:blue'}      # left grating orientation
DEFAULT_MAX_SEC = 3.0      # cap on each trial's stim period; a few trials last tens of seconds
DEFAULT_PRE_SEC = 2.0      # pre-stimulus control window (the trial pkls hold 2 s before onset)
N_SHUFFLES = 200
N_FOLDS = 5
TIMECOURSE_SEC = 3.0       # x range of the time-from-onset panel
MIN_TRIALS_PER_TIMEBIN = 5
KDE_DIMS = 3               # the KDE lives in the first three LVs
KDE_GRID = 160             # grid points per axis of the 2-D marginal maps
KDE_MASS_LEVELS = (0.9, 0.5, 0.25)   # contours enclose this much of a condition's probability mass
KDE_RATIO_MASS = 0.99      # log ratio shown where the two-condition mixture holds this much mass
KDE_RATIO_CLIP = 2.0       # colour range of the log10 density ratio; the tails run to +-20 and beyond
BIAS_POST_SEC = 2.0        # bias is also evaluated this long after stim offset (the pkls hold 2 s)
BIAS_VIEW_SEC = 5.0        # x range of the bias heat map after stim onset
BIAS_CLIP = 6.0            # colour range of the bias heat map (ln units)
BIAS_N_EXAMPLES = 2        # example trials per condition
KDE_VIEW_PCT = (1, 99)     # axis range = these percentiles of the scores (+ 25 % padding)


# ============================================================ response matrix

def bin_trial_pkl(pkl_file, bin_sec, max_sec=None, pre_sec=0.0, post_sec=0.0):
    """Binned counts of one task run, from pre_sec before stim onset to
    post_sec after stim offset, on a bin grid anchored at stim onset.

    Returns (counts (T x N), unit_ids, rows) where rows is a dict of per-row
    arrays: trial (index within this run), bin (index within the trial, 0 =
    first bin after stim onset, negative = pre-stimulus), fit (True for the
    bins the models are fit on: stim on and before max_sec), trial_dur (stim
    duration, s), left_ori, choice_left, correct.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    trials = data['trial_info']['all_trial_parameters']
    unit_ids = list(data['spike_data'])
    if pre_sec > data['extraction_params']['window_pre'] + 1e-9:
        raise ValueError(f"{Path(pkl_file).name} holds only {data['extraction_params']['window_pre']} s "
                         f"before stim onset, --pre-sec {pre_sec} asked")
    if post_sec > data['extraction_params']['window_post'] + 1e-9:
        raise ValueError(f"{Path(pkl_file).name} holds only {data['extraction_params']['window_post']} s "
                         f"after stim offset, {post_sec} asked")
    n_pre = int(round(pre_sec / bin_sec))
    n_bins, n_fit = [], []
    for t in trials:
        dur = t['trial_duration'] if max_sec is None else min(t['trial_duration'], max_sec)
        n_stim = int(np.floor(dur / bin_sec + 1e-9))
        n_fit.append(n_stim)
        n_all = int(np.floor((t['trial_duration'] + post_sec) / bin_sec + 1e-9))
        n_bins.append(n_pre + n_all if n_stim else 0)
    starts = np.concatenate(([0], np.cumsum(n_bins)))
    counts = np.zeros((starts[-1], len(unit_ids)), dtype=np.int32)
    for u, uid in enumerate(unit_ids):
        for k, row in enumerate(data['spike_data'][uid]):
            if n_bins[k] == 0:
                continue
            edges = np.arange(-n_pre, n_bins[k] - n_pre + 1) * bin_sec
            counts[starts[k]:starts[k + 1], u] = np.histogram(row['spike_times'], bins=edges)[0]
    rep = lambda values: np.repeat(np.asarray(values), n_bins)
    rows = {
        'trial': rep(np.arange(len(trials))),
        'bin': (np.concatenate([np.arange(n) - n_pre for n in n_bins]) if trials
                else np.array([], dtype=int)),
        'trial_dur': rep([float(t['trial_duration']) for t in trials]),
        'left_ori': rep([int(round(t['left_orientation'])) for t in trials]),
        'choice_left': rep([t['choice'] == 'left' for t in trials]),
        'correct': rep([bool(t['correct']) for t in trials]),
    }
    rows['fit'] = (rows['bin'] >= 0) & (rows['bin'] < rep(n_fit))
    return counts, unit_ids, rows


def stack_runs(pkl_files, bin_sec, max_sec=None, pre_sec=0.0, post_sec=0.0):
    """Stack every run's binned counts along time; trials get a day-wide id."""
    all_counts, all_rows, unit_ids, trial_offset = [], [], None, 0
    for r, pkl_file in enumerate(pkl_files):
        counts, ids, rows = bin_trial_pkl(pkl_file, bin_sec, max_sec, pre_sec, post_sec)
        if unit_ids is None:
            unit_ids = ids
        elif ids != unit_ids:
            raise ValueError(f"{Path(pkl_file).name}: units differ from the day's first run")
        n_trials = int(rows['trial'].max()) + 1 if rows['trial'].size else 0
        rows['trial'] = rows['trial'] + trial_offset
        rows['run'] = np.full(counts.shape[0], r)
        trial_offset += n_trials
        all_counts.append(counts)
        all_rows.append(rows)
    rows = {k: np.concatenate([d[k] for d in all_rows]) for k in all_rows[0]}
    return np.concatenate(all_counts), unit_ids, rows


def zscore(counts, stim):
    """(z, kept): every unit z-scored with its mean / sd over the stim rows;
    units that never vary there are dropped."""
    sd = counts[stim].std(axis=0)
    kept = sd > 0
    x = counts[:, kept].astype(float)
    return (x - x[stim].mean(axis=0)) / sd[kept], kept


# ============================================================ PLS

def fit_pls(x, y, n_components):
    """PLSRegression with LV signs set so every score covaries positively with y."""
    pls = PLSRegression(n_components=n_components, scale=False).fit(x, y)
    scores = pls.transform(x)
    sign = np.sign([np.cov(scores[:, c], y)[0, 1] or 1.0 for c in range(n_components)])
    return pls, scores * sign


def heldout_prediction(x, y, groups, n_components, fit_rows=None, n_folds=N_FOLDS):
    """Predicted y for every row from a model that never saw the row's trial.
    Only rows in fit_rows (default all) are used for fitting."""
    pred = np.full(len(y), np.nan)
    fit_rows = np.ones(len(y), dtype=bool) if fit_rows is None else fit_rows
    for train, test in GroupKFold(n_splits=n_folds).split(x, y, groups):
        train = train[fit_rows[train]]
        pls = PLSRegression(n_components=n_components, scale=False).fit(x[train], y[train])
        pred[test] = np.ravel(pls.predict(x[test]))
    return pred


def accuracies(pred, y, groups):
    """(per-bin, per-trial) balanced accuracy of sign(pred); a trial's
    prediction is the mean over its bins."""
    def balanced(p, t):
        return float(np.mean([np.mean(np.sign(p[t == c]) == c) for c in (-1, 1)]))
    trials = np.unique(groups)
    trial_pred = np.array([pred[groups == g].mean() for g in trials])
    trial_y = np.array([y[groups == g][0] for g in trials])
    return balanced(pred, y), balanced(trial_pred, trial_y), trials, trial_pred, trial_y


def shuffle_null(x, y, groups, n_components, n_shuffles, seed):
    """Held-out per-trial accuracy with the left-grating label permuted across trials."""
    rng = np.random.default_rng(seed)
    trials, inverse = np.unique(groups, return_inverse=True)
    trial_y = np.array([y[groups == g][0] for g in trials])
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        y_perm = rng.permutation(trial_y)[inverse]
        null[i] = accuracies(heldout_prediction(x, y_perm, groups, n_components), y_perm, groups)[1]
    return null


# ============================================================ KDE

def fit_kdes(scores, y):
    """{+1: kde, -1: kde}: one Gaussian KDE per left grating over the bins' LV scores."""
    return {c: gaussian_kde(scores[y == c].T) for c in (1, -1)}


def marginal_kde(kde, dims):
    """Exact marginal of a gaussian_kde on a subset of its dimensions: its
    kernel covariance is factor^2 * cov(data), so the marginal is the KDE of
    the same points in those dimensions with the same factor."""
    return gaussian_kde(kde.dataset[list(dims)], bw_method=kde.factor)


def log_density_ratio(kdes, points):
    """log p(points | 45 left) - log p(points | 315 left); points is (n, d)."""
    return kdes[1].logpdf(points.T) - kdes[-1].logpdf(points.T)


def heldout_bias(counts, y, groups, fit, n_components, n_folds=N_FOLDS):
    """B = ln p(z | 45 left) / p(z | 315 left) for every row, from a fold that
    never saw the row's trial.

    Per fold everything comes from the training trials' fit rows: each unit's
    mean / sd (units with sd = 0 there are left out of that fold), the PLS
    projection and the two KDEs. A test row's raw counts are normalised with
    that mean / sd, projected, and scored by both KDEs.
    """
    bias = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_folds).split(counts, y, groups):
        train = train[fit[train]]
        mu, sd = counts[train].mean(axis=0), counts[train].std(axis=0)
        keep = sd > 0
        z = lambda idx: (counts[idx][:, keep] - mu[keep]) / sd[keep]
        pls = PLSRegression(n_components=n_components, scale=False).fit(z(train), y[train])
        kdes = fit_kdes(pls.transform(z(train))[:, :KDE_DIMS], y[train])
        bias[test] = log_density_ratio(kdes, pls.transform(z(test))[:, :KDE_DIMS])
    return bias


def _mass_levels(density, masses):
    """Density values whose super-level sets hold the given fractions of the grid's mass, ascending."""
    d = np.sort(density.ravel())[::-1]
    cum = np.cumsum(d) / d.sum()
    return sorted(d[min(np.searchsorted(cum, m), d.size - 1)] for m in masses)


def plot_kde(scores, y, kdes, acc_bin, acc_trial, title, stamp, out):
    lo, hi = np.percentile(scores, KDE_VIEW_PCT, axis=0)
    pad = 0.25 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    names = {1: 45, -1: 315}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        ga, gb = np.linspace(lo[a], hi[a], KDE_GRID), np.linspace(lo[b], hi[b], KDE_GRID)
        mesh_a, mesh_b = np.meshgrid(ga, gb)
        grid = np.vstack([mesh_a.ravel(), mesh_b.ravel()])
        dens = {c: marginal_kde(kdes[c], (a, b))(grid).reshape(mesh_a.shape) for c in kdes}

        ax = axes[0, col]
        for c, d in dens.items():
            color = COLORS[names[c]]
            levels = _mass_levels(d, KDE_MASS_LEVELS)
            ax.contourf(mesh_a, mesh_b, d, levels=levels + [d.max()], colors=[color], alpha=0.22)
            ax.contour(mesh_a, mesh_b, d, levels=levels, colors=[color], linewidths=[0.6, 1.0, 1.4])
            ax.scatter(scores[y == c, a], scores[y == c, b], s=2, color=color, alpha=0.25, lw=0)
        ax.set_title(f'KDE marginal on LV{a + 1}-LV{b + 1}', fontsize=9)

        ax = axes[1, col]
        ratio = np.log10(dens[1]) - np.log10(dens[-1])
        mixture = 0.5 * (dens[1] + dens[-1])
        ratio = np.ma.masked_where(mixture < _mass_levels(mixture, [KDE_RATIO_MASS])[0], ratio)
        im = ax.pcolormesh(mesh_a, mesh_b, ratio, cmap='RdBu_r', vmin=-KDE_RATIO_CLIP,
                           vmax=KDE_RATIO_CLIP, shading='auto')
        ax.contour(mesh_a, mesh_b, ratio, levels=[0], colors='k', linewidths=0.8)
        fig.colorbar(im, ax=ax, extend='both', label='log10 p(45\N{DEGREE SIGN} left) / p(315\N{DEGREE SIGN} left)')
        ax.set_title(f'log density ratio of the marginals (black: equal density)', fontsize=9)
        for ax in axes[:, col]:
            ax.set_xlim(lo[a], hi[a])
            ax.set_ylim(lo[b], hi[b])
            ax.set_xlabel(f'LV{a + 1}')
            ax.set_ylabel(f'LV{b + 1}')
    for c in kdes:
        axes[0, 0].plot([], [], color=COLORS[names[c]], lw=1.4,
                        label=f'left grating {names[c]}\N{DEGREE SIGN}  ({(y == c).sum()} bins)')
    axes[0, 0].legend(fontsize=7, frameon=False, title='contours hold ' + ' / '.join(
        f'{m:.0%}' for m in sorted(KDE_MASS_LEVELS)) + ' of the mass', title_fontsize=7)
    fig.suptitle(f"{title}\n3-D KDE per condition on the in-sample scores, shown as 2-D marginals;   "
                 f"held-out sign of the 3-D log density ratio: balanced accuracy {acc_bin:.2f} per bin, "
                 f"{acc_trial:.2f} per trial", fontsize=11, x=0.01, ha='left', y=0.995)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.12, wspace=0.25, hspace=0.28)
    fig.text(0.005, 0.005, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_bias(rows, bias, bin_sec, acc_bin, acc_trial, title, stamp, out):
    """Held-out bias through the trial: every trial as a heat-map row, the
    condition means aligned to stim onset and to stim offset, example trials."""
    names = {45: '45\N{DEGREE SIGN}', 315: '315\N{DEGREE SIGN}'}
    trial, b, dur, left = rows['trial'], rows['bin'], rows['trial_dur'], rows['left_ori']
    centers_all = (b + 0.5) * bin_sec
    stim_on = (b >= 0) & ((b + 1) * bin_sec <= dur + 1e-9)
    ids = np.unique(trial)
    t_dur = np.array([dur[trial == g][0] for g in ids])
    t_left = np.array([left[trial == g][0] for g in ids])
    b_min = int(b.min())
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # every trial, sorted by condition then stim duration
    ax = axes[0, 0]
    order = np.lexsort((t_dur, t_left != 45))
    n_view = int(round(BIAS_VIEW_SEC / bin_sec))
    image = np.full((ids.size, n_view - b_min), np.nan)
    row_of = {g: r for r, g in enumerate(ids[order])}
    sel = b < n_view
    image[[row_of[g] for g in trial[sel]], b[sel] - b_min] = bias[sel]
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')
    im = ax.imshow(np.ma.masked_invalid(image), aspect='auto', cmap=cmap, vmin=-BIAS_CLIP, vmax=BIAS_CLIP,
                   extent=(b_min * bin_sec, BIAS_VIEW_SEC, ids.size - 0.5, -0.5), interpolation='nearest')
    fig.colorbar(im, ax=ax, extend='both', label='B = ln p(z | 45\N{DEGREE SIGN} left) / p(z | 315\N{DEGREE SIGN} left)')
    shown = t_dur[order] <= BIAS_VIEW_SEC
    ax.plot(t_dur[order][shown], np.arange(ids.size)[shown], 'k|', ms=4, mew=1.0, label='stim off (choice)')
    n_45 = int((t_left == 45).sum())
    ax.axhline(n_45 - 0.5, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks([n_45 / 2, n_45 + (ids.size - n_45) / 2])
    ax.set_yticklabels([f"left {names[45]}\n({n_45} trials)", f"left {names[315]}\n({ids.size - n_45} trials)"])
    ax.set_xlabel('time from stim onset (s)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title('held-out bias, one row per trial (sorted by stim duration; grey = outside the trial window)',
                 fontsize=9)

    def mean_trace(ax, index, rows_ok, lo, hi):
        """Median and quartiles over trials of bias at integer time index, per
        condition. Not mean +- sem: outside the fitted region (after stim
        offset above all) both densities are ~0 and single bins reach |B| of
        several hundred."""
        idx = np.arange(lo, hi)
        for ori, color in COLORS.items():
            q = np.full((3, idx.size), np.nan)
            for i, k in enumerate(idx):
                v = bias[rows_ok & (left == ori) & (index == k)]
                if v.size >= MIN_TRIALS_PER_TIMEBIN:
                    q[:, i] = np.percentile(v, [25, 50, 75])
            ax.plot((idx + 0.5) * bin_sec, q[1], color=color, label=f'left {names[ori]}')
            ax.fill_between((idx + 0.5) * bin_sec, q[0], q[2], color=color, alpha=0.25, lw=0)
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_ylabel('B  (median and quartiles over trials)')
        ax.set_ylim(-2.5 * BIAS_CLIP, 2.5 * BIAS_CLIP)

    ax = axes[0, 1]
    mean_trace(ax, b, (b < 0) | stim_on, b_min, int(round(TIMECOURSE_SEC / bin_sec)))
    if b_min < 0:
        ax.axvspan(b_min * bin_sec, 0, color='0.92', lw=0, zorder=0)
    ax.set_xlabel('time from stim onset (s)')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f'aligned to stim onset, stim-on bins only  (bins with \N{GREATER-THAN OR EQUAL TO} '
                 f'{MIN_TRIALS_PER_TIMEBIN} trials; grey = before onset)', fontsize=9)

    ax = axes[1, 0]
    from_off = np.floor((centers_all - dur) / bin_sec).astype(int)
    half = int(round(BIAS_POST_SEC / bin_sec))
    mean_trace(ax, from_off, b >= 0, -half, half)
    ax.axvspan(0, half * bin_sec, color='0.92', lw=0, zorder=0)
    ax.set_xlabel('time from stim offset = choice (s)')
    ax.set_title('aligned to stim offset  (grey = after the stimulus is off)', fontsize=9)

    # example trials: stim durations spread over the upper half of each condition's trials that fit the view
    ax = axes[1, 1]
    for ori, color in COLORS.items():
        ok = (t_left == ori) & (t_dur <= BIAS_VIEW_SEC - BIAS_POST_SEC)
        cand = ids[ok][np.argsort(t_dur[ok])]
        picks = cand[np.round(np.linspace(0.5, 0.95, BIAS_N_EXAMPLES) * (cand.size - 1)).astype(int)]
        for g in picks:
            m = (trial == g) & (centers_all < BIAS_VIEW_SEC)
            ax.step(centers_all[m], np.clip(bias[m], -2 * BIAS_CLIP, 2 * BIAS_CLIP), where='mid', color=color,
                    lw=0.9, alpha=0.8)
            ax.plot(t_dur[ids == g], 0, '|', color=color, ms=12, mew=1.5)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('time from stim onset (s)')
    ax.set_xlim(b_min * bin_sec, BIAS_VIEW_SEC)
    ax.set_ylabel(f'B  (clipped at \N{PLUS-MINUS SIGN}{2 * BIAS_CLIP:g})')
    ax.set_title(f'{BIAS_N_EXAMPLES} example trials per condition (ticks on the zero line: their stim offsets)',
                 fontsize=9)

    fig.suptitle(f"{title}\nheld-out track-bias-style log odds  B = ln p(z | 45\N{DEGREE SIGN} left) / "
                 f"p(z | 315\N{DEGREE SIGN} left);   sign(B) on the fitted stim bins: balanced accuracy "
                 f"{acc_bin:.2f} per bin, {acc_trial:.2f} per trial", fontsize=11, x=0.01, ha='left', y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.12, wspace=0.2, hspace=0.25)
    fig.text(0.005, 0.005, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ============================================================ figure

def plot_pls(scores, rows, pred, acc_bin, acc_trial, null, trial_stats, acc_pre, bin_sec, title, stamp, out):
    """scores and the first five panels cover the stim rows only; the last
    panel also shows the pre-stimulus control rows (rows['bin'] < 0)."""
    all_rows, all_pred = rows, pred
    stim = rows['bin'] >= 0
    rows = {k: v[stim] for k, v in rows.items()}
    pred = pred[stim]
    left = rows['left_ori']
    trials, trial_pred, trial_y = trial_stats
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def scatter_lv(ax, a, b):
        for ori, color in COLORS.items():
            m = left == ori
            ax.scatter(scores[m, a], scores[m, b], s=4, color=color, alpha=0.25, lw=0)
        for g in trials:                       # one marker per trial: mean over its bins
            m = rows['trial'] == g
            ax.scatter(scores[m, a].mean(), scores[m, b].mean(), s=28, color=COLORS[left[m][0]],
                       edgecolor='k', lw=0.5, zorder=3)
        ax.set_xlabel(f'LV{a + 1}')
        ax.set_ylabel(f'LV{b + 1}')

    scatter_lv(axes[0, 0], 0, 1)
    axes[0, 0].set_title('in-sample PLS scores (dots = bins, circles = trial means)', fontsize=9)
    for ori, color in COLORS.items():
        axes[0, 0].scatter([], [], s=28, color=color, edgecolor='k', lw=0.5,
                           label=f'left grating {ori}\N{DEGREE SIGN}  ({np.unique(rows["trial"][left == ori]).size} '
                                 f'trials, {(left == ori).sum()} bins)')
    axes[0, 0].legend(fontsize=7, frameon=False)
    scatter_lv(axes[0, 1], 0, 2)
    axes[0, 1].set_title('in-sample PLS scores', fontsize=9)

    ax = axes[0, 2]
    t = rows['bin'] * bin_sec
    sc = ax.scatter(scores[:, 0], scores[:, 1], s=4, c=np.minimum(t, TIMECOURSE_SEC), cmap='viridis', lw=0)
    fig.colorbar(sc, ax=ax, label=f'time from stim onset (s, clipped at {TIMECOURSE_SEC:g})')
    ax.set_xlabel('LV1')
    ax.set_ylabel('LV2')
    ax.set_title('same space, coloured by time in trial', fontsize=9)

    ax = axes[1, 0]
    lim = np.nanpercentile(np.abs(pred), 99.5)
    edges = np.linspace(-lim, lim, 41)
    for ori, color in COLORS.items():
        ax.hist(pred[left == ori], bins=edges, color=color, alpha=0.55)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('held-out PLS prediction per bin  (<0: 315\N{DEGREE SIGN} left, >0: 45\N{DEGREE SIGN} left)')
    ax.set_ylabel('bins')
    ax.set_title(f'held-out, per bin: balanced accuracy {acc_bin:.2f}', fontsize=9)

    ax = axes[1, 1]
    choice_left = np.array([rows['choice_left'][rows['trial'] == g][0] for g in trials])
    for is_left, marker in ((True, '<'), (False, '>')):
        m = choice_left == is_left
        ax.scatter(trials[m], trial_pred[m], s=30, marker=marker,
                   c=[COLORS[45 if v > 0 else 315] for v in trial_y[m]], edgecolor='k', lw=0.4,
                   label=f"chose {'left' if is_left else 'right'}")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('trial (day order)')
    ax.set_ylabel('held-out prediction, mean over the trial')
    p = (1 + np.sum(null >= acc_trial)) / (1 + null.size)
    ax.set_title(f'held-out, per trial: balanced accuracy {acc_trial:.2f}   (label shuffle '
                 f'{null.mean():.2f} \N{PLUS-MINUS SIGN} {null.std():.2f}, p = {p:.3f})', fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 2]
    bins = np.arange(all_rows['bin'].min(), int(round(TIMECOURSE_SEC / bin_sec)))
    centers = (bins + 0.5) * bin_sec
    for ori, color in COLORS.items():
        mean, sem = np.full(bins.size, np.nan), np.full(bins.size, np.nan)
        for i, b in enumerate(bins):
            v = all_pred[(all_rows['left_ori'] == ori) & (all_rows['bin'] == b)]
            if v.size >= MIN_TRIALS_PER_TIMEBIN:
                mean[i], sem[i] = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
        ax.plot(centers, mean, color=color)
        ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.25, lw=0)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='tab:green', lw=0.6)
    if bins[0] < 0:        # pre-stimulus control: not fitted, only projected
        ax.axvspan(bins[0] * bin_sec, 0, color='0.92', lw=0, zorder=0)
    ax.set_xlabel('time from stim onset (s)')
    ax.set_ylabel('held-out prediction (mean \N{PLUS-MINUS SIGN} sem over trials)')
    control = f'\npre-stim control (grey, not fitted): per-trial accuracy {acc_pre:.2f}' if acc_pre is not None else ''
    ax.set_title(f'held-out prediction vs time in trial  (bins with \N{GREATER-THAN OR EQUAL TO} '
                 f'{MIN_TRIALS_PER_TIMEBIN} trials){control}', fontsize=9)

    fig.suptitle(title, fontsize=11, x=0.01, ha='left', y=0.995)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12, wspace=0.25, hspace=0.28)
    fig.text(0.005, 0.005, stamp, fontsize=5.5, color='0.35', va='bottom')
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ============================================================ driver

def run_day(animal, date, bin_ms=100.0, n_components=3, max_sec=DEFAULT_MAX_SEC, pre_sec=DEFAULT_PRE_SEC,
            seed=0, out_dir=None, log=print):
    """max_sec None = whole stim period; pre_sec 0 = no pre-stimulus control."""
    key = f"{animal}_{short_date(date)}"
    config = load_registry().get(key)
    if config is None:
        raise KeyError(f"{key} is not in {REGISTRY_FILE} - register the day with task_day_gui.py")
    pkl_files = [t['trial_pkl'].replace('\\', '/') for t in config.get('tasks', []) if t.get('trial_pkl')]
    pkl_files = [p for p in pkl_files if Path(p).is_file()]
    if not pkl_files:
        raise FileNotFoundError(f"{key}: no trial pkl on disk - run extract_task_day.py first")
    bin_sec = bin_ms / 1000.0

    counts, unit_ids, rows = stack_runs(pkl_files, bin_sec, max_sec, pre_sec, BIAS_POST_SEC)
    stim = rows['fit']
    pre = rows['bin'] < 0
    oris = np.unique(rows['left_ori'])
    if set(oris.tolist()) != set(COLORS):
        raise ValueError(f"left orientations are {oris.tolist()}, expected {sorted(COLORS)}")
    x, kept = zscore(counts, stim)
    y = np.where(rows['left_ori'] == 45, 1.0, -1.0)
    groups = rows['trial']
    n_trials = np.unique(groups).size
    bins_per_trial = np.bincount(groups[stim])
    bins_per_trial = bins_per_trial[bins_per_trial > 0]
    top5 = np.sort(bins_per_trial)[-5:].sum() / bins_per_trial.sum()
    log(f"{key}: {len(pkl_files)} run(s), {n_trials} trials, X = {stim.sum()} stim bins x {x.shape[1]} units "
        f"({(~kept).sum()} silent units dropped); bins per trial median {np.median(bins_per_trial):.0f}, "
        f"max {bins_per_trial.max()}; the 5 longest trials hold {top5:.0%} of all bins")

    _, scores = fit_pls(x[stim], y[stim], n_components)
    pred = heldout_prediction(x, y, groups, n_components, fit_rows=stim)
    acc_bin, acc_trial, trials, trial_pred, trial_y = accuracies(pred[stim], y[stim], groups[stim])
    null = shuffle_null(x[stim], y[stim], groups[stim], n_components, N_SHUFFLES, seed)
    acc_pre = None
    if pre.any():           # control: the same held-out models on bins before the grating is on
        acc_pre_bin, acc_pre = accuracies(pred[pre], y[pre], groups[pre])[:2]
        log(f"  pre-stimulus control ({pre_sec:g} s before onset, not used for the fit): per bin "
            f"{acc_pre_bin:.3f}, per trial {acc_pre:.3f}  (chance 0.5)")
    log(f"  {n_components} LVs, held-out balanced accuracy: per bin {acc_bin:.3f}, per trial {acc_trial:.3f}; "
        f"label shuffle per trial {null.mean():.3f} +- {null.std():.3f}, "
        f"p = {(1 + np.sum(null >= acc_trial)) / (1 + null.size):.3f}")
    for k in range(1, 7):
        a_bin, a_trial = accuracies(heldout_prediction(x[stim], y[stim], groups[stim], k), y[stim],
                                    groups[stim])[:2]
        log(f"    {k} LV(s): per bin {a_bin:.3f}, per trial {a_trial:.3f}")

    sortout = Path(config['sortout_folder'].replace('\\', '/'))
    out_dir = Path(out_dir) if out_dir else sortout / 'pls_task'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"bin{bin_ms:g}ms" + (f"_max{max_sec:g}s" if max_sec is not None else '')
    out = out_dir / f"pls_task_{sortout.name}_{tag}.png"
    window = 'stim on -> stim off' if max_sec is None else f'stim on -> min(stim off, {max_sec:g} s)'
    title = (f"{key}  PLS on z-scored {bin_ms:g} ms spike counts vs left grating:  {n_trials} trials, "
             f"{stim.sum()} bins x {x.shape[1]} units,  {n_components} LVs,  window {window}")
    stamp = (f"Repro: python {Path(__file__).name} {animal} {short_date(date)} --bin-ms {bin_ms:g} "
             f"--n-components {n_components} --max-sec {max_sec or 0:g} --pre-sec {pre_sec:g} --seed {seed} | "
             f"{datetime.now().isoformat(timespec='seconds')}\n"
             f"inputs: {' ; '.join(pkl_files)}\n"
             f"units: all {len(unit_ids)} in the pkl (uncurated), {(~kept).sum()} silent dropped; z-score per unit "
             f"over the stim bins; PLSRegression(scale=False), y = +1 (45 left) / -1 (315 left); held-out = "
             f"GroupKFold({N_FOLDS}) by trial; shuffle = {N_SHUFFLES} trial-label permutations; 5 longest trials "
             f"hold {top5:.0%} of bins; pre-stim bins only projected through the held-out models")
    core = stim | pre
    plot_pls(scores, {k: v[core] for k, v in rows.items()}, pred[core], acc_bin, acc_trial, null,
             (trials, trial_pred, trial_y), acc_pre, bin_sec, title, stamp, out)
    bias = np.full(len(y), np.nan)
    kde_out = bias_out = None
    if n_components >= KDE_DIMS:
        kdes = fit_kdes(scores[:, :KDE_DIMS], y[stim])
        bias = heldout_bias(counts, y, groups, stim, n_components)
        kde_bin, kde_trial = accuracies(bias[stim], y[stim], groups[stim])[:2]
        log(f"  KDE on LV1-{KDE_DIMS} (Scott factor {kdes[1].factor:.3f} / {kdes[-1].factor:.3f}), held-out "
            f"bias B = ln p(45 left) / p(315 left), sign on the stim bins: per bin {kde_bin:.3f}, per trial "
            f"{kde_trial:.3f}")
        log(f"    B on the stim bins: median |B| {np.median(np.abs(bias[stim])):.2f}, 1-99 percentiles "
            f"{np.percentile(bias[stim], 1):.1f} .. {np.percentile(bias[stim], 99):.1f}")
        if pre.any():
            pre_bin, pre_trial = accuracies(bias[pre], y[pre], groups[pre])[:2]
            log(f"    pre-stimulus control: per bin {pre_bin:.3f}, per trial {pre_trial:.3f}  (chance 0.5)")
        kde_stamp = (f"\nKDE: scipy gaussian_kde, Scott bandwidth, fit in LV1-{KDE_DIMS}. Bias: per GroupKFold({N_FOLDS}) "
                     f"fold, unit mean / sd + PLS + both KDEs from the training trials' stim bins (first "
                     f"{'all' if max_sec is None else format(max_sec, 'g')} s); test bins from {pre_sec:g} s before "
                     f"onset to {BIAS_POST_SEC:g} s after stim offset; natural log")
        bias_out = out.with_name(out.stem + '_bias.png')
        plot_bias(rows, bias, bin_sec, kde_bin, kde_trial, title, stamp + kde_stamp, bias_out)
        kde_out = out.with_name(out.stem + '_kde.png')
        plot_kde(scores[:, :KDE_DIMS], y[stim], kdes, kde_bin, kde_trial, title,
                 stamp + f"\nKDE: scipy gaussian_kde, Scott bandwidth (factor {kdes[1].factor:.3f} for 45, "
                         f"{kdes[-1].factor:.3f} for 315; kernel cov = factor^2 * cov of the condition's scores), fit "
                         f"in LV1-{KDE_DIMS}; panels = exact 2-D marginals; view = {KDE_VIEW_PCT} percentiles of the "
                         f"scores; ratio masked outside {KDE_RATIO_MASS:.0%} of the mixture mass, colour clipped at +-{KDE_RATIO_CLIP:g}", kde_out)
    else:
        log(f"  KDE and bias figures skipped: need --n-components >= {KDE_DIMS}")
    np.savez_compressed(out.with_suffix('.npz'), counts=counts, unit_ids=np.array(unit_ids), unit_kept=kept,
                        stim_scores=scores, heldout_prediction=pred, shuffle_trial_accuracy=null,
                        heldout_bias=bias, bin_sec=bin_sec, **rows)
    log(f"  saved {out}\n  saved {out.with_suffix('.npz')}" + (f"\n  saved {kde_out}\n  saved {bias_out}" if kde_out else ''))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('animal')
    parser.add_argument('date', help='YYMMDD')
    parser.add_argument('--bin-ms', type=float, default=100.0)
    parser.add_argument('--n-components', type=int, default=3)
    parser.add_argument('--max-sec', type=float, default=DEFAULT_MAX_SEC,
                        help='use at most this much of each stim period; 0 = all of it (default: %(default)s)')
    parser.add_argument('--pre-sec', type=float, default=DEFAULT_PRE_SEC,
                        help='pre-stimulus control: project this much time before stim onset through the '
                             'held-out models; 0 = off (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='seed of the label shuffle')
    parser.add_argument('--out-dir', default=None, help='default: <sortout>/pls_task')
    args = parser.parse_args(argv)
    run_day(args.animal, args.date, args.bin_ms, args.n_components, args.max_sec or None, args.pre_sec,
            args.seed, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
