"""
Train the best merged task+passive visual-stimulus decoder, apply it to a
continuous sleep/rest interval, and visualize predicted reactivation patterns.

Workflow
--------
1. Prepare task and passive stim-type data at several bin sizes.
2. Align units shared by task and passive recordings.
3. Evaluate all classifiers on merged task+passive data.
4. Pick the best classifier and its best bin size.
5. Refit that classifier on balanced merged data at the best bin size.
6. Bin sleep spikes over [sleep_start_sec, sleep_end_sec] in the same spike-time
   frame as the sleep pickle, then decode each sleep bin.
7. Report predicted +1 and -1 events and save summary figures.

Expected pickle format
----------------------
The task/passive/sleep pickle files should contain:
    data["spike_data"][unit_label]["spike_times_sec"]

For task/passive files, the existing prepare_* scripts also require
trial_params and window timing fields.
"""

import sys
import pickle
import json
from datetime import datetime
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'reactivation' / 'VStimOnDecoding'))
sys.path.insert(0, str(code_dir / 'DiscriminationTask' / 'grating'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import logsumexp
from scipy.signal import find_peaks
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

try:
    import umap
except ImportError:
    umap = None

from decode_aode import AODEClassifier, binarize
from decode_utils import (
    balance_by_undersampling,
    bin_spikes,
    make_classifiers,
    run_cv_balanced_train,
)
from prepare_passive_stimtype import prepare_passive_stim_type
from prepare_task_stimtype import prepare_task_stim_type
from UPState import detect_up_down_from_rates, plot_up_down_summary, print_up_down_report


from params import (
    task_pkl, passive_pkl, sleep_blocks,
    bin_sizes_ms, n_splits, random_state,
    class_pos, class_neg, TASK_COL_MAP, PASSIVE_COL_MAP,
    event_threshold, event_min_distance_sec, top_n_events_per_class,
    plot_window_sec,
)


USE_CACHED_BEST_MODEL = True

FIG_DPI = 300
FIG_EXPORT_FORMATS = ("png", "pdf")
PUB_COLORS = {
    -1: "#D55E00",
    0: "#6E6E6E",
    1: "#0072B2",
    "rate": "#3B4CC0",
    "threshold": "#222222",
}
PUB_CLASS_NAMES = {-1: "Stim -1", 0: "ITI", 1: "Stim +1"}
PATTERN_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue_white_orange",
    ["#2166AC", "#F7F7F7", "#B2182B"],
)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

UPDOWN_SMOOTH_SIGMA_SEC = 0.05
UPDOWN_DOWN_Z_THRESHOLD = -0.5
UPDOWN_UP_Z_THRESHOLD = 0.0
UPDOWN_DOWN_PERCENTILE = 20.0
UPDOWN_MIN_STATE_DURATION_SEC = 0.05
UPDOWN_MERGE_GAP_SEC = 0.03


def _softmax_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / np.maximum(exp_s.sum(axis=1, keepdims=True), 1e-12)


def _poisson_log_proba(clf, X):
    scores = X @ clf._log_lam.T - clf._lam_sum + clf._log_prior
    log_norm = logsumexp(scores, axis=1, keepdims=True)
    return scores - log_norm


def _fit_classifier(name, clf_proto, X, y):
    classes = sorted(int(c) for c in np.unique(y))
    if name == "AODE":
        X_bin = binarize(X).astype(np.float64)
        tevents = {str(c): X_bin[y == c][:, :, np.newaxis] for c in classes}
        prior_probs = {str(c): float(np.mean(y == c)) for c in classes}
        clf = AODEClassifier()
        clf.train(tevents)
        return {"name": name, "clf": clf, "classes": np.array(classes), "prior_probs": prior_probs}

    clf = clone(clf_proto)
    clf.fit(X, y)
    return {"name": name, "clf": clf, "classes": np.asarray(clf.classes_, dtype=int)}


def _predict_with_confidence(model, X):
    name = model["name"]
    clf = model["clf"]

    if name == "AODE":
        X_bin = binarize(X).astype(np.float64)
        logp = clf.predict_log_proba(X_bin, model["prior_probs"])
        proba = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
        classes = model["classes"]
        pred = classes[np.argmax(proba, axis=1)]
        return pred, proba, classes

    pred = clf.predict(X)
    classes = np.asarray(clf.classes_, dtype=int)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
    elif name == "Poisson NB":
        proba = np.exp(_poisson_log_proba(clf, X))
    elif hasattr(clf, "decision_function"):
        proba = _softmax_scores(clf.decision_function(X))
    else:
        proba = np.zeros((len(pred), len(classes)), dtype=float)
        for i, p in enumerate(pred):
            proba[i, np.where(classes == p)[0][0]] = 1.0

    return pred.astype(int), proba, classes


def _class_col(classes, label):
    matches = np.where(np.asarray(classes, dtype=int) == int(label))[0]
    if matches.size == 0:
        raise ValueError(f"Class {label} is missing from classifier classes {classes}.")
    return int(matches[0])


def _align_columns(X, units, common_units):
    idx = [units.index(u) for u in common_units]
    return X[:, idx]


def _prepare_merged_training(bin_size_sec):
    X_t, y_t, _, units_t = prepare_task_stim_type(
        task_pkl, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False,
        random_state=random_state,
    )
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_sec,
        balance_classes=False,
        random_state=random_state,
    )

    common_units = sorted(set(units_t) & set(units_p))
    if not common_units:
        raise RuntimeError("No common units between task and passive data.")

    X_t = _align_columns(X_t, units_t, common_units)
    X_p = _align_columns(X_p, units_p, common_units)
    X = np.vstack([X_t, X_p])
    y = np.concatenate([y_t, y_p])
    return X, y, common_units


def choose_best_merged_decoder():
    classifiers = make_classifiers(random_state)
    rows = []
    best = None

    for bms in bin_sizes_ms:
        print(f"\n=== Merged CV: bin {bms} ms ===")
        X, y, common_units = _prepare_merged_training(bms / 1000.0)
        print(
            f"  X={X.shape}  +1={np.sum(y == 1)}  "
            f"-1={np.sum(y == -1)}  0={np.sum(y == 0)}"
        )

        for name, clf_proto in classifiers.items():
            folds, mean, chance, pc_means, pc_stds = run_cv_balanced_train(
                name,
                clf_proto,
                X,
                y,
                n_splits=n_splits,
                random_state=random_state,
            )
            row = {
                "classifier": name,
                "bin_ms": bms,
                "mean_acc": mean,
                "std_acc": float(np.std(folds)),
                "chance": chance,
                "per_class_means": pc_means,
                "per_class_stds": pc_stds,
                "n_units": len(common_units),
            }
            rows.append(row)
            print(f"  [{name}] mean={mean:.3f} +/- {row['std_acc']:.3f}")
            if best is None or mean > best["mean_acc"]:
                best = row

    print(
        "\nBest merged decoder: "
        f"{best['classifier']} @ {best['bin_ms']} ms, acc={best['mean_acc']:.3f}"
    )
    return best, rows


def fit_best_model(best):
    X, y, common_units = _prepare_merged_training(best["bin_ms"] / 1000.0)
    rng = np.random.default_rng(random_state)
    X_bal, y_bal = balance_by_undersampling(X, y, rng)
    clf_proto = make_classifiers(random_state)[best["classifier"]]
    model = _fit_classifier(best["classifier"], clf_proto, X_bal, y_bal)
    return model, X, y, X_bal, y_bal, common_units


def _file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _model_cache_key():
    return {
        "task_pkl": _file_signature(task_pkl),
        "passive_pkl": _file_signature(passive_pkl),
        "bin_sizes_ms": list(bin_sizes_ms),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "class_pos": dict(class_pos),
        "class_neg": dict(class_neg),
        "task_col_map": dict(TASK_COL_MAP),
        "passive_col_map": dict(PASSIVE_COL_MAP),
    }


def _cache_matches(cached_key, current_key):
    return json.dumps(cached_key, sort_keys=True) == json.dumps(current_key, sort_keys=True)


def load_or_fit_best_model(cache_dir):
    cache_path = cache_dir / "best_merged_decoder_model.pkl"
    current_key = _model_cache_key()

    if USE_CACHED_BEST_MODEL and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if _cache_matches(cached.get("cache_key"), current_key):
                print(f"Using cached best merged decoder -> {cache_path}")
                return (
                    cached["best"],
                    cached["cv_rows"],
                    cached["model"],
                    cached["X_train"],
                    cached["y_train"],
                    cached["X_bal"],
                    cached["y_bal"],
                    cached["common_units"],
                )
            print(f"Cached decoder settings/data changed; refitting -> {cache_path}")
        except Exception as exc:
            print(f"Could not load cached decoder ({exc}); refitting.")

    best, cv_rows = choose_best_merged_decoder()
    model, X_train, y_train, X_bal, y_bal, common_units = fit_best_model(best)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "cache_key": current_key,
                "best": best,
                "cv_rows": cv_rows,
                "model": model,
                "X_train": X_train,
                "y_train": y_train,
                "X_bal": X_bal,
                "y_bal": y_bal,
                "common_units": common_units,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Cached best merged decoder -> {cache_path}")
    return best, cv_rows, model, X_train, y_train, X_bal, y_bal, common_units


def load_sleep_matrix(sleep_pkl_path, start_sec, end_sec, common_units, bin_size_sec):
    if not sleep_pkl_path:
        raise ValueError("Set sleep_pkl_path before running.")

    with open(sleep_pkl_path, "rb") as f:
        data = pickle.load(f)
    spike_data = data["spike_data"]

    if end_sec is None:
        end_sec = float(data.get("window", {}).get("window_duration_sec", 0.0))
    if start_sec is None:
        start_sec = 0.0
    if end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be greater than start_sec ({start_sec}).")

    missing = sorted(set(common_units) - set(spike_data))
    if missing:
        raise RuntimeError(
            f"Sleep data is missing {len(missing)} common training units. "
            f"First missing units: {missing[:10]}"
        )

    spike_data = {u: spike_data[u] for u in common_units}
    n_bins = int(np.floor((float(end_sec) - float(start_sec)) / bin_size_sec))
    if n_bins < 1:
        raise ValueError("Sleep interval is shorter than the selected decoder bin size.")
    edges = float(start_sec) + np.arange(n_bins + 1) * bin_size_sec
    X_sleep, units_sleep = bin_spikes(spike_data, edges, bin_size_sec)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return X_sleep, centers, units_sleep, float(start_sec), float(end_sec)


def find_sleep_events(centers, proba, classes):
    events = {}
    min_dist_bins = max(1, int(round(event_min_distance_sec / np.median(np.diff(centers)))))
    for label in (-1, 1):
        col = _class_col(classes, label)
        peaks, props = find_peaks(proba[:, col], height=event_threshold, distance=min_dist_bins)
        order = np.argsort(props["peak_heights"])[::-1]
        peaks = peaks[order]
        heights = props["peak_heights"][order]
        if top_n_events_per_class is not None:
            peaks = peaks[:top_n_events_per_class]
            heights = heights[:top_n_events_per_class]
        events[label] = [
            {"time_sec": float(centers[p]), "confidence": float(h), "bin_index": int(p)}
            for p, h in zip(peaks, heights)
        ]
    return events


def _training_unit_zscore_params(X_train):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = np.maximum(X_train.std(axis=0, keepdims=True), 1e-9)
    return mu, sd


def _apply_unit_zscore(X, mu, sd):
    return (np.asarray(X, dtype=float) - mu) / sd


def _zscore_rows(M):
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    mu = M.mean(axis=1, keepdims=True)
    sd = np.maximum(M.std(axis=1, keepdims=True), 1e-9)
    return (M - mu) / sd


def _class_templates(X_train, y_train):
    templates = {}
    for label in (-1, 0, 1):
        if np.any(y_train == label):
            templates[label] = X_train[y_train == label].mean(axis=0)
    return templates


def _figure_timestamp():
    return datetime.now().astimezone()


def _timestamped_figure_name(stem, timestamp, ext):
    return f"{stem}_{timestamp.strftime('%Y%m%d_%H%M%S')}.{ext}"


def _save_publication_figure(fig, out_dir, stem, timestamp):
    paths = []
    for ext in FIG_EXPORT_FORMATS:
        fig_path = out_dir / _timestamped_figure_name(stem, timestamp, ext)
        fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        paths.append(fig_path)
    return paths


def _format_pub_axis(ax, *, grid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8, color="0.2")
    if grid:
        ax.grid(True, axis="y", color="0.88", linewidth=0.6)


def _add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def _figure_repro_info(best, out_dir, timestamp, extra=""):
    info = (
        f"Generated {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
        f"script={Path(__file__).name} | block={out_dir.name}\n"
        f"decoder={best['classifier']} @ {best['bin_ms']} ms, merged_cv_acc={best['mean_acc']:.3f} | "
        f"event_threshold={event_threshold}, min_distance={event_min_distance_sec}s, "
        f"top_n_events_per_class={top_n_events_per_class}, random_state={random_state}\n"
        f"classes: +1={class_pos}, -1={class_neg}, 0=ITI | "
        f"task={Path(task_pkl).name}, passive={Path(passive_pkl).name}"
    )
    if extra:
        info += f"\n{extra}"
    return info


def _stamp_figure(fig, best, out_dir, timestamp, extra="", y=0.01):
    fig.text(
        0.01,
        y,
        _figure_repro_info(best, out_dir, timestamp, extra=extra),
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="0.35",
    )


def plot_sleep_summary(out_dir, best, centers, X_sleep, pred, proba, classes, events, X_train, y_train):
    c_pos = _class_col(classes, 1)
    c_neg = _class_col(classes, -1)
    c_iti = _class_col(classes, 0)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 0.75, 1.0], "hspace": 0.18},
    )

    axes[0].plot(centers, proba[:, c_pos], color=PUB_COLORS[1], linewidth=1.2, label=PUB_CLASS_NAMES[1])
    axes[0].plot(centers, proba[:, c_neg], color=PUB_COLORS[-1], linewidth=1.2, label=PUB_CLASS_NAMES[-1])
    axes[0].plot(centers, proba[:, c_iti], color=PUB_COLORS[0], linewidth=1.0, label=PUB_CLASS_NAMES[0])
    axes[0].axhline(event_threshold, color=PUB_COLORS["threshold"], linestyle=":", linewidth=1.0)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Decoder confidence")
    axes[0].set_title(
        f"Sleep decoding: {best['classifier']} @ {best['bin_ms']} ms "
        f"(merged CV acc={best['mean_acc']:.3f})"
    )
    axes[0].legend(loc="upper right", ncol=3, frameon=False, handlelength=1.8)
    _add_panel_label(axes[0], "A")

    y_plot = np.zeros_like(pred, dtype=float)
    y_plot[pred == 1] = 1
    y_plot[pred == -1] = -1
    axes[1].step(centers, y_plot, where="mid", color="0.05", linewidth=0.9)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels([PUB_CLASS_NAMES[-1], PUB_CLASS_NAMES[0], PUB_CLASS_NAMES[1]])
    axes[1].set_ylim(-1.35, 1.35)
    axes[1].set_ylabel("Predicted class")
    _add_panel_label(axes[1], "B")

    pop_rate = X_sleep.mean(axis=1)
    axes[2].plot(centers, pop_rate, color=PUB_COLORS["rate"], linewidth=1.0)
    axes[2].set_ylabel("Mean spike count/bin")
    _add_panel_label(axes[2], "C")

    for label, color in [(1, PUB_COLORS[1]), (-1, PUB_COLORS[-1])]:
        for e in events[label]:
            for ax in axes:
                ax.axvline(e["time_sec"], color=color, alpha=0.22, linewidth=0.7)

    axes[2].set_xlabel("Sleep time in spike-time frame (s)")
    for ax in axes:
        _format_pub_axis(ax, grid=True)

    timestamp = _figure_timestamp()
    counts = {label: int(np.sum(pred == label)) for label in (-1, 0, 1)}
    _stamp_figure(
        fig,
        best,
        out_dir,
        timestamp,
        extra=f"predicted bins: -1={counts[-1]}, 0={counts[0]}, +1={counts[1]} | n_sleep_bins={len(pred)}",
    )
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig_paths = _save_publication_figure(fig, out_dir, "sleep_decoding_summary", timestamp)
    plt.close(fig)
    print(f"Summary figure saved -> {fig_paths[0]} and {fig_paths[1]}")

    plot_event_patterns(out_dir, best, X_sleep, events, X_train, y_train)
    plot_event_population_clusters(out_dir, best, centers, X_sleep, events, proba, classes, X_train, y_train)
    plot_event_confidence_heatmaps(out_dir, best, centers, proba, classes, events)


def _save_event_pattern_figure(
    out_dir,
    fig_stem,
    best,
    title,
    colorbar_label,
    template_mat,
    pos_mat,
    neg_mat,
    template_order,
    label_names,
):
    n_pos = pos_mat.shape[0]
    n_neg = neg_mat.shape[0]
    n_template = template_mat.shape[0]
    n_total = n_template + n_pos + n_neg
    if n_total == 0:
        return

    height_ratios = [max(n_template, 1), max(n_pos, 1), max(n_neg, 1)]
    fig, axes = plt.subplots(
        3, 1,
        figsize=(7.2, 1.7 + 0.16 * n_total),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    vmax = 2.5
    cmap = PATTERN_CMAP

    axes[0].imshow(template_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[0].set_yticks(range(len(template_order)))
    axes[0].set_yticklabels([label_names[lab] for lab in template_order], fontsize=9)
    axes[0].set_title(title, fontsize=10)
    _add_panel_label(axes[0], "A")

    if n_pos:
        axes[1].imshow(pos_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[1].set_ylabel(f"+1 events (n={n_pos})", color=PUB_COLORS[1], fontsize=9)
        axes[1].set_yticks([0, n_pos - 1])
        axes[1].set_yticklabels(["best", "weakest"], fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No +1 events above threshold", ha="center", va="center")
        axes[1].set_axis_off()
    _add_panel_label(axes[1], "B")

    if n_neg:
        im = axes[2].imshow(neg_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[2].set_ylabel(f"-1 events (n={n_neg})", color=PUB_COLORS[-1], fontsize=9)
        axes[2].set_yticks([0, n_neg - 1])
        axes[2].set_yticklabels(["best", "weakest"], fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "No -1 events above threshold", ha="center", va="center")
        axes[2].set_axis_off()
        im = axes[0].images[0]
    _add_panel_label(axes[2], "C")

    axes[2].set_xlabel("Common units (sorted: -1 preferring  ->  +1 preferring)")
    for ax in axes:
        if ax.axison:
            _format_pub_axis(ax)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=colorbar_label)

    timestamp = _figure_timestamp()
    _stamp_figure(
        fig,
        best,
        out_dir,
        timestamp,
        extra=(
            f"figure={fig_stem} | templates={n_template}, +1_events={n_pos}, -1_events={n_neg} | "
            f"normalization={colorbar_label}"
        ),
    )
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.18, top=0.92, hspace=0.35)
    fig_paths = _save_publication_figure(fig, out_dir, fig_stem, timestamp)
    plt.close(fig)
    print(f"Event-pattern figure saved -> {fig_paths[0]} and {fig_paths[1]}")


def _high_conf_zero_events(centers, proba, classes):
    try:
        col = _class_col(classes, 0)
    except ValueError:
        return []

    min_dist_bins = max(1, int(round(event_min_distance_sec / np.median(np.diff(centers)))))
    peaks, props = find_peaks(proba[:, col], height=event_threshold, distance=min_dist_bins)
    if len(peaks) == 0:
        peaks = np.arange(proba.shape[0])
        heights = proba[:, col]
    else:
        heights = props["peak_heights"]

    order = np.argsort(heights)[::-1]
    peaks = peaks[order]
    heights = heights[order]
    if top_n_events_per_class is not None:
        peaks = peaks[:top_n_events_per_class]
        heights = heights[:top_n_events_per_class]

    return [
        {"time_sec": float(centers[p]), "confidence": float(h), "bin_index": int(p)}
        for p, h in zip(peaks, heights)
    ]


def _event_matrix(X_sleep, events, zero_events=None):
    rows = []
    labels = []
    confs = []
    times = []
    events_by_label = {
        -1: events.get(-1, []),
        0: zero_events or [],
        1: events.get(1, []),
    }
    for label in (-1, 0, 1):
        for event in events_by_label[label]:
            rows.append(X_sleep[event["bin_index"]])
            labels.append(label)
            confs.append(event["confidence"])
            times.append(event["time_sec"])
    if not rows:
        return (
            np.zeros((0, X_sleep.shape[1]), dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(confs, dtype=float),
        np.asarray(times, dtype=float),
    )


def _two_dim_pca(X):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=float), None
    n_comp = min(2, X.shape[0], X.shape[1])
    coords = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
    if n_comp == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(X.shape[0])])
    return coords, None


EVENT_COLORS = {-1: PUB_COLORS[-1], 0: PUB_COLORS[0], 1: PUB_COLORS[1]}
EVENT_MARKERS = {-1: "v", 0: "o", 1: "^"}


def _scatter_events(ax, coords, event_labels, confidence, title, value_label):
    for label in (-1, 0, 1):
        idx = np.where(event_labels == label)[0]
        if idx.size == 0:
            continue
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=EVENT_COLORS[label],
            marker=EVENT_MARKERS[label],
            s=38 + 80 * confidence[idx],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.45,
            label=f"event {label:+d}" if label else "event 0",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(value_label + " 1")
    ax.set_ylabel(value_label + " 2")
    _format_pub_axis(ax, grid=True)


def plot_event_population_clusters(out_dir, best, centers, X_sleep, events, proba, classes, X_train, y_train):
    zero_events = _high_conf_zero_events(centers, proba, classes)
    _plot_event_population_cluster_figure(
        out_dir,
        best,
        X_sleep,
        events,
        zero_events,
        X_train,
        y_train,
        fig_stem="sleep_event_population_clusters_with_class0",
        title_suffix="with class 0/ITI",
    )
    _plot_event_population_cluster_figure(
        out_dir,
        best,
        X_sleep,
        events,
        None,
        X_train,
        y_train,
        fig_stem="sleep_event_population_clusters_without_class0",
        title_suffix="without class 0/ITI",
    )


def _plot_event_population_cluster_figure(
    out_dir,
    best,
    X_sleep,
    events,
    zero_events,
    X_train,
    y_train,
    fig_stem,
    title_suffix,
):
    X_events, event_labels, event_conf, event_times = _event_matrix(X_sleep, events, zero_events)
    if X_events.shape[0] < 2:
        print(f"Event population clustering skipped ({title_suffix}) - fewer than 2 detected events.")
        return

    train_mu, train_sd = _training_unit_zscore_params(X_train)
    X_train_z = _apply_unit_zscore(X_train, train_mu, train_sd)
    X_events_z = _apply_unit_zscore(X_events, train_mu, train_sd)

    pca_coords, _ = _two_dim_pca(X_events_z)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes = axes.ravel()

    _scatter_events(axes[0], pca_coords, event_labels, event_conf, "PCA event map", "PC")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    _add_panel_label(axes[0], "A")

    if umap is not None and X_events_z.shape[0] >= 3:
        n_neighbors = max(2, min(15, X_events_z.shape[0] - 1))
        umap_coords = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(X_events_z)
        _scatter_events(axes[1], umap_coords, event_labels, event_conf, "UMAP event map", "UMAP")
    else:
        msg = "UMAP unavailable" if umap is None else "Need >=3 events for UMAP"
        axes[1].text(0.5, 0.5, msg, ha="center", va="center")
        axes[1].set_axis_off()
    _add_panel_label(axes[1], "B")

    if X_events_z.shape[0] >= 4:
        perplexity = max(2, min(10, X_events_z.shape[0] - 1))
        tsne_coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(X_events_z)
        _scatter_events(axes[2], tsne_coords, event_labels, event_conf, "t-SNE event map", "t-SNE")
    else:
        axes[2].text(0.5, 0.5, "Need >=4 events for t-SNE", ha="center", va="center")
        axes[2].set_axis_off()
    _add_panel_label(axes[2], "C")

    classes = np.unique(y_train)
    lda_components = min(2, len(classes) - 1, X_train_z.shape[1])
    if lda_components >= 1 and X_events_z.shape[0] >= 2:
        lda = LinearDiscriminantAnalysis(n_components=lda_components)
        lda.fit(X_train_z, y_train)
        lda_coords = lda.transform(X_events_z)
        if lda_components == 1:
            lda_coords = np.column_stack([lda_coords[:, 0], np.zeros(X_events_z.shape[0])])
        _scatter_events(axes[3], lda_coords, event_labels, event_conf, "Supervised LDA projection", "LD")

        template_z = _class_templates(X_train_z, y_train)
        template_order = [lab for lab in (-1, 0, 1) if lab in template_z]
        template_mat = np.vstack([template_z[lab] for lab in template_order])
        template_coords = lda.transform(template_mat)
        if lda_components == 1:
            template_coords = np.column_stack([template_coords[:, 0], np.zeros(template_mat.shape[0])])
        for lab, xy in zip(template_order, template_coords):
            axes[3].scatter(
                xy[0],
                xy[1],
                marker="*",
                s=180,
                color=EVENT_COLORS[lab],
                edgecolor="white",
                linewidth=0.7,
            )
            axes[3].text(xy[0], xy[1], f" {lab:+d}" if lab else " 0", fontsize=8, va="center")
    else:
        axes[3].text(0.5, 0.5, "LDA unavailable", ha="center", va="center")
        axes[3].set_axis_off()
    _add_panel_label(axes[3], "D")

    counts = {label: int(np.sum(event_labels == label)) for label in (-1, 0, 1)}
    timestamp = _figure_timestamp()
    fig.suptitle(
        f"Detected sleep-event population clusters {title_suffix} "
        f"(n={X_events_z.shape[0]}; per-unit training z-score)",
        fontsize=12,
    )
    _stamp_figure(
        fig,
        best,
        out_dir,
        timestamp,
        extra=(
            f"figure={fig_stem} | plotted events: -1={counts[-1]}, 0={counts[0]}, +1={counts[1]} | "
            f"normalization=per-unit z-score from balanced merged training bins | "
            f"colors show decoder labels: -1={EVENT_COLORS[-1]}, 0={EVENT_COLORS[0]}, +1={EVENT_COLORS[1]} | "
            f"embeddings shown: PCA, UMAP, t-SNE, supervised LDA"
        ),
    )
    fig.tight_layout(rect=[0, 0.1, 1, 0.96])
    fig_paths = _save_publication_figure(fig, out_dir, fig_stem, timestamp)
    plt.close(fig)
    print(f"Event population cluster figure saved -> {fig_paths[0]} and {fig_paths[1]}")


def plot_event_patterns(out_dir, best, X_sleep, events, X_train, y_train):
    """Separate figure: training templates + sleep-event firing patterns,
    saved with both per-unit and row-wise z-scoring."""
    train_mu, train_sd = _training_unit_zscore_params(X_train)
    X_train_z = _apply_unit_zscore(X_train, train_mu, train_sd)
    X_sleep_z = _apply_unit_zscore(X_sleep, train_mu, train_sd)

    unit_templates = _class_templates(X_train_z, y_train)
    raw_templates = _class_templates(X_train, y_train)
    if -1 not in unit_templates or 1 not in unit_templates:
        print("Event-pattern figure skipped - training set lacks +1 or -1 class.")
        return

    diff = unit_templates[1] - unit_templates[-1]
    unit_order = np.argsort(diff)  # -1 preferring on the left, +1 on the right

    template_order = [lab for lab in (-1, 0, 1) if lab in unit_templates]
    label_names = {-1: "template -1", 0: "template 0 (ITI)", 1: "template +1"}
    ev_pos = [e["bin_index"] for e in events.get(1, [])]
    ev_neg = [e["bin_index"] for e in events.get(-1, [])]

    template_mat_unit_z = np.vstack([unit_templates[lab][unit_order] for lab in template_order])
    pos_mat_unit_z = X_sleep_z[ev_pos][:, unit_order] if ev_pos else np.zeros((0, X_sleep.shape[1]))
    neg_mat_unit_z = X_sleep_z[ev_neg][:, unit_order] if ev_neg else np.zeros((0, X_sleep.shape[1]))
    _save_event_pattern_figure(
        out_dir,
        "sleep_event_patterns_unit_zscore",
        best,
        "Training templates (each unit z-scored from training bins; units sorted by +1 minus -1 template)",
        "z-score by each unit's training activity",
        template_mat_unit_z,
        pos_mat_unit_z,
        neg_mat_unit_z,
        template_order,
        label_names,
    )

    template_mat_row_z = _zscore_rows(np.vstack([raw_templates[lab][unit_order] for lab in template_order]))
    pos_mat_row_z = _zscore_rows(X_sleep[ev_pos][:, unit_order]) if ev_pos else np.zeros((0, X_sleep.shape[1]))
    neg_mat_row_z = _zscore_rows(X_sleep[ev_neg][:, unit_order]) if ev_neg else np.zeros((0, X_sleep.shape[1]))
    _save_event_pattern_figure(
        out_dir,
        "sleep_event_patterns_across_units_zscore",
        best,
        "Training templates (each row z-scored across units; units sorted by +1 minus -1 template)",
        "z-score across units within each row",
        template_mat_row_z,
        pos_mat_row_z,
        neg_mat_row_z,
        template_order,
        label_names,
    )

def plot_event_confidence_heatmaps(out_dir, best, centers, proba, classes, events):
    """Per-class event-centered decoder confidence heatmap."""
    half_bins = max(1, int(round(plot_window_sec / (best["bin_ms"] / 1000.0))))
    for label, color_name in [(1, "pos"), (-1, "neg")]:
        idx = [e["bin_index"] for e in events[label]]
        snippets = []
        for p in idx:
            lo = p - half_bins
            hi = p + half_bins + 1
            if lo >= 0 and hi <= len(centers):
                snippets.append(proba[lo:hi, _class_col(classes, label)])
        if not snippets:
            continue
        snippets = np.asarray(snippets)
        rel_t = (np.arange(snippets.shape[1]) - half_bins) * (best["bin_ms"] / 1000.0)

        fig_e, ax_e = plt.subplots(figsize=(5.8, max(2.6, 0.18 * len(snippets))))
        im = ax_e.imshow(
            snippets,
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=[rel_t[0], rel_t[-1], len(snippets), 0],
        )
        ax_e.axvline(0, color="white", linewidth=1)
        ax_e.set_title(f"{PUB_CLASS_NAMES[label]} event-centered decoder confidence")
        ax_e.set_xlabel("Time from event peak (s)")
        ax_e.set_ylabel("Events sorted by confidence")
        _format_pub_axis(ax_e)
        cbar = fig_e.colorbar(im, ax=ax_e, pad=0.015, fraction=0.045)
        cbar.set_label("Decoder confidence")

        timestamp = _figure_timestamp()
        _stamp_figure(
            fig_e,
            best,
            out_dir,
            timestamp,
            extra=(
                f"figure=sleep_class_{color_name}_event_heatmap | class={label:+d} | "
                f"events_in_heatmap={len(snippets)} | window=+/-{plot_window_sec}s"
            ),
        )
        fig_e.tight_layout(rect=[0, 0.18, 1, 1])
        fig_e_paths = _save_publication_figure(fig_e, out_dir, f"sleep_class_{color_name}_event_heatmap", timestamp)
        plt.close(fig_e)
        print(f"Event heatmap saved -> {fig_e_paths[0]} and {fig_e_paths[1]}")


def print_report(best, events, pred, proba, classes, centers, start_sec, end_sec):
    print("\n=== Sleep decoding report ===")
    print(f"Best classifier : {best['classifier']}")
    print(f"Best bin size   : {best['bin_ms']} ms")
    print(f"Merged CV acc   : {best['mean_acc']:.3f} +/- {best['std_acc']:.3f}")
    print(f"Sleep interval  : {start_sec:.3f} to {end_sec:.3f} s")
    print(f"Sleep bins      : {len(centers)}")

    for label in (1, -1, 0):
        n = int(np.sum(pred == label))
        frac = n / max(len(pred), 1)
        mean_conf = float(np.mean(proba[:, _class_col(classes, label)]))
        print(f"Predicted {label:+d}: {n} bins ({frac:.2%}), mean confidence={mean_conf:.3f}")

    for label in (1, -1):
        print(f"\nTop stim-on {label:+d} events above threshold {event_threshold}:")
        if not events[label]:
            print("  none")
            continue
        for e in events[label][:10]:
            print(f"  t={e['time_sec']:.3f} s  confidence={e['confidence']:.3f}")


def main():
    session = Path(task_pkl).parent.name
    base_out_dir = Path(task_pkl).parent / "reactivation" / f"sleep_merged_decoder_{session}"
    best, cv_rows, model, X_train, y_train, X_bal, y_bal, common_units = load_or_fit_best_model(base_out_dir)

    training_confusion = confusion_matrix(
        y_bal, _predict_with_confidence(model, X_bal)[0], labels=[-1, 0, 1]
    )

    for label, pkl_path, start_sec, end_sec in sleep_blocks:
        if not pkl_path:
            print(f"\n[{label}] Skipped - no sleep_pkl path.")
            continue

        print(f"\n========== Decoding sleep block '{label}' ==========")
        X_sleep, centers, _, start_sec_eff, end_sec_eff = load_sleep_matrix(
            pkl_path, start_sec, end_sec, common_units, best["bin_ms"] / 1000.0
        )
        pred, proba, classes = _predict_with_confidence(model, X_sleep)
        events = find_sleep_events(centers, proba, classes)
        updown = detect_up_down_from_rates(
            X_sleep,
            centers,
            best["bin_ms"] / 1000.0,
            smooth_sigma_sec=UPDOWN_SMOOTH_SIGMA_SEC,
            down_z_threshold=UPDOWN_DOWN_Z_THRESHOLD,
            up_z_threshold=UPDOWN_UP_Z_THRESHOLD,
            down_percentile=UPDOWN_DOWN_PERCENTILE,
            min_state_duration_sec=UPDOWN_MIN_STATE_DURATION_SEC,
            merge_gap_sec=UPDOWN_MERGE_GAP_SEC,
        )

        out_dir = base_out_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print_report(best, events, pred, proba, classes, centers, start_sec_eff, end_sec_eff)
        plot_sleep_summary(out_dir, best, centers, X_sleep, pred, proba, classes, events, X_bal, y_bal)
        updown_fig = plot_up_down_summary(
            out_dir, centers, updown, decoder_pred=pred, decoder_proba=proba, classes=classes
        )
        print_up_down_report(updown, figure_path=updown_fig)

        out = {
            "best": best,
            "cv_rows": cv_rows,
            "common_units": common_units,
            "sleep_label": label,
            "sleep_pkl": pkl_path,
            "sleep_start_sec": start_sec_eff,
            "sleep_end_sec": end_sec_eff,
            "sleep_centers_sec": centers,
            "sleep_pred": pred,
            "sleep_proba": proba,
            "classes": classes,
            "events": events,
            "updown": updown,
            "event_threshold": event_threshold,
            "event_min_distance_sec": event_min_distance_sec,
            "training_confusion_balanced": training_confusion,
        }
        pkl_out = out_dir / "sleep_decoding_results.pkl"
        with open(pkl_out, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved -> {pkl_out}")

    plt.show()


if __name__ == "__main__":
    main()
