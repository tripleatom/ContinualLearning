"""
Apply the best merged-context visual-stimulus decoder (trained by
train_task_test_passive.py) to sleep blocks, and visualize predicted
reactivation patterns.

Workflow
--------
1. Locate the CV results pickle saved by train_task_test_passive.py
   (same out_dir / stem convention as that script).
2. Read best_per_condition[MERGED_CONDITION] to get the winning
   (classifier, bin_ms) for the merged task+passive decoder.
   Default condition is 'merged_neural_cv' — neural-only merged decoder,
   no kinematic columns, which is the natural input shape for sleep
   (sleep has no kinematics). Switch to 'merged_cv' to additionally
   zero-fill the kinematic columns at sleep time, matching how the
   passive half is zero-filled at train time.
3. Rebuild the merged training matrix at that bin size, aligned to the
   common task n passive units.
4. Re-balance via (class x group) undersampling (same scheme as the CV
   training folds in train_task_test_passive.py) and refit the chosen
   classifier on the full balanced pool.
5. For each sleep block: bin spikes at the same bin size in the sleep
   pkl's own spike-time frame, decode each bin, locate reactivation
   events, save figures + a results pickle.
"""

import sys
import pickle
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'reactivation' / 'VStimOnDecoding'))
sys.path.insert(0, str(code_dir / 'DiscriminationTask' / 'grating'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.signal import find_peaks
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from decode_aode import AODEClassifier, binarize
from decode_utils import (
    balance_by_class_group_undersampling,
    bin_spikes,
    make_classifiers,
)
from prepare_passive_stimtype import prepare_passive_stim_type
from prepare_task_stimtype import prepare_task_stim_type
from kinematics_utils import (
    N_KINEMATIC_FEATURES,
    build_task_kinematics,
    load_task_kinematic_samples,
)

from params import (
    task_pkl, passive_pkl, sleep_blocks,
    n_splits, random_state,
    class_pos, class_neg, TASK_COL_MAP, PASSIVE_COL_MAP,
    event_threshold, event_min_distance_sec, top_n_events_per_class,
    plot_window_sec,
)


# Which merged condition (from train_task_test_passive.py) to apply to sleep:
#   'merged_neural_cv' : task neural + passive neural, no kin columns (default)
#   'merged_cv'        : task (neural+kin) + passive (neural + kin=0);
#                        sleep is decoded with kin columns zero-filled
MERGED_CONDITION = 'merged_neural_cv'


def _stem_part(d):
    return '_'.join(f'{k}{v:g}' for k, v in d.items())


def _cv_pkl_path():
    session = Path(task_pkl).parent.name
    stem = (f"train_task_test_passive_{session}_pos-{_stem_part(class_pos)}_"
            f"neg-{_stem_part(class_neg)}_{n_splits}fold")
    return Path(task_pkl).parent / 'reactivation' / f"{stem}.pkl"


def load_best_merged_from_cv(condition=MERGED_CONDITION):
    """Load the saved CV summary and return best_per_condition[condition]."""
    p = _cv_pkl_path()
    if not p.exists():
        raise FileNotFoundError(
            f"CV results pickle not found at:\n  {p}\n"
            "Run reactivation/VStimOnDecoding/train_task_test_passive.py first."
        )
    with open(p, 'rb') as f:
        cv = pickle.load(f)
    if condition not in cv.get('best_per_condition', {}):
        raise KeyError(
            f"Condition '{condition}' missing from best_per_condition in {p}. "
            f"Available: {sorted(cv.get('best_per_condition', {}).keys())}"
        )
    best = dict(cv['best_per_condition'][condition])
    best['condition'] = condition
    print(f"Loaded CV results from:\n  {p}")
    print(f"Best {condition}: {best['classifier']} @ {best['bin_ms']} ms "
          f"-> acc={best['mean_acc']:.3f} +/- {best['std_acc']:.3f}")
    print(f"Best overall (across all conditions): "
          f"{cv['best_overall']['condition']} / "
          f"{cv['best_overall']['classifier']} @ "
          f"{cv['best_overall']['bin_ms']} ms = "
          f"{cv['best_overall']['mean_acc']:.3f}")
    return best, cv


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
        return {"name": name, "clf": clf, "classes": np.array(classes),
                "prior_probs": prior_probs}

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


def _prepare_merged_training(bin_size_sec, condition):
    """Rebuild the merged training matrix for the chosen condition,
    matching train_task_test_passive.py."""
    X_t, y_t, bc_t, units_t = prepare_task_stim_type(
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
        raise RuntimeError("No common units between task and passive recordings.")
    X_t_a = _align_columns(X_t, units_t, common_units)
    X_p_a = _align_columns(X_p, units_p, common_units)

    n_kin = 0
    if condition == 'merged_cv':
        kin_samples, _, _ = load_task_kinematic_samples(task_pkl)
        kin_t, keep_t = build_task_kinematics(bc_t, bin_size_sec, kin_samples)
        n_kin = N_KINEMATIC_FEATURES
        X_t_a = X_t_a[keep_t]
        y_t = y_t[keep_t]
        kin_t = kin_t[keep_t]
        kin_p_zero = np.zeros((X_p_a.shape[0], n_kin), dtype=np.float64)
        X_t_use = np.hstack([X_t_a, kin_t.astype(X_t_a.dtype)])
        X_p_use = np.hstack([X_p_a, kin_p_zero.astype(X_p_a.dtype)])
    elif condition == 'merged_neural_cv':
        X_t_use = X_t_a
        X_p_use = X_p_a
    else:
        raise ValueError(
            f"Unsupported MERGED_CONDITION={condition!r}. "
            "Use 'merged_neural_cv' or 'merged_cv'."
        )

    X = np.vstack([X_t_use, X_p_use])
    y = np.concatenate([y_t, y_p])
    groups = np.concatenate([
        np.zeros(X_t_use.shape[0], dtype=int),
        np.ones(X_p_use.shape[0], dtype=int),
    ])
    return X, y, groups, common_units, n_kin


def fit_best_model(best, condition):
    X, y, groups, common_units, n_kin = _prepare_merged_training(
        best['bin_ms'] / 1000.0, condition,
    )
    print(f"Merged training pool ({condition}): X={X.shape}, "
          f"task bins={int(np.sum(groups == 0))}, "
          f"passive bins={int(np.sum(groups == 1))}")

    rng = np.random.default_rng(random_state)
    X_bal, y_bal, g_bal = balance_by_class_group_undersampling(X, y, groups, rng)
    cell_counts = {
        (int(c), int(g)): int(np.sum((y_bal == c) & (g_bal == g)))
        for c in np.unique(y_bal) for g in (0, 1)
    }
    print(f"After (class x group) balancing: X={X_bal.shape}, "
          f"per-cell counts={cell_counts}")

    clf_proto = make_classifiers(random_state)[best['classifier']]
    model = _fit_classifier(best['classifier'], clf_proto, X_bal, y_bal)
    return model, X, y, X_bal, y_bal, common_units, n_kin


def load_sleep_matrix(sleep_pkl_path, start_sec, end_sec, common_units,
                      bin_size_sec, n_kin_zero_fill):
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
    if n_kin_zero_fill > 0:
        X_sleep = np.hstack([
            X_sleep,
            np.zeros((X_sleep.shape[0], n_kin_zero_fill), dtype=X_sleep.dtype),
        ])
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


def _class_templates(X_train, y_train):
    templates = {}
    for label in (-1, 0, 1):
        if np.any(y_train == label):
            templates[label] = X_train[y_train == label].mean(axis=0)
    return templates


def plot_sleep_summary(out_dir, best, centers, X_sleep, pred, proba, classes, events,
                       X_train, y_train):
    c_pos = _class_col(classes, 1)
    c_neg = _class_col(classes, -1)
    c_iti = _class_col(classes, 0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(centers, proba[:, c_pos], color="#2c7fb8", label="+1")
    axes[0].plot(centers, proba[:, c_neg], color="#d95f0e", label="-1")
    axes[0].plot(centers, proba[:, c_iti], color="0.45", label="ITI")
    axes[0].axhline(event_threshold, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Decoder confidence")
    axes[0].set_title(
        f"Sleep decoding: {best['classifier']} @ {best['bin_ms']} ms "
        f"(merged CV acc={best['mean_acc']:.3f}, condition={best['condition']})"
    )
    axes[0].legend(loc="upper right", ncol=3)

    y_plot = np.zeros_like(pred, dtype=float)
    y_plot[pred == 1] = 1
    y_plot[pred == -1] = -1
    axes[1].step(centers, y_plot, where="mid", color="black", linewidth=0.8)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["-1", "ITI/0", "+1"])
    axes[1].set_ylabel("Predicted class")

    pop_rate = X_sleep.mean(axis=1)
    axes[2].plot(centers, pop_rate, color="#4C72B0", linewidth=0.9)
    axes[2].set_ylabel("Mean firing rate")

    for label, color in [(1, "#2c7fb8"), (-1, "#d95f0e")]:
        for e in events[label]:
            for ax in axes:
                ax.axvline(e["time_sec"], color=color, alpha=0.25, linewidth=0.8)

    axes[2].set_xlabel("Sleep time in spike-time frame (s)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig_path = out_dir / "sleep_decoding_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Summary figure saved -> {fig_path}")

    plot_event_patterns(out_dir, X_sleep, events, X_train, y_train)
    plot_event_confidence_heatmaps(out_dir, best, centers, proba, classes, events)


def _zscore_rows(M):
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    mu = M.mean(axis=1, keepdims=True)
    sd = np.maximum(M.std(axis=1, keepdims=True), 1e-9)
    return (M - mu) / sd


def plot_event_patterns(out_dir, X_sleep, events, X_train, y_train):
    """Training templates + sleep-event firing patterns, z-scored per row,
    with units sorted by (+1 minus -1) template."""
    templates = _class_templates(X_train, y_train)
    if -1 not in templates or 1 not in templates:
        print("Event-pattern figure skipped - training set lacks +1 or -1 class.")
        return

    diff = templates[1] - templates[-1]
    unit_order = np.argsort(diff)  # -1 preferring on the left, +1 on the right

    template_order = [lab for lab in (-1, 0, 1) if lab in templates]
    label_names = {-1: "template -1", 0: "template 0 (ITI)", 1: "template +1"}
    template_mat = np.vstack([templates[lab][unit_order] for lab in template_order])
    template_mat_z = _zscore_rows(template_mat)

    ev_pos = [e["bin_index"] for e in events.get(1, [])]
    ev_neg = [e["bin_index"] for e in events.get(-1, [])]
    pos_mat = _zscore_rows(X_sleep[ev_pos][:, unit_order]) if ev_pos else np.zeros((0, X_sleep.shape[1]))
    neg_mat = _zscore_rows(X_sleep[ev_neg][:, unit_order]) if ev_neg else np.zeros((0, X_sleep.shape[1]))

    n_pos = pos_mat.shape[0]
    n_neg = neg_mat.shape[0]
    n_template = template_mat_z.shape[0]
    n_total = n_template + n_pos + n_neg
    if n_total == 0:
        return

    height_ratios = [max(n_template, 1), max(n_pos, 1), max(n_neg, 1)]
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 1.6 + 0.18 * n_total),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    vmax = 2.5
    cmap = "coolwarm"

    axes[0].imshow(template_mat_z, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(range(len(template_order)))
    axes[0].set_yticklabels([label_names[lab] for lab in template_order], fontsize=9)
    axes[0].set_title("Training templates (rows z-scored across units; "
                      "units sorted by +1 minus -1 template)",
                      fontsize=10)

    if n_pos:
        axes[1].imshow(pos_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[1].set_ylabel(f"+1 events (n={n_pos})", color="#2c7fb8", fontsize=10)
        axes[1].set_yticks([0, n_pos - 1])
        axes[1].set_yticklabels(["best", "weakest"], fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No +1 events above threshold", ha="center", va="center")
        axes[1].set_axis_off()

    if n_neg:
        im = axes[2].imshow(neg_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[2].set_ylabel(f"-1 events (n={n_neg})", color="#d95f0e", fontsize=10)
        axes[2].set_yticks([0, n_neg - 1])
        axes[2].set_yticklabels(["best", "weakest"], fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "No -1 events above threshold", ha="center", va="center")
        axes[2].set_axis_off()
        im = axes[0].images[0]

    axes[2].set_xlabel("Common units (sorted: -1 preferring  ->  +1 preferring)")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="z-score across units")

    fig_path = out_dir / "sleep_event_patterns.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Event-pattern figure saved -> {fig_path}")


def plot_event_confidence_heatmaps(out_dir, best, centers, proba, classes, events):
    """Per-class event-centered decoder confidence heatmap (one PNG per class)."""
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

        fig_e, ax_e = plt.subplots(figsize=(8, max(3, 0.25 * len(snippets))))
        ax_e.imshow(snippets, aspect="auto", cmap="magma", vmin=0, vmax=1,
                    extent=[rel_t[0], rel_t[-1], len(snippets), 0])
        ax_e.axvline(0, color="white", linewidth=1)
        ax_e.set_title(f"Class {label} event-centered decoder confidence")
        ax_e.set_xlabel("Time from event peak (s)")
        ax_e.set_ylabel("Events sorted by confidence")
        fig_e.tight_layout()
        fig_e_path = out_dir / f"sleep_class_{color_name}_event_heatmap.png"
        fig_e.savefig(fig_e_path, dpi=150, bbox_inches="tight")
        print(f"Event heatmap saved -> {fig_e_path}")


def print_report(best, events, pred, proba, classes, centers, start_sec, end_sec):
    print("\n=== Sleep decoding report ===")
    print(f"Condition       : {best['condition']}")
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
    best, cv = load_best_merged_from_cv(MERGED_CONDITION)
    model, X_train, y_train, X_bal, y_bal, common_units, n_kin = fit_best_model(
        best, MERGED_CONDITION,
    )

    session = Path(task_pkl).parent.name
    base_out_dir = (Path(task_pkl).parent / "reactivation"
                    / f"sleep_{MERGED_CONDITION}_{session}")

    training_confusion = confusion_matrix(
        y_bal, _predict_with_confidence(model, X_bal)[0], labels=[-1, 0, 1]
    )

    for label, pkl_path, start_sec, end_sec in sleep_blocks:
        if not pkl_path:
            print(f"\n[{label}] Skipped - no sleep_pkl path.")
            continue

        print(f"\n========== Decoding sleep block '{label}' ==========")
        X_sleep, centers, _, start_sec_eff, end_sec_eff = load_sleep_matrix(
            pkl_path, start_sec, end_sec, common_units,
            best["bin_ms"] / 1000.0, n_kin_zero_fill=n_kin,
        )
        pred, proba, classes = _predict_with_confidence(model, X_sleep)
        events = find_sleep_events(centers, proba, classes)

        out_dir = base_out_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print_report(best, events, pred, proba, classes, centers, start_sec_eff, end_sec_eff)
        plot_sleep_summary(out_dir, best, centers, X_sleep, pred, proba, classes, events,
                           X_bal, y_bal)

        out = {
            "best": best,
            "cv_best_per_condition": cv['best_per_condition'],
            "cv_best_overall": cv['best_overall'],
            "merged_condition": MERGED_CONDITION,
            "common_units": common_units,
            "n_kin_zero_fill": n_kin,
            "sleep_label": label,
            "sleep_pkl": pkl_path,
            "sleep_start_sec": start_sec_eff,
            "sleep_end_sec": end_sec_eff,
            "sleep_centers_sec": centers,
            "sleep_pred": pred,
            "sleep_proba": proba,
            "classes": classes,
            "events": events,
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
