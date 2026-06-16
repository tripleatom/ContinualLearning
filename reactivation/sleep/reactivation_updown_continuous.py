"""
Continuous reactivation score vs UP/DOWN state phase.

Motivation
----------
The merged decoder's best bin is 500 ms, so discrete 500-ms reactivation
"events" are coarser than the 10 ms UP/DOWN dynamics and cannot be aligned to
DOWN onset / UP onset / UP center (see reactivation_updown_alignment.py). Here
we instead build a CONTINUOUS reactivation trace by sliding the trained 500 ms
decoder window in fine (10 ms) steps over the whole sleep block, then ask how
that trace co-varies with UP/DOWN state.

    r(t) = P(+1 | window @ t) + P(-1 | window @ t) = 1 - P(ITI | window @ t)

Window width is kept at the decoder's trained bin (500 ms) so the firing-rate
features stay in-distribution for the Random Forest; only the SAMPLING is fine.
This means r(t) is inherently low-pass filtered at ~the window width: it can
resolve which slow phase (UP vs DOWN, ramp direction) reactivation favors, but
NOT sub-500-ms onset-vs-center timing. That limit is fundamental to a 500 ms
decoder and is stated on every figure.

Analyses (all on the shared 10 ms grid, with circular-shift nulls)
    1. Landmark-triggered average of r(t) around DOWN onset, UP onset
       (=DOWN->UP transition), UP center, UP offset.
    2. r(t) profile across normalized UP-state and DOWN-state phase.
    3. Mean r in UP vs DOWN; Pearson corr of r(t) with population-MUA z.

Run (use the env that trained the model; it was pickled with sklearn 1.5.0)
    .../envs/ms10/python.exe reactivation/sleep/reactivation_updown_continuous.py
    ... reactivation/sleep/reactivation_updown_continuous.py --sleep-label pre --hop-ms 10
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(code_dir / "reactivation" / "VStimOnDecoding"))
sys.path.insert(0, str(Path(__file__).parent))

from UPState import detect_up_down_from_rates, _mask_to_events
from params import (
    sleep_blocks, task_pkl, random_state,
    updown_bin_ms, updown_smooth_sigma_ms, updown_down_z_threshold,
    updown_up_z_threshold, updown_down_percentile, updown_min_state_ms,
    updown_merge_gap_ms,
)

ETA_HALF_WINDOW_SEC = 0.75
N_SHUFFLES = 200
N_PHASE_BINS = 12
PREDICT_CHUNK = 100_000

LANDMARK_TYPES = [
    ("down_onset", "DOWN onset"),
    ("up_onset",   "UP onset = DOWN->UP transition"),
    ("up_center",  "UP-state center"),
    ("up_offset",  "UP offset = UP->DOWN transition"),
]


# --------------------------------------------------------------------------- #
#  Decoder probability (matches apply_merged_decoder_to_sleep _predict logic)  #
# --------------------------------------------------------------------------- #
def _proba(model, X):
    """Return (proba, classes). Handles the Random Forest path and AODE."""
    name = model["name"]
    clf = model["clf"]
    if name == "AODE":
        from decode_aode import binarize
        from scipy.special import logsumexp
        Xb = binarize(X).astype(np.float64)
        logp = clf.predict_log_proba(Xb, model["prior_probs"])
        proba = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
        return proba, np.asarray(model["classes"], dtype=int)
    classes = np.asarray(clf.classes_, dtype=int)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X), classes
    if hasattr(clf, "decision_function"):
        s = np.asarray(clf.decision_function(X), dtype=float)
        if s.ndim == 1:
            s = np.column_stack([-s, s])
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12), classes
    raise RuntimeError(f"Classifier {name} exposes no probability method.")


def _class_col(classes, label):
    return int(np.where(np.asarray(classes, dtype=int) == label)[0][0])


# --------------------------------------------------------------------------- #
#  Sliding-window firing-rate features                                         #
# --------------------------------------------------------------------------- #
def _build_slide_rates(spike_data, common_units, centers, window_sec, start_sec, end_sec):
    """Firing rate (spikes/s) per unit in a window of fixed width centered at
    each grid point. Window is clipped to [start, end] at the edges and the
    count is divided by the clipped width, keeping the feature in spikes/s."""
    half = window_sec / 2.0
    left = np.clip(centers - half, start_sec, end_sec)
    right = np.clip(centers + half, start_sec, end_sec)
    width = np.maximum(right - left, 1e-9)
    X = np.empty((centers.size, len(common_units)), dtype=np.float32)
    for j, u in enumerate(common_units):
        st = np.asarray(spike_data[u].get("spike_times_sec", []), dtype=float)
        if st.size:
            st.sort()
            cnt = np.searchsorted(st, right, side="left") - np.searchsorted(st, left, side="left")
        else:
            cnt = np.zeros(centers.size, dtype=float)
        X[:, j] = (cnt / width).astype(np.float32)
    return X


def _predict_proba_chunked(model, X):
    cols = None
    out = None
    for lo in range(0, X.shape[0], PREDICT_CHUNK):
        hi = min(lo + PREDICT_CHUNK, X.shape[0])
        p, classes = _proba(model, X[lo:hi])
        if out is None:
            cols = classes
            out = np.empty((X.shape[0], p.shape[1]), dtype=np.float32)
        out[lo:hi] = p
    return out, cols


# --------------------------------------------------------------------------- #
#  Analyses                                                                     #
# --------------------------------------------------------------------------- #
def _landmark_indices(updown, key):
    up = updown["events"]["up"]
    down = updown["events"]["down"]
    if key == "down_onset":
        return np.array([e["start_bin"] for e in down], dtype=int)
    if key == "up_onset":
        return np.array([e["start_bin"] for e in up], dtype=int)
    if key == "up_center":
        return np.array([int(round(0.5 * (e["start_bin"] + e["end_bin"]))) for e in up], dtype=int)
    if key == "up_offset":
        return np.array([e["end_bin"] for e in up], dtype=int)
    raise KeyError(key)


def _triggered_average(trace, idxs, half_bins):
    """Mean of trace in [-half_bins, +half_bins] around each index (full slices only)."""
    n = trace.size
    keep = idxs[(idxs - half_bins >= 0) & (idxs + half_bins < n)]
    if keep.size == 0:
        return np.full(2 * half_bins + 1, np.nan), 0
    offs = np.arange(-half_bins, half_bins + 1)
    mat = trace[keep[:, None] + offs[None, :]]
    return mat.mean(axis=0), keep.size


def _eta_with_null(trace, idxs, half_bins, rng):
    obs, n_used = _triggered_average(trace, idxs, half_bins)
    n = trace.size
    sh = np.empty((N_SHUFFLES, 2 * half_bins + 1), dtype=float)
    for s in range(N_SHUFFLES):
        shifted = np.roll(trace, int(rng.integers(1, n)))
        sh[s], _ = _triggered_average(shifted, idxs, half_bins)
    return obs, np.nanmean(sh, 0), np.nanpercentile(sh, 2.5, 0), np.nanpercentile(sh, 97.5, 0), n_used


def _phase_profile(trace, events, n_bins):
    """Mean trace across normalized 0->1 phase within each state event."""
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for e in events:
        a, b = e["start_bin"], e["end_bin"]
        if b <= a:
            continue
        idx = np.arange(a, b + 1)
        phase = (idx - a) / (b - a)
        bins = np.minimum((phase * n_bins).astype(int), n_bins - 1)
        np.add.at(acc, bins, trace[idx])
        np.add.at(cnt, bins, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return acc / np.maximum(cnt, 1)


def analyze_block(label, decoder_pkl, model, common_units, decoder_bin_ms, hop_ms, out_dir):
    with open(decoder_pkl, "rb") as f:
        dec = pickle.load(f)
    sleep_pkl = dec["sleep_pkl"]
    start_sec = float(dec["sleep_start_sec"])
    end_sec = float(dec["sleep_end_sec"])

    with open(sleep_pkl, "rb") as f:
        spike_data = pickle.load(f)["spike_data"]

    # ---- 10 ms UP/DOWN detection (population MUA over common units).
    ud_bin = updown_bin_ms / 1000.0
    n_ud = int(np.floor((end_sec - start_sec) / ud_bin))
    ud_edges = start_sec + np.arange(n_ud + 1) * ud_bin
    ud_centers = 0.5 * (ud_edges[:-1] + ud_edges[1:])
    X_ud = _build_slide_rates_boxcar(spike_data, common_units, ud_edges)
    updown = detect_up_down_from_rates(
        X_ud, ud_centers, ud_bin,
        smooth_sigma_sec=updown_smooth_sigma_ms / 1000.0,
        down_z_threshold=updown_down_z_threshold,
        up_z_threshold=updown_up_z_threshold,
        down_percentile=updown_down_percentile,
        min_state_duration_sec=updown_min_state_ms / 1000.0,
        merge_gap_sec=updown_merge_gap_ms / 1000.0,
    )

    # ---- Continuous reactivation trace on the SAME grid as UP/DOWN.
    hop_sec = hop_ms / 1000.0
    if abs(hop_sec - ud_bin) < 1e-9:
        centers = ud_centers
    else:
        n_h = int(np.floor((end_sec - start_sec) / hop_sec))
        centers = start_sec + (np.arange(n_h) + 0.5) * hop_sec
    print(f"  building sliding {decoder_bin_ms:.0f}ms window at {hop_ms:.0f}ms hop "
          f"-> {centers.size} samples x {len(common_units)} units")
    X_slide = _build_slide_rates(spike_data, common_units, centers,
                                 decoder_bin_ms / 1000.0, start_sec, end_sec)
    proba, classes = _predict_proba_chunked(model, X_slide)
    p_pos = proba[:, _class_col(classes, 1)]
    p_neg = proba[:, _class_col(classes, -1)]
    r = (p_pos + p_neg).astype(float)            # = 1 - P(ITI)

    # If hop != ud_bin, map UP/DOWN landmark/state onto the r grid by nearest bin.
    if centers.size == ud_centers.size and np.allclose(centers, ud_centers):
        state_label = np.asarray(updown["state_label"], dtype=int)
        pop_z = updown["population_mua_z"]
        up_events = updown["events"]["up"]
        down_events = updown["events"]["down"]
        landmark_idx = {k: _landmark_indices(updown, k) for k, _ in LANDMARK_TYPES}
    else:
        nn = np.searchsorted(ud_centers, centers).clip(0, ud_centers.size - 1)
        state_label = np.asarray(updown["state_label"], dtype=int)[nn]
        pop_z = np.interp(centers, ud_centers, updown["population_mua_z"])
        scale = ud_bin / hop_sec
        up_events = [{"start_bin": int(e["start_bin"] * scale), "end_bin": int(e["end_bin"] * scale)}
                     for e in updown["events"]["up"]]
        down_events = [{"start_bin": int(e["start_bin"] * scale), "end_bin": int(e["end_bin"] * scale)}
                       for e in updown["events"]["down"]]
        landmark_idx = {}
        for k, _ in LANDMARK_TYPES:
            li = _landmark_indices(updown, k)
            landmark_idx[k] = (li * scale).round().astype(int).clip(0, centers.size - 1)

    rng = np.random.default_rng(random_state)
    half_bins = max(1, int(round(ETA_HALF_WINDOW_SEC / hop_sec)))
    lag = (np.arange(-half_bins, half_bins + 1)) * hop_sec

    eta = {}
    for key, _name in LANDMARK_TYPES:
        obs, mean, lo, hi, n_used = _eta_with_null(r, landmark_idx[key], half_bins, rng)
        eta[key] = {"obs": obs, "null_mean": mean, "null_lo": lo, "null_hi": hi, "n_landmarks": int(n_used)}

    up_profile = _phase_profile(r, up_events, N_PHASE_BINS)
    down_profile = _phase_profile(r, down_events, N_PHASE_BINS)

    in_up = state_label == 1
    in_down = state_label == 0
    mean_r_up = float(np.mean(r[in_up])) if in_up.any() else float("nan")
    mean_r_down = float(np.mean(r[in_down])) if in_down.any() else float("nan")
    # circular-shift null for the UP-DOWN mean difference
    obs_diff = mean_r_up - mean_r_down
    null_diff = np.empty(N_SHUFFLES)
    for s in range(N_SHUFFLES):
        rr = np.roll(r, int(rng.integers(1, r.size)))
        mu = np.mean(rr[in_up]) if in_up.any() else np.nan
        md = np.mean(rr[in_down]) if in_down.any() else np.nan
        null_diff[s] = mu - md
    diff_p = float(np.mean(np.abs(null_diff) >= abs(obs_diff)))
    corr_r, corr_p = pearsonr(r, pop_z)

    results = {
        "label": label, "sleep_pkl": sleep_pkl,
        "decoder_bin_ms": decoder_bin_ms, "hop_ms": hop_ms, "updown_bin_ms": updown_bin_ms,
        "n_samples": int(centers.size), "n_units": len(common_units),
        "lag_sec": lag, "eta": eta,
        "phase_bins": (np.arange(N_PHASE_BINS) + 0.5) / N_PHASE_BINS,
        "up_phase_profile": up_profile, "down_phase_profile": down_profile,
        "mean_r_up": mean_r_up, "mean_r_down": mean_r_down,
        "up_down_diff": obs_diff, "up_down_diff_p": diff_p,
        "corr_r_popz": float(corr_r), "corr_r_popz_p": float(corr_p),
        "frac_up_time": float(np.mean(state_label == 1)),
        "frac_down_time": float(np.mean(state_label == 0)),
        "n_shuffles": N_SHUFFLES,
    }
    _plot(out_dir, label, results)
    _report(results)
    pkl_out = out_dir / "reactivation_updown_continuous_results.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved -> {pkl_out}")
    return results


def _build_slide_rates_boxcar(spike_data, common_units, edges):
    """Non-overlapping bin firing rates (for UP/DOWN detection)."""
    bin_w = float(edges[1] - edges[0])
    X = np.empty((edges.size - 1, len(common_units)), dtype=np.float32)
    for j, u in enumerate(common_units):
        st = np.asarray(spike_data[u].get("spike_times_sec", []), dtype=float)
        cnt, _ = np.histogram(st, bins=edges)
        X[:, j] = (cnt / bin_w).astype(np.float32)
    return X


def _stamp(label, results, timestamp):
    return (
        f"Generated {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')} | script={Path(__file__).name} | block={label}\n"
        f"continuous reactivation r(t)=P(+1)+P(-1)=1-P(ITI); sliding {results['decoder_bin_ms']:.0f}ms RF window "
        f"@ {results['hop_ms']:.0f}ms hop, {results['n_samples']} samples x {results['n_units']} units\n"
        f"UP/DOWN @ {results['updown_bin_ms']}ms (smooth={updown_smooth_sigma_ms}ms, down_z={updown_down_z_threshold}, "
        f"up_z={updown_up_z_threshold}, down_pct={updown_down_percentile}) | sleep_pkl={Path(results['sleep_pkl']).name}\n"
        f"NOTE: 500ms window low-passes r(t) ~2Hz -> resolves slow UP/DOWN phase, NOT sub-500ms onset timing | "
        f"null=circular-shift n={results['n_shuffles']}, random_state={random_state}"
    )


def _plot(out_dir, label, results):
    out_dir.mkdir(parents=True, exist_ok=True)
    lag = results["lag_sec"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, (key, name) in zip(axes[:4], LANDMARK_TYPES):
        e = results["eta"][key]
        ax.plot(lag, e["obs"], color="#2c7fb8", linewidth=1.6, label="observed r(t)")
        ax.plot(lag, e["null_mean"], color="crimson", linewidth=1.0, label="shuffle mean")
        ax.fill_between(lag, e["null_lo"], e["null_hi"], color="crimson", alpha=0.15, label="shuffle 95%")
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"{name}  (n={e['n_landmarks']})", fontsize=9)
        ax.set_xlabel("Time from landmark (s)")
        ax.set_ylabel("Reactivation score r(t)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)

    ax = axes[4]
    pb = results["phase_bins"]
    ax.plot(pb, results["up_phase_profile"], color="#2c7fb8", marker="o", ms=3, label="within UP")
    ax.plot(pb, results["down_phase_profile"], color="#d95f0e", marker="o", ms=3, label="within DOWN")
    ax.set_title("r(t) across normalized state phase\n(0=onset, 1=offset)", fontsize=9)
    ax.set_xlabel("Normalized state phase")
    ax.set_ylabel("Mean r(t)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[5]
    ax.bar([0, 1], [results["mean_r_up"], results["mean_r_down"]],
           color=["#2c7fb8", "#d95f0e"], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["UP", "DOWN"])
    ax.set_ylabel("Mean r(t)")
    ax.set_title(
        f"Mean reactivation by state\nUP-DOWN diff={results['up_down_diff']:+.3f} "
        f"(p={results['up_down_diff_p']:.3f})\ncorr(r, popMUAz)={results['corr_r_popz']:+.3f}",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Continuous reactivation vs UP/DOWN phase - block '{label}' "
        f"({results['n_samples']} samples @ {results['hop_ms']:.0f}ms)",
        fontsize=13,
    )
    ts = datetime.now().astimezone()
    fig.text(0.01, 0.005, _stamp(label, results, ts), ha="left", va="bottom", fontsize=7)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    fp = out_dir / f"reactivation_updown_continuous_{ts.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {fp}")


def _report(results):
    print(f"\n=== Continuous reactivation vs UP/DOWN: block '{results['label']}' ===")
    print(f"Samples={results['n_samples']} @ {results['hop_ms']:.0f}ms | "
          f"UP time={results['frac_up_time']:.1%}, DOWN time={results['frac_down_time']:.1%}")
    print(f"Mean r(t): UP={results['mean_r_up']:.3f}  DOWN={results['mean_r_down']:.3f}  "
          f"diff={results['up_down_diff']:+.3f} (shuffle p={results['up_down_diff_p']:.3f})")
    print(f"corr(r(t), population-MUA z) = {results['corr_r_popz']:+.3f} (p={results['corr_r_popz_p']:.1e})")
    direction = "DOWN" if results["up_down_diff"] < 0 else "UP"
    print(f"-> Continuous reactivation score is higher during {direction} states "
          f"({'significant' if results['up_down_diff_p'] < 0.05 else 'n.s.'} vs circular-shift null)")


def main():
    ap = argparse.ArgumentParser(description="Continuous reactivation score vs UP/DOWN phase.")
    ap.add_argument("--sleep-label", default=None)
    ap.add_argument("--hop-ms", type=float, default=updown_bin_ms, help="Sampling hop (default = UP/DOWN bin).")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    session = Path(task_pkl).parent.name
    decoder_base = (Path(args.results_dir) if args.results_dir
                    else Path(task_pkl).parent / "reactivation" / f"sleep_merged_decoder_{session}")
    out_base = Path(task_pkl).parent / "reactivation" / f"reactivation_updown_continuous_{session}"

    with open(decoder_base / "best_merged_decoder_model.pkl", "rb") as f:
        cache = pickle.load(f)
    model = cache["model"]
    common_units = cache["common_units"]
    decoder_bin_ms = float(cache["best"]["bin_ms"])
    print(f"Loaded {cache['best']['classifier']} @ {decoder_bin_ms:.0f}ms, {len(common_units)} units.")

    ran = False
    for label, pkl_path, _s, _e in sleep_blocks:
        if args.sleep_label is not None and label != args.sleep_label:
            continue
        if not pkl_path:
            print(f"\n[{label}] Skipped - no sleep_pkl path.")
            continue
        decoder_pkl = decoder_base / label / "sleep_decoding_results.pkl"
        if not decoder_pkl.exists():
            print(f"\n[{label}] Skipped - decoder results not found: {decoder_pkl}")
            continue
        print(f"\n========== Continuous reactivation: block '{label}' ==========")
        analyze_block(label, decoder_pkl, model, common_units, decoder_bin_ms,
                      args.hop_ms, out_base / label)
        ran = True
    if not ran:
        raise SystemExit("No blocks analyzed.")


if __name__ == "__main__":
    main()
