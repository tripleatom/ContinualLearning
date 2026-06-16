"""
Test whether decoded sleep reactivation events are temporally aligned to
UP/DOWN state landmarks (DOWN onset, DOWN->UP transition / UP onset,
UP-state center, UP offset).

Why a separate script
---------------------
apply_merged_decoder_to_sleep_original.py already produces, on a SHARED coarse
grid (best["bin_ms"], 50-500 ms):
    - reactivation events  : peaks of decoder P(+1)/P(-1) above event_threshold
    - UP/DOWN state labels  : from population MUA
...but it never relates the two, and on a 50-500 ms grid an UP state is only a
few bins long, so "onset vs center" is unresolvable.

This script therefore:
    1. Loads the reactivation events from each block's saved
       sleep_decoding_results.pkl (decoder grid, unchanged).
    2. RE-detects UP/DOWN states at a fine bin (params.updown_bin_ms, 10 ms)
       directly from the sleep spike pkl, so transition timing is well resolved.
    3. Maps each reactivation event time onto the fine UP/DOWN landmarks and
       builds peri-landmark time histograms (PETH).
    4. Controls the firing-rate confound (both signals derive from the same
       population MUA) with a CIRCULAR-SHIFT shuffle baseline: only timing
       structure beyond shared occupancy survives the shuffle band.

Outputs, per sleep block, under
    <session>/reactivation/reactivation_updown_alignment_<session>/<label>/ :
    - reactivation_updown_alignment_<ts>.png   (stamped, reproducible)
    - reactivation_updown_alignment_results.pkl

Run
---
    python reactivation/sleep/reactivation_updown_alignment.py
    python reactivation/sleep/reactivation_updown_alignment.py --sleep-label pre
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(code_dir / "reactivation" / "VStimOnDecoding"))
sys.path.insert(0, str(Path(__file__).parent))

from UPState import load_sleep_rates_from_pkl, detect_up_down_from_rates
from params import (
    sleep_blocks, task_pkl, random_state,
    updown_bin_ms, updown_smooth_sigma_ms, updown_down_z_threshold,
    updown_up_z_threshold, updown_down_percentile, updown_min_state_ms,
    updown_merge_gap_ms,
)


# Peri-landmark analysis window.
PETH_HALF_WINDOW_SEC = 0.5
PETH_BIN_SEC = 0.02
N_SHUFFLES = 500
NEAR_WINDOW_SEC = 0.1          # window used for the "enrichment near landmark" stat

EVENT_COLORS = {-1: "#d95f0e", 0: "0.45", 1: "#2c7fb8", "all": "black"}

# Landmark types extracted from the fine UP/DOWN detection. The key is the
# label; the value describes which UP/DOWN event field supplies the time.
# NOTE: "up_onset" IS the down->up transition in this representation - an UP
# state always begins where population MUA crosses up; we report it under both
# names so the figure answers the question as the user phrased it.
LANDMARK_TYPES = [
    ("down_onset", "DOWN onset"),
    ("up_onset",   "UP onset = DOWN->UP transition"),
    ("up_center",  "UP-state center"),
    ("up_offset",  "UP offset = UP->DOWN transition"),
]


def _extract_landmarks(updown):
    """Return a dict landmark_key -> sorted array of landmark times (sec)."""
    up = updown["events"]["up"]
    down = updown["events"]["down"]
    landmarks = {
        "down_onset": np.array([e["start_sec"] for e in down], dtype=float),
        "up_onset":   np.array([e["start_sec"] for e in up], dtype=float),
        "up_center":  np.array([e["center_sec"] for e in up], dtype=float),
        "up_offset":  np.array([e["end_sec"] for e in up], dtype=float),
    }
    for k in landmarks:
        landmarks[k] = np.sort(landmarks[k])
    return landmarks


def _nearest_signed_latency(event_times, landmarks):
    """For each event, signed latency to the NEAREST landmark (event - landmark).

    Positive => event occurs AFTER the landmark.
    """
    event_times = np.asarray(event_times, dtype=float)
    landmarks = np.asarray(landmarks, dtype=float)
    if event_times.size == 0 or landmarks.size == 0:
        return np.array([], dtype=float)

    idx = np.searchsorted(landmarks, event_times)
    idx_left = np.clip(idx - 1, 0, landmarks.size - 1)
    idx_right = np.clip(idx, 0, landmarks.size - 1)
    cand_left = event_times - landmarks[idx_left]
    cand_right = event_times - landmarks[idx_right]
    pick_left = np.abs(cand_left) <= np.abs(cand_right)
    return np.where(pick_left, cand_left, cand_right)


def _circular_shift(event_times, start_sec, end_sec, shift):
    duration = end_sec - start_sec
    return start_sec + np.mod(event_times - start_sec + shift, duration)


def _peth_counts(latencies, edges):
    counts, _ = np.histogram(latencies, bins=edges)
    return counts.astype(float)


def _shuffle_band(event_times, landmarks, start_sec, end_sec, edges, rng):
    """Circular-shift null: returns (mean, lo, hi) PETH bands across shuffles."""
    duration = end_sec - start_sec
    n_bins = len(edges) - 1
    acc = np.zeros((N_SHUFFLES, n_bins), dtype=float)
    for s in range(N_SHUFFLES):
        shift = rng.uniform(0.0, duration)
        shifted = _circular_shift(event_times, start_sec, end_sec, shift)
        lat = _nearest_signed_latency(shifted, landmarks)
        acc[s] = _peth_counts(lat, edges)
    mean = acc.mean(axis=0)
    lo = np.percentile(acc, 2.5, axis=0)
    hi = np.percentile(acc, 97.5, axis=0)
    return mean, lo, hi


def _state_label_at(times, centers, state_label):
    """Nearest-bin UP/DOWN/uncertain label for each event time."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(centers, times)
    idx_left = np.clip(idx - 1, 0, centers.size - 1)
    idx_right = np.clip(idx, 0, centers.size - 1)
    pick_left = np.abs(times - centers[idx_left]) <= np.abs(times - centers[idx_right])
    nearest = np.where(pick_left, idx_left, idx_right)
    return state_label[nearest]


def _up_phase_positions(event_times, up_events):
    """Normalized 0->1 position within the containing UP state (onset->offset)."""
    positions = []
    starts = np.array([e["start_sec"] for e in up_events], dtype=float)
    ends = np.array([e["end_sec"] for e in up_events], dtype=float)
    for t in np.asarray(event_times, dtype=float):
        inside = np.where((starts <= t) & (t <= ends))[0]
        if inside.size == 0:
            continue
        i = inside[0]
        span = ends[i] - starts[i]
        if span <= 0:
            continue
        positions.append((t - starts[i]) / span)
    return np.array(positions, dtype=float)


def _enrichment_stat(observed_counts, shuffle_mean, edges, near_window):
    """Fraction of events within +/-near_window of the landmark, observed vs null."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    near = np.abs(centers) <= near_window
    obs_total = observed_counts.sum()
    null_total = shuffle_mean.sum()
    obs_frac = observed_counts[near].sum() / obs_total if obs_total > 0 else 0.0
    null_frac = shuffle_mean[near].sum() / null_total if null_total > 0 else 0.0
    enrichment = obs_frac / null_frac if null_frac > 0 else float("nan")
    return obs_frac, null_frac, enrichment


def _gather_event_times(decoder_results):
    """Pull reactivation event times (sec) for +1, -1, and combined from the
    saved decoder results dict."""
    events = decoder_results["events"]
    pos_t = np.array([e["time_sec"] for e in events.get(1, [])], dtype=float)
    neg_t = np.array([e["time_sec"] for e in events.get(-1, [])], dtype=float)
    all_t = np.sort(np.concatenate([pos_t, neg_t])) if (pos_t.size or neg_t.size) else np.array([])
    return {1: np.sort(pos_t), -1: np.sort(neg_t), "all": all_t}


def analyze_block(label, decoder_pkl, out_dir):
    with open(decoder_pkl, "rb") as f:
        dec = pickle.load(f)

    sleep_pkl = dec["sleep_pkl"]
    start_sec = float(dec["sleep_start_sec"])
    end_sec = float(dec["sleep_end_sec"])
    decoder_bin_ms = float(dec["best"]["bin_ms"])

    # Fine-grid UP/DOWN re-detection from the raw sleep spikes.
    fine_bin_sec = updown_bin_ms / 1000.0
    X_sleep, centers_fine, units, s_eff, e_eff = load_sleep_rates_from_pkl(
        sleep_pkl, start_sec=start_sec, end_sec=end_sec, bin_size_sec=fine_bin_sec
    )
    updown = detect_up_down_from_rates(
        X_sleep,
        centers_fine,
        fine_bin_sec,
        smooth_sigma_sec=updown_smooth_sigma_ms / 1000.0,
        down_z_threshold=updown_down_z_threshold,
        up_z_threshold=updown_up_z_threshold,
        down_percentile=updown_down_percentile,
        min_state_duration_sec=updown_min_state_ms / 1000.0,
        merge_gap_sec=updown_merge_gap_ms / 1000.0,
    )
    landmarks = _extract_landmarks(updown)
    state_label = np.asarray(updown["state_label"], dtype=int)

    event_times = _gather_event_times(dec)
    edges = np.arange(-PETH_HALF_WINDOW_SEC, PETH_HALF_WINDOW_SEC + PETH_BIN_SEC, PETH_BIN_SEC)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(random_state)

    # ---- Peri-landmark PETHs + shuffle bands, for combined reactivation events.
    peth = {}
    enrichment = {}
    for key, _name in LANDMARK_TYPES:
        lm = landmarks[key]
        obs_lat = _nearest_signed_latency(event_times["all"], lm)
        obs_counts = _peth_counts(obs_lat, edges)
        sh_mean, sh_lo, sh_hi = _shuffle_band(
            event_times["all"], lm, s_eff, e_eff, edges, rng
        )
        peth[key] = {
            "observed": obs_counts,
            "shuffle_mean": sh_mean,
            "shuffle_lo": sh_lo,
            "shuffle_hi": sh_hi,
            "pos_latency": _nearest_signed_latency(event_times[1], lm),
            "neg_latency": _nearest_signed_latency(event_times[-1], lm),
        }
        obs_frac, null_frac, enr = _enrichment_stat(obs_counts, sh_mean, edges, NEAR_WINDOW_SEC)
        enrichment[key] = {"obs_frac": obs_frac, "null_frac": null_frac, "enrichment": enr}

    # ---- UP-phase position (onset=0 -> offset=1) with shuffle band.
    up_pos_obs = _up_phase_positions(event_times["all"], updown["events"]["up"])
    phase_edges = np.linspace(0.0, 1.0, 11)
    phase_obs = _peth_counts(up_pos_obs, phase_edges)
    phase_sh = np.zeros((N_SHUFFLES, len(phase_edges) - 1), dtype=float)
    duration = e_eff - s_eff
    for s in range(N_SHUFFLES):
        shifted = _circular_shift(event_times["all"], s_eff, e_eff, rng.uniform(0, duration))
        phase_sh[s] = _peth_counts(
            _up_phase_positions(shifted, updown["events"]["up"]), phase_edges
        )
    phase_band = (phase_sh.mean(0), np.percentile(phase_sh, 2.5, 0), np.percentile(phase_sh, 97.5, 0))

    # ---- State occupancy of events vs overall time occupancy.
    ev_states = _state_label_at(event_times["all"], centers_fine, state_label)
    n_ev = max(ev_states.size, 1)
    n_bins_total = max(state_label.size, 1)
    occupancy = {
        "event_up": float(np.mean(ev_states == 1)),
        "event_down": float(np.mean(ev_states == 0)),
        "event_uncertain": float(np.mean(ev_states == -1)),
        "time_up": float(np.mean(state_label == 1)),
        "time_down": float(np.mean(state_label == 0)),
        "time_uncertain": float(np.mean(state_label == -1)),
        "n_events": int(ev_states.size),
    }

    results = {
        "label": label,
        "sleep_pkl": sleep_pkl,
        "decoder_bin_ms": decoder_bin_ms,
        "updown_bin_ms": updown_bin_ms,
        "n_events": {k: int(np.asarray(v).size) for k, v in event_times.items()},
        "peth_edges": edges,
        "peth_bin_centers": bin_centers,
        "peth": peth,
        "enrichment_near_window_sec": NEAR_WINDOW_SEC,
        "enrichment": enrichment,
        "phase_edges": phase_edges,
        "phase_observed": phase_obs,
        "phase_shuffle": phase_band,
        "occupancy": occupancy,
        "n_shuffles": N_SHUFFLES,
    }

    _plot_alignment(out_dir, label, results)
    _print_block_report(results)

    pkl_out = out_dir / "reactivation_updown_alignment_results.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved -> {pkl_out}")
    return results


def _repro_stamp(label, results, timestamp):
    return (
        f"Generated {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
        f"script={Path(__file__).name} | block={label}\n"
        f"reactivation events from decoder @ {results['decoder_bin_ms']} ms | "
        f"UP/DOWN re-detected @ {results['updown_bin_ms']} ms (smooth={updown_smooth_sigma_ms}ms, "
        f"down_z={updown_down_z_threshold}, up_z={updown_up_z_threshold}, "
        f"down_pct={updown_down_percentile}, min_state={updown_min_state_ms}ms, merge_gap={updown_merge_gap_ms}ms)\n"
        f"sleep_pkl={Path(results['sleep_pkl']).name} | "
        f"n_events: +1={results['n_events'][1]}, -1={results['n_events'][-1]}, all={results['n_events']['all']} | "
        f"PETH +/-{PETH_HALF_WINDOW_SEC}s @ {PETH_BIN_SEC*1000:.0f}ms bins | "
        f"shuffle=circular-shift n={results['n_shuffles']}, random_state={random_state}\n"
        f"latency convention: positive = reactivation AFTER landmark | "
        f"enrichment window=+/-{NEAR_WINDOW_SEC}s"
    )


def _plot_alignment(out_dir, label, results):
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = results["peth_edges"]
    bc = results["peth_bin_centers"]
    width = (edges[1] - edges[0]) * 0.9

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, (key, name) in zip(axes[:4], LANDMARK_TYPES):
        p = results["peth"][key]
        ax.bar(bc, p["observed"], width=width, color="0.7", edgecolor="none",
               label="observed (all events)")
        ax.plot(bc, p["shuffle_mean"], color="crimson", linewidth=1.2, label="shuffle mean")
        ax.fill_between(bc, p["shuffle_lo"], p["shuffle_hi"], color="crimson", alpha=0.15,
                        label="shuffle 95% band")
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        enr = results["enrichment"][key]["enrichment"]
        ax.set_title(f"{name}\nnear-landmark enrichment = {enr:.2f}x", fontsize=9)
        ax.set_xlabel("Reactivation time - landmark (s)")
        ax.set_ylabel("Event count")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)

    # UP-phase position panel.
    ax = axes[4]
    phase_edges = results["phase_edges"]
    pc = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    pw = (phase_edges[1] - phase_edges[0]) * 0.9
    sh_mean, sh_lo, sh_hi = results["phase_shuffle"]
    ax.bar(pc, results["phase_observed"], width=pw, color="0.7", edgecolor="none", label="observed")
    ax.plot(pc, sh_mean, color="crimson", linewidth=1.2, label="shuffle mean")
    ax.fill_between(pc, sh_lo, sh_hi, color="crimson", alpha=0.15)
    ax.set_title("Within-UP position of reactivations\n(0=onset, 1=offset)", fontsize=9)
    ax.set_xlabel("Normalized UP-state phase")
    ax.set_ylabel("Event count")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Occupancy panel.
    ax = axes[5]
    occ = results["occupancy"]
    cats = ["UP", "DOWN", "uncertain"]
    ev = [occ["event_up"], occ["event_down"], occ["event_uncertain"]]
    tm = [occ["time_up"], occ["time_down"], occ["time_uncertain"]]
    x = np.arange(len(cats))
    ax.bar(x - 0.2, ev, width=0.4, color="#2c7fb8", label="event fraction")
    ax.bar(x + 0.2, tm, width=0.4, color="0.6", label="time occupancy")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Fraction")
    ax.set_title("State occupancy: events vs time\n(equal bars = no preference)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Reactivation vs UP/DOWN alignment - block '{label}' "
        f"(n={results['n_events']['all']} events; UP/DOWN @ {results['updown_bin_ms']} ms)",
        fontsize=13,
    )

    timestamp = datetime.now().astimezone()
    fig.text(0.01, 0.005, _repro_stamp(label, results, timestamp), ha="left", va="bottom", fontsize=7)
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    fig_path = out_dir / f"reactivation_updown_alignment_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Alignment figure saved -> {fig_path}")
    return fig_path


def _print_block_report(results):
    print(f"\n=== Reactivation/UP-DOWN alignment: block '{results['label']}' ===")
    print(f"Reactivation events: +1={results['n_events'][1]}, "
          f"-1={results['n_events'][-1]}, all={results['n_events']['all']}")
    occ = results["occupancy"]
    print(f"Event state occupancy : UP={occ['event_up']:.2%}, DOWN={occ['event_down']:.2%}, "
          f"uncertain={occ['event_uncertain']:.2%}")
    print(f"Time  state occupancy : UP={occ['time_up']:.2%}, DOWN={occ['time_down']:.2%}, "
          f"uncertain={occ['time_uncertain']:.2%}")
    print(f"\nNear-landmark enrichment (events within +/-{NEAR_WINDOW_SEC}s, observed/shuffle):")
    best_key, best_enr = None, -np.inf
    for key, name in LANDMARK_TYPES:
        e = results["enrichment"][key]
        flag = ""
        if np.isfinite(e["enrichment"]):
            if e["enrichment"] > best_enr:
                best_enr, best_key = e["enrichment"], name
        print(f"  {name:<35s}: obs={e['obs_frac']:.2%}, null={e['null_frac']:.2%}, "
              f"enrichment={e['enrichment']:.2f}x{flag}")
    if best_key is not None:
        print(f"\n-> Reactivations are most enriched near: {best_key} ({best_enr:.2f}x shuffle)")
    print("   (enrichment ~1.0x => no alignment beyond shared firing-rate occupancy)")


def main():
    parser = argparse.ArgumentParser(
        description="Align decoded reactivation events to fine-grid UP/DOWN landmarks."
    )
    parser.add_argument("--sleep-label", default=None, help="Only run this block, e.g. pre or post.")
    parser.add_argument("--results-dir", default=None,
                        help="Override directory holding <label>/sleep_decoding_results.pkl.")
    args = parser.parse_args()

    session = Path(task_pkl).parent.name
    decoder_base = (Path(args.results_dir) if args.results_dir
                    else Path(task_pkl).parent / "reactivation" / f"sleep_merged_decoder_{session}")
    out_base = Path(task_pkl).parent / "reactivation" / f"reactivation_updown_alignment_{session}"

    ran_any = False
    for label, pkl_path, _start, _end in sleep_blocks:
        if args.sleep_label is not None and label != args.sleep_label:
            continue
        if not pkl_path:
            print(f"\n[{label}] Skipped - no sleep_pkl path.")
            continue
        decoder_pkl = decoder_base / label / "sleep_decoding_results.pkl"
        if not decoder_pkl.exists():
            print(f"\n[{label}] Skipped - decoder results not found: {decoder_pkl}\n"
                  f"          Run apply_merged_decoder_to_sleep_original.py first.")
            continue
        print(f"\n========== Aligning block '{label}' ==========")
        analyze_block(label, decoder_pkl, out_base / label)
        ran_any = True

    if not ran_any:
        raise SystemExit("No blocks analyzed. Check sleep-label / that the decoder has been run.")


if __name__ == "__main__":
    main()
