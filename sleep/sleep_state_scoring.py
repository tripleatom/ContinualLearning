"""
sleep_state_scoring.py
======================
Step 3 of the UP/DOWN state pipeline.

Classifies each 4-second epoch into NREM / REM / Wake using three signals:
    1. PC1 of the LFP spectrogram  (high PC1 → NREM: dominated by slow waves)
    2. Theta ratio (5–10 / 2–15 Hz) (high + low PC1 → REM)
    3. EMG proxy (high-frequency LFP coherence 200–600 Hz) (high → Awake)

Decision logic (applied per epoch, after smoothing):
    ├── emg_proxy  > wake_thresh   → WAKE
    ├── theta_ratio > rem_thresh   → REM
    └── pc1        > nrem_thresh   → NREM
    otherwise                      → UNSCORED

A minimum-bout filter is applied so that brief (<30 s) isolated epochs are
re-labelled as the surrounding state.

Output (per window):
    <OUTPUT_DIR>/<window_name>/<SESSION>_sh<SHANK>_<window_name>_sleep_states.pkl

Pickle keys:
    state_labels    (n_epochs,) int8  {0: UNSCORED, 1: NREM, 2: REM, 3: WAKE}
    nrem_mask_lfp   (n_lfp_samples,) bool  — aligned to LFP time axis
    epoch_times     (n_epochs,)       seconds from window start (epoch centres)
    epoch_sec       float
    pc1_mean        (n_epochs,)       mean PC1 across deep channels per epoch
    theta_ratio_mean (n_epochs,)
    emg_proxy_mean  (n_epochs,)
    thresholds      dict   {signal: threshold_value}
    state_summary   dict   {state_name: fraction_of_recording}
    session, shank, window_name
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sleep_config import (
    SESSION, SHANK, SCORING, DEEP_LAYER_RANGE_UM,
    SLEEP_WINDOWS, get_window_dir, validate_windows,
)

validate_windows()

# ─── state integer codes ──────────────────────────────────────────────────────
UNSCORED = 0
NREM     = 1
REM      = 2
WAKE     = 3
STATE_NAMES  = {UNSCORED: "UNSCORED", NREM: "NREM", REM: "REM", WAKE: "WAKE"}
STATE_COLORS = {UNSCORED: "lightgrey", NREM: "steelblue", REM: "tomato", WAKE: "gold"}


def epoch_mean(signal: np.ndarray, epoch_samples: int) -> np.ndarray:
    """Average a 1D signal into non-overlapping epochs."""
    n = len(signal) // epoch_samples * epoch_samples
    return signal[:n].reshape(-1, epoch_samples).mean(axis=1)


def apply_min_bout(labels: np.ndarray, min_epochs: int) -> np.ndarray:
    """
    Replace state bouts shorter than min_epochs with the label of the
    surrounding majority state.  Applied iteratively until stable.
    """
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(labels):
            j = i
            while j < len(labels) and labels[j] == labels[i]:
                j += 1
            bout_len = j - i
            if bout_len < min_epochs:
                # replace with neighbor label (prefer right, fall back to left)
                if j < len(labels):
                    new_label = labels[j]
                elif i > 0:
                    new_label = labels[i - 1]
                else:
                    new_label = UNSCORED
                if new_label != labels[i]:
                    labels[i:j] = new_label
                    changed = True
            i = j
    return labels


# ─── main loop ────────────────────────────────────────────────────────────────
for win in SLEEP_WINDOWS:
    wname   = win["name"]
    out_dir = get_window_dir(wname)

    # ── load band powers ──────────────────────────────────────────────────────
    bp_pkl = out_dir / f"{SESSION}_sh{SHANK}_{wname}_band_powers.pkl"
    if not bp_pkl.exists():
        print(f"\n[SKIP] Band powers not found: {bp_pkl}")
        print(f"       Run calculate_band_powers_sleep.py first.")
        continue

    print(f"\n{'='*70}")
    print(f"Window: {wname}")
    print(f"Loading band powers from: {bp_pkl.name}")

    with open(bp_pkl, "rb") as fh:
        bp_data = pickle.load(fh)

    lfp_time    = bp_data["lfp_time"]
    fs          = bp_data["sampling_rate"]
    ch_depths   = bp_data["channel_depths"]
    channel_ids = bp_data["channel_ids"]
    pc1_interp  = bp_data["pc1_interp"]       # (n_ch, n_samples)
    band_powers = bp_data["band_powers"]
    n_samples   = len(lfp_time)

    # ── select deep-layer channels for scoring ────────────────────────────────
    deep_mask = (ch_depths >= DEEP_LAYER_RANGE_UM[0]) & \
                (ch_depths <= DEEP_LAYER_RANGE_UM[1])
    deep_indices = np.where(deep_mask)[0]
    n_deep = len(deep_indices)
    print(f"  Deep-layer channels : {n_deep}  "
          f"({ch_depths[deep_indices[0]]:.0f}–{ch_depths[deep_indices[-1]]:.0f} µm)")

    # ── aggregate signals across deep channels ────────────────────────────────
    # PC1 mean (slow-wave index)
    pc1_mean_full = pc1_interp[deep_indices].mean(axis=0)   # (n_samples,)

    # Theta ratio mean
    theta_stack = np.stack(
        [band_powers[str(channel_ids[i])]["theta_ratio"] for i in deep_indices],
        axis=0,
    )
    theta_mean_full = theta_stack.mean(axis=0)

    # EMG proxy (high-frequency incoherence across ALL channels — captures muscle)
    emg_stack = np.stack(
        [band_powers[str(ch_id)]["emg_proxy"] for ch_id in channel_ids],
        axis=0,
    )
    emg_mean_full = emg_stack.mean(axis=0)

    # ── epoch the signals ──────────────────────────────────────────────────────
    epoch_sec     = SCORING["epoch_sec"]
    epoch_samples = int(epoch_sec * fs)
    n_epochs      = n_samples // epoch_samples

    pc1_epochs    = epoch_mean(pc1_mean_full,   epoch_samples)[:n_epochs]
    theta_epochs  = epoch_mean(theta_mean_full, epoch_samples)[:n_epochs]
    emg_epochs    = epoch_mean(emg_mean_full,   epoch_samples)[:n_epochs]
    epoch_times   = (np.arange(n_epochs) + 0.5) * epoch_sec   # epoch centres

    print(f"  Epoch length: {epoch_sec} s  |  n_epochs: {n_epochs}")

    # ── compute thresholds ────────────────────────────────────────────────────
    pc1_thresh  = pc1_epochs.mean()  + SCORING["nrem_pc1_thresh_sd"]  * pc1_epochs.std()
    rem_thresh  = theta_epochs.mean() + SCORING["rem_theta_thresh_sd"] * theta_epochs.std()
    emg_thresh  = emg_epochs.mean()  + SCORING["wake_emg_thresh_sd"]  * emg_epochs.std()

    thresholds = {
        "pc1_nrem"   : float(pc1_thresh),
        "theta_rem"  : float(rem_thresh),
        "emg_wake"   : float(emg_thresh),
    }
    print(f"  Thresholds  : PC1>{pc1_thresh:.3f} → NREM | "
          f"theta>{rem_thresh:.3f} → REM | "
          f"EMG>{emg_thresh:.3f} → WAKE")

    # ── classify epochs ───────────────────────────────────────────────────────
    labels = np.full(n_epochs, UNSCORED, dtype=np.int8)

    # Priority order: WAKE > REM > NREM
    labels[pc1_epochs  > pc1_thresh]  = NREM
    labels[theta_epochs > rem_thresh] = REM
    labels[emg_epochs  > emg_thresh]  = WAKE

    # Apply minimum-bout filter
    min_epochs = max(1, int(SCORING["min_bout_sec"] / epoch_sec))
    labels = apply_min_bout(labels, min_epochs)

    # ── state summary ─────────────────────────────────────────────────────────
    state_summary = {}
    for code, sname in STATE_NAMES.items():
        frac = np.mean(labels == code)
        state_summary[sname] = float(frac)
        print(f"  {sname:9s}: {frac*100:5.1f}%  "
              f"({int(frac*n_epochs*epoch_sec/60):.0f} min)")

    # ── build NREM mask on full LFP time axis ─────────────────────────────────
    nrem_mask_lfp = np.zeros(n_samples, dtype=bool)
    for ep in range(n_epochs):
        if labels[ep] == NREM:
            s = ep * epoch_samples
            e = min(s + epoch_samples, n_samples)
            nrem_mask_lfp[s:e] = True

    nrem_frac = nrem_mask_lfp.mean()
    print(f"  NREM mask coverage: {nrem_frac*100:.1f}% of window")

    # ── save ──────────────────────────────────────────────────────────────────
    out_name = f"{SESSION}_sh{SHANK}_{wname}_sleep_states.pkl"
    out_path = out_dir / out_name

    save_data = {
        "state_labels"      : labels,
        "nrem_mask_lfp"     : nrem_mask_lfp,
        "epoch_times"       : epoch_times,
        "epoch_sec"         : epoch_sec,
        "pc1_mean"          : pc1_epochs,
        "theta_ratio_mean"  : theta_epochs,
        "emg_proxy_mean"    : emg_epochs,
        "thresholds"        : thresholds,
        "state_summary"     : state_summary,
        "session"           : SESSION,
        "shank"             : SHANK,
        "window_name"       : wname,
    }

    print(f"Saving → {out_path}")
    with open(out_path, "wb") as fh:
        pickle.dump(save_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ── diagnostic plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"{SESSION}  sh{SHANK}  {wname}  —  Sleep State Scoring",
                 fontsize=13, fontweight="bold")

    # Shade state background on all panels
    def shade_states(ax):
        for ep in range(n_epochs):
            t0 = epoch_times[ep] - epoch_sec / 2
            t1 = epoch_times[ep] + epoch_sec / 2
            ax.axvspan(t0, t1,
                       color=STATE_COLORS[labels[ep]],
                       alpha=0.3, linewidth=0)

    for ax in axes:
        shade_states(ax)

    axes[0].plot(epoch_times, pc1_epochs, color="steelblue", lw=0.8)
    axes[0].axhline(pc1_thresh, color="steelblue", ls="--", lw=1, alpha=0.7,
                    label=f"NREM thresh ({pc1_thresh:.2f})")
    axes[0].set_ylabel("Spec PC1")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(epoch_times, theta_epochs, color="tomato", lw=0.8)
    axes[1].axhline(rem_thresh, color="tomato", ls="--", lw=1, alpha=0.7,
                    label=f"REM thresh ({rem_thresh:.3f})")
    axes[1].set_ylabel("Theta ratio")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(epoch_times, emg_epochs, color="goldenrod", lw=0.8)
    axes[2].axhline(emg_thresh, color="goldenrod", ls="--", lw=1, alpha=0.7,
                    label=f"Wake thresh ({emg_thresh:.3f})")
    axes[2].set_ylabel("EMG proxy")
    axes[2].legend(fontsize=8, loc="upper right")

    # Hypnogram
    from matplotlib.patches import Patch
    hyp_y = {NREM: 1, REM: 2, WAKE: 3, UNSCORED: 0}
    axes[3].step(epoch_times, [hyp_y[l] for l in labels],
                 where="mid", color="black", lw=1)
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(["UNSC.", "NREM", "REM", "WAKE"])
    axes[3].set_ylabel("State")
    axes[3].set_xlabel("Time (s from window start)")
    legend_patches = [Patch(facecolor=STATE_COLORS[c], label=STATE_NAMES[c])
                      for c in [NREM, REM, WAKE, UNSCORED]]
    axes[3].legend(handles=legend_patches, loc="upper right",
                   fontsize=8, ncol=4)

    plt.tight_layout()
    fig_path = out_dir / f"{SESSION}_sh{SHANK}_{wname}_hypnogram.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Diagnostic plot → {fig_path.name}")

print(f"\n{'='*70}")
print("sleep_state_scoring.py  DONE")
print("NOTE: Inspect the hypnogram PNG and adjust threshold SD values in")
print("      sleep_config.SCORING if the state boundaries look off.")
print(f"{'='*70}")
