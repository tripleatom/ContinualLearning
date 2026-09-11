"""
Per-channel PC1 / theta-ratio / velocity histograms, with a GMM-BIC test for
whether each distribution is bimodal (e.g. NREM vs wake) or unimodal.

Shank 0 only (this is a diagnostic/exploratory script, not a pipeline stage -
override with --shank if you want another one). For every active sleep
session (pre/post per sleep_pipeline_config.SESSION_FILTER) and every channel
on that shank, reads PC1 and theta_ratio from the band-powers pkl
(compute_sleep_features.py's output - already computed, nothing recomputed
here) and the session's velocity trace (proc_func_velocity.py's output),
then fits 1- vs 2-component Gaussian mixtures to each of the three
distributions: 2 components wins (lower BIC) -> bimodal, else unimodal. This
is the same idea as fitting a GMM to the broadbandSlowWave histogram to
separate NREM from wake (Watson et al. 2016), applied here as a diagnostic
across PC1, theta_ratio and velocity independently.

Two things about the velocity panel specifically:

  * It is built from the speed reduced to one point per second - each point the
    mean over its own 1 s, so consecutive points share no frames - NOT from the
    raw camera frames. The camera's frame rate is not fixed, so a per-frame
    histogram silently weights densely-sampled stretches more heavily. Pass
    --velocity-window-sec 10 to average over the NREM gate's window instead;
    that makes the threshold directly comparable to --velocity-thresh, at the
    cost of 90% overlap between neighbouring points.
  * It is fitted in log10. Speed is strictly positive and spans three orders of
    magnitude, so one behavioural state is log-normal; forcing Gaussians onto
    it in linear units produced components with 13-24% of their mass below zero
    speed, curves that missed the histogram entirely, and a "bimodal" verdict
    that reflected skewness rather than two states.

Where a distribution comes out bimodal the figure also carries the threshold
the fit implies - the point where the two weighted components cross, i.e. where
a sample stops being more likely to belong to the low mode than the high one
(`gmm_threshold`). It is drawn as a dashed line, annotated with the value (in
the original units) and the fraction of samples below it, and written to the
summary CSV. Note it is neither the midpoint between the means nor the
histogram trough: both ignore the components' widths and weights.

Values are used as-is - no artifact-period exclusion (see
sleep_pipeline_config.artifact_params) - so movement/cable artifacts can
thicken a distribution's tails and bias the bimodality call. The threshold is a
starting point read off one session's marginal distribution, not a scoring
decision.

Outputs (in low_freq/histograms/):
  {session}_sh{shank}_ch{NNN}_pc1_theta_velocity_hist.png
  {session}_sh{shank}_bimodality_summary.csv   (one row per channel; the
    velocity columns are session-level, so they repeat identically across
    every row of a given session. velocity_threshold_px_s is in px/s, ready to
    pass to score_nrem_delta_velocity.py --velocity-thresh)
"""
import argparse
import csv
import errno
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from sleep_pipeline_config import (
    rec_folder, session_name, video_folder, VELOCITY_SOURCE,
    VELOCITY_MIN_COVERAGE, VELOCITY_STEP_SEC, VELOCITY_WINDOW_SEC,
    sleep_sessions, active_sleep_sessions,
    resolve_existing_file, resolve_output_folder, mirror_on_backup_server,
)
from proc_func_velocity import aggregate_speed, velocity_output_name

DEFAULT_SHANK = 0


def parse_args():
    p = argparse.ArgumentParser(
        description="PC1 / theta-ratio / velocity histograms + GMM bimodality test.")
    p.add_argument("--shank", type=int, default=DEFAULT_SHANK,
                   help=f"shank to analyze (default {DEFAULT_SHANK})")
    # Window == step, so each point averages its own second and consecutive
    # points share no frames. A longer window on the same grid would overlap
    # (the NREM gate's 10 s window on a 1 s step reuses 90% of each sample),
    # which narrows the modes and makes the histogram's effective n far smaller
    # than its nominal one.
    p.add_argument("--velocity-window-sec", type=float, default=VELOCITY_STEP_SEC,
                   help="window the speed is averaged over before its histogram "
                        f"is built (default {VELOCITY_STEP_SEC:g}s = one grid step, "
                        "so the points are independent). Pass "
                        f"{VELOCITY_WINDOW_SEC:g} to match the NREM gate's window "
                        "instead.")
    p.add_argument("--velocity-step-sec", type=float, default=VELOCITY_STEP_SEC,
                   help="spacing of the regular grid the speed is placed on "
                        f"(default {VELOCITY_STEP_SEC:g}s)")
    return p.parse_args()


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def with_backup_mirrors(paths):
    """Interleave each path with its new-server mirror, preserving priority."""
    out = []
    for path in paths:
        mirrored = mirror_on_backup_server(path)
        if mirrored is not None:
            out.append(mirrored)
        out.append(path)
    return out


def load_velocity(session_cfg, window_sec=VELOCITY_STEP_SEC,
                  step_sec=VELOCITY_STEP_SEC):
    """Velocity for this session, on the analysis grid rather than per frame.

    Not synced to spectrogram time - this script only wants each signal's own
    marginal distribution, not a time-aligned comparison, so the camera clock
    is enough.

    It IS reduced onto a regular `step_sec` grid first, though, for two
    reasons. The camera does not run at a fixed frame rate, so a histogram of
    the raw per-frame array weights each frame equally regardless of how much
    time it stands for - stretches where frames happen to arrive closer
    together are over-represented, and what comes out is a per-frame
    distribution rather than a per-second-of-recording one. And the per-frame
    distribution is not the one anything downstream thresholds: the NREM gate
    in score_nrem_delta_velocity.py works on a `window_sec` windowed mean,
    whose spread is far narrower, so a threshold read off the raw frames could
    not be carried across. Aggregating here with the same window/step the gate
    uses makes the number on the figure the number you can actually pass to
    ``--velocity-thresh``.
    """
    proc_file = session_cfg.get("proc_file")
    if not proc_file:
        print("  No proc_file registered for this session - skipping velocity.")
        return None
    vel_name = velocity_output_name(proc_file, VELOCITY_SOURCE)
    velocity_file = first_existing(with_backup_mirrors(
        [Path(proc_file).parent / vel_name, video_folder / vel_name]))
    if velocity_file is None:
        print(f"  WARNING: velocity file not found: {vel_name} "
             f"(run proc_func_velocity.py --source {VELOCITY_SOURCE} first)")
        return None
    with open(velocity_file, "rb") as f:
        data = pickle.load(f)

    time_stamp = np.asarray(data["time_stamp"], dtype=float)
    per_frame = np.asarray(data["velocity"], dtype=float)
    grid = np.arange(time_stamp[0], time_stamp[-1], float(step_sec))
    windowed = aggregate_speed(time_stamp, per_frame, grid, window_sec,
                               min_coverage=VELOCITY_MIN_COVERAGE)
    values = windowed["mean"]
    frame_rate = 1.0 / float(np.median(np.diff(time_stamp)))
    print(f"  Loaded velocity: {velocity_file.name} "
          f"({per_frame.size:,} frames at {frame_rate:.1f} fps) -> "
          f"{values.size:,} points on a {step_sec:g}s grid "
          f"({window_sec:g}s window, {int(np.sum(~np.isfinite(values))):,} below "
          f"{VELOCITY_MIN_COVERAGE:.0%} coverage)")
    return values


def gmm_bimodality(values, label, log_space=False):
    """Fit 1- vs 2-component Gaussian mixtures; bimodal iff the 2-component
    model has the lower BIC (Bayesian info criterion - penalizes the extra
    component's parameters, so it doesn't win from overfitting alone).

    `log_space` fits log10(values) instead, for strictly positive, heavy-tailed
    quantities like speed - see the comment below for why that is not optional
    there.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if log_space:
        # Speed is strictly positive and spans three orders of magnitude, so a
        # single behavioural state is log-normal, not normal. Fitting Gaussians
        # to it in linear units is mis-specified badly enough to be misleading:
        # on this data the fitted components put 13-24% of their mass BELOW
        # ZERO speed, the curves miss the histogram, and 2 components still
        # beat 1 on BIC - not because there are two states but because two
        # Gaussians approximate a skewed shape better than one. In log10 the
        # modes are genuinely Gaussian and the split means something.
        x = np.log10(x[x > 0])
    result = {"label": label, "n": x.size, "x": x, "bic1": np.nan, "bic2": np.nan,
             "bimodal": False, "means": None, "stds": None, "weights": None,
             "threshold": None, "threshold_linear": None, "frac_below": np.nan,
             "log_space": bool(log_space)}
    if x.size < 10:
        return result
    X = x.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=3).fit(X)
    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(X)
    result["bic1"], result["bic2"] = gmm1.bic(X), gmm2.bic(X)
    result["bimodal"] = result["bic2"] < result["bic1"]
    if result["bimodal"]:
        order = np.argsort(gmm2.means_.ravel())
        result["means"] = gmm2.means_.ravel()[order]
        result["stds"] = np.sqrt(gmm2.covariances_.ravel())[order]
        result["weights"] = gmm2.weights_[order]
        # Only a two-component fit has a boundary to quote; a unimodal one is
        # deliberately left without a threshold rather than split down the middle.
        result["threshold"] = gmm_threshold(
            result["means"], result["stds"], result["weights"])
        if result["threshold"] is not None:
            result["frac_below"] = float(np.mean(x < result["threshold"]))
            # Reported back in the original units, which is what a threshold
            # is worth quoting in and what --velocity-thresh expects.
            result["threshold_linear"] = (10.0 ** result["threshold"] if log_space
                                          else result["threshold"])
    else:
        result["means"] = gmm1.means_.ravel()
        result["stds"] = np.sqrt(gmm1.covariances_.ravel())
        result["weights"] = gmm1.weights_
    return result


def norm_pdf(x, mean, std):
    std = max(float(std), 1e-12)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def gmm_threshold(means, stds, weights):
    """Where the two fitted components cross - the posterior-0.5 boundary.

    This is the threshold the fit itself implies: below it a sample is more
    likely to have come from the low component, above it from the high one. It
    is NOT the midpoint between the means and NOT the histogram trough - both
    ignore the components' widths and weights, and the two modes here are
    usually very unequal in both (a narrow, heavy still mode against a broad,
    lighter moving one), which pushes the true boundary well toward the narrow
    side.

    Setting w1*N(x;m1,s1) = w2*N(x;m2,s2) and taking logs leaves a quadratic in
    x. When the variances differ it has two roots; only the one lying between
    the means separates the modes, the other sits far out in a tail and means
    nothing. Returns None when no single root separates them - which happens
    when one component merely sits inside the other, and where quoting a
    threshold would be inventing a boundary the fit does not support.
    """
    if means is None or len(means) != 2:
        return None
    (m1, m2), (s1, s2), (w1, w2) = means, stds, weights
    m1, m2 = float(m1), float(m2)
    s1, s2 = max(float(s1), 1e-12), max(float(s2), 1e-12)
    w1, w2 = max(float(w1), 1e-300), max(float(w2), 1e-300)
    if not m2 > m1:
        return None

    a = 1.0 / (2 * s1 ** 2) - 1.0 / (2 * s2 ** 2)
    b = m2 / s2 ** 2 - m1 / s1 ** 2
    c = (m1 ** 2 / (2 * s1 ** 2) - m2 ** 2 / (2 * s2 ** 2)
         + np.log((w2 * s1) / (w1 * s2)))

    if abs(a) < 1e-12:                      # equal variances -> linear
        if abs(b) < 1e-300:
            return None
        roots = [-c / b]
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        root = np.sqrt(discriminant)
        roots = [(-b + root) / (2 * a), (-b - root) / (2 * a)]

    separating = [r for r in roots if m1 < r < m2]
    return float(separating[0]) if len(separating) == 1 else None


def plot_panel(ax, result, xlabel):
    """Histogram + fitted components, with the implied threshold when bimodal.

    A log-space fit is plotted in log space - the histogram, the curves and the
    threshold line all live on the same axis the mixture was fitted on, so what
    is drawn is what was tested. Only the threshold's printed value is
    converted back to the original units.
    """
    x = result["x"]
    if x.size < 10:
        ax.set_title(f"{result['label']}: too few samples (n={result['n']})", fontsize=9)
        ax.axis("off")
        return

    ax.hist(x, bins=60, density=True, color="#4C72B0", alpha=0.6, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 500)
    for mean, std, weight in zip(result["means"], result["stds"], result["weights"]):
        ax.plot(xs, weight * norm_pdf(xs, mean, std), lw=1.5, color="#C44E52")

    # The implied threshold, drawn only when the fit actually has two
    # components to separate (see gmm_threshold).
    threshold = result.get("threshold")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1.4, zorder=5)
        shown = result.get("threshold_linear", threshold)
        # Anchor the label on whichever side of the line has room, so a
        # threshold near either edge of the axis stays readable.
        lo, hi = ax.get_xlim()
        near_right = (threshold - lo) > 0.5 * (hi - lo)
        ax.annotate(f"thr = {shown:.3g}\n{result['frac_below']:.1%} below",
                    xy=(threshold, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-5 if near_right else 5, -5), textcoords="offset points",
                    ha="right" if near_right else "left", va="top",
                    fontsize=8, color="black")

    verdict = "BIMODAL" if result["bimodal"] else "unimodal"
    ax.set_title(f"{result['label']}: {verdict}  "
                f"(BIC1={result['bic1']:.0f}, BIC2={result['bic2']:.0f}, n={result['n']})",
                fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")


def save_figure(fig, out_dir, out_name):
    """Save, spilling to the backup server on ENOSPC (matches the rest of the pipeline)."""
    out_file = out_dir / out_name
    try:
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        backup_dir = mirror_on_backup_server(out_dir)
        if backup_dir is None:
            raise
        backup_dir.mkdir(parents=True, exist_ok=True)
        out_file = backup_dir / out_name
        print(f"Out of space - retrying on backup server: {out_file}")
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
    return out_file


def main():
    args = parse_args()
    shank = args.shank

    rec_folder_p = Path(rec_folder)
    low_freq_folder = rec_folder_p / "low_freq"
    out_dir = resolve_output_folder(low_freq_folder / "histograms")

    sessions = active_sleep_sessions(sleep_sessions)
    if not sessions:
        raise SystemExit("No active sleep session for this day - nothing to do.")

    for session_key, session_cfg in sessions.items():
        session_label = f"{session_name}{session_cfg['suffix']}"
        print(f"\n{'=' * 70}\n{session_label}  (shank {shank})\n{'=' * 70}")

        bp_file = resolve_existing_file(
            low_freq_folder / f"{session_label}_all_shanks_band_powers.pkl")
        print(f"Loading band powers: {bp_file}")
        with open(bp_file, "rb") as f:
            all_data = pickle.load(f)

        if shank not in all_data["shanks_data"]:
            print(f"  [skip] shank {shank} not in {bp_file.name}")
            continue
        sd = all_data["shanks_data"][shank]
        channel_ids = sd["channel_ids"]
        pc1_all = sd["pc1_spectrogram"]

        velocity_raw = load_velocity(session_cfg, args.velocity_window_sec,
                                     args.velocity_step_sec)
        velocity_result = (gmm_bimodality(velocity_raw, "velocity", log_space=True)
                           if velocity_raw is not None else None)

        summary_rows = []
        for ch_idx, ch_id in enumerate(channel_ids):
            pc1_result = gmm_bimodality(pc1_all[ch_idx], "PC1")
            theta_result = gmm_bimodality(sd["band_powers"][ch_id]["theta_ratio"],
                                          "theta ratio")

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            plot_panel(axes[0], pc1_result, "PC1 (z-scored log power)")
            plot_panel(axes[1], theta_result, "theta ratio (5-10Hz / 2-15Hz)")
            if velocity_result is not None:
                plot_panel(axes[2], velocity_result,
                           f"log10 velocity (px/s, {args.velocity_window_sec:g}s window)")
            else:
                axes[2].set_title("velocity: not available", fontsize=9)
                axes[2].axis("off")
            fig.suptitle(f"{session_label}  sh{shank}  ch{ch_id}", fontsize=11)

            def threshold_note(result, name):
                if result is None or result.get("threshold") is None:
                    return f"{name}=none"
                return f"{name}={result['threshold_linear']:.4g}"

            stamp = (
                f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by "
                f"plot_pc1_theta_velocity_histograms.py  |  "
                f"session={session_name} sleep_session={session_key} shank={shank} ch={ch_id}  |  "
                f"source={bp_file}  |  "
                f"velocity={args.velocity_window_sec:g}s window on a "
                f"{args.velocity_step_sec:g}s grid, fitted in log10  |  "
                f"bimodality test=GaussianMixture(1 vs 2 components), lower BIC wins  |  "
                f"threshold=component crossing (posterior 0.5): "
                f"{threshold_note(pc1_result, 'pc1')} "
                f"{threshold_note(theta_result, 'theta_ratio')} "
                f"{threshold_note(velocity_result, 'velocity')}px/s"
            )
            fig.text(0.005, 0.001, stamp, fontsize=6, color="0.4", ha="left", va="bottom")
            fig.tight_layout(rect=(0, 0.03, 1, 1))

            out_name = f"{session_label}_sh{shank}_ch{ch_id:03d}_pc1_theta_velocity_hist.png"
            out_file = save_figure(fig, out_dir, out_name)
            plt.close(fig)

            print(f"  ch{ch_id}: PC1={'bimodal' if pc1_result['bimodal'] else 'unimodal'}  "
                 f"theta_ratio={'bimodal' if theta_result['bimodal'] else 'unimodal'}  "
                 f"-> {out_file.name}")

            summary_rows.append({
                "channel": ch_id,
                "pc1_n": pc1_result["n"], "pc1_bic1": pc1_result["bic1"],
                "pc1_bic2": pc1_result["bic2"], "pc1_bimodal": pc1_result["bimodal"],
                "pc1_threshold": pc1_result["threshold"],
                "pc1_frac_below": pc1_result["frac_below"],
                "theta_ratio_n": theta_result["n"], "theta_ratio_bic1": theta_result["bic1"],
                "theta_ratio_bic2": theta_result["bic2"],
                "theta_ratio_bimodal": theta_result["bimodal"],
                "theta_ratio_threshold": theta_result["threshold"],
                "theta_ratio_frac_below": theta_result["frac_below"],
                "velocity_n": velocity_result["n"] if velocity_result else 0,
                "velocity_bic1": velocity_result["bic1"] if velocity_result else np.nan,
                "velocity_bic2": velocity_result["bic2"] if velocity_result else np.nan,
                "velocity_bimodal": velocity_result["bimodal"] if velocity_result else False,
                # In px/s (converted back from the log10 fit), so it can be
                # passed straight to score_nrem_delta_velocity --velocity-thresh.
                "velocity_threshold_px_s": (velocity_result["threshold_linear"]
                                            if velocity_result else None),
                "velocity_frac_below": (velocity_result["frac_below"]
                                        if velocity_result else np.nan),
                "velocity_window_sec": args.velocity_window_sec,
            })

        if summary_rows:
            summary_file = out_dir / f"{session_label}_sh{shank}_bimodality_summary.csv"
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\nSummary: {summary_file}")
            print("(velocity columns are session-level - identical across every "
                 "channel row)")


if __name__ == "__main__":
    main()
