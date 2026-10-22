r"""Robust probe-level NREM/REM/WAKE/QUIET scoring.

Channel evidence is normalized per channel, voted within shank, then voted
across shanks.  Physical velocity is session-level and is therefore applied
once, after spectral voting.  Artifact/missing-data epochs are represented by
``valid_mask=False`` rather than treated as a fifth biological state.

The NREM minimum-bout requirement is intentionally prominent and reviewable:
``--nrem-min-bout-sec`` controls it, and every run reports how 0, 30, and 60 s
requirements change the retained NREM time.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import argparse
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from band_powers_io import band_time, load_spectrograms
from proc_func_velocity import aggregate_speed, velocity_output_name
from sleep_artifact_detection import detect_broadband_artifacts, robust_z
from sleep_pipeline_config import (
    VELOCITY_MIN_COVERAGE,
    VELOCITY_SOURCE,
    VELOCITY_STEP_SEC,
    active_sleep_sessions,
    artifact_params,
    mirror_on_backup_server,
    original_fs,
    rec_folder,
    resolve_existing_file,
    resolve_output_folder,
    session_name,
    shanks as configured_shanks,
    sleep_detect_params,
    sleep_sessions,
    video_folder,
)


STATE_LABELS = ("NREM", "REM", "WAKE", "QUIET")
STATE_TO_CODE = {label: index for index, label in enumerate(STATE_LABELS)}
STATE_COLORS = {"NREM": "#4C72B0", "REM": "#C44E52",
                "WAKE": "#DD8452", "QUIET": "#8C8C8C"}

DEFAULT_NREM_MIN_BOUT_SEC = 60.0  # REVIEW FLAG: empirical heuristic, not ground truth.


def robust_channel_score(values, log=False):
    values = np.asarray(values, dtype=float)
    if log:
        values = np.log(np.maximum(values, 1e-12))
    return robust_z(values)


def gmm_threshold(values, *, log1p=False, min_samples=100):
    """Two-component GMM crossing when BIC supports bimodality, else None."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_samples:
        return None
    transformed = np.log1p(np.maximum(values, 0)) if log1p else values
    X = transformed[:, None]
    # scikit-learn emits one Windows/MKL warning for every initialization.
    # It is harmless here but can bury the scoring report for 100s of channels.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="KMeans is known to have a memory leak on Windows.*"
        )
        warnings.filterwarnings(
            "ignore", message="Could not find the number of physical cores.*"
        )
        one = GaussianMixture(1, random_state=0, n_init=1).fit(X)
        two = GaussianMixture(2, random_state=0, n_init=2).fit(X)
    if two.bic(X) >= one.bic(X):
        return None
    means = two.means_.ravel()
    order = np.argsort(means)
    lo, hi = means[order]
    grid = np.linspace(lo, hi, 1024)[:, None]
    weighted = np.exp(two._estimate_weighted_log_prob(grid))[:, order]
    crossing = int(np.argmin(np.abs(weighted[:, 0] - weighted[:, 1])))
    threshold = float(grid[crossing, 0])
    return float(np.expm1(threshold)) if log1p else threshold


def hysteresis_mask(score, enter, exit):
    score = np.asarray(score, dtype=float)
    mask = np.zeros(score.size, dtype=bool)
    active = False
    for index, value in enumerate(score):
        if not np.isfinite(value):
            active = False
        elif not active and value >= enter:
            active = True
        elif active and value < exit:
            active = False
        mask[index] = active
    return mask


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.where(changes == 1)[0], np.where(changes == -1)[0]))


def bridge_gaps(mask, max_gap_bins):
    output = np.asarray(mask, dtype=bool).copy()
    spans = runs(output)
    for (_, left_end), (right_start, _) in zip(spans[:-1], spans[1:]):
        if right_start - left_end <= max_gap_bins:
            output[left_end:right_start] = True
    return output


def remove_short_bouts(mask, min_bins):
    output = np.asarray(mask, dtype=bool).copy()
    for start, end in runs(output):
        if end - start < min_bins:
            output[start:end] = False
    return output


def intervals_from_mask(mask, times, epoch_sec):
    intervals = []
    for start, end in runs(mask):
        left = max(0.0, float(times[start] - epoch_sec / 2))
        right = float(times[end - 1] + epoch_sec / 2)
        intervals.append((left, right))
    return intervals


def first_existing(paths):
    for path in paths:
        if path is None:
            continue
        try:
            if Path(path).is_file():
                return Path(path)
        except OSError:
            continue
    return None


def with_mirror(path):
    mirror = mirror_on_backup_server(path)
    return ([mirror] if mirror is not None else []) + [Path(path)]


def load_velocity(session_cfg, grid, window_sec):
    proc_file = session_cfg.get("proc_file")
    if not proc_file:
        return np.full(len(grid), np.nan), None
    filename = velocity_output_name(proc_file, VELOCITY_SOURCE)
    velocity_path = first_existing(
        with_mirror(Path(proc_file).parent / filename)
        + with_mirror(Path(video_folder) / filename)
    )
    sync_path = first_existing(with_mirror(
        Path(rec_folder) / f"sync_times{session_cfg['suffix']}.pkl"
    ))
    if velocity_path is None or sync_path is None:
        return np.full(len(grid), np.nan), None
    with velocity_path.open("rb") as stream:
        velocity = pickle.load(stream)
    with sync_path.open("rb") as stream:
        sync = pickle.load(stream)
    camera_t = np.asarray(velocity["time_stamp"], dtype=float)
    speed = np.asarray(velocity["velocity"], dtype=float)
    proc = np.asarray(sync["proc_rising_time"], dtype=float)
    ephys = np.asarray(sync["SG_rising_time"], dtype=float) / float(original_fs)
    inside = (camera_t >= proc[0]) & (camera_t <= proc[-1])
    camera_t, speed = camera_t[inside], speed[inside]
    ephys_t = np.interp(camera_t, [proc[0], proc[-1]], [ephys[0], ephys[-1]])
    windowed = aggregate_speed(ephys_t, speed, grid, window_sec,
                               min_coverage=VELOCITY_MIN_COVERAGE)
    return np.asarray(windowed["mean"], dtype=float), velocity_path


def _align(values, length):
    values = np.asarray(values, dtype=float)
    output = np.full(length, np.nan, dtype=float)
    output[:min(length, values.size)] = values[:length]
    return output


def build_spectral_votes(all_data, selected_shanks, grid, band_path):
    """Return channel-level spectral evidence grouped by shank."""
    n = grid.size
    shank_nrem_scores = {}
    shank_theta = {}
    channel_qc = []
    shank_spectrograms = []
    artifact_times = None
    artifact_freqs = None

    for shank in selected_shanks:
        if shank not in all_data["shanks_data"]:
            continue
        data = all_data["shanks_data"][shank]
        other_grid = np.asarray(band_time(data), dtype=float)
        overlap = min(n, other_grid.size)
        if overlap == 0 or not np.allclose(other_grid[:overlap], grid[:overlap]):
            raise ValueError(f"shank {shank} is not on the shared time grid")
        channel_scores = []
        theta_traces = []
        for channel in data["channel_ids"]:
            delta = _align(data["band_powers"][channel]["delta"], n)
            gamma = _align(data["band_powers"][channel]["gamma"], n)
            theta = _align(data["band_powers"][channel]["theta_ratio"], n)
            finite = np.isfinite(delta) & np.isfinite(gamma) & (delta > 0) & (gamma > 0)
            coverage = float(np.mean(finite))
            delta_mad = np.nanmedian(np.abs(np.log(np.maximum(delta, 1e-12))
                                            - np.nanmedian(np.log(np.maximum(delta, 1e-12)))))
            gamma_mad = np.nanmedian(np.abs(np.log(np.maximum(gamma, 1e-12))
                                            - np.nanmedian(np.log(np.maximum(gamma, 1e-12)))))
            usable = coverage >= 0.8 and delta_mad > 1e-6 and gamma_mad > 1e-6
            channel_qc.append({"shank": int(shank), "channel": int(channel),
                               "finite_fraction": coverage,
                               "delta_log_mad": float(delta_mad),
                               "gamma_log_mad": float(gamma_mad),
                               "usable": bool(usable)})
            if not usable:
                continue
            score = robust_channel_score(delta, log=True) - robust_channel_score(gamma, log=True)
            channel_scores.append(score)
            theta_traces.append(theta)
        if channel_scores:
            shank_nrem_scores[shank] = np.stack(channel_scores)
            shank_theta[shank] = theta_traces

        spectrogram = load_spectrograms(data, band_path, shank)
        shank_spectrograms.append(np.nanmean(spectrogram, axis=0))
        if artifact_times is None:
            artifact_times = np.asarray(data["spectrogram_times"], dtype=float)
            artifact_freqs = np.asarray(data["spectrogram_freqs"], dtype=float)

    if not shank_nrem_scores:
        raise ValueError("No shank retained enough valid channels for scoring")
    min_spec = min(array.shape[1] for array in shank_spectrograms)
    mean_spec = np.nanmedian(
        np.stack([array[:, :min_spec] for array in shank_spectrograms]), axis=0
    )
    artifact_spec, artifact_info = detect_broadband_artifacts(
        mean_spec, artifact_freqs, artifact_times[:min_spec],
        n_mad=artifact_params["n_mad"],
        dilate_sec=artifact_params["dilate_sec"],
        fmax=artifact_params["fmax"],
    )
    artifact = np.interp(grid, artifact_times[:len(artifact_spec)],
                         artifact_spec.astype(float)) > 0.5
    return shank_nrem_scores, shank_theta, artifact, artifact_info, channel_qc


def score_block(band_path, session_key, session_cfg, args):
    with Path(band_path).open("rb") as stream:
        all_data = pickle.load(stream)
    available = sorted(int(value) for value in all_data["shanks_data"])
    default_shanks = sleep_detect_params.get("shanks") or configured_shanks
    selected = args.shanks if args.shanks is not None else [
        value for value in default_shanks if value in available
    ]
    first = all_data["shanks_data"][next(shank for shank in selected if shank in available)]
    grid = np.asarray(band_time(first), dtype=float)
    epoch_sec = float(np.nanmedian(np.diff(grid)))
    shank_scores, shank_theta, artifact, artifact_info, channel_qc = (
        build_spectral_votes(all_data, selected, grid, band_path)
    )

    # Explicit hierarchy: threshold each channel, majority-vote within each
    # shank, then majority-vote across shanks. This prevents a shank with many
    # channels from dominating the probe-level decision.
    shank_vote_fractions = []
    for scores in shank_scores.values():
        finite = np.isfinite(scores)
        valid_channels = np.sum(finite, axis=0)
        fraction = np.divide(
            np.sum((scores >= args.nrem_channel_score_threshold) & finite, axis=0),
            valid_channels,
            out=np.full(grid.size, np.nan),
            where=valid_channels > 0,
        )
        shank_vote_fractions.append(fraction)
    shank_vote_fractions = np.stack(shank_vote_fractions)
    valid_shanks = np.isfinite(shank_vote_fractions)
    n_valid_shanks = np.sum(valid_shanks, axis=0)
    shank_votes = (shank_vote_fractions >= args.nrem_channel_fraction) & valid_shanks
    nrem_confidence = np.divide(
        np.sum(shank_votes, axis=0),
        n_valid_shanks,
        out=np.full(grid.size, np.nan),
        where=n_valid_shanks > 0,
    )
    spectral_valid = n_valid_shanks >= args.min_valid_shanks
    raw_nrem = hysteresis_mask(
        nrem_confidence, args.nrem_enter_fraction, args.nrem_exit_fraction
    ) & spectral_valid & ~artifact
    bridged_nrem = bridge_gaps(
        raw_nrem, int(round(args.nrem_merge_gap_sec / epoch_sec))
    ) & ~artifact

    sensitivity = {}
    for seconds in (0.0, 30.0, 60.0):
        candidate = remove_short_bouts(
            bridged_nrem, int(round(seconds / epoch_sec))
        )
        spans = intervals_from_mask(candidate, grid, epoch_sec)
        sensitivity[str(int(seconds))] = {
            "n_bouts": len(spans),
            "total_sec": float(np.sum(candidate) * epoch_sec),
        }
    nrem = remove_short_bouts(
        bridged_nrem, int(round(args.nrem_min_bout_sec / epoch_sec))
    )

    speed, velocity_path = load_velocity(session_cfg, grid, args.velocity_window_sec)
    speed_threshold = args.velocity_threshold
    if speed_threshold is None:
        speed_threshold = gmm_threshold(speed, log1p=True)
    if speed_threshold is None or not np.isfinite(speed_threshold):
        speed_threshold = 5.0
    stationary = np.isfinite(speed) & (speed < speed_threshold)

    # Fit theta thresholds only to stationary, non-NREM samples. Thresholds are
    # per channel; votes are combined channel->shank->probe.
    theta_shank_votes = []
    theta_thresholds = {}
    theta_candidates = (~nrem) & stationary & (~artifact)
    for shank, traces in shank_theta.items():
        transformed = [np.log(np.maximum(trace, 1e-12)) for trace in traces]
        fitted = [gmm_threshold(trace[theta_candidates]) for trace in transformed]
        good = [value for value in fitted if value is not None and np.isfinite(value)]
        fallback = float(np.median(good)) if good else float(np.nanmedian(
            np.concatenate([trace[theta_candidates] for trace in transformed])
        ))
        channel_votes = []
        theta_thresholds[str(shank)] = []
        for trace, threshold in zip(transformed, fitted):
            threshold = fallback if threshold is None else threshold
            theta_thresholds[str(shank)].append(float(threshold))
            channel_votes.append(trace >= threshold)
        theta_shank_votes.append(
            np.mean(np.stack(channel_votes), axis=0) >= args.theta_channel_fraction
        )
    theta_confidence = np.mean(np.stack(theta_shank_votes), axis=0)
    theta_high = theta_confidence >= args.theta_shank_fraction

    # A session with no tracking at all (no PROC/DLC, or no sync) has no speed
    # on any epoch, which would make every epoch invalid. With
    # --allow-missing-velocity NREM is then scored from the spectral vote alone
    # and the other epochs stay unlabelled: without speed WAKE cannot be told
    # from QUIET/REM. A session that HAS tracking still rejects its NaN-speed
    # epochs - "could not see whether it moved" must not read as "was still".
    velocity_available = bool(np.isfinite(speed).any())
    spectral_only = (not velocity_available) and args.allow_missing_velocity
    if spectral_only:
        print(f"  NOTE [{session_key}]: no velocity for this session - NREM scored from the "
              "spectral vote alone, no movement gate; WAKE/REM/QUIET left unlabelled")
        valid = spectral_valid & ~artifact
    else:
        valid = spectral_valid & ~artifact & np.isfinite(speed)
    states = np.full(grid.size, "", dtype=object)
    states[valid & nrem] = "NREM"
    remaining = valid & ~nrem & (not spectral_only)
    states[remaining & ~stationary] = "WAKE"
    states[remaining & stationary & theta_high] = "REM"
    states[remaining & stationary & ~theta_high] = "QUIET"
    state_code = np.full(grid.size, -1, dtype=np.int8)
    for label, code in STATE_TO_CODE.items():
        state_code[states == label] = code

    state_intervals = {
        label: intervals_from_mask(states == label, grid, epoch_sec)
        for label in STATE_LABELS
    }
    consolidated = [interval for interval in state_intervals["NREM"]
                    if interval[1] - interval[0] >= args.nrem_consolidated_sec]
    longest = max(consolidated, key=lambda interval: interval[1] - interval[0],
                  default=None)
    return {
        "schema_version": 1,
        "session": f"{session_name}{session_cfg['suffix']}",
        "sleep_session": session_key,
        "shanks": sorted(int(value) for value in shank_scores),
        "epoch_sec": epoch_sec,
        "epoch_times": grid,
        "state_labels": STATE_LABELS,
        "state_code": state_code,
        "states": states,
        "valid_mask": valid,
        "artifact_mask": artifact,
        "nrem_confidence": nrem_confidence,
        # Compatibility name used by OFF-state correlation analyses. Unlike
        # the old z-score this is an interpretable 0..1 probe vote fraction.
        "sw_index": nrem_confidence,
        "nrem_shank_vote_fraction": shank_vote_fractions,
        "theta_confidence": theta_confidence,
        "velocity_px_s": speed,
        "velocity_available": velocity_available,
        "spectral_only": spectral_only,
        "nrem_epoch_mask": states == "NREM",
        "nrem_mask": states == "NREM",
        "bout_intervals_s": state_intervals["NREM"],
        "consolidated_windows_s": consolidated,
        "fully_asleep_window_s": longest,
        "state_intervals_s": state_intervals,
        "channel_qc": channel_qc,
        "theta_thresholds_log": theta_thresholds,
        "nrem_bout_sensitivity": sensitivity,
        "artifact_info": artifact_info,
        "sleep_start_sample": session_cfg.get("start_sample"),
        "sources": {"band_powers": str(band_path),
                    "velocity": str(velocity_path) if velocity_path else None},
        "params": {
            "nrem_channel_score_threshold": args.nrem_channel_score_threshold,
            "nrem_channel_fraction": args.nrem_channel_fraction,
            "nrem_enter_fraction": args.nrem_enter_fraction,
            "nrem_exit_fraction": args.nrem_exit_fraction,
            "nrem_merge_gap_sec": args.nrem_merge_gap_sec,
            "nrem_min_bout_sec": args.nrem_min_bout_sec,
            "nrem_min_bout_review_required": True,
            "nrem_consolidated_sec": args.nrem_consolidated_sec,
            "min_valid_shanks": args.min_valid_shanks,
            "theta_channel_fraction": args.theta_channel_fraction,
            "theta_shank_fraction": args.theta_shank_fraction,
            "velocity_threshold_px_s": float(speed_threshold),
            "velocity_window_sec": args.velocity_window_sec,
            "source": Path(__file__).name,
        },
    }


def print_report(result):
    print(f"\n{result['session']} ({result['epoch_sec']:.3g} s grid)")
    valid = result["valid_mask"]
    for label in STATE_LABELS:
        count = int(np.sum(result["states"] == label))
        print(f"  {label:<5}: {count:>6} epochs, "
              f"{count * result['epoch_sec'] / 60:7.1f} min, "
              f"{100 * count / max(np.sum(valid), 1):5.1f}% of valid")
    print(f"  invalid/artifact: {np.sum(~valid)} epochs")
    print("  REVIEW FLAG -- NREM minimum-bout requirement:")
    for seconds, row in result["nrem_bout_sensitivity"].items():
        print(f"    minimum {seconds:>2}s: {row['n_bouts']:>3} bouts, "
              f"{row['total_sec'] / 60:6.1f} min retained")
    chosen = result["params"]["nrem_min_bout_sec"]
    print(f"    CHOSEN: {chosen:g}s. This is a heuristic; inspect the 0/30/60s "
          "sensitivity before treating it as a biological requirement.")


def plot_result(result, path):
    time_min = result["epoch_times"] / 60.0
    figure, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True,
                                gridspec_kw={"height_ratios": (2, 2, 2, 0.7)})
    axes[0].plot(time_min, result["nrem_confidence"], color="#4C72B0", lw=0.8)
    axes[0].axhline(result["params"]["nrem_enter_fraction"], color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("NREM\nshank vote")
    axes[1].plot(time_min, result["theta_confidence"], color="#C44E52", lw=0.8)
    axes[1].axhline(result["params"]["theta_shank_fraction"], color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Theta\nshank vote")
    axes[2].plot(time_min, result["velocity_px_s"], color="#DD8452", lw=0.8)
    axes[2].axhline(result["params"]["velocity_threshold_px_s"], color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("Speed\n(px/s)")
    image = np.ma.masked_where(result["state_code"] < 0, result["state_code"])[None, :]
    from matplotlib.colors import ListedColormap
    axes[3].imshow(image, aspect="auto", interpolation="nearest",
                   extent=[time_min[0], time_min[-1], 0, 1],
                   cmap=ListedColormap([STATE_COLORS[label] for label in STATE_LABELS]),
                   vmin=0, vmax=len(STATE_LABELS) - 1)
    axes[3].set_yticks([])
    axes[3].set_xlabel("Time (minutes)")
    axes[0].set_title(
        f"{result['session']} robust four-state scoring | "
        f"NREM min bout={result['params']['nrem_min_bout_sec']:g}s (REVIEW REQUIRED)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", choices=("pre", "post"))
    parser.add_argument("--shanks", type=int, nargs="+", default=None,
                        help="default: sleep_detect_params['shanks'] from config")
    parser.add_argument("--min-valid-shanks", type=int, default=3)
    parser.add_argument("--nrem-channel-score-threshold", type=float,
                        default=sleep_detect_params.get("nrem_sw_z_thresh", 1.0))
    parser.add_argument("--nrem-channel-fraction", type=float, default=0.6,
                        help="fraction of valid channels needed for a shank vote")
    parser.add_argument("--nrem-enter-fraction", type=float, default=0.6)
    parser.add_argument("--nrem-exit-fraction", type=float, default=0.4)
    parser.add_argument("--nrem-merge-gap-sec", type=float,
                        default=sleep_detect_params.get("merge_gap_sec", 20.0))
    parser.add_argument("--nrem-min-bout-sec", type=float,
                        default=sleep_detect_params.get(
                            "min_bout_sec", DEFAULT_NREM_MIN_BOUT_SEC
                        ),
                        help="REVIEW FLAG: heuristic minimum NREM duration; compare "
                             "0, 30, and 60 s in every run (default 60)")
    parser.add_argument("--nrem-consolidated-sec", type=float,
                        default=sleep_detect_params.get("consolidated_sec", 120.0))
    parser.add_argument("--theta-channel-fraction", type=float, default=0.6)
    parser.add_argument("--theta-shank-fraction", type=float, default=0.6)
    parser.add_argument("--velocity-threshold", type=float, default=None,
                        help="px/s; default: two-component GMM, falling back to 5")
    parser.add_argument("--velocity-window-sec", type=float, default=VELOCITY_STEP_SEC)
    parser.add_argument("--allow-missing-velocity", action="store_true",
                        help="for a session with no tracking at all, score NREM from the "
                             "spectral vote alone instead of marking every epoch invalid")
    parser.add_argument("--no-save", action="store_true",
                        help="calculate and report without writing share artifacts")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0 <= args.nrem_exit_fraction <= args.nrem_enter_fraction <= 1:
        raise ValueError("NREM fractions must satisfy 0 <= exit <= enter <= 1")
    if not 0 <= args.nrem_channel_fraction <= 1:
        raise ValueError("--nrem-channel-fraction must be between 0 and 1")
    sessions = active_sleep_sessions(sleep_sessions)
    if args.sessions:
        sessions = {key: cfg for key, cfg in sessions.items() if key in args.sessions}
    for key, cfg in sessions.items():
        label = f"{session_name}{cfg['suffix']}"
        band_path = resolve_existing_file(
            Path(rec_folder) / "low_freq" / f"{label}_all_shanks_band_powers.pkl"
        )
        result = score_block(band_path, key, cfg, args)
        print_report(result)
        if args.no_save:
            continue
        out_dir = Path(resolve_output_folder(
            Path(band_path).parent / "sleep_segmentation"
        ))
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = out_dir / f"{label}_sleep_periods.pkl"
        with pkl_path.open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        png_path = out_dir / f"{label}_sleep_periods.png"
        plot_result(result, png_path)
        print(f"  saved: {pkl_path}\n         {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
