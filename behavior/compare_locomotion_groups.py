#!/usr/bin/env python
"""Compare implanted vs unimplanted locomotion from centroid trajectories."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting") from exc


def infer_group(session_id: str) -> str:
    if session_id.startswith("Centimani_implanted/"):
        return "implanted"
    if session_id.startswith("Comparison_unimplanted/"):
        return "unimplanted"
    return "unknown"


def load_manifest_maps(manifest: Path | None) -> tuple[dict[str, str], set[str] | None]:
    if manifest is None or not manifest.exists():
        return {}, None
    df = pd.read_csv(manifest)
    if "session_id" not in df:
        return {}, None
    if "include" in df:
        include = df["include"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        included_ids = set(df.loc[include, "session_id"].astype(str))
        df = df.loc[include].copy()
    else:
        included_ids = None
    if "group" not in df:
        return {}, included_ids
    return {str(row.session_id): str(row.group) for row in df.itertuples(index=False)}, included_ids


def trajectory_files(root: Path) -> list[Path]:
    paths = set(root.rglob("*_centroid_trajectory.csv.gz"))
    paths.update(root.rglob("centroid_trajectory.csv.gz"))
    return sorted(paths)


def trajectory_session_id(path: Path) -> str:
    try:
        return str(pd.read_csv(path, usecols=["session_id"], nrows=1)["session_id"].iloc[0])
    except Exception:
        return path.parent.name


def filter_trajectory_files(
    files: list[Path],
    included_ids: set[str] | None,
    exclude_regex: re.Pattern[str] | None,
) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    excluded: list[str] = []
    for path in files:
        session_id = trajectory_session_id(path)
        if included_ids is not None and session_id not in included_ids:
            excluded.append(session_id)
            continue
        if exclude_regex is not None and exclude_regex.search(session_id):
            excluded.append(session_id)
            continue
        selected.append(path)
    return selected, excluded


def occupancy_metrics(x: np.ndarray, y: np.ndarray, bin_size: float) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return {"coverage_fraction_bbox": math.nan, "occupancy_entropy_norm": math.nan}
    x_edges = np.arange(float(np.nanmin(x)), float(np.nanmax(x)) + bin_size, bin_size)
    y_edges = np.arange(float(np.nanmin(y)), float(np.nanmax(y)) + bin_size, bin_size)
    if len(x_edges) < 2:
        x_edges = np.array([float(np.nanmin(x)), float(np.nanmin(x)) + bin_size])
    if len(y_edges) < 2:
        y_edges = np.array([float(np.nanmin(y)), float(np.nanmin(y)) + bin_size])
    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    occupied = float(np.sum(hist > 0))
    total = float(hist.size)
    probs = hist[hist > 0] / np.sum(hist)
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else math.nan
    entropy_norm = entropy / math.log2(total) if total > 1 and np.isfinite(entropy) else math.nan
    return {
        "coverage_fraction_bbox": occupied / total if total else math.nan,
        "occupancy_entropy_norm": entropy_norm,
    }


def summarize_trajectory(path: Path, group_map: dict[str, str], bin_size: float, moving_threshold_m_s: float) -> dict[str, Any]:
    df = pd.read_csv(path)
    session_id = str(df["session_id"].iloc[0]) if "session_id" in df and len(df) else path.stem
    group = group_map.get(session_id, infer_group(session_id))
    has_m = {"centroid_x_smooth_m", "centroid_y_smooth_m", "speed_m_per_sec"}.issubset(df.columns)
    if has_m:
        x = df["centroid_x_smooth_m"].to_numpy(dtype=float)
        y = df["centroid_y_smooth_m"].to_numpy(dtype=float)
        speed = df["speed_m_per_sec"].to_numpy(dtype=float)
        speed_col = "speed_m_per_sec"
    else:
        x = df["centroid_x_smooth"].to_numpy(dtype=float)
        y = df["centroid_y_smooth"].to_numpy(dtype=float)
        speed = df["speed_px_per_sec"].to_numpy(dtype=float)
        speed_col = "speed_px_per_sec"
    t = df["frame_time"].to_numpy(dtype=float) if "frame_time" in df else np.arange(len(df), dtype=float)
    valid_xy = np.isfinite(x) & np.isfinite(y)
    if {"centroid_x_smooth", "centroid_y_smooth"}.issubset(df.columns):
        x_norm = df["centroid_x_smooth"].to_numpy(dtype=float)
        y_norm = df["centroid_y_smooth"].to_numpy(dtype=float)
        center = np.isfinite(x_norm) & np.isfinite(y_norm) & (x_norm >= 0.25) & (x_norm <= 0.75) & (y_norm >= 0.25) & (y_norm <= 0.75)
        center_fraction = float(np.mean(center[np.isfinite(x_norm) & np.isfinite(y_norm)]))
    else:
        center_fraction = math.nan
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    step_valid = valid_xy[:-1] & valid_xy[1:] & np.isfinite(dt) & (dt > 0)
    total_distance = float(np.nansum(np.sqrt(dx[step_valid] ** 2 + dy[step_valid] ** 2)))
    finite_speed = speed[np.isfinite(speed)]
    duration = float(np.nanmax(t) - np.nanmin(t)) if np.sum(np.isfinite(t)) > 1 else math.nan
    metrics = {
        "session_id": session_id,
        "group": group,
        "trajectory_path": str(path),
        "frames": int(len(df)),
        "duration_sec": duration,
        "centroid_valid_fraction": float(np.mean(df["centroid_valid"].astype(bool))) if "centroid_valid" in df else float(np.mean(valid_xy)),
        "centroid_any_interpolated_fraction": float(np.mean(df["centroid_any_interpolated"].astype(bool))) if "centroid_any_interpolated" in df else math.nan,
        "speed_column": speed_col,
        "speed_mean": float(np.nanmean(finite_speed)) if finite_speed.size else math.nan,
        "speed_median": float(np.nanmedian(finite_speed)) if finite_speed.size else math.nan,
        "speed_p95": float(np.nanpercentile(finite_speed, 95)) if finite_speed.size else math.nan,
        "total_distance": total_distance,
        "moving_fraction": float(np.nanmean(speed > moving_threshold_m_s)) if finite_speed.size and has_m else math.nan,
        "center_fraction": center_fraction,
    }
    metrics.update(occupancy_metrics(x, y, bin_size))
    return metrics


def mannwhitney_for_metrics(session_metrics: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    groups = sorted(group for group in session_metrics["group"].dropna().unique() if group != "unknown")
    if len(groups) != 2:
        return pd.DataFrame(rows)
    a_group, b_group = groups
    for metric in metrics:
        a = session_metrics.loc[session_metrics["group"] == a_group, metric].dropna().to_numpy(dtype=float)
        b = session_metrics.loc[session_metrics["group"] == b_group, metric].dropna().to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "group_a": a_group,
                "group_b": b_group,
                "n_a": len(a),
                "n_b": len(b),
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "difference_median_a_minus_b": float(np.median(a) - np.median(b)),
                "mannwhitney_u": float(stat),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_group_summary(session_metrics: pd.DataFrame, metrics: list[str], n_bootstrap: int, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []
    for group, group_df in session_metrics.groupby("group"):
        if group == "unknown":
            continue
        for metric in metrics:
            values = group_df[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            boots = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_bootstrap)]
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_sessions": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "bootstrap_mean_ci_low": float(np.percentile(boots, 2.5)),
                    "bootstrap_mean_ci_high": float(np.percentile(boots, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def make_metric_plots(session_metrics: pd.DataFrame, output_dir: Path, metrics: list[str]) -> list[Path]:
    paths = []
    groups = [group for group in ["implanted", "unimplanted"] if group in set(session_metrics["group"])]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        values = [session_metrics.loc[session_metrics["group"] == group, metric].dropna().to_numpy(dtype=float) for group in groups]
        if any(len(v) for v in values):
            ax.boxplot(values, labels=groups, showfliers=False)
            for idx, vals in enumerate(values, start=1):
                if len(vals):
                    jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) > 1 else np.array([0.0])
                    ax.scatter(np.full(len(vals), idx) + jitter, vals, color="#333333", s=28, zorder=3)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        path = output_dir / f"{metric}_by_group.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def trajectory_arrays(path: Path, group_map: dict[str, str]) -> dict[str, Any]:
    df = pd.read_csv(path)
    session_id = str(df["session_id"].iloc[0]) if "session_id" in df and len(df) else path.stem
    group = group_map.get(session_id, infer_group(session_id))
    has_m = {"centroid_x_smooth_m", "centroid_y_smooth_m", "speed_m_per_sec"}.issubset(df.columns)
    if has_m:
        x = df["centroid_x_smooth_m"].to_numpy(dtype=float)
        y = df["centroid_y_smooth_m"].to_numpy(dtype=float)
        speed = df["speed_m_per_sec"].to_numpy(dtype=float)
        coord_unit = "m"
        speed_unit = "m/s"
    else:
        x = df["centroid_x_smooth"].to_numpy(dtype=float)
        y = df["centroid_y_smooth"].to_numpy(dtype=float)
        speed = df["speed_px_per_sec"].to_numpy(dtype=float)
        coord_unit = "coordinate"
        speed_unit = "coordinate/s"
    return {
        "session_id": session_id,
        "group": group,
        "path": path,
        "x": x,
        "y": y,
        "speed": speed,
        "coord_unit": coord_unit,
        "speed_unit": speed_unit,
    }


def plot_pooled_velocity(data: list[dict[str, Any]], output_dir: Path, bins: int) -> list[Path]:
    paths: list[Path] = []
    groups = [group for group in ["implanted", "unimplanted"] if any(item["group"] == group for item in data)]
    speed_unit = next((item["speed_unit"] for item in data), "speed")
    pooled = {
        group: np.concatenate([item["speed"][np.isfinite(item["speed"])] for item in data if item["group"] == group])
        for group in groups
    }
    finite_all = np.concatenate([values for values in pooled.values() if len(values)]) if pooled else np.array([])
    upper = float(np.nanpercentile(finite_all, 99.5)) if finite_all.size else 1.0
    upper = upper if np.isfinite(upper) and upper > 0 else 1.0

    for group, values in pooled.items():
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        if len(values):
            ax.hist(values, bins=bins, range=(0, upper), density=True, color="#4c78a8", alpha=0.85)
        ax.set_title(f"{group} pooled velocity distribution")
        ax.set_xlabel(f"speed ({speed_unit})")
        ax.set_ylabel("density")
        path = output_dir / f"pooled_velocity_{group}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    colors = {"implanted": "#4c78a8", "unimplanted": "#f58518"}
    for group, values in pooled.items():
        if len(values):
            ax.hist(values, bins=bins, range=(0, upper), density=True, histtype="step", linewidth=2, color=colors.get(group), label=group)
    ax.set_title("Pooled velocity distribution by group")
    ax.set_xlabel(f"speed ({speed_unit})")
    ax.set_ylabel("density")
    ax.legend()
    path = output_dir / "pooled_velocity_groups_overlay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_pooled_trajectories(data: list[dict[str, Any]], output_dir: Path, stride: int) -> list[Path]:
    paths: list[Path] = []
    groups = [group for group in ["implanted", "unimplanted"] if any(item["group"] == group for item in data)]
    coord_unit = next((item["coord_unit"] for item in data), "coordinate")
    colors = {"implanted": "#4c78a8", "unimplanted": "#f58518"}

    for group in groups:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        for item in data:
            if item["group"] != group:
                continue
            valid = np.isfinite(item["x"]) & np.isfinite(item["y"])
            ax.plot(item["x"][valid][::stride], item["y"][valid][::stride], lw=0.45, alpha=0.45)
        ax.set_title(f"{group} pooled centroid trajectories")
        ax.set_xlabel(f"x ({coord_unit})")
        ax.set_ylabel(f"y ({coord_unit})")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        path = output_dir / f"pooled_trajectory_{group}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for item in data:
        if item["group"] not in groups:
            continue
        valid = np.isfinite(item["x"]) & np.isfinite(item["y"])
        ax.plot(
            item["x"][valid][::stride],
            item["y"][valid][::stride],
            lw=0.45,
            alpha=0.45,
            color=colors.get(item["group"]),
            label=item["group"],
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_title("Pooled centroid trajectories by group")
    ax.set_xlabel(f"x ({coord_unit})")
    ax.set_ylabel(f"y ({coord_unit})")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    path = output_dir / "pooled_trajectory_groups_overlay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_pooled_occupancy(data: list[dict[str, Any]], output_dir: Path, bin_size: float) -> list[Path]:
    paths: list[Path] = []
    groups = [group for group in ["implanted", "unimplanted"] if any(item["group"] == group for item in data)]
    coord_unit = next((item["coord_unit"] for item in data), "coordinate")
    for group in groups:
        xs = [item["x"][np.isfinite(item["x"]) & np.isfinite(item["y"])] for item in data if item["group"] == group]
        ys = [item["y"][np.isfinite(item["x"]) & np.isfinite(item["y"])] for item in data if item["group"] == group]
        if not xs or not any(len(x) for x in xs):
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        x_edges = np.arange(float(np.nanmin(x)), float(np.nanmax(x)) + bin_size, bin_size)
        y_edges = np.arange(float(np.nanmin(y)), float(np.nanmax(y)) + bin_size, bin_size)
        if len(x_edges) < 2:
            x_edges = np.array([float(np.nanmin(x)), float(np.nanmin(x)) + bin_size])
        if len(y_edges) < 2:
            y_edges = np.array([float(np.nanmin(y)), float(np.nanmin(y)) + bin_size])
        hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = ax.imshow(
            np.log1p(hist.T),
            origin="upper",
            aspect="equal",
            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
            cmap="magma",
        )
        fig.colorbar(image, ax=ax, label="log(pooled occupancy + 1)")
        ax.set_title(f"{group} pooled occupancy")
        ax.set_xlabel(f"x ({coord_unit})")
        ax.set_ylabel(f"y ({coord_unit})")
        path = output_dir / f"pooled_occupancy_{group}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def make_pooled_group_plots(
    files: list[Path],
    group_map: dict[str, str],
    output_dir: Path,
    occupancy_bin_size: float,
    velocity_bins: int,
    trajectory_stride: int,
) -> list[Path]:
    data = [trajectory_arrays(path, group_map) for path in files]
    pooled_dir = output_dir / "pooled_group_plots"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_pooled_velocity(data, pooled_dir, velocity_bins))
    paths.extend(plot_pooled_trajectories(data, pooled_dir, max(1, trajectory_stride)))
    paths.extend(plot_pooled_occupancy(data, pooled_dir, occupancy_bin_size))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare locomotion metrics between implanted and unimplanted groups.")
    parser.add_argument("trajectory_root", nargs="?", type=Path, default=Path("analysis_outputs/dlc_locomotion_calibrated_shared"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--occupancy-bin-size", type=float, default=0.02, help="Bin size in meters when meter columns exist.")
    parser.add_argument("--moving-threshold-m-s", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=123)
    parser.add_argument("--pooled-velocity-bins", type=int, default=100)
    parser.add_argument("--pooled-trajectory-stride", type=int, default=10)
    parser.add_argument("--exclude-session-filter", default=None, help="Regex applied to session_id values to exclude before statistics.")
    args = parser.parse_args()

    root = args.trajectory_root.resolve()
    output_dir = args.output_dir or root / "group_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_map, included_ids = load_manifest_maps(args.manifest)
    files = trajectory_files(root)
    exclude_regex = re.compile(args.exclude_session_filter) if args.exclude_session_filter else None
    files, excluded_sessions = filter_trajectory_files(files, included_ids, exclude_regex)
    if not files:
        raise SystemExit(f"No centroid trajectory files found under {root}")
    metrics_df = pd.DataFrame(
        [summarize_trajectory(path, group_map, args.occupancy_bin_size, args.moving_threshold_m_s) for path in files]
    )
    metrics = [
        "speed_mean",
        "speed_median",
        "speed_p95",
        "total_distance",
        "moving_fraction",
        "coverage_fraction_bbox",
        "occupancy_entropy_norm",
        "center_fraction",
    ]
    stats_df = mannwhitney_for_metrics(metrics_df, metrics)
    summary_df = bootstrap_group_summary(metrics_df, metrics, args.n_bootstrap, args.rng_seed)
    plot_paths = make_metric_plots(metrics_df, output_dir, metrics)
    pooled_plot_paths = make_pooled_group_plots(
        files,
        group_map,
        output_dir,
        args.occupancy_bin_size,
        args.pooled_velocity_bins,
        args.pooled_trajectory_stride,
    )
    metrics_path = output_dir / "group_locomotion_session_metrics.csv"
    summary_path = output_dir / "group_locomotion_summary.csv"
    stats_path = output_dir / "group_locomotion_tests.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "trajectory_root": str(root),
        "manifest": str(args.manifest) if args.manifest else None,
        "n_sessions": int(len(metrics_df)),
        "groups": metrics_df["group"].value_counts().to_dict(),
        "excluded_sessions": excluded_sessions,
        "exclude_session_filter": args.exclude_session_filter,
        "occupancy_bin_size": args.occupancy_bin_size,
        "moving_threshold_m_s": args.moving_threshold_m_s,
        "outputs": {
            "session_metrics": str(metrics_path),
            "group_summary": str(summary_path),
            "tests": str(stats_path),
            "plots": [str(path) for path in plot_paths],
            "pooled_group_plots": [str(path) for path in pooled_plot_paths],
        },
        "note": "Statistical tests use sessions, not frames, as the independent unit.",
    }
    report_path = output_dir / "group_locomotion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Compared {len(metrics_df)} sessions")
    print(f"Groups: {metrics_df['group'].value_counts().to_dict()}")
    print(f"Wrote {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
