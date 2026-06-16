#!/usr/bin/env python
"""Rebuild locomotion plots from compact centroid trajectory exports.

This script is intentionally independent from the full DLC tables. It reads the
`centroid_trajectory.csv.gz` or older `*_centroid_trajectory.csv.gz` files
written by `dlc_locomotion_pipeline.py`,
plus optional parameter/summary files, so plot styling and binning can be tuned
without reloading DLC keypoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting") from exc


def safe_stem(path: Path) -> str:
    suffix = "_centroid_trajectory.csv.gz"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    if path.name == "centroid_trajectory.csv.gz":
        return path.parent.name
    return path.stem


def trajectory_files(root: Path) -> list[Path]:
    paths = set(root.rglob("*_centroid_trajectory.csv.gz"))
    paths.update(root.rglob("centroid_trajectory.csv.gz"))
    return sorted(paths)


def load_bin_size(args: argparse.Namespace) -> float:
    if args.coverage_bin_size is not None:
        return args.coverage_bin_size
    if args.parameters_json and args.parameters_json.exists():
        params = json.loads(args.parameters_json.read_text(encoding="utf-8"))
        if params.get("coverage_bin_size_m") is not None:
            return float(params["coverage_bin_size_m"])
        return float(params.get("coverage_bin_size_px", 25.0))
    return 25.0


def plot_one_trajectory(path: Path, output_dir: Path, bin_size: float, title_prefix: str) -> list[Path]:
    df = pd.read_csv(path)
    if {"centroid_x_smooth_m", "centroid_y_smooth_m"}.issubset(df.columns):
        x = df["centroid_x_smooth_m"].to_numpy(dtype=float)
        y = df["centroid_y_smooth_m"].to_numpy(dtype=float)
        speed = df["speed_m_per_sec"].to_numpy(dtype=float) if "speed_m_per_sec" in df else np.full(len(df), np.nan)
        x_label = "x (m)"
        y_label = "y (m)"
        speed_label = "speed (m/s)"
    else:
        x = df["centroid_x_smooth"].to_numpy(dtype=float)
        y = df["centroid_y_smooth"].to_numpy(dtype=float)
        speed = df["speed_px_per_sec"].to_numpy(dtype=float) if "speed_px_per_sec" in df else np.full(len(df), np.nan)
        x_label = "corrected arena x"
        y_label = "corrected arena y"
        speed_label = "speed per second"
    valid = np.isfinite(x) & np.isfinite(y)
    session_name = safe_stem(path)
    title = title_prefix or session_name
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    if np.any(valid):
        ax.plot(x[valid], y[valid], lw=0.6, color="#1f77b4")
        ax.scatter(x[valid][0:1], y[valid][0:1], s=40, color="green", label="start")
        ax.scatter(x[valid][-1:], y[valid][-1:], s=40, color="red", label="end")
    ax.set_title(f"{title} centroid trajectory")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(loc="best")
    out = output_dir / f"{session_name}_centroid_trajectory.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    paths.append(out)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if np.any(valid):
        x_edges = np.arange(float(np.nanmin(x)), float(np.nanmax(x)) + bin_size, bin_size)
        y_edges = np.arange(float(np.nanmin(y)), float(np.nanmax(y)) + bin_size, bin_size)
        if x_edges.size < 2:
            x_edges = np.array([float(np.nanmin(x)), float(np.nanmin(x)) + bin_size])
        if y_edges.size < 2:
            y_edges = np.array([float(np.nanmin(y)), float(np.nanmin(y)) + bin_size])
        hist, _, _ = np.histogram2d(x[valid], y[valid], bins=(x_edges, y_edges))
        image = ax.imshow(
            np.log1p(hist.T),
            origin="upper",
            aspect="equal",
            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
            cmap="magma",
        )
        fig.colorbar(image, ax=ax, label="log(occupancy + 1)")
    ax.set_title(f"{title} occupancy")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    out = output_dir / f"{session_name}_occupancy_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    paths.append(out)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size:
        upper = min(float(np.percentile(finite_speed, 99.5)), float(np.nanmax(finite_speed)))
        if np.isfinite(upper) and upper > 0:
            ax.hist(finite_speed, bins=80, range=(0, upper), color="#4c78a8")
    ax.set_title(f"{title} velocity")
    ax.set_xlabel(speed_label)
    ax.set_ylabel("frames")
    out = output_dir / f"{session_name}_velocity_histogram.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    paths.append(out)

    return paths


def write_plot_index(output_dir: Path, generated: list[dict[str, str]], bin_size: float) -> Path:
    path = output_dir / "centroid_plot_outputs.json"
    payload = {
        "coverage_bin_size": bin_size,
        "trajectory_files": generated,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make plots from compact centroid trajectory files.")
    parser.add_argument(
        "trajectory_root",
        nargs="?",
        type=Path,
        default=Path("analysis_outputs/dlc_locomotion_calibrated_shared"),
        help="Locomotion output root or a directory containing trajectory files.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--coverage-bin-size", type=float, default=None)
    parser.add_argument("--parameters-json", type=Path, default=None)
    parser.add_argument("--session-filter", default=None, help="Substring filter applied to trajectory filenames.")
    args = parser.parse_args()

    root = args.trajectory_root.resolve()
    default_params = root / "dlc_locomotion_parameters.json"
    if args.parameters_json is None and default_params.exists():
        args.parameters_json = default_params
    output_dir = args.output_dir or root / "centroid_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_size = load_bin_size(args)

    trajectories = trajectory_files(root)
    if args.session_filter:
        trajectories = [path for path in trajectories if args.session_filter in path.name or args.session_filter in path.parent.name]
    if not trajectories:
        raise SystemExit(f"No centroid trajectory CSV files found under {root}")

    generated: list[dict[str, str]] = []
    for trajectory in trajectories:
        paths = plot_one_trajectory(trajectory, output_dir, bin_size, safe_stem(trajectory))
        generated.append(
            {
                "trajectory": str(trajectory),
                "plots": ";".join(str(path) for path in paths),
            }
        )
    index = write_plot_index(output_dir, generated, bin_size)
    print(f"Plotted {len(generated)} centroid trajectories")
    print(f"Wrote {output_dir}")
    print(f"Index: {index}")


if __name__ == "__main__":
    main()
