"""Apply arena calibration to raw camera-frame coordinates.

This module converts original video pixel coordinates into the corrected square
arena coordinate system saved by ``arena_calibration.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TRANSFORM_TYPE = "boundary_polynomial_raw_pixel_to_square"
DEFAULT_OUTPUT_SIZE = 520.0


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load a calibration JSON written by arena_calibration.py."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _transform_from_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    transform = calibration.get("transform", {})
    if transform.get("type") != TRANSFORM_TYPE:
        raise ValueError(f"Unsupported calibration transform: {transform.get('type')}")
    return transform


def correct_points(
    points: np.ndarray,
    calibration: dict[str, Any],
    output_size: float | None = DEFAULT_OUTPUT_SIZE,
) -> np.ndarray:
    """Convert an array of raw ``(x, y)`` coordinates to corrected coordinates.

    Parameters
    ----------
    points:
        Array-like coordinates with shape ``(N, 2)``. A single ``(2,)`` point is
        also accepted and returned as shape ``(1, 2)``.
    calibration:
        Parsed calibration JSON from :func:`load_calibration`.
    output_size:
        Corrected square arena side length for the returned coordinates. The
        default returns coordinates in a 520 x 520 square. Use ``None`` to keep
        the calibration's native target units, usually normalized 0..1.
    """
    transform = _transform_from_calibration(calibration)
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        if pts.shape[0] != 2:
            raise ValueError("A single point must have shape (2,).")
        pts = pts.reshape(1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2) or (2,).")

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    mapped = np.full_like(pts, np.nan, dtype=float)
    if not np.any(finite):
        return mapped

    terms = [(int(px), int(py)) for px, py in transform["terms"]]
    width = float(transform["source_width"])
    height = float(transform["source_height"])

    sx = pts[finite, 0] / max(1.0, width - 1)
    sy = pts[finite, 1] / max(1.0, height - 1)
    design = np.column_stack([(sx**px) * (sy**py) for px, py in terms])
    base = np.column_stack(
        [
            design @ np.asarray(transform["forward_coeff_x"], dtype=float),
            design @ np.asarray(transform["forward_coeff_y"], dtype=float),
        ]
    )

    homography = np.asarray(transform.get("post_homography", np.eye(3).tolist()), dtype=float)
    homog = np.column_stack([base, np.ones(len(base))]) @ homography.T
    mapped[finite] = homog[:, :2] / homog[:, 2:3]
    if output_size is not None:
        target_size = float(transform.get("target_size", calibration.get("target_size", 1.0)))
        if target_size <= 0:
            raise ValueError("Calibration target_size must be positive.")
        mapped *= float(output_size) / target_size
    return mapped


def correct_point(
    x: float,
    y: float,
    calibration: dict[str, Any],
    output_size: float | None = DEFAULT_OUTPUT_SIZE,
) -> tuple[float, float]:
    """Convert one raw camera-frame point to corrected square coordinates."""
    mapped = correct_points(np.array([[x, y]], dtype=float), calibration, output_size=output_size)
    return float(mapped[0, 0]), float(mapped[0, 1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw pixel coordinates to corrected arena coordinates.")
    parser.add_argument("calibration_json", type=Path)
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument(
        "--output-size",
        type=float,
        default=DEFAULT_OUTPUT_SIZE,
        help="Corrected square side length. Default: 520. Use 1 for normalized output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    calibration = load_calibration(args.calibration_json)
    x_corr, y_corr = correct_point(args.x, args.y, calibration, output_size=args.output_size)
    print(f"{x_corr:.10g},{y_corr:.10g}")


if __name__ == "__main__":
    main()
