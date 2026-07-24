"""Create an oriented sinusoidal grating and save it as an SVG.

Example
-------
python plot/plot_grating.py --orientation 45 --output grating_45.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_grating(
    orientation_deg: float,
    *,
    size: int = 512,
    spatial_frequency: float = 8.0,
    phase_deg: float = 0.0,
    contrast: float = 1.0,
    mean_luminance: float = 0.5,
) -> np.ndarray:
    """Return a grayscale sinusoidal grating image.

    Parameters
    ----------
    orientation_deg:
        Orientation of the grating bars in degrees. 0 gives vertical bars,
        90 gives horizontal bars.
    size:
        Width and height of the square output image in pixels.
    spatial_frequency:
        Number of sinusoidal cycles across the image.
    phase_deg:
        Phase offset in degrees.
    contrast:
        Michelson-like contrast scale in [0, 1].
    mean_luminance:
        Center luminance in [0, 1].
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if spatial_frequency <= 0:
        raise ValueError("spatial_frequency must be positive")
    if not 0 <= contrast <= 1:
        raise ValueError("contrast must be in [0, 1]")
    if not 0 <= mean_luminance <= 1:
        raise ValueError("mean_luminance must be in [0, 1]")

    coords = np.linspace(-0.5, 0.5, size, endpoint=False)
    x, y = np.meshgrid(coords, coords)

    theta = np.deg2rad(orientation_deg)
    phase = np.deg2rad(phase_deg)

    # The sinusoid changes along the normal to the bars.
    normal_projection = x * np.cos(theta) + y * np.sin(theta)
    grating = mean_luminance + 0.5 * contrast * np.sin(
        2 * np.pi * spatial_frequency * normal_projection + phase
    )
    return np.clip(grating, 0.0, 1.0)


def save_grating_svg(
    output_path: str | Path,
    orientation_deg: float,
    *,
    size: int = 512,
    spatial_frequency: float = 8.0,
    phase_deg: float = 0.0,
    contrast: float = 1.0,
    mean_luminance: float = 0.5,
) -> Path:
    """Create a grating and save it to an SVG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grating = make_grating(
        orientation_deg,
        size=size,
        spatial_frequency=spatial_frequency,
        phase_deg=phase_deg,
        contrast=contrast,
        mean_luminance=mean_luminance,
    )

    fig, ax = plt.subplots(figsize=(4, 4), dpi=size / 4)
    ax.imshow(grating, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an oriented sinusoidal grating and save it as SVG."
    )
    parser.add_argument(
        "--orientation",
        "-o",
        type=float,
        required=True,
        help="Orientation of the bars in degrees. 0 is vertical, 90 is horizontal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("grating.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Square image size in pixels.",
    )
    parser.add_argument(
        "--spatial-frequency",
        "--cycles",
        type=float,
        default=8.0,
        help="Number of grating cycles across the image.",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Phase offset in degrees.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast in [0, 1].",
    )
    parser.add_argument(
        "--mean-luminance",
        type=float,
        default=0.5,
        help="Mean luminance in [0, 1].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = save_grating_svg(
        args.output,
        args.orientation,
        size=args.size,
        spatial_frequency=args.spatial_frequency,
        phase_deg=args.phase,
        contrast=args.contrast,
        mean_luminance=args.mean_luminance,
    )
    print(f"Saved grating to {output_path}")


if __name__ == "__main__":
    main()
