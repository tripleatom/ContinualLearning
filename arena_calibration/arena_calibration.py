#!/usr/bin/env python
"""Interactive arena calibration for square freely roaming arenas.

The calibration maps raw camera pixels to a corrected square arena coordinate
system. It suggests 30 points per arena edge from the first frame, lets the user
edit those points in a matplotlib GUI, then fits a polynomial transform from
curved camera-frame edge points to a square target coordinate system.

Default target units are normalized arena coordinates: x/y in [0, 1]. If the
arena side length is known, pass --arena-side-length and --target-units to save
coordinates directly in physical units such as cm.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, UnivariateSpline
from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
from skimage import draw
from skimage import color, measure, morphology, transform, util


SIDES = ["top", "right", "bottom", "left"]
SIDE_COLORS = {"top": "tab:blue", "right": "tab:orange", "bottom": "tab:green", "left": "tab:red"}
SIDE_CORNER_INDEX = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}


@dataclass
class FitResult:
    order: int
    source_width: int
    source_height: int
    target_size: float
    terms: list[list[int]]
    forward_coeff_x: list[float]
    forward_coeff_y: list[float]
    inverse_coeff_x: list[float]
    inverse_coeff_y: list[float]
    post_homography: list[list[float]]


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    reader = imageio.get_reader(video_path)
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


def sample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    if len(points) == 0:
        return points
    if len(points) == 1:
        return np.repeat(points, n_points, axis=0)
    diffs = np.diff(points, axis=0)
    dist = np.concatenate([[0.0], np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))])
    if dist[-1] == 0:
        return np.repeat(points[:1], n_points, axis=0)
    target = np.linspace(0, dist[-1], n_points)
    x = np.interp(target, dist, points[:, 0])
    y = np.interp(target, dist, points[:, 1])
    return np.column_stack([x, y])


def fallback_edge_points(image: np.ndarray, points_per_edge: int) -> dict[str, np.ndarray]:
    height, width = image.shape[:2]
    margin_x = width * 0.08
    margin_y = height * 0.08
    return {
        "top": np.column_stack([np.linspace(margin_x, width - margin_x, points_per_edge), np.full(points_per_edge, margin_y)]),
        "right": np.column_stack([np.full(points_per_edge, width - margin_x), np.linspace(margin_y, height - margin_y, points_per_edge)]),
        "bottom": np.column_stack([np.linspace(margin_x, width - margin_x, points_per_edge), np.full(points_per_edge, height - margin_y)]),
        "left": np.column_stack([np.full(points_per_edge, margin_x), np.linspace(margin_y, height - margin_y, points_per_edge)]),
    }


class CornerEditor:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.corners: list[list[float]] = []
        self.selected: int | None = None
        self.confirmed = False
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.ax.imshow(image)
        self.scatter = None
        self.lines = None
        self.text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        self.redraw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def redraw(self) -> None:
        if self.scatter is not None:
            self.scatter.remove()
        if self.lines is not None:
            self.lines.remove()
        pts = np.asarray(self.corners, dtype=float) if self.corners else np.empty((0, 2))
        self.scatter = self.ax.scatter(
            pts[:, 0] if len(pts) else [],
            pts[:, 1] if len(pts) else [],
            c="yellow",
            s=60,
            edgecolors="black",
            zorder=3,
        )
        if len(pts) >= 2:
            closed = pts if len(pts) < 4 else np.vstack([pts, pts[0]])
            (self.lines,) = self.ax.plot(closed[:, 0], closed[:, 1], color="yellow", lw=1.5, zorder=2)
        else:
            (self.lines,) = self.ax.plot([], [])
        labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
        self.text.set_text(
            "Click rough corners in order: top-left, top-right, bottom-right, bottom-left.\n"
            "Drag to adjust. Backspace removes last. Enter confirms. q cancels.\n"
            f"Next: {labels[len(self.corners)] if len(self.corners) < 4 else 'press enter'}"
        )
        self.fig.canvas.draw_idle()

    def nearest_corner(self, x: float, y: float, max_dist: float = 15.0) -> int | None:
        if not self.corners:
            return None
        pts = np.asarray(self.corners)
        d = np.sqrt(np.sum((pts - np.array([x, y])) ** 2, axis=1))
        idx = int(np.argmin(d))
        return idx if d[idx] < max_dist else None

    def on_press(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        nearest = self.nearest_corner(float(event.xdata), float(event.ydata))
        if nearest is not None:
            self.selected = nearest
        elif event.button == 1 and len(self.corners) < 4:
            self.corners.append([float(event.xdata), float(event.ydata)])
            self.redraw()

    def on_motion(self, event: Any) -> None:
        if self.selected is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.corners[self.selected] = [float(event.xdata), float(event.ydata)]
        self.redraw()

    def on_release(self, event: Any) -> None:
        self.selected = None

    def on_key(self, event: Any) -> None:
        if event.key == "backspace" and self.corners:
            self.corners.pop()
            self.redraw()
        elif event.key == "enter":
            if len(self.corners) != 4:
                print("Need four corners before confirming.")
                return
            self.confirmed = True
            plt.close(self.fig)
        elif event.key == "q":
            self.confirmed = False
            plt.close(self.fig)

    def show(self) -> np.ndarray:
        plt.show()
        if not self.confirmed:
            raise RuntimeError("Corner selection was cancelled.")
        return np.asarray(self.corners, dtype=float)


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    labeled, n_labels = ndi.label(mask)
    if n_labels == 0:
        return None
    sizes = ndi.sum(mask, labeled, index=np.arange(1, n_labels + 1))
    label = int(np.argmax(sizes) + 1)
    return labeled == label


def suggest_floor_mask(image: np.ndarray) -> np.ndarray | None:
    rgb = util.img_as_float(image[..., :3])
    lab = color.rgb2lab(rgb)
    pixels = lab.reshape(-1, 3)
    sample = pixels[:: max(1, len(pixels) // 10000)]
    try:
        centroids, _ = kmeans2(sample, 4, minit="points", iter=30)
    except Exception:
        return None
    d = np.sum((pixels[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d, axis=1).reshape(image.shape[:2])

    height, width = image.shape[:2]
    cy, cx = height / 2, width / 2
    best_mask = None
    best_score = -np.inf
    for label in range(len(centroids)):
        mask = labels == label
        mask = morphology.remove_small_objects(mask, min_size=max(50, int(mask.size * 0.002)))
        mask = morphology.binary_closing(mask, morphology.disk(5))
        mask = ndi.binary_fill_holes(mask)
        comp = largest_component(mask)
        if comp is None:
            continue
        area = float(np.mean(comp))
        rows, cols = np.nonzero(comp)
        if len(rows) == 0:
            continue
        center_distance = math.hypot(float(np.mean(cols) - cx) / width, float(np.mean(rows) - cy) / height)
        contains_center = bool(comp[int(round(cy)), int(round(cx))])
        plausible_area = 0.15 <= area <= 0.95
        score = area - center_distance + (0.5 if contains_center else 0.0) + (0.25 if plausible_area else -0.25)
        if score > best_score:
            best_score = score
            best_mask = comp
    return best_mask


def segment_between(contour: np.ndarray, i: int, j: int) -> np.ndarray:
    if i <= j:
        path1 = contour[i : j + 1]
        path2 = np.vstack([contour[j:], contour[: i + 1]])
    else:
        path1 = np.vstack([contour[i:], contour[: j + 1]])
        path2 = contour[j : i + 1]
    length1 = np.sum(np.sqrt(np.sum(np.diff(path1, axis=0) ** 2, axis=1)))
    length2 = np.sum(np.sqrt(np.sum(np.diff(path2, axis=0) ** 2, axis=1)))
    return path1 if length1 <= length2 else path2


def suggest_edge_points(image: np.ndarray, points_per_edge: int) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    mask = suggest_floor_mask(image)
    if mask is None:
        return fallback_edge_points(image, points_per_edge), None
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return fallback_edge_points(image, points_per_edge), mask
    contour_yx = max(contours, key=len)
    contour = np.column_stack([contour_yx[:, 1], contour_yx[:, 0]])
    x, y = contour[:, 0], contour[:, 1]
    idx = {
        "top_left": int(np.argmin(x + y)),
        "top_right": int(np.argmax(x - y)),
        "bottom_right": int(np.argmax(x + y)),
        "bottom_left": int(np.argmin(y - x)),
    }
    edges = {
        "top": segment_between(contour, idx["top_left"], idx["top_right"]),
        "right": segment_between(contour, idx["top_right"], idx["bottom_right"]),
        "bottom": segment_between(contour, idx["bottom_left"], idx["bottom_right"]),
        "left": segment_between(contour, idx["top_left"], idx["bottom_left"]),
    }
    edges["top"] = edges["top"][np.argsort(edges["top"][:, 0])]
    edges["bottom"] = edges["bottom"][np.argsort(edges["bottom"][:, 0])]
    edges["right"] = edges["right"][np.argsort(edges["right"][:, 1])]
    edges["left"] = edges["left"][np.argsort(edges["left"][:, 1])]
    return {side: sample_polyline(edges[side], points_per_edge) for side in SIDES}, mask


def preprocess_gray(image: np.ndarray, sigma: float) -> np.ndarray:
    rgb = util.img_as_float(image[..., :3])
    gray = color.rgb2gray(rgb)
    p1, p99 = np.percentile(gray, [1, 99])
    if p99 > p1:
        gray = np.clip((gray - p1) / (p99 - p1), 0, 1)
    return ndi.gaussian_filter(gray, sigma=sigma)


def polygon_inside_mask(shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    rr, cc = draw.polygon(corners[:, 1], corners[:, 0], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coords = np.vstack([y, x])
    return ndi.map_coordinates(image, coords, order=1, mode="nearest")


def normal_toward_inside(point: np.ndarray, edge_vec: np.ndarray, inside_mask: np.ndarray) -> np.ndarray:
    normal = np.array([-edge_vec[1], edge_vec[0]], dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0.0, 1.0])
    normal /= norm
    height, width = inside_mask.shape
    p1 = point + normal * 5
    p2 = point - normal * 5
    def inside(p: np.ndarray) -> bool:
        x = int(np.clip(round(p[0]), 0, width - 1))
        y = int(np.clip(round(p[1]), 0, height - 1))
        return bool(inside_mask[y, x])
    if inside(p2) and not inside(p1):
        normal = -normal
    elif inside(p1) == inside(p2):
        centroid = np.mean(np.column_stack(np.nonzero(inside_mask))[:, ::-1], axis=0)
        if np.dot(centroid - point, normal) < 0:
            normal = -normal
    return normal


def detect_profile_edge(
    gray: np.ndarray,
    center: np.ndarray,
    normal_inside: np.ndarray,
    tangent: np.ndarray,
    search_radius: float,
    profile_half_width: int,
) -> np.ndarray:
    offsets = np.arange(-search_radius, search_radius + 1, 1.0)
    profiles = []
    for tangent_offset in range(-profile_half_width, profile_half_width + 1):
        points = center + offsets[:, None] * normal_inside + tangent_offset * tangent
        profiles.append(bilinear_sample(gray, points[:, 0], points[:, 1]))
    profile = np.mean(np.vstack(profiles), axis=0)
    gradient = np.abs(np.gradient(profile))
    inside_vals = []
    outside_vals = []
    for i in range(len(profile)):
        inside_slice = profile[i + 1 : min(len(profile), i + 8)]
        outside_slice = profile[max(0, i - 7) : i]
        inside_vals.append(float(np.mean(inside_slice)) if len(inside_slice) else float(profile[i]))
        outside_vals.append(float(np.mean(outside_slice)) if len(outside_slice) else float(profile[i]))
    inside_mean = np.asarray(inside_vals)
    outside_mean = np.asarray(outside_vals)
    contrast = inside_mean - outside_mean
    distance_penalty = np.abs(offsets) / max(1.0, search_radius)
    score = gradient + 0.75 * np.maximum(contrast, 0) - 0.15 * distance_penalty
    edge_buffer = max(2, int(search_radius * 0.15))
    score[:edge_buffer] = -np.inf
    score[-edge_buffer:] = -np.inf
    best = int(np.argmax(score))
    return center + offsets[best] * normal_inside


def smooth_edge_points(points: np.ndarray, side: str, smoothing: float, outlier_threshold: float) -> np.ndarray:
    if len(points) < 4:
        return points
    pts = points[np.argsort(points[:, 0] if side in {"top", "bottom"} else points[:, 1])]
    t = np.linspace(0, 1, len(pts))
    sx = UnivariateSpline(t, pts[:, 0], s=smoothing * len(pts), k=min(3, len(pts) - 1))
    sy = UnivariateSpline(t, pts[:, 1], s=smoothing * len(pts), k=min(3, len(pts) - 1))
    fitted = np.column_stack([sx(t), sy(t)])
    residual = np.sqrt(np.sum((pts - fitted) ** 2, axis=1))
    keep = residual <= outlier_threshold
    if np.sum(keep) >= 4 and np.any(~keep):
        sx = UnivariateSpline(t[keep], pts[keep, 0], s=smoothing * np.sum(keep), k=min(3, np.sum(keep) - 1))
        sy = UnivariateSpline(t[keep], pts[keep, 1], s=smoothing * np.sum(keep), k=min(3, np.sum(keep) - 1))
        fitted = np.column_stack([sx(t), sy(t)])
    return fitted


def suggest_edge_points_from_corners(
    image: np.ndarray,
    corners: np.ndarray,
    points_per_edge: int,
    search_radius: float,
    profile_half_width: int,
    gaussian_sigma: float,
    spline_smoothing: float,
    outlier_threshold: float,
) -> dict[str, np.ndarray]:
    gray = preprocess_gray(image, gaussian_sigma)
    inside_mask = polygon_inside_mask(gray.shape, corners)
    suggestions: dict[str, np.ndarray] = {}
    for side in SIDES:
        ia, ib = SIDE_CORNER_INDEX[side]
        a = corners[ia]
        b = corners[ib]
        edge_vec = b - a
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0:
            suggestions[side] = np.repeat(a[None, :], points_per_edge, axis=0)
            continue
        tangent = edge_vec / edge_len
        raw_points = []
        for t in np.linspace(0, 1, points_per_edge):
            center = a * (1 - t) + b * t
            normal = normal_toward_inside(center, edge_vec, inside_mask)
            raw_points.append(
                detect_profile_edge(gray, center, normal, tangent, search_radius, profile_half_width)
            )
        raw = np.asarray(raw_points)
        smooth = smooth_edge_points(raw, side, spline_smoothing, outlier_threshold)
        suggestions[side] = smooth
    return sort_edge_points(suggestions)


class EdgePointEditor:
    def __init__(self, image: np.ndarray, points: dict[str, np.ndarray], mask: np.ndarray | None = None):
        self.image = image
        self.points = {side: points[side].copy() for side in SIDES}
        self.active_side = "top"
        self.selected: tuple[str, int] | None = None
        self.confirmed = False
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.ax.imshow(image)
        if mask is not None:
            self.ax.contour(mask, levels=[0.5], colors="white", linewidths=0.7, alpha=0.6)
        self.scatters: dict[str, Any] = {}
        self.text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        self.redraw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def redraw(self) -> None:
        for scatter in self.scatters.values():
            scatter.remove()
        self.scatters = {}
        for side in SIDES:
            pts = self.points[side]
            size = 42 if side == self.active_side else 24
            self.scatters[side] = self.ax.scatter(
                pts[:, 0] if len(pts) else [],
                pts[:, 1] if len(pts) else [],
                s=size,
                c=SIDE_COLORS[side],
                label=side,
                edgecolors="black",
                linewidths=0.5,
            )
        self.text.set_text(
            "Keys: 1/2/3/4 active edge, a add, right-click delete, drag move, enter save, q cancel\n"
            f"Active: {self.active_side}. Points: "
            + ", ".join(f"{side}={len(self.points[side])}" for side in SIDES)
        )
        self.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        self.fig.subplots_adjust(right=0.82)
        self.fig.canvas.draw_idle()

    def nearest_point(self, x: float, y: float, max_dist: float = 12.0) -> tuple[str, int] | None:
        best = None
        best_dist = max_dist
        for side, pts in self.points.items():
            if len(pts) == 0:
                continue
            d = np.sqrt(np.sum((pts - np.array([x, y])) ** 2, axis=1))
            idx = int(np.argmin(d))
            if d[idx] < best_dist:
                best_dist = float(d[idx])
                best = (side, idx)
        return best

    def on_press(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        nearest = self.nearest_point(float(event.xdata), float(event.ydata))
        if event.button == 3 and nearest is not None:
            side, idx = nearest
            self.points[side] = np.delete(self.points[side], idx, axis=0)
            self.redraw()
        elif event.button == 1:
            self.selected = nearest

    def on_motion(self, event: Any) -> None:
        if self.selected is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        side, idx = self.selected
        self.points[side][idx] = [float(event.xdata), float(event.ydata)]
        self.redraw()

    def on_release(self, event: Any) -> None:
        self.selected = None

    def on_key(self, event: Any) -> None:
        if event.key in {"1", "2", "3", "4"}:
            self.active_side = SIDES[int(event.key) - 1]
            self.redraw()
        elif event.key == "a":
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            point = np.array([np.mean(xlim), np.mean(ylim)])
            self.points[self.active_side] = np.vstack([self.points[self.active_side], point])
            self.redraw()
        elif event.key == "enter":
            self.confirmed = True
            plt.close(self.fig)
        elif event.key == "q":
            self.confirmed = False
            plt.close(self.fig)

    def show(self) -> dict[str, np.ndarray]:
        plt.show()
        if not self.confirmed:
            raise RuntimeError("Calibration point editing was cancelled.")
        return self.points


def sort_edge_points(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sorted_points = {}
    for side, pts in points.items():
        pts = np.asarray(pts, dtype=float)
        if side in {"top", "bottom"}:
            pts = pts[np.argsort(pts[:, 0])]
        else:
            pts = pts[np.argsort(pts[:, 1])]
        sorted_points[side] = pts
    return sorted_points


def target_points_for_edges(points: dict[str, np.ndarray], target_size: float) -> dict[str, np.ndarray]:
    def arc_fraction(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.array([])
        if len(pts) == 1:
            return np.array([0.0])
        diffs = np.diff(pts, axis=0)
        dist = np.concatenate([[0.0], np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))])
        if dist[-1] == 0:
            return np.linspace(0, 1, len(pts))
        return dist / dist[-1]

    targets = {}
    for side, pts in points.items():
        frac = arc_fraction(np.asarray(pts, dtype=float))
        if side == "top":
            targets[side] = np.column_stack([frac * target_size, np.zeros(len(frac))])
        elif side == "bottom":
            targets[side] = np.column_stack([frac * target_size, np.full(len(frac), target_size)])
        elif side == "left":
            targets[side] = np.column_stack([np.zeros(len(frac)), frac * target_size])
        elif side == "right":
            targets[side] = np.column_stack([np.full(len(frac), target_size), frac * target_size])
    return targets


def polynomial_terms(order: int) -> list[tuple[int, int]]:
    terms = []
    for degree in range(order + 1):
        for px in range(degree + 1):
            terms.append((px, degree - px))
    return terms


def basis(x: np.ndarray, y: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    return np.column_stack([(x**px) * (y**py) for px, py in terms])


def fit_polynomial_transform(
    source: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    target_size: float,
    order: int,
    side_counts: list[int],
) -> FitResult:
    terms = polynomial_terms(order)
    sx = source[:, 0] / max(1, width - 1)
    sy = source[:, 1] / max(1, height - 1)
    design = basis(sx, sy, terms)
    coeff_x, *_ = np.linalg.lstsq(design, target[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, target[:, 1], rcond=None)

    base_mapped = np.column_stack([design @ coeff_x, design @ coeff_y])
    post = fit_post_homography_from_edge_lines(base_mapped, target_size, contiguous_side_indices(side_counts))
    final_target = apply_post_homography(base_mapped, post.params)
    inverse_target, inverse_source = dense_forward_grid_for_inverse(
        width,
        height,
        target_size,
        terms,
        coeff_x,
        coeff_y,
        post.params,
    )
    inverse_target = np.vstack([inverse_target, final_target])
    inverse_source = np.vstack([inverse_source, source])
    tx = inverse_target[:, 0] / target_size
    ty = inverse_target[:, 1] / target_size
    inv_design = basis(tx, ty, terms)
    inv_coeff_x, *_ = np.linalg.lstsq(inv_design, inverse_source[:, 0], rcond=None)
    inv_coeff_y, *_ = np.linalg.lstsq(inv_design, inverse_source[:, 1], rcond=None)
    return FitResult(
        order=order,
        source_width=width,
        source_height=height,
        target_size=target_size,
        terms=[[int(a), int(b)] for a, b in terms],
        forward_coeff_x=coeff_x.tolist(),
        forward_coeff_y=coeff_y.tolist(),
        inverse_coeff_x=inv_coeff_x.tolist(),
        inverse_coeff_y=inv_coeff_y.tolist(),
        post_homography=post.params.tolist(),
    )


def apply_post_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homog = np.column_stack([points, np.ones(len(points))]) @ homography.T
    return homog[:, :2] / homog[:, 2:3]


def dense_forward_grid_for_inverse(
    width: int,
    height: int,
    target_size: float,
    terms: list[tuple[int, int]],
    coeff_x: np.ndarray,
    coeff_y: np.ndarray,
    post_homography: np.ndarray,
    grid_n: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate dense final-corrected -> raw correspondences for image warping."""
    xs = np.linspace(0, width - 1, grid_n)
    ys = np.linspace(0, height - 1, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    source = np.column_stack([xx.ravel(), yy.ravel()])
    sx = source[:, 0] / max(1, width - 1)
    sy = source[:, 1] / max(1, height - 1)
    design = basis(sx, sy, terms)
    base = np.column_stack([design @ coeff_x, design @ coeff_y])
    final = apply_post_homography(base, post_homography)
    finite = np.isfinite(final).all(axis=1)
    return final[finite], source[finite]


def apply_polynomial_only(points: np.ndarray, fit: FitResult) -> np.ndarray:
    terms = [(a, b) for a, b in fit.terms]
    x = points[:, 0] / max(1, fit.source_width - 1)
    y = points[:, 1] / max(1, fit.source_height - 1)
    design = basis(x, y, terms)
    return np.column_stack([design @ np.asarray(fit.forward_coeff_x), design @ np.asarray(fit.forward_coeff_y)])


def contiguous_side_indices(side_counts: list[int]) -> list[np.ndarray]:
    indices = []
    start = 0
    for count in side_counts:
        stop = start + count
        indices.append(np.arange(start, stop))
        start = stop
    return indices


def fit_line(points: np.ndarray) -> np.ndarray:
    """Fit ax + by + c = 0 using total least squares."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    c = -float(np.dot(normal, centroid))
    line = np.array([normal[0], normal[1], c], dtype=float)
    norm = np.linalg.norm(line[:2])
    return line / norm if norm else line


def line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray:
    cross = np.cross(line_a, line_b)
    if abs(cross[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return cross[:2] / cross[2]


def fit_post_homography_from_edge_lines(
    base_points: np.ndarray,
    target_size: float,
    side_indices: list[np.ndarray],
) -> transform.ProjectiveTransform:
    lines = []
    for idx in side_indices:
        pts = base_points[idx]
        if len(pts) < 2:
            post = transform.ProjectiveTransform()
            post.params = np.eye(3)
            return post
        lines.append(fit_line(pts))
    top, right, bottom, left = lines
    corners = np.vstack(
        [
            line_intersection(top, left),
            line_intersection(top, right),
            line_intersection(bottom, right),
            line_intersection(bottom, left),
        ]
    )
    target_corners = np.array(
        [[0.0, 0.0], [target_size, 0.0], [target_size, target_size], [0.0, target_size]],
        dtype=float,
    )
    post = transform.ProjectiveTransform()
    if not np.all(np.isfinite(corners)) or not post.estimate(corners, target_corners):
        post.params = np.eye(3)
    return post


def apply_forward(points: np.ndarray, fit: FitResult) -> np.ndarray:
    terms = [(a, b) for a, b in fit.terms]
    x = points[:, 0] / max(1, fit.source_width - 1)
    y = points[:, 1] / max(1, fit.source_height - 1)
    design = basis(x, y, terms)
    base = np.column_stack([design @ np.asarray(fit.forward_coeff_x), design @ np.asarray(fit.forward_coeff_y)])
    h = np.asarray(fit.post_homography, dtype=float)
    return apply_post_homography(base, h)


def apply_forward_xy(x: np.ndarray, y: np.ndarray, fit: FitResult) -> tuple[np.ndarray, np.ndarray]:
    pts = np.column_stack([x.ravel(), y.ravel()])
    mapped = apply_forward(pts, fit)
    return mapped[:, 0].reshape(x.shape), mapped[:, 1].reshape(y.shape)


def apply_inverse(target_points: np.ndarray, fit: FitResult) -> np.ndarray:
    terms = [(a, b) for a, b in fit.terms]
    x = target_points[:, 0] / fit.target_size
    y = target_points[:, 1] / fit.target_size
    design = basis(x, y, terms)
    return np.column_stack([design @ np.asarray(fit.inverse_coeff_x), design @ np.asarray(fit.inverse_coeff_y)])


def inverse_grid_interpolated(target_points: np.ndarray, fit: FitResult, grid_n: int = 90) -> np.ndarray:
    terms = [(a, b) for a, b in fit.terms]
    inverse_target, inverse_source = dense_forward_grid_for_inverse(
        fit.source_width,
        fit.source_height,
        fit.target_size,
        terms,
        np.asarray(fit.forward_coeff_x),
        np.asarray(fit.forward_coeff_y),
        np.asarray(fit.post_homography),
        grid_n=grid_n,
    )
    lin_x = LinearNDInterpolator(inverse_target, inverse_source[:, 0], fill_value=np.nan)
    lin_y = LinearNDInterpolator(inverse_target, inverse_source[:, 1], fill_value=np.nan)
    src_x = lin_x(target_points)
    src_y = lin_y(target_points)
    missing = ~np.isfinite(src_x) | ~np.isfinite(src_y)
    if np.any(missing):
        near_x = NearestNDInterpolator(inverse_target, inverse_source[:, 0])
        near_y = NearestNDInterpolator(inverse_target, inverse_source[:, 1])
        src_x[missing] = near_x(target_points[missing])
        src_y[missing] = near_y(target_points[missing])
    return np.column_stack([src_x, src_y])


def inverse_numerical_forward(
    target_points: np.ndarray,
    fit: FitResult,
    initial: np.ndarray | None = None,
    iterations: int = 8,
) -> np.ndarray:
    if initial is None:
        source = inverse_grid_interpolated(target_points, fit, grid_n=70)
    else:
        source = initial.copy()
    source[:, 0] = np.clip(source[:, 0], 0, fit.source_width - 1)
    source[:, 1] = np.clip(source[:, 1], 0, fit.source_height - 1)
    eps = 1e-3
    for _ in range(iterations):
        mapped = apply_forward(source, fit)
        residual = mapped - target_points
        px = source.copy()
        mx = source.copy()
        py = source.copy()
        my = source.copy()
        px[:, 0] += eps
        mx[:, 0] -= eps
        py[:, 1] += eps
        my[:, 1] -= eps
        dfdx = (apply_forward(px, fit) - apply_forward(mx, fit)) / (2 * eps)
        dfdy = (apply_forward(py, fit) - apply_forward(my, fit)) / (2 * eps)
        a = dfdx[:, 0]
        b = dfdy[:, 0]
        c = dfdx[:, 1]
        d = dfdy[:, 1]
        det = a * d - b * c
        good = np.isfinite(det) & (np.abs(det) > 1e-12)
        delta = np.zeros_like(source)
        delta[good, 0] = (d[good] * residual[good, 0] - b[good] * residual[good, 1]) / det[good]
        delta[good, 1] = (-c[good] * residual[good, 0] + a[good] * residual[good, 1]) / det[good]
        delta = np.clip(delta, -25, 25)
        source[good] -= delta[good]
        source[:, 0] = np.clip(source[:, 0], -0.25 * fit.source_width, 1.25 * fit.source_width)
        source[:, 1] = np.clip(source[:, 1], -0.25 * fit.source_height, 1.25 * fit.source_height)
        if np.nanmedian(np.sqrt(np.sum(delta[good] ** 2, axis=1))) < 1e-3:
            break
    return source


def edge_qc(edge_points: dict[str, np.ndarray], fit: FitResult) -> dict[str, Any]:
    qc: dict[str, Any] = {}
    all_errors = []
    for side, pts in edge_points.items():
        mapped = apply_forward(pts, fit)
        if side == "top":
            err = np.abs(mapped[:, 1] - 0)
        elif side == "bottom":
            err = np.abs(mapped[:, 1] - fit.target_size)
        elif side == "left":
            err = np.abs(mapped[:, 0] - 0)
        else:
            err = np.abs(mapped[:, 0] - fit.target_size)
        qc[f"{side}_edge_error_mean"] = float(np.mean(err))
        qc[f"{side}_edge_error_max"] = float(np.max(err))
        all_errors.append(err)
    all_err = np.concatenate(all_errors) if all_errors else np.array([np.nan])
    qc["edge_error_mean"] = float(np.nanmean(all_err))
    qc["edge_error_max"] = float(np.nanmax(all_err))
    return qc


def save_transform_debug(
    edge_points: dict[str, np.ndarray],
    target_points: dict[str, np.ndarray],
    fit: FitResult,
    output_dir: Path,
    name: str,
) -> dict[str, str]:
    rows = []
    line_rows = []
    for side in SIDES:
        raw = np.asarray(edge_points[side], dtype=float)
        target = np.asarray(target_points[side], dtype=float)
        poly = apply_polynomial_only(raw, fit)
        final = apply_forward(raw, fit)
        for idx in range(len(raw)):
            rows.append(
                {
                    "side": side,
                    "index": idx,
                    "raw_x": raw[idx, 0],
                    "raw_y": raw[idx, 1],
                    "target_x": target[idx, 0],
                    "target_y": target[idx, 1],
                    "poly_x": poly[idx, 0],
                    "poly_y": poly[idx, 1],
                    "final_x": final[idx, 0],
                    "final_y": final[idx, 1],
                    "final_minus_target_x": final[idx, 0] - target[idx, 0],
                    "final_minus_target_y": final[idx, 1] - target[idx, 1],
                }
            )
        for stage, pts in [("polynomial_only", poly), ("final", final), ("target", target)]:
            if len(pts) >= 2:
                line = fit_line(pts)
                line_rows.append(
                    {
                        "side": side,
                        "stage": stage,
                        "line_a": line[0],
                        "line_b": line[1],
                        "line_c": line[2],
                        "angle_deg": math.degrees(math.atan2(-line[0], line[1])),
                    }
                )

    point_csv = output_dir / f"{name}_transform_debug_points.csv"
    line_csv = output_dir / f"{name}_transform_debug_lines.csv"
    summary_json = output_dir / f"{name}_transform_debug_summary.json"
    df = pd.DataFrame(rows)
    line_df = pd.DataFrame(line_rows)
    df.to_csv(point_csv, index=False)
    line_df.to_csv(line_csv, index=False)

    summary: dict[str, Any] = {
        "post_homography": fit.post_homography,
        "point_csv": str(point_csv),
        "line_csv": str(line_csv),
        "final_residual_abs_mean_x": float(np.mean(np.abs(df["final_minus_target_x"]))),
        "final_residual_abs_mean_y": float(np.mean(np.abs(df["final_minus_target_y"]))),
        "final_residual_abs_max_x": float(np.max(np.abs(df["final_minus_target_x"]))),
        "final_residual_abs_max_y": float(np.max(np.abs(df["final_minus_target_y"]))),
        "line_angles_deg": line_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "transform_debug_points": str(point_csv),
        "transform_debug_lines": str(line_csv),
        "transform_debug_summary": str(summary_json),
    }


def warp_frame(image: np.ndarray, fit: FitResult, output_pixels: int) -> np.ndarray:
    def inverse_map(coords: np.ndarray) -> np.ndarray:
        row = coords[:, 0]
        col = coords[:, 1]
        target = np.column_stack([col / max(1, output_pixels - 1) * fit.target_size, row / max(1, output_pixels - 1) * fit.target_size])
        source = apply_inverse(target, fit)
        return np.column_stack([source[:, 1], source[:, 0]])

    warped = transform.warp(image, inverse_map=inverse_map, output_shape=(output_pixels, output_pixels), preserve_range=True)
    return np.clip(warped, 0, 255).astype(np.uint8)


def transformed_frame_extent(fit: FitResult, samples_per_edge: int = 200) -> tuple[float, float, float, float]:
    width = fit.source_width
    height = fit.source_height
    top = np.column_stack([np.linspace(0, width - 1, samples_per_edge), np.zeros(samples_per_edge)])
    right = np.column_stack([np.full(samples_per_edge, width - 1), np.linspace(0, height - 1, samples_per_edge)])
    bottom = np.column_stack([np.linspace(width - 1, 0, samples_per_edge), np.full(samples_per_edge, height - 1)])
    left = np.column_stack([np.zeros(samples_per_edge), np.linspace(height - 1, 0, samples_per_edge)])
    mapped = apply_forward(np.vstack([top, right, bottom, left]), fit)
    pad = 0.03 * fit.target_size
    return (
        float(np.nanmin(mapped[:, 0]) - pad),
        float(np.nanmax(mapped[:, 0]) + pad),
        float(np.nanmin(mapped[:, 1]) - pad),
        float(np.nanmax(mapped[:, 1]) + pad),
    )


def output_grid_for_extent(
    extent: tuple[float, float, float, float],
    max_output_pixels: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float, float, float]]:
    x_min, x_max, y_min, y_max = extent
    width_units = x_max - x_min
    height_units = y_max - y_min
    if width_units <= 0 or height_units <= 0:
        raise ValueError("Invalid transformed frame extent.")
    if width_units >= height_units:
        out_w = max_output_pixels
        out_h = max(1, int(round(max_output_pixels * height_units / width_units)))
    else:
        out_h = max_output_pixels
        out_w = max(1, int(round(max_output_pixels * width_units / height_units)))
    rows, cols = np.mgrid[0:out_h, 0:out_w]
    target_x = x_min + cols.ravel() / max(1, out_w - 1) * width_units
    target_y = y_min + rows.ravel() / max(1, out_h - 1) * height_units
    return np.column_stack([target_x, target_y]), (out_h, out_w), extent


def sample_image_at_source(image: np.ndarray, source: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = output_shape
    sampled_channels = []
    for channel in range(image.shape[2] if image.ndim == 3 else 1):
        plane = image[..., channel] if image.ndim == 3 else image
        vals = ndi.map_coordinates(
            plane,
            [source[:, 1], source[:, 0]],
            order=1,
            mode="constant",
            cval=0,
        )
        sampled_channels.append(vals.reshape(out_h, out_w))
    if image.ndim == 3:
        return np.stack(sampled_channels, axis=2).astype(np.uint8)
    return sampled_channels[0].astype(np.uint8)


def warp_full_frame_with_inverse(
    image: np.ndarray,
    fit: FitResult,
    max_output_pixels: int,
    method: str,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    extent = transformed_frame_extent(fit)
    target_points, output_shape, extent = output_grid_for_extent(extent, max_output_pixels)
    if method == "inverse_polynomial":
        source = apply_inverse(target_points, fit)
    elif method == "grid_interpolation":
        source = inverse_grid_interpolated(target_points, fit)
    elif method == "numerical_forward":
        initial = inverse_grid_interpolated(target_points, fit)
        source = inverse_numerical_forward(target_points, fit, initial=initial)
    else:
        raise ValueError(f"Unknown inverse method: {method}")
    return sample_image_at_source(image, source, output_shape), extent


def raw_line_corner_homography(edge_points: dict[str, np.ndarray], target_size: float) -> transform.ProjectiveTransform:
    lines = {side: fit_line(np.asarray(edge_points[side], dtype=float)) for side in SIDES}
    corners = np.vstack(
        [
            line_intersection(lines["top"], lines["left"]),
            line_intersection(lines["top"], lines["right"]),
            line_intersection(lines["bottom"], lines["right"]),
            line_intersection(lines["bottom"], lines["left"]),
        ]
    )
    target = np.array([[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]], dtype=float)
    tform = transform.ProjectiveTransform()
    if not np.all(np.isfinite(corners)) or not tform.estimate(corners, target):
        tform.params = np.eye(3)
    return tform


def warp_full_frame_homography_baseline(
    image: np.ndarray,
    edge_points: dict[str, np.ndarray],
    fit: FitResult,
    max_output_pixels: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    tform = raw_line_corner_homography(edge_points, fit.target_size)
    width = fit.source_width
    height = fit.source_height
    border = np.vstack(
        [
            np.column_stack([np.linspace(0, width - 1, 200), np.zeros(200)]),
            np.column_stack([np.full(200, width - 1), np.linspace(0, height - 1, 200)]),
            np.column_stack([np.linspace(width - 1, 0, 200), np.full(200, height - 1)]),
            np.column_stack([np.zeros(200), np.linspace(height - 1, 0, 200)]),
        ]
    )
    mapped = tform(border)
    pad = 0.03 * fit.target_size
    extent = (
        float(np.nanmin(mapped[:, 0]) - pad),
        float(np.nanmax(mapped[:, 0]) + pad),
        float(np.nanmin(mapped[:, 1]) - pad),
        float(np.nanmax(mapped[:, 1]) + pad),
    )
    target_points, output_shape, extent = output_grid_for_extent(extent, max_output_pixels)
    source = tform.inverse(target_points)
    return sample_image_at_source(image, source, output_shape), extent


def save_edge_points_files(edge_points: dict[str, np.ndarray], output_base: Path) -> dict[str, str]:
    json_path = output_base.with_name(f"{output_base.stem}_edge_points.json")
    csv_path = output_base.with_name(f"{output_base.stem}_edge_points.csv")
    payload = {side: np.asarray(edge_points[side], dtype=float).round(4).tolist() for side in SIDES}
    json_path.write_text(json.dumps({"edge_points": payload}, indent=2), encoding="utf-8")
    rows = []
    for side in SIDES:
        for idx, point in enumerate(np.asarray(edge_points[side], dtype=float)):
            rows.append({"side": side, "index": idx, "x": point[0], "y": point[1]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"edge_points_json": str(json_path), "edge_points_csv": str(csv_path)}


def default_edge_points_path(calibration_output: Path) -> Path:
    return calibration_output.with_name(f"{calibration_output.stem}_edge_points.json")


def load_edge_points_file(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "edge_points" in payload:
        payload = payload["edge_points"]
    return sort_edge_points({side: np.asarray(payload[side], dtype=float) for side in SIDES})


def save_diagnostics(
    image: np.ndarray,
    edge_points: dict[str, np.ndarray],
    fit: FitResult,
    output_dir: Path,
    name: str,
    output_pixels: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The session folder already carries the long recording name. Keep
    # diagnostic filenames short so Windows/PIL does not hit path limits.
    original_path = output_dir / "source_points.png"
    warped_path = output_dir / "warped_square.png"
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(image)
    for side, pts in edge_points.items():
        ax.plot(pts[:, 0], pts[:, 1], ".-", color=SIDE_COLORS[side], label=side)
    ax.set_title("Arena edge calibration points")
    ax.legend(loc="lower right")
    fig.savefig(original_path, dpi=160)
    plt.close(fig)

    warped = warp_frame(image, fit, output_pixels)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(warped)
    ax.set_title("Arena-square crop diagnostic")
    ax.set_axis_off()
    fig.savefig(warped_path, dpi=160)
    plt.close(fig)

    full_outputs: dict[str, str] = {}
    for method, label in [
        ("inverse_polynomial", "Inverse polynomial"),
        ("grid_interpolation", "Dense-grid interpolated inverse"),
        ("numerical_forward", "Numerical inversion of forward map"),
    ]:
        full, extent = warp_full_frame_with_inverse(image, fit, output_pixels, method)
        path = output_dir / f"rectified_full_frame_{method}.png"
        save_full_frame_plot(full, extent, edge_points, fit, path, label)
        full_outputs[f"rectified_full_frame_{method}"] = str(path)

    full, extent = warp_full_frame_homography_baseline(image, edge_points, fit, output_pixels)
    homography_path = output_dir / "rectified_full_frame_homography_baseline.png"
    save_full_frame_plot(full, extent, edge_points, fit, homography_path, "Homography baseline")
    full_outputs["rectified_full_frame_homography_baseline"] = str(homography_path)

    compatibility_path = output_dir / "rectified_full_frame.png"
    save_full_frame_plot(
        *warp_full_frame_with_inverse(image, fit, output_pixels, "numerical_forward"),
        edge_points,
        fit,
        compatibility_path,
        "Numerical inversion of forward map",
    )
    full_outputs["rectified_full_frame"] = str(compatibility_path)
    result = {"source_points": str(original_path), "warped_square": str(warped_path)}
    result.update(full_outputs)
    return result


def save_full_frame_plot(
    full: np.ndarray,
    extent: tuple[float, float, float, float],
    edge_points: dict[str, np.ndarray],
    fit: FitResult,
    path: Path,
    title_suffix: str,
) -> None:
    x_min, x_max, y_min, y_max = extent
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(full, extent=[x_min, x_max, y_max, y_min])
    ax.plot(
        [0, fit.target_size, fit.target_size, 0, 0],
        [0, 0, fit.target_size, fit.target_size, 0],
        color="yellow",
        lw=1.8,
        label="ideal square",
    )
    for side, pts in edge_points.items():
        mapped = apply_forward(np.asarray(pts, dtype=float), fit)
        ax.plot(
            mapped[:, 0],
            mapped[:, 1],
            ".-",
            color=SIDE_COLORS[side],
            markersize=4,
            linewidth=1.0,
            label=f"{side} mapped points",
        )
    ax.set_title(f"Full-frame rectified diagnostic: {title_suffix}")
    ax.set_xlabel("corrected arena x")
    ax.set_ylabel("corrected arena y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration_payload(
    video_path: Path,
    frame_index: int,
    image: np.ndarray,
    edge_points: dict[str, np.ndarray],
    target_size: float,
    target_units: str,
    order: int,
    output_dir: Path,
    output_pixels: int,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    edge_points = sort_edge_points(edge_points)
    targets = target_points_for_edges(edge_points, target_size)
    source = np.vstack([edge_points[side] for side in SIDES])
    target = np.vstack([targets[side] for side in SIDES])
    fit = fit_polynomial_transform(
        source,
        target,
        width,
        height,
        target_size,
        order,
        [len(edge_points[side]) for side in SIDES],
    )
    qc = edge_qc(edge_points, fit)
    diagnostics = save_diagnostics(image, edge_points, fit, output_dir, safe_stem(video_path), output_pixels)
    edge_point_files = save_edge_points_files(edge_points, output_dir / safe_stem(video_path))
    transform_debug = save_transform_debug(edge_points, targets, fit, output_dir, safe_stem(video_path))
    return {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(video_path),
        "frame_index": frame_index,
        "image_width_px": width,
        "image_height_px": height,
        "arena_shape": "square",
        "target_units": target_units,
        "target_size": target_size,
        "target_square": [[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]],
        "edge_points": {side: edge_points[side].round(4).tolist() for side in SIDES},
        "transform": {
            "type": "boundary_polynomial_raw_pixel_to_square",
            "order": fit.order,
            "source_width": fit.source_width,
            "source_height": fit.source_height,
            "target_size": fit.target_size,
            "terms": fit.terms,
            "forward_coeff_x": fit.forward_coeff_x,
            "forward_coeff_y": fit.forward_coeff_y,
            "inverse_coeff_x": fit.inverse_coeff_x,
            "inverse_coeff_y": fit.inverse_coeff_y,
            "post_homography": fit.post_homography,
        },
        "qc": qc,
        "diagnostics": diagnostics,
        "edge_point_files": edge_point_files,
        "transform_debug": transform_debug,
        "assumptions": [
            "Arena is planar.",
            "Arena is square in corrected coordinates.",
            "Boundary points mark the arena floor edge.",
            "Polynomial warp corrects visible curved edges empirically from boundary points.",
            "Calibration can be reused only if camera and arena geometry are unchanged.",
        ],
    }


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:80]


def load_calibration(path: Path) -> tuple[dict[str, Any], FitResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tr = payload["transform"]
    fit = FitResult(
        order=int(tr["order"]),
        source_width=int(tr["source_width"]),
        source_height=int(tr["source_height"]),
        target_size=float(tr["target_size"]),
        terms=tr["terms"],
        forward_coeff_x=tr["forward_coeff_x"],
        forward_coeff_y=tr["forward_coeff_y"],
        inverse_coeff_x=tr["inverse_coeff_x"],
        inverse_coeff_y=tr["inverse_coeff_y"],
        post_homography=tr.get("post_homography", np.eye(3).tolist()),
    )
    return payload, fit


def command_calibrate(args: argparse.Namespace) -> None:
    image = read_video_frame(args.video, args.frame_index)
    mask = None
    saved_edge_points_path = args.edge_points_json or default_edge_points_path(args.output)
    if saved_edge_points_path.exists():
        print(f"Loading saved edge points: {saved_edge_points_path}")
        suggestions = load_edge_points_file(saved_edge_points_path)
    elif args.corners_json is not None:
        corners_payload = json.loads(args.corners_json.read_text(encoding="utf-8"))
        corners = np.asarray(corners_payload["corners"], dtype=float)
        suggestions = suggest_edge_points_from_corners(
            image,
            corners,
            args.points_per_edge,
            args.search_radius_px,
            args.profile_half_width_px,
            args.gaussian_sigma,
            args.spline_smoothing,
            args.outlier_threshold_px,
        )
    elif args.no_gui:
        suggestions, mask = suggest_edge_points(image, args.points_per_edge)
    else:
        corners = CornerEditor(image).show()
        suggestions = suggest_edge_points_from_corners(
            image,
            corners,
            args.points_per_edge,
            args.search_radius_px,
            args.profile_half_width_px,
            args.gaussian_sigma,
            args.spline_smoothing,
            args.outlier_threshold_px,
        )
    points = suggestions
    if not args.no_gui:
        editor = EdgePointEditor(image, suggestions, mask)
        points = editor.show()
    output_dir = args.output_dir or args.output.parent / "diagnostics"
    payload = calibration_payload(
        args.video.resolve(),
        args.frame_index,
        image,
        points,
        args.arena_side_length,
        args.target_units,
        args.polynomial_order,
        output_dir,
        args.output_pixels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload["edge_point_files"].update(save_edge_points_files(points, args.output.with_suffix("")))
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote calibration: {args.output}")
    print(f"Mean edge error ({args.target_units}): {payload['qc']['edge_error_mean']:.6f}")


def command_validate(args: argparse.Namespace) -> None:
    payload, fit = load_calibration(args.calibration)
    image = read_video_frame(args.video, args.frame_index)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_points = {side: np.asarray(payload["edge_points"][side], dtype=float) for side in SIDES}
    diagnostics = save_diagnostics(image, edge_points, fit, output_dir, safe_stem(args.video), args.output_pixels)
    summary = {
        "validated_video": str(args.video.resolve()),
        "calibration": str(args.calibration.resolve()),
        "frame_index": args.frame_index,
        "diagnostics": diagnostics,
        "note": "Validation uses saved edge points overlaid on this video frame; recalibrate if arena/camera shifted.",
    }
    out = output_dir / f"{safe_stem(args.video)}_validation.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote validation: {out}")


def find_videos(root: Path, include_no_use: bool) -> list[Path]:
    videos = sorted(root.rglob("*_VIDEO.avi"))
    if not include_no_use:
        videos = [path for path in videos if "no_use_videos" not in str(path)]
    return videos


def command_batch_validate(args: argparse.Namespace) -> None:
    videos = find_videos(args.root, args.include_no_use)
    for video in videos:
        subdir = args.output_dir / safe_stem(video)
        ns = argparse.Namespace(
            video=video,
            calibration=args.calibration,
            frame_index=args.frame_index,
            output_dir=subdir,
            output_pixels=args.output_pixels,
        )
        command_validate(ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and validate square arena calibration files.")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Suggest/edit edge points and save calibration JSON.")
    cal.add_argument("--video", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--frame-index", type=int, default=0)
    cal.add_argument("--points-per-edge", type=int, default=30)
    cal.add_argument("--arena-side-length", type=float, default=1.0)
    cal.add_argument("--target-units", default="normalized_arena_side")
    cal.add_argument("--polynomial-order", type=int, default=3)
    cal.add_argument("--output-pixels", type=int, default=700)
    cal.add_argument("--output-dir", type=Path, default=None)
    cal.add_argument("--corners-json", type=Path, default=None, help="Optional rough corners JSON for non-GUI testing.")
    cal.add_argument("--edge-points-json", type=Path, default=None, help="Optional saved edge-points JSON to load.")
    cal.add_argument("--search-radius-px", type=float, default=35.0)
    cal.add_argument("--profile-half-width-px", type=int, default=3)
    cal.add_argument("--gaussian-sigma", type=float, default=1.5)
    cal.add_argument("--spline-smoothing", type=float, default=8.0)
    cal.add_argument("--outlier-threshold-px", type=float, default=12.0)
    cal.add_argument("--no-gui", action="store_true", help="Use suggestions directly; mainly for smoke tests.")
    cal.set_defaults(func=command_calibrate)

    val = sub.add_parser("validate", help="Create diagnostic images for one video using a saved calibration.")
    val.add_argument("--video", type=Path, required=True)
    val.add_argument("--calibration", type=Path, required=True)
    val.add_argument("--frame-index", type=int, default=0)
    val.add_argument("--output-dir", type=Path, required=True)
    val.add_argument("--output-pixels", type=int, default=700)
    val.set_defaults(func=command_validate)

    batch = sub.add_parser("batch-validate", help="Validate a shared calibration across discovered videos.")
    batch.add_argument("--root", type=Path, default=Path("."))
    batch.add_argument("--calibration", type=Path, required=True)
    batch.add_argument("--frame-index", type=int, default=0)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--output-pixels", type=int, default=700)
    batch.add_argument("--include-no-use", action="store_true")
    batch.set_defaults(func=command_batch_validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
