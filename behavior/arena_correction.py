"""Arena view correction: lens undistortion + perspective warp to top-down.

Just hit Run — the CONFIG block below controls everything. No CLI needed.
You can still run from the terminal:
    python arena_correction.py                # uses CONFIG below
    python arena_correction.py calibrate      # interactive (k1/k2 + click corners)
    python arena_correction.py visualize      # 4-panel preview PNG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG — edit these and press Run.
# ----------------------------------------------------------------------------
VIDEO_PATH   = r"D:\cl\video\front_camera_CnL43_2026-05-11_1_VIDEO.avi"
PREVIEW_PATH = r"D:\cl\video\arena_preview.png"

# which mode to run when you press Run with no CLI args:
#   "visualize" — save a 4-panel PNG using CORNERS/K1/K2 below
#   "calibrate" — open interactive windows to pick k1, k2 then click corners
DEFAULT_MODE = "calibrate"

FRAME_IDX = 30000        # which frame to preview (0 = first)
K1, K2    = -0.10, 0.0   # lens distortion. More negative = stronger barrel correction.
CORNERS   = [            # TL, TR, BR, BL pixel coords on the RAW frame. None = auto-detect.
    (75,  65),
    (470, 90),
    (430, 420),
    (75,  420),
]
# ----------------------------------------------------------------------------

CALIB_PATH = Path(__file__).with_name("arena_calibration.json")
OUT_SIZE = 512  # side length of warped top-down view, in pixels


def grab_frame(video_path: str, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def undistort_maps(shape_hw: tuple[int, int], k1: float, k2: float):
    """Build remap LUTs for a simple radial-distortion model with principal point at center."""
    h, w = shape_hw
    fx = fy = max(h, w)  # rough focal length in pixels; tune via k1/k2
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
    return map1, map2, K, dist, new_K


def _draw_grid(img: np.ndarray, n: int = 10, color=(0, 200, 255)) -> np.ndarray:
    """Overlay an n×n reference grid. Straight arena edges should run parallel to it."""
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, n):
        x = int(w * i / n)
        y = int(h * i / n)
        cv2.line(out, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
    return out


def tune_undistortion(frame: np.ndarray, k1_init: float = -0.10,
                      k2_init: float = 0.0) -> tuple[float, float]:
    """Interactive trackbars to pick k1, k2 by eye. Returns chosen (k1, k2).

    Keys:
        a / s  — decrease / increase k1
        z / x  — decrease / increase k2
        [ / ]  — make step finer / coarser
        g      — toggle reference grid
        r      — reset to initial values
        Enter  — accept
        Esc    — cancel
    """
    win = "undistort  [a/s: k1, z/x: k2, [/]: step, g: grid, r: reset, Enter: ok]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 900)
    k1, k2 = k1_init, k2_init
    step = 0.01
    show_grid = True
    while True:
        map1, map2, *_ = undistort_maps(frame.shape[:2], k1, k2)
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        disp = _draw_grid(undist) if show_grid else undist.copy()
        cv2.putText(disp, f"k1={k1:+.3f}  k2={k2:+.3f}  step={step:.3f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            break
        if key == 27:
            raise KeyboardInterrupt
        if key == ord("a"): k1 -= step
        if key == ord("s"): k1 += step
        if key == ord("z"): k2 -= step
        if key == ord("x"): k2 += step
        if key == ord("["): step = max(0.001, step / 2)
        if key == ord("]"): step = min(0.2, step * 2)
        if key == ord("g"): show_grid = not show_grid
        if key == ord("r"): k1, k2 = k1_init, k2_init
    cv2.destroyWindow(win)
    return k1, k2


def pick_corners(frame: np.ndarray,
                 initial: np.ndarray | None = None) -> np.ndarray:
    """Click TL, TR, BR, BL (in that order); drag any existing point to nudge it.

    A zoomed inset around the cursor is shown in the top-right for sub-pixel placement.
    Keys: u undo, r reset, Enter accept, Esc cancel.
    Returns 4x2 float32 array in TL,TR,BR,BL order.
    """
    labels = ["TL", "TR", "BR", "BL"]
    pts: list[list[float]] = [list(p) for p in initial] if initial is not None else []
    state = {"cursor": (0, 0), "dragging": None}
    win = "pick corners (TL,TR,BR,BL)  [click, drag to nudge, u undo, r reset, Enter ok]"
    HIT_RADIUS = 12  # pixels to grab an existing point

    def on_mouse(event, x, y, flags, _):
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # try to grab an existing point first
            for i, (px, py) in enumerate(pts):
                if (px - x) ** 2 + (py - y) ** 2 <= HIT_RADIUS ** 2:
                    state["dragging"] = i
                    return
            if len(pts) < 4:
                pts.append([float(x), float(y)])
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
            pts[state["dragging"]] = [float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = None

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 900)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = frame.copy()
        # draw polygon between picked points
        if len(pts) >= 2:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [poly], len(pts) == 4, (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(disp, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(disp, labels[i], (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # status bar
        msg = (f"click {labels[len(pts)]}" if len(pts) < 4
               else "all 4 set — drag to nudge, Enter to accept")
        cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if len(pts) < 4 else (0, 255, 0), 2)

        # zoomed inset around the cursor (top-right)
        cx, cy = state["cursor"]
        rad = 30
        h, w = frame.shape[:2]
        x0, y0 = max(0, cx - rad), max(0, cy - rad)
        x1, y1 = min(w, cx + rad), min(h, cy + rad)
        if x1 - x0 > 4 and y1 - y0 > 4:
            patch = frame[y0:y1, x0:x1]
            zoom = cv2.resize(patch, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.line(zoom, (100, 0), (100, 200), (0, 0, 255), 1)
            cv2.line(zoom, (0, 100), (200, 100), (0, 0, 255), 1)
            disp[10:210, w - 210:w - 10] = zoom
            cv2.rectangle(disp, (w - 211, 9), (w - 9, 211), (255, 255, 255), 1)

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and pts:
            pts.pop()
        elif key == ord("r"):
            pts.clear()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            raise KeyboardInterrupt

    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.float32)


def perspective_matrix(src_corners: np.ndarray, out_size: int = OUT_SIZE) -> np.ndarray:
    dst = np.array([[0, 0],
                    [out_size - 1, 0],
                    [out_size - 1, out_size - 1],
                    [0, out_size - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_corners, dst)


def save_calibration(k1: float, k2: float, src_corners: np.ndarray,
                     shape_hw: tuple[int, int], out_size: int) -> None:
    payload = {
        "k1": float(k1),
        "k2": float(k2),
        "src_corners_TL_TR_BR_BL": src_corners.tolist(),
        "frame_h": int(shape_hw[0]),
        "frame_w": int(shape_hw[1]),
        "out_size": int(out_size),
    }
    CALIB_PATH.write_text(json.dumps(payload, indent=2))
    print(f"saved {CALIB_PATH}")


def auto_detect_corners(frame: np.ndarray) -> np.ndarray | None:
    """Threshold the bright floor and approximate it to a 4-corner polygon.

    Returns corners ordered TL, TR, BR, BL, or None if detection fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    # try progressively looser polygon approximation until we get 4 vertices
    peri = cv2.arcLength(cnt, True)
    quad = None
    for eps in np.linspace(0.005, 0.08, 20):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(quad)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def visualize(video: str, out_path: str, frame_idx: int = 0,
              k1: float = -0.30, k2: float = 0.0,
              manual_corners: np.ndarray | None = None) -> None:
    """Render a 4-panel preview: raw, raw+outline, undistorted+outline, top-down warp.

    If `manual_corners` (TL,TR,BR,BL on the RAW frame) is given, those are used
    instead of auto-detection.
    """
    frame = grab_frame(video, frame_idx)
    h, w = frame.shape[:2]

    map1, map2, *_ = undistort_maps((h, w), k1, k2)
    undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    if manual_corners is not None:
        raw_corners = manual_corners.astype(np.float32)
        # warp the corner pixels through the same undistortion to get their
        # locations in the undistorted image
        pts = raw_corners.reshape(-1, 1, 2)
        new_K = undistort_maps((h, w), k1, k2)[4]
        K = undistort_maps((h, w), k1, k2)[2]
        dist = undistort_maps((h, w), k1, k2)[3]
        und_corners = cv2.undistortPoints(pts, K, dist, P=new_K).reshape(-1, 2).astype(np.float32)
    else:
        raw_corners = auto_detect_corners(frame)
        und_corners = auto_detect_corners(undist)

    def draw_outline(img, corners, color=(0, 255, 0)):
        out = img.copy()
        if corners is None:
            cv2.putText(out, "no quad detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2)
        for i, (x, y) in enumerate(corners):
            cv2.circle(out, (int(x), int(y)), 6, color, -1)
            cv2.putText(out, ["TL", "TR", "BR", "BL"][i], (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    raw_outlined = draw_outline(frame, raw_corners, (0, 255, 255))
    und_outlined = draw_outline(undist, und_corners, (0, 255, 0))

    if und_corners is not None:
        M = perspective_matrix(und_corners, OUT_SIZE)
        warped = cv2.warpPerspective(undist, M, (OUT_SIZE, OUT_SIZE))
    else:
        warped = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
        cv2.putText(warped, "no warp", (20, OUT_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def fit(img, target_h):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h))

    panel_h = 600
    panels = [fit(frame, panel_h), fit(raw_outlined, panel_h),
              fit(und_outlined, panel_h), fit(warped, panel_h)]
    titles = ["raw", "raw + auto outline", f"undistorted (k1={k1:+.2f}) + outline", "top-down warp"]
    labeled = []
    for img, t in zip(panels, titles):
        bar = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        labeled.append(np.vstack([bar, img]))
    grid = np.hstack(labeled)
    cv2.imwrite(out_path, grid)
    print(f"saved preview to {out_path}")
    if raw_corners is not None:
        print(f"raw corners (TL,TR,BR,BL):\n{raw_corners}")
    if und_corners is not None:
        print(f"undistorted corners (TL,TR,BR,BL):\n{und_corners}")


def cmd_calibrate(video: str, frame_idx: int) -> None:
    frame = grab_frame(video, frame_idx)
    print("step 1/2: tune lens undistortion until arena edges look straight against the grid")
    print("          keys: a/s (k1 -/+), z/x (k2 -/+), [/] step, g grid, r reset, Enter accept")
    k1, k2 = tune_undistortion(frame, K1, K2)
    map1, map2, *_ = undistort_maps(frame.shape[:2], k1, k2)
    undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    # if the user already saved corners on the RAW frame in CONFIG, project them
    # forward onto the undistorted image as a starting point they can nudge.
    initial = None
    if CORNERS is not None:
        _, _, K, dist, new_K = undistort_maps(frame.shape[:2], k1, k2)
        raw_pts = np.array(CORNERS, dtype=np.float32).reshape(-1, 1, 2)
        initial = cv2.undistortPoints(raw_pts, K, dist, P=new_K).reshape(-1, 2)

    print("step 2/2: click TL, TR, BR, BL on the undistorted floor (drag to refine)")
    corners_undist = pick_corners(undist, initial=initial)

    # project corners back to the raw-frame coordinates so they can be saved
    # in a stable, image-independent way and visualized over the raw input.
    _, _, K, dist, new_K = undistort_maps(frame.shape[:2], k1, k2)
    # cv2.projectPoints expects 3-D points; use depth=1 and invert new_K
    homog = np.hstack([corners_undist, np.ones((4, 1), dtype=np.float32)])
    norm = (np.linalg.inv(new_K) @ homog.T).T  # rays in camera coords
    raw_pts, _ = cv2.projectPoints(norm.astype(np.float32),
                                   np.zeros(3), np.zeros(3), K, dist)
    corners_raw = raw_pts.reshape(-1, 2).astype(np.float32)
    save_calibration(k1, k2, corners_raw, frame.shape[:2], OUT_SIZE)

    # write the 4-panel preview PNG using the picked corners
    visualize(video, PREVIEW_PATH, frame_idx, k1, k2, manual_corners=corners_raw)
    M = perspective_matrix(corners_undist)
    warped = cv2.warpPerspective(undist, M, (OUT_SIZE, OUT_SIZE))
    cv2.imshow("preview — top-down warp (press any key to close)", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"saved calibration to {CALIB_PATH}")


def run_from_config(mode: str) -> None:
    """Run the chosen mode using the CONFIG values at the top of this file."""
    manual = np.array(CORNERS, dtype=np.float32) if CORNERS is not None else None
    if mode == "calibrate":
        cmd_calibrate(VIDEO_PATH, FRAME_IDX)
    elif mode == "visualize":
        visualize(VIDEO_PATH, PREVIEW_PATH, FRAME_IDX, K1, K2, manual)
    else:
        raise ValueError(f"unknown mode {mode!r}; use calibrate or visualize")


def main() -> None:
    # No CLI args → just use the CONFIG block above. Lets you press Run in the IDE.
    if len(sys.argv) == 1:
        print(f"[arena_correction] no CLI args; running mode={DEFAULT_MODE!r} from CONFIG")
        run_from_config(DEFAULT_MODE)
        return

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="interactively pick k1/k2 and 4 corners")
    pc.add_argument("--video", default=VIDEO_PATH)
    pc.add_argument("--frame", type=int, default=FRAME_IDX)

    pv = sub.add_parser("visualize", help="auto-detect arena outline and save a preview PNG")
    pv.add_argument("--video", default=VIDEO_PATH)
    pv.add_argument("--out", default=PREVIEW_PATH)
    pv.add_argument("--frame", type=int, default=FRAME_IDX)
    pv.add_argument("--k1", type=float, default=K1)
    pv.add_argument("--k2", type=float, default=K2)
    pv.add_argument("--corners", type=str, default=None,
                    help="manual corners as 'x1,y1;x2,y2;x3,y3;x4,y4' in TL,TR,BR,BL order")

    args = p.parse_args()
    if args.cmd == "calibrate":
        cmd_calibrate(args.video, args.frame)
    else:
        if args.corners:
            manual = np.array([[float(v) for v in pt.split(",")]
                               for pt in args.corners.split(";")], dtype=np.float32)
        else:
            manual = np.array(CORNERS, dtype=np.float32) if CORNERS is not None else None
        visualize(args.video, args.out, args.frame, args.k1, args.k2, manual)


if __name__ == "__main__":
    main()
