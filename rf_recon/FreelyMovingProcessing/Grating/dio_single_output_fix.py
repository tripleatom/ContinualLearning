"""
Load an exported DIO .npz file, pick one rising-edge range, repair only that
range, and save only that range as a smaller DIO .npz.

The file is expected to contain:
    rising_times
    falling_times

Range input uses edge indices, not sample values:
    100:140   -> fixes indices 100 through 139
    100-140   -> fixes indices 100 through 140
    blank     -> picks the full file
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -- Config --------------------------------------------------------------------
npz_path = Path(input("Enter path to exported DIO .npz: ").strip().strip('"'))
fs = 30_000
expected_iti_s = 3.0
tolerance = 0.02


# -- Helpers -------------------------------------------------------------------
def remove_glitches(edges, expected, tol):
    """
    Remove extra edges whose interval to the next edge is shorter than
    (1 - tol) * expected. When two consecutive edges are too close, keep
    the one whose next gap is closer to expected.
    """
    edges = list(edges)
    lo = (1 - tol) * expected
    log = []
    i = 0

    while i < len(edges) - 1:
        interval = edges[i + 1] - edges[i]
        if interval < lo:
            next_if_keep_i = edges[i + 2] - edges[i] if i + 2 < len(edges) else np.inf
            next_if_keep_i1 = edges[i + 2] - edges[i + 1] if i + 2 < len(edges) else np.inf

            if abs(next_if_keep_i - expected) <= abs(next_if_keep_i1 - expected):
                log.append(
                    f"  removed glitch at local index {i + 1} "
                    f"(sample {edges[i + 1]:.0f}, interval {interval / expected:.3f}x)"
                )
                edges.pop(i + 1)
            else:
                log.append(
                    f"  removed glitch at local index {i} "
                    f"(sample {edges[i]:.0f}, interval {interval / expected:.3f}x)"
                )
                edges.pop(i)
        else:
            i += 1

    return np.array(edges), log


def fill_missing(edges, expected, tol):
    """
    Insert interpolated edges where the gap between consecutive edges is
    longer than (1 + tol) * expected.
    """
    edges = list(edges)
    lo = (1 - tol) * expected
    hi = (1 + tol) * expected
    log = []
    i = 0

    while i < len(edges) - 1:
        gap = edges[i + 1] - edges[i]
        if gap > hi:
            n_missing = round(gap / expected) - 1
            good = [
                edges[j + 1] - edges[j]
                for j in range(max(0, i - 10), min(len(edges) - 1, i + 10))
                if lo <= edges[j + 1] - edges[j] <= hi
            ]
            step = int(round(np.mean(good))) if good else int(expected)

            for k in range(1, n_missing + 1):
                inserted = int(edges[i] + k * step)
                edges.insert(i + k, inserted)
                log.append(
                    f"  inserted edge at sample {inserted} "
                    f"(gap was {gap / expected:.2f}x expected, step={step})"
                )

            i += n_missing + 1
        else:
            i += 1

    return np.array(edges), log


def fix_jitter(edges, expected, tol):
    """
    Snap individual edges whose gap to a neighbor deviates from expected by
    less than 50 percent, using a neighboring good edge as the anchor.
    """
    edges = list(edges)
    lo = (1 - tol) * expected
    hi = (1 + tol) * expected
    log = []
    i = 0

    while i < len(edges) - 1:
        gap = edges[i + 1] - edges[i]
        if lo <= gap <= hi or gap > 1.5 * expected:
            i += 1
            continue

        prev_ok = i > 0 and lo <= edges[i] - edges[i - 1] <= hi
        next_ok = i + 2 < len(edges) and lo <= edges[i + 2] - edges[i + 1] <= hi

        if prev_ok:
            new = int(edges[i] + expected)
            if i + 2 >= len(edges) or lo <= edges[i + 2] - new <= hi:
                old = edges[i + 1]
                edges[i + 1] = new
                log.append(
                    f"  jitter fix local index {i + 1}: {old:.0f} -> {new} "
                    f"(gap was {gap / expected:.3f}x, anchored left)"
                )
        elif next_ok:
            new = int(edges[i + 1] - expected)
            if i == 0 or lo <= new - edges[i - 1] <= hi:
                old = edges[i]
                edges[i] = new
                log.append(
                    f"  jitter fix local index {i}: {old:.0f} -> {new} "
                    f"(gap was {gap / expected:.3f}x, anchored right)"
                )

        i += 1

    return np.array(edges), log


def parse_index_range(range_text, n_edges):
    text = range_text.strip()
    if not text:
        return 0, n_edges

    if ":" in text:
        start_text, end_text = text.split(":", 1)
        end_inclusive = False
    elif "-" in text:
        start_text, end_text = text.split("-", 1)
        end_inclusive = True
    else:
        idx = int(text)
        return idx, idx + 1

    start = int(start_text.strip()) if start_text.strip() else 0
    end = int(end_text.strip()) if end_text.strip() else n_edges
    if end_inclusive:
        end += 1

    start = max(0, start)
    end = min(n_edges, end)
    if start >= end:
        raise ValueError(f"Invalid range {range_text!r} for {n_edges} edge(s)")

    return start, end


def repair_edges(edges, expected_samples, tol):
    clean, glitch_log = remove_glitches(edges, expected_samples, tol)
    fixed, jitter_log = fix_jitter(clean, expected_samples, tol)
    filled, fill_log = fill_missing(fixed, expected_samples, tol)
    return clean, fixed, filled, glitch_log, jitter_log, fill_log


def estimate_stim_samples(rising, falling, default_samples):
    if falling is None or len(falling) != len(rising):
        return default_samples

    durations = falling - rising
    durations = durations[durations > 0]
    if len(durations) == 0:
        return default_samples

    return int(round(np.median(durations)))


def plot_selected_range(raw_edges, fixed_edges, filled_edges, start_index, end_index):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(
        f"{npz_path.name} selected range {start_index}:{end_index}",
        fontsize=11,
        fontweight="bold",
    )

    for ax, edges, color, label in [
        (axes[0], raw_edges, "steelblue", f"Raw selected range ({len(raw_edges)} edges)"),
        (axes[1], fixed_edges, "darkorange", f"After glitch/jitter fix ({len(fixed_edges)} edges)"),
        (axes[2], filled_edges, "green", f"After fill missing ({len(filled_edges)} edges)"),
    ]:
        if len(edges) > 1:
            iti = np.diff(edges) / fs
            ax.plot(iti, marker="o", ms=3, lw=0.8, color=color)
            ax.axhline(
                expected_iti_s,
                color="red",
                ls="--",
                lw=1.2,
                label=f"expected {expected_iti_s} s",
            )
            ax.axhline(expected_iti_s * (1 - tolerance), color="red", ls=":", lw=0.8)
            ax.axhline(expected_iti_s * (1 + tolerance), color="red", ls=":", lw=0.8)
        else:
            ax.text(0.5, 0.5, "Need at least 2 edges", ha="center", va="center")

        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Interval (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Local rising edge interval index in selected range")
    plt.tight_layout()
    plt.show()


# -- Load ----------------------------------------------------------------------
with np.load(npz_path, allow_pickle=True) as dio_data:
    print(f"Loaded: {npz_path}")
    print(f"Keys: {list(dio_data.files)}")

    if "rising_times" not in dio_data.files:
        raise KeyError(f"{npz_path} does not contain 'rising_times'")

    rising_raw = dio_data["rising_times"].ravel().astype(float)
    falling_raw = (
        dio_data["falling_times"].ravel().astype(float)
        if "falling_times" in dio_data.files
        else None
    )

print(f"\nRising edges : {len(rising_raw)}")
if falling_raw is not None:
    print(f"Falling edges: {len(falling_raw)}")
if len(rising_raw):
    print(f"Duration     : {rising_raw[-1] / fs:.1f} s")


# -- Pick and fix selected range ----------------------------------------------
expected_samples = expected_iti_s * fs
range_text = input(
    "\nEnter edge index range to fix [start:end, end exclusive; start-end inclusive; blank=all]: "
)
fix_start, fix_end = parse_index_range(range_text, len(rising_raw))
print(f"Picked rising_times[{fix_start}:{fix_end}] ({fix_end - fix_start} edge(s))")

range_raw = rising_raw[fix_start:fix_end]
range_falling_raw = (
    falling_raw[fix_start:fix_end]
    if falling_raw is not None and len(falling_raw) == len(rising_raw)
    else None
)
range_clean, range_fixed, range_filled, glitch_log, jitter_log, fill_log = repair_edges(
    range_raw,
    expected_samples,
    tolerance,
)

print(f"\nGlitch removal: {len(range_raw)} -> {len(range_clean)} selected edges")
for entry in glitch_log:
    print(entry)

print(f"\nJitter fix: {len(range_clean)} -> {len(range_fixed)} selected edges")
for entry in jitter_log:
    print(entry)

print(f"\nFill missing: {len(range_fixed)} -> {len(range_filled)} selected edges")
for entry in fill_log:
    print(entry)

plot_selected_range(range_raw, range_fixed, range_filled, fix_start, fix_end)


# -- Save only the selected range ---------------------------------------------
rising_out = range_filled
default_stim_samples = int(round(expected_iti_s * fs))
stim_samples = estimate_stim_samples(range_raw, range_falling_raw, default_stim_samples)
falling_out = rising_out + stim_samples

range_label = f"{fix_start}_{fix_end - 1}" if fix_end > fix_start else f"{fix_start}"
default_save_path = npz_path.with_name(f"{npz_path.stem}_range_{range_label}_DIO.npz")
save_text = input(f"Save path (blank={default_save_path}): ").strip().strip('"')
save_path = Path(save_text) if save_text else default_save_path

np.savez_compressed(
    save_path,
    rising_times=rising_out,
    falling_times=falling_out,
)
print(f"Saved {len(rising_out)} repaired selected edges -> {save_path}")
