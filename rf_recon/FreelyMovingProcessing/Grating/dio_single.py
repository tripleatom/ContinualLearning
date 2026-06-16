"""
Load and inspect raw DIO signal from a single .rec folder.
Applies glitch removal and jitter correction on rising edges.

Set the config section below, then run.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from process_func.DIO import get_dio_folders, concatenate_din_data

# ── Config ─────────────────────────────────────────────────────────────────────
rec_folder = Path(input("Enter path to .rec folder: ").strip())
channel_id        = 3      # Din channel (1-indexed)
fs                = 30_000
expected_iti_s    = 3.0    # expected inter-trial interval in seconds
tolerance         = 0.02   # ±2 % around expected ITI


# ── Helpers ────────────────────────────────────────────────────────────────────

def remove_glitches(edges, expected, tol):
    """
    Remove extra edges whose interval to the next is shorter than
    (1 - tol) * expected.  When two consecutive edges are too close,
    keep the one whose *next* gap is closer to expected.

    Returns cleaned edge array and a log of removals.
    """
    edges = list(edges)
    lo = (1 - tol) * expected
    log = []
    i = 0
    while i < len(edges) - 1:
        interval = edges[i + 1] - edges[i]
        if interval < lo:
            next_if_keep_i   = edges[i + 2] - edges[i]     if i + 2 < len(edges) else np.inf
            next_if_keep_i1  = edges[i + 2] - edges[i + 1] if i + 2 < len(edges) else np.inf
            if abs(next_if_keep_i - expected) <= abs(next_if_keep_i1 - expected):
                log.append(f"  removed glitch at index {i+1} "
                           f"(sample {edges[i+1]:.0f}, interval {interval/expected:.3f}x)")
                edges.pop(i + 1)
            else:
                log.append(f"  removed glitch at index {i} "
                           f"(sample {edges[i]:.0f}, interval {interval/expected:.3f}x)")
                edges.pop(i)
        else:
            i += 1
    return np.array(edges), log


def fill_missing(edges, expected, tol):
    """
    Insert interpolated edges where the gap between consecutive edges is
    longer than (1 + tol) * expected, indicating one or more missed trials.

    For each such gap, n_missing = round(gap / expected) - 1 edges are
    inserted at equal spacing between the two flanking edges.

    Returns the filled edge array and a log of insertions.
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
            # Use mean of nearby good gaps as step size
            good = [edges[j + 1] - edges[j]
                    for j in range(max(0, i - 10), min(len(edges) - 1, i + 10))
                    if lo <= edges[j + 1] - edges[j] <= hi]
            step = int(round(np.mean(good))) if good else int(expected)
            for k in range(1, n_missing + 1):
                inserted = int(edges[i] + k * step)
                edges.insert(i + k, inserted)
                log.append(f"  Inserted edge at sample {inserted} "
                           f"(gap was {gap/expected:.2f}x expected, step={step})")
            i += n_missing + 1
        else:
            i += 1
    return np.array(edges), log


def fix_jitter(edges, expected, tol):
    """
    Snap individual edges whose gap to the previous or next edge deviates
    from expected by less than 50 % (i.e. not a true missing trial).
    Uses the neighbouring good edge as the anchor.

    Returns corrected edge array and a log of corrections.
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
                log.append(f"  jitter fix index {i+1}: {old:.0f} → {new}  "
                           f"(gap was {gap/expected:.3f}x, anchored left)")
        elif next_ok:
            new = int(edges[i + 1] - expected)
            if i == 0 or lo <= new - edges[i - 1] <= hi:
                old = edges[i]
                edges[i] = new
                log.append(f"  jitter fix index {i}: {old:.0f} → {new}  "
                           f"(gap was {gap/expected:.3f}x, anchored right)")
        i += 1
    return np.array(edges), log


# ── Load ───────────────────────────────────────────────────────────────────────
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
print(f"Found {len(dio_folders)} DIO folder(s) in {rec_folder.name}")
for f in dio_folders:
    print(f"  {f.name}")

time, state = concatenate_din_data(dio_folders, channel_id)
time  = time.ravel().astype(float)
state = state.ravel()
time -= time[0]   # zero-align

rising_raw = time[state == 1]
print(f"\nTotal samples : {len(time)}")
print(f"Rising  edges : {len(rising_raw)}")
print(f"Duration      : {time[-1]/fs:.1f} s")

# ── Fix ────────────────────────────────────────────────────────────────────────
expected_samples = expected_iti_s * fs

rising_clean, glitch_log = remove_glitches(rising_raw, expected_samples, tolerance)
print(f"\nGlitch removal: {len(rising_raw)} → {len(rising_clean)} edges")
for entry in glitch_log:
    print(entry)

rising_fixed, jitter_log = fix_jitter(rising_clean, expected_samples, tolerance)
print(f"\nJitter fix: {len(rising_clean)} → {len(rising_fixed)} edges")
for entry in jitter_log:
    print(entry)

rising_filled, fill_log = fill_missing(rising_fixed, expected_samples, tolerance)
print(f"\nFill missing: {len(rising_fixed)} → {len(rising_filled)} edges")
for entry in fill_log:
    print(entry)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
fig.suptitle(rec_folder.name, fontsize=11, fontweight='bold')

for ax, edges, color, label in [
    (axes[0], rising_raw,    'steelblue',  f'Raw ({len(rising_raw)} edges)'),
    (axes[1], rising_fixed,  'darkorange', f'After glitch/jitter fix ({len(rising_fixed)} edges)'),
    (axes[2], rising_filled, 'green',      f'After fill missing ({len(rising_filled)} edges)'),
]:
    if len(edges) > 1:
        iti = np.diff(edges) / fs
        ax.plot(iti, marker='o', ms=3, lw=0.8, color=color)
        ax.axhline(expected_iti_s,               color='red',  ls='--', lw=1.2, label=f'expected {expected_iti_s} s')
        ax.axhline(expected_iti_s * (1 - tolerance), color='red', ls=':',  lw=0.8)
        ax.axhline(expected_iti_s * (1 + tolerance), color='red', ls=':',  lw=0.8)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('Interval (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Rising edge index')
plt.tight_layout()
plt.show()

save_choice = input(
    "\nSave which edges? [raw/original=no fix, fixed=glitch+jitter, filled=all fixes] "
    "(default: filled): "
).strip().lower()

if save_choice in ("raw", "original", "nofix", "no fix", "none"):
    rising_out = rising_raw
    save_label = "raw/original edges (no fix)"
elif save_choice in ("fixed", "glitch", "jitter"):
    rising_out = rising_fixed
    save_label = "glitch/jitter-fixed edges"
else:
    rising_out = rising_filled
    save_label = "filled/final fixed edges"

# ── Save ───────────────────────────────────────────────────────────────────────
stimulus_duration_s = expected_iti_s  # adjust if stimulus duration differs from ITI
stim_samples = int(stimulus_duration_s * fs)

falling_out = rising_out + stim_samples
save_path = rec_folder.parent / f"{rec_folder.stem}_DIO.npz"
np.savez_compressed(save_path, rising_times=rising_out, falling_times=falling_out)
print(f"Saved {len(rising_out)} {save_label} → {save_path}")
