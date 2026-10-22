r"""Plot and quantify cross-shank agreement of detected global ON/OFF periods.

This consumes the per-shank pickles written by ``detect_off_states.py``.  The
word "global" in those files means global across channels *within one shank*;
this script asks the separate, downstream question of whether those shank-level
states line up across the probe.

Three states are shown because the detector's accepted ON intervals do not
cover the whole complement of OFF: ON is restricted to 50--2000 ms gaps between
two accepted OFF periods.  Everything else is labelled "unclassified" rather
than silently called ON.

Outputs, once per epoch:

By default all three files are written directly to the session's experiment-
data folder (the parent of ``MUA``).  Pass ``--out`` to override that location.

``*_on_off_synchrony.png``
    One state row per shank plus the fraction of shanks OFF through time, and
    pairwise OFF Jaccard/kappa heat maps.
``*_on_off_pairwise.csv``
    Exact interval Jaccard, binned chance-corrected Cohen's kappa, overlap
    enrichment, and one-to-one OFF-onset matching within a tolerance.
``*_on_off_synchrony.json``
    Across-pair summaries and population consensus fractions.

Examples
--------
    python plot_on_off_synchrony.py --session-folder D:\data\CnL46_20260729
    python plot_on_off_synchrony.py --epochs post --shanks 4 5 6
    python plot_on_off_synchrony.py --tag f75_strict_r1hz_nobad --bin-ms 10
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

_HERE = Path(__file__).resolve().parent
_SLEEP = _HERE.parent
_ROOT = _SLEEP.parent
for _path in (str(_HERE), str(_SLEEP), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import detect_off_states as dos  # noqa: E402
from server_fallback import resolve_output_folder  # noqa: E402


OFF_COLOR = dos.PURPLE
ON_COLOR = dos.STEELBLUE
UNKNOWN_COLOR = "#eeeeee"
_FILE_RE = re.compile(
    r"^(?P<session>.+)_(?P<epoch>pre|post)_sh(?P<shank>\d+)_off_states_(?P<tag>.+)\.pkl$",
    re.I,
)


def find_off_dir(session_folder=None):
    """Return the first discoverable ``MUA/off_states`` directory."""
    for folder in dos.candidate_session_folders(session_folder):
        candidate = Path(folder) / "MUA" / "off_states"
        if candidate.is_dir():
            return candidate
    return None


def discover_variants(off_dir, session_name, epoch, tag_filter=None):
    """Map ``tag -> shank -> path`` for one session/epoch."""
    variants = {}
    for path in Path(off_dir).glob(f"*_{epoch}_sh*_off_states_*.pkl"):
        match = _FILE_RE.match(path.name)
        if not match or match.group("epoch").lower() != epoch.lower():
            continue
        if session_name and match.group("session") != session_name:
            continue
        tag = match.group("tag")
        if tag_filter and tag_filter not in tag:
            continue
        shank = int(match.group("shank"))
        previous = variants.setdefault(tag, {}).get(shank)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            variants[tag][shank] = path
    return variants


def choose_variant(variants, requested_shanks=None):
    """Choose one common parameter tag; never mix variants across shanks."""
    if not variants:
        return None, {}
    wanted = set(requested_shanks or ())

    def score(item):
        _, paths = item
        coverage = len(wanted.intersection(paths)) if wanted else len(paths)
        newest = max(path.stat().st_mtime for path in paths.values())
        return coverage, len(paths), newest

    tag, paths = max(variants.items(), key=score)
    if requested_shanks:
        paths = {shank: paths[shank] for shank in requested_shanks if shank in paths}
    return tag, paths


def load_results(paths):
    # NumPy 2 pickles refer to the private package as ``numpy._core`` whereas
    # NumPy 1.x exposes the same implementation as ``numpy.core``.  The lab's
    # processing and plotting environments can sit on opposite sides of that
    # rename.  Aliasing only these already-imported NumPy modules keeps these
    # simple ndarray-containing result pickles portable without rewriting them.
    if int(np.__version__.split(".", 1)[0]) < 2:
        sys.modules.setdefault("numpy._core", np.core)
        for module_name in ("numeric", "multiarray", "fromnumeric", "_multiarray_umath"):
            module = importlib.import_module(f"numpy.core.{module_name}")
            sys.modules.setdefault(f"numpy._core.{module_name}", module)
    results = {}
    for shank, path in sorted(paths.items()):
        with path.open("rb") as file:
            result = pickle.load(file)
        results[shank] = result
    return results


def clipped_intervals(intervals, start, end):
    """Clip intervals to [start, end], dropping empty intersections."""
    values = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if not len(values):
        return values
    values = values.copy()
    values[:, 0] = np.maximum(values[:, 0], start)
    values[:, 1] = np.minimum(values[:, 1], end)
    return values[values[:, 1] > values[:, 0]]


def interval_duration(intervals):
    values = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    return float(np.sum(values[:, 1] - values[:, 0])) if len(values) else 0.0


def intersection_duration(a, b):
    """Exact overlap duration of two sorted, non-overlapping interval sets."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i, 1], b[j, 1]) - max(a[i, 0], b[j, 0]))
        if a[i, 1] <= b[j, 1]:
            i += 1
        else:
            j += 1
    return total


def interval_jaccard(a, b):
    overlap = intersection_duration(a, b)
    union = interval_duration(a) + interval_duration(b) - overlap
    return overlap / union if union > 0 else float("nan")


def intervals_to_mask(intervals, centers):
    """True where bin centers lie in any interval."""
    mask = np.zeros(len(centers), dtype=bool)
    for start, end in np.asarray(intervals, dtype=np.float64).reshape(-1, 2):
        left = np.searchsorted(centers, start, side="left")
        right = np.searchsorted(centers, end, side="left")
        mask[left:right] = True
    return mask


def cohen_kappa(a, b):
    """Cohen's kappa for two boolean state traces."""
    if not len(a):
        return float("nan")
    observed = float(np.mean(a == b))
    pa, pb = float(np.mean(a)), float(np.mean(b))
    expected = pa * pb + (1.0 - pa) * (1.0 - pb)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else float("nan")


def overlap_enrichment(a, b):
    """Observed simultaneous-state occupancy / independence expectation."""
    expected = float(np.mean(a)) * float(np.mean(b))
    return float(np.mean(a & b)) / expected if expected > 0 else float("nan")


def match_onsets(a, b, tolerance_sec):
    """Greedy chronological one-to-one matching of event onsets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    i = j = 0
    lags = []
    while i < len(a) and j < len(b):
        delta = b[j] - a[i]
        if abs(delta) <= tolerance_sec:
            lags.append(delta)
            i += 1
            j += 1
        elif b[j] < a[i] - tolerance_sec:
            j += 1
        else:
            i += 1
    n_match = len(lags)
    precision = n_match / len(b) if len(b) else float("nan")
    recall = n_match / len(a) if len(a) else float("nan")
    denom = len(a) + len(b)
    f1 = 2.0 * n_match / denom if denom else float("nan")
    return {
        "matches": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_abs_lag_ms": (float(np.median(np.abs(lags))) * 1000.0
                              if lags else float("nan")),
        "median_signed_lag_ms": (float(np.median(lags)) * 1000.0
                                 if lags else float("nan")),
    }


def mean_finite(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) \
        else float("nan")


def analyse(results, start, end, bin_sec, onset_tolerance_sec):
    """Build state raster, pairwise rows, and summary metrics."""
    shanks = sorted(results)
    n_bins = int(math.floor((end - start) / bin_sec))
    if n_bins < 1:
        raise ValueError("Selected time range is shorter than one analysis bin")
    centers = start + (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_sec

    off_intervals, on_intervals = {}, {}
    off_masks, on_masks = {}, {}
    state = np.zeros((len(shanks), n_bins), dtype=np.uint8)
    for row, shank in enumerate(shanks):
        result = results[shank]
        off = clipped_intervals(result.get("global_off_intervals", []), start, end)
        on = clipped_intervals(result.get("global_on_intervals", []), start, end)
        off_intervals[shank], on_intervals[shank] = off, on
        off_masks[shank] = intervals_to_mask(off, centers)
        on_masks[shank] = intervals_to_mask(on, centers)
        state[row, on_masks[shank]] = 1
        state[row, off_masks[shank]] = 2  # OFF wins if malformed inputs overlap.

    pair_rows = []
    off_jaccard = np.eye(len(shanks), dtype=np.float64)
    off_kappa = np.eye(len(shanks), dtype=np.float64)
    for i, shank_a in enumerate(shanks):
        for j in range(i + 1, len(shanks)):
            shank_b = shanks[j]
            a_off, b_off = off_masks[shank_a], off_masks[shank_b]
            onset = match_onsets(off_intervals[shank_a][:, 0],
                                 off_intervals[shank_b][:, 0], onset_tolerance_sec)
            row = {
                "shank_a": shank_a,
                "shank_b": shank_b,
                "off_jaccard": interval_jaccard(off_intervals[shank_a],
                                                 off_intervals[shank_b]),
                "off_kappa": cohen_kappa(a_off, b_off),
                "off_overlap_enrichment": overlap_enrichment(a_off, b_off),
                "off_onset_matches": onset["matches"],
                "off_onset_f1": onset["f1"],
                "off_onset_median_abs_lag_ms": onset["median_abs_lag_ms"],
                "off_onset_median_signed_lag_ms_b_minus_a": onset["median_signed_lag_ms"],
                "on_jaccard": interval_jaccard(on_intervals[shank_a],
                                                on_intervals[shank_b]),
                "on_kappa": cohen_kappa(on_masks[shank_a], on_masks[shank_b]),
            }
            pair_rows.append(row)
            off_jaccard[i, j] = off_jaccard[j, i] = row["off_jaccard"]
            off_kappa[i, j] = off_kappa[j, i] = row["off_kappa"]

    n_off = np.sum(np.stack([off_masks[s] for s in shanks]), axis=0)
    n_shanks = len(shanks)
    summary = {
        "time_start_sec": start,
        "time_end_sec": start + n_bins * bin_sec,
        "bin_ms": bin_sec * 1000.0,
        "onset_tolerance_ms": onset_tolerance_sec * 1000.0,
        "shanks": shanks,
        "n_pairs": len(pair_rows),
        "mean_pairwise_off_jaccard": mean_finite([r["off_jaccard"] for r in pair_rows]),
        "mean_pairwise_off_kappa": mean_finite([r["off_kappa"] for r in pair_rows]),
        "mean_pairwise_off_overlap_enrichment": mean_finite(
            [r["off_overlap_enrichment"] for r in pair_rows]),
        "mean_pairwise_off_onset_f1": mean_finite([r["off_onset_f1"] for r in pair_rows]),
        "mean_pairwise_on_jaccard": mean_finite([r["on_jaccard"] for r in pair_rows]),
        "mean_pairwise_on_kappa": mean_finite([r["on_kappa"] for r in pair_rows]),
        "fraction_time_any_shank_off": float(np.mean(n_off >= 1)),
        "fraction_time_majority_shanks_off": float(
            np.mean(n_off >= math.floor(n_shanks / 2) + 1)),
        "fraction_time_all_shanks_off": float(np.mean(n_off == n_shanks)),
        "mean_fraction_shanks_off": float(np.mean(n_off / n_shanks)),
    }
    return centers, state, n_off, pair_rows, summary, off_jaccard, off_kappa


def write_pairwise_csv(rows, path):
    fields = [
        "shank_a", "shank_b", "off_jaccard", "off_kappa",
        "off_overlap_enrichment", "off_onset_matches", "off_onset_f1",
        "off_onset_median_abs_lag_ms", "off_onset_median_signed_lag_ms_b_minus_a",
        "on_jaccard", "on_kappa",
    ]
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(centers, state, n_off, shanks, off_jaccard, off_kappa,
                 summary, title, output_path, dpi):
    """State raster, population OFF fraction, and pairwise heat maps."""
    figure = plt.figure(figsize=(15, max(6.5, 2.8 + 0.38 * len(shanks))))
    grid = figure.add_gridspec(3, 2, height_ratios=[max(2.2, 0.34 * len(shanks)), 1.1, 2.4],
                               hspace=0.48, wspace=0.32)
    ax_state = figure.add_subplot(grid[0, :])
    cmap = ListedColormap([UNKNOWN_COLOR, ON_COLOR, OFF_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    extent = [centers[0] / 60.0, centers[-1] / 60.0, len(shanks) - 0.5, -0.5]
    ax_state.imshow(state, aspect="auto", interpolation="nearest", cmap=cmap,
                    norm=norm, extent=extent)
    ax_state.set_yticks(np.arange(len(shanks)), [f"shank {s}" for s in shanks])
    ax_state.set_xlabel("Time from epoch start (min)")
    ax_state.set_title(title)
    from matplotlib.patches import Patch
    ax_state.legend(handles=[Patch(color=ON_COLOR, label="accepted ON"),
                             Patch(color=OFF_COLOR, label="global OFF"),
                             Patch(color=UNKNOWN_COLOR, label="unclassified")],
                    loc="upper right", ncol=3, frameon=True, fontsize=8)

    ax_count = figure.add_subplot(grid[1, :], sharex=ax_state)
    fraction = n_off / len(shanks)
    ax_count.fill_between(centers / 60.0, 0, fraction, color=OFF_COLOR, alpha=0.75,
                          linewidth=0)
    ax_count.set_ylim(0, 1)
    ax_count.set_ylabel("fraction of\nshanks OFF")
    ax_count.set_xlabel("Time from epoch start (min)")
    ax_count.grid(axis="y", alpha=0.2)

    for ax, matrix, name, low, high in (
            (figure.add_subplot(grid[2, 0]), off_jaccard, "OFF interval Jaccard", 0, 1),
            (figure.add_subplot(grid[2, 1]), off_kappa, "OFF Cohen's kappa", -1, 1)):
        image = ax.imshow(matrix, vmin=low, vmax=high, cmap="viridis" if low == 0 else "coolwarm")
        ax.set_xticks(np.arange(len(shanks)), shanks)
        ax.set_yticks(np.arange(len(shanks)), shanks)
        ax.set_xlabel("shank")
        ax.set_ylabel("shank")
        ax.set_title(name)
        if len(shanks) <= 10:
            for row in range(len(shanks)):
                for col in range(len(shanks)):
                    value = matrix[row, col]
                    ax.text(col, row, f"{value:.2f}" if np.isfinite(value) else "NA",
                            ha="center", va="center", fontsize=7,
                            color="white" if np.isfinite(value) and value > 0.55 else "black")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    figure.text(
        0.01, 0.01,
        f"Mean pairwise OFF: Jaccard={summary['mean_pairwise_off_jaccard']:.3f}, "
        f"kappa={summary['mean_pairwise_off_kappa']:.3f}, "
        f"onset F1={summary['mean_pairwise_off_onset_f1']:.3f}; "
        f"all-shank OFF time={summary['fraction_time_all_shanks_off']:.3%}. "
        f"Raster/kappa bin={summary['bin_ms']:g} ms.",
        fontsize=8, color="0.25",
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-folder", type=Path, default=None)
    parser.add_argument("--session-name", default=None,
                        help="default: infer from session folder / result files")
    parser.add_argument("--epochs", nargs="+", choices=("pre", "post"),
                        default=("pre", "post"))
    parser.add_argument("--shanks", type=int, nargs="+", default=None)
    parser.add_argument("--tag", default=None,
                        help="substring selecting one common detector parameter variant")
    parser.add_argument("--bin-ms", type=float, default=10.0,
                        help="raster and kappa resolution (default: 10 ms)")
    parser.add_argument("--onset-tolerance-ms", type=float, default=50.0,
                        help="maximum lag for one-to-one OFF-onset matches")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: session experiment-data folder)")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bin_ms <= 0 or args.onset_tolerance_ms < 0:
        raise ValueError("--bin-ms must be > 0 and --onset-tolerance-ms must be >= 0")
    off_dir = find_off_dir(args.session_folder)
    if off_dir is None:
        raise FileNotFoundError("No MUA/off_states folder found; run detect_off_states.py first")

    session_name = args.session_name
    if session_name is None and args.session_folder is not None:
        session_name = args.session_folder.name
    # Keep downstream products with the animal-day by default.  ``off_dir`` is
    # <session>/MUA/off_states, so parents[1] is the session's experiment-data
    # folder itself.  An explicit --out remains useful for scratch/test runs.
    session_dir = off_dir.parents[1]
    output_dir = Path(args.out) if args.out else Path(resolve_output_folder(session_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"OFF-state folder: {off_dir}")
    print(f"Output folder   : {output_dir}")

    completed = 0
    for epoch in args.epochs:
        variants = discover_variants(off_dir, session_name, epoch, args.tag)
        tag, paths = choose_variant(variants, args.shanks)
        if len(paths) < 2:
            print(f"[{epoch}] need at least two shanks with one common tag; found {sorted(paths)}")
            continue
        results = load_results(paths)
        durations = [float(result["duration_sec"]) for result in results.values()]
        start = max(0.0, args.start_sec)
        end = min(durations) if args.end_sec is None else min(args.end_sec, min(durations))
        if end <= start:
            raise ValueError(f"[{epoch}] invalid common time range {start:g}-{end:g} s")
        shanks = sorted(results)
        print(f"[{epoch}] tag={tag}; shanks={shanks}; common range={start:.1f}-{end:.1f}s")

        analysis = analyse(results, start, end, args.bin_ms / 1000.0,
                           args.onset_tolerance_ms / 1000.0)
        centers, state, n_off, rows, summary, jaccard, kappa = analysis
        inferred_session = next(iter(results.values())).get("session") or session_name or "session"
        summary.update({
            "session": inferred_session,
            "epoch": epoch,
            "parameter_tag": tag,
            "source_files": {str(shank): str(paths[shank]) for shank in shanks},
            "interpretation": {
                "off_jaccard": "exact shared OFF duration / union OFF duration; 1 is identical",
                "off_kappa": "chance-corrected agreement of binned OFF/not-OFF; 1 is identical, 0 is chance",
                "off_overlap_enrichment": "simultaneous OFF occupancy divided by independence expectation; >1 is excess overlap",
                "off_onset_f1": "one-to-one OFF onset agreement within onset_tolerance_ms; 1 is perfect",
                "unclassified": "time outside accepted ON and OFF intervals; it is not assumed ON",
            },
        })
        stem = f"{inferred_session}_{epoch}_{tag}_on_off_synchrony"
        csv_path = output_dir / f"{stem}_pairwise.csv"
        json_path = output_dir / f"{stem}.json"
        figure_path = output_dir / f"{stem}.png"
        write_pairwise_csv(rows, csv_path)
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=True))
        plot_results(centers, state, n_off, shanks, jaccard, kappa, summary,
                     f"{inferred_session} {epoch}: cross-shank ON/OFF timing ({tag})",
                     figure_path, args.dpi)
        print(f"  mean OFF Jaccard={summary['mean_pairwise_off_jaccard']:.3f}, "
              f"kappa={summary['mean_pairwise_off_kappa']:.3f}, "
              f"onset F1={summary['mean_pairwise_off_onset_f1']:.3f}")
        print(f"  wrote {figure_path.name}, {csv_path.name}, {json_path.name}")
        completed += 1

    if not completed:
        raise FileNotFoundError("No epoch had at least two compatible per-shank result files")


if __name__ == "__main__":
    main()
