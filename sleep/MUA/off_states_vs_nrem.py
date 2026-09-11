r"""Test whether MUA global-OFF-period time/frequency tracks NREM state.

Pure downstream consumer of two earlier stages -- it never reads MUA spikes or
LFP directly:

  detect_off_states.py       -> ``<session>_<suffix>_sh<N>_off_states_*.pkl``
                                 (must be a WHOLE-EPOCH run, not --nrem-only /
                                 --restrict-sec, since this script needs OFF
                                 periods during both sleep and wake to compare)
  score_nrem_delta_velocity.py
  / score_nrem_epochs.py     -> ``<session>_<suffix>_sleep_periods.pkl``
                                 (default; --nrem-source delta-velocity)
  score_states_from_thresholds.py
                              -> ``<session>_<suffix>_sh<N>_ch<NNN>
                                 _state_scoring.pkl`` (--nrem-source thresholds
                                 --state-shank N --state-channel NNN)

Both `sleep_periods.pkl` scorers write ``epoch_times`` / ``epoch_sec`` /
``nrem_epoch_mask`` (bool) / ``sw_index`` (a continuous NREM-depth score: the
delta-gamma tilt for score_nrem_epochs.py, the pooled log-delta z-score for
score_nrem_delta_velocity.py -- higher is always more NREM-like) on one shared
epoch grid, so this script only needs that grid, never caring which one
produced it. ``--nrem-source thresholds`` adapts a THIRD, independent scorer
onto the same shape: ``nrem_epoch_mask`` becomes ``states == "NREM"`` (its
4-way NREM/REM/WAKE/QUIET labels collapsed to the binary this analysis needs)
and ``sw_index`` becomes that channel's own oriented PC1 trace -- its
continuous NREM-depth analogue, already sign-corrected against delta power by
that script.

What it does, per (epoch suffix, shank)
----------------------------------------
1. Bins ``global_off_intervals`` onto the NREM-scoring epoch grid: per bin,
   how many OFF periods START in it (rate) and what fraction of the bin they
   COVER (time). The two are reported separately because they can move
   independently -- e.g. OFF periods could get longer without getting more
   frequent.
2. Compares both quantities between NREM and non-NREM bins (Mann-Whitney U;
   the epochs are not independent samples, so read the p-value as
   descriptive, not as a formal test).
3. Correlates both quantities against the continuous ``sw_index`` (Spearman),
   which is the more graded question: does OFF activity scale with NREM
   *depth*, not just its presence.
4. Repeats the same two comparisons pooled across every requested shank
   (mean of each shank's own rate/fraction on the shared grid), since a
   single-shank result could be a probe-placement quirk.

Usage
-----
    python off_states_vs_nrem.py                    # every shank, pre + post
    python off_states_vs_nrem.py --shanks 5 --epochs post
    python off_states_vs_nrem.py --list              # what is discoverable
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent          # .../sleep/MUA
_SLEEP = _HERE.parent                            # .../sleep
_ROOT = _SLEEP.parent                            # repo root (server_fallback.py)
for _p in (str(_HERE), str(_SLEEP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server_fallback import resolve_output_folder  # noqa: E402
import detect_off_states as dos  # noqa: E402

NREM_SHADE = "mediumseagreen"    # matches score_nrem_delta_velocity.py's own figures
OFF_COLOR = dos.PURPLE           # matches detect_off_states.py's "global OFF" color
SW_COLOR = "0.25"


# =====================================================
# DISCOVERY
# =====================================================

def resolve_off_states_dir(folders):
    """First ``<folder>/MUA/off_states`` that exists among the candidates."""
    for folder in folders:
        candidate = folder / "MUA" / "off_states"
        if candidate.is_dir():
            return candidate
    return None


_SHANK_RE = re.compile(r"_sh(\d+)_off_states_")


def discover_shanks(off_dir, names, suffix):
    """Shank numbers with at least one off_states pkl for this suffix."""
    shanks = set()
    for name in names:
        for path in off_dir.glob(f"{name}_{suffix}_sh*_off_states_*.pkl"):
            match = _SHANK_RE.search(path.name)
            if match:
                shanks.add(int(match.group(1)))
    return sorted(shanks)


def find_off_states_result(off_dir, names, suffix, shank, tag_filter=None):
    """The newest WHOLE-EPOCH off_states pkl for one shank.

    Several parameter variants can exist per shank (different
    --global-fraction / --flank-mode runs); the newest by mtime wins unless
    ``tag_filter`` narrows the filename first. A variant that was itself
    restricted to NREM (``--nrem-only`` / ``--restrict-sec``) is unusable here
    -- this analysis needs OFF periods across the WHOLE epoch to compare NREM
    against non-NREM -- so those are skipped rather than silently used.

    Returns ``(result, path, skipped_notes)``; ``result`` is None when
    nothing usable was found.
    """
    candidates = []
    for name in names:
        candidates += list(off_dir.glob(f"{name}_{suffix}_sh{shank}_off_states_*.pkl"))
    if tag_filter:
        candidates = [c for c in candidates if tag_filter in c.stem]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    notes = []
    for path in candidates:
        with path.open("rb") as file:
            result = pickle.load(file)
        if result["params"].get("nrem_restricted"):
            notes.append(f"{path.name}: skipped (NREM-restricted; need a whole-epoch run)")
            continue
        return result, path, notes
    return None, None, notes


def resolve_state_scoring_pkl(folders, names, suffix, state_shank, state_channel):
    """``<name>_<suffix>_sh<state_shank>_ch<state_channel:03d>_state_scoring.pkl``.

    Written by score_states_from_thresholds.py, under ``low_freq/state_scoring``
    -- a sibling of ``low_freq/<name>_<suffix>_sleep_periods.pkl``, not under
    MUA/off_states, so it is resolved independently rather than through
    :func:`resolve_off_states_dir`.
    """
    return dos._first_existing([
        folder / "low_freq" / "state_scoring" /
        f"{name}_{suffix}_sh{state_shank}_ch{state_channel:03d}_state_scoring.pkl"
        for folder in folders for name in names
    ])


def periods_from_state_scoring(path):
    """Adapt a score_states_from_thresholds.py pkl into the periods-dict shape
    :func:`build_bin_table` expects: ``epoch_sec`` / ``epoch_times`` /
    ``nrem_epoch_mask`` / ``sw_index`` / ``sleep_start_sample``.

    The 4-way ``states`` label collapses to NREM-vs-everything-else here --
    REM, WAKE and QUIET are all "not NREM" for this analysis, which is only
    asking about the slow-wave state OFF periods are a hallmark of. The
    per-channel oriented ``pc1`` trace stands in for ``sw_index``: it is
    already continuous and already sign-corrected so high = more NREM-like,
    exactly what ``sw_index`` is for the other two scorers.
    """
    with Path(path).open("rb") as file:
        state = pickle.load(file)
    states = np.asarray(state["states"])
    return {
        "session": state.get("session"),
        "epoch_sec": float(state["epoch_sec"]),
        "epoch_times": np.asarray(state["epoch_times"], dtype=np.float64),
        "nrem_epoch_mask": states == "NREM",
        "sw_index": np.asarray(state["pc1"], dtype=np.float64),
        "sleep_start_sample": state.get("sleep_start_sample"),
    }, state


# =====================================================
# BINNING ONTO THE NREM-SCORING EPOCH GRID
# =====================================================

def bin_off_intervals(intervals, bin_starts, bin_sec):
    """Per-bin ``(onset_count, covered_seconds)`` against a uniform grid.

    ``bin_starts`` are left edges, spaced ``bin_sec`` apart, on the SAME clock
    as ``intervals`` (MUA-epoch seconds). ``onset_count`` keys each OFF period
    by its MIDPOINT -- the rate at which OFF periods start. ``covered_seconds``
    is the exact overlap; at 50-400 ms an OFF period spans at most two or three
    scoring-epoch bins, so a plain per-interval loop stays cheap even over a
    whole session's worth of them.
    """
    n_bins = len(bin_starts)
    onset_count = np.zeros(n_bins, dtype=np.int64)
    covered = np.zeros(n_bins, dtype=np.float64)
    if len(intervals) == 0:
        return onset_count, covered

    t0 = bin_starts[0]
    midpoints = intervals.mean(axis=1)
    idx = np.floor((midpoints - t0) / bin_sec).astype(np.int64)
    valid = (idx >= 0) & (idx < n_bins)
    np.add.at(onset_count, idx[valid], 1)

    bin_ends = bin_starts + bin_sec
    for start, end in intervals:
        k0 = max(0, int(np.floor((start - t0) / bin_sec)))
        k1 = min(n_bins - 1, int(np.floor((end - t0) / bin_sec)))
        for k in range(k0, k1 + 1):
            left, right = max(start, bin_starts[k]), min(end, bin_ends[k])
            if right > left:
                covered[k] += right - left
    return onset_count, covered


def build_bin_table(off_result, periods):
    """One row per NREM-scoring epoch: OFF rate/time-fraction, NREM label, sw_index.

    ``off_result`` doubles as the ``mua`` argument :func:`detect_off_states.
    clock_offset_sec` expects -- it already carries ``sampling_frequency``,
    ``epoch_start_sample`` and ``duration_sec`` from the MUA pkl it was built
    from, so no separate MUA load is needed here.
    """
    bin_sec = float(periods["epoch_sec"])
    n = min(len(np.asarray(periods["epoch_times"])),
            len(np.asarray(periods["nrem_epoch_mask"])),
            len(np.asarray(periods["sw_index"])))
    epoch_times = np.asarray(periods["epoch_times"], dtype=np.float64)[:n]
    offset = dos.clock_offset_sec(periods, off_result)
    bin_starts = epoch_times - bin_sec / 2.0 + offset

    intervals = np.asarray(off_result["global_off_intervals"],
                           dtype=np.float64).reshape(-1, 2)
    off_count, off_time = bin_off_intervals(intervals, bin_starts, bin_sec)

    duration = float(off_result["duration_sec"])
    in_range = (bin_starts >= 0.0) & (bin_starts + bin_sec <= duration)

    return {
        "bin_start_s": bin_starts,
        "bin_center_s": bin_starts + bin_sec / 2.0,
        "bin_sec": bin_sec,
        "nrem": np.asarray(periods["nrem_epoch_mask"], dtype=bool)[:n],
        "sw_index": np.asarray(periods["sw_index"], dtype=np.float64)[:n],
        "off_count": off_count,
        "off_time_s": off_time,
        "off_frac": off_time / bin_sec,
        "off_rate_per_min": off_count / bin_sec * 60.0,
        "in_range": in_range,
    }


def pool_shanks(tables):
    """Mean OFF fraction/rate across shanks on their shared epoch grid.

    The grid (bin times, NREM label, sw_index) is identical across shanks --
    one NREM scoring run per session, not per shank -- so only the two OFF
    metrics need combining. Uses the first table's grid as the reference and
    nan-means each shank in, dropping a bin from the pooled average wherever
    ANY shank has it out of range.
    """
    ref = next(iter(tables.values()))
    n = len(ref["bin_center_s"])
    frac_stack = np.full((len(tables), n), np.nan)
    rate_stack = np.full((len(tables), n), np.nan)
    in_range_all = np.ones(n, dtype=bool)
    for row, table in enumerate(tables.values()):
        frac_stack[row] = np.where(table["in_range"], table["off_frac"], np.nan)
        rate_stack[row] = np.where(table["in_range"], table["off_rate_per_min"], np.nan)
        in_range_all &= table["in_range"]
    with np.errstate(invalid="ignore"):
        return {
            "bin_start_s": ref["bin_start_s"],
            "bin_center_s": ref["bin_center_s"],
            "bin_sec": ref["bin_sec"],
            "nrem": ref["nrem"],
            "sw_index": ref["sw_index"],
            "off_frac": np.nanmean(frac_stack, axis=0),
            "off_rate_per_min": np.nanmean(rate_stack, axis=0),
            "in_range": in_range_all,
            "n_shanks": len(tables),
        }


# =====================================================
# STATISTICS
# =====================================================

def compare_by_state(table, mask):
    """Mann-Whitney U on ``off_frac`` / ``off_rate_per_min``, NREM vs non-NREM bins."""
    valid = mask & np.isfinite(table["off_frac"]) & np.isfinite(table["off_rate_per_min"])
    nrem = table["nrem"][valid]
    results = {}
    for key in ("off_frac", "off_rate_per_min"):
        values = table[key][valid]
        a, b = values[nrem], values[~nrem]
        row = {
            "n_nrem": int(a.size), "n_non_nrem": int(b.size),
            "median_nrem": float(np.median(a)) if a.size else float("nan"),
            "median_non_nrem": float(np.median(b)) if b.size else float("nan"),
        }
        if a.size and b.size:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            # Rank-biserial effect size, signed so +1 means every NREM bin
            # outranks every non-NREM bin, -1 the reverse, 0 full overlap.
            # scipy's U for (a, b) is the count of pairs with a_i > b_j (up to
            # ties), so it is n_a*n_b when a is uniformly higher -- verified
            # empirically, since the textbook formula's sign varies by source
            # depending on which of U1/U2 it assumes.
            row["mannwhitney_u"] = float(u)
            row["p_value"] = float(p)
            row["rank_biserial"] = float(2.0 * u / (a.size * b.size) - 1.0)
        else:
            row["mannwhitney_u"] = row["p_value"] = row["rank_biserial"] = float("nan")
        results[key] = row
    return results


def correlate_with_sw_index(table, mask):
    """Spearman correlation of ``off_frac`` / ``off_rate_per_min`` against ``sw_index``."""
    valid = mask & np.isfinite(table["sw_index"])
    sw = table["sw_index"][valid]
    results = {}
    for key in ("off_frac", "off_rate_per_min"):
        values = table[key][valid]
        finite = valid.copy()
        finite[valid] = np.isfinite(values)
        n = int(finite.sum())
        if n >= 3:
            rho, p = stats.spearmanr(table["sw_index"][finite], table[key][finite])
        else:
            rho, p = float("nan"), float("nan")
        results[key] = {"n": n, "spearman_r": float(rho), "p_value": float(p)}
    return results


# =====================================================
# OUTPUT
# =====================================================

def write_bin_csv(table, path, pooled=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["bin_start_s", "bin_center_s", "nrem", "sw_index",
              "off_frac", "off_rate_per_min"]
    if not pooled:
        fields += ["off_count", "off_time_s"]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for i in np.flatnonzero(table["in_range"]):
            row = [table["bin_start_s"][i], table["bin_center_s"][i],
                  bool(table["nrem"][i]), table["sw_index"][i],
                  table["off_frac"][i], table["off_rate_per_min"][i]]
            if not pooled:
                row += [int(table["off_count"][i]), table["off_time_s"][i]]
            writer.writerow(row)


def smooth(values, width):
    """Centred, NaN-ignoring boxcar -- display only, never used for statistics."""
    values = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return values
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones(int(width))
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(finite.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out


def bool_spans(mask, t):
    """Contiguous True runs of ``mask`` as ``(t_start, t_end)`` pairs."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    return [(t[s], t[e - 1]) for s, e in zip(starts, ends)]


def box_by_state(ax, values, nrem, ylabel, stat_row):
    groups = [values[nrem][np.isfinite(values[nrem])],
              values[~nrem][np.isfinite(values[~nrem])]]
    box = ax.boxplot(groups, tick_labels=["NREM", "non-NREM"], showfliers=False,
                     patch_artist=True, widths=0.55)
    for patch, color in zip(box["boxes"], (NREM_SHADE, "0.65")):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"Mann-Whitney p={stat_row['p_value']:.2g}, "
                 f"rank-biserial={stat_row['rank_biserial']:.2f}", fontsize=8.5)
    ax.tick_params(labelsize=8)


def scatter_with_fit(ax, sw, values, nrem, ylabel, corr_row, sw_label):
    ax.scatter(sw[~nrem], values[~nrem], s=4, color="0.6", alpha=0.35,
              linewidths=0, label="non-NREM")
    ax.scatter(sw[nrem], values[nrem], s=4, color=NREM_SHADE, alpha=0.55,
              linewidths=0, label="NREM")
    finite = np.isfinite(sw) & np.isfinite(values)
    if finite.sum() >= 2:
        slope, intercept = np.polyfit(sw[finite], values[finite], 1)
        xs = np.linspace(sw[finite].min(), sw[finite].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="black", lw=1.2, zorder=5)
    ax.set_xlabel(sw_label, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"Spearman r={corr_row['spearman_r']:.2f}, "
                 f"p={corr_row['p_value']:.2g}, n={corr_row['n']}", fontsize=8.5)
    ax.tick_params(labelsize=8)


def plot_off_vs_nrem(table, by_state, corr, *, title, output_path, stamp_text,
                     smooth_bins, dpi, sw_label="sw_index (continuous NREM depth, z)"):
    in_range = table["in_range"]
    t_min = table["bin_center_s"][in_range] / 60.0
    nrem = table["nrem"][in_range]
    off_frac = table["off_frac"][in_range]
    off_rate = table["off_rate_per_min"][in_range]
    sw = table["sw_index"][in_range]
    spans = bool_spans(nrem, t_min)

    figure = plt.figure(figsize=(14, 13))
    grid = figure.add_gridspec(5, 2, height_ratios=(0.25, 1.3, 1.3, 1.5, 1.7),
                               hspace=0.55, wspace=0.28,
                               left=0.07, right=0.95, top=0.93, bottom=0.05)

    def shade(ax):
        for a, b in spans:
            ax.axvspan(a, b, color=NREM_SHADE, alpha=0.15, lw=0, zorder=0)

    ax_bar = figure.add_subplot(grid[0, :])
    for a, b in spans:
        ax_bar.axvspan(a, b, color=NREM_SHADE, alpha=0.9, lw=0)
    ax_bar.set_xlim(t_min.min(), t_min.max())
    ax_bar.set_yticks([]); ax_bar.set_xticks([])
    ax_bar.set_ylabel("NREM", rotation=0, ha="right", va="center", fontsize=9)
    ax_bar.set_title(title, fontsize=12)

    ax_frac = figure.add_subplot(grid[1, :], sharex=ax_bar)
    shade(ax_frac)
    ax_frac.plot(t_min, smooth(off_frac, smooth_bins) * 100, color=OFF_COLOR, lw=1.0)
    ax_frac.set_ylabel("OFF time\n(% of bin)", fontsize=9, color=OFF_COLOR)
    ax_frac.tick_params(axis="y", labelcolor=OFF_COLOR, labelsize=8)
    ax_frac.tick_params(labelbottom=False)
    ax_sw1 = ax_frac.twinx()
    ax_sw1.plot(t_min, smooth(sw, smooth_bins), color=SW_COLOR, lw=0.7, alpha=0.75)
    ax_sw1.set_ylabel(sw_label.split(" (")[0], fontsize=9, color=SW_COLOR)
    ax_sw1.tick_params(axis="y", labelcolor=SW_COLOR, labelsize=8)

    ax_rate = figure.add_subplot(grid[2, :], sharex=ax_bar)
    shade(ax_rate)
    ax_rate.plot(t_min, smooth(off_rate, smooth_bins), color=OFF_COLOR, lw=1.0)
    ax_rate.set_ylabel("OFF rate\n(onsets/min)", fontsize=9, color=OFF_COLOR)
    ax_rate.tick_params(axis="y", labelcolor=OFF_COLOR, labelsize=8)
    ax_rate.set_xlabel("Time (minutes)", fontsize=9)
    ax_sw2 = ax_rate.twinx()
    ax_sw2.plot(t_min, smooth(sw, smooth_bins), color=SW_COLOR, lw=0.7, alpha=0.75)
    ax_sw2.set_ylabel(sw_label.split(" (")[0], fontsize=9, color=SW_COLOR)
    ax_sw2.tick_params(axis="y", labelcolor=SW_COLOR, labelsize=8)

    box_by_state(figure.add_subplot(grid[3, 0]), off_frac, nrem,
                "OFF time fraction", by_state["off_frac"])
    box_by_state(figure.add_subplot(grid[3, 1]), off_rate, nrem,
                "OFF rate (onsets/min)", by_state["off_rate_per_min"])

    ax_sc1 = figure.add_subplot(grid[4, 0])
    scatter_with_fit(ax_sc1, sw, off_frac, nrem, "OFF time fraction",
                     corr["off_frac"], sw_label)
    ax_sc1.legend(fontsize=7, loc="upper left", markerscale=2)
    scatter_with_fit(figure.add_subplot(grid[4, 1]), sw, off_rate, nrem,
                     "OFF rate (onsets/min)", corr["off_rate_per_min"], sw_label)

    dos.stamp_figure(figure, stamp_text, script="off_states_vs_nrem.py")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-folder", type=Path, default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--epochs", nargs="+", default=None,
                        help="epoch suffixes to process (default: pre post)")
    parser.add_argument("--shanks", type=int, nargs="+", default=None,
                        help="default: every shank with a whole-epoch off_states pkl")
    parser.add_argument("--tag", default=None,
                        help="substring filter to pick one off_states parameter "
                             "variant when several exist per shank (e.g. 'f75_strict')")
    parser.add_argument("--nrem-source", choices=("delta-velocity", "thresholds"),
                        default="delta-velocity",
                        help="'delta-velocity' (default): <suffix>_sleep_periods.pkl "
                             "from score_nrem_delta_velocity.py / score_nrem_epochs.py. "
                             "'thresholds': one channel's PC1/theta/velocity hypnogram "
                             "from score_states_from_thresholds.py (needs --state-channel)")
    parser.add_argument("--state-shank", type=int, default=0,
                        help="shank the --nrem-source thresholds pkl was scored on "
                             "(default 0, matching score_states_from_thresholds.py)")
    parser.add_argument("--state-channel", type=int, default=None,
                        help="channel the --nrem-source thresholds pkl was scored on "
                             "(required when --nrem-source thresholds)")
    parser.add_argument("--list", action="store_true",
                        help="report what is discoverable and exit")
    parser.add_argument("--smooth-bins", type=int, default=15,
                        help="boxcar width, in NREM-scoring epochs, for the "
                             "DISPLAYED time-series traces only -- statistics "
                             "always use the raw per-epoch values (default 15)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def summary_row(epoch, shank, table, by_state, corr, off_label, periods_path):
    frac, rate = by_state["off_frac"], by_state["off_rate_per_min"]
    cf, cr = corr["off_frac"], corr["off_rate_per_min"]
    return {
        "epoch": epoch, "shank": shank, "n_bins": int(table["in_range"].sum()),
        "off_frac_median_nrem": frac["median_nrem"],
        "off_frac_median_non_nrem": frac["median_non_nrem"],
        "off_frac_mwu_p": frac["p_value"], "off_frac_rank_biserial": frac["rank_biserial"],
        "off_rate_median_nrem": rate["median_nrem"],
        "off_rate_median_non_nrem": rate["median_non_nrem"],
        "off_rate_mwu_p": rate["p_value"], "off_rate_rank_biserial": rate["rank_biserial"],
        "off_frac_vs_sw_rho": cf["spearman_r"], "off_frac_vs_sw_p": cf["p_value"],
        "off_rate_vs_sw_rho": cr["spearman_r"], "off_rate_vs_sw_p": cr["p_value"],
        "off_states_source": off_label, "nrem_source_pkl": periods_path.name,
    }


def main():
    args = parse_args()
    if args.nrem_source == "thresholds" and args.state_channel is None:
        raise SystemExit("--nrem-source thresholds requires --state-channel")

    folders = dos.candidate_session_folders(args.session_folder)
    names = dos.candidate_session_names(args.session_name)
    epochs = args.epochs or ["pre", "post"]

    print("=" * 72)
    print("OFF-period vs NREM-state analysis")
    print(f"  session folders searched : {[str(f) for f in folders]}")
    print(f"  session names tried      : {names}")
    print(f"  NREM source              : {args.nrem_source}"
          + (f"  (shank {args.state_shank}, channel {args.state_channel})"
             if args.nrem_source == "thresholds" else ""))

    off_dir = resolve_off_states_dir(folders)
    print(f"  off_states folder        : {off_dir}")

    # sw_label: what the continuous NREM-depth trace actually is, for the
    # figure's x-axis/twin-axis -- different per source (see module docstring).
    sw_label = ("sw_index (continuous NREM depth, z)" if args.nrem_source == "delta-velocity"
               else f"PC1 ch{args.state_channel} (oriented, continuous NREM depth)")
    # File tag so a --nrem-source thresholds run never collides with the
    # default delta-velocity output for the same suffix/shank.
    source_tag = ("" if args.nrem_source == "delta-velocity"
                  else f"_thrsh{args.state_shank}ch{args.state_channel}")

    if args.list:
        for suffix in epochs:
            periods_path = dos.resolve_sleep_periods(suffix, folders, names)
            state_path = resolve_state_scoring_pkl(
                folders, names, suffix, args.state_shank,
                args.state_channel if args.state_channel is not None else 0)
            shanks = discover_shanks(off_dir, names, suffix) if off_dir else []
            print(f"\n[{suffix}] off_states shanks: {shanks}")
            print(f"  sleep periods (delta-velocity): {periods_path}")
            print(f"  state scoring (thresholds, shank {args.state_shank}, "
                  f"channel {args.state_channel if args.state_channel is not None else 0}): "
                  f"{state_path}")
        return

    if off_dir is None:
        raise FileNotFoundError(
            "No MUA/off_states folder found among the candidate session folders; "
            "run detect_off_states.py first."
        )

    summary_rows = []
    out_dir = None
    for suffix in epochs:
        print("\n" + "#" * 72)
        print(f"EPOCH {suffix}")

        if args.nrem_source == "thresholds":
            periods_path = resolve_state_scoring_pkl(
                folders, names, suffix, args.state_shank, args.state_channel)
            if periods_path is None:
                print(f"  no state_scoring pkl found for {suffix} shank "
                      f"{args.state_shank} channel {args.state_channel} -- skip "
                      f"(run score_states_from_thresholds.py --shank "
                      f"{args.state_shank} --channels {args.state_channel} first)")
                continue
            periods, _raw_state = periods_from_state_scoring(periods_path)
        else:
            periods_path = dos.resolve_sleep_periods(suffix, folders, names)
            if periods_path is None:
                print(f"  no sleep_periods pkl found for {suffix} -- skip "
                      f"(run score_nrem_delta_velocity.py --build or score_nrem_epochs.py)")
                continue
            with periods_path.open("rb") as file:
                periods = pickle.load(file)
        print(f"  NREM scoring: {periods_path}")

        shanks = args.shanks if args.shanks is not None else discover_shanks(off_dir, names, suffix)
        if not shanks:
            print(f"  no off_states pkl found for {suffix} -- skip")
            continue

        out_dir = resolve_output_folder(off_dir / "vs_nrem")
        tables = {}
        session_label = names[0]
        for shank in shanks:
            result, off_path, notes = find_off_states_result(
                off_dir, names, suffix, shank, args.tag)
            for note in notes:
                print(f"    shank {shank}: {note}")
            if result is None:
                print(f"    shank {shank}: no whole-epoch off_states pkl found -- skip")
                continue
            print(f"    shank {shank}: {off_path.name}")

            table = build_bin_table(result, periods)
            tables[shank] = table
            session_label = result["session"]
            by_state = compare_by_state(table, table["in_range"])
            corr = correlate_with_sw_index(table, table["in_range"])
            write_bin_csv(table, out_dir /
                          f"{result['session']}_{suffix}_sh{shank}_off_vs_nrem{source_tag}.csv")

            frac, rate = by_state["off_frac"], by_state["off_rate_per_min"]
            print(f"      off_frac  NREM={frac['median_nrem']:.3f} "
                  f"non-NREM={frac['median_non_nrem']:.3f}  "
                  f"MWU p={frac['p_value']:.2g}  rank-biserial={frac['rank_biserial']:.2f}")
            print(f"      off_rate  NREM={rate['median_nrem']:.1f}/min "
                  f"non-NREM={rate['median_non_nrem']:.1f}/min  "
                  f"MWU p={rate['p_value']:.2g}  rank-biserial={rate['rank_biserial']:.2f}")
            print(f"      vs sw_index  frac: rho={corr['off_frac']['spearman_r']:.2f} "
                  f"p={corr['off_frac']['p_value']:.2g}   "
                  f"rate: rho={corr['off_rate_per_min']['spearman_r']:.2f} "
                  f"p={corr['off_rate_per_min']['p_value']:.2g}")

            summary_rows.append(summary_row(suffix, shank, table, by_state, corr,
                                            off_path.name, periods_path))

            if not args.no_plot:
                fig_path = out_dir / f"{result['session']}_{suffix}_sh{shank}_off_vs_nrem{source_tag}.png"
                extra_args = (f"--nrem-source thresholds --state-shank {args.state_shank} "
                             f"--state-channel {args.state_channel}"
                             if args.nrem_source == "thresholds" else "")
                plot_off_vs_nrem(
                    table, by_state, corr,
                    title=f"{result['session']} {suffix} shank {shank}  |  "
                          f"OFF periods vs NREM state",
                    output_path=fig_path,
                    stamp_text=(
                        f"off_pkl={off_path.name}  |  nrem_pkl={periods_path.name}  |  "
                        f"bin={table['bin_sec']:g}s smooth={args.smooth_bins}bins  |  "
                        f"reproduce: python off_states_vs_nrem.py --epochs {suffix} "
                        f"--shanks {shank}{(' ' + extra_args) if extra_args else ''}"
                    ),
                    smooth_bins=args.smooth_bins, dpi=args.dpi, sw_label=sw_label,
                )
                print(f"      figure: {fig_path.name}")

        if len(tables) >= 2:
            pooled = pool_shanks(tables)
            by_state = compare_by_state(pooled, pooled["in_range"])
            corr = correlate_with_sw_index(pooled, pooled["in_range"])
            write_bin_csv(pooled, out_dir /
                          f"{session_label}_{suffix}_pooled_off_vs_nrem{source_tag}.csv",
                         pooled=True)

            frac, rate = by_state["off_frac"], by_state["off_rate_per_min"]
            print(f"\n  POOLED across {len(tables)} shanks {sorted(tables)}:")
            print(f"    off_frac  NREM={frac['median_nrem']:.3f} "
                  f"non-NREM={frac['median_non_nrem']:.3f}  "
                  f"MWU p={frac['p_value']:.2g}  rank-biserial={frac['rank_biserial']:.2f}")
            print(f"    off_rate  NREM={rate['median_nrem']:.1f}/min "
                  f"non-NREM={rate['median_non_nrem']:.1f}/min  "
                  f"MWU p={rate['p_value']:.2g}  rank-biserial={rate['rank_biserial']:.2f}")

            summary_rows.append(summary_row(
                suffix, "pooled", pooled, by_state, corr,
                f"pooled({len(tables)} shanks: {sorted(tables)})", periods_path))

            if not args.no_plot:
                fig_path = out_dir / f"{session_label}_{suffix}_pooled_off_vs_nrem{source_tag}.png"
                shank_list = " ".join(str(s) for s in sorted(tables))
                extra_args = (f"--nrem-source thresholds --state-shank {args.state_shank} "
                             f"--state-channel {args.state_channel}"
                             if args.nrem_source == "thresholds" else "")
                plot_off_vs_nrem(
                    pooled, by_state, corr,
                    title=f"{session_label} {suffix}  |  OFF periods vs NREM state "
                          f"(pooled, {len(tables)} shanks)",
                    output_path=fig_path,
                    stamp_text=(
                        f"pooled shanks={sorted(tables)}  |  nrem_pkl={periods_path.name}  |  "
                        f"bin={pooled['bin_sec']:g}s smooth={args.smooth_bins}bins  |  "
                        f"reproduce: python off_states_vs_nrem.py --epochs {suffix} "
                        f"--shanks {shank_list}{(' ' + extra_args) if extra_args else ''}"
                    ),
                    smooth_bins=args.smooth_bins, dpi=args.dpi, sw_label=sw_label,
                )
                print(f"    figure: {fig_path.name}")

    if summary_rows and out_dir is not None:
        summary_path = out_dir / f"off_vs_nrem_summary{source_tag}.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print("\n" + "=" * 72)
        print(f"{'epoch':>6} {'shank':>8} {'n_bins':>7} {'off_frac p':>11} "
              f"{'r_rb':>6} {'off_rate p':>11} {'r_rb':>6} {'rho(frac)':>10}")
        for row in summary_rows:
            print(f"{row['epoch']:>6} {str(row['shank']):>8} {row['n_bins']:>7} "
                  f"{row['off_frac_mwu_p']:>11.2g} {row['off_frac_rank_biserial']:>6.2f} "
                  f"{row['off_rate_mwu_p']:>11.2g} {row['off_rate_rank_biserial']:>6.2f} "
                  f"{row['off_frac_vs_sw_rho']:>10.2f}")
        print(f"\nSummary: {summary_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
