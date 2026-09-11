r"""Plot per-channel MUA rates from the epoch pickles find_sleep_mua.py writes.

Point it at a session folder; it finds ``MUA/*_mua_events.pkl`` itself and makes
one set of figures covering every epoch pickle it found (normally pre + post):

``*_mua_channel_rate_profile.png``
    Overall rate per channel against depth, one panel per shank, pre and post
    overlaid. This is the "is the detection sane?" figure — dead channels sit at
    zero, noisy ones stick out as spikes, and a whole shank shifting between
    epochs is visible as one curve displaced from the other.
``*_mua_channel_rate_heatmap.png``
    Per-channel rate over time: channel (in depth order) x time, one row per
    shank, pre and post side by side with a shared colour scale per shank and
    panel widths proportional to epoch duration. Sleep/wake structure shows up
    as vertical banding; a channel dying mid-epoch shows up as a row that stops.
``*_mua_channel_rate_timeseries.png``
    The same data as lines — one faint line per channel coloured by depth, with
    the across-channel mean in black. Easier than the heatmap for reading the
    size of a rate change; harder for spotting which channel it came from.
``*_mua_channel_rates.csv``
    Long-format table (epoch, shank, channel, depth, n_events, rate) behind all
    three figures.

Rates are events/second on a single channel, over the whole epoch. Note that
"presleep"/"postsleep" are recording blocks, not scored sleep — the animal wakes
and moves inside both — so these rates mix sleep and wake. Restricting to NREM
is a separate step (see detect_off_states.py's --nrem handling); nothing here
masks anything.

Time bins drop the final partial bin, so every bin covers exactly ``--bin-sec``
and no edge bin reads low purely from being short.

Usage
-----
    python plot_mua_rates.py \\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\260727\CnL46_20260727
    python plot_mua_rates.py <session> --shanks 4 5
    python plot_mua_rates.py <session> --epochs post --bin-sec 5 --smooth-sec 30
    python plot_mua_rates.py <session> --out D:\figures
"""

from __future__ import annotations

import argparse
import csv
import gc
import pickle
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


#: Epoch suffix -> (label, colour). Order fixes left-to-right panel order.
EPOCH_STYLE = {
    "pre": ("presleep", "#3B75AF"),
    "post": ("postsleep", "#E68624"),
}

#: Filename pattern written by find_sleep_mua.epoch_pkl_path().
_PKL_RE = re.compile(r"^(?P<base>.+)_(?P<suffix>pre|post)_mua_events\.pkl$", re.I)

#: Channels at or below this rate (Hz) are called dead in the printed summary.
DEAD_CHANNEL_HZ = 0.5

#: A run this long (seconds) with no events at all is reported as a dropout. At
#: the rates these recordings carry, even the quietest channel fires within a
#: few seconds, so a minute of complete silence is a dead site or a blanked
#: stretch rather than a slow channel. Measured in --bin-sec bins.
SILENT_GAP_SEC = 60.0

#: Figure margins in INCHES, plus the size of one stacked panel. Fixed in inches
#: rather than as axes fractions so a one-shank figure and an eight-shank figure
#: keep the same title clearance, label room and stamp room instead of the
#: margins shrinking as the figure grows.
MARGIN_IN = {"top": 1.05, "bottom": 0.95, "left": 1.0, "right": 0.85}
PANEL_WIDTH_IN = 6.0      # widest epoch panel; shorter epochs scale below it
PANEL_HEIGHT_IN = 2.0     # one shank's row
PANEL_GAP_IN = 0.62       # vertical gap between shanks


# =====================================================
# DISCOVERY
# =====================================================

def find_mua_pkls(session_folder: Path):
    """Locate the epoch pickles under ``<session>/MUA/``.

    Returns ``(base_name, {suffix: path})``. The MUA folder is searched both
    directly under the session folder and one level down, so passing either the
    session folder or its ``MUA`` folder works.
    """
    session_folder = Path(session_folder)
    if not session_folder.exists():
        raise FileNotFoundError(f"session folder not found: {session_folder}")

    # A session processed from a local disk keeps its results on the share, so
    # the server twin of the folder is searched first - that copy is the one
    # find_sleep_mua.py wrote.
    search_dirs = [session_folder / "MUA", session_folder]
    try:
        from server_fallback import mirror_on_backup_server
        twin = mirror_on_backup_server(session_folder)
    except Exception:
        twin = None
    if twin is not None:
        search_dirs.insert(0, twin / "MUA")
    found, base = {}, None
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*_mua_events.pkl")):
            match = _PKL_RE.match(path.name)
            if match is None:
                continue
            suffix = match.group("suffix").lower()
            if suffix in found:
                continue
            found[suffix] = path
            base = base or match.group("base")
        if found:
            break

    if not found:
        raise FileNotFoundError(
            f"no *_mua_events.pkl under {session_folder / 'MUA'} (or the folder "
            f"itself) — run find_sleep_mua.py for this session first"
        )
    ordered = {s: found[s] for s in EPOCH_STYLE if s in found}
    ordered.update({s: p for s, p in found.items() if s not in ordered})
    return base, ordered


def resolve_output_dir(session_folder: Path, explicit=None) -> Path:
    """``<session>/MUA/rate_plots/``, or the backup mirror when that is read-only."""
    if explicit is not None:
        out = Path(explicit)
        out.mkdir(parents=True, exist_ok=True)
        return out
    target = Path(session_folder) / "MUA" / "rate_plots"
    try:
        from server_fallback import resolve_output_folder
        return Path(resolve_output_folder(target))
    except Exception:
        target.mkdir(parents=True, exist_ok=True)
        return target


# =====================================================
# LOADING / SUMMARISING
# =====================================================

def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _lookup_channel(mapping, channel_id):
    """Look up a channel ID across Python/NumPy scalar and string types."""
    if channel_id in mapping:
        return mapping[channel_id]
    wanted = str(_scalar(channel_id))
    for key, value in mapping.items():
        if str(_scalar(key)) == wanted:
            return value
    raise KeyError(f"channel {channel_id!r} absent from the pickle")


def depth_sorted_channels(entry):
    """Channel IDs ordered superficial -> deep, with their depths in um.

    ``channel_locations`` is stored in acquisition order, which is not depth
    order on these probes, so plotting rows without this sort scrambles the
    depth axis.
    """
    channel_ids = [_scalar(v) for v in entry["channel_ids"]]
    locations = np.asarray(entry.get("channel_locations", []))
    if locations.ndim == 2 and locations.shape[0] == len(channel_ids):
        column = 1 if locations.shape[1] > 1 else 0
        order = np.argsort(locations[:, column], kind="stable")
        return [channel_ids[i] for i in order], locations[order, column].astype(float)
    return channel_ids, np.full(len(channel_ids), np.nan)


def longest_zero_run(rate_row):
    """Longest run of consecutive empty bins, in bins.

    A channel that simply fires slowly scatters isolated empty bins; a channel
    that drops out — a dead site, or a stretch blanked by artifact repair —
    produces one long run. The distinction is what makes this worth reporting
    separately from the mean rate.
    """
    empty = np.concatenate(([0], (rate_row == 0).astype(np.int8), [0]))
    changes = np.diff(empty)
    starts = np.flatnonzero(changes == 1)
    if starts.size == 0:
        return 0
    return int((np.flatnonzero(changes == -1) - starts).max())


def summarise_shank(entry, duration_sec, bin_sec):
    """Per-channel overall rate and binned rate-over-time for one shank.

    The event times themselves are not kept: an epoch holds tens of millions of
    them per shank, and everything plotted here is a per-channel reduction.
    """
    channel_ids, depths = depth_sorted_channels(entry)
    n_bins = int(np.floor(duration_sec / bin_sec))
    if n_bins < 1:
        raise ValueError(f"epoch of {duration_sec:.1f} s is shorter than one "
                         f"{bin_sec} s bin")
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_sec

    rate_over_time = np.zeros((len(channel_ids), n_bins), dtype=np.float32)
    counts = np.zeros(len(channel_ids), dtype=np.int64)
    stored = entry.get("channel_rate_hz", {})
    stored_mismatch = 0.0

    for row, channel_id in enumerate(channel_ids):
        times = np.asarray(_lookup_channel(entry["channel_spike_times"], channel_id),
                           dtype=np.float64)
        counts[row] = times.size
        rate_over_time[row] = np.histogram(times, bins=edges)[0] / bin_sec
        if stored:
            try:
                stored_mismatch = max(stored_mismatch, abs(
                    float(_lookup_channel(stored, channel_id)) - times.size / duration_sec))
            except KeyError:
                pass

    return {
        "channel_ids": channel_ids,
        "depths": depths,
        "n_events": counts,
        "rate_hz": counts / duration_sec,
        "rate_over_time": rate_over_time,
        "bin_centers": (np.arange(n_bins) + 0.5) * bin_sec,
        "binned_sec": n_bins * bin_sec,
        "max_silent_sec": np.array([longest_zero_run(row) * bin_sec
                                    for row in rate_over_time]),
        "stored_rate_mismatch": stored_mismatch,
        "artifact_source": entry.get("artifact_source"),
        "population_rate_hz": float(entry.get("population_rate_hz", counts.sum() / duration_sec)),
    }


def summarise_epoch(path: Path, shanks, bin_sec, verbose=True):
    """Load one epoch pickle and reduce every requested shank to summaries.

    Each shank is dropped from the loaded dict as soon as it is summarised —
    these pickles run to hundreds of MB, and the reductions are kilobytes.
    """
    if verbose:
        size_mb = path.stat().st_size / 1024 ** 2
        print(f"  loading {path.name} ({size_mb:.0f} MB) ...", flush=True)
    with path.open("rb") as file:
        data = pickle.load(file)

    available = sorted(data.get("shanks", {}))
    wanted = available if shanks is None else [s for s in shanks if s in available]
    missing = [] if shanks is None else [s for s in shanks if s not in available]
    if missing:
        print(f"  WARNING: shank(s) {missing} not in {path.name}; present: {available}")
    if not wanted:
        raise KeyError(f"none of the requested shanks are in {path.name}")

    # Epoch-level duration, falling back to the per-shank copy: every shank of
    # an epoch spans the same window, so the two always agree when both exist.
    duration = float(data.get("duration_sec",
                              data["shanks"][wanted[0]]["duration_sec"]))
    summary = {
        "path": path,
        "session": data.get("session"),
        "epoch": data.get("epoch"),
        "suffix": data.get("suffix"),
        "duration_sec": duration,
        "partial": bool(data.get("partial", False)),
        "params": dict(data.get("params", {})),
        "shanks": {},
    }
    for shank in wanted:
        summary["shanks"][shank] = summarise_shank(
            data["shanks"].pop(shank), duration, bin_sec)
        if verbose:
            entry = summary["shanks"][shank]
            print(f"    shank {shank}: {entry['n_events'].sum()} events on "
                  f"{len(entry['channel_ids'])} channels, "
                  f"{entry['rate_hz'].mean():.1f} Hz/channel")
    del data
    gc.collect()
    return summary


# =====================================================
# FIGURES
# =====================================================

def stamp_figure(figure, text):
    """Embed a reproducibility line (what made this, from what, when).

    Wrapped to the figure's own width before drawing: matplotlib does not wrap
    ``figure.text``, so one long line either runs off the canvas or, under
    ``bbox_inches="tight"``, silently widens it. ~17 monospace characters per
    inch at 5 pt.
    """
    body = (f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by plot_mua_rates.py"
            f"  |  {text}")
    width = max(80, int(figure.get_figwidth() * 17))
    figure.text(0.004, 0.004, textwrap.fill(body, width=width),
                fontsize=5, color="0.4", ha="left", va="bottom", family="monospace")


def stacked_layout(n_rows, panel_widths_in):
    """Figure size and gridspec margins for a rows=shanks, cols=epochs figure.

    Panel widths are given in inches (proportional to epoch duration), so pre
    and post share one time scale and a shorter epoch draws a narrower panel
    rather than a stretched one.
    """
    width = MARGIN_IN["left"] + MARGIN_IN["right"] + sum(panel_widths_in)
    height = (MARGIN_IN["top"] + MARGIN_IN["bottom"]
              + n_rows * PANEL_HEIGHT_IN + (n_rows - 1) * PANEL_GAP_IN)
    kwargs = {
        "left": MARGIN_IN["left"] / width,
        "right": 1.0 - MARGIN_IN["right"] / width,
        "top": 1.0 - MARGIN_IN["top"] / height,
        "bottom": MARGIN_IN["bottom"] / height,
        "hspace": PANEL_GAP_IN / PANEL_HEIGHT_IN,
        "wspace": 0.06,
    }
    return (width, height), kwargs


def suptitle_top(figure, text):
    """Title pinned just under the top edge, clear of the panel titles below."""
    figure.suptitle(text, fontsize=11, va="top",
                    y=1.0 - 0.12 / figure.get_figheight())


def _stamp_text(summaries, args, extra=""):
    sources = "  ".join(f"{s['suffix']}={s['path']}" for s in summaries.values())
    params = next(iter(summaries.values()))["params"]
    detect = (f"threshold={params.get('detect_threshold')} "
              f"sign={params.get('detect_sign')} "
              f"time_radius={params.get('detect_time_radius_msec')}ms "
              f"channel_radius={params.get('detect_channel_radius')} "
              f"scale={params.get('scale_mode')} "
              f"band={params.get('freq_min')}-{params.get('freq_max')}Hz")
    durations = "  ".join(f"{s['suffix']}={s['duration_sec']:.0f}s"
                          + (" PARTIAL" if s["partial"] else "")
                          for s in summaries.values())
    line = (f"src: {sources}  |  detect: {detect}  |  epochs: {durations}  |  "
            f"bin={args.bin_sec}s smooth={args.smooth_sec}s  |  "
            f"whole epoch, NOT restricted to scored sleep")
    if extra:
        line += f"  |  {extra}"
    return line + (f"  |  reproduce: python plot_mua_rates.py {args.session_folder}")


def _grid(n_panels, n_cols):
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, min(n_cols, n_panels)


def _any_entry(summaries, shank):
    """The first epoch's summary for a shank, whichever epoch has it."""
    for summary in summaries.values():
        if shank in summary["shanks"]:
            return summary["shanks"][shank]
    return None


def _depth_ticks(entry, every=4):
    """Row positions and depth labels for a channel axis, or None if no depths."""
    depths = entry["depths"]
    if not np.isfinite(depths).all():
        return None, None
    rows = np.arange(0, len(depths), every)
    return rows, [f"{depths[r]:.0f}" for r in rows]


def plot_rate_profile(summaries, shanks, args, output_path):
    """Overall per-channel rate against depth, one panel per shank."""
    n_rows, n_cols = _grid(len(shanks), args.profile_cols)
    figure, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True,
                                figsize=(3.0 * n_cols + 1.0, 3.6 * n_rows + 1.2))

    for ax, shank in zip(axes.ravel(), shanks):
        for suffix, summary in summaries.items():
            entry = summary["shanks"].get(shank)
            if entry is None:
                continue
            label, colour = EPOCH_STYLE.get(suffix, (suffix, "0.4"))
            depths = entry["depths"]
            y = depths if np.isfinite(depths).all() else np.arange(len(depths))
            ax.plot(entry["rate_hz"], y, "-o", ms=2.6, lw=1.0, color=colour,
                    label=f"{label} ({entry['rate_hz'].mean():.0f} Hz/ch)")
        ax.set_title(f"shank {shank}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=6, frameon=False, loc="best")
        ax.set_xlim(left=0)

    for ax in axes.ravel()[len(shanks):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("MUA rate (Hz)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("depth (um), superficial at top", fontsize=8)
    first = _any_entry(summaries, shanks[0])
    if first is not None and np.isfinite(first["depths"]).all():
        axes[0, 0].invert_yaxis()

    figure.suptitle(
        f"MUA rate per channel — {next(iter(summaries.values()))['session']}\n"
        f"one point per channel, whole epoch", fontsize=11)
    stamp_figure(figure, _stamp_text(summaries, args))
    # Bottom rect reserves the stamp's ~3 wrapped lines at 5 pt.
    figure.tight_layout(rect=[0, 0.38 / figure.get_figheight(), 1, 0.96])
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


def _panel_widths(summaries):
    """Epoch panel widths in inches, proportional to duration."""
    order = list(summaries)
    longest = max(summaries[s]["duration_sec"] for s in order)
    return [PANEL_WIDTH_IN * summaries[s]["duration_sec"] / longest for s in order]


def plot_rate_heatmap(summaries, shanks, args, output_path):
    """Channel x time rate images, one row per shank, one column per epoch."""
    order = list(summaries)
    panel_widths = _panel_widths(summaries)
    cbar_width_in = 0.16
    size, grid_kwargs = stacked_layout(len(shanks), panel_widths + [cbar_width_in])
    figure = plt.figure(figsize=size)
    grid = figure.add_gridspec(len(shanks), len(order) + 1,
                               width_ratios=panel_widths + [cbar_width_in],
                               **grid_kwargs)

    for row, shank in enumerate(shanks):
        present = [s for s in order if shank in summaries[s]["shanks"]]
        if not present:
            continue
        pooled = np.concatenate([summaries[s]["shanks"][shank]["rate_over_time"].ravel()
                                 for s in present])
        vmax = float(np.percentile(pooled, args.clip_percentile)) or 1.0
        image = None
        for col, suffix in enumerate(order):
            ax = figure.add_subplot(grid[row, col])
            entry = summaries[suffix]["shanks"].get(shank)
            if entry is None:
                ax.text(0.5, 0.5, f"shank {shank} not in {suffix}", fontsize=7,
                        color="0.5", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            data = entry["rate_over_time"]
            if args.smooth_sec > 0:
                data = gaussian_filter1d(data, args.smooth_sec / args.bin_sec, axis=1)
            image = ax.imshow(data, aspect="auto", origin="upper", cmap="magma",
                              vmin=0, vmax=vmax, interpolation="nearest",
                              extent=[0, entry["binned_sec"] / 60.0,
                                      len(entry["channel_ids"]) - 0.5, -0.5])
            label = EPOCH_STYLE.get(suffix, (suffix, None))[0]
            ax.set_title(f"shank {shank} — {label} "
                         f"({entry['rate_hz'].mean():.0f} Hz/ch)", fontsize=8)
            ax.tick_params(labelsize=7)
            ticks, tick_labels = _depth_ticks(entry)
            if ticks is not None:
                ax.set_yticks(ticks)
                ax.set_yticklabels(tick_labels if col == 0 else [])
            if col == 0:
                ax.set_ylabel("depth (um)\nsuperficial at top", fontsize=7)
            else:
                ax.set_yticklabels([])
            if row == len(shanks) - 1:
                ax.set_xlabel("time from epoch start (min)", fontsize=8)
        if image is not None:
            cax = figure.add_subplot(grid[row, len(order)])
            bar = figure.colorbar(image, cax=cax)
            bar.set_label("Hz", fontsize=7)
            bar.ax.tick_params(labelsize=6)

    suptitle_top(figure,
                 f"MUA rate over time per channel — "
                 f"{next(iter(summaries.values()))['session']}\n"
                 f"{args.bin_sec} s bins"
                 + (f", {args.smooth_sec} s Gaussian smoothing"
                    if args.smooth_sec > 0 else "")
                 + f"; colour clipped at the {args.clip_percentile:g}th "
                   f"percentile (shared within a shank)")
    stamp_figure(figure, _stamp_text(
        summaries, args, f"colour clip={args.clip_percentile:g}th pct per shank"))
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


def plot_rate_timeseries(summaries, shanks, args, output_path):
    """Rate over time as one line per channel, coloured by depth."""
    order = list(summaries)
    finite = np.concatenate([summaries[s]["shanks"][sh]["depths"]
                             for s in order for sh in shanks
                             if sh in summaries[s]["shanks"]])
    finite = finite[np.isfinite(finite)]
    norm = colors.Normalize(vmin=finite.min(), vmax=finite.max()) if finite.size \
        else colors.Normalize(0, 1)
    cmap = plt.get_cmap("viridis")

    panel_widths = _panel_widths(summaries)
    size, grid_kwargs = stacked_layout(len(shanks), panel_widths)
    figure, axes = plt.subplots(
        len(shanks), len(order), squeeze=False, sharey="row", figsize=size,
        gridspec_kw={"width_ratios": panel_widths, **grid_kwargs})

    for row, shank in enumerate(shanks):
        for col, suffix in enumerate(order):
            ax = axes[row, col]
            entry = summaries[suffix]["shanks"].get(shank)
            if entry is None:
                ax.text(0.5, 0.5, f"shank {shank} not in {suffix}", fontsize=7,
                        color="0.5", ha="center", va="center", transform=ax.transAxes)
                continue
            data = entry["rate_over_time"]
            if args.smooth_sec > 0:
                data = gaussian_filter1d(data, args.smooth_sec / args.bin_sec, axis=1)
            t_min = entry["bin_centers"] / 60.0
            for channel_row, depth in enumerate(entry["depths"]):
                colour = cmap(norm(depth)) if np.isfinite(depth) else "0.6"
                ax.plot(t_min, data[channel_row], lw=0.4, color=colour, alpha=0.55)
            ax.plot(t_min, data.mean(axis=0), lw=1.3, color="black",
                    label="mean over channels")
            label = EPOCH_STYLE.get(suffix, (suffix, None))[0]
            ax.set_title(f"shank {shank} — {label} "
                         f"({entry['rate_hz'].mean():.0f} Hz/ch)", fontsize=8)
            ax.set_xlim(0, t_min[-1])
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, lw=0.4)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, frameon=False, loc="upper right")
            if col == 0:
                ax.set_ylabel("Hz / channel", fontsize=7)
            if row == len(shanks) - 1:
                ax.set_xlabel("time from epoch start (min)", fontsize=8)

    # Colourbar in the reserved right margin, rather than stolen from the axes,
    # so the panels keep the duration-proportional widths set above.
    width, height = size
    cax = figure.add_axes([1.0 - 0.62 / width, grid_kwargs["bottom"],
                           0.16 / width, grid_kwargs["top"] - grid_kwargs["bottom"]])
    bar = figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_label("channel depth (um)", fontsize=7)
    bar.ax.tick_params(labelsize=6)

    suptitle_top(figure,
                 f"MUA rate over time, per channel — "
                 f"{next(iter(summaries.values()))['session']}\n"
                 f"one line per channel ({args.bin_sec} s bins"
                 + (f", {args.smooth_sec} s smoothing" if args.smooth_sec > 0 else "")
                 + "), black = mean across channels")
    stamp_figure(figure, _stamp_text(summaries, args))
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


# =====================================================
# TABLE / SUMMARY
# =====================================================

def write_csv(summaries, shanks, output_path):
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["session", "epoch", "suffix", "shank", "channel_id",
                         "depth_um", "n_events", "rate_hz", "max_silent_sec",
                         "duration_sec"])
        for suffix, summary in summaries.items():
            for shank in shanks:
                entry = summary["shanks"].get(shank)
                if entry is None:
                    continue
                for channel_id, depth, n, rate, silent in zip(
                        entry["channel_ids"], entry["depths"],
                        entry["n_events"], entry["rate_hz"],
                        entry["max_silent_sec"]):
                    writer.writerow([summary["session"], summary["epoch"], suffix,
                                     shank, channel_id, f"{depth:.1f}", int(n),
                                     f"{rate:.4f}", f"{silent:.1f}",
                                     f"{summary['duration_sec']:.2f}"])


def print_summary(summaries, shanks):
    """Per-shank rate statistics, and the pre/post ratio when both are present."""
    order = list(summaries)
    print("\n" + "=" * 78)
    print("PER-CHANNEL RATE SUMMARY (Hz on a single channel, whole epoch)")
    header = f'{"shank":>5} {"epoch":>6} {"n_ch":>5} {"mean":>8} {"median":>8} ' \
             f'{"min":>8} {"max":>8} {"dead":>5} {"gappy":>6} {"longest gap":>12} ' \
             f'{"artifacts":>10}'
    print(header)
    print("-" * len(header))
    for shank in shanks:
        for suffix in order:
            entry = summaries[suffix]["shanks"].get(shank)
            if entry is None:
                continue
            rates, silent = entry["rate_hz"], entry["max_silent_sec"]
            worst = int(np.argmax(silent))
            worst_label = f"{silent[worst]:.0f}s ch{entry['channel_ids'][worst]}"
            print(f"{shank:>5} {suffix:>6} {len(rates):>5} {rates.mean():>8.1f} "
                  f"{np.median(rates):>8.1f} {rates.min():>8.2f} {rates.max():>8.1f} "
                  f"{int(np.sum(rates <= DEAD_CHANNEL_HZ)):>5} "
                  f"{int(np.sum(silent >= SILENT_GAP_SEC)):>6} {worst_label:>12} "
                  f"{str(entry['artifact_source']):>10}")
    print(f"  'dead'  = channels at or below {DEAD_CHANNEL_HZ} Hz over the epoch")
    print(f"  'gappy' = channels with a run of {SILENT_GAP_SEC:.0f} s or more with "
          f"no events at all; 'longest gap' names the worst channel.")
    print("            Long gaps are dropouts (dead site, or a stretch the "
          "artifact repair blanked), not low rate.")

    if "pre" in order and "post" in order:
        print("\nPOST / PRE per-channel rate ratio")
        head = f'{"shank":>5} {"n_ch":>5} {"median":>8} {"IQR":>17} {"min":>8} {"max":>8}'
        print(head)
        print("-" * len(head))
        for shank in shanks:
            pre = summaries["pre"]["shanks"].get(shank)
            post = summaries["post"]["shanks"].get(shank)
            if pre is None or post is None:
                continue
            if [str(c) for c in pre["channel_ids"]] != [str(c) for c in post["channel_ids"]]:
                print(f"{shank:>5}  channel sets differ between epochs — skipped")
                continue
            valid = pre["rate_hz"] > DEAD_CHANNEL_HZ
            if not valid.any():
                continue
            ratio = post["rate_hz"][valid] / pre["rate_hz"][valid]
            q1, q3 = np.percentile(ratio, [25, 75])
            print(f"{shank:>5} {int(valid.sum()):>5} {np.median(ratio):>8.3f} "
                  f"{f'{q1:.3f}-{q3:.3f}':>17} {ratio.min():>8.3f} {ratio.max():>8.3f}")
        print(f"  computed over channels above {DEAD_CHANNEL_HZ} Hz in presleep")

    mismatch = max((e["stored_rate_mismatch"] for s in summaries.values()
                    for e in s["shanks"].values()), default=0.0)
    ok = mismatch < 1e-6
    print(f"\n  [{'PASS' if ok else 'FAIL'}] recomputed rates match the pickle's "
          f"channel_rate_hz (max diff {mismatch:.2e} Hz)")
    if not ok:
        print("         -> the stored per-channel rates disagree with the stored "
              "spike times; the pickle may have been written by a different version.")
    print("=" * 78)


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_folder", type=Path,
                        help=r"session folder holding MUA\*_mua_events.pkl "
                             r"(e.g. \\10.129.151.88\...\CnL46_20260727)")
    parser.add_argument("--epochs", nargs="+", default=None,
                        choices=sorted(EPOCH_STYLE),
                        help="epochs to plot (default: every pickle found)")
    parser.add_argument("--shanks", type=int, nargs="+", default=None,
                        help="shanks to plot (default: every shank in the pickles)")
    parser.add_argument("--bin-sec", type=float, default=1.0,
                        help="time bin for the rate-over-time figures (default 1.0)")
    parser.add_argument("--smooth-sec", type=float, default=10.0,
                        help="Gaussian smoothing of the rate traces, seconds; "
                             "0 disables (default 10)")
    parser.add_argument("--clip-percentile", type=float, default=99.0,
                        help="heatmap colour ceiling, as a percentile of that "
                             "shank's rates (default 99)")
    parser.add_argument("--profile-cols", type=int, default=4,
                        help="panels per row in the rate-profile figure (default 4)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", default=None,
                        help="output folder (default <session>/MUA/rate_plots)")
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    base, pkls = find_mua_pkls(args.session_folder)
    if args.epochs is not None:
        missing = [e for e in args.epochs if e not in pkls]
        if missing:
            raise SystemExit(f"no pickle for epoch(s) {missing}; found {sorted(pkls)}")
        pkls = {e: pkls[e] for e in args.epochs}

    out_dir = resolve_output_dir(args.session_folder, args.out)

    print("=" * 78)
    print(f"MUA rate plots — {base}")
    print(f"  session : {args.session_folder}")
    print(f"  output  : {out_dir}")
    for suffix, path in pkls.items():
        print(f"  {suffix:>4}    : {path.name}")
    print("=" * 78)

    summaries = {}
    for suffix, path in pkls.items():
        print(f"\n--- {suffix} ---")
        summaries[suffix] = summarise_epoch(path, args.shanks, args.bin_sec,
                                            args.verbose)
        if summaries[suffix]["partial"]:
            print(f"  WARNING: {path.name} is marked PARTIAL (written with "
                  f"--limit-sec) — it is not a full epoch")

    shanks = sorted({s for summary in summaries.values() for s in summary["shanks"]})
    print(f"\nplotting shanks {shanks}")

    figures = [
        (out_dir / f"{base}_mua_channel_rate_profile.png", plot_rate_profile),
        (out_dir / f"{base}_mua_channel_rate_heatmap.png", plot_rate_heatmap),
        (out_dir / f"{base}_mua_channel_rate_timeseries.png", plot_rate_timeseries),
    ]
    for path, function in figures:
        function(summaries, shanks, args, path)
        print(f"  wrote {path}")

    csv_path = out_dir / f"{base}_mua_channel_rates.csv"
    write_csv(summaries, shanks, csv_path)
    print(f"  wrote {csv_path}")

    print_summary(summaries, shanks)


if __name__ == "__main__":
    main()
