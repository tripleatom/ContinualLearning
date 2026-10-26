r"""Channel MUA trains vs sorted-unit trains, on the same time axis.

Both come from the same probe and the same sleep block:
- channel MUA: find_sleep_mua.py threshold crossings, one train per channel
  (256 = 8 shanks x 32), each spike counted on every channel it crosses;
- sorted units: sleep_spikes_<date>_sleep_<block>.pkl (raw sorter clusters),
  ordered by the sorter's unit location (monopolar triangulation).
Both are ordered by shank, then deep -> superficial (probe y, 0 = near surface).
PBEs shown are the MUA-based ones (detect_pbe.py --source mua).

Outputs, in <session>/PBE/figures/mua_vs_units/:
  <label>_mua_vs_units_NN_<t>s_<kind>.png  one window: 2 s and 0.5 s zoom side by
       side; rows = channel MUA raster, unit raster, pooled z of both overlaid.
  <label>_mua_vs_units_grid.png            all windows at 1 s on one page.
  <label>_mua_vs_units_stats.png           how closely pooled units track pooled MUA:
       correlation in NREM-ON vs bin size (whole probe and per shank), and
       per-shank spike counts of each.

Usage
-----
    python sleep/PBE/plot_mua_vs_units.py --day 260731 --block post
    python sleep/PBE/plot_mua_vs_units.py --day 260731 --block pre --n-random-examples 10
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_pbe as dp  # noqa: E402  (also bootstraps sleep_paths)
from plot_on_pbe import kept_events, pick_examples, shade, shank_ticks  # noqa: E402

import numpy as np  # noqa: E402

from reactivation.sleep_apply.sleep_states import SleepBlock, interval_mask  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GRATING_DIR = REPO_ROOT / "passive_visual" / "FreelyMovingProcessing" / "Grating"
PBE_COLOR = "#2b83ba"
UNIT_COLOR = "#d7301f"
ON_COLOR = "#2ca02c"


# ---------------------------------------------------------------- loading
def unit_depths(sortout_folder):
    """{"shankX_unitY": probe y (um)} from each shank's sorting_analyzer unit_locations."""
    sys.path.insert(0, str(GRATING_DIR))
    from GratingExport import _find_shank_raw_analyzers
    from spikeinterface import load_sorting_analyzer
    depth = {}
    for shank, path in _find_shank_raw_analyzers(Path(sortout_folder)):
        loc_file = Path(path) / "extensions" / "unit_locations" / "unit_locations.npy"
        if not loc_file.is_file():
            continue
        ids = list(load_sorting_analyzer(path, load_extensions=False).unit_ids)
        loc = np.load(loc_file)
        for uid, row in zip(ids, loc):
            depth[f"shank{shank}_unit{uid}"] = float(row[1])
    return depth


def load_inputs(animal, day, block):
    paths = dp.block_paths(animal, day, block)
    label = f"{paths['session']}_{block}"
    with (Path(paths["out_dir"]) / f"{label}_pbe_mua.pkl").open("rb") as stream:
        pbe = pickle.load(stream)
    blk = SleepBlock(**pbe["sleep_block"])

    with paths["mua"].open("rb") as stream:
        mua = pickle.load(stream)
    offset = (int(mua["epoch_start_sample"]) - blk.day_start_sample) / blk.sampling_frequency_hz
    mua_rows = []
    for shank, entry in sorted(mua["shanks"].items()):
        ids = list(np.asarray(entry["channel_ids"]).tolist())
        ys = np.asarray(entry["channel_locations"])[:, 1]
        for ch, t in entry["channel_spike_times"].items():
            mua_rows.append((int(shank), float(ys[ids.index(ch)]), np.asarray(t) + offset))
    del mua
    mua_rows.sort(key=lambda r: (r[0], -r[1]))

    with paths["spikes"].open("rb") as stream:
        spikes = pickle.load(stream)
    depth = unit_depths(Path(paths["spikes"]).parent)
    unit_rows = [(int(r.get("shank") or 0), depth.get(uid, np.nan), np.asarray(r["spike_times_sec"]))
                 for uid, r in spikes["spike_data"].items()]
    n_missing = int(sum(np.isnan(r[1]) for r in unit_rows))
    unit_rows.sort(key=lambda r: (r[0], -np.nan_to_num(r[1], nan=-1e9)))

    p = pbe["params"]
    nrem_on = pbe["nrem_on_intervals_sec"]
    edges, _, rate_u = dp.population_trace([r[2] for r in unit_rows], blk.duration_sec,
                                           p["bin_sec"], p["sigma_sec"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = interval_mask(centers, nrem_on)
    z_units = (rate_u - rate_u[mask].mean()) / rate_u[mask].std()

    return {"paths": paths, "label": label, "block": blk, "pbe": pbe,
            "mua": mua_rows, "units": unit_rows, "n_units_no_depth": n_missing,
            "z_mua": pbe["trace"]["z"], "z_units": z_units, "centers": centers}


def stamp(fig, d, extra=""):
    p = d["pbe"]["params"]
    text = (f"{datetime.now():%Y-%m-%d %H:%M}  |  sleep/PBE/plot_mua_vs_units.py {' '.join(sys.argv[1:])}  |  "
            f"{d['label']}  |  {len(d['mua'])} MUA channels, {len(d['units'])} sorted units "
            f"({d['n_units_no_depth']} without a location)  |  rows: shank, then deep->superficial  |  "
            f"pooled z: {p['bin_sec']*1e3:g} ms bins, sigma {p['sigma_sec']*1e3:g} ms, z on NREM-ON bins (units; MUA PBEs: {p.get('state', 'nrem_on')})  |  "
            f"PBEs: MUA source{extra}\n"
            f"MUA: {d['paths']['mua']}\nunits: {d['paths']['spikes']}\nON: {d['paths']['on']}  |  NREM: {d['paths']['nrem']}")
    fig.text(0.005, 0.003, text, fontsize=5.5, family="monospace", va="bottom")


# ---------------------------------------------------------------- drawing
def raster(ax, rows, t0, t1, color, lw):
    for i, (_, _, train) in enumerate(rows):
        t = train[(train >= t0) & (train < t1)]
        if len(t):
            ax.vlines(t, i, i + 0.9, lw=lw, color=color)
    shank_ticks(ax, np.array([r[0] for r in rows]))


def draw_window(axes, d, events, t0, t1, labels=True):
    ax_m, ax_u, ax_z = axes
    nrem_on = d["pbe"]["nrem_on_intervals_sec"]
    short = t1 - t0 <= 0.6
    for ax in (ax_m, ax_u):
        shade(ax, nrem_on, t0, t1, ON_COLOR, 0.07)
        shade(ax, events[:, :2], t0, t1, PBE_COLOR, 0.15)
    raster(ax_m, d["mua"], t0, t1, "0.1", 0.6 if short else 0.35)
    raster(ax_u, d["units"], t0, t1, UNIT_COLOR, 1.2 if short else 0.8)
    sel = (d["centers"] >= t0) & (d["centers"] < t1)
    shade(ax_z, nrem_on, t0, t1, ON_COLOR, 0.12)
    shade(ax_z, events[:, :2], t0, t1, PBE_COLOR, 0.3)
    ax_z.plot(d["centers"][sel], d["z_mua"][sel], color="k", lw=1.0, label="MUA")
    ax_z.plot(d["centers"][sel], d["z_units"][sel], color=UNIT_COLOR, lw=1.0, label="units")
    p = d["pbe"]["params"]
    ax_z.axhline(p["onset_z"], ls="--", lw=0.5, color="0.5")
    ax_z.axhline(p["peak_z"], ls="--", lw=0.5, color="0.3")
    for ax in axes:
        ax.set_xlim(t0, t1)
        ax.tick_params(labelsize=6)
        ax.xaxis.get_major_formatter().set_useOffset(False)
    for ax in (ax_m, ax_u):
        ax.tick_params(labelbottom=False)
    if labels:
        ax_m.set_ylabel(f"channel MUA ({len(d['mua'])})", fontsize=8)
        ax_u.set_ylabel(f"sorted units ({len(d['units'])})", fontsize=8, color=UNIT_COLOR)
        ax_z.set_ylabel("pooled z", fontsize=8)


def plot_window(d, events, t_center, out_png, title):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1.4], hspace=0.12, wspace=0.12,
                          left=0.06, right=0.99, top=0.93, bottom=0.12)
    for col, span in enumerate((2.0, 0.5)):
        t0, t1 = t_center - span / 2, t_center + span / 2
        axes = [fig.add_subplot(gs[r, col]) for r in range(3)]
        draw_window(axes, d, events, t0, t1, labels=(col == 0))
        axes[0].set_title(f"{span:g} s", fontsize=11)
        axes[2].set_xlabel("block time (s)", fontsize=8)
        if col == 0:
            axes[2].legend(fontsize=7, loc="upper left", ncol=2)
            for ax in axes:
                ax.axvspan(t_center - 0.25, t_center + 0.25, facecolor="none", edgecolor="tab:red", lw=1.0)
    fig.suptitle(f"{d['label']}  |  {title}  |  black = channel MUA, red = sorted units, "
                 f"blue = MUA PBE, green = NREM-ON, red box = 0.5 s zoom", fontsize=10, x=0.06, ha="left")
    stamp(fig, d, extra=f"  |  centred at {t_center:.3f} s")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_grid(d, events, examples, out_png, span=1.0):
    import matplotlib.pyplot as plt
    n = len(examples)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5 * ncol, 6.2 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.25, wspace=0.18, left=0.04, right=0.99,
                             top=1 - 0.4 / nrow / 6.2 * 6, bottom=0.6 / (6.2 * nrow))
    for i, (kind, t) in enumerate(examples):
        inner = outer[i // ncol, i % ncol].subgridspec(3, 1, height_ratios=[3, 3, 1.2], hspace=0.06)
        axes = [fig.add_subplot(inner[r]) for r in range(3)]
        draw_window(axes, d, events, t - span / 2, t + span / 2, labels=(i % ncol == 0))
        axes[0].set_title(f"{i + 1}. {kind} @ {t:.1f} s", fontsize=8)
    fig.suptitle(f"{d['label']}  |  channel MUA (black) vs sorted units (red), {span:g} s windows  |  "
                 f"blue = MUA PBE, green = NREM-ON", fontsize=11, x=0.04, ha="left")
    stamp(fig, d)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- stats
def binned(trains, edges):
    out = np.zeros(len(edges) - 1)
    for t in trains:
        out += np.histogram(t, edges)[0]
    return out


def plot_stats(d, out_png, bins_ms=(5, 10, 20, 50, 100, 250)):
    import matplotlib.pyplot as plt
    blk = d["block"]
    nrem_on = d["pbe"]["nrem_on_intervals_sec"]
    shanks = sorted({r[0] for r in d["mua"]})
    res = {"all": []} | {s: [] for s in shanks}
    for b in bins_ms:
        edges = np.arange(0, blk.duration_sec, b / 1e3)
        mask = interval_mask(0.5 * (edges[:-1] + edges[1:]), nrem_on)
        for key in res:
            pick = (lambda r: True) if key == "all" else (lambda r, k=key: r[0] == k)
            m = binned([r[2] for r in d["mua"] if pick(r)], edges)[mask]
            u = binned([r[2] for r in d["units"] if pick(r)], edges)[mask]
            res[key].append(np.corrcoef(m, u)[0, 1] if m.std() and u.std() else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.plot(bins_ms, res["all"], "o-", color="k", lw=2, label="whole probe")
    for s, color in zip(shanks, plt.cm.tab10(np.arange(len(shanks)))):
        ax.plot(bins_ms, res[s], "o-", color=color, lw=1, ms=3, label=f"shank {s}")
    ax.set_xscale("log")
    ax.set_xticks(bins_ms)
    ax.set_xticklabels([str(b) for b in bins_ms])
    ax.set_ylim(-0.05, 1)
    ax.set_xlabel("bin (ms)")
    ax.set_ylabel("Pearson r, pooled MUA vs pooled units")
    ax.set_title("agreement inside NREM-ON", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    on_sec = float((nrem_on[:, 1] - nrem_on[:, 0]).sum())
    def rate(rows, s):
        return sum(np.sum(interval_mask(r[2], nrem_on)) for r in rows if r[0] == s) / on_sec
    m_rate = [rate(d["mua"], s) for s in shanks]
    u_rate = [rate(d["units"], s) for s in shanks]
    x = np.arange(len(shanks))
    ax.bar(x - 0.2, m_rate, 0.4, color="0.3", label="channel MUA")
    ax.bar(x + 0.2, u_rate, 0.4, color=UNIT_COLOR, label="sorted units")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sh{s}" for s in shanks])
    ax.set_ylabel("spikes / s in NREM-ON (summed over shank)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("how many spikes each source has", fontsize=10)

    ax = axes[2]
    n_units = [sum(r[0] == s for r in d["units"]) for s in shanks]
    ax.bar(x, n_units, color=UNIT_COLOR)
    for i, v in enumerate(n_units):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"sh{s}" for s in shanks])
    ax.set_ylabel("sorted units (32 MUA channels per shank)")
    ax.set_title("units per shank", fontsize=10)
    stamp(fig, d, extra=f"  |  NREM-ON time {on_sec:.0f} s")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.25)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return dict(zip(bins_ms, res["all"]))


# ---------------------------------------------------------------- driver
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", default="CnL46")
    p.add_argument("--day", default="260731")
    p.add_argument("--block", default="post", choices=("pre", "post"))
    p.add_argument("--example-times", type=float, nargs="*", default=None)
    p.add_argument("--n-random-examples", type=int, default=7, help="random MUA PBEs")
    p.add_argument("--n-random-on", type=int, default=4, help="random NREM-ON windows")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    dp.install_pickle_compat()
    d = load_inputs(args.animal, args.day, args.block)
    events = kept_events(d["pbe"])
    out = Path(d["paths"]["out_dir"]) / "figures" / "mua_vs_units"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{d['label']}_mua_vs_units_*.png"):
        old.unlink()

    examples = ([("chosen", t) for t in args.example_times] if args.example_times
                else pick_examples(events, args.n_random_examples, d["pbe"]["nrem_on_intervals_sec"],
                                   args.n_random_on, args.seed))
    for n, (kind, t) in enumerate(examples, 1):
        f = out / f"{d['label']}_mua_vs_units_{n:02d}_{t:07.1f}s_{kind.replace(' ', '_')}.png"
        plot_window(d, events, t, f, f"window {n}/{len(examples)} ({kind}) at {t:.2f} s")
    print(f"{len(examples)} window figures -> {out}")
    plot_grid(d, events, examples, out / f"{d['label']}_mua_vs_units_grid.png")
    r = plot_stats(d, out / f"{d['label']}_mua_vs_units_stats.png")
    print("grid + stats written; r(MUA, units) in NREM-ON by bin: "
          + ", ".join(f"{b} ms {v:.2f}" for b, v in r.items()))
    print(f"units without a location: {d['n_units_no_depth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
