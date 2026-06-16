"""
detect_up_down.py  —  Stage 3 of the UP/DOWN pipeline
=====================================================
Segments UP vs DOWN cortical states inside a consolidated-NREM window found by
detect_sleep_periods.py (Stage 2).

Hybrid design (per the user's choice):
  * NOW (LFP-only): population activity is proxied by broadband-gamma (30-200 Hz)
    power on deep-layer channels. DOWN states = sustained troughs of this proxy
    (population silence), UP = the active periods between them.
  * LATER (MUA): set up_down_params['mua_pkl'] to a MUA.py output; the binned
    spike rate then replaces the LFP proxy and the SAME detector runs unchanged.

Why re-extract from the NWB: the Stage-1/2 LFP is 500 Hz (Nyquist 250), too low
for broadband gamma. We re-read just this short window at 1250 Hz.

Alignment: Stage-2 window times are in lfp_time = seconds from the sleep-window
start (sleep_start_sample). Raw sample = sleep_start_sample + t * original_fs.

Outputs (low_freq/up_down/):
  {session}_sh{shank}_up_down.pkl   — down/up intervals (window- and lfp-time), proxy
  {session}_sh{shank}_up_down.png   — zoom (LFP + proxy + UP/DOWN) + summary
"""
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from sleep_params import (rec_folder, session_name, nwb_session_name,
                          original_fs, sleep_start_sample, preproc_params,
                          plot_params, up_down_params as U)

rec_folder = Path(rec_folder)
low_freq_folder = rec_folder / "low_freq"
out_dir = low_freq_folder / "up_down"
out_dir.mkdir(exist_ok=True)
ish = U["shank"]

# ── pick the NREM window from Stage 2 ─────────────────────────────────────────
seg_pkl = low_freq_folder / "sleep_segmentation" / f"{session_name}_sleep_periods.pkl"
with open(seg_pkl, "rb") as f:
    seg = pickle.load(f)

if U["window"] == "longest":
    win = seg["fully_asleep_window_s"]
    if win is None:
        raise SystemExit("Stage 2 found no consolidated window; relax thresholds.")
else:
    win = tuple(U["window"])
win_start_s, win_end_s = float(win[0]), float(win[1])
print(f"NREM window (lfp_time): {win_start_s:.1f} - {win_end_s:.1f} s "
      f"({win_end_s - win_start_s:.1f} s)")

# ── re-extract this window from the NWB at higher rate ────────────────────────
nwb_path = f"{rec_folder}\\{nwb_session_name}sh{ish}.nwb"
print(f"Loading NWB: {nwb_path}")
rec = se.NwbRecordingExtractor(nwb_path)
orig_fs = rec.get_sampling_frequency()
assert abs(orig_fs - original_fs) < 1, f"orig_fs {orig_fs} != {original_fs}"

# map window (lfp_time, from sleep-window start) -> absolute raw samples
raw_start = int(sleep_start_sample) + int(round(win_start_s * orig_fs))
raw_end = int(sleep_start_sample) + int(round(win_end_s * orig_fs))
rec_win = rec.frame_slice(start_frame=raw_start, end_frame=raw_end)

# CAR (match Stage 1) -> bandpass -> integer decimate to target_fs
rec_car = spre.common_reference(rec_win, reference=preproc_params["reference"],
                                operator=preproc_params["operator"])
rec_band = spre.bandpass_filter(rec_car, freq_min=U["extract_filter"][0],
                                freq_max=U["extract_filter"][1])
factor = orig_fs / U["target_fs"]
dec = int(round(factor))
assert abs(factor - dec) < 1e-6, f"need integer decimation, got {factor}"
rec_lfp = spre.decimate(rec_band, dec)
fs = rec_lfp.get_sampling_frequency()

# depth-sorted channels, select deep layer
locs = rec_lfp.get_channel_locations()
ch_ids = rec_lfp.get_channel_ids()
ycoord = locs[:, 1]
order = np.argsort(ycoord)
ch_sorted = ch_ids[order]
y_sorted = ycoord[order]
deep_mask = (y_sorted >= U["deep_layer_um"][0]) & (y_sorted <= U["deep_layer_um"][1])
deep_ch = ch_sorted[deep_mask]
print(f"Extracting {len(deep_ch)}/{len(ch_ids)} deep channels "
      f"({U['deep_layer_um'][0]}-{U['deep_layer_um'][1]} um) at {fs:.0f} Hz")

traces = rec_lfp.get_traces(channel_ids=ch_sorted.tolist(),
                            return_scaled=True).astype("float32")  # (n, n_ch)
deep_idx = np.where(deep_mask)[0]
deep_traces = traces[:, deep_idx]                                   # (n, n_deep)
n = deep_traces.shape[0]
t_local = np.arange(n) / fs                                         # 0..win_len


# ── population-activity proxy ─────────────────────────────────────────────────
def bp_power(x, fs, lo, hi):
    sos = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                        output="sos")
    return signal.sosfiltfilt(sos, x, axis=0) ** 2


if U["mua_pkl"]:
    # MUA mode: binned spike rate on the window grid (drop-in replacement)
    with open(U["mua_pkl"], "rb") as f:
        mua = pickle.load(f)
    mt = mua["mua_time"]; mr = mua["mua_rate_smooth"]
    pa_raw = np.interp(t_local + win_start_s, mt, mr)
    pa = pa_raw
    proxy_label = "MUA rate (sp/s)"
else:
    # LFP mode: broadband-gamma power, averaged over deep channels, smoothed
    gp = bp_power(deep_traces, fs, *U["gamma_broad"]).mean(axis=1)  # (n,)
    sig = max(1, int(U["env_smooth_ms"] / 1000 * fs))
    pa = gaussian_filter1d(gp, sigma=sig)
    proxy_label = f"gamma {U['gamma_broad'][0]:.0f}-{U['gamma_broad'][1]:.0f}Hz power"

# z-score of log-proxy within the window
pa_log = np.log(pa + 1e-12)
pa_z = (pa_log - np.median(pa_log)) / (1.4826 * np.median(
    np.abs(pa_log - np.median(pa_log))) + 1e-12)

# slow-wave (delta) trace, mean over deep channels (for display / corroboration)
sos_d = signal.butter(4, [U["delta_band"][0] / (fs / 2),
                          U["delta_band"][1] / (fs / 2)], btype="band",
                      output="sos")
delta_trace = signal.sosfiltfilt(sos_d, deep_traces, axis=0).mean(axis=1)

# ── DOWN-state detection ──────────────────────────────────────────────────────
thresh = np.percentile(pa_z, U["down_percentile"])
below = pa_z < thresh


def runs(mask):
    out = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append([i, j])
            i = j
        else:
            i += 1
    return out


cand = runs(below)
# merge close DOWNs
merge_g = int(U["merge_gap_ms"] / 1000 * fs)
merged = []
for r in cand:
    if merged and (r[0] - merged[-1][1]) <= merge_g:
        merged[-1][1] = r[1]
    else:
        merged.append(r)
# duration filter
min_d = int(U["min_down_ms"] / 1000 * fs)
max_d = int(U["max_down_ms"] / 1000 * fs)
down = [r for r in merged if min_d <= (r[1] - r[0]) <= max_d]

# UP = gaps between DOWNs (within the window), with a min duration
min_u = int(U["min_up_ms"] / 1000 * fs)
up = []
prev = 0
for r in down:
    if (r[0] - prev) >= min_u:
        up.append([prev, r[0]])
    prev = r[1]
if (n - prev) >= min_u:
    up.append([prev, n])

down_s = [(t_local[a], t_local[min(b, n - 1)]) for a, b in down]
up_s = [(t_local[a], t_local[min(b, n - 1)]) for a, b in up]
down_durs = np.array([b - a for a, b in down_s])
win_len = win_end_s - win_start_s

print(f"DOWN states: {len(down_s)}  (rate {len(down_s)/win_len:.2f}/s, "
      f"mean {1000*down_durs.mean():.0f} ms, median {1000*np.median(down_durs):.0f} ms)"
      if len(down_s) else "DOWN states: 0")
print(f"UP states:   {len(up_s)}")

# ── save ──────────────────────────────────────────────────────────────────────
out_pkl = out_dir / f"{session_name}_sh{ish}_up_down.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump({
        "session": session_name, "shank": ish, "fs": fs,
        "window_lfp_s": (win_start_s, win_end_s),
        "raw_sample_range": (raw_start, raw_end),
        "proxy_mode": "mua" if U["mua_pkl"] else "lfp_gamma",
        "proxy_label": proxy_label,
        "t_local": t_local, "pa_z": pa_z, "thresh_z": float(thresh),
        # intervals in window-local seconds AND absolute lfp_time seconds
        "down_intervals_local_s": down_s,
        "up_intervals_local_s": up_s,
        "down_intervals_lfp_s": [(a + win_start_s, b + win_start_s) for a, b in down_s],
        "up_intervals_lfp_s": [(a + win_start_s, b + win_start_s) for a, b in up_s],
        "down_durations_s": down_durs,
        "params": U,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved → {out_pkl}")

# ── figure: zoom (LFP + proxy + UP/DOWN) + summary ────────────────────────────
z0 = min(U["zoom_offset_sec"], max(0, win_len - U["zoom_sec"]))
z1 = min(z0 + U["zoom_sec"], win_len)
zm = (t_local >= z0) & (t_local <= z1)

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1],
                      width_ratios=[3, 1])

# shade DOWN spans helper
def shade_down(ax):
    for a, b in down_s:
        if b >= z0 and a <= z1:
            ax.axvspan(a, b, color="steelblue", alpha=0.25, lw=0)

# LFP deep channels (zoom)
axL = fig.add_subplot(gs[0, 0])
step = max(1, deep_traces.shape[1] // 12)
sel = np.arange(0, deep_traces.shape[1], step)
off = np.median(np.std(deep_traces[zm][:, sel], axis=0)) * 6
for k, ci in enumerate(sel):
    axL.plot(t_local[zm], deep_traces[zm, ci] - k * off, "k", lw=0.4, alpha=0.8)
shade_down(axL)
axL.set_xlim(z0, z1); axL.set_yticks([]); axL.set_ylabel("Deep LFP")
axL.set_title(f"{session_name} sh{ish}  UP/DOWN  (DOWN shaded)  —  "
              f"NREM {win_start_s:.0f}-{win_end_s:.0f}s, zoom {z0:.0f}-{z1:.0f}s")

# delta trace (zoom)
axD = fig.add_subplot(gs[1, 0], sharex=axL)
axD.plot(t_local[zm], delta_trace[zm], "purple", lw=0.7)
shade_down(axD); axD.set_ylabel(f"Delta\n{U['delta_band'][0]}-{U['delta_band'][1]}Hz")
axD.set_xlim(z0, z1)

# proxy (zoom)
axP = fig.add_subplot(gs[2, 0], sharex=axL)
axP.plot(t_local[zm], pa_z[zm], "C1", lw=0.7)
axP.axhline(thresh, color="k", ls="--", lw=0.8, label=f"DOWN < {thresh:.2f}")
shade_down(axP)
axP.set_ylabel(f"{proxy_label}\n(log z)"); axP.set_xlabel("Time in window (s)")
axP.set_xlim(z0, z1); axP.legend(fontsize=8, loc="upper right")

# DOWN duration histogram
axH = fig.add_subplot(gs[1, 1])
if len(down_durs):
    axH.hist(down_durs * 1000, bins=30, color="steelblue", alpha=0.8)
    axH.axvline(np.median(down_durs) * 1000, color="k", ls="--", lw=1)
axH.set_xlabel("DOWN duration (ms)"); axH.set_ylabel("count")

# summary text
axS = fig.add_subplot(gs[0, 1]); axS.axis("off")
dr = len(down_s) / win_len if win_len else 0
summary = (f"proxy: {'MUA' if U['mua_pkl'] else 'LFP gamma'}\n"
           f"window: {win_len:.0f} s @ {fs:.0f} Hz\n"
           f"deep ch: {len(deep_ch)}\n"
           f"DOWN states: {len(down_s)}\n"
           f"DOWN rate: {dr:.2f} /s\n")
if len(down_durs):
    summary += (f"DOWN dur: {1000*down_durs.mean():.0f}±"
                f"{1000*down_durs.std():.0f} ms\n"
                f"median: {1000*np.median(down_durs):.0f} ms\n")
summary += f"UP states: {len(up_s)}"
axS.text(0, 1, summary, va="top", ha="left", fontsize=11, family="monospace")

stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by detect_up_down.py | "
         f"{session_name} sh{ish} | NWB={nwb_path} | window(lfp)={win} | params={U}")
fig.text(0.005, 0.001, stamp, fontsize=6, color="0.4", ha="left", va="bottom")

out_png = out_dir / f"{session_name}_sh{ish}_up_down.png"
fig.savefig(out_png, dpi=plot_params["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_png}")
