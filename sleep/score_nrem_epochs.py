"""
score_nrem_epochs.py  —  NREM scoring and consolidated-window selection
=======================================================================
Segments the period(s) where the animal is "fully asleep" (consolidated NREM)
from the 500 Hz feature pkl produced by compute_sleep_features.py.

NREM index  = PC1 of the spectrogram, oriented so high = high delta (slow waves)
Movement    = broadband power proxy (mean dB over freqs <= move_fmax); the same
              signal the artifact detector uses (high broadband -> moving/awake)

An epoch is NREM when slow-wave activity is high AND movement is low AND the
bin is not a broadband artifact. NREM epochs are gap-bridged and short bouts
dropped; bouts >= consolidated_sec are the "fully asleep" windows handed to the
UP/DOWN stage (score_cortical_up_down_states.py).

Outputs (in low_freq/sleep_segmentation/):
  {session}_sleep_periods.pkl   — nrem_mask, bouts, consolidated windows, scores
  {session}_sleep_periods.png   — Buzsaki-style figure with colored state bar
"""
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_pipeline_config import (rec_folder, session_name, shanks, plot_params,
                                   artifact_params, sleep_detect_params as P,
                                   resolve_existing_file, resolve_output_folder)
from band_powers_io import (band_time, band_fs, load_spectrograms,
                            epoch_average, epoch_times)
from sleep_artifact_detection import (broadband_level, robust_z,
                                      detect_broadband_artifacts, mask_to_spans)

rec_folder = Path(rec_folder)
low_freq_folder = rec_folder / "low_freq"
# Falls back to the backup server if a previous stage saved there due to low space.
bp_file = resolve_existing_file(low_freq_folder / f"{session_name}_all_shanks_band_powers.pkl")
# Use wherever the band-powers file actually was (primary or backup server) as
# the base for outputs too, so they land next to their source data.
low_freq_folder = bp_file.parent
out_dir = resolve_output_folder(low_freq_folder / "sleep_segmentation")

use_shanks = P["shanks"] if P["shanks"] is not None else shanks

print(f"Loading band powers: {bp_file}")
with open(bp_file, "rb") as f:
    all_data = pickle.load(f)


# ── aggregate slow-wave (PC1) + delta/sigma + movement across shanks/channels ──
pc1_oriented_list = []      # per channel, on band_t, oriented so high=NREM
delta_list, sigma_list, gamma_list = [], [], []
bb_specbase_list = []       # broadband movement level, on spectrogram time base
mean_spec_list = []         # mean log-power spectrogram for display

# `band_t` is the grid the band powers and PC1 are stored on: the spectrogram
# grid for a compact file, the 500 Hz LFP grid for one written before that
# change. Everything below works off those timestamps, so both layouts score
# identically. `band_rate` is the rate of THAT grid - not sd["sampling_rate"],
# which stays the 500 Hz rate the envelopes were computed at.
band_t = None
spec_times = None
freqs = None
band_rate = None

for ish in use_shanks:
    if ish not in all_data["shanks_data"]:
        print(f"  [skip] shank {ish} not in data")
        continue
    sd = all_data["shanks_data"][ish]
    if band_t is None:
        band_t = band_time(sd)
        spec_times = sd["spectrogram_times"]
        freqs = sd["spectrogram_freqs"]
        band_rate = band_fs(sd)
    ch_ids = sd["channel_ids"]
    pc1 = sd["pc1_spectrogram"]                 # (n_ch, n_samples) on band_t
    spec = load_spectrograms(sd, bp_file, ish)  # (n_ch, n_freqs, n_times)

    # orient each channel's PC1 by its correlation with delta, then collect
    for ci, ch in enumerate(ch_ids):
        delta = np.asarray(sd["band_powers"][ch]["delta"], float)
        sigma = np.asarray(sd["band_powers"][ch]["sigma"], float)
        gamma = np.asarray(sd["band_powers"][ch]["gamma"], float)
        p = np.asarray(pc1[ci], float)
        n = min(len(p), len(delta))
        c = np.corrcoef(p[:n], delta[:n])[0, 1]
        if not np.isfinite(c):
            c = 1.0
        pc1_oriented_list.append(p if c >= 0 else -p)
        delta_list.append(delta)
        sigma_list.append(sigma)
        gamma_list.append(gamma)

    # mean spectrogram across channels for this shank (movement + display)
    mean_spec_list.append(spec.mean(axis=0))    # (n_freqs, n_times)

# average the mean spectrograms across shanks
mean_spec = np.mean(np.stack([m[:, :len(spec_times)] for m in mean_spec_list]),
                    axis=0)                      # (n_freqs, n_times)

# align channel signals to a common length on band_t
n_band = min(min(len(a) for a in pc1_oriented_list), len(band_t))
band_t = band_t[:n_band]
pc1_mean = np.mean([a[:n_band] for a in pc1_oriented_list], axis=0)
delta_mean = np.mean([a[:n_band] for a in delta_list], axis=0)
sigma_mean = np.mean([a[:n_band] for a in sigma_list], axis=0)
gamma_mean = np.mean([a[:n_band] for a in gamma_list], axis=0)
_ = pc1_mean  # PC1 retained in output for reference; tilt index drives scoring

# spectral-tilt slow-wave index: z(log delta) - z(log gamma).
# High in NREM (delta up, gamma down); low during movement/wake (gamma up).
# This is immune to the delta-contamination that broke a broadband proxy.
delta_z_band = robust_z(np.log(delta_mean + 1e-12))
gamma_z_band = robust_z(np.log(gamma_mean + 1e-12))
sw_band = delta_z_band - gamma_z_band

# artifact mask on spectrogram base, mapped onto band_t
if P["use_artifact_mask"]:
    art_mask_spec, _ = detect_broadband_artifacts(
        mean_spec, freqs, spec_times[:mean_spec.shape[1]],
        n_mad=artifact_params["n_mad"], dilate_sec=artifact_params["dilate_sec"],
        fmax=artifact_params["fmax"])
    art_mask_band = np.interp(band_t, spec_times[:len(art_mask_spec)],
                              art_mask_spec.astype(float)) > 0.5
else:
    art_mask_spec = np.zeros(mean_spec.shape[1], bool)
    art_mask_band = np.zeros(n_band, bool)

# ── epoch the signals and score NREM ──────────────────────────────────────────
# Epochs come from the timestamps, not from a sample count: at 0.512 s bins a
# 4 s epoch is 7.8125 samples, and rounding that to 8 would stretch every epoch
# by 2.4% - minutes of drift over a session.
ep_sec = P["epoch_sec"]
sw_ep = epoch_average(sw_band, band_t, ep_sec)          # delta-gamma tilt (SD units)
move_ep = epoch_average(gamma_z_band, band_t, ep_sec)   # high-freq (gamma) proxy
art_ep = epoch_average(art_mask_band.astype(float), band_t, ep_sec) > 0.5
n_ep = sw_ep.size
ep_t = epoch_times(ep_sec, n_ep)

# light smoothing
k = max(1, int(P["smooth_epochs"]))
if k > 1:
    box = np.ones(k) / k
    sw_ep = np.convolve(sw_ep, box, mode="same")
    move_ep = np.convolve(move_ep, box, mode="same")

# NREM: high spectral tilt (delta >> gamma) and not an artifact bin. The tilt
# index already suppresses movement (gamma up -> tilt down), so the optional
# move gate only rejects extreme high-frequency epochs.
nrem_ep = (sw_ep > P["nrem_sw_z_thresh"]) & (~art_ep)
if P.get("move_z_thresh") is not None:
    nrem_ep &= (move_ep < P["move_z_thresh"])

# ── gap-bridge + min-bout on the epoch mask ───────────────────────────────────
def epochs_to_bouts(mask, ep_sec):
    spans = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            spans.append([i, j])      # [start_epoch, end_epoch)
            i = j
        else:
            i += 1
    return spans

bouts = epochs_to_bouts(nrem_ep, ep_sec)
# bridge short gaps
merge_gap_ep = int(round(P["merge_gap_sec"] / ep_sec))
merged = []
for b in bouts:
    if merged and (b[0] - merged[-1][1]) <= merge_gap_ep:
        merged[-1][1] = b[1]
    else:
        merged.append(b)
# drop short bouts
min_ep = int(round(P["min_bout_sec"] / ep_sec))
merged = [b for b in merged if (b[1] - b[0]) >= min_ep]

# rebuild epoch mask from merged bouts
nrem_ep_clean = np.zeros(n_ep, bool)
for b in merged:
    nrem_ep_clean[b[0]:b[1]] = True

# convert to (start_s, end_s) intervals and consolidated windows
bout_intervals = [(float(ep_t[b[0]] - ep_sec / 2),
                   float(ep_t[min(b[1], n_ep) - 1] + ep_sec / 2)) for b in merged]
consolidated = [iv for iv in bout_intervals
                if (iv[1] - iv[0]) >= P["consolidated_sec"]]
fully_asleep = max(consolidated, key=lambda iv: iv[1] - iv[0], default=None)

# NREM mask on the band-power time base, built from the bout times rather than
# an epoch-to-sample count, so it holds at whatever rate that grid is.
nrem_mask_band = np.zeros(n_band, bool)
for start, end in bout_intervals:
    nrem_mask_band |= (band_t >= start) & (band_t < end)

# ── report ────────────────────────────────────────────────────────────────────
total_nrem_s = float(nrem_mask_band.sum()) / band_rate
print(f"\nEpochs: {n_ep} x {ep_sec}s   NREM epochs: {nrem_ep_clean.sum()}")
print(f"NREM bouts (>= {P['min_bout_sec']}s): {len(bout_intervals)}")
print(f"Consolidated windows (>= {P['consolidated_sec']}s): {len(consolidated)}")
for i, iv in enumerate(consolidated):
    print(f"  [{i}] {iv[0]:8.1f} - {iv[1]:8.1f} s   ({iv[1]-iv[0]:6.1f} s)")
if fully_asleep:
    print(f"Longest 'fully asleep' window: {fully_asleep[0]:.1f} - "
          f"{fully_asleep[1]:.1f} s  ({fully_asleep[1]-fully_asleep[0]:.1f} s)")
else:
    print("No consolidated window found — relax thresholds in sleep_detect_params.")

# ── save ──────────────────────────────────────────────────────────────────────
out_pkl = out_dir / f"{session_name}_sleep_periods.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump({
        "session": session_name,
        "shanks": list(use_shanks),
        # The time base the scoring ran on, named for what it is: the band-power
        # grid (~2 Hz for a compact band-power file, 500 Hz for a legacy one).
        # The old "lfp_time"/"fs"/"nrem_mask_lfp" keys implied the LFP rate,
        # which is no longer what these are sampled at.
        "band_time": band_t,
        "band_fs": band_rate,
        "epoch_sec": ep_sec,
        "epoch_times": ep_t,
        "sw_index": sw_ep, "gamma_z": move_ep,
        "nrem_epoch_mask": nrem_ep_clean,
        "nrem_mask": nrem_mask_band,
        "bout_intervals_s": bout_intervals,
        "consolidated_windows_s": consolidated,
        "fully_asleep_window_s": fully_asleep,
        "params": P,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved → {out_pkl}")

# ── figure: Buzsaki-style with state bar on top ───────────────────────────────
t0, t1 = band_t[0], band_t[-1]
art_spans = mask_to_spans(spec_times[:len(art_mask_spec)], art_mask_spec)

# display spectrogram: dB power, robust z PER FREQUENCY over non-artifact bins
# (whitens 1/f so band/state structure is visible, as in Buzsaki)
gcol = ~art_mask_spec[:mean_spec.shape[1]]
if not gcol.any():
    gcol = np.ones(mean_spec.shape[1], bool)
log_spec = 10 * np.log10(mean_spec + 1e-12)
med_f = np.median(log_spec[:, gcol], axis=1, keepdims=True)
mad_f = np.median(np.abs(log_spec[:, gcol] - med_f), axis=1, keepdims=True)
spec_z = (log_spec - med_f) / (1.4826 * mad_f + 1e-10)
vmin = np.percentile(spec_z[:, gcol], 2)
vmax = np.percentile(spec_z[:, gcol], 98)

fig = plt.figure(figsize=(plot_params["figsize"][0], 10), constrained_layout=True)
gs = fig.add_gridspec(6, 1, height_ratios=[0.25, 2, 1, 1, 1, 1], hspace=0.25)


def shade(ax, color, alpha, spans):
    for (a, b) in spans:
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, zorder=5)

# 0: state bar
ax0 = fig.add_subplot(gs[0])
shade(ax0, "steelblue", 0.9, bout_intervals)
if fully_asleep:
    ax0.axvspan(*fully_asleep, color="navy", alpha=0.9, lw=0)
ax0.set_xlim(t0, t1); ax0.set_yticks([])
ax0.set_ylabel("NREM", rotation=0, ha="right", va="center", fontsize=9)
ax0.set_xticks([]); ax0.set_title(
    f"{session_name}  —  consolidated-sleep segmentation  "
    f"(navy = fully-asleep window)", fontsize=12)

# 1: spectrogram
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.pcolormesh(spec_times[:mean_spec.shape[1]], freqs, spec_z, shading="gouraud",
               cmap=plot_params["cmap"], vmin=vmin, vmax=vmax)
ax1.set_yscale("log"); ax1.set_ylim(plot_params["freq_min"], plot_params["freq_max"])
ax1.set_yticks([1, 4, 16, 64]); ax1.set_yticklabels(["1", "4", "16", "64"])
ax1.set_ylabel("f (Hz)"); ax1.set_xticklabels([])
shade(ax1, artifact_params["shade_color"], artifact_params["shade_alpha"], art_spans)

# 2-4: scores / bands
def trace_panel(gs_idx, x, y, ylabel, color, thresh=None):
    ax = fig.add_subplot(gs[gs_idx], sharex=ax0)
    shade(ax, "steelblue", 0.12, bout_intervals)
    shade(ax, artifact_params["shade_color"], artifact_params["shade_alpha"], art_spans)
    ax.plot(x, y, color=color, lw=0.6)
    if thresh is not None:
        ax.axhline(thresh, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_xlim(t0, t1)
    ax.set_xticklabels([])
    return ax

trace_panel(2, ep_t, sw_ep, "Slow-wave\nδ−γ tilt", "steelblue", P["nrem_sw_z_thresh"])
trace_panel(3, ep_t, move_ep, "Gamma\n40-100Hz (z)", "goldenrod", P.get("move_z_thresh"))
ax5 = trace_panel(4, band_t, robust_z(delta_mean), "Delta\n0.5-4 Hz", "k")
ax6 = trace_panel(5, band_t, robust_z(sigma_mean), "Sigma\n9-25 Hz", "k")
ax6.set_xlabel("Time (s)"); ax6.set_xticklabels(
    [f"{int(t)}" for t in ax6.get_xticks()])

stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by score_nrem_epochs.py | "
         f"session={session_name} shanks={list(use_shanks)} | source={bp_file} | "
         f"params={P}")
fig.text(0.005, 0.001, stamp, fontsize=6, color="0.4", ha="left", va="bottom")

out_png = out_dir / f"{session_name}_sleep_periods.png"
fig.savefig(out_png, dpi=plot_params["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_png}")
