"""Plot a spectrogram summary figure directly from an exported trace_data pkl.

Reads the pkl written by spectrogram_plot.py (z-scored spectrogram + band
traces + velocity, with artifact periods already excised and the clean parts
concatenated) and renders a publication-style figure:

    spectrogram (top)  ->  one panel per band trace  ->  velocity (bottom)

Differences from spectrogram_plot.py's own figure, per request:
  * NO seam dashed lines.
  * Band y-labels show the Hz range only (no band name).

Usage:
    python plot_from_pkl.py [path_to_trace_data.pkl] [--out OUT.png]
"""
from pathlib import Path
from datetime import datetime
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

# Default pkl (sh7 ch9, artifact-removed + concatenated).
DEFAULT_PKL = (r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260313"
               r"\CnL42SG_20260313\low_freq\spectrogram"
               r"\CnL42SG_20260313_sh7_ch009_trace_data_concat.pkl")

# Frequency-range-only labels (no band names).
BAND_LABELS = {
    'delta':       '0.5-4 Hz',
    '4_25':        '4-25 Hz',
    'theta_ratio': '5-10 / 2-15 Hz',
    'sigma':       '9-25 Hz',
    'gamma':       '40-100 Hz',
    'pc1':         'PC1',
}

CMAP = 'jet'
FREQ_MIN, FREQ_MAX = 0.5, 100.0


def parse_args():
    p = argparse.ArgumentParser(description="Plot a spectrogram figure from a trace_data pkl.")
    p.add_argument("pkl", nargs="?", default=DEFAULT_PKL,
                   help="path to a *_trace_data*.pkl exported by spectrogram_plot.py")
    p.add_argument("--out", default=None, help="output PNG path (default: alongside the pkl)")
    return p.parse_args()


args = parse_args()
pkl_path = Path(args.pkl)
print(f"Loading: {pkl_path}")
with open(pkl_path, 'rb') as f:
    d = pickle.load(f)

spec = d['spectrogram']
spec_x, freqs, spec_z = spec['time_s'], spec['freqs_hz'], spec['z']
vmin, vmax = spec['vmin'], spec['vmax']
band_x = d['trace_time_s']
traces = d['traces']                     # dict: key -> array (on band_x)
band_keys = list(traces.keys())
has_vel = 'velocity' in d
x_lo, x_hi = d['time_axis_s']
band_ylim = tuple(d['band_ylim'])

print(f"  channel {d['channel']} shank {d['shank']}  mode={d['remove_mode']}")
print(f"  bands: {band_keys}  velocity={has_vel}")

# --- Layout: spectrogram (tall) + one row per trace + velocity ---
n_trace = len(band_keys) + (1 if has_vel else 0)
n_sub = 1 + n_trace
height_ratios = [2] + [1] * n_trace

fig = plt.figure(figsize=(30, 2.0 * n_sub + 2), constrained_layout=True)
gs = fig.add_gridspec(n_sub, 1, height_ratios=height_ratios, hspace=0.3)

# --- Spectrogram ---
ax1 = fig.add_subplot(gs[0])
cmap = plt.get_cmap(CMAP).copy()
cmap.set_bad('white')
im = ax1.pcolormesh(spec_x, freqs, spec_z, shading='nearest',
                    cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_ylabel('Frequency (Hz)', fontsize=18)
ax1.set_ylim([FREQ_MIN, FREQ_MAX])
ax1.set_yscale('log')
ax1.set_yticks([1, 4, 16, 64])
ax1.set_yticklabels(['1', '4', '16', '64'])
ax1.tick_params(axis='y', labelsize=14, length=4)
ax1.set_xlim([x_lo, x_hi])
ax1.set_title(f"Spectrogram (Z-scored) - Ch{d['channel']} (Shank {d['shank']})",
              fontsize=15)
ax1.set_xticklabels([])
for sp in ax1.spines.values():
    sp.set_visible(False)
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Z-scored Power', fontsize=14)
cbar.ax.tick_params(labelsize=12)
cbar.outline.set_visible(False)

axes = [ax1]

# --- Band trace panels (Hz-range labels only, no name, no y-ticks) ---
for i, key in enumerate(band_keys):
    ax = fig.add_subplot(gs[1 + i], sharex=ax1)
    ax.plot(band_x, traces[key], 'k-', linewidth=0.5)
    ax.set_ylabel(BAND_LABELS.get(key, key), fontsize=13)
    ax.set_xlim([x_lo, x_hi])
    ax.set_ylim(band_ylim)
    ax.set_xticklabels([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    axes.append(ax)

# --- Velocity ---
if has_vel:
    ax = fig.add_subplot(gs[1 + len(band_keys)], sharex=ax1)
    v = d['velocity']
    ax.plot(v['time_s'], v['value_cm_s'], 'b-', linewidth=0.5)
    ax.set_ylabel('Velocity\n(cm/s)', fontsize=13)
    ax.set_xlim([x_lo, x_hi])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    axes.append(ax)

axes[-1].set_xlabel(
    'Time (s, artifact-free)' if d['remove_mode'] == 'concatenate' else 'Time (s)',
    fontsize=10)

# --- 500 s scale bar on the bottom panel ---
last = axes[-1]
x_end = x_hi - 100
x_start = x_end - 500
yl = last.get_ylim()
yr = yl[1] - yl[0]
yp = yl[0] - 0.15 * yr
last.plot([x_start, x_end], [yp, yp], color='black', lw=8,
          solid_capstyle='butt', clip_on=False)
last.text((x_start + x_end) / 2, yp - 0.1 * yr, '500s', ha='center', va='top',
          fontsize=14, fontweight='bold', clip_on=False)

# --- Reproducibility stamp ---
stamp = (
    f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by plot_from_pkl.py  |  "
    f"src={pkl_path.name}  |  session={d['session']} sh{d['shank']} ch{d['channel']}  |  "
    f"remove_mode={d['remove_mode']} removed={d['removed_duration_s']:.0f}s "
    f"kept={d['kept_duration_s']:.0f}s  |  reproduce: python plot_from_pkl.py \"{pkl_path}\""
)
fig.text(0.005, 0.001, stamp, fontsize=6, color='0.4', ha='left', va='bottom')

out = Path(args.out) if args.out else pkl_path.with_name(
    pkl_path.stem.replace('_trace_data', '') + '_from_pkl.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
