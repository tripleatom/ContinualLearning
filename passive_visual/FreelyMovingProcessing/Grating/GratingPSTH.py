import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from grating_utils import load_neural_data, resolve_data_path

# ── Load pre-extracted neural data pkl (per-unit spike_data already windowed and
# aligned to stimulus onset by GratingExport.py — no raw sorting/DIO access needed) ──
DATA_PATH = Path(resolve_data_path())
data = load_neural_data(DATA_PATH)

animal_id = data['metadata']['animal_id']
session_id = data['metadata']['session_id']
print(f"Processing {animal_id}/{session_id}  —  {len(data['unit_info'])} unit(s), "
      f"{data['metadata']['n_trials']} trial(s)")

orientations = np.array(data['trial_info']['orientations'])
unique_orientations = np.sort(np.unique(orientations))
n_stim_types = len(unique_orientations)
colors = plt.cm.viridis(np.linspace(0, 1, n_stim_types))
orientation2color = dict(zip(unique_orientations, colors))

print(f"Orientations to analyze: {unique_orientations}")
print(f"Number of stimulus types: {n_stim_types}")

out_folder = DATA_PATH.parent / 'grating_psth'
out_folder.mkdir(parents=True, exist_ok=True)

# Plotting window — must be within the window used when the pkl was extracted
# (GratingExport.py's window_pre/window_post; trimmed further here if narrower).
window_pre = 0.2   # seconds before stimulus onset
window_post = 1.0  # seconds after stimulus onset
bin_width = 0.010  # 10ms bins
bin_edges = np.arange(-window_pre, window_post + bin_width, bin_width)
bin_centers = bin_edges[:-1] + bin_width / 2

sigma_ms = 20  # smoothing width in ms
sigma_bins = sigma_ms / (bin_width * 1000)

run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

for unit_key, trials in tqdm(data['spike_data'].items(), desc="units", unit="units"):
    info = data['unit_info'][unit_key]
    quality = info.get('quality', 'unknown')

    if quality == 'noise':
        continue

    # Group each trial's (window-trimmed) spike times by orientation
    groups = {ori: [] for ori in unique_orientations}
    for t in trials:
        ori = t.get('orientation')
        if ori is None or ori not in groups:
            continue
        spikes = np.asarray(t['spike_times'], dtype=float)
        spikes = spikes[(spikes >= -window_pre) & (spikes < window_post)]
        groups[ori].append(spikes)

    # --- Combined raster + PSTH figure ---
    plt.style.use('default')
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('white')

    # --- Raster plot ---
    y_base = 0
    yticks, ylabels = [], []
    for ori in unique_orientations:
        trial_spikes_list = groups[ori]
        n = len(trial_spikes_list)
        for i, spikes in enumerate(trial_spikes_list):
            if len(spikes) > 0:
                y = y_base + i + 0.5
                ax_raster.scatter(spikes * 1000, np.full_like(spikes, y),
                                  s=8, color=orientation2color[ori], marker='|',
                                  alpha=0.8, linewidth=1.5)
        yticks.append(y_base + n / 2)
        ylabels.append(f"{ori}°")
        y_base += n

    ax_raster.set_ylim(0, y_base)
    ax_raster.set_yticks(yticks)
    ax_raster.set_yticklabels(ylabels, fontsize=11)
    ax_raster.set_ylabel('Trial Block (by orientation)', fontsize=12, fontweight='bold')
    ax_raster.set_title(f"{unit_key} — Quality: {quality}", fontsize=14, fontweight='bold', pad=20)
    ax_raster.grid(True, alpha=0.3, linestyle='--')
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.set_xlim(-window_pre * 1000, window_post * 1000)
    ax_raster.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stimulus onset')

    # --- PSTH with smoothing ---
    for ori in unique_orientations:
        trial_spikes_list = groups[ori]
        allspikes = np.concatenate(trial_spikes_list) if trial_spikes_list else np.array([])
        if len(allspikes) == 0:
            continue
        counts, _ = np.histogram(allspikes, bins=bin_edges)
        rate = counts / (len(trial_spikes_list) * bin_width)  # in Hz
        rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
        ax_psth.plot(bin_centers * 1000, rate_smooth,
                     label=f"{ori}°", color=orientation2color[ori],
                     linewidth=2.5, alpha=0.9)

    ax_psth.set_xlabel('Time from stimulus onset (ms)', fontsize=12, fontweight='bold')
    ax_psth.set_ylabel('Firing rate (Hz)', fontsize=12, fontweight='bold')
    ax_psth.set_title('Peri-Stimulus Time Histogram (smoothed)', fontsize=12, fontweight='bold')
    ax_psth.legend(title='Orientation', title_fontsize=9, fontsize=8,
                  ncol=min(3, n_stim_types), loc='upper right',
                  frameon=True, fancybox=True, shadow=True,
                  bbox_to_anchor=(0.98, 0.98))
    ax_psth.grid(True, alpha=0.3, linestyle='--')
    ax_psth.spines['top'].set_visible(False)
    ax_psth.spines['right'].set_visible(False)
    ax_psth.set_xlim(-window_pre * 1000, window_post * 1000)
    ax_psth.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)

    # Reproducibility stamp
    fig.text(0.01, 0.003,
             f"Generated {run_timestamp} | script=GratingPSTH.py | data={DATA_PATH.name} | "
             f"window=[-{window_pre}s, {window_post}s], bin={bin_width*1000:.0f}ms, "
             f"smooth_sigma={sigma_ms}ms",
             ha='left', va='bottom', fontsize=5, color='0.4')

    # Adjust layout and styling
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.subplots_adjust(hspace=0.3)

    fig.savefig(out_folder / f"{unit_key}_{quality}.png",
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

print("Processing complete!")
print(f"Figures saved to: {out_folder}")
