import errno
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import binary_dilation

from sleep_pipeline_config import rec_folder, session_name, shanks, plot_params
from sleep_pipeline_config import band_params, artifact_params
from sleep_pipeline_config import original_fs as fs
from sleep_pipeline_config import resolve_existing_file, resolve_output_folder, mirror_on_backup_server
from sleep_pipeline_config import sleep_sessions, active_sleep_sessions


# Y-axis labels for the optional trace panels below the spectrogram.
TRACE_META = {
    'pc1':         'PC1\nSpectrogram',
    'theta_ratio': 'Theta ratio\n5-10Hz/2-15Hz',
    'delta':       '0.5-4 Hz\n(Delta)',
    '4_25':        '4-25 Hz',
    'sigma':       '9-25Hz\n(Sigma)',
    'gamma':       '40-100Hz\n(Gamma)',
}


# =====================================================
# BROADBAND ARTIFACT DETECTION (Option B)
# =====================================================
def detect_broadband_artifacts(spec_ch, freqs, times, n_mad=5.0,
                               dilate_sec=5.0, fmax=None):
    """Flag spectrogram time bins that are broadband outliers (across ALL freqs).

    spec_ch : (n_freqs, n_times) linear power for one channel.
    Returns (mask, z) on the spectrogram time base, where mask is True for
    artifact bins and z is the robust z-score of broadband power.

    Two-sided: flags both saturated-HIGH bins (motion/EMG/cable) and dropped-LOW
    bins (signal dropout/disconnection). Both bias the per-frequency z-score
    baseline if left in, so both are excluded.
    """
    fmask = np.ones_like(freqs, bool) if fmax is None else (freqs <= fmax)
    # broadband level: mean power across frequency, in dB
    bb = np.mean(10 * np.log10(spec_ch[fmask] + 1e-12), axis=0)  # (n_times,)
    med = np.median(bb)
    mad = np.median(np.abs(bb - med)) + 1e-12
    z = (bb - med) / (1.4826 * mad)              # robust z-score
    mask = np.abs(z) > n_mad                      # two-sided: high AND low
    if dilate_sec and dilate_sec > 0 and len(times) > 1:
        dt = float(np.median(np.diff(times)))
        k = max(1, int(round(dilate_sec / dt)))
        mask = binary_dilation(mask, np.ones(2 * k + 1, bool))
    return mask, z


def mask_to_spans(times, mask):
    """Convert a boolean mask to a list of (t_start, t_end) span tuples."""
    spans = []
    if not np.any(mask):
        return spans
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    for s, e in zip(starts, ends):
        spans.append((times[s], times[min(e, len(times) - 1)]))
    return spans


def znorm_masked(x, bad):
    """Z-normalize x using only non-artifact samples, then NaN the bad ones."""
    x = np.asarray(x, dtype=float).copy()
    good = ~bad
    if np.any(good):
        mu = np.nanmean(x[good])
        sd = np.nanstd(x[good])
    else:
        mu, sd = np.nanmean(x), np.nanstd(x)
    z = (x - mu) / (sd + 1e-10)
    z[bad] = np.nan
    return z


def make_time_compressor(spans, t_min, t_max):
    """Map real time -> 'artifact-free' time with the (t0, t1) `spans` removed
    and the surviving pieces concatenated.

    Returns (f, seams, total): f(t) is a vectorized real->compressed mapping
    (monotonic, slope 1 in kept regions, flat across removed spans); `seams`
    are the join positions in compressed time; `total` is the kept duration.
    """
    spans = sorted((max(a, t_min), min(b, t_max))
                   for a, b in spans if min(b, t_max) > max(a, t_min))
    xs = [t_min]
    ys = [0.0]
    seams = []
    removed = 0.0
    for (a, b) in spans:
        xs.append(a); ys.append(a - t_min - removed)   # last kept point
        seams.append(ys[-1])                            # join position
        removed += (b - a)
        xs.append(b); ys.append(b - t_min - removed)    # collapses onto seam
    xs.append(t_max); ys.append(t_max - t_min - removed)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    total = float(t_max - t_min - removed)

    def f(t):
        return np.interp(np.asarray(t, float), xs, ys)

    return f, np.asarray(seams, float), total


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LFP spectrogram summary panels.")
    parser.add_argument("--shanks", nargs="+", type=int, default=None)
    parser.add_argument("--channels", nargs="+", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_velocity_file(base_folder: Path):
    candidates = [
        base_folder / "velocity_advanced.pkl",
        base_folder.parent / "velocity_advanced.pkl",
        base_folder.parent / "video" / "velocity_advanced.pkl",
    ]
    # Also check the backup-server mirror of each candidate, in case this
    # session's data was written there (e.g. primary was low on space).
    candidates += [c for c in (mirror_on_backup_server(p) for p in candidates) if c is not None]
    found = first_existing(candidates)
    if found is not None:
        return found

    # rglob under both the primary parent AND its backup mirror (the file may
    # only exist on one of the two servers, under a session-specific name that
    # doesn't match the exact-filename candidates above).
    search_roots = [base_folder.parent]
    mirrored_root = mirror_on_backup_server(base_folder.parent)
    if mirrored_root is not None:
        search_roots.append(mirrored_root)

    matches = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob("*velocity*advanced*.pkl"))
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]

# === CONFIGURATION ===
args = parse_args()
rec_folder = Path(rec_folder)  # Convert to Path object
if args.shanks is not None:
    shanks = args.shanks

# === LOAD SYNCHRONIZATION AND VELOCITY DATA ===
print("Loading velocity and synchronization data...")

# Velocity file
velocity_file = find_velocity_file(rec_folder)
if velocity_file is not None:
    with open(velocity_file, 'rb') as f:
        velocity_data = pickle.load(f)
    velocity_time_raw = velocity_data['time_stamp']
    velocity_raw = velocity_data['velocity']
    print(f"Loaded velocity data: {len(velocity_raw)} samples")
else:
    velocity_time_raw = None
    velocity_raw = None
    print("Velocity file not found")

# Sync times file
sync_times_candidates = [
    rec_folder / "sync_times.pkl",
    rec_folder.parent / "sync_times.pkl",
]
sync_times_candidates += [
    c for c in (mirror_on_backup_server(p) for p in sync_times_candidates) if c is not None
]
sync_times_file = first_existing(sync_times_candidates)
if sync_times_file is not None:
    with open(sync_times_file, 'rb') as f:
        sync_times = pickle.load(f)
    proc_rising_time = sync_times['proc_rising_time']
    SG_rising_time = sync_times['SG_rising_time'] / fs
    print(f"Loaded sync times")
    print(f"  Proc rising time range: {proc_rising_time[0]:.2f} - {proc_rising_time[-1]:.2f} s")
    print(f"  SG rising time range: {SG_rising_time[0]:.2f} - {SG_rising_time[-1]:.2f} s")
else:
    proc_rising_time = None
    SG_rising_time = None
    print("Sync times file not found")

# === SYNCHRONIZE VELOCITY WITH SPECTROGRAM ===
velocity_synced = None
velocity_time_synced = None

if velocity_time_raw is not None and proc_rising_time is not None and SG_rising_time is not None:
    print("\nSynchronizing velocity with spectrogram...")

    # Pick up velocity in the range of proc_rising_time[0:-1]
    vel_mask = (velocity_time_raw >= proc_rising_time[0]) & (velocity_time_raw <= proc_rising_time[-1])
    velocity_synced = velocity_raw[vel_mask]
    velocity_time_synced = velocity_time_raw[vel_mask]

    # Remap velocity time to match spectrogram time (SG_rising_time[0] to SG_rising_time[-1])
    # Linear mapping from proc_rising_time to SG_rising_time
    velocity_time_synced = np.interp(velocity_time_synced,
                                     [proc_rising_time[0], proc_rising_time[-1]],
                                     [SG_rising_time[0], SG_rising_time[-1]])

    print(f"Synchronized velocity")
    print(f"  Velocity samples: {len(velocity_synced)}")
    print(f"  Velocity time range: {velocity_time_synced[0]:.2f} - {velocity_time_synced[-1]:.2f} s")
    print(f"  Spectrogram time range: {SG_rising_time[0]:.2f} - {SG_rising_time[-1]:.2f} s")

    plot_velocity = True
else:
    plot_velocity = False
    print("\nCannot synchronize velocity - missing data or sync times")

# === LOOP THROUGH ALL ACTIVE SLEEP SESSIONS (pre/post) ===
# Each session has its own band_powers pkl (suffixed, e.g. "..._post_all_shanks_band_powers.pkl")
# and every output filename below is suffixed the same way, so pre/post plots
# and trace-data exports never collide/overwrite each other.
sessions_to_run = active_sleep_sessions(sleep_sessions)
if not sessions_to_run:
    print("No active sleep sessions (pre/post both start=end=None) - nothing to do.")

total_files_created = 0
for session_key, session_cfg in sessions_to_run.items():
  session_label = f"{session_name}{session_cfg['suffix']}"
  print(f"\n{'#'*60}")
  print(f"SLEEP SESSION: {session_key}")
  print(f"{'#'*60}")

  # === LOAD DATA ===
  low_freq_folder = rec_folder / "low_freq"
  print(f"\nLoading data from: {low_freq_folder}")

  # Load computed band powers and spectrograms (all in one pickle file).
  # Falls back to the backup server if a previous stage saved there due to low space.
  band_powers_file = resolve_existing_file(low_freq_folder / f'{session_label}_all_shanks_band_powers.pkl')
  print(f"Loading data from: {band_powers_file}")

  with open(band_powers_file, 'rb') as f:
      all_data = pickle.load(f)

  # Use wherever the band-powers file actually was (primary or backup server) as
  # the base for outputs too, so figures land next to their source data.
  low_freq_folder = band_powers_file.parent

  # === CREATE OUTPUT FOLDER ===
  output_folder = resolve_output_folder(low_freq_folder / "spectrogram")
  print(f"\nSaving plots to: {output_folder}")

  # === LOOP THROUGH ALL SHANKS ===
  session_files_created = 0
  for shank_id in shanks:
      print(f"\n{'='*60}")
      print(f"PROCESSING SHANK {shank_id}")
      print(f"{'='*60}")

      # Check if shank data exists
      if shank_id not in all_data['shanks_data']:
          print(f"Shank {shank_id} not found in data, skipping...")
          continue

      shank_data = all_data['shanks_data'][shank_id]

      # Extract all needed data
      lfp_time = shank_data['lfp_time']
      pc1_spectrogram = shank_data['pc1_spectrogram']
      channel_ids = shank_data['channel_ids']
      times = shank_data['spectrogram_times']
      freqs = shank_data['spectrogram_freqs']
      spectrograms = shank_data['spectrograms']  # (n_channels, n_freqs, n_times)
      sampling_rate = shank_data['sampling_rate']

      print(f"\nLoaded data for Shank {shank_id}:")
      print(f"  Spectrograms shape: {spectrograms.shape}")
      print(f"  Sampling rate: {sampling_rate} Hz")
      print(f"  Total duration: {lfp_time[-1]:.1f} s")
      print(f"  Number of channels: {len(channel_ids)}")

      # === CROP DATA TO SYNCHRONIZED TIME RANGE ===
      if plot_velocity and SG_rising_time is not None:
          print(f"\nCropping data to synchronized time range...")

          # Crop spectrogram and times to SG_rising_time range
          spec_mask = (times >= SG_rising_time[0]) & (times <= SG_rising_time[-1])
          times_cropped = times[spec_mask]
          spectrograms_cropped = spectrograms[:, :, spec_mask]

          # Crop LFP-based data (band powers, PC1) to same range
          lfp_mask = (lfp_time >= SG_rising_time[0]) & (lfp_time <= SG_rising_time[-1])
          lfp_time_cropped = lfp_time[lfp_mask]
          pc1_cropped = pc1_spectrogram[:,lfp_mask]

          print(f"  Cropped spectrogram time: {times_cropped[0]:.2f} - {times_cropped[-1]:.2f} s")
          print(f"  Cropped LFP time: {lfp_time_cropped[0]:.2f} - {lfp_time_cropped[-1]:.2f} s")
          print(f"  Velocity time: {velocity_time_synced[0]:.2f} - {velocity_time_synced[-1]:.2f} s")
      else:
          # Use full data if no synchronization
          times_cropped = times
          spectrograms_cropped = spectrograms
          lfp_time_cropped = lfp_time
          pc1_cropped = pc1_spectrogram

          # === COLOR SCALE WILL BE DETERMINED PER CHANNEL AFTER Z-SCORING ===

      # === PLOTTING ===
      total_duration = lfp_time_cropped[-1] - lfp_time_cropped[0]
      print(f"\nProcessing {len(channel_ids)} channels, full recording ({total_duration:.1f}s each)...")

      # Time range for full recording
      t_start = lfp_time_cropped[0]
      t_end = lfp_time_cropped[-1]

      # Determine subplot layout: spectrogram (tall) + one row per selected trace
      # panel + velocity (if available).
      trace_panels = list(plot_params.get('trace_panels',
                                          ['pc1', 'theta_ratio', 'delta',
                                           'sigma', 'gamma']))
      n_trace = len(trace_panels) + (1 if plot_velocity else 0)
      n_subplots = 1 + n_trace
      height_ratios = [2] + [1] * n_trace

      channel_indices = list(range(len(channel_ids)))
      if args.channels is not None:
          requested = set(args.channels)
          channel_indices = [
              idx for idx in channel_indices
              if int(channel_ids[idx]) in requested
          ]
      if args.max_channels is not None:
          channel_indices = channel_indices[:args.max_channels]
      if not channel_indices:
          print("  No channels selected for this shank, skipping...")
          continue

      # Loop through selected channels
      for n_selected, ch_idx in enumerate(channel_indices, start=1):
          ch_id = channel_ids[ch_idx]
          print(f"\n=== Processing Channel {ch_id} ({n_selected}/{len(channel_indices)} selected) ===")

          # Get data for this channel (already cropped)
          channel_spectrogram = spectrograms_cropped[ch_idx, :, :]

          # --- Broadband artifact detection (Option B) ---
          if artifact_params['enabled']:
              art_mask, art_z = detect_broadband_artifacts(
                  channel_spectrogram, freqs, times_cropped,
                  n_mad=artifact_params['n_mad'],
                  dilate_sec=artifact_params['dilate_sec'],
                  fmax=artifact_params['fmax'],
              )
              # Map the mask onto the (higher-res) LFP time base for band panels
              art_mask_lfp = np.interp(
                  lfp_time_cropped, times_cropped, art_mask.astype(float)) > 0.5
              # Optional velocity gate
              if (artifact_params['velocity_threshold'] is not None
                      and plot_velocity):
                  vel_on_lfp = np.interp(
                      lfp_time_cropped, velocity_time_synced, velocity_synced)
                  art_mask_lfp |= vel_on_lfp > artifact_params['velocity_threshold']
              artifact_spans = mask_to_spans(times_cropped, art_mask)
              frac = 100.0 * np.mean(art_mask)
              print(f"  Artifacts: {len(artifact_spans)} spans, "
                    f"{frac:.1f}% of bins flagged (n_mad={artifact_params['n_mad']})")
          else:
              art_mask = np.zeros(len(times_cropped), bool)
              art_mask_lfp = np.zeros(len(lfp_time_cropped), bool)
              artifact_spans = []
              frac = 0.0

          # Display transform: dB power, robustly z-scored PER FREQUENCY over
          # non-artifact bins. Linear power is dominated by the 1/f background and
          # by total-power swings, so a single global z-score just shows "total
          # power over time" (~movement). Normalizing each frequency row whitens
          # the 1/f so state-dependent band changes (delta rising in NREM, etc.)
          # become visible, as in Buzsaki's spectrogram.
          good_cols = ~art_mask
          if not np.any(good_cols):
              good_cols = np.ones(channel_spectrogram.shape[1], bool)
          log_spec = 10 * np.log10(channel_spectrogram + 1e-12)
          med_f = np.median(log_spec[:, good_cols], axis=1, keepdims=True)
          mad_f = np.median(np.abs(log_spec[:, good_cols] - med_f),
                            axis=1, keepdims=True)
          channel_spectrogram_zscored = (log_spec - med_f) / (1.4826 * mad_f + 1e-10)

          # How to handle artifact periods in the figure (see sleep_pipeline_config):
          #   'blank'       -> NaN the flagged bins so they render as white gaps
          #   'concatenate' -> drop the flagged bins and stitch the rest together
          remove_mode = artifact_params.get('remove_mode', 'blank') \
              if artifact_params['enabled'] else 'none'
          if remove_mode == 'blank':
              channel_spectrogram_zscored[:, art_mask] = np.nan

          # Print z-scored data range (over the kept, non-artifact bins)
          print(f"  Z-scored spectrogram range: [{np.nanmin(channel_spectrogram_zscored):.3f}, {np.nanmax(channel_spectrogram_zscored):.3f}]")


          # Determine color scale from the kept (non-artifact) bins only.
          zscored_values = channel_spectrogram_zscored[:, good_cols]
          zscored_values = zscored_values[np.isfinite(zscored_values)]

          if plot_params['color_scale_method'] == 'adaptive':
              median_val = np.median(zscored_values)
              mad = np.median(np.abs(zscored_values - median_val))
              vmin_base = median_val - plot_params['adaptive_n_mad'] * mad
              vmax_base = median_val + plot_params['adaptive_n_mad'] * mad

              # Extend by configured percentages
              range_val = vmax_base - vmin_base
              vmin = vmin_base - plot_params['vmin_extension'] * range_val
              vmax = vmax_base + plot_params['vmax_extension'] * range_val

          elif plot_params['color_scale_method'] == 'percentile':
              vmin_base = np.percentile(zscored_values, plot_params['vmin_percentile'])
              vmax_base = np.percentile(zscored_values, plot_params['vmax_percentile'])

              # Extend by configured percentages
              range_val = vmax_base - vmin_base
              vmin = vmin_base - plot_params['vmin_extension'] * range_val
              vmax = vmax_base + plot_params['vmax_extension'] * range_val

          else:  # manual
              vmin = plot_params['vmin_manual']
              vmax = plot_params['vmax_manual']

          print(f"  Color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

          # Load band powers for this channel from the nested dictionary
          bands_data_full = {
              'delta': shank_data['band_powers'][ch_id]['delta'],
              'theta_ratio': shank_data['band_powers'][ch_id]['theta_ratio'],
              'sigma': shank_data['band_powers'][ch_id]['sigma'],
              'gamma': shank_data['band_powers'][ch_id]['gamma'],
          }

          # Crop band powers to synchronized time range
          bands_data = {}
          for band_name, band_values in bands_data_full.items():
              bands_data[band_name] = band_values[lfp_mask] if plot_velocity else band_values

          # Custom 4-25 Hz band, integrated from the displayed spectrogram (the
          # pipeline bands come from LFP bandpass; this one is added on the fly,
          # smoothed with the same window, then put on the LFP time base).
          if '4_25' in trace_panels:
              fsel = (freqs >= 4) & (freqs <= 25)
              bp = channel_spectrogram[fsel, :].mean(axis=0)        # linear power
              dt_spec = float(np.median(np.diff(times_cropped)))
              w = max(1, int(round(band_params['smoothing_window'] / dt_spec)))
              bp = np.convolve(bp, np.ones(w) / w, mode='same')
              bands_data['4_25'] = np.interp(lfp_time_cropped, times_cropped, bp)

          # --- Build the time axis: concatenate (excise artifacts) or keep full ---
          # spec_x/spec_z drive the spectrogram; band_x + band_sel drive the trace
          # panels; vel_x + vel_sel drive velocity; [x_lo, x_hi] is the shared xlim.
          if remove_mode == 'concatenate' and artifact_spans:
              compress, seams_c, total_c = make_time_compressor(
                  artifact_spans, t_start, t_end)
              keep_spec = ~art_mask
              keep_lfp = ~art_mask_lfp
              spec_x = compress(times_cropped[keep_spec])
              spec_z = channel_spectrogram_zscored[:, keep_spec]
              band_x = compress(lfp_time_cropped[keep_lfp])
              band_sel = keep_lfp
              x_lo, x_hi = 0.0, total_c
              if plot_velocity:
                  vel_art = np.interp(velocity_time_synced, times_cropped,
                                      art_mask.astype(float)) > 0.5
                  vel_sel = ~vel_art
                  vel_x = compress(velocity_time_synced[vel_sel])
              removed_s = (t_end - t_start) - total_c
              print(f"  Concatenated: removed {removed_s:.0f}s of artifacts, "
                    f"{len(seams_c)} seams, kept timeline {total_c:.0f}s")
          else:
              seams_c = np.array([])
              spec_x = times_cropped
              spec_z = channel_spectrogram_zscored
              band_x = lfp_time_cropped
              band_sel = np.ones(len(lfp_time_cropped), bool)
              x_lo, x_hi = t_start, t_end
              if plot_velocity:
                  vel_sel = np.ones(len(velocity_time_synced), bool)
                  vel_x = velocity_time_synced

          # Create figure
          print(f"  Creating full recording plot...")

          # Create figure with subplots
          fig = plt.figure(figsize=plot_params['figsize'], constrained_layout=True)
          gs = fig.add_gridspec(n_subplots, 1, height_ratios=height_ratios, hspace=0.3)

          subplot_idx = 0

          # 1. Spectrogram
          ax1 = fig.add_subplot(gs[subplot_idx])
          subplot_idx += 1

          # Colormap with NaN (removed artifact bins) drawn as white gaps.
          spec_cmap = plt.get_cmap(plot_params['cmap']).copy()
          spec_cmap.set_bad('white')
          im = ax1.pcolormesh(
              spec_x,
              freqs,
              spec_z,
              shading='nearest',
              cmap=spec_cmap,
              vmin=vmin,
              vmax=vmax
          )

          ax1.set_ylabel('Frequency (Hz)', fontsize=18)
          ax1.set_ylim([plot_params['freq_min'], plot_params['freq_max']])
          ax1.set_yscale('log')
          ax1.set_yticks([1, 4, 16, 64])
          ax1.set_yticklabels(['1', '4', '16', '64'])
          ax1.tick_params(axis='y', labelsize=14, length=4)
          ax1.set_xlim([x_lo, x_hi])
          ax1.set_title(f'Spectrogram (Z-scored) - Ch{ch_id} (Shank {shank_id})', fontsize=15)
          ax1.set_xticklabels([])
          # Publication: no box outline around the spectrogram.
          for spine in ax1.spines.values():
              spine.set_visible(False)

          cbar = plt.colorbar(im, ax=ax1)
          cbar.set_label('Z-scored Power', fontsize=14)
          cbar.ax.tick_params(labelsize=12)
          cbar.outline.set_visible(False)



          # --- Trace panels (selectable via plot_params['trace_panels']) ---
          # trace_export holds the exact arrays drawn (kept/concatenated samples).
          trace_axes = []
          trace_export = {}
          for key in trace_panels:
              raw = pc1_cropped[ch_idx, :] if key == 'pc1' else bands_data[key]
              norm = znorm_masked(raw, art_mask_lfp)
              ax = fig.add_subplot(gs[subplot_idx], sharex=ax1)
              subplot_idx += 1
              ax.plot(band_x, norm[band_sel], 'k-', linewidth=0.5)
              ax.set_ylabel(TRACE_META.get(key, key), fontsize=13)
              ax.set_xlim([x_lo, x_hi])
              ax.set_ylim(plot_params['band_ylim'])
              ax.set_xticklabels([])
              ax.set_yticks([])
              for spine in ax.spines.values():
                  spine.set_visible(False)
              ax.tick_params(left=False, bottom=False)
              trace_axes.append(ax)
              trace_export[key] = norm[band_sel]

          # Velocity (if available)
          ax_vel = None
          if plot_velocity:
              ax_vel = fig.add_subplot(gs[subplot_idx], sharex=ax1)
              subplot_idx += 1
              ax_vel.plot(vel_x, velocity_synced[vel_sel], 'b-', linewidth=0.5)
              ax_vel.set_ylabel('Velocity\n(cm/s)', fontsize=13)
              ax_vel.set_xlim([x_lo, x_hi])
              for spine in ax_vel.spines.values():
                  spine.set_visible(False)
              ax_vel.tick_params(left=False, bottom=False)
              trace_axes.append(ax_vel)

          # X-axis label on the bottom-most panel.
          xlabel_txt = 'Time (s, artifact-free)' if remove_mode == 'concatenate' else 'Time (s)'
          trace_axes[-1].set_xlabel(xlabel_txt, fontsize=10)

          panel_axes = [ax1] + trace_axes

          if remove_mode == 'concatenate':
              # Mark the seams where clean segments were stitched together.
              for ax in panel_axes:
                  for s in seams_c:
                      ax.axvline(s, color='0.65', lw=0.6, ls=(0, (4, 3)),
                                 zorder=4)
          elif remove_mode == 'blank' and artifact_params['enabled'] and artifact_spans:
              # --- Shade broadband-artifact spans across every panel ---
              for ax in panel_axes:
                  for (t0, t1) in artifact_spans:
                      ax.axvspan(t0, t1, color=artifact_params['shade_color'],
                                 alpha=artifact_params['shade_alpha'],
                                 lw=0, zorder=5)

          # Add visible scale bar for 500s
          last_ax = fig.get_axes()[-1]

          # Calculate scale bar position and length
          # Draw the scale bar in data coordinates
          scale_length = 500  # seconds
          x_end = x_hi - 100  # 100s from the right edge
          x_start = x_end - scale_length

          # Get the y-axis limits
          y_limits = last_ax.get_ylim()
          y_range = y_limits[1] - y_limits[0]
          y_pos = y_limits[0] - 0.15 * y_range  # Position below the plot

          # Draw the scale bar
          last_ax.plot([x_start, x_end], [y_pos, y_pos],
                      color='black', linewidth=8, solid_capstyle='butt',
                      clip_on=False)  # Important: don't clip the line

          # Add text label below the bar
          last_ax.text((x_start + x_end) / 2, y_pos - 0.1 * y_range, '500s',
                      ha='center', va='top', fontsize=14, fontweight='bold',
                      clip_on=False)  # Important: don't clip the text

          # Save figure (session_label keeps pre/post outputs from colliding)
          output_file = output_folder / f'{session_label}_sh{shank_id}_ch{ch_id:03d}_full_recording{args.output_suffix}.png'

          # Reproducibility stamp embedded in the figure
          stamp = (
              f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by plot_sleep_spectrograms.py  |  "
              f"session={session_name} sleep_session={session_key} shank={shank_id} ch={ch_id}  |  "
              f"source={band_powers_file}  |  "
              f"artifact[enabled={artifact_params['enabled']} n_mad={artifact_params['n_mad']} "
              f"fmax={artifact_params['fmax']} dilate={artifact_params['dilate_sec']}s "
              f"vel_thr={artifact_params['velocity_threshold']}] -> "
              f"{len(artifact_spans)} spans, {frac:.1f}% bins flagged  |  "
              f"remove_mode={remove_mode}"
          )
          fig.text(0.005, 0.001, stamp, fontsize=6, color='0.4',
                   ha='left', va='bottom')

          # --- Export the exact plotted data (traces + spectrogram + velocity) ---
          export = {
              'session': session_name,
              'sleep_session': session_key,
              'shank': shank_id,
              'channel': int(ch_id),
              'remove_mode': remove_mode,
              'artifact': {
                  'n_mad': artifact_params['n_mad'],
                  'fmax': artifact_params['fmax'],
                  'dilate_sec': artifact_params['dilate_sec'],
                  'velocity_threshold': artifact_params['velocity_threshold'],
                  'n_spans': len(artifact_spans),
                  'frac_flagged_pct': float(frac),
              },
              'time_full_s': (float(t_start), float(t_end)),
              'time_axis_s': (float(x_lo), float(x_hi)),
              'kept_duration_s': float(x_hi - x_lo),
              'removed_duration_s': float((t_end - t_start) - (x_hi - x_lo)),
              'seams_s': seams_c,
              'band_ylim': plot_params['band_ylim'],
              'spectrogram': {
                  'time_s': spec_x, 'freqs_hz': freqs, 'z': spec_z,
                  'vmin': float(vmin), 'vmax': float(vmax),
              },
              # All trace panels share band_x; values are z-normalized (SD units).
              'trace_time_s': band_x,
              'traces': trace_export,
          }
          if plot_velocity:
              export['velocity'] = {
                  'time_s': vel_x, 'value_cm_s': velocity_synced[vel_sel],
              }

          # Real-time (full timeline, BEFORE artifact removal/concatenation):
          # raw band power on the actual recording time base, plus raw velocity.
          # These are NOT z-scored and NOT excised (artifact periods included).
          realtime_bands = {}
          for key in trace_panels:
              raw = pc1_cropped[ch_idx, :] if key == 'pc1' else bands_data.get(key)
              if raw is not None:
                  realtime_bands[key] = np.asarray(raw)
          export['realtime'] = {
              'lfp_time_s': lfp_time_cropped,        # real recording time (s)
              'band_power': realtime_bands,          # raw power per band, full
              'artifact_mask_lfp': art_mask_lfp,     # True where flagged artifact
          }
          if plot_velocity:
              export['realtime']['velocity_time_s'] = velocity_time_synced
              export['realtime']['velocity_cm_s'] = velocity_synced

          trace_pkl = output_folder / (
              f'{session_label}_sh{shank_id}_ch{ch_id:03d}_trace_data'
              f'{args.output_suffix}.pkl')
          try:
              with open(trace_pkl, 'wb') as f:
                  pickle.dump(export, f)
              print(f"  Exported trace data: {trace_pkl.name}")

              print(f"  Saving to: {output_file.name}")
              plt.savefig(output_file, dpi=plot_params['dpi'], bbox_inches='tight')
          except OSError as e:
              if e.errno != errno.ENOSPC:
                  raise
              backup_folder = mirror_on_backup_server(output_folder)
              if backup_folder is None:
                  raise
              backup_folder.mkdir(parents=True, exist_ok=True)
              output_folder = backup_folder
              trace_pkl = output_folder / trace_pkl.name
              output_file = output_folder / output_file.name
              print(f"Out of space while saving - switching to backup server: {output_folder}")
              with open(trace_pkl, 'wb') as f:
                  pickle.dump(export, f)
              print(f"  Exported trace data: {trace_pkl.name}")
              print(f"  Saving to: {output_file.name}")
              plt.savefig(output_file, dpi=plot_params['dpi'], bbox_inches='tight')
          plt.close()
          session_files_created += 1
          total_files_created += 1

          print(f"  Completed Shank {shank_id}, Channel {ch_id}")

  print(f"\n{'='*60}")
  print(f"SLEEP SESSION {session_key} PLOTTING COMPLETE")
  print(f"{'='*60}")
  print(f"Output directory: {output_folder}")
  print(f"Files created this session: {session_files_created} (one full recording per channel per shank)")
  print(f"Processed shanks: {shanks}")
  if plot_velocity:
      print(f"\nVelocity data included and synchronized")
  else:
      print(f"\nVelocity data not included")

print(f"\n{'='*60}")
print("ALL SLEEP SESSIONS PLOTTING COMPLETE")
print(f"{'='*60}")
print(f"Total files created: {total_files_created}")
