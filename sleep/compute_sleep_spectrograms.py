import argparse
import errno
import numpy as np
from scipy import signal

from sleep_pipeline_config import (
    session_name,
    shanks,
    low_freq_folder,
    spec_params,
    sleep_sessions,
    active_sleep_sessions,
    resolve_output_folder,
    resolve_existing_file,
    mirror_on_backup_server,
)

# Saved frequency axis: log-spaced bin centers, reduced from the linear FFT
# grid per channel below. This is the `freqs` every downstream consumer sees.
log_freqs = np.logspace(np.log10(spec_params["log_fmin"]),
                        np.log10(spec_params["log_fmax"]),
                        spec_params["n_log_bins"])


def logbin_rows(Sxx, f, centers):
    """Reduce linear-grid PSD rows (n_freqs, n_times) onto log-spaced centers.

    Bin edges are the geometric midpoints between neighboring centers. Rows of
    the linear grid falling inside a bin are averaged; a bin narrower than the
    linear spacing (the lowest few, where log bins are ~0.05 Hz against the
    0.1 Hz grid) instead gets the PSD linearly interpolated at its center, so
    no bin comes out empty.
    """
    inner = np.sqrt(centers[:-1] * centers[1:])
    edges = np.concatenate(([centers[0] ** 2 / inner[0]], inner,
                            [centers[-1] ** 2 / inner[-1]]))
    out = np.empty((centers.size, Sxx.shape[1]), dtype=Sxx.dtype)
    for i, center in enumerate(centers):
        rows = (f >= edges[i]) & (f < edges[i + 1])
        if rows.any():
            out[i] = Sxx[rows].mean(axis=0)
        else:
            j = np.searchsorted(f, center)
            w = (center - f[j - 1]) / (f[j] - f[j - 1])
            out[i] = (1.0 - w) * Sxx[j - 1] + w * Sxx[j]
    return out

parser = argparse.ArgumentParser(
    description="Compute per-channel spectrograms from extracted LFP traces.")
parser.add_argument(
    "--overwrite", action="store_true",
    help="Recompute even when the spectrograms file already exists "
         "(default: existing outputs are kept and the shank is skipped).")
args = parser.parse_args()

# === MAIN LOOP ===
sessions_to_run = active_sleep_sessions(sleep_sessions)
if not sessions_to_run:
    print("No active sleep sessions (pre/post both start=end=None) - nothing to do.")

for session_key, session_cfg in sessions_to_run.items():
    session_label = f"{session_name}{session_cfg['suffix']}"
    print(f"\n{'#'*70}")
    print(f"SLEEP SESSION: {session_key}")
    print(f"{'#'*70}")

    for ish in shanks:
        # Checked before loading the (large) LFP file - an already-done shank
        # should never pay for that load just to overwrite the same output.
        output_check = resolve_existing_file(
            low_freq_folder / f"{session_label}_sh{ish}_spectrograms.npz")
        if not args.overwrite and output_check.exists():
            print(f"\nShank {ish}: {output_check.name} already exists - "
                  f"skipping ({output_check})")
            continue

        print(f"\n{'='*70}")
        print(f"PROCESSING SHANK {ish}")
        print(f"{'='*70}\n")

        # Load LFP file (falls back to the backup server if it was saved there)
        lfp_file = resolve_existing_file(low_freq_folder / f"{session_label}_sh{ish}_lfp_traces.npz")
        if not lfp_file.exists():
            print(f"WARNING: LFP file not found → {lfp_file}")
            continue

        print(f"Loading LFP: {lfp_file.name}")
        lfp_data = np.load(lfp_file)

        traces = lfp_data["traces"]              # shape: (n_samples, n_channels)
        sampling_rate = int(lfp_data["sampling_rate"])
        channel_ids = lfp_data["channel_ids"]

        if "time_range" in lfp_data:
            start_time = float(lfp_data["time_range"][0])
        else:
            start_time = 0.0

        n_samples, n_channels = traces.shape
        print(f"  Channels: {n_channels}, Samples: {n_samples}, Duration: {n_samples/sampling_rate:.1f} s")

        # === COMPUTE SPECTROGRAMS ===
        print("\nComputing spectrograms...")
        spectrograms = []
        freqs = None
        times = None

        for ch_idx, ch_id in enumerate(channel_ids):
            if (ch_idx + 1) % 4 == 0 or ch_idx == 0:
                print(f"  Channel {ch_id}  ({ch_idx+1}/{n_channels})")

            trace = traces[:, ch_idx]

            # Spectrogram (linear power)
            f, t, Sxx = signal.spectrogram(
                trace,
                fs=sampling_rate,
                nperseg=spec_params["nperseg"],
                noverlap=spec_params["noverlap"],
                nfft=spec_params["nfft"],
                scaling=spec_params["scaling"],
                mode=spec_params["mode"],
            )

            # Save freq/time only once
            if freqs is None:
                freqs = log_freqs.astype("float32")
                times = t.astype("float32")

            # Reduce the full linear grid (0.1 Hz rows up to Nyquist) to the
            # log-spaced axis before keeping anything; float32 for file size.
            spectrograms.append(logbin_rows(Sxx, f, log_freqs).astype("float32"))

        spectrograms = np.array(spectrograms, dtype="float32")   # (n_channels, n_freqs, n_times)

        print(f"\n✓ DONE — spectrograms shape: {spectrograms.shape}")
        print(f"  Frequency axis: {len(freqs)} log-spaced bins, "
              f"{freqs[0]:.2f}-{freqs[-1]:.1f} Hz")
        print(f"  Time resolution: {times[1] - times[0]:.3f} s")

        # === SAVE RESULTS ===
        out_dir = resolve_output_folder(low_freq_folder)
        output_file = out_dir / f"{session_label}_sh{ish}_spectrograms.npz"
        print(f"\nSaving → {output_file.name}")

        savez_kwargs = dict(
            spectrograms=spectrograms,       # linear power (n_channels, n_freqs, n_times)
            freqs=freqs,
            times=times,
            channel_ids=channel_ids,
            sampling_rate=sampling_rate,
            start_time=start_time,
            spec_params=spec_params,
            n_channels=n_channels,
            n_freqs=len(freqs),
            n_times=len(times),
        )

        try:
            np.savez(output_file, **savez_kwargs)
        except OSError as e:
            if e.errno != errno.ENOSPC:
                raise
            backup_dir = mirror_on_backup_server(out_dir)
            if backup_dir is None:
                raise
            backup_dir.mkdir(parents=True, exist_ok=True)
            output_file = backup_dir / output_file.name
            print(f"Out of space while saving - retrying on backup server: {output_file}")
            np.savez(output_file, **savez_kwargs)

        file_size = output_file.stat().st_size / 1024**2
        print(f"  File size: {file_size:.2f} MB — saved successfully!")

print("\n" + "="*70)
print("ALL SLEEP SESSIONS / SHANKS SPECTROGRAM PROCESSING COMPLETE")
print("="*70)
