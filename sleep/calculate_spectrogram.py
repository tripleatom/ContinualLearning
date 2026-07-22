import errno
import numpy as np
from scipy import signal

from sleep_params import (
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
                freqs = f.astype("float32")
                times = t.astype("float32")

            # Convert to float32 to reduce file size
            spectrograms.append(Sxx.astype("float32"))

        spectrograms = np.array(spectrograms, dtype="float32")   # (n_channels, n_freqs, n_times)

        print(f"\n✓ DONE — spectrograms shape: {spectrograms.shape}")
        print(f"  Frequency resolution: {freqs[1] - freqs[0]:.3f} Hz")
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
