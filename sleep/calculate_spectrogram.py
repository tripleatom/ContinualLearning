from pathlib import Path
import numpy as np
from scipy import signal

# === CONFIGURATION ===
rec_folder = r"D:\cl\ephys\sleep\CnL42SG_20251112_170949.rec"
session_name = Path(rec_folder).stem.split('.')[0]
shanks = [0, 1, 2, 3, 4, 5, 6, 7]

# === OPTIMAL SPECTROGRAM PARAMETERS FOR SLEEP ANALYSIS ===
# These parameters give:
#   • ~4 s window smoothing
#   • ~0.25 Hz freq resolution
#   • ~0.5 s time steps
spec_params = {
    "nperseg": 1024,       # 1024 samples / 500 Hz ≈ 2.048 s 
    "noverlap": 768,       # 75% overlap = 1.5 s overlap
    "nfft": 2048,          # gives ~0.24 Hz freq resolution
    "scaling": "density",
    "mode": "psd",
}

# === MAIN LOOP ===
for ish in shanks:
    print(f"\n{'='*70}")
    print(f"PROCESSING SHANK {ish}")
    print(f"{'='*70}\n")

    low_freq_folder = Path(rec_folder) / "low_freq"

    # Load LFP file
    lfp_file = low_freq_folder / f"{session_name}_sh{ish}_lfp_traces.npz"
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
    output_file = low_freq_folder / f"{session_name}_sh{ish}_spectrograms.npz"
    print(f"\nSaving → {output_file.name}")

    np.savez(
        output_file,
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

    file_size = output_file.stat().st_size / 1024**2
    print(f"  File size: {file_size:.2f} MB — saved successfully!")

print("\n" + "="*70)
print("ALL SHANKS SPECTROGRAM PROCESSING COMPLETE")
print("="*70)
