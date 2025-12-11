# **Spectrogram Processing Pipeline for Sleep LFP Data**

This script computes time–frequency spectrograms for all channels across multiple shanks in a `.rec` electrophysiology recording. It uses parameters optimized for sleep analysis, producing stable slow-wave and spindle-band representations.

---

## **Overview**

The pipeline performs the following steps:

1. **Identify recording session and shanks** from the `.rec` folder.
2. **Load LFP traces** for each shank from the `low_freq/` directory.
3. **Compute spectrograms** channel-by-channel using `scipy.signal.spectrogram` with sleep-optimized parameters.
4. **Save output** (spectrograms + metadata) into compressed `.npz` files for further analysis (e.g., SWS detection, sleep scoring, coherence analysis).

---

## **Key Features**

### ✔ Sleep-optimized spectrogram settings

* **Window (nperseg = 1024)** → ~2.0 s windows at 500 Hz
* **75% overlap** → smooth time transitions
* **FFT size (2048)** → ~0.24 Hz frequency resolution
* Produces ~0.5 s time resolution suitable for NREM, REM, SWS detection.

### ✔ Efficient processing

* Computes spectrograms per channel for each shank.
* Saves frequency and time vectors only once to reduce redundancy.
* Stores floating-point values as `float32` to reduce file size.

### ✔ Structured output

For each shank, the script writes:

```
<session>_shX_spectrograms.npz
```

Containing:

* `spectrograms` — array of shape (n_channels, n_freqs, n_times)
* `freqs`, `times` — frequency/time axes
* `channel_ids`
* `sampling_rate`, `start_time`
* `spec_params`
* Basic metadata (n_channels, n_freqs, n_times)

---

## **Workflow**

1. **Loop over shanks**
   For each shank ID, check whether the LFP file exists.

2. **Load LFP data**

   ```
   traces: (n_samples, n_channels)
   sampling_rate: typically 500 Hz
   channel_ids: per-channel identifier
   ```

3. **Compute spectrogram per channel**

   * Extract individual channel trace
   * Run `signal.spectrogram()`
   * Store the output matrix (power vs frequency vs time)

4. **Aggregate and save**

   * Stack all channel spectrograms into a 3D array
   * Save to disk in compressed `.npz` format

---

## **Output Example**

A typical saved spectrogram file includes:

```
spectrograms: float32 array (64, 1025, 20000)
freqs: float32 array (1025,)
times: float32 array (~20000,)
channel_ids: int array
sampling_rate: int
start_time: float
```

---

## **Use Cases**

* Sleep stage classification (REM/NREM/SWS)
* Delta/theta/beta/gamma power tracking
* Coherence analysis across brain vs spinal channels
* Visualizing oscillatory bursts (sleep spindles, slow waves)
