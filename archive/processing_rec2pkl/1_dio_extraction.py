# ================================================================
# dio_extract_clean_segments_samples_with_dtplots.py
# Extract DIO edges from .rec (in SAMPLES), optionally clean,
# optionally segment by long gaps, visualize Δt between rises,
# and save NPZ(s). No TXT verification.
# ================================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# ---- DIO helpers ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from process_func.DIO import get_dio_folders, concatenate_din_data  # noqa: E402

# ======================
# User config
# ======================
rec_folder        = Path(r"Z:\xl_cl\Albert\20251022_psv;hf;2grtings\CnL42SG\CnL42SG_20251022_160759.rec")
pd_channel        = 3
fs                = 30000  # Hz
save_dir          = rec_folder.parent
out_stem          = rec_folder.stem + "_DIO_samples"

# Falls choice
use_real_falls        = False   # False → fall = rise + stimulus_duration*fs ; True → use measured falls
stimulus_duration_s   = 1.0     # used if use_real_falls == False

# Cleaning options
auto_clean            = True
min_high_s            = 0.10    # drop highs with ON duration < this
min_low_s             = 0.10    # collapse LOW gaps < this by deleting (fall_i, rise_{i+1})

# Manual fixes (indices after auto_clean)
manual_drop_rise_idxs: list[int] = []  # e.g., [56]
manual_drop_fall_idxs: list[int] = []  # e.g., [55]

# Segmentation
segment_by_gaps       = True
gap_threshold_s       = 10.0    # split when Δrise > this

# Δt plot options
make_dt_plots         = True
dt_tolerance_s        = 1000    # tolerance around median for flagging
save_plots            = True


# ======================
# Utilities
# ======================
@dataclass
class CleanLog:
    raw_highs: int
    dropped_short_high_idxs: list
    removed_low_blips_pairs: list
    manual_drop_rise_idxs: list
    manual_drop_fall_idxs: list
    clean_highs: int


def load_pd_samples(rec: Path, din_ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_samples:int64, state:int8) anchored to 0."""
    dio_folders = sorted(get_dio_folders(rec), key=lambda p: p.name)
    pd_time, pd_state = concatenate_din_data(dio_folders, din_ch)
    pd_time = pd_time - pd_time[0]
    return pd_time.astype(np.int64), np.asarray(pd_state, np.int8)


def edges_from_transitions(time_samples: np.ndarray, state01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect rising/falling edges from state transitions; return paired arrays in samples."""
    ds = np.diff(state01, prepend=state01[0])
    r = time_samples[ds == 1]
    f = time_samples[ds == -1]
    if r.size and f.size and f[0] < r[0]:
        f = f[1:]
    m = min(len(r), len(f))
    return r[:m].astype(np.int64), f[:m].astype(np.int64)


def clean_dio_simple_samples(r_samples: np.ndarray,
                             f_samples: np.ndarray,
                             fs_hz: int,
                             min_high_s: float,
                             min_low_s: float) -> tuple[np.ndarray, np.ndarray, CleanLog]:
    """Drop short highs and collapse short LOW blips; all in samples."""
    r = r_samples.copy()
    f = f_samples.copy()
    n0 = min(len(r), len(f))
    r, f = r[:n0], f[:n0]
    if n0 == 0:
        return r, f, CleanLog(0, [], [], [], [], 0)

    # Drop short highs
    dur = f - r
    keep_high = dur >= int(round(min_high_s * fs_hz))
    dropped = np.where(~keep_high)[0].tolist()
    r1, f1 = r[keep_high], f[keep_high]

    # Collapse short LOW blips
    removed_pairs = []
    if len(r1) > 1:
        low = r1[1:] - f1[:-1]
        to_remove = np.where(low < int(round(min_low_s * fs_hz)))[0]
        keep_r = np.ones(len(r1), dtype=bool)
        keep_f = np.ones(len(f1), dtype=bool)
        for i in to_remove:
            keep_f[i] = False
            keep_r[i + 1] = False
            removed_pairs.append((int(i), int(i + 1)))
        r2 = r1[keep_r]
        f2 = f1[keep_f]
    else:
        r2, f2 = r1, f1

    log = CleanLog(
        raw_highs=int(len(r)),
        dropped_short_high_idxs=dropped,
        removed_low_blips_pairs=removed_pairs,
        manual_drop_rise_idxs=[],
        manual_drop_fall_idxs=[],
        clean_highs=int(len(r2)),
    )
    return r2, f2, log


def apply_manual_drops(r_samples: np.ndarray,
                       f_samples: np.ndarray,
                       drop_rise_idxs: list[int],
                       drop_fall_idxs: list[int],
                       log: CleanLog) -> tuple[np.ndarray, np.ndarray, CleanLog]:
    """Apply manual index removals, then re-pair."""
    r = r_samples
    f = f_samples
    if drop_rise_idxs:
        keep_r = np.ones(len(r), dtype=bool)
        keep_r[drop_rise_idxs] = False
        r = r[keep_r]
    if drop_fall_idxs:
        keep_f = np.ones(len(f), dtype=bool)
        keep_f[drop_fall_idxs] = False
        f = f[keep_f]
    m = min(len(r), len(f))
    r, f = r[:m], f[:m]
    log.manual_drop_rise_idxs = list(drop_rise_idxs)
    log.manual_drop_fall_idxs = list(drop_fall_idxs)
    log.clean_highs = int(m)
    return r, f, log


def synthesize_falls_from_duration(r_samples: np.ndarray, fs_hz: int, stim_dur_s: float) -> np.ndarray:
    """Create ideal falls from rise times and fixed duration."""
    return r_samples + int(round(stim_dur_s * fs_hz))


def split_by_long_gaps(r_samples: np.ndarray, fs_hz: int, gap_threshold_s: float) -> list[np.ndarray]:
    """Return list of index arrays for each segment."""
    if len(r_samples) <= 1:
        return [np.arange(len(r_samples))]
    d = np.diff(r_samples) / fs_hz
    cut = np.where(d > gap_threshold_s)[0]
    return np.split(np.arange(len(r_samples)), cut + 1)


def save_npz_samples(path: Path,
                     rising_samples: np.ndarray,
                     falling_samples: np.ndarray,
                     fs_hz: int,
                     log: CleanLog,
                     segment_index: int | None = None):
    np.savez_compressed(
        path,
        rising_samples=rising_samples.astype(np.int64),
        falling_samples=falling_samples.astype(np.int64),
        fs=np.int64(fs_hz),
        segment_index=(np.int64(segment_index) if segment_index is not None else -1),
        cleaner_log=dict(
            raw_highs=log.raw_highs,
            dropped_short_high_idxs=log.dropped_short_high_idxs,
            removed_low_blips_pairs=log.removed_low_blips_pairs,
            manual_drop_rise_idxs=log.manual_drop_rise_idxs,
            manual_drop_fall_idxs=log.manual_drop_fall_idxs,
            clean_highs=log.clean_highs,
        ),
    )


def plot_rise_dt_diagnostics(rising_edges, fs_hz, *,
                             gap_threshold_s: float,
                             tolerance_s: float,
                             title_prefix: str,
                             save_path: Path | None,
                             show: bool):
    """Plot Δt between rising edges (seconds) as time-series + histogram."""
    rising_edges = np.asarray(rising_edges)
    in_samples = (rising_edges.dtype.kind in "iu") or (rising_edges.max() > 1e4)
    t_rise_s = rising_edges / fs_hz if in_samples else rising_edges.astype(float)

    if t_rise_s.size < 2:
        print("[Δt] not enough rises to compute deltas.")
        return

    dt = np.diff(t_rise_s)
    n = dt.size
    med = float(np.median(dt))
    p25, p75 = np.percentile(dt, [25, 75])
    iqr = float(p75 - p25)
    print(f"[Δt stats] n={n}  median={med:.3f}s  IQR={iqr:.3f}s  min={dt.min():.3f}s  max={dt.max():.3f}s")

    idx_long = np.where(dt > gap_threshold_s)[0]
    idx_outlier = np.where(np.abs(dt - med) > tolerance_s)[0]
    if idx_outlier.size:
        print(f"[Δt] {len(idx_outlier)} deltas deviate > {tolerance_s:.2f}s from median")

    if idx_long.size:
        print(f"[gaps] Δt > {gap_threshold_s:.1f}s at indices {idx_long.tolist()}")

    # Plot
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(dt, marker='.', lw=0.8)
    ax1.set_title(f"{title_prefix} Δt between rising edges")
    ax1.set_xlabel("Rise index (i → i+1)")
    ax1.set_ylabel("Δt (s)")
    ax1.grid(alpha=0.3)
    ax1.axhline(med, ls='--', alpha=0.6, label=f"median {med:.3f}s")
    ax1.axhline(gap_threshold_s, color='r', ls='--', alpha=0.7, label=f"gap {gap_threshold_s:.1f}s")
    ax1.legend(loc="best", fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(dt, bins=max(12, int(np.sqrt(n))))
    ax2.set_title("Δt histogram")
    ax2.set_xlabel("Δt (s)")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================
# Main
# ======================
def main():
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load DIN stream
    pd_time_samp, pd_state = load_pd_samples(rec_folder, pd_channel)
    r_samp, f_samp = edges_from_transitions(pd_time_samp, pd_state)

    # Auto clean
    if auto_clean:
        r_samp, f_samp, clog = clean_dio_simple_samples(r_samp, f_samp, fs, min_high_s, min_low_s)
    else:
        clog = CleanLog(len(r_samp), [], [], [], [], len(r_samp))

    # Manual drops
    if manual_drop_rise_idxs or manual_drop_fall_idxs:
        r_samp, f_samp, clog = apply_manual_drops(r_samp, f_samp, manual_drop_rise_idxs, manual_drop_fall_idxs, clog)

    # Ideal vs real falls
    if not use_real_falls:
        f_samp = synthesize_falls_from_duration(r_samp, fs, stimulus_duration_s)
        m = min(len(r_samp), len(f_samp))
        r_samp, f_samp = r_samp[:m], f_samp[:m]

    # Δt diagnostics (before segmentation)
    if make_dt_plots:
        plot_path = (save_dir / f"{out_stem}_dt_all.png") if save_plots else None
        plot_rise_dt_diagnostics(
            rising_edges=r_samp,
            fs_hz=fs,
            gap_threshold_s=gap_threshold_s,
            tolerance_s=dt_tolerance_s,
            title_prefix=rec_folder.stem,
            save_path=plot_path,
            show=not save_plots,
        )

    # Segmentation
    if segment_by_gaps:
        segs = split_by_long_gaps(r_samp, fs, gap_threshold_s)
        print(f"[DIO] Found {len(segs)} segment(s) with gap > {gap_threshold_s:.1f}s")
        for i, idx in enumerate(segs, 1):
            r_seg = r_samp[idx]
            f_seg = f_samp[idx]
            out_path = save_dir / f"{out_stem}_segment{i:02d}.npz"
            save_npz_samples(out_path, r_seg, f_seg, fs, clog, segment_index=i)
            print(f"  [Saved] {out_path.name} | highs={len(r_seg)}")

            if make_dt_plots:
                seg_plot = (save_dir / f"{out_stem}_dt_segment{i:02d}.png") if save_plots else None
                plot_rise_dt_diagnostics(
                    rising_edges=r_seg,
                    fs_hz=fs,
                    gap_threshold_s=gap_threshold_s,
                    tolerance_s=dt_tolerance_s,
                    title_prefix=f"{rec_folder.stem} seg{i:02d}",
                    save_path=seg_plot,
                    show=False if save_plots else True,
                )
    else:
        out_path = save_dir / f"{out_stem}.npz"
        save_npz_samples(out_path, r_samp, f_samp, fs, clog)
        print(f"[Saved] {out_path.name} | highs={len(r_samp)}")

    # Summary
    print(f"[clean] Raw highs: {clog.raw_highs}")
    print(f"[clean] Dropped short highs: {clog.dropped_short_high_idxs}")
    print(f"[clean] Removed LOW blips (pairs): {clog.removed_low_blips_pairs}")
    print(f"[clean] Manual drops (rise): {clog.manual_drop_rise_idxs} | (fall): {clog.manual_drop_fall_idxs}")
    print(f"[clean] Clean highs: {clog.clean_highs}")


if __name__ == "__main__":
    main()
