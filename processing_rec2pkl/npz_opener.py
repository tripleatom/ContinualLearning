# ================================================================
# compare_dio_npz_samples_only.py
# Compare rising/falling edges between two DIO NPZ files in SAMPLES.
# Treats both *_samples and *_times arrays as SAMPLES (rounds to int64).
# ================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# USER CONFIG
# ----------------------------
npz_a = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\old\CnL39_drifting_grating_exp_20251031_085247_DIO_WORKING.npz")
npz_b = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39SG_20251031_085159_DIO_samples_segment01.npz")

tol_samples   = 0      # allowable difference in samples
zero_anchor   = True   # subtract each array's first value before compare
make_plots    = True
save_plots    = True
save_dir      = npz_b.parent / "comparison_outputs"
save_dir.mkdir(exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def load_edges_as_samples(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    # Accept either naming; ALWAYS interpret as samples, coerce to int64
    if "rising_samples" in d and "falling_samples" in d:
        r = d["rising_samples"]
        f = d["falling_samples"]
    elif "rising_times" in d and "falling_times" in d:
        r = d["rising_times"]
        f = d["falling_times"]
    else:
        raise KeyError(f"{npz_path.name}: missing rising_* / falling_* arrays.")

    # Coerce to int64 samples (round if float)
    r = np.round(r).astype(np.int64)
    f = np.round(f).astype(np.int64)

    seg = int(d.get("segment_index", -1))
    return r, f, seg

def summarize_diff(name, diff, tol):
    n = diff.size
    if n == 0:
        print(f"[{name}] No edges to compare.")
        return
    med = np.median(diff)
    sd  = np.std(diff)
    out = np.sum(np.abs(diff) > tol)
    print(f"[{name}] n={n}  median Δ={med:.1f} samp  std={sd:.1f}  "
          f"min={diff.min():.0f}  max={diff.max():.0f}  | {out} > tol ({tol})")
    if out:
        idxs = np.where(np.abs(diff) > tol)[0][:10]
        print(f"  > First mismatch indices: {idxs.tolist()}")

def plot_diffs(dr, df, save_prefix: Path | None = None):
    for label, d in [("rising", dr), ("falling", df)]:
        if d.size == 0:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(d, marker='.', lw=0.7)
        plt.axhline(np.median(d), ls='--', color='gray')
        plt.title(f"Δ {label} (B - A) [samples]")
        plt.xlabel("index")
        plt.ylabel("Δ samples")
        plt.grid(alpha=0.3)
        if save_prefix:
            out = save_prefix.with_name(save_prefix.stem + f"_{label}_diff.png")
            plt.savefig(out, dpi=150)
            print(f"[saved] {out}")
            plt.close()
        else:
            plt.show()

# ----------------------------
# Main
# ----------------------------
print(f"Comparing (samples only):\nA = {npz_a}\nB = {npz_b}")

rA, fA, segA = load_edges_as_samples(npz_a)
rB, fB, segB = load_edges_as_samples(npz_b)

if zero_anchor:
    if rA.size: rA = rA - rA[0]
    if fA.size: fA = fA - fA[0]
    if rB.size: rB = rB - rB[0]
    if fB.size: fB = fB - fB[0]

# Trim to common length
nR = min(len(rA), len(rB))
nF = min(len(fA), len(fB))
rA, rB, fA, fB = rA[:nR], rB[:nR], fA[:nF], fB[:nF]

# Differences in samples
dr = rB - rA
df = fB - fA

print(f"\n[SAMPLES] Comparing {nR} rising and {nF} falling edges (segA={segA}, segB={segB})")
summarize_diff("rising", dr, tol_samples)
summarize_diff("falling", df, tol_samples)

# Quick offset readout
if dr.size:
    print(f"[rising offset] median={np.median(dr):.1f} samp, std={np.std(dr):.1f}")
if df.size:
    print(f"[falling offset] median={np.median(df):.1f} samp, std={np.std(df):.1f}")

# Plots
if make_plots:
    save_prefix = (save_dir / f"{npz_b.stem}_vs_{npz_a.stem}") if save_plots else None
    plot_diffs(dr, df, save_prefix)
