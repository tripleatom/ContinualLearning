"""
SVM orientation-decoding accuracy across passive grating days.

For every recorded day, this loads that day's merged grating pickle, runs the
exact same SVM pipeline used by GratingSVM (balanced classes, scaled-in-fold
cross-validation, shuffled-label baseline), and records the cross-validated
multi-class orientation-decoding accuracy. It then plots accuracy vs. time:

    x-axis : session date (earliest -> latest)
    y-axis : decoding accuracy (mean +/- std across CV folds)
             + shuffled-label baseline + theoretical chance level

Each day uses ALL units present in that day's merged pickle, so the decoded
population (and unit count) varies day to day; the per-day n_units is annotated.

Data source per day (see GratingExport.py):
    <base_dir>/<session>/passive_embedding_analysis/*_grating_data_merged.pkl

Example:
    python plot_svm_accuracy_over_days.py --exclude-session 260226 260313
    python plot_svm_accuracy_over_days.py --time-window 0.05 1.5 --kernel rbf
"""

from __future__ import annotations

import argparse
import re
import platform
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grating_utils import load_neural_data, calculate_firing_rates
from GratingSVM import perform_svm_analysis


# ── Defaults (mirror the by-day raster script so the same sessions are used) ──
if platform.system() == "Darwin":
    _SORTOUT_ROOT = Path(r"/Volumes/xieluanlabs/xl_cl/sortout")
else:
    _SORTOUT_ROOT = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout")

DEFAULT_ANIMAL = "CnL42SG"

# ── EDIT HERE to exclude sessions ────────────────────────────────────────────
# Add the date token (YYMMDD, e.g. "260226") of any session you want to drop
# from the plot — bad data quality, missing pickle, etc. One per line is fine.
# These are skipped unless overridden on the command line with --exclude-session.
EXCLUDE_SESSIONS = [
    "260226",
    "260313",
    "260305",
]

# Only plot sessions on or after this date (YYMMDD). Set to None for no cutoff.
# Overridable on the command line with --start-session.
START_SESSION = "260316"
# ─────────────────────────────────────────────────────────────────────────────


def find_day_pkls(base_dir: Path, excluded: set[str]) -> dict[str, Path]:
    """Newest *_grating_data_merged.pkl per session folder under base_dir.

    Keys are the full session folder name (e.g. CnL42SG_20260123). A session is
    skipped if its 6-digit (YYMMDD) or 8-digit (YYYYMMDD) date is in `excluded`.
    """
    pkls: dict[str, Path] = {}
    for analysis_dir in sorted(base_dir.glob("*/passive_embedding_analysis")):
        session = analysis_dir.parent.name
        digits = re.search(r"(\d{8}|\d{6})", session)
        if digits:
            d = digits.group(1)
            if d in excluded or (len(d) == 8 and d[2:] in excluded):
                continue
        candidates = sorted(analysis_dir.glob("*_grating_data_merged.pkl"))
        if not candidates:
            candidates = sorted(analysis_dir.glob("*_grating_data*.pkl"))
        if not candidates:
            continue
        pkls[session] = max(candidates, key=lambda p: p.stat().st_mtime)
    return pkls


def session_date(session: str) -> datetime:
    """Parse YYYYMMDD out of a session token like CnL42SG_20260123."""
    digits = re.search(r"(\d{8}|\d{6})", session).group(1)
    if len(digits) == 6:
        digits = "20" + digits
    return datetime.strptime(digits, "%Y%m%d")


def decode_one_day(pkl_path: Path, time_window, kernel, C, gamma):
    """Run the SVM orientation decoder on one day. Returns a result dict or None.

    Pools across spatial frequencies (all units, all SFs) so there is one
    overall multi-class orientation-decoding accuracy per day.
    """
    data = load_neural_data(pkl_path)
    firing_rates, ori_labels, unit_ids, _ = calculate_firing_rates(
        data, time_window=time_window
    )
    if len(ori_labels) == 0 or len(np.unique(ori_labels)) < 2:
        print(f"  Skipping {pkl_path.name}: <2 orientations or no valid trials.")
        return None

    res = perform_svm_analysis(firing_rates, ori_labels,
                               kernel=kernel, C=C, gamma=gamma)
    return {
        "n_units": len(unit_ids),
        "n_trials": len(ori_labels),
        "n_orientations": len(res["unique_orientations"]),
        "acc_mean": float(res["cv_scores"].mean()),
        "acc_std": float(res["cv_scores"].std()),
        "shuf_mean": float(res["cv_scores_shuffled"].mean()),
        "shuf_std": float(res["cv_scores_shuffled"].std()),
        "chance": float(res["chance_accuracy"]),
    }


def plot_accuracy_over_days(rows, out_path, *, time_window, kernel, C, gamma,
                            base_dir):
    """Plot decoding accuracy (mean +/- std) vs. date, with baselines."""
    dates = [r["date"] for r in rows]
    acc = np.array([r["acc_mean"] for r in rows])
    acc_sd = np.array([r["acc_std"] for r in rows])
    shuf = np.array([r["shuf_mean"] for r in rows])
    shuf_sd = np.array([r["shuf_std"] for r in rows])
    chance = float(np.mean([r["chance"] for r in rows]))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.errorbar(dates, acc, yerr=acc_sd, marker="o", markersize=9,
                linewidth=2.5, capsize=5, color="#2E86AB",
                label="CV accuracy", zorder=3)
    ax.errorbar(dates, shuf, yerr=shuf_sd, marker="s", markersize=7,
                linewidth=2.0, capsize=4, color="#999999", alpha=0.9,
                label="Shuffled labels", zorder=2)
    ax.axhline(chance, color="black", linestyle="--", linewidth=2.0,
               label=f"Chance ({chance:.3f})", zorder=1)

    ax.set_xlabel("Session date", fontsize=18, fontweight="bold")
    ax.set_ylabel("Orientation decoding accuracy", fontsize=18, fontweight="bold")
    ax.set_title("Orientation decoding across days", fontsize=20,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="both", labelsize=14, width=1.8, length=6)
    fig.autofmt_xdate(rotation=45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.8)
    ax.legend(fontsize=14, frameon=False, loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Reproducibility stamp (embedded in the figure) ──
    stamp = (
        f"plot_svm_accuracy_over_days.py | generated {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"base_dir={base_dir}\n"
        f"time_window={time_window}s  kernel={kernel}  C={C}  gamma={gamma}  "
        f"all units/day, classes balanced, StratifiedKFold CV\n"
        f"sessions: " + ", ".join(r["session"] for r in rows)
    )
    fig.text(0.005, 0.005, stamp, fontsize=6, color="#555555",
             ha="left", va="bottom", family="monospace")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path,
                        default=_SORTOUT_ROOT / DEFAULT_ANIMAL,
                        help="Sortout animal folder containing <session>/passive_embedding_analysis/")
    parser.add_argument("--exclude-session", nargs="*",
                        default=list(EXCLUDE_SESSIONS),
                        help="Session date tokens (YYMMDD) to skip. "
                             "Overrides the EXCLUDE_SESSIONS list at the top of the file.")
    parser.add_argument("--start-session", default=START_SESSION,
                        help="Only include sessions on/after this date (YYMMDD). "
                             "Use 'none' for no cutoff.")
    parser.add_argument("--time-window", nargs=2, type=float, default=(0.05, 1.5),
                        metavar=("START", "END"),
                        help="Post-stimulus window (s) for firing-rate features.")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output figure path (default: <base_dir>/svm_accuracy_over_days.png)")
    args = parser.parse_args()

    excluded = set(args.exclude_session)
    time_window = tuple(args.time_window)

    pkls = find_day_pkls(args.base_dir, excluded)
    if not pkls:
        raise SystemExit(f"No merged grating pickles found under {args.base_dir}")

    # Apply the start-date cutoff (sessions on/after --start-session).
    start = args.start_session
    if start and str(start).lower() != "none":
        cutoff = session_date(start)
        pkls = {s: p for s, p in pkls.items() if session_date(s) >= cutoff}
        print(f"Start-session cutoff: {cutoff:%Y-%m-%d} "
              f"({len(pkls)} session(s) remain).")
        if not pkls:
            raise SystemExit(f"No sessions on/after {cutoff:%Y-%m-%d}.")

    print(f"Found {len(pkls)} session(s) with merged grating pickles.")
    rows = []
    for session in sorted(pkls, key=session_date):
        print(f"\n{'='*60}\n{session}  ->  {pkls[session].name}\n{'='*60}")
        try:
            r = decode_one_day(pkls[session], time_window,
                               args.kernel, args.C, args.gamma)
        except Exception as e:  # one bad day shouldn't kill the whole sweep
            print(f"  ERROR on {session}: {e}")
            r = None
        if r is None:
            continue
        r["session"] = session
        r["date"] = session_date(session)
        rows.append(r)

    if not rows:
        raise SystemExit("No day produced a valid decoding result.")

    print(f"\n{'#'*60}\nSummary ({len(rows)} day(s)):")
    for r in rows:
        print(f"  {r['session']}: acc={r['acc_mean']:.3f}+/-{r['acc_std']:.3f}  "
              f"shuf={r['shuf_mean']:.3f}  chance={r['chance']:.3f}  "
              f"n_units={r['n_units']}  n_trials={r['n_trials']}")

    out = args.output or (args.base_dir / "svm_accuracy_over_days.png")
    plot_accuracy_over_days(rows, out, time_window=time_window,
                            kernel=args.kernel, C=args.C, gamma=args.gamma,
                            base_dir=args.base_dir)


if __name__ == "__main__":
    main()
