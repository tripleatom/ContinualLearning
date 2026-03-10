"""
Window Search for Optimal LDA Time Window

Sweeps over (start, end) combinations, runs LDA cross-validation for each,
and plots mean CV accuracy as a heatmap.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from behaviorLDA import load_neural_data, calculate_firing_rates, perform_lda_analysis


def sweep_windows(data, starts, ends):
    """
    Run LDA CV for every valid (start, end) combination.

    Parameters
    ----------
    data : dict
        Neural data dict from load_neural_data()
    starts : ndarray
        Window start times in seconds (relative to trial onset)
    ends : ndarray
        Window end times in seconds

    Returns
    -------
    cv_matrix : ndarray, shape (len(starts), len(ends))
        Mean CV accuracy for each (start, end) pair; NaN for invalid/skipped.
    """
    cv_matrix = np.full((len(starts), len(ends)), np.nan)
    total = len(starts) * len(ends)
    done = 0

    for i, start in enumerate(starts):
        for j, end in enumerate(ends):
            done += 1
            if done % 500 == 0:
                valid_so_far = np.sum(~np.isnan(cv_matrix))
                print(f"  [{done}/{total}] valid windows so far: {valid_so_far}")

            if end - start <= 0.05:
                continue

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    firing_rates, condition_labels, unit_ids, _ = calculate_firing_rates(
                        data, time_window=(float(start), float(end))
                    )

                if len(condition_labels) == 0:
                    continue

                with contextlib.redirect_stdout(io.StringIO()):
                    results = perform_lda_analysis(firing_rates, condition_labels)

                cv_matrix[i, j] = results['cv_scores'].mean()

            except Exception:
                pass

    return cv_matrix


def plot_heatmap(cv_matrix, starts, ends, save_path=None):
    """
    Plot CV accuracy as a heatmap with start on y-axis and end on x-axis.

    Parameters
    ----------
    cv_matrix : ndarray, shape (len(starts), len(ends))
    starts : ndarray
    ends : ndarray
    save_path : str or Path, optional
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Mask NaN cells
    masked = np.ma.masked_invalid(cv_matrix)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(
        masked,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=0.5,
        vmax=1.0,
        extent=[ends[0], ends[-1], starts[0], starts[-1]],
        interpolation='nearest',
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean CV Accuracy', fontsize=12)
    cbar.ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Chance')

    # Mark best window
    if not np.all(np.isnan(cv_matrix)):
        best_flat = np.nanargmax(cv_matrix)
        bi, bj = np.unravel_index(best_flat, cv_matrix.shape)
        best_start, best_end = starts[bi], ends[bj]
        best_cv = cv_matrix[bi, bj]
        ax.plot(best_end, best_start, 'r*', markersize=18,
                label=f'Best: [{best_start:.2f}, {best_end:.2f}] s  CV={best_cv:.3f}')
        ax.legend(fontsize=11, loc='upper right')

    # Ticks every 0.1 s
    x_ticks = np.arange(np.ceil(ends[0] * 10) / 10, ends[-1] + 0.01, 0.1)
    y_ticks = np.arange(np.ceil(starts[0] * 10) / 10, starts[-1] + 0.01, 0.1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{t:.1f}' for t in x_ticks], rotation=45, fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{t:.1f}' for t in y_ticks], fontsize=8)

    ax.set_xlabel('Window End (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Window Start (s)', fontsize=13, fontweight='bold')
    ax.set_title('LDA CV Accuracy — Window Search', fontsize=15, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")

    return fig


if __name__ == '__main__':
    DATA_PATH = r"/Volumes/xieluanlabs/xl_cl/sortout/CnL42SG/CnL42SG_20260304/behavior_trial_embedding_20260309_2000.pkl"

    starts = np.arange(0.00, 1.01, 0.01)   # 0.00 … 1.00 s
    ends   = np.arange(0.10, 2.01, 0.01)   # 0.10 … 2.00 s

    print(f"Loading data from: {DATA_PATH}")
    data = load_neural_data(DATA_PATH)

    n_valid = sum(1 for s in starts for e in ends if e - s > 0.05)
    print(f"Grid: {len(starts)} starts × {len(ends)} ends = {n_valid} valid windows")

    print("\nRunning window sweep...")
    cv_matrix = sweep_windows(data, starts, ends)

    # Report best window
    if not np.all(np.isnan(cv_matrix)):
        best_flat = np.nanargmax(cv_matrix)
        bi, bj = np.unravel_index(best_flat, cv_matrix.shape)
        print(f"\nBest window: start={starts[bi]:.2f}s  end={ends[bj]:.2f}s  "
              f"duration={ends[bj]-starts[bi]:.2f}s  CV={cv_matrix[bi, bj]:.3f}")
    else:
        print("\nNo valid results found.")

    # Save heatmap next to PKL
    out_path = Path(DATA_PATH).with_suffix('.window_search.png')
    plot_heatmap(cv_matrix, starts, ends, save_path=out_path)
    plt.show()
