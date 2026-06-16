"""
Inspect the (class x context) distribution of the merged training pool used by
apply_merged_decoder_to_sleep_original.py.

For each bin size in `bin_sizes_ms`, this:
  1. Prepares task and passive bins exactly the way the original script does
     (no kinematics, common units, vstack(task, passive)).
  2. Reports per-(class x group) cell counts and the within-class task fraction
     P(group=task | class=c) — this is the confound the original's class-only
     undersampling does NOT correct, and which the decoder can exploit.
  3. Simulates one draw of `balance_by_undersampling` (class-only, same RNG
     seed as the original) and reports the same breakdown for the balanced
     pool, so you can see whether class-only balancing changes the within-class
     task/passive ratio. (It doesn't, on average — but the realised draw can
     deviate slightly.)
  4. Saves a stacked-bar PNG per bin size and a single CSV with all counts.
"""

import sys
import csv
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(code_dir / 'reactivation' / 'VStimOnDecoding'))
sys.path.insert(0, str(code_dir / 'DiscriminationTask' / 'grating'))

import numpy as np
import matplotlib.pyplot as plt

from decode_utils import balance_by_undersampling
from prepare_task_stimtype import prepare_task_stim_type
from prepare_passive_stimtype import prepare_passive_stim_type

from params import (
    task_pkl, passive_pkl,
    bin_sizes_ms, random_state,
    class_pos, class_neg, TASK_COL_MAP, PASSIVE_COL_MAP,
)

CLASS_LABELS = (-1, 0, 1)
CLASS_NAMES  = {-1: '-1 (stim-on neg)', 0: '0 (ITI)', 1: '+1 (stim-on pos)'}
GROUP_NAMES  = {0: 'task', 1: 'passive'}
GROUP_COLORS = {0: '#E69F00', 1: '#0072B2'}


def _prepare_merged_with_groups(bin_size_sec):
    """Same merge as the original script, but also keep a group label
    (0 = task, 1 = passive) per row."""
    X_t, y_t, _, units_t = prepare_task_stim_type(
        task_pkl, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bin_size_sec, balance_classes=False,
        random_state=random_state,
    )
    X_p, y_p, _, units_p = prepare_passive_stim_type(
        passive_pkl, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_sec, balance_classes=False,
        random_state=random_state,
    )
    common = sorted(set(units_t) & set(units_p))
    X_t = X_t[:, [units_t.index(u) for u in common]]
    X_p = X_p[:, [units_p.index(u) for u in common]]
    X = np.vstack([X_t, X_p])
    y = np.concatenate([y_t, y_p])
    groups = np.concatenate([
        np.zeros(X_t.shape[0], dtype=int),
        np.ones(X_p.shape[0], dtype=int),
    ])
    return X, y, groups, len(common)


def _cell_counts(y, groups):
    """Return {(class, group): count} for all combinations seen."""
    return {
        (int(c), int(g)): int(np.sum((y == c) & (groups == g)))
        for c in CLASS_LABELS for g in (0, 1)
    }


def _print_table(title, counts):
    print(f"\n{title}")
    print(f"  {'class':>20}  {'task':>8}  {'passive':>8}  {'total':>8}  {'P(task|class)':>14}")
    for c in CLASS_LABELS:
        n_t = counts.get((c, 0), 0)
        n_p = counts.get((c, 1), 0)
        tot = n_t + n_p
        frac = (n_t / tot) if tot else float('nan')
        print(f"  {CLASS_NAMES[c]:>20}  {n_t:>8d}  {n_p:>8d}  {tot:>8d}  {frac:>14.3f}")
    total_t = sum(counts.get((c, 0), 0) for c in CLASS_LABELS)
    total_p = sum(counts.get((c, 1), 0) for c in CLASS_LABELS)
    grand   = total_t + total_p
    print(f"  {'--- total ---':>20}  {total_t:>8d}  {total_p:>8d}  {grand:>8d}")


def _plot_bin(out_dir, bms, raw_counts, bal_counts):
    """Stacked bar: raw and post-undersample counts, per class, split task/passive."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax, counts, title in zip(
        axes,
        [raw_counts, bal_counts],
        [f'Raw merged pool ({bms} ms)',
         f'After balance_by_undersampling ({bms} ms)'],
    ):
        xs = np.arange(len(CLASS_LABELS))
        task_n = np.array([counts.get((c, 0), 0) for c in CLASS_LABELS])
        pass_n = np.array([counts.get((c, 1), 0) for c in CLASS_LABELS])
        ax.bar(xs, task_n, color=GROUP_COLORS[0], label='task')
        ax.bar(xs, pass_n, bottom=task_n, color=GROUP_COLORS[1], label='passive')
        for x, t, p in zip(xs, task_n, pass_n):
            tot = t + p
            if tot:
                ax.text(x, tot, f"{t/tot:.0%} task",
                        ha='center', va='bottom', fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([CLASS_NAMES[c] for c in CLASS_LABELS], fontsize=8)
        ax.set_ylabel('bins')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    p = out_dir / f"training_distribution_{bms}ms.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figure -> {p}")


def main():
    session = Path(task_pkl).parent.name
    out_dir = (Path(task_pkl).parent / 'reactivation'
               / f'training_distribution_{session}')
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for bms in bin_sizes_ms:
        print('\n' + '=' * 78)
        print(f'  bin {bms} ms')
        print('=' * 78)
        X, y, groups, n_units = _prepare_merged_with_groups(bms / 1000.0)
        print(f'  merged shape: X={X.shape}  common units={n_units}')

        raw = _cell_counts(y, groups)
        _print_table('Raw merged pool:', raw)

        # Simulate one balance_by_undersampling draw with the same seed the
        # original uses for the final refit (random_state). CV folds use the
        # same scheme inside StratifiedKFold's training half; the raw within-
        # class task/passive ratio is what matters for the bias either way.
        rng = np.random.default_rng(random_state)
        X_bal, y_bal = balance_by_undersampling(X, y, rng)
        # Recover the group of each balanced row by matching against the raw
        # index list (cheap since balance_by_undersampling returns a sorted
        # selection from the full pool).
        rng2 = np.random.default_rng(random_state)
        classes = np.unique(y)
        min_count = min(int(np.sum(y == c)) for c in classes)
        sel = np.sort(np.concatenate([
            rng2.choice(np.where(y == c)[0], size=min_count, replace=False)
            for c in classes
        ]))
        groups_bal = groups[sel]
        bal = _cell_counts(y_bal, groups_bal)
        _print_table(f'After balance_by_undersampling (per-class -> {min_count}):', bal)

        _plot_bin(out_dir, bms, raw, bal)

        for c in CLASS_LABELS:
            for g in (0, 1):
                csv_rows.append({
                    'bin_ms': bms,
                    'class': c,
                    'group': GROUP_NAMES[g],
                    'n_raw': raw.get((c, g), 0),
                    'n_balanced': bal.get((c, g), 0),
                })

    csv_path = out_dir / 'training_distribution.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bin_ms', 'class', 'group',
                                          'n_raw', 'n_balanced'])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV summary -> {csv_path}")
    print("\nInterpretation:")
    print("  P(task | class=c) far from 0.5 means the decoder can use task-vs-passive")
    print("  features to predict class -- a confound the original's class-only")
    print("  undersampling does NOT correct (the column 'P(task|class)' stays roughly")
    print("  the same after balancing).")


if __name__ == '__main__':
    main()
