import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pickle

from decode_utils import bin_spikes, balance_by_undersampling
from prepare_task_stimtype import _match_all


def prepare_passive_stim_type(
    pkl_file,
    class_pos,
    class_neg,
    col_map,
    bin_size_sec=0.05,
    tol=1e-6,
    balance_classes=False,
    random_state=42,
):
    """
    Bin passive-session spike data and label each bin by left-side grating identity.

    Same joint-label semantics as prepare_task_stim_type but operates on a
    passive_spikes_*.pkl whose trial_params use passive-side column names
    (e.g. 'L_Orient', 'L_SF') — pass PASSIVE_COL_MAP from params.py.

    Labels
    ------
    +1   bin centre in a stim epoch whose trial matches every key of class_pos
    -1   bin centre in a stim epoch whose trial matches every key of class_neg
     0   bin centre outside all stim epochs (ITI)

    Bins from trials whose stimulus matches neither class are excluded.

    Returns
    -------
    X            : (n_bins_kept, n_units)  float32  firing rate in spikes/s
    y            : (n_bins_kept,)          int8     +1 / -1 / 0
    bin_centers  : (n_bins_kept,)          seconds, t=0 = window start
    unit_labels  : list of str             unit identifiers (columns of X)
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    trial_onsets  = np.asarray(data['window']['trial_onsets_sec'])
    trial_offsets = np.asarray(data['window']['trial_offsets_sec'])
    trial_params  = data['trial_params']
    window_dur    = data['window']['window_duration_sec']
    spike_data    = data['spike_data']

    if len(trial_params) == 0:
        raise ValueError("trial_params is empty — cannot label by stimulus type.")

    is_pos = _match_all(trial_params, class_pos, col_map, tol)
    is_neg = _match_all(trial_params, class_neg, col_map, tol)
    if (is_pos & is_neg).any():
        raise ValueError(
            "class_pos and class_neg both match the same trial — class definitions overlap."
        )

    bin_edges   = np.arange(0.0, window_dur + bin_size_sec, bin_size_sec)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins      = len(bin_centers)

    X_full, unit_labels = bin_spikes(spike_data, bin_edges, bin_size_sec)

    y_full = np.zeros(n_bins, dtype=np.int8)
    keep   = np.ones(n_bins, dtype=bool)

    for i, t in enumerate(bin_centers):
        in_epoch = (t >= trial_onsets) & (t < trial_offsets)
        if in_epoch.any():
            trial_idx = int(np.argmax(in_epoch))
            if is_pos[trial_idx]:
                y_full[i] = np.int8(1)
            elif is_neg[trial_idx]:
                y_full[i] = np.int8(-1)
            else:
                keep[i] = False

    X           = X_full[keep]
    y           = y_full[keep]
    bin_centers = bin_centers[keep]

    print(f"[passive stim type]  class_pos={class_pos}  class_neg={class_neg}")
    print(f"  +1={np.sum(y == 1)}  -1={np.sum(y == -1)}  "
          f"0 (ITI)={np.sum(y == 0)}  excluded={np.sum(~keep)}  kept={len(y)}")

    if balance_classes:
        rng = np.random.default_rng(random_state)
        X, y, bin_centers = balance_by_undersampling(X, y, rng, bin_centers=bin_centers)
        print(f"  balanced → {min(np.sum(y==1), np.sum(y==-1), np.sum(y==0))} bins per class  "
              f"(+1={np.sum(y==1)}  -1={np.sum(y==-1)}  0={np.sum(y==0)})")

    return X, y, bin_centers, unit_labels


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from decode_utils import run_cv

    from params import (
        passive_pkl as pkl_file,
        prep_default_bin_ms as bin_size_ms,
        n_splits,
        class_pos, class_neg, PASSIVE_COL_MAP,
        random_state,
    )
    balance_classes = True
    output          = None   # set to a path string to save results

    X, y, bin_centers, unit_labels = prepare_passive_stim_type(
        pkl_file, class_pos, class_neg, PASSIVE_COL_MAP,
        bin_size_sec=bin_size_ms / 1000.0,
        balance_classes=balance_classes,
        random_state=random_state,
    )
    print(f"X shape: {X.shape}  units: {len(unit_labels)}")

    fold_accs, mean_acc, chance, per_class_means, per_class_stds = run_cv(
        'AODE', None, X, y, n_splits=n_splits, random_state=random_state,
    )
    print(f"\n[{n_splits}-fold CV — stim type — AODE]")
    print(f"  Per-fold acc : {[f'{a:.3f}' for a in fold_accs]}")
    print(f"  Mean acc     : {mean_acc:.3f}")
    print(f"  Chance level : {chance:.3f}")
    print(f"  Per-class    : {per_class_means}")

    if output is not None:
        out = {
            'X': X, 'y': y, 'bin_centers': bin_centers,
            'unit_labels': unit_labels, 'bin_size_sec': bin_size_ms / 1000.0,
            'class_pos': class_pos, 'class_neg': class_neg, 'col_map': PASSIVE_COL_MAP,
            'fold_accs': fold_accs, 'mean_acc': mean_acc, 'chance': chance,
            'per_class_means': per_class_means, 'per_class_stds': per_class_stds,
        }
        with open(output, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved → {output}")
