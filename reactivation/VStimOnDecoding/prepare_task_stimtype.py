import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pickle

from decode_utils import bin_spikes, balance_by_undersampling


def _match_all(trial_params, class_dict, col_map, tol):
    """Boolean mask: which trials match every (canonical_key -> value) in class_dict.

    class_dict keys are canonical (e.g. 'orientation', 'spatial_freq');
    col_map translates them to the pkl's trial_params column names.
    """
    mask = np.ones(len(trial_params), dtype=bool)
    for canon_key, val in class_dict.items():
        col = col_map[canon_key]
        vals = np.array([tp[col] for tp in trial_params], dtype=float)
        mask &= np.abs(vals - float(val)) <= tol
    return mask


def infer_rewarded_combination(trial_params):
    """Return canonical {'orientation': float, 'spatial_freq': float} of the rewarded
    grating in this task session.

    The rewarded side varies trial-to-trial (rewardedOnLeft is True or False) but the
    *grating identity* on the rewarded side should be constant — we read it from the
    rewarded side of every trial and assert that exactly one (ori, SF) pair appears.

    Raises ValueError if the mapping is inconsistent across trials.
    """
    combos = set()
    for tp in trial_params:
        if tp['rewardedOnLeft']:
            ori, sf = tp['leftOrientation'], tp['leftSpatialFreq']
        else:
            ori, sf = tp['rightOrientation'], tp['rightSpatialFreq']
        combos.add((float(ori), float(sf)))
    if len(combos) != 1:
        raise ValueError(
            f"Inconsistent reward mapping across trials: {sorted(combos)}. "
            "Expected exactly one (orientation, spatial_freq) combination to be the "
            "rewarded stimulus regardless of which side it appeared on."
        )
    ori, sf = combos.pop()
    return {"orientation": ori, "spatial_freq": sf}


def prepare_task_stim_type(
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
    Bin task-session spike data and label each bin by left-side grating identity.

    Labels
    ------
    +1   bin centre falls inside a stimulus epoch whose trial matches every
         (canonical_key → value) in class_pos
    -1   bin centre falls inside a stimulus epoch whose trial matches every
         (canonical_key → value) in class_neg
     0   bin centre is outside all stimulus epochs (ITI)

    Bins from trials whose left grating matches neither class_pos nor class_neg
    are excluded from the output.

    Parameters
    ----------
    pkl_file        : str or Path  – task_spikes_*.pkl
    class_pos       : dict         – canonical-keyed dict mapped to label +1
                                     e.g. {"orientation": 0.0, "spatial_freq": 0.04}
    class_neg       : dict         – canonical-keyed dict mapped to label -1
    col_map         : dict         – canonical_key → trial_params column name,
                                     e.g. {"orientation": "leftOrientation",
                                           "spatial_freq": "leftSpatialFreq"}
    bin_size_sec    : float        – bin width in seconds (default 0.05 = 50 ms)
    tol             : float        – tolerance for float comparison (default 1e-6)
    balance_classes : bool         – if True, undersample each class to the size of
                                     the smallest class so all classes are equally
                                     represented (default False)
    random_state    : int          – random seed used when balance_classes=True

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

    print(f"[task stim type]  class_pos={class_pos}  class_neg={class_neg}")
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
    from params import (
        task_pkl as pkl_file,
        prep_default_bin_ms as bin_size_ms,
        class_pos, class_neg, TASK_COL_MAP,
        random_state,
    )
    balance_classes = True

    X, y, bin_centers, unit_labels = prepare_task_stim_type(
        pkl_file, class_pos, class_neg, TASK_COL_MAP,
        bin_size_sec=bin_size_ms / 1000.0,
        balance_classes=balance_classes,
        random_state=random_state,
    )
    print(f"X shape: {X.shape}  units: {len(unit_labels)}")

    with open(pkl_file, 'rb') as f:
        tp = pickle.load(f)['trial_params']
    rewarded = infer_rewarded_combination(tp)
    print(f"Rewarded grating in this session: {rewarded}")
