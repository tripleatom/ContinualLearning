import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_dir))

import numpy as np
import pickle


def prepare_classification_data(pkl_file, bin_size_sec=0.05):
    """
    Bin spike data from a session_spikes pickle (extract_session_spikes.py) into
    firing rates and produce per-bin labels for classification.

    Labels
    ------
    +1  bin centre falls inside a stimulus epoch AND left stim is the rewarded stim
    -1  bin centre falls inside a stimulus epoch AND left stim is the non-rewarded stim
     0  outside any stimulus epoch

    Parameters
    ----------
    pkl_file     : str or Path  – path to session_spikes_*.pkl
    bin_size_sec : float        – bin width in seconds (default 0.05 = 50 ms)

    Returns
    -------
    X            : np.ndarray, shape (n_bins, n_units), float32
                   Firing rate in spikes/s for each unit × time bin.
    y            : np.ndarray, shape (n_bins,), int8
                   Per-bin label (+1 / -1 / 0).
    bin_centers  : np.ndarray, shape (n_bins,)
                   Time (s) of each bin centre, t=0 is window start.
    unit_labels  : list of str
                   Unit identifiers corresponding to columns of X.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    window_dur    = data['window']['window_duration_sec']
    trial_onsets  = np.asarray(data['window']['trial_onsets_sec'])
    trial_offsets = np.asarray(data['window']['trial_offsets_sec'])
    trial_params  = data['trial_params']
    spike_data    = data['spike_data']

    rewarded_on_left = np.array([tp['rewardedOnLeft'] for tp in trial_params], dtype=bool)

    # bin edges and centres
    bin_edges   = np.arange(0.0, window_dur + bin_size_sec, bin_size_sec)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins      = len(bin_centers)

    # label each bin
    y = np.zeros(n_bins, dtype=np.int8)
    for i, t in enumerate(bin_centers):
        in_epoch = (t >= trial_onsets) & (t < trial_offsets)
        if in_epoch.any():
            trial_idx = np.argmax(in_epoch)
            y[i] = np.int8(1) if rewarded_on_left[trial_idx] else np.int8(-1)

    # bin spike trains → firing rates
    unit_labels = sorted(spike_data.keys())
    X = np.zeros((n_bins, len(unit_labels)), dtype=np.float32)

    for col, uid in enumerate(unit_labels):
        spikes         = np.asarray(spike_data[uid]['spike_times_sec'])
        counts, _      = np.histogram(spikes, bins=bin_edges)
        X[:, col]      = counts / bin_size_sec   # spikes/s

    return X, y, bin_centers, unit_labels


if __name__ == '__main__':
    # =========================================================
    # Edit these paths / parameters before running
    # =========================================================
    pkl_file     = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313\task_spikes_20260313_1800.pkl"
    bin_size_ms  = 50.0     # bin width in milliseconds
    output       = None     # set to a path string to save the prepared data, e.g.:
                            # r"\\...\\clf_data.pkl"
    # =========================================================

    bin_size_sec = bin_size_ms / 1000.0
    X, y, bin_centers, unit_labels = prepare_classification_data(
        pkl_file, bin_size_sec=bin_size_sec
    )

    print(f"X shape:      {X.shape}  (n_bins x n_units)")
    print(f"y shape:      {y.shape}")
    print(f"Bin size:     {bin_size_ms} ms")
    print(f"Label counts: +1={np.sum(y == 1)}, -1={np.sum(y == -1)}, 0={np.sum(y == 0)}")

    pkl_path = Path(pkl_file)
    if output is None:
        output = pkl_path.parent / f'clf_data_{pkl_path.stem}.pkl'

    out = {
        'X':            X,
        'y':            y,
        'bin_centers':  bin_centers,
        'unit_labels':  unit_labels,
        'bin_size_sec': bin_size_sec,
    }
    with open(output, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {output}")
