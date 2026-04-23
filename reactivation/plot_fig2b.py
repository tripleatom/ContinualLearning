import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from params_fig2 import PKL_PATH, OUTPUT_DIR, RESP_WINDOW


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def check_spike_time_units(data):
    for unit_trials in data['spike_data'].values():
        for trial in unit_trials:
            st = np.asarray(trial['spike_times'], dtype=float)
            if len(st) > 0:
                if not np.all(st % 1 == 0):
                    return 'seconds'
    return 'frames'


def get_spikes_sec(trial, fs, time_unit):
    st = np.asarray(trial['spike_times'], dtype=float)
    return st / fs if time_unit == 'frames' else st


def build_conditions(data):
    sf_vals = sorted(set(
        round(tp['left_spatial_freq'], 4)
        for tp in data['trial_info']['all_trial_parameters']
    ))
    sf_to_cond = {sf: i for i, sf in enumerate(sf_vals)}
    label_map = {
        tp['trial_index']: sf_to_cond[round(tp['left_spatial_freq'], 4)]
        for tp in data['trial_info']['all_trial_parameters']
    }
    condition_names = [f'Left SF={sf:.3f} cpd' for sf in sf_vals]
    return label_map, condition_names


if __name__ == '__main__':
    pkl_path = Path(PKL_PATH)
    out_dir  = Path(OUTPUT_DIR) if OUTPUT_DIR else pkl_path.parent / 'figures'
    out_dir.mkdir(exist_ok=True)

    data      = load_pkl(pkl_path)
    fs        = data['metadata']['sampling_frequency']
    time_unit = check_spike_time_units(data)
    print(f"Spike time unit: {time_unit}  (fs={fs} Hz)")

    condition_labels, condition_names = build_conditions(data)
    n_cond = len(condition_names)
    cond_trials = {c: sorted([i for i, v in condition_labels.items() if v == c])
                   for c in range(n_cond)}

    unit_ids = list(data['spike_data'].keys())
    n_units  = len(unit_ids)

    # Load sort order from fig2a if available, otherwise use original order
    sort_order_file = out_dir / 'sort_order.npy'
    if sort_order_file.exists():
        sort_order = np.load(sort_order_file)
        print(f"Loaded sort_order from {sort_order_file}")
    else:
        sort_order = np.arange(n_units)
        print("sort_order.npy not found — using original unit order (run plot_fig2a.py first)")

    unit_ids_sorted = [unit_ids[i] for i in sort_order]

    pref_file = out_dir / 'pref_sorted.npy'
    pref_sorted = np.load(pref_file) if pref_file.exists() else np.zeros(n_units, dtype=int)

    dur = RESP_WINDOW[1] - RESP_WINDOW[0]
    base_dur = -RESP_WINDOW[0] if RESP_WINDOW[0] < 0 else 0.5

    print(f"Building single-trial matrices for {n_units} units ...")
    mats = []
    all_rates = np.zeros((n_units, 0))
    for c in range(n_cond):
        trials = cond_trials[c]
        mat = np.zeros((n_units, len(trials)))
        for i, uid in enumerate(unit_ids_sorted):
            trial_map = {t['trial_index']: t for t in data['spike_data'][uid]}
            for j, tidx in enumerate(trials):
                if tidx not in trial_map:
                    continue
                st   = get_spikes_sec(trial_map[tidx], fs, time_unit)
                mask = (st >= RESP_WINDOW[0]) & (st < RESP_WINDOW[1])
                mat[i, j] = mask.sum() / dur
        mats.append(mat)
        all_rates = np.concatenate([all_rates, mat], axis=1)

    # Per-neuron z-score using distribution across all trials
    mu = all_rates.mean(axis=1, keepdims=True)
    sigma = all_rates.std(axis=1, keepdims=True)
    sigma[sigma < 1e-10] = 1.0
    mats = [(m - mu) / sigma for m in mats]

    vmax      = np.percentile(np.abs(np.concatenate([m.ravel() for m in mats])), 97)
    n_max     = max(len(cond_trials[c]) for c in range(n_cond))
    fig_w     = max(4, n_max * 0.12) * n_cond
    fig_h     = max(5, n_units * 0.035)

    fig, axes = plt.subplots(1, n_cond, figsize=(fig_w, fig_h), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        im = ax.imshow(mats[c], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel(f'Trial  ({condition_names[c]})')
        ax.set_title(condition_names[c])
        if c == 0:
            ax.set_ylabel('Neurons (sorted by preferred condition)')
        for g in range(n_cond - 1):
            ax.axhline(int(np.sum(pref_sorted <= g)) - 0.5, color='k', lw=0.8)

    plt.colorbar(im, ax=axes[-1], label='Z-scored FR', shrink=0.6)
    plt.suptitle(f'Fig 2b — Single-trial responses  [{RESP_WINDOW[0]}–{RESP_WINDOW[1]} s]',
                 fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = out_dir / 'fig2b_single_trial.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
