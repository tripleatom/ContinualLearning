import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from params_fig2 import PKL_PATH, OUTPUT_DIR, PSTH_WINDOW, BIN_SIZE


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


def compute_psth(unit_trials, trial_indices, fs, time_unit, window, bin_size):
    edges   = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    trial_map = {t['trial_index']: t for t in unit_trials}
    rates = []
    for idx in trial_indices:
        if idx not in trial_map:
            continue
        st = get_spikes_sec(trial_map[idx], fs, time_unit)
        counts, _ = np.histogram(st, bins=edges)
        rates.append(counts / bin_size)
    if not rates:
        return np.zeros(len(centers)), centers, np.zeros((0, len(centers)))
    trial_rates = np.array(rates)
    return trial_rates.mean(axis=0), centers, trial_rates


def zscore_psth(psth, centers, trial_rates, baseline=(-0.2, 0.0)):
    mask = (centers >= baseline[0]) & (centers < baseline[1])
    vals = trial_rates[:, mask].ravel() if trial_rates.shape[0] > 0 else psth[mask]
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-10:
        return np.zeros_like(psth)
    return (psth - mu) / sigma


if __name__ == '__main__':
    pkl_path = Path(PKL_PATH)
    out_dir  = Path(OUTPUT_DIR) if OUTPUT_DIR else pkl_path.parent / 'figures'
    out_dir.mkdir(exist_ok=True)

    data      = load_pkl(pkl_path)
    fs        = data['metadata']['sampling_frequency']
    time_unit = check_spike_time_units(data)
    print(f"Spike time unit: {time_unit}  (fs={fs} Hz)")

    condition_labels, condition_names = build_conditions(data)
    n_cond   = len(condition_names)
    cond_trials = {c: [i for i, v in condition_labels.items() if v == c]
                   for c in range(n_cond)}

    unit_ids = list(data['spike_data'].keys())
    n_units  = len(unit_ids)

    edges   = np.arange(PSTH_WINDOW[0], PSTH_WINDOW[1] + BIN_SIZE, BIN_SIZE)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins  = len(centers)
    resp_mask = (centers >= 0) & (centers < 0.5)

    Z         = np.zeros((n_units, n_cond, n_bins))
    preferred = np.zeros(n_units, dtype=int)

    print(f"Computing PSTHs for {n_units} units ...")
    for i, uid in enumerate(unit_ids):
        unit_trials = data['spike_data'][uid]
        for c in range(n_cond):
            psth, cbins, trial_rates = compute_psth(
                unit_trials, cond_trials[c], fs, time_unit, PSTH_WINDOW, BIN_SIZE)
            Z[i, c, :] = zscore_psth(psth, cbins, trial_rates)
        preferred[i] = int(np.argmax([Z[i, c, resp_mask].mean() for c in range(n_cond)]))

    for i in range(n_units):
        for c in range(n_cond):
            Z[i, c, :] = gaussian_filter1d(Z[i, c, :], sigma=2)

    sort_order  = np.argsort(preferred, kind='stable')
    Z_sorted    = Z[sort_order]
    pref_sorted = preferred[sort_order]

    vmax  = np.percentile(np.abs(Z_sorted), 97)
    fig_h = max(5, n_units * 0.035)
    fig, axes = plt.subplots(1, n_cond, figsize=(3.5 * n_cond, fig_h), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        im = ax.imshow(Z_sorted[:, c, :], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[PSTH_WINDOW[0], PSTH_WINDOW[1], n_units, 0])
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('Time from stim onset (s)')
        ax.set_title(condition_names[c])
        if c == 0:
            ax.set_ylabel('Neurons (sorted by preferred condition)')
        for g in range(n_cond - 1):
            ax.axhline(int(np.sum(pref_sorted <= g)), color='k', lw=0.8)

    plt.colorbar(im, ax=axes[-1], label='Z-scored FR', shrink=0.6)
    plt.suptitle('Fig 2a — Mean cue response heatmap', fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = out_dir / 'fig2a_mean_response.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")

    # Save sort_order and pref_sorted for use by fig2b / fig2d
    np.save(out_dir / 'sort_order.npy', sort_order)
    np.save(out_dir / 'pref_sorted.npy', pref_sorted)
    print(f"Saved sort_order.npy and pref_sorted.npy → {out_dir}")
