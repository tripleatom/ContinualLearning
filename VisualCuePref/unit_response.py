"""
Simple per-unit response heatmap (no significance test, no top-N filter).
One column per cue, rows = all units, time window -2 to 5 s rel trial start.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from params import task_spikes_file
from cue_preference import load_data, get_cue_labels, cue_is_rewarded, plot_heatmap


def get_save_dir():
    """VisualResponse/ subfolder next to the sortout session pkl."""
    out = os.path.join(os.path.dirname(task_spikes_file), 'VisualResponse')
    os.makedirs(out, exist_ok=True)
    return out


def build_psth_all(data, t_start=-1.0, t_stop=1.0, bin_w=0.05, sigma_s=0.075):
    """psth[cell, trial, time_bin] in spikes/s, Gaussian-smoothed along time."""
    edges = np.arange(t_start, t_stop + bin_w / 2, bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    unit_ids = list(data['spike_data'].keys())
    n_cells = len(unit_ids)
    n_trials = data['metadata']['n_trials']

    psth = np.zeros((n_cells, n_trials, len(edges) - 1))
    for c, uid in enumerate(unit_ids):
        for trial in data['spike_data'][uid]:
            t_idx = trial['trial_index']
            spikes = np.asarray(trial['spike_times'], dtype=float)
            spikes = spikes[(spikes >= t_start) & (spikes < t_stop)]
            counts, _ = np.histogram(spikes, bins=edges)
            psth[c, t_idx, :] = counts / bin_w
    psth = gaussian_filter1d(psth, sigma=sigma_s / bin_w, axis=-1)
    return psth, centers, unit_ids


def sort_by_peak(psth, cues, unique_cues, centers, peak_window=(0.0, 1.0)):
    """Sort cells by time of peak mean response (across cues), within peak_window."""
    n_cells, _, n_time = psth.shape
    mean_psth = np.zeros((n_cells, len(unique_cues), n_time))
    for ci, cue in enumerate(unique_cues):
        m = cues == cue
        if m.any():
            mean_psth[:, ci, :] = psth[:, m, :].mean(axis=1)
    pooled = mean_psth.mean(axis=1)  # [cell, time]
    win = (centers >= peak_window[0]) & (centers < peak_window[1])
    if win.any():
        peak_t = np.argmax(pooled[:, win], axis=1)
    else:
        peak_t = np.argmax(pooled, axis=1)
    return np.argsort(peak_t)


def main(cue_mode='left_stim', zscore='baseline'):
    data = load_data(task_spikes_file)
    cues, cue_names = get_cue_labels(data, mode=cue_mode)
    print(f"Loaded {data['metadata']['n_trials']} trials, "
          f"{len(data['spike_data'])} units")

    psth, centers, _ = build_psth_all(
        data, t_start=-1.0, t_stop=1.0, bin_w=0.05, sigma_s=0.075,
    )

    unique_cues = np.unique(cues)
    order = sort_by_peak(psth, cues, unique_cues, centers, peak_window=(0.0, 1.0))
    psth = psth[order]

    is_rew_full = cue_is_rewarded(data, cue_mode, len(cue_names), cues)
    is_rew_plot = (None if is_rew_full is None
                   else [is_rew_full[c] for c in unique_cues])

    fig = plot_heatmap(
        psth, cues, unique_cues, centers,
        cue_on=0.0, cue_off=None,
        cue_names=[cue_names[c] for c in unique_cues],
        cue_is_rewarded=is_rew_plot,
        row_pref=None,
        row_sign=None,
        zscore=zscore,
        baseline_t=(-1.0, 0.0),
        save_path=os.path.join(get_save_dir(), 'unit_response_psth.png'),
    )
    plt.show()
    return order, psth, fig


if __name__ == '__main__':
    main()
