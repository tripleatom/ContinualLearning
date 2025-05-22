import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import h5py
import os
from pathlib import Path
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.colors as mcolors

from rf_recon.rf_func import dereference, hex_offsets
from rec2nwb.preproc_func import parse_session_info
from spikeinterface.extractors import PhySortingExtractor

# ————— Inputs —————
rec_folder     = Path(input("Enter full path to recording folder: ").strip())
stimdata_file  = Path(input("Enter full path to .mat file: ").strip().strip('"'))

# load digital‐input rising edges
peaks_data     = scipy.io.loadmat(rec_folder / "peaks.mat", struct_as_record=False, squeeze_me=True)
rising_edges   = peaks_data['locs']

# load stimulus timing info
with h5py.File(rec_folder / "DIN.mat", 'r') as f:
    freq_params = f["frequency_parameters"]
    data        = freq_params['board_dig_in_sample_rate'][:]
    digInFreq   = data[0][0] if isinstance(data, np.ndarray) else data

# parse IDs
animal_id, session_id, _ = parse_session_info(rec_folder)

# load drifting‐grating parameters
with h5py.File(stimdata_file, 'r') as f:
    pp = f['Stimdata']['movieParams']
    orientation_data = pp['orientation'][()]
    stim_orientation = np.array([dereference(r, f) for r in orientation_data]).flatten().astype(float)
    tf_data          = pp['temporalFreq'][()]
    stim_tempFreq    = np.array([dereference(r, f) for r in tf_data]).flatten().astype(float)

n_stim = stim_orientation.size
drift_edges = rising_edges[-n_stim:]
print(f"Drifting‐grating stimuli: {n_stim}, rising edges: {len(drift_edges)}")

# custom colormap
pink_reds = LinearSegmentedColormap.from_list(
    'PinkReds',
    [(1,0.9,0.9), (1,0.6,0.6), (0.8,0,0), (0.6,0,0)]
)

# container for all metrics
all_shank_info = {}
shanks = ['0','1','2','3']

for ish in shanks:
    print(f"\nShank {ish}:")
    # find sorting folders
    code_folder = Path(__file__).parent.parent.parent
    shank_folder = code_folder / f"sortout/{animal_id}/{session_id}/{ish}"
    sorting_results_folders = []
    for root, dirs, files in os.walk(shank_folder):
        for dir_name in dirs:
            if dir_name.startswith('sorting_results_'):
                sorting_results_folders.append(os.path.join(root, dir_name))

    unit_info = {}

    for sorting_results_folder in sorting_results_folders:
        phy_folder    = Path(sorting_results_folder) / 'phy'
        out_fig_folder = Path(sorting_results_folder) / 'drifting_grating'
        out_fig_folder.mkdir(parents=True, exist_ok=True)

        sorting = PhySortingExtractor(phy_folder)
        fs      = sorting.sampling_frequency
        unit_ids = sorting.unit_ids
        qualities = sorting.get_property('quality')

        for idx, unit_id in enumerate(unit_ids):
            spike_train = sorting.get_unit_spike_train(unit_id)

            # blank baseline
            blank_win = 10.0  # seconds
            b_start = drift_edges[0] - int(blank_win * fs)
            b_end   = drift_edges[0]
            blank_spikes = np.sum((spike_train>=b_start)&(spike_train<b_end))
            blank_mean   = blank_spikes / blank_win

            # compute responses
            responses = np.zeros(n_stim)
            delay_pts = int(0.05 * fs)
            win_pts   = int(0.2  * fs)
            for j, edge in enumerate(drift_edges):
                start = edge + delay_pts
                stop  = start + win_pts
                responses[j] = np.sum((spike_train>=start)&(spike_train<stop)) / 0.2

            # unique conditions
            un_ori = np.unique(stim_orientation)
            un_tf  = np.unique(stim_tempFreq)
            n_ori, n_tf = un_ori.size, un_tf.size
            repeats = n_stim // (n_ori * n_tf)

            # reshape to ori × tf × repeats
            resp = responses.reshape(n_ori, n_tf, repeats, order='C')

            # — Metrics —
            # OSI
            mean_ot = resp.mean(axis=2)
            i_ori_sel, i_tf_sel = np.unravel_index(mean_ot.argmax(), mean_ot.shape)
            R_pref = mean_ot[i_ori_sel, i_tf_sel]
            i_orth = (i_ori_sel + n_ori//2) % n_ori
            R_orth = mean_ot[i_orth, i_tf_sel]
            OSI    = (R_pref - R_orth) / (R_pref + R_orth)

            # gOSI
            R_theta = mean_ot.mean(axis=1)
            theta   = np.deg2rad(un_ori)
            gOSI    = np.abs((R_theta * np.exp(1j*2*theta)).sum() / R_theta.sum())

            # TF Discrimination Index
            mean_by_ori = resp.mean(axis=(1,2))
            ori_pref    = mean_by_ori.argmax()
            tf_tuning   = resp[ori_pref,:,:].mean(axis=1)
            R_max, R_min = tf_tuning.max(), tf_tuning.min()
            TF_DI = (R_max - R_min) / (R_max + R_min) if (R_max+R_min)>0 else 0.0

            print(f" Unit {unit_id}: OSI={OSI:.2f}, gOSI={gOSI:.2f}, TF-DI={TF_DI:.2f}")

            unit_info[str(unit_id)] = {
                'OSI': OSI, 'gOSI': gOSI, 'TF_DI': TF_DI,
                'pref_ori': float(un_ori[i_ori_sel]),
                'pref_tf' : float(un_tf[i_tf_sel])
            }

            # 4. Generate polar (fan) plot for drifting‐grating response visualization
            #    (mimics your static‐grating code, but for orientation × temporalFreq)

            overall_std = np.std(resp)                # resp shape: (n_ori, n_tf, n_repeats)
            vmin = blank_mean
            vmax = blank_mean + 3.0 * overall_std
            norm = Normalize(vmin=vmin, vmax=vmax)

            # radial positions for each temporal frequency
            even_r = np.linspace(1, 8, n_tf)  # Increased range for better spacing

            # build a grid of (theta, r) centers
            theta_grid, r_grid = np.meshgrid(
                np.deg2rad(un_ori),    # unique orientations in degrees → radians
                even_r,                # radial levels for each TF
                indexing='ij'
            )
            theta_flat = theta_grid.flatten()
            r_flat     = r_grid.flatten()

            fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(10,6), dpi=300)  # Increased figure size
            for idx_val in range(len(theta_flat)):
                theta_val = theta_flat[idx_val]
                r_val     = r_flat[idx_val]
                o_idx     = idx_val // n_tf
                tf_idx    = idx_val % n_tf

                # get all repeats for this (orientation, TF)
                data_here      = resp[o_idx, tf_idx, :].flatten()
                data_sorted    = np.sort(data_here)[::-1]        # largest first
                local_uv       = hex_offsets(len(data_sorted), radius=0.15)  # Increased radius for better spread
                
                # jitter each repeat around the hub
                for rep_idx, (du, dv) in enumerate(local_uv):
                    dtheta = du / r_val if r_val != 0 else du
                    new_r     = r_val + dv
                    new_theta = theta_val + dtheta
                    fr_value  = data_sorted[rep_idx]
                    
                    ax.scatter(
                        new_theta, new_r,
                        c=fr_value,
                        cmap=pink_reds,
                        s=15,  # Increased point size
                        norm=norm,
                        clip_on=False,
                        alpha=0.8,  # Slightly reduced alpha for better visibility
                        marker='h',
                        edgecolors='none'
                    )

            plt.colorbar(ax.collections[0], ax=ax, label='Firing Rate (Hz)')
            ax.set_thetamin(0)
            ax.set_thetamax(360)
            ax.set_thetagrids(un_ori, labels=[f"{int(o)}°" for o in un_ori])
            ax.set_rticks(even_r)
            # Modify y-axis labels to only show Hz for 15 Hz
            y_labels = []
            for tf in un_tf:
                if abs(tf - 15.0) < 0.01:  # Check if close to 15 Hz
                    y_labels.append(f"{tf:.2f} Hz")
                else:
                    y_labels.append(f"{tf:.2f}")
            ax.set_yticklabels(y_labels)
            ax.spines['polar'].set_visible(False)
            ax.set_frame_on(False)
            ax.set_title(f"Unit {unit_id}: {qualities[idx]}")

            plt.tight_layout()
            fig.text(
                0.5, 0.01,
                f"Pref ori={un_ori[i_ori_sel]:.1f}°, OSI={OSI:.2f}, "
                f"gOSI={gOSI:.2f}, Pref TF={un_tf[i_tf_sel]:.2f} Hz, TF-DI={TF_DI:.2f}",
                ha='center', va='bottom', fontsize=10
            )

            out_file = out_fig_folder / f'unit_{unit_id}_fan_plot.png'
            plt.savefig(out_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved fan plot for unit {unit_id} to {out_file}")


        # end unit loop

    all_shank_info[ish] = unit_info

# save all metrics
out_npz = code_folder / f"sortout/{animal_id}/{session_id}/drifting_grating_tuning_metrics.npz"
np.savez_compressed(out_npz, all_shank_info=all_shank_info)
print(f"\nSaved metrics to {out_npz}")