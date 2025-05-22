import spikeinterface.preprocessing as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import numpy as np
import pandas as pd
from pathlib import Path
from spikeinterface import extractors as se
from read_intan import get_ch_index_on_shank
from rec2nwb.preproc_func import get_or_set_device_type


def format_impedance(imp):
    """
    Format the impedance value:
      - If imp >= 1e6: show as x.xx Mohm.
      - If imp >= 1e3: show as xxxk Ω.
      - Else, show as integer ohms.
    """
    if imp >= 1e6:
        return f"{imp/1e6:.2f} Mohm"
    elif imp >= 1e3:
        return f"{imp/1e3:.0f}k Ω"
    else:
        return f"{imp:.0f} Ω"


def mannual_bad_ch_id(
    rhd_folder: Path,
    first_file: Path,
    n_shank: int,
    impedance_path: Path,
    device_type: str = "4shank16intan",) -> list:
    """
    Manually screen bad channels across shanks, saving segment screenshots.
    """
    bad_file = rhd_folder / "bad_channels.txt"
    if bad_file.exists():
        answer = input(f"{bad_file} already exists. Redo screening? (y/n): ")
        if answer.lower() not in ['y', 'yes']:
            print("Screening aborted. Using existing bad channel file.")
            return [line.strip() for line in open(bad_file)]

    # derive animal_id and session_id from folder structure
    # assumes: .../<animal_id>/<session_date>/<session_id_folder>
    animal_id  = rhd_folder.parent.name
    session_id = rhd_folder.name
    out_root   = Path("sortout") / animal_id / session_id

    print("Loading impedance file...")
    impedance_table = pd.read_csv(impedance_path)
    impedance       = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()
    channel_name    = impedance_table['Channel Name'].to_numpy()

    all_bad_ch_ids = []

    for ishank in range(n_shank):
        # make output folder for this shank
        out_folder = out_root / f"shank{ishank}"
        out_folder.mkdir(parents=True, exist_ok=True)

        recording = se.read_intan(first_file, stream_id='0')
        # Bandpass filter to remove both low and high frequency noise
        rec_filter = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
        fs         = recording.sampling_frequency

        # Get channel indices and coordinates for this shank
        channel_index, xcoord, ycoord = get_ch_index_on_shank(ishank, device_type)
        impedance_sh = impedance[channel_index]
        channel_ids   = channel_name[channel_index]

        # Create groups for each shank
        all_groups = []
        for s in range(n_shank):
            ch_idx, _, _ = get_ch_index_on_shank(s, device_type)
            all_groups.append(channel_name[ch_idx].tolist())

        # Apply common reference with shank-based groups
        rec_cr     = sp.common_reference(rec_filter, reference='global', operator='median', groups=all_groups)
        trace      = rec_cr.get_traces(channel_ids=channel_ids)
        segment_duration   = 3   # seconds per segment
        n_samples_segment  = int(segment_duration * fs)
        screening_duration = 30  # total seconds to screen
        total_samples      = int(screening_duration * fs)
        segment_starts     = np.arange(0, total_samples, n_samples_segment)

        shank_bad = set()

        for seg_start in segment_starts:
            seg_end    = min(seg_start + n_samples_segment, total_samples)
            time_axis  = np.arange(seg_start, seg_end) / fs

            fig = plt.figure(figsize=(12, 8))
            ax_anno = fig.add_axes([0.1, 0.1, 0.10, 0.8])
            ax_main = fig.add_axes([0.1, 0.1, 0.75, 0.8])
            rax     = fig.add_axes([0.85, 0.1, 0.12, 0.8])

            seg_stds = np.std(trace[seg_start:seg_end, :], axis=0)
            offset_multiplier = np.median(seg_stds) * 15
            offsets = -np.arange(len(channel_ids)) * offset_multiplier

            for i, cid in enumerate(channel_ids):
                ax_main.plot(time_axis, trace[seg_start:seg_end, i] + offsets[i],
                             color='k', lw=0.8)

            ax_main.set_title(f"Shank {ishank} | {seg_start/fs:.1f}-{seg_end/fs:.1f}s")
            ax_main.set_xlabel("Time (s)")
            ax_main.set_xlim([time_axis[0], time_axis[-1]])
            ax_main.set_ylim([offsets.min() - offset_multiplier,
                              offsets.max() + offset_multiplier])
            ax_main.set_yticks([])

            ax_anno.set_ylim(ax_main.get_ylim())
            ax_anno.set_yticks(offsets)
            anno_labels = [
                f"{cid}:{format_impedance(impedance_sh[i])}"
                for i, cid in enumerate(channel_ids)
            ]
            ax_anno.set_yticklabels(anno_labels, fontsize=8)
            ax_anno.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            for spine in ["top", "right", "bottom"]:
                ax_anno.spines[spine].set_visible(False)

            seg_bad_flags = {cid: (cid in shank_bad) for cid in channel_ids}
            visibility    = [seg_bad_flags[cid] for cid in channel_ids]
            check         = CheckButtons(rax, channel_ids, visibility)

            def checkbox_callback(label):
                seg_bad_flags[label] = not seg_bad_flags[label]
                if seg_bad_flags[label]:
                    print(f"Channel {label} marked bad.")
                else:
                    print(f"Channel {label} unmarked.")

            check.on_clicked(checkbox_callback)

            # Add a 'Finish' button to exit the loop
            finish_ax = fig.add_axes([0.85, 0.9, 0.12, 0.05])
            finish_button = Button(finish_ax, 'Finish')
            finish_button.label.set_fontsize(10)

            # Flag to check if the loop should be exited
            exit_loop = {'flag': False}

            def finish_callback(event):
                print("Exiting screening loop.")
                exit_loop['flag'] = True

            finish_button.on_clicked(finish_callback)

            print(f"Reviewing shank {ishank}, segment {seg_start/fs:.1f}-{seg_end/fs:.1f}s.")
            plt.show()

            # update bad set before checking if we should break
            for cid, is_bad in seg_bad_flags.items():
                if is_bad:
                    shank_bad.add(cid)
                else:
                    shank_bad.discard(cid)

            # save figure after closing
            seg_label = f"{int(seg_start/fs)}-{int(seg_end/fs)}s"
            img_path  = out_folder / f"shank{ishank}_seg_{seg_label}.png"
            fig.savefig(img_path, dpi=150)
            print(f"Saved screening image to: {img_path}")
            plt.close(fig)

            # Check if the finish button was clicked
            if exit_loop['flag']:
                break

        # Save bad channels for the current shank
        all_bad_ch_ids.extend(sorted(shank_bad))
        print(f"Shank {ishank} bad channels: {sorted(shank_bad)}")

    # write out bad channels
    with open(bad_file, "w") as f:
        for cid in all_bad_ch_ids:
            f.write(cid + "\n")

    print(f"Bad channel IDs saved to {bad_file}")
    return all_bad_ch_ids


if __name__ == "__main__":
    # 1) figure out the animal_id from your folder structure:
    rhd_folder = Path(input("Enter full path to RHD folder: ").strip())
    # e.g. /…/250504/CoI06/CoI06_250504_205955
    impedance_path   = input("Please enter the full path to the impedance file: ").strip().strip('"')
    impedance_file   = Path(impedance_path)
    print("Using impedance file:", impedance_file)

    # animal_id might be two levels up
    animal_id = rhd_folder.parent.name

    # 2) get (or choose) the device_type
    device_type = get_or_set_device_type(animal_id)
    print("Using device type:", device_type)
    # 3) get the number of shanks
    n_shank = int(input("Enter the number of shanks: ").strip())

    # pick first .rhd or .rhs, exclude macOS system files like ._*
    data_files = sorted(
        p for p in rhd_folder.iterdir()
        if p.suffix.lower() in ('.rhd', '.rhs') and not p.name.startswith("._")
    )
    if not data_files:
        raise FileNotFoundError("No .rhd or .rhs files found.")
    first_file = data_files[0]

    mannual_bad_ch_id(rhd_folder, first_file, n_shank, impedance_file, device_type)