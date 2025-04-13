import spikeinterface.preprocessing as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
import pandas as pd
import os
from pathlib import Path
from spikeinterface import extractors as se
from read_intan import get_ch_index_on_shank

# This script is used to manually screen bad channels in a neural recording.

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


def mannual_bad_ch_id(rhd_folder: Path, 
                       first_rhd_file: Path, 
                       n_shank: int, 
                       impedance_path: Path, 
                       device_type: str = "4shank16intan") -> list:
    """
    Identify bad channels by plotting traces in segments (e.g., 3-second segments from a 30-second screening)
    and allowing interactive marking via CheckButtons. The impedance for each channel is displayed
    in a separate (left-side) annotation axis.
    
    Steps:
      1. Load the impedance from a CSV file.
      2. For each shank:
            - Load and filter the recording.
            - For each screening segment, plot:
                * ax_anno (left) shows channel IDs and formatted impedances.
                * ax_main (center) shows the raw signal traces with vertical offsets.
                * CheckButtons (right) allow marking bad channels.
            - The bad channel status is carried forward to subsequent segments.
      3. Save the accumulated bad channel IDs to a file and return them.
    """
    bad_file = Path(rhd_folder) / "bad_channels.txt"
    # Check if the bad channel file already exists.
    if bad_file.exists():
        answer = input(f"{bad_file} already exists. Redo screening? (y/n): ")
        if answer.lower() not in ['y', 'yes']:
            print("Screening aborted. Using existing bad channel file.")
            with open(bad_file, "r") as f:
                existing_bad = [line.strip() for line in f.readlines()]
            return existing_bad


    # This list will hold bad channel IDs from all shanks.
    all_bad_ch_ids = []

    # Load the impedance file (assumes one CSV for all channels).
    print("Loading impedance file...")
    impedance_table = pd.read_csv(impedance_path)
    impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()

    # Loop over each shank.
    for ishank in range(n_shank):
        # Get the channel indices for this shank.
        channel_index, _, _ = get_ch_index_on_shank(ishank, device_type)
        impedance_sh = impedance[channel_index]
        channel_ids = [f"A-{i:03d}" for i in channel_index]

        # Load the recording and filter the data.
        recording = se.read_intan(first_rhd_file, stream_id='0')
        rec_filter = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
        fs = recording.sampling_frequency
        rec_cr = sp.common_reference(rec_filter, reference='global', operator='median')
        trace = rec_cr.get_traces(channel_ids=channel_ids)
        # 'trace' shape is [n_time_points, n_channels]

        # Plotting parameters.
        segment_duration = 3  # seconds per segment for screening
        n_samples_segment = int(segment_duration * fs)
        screening_duration = 30  # total screening duration in seconds
        total_samples = int(screening_duration * fs)
        segment_starts = np.arange(0, total_samples, n_samples_segment)

        # Persistent set to accumulate bad channels for this shank.
        shank_bad = set()

        # Loop over segments.
        for seg_start in segment_starts:
            seg_end = min(seg_start + n_samples_segment, total_samples)
            time_axis = np.arange(seg_start, seg_end) / fs  # in seconds

            # Create a new figure for this segment.
            # Three axes: annotation (left), main plot (center), and CheckButtons (right).
            fig = plt.figure(figsize=(12, 8))
            
            # Annotation axis: left side.
            ax_anno = fig.add_axes([0.1, 0.1, 0.10, 0.8])
            # Main signal axis.
            ax_main = fig.add_axes([0.1, 0.1, 0.75, 0.8])
            # CheckButtons axis.
            rax = fig.add_axes([0.85, 0.1, 0.12, 0.8])
            
            # Compute a robust offset based on the median standard deviation.
            seg_stds = np.std(trace[seg_start:seg_end, :], axis=0)
            offset_multiplier = np.median(seg_stds) * 15  # adjust multiplier if needed
            # Use negative offsets so that channel order appears top-to-bottom.
            offsets = -np.arange(len(channel_ids)) * offset_multiplier
            
            # Plot each channel trace on the main axis.
            for i, cid in enumerate(channel_ids):
                ch_trace = trace[seg_start:seg_end, i]
                ax_main.plot(time_axis, ch_trace + offsets[i], color='k', lw=0.8)
            
            ax_main.set_title(f"Shank {ishank} | Time: {seg_start/fs:.1f}-{seg_end/fs:.1f} s")
            ax_main.set_xlabel("Time (s)")
            ax_main.set_xlim([time_axis[0], time_axis[-1]])
            ax_main.set_ylim([np.min(offsets) - offset_multiplier, np.max(offsets) + offset_multiplier])
            ax_main.set_yticks([])  # Hide y ticks on the main axis

            # --- Annotation Axis Setup ---
            ax_anno.set_ylim(ax_main.get_ylim())
            ax_anno.set_yticks(offsets)
            anno_labels = []
            for i, cid in enumerate(channel_ids):
                imp_str = format_impedance(impedance_sh[i])
                anno_labels.append(f"{cid}:{imp_str}")
            ax_anno.set_yticklabels(anno_labels, fontsize=8)
            ax_anno.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            # Hide extra spines.
            for spine in ["top", "right", "bottom"]:
                ax_anno.spines[spine].set_visible(False)
            
            # --- CheckButtons Setup ---
            # Initialize the persistent state into the segment.
            # If a channel was already marked as bad, pre-check its box.
            seg_bad_flags = {cid: (cid in shank_bad) for cid in channel_ids}
            visibility = [seg_bad_flags[cid] for cid in channel_ids]
            check = CheckButtons(rax, channel_ids, visibility)
            
            # Callback to toggle channel bad status.
            def checkbox_callback(label):
                # Once a channel is flagged as bad, we keep it bad.
                seg_bad_flags[label] = True  
                # Optionally, you can print the change.
                print(f"Channel {label} marked as bad.")
            check.on_clicked(checkbox_callback)
            
            # Display the interactive window.
            print(f"Reviewing shank {ishank}, segment {seg_start/fs:.1f}-{seg_end/fs:.1f} s. "
                  f"Mark any bad channels using the check buttons and then close the window to continue...")
            plt.show()
            plt.close(fig)
            
            # Update persistent state: once bad, always keep it marked.
            for cid, is_bad in seg_bad_flags.items():
                if is_bad:
                    shank_bad.add(cid)
                    
        # After processing all segments for this shank, record the bad channels.
        all_bad_ch_ids.extend(sorted(shank_bad))
        print(f"Shank {ishank}: Marked bad channels: {sorted(shank_bad)}")

    # Save the complete list of bad channel IDs to a file.
    with open(bad_file, "w") as f:
        for cid in all_bad_ch_ids:
            f.write(cid + "\n")

    print(f"\nBad channel IDs saved in: {bad_file}")
    return all_bad_ch_ids

if __name__ == "__main__":
    # Define folder and file paths
    rhd_folder_input = input("Please enter the full path to the RHD folder: ")
    # rhd_folder_input = '/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/20250411/CnL36/CnL36_250412_200459'
    rhd_folder = Path(rhd_folder_input)
    impedance_path = input("Please enter the full path to the impedance file: ")
    # impedance_path = '/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed/20250411/CnL36/CnL36.csv'
    # impedance_path = impedance_path.strip('"')

    impedance_file = Path(impedance_path)
    print("Using impedance file:", impedance_file)

    device_type = "4shank16intan"
    n_shank = 4

    session_description = os.path.basename(rhd_folder)

    # Gather all .rhd files in the folder and sort them
    rhd_files = sorted(rhd_folder.glob('*.rhd'))
    if not rhd_files:
        raise FileNotFoundError("No .rhd files found in the specified folder.")
    first_rhd_file = rhd_files[0]
    mannual_bad_ch_id(rhd_folder, first_rhd_file, n_shank, impedance_file, device_type)