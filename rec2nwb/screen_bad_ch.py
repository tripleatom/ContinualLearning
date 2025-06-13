import spikeinterface.preprocessing as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import numpy as np
import pandas as pd
from pathlib import Path
from spikeinterface import extractors as se
import os
import shutil
import re
import json
from rec2nwb.preproc_func import get_or_set_device_type


def get_ch_index_on_shank(ish: int, device_type: str) -> tuple:
    """
    Return the channel indices on a given shank.
    Returns: (channel indices, x-coordinates, y-coordinates)
    """
    script_dir = Path(__file__).resolve().parent
    mapping_file = script_dir / "mapping" / f"{device_type}.csv"

    channel_map = pd.read_csv(mapping_file)
    xcoord = channel_map['xcoord'].astype(float).to_numpy()
    ycoord = channel_map['ycoord'].astype(float).to_numpy()
    sh = channel_map['sh'].astype(int).to_numpy()

    ch_index = np.where(sh == ish)[0]
    return ch_index, xcoord[ch_index], ycoord[ch_index]


def format_impedance(imp):
    """
    Format the impedance value:
      - If imp >= 1e6: show as x.xx Mohm.
      - If imp >= 1e3: show as xxxk Ω.
      - Else, show as integer ohms.
    """
    if pd.isna(imp) or imp == 0:
        return "N/A"
    if imp >= 1e6:
        return f"{imp/1e6:.2f} Mohm"
    elif imp >= 1e3:
        return f"{imp/1e3:.0f}k Ω"
    else:
        return f"{imp:.0f} Ω"


class BadChannelScreener:
    """Unified bad channel screener for Intan and SpikeGadgets data."""
    
    def __init__(self, recording_method: str):
        """
        Initialize screener with recording method.
        
        Args:
            recording_method: Either 'intan' or 'spikegadget'
        """
        if recording_method not in ['intan', 'spikegadget']:
            raise ValueError("Recording method must be 'intan' or 'spikegadget'")
        self.recording_method = recording_method

    def _setup_spikegadget_files(self, data_file: Path):
        """Setup required files for SpikeGadgets reading."""
        if self.recording_method != 'spikegadget':
            return
            
        # For .rec files, the parent directory is where we need the files
        rec_folder = data_file.parent
        script_dir = Path(__file__).resolve().parent
        params_path = script_dir / "params.json"
        geom_path = script_dir / "geom.csv"
        
        if params_path.exists() and geom_path.exists():
            shutil.copy2(params_path, rec_folder)
            shutil.copy2(geom_path, rec_folder)
        else:
            print("Warning: params.json or geom.csv not found. SpikeGadgets reading may fail.")

    def _get_spikegadget_parts(self, base_rec_file: Path) -> list:
        """
        Get all parts of a SpikeGadgets recording.
        
        Args:
            base_rec_file: The base .rec file (could be part1 or any part)
            
        Returns:
            List of .rec files in order (part1, part2, part3, etc.)
        """
        rec_folder = base_rec_file.parent
        base_name = base_rec_file.stem
        
        # Remove any existing part number from the base name
        base_name_clean = re.sub(r'\.part\d+$', '', base_name)
        
        # Find all parts
        rec_parts = []
        
        # Look for the file without part number (this is part 1)
        part1_file = rec_folder / f"{base_name_clean}.rec"
        if part1_file.exists():
            rec_parts.append(part1_file)
        
        # Look for numbered parts (part2, part3, etc.)
        part_num = 2
        while True:
            part_file = rec_folder / f"{base_name_clean}.part{part_num}.rec"
            if part_file.exists():
                rec_parts.append(part_file)
                part_num += 1
            else:
                break
        
        if not rec_parts:
            # If no parts found, return the original file
            rec_parts = [base_rec_file]
        
        return sorted(rec_parts)

    def _read_recording(self, data_file: Path):
        """Read recording data based on recording method."""
        if self.recording_method == 'intan':
            recording = se.read_intan(data_file, stream_id='0')
        else:  # spikegadget
            self._setup_spikegadget_files(data_file)
            # For SpikeGadgets, read the recording directly
            recording = se.read_spikegadgets(data_file)
        return recording

    def _get_channel_info(self, ishank: int, device_type: str, impedance_path: Path = None, recording=None):
        """Get channel information for a specific shank."""
        # Get channel indices and coordinates for this shank
        channel_index, xcoord, ycoord = get_ch_index_on_shank(ishank, device_type)
        
        # Handle impedance data and channel names
        impedance_sh = None
        channel_ids = None
        
        if impedance_path and impedance_path.exists():
            impedance_table = pd.read_csv(impedance_path)
            impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()
            channel_name = impedance_table['Channel Name'].to_numpy()
            impedance_sh = impedance[channel_index]
            channel_ids = channel_name[channel_index]
        else:
            # Get channel names from recording if available
            if recording is not None:
                all_channel_ids = recording.get_channel_ids()
                if self.recording_method == 'intan':
                    # For Intan, use the channel names from recording
                    channel_ids = np.array([all_channel_ids[i] for i in channel_index if i < len(all_channel_ids)])
                else:
                    # For SpikeGadgets, map electrode indices to actual channel IDs
                    channel_ids = []
                    available_channel_ids = [str(ch_id) for ch_id in all_channel_ids]
                    for ch_idx in channel_index:
                        if str(ch_idx) in available_channel_ids:
                            channel_ids.append(str(ch_idx))
                    channel_ids = np.array(channel_ids)
            else:
                # Fallback to default names
                if self.recording_method == 'intan':
                    channel_ids = np.array([f"ch{i}" for i in channel_index])
                else:
                    channel_ids = np.array([str(i) for i in channel_index])
            impedance_sh = np.full(len(channel_ids), np.nan)
        
        return channel_index[:len(channel_ids)], channel_ids, impedance_sh, xcoord[:len(channel_ids)], ycoord[:len(channel_ids)]

    def _get_all_channel_groups(self, n_shank: int, device_type: str, impedance_path: Path = None, recording=None):
        """Get all channel groups for common reference."""
        all_groups = []
        for s in range(n_shank):
            ch_idx, ch_ids, _, _, _ = self._get_channel_info(s, device_type, impedance_path, recording)
            if self.recording_method == 'intan':
                all_groups.append(ch_ids.tolist())
            else:
                # For SpikeGadgets, use the string channel IDs
                all_groups.append(ch_ids.tolist())
        return all_groups

    def get_data_files(self, data_folder: Path) -> list:
        """Get list of data files based on recording method."""
        if self.recording_method == 'intan':
            # Get .rhd or .rhs files, exclude macOS system files
            data_files = sorted(
                p for p in data_folder.iterdir()
                if p.suffix.lower() in ('.rhd', '.rhs') and not p.name.startswith("._"))
        else:  # spikegadget
            # Find all .rec files in the folder
            rec_files = list(data_folder.glob('*.rec'))
            
            if not rec_files:
                raise FileNotFoundError("No .rec files found in the specified folder.")
            
            # Group files by their base name (without part numbers)
            file_groups = {}
            for rec_file in rec_files:
                # Remove .part# from the filename to get the base name
                base_name = re.sub(r'\.part\d+$', '', rec_file.stem)
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(rec_file)
            
            # For each group, get all parts in order
            all_data_files = []
            for base_name, files in file_groups.items():
                # Get the first file as reference to find all parts
                ref_file = files[0]
                parts = self._get_spikegadget_parts(ref_file)
                all_data_files.extend(parts)
            
            data_files = all_data_files
        
        if not data_files:
            file_types = ".rhd/.rhs" if self.recording_method == 'intan' else ".rec"
            raise FileNotFoundError(f"No {file_types} files found in the specified folder.")
        
        return data_files

    def get_session_description(self, data_folder: Path) -> str:
        """Extract session description from folder path."""
        if self.recording_method == 'spikegadget':
            # For SpikeGadgets: use folder name directly
            return data_folder.name
        else:
            # For Intan: use folder name
            return data_folder.name

    def manual_bad_ch_id(self, data_folder: Path, first_file: Path, n_shank: int, 
                        impedance_path: Path = None, device_type: str = "4shank16") -> list:
        """
        Manually screen bad channels across shanks, saving segment screenshots.
        """
        bad_file = data_folder / "bad_channels.txt"
        if bad_file.exists():
            answer = input(f"{bad_file} already exists. Redo screening? (y/n): ")
            if answer.lower() not in ['y', 'yes']:
                print("Screening aborted. Using existing bad channel file.")
                return [line.strip() for line in open(bad_file)]

        # Derive animal_id and session_id from folder structure
        animal_id = data_folder.parent.name
        session_id = self.get_session_description(data_folder)
        out_root = Path("sortout") / animal_id / session_id

        # Load recording
        print("Loading recording...")
        recording = self._read_recording(first_file)
        fs = recording.sampling_frequency
        
        print(f"Recording info: {recording.get_num_channels()} channels, {recording.get_num_samples()} samples, {fs} Hz")
        
        # For SpikeGadgets, limit the data size for faster processing
        if self.recording_method == 'spikegadget':
            max_samples = min(recording.get_num_samples(), int(60 * fs))  # Limit to 60 seconds max
            print(f"Using first {max_samples/fs:.1f} seconds of data for screening")
        else:
            max_samples = recording.get_num_samples()
        
        # Get a subset of data for faster filtering
        print("Applying bandpass filter...")
        if self.recording_method == 'spikegadget':
            # For SpikeGadgets, work with smaller chunks to avoid memory issues
            rec_subset = recording.frame_slice(start_frame=0, end_frame=max_samples)
            rec_filter = sp.bandpass_filter(rec_subset, freq_min=300, freq_max=6000, dtype=np.float32)
        else:
            rec_filter = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)

        # Get all channel groups for common reference
        print("Setting up common reference...")
        all_groups = self._get_all_channel_groups(n_shank, device_type, impedance_path, recording)
        
        # Apply common reference with shank-based groups
        rec_cr = sp.common_reference(rec_filter, reference='global', operator='median', groups=all_groups)

        all_bad_ch_ids = []

        for ishank in range(n_shank):
            print(f"Processing shank {ishank}...")
            # Make output folder for this shank
            out_folder = out_root / f"shank{ishank}"
            out_folder.mkdir(parents=True, exist_ok=True)

            # Get channel information for this shank
            channel_index, channel_ids, impedance_sh, xcoord, ycoord = self._get_channel_info(
                ishank, device_type, impedance_path, recording)

            if len(channel_ids) == 0:
                print(f"Warning: No valid channels found for shank {ishank}. Skipping.")
                continue

            # Get traces for this shank
            print(f"Loading traces for shank {ishank}...")
            if self.recording_method == 'intan':
                trace = rec_cr.get_traces(channel_ids=channel_ids.tolist())
                display_ids = channel_ids
            else:
                # For SpikeGadgets, channel_ids are already strings
                trace = rec_cr.get_traces(channel_ids=channel_ids.tolist())
                display_ids = [f"ch{ch_id}" for ch_id in channel_ids]

            # Sort traces by depth (y-coordinate) - shallow channels first (0μm at top)
            depth_order = np.argsort(ycoord)  # Sort ascending: 0μm, 25μm, 50μm, etc.
            
            # Reorder everything according to depth
            trace = trace[:, depth_order]
            display_ids = [display_ids[i] for i in depth_order]
            impedance_sh_sorted = impedance_sh[depth_order]
            ycoord_sorted = ycoord[depth_order]
            xcoord_sorted = xcoord[depth_order]
            channel_ids_sorted = channel_ids[depth_order]

            print(f"Trace shape: {trace.shape}")
            print(f"Depth range: {ycoord_sorted.min():.1f} to {ycoord_sorted.max():.1f} μm (shallow to deep)")

            segment_duration = 3   # seconds per segment
            n_samples_segment = int(segment_duration * fs)
            screening_duration = min(30, trace.shape[0] / fs)  # Adapt to available data
            total_samples = int(screening_duration * fs)
            segment_starts = np.arange(0, min(total_samples, trace.shape[0]), n_samples_segment)

            print(f"Will screen {len(segment_starts)} segments of {segment_duration}s each")

            shank_bad = set()

            for seg_idx, seg_start in enumerate(segment_starts):
                seg_end = min(seg_start + n_samples_segment, trace.shape[0])
                time_axis = np.arange(seg_start, seg_end) / fs

                print(f"Displaying segment {seg_idx+1}/{len(segment_starts)}: {seg_start/fs:.1f}-{seg_end/fs:.1f}s")

                fig = plt.figure(figsize=(12, 8))
                ax_anno = fig.add_axes([0.05, 0.1, 0.12, 0.8])  # Moved closer to main plot
                ax_main = fig.add_axes([0.18, 0.1, 0.70, 0.8])   # Adjusted main plot position
                rax = fig.add_axes([0.89, 0.1, 0.10, 0.8])       # Adjusted checkbox position

                seg_stds = np.std(trace[seg_start:seg_end, :], axis=0)
                offset_multiplier = np.median(seg_stds) * 15
                offsets = -np.arange(len(display_ids)) * offset_multiplier

                for i, cid in enumerate(display_ids):
                    ax_main.plot(time_axis, trace[seg_start:seg_end, i] + offsets[i],
                                 color='k', lw=0.8)

                ax_main.set_title(f"Shank {ishank} | {seg_start/fs:.1f}-{seg_end/fs:.1f}s | {self.recording_method.upper()} | 0μm→deep")
                ax_main.set_xlabel("Time (s)")
                ax_main.set_xlim([time_axis[0], time_axis[-1]])
                ax_main.set_ylim([offsets.min() - offset_multiplier,
                                  offsets.max() + offset_multiplier])
                ax_main.set_yticks([])
                # Remove y-axis spine
                ax_main.spines['left'].set_visible(False)

                ax_anno.set_ylim(ax_main.get_ylim())
                ax_anno.set_yticks(offsets)
                anno_labels = [
                    f"{cid}:{format_impedance(impedance_sh_sorted[i])} @{ycoord_sorted[i]:.0f}μm"
                    for i, cid in enumerate(display_ids)
                ]
                ax_anno.set_yticklabels(anno_labels, fontsize=8, ha='right')  # Right-align and larger font
                ax_anno.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax_anno.tick_params(axis='y', which='both', left=False, right=False, labelright=True, labelleft=False)  # Remove tick marks
                # Remove all spines for cleaner look
                for spine in ax_anno.spines.values():
                    spine.set_visible(False)

                # Use the actual channel IDs for checkbox labels (for saving to bad_channels.txt)
                checkbox_labels = [str(channel_ids_sorted[i]) for i in range(len(display_ids))]
                seg_bad_flags = {label: (label in shank_bad) for label in checkbox_labels}
                visibility = [seg_bad_flags[label] for label in checkbox_labels]
                check = CheckButtons(rax, checkbox_labels, visibility)

                def checkbox_callback(label):
                    seg_bad_flags[label] = not seg_bad_flags[label]
                    if seg_bad_flags[label]:
                        print(f"Channel {label} marked bad.")
                    else:
                        print(f"Channel {label} unmarked.")

                check.on_clicked(checkbox_callback)

                # Add a 'Finish' button to exit the loop
                finish_ax = fig.add_axes([0.89, 0.92, 0.10, 0.05])  # Adjusted position
                finish_button = Button(finish_ax, 'Finish')
                finish_button.label.set_fontsize(10)

                # Flag to check if the loop should be exited
                exit_loop = {'flag': False}

                def finish_callback(event):
                    print("Exiting screening loop.")
                    exit_loop['flag'] = True
                    plt.close(fig)

                finish_button.on_clicked(finish_callback)

                print(f"Reviewing shank {ishank}, segment {seg_start/fs:.1f}-{seg_end/fs:.1f}s.")
                plt.show()

                # Update bad set before checking if we should break
                for cid, is_bad in seg_bad_flags.items():
                    if is_bad:
                        shank_bad.add(cid)
                    else:
                        shank_bad.discard(cid)

                # Save figure after closing
                seg_label = f"{int(seg_start/fs)}-{int(seg_end/fs)}s"
                img_path = out_folder / f"shank{ishank}_seg_{seg_label}.png"
                fig.savefig(img_path, dpi=150)
                print(f"Saved screening image to: {img_path}")
                plt.close(fig)

                # Check if the finish button was clicked
                if exit_loop['flag']:
                    break

            # Save bad channels for the current shank
            all_bad_ch_ids.extend(sorted(shank_bad))
            print(f"Shank {ishank} bad channels: {sorted(shank_bad)}")

        # Write out bad channels
        with open(bad_file, "w") as f:
            for cid in all_bad_ch_ids:
                f.write(str(cid) + "\n")

        print(f"Bad channel IDs saved to {bad_file}")
        return all_bad_ch_ids


def main():
    """Main function to run the bad channel screener."""
    # Choose recording method
    print("Choose recording method:")
    print("1. Intan")
    print("2. SpikeGadgets")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        recording_method = 'intan'
        folder_prompt = "Enter full path to Intan data folder: "
    elif choice == '2':
        recording_method = 'spikegadget'
        folder_prompt = "Enter full path to folder containing .rec files: "
    else:
        print("Invalid choice. Exiting.")
        return

    # Initialize screener
    screener = BadChannelScreener(recording_method)
    
    # Get inputs
    data_folder = Path(input(folder_prompt).strip().strip('"').strip("'"))
    
    impedance_path_input = input("Please enter the full path to the impedance file (optional, press Enter to skip): ").strip().strip('"').strip("'")
    impedance_file = Path(impedance_path_input) if impedance_path_input else None
    
    if impedance_file and impedance_file.exists():
        print("Using impedance file:", impedance_file)
    else:
        print("No impedance file provided or file not found. Using default channel info.")
        impedance_file = None

    # Get animal_id and device_type
    animal_id = data_folder.parent.name
    device_type = get_or_set_device_type(animal_id)
    print("Using device type:", device_type)
    
    # Get number of shanks
    n_shank = int(input("Enter the number of shanks: ").strip())

    # Get data files
    try:
        data_files = screener.get_data_files(data_folder)
        first_file = data_files[0]
        print(f"Found {len(data_files)} data files. Using first file: {first_file.name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Run screening
    try:
        bad_channels = screener.manual_bad_ch_id(
            data_folder, first_file, n_shank, impedance_file, device_type)
        print(f"Screening completed. Found {len(bad_channels)} bad channels total.")
    except Exception as e:
        print(f"Error during screening: {e}")
        return


if __name__ == "__main__":
    main()