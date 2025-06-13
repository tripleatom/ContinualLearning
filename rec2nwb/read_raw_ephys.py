import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import sys
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import neo.rawio
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO
from rec2nwb.preproc_func import get_or_set_device_type
import os
import shutil
import json


class EphysToNWBConverter:
    """Unified converter for Intan and SpikeGadgets data to NWB format."""
    
    def __init__(self, recording_method: str):
        """
        Initialize converter with recording method.
        
        Args:
            recording_method: Either 'intan' or 'spikegadget'
        """
        if recording_method not in ['intan', 'spikegadget']:
            raise ValueError("Recording method must be 'intan' or 'spikegadget'")
        self.recording_method = recording_method
    
    def get_stream_ids(self, file_path: str) -> any:
        """
        Get the stream ids from an Intan file.
        Only applies to Intan recordings.
        """
        if self.recording_method != 'intan':
            return None
            
        file_path = str(file_path)
        reader = neo.rawio.IntanRawIO(filename=file_path)
        reader.parse_header()
        header = reader.header
        return header['signal_streams']['id']

    def get_timestamp(self, file_path: Path) -> datetime:
        """
        Extract the recording start time from the filename.
        Handles both Intan and SpikeGadgets formats.
        """
        if self.recording_method == 'intan':
            # Expected format: <prefix>_yymmdd_HHMMSS.rh[s|d]
            match = re.match(
                r"([a-zA-Z0-9_]+)_([0-9]+_[0-9]+).rh(?:s|d)", file_path.name)
            if match:
                rec_datetimestr = match.group(2)  # yymmdd_HHMMSS
                return datetime.strptime(rec_datetimestr, "%y%m%d_%H%M%S")
        else:
            # Expected format: contains _YYYYMMDD_HHMMSS
            match = re.search(r"_(\d{8})_(\d{6})", file_path.name)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        
        raise ValueError("Filename does not match expected pattern.")

    def get_ch_index_on_shank(self, ish: int, device_type: str) -> tuple:
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

    def _setup_spikegadget_files(self, data_file: Path):
        """Setup required files for SpikeGadgets reading."""
        if self.recording_method != 'spikegadget':
            return
            
        # For .rec files, the parent directory is where we need the files
        rec_folder = data_file.parent
        script_dir = Path(__file__).resolve().parent
        params_path = script_dir / "params.json"
        # geom_path = script_dir / "geom.csv"
        shutil.copy2(params_path, rec_folder)
        # shutil.copy2(geom_path, rec_folder)

    def _get_sampling_rate(self, data_file: Path) -> float:
        """
        Get sampling rate based on recording method.
        For SpikeGadgets, read from params.json. For Intan, use recording's rate.
        """
        if self.recording_method == 'spikegadget':
            # Read sampling rate from params.json in the script directory
            script_dir = Path(__file__).resolve().parent
            params_file = script_dir / "params.json"
            
            if params_file.exists():
                with open(params_file, 'r') as f:
                    params = json.load(f)
                return float(params.get("samplerate", 30000))  # Default to 30000 if not found
            else:
                print(f"Warning: params.json not found in {script_dir}, using default rate 30000")
                return 30000.0
        else:
            # For Intan, we'll get it from the recording object later
            return None

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

    def _read_recording(self, data_file: Path, channel_ids: list = None, return_trace: bool = True):
        """Read recording data based on recording method.
        If return_trace is False, only return the recording object and sampling rate
        """
        trace = None
        conversion = None
        offset = None
        sampling_rate = None
        if self.recording_method == 'intan' and return_trace:
            recording = se.read_intan(data_file, stream_id='0')
            if channel_ids:
                trace = recording.get_traces(channel_ids=channel_ids)
            else:
                trace = recording.get_traces()
            conversion = recording.get_channel_gains()[0] / 1e6  # convert uV to V
            offset = recording.get_channel_offsets()[0] / 1e6
            sampling_rate = recording.get_sampling_frequency()
        elif self.recording_method == 'intan' and not return_trace:
            recording = se.read_intan(data_file, stream_id='0')
            sampling_rate = recording.get_sampling_frequency()
        elif self.recording_method == 'spikegadget' and return_trace:
            self._setup_spikegadget_files(data_file)
            # For SpikeGadgets, read the recording directly
            recording = se.read_spikegadgets(data_file)
            if channel_ids:
                # Convert channel_ids to strings since SpikeGadgets uses string IDs
                channel_ids_str = [str(ch_id) for ch_id in channel_ids]
                trace = recording.get_traces(channel_ids=channel_ids_str)
            else:
                trace = recording.get_traces()
            conversion = 0.195 / 1e6  # Convert to V
            offset = 0.0 / 1e6
            # Get sampling rate from params.json instead of recording
            sampling_rate = self._get_sampling_rate(data_file)
        elif self.recording_method == 'spikegadget' and not return_trace:
            recording = se.read_spikegadgets(data_file)
            sampling_rate = self._get_sampling_rate(data_file)
        else:
            raise ValueError("Invalid recording method or return_trace flag")
        return recording, trace, conversion, offset, sampling_rate

    def initiate_nwb(self, data_file: Path, nwb_path: Path, ishank: int = 0,
                     impedance_path: str = None, bad_ch_ids: list = None,
                     metadata: dict = None) -> list:
        """
        Create and write an NWB file from recording data.
        """
        metadata = metadata or {}
        print("Initiating NWB file...")
        
        session_start_time = self.get_timestamp(data_file)
        nwb_description = metadata.get("session_desc", f"NWB file for {self.recording_method} data")
        experimenter = metadata.get("experimenter", "Zhang, Xiaorong")
        lab = metadata.get("lab", "XL Lab")
        institution = metadata.get("institution", "Rice University")
        exp_desc = metadata.get("exp_desc", "None")
        session_id = metadata.get("session_id", "None")
        electrode_location = metadata.get("electrode_location", None)
        device_type = metadata.get("device_type", "4shank16intan" if self.recording_method == 'intan' else "4shank16")

        nwbfile = NWBFile(
            session_description=nwb_description,
            identifier=str(uuid4()),
            session_start_time=session_start_time,
            experimenter=[experimenter],
            lab=lab,
            institution=institution,
            experiment_description=exp_desc,
            session_id=session_id,
        )

        print("Adding device...")
        channel_index, xcoord, ycoord = self.get_ch_index_on_shank(ishank, device_type)
        
        # Create a device and add electrode metadata
        device = nwbfile.create_device(
            name="--", description="--", manufacturer="--")
        nwbfile.add_electrode_column(
            name="label", description="label of electrode")

        electrode_group = nwbfile.create_electrode_group(
            name=f"shank{ishank}",
            description=f"electrode group for shank {ishank}",
            device=device,
            location=electrode_location,
        )

        # Handle impedance and channel names
        impedance_sh = None
        channel_name_sh = None
        if impedance_path is not None:
            impedance_table = pd.read_csv(impedance_path)
            impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()
            impedance_sh = impedance[channel_index]
            channel_name = impedance_table['Channel Name'].to_numpy()
            channel_name_sh = channel_name[channel_index]
        else:
            # Create default channel names and impedances
            channel_name_sh = [f"{i}" for i in channel_index] #SpikeGadgets uses string channel names
            impedance_sh = [np.nan] * len(channel_index)

        # Create electrode DataFrame
        electrode_df = pd.DataFrame({
            'channel_name': channel_name_sh,
            'impedance': impedance_sh,
            'x': xcoord,
            'y': ycoord,
            'channel_index': channel_index
        })

        # Remove bad channels from the DataFrame
        if bad_ch_ids is not None:
            electrode_df = electrode_df[~electrode_df['channel_name'].isin(bad_ch_ids)]

        n_electrodes = len(electrode_df)
        print(f"Number of good electrodes: {n_electrodes}")
        
        # Add electrodes to NWB file
        for idx, row in electrode_df.iterrows():
            nwbfile.add_electrode(
                group=electrode_group,
                label=f"shank{ishank}:{row['channel_name']}",
                location=electrode_location,
                rel_x=float(row['x']),
                rel_y=float(row['y']),
                imp=float(row['impedance']) if not np.isnan(row['impedance']) else 0.0,
            )

        electrode_table_region = nwbfile.create_electrode_table_region(
            list(range(n_electrodes)), "all electrodes"
        )

        # Read recording data
        print("Adding electrical data...")
        
        # First read the recording to get available channel IDs
        recording, _, _, _, sampling_rate = self._read_recording(data_file, None, return_trace=False)
        
        if self.recording_method == 'spikegadget':
            # Get the actual channel IDs from the recording
            actual_channel_ids = recording.get_channel_ids()
            # print(f"Available channel IDs: {actual_channel_ids[:10]}...")  # Show first 10
            
            # Map electrode indices to actual channel IDs
            electrode_to_channel_map = {}
            good_channel_ids_for_recording = []
            
            for idx, row in electrode_df.iterrows():
                ch_index = row['channel_index']
                if str(ch_index) in actual_channel_ids:
                    electrode_to_channel_map[ch_index] = str(ch_index)
                    good_channel_ids_for_recording.append(str(ch_index))
                else:
                    print(f"Warning: Channel index {ch_index} not found in recording")
            
            # Get traces with the mapped channel IDs
            recording, trace, conversion, offset, sampling_rate = self._read_recording(data_file, good_channel_ids_for_recording)
            good_channel_ids = [int(ch_id) for ch_id in good_channel_ids_for_recording]  # Keep as integers for consistency
            
        else:  # intan
            if impedance_path is not None:
                good_channel_ids = electrode_df['channel_name'].tolist()
            else:
                good_channel_ids = electrode_df['channel_name'].tolist()
            recording, trace, conversion, offset, sampling_rate = self._read_recording(data_file, good_channel_ids)

        # Use the appropriate sampling rate
        if self.recording_method == 'intan':
            rate = recording.get_sampling_frequency()
        else:
            rate = sampling_rate

        electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=H5DataIO(data=trace, maxshape=(None, trace.shape[1])),
            electrodes=electrode_table_region,
            starting_time=0.0, #FIXME: starting_time could be the real starting time of the recording
            rate=rate,
            conversion=conversion,
            offset=offset,
        )
        nwbfile.add_acquisition(electrical_series)

        # Handle digital input for Intan
        if self.recording_method == 'intan':
            stream_ids = self.get_stream_ids(data_file)
            if '4' in stream_ids:
                print("Found digital input channels...")
                # TODO: Implement digital input handling if needed
                pass

        print("Writing NWB file...")
        with NWBHDF5IO(nwb_path, "w") as io:
            io.write(nwbfile)

        return good_channel_ids

    def _append_nwb_dset(self, dset, data_to_append, append_axis: int) -> None:
        """
        Append data along a specified axis in an HDF5 dataset.
        """
        dset_shape = dset.shape
        dset_len = dset_shape[append_axis]
        app_len = data_to_append.shape[append_axis]
        new_len = dset_len + app_len

        # Prepare slicer to index the appended region
        slicer = [slice(None)] * len(dset_shape)
        slicer[append_axis] = slice(-app_len, None)

        dset.resize(new_len, axis=append_axis)
        dset[tuple(slicer)] = data_to_append

    def append_nwb(self, nwb_path: Path, data_file: Path, channel_ids: list = None,
                   metadata: dict = None) -> None:
        """
        Append additional recording data to an existing NWB file.
        """
        metadata = metadata or {}
        with NWBHDF5IO(nwb_path, "a") as io:
            nwb_obj = io.read()
            
            if self.recording_method == 'spikegadget':
                # Convert channel_ids to strings for SpikeGadgets
                channel_ids_str = [str(ch_id) for ch_id in channel_ids] if channel_ids else None
                _, trace, _, _, _ = self._read_recording(data_file, channel_ids_str)
            else:
                _, trace, _, _, _ = self._read_recording(data_file, channel_ids)
                
            self._append_nwb_dset(
                nwb_obj.acquisition['ElectricalSeries'].data, trace, 0)
            io.write(nwb_obj)

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

            def part_key(f:Path):
                m = re.search(r"\.part(\d+)\.rec$", f.name)
                return (1, int(m.group(1))) if m else (0, 0)
            
            data_files = sorted(rec_files, key=part_key)
        
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


def load_bad_ch(bad_file: Path) -> list:
    """
    Load bad channels from a file.
    """
    if not bad_file.exists():
        print(f"No bad channels file found at {bad_file}. Using all channels.")
        return []
    with open(bad_file, "r") as f:
        bad_channels = [line.strip() for line in f.readlines()]
    return bad_channels


def main():
    """Main function to run the unified converter."""
    # Choose recording method
    print("Choose recording method:")
    print("1. Intan")
    print("2. SpikeGadgets")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        recording_method = 'intan'
        folder_prompt = "Please enter the full path to the Intan data folder: "
    elif choice == '2':
        recording_method = 'spikegadget'
        folder_prompt = "Please enter the full path to the folder containing .rec files: "
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Initialize converter
    converter = EphysToNWBConverter(recording_method)
    
    # Get inputs
    data_folder = Path(input(folder_prompt).strip().strip("'").strip('"'))
    impedance_file_input = input("Please enter the full path to the impedance file (optional, press Enter to skip): ").strip().strip("'").strip('"')
    impedance_file = Path(impedance_file_input) if impedance_file_input else None
    electrode_location = input("Please enter the electrode location: ").strip()
    exp_desc = input("Please enter the experiment description: ").strip() or "None"
    
    animal_id = data_folder.parent.name
    device_type = get_or_set_device_type(animal_id)
    raw = input("Please enter the shank numbers (e.g. 0,1,2,3 or [0,1,2,3]): ")
    shanks = [int(x) for x in re.findall(r'\d+', raw)]
    print(f"Processing shanks: {shanks}")
    
    session_description = converter.get_session_description(data_folder)
    
    if not data_folder.exists():
        print(f"Folder {data_folder} does not exist, exiting")
        sys.exit(1)

    # Get data files
    data_files = converter.get_data_files(data_folder)
    first_file = data_files[0]
    print(f"First file: {first_file.name}")

    # Load bad channels
    bad_file = data_folder / "bad_channels.txt"
    bad_ch_ids = load_bad_ch(bad_file)

    # Process each shank
    for ish in shanks:
        nwb_path = data_folder / f"{session_description}sh{ish}.nwb"
        print(f"Creating NWB file {nwb_path.name}")

        # Create the NWB from first file
        good_ch = converter.initiate_nwb(
            first_file, nwb_path, ishank=ish,
            impedance_path=impedance_file,
            bad_ch_ids=bad_ch_ids,
            metadata={
                'device_type': device_type,
                'session_desc': session_description,
                'n_channels_per_shank': 32,
                'electrode_location': electrode_location,
                'exp_desc': exp_desc,
            }
        )

        if len(data_files) == 1:
            print(f"Only one file ({first_file.name}) found, skipping appending.")
            continue

        # Append the rest
        for f in data_files[1:]:
            print(f"Appending {f.name} → {nwb_path.name}")
            converter.append_nwb(
                nwb_path, f,
                channel_ids=good_ch,
                metadata={'device_type': device_type}
            )

    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()