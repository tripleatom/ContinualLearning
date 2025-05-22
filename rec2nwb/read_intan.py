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

def get_stream_ids(file_path: str) -> any:
    """
    Get the stream ids from an Intan file.

    For RHD:
        0: 'RHD2000' amplifier channel
        1: 'RHD2000 auxiliary input channel'
        2: 'RHD2000 supply voltage channel'
        3: 'USB board ADC input channel'
        4: 'USB board digital input channel'
        5: 'USB board digital output channel'

    And for RHS:
        0: 'RHS2000 amplifier channel'
        3: 'USB board ADC input channel'
        4: 'USB board ADC output channel'
        5: 'USB board digital input channel'
        6: 'USB board digital output channel'
        10: 'DC Amplifier channel'
        11: 'Stim channel' (default will remove this channel)
    """
    file_path = str(file_path)
    reader = neo.rawio.IntanRawIO(filename=file_path)
    reader.parse_header()
    header = reader.header
    return header['signal_streams']['id']


def get_intan_timestamp(rhs_filename: Path) -> datetime:
    """
    Extract the recording start time from the filename.
    Expected filename format: <prefix>_yymmdd_HHMMSS.rh[s|d]
    """
    match = re.match(
        r"([a-zA-Z0-9_]+)_([0-9]+_[0-9]+).rh(?:s|d)", rhs_filename.name)
    if match:
        rec_datetimestr = match.group(2)  # yymmdd_HHMMSS
        return datetime.strptime(rec_datetimestr, "%y%m%d_%H%M%S")
    raise ValueError("Filename does not match expected pattern.")


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


def initiate_nwb(intan_file: Path, nwb_path: Path, ishank: int = 0,
                 impedance_path: str = None, bad_ch_ids: list = None,
                 metadata: dict = None) -> None:
    """
    Create and write an NWB file from an Intan recording.
    """
    metadata = metadata or {}
    print("Initiating NWB file...")
    session_start_time = get_intan_timestamp(intan_file)
    nwb_description = metadata.get("session_desc", "NWB file for RHD data")
    experimenter = metadata.get("experimenter", "Zhang, Xiaorong")
    lab = metadata.get("lab", "XL Lab")
    institution = metadata.get("institution", "Rice University")
    exp_desc = metadata.get("exp_desc", "None")
    session_id = metadata.get("session_id", "None")
    electrode_location = metadata.get("electrode_location", None)
    device_type = metadata.get("device_type", "4shank16intan")

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
    channel_index, xcoord, ycoord = get_ch_index_on_shank(ishank, device_type)
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

    # Add impedance data if available
    impedance_sh = None
    if impedance_path is not None:
        impedance_table = pd.read_csv(impedance_path)
        impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()
        impedance_sh = impedance[channel_index]
        channel_name = impedance_table['Channel Name'].to_numpy()
        channel_name_sh = channel_name[channel_index]

    electrode_df = pd.DataFrame({
        'channel_name': channel_name_sh,
        'impedance': impedance_sh,
        'x': xcoord,
        'y': ycoord
    })

    # Remove bad channels from the DataFrame
    electrode_df = electrode_df[~electrode_df['channel_name'].isin(bad_ch_ids)]

    n_electrodes = len(electrode_df)
    print(f"Number of good electrodes: {n_electrodes}")
    # Loop through each row in electrode_df and add an electrode entry to the NWB file.
    for idx, row in electrode_df.iterrows():
        nwbfile.add_electrode(
            group=electrode_group,
            label=f"shank{ishank}:{row['channel_name']}",
            location=electrode_location,
            rel_x=float(row['x']),
            rel_y=float(row['y']),
            imp=float(row['impedance']),
        )

    electrode_table_region = nwbfile.create_electrode_table_region(
        list(range(n_electrodes)), "all electrodes"
    )

    stream_ids = get_stream_ids(intan_file)
    if '0' in stream_ids:
        print("Found amplifier channels...")
        channel_ids = electrode_df["channel_name"].tolist()
        recording = se.read_intan(intan_file, stream_id='0')

        trace = recording.get_traces(channel_ids=channel_ids)
        electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=H5DataIO(data=trace, maxshape=(None, trace.shape[1])),
            electrodes=electrode_table_region,
            starting_time=0.0,
            rate=recording.get_sampling_frequency(),
            conversion=recording.get_channel_gains()[
                0] / 1e6,  # convert uV to V
            offset=recording.get_channel_offsets()[0] / 1e6,
        )
        nwbfile.add_acquisition(electrical_series)

    if '4' in stream_ids:
        print("Found digital input channels...")
    #     #TODO: digital output is problematic, need to check, reading intan DIN with matlab now.
    #     # Note: The stream id used for digital input differs between initiate and append.
    #     recording = se.read_intan(intan_file, stream_id='4')
    #     trace = recording.get_traces()
    #     digin_series = TimeSeries(
    #         name="DigInSeries",
    #         data=H5DataIO(data=trace, maxshape=(None, trace.shape[1])),
    #         starting_time=0.0,
    #         rate=recording.get_sampling_frequency(),
    #         unit="bit"
    #     )
    #     nwbfile.add_acquisition(digin_series)
        pass

    print("Writing NWB file...")
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    return channel_ids


def _append_nwb_dset(dset, data_to_append, append_axis: int) -> None:
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


def append_nwb(nwb_path: Path, append_intan_path: Path, channel_ids: list = None,
               metadata: dict = None) -> None:
    """
    Append additional Intan data to an existing NWB file.
    """
    metadata = metadata or {}
    stream_ids = get_stream_ids(append_intan_path)
    with NWBHDF5IO(nwb_path, "a") as io:
        nwb_obj = io.read()
        rec_ephys = se.read_intan(append_intan_path, stream_id='0') \
                      .get_traces(channel_ids=channel_ids)
        _append_nwb_dset(
            nwb_obj.acquisition['ElectricalSeries'].data, rec_ephys, 0)

        # if '4' in stream_ids:
        #     rec_digin = se.read_intan(append_intan_path, stream_id='4') \
        #                   .get_traces()
        #     _append_nwb_dset(
        #         nwb_obj.acquisition['DigInSeries'].data, rec_digin, 0)

        io.write(nwb_obj)

def load_bad_ch(bad_file: Path) -> list:
    """
    Load bad channels from a file.
    """
    if not bad_file.exists():
        print(f"No bad channels file found at {bad_file}. Please run the screening script.")
        sys.exit(1)
    with open(bad_file, "r") as f:
        bad_channels = [line.strip() for line in f.readlines()]
    return bad_channels



if __name__ == "__main__":
    # Inputs
    rhd_folder = Path(input("Please enter the full path to the RHD folder: ").strip().strip("'").strip('"'))
    impedance_file = Path(input("Please enter the full path to the impedance file: ").strip().strip("'").strip('"'))
    electrode_location = input("Please enter the electrode location: ").strip()
    exp_desc = input("Please enter the experiment description: ").strip() or "None"
    animal_id = rhd_folder.parent.name
    device_type = get_or_set_device_type(animal_id)
    raw = input("Please enter the shank numbers (e.g. 0,1,2,3 or [0,1,2,3]): ")
    # extract all integer substrings
    shanks = [int(x) for x in re.findall(r'\d+', raw)]
    print(shanks)
    
    session_description = rhd_folder.name

    # Gather all .rhd or .rhs files
    # pick first .rhd or .rhs, exclude macOS system files like ._*
    data_files = sorted(
        p for p in rhd_folder.iterdir()
        if p.suffix.lower() in ('.rhd', '.rhs') and not p.name.startswith("._"))

    if not data_files:
        raise FileNotFoundError("No .rhd or .rhs files found in the specified folder.")
    first_file = data_files[0]

    # Load bad channels
    bad_file = rhd_folder / "bad_channels.txt"
    bad_ch_ids = load_bad_ch(bad_file)

    # Process each shank
    for ish in shanks:
        nwb_path = rhd_folder / f"{session_description}sh{ish}.nwb"

        # create the NWB from first file
        good_ch = initiate_nwb(
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

        # append the rest
        for f in data_files[1:]:
            print(f"Appending {f.name} → {nwb_path.name}")
            append_nwb(
                nwb_path, f,
                channel_ids=good_ch,
                metadata={'device_type': device_type}
            )
