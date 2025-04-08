import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

import spikeinterface.extractors as se
import neo.rawio
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO


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
                 impedance_path: str = None, metadata: dict = None) -> None:
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
        impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy(
        )
        impedance_sh = impedance[channel_index]

    electrode_counter = 0
    for i, ch in enumerate(channel_index):
        imp_value = float(impedance_sh[i]) if impedance_sh is not None else 0.0
        nwbfile.add_electrode(
            group=electrode_group,
            label=f"shank{ishank}elec{ch}",
            location=electrode_location,
            rel_x=float(xcoord[i]),
            rel_y=float(ycoord[i]),
            imp=imp_value,
        )
        electrode_counter += 1

    electrode_table_region = nwbfile.create_electrode_table_region(
        list(range(electrode_counter)), "all electrodes"
    )

    stream_ids = get_stream_ids(intan_file)
    if '0' in stream_ids:
        print("Found amplifier channels...")
        channel_ids = [f"D-{i:03d}" for i in channel_index]
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

    # if '4' in stream_ids:
    #     #TODO: digital output is problematic, need to check, reading intan DIN with matlab now.
    #     print("Found digital input channels...")
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

    print("Writing NWB file...")
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)


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


def append_nwb(nwb_path: Path, append_intan_path: Path, ishank: int = 0,
               metadata: dict = None) -> None:
    """
    Append additional Intan data to an existing NWB file.
    """
    metadata = metadata or {}
    stream_ids = get_stream_ids(append_intan_path)
    with NWBHDF5IO(nwb_path, "a") as io:
        nwb_obj = io.read()

        device_type = metadata.get("device_type", "4shank16intan")
        channel_index, _, _ = get_ch_index_on_shank(ishank, device_type)
        channel_ids = [f"D-{i:03d}" for i in channel_index]

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


if __name__ == "__main__":
    # Define folder and file paths
    rhd_folder = Path(
        r'C:\STA_temp\250406\CnL22\CnL22_250406_002515')
    impedance_file = None
    device_type = "4shank16intan"
    n_shank = 4

    session_description = os.path.basename(rhd_folder)

    # Gather all .rhd files in the folder and sort them
    rhd_files = sorted(rhd_folder.glob('*.rhd'))
    if not rhd_files:
        raise FileNotFoundError("No .rhd files found in the specified folder.")
    first_rhd_file = rhd_files[0]

    # Process each shank: create an NWB file then append additional files if present
    for ish in range(n_shank):
        nwb_path = rhd_folder / f"{session_description}sh{ish}.nwb"
        initiate_nwb(first_rhd_file, nwb_path, ishank=ish,
                     impedance_path=impedance_file,
                     metadata={'device_type': device_type,
                               "session_desc": session_description,
                               "n_channels_per_shank": 32,
                               "electrode_location": "V1", 
                               "exp_desc": "grating",}
                     )

        if len(rhd_files) == 1:
            print(
                f"Only one file ({first_rhd_file.name}) found, skipping appending.")
            continue

        for rhd_file in rhd_files[1:]:
            print(f"Appending file {rhd_file.name} to {nwb_path.name}")
            append_nwb(nwb_path, rhd_file, ishank=ish,
                       metadata={'device_type': device_type})
