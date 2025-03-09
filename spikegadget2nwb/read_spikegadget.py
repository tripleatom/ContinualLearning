import spikeinterface.extractors as se
import neo.rawio
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
import re
from datetime import datetime
from uuid import uuid4
import pandas as pd
from pathlib import Path
from pynwb.ecephys import ElectricalSeries, TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO
import platform
import os


def get_rec_timestamp(rec_file: Path):
    match = re.search(r"_(\d{8})_(\d{6})", rec_file.name)

    date_str = match.group(1)
    time_str = match.group(2)

    # Combine date and time strings and parse into a datetime object
    datetime_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")

    return datetime_obj


def get_ch_index_on_shank(ish, device_type):
    # return channel index on a shank
    # return: index, xcoord, ycoord
    channel_map = pd.read_csv('mapping/' + device_type + '.csv')
    xcoord = np.array(channel_map['xcoord'], dtype=float)
    ycoord = np.array(channel_map['ycoord'], dtype=float)
    sh = np.array(channel_map['sh'], dtype=int)

    ch_index = np.where(sh == ish)[0]
    return ch_index, xcoord[ch_index], ycoord[ch_index]


def initiate_nwb(rec_file, nwb_path, metadata=None, ishank=0):
    session_start_time = get_rec_timestamp(rec_file)
    nwbhead_description = metadata.get("session_desc", "NWB file for RHS data")
    nwbhead_experimenter = metadata.get("experimenter", "Zhang, Xiaorong")
    nwbhead_lab = metadata.get("lab", "XL Lab")
    nwbhead_institution = metadata.get("institution", "Rice University")
    nwbhead_expdesc = metadata.get("exp_desc", "None")
    nwbhead_session_id = metadata.get("session_id", "None")
    electrode_location = metadata.get("electrode_location", None)
    device_type = metadata.get("device_type", "4shank16")

    nwbfile = NWBFile(
        session_description=nwbhead_description,
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        experimenter=[nwbhead_experimenter],
        lab=nwbhead_lab,
        institution=nwbhead_institution,
        experiment_description=nwbhead_expdesc,
        session_id=nwbhead_session_id,
    )

    # add device
    print("Adding device...")

    channel_index, xcoord, ycoord = get_ch_index_on_shank(ishank, device_type)

    PL_device = nwbfile.create_device(
        name="--", description="--", manufacturer="--"
    )
    nwbfile.add_electrode_column(
        name="label", description="label of electrode")

    electrode_counter = 0

    electrode_group = nwbfile.create_electrode_group(
        name="shank{}".format(ishank),
        description="electrode group for shank {}".format(ishank),
        device=PL_device,
        location=electrode_location,
    )

    for i, ich in enumerate(channel_index):
        # create an electrode group for this shank
        # add electrodes to the electrode table

        nwbfile.add_electrode(
            group=electrode_group,
            label="shank{}elec{}".format(ishank, ich),
            location=electrode_location,
            rel_x=float(xcoord[i]),
            rel_y=float(ycoord[i]),
        )
        electrode_counter += 1

    electrode_table_region = nwbfile.create_electrode_table_region(
        list(range(electrode_counter)), "all electrodes"
    )

    print("Adding electrical data...")
    # recording = se.read_spikegadgets(rec_file)
    mda_folder = os.path.dirname(rec_file)
    mda_file = os.path.basename(rec_file)
    recording = se.read_mda_recording(mda_folder, mda_file, params_fname=r"\\10.129.151.108\xieluanlabs\xl_cl\code\mapping\params.json",
                                      geom_fname=r"\\10.129.151.108\xieluanlabs\xl_cl\code\mapping\geom.csv")

    trace = recording.get_traces(channel_ids=channel_index)
    electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=H5DataIO(data=trace, maxshape=(None, np.shape(trace)[1])),
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=recording.get_sampling_frequency(),
        conversion=.195/1e6,
        offset=0./1e6,
    )

    nwbfile.add_acquisition(electrical_series)

    # todo: add digital input

    # if '5' in stream_ids:
    #     recording = se.read_intan(intan_file, stream_id='5')
    #     trace = recording.get_traces()
    #     digin_series = TimeSeries(
    #         name="DigInSeries",
    #         data=H5DataIO(data=trace, maxshape=(
    #             None, np.shape(trace)[1])),  # use H2DataIO to make the data chunked
    #         starting_time=0.0,
    #         rate=recording.get_sampling_frequency(),
    #         unit="bit"
    #     )

    #     nwbfile.add_acquisition(digin_series)

    print('writing nwb file...')
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)


def _append_nwb_dset(dset, data_to_append, append_axis):
    dset_shape = dset.shape
    dset_len = dset_shape[append_axis]
    app_len = data_to_append.shape[append_axis]

    dset_len += app_len

    my_slicer = [slice(None) for _ in range(len(dset_shape))]
    my_slicer[append_axis] = slice(-app_len, None)
    dset.resize(dset_len, axis=append_axis)
    dset[tuple(my_slicer)] = data_to_append


def append_nwb(nwb_path, rec_file, ishank=0, metadata=None):

    device_type = metadata.get("device_type", "4shank16")
    channel_index, _, _ = get_ch_index_on_shank(ishank, device_type)
    # stream_ids = get_stream_ids(append_intan_path)
    io = NWBHDF5IO(nwb_path, "a")
    nwb_obj_ = io.read()

    mda_folder = os.path.dirname(rec_file)
    mda_file = os.path.basename(rec_file)
    rec_ephys = se.read_mda_recording(
        mda_folder, mda_file, params_fname=r"\\10.129.151.108\xieluanlabs\xl_cl\code\mapping\params.json",
        geom_fname=r"\\10.129.151.108\xieluanlabs\xl_cl\code\mapping\geom.csv").get_traces(channel_ids=channel_index)
    _append_nwb_dset(
        nwb_obj_.acquisition['ElectricalSeries'].data, rec_ephys, 0)

    # if '5' in stream_ids:
    #     rec_digin = se.read_intan(
    #         append_intan_path, stream_id='5').get_traces()
    #     _append_nwb_dset(
    #         nwb_obj_.acquisition['DigInSeries'].data, rec_digin, 0)

    io.write(nwb_obj_)
    io.close()

def get_session_description(rec_folder):
    """
    Extract session description from recording folder path using different methods.
    
    Args:
        rec_folder (str): Path to recording folder
        
    Returns:
        str: Session description
    """
    # Method 1: Using regex to match the pattern before .rec
    pattern = r'[\\\/]([^\\\/]+)\.rec$'
    match = re.search(pattern, rec_folder)
    if match:
        return match.group(1)
    
    # Method 2: Using os.path to get basename and remove extension
    basename = os.path.basename(rec_folder)
    return os.path.splitext(basename)[0]


if __name__ == "__main__":
    device_type = '4shank16'
    n_shank = 4


    rec_folder = r"D:\cl\ephys\CnL22_20250216_212206.rec"
    session_description = get_session_description(rec_folder)

    if not rec_folder.exists():
        print(f"Folder {rec_folder} does not exist, exiting")
        exit()

    folders = sorted(rec_folder.glob('*.mountainsort'),
                     key=lambda x: os.path.getmtime(x))

    # Extract group number and sort files
    rec_files = [x for folder in folders for x in folder.glob('*group0.mda')]

    rec_file0 = rec_files[0]

    # impedance_file = subject_folder / \
    #     Path(subject_id + '_' + exp_date + '.csv')


    for ish in range(n_shank):
        nwb_path = rec_folder / Path(session_description + f'sh{ish}.nwb')
        print(f"Creating NWB file {nwb_path.name}")

        initiate_nwb(rec_file0, nwb_path, ishank=ish,
                     metadata={'device_type': device_type,
                               "session_desc": session_description,
                               "n_channels_per_shank": 32,
                               "electrode_location": "V1", })

        if len(rec_files) == 1:
            print(f"Only one file {rec_file0.name} found, exiting")
            continue

        for i, rec_file in enumerate(rec_files[1:]):
            print(f"Appending file {rec_file.name} to {nwb_path.name}")
            append_nwb(nwb_path, rec_file, ishank=ish,
                       metadata={'device_type': device_type})
