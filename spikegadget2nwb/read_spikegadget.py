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


def get_ephys_folder():

    pf = platform.system()

    if pf == 'Darwin':
        NotImplementedError('MacOS is not supported yet!')
    elif pf == 'Windows':
        ephys_folder = Path(r'\\10.129.151.108\xieluanlabs\xl_cl\ephys')
    elif pf == 'Linux':
        NotImplementedError('Linux is not supported yet!')
    if not ephys_folder.exists():
        raise ValueError('The folder does not exist!')

    return ephys_folder

def get_rec_timestamp(rec_file: Path):
    x = re.match(r'(.*)_(\d{8})_(\d{6}).rec', rec_file.name).groups()
    rec_datetime = datetime.strptime(x[1] + x[2], '%Y%m%d%H%M%S')
    return rec_datetime


def initiate_nwb(rec_file, nwb_path, metadata=None):
    print("Initiating NWB file...")
    session_start_time = get_rec_timestamp(rec_file)
    nwbhead_description = metadata.get("session_desc", "NWB file for RHS data")
    nwbhead_experimenter = metadata.get("experimenter", "Zhang, Xiaorong")
    nwbhead_lab = metadata.get("lab", "XL Lab")
    nwbhead_institution = metadata.get("institution", "Rice University")
    nwbhead_expdesc = metadata.get("exp_desc", "None")
    nwbhead_session_id = metadata.get("session_id", "None")

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
    device_type = metadata.get("device_type", "4shank16")
    channel_map = pd.read_csv('mapping/' + device_type + '.csv')

    xcoord = np.array(channel_map['xcoord'], dtype=float)
    ycoord = np.array(channel_map['ycoord'], dtype=float)
    sh = np.array(channel_map['sh'], dtype=int)

    # impedance_table = pd.read_csv(impedance_path)
    # impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy(
    # )
    # impedance_reordered = impedance[device_index]

    PL_device = nwbfile.create_device(
        name="--", description="--", manufacturer="--"
    )
    nwbfile.add_electrode_column(
        name="label", description="label of electrode")
    
    electrode_location = metadata.get("electrode_location", None)
    nchannels_per_shank = metadata.get("n_channels_per_shank", 32)
    nshanks = metadata.get("n_shanks", 4)
    electrode_counter = 0
    n_electrodes = nchannels_per_shank * nshanks

    electrode_groups = []
    for ish in range(nshanks):
        electrode_group = nwbfile.create_electrode_group(
            name="shank{}".format(ish),
            description="electrode group for shank {}".format(ish),
            device=PL_device,
            location=electrode_location,
        )
        electrode_groups.append(electrode_group)

    for ich in range(n_electrodes):
        # create an electrode group for this shank
        ishank = sh[ich]
        # add electrodes to the electrode table

        nwbfile.add_electrode(
            group=electrode_groups[ishank],
            label="shank{}elec{}".format(ishank, ich),
            location=electrode_location,
            rel_x=float(xcoord[ich]),
            rel_y=float(ycoord[ich]),
        )
        electrode_counter += 1

    electrode_table_region = nwbfile.create_electrode_table_region(
        list(range(electrode_counter)), "all electrodes"
    )

    print("Adding electrical data...")
    recording = se.read_spikegadgets(rec_file)
    trace = recording.get_traces()
    electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=H5DataIO(data=trace, maxshape=(None, np.shape(trace)[1])),
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=recording.get_sampling_frequency(),
        conversion=recording.get_channel_gains()[0]/1e6,
        offset=recording.get_channel_offsets()[0]/1e6,
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


def append_nwb(nwb_path, append_intan_path, metadata=None):
    # stream_ids = get_stream_ids(append_intan_path)
    io = NWBHDF5IO(nwb_path, "a")
    nwb_obj_ = io.read()

    rec_ephys = se.read_spikegadgets(append_intan_path).get_traces()
    _append_nwb_dset(
        nwb_obj_.acquisition['ElectricalSeries'].data, rec_ephys, 0)

    # if '5' in stream_ids:
    #     rec_digin = se.read_intan(
    #         append_intan_path, stream_id='5').get_traces()
    #     _append_nwb_dset(
    #         nwb_obj_.acquisition['DigInSeries'].data, rec_digin, 0)

    io.write(nwb_obj_)
    io.close()


if __name__ == "__main__":
    subject_id = "CnL20"
    exp_date = "20240811"
    exp_time = "144358"
    session_description = subject_id + '_' + exp_date + '_' + exp_time + '.rec'
    ephys_folder = get_ephys_folder()
    folder = ephys_folder / session_description

    rec_files = sorted(folder.glob('*.rec'))
    rec_file = rec_files[0]

    # impedance_file = subject_folder / \
    #     Path(subject_id + '_' + exp_date + '.csv')
    nwb_path = folder / Path(session_description + '.nwb')

    initiate_nwb(rec_file, nwb_path,
                 metadata={'device_type': '4shank16',
                           "session_desc": session_description,
                           'n_shanks': 4,
                           'n_channels_per_shank': 32,
                           "electrode_location": "V1", })

    # for i, rhs_file in enumerate(rhs_files[1:]):
    #     print(f"Appending file {rhs_file.name} to {nwb_path.name}")
    #     append_nwb(nwb_path, rhs_file, metadata={'device_type': 'pin32'})
