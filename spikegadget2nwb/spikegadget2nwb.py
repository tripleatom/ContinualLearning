from read_spikegadget import get_ephys_folder
from pathlib import Path
from read_spikegadget import initiate_nwb, append_nwb
import os


subject_id = "CnL22"
exp_date = "20241030"
exp_time = "233456"
device_type = '4shank32'
ephys_folder = Path(r"D:\cl\ephys")
session_description = subject_id + '_' + exp_date + '_' + exp_time + '.rec'

rec_folder = ephys_folder / session_description

folders = sorted(rec_folder.glob('*.mountainsort'),
                    key=lambda x: os.path.getmtime(x))

# Extract group number and sort files
rec_files = [x for folder in folders for x in folder.glob('*group0.mda')]

rec_file0 = rec_files[0]

# impedance_file = subject_folder / \
#     Path(subject_id + '_' + exp_date + '.csv')

n_shank = 4
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
