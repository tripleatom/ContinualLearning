from read_spikegadget import get_ephys_folder
from pathlib import Path
from read_spikegadget import initiate_nwb, append_nwb
import os


subject_id = "CnL14"
exp_date = "20241004"
exp_time = "153555"
session_description = subject_id + '_' + exp_date + '_' + exp_time + '.rec'
# ephys_folder = get_ephys_folder()
ephys_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\rf_recons")
folder = ephys_folder / session_description

rec_files = sorted(folder.glob('*.rec'), key=lambda x: os.path.getmtime(x))
rec_file = rec_files[0]

# impedance_file = subject_folder / \
#     Path(subject_id + '_' + exp_date + '.csv')
nwb_path = folder / Path(session_description + '.nwb')

initiate_nwb(rec_file, nwb_path,
             metadata={'device_type': '4shank16',
                       "session_desc": session_description,
                       "n_channels": 128,
                       "electrode_location": "V1", })

if len(rec_files) == 1:
    print(f"Only one file {rec_file.name} found, exiting")
    exit()

for i, rec_file in enumerate(rec_files[1:]):
    print(f"Appending file {rec_file.name} to {nwb_path.name}")
    append_nwb(nwb_path, rec_file, metadata={'device_type': '4shank16'})
