import numpy as np
import pandas as pd

device_type = '4shank16'

ch_map = pd.read_csv('mapping/' + device_type + '.csv')

chanMap = np.array(ch_map['spikegadget'], dtype=int)
xc = np.array(ch_map['xcoord'], dtype=float)
yc = np.array(ch_map['ycoord'], dtype=float)
kcoords = np.array(ch_map['sh'], dtype=int)
n_chan = len(chanMap)

probe = {
    'chanMap': chanMap,
    'xc': xc,
    'yc': yc,
    'kcoords': kcoords,
    'n_chan': n_chan
}

from kilosort.io import save_probe

save_probe(probe, 'mapping/' + device_type + '.json')