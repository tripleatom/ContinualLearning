import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
import spikeinterface as si
from pathlib import Path
import time
import matplotlib.pyplot as plt
from Timer import Timer

rec_folder = r"D:\cl\rf_reconstruction\head_fixed\CnL22_20241113_155342.rec"
animal_id = 'CnL22'
session_id = '20241113_155342'
ish = '3'
nwb_folder = Path(rec_folder) / f'{animal_id}_{session_id}.recsh{ish}.nwb'
out_folder = Path('sortout') / animal_id / session_id / ish
if not out_folder.exists():
    out_folder.mkdir(parents=True)


rec = se.NwbRecordingExtractor(nwb_folder)
print(rec)
# print(rec.get_channel_gains())
# print(rec.get_channel_locations())


#%%
rec_filt = sp.bandpass_filter(rec, freq_min=300, freq_max=6000, dtype='int32')
rec_cr = sp.common_reference(rec_filt, reference='global', operator='median')

rec_whiten = sp.whiten(rec_cr, dtype='float32')
rec_for_wvf_extraction = rec_filt

rec_preprocessed = rec_whiten

import numpy as np
start_times = np.arange(0, rec.get_total_duration(), 10)

ts_whiten_out_folder = out_folder / 'timeseries_whiten'
ts_cr_out_folder = out_folder / 'timeseries_cr'
if not ts_whiten_out_folder.exists():
    ts_whiten_out_folder.mkdir(parents=True)
if not ts_cr_out_folder.exists():
    ts_cr_out_folder.mkdir(parents=True)

for i, start_time in enumerate(start_times):
    time_range = [start_time, start_time + .2]
    sw.plot_traces(rec_whiten, backend='matplotlib', time_range=time_range,
                   order_channel_by_depth=True,)
    plt.savefig(ts_whiten_out_folder / f'timeseries_{i}.png')
    plt.close()
    sw.plot_traces(rec_cr, backend='matplotlib', time_range=time_range,
                   order_channel_by_depth=True,)
    plt.savefig(ts_cr_out_folder / f'timeseries_{i}.png')
    plt.close()


#%%
import mountainsort5 as ms5

threshold = 5.5
phase1_detect_time_radius_msec = .4
npca_ch=3
npca_sub=10

timer = Timer('ms5')
sorting_params = ms5.Scheme1SortingParameters(
    detect_time_radius_msec=phase1_detect_time_radius_msec, detect_threshold=threshold,
    npca_per_channel=npca_ch, npca_per_subdivision=npca_sub)
sorting = ms5.sorting_scheme1(
    recording=rec_preprocessed, sorting_parameters=sorting_params)
t_end = time.time()
timer.report()

import os
import json
current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
folder_name = 'sorting_results_' + current_time
sort_out_folder = out_folder / folder_name
if not os.path.exists(sort_out_folder):
    os.makedirs(sort_out_folder)
with open(sort_out_folder / 'sorting_params.json', 'w') as f:
    json.dump(sorting_params.__dict__, f)



print(sorting.unit_ids)

print(sorting.count_num_spikes_per_unit())

sorting.register_recording(rec_cr)
sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec_filt ,format='binary_folder', folder=sort_out_folder/'sorting_analyzer')


print(sorting_analyzer)

sorting_analyzer.compute("random_spikes")
sorting_analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
sorting_analyzer.compute(["templates", "quality_metrics","noise_levels", "amplitude_scalings", "template_metrics", "spike_amplitudes"])

for unit_id in sorting.get_unit_ids():
    sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
    plt.savefig(sort_out_folder / f'unit_summary_{unit_id}.png')
    plt.close()

for unit_id in sorting.unit_ids:
    sw.plot_unit_templates(sorting_analyzer, unit_ids=[unit_id])
    plt.savefig(sort_out_folder / f'unit_templates_{unit_id}.png')
    plt.close()