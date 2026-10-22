"""Numerical checks: CSD sign/units, gap handling, and known-power NWB."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from plot_layers import (csd_curvature, fill_rejected, find_outlier_channels, find_power_outliers,
                         floor_scaled_power, process_shank, read_timing,
                         passive_psd, smooth_laminar_csd)


class LayerTests(unittest.TestCase):
    def test_quadratic_csd_and_gap(self):
        z = np.array([0, 25, 50, 75, 125, 150, 175.])
        v = np.tile((z / 1000) ** 2, (4, 1))
        csd = csd_curvature(v, z)
        np.testing.assert_allclose(csd[:, [1, 2, 5]], -2, atol=1e-12)
        self.assertTrue(np.isnan(csd[:, [0, 3, 4, 6]]).all())

    def test_synthetic_nwb_power_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            fs = 12000
            t = np.arange(12 * fs) / fs
            signal = sum(np.sin(2 * np.pi * hz * t) for hz in (8, 200, 450, 800, 4000))
            values = signal[:, None] * np.array([1, 2, 3, 4])[None, :]
            with h5py.File(p / 'test.nwb', 'w') as f:
                es = f.create_group('acquisition/ElectricalSeries')
                data = es.create_dataset('data', data=values * 1e-6)
                data.attrs.update(conversion=1., offset=0., unit='volts')
                es.create_dataset('starting_time', data=0.).attrs['rate'] = fs
                table = f.create_group('electrodes')
                table['label'] = np.array([b'shank0:0', b'shank0:1', b'shank0:2', b'shank0:3'])
                table['rel_y'] = [0, 25, 50, 75]
                table['rel_x'] = [0, 0, 0, 0]
                es.create_dataset('electrodes', data=np.arange(4)).attrs['table'] = table.ref
            np.savez(p / 'dio.npz', rising_times=np.array([4, 7]) * fs, falling_times=np.array([5, 8]) * fs)
            args = SimpleNamespace(dio=p / 'dio.npz', dio_fs=fs, offset_sec=0,
                                   pre_sec=.5, bad_channels=p / 'bad.txt', batch_trials=2)
            on, off, duration = read_timing(args)
            epochs = {}
            r = process_shank(p / 'test.nwb', args, on, off, duration,
                              epoch_callback=lambda index, epoch: epochs.update({index: epoch.copy()}),
                              filter_bands=((.5, 100), (.5, 300), (None, 300)))
            self.assertEqual(sorted(epochs), [0, 1])
            np.testing.assert_allclose((epochs[0] + epochs[1]) / 2, r['lfp_uv'], atol=1e-12)
            spectrum = passive_psd(p / 'test.nwb', r['channel_labels'], (0, len(t)), batch_windows=3)
            np.testing.assert_allclose(spectrum['normalized_power'], ((np.arange(1, 5) / 4)[:, None] ** 2), rtol=1e-5)
            np.testing.assert_allclose(spectrum['band_power_uv2'][0], 1., rtol=.02)
            self.assertEqual(spectrum['psd_sample_rate_hz'], fs)
            # Wider temporal passband must retain substantially more 200-Hz activity.
            means = r['filter_lfp_uv']
            select = (r['time_sec'] >= 0) & (r['time_sec'] < .8)
            carrier = np.exp(-2j * np.pi * 200 * r['time_sec'][select])
            amplitude = abs(carrier @ means[:, select, 0].T) * 2 / select.sum()
            self.assertGreater(amplitude[1], 10 * amplitude[0])
            self.assertGreater(amplitude[2], 10 * amplitude[0])
            self.assertEqual(spectrum['psd_window_count'], 11)
            np.testing.assert_allclose(r['lfp_uv'][:1000].mean(axis=0), 0, atol=1e-10)
            self.assertAlmostEqual(r['time_sec'][1000], 0)
            self.assertAlmostEqual(r['time_sec'][-1], 1)
            np.testing.assert_allclose(r['csd_uv_per_mm2'][:, 1:-1], 0, atol=.05)
            # Strong idle-period signal must contribute to continuous PSD,
            # even though the stimulus epochs are confined to the quiet middle.
            with h5py.File(p / 'test.nwb', 'r+') as f:
                envelope = np.where((t < 3) | (t >= 9), 3., 1.)
                f['acquisition/ElectricalSeries/data'][:] = values * envelope[:, None] * 1e-6
            whole = passive_psd(p / 'test.nwb', r['channel_labels'], (0, len(t)), batch_windows=3)
            middle = passive_psd(p / 'test.nwb', r['channel_labels'], (4 * fs, 8 * fs), batch_windows=3)
            self.assertGreater(whole['band_density_uv2_per_hz'][0, 0], 3 * middle['band_density_uv2_per_hz'][0, 0])

    def test_power_outliers_and_floor_scaling(self):
        z = np.arange(12) * 25.
        power = 100 + 4. * np.arange(12)
        power[3] *= .2    # attenuated contact
        power[8] *= 3     # noisy contact: broadband excess
        power[0] *= 4     # beside a large unit: excess confined to the spike band
        power[11] *= 1.5  # hot, but within the ratio: kept
        noise = np.ones(12)
        noise[8] = 3
        # Without a noise estimate only attenuated contacts can be rejected.
        self.assertEqual(np.flatnonzero(find_power_outliers(power, z)).tolist(), [3])
        rejected = find_power_outliers(power, z, noise=noise)
        self.assertEqual(np.flatnonzero(rejected).tolist(), [3, 8])
        self.assertFalse(find_power_outliers(power, z, ratio=0, noise=noise).any())
        # Excluded contacts are neither neighbors nor returned.
        self.assertEqual(np.flatnonzero(find_power_outliers(power, z, exclude=np.arange(12) == 3, noise=noise)).tolist(), [8])
        profile, floor, top = floor_scaled_power(power, rejected)
        self.assertTrue(np.isnan(profile[[3, 8]]).all())
        good = power[~rejected]
        np.testing.assert_allclose([floor, top], np.percentile(good, [5, 95]))
        # A pedestal shared by every contact leaves the profile unchanged.
        np.testing.assert_allclose(floor_scaled_power(power + 1000, rejected)[0], profile, equal_nan=True)
        # The unit contact is kept and runs past 1 without rescaling the others.
        self.assertGreater(profile[0], 1)
        self.assertLess(np.nanmax(profile[1:]), 1)

    def test_band_power_is_median_over_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            fs = 12000
            t = np.arange(40 * fs) / fs
            values = np.sin(2 * np.pi * 1000 * t)[:, None] * np.ones((1, 2))
            values[(t >= 10) & (t < 12)] *= 30    # brief large event, 5% of the recording
            with h5py.File(Path(directory) / 'test.nwb', 'w') as f:
                es = f.create_group('acquisition/ElectricalSeries')
                data = es.create_dataset('data', data=values * 1e-6)
                data.attrs.update(conversion=1., offset=0., unit='volts')
                es.create_dataset('starting_time', data=0.).attrs['rate'] = fs
                table = f.create_group('electrodes')
                table['label'] = np.array([b'shank0:0', b'shank0:1'])
                es.create_dataset('electrodes', data=np.arange(2)).attrs['table'] = table.ref
            s = passive_psd(Path(directory) / 'test.nwb', ['shank0:0', 'shank0:1'], (0, len(t)), batch_windows=5)
            self.assertEqual(s['window_band_power_uv2'].shape, (39, 2, 1))
            np.testing.assert_allclose(s['band_power_uv2'], .5, rtol=.01)
            self.assertGreater(s['band_power_mean_uv2'][0, 0], 10 * s['band_power_uv2'][0, 0])
            np.testing.assert_allclose(s['window_band_power_uv2'].mean(axis=0), s['band_power_mean_uv2'], rtol=1e-4)

    def test_outlier_rejection_and_fill(self):
        z = np.arange(10) * 25.
        t = np.linspace(-.1, .4, 501)
        v = np.sin(2 * np.pi * 10 * t)[:, None] * (1 + z / 500)[None, :]
        v[:, 4] *= .4     # flat contact
        v[:, 7] *= 2      # oversized contact
        rejected = find_outlier_channels(v, t, z)
        self.assertEqual(np.flatnonzero(rejected).tolist(), [4, 7])
        self.assertFalse(find_outlier_channels(v, t, z, threshold=0).any())
        filled, used = fill_rejected(v, z, rejected)
        self.assertTrue(used.all())
        np.testing.assert_allclose(filled[:, 4], np.sin(2 * np.pi * 10 * t) * (1 + z[4] / 500), atol=1e-12)
        # Endpoint smoothing replicates values, so only the interior is exactly flat.
        np.testing.assert_allclose(smooth_laminar_csd(v, z, rejected)[:, 2:-2], 0, atol=1e-6)
        # Two adjacent rejects are dropped, not bridged.
        _, used = fill_rejected(v, z, np.isin(np.arange(10), [4, 5]))
        self.assertEqual(np.flatnonzero(~used).tolist(), [4, 5])

    def test_smoothed_csd_interior_and_no_gap_crossing(self):
        z = np.arange(9) * 25.
        v = np.tile((z / 1000) ** 2, (2, 1))
        np.testing.assert_allclose(smooth_laminar_csd(v, z)[:, 2:-2], -2, atol=1e-10)
        z = np.array([0, 25, 50, 100, 125, 150.])
        v = np.tile([0, 0, 0, 100, 100, 100.], (2, 1))
        smooth = smooth_laminar_csd(v, z)
        np.testing.assert_allclose(smooth[:, [1, 4]], 0, atol=1e-10)
        self.assertTrue(np.isnan(smooth[:, [0, 2, 3, 5]]).all())


if __name__ == '__main__':
    unittest.main()
