import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sleep' / 'spindles'))
from spindle_bosc import (CHI2_GEOMEAN, FREQUENCIES, combine_events, detect_bosc,
                          morlet_power, optimized_background, sustained_mask)


class BOSCTests(unittest.TestCase):
    def test_wavelet_crop_matches_direct_convolution(self):
        x = np.random.default_rng(1).normal(size=1000)
        fs, f, width = 250, 12, 6
        sigma = width / (2*np.pi*f)
        t = -3.6*sigma + np.arange(int(np.floor(7.2*sigma*fs))+1)/fs
        wave = np.exp(-t*t/(2*sigma*sigma))*np.exp(2j*np.pi*f*t)/np.sqrt(sigma*np.sqrt(np.pi))
        start = int(np.ceil(len(wave)/2))-1
        expected = abs(np.convolve(x, wave)[start:start+len(x)])**2
        np.testing.assert_allclose(morlet_power(x, fs, f), expected, atol=1e-10)

    def test_background_geometric_mean_threshold(self):
        rng = np.random.default_rng(7)
        arithmetic = 100 / FREQUENCIES
        power = rng.exponential(arithmetic[:, None], size=(21, 30000))
        bg, coeff, _, _ = optimized_background(power)
        estimated = chi2.ppf(.99, 2)*bg/CHI2_GEOMEAN
        np.testing.assert_allclose(estimated, -np.log(.01)*arithmetic, rtol=.05)
        self.assertAlmostEqual(coeff[0], -1, delta=.03)

    def test_cycle_rule_precedes_frequency_union(self):
        valid = np.ones(100, bool)
        a = np.zeros(100); b = a.copy()
        a[20:35] = 2; b[35:50] = 2
        da = sustained_mask(a, 1, 20, valid)
        db = sustained_mask(b, 1, 20, valid)
        self.assertFalse(da.any() or db.any())

    def test_invalid_gap_never_merged(self):
        valid = np.ones(500, bool); valid[220:225] = False
        d = np.zeros((7, 500), bool); d[:, 180:220] = True; d[:, 225:270] = True
        self.assertEqual(combine_events(d, d.astype(float)*2, valid, 100), [])

    def test_bursts_and_scale_invariance(self):
        fs = 125
        rng = np.random.default_rng(10)
        t = np.arange(120*fs)/fs
        # Stationary pink-like background with planted multi-cycle bursts.
        f = np.fft.rfftfreq(len(t), 1/fs)
        noise = np.fft.irfft(np.fft.rfft(rng.normal(size=len(t)))/np.sqrt(np.maximum(f, .1)), n=len(t))
        noise /= noise.std()
        x = noise.copy()
        for center in (30, 60, 90):
            x += 3*np.exp(-.5*((t-center)/.5)**2)*np.sin(2*np.pi*12*t)
        result = detect_bosc(x, fs, np.ones(len(t), bool))
        other = detect_bosc(x*100, fs, np.ones(len(t), bool))
        np.testing.assert_array_equal(result['detected'], other['detected'])
        for center in (30, 60, 90):
            self.assertTrue(any(e['onset_s'] < center < e['offset_s'] for e in result['events']))
        self.assertTrue(all(e['duration_s'] >= .35 for e in result['events']))
        self.assertFalse(result['valid'][0])

    def test_no_nrem_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            detect_bosc(np.ones(2000), 125, np.zeros(2000, bool))

    def test_seeded_bursts_versus_single_cycle_pulses(self):
        fs = 125
        t = np.arange(180*fs)/fs
        rng = np.random.default_rng(11)
        f = np.fft.rfftfreq(len(t), 1/fs)
        noise = np.fft.irfft(np.fft.rfft(rng.normal(size=len(t)))/np.sqrt(np.maximum(f, .1)), n=len(t))
        noise /= noise.std()
        centers = np.arange(20, 161, 20)
        for kind, expected in [('noise', 0), ('burst', 8), ('single_cycle', 0)]:
            x = noise.copy()
            for center in centers:
                u = t - center
                if kind == 'burst':
                    x += 3*np.exp(-.5*(u/.5)**2)*np.sin(2*np.pi*12*t)
                elif kind == 'single_cycle':
                    x += 3*np.sin(2*np.pi*12*u)*((u >= 0) & (u < 1/12))
            result = detect_bosc(x, fs, np.ones(len(x), bool))
            hits = sum(any(e['onset_s'] <= c+1/24 <= e['offset_s'] for e in result['events']) for c in centers)
            self.assertEqual(hits, expected, kind)
            self.assertEqual(len(result['events']), expected, kind)


if __name__ == '__main__':
    unittest.main()
