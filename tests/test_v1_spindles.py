import sys
import unittest
import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sleep' / 'spindles'))
from detect_v1_spindles import analyze, detect_events, nrem_samples, process_file


class SpindleTests(unittest.TestCase):
    def test_epoch_support_and_artifact_exclusion(self):
        states = dict(epoch_times=[1., 2., 5.], epoch_sec=1.,
                      states=['NREM'] * 3, valid_mask=[True] * 3,
                      artifact_mask=[False, True, False])
        mask = nrem_samples(states, 60, 10)
        expected = np.zeros(60, bool)
        expected[5:15] = True
        expected[45:55] = True
        np.testing.assert_array_equal(mask, expected)

    def test_events_do_not_bridge_invalid_gap(self):
        fs = 100
        amp = np.ones(1000)
        amp[300:350] = 10
        amp[355:400] = 10
        valid = np.ones(1000, bool)
        valid[350:355] = False
        events, _ = detect_events(amp, valid, fs)
        self.assertEqual(events, [])  # both events are truncated at the artifact
        events, _ = detect_events(amp, np.ones(1000, bool), fs)
        self.assertEqual(len(events), 1)

    def test_laminar_bursts_and_dead_channel(self):
        fs = 200
        t = np.arange(60 * fs) / fs
        rng = np.random.default_rng(4)
        traces = rng.normal(0, .3, (len(t), 8))
        for center in (10, 20, 30, 40, 50):
            envelope = np.exp(-.5 * ((t - center) / .22) ** 2)
            traces[:, 2:5] += (4 * envelope * np.sin(2 * np.pi * 12 * t))[:, None]
        traces[:, 7] = 0
        result = analyze(traces, fs, np.arange(8) * 25, np.ones(len(t), bool))
        self.assertIn(result['best_index'], [2, 3, 4])
        self.assertFalse(result['good_channel'][7])
        self.assertEqual(len(result['events']), 5)
        np.testing.assert_allclose([e['peak_s'] for e in result['events']], [10, 20, 30, 40, 50], atol=.1)
        self.assertEqual(result['triggered_event_count'], 5)

    def test_no_nrem_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient valid NREM'):
            analyze(np.zeros((100, 4)), 200, np.arange(4), np.zeros(100, bool))

    def test_large_physical_gaps_prevent_region(self):
        rng = np.random.default_rng(0)
        result = analyze(rng.normal(size=(4000, 4)), 200, np.arange(4) * 100,
                         np.ones(4000, bool))
        self.assertEqual(result['best_index'], -1)
        self.assertEqual(result['events'], [])

    def test_file_pipeline_writes_review_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            lfp = folder / 'example_pre_sh5_lfp_traces.npz'
            state = folder / 'example_pre_sleep_periods.pkl'
            rng = np.random.default_rng(5)
            np.savez(lfp, traces=rng.normal(size=(6000, 5)), sampling_rate=200.,
                     channel_ids=np.arange(5), ycoord=np.arange(5) * 25.,
                     sleep_start_sample=30000, original_fs=30000., shank=5,
                     session_name='example', sleep_session='pre')
            with state.open('wb') as stream:
                pickle.dump(dict(session='example_pre', sleep_start_sample=30000,
                                 epoch_times=np.arange(.5, 30, 1), epoch_sec=1.,
                                 states=np.repeat('NREM', 30), valid_mask=np.ones(30, bool),
                                 artifact_mask=np.zeros(30, bool)), stream)
            args = argparse.Namespace(shank=5, exclude_channels=[], depth_range=None,
                                      peak_threshold=3., boundary_threshold=1.5,
                                      min_duration=.3, max_duration=3., merge_gap=.2,
                                      max_depth_gap_um=50., min_region_channels=3,
                                      out_dir=folder)
            process_file(lfp, state, args)
            stem = folder / 'example_pre_sh5_v1_spindles'
            for ext in ('.csv', '.npz', '.png', '.json'):
                self.assertTrue(stem.with_suffix(ext).is_file())
            metadata = json.loads(stem.with_suffix('.json').read_text())
            self.assertEqual(metadata['band_hz'], [9, 17])


if __name__ == '__main__':
    unittest.main()
