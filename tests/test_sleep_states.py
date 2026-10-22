import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SLEEP_APPLY_DIR = Path(__file__).resolve().parents[1] / "reactivation" / "sleep_apply"
if str(SLEEP_APPLY_DIR) not in sys.path:
    sys.path.insert(0, str(SLEEP_APPLY_DIR))

from sleep_states import (
    SleepBlock, annotate_decoder_results, interval_mask, load_mua_on_intervals, load_sleep_block,
)


class SleepReactivationTests(unittest.TestCase):
    def test_time_round_trip_and_out_of_range_rejection(self):
        block = SleepBlock("pre", 30000.0, 90000, 390000, 10.0)
        seconds = np.array([0.0, 1.25, 10.0])
        samples = block.local_sec_to_day_sample(seconds)
        np.testing.assert_allclose(block.day_sample_to_local_sec(samples), seconds)
        with self.assertRaisesRegex(ValueError, "Whole-day sample indices"):
            block.validate_local_seconds([90000])

    def test_interval_mask_uses_half_open_intervals(self):
        mask = interval_mask([0.0, 1.0, 1.999, 2.0], [[1.0, 2.0]])
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_mua_windows_are_mapped_from_day_samples(self):
        block = SleepBlock("pre", 10.0, 100, 200, 10.0)
        doc = {
            "sampling_frequency": 10.0,
            "rows": [
                {"shank": 0, "on_start_daysample": 110, "on_end_daysample": 116},
                {"shank": 1, "on_start_daysample": 112, "on_end_daysample": 118},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "on.pkl"
            with path.open("wb") as stream:
                pickle.dump(doc, stream)
            windows, metadata = load_mua_on_intervals(
                path, block, min_shank_fraction=1.0, max_duration_sec=None
            )
        np.testing.assert_allclose(windows, [[1.2, 1.6]])
        self.assertEqual(metadata["required_shanks_on"], 2)

    def test_decoder_annotation_marks_only_nrem_on_as_primary(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sleep_path = folder / "sleep.pkl"
            nrem_path = folder / "nrem.pkl"
            on_path = folder / "on.pkl"
            with sleep_path.open("wb") as stream:
                pickle.dump({
                    "metadata": {"sleep_id": "pre", "sampling_frequency": 10.0},
                    "window": {
                        "sleep_start_sample": 100,
                        "sleep_end_sample": 200,
                        "window_duration_sec": 10.0,
                    },
                    "spike_data": {"u": {"spike_times_sec": np.array([1.0, 5.0])}},
                }, stream)
            with nrem_path.open("wb") as stream:
                pickle.dump({"bout_intervals_s": [(2.0, 8.0)]}, stream)
            with on_path.open("wb") as stream:
                pickle.dump({
                    "sampling_frequency": 10.0,
                    "rows": [{
                        "shank": 0,
                        "on_start_daysample": 140,
                        "on_end_daysample": 160,
                    }],
                }, stream)
            results = {
                "sleep_pkl": str(sleep_path),
                "sleep_label": "pre",
                "sleep_centers_sec": np.array([1.0, 3.0, 5.0, 7.0]),
                "events": {
                    1: [{"time_sec": 5.0, "confidence": 0.9}],
                    -1: [{"time_sec": 3.0, "confidence": 0.8}],
                },
            }
            output = annotate_decoder_results(
                results, nrem_path=nrem_path, on_path=on_path,
                min_shank_fraction=1.0,
            )
        self.assertTrue(output["events"][1][0]["is_primary_reactivation"])
        self.assertFalse(output["events"][-1][0]["is_primary_reactivation"])
        self.assertEqual(output["state_annotation"]["event_counts"]["nrem_on"], 1)


if __name__ == "__main__":
    unittest.main()
