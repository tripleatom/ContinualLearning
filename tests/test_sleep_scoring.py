import sys
import unittest
from pathlib import Path

import numpy as np


SLEEP_DIR = Path(__file__).resolve().parents[1] / "sleep" / "staging"
if str(SLEEP_DIR) not in sys.path:
    sys.path.insert(0, str(SLEEP_DIR))

from score_sleep_states import bridge_gaps, hysteresis_mask, remove_short_bouts


class SleepScoringTests(unittest.TestCase):
    def test_hysteresis_uses_separate_entry_and_exit_thresholds(self):
        score = np.array([0.2, 0.7, 0.5, 0.39, 0.8])
        np.testing.assert_array_equal(
            hysteresis_mask(score, enter=0.6, exit=0.4),
            [False, True, True, False, True],
        )

    def test_gap_bridging_and_minimum_bout_are_independent(self):
        mask = np.array([True, True, False, True, True, False, False, True])
        bridged = bridge_gaps(mask, max_gap_bins=1)
        np.testing.assert_array_equal(
            bridged, [True, True, True, True, True, False, False, True]
        )
        np.testing.assert_array_equal(
            remove_short_bouts(bridged, min_bins=2),
            [True, True, True, True, True, False, False, False],
        )


if __name__ == "__main__":
    unittest.main()
