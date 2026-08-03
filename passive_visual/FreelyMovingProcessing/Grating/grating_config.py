import platform
from pathlib import Path

ANIMAL_ID = "CnL46"       # used for experiment data paths and CSV log
SORTOUT_ANIMAL_ID = "CnL46"  # used for sortout folder (may differ from ANIMAL_ID)
EXPERIMENT_DATE = "260726"

# Which lab drive holds this session's sortout: "xieluanlabs" or "xieluanlabs2".
SORTOUT_DRIVE = "xieluanlabs2"

_SORTOUT_ROOTS = {
    "xieluanlabs": {
        "Darwin": r"/Volumes/xieluanlabs/xl_cl/sortout",
        "other": r"\\10.129.151.108\xieluanlabs\xl_cl\sortout",
    },
    "xieluanlabs2": {
        "Darwin": r"/Volumes/xieluanlabs2/xl_cl/sortout",
        "other": r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout",
    },
}

_sortout_root = Path(
    _SORTOUT_ROOTS[SORTOUT_DRIVE]["Darwin" if platform.system() == "Darwin" else "other"]
)

SORTOUT_FOLDER = _sortout_root / SORTOUT_ANIMAL_ID / f"{SORTOUT_ANIMAL_ID}_20{EXPERIMENT_DATE}"

# Sample index in the concatenated sorter recording where the passive session starts.
# Set to 0 if the sorter ran only on the passive rec_folders (most common).
# Set to the actual offset if the sorter included recordings before the passive session.
PASSIVE_START = 62734834
# Sample index where the passive session ends (None = use last spike in sorter output).
PASSIVE_END   = None

# Optional per-task passive windows in the day-concatenated sorter space.
# Use this when each task's DIO.npz is zero-based within its own passive recording,
# but the curated_analyzer spike trains are from the full day concatenation.
#
# Entries match the semicolon-separated TaskFile order in experiment_log/<animal>.csv.
# Each entry can be a dict:
#   {"passive_start": 0, "passive_end": 123456}
# or a tuple:
#   (0, 123456)
#
# Leave as None to use PASSIVE_START/PASSIVE_END for every task.
PASSIVE_WINDOWS = None

SLEEP_BLOCKS = [
    ("pre", 181605860, 254815776),
    ("post", 308945341, 462415870),
]
