import platform
from pathlib import Path

ANIMAL_ID = "CnL42"       # used for experiment data paths and CSV log
SORTOUT_ANIMAL_ID = "CnL42SG"  # used for sortout folder (may differ from ANIMAL_ID)
EXPERIMENT_DATE = "260313"

if platform.system() == "Darwin":
    _sortout_root = Path(r"/Volumes/xieluanlabs/xl_cl/sortout")
else:
    _sortout_root = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout")

SORTOUT_FOLDER = _sortout_root / SORTOUT_ANIMAL_ID / f"{SORTOUT_ANIMAL_ID}_20{EXPERIMENT_DATE}"

# Sample index in the concatenated sorter recording where the passive session starts.
# Set to 0 if the sorter ran only on the passive rec_folders (most common).
# Set to the actual offset if the sorter included recordings before the passive session.
PASSIVE_START = 0
# Sample index where the passive session ends (None = use last spike in sorter output).
PASSIVE_END   = None
