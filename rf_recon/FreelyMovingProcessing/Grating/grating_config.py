import platform
from pathlib import Path

ANIMAL_ID = "CnL42"
EXPERIMENT_DATE = "260313"

if platform.system() == "Darwin":
    _sortout_root = Path(r"/Volumes/xieluanlabs/xl_cl/sortout")
else:
    _sortout_root = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout")

SORTOUT_FOLDER = _sortout_root / ANIMAL_ID / f"{ANIMAL_ID}_20{EXPERIMENT_DATE}"
