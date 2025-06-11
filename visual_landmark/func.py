from datetime import datetime, timedelta
from pathlib import Path

def generate_dates(start_date: str, end_date: str) -> list[str]:
    """
    Return a list of 'yymmdd' strings from start_date to end_date inclusive.
    """
    fmt = "%y%m%d"
    start = datetime.strptime(start_date, fmt)
    end   = datetime.strptime(end_date,   fmt)
    dates = []
    curr  = start
    while curr <= end:
        dates.append(curr.strftime(fmt))
        curr += timedelta(days=1)
    return dates


def find_session_folders(BASE_DIR: Path, date: str, animal_id: str) -> list[Path]:
    """
    Look for folders named <animal_id>_<date>_* under BASE_DIR/date/animal_id.
    """
    folder = BASE_DIR / date / animal_id
    if not folder.exists():
        return []
    return list(folder.glob(f"{animal_id}_{date}_*"))