"""Put the sleep package's folders on sys.path so its flat imports keep working.

The scripts in sync/, lfp/, staging/ and spindles/ import each other (and the
shared sleep_pipeline_config / sleep_day_registry in this folder) by bare module
name, and reach the repo root for server_fallback / trodes_io.  Import this first:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import sleep_paths  # noqa: F401
"""

import sys
from pathlib import Path

SLEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SLEEP_DIR.parent

for _path in (SLEEP_DIR, REPO_ROOT,
              *sorted(d for d in SLEEP_DIR.iterdir()
                      if d.is_dir() and not d.name.startswith(("_", ".")))):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))
