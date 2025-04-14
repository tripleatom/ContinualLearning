## Installation

After cloning the repository, run the following command to install the package in editable mode:

```bash
pip install -e .
```

This ensures that local functions are properly installed and available.

## Folder Structure

- **`rec2nwb/`**  
  - `screen_bad_ch.py`  
    Contains functions for visualizing and removing bad channels from the recording.
    ![remove bad channels](images/bad_channel.png "remove bad channels")
  - `read_intan.py`, convert .rhd files to .nwb files.
  - `read_spikegadget.py`, convert .mda files from spikegadget to .nwb files.

- **`spikesorting/`**  
  Contains the spike sorting pipeline, including sorting with mountainsort, export to phy to manually curate the results.
