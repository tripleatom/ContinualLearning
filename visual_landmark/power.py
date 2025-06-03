import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR     = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed")
DATE_START   = "250307"
DATE_END     = "250308"
ANIMAL_IDS   = ["CnL22"]
SHANK_IDS    = [0,1,2,3]
OUT_SUBDIR   = "sortout"  # relative to script folder

# ─── Helper Functions ──────────────────────────────────────────────────────────

def generate_dates(start_date: str, end_date: str) -> list[str]:
    """
    Given 'yymmdd' strings, return a list of all dates between them (inclusive)
    in the same 'yymmdd' format.
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


def find_session_folders(date: str, animal_id: str) -> list[Path]:
    """
    Look for folders named like <animal_id>_<date>_* under BASE_DIR/date/animal_id.
    Returns a list of Path objects.
    """
    pattern = BASE_DIR / date / animal_id
    if not pattern.exists():
        return []
    return list(pattern.glob(f"{animal_id}_{date}_*"))


def process_shank(session_folder: Path, session_name: str, shank_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    For a given session_folder and shank_id, load the corresponding .nwb file,
    compute the mean squared power per channel, sort by depth, and return
    (sorted_power, sorted_depths). Returns None if file is missing or error.
    """
    nwb_filename = session_folder / f"{session_name}sh{shank_id}.nwb"
    if not nwb_filename.exists():
        print(f"    [Warning] NWB file not found: {nwb_filename.name}")
        return None

    try:
        rec = se.NwbRecordingExtractor(str(nwb_filename))
        rec_filt = sp.bandpass_filter(rec, freq_min=500, freq_max=5000)

        # Extract all traces (n_channels x n_samples)
        traces = rec_filt.get_traces().astype(np.float32)

        # Compute mean squared power per channel
        power = np.mean(traces ** 2, axis=1)

        # Get channel locations and depths
        channel_ids = rec_filt.get_channel_ids()
        locations   = np.array([
            rec_filt.get_channel_property(ch, "location") 
            for ch in channel_ids
        ])
        depths = locations[:, 1]  # Y-axis = cortical depth

        # Sort by depth (ascending → top to bottom)
        order = np.argsort(depths)
        sorted_power  = power[order]
        sorted_depths = depths[order]

        return sorted_power, sorted_depths

    except Exception as e:
        print(f"    [Error] Failed processing {nwb_filename.name}: {e}")
        return None

    finally:
        # Close extractors if they were created
        try:
            rec_filt.close()
        except:
            pass
        try:
            rec.close()
        except:
            pass


def plot_and_save(powers_list: list[np.ndarray],
                  depths_list: list[np.ndarray],
                  shank_ids: list[int],
                  animal_id: str,
                  date: str,
                  session_id: str,
                  output_dir: Path) -> None:
    """
    Given lists of (power, depth) for each shank, plot them in a stacked figure
    and save both PDF and .npz data.
    """
    n_shanks = len(shank_ids)
    colors = ["black", "red", "blue", "green"]  # extend if needed

    # Create subplots
    fig, axes = plt.subplots(n_shanks, 1, figsize=(8, 4 * n_shanks), squeeze=False)
    axes = axes.flatten()

    # Plot each shank’s normalized power vs. depth
    for idx, (power, depths, shank_id) in enumerate(zip(powers_list, depths_list, shank_ids)):
        ax = axes[idx]
        norm_power = power / np.max(power)
        ax.plot(norm_power, depths, color=colors[idx % len(colors)])
        ax.invert_yaxis()  # depth increases downward
        ax.set_xlabel("normalized power\n(500 Hz – 5 kHz)", fontsize=10)
        ax.set_ylabel("depth (μm)",          fontsize=10)
        ax.set_title(f"Shank {shank_id}",     fontsize=12)

    # Main title
    fig.suptitle(
        f"High-Frequency Power by Cortical Depth\n"
        f"{animal_id} {date} {session_id}",
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save figure as PDF
    pdf_path = output_dir / f"power_{session_id}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Saved plot: {pdf_path.name}")

    # Save raw data as .npz
    npz_path = output_dir / f"power_{session_id}.npz"
    np.savez(
        npz_path,
        shank_ids=np.array(shank_ids),
        powers=[p / np.max(p) for p in powers_list],
        depths=depths_list,
        animal_id=animal_id,
        date=date,
        session_id=session_id
    )
    print(f"    → Saved data: {npz_path.name}")


# ─── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    # Determine script folder (for relative output)
    script_folder = Path(__file__).parent

    # Generate all dates in the range
    dates = generate_dates(DATE_START, DATE_END)

    for date in dates:
        for animal_id in ANIMAL_IDS:
            session_folders = find_session_folders(date, animal_id)

            if not session_folders:
                print(f"No sessions for {animal_id} on {date}")
                continue

            print(f"Found {len(session_folders)} session(s) for {animal_id} on {date}")

            for session_folder in session_folders:
                session_name = session_folder.name
                session_id   = session_name.split("_")[-1]
                print(f" Processing session: {session_name}")

                powers_list = []
                depths_list = []
                valid_shanks = []

                # Process each shank with progress print
                total_shanks = len(SHANK_IDS)
                for idx, shank_id in enumerate(SHANK_IDS):
                    print(f"  - [Shank {idx + 1}/{total_shanks}] Processing shank {shank_id}...")
                    result = process_shank(session_folder, session_name, shank_id)
                    if result is None:
                        continue
                    sorted_power, sorted_depths = result
                    powers_list.append(sorted_power)
                    depths_list.append(sorted_depths)
                    valid_shanks.append(shank_id)

                # If any shank data was valid, plot & save
                if powers_list:
                    out_dir = (
                        script_folder
                        / OUT_SUBDIR
                        / animal_id
                        / f"{date}_{session_id}"
                        / "VisualLandmark"
                    )
                    plot_and_save(
                        powers_list,
                        depths_list,
                        valid_shanks,
                        animal_id,
                        date,
                        session_id,
                        out_dir
                    )
                else:
                    print(f"  [Info] No valid shank data for {session_name}")

    print("All done.")


if __name__ == "__main__":
    main()
