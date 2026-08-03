import errno
import re
import numpy as np
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from sleep_pipeline_config import (
    rec_folder,
    session_name,
    nwb_session_name,
    shanks,
    sleep_sessions,
    active_sleep_sessions,
    preproc_params,
    DOWNSAMPLE_METHOD,
    N_JOBS,
    CHUNK_DURATION,
    resolve_output_folder,
    mirror_on_backup_server,
)


def resolve_nwb_path(folder, session_prefix, shank):
    """Find a shank NWB while allowing either YYMMDD or YYYYMMDD dates."""
    folder = Path(folder)
    suffix = f"sh{shank}.nwb"
    prefixes = [session_prefix]

    # NWB exports are inconsistent about using YYMMDD versus YYYYMMDD.
    date_match = re.fullmatch(r"(.+_)(\d{6}|\d{8})", session_prefix)
    if date_match:
        stem, date = date_match.groups()
        alternate_date = f"20{date}" if len(date) == 6 else date[2:]
        prefixes.append(f"{stem}{alternate_date}")

    attempted = []
    for prefix in prefixes:
        candidate = folder / f"{prefix}{suffix}"
        attempted.append(candidate)
        if candidate.is_file():
            if prefix != session_prefix:
                print(f"Configured NWB name not found; using date-format match: {candidate}")
            return candidate

    # Last resort: accept a unique file for the same animal/session prefix and
    # shank. Never silently choose when multiple recordings match.
    animal_prefix = session_prefix.rsplit("_", 1)[0]
    matches = sorted(
        path for path in folder.glob(f"{animal_prefix}_*{suffix}")
        if path.is_file()
    )
    if len(matches) == 1:
        print(f"Configured NWB name not found; using unique shank match: {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        choices = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple NWB files match {animal_prefix}_*{suffix}; "
            f"cannot choose safely:\n  {choices}"
        )

    tried = "\n  ".join(str(path) for path in attempted)
    raise FileNotFoundError(
        f"Could not find an NWB file for shank {shank}. Tried:\n  {tried}\n"
        f"Also searched for: {folder / f'{animal_prefix}_*{suffix}'}"
    )


def build_lfp_recording(rec):
    """CAR -> downsample -> LFP band. Returns the lazy LFP recording."""
    # 1. CAR
    print("\n1. Applying CAR...")
    rec_car = spre.common_reference(
        rec,
        reference=preproc_params['reference'],
        operator=preproc_params['operator'],
    )

    if DOWNSAMPLE_METHOD == "resample":
        # 2. RESAMPLE (anti-alias lowpass included by SpikeInterface)
        print("2. Downsampling (resample)...")
        rec_ds = spre.resample(rec_car, preproc_params['target_fs'])
        # 3. LFP BANDPASS
        print("3. Bandpass filtering (LFP band)...")
        rec_lfp = spre.bandpass_filter(
            rec_ds,
            freq_min=preproc_params['lfp_min'],
            freq_max=preproc_params['lfp_max'],
        )

    elif DOWNSAMPLE_METHOD == "decimate":
        orig_fs = rec_car.get_sampling_frequency()
        factor = orig_fs / preproc_params['target_fs']
        decimation_factor = int(round(factor))
        if abs(factor - decimation_factor) > 1e-6:
            raise ValueError(
                f"decimate requires integer orig_fs/target_fs, got "
                f"{orig_fs}/{preproc_params['target_fs']} = {factor:.4f}. "
                f"Use DOWNSAMPLE_METHOD='resample' instead."
            )
        new_nyquist = (orig_fs / decimation_factor) / 2
        if preproc_params['lfp_max'] >= new_nyquist:
            raise ValueError(
                f"lfp_max ({preproc_params['lfp_max']} Hz) must be < new Nyquist "
                f"({new_nyquist} Hz) so the bandpass also serves as anti-alias."
            )
        # 2. LFP BANDPASS at full rate -> doubles as the anti-alias filter
        print("2. Bandpass filtering (LFP band, also anti-alias)...")
        rec_band = spre.bandpass_filter(
            rec_car,
            freq_min=preproc_params['lfp_min'],
            freq_max=preproc_params['lfp_max'],
        )
        # 3. DECIMATE (integer factor)
        print(f"3. Decimating by {decimation_factor}x...")
        rec_lfp = spre.decimate(rec_band, decimation_factor)

    else:
        raise ValueError(f"Unknown DOWNSAMPLE_METHOD: {DOWNSAMPLE_METHOD!r}")

    return rec_lfp


def process_shank(ish, session_key, session_cfg, job_kwargs):
    print("\n" + "=" * 75)
    print(f"PROCESSING SHANK {ish}  [session={session_key}]")
    print("=" * 75 + "\n")

    # Load NWB
    rec_path = resolve_nwb_path(rec_folder, nwb_session_name, ish)
    print(f"Loading NWB: {rec_path}")
    rec = se.read_nwb_recording(rec_path)

    # Basic info
    orig_fs = rec.get_sampling_frequency()
    orig_dur = rec.get_total_duration()
    n_channels = rec.get_num_channels()
    print(f"Original FS: {orig_fs} Hz")
    print(f"Duration:    {orig_dur:.2f} sec")
    print(f"Channels:    {n_channels}")

    # =====================================================
    # RESTRICT TO SLEEP PERIOD (in original-FS samples)
    # =====================================================
    # Slice before any preprocessing so CAR/downsample/bandpass only see
    # the sleep window. Bounds are in original-FS sample indices.
    total_frames = rec.get_num_frames()
    start_sample = session_cfg['start_sample']
    end_sample = session_cfg['end_sample']
    slice_start = 0 if start_sample is None else int(start_sample)
    slice_end = total_frames if end_sample is None else int(end_sample)
    if not (0 <= slice_start < slice_end <= total_frames):
        raise ValueError(
            f"Invalid sleep window [{slice_start}, {slice_end}) for recording "
            f"with {total_frames} samples"
        )
    rec = rec.frame_slice(start_frame=slice_start, end_frame=slice_end)
    print(
        f"Sleep window:  samples [{slice_start}, {slice_end}) "
        f"= {slice_start / orig_fs:.2f}–{slice_end / orig_fs:.2f} s "
        f"({(slice_end - slice_start) / orig_fs:.2f} s)"
    )

    # =====================================================
    # PREPROCESSING PIPELINE (lazy)
    # =====================================================
    rec_lfp = build_lfp_recording(rec)

    # =====================================================
    # CHANNEL ORDERING BY DEPTH
    # =====================================================
    channel_ids = rec_lfp.get_channel_ids()
    chan_locs = rec_lfp.get_channel_locations()
    xcoord = chan_locs[:, 0]
    ycoord = chan_locs[:, 1]
    depth_order = np.argsort(ycoord)
    sorted_channels = channel_ids[depth_order]
    print(f"\nSorted channels by depth ({len(sorted_channels)} channels)")

    # =====================================================
    # EXTRACT LFP TRACES (parallel)
    # =====================================================
    # save(format="memory") runs CAR/downsample/bandpass across all cores via
    # job_kwargs and returns an in-memory recording; get_traces() is then instant.
    fs = rec_lfp.get_sampling_frequency()
    duration = rec_lfp.get_total_duration()
    print("\nExtracting LFP traces (parallel)...")
    print(f"  Target sampling rate: {fs} Hz")
    print(f"  Duration: {duration:.2f} sec")

    rec_mem = rec_lfp.save(format="memory", **job_kwargs)

    # Full array in native channel order, then reorder columns by depth.
    traces = rec_mem.get_traces()                      # (n_samples, n_channels)
    traces = np.ascontiguousarray(traces[:, depth_order], dtype="float32")

    print(f"\n  Final LFP shape: {traces.shape} (time x channels)")

    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    out_dir = resolve_output_folder(Path(rec_path).parent / "low_freq")
    out_file = out_dir / f"{session_name}{session_cfg['suffix']}_sh{ish}_lfp_traces.npz"

    savez_kwargs = dict(
        traces=traces,
        sampling_rate=fs,
        channel_ids=sorted_channels,
        channel_locations=chan_locs[depth_order],
        xcoord=xcoord[depth_order],
        ycoord=ycoord[depth_order],
        depth_order=depth_order,
        duration=duration,
        n_channels=len(sorted_channels),
        n_timepoints=traces.shape[0],
        original_fs=orig_fs,
        original_duration=orig_dur,
        sleep_start_sample=slice_start,
        sleep_end_sample=slice_end,
        downsample_method=DOWNSAMPLE_METHOD,
        session_name=session_name,
        sleep_session=session_key,
        shank=ish,
    )

    print(f"\nSaving → {out_file}")
    try:
        np.savez(out_file, **savez_kwargs)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        backup_dir = mirror_on_backup_server(out_dir)
        if backup_dir is None:
            raise
        backup_dir.mkdir(parents=True, exist_ok=True)
        out_file = backup_dir / out_file.name
        print(f"Out of space while saving - retrying on backup server: {out_file}")
        np.savez(out_file, **savez_kwargs)
    print("Done!")


# =====================================================
# ENTRY POINT
#   The __main__ guard is REQUIRED on Windows: n_jobs > 1 spawns worker
#   processes that re-import this module, and without the guard that would
#   recursively launch the whole pipeline.
# =====================================================
if __name__ == "__main__":
    job_kwargs = dict(n_jobs=N_JOBS, chunk_duration=CHUNK_DURATION, progress_bar=True)

    sessions_to_run = active_sleep_sessions(sleep_sessions)
    if not sessions_to_run:
        print("No active sleep sessions (pre/post both start=end=None) - nothing to do.")

    for session_key, session_cfg in sessions_to_run.items():
        print("\n" + "#" * 75)
        print(f"SLEEP SESSION: {session_key}")
        print("#" * 75)
        for ish in shanks:
            process_shank(ish, session_key, session_cfg, job_kwargs)

    print("\n" + "#" * 75)
    print("ALL SHANKS PROCESSED")
    print("#" * 75)
