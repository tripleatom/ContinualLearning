from grating_utils import parse_grating_experiment
import numpy as np
from pathlib import Path
from datetime import datetime
from trodes_io.DIO import get_dio_folders, concatenate_din_data
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons


def _stamp_figure(fig, info, y=0.01):
    """Embed a small reproducibility stamp (params/paths/timestamp) at the bottom of a figure."""
    fig.text(0.01, y, info, ha='left', va='bottom', fontsize=6, color='0.35')


def load_recording_level_dio_edges(rec_folders, channel_id=3):
    """
    Load cleaned recording-level DIO files made by dio_single.py.

    dio_single.py saves one zero-aligned file beside each .rec folder:
        <rec_stem>_DIO.npz

    This function concatenates those cleaned rising edges into the same
    recording-order sample coordinate system used when raw DIO folders are
    concatenated. It uses raw DIO only to recover each recording's duration, not
    to recover stimulus edges.
    """
    all_rising = []
    offset = 0
    loaded_paths = []

    for rec_folder in rec_folders:
        rec_stem = rec_folder.name[:-4] if rec_folder.name.endswith('.rec') else rec_folder.name
        cleaned_path = rec_folder.parent / f"{rec_stem}_DIO.npz"
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Recording-level DIO not found: {cleaned_path}")

        cleaned = np.load(cleaned_path)
        rising = cleaned['rising_times'].ravel().astype(float)
        if len(rising) == 0:
            raise ValueError(f"No rising_times found in {cleaned_path}")

        dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
        raw_time, _ = concatenate_din_data(dio_folders, channel_id)
        raw_time = raw_time.ravel().astype(float)
        raw_duration = int(raw_time[-1] - raw_time[0] + 1) if len(raw_time) else int(rising[-1] + 1)

        all_rising.append(rising + offset)
        loaded_paths.append(cleaned_path)
        print(f"Loaded cleaned rec DIO: {cleaned_path.name} "
              f"({len(rising)} rising edges, offset={offset})")
        offset += raw_duration

    return np.concatenate(all_rising), loaded_paths


def _local_good_avg(edges, center_idx, lo, hi, fallback, window=30):
    """
    Return the mean of good gap sizes within `window` positions of center_idx.
    A gap is good if it falls inside [lo, hi].
    Falls back to `fallback` if no good gaps are found nearby.
    """
    start = max(0, center_idx - window)
    end   = min(len(edges) - 1, center_idx + window)
    good  = [edges[j + 1] - edges[j] for j in range(start, end)
             if lo <= edges[j + 1] - edges[j] <= hi]
    return int(round(np.mean(good))) if good else int(fallback)


def fix_rising_edges(rising_times, n_trials, trial_duration_samples, tolerance=0.02,
                     pre_screen_fraction=0.5, max_gap_samples=None):
    """
    Automatically detect and fix abnormal rising edges.

    Handles two classes of anomalies:
      - Glitch (extra edge): interval between consecutive risings is much shorter
        than expected (< (1-tolerance) * trial_duration). The edge that leaves the
        worse next-gap is removed.
      - Missing trial: interval is much longer than expected
        (> (1+tolerance) * trial_duration). Missing edges are interpolated at
        regular multiples of trial_duration_samples.

    A pre-screening pass runs first: any run of edges whose consecutive spacing
    is < pre_screen_fraction * trial_duration_samples is collapsed to just the
    first edge of that run.

    Parameters
    ----------
    rising_times : np.ndarray
        Raw rising edge sample indices.
    n_trials : int
        Expected number of trials from the task file.
    trial_duration_samples : int
        Expected inter-trial interval in samples (stimulus + ITI).
    tolerance : float
        Fractional tolerance around the expected interval (default 0.02 = ±2%,
        i.e. ±60 ms for a 3 s trial). Intervals outside [1-tol, 1+tol]*expected
        are treated as glitches or missing trials.
    pre_screen_fraction : float
        Edges whose spacing is below this fraction of the expected interval are
        considered "obviously too close" and collapsed (default 0.5 = 50% of ITI).

    Returns
    -------
    fixed : np.ndarray
        Corrected rising edge sample indices.
    screened : np.ndarray
        Rising edges after pre-screening but before glitch/missing-trial fixes.
    log : list of str
        Description of each fix applied.
    """
    expected = trial_duration_samples
    min_interval = (1 - tolerance) * expected
    max_interval = (1 + tolerance) * expected

    # ── Pre-screening pass ────────────────────────────────────────────────────
    # Collapse runs of edges that are obviously too close (< pre_screen_fraction
    # of the expected interval) into just the first edge of each run.
    pre_screen_threshold = pre_screen_fraction * expected
    raw_list = rising_times.ravel().tolist()
    screened = []
    log = []
    i = 0
    while i < len(raw_list):
        screened.append(raw_list[i])
        j = i + 1
        while j < len(raw_list) and (raw_list[j] - raw_list[j - 1]) < pre_screen_threshold:
            log.append(f"Pre-screen: removed edge at sample {raw_list[j]} "
                       f"(spacing {(raw_list[j] - raw_list[j-1])/expected:.3f}x expected, "
                       f"kept first at {raw_list[i]})")
            j += 1
        i = j
    if len(screened) < len(raw_list):
        log.append(f"Pre-screen: {len(raw_list) - len(screened)} edge(s) removed, "
                   f"{len(screened)} remain")

    fixed = screened
    i = 0

    while i < len(fixed) - 1:
        interval = fixed[i + 1] - fixed[i]

        if interval < min_interval:
            # Glitch: two edges too close together.
            # Keep the one whose next gap is closer to expected.
            next_if_remove_i1 = fixed[i + 2] - fixed[i] if i + 2 < len(fixed) else np.inf
            next_if_remove_i0 = fixed[i + 2] - fixed[i + 1] if i + 2 < len(fixed) else np.inf
            if abs(next_if_remove_i1 - expected) <= abs(next_if_remove_i0 - expected):
                log.append(f"Removed glitch at index {i+1} (sample {fixed[i+1]}, "
                           f"interval {interval/expected:.2f}x expected)")
                fixed.pop(i + 1)
            else:
                log.append(f"Removed glitch at index {i} (sample {fixed[i]}, "
                           f"interval {interval/expected:.2f}x expected)")
                fixed.pop(i)
            # Re-check same position after removal

        elif interval > max_interval:
            if max_gap_samples is not None and interval >= max_gap_samples:
                # Large gap treated as a task/session boundary — do not fill.
                log.append(f"Skipped large gap at index {i} "
                           f"(interval {interval/expected:.2f}x expected, "
                           f"{interval/30000:.1f}s ≥ max_gap threshold)")
                i += 1
            else:
                # If it's the very first interval, the first edge is likely a
                # stray pre-trial pulse — remove it instead of inserting dots.
                if i == 0:
                    log.append(f"Removed leading edge at sample {fixed[0]} "
                               f"(first interval {interval/expected:.2f}x expected)")
                    fixed.pop(0)
                    # i stays at 0 to re-check the new first interval
                else:
                    # Missing trial(s): gap is ~N times the expected interval.
                    n_missing = round(interval / expected) - 1
                    step = _local_good_avg(fixed, i, min_interval, max_interval, expected)
                    for k in range(1, n_missing + 1):
                        inserted = int(fixed[i] + k * step)
                        fixed.insert(i + k, inserted)
                        log.append(f"Inserted missing edge at sample {inserted} "
                                   f"(gap was {interval/expected:.2f}x expected, "
                                   f"step={step} from local avg)")
                    i += n_missing + 1

        else:
            i += 1

    if len(fixed) != n_trials:
        log.append(f"WARNING: {len(fixed)} edges after fix, but {n_trials} trials expected")
    else:
        log.append(f"OK: {len(fixed)} edges match {n_trials} expected trials")

    return np.array(fixed), np.array(screened), log


def segment_rising_edges_by_task(rising_times, task_n_trials_list, task_trial_durations_samples,
                                  buffer_samples=None, fs=30000):
    """
    Split a continuous array of rising edges into per-task segments.

    The boundary after task k is placed at:
        first_edge_of_task_k  +  n_trials_k * trial_duration_k  +  buffer

    Everything before that boundary belongs to task k; the remainder is passed
    to the next task.

    Parameters
    ----------
    rising_times : np.ndarray
        All rising edge sample indices from the full recording.
    task_n_trials_list : list of int
        Expected number of trials for each task, in time order.
    task_trial_durations_samples : list of int
        Expected trial duration (samples) for each task, in time order.
    buffer_samples : int or None
        Extra samples added after the last expected trial of each task before
        the cut. Defaults to 10 * fs (10 seconds).
    fs : int
        Sampling rate, used only to set the default buffer (default 30000).

    Returns
    -------
    segments : list of np.ndarray
        One rising-edge array per task.
    """
    n_tasks = len(task_n_trials_list)
    if n_tasks == 1:
        return [rising_times]

    if buffer_samples is None:
        buffer_samples = 10 * fs

    segments = []
    remaining = rising_times.copy()

    for k in range(n_tasks - 1):
        if len(remaining) == 0:
            segments.append(remaining)
            continue

        task_start = remaining[0]
        cutoff = task_start + task_n_trials_list[k] * task_trial_durations_samples[k] + buffer_samples

        split_idx = np.searchsorted(remaining, cutoff, side='right')
        segments.append(remaining[:split_idx])
        remaining = remaining[split_idx:]

        print(f"  Task {k+1}: cutoff at sample {cutoff} "
              f"({cutoff/fs:.1f}s from task start), "
              f"{len(segments[-1])} edges assigned "
              f"(expected {task_n_trials_list[k]})")

    # Last task gets everything remaining
    segments.append(remaining)
    print(f"  Task {n_tasks}: {len(remaining)} edges assigned "
          f"(expected {task_n_trials_list[-1]})")

    return segments


def segment_by_large_gaps(rising_times, gap_threshold_samples, fs=30000):
    """
    Segment rising edges into groups separated by gaps larger than gap_threshold_samples.

    Parameters
    ----------
    rising_times : np.ndarray
        Rising edge sample indices (pre-screened or raw).
    gap_threshold_samples : int
        Gaps strictly larger than this value mark a segment boundary.
    fs : int
        Sampling rate, used only for printing.

    Returns
    -------
    segments : list of np.ndarray
        One array of rising edge indices per segment.
    """
    if len(rising_times) < 2:
        return [rising_times]

    diffs = np.diff(rising_times)
    split_positions = np.where(diffs > gap_threshold_samples)[0] + 1
    segments = np.split(rising_times, split_positions)

    print(f"Found {len(segments)} segment(s) separated by gaps > {gap_threshold_samples/fs:.1f}s:")
    for k, seg in enumerate(segments):
        duration_s = (seg[-1] - seg[0]) / fs if len(seg) > 1 else 0.0
        print(f"  Segment {k+1}: {len(seg)} edges, "
              f"start={seg[0]/fs:.1f}s, end={seg[-1]/fs:.1f}s, "
              f"span={duration_s:.1f}s")
    return segments


def plot_rising_edge_segments(rising_times, segments, task_metas, fs=30000,
                               gap_threshold_s=10.0, save_path=None):
    """
    Plot all rising edges (ITI vs edge index) colored by segment, with a bar chart
    comparing detected vs expected edge counts per segment.

    Parameters
    ----------
    rising_times : np.ndarray
        All rising edge sample indices (after global pre-screen).
    segments : list of np.ndarray
        Output of segment_by_large_gaps.
    task_metas : list of dict
        Each dict has keys 'n_trials', 'trial_duration', 'path'.
    fs : int
        Sampling rate.
    gap_threshold_s : float
        Threshold used for segmentation (drawn as a reference line).
    save_path : Path or None
        If given, save the figure there.
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(f"DIO segments — {len(rising_times)} total edges, "
                 f"{len(segments)} segment(s), "
                 f"threshold={gap_threshold_s:.0f}s",
                 fontsize=12, fontweight='bold')

    # ── Top: ITI vs global edge index, colored by segment ──────────────────────
    offset = 0
    for k, seg in enumerate(segments):
        color = colors[k % len(colors)]
        n = len(seg)
        if n > 1:
            diffs = np.diff(seg) / fs
            idx = np.arange(offset + 1, offset + n)
            axes[0].plot(idx, diffs, color=color, lw=0.8, marker='.', ms=3,
                         label=f'Seg {k+1} ({n} edges)')
        # Segment boundary line
        if k < len(segments) - 1:
            axes[0].axvline(offset + n - 0.5, color='gray', linestyle=':', lw=1.2)
        # Annotate count
        mid_idx = offset + n / 2
        axes[0].annotate(f'{n}', xy=(mid_idx, gap_threshold_s * 0.92),
                         ha='center', fontsize=9, fontweight='bold',
                         color=color)
        offset += n

    # Expected ITI lines per task
    for k, m in enumerate(task_metas):
        axes[0].axhline(m['trial_duration'], color=colors[k % len(colors)],
                        linestyle='--', lw=1.2, alpha=0.7,
                        label=f'Task {k+1} exp. ITI ({m["trial_duration"]:.2f}s)')

    axes[0].axhline(gap_threshold_s, color='red', linestyle='-', lw=1.5, alpha=0.6,
                    label=f'Seg threshold ({gap_threshold_s:.0f}s)')
    axes[0].set_ylabel('Inter-trial interval (s)', fontsize=10)
    axes[0].set_xlabel('Edge index', fontsize=10)
    axes[0].set_title('ITI vs edge index — colored by segment', fontsize=10)
    axes[0].legend(fontsize=8, loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.3)

    # ── Bottom: bar chart — detected vs expected per segment ───────────────────
    seg_counts = [len(s) for s in segments]
    exp_counts = [m['n_trials'] for m in task_metas]
    # Pad shorter list with zeros
    n_bars = max(len(segments), len(task_metas))
    seg_counts += [0] * (n_bars - len(seg_counts))
    exp_counts += [0] * (n_bars - len(exp_counts))

    x = np.arange(n_bars)
    w = 0.35
    bars_det = axes[1].bar(x - w / 2, seg_counts, w,
                           color=[colors[k % len(colors)] for k in range(n_bars)],
                           label='Detected')
    bars_exp = axes[1].bar(x + w / 2, exp_counts, w,
                           color='lightgray', edgecolor='black', linewidth=0.8,
                           label='Expected')

    for bar, cnt in zip(bars_det, seg_counts):
        if cnt:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         str(cnt), ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, cnt in zip(bars_exp, exp_counts):
        if cnt:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         str(cnt), ha='center', va='bottom', fontsize=9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Seg {k+1}' for k in range(n_bars)], fontsize=9)
    axes[1].set_ylabel('Edge count', fontsize=10)
    axes[1].set_title('Detected vs expected edges per segment', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stamp = (
        f"Generated {timestamp} | script={Path(__file__).name}\n"
        f"gap_threshold={gap_threshold_s:.0f}s, fs={fs} | "
        f"segments={len(segments)} (sizes={[len(s) for s in segments]}) | "
        f"expected_per_task={[m['n_trials'] for m in task_metas]}\n"
        f"save_path={save_path}"
    )
    _stamp_figure(fig, stamp)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Segment plot saved to {save_path}")
    plt.show()


def fix_residual_jitter(rising_edges, trial_duration_samples, tolerance=0.02, group_window=10):
    """
    Fix residual timing errors after fix_rising_edges.

    Bad gaps (outside [1-tol, 1+tol] * expected, and < 1.5 * expected) that fall
    within group_window edge-indices of each other are treated as a cluster and
    fixed together by interpolating evenly between the good anchor edges on either
    side of the cluster.  Isolated bad gaps are fixed individually by snapping to
    the correct neighbour.

    Parameters
    ----------
    rising_edges : np.ndarray
        Edge indices (output of fix_rising_edges).
    trial_duration_samples : int
        Expected inter-trial interval in samples.
    tolerance : float
        Fractional tolerance (default 0.02 = ±2 %).
    group_window : int
        Bad gap indices within this many positions of each other are merged into
        one cluster (default 10).

    Returns
    -------
    fixed : np.ndarray
        Corrected edge indices.
    log : list of str
        Description of each fix applied.
    """
    edges = list(rising_edges)
    expected = trial_duration_samples
    lo = (1 - tolerance) * expected
    hi = (1 + tolerance) * expected
    log = []

    def _bad_gap_indices(e):
        """Return indices i where gap e[i+1]-e[i] is outside tolerance but < 1.5x."""
        bad = []
        for i in range(len(e) - 1):
            g = e[i + 1] - e[i]
            if not (lo <= g <= hi) and g <= 1.5 * expected:
                bad.append(i)
        return bad

    def _group_indices(bad, window):
        """Merge bad gap indices within `window` of each other into clusters."""
        if not bad:
            return []
        groups, cur = [], [bad[0]]
        for idx in bad[1:]:
            if idx - cur[-1] <= window:
                cur.append(idx)
            else:
                groups.append(cur)
                cur = [idx]
        groups.append(cur)
        return groups

    # ── Cluster fix pass ───────────────────────────────────────────────────────
    # Repeat until no new clusters form (a cluster fix may reveal new bad gaps).
    for _ in range(20):
        bad = _bad_gap_indices(edges)
        if not bad:
            break
        groups = _group_indices(bad, group_window)
        any_fixed = False

        for grp in groups:
            if len(grp) == 1:
                continue  # handled by single-point pass below

            # Edge indices spanning the cluster:
            #   left anchor  = grp[0]          (edge before first bad gap)
            #   right anchor = grp[-1] + 1     (edge after last bad gap)
            la = grp[0]
            ra = grp[-1] + 1
            if ra >= len(edges):
                log.append(f"  Cluster [{la}:{ra}]: right anchor out of bounds, skipped")
                continue

            # Verify anchors are reliable
            left_ok  = (la == 0 or lo <= edges[la] - edges[la - 1] <= hi)
            right_ok = (ra + 1 >= len(edges) or lo <= edges[ra + 1] - edges[ra] <= hi)

            if not (left_ok or right_ok):
                log.append(f"  Cluster [{la}:{ra}]: neither anchor reliable, skipped")
                continue

            n_gaps   = ra - la          # number of intervals to fill
            span     = edges[ra] - edges[la]
            expected_span = n_gaps * expected

            if abs(span - expected_span) / expected_span > 0.15:
                log.append(f"  Cluster [{la}:{ra}]: span {span/expected:.3f}x expected "
                           f"({n_gaps} gaps), too far off — skipped")
                continue

            # Interpolate evenly between the two anchor edges using local avg step
            step = _local_good_avg(edges, la, lo, hi, expected)
            for k in range(1, n_gaps):
                old = edges[la + k]
                new = int(edges[la] + k * step)
                edges[la + k] = new
                log.append(f"  Cluster [{la}:{ra}] index {la+k}: {old} → {new}  "
                           f"({n_gaps}-gap cluster, span={span/expected:.3f}x, "
                           f"step={step} from local avg)")
            any_fixed = True

        if not any_fixed:
            break

    # ── Single-point fix pass ──────────────────────────────────────────────────
    i = 0
    while i < len(edges) - 1:
        gap = edges[i + 1] - edges[i]

        if lo <= gap <= hi or gap > 1.5 * expected:
            i += 1
            continue

        prev_good = i > 0 and lo <= edges[i] - edges[i - 1] <= hi
        next_good = i + 2 < len(edges) and lo <= edges[i + 2] - edges[i + 1] <= hi

        if prev_good:
            step = _local_good_avg(edges, i, lo, hi, expected)
            new = int(edges[i] + step)
            right_neighbor_ok = (i + 2 >= len(edges) or lo <= edges[i + 2] - new <= hi)
            if right_neighbor_ok:
                bad = edges[i + 1]
                edges[i + 1] = new
                log.append(f"  index {i+1}: {bad} → {new}  "
                           f"(gap {gap/expected:.3f}x, anchored to left edge {edges[i]}, "
                           f"step={step} from local avg)")
                i += 1
            else:
                log.append(f"  index {i+1}: gap {gap/expected:.3f}x — "
                           f"proposed snap {new} would break right gap "
                           f"({(edges[i+2]-new)/expected:.3f}x), skipped")
                i += 1

        elif next_good:
            step = _local_good_avg(edges, i + 1, lo, hi, expected)
            new = int(edges[i + 1] - step)
            left_neighbor_ok = (i == 0 or lo <= new - edges[i - 1] <= hi)
            if left_neighbor_ok:
                bad = edges[i]
                edges[i] = new
                log.append(f"  index {i}: {bad} → {new}  "
                           f"(gap {gap/expected:.3f}x, anchored to right edge {edges[i + 1]}, "
                           f"step={step} from local avg)")
                i = max(0, i - 1)
            else:
                log.append(f"  index {i}: gap {gap/expected:.3f}x — "
                           f"proposed snap {new} would break left gap "
                           f"({(new-edges[i-1])/expected:.3f}x), skipped")
                i += 1

        else:
            log.append(f"  index {i}: gap {gap/expected:.3f}x — no correct neighbour, skipped")
            i += 1

    return np.array(edges), log


def parse_txt_trial_times(task_file_path):
    """
    Extract per-trial stimulus Start/End wall-clock times from a grating .txt
    task file, in seconds relative to the first trial's Start.

    Reads the Start/End columns (HH:MM:SS.mmm) of the TRIAL DATA table already
    parsed into a DataFrame by parse_grating_experiment.

    Returns
    -------
    start_sec, end_sec : np.ndarray
        Per-trial Start / End times in seconds, zeroed to trial 1's Start.
    """
    task = parse_grating_experiment(task_file_path)
    df = task.get('trial_data')
    if df is None or 'Start' not in df or 'End' not in df:
        raise ValueError(f"No parseable trial Start/End table in {task_file_path}")

    fmt = "%H:%M:%S.%f"
    starts = [datetime.strptime(s, fmt) for s in df['Start']]
    ends   = [datetime.strptime(s, fmt) for s in df['End']]
    t0 = starts[0]
    start_sec = np.array([(t - t0).total_seconds() for t in starts])
    end_sec   = np.array([(t - t0).total_seconds() for t in ends])
    return start_sec, end_sec


def align_edges_to_txt(rising_edges, txt_start_sec, fs=30000):
    """
    Locate which contiguous block of txt trials a (possibly incomplete) rising-
    edge train corresponds to, and fit the linear map txt-seconds -> ephys-sample.

    Slides the rising-edge inter-trial-interval (ITI) sequence over every start
    offset in the txt ITI sequence and keeps the lowest mean-squared-error match.
    Comparing ITIs (rather than absolute times) makes this invariant to the
    constant clock offset between the DIO reference and the txt wall clock.

    Parameters
    ----------
    rising_edges : np.ndarray
        Rising-edge sample indices (0-based) of the recorded (incomplete) DIO.
    txt_start_sec : np.ndarray
        Per-trial Start times in seconds (from parse_txt_trial_times).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict with keys:
        offset      : int        - 0-based txt trial index where rising_edges[0] lands
        a, b        : float      - linear fit  rising_sec = a * txt_start_sec + b
        mse         : float      - mean squared ITI error at the best offset
        residual_ms : np.ndarray - per-matched-trial fit residual, in ms
    """
    rising_sec = np.asarray(rising_edges, dtype=float) / fs
    n_edges = len(rising_sec)
    n_txt = len(txt_start_sec)
    if n_edges < 2:
        raise ValueError("Need >= 2 rising edges to align against the txt log")
    if n_edges > n_txt:
        raise ValueError(f"More rising edges ({n_edges}) than txt trials ({n_txt})")

    iti_edges = np.diff(rising_sec)
    iti_txt   = np.diff(txt_start_sec)

    best_offset, best_err = 0, np.inf
    for offset in range(0, n_txt - n_edges + 1):
        window = iti_txt[offset:offset + n_edges - 1]
        err = np.mean((window - iti_edges) ** 2)
        if err < best_err:
            best_offset, best_err = offset, err

    matched = txt_start_sec[best_offset:best_offset + n_edges]
    A = np.vstack([matched, np.ones_like(matched)]).T
    (a, b), *_ = np.linalg.lstsq(A, rising_sec, rcond=None)
    residual_ms = (rising_sec - (a * matched + b)) * 1000
    return {'offset': best_offset, 'a': float(a), 'b': float(b),
            'mse': float(best_err), 'residual_ms': residual_ms}


def fix_missing_trials_from_txt(rising_edges, task_file_path, stimulus_duration, fs=30000):
    """
    Complete a partial rising-edge train using the wall-clock trial Start times
    logged in a grating .txt file.

    Use when the recorded DIO only captured a contiguous subset of the task's
    trials - e.g. the photodiode/trigger line dropped out mid-session, or the
    recording started or stopped partway through the task - but the txt log
    recorded every trial. A linear txt-seconds -> ephys-sample map is fit on the
    trials that WERE recorded (see align_edges_to_txt), then sample indices for
    the txt trials missing before/after the recorded block are extrapolated from
    it. Every recorded edge is preserved unchanged; only missing trials are
    filled in.

    Assumes the recorded edges form ONE contiguous block of txt trials (missing
    trials are all at the start and/or end). Trials missing in the middle of the
    recorded block are better handled by fix_rising_edges' interpolation; a large
    residual std here is the tell-tale that this contiguity assumption is broken.

    Falling edges are set to rising + stimulus_duration for every trial, matching
    the 'fixed' convention used elsewhere in this module.

    Parameters
    ----------
    rising_edges : np.ndarray
        Recorded (partial) rising-edge sample indices, ideally glitch-screened.
    task_file_path : str or Path
        Grating .txt task file.
    stimulus_duration : float
        Stimulus ON duration in seconds.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict with keys:
        rising_times, falling_times : np.ndarray  (length == n txt trials)
        is_reconstructed            : np.ndarray of bool  (True for filled-in trials)
        offset, n_real, n_reconstructed, fit
    """
    rising_edges = np.asarray(rising_edges, dtype=np.int64)
    txt_start_sec, _ = parse_txt_trial_times(task_file_path)
    n_txt = len(txt_start_sec)
    n_real = len(rising_edges)

    fit = align_edges_to_txt(rising_edges, txt_start_sec, fs=fs)
    offset, a, b = fit['offset'], fit['a'], fit['b']

    new_rising = np.empty(n_txt, dtype=np.int64)
    is_recon = np.ones(n_txt, dtype=bool)

    # Recorded trials: preserved exactly.
    new_rising[offset:offset + n_real] = rising_edges
    is_recon[offset:offset + n_real] = False

    # Missing trials before and/or after the recorded block: extrapolate.
    missing_idx = np.r_[np.arange(0, offset), np.arange(offset + n_real, n_txt)]
    for i in missing_idx:
        new_rising[i] = int(round((a * txt_start_sec[i] + b) * fs))

    falling = new_rising + int(round(stimulus_duration * fs))

    if n_real >= 1 and not np.all(np.diff(new_rising) > 0):
        print("  WARNING: reconstructed rising edges are not strictly increasing "
              "- check txt/DIO alignment.")

    res = fit['residual_ms']
    print(f"  txt-referring fix: {n_real} recorded trial(s) matched to txt trials "
          f"{offset + 1}-{offset + n_real} "
          f"(a={a:.7f}, b={b:.3f}s, residual std={res.std():.1f} ms, "
          f"max|res|={np.abs(res).max():.1f} ms); "
          f"reconstructed {len(missing_idx)} missing trial(s) "
          f"({offset} before, {n_txt - offset - n_real} after).")

    return {'rising_times': new_rising, 'falling_times': falling,
            'is_reconstructed': is_recon, 'offset': offset,
            'n_real': n_real, 'n_reconstructed': len(missing_idx), 'fit': fit}


def process_task(task_file_path, rising_segment, fs=30000):
    """
    Fix DIO edges, plot diagnostics, and save results for a single task.

    Parameters
    ----------
    task_file_path : Path
        Path to the .txt task file.
    rising_segment : np.ndarray
        Rising edge sample indices that belong to this task.
    fs : int
        Sampling rate (default 30000).
    """
    task_file = parse_grating_experiment(task_file_path)
    task_id = task_file_path.stem
    folder_path = task_file_path.parent

    stimulus_duration = float(task_file['parameters']['stimulus_duration'].rstrip('s'))
    ITI_duration = float(task_file['parameters']['iti_duration'].rstrip('s'))
    n_repeats = task_file['parameters']['total_trials']
    trial_duration = stimulus_duration + ITI_duration

    print(f"\n  Task: {task_id}")
    print(f"  stimulus={stimulus_duration}s  ITI={ITI_duration}s  "
          f"trial_duration={trial_duration}s  n_trials={n_repeats}")
    print(f"  Segment edges: {len(rising_segment)} (expected {n_repeats})")

    trial_duration_samples = int(trial_duration * fs)

    # Step 1: remove glitch bursts and insert missing trials
    rising_fixed, rising_screened, fix_log = fix_rising_edges(
        rising_segment, n_repeats, trial_duration_samples)

    # Step 2: fix residual single-edge jitter (e.g. 4 s or 2.75 s gaps)
    rising_fixed, jitter_log = fix_residual_jitter(rising_fixed, trial_duration_samples)

    n_removed_prescreen = len(rising_segment) - len(rising_screened)
    all_log = fix_log + jitter_log
    print(f"  Pre-screen: {len(rising_segment)} → {len(rising_screened)} "
          f"({n_removed_prescreen} removed)")
    print("  Fix log:")
    for entry in all_log:
        print(f"    {entry}")

    falling_fixed = rising_fixed + int(stimulus_duration * fs)

    # Step 3: txt-referring reconstruction of any trials missing from the DIO.
    # Uses the .txt trial Start times to fill in a contiguous block of trials
    # missing at the start/end of the recording (e.g. trigger dropout), while
    # preserving every recorded edge. Aligned from the glitch-screened edges.
    txt_result = None
    try:
        txt_result = fix_missing_trials_from_txt(
            rising_screened, task_file_path, stimulus_duration, fs=fs)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"  txt-referring fix unavailable: {e}")

    # Diagnostics plot — RAW / PRE-SCREENED / FIXED (+ TXT-REFERRING if available)
    n_panels = 4 if txt_result is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=False)
    fig.suptitle(f"{task_id}\n"
                 f"raw={len(rising_segment)}  screened={len(rising_screened)}  "
                 f"fixed={len(rising_fixed)}"
                 + (f"  txt_fixed={len(txt_result['rising_times'])}" if txt_result else "")
                 + f"  expected={n_repeats}",
                 fontsize=11)

    raw_diff      = np.diff(rising_segment)  / fs
    screened_diff = np.diff(rising_screened) / fs
    fixed_diff    = np.diff(rising_fixed)    / fs

    tolerance = 0.04
    lo = (1 - tolerance) * trial_duration
    hi = (1 + tolerance) * trial_duration

    for ax, diff, color, title in [
        (axes[0], raw_diff,      'steelblue',  f'RAW ({len(rising_segment)} edges)'),
        (axes[1], screened_diff, 'darkorange',
         f'PRE-SCREENED ({len(rising_screened)} edges, {n_removed_prescreen} removed)'),
        (axes[2], fixed_diff,    'green',      f'FIXED ({len(rising_fixed)} edges)'),
    ]:
        ax.plot(diff, marker='o', ms=3, lw=0.8, color=color)
        ax.axhline(trial_duration, color='r', linestyle='--',
                   lw=1.2, label=f'expected ({trial_duration}s)')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Interval (s)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Mark points that need fixing in the PRE-SCREENED panel
    glitch_idx  = np.where(screened_diff < lo)[0]
    missing_idx = np.where(screened_diff > hi)[0]
    if len(glitch_idx):
        axes[1].scatter(glitch_idx, screened_diff[glitch_idx],
                        color='red', zorder=5, s=40, label=f'glitch ({len(glitch_idx)})')
    if len(missing_idx):
        axes[1].scatter(missing_idx, screened_diff[missing_idx],
                        color='purple', zorder=5, s=40, marker='^',
                        label=f'missing ({len(missing_idx)})')
    if len(glitch_idx) or len(missing_idx):
        axes[1].legend(fontsize=8)

    # TXT-REFERRING panel: full-length train with recorded vs reconstructed edges
    if txt_result is not None:
        ax = axes[3]
        tr = txt_result['rising_times']
        recon = txt_result['is_reconstructed']
        txt_diff = np.diff(tr) / fs
        eidx = np.arange(len(txt_diff))
        # Colour each ITI by whether the trial it leads INTO was reconstructed.
        real_mask = ~recon[1:]
        ax.plot(eidx, txt_diff, lw=0.6, color='0.75', zorder=1)
        ax.scatter(eidx[real_mask], txt_diff[real_mask], s=10, color='green',
                   zorder=2, label=f'recorded ({txt_result["n_real"]})')
        ax.scatter(eidx[~real_mask], txt_diff[~real_mask], s=10, color='red',
                   zorder=2, label=f'reconstructed ({txt_result["n_reconstructed"]})')
        ax.axhline(trial_duration, color='r', linestyle='--', lw=1.2)
        res_std = txt_result['fit']['residual_ms'].std()
        ax.set_title(f'TXT-REFERRING FIX ({len(tr)} edges, '
                     f'{txt_result["n_reconstructed"]} reconstructed, '
                     f'fit residual std={res_std:.1f} ms)', fontsize=10)
        ax.set_ylabel('Interval (s)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Edge index', fontsize=9)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stamp = (
        f"Generated {timestamp} | script={Path(__file__).name} | task={task_id}\n"
        f"folder={folder_path} | stimulus={stimulus_duration}s, ITI={ITI_duration}s, "
        f"trial_duration={trial_duration}s, display_tolerance=±{tolerance*100:.0f}% | "
        f"n_trials_expected={n_repeats}\n"
        f"raw={len(rising_segment)}, screened={len(rising_screened)} "
        f"({n_removed_prescreen} pre-screened out), fixed={len(rising_fixed)}"
    )
    _stamp_figure(fig, stamp)

    fig.tight_layout(rect=[0, 0.08, 0.83, 1])
    fig.savefig(folder_path / f"{task_id}_DIO_fix.png", dpi=150)

    # ── Select which version to save to the .npz ───────────────────────────────
    version_data = {
        'raw':      (rising_segment,  rising_segment  + int(stimulus_duration * fs)),
        'screened': (rising_screened, rising_screened + int(stimulus_duration * fs)),
        'fixed':    (rising_fixed,    falling_fixed),
    }
    version_options = ['raw', 'screened', 'fixed']
    if txt_result is not None:
        version_options.append('txt_fixed')
        version_data['txt_fixed'] = (txt_result['rising_times'], txt_result['falling_times'])

    # Default to the txt reconstruction when the interval-based fix did not reach
    # the expected trial count but the txt fix did; otherwise default to 'fixed'.
    default_version = 'fixed'
    if txt_result is not None and len(rising_fixed) != n_repeats \
            and len(txt_result['rising_times']) == n_repeats:
        default_version = 'txt_fixed'

    rax = fig.add_axes([0.85, 0.45, 0.13, 0.18])
    rax.set_title('Save version', fontsize=9)
    radio = RadioButtons(rax, version_options, active=version_options.index(default_version))

    save_choice = {'version': default_version}

    def _on_select(label):
        save_choice['version'] = label

    radio.on_clicked(_on_select)
    plt.show()

    chosen_rising, chosen_falling = version_data[save_choice['version']]

    save_path = folder_path / f"{task_id}_DIO.npz"
    np.savez_compressed(save_path, rising_times=chosen_rising, falling_times=chosen_falling)
    print(f"  Saved '{save_choice['version']}' version to {save_path}")

    return {
        'task_id': task_id,
        'rising_times': chosen_rising,
        'falling_times': chosen_falling,
        'trial_duration_samples': trial_duration_samples,
        'stimulus_duration': stimulus_duration,
        'saved_version': save_choice['version'],
    }


# ── Paths ──────────────────────────────────────────────────────────────────────
from grating_utils import load_session_paths
from grating_config import ANIMAL_ID as Animal_id, EXPERIMENT_DATE as experiment_date

rec_folders, task_file_paths = load_session_paths(Animal_id, experiment_date)
animal_id = Animal_id
session_id = rec_folders[0].parent.name  # EphysFolder name
print(f"Processing {animal_id}/{session_id}  —  {len(rec_folders)} rec folder(s), {len(task_file_paths)} task(s)")

# ── Parse all task files ────────────────────────────────────────────────────────
task_metas = []
for p in task_file_paths:
    tf = parse_grating_experiment(p)
    stim = float(tf['parameters']['stimulus_duration'].rstrip('s'))
    iti = float(tf['parameters']['iti_duration'].rstrip('s'))
    n = tf['parameters']['total_trials']
    task_metas.append({'path': p, 'n_trials': n, 'trial_duration': stim + iti})
    print(f"  {p.stem}: {n} trials, {stim+iti}s/trial")

# ── Raw DIO signal ─────────────────────────────────────────────────────────────
fs = 30000
# If True, use cleaned recording-level <rec_stem>_DIO.npz files generated by
# dio_single.py when every rec_folder has one. This script still segments those
# cleaned rising edges by task and writes task-level <task_stem>_DIO.npz files.
PREFER_RECORDING_LEVEL_DIO = True

if PREFER_RECORDING_LEVEL_DIO:
    try:
        rising_times, loaded_cleaned_paths = load_recording_level_dio_edges(rec_folders, channel_id=3)
        print(f"Using {len(loaded_cleaned_paths)} cleaned recording-level DIO file(s).")
    except (FileNotFoundError, ValueError) as e:
        print(f"Cleaned recording-level DIO unavailable: {e}")
        print("Falling back to raw DIO folders.")
        PREFER_RECORDING_LEVEL_DIO = False

if not PREFER_RECORDING_LEVEL_DIO:
    dio_folders = sorted(
        [dio for rf in rec_folders for dio in get_dio_folders(rf)],
        key=lambda x: x.name,
    )
    pd_time, pd_state = concatenate_din_data(dio_folders, 3)
    pd_time = pd_time.ravel()
    pd_state = pd_state.ravel()
    pd_time = pd_time - pd_time[0]

    rising_times = pd_time[np.where(pd_state == 1)[0]]
    falling_times = pd_time[np.where(pd_state == 0)[0]]

    if len(falling_times) > 0 and len(rising_times) > 0 and falling_times[0] < rising_times[0]:
        falling_times = falling_times[1:]
        print("Discarded leading falling edge (no matching rising edge)")

total_expected = sum(m['n_trials'] for m in task_metas)
print(f"Raw edges: {len(rising_times)} rising (total expected {total_expected})")

# ── Global fix before segmentation ────────────────────────────────────────────
# Fixes glitches (edges too close) AND inserts missing trials (gaps < gap_threshold_s).
# Gaps >= gap_threshold_s are left intact as task/session boundaries.
gap_threshold_s = 30.0
gap_threshold_samples = int(gap_threshold_s * fs)

# Use the common trial duration across tasks (assumed identical; adjust if not).
# Ask for it interactively since different experiments use different values
# (e.g. 3s vs 2s trial) and the parsed task-file value is worth confirming.
global_trial_duration_s = task_metas[0]['trial_duration']
ASK_TRIAL_DURATION = True
if ASK_TRIAL_DURATION:
    try:
        user_input = input(
            f"Expected trial duration (stimulus+ITI) in seconds? "
            f"[Enter for default = {global_trial_duration_s:g}s, parsed from task file]: "
        ).strip()
    except EOFError:
        user_input = ""
    if user_input:
        global_trial_duration_s = float(user_input)
        print(f"Using user-specified trial duration: {global_trial_duration_s:g}s")
global_trial_duration_samples = int(global_trial_duration_s * fs)
global_n_trials = total_expected

print(f"Global fix: trial_duration={global_trial_duration_s}s  "
      f"max_gap={gap_threshold_s}s  total_expected={global_n_trials}")

if PREFER_RECORDING_LEVEL_DIO:
    rising_prescreened = rising_times
    global_fix_log = []
    print("Global fix skipped because recording-level DIO was already cleaned.")
else:
    rising_prescreened, _, global_fix_log = fix_rising_edges(
        rising_times,
        n_trials=global_n_trials,
        trial_duration_samples=global_trial_duration_samples,
        max_gap_samples=gap_threshold_samples,
    )
print(f"Global fix: {len(rising_times)} → {len(rising_prescreened)} edges")
for entry in global_fix_log:
    print(f"  {entry}")

# ── Segment by large gaps (> gap_threshold_s) ──────────────────────────────────
auto_segments = segment_by_large_gaps(rising_prescreened, gap_threshold_samples, fs=fs)

# Print a clear summary so you can set manual_task_edge_ranges below
print("\n── Detected segments (edge indices into rising_prescreened) ──")
cumulative = 0
for k, seg in enumerate(auto_segments):
    print(f"  Segment {k+1}: edges [{cumulative} : {cumulative + len(seg)}]  "
          f"({len(seg)} edges)  "
          f"start={seg[0]/fs:.1f}s  end={seg[-1]/fs:.1f}s")
    cumulative += len(seg)
print(f"  Total: {len(rising_prescreened)} edges\n")

# ── Ask how many real task pieces there are ─────────────────────────────────────
# Gap-based auto-segmentation can over-split on a single stray/glitch edge (a
# lone pulse far from the main train shows up as its own 1-edge "segment").
# Tell it how many real pieces to expect and it keeps only the N largest
# auto-detected segments (by edge count), dropping the rest as spurious.
# Set to False for non-interactive/batch runs (auto_segments is used as-is).
ASK_NUM_SEGMENTS = True

if ASK_NUM_SEGMENTS and len(auto_segments) > 1:
    default_n_pieces = len(task_metas)
    try:
        user_input = input(
            f"How many real task piece(s) are there? "
            f"[Enter for default = {default_n_pieces}]: "
        ).strip()
    except EOFError:
        user_input = ""
    n_pieces = int(user_input) if user_input else default_n_pieces

    if 0 < n_pieces < len(auto_segments):
        order = sorted(range(len(auto_segments)), key=lambda i: len(auto_segments[i]), reverse=True)
        kept_idx = sorted(order[:n_pieces])
        dropped_idx = [i for i in range(len(auto_segments)) if i not in kept_idx]
        print(f"Keeping the {n_pieces} largest segment(s): {[i + 1 for i in kept_idx]}; "
              f"dropping spurious segment(s): {[i + 1 for i in dropped_idx]} "
              f"({[len(auto_segments[i]) for i in dropped_idx]} edges each)\n")
        auto_segments = [auto_segments[i] for i in kept_idx]
    elif n_pieces >= len(auto_segments):
        print(f"Requested {n_pieces} piece(s) — nothing to drop "
              f"({len(auto_segments)} segment(s) detected).\n")
    else:
        print(f"Invalid piece count ({n_pieces}) — keeping all {len(auto_segments)} segment(s).\n")

# ── Manual task assignment ─────────────────────────────────────────────────────
# Set each tuple to (start_edge_idx, end_edge_idx) — end is EXCLUSIVE, Python-style.
# Example: task 1 = first 360 edges, task 2 = next 360 edges
#   manual_task_edge_ranges = [(0, 360), (360, 720)]
# Set to None to use the auto segments directly (requires segment count == task count).
manual_task_edge_ranges = None  # Fill in after seeing the printed summary if auto segmentation is wrong.

if manual_task_edge_ranges is not None:
    segments = [rising_prescreened[s:e] for s, e in manual_task_edge_ranges]
    print("Using manual task segments:")
    for k, (rng, seg) in enumerate(zip(manual_task_edge_ranges, segments)):
        print(f"  Task {k+1}: edges {rng} → {len(seg)} edges")
else:
    segments = auto_segments
    if len(segments) != len(task_metas):
        print(f"WARNING: {len(segments)} segment(s) found but {len(task_metas)} task file(s) provided.")
        print("Set manual_task_edge_ranges to assign edges to tasks manually.\n")

# ── Segment overview plot ──────────────────────────────────────────────────────
seg_plot_path = rec_folders[0].parent / f"{session_id}_DIO_segments.png"
plot_rising_edge_segments(rising_prescreened, segments, task_metas, fs=fs,
                          gap_threshold_s=gap_threshold_s, save_path=seg_plot_path)

# ── Per-task processing ────────────────────────────────────────────────────────
for k, (meta, seg) in enumerate(zip(task_metas, segments)):
    print(f"\nTask {k+1}/{len(task_metas)}: {meta['path'].stem}  —  {len(seg)} edges in segment")
    process_task(meta['path'], seg, fs=fs)
