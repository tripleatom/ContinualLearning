"""Regenerate Combined_Pattern_Registrations.pkl from stimulation .mat files.

This script reconstructs the list-of-dicts format used by the shared LS19
dataset from the MATLAB files in a recording branch folder.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import scipy.io as sio


DEFAULT_BRANCH_DIR = Path(
    r"D:\Cemtimani\SG_LS19_12182025\18-Dec-2025\Branch3"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\Windows\Desktop\LS19_Dec25\LS19_Dec25"
    r"\Combined_Pattern_Registrations_regenerated.pkl"
)


def _as_1d(value: Any) -> np.ndarray:
    """Return MATLAB scalar/empty/vector values as a one-dimensional array."""
    if value is None:
        return np.array([])
    return np.atleast_1d(np.asarray(value)).ravel()


def _loadmat(path: Path) -> dict[str, Any]:
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _single_file(branch_dir: Path, pattern: str) -> Path:
    matches = sorted(branch_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern!r} in {branch_dir}, found {len(matches)}"
        )
    return matches[0]


def _stim_times_by_channel(branch_dir: Path) -> dict[int, np.ndarray]:
    mat = _loadmat(branch_dir / "stim_times_arr.mat")
    stim_times_arr = np.asarray(mat["stim_times_arr"], dtype=object)

    out: dict[int, np.ndarray] = {}
    for channel, timestamps in stim_times_arr:
        out[int(channel)] = np.asarray(timestamps, dtype=np.int64).ravel()
    return out


def _hdf5_array(dataset: Any) -> np.ndarray:
    return np.asarray(dataset[()]).ravel()


def _hdf5_scalar(dataset: Any) -> float:
    return float(_hdf5_array(dataset)[0])


def _hdf5_step(file: h5py.File, ref: Any) -> SimpleNamespace:
    group = file[ref]
    return SimpleNamespace(
        cfg_indices=_hdf5_array(group["cfg_indices"]),
        channels_vec=_hdf5_array(group["channels_vec"]),
        currents_vec=_hdf5_array(group["currents_vec"]),
        delay_vec=_hdf5_array(group["delay_vec"]),
    )


def _hdf5_pattern(file: h5py.File, ref: Any) -> SimpleNamespace:
    group = file[ref]
    steps = [_hdf5_step(file, step_ref) for step_ref in _hdf5_array(group["steps"])]
    return SimpleNamespace(
        steps=np.asarray(steps, dtype=object),
        lambda_=_hdf5_scalar(group["lambda"]),
        global_id=int(_hdf5_scalar(group["global_id"])),
    )


def _hdf5_pattern_libraries(exp_keys_path: Path) -> tuple[np.ndarray, np.ndarray]:
    patterns: dict[str, np.ndarray] = {}
    with h5py.File(exp_keys_path, "r") as file:
        for key in ("Patterns_sample", "Patterns_oracle"):
            pattern_refs = _hdf5_array(file[key])
            patterns[key] = np.asarray(
                [_hdf5_pattern(file, ref) for ref in pattern_refs],
                dtype=object,
            )
    return patterns["Patterns_sample"], patterns["Patterns_oracle"]


def _pattern_libraries(branch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    patterns_dir = branch_dir / "patterns"
    sample_path = patterns_dir / "Patterns_Sample.mat"
    oracle_path = patterns_dir / "Patterns_ORACLE.mat"
    if sample_path.exists() and oracle_path.exists():
        sample = _loadmat(sample_path)
        oracle = _loadmat(oracle_path)
        return (
            np.asarray(sample["Patterns_sample"], dtype=object).ravel(),
            np.asarray(oracle["Patterns_oracle"], dtype=object).ravel(),
        )

    exp_keys_path = _single_file(branch_dir, "exp_keys_*.mat")
    return _hdf5_pattern_libraries(exp_keys_path)


def _pattern_lambda(pattern: Any) -> float:
    if hasattr(pattern, "lambda_"):
        return float(pattern.lambda_)
    return float(getattr(pattern, "lambda"))


def _get_pattern(
    pattern_name: int, sample_patterns: np.ndarray, oracle_patterns: np.ndarray
) -> Any:
    if 1 <= pattern_name <= len(sample_patterns):
        return sample_patterns[pattern_name - 1]

    oracle_index = pattern_name - len(sample_patterns) - 1
    if 0 <= oracle_index < len(oracle_patterns):
        return oracle_patterns[oracle_index]

    raise ValueError(f"Pattern id {pattern_name} is outside sample/oracle libraries")


def _channel_delays(step: Any) -> list[dict[str, int]]:
    channels = _as_1d(getattr(step, "channels_vec", []))
    delays = _as_1d(getattr(step, "delay_vec", []))

    if len(channels) == 2 and np.all(channels == 0) and np.all(delays == 0):
        return []

    if len(channels) != len(delays):
        raise ValueError(
            "Pattern step has mismatched channel and delay vector lengths: "
            f"{len(channels)} vs {len(delays)}"
        )

    return [
        {"channel": int(channel), "delay_mode": int(delay)}
        for channel, delay in zip(channels, delays)
    ]


def _delay_offset_samples(delay_mode: int, stim_params: Any) -> int:
    """Return the approximate search offset for delayed stimulation modes."""
    if delay_mode != 1:
        return 0

    fs = 30_000
    fine_tune_adjust = float(getattr(stim_params, "fine_tune_adjust", 0.01))
    ipi = int(getattr(stim_params, "IPI", 3))
    return int(round(fine_tune_adjust * fs)) + ipi


def _step_start_timestamp(
    channel_delays: list[dict[str, int]],
    expected_timestamp: int,
    stim_times: dict[int, np.ndarray],
    stim_params: Any,
    backward_tolerance_samples: int = 0,
    max_forward_tolerance_samples: int = 1_500,
) -> int:
    """Find the actual onset of a non-empty step from channel stim timestamps."""
    candidates: list[int] = []

    for channel_delay in channel_delays:
        channel = channel_delay["channel"]
        delay_mode = channel_delay["delay_mode"]
        offset = _delay_offset_samples(delay_mode, stim_params)

        if channel not in stim_times:
            raise KeyError(f"Channel {channel} is missing from stim_times_arr.mat")

        times = stim_times[channel]
        if backward_tolerance_samples > 100 or offset:
            search_timestamp = (
                expected_timestamp + offset - backward_tolerance_samples
            )
        else:
            search_timestamp = expected_timestamp + offset
        idx = int(np.searchsorted(times, search_timestamp, side="left"))
        if idx >= len(times):
            continue

        candidate = int(times[idx])
        if candidate <= expected_timestamp + max_forward_tolerance_samples:
            candidates.append(candidate)

    if not candidates:
        return int(expected_timestamp)

    return min(candidates)


def regenerate_combined_pattern_registrations(
    branch_dir: Path,
    sample_rate: int = 30_000,
) -> list[dict[str, Any]]:
    pattern_starts_mat = _loadmat(_single_file(branch_dir, "pattern_starts_*.mat"))
    pattern_starts = np.asarray(pattern_starts_mat["PatternStarts"])
    stim_params = pattern_starts_mat.get("stim_params")

    if stim_params is None:
        stim_params = _loadmat(_single_file(branch_dir, "stim_params_*.mat"))[
            "stim_params"
        ]

    sample_patterns, oracle_patterns = _pattern_libraries(branch_dir)
    stim_times = _stim_times_by_channel(branch_dir)

    step_samples = int(round(float(getattr(stim_params, "step_t")) * sample_rate))
    inter_pattern_samples = int(
        round(float(getattr(stim_params, "inter_pattern_time")) * sample_rate)
    )
    registrations: list[dict[str, Any]] = []
    next_pattern_expected_timestamp: int | None = None

    for timing_index, row in enumerate(pattern_starts):
        pattern_flag_start_timestamp = int(row[0] * sample_rate)
        pattern_name = int(row[2])
        pattern = _get_pattern(pattern_name, sample_patterns, oracle_patterns)
        pattern_lambda = _pattern_lambda(pattern)

        registration = {
            "pattern_name": pattern_name,
            "pattern_lambda": pattern_lambda,
            "pattern_flag_start_timestamp": pattern_flag_start_timestamp,
            "pattern_timing_index": timing_index,
            "steps": [],
        }

        if next_pattern_expected_timestamp is None:
            expected_timestamp = pattern_flag_start_timestamp
        else:
            expected_timestamp = next_pattern_expected_timestamp

        steps = np.asarray(pattern.steps, dtype=object).ravel()
        pending_empty_step_indices: list[int] = []
        resolved_any_step = False

        for step_index, step in enumerate(steps):
            channel_delays = _channel_delays(step)
            if channel_delays:
                try:
                    start_timestamp = _step_start_timestamp(
                        channel_delays,
                        expected_timestamp,
                        stim_times,
                        stim_params,
                        backward_tolerance_samples=(
                            3_000 if not resolved_any_step else 10
                        ),
                        max_forward_tolerance_samples=(
                            10_000 if not resolved_any_step else step_samples
                        ),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed at pattern index {timing_index}, "
                        f"pattern {pattern_name}, step {step_index}, "
                        f"expected sample {expected_timestamp}"
                    ) from exc
                if pending_empty_step_indices:
                    first_pending_start = start_timestamp - (
                        len(pending_empty_step_indices) * step_samples
                    )
                    for pending_offset, pending_step_index in enumerate(
                        pending_empty_step_indices
                    ):
                        registration["steps"][pending_step_index][
                            "start_timestamp"
                        ] = int(first_pending_start + pending_offset * step_samples)
                    pending_empty_step_indices.clear()
                resolved_any_step = True
            else:
                start_timestamp = expected_timestamp

            registration["steps"].append(
                {
                    "index": step_index,
                    "channel_delays": channel_delays,
                    "start_timestamp": int(start_timestamp),
                }
            )
            if not channel_delays and not resolved_any_step:
                pending_empty_step_indices.append(step_index)

            expected_timestamp = int(start_timestamp) + step_samples

        if pending_empty_step_indices:
            for pending_step_index in pending_empty_step_indices:
                registration["steps"][pending_step_index]["start_timestamp"] = int(
                    registration["steps"][pending_step_index]["start_timestamp"]
                )

        last_step_timestamp = int(registration["steps"][-1]["start_timestamp"])
        next_pattern_expected_timestamp = (
            last_step_timestamp + step_samples + inter_pattern_samples
        )
        registrations.append(registration)

    return registrations


def _compare(left: Any, right: Any) -> None:
    if left == right:
        print("Comparison OK: regenerated registrations exactly match reference.")
        return

    print("Comparison FAILED: regenerated registrations differ from reference.")
    if len(left) != len(right):
        print(f"Length differs: regenerated={len(left)}, reference={len(right)}")
        return

    for pattern_index, (left_pattern, right_pattern) in enumerate(zip(left, right)):
        if left_pattern == right_pattern:
            continue
        print(f"First differing pattern index: {pattern_index}")
        print("Regenerated:", left_pattern)
        print("Reference:   ", right_pattern)
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Combined_Pattern_Registrations.pkl from .mat files."
    )
    parser.add_argument("--branch-dir", type=Path, default=DEFAULT_BRANCH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=int, default=30_000)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional existing pickle to compare against after writing.",
    )
    args = parser.parse_args()

    registrations = regenerate_combined_pattern_registrations(
        args.branch_dir,
        sample_rate=args.sample_rate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(registrations, f)

    print(f"Wrote {len(registrations)} registrations to {args.output}")

    if args.compare is not None:
        with args.compare.open("rb") as f:
            reference = pickle.load(f)
        _compare(registrations, reference)


if __name__ == "__main__":
    main()
