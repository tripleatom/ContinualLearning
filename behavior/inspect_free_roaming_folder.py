#!/usr/bin/env python
"""Inspect freely roaming behavior data folders without loading large files.

The script summarizes the file inventory, groups matching VIDEO/TS/PROC/DLC
files into recording sessions, and prints lightweight metadata from common
file types used in this folder.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import pickle
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


SESSION_SUFFIXES = {
    "VIDEO": re.compile(r"^(?P<base>.+)_VIDEO\.avi$", re.IGNORECASE),
    "TS": re.compile(r"^(?P<base>.+)_TS\.npy$", re.IGNORECASE),
    "DLC": re.compile(r"^(?P<base>.+)_DLC\.hdf5?$", re.IGNORECASE),
    "PROC": re.compile(r"^(?P<base>.+)_PROC$", re.IGNORECASE),
}


def human_bytes(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n_bytes} B"


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace(os.sep, "/")


def detect_kind(path: Path) -> str:
    name = path.name
    for kind, pattern in SESSION_SUFFIXES.items():
        if pattern.match(name):
            return kind
    suffix = path.suffix.lower().lstrip(".")
    return suffix.upper() if suffix else "NO_EXT"


def session_base(path: Path) -> str | None:
    for pattern in SESSION_SUFFIXES.values():
        match = pattern.match(path.name)
        if match:
            return match.group("base")
    return None


def summarize_npy(path: Path, sample_count: int) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    flat = arr.reshape(-1) if arr.size else arr
    sample = flat[:sample_count].tolist()
    info: dict[str, Any] = {
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "sample": sample,
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = np.asarray(flat[np.isfinite(flat)]) if np.issubdtype(arr.dtype, np.floating) else flat
        if finite.size:
            info.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "first": float(flat[0]),
                    "last": float(flat[-1]),
                }
            )
            if arr.size > 1:
                diffs = np.diff(flat[: min(arr.size, 10000)])
                info["median_step_first_10k"] = float(np.median(diffs))
    return info


def summarize_hdf5(path: Path, sample_count: int) -> dict[str, Any]:
    info: dict[str, Any] = {"hdf5_nodes": []}
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            node: dict[str, Any] = {"path": f"/{name}", "type": obj.__class__.__name__}
            if isinstance(obj, h5py.Dataset):
                node["shape"] = tuple(int(x) for x in obj.shape)
                node["dtype"] = str(obj.dtype)
            info["hdf5_nodes"].append(node)

        h5.visititems(visitor)

    try:
        with pd.HDFStore(path, mode="r") as store:
            info["pandas_keys"] = store.keys()
            for key in store.keys()[:3]:
                storer = store.get_storer(key)
                info.setdefault("pandas_tables", []).append(
                    {
                        "key": key,
                        "nrows": getattr(storer, "nrows", None),
                        "shape": getattr(storer, "shape", None),
                    }
                )
                try:
                    frame = store.select(key, start=0, stop=sample_count)
                    info.setdefault("pandas_samples", {})[key] = {
                        "shape": tuple(int(x) for x in frame.shape),
                        "columns": [str(x) for x in frame.columns[:10]],
                        "head": json.loads(frame.head(min(sample_count, 3)).to_json(orient="split")),
                    }
                except Exception as exc:  # pragma: no cover - best effort display
                    info.setdefault("pandas_sample_errors", {})[key] = str(exc)
    except Exception as exc:
        info["pandas_error"] = str(exc)
    return info


def parse_avi_header(path: Path) -> dict[str, Any]:
    """Return basic AVI metadata by scanning for the avih chunk."""
    info: dict[str, Any] = {}
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12:
            return {"error": "file too small"}
        info["riff_header"] = header[:4].decode("ascii", errors="replace")
        info["riff_type"] = header[8:12].decode("ascii", errors="replace")
        scan = f.read(1024 * 1024)

    idx = scan.find(b"avih")
    if idx >= 0 and idx + 8 + 40 <= len(scan):
        size = struct.unpack_from("<I", scan, idx + 4)[0]
        chunk = scan[idx + 8 : idx + 8 + min(size, 56)]
        if len(chunk) >= 40:
            fields = struct.unpack_from("<10I", chunk, 0)
            microsec_per_frame = fields[0]
            total_frames = fields[4]
            width = fields[8]
            height = fields[9]
            fps = 1_000_000.0 / microsec_per_frame if microsec_per_frame else None
            duration = total_frames / fps if fps else None
            info.update(
                {
                    "width": width,
                    "height": height,
                    "total_frames_header": total_frames,
                    "fps_header": fps,
                    "duration_sec_header": duration,
                }
            )
    return info


def summarize_proc(path: Path, sample_bytes: int) -> dict[str, Any]:
    with path.open("rb") as f:
        data = f.read(sample_bytes)
    printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in data)
    text_fraction = printable / len(data) if data else 0
    summary: dict[str, Any] = {
        "first_bytes_hex": data[:64].hex(" "),
        "printable_fraction_first_block": round(text_fraction, 3),
    }
    if text_fraction > 0.8:
        summary["text_preview"] = data.decode("utf-8", errors="replace")[:1000]
    return summary


def summarize_value(value: Any, sample_count: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, np.ndarray):
        summary.update({"shape": tuple(int(x) for x in value.shape), "dtype": str(value.dtype)})
        if value.size:
            summary["sample"] = value.reshape(-1)[:sample_count].tolist()
    elif isinstance(value, (list, tuple)):
        summary.update({"len": len(value), "sample": list(value[:sample_count])})
    elif isinstance(value, dict):
        summary.update({"len": len(value), "keys": [str(k) for k in list(value)[:20]]})
    elif isinstance(value, (str, int, float, bool)) or value is None:
        summary["value"] = value
    else:
        summary["repr"] = repr(value)[:500]
    return summary


def summarize_proc_pickle(path: Path, sample_count: int) -> dict[str, Any]:
    with path.open("rb") as f:
        value = pickle.load(f)
    if isinstance(value, dict):
        return {
            "pickle_type": "dict",
            "keys": [str(k) for k in value.keys()],
            "values": {str(k): summarize_value(v, sample_count) for k, v in value.items()},
        }
    return {"pickle_type": type(value).__name__, "value": summarize_value(value, sample_count)}


def print_json_block(title: str, value: Any) -> None:
    print(f"\n{title}")
    print(json.dumps(value, indent=2, default=str))


def is_excluded(path: Path, root: Path, include_code: bool) -> bool:
    if include_code:
        return False
    relative_parts = path.relative_to(root).parts
    if "__pycache__" in relative_parts:
        return True
    if path.suffix.lower() in {".py", ".pyc"}:
        return True
    return False


def inspect(
    root: Path,
    sample_count: int,
    sample_bytes: int,
    max_files: int | None,
    include_code: bool,
    load_proc_pickle: bool,
) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and not is_excluded(p, root, include_code))
    dirs = sorted(p for p in root.rglob("*") if p.is_dir())
    if max_files is not None:
        files_to_detail = files[:max_files]
    else:
        files_to_detail = files

    print(f"Root: {root}")
    print(f"Directories: {len(dirs)}")
    print(f"Files: {len(files)}")
    print(f"Total size: {human_bytes(sum(p.stat().st_size for p in files))}")

    by_kind = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_folder = defaultdict(lambda: {"count": 0, "bytes": 0})
    for path in files:
        size = path.stat().st_size
        kind = detect_kind(path)
        folder = rel(path.parent, root)
        by_kind[kind]["count"] += 1
        by_kind[kind]["bytes"] += size
        by_folder[folder]["count"] += 1
        by_folder[folder]["bytes"] += size

    print_json_block(
        "File types",
        {
            k: {"count": v["count"], "size": human_bytes(v["bytes"])}
            for k, v in sorted(by_kind.items())
        },
    )
    print_json_block(
        "Folders",
        {
            k: {"count": v["count"], "size": human_bytes(v["bytes"])}
            for k, v in sorted(by_folder.items())
        },
    )

    sessions: dict[str, dict[str, str]] = defaultdict(dict)
    for path in files:
        base = session_base(path)
        if base:
            sessions[f"{rel(path.parent, root)}/{base}"][detect_kind(path)] = rel(path, root)
    print_json_block("Recording sessions", sessions)

    print("\nDetailed file summaries")
    for path in files_to_detail:
        stat = path.stat()
        kind = detect_kind(path)
        summary: dict[str, Any] = {
            "path": rel(path, root),
            "kind": kind,
            "size": human_bytes(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
        try:
            if kind == "TS":
                summary["content"] = summarize_npy(path, sample_count)
            elif kind == "DLC":
                summary["content"] = summarize_hdf5(path, sample_count)
            elif kind == "VIDEO":
                summary["content"] = parse_avi_header(path)
            elif kind == "PROC":
                summary["content"] = summarize_proc(path, sample_bytes)
                if load_proc_pickle:
                    summary["content"]["pickle"] = summarize_proc_pickle(path, sample_count)
            else:
                summary["content"] = {"note": "No specialized parser for this file type."}
        except Exception as exc:
            summary["content_error"] = f"{exc.__class__.__name__}: {exc}"
        print_json_block(summary["path"], summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize freely roaming VIDEO/TS/PROC/DLC files for pipeline planning."
    )
    parser.add_argument("root", nargs="?", default=".", type=Path, help="Folder to inspect.")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of array/table items to preview.")
    parser.add_argument("--sample-bytes", type=int, default=4096, help="Bytes to preview from PROC files.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit detailed summaries to the first N files.")
    parser.add_argument("--include-code", action="store_true", help="Include .py/.pyc files and __pycache__.")
    parser.add_argument(
        "--load-proc-pickle",
        action="store_true",
        help="Load *_PROC pickle files and summarize keys. Only use for trusted local data.",
    )
    args = parser.parse_args()

    inspect(
        args.root.resolve(),
        args.sample_count,
        args.sample_bytes,
        args.max_files,
        args.include_code,
        args.load_proc_pickle,
    )


if __name__ == "__main__":
    main()
