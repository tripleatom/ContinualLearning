#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def explore(obj, prefix=""):
    """Recursively print keys with indentation."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}")
            explore(v, prefix + "  ")
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # show keys of first element in list
        explore(obj[0], prefix + "[0].")
    # otherwise ignore primitive values

def main(path):
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Top‑level keys in {path}:")
    explore(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_json.py /path/to/file.json")
        sys.exit(1)
    main(sys.argv[1])