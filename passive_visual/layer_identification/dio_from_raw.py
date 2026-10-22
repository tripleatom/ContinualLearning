"""Stimulus timing from a raw Trodes photodiode DIO, keeping only clean pulses.

For sessions with no reviewed `<task>_DIO.npz`. Onsets are raw rising edges,
zeroed to the DIO file's first timestamp; a trial is kept only if its pulse
width is within `--tolerance-sec` of the session median, which drops stalled
or split stimulus presentations instead of repairing them. Offsets are
onset + the nominal stimulus duration, as in the reviewed files.
"""
import argparse
from pathlib import Path

import numpy as np


def read_din(path):
    raw = Path(path).read_bytes()
    marker = b'<End settings>\n'
    body = raw[raw.index(marker) + len(marker):]
    records = np.frombuffer(body, dtype=[('time', '<u4'), ('state', 'u1')])
    return records['time'].astype(np.int64), records['state']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('din', type=Path, help='*.dio_Controller_Din3.dat')
    parser.add_argument('output', type=Path)
    parser.add_argument('--fs', type=float, default=30000)
    parser.add_argument('--stimulus-sec', type=float, default=1)
    parser.add_argument('--tolerance-sec', type=float, default=.03)
    args = parser.parse_args()
    time, state = read_din(args.din)
    if np.any(np.diff(state.astype(int)) == 0):
        raise ValueError('DIO states do not alternate')
    rising, falling = time[state == 1], time[state == 0]
    falling = falling[falling > rising[0]]
    rising = rising[:len(falling)]
    width = (falling - rising) / args.fs
    keep = np.flatnonzero(abs(width - np.median(width)) <= args.tolerance_sec)
    onset = rising[keep] - time[0]
    np.savez_compressed(args.output, rising_times=onset,
                        falling_times=onset + round(args.stimulus_sec * args.fs),
                        raw_rising_edge_index=keep, source=str(args.din),
                        median_pulse_width_sec=np.median(width), tolerance_sec=args.tolerance_sec)
    dropped = np.setdiff1d(np.arange(len(rising)), keep)
    span = f' (raw edge indices {dropped.min()}..{dropped.max()})' if len(dropped) else ''
    print(f'{len(rising)} raw pulses, kept {len(keep)}, dropped {len(dropped)}{span}; '
          f'median width {np.median(width):.4f} s', flush=True)


if __name__ == '__main__':
    main()
