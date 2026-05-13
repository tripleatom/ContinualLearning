# VStimOnDecoding

Decoding analyses for task, passive, merged task/passive, and sleep reactivation
data.

## Core files

- `params.py` centralizes data paths, joint (orientation, SF) class definitions,
  per-context column maps, bin sizes, CV settings, and sleep-event settings.
- `decode_utils.py`, `decode_aode.py`, and `decode_naive_bayes.py` contain
  classifier and cross-validation helpers.
- `kinematics_utils.py` contains shared DLC-cleaning, time-frame alignment, and
  per-bin kinematic feature helpers.
- `prepare_*.py` scripts build binned feature matrices and labels from spike
  pickles. Both `prepare_task_stimtype.py` and `prepare_passive_stimtype.py`
  accept the canonical `class_pos` / `class_neg` dicts plus a per-context
  `col_map`, so task and passive use the same physical label definition.
  `prepare_task_stimtype.infer_rewarded_combination()` recovers the rewarded
  (orientation, SF) pair from `trial_params` as a side channel — the rewarded
  grating may appear on either side trial to trial.

## Labels

Both contexts share the same joint label scheme defined in `params.py`:

```python
class_pos = {"orientation": 0.0, "spatial_freq": 0.04}   # → +1
class_neg = {"orientation": 0.0, "spatial_freq": 0.16}   # → -1
```

A bin gets `+1` iff its trial's left grating matches every key of `class_pos`
(via TASK_COL_MAP / PASSIVE_COL_MAP); `-1` iff it matches every key of
`class_neg`; `0` if it falls outside any stimulus epoch. Any other left-grating
combination is dropped.

## Analysis scripts

- `compare_task_decoders*.py` compare task stim-type decoders with optional
  velocity, heading, kinematic, or velocity-matched controls.
- `compare_passive_stimtype_decoders.py` compares passive stim-type decoders.
- `compare_merged_vs_unmerged_decoders.py` compares task, passive, and merged
  decoders using spikes only.
- `compare_merged_vs_unmerged_decoders_with_kinematics.py` compares task,
  passive, and merged decoders with appended kinematic columns.
- `compare_merged_with_vs_without_kinematics.py` directly tests whether the
  merged decoder improves when kinematic columns are added.
- `apply_merged_decoder_to_sleep.py` fits the best merged decoder and applies it
  to sleep blocks (note: still references the old single-feature label API —
  needs an update before it can be re-run against the new params).

## Cleanup notes

Keep generated outputs outside this code directory when possible. The repository
already ignores `__pycache__`, `*.pkl`, and `*.png`, so generated cache/results
should stay untracked.
