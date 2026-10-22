# decoders

Decoding analyses for task, passive, merged task/passive, and sleep reactivation
data.

## Decoder workbench (start here)

`decoder_gui.py` (or `decoder_workbench.py` on the command line) is the single entry
point for stimulus-decoding comparisons. Choose:

- **context**: task, passive, or merged (task + passive stacked);
- **inputs**: firing rate is always included; optionally add speed, heading
  (sin/cos), head angle (sin/cos) and/or arena position (x, y);
- **classifiers**: Gaussian NB, Poisson NB, LDA, logistic regression, linear SVM,
  random forest, AODE;
- **bin sizes, CV folds, repeats, label-shuffle null, seed**;
- **speed matching** (task only): controls for running speed by sub-sampling the ITI
  until its speed histogram matches the stim-on one, and compares that with a uniformly
  sub-sampled ITI of the same size (water-fill matching; KS check of the histograms is
  saved as a figure). It replaces the with/without-features comparison; inputs can still
  be added to both arms.

With inputs added it also runs a rate-only baseline on the same bins and reports the
paired gain (bootstrap CI and p-value); with shuffles it reports z-scores against the
null at the best bin. Output goes to `<session folder>/reactivation/` as a pkl, a
summary figure, and a bar chart at one bin, each stamped with the command that
reproduces it.

    python reactivation/decoders/decoder_gui.py
    python reactivation/decoders/decoder_workbench.py --context merged --features speed heading --shuffles 10

Two options reproduce the older scripts exactly:

- `--merged-balance class` balances merged training folds by class only (the
  former `compare_merged_*` scripts); the default balances class x context, as
  `train_task_test_passive.py` does.
- `--count-models-see-features` also feeds the added inputs to Poisson NB and AODE
  (the former kinematics scripts did); by default those two see firing rates only.

These replaced `compare_task_decoders*.py` (rate only, with velocity / heading /
kinematics), `compare_passive_stimtype_decoders.py`, `compare_merged_*` and
`best_decoder.py`, and `compare_task_decoders_velocity_matched.py` (its matching code now
lives in `speed_matching.py`). The workbench was checked against them number for number
(identical accuracies, per-class recall, gains and p-values); they remain in git
history under `reactivation/VStimOnDecoding/`.

## Core files

- `params.py` centralizes data paths, joint (orientation, SF) class definitions,
  per-context column maps, bin sizes, CV settings, and sleep-event settings.
- `decode_utils.py` holds the classifiers (Gaussian/Poisson NB, LDA, logistic regression,
  linear SVM, random forest, and AODE, a scikit-learn-style `AODEClassifier` that binarises
  firing rates internally) returned by `make_classifiers()`, plus the cross-validation
  runners. Every classifier goes through the same code path.
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

## Other scripts

- `speed_matching.py` holds the ITI speed-matching functions used by `--speed-match`.
- `train_task_test_passive.py` runs the within-context decoding (passive, merged,
  task with and without kinematics) and saves the CV results that
  `apply_merged_decoder_to_sleep.py` loads to pick the best merged decoder.
- `plot_decoding_schematic.py` draws the schematic of what the decoder reads.
- `../sleep_apply/apply_merged_decoder_to_sleep.py` fits the best merged decoder and
  applies it to complete, block-relative sleep spike files. When registered
  sleep artifacts are accessible it also annotates events with canonical NREM
  and channel-MUA ON/OFF state through `reactivation/sleep_apply/sleep_states.py`.

## Cleanup notes

Keep generated outputs outside this code directory when possible. The repository
already ignores `__pycache__`, `*.pkl`, and `*.png`, so generated cache/results
should stay untracked.
