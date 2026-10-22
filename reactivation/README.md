# Reactivation

| Folder | Contents |
|---|---|
| `decoders/` | **`decoder_gui.py` / `decoder_workbench.py`: configurable decoder (context, kinematic inputs, classifiers, ITI speed matching).** Stimulus decoders trained on task / passive data: Gaussian and Poisson NB, LDA, logistic regression, linear SVM, random forest, AODE (`decode_utils.py`, `speed_matching.py`, `train_task_test_passive.py`) |
| `decoders/pls/` | PLS-based reactivation: `pls_task_trials.py` fits the awake model; `pls_sleep_bias.py`, `pls_spindle_bias.py` and `pls_spindle_timing_shuffle.py` score NREM and spindles against it |
| `sleep_apply/` | Apply a decoder to sleep blocks and relate it to NREM and MUA ON/OFF state; `sleep_states.py` is the shared NREM / ON-interval loader |
| `tuning_coupling/` | Whether similarly tuned pairs co-fire more in NREM UP states (not a decoder) |

Sleep state comes from the scoring pipeline in `sleep/` (`staging/`, `mua_onoff/`),
never from the sorted units a decoder is tested on.
