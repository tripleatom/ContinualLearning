# Sleep reactivation analyses

The active decoder entry point is `apply_merged_decoder_to_sleep.py`. It uses
block-relative seconds from the extracted sleep-spike pickle and, when the
registered artifacts are available, annotates events with the completed sleep
pipeline's NREM and channel-MUA ON/OFF results.

`sleep_states.py` is the shared loader: `SleepBlock` time base, NREM and MUA ON
intervals, artifact resolution from `sleep/sleep_day_configs.json`, and
`annotate_decoder_results`. State-dependent analyses use the independent
channel-level MUA output written by `sleep/mua_onoff/detect_off_states.py`; states
are not inferred from the sorted population the decoder uses, avoiding circular
coupling/reactivation tests.

All sleep-decoder results live in one folder, `<session folder>/reactivation/sleep_merged_neural_cv_<session>/`,
named by `sleep_states.decoder_results_dir()`. `apply_merged_decoder_to_sleep.py` writes
`best_merged_decoder_model.pkl` and `<block>/sleep_decoding_results.pkl` there, and `template_loo_correlation.py` writes
`template_diagnostics/` into it. Change `MERGED_CONDITION` in `sleep_states.py` to switch
condition everywhere at once.
