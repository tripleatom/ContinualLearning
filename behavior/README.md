# Centimani Freely Roaming Analysis

Quick start: see [docs/data_understanding.md](docs/data_understanding.md#quickstart).

The current full analysis dataset is organized under:

- `Centimani_implanted/`
- `Comparison_unimplanted/`

Use `no_use_videos/` only for QC or explicitly exploratory runs. The main
analysis workflow expects each included session to have:

- `*_VIDEO.avi`
- `*_TS.npy`
- `*_DLC.hdf5`

For the calibrated locomotion workflow, start with:

```powershell
python inspect_free_roaming_folder.py --max-files 0
python analyze_tracking_summary.py --output-dir analysis_outputs\tracking_summary_full_set
python review_video_calibrations.py --default-edge-points-json analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration_edge_points.json --arena-edge-length-cm 52 --output-dir analysis_outputs\arena_calibration_full_set --manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv
python dlc_locomotion_pipeline.py --calibration-manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv --arena-edge-length-cm 52 --coverage-bin-size 0.025 --coverage-bin-size-m 0.025 --output-dir analysis_outputs\dlc_locomotion_full_set_52cm
python compare_locomotion_groups.py analysis_outputs\dlc_locomotion_full_set_52cm --manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv --output-dir analysis_outputs\dlc_locomotion_full_set_52cm\group_comparison
```
