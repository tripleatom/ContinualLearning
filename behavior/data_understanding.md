# Freely Roaming Data Understanding

This document is the living notebook for understanding the current freely
roaming behavior folder and shaping it into a repeatable processing pipeline.
Update it whenever the analysis scripts reveal a new data assumption, QC rule,
or processing decision.

## Current Folder Structure

The data folder currently contains two top-level experiment groups:

- `Centimani_implanted`
- `Comparison_unimplanted`

Each recording session currently follows this four-file pattern:

- `*_VIDEO.avi`: front-camera video.
- `*_TS.npy`: video frame timestamps.
- `*_DLC.hdf5`: DeepLabCut tracking output.
- `*_PROC`: trusted local Python pickle containing processed local-motion and signal data.

Some sessions are stored under `no_use_videos/`; these are still discoverable
by the scripts but should be treated as excluded or QC-only unless explicitly
included in a final analysis.

Current full-set inventory after adding the remaining videos and DLC outputs:

- `16` discovered session folders/filesets.
- `15` sessions have video, timestamp, and DLC files.
- `13` sessions also have `_PROC` files.
- `9` main-analysis sessions outside `no_use_videos/` have video, timestamp,
  and DLC files.

Main analysis sessions currently present:

| Group | Main Sessions With Video/DLC/TS |
|---|---|
| `Centimani_implanted` | `front_camera_LS16_2026-05-16_3`, `front_camera_LS18_2026-04-02_1`, `front_camera_LS19_2026-04-02_3`, `front_camera_LS20_2026-04-02_1`, `Imaging_source_bot01_2024-12-06_1` |
| `Comparison_unimplanted` | `front_camera_C57_0005_C1_6677_2026-05-03_3`, `front_camera_C57_0008_C1_6673_2026-05-03_1`, `front_camera_GCamp-0001-G1_2026-05-14_1`, `front_camera_GCamp-0002-G1_2026-05-14_1` |

`no_use_videos/` currently includes QC/excluded recordings. One excluded LS16
video has only `*_VIDEO.avi`; do not include it in DLC analysis unless matching
timestamp and DLC files are added.

## Quickstart

This quickstart is the recommended path for the full analysis set. It assumes
each included session has these files in either `Centimani_implanted/` or
`Comparison_unimplanted/`:

- `*_VIDEO.avi`
- `*_TS.npy`
- `*_DLC.hdf5`

The `_PROC` file is useful for comparison with older processed local-motion
signals, but it is not required for the DLC centroid locomotion pipeline.
Sessions under `no_use_videos/` are excluded by default.

For the planned direct-folder 30-minute analysis with full console logging, use
the dedicated procedure:

```text
docs/full_set_30min_analysis_procedure.md
```

### 1. Inspect the Full Set

Use this first after adding or moving files:

```powershell
python inspect_free_roaming_folder.py --max-files 0
```

Optional deeper preview:

```powershell
python inspect_free_roaming_folder.py --max-files 4 --sample-count 3 --load-proc-pickle
```

### 2. Refresh the Broad Data Summary

This gives a first-pass overview of videos, DLC confidence, `_PROC` availability,
local-motion summaries when `_PROC` exists, and simple arena coverage:

```powershell
python analyze_tracking_summary.py --output-dir analysis_outputs\tracking_summary_full_set
```

Main outputs:

- `analysis_outputs/tracking_summary_full_set/session_summary.csv`
- `analysis_outputs/tracking_summary_full_set/dlc_keypoint_summary.csv`
- `analysis_outputs/tracking_summary_full_set/tracking_data_understanding_report.md`
- `analysis_outputs/tracking_summary_full_set/plots/`

### 3. Review Per-Video Arena Calibration

The preferred full-set workflow is per-video calibration from a manifest. This
handles the new LS16 and GCamp sessions and avoids blindly applying an older
shared transform to newly added videos.

Use the accepted shared edge points as a starting point:

```text
analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration_edge_points.json
```

Interactive full-set calibration review:

```powershell
python review_video_calibrations.py --default-edge-points-json analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration_edge_points.json --arena-edge-length-cm 52 --output-dir analysis_outputs\arena_calibration_full_set --manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv
```

For each video, check the proposed edge points and save a session-specific
calibration JSON. The manifest becomes the authoritative input for the
full-set locomotion run.

If you only need a fast non-GUI smoke check from the existing reviewed edge
points:

```powershell
python review_video_calibrations.py --default-edge-points-json analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration_edge_points.json --arena-edge-length-cm 52 --output-dir analysis_outputs\arena_calibration_full_set_smoke --manifest analysis_outputs\arena_calibration_full_set_smoke\calibration_manifest.csv --no-gui
```

Use the smoke manifest for testing mechanics only; use reviewed per-video
calibrations for final analysis.

### 4. Run Full-Set Calibrated Locomotion

Run calibrated DLC-only locomotion from the reviewed full-set manifest:

```powershell
python dlc_locomotion_pipeline.py --calibration-manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv --arena-edge-length-cm 52 --coverage-bin-size 0.025 --coverage-bin-size-m 0.025 --output-dir analysis_outputs\dlc_locomotion_full_set_52cm
```

Main outputs:

- `analysis_outputs/dlc_locomotion_full_set_52cm/dlc_locomotion_session_summary.csv`
- `analysis_outputs/dlc_locomotion_full_set_52cm/dlc_locomotion_report.md`
- `analysis_outputs/dlc_locomotion_full_set_52cm/sessions/*/*_dlc_locomotion.parquet`
- `analysis_outputs/dlc_locomotion_full_set_52cm/sessions/*/*_centroid_trajectory.csv.gz`
- `analysis_outputs/dlc_locomotion_full_set_52cm/sessions/*/*.png`

Important physical-unit columns:

- `centroid_x_m`
- `centroid_y_m`
- `centroid_x_smooth_m`
- `centroid_y_smooth_m`
- `speed_m_per_sec`

Notes:

- By default, `no_use_videos/` sessions are excluded.
- `--arena-edge-length-cm 52` stores the cage floor edge length and enables
  meter-scale trajectory and velocity outputs.
- `--coverage-bin-size 0.025` bins normalized arena coordinates.
- `--coverage-bin-size-m 0.025` bins meter-coordinate occupancy in `2.5 cm`
  bins when meter columns are available.

### 5. Compare Implanted vs Unimplanted Groups

Use compact centroid trajectory files from the full-set run. The script treats
each session as the independent unit:

```powershell
python compare_locomotion_groups.py analysis_outputs\dlc_locomotion_full_set_52cm --manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv --output-dir analysis_outputs\dlc_locomotion_full_set_52cm\group_comparison
```

Example outputs:

- `group_locomotion_session_metrics.csv`
- `group_locomotion_summary.csv`
- `group_locomotion_tests.csv`
- `group_locomotion_report.json`
- `*_by_group.png`
- `pooled_group_plots/`

### 6. Rebuild Plots From Compact Centroid Trajectories

Use this when only plot style or binning needs adjustment, without reloading the
larger DLC-derived tables:

```powershell
python plot_centroid_trajectories.py analysis_outputs\dlc_locomotion_full_set_52cm --output-dir analysis_outputs\dlc_locomotion_full_set_52cm\centroid_plots_from_trajectory
```

The script reads the compact `*_centroid_trajectory.csv.gz` files and, when
available, uses `dlc_locomotion_parameters.json` from the same output root.

### 7. Optional Overlay QC

Generate an annotated preview for a selected session:

```powershell
python dlc_locomotion_pipeline.py --calibration-manifest analysis_outputs\arena_calibration_full_set\calibration_manifest.csv --coverage-bin-size 0.025 --session-filter front_camera_LS16_2026-05-16_3 --overlay-mode preview --overlay-seconds 60 --overlay-fps 20 --output-dir analysis_outputs\dlc_locomotion_overlay_qc
```

The overlay uses raw video pixels for display and timestamp matching between
`*_TS.npy` and DLC `frame_time` to avoid frame-index asynchrony.

## Legacy And Calibration Notes

For the earlier shared front-camera sessions, the accepted calibration is:

```text
analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration.json
```

To create a new shared calibration from a representative video:

```powershell
python arena_calibration.py calibrate --video Centimani_implanted\front_camera_LS18_2026-04-02_1_VIDEO.avi --output analysis_outputs\arena_calibration\shared_arena_calibration.json
```

The GUI asks for four rough arena corners, suggests edge points, then lets the
user edit and confirm the final edge points. Reviewed edge points are saved next
to the calibration JSON and are auto-loaded on later runs with the same output
path.

Validate a shared calibration on all main videos:

```powershell
python arena_calibration.py batch-validate --root . --calibration analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration.json --output-dir analysis_outputs\arena_calibration_batch_validation
```

Earlier decision: the original five shared front-camera sessions looked good
with the shared calibration, while
`Centimani_implanted/Imaging_source_bot01_2024-12-06_1` should receive a
separate calibration.

### Per-Video Calibration and Physical Scale

The cage floor edge length is currently:

```text
cage_floor_edge_length_cm = 52
```

The correction first maps the cage floor to normalized square arena coordinates.
Physical scaling is then applied as:

```text
centroid_x_m = centroid_x_normalized * 0.52
centroid_y_m = centroid_y_normalized * 0.52
speed_m_per_sec = speed_normalized_arena_side_per_sec * 0.52
```

This assumes the corrected square arena edge corresponds to the real cage floor
edge-to-edge distance of `52 cm`.

Preferred calibration workflow is now per-video rather than blind batch
application. For each selected video, load a default edge-points JSON, fine-tune
if needed, save a per-video calibration JSON, and write a manifest for the
locomotion pipeline.

Interactive command:

```powershell
python review_video_calibrations.py --default-edge-points-json analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration_edge_points.json --arena-edge-length-cm 52 --exclude-session-filter Imaging_source_bot01_2024-12-06_1 --output-dir analysis_outputs\arena_calibration_per_video --manifest analysis_outputs\arena_calibration_per_video\calibration_manifest.csv
```

For each video, the script prints `cage_floor_edge_length_cm = 52`, then prompts
for:

- edge-points JSON to load, defaulting to the provided
  `--default-edge-points-json`;
- output calibration JSON filename, defaulting to the per-session output folder.

Example non-GUI run using the previously reviewed edge points directly:

```powershell
python review_video_calibrations.py --default-edge-points-json analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration_edge_points.json --arena-edge-length-cm 52 --exclude-session-filter Imaging_source_bot01_2024-12-06_1 --output-dir analysis_outputs\arena_calibration_per_video_example --manifest analysis_outputs\arena_calibration_per_video_example\calibration_manifest.csv --no-gui
```

The script prints `cage_floor_edge_length_cm = 52` to the console and stores the
same value in the manifest. Each per-video folder contains calibration JSON,
edge-points JSON/CSV, diagnostics, and transform debug files.

Run locomotion from the per-video calibration manifest:

```powershell
python dlc_locomotion_pipeline.py --calibration-manifest analysis_outputs\arena_calibration_per_video_example\calibration_manifest.csv --arena-edge-length-cm 52 --coverage-bin-size 0.025 --output-dir analysis_outputs\dlc_locomotion_per_video_calibrated_52cm
```

The locomotion run prints `cage_floor_edge_length_cm = 52` and saves it in:

```text
analysis_outputs\dlc_locomotion_per_video_calibrated_52cm\processing_log.json
```

Important physical-unit outputs:

- `centroid_x_m`
- `centroid_y_m`
- `centroid_x_smooth_m`
- `centroid_y_smooth_m`
- `speed_m_per_sec`

The compact trajectory files include both normalized and meter columns.

### 8. Implanted vs Unimplanted Locomotion Comparison

Use `compare_locomotion_groups.py` to compare groups from compact centroid
trajectory files. The script treats each session as the independent unit, not
each frame.

Example run:

```powershell
python compare_locomotion_groups.py analysis_outputs\dlc_locomotion_per_video_calibrated_52cm --manifest analysis_outputs\arena_calibration_per_video_example\calibration_manifest.csv --output-dir analysis_outputs\dlc_locomotion_per_video_calibrated_52cm\group_comparison
```

Example outputs:

- `group_locomotion_session_metrics.csv`
- `group_locomotion_summary.csv`
- `group_locomotion_tests.csv`
- `group_locomotion_report.json`
- `*_by_group.png`
- `pooled_group_plots/pooled_velocity_implanted.png`
- `pooled_group_plots/pooled_velocity_unimplanted.png`
- `pooled_group_plots/pooled_velocity_groups_overlay.png`
- `pooled_group_plots/pooled_trajectory_implanted.png`
- `pooled_group_plots/pooled_trajectory_unimplanted.png`
- `pooled_group_plots/pooled_trajectory_groups_overlay.png`
- `pooled_group_plots/pooled_occupancy_implanted.png`
- `pooled_group_plots/pooled_occupancy_unimplanted.png`

Current example run:

- Sessions compared: `5`
- Groups: `implanted = 3`, `unimplanted = 2`
- Output folder:
  `analysis_outputs/dlc_locomotion_per_video_calibrated_52cm/group_comparison/`

## Video Metadata

All currently discovered `*_VIDEO.avi` files report the same AVI header format:

- Frame rate: `100 fps`
- Frame size: `530 x 510 px` (`W x H`)

The script records these columns in `analysis_outputs/session_summary.csv`:

- `video_width_px`
- `video_height_px`
- `video_fps_header`
- `video_total_frames_header`
- `video_duration_sec_header`

Current complete video/DLC/timestamp sessions without `_PROC`:

- `Centimani_implanted/Imaging_source_bot01_2024-12-06_1`
- `Centimani_implanted/no_use_videos/Imaging_source_Bot-1_2024-12-14_1`

Current excluded/incomplete session:

- `Centimani_implanted/no_use_videos/front_camera_LS16_2026-05-16_1`
  currently has only `*_VIDEO.avi`.

## First Full Summary Run

Run date: 2026-05-09

Command:

```powershell
python analyze_tracking_summary.py
```

That initial run completed without analysis errors and wrote outputs to
`analysis_outputs/`.

Generated QC figures:

- `analysis_outputs/plots/dlc_median_likelihood_heatmap.png`
- `analysis_outputs/plots/dlc_usable_fraction_heatmap.png`
- `analysis_outputs/plots/velocity_histograms.png`
- `analysis_outputs/plots/arena_occupancy_maps.png`

Generated report:

- `analysis_outputs/tracking_data_understanding_report.md`

## Scripts

### `inspect_free_roaming_folder.py`

Lightweight file inventory and file-content preview. Use it when first checking
whether files are present and readable.

```powershell
python inspect_free_roaming_folder.py --max-files 0
python inspect_free_roaming_folder.py --max-files 4 --sample-count 3 --load-proc-pickle
```

### `analyze_tracking_summary.py`

Exploratory data-understanding script for keypoint confidence, local motion,
velocity, and arena coverage.

```powershell
python analyze_tracking_summary.py
```

Outputs are written to `analysis_outputs/`:

- `session_summary.csv`
- `dlc_keypoint_summary.csv`
- `localmotion_velocity_summary.csv`
- `arena_coverage_summary.csv`
- `analysis_errors.csv`
- `overall_summary.json`
- `tracking_data_understanding_report.md`

### `dlc_locomotion_pipeline.py`

Robust DLC-only locomotion pipeline using a fixed-anatomy centroid from the five
most reliable trunk points.

Default command for the main analysis set:

```powershell
python dlc_locomotion_pipeline.py
```

By default, sessions under `no_use_videos/` are excluded. Include them only for
QC or explicit exploratory analysis:

```powershell
python dlc_locomotion_pipeline.py --include-no-use
```

Generate a lightweight annotated video preview for selected sessions:

```powershell
python dlc_locomotion_pipeline.py --session-filter front_camera_LS18_2026-04-02_1 --overlay-mode preview --overlay-seconds 60 --overlay-fps 20
```

Generate a full annotated overlay video when needed:

```powershell
python dlc_locomotion_pipeline.py --session-filter front_camera_LS18_2026-04-02_1 --overlay-mode full --overlay-fps 100
```

Primary outputs are written to `analysis_outputs/dlc_locomotion/`:

- `dlc_locomotion_session_summary.csv`
- `dlc_locomotion_parameters.json`
- `dlc_locomotion_outputs.json`
- `dlc_locomotion_errors.csv`
- `dlc_locomotion_report.md`
- `sessions/<session>/..._dlc_locomotion.parquet`
- `sessions/<session>/..._centroid_trajectory.csv.gz`
- `sessions/<session>/..._centroid_trajectory.png`
- `sessions/<session>/..._occupancy_heatmap.png`
- `sessions/<session>/..._velocity_histogram.png`
- optional `sessions/<session>/..._overlay.mp4`

Session output folders use a shortened stable name plus hash when needed. This
avoids Windows path-length failures for long session IDs while preserving the
full `session_id` inside the exported tables and summaries.

Earlier uncalibrated main run before the full dataset was added:

- Command: `python dlc_locomotion_pipeline.py`
- Sessions processed: `6` main DLC sessions, excluding `no_use_videos/`
- Error count: `0`
- Report: `analysis_outputs/dlc_locomotion/dlc_locomotion_report.md`

Earlier all-session smoke test before the full dataset was added:

- Command: `python dlc_locomotion_pipeline.py --include-no-use --no-plots --output-dir analysis_outputs/dlc_locomotion_all_sessions_test`
- Sessions processed: `11`
- Error count: `0`

Use a reviewed arena calibration file:

```powershell
python dlc_locomotion_pipeline.py --calibration-json analysis_outputs\arena_calibration\shared_arena_calibration.json
```

When a calibration file is provided, raw-pixel DLC quality control still happens
first. Cleaned keypoints are then transformed into corrected square arena
coordinates before centroid, velocity, and coverage are computed.

Earlier accepted shared-calibration locomotion run:

```powershell
python dlc_locomotion_pipeline.py --calibration-json analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration.json --coverage-bin-size 0.025 --session-filter "^(?!.*Imaging_source_bot01_2024-12-06_1).*" --output-dir analysis_outputs\dlc_locomotion_calibrated_shared
```

- Sessions processed: `5`.
- Excluded for separate calibration: `Centimani_implanted/Imaging_source_bot01_2024-12-06_1`.
- Error count: `0`.
- Corrected coordinate system: normalized square arena coordinates.
- Coverage bin size: `0.025` corrected arena units.
- Report: `analysis_outputs/dlc_locomotion_calibrated_shared/dlc_locomotion_report.md`.
- Compact centroid trajectory files:
  `analysis_outputs/dlc_locomotion_calibrated_shared/sessions/*/*_centroid_trajectory.csv.gz`.

### `plot_centroid_trajectories.py`

Plot-only helper that rebuilds trajectory, occupancy, and velocity plots from
the compact centroid trajectory files rather than the larger DLC-derived frame
tables.

Earlier plot-only run:

```powershell
python plot_centroid_trajectories.py analysis_outputs\dlc_locomotion_calibrated_shared --output-dir analysis_outputs\dlc_locomotion_calibrated_shared\centroid_plots_from_trajectory
```

The script automatically uses
`dlc_locomotion_parameters.json` from the locomotion output root when present,
including the calibrated coverage bin size.

### `arena_calibration.py`

Interactive calibration tool for mapping raw camera pixels to a corrected square
arena coordinate system.

Create a shared calibration from a representative video:

```powershell
python arena_calibration.py calibrate --video Centimani_implanted\front_camera_LS18_2026-04-02_1_VIDEO.avi --output analysis_outputs\arena_calibration\shared_arena_calibration.json
```

The calibration GUI is now two-stage:

1. Click rough corners in order: top-left, top-right, bottom-right,
   bottom-left.
2. The program uses those corners to suggest 30 points per arena edge from a
   local edge search.
3. Review and edit the proposed edge points before saving.

Corner selection controls:

- Left-click to place four rough corners.
- Drag a corner to adjust it.
- `backspace`: remove the last corner.
- `enter`: confirm corners.
- `q`: cancel.

Edge review controls:

- `1`, `2`, `3`, `4`: select `top`, `right`, `bottom`, `left` edge.
- Drag a point to move it.
- Right-click a point to delete it.
- `a`: add a point to the active edge.
- `enter`: confirm and save.
- `q`: cancel.

The reviewed edge feature points are saved separately next to the calibration
JSON:

- `<calibration_stem>_edge_points.json`
- `<calibration_stem>_edge_points.csv`

Diagnostic copies are also saved in the diagnostics folder. On the next
`calibrate` run with the same `--output`, the script automatically loads
`<calibration_stem>_edge_points.json` and uses those reviewed points instead of
running default auto-detection again. This makes iterative fine-tuning easier.

Validate a shared calibration on another session:

```powershell
python arena_calibration.py validate --video Centimani_implanted\front_camera_LS19_2026-04-02_3_VIDEO.avi --calibration analysis_outputs\arena_calibration\shared_arena_calibration.json --output-dir analysis_outputs\arena_calibration\validate_LS19
```

Batch-validate across main sessions:

```powershell
python arena_calibration.py batch-validate --root . --calibration analysis_outputs\arena_calibration\shared_arena_calibration.json --output-dir analysis_outputs\arena_calibration\batch_validate
```

Earlier shared-calibration batch validation used the working numerical-forward
full-frame rectification as the default diagnostic:

```powershell
python arena_calibration.py batch-validate --root . --calibration analysis_outputs\arena_calibration_inverse_compare\shared_arena_calibration.json --output-dir analysis_outputs\arena_calibration_batch_validation
```

This generated one folder per main video under
`analysis_outputs/arena_calibration_batch_validation/`. Each folder contains:

- `*_source_points.png`
- `*_warped_square.png`
- `*_rectified_full_frame.png` using numerical forward inversion
- inverse-method comparison images
- `*_validation.json`

Default target units are normalized square arena coordinates, `0-1` along each
side. This is the most compatible default because it does not require knowing
the physical arena size. If the side length is known, pass
`--arena-side-length` and `--target-units`, for example `cm`, to export
corrected coordinates directly in physical units.

## Current Data Interpretation

DLC files contain one pandas key, usually `/df_with_missing`. The pose table has
body-part columns with `x`, `y`, and `likelihood`, plus `frame_time` and
`pose_time`.

Detected DLC bodyparts from the initial inspection:

- `nose`
- `left_eye`
- `right_eye`
- `left_bar`
- `right_bar`
- `cable_base`
- `left_midside`
- `right_midside`
- `left_hip`
- `right_hip`
- `tail_base`
- `tail_end`

The `_PROC` files are Python pickles with these keys:

| Key | Type | Meaning In Current Analysis |
|---|---|---|
| `start_time` | scalar float | Recording/process start timestamp. |
| `frame_time` | 1D float array | Frame-associated timestamps in the processed record. |
| `time_stamp` | 1D float array | Processed sample timestamps used for velocity timing. |
| `step` | 1D int array | Processed sample index/step. |
| `center_x` | 1D float array | Processed animal center x-position in pixels. |
| `center_y` | 1D float array | Processed animal center y-position in pixels. |
| `heading_direction` | 1D float array | Processed heading direction in degrees. |
| `head_angle` | 1D float array | Processed head angle in radians. |
| `signal` | 1D float array | External/behavioral signal trace. |
| `signal_time` | 1D float array | Timestamps for `signal`. |
| `config` | dict | Acquisition/configuration metadata. |
| `signal_summary` | dict | Summary statistics for the signal trace. |

For the first-pass local-motion summary, `center_x` and `center_y` are treated
as the animal position in pixels. `(0, 0)` positions are excluded as invalid
placeholders.

## Can We Analyze With Only DLC?

Yes for several core analyses, but not exactly the same analysis currently
implemented from `_PROC`.

What `_DLC.hdf5` alone supports:

- Keypoint detection summary.
- Per-bodypart likelihood and usable-frame fraction.
- Body posture and body-axis metrics from reliable points.
- DLC-derived animal position if we define a body center, for example the
  centroid of `left_hip`, `right_hip`, `left_midside`, `right_midside`, and
  `tail_base`.
- DLC-derived velocity and arena occupancy, using DLC `frame_time` or
  `pose_time`, after confidence filtering and gap interpolation.

What `_PROC` adds:

- Already-computed `center_x` and `center_y`.
- Already-computed `heading_direction` and `head_angle`.
- Signal and signal timing arrays.
- A processed local-motion representation that avoids rebuilding center and
  heading from raw DLC keypoints.

Important distinction: the velocity and arena coverage summaries generated so
far use `_PROC` center positions, not DLC-only positions. To make the pipeline
DLC-only, the next step is to implement a reproducible DLC-derived center,
filter low-confidence points, interpolate short gaps, and compare that trajectory
against `_PROC` center for the sessions where both exist. In the current
full-set inventory, `13` sessions have `_PROC` files.

## DLC-Only Locomotion Pipeline Assumptions

The implemented DLC-only locomotion pipeline intentionally avoids a
variable-point centroid. The centroid is never calculated from "whatever points
are present" in a frame, because that would skew the body center when one side
or one body part drops out.

Reliable trunk points used for locomotion:

- `tail_base`
- `left_hip`
- `right_hip`
- `left_midside`
- `right_midside`

Default parameters:

- DLC likelihood threshold: `0.8`
- Maximum keypoint interpolation gap: `50 frames` (`0.5 s` at `100 fps`)
- Keypoint jump rejection threshold: `3000 px/s`
- Centroid speed artifact cap: `2000 px/s`
- Median filter window: `5 frames`
- Savitzky-Golay smoothing window: `21 frames`
- Coverage bin size: `25 px`

Robust missing-data handling:

- Each of the five keypoints is filtered independently by confidence, finite
  coordinates, frame bounds, and isolated jump artifacts.
- Each keypoint is interpolated independently for short gaps only.
- Long gaps remain missing.
- The centroid is the non-weighted mean of all five cleaned keypoints.
- A centroid is produced only when all five cleaned keypoints are available,
  either observed or short-gap interpolated.
- Smoothing is applied segment-by-segment within contiguous valid centroid
  samples. Missing samples are not included in any smoothing window and filters
  never cross missing gaps.

Saved per-frame data:

- Raw-video frame mapping columns: `video_frame_index` and
  `video_time_error_sec`.
- Raw x/y/likelihood for each of the five keypoints.
- Valid and interpolated flags for each keypoint.
- Cleaned x/y for each keypoint.
- Raw complete-frame centroid.
- Cleaned centroid.
- Smoothed centroid.
- Speed in px/s.
- `_PROC` comparison columns when `_PROC` exists.

This per-frame table is saved as Parquet when possible, with gzip CSV fallback.

## Overlay Synchronization Note

The raw AVI frame index and DLC row index are not the same. DLC `frame_time`
values align to a subset of the raw video timestamp array in `*_TS.npy`.
Therefore, overlaying DLC row `i` on raw video frame `i` causes visible
asynchronization.

The DLC locomotion pipeline now maps each DLC row to the nearest raw video
timestamp:

- `video_frame_index`: nearest raw video frame for that DLC row.
- `video_time_error_sec`: timestamp mismatch after matching.

For annotated videos, the overlay uses raw video frame timestamps and selects
the nearest DLC row within `overlay_time_tolerance_sec`, default `0.03 s`.
The overlay text shows both the raw video frame and DLC row plus the timing
offset in milliseconds.

Example finding from `front_camera_LS18_2026-04-02_1`:

- DLC row `0` maps to raw video frame `6`.
- DLC row `183995` maps to raw video frame `216735`.
- This explains why 1:1 row/frame overlay was asynchronous.

## Camera Distortion And Tilt Correction Pipeline

This calibration pipeline has been implemented in `arena_calibration.py`.

Goal: map raw camera pixel coordinates into a corrected square arena coordinate
system so trajectories, velocities, and coverage can be compared in common
arena coordinates.

Implemented workflow:

1. Select one representative calibration frame, usually the first clear frame
   of each video or a shared reference frame for a fixed camera setup.
2. User marks four rough corners in a matplotlib GUI.
3. Suggest arena boundary points from a corner-guided local edge search.
   - The rough corners define the expected arena quadrilateral.
   - Each arena side is sampled at `30` positions by default.
   - At each position, the tool searches along the local edge normal.
   - The edge score combines local gradient strength, expected floor-to-wall
     brightness contrast, and distance from the rough edge.
   - The result is smoothed per edge and obvious outliers are down-weighted.
4. User edits the suggested points in a matplotlib GUI.
   - Points that are clearly off can be deleted.
   - Decent points can be dragged into position.
   - Extra points can be added if needed.
5. Define the target square coordinate system.
   - Default target: normalized square coordinates from `0` to `1`.
   - Optional target: physical units if `--arena-side-length` is known.
6. Fit the spatial transform.
   - The implemented transform is a boundary-constrained polynomial mapping
     from raw pixels to the corrected square.
   - This is more appropriate than a four-corner homography when arena edges
     are visibly curved.
   - The transform is empirical and should be validated visually for each
     camera setup.
   - Given refined edge points, top-edge points are constrained to `y = 0`,
     bottom to `y = 1`, left to `x = 0`, and right to `x = 1` by default.
   - The along-edge coordinate is not assumed to be evenly spaced by point
     index. It is assigned from cumulative arc length along each reviewed edge,
     preserving the relative spacing after user edits.
   - A least-squares polynomial transform is fit from raw image coordinates to
     these corrected square coordinates.
   - A final projective normalization matrix, saved as `post_homography`, is
     then fit from best-fit edge-line intersections to the ideal square. This
     handles residual global rotation/shear/perspective after curved-edge
     correction.
   - The applied mapping is therefore:
     `raw pixels -> polynomial correction -> post_homography -> corrected arena`.
7. Validate the transform visually.
   - `*_warped_square.png` shows the arena-square crop: the output covers only
     corrected arena coordinates from `0` to `1`, so it intentionally crops to
     the arena.
   - `*_rectified_full_frame.png` shows the maximum practical transformed
     boundary of the original frame. It transforms the source-frame border,
     computes the bounding box in corrected coordinates, and draws the corrected
     square arena outline.
   - Confirm arena edges are straight and square.
   - Confirm animal scale and location look plausible across several frames.
8. Apply the transform consistently.
   - The implemented locomotion pipeline performs raw tracking QC in camera
     pixels, transforms cleaned keypoints, then recomputes
     centroid/velocity/coverage in corrected arena coordinates.
9. Save calibration outputs.
   - Marked source points.
   - Separate reusable edge-point JSON/CSV files.
   - Target square points.
   - Polynomial transform coefficients.
   - Units and arena side length.
   - Diagnostic source-point, arena-crop, and full-frame rectified images.

Default edge-detection parameters:

- `points_per_edge`: `30`
- `search_radius_px`: `35`
- `profile_half_width_px`: `3`
- `gaussian_sigma`: `1.5`
- `spline_smoothing`: `8`
- `outlier_threshold_px`: `12`

Important assumptions:

- The arena is planar.
- The arena is square in real-world coordinates.
- Camera and arena geometry do not move during a session.
- A four-corner homography corrects perspective/camera tilt but does not fully
  correct radial lens distortion if edges are visibly curved; this is why the
  implemented tool uses many boundary points and a polynomial transform.
- If the same camera setup is fixed across sessions, one calibration may be
  reusable, but this should be verified per session.

Recommended staged approach:

1. Calibrate one representative session with the GUI.
2. Validate the shared calibration on each session.
3. Recalibrate any session where the arena/camera appears shifted.
4. Once accepted, run `dlc_locomotion_pipeline.py --calibration-json ...`.
5. Report locomotion in corrected arena units rather than raw pixels.

## First-Pass Findings

The original first-pass findings below were generated before the full dataset
was added. They covered `11` discoverable sessions with video, timestamp, and
DLC files, of which `9` also had `_PROC` files. The full-set inventory is now
larger; rerun the quickstart summary command before treating these values as
current.

Session durations from timestamp arrays range from short test recordings
(`~8-42 s`) to full recordings (`~2175-3009 s`). This matters because coverage
and velocity distributions should be interpreted separately for short versus
long sessions.

DLC keypoint quality at likelihood threshold `0.8`:

- Strong body tracking: `tail_base`, `left_hip`, `right_hip`,
  `left_midside`, and `right_midside` have high median likelihoods and high
  usable fractions.
- Moderate anterior tracking: `nose`, `left_eye`, and `right_eye` are usable
  but have lower low-percentile confidence, so they likely need filtering or
  interpolation for head-orientation analyses.
- Weak environmental/cable markers: `left_bar`, `right_bar`, and `cable_base`
  have low median likelihood and low usable fractions, so they should not be
  treated as reliable animal body points.
- `tail_end` is mostly good but less reliable than `tail_base`.

Local-motion summaries from `_PROC` center positions:

- Valid processed center positions are essentially complete across sessions
  after excluding `(0, 0)`.
- Median speed is typically `~18-41 px/s`, except
  `Comparison_unimplanted/front_camera_C57_0008_C1_6673_2026-05-03_1`, whose
  median speed is `0 px/s`. That session needs visual QC to determine whether
  the animal was often stationary, position estimates were quantized/repeated,
  or the processed center signal needs smoothing/recalculation.
- P95 speeds are roughly `~196-253 px/s`. The current script caps
  instantaneous speeds above `2000 px/s` as likely artifacts.

Arena coverage from processed center positions:

- Full-length sessions generally cover most of the observed bounding box
  (`~0.86-0.91` coverage fraction with `25 px` bins).
- Short sessions have lower coverage, as expected from their duration
  (`~0.30-0.65`).
- Coverage is currently relative to each session's observed bounding box. It is
  useful for within-session QC, but a fixed arena mask or calibration is needed
  for rigorous cross-session comparison.

## First-Pass QC Rules

- DLC usable frames are summarized with a default likelihood threshold of `0.8`.
- Velocity is computed in pixels per second from processed center positions and timestamps.
- Time steps below `0.001` seconds are ignored.
- Speeds above `2000 px/s` are treated as likely artifacts for this initial summary.
- Arena coverage is measured as occupied bins in the observed position bounding box.
- Coverage currently uses a `25 px` bin size and is not yet normalized to a fixed arena mask.

## Pipeline Questions To Resolve

- Should the animal position be based on `_PROC` center, DLC body centroid, tail base, or a filtered body-axis midpoint?
- What DLC confidence threshold should be used for final filtering?
- Should low-confidence keypoints be interpolated, masked, or replaced using neighboring body geometry?
- Do we have a stable arena calibration or fixed arena mask for cross-session coverage comparison?
- Should velocity be smoothed before distribution summaries?
- How should short test recordings be handled relative to full recordings?

## Proposed Next Pipeline Layer

1. Add per-session visual QC panels combining DLC confidence, center trajectory,
   occupancy, and velocity histogram.
2. Define a canonical animal position. The current best candidate is processed
   `_PROC` center for locomotion, with DLC body-axis points used for posture.
3. Establish DLC filtering rules:
   - retain reliable body points directly,
   - interpolate short gaps for moderate points,
   - exclude weak bar/cable markers from animal tracking summaries.
4. Add an arena calibration step so coverage can be reported in common arena
   coordinates instead of per-session observed bounding boxes.
5. Add export tables for downstream analysis: per-frame clean position,
   per-frame speed, per-session velocity distribution, and per-session coverage
   metrics.
