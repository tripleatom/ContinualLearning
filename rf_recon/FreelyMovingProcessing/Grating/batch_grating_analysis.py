"""
Batch grating analysis runner.

Default workflow:
1. Export the current day from grating_config.py with GratingExport.py logic.
2. Run LDA, SVM, embedding, and tuning-curve analyses on the exported pkl.
3. Run OSI_distribution.py on the tuning_statistics.csv from tuning curves.

You can also skip export and run downstream analyses on an existing pkl:
    python batch_grating_analysis.py --pkl path/to/grating_data_merged.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _output_base_from_sortout(sortout_folder: Path) -> Path:
    return sortout_folder.parent if sortout_folder.name == "curated_analyzer" else sortout_folder


def _save_pkl(neural_data: dict, filepath: Path) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving exported grating data to: {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(neural_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def export_from_grating_config() -> Path:
    """Export current grating_config.py session and return the merged pkl path."""
    from rf_recon.FreelyMovingProcessing.Grating.GratingExport import (
        _normalize_passive_window,
        _rec_folders_for_task_files,
        extract_grating_neural_data_for_embedding,
        merge_grating_neural_data,
    )
    from rf_recon.FreelyMovingProcessing.Grating.grating_config import (
        ANIMAL_ID,
        EXPERIMENT_DATE,
        SORTOUT_FOLDER,
        PASSIVE_START,
        PASSIVE_END,
        PASSIVE_WINDOWS,
    )
    from rf_recon.FreelyMovingProcessing.Grating.grating_utils import load_session_paths

    rec_folders, passive_log_paths = load_session_paths(ANIMAL_ID, EXPERIMENT_DATE)
    if not rec_folders:
        raise RuntimeError(f"No recording folders found for {ANIMAL_ID} {EXPERIMENT_DATE}")
    if not passive_log_paths:
        raise RuntimeError(f"No grating task files found for {ANIMAL_ID} {EXPERIMENT_DATE}")

    if PASSIVE_WINDOWS is None:
        passive_windows = [
            {"passive_start": PASSIVE_START, "passive_end": PASSIVE_END}
            for _ in passive_log_paths
        ]
    else:
        if len(PASSIVE_WINDOWS) != len(passive_log_paths):
            raise ValueError(
                f"PASSIVE_WINDOWS has {len(PASSIVE_WINDOWS)} entries, but "
                f"{len(passive_log_paths)} task file(s) were found in the CSV."
            )
        passive_windows = [_normalize_passive_window(w) for w in PASSIVE_WINDOWS]

    missing = [p for p in passive_log_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Task file does not exist: " + str(missing[0]))

    sortout_folder = Path(SORTOUT_FOLDER)
    curated_analyzer_path = (
        sortout_folder if sortout_folder.name == "curated_analyzer"
        else sortout_folder / "curated_analyzer"
    )
    if not curated_analyzer_path.exists():
        raise FileNotFoundError(f"curated_analyzer path does not exist: {curated_analyzer_path}")

    print("=" * 72)
    print("EXPORT")
    print("=" * 72)
    print(f"Animal/date: {ANIMAL_ID} {EXPERIMENT_DATE}")
    print(f"Sortout folder: {sortout_folder}")
    print(f"Task files: {len(passive_log_paths)}")

    rec_folders_for_tasks = _rec_folders_for_task_files(rec_folders, passive_log_paths)
    rec_folder = rec_folders_for_tasks[0]

    all_data = []
    for task_file_path, passive_window, rec_folder_for_task in zip(
        passive_log_paths, passive_windows, rec_folders_for_tasks
    ):
        print("\n" + "-" * 72)
        print(f"Extracting: {task_file_path.stem}")
        print(
            f"Passive window: start={passive_window['passive_start']} "
            f"end={passive_window['passive_end']}"
        )
        neural_data = extract_grating_neural_data_for_embedding(
            rec_folder_for_task,
            task_file_path,
            sortout_folder,
            passive_start=passive_window["passive_start"],
            passive_end=passive_window["passive_end"],
        )
        if neural_data is None:
            raise RuntimeError(f"Failed to extract {task_file_path.stem}")
        all_data.append(neural_data)

    merged = merge_grating_neural_data(all_data)
    rec_name = rec_folder.name.replace(".rec", "")
    output_dir = _output_base_from_sortout(sortout_folder) / "passive_embedding_analysis"
    output_pkl = output_dir / f"{rec_name}_grating_data_merged.pkl"
    _save_pkl(merged, output_pkl)

    print(f"Exported units: {len(merged['spike_data'])}")
    print(f"Exported trials: {merged['metadata']['n_trials']}")
    print(f"Orientations: {merged['trial_info']['unique_orientations']}")
    return output_pkl


def _run_stage(stage_name: str, fn, continue_on_error: bool):
    print("\n" + "=" * 72)
    print(stage_name.upper())
    print("=" * 72)
    try:
        return fn()
    except Exception as exc:
        print(f"{stage_name} failed: {exc}")
        traceback.print_exc()
        if not continue_on_error:
            raise
        return None
    finally:
        plt.close("all")


def run_downstream(
    pkl_path: Path,
    time_window: tuple[float, float],
    svm_kernel: str,
    svm_c: float,
    svm_gamma,
    create_tuning_summary: bool,
    continue_on_error: bool,
) -> dict:
    """Run all downstream analyses from an exported grating pkl."""
    import GratingEmbedding
    import GratingLDA
    import GratingSVM
    import GratingTuningCurve
    import OSI_distribution

    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Input pkl does not exist: {pkl_path}")

    analysis_prefix = pkl_path.with_suffix("")
    tuning_folder = pkl_path.parent / f"{pkl_path.stem}_tuning_curves"
    tuning_csv = tuning_folder / "tuning_statistics.csv"
    osi_png = tuning_folder / "OSI_distribution.png"

    outputs = {
        "pkl": pkl_path,
        "analysis_prefix": analysis_prefix,
        "tuning_folder": tuning_folder,
        "tuning_csv": tuning_csv,
        "osi_png": osi_png,
    }

    _run_stage(
        "LDA",
        lambda: GratingLDA.run_analysis(
            data_path=pkl_path,
            time_window=time_window,
            save_plots=True,
            output_path=analysis_prefix,
        ),
        continue_on_error,
    )

    _run_stage(
        "SVM",
        lambda: GratingSVM.run_analysis(
            data_path=pkl_path,
            time_window=time_window,
            save_plots=True,
            output_path=analysis_prefix,
            kernel=svm_kernel,
            C=svm_c,
            gamma=svm_gamma,
        ),
        continue_on_error,
    )

    _run_stage(
        "Embedding",
        lambda: GratingEmbedding.run_analysis(
            data_path=pkl_path,
            time_window=time_window,
            save_plots=True,
            output_path=analysis_prefix,
        ),
        continue_on_error,
    )

    _run_stage(
        "Tuning curves",
        lambda: GratingTuningCurve.generate_tuning_curves(
            data_path=pkl_path,
            time_window=time_window,
            output_folder=tuning_folder,
            create_summary=create_tuning_summary,
        ),
        continue_on_error,
    )

    if not tuning_csv.exists():
        msg = f"Cannot run OSI distribution; tuning CSV does not exist: {tuning_csv}"
        if continue_on_error:
            print(msg)
        else:
            raise FileNotFoundError(msg)
    else:
        _run_stage(
            "OSI distribution",
            lambda: OSI_distribution.plot_osi_distribution(
                csv_path=tuning_csv,
                save_path=osi_png,
            ),
            continue_on_error,
        )

    return outputs


def _parse_gamma(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export grating data, then run LDA, SVM, embedding, tuning curves, "
            "and OSI distribution."
        )
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="Existing exported grating pkl. If provided, export is skipped.",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=0.05,
        help="Analysis window start in seconds relative to stimulus onset.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=1.0,
        help="Analysis window end in seconds relative to stimulus onset.",
    )
    parser.add_argument("--svm-kernel", default="rbf", help="SVM kernel.")
    parser.add_argument("--svm-c", type=float, default=1.0, help="SVM C parameter.")
    parser.add_argument(
        "--svm-gamma",
        default="scale",
        help="SVM gamma parameter: scale, auto, or a number.",
    )
    parser.add_argument(
        "--no-tuning-summary",
        action="store_true",
        help="Skip multi-page tuning curve summary figures.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later stages if one downstream stage fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_window = (args.t0, args.t1)
    svm_gamma = _parse_gamma(args.svm_gamma)

    pkl_path = args.pkl if args.pkl is not None else export_from_grating_config()

    outputs = run_downstream(
        pkl_path=pkl_path,
        time_window=time_window,
        svm_kernel=args.svm_kernel,
        svm_c=args.svm_c,
        svm_gamma=svm_gamma,
        create_tuning_summary=not args.no_tuning_summary,
        continue_on_error=args.continue_on_error,
    )

    print("\n" + "=" * 72)
    print("BATCH COMPLETE")
    print("=" * 72)
    print(f"Input pkl: {outputs['pkl']}")
    print(f"Analysis prefix: {outputs['analysis_prefix']}")
    print(f"Tuning folder: {outputs['tuning_folder']}")
    print(f"Tuning CSV: {outputs['tuning_csv']}")
    print(f"OSI plot: {outputs['osi_png']}")


if __name__ == "__main__":
    main()
