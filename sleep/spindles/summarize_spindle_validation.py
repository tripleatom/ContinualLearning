"""Write review indexes for the spindle audit, beside the band-power outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402  (sleep/ subfolders + repo root on sys.path)
import json


def main():
    registry = json.loads((sleep_paths.SLEEP_DIR / 'sleep_day_configs.json').read_text())
    for day in ('260729', '260730'):
        folder = Path(registry['CnL46_' + day]['rec_folder']) / 'low_freq/v1_spindles/validation'
        lines = ['# Spindle candidate audit: CnL46 ' + day, '',
                 'The plots are diagnostic evidence, not proof that every candidate is a physiological spindle. '
                 'Inspection of July 29 shanks 0 and 5 shows sharp broadband transients coinciding with some detections. '
                 'The original amplitude-only detector can include these. Treat its counts as candidate counts.', '',
                 '## Reading the examples', '',
                 'Each file shows three events selected at the 10th, 50th and 90th percentiles of peak amplitude '
                 '(not hand-picked for appearance). Green lines mark the saved onset and offset. Rows show:', '',
                 '1. Saved broadband LFP, with its median removed; this is the existing preprocessed LFP, before the spindle-band filter.',
                 '2. The 9–17 Hz signal and smoothed Hilbert amplitude.',
                 '3. The reproduced NREM amplitude z-score, with boundary and peak thresholds.',
                 '4. A spectrogram computed from broadband LFP (0.5-second windows). Dashed lines mark 9 and 17 Hz. Color scales are local, not comparable across panels.',
                 '5. The broadband event spectrum versus a nearby, equal-duration, non-event NREM control.', '',
                 'Look for repeated oscillatory cycles already supported by the broadband signal, a sustained frequency-localized burst, '
                 'and a rising/falling envelope. A single sharp deflection followed by oscillation only in the filtered signal is concerning. '
                 'A threshold crossing or band-filtered oscillation alone cannot distinguish these cases.', '',
                 '## Methods and limits', '',
                 '- Every audited event was checked against the clean NREM mask and its saved peak z-score was reproduced to within 0.0001.',
                 '- Controls are the nearest eligible clean-NREM windows 3–30 seconds before or after an event, of equal duration, avoiding other candidates by one second. Controls may be reused.',
                 '- Spectral peaks are measured on broadband LFP over each detected interval, searching 6–30 Hz. Short intervals have limited frequency resolution; zero-padding does not improve that resolution.',
                 '- Sigma enrichment is expected because detection selects high sigma amplitude. It is not an independent significance test or an estimated accuracy.',
                 '- Characteristics plots show duration, broadband spectral peaks, normalized event/control spectra, and incidence per clean NREM minute in 2-minute bins. Bins with less than 15 seconds of clean NREM are omitted.',
                 '- Shared events across shanks are not deduplicated. Probe coordinates do not identify a cortical layer.',
                 '- July 30 NREM is provisional and based on shank 0; post-sleep has no tracking. Original detections have not been altered by this audit.', '',
                 '## Results and plots', '',
                 '| Recording | Candidates | Median duration (s) | Raw peak in 9–17 Hz | Median sigma/control | Plots |',
                 '|---|---:|---:|---:|---:|---|']
        for path in sorted(folder.glob('*_audit.json')):
            data = json.loads(path.read_text())
            stem = data['file']
            lines.append(f'| {stem} | {data["n"]} | {data["median_duration_s"]:.3f} | '
                         f'{100 * data["raw_peak_in_9_17_fraction"]:.1f}% | {data["median_sigma_vs_control"]:.2f} | '
                         f'[Examples]({stem}_examples.png) · [Characteristics]({stem}_characteristics.png) · [Metrics]({stem}_event_metrics.csv) |')
        (folder / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(folder / 'README.md')


if __name__ == '__main__':
    main()
