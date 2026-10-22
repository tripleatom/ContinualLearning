# Tuning similarity vs fine-timescale synchrony in NREM UP states

Tests one hypothesis:

> During NREM UP states, pairs of neurons with more similar visual tuning fire
> spikes that are **more precisely aligned** (within ±5–10 ms) than expected from
> each neuron's own firing pattern, after accounting for firing rate, recording
> distance and cell type.

A raw cross-correlogram peak cannot answer this. Inside an UP state every neuron
fires more, so shared changes in firing probability put spikes near each other
for nearly every pair. The **jitter correction** removes that part and keeps
only alignment finer than the jitter width.

## Files

| file | what it does |
| --- | --- |
| `tc_similarity.py` | Tuning similarity per pair: Δ preferred orientation (circular, 0° = 180°, so 0–90°) and cross-validated tuning-curve correlation. Also per-pair covariates (same shank, depth/lateral offset, cell-type pair). |
| `tc_upstates.py` | Sleep spikes; NREM UP windows from MUA ON states; restriction to scored NREM; the clock check. |
| `tc_jitter.py` | Coincidence counts, jitter surrogates, excess synchrony, jitter-corrected CCG. |
| `tc_stats.py` | Pair-level regression with a unit-shuffle p-value; pre vs post. |
| `run_tuning_coupling.py` | Runs everything for one session and writes figures, CSV and pkl. |
| `tc_selftest.py` | Synthetic data with known answers. |
| `cofiring.py` | Descriptive co-firing measures: pooled CCG with baselines, spike-count correlation, population coupling, STTC, UP-aligned rates, spike spectrum. |
| `describe_cofiring.py` | Figure pages describing co-firing for one session (no tests). |
| `plot_pair_ccgs.py` | Cross-correlograms of individual pairs: example, gallery, unit grid, all-pairs heatmap, artifact flags. |

## Describing co-firing first (no tests)

`describe_cofiring.py` (measures in `cofiring.py`) shows what co-firing looks
like before any hypothesis is tested. It uses every sorted unit with ≥ 50
spikes in UP states and writes five pages per sleep block to `<sortout>/cofiring/`:

| page | shows |
| --- | --- |
| 0 data check | fine CCG vs jitter for same- and cross-shank pairs (sorting collisions, artifacts); cross-unit part of the population spike spectrum |
| A UP states | every unit's rate around UP onset and offset, onset latency vs depth, population coupling |
| B CCG | pooled CCG at ±1 s / ±100 ms / ±15 ms against three baselines (rate only, UP membership, 50 ms jitter), and the split into UP-membership, slow within-UP and fine co-firing |
| C timescale | spike-count correlation vs bin width (1–100 ms), all NREM vs inside UP vs UP-membership baseline; same vs cross shank |
| D STTC | spike time tiling coefficient inside UP states vs Δt, pair matrix by shank/depth, rate dependence |
| E UP vs whole | r_sc, STTC and population coupling for the whole block (incl. wake), all NREM and UP states only; per-pair UP-vs-whole scatter; whether the same pairs rank high in each (Spearman ρ vs timescale) |

```bash
python describe_cofiring.py                  # grating_config session, pre + post
python describe_cofiring.py --date 260728 --blocks post
```

`plot_pair_ccgs.py` shows CCGs **pair by pair**: B's firing rate (Hz) around
A's spikes, with B's mean rate and the 50 ms-jitter expectation for reference.
It writes four pages per block to `<sortout>/cofiring/pair_ccgs/`:

- an annotated example pair ("how to read a CCG")
- a gallery of strongest / strongest cross-shank / typical / random pairs
- an 8-unit grid (like Phy's correlogram view, autocorrelograms on the diagonal)
- a heatmap of every pair, one row each

Pairs whose shape suggests a problem are shown in a separate "check" row and
kept out of the others. A *zero-lag spike* is the same event detected in both
units. A *split-like* CCG looks like an autocorrelogram: one neuron sorted as
two units. The CCGs are cached (`.npz`), so `--reuse --units a,b,c` re-plots
any units in seconds.

```bash
python plot_pair_ccgs.py --blocks post                 # all NREM, ~2.5 min per block
python plot_pair_ccgs.py --blocks post --scope up      # UP states only
python plot_pair_ccgs.py --blocks post --reuse --units shank1_unit7,shank2_unit48,shank3_unit39
python plot_pair_ccgs.py --scope whole                 # whole day, ~30 min
python plot_pair_ccgs.py --scope whole --recordings presleep,postsleep
```

`--scope whole` reads every unit's full spike train straight from each
shank's `shank{N}/sorting_results_*/sorting_analyzer`. Those spike trains are
in the day's concatenated sorter space, so each recording's sample range comes
from `conversion_list.txt`. Recordings are kept apart, so no lag or jitter
window spans the join between two files. Across the whole day there are no UP
states, so the orange baseline there is plain 50 ms jitter within each
recording: the co-firing expected from *any* shared rate change slower than
50 ms (behavioural state, arousal, UP/DOWN).

Spike times are integer 30 kHz samples. All lag binning therefore uses
whole-sample bin widths (enforced), and jitter moves spikes by whole samples.
Otherwise alternate CCG bins hold different numbers of sample lags and the
CCG zig-zags like a periodic artifact.

## Run

```bash
python run_tuning_coupling.py --check        # which inputs exist
python run_tuning_coupling.py --self-test    # 15 known-answer checks, ~1 min
python run_tuning_coupling.py                # grating_config session, pre + post
python run_tuning_coupling.py --date 260728  # another day
```

Use the `ms10` conda env. With default settings a block takes roughly 10–20
minutes, mostly surrogates and unit shuffles.

Outputs go to `<sortout>/tuning_coupling_jitter/`:

- `jitter_sync_<block>_w5_D50_*.png`: the detailed figure for the headline setting
- `jitter_sync_beta_vs_D_*.png`: the tuning coefficient for every w and D, for pre, post and post − pre
- `jitter_sync_pairs_<block>_*.csv`: one row per pair, with every measure
- `jitter_sync_results_*.pkl`: tables, fitted models (with null distributions), UP-state summary and CCGs

## The analysis, step by step

**1. UP windows.** MUA ON windows from `sleep/mua_onoff/detect_off_states.py` are
combined across shanks: an UP window is a stretch where ≥ 50% of shanks are ON
(`--min-shank-fraction`, via `sleep_states.load_mua_on_intervals`). These windows are intersected with scored NREM
(`*_sleep_periods.pkl`) and must pass the clock check: sorted units fire ≥ 1.25×
more inside UP than in DOWN time within NREM (`--min-alignment-ratio`). UP
windows come from channel MUA rather than from the sorted units being tested, so
the units' co-firing is not built into the windows.

**2. Observed coincidences.** For each pair, the analysis counts spike pairs
with `exclude ≤ |lag| ≤ w`, for w = 5 and 10 ms (`--coinc-ms`). Only pairs in
the *same* UP window are counted: each window gets its own stretch of time,
separated from the next by 1 s of empty time.

**3. Jitter surrogates.** Each UP window is cut into cells of width D, starting
at UP onset (`--jitter-ms`, default 25, 50 and 100 ms; the last cell ends at UP
offset). In each surrogate, every spike of both neurons is moved to a random
time inside its own cell. Each neuron keeps its exact spike count in every
cell, so its rate, its per-UP excitability and its time course relative to UP
onset (at resolution D) are all preserved. Only alignment finer than D is
destroyed. The default is 500 surrogates.

**4. Excess synchrony.** Observed count minus the mean count across surrogates.
The tested quantity is

```text
excess_norm = (observed − jitter mean) / sqrt(n_i · n_j)
```

i.e. excess coincidences per spike. It is comparable across pairs with
different rates and across blocks with different amounts of UP time. The CSV
also has the raw `excess`, `excess_z` (in surrogate SDs) and a pair-level
`p_jitter`.

**5. The tuning test.**

```text
excess_norm ~ tuning similarity + log geometric-mean rate + |log rate ratio|
              + same shank + |Δdepth| + |Δx| + cell-type pair
```

The model is fitted separately with Δ preferred orientation (hypothesis:
β < 0) and with tuning-curve correlation (hypothesis: β > 0). The p-value
comes from shuffling tuning across units, 1000 times. Each unit appears in
many pairs, so pairs are not independent and ordinary regression p-values
would be far too small. The unit shuffle keeps every unit's rate, position
and cell type, and scrambles only which tuning curve it has.

**6. Pre vs post.** The same model is fitted on the within-pair change
(post − pre). Anything stable about a pair cancels, so a slope there is the
experience-dependent claim.

## Choosing w and D

- **w** is the precision being asked about (±5 ms ≈ monosynaptic or common-input
  timescale; ±10 ms is looser).
- **D** decides what counts as "precise". Co-modulation slower than D stays in
  the surrogates and is *not* excess; anything faster than D *is* excess. D must
  be much wider than w (the code warns when D < 2w). An effect that exists only
  at large D is really about 20–100 ms co-modulation, not precise spike timing.
  That is why every D is reported (`jitter_sync_beta_vs_D`) rather than one.

## Spike-sorting caveats

- **Collisions.** Kilosort rarely detects two units on nearby channels within
  about 1 ms of each other, which carves a dip at 0 lag. Lags below
  `--exclude-ms` (default 1 ms) are therefore never counted, for any pair.
- **Duplicates.** A unit split into two gives a huge 0-lag peak. A pair is
  flagged `possible_duplicate` when > 20% of the sparser unit's spikes have a
  partner within 0.5 ms. Flagged pairs stay in the CSV but are left out of
  every test.

## Validation (`--self-test`)

| check | what is planted | what must come out |
| --- | --- | --- |
| mechanics | small hand-made spike trains | counts match brute force; jitter never moves a spike out of its UP window or cell; nothing counted across UP windows; sample-grid spikes give a flat fine CCG |
| `slow_common` | shared per-UP gain + within-UP envelope | excess ≈ 0, no tuning effect, the planted duplicate unit flagged |
| `mid_common` | shared ~20 ms rate fluctuation | excess for nearly every pair, growing with D; no tuning effect |
| calibration | 40 random tuning sets on `mid_common` data | p < 0.05 in ≤ 12.5% of draws |
| `effect` | ±1.5 ms synchronous events among similarly tuned units | β > 0 on tuning correlation, β < 0 on Δ preferred orientation |

## Geometry caveat

The channel map gives x ∈ {0, 300, 600, 900} µm for eight shanks, so shanks
0/4, 1/5, 2/6 and 3/7 share an x coordinate and true inter-shank distance
cannot be recovered. Distance therefore enters the model as the components
that *are* defined: same shank, depth difference and map-x difference.
