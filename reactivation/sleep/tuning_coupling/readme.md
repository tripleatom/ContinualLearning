# Tuning similarity vs NREM UP-state co-activation

Tests one hypothesis:

> During NREM UP states, pairs of neurons with more similar visual tuning show
> greater **excess** co-activation than dissimilarly tuned pairs, after
> controlling for firing rates, UP-state-wide drive, recording distance and cell
> type.

"Excess" carries the weight. Inside an UP state every neuron fires more, so raw
co-firing rises for nearly every pair whether or not the two are functionally
related. Everything here is built to remove that shared drive before coupling is
reported, and the pipeline keeps the raw measure alongside the corrected one
specifically so the difference stays visible.

## Files

| file | what it does |
|---|---|
| `tc_similarity.py` | tuning similarity per pair + per-unit covariates |
| `tc_coupling.py` | UP-state windows, binning, excess/residual co-activation |
| `tc_stats.py` | pair-level regression, unit-label permutation test, pre/post |
| `run_tuning_coupling.py` | orchestrator, figures, CSV/pkl outputs |
| `tc_selftest.py` | synthetic sessions with a known answer |

## Run

```bash
python run_tuning_coupling.py --check       # what this session has, what is missing
python run_tuning_coupling.py --self-test   # synthetic validation, ~4 min
python run_tuning_coupling.py               # full run on the grating_config session
```

The session comes from `grating_config.py` (`ANIMAL_ID` / `EXPERIMENT_DATE`);
`--sortout` overrides it.

## The five steps

**1. Tuning similarity** — from `all_units_tuning.pkl`, two measures per pair:
`delta_pref_deg` (preferred-orientation difference wrapped onto the orientation
circle, so 0 deg == 180 deg, giving 0-90) and `signal_corr` (tuning-curve
correlation). The correlation is cross-validated by default: unit A's curve comes
from one random half of the trials and unit B's from the other half, averaged
over many splits, so the two curves never share a trial and the measure cannot be
inflated by noise the pair happens to share. Only units that are visually
responsive *and* split-half reliable enter (`--keep-unreliable` to relax) —
similarity between two unreliable tuning curves is mostly noise.

**2. UP states** — **required** source is the MUA OFF/ON detection
(`sleep/MUA/detect_off_states.py`), because it comes from channel-level multi-unit
activity and is therefore independent of the sorted units whose co-firing is being
measured. Per-shank ON windows are combined into probe-wide UP states (a stretch
where at least half the scored shanks are simultaneously ON). A sleep block with no
`detect_off_states.py` output is **skipped**, not silently replaced — using the
sorted population rate instead would define UP states from the very units whose
coupling is being tested. Pass `--population-up` to opt into that fallback anyway
(`reactivation/sleep/UPState.py`); the circularity it introduces is printed, not
hidden. UP states can be restricted to scored NREM when `*_sleep_periods.pkl` exists.

*Clocks are verified, not assumed.* MUA windows live in whole-day samples and
sleep spikes in block-relative seconds; after mapping, `check_up_alignment`
compares sorted-unit firing inside the windows against the gaps. A correct
mapping gives a clearly larger-than-1 ratio; anything near 1 aborts the run.

**3. Excess co-activation** — spikes are binned at 25 ms inside UP states only.
Three numbers per pair:

- `raw_corr` — plain correlation of the binned counts. The comparator that should
  *not* be used to test the hypothesis; it is the quantity that rises for everyone.
- `excess_z` (**primary**) — observed coincidences minus a rate-preserving null,
  in null SDs. The null permutes each unit's counts within short blocks that never
  cross an UP-state boundary, so each unit keeps its rate, its per-UP-state
  excitability and its slow within-UP time course.
- `resid_corr` — correlation of residuals after each unit's counts are predicted
  from UP-state identity x within-UP phase. An independent route to the same
  question, used as a robustness check.

**4. The test** — `coupling ~ tuning similarity + log rate (geometric mean) +
rate imbalance + same shank + depth difference + lateral offset + cell-type pair`.
The p-value is a **unit-label permutation** p-value: tuning curves are scrambled
across units while every unit keeps its rate, position and cell type. This is
necessary, not decorative — with n units each unit appears in n-1 pairs, so an OLS
standard error computed as if pairs were independent is far too small. OLS SEs are
still reported, flagged as anticonservative.

**5. Pre vs post** — the same model on the *within-pair change* in coupling,
restricted to pairs measured in both blocks. Differencing cancels any stable
functional architecture, so a slope there is the experience-dependent claim, while
a slope in each block separately with no pre/post difference is the stable-V1
result.

## Geometry caveat

This probe's channel map gives x in {0, 300, 600, 900} um for eight shanks, so
shanks 0/4, 1/5, 2/6 and 3/7 share an x coordinate and a true inter-shank distance
cannot be recovered from the map. Rather than invent one, separation enters the
model as the components that *are* defined — depth difference, map-x difference,
and a same-shank indicator. `dist_um` is filled in only for same-shank pairs.

## Validation

`--self-test` simulates three sessions and analyses them with the production code:

| dataset | what is planted | what must come out |
|---|---|---|
| `slow_common` | UP-state excitability + within-UP envelope, shared by all | excess z near 0, no tuning effect |
| `fast_common` | global drive faster than the shuffle block | excess z high for everyone, still no tuning effect |
| `effect` | orientation-specific fast latents | excess z tracks tuning similarity |

The middle row is the important one: it is the case the "excess" framing exists
for, and a method that cannot tell it from the bottom row would answer this
hypothesis wrongly. All 11 checks pass.
