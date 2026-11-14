# E0 provenance addendum 1: population_rates call form

Status: additive record, 2026-11-14. The sealed record is unchanged: `docs/laminar_motif/E0_SEAL.md`, `results/laminar_e0/e0_result.json`, `results/laminar_e0/e0_seed1..5.json`, `docs/laminar_motif/E0_PILOT_RECEIPT.json`.
Authority: owner ruling, 2026-11-14, Option A ("authorize the seconds-form API on the frozen JaxFNE 0.5.8 baseline"). Scope is the call form only. No value, gate, threshold, seed, reference, margin or verdict changes.

## Invocation as sealed

Amendment 1 (owner, 2026-11-06; commit 0465b36) ran the E0 analysis on JaxFNE 0.5.9 (tag `v0.5.9`, `e0160dfc`) with:

```
population_rates(arr, sizes=sizes, pops=POPS, dt_ms=dt * 1000.0, discard_ms=DISCARD_S * 1000.0)
```

The scored runs record `jaxfne_versions: ["0.5.9"]` in `results/laminar_e0/e0_result.json`.

## Corrected invocation (this addendum)

```
population_rates(arr, dt=dt, sizes=sizes, pops=POPS, discard_s=DISCARD_S)
```

Code: commit `6092f75` on `origin/_temp_open47_population_rates` (one call, `jomission/laminar/caminos_analysis.py:105`). Not yet on `dev`; integration waits for the OPEN-47 gate classification.

## Why the two forms give the same quantity

Read from source (re-checked for this addendum):

| Step | 0.5.8 seconds form (`metrics.py:1082`) | 0.5.9 ms form (`metrics.py:1103`) |
|---|---|---|
| Units in | `dt` s, `discard_s` s | `dt_ms`, `discard_ms` converted to s (`/1000`) |
| Cut | `round(discard_s / dt)` | `round(discard / dt_s)` |
| Duration | `keep.shape[0] * dt` | `keep.shape[0] * dt_s` |
| Rate | `sum / dur / size` | `sum / dur / size` |

Both raise `ValueError` when the run is shorter than the discard.

Numeric check (branch receipt `docs/receipts/open47/`, `poprates_equiv.py`, `pr_058.json`, `pr_059.json`): 3 seeds, 3 populations, 2 comparisons = 18 values. Worst relative difference reported as 0. The 0.5.9 seconds form emits `DeprecationWarning`.

## Limits of this verification

- The 18-value check uses synthetic Poisson counts (dt 1e-4 s, discard 2 s, 12 s runs). It does not use the sealed E0 spike trains, which are not in the repo. Bit-level equality on the sealed counts: unknown.
- The receipt was produced by the OPEN-47 author. The reviewer re-read the source and confirmed the formulas above; the receipt run itself was not repeated here.

## What the corrected form does and does not change

- Engine: the corrected form runs on JaxFNE 0.5.8 (the execution authority). On 0.5.9 it runs with a deprecation warning. No engine code or pin changed.
- Sealed results: not re-run, not re-scored, not edited.
- Stale text: `docs/laminar_motif/E0_SEAL.md:56` still describes the `dt_ms` call. It stays as sealed text. This addendum supersedes the call form only.
- 0.5.8 pilot (2026-11-06, seed 0, unscored; `E0_SEAL.md` Amendment 1): ran with the pre-amendment positional form. Not affected by this addendum.
