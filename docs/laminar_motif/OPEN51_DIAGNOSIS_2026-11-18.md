# OPEN-51 diagnosis, 2026-11-18: where the R1 divergence originates (static; pending independent review)

Authority: owner 2026-11-18, bounded diagnosis. Static inspection and analysis of existing outputs only. No simulation,
no protocol change, no tuning. The frozen protocol, the R1 outputs, the receipts and the NOT REPRODUCED
classification are unchanged. Evidence and scripts are in `open51_diag/`.

## Question

Does the first divergence of R1 from the original E0 seed-1 counts (step 8088, 808.8 ms) originate in:
- stochastic state (PRNG streams, drive, construction);
- numerical execution (how the compiled program evaluates the same model);
- or simulation state evolution (run-to-run nondeterminism in the same environment)?

## Findings

1. **The original compiled program survives.**
   - The persistent JAX compilation cache `~/.cache/jomission-jax` holds 306 entries dated 2026-11-06 to 2026-11-08.
   - Among them is the `jit_scan` executable compiled at 2026-11-06 18:42, when the original E0 seed-1 run started
     (its counts file is from 19:17, with `wall_s` 2128).
   - It embeds `E:\repos\_wt\jaxfne-0.5.9-site\jaxfne\...`, `E:\repos\_wt\jomission-handoff\scripts\run_e0_scored.py`
     and `caminos_e0.py`.
   - Both it and R1's executable are copied to `open51_diag/*.cache`. Hashes are in `cache_entries.txt`.
2. **Same program input, same construction size.**
   - Both executables deserialize under jaxlib 0.10.1. Their entry signatures are identical (`main.43`), with
     6 236 657 edges in both, equal to R1's realized edge count.
   - The 33 JaxFNE source locations (file, function, line) in their stack frames are identical (`frames.txt`).
   - The traced JaxFNE program is therefore the same. This is direct, line-level evidence that the original E0 build
     matches `e0160dfc`, which was previously only declared.
3. **Different compiled evaluation of the membrane update** (`hlodiff.txt`; full dumps in `*.hlo.txt.gz`).
   - With instruction names canonicalized, the two optimized programs differ only in how `v + 0.1·dv` and the
     threshold test are fused:
     - the original computes the update once in a `ynn_fusion` (XLA:CPU YNNPACK path), stores it, and compares the
       stored value;
     - R1 has no YNN fusion: an `add_divide_fusion` feeds four consumers, each recomputing `v + 0.1·dv` inline before
       `compare(... GE ...)`.
   - The PRNG (threefry `u32[8750]` loops) and the scatter are identical up to naming.
   - The loop `backend_config` differs only in a field name: `dynamic_variables` in the original,
     `dynamic_variable_tuple_indices` in R1.
4. **The compiler configuration differs, consistently** (`ynn.txt`, a census of 422 cached `jit_scan` executables).

   | Executables | YNN fusion | Loop config field |
   |---|---|---|
   | Original era, 2026-11-06 to 11-09 (side-install and handoff venv) | present in all 25 | `dynamic_variables` (checked in E0 seed 1) |
   | jax 0.10.2 environments (`jomission_pin058` 11-14; 351f6048 runs 11-16 and 11-17) | present | `dynamic_variables` (checked in pin058) |
   | jax 0.10.1 re-run environment (R1 and two smokes) | absent in all 3 | `dynamic_variable_tuple_indices` |

   - Both jaxlib 0.10.1 and 0.10.2 binaries contain the flag `xla_cpu_experimental_ynn_fusion_type`. The difference
     is therefore the build's default or a setting.
   - Inferred: the deleted handoff venv ran jaxlib 0.10.2, or a build that shares both markers. The protocol's
     candidate rule could not see it, because the venv no longer exists.
5. **The divergence boundary is one spike moved by one step** (`boundary.txt`).
   - At step 8088, R1 has one extra E spike; at step 8089, the original has one E spike that R1 lacks. All other
     populations are equal.
   - Only 2 steps differ over the next 2 ms. The difference then grows: 41% of steps differ in the first 100 ms and
     65-96% in later 100 ms windows.
   - Everything before step 8088 is equal: 8087 steps of E, PV and SOM counts across 9500 cells.

## Assessment against the three hypotheses

| Hypothesis | Evidence | Status |
|---|---|---|
| Stochastic state (PRNG, drive, construction) | identical PRNG HLO; identical edge count and signature; 808.8 ms of identical counts, which a different drive stream would not give | disfavoured (inferred), not excluded: input array values were not compared |
| Numerical execution (compiler) | the original and R1 executables differ exactly in the float32 evaluation of `v + 0.1·dv` before the threshold test; the first difference is one threshold crossing moved by 0.1 ms, the signature of a rounding difference at threshold; the configuration difference matches jax 0.10.2 against 0.10.1 on two markers | favoured (inferred); the mechanism is not demonstrated |
| State evolution or run-to-run nondeterminism | R2 did not run | untested |

A difference that changes how a float32 sum is fused can change the last bit of `v` (for example through FMA
contraction or operation order) and move a crossing by one step. The network then amplifies it. This is a candidate
mechanism; it has not been shown at the bit level.

## Proposed discriminating experiment (not run; needs owner authorization)

**Replay the original executable.**
- Construct E0 seed 1 in the re-run environment (worktree 3257ffc, JaxFNE `e0160dfc`), with no simulation, to obtain
  the executable's input arrays.
- Run the deserialized original `jit_scan` executable (`open51_diag/original_e0s1_jit_scan.cache`) on them, and as
  control R1's own executable on the same inputs.
- Compare per-step counts with the original and with R1.

| Outcome of the original executable | Reading |
|---|---|
| equal to the original counts for all 120 000 steps | numerical execution is the cause; inputs and stochastic state are identical |
| first difference at a step earlier than 8088 | inputs differ (stochastic state or construction) |
| equal up to 8087, then R1's divergence again | the executable is not the cause; state evolution or runtime remains |
| any other first difference | runtime or host effects; reported as found |

- **Control:** R1's executable on the same inputs must reproduce R1's counts exactly. That also tests same-environment
  determinism for this run. If the control fails, the replay is uninterpretable and stops.
- **Precondition, checked before running:** the input arrays built for the replay hash equal to the inputs R1
  consumed. The edge table is already equal to R1's, per its receipt.
- **Cost:**
  - 2 simulations of 12 s each (the trip count of 120 000 is compiled in), about 22 min each, so about 45 min CPU
    under `suite_lock.py`;
  - about 1 h of setup to assemble the argument list.
  - Risk: the argument order must match `main.43`. Both signatures are identical, which bounds this risk.
- **Stopping criterion:** one replay and one control, no retries, no parameter changes. Any outcome is reported.
- **Scope:** diagnostic only. The protocol classification stays NOT REPRODUCED whatever the outcome. Using a different
  jax version or the original executable for acceptance would need a protocol amendment, which is the owner's
  decision.

A cheaper, zero-simulation alternative would compile E0 seed 1 under a fresh jax 0.10.2 venv and compare its optimized
HLO with the original's. It would test the version hypothesis only; it cannot test causality.
