# OPEN-51 diagnosis, 2026-11-18: where the R1 divergence originates (static diagnosis reviewed ACCEPT-WITH-CHANGES, changes applied; replay run, causal reading reviewed ACCEPT-WITH-CHANGES, changes applied)

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
   - The persistent JAX compilation cache `~/.cache/jomission-jax` holds 4867 entries from 2026-11-05 to 2026-11-17;
     306 of them are dated 2026-11-06 to 2026-11-08.
   - Among them is a `jit_scan` executable with file time 2026-11-06 18:42. The original E0 seed-1 counts file is from
     19:17 with `wall_s` 2128, so that run started near 18:42; the match is inferred from these times (the entry's
     4-byte header field is 0, so it carries no timestamp).
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
   - With instruction names canonicalized (1331 vs 1338 instruction lines, 167 diff lines), the differences lie in the
     membrane update and threshold test:
     - the original holds the whole `dv` chain in a `ynn_fusion` (XLA:CPU YNNPACK path): a reduce-sum over the
       2 receptors, subtract, multiply, divide, multiply, add; it stores `v + 0.1·dv` and compares the stored value;
     - R1 has no YNN fusion and no separate fused reduce: the chain is in an `add_divide_fusion` feeding four
       consumers, each recomputing `v + 0.1·dv` inline (an extra 0.1 constant and parameters) before
       `compare(... GE ...)`.
   - Reduction order, divide and FMA contraction are all candidate sources of a last-bit difference.
   - The PRNG (threefry `u32[8750]` loops) and the scatter are identical up to naming.
   - The loop `backend_config` differs only in a field name: `dynamic_variables` in the original,
     `dynamic_variable_tuple_indices` in R1.
4. **The compiler configuration differs between the two environments** (`ynn.txt`, a census of 420 cached `jit_scan`
   executables; 71 contain a YNN fusion).

   | Executables | YNN fusion | Loop config field |
   |---|---|---|
   | Original era, 2026-11-06 to 11-09 (side-install and handoff venv) | present in all 25 | `dynamic_variables` (checked in E0 seed 1) |
   | jax 0.10.2 environments (`jomission_pin058` 11-14; 351f6048 runs 11-16 and 11-17) | present | `dynamic_variables` (checked in pin058, `pin058_markers.txt`) |
   | jax 0.10.1 re-run environment (R1 and two smokes) | absent in all 3 | `dynamic_variable_tuple_indices` |

   - The 349 YNN-free executables also include 17 from 2026-11-05, 237 from 11-11 and 76 from 11-14. Their jax
     versions were not determined, and YNN fusion also depends on the program (many small programs have none). The
     0.10.1 side of the version link therefore rests on the three re-run executables.
   - Both jaxlib 0.10.1 and 0.10.2 `jax_common.dll` contain the flag string `xla_cpu_experimental_ynn_fusion_type`
     (`pin058_markers.txt`). The difference is therefore a build default or a setting, not the flag's presence.
   - Inferred: the deleted handoff venv ran jaxlib 0.10.2, or a build that shares both markers. The protocol's
     candidate rule could not see it, because the venv no longer exists.
5. **The divergence boundary is one E spike count moved by one step** (`boundary.txt`).
   - At step 8088, R1 has one extra E spike; at step 8089, the original has one E spike that R1 lacks. All other
     populations are equal. The original saved counts only, so whether the same cell fired is unknown.
   - Only 2 steps differ over the next 2 ms. The difference then grows: 41% of steps differ in the first 100 ms and
     62-96% in later 100 ms windows.
   - Everything before step 8088 is equal: 8087 steps of E, PV and SOM counts across 9500 cells.

## Assessment against the three hypotheses

| Hypothesis | Evidence | Status |
|---|---|---|
| Stochastic state (PRNG, drive, construction) | identical PRNG HLO; identical edge count and signature; 808.8 ms of identical counts, which a different drive stream would not give | disfavoured (inferred), not excluded: input array values were not compared |
| Numerical execution (compiler) | a compiled-code difference in the float32 evaluation of `dv` and `v + 0.1·dv` before the threshold test is observed; the configuration difference matches jax 0.10.2 against 0.10.1 on two markers | favoured: the code difference is observed, its causal link is untested |
| State evolution or run-to-run nondeterminism | R2 did not run; a one-step shift followed by growth is equally the signature of any small perturbation, nondeterminism included | untested |

The counts cannot distinguish a last-bit rounding difference from other small perturbations: a chaotic network
amplifies either into the same pattern. The compiled-code difference is the only observed candidate.

## Replay result, 2026-11-18 (owner authorized the replay; receipts in `open51_replay/`)

| Run | Executable | Counts vs original | Counts vs R1 | Raster vs R1 |
|---|---|---|---|---|
| control, 02:53-03:15Z | R1's (`r1_e0s1_jit_scan.cache`, YNN 0) | first difference at step 8088, 91 201 steps differ | equal at all 120 000 steps; array sha256 `236d9612…` = R1's | equal (sha256 `fc602009…` = R1's) |
| replay, 03:15-03:35Z | original (`original_e0s1_jit_scan.cache`, YNN 7) | equal at all 120 000 steps; array sha256 `ffcc8a7b…` = the original's | first difference at step 8088 | differs |

- Both runs used the re-run environment unchanged: worktree 3257ffc (clean), JaxFNE `e0160dfc`, jax 0.10.1, the edge
  table equal to R1's (sha256 `06f6c60f…`), and a fresh, empty JAX cache. Only the `jit_scan` executable differed:
  the driver `scripts/open51_replay.py` (caaebf7) substitutes it at the compile call, exactly once per run.
- The original executable loaded as an XLA:CPU AOT result (machine code, no recompile), and its loaded HLO held the
  YNN fusion.
- Reading, per the pre-stated outcome table: **numerical execution is the cause.** Swapping only the `jit_scan`
  executable switches the output between the two trajectories. The edge table and the PRNG HLO are equal; the other
  input array values were not hashed. An input difference that both executables mask is not excluded, but it is
  unlikely, since each executable reproduced its own trajectory exactly on the same inputs.
- Run-to-run nondeterminism: none observed, in one repeat (control against R1). The scope is seed 1, one executable
  per arm, CPU, one machine. The control run also serves as the sanity check of the substitution path that the
  single-step smoke could not provide.
- The comparison covers per-step counts of E, PV and SOM. No cell-level original raster exists, so equal counts do not
  show that the same cells fired.
- Not established:
  - which instruction in the executable produces the first differing bit. The cause is the executable; the YNN
    fusion of the `dv` chain is the only differing region in the canonicalized HLO, but the AOT machine code was not
    compared and the bit-level path is not traced;
  - that jaxlib 0.10.2 produced the original executable (inferred from two markers).
- Deviations from the proposal:
  1. The smoke did not execute a single step, because the trip count of 120 000 is compiled in. Instead, the AOT
     load and the YNN count were checked before running.
  2. The first control attempt stopped in setup (02:52Z, exit 7, no simulation). The cache key under the replay
     driver differs from R1's, so the entry was not used. The driver was fixed to substitute at the compile call, and
     the attempt's receipt is kept. Unexplained: the control and replay requested different keys (`4f39b75f…`,
     `d0a9058c…`) for the same program; the substitution makes the key irrelevant to the result.
- Scope: diagnostic only. OPEN-51 stays NOT REPRODUCED under the frozen protocol. The replay's raster
  (`open51_replay/output_files.sha256`, kept on D:) is a raster whose counts equal the original's. It was produced by
  the original executable in another environment, and it has no standing as the recovered original raster unless the
  owner amends the protocol.

## Proposed discriminating experiment (as approved; run above)

**Replay the original executable.**
- Construct E0 seed 1 in the re-run environment (worktree 3257ffc, JaxFNE `e0160dfc`), with no simulation, to obtain
  the executable's input arrays.
- Run the deserialized original `jit_scan` executable (`open51_diag/original_e0s1_jit_scan.cache`) on them, and as
  control R1's own executable on the same inputs.
- Compare per-step counts with the original and with R1.

| Outcome of the original executable (control passed) | Reading |
|---|---|
| equal to the original counts for all 120 000 steps | numerical execution is the cause; inputs and stochastic state are identical |
| first difference at a step earlier than 8088 | inputs differ (stochastic state or construction) |
| equal to R1's counts for all 120 000 steps | the executable is not the cause; the compiled-code difference has no effect here |
| equal up to 8087, diverging from both the original and R1 at or after 8088 | the replay itself is not deterministic, or the runtime differs from the original's; reported as found |
| fails to load or execute | replay not possible on this jaxlib; no reading |

- **Control:** R1's executable on the same inputs, run first. If it reproduces R1's counts exactly,
  same-environment determinism holds for this run. If it does not, that is a finding for run-to-run nondeterminism
  (hypothesis 3); it is reported, and the replay still runs but its reading is limited to the first difference.
- **Precondition, checked before running:** the input arrays built for the replay hash equal to the inputs R1
  consumed. The edge table is already equal to R1's, per its receipt.
- **Portability smoke, before the long runs:**
  - deserialize the original executable under jaxlib 0.10.1, confirm the loaded executable's HLO still contains
    `ynn_fusion` (evidence that it did not recompile without YNN), and run it on a single step;
  - deserializing and dumping HLO, done so far, is not execution, and whether the serialized form holds machine
    code or recompiles under 0.10.1 is unknown;
  - if the smoke fails, stop and report: the replay is not possible on this jaxlib.
- Each run gets freshly built input arrays (buffer donation).
- **Cost:**
  - 2 simulations of 12 s each (the trip count of 120 000 is compiled in), about 22 min each, so about 45 min CPU
    under `suite_lock.py`;
  - about 1 h of setup to assemble the argument list.
  - the 22 min per run matches R1's `wall_s` of 1293.6 s; each run needs about 6.6 GB working set, and a raster
    output (pred[120000,9500]) of about 1.1 GB, kept on D:;
  - Risk: the argument order must match `main.43`. Both signatures are identical, which bounds this risk. PRNG keys
    are inputs and the threefry HLO is identical.
- **Stopping criterion:** one smoke, one control and one replay, no retries, no parameter changes. Any outcome is
  reported.
- **Scope:** diagnostic only. The protocol classification stays NOT REPRODUCED whatever the outcome. Using a different
  jax version or the original executable for acceptance would need a protocol amendment, which is the owner's
  decision.

A cheaper, zero-simulation alternative would compile E0 seed 1 under a fresh jax 0.10.2 venv and compare its optimized
HLO with the original's. It would test the version hypothesis only; it cannot test causality.
