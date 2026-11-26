# Re-review of the R2 spec, draft 4 (d2df9d5), 2026-11-26

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` at d2df9d5
(branch `_temp_lansa_r2_spec_20261125`, based on dev 54048dc). Earlier reviews: draft 1 (6b95a85) REVISE; draft 2 (6185106)
REVISE; draft 3 (5d4acf5) REVISE, `docs/receipts/lansa_r2_spec_rev3_review_2026-11-26.md`. Verdict: ACCEPT, with three
minor wording fixes to make in the seal text. No major is open. The device is an owner item (section 5, D1 or D2).

## Status of the draft 3 review points

| Point | Draft 4 | Status |
|---|---|---|
| A2: device and precision | section 4 requires one engine, device and precision for every bitwise check and scored run; section 5 makes the device an owner item (D1 CPU, recommended; D2 GPU at `highest`) | addressed; the decision is the owner's |
| B1: "every scored run" | section 7 names seeds 7-9; baseline and default-route check record `default` | addressed |
| B2: sigma caveat in section 9 | "no data basis" present | addressed |
| B3: comparison object | section 4 lists (a) the `.npz` arrays and (b) the JSON fields | addressed; minor M1 |
| GPU venv | built 2026-11-26; `pip show` here confirms jaxfne 0.5.8, jax and jaxlib 0.10.2, and `jax-cuda12-plugin` 0.10.2 | addressed |
| Cost reference | T1 confirmation, CPU, event route, 181-194 s per 12 s run | verified below |

## Verified (observed)

- 1bef838 records at cowork 21b2697. `evchk_kd1p41_sst15_n4000_s1_T8`: `edge_delivery` "event", `jax_devices` ["cpu:0"],
  no precision field. `t1_kd1p41_n4000_s1_T8`: `jax_devices` ["cpu:0"], no `edge_delivery` field (default route), no
  precision field. The draft's description matches both.
- T1 confirmation records (seeds 4-6): `edge_delivery` "event", `jax_devices` ["cpu:0"], simulate 185.7, 193.7 and
  181.4 s. The draft's 181-194 s range is correct. Three runs are about 9.4 min, so "about 10 min" is correct.
- The driver saves `np.savez_compressed` with `counts`, `block_keys`, `lfp_pA`, `g_nS` and the raster arrays. The run
  JSON carries `depth_rel_L4`, `p_inter`, `jax_devices` and the other fields the draft lists. No precision field is written.
- No `gpu:` device string appears under `results/laminar_e2` at cowork 21b2697. The draft's "no GPU result exists for any
  E2 family" holds for that folder only (see M3).

## Findings (minor; none blocks the seal)

MINOR M1. Section 4 says "(a) every array of the `.npz` ... key sets and bytes equal". The file is written with
`savez_compressed`, so the container bytes can differ between two runs with identical arrays (zip timestamps; INFERRED from
Python's `zipfile`, not checked in source). Fix: state that (a) compares each decompressed array's dtype, shape and raw bytes
(`np.load` then `tobytes`), and never the container file's bytes.

MINOR M2. Section 5 says D1 "matches 1bef838 and the T1 confirmation". The device matches (`cpu:0`). The venv is not
recorded in either 1bef838 record: the records carry the engine git SHA 351f6048 but no venv path. Fix: say "device matches;
venv of 1bef838 not recorded".

MINOR M3. Section 5 says "No GPU result exists for any E2 family". The check covers cowork `results/laminar_e2` at 21b2697
only. Fix: qualify the scope ("none under `results/laminar_e2` at 21b2697"), or extend the search to the other result folders.

## Not reproduced here

- The TF32 probe figures (512^2, 2026-11-26: 3.5e-2 default, 9.5e-5 `highest`). I did not run the probe. They agree in
  direction with the host note (4096^2, 2026-11-24, 1.1e-1 default against 3e-4 to 1e-3 `highest`).
- The device-to-venv mapping for D1: that `JAX_PLATFORMS=cpu` in the CUDA-capable venv reproduces the CPU-only venv's
  output. The precondition compares two sides inside one venv, so the spec does not need that mapping; it would matter only
  if someone compared with 1bef838's venv.

## Verdict

ACCEPT, with M1 to M3 fixed in the seal text. The owner items remain open and are the owner's: device (D1 or D2), support and
falsification rule, and the manifest entry `SCI-LANSA-GEOM-R2`. The bitwise baseline and the delivery check run only after the
device is chosen.
