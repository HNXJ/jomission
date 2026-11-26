# Re-review of the R2 spec, draft 3 (5d4acf5), 2026-11-26

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` at 5d4acf5
(branch `_temp_lansa_r2_spec_20261125`, based on dev 54048dc). Previous reviews: draft 1 (6b95a85) REVISE, draft 2 (6185106)
REVISE. Verdict: REVISE, one major and three minors. No design point is open except A2.

## Status of the draft 2 review points

| Point | Draft 3 | Status |
|---|---|---|
| Major A: bitwise baseline across delivery modes | baseline on the default route for both sides; event checked separately at the seal commit for 1 s; 1bef838 cited, stated as not covering P0 | addressed |
| Minor 1: name the switch | `JAXFNE_EDGE_DELIVERY` in sections 5 and 7; `runtime_kw` lines 40-47 | addressed; matches dev `scripts/run_e2_stack.py` |
| Minor 2: stale GPU item | "decided (Hamm 2026-11-25) and not yet built" | addressed |
| Minor 3: sigma_alt caveat | "no data basis" in sections 1 and 6 | addressed in two places; see B2 |
| Minor 4: "1.41 is the T1 axis-1 winner" | FINDINGS_2026-11-10.md line 72 | verified: line 72 under "2b. T1 axis 1, kd" reads "Winner kd 1.41 (seed-mean r_proto 0.361, SD 0.021)" |

## Findings

MAJOR A2. The bitwise checks name no device and no precision. Section 5 says the gate venv `jomission058` is CPU, and a
0.5.8 GPU venv is decided. The bitwise baseline (section 4), the event check and the seal runs can run on either. Float32
results differ between CPU and GPU. The host rules (`rules/host.md`, 2026-11-24) record that JAX's default float32 matmul
runs in TF32 on this GPU: a 4096^2 product had max error 1.1e-1 at the default precision against 3e-4 to 1e-3 at `highest`.
So "bitwise equal" means something only within one device and one precision setting. The only prior equality evidence,
1bef838, does not say which device it ran on. Fix: name one device and one precision for every bitwise check and every seal
run. If the GPU is used, set `jax_default_matmul_precision="highest"` (or `JAX_DEFAULT_MATMUL_PRECISION=highest`) on both
sides and in the seal, and record it in the run JSON. Record the device of 1bef838 in section 4.

MINOR B1. Section 7 says "each run record's `edge_delivery` field must read `event`". The baseline and the default-route
check record "default" (`scripts/run_e2_stack.py` line 122, which writes `os.environ.get("JAXFNE_EDGE_DELIVERY") or
"default"`). Fix: "every scored run (seeds 7-9)".

MINOR B2. Section 9 records "sigma_alt 0.25" as decided without "no data basis". Sections 1 and 6 carry the caveat. Fix: add
it in section 9.

MINOR B3. The comparison object is not defined. "The constructed network (connectivity, weights, delays, parameters)" is
produced by two builders that differ: `build_config` took its receptor naming in 89a1272, and `caminos_stack.py` changed by
218 lines after cc33b37. Fix: name the serialisation each side produces (for example the edge table and the per-receptor
arrays after construction), and state that each side uses its own code to produce it.

## Checked and correct in draft 3

- `JAXFNE_EDGE_DELIVERY` gates the event `RuntimeConfig` in `runtime_kw` (dev lines 40-47). Unset gives the default route.
- The record field follows the variable (dev line 122).
- cc33b37 has no delivery switch (no `JAXFNE_EDGE` or `edge_delivery` match under `scripts/` or `jomission/`).
- 1bef838 is described as one 8 s run on 351f6048, equal across the two routes, 22 arrays. The draft correctly limits its scope.

## Verdict

REVISE. A2 must be fixed before any precondition runs, because the device decides whether "bitwise" can be read at all.
B1 to B3 are wording and record fixes. After A2 is fixed, a fourth review is needed; it is not the writer's.
