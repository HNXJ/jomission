# Geometry R2 spec, draft 4 (not sealed; execution not authorised)

Status: DRAFT for owner review. Nothing here is sealed. No run starts until (a) every OWNER item in section 9 is decided,
(b) a manifest entry authorises the lineage, and (c) the final text is committed and pushed as the seal (AGENTS.md).

## 1. Owner decision

Hamm, 2026-11-25, by direct answer in session jxm-dev (relayed to other lanes): route R2 for geometry Phase 1
(`GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md` section 3): new spectra whose parameters are fixed before any comparison with
the observed map Y. R1 is not run. Same session, same day: parameter set P0; `sigma_alt` 0.25 (no data basis); depths confirmed as listed
(+0.095, +0.198 primary, +0.406 sensitivity; mouse-derived, macaque transfer assumed); a 0.5.8 GPU venv for execution.
Hamm, 2026-11-26, same session, to the open items of draft 4.1 (a4b9a89: device D1 recommended, section 7 rule,
manifest entry `SCI-LANSA-GEOM-R2`): "proceed with recommendations". Decided: device D1 (CPU); the section 7 rule as
written; the manifest entry.
Reviews: draft 1 (6b95a85) REVISE, `docs/receipts/lansa_r2_spec_review_2026-11-25.md` (0da32ff); draft 2 (6185106)
REVISE (a58f3e9); draft 3 (5d4acf5) REVISE, `docs/receipts/lansa_r2_spec_rev3_review_2026-11-26.md` (a1d8ed9). All
addressed below.

## 2. Question

Given unit spectra generated without any Y-informed parameter, does an anatomically placed readout (fixed depths, fixed
sigma) reproduce each held-out area's gamma depth profile? Same question as protocol section 1, on Y-independent spectra.

## 3. Y-independence of the parameter set (evidence)

Y entered the repo at 5992eb7 (2026-11-09 22:25:22 -0600, `docs/laminar_motif/data/lansa_lfp_Y.json`, sha256 aad7f702).
Its parent cc33b37 (21:48:56) is the last Y-free code state.

| Knob | Value used by T1 / sw_* / lfs_* | First appearance | Y-exposed? |
|---|---|---|---|
| `sstNN` (SST->E decay) | 15 ms | 89a1272, 22:31:47, commit reports "r_proto 0.38" against Y | yes (observed) |
| `kdB` (local in-degree vs density) | B = 1, then 1.41 | kd1 was queued as a plan at cc33b37 ("kd1 x SST 15/30"); first run and named best in 734fede ("kd1 + SST->E 15 ms, r_proto 0.376"); 1.41 is the T1 axis-1 winner (FINDINGS_2026-11-10.md line 72) | planned before Y, selected after Y: exposed |
| `sstdNN` | 21 ms | T1 axis 3 winner by r_proto (FINDINGS_2026-11-10.md line 137) | yes (observed) |
| `dens` (E cells per unit proportional to LANSA density) | on | modifier documented at cc33b37 (`caminos_stack.py` `variant` docstring) | no for the code; see X below |
| `kX` (inter-unit in-degree K_INTER * X) | 1 | modifier at cc33b37; 1 is the identity | no |
| SST->E decay with no `sst` token | shared inhibitory receptor, `TAU_I_MS` = 8.0 ms | `caminos_e0.py` line 18 at cc33b37 | no |

Density input X (`docs/laminar_motif/data/lansa_density_X.json`) is a cell-density table, not the LFP map. Whether X and Y
come from the same animals or sessions is UNKNOWN; ASSUMED independent as measurements.

## 4. Parameter set (OWNER, proposed)

Ranked proposals; one is chosen and sealed.

| ID | Layout | Basis | Exposure |
|---|---|---|---|
| P0 (recommended) | `e2_3unit_n4000_dens_k1` | every value present at cc33b37, before Y; SST->E on the shared 8 ms receptor; no kd, sst or sstd token | none known from Y; X assumed independent |
| P1 | `e2_3unit_n4000_dens_k1_kd1` | B = 1 is the linear density scaling, an a-priori form | value first used after Y and named best by r_proto: exposed |
| P2 | P0 with an SST->E decay from a literature source | needs a cited source, none in the repo | none if the source predates and ignores Y |

Decided: P0 (Hamm 2026-11-25).

Code-path condition (review major 2): the P0 path changed after cc33b37 (`caminos_stack.py` +218/-47 lines with
`caminos_e0.py` and `run_e2_stack.py`; 7a83ac6 adds a refusal guard). Before the seal: run P0 with the same seed for 1 s
at cc33b37 and at the seal commit, both on the default delivery route (`JAXFNE_EDGE_DELIVERY` unset; cc33b37 has no
delivery switch), and compare bitwise. The cc33b37 side runs from a `git archive cc33b37` tree in session scratch, with
`PYTHONPATH` set to that tree; the seal side runs from the seal commit. Any difference blocks the seal and is recorded;
the seal then names which code state is P0.

Comparison object (review B3): each side produces its own output with its own code, `scripts/run_e2_stack.py` of that
commit. Compared: (a) every array of the `.npz` (`counts`, `block_keys`, `g_nS`, `lfp_pA`, the per-block raster samples),
equal key sets and, per array after loading, equal dtype, shape and `tobytes()` (the compressed container file is not
compared); (b) the run JSON fields present at both commits that describe the network: `sizes`, `edges`
(`caminos_stack.edge_table`, edge counts per projection), `p_local`, `p_inter`, `rho`, `depth_rel_L4`, `tau_sst_e_ms`,
`g_columns`, `g_reversal_mV`, `g_tau_ms`, equal as parsed JSON. Synaptic weights and delays are not serialised as such by
either driver; they are tested only through (a), whose `g_nS` and spike arrays depend on them (INFERRED sufficiency).

Delivery check, separate: at the seal commit, the same P0 seed for 1 s on the default route and with
`JAXFNE_EDGE_DELIVERY=event`, compared bitwise on (a) and (b). Prior evidence: 1bef838 (FINDINGS section 5): run
`evchk_kd1p41_sst15_n4000_s1_T8` (351f6048, event) equal to `t1_kd1p41_n4000_s1_T8` (0.5.8, default) in 22 arrays; both
records read `jax_devices` `["cpu:0"]`, no matmul-precision setting recorded. It does not cover P0.

Device and precision (review A2): every bitwise check and every scored run uses one engine (JaxFNE 0.5.8, venv
`/home/hamm/.venvs/jomission058gpu`), one device and one precision, named in section 5. A bitwise result across devices
or precisions is not read.

P0 may fail the motif outright (FINDINGS shows the motif depends on kd and sst). A failure is a result, not a reason to
retune: any later change is a new lineage.

## 5. Execution (proposed)

- Engine: JaxFNE 0.5.8, the execution authority (`manifests/current_state.json` `jaxfne_version`). Venv
  `/home/hamm/.venvs/jomission058gpu` (decided Hamm 2026-11-25; built 2026-11-26 from tag `v0.5.8` = 11957afa): jaxfne
  0.5.8, jax/jaxlib 0.10.2 with the CUDA 12 plugin, `jax.default_backend()` = `gpu`; jomission is not installed there and
  is imported through `PYTHONPATH`. The gate venv `/home/hamm/.venvs/jomission058` (CPU) is not used for runs.
- Device: D1, decided (Hamm 2026-11-26). Options as put to him:
  - D1 (recommended): CPU, `JAX_PLATFORMS=cpu` in the same venv. Same device as 1bef838 and the T1 confirmation (neither
    1bef838 record names its venv; all three
    `t1_confirm_*` records read `jax_devices` `["cpu:0"]`, event route, simulate 181-194 s per 12 s run). Every record's
    `jax_devices` must read `["cpu:0"]`.
  - D2: GPU with `JAX_DEFAULT_MATMUL_PRECISION=highest` on every side. On this GPU the default float32 matmul runs in TF32
    (probe 2026-11-26, 512^2 normal product, seed 0, max error vs float64: 3.5e-2 default, 9.5e-5 `highest`). The driver
    does not record precision; D2 needs a driver change to record it, which the seal commit must contain. No GPU result
    appears in cowork `results/laminar_e2` at 21b2697 (127 run records with `jax_devices`, all `cpu:0`; 8 analysis files
    carry no device field), so D2 has no continuity with T1 there; other trees were not checked.
- Seeds 7, 8, 9 (T1 used 1-6). T = 12 s, dt 0.1 ms, `JAXFNE_EDGE_DELIVERY=event`: the T1 confirmation settings (`T_s`
  12.0, `dt_ms` 0.1 in their run records). Discard as the scorer applies it, unchanged.
- Driver: `scripts/run_e2_stack.py` (`main(seed, T_s, tag, lay)`) with the sealed layout. It builds the event
  `RuntimeConfig` only when `JAXFNE_EDGE_DELIVERY=event` is set (`runtime_kw`, lines 40-47); unset is the default route.
  The code SHA is the seal commit. Tags: `r2_p0_n4000_s{7,8,9}_T12` (scored), `r2_smoke_p0_n4000_s7_T1` (smoke).
- Cost reference: T1 confirmation, CPU, event route, simulate 181-194 s per 12 s run (`wall_s`); 3 runs about 10 min
  (INFERRED; P0 has no kd or sstd token and may differ in spike count).
- Smoke first: one seed, 1 s, checks construction, finite output and the V0/V1 artifacts; it is not scored against Y.

## 6. Readout conditions and controls

As `GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md` sections 4-6, unchanged: estimand per held-out area (LOAO over 11 areas,
median summary); conditions G0, G1, G2, G3, G1'; controls K1 (flat spectrum) and K2 (depth-shuffled null) required.
`sigma_alt` = 0.25 (Hamm 2026-11-25; no data basis, a contrast to the baseline 0.15, itself an assumption).

## 7. Support and falsification (decided, Hamm 2026-11-26)

- Support: G1 median LOAO r_gm >= 0.6, G1 exceeds K1 at the same geometry, and median r_ab >= 0.6.
- Falsification: G1 median r_gm < 0.6, or median r_ab < 0.6, or G1 does not exceed K1.
- Threshold provenance (review major 1): `R_MIN = 0.6` was first set in 9681fc5 (2026-11-09 23:58, after Y), the commit
  that reports the best Y run failing it. It is Y-exposed. It is retained by the 2026-11-25 ruling and not changed here.
  Sensitivity, descriptive only: the continuous median r_gm and r_ab are reported, with pass/fail at 0.5 and 0.7 beside
  0.6. Only 0.6 carries the claim.
- Event delivery is set explicitly with `JAXFNE_EDGE_DELIVERY=event`, which makes the driver pass
  `RuntimeConfig(recurrent_backend="edge_list", edge_delivery="event")`; every scored run's record (seeds 7-9) must read
  `edge_delivery` `event`. The baseline and the default-route check record `default`.

## 8. Visualization contract

This lineage builds a model: V0 schematic and V1 initial 1 s raster before any scoring; V2 final raster and V3 atlas before
the lineage is declared complete (`manifests/visualization_contract.json`). The record's `visualization` block names all four.

## 9. OWNER items before sealing

Decided 2026-11-25: parameter set P0; `sigma_alt` 0.25 (no data basis); depths as listed; 0.5.8 GPU venv (built
2026-11-26). Decided 2026-11-26: device D1 (CPU); the section 7 rule; manifest entry `SCI-LANSA-GEOM-R2`. Open:

1. Before the seal (agent work): the bitwise P0 baseline and the delivery check (section 4), CPU.

Fixed and not reopened: the T1 record and FAIL, the gamma criterion, Y, the scorer.
