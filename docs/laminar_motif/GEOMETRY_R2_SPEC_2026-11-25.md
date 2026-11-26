# Geometry R2 spec, draft 1 (not sealed; execution not authorised)

Status: DRAFT for owner review. Nothing here is sealed. No run starts until (a) every OWNER item in section 9 is decided,
(b) a manifest entry authorises the lineage, and (c) the final text is committed and pushed as the seal (AGENTS.md).

## 1. Owner decision

Hamm, 2026-11-25: route R2 for geometry Phase 1 (`GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md` section 3): new spectra whose
parameters are fixed before any comparison with the observed map Y. R1 is not run.

## 2. Question

Given unit spectra generated without any Y-informed parameter, does an anatomically placed readout (fixed depths, fixed
sigma) reproduce each held-out area's gamma depth profile? Same question as protocol section 1, on Y-independent spectra.

## 3. Y-independence of the parameter set (evidence)

Y entered the repo at 5992eb7 (2026-11-09 22:25:22 -0600, `docs/laminar_motif/data/lansa_lfp_Y.json`, sha256 aad7f702).
Its parent cc33b37 (21:48:56) is the last Y-free code state.

| Knob | Value used by T1 / sw_* / lfs_* | First appearance | Y-exposed? |
|---|---|---|---|
| `sstNN` (SST->E decay) | 15 ms | 89a1272, 22:31:47, commit reports "r_proto 0.38" against Y | yes (observed) |
| `kdB` (local in-degree vs density) | B = 1, then 1.41 | value first in 734fede, "Best 3-unit run so far: kd1 + SST->E 15 ms, r_proto 0.376" | yes (observed) |
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

P0 may fail the motif outright (FINDINGS shows the motif depends on kd and sst). A failure is a result, not a reason to
retune: any later change is a new lineage.

## 5. Execution (proposed)

- Engine: JaxFNE 0.5.8, the execution authority (`manifests/current_state.json` `jaxfne_version`). The gate venv
  `/home/hamm/.venvs/jomission058` has CPU jaxlib only; a 0.5.8 GPU environment is an OWNER/environment item.
- Seeds 7, 8, 9 (T1 used 1-6). T = 12 s, dt 0.1 ms, event delivery: the T1 confirmation settings (`T_s` 12.0, `dt_ms`
  0.1 in their run records). Discard as the scorer applies it, unchanged.
- Driver: `scripts/run_e2_stack.py` (`main(seed, T_s, tag, lay)`, `edge_delivery="event"`) with the sealed layout; the
  code SHA is the seal commit.
- Cost reference: `lfs_n4000_s2_T8` simulate 1518 s for 8 s on its recorded machine (`wall_s`); 3 runs of 12 s are
  roughly 3 x 38 min there (INFERRED by scaling; hardware differs).
- Smoke first: one seed, 1 s, checks construction, finite output and the V0/V1 artifacts; it is not scored against Y.

## 6. Readout conditions and controls

As `GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md` sections 4-6, unchanged: estimand per held-out area (LOAO over 11 areas,
median summary); conditions G0, G1, G2, G3, G1'; controls K1 (flat spectrum) and K2 (depth-shuffled null) required.
`sigma_alt` is OWNER (section 9).

## 7. Support and falsification (OWNER to confirm)

- Support: G1 median LOAO r_gm >= 0.6, G1 exceeds K1 at the same geometry, and median r_ab >= 0.6.
- Falsification: G1 median r_gm < 0.6, or G1 does not exceed K1.
- r_gm threshold 0.6 and the 50-150 Hz band are the T1 criterion (`scripts/e2_scorer.py`), retained by ruling.

## 8. Visualization contract

This lineage builds a model: V0 schematic and V1 initial 1 s raster before any scoring; V2 final raster and V3 atlas before
the lineage is declared complete (`manifests/visualization_contract.json`). The record's `visualization` block names all four.

## 9. OWNER items before sealing

1. Parameter set: P0 (recommended), P1 or P2.
2. `sigma_alt` for G2/G3 (protocol proposes 0.25; no data basis).
3. Depth values: deep threshold +0.095; L5 mid +0.198 primary; L6 mid +0.406 sensitivity (ruling numbers blank in
   transmission).
4. Support and falsification rule (section 7) as written.
5. Execution environment: a 0.5.8 GPU venv, or CPU runs.
6. Manifest entry authorising the lineage (proposed id `SCI-LANSA-GEOM-R2`).

Fixed and not reopened: the T1 record and FAIL, the gamma criterion, Y, the scorer.
