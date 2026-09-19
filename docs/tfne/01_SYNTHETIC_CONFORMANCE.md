# TFNE Algebra v1: synthetic conformance suite

> **SUPERSEDED by `TFNE_V2_SPEC.md` (tfne/2, 2026-09-17).** `V1_FIXTURE`, kept as the v1 record.

> **`V1_FIXTURE` (2026-09-17).** Superseded for v2 semantics: written against the v1
> spelling (`Q`, `→`, `↛`, `()`) and the v1 error codes. Preserved as the historical
> record; not mechanically translated. The v2 corpus is `06_V2_ADVERSARIAL_CORPUS.md`.
> The JaxFNE probes under `scripts/tfne/probes/` are unaffected: they execute the engine
> and parse no TFNE notation, so their 18 pinned values stay valid receipts.

**Owner:** Jomission
**Algebra version:** `tfne-algebra/1`
**Date:** 2026-09-16
**Status:** SPECIFIED; no compiler exists.
**Semantics:** `00_TFNE_ALGEBRA_SPEC.md`

This suite tests the algebra and its compiler before any biology is frozen. Structural tests need no motifs. Projection-level tests use the registry below.

> **SYNTHETIC.** Every motif in this registry is a test fixture with no biological meaning. It must never be used as a canonical motif, cited as architecture, or used in a scientific run. Registry ids carry the prefix `synthetic/` and the registry sets `"synthetic": true`. A compiler loading it for anything other than tests should refuse.

## 1. Fixture registry

```json
{
  "schema": "tfne_motif_registry_v1",
  "registry_id": "synthetic/conformance",
  "registry_version": "synthetic-1",
  "synthetic": true,
  "units": {
    "synthetic/H": {
      "status": "defined",
      "layers": {"L1": ["E"], "L2": ["E"], "L3": ["E"], "L4": ["E", "PV"], "L5": ["E"], "L6": ["E"]},
      "aliases": {"L23": ["L2", "L3"], "L56": ["L5", "L6"]},
      "input_frontier": ["L4.E"],
      "output_frontier": ["L23.E"],
      "within_unit": [],
      "params": {"N": {"type": "int", "min": 1, "default": 100}}
    }
  },
  "O": {
    "synthetic/O": {
      "status": "defined",
      "ff": [{"from_lower": "L23.E", "to_higher": "L4.E", "mechanisms": ["AMPA"],
              "allowed_mechanisms": ["AMPA", "NMDA"]}],
      "fb": [{"from_higher": "L56.E", "to_lower": "L1.E", "mechanisms": ["AMPA", "NMDA"],
              "allowed_mechanisms": ["AMPA", "NMDA", "GABA_B"]}],
      "params": {"w": {"type": "float", "default": 1.0}, "p": {"type": "float", "min": 0, "max": 1, "default": 1.0}}
    }
  },
  "Q": {
    "synthetic/Q": {
      "status": "defined",
      "lateral": [{"between": "L23.E", "and": "L23.E", "mech": "AMPA", "symmetric": true}],
      "params": {"w": {"type": "float", "default": 1.0}, "p": {"type": "float", "min": 0, "max": 1, "default": 1.0}}
    }
  },
  "X": {
    "synthetic/X": {
      "status": "defined",
      "directed": [{"from": "L23.E", "to": "L4.E", "mech": "AMPA"}],
      "params": {"w": {"type": "float", "default": 1.0}}
    }
  },
  "projection_defaults": {
    "synthetic/P": {
      "status": "defined",
      "unit_level": [{"from": "L56.E", "to": "L4.E", "mech": "AMPA"}],
      "params": {"mech": {"type": "str"}, "w": {"type": "float", "default": 1.0}, "p": {"type": "float", "min": 0, "max": 1, "default": 1.0}}
    }
  },
  "bindings": {"unit": "synthetic/H", "O": "synthetic/O", "Q": "synthetic/Q", "X": "synthetic/X", "->": "synthetic/P"}
}
```

**Expansion rules for this fixture:**

- An alias expands to the Cartesian product of its layers.
- `X[<->]` applies `directed` in both directions.
- `symmetric: true` applies `lateral` in both directions.
- An O route generates one projection per entry of `mechanisms`; `allowed_mechanisms` bounds explicit additions on that route.

**Base expansions**, used by the tests. Every projection is written `src -> dst [mech]`.

| Expression | Projections |
|---|---|
| `V1 O V4` (6) | `V1.L2.E -> V4.L4.E [AMPA]`, `V1.L3.E -> V4.L4.E [AMPA]`, `V4.L5.E -> V1.L1.E [AMPA]`, `V4.L5.E -> V1.L1.E [NMDA]`, `V4.L6.E -> V1.L1.E [AMPA]`, `V4.L6.E -> V1.L1.E [NMDA]` |
| `V1_L Q V1_R` (8) | `V1_L.La.E -> V1_R.Lb.E [AMPA]` and `V1_R.La.E -> V1_L.Lb.E [AMPA]`, for `La, Lb ∈ {L2, L3}` |
| `V1 X[->] PFC` (2) | `V1.L2.E -> PFC.L4.E [AMPA]`, `V1.L3.E -> PFC.L4.E [AMPA]` |
| `V1 X[<->] PFC` (4) | the 2 above, plus `PFC.L2.E -> V1.L4.E [AMPA]`, `PFC.L3.E -> V1.L4.E [AMPA]` |
| `V1; PFC; V1 -> PFC` (2) | `V1.L5.E -> PFC.L4.E [AMPA]`, `V1.L6.E -> PFC.L4.E [AMPA]` |

## 2. Structural tests (no registry)

### 2.1 Equivalence: equal `h_S`

| # | A | B |
|---|---|---|
| S1 | `V1 O V4 O PFC` | `(V1 O V4) O PFC`; `V1 O (V4 O PFC)` |
| S2 | `V1_L Q V1_R` | `V1_R Q V1_L` |
| S3 | `A Q B Q C` | `(A Q B) Q C`; `A Q (B Q C)` |
| S4 | `(V1_L Q V1_R) O (V4_R Q V4_L)` | `(V1_L Q V1_R) O (V4_L Q V4_R)` |
| S5 | `(V1_L Q V1_R) O (V4_L Q V4_R)` | `(V1_L O V4_L) Q (V1_R O V4_R)` |
| S6 | `x:(A Q B)` | `(x:A) Q (x:B)` |
| S7 | `x:(A O B)` | `(x:A) O B` |
| S8 | section 8 program of the spec | its product spelling |
| S9 | any program with `→`, `↛` | the same with `->`, `-/->`, other whitespace, statements reordered |

### 2.2 Structure

| # | Expression | Expected |
|---|---|---|
| C1 | `H Q^3` | 3 units; Q pairs `{H_1,H_2}`, `{H_1,H_3}`, `{H_2,H_3}` |
| C2 | `H O^3` | 3 units; O `H_1→H_2`, `H_2→H_3` |
| C3 | `(A O B) Q (C O D)` | O `A→B`, `C→D`; Q `{A,C}`, `{B,D}` only |
| C4 | `(A O B) Q (C O D) Q (E O F)` | Q `{A,C}`, `{A,E}`, `{C,E}`, `{B,D}`, `{B,F}`, `{D,F}` |
| C5 | spec section 8 program | 8 units, 6 O, 4 Q, 2 X, 2 input bindings |
| C6 | `V1 O OFC`; `V1OOFC` | 2 units and 1 O; 1 unit |
| C7 | `V1 O V4; V1 -> V4` | valid: `→` is not an architectural relation |

### 2.3 Invalid

| # | Expression | Error |
|---|---|---|
| I1 | `A Q B O C` | `E_UNGROUPED_MIX` |
| I2 | `(A_1 Q A_2) O (B_1 Q B_2 Q B_3)` | `E_MATCH` |
| I3 | `(A_L Q A_R) O (B_1 Q B_2)` | `E_MATCH` |
| I4 | `(H1 Q^2) O (H2 Q^3)` | `E_MATCH`; valid with a complete `O[map=...]` |
| I5 | `(A O B) Q C` | `E_SHAPE` |
| I6 | `(H Q^2) O (H Q^2)` | `E_ID_COLLISION` |
| I7 | `A O B; B O C; C O A` | `E_HIERARCHY_CYCLE` |
| I8 | `V1 O V4; V1 Q V4` | `E_RELATION_CONFLICT` |
| I9 | `V1 O V4; V1 X[->] V4` | `E_RELATION_CONFLICT` |
| I10 | `(A O B) Q^2` | `E_REP_COMPOSITE` |
| I11 | `O Q V1` | `E_TOKEN` |
| I12 | `V1 O V4; V9 -/-> V1` | `E_ADDRESS_UNKNOWN` (unit `V9` is declared nowhere) |
| I13 | `V1; PFC; V1 -> PFCl` | `E_ADDRESS_UNKNOWN` (a reference never declares a unit) |

## 3. Projection-level tests (fixture registry)

### 3.1 Expansion

| # | Program | Expected |
|---|---|---|
| P1 | `V1 O V4` | exactly the 6 base projections |
| P2 | `V1_L Q V1_R` | exactly the 8 base projections |
| P3 | `V1 X[->] PFC`; `V1 X[<->] PFC` | 2; 4 |
| P4 | `V1; PFC; V1 -> PFC` | 2. With `synthetic/P` removed from the registry: `E_MOTIF_UNDEFINED`. Without the `V1; PFC` declarations: `E_ADDRESS_UNKNOWN` |
| P5 | `V1 O V4; V1.L5.E ->[mech=AMPA] V4.L4.E` | 7 |
| P6 | `V1 O V4; V1.L5.E -> V4.L4.E` | `E_MECHANISM_UNRESOLVED` (no arrow mechanism; no motif generates this projection) |
| P7 | `x:(V1 O V4)` | one input binding: `x → V1.L4.E` |
| P8 | `V1 X PFC` | `h_S` computable; projection NF raises `E_X_UNDIRECTED` |
| P9 | spec section 8 program | 72 projections (6×6 O + 4×8 Q + 2×2 X); bindings `x → V1_L.L4.E`, `x → V1_R.L4.E` |
| P10 | `H Q^3` | 24 |
| P11 | `V1 O V4; V1 -> V4` | 8 (6 from O plus `V1.L5.E -> V4.L4.E [AMPA]`, `V1.L6.E -> V4.L4.E [AMPA]`) |
| P12 | `V1 O V4; V1.L7.E ->[mech=AMPA] V4.L4.E` | `E_ADDRESS_UNKNOWN` (layer `L7` is absent from `synthetic/H`) |
| P13 | `V1; PFC; V1 ->[mech=NMDA] PFC` | `E_MECHANISM_CONFLICT` (`synthetic/P` generates AMPA) |
| P14 | `V1; PFC; V1 ->[mech=AMPA] PFC` | `E_PROJECTION_REDUNDANT` (`synthetic/P` already supplies AMPA) |
| P15 | `V1 O V4; V1.L1.PV ->[mech=GABA_A] V4.L1.E` | `E_ADDRESS_UNKNOWN` (`synthetic/H` has no PV population in L1: `p_{L1,PV} = 0`) |

### 3.2 Exceptions (mechanism-inclusive identity)

| # | Program | Expected |
|---|---|---|
| E1 | `V1 O V4; V4 -/-> V1` | 2 remain (FF only) |
| E2 | `V1 O V4; V4.L5.E -/->[mech=NMDA] V1.L1.E` | 5 remain; `V4.L5.E -> V1.L1.E [AMPA]` retained |
| E3 | `V1 O V4; V4.L5.E -/-> V1.L1.E` | 4 remain (AMPA and NMDA removed) |
| E4 | `V1 O V4; V1.L5.E -/-> V4.L1.E` | `E_EXCL_NOT_IN_G0` (FB runs higher → lower) |
| E5 | `V1 O V4; V4.L5.E -/->[mech=GABA_A] V1.L1.E` | `E_EXCL_NOT_IN_G0` (mechanism absent) |
| E6 | `V1 O V4; V1.L23.E ->[mech=AMPA] V4.L4.E` | `E_PROJECTION_REDUNDANT` |
| E7 | `V1 O V4; V1.L23.E ->[mech=NMDA] V4.L4.E` | valid; 8 (distinct identities; NMDA ∈ `allowed_mechanisms` of the ff route) |
| E7b | `V1 O V4; V1.L23.E ->[mech=GABA_B] V4.L4.E` | `E_MECHANISM_NOT_PERMITTED` (GABA_B ∉ ff `allowed_mechanisms`) |
| E7c | `V1 O V4; V4.L5.E ->[mech=GABA_B] V1.L1.E` | valid; 7 (GABA_B ∈ fb `allowed_mechanisms`) |
| E8 | `V1 O V4; V4.L5.E -/->[mech=AMPA] V1.L1.E; V4.L5.E ->[mech=AMPA] V1.L1.E` | `E_EXC_CONFLICT` and `E_PROJECTION_REDUNDANT` |
| E9 | `V1 O V4; V4 -/-> V1` vs `V1 O V4; V4.L56.E -/-> V1.L1.E` | equal `h_T`, `h_P`; different `h_S` |
| E10 | `V1 O V4; V4 -/-> V1; V1.L5.E ->[mech=AMPA] V4.L4.E` in all 6 statement orders | identical `h_S`, `h_T`, `h_P` |
| E11 | `V1 O V4; V1.L23.E ->[mech=AMPA] V4.L4` | `E_PROJECTION_REDUNDANT` |

| E12 | `V1 O V4; V1.L56.E ->[mech=NMDA] V4.L4` | valid; 10 (all 4 resolved identities are new) |

E11 and E12 test transactional statements (spec 3.13). In E11, `V4.L4` expands to classes E and PV; the E targets are already in G0 and the PV targets are new, so the whole statement is rejected and the PV projections are not added. E12 has the same shape with every identity new.

### 3.3 Hashes

| # | A vs B | h_S | h_T | h_P | h_M |
|---|---|---|---|---|---|
| H1 | `V1 O V4; V1.L5.E ->[mech=AMPA, w=0.5] V4.L4.E` vs same with `w=0.7` | ≠ | = | ≠ | ≠ |
| H2 | `V1 O[p=0.5] V4` vs `V1 O V4` | ≠ | = | ≠ | ≠ |
| H3 | `V1 O[p=1.0] V4` vs `V1 O V4` (registry default 1.0) | ≠ | = | = | = |
| H4 | `V1 O V4` vs `V1 O V4; V1.L23.E ->[mech=NMDA] V4.L4.E` | ≠ | ≠ | ≠ | ≠ |
| H5 | same program; `registry_version` `synthetic-1` vs `synthetic-1b`, identical motif content | = | = | = | ≠ |
| H6 | `V1_L Q V1_R` vs `V1_R Q V1_L` (also Unicode vs ASCII, whitespace) | = | = | = | = |
| H7 | `V1[N=100] O V4` vs `V1[N=100.0] O V4` | may ≠ | = | = | = |
| H8 | `V1[N=200] O V4` vs `V1[N=100] O V4` | ≠ | = | ≠ | ≠ |
| H9 | `(V1_L Q V1_R) O (V4_L Q V4_R)` vs `(V1_L O V4_L) Q (V1_R O V4_R)` | = | = | = | = |

Notes:

- `h_S` is registry-free, so it cannot coerce types or resolve defaults (H3, H7). `h_S` equality implies `h_T` and `h_P` equality; the converse does not hold.
- In H4, `h_P` changes because the parameter set is keyed by projection identity and a new identity appears.

## 4. Acceptance

A compiler conforms to `tfne-algebra/1` when:

1. every row in sections 2–3 produces the stated result, with the error code among the reported violations;
2. every hash column relation holds;
3. no output derived from this fixture can be emitted without its `synthetic: true` marker.
