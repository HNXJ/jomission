# TFNE v2 adversarial normalization corpus

**Date:** 2026-09-17 (second pass: decisions R1–R7 applied, all cases re-derived)
**Algebra:** `TFNE_ALGEBRA_PROJECT_SOURCE.md`; decisions in `05_V2_MIGRATION_DECISIONS.md`.
**Contains:** valid corpus, invalid probes, expected structural and type properties, manual
derivations, and the residual items the second pass exposed.
**Does not contain:** a parser, a compiler, new vocabulary, or any decision of mine.

Pipeline under derivation:

```text
source → parse → type/address resolution → replication expansion → structural normalization
       → frontier resolution → O/X-rule expansion → projection generation G_0
       → exception resolution → NF(A) --R, ρ--> (s, h_0, I)
```

`R` is the declared realization/allocation policy; `ρ` is the realization RNG identity.
`I` is derived, never authored.

## 1. Expected structural and type properties

| ID | Property | Stage |
|---|---|---|
| T1 | `N[A]` integer, `Σ_c N[A.c] = N[A]`, produced by `R` | realize |
| T2 | `0 ≤ P_A[c] ≤ 1`, `Σ_{c∈C_A} P_A[c] = 1` over the declared member set | type |
| T3 | every address resolves to a declared path; references never create objects | type |
| T4 | a relation is `O` or `X` with at most one rule; a bare relation generates no projections | normalize |
| T5 | direction comes only from `>`, `<`, `<>`, generated inside rule bodies via `$L`, `$R` | rule expansion |
| T6 | `x` and `y` carry declared types | type |
| T7 | `{...}` survives into `NF` as an object boundary; no silent flattening | normalize |
| T8 | `A^n` yields `n` indexed instances and no relations; `O[k](A^n)` yields the chain | replication |
| T9 | exactly one expansion exists for a fully defined expression | normalize |
| T10 | `I` is total, injective, and reproducible for fixed `(NF, R, ρ)` under canonical traversal | realize |
| T11 | `E_- ⊆ G_0` and `E_+ ∩ G_0 = ∅` | exception resolution |

Canonical traversal (R6): structural path, replicated numeric index, canonical child key,
population identity, local realization index; keys sort lexicographically.

## 2. Valid corpus, re-derived

### C1 — single HH cell

```text
CELL := [ C := {hh} ; N = 1 ; P = [hh:1.0] ; model[hh] = HH ; G = [point] ]
x : CELL : y                    ; x : [I_inj, pA] ; y : [V_m, mV]
```

Objects: `CELL`. No relations, projections or exclusions. `N[CELL.hh] = 1`;
`I[CELL] = [0,1)`. `h = {h_dyn}`. **Derives uniquely.**

### C2 — nonlaminar nucleus

```text
LGN := [ C := {inter, relay} ; N = 1000 ; P = [inter:0.2, relay:0.8] ;
         model = Izhikevich ; G = [ellipsoid] ; X[local] := [ $L.relay >[AMPA] $R.inter ] ]
x : LGN : y
```

`R` gives `N[LGN.inter] = 200`, `N[LGN.relay] = 800`. Canonical traversal sorts children
lexicographically, so `I[LGN.inter] = [0,200)` and `I[LGN.relay] = [200,1000)` — **not**
declaration order, which is the point of R6. **Derives uniquely.**

### C3 — six-layer cortical `CTX`

```text
CTX[k] := [ L := {L1,L2,L3,L4,L5,L6} ; C := {E,PV,SST,VIP} ;
            P     = [L1:0.02, L2:0.15, L3:0.20, L4:0.22, L5:0.23, L6:0.18] ;
            P[L1] = [PV:0.10, SST:0.35, VIP:0.55] ;
            P[L4] = [E:0.70, PV:0.15, SST:0.10, VIP:0.05] ;
            in := [L4] ; out := [L2, L3] ;
            model = Izhikevich ; G = [laminar depth profile] ; X[local] := [...] ]

V1 := CTX[v1] ; N[V1] = 10000
```

Layers: 200, 1500, 2000, 2200, 2300, 1800 (sum 10000). `L4`: E 1540, PV 330, SST 220,
VIP 110. `L1` has no `E` member: PV 20, SST 70, VIP 110.

`I[V1.L4] = [3700, 5900)`, and inside it `E [3700,5240)`, `PV [5240,5570)`,
`SST [5570,5790)`, `VIP [5790,5900)`. D4a–D4f are properties of this `CTX` candidate;
SL0/SL1/SL2 qualify it. **Derives uniquely.**

### C4 — replicated spinal chain (was blocked by F1, F2)

```text
SEG := [ C := {inter, motor, sensory} ; N = 500 ;
         P = [inter:0.6, motor:0.2, sensory:0.2] ; model = Izhikevich ;
         in := [inter] ; out := [inter] ]
O[spinal] := [ $L.out >[AMPA] $R.in ]
CORD := { O[spinal](SEG^8) }
x : CORD : y
```

- Replication expansion: instances `SEG.1 … SEG.8` (R2 addressing), no relations (T8).
- `O[spinal](SEG^8)` (R1) applies the rule between adjacent instances: 7 relations.
- Rule expansion (R4): `SEG.i.inter >[AMPA] SEG.(i+1).inter`, `i = 1…7`; `G_0` has 7
  projection identities at population level.
- Realize: per instance inter 300, motor 100, sensory 100; 4000 units.
  `I[CORD.SEG.3] = [1000,1500)`, `I[CORD.SEG.3.inter] = [1000,1300)`,
  `.motor = [1300,1400)`, `.sensory = [1400,1500)`.
- Chain frontier: `in(CORD) = in(SEG.1)`, `out(CORD) = out(SEG.8)` — see assumption A2.

**Derives uniquely.**

### C5 — heterogeneous multi-area system (was blocked by F3, F4, F5, F7)

```text
VIS  := { V1 X[callosal] V1_R }
SYS  := { Retina O[retino] LGN O[thalamo] VIS }
X[callosal] := [ $L.L23.E <>[AMPA] $R.L23.E ]
O[thalamo]  := [ $L.out >[AMPA] $R.in ]
x : SYS : y ;  V1 ≠< V1_R
```

- Structure: sequence `[Retina, LGN, VIS]` with `retino` on the first adjacency and
  `thalamo` on the second (R5, plus assumption A1 for the mixed-rule chain); `VIS` is a
  preserved composite containing `V1` and `V1_R`.
- Frontier resolution (R3): `in(VIS) = in(V1) ∪ in(V1_R) = {V1.L4, V1_R.L4}`. `O[thalamo]`
  expands `$L.out = LGN.out` to both, giving two projection families, not a guess.
- `X[callosal]` generates `V1.L23.E > V1_R.L23.E` and `V1_R.L23.E > V1.L23.E` (`<>`).
- Exception resolution (R7): `V1 ≠< V1_R` resolves to the second of those identities,
  which is in `G_0`, so `G = G_0 \ E_-` keeps only the left-to-right callosal family. Had
  the callosal rule generated only `>`, the same line would raise `E_EXCLUSION_UNKNOWN`
  instead of silently doing nothing.
- Composites are named, which matters: see residual G1.

**Derives uniquely.**

### C6 — group sensitivity

```text
A O[k] B O[k] C   ;   {A O[k] B} O[k] C   ;   A O[k] {B O[k] C}
```

Under R3 and R5 all three derive, and they differ **structurally**: the first normalizes to
the sequence `[A, B, C]`; the second and third carry a composite object, so addresses,
`I` keys and object identity differ, and `T7` keeps the boundary.

Their generated projection sets, however, coincide: with `in({A O B}) = in(A)` and
`out({A O B}) = out(B)`, every form yields `out(A) → in(B)` and `out(B) → in(C)`. Braces
change the object, not the topology, unless a rule distributes over a composite differently
than over a plain object. That is a consequence of A2, not a decision — flagged as
observation O1 below, since "not generally equivalent" may have been meant to bite on
topology too.

## 3. Invalid probes

| ID | Expression | Violates | Reject at |
|---|---|---|---|
| N1 | `NUC := [ C := {a,b} ; N = 100 ; P = [a:0.7, b:0.5] ]` | T2 | type |
| N2 | `V1.L1.E > V1.L4.PV` with `C_{L1} = {PV,SST,VIP}` | T3 | type/address |
| N3 | `x : SEG^8 : y` used as a chain | T8: no relation exists; `O[spinal](SEG^8)` is the correct form | replication/normalize |
| N4 | `NUC := [ C := {E,PV} ; N = 101 ; P = [E:0.8, PV:0.2] ]` **with no declared `R`** | T1: allocation policy missing. With `R` declared this is valid: `(80.8, 20.2) → (81, 20)`, `Σ = 101` | realize |
| N5 | `A X[k] B X[k] C` unbraced, `k` declaring no associative policy | R5: `X` is not globally associative | normalize |
| N6 | `V1 O[ff] {P X[lat] Q}` where `Q` declares no `in` frontier | R3: `E_FRONTIER_UNRESOLVED`, no guessing | frontier resolution |

N3 stays the probe that matters: replication must never imply adjacency. N4 is corrected
per the owner: non-integer products are `R`'s job; the failure is a missing policy.

Failure classes observed (input to the minimum error vocabulary, decision 3):

| Class | Trigger | Probe |
|---|---|---|
| A | proportions do not normalize | N1 |
| B | address unresolved or member absent | N2 |
| C | topology assumed over replicated instances | N3 |
| D | no allocation policy `R` | N4 |
| E | projection with no mechanism from any rule | a bare `V1 > MT` |
| F | more than one admissible expansion | G1 below |
| G | frontier not uniquely derivable (`E_FRONTIER_UNRESOLVED`) | N6 |
| H | exclusion matches nothing (`E_EXCLUSION_UNKNOWN`) | R7 |
| I | addition already generated (`E_+ ∩ G_0 ≠ ∅`) | R7 |
| J | unbraced non-associative `X` | N5 |

## 4. Residual items from the second pass

Assumptions I had to make to complete a derivation, and one genuine gap. No vocabulary was
added for any of them.

| ID | Item | Status |
|---|---|---|
| A1 | The acceptance example chains two **different** rules unbraced (`Retina O[retino] LGN O[thalamo] VIS`), while R5 states associativity for a homogeneous `O[k]` chain. Derived as the ordered sequence with each rule attached to its own adjacency. | needs confirmation; the alternative is to require braces for mixed-rule chains |
| A2 | Frontier of an *ordered* composite: `in({A O B}) = in(A)`, `out({A O B}) = out(B)`. R3 gives the union rule for `X` only. | needs confirmation |
| A3 | `in` and `out` become reserved path components (`$L.out`, `$R.in`, and `in := [...]` in a definition). | implied by R3 and R4; worth stating |
| G1 | **Anonymous composites have no canonical key.** R6 sorts children by canonical key, but `{V2 X[lateral] V3}` written inline has no name, so its position in the traversal is undefined and two sibling anonymous composites are indistinguishable. C5 avoids this only because every composite was named. | breaks determinism as written; fixable by requiring a name, or by keying a composite on its ordered member list. Needs no new grammar either way |
| O1 | Braces preserve object identity and addressing but, under A2, generate the same projections as the unbraced chain. | observation; confirm whether group sensitivity is meant to reach topology |
| O2 | Lexicographic key order means `L10` sorts before `L2`. Any object with more than nine same-prefixed children needs zero-padded keys or an explicit order in its definition (R6 already routes biological order into `NF`). | hazard, not a gap |

## 5. Universality result

| Case | Derives uniquely |
|---|---|
| single HH cell | yes (C1) |
| nonlaminar nucleus | yes (C2) |
| six-layer cortical `CTX` | yes (C3) |
| replicated spinal chain | yes (C4, via R1 and R2) |
| heterogeneous multi-area system | yes (C5, via R3, R4, R5, R7) |
| group sensitivity | yes, structurally (C6, see O1) |

All five universality cases derive with no new primitive and no biology-specific grammar.

**Stopping test.** The corpus broke deterministic normalization in exactly one place, G1,
the canonical key of an anonymous composite. That is a naming rule, not biology. A1, A2 and
A3 are confirmations rather than gaps. With G1 settled, the corpus no longer produces a
non-unique expansion, and one permanent v2 specification can replace the migration stack.
