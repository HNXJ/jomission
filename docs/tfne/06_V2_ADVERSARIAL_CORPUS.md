# TFNE v2 adversarial normalization corpus

**Date:** 2026-09-17
**Algebra:** `TFNE_ALGEBRA_PROJECT_SOURCE.md`; decisions in `05_V2_MIGRATION_DECISIONS.md`.
**Contains:** proposed valid corpus, four invalid probes, expected structural and type
properties, and the manually derived normalized representation of each expression.
**Does not contain:** a parser, a compiler, a grammar fix, or any decision. Where the
derivation could not be completed, the gap is recorded as a finding, not resolved.

Every expression is written in the source's own vocabulary. Nothing here adds a primitive.

## 0. What a derivation contains

Derived by hand, one deterministic expansion per expression:

| Part | Content |
|---|---|
| `NF.objects` | path → kind, declared member set, `N`, `P`, `model`, `G`, parameters |
| `NF.relations` | ordered (`O`) and cross (`X`) relations with their rule name; composite boundaries preserved |
| `NF.projections` | directed projections (`>`, `<`, `<>`) with source and target addresses and their originating rule |
| `NF.exclusions` | `≠>`, `≠<` entries with what they remove |
| `NF.boundaries` | declared types of `x` and `y` |
| realization | integer counts from `P` and `N` under the declared allocation `ρ`, then `I` |

`I` is derived, never written by the author (`05` refinement).

## 1. Expected structural and type properties

| ID | Property | Checked where |
|---|---|---|
| T1 | `N[A]` is an integer and `Σ_c N[A.c] = N[A]` | realize |
| T2 | `0 ≤ P_A[c] ≤ 1` and `Σ_{c∈C_A} P_A[c] = 1` over the declared member set | type |
| T3 | every address resolves to a declared path; references never create objects | type |
| T4 | every relation is `O` or `X`, optionally carrying one rule `[k]`; a bare relation generates no projections | normalize |
| T5 | projection direction comes only from `>`, `<`, `<>`; a relation never implies it | normalize |
| T6 | `x` and `y` carry declared types | type |
| T7 | `{...}` survives into `NF` as an object boundary; no silent flattening | normalize |
| T8 | `A^n` yields `n` indexed instances and no relations among them | normalize |
| T9 | exactly one expansion exists for a fully defined expression | normalize |
| T10 | `I` is total over realized units, injective, and reproducible for fixed `(NF, R, ρ)` | realize |

Allocation `ρ` used throughout: largest remainder, ties broken by declared member order.

## 2. Valid corpus

### C1 — single HH cell

```text
CELL := [ C := {hh} ; N = 1 ; P = [hh:1.0] ; model[hh] = HH ; G = [point] ]
x : CELL : y                    ; x : [I_inj, pA] ; y : [V_m, mV]
```

- `NF.objects`: `CELL` (members `{hh}`, `N=1`, `model=HH`, `G=point`).
- `NF.relations`, `NF.projections`, `NF.exclusions`: empty.
- `NF.boundaries`: `x : I_inj (pA)`, `y : V_m (mV)`.
- realize: `N[CELL.hh] = 1` (T1); `I[CELL] = I[CELL.hh] = [0,1)` (T10).
- `h = {h_dyn}` only: no plasticity, no history, no RNG stream declared.

Degenerate validity holds: no separate single-cell language, no new primitive. **PASS.**

### C2 — nonlaminar nucleus

```text
LGN := [ C := {relay, inter} ; N = 1000 ; P = [relay:0.8, inter:0.2] ;
         model[relay] = Izhikevich ; model[inter] = Izhikevich ; G = [ellipsoid] ;
         X[local] := [...] ]
x : LGN : y                     ; x : [retinal spike trains] ; y : [relay spike trains]
```

- `NF.objects`: `LGN` with two members; one internal cross relation `LGN X[local] LGN`.
- realize: `N[LGN.relay] = 800`, `N[LGN.inter] = 200`, sum `1000` (T1, T2).
- `I[LGN.relay] = [0,800)`, `I[LGN.inter] = [800,1000)`.

No layers are required, so optional hierarchy holds. **PASS.**

### C3 — six-layer cortical `CTX`

```text
CTX[k] := [ L := {L1,L2,L3,L4,L5,L6} ; C := {E,PV,SST,VIP} ;
            P      = [L1:0.02, L2:0.15, L3:0.20, L4:0.22, L5:0.23, L6:0.18] ;
            P[L1]  = [PV:0.10, SST:0.35, VIP:0.55] ;          # E absent by declaration
            P[L4]  = [E:0.70, PV:0.15, SST:0.10, VIP:0.05] ;  # L2,L3,L5,L6 likewise
            model  = Izhikevich ; G = [laminar depth profile] ; X[local] := [...] ]

V1 := CTX[v1] ; N[V1] = 10000
```

- Layer counts: 200, 1500, 2000, 2200, 2300, 1800 (sum 10000, T1).
- `L4` split: `E 1540, PV 330, SST 220, VIP 110` (sum 2200).
- `L1` has no `E` member: `PV 20, SST 70, VIP 110` (sum 200). An absent class is a
  declaration, not grammar; `P_{L1}` normalizes over `{PV,SST,VIP}` (T2).
- `I[V1.L4] = [3700, 5900)`; inside it `I[V1.L4.E] = [3700, 5240)`,
  `I[V1.L4.PV] = [5240, 5570)`, `I[V1.L4.SST] = [5570, 5790)`, `I[V1.L4.VIP] = [5790, 5900)`.
- `V1.L4.E.v` addresses the membrane-potential slice of that range (path semantics, T3).

D4a–D4f are now properties of this `CTX` candidate, not of the grammar. **PASS.**

### C4 — replicated spinal chain

```text
SEG := [ C := {motor, inter, sensory} ; N = 500 ;
         P = [motor:0.2, inter:0.6, sensory:0.2] ; model = Izhikevich ]
O[rc] := [ ... ]                       # rostro-caudal ordered rule
CORD := { SEG^8 }
x : CORD : y
```

- realize per instance: `motor 100, inter 300, sensory 100` (T1, T2); eight instances,
  4000 units total.
- `NF.relations` is **empty**: `A^n` gives indexed instances only (T8). The chain is not
  expressed.
- Writing it out, `SEG_1 O[rc] SEG_2 O[rc] … O[rc] SEG_8`, needs eight declared names
  rather than `SEG^8`, and then `{SEG^8}` is not what is composed.

**BLOCKED** by F1 and F2 below. The counts and `I` derive cleanly; the topology does not.

### C5 — heterogeneous multi-area system

```text
x : LGN O[ff] V1 O[ff] {V2 X[lateral] V3} O[ff] MT : y ;
V1 > MT ;
V2 ≠< V3
```

- `NF.objects`: `LGN`, `V1`, `MT`, and the composite `{V2 X[lateral] V3}` preserved as a
  boundary (T7), containing `V2` and `V3`.
- `NF.relations`: `LGN O[ff] V1`; `V1 O[ff] {…}`; `{…} O[ff] MT`; `V2 X[lateral] V3`.
- `NF.projections`: `V1 > MT` — direction explicit, mechanism absent.
- `NF.exclusions`: `V2 ≠< V3`.

Three parts of the derivation cannot be completed: what `O[ff]` connects when an operand
is a composite (F3), how the rule body names its two operands and the direction it
generates (F4), and whether the unbraced chain is flat or nested (F5). The exclusion
cannot be resolved without knowing the generated set it subtracts from (F7).

**BLOCKED** by F3, F4, F5, F7.

### C6 — group sensitivity

```text
A O B O C  ;  {A O B} O C  ;  A O {B O C}
```

The source states these are not generally equivalent. Derivation: the three differ only
once a rule's behaviour on a composite operand is defined; with the composite treated as
transparent they collapse to the same pairwise relation set. The property is asserted but
not yet computable. **BLOCKED** by F3 and F5 — and this is the sharpest test of both.

## 3. Invalid probes

Each probe is expected to be rejected, and names the stage that should reject it.

| ID | Expression | Violates | Reject at |
|---|---|---|---|
| N1 | `NUC := [ C := {a,b} ; N = 100 ; P = [a:0.7, b:0.5] ]` | T2, proportion normalization | type |
| N2 | `V1.L1.E > V1.L4.PV`, with `C_{L1} = {PV,SST,VIP}` (C3) | T3, address resolution to an absent member | type |
| N3 | `x : SEG^8 : y` used as a rostro-caudal chain | T8, replication independence: no relation exists to carry the flow | normalize |
| N4 | `NUC := [ C := {E,PV} ; N = 101 ; P = [E:0.8, PV:0.2] ]` with no declared allocation | T1, exact cardinality: `80.8` is not an integer and `ρ` is undeclared | realize |

N3 is the probe that matters: it fails silently in any language where replication implies
adjacency. Here it must produce an empty relation set and be caught, not quietly realize
eight isolated segments as if they were a cord.

## 4. Findings: what the corpus exposes

Grammar insufficiencies, in the order they blocked a derivation. None is decided here.

| ID | Gap | Options seen |
|---|---|---|
| F1 | No schema relates the instances of `A^n`. A chain, ring, or nearest-neighbour lattice over replicated instances cannot be written. | (a) let a rule apply to a replicated object, `SEG^8 O[rc]`, with the rule defining the index relation; (b) an indexed relation schema over `i`; (c) accept explicit enumeration and treat `^n` as sugar for naming only |
| F2 | No spelling addresses the i-th instance. `[]` already carries three readings (selection, rule binding, specialization); an index would be a fourth. | (a) `SEG.3` as a path segment; (b) `SEG[3]` as a fourth `[]` reading; (c) require explicit names |
| F3 | A composite operand has no frontier. `V1 O[ff] {V2 X[lateral] V3}` does not say whether `V1` reaches `V2`, `V3`, both, or a declared frontier of the composite. | (a) declare frontiers inside the composite; (b) make it the rule's business, `O[k]` deciding how to distribute over a composite; (c) a default (all members) with an override |
| F4 | Rule bodies have no operand binder and no direction vocabulary. `O[ff] := [...]` cannot refer to "the lower operand" or state that it generates `>`. | (a) reserved binders inside a rule body; (b) positional forms; (c) rules written as templates over addresses |
| F5 | Associativity of unbraced `O` and `X` is unstated, while group sensitivity presupposes an answer. | (a) unbraced is n-ary flat, braces the only nesting; (b) left-associative with braces as override |
| F6 | `I` needs a declared traversal order for the flat index space; the ranges in C2, C3 assume declaration order. | (a) declaration order; (b) canonical sort by path; either way it must be stated, since `I` is required to be reproducible |
| F7 | `≠>` and `≠<` have no resolution rule: what generated set they subtract from, and whether an exclusion selecting nothing is an error. | the v1 exception algebra `G = (G0 \ E−) ∪ E+` is available as a starting point, but is not in the source |

Failure classes for the future minimum error vocabulary (decision 3), observed rather
than imported from v1:

| Class | Trigger | Probe |
|---|---|---|
| A | proportions do not normalize | N1 |
| B | address does not resolve, or names an absent member | N2 |
| C | topology assumed over replicated instances | N3 |
| D | non-integer allocation with no declared `ρ` | N4 |
| E | projection with no mechanism from any rule | C5, `V1 > MT` |
| F | more than one admissible expansion | C5, C6 (via F3, F5) |

## 5. Universality result

| Case | Expressible with no new primitive |
|---|---|
| single HH cell | yes (C1) |
| nonlaminar nucleus | yes (C2) |
| six-layer cortical `CTX` | yes (C3) |
| replicated spinal chain | not yet: F1, F2 |
| heterogeneous multi-area system | not yet: F3, F4, F5, F7 |

Three of five pass as written. The two failures are about composition of composites and of
replicas. F3, F4 and F7 are plausibly answerable inside rule definitions, which would keep
the sealing criterion intact. F1, F2, F5 and F6 are spelling and normalization decisions
that no definition can supply.

Review these before one replacement permanent spec is written and the v1 documents retire.
