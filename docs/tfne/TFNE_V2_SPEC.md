# TFNE v2: permanent specification

```
Algebra version:  tfne/2
Status:           SEALED (language). Parser: NOT IMPLEMENTED. Compiler: NOT IMPLEMENTED.
Owner:            Jomission (Hamm)
Date:             2026-09-17
Authority:        owner decisions of 2026-09-17 (05_V2_MIGRATION_DECISIONS.md R1-R7, G1, A1-A3, O1-O2)
Source:           TFNE_ALGEBRA_PROJECT_SOURCE.md (verbatim owner text; incorporated here)
Supersedes:       00_TFNE_ALGEBRA_SPEC.md, 01_SYNTHETIC_CONFORMANCE.md, 04_SOURCE_MIGRATION.md,
                  05_V2_MIGRATION_DECISIONS.md, 06_V2_ADVERSARIAL_CORPUS.md (kept, not deleted)
Scientific authority: NONE. This is a specification language, not evidence.
```

## 0. Invariants

$$\textbf{biology introduces definitions, not grammar}$$

$$\mathcal A \xrightarrow{\text{normalize}} NF(\mathcal A) \xrightarrow[\rho]{\text{realize},R} (s,h_0,I)$$

$$(h_t,x_t;s)\mapsto(h_{t+1},y_t)$$

$$\text{structural identity} \neq \text{instance identity} \neq \text{realization identity}$$

A new nervous-system structure is expressed by new definitions — objects, rules,
proportions, geometry, models — never by new operators. The primitive set is closed.

## 1. Core form

$$x:\mathcal A:y$$

An explicitly modelled system $\mathcal A$ between a typed upstream representation $x$ and a
typed downstream representation $y$. Both are interfaces, not particular physical
quantities; each carries a declared type where ambiguity matters. Either may stand for
omitted causes upstream or omitted consequences downstream.

## 2. Lexicon

Reserved fundamental names, complete:

| Name | Meaning |
|---|---|
| `O` | ordered architectural composition |
| `X` | cross / lateral (non-ordered) architectural composition |
| `x`, `y` | upstream, downstream representation |
| `H` | H-state tensor (nothing else) |
| `h`, `s` | mutable execution state, static execution state |
| `N`, `P`, `G`, `C` | cardinality, proportion, geometry, cell-type domain maps |
| `:` | system boundary |
| `:=` | definition |
| `{}` | composite structural boundary |
| `[]` | typed selection / specialization / rule binding |
| `.` | hierarchical addressing |
| `>`, `<`, `<>` | projection direction |
| `≠>`, `≠<` | projection exclusion |

Reserved **interface path components** (A3): `A.in`, `A.out`. They are addresses, not
operators. A definition may declare `in := [...]` and `out := [...]`; otherwise they are
derived (section 5).

Reserved **rule metavariables** (R4): `$L`, `$R`, bound to the left and right operand of
the invocation whose rule body they appear in.

Case convention: `Capital…` names an object; `lowercase…` names a value, process,
parameter, rule or representation.

## 3. Objects and definitions

`A := E` defines `A`. Definitions carry, as applicable:

| Field | Meaning | Constraint |
|---|---|---|
| `C` | declared member / cell-type domain | members are addressable |
| `P` | proportion map over the declared member set | $0\le P_A[c]\le1$, $\sum_{c\in C_A}P_A[c]=1$ |
| `N` | cardinality | integer; $\sum_c N[A.c]=N[A]$ after realization |
| `G` | geometry | first-class structure; compiles into `s` when fixed; never connectivity or dynamics |
| `model` | dynamical realization | biological identity and dynamical model are distinct; HH, Izhikevich, LIF or another declared model |
| `in`, `out` | interfaces | declared or derived (section 5) |

A member absent from an object is absent from its declared set: `P` normalizes over what is
declared, and an address to an undeclared member fails. Absence is a declaration, never
grammar.

The algebra is recursive and scale-free: cell substructure, cell, population, structural
subdivision, area or nucleus, multi-area system, whole nervous system. It stays valid at
cardinality one, and layers are one specialization of structural subdivision, never
mandatory. A cortical unit is therefore a definition, conventionally `CTX[k]`, with
instances `V1 := CTX[v1]`; `LGN`, `SC`, spinal segments and single cells likewise.

## 4. Composition

| Form | Meaning |
|---|---|
| `A O B` | ordered architectural composition; implies no projection direction |
| `A X B` | cross / lateral composition; implies no projection direction |
| `A O[k] B`, `A X[k] B` | composition carrying connection rule `k` |
| `{E}` | composite object: a preserved structural boundary, never silently flattened |
| `A^n` | `n` indexed instances `A.1 … A.n`, **no connectivity implied** |
| `O[k](A^n)` | derived application form: `A.1 O[k] A.2 O[k] … O[k] A.n` |

**Associativity (R5, A1).** An unbraced `O` chain normalizes to an adjacency-labelled
ordered sequence, and rules need not be equal along it:

```text
A O[k] B O[j] C   →   (A, k, B, j, C)
```

Each rule belongs to its own adjacency. `X` is not globally associative: `A X[k] B X[k] C`
must be grouped unless rule `k` declares an associative composition policy, else
`E_X_GROUPING`.

**Braces (O1).** Braces change object identity, scope, addressing, frontiers and future
composition semantics. They need not change the projections generated inside them:
`A O B` and `{A O B}` may generate the same internal projections, while `A O B O C` and
`{A O B} O C` differ because the next `O` acts on different exposed frontiers.

**Replication independence.** `A^n` alone is `n` disconnected instances. Any relation among
them is explicit, via `O[k](A^n)` or written out. A system built from `A^n` with no rule is
well formed and has zero inter-instance projections; the realization receipt shows that
directly (appendix B, N3).

## 5. Frontiers

Frontiers are never inferred from grouping. Every object exposes `in` and `out`, declared
in its definition or derived:

$$in(\{A_1 O \cdots O A_n\}) = in(A_1),\qquad out(\{A_1 O \cdots O A_n\}) = out(A_n)$$

$$in(\{A\,X\,B\}) = in(A)\cup in(B),\qquad out(\{A\,X\,B\}) = out(A)\cup out(B)$$

unless the `X[k]` rule overrides the frontier. A rule applies from `out` of its left operand
to `in` of its right operand. If a required interface is not uniquely derivable:
`E_FRONTIER_UNRESOLVED`. No guessing.

## 6. Rules, projections, exclusions

A connection rule may independently define topology, mechanism, parameters, geometry and
delay. Direction implies nothing about excitation, inhibition, mechanism, density, weight,
delay or plasticity.

```text
O[ff] := [ $L.out >[AMPA] $R.in ]
X[callosal] := [ $L.L23.E <>[AMPA] $R.L23.E ]
```

For `X[k]`, `$L` and `$R` carry syntactic identity only, with no implied order.

Canonical expansion of rules and projections produces the generated set $G_0$. Explicit
statements then apply:

$$E_-\subseteq G_0,\qquad E_+\cap G_0=\varnothing,\qquad G=(G_0\setminus E_-)\cup E_+$$

An exclusion matching nothing raises `E_EXCLUSION_UNKNOWN`, so a stale exclusion cannot
survive silently; an addition already generated raises `E_ADDITION_REDUNDANT`. A projection
that obtains no mechanism from any rule raises `E_MECHANISM_UNRESOLVED`.

## 7. Normalization

```text
source → parse → type / address resolution → replication expansion
       → structural normalization → frontier resolution → O/X-rule expansion
       → projection generation G_0 → exception resolution → NF(A)
```

Each stage fails locally. A fully defined expression has exactly one expansion; otherwise
`E_AMBIGUOUS_EXPANSION`.

`NF(A)` contains: objects with their declarations; composite boundaries; ordered sequences
with their adjacency rules; cross relations; projection identities after exceptions;
declared boundary types; and any explicitly declared order.

**Canonical keys and traversal (R6, G1, O2).** Traversal order is canonical, never
declaration order, so that reordering the source cannot change realization identity:

1. structural path;
2. replicated numeric index;
3. canonical child key;
4. population / cell identity;
5. local realization index.

Path components are **typed natural tokens** — name stem, numeric suffix or index,
specialization — so `L1 < L2 < … < L9 < L10` and `SEG.2 < SEG.10`. Zero padding is never
required. An explicit biological order declared in an object overrides canonical ordering
and enters `NF`.

An **anonymous composite** `{E}` takes a structural key derived from its normalized
contents: the deterministic canonical serialization of `NF(E)` orders traversal, and
`hash(serialization(NF(E)))` is the compact identity receipt. The serialized structural form
is the authority; its hash is a receipt, never the ordering key. Two structurally identical
anonymous siblings share a structural key and are separated by deterministic occurrence
indices assigned after canonical sorting: $C^{(1)}, C^{(2)}, \dots$ Hence structural
identity is not instance identity.

## 8. Realization

$$NF(\mathcal A)\xrightarrow[\rho]{R}(s,h_0,I)$$

- `R` is the declared realization and allocation policy. Non-integer products are `R`'s
  business: `N = 101` with `P = (0.8, 0.2)` realizes as `(81, 20)`, any deterministic
  allocation satisfying $\sum_c N_c = N$. A missing policy is `E_ALLOCATION_POLICY_MISSING`.
- `ρ` is the realization RNG identity, not the allocation rule. A deterministic `R` needs no
  RNG.
- `s` holds everything fixed after realization: realized topology, fixed parameters,
  geometry, constants.
- `h` holds everything needed from the current state to determine future evolution, kept as
  distinct components: $h=\{h_{\mathrm{dyn}},h_H,h_{\mathrm{plastic}},h_{\mathrm{history}},h_{\mathrm{rng}}\}$.
  Any declared mutable parameter $\theta\in h_{\mathrm{plastic}}$ may evolve by an explicit
  state-dependent rule; plasticity is not restricted to synaptic weight.
- `I` is the derived index map, never authored: $I[\mathrm{V1.L4.E}]=[i_0,i_1)$. It is total
  over realized units, injective, and reproducible for fixed $(NF, R, \rho)$. It must resolve
  both directions — address to index range and back — plus rule to realized edges, edges to
  their originating relation, parameter and state targeting by TFNE identity, and inspection
  of realized cardinality, geometry, parameters, state and connectivity.

Given `s`, future inputs $x_{t:}$ and deterministic RNG semantics, $h_t$ suffices to
reproduce future execution.

## 9. Execution

$$(h_t,x_t;s)\mapsto(h_{t+1},y_t)$$

Flattening changes representation, not TFNE semantics. The recursive hierarchy imposes no
recursive traversal inside the simulation loop: a runtime may use flat, indexed, vectorized,
JIT-compatible arrays provided `I` preserves inspectability. TFNE is a specification language
for JaxFNE, not a second simulator.

$$\textbf{factor at specification time; flatten at execution time}$$

## 10. Minimum error vocabulary

Derived from failures the conformance corpus actually produced, not inherited from v1.

| Code | Stage | Condition |
|---|---|---|
| `E_PROPORTION_INVALID` | type | `P` outside `[0,1]` or not summing to 1 over the declared member set |
| `E_ADDRESS_UNRESOLVED` | type | address names an undeclared path or an absent member |
| `E_X_GROUPING` | normalize | unbraced `X` chain whose rule declares no associative policy |
| `E_FRONTIER_UNRESOLVED` | frontier | required `in`/`out` not uniquely derivable |
| `E_MECHANISM_UNRESOLVED` | rule expansion | resolved projection with no mechanism from any rule |
| `E_EXCLUSION_UNKNOWN` | exception | exclusion matches nothing in `G_0` |
| `E_ADDITION_REDUNDANT` | exception | addition already in `G_0` |
| `E_AMBIGUOUS_EXPANSION` | normalize | more than one admissible expansion |
| `E_ALLOCATION_POLICY_MISSING` | realize | no declared `R` for a non-integer allocation |

The v1 catalogue is historical (`00_TFNE_ALGEBRA_SPEC.md` §5); a code re-enters only when a
corpus case demonstrates its failure class.

## 11. What is outside the language

- **Definitions**: `CTX` and its D4a–D4f content, `LGN`, spinal segments, models, geometry.
- **Qualification of a candidate `CTX`**: SL0 native generators; SL1 a qualified physical
  forward model; SL2 the spectrolaminar phenotype from SL0 + SL1. A candidate `CTX` is never
  tuned to make a phenomenological proxy reproduce SL2; `lfp_proxy` and `csd_proxy` are
  visualization and instrumentation, never the objective.
- **Realization and compiler work**, and every scientific verdict.

## Appendix A — conformance corpus

Full derivations: `06_V2_ADVERSARIAL_CORPUS.md`. The arithmetic below is derived by
`scripts/tfne/v2_conformance.py` from the declarations, and `tests/test_tfne_v2_spec.py`
checks this document against that derivation.

| Case | Expression | Exercises |
|---|---|---|
| C1 | `CELL := [ C := {hh} ; N = 1 ; model[hh] = HH ]`, `x : CELL : y` | degenerate validity |
| C2 | `LGN := [ C := {inter, relay} ; N = 1000 ; P = [inter:0.2, relay:0.8] ]` | optional hierarchy, canonical order over declaration order |
| C3 | `CTX[k] := [...]`, `V1 := CTX[v1] ; N[V1] = 10000` | proportions, absent member in `L1`, exact cardinality, `I` |
| C4 | `CORD := { O[spinal](SEG^8) }` | replication independence, `O[k](A^n)`, numeric path components |
| C5 | `SYS := { Retina O[retino] LGN O[thalamo] VIS }`, `VIS := { V1 X[callosal] V1_R }`, `V1 ≠< V1_R` | mixed-rule chain, frontiers, `$L`/`$R`, exclusion against `G_0` |
| C6 | `A O B O C` vs `{A O B} O C` vs `A O {B O C}` | braces: identity and frontiers, not internal topology |

```json
{
 "C2_LGN": {
  "counts": {"inter": 200, "relay": 800},
  "index_map": {"inter": [0, 200], "relay": [200, 1000]}
 },
 "C3_V1": {
  "L1_counts": {"PV": 20, "SST": 70, "VIP": 110},
  "L4_counts": {"E": 1540, "PV": 330, "SST": 220, "VIP": 110},
  "index_map_selected": {"L1": [0, 200], "L4": [3700, 5900], "L4.E": [3700, 5240],
                         "L4.PV": [5240, 5570], "L4.SST": [5570, 5790], "L4.VIP": [5790, 5900]},
  "layer_counts": {"L1": 200, "L2": 1500, "L3": 2000, "L4": 2200, "L5": 2300, "L6": 1800},
  "total": 10000
 },
 "C4_CORD": {
  "index_map_selected": {"SEG.3": [1000, 1500], "SEG.3.inter": [1000, 1300],
                         "SEG.3.motor": [1300, 1400], "SEG.3.sensory": [1400, 1500]},
  "per_instance_counts": {"inter": 300, "motor": 100, "sensory": 100},
  "projections": [
   {"dst": "SEG.2.inter", "rule": "spinal", "src": "SEG.1.inter"},
   {"dst": "SEG.3.inter", "rule": "spinal", "src": "SEG.2.inter"},
   {"dst": "SEG.4.inter", "rule": "spinal", "src": "SEG.3.inter"},
   {"dst": "SEG.5.inter", "rule": "spinal", "src": "SEG.4.inter"},
   {"dst": "SEG.6.inter", "rule": "spinal", "src": "SEG.5.inter"},
   {"dst": "SEG.7.inter", "rule": "spinal", "src": "SEG.6.inter"},
   {"dst": "SEG.8.inter", "rule": "spinal", "src": "SEG.7.inter"}
  ],
  "total": 4000
 },
 "N4_valid": {"counts": {"E": 81, "PV": 20}, "exact": {"E": 80.8, "PV": 20.2}},
 "natural_order_probe": ["L1", "L2", "L10", "SEG.2", "SEG.10"]
}
```

## Appendix B — invalid probes

| ID | Expression | Outcome |
|---|---|---|
| N1 | `P = [a:0.7, b:0.5]` | `E_PROPORTION_INVALID` (type) |
| N2 | `V1.L1.E > V1.L4.PV` with `C_{L1} = {PV,SST,VIP}` | `E_ADDRESS_UNRESOLVED` (type) |
| N3 | `x : SEG^8 : y` intended as a chain | **well formed, no error**: eight disconnected instances, zero inter-instance projections. The language prevents the mistake by never implying adjacency, and the realization receipt makes the absence visible. Use `O[spinal](SEG^8)` |
| N4 | `N = 101`, `P = [E:0.8, PV:0.2]`, no declared `R` | `E_ALLOCATION_POLICY_MISSING` (realize). With `R` declared: `(81, 20)` |
| N5 | `A X[k] B X[k] C` unbraced, no associative policy | `E_X_GROUPING` (normalize) |
| N6 | `V1 O[ff] {P X[lat] W}` with `W` declaring no `in` | `E_FRONTIER_UNRESOLVED` (frontier) |

N3 is the case worth keeping in view: it is the one mis-specification the language answers
by construction rather than by an error.

## Appendix C — superseded documents

`00_TFNE_ALGEBRA_SPEC.md` (v1 semantics), `01_SYNTHETIC_CONFORMANCE.md` (`V1_FIXTURE`),
`04_SOURCE_MIGRATION.md`, `05_V2_MIGRATION_DECISIONS.md`, `06_V2_ADVERSARIAL_CORPUS.md`.
All retained as the lineage record. `02_JAXFNE_COMPATIBILITY.md` is unaffected: it records
engine facts, and its probes parse no TFNE notation.
