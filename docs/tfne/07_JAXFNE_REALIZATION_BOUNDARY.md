# JaxFNE realization boundary for the whole-system checkpoint

```
Branch:     whole-system-realizer
Engine:     jaxfne 0.4.24, jax 0.10.1
Authority:  TFNE/2 SEALED (docs/tfne/TFNE_V2_SPEC.md, imported from tfne-v2-normalization)
Scope:      which JaxFNE construction APIs may realize a TFNE normal form without adding
            projections the expression does not declare. No simulation.
```

Checkpoint §12.5-12.6 require inspecting compiler conformance and proving that no
topology-changing defect can corrupt the architecture. The identifier `TFNE2-08` from the other
session exists nowhere in this repository; the semantic failure class it names — a builder
generating projections beyond the normal form — is real and is what these probes test.

## Probe 1: replication without connectivity

`A^n` means *n* instances and no connectivity. The scaffold must reproduce that exactly.

```python
cfg = build_multi_area_columns(areas=("A1","A2","A3"), n_per_area=60,
                               layers=CANONICAL_LAYERS_6L, p_feedforward=0.0, p_feedback=0.0)
```

180 neurons, 10620 edges, **cross-area edges: none**. The multi-area builder at zero FF/FB
probability is a faithful scaffold for replication: within-column microcircuitry only.

## Probe 2: `connect_columns` is not usable as an `O[k]` realization

```python
cfg = connect_columns(cfg, "A1", "A2", mode="sparse", feedforward_gain=0.65, feedback_gain=0.5)
```

Realized cross-area edges: 62, **all of them A1 to A2**; A2 to A1 is empty despite
`feedback_gain=0.5`. The realized A1 to A2 motif is compound:

| source | target | edges |
|---|---|---|
| L6.E | L5.E | 18 |
| L3.E | L4.E | 14 |
| L2.E | L4.E | 9 |
| L2.E | L4.PV | 4 |
| L3.E | L4.VIP | 4 |
| L6.E | L5.PV | 3 |

Two defects against TFNE/2 semantics:

1. **A directional convenience call emits a mixed motif.** `L2/L3.E -> L4` is the feedforward
   projection; `L6.E -> L5.E` in the same call is a second, anatomically feedback-like
   projection travelling in the feedforward direction. One TFNE rule clause cannot own both, so
   a single `connect_columns` call cannot be the realization of a single declared projection.
2. **The declared feedback gain realizes nothing.** A rule that declares a reciprocal projection
   would realize `E_missing` under this builder.

Either way the realized graph is not `G` derived from `NF(A)`, so `connect_columns` is rejected
for this lineage. This is the failure class, demonstrated rather than assumed.

## Probe 3: the admissible realization primitive

`Configuration.connections()` — "Declare a connection rule that `construct()` compiles into
edges" — takes one named rule with explicit `source` and `target` selectors, probability,
weight, sign and mechanism. `AreaConnection` carries the same identity shape:

```text
source_area, source_layer, source_neuron_type,
target_area, target_layer, target_neuron_type,
mechanism, static, plastic
```

That is a TFNE projection identity, one declaration per generated projection. The realizer will
therefore build the scaffold with `build_multi_area_columns(p_feedforward=0.0, p_feedback=0.0)`
and emit one `connections()` declaration per projection in `G`, then reconcile realized edges
against `G` and require `E_missing = E_extra = 0`.

## Consequence for the realizer

```text
NF(A)  ->  one connections() declaration per projection in G  ->  construct()  ->  edge receipt
```

No convenience inter-area builder appears in that path. `build_multi_area_columns` is used only
as a zero-connectivity scaffold, which probe 1 shows is exactly what it is at zero FF/FB.
