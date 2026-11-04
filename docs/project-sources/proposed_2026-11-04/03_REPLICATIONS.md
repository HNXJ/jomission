# JaxFNE development replication audit for Jomission agents
Source: dev `docs/start/replications.md` generated from `dashboard/sources.json`. Dashboard statuses are reported, NOT independently rerun. Distinguish paper-level phenotype, public library-path P6, and core method tests. Replica score is panel coverage, not probability of correctness.

| Study | Dev evidence | Implication |
|---|---|---|
| CAMINOS PLoS CB 2026 Fig.1 | P6, 48/100; library PASS; paper mixed; PV→SOM lag 6.47 vs 8.5 ms | API route works; not a full interneuron physiology replication; delay sensitivity matters |
| CAMINOS Cell Reports 2025 | P2, 71/100 | In progress |
| Izhikevich 2003 | P6, 58/100; library PASS | Emitter path works, incomplete paper panels |
| Brunel 2000 | P5, conditional reproduction; CV 0.40–0.41, κ 0.20 at C | AI classifier was amended; not proof of generic cortical irregularity |
| Potjans–Diesmann 2014 | P3, 2/100 | Laminar microcircuit not qualified |
| Vogels 2011 | P3, 0/100 | Plasticity method checks ≠ paper-level balanced AI network |
| Destexhe 2009 | P3, 4/100 | Self-sustained AI state not established |
| GSDR 2026 | P3, 0/100 | Spectral phenotype not established |
| Mejias 2016 | P2, 0/100 | Functional multi-area hierarchy not established |
| Predictive coding light Fig.3 | P5, 67/100 | Partial target panel coverage |
| Emitter/field/plasticity/numerics | method PASS | Primitive tests, not physiological validation |

## Required interpretation
A failed or pending paper reproduction is not automatically a core defect. A method PASS does not imply a network operating regime. P-054 shows that copying a/b/c/d without the source voltage equation can invert excitability. For Jomission, test phase planes and current thresholds per cell class, per-cell ISI and synchrony, native history dependence, causal FF/FB, and H/HDP behavior separately. Before borrowing a paper threshold, verify estimator, configuration, units and source-equation equivalence.
