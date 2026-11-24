# SENSORY-INPUT AUDIT — external-drive construction vs cortical mechanism

**Audit:** read-only; no configs or sealed artifacts changed.
**Scope:** A_RFoff_RateRef (4f9fdeae7428199a, uniform drive 5.0), B_RFon_RateRef (e5e331a140ebd37e, RF graded), C_RFoff_RateSlow.
**Hash gate:** every consumed artifact verified against the sealed `artifact_hashes.json` of its cell — all MATCH.

---

## 1. The 185× drive-energy discrepancy — CONFIRMED (exact numbers)

Energy = Σ amplitude over steps (dt = 0.1 ms, 5310 steps/slot), from sealed `external_drive_examples.npz` (StimulusSchedule.to_array output).

| | A_RFoff (uniform) | B_RFon (graded V1) | ratio A/B |
|---|---|---|---|
| AAAB trial total | **42,480,000.0** | **230,180.8** | **184.55×** |
| AXAB trial total | 31,860,000.0 | 172,652.5 | 184.55× |
| per-slot (p1..p4) | 10,620,000.0 | 57,528.4 / 57,528.4 / 57,528.4 / 57,595.9 | 184.6× |
| per-unit per-slot | 26,550.0 (5310 × 5.0) | graded 2.91–5.0 amplitude | — |
| per-unit per-trial | 106,200.0 | — | — |
| units driven | **400 / 400** | **6 / 12 targets** (V1 L4 E 41–43 A-blob; 46–48 B-blob) | — |

Construction trace: A uses `condition_to_stimulus_schedule(n_neurons=400, drive_amplitude=5.0)` (jomission/paradigm/spec.py:108) → **uniform** amplitude on p1–p4 for **all 400 units**. B uses `RFOperator.to_stimulus_schedule(tier="graded", base_amplitude=5.0)` (jomission/network/rf.py:569) → per-unit graded amplitudes applied only to the 12 V1 L4 E/PV `target_indices`; in AAAB only 6 of those exceed the sparsity threshold. The claim's "12 targets" = configured target set; actual AAAB drive touches 6.

**The ~185× claim is exact (184.55×).**

## 2. Where heterogeneity disappears — measured variance at each stage

Chain: **parameters → synaptic input → V_m → spike timing → integrated count.**

| Stage | Data | Measurement | Verdict |
|---|---|---|---|
| External drive | sealed `external_drive_examples.npz` | A/C: identical 26,550/unit/slot for all 400 (uniform). B: graded, 6 units only. | **Heterogeneity absent at input in RFoff**; present in RFon (V1 only) |
| Izhikevich params | in-memory model, config_hash re-verified = 4f9fdeae7428199a | `a,b,c,d,drive,sign,u0` **identical within (area, cell-type) class** (unique=1); only `v0` is per-neuron random (400 unique) | **Class-constant params** |
| Connectivity W | model `emitter.W` (400×400) | E rows **not** identical (max row diff 0.075–0.099; 99–109 nonzeros/row) | Heterogeneity present |
| Synaptic input | no current traces retained | — | **UNRESOLVED** |
| V_m | only `V_m[:1000,:10]` in B/C (units 0–9, no metadata) | sample col-std 0.90–10.24 mV | **UNRESOLVED (population)** |
| Spike timing | C rasters (46240×400) | V1 E pairwise mismatch **0.31%** of steps (not identical) | Heterogeneity present but tiny |
| Integrated count | sealed `spikes_example_*.npz` | E counts **exactly identical** (std=0): A 74/67/73, C 74/68; B 49 (V4/FEF/PFC) | **Degenerate observable** |

**Where it disappears:** the integrated E count is a degenerate function of *(class-identical a,b,c,d × uniform drive)*. The per-neuron `v0` and the heterogeneous connectivity W only produce ~0.3% spike-timing jitter — too small to change counts. In RFon, count heterogeneity appears **only** in the graded-drive V1 target units (49–68); everywhere else still var=0.

## 3. Omission-vs-intact drive difference vs rate decrement

- **A (RFoff):** omission (AXAB − AAAB) removes **exactly 26,550 drive-units from every one of the 400 units** — a **spatially uniform global flash-off**, not a local retinotopic omission. Every E unit loses **7 spikes** (−1.51 Hz); area decrements −2.64 (V1) to −3.19 Hz (PFC).
- **B (RFon):** omission removes drive **only from units 41–43** (15,479 / 26,550 / 15,479); units 41–43 decrement 4–6 spikes, all other units unchanged. Area decrements −0.004 to −0.027 Hz (**~1% of A**).
- **Coupling:** the 184.55× drive gap maps to a ~98× omission-effect gap → the phenotype amplitude is a near-linear readout of **removed drive energy**.

## 4. Verdict: **DRIVE_DOMINATED** (model property, not a bug)

The omission phenotype amplitude, its spatial uniformity, and the zero E spike-count variance are controlled by **how external drive is constructed** (uniform-global vs retinotopic-graded) and its energy — not by hierarchy/HDP. The "omission" in RFoff is a global 400-unit drive step-down; the E population is a single degenerate unit (identical counts), and the effect scales linearly with removed drive.

**Caveats:** hierarchy is not inert (area decrements differ: V1 −2.64 vs PFC −3.19); per-unit synaptic input / population V_m / per-unit H are UNRESOLVED (not retained); B's omission effect is near the noise floor.

**Recommendation:**
- **v0.1:** record as **F-01** in the failure atlas — a MODEL property (drive-construction dominates; RF factor confounded with 184.55× drive energy). Do not repair the model.
- **v0.2:** before any RF interpretation, the factorial design MUST add a **drive-energy-matched control** (RFoff at matched total energy, or normalized drive) so the RF factor is not co-varied with drive magnitude. Design constraint, not a model fix.