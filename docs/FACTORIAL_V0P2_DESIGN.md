# 2×2 Factorial Plasticity-Rate × RF — v0.2 Frozen Design (RF×Rate)

**Version:** `rf_rate_factorial.v0.2.0` — **FROZEN**, preregistered, not tuned to results.
**Manifest:** `manifests/factorial_v0p2_design.json` (`sha256[:16] 378492a24c8b331d` of the design body)
**Supersedes:** `rf_rate_factorial.v0.1.0` after the three identification failures at `14165f9`.

## 1. Fixes the three v0.1 identification failures

1. **Corrected A identity.** Cell A (`A_RFoff_RateStd`) must run with `hp_hash bb8277e7a8e0bca2` (`K_HDP = 0.0`), `config_hash 4f9fdeae7428199a`. v0.1's `exec_A` froze the RF-ON canonical hash `f327f9d2` and never applied the `K_HDP=0.0` override. Exact hp dict (all keys): `K_HDP 0.0`, `K_w_ctrl 0.001`, `K_ctrl 0.15`, `tau_0_ms 5.0`, `alpha 0.05`, `gamma 0.5`, `beta 0.0`, `delta 0.0`, `C_spike 0.0`, `rho_passive 0.0`, `H_min 0.1`, `H_max 10.0`, `w_floor 0.01`, `w_ceiling 10.0`, `barrier_*`, `H_boost_gain 4.0`, `size_scale_by_cell_type {E:5, PV:1, Inl:1, SST:1.5, Ing:1.5, VIP:1.5}`. Verified live at freeze. Cell → `(hp_hash, config_hash)`: A `bb8277e7a8e0bca2`/`4f9fdeae7428199a`, B `f72a489841810a4b`/`4f9fdeae7428199a`, C `f327f9d2ad64cc88`/`e5e331a140ebd37e`, D `b326f7201c59b803`/`e5e331a140ebd37e`. Runtime asserts the `(factor, hp_hash, config_hash)` tuple; the v0.1 B/C label swap is deprecated.

2. **Energy-matched RF.** v0.1 imbalance measured live: RFoff uniform 42,480,000 vs RFon V1-graded 230,181 per AAAB trial (ratio 184.55). Frozen fix: per-stimulus scale `A_on(s) = 2000/G(s)` with `G(s) = Σ_active (d_u/max_u d_u)`, so every intact p-slot delivers `E_ref = (531/0.1)·400·5.0 = 10,620,000` (L1 over time×units). Sealed table: `A_on(A)=923.0238702332183`, `A_on(B)=921.9403481446361`, `A_on(R)=188.4933551695342`. Frozen constraint: `|E_on/E_ref − 1| < 1%` per intact slot, zero in omission slots — verified pre-run (materialize `to_array`, per-slot L1, ratio assert). `jaxfne.StimulusSchedule.to_array` **does** support per-unit amplitude scaling (per-event `target_indices=[unit]` + distinct amplitude, additive) — confirmed by source inspection and live materialization (AAAB slot ratios 1.000000).

3. **LONG as trajectory/timescale.** Estimands at preregistered ages `t_e ∈ {0, 50.864, 203.456, 601.12, 1202.24} s` (exposure trials 0/11/44/130/260), not endpoint run-means:
   - `D_Θ(t_e) = Θ(t_e; τ_ref) − Θ(t_e; τ_LONG)` (seed-matched, = −Rate-main at t_e) from checkpoint Θ-trajectories.
   - `ΔY(t_e) = Y_omission(t_e) − Y_omission(0)`, where `Y_omission` = slot-resolved `[0,531] ms` omission-vs-intact per position p2/p3/p4, per area (rate + low-gamma field), from 96-trial probes interleaved into the 260-trial exposure (total 740 trials).

## 2. Other frozen requirements

- **Recording (#4):** slot-resolved `[0,531] ms` low-gamma field (30–50 Hz, `area_local` proxy_readout) and per-area rates recorded for **all** cells, all 5 probes, both intact and omission — fixes the v0.1 pre-only `field_proxy` gap. Plus Θ(t)/H(t) at every exposure checkpoint.
- **Seeds (#5):** `{0,1,2,3}` minimum, identical across cells, paired by replicate. n<4 → INCOMPLETE.
- **Completion (#6):** atomic-save **5-flag** (`simulation_terminal ∧ observations_persisted ∧ artifacts_readable ∧ hashes_verified ∧ manifest_committed`) AND n≥4 AND energy-matched verification receipt AND slot-resolved observables present, over phases `[pre, probe_t1, probe_t2, probe_t3, exposure, post, recovery]` — plus identity hashes and schedule gates (P1–P6).
- **Statistics:** 2-way ANOVA RF×Rate per t_e, per position; trajectory interpretation (POSITIVE timescale = growing Rate effect at t_e ≥ 200 s + consistent D_Θ); Θ below `w_floor` → UNRESOLVED, not reinterpreted.

No code was modified. Future execution must implement this spec verbatim under the completion predicate.