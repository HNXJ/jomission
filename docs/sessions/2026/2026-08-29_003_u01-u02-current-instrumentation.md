# Session: `2026-08-29_003_u01-u02-current-instrumentation` — U01/U02 Current Instrumentation: Passive Delayed Grouped Currents

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Session ID | `2026-08-29_003_u01-u02-current-instrumentation` |
| UTC | `2026-08-29T20:33:00Z` |
| Local | `2026-08-29 16:33 (EDT)` |
| Branch | `main` |
| HEAD / code SHA | `f9af396` (full: `f9af396a0235f405eb75d786b01d89166c7f2a9a`) |
| Parent model | `manifests/w4a_blocker_seal.json` (`config_hash: a51711101576c7c3`, `hp_hash: f327f9d2ad64cc88`, `rf_hash: 18c2f0f55229c307`) — GEN2_C021; also `manifests/canonical_full_exposure_seal.json` (`config_hash: 4f9fdeae7428199a`, `hp_hash: f327f9d2ad64cc88`) sealed |
| Config / hash | `config_hash=a51711101576c7c3`, `hp_hash=f327f9d2ad64cc88` |
| Agent roles | `Worker: U01 Passivity Implementer (J) + U02 Grouped Reduction Implementer (J) | Reviewer: FG-01 Production-Path Verifier (A1, R mode) | Session Backfill: G` |
| Env | `Python 3.13.7 | JAX 0.10.1 | jaxfne 0.4.17 | manifests sha256[:16]=e80d53817457dcba (canonical_full_exposure_seal.json)` |

## 2. Goal

- **Research question:** Does passive grouped current recording `I[t,G]=Σ_{e∈g} w_e·syn_state_e[t]` provide production-scale observability for delayed laminar networks without altering dynamics?
- **Acceptance predicate (before results):** POSITIVE if (i) delayed non-HDP `record_grouped_current` PASS with `G≈16` producing ~0.6 MiB vs 0.40 GiB raw, passivity `ΔV_m=0 Δspikes=0`, grouped reduction `max|gc − segment_sum(ec,gids,G)| < 1e-5`, cache/config identity and units/sign/group-ID correct; (ii) zero-delay HDP `record_edge_current` PASS; (iii) delayed+HDP correctly BLOCKED at `jaxfne/_model_simulate.py:280` (`ValueError`); else CONDITIONAL/BLOCKED split per path.
- **Out of scope / non-goals:** HDP finite-delay ring kernel design (`_model_simulate.py:280` guard lift); `jomission/network/builder.py` topology/RF/weight mutation; `manifests/*.json` ledger mutation; frozen evidence re-grade; upstream JaxFNE release beyond `0.4.17` emitters `~193 KiB`.

## 3. Starting authoritative state

Each claim tagged `OBSERVED` / `DERIVED` / `INFERRED` / `MODEL_ASSUMPTION` / `LITERATURE_PRIOR` / `UNRESOLVED`.

| # | Claim | Tag | Authority (manifest / ledger / commit / paper) |
|---|-------|-----|-----------------------------------------------|
| 3.1 | Delayed non-HDP `record_edge_current` (raw `I[t,E]`) opt-in passive at `jaxfne/emitters.py:714` `_simulate_edge_recurrent_izhikevich_delayed` was missing before U01; HDP seam `emitters.py:2846` only zero-delay path existed | `OBSERVED` | `scratch/jaxfne_issue_delayed_edge_current.md:19`, `scratch/jaxfne_issue_delayed_edge_current_v2.md:18` citing `emitters.py:714:896`, `_model_simulate.py:280`, `_pipeline.py:395:538` |
| 3.2 | U01 implements passive raw `edge_current_trace (n_steps,n_edges)` as already-computed `w*syn_state` stacked via `lax.scan` tail, `O(n_steps·n_edges)` ~0.40 GiB (1 s, 10k×10590×4 B) / 0.79 GiB (2 s, 20k×10590) — off by default, `ΔV_m=0 Δspikes=0` | `OBSERVED` | `scratch/jaxfne_U01_passivity.md:15` (`emitters.py:714,996`, `_model_simulate.py:386`, `_pipeline.py:395`) + `scratch/fg01_U01_U02_verification.md:33` |
| 3.3 | U02 implements passive grouped `grouped_current_trace (n_steps,G)` as `segment_sum(edge_current, edge_group_ids, G)` inside kernel, `O(n_steps·G)` ~0.6 MiB at `G≈16` vs 0.40 GiB raw (600×); `G=759` full cross-product at N=400 still 29 MiB, `G=6` minimal `L4_E/PV/SST/VIP→L2/3_E` ~0.23 MiB; `ΔV_m=0 Δspikes=0 ΔH=0` | `OBSERVED` | `scratch/jaxfne_U02_grouped.md:15` (`emitters.py:714,1203`, `_model_simulate.py:421`, `_pipeline.py:389`, `jomission/recording/observables.py:30,92`) + `scratch/fg01_U01_U02_verification.md:33` |
| 3.4 | Delayed non-HDP grouped available at `jaxfne/emitters.py:714` (`record_grouped_current, edge_group_ids, grouped_num_segments`), `_model_simulate.py:386→421` non-HDP 4-way dispatch, `_pipeline.py:395` `compile_step_fn` forwarding, `jomission/recording/observables.py:30` `EdgeCurrentRecording` → `observables.py:81` `build_edge_group_ids` | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:33` live probe `n=20` `(50,9)` and `n=400` `E=10666 G=759→6`; `scratch/jaxfne_U02_grouped.md:97` |
| 3.5 | Grouped reduction faithful: `max|gc − Σ_{e:g} ec| = 1.91e-06` (float32 sum-order) `<1e-5` for generic `G=9` class grouping, `0.0` exact for `G=6` minimal partition; both direct kernel and `Model.simulate` | `OBSERVED` | `scratch/jaxfne_U02_grouped.md:71-93`, `scratch/fg01_U01_U02_verification.md:33` |
| 3.6 | Passivity exact: delayed non-HDP OFF vs edge/grouped/both `ΔV_m=0.0`, `Δspikes=0.0`; HDP zero-delay OFF vs ON `ΔH=0 Δw=0 ΔV_m=0`; continuation `delay_state/prng_key/step_index` identity; `ΔG` via `ΔV_m/Δspikes` grouped ON vs OFF =0 | `OBSERVED` | `scratch/jaxfne_U01_passivity.md:55-88`, `scratch/jaxfne_U02_grouped.md:58-104`, `scratch/fg01_U01_U02_verification.md:33` |
| 3.7 | Cache/config identity: `config_hash` invariant `c4dca2b95df4ecaf` OFF vs ON (topology unchanged, `jaxfne/io.py:42`); compiled-cache key includes `_gid_hash = hash(tuple(ids[:2048]), G, n_edges)` at `_model_simulate.py:466,474` preventing stale JIT capture | `DERIVED` | `scratch/fg01_U01_U02_verification.md:33` check 5; `jaxfne/_model_simulate.py:466` |
| 3.8 | Units/sign correct: native `I_e = w·syn_state` signed (`E +1.0`, `PV/SST/VIP -1.0`), `syn = segment_sum(edge_current, post)` in `current_native = drive + syn + noise_coef·noise`; `is_exc_pre` via `cell_type` `E` prefix else `weight>0` | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:33` check 6; `jaxfne/emitters.py:6`, `IZHIKEVICH_CELL_TYPE_DEFAULTS:55`, `_model_simulate.py:477,863` |
| 3.9 | Group IDs map correctly to `(area,layer,class[,subtype])` via `build_edge_group_ids` key `pre_area×pre_layer×pre_class->post_area×post_layer×post_class`, `G≈16` production collapsed from `G=759` full cross-product at N=400 | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:33` check 7; `jomission/recording/observables.py:81,118` |
| 3.10 | Zero-delay HDP `record_edge_current` PASS — `(n_steps,n_edges)` when enabled, `(50,380)` mean|abs| ~0.026 finite; guard at `_model_simulate.py:50` `_hdp_kernel_kwargs` | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:33` check 2; `jaxfne/emitters.py:2846`, `_model_simulate.py:50,311` |
| 3.11 | Delayed+HDP (`enable_hdp=True` + `delay_steps [20,80,120]`) correctly BLOCKED — `ValueError` at `_model_simulate.py:280` (also `:600`, `:1112`, `emitters.py:1112`) — HDP has no finite-delay ring; not claimed solved by U01/U02 | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:21` matrix row D; `scratch/jaxfne_issue_delayed_edge_current_v2.md:89`, reproduced at `scratch/jaxfne_U01_passivity.md:61` |
| 3.12 | Before this session, delayed+HDP temporal `Corr(I_private, I_SST[t+τ])` and W4a `spikes_{L4_E}→I_{L4_E→L2/3}→V_{m,L2/3}` mediation remained `PROXY W·r·τ` not `REALIZED`; W4a `MECHANISM_OBSERVABILITY_BLOCKER` sealed at `GEN2_C022` | `OBSERVED` | `scratch/jaxfne_issue_delayed_edge_current_v2.md:25`, `manifests/gen2_modification_ledger.jsonl:GEN2_C022` (`w4a_blocker_seal.json`) |
| 3.13 | Seams consumed without `builder.py` mutation: `builder.py:867` `_apply_laminar_delays` → `emitters.py:545` `edge_list_with_delay_ms` delays `[20,80,120]` max120 buf121; `builder.py:49` Poisson `rate 2000 Hz amp 2.0`; `recording/observables.py:161` `partition_currents_by_motif`; `simulation/lifecycle.py:141` diagnostics pull | `DERIVED` | `scratch/fg01_U01_U02_verification.md:43` (`builder.py:49,90,867,1902`) |

## 4. Work performed

Execution ≠ verification — note which artifacts were verified by tests/gates.

| Worker | Assignment | Action | Artifact | Result |
|--------|------------|--------|----------|--------|
| U01 Passivity Implementer (J) | Expose passive raw `edge_current_trace` on delayed non-HDP path (Option A) | Edit `jaxfne/emitters.py:714` sig `record_edge_current=False` + 4-way `lax.scan` dispatch; `emitters.py:996` dispatch; `_model_simulate.py:386` non-HDP branch read `hdp_params["record_edge_current"]` + cache key; `_pipeline.py:395` forward + `542` tail handling — no `H/w/delay_state` mutation | `jaxfne/emitters.py:714`, `jaxfne/_model_simulate.py:386`, `jaxfne/_pipeline.py:395` | done — verified by FG-01 `§2.1 delayed non-HDP Δ0 PASS` |
| U02 Grouped Reduction Implementer (J) | Implement streamed grouped `I[t,G]` as `segment_sum` reduction inside kernel | Extend same seams: `emitters.py:714,1203` sig `record_grouped_current, edge_group_ids, grouped_num_segments` + `879 grouped=_segment_sum`; `_model_simulate.py:421` 4-way dispatch + `_gid_hash`; `_pipeline.py:389` `compile_step_fn` forwarding; `jomission/recording/observables.py:30,92` `EdgeCurrentRecording` + `build_edge_group_ids` | `jaxfne/emitters.py:714`, `jaxfne/_model_simulate.py:421`, `jaxfne/_pipeline.py:389`, `jomission/recording/observables.py:30` | done — verified by FG-01 `§2.3 max|gc-sum_g(ec)| 1.91e-06 PASS` |
| U02 | Jomission seam extension for W4 production scale | Add `OPTIONAL_OBSERVABLES I_grouped_current`, `EdgeCurrentRecording(mode="grouped", group_by=("area","layer","class"))` + `to_hdp_params` translator, docs `G≈16 →0.6 MiB vs 0.40 GiB` | `jomission/recording/observables.py:28,81` | done — verified via N=400 `G=759→6` demo (§6) |
| FG-01 Production-Path Verifier (A1, R) | Read-only integration verification of U01+U02 + blocked path confirmation | Inspect canonical seams (`emitters.py:714`, `_model_simulate.py:280/386`, `_pipeline.py:395`, `observables.py:30`, `builder.py:49/90`) + minimal replay probes `n=20 dt0.1 50 steps [20,80,120]` via `edge_list_with_delay_ms`; `n=400` grouped reduction probe — no `builder.py` mutation, no ledger mutation | `scratch/fg01_U01_U02_verification.md` | done — CONDITIONAL PASS with explicit scoping (see §7) |
| FG-01 | Verify `MECHANISM_OBSERVABILITY_BLOCKER` scoping | Probe `delayed+HDP` at `_model_simulate.py:280` and `:600/1112` — confirm intentionally retained `ValueError` and that any "U01 solved delayed+HDP" is contradicted | `scratch/jaxfne_issue_delayed_edge_current_v2.md:89` + `scratch/fg01_U01_U02_verification.md:25` | done — BLOCKED correctly verified |
| FG-09b Bridge (mechanistic) | HDP_OFF supplementary assay using U01/U02 grouped to test `I_{L4_E→L2/3}→V_m` bottleneck | `RuntimeConfig(enable_hdp=False, hdp_params={record_grouped_current, edge_group_ids, grouped_num_segments})` `G=748-759` → `0.6 MiB at G=16` on frozen C019/C020/C021 hashes, `I(S;Y)` KSG `k=3`, fast-state equivalence `Δbar<0.05 Hz ΔVm<0.02 mV` | `scratch/fg09b_HDPOFF_bridge_raw.json`, `scratch/fg09b_HDP_OFF_bridge.md` | done — supplementary (see §13) |

## 5. Evidence

| Ref | Path | Kind | Hash / receipt | Notes |
|-----|------|------|----------------|-------|
| E5.1 | `manifests/canonical_full_exposure_seal.json` | `manifest` | `sha256[:16]=e80d53817457dcba` | Sealed canonical model; `config_hash 4f9fdeae7428199a hp_hash f327f9d2ad64cc88` |
| E5.2 | `manifests/w4a_blocker_seal.json` | `manifest` | `sha256[:16]=a51711101576c7c3` | GEN2_C021 + BLOCKER seal; `config_hash a51711101576c7c3 hp_hash f327f9d2ad64cc88 rf_hash 18c2f0f55229c307` |
| E5.3 | `manifests/gen2_modification_ledger.jsonl` | `ledger` | — | `GEN2_C019/020/021/022` history; C021 `a5171110` is parent for this session |
| E5.4 | `scratch/fg01_U01_U02_verification.md` | `other` | — | **Authoritative FG-01 verification** — production-path compatibility matrix (8 rows) + 7 check-level verdicts with file:line evidence + repro commands |
| E5.5 | `scratch/jaxfne_U01_passivity.md` | `other` | — | U01 passivity proof — 4 pairs OFF vs ON `ΔV_m/Δspikes/ΔH 0.0`, delayed non-HDP `edge_current_trace (100,380)` mean 0.0104, continuation RNG identity |
| E5.6 | `scratch/jaxfne_U02_grouped.md` | `other` | — | U02 grouped verification — direct kernel/Model/continuation passivity + reduction `1.91e-06 <1e-5`, N=400 `G=759→6` `0.40 GiB→0.23 MiB` table |
| E5.7 | `scratch/jaxfne_issue_delayed_edge_current_v2.md` | `other` | — | Upstream issue v2 — W4a blocker definition, reproduction logs Config A/B/C/D, triage `MISSING_CAPABILITY/BUG/DOC_GAP`, proposed fixes Option A / A-grouped / B / C |
| E5.8 | `scratch/jaxfne_issue_delayed_edge_current.md` | `other` | — | Upstream issue v1 (287 lines) — C012/C013 branch, same guard at `_model_simulate.py:280` + `:572/:1063` |
| E5.9 | `scratch/fg09b_HDPOFF_bridge_raw.json` | `other` | — | Mechanistic bridge raw behind U01a-grouped consumer — `6 keys N=24 per C/seed T=10000 gc (10000,752-759) ~30 MiB full / 0.6 MiB at G=16` passivity `0.0` |
| E5.10 | `scratch/fg09b_HDP_OFF_bridge.md` | `other` | — | FG-09b bridge synthesis — `I_{L4_E→L2/3}` `d>240 p0.001 I0.67` in 6/6, `V_m` heterogeneous strong `G 0.013` shunt, fast-state equivalence PASS `Δbar<0.05 ΔVm<0.02` |
| E5.11 | `jaxfne/emitters.py:714` | `other` | `~193 KiB emitters.py` | U01+U02 seam — `_simulate_edge_recurrent_izhikevich_delayed(record_grouped_current, edge_group_ids, grouped_num_segments)` + `840-1210` 4-way dispatch, doc `O(n_steps·G) 10k×16 ~0.6 MiB` |
| E5.12 | `jaxfne/_model_simulate.py:280` | `other` | — | Guard intentionally retained — `if int(jnp.asarray(edges.delay_steps).sum())!=0: raise ValueError("enable_hdp does not support nonzero edge delay_steps ...")` also `:600`, `:1112`, `emitters.py:1112` |
| E5.13 | `jaxfne/_model_simulate.py:386` | `other` | — | U01 non-HDP dispatch (`record_edge_current`) |
| E5.14 | `jaxfne/_model_simulate.py:421` | `other` | — | U02 non-HDP 4-way dispatch (`record_grouped_current` + `_gid_hash`) at `:474 cache_key = ("simulate_recurrent",..., record_edge, record_grouped, _gid_hash)` |
| E5.15 | `jaxfne/_pipeline.py:389` | `other` | — | `compile_step_fn(record_grouped_current, edge_group_ids, grouped_num_segments)` + `:472 _rec_grouped` → `:489 kernel_kw["record_grouped_current"]=True` |
| E5.16 | `jomission/recording/observables.py:30` | `other` | — | `EdgeCurrentRecording(mode="grouped", group_by=("area","layer","class"))` + `observables.py:81 build_edge_group_ids` |
| E5.17 | `scratch/fg09b_HDPOFF_bridge_runner.py` | `other` | — | HDP_OFF bridge runner (copy-paste repro for `I[t,G]→V_m` + `corr(I_private, I_L4→L23[t+τ])`) |

- **EvidenceRefs:** See E5.4–E5.10 above — FG-01 verification + U01/U02 receipts + upstream v2 bundle + FG-09b bridge.
- **Visualization report:** None this session — no `results/figures/*.png` generated; mechanistic visualization gated on FG-09b grouped measurement (see §14).

## 6. Observations

Compact quantitative tables only — no interpretation.

| ID | Metric | Value | Units | n | Artifact | Notes |
|----|--------|-------|-------|---|----------|-------|
| Obs 6.1 | Delayed non-HDP `grouped_current_trace` available `G≈16` production collapsed (full `G=759` at N=400, minimal `G=6` L4→L2/3) | `G≈16 (759→6 demo)` | groups | 400 neurons `E=10666` | `E5.6` `E5.4` | `n=400 10k×10666 0.40 GiB raw vs 10k×16 0.6 MiB` (≈625 KiB) 600×; 10k×6 0.23 MiB; 20k×16 1.2 MiB. Live probe `n=20 (50,9)` class grouping. |
| Obs 6.2 | Passivity — delayed non-HDP OFF vs `record_grouped_current` / `record_edge_current` / both | `ΔV_m=0.0 Δspikes=0.0` | — | 50 steps `n=20` `[20,80,120]` max120 buf121 | `E5.5` `E5.6` `E5.4` | Zero across 100 steps (U01 §2.1 100 steps) and 50 steps (U02 §2.1-2.3). `ΔH=0 ΔΘ=0` for HDP zero-delay control. |
| Obs 6.3 | Passivity — zero-delay HDP OFF vs ON | `ΔV_m=0 Δspikes=0 ΔH=0 Δw=0` | — | 100 steps `n=20` delays `0` | `E5.5` `E5.4` | `enable_hdp=True` `diag keys ['H_final','H_trace','w_final','w_trace','edge_current_trace']` `edge_current_trace (50,380)` mean|abs| ~0.026 finite |
| Obs 6.4 | Passivity — continuation / RNG identity grouped ON vs OFF | `Δdelay_state=0 Δv=0 Δu=0 Δprng_key array_equal True step_index 10 vs 10` | — | 10 steps `delay_state (121,20)` | `E5.5` `E5.6` `E5.4` | `outputs[0..2] Δ0.0`, `out len OFF 5 vs ON 6` (extra grouped), second segment `20 vs 20` |
| Obs 6.5 | Grouped reduction fidelity — `max\|gc − Σ_{e:g} ec\|` | `1.91e-06` (generic G=9) and `0.0` (N=400 G=6 minimal) | native current | `n=20 50 steps` + `n=400 10 steps` | `E5.6` `E5.4` | Threshold `<1e-5` pass (float32 `segment_sum` vs `numpy` sum-order). `both` case `gc_b` vs recomputed same `1.91e-06`. |
| Obs 6.6 | Raw vs grouped memory at W4 scale | `0.40 GiB raw (10k×10666×4 B) → 0.6 MiB grouped (10k×16×4 B)` ; full `G=759 → 29 MiB`; `0.79 GiB raw at 20k×10666` → `1.2 MiB at G=16` | bytes | `n_steps=10000/20000 dt0.1` | `E5.6` `E5.7` | Hazard `10k×2M×4 B=80 GiB` at `_pipeline.py:395` doc; grouped avoids it (`G≪E`) |
| Obs 6.7 | Cache/config identity — `config_hash` invariant OFF vs ON; `_gid_hash` in compile cache key | `c4dca2b95df4ecaf invariant` ; `_gid_hash hash(tuple(ids[:2048]), G, n_edges)` in `cache_key` | — | — | `E5.4` | `jaxfne/io.py:42` `config_hash` hashes only topology (`Configuration`), not `RuntimeConfig` recording flag; `_model_simulate.py:466` covers first 2048 ids + `G` + `n_edges` |
| Obs 6.8 | Units/sign conventions — `E sign +1.0`, `PV/SST/VIP/I -1.0` calibrations | `+1.0 / -1.0` | native gain | `n_neurons` | `E5.4` | `IZHIKEVICH_CELL_TYPE_DEFAULTS:55`, `EdgeList:597`, `emitters.py:822 syn=segment_sum(edge_current,post)` preserves signed `weight` |
| Obs 6.9 | Group ID construction — `edge_group_ids int[n_edges] ∈ [0,G)` via `area×layer×class` key ordering by first appearance | `G=759 full, G≈16 collapsed, G=6 minimal` | groups | `n_edges 10666` | `E5.4` `E5.16` | `observables.py:81,118` alias `class/cell_type/subtype → cell_type` |
| Obs 6.10 | Delayed non-HDP grouped shape — `grouped_current_trace (n_steps,G)` exists, `edge_current_trace` absent when grouped-only | `(50,9)` delayed `n=20` ; `(10000,752-759)` W4 `n=400` | — | 50–10000 steps | `E5.4` `E5.6` `E5.9` | Probe `diag["grouped_current_trace"] (50,9)` exists, `diag["edge_current_trace"]` absent (grouped-only); both True → `(50,380)+(50,9)` |
| Obs 6.11 | Zero-delay HDP edge semantics — `edge_current_trace (n_steps,n_edges)` shape when `record_edge_current=True` | `(50,380) mean 0.026 finite` ; `(100,380) mean 0.0266` | native | 50–100 steps | `E5.4` `E5.5` | HDP seam `emitters.py:2846` |
| Obs 6.12 | Delayed HDP blocked — `ValueError` at `_model_simulate.py:280` (also `:600`, `:1112`, `emitters.py:1112`) | `enable_hdp does not support nonzero edge delay_steps ... Use the non-HDP finite-delay recurrent kernel (recurrent_backend='edge_list', enable_hdp=False)` | error message | `[20,80,120] sum 27960` | `E5.4` `E5.7` | Reproduced verbatim `n=20 380 edges 27960` — HDP signature lacks `record_grouped_current` (`inspect.signature` → `False`) |
| Obs 6.13 | Continuation pipeline forwarding — `compile_step_fn(record_grouped_current)` → `kernel="baseline"` delayed at `_pipeline.py:461:489` | `use_delays and record_grouped_current → kernel_kw["record_grouped_current"]=True` | — | — | `E5.4` `E5.7` | `_pipeline.py:395→472 _rec_grouped` → `489` forwarding; outputs tail `baseline delayed → gc[0] if gc.ndim==2` |
| Obs 6.14 | Fast-state equivalence HDP_OFF bridge (uses U01a grouped) — `Δbar_r <0.05 Hz`, `ΔV_m <0.02 mV` del vs zeroOff, `passivity ΔV_m/Δspikes=0` | `HDP disabling changes bar_r <0.4% (11-15 Hz), V_m mean <0.03% (−65 mV)` | Hz / mV | `N=400 6 cells C019/C020/C021 2 seeds` | `E5.10` | `fg09b_HDP_OFF_bridge.md:72` 6/6 `Δbar −0.027 to +0.008` `ΔV_m −0.006 to +0.005`; FG-01 `ΔV_m0 Δsp0` confirms `emitters.py:816 edge_current=w*syn_state` already computed |
| Obs 6.15 | FG-09b `I_{L4_E→L2/3_E}` informativeness (grouped consumer) — `I_syn Δ 0.33-1.69 native d>240 p0.001 I_corr 0.674-0.683 acc1.00` in all 6/6 | `d 247-979 p0.001 I0.67` | native | `N=24 per C/seed` `W=[100,531) B_proxy=[0,50)` | `E5.10` | `I_syn` carries `d>240` in all 6; `V_m,L23` heterogeneous: `2/6` invariant `|d|<0.5 p>0.05` strong shunt `G 0.013`, `4/6` small absolute carry `ΔVm 0.13-0.29 mV d0.82-1.92` |

_Figures:_ `None`

## 7. Interpretation

Observation ≠ interpretation. Each claim cites supporting observation ID(s) from §6.

| # | Interpretation | Supporting observations | Confidence |
|---|----------------|-------------------------|------------|
| 7.1 | Delayed non-HDP grouped current `I[t,G]` is production-available, passive, and memory-appropriate — `G≈16` yields ~0.6 MiB at 1 s (600× vs raw 0.40 GiB), enabling W4 mechanistic replay `spikes_{L4_E}→I_{L4_E→L2/3}(t)→V_{m,L2/3}(t)` without `edge×time` materialization via `EdgeCurrentRecording` | `Obs 6.1`, `Obs 6.2`, `Obs 6.6`, `Obs 6.10` | `HIGH` |
| 7.2 | Grouped reduction is numerically faithful to raw per-edge sum — `max|gc−Σ_g ec| <1e-5` (observed `1.91e-06` generic, `0.0` minimal) so W4 motif-level questions can be answered from grouped without loss beyond float32 sum-order | `Obs 6.5` | `HIGH` |
| 7.3 | Recording is chemically passive — OFF≡ON for `V_m/spikes/H/w` and `ContinuationState/RNG` because `edge_current = w·syn_state` is already computed then `syn = segment_sum(edge_current,post)`; recording only stacks tail via Python-bool 4-way dispatch | `Obs 6.2`, `Obs 6.3`, `Obs 6.4`, `Obs 6.14` | `HIGH` |
| 7.4 | Cache/config identity is correct: `config_hash` invariant (topology unchanged) and `_gid_hash` disambiguates JIT compilation so stale `group_ids` capture cannot silently mis-route currents | `Obs 6.7` | `HIGH` |
| 7.5 | Sign/units and group-ID mapping are scientifically correct — excitatory `+` inhibitory `−` preserved, and `(area,layer,class)` key construction via `build_edge_group_ids` matches `observables.py:161` `partition_currents_by_motif` intent (`I_e` vs `I_i` split, `Efrac`) | `Obs 6.8`, `Obs 6.9` | `HIGH` |
| 7.6 | Zero-delay HDP edge recording semantics unchanged — shape `(n_steps,n_edges)` when enabled, passivity `ΔH=0` | `Obs 6.3`, `Obs 6.11` | `HIGH` |
| 7.7 | Delayed+HDP remains correctly BLOCKED — HDP has no finite-delay ring (`bufsize D_max+1`) so `enable_hdp=True` + `delay_steps [20,80,120]` raising `ValueError` at `_model_simulate.py:280` is scientifically intentional, not a bug to be silently bypassed | `Obs 6.12` | `HIGH` |
| 7.8 | U01/U02 only solved delayed non-HDP (B) + zero-delay HDP (C); interpreting "U01 solved delayed+HDP" is contradicted by reproduction — delayed+HDP production claims must remain `PROXY W·r·τ` and `UNRESOLVED` temporal `Corr(spikes,I_delay)` until dedicated HDP delay kernel | `Obs 6.12`, `Obs 6.13` | `HIGH` |
| 7.9 | FG-09b demonstrates U01a-grouped consumer viability: fast-state equivalence (`Δbar<0.05 Hz ΔVm<0.02 mV`) makes HDP_OFF assay identifying for fast currents/`V_m`; heterogeneous `I→V_m` attenuation (`G 0.013` strong shunt in `C021_s0`) together with universal `I_syn d>240` confirms `current→V_m` not `I_syn` generation is bottleneck in W4a | `Obs 6.14`, `Obs 6.15`, `Obs 6.2` | `MED` |

## 8. Negative / insufficient results

Bounded vocabulary: `NEGATIVE_INSTANCE` | `NULL_RESULT` | `INSUFFICIENT_POWER` | `INCONCLUSIVE` | `RESOURCE_BOUNDARY` | `BLOCKED`; `FALSIFIED` reserved for `L0`–`L5` criterion failures.

| # | Claim | Verdict | Level / threshold | Notes |
|---|-------|---------|-------------------|-------|
| 8.1 | Delayed+HDP (`enable_hdp=True` + `delay_steps [20,80,120]` + `record_edge_current` or `record_grouped_current`) — production delayed plasticity current observation | `BLOCKED` | `ValueError at jaxfne/_model_simulate.py:280` (also `:600`, `:1112`, `emitters.py:1112`) — identical guard | Intentionally retained; HDP has no ring buffer. Must keep `PROXY W·r·τ` + `UNRESOLVED Corr(I_private, I_SST[t+τ])` and `spikes_{L4_E}→I_{L4_E→L2/3}→V_m` for any HDP-delay plasticity claim until dedicated kernel. |
| 8.2 | Delayed+HDP grouped reduction path (HDP seam lacks `record_grouped_current` signature) | `BLOCKED` | `inspect.signature` `record_grouped_current` → `False` on HDP kernel | Same guard fires before any recording dispatch; no `grouped_current_trace` possible on HDP delayed. |
| 8.3 | `C019/C020/C021` delayed+HDP information transmission claim | `BLOCKED` | `_model_simulate.py:280` blocks realization; W4a `GEN2_C022` BLOCKER sealed `MECHANISM_OBSERVABILITY_BLOCKER` | Deployed delayed+HDP numbers remain `PROXY` per `gen2_C012_verify.json:38` + `review_C013_transfer.md:2.1`; see E5.7 v2. |
| 8.4 | Zero-delay HDP vs delayed non-HDP numerical equivalence of `I[t]` | `INCONCLUSIVE` | Delays change dynamics — same `(n_steps,n_edges)` shape but temporally shifted (`I[t] ↔ spikes[t−D]`) | Valid comparison is OFF≡ON within same delay regime, not cross-regime equality (see U01 §2.4). |

_If none:_ `No negative/insufficient results — all acceptance predicates passed (see §6).` — **not applicable; 8.1–8.3 BLOCKED as above.**

## 9. Unexpected findings

| # | Finding | Why unexpected | Follow-up priority |
|---|---------|----------------|-------------------|
| 9.1 | Fast-state equivalence between HDP_OFF and HDP_ON/del vs zero-delay is tighter than expected: `Δbar<0.05 Hz` (<0.4% of `bar 11-15 Hz`), `ΔV_m<0.02 mV` (<0.03% of `−65 mV`) across 6 C/seed cells, while `H` differs (null vs ~1.0) — separating fast plasticity observability from fast drive | Before U01, HDP was assumed to materially modulate fast `bar_r/V_m` (tonic 3.0 + `μ_total 3.405`); instead HDP is slow, not fast drive, so HDP_OFF grouped assay is identifying for `I→V_m` | `high` — enables FG-09b bridge without confounding fast dynamics; still not canonical for `H/Θ` claims |
| 9.2 | Grouped reduction exact `0.0` for minimal `G=6` partition at N=400 (vs `1.91e-06` for generic `G=9`) due to partition structure — float32 sum-order coincidence | Expected uniformly small <1e-5, but exact 0.0 suggests specific grouping reduces rounding dispersion | `low` — no action; both <1e-5 PASS |

_If none:_ `None.` — addressed above.

## 10. Tool / workflow friction

| Friction | Cause | Workaround | Permanent repair? |
|----------|-------|------------|-------------------|
| `enable_hdp does not support nonzero edge delay_steps` guard at `jaxfne/_model_simulate.py:280` (also `:600`, `:1112`) blocks production delayed+HDP current observability; redirect message suggests `enable_hdp=False` alternative that also lacked `record_*_current` before U01/U02 | JaxFNE HDP kernel has no finite-delay ring (`bufsize D_max+1` at `emitters.py:714`); U01/U02 intentionally kept guard for HDP (scientific trajectory) | Delayed non-HDP now solved via U01/U02 (B path); delayed+HDP remains documented `PROXY`; FG-09b HDP_OFF bridge provides mechanistic assay with fast-state equivalence `Δbar<0.05` | `JAXFNE_UPSTREAM` — HDP finite-delay ring kernel or explicit `record_grouped_current` on HDP + documentation of `PROXY` bound and `UNRESOLVED` temporal mediation (proposed in §12) |

## 11. Learned lessons

| Lesson | Trigger | Cause | Repair | Evidence | Scope | Confidence | Persistence |
|--------|---------|-------|--------|----------|-------|------------|-------------|
| Passive already-computed quantity should be exposed via `lax.scan` tail stacking, not new state — `edge_current = w·syn_state` was computed at `emitters.py:821,990` then `syn = segment_sum(edge_current,post)` so recording only appends diagnostic buffer | Need to observe `I[t,E]=W·syn_state[t]` without `ΔV_m` | New buffer would duplicate logic downstream (NEVER-02/03) | 4-way Python-bool dispatch `edge/current × grouped` at `emitters.py:840`, no `drive/syn/decay/spike` change | `E5.5` `Obs 6.2` `Obs 6.4` | `project` | `HIGH` | `YES` |
| Grouped `I[t,G]=segment_sum(I[t,E], gids, G)` inside kernel gives 600× memory saving and is numerically faithful within `1e-5` for `G≪E` | `O(n_steps·n_edges)` `0.40 GiB raw` dominated W4 | Materializing `edge×time` before grouping | Streamed `grouped = _segment_sum(edge_current, _group_ids, _G)` per step at `emitters.py:879` + `observables.py:81` `build_edge_group_ids` | `Obs 6.5` `Obs 6.6` `E5.6` | `project` | `HIGH` | `YES` |
| `config_hash` (topology) must stay invariant while `RuntimeConfig` recording flags disambiguate via `_gid_hash` in JIT cache key | Caller needs identical `config_hash c4dca2b95df4ecaf` OFF vs ON but stale JIT capture would mis-route groups | Single hash for both config and compilation | `jaxfne/io.py:42` topology hash + `_model_simulate.py:466` `hash(tuple(ids[:2048]), G, n_edges)` at `:474` | `Obs 6.7` `E5.4` | `harness` | `MED` | `YES` |
| `MECHANISM_OBSERVABILITY_BLOCKER` (W4a `GEN2_C022`) correctly distinguishes delayed non-HDP (solved) from delayed+HDP (blocked) — do not conflate U01a pass with delayed+HDP claim | Three large interventions left `V1 L2/3 Δ0.02 acc0.542` invariant while presumed causes changed `ΔM +2652` | Missing `spikes→I→V_m` chain inside `D_max 120 buf121` ring | Keep delayed+HDP `PROXY` + `UNRESOLVED`, use HDP_OFF grouped only for fast dynamics (not `H/Θ`) per FG-09b decision matrix | `Obs 6.12` `E5.2` `E5.10` | `project` | `HIGH` | `YES` |

## 12. Harness / tool proposals

Type ∈ `PROJECT_RULE` | `TEST/GATE` | `JOMISSION_TOOL` | `JAXFNE_UPSTREAM` | `DOCUMENTATION` | `VISUALIZATION` | `AGENT_WORKFLOW`

| # | Type | Target | Rationale | Evidence |
|---|------|--------|-----------|----------|
| 12.1 | `JAXFNE_UPSTREAM` | `jaxfne/emitters.py:714` `jaxfne/_model_simulate.py:280/600/1112` `jaxfne/_pipeline.py:395` — persist upstream grouped streaming `record_grouped_current` as first-class affordance for motif-level `I[t,src_group,tgt_group]` without `edge×time` materialization (`G≈16→0.6 MiB`) | Broadly useful beyond W4a; avoids `80 GiB at 10k×2M` hazard (`_pipeline.py:395` doc) while exposing `I_{L4→L2/3}→V_m` mediation + inhibitory components | `Obs 6.1` `Obs 6.6` `E5.6` |
| 12.2 | `TEST/GATE` | `docs/sessions/validate.py` + `tests/` — add explicit non-HDP delayed `record_grouped_current` passivity gate (`ΔV_m=0 Δspikes=0 max\|gc−Σ_g ec\|<1e-5` at `[20,80,120]`) and negative gate asserting `ValueError` at `_model_simulate.py:280` for delayed+HDP | Prevents regression to silent drop (Config C) and prevents false "solved delayed+HDP" claim | `Obs 6.2` `Obs 6.5` `Obs 6.12` |
| 12.3 | `JAXFNE_UPSTREAM` | `jaxfne/_model_simulate.py:281` error message + `jaxfne/emitters.py:3066` docstring — clarify redirect: state that `enable_hdp=False` non-HDP **now** supports `record_*_current` via U01/U02, but `enable_hdp=True` + delays remains unsupported and `PROXY W·r·τ` temporal coherence stays `UNRESOLVED` without it | Current 0.4.17 message redirects to non-HDP without mentioning diagnostic; now partially misleading since non-HDP diagnostic exists but HDP+delay does not | `Obs 6.12` `E5.7` §5 Option C |
| 12.4 | `JOMISSION_TOOL` | `jomission/recording/observables.py:161` `partition_currents_by_motif` — consume either `edge_current_trace (n_steps,n_edges)` or `grouped_current_trace (n_steps,G)` and emit `Efrac` / `E/I` split / `η_rec = ‖I_rec‖/‖I_total‖` for FG-10 | Enables `FG-10` 0-DOF measurement of `η_rec` and `corr(I_private, I_{L4→L2/3}[t+τ]) τ 10-20 ms` on frozen hashes before any `builder.py` DOF | `Obs 6.10` `E5.10` |

## 13. Scientific state transition

| Field | Value |
|-------|-------|
| **Before** | `MECHANISM_OBSERVABILITY_BLOCKER` sealed (`GEN2_C022` `w4a_blocker_seal.json` `a51711101576c7c3`): `spikes_{L4_E}→I_{L4_E→L2/3}→V_{m,L2/3}` at `delays [20,80,120] max120 buf121` blocked — `record_edge_current` existed only for zero-delay HDP (`emitters.py:2846`), non-HDP delayed kernel `emitters.py:714` had no diagnostic, `compile_step_fn` HDP-only guard at `_pipeline.py:538`, `PROXY W·r·τ` flagged across C012/C020/C021 (`E5.7`). |
| **After** | Delayed non-HDP `I[t,G]` streamable: `record_grouped_current` + `edge_group_ids` at `emitters.py:714` via `_model_simulate.py:421` 4-way dispatch and `_pipeline.py:389` forwarding + `observables.py:30,81` grouping; `G≈16 →0.6 MiB` (600× vs 0.40 GiB raw) passive `Δ0` and reduction `<1e-5`; zero-delay HDP unchanged; delayed+HDP correctly BLOCKED at `_model_simulate.py:280` (not claimed solved). |
| **Unlocked** | W4 mechanistic replay for `spikes_{L4_E}→I_{L4_E→L2/3}(t)→V_{m,L23}(t)` without `edge×time` materialization — `REALIZED` `grouped_current_trace (n_steps,G)` partitioning via `observables.py:161` (E/PV/SST/VIP `I_e/I_i`) + fast-state-equivalent HDP_OFF assay (`Δbar<0.05 ΔVm<0.02` per `E5.10`) for `B→C` bottleneck test (`I→V_m` shunting `G 0.013` strong vs heterogeneous graded). |
| **Still gated** | Canonical delayed+HDP `I[t,G]` for plasticity (`H/Θ`) claims — `enable_hdp=True` + `[20,80,120]` still raises `ValueError` (`E5.12`). W4b/W5, typed vertical beyond `GEN2_C020 Ne16→54`, and `I_bg(λ)` builder sweeps remain gated until FG-10 measures `η_rec` and `corr(I_private, I_{L4→L2/3}[t+τ])` on grouped. Zero-delay is reference not expected identical to delayed (delay ring changes temporal dynamics). |
| **Invalidated** | Claim that "U01 solved delayed+HDP" — explicitly `NEGATIVE claim contradicted by reproduction` (`E5.4` §1 matrix rows D). No longer valid to treat `PROXY W·r·τ` as equivalent to `REALIZED` for temporal `Corr(spikes,I_delay)` on delayed+HDP. |
| **New uncertainty** | Whether `η_rec = ‖I_rec‖/‖I_total‖` is `≪0.1` on grouped trace (needed before typing vertical `Ne/k/C` `builder.py:210/220` `0.12 max40` or `I_bg(λ)` `g_bg 1.288` escalation, 1 DOF at a time); whether upstream JaxFNE will add HDP finite-delay ring (Option A-grouped as upstream affordance vs documentation-only Option C). |

## 14. Next action

- **Primary (highest-value):** **FG-09b→FG-10 bridge: measure `I_{L4_E→L2/3}(t)` via delayed non-HDP grouped `G≈16` (or minimal `G=6` `L4_E/PV/SST/VIP→L2/3_E` + other `observables.py:81`) on frozen `GEN2_C021 a51711101576c7c3` hashes across `C019/C020/C021 2 seeds` with `I(S;Y)` + `corr(I_private, I_{L4→L2/3}[t+τ]) τ10-20ms` + `dV/dI` and `η_rec` partition** — why highest value: achievability `immediate` (U01a PASS grouped `0.6 MiB` passive, `E5.10` fast-state equivalent), unlocks localization of W4a `current→V_m` bottleneck (`G 0.013` strong shunt vs graded `0.15-0.85`) without builder DOF; **ready predicate:** none — `RuntimeConfig(enable_hdp=False, hdp_params={record_grouped_current, edge_group_ids, grouped_num_segments})` available now via `E5.9` runner; `C_next FORBIDDEN until J` (`mechanistic_replay_freeze.md:162`). Produce `transfer_matrix.csv` then escalate to `I_bg(λ)` only if `η_rec≪0.1`.
- **Secondary (optional):** **Propose upstream JaxFNE grouped streaming as first-class `JAXFNE_UPSTREAM` affordance** (emit `JAXFNE_REQUEST` with `emitters.py:714` + `_pipeline.py:395` diff + `build_edge_group_ids` spec) — why: makes grouped `O(n_steps·G)` broadly useful beyond W4 and avoids `80 GiB` hazard for all delayed models. Ready when FG-10 measurement receipts posted.

## 15. Progress score

| Field | Value |
|-------|-------|
| Current | `74` |
| Previous | `73` |
| Delta | `+1` |
| Reason | `U01a PASS — grouped 0.6 MiB passive delayed non-HDP via jaxfne/emitters.py:714 → _model_simulate.py:421 → _pipeline.py:389 → observables.py:30, verified FG-01 fg01_U01_U02_verification.md:33 (ΔV_m/Δspikes=0, grouped reduction 1.91e-06 <1e-5, G≈16 625 KiB vs 0.40 GiB, cache _gid_hash, units/sign, group IDs PASS; zero-delay HDP PASS; delayed+HDP correctly BLOCKED at _model_simulate.py:280), HDP_OFF fast-state equivalence Δbar<0.05 ΔVm<0.02 in fg09b_HDP_OFF_bridge.md:72; U01b still blocked — no canonical delayed+HDP I[t,G]. Score reflects verified capability (tests/gates/manifests), not activity.` |

---

**Checklist before seal:**

- [x] All 15 sections present and in order
- [x] §1 `session_id` matches filename stem; timestamps ISO-8601
- [x] §5 every artifact exists; figures link to visualization
- [x] §6 tables have units, n, artifact refs; no interpretation
- [x] §7 each claim cites `Obs 6.x`
- [x] §8 bounded vocabulary used correctly; `FALSIFIED` only with `L0`–`L5` ref
- [x] §10 friction normalized (for recurrent detection)
- [x] §11 `Persistence` and `Confidence` enums correct
- [x] §15 score delta justified by verified capability
- [x] JSON sidecar `*.json` created and matches `SCHEMA.md`
- [x] `python docs/sessions/validate.py <this file>` passes
