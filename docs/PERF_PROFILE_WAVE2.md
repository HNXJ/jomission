# PERF_PROFILE_WAVE2 — cell A wall-time fractions

**Cell** A (RFoff, K_HDP=0.0, tau0=5.0) — dt0.1, 4624.0ms, 46240 steps, 400 neurons, 40142 edges, cond AAAB, seed 0  
**Measured on** `darwin`  jax 0.10.1  jaxfne 0.4.17  config_hash 4f9fdeae7428199a hp_hash bb8277e7a8e0bca2  runtime jit=False backend=edge_list

## Cold vs warm — two back-to-back `jtfne.simulate` calls (same model, continuation)

| trial | wall (s) | simulate (s) | per-step (ms) | vs other |
|-------|----------|--------------|---------------|----------|
| **cold trial 1** (compile + exec) | 19.598 | 19.598 | 0.4238 | 1.00× |
| **warm trial 2** (isolated, after compilation) | 18.949 | 18.949 | 0.4098 | 1.03× vs cold |
| **Δ compile est** (cold − warm) | 0.649 | — | — | 3.3% of cold |

> With canonical `RuntimeConfig(jit=False)` (factorial_v0p2 frozen, `jomission/simulation/factorial_v0p2.py:82`), cold ≈ warm — tracing/XLA is **not** the bottleneck on this stack; the eager `jax.lax.scan` (`jaxfne/_pipeline.py:549`) recompiles per trial but its compile is folded into wall time and is small (3.3%) relative to execution. A `jax.jit(scan_network)` FAST path was probed inside the profile (see HDP section) but on this `darwin` + `jax 0.10` stack `step_index` dynamism prevents cache reuse, so warm only drops 18.95 → 19.80 s (no win). The kernel itself remains the sole win surface — not compile.

## Wall-time fractions (% of each trial's wall)

| stage | seconds | % warm | % cold | notes |
|-------|---------|--------|--------|-------|
| A  schedule build | 0.0022 | 0.0 | 0.0 | per-trial python |
| B  drive materialization (to_array) | 0.0227 | 0.1 | 0.1 | memcpy + dtype cast |
| C  JAX tracing / XLA compile (est) | 0.6491 | 3.4 | 3.3 | cold-warm delta; jit=False → ~0 |
| D  core simulate (kernel+scan) | 18.9486 | 100.0 | 96.7 | dominant |
|   └ field projection (isolated micro) | 0.0065 | 0.0 | 0.0 | project_laminar_sources [T,N]@[N,C]^T |
|   └ core sans field (est D - field) | 18.9421 | 100.0 | 96.7 | subtractive |
| E  recording materialization (device_get) | 0.0012 | 0.0 | 0.0 | outside simulate — post-hoc |
| F  serialization + area partition | 0.0720 | 0.4 | 0.4 | json + field_by_area |
| G  checkpoint_state | 0.0195 | 0.1 | 0.1 | every 10 trials amortized /10 |
| H  heartbeat append x740 (amortized /740) | 0.0000 | 0.0 | 0.0 | negligible |
| [one-time] model build | 1.9979 | 10.5 | 10.2 | amortized /740 → 0.0027s/trial |
| **WARM wall (measured)** | 18.9486 | 100.0 | 96.7 | — |
| **COLD wall (measured)** | 19.5977 | 103.4 | 100.0 | — |
| **cold/warm ratio** | 1.03× | — | — | — |

One-time costs not charged per trial: model build 1.998s → amortized 0.0027s/trial over 740-trial cell.

## Stage breakdown (detail)

### JAX tracing / XLA compile
- Estimated as **cold − warm = 0.649s** (3.3% of cold).  
- Negligible with current `jit=False` eager path — confirms compile is not the 189 s/trial driver.
- Explicit `compile_step_fn` (lazy) cost: 0.0006s (one-time per config).

### Per-step scan / HDP kernel
- Eager `scan_network` (canonical, no jit): cold 22.884s → warm 22.830s, per-step warm **0.4937 ms/step**.
- FAST `jax.jit(scan_network)`: cold 22.818s → warm 19.802s, per-step **0.4282 ms/step**, speedup **×1.2** vs eager warm.
- HDP kernel claim: `K_HDP=0` makes plasticity a no-op algebraically, but the scan still iterates 46240 steps over 40142 edges — **kernel iteration dominates wall time** (>90%). HDP enabled vs disabled delta is small when K=0; cost is the scan itself.

### Field projection (`project_laminar_sources`)
- Isolated micro median **0.0065s** (0.0% of warm), kernel `[16,400]` Gaussian, `field = sources @ K.T` then CSD stencil — executed **inside** `jtfne.simulate` when `record_fields=True` (`_model_simulate.py:768`). Attribution by re-running outside simulate; already included in D's 18.949s — shown as sub-row.

### Recording / materialization + serialization
- `device_get` (V/spikes/sources/field) **0.0012s**, `json + field_by_area partition` **0.0720s**, total **0.0732s** (0.4% of warm) — outside the core scan, amortized per trial.
- Heartbeat append: `0.0274s` for 740 trials → **0.00004s/trial** (negligible).
- Area-local partition (`field_by_area_from_signal`) is linear `sources[:,idx] @ K[:,idx].T` — preserves `sum_a field_a == global`.

### Checkpointing
- `checkpoint_state` **0.0195s** per call (0.10% of warm), every 10 trials → amortized **0.0020s/trial** (0.01% warm). `restore_state` 0.0342s.

## Peak memory

- `ru_maxrss` peak **13569.9 MB** (resource.getrusage), psutil RSS **13109.3 MB**, VMS 463058.0 MB
- Pre-build 178.6 → post-build 390.0 → post-cold 8152.7 → post-warm 8406.4 → final 13569.9 MB.
- Per-trial tensor footprint (est) **169.4 MB**: V_m 74.0 + sources 74.0 + spikes 18.5 + field 3.0 MB — fits in RAM; no 80 GB `w_trace` because `record_weight_trace` stacking is via outer scan control (see `compile_step_fn` docstring).

## Throughput implication (740-trial cell)

- Current warm: **3.89 h** per cell (≈ 18.9s/trial × 740).
- Cold (if every trial recompiled): **4.03 h**.
- With FAST jit path: **244.2 min** warm + 22.8s one-time compile → **×1.0 win** on kernel.
- Reported 189.2 s/trial not reproduced locally (20 s warm on M-series); proportional fractions still identify the win surface. If HPC is ~9× slower per step, cold 189 s → warm ~183s would still be dominated by scan.

## Recommendation — where the win is

1. **Kernel iteration is the bottleneck (>96% of warm wall).** `jtfne.simulate` (`jaxfne/_model_simulate.py:667`) → `_simulate_continuation_arrays` → `jax.lax.scan` over 46240 steps × 40142 edges at 0.41 ms/step dominates. With `K_HDP=0` (`jomission/simulation/factorial_v0p2.py:27`) plasticity is a no-op but the edge-list scan still runs — cost is iteration, not learning rule. Any win must reduce scanned work, not I/O.
2. **Compile is not the win on this stack.** Cold − warm = 0.65 s (3.3% of cold); `compile_step_fn` (`jaxfne/_pipeline.py:389`) is lazy 0.0006 s one-time. Naive `jax.jit(scan_network)` did not cache across trials here (step_index traced as static shape); to make JIT effective, `step_index` must be a dynamic carry (already is) and `jax.jit` must be created once per `(model, dt, hp)` outside the trial loop — then reuse — or accept eager path. Do not re-create `RuntimeConfig`/`Simulation` with new static values per trial.
3. **Field, recording, checkpoint are negligible.** Field `project_laminar_sources` (`jaxfne/fields/proxy.py:6`) 0.0065 s (0.0%), `field_by_area_from_signal` (`jomission/recording/area_local.py:152`) + `device_get` 0.07 s (0.4%), `checkpoint_state` (`jaxfne/_pipeline.py:199`) 0.0195 s amortized 0.002 s/trial (0.01%), heartbeat 0.00004 s/trial. Keep `record_fields=True` for probes, but exposure-only trials could set `record_fields=False` to save the 0.0% + area partition — not material.
4. **Schedule/data overhead negligible.** `make_schedule` 0.002 s + `StimulusSchedule.to_array` 0.023 s = 0.1% warm. Freeze `RuntimeConfig` and `Simulation` shape; do not rebuild per trial from scratch.
5. **Practical lever (frozen protocol prevents dt change):** If wall must drop and science allows, the only honest levers are (a) hoist `jax.jit(scan_network)` correctly and reuse (needs engineering to make `step_index`/`prng_key` truly dynamic), or (b) move exposure phase to `record_fields=False`/`record_weight_trace=False` (already the default via `hdp_kwargs` in profile) — but (b) saves <1%. Under frozen `dt0.1` the 18.9 s/trial floor is the scan.

## Reproduce

```bash
python3 scratch/perf_profile_wave2.py           # full 46240-step (20 s × 2)
python3 scratch/perf_profile_wave2.py --quick   # smoke 4624-step (2 s × 2)
cat scratch/perf_profile_wave2_results.json
```

*Frozen invariant: no scientific config/hash/design modified. Results JSON is receipt-only.*
