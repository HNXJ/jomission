# VIP equation by construction path, 2026-11-14

Status: observed by construction probe. No simulation was run. Ruling: owner, 2026-11-14 (VIP hybrid not adopted; JaxFNE 0.5.8 authoritative; route kept open). Tracked as OPEN-51.

## Question

Which VIP v-equation does each construction path produce, under JaxFNE 0.5.8 and 0.5.9? Are coefficients consistent across `jomission/network/builder.py`, `jomission/qualification/cmin.py`, and the constructed model?

## Method

- Script and outputs: `docs/receipts/vip_equation/` (`vip_probe.py`, `vip_058.json`, `vip_059.json`, logs).
- Paths: `build_jomission_model(n_per_area=100, seed=0)` (builder); `cmin.build_cmin(seed=0)` raw; `cmin.repair_jitter(cmin.repair_tonic(cmin.build_cmin(seed=0)), seed=0)` (repaired).
- Read from each constructed model: per-type `v_lin`, `v_const`; VIP `a`, `b`, `c`, `d`.
- Engines: 0.5.8 = tag `v0.5.8` (`11957afa`; checked as git HEAD of `D:\repos\_work\jaxfne_v058`). 0.5.9 = tag `v0.5.9` (`e0160dfc`); the side tree `D:\tmp\jaxfne_v059` is byte-identical to `git archive v0.5.9 jaxfne` (ignoring `__pycache__`).
- Interpreter: `D:\venvs\jomission\Scripts\python.exe` with `PYTHONPATH` set to the engine tree and the jomission checkout.

## Result

| Path | Engine | VIP v-equation (`v_lin`, `v_const`) | VIP b | VIP a, c, d | E, PV, SST |
|---|---|---|---|---|---|
| builder, 64 VIP cells | 0.5.8 | none (engine standard 0.04v²+5v+140) | 0.1808 to 0.2198 | a 0.0181 to 0.0218; c −57.97 to −52.02; d 5.41 to 6.59 | standard |
| builder, 64 VIP cells | 0.5.9 | **4.1 / 108 (class-1)** | 0.1808 to 0.2198 | same | 5 / 140 |
| cmin raw, 28 VIP cells | 0.5.8 | none (standard) | −0.10 | a 0.02; c −55; d 6 | standard |
| cmin raw, 28 VIP cells | 0.5.9 | **4.1 / 108 (class-1)** | −0.10 | same | 5 / 140 |
| cmin repaired (repair_tonic, repair_jitter), 28 VIP cells | 0.5.8 | none (standard) | −0.1164 to −0.0860 | a 0.02; c −55; d 6 | standard |
| cmin repaired, 28 VIP cells | 0.5.9 | `v_lin = None` for all types: standard | −0.1164 to −0.0860 | a 0.02; c −55; d 6 | standard (`v_lin = None`) |

Full per-type pairs: `vip_059.json`, field `per_type`.

## What this shows (observed unless marked)

1. Under 0.5.8 every path gives the standard VIP equation. The field does not exist in 0.5.8 and the standard is hard-coded. Every sealed 0.5.8 result therefore uses the standard VIP equation (derived).
2. Under 0.5.9 the paths disagree. Builder and raw cmin give class-1 VIP (4.1/108). Repaired cmin gives standard VIP. One 0.5.9 run that mixes these paths integrates two VIP equations.
3. The mechanism is the 0.5.9 preset (`jaxfne/emitters.py:124-134`: VIP and I carry `v_lin 4.1`, `v_const 108`; other presets fall back to 5/140). The builder's VIP b override and jitter change fields through `dataclasses.replace`, which keeps `v_lin`/`v_const`. `repair_jitter` calls `with_emitter_parameters`, which calls `refresh_v_coefs`; that sets `v_lin` to `None` (standard) for every type. The repaired E, PV and SST equations are unchanged (5/140), only their field representation changes (derived from source; probe confirms the output).
4. jaxfne's own contract says a b override restores the standard equation (`emitters.py:454-457`; test `tests/test_izh_class1_presets.py:111-114`). The builder route departs from that contract (derived).
5. Scope of the probe: construction only, seed 0, one population size. Dynamic equivalence (trajectories under the two equations) was not tested.

## Provisional definition (for owner confirmation; not a biophysics choice)

Until OPEN-51 is ruled: the intended VIP equation for jomission is the **standard** equation, `v_lin = 5`, `v_const = 140` (0.5.8 behaviour, matching every sealed result). The class-1 preset is not adopted. Any engine migration must either carry the standard coefficients explicitly on every path, or receive a written owner ruling that adopts class-1 for VIP and states its VIP b, a, c, d.

## Required next checks (not run)

- Consistency test: assert that builder, cmin raw, cmin repaired and the constructed model all yield the same `(v_lin, v_const)` per type. Under the 0.5.8 pin it should pass; under 0.5.9 it should fail on the builder and raw-cmin routes. Add to tier 0 only when the 0.5.8 pin is the one under test.
- Repeat the probe for seeds 1 to 4 and for the other cmin repair paths (`d_per_neuron` and `b_per_neuron` in `qualification/cmin.py`) before any migration decision.
- Dynamic check (not in this record): short-run VIP trace under each equation, if OPEN-51 needs an effect size.

## Not changed

No engine code, pin, sealed result, manifest or gate was changed. `E0_SEAL.md` still describes the 0.5.9 call (see `docs/laminar_motif/E0_PROVENANCE_ADDENDUM_1.md`).
