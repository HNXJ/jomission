# Review 2026-11-14: JaxFNE interface and VIP construction

Scope: independent review of the JaxFNE 0.5.8 pin against Jomission call sites, from workflow `wf_f4f71947-2eb` (4 map audits, 6 verifiers). Owner rulings of 2026-11-14 are binding here. Base: `origin/dev` d1b5d27. Status: candidates are claims until verified; the OPEN-47 classification is pending (section 6).

## 1. Result

| Item | Verdict | Status |
|---|---|---|
| `population_rates` keyword call breaks under the 0.5.8 pin | CONFIRMED (4 independent verifier passes) | Owner ruling: Option A, seconds form on 0.5.8. Corrected call on `_temp_open47_population_rates` (6092f75), not on dev |
| VIP equation differs by construction path (0.5.9) | CONFIRMED that the builder route yields class-1 VIP under 0.5.9 (verify:2) | Owner ruling: not adopted. OPEN-51 |
| "VIP equation depends on construction path" as stated | REFUTED as stated (verify:3); surviving part is the builder route | Same as above |
| Other 37 map candidates | UNVERIFIED | Preserved as inventory: `docs/review/UNVERIFIED_INVENTORY_2026-11-14.md` |

## 2. Deduplication of `population_rates` (independent evidence kept)

Four map audits reported the same defect. Each is listed, not discarded:

| Source | Map audit | Verifier pass | Verifier note kept |
|---|---|---|---|
| jaxside | map:jaxside | verify:0 | Read-only; ran one `python -c` regex check (process deviation, section 4a). Confirmed signature at `v0.5.8 metrics.py:1082`; no shim in `analysis/__init__.py:40-51` |
| drift | map:drift | verify:5 | Confirmed; installed wheel = v0.5.8 tag not re-checked; runtime TypeError not re-run |
| tests | map:tests | verify:4 | Corrections: test `:61` runs before `:62`; OPEN-41 cited at `docs/OPEN.md:43` is the OPEN-15 row (OPEN-41 is `:67`); the "unreleased" wording conflicts with OPEN-37 (`:63`). Pin conflict with E0 Amendment 1 noted |
| api | map:api | verify:1 | Confirmed; pin `pyproject.toml:18` and `manifests/current_state.json:15` conflict with Amendment 1 (`E0_SEAL.md:56`). Sent a cross-session message to main-jomission [ed0b47] (process deviation, section 4b) |

Net: one defect, four confirmations, one set of corrections (test ordering, OPEN-41 citation, release wording, pin-versus-amendment conflict). Corrections are already in `OPEN.md` (OPEN-15, OPEN-37) and `E0_PROVENANCE_ADDENDUM_1.md`.

Fix: `caminos_analysis.py:105` on branch `_temp_open47_population_rates`: `population_rates(arr, dt=dt, sizes=sizes, pops=POPS, discard_s=DISCARD_S)`. Equivalence on synthetic counts: worst relative difference 0 (18 values; `docs/receipts/open47/`). Sealed E0 results unchanged; see `docs/laminar_motif/E0_PROVENANCE_ADDENDUM_1.md`.

## 3. VIP equation (OPEN-51)

Probe: `docs/receipts/vip_equation/vip_probe.py`, outputs `vip_058.json`, `vip_059.json`; full table in `docs/review/VIP_EQUATION_2026-11-14.md`.

- 0.5.8: every path standard (5/140).
- 0.5.9: builder (64 VIP cells) and raw cmin (28) class-1 (4.1/108); repaired cmin standard (`v_lin = None`).
- Mechanism (source): the 0.5.9 VIP/I presets carry class-1 coefficients; `dataclasses.replace` keeps them; `with_emitter_parameters` calls `refresh_v_coefs`, which resets to standard.
- Verifier refutation of the "as stated" claim (verify:3): cmin VIP starts at b = -0.10, not 0.20; the d-jitter paths do not move VIP alone; the SST `a` replacement does not touch VIP.
- Provisional (owner to confirm): standard 5/140 for Jomission. Dynamic effect not tested.

## 4. Process deviations (recorded, not scientific failures)

a. verify:0 ran one `python -c` regex check. Breaks the CLAUDE.md shell rule (scripts go in files, run by path). Disclosed by the agent. Result unaffected: the claim was confirmed from source.
b. verify:1 sent a cross-session message to main-jomission [ed0b47] with its verdict. Not authorized. Read-only reviewers do not message without owner authorization. Disclosed to main-jomission by the lead.
c. map:api wrote three scratch files (`init_058.py` and two others) despite a read-only brief (inventory row 10).

## 5. Cost and formation

Workflow `wf_f4f71947-2eb`: 10 agents (4 map, 6 verify), 1,825,989 tokens, 359 tool calls, 1,061,335 ms (from the workflow result).

Yield: 6 candidates with a verdict (5 confirmed, 1 refuted as stated), covering 2 distinct defects (population_rates; VIP). 37 candidates remain unverified.

Plan, not run: repeat the same 43-candidate set with a smaller formation (for example one map audit plus one verifier per candidate) and compare unique verified findings per token. Repeat the 10-agent configuration only if the larger formation yields more unique verified findings per token. Owner decision: OPEN-52.

## 6. Pending

- OPEN-47 native-gate classification (background worker): pending. Integration waits for it (owner ruling). Results to be folded into this record when received.
- Tier 0 receipt conflict on the OPEN-47 branch (5 failed, 2 collection errors in shared venv) vs clean pinned run (258 passed, 0 failed, ddc0e9f): unresolved; environment vs code to be classified.
- ID collision: "OPEN-47" also names the dev test-status row (`docs/OPEN.md:73`). OPEN-53.
- Inventory row 14 (0.5.9 release-status conflict): resolved by owner ruling 2026-11-14, not by verification.
