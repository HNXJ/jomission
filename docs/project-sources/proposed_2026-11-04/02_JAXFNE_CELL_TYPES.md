# JaxFNE dev: cell types, engine semantics, migration boundary
Sources: dev `jaxfne/emitters.py`, `tests/test_izh_class1_presets.py`, `artifacts/archive/0.5.x/issues_closed/issues_closed_2026-11-03.md` P-054; Jomission `docs/JAXFNE_058_AUDIT.md`, `docs/OPEN.md` OPEN-15.

## P-054: source-equation mismatch corrected in dev
0.5.8 VIP and generic I use Izhikevich `a=.02,b=-.1,c=-55,d=6`, drive=3, with standard `dv/dt=.04v²+5v+140-u+I`; formal rheobase 22.56. Dev uses class-1 source `dv/dt=.04v²+4.1v+108-u+I`; formal rheobase 2.25. `du/dt=a(bv-u)`, reset `v=c,u+=d`. Dev tests assert VIP/I class-1 coefficients, spiking at preset drive, standard E unaffected, mixed-cell realization. This changes the scientific plant, not merely a label.

**Critical Jomission caveat:** builder overrides VIP b to **0.20** (`VIP_B_CORRECTED`, OPEN-15). Published 22.56/2.25 numbers are for b=-0.10, NOT necessarily Jomission's realized VIP. For b=.20, the formal source-equation quadratic threshold is -12.9375; this is not a verified dynamic firing threshold. Explicitly inspect constructed b, v_lin/v_const, initial v/u, reset, effective current, noise and firing before interpreting.

## 0.5.8 API refusals
Jomission migration audit records refusal of stored/unconsumed `delay_ms_or_status`, identity `cell_type_to_cell_type_map`, `connectivity(within_area='spatial_local')`, `spatial_sigma`, `field(source_mode=...)`. Jomission uses an edge-delay shim and locality pruning shim; engine native sparse connectivity requires mechanism-aware rules. `Simulation.seed` and `runtime.seed` must agree. Verify actual edge list, weights, delays, source/field output, tonic + scheduled drive, noise, RNG. Configured ≠ realized ≠ executed ≠ causally effective.

## Proposed migration verification, not authorization
Owner decides whether/when to migrate after release. Pin wheel/source hashes. Run deterministic single-cell E/PV/SST/VIP/I steps on 0.5.8 and candidate, both default and Jomission-overridden VIP, varying initial v/u. Check construction identity, labels, geometry, edges, weights, delays, config hashes, and expected scientific equation differences. Run tier 0–2 and frozen local-operation gate, V0–V3. Preserve FAIL if it fails. Do not expect bitwise spike parity when equations change.
