# H07 term 4 ruling Q18 (in effect on commit; 2026-11-05)

Owner ruling (Hamm, Q18, 2026-11-05; OPEN-29): H07 term 4 passes iff locality_index >= 0.9 x the canonical model's locality_index. This matches the plus or minus 10 percent tolerance ruled for edge counts (Q-a). The sealed spec `SCI-AGSDR-H07H11_spec.md` and its receipts are not edited.

Reason: under the Q-c definition, the canonical reference scores 1.92 (3D) and 1.52 (horizontal distances only), both against a threshold of > 3. A gate that fails its own reference cannot be interpreted.

Recompute: `scripts/agsdr_h07_q18.py` reads `docs/receipts/agsdr_h07h11/AGS_S0_*.json` and the canonical reference, applies the ruled term 4, and writes `docs/receipts/agsdr_h07h11/h07_q18_summary.json`. It does not simulate. Controls: the canonical model passes; the globalized model (control 2 of the sealed spec, locality_index 1.00) fails. The final hard-feasible set is reported for the unamended stage 0 (`stage0_correction_v1.json`) and for stage 0 under amendment 4 (`docs/receipts/agsdr_stage0_a4/summary.json`).
