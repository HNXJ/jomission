# SCI-AGSDR-CALIBRATION amendment 4 (in effect on commit; 2026-11-05)

Owner rulings (Hamm, 2026-11-05): Q16 accepts SCI-VIP-DRIVE-PROBE as the viable-regime verdict (OPEN-25); Q17 sets the form: a frozen VIP drive scale of 2.25. Nothing here changes a sealed file in place.

## Problem
Under the sealed design, VIP is 0.4 to 1.7 Hz in 24 of 24 stage-0 candidates, against the B1 band of 8 to 15 Hz; no design coordinate acts on VIP drive. The builder boosts PV drive (x1.7, GEN2_C016) and not VIP; VIP drive after tonic 0.6 is about 1.78 against a VIP rheobase near 4. SCI-VIP-DRIVE-PROBE (spec 2fcac30, docs/receipts/vip_drive_probe/summary.json) found that VIP drive x2.0 and x2.5 put AGS_S0_06 and AGS_S0_20 in all bands at 2 seeds. SCI-PV-HYSTERESIS (79dfe68) found no hysteresis on the VIP-drive or global-drive axes.

## Amendment
1. Frozen (not free): VIP drive scale 2.25, the midpoint of the viable interval [2.0, 2.5]. Each VIP cell's emitter drive is multiplied by 2.25 after the builder has finished, then clipped to the builder bounds [1, 8]. Same per-class drive mechanism as the PV boost, applied as `vip_drive_probe.scale_vip_drive(model, 2.25)`.
2. Seam: applied in the AGSDR runners after `build_candidate_model`. `jomission/network/builder.py` and the canonical model are unchanged, so tests, the edge identity and config hashes of the canonical build are unaffected. Receipts record the scale.
3. Stage 0 is rerun under this amendment (24 candidates, seed 0, the stage-0 simulation and scoring, the tau-corrected H09 and the H05 all-E ruling), under its own sealed spec (`SCI-AGSDR-STAGE0-A4_spec.md`). The stage-0 receipts in `docs/receipts/agsdr_local/` stay as the record of the unamended design.
4. Structural gates: the amendment changes no edge or metadata field. H07 and H11 results carry over (`docs/receipts/agsdr_h07h11`). H07 term 4 remains for owner ruling (OPEN-29).

## Does not
Change any threshold, weight, sampling rule, candidate coordinate or the 11-dimensional search box. It does not select candidates or authorise stage 1. It does not resolve OPEN-18, OPEN-23, OPEN-29 or OPEN-33.
