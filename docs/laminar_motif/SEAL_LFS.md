# Seal: L-FS finite-size ladder (lineage `lfs`), 2026-11-10

Parent seal under PLAN_PAPER_JOMISSION_LANSA.md v2 §2 (owner order 2026-11-10, todo SCI-LAMINAR-MOTIF-1). This
file is committed and pushed before the first ladder run.

- **Question:** the smallest E-per-unit size at which the reference model kd1_sst15
  (`e2_3unit_nN_dens_k1_kd1_sst15`) keeps its spectrolaminar map. In-degrees K_E and K_INTER are preserved;
  the W1 guard refuses any clipped p.
- **Grid:** N ∈ {4000, 2000, 1000} E per unit (mean, density-scaled). Seeds: 1, 2, 3 at every N; n4000 seed 1
  exists. Run length 8 s, 2 s discarded. Fixed log-ratio grid; no other parameter changes.
- **Software:** JaxFNE 0.5.8, the execution authority. W0 (2026-11-10) found 0.5.8 and 0.5.9 give identical
  per-step block spike counts on this stack (0.5 s, 35028 spikes, max relative difference 0).
  - If an 8 s run fails on 0.5.8 for memory, the ladder falls back to the 0.5.9 install
    `~/.local/lib/jaxfne-sites/v0.5.9-e0160dfc`, labelled off-authority.
  - The existing n4000 seed 1 run (0.5.9 scratch install) is a 0.5.9 record. It counts in the ladder because the
    W0 equality holds.
- **Estimands per N:**
  - mean and SD over seeds of r_proto (sum row);
  - per-unit LFP peak Hz;
  - the full `scripts/e2_scorer.py` verdict fields;
  - E rate per unit.
- **Rule (from the plan; not retuned):** accept the smallest N whose seed-mean r_proto is within 2 seed-SD of
  n4000's and whose per-unit peaks are within 3 Hz of n4000's. That N becomes the T1 size.
  - If no smaller N holds, T1 runs at n4000.
  - n4000 needs 3 seeds before the rule is applied.
- **Stop states:**
  - a run refused by the W1 guard (report it; do not change N);
  - any unit with an E rate above 50 Hz, or a deep unit with an E rate below 0.1 Hz (report it; the rule
    still applies to the remaining N).
- **Visualization (contract):** `results/viz/lfs/`.
  - V0: architecture schematic and details table (W3 kit) for n1000.
  - V1: 1 s raster of n1000 seed 1, before any reading of the ladder.
  - V2: final rasters per N.
  - V3: an atlas of SL maps per N and seed.
- **Not decided here:** the literal 1000-total-cell model (`L-FS1k`, weight scaling). It gets its own seal.
