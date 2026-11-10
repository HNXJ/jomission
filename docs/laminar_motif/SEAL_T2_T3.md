# Seal: T2 E/I and T3 PV/SST ratio (lineages `t2`, `t3`), 2026-11-10

Under PLAN_PAPER_JOMISSION_LANSA.md v2 §2. Committed and pushed before the first T2 or T3 run.

- **Reference:** the frozen T1 winner (SEAL_T1.md) at the T1 size N. If T1 has not finished when a slot is
  free, T2 and T3 run on the L-FS reference `e2_3unit_n{N}_dens_k1_kd1_sst15` instead, labelled
  `ref=lfs`; the T1-winner runs follow when T1 freezes. Both are reported, never merged.
- **T2, E/I:** every GABA weight (pv_e, pv_pv, sst_e, sst_pv) times g; modifier `giX`. Excitatory weights,
  in-degrees and drive unchanged. N_I/N_E is deferred (plan §5).
- **T3, PV/SST ratio:** every unit's N_SST/N_PV times s, keeping the ramp shape (low superficial, high deep)
  and each unit's inhibitory total; modifier `rhosX`. LANSA CB/PV is not used as SST/PV (OPEN-38).
- **Levels (log, factor 2; owner order "logarithmic sweep"):** g and s in {0.25, 0.5, 1, 2, 4}. Level 1 is the
  reference. The plan's {0.5, 2} are inside this grid. Seeds 1-3, 8 s, 2 s discard.
- **Estimands per level (seed mean and SD):** r_proto, r_ab, r_gm, the scorer verdict, per-unit LFP peak Hz,
  E/PV/SOM rate per unit, absolute alpha-beta and gamma band power per unit (RWS, pA²/Hz).
- **No pass criterion:** T2 and T3 are dose-response measurements. A level is reported whatever its
  verdict. The reported effect is the slope of each estimand on log2(level), with its SE over seeds.
- **Stop states:** a W1 refusal (level dropped, reported); any unit E rate above 50 Hz or deep E rate below
  0.1 Hz (the level is reported as a stop state, not dropped).
- **Not retuned:** levels and estimands are fixed here.
- **Visualization:** `results/viz/t2/`, `results/viz/t3/`. V0: the W3 block of the reference. V1: 1 s raster
  of level 1, seed 1, before reading any other level. V2: final raster per level (seed 1). V3: SL map atlas
  per level, and dose-response panels.
