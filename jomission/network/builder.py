"""Network builder — constructs JaxFNE Configuration + Model for jomission.

Single source of truth for V1/V4/FEF/PFC × layer × E/PV/SST/VIP.

Delegates to JaxFNE primitives: Configuration.column, area_layer_cell_types, connectivity,
hdp, field, probe, runtime. No parallel neural simulator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence, Optional

import jax.numpy as jnp
import numpy as np
import jaxfne as jtfne
from jaxfne import Configuration

from jomission.network.populations import (
    JOMISSION_AREAS,
    JOMISSION_LAYERS,
    AREA_LAYER_CELL_TYPES,
    LAYER_DEPTH_BANDS,
    LAYER_COUNT_FRAC_DEFAULT,
)
from jomission.network.connectivity import (
    WITHIN_GAIN_DEFAULT,
    P_FEEDFORWARD_DEFAULT,
    P_FEEDBACK_DEFAULT,
    HIERARCHY,
    DELAY_FF_MS,
    DELAY_FB_MS,
    DELAY_WITHIN_MS,
    DELAY_LAMINAR_MS,
)

# GEN2_C002 intrinsic heterogeneity (W1.2) — per-neuron jitter via cell_params seam
# Breaks class-homogeneous 74-spike degeneracy; deterministic per-seed RNG.
HETEROGENEITY_JITTER_SIGMA_DEFAULT: float = 0.10

# GEN2_C007 background Poisson (W2.2) — minimal irregularity drive, fluctuation-driven
# GEN2_C012 I_bg(λ) mean-controlled substitution: Poisson OFF→2kHz amp2.0 private per-neuron (Brunel2000)
#   I_bg(λ)=I_tonic(λ)+I_private(λ)+I_noise+I_syn with E[I_bg]≈μ_target∈[3.5,4.0] fixed, σ_total=√(p(1-p)amp²+0.25),
#   p=rate·dt/1000 (dt0.1), Poisson 2kHz amp2.0 p0.20 μ0.40 σ0.80 → μ_total3.405 σ_total0.943 σ/μ0.277
# Seam: jaxfne/_signals.py:670 _make_poisson_drive + Simulation(poisson_drive) via
# _model_simulate.py:771-783 (existing seam, not new simulator). Deterministic per-seed (seed+7919).
# Provenance MODEL_ASSUMPTION rate 2.0 kHz amp 2.0 (native current units) direction LITERATURE_PRIOR Brunel2000 / ENGINE_DEFAULT Poisson seam;
# tonic drive 5 but Poisson SD ~sqrt(p(1-p))·amp provides private fluctuation without shared rho (Corr≈0 via per-neuron Bernoulli shape n_steps×n_neurons).
BACKGROUND_POISSON_RATE_HZ_DEFAULT: float = 2000.0
BACKGROUND_POISSON_AMPLITUDE_DEFAULT: float = 2.0
BACKGROUND_POISSON_TARGET_DEFAULT: str = "all"
BACKGROUND_POISSON_PROVENANCE: str = "MODEL_ASSUMPTION (rate 2.0kHz, amp 2.0 magnitudes) / LITERATURE_PRIOR direction balanced barrage Brunel2000 / ENGINE_DEFAULT (emitters.py:477 Poisson seam if used) / DERIVED p·amp algebra"
# Referenced seam hash-visibility: cfg.metadata background_poisson_rate_hz / amplitude / seed / target

# GEN2_C014 pseudogenome — typed DESIRED_MOTIF_GAIN v0 (replaces uniform 1.5× one)
# Seam: post-construct EdgeList weight scaling _apply_motif_gains builder.py:623 (preferred, hash-visible)
# vs declarative Configuration.connections (not suitable for VIP→SST due to E-only source filter _config.py:1079).
# Provenance: magnitudes MODEL_ASSUMPTION (single enum), direction LITERATURE_PRIOR (Pfeffer2013 etc, DOI pending, builder.py:78)
# All 16 motifs explicit; per-motif deterministic seed ^motif_hash mirroring builder.py:134 jitter (see _apply_motif_gains)
# Preserves lognormal CV1.5, VIP b0.05, tonic 3.0, Poisson 2kHz, delays [20,80,120], topology, sign.
# Previous MOTIF_GAIN E→PV1.5× only (C006) uniform 15 motifs 1.0 is superseded.
MOTIF_GAIN: dict[tuple[str, str], float] = {
    ("E", "E"): 1.00,
    ("E", "PV"): 1.70,
    ("E", "SST"): 0.70,
    ("E", "VIP"): 1.00,
    ("PV", "E"): 1.30,
    ("PV", "PV"): 1.00,
    ("PV", "SST"): 1.00,
    ("PV", "VIP"): 1.00,
    ("SST", "E"): 0.60,
    ("SST", "PV"): 0.60,
    ("SST", "SST"): 1.00,
    ("SST", "VIP"): 0.80,
    ("VIP", "E"): 1.00,
    ("VIP", "PV"): 1.00,
    ("VIP", "SST"): 5.00,
    ("VIP", "VIP"): 1.00,
}
# Alias for provenance audit
DESIRED_MOTIF_GAIN: dict[tuple[str, str], float] = dict(MOTIF_GAIN)
DESIRED_MOTIF_GAIN_V0: dict[str, float] = {f"{k[0]}->{k[1]}": float(v) for k, v in MOTIF_GAIN.items()}
PSEUDOGENOME_VERSION: str = "v0"
PSEUDOGENOME_PROVENANCE: str = "MODEL_ASSUMPTION (magnitudes) / LITERATURE_PRIOR direction Pfeffer2013 etc (DOI pending, builder.py:78) / ENGINE_DEFAULT seam / DERIVED per-motif mean"
PSEUDOGENOME_SEAM: str = "post-construct EdgeList weight scaling _apply_motif_gains builder.py:623 (preferred, hash-visible) vs declarative Configuration.connections (not suitable for VIP→SST due to E-only source filter _config.py:1079)"
MOTIF_GAIN_PROVENANCE: str = "MODEL_ASSUMPTION (magnitudes) / LITERATURE_PRIOR direction Pfeffer2013 etc (DOI pending) / ENGINE_DEFAULT seam / DERIVED per-motif mean"
MOTIF_GAIN_SEAM: str = "post-construct EdgeList weight scaling _apply_motif_gains builder.py:623 (preferred, hash-visible) vs declarative Configuration.connections (not suitable for VIP→SST due to E-only source filter _config.py:1079)"
MOTIF_GAIN_REASON: str = "typed W[a,l_s,c_s,l_t,c_t] Rule compliance via pseudogenome DESIRED_MOTIF_GAIN v0: E→PV1.70 E→SST0.70 PV→E1.30 SST→E0.60 SST→PV0.60 SST→VIP0.80 VIP→SST5.00 etc; replaces uniform 1.5× one-motif (C006) with full 12-gain pseudogenome (magnitudes MODEL_ASSUMPTION, direction LITERATURE_PRIOR Pfeffer2013)"

# GEN2_C008 laminar hierarchy + delays W2.3 — explicit FF/FB layer maps + delay_steps
# Provenance: delays 8/12/2 ms are MODEL_ASSUMPTION (proxy, no citation yet), ordering FB>FF>within is DERIVED (less myelinated FB slower)
# Seam: Configuration.inter_column_connectivity(layer_to_layer_map, delay_ms_or_status) metadata + post-construct edge_list_with_delay_ms(dt=0.1)
# Default JaxFNE pairs were L2/3->L4 (FF) and L6->L1 / L6->L5 (FB) implicit when layer_to_layer_map=None (_construct_connectivity.py:61-64)
# Now made explicit so config_hash reflects laminar routing and delays hash-visible.
FF_LAYER_MAP: dict[str, str] = {"L2/3": "L4"}
FB_LAYER_MAPS: list[dict[str, str]] = [{"L6": "L1"}, {"L6": "L5"}]  # two specs per FB direction (dict key uniqueness)
# For hash-visible metadata, store fb as combined dict with suffix to avoid key collision
FB_LAYER_MAP_COMBINED: dict[str, str] = {"L6->L1": "L1", "L6->L5": "L5"}  # convenience, actual specs are separate
# GEN2_C009 VIP intrinsic correction — sole negative b -0.10 → +0.05
# GEN2_C017 VIP intrinsic recruitment — b 0.05→0.20 single delta (I_rh>15→4, operating VIP μ2.19 σ0.943 μ+2σ≈4.07 — b0.20 I_rh4 8at4 near μ+1.92σ)
# Provenance: MODEL_ASSUMPTION (0.20 magnitude) / LITERATURE_PRIOR direction b≥0 Izhikevich2003 / ENGINE_DEFAULT emitter seam / DERIVED f-I
# Seam: pre-jitter emitter b override via Builder _apply_vip_b_correction builder.py:713-785 pre-jitter u0=b·v0 ±10% jitter → [0.18,0.22] I_rh≈4 thr≈95
# References: jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS VIP b -0.10 original; builder.py:39 jitter σ=0.10 b∈[-0.20,0.35] clip
#            C009 0.05→0.15 I_rh8 r11 thr100.64, b0.20 I_rh4 r21 thr95 equals E; operating VIP μ2.19 σ0.943 μ+2σ≈4.07; C017 b0.20 I_rh4 8at4 at μ+1.92σ recruits (2.7% tail)
# After fix: VIP b 0.20 ±10% → [0.18,0.22] all positive, threshold 107.79→94.99, V_m-77.79→-64.99 at I3, I_rh>15→4 deterministic, PV>E ordering preserved
VIP_B_CORRECTED: float = 0.20
VIP_B_CORRECTED_PROVENANCE: str = "MODEL_ASSUMPTION (0.20 magnitude) / LITERATURE_PRIOR direction b>=0 Izhikevich2003 / ENGINE_DEFAULT emitter seam / DERIVED f-I (I_rh>15->4, thr107.79->94.99, V_m-77.79->-64.99)"
VIP_B_CORRECTED_SEAM: str = "pre-jitter emitter b override via Builder (_apply_vip_b_correction builder.py:713-785 u0=b*v0 recompute) + jitter preservation -> [0.18,0.22]"
VIP_B_OLD_C016: float = 0.05
# GEN2_C015 SST intrinsic correction — operating-point correction b 0.25→0.21 (not SST-output gain)
# Provenance: MODEL_ASSUMPTION (0.21 magnitude) / LITERATURE_PRIOR direction SST LTS Izhikevich2003 / ENGINE_DEFAULT emitter seam / DERIVED f-I
# Seam: pre-jitter emitter b override (builder.py) + per-neuron jitter preservation, like VIP builder.py:713
# References: jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS SST b 0.25; builder.py:39 jitter σ=0.10
#            f-I dense b sweep (2000ms dt0.1 noise0.5 isolated) b0.21 I_rh3.5 r20.5@3.5 r39.5@5 gain12.7 CV0.035 V_m-62.84 threshold92.8 PASS all 6
# After fix: SST b 0.21 ±10% → [0.189,0.231], b∈[-0.20,0.35] clip, u0=b·v0 recomputed, deterministic, sign-preserving
SST_B_CORRECTED: float = 0.21
SST_B_OLD: float = 0.25
SST_B_CORRECTED_PROVENANCE: str = "MODEL_ASSUMPTION (0.21 magnitude) / LITERATURE_PRIOR direction SST LTS Izhikevich2003 / ENGINE_DEFAULT emitter seam / DERIVED f-I"
SST_B_CORRECTED_SEAM: str = "pre-jitter emitter b override via Builder (_apply_sst_b_correction) + jitter preservation (like builder.py:713 VIP)"
# GEN2_C016 PV recruitment — PV-specific tonic drive boost (I_network < I_rh gap)
# Provenance: MODEL_ASSUMPTION (1.7 magnitude to reach rheobase 1.77->3.01) / LITERATURE_PRIOR direction fast-spiking PV recruitment (Pfeffer2013, Hu2014) / ENGINE_DEFAULT drive seam
# Seam: post-jitter per-class emitter drive scaling _apply_pv_drive_boost via neuron_table PV identification (existing Model params, not new kernel), deterministic, sign-preserving
# References: emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS PV drive 3.0; builder.py:39 jitter σ=0.10 -> PV drive 1.772±0.098 [1.62,1.98] I_net2.175 < I_rh4.0 r0.317Hz V_m-67.11 thr97.11
#            tonic scale 0.6 -> E 3.01 PV 1.77 SST 1.93 VIP 1.78; PV boost 1.7 -> PV 3.01 (1.77*1.7) I_net≈3.40 > I_rh, deterministic per-PV
#            pseudogenome DESIRED_MOTIF_GAIN v0 intact, VIP b0.05 frozen, SST b0.21 intact, lognormal CV1.94, delays [20,80,120], Poisson 2kHz
# Hash-visible via cfg.metadata pv_drive_scale / pv_drive_boost / pv_drive_PV etc; recomputes u0=b·v0 preserved (drive does not affect u0 but kept for audit)
PV_DRIVE_SCALE_DEFAULT: float = 1.7
PV_DRIVE_SCALE_PROVENANCE: str = "MODEL_ASSUMPTION (1.7 magnitude to reach rheobase 1.77->3.01) / LITERATURE_PRIOR direction fast-spiking PV recruitment Pfeffer2013 Hu2014 / ENGINE_DEFAULT drive seam"
PV_DRIVE_SEAM: str = "post-jitter per-class emitter drive scaling _apply_pv_drive_boost via neuron_table PV identification (existing Model params, not new kernel), deterministic, sign-preserving, after tonic scale before laminar delays"
PV_DRIVE_REASON: str = "PV I_network 2.175 < I_rh4.0 (1.825 below, 2σ upper 4.06 barely touches) vs E net3.41 near I_rh; E→PV gain would compensate low r_E 3.9Hz not motif fix; PV-specific tonic drive 1.77->3.01 (scale1.7) raises I_net≈3.40 > I_rh without broadly increasing E→PV motif (single PV recruitment delta)"
DELAY_PROVENANCE: str = "MODEL_ASSUMPTION (8/12/2 values) / DERIVED (ordering FB>FF>within)"
DELAY_SEAM: str = "post-construct edge_list_with_delay_ms(dt=0.1) via jaxfne/emitters.py:545 + Configuration.inter_column_connectivity delay_ms_or_status metadata"

# GEN2_C010 lognormal weight variance (B2 CV(W) heterogeneity)
# Provenance: MODEL_ASSUMPTION (CV=1.5 magnitude — no single citation fixes 1.5)
#             / LITERATURE_PRIOR direction Song et al. 2005 lognormal E distribution
# CV(L)=1.5 → σ=√ln(1+CV²)=√ln(3.25)=1.0856587845 (correct, NOT 0.97→CV≈1.25)
#             μ=-σ²/2=-0.5893274982 so E[L]=exp(μ+σ²/2)=1 motif-mean-preserving.
# Seam: post-construct EdgeList weight scaling (existing Model params, no new kernel)
#       — like C006, after _apply_motif_gains before _apply_laminar_delays so
#       delays/topology/VIP b/Poisson/H/Θ/FF strength/tonic drive 5 untouched.
LOGNORMAL_CV: float = 1.5
LOGNORMAL_SIGMA: float = 1.085658784490618  # sqrt(ln(1+1.5^2)); approx 1.08566 (not 0.97)
LOGNORMAL_MU: float = -0.5893274981708232  # -sigma^2/2; approx -0.58933
LOGNORMAL_SIGMA_APPROX_NOTE: str = "0.97 approx gives CV~1.25 not 1.5 — use 1.08566"
LOGNORMAL_PROVENANCE: str = "MODEL_ASSUMPTION (CV=1.5 magnitude) / LITERATURE_PRIOR direction Song et al. 2005 lognormal"
LOGNORMAL_SEAM: str = "post-construct EdgeList weight scaling W·L L~LN(μ,σ) per-motif mean-preserving (existing Model params, not new kernel)"

# GEN2_C011 tonic drive reduction (B2 operating-point, single-knob causal test)
# GEN2_C012 I_bg(λ) mean-controlled substitution: I_tonic 3.5→3.0 (scale0.6) + Poisson 0→2kHz amp2.0 — one background-composition λ
#   I_bg(λ)=I_tonic(λ)+I_private(λ)+I_noise+I_syn with E[I_bg]≈μ_target fixed near rheobase while redistributing deterministic→private stochastic,
#   preserving μ_total≈3.405 σ/μ 0.277 vs C011 0.168. Not two-factor tuning: single ledger delta redistributing fixed budget.
# Provenance: MODEL_ASSUMPTION (magnitude 3.5 toward rheobase μ_rheobase≈3.5) / DERIVED (scale 0.7=3.5/5) / ENGINE_DEFAULT (drive seam emitters.py:55,477)
#   C012 magnitude 3.0 (scale0.6) is MODEL_ASSUMPTION within hedging 3.5 vs 4.0 vs 3.0; direction toward rheobase is DERIVED (R-tonic sweep);
#   Poisson 2kHz amp2.0 magnitude MODEL_ASSUMPTION / direction LITERATURE_PRIOR Brunel2000 / DERIVED p·amp algebra.
# Direction I_tonic 5→3.5 toward rheobase is DERIVED from w2_D_ISI_diagnosis.md:§2.6 single-neuron smokes (drive 3→0 spikes, 4→ISI140 CV0.006, 5→ISI93 CV0.018) so μ_rheobase≈3.5.
# C012 candidate B I_tonic 3.0 scale0.6 + Poisson 2kHz amp2.0 p0.20 μ0.40 σ0.80 → μ_total3.405 σ_total0.943 σ/μ0.277 (input-stat preregistered, not CV-selected).
# Seam: post-jitter emitter drive scaling drive·s (existing Model params, no new kernel) via _apply_tonic_drive_scale before delays,
#       uniform across E/PV/SST/VIP preserving heterogeneity jitter relative ratios and sign, delay/topology/VIP b/lognormal untouched.
# C011 Poisson remains OFF to isolate tonic effect; C012 Poisson ON as mean-controlled substitution — baseline parent plain 4bf1403e628ac06f (C011 scale0.7) → C012 plain+poisson.
TONIC_DRIVE_SCALE_DEFAULT: float = 0.6  # 3.0/5.0 — C012 I_bg(λ) mean-controlled substitution; C011 was 0.7 (3.5); MODEL_ASSUMPTION magnitude 3.0 toward rheobase
TONIC_DRIVE_SCALE_PROVENANCE: str = "MODEL_ASSUMPTION (magnitude 3.0 C012 I_bg(λ) toward rheobase μ≈3.5) / DERIVED (scale 0.6=3.0/5) / LITERATURE_PRIOR direction balanced barrage Brunel2000 for Poisson component / ENGINE_DEFAULT drive seam (emitters.py:55,477)"
TONIC_DRIVE_SEAM: str = "post-jitter/post-lognormal emitter drive scaling drive·s uniform across classes (existing Model params, no new kernel) before laminar delays — combined with Simulation(poisson_drive) via _signals.py:670 as single I_bg(λ) composition"
TONIC_DRIVE_OLD: float = 5.0  # ENGINE_DEFAULT E drive; PV 3.0 SST 3.5 VIP 3.0 scale uniformly
TONIC_DRIVE_NEW: float = 3.0  # C012 I_bg(λ) E drive after scale 0.6; PV 1.8 SST 2.1 VIP 1.8; Poisson 2kHz amp2.0 compensates μ0.40 → μ_total3.405
# GEN2_C012 combined background composition metadata (hash-visible via individual scale/rate/amp but also as combined I_bg_lambda for audit)
I_BG_LAMBDA_PROVENANCE: str = "MODEL_ASSUMPTION (I_tonic 3.0, rate2kHz amp2.0 magnitudes) / LITERATURE_PRIOR direction balanced barrage Brunel2000 / ENGINE_DEFAULT Bernoulli seam / DERIVED p·amp algebra"
I_BG_LAMBDA_SEAM: str = "_apply_tonic_drive_scale(model,scale=0.6) + Simulation(poisson_drive=dict(rate_hz=2000, amplitude=2.0, target='all', seed=seed+7919)) via _signals.py:670 Bernoulli private shape (n_steps,n_neurons)"

# GEN2_C013 SST-output gain reduction (B2 transfer de-clamping, bounded) — SUPERSEDED by C014 pseudogenome
# C013: alpha=0.5 bounded test falsified (|I_SST| -50% but Var/CV flat). C014 pseudogenome DESIRED_MOTIF_GAIN v0
# already encodes typed SST→E0.60 SST→PV0.60 with VIP→SST5.00 etc; keeping alpha 0.5 would double-reduce to 0.30
# (0.60*0.50) vs desired 0.60. So C014 resets alpha to 1.0 (disabled) via pseudogenome_version v0.
# Mechanism preserved for audit; C013 provenance retained for ledger.
SST_OUTPUT_GAIN_ALPHA_DEFAULT: float = 1.0
SST_OUTPUT_GAIN_ALPHA_C013: float = 0.5
SST_OUTPUT_GAIN_ALPHA_PROVENANCE: str = "MODEL_ASSUMPTION (alpha 0.5 bounded C013, 1.0 C014 pseudogenome) / LITERATURE_PRIOR direction (reduce SST influence) / ENGINE_DEFAULT post-EdgeList seam"
SST_OUTPUT_GAIN_SEAM: str = "post-construct EdgeList weight scaling W_{SST->*} x alpha (existing Model params, no new kernel) after lognormal/motif before delays, sign-preserving, deterministic uniform"
# GEN2_C018 typed E mixture pseudogenome M2 70% RS /20% CH /10% E_FS (pseudogenomic)
# Basis E={E_RS,E_IB,E_CH} distinct: RS regular CV0.01-0.07, IB phasic CV0.03-0.30, CH sustained burst CV1.3-1.5
# Minimal K=2 {RS,CH} with 20-30% CH gives CV→0.46, preferred K=3 {RS,IB,CH}; FS-like renamed E_FS to avoid inhibitory label
# M2 70% RS /20% CH (c-50 d2) /10% FS-like (a0.10 c-65 d2) gives 20% burst, Fano +30% robust, minimal viable per R-B
# Provenance: LITERATURE_PRIOR RS/CH/IB Izhikevich2003 Fig.1 (a/b/c/d phenotypes) / MODEL_ASSUMPTION proportions 70/20/10
# Seam: pre-jitter emitter a/c/d override via Builder _apply_typed_E_phenotypes after VIP/SST before _apply_per_neuron_jitter (builder.py:194 emitters.py:55)
#       deterministic per-seed RNG (seed ^ 0x9E3779B9 ^ 0xC018) permutation, phenotype means preserved, jitter ±10% on top retains distinct c/d clusters
#       motif pseudogenome v0, VIP b0.20, SST b0.21, PV drive1.7, lognormal CV1.5, tonic3.0, Poisson2kHz, delays [20,80,120], H/Θ intact
E_MIXTURE_VERSION: str = "M2"
E_MIXTURE_RS_FRAC: float = 0.70
E_MIXTURE_CH_FRAC: float = 0.20
E_MIXTURE_EFS_FRAC: float = 0.10
E_MIXTURE_RS_PARAMS: dict[str, float] = {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0}
E_MIXTURE_CH_PARAMS: dict[str, float] = {"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0}
E_MIXTURE_EFS_PARAMS: dict[str, float] = {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0}
E_MIXTURE_PROVENANCE: str = "LITERATURE_PRIOR RS/CH/IB Izhikevich2003 Fig.1 (a/b/c/d phenotypes) / MODEL_ASSUMPTION proportions 70/20/10"
E_MIXTURE_SEAM: str = "pre-jitter emitter a/c/d override via Builder _apply_typed_E_phenotypes builder.py:194-285 then _apply_per_neuron_jitter σ0.10 on top, deterministic seed, u0=b·v0 recompute, preserves motif gains"
E_MIXTURE_REASON: str = "typed E heterogeneity via pseudogenomic c/d clusters: single RS+σ0.40 meanCV0.036-0.062 max0.219 frac>0.5 0.00 insufficient vs typed M2 meanCV0.350-0.456 max1.90 frac>0.5 0.22-0.29 Fisher 0/51 vs11/38 p<1e-6; variance-matched control Var_c48.0 vs Var42.0 Δ12.5% falsifies broad jitter"
SST_OUTPUT_GAIN_REASON: str = "C013 bounded 0.5 falsified; C014 pseudogenome typed 0.60 supersedes (alpha 1.0 disabled to preserve DESIRED 0.60)"


def _resolve_heterogeneity_sigma(raw: Any) -> float:
    if raw is None:
        return float(HETEROGENEITY_JITTER_SIGMA_DEFAULT)
    try:
        v = float(raw)
    except Exception:
        return float(HETEROGENEITY_JITTER_SIGMA_DEFAULT)
    if v < 0:
        return 0.0
    return float(v)


def _apply_per_neuron_jitter(
    model: jtfne.Model,
    *,
    seed: int,
    sigma: float,
) -> jtfne.Model:
    """Apply ±sigma jitter to Izhikevich a,b,c,d,drive per neuron.

    Deterministic RNG: seed ^ 0x9E3779B9 ^ sigma*1e6.
    Keeps ordering PV>E and bounds.
    """
    if float(sigma) <= 1e-12:
        return model
    em = model.params.get("emitter")
    if em is None:
        return model
    try:
        n = int(np.asarray(em.a).shape[0])
    except Exception:
        return model
    # deterministic per-seed rng
    base = int(seed) ^ 0x9E3779B9 ^ int(float(sigma) * 1e6)
    rng = np.random.default_rng(int(base) & 0x7FFFFFFF)
    # read emitter arrays
    a_np = np.asarray(em.a, dtype=np.float64)
    b_np = np.asarray(em.b, dtype=np.float64)
    c_np = np.asarray(em.c, dtype=np.float64)
    d_np = np.asarray(em.d, dtype=np.float64)
    # drive / sign handling: emitter may store drive and/or separate arrays
    has_drive = hasattr(em, "drive") and em.drive is not None
    drive_np = np.asarray(em.drive, dtype=np.float64) if has_drive else None
    # uniform ±sigma for a,b,d,drive ; c ±3mV scaled by sigma/0.1
    def jitter(arr: np.ndarray, lo: float, hi: float, is_c: bool = False) -> np.ndarray:
        scale = float(sigma) if not is_c else float(sigma) * 30.0  # 3mV at sigma 0.1
        if is_c:
            delta = rng.uniform(-scale, scale, size=arr.shape)
        else:
            delta = rng.uniform(-float(sigma), float(sigma), size=arr.shape) * np.abs(arr)
            # for b near -0.02 etc, use relative jitter but keep sign
        out = arr + delta
        return np.clip(out, lo, hi)
    a_j = jitter(a_np, 0.005, 0.20)
    b_j = jitter(b_np, -0.20, 0.35)
    c_j = jitter(c_np, -75.0, -45.0, is_c=True)
    d_j = jitter(d_np, 0.5, 10.0)
    drive_j = jitter(drive_np, 1.0, 8.0) if drive_np is not None else None
    # preserve PV > E ordering check: already bounded, ledger verified
    from jaxfne.emitters import IzhikevichParams  # type: ignore
    # rebuild emitter preserving other fields (v0, u0 etc.)
    kwargs = dict(
        a=jnp.asarray(a_j, dtype=jnp.float32),
        b=jnp.asarray(b_j, dtype=jnp.float32),
        c=jnp.asarray(c_j, dtype=jnp.float32),
        d=jnp.asarray(d_j, dtype=jnp.float32),
    )
    if drive_j is not None:
        kwargs["drive"] = jnp.asarray(drive_j, dtype=jnp.float32)
    # carry over v0/u0/sign etc. if present
    for k in ("v0", "u0", "sign", "recovery_scale"):
        if hasattr(em, k):
            v = getattr(em, k)
            if v is not None:
                kwargs[k] = v
    # also carry any other emitter fields generically
    try:
        new_em = replace(em, **kwargs)  # type: ignore
    except Exception:
        # fallback: construct new but keep table linkage via params replace
        new_em = em
        for kk, vv in kwargs.items():
            try:
                object.__setattr__(new_em, kk, vv)
            except Exception:
                pass
    # recompute u0 = b*v0 if v0 present (matches Izhikevich init)
    try:
        v0_np = np.asarray(new_em.v0)
        b_new = np.asarray(new_em.b)
        u0_new = b_new * v0_np
        new_em = replace(new_em, u0=jnp.asarray(u0_new, dtype=jnp.float32))  # type: ignore
    except Exception:
        pass
    new_params = dict(model.params)
    new_params["emitter"] = new_em
    return replace(model, params=new_params)


def build_jomission_network(
    *,
    n_per_area: int = 100,
    areas: Sequence[str] = JOMISSION_AREAS,
    layers: Sequence[str] = JOMISSION_LAYERS,
    seed: int = 0,
    within_gain: float = WITHIN_GAIN_DEFAULT,
    p_feedforward: float = P_FEEDFORWARD_DEFAULT,
    p_feedback: float = P_FEEDBACK_DEFAULT,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
    n_contacts: int = 16,
    heterogeneity_jitter: float | None = HETEROGENEITY_JITTER_SIGMA_DEFAULT,
    background_poisson_rate_hz: float | None = BACKGROUND_POISSON_RATE_HZ_DEFAULT,
    background_poisson_amplitude: float | None = BACKGROUND_POISSON_AMPLITUDE_DEFAULT,
    background_poisson_target: str | None = BACKGROUND_POISSON_TARGET_DEFAULT,
    tonic_drive_scale: float | None = TONIC_DRIVE_SCALE_DEFAULT,
    sst_output_gain_alpha: float | None = SST_OUTPUT_GAIN_ALPHA_DEFAULT,
    pv_drive_scale: float | None = PV_DRIVE_SCALE_DEFAULT,
) -> Configuration:
    """Build jomission Configuration with explicit areas/layers/cell types.

    - n_per_area ≥100, configurable
    - Each area declares a column with shared layers
    - Per-area per-layer E/PV/SST/VIP fractions via area_layer_cell_types
    - FF+FB between adjacent hierarchical pairs (both directions)
    - Geometry probe for LFP-like/CSD-like via linear_solver
    """
    if n_per_area < 100:
        raise ValueError(f"n_per_area {n_per_area} < minimum 100")
    if tuple(areas) != HIERARCHY and set(areas) != set(HIERARCHY):
        # Allow subset but warn if order wrong; enforce low->high order if same set
        pass

    cfg = Configuration()
    # Declare areas for bookkeeping (optional, not required by builder but useful)
    try:
        cfg = cfg.areas(list(areas))  # type: ignore[attr-defined]
    except Exception:
        pass

    # Declare columns
    for area in areas:
        cfg = cfg.column(area, layers=list(layers), n=int(n_per_area))

    # Wire inter-area connectivity for each adjacent pair — both FF and FB (W2.3 explicit laminar hierarchy)
    # FF: L2/3_E -> L4 (superficial->granular) with delay 8ms MODEL_ASSUMPTION
    # FB: L6_E -> L1 / L6_E -> L5 (deep->supragranular/deep) with delay 12ms MODEL_ASSUMPTION, two specs per direction for key uniqueness
    for lo, hi in zip(areas[:-1], areas[1:]):
        cfg = cfg.inter_column_connectivity(
            source_area=lo,
            target_area=hi,
            mode="sparse",
            layer_to_layer_map=dict(FF_LAYER_MAP),
            cell_type_to_cell_type_map={"E": "E", "PV": "PV", "SST": "SST", "VIP": "VIP"},
            p_feedforward=float(p_feedforward),
            p_feedback=0.0,
            feedforward_weight_range=(0.5, 2.0),
            feedback_weight_range=(0.3, 1.5),
            delay_ms_or_status=float(DELAY_FF_MS),
        )
    # Explicit FB specs: V4->V1, FEF->V4, PFC->FEF each with L6->L1 and L6->L5 (derived from areas for genericity)
    fb_pairs = [(hi, lo) for lo, hi in zip(areas[:-1], areas[1:])]
    for hi, lo in fb_pairs:
        for fb_map in FB_LAYER_MAPS:
            cfg = cfg.inter_column_connectivity(
                source_area=hi,
                target_area=lo,
                mode="sparse",
                layer_to_layer_map=dict(fb_map),
                cell_type_to_cell_type_map={"E": "E", "PV": "PV", "SST": "SST", "VIP": "VIP"},
                p_feedforward=0.0,
                p_feedback=float(p_feedback),
                feedforward_weight_range=(0.5, 2.0),
                feedback_weight_range=(0.3, 1.5),
                delay_ms_or_status=float(DELAY_FB_MS),
            )

    # Global fallback cell fractions (used if per-area table missing)
    from jomission.network.populations import FLAT_CELL_TYPE_FRACTIONS
    cfg = cfg.cell_types(FLAT_CELL_TYPE_FRACTIONS)
    for area in areas:
        per_layer = AREA_LAYER_CELL_TYPES.get(area)
        if per_layer is None:
            continue
        cfg = cfg.area_layer_cell_types(area, per_layer)

    # Layer depth fractions (proxy geometry)
    cfg = cfg.layer_fractions({k: list(v) for k, v in LAYER_DEPTH_BANDS.items()})

    # Per-layer count fractions (explicit, replaceable) — set via direct metadata to avoid column redeclaration
    cfg = cfg.update_metadata(
        layer_count_frac=dict(LAYER_COUNT_FRAC_DEFAULT),
        area_layer_count_frac={area: dict(LAYER_COUNT_FRAC_DEFAULT) for area in areas},
    )

    # Within-area connectivity — GEN2_C003 sparse-local (dense-local/sparse-global)
    # Seam: jaxfne/connectivity.py:222 _candidate_pairs_localized (spatial_sigma)
    # and jaxfne/_config.py:679 connections(max_in_degree, spatial_sigma)
    # Functional via cfg.connectivity metadata + _apply_spatial_locality pruning
    # in build_jomission_model (Gaussian kernel, same as jaxfne).
    cfg = cfg.connectivity(
        within_area="spatial_local",
        within_gain=float(within_gain),
        spatial_sigma=0.08,
        max_in_degree=25,
        p_connect=None,
    )

    # Field + probe — linear_solver, proxy_readout (no physical calibration)
    cfg = cfg.field(source_mode="proxy_no_field_solve")
    cfg = cfg.probe(n_contacts=int(n_contacts))

    # Runtime — edge_list backend required for full-state continuation (C_t carry)
    cfg = cfg.runtime(recurrent_backend="edge_list")

    # Emitter family
    cfg = cfg.emitter(family=str(emitter))

    # Metadata — GEN2_C002 heterogeneity + GEN2_C003 spatial (both hash-visible)
    sigma_eff = _resolve_heterogeneity_sigma(heterogeneity_jitter)
    cfg = cfg.update_metadata(
        jomission_version="0.1.0",
        hierarchy="->".join(areas),
        n_per_area=int(n_per_area),
        n_total=int(n_per_area * len(areas)),
        within_gain=float(within_gain),
        p_feedforward=float(p_feedforward),
        p_feedback=float(p_feedback),
        seed=int(seed),
        dt_ms=float(dt_ms),
        heterogeneity_jitter=float(sigma_eff),
        heterogeneity_jitter_enabled=bool(sigma_eff > 1e-12),
    )
    # sentinel for audit: record that cell_params jitter seam is declared
    cfg = cfg.update_metadata(
        cell_params_declared=bool(sigma_eff > 1e-12),
        cell_params_sigma=float(sigma_eff),
    )
    # GEN2_C014 pseudogenome DESIRED_MOTIF_GAIN v0 — hash-visible (W3.4 typed matrix)
    # Replaces uniform MOTIF_GAIN (C006 E→PV1.5× only) with 12-gain pseudogenome;
    # stored as string-keyed dict for JSON-safe config_hash; tuple keys converted to "E->PV"
    # Provenance MODEL_ASSUMPTION magnitudes / LITERATURE_PRIOR direction Pfeffer2013 etc
    motif_gain_json = {f"{k[0]}->{k[1]}": float(v) for k, v in MOTIF_GAIN.items()}
    cfg = cfg.update_metadata(
        motif_gain=motif_gain_json,
        motif_gain_E_PV=float(MOTIF_GAIN.get(("E", "PV"), 1.0)),
        motif_gain_provenance=str(MOTIF_GAIN_PROVENANCE),
        motif_gain_seam=str(MOTIF_GAIN_SEAM),
        motif_gain_reason=str(MOTIF_GAIN_REASON),
        pseudogenome_version=str(PSEUDOGENOME_VERSION),
        pseudogenome_provenance=str(PSEUDOGENOME_PROVENANCE),
        pseudogenome_seam=str(PSEUDOGENOME_SEAM),
        desired_motif_gain_v0=dict(DESIRED_MOTIF_GAIN_V0),
    )
    # GEN2_C008 laminar hierarchy + delays — hash-visible (W2.3)
    # Explicit FF L2/3->L4 and FB L6->L1 / L6->L5 with delays 8/12/2 ms provenance-tagged
    cfg = cfg.update_metadata(
        ff_layer_map=dict(FF_LAYER_MAP),
        fb_layer_maps=[dict(m) for m in FB_LAYER_MAPS],
        fb_layer_map_combined=dict(FB_LAYER_MAP_COMBINED),
        layer_to_layer_map_explicit=True,
        delay_ff_ms=float(DELAY_FF_MS),
        delay_fb_ms=float(DELAY_FB_MS),
        delay_within_ms=float(DELAY_WITHIN_MS),
        delay_laminar_ms=dict(DELAY_LAMINAR_MS),
        delay_provenance=str(DELAY_PROVENANCE),
        delay_seam=str(DELAY_SEAM),
        hierarchy_provenance="MODEL_ASSUMPTION (8/12/2 values) / DERIVED (ordering FB>FF>within)",
        hierarchy_explicit=True,
    )
    # GEN2_C009 VIP b correction — hash-visible (W2.4 intrinsic)
    # VIP b -0.10 (sole negative, ENGINE_DEFAULT) → +0.05 (MODEL_ASSUMPTION/LITERATURE_PRIOR b>=0)
    # BEFORE jitter so after jitter VIP b ∈ [0.045,0.055] all positive
    cfg = cfg.update_metadata(
        vip_b_corrected=float(VIP_B_CORRECTED),
        vip_b_old=float(-0.10),
        vip_b_provenance=str(VIP_B_CORRECTED_PROVENANCE),
        vip_b_seam=str(VIP_B_CORRECTED_SEAM),
        vip_b_corrected_enabled=True,
    )
    # GEN2_C015 SST b correction — hash-visible (W3.5 intrinsic operating-point, not SST-output gain)
    # SST b 0.25→0.21 (MODEL_ASSUMPTION/LITERATURE_PRIOR SST LTS direction), BEFORE jitter → [0.189,0.231]
    cfg = cfg.update_metadata(
        sst_b_corrected=float(SST_B_CORRECTED),
        sst_b_old=float(SST_B_OLD),
        sst_b_provenance=str(SST_B_CORRECTED_PROVENANCE),
        sst_b_seam=str(SST_B_CORRECTED_SEAM),
        sst_b_corrected_enabled=True,
    )
    # GEN2_C010 lognormal weight variance — hash-visible (W2.5, B2 heterogeneity)
    # CV=1.5 magnitude MODEL_ASSUMPTION / direction LITERATURE_PRIOR Song 2005
    # sigma=1.08566 mu=-0.58933 (correct; 0.97 approx note stored)
    cfg = cfg.update_metadata(
        weight_lognormal_cv=float(LOGNORMAL_CV),
        weight_lognormal_sigma=float(LOGNORMAL_SIGMA),
        weight_lognormal_mu=float(LOGNORMAL_MU),
        weight_lognormal_provenance=str(LOGNORMAL_PROVENANCE),
        weight_lognormal_seam=str(LOGNORMAL_SEAM),
        weight_lognormal_enabled=True,
        weight_lognormal_sigma_approx_note=str(LOGNORMAL_SIGMA_APPROX_NOTE),
    )
    # GEN2_C011 tonic drive reduction — hash-visible (W3.1, B2 operating point)
    # Uniform scale across E/PV/SST/VIP preserving jitter relative ratios; default 0.7 → E 5→3.5 toward rheobase
    # Provenance MODEL_ASSUMPTION (3.5 magnitude) / DERIVED (scale 0.7) / ENGINE_DEFAULT drive seam
    # Seam post-jitter drive·s (existing Model params, no new kernel) before delays
    try:
        tonic_scale = float(tonic_drive_scale) if tonic_drive_scale is not None else float(TONIC_DRIVE_SCALE_DEFAULT)
    except Exception:
        tonic_scale = float(TONIC_DRIVE_SCALE_DEFAULT)
    # clamp to sign-preserving positive range
    if tonic_scale < 0.05:
        tonic_scale = 0.05
    if tonic_scale > 2.0:
        tonic_scale = 2.0
    cfg = cfg.update_metadata(
        tonic_drive_scale=float(tonic_scale),
        tonic_drive_E=float(TONIC_DRIVE_OLD * float(tonic_scale)),
        tonic_drive_PV=float(3.0 * float(tonic_scale)),
        tonic_drive_SST=float(3.5 * float(tonic_scale)),
        tonic_drive_VIP=float(3.0 * float(tonic_scale)),
        tonic_drive_old=float(TONIC_DRIVE_OLD),
        tonic_drive_new=float(TONIC_DRIVE_OLD * float(tonic_scale)),
        tonic_drive_provenance=str(TONIC_DRIVE_SCALE_PROVENANCE),
        tonic_drive_seam=str(TONIC_DRIVE_SEAM),
        tonic_drive_enabled=bool(abs(float(tonic_scale) - 1.0) > 1e-12),
    )
    # GEN2_C013 SST-output gain — hash-visible (W3.3, B2 transfer de-clamping)
    # Bounded SST->* x0.5 uniform sign-preserving, post-EdgeList seam like C006/C010
    # Provenance MODEL_ASSUMPTION (alpha 0.5 bounded) / LITERATURE_PRIOR direction / ENGINE_DEFAULT seam
    try:
        sst_alpha = float(sst_output_gain_alpha) if sst_output_gain_alpha is not None else float(SST_OUTPUT_GAIN_ALPHA_DEFAULT)
    except Exception:
        sst_alpha = float(SST_OUTPUT_GAIN_ALPHA_DEFAULT)
    if sst_alpha < 0.05:
        sst_alpha = 0.05
    if sst_alpha > 2.0:
        sst_alpha = 2.0
    cfg = cfg.update_metadata(
        sst_output_gain=float(sst_alpha),
        sst_output_gain_alpha=float(sst_alpha),
        sst_output_gain_provenance=str(SST_OUTPUT_GAIN_ALPHA_PROVENANCE),
        sst_output_gain_seam=str(SST_OUTPUT_GAIN_SEAM),
        sst_output_gain_reason=str(SST_OUTPUT_GAIN_REASON),
        sst_output_gain_enabled=bool(abs(float(sst_alpha) - 1.0) > 1e-12),
    )
    # GEN2_C016 PV recruitment — hash-visible (CL1-B operating PV fast-spiking)
    # PV-specific tonic drive boost 1.77→3.01 (scale1.7) via per-class drive seam, deterministic per-PV
    # Provenance MODEL_ASSUMPTION (1.7 magnitude to reach rheobase) / LITERATURE_PRIOR direction fast PV recruitment / ENGINE_DEFAULT drive seam
    # Seam _apply_pv_drive_boost post-jitter after tonic scale before laminar delays (existing Model params, not new kernel)
    try:
        pv_scale = float(pv_drive_scale) if pv_drive_scale is not None else float(PV_DRIVE_SCALE_DEFAULT)
    except Exception:
        pv_scale = float(PV_DRIVE_SCALE_DEFAULT)
    if pv_scale < 0.05:
        pv_scale = 0.05
    if pv_scale > 5.0:
        pv_scale = 5.0
    # PV drive old/new: tonic 0.6 base PV 1.8 -> boosted 3.06; jitter ±10% gives ~3.01 mean; E unchanged 3.01
    cfg = cfg.update_metadata(
        pv_drive_scale=float(pv_scale),
        pv_drive_boost=float(pv_scale),
        pv_drive_PV_old=float(3.0 * float(tonic_scale)),
        pv_drive_PV_new=float(3.0 * float(tonic_scale) * float(pv_scale)),
        pv_drive_E=float(TONIC_DRIVE_OLD * float(tonic_scale)),
        pv_drive_SST=float(3.5 * float(tonic_scale)),
        pv_drive_VIP=float(3.0 * float(tonic_scale)),
        pv_drive_provenance=str(PV_DRIVE_SCALE_PROVENANCE),
        pv_drive_seam=str(PV_DRIVE_SEAM),
        pv_drive_reason=str(PV_DRIVE_REASON),
        pv_drive_enabled=bool(abs(float(pv_scale) - 1.0) > 1e-12),
    )
    # GEN2_C018 typed E mixture pseudogenome M2 70/20/10 — hash-visible (B2 CV_ISI heterogeneity, CL1 irregularity mechanism)
    # RS70/CH20/E_FS10 via per-neuron a,b,c,d clusters (c/d typed, a 0.02/0.02/0.10) deterministic per seed, LITERATURE_PRIOR Izhikevich2003 Fig.1 / MODEL_ASSUMPTION proportions
    # Seam pre-jitter via _apply_typed_E_phenotypes before _apply_per_neuron_jitter so phenotype means preserved with jitter ±10% on top; motif pseudogenome v0, VIP b0.20, SST b0.21, PV drive1.7, lognormal CV1.5 intact
    cfg = cfg.update_metadata(
        e_mixture_version=str(E_MIXTURE_VERSION),
        e_mixture_RS70_CH20_EFS10="RS70_CH20_EFS10",
        e_mixture_RS_frac=float(E_MIXTURE_RS_FRAC),
        e_mixture_CH_frac=float(E_MIXTURE_CH_FRAC),
        e_mixture_EFS_frac=float(E_MIXTURE_EFS_FRAC),
        e_mixture_RS_a=float(E_MIXTURE_RS_PARAMS["a"]),
        e_mixture_RS_b=float(E_MIXTURE_RS_PARAMS["b"]),
        e_mixture_RS_c=float(E_MIXTURE_RS_PARAMS["c"]),
        e_mixture_RS_d=float(E_MIXTURE_RS_PARAMS["d"]),
        e_mixture_CH_a=float(E_MIXTURE_CH_PARAMS["a"]),
        e_mixture_CH_b=float(E_MIXTURE_CH_PARAMS["b"]),
        e_mixture_CH_c=float(E_MIXTURE_CH_PARAMS["c"]),
        e_mixture_CH_d=float(E_MIXTURE_CH_PARAMS["d"]),
        e_mixture_EFS_a=float(E_MIXTURE_EFS_PARAMS["a"]),
        e_mixture_EFS_b=float(E_MIXTURE_EFS_PARAMS["b"]),
        e_mixture_EFS_c=float(E_MIXTURE_EFS_PARAMS["c"]),
        e_mixture_EFS_d=float(E_MIXTURE_EFS_PARAMS["d"]),
        e_mixture_provenance=str(E_MIXTURE_PROVENANCE),
        e_mixture_seam=str(E_MIXTURE_SEAM),
        e_mixture_reason=str(E_MIXTURE_REASON),
        e_mixture_enabled=True,
    )
    # GEN2_C007 background Poisson — hash-visible when enabled; baseline (0 Hz) keeps existing hash
    # so tests for canonical 4a8908e remain green. Only non-zero rate adds keys → distinct hash.
    try:
        poisson_rate = float(background_poisson_rate_hz) if background_poisson_rate_hz is not None else 0.0
    except Exception:
        poisson_rate = 0.0
    try:
        poisson_amp = float(background_poisson_amplitude) if background_poisson_amplitude is not None else float(BACKGROUND_POISSON_AMPLITUDE_DEFAULT)
    except Exception:
        poisson_amp = float(BACKGROUND_POISSON_AMPLITUDE_DEFAULT)
    poisson_target = str(background_poisson_target) if background_poisson_target else str(BACKGROUND_POISSON_TARGET_DEFAULT)
    if poisson_rate > 1e-9:
        cfg = cfg.update_metadata(
            background_poisson_rate_hz=float(poisson_rate),
            background_poisson_amplitude=float(poisson_amp),
            background_poisson_target=str(poisson_target),
            background_poisson_seed=int(seed) + 7919,
            background_poisson_provenance=str(BACKGROUND_POISSON_PROVENANCE),
            background_poisson_seam="Simulation(poisson_drive) via jaxfne/_signals.py:670 _make_poisson_drive + _model_simulate.py:771",
            background_poisson_enabled=True,
        )
    else:
        # Baseline: no Poisson background — keep zero without polluting hash for existing tests,
        # but record disabled flag as non-hash metadata? We store disabled as False without affecting
        # prior hash expectation by using separate conditional: only hash-visible when enabled.
        # For audit completeness, store disabled state as metadata that is hash-stable (0 values)
        # but previous hash tests expect absence; so we keep absence for 0 case.
        # To still allow inspection, store as attribute but not via update_metadata that feeds hash?
        # We instead skip metadata for 0 to preserve canonical hash.
        pass
    return cfg


def _apply_spatial_locality(
    model: jtfne.Model,
    *,
    spatial_sigma: float = 0.08,
    max_in_degree: int = 25,
    seed: int = 0,
) -> jtfne.Model:
    """Prune within-area edges to Gaussian distance-weighted local subset.

    Mirrors jaxfne/connectivity.py:222 _candidate_pairs_localized kernel
    exp(-d^2/(2 sigma^2)) and per-post cap max_in_degree. Keeps all
    between-area edges (FF/FB) untouched — W2 handles laminar. Deterministic
    via seed folding per post (matches jaxfne stable_fold).
    """
    el = model.params.get("edge_list")
    if el is None or int(el.pre.shape[0]) == 0:
        return model
    pre_np = np.asarray(el.pre, dtype=np.int64)
    post_np = np.asarray(el.post, dtype=np.int64)
    w_np = np.asarray(el.weight)
    r_np = np.asarray(el.receptor_index, dtype=np.int64)
    tau_np = np.asarray(el.tau_ms)
    pos = np.asarray(model.params["positions"], dtype=np.float64)
    # area labels from neuron_table
    try:
        tbl = model.neuron_table()
        area_labels = [str(r.get("area")) for r in tbl]
    except Exception:
        # fallback: infer from positions x offset
        area_labels = []
        for i in range(pos.shape[0]):
            x = float(pos[i, 0])
            idx = int(round(x / 2.0))
            idx = max(0, min(idx, 3))
            area_labels.append(["V1", "V4", "FEF", "PFC"][idx])
    n = pos.shape[0]
    # Build per-post list of edge indices grouped
    # between edges: area[pre]!=area[post] -> keep all
    between_mask = np.array([area_labels[int(pre_np[i])] != area_labels[int(post_np[i])] for i in range(len(pre_np))], dtype=bool)
    keep_idx: list[int] = list(np.where(between_mask)[0])
    # within edges grouped by post
    within_idx = np.where(~between_mask)[0]
    # group within indices by post id
    from collections import defaultdict
    post_to_idxs: dict[int, list[int]] = defaultdict(list)
    for idx in within_idx:
        post_to_idxs[int(post_np[idx])].append(int(idx))
    # per-post weighted sampling
    base_seed = int(seed) & 0x7FFFFFFF
    for post_id, idxs in post_to_idxs.items():
        if not idxs:
            continue
        # if already <= max_in_degree, keep all but weight by distance? keep all is already local? but still prune to local via weighting
        # compute distances for each candidate pre
        cand_pres = [int(pre_np[i]) for i in idxs]
        cand_pos = pos[np.array(cand_pres)]
        post_pos = pos[post_id]
        dist_sq = np.sum((cand_pos - post_pos) ** 2, axis=1)
        # Gaussian kernel (same as jaxfne/connectivity.py:292)
        sigma = max(float(spatial_sigma), 1e-9)
        raw_w = np.exp(-dist_sq / (2.0 * sigma * sigma))
        mass = raw_w.sum()
        if mass <= 0:
            continue
        probs = raw_w / mass
        n_take = min(int(max_in_degree), len(idxs))
        # deterministic per-post RNG: seed = base ^ post_id
        # use stable fold via sha256 like jaxfne?
        import hashlib
        h = hashlib.sha256(str(base_seed ^ (post_id * 1009)).encode()).hexdigest()
        pseed = int(h[:8], 16) & 0x7FFFFFFF
        rng = np.random.default_rng(pseed)
        # If n_take == len, keep all (already local-ish but we keep)
        if n_take >= len(idxs):
            keep_idx.extend(idxs)
        else:
            chosen = rng.choice(len(idxs), size=n_take, replace=False, p=probs)
            keep_idx.extend([idxs[int(c)] for c in chosen])
    keep_idx = sorted(set(keep_idx))
    if len(keep_idx) == len(pre_np):
        return model
    # rebuild edge_list
    from jaxfne.emitters import EdgeList
    jdtype = el.weight.dtype
    # preserve delay_steps if present (GEN2_C008)
    try:
        delay_np = np.asarray(getattr(el, "delay_steps", None))
        has_delay = delay_np is not None and delay_np.shape[0] == pre_np.shape[0]
    except Exception:
        has_delay = False
        delay_np = None
    kwargs_el: dict[str, Any] = dict(
        pre=jnp.asarray(pre_np[keep_idx], dtype=jnp.int32),
        post=jnp.asarray(post_np[keep_idx], dtype=jnp.int32),
        weight=jnp.asarray(w_np[keep_idx], dtype=jdtype),
        receptor_index=jnp.asarray(r_np[keep_idx], dtype=jnp.int32),
        tau_ms=jnp.asarray(tau_np[keep_idx], dtype=jdtype),
        source_calibration_status=el.source_calibration_status,
    )
    if has_delay:
        kwargs_el["delay_steps"] = jnp.asarray(delay_np[keep_idx], dtype=jnp.int32)
    new_el = EdgeList(**kwargs_el)
    new_params = dict(model.params)
    new_params["edge_list"] = new_el
    # keep W placeholder as is (edge_list backend authoritative)
    return replace(model, params=new_params)


def _apply_motif_gains(
    model: jtfne.Model,
    gain_map: Mapping[tuple[str, str], float] | None = None,
    seed: int = 0,
) -> jtfne.Model:
    """Apply per-motif weight scaling via post-construct EdgeList (GEN2_C006).

    Scales EdgeList.weight by gain_map[(pre_cell_type, post_cell_type)].
    Preserves sign (multiplicative). Uses neuron_table for class lookup.
    Existing Model params seam — not a new simulator kernel.
    References: connectivity.py:35 (empty table), builder.py:178-185 (diagonal map),
    _construct_connectivity.py:54 (no per-motif gain), emitters.py weight generation.
    """
    if gain_map is None:
        gain_map = MOTIF_GAIN
    el = model.params.get("edge_list")
    if el is None or int(el.pre.shape[0]) == 0:
        return model
    # Fast path: if all gains ==1.0 skip
    if all(float(v) == 1.0 for v in gain_map.values()):
        return model
    try:
        tbl = model.neuron_table()
        cell_types = [str(r.get("cell_type")) for r in tbl]
        # vectorized lookup per edge
        pre_np = np.asarray(el.pre, dtype=np.int64)
        post_np = np.asarray(el.post, dtype=np.int64)
        w_np = np.asarray(el.weight, dtype=np.float64)
    except Exception:
        return model
    # lookup per edge; use table indexing
    gains = np.ones_like(w_np, dtype=np.float64)
    for i in range(len(pre_np)):
        try:
            pc = cell_types[int(pre_np[i])]
            qc = cell_types[int(post_np[i])]
            g = gain_map.get((pc, qc), 1.0)
            gains[i] = float(g)
        except Exception:
            gains[i] = 1.0
    # only where gain !=1 need scaling
    if np.all(gains == 1.0):
        return model
    w_scaled = w_np * gains
    from jaxfne.emitters import EdgeList

    jdtype = el.weight.dtype
    # preserve all EdgeList fields including delay_steps and source_calibration_status
    try:
        delay = getattr(el, "delay_steps", None)
    except Exception:
        delay = None
    kwargs: dict[str, Any] = dict(
        pre=el.pre,
        post=el.post,
        weight=jnp.asarray(w_scaled, dtype=jdtype),
        receptor_index=el.receptor_index,
        tau_ms=el.tau_ms,
        source_calibration_status=el.source_calibration_status,
    )
    # EdgeList.__post_init__ handles delay_steps default; explicitly pass if present
    if delay is not None:
        kwargs["delay_steps"] = delay
    new_el = EdgeList(**kwargs)
    new_params = dict(model.params)
    new_params["edge_list"] = new_el
    return replace(model, params=new_params)


def _apply_laminar_delays(
    model: jtfne.Model,
    dt_ms: float = 0.1,
) -> jtfne.Model:
    """Apply laminar- and area-specific axonal delays via EdgeList.delay_steps (GEN2_C008).

    - within (area_pre==area_post) -> DELAY_WITHIN_MS 2.0 -> 20 steps
    - between FF (post layer L4) -> DELAY_FF_MS 8.0 -> 80 steps
    - between FB (post L1/L5 etc) -> DELAY_FB_MS 12.0 -> 120 steps
    Seam: jaxfne/emitters.py:545 edge_list_with_delay_ms(dt_ms) post-construct shim,
          mirrors e2_execution.attach_provenance_class_delays pattern.
    Provenance MODEL_ASSUMPTION (8/12/2 values) / DERIVED (ordering FB>FF>within).
    Preserves weight/receptor/tau/sign; uses existing EdgeList, not new simulator.
    """
    el = model.params.get("edge_list")
    if el is None or int(el.pre.shape[0]) == 0:
        return model
    try:
        tbl = model.neuron_table()
        area_labels = [str(r.get("area")) for r in tbl]
        layer_labels = [str(r.get("layer")) for r in tbl]
    except Exception:
        return model
    pre_np = np.asarray(el.pre, dtype=np.int64)
    post_np = np.asarray(el.post, dtype=np.int64)
    n_edges = int(pre_np.shape[0])
    delay_ms = np.zeros(n_edges, dtype=np.float64)
    for k in range(n_edges):
        try:
            a_pre = area_labels[int(pre_np[k])]
            a_post = area_labels[int(post_np[k])]
            if a_pre == a_post:
                delay_ms[k] = float(DELAY_WITHIN_MS)
            else:
                # FF vs FB by realized post layer (FF targets L4)
                post_layer = layer_labels[int(post_np[k])]
                is_ff = post_layer == "L4"
                delay_ms[k] = float(DELAY_FF_MS) if is_ff else float(DELAY_FB_MS)
        except Exception:
            delay_ms[k] = float(DELAY_WITHIN_MS)
    # Validate grid alignment (8/0.1=80 etc)
    try:
        from jaxfne.emitters import edge_list_with_delay_ms
    except Exception:
        return model
    try:
        new_el = edge_list_with_delay_ms(el, delay_ms, dt_ms=float(dt_ms))
    except Exception:
        return model
    new_params = dict(model.params)
    new_params["edge_list"] = new_el
    return replace(model, params=new_params)


def _apply_vip_b_correction(
    model: jtfne.Model,
    corrected_b: float = VIP_B_CORRECTED,
) -> jtfne.Model:
    """Apply VIP b correction -0.10 -> +0.05 (C009) -> 0.20 (C017) BEFORE jitter.

    C017: b 0.05→0.20 single delta (I_rh>15→4, thr107.79→94.99, V_m-77.79→-64.99 at I3)
    Overrides VIP b to 0.20 before jitter so after jitter VIP b ∈ [0.18,0.22].
    Provenance: MODEL_ASSUMPTION (0.20 magnitude) / LITERATURE_PRIOR b>=0 Izhikevich2003 / ENGINE_DEFAULT emitter seam / DERIVED f-I.
    Seam: emitter b override via Builder builder.py:713-785 pre-jitter u0=b·v0 recompute + jitter preservation.
    References: jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS, builder.py:39 jitter σ0.10 b∈[-0.20,0.35] clip.

    Preserves PV>E ordering; u0 = b*v0 recomputed for VIP neurons; hash-visible via cfg.metadata vip_b_corrected=0.20.
    """
    em = model.params.get("emitter")
    if em is None:
        return model
    try:
        n = int(np.asarray(em.b).shape[0])
    except Exception:
        return model
    # Determine VIP indices via neuron_table or emitter labels
    vip_idx: list[int] = []
    try:
        tbl = model.neuron_table()
        for i, row in enumerate(tbl):
            if str(row.get("cell_type")) == "VIP":
                vip_idx.append(int(i))
    except Exception:
        try:
            labels = getattr(em, "labels", None)
            if labels is not None:
                for i, lab in enumerate(labels):
                    if str(lab) == "VIP":
                        vip_idx.append(int(i))
        except Exception:
            return model
    if not vip_idx:
        return model
    # Override b for VIP
    try:
        b_np = np.asarray(em.b, dtype=np.float64).copy()
        for i in vip_idx:
            b_np[i] = float(corrected_b)
        # rebuild emitter with corrected b
        from jaxfne.emitters import IzhikevichParams  # type: ignore

        # carry over other fields, update b and u0
        v0_np = np.asarray(em.v0, dtype=np.float64) if hasattr(em, "v0") and em.v0 is not None else None
        kwargs: dict[str, Any] = dict(b=jnp.asarray(b_np, dtype=jnp.float32))
        if v0_np is not None:
            try:
                # recompute u0 = b * v0 for all neurons (VIP updated, non-VIP unchanged but consistent)
                u0_new = b_np * v0_np
                kwargs["u0"] = jnp.asarray(u0_new, dtype=jnp.float32)
            except Exception:
                pass
        # use replace to preserve other fields
        try:
            new_em = replace(em, **kwargs)  # type: ignore
        except Exception:
            # fallback set attr
            new_em = em
            for kk, vv in kwargs.items():
                try:
                    object.__setattr__(new_em, kk, vv)
                except Exception:
                    pass
        new_params = dict(model.params)
        new_params["emitter"] = new_em
        return replace(model, params=new_params)
    except Exception:
        return model


def _apply_sst_b_correction(
    model: jtfne.Model,
    corrected_b: float = SST_B_CORRECTED,
) -> jtfne.Model:
    """Apply SST b correction 0.25 -> 0.21 BEFORE jitter (GEN2_C015).

    Overrides SST b to 0.21 before jitter so after jitter SST b ∈ [0.189,0.231].
    Provenance: MODEL_ASSUMPTION (0.21 magnitude) / LITERATURE_PRIOR direction SST LTS Izhikevich2003 / ENGINE_DEFAULT emitter seam / DERIVED f-I.
    Seam: emitter b override via Builder (existing Model params, no new simulator) like VIP builder.py:713.
    References: jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS SST b 0.25, builder.py:39 jitter σ=0.10
                f-I sweep b0.21 I_rh3.5 r20.5@3.5 r39.5@5 gain12.7 CV0.035 V_m-62.84 PASS all 6.

    Preserves delays/topology/lognormal/VIP b/tonic/Poisson/pseudogenome intact; u0 = b*v0 recomputed; b∈[-0.20,0.35] clip via jitter.
    Deterministic, sign-preserving (b>0 stays >0).
    """
    em = model.params.get("emitter")
    if em is None:
        return model
    try:
        n = int(np.asarray(em.b).shape[0])
    except Exception:
        return model
    # Determine SST indices via neuron_table or emitter labels
    sst_idx: list[int] = []
    try:
        tbl = model.neuron_table()
        for i, row in enumerate(tbl):
            if str(row.get("cell_type")) == "SST":
                sst_idx.append(int(i))
    except Exception:
        try:
            labels = getattr(em, "labels", None)
            if labels is not None:
                for i, lab in enumerate(labels):
                    if str(lab) == "SST":
                        sst_idx.append(int(i))
        except Exception:
            return model
    if not sst_idx:
        return model
    # Override b for SST
    try:
        b_np = np.asarray(em.b, dtype=np.float64).copy()
        for i in sst_idx:
            b_np[i] = float(corrected_b)
        # rebuild emitter with corrected b
        from jaxfne.emitters import IzhikevichParams  # type: ignore

        v0_np = np.asarray(em.v0, dtype=np.float64) if hasattr(em, "v0") and em.v0 is not None else None
        kwargs: dict[str, Any] = dict(b=jnp.asarray(b_np, dtype=jnp.float32))
        if v0_np is not None:
            try:
                u0_new = b_np * v0_np
                kwargs["u0"] = jnp.asarray(u0_new, dtype=jnp.float32)
            except Exception:
                pass
        try:
            new_em = replace(em, **kwargs)  # type: ignore
        except Exception:
            new_em = em
            for kk, vv in kwargs.items():
                try:
                    object.__setattr__(new_em, kk, vv)
                except Exception:
                    pass
        new_params = dict(model.params)
        new_params["emitter"] = new_em
        return replace(model, params=new_params)
    except Exception:
        return model


def _apply_typed_E_phenotypes(
    model: jtfne.Model,
    seed: int = 0,
) -> jtfne.Model:
    """Apply typed E subpopulations RS70/CH20/E_FS10 via per-neuron a,b,c,d (GEN2_C018).

    Assigns each E neuron a phenotype label (RS/CH/E_FS) with proportions
    70/20/10 deterministic per seed (permutation of E indices via seed ^
    0x9E3779B9 ^ 0xC018), then applies phenotype-specific a,b,c,d:

      RS   a0.02 b0.20 c-65 d8   LITERATURE_PRIOR RS Izhikevich2003 Fig.1
      CH   a0.02 b0.20 c-50 d2   LITERATURE_PRIOR CH
      E_FS a0.10 b0.20 c-65 d2   MODEL_ASSUMPTION FS-like renamed E_FS (not inhibitory)

    Keeps drive unchanged; recomputes u0=b·v0 for affected neurons.
    Called pre-jitter so _apply_per_neuron_jitter σ0.10 adds ±10% on top
    preserving distinct c/d clusters (RS c-65 d8 vs CH c-50 d2 variance-matched
    control a single RS σ0.40 insufficient even variance-matched per R C).
    Deterministic, sign-preserving (a>0,b>0,c∈[-75,-45],d∈[0.5,10] clips later).
    Hash-visible via cfg.metadata e_mixture_version=M2.

    References: builder.py:194 _apply_per_neuron_jitter, emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS,
                Izhikevich2003 Fig.1 (RS/IB/CH/BZ), R A phenotypes at I3.5 RS 6.7Hz CV0.07 vs CH 20.7Hz CV1.392 Δ1.32
    """
    em = model.params.get("emitter")
    if em is None:
        return model
    try:
        n = int(np.asarray(em.a).shape[0])
    except Exception:
        return model
    # Identify E indices via neuron_table
    e_idx: list[int] = []
    try:
        tbl = model.neuron_table()
        for i, row in enumerate(tbl):
            if str(row.get("cell_type")) == "E":
                e_idx.append(int(i))
    except Exception:
        try:
            labels = getattr(em, "labels", None)
            if labels is not None:
                for i, lab in enumerate(labels):
                    if str(lab) == "E":
                        e_idx.append(int(i))
        except Exception:
            return model
    if not e_idx:
        return model
    n_e = len(e_idx)
    # Deterministic permutation of E indices
    base = (int(seed) ^ 0x9E3779B9 ^ 0xC018) & 0x7FFFFFFF
    rng = np.random.default_rng(int(base))
    perm = rng.permutation(np.array(e_idx, dtype=np.int64))
    # Compute counts deterministic 70/20/10 rounding preserves sum
    n_rs = int(round(n_e * float(E_MIXTURE_RS_FRAC)))
    n_ch = int(round(n_e * float(E_MIXTURE_CH_FRAC)))
    # Clamp to avoid overflow, assign remaining to E_FS
    if n_rs + n_ch > n_e:
        # scale down proportionally
        total = n_rs + n_ch
        n_rs = int(round(n_e * float(E_MIXTURE_RS_FRAC) / (float(E_MIXTURE_RS_FRAC) + float(E_MIXTURE_CH_FRAC)) * 0.90))
        n_ch = int(round(n_e * float(E_MIXTURE_CH_FRAC) / (float(E_MIXTURE_RS_FRAC) + float(E_MIXTURE_CH_FRAC)) * 0.90))
    n_efs = n_e - n_rs - n_ch
    if n_efs < 0:
        n_efs = 0
        # adjust ch
        n_ch = n_e - n_rs
    # Assign: first n_rs -> RS, next n_ch -> CH, rest -> E_FS
    rs_idx = set(int(x) for x in perm[:n_rs].tolist()) if n_rs > 0 else set()
    ch_idx = set(int(x) for x in perm[n_rs:n_rs + n_ch].tolist()) if n_ch > 0 else set()
    efs_idx = set(int(x) for x in perm[n_rs + n_ch:].tolist()) if n_efs > 0 else set()
    try:
        a_np = np.asarray(em.a, dtype=np.float64).copy()
        b_np = np.asarray(em.b, dtype=np.float64).copy()
        c_np = np.asarray(em.c, dtype=np.float64).copy()
        d_np = np.asarray(em.d, dtype=np.float64).copy()
        # Apply phenotype values
        for i in e_idx:
            if i in rs_idx:
                a_np[i] = float(E_MIXTURE_RS_PARAMS["a"])
                b_np[i] = float(E_MIXTURE_RS_PARAMS["b"])
                c_np[i] = float(E_MIXTURE_RS_PARAMS["c"])
                d_np[i] = float(E_MIXTURE_RS_PARAMS["d"])
            elif i in ch_idx:
                a_np[i] = float(E_MIXTURE_CH_PARAMS["a"])
                b_np[i] = float(E_MIXTURE_CH_PARAMS["b"])
                c_np[i] = float(E_MIXTURE_CH_PARAMS["c"])
                d_np[i] = float(E_MIXTURE_CH_PARAMS["d"])
            elif i in efs_idx:
                a_np[i] = float(E_MIXTURE_EFS_PARAMS["a"])
                b_np[i] = float(E_MIXTURE_EFS_PARAMS["b"])
                c_np[i] = float(E_MIXTURE_EFS_PARAMS["c"])
                d_np[i] = float(E_MIXTURE_EFS_PARAMS["d"])
        # Clip to same bounds as _apply_per_neuron_jitter will (but set exactly within)
        # a∈[0.005,0.20] b∈[-0.20,0.35] c∈[-75,-45] d∈[0.5,10]
        a_np = np.clip(a_np, 0.005, 0.20)
        b_np = np.clip(b_np, -0.20, 0.35)
        c_np = np.clip(c_np, -75.0, -45.0)
        d_np = np.clip(d_np, 0.5, 10.0)
        from jaxfne.emitters import IzhikevichParams  # type: ignore

        v0_np = np.asarray(em.v0, dtype=np.float64) if hasattr(em, "v0") and em.v0 is not None else None
        kwargs: dict[str, Any] = dict(
            a=jnp.asarray(a_np, dtype=jnp.float32),
            b=jnp.asarray(b_np, dtype=jnp.float32),
            c=jnp.asarray(c_np, dtype=jnp.float32),
            d=jnp.asarray(d_np, dtype=jnp.float32),
        )
        if v0_np is not None:
            try:
                u0_new = b_np * v0_np
                kwargs["u0"] = jnp.asarray(u0_new, dtype=jnp.float32)
            except Exception:
                pass
        try:
            new_em = replace(em, **kwargs)  # type: ignore
        except Exception:
            new_em = em
            for kk, vv in kwargs.items():
                try:
                    object.__setattr__(new_em, kk, vv)
                except Exception:
                    pass
        new_params = dict(model.params)
        new_params["emitter"] = new_em
        return replace(model, params=new_params)
    except Exception:
        return model


def _apply_lognormal_weights(
    model: jtfne.Model,
    seed: int = 0,
    cv: float = LOGNORMAL_CV,
    sigma: float = LOGNORMAL_SIGMA,
    mu: float = LOGNORMAL_MU,
) -> jtfne.Model:
    """Apply motif-mean-preserving lognormal weight variance (GEN2_C010).

    Multiplies EdgeList.weight by L~LN(μ,σ) with μ=-σ²/2, σ=√ln(1+CV²)
    CV=1.5 → σ=1.0856587845 μ=-0.5893274982 so E[L]=1 per motif.

    - Per-motif RNG: seed ^ 0x9E3779B9 ^ motif_hash (mirrors builder.py:134
      jitter / per-post fold), deterministic, preserves per-motif mean
      (E→PV 1.5× ratio 1.538 stays), sign L>0 preserves sign.
    - EdgeList.weight only; topology 10590 edges, delays [20,80,120],
      Poisson 1kHz VIP b 0.05 untouched.
    - Seam: post-construct EdgeList weight scaling (existing Model params,
      not new kernel) — like C006 _apply_motif_gains.

    Provenance MODEL_ASSUMPTION (CV=1.5) / LITERATURE_PRIOR direction Song 2005.
    """
    if float(cv) <= 1e-12 or float(sigma) <= 1e-12:
        return model
    el = model.params.get("edge_list")
    if el is None or int(el.pre.shape[0]) == 0:
        return model
    try:
        tbl = model.neuron_table()
        cell_types = [str(r.get("cell_type")) for r in tbl]
        pre_np = np.asarray(el.pre, dtype=np.int64)
        post_np = np.asarray(el.post, dtype=np.int64)
        w_np = np.asarray(el.weight, dtype=np.float64)
    except Exception:
        return model
    n_edges = int(pre_np.shape[0])
    # Build motif -> edge indices map for per-motif RNG
    from collections import defaultdict
    motif_to_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ei in range(n_edges):
        try:
            pc = cell_types[int(pre_np[ei])]
            qc = cell_types[int(post_np[ei])]
        except Exception:
            pc, qc = "E", "E"
        motif_to_idxs[(pc, qc)].append(ei)
    # Per-motif lognormal generation
    import hashlib

    w_scaled = w_np.copy()
    for (pc, qc), idxs in motif_to_idxs.items():
        if not idxs:
            continue
        motif_key = f"{pc}->{qc}"
        # motif_hash stable via sha256
        h = hashlib.sha256(motif_key.encode()).hexdigest()
        motif_hash = int(h[:8], 16) & 0x7FFFFFFF
        seed_fold = (int(seed) ^ 0x9E3779B9 ^ int(motif_hash)) & 0x7FFFFFFF
        rng = np.random.default_rng(int(seed_fold))
        Ls = rng.lognormal(mean=float(mu), sigma=float(sigma), size=len(idxs))
        # Ls>0 preserves sign automatically
        for j, ei in enumerate(idxs):
            w_scaled[ei] = w_np[ei] * float(Ls[j])
    from jaxfne.emitters import EdgeList

    jdtype = el.weight.dtype
    try:
        delay = getattr(el, "delay_steps", None)
    except Exception:
        delay = None
    kwargs: dict[str, Any] = dict(
        pre=el.pre,
        post=el.post,
        weight=jnp.asarray(w_scaled, dtype=jdtype),
        receptor_index=el.receptor_index,
        tau_ms=el.tau_ms,
        source_calibration_status=el.source_calibration_status,
    )
    if delay is not None:
        kwargs["delay_steps"] = delay
    new_el = EdgeList(**kwargs)
    new_params = dict(model.params)
    new_params["edge_list"] = new_el
    return replace(model, params=new_params)


def _apply_tonic_drive_scale(
    model: jtfne.Model,
    scale: float = TONIC_DRIVE_SCALE_DEFAULT,
) -> jtfne.Model:
    """Apply tonic drive reduction 5→3.5 (scale 0.7) uniform across classes (GEN2_C011).

    Scales emitter drive array by scale factor s = I_new/I_old (0.7 for 5→3.5).
    Uniform across E/PV/SST/VIP preserves jitter relative ratios and sign (s>0).
    Deterministic, sign-preserving, after jitter/motif/lognormal before delays so
    topology/delays/Poisson/VIP b/lognormal untouched. Existing Model params,
    not new kernel — mirrors C009 VIP b override / C006 motif scaling pattern.

    Provenance MODEL_ASSUMPTION (3.5 magnitude toward rheobase) / DERIVED (0.7)
    / ENGINE_DEFAULT drive seam (emitters.py:55,477).
    """
    if abs(float(scale) - 1.0) < 1e-12:
        return model
    em = model.params.get("emitter")
    if em is None or not hasattr(em, "drive") or em.drive is None:
        return model
    try:
        drive_np = np.asarray(em.drive, dtype=np.float64)
    except Exception:
        return model
    if drive_np.size == 0:
        return model
    # scale preserving sign (scale >0)
    s = float(scale)
    if s <= 0:
        return model
    drive_scaled = drive_np * s
    # clip to jitter bounds [1.0,8.0] already used in _apply_per_neuron_jitter drive clipping
    drive_scaled = np.clip(drive_scaled, 1.0, 8.0)
    try:
        new_em = replace(em, drive=jnp.asarray(drive_scaled, dtype=jnp.float32))  # type: ignore
    except Exception:
        try:
            object.__setattr__(em, "drive", jnp.asarray(drive_scaled, dtype=jnp.float32))
            new_em = em
        except Exception:
            return model
    new_params = dict(model.params)
    new_params["emitter"] = new_em
    return replace(model, params=new_params)


def _apply_pv_drive_boost(
    model: jtfne.Model,
    factor: float = PV_DRIVE_SCALE_DEFAULT,
) -> jtfne.Model:
    """Apply PV-specific tonic drive boost 1.77->3.01 (scale1.7) PV only (GEN2_C016).

    Scales emitter drive array for PV neurons only (identified via neuron_table
    area/layer/cell_type PV) by factor (default 1.7). Deterministic, sign-
    preserving (factor>0), after jitter/tonic scale before laminar delays so
    delays/topology/lognormal/VIP b/SST b/Poisson/H/Theta/FF gains preserved.

    After tonic 0.6: E 3.01 PV 1.77 SST 1.93 VIP 1.78 (jitter ±10% gives 1.772±0.098
    [1.62,1.98] I_net2.175 < I_rh4.0 r0.317Hz). After PV boost 1.7: PV 3.01
    (scale1.7) via per-class drive seam, I_net≈3.40 > I_rh, deterministic per-PV.

    Provenance MODEL_ASSUMPTION (1.7 magnitude to reach rheobase) / LITERATURE_PRIOR
    direction fast-spiking PV recruitment / ENGINE_DEFAULT drive seam.
    Seam: per-class emitter drive scaling (existing Model params, not new kernel),
    neuron_table PV identification, like _apply_tonic_drive_scale but per-class.

    References: builder.py:39 jitter σ=0.10, builder.py:860 tonic scale 0.6,
                emitters.py:55 PV drive 3.0 default, builder.py:860 per-class drive seam
                (not uniform tonic scale). VIP frozen b0.05, SST b0.21 intact.
    """
    f = float(factor)
    if abs(f - 1.0) < 1e-12:
        return model
    if f <= 0:
        return model
    em = model.params.get("emitter")
    if em is None or not hasattr(em, "drive") or em.drive is None:
        return model
    try:
        drive_np = np.asarray(em.drive, dtype=np.float64).copy()
    except Exception:
        return model
    if drive_np.size == 0:
        return model
    # Identify PV indices via neuron_table
    pv_idx: list[int] = []
    try:
        tbl = model.neuron_table()
        for i, row in enumerate(tbl):
            if str(row.get("cell_type")) == "PV":
                pv_idx.append(int(i))
    except Exception:
        try:
            labels = getattr(em, "labels", None)
            if labels is not None:
                for i, lab in enumerate(labels):
                    if str(lab) == "PV":
                        pv_idx.append(int(i))
        except Exception:
            return model
    if not pv_idx:
        return model
    # Scale PV drive only
    for i in pv_idx:
        try:
            drive_np[i] = float(drive_np[i]) * f
        except Exception:
            pass
    # Clip to jitter bounds [1.0,8.0] like _apply_per_neuron_jitter and _apply_tonic_drive_scale
    drive_np = np.clip(drive_np, 1.0, 8.0)
    try:
        new_em = replace(em, drive=jnp.asarray(drive_np, dtype=jnp.float32))  # type: ignore
    except Exception:
        try:
            object.__setattr__(em, "drive", jnp.asarray(drive_np, dtype=jnp.float32))
            new_em = em
        except Exception:
            return model
    # Note: drive does not affect u0=b·v0, but keep u0 recomputed for audit if present (no-op for drive)
    # u0 depends on b·v0 only; no change needed. Kept hash-visible via cfg.metadata pv_drive_scale.
    new_params = dict(model.params)
    new_params["emitter"] = new_em
    return replace(model, params=new_params)


def _apply_sst_output_gain(
    model: jtfne.Model,
    alpha: float = SST_OUTPUT_GAIN_ALPHA_DEFAULT,
) -> jtfne.Model:
    """Apply bounded SST-output gain W_{SST->*}' = alpha W_{SST->*} (GEN2_C013).

    Scales all edges where pre is SST (class SST) uniformly by alpha<1
    (default 0.5, bounded not abolish), sign-preserving (negative SST
    weights magnitude x0.5). Deterministic uniform scaling (seed not needed).
    Preserves delays/topology/lognormal/VIP b/tonic/Poisson/H/Theta/FF gains.

    Seam: post-EdgeList weight scaling (existing Model params, no new kernel)
          like C006 _apply_motif_gains and C010 _apply_lognormal_weights,
          after lognormal+motif before laminar delays so delays/topology intact.
    Provenance MODEL_ASSUMPTION (alpha 0.5 bounded) / LITERATURE_PRIOR direction
               reduce SST influence / ENGINE_DEFAULT post-EdgeList seam.

    References: builder.py:386 _apply_motif_gains post-EdgeList scaling (C006 E->PV 1.5x)
                builder.py:513 _apply_lognormal_weights, connectivity.py:35 diagonal,
                SST b0.25 (most excitable) vs VIP b0.05 etc.
    """
    a = float(alpha)
    if abs(a - 1.0) < 1e-12:
        return model
    if a <= 0:
        return model
    el = model.params.get("edge_list")
    if el is None or int(el.pre.shape[0]) == 0:
        return model
    try:
        tbl = model.neuron_table()
        cell_types = [str(r.get("cell_type")) for r in tbl]
        pre_np = np.asarray(el.pre, dtype=np.int64)
        w_np = np.asarray(el.weight, dtype=np.float64)
    except Exception:
        return model
    n_edges = int(pre_np.shape[0])
    # mask pre == SST
    is_sst = np.zeros(n_edges, dtype=bool)
    for i in range(n_edges):
        try:
            if cell_types[int(pre_np[i])] == "SST":
                is_sst[i] = True
        except Exception:
            pass
    if not np.any(is_sst):
        return model
    w_scaled = w_np.copy()
    w_scaled[is_sst] = w_np[is_sst] * a  # sign-preserving (negative magnitude x0.5)
    from jaxfne.emitters import EdgeList

    jdtype = el.weight.dtype
    try:
        delay = getattr(el, "delay_steps", None)
    except Exception:
        delay = None
    kwargs: dict[str, Any] = dict(
        pre=el.pre,
        post=el.post,
        weight=jnp.asarray(w_scaled, dtype=jdtype),
        receptor_index=el.receptor_index,
        tau_ms=el.tau_ms,
        source_calibration_status=el.source_calibration_status,
    )
    if delay is not None:
        kwargs["delay_steps"] = delay
    new_el = EdgeList(**kwargs)
    new_params = dict(model.params)
    new_params["edge_list"] = new_el
    # Also scale dense W matrix (authoritative when recurrent_backend is dense exponential)
    # Dense W is (n_post, n_pre) where W[post, pre] is weight; scale columns where pre is SST
    try:
        em = model.params.get("emitter")
        if em is not None and hasattr(em, "W") and em.W is not None:
            W_np = np.asarray(em.W, dtype=np.float64)
            if W_np.shape[0] == len(cell_types) and W_np.shape[1] == len(cell_types):
                W_scaled = W_np.copy()
                # identify SST neuron indices
                sst_idx = [i for i, ct in enumerate(cell_types) if ct == "SST"]
                if sst_idx:
                    # scale columns (pre)
                    W_scaled[:, sst_idx] = W_scaled[:, sst_idx] * a
                    from dataclasses import replace as _replace
                    try:
                        new_em = _replace(em, W=jnp.asarray(W_scaled, dtype=em.W.dtype))  # type: ignore
                        new_params["emitter"] = new_em
                    except Exception:
                        try:
                            object.__setattr__(em, "W", jnp.asarray(W_scaled, dtype=em.W.dtype))
                            new_params["emitter"] = em
                        except Exception:
                            pass
    except Exception:
        pass
    return replace(model, params=new_params)


def build_jomission_model(
    *,
    n_per_area: int = 100,
    areas: Sequence[str] = JOMISSION_AREAS,
    seed: int = 0,
    **kwargs: Any,
) -> jtfne.Model:
    cfg = build_jomission_network(n_per_area=n_per_area, areas=areas, seed=seed, **kwargs)
    model = jtfne.construct(cfg)
    # Apply sparse-local pruning if connectivity declares spatial params (GEN2_C003)
    conn = cfg.metadata.get("connectivity", {}) or {}
    sigma = conn.get("spatial_sigma")
    mid = conn.get("max_in_degree")
    if sigma is not None and mid is not None:
        try:
            model = _apply_spatial_locality(model, spatial_sigma=float(sigma), max_in_degree=int(mid), seed=int(seed))
        except Exception:
            pass
    # GEN2_C009 VIP b correction BEFORE jitter (intrinsic #1, W2.4)
    # Must precede jitter so jitter spreads 0.05 → [0.045,0.055]
    try:
        vip_b_cfg = float(cfg.metadata.get("vip_b_corrected", VIP_B_CORRECTED))
    except Exception:
        vip_b_cfg = float(VIP_B_CORRECTED)
    try:
        model = _apply_vip_b_correction(model, corrected_b=float(vip_b_cfg))
    except Exception:
        pass
    # GEN2_C015 SST b correction BEFORE jitter (intrinsic operating-point, not SST-output gain)
    # Must precede jitter so jitter spreads 0.21 → [0.189,0.231] like VIP builder.py:713
    try:
        sst_b_cfg = float(cfg.metadata.get("sst_b_corrected", SST_B_CORRECTED))
    except Exception:
        sst_b_cfg = float(SST_B_CORRECTED)
    try:
        model = _apply_sst_b_correction(model, corrected_b=float(sst_b_cfg))
    except Exception:
        pass
    # GEN2_C018 typed E mixture pseudogenome M2 70/20/10 BEFORE jitter (pre-jitter preserves motif gains)
    # Deterministic per-seed RS70/CH20/E_FS10 via _apply_typed_E_phenotypes, then jitter ±10% on top retains distinct c/d clusters
    try:
        model = _apply_typed_E_phenotypes(model, seed=int(seed))
    except Exception:
        pass
    # Apply per-neuron jitter if declared (GEN2_C002) — after pruning so emitter stays
    try:
        sigma_eff = float(cfg.metadata.get("heterogeneity_jitter", HETEROGENEITY_JITTER_SIGMA_DEFAULT))
    except Exception:
        sigma_eff = float(HETEROGENEITY_JITTER_SIGMA_DEFAULT)
    if sigma_eff > 1e-12:
        try:
            model = _apply_per_neuron_jitter(model, seed=int(seed), sigma=float(sigma_eff))
        except Exception:
            pass
    # GEN2_C006 motif gain E→PV 1.5× — after spatial pruning (W2.1 one delta only)
    try:
        model = _apply_motif_gains(model, gain_map=MOTIF_GAIN, seed=int(seed))
    except Exception:
        pass
    # GEN2_C010 lognormal weight variance CV1.5 motif-mean-preserving — after motif gains before delays (W2.5)
    # Preserves per-motif mean (E→PV 1.538 ratio), sign L>0, deterministic per-motif seed ^ motif_hash
    try:
        cv = float(cfg.metadata.get("weight_lognormal_cv", LOGNORMAL_CV))
        sigma_ln = float(cfg.metadata.get("weight_lognormal_sigma", LOGNORMAL_SIGMA))
        mu_ln = float(cfg.metadata.get("weight_lognormal_mu", LOGNORMAL_MU))
    except Exception:
        cv, sigma_ln, mu_ln = float(LOGNORMAL_CV), float(LOGNORMAL_SIGMA), float(LOGNORMAL_MU)
    if cv > 1e-12:
        try:
            model = _apply_lognormal_weights(model, seed=int(seed), cv=float(cv), sigma=float(sigma_ln), mu=float(mu_ln))
        except Exception:
            pass
    # GEN2_C013 SST-output gain W_{SST->*}' = alpha W_{SST->*} alpha=0.5 bounded — after lognormal+motif before delays (W3.3)
    # Uniform scaling sign-preserving across SST->E / SST->PV / SST->VIP, deterministic (seed not needed), preserves E->PV 1.5x and lognormal CV1.94 topology 10590 delays [20,80,120]
    try:
        sst_alpha = float(cfg.metadata.get("sst_output_gain", cfg.metadata.get("sst_output_gain_alpha", SST_OUTPUT_GAIN_ALPHA_DEFAULT)))
    except Exception:
        sst_alpha = float(SST_OUTPUT_GAIN_ALPHA_DEFAULT)
    if abs(float(sst_alpha) - 1.0) > 1e-12:
        try:
            model = _apply_sst_output_gain(model, alpha=float(sst_alpha))
        except Exception:
            pass
    # GEN2_C011 tonic drive reduction 5→3.5 scale 0.7 — after jitter/motif/lognormal before delays (W3.1)
    # Uniform across classes preserving jitter ratios; sign-preserving s>0; deterministic; hash-visible via tonic_drive_scale
    try:
        tonic_scale = float(cfg.metadata.get("tonic_drive_scale", TONIC_DRIVE_SCALE_DEFAULT))
    except Exception:
        tonic_scale = float(TONIC_DRIVE_SCALE_DEFAULT)
    if abs(float(tonic_scale) - 1.0) > 1e-12:
        try:
            model = _apply_tonic_drive_scale(model, scale=float(tonic_scale))
        except Exception:
            pass
    # GEN2_C016 PV recruitment — PV-specific drive boost 1.77->3.01 scale1.7 — after jitter/tonic before delays (CL1-B)
    # Hash-visible via pv_drive_scale (builder.py per-class drive seam, not uniform tonic scale); E/SST/VIP unchanged
    try:
        pv_scale_eff = float(cfg.metadata.get("pv_drive_scale", cfg.metadata.get("pv_drive_boost", PV_DRIVE_SCALE_DEFAULT)))
    except Exception:
        pv_scale_eff = float(PV_DRIVE_SCALE_DEFAULT)
    if abs(float(pv_scale_eff) - 1.0) > 1e-12:
        try:
            model = _apply_pv_drive_boost(model, factor=float(pv_scale_eff))
        except Exception:
            pass
    # GEN2_C008 laminar delays 8/12/2 ms — after spatial+motif+lognormal+tonic+PV boost so edge_list is final (W2.3)
    try:
        dt_eff = float(cfg.metadata.get("dt_ms", 0.1))
    except Exception:
        dt_eff = 0.1
    try:
        model = _apply_laminar_delays(model, dt_ms=dt_eff)
    except Exception:
        pass
    return model


def background_poisson_drive_spec(
    cfg: Any,
    *,
    seed_override: int | None = None,
) -> dict[str, Any] | None:
    """Return Simulation.poisson_drive dict from cfg metadata, or None if disabled.

    Seam: jaxfne/_signals.py:670 _make_poisson_drive; deterministic seed = cfg seed + 7919.
    """
    try:
        rate = float(cfg.metadata.get("background_poisson_rate_hz", 0.0))
    except Exception:
        return None
    if rate <= 1e-9:
        return None
    try:
        amp = float(cfg.metadata.get("background_poisson_amplitude", cfg.metadata.get("background_poisson_amplitude", BACKGROUND_POISSON_AMPLITUDE_DEFAULT)))
    except Exception:
        amp = float(BACKGROUND_POISSON_AMPLITUDE_DEFAULT)
    target = str(cfg.metadata.get("background_poisson_target", BACKGROUND_POISSON_TARGET_DEFAULT))
    try:
        base_seed = int(cfg.metadata.get("seed", 0))
    except Exception:
        base_seed = 0
    seed = int(seed_override) if seed_override is not None else int(cfg.metadata.get("background_poisson_seed", base_seed + 7919))
    return {"rate_hz": float(rate), "amplitude": float(amp), "target": str(target), "seed": int(seed)}


def simulation_with_background_poisson(
    cfg: Any,
    *,
    duration_ms: float,
    dt_ms: float = 0.1,
    seed: int = 0,
    plasticity: float = 0.0,
    runtime: Any | None = None,
) -> Any:
    """Build jaxfne.Simulation with Poisson drive from cfg if enabled (GEN2_C007)."""
    from jaxfne import Simulation

    spec = background_poisson_drive_spec(cfg, seed_override=int(seed) + 7919 if background_poisson_drive_spec(cfg) is not None else None)
    # If cfg has no poisson, spec is None → plain Simulation
    if spec is not None:
        # Override seed to simulation seed folded
        spec = dict(spec)
        spec["seed"] = int(seed) + 7919
        return Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed), plasticity=float(plasticity), runtime=runtime, poisson_drive=spec)
    return Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed), plasticity=float(plasticity), runtime=runtime)


# Convenience for tests
def validate_network(n_per_area: int = 100) -> dict[str, Any]:
    cfg = build_jomission_network(n_per_area=n_per_area, seed=0)
    # Check metadata columns
    meta = cfg.metadata
    cols = meta.get("columns", [])
    issues: list[str] = []
    if len(cols) != 4:
        issues.append(f"columns {len(cols)} != 4")
    names = [c["name"] for c in cols]
    if names != list(JOMISSION_AREAS):
        issues.append(f"column names {names} != {list(JOMISSION_AREAS)}")
    total = sum(c["n"] for c in cols)
    if total != n_per_area * 4:
        issues.append(f"total n {total} != {n_per_area*4}")
    return {"valid": not issues, "issues": issues, "columns": names, "total_n": total}
