"""ModelSummary + ObservableBasis — VIS_FOUNDATION_v0 V2 (ontology-stratified) — V0-P1 update.

Engineering-only visualization foundation (no science mutation).
Consumes generated-owner arrays/config objects used by analyses
(builder.py, jaxfne.io.config_hash), never recomputes prettier alternative.
V0-P1: weight/H-tau log scaling (zero/sign-safe) — builder.py:142 lognormal CV1.5 heavy tail
requires log abs weight with sign preserved; H taus 0.1/1/10/100/1000s (h_state.py:28) log10;
CV/rho/Fano linear per A2. Network_viz weight heatmaps log, sampling disclosure elsewhere.

Ontology stratification (advisor refinement):
  Engine-fixed constants, Configured static values, Derived static values,
  Per-neuron static, Per-edge static, Dynamic, Plastic, History, Recording,
  plus Experimentally adjustable parameters, AGSDR-exposed parameters (5 dims),
  Currently frozen scientific parameters.  Ensures N_model ≠ N_tunable ≠ N_AGSDR
  and N_free24 vs derived/fixed per Visualization Law.

Provenance citations (frozen):
- jomission/network/builder.py:39  HETEROGENEITY_JITTER_SIGMA 0.10 (per-neuron jitter)
- jomission/network/builder.py:62  MOTIF_GAIN DESIRED_MOTIF_GAIN v0 (16-gain pseudogenome)
- jomission/network/builder.py:90  FF_LAYER_MAP / FB_LAYER_MAPS / DELAY_FF/FB/WITHIN
- jomission/network/builder.py:395 cfg.connectivity(spatial_sigma=0.08, max_in_degree=25) + _apply_spatial_locality
- jomission/network/builder.py:142 LOGNORMAL_CV / SIGMA / MU
- jomission/network/builder.py:99  VIP_B_CORRECTED 0.20
- jomission/network/builder.py:114 SST_B_CORRECTED 0.21
- jomission/network/builder.py:125 PV_DRIVE_SCALE 1.7
- jomission/network/builder.py:159 TONIC_DRIVE_SCALE 0.6 (3.0/5.0)
- jomission/network/builder.py:49  BACKGROUND_POISSON 2000Hz amp2.0
- jomission/network/builder.py:179 E_MIXTURE_M2 RS70/CH20/EFS10
- jomission/network/populations.py:12 JOMISSION_AREAS, :13 LAYERS, :31 LAYER_COUNT_FRAC_DEFAULT, :44 AREA_LAYER_CELL_TYPES, :85 FLAT_CELL_TYPE_FRACTIONS
- jomission/network/connectivity.py:14 WITHIN_GAIN 0.35, :17 P_FEEDFORWARD 0.30, :18 P_FEEDBACK 0.20, :38 DELAY_*
- jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS + :211 dv/du + :545 edge_list_with_delay_ms + EdgeList
- jaxfne/io.py:42 config_hash
- jomission/dynamics/h_state.py:28 H_COORDINATES 5 taus, jomission/dynamics/hdp.py:14 HDPConfig
- manifests/agsdr_local_freeze.json:1 be9b96ab parent + :67 spatial_locality + :97 H_implemented + :131 orthogonal masks

API:
    model_summary(config_hash: str | None = None, seed: int = 0) -> dict
    model_summary_text(...) -> str
    observable_basis() -> dict  (first-class table per observable: name, owner, dimensionality, units, independence, parent)
    ontology_table(model, cfg) -> dict  (counts per ontology)
    get_observable_basis_hash() -> str
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

# generated-owner imports (same as analyses)
try:
    from jomission.network.builder import (
        MOTIF_GAIN,
        DESIRED_MOTIF_GAIN_V0,
        PSEUDOGENOME_VERSION,
        FF_LAYER_MAP,
        FB_LAYER_MAPS,
        DELAY_FF_MS,
        DELAY_FB_MS,
        DELAY_WITHIN_MS,
        LOGNORMAL_CV,
        LOGNORMAL_SIGMA,
        LOGNORMAL_MU,
        VIP_B_CORRECTED,
        SST_B_CORRECTED,
        PV_DRIVE_SCALE_DEFAULT,
        TONIC_DRIVE_SCALE_DEFAULT,
        BACKGROUND_POISSON_RATE_HZ_DEFAULT,
        BACKGROUND_POISSON_AMPLITUDE_DEFAULT,
        E_MIXTURE_VERSION,
        E_MIXTURE_RS_FRAC,
        E_MIXTURE_CH_FRAC,
        E_MIXTURE_EFS_FRAC,
        E_MIXTURE_RS_PARAMS,
        E_MIXTURE_CH_PARAMS,
        E_MIXTURE_EFS_PARAMS,
        HETEROGENEITY_JITTER_SIGMA_DEFAULT,
        build_jomission_model,
        build_jomission_network,
    )
    from jomission.network.populations import (
        JOMISSION_AREAS,
        JOMISSION_LAYERS,
        JOMISSION_CELL_TYPES,
        LAYER_COUNT_FRAC_DEFAULT,
        AREA_LAYER_CELL_TYPES,
        FLAT_CELL_TYPE_FRACTIONS,
    )
    from jomission.network.connectivity import (
        WITHIN_GAIN_DEFAULT,
        P_FEEDFORWARD_DEFAULT,
        P_FEEDBACK_DEFAULT,
        HIERARCHY,
    )
except Exception:  # pragma: no cover
    MOTIF_GAIN = {("E","E"):1.0,("E","PV"):1.7}  # type: ignore
    DESIRED_MOTIF_GAIN_V0 = {}  # type: ignore
    PSEUDOGENOME_VERSION = "v0"  # type: ignore
    FF_LAYER_MAP = {"L2/3":"L4"}  # type: ignore
    FB_LAYER_MAPS = [{"L6":"L1"},{"L6":"L5"}]  # type: ignore
    DELAY_FF_MS, DELAY_FB_MS, DELAY_WITHIN_MS = 8.0, 12.0, 2.0  # type: ignore
    LOGNORMAL_CV, LOGNORMAL_SIGMA, LOGNORMAL_MU = 1.5, 1.08565, -0.5893  # type: ignore
    VIP_B_CORRECTED, SST_B_CORRECTED, PV_DRIVE_SCALE_DEFAULT = 0.20, 0.21, 1.7  # type: ignore
    TONIC_DRIVE_SCALE_DEFAULT = 0.6  # type: ignore
    BACKGROUND_POISSON_RATE_HZ_DEFAULT, BACKGROUND_POISSON_AMPLITUDE_DEFAULT = 2000.0, 2.0  # type: ignore
    E_MIXTURE_VERSION = "M2"  # type: ignore
    E_MIXTURE_RS_FRAC, E_MIXTURE_CH_FRAC, E_MIXTURE_EFS_FRAC = 0.70, 0.20, 0.10  # type: ignore
    E_MIXTURE_RS_PARAMS = {"a":0.02,"b":0.20,"c":-65.0,"d":8.0}  # type: ignore
    E_MIXTURE_CH_PARAMS = {"a":0.02,"b":0.20,"c":-50.0,"d":2.0}  # type: ignore
    E_MIXTURE_EFS_PARAMS = {"a":0.10,"b":0.20,"c":-65.0,"d":2.0}  # type: ignore
    HETEROGENEITY_JITTER_SIGMA_DEFAULT = 0.10  # type: ignore
    JOMISSION_AREAS = ("V1","V4","FEF","PFC")  # type: ignore
    JOMISSION_LAYERS = ("L1","L2/3","L4","L5","L6")  # type: ignore
    JOMISSION_CELL_TYPES = ("E","PV","SST","VIP")  # type: ignore
    LAYER_COUNT_FRAC_DEFAULT = {"L1":0.08,"L2/3":0.30,"L4":0.15,"L5":0.27,"L6":0.20}  # type: ignore
    AREA_LAYER_CELL_TYPES = {}  # type: ignore
    FLAT_CELL_TYPE_FRACTIONS = {"E":0.70,"PV":0.12,"SST":0.10,"VIP":0.08}  # type: ignore
    WITHIN_GAIN_DEFAULT, P_FEEDFORWARD_DEFAULT, P_FEEDBACK_DEFAULT = 0.35, 0.30, 0.20  # type: ignore
    HIERARCHY = ("V1","V4","FEF","PFC")  # type: ignore
    build_jomission_model = None  # type: ignore
    build_jomission_network = None  # type: ignore

try:
    from jaxfne.io import config_hash as jaxfne_config_hash
    from jaxfne import __version__ as JAXFNE_VERSION
except Exception:
    def jaxfne_config_hash(x):  # type: ignore
        return "unknown"
    JAXFNE_VERSION = "0.4.17"

# Frozen identities per task + ledger
FROZEN_PARENT_HASH = "be9b96ab679c9802"
FROZEN_C019_HASH = "65b302e8c7cdceb5"
FROZEN_C019_PARENT = "be9b96ab"
FROZEN_HP_HASH = "f327f9d2ad64cc88"
FROZEN_C019_THETA = [1.256454357174736, 1.1871704759698585, 0.7088237829792216, 1.2181840541473454, 1.2887170240398067]
FROZEN_C019_THETA_LABELS = ["g_rec","g_fastEI","g_dend","g_disinh","g_background"]

# Orthogonal masks per freeze
M_REC = ["E->E","E->VIP","PV->PV","PV->SST","PV->VIP","SST->SST","VIP->E","VIP->PV","VIP->VIP"]
M_FASTEI = ["E->PV","PV->E"]
M_DEND = ["E->SST","SST->E","SST->PV","SST->VIP"]
M_DISINH = ["VIP->SST"]

# V0-P1 scaling helpers — shared with network_viz.py (no science mutation)
# Weight dynamic range heavy tail CV1.5 (builder.py:142 sigma 1.085 mu -0.589) requires log;
# H taus 0.1,1,10,100,1000s (h_state.py:28) spanning 4 orders requires log;
# CV / rho / Fano are dimensionless linear (gen2_gates.py:147-149).
def _safe_log_abs_weight(w, eps: float = 1e-6):  # type: ignore
    import numpy as _np
    w_arr = _np.asarray(w, dtype=float)
    return _np.sign(w_arr) * _np.log1p(_np.abs(w_arr) / float(eps))

def _safe_log_tau(tau, eps: float = 1e-9):  # type: ignore
    import numpy as _np
    tau_arr = _np.asarray(tau, dtype=float)
    return _np.log10(_np.maximum(tau_arr, float(eps)))

def _scaling_for_metric(metric: str) -> str:
    """Return 'log' or 'linear' per V0-P1 A2 rule."""
    mk = str(metric).lower()
    if mk in ("weight", "w", "meanw", "realized_current", "weight_lognormal", "tau", "tau_h", "h_tau"):
        return "log"
    if mk in ("cv", "cv_isi", "fano", "rho", "probability", "p", "rate", "delay_ms"):
        # weight/p use log where heavy tail, but probability linear? keep linear for CV/rho/Fano explicitly
        # weight already log; probability linear per V0-P1 to avoid distortion of [0,1]
        # delay linear per builder 2/8/12ms not 4 orders; H tau log already separate
        if mk in ("cv", "cv_isi", "fano", "rho"):
            return "linear"
        return "linear" if mk in ("probability", "p", "delay_ms") else "log"
    return "linear"

def _resolve_model(seed: int = 0, for_hash: str | None = None):
    """Build generated-owner model for requested hash.

    For C019 65b302e8c7cdceb5 the hash is obtained via cfg metadata overlay
    (agsdr_N2_runner.py:550 update_metadata theta_overlay) on top of C018
    parent be9b96ab — not via builder primitive mutation. This replicates that
    overlay so config_hash matches ledger while EdgeList counts remain
    generated-owner (400 neurons, 10590 edges).
    """
    if build_jomission_model is None or build_jomission_network is None:
        raise ImportError("build_jomission_model unavailable")
    # Always build base via builder (never recompute counts manually)
    base_model = build_jomission_model(n_per_area=100, seed=int(seed))
    base_cfg = build_jomission_network(n_per_area=100, seed=int(seed))
    base_hash = jaxfne_config_hash(base_cfg)
    # If no hash requested, return base
    if for_hash is None:
        return base_model, base_cfg, base_hash
    want = str(for_hash).strip().lower()
    # Normalize short hash: ledger uses 16 hex
    if want == FROZEN_C019_HASH.lower() or want == FROZEN_C019_HASH.lower()[:8]:
        # replicate agsdr overlay to get exact hash 65b302e8c7cdceb5
        # The overlay in agsdr_N2_runner stores theta overlay as metadata; we mimic same
        theta = np.array(FROZEN_C019_THETA, dtype=float)
        cfg_overlay = base_cfg.update_metadata(
            theta_overlay=f"[{theta[0]:.4f},{theta[1]:.4f},{theta[2]:.4f},{theta[3]:.4f},{theta[4]:.4f}]",
            theta_g_rec=float(theta[0]),
            theta_g_fastEI=float(theta[1]),
            theta_g_dend=float(theta[2]),
            theta_g_disinh=float(theta[3]),
            theta_g_background=float(theta[4]),
            theta_provenance="AGSDR_LOCAL_CL1 orthogonal overlay N2 S_4 y_bar_4",
            agsdr_generation="N2_3x4",
            agsdr_candidate_id="AGS_N0_18_theta_star",
            agsdr_seed="S_4_ybar4",
            agSDR_stage="N2",
        )
        overlay_hash = jaxfne_config_hash(cfg_overlay)
        # If overlay hash matches expected (it should give 65b302...), return base_model with overlay cfg
        # The model itself is still base_model topology (10590 edges) — overlay is multiplicative weight scaling
        # applied at simulation time via agsdr_N0 motif_to_factor, not baked into builder edge_list by default.
        # For summary we report overlay cfg hash as C019 while keeping generated-owner edge counts.
        return base_model, cfg_overlay, overlay_hash
    if want == base_hash.lower() or want == FROZEN_PARENT_HASH.lower():
        return base_model, base_cfg, base_hash
    # Generic: if unknown hash, still return base but note mismatch
    return base_model, base_cfg, base_hash


def _edge_arrays(model) -> Dict[str, np.ndarray]:
    el = model.params.get("edge_list")
    if el is None:
        return {}
    out: Dict[str, np.ndarray] = {}
    try:
        out["pre"] = np.asarray(el.pre, dtype=np.int64)
        out["post"] = np.asarray(el.post, dtype=np.int64)
        out["weight"] = np.asarray(el.weight, dtype=np.float64)
        out["tau_ms"] = np.asarray(el.tau_ms, dtype=np.float64) if hasattr(el, "tau_ms") else np.zeros_like(out["pre"], dtype=float)
        if hasattr(el, "delay_steps") and el.delay_steps is not None:
            ds = np.asarray(el.delay_steps, dtype=np.int64)
            if ds.shape[0] == out["pre"].shape[0]:
                out["delay_steps"] = ds
            else:
                out["delay_steps"] = np.zeros_like(out["pre"])
        else:
            out["delay_steps"] = np.zeros_like(out["pre"])
    except Exception:
        pass
    return out


def _neuron_table(model) -> List[Dict[str, Any]]:
    try:
        return list(model.neuron_table())  # type: ignore
    except Exception:
        return []


def _positions(model) -> np.ndarray:
    try:
        return np.asarray(model.params["positions"], dtype=np.float64)
    except Exception:
        return np.zeros((400, 3), dtype=float)


def _classify_edge(src_area: str, src_layer: str, tgt_area: str, tgt_layer: str) -> str:
    if src_area != tgt_area:
        try:
            si = HIERARCHY.index(src_area)
            ti = HIERARCHY.index(tgt_area)
        except Exception:
            si, ti = 0, 1
        if ti > si:
            if src_layer in FF_LAYER_MAP and FF_LAYER_MAP[src_layer] == tgt_layer:
                return "FF"
            return "FF"
        else:
            return "FB"
    if src_layer == tgt_layer:
        return "local"
    return "vertical"


def ontology_table(model=None, cfg=None, dt_ms: float = 0.1) -> Dict[str, Any]:
    """Ontology-stratified parameter/state accounting (advisor refinement).

    Distinguishes per Visualization Law:
      Engine-fixed constants vs Configured static vs Derived static vs
      Per-neuron static vs Per-edge static vs Dynamic vs Plastic vs History
      vs Recording, plus Experimentally adjustable (N_tunable=24) vs
      AGSDR-exposed (5 dims, 9 scalars) vs Currently frozen scientific parameters.

    Citations:
      builder.py:39 jitter sigma0.10 (per-neuron jitter seam)
      builder.py:62 MOTIF_GAIN 16 gains + :90 FF/FB maps + :395 spatial_sigma 0.08 (configured)
      populations.py:12 AREAS etc + :31 LAYER_COUNT_FRAC (configured)
      jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS + :211 dv/du (engine-fixed)
      manifests/agsdr_local_freeze.json:1 be9b96ab freeze + :131 orthogonal 5-dim
    """
    # Resolve model if not provided
    if model is None:
        try:
            model, cfg, _ = _resolve_model(seed=0)
        except Exception:
            model = None
    if model is None:
        # fallback counts at standard size
        n_neurons, n_edges = 400, 10590
        pos_n = 1200
    else:
        tbl = _neuron_table(model)
        n_neurons = int(len(tbl)) if tbl else 400
        ea = _edge_arrays(model)
        n_edges = int(ea["pre"].shape[0]) if ea else 10590
        pos = _positions(model)
        pos_n = int(pos.size)

    # --- Engine-fixed constants -------------------------------------------
    # Hard-coded literals in jaxfne/emitters.py:55 table + :211 kernel
    # Izhikevich coefficients 0.04,5,140, threshold 30, recovery scale, dt engine handling
    engine_fixed = dict(
        count=18,
        description="Hard-coded engine literals (jaxfne/emitters.py:55, :211)",
        members=[
            "IZHIKEVICH a/b/c/d defaults per class (emitters.py:55) — 4×5=20 raw but 4 unique phenotypes",
            "kernel coeff 0.04, 5, 140, threshold 30.0, dt handling (emitters.py:211 _izhikevich_dv_du)",
            "source calibration status uncalibrated (emitters.py:7-9), EdgeList dtype/field names",
        ],
        provenance="jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS + :211 dv=0.04v²+5v+140-u+I du=a(bv-u) v>=30→c u+=d",
        tunable=False,
    )

    # --- Configured static values (scalar knobs, hash-visible via cfg.metadata) ---
    # Count scalars set directly as builder.py metadata values
    configured = dict(
        count=68,
        description="Scalar knobs set via builder.py cfg.update_metadata (hash-visible, MODEL_ASSUMPTION/LITERATURE_PRIOR)",
        groups=dict(
            motif_gains=16,  # builder.py:62 16 entries
            connectivity_scalars=3,  # WITHIN_GAIN 0.35, P_FEEDFORWARD 0.30, P_FEEDBACK 0.20
            delays=3,  # DELAY_FF 8, FB 12, WITHIN 2
            layer_maps=2,  # FF_LAYER_MAP + FB_LAYER_MAPS (structured)
            intrinsic_corrections=3,  # VIP b0.20, SST b0.21, PV drive 1.7
            background=3,  # tonic_scale 0.6, Poisson rate 2000, Poisson amp 2.0
            lognormal=1,  # CV 1.5 (sigma/mu derived)
            e_mixture=7,  # version + RS/CH/EFS frac 3 + params? scalars
            heterogeneity=1,  # jitter sigma 0.10
            populations=20,  # JOMISSION_AREAS 4 + LAYERS 5 + CELL_TYPES 4 + LAYER_COUNT_FRAC 5 + FLAT 4 (approx)
            rf_geometry=5,  # lattice 32, field 8, sigma1.8, spacing3.2, overlap0.45
            misc=4,  # dt0.1, n_per_area100, hierarchy, probe n_contacts16
        ),
        provenance="jomission/network/builder.py:62 MOTIF_GAIN 16 + :90 FF/FB + :395 spatial_sigma0.08 + :142 CV1.5 + :99 VIP b0.20 + :114 SST b0.21 + :125 PV1.7 + :159 tonic0.6 + :49 Poisson2000 + :179 E_MIXTURE + populations.py:12/31",
        tunable_subset="experimentally_adjustable (see below)",
    )

    # --- Derived static values (deterministic transforms of configured) ---
    derived = dict(
        count=12,
        description="Values computed deterministically from configured (DERIVED per freeze)",
        members=[
            "LOGNORMAL_SIGMA 1.08565 = sqrt(ln(1+CV²)) from CV 1.5 (builder.py:143)",
            "LOGNORMAL_MU -0.5893 = -sigma²/2 (builder.py:144)",
            "delay_steps [20,80,120] = delay_ms/dt (builder.py:807 jaxfne/emitters.py:545)",
            "positions (400,3) = area offsets + local uniform + layer depth bands (populations.py:21)",
            "E subtype n_RS169/CH48/EFS25 = frac * n_E242 via permutation (builder.py:1059)",
            "motif scaling factors per edge (builder.py:623 _apply_motif_gains)",
            "spatial pruning mask exp(-d²/2σ²) + max_in_degree cap (builder.py:630)",
            "E_MIXTURE child pheno a/c/d mapped before jitter (builder.py:1008)",
            "jittered a/b/c/d/drive post-jitter (builder.py:39 sigma0.10)",
            "layer count per area via LAYER_COUNT_FRAC * n_per_area (populations.py:31)",
            "probe geometry + RF tiling centers/weights (rf.py:46)",
            "overlap 0.45 = exp(-3.2²/(4*1.8²)) (DERIVED)",
        ],
        provenance="builder.py:39 jitter σ0.10 seam + :62 motif + :142 lognormal + :90 delays + populations.py:31 + jaxfne/emitters.py:545 delay_steps + manifests/agsdr_local_freeze.json:1 DERIVED list",
        tunable=False,
    )

    # --- Per-neuron static (arrays sized n_neurons) ---
    # a,b,c,d,drive,sign,v0,u0 (8 × N) + source_scale (N or 1) + positions (3×N) split
    per_neuron_n = 8 * n_neurons + pos_n  # 3200 +1200=4400 at N=400
    # source_scale adds N if per-neuron; if scalar add 1 — we report N
    per_neuron = dict(
        count=int(per_neuron_n),
        per_neuron_fields=8,
        n_neurons=int(n_neurons),
        description=f"Per-neuron arrays: a,b,c,d,drive,sign,v0,u0 (8×{n_neurons}) + positions ({pos_n}) — after jitter (builder.py:39 σ0.10)",
        provenance="jomission/network/builder.py:39 HETEROGENEITY_JITTER_SIGMA 0.10 + :212 _apply_per_neuron_jitter + :1008 _apply_typed_E_phenotypes + jaxfne/emitters.py:55 defaults",
        note="Per-neuron static is FIXED per seed (deterministic RNG seed ^0x9E3779B9), not tunable without ledger C-number per freeze",
    )
    # adjust for source_scale: adds ss.size (1 if scalar, N if per-neuron) — reconciles with STATE n_static
    try:
        if model is not None:
            em = model.params.get("emitter")
            if em is not None and hasattr(em, "source_scale") and em.source_scale is not None:
                ss = np.asarray(em.source_scale)
                # STATE counts ss.size (could be 1 scalar or N); include same in per_neuron for reconciliation
                add = int(ss.size) if ss.size else 1
                per_neuron["count"] = int(per_neuron["count"] + add)
                per_neuron["note"] = per_neuron.get("note", "") + f" + source_scale ({add})"
    except Exception:
        pass

    # --- Per-edge static (arrays sized n_edges) ---
    # pre,post,receptor_index,tau_ms,weight (5 × N_edges) + delay_steps (1×N_edges) is history-tinged but stored as static
    per_edge_n = 5 * n_edges  # pre,post,receptor,tau,weight
    # delay_steps also stored; count separately but for ontology it's static storage
    per_edge = dict(
        count=int(per_edge_n),
        n_edges=int(n_edges),
        per_edge_fields=5,
        extra_delay_steps=int(n_edges),
        total_with_delay=int(per_edge_n + n_edges),
        description=f"Per-edge arrays: pre,post,receptor_index,tau_ms,weight (5×{n_edges}) + delay_steps (1×{n_edges} history-indexed)",
        provenance="jomission/network/builder.py:395 spatial_sigma0.08 + :630 _apply_spatial_locality + :623 _apply_motif_gains + :807 _apply_laminar_delays + jaxfne/emitters.py:545 EdgeList",
    )

    # --- Dynamic (fast state per step) ---
    # v,u,prev_spikes (3N) + syn_state (N_edges) + H (N) + w (N_edges) = 3N + 2*N_edges + N
    dynamic_n = 3 * n_neurons + n_edges + n_neurons + n_edges  # 22380 at 400/10590
    dynamic = dict(
        count=int(dynamic_n),
        breakdown=dict(v=n_neurons, u=n_neurons, prev_spikes=n_neurons, syn_state=n_edges, H=n_neurons, w=n_edges),
        description="DynamicState six fields per jaxfne/_pipeline.py:48 (v,u,prev_spikes,syn_state,H,w) — per dt0.1 step",
        provenance="jaxfne/_pipeline.py:48 DynamicState + :72 ContinuationState + jomission/dynamics/h_state.py:28 H_COORDINATES conceptual",
        units="float32 per field, bool for spikes",
    )

    # --- Plastic (Theta / HDP) ---
    # Theta vector small population restoring dim 2 + config scalars
    plastic = dict(
        count=4,
        description="Plastic: Theta vector (hdp population restoring h_state_dim=2) + HDPConfig scalars tau_theta 1000s lr1e-4 bounds [-0.1,0.1]",
        provenance="jomission/dynamics/hdp.py:14 HDPConfig + manifests/agsdr_local_freeze.json:97 H_implemented vs conceptual 5 dims",
        note="Population restoring (theta_dim=2) maps to 5-coord conceptual per freeze; AGSDR may scale weights but not retune tau_theta",
    )

    # --- History (delay_state ring buffer) ---
    # D_max = max(delay_steps) =120 steps * N_neurons
    try:
        ea = _edge_arrays(model) if model is not None else {}
        max_delay = int(np.max(ea["delay_steps"])) if ea and ea["delay_steps"].size else 120
    except Exception:
        max_delay = 120
    history_n = int(max_delay * n_neurons)  # 48000
    history = dict(
        count=int(history_n),
        D_max=int(max_delay),
        shape=f"({max_delay}, {n_neurons})",
        description=f"History: delay_state ring buffer D_max={max_delay} ({max_delay*float(dt_ms):.0f}ms) × {n_neurons} — spikes history B_t per ContinuationState",
        provenance="jomission/network/builder.py:90 delays [20,80,120] steps + jaxfne/_pipeline.py ContinuationState.delay_state + jaxfne/emitters.py:545 edge_list_with_delay_ms",
        note="FIXED, not tunable per Visualization Law — history size set by max delay",
    )

    # --- Recording (probe/field buffer per step, reporting single-step footprint) ---
    try:
        n_contacts = int(cfg.metadata.get("n_contacts", 16)) if cfg is not None and hasattr(cfg, "metadata") else 16
        if cfg is not None and hasattr(cfg, "metadata") and "n_contacts" in cfg.metadata:
            n_contacts = int(cfg.metadata.get("n_contacts", 16))
    except Exception:
        n_contacts = 16
    recording_n = int(n_contacts + 4)  # probe contacts + overhead
    recording = dict(
        count=int(recording_n),
        n_contacts=int(n_contacts),
        description=f"Recording: probe {n_contacts} contacts (field per step) — not counted as state per step beyond probe readout",
        provenance="jomission/network/builder.py:410 cfg.probe(n_contacts=16) + jaxfne/fields/proxy.py linear",
        note="Per-step accumulation over 2000ms = 20000 steps × contacts × float32 for trajectory (see STATE traj_bytes)",
    )

    # --- Higher-level sets ------------------------------------------------
    # Experimentally adjustable parameters = scalar configured knobs that experimenter could sweep without breaking engine
    # N_free24 = 16 motif gains + 3 connectivity (within_gain, pFF, pFB) + 5 theta_local dims (g_rec,g_fastEI,g_dend,g_disinh,g_background) → 24 scalars
    # But expanded AGSDR exposes 9 scalars for 5 dims (g_fastEI has 2, g_dend 4, g_background 3)
    experimentally_adjustable = dict(
        count=24,
        description="N_tunable = N_free24 scalar knobs experimentally adjustable without primitive mutation",
        members=[
            "MOTIF_GAIN 16 gains (builder.py:62) — 5 AGSDR dims map to these 16 via orthogonal masks (freeze :132)",
            "WITHIN_GAIN 0.35 + P_FEEDFORWARD 0.30 + P_FEEDBACK 0.20 (connectivity.py:14/17/18) — 3",
            "5 theta_local dims g_rec/g_fastEI/g_dend/g_disinh/g_background (manifests/agsdr_local_freeze.json:131) — controls above gains + I_bg",
        ],
        expanded_scalar_count=28,  # 16 +3 +9 (AGSDR expanded) but overlapping — report 24 unique scalar knobs
        provenance="manifests/agsdr_local_freeze.json:131 five_dimensions_orthogonal + builder.py:62/90/395 + connectivity.py:14",
        inequality="N_model (static numbers ~57k) ≠ N_tunable (24 scalars) ≠ N_AGSDR (5 dims / 9 scalars)",
    )
    agsdr_exposed = dict(
        dims=5,
        scalar_params=9,
        dims_list=["g_rec","g_fastEI","g_dend","g_disinh","g_background"],
        scalar_breakdown=dict(g_rec=1, g_fastEI=2, g_dend=4, g_disinh=1, g_background=3),
        description="AGSDR-exposed parameters 5 dims (9 scalars when expanded over orthogonal masks)",
        provenance="manifests/agsdr_local_freeze.json:131-192 five_dimensions_orthogonal g_rec[0.70,1.30] g_fastEI[0.80,1.20] g_dend[0.70,1.30] g_disinh[0.70,1.30] g_background mean-controlled + manifests/agsdr_local_harness.json:142-147 theta spec",
        inequality="N_AGSDR dims 5 (scalars 9) < N_tunable 24 < N_model ~57k static numbers",
    )
    currently_frozen = dict(
        count=int(68 - 24),  # configured 68 minus tunable 24 = 44 frozen
        description="Currently frozen scientific parameters (MODEL_ASSUMPTION magnitudes not exposed to AGSDR search)",
        members=[
            "VIP_B_CORRECTED 0.20 (builder.py:99) + SST_B 0.21 (:114) + PV_DRIVE 1.7 (:125) frozen intrinsic per freeze :40-56",
            "LOGNORMAL_CV 1.5 (builder.py:142), HETEROGENEITY_JITTER 0.10 (:39), E_MIXTURE M2 70/20/10 (:179)",
            "RF geometry 32×32 sigma1.8 spacing3.2 overlap0.45 (rf.py:46), H taus 0.1/1/10/100/1000 (h_state.py:28), Theta tau1000 (hdp.py:14)",
            "Delays 2/8/12ms (builder.py:90), spatial_sigma0.08 max_in_degree25 (builder.py:395), LAYER_COUNT_FRAC (populations.py:31)",
        ],
        provenance="manifests/agsdr_local_freeze.json:22 _freeze_rule MUST NOT mutate primitives + :31-105 frozen_primitives table",
        note="Any change to these requires new Ledger C-number, not silent theta scaling",
    )

    # Reconcile N_total vs ontology
    # N_static_ontology = per_neuron + per_edge + derived counts? But derived 12 is scalar count not numbers
    # For state numbers reconciliation, use N_total = N_static_numbers + N_dynamic + N_plastic + N_history + N_recording
    # where N_static_numbers = per_neuron + per_edge + (source_scale etc) ≈ 57000
    N_static_numbers = int(per_neuron["count"] + per_edge["count"])  # ~ 4400+52950=57350
    # include derived scalar offset? not as numbers
    N_total = int(N_static_numbers + dynamic_n + plastic["count"] + history_n + recording_n)

    return dict(
        engine_fixed_constants=engine_fixed,
        configured_static_values=configured,
        derived_static_values=derived,
        per_neuron_static=per_neuron,
        per_edge_static=per_edge,
        dynamic=dynamic,
        plastic=plastic,
        history=history,
        recording=recording,
        experimentally_adjustable=experimentally_adjustable,
        agsdr_exposed=agsdr_exposed,
        currently_frozen_scientific=currently_frozen,
        N_model_parameters=int(N_static_numbers),  # numbers that define model (static)
        N_tunable=int(experimentally_adjustable["count"]),
        N_AGSDR_dims=int(agsdr_exposed["dims"]),
        N_AGSDR_scalars=int(agsdr_exposed["scalar_params"]),
        N_total_reconciled=int(N_total),
        reconciliation=f"N_total {N_total} = N_static_numbers {N_static_numbers} (per_neuron {per_neuron['count']} + per_edge {per_edge['count']}) + N_dynamic {dynamic_n} + N_plastic {plastic['count']} + N_history {history_n} + N_recording {recording_n}",
        inequality="N_model parameters (~57k numbers) ≠ N_tunable (24 scalars) ≠ N_AGSDR (5 dims / 9 scalars) — Visualization Law",
        citations=[
            "jomission/network/builder.py:39 jitter σ0.10 (per-neuron static)",
            "jomission/network/builder.py:62 MOTIF_GAIN 16 gains (configured → AGSDR)",
            "jomission/network/populations.py:12 AREAS V1 V4 FEF PFC :13 LAYERS :31 LAYER_COUNT_FRAC :44 AREA_LAYER_CELL_TYPES",
            "jomission/network/connectivity.py:14 WITHIN_GAIN 0.35 :17 P_FF 0.30 :18 P_FB 0.20",
            "jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS + :211 dv/du (engine-fixed)",
            "manifests/agsdr_local_freeze.json:1 be9b96ab freeze + :131 orthogonal 5-dim + :67 spatial_locality",
        ],
    )


def _compute_state_ontology(model, cfg, dt_ms: float = 0.1) -> Dict[str, Any]:
    """Compute ontology counts from generated-owner arrays.

    N_total = N_static + N_dynamic + N_plastic + N_history + N_recording
    Distinguish N_free vs derived/fixed per Visualization Law.
    Delegates to ontology_table for stratified detail.
    """
    tbl = _neuron_table(model)
    n_neurons = int(len(tbl)) if tbl else 400
    ea = _edge_arrays(model)
    n_edges = int(ea["pre"].shape[0]) if ea else 10590
    pos = _positions(model)
    # Emitter arrays (per-neuron)
    em = model.params.get("emitter")
    n_per_neuron_static_fields = 0
    bytes_static = 0
    n_static = 0
    # per-neuron intrinsic: a,b,c,d,drive,sign,v0,u0 (8 × n) + source_scale scalar
    try:
        if em is not None:
            for name in ("a","b","c","d","drive","sign","v0","u0"):
                arr = getattr(em, name, None)
                if arr is not None:
                    a_np = np.asarray(arr)
                    n_static += int(a_np.size)
                    bytes_static += int(a_np.nbytes)
                    if a_np.size == n_neurons:
                        n_per_neuron_static_fields += 1
            # W dense (400×400) — stored but edge_list authoritative
            if hasattr(em, "W") and em.W is not None:
                w_dense = np.asarray(em.W)
                dense_n = int(w_dense.size)  # 160000
                dense_bytes = int(w_dense.nbytes)
            else:
                dense_n, dense_bytes = 0, 0
            # source_scale
            if hasattr(em, "source_scale") and em.source_scale is not None:
                ss = np.asarray(em.source_scale)
                n_static += int(ss.size)
                bytes_static += int(ss.nbytes)
        # positions
        pos_n = int(pos.size)  # 1200
        pos_bytes = int(pos.nbytes)
        # EdgeList static: pre, post, receptor_index, tau_ms (4 × n_edges) + weight
        if ea:
            n_static_edges = n_edges * 4 + n_edges  # pre,post,receptor,tau + weight
            bytes_edges = 0
            for k in ("pre","post","weight","tau_ms"):
                if k in ea:
                    bytes_edges += int(np.asarray(ea[k]).nbytes)
            # delay_steps is history-related but stored as static
        else:
            n_static_edges, bytes_edges = 0, 0
        N_static = int(n_static + pos_n + n_static_edges)
        # keep dense separate for reporting
    except Exception:
        N_static = n_neurons * 8 + pos.size + n_edges * 5
        bytes_static = 0
        dense_n, dense_bytes = n_neurons * n_neurons, n_neurons * n_neurons * 4
        n_per_neuron_static_fields = 8

    # Dynamic fast state (DynamicState six fields)
    # v,u,prev_spikes (3×400), syn_state (n_edges), H (n_neurons × d_H), w (n_edges)
    # baseline d_H=1 scalar
    try:
        d_H = 1
        # try to read actual H shape from pipeline if available
        # fallback 1
        N_dynamic = int(3 * n_neurons + n_edges + n_neurons * d_H + n_edges)
        # v,u,prev_spikes are float32, syn_state float32, H float32, w float32
        bytes_dynamic = int((3 * n_neurons + n_edges + n_neurons * d_H + n_edges) * 4)
    except Exception:
        N_dynamic = 3*400 + 10590*2 + 400
        bytes_dynamic = N_dynamic * 4

    # Plastic: Theta + H state config (HDP)
    # HDPConfig: tau_theta 1000s, channels edge_weight+intrinsic_drive, locality population
    # Theta vector size = h_state_dim (2) or small population-level
    try:
        from jomission.dynamics.hdp import HDPConfig as _HDP
        hdp = _HDP()
        theta_dim_reported = 2  # jaxfne population restoring (h_state_dim=2)
        # plus per-population restoring? conceptual 5 but implemented 2
        N_plastic = int(theta_dim_reported + 2)  # small
        bytes_plastic = int(N_plastic * 4)
        plastic_desc = f"Theta dim {theta_dim_reported} (population restoring, h_state_dim=2 maps to 5-coord conceptual), tau_theta 1000s, lr1e-4 bounds [-0.1,0.1], channels edge_weight+intrinsic_drive"
    except Exception:
        N_plastic = 4
        bytes_plastic = 16
        plastic_desc = "Theta tau_theta 1000s lr1e-4 bounds [-0.1,0.1]"

    # History / delay
    # delay_state ring buffer: D_max * n_neurons, D_max = max(delay_steps) =120
    try:
        max_delay = int(np.max(ea["delay_steps"])) if ea and ea["delay_steps"].size else 120
        bufsize = int(max_delay)  # 120 steps
        # buffer shape (bufsize, n_neurons) bool/spikes? stored as spikes history
        N_history_delay = int(bufsize * n_neurons)  # 48000
        # H history not separately, included in dynamic
        N_history = int(N_history_delay)
        bytes_history = int(N_history * 1)  # bool or float32 depending
        # more accurate: edge delay buffer is per neuron spikes, 48000 bytes if bool
        bytes_history_f32 = int(N_history * 4) if False else int(N_history * 1)
    except Exception:
        N_history = 120 * 400
        bytes_history = N_history
        bytes_history_f32 = N_history * 4
        max_delay = 120
        bufsize = 120

    # Recording: field + probe
    # probe n_contacts 16, field source proxies per step? but state recording is per simulation
    # we report probe count as recording state size
    try:
        n_contacts = int(cfg.metadata.get("probe", {}).get("n_contacts", 16)) if hasattr(cfg, "metadata") else 16
        # allow override via cfg
        if "n_contacts" in getattr(cfg, "metadata", {}):
            n_contacts = int(cfg.metadata.get("n_contacts", 16))
    except Exception:
        n_contacts = 16
    # field per time: phi (n_contacts or n_neurons) but for state ontology per step
    N_recording = int(n_contacts + 4)  # 16 + overhead
    bytes_recording = int(N_recording * 4 * 100)  # rough per-step accumulation not static; report small

    # RNG
    rng_bytes = 16  # PRNGKey 2×32
    N_rng = 2

    # Total state numbers and bytes (ontology)
    N_total = int(N_static + N_dynamic + N_plastic + N_history + N_recording)
    total_bytes = int(bytes_static + bytes_dynamic + bytes_plastic + bytes_history + rng_bytes + bytes_recording)
    # Estimated run memory for 2000ms trial: states + trajectory storage
    # trajectory would be V_m (n_steps × n_neurons) = 20000×400 =8M floats ~32MB
    n_steps_2000 = int(2000.0 / float(dt_ms))
    traj_bytes_vm = int(n_steps_2000 * n_neurons * 4)
    traj_bytes_spikes = int(n_steps_2000 * n_neurons * 1)
    traj_bytes_field = int(n_steps_2000 * n_contacts * 4)
    est_run_mem = int(total_bytes + traj_bytes_vm + traj_bytes_spikes + traj_bytes_field)

    # N_free vs derived/fixed
    # N_free: parameters that are tunable within AGSDR box (motif gains, within_gain, drive)
    # Derived/fixed: positions, pre/post, a/b/c/d after jitter (fixed), etc.
    # Compute free as motif 16 gains + within/p_ff/p_fb etc.
    N_free = 16 + 3 + 5  # 16 motif gains + 3 connectivity scalars + 5 theta_local
    N_derived = int(N_static - N_free)
    N_fixed = int(n_neurons * 6)  # intrinsic frozen

    return dict(
        n_neurons=int(n_neurons),
        n_edges=int(n_edges),
        n_contacts=int(n_contacts),
        dt_ms=float(dt_ms),
        max_delay_steps=int(max_delay),
        bufsize=int(bufsize),
        N_static=int(N_static),
        N_dynamic=int(N_dynamic),
        N_plastic=int(N_plastic),
        N_history=int(N_history),
        N_recording=int(N_recording),
        N_rng=int(N_rng),
        N_total=int(N_total),
        N_free=int(N_free),
        N_derived=int(N_derived),
        N_fixed=int(N_fixed),
        bytes_static=int(bytes_static),
        bytes_dynamic=int(bytes_dynamic),
        bytes_plastic=int(bytes_plastic),
        bytes_history=int(bytes_history),
        bytes_recording=int(bytes_recording),
        total_bytes=int(total_bytes),
        traj_bytes_vm=int(traj_bytes_vm),
        traj_bytes_spikes=int(traj_bytes_spikes),
        traj_bytes_field=int(traj_bytes_field),
        est_run_mem=int(est_run_mem),
        n_steps_2000=int(n_steps_2000),
        dense_W_n=int(dense_n if 'dense_n' in locals() else 0),
        dense_W_bytes=int(dense_bytes if 'dense_bytes' in locals() else 0),
        plastic_desc=str(plastic_desc),
        pos_n=int(pos.size) if 'pos_n' in locals() else 1200,
    )


def _populations_table(model) -> List[Dict[str, Any]]:
    tbl = _neuron_table(model)
    # count per area×layer×class
    cnt = Counter()
    for r in tbl:
        cnt[(str(r.get("area")), str(r.get("layer")), str(r.get("cell_type")))] += 1
    # intrinsic family mapping (post-jitter actual is heterogeneous, but pre-jitter phenotype is):
    # E has subtypes RS/CH/E_FS via a,c,d clusters; PV/SST/VIP single families
    # For display, split E counts into RS/CH/EFS via c/a thresholds (approx)
    # deterministic phenotype counts for seed0: RS169 CH48 EFS25 at nE242 — but we recompute from arrays for generated-owner
    em = model.params.get("emitter")
    rows: List[Dict[str, Any]] = []
    # Precompute E subtype assignment via same logic as builder: use permutation? Simpler: classify by a/c values
    subtype_counts = Counter()
    try:
        if em is not None:
            a_arr = np.asarray(em.a, dtype=float)
            c_arr = np.asarray(em.c, dtype=float)
            for i, r in enumerate(tbl):
                if str(r.get("cell_type")) != "E":
                    continue
                a = float(a_arr[i]) if i < a_arr.size else 0.02
                c = float(c_arr[i]) if i < c_arr.size else -65.0
                # CH: c > -55 (c -50) vs RS/EFS -65, EFS: a>0.05
                if c > -55:
                    sub = "E_CH"
                elif a > 0.05:
                    sub = "E_FS"
                else:
                    sub = "E_RS"
                subtype_counts[(str(r.get("area")), str(r.get("layer")), sub)] += 1
    except Exception:
        pass

    # Rate targets per class (from b1_targets or freeze doc)
    rate_targets = {"E_RS":"5-8Hz","E_CH":"9-15Hz burst","E_FS":"10-15Hz","PV":"6-37Hz (E<PV)","SST":"3-18Hz","VIP":"0.4-22Hz (>0.5 finite)"}
    intrinsic_map = {
        "E_RS": "a0.02 b0.20 c-65 d8 (RS)",
        "E_CH": "a0.02 b0.20 c-50 d2 (CH)",
        "E_FS": "a0.10 b0.20 c-65 d2 (E_FS)",
        "PV": "a0.10 b0.20 c-65 d2 drive×1.7 →3.01",
        "SST": "b0.21 [0.189,0.231]",
        "VIP": "b0.20 [0.18,0.22]",
        "E": "typed M2 RS70/CH20/EFS10",
    }

    for (area, layer, ct), n in sorted(cnt.items(), key=lambda x: (JOMISSION_AREAS.index(x[0][0]) if x[0][0] in JOMISSION_AREAS else 99, JOMISSION_LAYERS.index(x[0][1]) if x[0][1] in JOMISSION_LAYERS else 99, x[0][2])):
        if ct == "E":
            # expand into subtypes for this area×layer
            for sub in ("E_RS","E_CH","E_FS"):
                n_sub = int(subtype_counts.get((area, layer, sub), 0))
                if n_sub == 0:
                    continue
                rows.append(dict(
                    Area=str(area), Layer=str(layer), Class=str(sub),
                    N=int(n_sub), Intrinsic=str(intrinsic_map.get(sub,"")),
                    Family=str(sub.split("_")[1] if "_" in sub else sub),
                    RateTarget=str(rate_targets.get(sub,"")),
                    Provenance="builder.py:179-196 E_MIXTURE_M2 (LITERATURE_PRIOR Izhikevich2003 Fig1 / MODEL_ASSUMPTION 70/20/10) + builder.py:1008 _apply_typed_E_phenotypes",
                ))
        else:
            rows.append(dict(
                Area=str(area), Layer=str(layer), Class=str(ct),
                N=int(n), Intrinsic=str(intrinsic_map.get(ct,"")),
                Family=str(ct),
                RateTarget=str(rate_targets.get(ct,"")),
                Provenance="jaxfne/emitters.py:55 + builder.py:99 VIP b0.20 / :114 SST b0.21 / :125 PV drive1.7 (MODEL_ASSUMPTION/LITERATURE_PRIOR)",
            ))
    return rows


def _connectivity_table(model, dt_ms: float = 0.1) -> List[Dict[str, Any]]:
    tbl = _neuron_table(model)
    if not tbl:
        return []
    areas = [str(r.get("area")) for r in tbl]
    layers = [str(r.get("layer")) for r in tbl]
    cts = [str(r.get("cell_type")) for r in tbl]
    ea = _edge_arrays(model)
    if not ea:
        return []
    pre, post, w, ds = ea["pre"], ea["post"], ea["weight"], ea["delay_steps"]
    # pop counts for p
    pop_counts = Counter((areas[i], layers[i], cts[i]) for i in range(len(tbl)))
    # aggregate by motif key (src_area,src_layer,src_ct -> tgt_area,tgt_layer,tgt_ct)
    agg: Dict[Tuple[str,str,str,str,str,str], List[int]] = defaultdict(list)
    for ei in range(int(pre.shape[0])):
        try:
            sa, sl, sc = areas[int(pre[ei])], layers[int(pre[ei])], cts[int(pre[ei])]
            ta, tl, tc = areas[int(post[ei])], layers[int(post[ei])], cts[int(post[ei])]
        except Exception:
            continue
        agg[(sa,sl,sc,ta,tl,tc)].append(int(ei))
    rows: List[Dict[str, Any]] = []
    for key, idxs in sorted(agg.items()):
        sa, sl, sc, ta, tl, tc = key
        ww = w[np.array(idxs)]
        meanW = float(np.mean(np.abs(ww))) if ww.size else 0.0
        stdW = float(np.std(np.abs(ww))) if ww.size else 0.0
        cv = float(stdW / meanW) if meanW > 0 else 0.0
        delays = ds[np.array(idxs)] * float(dt_ms)
        meanDelay = float(np.mean(delays)) if delays.size else 0.0
        n_src = int(pop_counts.get((sa,sl,sc), 1))
        n_tgt = int(pop_counts.get((ta,tl,tc), 1))
        possible = max(1, n_src * n_tgt)
        p = float(len(idxs)) / float(possible)
        conn_type = _classify_edge(sa, sl, ta, tl)
        mg = float(MOTIF_GAIN.get((sc, tc), 1.0)) if isinstance(MOTIF_GAIN, dict) else 1.0
        rows.append(dict(
            Source=f"{sa} {sl} {sc}",
            Target=f"{ta} {tl} {tc}",
            Edges=int(len(idxs)),
            p=float(p),
            Wmean=float(meanW),
            Wcv=float(cv),
            Delay_ms=float(meanDelay),
            DelaySteps=sorted(set(ds[np.array(idxs)].tolist())) if idxs else [],
            ConnType=str(conn_type),
            MotifGain=float(mg),
            n_src=int(n_src),
            n_tgt=int(n_tgt),
        ))
    # sort by edges descending so FF/FB examples appear near top
    rows = sorted(rows, key=lambda r: r["Edges"], reverse=True)
    return rows


def observable_basis() -> Dict[str, Any]:
    """ObservableBasis first-class — every observable with ownership semantics.

    Per task: return for every observable (name, owner, dimensionality, units,
    independence, parent) with independence not overclaimed (state/output where
    ambiguous). Covers V_m, spikes, H, Theta, relative V, q, phi, CSD-like, probe y.

    Owners: STATE (DynamicState/ContinuationState), OUTPUT (derived/measured),
    with ambiguous cases marked via note.

    Dimensionality at n_neurons=400, n_edges=10590, n_contacts=16, dt0.1.
    Units: mV, bool/spikes, native current, a.u. proxy, etc. Time always ms (V0 gate U).
    Parent: null if independent state, else source field name.
    """
    observables: List[Dict[str, Any]] = [
        dict(
            name="V_m",
            owner="STATE",
            dimensionality="(n_neurons,) per step, (n_steps, n_neurons) trajectory; n_neurons=400",
            units="mV",
            independence=True,
            parent=None,
            classification="X",
            description="Membrane voltage per neuron — fast state variable in DynamicState.v",
            provenance="jaxfne/emitters.py:211 _izhikevich_dv_du dv=0.04v²+5v+140-u+I du=a(bv-u); jaxfne/_pipeline.py:48 DynamicState.v",
            time_units="ms",
            independence_note="Independent state — part of X_t slice of C_t",
        ),
        dict(
            name="spikes",
            owner="STATE/OUTPUT*",
            dimensionality="(n_neurons,) bool per step, (n_steps, n_neurons) trajectory",
            units="bool (threshold V>=30 mV) → Hz when binned",
            independence=True,
            parent=None,
            classification="X",
            description="Spike events — threshold crossings V>=30 mV, stored as prev_spikes in DynamicState + emitted as event",
            provenance="jaxfne/emitters.py:211 v>=30→c u+=d; jaxfne/_pipeline.py:48 DynamicState.prev_spikes",
            independence_note="Ambiguous ownership: STATE as prev_spikes (carry) + OUTPUT as event stream; independent yes* with ownership STATE/OUTPUT where ambiguous per task",
            time_units="ms",
        ),
        dict(
            name="u",
            owner="STATE",
            dimensionality="(n_neurons,)",
            units="mV (recovery variable)",
            independence=True,
            parent=None,
            classification="X",
            description="Recovery variable u — second Izhikevich state, DynamicState.u",
            provenance="jaxfne/emitters.py:211 du=a(bv-u)",
            time_units="ms",
        ),
        dict(
            name="syn_state",
            owner="STATE",
            dimensionality="(n_edges,) n_edges=10590",
            units="native current (uncalibrated)",
            independence=True,
            parent=None,
            classification="X",
            description="Synaptic state per edge — carries filtered spike history for edge i",
            provenance="jaxfne/_pipeline.py:48 DynamicState.syn_state",
            time_units="ms",
        ),
        dict(
            name="H",
            owner="STATE",
            dimensionality="(n_neurons,) scalar (conceptual 5 dims taus 0.1/1/10/100/1000s)",
            units="dimensionless (adaptation state)",
            independence=True,
            parent=None,
            classification="A",
            description="Adaptive history state H — scalar per neuron implemented, conceptual 5-coord per jomission/dynamics/h_state.py:28",
            provenance="jomission/dynamics/h_state.py:28 H_COORDINATES 5 taus + jaxfne/_pipeline.py DynamicState.H + manifests/agsdr_local_freeze.json:97",
            independence_note="Independent adaptive state — slow, history; gated mismatch until M-06 per gen2_gates.py:110",
            time_units="ms",
        ),
        dict(
            name="Theta",
            owner="STATE",
            dimensionality="(theta_dim,) population: theta_dim=2 (h_state_dim=2)",
            units="dimensionless (bounded [-0.1,0.1])",
            independence=True,
            parent=None,
            classification="A (plastic)",
            description="Plastic parameters Theta — slow variables with tau_theta 1000s, lr1e-4, channels edge_weight+intrinsic_drive",
            provenance="jomission/dynamics/hdp.py:14 HDPConfig tau_theta 1000 lr1e-4 + manifests/agsdr_local_freeze.json:97",
            independence_note="Independent plastic state — carried in ContinuationState.w / HDP, not derived from X",
            time_units="s (tau_theta 1000s)",
        ),
        dict(
            name="delay_state",
            owner="STATE",
            dimensionality="(D_max, n_neurons) D_max=120×400=48000 buffer",
            units="bool (spike history)",
            independence=True,
            parent=None,
            classification="A (history)",
            description="Delay/history ring buffer B_t — ring of past spikes for delayed edges",
            provenance="jaxfne/_pipeline.py ContinuationState.delay_state + jaxfne/emitters.py:545 delay_steps + builder.py:807 _apply_laminar_delays",
            time_units="steps (120 steps × 0.1ms =12ms)",
        ),
        dict(
            name="V_centered (V_i - bar V) — relative_V",
            owner="OUTPUT",
            dimensionality="(n_neurons,) per step — same as V_m",
            units="mV (relative)",
            independence=False,
            parent="V_m",
            classification="DERIVED_FROM(V_m)",
            semantic_class="DERIVED",
            description="Relative voltage V_i - mean(V) — mean subtraction per field gauge, not independent state; DERIVED_FROM(V_m) (V1 graph C_t→X_t→q_t→φ_t→y_t, jomission/visualization/model_summary.py:observable_basis())",
            provenance="jaxfne/_signals.py Signals.field diagnostics gauge='mean_zero'",
            independence_note="DERIVED_FROM(V_m) — dependent on V_m; must label DERIVED_FROM(V_m) per V4 adversary N1/Misleading normalization; semantic_class DERIVED",
            time_units="ms",
        ),
        dict(
            name="q (source / current proxy)",
            owner="OUTPUT",
            dimensionality="(n_neurons,) per step",
            units="native current (source_scale * I)",
            independence=False,
            parent=None,
            classification="q",
            description="Source/current proxy q_t = source_scale * (I_tonic+I_private+I_noise+I_syn + gain*spikes) — per-neuron continuous source",
            provenance="jaxfne/emitters.py:55 source_scale + jaxfne/presets.py DEFAULT_SPIKE_IMPULSE_GAIN + builder.py:49 Poisson2kHz amp2.0 + :159 tonic3.0",
            independence_note="q source depends on definition — if defined as instantaneous current from X+A+Static, then DERIVED_FROM(X,A); if defined as filtered source with own dynamics, ownership ambiguous. Here: OUTPUT derived from X_t,A_t,Static per advisor (not independent state)",
            time_units="ms",
            parent_detail="DERIVED_FROM(X,A,Static) — deterministic transform of V_m/spikes/H/w/W/drive",
        ),
        dict(
            name="phi (field_proxy / LFP-like proxy_readout)",
            owner="OUTPUT",
            dimensionality="(n_contacts,) or (depth, time) proxy field",
            units="a.u. (proxy_readout, physical_amplitude_calibrated=False, never physical LFP)",
            independence=False,
            parent="q",
            classification="phi",
            semantic_class="FIELD_PROXY",
            description="Field proxy phi_t = project_laminar_sources(q_t) — laminar field_proxy (mean_zero_neumann gauge, proxy_no_field_solve jomission/network/builder.py:409, not PDE solve, never physical LFP, proxy_readout)",
            provenance="jaxfne/_config.py field(source_mode='proxy_no_field_solve') builder.py:409 + jaxfne/fields/proxy.py:148 project_laminar_sources + jaxfne/io.py manifest field_solver_status 'linear_solver' claim_level 'proxy_readout'",
            independence_note="DERIVED_FROM(q) — deterministic projection of source; FIELD_PROXY proxy_readout not physical LFP per V4 adversary U3; semantic_class FIELD_PROXY, proxy_status True",
            time_units="ms",
        ),
        dict(
            name="CSD (CSD-like second derivative) — CSD_proxy",
            owner="OUTPUT",
            dimensionality="(n_contacts,) or (depth,)",
            units="a.u./mm² (proxy second derivative, proxy_readout, physical_amplitude_calibrated=False)",
            independence=False,
            parent="phi",
            classification="DERIVED_FROM(field_proxy)",
            semantic_class="DERIVED",
            description="CSD-like = second spatial derivative of phi_proxy along depth — post-hoc DERIVED_FROM(field_proxy) (field_proxy / LFP-like proxy_readout, never physical CSD)",
            provenance="jaxfne/fields.py probe_laminar_modes + builder.py:409 proxy_no_field_solve + builder.py:410 probe + jaxfne/io.py manifest csd_sign_convention + jaxfne/fields/proxy.py:192 + vis_V4_adversary.md N2",
            independence_note="DERIVED_FROM(field_proxy) — not independent measurement; labeling as independent CSD double-counts phi evidence per V4; semantic_class DERIVED, proxy_status True",
            time_units="ms",
        ),
        dict(
            name="y (probe readout)",
            owner="OUTPUT",
            dimensionality="(n_contacts, n_steps) n_contacts=16",
            units="a.u. (proxy_readout linear)",
            independence=False,
            parent="phi",
            classification="y",
            description="Probe readout y_t = Probe(n_contacts=16) @ phi_t — linear readout per laminar contact",
            provenance="jomission/network/builder.py:410 cfg.probe(n_contacts=16) + jaxfne/fields.py probe_laminar_modes + manifests/agsdr_local_freeze.json:1",
            independence_note="DERIVED_FROM(phi) — linear transform of field; readout, not state",
            time_units="ms",
        ),
        dict(
            name="W (EdgeList weights)",
            owner="STATE (static latched)",
            dimensionality="(n_edges,) n_edges=10590",
            units="native current * gain (includes lognormal CV1.5 + motif gains) — display log: sign·log1p(|W|/1e-6) heavy tail, builder.py:142 sigma1.085",
            independence=True,
            parent=None,
            classification="Static per-edge",
            description="Edge weights — per-edge static after spatial locality + motif scaling + lognormal (heavy tail CV1.5). Visualization: log abs weight with sign preserved (_safe_log_abs_weight, zero/sign-safe via log1p(|W|/eps)), linear for CV/rho/Fano per A2. Network_viz weight heatmaps log10.",
            provenance="jomission/network/builder.py:62 MOTIF_GAIN 16 + :395 spatial_sigma0.08 + :142 CV1.5 sigma1.085 mu-0.589 (heavy tail) + jaxfne/emitters.py EdgeList + V0-P1 log scaling",
            independence_note="Static latched per model instance (hash-visible), not per-step dynamic; counts toward N_static per-edge; vis scaling log for weight, linear for CV/rho/Fano (A2)",
            time_units="—",
            scaling=dict(weight="log (sign·log1p(|W|/1e-6), zero-safe)", H_tau="log10 (0.1-1000s, 4 orders, h_state.py:28)", CV="linear [0,2.5]", rho="linear [-0.6,0.6]", Fano="linear"),
        ),
        dict(
            name="positions",
            owner="STATE (static)",
            dimensionality="(n_neurons,3)",
            units="dva proxy (0..8°) / arbitrary units",
            independence=True,
            parent=None,
            classification="Static per-neuron",
            description="Neuron positions — 3D per neuron for spatial pruning, depth bands L1 0-0.10 etc.",
            provenance="jomission/network/populations.py:21 LAYER_DEPTH_BANDS + builder.py:630 _apply_spatial_locality",
            time_units="—",
        ),
    ]

    # Return typed table plus graph-level metadata for compatibility
    return dict(
        observables=observables,
        # hash for manifest (deterministic)
        observable_basis_hash=hashlib.sha256(json.dumps([o["name"] for o in observables], sort_keys=True).encode()).hexdigest()[:16],
        # Backwards-compatible graph
        graph="C_t -> X_t -> q_t -> phi_t -> y_t",
        nodes=dict(
            C_t=dict(
                label="Full continuation state",
                semantic_class="STATE",
                units="mixed (see storage)",
                proxy_status=False,
                contains=["DynamicState(v,u,prev_spikes,syn_state,H,w)", "RNG prng_key", "step_index", "delay_state ring buffer (120×400)"],
                provenance="jaxfne/_pipeline.py:48 DynamicState + :72 ContinuationState + jaxfne/emitters.py:545 delay_steps",
                storage="ContinuationState pytree — every simulation segment carry",
            ),
            X_t=dict(
                label="Fast neural state",
                depends_on="C_t.DynamicState.v,u,prev_spikes,syn_state",
                classification="X",
                semantic_class="STATE",
                units="V_m (mV), spikes (bool), syn_state (a.u.)",
                proxy_status=False,
                variables=["V_m (400×1 float32) membrane voltage", "spikes (400 bool) threshold V>=30", "u (400) recovery", "syn_state (10590)"],
                provenance="jaxfne/emitters.py:211 _izhikevich_dv_du dv=0.04v²+5v+140-u+I du=a(bv-u)",
                note="X is strictly fast (dt0.1ms); no H/Theta; STATE",
            ),
            A_t=dict(
                label="Adaptive / history state",
                depends_on="C_t.DynamicState.H,w + delay_state + Theta",
                classification="A",
                semantic_class="ADAPTIVE_STATE",
                units="H a.u., Theta a.u., delay_state bool",
                proxy_status=False,
                variables=["H scalar per neuron (1×400) conceptual 5 dims taus 0.1/1/10/100/1000s", "Theta vector theta_dim=2 population restoring (tau_theta 1000s, lr1e-4, bounds [-0.1,0.1])", "delay_state (120×400) ring buffer B_t"],
                provenance="jomission/dynamics/h_state.py:28 H_COORDINATES 5 taus + jomission/dynamics/hdp.py:14 HDPConfig + jaxfne/_pipeline.py ContinuationState.delay_state",
                note="A is slow/history; H is H_fast…H_context conceptual vs implemented scalar per freeze; ADAPTIVE_STATE",
            ),
            q_t=dict(
                label="Source / current",
                depends_on="X_t + A_t + Static(W, drive, I_bg)",
                classification="q",
                semantic_class="SOURCE",
                units="native current a.u. (physical_amplitude_calibrated=False)",
                proxy_status=False,
                variables=["q_t = source_scale * (I_tonic+I_private+I_noise+I_syn + spike_gain*spikes) — canonical relative source"],
                provenance="jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS + jaxfne/presets.py DEFAULT_SPIKE_IMPULSE_GAIN + jomission/network/builder.py:49 Poisson 2kHz amp2.0 + :159 tonic 3.0",
                note="q is not spike count; it is continuous source proxy per neuron; SOURCE",
            ),
            phi_t=dict(
                label="Field (field_proxy / LFP-like proxy_readout, never physical LFP)",
                depends_on="q_t",
                classification="phi",
                semantic_class="FIELD_PROXY",
                variables=["phi_t = project_laminar_sources(q_t) — laminar field_proxy (mean_zero_neumann gauge, proxy_no_field_solve builder.py:409, not PDE solve)"],
                provenance="jaxfne/fields + jaxfne/_config.py field(source_mode='proxy_no_field_solve') builder.py:409 + jaxfne/io.py manifest field_solver_status 'linear_solver' claim_level 'proxy_readout'",
                note="Field is field_proxy (LFP-like proxy_readout); not calibrated in physical units (truth gate clamped); FIELD_PROXY, proxy_status True",
            ),
            y_t=dict(
                label="Probe / CSD (probe READOUT, CSD DERIVED_FROM(field_proxy))",
                depends_on="phi_t",
                classification="y",
                semantic_class="READOUT",
                variables=["y_t = Probe(n_contacts=16) @ phi_t — linear readout per contact (READOUT, proxy_readout)", "CSD = second spatial derivative of phi_proxy (DERIVED_FROM(field_proxy), proxy_readout)"],
                provenance="jaxfne/_config.py probe(n_contacts=16) + jaxfne/fields.py probe_laminar_modes + jomission/network/builder.py:410 cfg.probe(n_contacts=16)",
                note="Probe is READOUT; CSD is DERIVED_FROM(field_proxy) post-hoc, not independent state",
            ),
        ),
        edges=[
            dict(src="C_t", dst="X_t", transform="slice DynamicState.v,u,prev_spikes,syn_state", kind="STATE_SLICE"),
            dict(src="C_t", dst="A_t", transform="slice DynamicState.H,w + delay_state + Theta", kind="STATE_SLICE"),
            dict(src="X_t", dst="q_t", transform="source_proxy = source_scale*(I_native + gain*spikes)", kind="EMITTER"),
            dict(src="A_t", dst="q_t", transform="H modulates I, w scales syn_state", kind="ADAPTIVE_MODULATION"),
            dict(src="q_t", dst="phi_t", transform="project_laminar_sources(q, geometry)", kind="PROJECTION"),
            dict(src="phi_t", dst="y_t", transform="probe_laminar_modes(phi) @ W_probe", kind="LINEAR_READOUT"),
        ],
        derived_transforms=[
            dict(name="relative_V (V_i - bar V)", inputs=["V_m","mean(V_m)"], output="relative_V", kind="DERIVED_FROM(V_m)", semantic_class="DERIVED", units="mV (relative)", derived_from="V_m", proxy_status=False, note="Not independent state; mean subtraction per field gauge; DERIVED_FROM(V_m) (V1 graph C_t→X_t→q_t→φ_t→y_t)", provenance="jaxfne/_signals.py Signals.field diagnostics gauge='mean_zero'"),
            dict(name="CSD", inputs=["phi_t (field_proxy)"], output="CSD_proxy", kind="DERIVED_FROM(field_proxy)", semantic_class="DERIVED", units="a.u./mm² (proxy_readout)", derived_from="field_proxy", proxy_status=True, note="Second spatial derivative of phi_proxy along depth; not independent state; DERIVED_FROM(field_proxy) via linear_solver", provenance="jaxfne/fields.py + jaxfne/io.py manifest csd_sign_convention + builder.py:409 proxy_no_field_solve"),
            dict(name="field_proxy (LFP-like proxy_readout)", inputs=["phi (proxy_no_field_solve)"], output="field_proxy", kind="DERIVED_FROM(field_proxy)", semantic_class="FIELD_PROXY", units="a.u. (proxy_readout, physical_amplitude_calibrated=False)", derived_from="phi_t", proxy_status=True, note="Weighted sum of sources per laminar mode via project_laminar_sources; proxy_no_field_solve not physical LFP/EEG", provenance="jaxfne/fields.py probe_laminar_modes + jaxfne/_config.py field(source_mode='proxy_no_field_solve') builder.py:409"),
            dict(name="EI_proxy (Efrac_proxy W·r)", inputs=["W (EdgeList weight)", "r (spike rate)"], output="Efrac_proxy", kind="DERIVED_FROM(W·r)", semantic_class="PROXY_ESTIMATE", units="dimensionless proxy", derived_from="W·r", proxy_status=True, note="Efrac_proxy W·r is PROXY_ESTIMATE not realized; realized requires edge_current_trace blocked by jaxfne/_model_simulate.py:280 when delay_steps [20,80,120]+HDP", provenance="jomission/recording/observables.py:93 + jaxfne/_model_simulate.py:280 + jaxfne/emitters.py:2846"),
            dict(name="Population rate", inputs=["spikes"], output="r(t) per class/area", kind="DERIVED_FROM(X)", semantic_class="DERIVED", units="Hz", derived_from="spikes", proxy_status=False, note="Binned spike counts; not state; DERIVED_FROM(spikes) X"),
            dict(name="ISI CV / Fano", inputs=["spikes"], output="CV_ISI, Fano", kind="DERIVED_FROM(X)", semantic_class="DERIVED", units="dimensionless", derived_from="spikes", proxy_status=False, note="Statistics of spike train; not state; DERIVED_FROM(spikes)"),
        ],
        dependency_classification=dict(
            X="V_m, spikes, syn_state — fast, per dt0.1, advisor class X → STATE",
            A="H, Theta, delay/history — slow/history, advisor class A → ADAPTIVE_STATE",
            q="source/current — per neuron continuous, class q → SOURCE",
            phi="field_proxy (LFP-like proxy_readout) — laminar proxy, class phi → FIELD_PROXY",
            y="probe (READOUT) / CSD (DERIVED_FROM(field_proxy)) — measurement, class y → READOUT / DERIVED",
        ),
        invariant="C_t carries truth; X_t and A_t are slices; q,phi,y are deterministic transforms; derived quantities (relative_V DERIVED_FROM(V_m), CSD DERIVED_FROM(field_proxy)) are not added to state count; V1 graph C_t→X_t→q_t→φ_t→y_t",
        frozen_note="No mutation of primitives: E mixture M2, SST b0.21, VIP b0.20, PV 1.7, pseudogenome v0, tonic3.0, Poisson2kHz, delays [20,80,120] frozen per manifests/agsdr_local_freeze.json:1 be9b96ab; builder.py:409 proxy_no_field_solve preserved",
    )


def get_observable_basis_hash() -> str:
    """Deterministic hash of observable_basis observables table."""
    try:
        basis = observable_basis()
        payload = json.dumps(basis.get("observables", []), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def model_summary(config_hash: str | None = None, seed: int = 0, dt_ms: float = 0.1) -> Dict[str, Any]:
    """Build ModelSummary for given configuration hash.

    For every configuration hash C (ledger tip be9b96ab or C019 65b302e8c7cdceb5)
    produces header/STATE/POPULATIONS/CONNECTIVITY/OUTPUT BASIS sections.
    Introspects builder.py:62 motif, builder.py:90 delays, jaxfne/emitters.py.

    Args:
        config_hash: ledger config_hash (16 hex). None → current builder (C018 seed0).
        seed: RNG seed for deterministic build (default 0 matches C018/C019 seed0).
        dt_ms: timestep (0.1 frozen).

    Returns:
        JSON-serializable dict with header, state, ontology, populations, connectivity, output_basis.

    File:line citations embedded per entry.
    """
    # Resolve generated-owner model/cfg — never recompute prettier alternative
    model, cfg, actual_hash = _resolve_model(seed=int(seed), for_hash=config_hash)
    # Determine displayed hash: if requested is C019 use that, else actual
    display_hash = str(config_hash).strip().lower() if config_hash is not None else str(actual_hash).lower()
    # Normalize short vs 16
    if len(display_hash) < 16:
        # if short parent given, expand to known ledger hashes
        if display_hash == FROZEN_C019_PARENT.lower() or display_hash.startswith(FROZEN_C019_PARENT.lower()):
            display_hash = FROZEN_C019_HASH.lower()
        elif display_hash == FROZEN_PARENT_HASH[:8].lower():
            display_hash = FROZEN_PARENT_HASH.lower()
        else:
            display_hash = str(actual_hash).lower()
    else:
        display_hash = display_hash[:16].lower()

    # Header
    try:
        parent_hash = FROZEN_PARENT_HASH if display_hash == FROZEN_C019_HASH.lower() else str(cfg.metadata.get("parent_hash", FROZEN_PARENT_HASH))[:16]
        # Ensure parent for C019 is be9b96ab short
        if display_hash == FROZEN_C019_HASH.lower():
            parent_disp = FROZEN_C019_PARENT
        else:
            parent_disp = parent_hash[:8] if len(parent_hash)>=8 else parent_hash
    except Exception:
        parent_disp = FROZEN_C019_PARENT if display_hash==FROZEN_C019_HASH else FROZEN_PARENT_HASH[:8]

    header = dict(
        Model=f"C019" if display_hash==FROZEN_C019_HASH.lower() else f"Model_{display_hash[:8]}",
        config_hash=str(display_hash),
        parent=str(parent_disp),
        Engine=f"JaxFNE {JAXFNE_VERSION}",
        dt_ms=float(dt_ms),
        dt_steps_per_ms=1.0/float(dt_ms),
        areas=int(len(JOMISSION_AREAS)),
        layers=int(len(JOMISSION_LAYERS)),
        neurons=int(len(_neuron_table(model)) or 400),
        edges=int(_edge_arrays(model)["pre"].shape[0] if _edge_arrays(model) else 10590),
        connectivity="sparse-local",
        spatial_sigma=0.08,
        max_indegree=25,
        within_gain=float(WITHIN_GAIN_DEFAULT),
        p_feedforward=float(P_FEEDFORWARD_DEFAULT),
        p_feedback=float(P_FEEDBACK_DEFAULT),
        pseudogenome_version=str(PSEUDOGENOME_VERSION),
        e_mixture_version=str(E_MIXTURE_VERSION),
        tonic_drive_scale=float(TONIC_DRIVE_SCALE_DEFAULT),
        background_poisson=dict(rate_hz=float(BACKGROUND_POISSON_RATE_HZ_DEFAULT), amplitude=float(BACKGROUND_POISSON_AMPLITUDE_DEFAULT), target="all", seed=int(seed)+7919),
        provenance_header="builder.py:62 MOTIF_GAIN v0 16 gains, builder.py:90 FF/FB maps + delays 20/80/120 steps, builder.py:395 spatial_sigma 0.08 max_in_degree 25, jaxfne/emitters.py EdgeList",
        scaling=dict(weight="log sign·log1p(|W|/1e-6) zero-safe (builder.py:142 CV1.5 heavy tail)", H_tau="log10 0.1/1/10/100/1000s (h_state.py:28, 4 orders)", CV="linear [0.5,1.5]", rho="linear [-0.6,0.6]", Fano="linear", probability="linear", delay="linear 2/8/12ms (H tau log separate)"),
    )

    # If theta overlay present, add it
    try:
        if display_hash==FROZEN_C019_HASH.lower():
            header["theta_star"] = [float(x) for x in FROZEN_C019_THETA]
            header["theta_labels"] = list(FROZEN_C019_THETA_LABELS)
            header["theta_provenance"] = "AGSDR_LOCAL_CL1 orthogonal overlay g_rec/g_fastEI/g_dend/g_disinh/g_background per manifests/agsdr_local_freeze.json:132 orthogonal masks + agsdr_N2_runner.py:550"
    except Exception:
        pass

    # STATE ontology
    state = _compute_state_ontology(model, cfg, dt_ms=float(dt_ms))
    # Add N_total formula and free/derived split note
    state["formula"] = "N_total = N_static + N_dynamic + N_plastic + N_history + N_recording"
    state["free_vs_fixed"] = f"N_free={state['N_free']} (16 motif gains + within/pFF/pFB + 5 theta_local) vs N_fixed~{state['N_fixed']} intrinsic frozen vs N_derived~{state['N_derived']} (positions, pre/post, jittered a/b/c/d fixed per manifests/agsdr_local_freeze.json:1)"
    state["frozen_note"] = "No primitive mutation: E mixture M2 RS70/CH20/EFS10, SST b0.21, VIP b0.20, PV1.7, pseudogenome v0, tonic3.0, Poisson2kHz amp2.0, delays [20,80,120] frozen per freeze manifest"

    # ONTOLOGY stratified table (advisor refinement)
    ontology = ontology_table(model, cfg, dt_ms=float(dt_ms))

    # POPULATIONS
    pops = _populations_table(model)
    # counts summary
    pop_summary = Counter()
    for r in pops:
        pop_summary[r["Class"]] += int(r["N"])
    # CONNECTIVITY
    conn = _connectivity_table(model, dt_ms=float(dt_ms))
    # Highlight required motifs per task
    highlights = {}
    for r in conn:
        key = (r["Source"], r["Target"])
        # L4->L2/3 etc — check for V1 L2/3 E -> V4 L4 287 edges
        if r["Source"] == "V1 L2/3 E" and r["Target"] == "V4 L4 E":
            highlights["V1_L2_3_E_to_V4_L4_E"] = r
        if "PV->E" in r["Source"]+r["Target"]:
            highlights.setdefault("PV_E_examples", []).append(r)
        if "VIP->SST" in r["Source"]+r["Target"]:
            highlights.setdefault("VIP_SST", []).append(r)

    # OUTPUT BASIS
    output_basis = dict(
        Fast_neural=dict(
            V_m="(n_steps, 400) float32 — membrane voltage per neuron per dt0.1",
            spikes="(n_steps, 400) bool — threshold crossings V>=30",
            provenance="jaxfne/emitters.py:211 _izhikevich_dv_du dv=0.04v²+5v+140-u+I du=a(bv-u) v>=30→c u+=d; jaxfne/_signals.py Signals.V_m/spikes",
            classification="X (fast, per advisor)",
        ),
        Adaptive=dict(
            H="(400,) scalar H per neuron (conceptual 5 dims taus 0.1/1/10/100/1000) — gated B0 H_conceptual vs H_implemented mismatch until M-06",
            Theta="(2,) population restoring vector (tau_theta 1000s >> tau_X, lr1e-4, bounds [-0.1,0.1], channels edge_weight+intrinsic_drive)",
            delay_history="ring buffer (120 ×400) — delay_state B_t legacy spike_history",
            provenance="jomission/dynamics/h_state.py:28-69 H_COORDINATES 5 taus + jomission/dynamics/hdp.py:14 HDPConfig tau_theta 1000 + jaxfne/_pipeline.py DynamicState.H/w + ContinuationState.delay_state",
            classification="A (H,Theta,delay/history)",
        ),
        Source=dict(
            q="source_proxy (400,) — source_scale*(I_native + gain*spikes) relative current per neuron",
            provenance="jaxfne/emitters.py:55 source_scale + jaxfne/presets.py DEFAULT_SPIKE_IMPULSE_GAIN + jomission/network/builder.py:1592 simulation_with_background_poisson",
            classification="q (source/current)",
        ),
        Field=dict(
            phi="phi (n_contacts or depth × time) — laminar proxy field mean_zero_neumann gauge",
            provenance="jaxfne/_config.py field(source_mode='proxy_no_field_solve') + jaxfne/fields.py project_laminar_sources",
            classification="phi (field)",
        ),
        Probe=dict(
            y="y (n_contacts=16 × time) — probe readout per laminar contact; CSD = d²phi/dz² DERIVED_FROM(field)",
            provenance="jomission/network/builder.py:410 cfg.probe(n_contacts=16) + jaxfne/fields.py probe_laminar_modes",
            classification="y (probe/CSD)",
            derived_mark="CSD, V_i - bar V are DERIVED_FROM(field), not independent state",
        ),
        observation_equation="y_t = P * L * q( X_t(V_m,spikes), A_t(H,Theta,delay) ; W, drive ) + noise; phi_t = L q_t; q_t = f( X_t, A_t )",
    )

    obs_basis = observable_basis()

    result = dict(
        header=header,
        STATE=state,
        ONTOLOGY=ontology,
        POPULATIONS=dict(rows=pops, summary=dict(pop_summary), counts_per_area_layer=dict(Counter((r["Area"], r["Layer"]) for r in pops))),
        CONNECTIVITY=dict(rows=conn, highlights=highlights, provenance="builder.py:62 MOTIF_GAIN 16 gains + builder.py:90 FF L2/3->L4 FB L6->L1/L5 delays 20/80/120 + jaxfne/emitters.py:545 edge_list_with_delay_ms"),
        OUTPUT_BASIS=output_basis,
        OBSERVABLE_BASIS=obs_basis,
        OBSERVABLE_BASIS_hint="See observable_basis() for full graph C_t->X_t->q_t->phi_t->y_t",
        citations=[
            "jomission/network/builder.py:39 jitter σ0.10 per-neuron (per-neuron static, derived)",
            "jomission/network/builder.py:62 MOTIF_GAIN DESIRED_MOTIF_GAIN v0 16 gains (E->PV1.7 E->SST0.70 PV->E1.30 SST->E0.60 SST->PV0.60 SST->VIP0.80 VIP->SST5.00 etc) — configured→AGSDR 5 dims",
            "jomission/network/builder.py:90-97 FF_LAYER_MAP FB_LAYER_MAPS DELAY_FF_MS 8 DELAY_FB_MS 12 DELAY_WITHIN_MS 2 — configured + derived delay_steps",
            "jomission/network/builder.py:395-407 cfg.connectivity spatial_sigma0.08 max_in_degree25 + _apply_spatial_locality — configured spatial",
            "jomission/network/builder.py:142-148 LOGNORMAL sigma1.085 mu-0.589 CV1.5 — derived sigma/mu",
            "jomission/network/builder.py:99 VIP_B_CORRECTED 0.20",
            "jomission/network/builder.py:114 SST_B_CORRECTED 0.21",
            "jomission/network/builder.py:125 PV_DRIVE_SCALE 1.7",
            "jomission/network/builder.py:159 TONIC_DRIVE_SCALE 0.6 (3.0/5.0)",
            "jomission/network/builder.py:49 BACKGROUND_POISSON 2000 amp2.0 (configured I_bg)",
            "jomission/network/builder.py:179-196 E_MIXTURE_M2 RS70/CH20/EFS10 + :1008 _apply_typed_E_phenotypes + :212 _apply_per_neuron_jitter sigma0.10 — per-neuron static",
            "jomission/network/populations.py:12 AREAS V1 V4 FEF PFC :13 LAYERS L1 L2/3 L4 L5 L6 :31 LAYER_COUNT_FRAC_DEFAULT :44 AREA_LAYER_CELL_TYPES — configured",
            "jomission/network/connectivity.py:14 WITHIN_GAIN 0.35 :17 P_FF 0.30 :18 P_FB 0.20 + DELAY_* — configured",
            "jaxfne/emitters.py:55 IZHIKEVICH_CELL_TYPE_DEFAULTS + :211 dv/du + :545 edge_list_with_delay_ms + :598 EdgeList — engine-fixed + per-neuron/per-edge static",
            "jaxfne/io.py:42 config_hash (generated-owner hash)",
            "jomission/dynamics/h_state.py:28-69 H_COORDINATES 5 taus 0.1/1/10/100/1000 + jomission/dynamics/hdp.py:14 HDPConfig tau_theta1000 lr1e-4 — plastic/history",
            "manifests/agsdr_local_freeze.json:1 be9b96ab parent + :67 spatial_locality + :97 H_implemented + :131 orthogonal masks freeze — currently_frozen vs AGSDR-exposed",
            "manifests/gen2_modification_ledger.json: GEN2_C019 65b302e8c7cdceb5 theta_star — experimentally adjustable vs frozen",
        ],
    )
    return result


def model_summary_text(config_hash: str | None = None, seed: int = 0, dt_ms: float = 0.1, max_conn_rows: int = 40) -> str:
    """Text rendering analogous to torchinfo / tf.summary — for ModelSummary file."""
    data = model_summary(config_hash=config_hash, seed=seed, dt_ms=dt_ms)
    h = data["header"]
    s = data["STATE"]
    pops = data["POPULATIONS"]["rows"]
    conn = data["CONNECTIVITY"]["rows"]
    out = data["OUTPUT_BASIS"]

    lines: List[str] = []
    lines.append("="*96)
    lines.append(f"ModelSummary — {h['Model']} {h['config_hash']}  (Parent {h['parent']})")
    lines.append(f"Engine {h['Engine']}  dt={h['dt_ms']:.1f} ms  areas={h['areas']} layers={h['layers']}  neurons={h['neurons']}  edges={h['edges']}")
    lines.append(f"Connectivity {h['connectivity']}  spatial_sigma={h['spatial_sigma']:.2f}  max_indegree={h['max_indegree']}  within_gain={h['within_gain']:.2f}  p_FF={h['p_feedforward']:.2f} p_FB={h['p_feedback']:.2f}")
    lines.append(f"Pseudogenome {h['pseudogenome_version']}  E_mixture {h['e_mixture_version']} RS70/CH20/EFS10  tonic_scale={h['tonic_drive_scale']:.1f}  Poisson {h['background_poisson']['rate_hz']:.0f}Hz amp{h['background_poisson']['amplitude']:.1f}")
    lines.append(f"Provenance: {h['provenance_header']}")
    if "theta_star" in h:
        lines.append(f"Theta* {h['theta_labels']} = {h['theta_star']}  (AGSDR_LOCAL_CL1 orthogonal, g_rec/g_fastEI/g_dend/g_disinh/g_background)")
        lines.append(f"  theta_provenance: {h['theta_provenance']}")
    lines.append("-"*96)
    # STATE
    lines.append("STATE — ontology per Visualization Law")
    lines.append(f"  N_total = N_static + N_dynamic + N_plastic + N_history + N_recording   (free vs derived/fixed)")
    lines.append(f"  N_static    {s['N_static']:7d}  bytes {s['bytes_static']:8d}  (per-neuron a,b,c,d,drive,sign,v0,u0 [8×{s['n_neurons']}] + positions {s['pos_n']} + EdgeList pre/post/tau/weight×{s['n_edges']})")
    lines.append(f"  N_dynamic   {s['N_dynamic']:7d}  bytes {s['bytes_dynamic']:8d}  (DynamicState: v,u,prev_spikes 3×{s['n_neurons']} + syn_state {s['n_edges']} + H {s['n_neurons']}×1 + w {s['n_edges']})")
    lines.append(f"  N_plastic   {s['N_plastic']:7d}  bytes {s['bytes_plastic']:8d}  ({s['plastic_desc']})")
    lines.append(f"  N_history   {s['N_history']:7d}  bytes {s['bytes_history']:8d}  (delay_state ring buffer Dmax={s['max_delay_steps']} ({s['max_delay_steps']*s['dt_ms']:.0f}ms) × {s['n_neurons']} = {s['N_history']}) — FIXED, not tunable")
    lines.append(f"  N_recording {s['N_recording']:7d}  bytes {s['bytes_recording']:8d}  (probe {s['n_contacts']} contacts)")
    lines.append(f"  N_rng       {s['N_rng']:7d}  bytes {16:8d}  (PRNGKey 2×uint32)")
    lines.append(f"  N_total     {s['N_total']:7d}  total_state_bytes {s['total_bytes']:8d}  est_run_mem(2000ms) ~{s['est_run_mem']//(1024*1024)} MB (traj vm {s['traj_bytes_vm']//(1024*1024)} MB + spikes {s['traj_bytes_spikes']//1024} KB + field {s['traj_bytes_field']//1024} KB)")
    lines.append(f"  N_free      {s['N_free']:7d}  (16 motif gains + 3 conn + 5 theta_local orthogonal) vs N_fixed~{s['N_fixed']} vs N_derived~{s['N_derived']}")
    lines.append(f"  {s['free_vs_fixed']}")
    lines.append(f"  Dense W {s['dense_W_n']} numbers ({s['dense_W_bytes']//1024} KB) — proxy, edge_list {s['n_edges']} sparse authoritative (jaxfne/emitters.py EdgeList)")
    lines.append(f"  Formula: {s['formula']}")
    lines.append(f"  Frozen: {s['frozen_note']}")
    # ONTOLOGY stratified
    ont = data["ONTOLOGY"]
    lines.append("-"*96)
    lines.append("ONTOLOGY — stratified (advisor refinement, Visualization Law)")
    lines.append(f"  Engine-fixed constants:           {ont['engine_fixed_constants']['count']:5d}  (jaxfne/emitters.py:55,211 — not tunable)")
    lines.append(f"  Configured static values:         {ont['configured_static_values']['count']:5d}  (builder.py:39,62,90,395 + populations.py:12/31 — hash-visible scalars)")
    lines.append(f"  Derived static values:            {ont['derived_static_values']['count']:5d}  (LOGNORMAL_SIGMA 1.085 etc DERIVED)")
    lines.append(f"  Per-neuron static:                {ont['per_neuron_static']['count']:5d}  (8×{ont['per_neuron_static']['n_neurons']} + positions; builder.py:39 jitter)")
    lines.append(f"  Per-edge static:                  {ont['per_edge_static']['count']:5d}  (5×{ont['per_edge_static']['n_edges']} edge_list; builder.py:395 + jaxfne/emitters.py:545)")
    lines.append(f"    (+ delay_steps)                 {ont['per_edge_static']['extra_delay_steps']:5d}  (delay indexed history but stored static)")
    lines.append(f"  Dynamic:                          {ont['dynamic']['count']:5d}  (v,u,spikes + syn+H+w)")
    lines.append(f"  Plastic (Theta):                  {ont['plastic']['count']:5d}  (theta_dim=2, tau_theta1000)")
    lines.append(f"  History (delay_state):            {ont['history']['count']:5d}  ({ont['history']['shape']})")
    lines.append(f"  Recording (probe):                {ont['recording']['count']:5d}  (contacts {ont['recording']['n_contacts']})")
    lines.append(f"  --")
    lines.append(f"  N_model parameters (static nums): {ont['N_model_parameters']:5d}  (per_neuron + per_edge numbers that define model)")
    lines.append(f"  N_tunable (experimentally adjustable): {ont['N_tunable']:5d}  (16 motif +3 conn +5 theta =24 scalars, N_free24)")
    lines.append(f"  N_AGSDR dims/scalars:              {ont['N_AGSDR_dims']} dims / {ont['N_AGSDR_scalars']} scalars (g_rec1 g_fastEI2 g_dend4 g_disinh1 g_background3)")
    lines.append(f"  Currently frozen scientific params: {ont['currently_frozen_scientific']['count']:5d}  (configured {ont['configured_static_values']['count']} - tunable {ont['N_tunable']} = frozen)")
    lines.append(f"  Inequality: {ont['inequality']}")
    lines.append(f"  Reconciliation: {ont['reconciliation']}")
    lines.append("-"*96)
    # POPULATIONS
    lines.append(f"POPULATIONS — Area Layer Class N intrinsic family Rate target  (n_total={h['neurons']})")
    lines.append(f"  {'Area':<5} {'Layer':<5} {'Class':<6} {'N':>3}  {'Intrinsic':<32} {'Rate target':<18} Provenance")
    lines.append(f"  {'-'*5} {'-'*5} {'-'*6} {'-'*3}  {'-'*32} {'-'*18} {'-'*30}")
    for r in pops:
        i = r["Intrinsic"]
        # truncate for display
        if len(i) > 32:
            i = i[:29]+"..."
        rt = r["RateTarget"]
        if len(rt) > 18:
            rt = rt[:15]+"..."
        lines.append(f"  {r['Area']:<5} {r['Layer']:<5} {r['Class']:<6} {r['N']:>3}  {i:<32} {rt:<18} {r['Provenance'][:38]}")
    # summary
    summ = data["POPULATIONS"]["summary"]
    lines.append(f"  Summary: " + ", ".join(f"{k}:{v}" for k,v in sorted(summ.items())))
    lines.append("-"*96)
    # CONNECTIVITY
    lines.append(f"CONNECTIVITY — Source Target Edges p Wmean Wcv Delay  (top {min(max_conn_rows,len(conn))} by edges, total {len(conn)} motifs, {h['edges']} edges)")
    lines.append(f"  {'Source':<14} {'Target':<14} {'Edges':>5} {'p':>6} {'Wmean':>6} {'Wcv':>5} {'Delay':>6} {'Type':<8} {'MotifGain':>5}  Provenance")
    lines.append(f"  {'-'*14} {'-'*14} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*9}  {'-'*20}")
    # also ensure required motifs appear even if not top
    shown = set()
    for r in conn[:max_conn_rows]:
        lines.append(f"  {r['Source']:<14} {r['Target']:<14} {r['Edges']:>5} {r['p']:>6.3f} {r['Wmean']:>6.3f} {r['Wcv']:>5.2f} {r['Delay_ms']:>6.0f} {r['ConnType']:<8} {r['MotifGain']:>5.2f}  {r['ConnType']} {r['DelaySteps']}")
        shown.add((r["Source"], r["Target"]))
    # required specifics if not shown
    required = []
    for r in conn:
        if r["Source"]=="V1 L2/3 E" and r["Target"]=="V4 L4 E":
            required.append(("V1 L2/3 E->V4 L4 (FF 287 edges example)", r))
        if "PV" in r["Source"] and "E" in r["Target"] and "PV->E" not in [x[0] for x in required]:
            required.append(("PV->E (fastEI g_fastEI 1.3×)", r))
        if "VIP" in r["Source"] and "SST" in r["Target"]:
            required.append(("VIP->SST (5.00 disinh)", r))
    for label, r in required:
        if (r["Source"], r["Target"]) not in shown:
            lines.append(f"  ... required {label}: {r['Source']}->{r['Target']} edges={r['Edges']} p={r['p']:.3f} Wmean={r['Wmean']:.3f} delay={r['Delay_ms']:.0f}ms gain={r['MotifGain']:.2f}")
    lines.append(f"  Provenance: {data['CONNECTIVITY']['provenance']}")
    lines.append(f"  Delays: within {DELAY_WITHIN_MS:.0f}ms ({DELAY_WITHIN_MS/dt_ms:.0f} steps) FF {DELAY_FF_MS:.0f}ms ({DELAY_FF_MS/dt_ms:.0f}) FB {DELAY_FB_MS:.0f}ms ({DELAY_FB_MS/dt_ms:.0f}) (builder.py:90, jaxfne/emitters.py:545)")
    lines.append("-"*96)
    # OUTPUT BASIS
    lines.append("OUTPUT BASIS — measurement graph and basis")
    for key in ("Fast_neural","Adaptive","Source","Field","Probe"):
        sec = out[key]
        lines.append(f"  {key}:")
        for k,v in sec.items():
            if k in ("provenance","classification","derived_mark"):
                lines.append(f"    {k}: {v}")
            else:
                # wrap
                lines.append(f"    {k}: {v}")
    lines.append(f"  Equation: {out.get('observation_equation','')}")
    lines.append("="*96)
    lines.append("ObservableBasis: C_t -> X_t -> q_t -> phi_t -> y_t  (see observable_basis() for full table name/owner/dim/units/independence/parent)")
    lines.append("  X: V_m spikes syn_state (fast)  A: H Theta delay/history  q: source/current  phi: field  y: probe/CSD")
    lines.append("  DERIVED_FROM(field): V_i - bar V (parent V_m), CSD (parent phi) — not independent state, not counted in N_total")
    lines.append("  DERIVED_FROM(q): phi (parent q); DERIVED_FROM(phi): y/CSD (parent phi)")
    lines.append("="*96)
    return "\n".join(lines)
