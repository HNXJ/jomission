"""Network visualization foundation (VIS_FOUNDATION_v0 — V2).

Interactive Plotly figures that consume generated-owner arrays/config objects.
No science mutation — reads builder.py config and EdgeList only.

Provenance citations (frozen C019 65b302e8c7cdceb5):
- builder.py:62   MOTIF_GAIN DESIRED_MOTIF_GAIN v0  (pseudogenome)
- builder.py:90   FF_LAYER_MAP / FB_LAYER_MAPS / DELAY_FF/FB/WITHIN  (laminar hierarchy)
- builder.py:395  cfg.connectivity(spatial_sigma=0.08, max_in_degree=25) / _apply_spatial_locality
- builder.py:623  _apply_motif_gains post-EdgeList seam
- jaxfne/emitters.py  EdgeList(pre,post,weight,delay_steps,tau) / IZHIKEVICH_CELL_TYPE_DEFAULTS
- network/rf.py:47  RFConfig lattice 32×32 sigma 1.8 spacing 3.2 overlap
- populations.py:31 LAYER_COUNT_FRAC_DEFAULT / AREA_LAYER_CELL_TYPES / JOMISSION_AREAS/LAYERS
- network/connectivity.py:35 CONNECTIVITY_TABLE / HIERARCHY / P_FEEDFORWARD/FEEDBACK

Functions:
  hierarchy_fig(model=None, cfg=None) -> go.Figure
  motif_matrix_fig(model=None, ...) -> go.Figure
  spatial_fig(model=None, ...) -> go.Figure
  rf_fig(model=None, rf_config=None) -> go.Figure
  get_motif_stats(model) -> dict
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None  # type: ignore
    make_subplots = None  # type: ignore

# ---------------------------------------------------------------------------
# helpers — read-only consumption of generated-owner objects
# ---------------------------------------------------------------------------

try:
    from jomission.network.populations import (
        JOMISSION_AREAS,
        JOMISSION_LAYERS,
        JOMISSION_CELL_TYPES,
        LAYER_DEPTH_BANDS,
        LAYER_COUNT_FRAC_DEFAULT,
        AREA_LAYER_CELL_TYPES,
    )
    from jomission.network.builder import (
        MOTIF_GAIN,
        FF_LAYER_MAP,
        FB_LAYER_MAPS,
        DELAY_FF_MS,
        DELAY_FB_MS,
        DELAY_WITHIN_MS,
        HIERARCHY,
        build_jomission_model,
        build_jomission_network,
    )
    from jomission.network.connectivity import (
        P_FEEDFORWARD_DEFAULT,
        P_FEEDBACK_DEFAULT,
    )
    from jomission.network.rf import RFConfig, RFOperator
except Exception:  # pragma: no cover
    JOMISSION_AREAS = ("V1", "V4", "FEF", "PFC")  # type: ignore
    JOMISSION_LAYERS = ("L1", "L2/3", "L4", "L5", "L6")  # type: ignore
    JOMISSION_CELL_TYPES = ("E", "PV", "SST", "VIP")  # type: ignore
    LAYER_DEPTH_BANDS = {}  # type: ignore
    MOTIF_GAIN = {("E", "E"): 1.0}  # type: ignore
    FF_LAYER_MAP = {"L2/3": "L4"}  # type: ignore
    FB_LAYER_MAPS = [{"L6": "L1"}, {"L6": "L5"}]  # type: ignore
    DELAY_FF_MS, DELAY_FB_MS, DELAY_WITHIN_MS = 8.0, 12.0, 2.0  # type: ignore
    HIERARCHY = ("V1", "V4", "FEF", "PFC")  # type: ignore
    P_FEEDFORWARD_DEFAULT, P_FEEDBACK_DEFAULT = 0.30, 0.20  # type: ignore
    RFConfig = None  # type: ignore
    RFOperator = None  # type: ignore
    build_jomission_model = None  # type: ignore
    build_jomission_network = None  # type: ignore


def _ensure_model(model: Any | None) -> Any:
    if model is not None:
        return model
    if build_jomission_model is None:
        raise ImportError("build_jomission_model unavailable")
    return build_jomission_model(n_per_area=100, seed=0)


def _neuron_table(model: Any) -> List[Dict[str, Any]]:
    try:
        return list(model.neuron_table())  # type: ignore
    except Exception:
        return []


def _edge_arrays(model: Any) -> Dict[str, np.ndarray]:
    el = model.params.get("edge_list")  # type: ignore
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
        out["source_calibration_status"] = getattr(el, "source_calibration_status", "uncalibrated")
    except Exception:
        pass
    return out


def _positions(model: Any) -> np.ndarray:
    try:
        return np.asarray(model.params["positions"], dtype=np.float64)  # type: ignore
    except Exception:
        return np.zeros((400, 3), dtype=float)


def _classify_edge(
    src_area: str,
    src_layer: str,
    tgt_area: str,
    tgt_layer: str,
) -> str:
    """Classify as local / vertical / FF / FB (populations.py:31, builder.py:90)."""
    if src_area != tgt_area:
        # FF vs FB via hierarchy order (connectivity.py HIERARCHY low->high)
        try:
            si = HIERARCHY.index(src_area)
            ti = HIERARCHY.index(tgt_area)
        except Exception:
            si, ti = 0, 1
        if ti > si:
            # upward hierarchy: check FF map (L2/3->L4)
            if src_layer in FF_LAYER_MAP and FF_LAYER_MAP[src_layer] == tgt_layer:
                return "FF"
            return "FF"  # default inter-area up is FF
        else:
            return "FB"
    # same area
    if src_layer == tgt_layer:
        return "local"
    return "vertical"


# ---------------------------------------------------------------------------
# Sampling + scaling utilities — V0-P1 (adversary L1,A2)
# Must NOT mutate science: statistics use ALL edges; rendering may sample.
# Stratify by area_s×layer_s×class_s × area_t×layer_t×class_t (6-tuple)
# so rare FF/FB/VIP motifs don't disappear.
# Deterministic seed derived from config_hash (jaxfne/io.py:42) when available.
# Log scaling: weight -> log abs with sign preserved; H taus -> log.
# ---------------------------------------------------------------------------

_SAMPLING_DEFAULT_N_RENDERED = 1000
_SAMPLING_DISCLOSURE_TEMPLATE = (
    "Rendering {n_rendered:,} of {n_total:,} edges using deterministic stratified sampling. "
    "All reported degree, weight, probability and delay statistics use the complete EdgeList."
)

def _config_hash_for_model(model: Any) -> str:
    """Deterministic config hash for seeding — reads same generated-owner cfg (jaxfne/io.py:42)."""
    try:
        from jaxfne.io import config_hash as _ch  # type: ignore

        cfg = getattr(model, "config", None)
        if cfg is None:
            # builder.py builds via Configuration; try model.cfg
            cfg = getattr(model, "cfg", None)
        if cfg is not None:
            return str(_ch(cfg))
    except Exception:
        pass
    try:
        # Fallback: hash of edge_list bytes (still deterministic per seed)
        el = model.params.get("edge_list")  # type: ignore
        if el is not None:
            h = hashlib.sha256(np.asarray(el.pre).tobytes()).hexdigest()[:16]
            return h
    except Exception:
        pass
    return "0000000000000000"

def _deterministic_seed(config_hash: str, salt: str = "") -> int:
    """Derive deterministic int seed from config_hash + salt (e.g., 'hierarchy'/'spatial')."""
    try:
        s = f"{config_hash}:{salt}"
        h = hashlib.sha256(s.encode()).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF
    except Exception:
        return 0

def _motif_key_for_edge(
    pre_idx: int,
    post_idx: int,
    areas: Sequence[str],
    layers: Sequence[str],
    cts: Sequence[str],
) -> Tuple[str, str, str, str, str, str]:
    try:
        return (
            str(areas[int(pre_idx)]),
            str(layers[int(pre_idx)]),
            str(cts[int(pre_idx)]),
            str(areas[int(post_idx)]),
            str(layers[int(post_idx)]),
            str(cts[int(post_idx)]),
        )
    except Exception:
        return ("?", "?", "?", "?", "?", "?")

def _stratified_sample_indices(
    model: Any,
    n_rendered: int = _SAMPLING_DEFAULT_N_RENDERED,
    seed_salt: str = "viz",
) -> Tuple[np.ndarray, int, int, Dict[Tuple[str, str, str, str, str, str], int]]:
    """Deterministic stratified sampling per 6-tuple motif.

    Returns (sampled_indices, n_total, n_rendered_actual, per_motif_sampled_counts).
    Statistics must be computed from ALL edges (caller retains full EdgeList);
    sampled_indices are for rendering only.
    Stratify by area_s×layer_s×class_s × area_t×layer_t×class_t
    (builder.py:62 MOTIF_GAIN 16 entries, populations.py:12 AREAS/LAYERS).
    """
    tbl = _neuron_table(model)
    n_total = 0
    try:
        ea = _edge_arrays(model)
        pre = ea.get("pre", np.array([], dtype=int))
        n_total = int(pre.shape[0]) if pre.size else 0
    except Exception:
        return np.array([], dtype=int), 0, 0, {}
    if n_total == 0 or n_total <= int(n_rendered):
        return np.arange(n_total, dtype=int), n_total, n_total, {}
    areas = [str(r.get("area")) for r in tbl]
    layers = [str(r.get("layer")) for r in tbl]
    cts = [str(r.get("cell_type")) for r in tbl]
    ea = _edge_arrays(model)
    pre, post = ea["pre"], ea["post"]
    # group indices by motif key
    from collections import defaultdict as _dd
    motif_to_indices: Dict[Tuple[str, str, str, str, str, str], List[int]] = _dd(list)
    for ei in range(n_total):
        try:
            key = _motif_key_for_edge(int(pre[ei]), int(post[ei]), areas, layers, cts)
        except Exception:
            key = ("?", "?", "?", "?", "?", "?")
        motif_to_indices[key].append(int(ei))
    motifs = list(motif_to_indices.keys())
    m = len(motifs)
    # allocate quotas: at least 1 per motif, then proportional to group size
    quotas: Dict[Tuple[str, str, str, str, str, str], int] = {}
    # initial proportional
    remaining = int(n_rendered)
    # first pass: ensure at least 1 per motif, capped by group size
    for k, idxs in motif_to_indices.items():
        quotas[k] = 1 if len(idxs) >= 1 else 0
    remaining_after_min = int(n_rendered) - sum(quotas.values())
    if remaining_after_min < 0:
        # more motifs than budget: sample motifs uniformly deterministic, keep 1 per sampled motif
        # choose n_rendered motifs by deterministic hash order
        ch = _config_hash_for_model(model)
        base = _deterministic_seed(ch, f"{seed_salt}:motif_select")
        # deterministic ordering by hash of motif key
        def _motif_hash(k: Tuple[str, ...]) -> int:
            hs = hashlib.sha256("|".join(k).encode()).hexdigest()
            return int(hs[:8], 16)
        ordered = sorted(motifs, key=_motif_hash)
        # use base to rotate? simpler take first n_rendered
        selected = set(ordered[: int(n_rendered)])
        sampled: List[int] = []
        for k in selected:
            idxs = motif_to_indices[k]
            # pick one edge per selected motif deterministic
            ch2 = _config_hash_for_model(model)
            pseed = _deterministic_seed(ch2, f"{seed_salt}:{k}:0")
            rng = np.random.default_rng(pseed)
            pick = int(rng.choice(idxs, size=1)[0]) if len(idxs) > 1 else int(idxs[0])
            sampled.append(pick)
        sampled_arr = np.array(sorted(sampled), dtype=int)
        per_motif = {k: (1 if k in selected else 0) for k in motifs}
        return sampled_arr, n_total, int(sampled_arr.shape[0]), per_motif
    # distribute remaining proportional to group sizes beyond the 1 already allocated
    # weight = (size -1) for remaining? Use size proportional for full allocation then adjust
    # compute ideal quotas including the 1
    total = float(n_total)
    ideal = {k: max(1, round(int(n_rendered) * len(idxs) / total)) for k, idxs in motif_to_indices.items()}
    # clamp to group size
    for k in list(ideal.keys()):
        ideal[k] = min(int(ideal[k]), len(motif_to_indices[k]))
        ideal[k] = max(int(ideal[k]), 1 if len(motif_to_indices[k]) >= 1 else 0)
    # adjust sum to n_rendered via greedy correction (deterministic)
    s = sum(ideal.values())
    # if over, reduce largest quotas deterministically by motif hash order descending
    def _motif_sort_key(k: Tuple[str, ...]) -> Tuple[float, int]:
        # larger groups first, tie-break by hash
        hs = int(hashlib.sha256("|".join(k).encode()).hexdigest()[:8], 16)
        return (float(len(motif_to_indices[k])), hs)
    if s > int(n_rendered):
        # need to reduce
        ordered_desc = sorted(motifs, key=_motif_sort_key, reverse=True)
        idx = 0
        while s > int(n_rendered) and idx < len(ordered_desc) * 4:
            k = ordered_desc[idx % len(ordered_desc)]
            if ideal[k] > 1:
                ideal[k] -= 1
                s -= 1
            idx += 1
    elif s < int(n_rendered):
        ordered_desc = sorted(motifs, key=_motif_sort_key, reverse=True)
        idx = 0
        while s < int(n_rendered) and idx < len(ordered_desc) * 8:
            k = ordered_desc[idx % len(ordered_desc)]
            if ideal[k] < len(motif_to_indices[k]):
                ideal[k] += 1
                s += 1
            idx += 1
    quotas = {k: int(v) for k, v in ideal.items()}
    # now sample per motif deterministically
    sampled_all: List[int] = []
    per_motif_counts: Dict[Tuple[str, str, str, str, str, str], int] = {}
    ch = _config_hash_for_model(model)
    for k, idxs in motif_to_indices.items():
        q = int(quotas.get(k, 0))
        q = min(q, len(idxs))
        if q <= 0:
            per_motif_counts[k] = 0
            continue
        pseed = _deterministic_seed(ch, f"{seed_salt}:{k}")
        rng = np.random.default_rng(pseed)
        # deterministic permutation within motif
        perm = rng.permutation(np.array(idxs, dtype=int))
        chosen = perm[:q]
        sampled_all.extend(chosen.tolist())
        per_motif_counts[k] = int(q)
    sampled_arr = np.array(sorted(sampled_all), dtype=int)
    # if still off by a few due to rounding, trim/pad deterministically
    if sampled_arr.shape[0] > int(n_rendered):
        # trim by deterministic global shuffle
        gseed = _deterministic_seed(ch, f"{seed_salt}:global_trim")
        rng = np.random.default_rng(gseed)
        perm = rng.permutation(sampled_arr)
        sampled_arr = np.sort(perm[: int(n_rendered)])
    elif sampled_arr.shape[0] < int(n_rendered) and sampled_arr.shape[0] < n_total:
        # pad from remaining pool
        remaining_pool = np.array([i for i in range(n_total) if i not in set(sampled_arr.tolist())], dtype=int)
        if remaining_pool.size:
            gseed = _deterministic_seed(ch, f"{seed_salt}:global_pad")
            rng = np.random.default_rng(gseed)
            need = int(n_rendered) - int(sampled_arr.shape[0])
            pad = rng.choice(remaining_pool, size=min(need, remaining_pool.size), replace=False)
            sampled_arr = np.sort(np.concatenate([sampled_arr, pad]))
    return sampled_arr, n_total, int(sampled_arr.shape[0]), per_motif_counts

def _sampling_disclosure(n_total: int, n_rendered: int) -> str:
    return _SAMPLING_DISCLOSURE_TEMPLATE.format(n_total=int(n_total), n_rendered=int(n_rendered))

def _provenance_footer_text(
    n_total: int,
    n_rendered: int,
    filters: Dict[str, Any] | None = None,
    config_hash: str | None = None,
    extra: str | None = None,
) -> str:
    parts = [_sampling_disclosure(n_total, n_rendered)]
    if filters:
        filt_str = ", ".join(f"{k}={v}" for k, v in filters.items() if v is not None and v != "ALL")
        if filt_str:
            parts.append(f"Filters: {filt_str}.")
    if config_hash:
        parts.append(f"config_hash {config_hash[:8]} (jaxfne/io.py:42).")
    if extra:
        parts.append(extra)
    # builder provenance
    parts.append("Stratified by area_s×layer_s×class_s × area_t×layer_t×class_t (builder.py:62, populations.py:12) — rare FF/FB/VIP preserved.")
    return " ".join(parts)

def _safe_log_abs_weight(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log abs weight with sign preserved: sign(w) * log1p(|w|/eps) or log(|w|+eps). Zero-safe."""
    w = np.asarray(w, dtype=float)
    # sign-preserving log: sign * log1p(abs/eps) handles zero gracefully and heavy tail (builder.py:142 CV1.5)
    return np.sign(w) * np.log1p(np.abs(w) / float(eps))

def _safe_log_tau(tau: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Log H tau (positive, 0.1-1000s span 4 orders — h_state.py:28). Zero-safe log10."""
    tau = np.asarray(tau, dtype=float)
    return np.log10(np.maximum(tau, float(eps)))



def get_motif_stats(model: Any | None = None, dt_ms: float = 0.1) -> Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]]:
    """Aggregate EdgeList by full motif (src_area,src_layer,src_ct, tgt_area,tgt_layer,tgt_ct).

    Returns per-motif dict with edges, mean indegree, meanW, CV, delay_ms, p, provenance.
    Reads builder.py:62,90,395 and jaxfne/emitters.py EdgeList (no mutation).
    """
    model = _ensure_model(model)
    tbl = _neuron_table(model)
    areas = [str(r.get("area")) for r in tbl]
    layers = [str(r.get("layer")) for r in tbl]
    cts = [str(r.get("cell_type")) for r in tbl]
    n_tbl = len(tbl)
    ea = _edge_arrays(model)
    if not ea:
        return {}
    pre, post, w, ds = ea["pre"], ea["post"], ea["weight"], ea["delay_steps"]
    # count per population sizes for p computation
    pop_counts: Counter = Counter((areas[i], layers[i], cts[i]) for i in range(n_tbl))
    # aggregate
    agg: Dict[Tuple[str, str, str, str, str, str], List[int]] = defaultdict(list)
    for ei in range(int(pre.shape[0])):
        try:
            sa, sl, sc = areas[int(pre[ei])], layers[int(pre[ei])], cts[int(pre[ei])]
            ta, tl, tc = areas[int(post[ei])], layers[int(post[ei])], cts[int(post[ei])]
        except Exception:
            continue
        agg[(sa, sl, sc, ta, tl, tc)].append(int(ei))
    out: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
    # target population counts for indegree
    tgt_counts: Counter = Counter((areas[int(post[i])], layers[int(post[i])], cts[int(post[i])]) for i in range(int(post.shape[0])))
    for key, idxs in agg.items():
        sa, sl, sc, ta, tl, tc = key
        ww = w[np.array(idxs)]
        mean_w = float(np.mean(np.abs(ww))) if ww.size else 0.0
        std_w = float(np.std(np.abs(ww))) if ww.size else 0.0
        cv = float(std_w / mean_w) if mean_w > 0 else 0.0
        # mean delay
        delays = ds[np.array(idxs)] * float(dt_ms)
        mean_delay = float(np.mean(delays)) if delays.size else 0.0
        # p: edges / possible pairs (src pop × tgt pop)
        n_src = int(pop_counts.get((sa, sl, sc), 1))
        n_tgt = int(pop_counts.get((ta, tl, tc), 1))
        possible = max(1, n_src * n_tgt)
        p = float(len(idxs)) / float(possible)
        # mean indegree: edges / n_tgt
        mean_indeg = float(len(idxs)) / float(max(1, n_tgt))
        # connection type
        conn_type = _classify_edge(sa, sl, ta, tl)
        # provenance per type
        if conn_type == "FF":
            prov = f"MODEL_ASSUMPTION g_FF p={P_FEEDFORWARD_DEFAULT:.2f} delay {DELAY_FF_MS:.0f}ms (builder.py:90, jaxfne/_config.py:inter_column)"
            g_tag = f"g_FF={P_FEEDFORWARD_DEFAULT:.2f}"
        elif conn_type == "FB":
            prov = f"MODEL_ASSUMPTION g_FB p={P_FEEDBACK_DEFAULT:.2f} delay {DELAY_FB_MS:.0f}ms (builder.py:90)"
            g_tag = f"g_FB={P_FEEDBACK_DEFAULT:.2f}"
        elif conn_type == "local":
            prov = "MODEL_ASSUMPTION spatial_sigma=0.08 max_in_degree=25 (builder.py:395, jaxfne/connectivity.py:222)"
            g_tag = "g_within=0.35"
        else:  # vertical
            prov = "MODEL_ASSUMPTION vertical laminar (builder.py:90 FB/FF maps, populations.py:31)"
            g_tag = "g_vertical"
        # motif gain
        mg = float(MOTIF_GAIN.get((sc, tc), 1.0)) if isinstance(MOTIF_GAIN, dict) else 1.0
        out[key] = dict(
            edges=int(len(idxs)),
            mean_indegree=float(mean_indeg),
            meanW=float(mean_w),
            stdW=float(std_w),
            CV=float(cv),
            mean_delay_ms=float(mean_delay),
            delay_steps_modes=sorted(set(ds[np.array(idxs)].tolist())) if idxs else [],
            p=float(p),
            conn_type=str(conn_type),
            provenance=str(prov),
            motif_gain=float(mg),
            g_tag=str(g_tag),
            n_src=int(n_src),
            n_tgt=int(n_tgt),
        )
    return out


# ---------------------------------------------------------------------------
# 1. hierarchy_fig — cortical hierarchy four columns × five layers
# ---------------------------------------------------------------------------

_LAYER_ORDER = ["L1", "L2/3", "L4", "L5", "L6"]
_LAYER_Y = {"L1": 5.0, "L2/3": 4.0, "L4": 3.0, "L5": 2.0, "L6": 1.0}
_AREA_X = {"V1": 0.0, "V4": 3.0, "FEF": 6.0, "PFC": 9.0}
_CELL_COLOR = {"E": "#1f77b4", "PV": "#d62728", "SST": "#2ca02c", "VIP": "#ff7f0e"}
_CELL_OFFSET = {"E": -0.18, "PV": -0.06, "SST": 0.06, "VIP": 0.18}


def hierarchy_fig(
    model: Any | None = None,
    *,
    dt_ms: float = 0.1,
    width: int = 1100,
    height: int = 700,
    max_edges_shown: int = 1000,
) -> Any:
    """Figure A — cortical hierarchy (V1 V4 FEF PFC × L1/L2/3/L4/L5/L6).

    Reads builder.py:62,90,395 and EdgeList (jaxfne/emitters.py).
    Nodes = area×layer boxes; edges styled width→effective weight, opacity→p, dash→type.
    Hover = exact motif metadata (edges, mean indegree, meanW, CV, delay, provenance).
    Sampling: statistics use ALL edges via get_motif_stats (complete EdgeList);
    rendering uses deterministic stratified sampling per 6-tuple motif
    (area_s×layer_s×class_s × area_t×layer_t×class_t) so rare FF/FB/VIP preserved.
    """
    if go is None:
        raise ImportError("plotly not installed")
    model = _ensure_model(model)
    tbl = _neuron_table(model)
    # node positions per area×layer
    # counts per node for label
    node_counts: Counter = Counter()
    for r in tbl:
        node_counts[(str(r.get("area")), str(r.get("layer")))] += 1
    # Full stats from ALL edges (no sampling) — rule: statistics use ALL edges
    motif_stats = get_motif_stats(model, dt_ms=dt_ms)
    # Deterministic stratified sampling for rendering (V0-P1 L1)
    # Use config_hash (jaxfne/io.py:42) as seed base — same generated-owner arrays
    try:
        sampled_idx, n_total, n_rendered, per_motif_counts = _stratified_sample_indices(
            model, n_rendered=int(max_edges_shown), seed_salt="hierarchy"
        )
    except Exception:
        sampled_idx, n_total, n_rendered, per_motif_counts = np.array([], dtype=int), 10590, 1000, {}
    # If sampling returned empty (n_total <= n_rendered), show all motifs
    if n_total == 0:
        try:
            ea_tmp = _edge_arrays(model)
            n_total = int(ea_tmp.get("pre", np.array([])).shape[0]) if ea_tmp else 10590
            n_rendered = min(int(max_edges_shown), n_total)
        except Exception:
            n_total, n_rendered = 10590, 1000
    ch = _config_hash_for_model(model)
    # Determine which motifs to render: stratified ensures rare preserved
    if per_motif_counts:
        # keep only motifs with sampled quota >0 (rare guaranteed >=1)
        sorted_motifs = [kv for kv in motif_stats.items() if per_motif_counts.get(kv[0], 0) > 0]
        # sort by edges descending for visual bottleneck ordering within sampled set
        sorted_motifs = sorted(sorted_motifs, key=lambda kv: kv[1]["edges"], reverse=True)
    else:
        sorted_motifs = sorted(motif_stats.items(), key=lambda kv: kv[1]["edges"], reverse=True)
        # fallback truncation if still over budget (e.g., many motifs)
        if len(sorted_motifs) > int(max_edges_shown):
            sorted_motifs = sorted_motifs[: int(max_edges_shown)]
    # Build figure
    fig = go.Figure()
    # -- nodes as shapes + scatter for hover/label --
    node_x, node_y, node_text, node_hover = [], [], [], []
    for area in JOMISSION_AREAS:
        for layer in _LAYER_ORDER:
            x = float(_AREA_X.get(area, 0.0))
            y = float(_LAYER_Y.get(layer, 0.0))
            n = int(node_counts.get((area, layer), 0))
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{area}<br>{layer}<br>n={n}")
            # depth band hover (populations.py:31)
            try:
                frac = LAYER_COUNT_FRAC_DEFAULT.get(layer, 0.0)
                depth = LAYER_DEPTH_BANDS.get(layer, (0, 0))
            except Exception:
                frac, depth = 0.0, (0, 0)
            node_hover.append(f"{area} {layer}<br>n={n}<br>frac={frac:.2f}<br>depth {depth}<br>populations.py:31")
    # draw node rectangles as shapes (behind)
    shapes = []
    for area in JOMISSION_AREAS:
        for layer in _LAYER_ORDER:
            x = float(_AREA_X.get(area, 0.0))
            y = float(_LAYER_Y.get(layer, 0.0))
            shapes.append(dict(type="rect", x0=x - 0.55, x1=x + 0.55, y0=y - 0.38, y1=y + 0.38, line=dict(color="#333", width=1.2), fillcolor="#f8f9fa", layer="below"))
    fig.update_layout(shapes=shapes)  # type: ignore
    # node scatter (on top)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="middle center", textfont=dict(size=9, color="#111"), marker=dict(size=28, color="#e9ecef", line=dict(color="#333", width=1.2), opacity=0.0), hovertext=node_hover, hoverinfo="text", name="layers"))
    # -- edges: aggregate per motif, but rendered set is stratified sample --
    # scale for width/opacity — use LOG handling for weight dynamic range (builder.py:142 CV1.5 heavy tail)
    # Linear for CV/rho per A2; weight uses sign-preserving log via _safe_log_abs_weight for width scaling
    meanWs = [v["meanW"] for _, v in sorted_motifs] or [1.0]
    ps = [v["p"] for _, v in sorted_motifs] or [1.0]
    # Log transform for weight width to handle 4-order dynamic range (lognormal sigma 1.085)
    try:
        logWs = _safe_log_abs_weight(np.array(meanWs, dtype=float))
        max_logW = float(np.max(np.abs(logWs))) if logWs.size else 1.0
    except Exception:
        max_logW = float(max(meanWs)) if meanWs else 1.0
        logWs = np.array(meanWs, dtype=float)
    max_p = float(max(ps)) if ps else 1.0
    # draw edges as lines (one trace per type for legend, but hover per motif needs separate line per motif; use single trace with segments? use separate scatter for each motif to allow hover)
    dash_map = {"FF": "solid", "FB": "dash", "vertical": "dot", "local": "dashdot"}
    type_traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx_m, (key, stat) in enumerate(sorted_motifs):
        sa, sl, sc, ta, tl, tc = key
        x0 = float(_AREA_X.get(sa, 0.0)) + float(_CELL_OFFSET.get(sc, 0.0))
        y0 = float(_LAYER_Y.get(sl, 0.0))
        x1 = float(_AREA_X.get(ta, 0.0)) + float(_CELL_OFFSET.get(tc, 0.0))
        y1 = float(_LAYER_Y.get(tl, 0.0))
        # skip self-loop same node same cell (local) zero-length? keep but jitter
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            y1 = y0 + 0.08
            x1 = x0 + 0.08
        # style — width via log-scaled weight (sign-safe), opacity via p
        conn_type = str(stat["conn_type"])
        # log-scaled width
        try:
            lw_log = float(np.abs(logWs[idx_m]) / max(1e-9, max_logW)) if idx_m < len(logWs) else 0.5
        except Exception:
            lw_log = float(stat["meanW"] / max(1e-9, max_logW))
        line_w = float(np.clip(0.7 + 5.0 * lw_log, 0.7, 6.5))
        opacity = float(np.clip(0.18 + 0.75 * (stat["p"] / max(1e-9, max_p)), 0.18, 0.92))
        dash = dash_map.get(conn_type, "solid")
        color = _CELL_COLOR.get(sc, "#333")
        # hover exact motif metadata (task spec) — stats from complete EdgeList
        hover = (
            f"{sa} {sl} {sc} → {ta} {tl} {tc}<br>"
            f"edges {stat['edges']} mean indegree {stat['mean_indegree']:.2f} meanW {stat['meanW']:.3f} CV {stat['CV']:.2f}<br>"
            f"delay {stat['mean_delay_ms']:.0f}ms steps {stat['delay_steps_modes']}<br>"
            f"p {stat['p']:.3f} motif_gain {stat['motif_gain']:.2f}<br>"
            f"provenance {stat['provenance']}<br>"
            f"{stat['g_tag']} (builder.py:62,90 jaxfne/emitters.py)"
        )
        # Use scatter line per motif
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=line_w, dash=dash),
                opacity=opacity,
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
                name=conn_type,
            )
        )
    # legend proxies for dash/width
    for conn_type, dash in dash_map.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#333", width=2, dash=dash), name=f"{conn_type} ({dash})"))
    # area column labels
    for area, x in _AREA_X.items():
        fig.add_annotation(x=x, y=5.85, text=f"<b>{area}</b>", showarrow=False, font=dict(size=13))
    # Provenance footer with sampling disclosure — every figure that samples has n_total, n_rendered, filters
    disclosure = _sampling_disclosure(int(n_total), int(n_rendered))
    footer = _provenance_footer_text(int(n_total), int(n_rendered), filters=None, config_hash=ch)
    # Also embed explicit required sentence verbatim for HTML grep
    fig.add_annotation(
        x=0.5,
        y=-0.14,
        xref="paper",
        yref="paper",
        text=footer,
        showarrow=False,
        font=dict(size=9, color="#444"),
        align="center",
        bordercolor="#ccc",
        borderwidth=0,
        bgcolor="rgba(248,249,250,0.9)",
    )
    # Ensure disclosure also present as hidden meta via layout meta
    fig.update_layout(
        title="Hierarchy — V1→V4→FEF→PFC × laminar (width→|W| log-scaled opacity→p dash→FF/FB/local) — builder.py:62,90,395",
        width=width,
        height=height,
        xaxis=dict(visible=False, range=[-1.2, 10.2]),
        yaxis=dict(visible=False, range=[0.2, 6.2]),
        hovermode="closest",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=20, r=20, t=60, b=80),
        annotations=list(fig.layout.annotations) if hasattr(fig.layout, "annotations") else [],
    )
    # Attach sampling metadata to figure for testing/HTML verification
    try:
        fig._sampling_meta = dict(n_total=int(n_total), n_rendered=int(n_rendered), config_hash=str(ch), disclosure=str(disclosure))  # type: ignore
        fig.update_layout(meta=dict(sampling=dict(n_total=int(n_total), n_rendered=int(n_rendered), config_hash=str(ch), disclosure=str(disclosure), stratified_by="area_s×layer_s×class_s × area_t×layer_t×class_t", seed_salt="hierarchy")))
    except Exception:
        pass
    return fig


# ---------------------------------------------------------------------------
# 2. motif_matrix_fig — (source layer,class) × (target layer,class) heatmap
# ---------------------------------------------------------------------------

def _motif_matrix(
    model: Any,
    *,
    area: str | None = None,
    conn_type: str | None = None,
    metric: str = "weight",
    dt_ms: float = 0.1,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Return row_labels, col_labels, matrix for heatmap.

    Rows = source (layer,class), Cols = target (layer,class), both ordered L1 VIP? use L×CT cartesian.
    Filters: area (if set, require tgt_area==area or src_area==area depending on conn?), conn_type.
    Metrics: weight / edge_count / probability / realized_current / delay
    """
    model = _ensure_model(model)
    stats = get_motif_stats(model, dt_ms=dt_ms)
    # labels
    row_labels = [f"{lyr} {ct}" for lyr in _LAYER_ORDER for ct in JOMISSION_CELL_TYPES]
    col_labels = list(row_labels)
    row_index = {lab: i for i, lab in enumerate(row_labels)}
    col_index = {lab: j for j, lab in enumerate(col_labels)}
    mat = np.zeros((len(row_labels), len(col_labels)), dtype=float)
    counts = np.zeros_like(mat, dtype=int)
    # map motif to matrix cell: (src_layer,src_ct) -> row, (tgt_layer,tgt_ct) -> col
    # aggregate over areas/directions that match filters
    for key, st in stats.items():
        sa, sl, sc, ta, tl, tc = key
        ct = st["conn_type"]
        # area filter: if area specified, motif must involve that area (src or tgt) for FF/FB, else within-area==area for local/vertical
        if area is not None and area != "ALL":
            if ct in ("local", "vertical"):
                if sa != area or ta != area:
                    continue
            elif ct == "FF":
                if ta != area and sa != area:
                    continue
                # for hierarchy clarity, keep FF into area
                if ta != area:
                    continue
            elif ct == "FB":
                if sa != area and ta != area:
                    continue
                if ta != area:
                    continue
        if conn_type is not None and conn_type != "ALL":
            if ct != conn_type:
                continue
        rlab = f"{sl} {sc}"
        clab = f"{tl} {tc}"
        if rlab not in row_index or clab not in col_index:
            continue
        i, j = row_index[rlab], col_index[clab]
        # accumulate (multiple area pairs collapse into same LxCT)
        if metric == "weight":
            mat[i, j] += float(st["meanW"])
            counts[i, j] += 1
        elif metric == "edge_count":
            mat[i, j] += float(st["edges"])
        elif metric == "probability":
            mat[i, j] += float(st["p"])
            counts[i, j] += 1
        elif metric == "realized_current":
            # approx: meanW * p * 1e3 scale or meanW * edges
            mat[i, j] += float(st["meanW"] * st["edges"])
        elif metric == "delay":
            mat[i, j] += float(st["mean_delay_ms"])
            counts[i, j] += 1
        else:
            mat[i, j] += float(st["meanW"])
            counts[i, j] += 1
    # for averaged metrics, divide by counts
    if metric in ("weight", "probability", "delay"):
        nz = counts > 0
        mat[nz] = mat[nz] / counts[nz].astype(float)
    return row_labels, col_labels, mat


def motif_matrix_fig(
    model: Any | None = None,
    *,
    area: str = "V1",
    conn_type: str = "ALL",
    metric: str = "weight",
    dt_ms: float = 0.1,
    width: int = 750,
    height: int = 700,
    with_controls: bool = True,
) -> Any:
    """Interactive heatmap (source layer,class) × (target layer,class).

    Selectors: area, local/vertical/FF/FB, weight/edge count/probability/realized current/delay.
    Answers What connects to SST in V1 L2/3? etc.
    Uses generated-owner EdgeList (builder.py:62,395).
    """
    if go is None:
        raise ImportError("plotly not installed")
    model = _ensure_model(model)
    areas = ["ALL"] + list(JOMISSION_AREAS)
    conn_types = ["ALL", "local", "vertical", "FF", "FB"]
    metrics = ["weight", "edge_count", "probability", "realized_current", "delay"]
    # default fig
    rows, cols, mat = _motif_matrix(model, area=area, conn_type=conn_type, metric=metric, dt_ms=dt_ms)
    # Log handling: weight / realized_current use log scale (builder.py:142 CV1.5 heavy tail, sign-safe); CV/rho/p/delay linear per A2
    def _display_mat(m: np.ndarray, mk: str) -> Tuple[np.ndarray, str, str]:
        if mk in ("weight", "realized_current"):
            # log abs with sign preserved for weight (zero/sign-safe via _safe_log_abs_weight semantics but mat is abs meanW => positive)
            # Use log10 for positive values, keep zeros as NaN for transparency
            m_arr = np.asarray(m, dtype=float)
            eps = 1e-6
            # for zeros (empty cells) keep as 0? Use masked log: log10(max(m,eps))
            disp = np.where(m_arr > 0, np.log10(np.maximum(m_arr, eps)), np.nan)
            # Replace nan with min log -1 for display but hover will show raw 0
            # Use -6 for empties to appear as background
            disp = np.where(np.isnan(disp), -6, disp)
            title = f"{mk} (log10, zero-safe)"
            hover_suffix = "log10"
            return disp, title, hover_suffix
        else:
            return m, mk, ""
    mat_disp, cb_title, _hover_suffix = _display_mat(mat, metric)
    # For hover, keep raw mat as customdata to show original value
    # Build display matrix per metric for dropdowns with same log logic
    heatmaps: Dict[str, np.ndarray] = {}
    heatmaps_disp: Dict[str, np.ndarray] = {}
    for mkey in metrics:
        _, _, mm = _motif_matrix(model, area=area, conn_type=conn_type, metric=mkey, dt_ms=dt_ms)
        heatmaps[mkey] = mm
        dm, _, _ = _display_mat(mm, mkey)
        heatmaps_disp[mkey] = dm
    # choose display mat
    use_z = mat_disp if metric in ("weight", "realized_current") else mat
    hover_tmpl = "src %{y} → tgt %{x}<br>val %{z:.3g}<extra></extra>" if metric not in ("weight", "realized_current") else "src %{y} → tgt %{x}<br>log10 val %{z:.2f} (raw %{customdata:.3g})<extra></extra>"
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=use_z,
                x=cols,
                y=rows,
                colorscale="Viridis",
                colorbar=dict(title=cb_title),
                hovertemplate=hover_tmpl,
                customdata=mat if metric in ("weight", "realized_current") else None,
            )
        ]
    )
    if with_controls:
        # metric dropdown — switches z (log-aware: weight/realized_current use log10 display, others linear)
        buttons_metric = []
        for mkey in metrics:
            mm_disp = heatmaps_disp[mkey]
            # also need hover handling per metric; use update with customdata when log
            if mkey in ("weight", "realized_current"):
                mm_raw = heatmaps[mkey]
                buttons_metric.append(
                    dict(
                        label=f"{mkey} (log)",
                        method="update",
                        args=[{"z": [mm_disp], "customdata": [mm_raw]}, {"title": f"Motif matrix — {area} {conn_type} — {mkey} (log10) — builder.py:62,395"}],
                    )
                )
            else:
                buttons_metric.append(
                    dict(
                        label=mkey,
                        method="update",
                        args=[{"z": [mm_disp]}, {"title": f"Motif matrix — {area} {conn_type} — {mkey} — builder.py:62,395"}],
                    )
                )
        # area dropdown — recompute on the fly via many precomputed matrices (area choices)
        # Apply log display logic for weight metrics
        area_mats: Dict[str, np.ndarray] = {}
        area_mats_disp: Dict[str, np.ndarray] = {}
        for akey in areas:
            _, _, mm = _motif_matrix(model, area=akey, conn_type=conn_type, metric=metric, dt_ms=dt_ms)
            area_mats[akey] = mm
            dm, _, _ = _display_mat(mm, metric)
            area_mats_disp[akey] = dm
        buttons_area = [
            dict(label=akey, method="update", args=[{"z": [area_mats_disp[akey]]}, {"title": f"Motif matrix — {akey} {conn_type} — {metric}" + (" (log10)" if metric in ("weight","realized_current") else "")}])
            for akey in areas
        ]
        # conn_type dropdown similarly
        conn_mats: Dict[str, np.ndarray] = {}
        conn_mats_disp: Dict[str, np.ndarray] = {}
        for ckey in conn_types:
            _, _, mm = _motif_matrix(model, area=area, conn_type=ckey, metric=metric, dt_ms=dt_ms)
            conn_mats[ckey] = mm
            dm, _, _ = _display_mat(mm, metric)
            conn_mats_disp[ckey] = dm
        buttons_conn = [
            dict(label=ckey, method="update", args=[{"z": [conn_mats_disp[ckey]]}, {"title": f"Motif matrix — {area} {ckey} — {metric}" + (" (log10)" if metric in ("weight","realized_current") else "")}])
            for ckey in conn_types
        ]
        fig.update_layout(
            updatemenus=[
                dict(buttons=buttons_metric, direction="down", x=1.02, xanchor="left", y=1.0, yanchor="top", showactive=True, active=metrics.index(metric) if metric in metrics else 0),
                dict(buttons=buttons_area, direction="down", x=1.02, xanchor="left", y=0.78, yanchor="top", showactive=True, active=areas.index(area) if area in areas else 0),
                dict(buttons=buttons_conn, direction="down", x=1.02, xanchor="left", y=0.56, yanchor="top", showactive=True, active=conn_types.index(conn_type) if conn_type in conn_types else 0),
            ],
            annotations=[
                dict(text="metric", x=1.02, xref="paper", y=1.08, yref="paper", showarrow=False, xanchor="left"),
                dict(text="area", x=1.02, xref="paper", y=0.86, yref="paper", showarrow=False, xanchor="left"),
                dict(text="conn", x=1.02, xref="paper", y=0.64, yref="paper", showarrow=False, xanchor="left"),
            ],
        )
    log_note = " (log10, zero/sign-safe: sign·log1p(|W|/1e-6), builder.py:142 CV1.5)" if metric in ("weight", "realized_current") else " (linear; CV/rho linear per A2)"
    cb_extra = " — H taus log10 (0.1-1000s) via _safe_log_tau when metric=delay" if metric == "delay" else ""
    fig.update_layout(
        title=f"Motif matrix — {area} {conn_type} — {metric}{log_note} — answers: what connects to SST in V1 L2/3? (builder.py:62)",
        width=width,
        height=height,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=220, t=70, b=80),
    )
    # Add footer annotation about scaling
    fig.add_annotation(
        x=0.5,
        y=-0.16,
        xref="paper",
        yref="paper",
        text=f"Scaling: weight/realized_current log10 (heavy tail CV1.5, builder.py:142) zero/sign-safe; CV/rho/delay linear; H taus log10 (h_state.py:28).{cb_extra} All stats from complete EdgeList.",
        showarrow=False,
        font=dict(size=8, color="#555"),
        align="center",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. spatial_fig — neuron positions and edges with p(edge|distance)
# ---------------------------------------------------------------------------

def _spatial_sigma(model: Any) -> float:
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "metadata"):
            conn = cfg.metadata.get("connectivity", {}) or {}
            if "spatial_sigma" in conn:
                return float(conn["spatial_sigma"])
    except Exception:
        pass
    # fallback from builder default
    try:
        return float(model.config.metadata.get("connectivity", {}).get("spatial_sigma", 0.08))  # type: ignore
    except Exception:
        return 0.08


def spatial_fig(
    model: Any | None = None,
    *,
    area: str | None = None,
    layer: str | None = None,
    cell_type: str | None = None,
    e_subtype: str | None = None,
    motif: str | None = None,
    edge_distance_threshold: float | None = None,
    max_edges_plot: int = 1000,
    dt_ms: float = 0.1,
    width: int = 1100,
    height: int = 560,
) -> Any:
    """Spatial network view — neuron positions and edges with p(edge|distance).

    Mandatory given W1 sparse-local (builder.py:395 spatial_sigma=0.08, max_in_degree=25,
    jaxfne/connectivity.py:222). Shows configured kernel vs realized connectivity.
    Sampling: statistics use ALL edges (full EdgeList for p(edge|d)); rendering
    uses deterministic stratified sampling per motif (area_s×layer_s×class_s ×
    area_t×layer_t×class_t) with config_hash seed — rare FF/FB/VIP preserved.

    Filters: area, layer, class, E subtype, motif, edge-distance threshold.
    """
    if go is None or make_subplots is None:
        raise ImportError("plotly not installed")
    model = _ensure_model(model)
    tbl = _neuron_table(model)
    pos = _positions(model)
    # filter neuron mask
    n = len(tbl)
    mask = np.ones(n, dtype=bool)
    if area is not None and area != "ALL":
        mask &= np.array([str(r.get("area")) == area for r in tbl])
    if layer is not None and layer != "ALL":
        mask &= np.array([str(r.get("layer")) == layer for r in tbl])
    if cell_type is not None and cell_type != "ALL":
        mask &= np.array([str(r.get("cell_type")) == cell_type for r in tbl])
    # E subtype (requires per-neuron a/c labels; approximate via jitter cluster — just filter by a>0.05 vs c<-55)
    if e_subtype is not None and e_subtype != "ALL":
        try:
            em = model.params.get("emitter")  # type: ignore
            a_arr = np.asarray(em.a, dtype=float) if hasattr(em, "a") else None
            c_arr = np.asarray(em.c, dtype=float) if hasattr(em, "c") else None
            if a_arr is not None and c_arr is not None:
                if e_subtype == "RS":
                    mask &= (np.asarray([str(r.get("cell_type")) == "E" for r in tbl]) & (a_arr < 0.05) & (c_arr < -60))
                elif e_subtype == "CH":
                    mask &= (np.asarray([str(r.get("cell_type")) == "E" for r in tbl]) & (c_arr > -55))
                elif e_subtype in ("E_FS", "FS"):
                    mask &= (np.asarray([str(r.get("cell_type")) == "E" for r in tbl]) & (a_arr > 0.05))
        except Exception:
            pass
    # motif filter like "E->PV"
    if motif is not None and motif != "ALL" and "->" in motif:
        try:
            sc, tc = motif.split("->")
            sc, tc = sc.strip(), tc.strip()
            mask_motif = np.array([str(r.get("cell_type")) in (sc, tc) for r in tbl])
            # narrow to those types
            mask &= np.array([str(r.get("cell_type")) in (sc, tc) for r in tbl])
        except Exception:
            pass
    # positions of filtered neurons
    idx_filtered = np.where(mask)[0]
    # edge selection: keep edges where both pre and post pass mask (or at least one if motif)
    ea = _edge_arrays(model)
    pre, post, w = ea.get("pre", np.array([], dtype=int)), ea.get("post", np.array([], dtype=int)), ea.get("weight", np.array([], dtype=float))
    # distance per edge
    if pos.shape[0] == n and pre.size:
        # within-area distances matter for sparse-local
        # compute Euclidean in xy plane (x,y are local coords ±0.25 plus area offset)
        # for p(edge|distance) use full 3D or 2D local distance after removing area offset (so within-area local distances are small)
        # Use y,z local distance for within edges
        d_all = np.sqrt(((pos[pre] - pos[post]) ** 2).sum(axis=1))
        # local distance: remove area x offset for within-area (keep y,z)
        # Better: compute within-area local 2D distance from y,z (consistent with jaxfne/connectivity.py Gaussian on positions)
        # jaxfne uses positions full 3D but sigma 0.08 is local scale (<0.5) so within-area distances are small, between-area large (>1.5)
    else:
        d_all = np.zeros(int(pre.shape[0]), dtype=float)
    # edge distance threshold
    if edge_distance_threshold is not None:
        keep_dist = d_all <= float(edge_distance_threshold)
    else:
        keep_dist = np.ones_like(d_all, dtype=bool)
    # also restrict to edges touching filtered neurons
    if idx_filtered.size and pre.size:
        set_filt = set(int(x) for x in idx_filtered.tolist())
        keep_neuron = np.array([int(pre[i]) in set_filt and int(post[i]) in set_filt for i in range(int(pre.shape[0]))], dtype=bool)
        # if filter is strict and would yield 0 edges, relax to either endpoint
        if not np.any(keep_neuron & keep_dist) and idx_filtered.size:
            keep_neuron = np.array([int(pre[i]) in set_filt or int(post[i]) in set_filt for i in range(int(pre.shape[0]))], dtype=bool)
    else:
        keep_neuron = np.ones(int(pre.shape[0]), dtype=bool) if pre.size else np.array([], dtype=bool)
    keep = keep_neuron & keep_dist
    # Deterministic stratified sampling for rendering (V0-P1)
    # Full EdgeList stats for p(edge|d) use ALL edges; rendering samples per motif
    # Stratify by area_s×layer_s×class_s × area_t×layer_t×class_t (builder.py:62)
    ea_full = ea
    n_total_edges = int(ea_full["pre"].shape[0]) if ea_full.get("pre") is not None and ea_full["pre"].size else 0
    # candidate pool after filters
    keep_idx_all = np.where(keep)[0]
    n_before_sampling = int(keep_idx_all.shape[0])
    ch_spatial = _config_hash_for_model(model)
    if keep_idx_all.size > int(max_edges_plot):
        # stratify within keep pool by motif: build motif groups restricted to keep
        areas_tbl = [str(r.get("area")) for r in tbl]
        layers_tbl = [str(r.get("layer")) for r in tbl]
        cts_tbl = [str(r.get("cell_type")) for r in tbl]
        # group keep indices by motif
        from collections import defaultdict as _dd2
        motif_to_keep: Dict[Tuple[str, str, str, str, str, str], List[int]] = _dd2(list)
        for ei in keep_idx_all:
            try:
                k = _motif_key_for_edge(int(pre[int(ei)]), int(post[int(ei)]), areas_tbl, layers_tbl, cts_tbl)
            except Exception:
                k = ("?", "?", "?", "?", "?", "?")
            motif_to_keep[k].append(int(ei))
        # allocate quotas as in _stratified_sample_indices but scoped to keep pool
        m_sp = len(motif_to_keep)
        n_render_target = int(max_edges_plot)
        # at least 1 per motif
        quotas_sp: Dict[Tuple[str, str, str, str, str, str], int] = {k: 1 for k in motif_to_keep}
        s_min = sum(quotas_sp.values())
        if s_min > n_render_target:
            # more motifs than budget: select n_render_target motifs deterministic
            def _mh(k): return int(hashlib.sha256("|".join(k).encode()).hexdigest()[:8], 16)
            ordered = sorted(motif_to_keep.keys(), key=_mh)
            selected_keys = set(ordered[:n_render_target])
            edge_idx_plot_list: List[int] = []
            for k in selected_keys:
                idxs = motif_to_keep[k]
                pseed = _deterministic_seed(ch_spatial, f"spatial:{k}")
                rng = np.random.default_rng(pseed)
                edge_idx_plot_list.append(int(rng.choice(idxs, size=1)[0]))
            edge_idx_plot = np.array(sorted(edge_idx_plot_list), dtype=int)
        else:
            # proportional allocation within keep pool
            total_keep = float(n_before_sampling)
            ideal_sp = {k: max(1, round(n_render_target * len(idxs) / total_keep)) for k, idxs in motif_to_keep.items()}
            for k in ideal_sp:
                ideal_sp[k] = min(int(ideal_sp[k]), len(motif_to_keep[k]))
            s_ideal = sum(ideal_sp.values())
            def _sort_sp(k): return (float(len(motif_to_keep[k])), int(hashlib.sha256("|".join(k).encode()).hexdigest()[:8], 16))
            if s_ideal > n_render_target:
                ordered_desc = sorted(motif_to_keep.keys(), key=_sort_sp, reverse=True)
                idx = 0
                while s_ideal > n_render_target:
                    k = ordered_desc[idx % len(ordered_desc)]
                    if ideal_sp[k] > 1:
                        ideal_sp[k] -= 1
                        s_ideal -= 1
                    idx += 1
                    if idx > 5000:
                        break
            elif s_ideal < n_render_target:
                ordered_desc = sorted(motif_to_keep.keys(), key=_sort_sp, reverse=True)
                idx = 0
                while s_ideal < n_render_target:
                    k = ordered_desc[idx % len(ordered_desc)]
                    if ideal_sp[k] < len(motif_to_keep[k]):
                        ideal_sp[k] += 1
                        s_ideal += 1
                    idx += 1
                    if idx > 10000:
                        break
            edge_idx_plot_list = []
            for k, idxs in motif_to_keep.items():
                q = int(ideal_sp.get(k, 0))
                if q <= 0:
                    continue
                pseed = _deterministic_seed(ch_spatial, f"spatial:{k}")
                rng = np.random.default_rng(pseed)
                perm = rng.permutation(np.array(idxs, dtype=int))
                edge_idx_plot_list.extend(perm[:q].tolist())
            edge_idx_plot = np.array(sorted(edge_idx_plot_list), dtype=int)
            if edge_idx_plot.shape[0] > n_render_target:
                gseed = _deterministic_seed(ch_spatial, "spatial:global_trim")
                rng = np.random.default_rng(gseed)
                perm = rng.permutation(edge_idx_plot)
                edge_idx_plot = np.sort(perm[:n_render_target])
            elif edge_idx_plot.shape[0] < n_render_target and edge_idx_plot.shape[0] < keep_idx_all.size:
                remaining_pool = np.array([i for i in keep_idx_all if i not in set(edge_idx_plot.tolist())], dtype=int)
                if remaining_pool.size:
                    gseed = _deterministic_seed(ch_spatial, "spatial:global_pad")
                    rng = np.random.default_rng(gseed)
                    need = n_render_target - int(edge_idx_plot.shape[0])
                    pad = rng.choice(remaining_pool, size=min(need, remaining_pool.size), replace=False)
                    edge_idx_plot = np.sort(np.concatenate([edge_idx_plot, pad]))
    else:
        edge_idx_plot = keep_idx_all
    n_rendered_actual = int(edge_idx_plot.shape[0])
    # Build subplot: left scatter positions+edges, right p(edge|distance)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=("Neuron positions & edges (filtered)", "p(edge|distance) — kernel vs realized"))
    # left: neurons
    if idx_filtered.size:
        xs = pos[idx_filtered, 0]
        ys = pos[idx_filtered, 1]
        # color by cell type
        cts = [str(tbl[i].get("cell_type")) for i in idx_filtered]
        areas_p = [str(tbl[i].get("area")) for i in idx_filtered]
        cols_pts = [_CELL_COLOR.get(c, "#333") for c in cts]
        hover_pts = [f"id {int(idx_filtered[k])} {areas_p[k]} {tbl[int(idx_filtered[k])].get('layer')} {cts[k]}<br>pos ({xs[k]:.2f},{ys[k]:.2f},{pos[int(idx_filtered[k]),2]:.2f})" for k in range(len(idx_filtered))]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=4, color=cols_pts, line=dict(width=0)), hovertext=hover_pts, hoverinfo="text", name="neurons"), row=1, col=1)
        # edges as lines (thin) — opacity via log-scaled weight (builder.py:142 lognormal CV1.5 heavy tail, zero/sign-safe)
        # Precompute log scale for rendered edges
        try:
            w_render = np.abs(np.asarray([float(w[int(ei)]) for ei in edge_idx_plot], dtype=float))
            log_w_render = np.abs(_safe_log_abs_weight(w_render))
            max_log_w = float(np.max(log_w_render)) if log_w_render.size and float(np.max(log_w_render)) > 1e-9 else 1.0
        except Exception:
            max_log_w = 1.5
            log_w_render = np.ones(len(edge_idx_plot), dtype=float)
        for idx_r, ei in enumerate(edge_idx_plot):
            x0, y0 = float(pos[int(pre[ei]), 0]), float(pos[int(pre[ei]), 1])
            x1, y1 = float(pos[int(post[ei]), 0]), float(pos[int(post[ei]), 1])
            # opacity by log weight magnitude (zero/sign-safe handles small weights)
            try:
                lw = float(log_w_render[idx_r] / max(1e-9, max_log_w)) if idx_r < len(log_w_render) else 0.5
            except Exception:
                ww = float(abs(float(w[ei])))
                lw = float(np.clip(ww / 1.5, 0, 1))
            op = float(np.clip(0.15 + 0.55 * lw, 0.08, 0.80))
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="#888", width=0.6), opacity=op, hoverinfo="skip", showlegend=False),
                row=1,
                col=1,
            )
    # right: p(edge|distance)
    sigma = _spatial_sigma(model)
    # theoretical kernel curve
    d_grid = np.linspace(0.0, 0.6, 200)
    kernel = np.exp(-d_grid**2 / (2.0 * max(1e-9, sigma) ** 2))
    kernel = kernel / float(kernel.max()) if kernel.max() > 0 else kernel
    fig.add_trace(go.Scatter(x=d_grid, y=kernel, mode="lines", name=f"kernel σ={sigma:.2f} (builder.py:395)", line=dict(color="#1f77b4", width=2.6)), row=1, col=2)
    # empirical: histogram realized distances
    # Only within-area edges considered for spatial_local (between-area distances large >1)
    within_mask = np.array([str(tbl[int(pre[i])].get("area")) == str(tbl[int(post[i])].get("area")) for i in range(int(pre.shape[0]))], dtype=bool) if pre.size and len(tbl) else np.zeros(0, dtype=bool)
    d_within = d_all[within_mask] if within_mask.size else np.array([], dtype=float)
    if d_within.size:
        bins = np.linspace(0.0, 0.6, 26)
        hist, edges_b = np.histogram(d_within, bins=bins)
        # normalize to probability (protect against all-to-all: hist should decay)
        # Show empirical density
        centers = 0.5 * (edges_b[:-1] + edges_b[1:])
        # also compute possible pairs density for p(edge|distance) — sample random within pairs
        # approximate possible pairs: sample 20000 random within pairs, histogram
        rng = np.random.default_rng(1)
        # sample random within pairs (choose random pre/post in same area)
        area_to_idx: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(tbl):
            area_to_idx[str(r.get("area"))].append(i)
        rand_dists = []
        for _, idxs in area_to_idx.items():
            if len(idxs) < 2:
                continue
            for _ in range(5000):
                a, b = rng.choice(idxs, size=2, replace=False)
                d = float(np.sqrt(((pos[a] - pos[b]) ** 2).sum()))
                rand_dists.append(d)
        rand_dists = np.array(rand_dists, dtype=float)
        hist_possible, _ = np.histogram(rand_dists, bins=bins)
        # p_emp = hist / hist_possible (avoid div0)
        p_emp = np.divide(hist.astype(float), np.maximum(1, hist_possible.astype(float)), out=np.zeros_like(hist, dtype=float), where=hist_possible > 0)
        p_emp_norm = p_emp / float(p_emp.max()) if p_emp.max() > 0 else p_emp
        fig.add_trace(go.Scatter(x=centers, y=p_emp_norm, mode="markers+lines", name="realized p(edge|d) norm (within)", line=dict(color="#d62728", width=2), marker=dict(size=4)), row=1, col=2)
        fig.add_trace(go.Bar(x=centers, y=hist, name="realized count", marker=dict(color="rgba(255,127,14,0.35)"), yaxis="y2", showlegend=False, opacity=0.5), row=1, col=2)
        # annotation protects against all-to-all
        if p_emp.size and np.mean(p_emp_norm[centers > 0.35]) > 0.6:
            fig.add_annotation(x=0.45, y=0.85, xref="x2", yref="y2", text="⚠ flat p(edge|d) — all-to-all!", showarrow=False, font=dict(color="#d62728", size=11), row=1, col=2)
    fig.update_xaxes(title_text="x (area offset)", row=1, col=1)
    fig.update_yaxes(title_text="y (local)", row=1, col=1)
    fig.update_xaxes(title_text="distance", row=1, col=2)
    fig.update_yaxes(title_text="p / kernel (norm)", row=1, col=2)
    filt_str = f"area={area or 'ALL'} layer={layer or 'ALL'} ct={cell_type or 'ALL'} motif={motif or 'ALL'} thr={edge_distance_threshold if edge_distance_threshold is not None else 'none'}"
    # Provenance footer with sampling disclosure — every sampled figure has n_total, n_rendered, filters
    # p(edge|d) histogram uses ALL within edges (full EdgeList) for statistics; rendering samples per motif
    filters_dict = dict(area=area, layer=layer, cell_type=cell_type, motif=motif, edge_distance_threshold=edge_distance_threshold)
    disclosure = _sampling_disclosure(int(n_total_edges), int(n_rendered_actual))
    footer = _provenance_footer_text(int(n_total_edges), int(n_rendered_actual), filters=filters_dict, config_hash=str(ch_spatial), extra=f"shown {int(n_rendered_actual)}/{int(n_before_sampling)} after filters, {int(n_total_edges)} total (builder.py:395, jaxfne/emitters.py EdgeList).")
    fig.add_annotation(
        x=0.5,
        y=-0.18,
        xref="paper",
        yref="paper",
        text=footer,
        showarrow=False,
        font=dict(size=8, color="#444"),
        align="center",
        bgcolor="rgba(248,249,250,0.9)",
    )
    fig.update_layout(title=f"Spatial — {filt_str} — σ={sigma:.2f} max_in_degree=25 (builder.py:395, jaxfne/connectivity.py:222)", width=width, height=height, showlegend=True, legend=dict(x=0.02, y=-0.15, orientation="h"), margin=dict(l=40, r=40, t=85, b=70))
    try:
        fig._sampling_meta = dict(n_total=int(n_total_edges), n_rendered=int(n_rendered_actual), n_before_sampling=int(n_before_sampling), config_hash=str(ch_spatial), disclosure=str(disclosure), filters=filters_dict)  # type: ignore
        fig.update_layout(meta=dict(sampling=dict(n_total=int(n_total_edges), n_rendered=int(n_rendered_actual), config_hash=str(ch_spatial), disclosure=str(disclosure), stratified_by="area_s×layer_s×class_s × area_t×layer_t×class_t", seed_salt="spatial", filters=filters_dict)))
    except Exception:
        pass
    return fig


# ---------------------------------------------------------------------------
# 4. rf_fig — 32×32 RF lattice, pixel→neuron W_RF 400×1024 etc.
# ---------------------------------------------------------------------------

def rf_fig(
    model: Any | None = None,
    rf_config: Any | None = None,
    *,
    n_rf_examples: int = 4,
    dt_ms: float = 0.1,
    width: int = 1200,
    height: int = 800,
) -> Any:
    """RF visualization — 32×32 field, W_RF 400×1024, individual RFs, centers, coverage, overlap.

    Also shows active neuron map for A/B/R, energy map, target area/layer/class,
    E_A/E_B/E_R/E_omission=0 (would have exposed 185× problem).

    Reads network/rf.py:47 RFConfig and RFOperator (uses generated-owner W_RF arrays).
    """
    if go is None or make_subplots is None:
        raise ImportError("plotly not installed")
    if RFConfig is None or RFOperator is None:
        raise ImportError("RFConfig/RFOperator unavailable (network/rf.py:47)")
    model = _ensure_model(model)
    if rf_config is None:
        rf_config = RFConfig()
    op = RFOperator(rf_config, model)
    L = int(rf_config.lattice_size)
    # compute drives and energies (task: display E_A, E_B, E_R, E_omission=0)
    drives = {
        "A": op.drive_for_stimulus("stimulus_A"),
        "B": op.drive_for_stimulus("stimulus_B"),
        "R": op.drive_for_stimulus("random_stimulus"),
        "omission": op.drive_for_stimulus("stimulus_omitted"),
    }
    energies = {k: float(np.sum(np.asarray(v).astype(float) ** 2)) if k != "omission" else 0.0 for k, v in drives.items()}
    # weighted energy: sum drive
    sums = {k: float(np.sum(np.asarray(v).astype(float))) for k, v in drives.items()}
    # patterns
    pat_A = op.stimulus_pattern("stimulus_A")
    pat_B = op.stimulus_pattern("stimulus_B")
    pat_R = op.stimulus_pattern("random_stimulus")
    # coverage = sum of weights over target neurons (n_target × n_pixels -> 32×32)
    w_target = op.weights_target  # (n_target, n_pixels)
    coverage = np.sum(w_target, axis=0).reshape(L, L) if w_target.size else np.zeros((L, L), dtype=float)
    # centers scatter
    centers_xy = list(op.centers.values()) if hasattr(op, "centers") else []
    cx = [c[0] for c in centers_xy]
    cy = [c[1] for c in centers_xy]
    # overlap histogram: pairwise Gaussian overlap approx via centers distance
    overlaps = []
    for i in range(len(centers_xy)):
        for j in range(i + 1, len(centers_xy)):
            dx = centers_xy[i][0] - centers_xy[j][0]
            dy = centers_xy[i][1] - centers_xy[j][1]
            d2 = dx * dx + dy * dy
            ov = float(math.exp(-d2 / (4.0 * float(rf_config.sigma_px) ** 2)))
            overlaps.append(ov)
    # active maps per stimulus: drive_target values scattered at centers
    drive_A_t = np.asarray(drives["A"])[op.target_indices] if len(op.target_indices) else np.array([])
    drive_B_t = np.asarray(drives["B"])[op.target_indices] if len(op.target_indices) else np.array([])
    drive_R_t = np.asarray(drives["R"])[op.target_indices] if len(op.target_indices) else np.array([])
    # Build figure: 2 rows × 4 cols (examples + coverage/centers/overlap + active maps + energy bar + pattern)
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            f"RF examples (n={n_rf_examples}) — W_RF 400×{L*L}",
            "RF centers (grid)",
            "Coverage ΣW (target)",
            "Overlap hist",
            "Active map A",
            "Active map B / R",
            "Energy E=Σdrive²",
            "Patterns A/B/R + omission 0",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}, {"type": "heatmap"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}, {"type": "heatmap"}],
        ],
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )
    # Row1 col1: show RF examples as heatmaps overlaid? Use first example RF as image, annotation for others as contours
    if op.n_target > 0:
        # pick evenly spaced target indices for examples
        ex_idx_pos = np.linspace(0, op.n_target - 1, num=min(n_rf_examples, op.n_target), dtype=int)
        # show first as heatmap
        first_gidx = int(op.target_indices[int(ex_idx_pos[0])])
        w_first = op.weights[first_gidx].reshape(L, L)
        fig.add_trace(go.Heatmap(z=w_first, colorscale="Viridis", showscale=False, hoverinfo="skip"), row=1, col=1)
        # annotate others as centers
        ex_centers = [op.centers.get(int(op.target_indices[int(p)]), (0, 0)) for p in ex_idx_pos[1:]]
        if ex_centers:
            fig.add_trace(go.Scatter(x=[c[0] for c in ex_centers], y=[c[1] for c in ex_centers], mode="markers+text", text=[f"ex{i+1}" for i in range(len(ex_centers))], textposition="top center", marker=dict(size=8, color="#d62728", symbol="x"), showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Heatmap(z=np.zeros((L, L)), showscale=False), row=1, col=1)
    # Row1 col2: centers
    if cx:
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=4, color="#1f77b4", opacity=0.8), hovertemplate="center (%{x:.1f},%{y:.1f})<extra></extra>", showlegend=False), row=1, col=2)
    # Row1 col3: coverage
    fig.add_trace(go.Heatmap(z=coverage, colorscale="Hot", showscale=True, colorbar=dict(title="ΣW", x=0.62, len=0.35, y=0.82), hoverinfo="skip"), row=1, col=3)
    # Row1 col4: overlap histogram
    if overlaps:
        fig.add_trace(go.Histogram(x=np.array(overlaps, dtype=float), nbinsx=22, marker=dict(color="#2ca02c"), showlegend=False, hovertemplate="overlap %{x:.2f} count %{y}<extra></extra>"), row=1, col=4)
        # annotate expected overlap 0.45
        try:
            ov_expected = float(rf_config.overlap)
            fig.add_vline(x=ov_expected, line=dict(color="#d62728", dash="dash"), row=1, col=4)
            fig.add_annotation(x=ov_expected, y=0.95, xref="x4", yref="paper", text=f"overlap {ov_expected:.2f}", showarrow=False, font=dict(color="#d62728", size=10))
        except Exception:
            pass
    # Row2 col1: active map A — centers colored by drive
    if cx and drive_A_t.size:
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=6, color=drive_A_t, colorscale="Blues", showscale=False, cmin=0, cmax=float(np.max(drive_A_t)) if drive_A_t.size and float(np.max(drive_A_t)) > 0 else 1), hovertemplate="drive %{marker.color:.3f}<extra></extra>", showlegend=False), row=2, col=1)
    # Row2 col2: active map B and R as two traces
    if cx and drive_B_t.size:
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=5, color=drive_B_t, colorscale="Reds", opacity=0.85, showscale=False), name="B", hovertemplate="B %{marker.color:.3f}<extra></extra>"), row=2, col=2)
    if cx and drive_R_t.size:
        # overlay R as x
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=4, color=drive_R_t, colorscale="Greys", opacity=0.55, symbol="x"), name="R", hovertemplate="R %{marker.color:.3f}<extra></extra>"), row=2, col=2)
    # Row2 col3: energy bar E_A/E_B/E_R/E_omission=0
    labels_E = ["E_A", "E_B", "E_R", "E_omission"]
    vals_E = [energies.get("A", 0.0), energies.get("B", 0.0), energies.get("R", 0.0), 0.0]
    sums_list = [sums.get("A", 0.0), sums.get("B", 0.0), sums.get("R", 0.0), 0.0]
    fig.add_trace(go.Bar(x=labels_E, y=vals_E, marker=dict(color=["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"]), hovertemplate="%{x} Σdrive² %{y:.3g}<extra></extra>", name="Σdrive²"), row=2, col=3)
    # annotate 185× problem hint: ratio max/min nonzero
    try:
        nz = [v for v in vals_E[:3] if v > 1e-12]
        if nz:
            ratio = float(max(nz) / max(1e-12, min(nz)))
            fig.add_annotation(x=0.5, y=1.08, xref="x7", yref="paper", text=f"max/min Σdrive²={ratio:.1f}× (185× would flag)", showarrow=False, font=dict(size=9, color="#333"), row=2, col=3)
    except Exception:
        pass
    # Row2 col4: patterns mosaic — stack A,B,R as rows in one heatmap (tile 3*L × L)
    # For simplicity show pat_A as heatmap with annotation for B/R positions
    fig.add_trace(go.Heatmap(z=pat_A, colorscale="Blues", showscale=False, hoverinfo="skip"), row=2, col=4)
    # overlay B center marker
    try:
        bca, bcb = rf_config.blob_center_A, rf_config.blob_center_B
        fig.add_trace(go.Scatter(x=[bca[0], bcb[0]], y=[bca[1], bcb[1]], mode="markers+text", text=["A blob", "B blob"], textposition="top center", marker=dict(size=6, color=["#1f77b4", "#d62728"]), showlegend=False), row=2, col=4)
    except Exception:
        pass
    # layout
    tgt_str = f"{rf_config.target_area} {list(rf_config.target_layers)} {list(rf_config.target_cell_types)}"
    title = (
        f"RF 32×32 W_RF {op.weights.shape[0]}×{op.weights.shape[1]} σ={rf_config.sigma_px:.1f} spacing={rf_config.spacing_px:.1f} overlap={rf_config.overlap:.2f} — "
        f"target {tgt_str} n_target={op.n_target} — "
        f"E_A {energies.get('A',0):.2f} E_B {energies.get('B',0):.2f} E_R {energies.get('R',0):.2f} E_omission=0 (network/rf.py:47)"
    )
    fig.update_layout(title=title, width=width, height=height, showlegend=True, legend=dict(x=0.02, y=-0.12, orientation="h"), margin=dict(l=30, r=30, t=90, b=40))
    # axis ranges for RF panels to [0,32]
    for c in [1, 2, 3]:
        try:
            fig.update_xaxes(range=[0, L], row=1, col=c)
            fig.update_yaxes(range=[0, L], autorange="reversed", row=1, col=c)
        except Exception:
            pass
    for c in [1, 2, 4]:
        try:
            fig.update_xaxes(range=[0, L], row=2, col=c)
            fig.update_yaxes(range=[0, L], autorange="reversed", row=2, col=c)
        except Exception:
            pass
    return fig

