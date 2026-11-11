"""Supplement block for one saved E2 run: details table, V0 schematic and the paper figure.

Usage:
  python scripts/e2_paper_block.py TAG --out-dir DIR [--results-dir DIR] [--discard S]

Reads DIR/TAG.json and DIR/TAG.npz (written by scripts/run_e2_stack.py). Writes to --out-dir:
  TAG_details.json, TAG_details.md  module sizes, edges per rule, every parameter with its source
  TAG_V0_architecture.png           V0 schematic: modules, cell types, edge counts
  TAG_block.png                     A raster, B RWS traces, C firing-rate spectra, D RWS SL map,
                                    E firing-rate SL map, F observed LANSA prototype (area mean)
Layout and colours follow the LANSA conventions (docs/laminar_motif/PLAN_PAPER_JOMISSION_LANSA.md, section 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jomission.laminar import caminos_e0 as c0  # noqa: E402
from jomission.laminar import caminos_stack as cs  # noqa: E402
import e2_motif_figure as mf  # noqa: E402
import e2_motif_similarity as ms  # noqa: E402

PORT = "PORT_SPEC (Tahvili, Vinck, di Volo)"
AGENT = "agent choice"
OWNER_BANDS = "owner ruling 2026-11-09 (PLAN_LANSA_E2.md, Owner rulings)"
MISSING = "not recorded in run JSON"
POPS = ("E", "PV", "SOM")
RULE_KEY = {"E": "e", "PV": "pv", "SOM": "sst"}  # rule-name stem of each cell type (caminos_e0.RULES)
P_LOCAL_NAME = "within-module p ({})"
# PORT_SPEC Table 1 (E, PV, SOM). Same values as tests/test_caminos_e0.py TABLE1 and as the JaxFNE 0.5.9
# ADEX_CELL_TYPE_DEFAULTS (jaxfne/emitters_adex.py), which the stack builds from.
NEURON_TABLE1 = {
    "VT_mV": (-50.0, -50.0, -50.0),
    "Delta_mV": (2.0, 0.5, 2.0),
    "Tref_ms": (5.0, 5.0, 5.0),
    "tauw_ms": (500.0, 500.0, 500.0),
    "a_nS": (4.0, 0.0, 4.0),
    "b_pA": (50.0, 0.0, 80.0),
    "C_pF": (200.0, 200.0, 200.0),
    "gL_nS": (10.0, 10.0, 10.0),
    "EL_mV": (-65.0, -65.0, -55.0),
    "Vreset_mV": (-65.0, -65.0, -65.0),
}
MEASURE_UNITS = {"voltage": "mV", "time": "ms", "capacitance": "pF", "conductance": "nS", "current": "pA",
                 "rate": "Hz", "source": "caminos_e0.py docstring (JaxFNE units)"}
RASTER_SAMPLE = {"E": 200, "PV": 50, "SOM": 50}  # cells per block kept in the raster (run_e2_stack.py RASTER_N)
DISCARD_S = 2.0  # PLAN_LANSA_E2.md, C6: 2 s discarded
WINDOW_S = 2.0  # raster and trace window: the 2 s after the discard
F_MAX_HZ = 150.0
RASTER_GAP = 12  # blank rows between modules in the raster
BAND_DASHES_HZ = (4.0, 8.0, 12.0, 30.0, 90.0)  # LANSA convention
TYPE_COLOR = {"E": "#8e24aa", "PV": "#1565c0", "SOM": "#ff9800"}  # purple, blue, orange (match-visual-style)
MODULE_LS = ("-", "--", ":")  # module line style in panel C: a second cue to the colour
TYPE_MARKER = {"E": "o", "PV": "^", "SOM": "s"}  # panel C: a non-colour cue per cell type
MARK_EVERY_HZ = 15  # panel C: one marker per 15 Hz on the 1 Hz grid
STACK_GAP = 0.5  # panel B: z units added beyond 2 x max|z| so adjacent traces never touch
FIG_W, FIG_H = 7.2, 6.7  # inches; 7.2 in canvas printed at the 6.7677 in text width (scale 0.940), so fonts carry
# FONT_SCALE below to reach their floors at print. The layout uses absolute inch positions (COL_X, AX_W), so the canvas
# width is not shrunk to the text width: that would push the third column past the page edge.
FONT_SCALE = 1.0638  # 7.2 / 6.7677: native pt x scale = pt at print
AX_W, AX_H = 1.75, 2.2  # inches, one panel
COL_X = (0.95, 3.05, 5.15)  # left edge of each panel column, inches
ROW_Y = (4.1, 1.2)  # bottom edge of the top and bottom panel row, inches
# Every pt below is native to the 7.2 in canvas; x FONT_SCALE gives the size at print (scale 0.940).
# Floors at print: 6 pt text, 5.5 pt ticks, 9 pt panel letters (match-visual-style).
TEXT_PT = 6.5 * FONT_SCALE  # 6.91 native -> 6.5 pt at print (tick and note size)
LABEL_PT = 7.0 * FONT_SCALE  # 7.45 native -> 7.0 pt at print (axis labels)
LETTER_PT = 9.6 * FONT_SCALE  # 10.2 native -> 9.6 pt at print (panel letters, floor 9)


def _sgn(d: float) -> str:
    return f"{d:+.1f}".replace("-", "−")


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _param(group: str, name: str, value: object, unit: str, source: str) -> dict:
    return {"group": group, "name": name, "value": value, "unit": unit, "source": source}


def load_run(results_dir: Path, tag: str) -> tuple[dict, dict[str, np.ndarray]]:
    rec = json.loads((results_dir / f"{tag}.json").read_text(encoding="utf-8"))
    with np.load(results_dir / f"{tag}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return rec, arrays


def _tau_by_module(rec: dict) -> dict[str, float | None]:
    """SOM->E decay per module. A scalar in older run JSON (all modules); a per-module dict in current runs."""
    tau = rec["tau_sst_e_ms"]
    if isinstance(tau, dict):
        return {m: (None if tau.get(m) is None else float(tau[m])) for m in rec["units"]}
    return {m: (None if tau is None else float(tau)) for m in rec["units"]}


def _sample_size(pop: str, size: int) -> int:
    return min(RASTER_SAMPLE[pop], size)


def rule_rows(rec: dict) -> list[dict]:
    """One row per realized rule: local blocks of each module, then the inter-module E projections."""
    units = list(rec["units"])
    tau = _tau_by_module(rec)
    rows: list[dict] = []
    for m in units:
        for name, pre, post, q, mech in c0.RULES:
            weight, mechanism, source = q, mech, f"{PORT} Q; {mech} receptor"
            if name == "sst_e" and tau[m] is not None:
                weight = q * c0.TAU_I_MS / tau[m]
                mechanism = f"gaba_sst ({tau[m]:g} ms)"
                source = f"{PORT} Q; weight x TAU_I/tau (caminos_stack.build_config); tau {AGENT}"
            rows.append({"rule": f"{name}_{m}", "pre": pre, "post": post, "module": m, "Q_nS": q,
                         "weight_nS": weight, "mechanism": mechanism, "source": source})
    for a, b in zip(units[:-1], units[1:]):
        for src, dst in ((a, b), (b, a)):
            for post, q in (("E", c0.Q_E_E), ("PV", c0.Q_E_INH), ("SOM", c0.Q_E_INH)):
                rows.append({"rule": f"x_{src}_{dst}_e_{post.lower()}", "pre": "E", "post": post,
                             "module": f"{src}->{dst}", "Q_nS": q, "weight_nS": q, "mechanism": "ampa",
                             "source": f"{PORT} Q applied to inter-module edges (caminos_stack); K_INTER {AGENT}"})
    return rows


def module_rows(rec: dict) -> list[dict]:
    out = []
    for m in rec["units"]:
        s = rec["sizes"][m]
        out.append({"name": m, "depth_rel_L4": float(rec["depth_rel_L4"][m]), "rho": float(rec["rho"][m]),
                    "N_E": int(s["E"]), "N_PV": int(s["PV"]), "N_SOM": int(s["SOM"]),
                    "N": int(sum(s.values()))})
    return out


def parameter_rows(rec: dict) -> list[dict]:
    units = list(rec["units"])
    mods = rec["modifiers"]
    tau = _tau_by_module(rec)
    k_scale = float(mods.get("k", 1.0))
    rows: list[dict] = []
    for field, vals in NEURON_TABLE1.items():
        for pop, v in zip(POPS, vals):
            rows.append(_param("neuron", f"{pop} {field}", v, field.rsplit("_", 1)[-1], PORT))
    rows.append(_param("neuron", "spike registered at", "V >= VT + 5 Delta", "mV", PORT))
    rows += [
        _param("synapse", "tau_delay", c0.DELAY_MS, "ms", PORT),
        _param("synapse", "tau_E (AMPA)", c0.TAU_E_MS, "ms", PORT),
        _param("synapse", "tau_I (GABA)", c0.TAU_I_MS, "ms", PORT),
        _param("synapse", "E_rev E (AMPA)", c0.REV_E_MV, "mV", PORT),
        _param("synapse", "E_rev I (GABA)", c0.REV_I_MV, "mV", PORT),
        _param("synapse", "Q E->E", c0.Q_E_E, "nS", PORT),
        _param("synapse", "Q E->PV, E->SOM", c0.Q_E_INH, "nS", PORT),
        _param("synapse", "Q PV->E, PV->PV, SOM->E, SOM->PV", c0.Q_INH, "nS", PORT),
        _param("synapse", "Q PV->SOM, SOM->SOM", 0.0, "nS", PORT + " (absent in the reference model)"),
        _param("drive", "Q external -> E, PV", c0.Q_EXT, "nS", PORT),
        _param("drive", "external sources per E and PV cell", c0.ext_source_count(0.07), "count",
               PORT + " (round(p NE), p 0.07; per-cell sources are the port's assumption)"),
        _param("drive", "external rate per source", c0.EXT_RATE_HZ, "Hz", PORT),
        _param("drive", "external drive to SOM", "none", "", PORT),
        _param("network", "dt", c0.DT_MS, "ms", PORT + " (Brian2 Euler dt; caminos_e0 DT_MS)"),
        _param("network", "N_E, N_PV, N_SOM (reference)", f"{c0.NE}, {c0.NPV}, {c0.NSST}", "count",
               PORT),
        _param("network", "within-module p (CAMINOS)", cs.P_LOCAL, "", PORT),
        _param("network", "K_E = N_E p (E inputs per cell)", cs.K_E, "count", PORT + " (caminos_stack.K_E)"),
        _param("network", "K_INTER (inter-module E inputs per cell per neighbour)", cs.K_INTER * k_scale, "count",
               f"{AGENT} (caminos_stack.K_INTER; the code comment cites owner ruling 2026-11-09 for p 0.01 of 8000)"),
        _param("network", "k (inter-module in-degree scale)", k_scale, "",
               f"{AGENT} ({'set' if 'k' in mods else 'code default 1.0, not in run modifiers'})"),
        _param("network", "inter-module in-degree per target E cell",
               f"{cs.K_INTER * k_scale:g} from each neighbour; end module {2 * cs.K_INTER * k_scale:g} from its one "
               "neighbour", "count", f"caminos_stack.build_config; K_INTER {AGENT}"),
        _param("network", "inter-module p", rec["p_inter"] if rec["p_inter"] is not None else
               "null: K_INTER based", "", "run JSON p_inter; caminos_stack.build_config"),
    ]
    for m in units:
        pl = rec.get("p_local")
        val = pl[m] if pl else "K_E/N_E (no kd)"
        rows.append(_param("network", P_LOCAL_NAME.format(m), val, "",
                           "derived: K_E/N_E x (density/mean density)^kd (caminos_stack.local_p); kd " + AGENT))
    for m in units:
        t = tau[m]
        rows.append(_param("synapse", f"SOM->E tau ({m})", "CAMINOS 8 ms" if t is None else t, "ms",
                           f"{AGENT} (sst modifier)"))
        w = c0.Q_INH if t is None else c0.Q_INH * c0.TAU_I_MS / t
        rows.append(_param("synapse", f"SOM->E weight ({m})", w, "nS",
                           f"derived: Q_INH x TAU_I/tau (caminos_stack.build_config); tau {AGENT}"))
    for m in units:
        rows.append(_param("module", f"rho = N_SOM/N_PV ({m})", float(rec["rho"][m]), "",
                           "owner ruling 2026-11-09 (caminos_stack.RHO comment); rho 1 = PORT_SPEC 750/750"))
        rows.append(_param("module", f"depth rel L4 ({m})", float(rec["depth_rel_L4"][m]), "",
                           f"{AGENT} (caminos_stack.layout)"))
    zs, _ = cs.lansa_bins()
    for m in units:
        sel = cs.MODULE_BINS[m](zs)
        rows.append(_param("module", f"MODULE_BINS ({m})",
                           f"{float(zs[sel].min()):g} to {float(zs[sel].max()):g} ({int(sel.sum())} bins)", "",
                           f"{AGENT} (caminos_stack.MODULE_BINS on lansa_density_X.json bins)"))
    rows += [
        _param("modifier", "n (mean E cells per module)", mods.get("n", "not set"), "count",
               f"{AGENT} (layout modifier n)"),
        _param("modifier", "dens (E cells proportional to LANSA density)", "on" if "dens" in mods else "off", "",
               f"{AGENT} (layout modifier dens)"),
        _param("modifier", "kd (local in-degree exponent)", mods.get("kd", "off"), "",
               f"{AGENT} (layout modifier kd)"),
        _param("modifier", "sst (SOM->E decay, all modules)", mods.get("sst", "not set"), "ms",
               f"{AGENT} (layout modifier sst)"),
        _param("analysis", "sigma (probe Gaussian, relative depth)", mf.SIGMA, "",
               f"{AGENT} (e2_motif_figure.SIGMA)"),
        _param("analysis", "V_ref (current proxy)", mf.V_REF_MV, "mV", f"{AGENT} (e2_motif_figure.V_REF_MV)"),
        _param("analysis", "discard", DISCARD_S, "s", "PLAN_LANSA_E2.md, C6 (2 s discarded)"),
        _param("analysis", "multitaper half-bandwidth (smoothing)", mf.SMOOTH_HZ, "Hz",
               "e2_motif_figure.SMOOTH_HZ (as the published analysis)"),
        _param("analysis", "RWS delay", mf.RWS_DELAY_MS, "ms", "Mazzoni et al. 2015 (e2_motif_figure.py; plan section 0)"),
        _param("analysis", "RWS alpha", mf.RWS_ALPHA, "", "Mazzoni et al. 2015 (e2_motif_figure.py; plan section 0)"),
        _param("analysis", "bands alpha-beta / gamma",
               f"{mf.BANDS[mf.FIG_BANDS][0][0]:g}-{mf.BANDS[mf.FIG_BANDS][0][1]:g} / "
               f"{mf.BANDS[mf.FIG_BANDS][1][0]:g}-{mf.BANDS[mf.FIG_BANDS][1][1]:g}", "Hz", OWNER_BANDS),
        _param("analysis", "probe contacts", f"{mf.CHANNELS.size} at {mf.CHANNELS[0]:g} to {mf.CHANNELS[-1]:g} rel. L4",
               "", "LANSA bin centres (lansa_density_X.json axes; e2_motif_figure.CHANNELS)"),
    ]
    return rows


def _sha_line(src: str | None) -> str:
    """First line of the recorded jaxfne_source that names a sha; 'not recorded' when there is none."""
    for line in (src or "").splitlines():
        if "sha" in line.lower():
            return line.strip()
    return "not recorded"


def run_rows(rec: dict) -> list[dict]:
    return [
        {"key": "layout", "value": rec.get("layout", MISSING)},
        {"key": "experiment", "value": rec.get("experiment", MISSING)},
        {"key": "seed", "value": rec["seed"]},
        {"key": "T_s", "value": rec["T_s"]},
        {"key": "dt_ms", "value": rec["dt_ms"]},
        {"key": "jaxfne_version", "value": rec["jaxfne_version"]},
        {"key": "jaxfne_source sha", "value": _sha_line(rec.get("jaxfne_source"))},
        {"key": "jax_devices", "value": rec.get("jax_devices", MISSING)},
        {"key": "git_sha (run)", "value": rec["git_sha"]},
        {"key": "git_dirty", "value": rec.get("git_dirty", MISSING)},
        {"key": "wall_s", "value": rec.get("wall_s", {}).get("total", MISSING)},
    ]


def _notes(rec: dict) -> list[str]:
    notes = [
        "Exploratory, unsealed E2 run (run JSON experiment). The recorded JaxFNE version is "
        f"{rec['jaxfne_version']}; PLAN_LANSA_E2.md section 0 and OPEN-37 name 0.5.8 as the project authority.",
        f"Raster: sampled cells only ({RASTER_SAMPLE['E']} E, {RASTER_SAMPLE['PV']} PV and "
        f"{RASTER_SAMPLE['SOM']} SOM per module, a random sample by run_e2_stack.py).",
        "Module N is the sum of the recorded cell counts; the run JSON has no N key.",
        "Local p and the inter-module edges are as recorded; the inter-module autapse rule is not checked here.",
    ]
    missing = [k for k in ("git_dirty", "jaxfne_source") if k not in rec]
    if missing:
        notes.append(f"Run JSON has no {', '.join(missing)} key (written by later runner versions); reported as not recorded.")
    if not isinstance(rec["tau_sst_e_ms"], dict):
        notes.append("tau_sst_e_ms is a scalar in the run JSON; the current runner writes a per-module dict "
                     "(caminos_stack.sst_taus). Both are read the same way here.")
    return notes


def build_details(rec: dict, tag: str) -> dict:
    units = list(rec["units"])
    sizes = {m: {p: int(rec["sizes"][m][p]) for p in POPS} for m in units}
    n_total = sum(sum(sizes[m].values()) for m in units)
    if n_total != len(cs.neuron_rows(sizes)):
        raise RuntimeError(f"sizes sum to {n_total}, neuron rows {len(cs.neuron_rows(sizes))}")
    zs, _ = cs.lansa_bins()
    if not np.allclose(mf.CHANNELS, zs, atol=1e-3):
        raise RuntimeError("probe contacts differ from the LANSA bin centres")
    rules = rule_rows(rec)
    names = {r["rule"] for r in rules}
    if names != set(rec["edges"]):
        raise ValueError(f"rule names differ from run edges: {sorted(names ^ set(rec['edges']))}")
    for r in rules:
        r["edges"] = int(rec["edges"][r["rule"]])
    return {
        "tag": tag,
        "layout": rec.get("layout", MISSING),
        "experiment": rec.get("experiment", MISSING),
        "N": n_total,
        "measure_units": MEASURE_UNITS,
        "modules": module_rows(rec),
        "rules": rules,
        "edges": {r["rule"]: r["edges"] for r in rules},
        "parameters": parameter_rows(rec),
        "run": run_rows(rec),
        "notes": _notes(rec),
    }


def to_markdown(det: dict) -> str:
    out = [f"# Details: {det['tag']}", "",
           f"Layout `{det['layout']}`. {det['experiment']}.", "",
           "Units: " + ", ".join(f"{k} {v}" for k, v in det["measure_units"].items() if k != "source")
           + f" (source: {det['measure_units']['source']}).", "",
           "Source labels: `" + PORT + "` (values from caminos_e0 or the port spec), `" + AGENT
           + "` (set by a modifier or this project), or a named derivation.", "",
           "## Modules", "",
           "| module | depth rel. L4 | rho | N_E | N_PV | N_SOM | N |", "|---|---|---|---|---|---|---|"]
    for m in det["modules"]:
        out.append(f"| {m['name']} | {_sgn(m['depth_rel_L4'])} | {m['rho']:g} | {m['N_E']:,} | {m['N_PV']:,} | "
                   f"{m['N_SOM']:,} | {m['N']:,} |")
    out += ["", f"Total N = {det['N']:,}.", "",
            "## Edges per rule", "",
            "| rule | pre to post | module or pair | Q (nS) | weight (nS) | mechanism | edges |",
            "|---|---|---|---|---|---|---|"]
    for r in det["rules"]:
        out.append(f"| {r['rule']} | {r['pre']} to {r['post']} | {r['module']} | {_fmt(r['Q_nS'])} | "
                   f"{r['weight_nS']:.4g} | {r['mechanism']} | {r['edges']:,} |")
    out += ["", "## Parameters", "", "| group | parameter | value | unit | source |", "|---|---|---|---|---|"]
    for p in det["parameters"]:
        out.append(f"| {p['group']} | {p['name']} | {_fmt(p['value'])} | {p['unit']} | {p['source']} |")
    out += ["", "## Run", "", "| key | value |", "|---|---|"]
    out += [f"| {r['key']} | {_fmt(r['value'])} |" for r in det["run"]]
    out += ["", "## Notes", ""] + [f"- {n}" for n in det["notes"]]
    return "\n".join(out) + "\n"


def _set_style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": TEXT_PT, "axes.labelsize": LABEL_PT, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.labelsize": TEXT_PT, "ytick.labelsize": TEXT_PT, "legend.fontsize": TEXT_PT})


def _tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _contact_edges(ys: np.ndarray) -> np.ndarray:
    return np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2, [ys[-1] + (ys[-1] - ys[-2]) / 2]])


def draw_v0(det: dict, path: Path) -> None:
    """V0 schematic: one box per module at its depth order, with its cell counts, local rules and inter-module edges."""
    _set_style()
    h = 5.6
    fig = plt.figure(figsize=(FIG_W, h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, h)
    ax.axis("off")
    rules = {r["rule"]: r for r in det["rules"]}
    mods = det["modules"]
    x0, box_w, box_h, step = 0.35, 6.5, 1.15, 1.6
    centres = [4.9 - i * step for i in range(len(mods))]
    for mod, yc in zip(mods, centres):
        m = mod["name"]
        ax.add_patch(FancyBboxPatch((x0, yc - box_h / 2), box_w, box_h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="none", ec="#555555", lw=0.8))
        ax.text(x0 + 0.12, yc + box_h / 2 - 0.18, f"{m}   depth {_sgn(mod['depth_rel_L4'])}   rho {mod['rho']:g}   "
                f"N {mod['N']:,}", fontsize=LABEL_PT, weight="bold", va="center")
        nodes = {"E": (x0 + 0.6, yc - 0.1, mod["N_E"]), "PV": (x0 + 1.75, yc + 0.08, mod["N_PV"]),
                 "SOM": (x0 + 1.75, yc - 0.3, mod["N_SOM"])}
        for pop, (px, py, n) in nodes.items():
            ax.add_patch(Circle((px, py), 0.12, color=TYPE_COLOR[pop], zorder=3))
            ax.text(px, py, pop, color="white", fontsize=6.9 if pop == "SOM" else TEXT_PT, ha="center",
                    va="center", zorder=4, weight="bold")
            if pop == "E":  # the E<->PV and E<->SOM arcs leave E on the right and below; the label goes above
                ax.text(px, py + 0.2, f"N {n:,}", fontsize=TEXT_PT, va="center", ha="center")
            else:
                ax.text(px + 0.17, py, f"N {n:,}", fontsize=TEXT_PT, va="center")
        for a, b in (("E", "PV"), ("PV", "E"), ("E", "SOM"), ("SOM", "E")):
            pa, pb_ = nodes[a], nodes[b]
            rad = 0.25 if (a, b) in (("E", "PV"), ("E", "SOM")) else -0.25
            ax.add_patch(FancyArrowPatch((pa[0], pa[1]), (pb_[0], pb_[1]), arrowstyle="-|>", mutation_scale=6,
                                         shrinkA=9, shrinkB=9, connectionstyle=f"arc3,rad={rad}", lw=0.6,
                                         color="#444444", zorder=2))
        lines = []
        for pre, post in (("E", "E"), ("E", "PV"), ("E", "SOM"), ("PV", "E"), ("PV", "PV"), ("SOM", "E"),
                          ("SOM", "PV")):
            r = rules[f"{RULE_KEY[pre]}_{RULE_KEY[post]}_{m}"]
            lines.append(f"{pre} to {post}: {r['weight_nS']:.3g} nS, {r['edges']:,}")
        for i, line in enumerate(lines):
            ax.text(x0 + 3.05, yc + 0.28 - i * 0.115, line, fontsize=TEXT_PT, va="center")
    for i in range(len(mods) - 1):
        a, b = mods[i]["name"], mods[i + 1]["name"]
        y_mid = (centres[i] + centres[i + 1]) / 2
        for j, (src, dst) in enumerate(((a, b), (b, a))):
            parts = [f"{src} to {dst}:"] + [
                f"E to {post} {det['edges'][f'x_{src}_{dst}_e_{post.lower()}']:,}" for post in POPS]
            ax.text(x0 + 0.2, y_mid + (0.1 if j == 0 else -0.1), "   ".join(parts), fontsize=TEXT_PT, va="center")
    pv = {p["name"]: p["value"] for p in det["parameters"]}
    foot = [
        f"Synapses: conductance based; tau_E {c0.TAU_E_MS:g} ms, tau_I {c0.TAU_I_MS:g} ms, E_rev {c0.REV_E_MV:g} and "
        f"{c0.REV_I_MV:g} mV, delay {c0.DELAY_MS:g} ms ({PORT}).",
        f"External: {c0.ext_source_count(0.07)} private {c0.EXT_RATE_HZ:g} Hz Poisson sources per E and PV cell, "
        f"{c0.Q_EXT:g} nS; SOM undriven ({PORT}).",
        "Within-module p: " + ", ".join(f"{m['name']} {_fmt(pv[P_LOCAL_NAME.format(m['name'])])}" for m in mods)
        + f" (kd: {AGENT}).",
        f"Every value, its source and the edge counts: {det['tag']}_details.md.",
        "Arcs: E<->PV and E<->SOM loops; all rules and counts are in the text column.",
    ]
    for i, line in enumerate(foot):
        ax.text(x0, 1.05 - i * 0.2, line, fontsize=TEXT_PT, va="center")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def model_signals(rec: dict, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(rate [ms, 3U] Hz per cell in block order, RWS [ms, U] pA per E cell) from the saved counts and g_nS."""
    units = list(rec["units"])
    dt = float(rec["dt_ms"])
    per = int(round(1.0 / dt))  # steps per ms
    keys = [str(k) for k in arrays["block_keys"]]
    counts = arrays["counts"].astype(np.float64)
    n_ms = counts.shape[0] // per
    spikes_ms = counts[: n_ms * per].reshape(n_ms, per, -1).sum(axis=1)
    cols = [keys.index(f"{m}_{p}") for m in units for p in POPS]
    n_cell = np.array([rec["sizes"][m][p] for m in units for p in POPS], dtype=float)
    rate = spikes_ms[:, cols] / n_cell * 1000.0
    rws = mf.rws(arrays["g_nS"].astype(np.float64), dt, rec, units)
    n_r = rws.shape[0] // per
    rws_ms = rws[: n_r * per].reshape(n_r, per, -1).mean(axis=1)
    n_e = np.array([rec["sizes"][m]["E"] for m in units], dtype=float)
    n = min(rate.shape[0], rws_ms.shape[0])
    return rate[:n], rws_ms[:n] / n_e


def relative_map(x: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x [T, U] at 1 kHz -> (f, R [contacts, f]) with R = P / max over contacts (e2_motif_figure.row)."""
    K = np.exp(-((mf.CHANNELS[:, None] - depth[None, :]) ** 2) / (2 * mf.SIGMA ** 2))
    f, S = mf.cross_spectra(x, 1000.0, min(1000, x.shape[0]))
    P = mf.probe_power(S.mean(axis=0), K)
    return f, P / P.max(axis=0, keepdims=True)


def stack_offsets(z: np.ndarray) -> np.ndarray:
    """Vertical offset of each trace of z [T, U], unit 0 on top. Each trace spans +-max|z|, so a step above 2 max|z| keeps them apart."""
    n_u = z.shape[1]
    step = 2.0 * float(np.abs(z).max()) + STACK_GAP
    return np.array([(n_u - 1 - i) * step for i in range(n_u)])


def _panel(fig, col: int, row: int, dx: float = 0.0, w: float = AX_W):
    return fig.add_axes([(COL_X[col] + dx) / FIG_W, ROW_Y[row] / FIG_H, w / FIG_W, AX_H / FIG_H])


def _letter(fig, ax, letter: str) -> None:
    box = ax.get_position()
    fig.text((box.x0 * FIG_W - 0.2) / FIG_W, (box.y1 * FIG_H + 0.1) / FIG_H, letter, fontsize=LETTER_PT, weight="bold",
             va="bottom")


def _band_bars(ax) -> None:
    tr = ax.get_xaxis_transform()
    (a0, a1), (g0, g1) = mf.BANDS[mf.FIG_BANDS]
    for (lo, hi), col, name in (((a0, a1), mf.C_AB, "α–β"), ((g0, g1), mf.C_GM, "γ")):
        lab = f"{name}\n{lo:g}–{hi:g} Hz"
        ax.plot([lo, hi], [1.03, 1.03], color=col, lw=3.5, transform=tr, clip_on=False, solid_capstyle="butt")
        ax.text((lo + hi) / 2, 1.06, lab, transform=tr, ha="center", va="bottom", fontsize=TEXT_PT)


def _sl_map(ax, R: np.ndarray, depth_ch: np.ndarray, xlabel: str, show_y: bool):
    edges = _contact_edges(depth_ch)
    im = ax.imshow(R, aspect="auto", cmap=mf.PARULA, vmin=0.5, vmax=1.0, interpolation="nearest",
                   extent=(0.5, F_MAX_HZ + 0.5, edges[-1], edges[0]))
    ax.set_xlim(0.5, F_MAX_HZ + 0.5)
    ax.set_ylim(edges[-1], edges[0])
    ax.set_xticks([50, 100, 150])
    depth_ticks = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
    ax.set_yticks(depth_ticks)
    ax.set_yticklabels([f"{v:.1f}".replace("-", "−") for v in depth_ticks] if show_y else [])
    ax.set_xlabel(xlabel, fontsize=LABEL_PT, linespacing=1.2)
    _band_bars(ax)
    _tidy(ax)
    return im


def draw_block(rec: dict, arrays: dict[str, np.ndarray], proto: np.ndarray, discard_s: float, path: Path,
               tag: str) -> None:
    _set_style()
    units = list(rec["units"])
    n_u = len(units)
    dt = float(rec["dt_ms"])
    depth = np.array([float(rec["depth_rel_L4"][m]) for m in units])
    d0 = int(round(discard_s * 1000))
    w1 = d0 + int(round(WINDOW_S * 1000))
    rate, rws = model_signals(rec, arrays)
    if rate.shape[0] <= w1:
        raise ValueError(f"run has {rate.shape[0]} ms; the window ends at {w1} ms")
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    t_lo, t_hi = d0 / 1000.0, w1 / 1000.0

    # A: raster of the sampled cells, modules top to bottom in depth order
    axA = _panel(fig, 0, 0)
    y = 0
    ticks, tick_labels = [], []
    for i, m in enumerate(units):
        top = y
        for p in POPS:
            n_row = _sample_size(p, int(rec["sizes"][m][p]))
            cell = arrays[f"{m}_{p}_c"]
            if cell.size and int(cell.max()) >= n_row:
                raise ValueError(f"{m}_{p}: cell index {int(cell.max())} outside the {n_row}-row sample")
            tt = arrays[f"{m}_{p}_t"] * dt / 1000.0
            sel = (tt >= t_lo) & (tt < t_hi)
            axA.scatter(tt[sel], y + cell[sel], s=0.9, marker="|", linewidths=0.35, color=TYPE_COLOR[p])
            y += n_row
        ticks.append((top + y) / 2)
        tick_labels.append(f"{m} {_sgn(depth[i])}")
        y += RASTER_GAP
    axA.set_xlim(t_lo, t_hi)
    axA.set_ylim(y - RASTER_GAP, -1)
    axA.set_yticks(ticks)
    axA.set_yticklabels(tick_labels)
    axA.set_xticks([t_lo, (t_lo + t_hi) / 2, t_hi])
    axA.set_xlabel("Time (s); sampled cells", fontsize=LABEL_PT)
    handles = [Line2D([], [], ls="", marker="|", ms=5, color=TYPE_COLOR[p]) for p in POPS]
    axA.legend(handles, list(POPS), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
               handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    _tidy(axA)

    # B: RWS per E cell, z-scored, stacked by module
    axB = _panel(fig, 1, 0)
    seg = rws[d0:w1]
    tt = (np.arange(seg.shape[0]) + d0) / 1000.0
    z = (seg - seg.mean(axis=0)) / seg.std(axis=0)
    offs = stack_offsets(z)
    for i in range(n_u):
        axB.plot(tt, z[:, i] + offs[i], color="black", lw=0.6)
    axB.set_yticks(offs)
    axB.set_yticklabels(units)
    axB.set_xlim(t_lo, t_hi)
    axB.set_xticks([t_lo, (t_lo + t_hi) / 2, t_hi])
    axB.set_xlabel("Time (s)", fontsize=LABEL_PT)
    axB.set_ylabel("RWS, z", fontsize=LABEL_PT)
    _tidy(axB)

    # C: firing-rate spectra, one E-population trace per module (the per-type set is 9 lines and unreadable at
    # this panel width); module distinguished by line style and by a direct label at each trace's gamma peak
    axC = _panel(fig, 2, 0, dx=0.15, w=AX_W - 0.15)  # room for the log tick labels and the y label
    nper = min(1000, rate.shape[0] - d0)
    f, S = mf.cross_spectra(rate[d0:], 1000.0, nper)
    pw = np.einsum("fuu->uf", S.mean(axis=0)).real
    sel = (f >= 1) & (f <= F_MAX_HZ)
    e_cols = [i * len(POPS) for i in range(n_u)]
    for i, col in enumerate(e_cols):
        axC.plot(f[sel], pw[col, sel], color=TYPE_COLOR["E"], ls=MODULE_LS[i], lw=1.2)
    (a0, a1), (g0, g1) = mf.BANDS[mf.FIG_BANDS]
    axC.axvspan(a0, a1, color=mf.C_AB, alpha=0.10, lw=0)
    axC.axvspan(g0, g1, color=mf.C_GM, alpha=0.10, lw=0)
    for fd in BAND_DASHES_HZ:
        axC.axvline(fd, color="#9e9e9e", lw=0.5, ls="--")
    axC.set_yscale("log")
    axC.set_xlim(1, F_MAX_HZ)
    axC.set_xticks([50, 100, 150])
    axC.set_xlabel("Frequency (Hz)", fontsize=LABEL_PT)
    axC.set_ylabel("E firing-rate PSD (Hz²/Hz)", fontsize=LABEL_PT)
    mod_handles = [Line2D([], [], color=TYPE_COLOR["E"], lw=1.2, ls=MODULE_LS[i]) for i in range(n_u)]
    axC.legend(mod_handles, units, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=n_u,
               handlelength=1.6, columnspacing=0.8, borderaxespad=0.1, title="E population",
               title_fontsize=TEXT_PT)
    _tidy(axC)

    # D and E: SL maps (RWS per E cell; firing rate of E cells), F: observed LANSA prototype
    f_r, R_rws = relative_map(rws[d0:], depth)
    e_cols = [i * len(POPS) for i in range(n_u)]
    _, R_fr = relative_map(rate[d0:][:, e_cols], depth)
    k_mod = slice(1, int(F_MAX_HZ) + 1)  # 1-150 Hz: f_r is 1 Hz spaced
    if f_r[1] != 1.0:
        raise RuntimeError("spectra are not on a 1 Hz grid")
    axD = _panel(fig, 0, 1)
    im = _sl_map(axD, R_rws[:, k_mod], mf.CHANNELS, "Frequency (Hz)\nRWS LFP (model)", True)
    axD.set_ylabel("Position from L4", fontsize=LABEL_PT)
    axE = _panel(fig, 1, 1)
    _sl_map(axE, R_fr[:, k_mod], mf.CHANNELS, "Frequency (Hz)\nFiring rate, E (model)", False)
    axF = _panel(fig, 2, 1)
    _sl_map(axF, proto[:, :int(F_MAX_HZ)], mf.CHANNELS, "Frequency (Hz)\nLANSA observed (area mean)", False)
    cax = fig.add_axes([COL_X[1] / FIG_W, 0.5 / FIG_H, AX_W / FIG_W, 0.1 / FIG_H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.5, 0.75, 1.0])
    cb.ax.set_xticklabels(["0.5", "0.75", "1"], fontsize=TEXT_PT)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Relative power", fontsize=LABEL_PT, labelpad=1.5)
    seed, T_s = rec["seed"], rec["T_s"]
    # The note sits in the band above the bottom edge, left of the colourbar's tick column, wrapped to four short
    # lines so its right edge stays inside the canvas (a single wrapped line reached 19 px past the 720 px width).
    note = (f"Run {tag} (seed {seed}, {T_s:g} s;\n"
            f"first {discard_s:g} s discarded; n = 1 seed, no SEM).\n"
            "Model: stationary power, 1 s windows, 2 Hz smoothing,\n"
            "21-contact probe (sigma 0.15).\n"
            "LANSA: area mean of relative power,\n"
            "1 s peri-stimulus windows.")
    fig.text(0.30 / FIG_W, 0.005 / FIG_H, note, fontsize=TEXT_PT - 0.5, va="bottom", linespacing=1.1)
    for ax, letter in zip((axA, axB, axC, axD, axE, axF), "ABCDEF"):
        _letter(fig, ax, letter)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "laminar_e2")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--discard", type=float, default=DISCARD_S)
    a = ap.parse_args(argv)
    rec, arrays = load_run(a.results_dir, a.tag)
    det = build_details(rec, a.tag)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{a.tag}_details.json").write_text(json.dumps(det, indent=1), encoding="utf-8")
    (a.out_dir / f"{a.tag}_details.md").write_text(to_markdown(det), encoding="utf-8")
    draw_v0(det, a.out_dir / f"{a.tag}_V0_architecture.png")
    draw_block(rec, arrays, ms.PROTO, a.discard, a.out_dir / f"{a.tag}_block.png", a.tag)
    print(json.dumps({"tag": a.tag, "N": det["N"], "rules": len(det["rules"]), "parameters": len(det["parameters"]),
                      "out_dir": str(a.out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
