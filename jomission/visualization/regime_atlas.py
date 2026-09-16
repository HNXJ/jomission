"""Regime atlases: one atlas per sealed regime, rendered from results JSON only.

Reads ``results/<family>_<regime>.json`` and ``results/<family>_lineage.json``
as written by ``tests/test_v21_operation.py::run_regime`` / ``test_v21_verdict``.
Never simulates and never imports jaxfne: every plotted number is a sealed
value (OBSERVED) or arithmetic on sealed values (DERIVED, labelled).

Sealed record schema (per regime; lists are indexed by analysis window, i.e.
the last three 5 s chunks of a 20 s run, not by seed):
  tonic    {class: drive}
  rates    [{class: Hz}] x3
  cv       [{cv_evaluable, frac_in, E_rate_CV}] x3   (frac_in, E_rate_CV rounded to 3 dp)
  sync     [float] x3
  drift    {class: (max-min)/mean over windows}
  currents [{class: {syn_mean}, pathway: {I_mean}}] x3
  checks   {gate: bool};  pass: bool

Output layout (``build_regime_atlases``)::

  <out_dir>/index.html            cross-regime overview (gate matrix, rates vs regime)
  <out_dir>/manifest.json
  <out_dir>/<family>_<regime>/{index,gates,rates,irregularity,stability,currents}.html

Output is byte-deterministic for fixed inputs (no timestamps, fixed div ids).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from typing import Any, Dict, List, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jomission.visualization.theme import CLASS_COLORS, apply_dark_theme

CLASSES = ("E", "PV", "SST", "VIP")
INTERNEURONS = ("PV", "SST", "VIP")
FAMILY_LABELS = {"v21": "V2.1"}
FAMILY_PRODUCERS = {"v21": "tests/test_v21_operation.py::run_regime"}

# V2.1 gate thresholds, copied from tests/test_v21_operation.py::run_regime.
# test_regime_atlas.py asserts that recomputing checks with these reproduces
# every sealed ``checks`` dict, so drift from the battery fails loudly.
E_BAND_HZ = (2.0, 30.0)
INTERNEURON_MIN_HZ = 1.0
ISI_CV_FRAC_MIN = 0.8
E_RATE_CV_MIN = 0.3
SYNC_MAX = 0.01  # strict: max(sync) < SYNC_MAX
DRIFT_MAX = 0.20

PANELS = (
    ("gates.html", "Gates"),
    ("rates.html", "Rates"),
    ("irregularity.html", "Irregularity"),
    ("stability.html", "Stability"),
    ("currents.html", "Currents"),
)

UNAVAILABLE = {
    "raster / membrane traces / spectral": "spike trains and voltages are not stored in the sealed record",
    "network 3D / connectivity": "the sealed record stores no neuron or edge tables",
}

ACCENT = "#58a6ff"
CLASS_SYMBOLS = {"E": "circle", "PV": "square", "SST": "diamond", "VIP": "triangle-up"}
WINDOWS_NOTE = "analysis windows = last three 5 s chunks of a 20 s run"
REQUIRED_KEYS = ("regime", "tonic", "rates", "cv", "sync", "drift", "currents", "checks", "pass")


# ---------------------------------------------------------------- loading


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_regimes(
    results_dir: str = "results",
    family: str = "v21",
    regimes: Sequence[str] = ("low", "mid", "high"),
) -> Dict[str, Any]:
    """Load sealed regime records + lineage. Raises on any missing file or key."""
    records = []
    for name in regimes:
        path = os.path.join(results_dir, f"{family}_{name}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"sealed regime record missing: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        n_win = len(data["rates"])
        if not (len(data["cv"]) == len(data["sync"]) == len(data["currents"]) == n_win):
            raise ValueError(f"{path}: per-window lists disagree in length")
        records.append({"id": f"{family}_{name}", "source": path.replace(os.sep, "/"),
                        "sha256": _sha256(path), "data": data})
    lineage_path = os.path.join(results_dir, f"{family}_lineage.json")
    if not os.path.isfile(lineage_path):
        raise FileNotFoundError(f"sealed lineage missing: {lineage_path}")
    with open(lineage_path, encoding="utf-8") as f:
        lineage = json.load(f)
    return {"family": family, "records": records,
            "lineage": {"source": lineage_path.replace(os.sep, "/"),
                        "sha256": _sha256(lineage_path), "data": lineage}}


def recheck_gates(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Recompute each V2.1 gate from sealed values; return worst value + verdict."""
    rates, cv = data["rates"], data["cv"]
    e = [r["E"] for r in rates]
    inter_min = {c: min(r[c] for r in rates) for c in INTERNEURONS}
    worst_inter = min(inter_min, key=inter_min.get)
    frac = [w["frac_in"] for w in cv]
    ecv = [w["E_rate_CV"] for w in cv]
    worst_drift = max(data["drift"], key=data["drift"].get)
    return {
        "E_band": {"rule": f"{E_BAND_HZ[0]:g} <= E <= {E_BAND_HZ[1]:g} Hz (every window)",
                   "worst": f"E {min(e):.3g}-{max(e):.3g} Hz",
                   "pass": all(E_BAND_HZ[0] <= x <= E_BAND_HZ[1] for x in e)},
        "classes_active": {"rule": f"PV, SST, VIP >= {INTERNEURON_MIN_HZ:g} Hz (every window)",
                           "worst": f"{worst_inter} {inter_min[worst_inter]:.3g} Hz",
                           "pass": all(v >= INTERNEURON_MIN_HZ for v in inter_min.values())},
        "isi_cv": {"rule": f"ISI-CV in [0.5, 1.5] for >= {ISI_CV_FRAC_MIN:.0%} of evaluable cells",
                   "worst": f"{min(frac):.3g}",
                   "pass": all(x >= ISI_CV_FRAC_MIN for x in frac)},
        "E_het": {"rule": f"E rate-CV >= {E_RATE_CV_MIN:g}",
                  "worst": f"{min(ecv):.3g}",
                  "pass": all(x >= E_RATE_CV_MIN for x in ecv)},
        "sync": {"rule": f"max sync < {SYNC_MAX:g}",
                 "worst": f"{max(data['sync']):.3g}",
                 "pass": max(data["sync"]) < SYNC_MAX},
        "drift": {"rule": f"class-rate drift <= {DRIFT_MAX:g}",
                  "worst": f"{worst_drift} {data['drift'][worst_drift]:.3g}",
                  "pass": all(v <= DRIFT_MAX for v in data["drift"].values())},
    }


def recompute_drift(data: Dict[str, Any]) -> Dict[str, float]:
    """Drift per class from sealed rates, same formula as run_regime."""
    out = {}
    n = len(data["rates"])
    for c in CLASSES:
        rs = [r[c] for r in data["rates"]]
        out[c] = float((max(rs) - min(rs)) / max(sum(rs) / n, 1e-9))
    return out


# ---------------------------------------------------------------- rendering helpers


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _fmt(x: Any) -> str:
    return f"{x:.4g}" if isinstance(x, float) else _esc(x)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _status(ok: bool) -> str:
    return "<span class='pass'>&#10003; PASS</span>" if ok else "<span class='fail'>&#10007; FAIL</span>"


_CSS = """*{box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;margin:0;padding:24px 16px;
font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.5}
.container{max-width:1180px;margin:0 auto}a{color:#58a6ff}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:24px 0 8px}.sub{color:#8b949e;margin:0 0 16px}
.box{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 16px;margin-top:16px;overflow-x:auto}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;color:#fff;margin-right:6px}
.OBSERVED{background:#238636}.DERIVED{background:#1f6feb}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
th,td{text-align:left;padding:4px 12px 4px 0;border-bottom:1px solid #21262d;white-space:nowrap}
th{color:#8b949e;font-weight:600}.pass{color:#3fb950}.fail{color:#f85149}
.fig{width:100%;overflow-x:auto}nav a{margin-right:14px}"""


def _page(title: str, subtitle: str, nav: str, sections: Sequence[str]) -> str:
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class='container'>"
            f"<nav>{nav}</nav><h1>{_esc(title)}</h1><p class='sub'>{subtitle}</p>"
            + "".join(sections) + "</div></body></html>")


def _fig_html(fig: go.Figure, div_id: str) -> str:
    return "<div class='fig'>" + fig.to_html(include_plotlyjs="cdn", full_html=False,
                                             div_id=div_id, config={"displaylogo": False}) + "</div>"


def _box(evidence: str, caption: str, inner: str = "") -> str:
    return f"<div class='box'><span class='badge {evidence}'>{evidence}</span>{caption}{inner}</div>"


def _style(fig: go.Figure, title: str, subtitle: str, height: int = 460) -> go.Figure:
    apply_dark_theme(fig, title, subtitle)
    fig.update_layout(height=height, autosize=True, showlegend=False,
                      margin=dict(l=60, r=30, t=90, b=60))
    return fig


def _threshold(fig: go.Figure, y: float, label: str, x0: float, x1: float,
               row: int | None = None, col: int | None = None) -> None:
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color="#8b949e", width=1, dash="dash"), **kw)
    fig.add_annotation(x=x0, y=y, text=label, showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color="#8b949e"), **kw)


# ---------------------------------------------------------------- panels


def _panel_gates(rec: Dict[str, Any]) -> List[str]:
    d = rec["data"]
    gates = recheck_gates(d)
    rows = []
    for g, info in gates.items():
        sealed = bool(d["checks"].get(g))
        agree = "yes" if sealed == info["pass"] else "MISMATCH"
        rows.append((g, info["rule"], info["worst"], _status(sealed), _status(info["pass"]), agree))
    head = "".join(f"<th>{h}</th>" for h in ("gate", "rule", "worst observed", "sealed check (OBSERVED)",
                                               "recomputed (DERIVED)", "agree"))
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return [_box("OBSERVED", f" Sealed <code>pass</code> = <b>{d['pass']}</b>. Recomputed column applies the "
                             "rules to sealed values; frac_in and E rate-CV are stored rounded to 3 dp.", table)]


def _panel_rates(rec: Dict[str, Any]) -> List[str]:
    d = rec["data"]
    n = len(d["rates"])
    fig = go.Figure()
    for i, c in enumerate(CLASSES):
        vals = [r[c] for r in d["rates"]]
        mean = sum(vals) / n
        fig.add_trace(go.Bar(x=[c], y=[mean], marker=dict(color=CLASS_COLORS[c], cornerradius=4), width=0.3,
                             text=[f"{mean:.3g} Hz"], textposition="outside", cliponaxis=False,
                             hovertemplate=f"{c}: mean of {n} windows %{{y:.4g}} Hz<extra></extra>"))
        fig.add_trace(go.Scatter(x=[c] * n, y=vals, mode="markers",
                                 marker=dict(size=8, color="#c9d1d9", line=dict(color="#161b22", width=2)),
                                 customdata=list(range(1, n + 1)),
                                 hovertemplate=f"{c} window %{{customdata}}: %{{y:.4g}} Hz<extra></extra>"))
    _threshold(fig, E_BAND_HZ[0], f"E min {E_BAND_HZ[0]:g}", -0.45, 0.45)
    _threshold(fig, E_BAND_HZ[1], f"E max {E_BAND_HZ[1]:g}", -0.45, 0.45)
    _threshold(fig, INTERNEURON_MIN_HZ, f"PV/SST/VIP min {INTERNEURON_MIN_HZ:g}", 0.55, 3.45)
    _style(fig, "Rate by class", f"bar = mean over windows (DERIVED); dots = per-window sealed rates; {WINDOWS_NOTE}")
    top = max(max(r[c] for c in CLASSES) for r in d["rates"])
    fig.update_yaxes(title_text="rate (Hz)", range=[-0.03 * max(top, E_BAND_HZ[1]), 1.12 * max(top, E_BAND_HZ[1])])
    rows = [(c, *[r[c] for r in d["rates"]], sum(r[c] for r in d["rates"]) / n, d["tonic"].get(c, "")) for c in CLASSES]
    table = _table(("class", *[f"window {i + 1} Hz" for i in range(n)], "mean Hz (DERIVED)", "tonic drive"), rows)
    return [_fig_html(fig, f"{rec['id']}-rates"),
            _box("OBSERVED", " Per-window class rates from the sealed record. Tonic drive is the predeclared "
                             "per-class vector (drive units, shown in the table only).", table)]


def _panel_irregularity(rec: Dict[str, Any]) -> List[str]:
    d = rec["data"]
    w = list(range(1, len(d["cv"]) + 1))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Fraction of cells with ISI-CV in [0.5, 1.5]", "E rate-CV across cells"))
    for col, key, thr, label in ((1, "frac_in", ISI_CV_FRAC_MIN, f"gate {ISI_CV_FRAC_MIN:g}"),
                                 (2, "E_rate_CV", E_RATE_CV_MIN, f"gate {E_RATE_CV_MIN:g}")):
        fig.add_trace(go.Scatter(x=w, y=[v[key] for v in d["cv"]], mode="markers+lines",
                                 line=dict(color=ACCENT, width=2), marker=dict(size=9, color=ACCENT),
                                 hovertemplate=f"window %{{x}}: {key} %{{y:.3g}}<extra></extra>"), row=1, col=col)
        _threshold(fig, thr, label, 0.8, len(w) + 0.2, row=1, col=col)
        fig.update_xaxes(title_text="analysis window", tickvals=w, range=[0.7, len(w) + 0.3], row=1, col=col)
    fig.update_yaxes(range=[-0.04, 1.05], row=1, col=1)
    ecv_top = max(max(v["E_rate_CV"] for v in d["cv"]), E_RATE_CV_MIN)
    fig.update_yaxes(range=[-0.04 * ecv_top, 1.15 * ecv_top], row=1, col=2)
    _style(fig, "Irregularity", WINDOWS_NOTE)
    table = _table(("window", "CV-evaluable cells", "frac_in", "E rate-CV"),
                   [(i, v["cv_evaluable"], v["frac_in"], v["E_rate_CV"]) for i, v in zip(w, d["cv"])])
    return [_fig_html(fig, f"{rec['id']}-irregularity"),
            _box("OBSERVED", " CV-evaluable = cells with >= 4 spikes in the window. Values are stored rounded "
                             "to 3 dp.", table)]


def _panel_stability(rec: Dict[str, Any]) -> List[str]:
    d = rec["data"]
    w = list(range(1, len(d["sync"]) + 1))
    derived = recompute_drift(d)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Class-rate drift across windows", "Population sync per window"))
    for c in CLASSES:
        fig.add_trace(go.Bar(x=[c], y=[d["drift"][c]], marker=dict(color=CLASS_COLORS[c], cornerradius=4),
                             width=0.3, text=[f"{d['drift'][c]:.3g}"], textposition="outside", cliponaxis=False,
                             hovertemplate=f"{c}: drift %{{y:.3g}}<extra></extra>"), row=1, col=1)
    _threshold(fig, DRIFT_MAX, f"gate {DRIFT_MAX:g}", -0.45, 3.45, row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=d["sync"], mode="markers+lines", line=dict(color=ACCENT, width=2),
                             marker=dict(size=9, color=ACCENT),
                             hovertemplate="window %{x}: sync %{y:.3g}<extra></extra>"), row=1, col=2)
    _threshold(fig, SYNC_MAX, f"gate < {SYNC_MAX:g}", 0.8, len(w) + 0.2, row=1, col=2)
    fig.update_xaxes(title_text="analysis window", tickvals=w, range=[0.7, len(w) + 0.3], row=1, col=2)
    dtop = max(max(d["drift"].values()), DRIFT_MAX)
    fig.update_yaxes(range=[-0.04 * dtop, 1.2 * dtop], row=1, col=1)
    stop = max(max(d["sync"]), SYNC_MAX)
    fig.update_yaxes(range=[-0.04 * stop, 1.3 * stop], row=1, col=2)
    _style(fig, "Stability", WINDOWS_NOTE)
    rows = [(c, d["drift"][c], derived[c]) for c in CLASSES]
    table = _table(("class", "drift (sealed)", "drift from sealed rates (DERIVED)"), rows)
    table += _table(("window", "sync"), list(zip(w, d["sync"])))
    return [_fig_html(fig, f"{rec['id']}-stability"),
            _box("OBSERVED", " Drift = (max - min) / mean of class rate over windows. Sync = fraction of time "
                             "bins where more than half of E cells spike.", table)]


def _panel_currents(rec: Dict[str, Any]) -> List[str]:
    d = rec["data"]
    cur = d["currents"]
    n = len(cur)
    pathways = [k for k in cur[0] if "I_mean" in cur[0][k]]
    classes = [c for c in CLASSES if c in cur[0] and "syn_mean" in cur[0][c]]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, column_widths=[0.4, 0.6],
                        subplot_titles=("Presynaptic trace mean by class", "Pathway current mean"))
    for c in classes:
        m = sum(w[c]["syn_mean"] for w in cur) / n
        fig.add_trace(go.Bar(x=[c], y=[m], marker=dict(color=CLASS_COLORS[c], cornerradius=4), width=0.4,
                             hovertemplate=f"{c}: %{{y:.4g}}<extra></extra>"), row=1, col=1)
    means = [sum(w[p]["I_mean"] for w in cur) / n for p in pathways]
    fig.add_trace(go.Bar(x=pathways, y=means, marker=dict(color=ACCENT, cornerradius=4), width=0.4,
                         hovertemplate="%{x}: %{y:.4g}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="#30363d", width=1), row=1, col=2)
    fig.update_yaxes(title_text="a.u.", row=1, col=1)
    fig.update_yaxes(title_text="a.u.", row=1, col=2)
    _style(fig, "Recorded currents", f"mean of {n} windows (DERIVED); recorded, never tuned toward")
    rows = [(k, *[w[k]["syn_mean"] for w in cur]) for k in classes]
    rows += [(k, *[w[k]["I_mean"] for w in cur]) for k in pathways]
    table = _table(("quantity", *[f"window {i + 1}" for i in range(n)]), rows)
    return [_fig_html(fig, f"{rec['id']}-currents"),
            _box("OBSERVED", " Class rows: mean synaptic trace (syn_mean). Pathway rows (pre+post class, e.g. "
                             "SSTE = SST to E): mean over edges of weight x presynaptic class-mean trace "
                             "(I_mean). run_regime uses the class-mean trace, exact only for uniform rates "
                             "within a class. Proxy units.", table)]


_PANEL_FNS = {"gates.html": _panel_gates, "rates.html": _panel_rates,
              "irregularity.html": _panel_irregularity, "stability.html": _panel_stability,
              "currents.html": _panel_currents}


# ---------------------------------------------------------------- build


def _label(family: str, regime: str) -> str:
    return f"{FAMILY_LABELS.get(family, family)} {regime.upper()}"


def _prov_table(rec: Dict[str, Any], lineage: Dict[str, Any], family: str, commit: str | None) -> str:
    rows = [("source", rec["source"]), ("sha256", rec["sha256"]),
            ("lineage", lineage["source"]), ("lineage sha256", lineage["sha256"]),
            ("parent", lineage["data"].get("parent", "")), ("family verdict", lineage["data"].get("verdict", "")),
            ("producer", FAMILY_PRODUCERS.get(family, "unknown")), ("simulation", "none (sealed values only)")]
    if commit:
        rows.append(("repo commit", commit))
    return _table(("field", "value"), rows)


def build_regime_atlases(
    out_dir: str = "outputs/atlases",
    results_dir: str = "results",
    family: str = "v21",
    regimes: Sequence[str] = ("low", "mid", "high"),
    commit: str | None = None,
) -> Dict[str, Any]:
    """Write one atlas per sealed regime plus a cross-regime index. Returns the manifest."""
    loaded = load_regimes(results_dir, family, regimes)
    records, lineage = loaded["records"], loaded["lineage"]
    os.makedirs(out_dir, exist_ok=True)
    manifest_atlases = []

    for rec in records:
        d = rec["data"]
        adir = os.path.join(out_dir, rec["id"])
        os.makedirs(adir, exist_ok=True)
        title = _label(family, d["regime"])
        nav_links = ["<a href='../index.html'>&larr; all regimes</a>", "<a href='index.html'>atlas</a>"]
        nav_links += [f"<a href='{f}'>{p}</a>" for f, p in PANELS]
        nav = "".join(nav_links)
        sub = f"{_status(bool(d['pass']))} &middot; sealed record <code>{_esc(rec['source'])}</code>"
        files = []
        for fname, pname in PANELS:
            html_text = _page(f"{title} — {pname}", sub, nav, _PANEL_FNS[fname](rec))
            path = os.path.join(adir, fname)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(html_text)
            files.append({"file": fname, "panel": pname, "sha256": _sha256(path)})
        gates = recheck_gates(d)
        gate_rows = "".join(f"<tr><td>{g}</td><td>{_status(bool(d['checks'].get(g)))}</td>"
                            f"<td>{_esc(info['worst'])}</td></tr>" for g, info in gates.items())
        panel_list = "".join(f"<tr><td><a href='{f}'>{p}</a></td></tr>" for f, p in PANELS)
        missing = _table(("not rendered", "reason"), list(UNAVAILABLE.items()))
        sections = [
            _box("OBSERVED", " Sealed gate checks",
                 f"<table><thead><tr><th>gate</th><th>sealed</th><th>worst observed</th></tr></thead>"
                 f"<tbody>{gate_rows}</tbody></table>"),
            f"<div class='box'><table><thead><tr><th>panels</th></tr></thead><tbody>{panel_list}</tbody></table></div>",
            f"<div class='box'>{missing}</div>",
            f"<div class='box'>{_prov_table(rec, lineage, family, commit)}</div>",
        ]
        with open(os.path.join(adir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
            f.write(_page(title, sub, nav, sections))
        manifest_atlases.append({"id": rec["id"], "regime": d["regime"], "source": rec["source"],
                                 "source_sha256": rec["sha256"], "sealed_pass": bool(d["pass"]),
                                 "recomputed_pass": all(g["pass"] for g in gates.values()),
                                 "panels": files})

    _write_overview(out_dir, family, records, lineage, commit)
    manifest = {"suite": "regime_atlas.v1", "family": family, "simulation": "none",
                "lineage": {"source": lineage["source"], "sha256": lineage["sha256"],
                            "verdict": lineage["data"].get("verdict")},
                "unavailable": UNAVAILABLE, "atlases": manifest_atlases}
    if commit:
        manifest["commit"] = commit
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _write_overview(out_dir: str, family: str, records: List[Dict[str, Any]],
                    lineage: Dict[str, Any], commit: str | None) -> None:
    labels = [r["data"]["regime"].upper() for r in records]
    gate_names = list(recheck_gates(records[0]["data"]))
    head = "<th>gate</th>" + "".join(f"<th><a href='{r['id']}/index.html'>{_esc(l)}</a></th>"
                                     for r, l in zip(records, labels))
    body = "".join("<tr><td>" + g + "</td>" + "".join(f"<td>{_status(bool(r['data']['checks'].get(g)))}</td>"
                                                      for r in records) + "</tr>" for g in gate_names)
    body += "<tr><td><b>pass</b></td>" + "".join(f"<td>{_status(bool(r['data']['pass']))}</td>"
                                                for r in records) + "</tr>"
    gate_table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    fig = go.Figure()
    means = {c: [sum(w[c] for w in r["data"]["rates"]) / len(r["data"]["rates"]) for r in records] for c in CLASSES}
    top = max(max(v) for v in means.values())
    min_gap = 0.06 * max(top, 1.0)
    label_y: Dict[str, float] = {}
    for c in sorted(CLASSES, key=lambda k: means[k][-1]):  # spread end labels upward
        y = means[c][-1]
        if label_y:
            y = max(y, max(label_y.values()) + min_gap)
        label_y[c] = y
    for c in CLASSES:
        fig.add_trace(go.Scatter(x=labels, y=means[c], mode="lines+markers", name=c,
                                 line=dict(color=CLASS_COLORS[c], width=2),
                                 marker=dict(size=10, symbol=CLASS_SYMBOLS[c], color=CLASS_COLORS[c],
                                             line=dict(color="#161b22", width=2)),
                                 hovertemplate=f"{c} %{{x}}: %{{y:.4g}} Hz<extra></extra>"))
        fig.add_annotation(x=labels[-1], y=label_y[c], text=f"{c} {means[c][-1]:.3g}", showarrow=False,
                           xanchor="left", xshift=12, font=dict(color="#c9d1d9", size=12))
    _style(fig, "Class rate by regime", "mean over analysis windows (DERIVED) · x is regime order, not a scale",
           height=460)
    fig.update_layout(margin=dict(l=60, r=110, t=90, b=60))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="rate (Hz)", rangemode="tozero")
    fig.update_xaxes(title_text="tonic regime")
    rate_table = _table(("class", *labels), [(c, *means[c]) for c in CLASSES])
    tonic_table = _table(("class", *labels), [(c, *[r["data"]["tonic"].get(c, "") for r in records]) for c in CLASSES])

    lin = lineage["data"]
    title = f"{FAMILY_LABELS.get(family, family)} regime atlases"
    sub = (f"family verdict <b>{_esc(lin.get('verdict', ''))}</b> &middot; passing subset "
           f"{_esc(lin.get('passing_subset', []))} &middot; selection {_esc(lin.get('selection', ''))}")
    nav = "".join(f"<a href='{r['id']}/index.html'>{_esc(l)}</a>" for r, l in zip(records, labels))
    prov = [("lineage", lineage["source"]), ("lineage sha256", lineage["sha256"])]
    prov += [(r["source"], r["sha256"]) for r in records]
    prov += [("simulation", "none (sealed values only)")]
    if commit:
        prov.append(("repo commit", commit))
    sections = [
        _box("OBSERVED", " Sealed gate checks per regime", gate_table),
        _fig_html(fig, f"{family}-overview-rates"),
        _box("DERIVED", " Mean class rate per regime (Hz)", rate_table),
        _box("OBSERVED", " Predeclared tonic drive per class", tonic_table),
        f"<div class='box'>{_table(('not rendered', 'reason'), list(UNAVAILABLE.items()))}</div>",
        f"<div class='box'>{_table(('provenance', 'value'), prov)}</div>",
    ]
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(_page(title, sub, nav, sections))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/atlases")
    ap.add_argument("--results", default="results")
    ap.add_argument("--family", default="v21")
    ap.add_argument("--regimes", default="low,mid,high")
    args = ap.parse_args(argv)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = None
    m = build_regime_atlases(args.out, args.results, args.family, args.regimes.split(","), commit)
    for a in m["atlases"]:
        print(f"{a['id']}: sealed_pass={a['sealed_pass']} recomputed_pass={a['recomputed_pass']} "
              f"panels={len(a['panels'])}")
    print(f"wrote {os.path.join(args.out, 'index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
