"""Regenerate the Pages spectrolaminar page: every motif figure and the run table, registered in site-src/manifest.json.

Usage:
  python scripts/e2_pages_spectrolaminar.py

Owns, and rewrites on every call: site-src/pages/spectrolaminar.html, manifest panels whose id starts with "sl-",
and viz_allowlist keys under OWNED_DIRS. Everything else in the manifest is left as is. Figures are copied by
scripts/build_pages.py from results/viz/; this script only lists those present on disk (sorted, so the output is
deterministic). Run scripts/e2_concept_panels.py first for results/viz/spectrolaminar/runs/ and runs.json.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "results" / "viz"
MAN = ROOT / "site-src" / "manifest.json"
PAGE = ROOT / "site-src" / "pages" / "spectrolaminar.html"
OWNED_DIRS = ("schema/", "t1/", "paper_w4/", "paper_w5/", "paper_w6/", "spectrolaminar/")
RUN_FAMILIES = (("t1_", "T1 axes (kd, sst), n4000, three seeds"), ("lfs_", "Size ladder (n1000-n4000)"),
                ("sw_", "First-pass sweep (k, kd, dsz, rho)"), ("e2x_", "Exploratory per-unit weights (seed 1, unsealed)"),
                ("", "Other E2 runs"))


def dump(obj, kw: dict, nl: bytes, tail: bytes) -> bytes:
    """The manifest's own byte form: json.dumps with its options, its newline, its trailing newline or none."""
    return json.dumps(obj, **kw).encode().replace(b"\n", nl) + tail


def figs(sub: str) -> list[str]:
    return sorted(f"{sub}{p.name}" for p in (VIZ / sub).glob("*.png"))


def main() -> None:
    raw = MAN.read_bytes()
    man = json.loads(raw)
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    tail = nl if raw.endswith(nl) else b""
    forms = [kw for kw in ({"indent": 2}, {"indent": 2, "sort_keys": True})
             if dump(man, kw, nl, tail) == raw]
    if not forms:
        sys.exit("manifest.json is not in a json.dumps(indent=2) form; refusing to rewrite it")
    runs = json.loads((VIZ / "spectrolaminar" / "runs.json").read_text(encoding="utf-8"))["runs"]
    by_tag = {r["tag"]: r for r in runs}
    man["panels"] = {k: v for k, v in man["panels"].items() if not k.startswith("sl-")}
    man["viz_allowlist"] = {k: v for k, v in man["viz_allowlist"].items() if not k.startswith(OWNED_DIRS)}
    body: list[str] = []
    n = 0

    def add(key: str, title: str, ev: str = "OBSERVED", status: str | None = None, note: str | None = None) -> None:
        nonlocal n
        n += 1
        pid = f"sl-{n:03d}"
        man["viz_allowlist"][key] = {"evidence_class": ev}
        p = {"evidence_class": ev, "page": "spectrolaminar", "source": f"results/viz/{key}", "title": title,
             "type": "viz", "viz": key}
        if status:
            p["status"] = status
        if note:
            p["note"] = note
        man["panels"][pid] = p
        body.append(f'<div class="panel-slot" data-panel="{pid}"></div>')

    n_pass = sum(r["pass"] for r in runs)
    best = max(runs, key=lambda r: r["r_proto"])
    body += ["<h1>Spectrolaminar motif from cell densities</h1>",
             '<p class="status-note">Unsealed working figures (layout after Herrera 2026: data and circuit, then '
             f'observed vs model relative-power maps). {len(runs)} scored runs, {n_pass} pass the six-part scorer; '
             f'highest r_proto {best["r_proto"]:.3f} ({html.escape(best["tag"])}). Scorer and bands: '
             "docs/laminar_motif/FINDINGS_2026-11-10.md.</p>",
             "<h2>Data and circuit</h2>"]
    for k in figs("schema/"):
        add(k, "LANSA densities, observed motif, model column and built vs LANSA depth profiles",
            note="Rows: data (no simulation); the model as built. LANSA densities are relative per type: "
                 "depth profiles are compared, never fractions across types.")
    body.append("<h2>Architecture and reference run</h2>")
    for k in figs("t1/"):  # paper_w3 PNGs are byte copies of t1/V0 and t1/V1; one home
        add(k, Path(k).stem.replace("_", " "))
    body += ["<h2>All scored runs: configuration and verdict</h2>",
             '<div class="panel-slot" data-panel="sl-table"></div>']
    man["panels"]["sl-table"] = {"evidence_class": "DERIVED", "keys": ["runs"], "page": "spectrolaminar",
                                 "source": "results/viz/spectrolaminar/runs.json",
                                 "title": "E2 runs: sizes E/PV/SOM per unit, p, SOM->E tau, weight modifiers, scores",
                                 "type": "table"}
    man["viz_allowlist"]["spectrolaminar/runs.json"] = {"evidence_class": "DERIVED"}
    body.append("<h2>CSD, probe geometry, seed differences</h2>")
    for k in figs("paper_w4/") + figs("paper_w5/") + figs("paper_w6/"):
        add(k, Path(k).stem.replace("_", " "))
    left = sorted(by_tag)
    for prefix, head in RUN_FAMILIES:
        fam = [t for t in left if t.startswith(prefix)]
        left = [t for t in left if t not in fam]
        if not fam:
            continue
        body.append(f"<h2>{html.escape(head)}</h2>")
        for t in fam:
            r = by_tag[t]
            add(f"spectrolaminar/runs/{t}.png", f"{t}: observed vs model map", status="PASS" if r["pass"] else "FAIL",
                note=f"r_proto {r['r_proto']:.3f}, r_ab {r['r_ab']:.3f}, r_gm {r['r_gm']:.3f}, "
                     f"crossing {r['crossing_2se']}; w {r['w_mods']}")
    MAN.write_bytes(dump(man, forms[0], nl, tail))
    PAGE.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"{n} figure panels + 1 table; {len(runs)} runs, {n_pass} pass")


if __name__ == "__main__":
    main()
