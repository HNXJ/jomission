"""Pages build policy tests (workstream B).

Mechanical checks: deterministic build, values equal sources, evidence
machinery, firewall, quarantine flags, no path leaks, link integrity,
scientific-tree identity. No simulations.
"""

import filecmp
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(REPO, "scripts", "build_pages.py")
TMP = os.path.join(REPO, ".pages_test_tmp")

SCIENTIFIC_DIRS = ["results", "manifests", "jomission", "tests", "scripts", "site-src"]


def tree_hash():
    h = hashlib.sha256()
    for d in SCIENTIFIC_DIRS:
        root = os.path.join(REPO, d)
        for base, _, files in os.walk(root):
            for fn in sorted(files):
                if fn.endswith((".pyc", ".pyo")) or "__pycache__" in base:
                    continue
                p = os.path.join(base, fn)
                rel = os.path.relpath(p, REPO).replace(os.sep, "/")
                h.update(rel.encode())
                with open(p, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def run_build(out, commit="testcommit0"):
    env = dict(os.environ, JOMISSION_PAGES_COMMIT=commit)
    r = subprocess.run([sys.executable, BUILD, "--out", out], cwd=REPO,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return out


@pytest.fixture(scope="module")
def site_a():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    before = tree_hash()
    a = run_build(os.path.join(TMP, "a"))
    yield a
    assert tree_hash() == before, "build mutated scientific tree"
    shutil.rmtree(TMP, ignore_errors=True)


def _files(d):
    out = {}
    for base, _, files in os.walk(d):
        for fn in files:
            p = os.path.join(base, fn)
            out[os.path.relpath(p, d).replace(os.sep, "/")] = p
    return out


def test_build_deterministic(site_a):
    b = run_build(os.path.join(TMP, "b"))
    fa, fb = _files(site_a), _files(b)
    assert set(fa) == set(fb)
    for k in fa:
        assert filecmp.cmp(fa[k], fb[k], shallow=False), k


def test_manifest_valid():
    import sys
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    man = json.load(open(os.path.join(REPO, "site-src", "manifest.json")))
    for fn in man["results_allowlist"]:
        json.load(open(os.path.join(REPO, "results", fn)))
    for fn in man["results_manifests_allowlist"]:
        json.load(open(os.path.join(REPO, "manifests", fn)))
    import build_pages as bp
    for pid, p in man.get("panels", {}).items():
        assert p.get("evidence"), pid
        assert p.get("source") or p.get("note"), pid
        if p.get("type", "figure") == "figure":
            assert p.get("renderer") in bp.RENDERERS, pid
            src = p["source"]
            assert src is None or src.split("/")[0] in ("results", "manifests"), pid
        if p.get("type") == "embed":
            assert p["plotly"] in man["plotly_allowlist"], pid
    for g in man["gates"]:
        assert g["status"] in ("PASS", "FAIL", "LOCKED"), g


def test_renderers_deterministic_and_labeled():
    import sys
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import build_pages as bp
    man = json.load(open(os.path.join(REPO, "site-src", "manifest.json")))
    ctx = {"manifest": man, "commit": "unittest", "_out": TMP}
    a = bp.render_gate_graph(ctx)[1]
    b = bp.render_gate_graph(ctx)[1]
    assert a == b
    for g in man["gates"]:
        assert g["status"] in a


def test_values_equal_sources(site_a):
    # End-to-end: generated figures carry source values verbatim.
    # Present once panels are wired (commit 3 population); skeleton has none.
    man = json.load(open(os.path.join(REPO, "site-src", "manifest.json")))
    if not man.get("panels"):
        pytest.skip("no panels registered yet")
    nt = json.load(open(os.path.join(REPO, "results", "native_transformation.json")))
    svg = open(os.path.join(site_a, "assets", "svg", "g_curve.svg"), encoding="utf-8").read()
    for name in ("rest", "transition", "active", "above"):
        r = nt[name]
        assert f"{r['rate']:.4g}" in svg, name
        assert f"{r['w']:.4g}" in svg, name
    gg = open(os.path.join(site_a, "assets", "svg", "gate_graph.svg"), encoding="utf-8").read()
    for g in man["gates"]:
        assert g["status"] in gg
        assert g["label"].split()[0] in gg


def test_evidence_machinery(site_a):
    # Every rendered panel section carries evidence badge + source footer.
    for base, _, files in os.walk(site_a):
        for fn in files:
            if fn.endswith(".html") and "/assets/plotly/" not in base:
                t = open(os.path.join(base, fn), encoding="utf-8").read()
                for m in re.finditer(r'<section class="panel" id="([^"]+)">', t):
                    seg = t[m.start():m.start() + 4000]
                    assert "badge-" in seg, m.group(1)
                    assert "Source:" in seg, m.group(1)


def test_firewall_and_quarantine(site_a):
    man = json.load(open(os.path.join(REPO, "site-src", "manifest.json")))
    blob = ""
    for base, _, files in os.walk(site_a):
        for fn in files:
            if fn.endswith((".html", ".svg", ".json")):
                blob += open(os.path.join(base, fn), encoding="utf-8",
                             errors="ignore").read()
    for bad in man.get("prohibited_strings", []):
        assert bad not in blob, bad
    assert "QUARANTINED" in blob or "quarantine" in blob.lower()


def test_no_absolute_paths(site_a):
    pat = re.compile(r"[A-Z]:[\\/]|/home/|/mnt/|/tmp/|\\\\")
    for base, _, files in os.walk(site_a):
        for fn in files:
            if fn.endswith((".html", ".svg", ".css")):
                t = open(os.path.join(base, fn), encoding="utf-8", errors="ignore").read()
                assert not pat.search(t), f"{fn}: path leak"


def test_links_resolve(site_a):
    files = _files(site_a)
    for rel, p in files.items():
        if not rel.endswith(".html") or "/assets/plotly/" in rel:
            continue
        t = open(p, encoding="utf-8").read()
        for m in re.finditer(r'(?:href|src)="([^"#]+)"', t):
            url = m.group(1)
            if url.startswith(("http", "mailto:")):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(rel), url)).replace(os.sep, "/")
            assert target in files, f"{rel} -> missing {url}"
