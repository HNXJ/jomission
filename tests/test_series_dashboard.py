"""Series dashboard: values come from repo files, five pages exist, the omission firewall holds."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_series_dashboard as bsd  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = bsd.build(tmp_path_factory.mktemp("dash"))
    html = path.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    return html, json.loads(m.group(1).replace("<\\/", "</"))


def test_figure_plan_counts_and_fields(built):
    _, d = built
    series = {s["id"]: s for s in d["figures"]["series"]}
    assert set(series) == {"omission", "lansa"}
    for s in series.values():
        assert s["main"] == 7, s["id"]
        ids = [f["id"] for f in s["figures"]]
        assert len(ids) == len(set(ids))
        assert sum(s["by_status"].values()) == s["main"] + s["supplements"]
    n = series["lansa"]["supplements"]
    assert 2 * 7 <= n <= 3 * 7, "supplements about 2 to 3 times the main count"
    raw = json.loads((ROOT / "docs/FIGURE_PLAN.json").read_text(encoding="utf-8"))
    for s in raw["series"].values():
        for f in s["figures"]:
            assert f["status"] in {"PLANNED", "BLOCKED", "TBD"}, f["id"]
            if f["status"] == "BLOCKED":
                assert f["blocked_by"], f"{f['id']} is BLOCKED with no blocker"
            if f["status"] == "TBD":
                assert f["owner_status"] == "OWNER_TBD", f["id"]
            if f["owner_status"] != "OWNER_TBD":
                assert f["title"] and f["content"] and f["sources"], f["id"]


def test_minimal_model_status_matches_repo(built):
    _, d = built
    mm = d["laminar"]["minimal"]
    ids = [e["id"] for e in mm["experiments"]]
    assert ids[0] == "E0" and len(ids) == len(set(ids)) == 8
    assert [e["id"] for e in mm["experiments"] if e["authorized"]] == ["E0"], "only E0 is sealed and authorized"
    assert all(e["status"] == "SEALED" for e in mm["experiments"] if e["authorized"])
    assert (ROOT / mm["spec"]).exists() and (ROOT / "docs/laminar_motif/E0_SEAL.md").exists()
    for g in mm["e0_gates"]:
        assert g["state"] in {"PASS", "OPEN", "QUEUED", "FAIL"}, g
        if g["state"] == "PASS":
            path, _, fn = g["evidence"].partition("::")
            text = (ROOT / path).read_text(encoding="utf-8")
            assert f"def {fn}(" in text, f"gate {g['id']} cites a missing test"
        else:
            assert g["evidence"] == "", "a gate without a pass carries no evidence"


def test_conventions_present_and_lowercase_panels(built):
    _, d = built
    c = d["figures"]["conventions"]
    assert any("lowercase" in m and "panel" in m for m in c["match"]), "mains use bold lowercase panel letters"
    assert "cell-type" in c["constraint"]


def test_omission_figures_hold_no_outcome(built):
    _, d = built
    om = next(s for s in d["figures"]["series"] if s["id"] == "omission")
    for f in om["figures"]:
        if f["id"] not in ("om-01", "om-s01", "om-s02", "om-s03"):
            assert f["status"] == "TBD" and not f["content"], f["id"]


def test_five_pages_present(built):
    html, _ = built
    assert not re.search(r"<!doctype|<html|<head|<body", html, re.I), "publish step adds the skeleton"
    assert "<title>" in html and 'name="claude:viewport-fit-sides"' in html
    for pid in ("overview", "omission", "laminar", "figures", "gates", "todo"):
        assert f"['{pid}'," in html, pid


def test_counts_match_sources(built):
    _, d = built
    todo = json.loads((ROOT / "manifests/todo.json").read_text(encoding="utf-8"))["items"]
    lin = json.loads((ROOT / "manifests/lineage_registry.json").read_text(encoding="utf-8"))["lineages"]
    n_open = len(re.findall(r"^\| OPEN-\d+ \|", (ROOT / "docs/OPEN.md").read_text(encoding="utf-8"), re.M))
    assert len(d["todo"]["items"]) == len(todo) > 0
    assert d["overview"]["lineages"] == len(lin) > 0
    assert d["overview"]["open_rows"] == len(d["todo"]["open"]) == n_open > 0
    assert sum(d["overview"]["todo_by_status"].values()) == len(todo)


def test_checklist_score_is_sum_of_rows(built):
    _, d = built
    plan = d["laminar"]["plan"]
    assert len(plan) == 25, "the plan has 25 items"
    assert d["laminar"]["scores"]["builder"] == sum(r["builder"] for r in plan)
    assert d["laminar"]["scores"]["builder"] == 4 * sum(1 for r in plan if r["builder"] == 4)
    assert all(r["builder"] in (0, 4) for r in plan), "no partial credit"
    assert d["overview"]["builder_score"] == d["laminar"]["scores"]["builder"]


def test_firewall_holds_and_can_fail(built, monkeypatch, tmp_path):
    html, _ = built
    bad_list = bsd.prohibited()
    assert bad_list, "the firewall list must not be empty"
    for bad in bad_list:
        assert bad not in html
    real = bsd.collect

    def poisoned():
        x = real()
        x["omission"]["paradigm_note"] = "T1 POSITIVE"
        return x

    monkeypatch.setattr(bsd, "collect", poisoned)
    with pytest.raises(SystemExit):
        bsd.build(tmp_path)


def test_no_innerhtml_and_no_absolute_paths(built):
    html, _ = built
    assert "innerHTML" not in html
    assert not re.search(r"[A-Z]:[\\/]|/home/|/mnt/|/tmp/", html)
