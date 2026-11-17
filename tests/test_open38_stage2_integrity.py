"""Stage-2 integrity with execution manifests, on synthetic runs: a planted complete batch passes; execution IDs are
unique, propagate to the stage log and bind the outputs; a duplicated, retagged, unfinished or tampered record fails;
a stop state is reported without failing; the smoke subset is checked alone."""
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
SPEC = importlib.util.spec_from_file_location("integ2", ROOT / "scripts" / "open38_stage1_integrity.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
import open38_exec_manifest as em  # noqa: E402
import test_open38_smoke_check as fx  # noqa: E402

HEAD = "f" * 40
LOG = "o38s_s2_stage.log"


def _plant_one(d: Path, i: int, arm: str, s: int, deep_spikes=5, log=True):
    t = mod.tag2(arm, s)
    eid = em.begin(d, t, HEAD)
    fx._write(d, t, r_ab=0.5, sha=None, counts_bump=s + 10 * i, deep_spikes=deep_spikes)
    rec = json.loads((d / f"{t}.json").read_text())
    rec.update(seed=s, layout=mod.LAYOUT2[arm], T_s=8.0, git_sha=HEAD, git_dirty=False,
               jaxfne_file="D:/repos/_work/_temp_jaxfne_351f6048/jaxfne/__init__.py",
               wall_s={"construct": 9.0 + 0.1 * i, "simulate": 175.0 + s, "total": 190.0 + s + i})
    (d / f"{t}.json").write_text(json.dumps(rec))
    (d / f"{t}_similarity.json").write_text("{}")
    (d / f"{t}_motif.json").write_text("{}")
    em.end(d, t, 0)
    if log:
        with (d / LOG).open("a", encoding="utf-8") as f:
            f.write(f"2026-11-17T00:00:00Z start {t} {mod.LAYOUT2[arm]} exec={eid}\n")
    return t, eid


def _plant(d: Path):
    return {mod.tag2(a, s): _plant_one(d, i, a, s)[1] for i, a in enumerate(mod.ARMS2) for s in range(1, 6)}


def _run(d: Path, runs=None):
    rc = mod.main(str(d / "integ.json"), d, HEAD, 2, runs, d / LOG)
    rep = json.loads((d / "integ.json").read_text())
    return rc, rep["checks"], rep["warnings"]


def _rewrite(d, t, **kw):
    p = em.path(d, t)
    rec = json.loads(p.read_text())
    rec.update(kw)
    p.write_text(json.dumps(rec))


def test_complete_stage2_batch_passes_with_unique_ids(tmp_path):
    ids = _plant(tmp_path)
    rc, c, _ = _run(tmp_path)
    assert rc == 0, c
    assert len(ids) == 30 and len(set(ids.values())) == 30


def test_manifest_refuses_reuse_and_double_end(tmp_path):
    t = mod.tag2("v1", 1)
    em.begin(tmp_path, t, HEAD)
    with pytest.raises(SystemExit):
        em.begin(tmp_path, t, HEAD)
    em.end(tmp_path, t, 0)
    with pytest.raises(SystemExit):
        em.end(tmp_path, t, 0)
    with pytest.raises(SystemExit):
        em.end(tmp_path, mod.tag2("mt", 1), 0)  # never begun


def test_propagation_requires_the_logged_id(tmp_path):
    _plant(tmp_path)
    t = mod.tag2("teo", 2)
    log = (tmp_path / LOG).read_text()
    eid = json.loads(em.path(tmp_path, t).read_text())["exec_id"]
    (tmp_path / LOG).write_text(log.replace(f"exec={eid}", "exec=" + "0" * 32))
    rc, c, _ = _run(tmp_path)
    assert rc == 1 and c["runs"]["ok"] and c["propagation"]["missing"] == [t]


def test_duplicated_execution_record_fails(tmp_path):
    _plant(tmp_path)
    a, b = mod.tag2("chc", 1), mod.tag2("chc", 3)
    rec_a = json.loads(em.path(tmp_path, a).read_text())
    rec_b = json.loads(em.path(tmp_path, b).read_text())
    rec_b["exec_id"] = rec_a["exec_id"]  # one execution claimed by two tags, otherwise consistent
    em.path(tmp_path, b).write_text(json.dumps(rec_b))
    log = (tmp_path / LOG).read_text()
    (tmp_path / LOG).write_text(re.sub(rf"(start {b} \S+ exec=)\w+", rf"\g<1>{rec_a['exec_id']}", log))
    rc, c, _ = _run(tmp_path)
    assert rc == 1 and c["runs"]["ok"] and c["propagation"]["ok"], "fixture must pass every other check"
    assert [a, b] in c["provenance_unique"]["shared_exec_id"]


def test_copied_manifest_fails_on_tag_and_hashes(tmp_path):
    _plant(tmp_path)
    a, b = mod.tag2("mst", 4), mod.tag2("mst", 5)
    shutil.copyfile(em.path(tmp_path, a), em.path(tmp_path, b))
    rc, c, _ = _run(tmp_path)
    probs = " ".join(c["runs"]["failed"][b])
    assert rc == 1 and "tag" in probs and "hash differs" in probs


def test_output_changed_after_end_fails(tmp_path):
    _plant(tmp_path)
    t = mod.tag2("shape", 2)
    (tmp_path / f"{t}_verdict.json").write_text(json.dumps({"r_ab": 0.51, "r_gm": 0.4, "r_proto": 0.3}))
    rc, c, _ = _run(tmp_path)
    assert rc == 1 and any("_verdict.json hash differs" in p for p in c["runs"]["failed"][t])


def test_unfinished_or_failed_execution_fails(tmp_path):
    _plant(tmp_path)
    _rewrite(tmp_path, mod.tag2("v1", 3), rc=1)
    p = em.path(tmp_path, mod.tag2("mt", 3))
    rec = json.loads(p.read_text())
    del rec["end_utc"]
    p.write_text(json.dumps(rec))
    rc, c, _ = _run(tmp_path)
    f = c["runs"]["failed"]
    assert rc == 1 and {mod.tag2("v1", 3), mod.tag2("mt", 3)} <= set(f)


def test_stop_state_is_reported_not_failed(tmp_path):
    _plant_one(tmp_path, 0, "v1", 1, deep_spikes=0)
    rc, c, w = _run(tmp_path, [("v1", 1)])
    assert rc == 0, c
    assert w["stop_states"] == {mod.tag2("v1", 1): f"deep-unit E rate < {mod.sc.DEEP_MIN_HZ} Hz"}


def test_smoke_subset_ignores_other_lineages_but_not_stage2_extras(tmp_path):
    _plant_one(tmp_path, 0, "v1", 1)
    for other in ("o38s_ref_n4000_s1_T8", "o38s_nc_n4000_s1_T8", "o38s_pc_ref_n4000_s1_T8"):
        (tmp_path / f"{other}.json").write_text("{}")
    rc, c, _ = _run(tmp_path, [("v1", 1)])
    assert rc == 0, c
    _plant_one(tmp_path, 1, "mt", 1)
    rc, c, _ = _run(tmp_path, [("v1", 1)])
    assert rc == 1 and c["no_extra"]["extra"] == [mod.tag2("mt", 1)]


def test_layouts_equal_spec_table():
    spec = (ROOT / "docs/laminar_motif/SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| (v1|mt|mst|teo|chc|shape) \|.*\| `(_rhou\w+)` \|$", spec, re.M))
    assert len(rows) == 6 and {a: mod.BASE + s for a, s in rows.items()} == mod.LAYOUT2
