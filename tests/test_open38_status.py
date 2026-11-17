"""OPEN-38 status source on synthetic receipts: a launched or unfinished run is never PASS; STOP is BLOCKED; a stale
log is UNKNOWN; PASS needs DONE plus every named checker receipt, all from one source revision (or a verified
compatibility entry); the public view shows a status only from committed receipts."""
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "st", Path(__file__).resolve().parents[1] / "scripts" / "open38_status.py")
st = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(st)

T = "2026-11-17T00:00:00Z"
EXE = "a" * 40


def _log(repo: Path, name: str, lines: list[str], age_min: float = 0.0):
    p = repo / st.R / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{T} {x}\n" for x in lines), encoding="utf-8")
    t = time.time() - age_min * 60
    os.utime(p, (t, t))


def _rec(repo: Path, name: str, ok: bool, **kw):
    (repo / st.R).mkdir(parents=True, exist_ok=True)
    (repo / st.R / name).write_text(json.dumps({"pass": ok, **kw}), encoding="utf-8")


def _git(repo: Path, *a) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True,
                          stdin=subprocess.DEVNULL).stdout.strip()


def _repo(tmp: Path) -> Path:
    _git(tmp, "init", "-q")
    _git(tmp, "config", "user.email", "t@t")
    _git(tmp, "config", "user.name", "t")
    (tmp / "jomission").mkdir()
    (tmp / "jomission" / "model.py").write_text("x = 1\n")
    (tmp / "scripts").mkdir()
    (tmp / "scripts" / "run_e2_stack.py").write_text("run = 1\n")
    _git(tmp, "add", "-A")
    _git(tmp, "commit", "-qm", "exe")
    return tmp


def _commit_all(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


SMOKE_HEAD = ["mode smoke", f"HEAD {EXE}", "construction invariance: PASS", "start o38s_v1_n4000_s1_T8 L exec=x"]


def _stage1(repo: Path, exe: str, integ_head: str | None = None, run_sha: str | None = None):
    _log(repo, "o38s_pc_stage.log", [f"HEAD {exe}", "start o38s_pc_ref_n4000_s1_T8 L", "o38s_pc_ref_n4000_s1_T8 rc=0",
                                      "stage check: PASS", "DONE"])
    _rec(repo, "o38s_pc_stage1_check.json", True, runs={"ref": {"1": {"git_sha": run_sha or exe}}})
    _rec(repo, "o38s_pc_integrity.json", True, head=integ_head or exe)
    _cons(repo)


N_RULES = 260  # as stage 1: ruling-B intervals are wider than the frozen 99.9 % ones


def _cons(repo: Path, counts=None, widen=1.0, fake_ok=None, n_tests=N_RULES, sizes_ok=True):
    """Frozen receipt (99.9 % intervals, half-width 100 around 1000) and its ruling-B re-check, built as the checker does."""
    from statistics import NormalDist
    counts = counts or {}
    nd = NormalDist()
    scale = nd.inv_cdf(1 - 0.05 / N_RULES / 2) / nd.inv_cdf(1 - 0.001 / 2)

    def rec(half, fam, forge=None):
        rules = {}
        for i in range(N_RULES):
            c = counts.get(i, 1000)
            lo, hi = 1000 - half, 1000 + half
            ok = lo <= c <= hi if forge is None or i not in counts else forge
            rules[f"r{i}"] = {"count": c, "interval": [lo, hi], "params_equal": True, "ok": ok}
        inh_ok = all(v["ok"] for v in rules.values())
        out = {"arms": ["L"], "seeds": [1], "pass": inh_ok,
               "results": {"L|s1": {"sizes": {"ok": True}, "inh_rules": {"ok": inh_ok, "rules": rules}}}}
        return {**out, **({"family": fam} if fam else {})}

    _write(repo, "o38s_pc_construction.json", rec(100.0, None))
    b = rec(100.0 * scale * widen, {"alpha": 0.05, "n_tests": n_tests, "per_test_level": 1 - 0.05 / n_tests}, fake_ok)
    b["results"]["L|s1"]["sizes"]["ok"] = sizes_ok
    _write(repo, "o38s_pc_construction_rulingB.json", b)


def _write(repo: Path, name: str, obj):
    (repo / st.R).mkdir(parents=True, exist_ok=True)
    (repo / st.R / name).write_text(json.dumps(obj), encoding="utf-8")


def _compat(repo: Path, exe: str, src: str, paths=("jomission", "scripts/run_e2_stack.py"), equiv=True, rule=True,
            n=N_RULES):
    ent = {"receipt": f"{st.R}/o38s_pc_construction_rulingB.json", "execution_sha": exe, "source_sha": src,
           "identical_paths": list(paths), "expected_n_tests": n, "why": "test"}
    if equiv:
        ent.update(equivalent_to=f"{st.R}/o38s_pc_construction.json", ignore_keys=["interval", "ok", "pass", "family"])
    if rule:
        ent["rule"] = "ruling_b_construction"
    _write(repo, "o38s_receipt_compat.json", {"schema": "o38s_compat/1", "entries": [ent]})


def test_launched_run_is_running_never_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD)
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "RUNNING" and s["runs_exit0"] == 0 and s["public_status"] == "UNKNOWN"


def test_stale_running_log_is_unknown(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD, age_min=45)
    assert st.stage2(tmp_path, 30)["status"] == "UNKNOWN"


def test_stop_is_blocked_and_attempts_listed(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", ["mode smoke", f"HEAD {EXE}", "construction invariance: FAIL",
                                         "STOP: construction invariance failed; no run started"])
    _log(tmp_path, "o38s_s2_attempt1_stage.log", ["mode smoke", "HEAD old", "construction invariance: FAIL", "STOP: x"])
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "BLOCKED" and s["phases"][0]["status"] == "BLOCKED" and "STOP" in s["reason"]
    assert s["attempts"] == [{"attempt": 1, "mode": "smoke", "head": "old", "construction": "FAIL", "stop": "STOP: x"}]


def test_smoke_pass_without_rest_is_blocked_not_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=0", "smoke integrity: PASS", "DONE smoke"])
    s = st.stage2(tmp_path, 30)
    assert s["phases"][0]["status"] == "PASS" and s["status"] == "BLOCKED" and s["runs_exit0"] == 1


REST_TAGS = st.S2_TAGS[1:]  # the smoke ran S2_TAGS[0] (v1, seed 1)
REST = SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=0", "smoke integrity: PASS", "DONE smoke", "mode rest", f"HEAD {EXE}"] + [
    x for t in REST_TAGS for x in (f"start {t} L exec={t}", f"{t} rc=0")] + ["stage-2 integrity: PASS", "DONE rest"]


def _s2(repo: Path, exe: str = EXE, integ: bool = True, run_sha: str | None = None, construction: dict | None = None,
        integ_head: str | None = None):
    """Full stage-2 evidence as the driver leaves it: construction (ruling B, 6 arms x 5 seeds x 26 rules), run
    record, execution manifest and verdict per tag, smoke and stage integrity receipts."""
    rules = {f"r{i}": {"count": 1000, "interval": [900, 1100], "params_equal": True, "ok": True}
             for i in range(st.S2_RULES_PER_PAIR)}
    arms = [f"A{a}" for a in st.S2_ARMS]
    n = st.S2_N_TESTS
    cons = construction or {
        "arms": arms, "seeds": list(st.S2_SEEDS), "pass": True,
        "family": {"alpha": 0.05, "n_tests": n, "per_test_level": 1 - 0.05 / n},
        "results": {f"{a}|s{s}": {"sizes": {"ok": True}, "inh_rules": {"ok": True, "rules": json.loads(json.dumps(rules))}}
                    for a in arms for s in st.S2_SEEDS}}
    _write(repo, "o38s_s2_construction.json", cons)
    for t in st.S2_TAGS:
        _write(repo, f"{t}.json", {"git_sha": run_sha or exe, "git_dirty": False, "seed": int(t.split("_s")[-1][0]),
                                   "tag_probe": t})
        _write(repo, f"{t}_verdict.json", {"r_ab": -0.7, "tag_probe": t})
        _write(repo, f"{t}_similarity.json", {"tag_probe": t})
        _write(repo, f"{t}_motif.json", {"tag_probe": t})
        outs = {s: hashlib.sha256((repo / st.R / f"{t}{s}").read_bytes()).hexdigest() for s in st.S2_HASHED}
        _write(repo, f"{t}_exec.json", {"tag": t, "rc": 0, "exec_id": t, "head": exe, "outputs": outs})
    _rec(repo, "o38s_s2_smoke_integrity.json", True, head=exe)
    _rec(repo, "o38s_s2_integrity.json", integ, head=integ_head or exe, checks={"runs": {"ok": True}, "propagation": {"ok": True}},
         runs=[{"tag": t, "ok": True, "exec_id": t, "npz_sha256": None} for t in st.S2_TAGS])


def test_rest_pass_needs_integrity_receipt(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", REST)
    assert st.stage2(tmp_path, 30)["status"] == "UNKNOWN"
    _s2(tmp_path)
    assert st.stage2(tmp_path, 30)["status"] == "PASS"
    _rec(tmp_path, "o38s_s2_integrity.json", False, head=EXE)
    assert st.stage2(tmp_path, 30)["status"] != "PASS"


def test_stage2_receipts_from_another_revision_are_not_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", REST)
    _s2(tmp_path, integ_head="b" * 40)  # integrity re-run at another commit
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "UNKNOWN" and "revisions differ" in s["reason"]
    _s2(tmp_path, run_sha="c" * 40)  # one source for every run record, but not the logged HEAD
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "UNKNOWN" and "revisions differ" in s["reason"]


def test_stage2_pass_is_rederived_from_its_receipts(tmp_path):  # review 2 (2026-11-17)
    _log(tmp_path, "o38s_s2_stage.log", REST)
    _s2(tmp_path)
    assert st.stage2(tmp_path, 30)["status"] == "PASS"
    R = tmp_path / st.R
    t0 = st.S2_TAGS[3]

    def edit(name, fn):
        p = R / name
        x = json.loads(p.read_text())
        fn(x)
        p.write_text(json.dumps(x))

    cases = {
        "construction pass false": lambda: edit("o38s_s2_construction.json", lambda c: c.update({"pass": False})),
        "construction family forged": lambda: edit("o38s_s2_construction.json", lambda c: c.update(
            family={"alpha": 0.5, "n_tests": 1, "per_test_level": 0.5})),
        "construction rule dropped": lambda: edit("o38s_s2_construction.json", lambda c: next(iter(
            c["results"].values()))["inh_rules"]["rules"].pop("r0")),
        "construction ok forged": lambda: edit("o38s_s2_construction.json", lambda c: next(iter(
            c["results"].values()))["inh_rules"]["rules"]["r0"].update(count=5000)),
        "construction deleted": lambda: (R / "o38s_s2_construction.json").unlink(),
        "run record dirty": lambda: edit(f"{t0}.json", lambda r: r.update(git_dirty=True)),
        "one run on another SHA": lambda: edit(f"{t0}.json", lambda r: r.update(git_sha="d" * 40)),
        "manifest rc 1": lambda: edit(f"{t0}_exec.json", lambda r: r.update(rc=1)),
        "manifest deleted": lambda: (R / f"{t0}_exec.json").unlink(),
        "verdict deleted": lambda: (R / f"{t0}_verdict.json").unlink(),
        "integrity check not ok": lambda: edit("o38s_s2_integrity.json", lambda r: r["checks"]["runs"].update(ok=False)),
        "integrity run missing": lambda: edit("o38s_s2_integrity.json", lambda r: r["runs"].pop()),
        # review 3 (2026-11-17)
        "intervals widened, ok recomputed": lambda: edit("o38s_s2_construction.json", lambda c: [
            x.update(interval=[-1e9, 1e9], count=5000) for r in c["results"].values()
            for x in r["inh_rules"]["rules"].values()]),
        "one seed's interval moved to its count": lambda: edit("o38s_s2_construction.json", lambda c: next(iter(
            c["results"].values()))["inh_rules"]["rules"]["r0"].update(count=1200, interval=[1100, 1300])),
        "exec head of another revision": lambda: edit(f"{t0}_exec.json", lambda r: r.update(head="e" * 40)),
        "exec output hash altered": lambda: edit(f"{t0}_exec.json", lambda r: r["outputs"].update({".json": "0" * 64})),
        "run record replaced by another run's": lambda: (R / f"{t0}.json").write_bytes(
            (R / f"{st.S2_TAGS[4]}.json").read_bytes()),
        "verdict content altered": lambda: edit(f"{t0}_verdict.json", lambda r: r.update(r_ab=0.9)),
        # review 4 (2026-11-17)
        "manifest exec_id altered": lambda: edit(f"{t0}_exec.json", lambda r: r.update(exec_id="forged")),
        "integrity npz hash altered": lambda: edit("o38s_s2_integrity.json", lambda r: r["runs"][3].update(
            npz_sha256="1" * 64)),
    }
    for name, fn in cases.items():
        _s2(tmp_path)
        fn()
        s = st.stage2(tmp_path, 30)
        assert s["status"] in ("UNKNOWN", "FAIL"), f"{name}: {s['status']} {s['reason']}"
    _s2(tmp_path)
    _log(tmp_path, "o38s_s2_stage.log", [x for x in REST if t0 not in x])  # one run's start and rc removed
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "UNKNOWN" and "29 rc=0 of 30" in s["reason"], s["reason"]


def test_nonzero_exit_fails_and_missing_exit_is_unknown(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=1", "smoke integrity: FAIL", "DONE smoke"])
    assert st.stage2(tmp_path, 30)["status"] == "FAIL"
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["smoke integrity: PASS", "DONE smoke"])
    assert st.stage2(tmp_path, 30)["phases"][0]["status"] == "UNKNOWN"


def test_stage1_pass_needs_every_receipt(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe)
    (repo / st.R / "o38s_pc_construction_rulingB.json").unlink()
    assert st.stage1(repo, 30)["status"] == "UNKNOWN"
    _cons(repo)
    _compat(repo, exe, exe)
    assert st.stage1(repo, 30)["status"] == "PASS"
    _rec(repo, "o38s_pc_integrity.json", False, head=exe)
    assert st.stage1(repo, 30)["status"] == "FAIL"


def test_stage1_mixed_revisions_are_not_pass(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe, integ_head="c" * 40)
    _compat(repo, exe, exe)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "revisions differ" in s["reason"]
    _stage1(repo, exe, run_sha="d" * 40)
    assert st.stage1(repo, 30)["status"] == "UNKNOWN"


def test_foreign_receipt_needs_verified_compat_entry(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    (repo / "scripts" / "check.py").write_text("level = 2\n")  # the re-check's own code changed
    src_same = _commit_all(repo, "checker only")
    _stage1(repo, exe)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "no compatibility entry" in s["reason"]
    _compat(repo, exe, src_same)
    assert st.stage1(repo, 30)["status"] == "PASS"
    (repo / "jomission" / "model.py").write_text("x = 2\n")  # model code changed after execution
    src_diff = _commit_all(repo, "model change")
    _compat(repo, exe, src_diff)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "declared paths differ" in s["reason"]


def test_public_status_needs_committed_receipts(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe)
    _compat(repo, exe, exe)
    s = st.stage1(repo, 30)
    assert s["status"] == "PASS" and s["public_status"] == "UNKNOWN"  # receipts untracked
    _commit_all(repo, "receipts")
    assert st.stage1(repo, 30)["public_status"] == "PASS"
    _rec(repo, "o38s_pc_integrity.json", True, head=exe, note="edited")  # modified after commit
    assert st.stage1(repo, 30)["public_status"] == "UNKNOWN"


def _s1(repo, exe, **cons):
    _stage1(repo, exe)
    _cons(repo, **cons)
    return st.stage1(repo, 30)


def test_compat_needs_experimental_equivalence_not_only_identical_code(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _compat(repo, exe, exe)
    assert _s1(repo, exe)["status"] == "PASS"
    _stage1(repo, exe)
    b = json.loads((repo / st.R / "o38s_pc_construction_rulingB.json").read_text())
    b["results"]["L|s1"]["inh_rules"]["rules"]["r3"]["count"] = 1001  # one realized edge more than the frozen build
    _write(repo, "o38s_pc_construction_rulingB.json", b)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "differs from" in s["reason"], "a different realized count is not equivalent"
    _cons(repo)
    _compat(repo, exe, exe, equiv=False)
    assert "no readable equivalent_to" in st.stage1(repo, 30)["reason"]
    _compat(repo, exe, exe, rule=False)
    assert "without a re-derivation rule" in st.stage1(repo, 30)["reason"]


def test_ignored_fields_are_rederived_not_trusted(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _compat(repo, exe, exe)
    out = {0: 1130}  # outside the ruling-B interval (half-width 100 x z_B/z_0.999 ~ 125)
    s = _s1(repo, exe, counts=out)
    assert s["status"] == "FAIL", "a count outside ruling B fails honestly"
    # the frozen receipt fails this count honestly, so stage 1 is FAIL whatever ruling B says; the compat check
    # must still reject the forged ruling-B receipt on its own
    s = _s1(repo, exe, counts=out, widen=1.2, fake_ok=True)  # intervals widened beyond ruling B and ok forged
    assert s["status"] != "PASS" and "width" in str(s["consistency"]["problems"])
    s = _s1(repo, exe, counts=out, fake_ok=True)  # ruling-B widths, but ok forged for the outside count
    assert s["status"] != "PASS" and "recomputed" in str(s["consistency"]["problems"])
    s = _s1(repo, exe, n_tests=26)  # family claims fewer tests than the receipt holds
    assert s["status"] == "UNKNOWN" and "family" in s["reason"]
    s = _s1(repo, exe, sizes_ok=False)  # a level-independent gate's ok changed
    assert s["status"] != "PASS"


def _edit_b(repo, fn):
    p = repo / st.R / "o38s_pc_construction_rulingB.json"
    b = json.loads(p.read_text())
    fn(b)
    _write(repo, p.name, b)
    return st.stage1(repo, 30)


def test_comparison_fails_closed_on_unexpected_or_missing_fields(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _compat(repo, exe, exe)
    assert _s1(repo, exe)["status"] == "PASS"
    rules = lambda b: b["results"]["L|s1"]["inh_rules"]["rules"]  # noqa: E731
    cases = {
        "extra field": lambda b: rules(b)["r0"].update(seed=9),
        "missing field": lambda b: rules(b)["r0"].pop("params_equal"),
        "extra pair": lambda b: b["results"].update({"L|s2": b["results"]["L|s1"]}),
        "ignored key in an undeclared slot": lambda b: b["results"]["L|s1"]["sizes"].update(interval=[0, 1]),
        "re-derived field missing": lambda b: rules(b)["r0"].pop("interval"),
        "family missing": lambda b: b.pop("family"),
        "pass missing": lambda b: b.pop("pass"),
    }
    for name, fn in cases.items():
        _stage1(repo, exe)
        s = _edit_b(repo, fn)
        assert s["status"] == "UNKNOWN", f"{name}: {s['status']} {s.get('reason')}"


def _edit_both(repo, fn):
    for f in ("o38s_pc_construction.json", "o38s_pc_construction_rulingB.json"):
        p = repo / st.R / f
        x = json.loads(p.read_text())
        fn(x)
        _write(repo, f, x)
    return st.stage1(repo, 30)


def test_consistent_forgery_in_both_receipts_is_not_pass(tmp_path):  # review 2026-11-17 gaps (b), (c)
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _compat(repo, exe, exe)
    assert _s1(repo, exe)["status"] == "PASS"
    rules = lambda b: b["results"]["L|s1"]["inh_rules"]["rules"]  # noqa: E731
    _stage1(repo, exe)  # count outside the frozen interval, inside ruling B, frozen ok left true
    s = _edit_both(repo, lambda b: rules(b)["r0"].update(count=1105))
    assert s["status"] == "UNKNOWN" and "frozen" in s["reason"], s["reason"]
    _stage1(repo, exe)  # one rule dropped in both receipts
    s = _edit_both(repo, lambda b: rules(b).pop("r0"))
    assert s["status"] == "UNKNOWN" and "expected_n_tests" in s["reason"], s["reason"]
    _stage1(repo, exe)  # declared seeds name a pair both receipts lack
    s = _edit_both(repo, lambda b: b["seeds"].append(2))
    assert s["status"] == "UNKNOWN" and "arms x seeds" in s["reason"], s["reason"]


def test_identical_paths_must_cover_model_code_and_exist(tmp_path):  # review 2026-11-17 gap (a)
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe)
    _compat(repo, exe, exe, paths=("jomission",))  # exists, but omits the stack runner
    assert "omit" in st.stage1(repo, 30)["reason"]
    _compat(repo, exe, exe, paths=("jomission", "scripts/run_e2_stack.py", "nonexistent_dir"))
    assert "absent" in st.stage1(repo, 30)["reason"]


def test_committing_evidence_keeps_the_execution_identity(tmp_path):  # owner 2026-11-17 closure invariant
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    log = [ln.replace(EXE, exe) for ln in REST]
    _log(repo, "o38s_s2_stage.log", log)
    _s2(repo, exe)
    before = st.stage2(repo, 30)
    assert before["status"] == "PASS" and before["public_status"] == "UNKNOWN", "uncommitted evidence is not public"
    ev = _commit_all(repo, "stage-2 evidence")
    assert ev != exe
    after = st.stage2(repo, 30)
    assert after["public_status"] == "PASS"
    assert after["consistency"]["execution_sha"] == exe, "the evidence commit must not become the execution identity"
    assert st.git(repo)["sha"] == ev
    _s2(repo, exe, integ_head=ev)  # integrity re-stamped with the evidence commit
    _commit_all(repo, "restamp")
    s = st.stage2(repo, 30)
    assert s["status"] == "UNKNOWN" and "revisions differ" in s["reason"]


def test_stage2_pass_needs_finished_smoke(tmp_path):  # review 2026-11-17 MAJOR
    unfinished = ["mode smoke", f"HEAD {EXE}", "construction invariance: PASS", "start o38s_v1_n4000_s1_T8 L exec=x",
                  "mode rest", f"HEAD {EXE}", "start o38s_mt_n4000_s1_T8 L exec=y", "o38s_mt_n4000_s1_T8 rc=0",
                  "stage-2 integrity: PASS", "DONE rest"]
    _log(tmp_path, "o38s_s2_stage.log", unfinished)
    _s2(tmp_path)
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "UNKNOWN" and s["phases"][0]["status"] == "UNKNOWN", s["reason"]
    assert REST[len(SMOKE_HEAD) + 3] == "mode rest"
    _log(tmp_path, "o38s_s2_stage.log", REST[len(SMOKE_HEAD) + 3:])  # a rest segment with no smoke before it
    assert st.stage2(tmp_path, 30)["status"] != "PASS"
