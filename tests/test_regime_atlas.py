"""Regime atlases render sealed V2.1 records only (no simulation)."""

import json
import pathlib
import shutil

import pytest

from jomission.visualization import regime_atlas as RA

REGIMES = ("low", "mid", "high")


def test_recomputed_gates_match_sealed_checks():
    loaded = RA.load_regimes("results", "v21", REGIMES)
    for rec in loaded["records"]:
        d = rec["data"]
        recomputed = {g: info["pass"] for g, info in RA.recheck_gates(d).items()}
        assert recomputed == d["checks"], rec["id"]
        assert all(recomputed.values()) == d["pass"], rec["id"]


def test_recomputed_drift_matches_sealed():
    for rec in RA.load_regimes("results", "v21", REGIMES)["records"]:
        derived = RA.recompute_drift(rec["data"])
        for c in RA.CLASSES:
            assert derived[c] == pytest.approx(rec["data"]["drift"][c], abs=1e-12), (rec["id"], c)


def test_build_never_simulates(tmp_path, monkeypatch):
    import jaxfne

    def _forbidden(*a, **k):
        raise AssertionError("regime atlas must not simulate")

    for name in ("simulate", "run_continuation", "compile_step_fn", "construct"):
        if hasattr(jaxfne, name):
            monkeypatch.setattr(jaxfne, name, _forbidden)
    manifest = RA.build_regime_atlases(str(tmp_path), "results", "v21", REGIMES)

    assert manifest["simulation"] == "none"
    assert [a["regime"] for a in manifest["atlases"]] == list(REGIMES)
    assert (tmp_path / "index.html").exists()
    for a in manifest["atlases"]:
        adir = tmp_path / a["id"]
        assert (adir / "index.html").exists()
        assert [p["file"] for p in a["panels"]] == [f for f, _ in RA.PANELS]
        for f, _ in RA.PANELS:
            assert (adir / f).stat().st_size > 0
        src = pathlib.Path(a["source"])
        assert a["source_sha256"] == RA._sha256(str(src))
        assert a["sealed_pass"] == json.loads(src.read_text())["pass"]
        assert a["sealed_pass"] == a["recomputed_pass"]
        assert "MISMATCH" not in (adir / "gates.html").read_text(encoding="utf-8")
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["lineage"]["verdict"] == json.load(open("results/v21_lineage.json"))["verdict"]


def test_build_is_byte_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    RA.build_regime_atlases(str(a), "results", "v21", REGIMES)
    RA.build_regime_atlases(str(b), "results", "v21", REGIMES)
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b and len(files_a) == 2 + len(REGIMES) * (1 + len(RA.PANELS))
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_missing_sealed_record_fails_loudly(tmp_path):
    for name in ("v21_low.json", "v21_lineage.json"):
        shutil.copy(pathlib.Path("results") / name, tmp_path / name)
    with pytest.raises(FileNotFoundError, match="v21_mid.json"):
        RA.load_regimes(str(tmp_path), "v21", REGIMES)


def test_missing_key_fails_loudly(tmp_path):
    shutil.copy("results/v21_lineage.json", tmp_path / "v21_lineage.json")
    rec = json.load(open("results/v21_low.json"))
    del rec["currents"]
    (tmp_path / "v21_low.json").write_text(json.dumps(rec))
    with pytest.raises(KeyError, match="currents"):
        RA.load_regimes(str(tmp_path), "v21", ("low",))
