"""Tier 0: a registered sealed artifact cannot be overwritten by default.

On 2026-09-17 a bare full-suite pytest ran the tier-3 batteries and overwrote 55 sealed result
files in one command. manifests/sealed_artifacts.json already said "rerun only under an authorized
lineage"; nothing enforced it. These checks enforce it, and check that the guard denies before the
file is touched rather than after.
"""

import hashlib
import json
from pathlib import Path

import pytest

from jomission.harness import seals

ROOT = Path(__file__).resolve().parents[1]


def _a_sealed_path():
    for rel in sorted(seals.sealed_paths()):
        if (ROOT / rel).exists():
            return rel
    pytest.skip("no sealed artifact present in the working tree")


def _digest(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def test_guard_is_installed_for_this_session():
    assert seals.active()


def test_overwrite_of_a_sealed_artifact_fails_before_mutation():
    rel = _a_sealed_path()
    before = _digest(rel)
    with pytest.raises(seals.SealedArtifactWriteError):
        open(ROOT / rel, "w", encoding="utf-8").write("clobbered")
    assert _digest(rel) == before


def test_append_and_truncating_update_are_also_denied():
    rel = _a_sealed_path()
    before = _digest(rel)
    for mode in ("a", "r+", "wb"):
        with pytest.raises(seals.SealedArtifactWriteError):
            open(ROOT / rel, mode)
    assert _digest(rel) == before


def test_path_write_text_is_covered():
    rel = _a_sealed_path()
    before = _digest(rel)
    with pytest.raises(seals.SealedArtifactWriteError):
        (ROOT / rel).write_text("clobbered", encoding="utf-8")
    assert _digest(rel) == before


def test_reading_a_sealed_artifact_is_unaffected():
    """The registry seals code as well as results, so this only checks that reads still work."""
    rel = _a_sealed_path()
    assert (ROOT / rel).read_text(encoding="utf-8").strip()
    assert (ROOT / rel).read_bytes()


def test_a_new_or_unsealed_path_stays_writable(tmp_path):
    """An unsealed results path is not denied, and writing outside the repo is untouched.

    The denial is checked without writing into results/, because a tier-0 test that writes a
    result would itself violate the tier rule this suite exists to protect.
    """
    unsealed = ROOT / "results" / "_guard_probe_unsealed.json"
    assert "results/_guard_probe_unsealed.json" not in seals.sealed_paths()
    seals.check_write(unsealed)          # does not raise
    assert not unsealed.exists()
    scratch = tmp_path / "x.json"
    scratch.write_text("{}", encoding="utf-8")
    assert scratch.read_text(encoding="utf-8") == "{}"


@pytest.fixture
def fake_seal(tmp_path, monkeypatch):
    """A registry of our own under tmp_path, so authorization is exercised without touching evidence."""
    a, b = tmp_path / "results" / "one.json", tmp_path / "results" / "two.json"
    a.parent.mkdir(parents=True)
    a.write_text('{"a": 1}', encoding="utf-8")
    b.write_text('{"b": 2}', encoding="utf-8")
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"artifacts": [{"path": "results/one.json"}, {"path": "results/two.json"}]}),
                   encoding="utf-8")
    monkeypatch.setattr(seals, "ROOT", tmp_path)
    monkeypatch.setattr(seals, "REGISTRY", reg)
    return a, b


def test_the_fake_registry_is_denied_like_the_real_one(fake_seal):
    a, _ = fake_seal
    with pytest.raises(seals.SealedArtifactWriteError):
        a.write_text("clobbered", encoding="utf-8")
    assert a.read_text(encoding="utf-8") == '{"a": 1}'


def test_an_authorized_lineage_may_rewrite_the_path_it_names(fake_seal, monkeypatch):
    a, _ = fake_seal
    monkeypatch.setenv(seals.ENV, "results/one.json")
    a.write_text("authorized rewrite", encoding="utf-8")
    assert a.read_text(encoding="utf-8") == "authorized rewrite"


def test_authorization_is_scoped_to_the_named_path(fake_seal, monkeypatch):
    a, b = fake_seal
    monkeypatch.setenv(seals.ENV, "results/one.json")
    with pytest.raises(seals.SealedArtifactWriteError):
        b.write_text("clobbered", encoding="utf-8")
    assert b.read_text(encoding="utf-8") == '{"b": 2}'
    a.write_text("ok", encoding="utf-8")


def test_the_guard_does_not_mutate_the_registry_or_the_seals():
    reg = ROOT / "manifests" / "sealed_artifacts.json"
    before = hashlib.sha256(reg.read_bytes()).hexdigest()
    seals.sealed_paths()
    seals.check_write(ROOT / "results" / "_not_sealed_probe.json")
    assert hashlib.sha256(reg.read_bytes()).hexdigest() == before
