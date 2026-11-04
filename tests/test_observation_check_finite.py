"""Receipt observation checks must reject non-finite values (a NaN comparison is false, so abs(NaN - e) > tol passes)."""

import pytest

from jomission.harness import validate as v

REC = {"lineage_id": "L", "commit": "HEAD"}


def _obs(expected=1, tol=0):
    return {"check": {"path": "results/x.json", "json_pointer": "/value", "expected": expected, "tol": tol}}


def _with_text(monkeypatch, text):
    monkeypatch.setattr(v, "read_artifact", lambda ref: text)


def test_finite_matching_value_passes(monkeypatch):
    _with_text(monkeypatch, '{"value": 1}')
    assert v._observation_check(REC, _obs()) == []


def test_finite_contradiction_is_reported(monkeypatch):
    _with_text(monkeypatch, '{"value": 2}')
    assert v._observation_check(REC, _obs()) != []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_rejected(monkeypatch, token):
    _with_text(monkeypatch, '{"value": %s}' % token)
    assert v._observation_check(REC, _obs()) != []


def test_non_finite_expected_is_rejected(monkeypatch):
    _with_text(monkeypatch, '{"value": 1}')
    assert v._observation_check(REC, _obs(expected=float("nan"))) != []


def test_non_finite_tolerance_is_rejected(monkeypatch):
    _with_text(monkeypatch, '{"value": 1}')
    assert v._observation_check(REC, _obs(tol=float("inf"))) != []
