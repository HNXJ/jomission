"""Tests for T1-T7 and ablation matrix."""

from jomission.analysis.targets import FALSIFICATION_TARGETS, DELTA_EXPOSURE_DEFINITION, validate_targets
from jomission.ablations.matrix import ABLATION_MATRIX, validate_ablations


def test_targets_frozen():
    v = validate_targets()
    assert v["valid"]
    assert len(FALSIFICATION_TARGETS) == 7
    assert [t.id for t in FALSIFICATION_TARGETS] == ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    assert "Delta_exposure" in DELTA_EXPOSURE_DEFINITION


def test_ablation_matrix():
    v = validate_ablations()
    assert v["valid"]
    assert len(ABLATION_MATRIX) == 11
    names = [a.name for a in ABLATION_MATRIX]
    assert "full" in names and "no-H" in names and "FB-off" in names
