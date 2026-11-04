"""B10 audit matching (amendment 1): whole-word only."""

from jomission.qualification.gen2_gates import audit_b10_objective


def test_jomission_is_not_a_hit():
    assert audit_b10_objective("from jomission.network import builder")["clean"] is True


def test_whole_word_omission_is_a_hit():
    assert "omission" in audit_b10_objective("score the omission trials")["hits"]


def test_whole_word_T1_is_a_hit():
    assert "T1" in audit_b10_objective("uses T1 here")["hits"]


def test_T10_is_not_a_hit():
    assert "T1" not in audit_b10_objective("bin T10 is fine")["hits"]
