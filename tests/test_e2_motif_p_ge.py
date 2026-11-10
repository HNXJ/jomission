"""scripts/e2_motif_figure.p_ge: the identity permutation, equal to M up to float rounding, counts toward p."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_motif_figure as mf  # noqa: E402


def test_identity_within_rounding_counts():
    M = 0.21
    null = [M - 1.4e-17 * 10, M + 0.1, M - 0.1, M - 0.05]  # identity 1.4e-16 below M (rounding), one real exceedance
    assert null[0] < M, "fixture must put the identity below M in float"
    assert mf.p_ge(null, M) == 0.5


def test_real_shortfall_does_not_count():
    assert mf.p_ge([0.2, 0.1], 0.21) == 0.0
