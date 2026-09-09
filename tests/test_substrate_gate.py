"""Universal substrate gate: zero-tonic self-sustained activity.

SKIPPED until a candidate recurrent substrate arrives (see
results/substrate_gate.json). Unskip — do not delete — to qualify any
future architecture. The gate must pass BEFORE any T_R question.
"""

import pytest


@pytest.mark.skip(reason="substrate gate: current plant has no zero-tonic "
                         "regime (see results/substrate_gate.json); unskip "
                         "for candidate substrates")
def test_zero_tonic_self_sustained_regime():
    raise NotImplementedError("gate body to be written against candidate")
