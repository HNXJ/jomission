"""E1b scorer pieces (docs/laminar_motif/E1B_SEAL.md): multimodal flag, null gate."""

import importlib.util
from pathlib import Path

import numpy as np
from jaxfne.analysis import aperiodic_spectral_peak

_spec = importlib.util.spec_from_file_location("e1b_score", Path(__file__).resolve().parents[1] / "scripts/e1b_score.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def _spectrum(bumps):
    f = np.arange(0.0, 501.0)
    p = 1.0 / (1.0 + f ** 1.5)
    for fc, gain in bumps:
        p = p * (1.0 + gain * np.exp(-0.5 * ((f - fc) / 1.5) ** 2))
    return f, p


def _flag(bumps):
    f, p = _spectrum(bumps)
    pk = aperiodic_spectral_peak(f, p, band_lo_hz=10.0, band_hi_hz=80.0)
    return pk, sc.multimodal(f, p, pk, (10.0, 80.0))


def test_multimodal_known_positive():
    pk, m = _flag([(22.0, 20.0), (50.0, 12.0)])
    assert pk["peak_freq_Hz"] == 22.0, "fixture: primary at 22 Hz"
    assert m["multimodal"] and m["second_peak_Hz"] == 50.0


def test_multimodal_known_negative_single_peak():
    pk, m = _flag([(22.0, 20.0)])
    assert pk["peak_excess"] >= 4, "fixture must have a primary peak"
    assert not m["multimodal"]


def test_multimodal_weak_second_peak_not_flagged():
    _, m = _flag([(22.0, 20.0), (50.0, 1.0)])
    assert not m["multimodal"]


def test_null_gate_does_not_pass():
    g = sc.gate4_null()
    assert g["pass"] and g["p_value"] >= 0.05
