"""MIN-08 realization: VIP effective drive is 25.0; E, PV and SST keep tonic + 1.0; nothing else changes.

Spec: docs/minimal/s13_vip_drive_spec.json. The patch sets params.drive at the VIP rows only; u0, v0, a, b,
c, d and the labels are unchanged. Written test-first: it must fail against the build before
realize_vip_drive exists.
"""

import numpy as np

from jomission.minimal.builder import build_minimal_model
from jomission.minimal.s13_vip_drive import DC_CURRENT, VIP_DRIVE, realize_vip_drive


def test_vip_drive_25_other_drives_tonic_plus_dc_and_rest_unchanged():
    base = build_minimal_model().params["emitter"]
    labels = np.asarray(base.labels)
    vip = labels == "VIP"
    assert vip.sum() == 5, "fixture must have five VIP cells at N = 100"
    tonic = np.asarray(base.drive, dtype=float)

    model = realize_vip_drive(build_minimal_model(), vip_drive=VIP_DRIVE, dc_current=DC_CURRENT)
    patched = model.params["emitter"]

    assert np.allclose(np.asarray(patched.drive)[vip], VIP_DRIVE)
    assert np.allclose(np.asarray(patched.drive)[~vip], tonic[~vip] + DC_CURRENT)
    for name in ("u0", "v0", "a", "b", "c", "d"):
        assert np.array_equal(np.asarray(getattr(patched, name)), np.asarray(getattr(base, name))), name
    assert tuple(patched.labels) == tuple(base.labels)
