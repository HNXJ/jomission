"""V2.1 plant provenance: the executed plant carries JaxFNE internal noise.

Reproduces the finding recorded in results/v21_provenance_correction.json with
the frozen V2.1 build_plant (loaded by path, unchanged): at V_HIGH over 5000
steps, engine seeds 11 and 12 diverge under the default noise_scale, and the
divergence vanishes at noise_scale=0.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_v21_plant_carries_internal_noise():
    import jax
    import jax.numpy as jnp
    import jaxfne as jtfne

    spec = importlib.util.spec_from_file_location("_v21_frozen", ROOT / "tests" / "test_v21_operation.py")
    v21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v21)
    corr = json.load(open(ROOT / "results" / "v21_provenance_correction.json"))
    vec = json.load(open(ROOT / "results" / "v21_vectors.json"))["vectors"]["V_HIGH"]
    model, step_fn, cls = v21.build_plant(vec)
    sched = jnp.zeros((5000, len(cls)), dtype=model.params["emitter"].v0.dtype)

    def run(seed, fn):
        st = jtfne.ContinuationState(dynamic=jtfne.dynamic_state_from_model(model),
                                     prng_key=jax.random.PRNGKey(seed), step_index=0, delay_state=None)
        _, out = jtfne.run_continuation(fn, st, sched)
        return np.asarray(out[0], float), np.asarray(out[1], float)

    va, sa = run(v21.SEED, step_fn)
    vb, sb = run(v21.SEED + 1, step_fn)
    step0, _ = jtfne.compile_step_fn(model, dt_ms=v21.DT_MS, kernel="baseline", noise_scale=0.0)
    za, _ = run(v21.SEED, step0)
    zb, _ = run(v21.SEED + 1, step0)

    exp = corr["observed"]
    assert np.abs(va - vb).max() > exp["min_seed_divergence_mV"]
    assert np.abs(sa - sb).sum() > exp["min_differing_spike_events"]
    assert np.abs(za - zb).max() == 0.0
