"""Run a sealed SCI-EI driver with the sealed native-noise law pinned (SCI-EI-REPLAY-058, amendment 2).

Usage (from the repo root): python scripts/ei_replay_058_run.py <driver.py> [driver arguments...]

JaxFNE 0.5.6 and later scale the native noise coefficient by sqrt(DT_REF_MS / dt_ms) (emitters.py,
_noise_coef). The sealed runs used 0.5 at every dt; at dt 0.1 ms the default realizes 1.1180. This
wrapper passes noise_scale = NOISE_BASE to each compile_step_fn call, so the realized coefficient is
0.5 at dt 0.1 ms, the sealed value. The driver file runs unchanged (runpy); only the compile call is
wrapped. It refuses any other dt, kernel or a driver-supplied noise_scale, so the pin cannot apply
silently to a run it was not calibrated for. Exit code is the driver's.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import jaxfne
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DT_REF_MS = 0.5
SEALED_DT_MS = 0.1
SEALED_COEF = 0.5
# float32 value whose realization base * sqrt(DT_REF_MS / dt) equals SEALED_COEF exactly at dt 0.1
# (checked in float32 with the engine's operation order; the drivers run float32, no x64).
NOISE_BASE = 0.2236068

_compile_step_fn = jaxfne.compile_step_fn


def _realized(dt_ms):
    ref, dt = np.float32(DT_REF_MS), np.float32(dt_ms)
    return np.float32(NOISE_BASE) * np.sqrt(ref / dt)


def _pinned_compile_step_fn(model, *args, **kwargs):
    dt_ms = kwargs.get("dt_ms")
    if dt_ms is None or abs(float(dt_ms) - SEALED_DT_MS) > 1e-12:
        raise ValueError(f"noise pin is calibrated at dt {SEALED_DT_MS} ms, got dt_ms={dt_ms!r}")
    if kwargs.get("kernel") != "baseline":
        raise ValueError(f"noise pin covers the baseline kernel, got kernel={kwargs.get('kernel')!r}")
    if "noise_scale" in kwargs:
        raise ValueError("driver passes noise_scale; the pin would overwrite it")
    realized = _realized(dt_ms)
    if realized != np.float32(SEALED_COEF):
        raise ValueError(f"pinned coefficient realizes {realized!r}, not the sealed {SEALED_COEF}")
    kwargs["noise_scale"] = NOISE_BASE
    print(
        f"noise pin: noise_scale={NOISE_BASE} at dt {dt_ms} ms realizes {float(realized)} "
        f"(sealed {SEALED_COEF}; default would realize "
        f"{float(np.float32(0.5) * np.sqrt(np.float32(DT_REF_MS) / np.float32(dt_ms)))})",
        flush=True,
    )
    return _compile_step_fn(model, *args, **kwargs)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    driver = (ROOT / sys.argv[1]).resolve()
    if not driver.is_file():
        print(f"no driver at {driver}", file=sys.stderr)
        return 2
    jaxfne.compile_step_fn = _pinned_compile_step_fn
    sys.argv = [str(driver), *sys.argv[2:]]
    runpy.run_path(str(driver), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
