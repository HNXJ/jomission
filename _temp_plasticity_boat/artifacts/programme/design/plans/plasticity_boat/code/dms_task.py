"""Memory-guided spatial DMS task backend (memory_dms_spec_v1.md; D3 dispositions).

Deterministic, float64, numpy only. No controller, no neural state, no learning.

Plant (same as jaxfne.paradigm.sailor_step with a calm disturbance):
    v' = v + dt * (-beta * v + F)
    p' = p + dt * (v + v') / 2

Trial phases on the step grid (step index k, durations in whole steps):
    CUE   [s, s+c)            target visible as s = q - p(s), cue_valid = 1
    DELAY [s+c, s+c+d)        target hidden
    GO    [t2, t3)            motor enabled; trial closes at the first step k in [t2, t3]
                              with |p_k - q| <= radius (arrival) or at k = t3 (deadline)
    END   step t_END          result recorded; t_END = min(arrival, deadline)

Motor output is gated to GO. The plant keeps integrating in every phase.

Provisional, not calibrated (D3.4): radius, timings, max_force.
Feasibility (D3.2): analytic bound at GO onset is necessary; oracle success is demonstrated
reachability only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

IDLE, CUE, DELAY, GO, END = range(5)
PHASE_NAMES = ("IDLE", "CUE", "DELAY", "GO", "END")

# observation layout: one-hot phase (5), GO flag, cue_valid, s (2), b = (p, v) (4)
IDX_GO = 5
IDX_CUE_VALID = 6
IDX_S = slice(7, 9)
IDX_B = slice(9, 13)
N_OBS = 13

# privileged test oracle gains: critically damped at beta = 0.5 (beta + KD = 2 sqrt(KP))
KP, KD = 1.0, 1.5


@dataclass(frozen=True)
class Event:
    onset_s: float
    x: float
    y: float


@dataclass(frozen=True)
class TaskConfig:
    events: tuple = ()
    dt: float = 0.02
    damping: float = 0.5
    cue_s: float = 0.5
    delay_s: float = 2.0
    go_s: float = 5.0
    radius: float = 0.5
    max_force: float | None = None  # None: feasibility undefined; primary evaluation needs a value
    proprioception: bool = True


@dataclass(frozen=True)
class Trial:
    index: int
    onset: int
    t1: int
    t2: int
    t3: int
    q: tuple


def _steps(seconds: float, dt: float, name: str) -> int:
    n = seconds / dt
    k = round(n)
    if abs(n - k) > 1e-9:
        raise ValueError(f"{name}={seconds} s is not a whole number of steps at dt={dt}")
    return int(k)


def schedule(cfg: TaskConfig) -> list[Trial]:
    """Validate and convert events to step indices. Rejects; never reschedules."""
    if not cfg.dt > 0:
        raise ValueError("dt must be > 0")
    if not cfg.damping >= 0:
        raise ValueError("damping must be >= 0")
    if not cfg.radius > 0:
        raise ValueError("radius must be > 0")
    if cfg.max_force is not None and not cfg.max_force > 0:
        raise ValueError("max_force must be > 0 when set")
    cue = _steps(cfg.cue_s, cfg.dt, "cue_s")
    delay = _steps(cfg.delay_s, cfg.dt, "delay_s")
    go = _steps(cfg.go_s, cfg.dt, "go_s")
    if cue < 1 or delay < 0 or go < 1:
        raise ValueError("cue_s and go_s must be >= one step; delay_s must be >= 0")
    trials: list[Trial] = []
    prev_t3 = -1
    onsets = [ev.onset_s for ev in cfg.events]
    if onsets != sorted(onsets):
        raise ValueError("events must be given in nondecreasing onset order")
    for i, ev in enumerate(cfg.events):
        s = _steps(ev.onset_s, cfg.dt, "onset_s")
        if s < 0 or s <= prev_t3:
            raise ValueError(f"event {i} at {ev.onset_s} s overlaps the previous trial or precedes t=0")
        t1 = s + cue
        t2 = t1 + delay
        t3 = t2 + go
        trials.append(Trial(i, s, t1, t2, t3, (float(ev.x), float(ev.y))))
        prev_t3 = t3
    return trials


class DMSEnv:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self.trials = schedule(cfg)
        self.dt = float(cfg.dt)
        self.beta = float(cfg.damping)
        self.k = 0
        self.p = np.zeros(2)
        self.v = np.zeros(2)
        self.results: list[dict] = []
        self._active: int | None = None
        self._cue = np.zeros(2)
        self._min = math.inf
        self._feas = None  # (feasible, bound) once computed for the active trial
        self._end_at: int | None = None
        self._phase = IDLE
        self._rec: list[tuple] = []
        self._update_phase()

    # --- phase machinery -------------------------------------------------

    def _open(self, i: int) -> None:
        self._active = i
        self._cue = np.asarray(self.trials[i].q, dtype=float) - self.p
        self._min = math.inf
        self._feas = None

    def _feasibility(self, tr: Trial):
        fm = self.cfg.max_force
        if fm is None or self.beta <= 0:
            return None, None
        go_s = (tr.t3 - tr.t2) * self.dt
        bound = go_s * max(float(np.linalg.norm(self.v)), fm / self.beta)
        dist = float(np.linalg.norm(np.asarray(tr.q, dtype=float) - self.p))
        return dist <= bound, bound

    def _ensure_feas(self) -> None:
        if self._feas is None:
            self._feas = self._feasibility(self.trials[self._active])

    def _update_phase(self) -> None:
        k = self.k
        if self._active is None and self._end_at != k:
            for i, tr in enumerate(self.trials):
                if tr.onset == k:
                    self._open(i)
                    break
        if self._active is None:
            self._phase = END if self._end_at == k else IDLE
            return
        tr = self.trials[self._active]
        if k < tr.t1:
            self._phase = CUE
        elif k < tr.t2:
            self._phase = DELAY
        else:
            self._phase = GO
            self._ensure_feas()

    def _check_close(self) -> None:
        if self._active is None:
            return
        tr = self.trials[self._active]
        if self.k < tr.t2:
            return
        self._ensure_feas()
        d = float(np.linalg.norm(self.p - np.asarray(tr.q, dtype=float)))
        self._min = min(self._min, d)
        if d <= self.cfg.radius:
            self._close(True, self.k)
        elif self.k >= tr.t3:
            self._close(False, self.k)

    def _close(self, success: bool, kc: int) -> None:
        i = self._active
        tr = self.trials[i]
        feas, bound = self._feas
        d_end = float(np.linalg.norm(self.p - np.asarray(tr.q, dtype=float)))
        self.results.append({
            "index": i, "onset": tr.onset, "t2": tr.t2, "t3": tr.t3, "t_end": kc,
            "q": tr.q, "success": success,
            "latency_s": (kc - tr.t2) * self.dt if success else None,
            "terminal_error": d_end, "min_dist": self._min,
            "feasible": feas, "bound": bound,
        })
        self._active = None
        self._end_at = kc

    # --- public interface ------------------------------------------------

    def observe(self) -> np.ndarray:
        ph = self._phase
        m = np.zeros(5)
        m[ph] = 1.0
        o = np.zeros(N_OBS)
        o[:5] = m
        o[IDX_GO] = 1.0 if ph == GO else 0.0
        o[IDX_CUE_VALID] = 1.0 if ph == CUE else 0.0
        if ph == CUE:
            o[IDX_S] = self._cue
        if self.cfg.proprioception:
            o[IDX_B] = np.concatenate([self.p, self.v])
        return o

    def step(self, action) -> np.ndarray:
        obs = self.observe()
        a = np.asarray(action, dtype=float).reshape(2)
        gate = 1.0 if self._phase == GO else 0.0
        F = gate * a
        fm = self.cfg.max_force
        if fm is not None:
            F = np.clip(F, -fm, fm)
        self._rec.append((self.k, self._phase, self.p.copy(), self.v.copy(), F.copy(), obs))
        v_new = self.v + self.dt * (-self.beta * self.v + F)
        p_new = self.p + 0.5 * self.dt * (self.v + v_new)
        self.v, self.p = v_new, p_new
        self.k += 1
        self._check_close()
        self._update_phase()
        return self.observe()

    def log_arrays(self) -> dict:
        k, ph, p, v, F, obs = zip(*self._rec)
        return {
            "k": np.array(k), "phase": np.array(ph), "p": np.array(p),
            "v": np.array(v), "F": np.array(F), "obs": np.array(obs),
        }


def run(env: DMSEnv, n_steps: int, policy) -> DMSEnv:
    for _ in range(n_steps):
        env.step(policy(env))
    return env


def oracle_action(env: DMSEnv) -> np.ndarray:
    """PRIVILEGED test oracle: reads the hidden target. Reference for T06 and T07 only."""
    if env._active is None:
        return np.zeros(2)
    q = np.asarray(env.trials[env._active].q, dtype=float)
    a = KP * (q - env.p) - KD * env.v
    fm = env.cfg.max_force
    if fm is not None:
        a = np.clip(a, -fm, fm)
    return a
