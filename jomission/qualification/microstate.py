"""N5 causal microstate decomposition: matched interventions at the onset.

Parent: N1-N4 (NO_INTERMEDIATE). Candidates: s50 primary (amp 3.75),
s57 replication (amp 4.0). Snapshot times t* = 5 ms (primary) and 8 ms
(replicate robustness; NOT a deterministic tautology: different
microstate, same trajectory). No plant changes except the transient,
diagnostic interventions below (plant restored after each fork).

Source run: fresh 8 s uniform-E prep (N1 paradigm, driven throughout);
snapshot full dynamic carry at t*; control continues the identical
schedule (snapshot-fidelity check: must reproduce the sealed outcome).

Arms (7: control + 6), all from the same snapshot where valid:
  CTRL    continue unchanged schedule
  PERMUTE within-class joint (V,u,prev_spikes) relabeling + within-pathway
          syn_state permutation (seed 0); marginals/energy/spike-counts
          exactly preserved (asserted); aux/w/H untouched (asserted).
          Faithful by construction -> synchrony/microstate-phase test.
  UCLAMP  a=d=0 variant model for 200 ms, then restore (u frozen exactly;
          drift asserted ~=0). Pre-state unchanged by construction.
  SYNHALF one-shot syn_state x0.5; MEASURED washout expected (volley
          spikes re-saturate syn in ~15 steps): recorded, void as an
          ownership test, never a flipper. Lesson pre-registered: if
          washed out, synaptic state is spike-slaved at volley gain.
  KW0     k_w=0 rule from snapshot (freezes ONGOING HDP evolution;
          accumulated aux/w intact -- the demanded distinction).
  BLIPP   uniform E +2 units x 50 ms (amplitude control, not state).
  BLIPM   uniform E -2 units x 50 ms (amplitude control).
Outcome: driven endpoint SILENT (<1 Hz) / SYNC (>=4000 sustained) / OTHER;
latency t_sustain recorded (delay-without-flip = modulatory, not ownership).

Ownership (predeclared): arm flips control fate at BOTH t* (5, 8 ms) ->
owns (pending s57 replication for acceptance). >=2 flippers -> one
conjunction fork (pair) + latency ordering. None flip -> NONE_IDENTIFIED.
Session validity: control must go SYNC sustained, else VOID (max 2
sessions per candidate, else NONE_CONCLUSIVE for that candidate).
Acceptance classes: SPIKE_RESET_MICROSTATE | ADAPTATION | SYNAPTIC_STATE
| HDP_EVOLUTION | E_I_BALANCE | INTERACTION | NONE_IDENTIFIED.
"""

from __future__ import annotations

import numpy as np

PRIMARY = (50.0, 3.75)
REPLICATE = (57.0, 4.0)
SNAPS_MS = (5.0, 8.0)

CLAMP_MS = 200.0
SYN_SCALE = 0.5
BLIP_AMP = 2.0
BLIP_MS = 50.0
PERM_SEED = 0

DT_MS = 0.1
SEED = 11
RUN_S = 8.0

ARMS = ("CTRL", "PERMUTE", "UCLAMP", "SYNHALF", "KW0", "BLIPP", "BLIPM")


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")
