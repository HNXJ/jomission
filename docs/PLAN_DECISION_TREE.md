# Decision tree: live TODO items, end to end

Date: 2026-10-31. Source: `manifests/todo.json` (live items only). Nothing here is authorized;
`next_authorized_task` stays null until the owner names a node. Each node runs only after its
parent node's receipt is committed. A FAIL or STOP ends the branch and goes back to the owner.

## Node 0: owner go-ahead

- Input: owner names the first node to run. Nothing runs before this.
- Owner decisions queued for this tree: CB/CR marker mapping and PV→VIP confirmation (node 1),
  habituation schedule (node 3), LANSA cell densities (node 4), reviewer authorization (nodes 5, 6).

## Node 1: MIN-02, connectivity citations

- Goal: every rule and weight in `docs/minimal/connectivity.json` traces to a cited passage.
- Action: owner decides the CB/CR mapping and confirms PV→VIP (bound) and VIP→VIP (absent).
  Reviewer checks each citation against the text.
- Branch:
  - All rules cited → receipts for S-07 to S-11 → node 2.
  - A rule has no source → list it as open; the rule stays open; node 2 runs with that rule frozen out.

## Node 2: MIN-06, S-13 lineage (DC drive 1.0)

- Goal: E, PV, SST and VIP each fire in the 1000 ms run.
- Entry: node 1 done; owner authorizes MIN-06.
- Action: (a) read the JaxFNE units of DC 1.0 from the convention, not from assumption; (b) commit
  `docs/minimal/s13_spec.json` before any run; (c) log-spaced grid for any search parameter, stated
  in the spec; (d) one 1000 ms smoke per grid point; (e) commit `s13_receipt.json` separately.
- Branch:
  - Some grid point has all four classes firing → score S-13 PASS → node 3.
  - A class silent at every grid point → STOP and report. No retune after results. Owner decides
    the next lineage.

## Node 3: MIN-04, checklist A (CK-A, 10 items)

- Entry: owner gives the habituation schedule (stimulus sequence, trial order, seed, ≥ 1000 s
  exposure). Node 2 done (the drive it uses is fixed there).
- Action: spec first; naive and habituated arms on one seed set; perturb each new parameter at 0,
  boundary and non-default; omission firewall kept.
- Branch:
  - Each item receipt PASS → CK-A scored → node 4.
  - Any item FAIL → STOP at that item and report. Later items wait.

## Node 4: MIN-05, checklist B (CK-B, 10 items)

- Entry: owner gives the LANSA cell densities, or confirms the local `.mat` is the input. The
  statistic is fixed by the LANSA pipeline: PCA cumulative variance ≥ 90% (2 PCs), against a
  surrogate chance floor.
- Action: spec with the statistic and null first; pin LANSA by blob; build 11 areas × 21 bins;
  check realized fractions; reference-check the ported PCA against MATLAB on observed data; held-out
  areas listed in the spec. H is never tuned to match the pattern.
- Branch:
  - Reference PCA check fails → STOP. Fix the port, then re-check. No result is taken before it passes.
  - All items PASS → CK-B scored → node 5.

## Node 5: SCI-AGSDR-CALIBRATION

- Entry: SCI-EI-INTERACTION-1 verdict exists, and the reviewer authorizes the calibration design.
  Nodes 2 to 4 are not required for this node, but its plant must be the EI-defined one.
- Action: write and seal the calibration design; execute; seal the result.
- Branch:
  - Verdict viable → node 6 may be authorized.
  - Verdict not viable → STOP. Node 6 stays LOCKED.

## Node 6: SCI-PROP-QUALIFIED-1

- Entry: node 5 result accepted and the reviewer authorizes the design. Locked until then.
- Action: test ordered trajectory propagation in the AGSDR-defined viable regime.
- Branch: result sealed as recorded, PASS or FAIL. No further node is listed.

## Parallel tracks (no node depends on them)

- Long-term plans P1 to P10 in `docs/PLAN_LONGTERM.md` run in lanes beside this tree. They use the
  same gates. P1 overlaps nodes 2 and 3; P2 is node 3; P3 is node 4.
