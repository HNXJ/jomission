# Jomission context audit and near-term plan — 2026-11-13

Scope: documentation-only navigation, not a new scientific authorization or a substitute for the manifest TODO. Source: live GitHub `dev@0366d2baa4f4ec475c781387d2b59d0e9c9a91c3`; no local runtime or scientific tests executed.

## Authority and current evidence

- `AGENTS.md` directs execution through `python scripts/project_check.py`, `manifests/current_state.json`, `manifests/todo.json`, and the smallest authorized task. `manifests/sealed_artifacts.json` protects immutable evidence. Do not modify sealed artifacts.
- GitHub `main@b0b7fb84` and `dev@0366d2ba` differ at inspection; latest inspected dev CI run completed SUCCESS. The successful baseline does not verify this documentation change.
- `manifests/current_state.json` has `updated: 2026-11-05`, `jaxfne_version: 0.5.8`, and current gate `V2_LOCAL_OPERATION: FAIL`. Treat as recorded authority with its declared evidence, not proof of live installation. Other narrative references to 0.4.17/0.5.9 describe historical milestones or separate scientific lineages, not a reason to change engine identity.
- `manifests/todo.json` currently has 49 DONE, 9 RETIRED, 2 OPEN, 1 LOCKED (parsed from the inspected file). The two OPEN items are `SCI-AGSDR-CALIBRATION` and `SCI-LAMINAR-MOTIF-1`. A status of OPEN does not itself grant permission to run a blocked stage.
- Open scientific failures, inconclusive outcomes and unresolved variables remain evidence, not targets to retune. Simulation results are not empirical omission evidence.

## Short term — follow the authorized manifest, not this plan

1. **Reconcile provenance:** run the project check in the intended environment, confirm live JaxFNE checkout/version and current `next_authorized_task`; report any mismatch between recorded `0.5.8`, later lineage references and actual environment. No silent engine upgrade.
2. **AGSDR calibration:** resolve the owner/reviewer prerequisites in `SCI-AGSDR-CALIBRATION`, including shortfall and objective amendments, feasible-regime status, and reviewer authorization. Do not design or execute calibration while blocked. Closing check is the item's own sealed design, authorized run and receipts.
3. **Laminar motif:** continue only the already authorized portions of `SCI-LAMINAR-MOTIF-1`; obtain owner values for UNSET thresholds and required reviewer approvals before scoring the 25-item acceptance. Preserve E1 INCONCLUSIVE, E2/T8 outcomes and all negative controls.
4. **Scientific gate:** retain `V2_LOCAL_OPERATION: FAIL` and related quarantined or non-identifying results until a new predeclared lineage and independent verdict establish otherwise. Do not alter the omission firewall.
5. **Presentation and publication:** figures, paper and Pages are downstream of sealed, qualified model evidence; manuscript completeness or rendered figures alone do not qualify scientific results.

## Midterm dependencies

- **Substrate:** establish an identifiable, stable-enough laminar E/PV/SST/VIP model with explicit density, synaptic and field-proxy semantics; validate against predeclared nulls, shuffles and perturbations.
- **Hierarchy and continuity:** only after substrate gates, construct the V1/V4/FEF/PFC hierarchy and uninterrupted exposure-to-omission trajectory using frozen JaxFNE execution authority; verify no state reset and exact omission timing.
- **Omission evaluation:** run frozen T1–T7 only after substrate construction is independently qualified and the omission firewall permits it. Preserve null/negative findings; never calibrate the model to omission outcomes and then claim held-out validation.
- **Cross-project:** JaxFNE is the simulation engine, jnwb is a generic analysis library, and omission is an empirical study. The simulation may generate falsifiable predictions but does not validate empirical claims or JaxFNE's own physics.

## Audit findings / action boundaries

- The public README's first-milestone checklist and engine references are historical; the live execution authority is the manifests. This note is linked from README without rewriting scientific timing, thresholds or checked boxes.
- Project status is manifest-based; no `artifacts/state.md`, `artifacts/open_stack.md` or `artifacts/todo_stack.md` exists on dev. Do not manufacture parallel stacks.
- Several temporary branches exist; no open PRs at inspection. Reconcile ownership before branch cleanup.
- Documentation/manifest changes map to tier 0 under `manifests/test_tiers.json`. Owner's no-test request does not by itself establish a passing tier-0 gate. This pass records **NOT RUN**.

Closing evidence: exact SHA, project-check verdict, tier-0 receipt, unchanged sealed-artifact hashes, independent reviewer and owner authorization for any scientific scope change. None of those checks was executed by this GitHub-only audit.
