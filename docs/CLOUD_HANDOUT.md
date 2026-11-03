# Cloud handout (jomission, jaxfne 0.5.8)

Status: written from the cloud container session on 2026-11-03. Setup steps were run
in this container only, not on a fresh cloud image.

## Setup
1. `git clone https://github.com/HNXJ/jomission && cd jomission && git checkout cowork/focused-ptolemy-dz7hwc`
2. `python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`
   This installs the pinned engine `jaxfne==0.5.8` and `plotly` (dev extra).

## Checks
- Quick contract: `python scripts/project_check.py --offline --no-tests`
- Full check (network + tier 0): `python scripts/project_check.py`
- Tier 1 and tier 2 lists: `manifests/test_tiers.json`. Tier 3 needs an authorizing lineage and is not run.

## State
- Authority: `manifests/current_state.json` (jaxfne 0.5.8, record `results/jaxfne_058_migration.json`).
- Scorecard: `checklist-score.md`.

## Known open items
- `tests/test_v21b_operation.py` Unicode read: needs an owner decision (see checklist item 20).
- `jomission/simulation/exposure.py`: second simulate in the multi-trial loop has no continuation.
- The lite model is not defined.

## Not in git (not available in the cloud image)
- `local/` inputs (LANSA, bibliography, Campagnola PDF). Analyses that need them cannot run here.
