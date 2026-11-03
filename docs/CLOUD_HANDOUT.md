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
- Scorecard: `checklist-score.md` (v2). Open stack: `docs/OPEN.md`.

## Known open items
- Sealed v21b/v21c reads use cp1252 (checklist item 25); the drive-guard allowlist entry is owner-approved.
- `jomission/simulation/exposure.py`: second simulate in the multi-trial loop has no continuation.
- Lite model: canonical builder on V1 and V4 at n_per_area=100 (200 neurons), owner-defined; builds and simulates finitely in tests/test_lite_model.py.

## Not in git (not available in the cloud image)
- `local/` inputs (LANSA, bibliography, Campagnola PDF). Analyses that need them cannot run here.
