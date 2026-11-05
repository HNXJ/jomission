# Cloud handout (jomission, jaxfne 0.5.8)

Status: written 2026-11-03. Reproduced 2026-11-05 on a fresh clone with a fresh venv at 07332b4 (steps 1 and 2 as written; tier 0 216 passed; `project_check.py --offline --no-tests` PROJECT_CHECK_PASS; Python 3.13). Same container image; a fresh cloud image is not tested.

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
- Scorecard: `checklist-score.md` (v4). Open stack: `docs/OPEN.md`.

## Known open items
- Sealed v21b/v21c reads use cp1252 (OPEN-04); the drive-guard allowlist entry is owner-approved.
- `jomission/simulation/exposure.py`: second simulate in the multi-trial loop has no continuation.
- Lite model: canonical builder on V1 and V4 at n_per_area=100 (200 neurons), owner-defined; builds and simulates finitely in tests/test_lite_model.py.

## Not in git (not available in the cloud image)
- `local/` inputs (LANSA, bibliography, Campagnola PDF). Analyses that need them cannot run here.
