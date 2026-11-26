# Receipt: project_check in the jaxfne 0.5.8 gate venv (2026-11-25)

Owner word: Hamm, 2026-11-25, "proceed with your recommendations" (recommended option A': a separate venv, leaving the
editable 0.5.9 in `/home/hamm/.venvs/jaxfne` untouched).

## Build

| Step | Command | Result |
|---|---|---|
| source | `git -C /mnt/e/repos/jaxfne archive v0.5.8` into session scratch | `pyproject.toml` version 0.5.8; tag commit 11957afa4a25749b42ed52bcb16779fc624c9c56 |
| venv | `python3 -m venv /home/hamm/.venvs/jomission058` | python 3.14 |
| jaxfne | `pip install <archive of v0.5.8>` (not editable) | jaxfne 0.5.8 |
| jomission | `pip install -e /mnt/d/jomission` | jomission 0.1.0, editable, `/mnt/d/jomission/jomission/__init__.py` |
| resolved deps | `pip list` | jax 0.10.2, jaxlib 0.10.2 (CPU build), numpy 2.5.5, scipy 1.18.1 |

The jaxfne repository and `/mnt/d/jomission` were not modified (`git status --porcelain` empty after install).

## Gate

`cd /mnt/d/jomission && /home/hamm/.venvs/jomission058/bin/python scripts/project_check.py --offline --no-tests`
at `_temp_integ_20261124` @ ef0a766, exit 0:

```
[ok] workspace: git toplevel /mnt/d/jomission
      branch _temp_integ_20261124 @ ef0a766
[ok] tree clean: 0 modified tracked files
[ok] scientific head ancestry: main be0db96 in HEAD history
[ok] jaxfne identity: installed 0.5.8, execution authority 0.5.8
[ok] editable install: jomission -> /mnt/d/jomission/jomission/__init__.py
[ok] todo
[ok] current_state
[ok] sealed artifacts
[ok] lineage registry
[ok] skills
[ok] sealed-write guard: denies docs/designs/H_ACTIVE_TEST_spec.md; installed by tests/conftest.py
      next authorized task: SCI-AGSDR-CALIBRATION
      current gate: V2_LOCAL_OPERATION = FAIL
PROJECT_CHECK_PASS
```

## Limits

- Quick contract only. The full pass (network remote comparison and tier-0 tests) was not run.
- `V2_LOCAL_OPERATION` = FAIL is a project state, not an environment failure; unchanged.
- The venv has CPU jaxlib; it is a gate environment, not a GPU execution environment.
