# local/

Owner-supplied inputs the repo needs but does not hold in git. Everything here except this
file is ignored by git (`.gitignore`). Each file is listed in `manifests/local_inputs.json`
with its sha256 and its original location. Check with:

    python scripts/check_local_inputs.py

A missing or changed file is a finding. Restore it from its original location (the manifest
names it), then re-run the check. Do not commit anything in this folder.
