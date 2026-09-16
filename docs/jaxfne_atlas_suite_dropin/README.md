# jaxfne drop-in: atlas_suite

Verbatim-ready files for the jaxfne repo:

1. `atlas_suite.py` → `jaxfne/jaxfne/vis/atlas_suite.py`
2. Add to `jaxfne/jaxfne/vis/__init__.py`:
   `from .atlas_suite import build_atlas, PANELS` (+ `__all__` entries).
3. `atlas_suite_docs.md` → `jaxfne/docs/guides/atlas_suite.md`
4. `mkdocs_nav_snippet.yml` → merge the one line under `Guides:` in `mkdocs.yml`.

Source of truth while unmerged: `jomission/jomission/visualization/atlas_suite.py`
(handout: `jomission/docs/ATLAS_SUITE_HANDOUT.md`). Verified N=1 + small net
via `jomission/tests/test_atlas_suite.py`.
