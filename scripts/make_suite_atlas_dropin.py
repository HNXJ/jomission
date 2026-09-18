"""Regenerate the standalone drop-in from jomission/visualization/jaxfne_suite_atlas.py.

The drop-in is the same module with theme.py inlined, so it has no jomission dependency and can
be handed to the jaxfne team as one file. Edit the package copy; run this to republish.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "jomission" / "visualization" / "jaxfne_suite_atlas.py"
DST = ROOT / "docs" / "jaxfne_suite_atlas_dropin" / "jaxfne_suite_atlas.py"

THEME_IMPORT = '''from jomission.visualization.theme import (
    AREA_COLORS,
    CLASS_COLORS,
    PROJ_COLORS,
    apply_dark_theme,
    wrap_figure_with_provenance_html,
)

SUITE = "jaxfne_suite_atlas.v1"'''


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    cur = DST.read_text(encoding="utf-8")
    # The drop-in's own header (docstring + inlined theme) is everything before the marker.
    marker = "# --------------------------------------------------------------------------------------\n# introspection"
    head = cur.split(marker)[0]
    body = marker + src.split(marker, 1)[1]
    assert THEME_IMPORT not in body, "drop-in must not import jomission"
    DST.write_text(head + body, encoding="utf-8", newline="\n")
    print(f"regenerated {DST.relative_to(ROOT)} ({len((head + body).splitlines())} lines)")


if __name__ == "__main__":
    main()
