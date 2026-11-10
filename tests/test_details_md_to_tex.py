"""scripts/details_md_to_tex.py: every markdown table row reaches LaTeX, escaped, values unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import details_md_to_tex as d  # noqa: E402

MD = """# Details: x_T8

prose that is dropped

## Modules

| module | depth | N_E |
|---|---|---|
| S | −0.4 | 5,785 |
| D | +0.3 | 2,547 |

## Parameters

| group | parameter | value | unit | source |
|---|---|---|---|---|
| synapse | Q E->E | 2 | nS | PORT_SPEC (Tahvili) 50% & `x` |
"""


def test_tables_and_rows():
    t = d.tables(MD)
    assert [s for s, _ in t] == ["Modules", "Parameters"]
    assert [len(r) for _, r in t] == [3, 2]  # header + rows; separator lines dropped


def test_escape_and_values_kept():
    tex = d.to_tex("x_T8", MD)
    assert "S & $-$0.4 & 5,785 \\\\" in tex
    assert r"Q E$\to$E" in tex and r"50\% \& x" in tex and r"PORT\_SPEC" in tex
    assert "prose that is dropped" not in tex
    assert tex.count(r"\begin{longtable}") == 1 and tex.count(r"\begin{tabular}") == 1
