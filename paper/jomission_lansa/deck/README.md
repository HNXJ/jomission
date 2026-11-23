# jomission spectrolaminar deck

Card title: **Testing cell-type density as a circuit mechanism of primate spectrolaminar oscillations**

Card description: A column sized from macaque cell densities places alpha-beta deep on its reference seeds, though not density-specifically, and does not generate gamma near layer 4; changing density or the SOM/PV ramp hands the map to one unit; a step-by-step readout search shows the readout could carry the observed gamma profile, so under that readout the gap is in the circuit's spectra; readout depth remains a confound.

Status: preliminary, unpublished results (manuscript draft 2026-11-10). Hamm approved posting to the BastosLab site conditional on review (2026-11-10); posting is jarness-keeper's step after its independent review. Untracked and uncommitted until then.

Files: `slides.pdf` (25 pages, 16:9), `pages/page-01.png`..`page-25.png` (150 dpi), `thumb.png` (= page 1).
Sources: `main.tex` (Beamer), `main.md` (storyboard with the Table S6 record behind each page), `make_figs.py`
(redraws figures from the records; run from the repo root; asserts each rebuilt profile and map statistic against
the stored values). Record IDs appear only on the final Sources page. Scope (Hamm, 2026-11-10): the motif and
maximal similarity only.

Build: `python3 paper/jomission_lansa/deck/make_figs.py`, then `latexmk -pdf main.tex` in this folder, then
`pdftoppm -r 150 -png main.pdf pages/page`; `slides.pdf` is a copy of `main.pdf`.
