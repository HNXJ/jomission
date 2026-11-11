# Session log: jomission-LANSA Phase 0 and A3 audit, 2026-11-11

Actor: Cursor agent (DeepSeek V4.1 Flash), continuing from the Claude handoff in
`HANDOUT_CURSOR_LANSA_100.md`. The paper was read before the round-3 verdict landed; the A3 figure work of §6
came after and is scoped separately in the handout's state section.
Every line below is OBSERVED with its command, unless tagged INFERRED.

## 0. Check-in and gate receipts

- Entry state: `dev` @ `e36a566`, 4 commits ahead of `origin/dev` (`f4c7970`), 0 modified tracked files.
- `python3 scripts/project_check.py --offline --no-tests` in the live tree: `PROJECT_CHECK_PASS`.
- Tier-0 at HEAD `f60973c` in `git clone --shared` (`/tmp/jomission_tier0_f60973c`):
  `1 failed, 257 passed, 5 warnings in 24.20s`; the one failure is
  `tests/test_environment_provenance.py::test_jomission_import_resolves_to_this_checkout`, which fails in a
  clone by construction (the editable install points at the live tree). Same count as the `2c51ab3` receipt in
  the handout, so no regression.
- Manuscript build at HEAD: 0 undefined refs/cites, 0 overfull, 0 "Float too large", 35 pages
  (`paper/jomission_lansa/main.log`).

## 1. Handout state section updated (commit `f60973c`)

Named the manuscript SHA the round-3 review binds (`e36a566`), its timestamp, and the build receipt; removed the
self-referential "the commit that adds this state line" phrasing, which could never stay true.

## 2. Independent number checks at `e36a566` (verification of paper claims)

| Claim | Where | Source checked | Result |
|---|---|---|---|
| Abstract <= 250 words | `abstract.tex` | word count after stripping LaTeX | **248** — holds |
| Fig 4 plots 93 runs | `results.tex` caption | `results/viz/paper_w7/band_scatter.json` group sizes 9+54+3+12+3+6+6 | **93** — holds |
| Fig 4 "90 distinct simulations" | `results.tex` caption | 3 T8-reference tags carry `(r_ab, r_gm)` identical to 12 dp with the ladder n4000 runs | **90** — holds |
| D6 manifest complete | `data/sources_sha256.txt` | re-hashed every one of 158 entries from disk | **158 ok, 0 missing, 0 mismatch** |
| All figures resolve, none stubbed | 23 `\includegraphics` | build log lists all 23, `pending` count 0 | holds |
| Discussion cut 1459 -> 941 words | `response_review4` | word counts at `12fc805`..`e36a566`: 925, 932, 947; `f4c7970` 1456 | **947, not 941** (minor, response file) |
| Response "done" rows hold in the text | `response_review4` | 24 literal checks across 7 sections | 24/24 hold after resolving 3 grep artefacts |

Notes: the 941-vs-947 gap is in the response file only; the manuscript text states no Discussion word count, so
nothing in the paper is wrong. Tag `T` for cleanup: correct the number in `response_review4` or say "about 950".

## 3. Figures: tracking and print-size audit (Phase A3 evidence)

- 13 unique `\includegraphics` targets; 11 tracked at HEAD, **2 untracked**
  (`results/viz/paper_w3/..._block.png`, `results/viz/t1/V3_sstd_n4000_atlas.png`).
  Methods discloses both and D6 hashes both, so this is recorded, not a silent gap.
- Print size = `\textwidth` 489.10 pt = 6.7677 in. Font floors: 6 pt text, 5.5 pt ticks, 9 pt panel letters.

| Figure | Script | Canvas | Scale | Under floor at print |
|---|---|---|---|---|
| 1 schema | `e2_circuit_schema.py` | 7.2 in | 0.940 | none (legend 6.11, ticks 6.58) |
| 2 reference block | `e2_paper_block.py` | 7.2 in | 0.940 | SOM in-cell label **5.17 pt**, panel letters **8.46 pt** |
| 3 scorer view | `e2_concept_panels.py` | 6.8 in | 0.995 | panel D monospace **5.97 pt**, panel letters **7.96 pt** |
| 4 band scatter | `e2_band_scatter.py` | 4.6 in | 1.000 | none |

Fig 3 panel D monospace at 5.97 pt is the round-2 "5-6 pt" finding, now measured. Single cheapest fix: draw every
canvas at 6.77 in wide (scale 1.000) and raise Fig 2's SOM label to 6 pt and the panel letters to 9 pt.
`data/sources_sha256.txt` is hand-maintained (no generator script exists in the repo), so every regenerated PNG
needs its hash updated in the same commit.

## 4. Declaration placeholders (Phase A1, blocked on the owner)

`grep -c MISSING` at HEAD: `main.tex` 1, `sections/backmatter.tex` 8, every other section 0. The nine items:
authors/affiliations/ORCID/corresponding author; author contributions (CRediT); funding and grant numbers;
competing interests; acknowledgements; AI-use statement; data availability (LANSA acquisition + archive/DOI);
code availability (archive/DOI). All are owner-only facts and none is generated.

## 5. Queue (top first) and blockers

| # | Item | Closing check | Blocked by |
|---|---|---|---|
| 1 | Round-3 critic verdict (running, GPT-5.6, sees neither earlier round) | `VERDICT: ACCEPT` on `e36a566` | — |
| 2 | On ACCEPT: push `dev`, read CI for the SHA | `git log origin/dev..dev` empty; CI success | 1 |
| 3 | A3 figures: font floors fixed and measured (done, `5a89597`, `f20fe5c`, `c6941d9`, `e6b33ac`, `458fb2c`); needs the visual reviewer's verdict | visual `VERDICT: ACCEPT` | — |
| 4 | A1 declarations (9 items) | `grep -c MISSING` = 0 | Hamm |
| 5 | A2 archive + DOI, D6 re-verify from the download | 0 hash mismatches from the archive | Hamm |
| 6 | OPEN-36 / OPEN-38 owner values | owner ruling | Hamm |

## 6. A3 figure work: what changed and what is still open (2026-11-11)

Commits: `5a89597` (font floors), `f20fe5c` (Fig 1E split into E and F), `c6941d9` (Fig 2C 9 traces -> 3),
`e6b33ac` (Fig S3 redrawn at print size), `458fb2c` (Fig S1 architecture full width, map at print size).

Measured at print size, before -> after: Fig 1 legend 6.11 -> 7.00; Fig 2 SOM label 5.17 -> 6.10, letters
8.46 -> 9.02; Fig 3 panel D 5.97 -> 6.50, letters 7.96 -> 9.50; Fig S3 1.70 -> 6.51; Fig S1 architecture
2.57 -> 6.11, S1 map 5.77 -> 6.50. Every item now clears the 6 pt floor (5.5 ticks, 9 letters).

A method error to avoid repeating: the first pass divided pixel width by an assumed 300 dpi. Three sources save at
150 dpi, so those figures' native size, and therefore their printed type, was understated by half. Read the dpi
from the file (`PIL.Image.open(p).info["dpi"]`), never assume it.

Still open in A3, recorded rather than fixed:
- **The three T1 atlases** (`results/viz/t1/V3_{kd,sst,sstd}_n4000_atlas.png`) print at scale 0.896, so their
  6.5 pt titles land at 5.8 pt. No generator for them exists in `scripts/` (`grep` for `V3_kd`, `_n4000_atlas`
  finds only the tex references), so they cannot be redrawn reproducibly from this repo. Either the generator is
  found and committed, or the atlases are redrawn by a new script under the visualization contract. Not guessed at.
- A second, independent visual review of the four redrawn figures: running.

