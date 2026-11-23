# Jomission v2 crosswalk (draft r2, not ratified, not committed)

Sources, read from `origin/dev`:
- Historical checklist: `checklist-score.md` v4 (50 items × 2 pts; current 69/100; interim note 2026-11-11).
- Historical LANSA rubric: `paper/jomission_lansa/quality_rubric.md` (gates M1–M5; criteria C1–C8, sum 100). Last changed at `6f0e51c`.
- Proposed v2: goals G0–G4 and checklists L01–L14, F01–F08, O01–O10, E01–E08 (pasted draft, 2.0-draft.1). Criterion text in `_temp_v2_criteria.md`.

Scope: criteria mapping only. No rescoring, no weight change, nothing ratified.

Status key:
- **Direct**: the v2 criterion tests the same thing.
- **Partial**: the v2 criterion covers part of the v4 item, or overlaps it.
- **Gap**: no v2 criterion covers it. The nearest v2 ID is listed for reference only.

Rule applied to lineage rows: v4 rows about the EI-regime, AGSDR and H-balance lineages are Gap. Section 3.4 states that no v2 goal covers those lineages. Mapping them to O-criteria would assume a goal the owner has not approved.

## 1. v4 items → v2 criteria

| v4 # | v4 category | v4 objective (short) | v2 ID | Status | Note |
|---|---|---|---|---|---|
| 1 | Science | Tier 3 scientific batteries under an authorized TODO | E04 | Partial | Battery-specific criteria absent in v2 |
| 2 | Science | V2 local-operation gate PASS | E04 | Partial | Gate is recorded FAIL. E04 tests gates passing, not the lineage verdict |
| 3 | Science | Viable operating regime found | — (nearest O04) | Gap | EI-regime lineage |
| 4 | Science | Regime verdict recorded | — | Gap | EI-regime lineage |
| 5 | Science | SCI-EI-NATIVE-STATE-1 sealed verdict | — (nearest O05) | Gap | EI-regime lineage |
| 6 | Science | SST repeated-spiking question answered | — (nearest O08) | Gap | EI-regime lineage |
| 7 | Science | f-I assay (exploratory) | — | Gap | |
| 8 | Science | Engine-equivalence of f-I assay | — | Gap | |
| 9 | Science | V0 and V1 exist for canonical lineage | — (nearest E06) | Gap | Visualization stages have no v2 criterion. O04 covers readouts only |
| 10 | Science | V2 and V3 exist for canonical lineage | — (nearest E06) | Gap | Same |
| 11 | Science | AGSDR design sealed, amendments 1–3 | — (nearest E03) | Gap | AGSDR lineage |
| 12 | Science | AGSDR stage 0 executed with receipts | — (nearest O05) | Gap | AGSDR lineage |
| 13 | Science | AGSDR stage 0 selection under frozen rule | — (nearest O05) | Gap | AGSDR lineage |
| 14 | Science | AGSDR stages 1 and 2 executed | — (nearest O05) | Gap | AGSDR lineage |
| 15 | Science | B10 term scan clean | — | Gap | Term scan of the runner; not a test or gate pass |
| 16 | Science | Efrac tau correction applied | — (nearest O05) | Gap | AGSDR lineage |
| 17 | Science | H balance rule acts under sealed test | — (nearest O06) | Gap | H-balance lineage |
| 18 | Science | PV drive scan (matched-state) | — | Gap | Owner kept it (Q15); no spec |
| 19 | Science | H rho and Fano rise resolved (H4) | — (nearest O06) | Gap | H-balance lineage |
| 20 | Science | H rule in calibrated model | — (nearest O02) | Gap | H-balance lineage |
| 21 | Documentation | Cloud setup handout | E05 | Partial | |
| 22 | Documentation | Fresh-clone reproduction | E05 | Direct | |
| 23 | Documentation | Checklist versioned with history | E08 | Partial | |
| 24 | Documentation | Results inventory on dashboard | E06 | Partial | |
| 25 | Documentation | Open stack current | E08 | Direct | |
| 26 | Documentation | Handoff regenerated and current | E08 | Direct | |
| 27 | Documentation | TODO consistent with state | E08 | Direct | |
| 28 | Documentation | Every claim has a receipt | E02 | Partial | LANSA L08 covers manuscript numbers only |
| 29 | Documentation | Sealed-artifact policy enforced | E03 | Direct | |
| 30 | Documentation | Commit and PR provenance | E02 | Partial | |
| 31 | Documentation | Gate thresholds in one place | E04 | Partial | v4 tests a design property; E04 tests that gates pass |
| 32 | Documentation | Design-to-code traceability (AGSDR) | E02 | Partial | |
| 33 | Documentation | Goals document ratified | — | Gap | Governance; covered by the adoption gate, not a criterion |
| 34 | Documentation | Receipts name generator and hash | E02 | Direct | |
| 35 | Documentation | Compile cache trust documented | E02 | Partial | |
| 36 | Dashboard | Dashboard committed, no external resources | E06 | Partial | |
| 37 | Dashboard | Deterministic dashboard build | E06 | Direct | |
| 38 | Dashboard | Tier 0 test-count chart | E06 | Partial | Counts only |
| 39 | Dashboard | Tier 1 test-count chart | E06 | Partial | Counts only |
| 40 | Dashboard | Tier 2 test-count chart | E06 | Partial | Counts only |
| 41 | Dashboard | Tier 3 shown as open | E08 | Partial | |
| 42 | Dashboard | Edge-identity table | E02 | Partial | |
| 43 | Dashboard | Checklist score chart | E06 | Partial | |
| 44 | Dashboard | GitHub Pages current | E06 | Partial | No Pages criterion in v2 |
| 45 | Dashboard | Open items shown with reasons | E08 | Partial | |
| 46 | Rest | jaxfne 0.5.8 pinned | E01 | Direct | |
| 47 | Rest | Canonical network identity | E02 | Partial | |
| 48 | Rest | Tiers 0–2 pass on head | E04 | Direct | |
| 49 | Rest | Sealed registry and guards pass | E04 | Partial | |
| 50 | Rest | CI green, tree clean, pushed | E04 | Partial | Clean tree and push are outside E04 |

Counts: Direct 9, Partial 22, Gap 19. Total 50. Gap = items 3–20 (18 lineage rows) plus item 33.

## 2. LANSA rubric → v2 LANSA criteria

| Rubric item | Weight | v2 ID | Status | Note |
|---|---|---|---|---|
| M1 build clean, refs resolved | gate | L12 | Partial | v2 has no build gate; L12 is 10 pts |
| M2 no claim exceeds verdict | gate | L09, L02 | Partial | |
| M3 every number traces to source | gate | L08 | Direct | |
| M4 cited sources hashed or stated unavailable | gate | L07, L08 | Partial | |
| M5 fresh independent skeptical review, no open major finding | gate | L11 | Direct | |
| C1 numerical accuracy | 25 | L08 | Partial | Weight 25 vs 10 |
| C2 claim-evidence calibration | 20 | L09, L10 | Partial | Weight 20 vs 15 combined |
| C3 statistics | 15 | L06 | Partial | Weight 15 vs 10 |
| C4 reproducibility | 15 | L07, E05 | Partial | Weight 15 vs 10 + E05 15 |
| C5 internal consistency | 10 | L12, L08 | Gap | Judgment call: overlaps L12 and L08 in part; no direct match |
| C6 figures (print size, caption, traces to producer) | 5 | L12, L08 | Partial | "Traces to producer" is the L08 test |
| C7 methods completeness | 5 | L03, L07 | Partial | |
| C8 prose | 5 | L12 | Partial | |
| (no rubric item) | — | L01, L04, L05, L13, L14 | v2 only | See §4 |

## 3. Differences that block direct comparison

1. **Gate semantics.** LANSA caps the score: any DEFECT gate caps at 49; any UNKNOWN gate caps at 89; 90+ needs all five gates PASS. v2 has no numeric cap; an unmet gate "prevents final ACCEPT". The GATE-A..F definitions are not in the extract used here, so this comparison is unchecked against them.
2. **Scoring of unknowns.** LANSA: UNKNOWN scores 0 per item; a criterion scores its weight times the fraction of its items that PASS, so partial credit is possible. v2: PARTIAL is itemized; UNKNOWN and FAILED score 0. v4: a single 0 covers "not met" and "not verifiable". A v4 zero may be UNKNOWN under v2, so v4 and v2 scores are not comparable.
3. **Weights.** LANSA C1 (25) and C2 (20) carry more weight than the v2 criteria they map to (L08 10; L09 10 plus L10 5). Equal scores under each would describe different things.
4. **Scope (owner decision).** v4 Science items 3–20 cover EI-regime, AGSDR and H-balance lineages. No v2 goal covers them. Section 1 marks these Gap and lists a nearest v2 ID for reference only. If the owner maps these lineages to G3 (omission modeling), the nearest IDs become Partial; that mapping is not assumed here.
5. **Fixation.** No v4 item covers G2. The v2 F-list has no historical score to crosswalk.
6. **Prospective rule.** Under v2, new criteria are prospective. The v4 items 17, 19 and 20 were defined after the H results, which the v4 notes record. They are the case the rule is meant to block.

## 4. v2 criteria with no historical counterpart

- L01, L04, L05, L13, L14 (LANSA). L05 (frozen protocols and amendments) has no LANSA rubric item, although the manuscript depends on it.
- F01–F08 (fixation). No historical item.
- O01, O03, O07, O09, O10 (OXM). No historical item; O05 and O06 are only partly covered by v4 lineage rows marked Gap.
- E07 (independent engineering review ACCEPT). No v4 item.

## 5. Open for the owner

1. Map v4 lineage items 3–20 to a v2 goal (for example G3), or add a goal for the EI-regime and AGSDR lineages. Until then they stay Gap.
2. Choose between LANSA's numeric caps and v2's gate rule (or keep both).
3. Choose whether v4 zeros are re-scored under the v2 UNKNOWN/FAILED split. Under v4 they stay as recorded.
4. Provide the GATE-A..F definitions so the gate comparison can be checked.
5. Confirm the rubric source: `6f0e51c` is the last change to the LANSA rubric file; the handoff cites `d289583`.

## Review history

- Independent review (2 reviewers, workflow `wf_71265266-895`, 2026-11-23). v4 mapping: REVISE, 3 major (lineage rows, row 2, row 31). Fixed in r2: lineage rows → Gap with nearest ID; row 2 → Partial; row 31 → Partial; row 11 → Gap; row 50 → Partial; rows 9, 10 → Gap; row 15 → Gap. LANSA mapping: ACCEPT, 2 minor. Fixed in r2: UNKNOWN wording; C6 → L08 added; C5 judgment note; GATE-A..F gap noted.
- Owner decision (2026-11-23): lineage rows (v4 items 3–20) stay Gap. No mapping to a v2 goal is assumed.
