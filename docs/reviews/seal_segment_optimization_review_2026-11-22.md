# Seal and atomic-save segment: repository review

Date: 2026-11-22
Code reviewed: jomission at commit `83b83a3`
Change in this review: report and evidence files only (`docs/reviews/`). No code changed.

Priority scale:

- P0: sealed evidence can change through the production save path, or the seal is inactive there.
- P1: a measured route changes or removes a sealed path while the guard is installed.
- P2: a latent or inferred gap, or a measured cost that limits the tests.
- P3: hygiene, or a cost with no effect on seal integrity.

Status labels: measured, observed, inferred, hypothesis, unknown.

## 1 Scope and current behavior

- Scope: the seal segment of jomission at commit `83b83a3`. Probes and benchmarks ran against that commit. The seal and save modules were not edited.
- Other agents edited README and skill files in the same clone while these runs were going. Neither measured file changed (`jomission/harness/seals.py`, `jomission/simulation/atomic_save.py`).
- The independent review named branch `_temp_seal_guard_fix`. This review does not check that the branch points at `83b83a3`.
- Guard: `jomission/harness/seals.py`.
  - `install()` at L74-82 rebinds `builtins.open` and `io.open` to `_guarded_open` (L68-71). A second call changes nothing (L77).
  - Other functions: `uninstall()` L85, `active()` L95, `check_write()` L54, `sealed_paths()` L33, `authorized()` L38.
  - Env variable `JOMISSION_ALLOW_SEAL_WRITE` at L23.
  - Registry: `manifests/sealed_artifacts.json` (L22). It has 396 entries. Entries compare as raw strings (L35).
  - For each open with a mode that contains `w`, `a`, `x` or `+`, the guard resolves the path and reads the registry (L69).
- Install point: `tests/conftest.py` L6 only. No production module installs the guard.
  - `jomission/simulation/atomic_save.py` does not import `seals`.
  - `jomission/simulation/factorial_v0p2_fast.py` imports `atomic_write_json` (L71) and calls it at L396 and L414. It does not import `seals`.
  - `scripts/project_check.py` L97-108 imports `seals` and calls `check_write` as a probe only.
- Save path: `jomission/simulation/atomic_save.py`.
  - Temp file from `tempfile.mkstemp`, mode 0o600 (L57).
  - Commit by `os.replace` in `_write_temp_and_replace_bytes` (L67).
  - Callers: `atomic_write_json` (L78-81), `persist_phase_snapshot` (L121-165), `append_manifest_receipt` (L95-101), `append_trial_snapshot` (L197), `verify_artifacts_readable` (L235-242).

## 2 Assessment

The seal guard checks one call type. It patches `builtins.open` and `io.open` only, and it is installed only under pytest. The probe measured 11 of 19 cases where a sealed path changed or vanished with no error while the guard was active (n=1 per case, one fresh process per case). Two of those cases are the production save functions, `atomic_write_json` and `persist_phase_snapshot`. Both replace sealed paths through `os.replace` at atomic_save.py:67, and no guard runs on that production path at all.

Cost (measured, test-only): with the guard installed, a registered-root open-write-close costs about 229 to 254 us more than without it (two runs of 5 reps each, see section 3). Opens outside the root cost about 44 to 50 us more. `os.write` on an open descriptor is not patched and shows no measurable change.

The review recommends an explicit seal check at the save commit points now, and a route-independent digest gate after that. Until then, the seal should not be described as protecting sealed files.

## 3 Findings

Counts: P0 3, P1 6, P2 9, P3 6. Total 24.

F4 and F5 are the two findings from the independent review of `_temp_seal_guard_fix`. The probe reproduced both at `83b83a3`. Their status is measured. Each still rests on one case, so confidence in the route is high and confidence in other forms of the same route is lower.

| ID | Priority | Finding | Status | Confidence | Evidence | Impact | Recommendation |
|---|---|---|---|---|---|---|---|
| F1 | P0 | `atomic_write_json` replaces a sealed path with the guard active. | measured (n=1, case `atomic_write_json_onto_sealed`) | High | atomic_save.py:67, :78-81 | Sealed file replaced, 17 B. No error. | Add a seal check before `os.replace` at atomic_save.py:67. Red test first (check 1). |
| F2 | P0 | `persist_phase_snapshot` replaces a registered `pre_snapshot.json`. | measured (n=1, case `persist_phase_snapshot_onto_registered`) | High | atomic_save.py:121-165, :141-149, :67 | Snapshot replaced, 203 B. No error. | Same commit-point check as F1. |
| F3 | P0 | The guard runs only under pytest. No production module installs it. | observed (grep of tree) | High | seals.py:74-82; tests/conftest.py:6; scripts/project_check.py:97-108 (probe only) | Production writes to sealed paths run unguarded. | Owner decides where the guard installs for production, or stops calling it a production control. |
| F4 | P1 | `io.FileIO(path, 'w')` writes a sealed path with no raise. Reproduces independent review finding 1. | measured (n=1, case `io_fileio_w`) | High for this route | seals.py:79-81 (only `open` names patched) | Sealed file truncated to 1 B. | Route-independent digest gate (option D). Audit hook (option B) after its coverage is measured. |
| F5 | P1 | `os.rename` of a parent directory moves a sealed file out of its registered path. Reproduces independent review finding 2. | measured (n=1, case `dir_rename_parent`) | High | seals.py:79-81 (`os.rename` not patched) | Registered path gone. Bytes intact at `results_moved/sealed.json`. The guard saw no call. | Check ancestry on rename and remove (option A or B). Digest check on lookup (option D). |
| F6 | P1 | `os.replace` onto a sealed path replaces its bytes. | measured (n=1, case `os_replace_onto_sealed`) | High | seals.py:79-81; atomic_save.py:67 | Sealed bytes replaced, 20 B. | Option D. Seal check at commit points (option E). |
| F7 | P1 | `os.open(path, O_WRONLY or O_TRUNC)` empties a sealed path. | measured (n=1, case `os_open_o_trunc`) | High | seals.py:79-81 | Size 0 at once after open. | Option B (open event with flags) or option D. |
| F8 | P1 | `os.unlink` removes a sealed path. | measured (n=1, case `os_unlink_sealed`) | High | seals.py:79-81 | File removed, no raise. | Option B (remove event), or option D (lookup finds no file). |
| F9 | P1 | `append_manifest_receipt` rewrites the manifest through unguarded `os.replace`. | inferred (mechanism measured in F6; manifest case not probed) | Medium | atomic_save.py:95-101, :67 | A registered manifest path would be overwritten with the guard active. | Probe the manifest case (check 4). Cover it with the commit-point check (option E). |
| F10 | P2 | `os.write` on a descriptor opened before install changes sealed bytes. | measured (n=1, case `fd_opened_before_guard`) | High for this route. Narrow: needs a descriptor opened before install. | seals.py:68, :74 | Sealed bytes changed. Size stays 32. | Option D catches it. The guard cannot. |
| F11 | P2 | Registry entries compare as raw strings. A symlink name, a `./` entry or an absolute entry fails open. | measured (n=1 per form). Latent: 0 of 396 registry entries use these forms (grep). | High for the mechanism | seals.py:35, :43-51 | Writes pass with no raise. Latent until an entry uses one of these forms. | Resolve entries the same way as candidates. Reject other forms when the registry loads (check 11). |
| F12 | P2 | `_relative` returns None on TypeError, ValueError or OSError. `check_write` then passes. | inferred (code read; not probed) | Low | seals.py:43-51, :54-65 | Non-path targets and unresolvable paths are unguarded. | Probe (check 7). Then decide the None case. |
| F13 | P2 | `ROOT` is `parents[2]` of `__file__`. A moved or vendored `seals.py` guards no path. | inferred (code read; not probed) | Medium | seals.py:21 | Silent total fail-open on relocation. | Probe a relocated copy (check 8). Set ROOT from the repository root explicitly. |
| F14 | P2 | The registry JSON is read on every write-mode open inside ROOT. | measured (2 runs, 5 reps each, 20000 ops per rep, median) | High | seals.py:34 | Run 2: 35.07 us bare, 288.98 us guarded per open-write-close (+254 us). Run 1 (not saved): 31.96 and 261.25 us (+229 us). Test-only. Each open pays this; the cost in the real trial loop is not measured. | Cache the registry, keyed on mtime, or load it once per install. Re-run the bench (check 6). |
| F15 | P3 | `resolve()` runs on every write-mode open, including paths outside ROOT. | measured total (2 runs, 5 reps each). Attribution to `resolve()` inferred. | Medium | seals.py:45 | Run 2: 36.22 us bare, 86.64 us guarded (+50 us). Run 1: +44 us. | Compare the mode and the path prefix before `resolve()`. Measure again. |
| F16 | P2 | `fsync(fd)` alone may not flush the macOS drive cache. The durability claim in the module docstring rests on it. | unknown (no source checked here; not measured) | Low | atomic_save.py:65; module docstring L9 | A phase reported durable may be lost on power loss on this machine (Darwin arm64). | Owner decides. Either state the limit in the docstring, or verify the behavior and switch to `F_FULLFSYNC` if needed (check 9). Power-loss behavior cannot be tested here. |
| F17 | P2 | `_fsync_dir` swallows `OSError` from `os.open` of the directory. | observed (code read) | High | atomic_save.py:41-42 | A skipped directory fsync gives no signal. | Return or log a flag. Test by forcing the open to fail. |
| F18 | P2 | `append_manifest_receipt` reads the whole manifest, then rewrites it, with no lock. | inferred (code read; not measured) | Low | atomic_save.py:95-101 | Concurrent appenders can lose a receipt. | Run two concurrent writers on a copy and count receipts (check 10). Add a lock only if receipts are lost. |
| F19 | P2 | `verify_artifacts_readable` detects accidental corruption, not tampering. | inferred (code read; not run) | High | atomic_save.py:242 | A rewritten snapshot and a rewritten receipt would both pass as verified. | State the limit in the docstring. Store digests on separate storage (option D). |
| F20 | P3 | The env token `all` disables the guard for the whole process. | observed (code read; not run) | High | seals.py:38, :60 | One exported variable silences all sealed-path checks. | Log when the token is set. Owner decides whether to keep it. |
| F21 | P3 | Temp files and replaced artifacts have mode 0o600. | observed (temp-file mode from code); measured (n=1, replaced artifact) | High | atomic_save.py:57 | Group and other readers lose access. Effect on the pipeline is not checked. | Owner decides the intended mode. Test with the pipeline readers. |
| F22 | P3 | `verify_artifacts_readable` reads each snapshot twice. `read_text` has no encoding at L96, L110, L235. | observed (code read) | High | atomic_save.py:235, :242, :96, :110 | Double read I/O. Decoding may depend on the locale. Writes use utf-8 (L100). | Read once. Pass `encoding="utf-8"` on read calls. |
| F23 | P3 | `Path.touch` on a sealed path changes mtime and raises nothing. | measured (n=1, case `path_touch_sealed`) | High | seals.py:79-81 (`os.utime` not patched) | Metadata only. Content intact. | Record as known. Owner decides whether mtime counts as a change. |
| F24 | P3 | The per-trial `fsync` cost in `append_trial_snapshot` is unmeasured. | unknown (cost not measured) | High (that it is unmeasured) | atomic_save.py:197 | Per-trial cost is the guard cost (F14) plus an unknown `fsync` cost. | Measure `fsync` latency per call on the target volume before any change. |

Reconciliation:

- Overhead, inside ROOT, run 2: 288.98 us minus 35.07 us is 253.9 us. Run 1: 261.25 us minus 31.96 us is 229.3 us. Both runs agree on the order of magnitude. The bench reports medians of 5 reps. The rep ranges are in the saved JSON.
- Overhead, outside ROOT, run 2: 86.64 us minus 36.22 us is 50.4 us. Run 1: 73.99 us minus 30.24 us is 43.75 us.
- `os.write`, run 2: 2.1365 us guarded and 2.0674 us bare. The median difference is 0.07 us, and the rep ranges overlap (bare 2.06 to 2.19 us, guarded 2.08 to 2.60 us). No effect is detected at 5 reps. `os.write` is not patched, so none is expected.
- Version mismatch: the probe and bench ran on CPython 3.13.7. The audit table and `fileio.c` cited in section 5 are 3.13.18 docs and branch 3.13 at fetch time. Options that depend on audit events are not confirmed for 3.13.7 until checked.
- Contradiction, kept and flagged: PEP 578 says a raising hook should generally terminate the runtime. The 3.13 `sys` docs say a hook can raise to abort the operation. Both statements stay. The PEP is historical.

## 4 Options and trade-offs

| Option | Covers | Gaps | Status | Cost |
|---|---|---|---|---|
| A. Extend the patch list (research option A) | By design: the open family, rename, replace, remove, truncate, link, symlink, mkdir, rmdir. Would address F5, F6, F7 and F8. F4 only if a `FileIO` patch works on 3.13 (unknown). | Captured references (`from os import rename`). Writes on open descriptors (F10). mmap. C code. | hypothesis (coverage by design; not run). | More patched names. A check on every call in the hot path. Not measured. |
| B. Audit hook (research option B) | Per the 3.13.18 table: open (builtins, io, os, FileIO), mkdir, rmdir, remove, rename (rename, replace, renames), truncate, link, symlink. | No event for writes on descriptors or mmap (absence from the table only). C code and ctypes. Removal and raise rules unresolved (PEP 578 against 3.13 docs). Descriptors opened before install (F10). | hypothesis (docs and source read; not run). Not checked on 3.13.7. | One install point. Hook cost not measured. |
| C. Mode bits and flags (research option C) | Other users. Tools that honor mode bits. The directory write bit blocks rename and unlink (rename(2)). | The same user can `chmod` back or clear `uchg`. Root bypasses it. An ordinary user cannot set `chattr +i`. Current macOS `uchg` behavior not verified (the chflags source is from 1993). Inferred: the save path needs write access, so C also blocks saves unless the save step changes modes. | hypothesis (Linux man pages 2026; a 1993 Apple page). | Low setup cost. Blocks saves (inferred). Not measured. |
| D. Digest verification on read and at commit (research option D) | Any write route that changes content: F4 to F7, F10. Does not depend on the route. A check reads the file itself. | Detects, does not prevent. The sealed file stays changed until the check runs. The manifest must sit on separate storage. Renames and deletes seen only by path lookup. Gap between check and use. | inferred (design; not run). No external source needed. | A full read of each sealed file per check. Not measured. |
| E. Seal check at save commit points (new; from F1, F2, F9; not in research) | The save path: F1, F2, F9. | Only the save module. Routes F4 to F8 and F10 stay open. | inferred (from the probe; not tested). | One call per commit. Not measured. |

Recommendation: D is the required control, and E is the fix for the production path now.

- D does not depend on the write route. It covers the measured routes F4 to F7 and F10 without patching more names.
- The guard cannot close those routes, as the probe shows.
- D detects a change at the next check. It does not stop the change. E stops the production save path, which D does not.
- The research put B before D. This review puts B after D. B has an unresolved raise conflict and no measurement here. B is the next layer, after its coverage is measured on 3.13.7.
- A is not a control on its own. C is an accident guard for other users only.
- The cost of D is not measured. The owner decides where the manifest sits.

## 5 Current-research validation

Every external claim below is a hypothesis until it is reproduced here. No new web access was used in this review. The sources and dates come from the research step.

| Source | Page date or version | Claim it supports | Limitation | Status |
|---|---|---|---|---|
| https://docs.python.org/3.13/library/audit_events.html | 3.13.18 docs, footer Oct 26 2026 | Audit events for open, mkdir, rmdir, remove, rename (rename, replace, renames), truncate, link, symlink. Supports option B. | Table read in research. Not reproduced. Probe ran on 3.13.7. | hypothesis |
| https://docs.python.org/3.13/library/sys.html | 3.13.18 docs, Oct 26 2026 | `sys.addaudithook` exists. A hook can raise to abort the operation. | Hook removal not covered by the page. | hypothesis |
| https://docs.python.org/3.13/library/io.html | 3.13.18 docs, Oct 26 2026 | `open` and `io.open` are the same object (per research). | Not reproduced. The guard patches both names separately. | hypothesis |
| https://raw.githubusercontent.com/python/cpython/3.13/Modules/_io/fileio.c | Branch 3.13, fetched 2026-11-22 | `FileIO.__init__` raises audit event `open` before it gets a descriptor. Supports option B for F4. | Branch tip, not the 3.13.7 build used in the probe. | hypothesis |
| https://peps.python.org/pep-0578/ | Final. Created 2018-06-16. Modified 2024-06-03. | Hooks cannot be removed. A raising hook should generally terminate the runtime. | Conflicts with the 3.13 docs. Historical. | hypothesis (conflict kept). |
| https://man7.org/linux/man-pages/man2/rename.2.html | Linux man-pages 6.19, dated 2026-02-08 | Rename fails with EACCES when the directory is not writable. Supports option C. | Linux only. macOS not tested. | hypothesis |
| https://man7.org/linux/man-pages/man2/chmod.2.html | Linux man-pages 6.19, dated 2026-02-08 | Only the owner or a privileged process can change mode. EPERM for immutable files. | Linux only. | hypothesis |
| https://man7.org/linux/man-pages/man1/chattr.1.html | e2fsprogs 1.47.4, March 2025 | `+i` blocks modify, delete and rename. | An ordinary user cannot set it. | hypothesis |
| https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/chflags.2.html | Apple archive, dated June 9 1993 | `UF_IMMUTABLE` can be set and cleared by the owner or superuser. | Old page. Behavior on macOS 27.0.1 not verified. | hypothesis (stale source). |
| None (option D) | Inferred design | A digest recheck catches a change by any route. | No external source. `hashlib` not checked. | inferred |

## 6 Verification plan

Environment for every check: Darwin arm64, macOS 27.0.1, CPython 3.13.7, clone at `83b83a3`.

Saved evidence: `docs/reviews/seal_segment_evidence/`.

- `seal_probe_results.json`: 19 probe cases, one fresh process each (n=1).
- `bench_bare_run2.json` and `bench_guard_run2.json`: overhead, run 2, 5 reps each.
- `bench_overhead.py bare|guard` and `probe_seal.py`: the scripts that produced them. Both use the repository root from their own location and write scratch files to the system temp directory.

Run 1 of the bench (31.96 / 261.25 us inside root; 30.24 / 73.99 us outside) was not saved. Run 2 is the reference.

The current baseline has n=1 per probe case. Check 1 plans three runs per case before a case is treated as stable.

| Check | Exact probe or test | Baseline at 83b83a3 | Pass |
|---|---|---|---|
| 1. Red tests | One pytest case per entry in `bypass_checks` of `seal_probe_results.json`. New test file only. Run against `83b83a3`. | 11 silent changes (F4 to F8, F10, F11 x3, F1, F2). 1 metadata change (F23). 7 expected results. | The 11 silent cases fail. The 7 expected cases pass. The F23 expectation is set by the owner. |
| 2. Guard fix | The same 19 cases, after the fix. | As check 1. | All 11 raise and leave sealed bytes unchanged. The 7 controls still pass. |
| 3. Production path (F1, F2, F3) | Plain `python` process, no pytest, a copy of the registry. Call `atomic_write_json` and `persist_phase_snapshot` on registered names. | Replaced: 17 B and 203 B. | Each call raises and leaves bytes unchanged. |
| 4. Manifest (F9) | Call `append_manifest_receipt` on a registered manifest name. | Not probed. | Raises, and bytes unchanged. |
| 5. Digest gate (D) | Digests for the 396 registry entries, held outside the writer's reach. Run the 11 silent cases, then the verifier. Time the full verify, 5 reps. | None. New. | Verifier fails on all 11. Timing reported as median and range. No speed claim before it is measured. |
| 6. Bench (F14, F15) | `bench_overhead.py`, 5 reps, 20000 ops per rep, median and range. Run before and after any cache change. | Run 2 inside ROOT: 35.07 us bare, 288.98 us guarded. Outside: 36.22 us bare, 86.64 us guarded. `os.write`: 2.07 us bare, 2.14 us guarded. | Inside-ROOT guard cost falls by more than the rep-to-rep range. The range is reported. |
| 7. Non-path target (F12) | Call `check_write` with an int descriptor, and with a path holding a NUL byte (ValueError). | Not probed. | Result stated for each. The None case is then decided by the owner. |
| 8. Relocation (F13) | Copy `seals.py` to a temp tree at a different depth. Install. Attempt a sealed write. | Not probed. | Raises. |
| 9. `fsync` behavior (F16) | Time `os.fsync` and `fcntl.F_FULLFSYNC` on the target volume. Check the macOS documentation for `fsync` and `F_FULLFSYNC` first. 5 reps, median and range. | Not measured. | Reports latency only. Power-loss durability cannot be tested here. |
| 10. Manifest concurrency (F18) | Two processes append to one copy of a manifest. Count receipts against appends made. Repeat. | Not measured. | Zero receipts lost over repeated runs. Any loss is reported with counts. |
| 11. Registry forms (F11) | Symlink, `./` and absolute entries in a temp registry. | 3 fail-open (measured). | Raises, or the entry is rejected at load. |

## 7 Implementation boundary

- This review changes no code. It edits no file under `jomission/`, and it makes no change to sealed artifacts. It adds only this report and its evidence folder.
- Proposed next steps, after owner approval:
  1. Add the red tests of check 1 in a new test file. Run them against `83b83a3` and confirm the failures.
  2. Add the commit-point seal check (option E) in `atomic_save.py`, at L67 and in the writers that call it. Then run checks 2 and 3.
- Out of scope for these steps:
  - The audit hook (option B). Wait until its coverage is measured on 3.13.7.
  - Mode bits and flags (option C) on save paths.
  - Registry format and ROOT changes (F11, F13). Wait for checks 8 and 11.
  - Mode 0o600 (F21), the `fsync` durability claim (F16) and the manifest lock (F18). Owner decisions first.
  - The digest gate (option D). Wait for the owner to set where the manifest is stored.
  - Other parts of `seals.py` beyond the seal check.
- Owner decisions needed:
  - Where the guard installs for production (F3).
  - The intended file mode (F21).
  - Whether to keep the `fsync` durability claim, or to change it (F16).
  - Where the digest manifest is stored (option D).
  - Whether to keep the `all` token (F20).

## Revision notes

Corrections made after the independent review of this report (REVISE):

- Removed the claim that the clone held unrelated changes before the runs. The changes came from parallel edits during the run.
- Replaced `derived` status labels with `inferred`, and F16 with `unknown`.
- Added the bench files and results to `docs/reviews/seal_segment_evidence/`. The timing figures now cite them. Added run 2 figures and their range.
- F10: replaced the unsupported "first 20 B overwritten" with the recorded result (sealed bytes changed, size 32).
- Callers: corrected the `factorial_v0p2_fast.py` references (import L71, calls L396 and L414). Added `scripts/project_check.py` as the one production-adjacent importer.
- F19 and F22 statuses corrected to inferred (code read; not run). Removed the unmeasured "adds up" claim from F14.
- Check 1 now states that the baseline has n=1 and that the plan asks for three runs.
