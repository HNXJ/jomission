# R2 pre-seal checks, 2026-11-26

Spec: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` section 4. Device D1 (CPU, `JAX_PLATFORMS=cpu`), venv
`/home/hamm/.venvs/jomission058gpu` (jaxfne 0.5.8 from tag `v0.5.8` = 11957afa, jax/jaxlib 0.10.2). Layout
`e2_3unit_n4000_dens_k1`, seed 7, T 1 s. Outputs in session scratch, not committed (`.npz` files are untracked by rule).

## 1. P0 baseline, default route: EQUAL

| Side | Code | Tag | construct s | simulate s |
|---|---|---|---|---|
| A | `git archive cc33b37` | `r2bl_cc33_default_s7_T1` | 16.3 | 212.7 |
| B | `git archive a4b9a89` (code identical to the seal commit: later commits change only docs and manifests) | `r2bl_seal_default_s7_T1` | 10.6 | 211.3 |

Comparator `scripts/r2_compare_runs.py` (sha256 `6866a6d08e5b3c9fb19c6f678f3704fdecffd4eff6357a65a3eae7d281c8a024`;
first run as a session-scratch copy with the same logic, re-run from the committed file with the same verdicts):
`.npz` key sets, then per array dtype, shape and `tobytes()`; JSON fields
`sizes`, `edges`, `p_local`, `p_inter`, `rho`, `depth_rel_L4`, `tau_sst_e_ms`, `g_columns`, `g_reversal_mV`, `g_tau_ms`.

Output: keys equal (22 vs 22); all 22 arrays equal (including `counts` int32 (10000, 9), `g_nS` float32 (10000, 6),
`lfp_pA` float32 (10000, 3)); all 10 fields equal; 48 491 spikes; both `jax_devices` `["cpu:0"]`, both `jaxfne_version`
0.5.8; `edge_delivery` absent at cc33b37, `default` at B. Verdict line: `EQUAL`, exit 0.

Known negative: B copied with the lowest bit of `g_nS[5000, 0]` flipped. Output: `npz g_nS: DIFFER`, `DIFFERENT`, exit 1.

sha256 of the compared files:

| File | sha256 |
|---|---|
| A `.npz` | `5b9876a09b7d5759778b2e2b2e5ba6e48998fc7570a483c7f996742c5c10d7a2` |
| B `.npz` | `5b9876a09b7d5759778b2e2b2e5ba6e48998fc7570a483c7f996742c5c10d7a2` (container bytes also identical) |
| A `.json` | `fab791164fd1d5da0df5fc8201ea1293e41c8f631a8f26bc799670009b3cf315` |
| B `.json` | `c2e78424ec7259ff72ebba7689987e8568bbaae012835edbe461d0f210273918` (differs: wall times, SHA, added fields) |
| negative `.npz` | `3eb9f7fc89b24e7f8d070bd5c80a681a2cf801adc0d20b3682c7384f6953deed` |

Thread environment of section 1: `JAX_PLATFORMS=cpu` only, BLAS threads at default, the two sides run one after the other.

## 2. Delivery check: not runnable on 0.5.8

`JAXFNE_EDGE_DELIVERY=event` on B failed before simulation:
`TypeError: RuntimeConfig.__init__() got an unexpected keyword argument 'edge_delivery'`.
`git grep edge_delivery v0.5.8` in the JaxFNE repo: no match. 351f6048 = `v0.5.9-107-g351f6048`, a descendant of
`v0.5.8`. Consequence (spec section 5): scored runs use the default route.
