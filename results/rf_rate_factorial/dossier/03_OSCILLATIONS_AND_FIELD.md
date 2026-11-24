# 03 — OSCILLATIONS & FIELD (cell A_RFoff_RateRef)

_Spectral (PSD/TFR band) characterization of the field proxy and its coupling to firing rate. Field is `proxy_readout` (area-attributed linear partition of shared laminar contacts, depth-only kernel, 16 contacts), `physical_amplitude_calibrated=False` → all power is in arbitrary units and all quantitative claims are relative/log. See `field_proxy_manifest.json`._

- Field provenance: `5e993844b1af5f55` matches sealed manifest [MEASURED]; areas ['V1', 'V4', 'FEF', 'PFC'] [MEASURED]; 16 contacts, depth z∈[0,1] [MEASURED from meta]; reconstruction `sum_a field_a == global` verified [MEASURED, manifest].
- **Provenance flag:** `field_proxy_example_AAAB.npz` is **byte-identical** to `field_proxy_pre_0000.npz` (same field array, files differ only in meta key) [MEASURED, array_equal=True]. The 'example' AAAB field is therefore a pre-exposure trial, **not** a distinct post-exposure recording; any pre-vs-post field contrast reusing it collapses to zero. Field 'post' provenance for example files is `UNRESOLVED`.

## 1. Spectral shape (Welch PSD, 10 kHz, nperseg 8192, contact-averaged)

PSD is MEASURED per area per trial (Welch of the recorded field proxy). Peak frequency in 1–100 Hz (dominant spectral feature):

| trial | V1 | V4 | FEF | PFC | tag |
|---|---|---|---|---|---|
| pre_AAAB | 1.2 | 1.2 | 1.2 | 1.2 | DERIVED |
| pre_AXAB | 1.2 | 1.2 | 1.2 | 1.2 | DERIVED |
| ex_AAAB | 1.2 | 1.2 | 1.2 | 1.2 | DERIVED |
| ex_AXAB | 1.2 | 1.2 | 1.2 | 1.2 | DERIVED |
| ex_BBBA | 1.2 | 1.2 | 1.2 | 1.2 | DERIVED |

## 2. Band power (frozen bands [4,8,14,30,50,80] → θ/α/β/γ_lo/γ_hi) [MODEL_ASSUMPTION]

Absolute band power (a.u.²) per area, trial `ex_AAAB` (intact) and `ex_AXAB` (p2 omission):

**ex_AAAB**
| band | range | V1 | V4 | FEF | PFC | tag |
|---|---|---|---|---|---|---|
| theta | 4-8 Hz | 126.6 | 129.1 | 126.6 | 128.3 | MEASURED |
| alpha | 8-14 Hz | 61.47 | 62.25 | 60.93 | 61.2 | MEASURED |
| beta | 14-30 Hz | 47.48 | 48.49 | 47.33 | 46.77 | MEASURED |
| low-gamma | 30-50 Hz | 18.46 | 18.73 | 17.92 | 17.65 | MEASURED |
| high-gamma | 50-80 Hz | 10.47 | 10.49 | 9.988 | 9.849 | MEASURED |

**ex_AXAB**
| band | range | V1 | V4 | FEF | PFC | tag |
|---|---|---|---|---|---|---|
| theta | 4-8 Hz | 92.4 | 93.72 | 92.41 | 93.33 | MEASURED |
| alpha | 8-14 Hz | 46.36 | 46.95 | 45.99 | 45.63 | MEASURED |
| beta | 14-30 Hz | 34.11 | 34.96 | 34.05 | 34.12 | MEASURED |
| low-gamma | 30-50 Hz | 14.41 | 14.83 | 13.93 | 13.59 | MEASURED |
| high-gamma | 50-80 Hz | 8.077 | 8.104 | 7.73 | 7.501 | MEASURED |

Power is MEASURED (arbitrary units). Because amplitude is uncalibrated, absolute power has no physical meaning; only within-trial relative comparisons and log-ratios are interpretable. Fractional allocation of power across bands (DERIVED) and omission-vs-intact log-ratios follow.

## 3. Omission vs intact log band ratio (ln power_omission/power_intact), AXAB/AAAB

| band | V1 | V4 | FEF | PFC | tag |
|---|---|---|---|---|---|
| theta | -0.315 | -0.321 | -0.315 | -0.319 | DERIVED |
| alpha | -0.282 | -0.282 | -0.281 | -0.294 | DERIVED |
| beta | -0.331 | -0.327 | -0.329 | -0.315 | DERIVED |
| low-gamma | -0.248 | -0.233 | -0.252 | -0.261 | DERIVED |
| high-gamma | -0.260 | -0.258 | -0.256 | -0.272 | DERIVED |

Single-replicate (n=1), descriptive only. No inferential threshold applied (design requires n≥4); polarity `UNRESOLVED`.

## 4. T4-relevant low-gamma (30–50 Hz) p2-slot log ratio [DERIVED]

Frozen design field_low_gamma band = 30–50 Hz [MODEL_ASSUMPTION, design doc:54]; window relative to p2 onset (1031 ms): slot [0,531], baseline [-250,-50]. `frontal_minus_V1` = mean(FEF,PFC) − V1.

| trial set | window | area | log ratio | tag |
|---|---|---|---|---|
| pre | slot | V1 | -0.883 | DERIVED |
| pre | baseline | V1 | -0.129 | DERIVED |
| pre | slot | V4 | -0.868 | DERIVED |
| pre | baseline | V4 | 0.022 | DERIVED |
| pre | slot | FEF | -0.675 | DERIVED |
| pre | baseline | FEF | -0.153 | DERIVED |
| pre | slot | PFC | -0.565 | DERIVED |
| pre | baseline | PFC | 0.030 | DERIVED |
| pre | slot | frontal_minus_V1 | 0.263 | DERIVED |
| pre | baseline | frontal_minus_V1 | 0.068 | DERIVED |
| ex | slot | V1 | -0.793 | DERIVED |
| ex | baseline | V1 | 0.614 | DERIVED |
| ex | slot | V4 | -0.827 | DERIVED |
| ex | baseline | V4 | 0.618 | DERIVED |
| ex | slot | FEF | -0.767 | DERIVED |
| ex | baseline | FEF | 0.458 | DERIVED |
| ex | slot | PFC | -0.594 | DERIVED |
| ex | baseline | PFC | 0.540 | DERIVED |
| ex | slot | frontal_minus_V1 | 0.113 | DERIVED |
| ex | baseline | frontal_minus_V1 | -0.116 | DERIVED |

Note: the companion `E_A_analysis.json` computed this T4 proxy with a **20–50 Hz** band [MEASURED from that artifact], differing from the frozen 30–50 Hz band [MODEL_ASSUMPTION]. The two conventions are not directly comparable; the band convention discrepancy is flagged `UNRESOLVED` at the provenance level.

## 5. TFR — epoch-resolved low-gamma (30–50 Hz) band power time course (single-trial)

STFT (nperseg 512 ms, 256 ms hop), mean |S|² in 30–50 Hz, contact-averaged, per epoch. Single-trial estimates; high variance.

**AAAB (intact)**
| area | p1 | d1 | p2 | d2 | p3 | d3 | p4 | d4 | rw | tag |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 5.82 | 1.89 | 1.69 | 2.08 | 1.81 | 2.02 | 1.96 | 0.123 | 1.39 | DERIVED |
| V4 | 5.83 | 1.74 | 1.97 | 2.26 | 1.68 | 2.04 | 1.9 | 0.135 | 1.4 | DERIVED |
| FEF | 5.66 | 1.7 | 1.9 | 2.19 | 1.6 | 1.99 | 1.78 | 0.117 | 1.29 | DERIVED |
| PFC | 5.58 | 1.59 | 2.12 | 2.15 | 1.6 | 2.15 | 1.59 | 0.098 | 1.2 | DERIVED |

**AXAB (p2 omission)**
| area | p1 | d1 | p2 | d2 | p3 | d3 | p4 | d4 | rw | tag |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 5.95 | 0.111 | 0.102 | 2.03 | 2 | 2.12 | 1.99 | 0.114 | 1.39 | DERIVED |
| V4 | 6.08 | 0.112 | 0.0971 | 2.09 | 1.98 | 2.16 | 2.03 | 0.12 | 1.38 | DERIVED |
| FEF | 5.88 | 0.0931 | 0.0867 | 1.99 | 1.86 | 2.04 | 1.86 | 0.108 | 1.3 | DERIVED |
| PFC | 5.9 | 0.0958 | 0.0788 | 2.13 | 1.59 | 2.02 | 1.68 | 0.103 | 1.21 | DERIVED |

The AXAB trial shows reduced low-gamma power across the p2 slot relative to AAAB (consistent with the omission slot), but this is a single-trial, descriptive observation [INFERENCE]; no TFR significance is claimed.

## 6. Rate ↔ field coupling [DERIVED, n=3, descriptive]

Per-area Pearson correlation between trial-mean firing rate (spikes_example counts → Hz) and band power (field example) across the 3 matched example trials (AAAB, AXAB, BBBA). Caveat: the AAAB field is a pre-exposure duplicate (§header), so this is a descriptive pairing, not a clean post-exposure coupling.

| band | V1 | V4 | FEF | PFC | pooled(12) | tag |
|---|---|---|---|---|---|---|
| theta | +1.00 | +1.00 | +1.00 | +1.00 | +0.79 | DERIVED |
| alpha | +0.99 | +0.99 | +0.99 | +0.99 | +0.74 | DERIVED |
| beta | +1.00 | +1.00 | +1.00 | +1.00 | +0.76 | DERIVED |
| low-gamma | +0.95 | +0.95 | +0.93 | +0.95 | +0.64 | DERIVED |
| high-gamma | +0.98 | +0.98 | +0.97 | +0.97 | +0.64 | DERIVED |

With n=3 per area, correlations are not statistically resolvable; the design declares n<4 as `UNRESOLVED`/underpowered [MODEL_ASSUMPTION]. The T5 design estimand (trial-resolved gamma×rate covariance across replicates) is `UNRESOLVED` from artifacts because only single-trial examples are retained; the pooled n=12 is descriptive only. Pre-exposure pairing yields n=2 (`UNRESOLVED`).

## 7. Cross-area field correlation (contact-averaged, per trial)

| trial | V1-V4 | V4-FEF | FEF-PFC | V1-PFC | tag |
|---|---|---|---|---|---|
| ex_AAAB | 1.00 | 1.00 | 1.00 | 1.00 | DERIVED |
| ex_AXAB | 1.00 | 1.00 | 1.00 | 1.00 | DERIVED |

Cross-area correlations are high across the hierarchy in both trials [DERIVED]; this reflects the shared-laminar-contact proxy construction (area-attributed partitions of one shared field), so high correlation is expected and is NOT evidence for empirical cross-area coherence [INFERENCE]. T7 (between-area lead/lag) is `UNRESOLVED` from artifacts.

## 8. Laminar depth profile (low-gamma 30–50 Hz, V1, ex_AAAB)

| depth bin (4 contacts each) | low-gamma power (a.u.², Welch) |
|---|---|
| bin 1 (contacts 0-3) | 9.632 |
| bin 2 (contacts 4-7) | 19.25 |
| bin 3 (contacts 8-11) | 29.19 |
| bin 4 (contacts 12-15) | 15.78 |

Depth profile is MEASURED (Welch band power, consistent with §2 scaling; relative, arbitrary units); no physical amplitude calibration, so the laminar gradient is descriptive only.

## 9. Summary of open/UNRESOLVED items

- Physical amplitude calibration of the field proxy: `UNRESOLVED` (uncalibrated).
- Post-exposure field provenance of example files: `UNRESOLVED` (ex_AAAB == pre_0000 duplicate).
- Field-level delta_exposure for the intact condition: `UNRESOLVED` (post AAAB field not a distinct recording).
- T4/T5 multi-replicate spectral contrasts: `UNRESOLVED` (single-trial only).

