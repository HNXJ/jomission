"""Machine-readable T1–T7 comparison matrix — frozen before production.

Schema specified by hypothesis plan; each target has estimand, matching, windows,
and falsification rule. Do not expand opportunistically after seeing results.
"""

from __future__ import annotations

COMPARISON_MATRIX: dict = {
    "matrix_version": "jomission_comparison_matrix.v0.1.0",
    "delta_exposure": "Δ_exposure = Y_omission^{post} - Y_omission^{pre}  (post after ≥1000s AAAB/BBBA, pre naive)",
    "pooling_rule": "DO NOT pool p2/p3/p4 until position dependence explicitly tested (Q11)",
    "windows": {
        "omission_local": (-1000.0, 1000.0),
        "omission_baseline": (-250.0, -50.0),
        "omission_slot": (0.0, 531.0),
        "post_omission": (531.0, 1000.0),
        "trial_baseline": (-500.0, 0.0),
    },
    "targets": [
        {
            "id": "T1",
            "label": "sparse omission-linked spiking",
            "estimand": "fraction_significant_units",
            "contrast": "omission (AXAB/BXBA/RXRR etc) vs intact (AAAB/BBBA/RRRR) per position",
            "matching": "within-replicate, same seed, intact vs omission at same p position",
            "test": "per-unit rate comparison (omission slot [0,531] vs baseline [-250,-50]), FDR-corrected",
            "threshold": "<0.10 significant",
            "falsification": "if fraction >=0.10 or 0, record as is — do not tune to force sparsity",
            "null": "shuffled_timing control: omission→stimulus",
        },
        {
            "id": "T2",
            "label": "higher-order bias",
            "estimand": "area enrichment of T1+ units",
            "contrast": "FEF/PFC vs V1/V4 among T1+",
            "matching": "same T1 definition",
            "test": "chi2 / enrichment (observed vs expected by area size)",
            "threshold": "FEF/PFC enrichment >1.5x",
            "falsification": "if V1 enriched, report; do not rewire to force frontal bias",
        },
        {
            "id": "T3",
            "label": "weak V1 population omission spiking",
            "estimand": "V1 population PSTH omission vs intact",
            "contrast": "V1 mean rate [0,531] omission vs intact",
            "matching": "within-replicate, V1 all units",
            "test": "paired t per ms + effect size (Cohen d)",
            "threshold": "d <0.2 or non-significant",
            "falsification": "if V1 shows strong burst, report; not null",
        },
        {
            "id": "T4",
            "label": "frontal low-gamma omission effect",
            "estimand": "LFP-like band power (20-50 Hz) omission vs intact",
            "contrast": "FEF/PFC vs V1, corrected",
            "matching": "same trial, virtual contacts laminar",
            "test": "bandpower_jax, permutation / cluster-corrected",
            "threshold": "frontal p<0.05 corrected, V1 weaker",
            "field_claim": "proxy_readout, physical_amplitude_calibrated=False",
        },
        {
            "id": "T5",
            "label": "gamma-rate coupling",
            "estimand": "trial gamma power vs spike rate correlation",
            "contrast": "per unit, across trials",
            "matching": "same unit, same area",
            "test": "Pearson r",
            "threshold": "mean r >0",
        },
        {
            "id": "T6",
            "label": "weaker field coupling for omission-selective units",
            "estimand": "spike-field coupling (SFC / gamma coupling) T1+ vs T1-",
            "contrast": "T1+ vs T1- within same area",
            "matching": "same area, same recording geometry",
            "test": "compare coupling distributions",
            "threshold": "T1+ < T1-",
        },
        {
            "id": "T7",
            "label": "absence of strong fixed between-area lead/lag",
            "estimand": "cross-area field/rate cross-correlation peak lag",
            "contrast": "all area pairs, omission and intact",
            "matching": "same trials",
            "test": "peak lag distribution, test for consistent non-zero lag",
            "threshold": "no consistent fixed lag (p>0.05)",
            "falsification": "if strong lead/lag found, report as positive — this T expects null",
        },
    ],
    "language_rule": "lfp_proxy/csd_proxy remain proxy; never promote to physical LFP/CSD in manuscript",
    "freeze_note": "Frozen before production exposure; any expansion requires new version and review",
}
