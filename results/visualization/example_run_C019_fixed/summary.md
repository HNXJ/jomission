# Run Report — example_run_C019

- config_hash `be9b96ab679c9802` hp_hash `f327f9d2ad64cc88` run_hash `b9494d4d2604` seed 0 dt 0.1 ms duration 500 ms
- model C019 freeze 65b302e8c7cdceb5 (builder.py:62,90,395) — E mixture M2 RS70/CH20/E_FS10, VIP b0.20, SST b0.21, PV 1.7, lognormal CV1.5, spatial σ0.08 max25, delays [20,80,120]
- numerical_valid True state_identity `5bb3194262c0f9a6` completion {"all": true, "note": "spontaneous baseline single-trial, terminated_by_schedule true"}
- phases ['spontaneous'] — spontaneous 2000 ms baseline when duration≤5 s else lifecycle
- arrays: `arrays/spikes.npz` (5000, 400) `arrays/V_m.npz` `arrays/field.npz` (proxy_readout, jaxfne/fields/proxy.py:192) `arrays/H_t.npz` when HDP available (guard jaxfne/_model_simulate.py:280)
- figures: `figures/*.html` per tab + `report.html` fixed tabs [Overview, Raster, Rates, Membrane, E/I, H/Θ, Spectra, Field, Stimulus, Connectivity, Diagnostics, Provenance]
- provenance table rows 15 — claim→estimator→array→hash (see Provenance tab)

## Diagnostics (same spikes array)
- CV_ISI mean 0.286 max 2.034 frac>0.5 0.169 Fano 0.946 ρ 0.666 rate 11.8 Hz

Generated via jomission/visualization/run_report.py:run_report — consumes same generated-owner arrays as analyses (spikes, V_m, field, H, Theta, q) not recomputed prettier alternatives.
Wall time 1.1 s.
