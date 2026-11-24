# Run Report — example_run_C019

- config_hash `be9b96ab679c9802` hp_hash `f327f9d2ad64cc88` run_hash `05893c398d0a` seed 0 dt 0.1 ms duration 2000 ms
- model C019 freeze 65b302e8c7cdceb5 (builder.py:62,90,395) — E mixture M2 RS70/CH20/E_FS10, VIP b0.20, SST b0.21, PV 1.7, lognormal CV1.5, spatial σ0.08 max25, delays [20,80,120]
- numerical_valid True state_identity `6aed39f5b241d806` completion {"all": true, "note": "spontaneous baseline single-trial, terminated_by_schedule true"}
- phases ['spontaneous'] — spontaneous 2000 ms baseline when duration≤5 s else lifecycle
- arrays: `arrays/spikes.npz` (20000, 400) `arrays/V_m.npz` `arrays/field.npz` (proxy_readout, jaxfne/fields/proxy.py:192) `arrays/H_t.npz` when HDP available (guard jaxfne/_model_simulate.py:280)
- figures: `figures/*.html` per tab + `report.html` fixed tabs [Overview, Raster, Rates, Membrane, E/I, H/Θ, Spectra, Field, Stimulus, Connectivity, Diagnostics, Provenance]
- provenance table rows 12 — claim→estimator→array→hash (see Provenance tab)

## Diagnostics (same spikes array)
- CV_ISI mean 0.431 max 2.811 frac>0.5 0.273 Fano 1.035 ρ 0.661 rate 10.0 Hz

Generated via jomission/visualization/run_report.py:run_report — consumes same generated-owner arrays as analyses (spikes, V_m, field, H, Theta, q) not recomputed prettier alternatives.
Wall time 3.1 s.
