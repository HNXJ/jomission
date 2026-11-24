# Run Report — example_run_C021_transfer

- config_hash `a51711101576c7c3` hp_hash `f327f9d2ad64cc88` run_hash `05893c398d0a` seed 0 dt 0.1 ms duration 2000 ms
- model C019 freeze 65b302e8c7cdceb5 (builder.py:62,90,395) — E mixture M2 RS70/CH20/E_FS10, VIP b0.20, SST b0.21, PV 1.7, lognormal CV1.5, spatial σ0.08 max25, delays [20,80,120]
- numerical_valid True state_identity `59ce40ca53307787` completion {"all": true, "note": "C021 transfer spontaneous baseline; Transfer Function shows 6 stages V1 L4E\u2192L23E chain, machine NOT OBSERVABLE for current when blocked"}
- phases ['spontaneous'] — spontaneous 2000 ms baseline when duration≤5 s else lifecycle
- arrays: `arrays/spikes.npz` (20000, 400) `arrays/V_m.npz` `arrays/field.npz` (proxy_readout, jaxfne/fields/proxy.py:192) `arrays/H_t.npz` when HDP available (guard jaxfne/_model_simulate.py:280)
- figures: `figures/*.html` per tab + `report.html` fixed tabs [Overview, Raster, Rates, Membrane, E/I, H/Θ, Spectra, Field, Stimulus, Connectivity, Transfer Function, Transfer Summary, Diagnostics, Provenance]
- provenance table rows 21 — claim→estimator→array→hash (see Provenance tab)

## Diagnostics (same spikes array)
- CV_ISI mean 0.429 max 2.811 frac>0.5 0.273 Fano 1.035 ρ 0.661 rate 10.0 Hz

Generated via jomission/visualization/run_report.py:run_report — consumes same generated-owner arrays as analyses (spikes, V_m, field, H, Theta, q) not recomputed prettier alternatives.
Wall time 22.0 s.
