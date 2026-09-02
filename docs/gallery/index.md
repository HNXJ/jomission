# Jomission Interactive Reference Gallery

Welcome to the interactive reference gallery for **Jomission** and **JaxFNE**. These standalone, publication-grade Plotly visualizations provide full interactive exploration of network architecture, receptive field mapping, population dynamics, spectrolaminar profiles, longitudinal plasticity, and qualification diagnostics.

---

## Flagship Interactive Figures

### 1. Flagship 3D Network Explorer
<span class="badge badge-success">OBSERVED</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/network_3d.html)

Interactive 3D connectome visualization of the 4-area Jomission canonical cortical hierarchy. Neurons are spatially mapped along the hierarchical stream (X-axis: V1, V4, FEF, PFC), column transverse space (Y-axis), and cortical laminar depth (Z-axis: L1 to L6). Cell classes are color-coded: Excitatory (Cyan), PV (Crimson), SST (Amber), and VIP (Violet). Inter-areal Feedforward projections (Green: L2/3 → L4) and Feedback projections (Orange: L6 → L1/L5) are interactively filterable alongside recurrent intra-area circuits (Slate).

<iframe src="../_static/plotly/network_3d.html" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

### 2. Visual Field & RF Architecture
<span class="badge badge-info">DERIVED</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/visual_field_mapping.html)

Interactive retinotopic visual field mapping of the Jomission input layer. Left: 2D stimulus space spanning an 8° visual angle at 0.25°/px. Gaussian stimulus blobs for Item A (centered at (8,8)) and Item B (centered at (24,24)) are separated by >12σ with Jaccard overlap = 0.0. Right: Topographic receptive field centers of all 100 V1 neurons. Activated units for Stimulus A (Cyan, 9 units) and Stimulus B (Crimson, 9 units) show complete spatial orthogonality, establishing rigorous sensory input boundaries.

<iframe src="../_static/plotly/visual_field_mapping.html" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

### 3. Interactive Raster & Population Rates
<span class="badge badge-success">OBSERVED</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/raster_population.html)

Interactive population spiking and time-resolved rate dynamics during structured exposure. Top: Full raster plot of all 400 cortical neurons sorted by hierarchy (V1, V4, FEF, PFC) and laminar depth, color-coded by cell class (E: Cyan, PV: Crimson, SST: Amber, VIP: Violet). Shaded vertical regions mark stimulus presentation epochs (p1, p2, p3 in blue; p4 in red). Bottom: Binned population firing rate trajectories (10 ms sliding window) showing fast, transient recruitment of PV interneurons alongside sustained excitatory pyramidal drive.

<iframe src="../_static/plotly/raster_population.html" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

### 4. Spectral Response & Time-Frequency
<span class="badge badge-info">DERIVED</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/spectral_response.html)

Interactive spectral decomposition and time-frequency dynamics of the cortical hierarchy. Left: Area-resolved Power Spectral Density (Welch PSD estimate) for V1 (Blue), V4 (Green), FEF (Orange), and PFC (Purple), highlighting canonical physiological bands (Theta 4-8 Hz, Alpha 8-12 Hz, Beta 15-30 Hz, Gamma 30-80 Hz). Right: Spectrogram of V1 population potential across the 2000 ms trial displaying spectral power variations across sensory drive slots.

<iframe src="../_static/plotly/spectral_response.html" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

### 5. Plasticity Trajectory & Circuit Matrix
<span class="badge badge-success">OBSERVED</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/plasticity_trajectory.html)

Multi-scale empirical characterization of HDP synaptic plasticity. Top-Left: Weight gain G(t) across 100 s showing fast 2 s overshoot followed by asymptotic relaxation: recurrent circuits stay permanently remodeled (+33.1%), whereas feedforward (+1.4%) and feedback (+3.1%) relax back to baseline. Top-Right: Normalized structural displacement D_2(t) confirming recurrent dominance. Bottom-Left: 4×4 source-target remodeling matrix at t = 100 s, revealing profound potentiation of SST-associated recurrent weight remodeling (model specific) (SST→E +92.6%, SST→PV +72.2%) contrasting with invariant VIP disinhibition (VIP→SST +2.0%). Bottom-Right: Empirical temporal memory kernel D_order(ΔT) demonstrating that the learning rule is fundamentally order-insensitive across all intervals (D_order = 0.29% at ΔT = 50 ms).

<iframe src="../_static/plotly/plasticity_trajectory.html" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

### 6. B1-B3 Qualification & Root Causes
<span class="badge badge-success">OBSERVED</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/b1_b2_b3_dashboard.html)

Root-cause diagnostic decomposition of cortical qualification failures (B1, B2, B3). Top-Left: High spike correlation (ρ) in B2 is resolved as a startup transient artifact (Early window ρ = 0.530); in the late steady-state window (500-2000 ms), ρ is naturally zero (mean ρ = -0.0008), and subthreshold phase jitter immediately lowers full-window ρ to 0.189 (<0.20 ceiling). Top-Right: All populations operate subthreshold (I_native < I_rh), but VIP suffers a severe -1.56 current unit deficit below rheobase (I_native = 2.19 vs I_rh = 3.75). Bottom-Left: Subthreshold positioning collapses VIP firing rate to 0.39 Hz (23.4% silence). Bottom-Right: Recurrent synaptic currents (I_E, I_I < 0.005) represent <0.15% of external drive (I_ext ≈ 3.41), proving that the spontaneous cortical baseline is currently operating in a recurrently decoupled regime.

<iframe src="../_static/plotly/b1_b2_b3_dashboard.html" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

