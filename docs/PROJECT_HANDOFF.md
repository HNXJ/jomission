# Project Handoff: Recurrent Scaling System Identification & Phase C Closure

**Date**: 2026-09-02  
**Publication Readiness Score**: **42/100**  
*(Definition: Baseline computational/observability harness verified, repository synchronization clean, unitary and recurrent transfer functions directly identified; but primary neurocomputational benchmarks B1-B3 fail, B6 deferred, and sequence context memory mechanism remains missing).*

---

## 1. Verified Scientific State

| Dimension | State | Scope & Empirical Receipts |
|---|---|---|
| **B1 (Substrate Recruitment)** | `FAIL` | Coverage 0.57 < 0.60; VIP cells remain substantially below useful operating regime ($0.39\text{--}0.83\,\text{Hz}$). |
| **B2 (Firing Regularity / Correlation)** | `FAIL` | $CV_{\rm ISI} \approx 0.31\text{--}0.36 < 0.50$ persists across all $g_R \le 32$; late pairwise correlation $\rho \approx 0.00$ passing. |
| **B3 (E/I Contribution Ratio)** | `FAIL` | $E_{\rm frac} \approx 0.94 > 0.60$ under canonical model; recurrent current remains $<3.1\%$ of external drive. |
| **B6 (Cross-Area Modulation)** | `DEFERRED` | Frozen as deferred until local laminar substrate B1–B3 qualifies. |
| **Sequence Context Memory ($M$)** | `MISSING` | Latent structural state from HDP lacks temporal order specificity ($AAAB \approx BAAA$ at $18.6\,\text{s}$ and $50\,\text{ms}$). Biologically defensible mechanism spanning $531\,\text{ms}$ slot remains missing. |
| **Theta Structural Acquisition** | `PASS` | Local recurrent remodeling dominant: $G_{\rm rec} = 1.331$, $D_{2,\rm rec} = 0.216$. |
| **Theta Boundedness** | `PASS_THROUGH_100S` | Weight trajectory stationary by $60\text{--}100\,\text{s}$ ($\Delta_{60\to100} = 0.00110$). |
| **Theta Functional Consequence** | `SPARSE/SMALL` | Large latent synaptic memory formed with negligible shift in mean firing operating point ($\Delta r_{\rm global} \sim -0.3\%$). |
| **Theta Order Specificity** | `ESSENTIALLY_ABSENT` | $\cos(\Delta w_{AAAB}, \Delta w_{BAAA}) = 0.999981$; relative vector divergence $<0.62\%$. |
| **HALC Architecture** | `FROZEN` | Layer-resolved multi-timescale architecture specification frozen. |
| **HALC Parameters / Implementation** | `UNRESOLVED / NOT_AUTHORIZED` | Prohibited from implementation to prevent compensating for unqualified substrate. |
| **Recurrent Decoupling** | `OBSERVED` | Canonical $I_{\rm syn} < 0.15\%$ of $I_{\rm ext}$; $\frac{I_E}{I_{\rm ext}} \approx 10^{-3}$ confirmed not an artifact of E/I cancellation or measurement bug. |
| **Unitary Synaptic Transfer** | `DIRECTLY_MEASURED` | Directly simulated in nonlinear Izhikevich dynamics: $E \to E$ uEPSP $= +13.6\,\mu\text{V}$, $PV \to E$ uIPSP $= -24.2\,\mu\text{V}$ at canonical drive. |
| **Topology Normalization** | `K^-1_MEAN_CONSERVING` | Linear scaling $w \propto K^{-1}$ strictly conserves mean recurrent input currents $\mu(I_E), \mu(\|I_I\|)$ invariant across $K_{\rm in} \in [10, 100]$. |
| **Recurrent Gain vs $CV_{\rm ISI}$** | `REJECTED_AS_SUFFICIENT_CAUSE` | Increasing recurrent weight scale by $32\times$ ($g_R = 1 \to 32$) amplifies $R_E$ from $0.10\% \to 3.03\%$ and uEPSP to $+0.44\,\text{mV}$, but $CV_{\rm ISI}$ remains pinned at $0.316 \to 0.357 \ll 0.50$. |
| **Canonical Network Parameters** | `UNCHANGED` | Zero mutations applied to canonical scientific parameters in `jomission` or `jaxfne`. |

---

## 2. Latest Phase C Empirical Receipts

Receipts from [`results/phase_c_recurrent_gain_sweep_results.json`](results/phase_c_recurrent_gain_sweep_results.json):

| $g_R$ | $R_E = \frac{\langle I_E \rangle}{\langle I_{\rm ext} \rangle}$ | $R_I = \frac{\langle \|I_I\| \rangle}{\langle I_{\rm ext} \rangle}$ | Firing Rate | $r_E$ | $r_{\rm PV}$ | $r_{\rm SST}$ | $r_{\rm VIP}$ | $CV_{\rm ISI}$ | Median Fano | Late $\rho$ | uEPSP ($E \to E$) | uIPSP ($PV \to E$) | Stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **$1.0$** | $0.10\%$ | $0.07\%$ | $4.93\,\text{Hz}$ | $5.74$ | $6.41$ | $5.33$ | $0.39$ | $0.3162$ | $0.60$ | $-0.0028$ | $+0.0057\,\text{mV}$ | $-0.0324\,\text{mV}$ | Stable |
| **$2.0$** | $0.17\%$ | $0.13\%$ | $4.92\,\text{Hz}$ | $5.71$ | $6.47$ | $5.40$ | $0.38$ | $0.3141$ | $0.60$ | $+0.0084$ | $+0.0195\,\text{mV}$ | $-0.0563\,\text{mV}$ | Stable |
| **$4.0$** | $0.32\%$ | $0.26\%$ | $4.94\,\text{Hz}$ | $5.73$ | $6.53$ | $5.36$ | $0.38$ | $0.3175$ | $0.60$ | $-0.0041$ | $+0.0471\,\text{mV}$ | $-0.1040\,\text{mV}$ | Stable |
| **$8.0$** | $0.63\%$ | $0.55\%$ | $5.07\,\text{Hz}$ | $5.88$ | $6.81$ | $5.44$ | $0.38$ | $0.3267$ | $0.60$ | $+0.0076$ | $+0.1026\,\text{mV}$ | $-0.1982\,\text{mV}$ | Stable |
| **$16.0$** | $1.31\%$ | $1.15\%$ | $5.32\,\text{Hz}$ | $6.18$ | $7.22$ | $5.50$ | $0.42$ | $0.3487$ | $0.63$ | $+0.0106$ | $+0.2147\,\text{mV}$ | $-0.3825\,\text{mV}$ | Stable |
| **$32.0$** | $3.03\%$ | $2.86\%$ | $6.29\,\text{Hz}$ | $7.17$ | $9.21$ | $5.89$ | $0.83$ | **$0.3566$** | **$0.67$** | $+0.0021$ | **$+0.4437\,\text{mV}$** | **$-0.7358\,\text{mV}$** | Stable |

---

## 3. Critical Negative Evidence Preserved

1. **Weak Recurrence $\ne$ Low $CV_{\rm ISI}$**:
   Even when recurrent synaptic weights are scaled by $32\times$, driving $E \to E$ uEPSP to $+0.44\,\text{mV}$ and $PV \to E$ uIPSP to $-0.74\,\text{mV}$, $CV_{\rm ISI}$ remains tightly clustered in $[0.31, 0.36]$. Therefore, weak recurrent coupling does not explain the clock-like firing irregularity defect.
2. **VIP Silence is Intrinsic, Not Fixed by Homogeneous Recurrence**:
   VIP firing rates remain $<0.85\,\text{Hz}$ even under $g_R = 32$, because VIP sits $-1.559$ current units below rheobase. Recurrent excitation accounts for only $+0.08$ current units into VIP at $g_R = 32$.
3. **Startup Transient $\rho$ vs Steady-State $\rho$**:
   Full-window pairwise correlation $\rho \approx 0.39$ is entirely driven by the initial $0\text{--}500\,\text{ms}$ rest release transient. In the steady-state window ($500\text{--}2000\,\text{ms}$), baseline pairwise correlation is passing ($\rho = -0.0008 \approx 0.00$).

---

## 4. Current Critical Path & Next Experimental Actions

### B2 (Firing Irregularity)
- **Observed**: Low $CV_{\rm ISI}$ persists through $g_R = 32$.
- **Observed**: Late $\rho$ is near zero in steady state.
- **Inferred Candidate**: High-rate external Poisson shot noise ($\lambda = 2000\,\text{Hz}, \text{amp} = 2.0$) acts as a continuous low-pass integrator, producing clock-like first-passage threshold crossings.
- **Required Next Test**: Controlled input **mean-versus-variance system identification** (vary Poisson rate $\lambda$ and amplitude $A$ while preserving total mean drive $\mu = \lambda A \tau$, measuring $CV_{\rm ISI}(A, \lambda)$). Do not mutate canonical drive yet.

### B1 (VIP Operating Point)
- **Observed**: VIP operating current sits $-1.56$ below rheobase.
- **Required Question**: Determine why the generic substrate positions VIP at this subthreshold coordinate before intervening.
- **Constraint**: Do not add arbitrary VIP tonic drive merely to force B1 to pass.

### B3 (E/I Balance)
- Reassess after input drive and substrate system identification are complete. Do not adopt $g_R = 32$ merely because recurrence is stronger.

### B6 (Cross-Area Modulation)
- Remains deferred until local laminar substrate B1–B3 qualifies.

### Sequence Context Memory
- Current HDP is order-insensitive ($AAAB \approx BAAA$).
- A biologically defensible order-sensitive state spanning the $531\,\text{ms}$ inter-stimulus slot remains missing.
- Invariant: Do not design or tune memory mechanisms to compensate for an unqualified substrate.

---

## 5. Repository Provenance & Verification

- **Jomission**:
  - URL: `https://github.com/HNXJ/jomission`
  - Tracking: `main == origin/main`
  - Commit: `664b679` (clean working tree)
- **JaxFNE**:
  - URL: `https://github.com/HNXJ/jaxfne`
  - Tracking: `dev == origin/dev`
  - Commit: `ad88756` (clean working tree)
- **Harness Verification**:
  - `AGENTS.md`: Established at repository root.
  - Test: `tests/test_agents_policy.py` PASS (2/2).
  - Visualization: Plotly reference gallery infrastructure operational with audited captions.
