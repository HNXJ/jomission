# Details: t8_rflat_n4000_s3_T8

Layout `e2_3unit_n4000_dens_k1_kd1_sst15_rhof0p794`. E2 stack, first pass (unsealed; owner rulings 2026-11-09).

Units: voltage mV, time ms, capacitance pF, conductance nS, current pA, rate Hz (source: caminos_e0.py docstring (JaxFNE units)).

Source labels: `PORT_SPEC (Tahvili, Vinck, di Volo)` (values from caminos_e0 or the port spec), `agent choice` (set by a modifier or this project), or a named derivation.

## Modules

| module | depth rel. L4 | rho | N_E | N_PV | N_SOM | N |
|---|---|---|---|---|---|---|
| S | −0.4 | 0.794 | 5,785 | 605 | 480 | 6,870 |
| M | +0.0 | 0.794 | 3,668 | 384 | 304 | 4,356 |
| D | +0.3 | 0.794 | 2,547 | 266 | 212 | 3,025 |

Total N = 14,251.

## Edges per rule

| rule | pre to post | module or pair | Q (nS) | weight (nS) | mechanism | edges |
|---|---|---|---|---|---|---|
| e_e_S | E to E | S | 2 | 2 | ampa | 4,684,103 |
| e_pv_S | E to PV | S | 3 | 3 | ampa | 490,623 |
| e_sst_S | E to SOM | S | 3 | 3 | ampa | 388,871 |
| pv_e_S | PV to E | S | 5 | 5 | gaba | 491,086 |
| pv_pv_S | PV to PV | S | 5 | 5 | gaba | 51,063 |
| sst_e_S | SOM to E | S | 5 | 2.667 | gaba_sst (15 ms) | 388,285 |
| sst_pv_S | SOM to PV | S | 5 | 5 | gaba | 40,634 |
| e_e_M | E to E | M | 2 | 2 | ampa | 1,882,472 |
| e_pv_M | E to PV | M | 3 | 3 | ampa | 197,271 |
| e_sst_M | E to SOM | M | 3 | 3 | ampa | 156,220 |
| pv_e_M | PV to E | M | 5 | 5 | gaba | 197,064 |
| pv_pv_M | PV to PV | M | 5 | 5 | gaba | 20,699 |
| sst_e_M | SOM to E | M | 5 | 2.667 | gaba_sst (15 ms) | 156,128 |
| sst_pv_M | SOM to PV | M | 5 | 5 | gaba | 16,444 |
| e_e_D | E to E | D | 2 | 2 | ampa | 907,557 |
| e_pv_D | E to PV | D | 3 | 3 | ampa | 94,819 |
| e_sst_D | E to SOM | D | 3 | 3 | ampa | 75,081 |
| pv_e_D | PV to E | D | 5 | 5 | gaba | 95,440 |
| pv_pv_D | PV to PV | D | 5 | 5 | gaba | 9,832 |
| sst_e_D | SOM to E | D | 5 | 2.667 | gaba_sst (15 ms) | 75,693 |
| sst_pv_D | SOM to PV | D | 5 | 5 | gaba | 7,914 |
| x_S_M_e_e | E to E | S->M | 2 | 2 | ampa | 292,467 |
| x_S_M_e_pv | E to PV | S->M | 3 | 3 | ampa | 30,728 |
| x_S_M_e_som | E to SOM | S->M | 3 | 3 | ampa | 24,522 |
| x_M_S_e_e | E to E | M->S | 2 | 2 | ampa | 927,457 |
| x_M_S_e_pv | E to PV | M->S | 3 | 3 | ampa | 96,712 |
| x_M_S_e_som | E to SOM | M->S | 3 | 3 | ampa | 76,677 |
| x_M_D_e_e | E to E | M->D | 2 | 2 | ampa | 406,534 |
| x_M_D_e_pv | E to PV | M->D | 3 | 3 | ampa | 42,703 |
| x_M_D_e_som | E to SOM | M->D | 3 | 3 | ampa | 33,616 |
| x_D_M_e_e | E to E | D->M | 2 | 2 | ampa | 293,783 |
| x_D_M_e_pv | E to PV | D->M | 3 | 3 | ampa | 30,878 |
| x_D_M_e_som | E to SOM | D->M | 3 | 3 | ampa | 24,207 |

## Parameters

| group | parameter | value | unit | source |
|---|---|---|---|---|
| neuron | E VT_mV | -50 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV VT_mV | -50 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM VT_mV | -50 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E Delta_mV | 2 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV Delta_mV | 0.5 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM Delta_mV | 2 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E Tref_ms | 5 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV Tref_ms | 5 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM Tref_ms | 5 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E tauw_ms | 500 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV tauw_ms | 500 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM tauw_ms | 500 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E a_nS | 4 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV a_nS | 0 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM a_nS | 4 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E b_pA | 50 | pA | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV b_pA | 0 | pA | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM b_pA | 80 | pA | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E C_pF | 200 | pF | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV C_pF | 200 | pF | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM C_pF | 200 | pF | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E gL_nS | 10 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV gL_nS | 10 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM gL_nS | 10 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E EL_mV | -65 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV EL_mV | -65 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM EL_mV | -55 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | E Vreset_mV | -65 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | PV Vreset_mV | -65 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | SOM Vreset_mV | -65 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| neuron | spike registered at | V >= VT + 5 Delta | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | tau_delay | 2 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | tau_E (AMPA) | 2 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | tau_I (GABA) | 8 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | E_rev E (AMPA) | 0 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | E_rev I (GABA) | -80 | mV | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | Q E->E | 2 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | Q E->PV, E->SOM | 3 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | Q PV->E, PV->PV, SOM->E, SOM->PV | 5 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| synapse | Q PV->SOM, SOM->SOM | 0 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) (absent in the reference model) |
| drive | Q external -> E, PV | 2.5 | nS | PORT_SPEC (Tahvili, Vinck, di Volo) |
| drive | external sources per E and PV cell | 560 | count | PORT_SPEC (Tahvili, Vinck, di Volo) (round(p NE), p 0.07; per-cell sources are the port's assumption) |
| drive | external rate per source | 4 | Hz | PORT_SPEC (Tahvili, Vinck, di Volo) |
| drive | external drive to SOM | none |  | PORT_SPEC (Tahvili, Vinck, di Volo) |
| network | dt | 0.1 | ms | PORT_SPEC (Tahvili, Vinck, di Volo) (Brian2 Euler dt; caminos_e0 DT_MS) |
| network | N_E, N_PV, N_SOM (reference) | 8000, 750, 750 | count | PORT_SPEC (Tahvili, Vinck, di Volo) |
| network | within-module p (CAMINOS) | 0.07 |  | PORT_SPEC (Tahvili, Vinck, di Volo) |
| network | K_E = N_E p (E inputs per cell) | 560 | count | PORT_SPEC (Tahvili, Vinck, di Volo) (caminos_stack.K_E) |
| network | K_INTER (inter-module E inputs per cell per neighbour) | 80 | count | agent choice (caminos_stack.K_INTER; the code comment cites owner ruling 2026-11-09 for p 0.01 of 8000) |
| network | k (inter-module in-degree scale) | 1 |  | agent choice (set) |
| network | inter-module in-degree per target E cell | 80 from each neighbour; end module 160 from its one neighbour | count | caminos_stack.build_config; K_INTER agent choice |
| network | inter-module p | null: K_INTER based |  | run JSON p_inter; caminos_stack.build_config |
| network | within-module p (S) | 0.139997 |  | derived: K_E/N_E x (density/mean density)^kd (caminos_stack.local_p); kd agent choice |
| network | within-module p (M) | 0.14 |  | derived: K_E/N_E x (density/mean density)^kd (caminos_stack.local_p); kd agent choice |
| network | within-module p (D) | 0.140008 |  | derived: K_E/N_E x (density/mean density)^kd (caminos_stack.local_p); kd agent choice |
| synapse | SOM->E tau (S) | 15 | ms | agent choice (sst modifier) |
| synapse | SOM->E weight (S) | 2.66667 | nS | derived: Q_INH x TAU_I/tau (caminos_stack.build_config); tau agent choice |
| synapse | SOM->E tau (M) | 15 | ms | agent choice (sst modifier) |
| synapse | SOM->E weight (M) | 2.66667 | nS | derived: Q_INH x TAU_I/tau (caminos_stack.build_config); tau agent choice |
| synapse | SOM->E tau (D) | 15 | ms | agent choice (sst modifier) |
| synapse | SOM->E weight (D) | 2.66667 | nS | derived: Q_INH x TAU_I/tau (caminos_stack.build_config); tau agent choice |
| module | rho = N_SOM/N_PV (S) | 0.794 |  | owner ruling 2026-11-09 (caminos_stack.RHO comment); rho 1 = PORT_SPEC 750/750 |
| module | depth rel L4 (S) | -0.4 |  | agent choice (caminos_stack.layout) |
| module | rho = N_SOM/N_PV (M) | 0.794 |  | owner ruling 2026-11-09 (caminos_stack.RHO comment); rho 1 = PORT_SPEC 750/750 |
| module | depth rel L4 (M) | 0 |  | agent choice (caminos_stack.layout) |
| module | rho = N_SOM/N_PV (D) | 0.794 |  | owner ruling 2026-11-09 (caminos_stack.RHO comment); rho 1 = PORT_SPEC 750/750 |
| module | depth rel L4 (D) | 0.3 |  | agent choice (caminos_stack.layout) |
| module | MODULE_BINS (S) | -0.6 to -0.2 (9 bins) |  | agent choice (caminos_stack.MODULE_BINS on lansa_density_X.json bins) |
| module | MODULE_BINS (M) | -0.15 to 0.15 (7 bins) |  | agent choice (caminos_stack.MODULE_BINS on lansa_density_X.json bins) |
| module | MODULE_BINS (D) | 0.2 to 0.4 (5 bins) |  | agent choice (caminos_stack.MODULE_BINS on lansa_density_X.json bins) |
| modifier | n (mean E cells per module) | 4000 | count | agent choice (layout modifier n) |
| modifier | dens (E cells proportional to LANSA density) | on |  | agent choice (layout modifier dens) |
| modifier | kd (local in-degree exponent) | 1 |  | agent choice (layout modifier kd) |
| modifier | sst (SOM->E decay, all modules) | 15 | ms | agent choice (layout modifier sst) |
| analysis | sigma (probe Gaussian, relative depth) | 0.15 |  | agent choice (e2_motif_figure.SIGMA) |
| analysis | V_ref (current proxy) | -65 | mV | agent choice (e2_motif_figure.V_REF_MV) |
| analysis | discard | 2 | s | PLAN_LANSA_E2.md, C6 (2 s discarded) |
| analysis | multitaper half-bandwidth (smoothing) | 2 | Hz | e2_motif_figure.SMOOTH_HZ (as the published analysis) |
| analysis | RWS delay | 6 | ms | Mazzoni et al. 2015 (e2_motif_figure.py; plan section 0) |
| analysis | RWS alpha | 1.65 |  | Mazzoni et al. 2015 (e2_motif_figure.py; plan section 0) |
| analysis | bands alpha-beta / gamma | 10-30 / 50-150 | Hz | owner ruling 2026-11-09 (PLAN_LANSA_E2.md, Owner rulings) |
| analysis | probe contacts | 21 at -0.6 to 0.4 rel. L4 |  | LANSA bin centres (lansa_density_X.json axes; e2_motif_figure.CHANNELS) |

## Run

| key | value |
|---|---|
| layout | e2_3unit_n4000_dens_k1_kd1_sst15_rhof0p794 |
| experiment | E2 stack, first pass (unsealed; owner rulings 2026-11-09) |
| seed | 3 |
| T_s | 8 |
| dt_ms | 0.1 |
| jaxfne_version | 0.5.9 |
| jaxfne_source sha | not recorded |
| jax_devices | ['cpu:0'] |
| git_sha (run) | 2afef4da71b385277b9a404b3e91d7817b37c1da |
| git_dirty | True |
| wall_s | 331.391 |

## Notes

- Exploratory, unsealed E2 run (run JSON experiment). The recorded JaxFNE version is 0.5.9; PLAN_LANSA_E2.md section 0 and OPEN-37 name 0.5.8 as the project authority.
- Raster: sampled cells only (200 E, 50 PV and 50 SOM per module, a random sample by run_e2_stack.py).
- Module N is the sum of the recorded cell counts; the run JSON has no N key.
- Local p and the inter-module edges are as recorded; the inter-module autapse rule is not checked here.
