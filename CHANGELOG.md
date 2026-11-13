# Changelog

All notable changes to **MicroRing Studio** are documented in this file.

## [3.7.1] - 2026-11-13

### Changed & Calibrated Against Peer-Reviewed Experimental Publications
- **48-Publication / 50-Case Comprehensive Experimental Validation Suite (`benchmarks/run_publication_validation.py` & `docs/VERIFICATION_AND_BENCHMARKS.md`)**:
  - Expanded the head-to-head first-principles validation benchmark across **48 peer-reviewed Tier 1–4 experimental publications (`50` hardware cases, `50/50 PASS`, zero hardcoded output presets)** spanning all **7 native solver engines** (`2D Conformal Bent Vector Maxwell FEM`, `Spatial Cylindrical CMT + CW/CCW Split-Doublet Fitter`, `2D-FEM Butt/Edge & 2D-FDFD Grating Couplers`, `Twin-Core Supermode & Multi-Mode EME MZI Splitters`, `3D Crystal-Tensor EO Wafer Orientation + Sub-THz 300+ GHz T-Rail CPW + Detuned MRM`, `2D Nonlinear Poisson-Boltzmann Si PN/PIN TCAD`, and `2D Steady/Harmonic/Crank-Nicolson Thermal FEM`) from **`532 nm` Visible to `4200 nm` Mid-IR** across `Si`, `Ge/CaF2`, `Si3N4`, `TiO2`, `Ta2O5`, `Diamond`, `4H-SiC`, `3C-SiC`, `AlN`, `ScAlN (10%/20%/30%)`, `X/Z-cut TFLN`, `X-cut LiTaO3 (LTOI)`, `Epitaxial BTO/Si`, `Poled Thin-Film PZT`, `GaAs/AlGaAs`, `InP MQW`, and `GSST PCM`.
- **Sputtered `AlN`, `Sc_xAl_{1-x}N` & `4H-SiC` Pockels Tensor Calibration (`materials.py`, `active_tuning.py`)**:
  - Updated `n_aln_film` (`r33 = +1.50 pm/V`, `r13 = -0.92 pm/V`, opposite signs) and `n_scaln_film(x)` (`x = 0.10, 0.20, 0.30, 0.319`) to match **Y. Yang et al., *Nat. Commun.* 15, 6168 (2024)** (`<= 20% Sc` dip with `r13` sign flip near `x ~ 0.16`, `>= 30% Sc` enhancement to `r33 = 2.07 pm/V`, `r13 = +1.29 pm/V` at `x = 0.319`, plus `S. Wang et al., Laser Photonics Rev. 19, 2401178, 2025` vertical-vs-coplanar electrode gain) and composition-dependent permittivity (`Wingqvist JAP 2010`, `Fichtner JAP 2019`).
  - Calibrated `n_4h_sic_anisotropic` (`r13 = -0.50 pm/V`, `r33 = 0.03 pm/V`) to **R. Wang, J. Li, L. Cai, & Q. Li, *Opt. Lett.* 48, 1482–1485 (2023)** (measured `|r13| = 0.3–0.7 pm/V` dominating over nearly vanishing `|r33| <= 0.03 pm/V` in high-$Q$ $c$-axis `4H-SiCOI` microresonators, replacing legacy unverified `r33 = 2.7 pm/V`).
- **Clamped High-Speed Thin-Film `PZT` Calibration (`materials.py`, `active_tuning.py`, `eom_electrode.py`)**:
  - Calibrated `PZT` clamped RF Pockels tensor (`r33 = 67.0 pm/V`, `r13 = 20.0 pm/V`) and clamped GHz permittivity (`eps_RF = 320, 380`) to **Alexander et al., *Nat. Commun.* 9, 3444 (2018)** (`La2O5Ti`-seeded thin-film PZT), replacing legacy unclamped low-frequency values (`110 pm/V`, `eps_DC = 850`).
- **Sub-THz (`100–350 GHz`) RF Electrode Loss & Dispersion Physics (`eom_electrode.py`)**:
  - Added **Hammerstad-Jensen (1980)** frequency-dependent conductor surface-roughness skin resistance (`Rq = 70 nm` electroplated `Au`, `35 nm` `Cu`, `45 nm` `Al`), **Heinrich (IEEE MTT 1993)** finite-thickness vertical-sidewall current relief + exact Gupta Eq. (7.98) half-width arguments (`4*pi*a = 2*pi*W_S`), damped polar optical-phonon / ferroelectric GHz dielectric loss slope $\tan\delta(f)$ (`Unferdorben J. Infrared Millim. THz Waves 2015`, `Eltes JLT 2019`), and physical transverse Signal + Ground T-stem skin resistance $R_{\mathrm{stem,cell}}(f)$ (replacing the fixed `Q_stem = 18` approximation).
- **2D Thermal FEM Isolation & CMOS `W/M1` Interconnect Heat Sinking (`multiphysics_tcad.py`)**:
  - Expanded deep Bosch DRIE isolation trench geometry (`42 um` trench depth with adaptive substrate domain height) and modeled periodic CMOS tungsten contact vias + `AlCu` Metal-1 busbar heat sinking on `doped_si_rib` heaters, matching **Jacques et al., *Opt. Express* 27, 10456 (2019)** and **Fang et al., *Opt. Express* 19, 5809 (2011)** across all four heater topologies (`top_heater`, `doped_si_rib`, `deep_trench`, `undercut_suspended`).

## [3.7.0] - 2026-11-12

### Added
- **Sub-THz (`100–300+ GHz`) 2D Quasi-TEM RF Traveling-Wave / Slow-Wave T-Rail EOM Electrode Solver & Mask Generator (`eom_electrode.py`, Zero HFSS / COMSOL)**:
  - Solves anisotropic 2D Laplace electrostatics (`scikit-fem` `ElementTriP1`) across `gsg_cpw`, capacitively-loaded slow-wave `t_rail_cpw` (3-bar Signal + 4-bar Ground periodic unit cell with Step-2 ferroelectric slab etch corridors and localized $\mathrm{XeF}_2$ undercut pockets), `gs_coplanar`, `vertical_microstrip`, and split crystal-axis `lumped_ring_gsg`, using a continuous 2D `(W_S, L_rail)` root-finder to simultaneously match `Z_0 = 50 Ohm` and `n_RF = n_g,opt` (`|Δn| < 0.005`, reproducing wide `W_S ~ 81 um`, `L_rail ~ 10.6 um` on Quartz).
  - Evaluates exact Ghione (1993) conformal incremental-inductance conductor skin loss (`Au`/`Cu`/`Al`), multi-zone unwrapped symmetric ABCD Bloch-Floquet periodic unit-cell dispersion (`f_bragg_cutoff_GHz`) + radial T-stem LC resonance, Frankel-Rutledge Cherenkov substrate shockwave radiation ($\alpha_{\mathrm{rad}} \propto f^3 D_{\mathrm{tot}}^2$ when $n_{\mathrm{RF}} < \sqrt{\varepsilon_{\mathrm{sub}}}$), dielectric $\tan\delta$, semiconducting Si handle loss, Si PN $R_s C_j$ diode loss, two-sideband detuned CMT microring optical peaking, wideband electro-optic $S_{21,\mathrm{EO}}(f)$ (`-3 dBe` & `-6 dBe / -3 dBo` bandwidths up to `600 GHz`, achieving `f_3dBe > 300 GHz` and `S21_EO(300 GHz) = -1.56 dBe` on Quartz/Undercut-Si TFLN), coupled 2D vector Maxwell $V_\pi L$, metal proximity optical loss $\alpha_{\mathrm{metal}}$ (`dB/cm`), and $E_{\mathrm{bit}}$ (`fJ/bit`).
  - Exports DRC/LVS-clean foundry `.GDS` + KLayout `.lyp` masks (`100/125/150/200 um` GSG probe pads with `METAL_2` `13/0` + `VIA` `12/0` crossover bridges eliminating Metal-1-to-Ground shorts and Metal-1-over-core overlaps, periodic slow-wave `T-rail` fingers, optional on-chip `50 Ohm` TiN load, and Layer `66/0` Wafer Primary Flat + Crystal-Axis Compass Fiducial), IEEE Touchstone `.s2p`, and Ansys Lumerical INTERCONNECT `TWMOD` `.zip` bundles.
- **3D Crystal-Tensor EO Wafer Orientation & `0°–360°` In-Plane Azimuth Finder (`analyze_eo_wafer_orientation` in `eom_electrode.py`)**:
  - Evaluates exact 3D Voigt Pockels tensor contraction $\Delta(\varepsilon^{-1})_{ij} = \sum_k r_{ijk} E_k$ across `x_cut_ln`, `y_cut_ln`, `z_cut_ln` (`3m` $\mathrm{LiNbO}_3$), `x_cut_lt`, `z_cut_lt` (`3m` $\mathrm{LiTaO}_3$ `TFLT`), `a_cut_bto_si`, `c_cut_bto` (`4mm` epitaxial $\mathrm{BaTiO}_3$), `inplane_poled_pzt` (`4mm` electrode-self-poled $\mathrm{PbZr}_{0.52}\mathrm{Ti}_{0.48}\mathrm{O}_3$), `c_plane_6mm` & `a_plane_6mm` (`6mm` `AlN`/`ScAlN`/`4H-SiC`/`GaN`), and `cubic_001_43m` & `cubic_110_43m` (`-43m` `3C-SiC`/III-V), computing zero-walk-off principal-axis $\phi_{\mathrm{opt}}$ vs unconstrained shear-lobe peaks, signed $r_{\mathrm{eff}}(\phi)$, $V_\pi L(\phi)$, optical birefringence $n_{\mathrm{opt}}(\phi)$, RF $\varepsilon_{\mathrm{RF}}(\phi)$, SEMI primary wafer-flat alignment rules, and circular microring unsplit vs split push-pull efficiency (`0%` vs `63.7%` on X-cut TFLN/TFLT; `100%` on C-plane wurtzite, Z-cut LN/LT, and radially self-poled PZT).
- **5 New Ferroelectric Electro-Optic Platforms (`materials.py`, `active_tuning.py`, `eom_electrode.py`, `24` -> `29` Materials)**:
  - **`LT_Xcut` & `LT_Zcut` (Thin-Film Lithium Tantalate $\mathrm{LiTaO}_3$, `3m`)**: Bruner et al. *JAP* 2003 + Wang et al. *Nature* 629, 788 (2024) ($r_{33} = 30.5\text{ pm/V}$, ultra-low birefringence $\Delta n = +0.0038$ = 19× lower than TFLN).
  - **`BTO` & `BTO_ccut` (Epitaxial Tetragonal Barium Titanate $\mathrm{BaTiO}_3$, `4mm`)**: Wemple et al. 1968 + Abel et al. *Nat. Mater.* 2019 + Eltes et al. *Nat. Mater.* 2020 (clamped high-speed RF $r_{42}^S = 360\text{ pm/V}$, unlocking $|r_{\mathrm{eff}}| \approx 286\text{ pm/V}$ and sub-`0.5 V*cm` $V_\pi L$ along $\phi \approx 51^\circ$ near the `Si [110]` SEMI flat on `a_cut_bto_si`).
  - **`PZT` (Lanthanide-Seeded MPB Lead Zirconate Titanate $\mathrm{PbZr}_{0.52}\mathrm{Ti}_{0.48}\mathrm{O}_3$, `4mm`)**: Alexander et al. *Nat. Commun.* 9, 3444 (2018) ($r_{\mathrm{eff}} = 110\text{ pm/V}$ with post-fab electrode-self-aligned $\mathbf{P}_s \parallel \mathbf{E}_{\mathrm{RF}}$ across all `0°–360°` azimuths, enabling `100%` unsplit circular microring modulation).
- **2D Nonlinear Poisson-Boltzmann Silicon PN / PIN Doping TCAD (`multiphysics_tcad.py`, Zero Lumerical CHARGE)**:
  - Solves $-\nabla\cdot(\varepsilon\nabla\psi) = q(p - n + N_D^+ - N_A^-)$ via damped Newton-Raphson across `lateral_pn`, `l_shaped_pn`, `u_shaped_pn`, and `pin_diode` junctions with 3-stage implants (`p/n` core, `p+/n+` intermediate slab, `p++/n++` heavy silicide contact + Gaussian implant straggle), extracting 2D potential $\psi(x,y)$ & carrier maps, differential $C_j(V) = dQ/dV$ (`fF/um`), Caughey-Thomas 2D slab $R_s$ (`Ohm*cm`), intrinsic & lumped $f_{\mathrm{RC}}$, coupled 2D vector Maxwell Soref-Bennett $V_\pi L(V)$ & $\alpha_{\mathrm{FCA}}(V)$, and multi-implant `.GDS` (`P_DOPING` `5/0`, `N_DOPING` `6/0`, `VIA` `12/0`, `METAL_1` `11/0`).
- **2D Steady-State + Crank-Nicolson Transient & Harmonic Thermal FEM (`multiphysics_tcad.py`, Zero Lumerical HEAT / COMSOL)**:
  - Assembles 2D thermal stiffness $\mathbf{K}_{\mathrm{th}}$ and heat-capacity mass $\mathbf{M}_{\mathrm{th}}$ matrices across `top_heater`, `deep_trench` (etched air-isolation trenches `40/0`), `undercut_suspended` ($\mathrm{XeF}_2$ undercut cavity + longitudinal oxide suspension-bridge anchor conductance, sub-`1 mW` $P_\pi$), and `doped_si_rib`, extracting $P_\pi$, $V_\pi$, $I_\pi$, $R_{\mathrm{th}}$, Crank-Nicolson transient heating/cooling $\Delta T_{\mathrm{core}}(t)$ ($t_{10\text{--}90\%}$ rise/fall), harmonic $|H_{\mathrm{th}}(f)|^2$ (`f_3dB_thermal_kHz`), neighbor thermal crosstalk decay, and heater + trench `.GDS` + `.zip` bundles.
  - Wired across Stage 10 Web/Desktop UI (`📡 EOM · PN · Thermal`), Bilingual Copilot (`EN | 中文`), CLI (`microring agent eom`), MCP (`12` tools including `microring_eom_multiphysics`), Cookbook `08`, and Test Suite `14` (`14/14` suites passing).

## [3.6.0] - 2026-11-02

### Added
- **2D-FEM + EME Mach-Zehnder Interferometer (`MZI`) & High-Speed Modulator Suite (`mzi_interferometer.py`, Zero 3D FDTD)**:
  - **4 Selectable `50:50` Splitter / Combiner Topologies Solved Directly from 2D Conformal Vector Maxwell Eigenmodes**:
    - **`directional_coupler` (`2x2` DC)**: Mirror-symmetric Twin-Core 2D conformal vector FEM Even/Odd supermodes ($n_{\mathrm{even}}(\lambda), n_{\mathrm{odd}}(\lambda)$, $L_\pi(\lambda) = \frac{\lambda}{2\Delta n}$) + S-bend transition coupling integral (`Split = 49.5/49.5%, IL = 0.044 dB` on `220 nm` SOI).
    - **`mmi_2x2` & `mmi_1x2` (`2x2` and `1x2` Multimode Interference Couplers)**: Multi-mode 2D conformal vector FEM Eigenmode Expansion (`EME`, up to 11 guided lateral modes) + complex Poynting vector access-taper overlap + optimal self-imaging length $L_{\mathrm{MMI}}$ synthesis + top-view 2D $|E(x,z)|$ self-imaging interference field map (`IL = 0.55 dB` on `1x2 MMI`, `0.93 dB` on `2x2 MMI`).
    - **`y_branch` (`1x2` Symmetric Adiabatic Y-Junction)**: 2D vector FEM waist-narrowed stem-to-twin-core Even supermode complex Poynting overlap across the foundry minimum lithographic apex gap (`y_apex_gap_um = 130 nm` -> `IL = 0.273 dB` on `220 nm` SOI).
  - **Passive & Active MZI Modulator Performance (`simulate_mzi`)**:
    - Supports balanced or unbalanced arms ($\Delta L$ or target `FSR` in `nm`/`GHz`), dual-width dispersion-engineered arms ($W_1, W_2$), and **Ring-Assisted MZI (`RAMZI`)** linearized transfer functions.
    - Couples straight phase-shifter arms (`push_pull` differential or `single_arm`) to our native 2D-FEM active tuning cross-sections (`si_pn_depletion`, `pockels_eo`, `thermo_optic`, `si_pin_injection`) across `V`, `mA`, `mW`, `ΔT`, or `E`, extracting **`V_π·L [V·cm]`**, **`V_π [V]`**, **`P_π [mW]`**, non-cavity electrical RC / thermal **`3-dB` bandwidth**, dynamic extinction ratio **`ER [dB]`**, ON-state insertion loss **`IL_on [dB]`**, multi-bias `Bar`/`Cross` spectra, and 241-point switching transfer curves $T(u)$.
  - **Foundry MZI `.GDS` & Ansys Lumerical INTERCONNECT Export (`export_mzi_gds`, `export_mzi_interconnect_bundle`)**:
    - Exports complete `Y-Branch`, `DC`, or `MMI` MZI `.GDS` + KLayout `.lyp` layouts (with `TiN` heaters `10/0` or `GSG` push-pull electrodes `11/0`, recognized on upload by `analyze_uploaded_gds`), plus passive-verified N-port `SPAR .dat`, active arm `OMM .txt` lookup tables, and `.lsf` schematic builders for **Ansys Lumerical INTERCONNECT**.
  - Wired across Stage 9 Web/Desktop UI (`🔀 MZI`), Bilingual Copilot (`EN | 中文`), CLI (`microring agent mzi`), MCP (`11` tools including `microring_design_mzi`), Cookbook `07` (`07_mzi_spectra_dc_mmi_and_modulators.py`), and Test Suite `13` (`13/13` suites passing).

## [3.5.0] - 2026-11-01

### Added
- **Photonic IC Chip I/O Coupler Suite (`io_couplers.py`, Zero Lumerical / COMSOL / RSoft)**:
  - **Butt / Edge Coupler (`design_butt_edge_coupler`)**: Solves expanded inverse-taper facet eigenmodes via 2D conformal vector Maxwell FEM with wide-to-narrow shift-invert $n_{\mathrm{eff}}$ tracking (`~2.1 s` for 5-tip optimization), evaluates exact complex Poynting vector overlap against 8 target fiber and III-V laser diode presets (`SMF28_1550`, `SMF28_1310`, `Lensed_2p5um`, `Lensed_4p0um`, `UHNA_3`, `DFB_Laser_Butt`, `SMF_Vis780`, `MidIR_InF3`), computes Fresnel facet reflection (bare vs AR-coated), substrate `BOX` leakage, adiabatic EME taper length $L_{\mathrm{taper}}$, and lateral/vertical `1-dB` & `3-dB` alignment tolerances ($\pm \Delta x, \pm \Delta y$).
  - **Surface Grating Coupler (`design_grating_coupler`)**: Two-stage license-free solver combining 2nd-order Rytov Bloch-index Bragg and Gaussian-apodized tooth synthesis (`synthesize_grating_teeth`) with a native **2D Full-Wave Maxwell Frequency-Domain Finite-Difference (`2D FDFD`)** solver (`scipy.sparse.linalg.spsolve`) featuring Uniaxial Perfectly Matched Layers (`UPML`), subpixel permittivity averaging, exact total-field/scattered-field guided-mode port launch, bottom Si handle / `BOX` Fabry-Perot reflection, and tilted Gaussian fiber overlap (`~2.5 s` broadband spectrum + 2D radiated $|E(z,y)|$ field map).
  - **Foundry Coupler GDSII & Lumerical INTERCONNECT S-Parameter Export (`export_io_coupler_gds`, `export_io_coupler_interconnect`)**: Generates inverse-taper edge couplers (with deep facet dicing trench layer `40/0`), curved confocal focusing or linear surface grating couplers (shallow etch layer `4/0`), complete `127 um` pitch fiber-array microring loopback + calibration reference `.GDS` frames, and passive-verified 2-port complex optical S-parameter (`SPAR .dat`) `.zip` bundles for Ansys Lumerical INTERCONNECT.
- **Instant Ultra-High-$Q$ (`Q = 10^4` to `10^8+`) Convergence & 3D-FDTD Ring-Down Diagnostic (`mrr_topologies.py`)**:
  - Upgraded `simulate_mrr_topology` with Lorentzian-graded `sinh` resonance grid clustering (`0.1 fm` resolution) to resolve ultra-high-$Q$ resonances (`Q > 3e7`) instantaneously (`~1 s`) without time-domain ring-down stall.
  - Added `analyze_high_q_fdtd_convergence` quantifying photon lifetime $\tau_{\mathrm{ph}} = Q_L / \omega_0$, required `1e-5` auto-shutoff FDTD simulation time ($t_{10^{-5}} = 11.51\,\tau_{\mathrm{ph}}$ and cavity round-trips), and exact finite-$M$ round-trip impulse-response truncation spectra demonstrating FFT sinc-ringing and collapsed extinction ratio vs exact $M \to \infty$ steady-state TMM.
  - Added `export_coupler_cutout_gds` exporting only the non-resonant bus-to-ring coupler region (`.gds` + `.lyp` + automated Lumerical 3D FDTD `.lsf` script, `t_sim < 1 ps`, `>10,000x` faster than full-ring 3D FDTD).
  - Wired across Stage 8 Web/Desktop UI (`🔌 Butt/GC & High-Q`), Bilingual Copilot (`EN | 中文`), CLI (`microring agent io-coupler`, `microring agent high-q`), MCP (`10` tools including `microring_io_coupler` & `microring_high_q_fdtd`), Cookbook `06`, and Test Suite `12` (`12/12` suites passing).

### Changed
- **Project & Package Rename for Photonic IC Discoverability**:
  - Renamed repository and package from `openmrr-studio` (`openmrr`) to **`MicroRing Studio`** (`repo: microring-studio`, `CLI & import: microring`).

## [3.4.0] - 2026-11-01

### Added
- **Native 2D-FEM Multiphysics Active Tuning Engine (`active_tuning.py`, `eigenmode_solver.py` - Zero COMSOL)**:
  - Added spatial complex refractive-index perturbation `solve_perturbed_neff_sweep_fem` (`ElementDG(ElementTriP1())` inside the 2nd-order curl-conforming `Nedelec1 * Lagrange P2` Maxwell FEM solver) with complex transverse `B`-inner-product eigenvector tracking across arbitrary multi-state sweeps on a single shared mesh.
  - Added **2D Fourier Heat Conduction FEM (`solve_2d_heater_thermal_fem`)** solving $-\nabla\cdot(\kappa_{\mathrm{th}}\nabla \Delta T) = Q_{\mathrm{vol}}$ across the actual heater cross-section (`TiN`, `Pt`, `W`, `NiCr`, `doped_Si`), heater width/thickness/gap/offset, and buried oxide (`BOX`) thickness above the isothermal Si handle substrate to extract $R_{\mathrm{th}}$ (`K/mW`), $R_{\mathrm{elec}}$ (`Ohm`), $\tau_{\mathrm{th}}$ (`us`), $f_{3\mathrm{dB,TO}}$ (`kHz`), and $P_\pi$ (`mW`), plus direct temperature $\Delta T$ (`K`) input.
  - Added **2D Anisotropic Electrostatic Laplace FEM (`solve_2d_electrostatic_eo_fem`)** solving $-\nabla\cdot(\varepsilon_0\boldsymbol{\varepsilon}_{\mathrm{DC}}\nabla\phi) = 0$ on coplanar `GSG` or vertical electrode geometries (`electrode_gap_um`, `electrode_width_um`, `electrode_thickness_um`) with dielectric screening, crystal-axis tensor projection (`LN_Xcut`, `LN_Zcut`, `4H-SiC`, `3C-SiC`, `AlN`, `ScAlN`, `GaN`), line capacitance $C_{\mathrm{elec}}$ (`fF/um`), 50-Ohm RC bandwidth, and $V_\pi L$ (`V*cm`), plus direct electric-field $E$ (`V/um`) input.
  - Added **2D Silicon Lateral PN Depletion (`si_pn_depletion`) & PIN Forward Injection (`si_pin_injection`)** modeling Poisson depletion width $W_{\mathrm{dep}}(V)$, Debye-screened carrier profiles $(p(x,y), n(x,y))$, `p++`/`n++` contact wings, slab series resistance $R_s$, junction capacitance $C_j(V)$, RC/carrier-lifetime $f_{3\mathrm{dB,EO}}$, and Soref-Bennett / Nedeljkovic complex plasma dispersion $\Delta n + i\Delta k$ (capturing simultaneous phase shift and modal free-carrier absorption in `dB/cm`).
- **Turnkey Ansys Lumerical INTERCONNECT Compact Model Exporter (`export_lumerical_interconnect_bundle`)**:
  - Exports verified passive multi-port S-parameters (`microring_passive_sparams.dat` for `Optical N-Port S-Parameter` with strictly ascending frequency `[Hz]`, unwrapped phase, and passivity check $\sum |S|^2 \le 1$), active complex index perturbation tables (`microring_active_neff_table.txt` for `Optical Modulator Measured` with strictly ascending SI bias and $\mathrm{Im}(n_{\mathrm{eff}}) \ge 0$), automated INTERCONNECT circuit builder (`build_microring_interconnect.lsf`), Python `lumapi.INTERCONNECT()` runner (`run_interconnect_lumapi.py`), CSVs, and single-click `.zip` bundles.
  - Integrated across the Desktop/Web UI (`⚡ Tuning & IC` tab), Bilingual Copilot (`EN | 中文`), Agent CLI (`microring agent active-tuning --interconnect`), MCP server (`microring_active_tuning`), Cookbook `05` (`05_active_tuning_pn_eo_to_interconnect.py`), and Test Suite `11`.

## [3.3.0] - 2026-11-01

### Added
- **Cleanroom Lithography Over-Dose / Under-Dose (`Positive PR` vs `Negative PR`) & CD-Bias Error Engine (`fab_tolerance.py`)**:
  - Coupled 2D conformal vector FEM + spatial CMT corner evaluator (`run_fabrication_error_analysis`) modeling resist-tone physics (`Positive PR`: ZEP520A / PMMA / KrF / ArF where over-dose shrinks $W$ and widens $\mathrm{gap}$, shifting resonators under-coupled and blue-shifting $\lambda_0$; `Negative PR`: HSQ / ma-N 2403 / SU-8 where over-dose blooms $W$ and shrinks $\mathrm{gap}$, shifting resonators over-coupled, red-shifting $\lambda_0$, and flagging sub-`80 nm` gap bridging / RIE lag).
  - Reports `ER >= 15 dB` and `ER >= 10 dB` CD & exposure-dose yield windows, resonance wavelength sensitivity $d\lambda_0/d\mathrm{CD}$ (`pm/nm`), TiN microheater $\Delta T$ thermal trim budget, and 3-corner overlaid resonance dips.
  - Added `export_fab_bias_array_gds` to generate 5-cell pre-biased mask compensation brackets (`ΔCD_drawn = [-20, -10, 0, +10, +20] nm`, adjusting both $W + \delta$ and $\mathrm{gap} - \eta\delta$) + KLayout `.lyp`.
  - Integrated across the Desktop/Web UI (`🧪 Fab & Dose` tab + Retina 2x dual-axis canvas plots), Bilingual Copilot (`EN | 中文` natural-language queries), Agent CLI (`microring agent fab-tolerance`), MCP server (`microring_fab_tolerance`), and cookbook script `examples/01_devices_and_sweeps/04_lithography_dose_pr_tolerance.py`.

### Changed
- **Clean Repository Organization**:
  - Consolidated secondary documentation (`README_zh-CN.md`, `CONTRIBUTING.md`, `llms.txt`) into `docs/`, flattened validation and figure scripts into `benchmarks/`, and streamlined `packaging/` and top-level root around [`AGENTS.md`](./AGENTS.md) and [`.mcp.json`](./.mcp.json).

## [3.2.0] - 2026-10-31

### Added
- **6-Step Guided Bilingual (`EN | 中文`) Conversational UI (`ui_frontend.py`, `copilot.py`)**:
  - Redesigned the studio into a spacious, zero-overload step-by-step guided chatbot (`1. Goal` → `2. Waveband` → `3. Material` → `4. Stack & 2D Mode` → `5. Propagation Loss & Ring/FSR` → `6. Critical Coupling & GDS`) with a progressive right-hand visual stage and an optional slide-over `⚙ Expert Controls` drawer.
  - Added a top-bar **`EN | 中文`** language switch translating all UI labels, stepper badges, placeholders, plots, and Copilot prompts/chips instantaneously without losing design state.
- **Bidirectional Waveguide Propagation Loss (`dB/cm`) Workflow (`mrr_cmt_fit_predict.py`, `sweeps.py`, `agent_tools.py`)**:
  - Added exact group-index conversions `loss_db_per_cm_to_q_int` and `q_int_to_loss_db_per_cm`, propagation loss sweeps (`alpha_loss_db_per_cm`), and `--loss` CLI/MCP support.
  - Upgraded the Lab Spectrum CMT Fitter to accept both linear-power and logarithmic `dB`/`dBm` `.CSV`/`.TXT` traces, extract propagation loss $\alpha$ (`dB/cm`) + $Q_{\mathrm{int}}$ + self-consistent multi-FSR $n_g$, and hand off extracted loss to new ring critical-coupling synthesis in 1 click.
- **Native Desktop Installers (`.dmg` & `.exe`) + Offline/Online (`Claude` & `Codex`) Modes**:
  - Added `src/microring/build_standalone.py`, `packaging/windows_installer.iss`, and `packaging/build_installers_workflow.yml` to compile zero-Python standalone installers (`MicroRing-Studio-v3.2.0-macOS-AppleSilicon.dmg` and `MicroRing-Studio-v3.2.0-Windows-Setup-x64.exe`).
  - Added a 4-mode AI Engine switcher in the Studio UI (`⚡ Offline` | `🖥 Local Ollama/MLX` | `🟣 Online · Claude` | `🟢 Online · Codex / OpenAI`), supporting both locally signed-in `claude` / `codex` CLIs (zero API key needed) and optional in-memory API keys, plus `.devcontainer/devcontainer.json` for 1-click browser launch on GitHub Codespaces.

### Changed
- Removed legacy unused `gmsh`/`femwell` branch in `eigenmode_solver.py` so 100% of core code and dependencies are permissive (`MIT` / `BSD-3-Clause` / `Boost-1.0`, zero GPL).

## [3.1.0] - 2026-10-31

### Added
- **Full Visible (`400 nm`) to Mid-IR (`14.0 µm`) Multi-Waveband Engine**:
  - Expanded `MATERIAL_CATALOG` from 18 to **24 clean-room optical materials**, adding `TiO2` (anatase, athermal $dn/dT < 0$), `Ta2O5`, `GaN` (uniaxial wurtzite), `Diamond` (UV–LWIR), `CaF2`, and `MgF2`.
  - Added realistic interband absorption $k(\lambda) > 0$ below bandgap (`Si < 1.12 µm`, `a-Si < 0.85 µm`, `Ge < 1.90 µm`) and Mid-IR multiphonon absorption $k(\lambda) > 0$ for `SiO2` above `3.60 µm`, coupled directly into cavity $Q_{\mathrm{int}}$ and live UI compatibility banners (`check_wavelength_compatibility`).
  - Added `WAVEBAND_PRESETS`, UI Waveband & Bottom Cladding (`BOX/Sub`) selectors, multi-waveband Copilot quick-reply chips, and cookbook script `examples/01_devices_and_sweeps/03_multiband_vis_1310_midir.py`.
- **Cross-Solver & Commercial Lumerical Validation Suite (`benchmarks/lumerical_validation/`)**:
  - Added `generate_and_audit_benchmarks.py` verifying `< 1.2e-3` $|\Delta n_{\mathrm{eff}}|$ agreement (`1.81e-4` on O-band SOI, `3.52e-4` on C-band 4H-SiC, `7.82e-4` on Mid-IR Ge/CaF2) between MicroRing's BSD Conformal Vector FEM and Yee-grid FDE across `532 nm` to `4200 nm`.
  - Added `run_lumapi_validation.py` for automated head-to-head Ansys Lumerical `MODE FDE` + `3D FDTD` coupler S-parameter validation.
- **AI Agent & MCP Discovery Infrastructure**:
  - Added root `llms.txt`, Model Context Protocol Registry `server.json`, `--clad` / `--top-clad` CLI flags on `microring agent design`, and cross-platform GitHub Actions CI (`.github/workflows/ci.yml`).

### Fixed
- Fixed air-clad / asymmetric-cladding dielectric perturbation $\Delta\varepsilon$ in `mrr_topologies.py` to reference the upper cladding surrounding the etched waveguide ridge (`top_clad_material`) rather than the bottom substrate (`clad_material`), and included `L_rt(L_s)` round-trip loss scaling during racetrack critical-coupling root polishing.

## [3.0.0] - 2026-10-31

### Added
- Initial full-stack release of **MicroRing Studio**:
  - Pure-BSD 2nd-order curl-conforming 2D cylindrical bent/straight Maxwell FEM eigenmode solver (`conformal_mesher.py`, `eigenmode_solver.py`).
  - 7 microring resonator topologies (`point_circular_ap/ad`, `racetrack_ap/ad`, `pulley_ap/ad`, `vernier_crow_ad`) with spatial CMT integration and 1-click critical coupling.
  - Bilingual (`EN` / `中文`) Conversational Photonics Copilot + Retina 2D Web/Desktop UI (`pywebview` + `Starlette`).
  - Bidirectional GDSII / OASIS bridge (`upload .gds -> polygon caliper measurement -> simulate` and `export single/MPW array .gds + KLayout .lyp`).
  - BIC-guided experimental lab spectrum fitter (`Lorentzian`, `Split-Doublet`, `Fano`, `Multi-FSR`).
  - Ultra-low-token (`<150 tokens/call`) MCP stdio server (`microring mcp`) and agent CLI (`microring agent`) with local SSD mode caching.
