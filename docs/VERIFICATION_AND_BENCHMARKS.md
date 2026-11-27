# Cross-Solver Verification & Ansys Lumerical Benchmark Protocol

Scientific credibility requires stating exact physical equations, independent cross-solver agreement, and validity boundaries where full 3D time-domain simulation (`3D FDTD`) remains mandatory before foundry tape-out.

---

## 1. Independent Dual-Engine 2D Bent Maxwell Eigenmode Audit (`532 nm` $\rightarrow$ `4200 nm`)

Reproducible locally with zero commercial licenses via:

```bash
python benchmarks/generate_and_audit_benchmarks.py
```

This benchmark solves the **exact same 2D cylindrical bent waveguide cross-sections** across **five integrated-photonics wavebands** (`Visible 532 nm`, `VNIR 780 nm`, `O-Band 1310 nm`, `C-Band 1550 nm`, and `Mid-IR 4200 nm`) using two mathematically independent Maxwell discretizations:
1. **`MicroRing Conformal Vector FEM` (`scikit-fem`)**: Unstructured 2nd-order curl-conforming `Nedelec1` transverse + `Lagrange P2` azimuthal finite elements (`ElementTriN2 * ElementTriP2`) on an interface-conforming pure-BSD Delaunay mesh with exact cylindrical coordinate transformation $\rho(x) = 1 + x/R$.
2. **`Structured Yee-Grid FDE` (`tidy3d` local CPU `ModeSolver`)**: Finite-difference frequency-domain eigenmode discretization on a staggered Yee lattice (the same discretization family used by commercial FDE mode solvers).

### Verified Cross-Solver Results (`Darwin-arm64`, Python 3.12)

| Case ID | $\lambda$ (nm) | Platform & Stack | $R$ (µm) | $n_{\mathrm{eff}}$ (Conformal FEM) | $n_{\mathrm{eff}}$ (Yee-Grid FDE) | $|\Delta n_{\mathrm{eff}}|$ | $n_g$ | FSR (nm) | $\kappa^2$ (CMT) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Vis_532nm_Si3N4_Strip` | `532.0` | `180 nm Si3N4 / SiO2` | `25.0` | `1.766850` | `1.768019` | **`1.17e-03`** | `2.2519` | `0.800` | `0.0535` |
| `VNIR_780nm_TiO2_Strip` | `780.0` | `200 nm TiO2 / SiO2` (`86°`) | `30.0` | `1.877503` | `1.877864` | **`3.60e-04`** | `2.6911` | `1.199` | `0.0626` |
| `OBand_1310nm_SOI_220nm` | `1310.0` | `220 nm Si / SiO2` | `15.0` | `2.562140` | `2.562321` | **`1.81e-04`** | `4.3593` | `4.177` | `0.0353` |
| `CBand_1550nm_4HSiC_Pulley` | `1550.0` | `600 nm 4H-SiC / SiO2` (`82°`) | `40.0` | `2.275336` | `2.274984` | **`3.52e-04`** | `2.7978` | `3.417` | `0.0365` |
| `MidIR_4200nm_Ge_CaF2` | `4200.0` | `1.4 µm Ge rib (0.3 µm slab) / CaF2 / Air` | `65.0` | `3.704086` | `3.703304` | **`7.82e-04`** | `4.3295` | `9.168` | `0.0063` |

> **Why Conformal FEM Matters on Slanted High-Index Sidewalls**: On high-contrast slanted sidewalls (`Si` or `Ge` with `82°–88°` etch angles), a coarse Cartesian Yee grid without subpixel permittivity averaging snaps the slanted wall to the nearest Yee cell face, introducing `1e-2` staircase bias unless `dl <= 5–10 nm`. MicroRing's transfinite conformal triangular mesher aligns element edges directly along angled sidewalls at any angle `70°–90°` with zero staircasing.

---

## 2. Independent 4-Way Commercial Solver Validation (`Ansys Lumerical 2026 R1`, `COMSOL Multiphysics 6.4` & `Ansys HFSS 2026 R1`)

- **Full Validation Report & Figures**: [**`docs/COMMERCIAL_SOLVER_VALIDATION_REPORT.md`**](./COMMERCIAL_SOLVER_VALIDATION_REPORT.md) (**`10/10` automated physics audits `PASS`** across all **11 physical test suites**, two 6-panel overview figures [`commercial_validation_overview.png`](./images/commercial_validation_overview.png) and [`commercial_validation_p2_overview.png`](./images/commercial_validation_p2_overview.png), and 12 unit-labeled CSV plus 9 JSON archives in `benchmarks/outputs/`).

All four solvers (`MicroRing Studio v3.9.0`, `Ansys Lumerical 2026 R1`, `COMSOL Multiphysics 6.4`, and `Ansys HFSS 2026 R1`) execute independently from neutral specification JSONs (`commercial_validation_specs.json` and `commercial_validation_specs_p2.json`, zero solver outputs) with no post-hoc parameter fitting:

| Suite | Physical Engine Validated | Commercial Reference(s) | Measured Agreement |
| :--- | :--- | :--- | :--- |
| **Suite 1** | 2D Bent & Straight Vector Maxwell FEM (`532–4200 nm`, angled walls, uniaxial, rib) | Lumerical `MODE FDE` & COMSOL `ewfd` | Max $|\Delta n_{\mathrm{eff}}| = 1.6\times 10^{-5}$ (COMSOL) / $5.35\times 10^{-4}$ (Lumerical); $n_g$ within `0.01%–0.05%` |
| **Suite 2** | 3D Bus-to-Ring Evanescent Couplers (5 wavebands + `100–250 nm` SOI gap sweep) | Lumerical `3D FDTD` (RTX 5090 GPU) | `9/9` GPU runs converged (`passivity <= 1.0034`); SOI `100–250 nm` within `0.1%–1.5%`; `4H-SiC` pulley `0.97%`, `Ge` racetrack `2.1%`, `532 nm` `3.7%`, `780 nm` `5.8%` |
| **Suite 3** | MZI `2x2` DC Supermodes, `1x2` MMI / Y-Branch & 26-Tooth Uniform Surface GC | Lumerical `MODE`/`3D FDTD`/`2D FDTD` & COMSOL `ewfd` | DC $L_\pi$ within `0.16%` (COMSOL) / `0.36%`; `1x2` MMI `0.242 vs 0.242 dB`, Y-Branch `1.02 vs 1.07 dB`, uniform GC `3.30 / 3.33 / 3.26 dB` |
| **Suite 4** | 2D Steady & Transient Thermo-Optic Microheaters (`Top TiN`, `Deep Trench`, `BOX` scaling) | Lumerical `HEAT` & COMSOL `ht` + `ewfd` | $R_{\mathrm{th}}$ within `0.23%–0.38%` (COMSOL) / `0.51%–1.16%` (`HEAT`); exact `t_rise == t_fall`; `20 µm` crosstalk `< 0.01%` |
| **Suite 5** | 2D Anisotropic Electrostatic Pockels EO (`X-cut TFLN`, `ScAlN_20`) | COMSOL `es` + `ewfd` & Lumerical `MODE` | Single-arm $V_\pi L$ within `0.20%–0.87%` across all 3 devices (incl. `ScAlN_20` `0.87%`); `X-cut TFLN` $C_{\mathrm{line}}$ within `0.26%` |
| **Suite 6** | Chip I/O: Butt/Edge Inverse Tapers (SOI & Si3N4 DFB) + Apodized Focusing GC | Lumerical `MODE`/`EME`/`2D FDTD` & COMSOL `ewfd` | Facet IL within `0.005–0.057 dB`, `1-dB` alignment tol exact (`0.50 / 0.32 µm`); Apodized GC peak `1553.2 vs 1553.4 nm`, IL `2.95 vs 2.91 dB` |
| **Suite 7** | `2x2` MMI Self-Imaging, Full Ring (`R=5 µm`) & Native `INTERCONNECT` SPAR Load | Lumerical `3D FDTD` GPU & `INTERCONNECT ONA` | Full-ring (`R=5 µm`) FSR `17.83 vs 18.19 nm` & $Q_L$ `7926 vs 8174` (`3.0%`); `INTERCONNECT` MZI FSR `7.000 vs 7.012 nm`, ER `33.35 vs 33.33 dB` |
| **Suite 8** | Sub-THz EOM RF Electrodes (`GSG CPW` on HR-Si/Quartz/Undercut-Si + `10–300 GHz` 3D Periodic Slow-Wave `T-Rail`) | **Ansys HFSS 3D** (`2026 R1`) & COMSOL `es`/`emw`/3D Periodic | Uniform CPW (`40 GHz`) $Z_0$ within `0.3%–1.1%`, $n_{\mathrm{RF}}$ within `0.25%–0.70%` vs HFSS 3D (`C` `0.006%–0.08%`, `L` `0.22%–0.30%` vs COMSOL `es`); 3D `T-Rail` $Z_0 = 49.51$ vs COMSOL 3D `49.24 Ω` (`0.56%`), `40–300 GHz` $\alpha_{\mathrm{RF}}$ `1.20–7.59` vs HFSS `1.37–7.46 dB/cm` |
| **Suite 9** | 2D Nonlinear Poisson-Boltzmann Silicon PN TCAD (`Lateral`, `L-Shaped`, `U-Shaped`) | Lumerical `CHARGE`+`MODE` & COMSOL `g`+`ewfd` | $C_j(0/-3\text{ V})$ within `0.17%–3.2%` across all 3 solvers; Soref-Bennett FCA `10.76 vs 10.78 dB/cm` (`0.19%`), $V_\pi L$ `1.48 vs 1.56 V·cm` |
| **Suite 10** | Advanced Microheaters (`XeF2 Undercut`, `Doped-Rib`, `Co-Int PN+TiN`) + 3D Thermal Check | COMSOL `ht` (2D & 3D `L=200 µm`) | 2D $R_{\mathrm{th}}$ within `0.54%–0.78%` (`34.47 vs 34.67 K/mW` Undercut); confirms `+54.4%` PN contact heat-sinking; 3D end-loss `-2.0%` |
| **Suite 11** | Cleanroom Lithography CD Bias Sensitivity (`±20 nm` SOI `220 nm`) | Lumerical `MODE`+`3D FDTD` GPU & COMSOL `ewfd` | $\mathrm{d}\lambda/\mathrm{dCD} = 775.79 / 775.97 / 776.36\text{ pm/nm}$ (**`0.07%`** 3-way spread); $\kappa^2$ within `0.3%–4.8%` of 3D FDTD |

Reproduce the complete 4-way validation locally on a workstation with Ansys Lumerical, COMSOL Multiphysics, and Ansys HFSS:

```bash
# Phase 1 (Suites 1-5): neutral spec + independent solver runs
python benchmarks/commercial_validation_specs.py
python benchmarks/run_microring_independent.py
python benchmarks/run_lumerical_independent.py
python benchmarks/run_comsol_independent.py

# Phase 2 (Suites 6-11): neutral spec + independent 4-way solver runs
python benchmarks/commercial_validation_specs_p2.py
python benchmarks/run_microring_independent_p2.py
python benchmarks/run_lumerical_independent_p2.py
python benchmarks/run_comsol_independent_p2.py
python benchmarks/run_hfss_independent_p2.py

# Unified audit: 10/10 physics checks, 12 CSVs, two 6-panel figures, and Markdown report
python benchmarks/compile_commercial_validation_report.py
```

For a standalone Lumerical-only eigenmode and 3D coupler check (`MODE FDE` + `3D FDTD`), the legacy script remains available:

```bash
python benchmarks/generate_and_audit_benchmarks.py
python benchmarks/run_lumapi_validation.py
```

---

## 3. Native 2D-FEM Active Tuning & Lumerical INTERCONNECT Passivity Verification (`v3.4.0`)

Reproducible locally with zero COMSOL or Lumerical licenses via:

```bash
python examples/01_devices_and_sweeps/05_active_tuning_pn_eo_to_interconnect.py
python tests/test_microring_all.py
```

MicroRing Studio `v3.4.0` solves cross-section active perturbations natively in `scikit-fem` and injects cell-wise $\Delta\varepsilon(x,y) = (n_0 + \Delta n + i\Delta k)^2 - n_0^2$ (`ElementDG(ElementTriP1())`) directly into the 2D curl-conforming `Nedelec1 * Lagrange P2` Maxwell eigenmode solver (`solve_perturbed_neff_sweep_fem`):

| Active Mechanism | Cross-Section PDE / Carrier Physics | Verified Benchmark (`Darwin-arm64`) | Extracted Figure of Merit |
| :--- | :--- | :--- | :--- |
| **`si_pn_depletion`** (`220 nm` Si rib, `90 nm` slab, `0..-5 V`) | Poisson $W_{\mathrm{dep}}(V)$ + Debye-screened 2D carriers + Soref-Bennett $\Delta n + i\Delta k$ | `R = 20 µm`, `1550 nm` | `11.53 pm/V`, $V_\pi L = 2.088\text{ V}\cdot\text{cm}$, $f_{3\mathrm{dB}} = 6.29\text{ GHz}$ |
| **`pockels_eo`** (`600 nm` X-cut TFLN, `GSG` electrodes) | 2D anisotropic Laplace $-\nabla\cdot(\varepsilon_0\boldsymbol{\varepsilon}_{\mathrm{DC}}\nabla\phi)=0$ + tensor $r_{ij}$ | `G = 5.0 -> 3.2 µm` | $V_\pi L = 5.358 \rightarrow 3.603\text{ V}\cdot\text{cm}$, $f_{3\mathrm{dB}} = 19.02\text{ GHz}$ |
| **`thermo_optic`** (`400 nm` Si3N4 + `TiN` microheater) | 2D Fourier heat conduction $-\nabla\cdot(\kappa_{\mathrm{th}}\nabla\Delta T)=Q_{\mathrm{vol}}$ | `BOX = 2.0 -> 3.5 µm` | $R_{\mathrm{th}} = 1.153 \rightarrow 1.566\text{ K/mW}$, $P_\pi = 156.8 \rightarrow 115.5\text{ mW}$ |
| **Lumerical INTERCONNECT (`SPAR .dat`)** | Multi-state complex S-matrix export (`|S21|^2 + |S31|^2 <= 1`) | `1875` frequency pts/state | `passivity = True` across all bias states |

---

## 4. License-Free Chip I/O (`2D-FEM Butt` & `2D-FDFD Grating Couplers`) & High-$Q$ Convergence (`v3.5.0`)

Reproducible locally with zero Lumerical, COMSOL, or RSoft licenses via:

```bash
python examples/01_devices_and_sweeps/06_butt_gc_couplers_and_high_q_fdtd.py
```

### Verified Benchmarks (`Darwin-arm64`, Python 3.12)

| Module & Benchmark Case | Numerical Formulation | Wall Time | Key Physical Outputs |
| :--- | :--- | :---: | :--- |
| **Butt Coupler (`220 nm` SOI $\rightarrow$ `2.5 µm` Lensed Fiber)** | 2D Conformal Vector Maxwell FEM + Complex Poynting Overlap + Adiabatic EME | `2.10 s` | Optimal $W_{\mathrm{tip}} = 214\text{ nm}$, Total $\mathrm{IL} = 0.56\text{ dB}$ (`87.9%`), `1-dB` Align Tol $= (\pm 0.53, \pm 0.52)\,\mu\text{m}$ |
| **Butt Coupler (`300 nm` Si3N4 O-Band $\rightarrow$ III-V DFB `2.4×1.2 µm`)** | Elliptical Laser Mode Complex Poynting Overlap + Facet AR | `2.15 s` | Optimal $W_{\mathrm{tip}} = 304\text{ nm}$, Total $\mathrm{IL} = 0.61\text{ dB}$ (`86.8%`), `1-dB` Align Tol $= (\pm 0.51, \pm 0.32)\,\mu\text{m}$ |
| **Uniform `220 nm` SOI Grating Coupler (`70 nm` etch, `2 µm` BOX, `10°`)** | 2nd-Order Rytov Bloch Synthesis + Native 2D Full-Wave Maxwell `FDFD` (`UPML`) | `2.45 s` | Pitch $\Lambda = 634.1\text{ nm}$, Peak $\mathrm{IL} = 2.85\text{ dB}$ (`51.9%`), `1-dB / 3-dB BW` $= 39.6 / 71.2\text{ nm}$, $S_{11} = -27.5\text{ dB}$ |
| **Apodized Focusing `220 nm` SOI Grating Coupler (`ff` `0.87 -> 0.48`)** | Variable-Fill Gaussian Apodization + 2D Full-Wave `FDFD` + `SPAR .dat` | `2.55 s` | Peak $\mathrm{IL} = 2.62\text{ dB}$ (`54.7%`), `1-dB / 3-dB BW` $= 34.8 / 61.8\text{ nm}$, $S_{11} = -22.7\text{ dB}$, `passivity = True` |
| **Ultra-High-$Q$ `Si3N4` Pulley Ring (`R = 100 µm`, `α = 0.08 dB/cm`)** | Sub-fm `sinh` Grid + Exact $M \to \infty$ TMM vs Finite-$M$ Truncated FDTD | `1.10 s` | $Q_L = 3,022,829$ ($\mathrm{FWHM} = 0.513\text{ pm}$, $\tau_{\mathrm{ph}} = 2.49\text{ ns}$), `1e-5` FDTD needs `28.64 ns` (`8,287` RTs) vs `0.67 ps` Coupler Cutout (`42,488x` faster) |

---

## 5. 2D-FEM + EME Mach-Zehnder Interferometer (`MZI`) Splitters & Modulators (`v3.6.0`)

Reproducible locally with zero 3D FDTD, Lumerical, or COMSOL licenses via:

```bash
python examples/01_devices_and_sweeps/07_mzi_spectra_dc_mmi_and_modulators.py
```

### Verified Benchmarks (`Darwin-arm64`, Python 3.12, `ΔL = 80 µm` on `220 nm` SOI)

| Module & Benchmark Case | Numerical Formulation | Wall Time | Key Physical Outputs |
| :--- | :--- | :---: | :--- |
| **`2x2` Directional Coupler (`DC`) MZI** | Mirror-Symmetric Twin-Core 2D-FEM Even/Odd Supermodes + S-Bend Integral | `4.2 s` | Split `49.50 / 49.50%`, $\mathrm{IL} = 0.044\text{ dB}$, $\mathrm{Imb} = 0.000\text{ dB}$, $\mathrm{FSR} = 7.048\text{ nm}$, $\mathrm{ER} = 42.4\text{ dB}$ |
| **`2x2` Multimode Interference (`MMI`) MZI** | Multi-Mode 2D-FEM Eigenmode Expansion (`EME`, up to 11 modes) + Poynting Overlap | `6.8 s` | Split `39.79 / 40.89%`, $\mathrm{IL} = 0.932\text{ dB}$, $\mathrm{Imb} = -0.119\text{ dB}$, 2D $|E(x,z)|$ self-imaging map |
| **`1x2` Symmetric `MMI` MZI** | Symmetric Center-Fed Multi-Mode 2D-FEM `EME` + Taper Poynting Overlap | `5.9 s` | Split `44.04 / 44.04%`, $\mathrm{IL} = 0.551\text{ dB}$, $\mathrm{Imb} = 0.000\text{ dB}$, $\mathrm{ER} = 29.7\text{ dB}$ |
| **`1x2` Linear `Y-Branch` MZI** | 2D-FEM Blunt-Apex Stem-to-Twin-Core Supermode Overlap (`130 nm` apex gap) | `5.8 s` | Split `39.5 / 39.5%`, $\mathrm{IL} = 1.024\text{ dB}$ (matches Lumerical `3D FDTD` `1.069 dB`; PSO-sculpted spline requires 3D shape optimization) |
| **`Si` Lateral `PN` Push-Pull MZI (`1.5 mm`)** | 2D-FEM Soref-Bennett Depletion + `2x2 MMI` + INTERCONNECT `SPAR` + `OMM` | `14.5 s` | $V_\pi L = 0.826\text{ V}\cdot\text{cm}$ ($V_\pi = 5.74\text{ V}$), Non-Cavity $f_{3\mathrm{dB}} = 7.08\text{ GHz}$, `passivity = True` |
| **`X-Cut TFLN` `GSG` Push-Pull EO MZI (`5 mm`)** | 2D Anisotropic Laplace FEM + Tensor $r_{33}$ + `Y-Branch` | `15.8 s` | $V_\pi L = 2.079\text{ V}\cdot\text{cm}$ ($V_\pi = 4.29\text{ V}$), RC $f_{3\mathrm{dB}} = 16.59\text{ GHz}$ |
| **`Si3N4` `TiN` Thermo-Optic MZI + `RAMZI`** | 2D Fourier Heat FEM + `2x2 DC` + All-Pass Ring Assisted Phase Arm | `13.7 s` | $P_\pi = 159.4\text{ mW}$, Thermal $f_{3\mathrm{dB}} = 19.11\text{ kHz}$, `RAMZI` $Q_L = 7,364$ |

---

## 6. Sub-THz `300+ GHz` Slow-Wave T-Rail EOM Electrodes, Si PN TCAD & Transient Thermal FEM (`v3.7.1`)

Reproducible locally with zero Ansys HFSS, Lumerical CHARGE/HEAT, or COMSOL licenses via:

```bash
python examples/01_devices_and_sweeps/08_eom_rf_electrode_pn_tcad_and_thermal.py
```

### Verified Benchmarks (`Darwin-arm64`, Python 3.12)

| Module & Benchmark Case | Numerical Formulation | Key Physical Outputs |
| :--- | :--- | :--- |
| **`X-Cut TFLN` Slow-Wave `T-Rail` on Quartz (`Λ = 25 µm`, `t_Au = 1.5 µm`, `L = 4.5 mm`)** | 2D Laplace FEM + Hammerstad-Jensen Roughness + ABCD Bloch-Floquet + Polar-Phonon $\tan\delta(f)$ | $W_S = 82.0\,\mu\text{m}$, $L_{\mathrm{rail}} = 10.8\,\mu\text{m}$, $Z_0 = 50.19\,\Omega$, $n_{\mathrm{RF}} = 2.263$ vs $n_g = 2.266$, $\alpha_{\mathrm{RF}}(40/300\text{G}) = 1.24 / 8.56\text{ dB/cm}$, $S_{21,\mathrm{EO}}(300\text{ GHz}) = -1.94\text{ dBe}$, $f_{3\mathrm{dBe}} = 396.5\text{ GHz}$, $V_\pi L = 2.43\text{ V}\cdot\text{cm}$ |
| **`X-Cut TFLN` Slow-Wave `T-Rail` on Quartz (`Λ = 220 µm` Coarse Pitch)** | Multi-Zone Unwrapped ABCD Bloch-Floquet Unit-Cell Stopband | $f_{\mathrm{Bragg}} = 301\text{ GHz}$, $S_{21,\mathrm{EO}}(300\text{ GHz}) = -9.95\text{ dBe}$, $f_{3\mathrm{dBe}} = 236.4\text{ GHz}$ (`bragg_periodic_cutoff`) |
| **`X-Cut TFLN` Slow-Wave `T-Rail` on Undercut-Si vs Unthinned HR-Si** | Gap-Localized $\mathrm{XeF}_2$ Air Cavity vs Frankel-Rutledge $f^3$ Cherenkov Shockwave | Undercut-Si: $f_{3\mathrm{dBe}} = 404.9\text{ GHz}$ (`0 dB/cm` Cherenkov) vs HR-Si: $f_{3\mathrm{dBe}} = 104.9\text{ GHz}$ ($\alpha_{\mathrm{RF}}(300\text{G}) = 23.8\text{ dB/cm}$) |
| **`BTO` Microring Modulator (`Q_L = 12,000`, Detuning $\delta = 0.577 \rightarrow 1.45$)** | Two-Sideband Detuned CMT Lorentzian Optical Peaking | $f_{3\mathrm{dBe}} = 7.18 \rightarrow 16.36\text{ GHz}$ (`+0.58 dBe` optical peaking) |
| **`220 nm` SOI `L-Shaped` PN Doping TCAD (`0 -> -3 V`)** | 2D Damped Newton-Raphson Poisson-Boltzmann + Caughey-Thomas $R_s$ | $V_\pi L = 0.598\text{ V}\cdot\text{cm}$, $\alpha_{\mathrm{FCA}}(-3\text{V}) = 7.62\text{ dB/cm}$, $f_{\mathrm{RC,intr}} = 60.1\text{ GHz}$, $f_{\mathrm{TW,3dBo}} = 116.9\text{ GHz}$ |
| **`XeF2` Undercut, Deep-Trench & `N++` Doped-Rib vs Top `TiN` Thermal Shifter** | 2D Steady + Crank-Nicolson Transient Thermal FEM ($\mathbf{K}_{\mathrm{th}}, \mathbf{M}_{\mathrm{th}}$) | Top: $P_\pi = 17.06\text{ mW}$ (`31.0 kHz`), Trench: $11.24\text{ mW}$ (`0.08%` xtalk), Doped-Rib: $28.15\text{ mW}$ (`42.1 kHz`), Undercut: $0.56\text{ mW}$ (`0.04%` xtalk) |

---

## 7. Live First-Principles Validation Against 369 Unique Peer-Reviewed Experimental Publications (`v3.9.0`)

Reproducible locally with zero commercial solver licenses via:

```bash
python benchmarks/run_publication_validation.py --workers 3 --chunk 6 --save-csv --write-ledger
```

- **Full 369-Paper Ledger**: [**`docs/PUBLICATION_VALIDATION_LEDGER.md`**](./PUBLICATION_VALIDATION_LEDGER.md) (`Pub-001` .. `Pub-369`, **`367/369 PASS` (`99.5%`)**, `100%` 1-to-1 unique DOIs, zero MDPI venues, zero SPIE/conference proceedings, zero preprints, zero hardcoded output presets; the 2 non-passing rows are non-uniform 3D geometries outside uniform-section 2D-FEM/EME scope: `Pub-074` Floquet sinusoidally-modulated artificial-gauge-field coupler and `Pub-219` 13-parameter PSO shape-optimized Y-branch, retained unmodified for transparency).
- **8-Domain Coverage (`369` Unique Journal DOIs)**:
  1. *I. Waveguides, Passive Microrings & CMT* (`88` papers)
  2. *II. Kerr Microresonator Frequency Combs* (`36` papers)
  3. *III. Phase-Change & Non-Volatile Photonics* (`71` papers: `GST`, `GSST`, `Sb2S3`, `Sb2Se3`, `BTO`, `PZT`, `FeFET/TFLN`)
  4. *IV. Chip-to-Fiber Couplers & Cleanroom Fabrication Tolerance* (`20` papers)
  5. *V. MZI Splitters & Unitary Meshes* (`45` papers)
  6. *VI. Electro-Optic Crystal & Sub-THz Modulators* (`47` papers)
  7. *VII. Silicon Carrier PN/PIN Plasma TCAD* (`33` papers)
  8. *VIII. Thermo-Optic Phase Shifters* (`29` papers)

Every metric is solved organically from the published cross-sectional device geometry and literature-cited material tensors across all **7 native physical solver engines** (`2D Conformal Bent Vector Maxwell FEM`, `Spatial Cylindrical CMT + CW/CCW Split-Doublet Fitter`, `2D-FEM Butt/Edge & 2D-FDFD Grating Couplers`, `Twin-Core Supermode & Multi-Mode EME MZI Splitters`, `3D Crystal-Tensor EO Wafer Orientation + 2D Quasi-TEM CPW Laplace FEM + ABCD Bloch-Floquet 300+ GHz T-Rail + Detuned MRM`, `2D Nonlinear Poisson-Boltzmann PN/PIN TCAD`, and `2D Steady/Harmonic/Crank-Nicolson Thermal FEM`) from `450 nm` Visible to `4200 nm` Mid-IR. Representative anchor cases are summarized below; see [**`PUBLICATION_VALIDATION_LEDGER.md`**](./PUBLICATION_VALIDATION_LEDGER.md) for the complete 369-entry table.

### Domain I: Waveguides, Passive Microrings, Pulley Couplers, PCM & CMT Fitter (`Pub-01` .. `Pub-12`)

| ID | Experimental Publication | Fabricated Device Platform | Measured (Published) | Simulated (`MicroRing Studio` 2D-FEM) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`Pub-01`** | S. Romero-Garcia *et al.*, *Opt. Express* **21**, 14036 (2013) | `180 nm` LPCVD $\mathrm{Si}_3\mathrm{N}_4$ Strip Ring @ `532 nm` (`W = 380 nm`, `R = 25 µm`) | $n_{\mathrm{eff}} = 1.72\text{--}1.85$, $n_g = 2.20\text{--}2.32$, $\mathrm{FSR} \approx 0.78\text{--}0.83\text{ nm}$ | $n_{\mathrm{eff}} = 1.7627$, $n_g = 2.2524$, $\mathrm{FSR} = 0.800\text{ nm}$ | `PASS` |
| **`Pub-02`** | J. D. B. Bradley *et al.*, *Opt. Express* **20**, 23821 (2012); J. T. Choy *et al.*, *Opt. Lett.* **37**, 539 (2012) | `200 nm` $\mathrm{TiO}_2/\mathrm{SiO}_2$ Strip Ring @ `780 nm` (`W = 460 nm`, `R = 30 µm`) | $n_{\mathrm{eff}} = 1.82\text{--}1.95$, $n_g = 2.60\text{--}2.78$, $\mathrm{FSR} = 1.15\text{--}1.25\text{ nm}$ | $n_{\mathrm{eff}} = 1.9006$, $n_g = 2.6851$, $\mathrm{FSR} = 1.202\text{ nm}$ | `PASS` |
| **`Pub-03`** | H. Jung *et al.*, *Optica* **8**, 811 (2021) | `570 nm` $\mathrm{Ta}_2\mathrm{O}_5$ (Tantala) Strip Ring @ `1550 nm` (`W = 1.6 µm`, $\alpha = 0.10\text{ dB/cm}$) | $n_g = 2.15\text{--}2.25$, $\mathrm{FSR} \approx 2.85\text{--}2.98\text{ nm}$, $Q_{\mathrm{int}} = 3.8\times 10^6$ | $n_{\mathrm{eff}} = 1.8468$, $n_g = 2.1843$, $\mathrm{FSR} = 2.917\text{ nm}$, $Q_{\mathrm{int}} = 3.85\times 10^6$ | `PASS` |
| **`Pub-04`** | B. J. M. Hausmann *et al.*, *Nat. Nanotechnol.* **9**, 318 (2014) | `600 nm` CVD Diamond Racetrack Ring @ `1550 nm` (`W = 950 nm`, `R = 30 µm`, $\alpha = 0.34\text{ dB/cm}$) | $n_{\mathrm{Diamond}} = 2.386$, $n_g = 2.55\text{--}2.82$, $Q_{\mathrm{int}} \approx 1.0\times 10^6$ | $n_{\mathrm{eff}} = 2.0752$, $n_g = 2.5873$, $Q_{\mathrm{int}} = 1.34\times 10^6$ | `PASS` |
| **`Pub-05`** | W. Bogaerts *et al.*, *Laser Photonics Rev.* **6**, 47 (2012) | `220 nm` SOI Strip Rings @ `1550 nm` (`R = 5 µm`) & `1310 nm` (`R = 15 µm`) | `1550`: $n_g = 4.18\pm 0.08$, $\mathrm{FSR} \approx 18.3\text{ nm}$; `1310`: $n_g = 4.35\pm 0.08$, $\mathrm{FSR} \approx 4.18\text{ nm}$ | `1550`: $n_g = 4.327$, $\mathrm{FSR} = 17.68\text{ nm}$; `1310`: $n_g = 4.408$, $\mathrm{FSR} = 4.13\text{ nm}$ | `PASS` |
| **`Pub-06`** | X. Ji *et al.*, *Optica* **4**, 619 (2017) | `730×1800 nm` CMP $\mathrm{Si}_3\mathrm{N}_4$ Ultra-High-$Q$ Ring (`R = 115 µm`, $\alpha = 0.01\text{ dB/cm}$) | $n_g = 2.08\text{--}2.14$, $Q_{\mathrm{int}} = (3.7\pm 0.7)\times 10^7$, $Q_L \approx 1.85\times 10^7$, $\mathrm{FWHM} \approx 0.08\text{ pm}$ | $n_g = 2.116$, $Q_{\mathrm{int}} = 3.73\times 10^7$, $Q_L = 1.86\times 10^7$, $\mathrm{FWHM} = 0.083\text{ pm}$ | `PASS` |
| **`Pub-07`** | M. A. Guidry *et al.*, *Optica* **7**, 1139 (2020); D. M. Lukin *et al.*, *Nat. Photonics* **14**, 330 (2020) | `600 nm` `4H-SiCOI` Wrap-Around Pulley Ring @ `1550 nm` (`R = 40 µm`, $\alpha = 0.30\text{ dB/cm}$) | $n_{\mathrm{eff}} = 2.24\text{--}2.32$, $n_g = 2.75\text{--}2.85$, $\mathrm{FSR} \approx 3.42\text{ nm}$, $Q_{\mathrm{int}} \approx 1.6\times 10^6$ | $n_{\mathrm{eff}} = 2.2860$, $n_g = 2.7886$, $\mathrm{FSR} = 3.428\text{ nm}$, $Q_{\mathrm{int}} = 1.64\times 10^6$ | `PASS` |
| **`Pub-08`** | S. Lipkowitz *et al.*, *Optica* **11**, 417 (2024) | `550×1300 nm` Sputtered $c$-$\mathrm{AlN}$ Ring @ `1550 nm` ($\alpha = 1.74\text{ dB/cm}$) | $\alpha = 1.74\text{ dB/cm} \leftrightarrow Q_{\mathrm{int}} = 2.2\times 10^5$ ($n_g \approx 2.12\text{--}2.22$) | $n_g = 2.144$, $Q_{\mathrm{int}} = 2.17\times 10^5$ | `PASS` |
| **`Pub-09`** | S. Wang *et al.*, *APL Photonics* **9**, 016102 (2024); L. Xu *et al.*, *Photon. Res.* **12**, 793 (2024) | `400 nm` $\mathrm{Sc}_{0.10}\mathrm{Al}_{0.90}\mathrm{N}$ (`2.4 dB/cm`) & `500 nm` $\mathrm{Sc}_{0.20}\mathrm{Al}_{0.80}\mathrm{N}$ (`1.7 dB/cm`) Rings | `10% Sc`: $Q_{\mathrm{int}} = 1.47\times 10^5$; `20% Sc`: $Q_{\mathrm{int}} \approx 2.1\text{--}2.3\times 10^5$ | `10% Sc`: $n_g = 2.244$, $Q_{\mathrm{int}} = 1.65\times 10^5$; `20% Sc`: $Q_{\mathrm{int}} = 2.32\times 10^5$ | `PASS` |
| **`Pub-10`** | Y. Zhang *et al.*, *Nat. Commun.* **10**, 4279 (2019) | $\mathrm{Ge}_2\mathrm{Sb}_2\mathrm{Se}_4\mathrm{Te}_1$ (`GSST`) Amorphous vs Crystalline PCM @ `1550 nm` | `a-GSST`: $n = 3.32\pm 0.05$, $k < 10^{-4}$; `c-GSST`: $n = 5.08\pm 0.10$, $k = 0.35\text{--}0.45$ ($\Delta n \approx +1.75$) | `a-GSST`: $n = 3.320$, $k = 0$; `c-GSST`: $n = 5.080$, $k = 0.365$ ($\Delta n = +1.76$) | `PASS` |
| **`Pub-11`** | Y.-C. Chang *et al.*, *Opt. Express* **20**, 13252 (2012); M. Nedeljkovic *et al.*, *Opt. Express* **25**, 27431 (2017) | `1.4 µm` Ge Rib (`0.3 µm` slab) on $\mathrm{CaF}_2$ @ Mid-IR `4200 nm` (`R = 65 µm`) | $n_{\mathrm{Ge}} \approx 4.02$, $n_{\mathrm{eff}} = 3.65\text{--}3.78$, $n_g = 4.25\text{--}4.40$, `0` $\mathrm{SiO}_2$ multiphonon warnings | $n_{\mathrm{eff}} = 3.6928$, $n_g = 4.3414$, $\mathrm{FSR} = 9.95\text{ nm}$, `warnings = 0` | `PASS` |
| **`Pub-12`** | T. J. Kippenberg *et al.*, *Opt. Lett.* **27**, 1669 (2002); Q. Li *et al.*, *Opt. Express* **18**, 8367 (2010) | CW/CCW Backscattering Split-Doublet CMT Extraction (`dBm` Lab Trace) | True $Q_{\mathrm{int}} = 1.07\times 10^6$, $\mu_{\mathrm{split}} = 0.26\text{ GHz}$ ($2\mu = 0.52\text{ GHz}$) | Fit $Q_{\mathrm{int}} = 1.07\times 10^6$ (`0.07%` err), $\mu = 0.260\text{ GHz}$, $\alpha = 0.344\text{ dB/cm}$, $R^2 = 1.0000$ | `PASS` |

### Domain II: Butt/Edge Couplers & 2D-FDFD Surface Grating Couplers (`Pub-13` .. `Pub-16`)

| ID | Experimental Publication | Fabricated Device Platform | Measured (Published) | Simulated (`MicroRing Studio` 2D-FEM / FDFD) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`Pub-13`** | V. R. Almeida *et al.*, *Opt. Lett.* **28**, 1302 (2003); J. Cardenas *et al.*, *IEEE Photon. Technol. Lett.* **26**, 2380 (2014) | `220 nm` SOI Inverse-Taper Butt Coupler $\rightarrow$ `2.5 µm` Lensed Fiber (`1550 nm`) | $W_{\mathrm{tip}} = 180\text{--}230\text{ nm}$, $\mathrm{IL} = 0.45\text{--}0.85\text{ dB/facet}$, `1-dB` tol $\approx \pm 0.5\text{--}0.6\,\mu\text{m}$ | $W_{\mathrm{tip}} = 207\text{ nm}$, $\mathrm{IL} = 0.39\text{ dB}$, `1-dB` tol $= (\pm 0.54, \pm 0.54)\,\mu\text{m}$ | `PASS` |
| **`Pub-14`** | T. Barwicz *et al.*, *Opt. Express* **23**, 27912 (2015); P.-I. Dietrich *et al.*, *Laser Photonics Rev.* **12**, 1700258 (2018) | `300 nm` $\mathrm{Si}_3\mathrm{N}_4$ Edge Coupler $\rightarrow$ Elliptical III-V DFB Laser (`1310 nm`) | $\mathrm{IL} = 0.5\text{--}1.1\text{ dB}$, vertical `1-dB` tol $= \pm 0.28\text{--}0.40\,\mu\text{m}$ | $W_{\mathrm{tip}} = 304\text{ nm}$, $\mathrm{IL} = 0.70\text{ dB}$, $\Delta y_{1\mathrm{dB}} = \pm 0.37\,\mu\text{m}$ | `PASS` |
| **`Pub-15`** | D. Taillaert *et al.*, *Jpn. J. Appl. Phys.* **45**, 6071 (2006); D. Vermeulen *et al.*, *Opt. Express* **18**, 18278 (2010) | `220 nm` SOI Uniform `70 nm`-Etch Grating Coupler (`2 µm` BOX, `10°` SMF-28) | $\Lambda \approx 630\text{ nm}$, Peak $\mathrm{IL} = 2.6\text{--}3.1\text{ dB}$, `1-dB BW` $\approx 38\text{--}43\text{ nm}$ | $\Lambda = 632.3\text{ nm}$, $\mathrm{IL} = 2.87\text{ dB}$, `1-dB / 3-dB BW` $= 41.2 / 71.2\text{ nm}$ | `PASS` |
| **`Pub-16`** | R. Marchetti *et al.*, *Photon. Res.* **7**, 201 (2019); D. Benedikovic *et al.*, *Opt. Express* **23**, 22628 (2015) | `220 nm` SOI Gaussian-Apodized Focusing `70 nm`-Etch Grating Coupler | Peak $\mathrm{IL} = 1.9\text{--}2.7\text{ dB}$ ($<$ uniform GC), $S_{11} < -20\text{ dB}$, `1-dB BW` $\approx 32\text{--}40\text{ nm}$ | $\mathrm{IL} = 2.67\text{ dB}$ ($< 2.87\text{ dB}$), $S_{11} = -22.6\text{ dB}$, `1-dB BW` $= 38.0\text{ nm}$ | `PASS` |

### Domain III: MZI `50:50` Splitters (`Y-Branch`, `1x2 MMI`, `2x2 MMI`, `2x2 DC`) (`Pub-17` .. `Pub-20`)

| ID | Experimental Publication | Fabricated Device Platform | Measured (Published) | Simulated (`MicroRing Studio` 2D-FEM + EME) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`Pub-17`** | Y. Zhang *et al.*, *Opt. Express* **21**, 1310 (2013) | `220 nm` SOI `1x2` Y-Branch (`W = 480 nm`, `130 nm` Litho Apex, `1550 nm`, 13-param PSO spline in paper) | PSO-sculpted $\mathrm{IL} = 0.28 \pm 0.02\text{ dB}$; unoptimized linear `130 nm` apex `3D FDTD` $= 1.069\text{ dB}$ | Linear blunt apex: $\mathrm{IL} = 1.024\text{ dB}$ (matches `3D FDTD`; PSO inverse design outside 2D linear model) | `SCOPE` |
| **`Pub-18`** | Z. Sheng *et al.*, *IEEE Photon. Technol. Lett.* **24**, 1772 (2012) | `220 nm` SOI Symmetric Tapered `1x2` MMI Splitter (`1550 nm`) | Excess $\mathrm{IL} = 0.35\text{--}0.65\text{ dB}$, Imbalance $= 0.00\text{ dB}$ | $\mathrm{IL} = 0.629\text{ dB}$, $\mathrm{Imb} = 0.000\text{ dB}$, `9` guided modes | `PASS` |
| **`Pub-19`** | X. Guan *et al.*, *Opt. Express* **22**, 9636 (2014); R. Halir *et al.*, *Laser Photonics Rev.* **9**, 25 (2015) | `220 nm` SOI Paired-Interference Tapered `2x2` MMI (`W_MMI = 6 µm`, `1550 nm`) | Excess $\mathrm{IL} = 0.5\text{--}1.0\text{ dB}$, Imbalance $< 0.20\text{ dB}$ | $\mathrm{IL} = 0.837\text{ dB}$, $\mathrm{Imb} = -0.104\text{ dB}$, `11` guided modes | `PASS` |
| **`Pub-20`** | H. Yamada *et al.*, *Opt. Express* **13**, 8645 (2005); W. Bogaerts *et al.*, *IEEE JSTQE* **16**, 33 (2010) | `220 nm` SOI `2x2` Directional Coupler Asymmetric MZI (`W = 450 nm`, $\Delta L = 80\,\mu\text{m}$) | DC $\mathrm{IL} < 0.15\text{ dB}$, $\mathrm{FSR}(\Delta L = 80\,\mu\text{m}) = 6.9\text{--}7.2\text{ nm}$, $\mathrm{ER} > 30\text{ dB}$ | $\mathrm{IL} = 0.043\text{ dB}$, $\mathrm{FSR} = 6.949\text{ nm}$, $\mathrm{ER} = 39.2\text{ dB}$ | `PASS` |

### Domain IV: 3D Crystal-Tensor EO Wafer Orientation, Traveling-Wave, Sub-THz `T-Rail` & Microring Modulators (`Pub-21` .. `Pub-39`)

| ID | Experimental Publication | Fabricated Device Platform | Measured (Published) | Simulated (`MicroRing Studio` 2D-FEM) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`Pub-21`** | C. Wang *et al.*, *Nature* **562**, 101 (2018) | `600/300 nm` X-cut TFLN GSG on Si (`L = 5 & 20 mm`) | $V_\pi L = 2.2\text{--}2.5\text{ V}\cdot\text{cm}$, $\mathrm{BW}(5\text{ mm}) = 100\text{ GHz}$, $\mathrm{BW}(20\text{ mm}) = 20\text{--}22\text{ GHz}$, $\alpha_{50\mathrm{G}} \approx 5\text{ dB/cm}$ | $V_\pi L = 2.08\text{ V}\cdot\text{cm}$, $\mathrm{BW}(5\text{ mm}) = 106.7\text{ GHz}$, $\mathrm{BW}(20\text{ mm}) = 18.7\text{ GHz}$, $\alpha_{50\mathrm{G}} = 6.49\text{ dB/cm}$ | `PASS` |
| **`Pub-22`** | M. He *et al.*, *Nat. Photonics* **13**, 359 (2019) | X-cut TFLN Push-Pull GSG MZM on Si (`L = 5 mm`, `G = 5.5 µm`) | $V_\pi L = 2.2\text{--}2.5\text{ V}\cdot\text{cm}$, $Z_0 \approx 50\,\Omega$, $f_{3\mathrm{dBe}} > 70\text{ GHz}$ | $V_\pi L = 2.38\text{ V}\cdot\text{cm}$, $Z_0 = 49.1\,\Omega$, $f_{3\mathrm{dBe}} = 130.4\text{ GHz}$ | `PASS` |
| **`Pub-23`** | P. Kharel *et al.*, *Optica* **8**, 357 (2021) | `20 mm` X-cut TFLN Segmented `T-Rail` on Quartz ($\Lambda = 50\,\mu\text{m}$) | $V_\pi = 1.3\text{ V}$ ($2.6\text{ V}\cdot\text{cm}$), $S_{21,\mathrm{EO}}(50\text{G}) = -1.80\text{ dBe}$, $\alpha_{50\mathrm{G}} \approx 1.9\text{ dB/cm}$, $n_{\mathrm{RF}} \approx 2.26$ | $V_\pi = 1.18\text{ V}$ ($2.35\text{ V}\cdot\text{cm}$), $S_{21,\mathrm{EO}}(50\text{G}) = -1.50\text{ dBe}$, $\alpha_{50\mathrm{G}} = 1.86\text{ dB/cm}$, $n_{\mathrm{RF}} = 2.276$ | `PASS` |
| **`Pub-24`** | M. Xu *et al.*, *Optica* **9**, 61 (2022) | `10 mm` X-cut TFLN Slow-Wave `T-Rail` on Quartz (`G = 5 µm`, $\Lambda = 50\,\mu\text{m}$) | $V_\pi L = 2.2\text{--}2.45\text{ V}\cdot\text{cm}$, $Z_0 \approx 50\,\Omega$, $f_{3\mathrm{dBe}} > 110\text{ GHz}$ | $V_\pi L = 2.39\text{ V}\cdot\text{cm}$, $Z_0 = 50.2\,\Omega$, $n_{\mathrm{RF}} = 2.255$, $f_{3\mathrm{dBe}} = 204.9\text{ GHz}$ | `PASS` |
| **`Pub-25`** | F. Valdez *et al.*, *Opt. Express* **31**, 5273 (2023) | `10 mm` X-cut TFLN `T-Rail`: Undercut-Si Air Cavity vs Unthinned HR-Si | Undercut-Si eliminates Si Cherenkov/walk-off $\rightarrow \mathrm{BW} > 110\text{ GHz}$ ($> 2\times$ HR-Si) | Undercut-Si: $n_{\mathrm{RF}} = 2.257$, $\mathrm{BW} = 210.6\text{ GHz}$ vs HR-Si: $n_{\mathrm{RF}} = 2.521$, $\mathrm{BW} = 35.0\text{ GHz}$ | `PASS` |
| **`Pub-26`** | A. J. Mercante *et al.*, *Nat. Photonics* **12**, 747 (2018); Sub-THz Quartz CL-TWE | `5 mm` X-cut TFLN `T-Rail` on Quartz ($\Lambda = 25\,\mu\text{m}$, $t_{\mathrm{Au}} = 1.5\,\mu\text{m}$) | $V_\pi L = 2.2\text{--}2.6\text{ V}\cdot\text{cm}$, $\alpha_{\mathrm{RF}}(300\text{G}) \approx 9\text{--}12\text{ dB/cm}$, $S_{21,\mathrm{EO}}(300\text{G}) \approx -2.2\text{ to }-3.5\text{ dBe}$ | $V_\pi L = 2.40\text{ V}\cdot\text{cm}$, $\alpha_{300\mathrm{G}} = 8.63\text{ dB/cm}$, $S_{21,\mathrm{EO}}(300\text{G}) = -2.16\text{ dBe}$, $f_{3\mathrm{dBe}} = 373.0\text{ GHz}$ | `PASS` |
| **`Pub-27`** | A. Guarino *et al.*, *Nat. Photonics* **1**, 407 (2007); C. Wang *et al.*, *Opt. Express* **26**, 1547 (2018) | Z-cut TFLN (`TM0` Vertical $E_y$) vs X-cut TFLN (`TE0` In-Plane $E_x$) Ring | Z-cut `TM0`: $r_{33} = 30.9\text{ pm/V}$, `100%` unsplit ring; X-cut `TE0`: `0%` unsplit (cancels), split push-pull | Z-cut `TM0`: $r_{\mathrm{eff}} = 30.9\text{ pm/V}$, `100%` unsplit; X-cut `TE0`: `0.0%` unsplit, `94.4%` split | `PASS` |
| **`Pub-28`** | C. Wang *et al.*, *Nature* **629**, 788 (2024) | `600/300 nm` X-cut $\mathrm{LiTaO}_3$ (`LTOI`) GSG on Trap-Rich Si (`L = 7 mm`) | $|n_e - n_o| \approx 0.0038$ ($19\times < \mathrm{LN}$), $V_\pi L = 1.9\text{--}2.2\text{ V}\cdot\text{cm}$, $Z_0 \approx 50\,\Omega$, $f_{3\mathrm{dBe}} \approx 40\text{ GHz}$ | $|\Delta n| = 0.0035$, $V_\pi L = 2.16\text{ V}\cdot\text{cm}$, $Z_0 = 49.0\,\Omega$, $f_{3\mathrm{dBe}} = 45.4\text{ GHz}$ | `PASS` |
| **`Pub-29`** | S. Abel *et al.*, *Nat. Mater.* **18**, 42 (2019); F. Eltes *et al.*, *JLT* **37**, 1456 (2019) | Epitaxial $a$-axis BTO on $\mathrm{SiO}_2$/Si (`G = 4.5 µm`, `L = 2 mm`) | $\phi_{\mathrm{opt}} \approx 45^\circ\text{--}54^\circ$ ($r_{42}$ shear lobe), $V_\pi L = 0.20\text{--}0.45\text{ V}\cdot\text{cm}$, $\mathrm{BW} \approx 25\text{--}55\text{ GHz}$ | $\phi_{\mathrm{opt}} = 53.0^\circ$, $V_\pi L = 0.392\text{ V}\cdot\text{cm}$, $f_{3\mathrm{dBe}} = 55.0\text{ GHz}$ | `PASS` |
| **`Pub-30`** | F. Eltes *et al.*, *Nat. Mater.* **19**, 1164 (2020); C. Xiong *et al.*, *Nano Lett.* **14**, 1419 (2014) | BTO Microring Modulator ($Q_L = 12,000$, Max-Slope $\delta = 0.58$ vs Detuned $1.45$) | $Q = 12\text{k}$ cavity limit $\approx 7\text{--}9\text{ GHz}$ at quadrature; laser detuning extends $\mathrm{BW} > 2\times$ ($> 15\text{ GHz}$) | Slope ($\delta = 0.58$): $f_{3\mathrm{dBe}} = 7.14\text{ GHz} \rightarrow$ Detuned ($\delta = 1.45$): $16.17\text{ GHz}$ | `PASS` |
| **`Pub-31`** | K. Alexander *et al.*, *Nat. Commun.* **9**, 3444 (2018) | $\mathrm{La}_2\mathrm{O}_5\mathrm{Ti}$-Seeded Self-Poled Thin-Film PZT (`h = 170 nm`, `L = 3.2 mm`) | Clamped $r_{\mathrm{eff}} \approx 67\text{ pm/V}$, `100%` self-poled ring, single-arm $V_\pi L \approx 1.6\text{--}3.2\text{ V}\cdot\text{cm}$ | $r_{\mathrm{eff}} = 67\text{ pm/V}$, `unsplit = 100%`, $V_\pi L = 1.63\text{ V}\cdot\text{cm}$, $f_{3\mathrm{dBe}} = 53.3\text{ GHz}$ | `PASS` |
| **`Pub-32`** | C. Xiong *et al.*, *Nano Lett.* **12**, 3562 (2012); S. Zhu & G.-Q. Lo, *Opt. Express* **24**, 12501 (2016) | `450 nm` Sputtered $c$-$\mathrm{AlN}$ Vertical-Field EO Modulator (`1550 nm`, `TM0`) | $r_{33} \approx 1.0\text{--}1.5\text{ pm/V}$, vertical-field $V_\pi L \approx 50\text{--}120\text{ V}\cdot\text{cm}$ | $r_{33} = 1.50\text{ pm/V}$, $V_\pi L = 82.8\text{ V}\cdot\text{cm}$, $f_{3\mathrm{dBe}} = 30.7\text{ GHz}$ | `PASS` |
| **`Pub-33`** | Y. Yang *et al.*, *Nat. Commun.* **15**, 6168 (2024) | $c$-axis $\mathrm{AlN}$ & $\mathrm{Sc}_{0.10/0.20}\mathrm{AlN}$ Pockels Tensor (`1550 nm`) | AlN: $+1.50/-0.92\text{ pm/V}$; $\le 20\%$ Sc: $r_{33} < 1.50$ (no gain), $r_{13}$ flips sign near $x \approx 0.16$ | AlN: $+1.50/-0.92$; $10\%$: $+1.23/-0.36$; $20\%$: $+1.34/+0.32\text{ pm/V}$ | `PASS` |
| **`Pub-34`** | Y. Yang *et al.*, *Nat. Commun.* **15**, 6168 (2024); H. Yoshioka *et al.*, *Nanophotonics* **13**, 3239 (2024) | High-Sc Sputtered $c$-axis $\mathrm{Sc}_{0.32}\mathrm{Al}_{0.68}\mathrm{N}$ Ring (`1550 nm`) | $r_{33} = 2.07 \pm 0.45\text{ pm/V}$, $r_{13} = +1.29 \pm 0.25\text{ pm/V}$ ($> \mathrm{AlN}$) | $x = 0.319$: $+2.07/+1.29\text{ pm/V}$ (`ScAlN_30` TM0 $r_{\mathrm{eff}} = 1.91\text{ pm/V}$) | `PASS` |
| **`Pub-35`** | S. Wang *et al.*, *Laser Photonics Rev.* **19**, 2401178 (2025) | `800 nm` $\mathrm{Sc}_{0.10}\mathrm{Al}_{0.90}\mathrm{N}$ Ring: Vertical Top-Bottom vs Coplanar GSG | Vertical top-bottom plates deliver $> 2.5\times$ stronger $c$-axis EO tuning than coplanar GSG | $|E_{y,\mathrm{vert}}| = 0.268$ vs $|E_{y,\mathrm{cop}}| = 0.000\text{ V}/\mu\text{m/V}$, slope $= 0.776$ vs $0.002\text{ pm/V}$ | `PASS` |
| **`Pub-36`** | K. Powell *et al.*, *Nat. Commun.* **13**, 1851 (2022); T. Fan *et al.*, *Opt. Express* **28**, 25478 (2020) | Cubic `(001)` `3C-SiCOI` ($\bar{4}3m$, `TE0` + Vertical $E_y$, $r_{41} = 1.5\text{--}2.7\text{ pm/V}$) | `[100]` (`0°`): $r_{\mathrm{eff}} = 0$; `[110]` (`45°`): $|r_{41}| = 1.5\text{ pm/V}$ (thin film) to $2.7\text{ pm/V}$ (bulk), $V_\pi L \approx 13\text{--}28\text{ V}\cdot\text{cm}$ | $\phi = 0^\circ$: $0.00\text{ pm/V}$; $\phi_{\mathrm{opt}} = 45^\circ$ (bulk $2.70$), $V_\pi L(1.5\text{ pm/V}) = 24.1\text{ V}\cdot\text{cm}$ | `PASS` |
| **`Pub-37`** | R. Wang, J. Li, L. Cai, & Q. Li, *Opt. Lett.* **48**, 1482 (2023) | Hexagonal $c$-axis `4H-SiCOI` Ring/Modulator (`6mm`, `TE0` + Vertical $E_y$) | $|r_{13}| = 0.3\text{--}0.7\text{ pm/V}$ ($\sim 0.50$) $\gg |r_{33}| \le 0.03\text{ pm/V}$, `100%` unsplit ring symmetry | $|r_{13}| = 0.50 \gg |r_{33}| = 0.03\text{ pm/V}$, `unsplit = 100%`, $V_\pi L = 59.2\text{ V}\cdot\text{cm}$ | `PASS` |
| **`Pub-38`** | R. Spickermann *et al.*, *Electron. Lett.* **32**, 1095 (1996); J. Shin *et al.*, *IEEE JQE* **43**, 479 (2007) | $\mathrm{GaAs}/\mathrm{AlGaAs}$ Slow-Wave Capacitively-Loaded `T-Rail` MZM ($n_0 = 3.374$, $r_{41} = 1.43\text{ pm/V}$) | Slow-wave `T-Rail` $n_{\mathrm{RF}} \approx n_g$ ($\sim 3.5$), $Z_0 \approx 50\,\Omega$, push-pull $V_\pi L = 11\text{--}16\text{ V}\cdot\text{cm}$, $\mathrm{BW} > 40\text{ GHz}$ | $\phi_{\mathrm{opt}} = 55^\circ$, $n_{\mathrm{RF}} = 3.550$ ($n_g = 3.568$), $Z_0 = 49.8\,\Omega$, $V_\pi L = 13.3\text{ V}\cdot\text{cm}$, $\mathrm{BW} = 198.9\text{ GHz}$ | `PASS` |
| **`Pub-39`** | Y. Ogiso *et al.*, *J. Lightwave Technol.* **38**, 249 (2020) | $\mathrm{InP}$ $n\text{-}i\text{-}p\text{-}n$ $\mathrm{InGaAsP}$ MQW `CL-TWE` Slow-Wave `T-Rail` MZM (`L = 4 mm`) | `CL-TWE` $Z_0 \approx 50\,\Omega$, $n_{\mathrm{RF}} \approx n_g$ ($\sim 3.34$), MQW $V_\pi L \approx 0.6\text{--}1.1\text{ V}\cdot\text{cm}$ (bulk $\sim 15$), $\mathrm{BW} \ge 80\text{ GHz}$ | $n_{\mathrm{RF}} = 3.323$ ($n_g = 3.338$), $Z_0 = 49.8\,\Omega$, MQW $V_\pi L = 1.00$ (bulk $15.1$) $\text{V}\cdot\text{cm}$, $\mathrm{BW} = 236.0\text{ GHz}$ | `PASS` |

### Domain V: Silicon Carrier-Depletion (`Lateral`, `L-Shaped`, `U-Shaped` TCAD) & `PIN` Injection (`Pub-40` .. `Pub-44`)

| ID | Experimental Publication | Fabricated Device Platform | Measured (Published) | Simulated (`MicroRing Studio` 2D-TCAD + FEM) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`Pub-40`** | P. Dong *et al.*, *Opt. Express* **20**, 6163 (2012) / IMEC `iSiPP50G` | `220 nm` SOI Lateral PN Depletion MZM ($5\times 10^{17}\text{ cm}^{-3}$, $-2\text{ V}$) | Single-arm $V_\pi L = 1.4\text{--}1.8\text{ V}\cdot\text{cm}$ (PP $0.7\text{--}0.9$), $\mathrm{FCA} = 10\text{--}16\text{ dB/cm}$, $C_j = 0.22\text{--}0.32\text{ fF}/\mu\text{m}$, $\mathrm{BW} \approx 30\text{ GHz}$ | $V_\pi L = 1.35\text{ V}\cdot\text{cm}$ (PP $0.67$), $\mathrm{FCA} = 11.5\text{ dB/cm}$, $C_j = 0.266\text{ fF}/\mu\text{m}$, $\mathrm{BW} = 30.0\text{ GHz}$ | `PASS` |
| **`Pub-41`** | M. R. Watts *et al.*, *Opt. Express* **19**, 22354 (2011); AIM Photonics PDK | `220 nm` SOI L-Shaped Vertical+Horizontal PN Depletion MZM ($-2\text{ V}$) | Single-arm $V_\pi L = 0.85\text{--}1.15\text{ V}\cdot\text{cm}$ ($\sim 23\%$ lower than lateral), $C_j = 0.30\text{--}0.42\text{ fF}/\mu\text{m}$ | $V_\pi L = 1.04\text{ V}\cdot\text{cm}$ ($< 1.35$), $V_\pi L\alpha = 8.8\text{ V}\cdot\text{dB}$, $C_j = 0.325\text{ fF}/\mu\text{m}$ | `PASS` |
| **`Pub-42`** | E. Timurdogan *et al.*, *Nat. Commun.* **5**, 4008 (2014); Z. Yong *et al.*, *Opt. Express* **25**, 6112 (2017) | `220 nm` SOI U-Shaped Wrap-Around PN Junction ($7\times 10^{17}\text{ cm}^{-3}$, $-2\text{ V}$) | Single-arm $V_\pi L = 0.65\text{--}1.00\text{ V}\cdot\text{cm}$ (push-pull $0.35\text{--}0.50\text{ V}\cdot\text{cm}$) | $V_\pi L = 0.69\text{ V}\cdot\text{cm}$ (PP $0.35\text{ V}\cdot\text{cm}$), $C_j = 0.570\text{ fF}/\mu\text{m}$ | `PASS` |
| **`Pub-43`** | W. M. J. Green *et al.*, *Opt. Express* **15**, 17106 (2007); S. Akiyama *et al.*, *Opt. Express* **20**, 2911 (2012) | `220 nm` SOI Forward-Biased PIN Carrier-Injection Ring (`0 -> 1.5 mA`) | Strong carrier-plasma blue-shift ($\Delta\lambda < -100\text{ pm}$ at $\sim 1.5\text{ mA}$) + rising FCA | $\Delta\lambda = -312.3\text{ pm}$ (blue-shift), $\mathrm{FCA}(1.5\text{ mA}) = 17.40\text{ dB/cm}$ | `PASS` |
| **`Pub-44`** | X. Xiao *et al.*, *Opt. Express* **21**, 4116 (2013); P. Dong *et al.*, *Opt. Express* **17**, 22484 (2009) | `220 nm` SOI Reverse-Biased PN Depletion Microring (`R = 20 µm`, `0 -> -5 V`) | Red-shift slope $= 10\text{--}22\text{ pm/V}$, modal FCA drops as depletion widens | Slope $= 10.90\text{ pm/V}$, $\mathrm{FCA}: 12.52 \rightarrow 9.28\text{ dB/cm}$ | `PASS` |

### Domain VI: Thermo-Optic Phase Shifters Across `SOI`, `Si3N4` & `TFLN` (`Pub-45` .. `Pub-50`)

| ID | Experimental Publication | Fabricated Device Platform | Measured (Published) | Simulated (`MicroRing Studio` 2D-Thermal FEM) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`Pub-45`** | M. Jacques *et al.*, *Opt. Express* **27**, 10456 (2019) | `220 nm` SOI Top `TiN` Cladding Heater (`h_BOX = 2 µm`, `g_ox = 0.95 µm`) | $P_\pi = 19\text{--}21.4\text{ mW}$, $\tau = 5.6\,\mu\text{s}$ ($f_{3\mathrm{dB}} \approx 28\text{ kHz}$), $P_\pi\tau \approx 110\text{ mW}\cdot\mu\text{s}$ | $P_\pi = 17.06\text{ mW}$, $\tau = 5.13\,\mu\text{s}$ ($f_{3\mathrm{dB}} = 31.0\text{ kHz}$), $P_\pi\tau = 87.5\text{ mW}\cdot\mu\text{s}$ | `PASS` |
| **`Pub-46`** | M. Jacques *et al.*, *Opt. Express* **27**, 10456 (2019); N. C. Harris *et al.*, *Opt. Express* **22**, 10487 (2014) | `220 nm` SOI `N++` Doped-Rib Heater (`90 nm` slab + CMOS W/M1 vias) | $P_\pi = 22.8\text{--}26\text{ mW}$, fast $\tau = 2.2\text{--}3.8\,\mu\text{s}$ (faster than top cladding heater) | $P_\pi = 28.36\text{ mW}$, $\tau = 3.79\,\mu\text{s}$ ($f_{3\mathrm{dB}} = 42.0\text{ kHz}$) | `PASS` |
| **`Pub-47`** | M. Jacques *et al.*, *Opt. Express* **27**, 10456 (2019); P. Dong *et al.*, *Opt. Express* **18**, 20298 (2010) | `220 nm` SOI Deep Air-Trench Isolated `TiN` Heater | $P_\pi = 9\text{--}13\text{ mW}$, $\tau = 9\text{--}20\,\mu\text{s}$, neighbor crosstalk $< 0.2\%$ | $P_\pi = 11.24\text{ mW}$, $\tau = 9.25\,\mu\text{s}$, crosstalk $= 0.08\%$ | `PASS` |
| **`Pub-48`** | Q. Fang *et al.*, *Opt. Express* **19**, 5809 (2011) | `220 nm` SOI $\mathrm{XeF}_2$ Isotropically Undercut Suspended Bridge | $P_\pi = 0.49\text{--}1.8\text{ mW}$, $\tau = 130\text{--}260\,\mu\text{s}$ ($f_{3\mathrm{dB}} \approx 0.6\text{--}1.2\text{ kHz}$) | $P_\pi = 0.56\text{ mW}$, $\tau = 231.8\,\mu\text{s}$ ($f_{3\mathrm{dB}} = 0.69\text{ kHz}$), crosstalk $= 0.04\%$ | `PASS` |
| **`Pub-49`** | X. Xue *et al.*, *Opt. Express* **24**, 687 (2016); A. Arbabi *et al.*, *Opt. Lett.* **38**, 3878 (2013) | `400 nm` LPCVD $\mathrm{Si}_3\mathrm{N}_4$ Top `TiN` Heater (`h_BOX = 3 µm`, $dn/dT = 2.45\times 10^{-5}\text{ K}^{-1}$) | $P_\pi = 100\text{--}185\text{ mW}$ ($\sim 7.6\times$ smaller $dn/dT$ than Si), $\tau = 7\text{--}18\,\mu\text{s}$ | $P_\pi = 138.9\text{ mW}$, $\tau = 9.27\,\mu\text{s}$ ($f_{3\mathrm{dB}} = 17.2\text{ kHz}$) | `PASS` |
| **`Pub-50`** | M. Jin *et al.*, *Opt. Lett.* **44**, 1265 (2019) | `600/300 nm` X-cut TFLN Trench-Isolated `TiN` Heater ($dn_e/dT = 3.34\times 10^{-5}\text{ K}^{-1}$) | Trench-isolated TFLN `TiN` heater: $P_\pi = 30\text{--}95\text{ mW}$ (un-trenched $110\text{--}180\text{ mW}$), xtalk $< 0.5\%$ | $P_\pi = 63.4\text{ mW}$, $\tau = 11.22\,\mu\text{s}$, crosstalk $= 0.09\%$ | `PASS` |

---

## 8. Physical Scope Boundaries

- **High-Confidence Regime (`MicroRing Studio` 2D Bent FEM + Spatial CMT + 2D FDFD + EME + 2D Multiphysics FEM)**:
  - Guided straight and moderately bent waveguide dispersion ($n_{\mathrm{eff}}$, $n_g$, FSR, single-mode cutoff $W_{\mathrm{SM}}$) and steady-state High-$Q$ resonance spectra ($Q = 10^4$ to $10^8+$);
  - Weak-to-moderate evanescent couplers (`gap >= 150 nm` in NIR, `>= 300 nm` in Mid-IR) including phase-mismatched wrap-around pulley couplers;
  - Conventional MZI `50:50` splitters and combiners: `2x2` Directional Couplers (twin-core supermodes), tapered rectangular `1x2` and `2x2` `MMI` couplers (multi-mode 2D-FEM `EME`), and adiabatic `1x2` `Y-Branch` splitters;
  - Inverse-taper Butt/Edge couplers (`2D Vector FEM` + complex Poynting overlap) and 1D-periodic uniform or apodized curved-focusing Surface Grating Couplers (`2D Full-Wave Sagittal FDFD` + lateral width overlap);
  - Cross-sectional 2D Fourier thermo-optic (`TiN`/`Pt`/`W` microheaters), 2D Laplace Pockels electro-optic (`GSG`/vertical electrodes), and Silicon PN depletion / PIN injection perturbed eigenmode shifts;
  - Positive vs. negative photoresist exposure-dose / CD-bias coupling drift and 5-cell pre-biased mask synthesis;
  - Post-fabrication Lorentzian, CW/CCW Split-Doublet, Fano, and Multi-FSR transmission fitting.
- **Regimes Requiring Full `3D FDTD` / `3D TCAD` Before Final Foundry Tape-Out**:
  - Subwavelength-grating (`SWG`) metamaterial MMIs, pixelated / topology-optimized inverse-designed splitters, or abrupt non-tapered MMI corner back-reflections ($S_{11}$);
  - Ultra-tight sub-`120 nm` high-index-contrast gaps where strong supermode distortion and parasitic coupler radiation loss ($1 - |S_{21}|^2 - |S_{31}|^2 > 1\%$) occur (use `export_coupler_cutout_gds` to simulate only the `< 1 ps` single-pass coupler region in 3D FDTD);
  - 2D pillar subwavelength metamaterial (`SWG`) grating couplers, polarization-splitting 2D gratings, or ultra-short non-adiabatic 3D multi-tip spot-size converters;
  - Abrupt straight-to-bend racetrack junctions without adiabatic Euler transitions, or 3D in-plane longitudinal heat spreading near short heater corners / thermal isolation trenches;
  - Ultra-sharp bends ($R < 5\,\mu\text{m}$ in `Si3N4` or thin-BOX substrate leakage) where radiative $Q_{\mathrm{rad}}$ limits unloaded $Q_{\mathrm{int}}$.
