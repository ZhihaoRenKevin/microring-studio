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

## 2. Commercial Ansys Lumerical (`MODE FDE` + `3D FDTD`) Validation Runner

On a workstation equipped with Ansys Lumerical (`lumapi`), execute a head-to-head validation of both 2D bent eigenmodes (`MODE FDE`) and 3D bus-to-ring coupler S-parameters (`3D FDTD`, `auto shutoff min = 1e-5`) across point, racetrack, and $C^1$-smooth concentric pulley couplers:

```bash
# Step 1: Export canonical benchmark specifications & .gds masks
python benchmarks/generate_and_audit_benchmarks.py

# Step 2: Run head-to-head Ansys Lumerical MODE FDE + 3D FDTD comparison
python benchmarks/run_lumapi_validation.py
```

This generates `benchmarks/outputs/lumerical_vs_microring_validation.csv` and `LUMERICAL_VALIDATION_REPORT.md` with matched diagonal refractive index tensors `(nx, ny, nz)`, sidewall angles, slab heights, and fundamental-mode port expansions (`|S21|^2`).

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
| **`1x2` Adiabatic `Y-Branch` MZI** | 2D-FEM Waist-Narrowed Stem-to-Twin-Core Supermode Overlap (`130 nm` apex gap) | `5.8 s` | Split `46.96 / 46.96%`, $\mathrm{IL} = 0.273\text{ dB}$ (cf. Zhang *et al.*, *Opt. Express* 2013: `0.28 dB`) |
| **`Si` Lateral `PN` Push-Pull MZI (`1.5 mm`)** | 2D-FEM Soref-Bennett Depletion + `2x2 MMI` + INTERCONNECT `SPAR` + `OMM` | `14.5 s` | $V_\pi L = 0.826\text{ V}\cdot\text{cm}$ ($V_\pi = 5.74\text{ V}$), Non-Cavity $f_{3\mathrm{dB}} = 7.08\text{ GHz}$, `passivity = True` |
| **`X-Cut TFLN` `GSG` Push-Pull EO MZI (`5 mm`)** | 2D Anisotropic Laplace FEM + Tensor $r_{33}$ + `Y-Branch` | `15.8 s` | $V_\pi L = 2.079\text{ V}\cdot\text{cm}$ ($V_\pi = 4.29\text{ V}$), RC $f_{3\mathrm{dB}} = 16.59\text{ GHz}$ |
| **`Si3N4` `TiN` Thermo-Optic MZI + `RAMZI`** | 2D Fourier Heat FEM + `2x2 DC` + All-Pass Ring Assisted Phase Arm | `13.7 s` | $P_\pi = 159.4\text{ mW}$, Thermal $f_{3\mathrm{dB}} = 19.11\text{ kHz}$, `RAMZI` $Q_L = 7,364$ |

---

## 6. Physical Scope Boundaries

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
