# Changelog

All notable changes to **MicroRing Studio** are documented in this file.

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
