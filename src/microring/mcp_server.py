"""Zero-Dependency Model Context Protocol (`MCP`) Stdio Server (`microring mcp`).

Connects `MicroRing Studio` directly to **Claude Code**, **Claude Desktop**, **OpenAI Codex**,
**Cursor**, and **Antigravity** over JSON-RPC 2.0 (`stdin`/`stdout`).

Token-efficiency guarantees:
  - Compact tool schemas across all 15 tools.
  - Heavy 2D FEM, spatial CMT, GPU/AMX TMM spectra, and GDS polygon parsing run 100% locally.
  - Tool responses return single-line scalar KPI JSON (<150 tokens) + local CSV/GDS paths.
"""
from __future__ import annotations

import json
import sys
import traceback

from . import __version__
from .agent_tools import (
    agent_active_tuning,
    agent_advanced_mrr,
    agent_design_mrr,
    agent_design_mzi,
    agent_eom_multiphysics,
    agent_fab_tolerance,
    agent_fit_spectrum,
    agent_high_q_fdtd,
    agent_inspect_gds,
    agent_io_coupler,
    agent_nanoantenna_tcmt,
    agent_2d_material_mrr,
    agent_inverse_adjoint,
    agent_nonlinear_lle,
    agent_pcm_photonics,
    agent_pdk_drc,
    agent_photodetector,
    agent_photonic_computing,
    agent_pic_netlist,
    agent_polarization_mdm,
    agent_prism_optimize,
    agent_saw_sbs,
    agent_status,
    agent_sweep_mrr,
    agent_swg_phc,
    agent_wbg_awg,
    agent_wdm_link,
    agent_width_cutoff,
)

MCP_TOOLS = [
    {
        "name": "microring_status",
        "description": "Probe local GPU/AMX/CPU accelerator & list supported 29 materials and 7 MRR topologies.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "microring_width_cutoff",
        "description": "Sweep waveguide width via local 2D FEM to find exact single-mode cutoff W_SM and n_eff/n_g.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mat": {"type": "string", "description": "Core material key (e.g. Si, Si3N4, 4H-SiC, LN_Xcut, AlN, ScAlN_20)"},
                "h_um": {"type": "number", "description": "Film height [um]"},
                "slab_um": {"type": "number", "description": "Slab height [um] (0 for strip)"},
                "wl_nm": {"type": "number", "description": "Wavelength [nm]"},
                "pol": {"type": "string", "description": "TE0 or TM0"},
                "w_min_um": {"type": "number"},
                "w_max_um": {"type": "number"},
            },
            "required": ["mat", "h_um"],
        },
    },
    {
        "name": "microring_design_mrr",
        "description": "Run local 2D bent FEM + spatial CMT + TMM spectrum for an MRR; optional target FSR, critical coupling & GDS export.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top": {"type": "string", "description": "point_circular_ap/ad, racetrack_ap/ad, pulley_ap/ad, vernier_crow_ad"},
                "mat": {"type": "string"},
                "h_um": {"type": "number"},
                "slab_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "w_bus_um": {"type": "number"},
                "r_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "ls_um": {"type": "number"},
                "theta_deg": {"type": "number"},
                "wl_nm": {"type": "number"},
                "pol": {"type": "string"},
                "target_fsr_ghz": {"type": "number", "description": "Auto-solve radius R for target FSR [GHz]"},
                "apply_cc": {"type": "boolean", "description": "Auto-apply synthesized critical coupling"},
                "export_gds": {"type": "boolean", "description": "Write .gds and KLayout .lyp locally"},
            },
        },
    },
    {
        "name": "microring_sweep_mrr",
        "description": "Sweep gap, pulley angle, straight length, radius, temp, or voltage locally (reusing cached FEM modes).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top": {"type": "string"},
                "mat": {"type": "string"},
                "h_um": {"type": "number"},
                "r_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "w_bus_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "param_name": {"type": "string", "description": "gap_thru_um, pulley_angle_thru_deg, straight_length_um, delta_temp_k, voltage_v"},
                "v_min": {"type": "number"},
                "v_max": {"type": "number"},
                "num_points": {"type": "integer"},
                "export_array_gds": {"type": "boolean"},
            },
            "required": ["param_name", "v_min", "v_max"],
        },
    },
    {
        "name": "microring_inspect_gds",
        "description": "Extract MRR topology, R, W_ring, W_bus, gap, L_s & pulley angle from a local .gds/.oas file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gds_path": {"type": "string"},
                "cell_name": {"type": "string"},
            },
            "required": ["gds_path"],
        },
    },
    {
        "name": "microring_fit_spectrum",
        "description": "Fit a local lab spectrum CSV/TXT via CMT (auto BIC: Lorentzian / Split-Doublet / Fano / Multi-FSR).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "model_type": {"type": "string", "description": "auto, lorentzian, split_doublet, fano, multi_fsr"},
                "known_regime": {"type": "string", "description": "under or over"},
                "ring_length_um": {"type": "number"},
            },
            "required": ["csv_path"],
        },
    },
    {
        "name": "microring_fab_tolerance",
        "description": "Analyze cleanroom lithography over-dose/under-dose (positive_pr vs negative_pr) and CD-bias errors on ER, Q_L, and resonance shift; optional pre-biased OPC .gds array.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resist": {"type": "string", "description": "positive_pr, negative_pr, or direct_cd"},
                "dose_pct": {"type": "number", "description": "Max exposure dose error [+/- %]"},
                "slope_nm_pct": {"type": "number", "description": "CD shift slope [nm per % dose]"},
                "cd_nm": {"type": "number", "description": "Max CD bias [nm] when resist=direct_cd"},
                "top": {"type": "string"},
                "mat": {"type": "string"},
                "h_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "w_bus_um": {"type": "number"},
                "r_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "apply_cc": {"type": "boolean"},
                "export_bias_gds": {"type": "boolean"},
                "corners": {"type": "boolean", "description": "Run multi-parameter foundry process corners (Nominal/SS/FF/SF/FS + wafer Monte-Carlo)"},
                "pdk": {"type": "string", "description": "Foundry PDK profile key (gsip_soi220, cornerstone_soi220, siepic_ebeam_soi220, tfln_xcut_600, scaln_soi_bilayer)"},
            },
        },
    },
    {
        "name": "microring_active_tuning",
        "description": "Compute 2D-FEM TO/EO/Si-PN/Si-PIN tuning efficiency & multi-state spectra (V, I, P, Delta_T, or E-field) linked to cross-section geometry (zero COMSOL), plus optional Lumerical INTERCONNECT export.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mechanism": {"type": "string", "description": "auto, si_pn_depletion, pockels_eo, thermo_optic, si_pin_injection"},
                "drive_type": {"type": "string", "description": "auto, voltage_v, current_ma, power_mw, delta_temp_k, e_field_v_per_um"},
                "bias_values": {"type": "array", "items": {"type": "number"}},
                "top": {"type": "string"},
                "mat": {"type": "string"},
                "h_um": {"type": "number"},
                "slab_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "r_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "heater_gap_um": {"type": "number"},
                "box_thickness_um": {"type": "number"},
                "electrode_gap_um": {"type": "number"},
                "na_p_cm3": {"type": "number"},
                "nd_n_cm3": {"type": "number"},
                "pn_offset_nm": {"type": "number"},
                "apply_cc": {"type": "boolean"},
                "export_interconnect": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_io_coupler",
        "description": "Design Butt/Edge Inverse-Taper Couplers (2D FEM + Poynting overlap + alignment tolerance) or Surface Grating Couplers (Bragg/apodization + Native 2D Full-Wave Maxwell FDFD), plus focusing .gds, 127um FA ring loopback .gds & INTERCONNECT export.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "coupler_type": {"type": "string", "description": "butt_edge or grating_coupler"},
                "mat": {"type": "string", "description": "Core material (Si, Si3N4, 4H-SiC, LN_Xcut, AlN, Ge)"},
                "wl_nm": {"type": "number"},
                "pol": {"type": "string"},
                "h_um": {"type": "number"},
                "etch_depth_um": {"type": "number"},
                "w_bus_um": {"type": "number"},
                "w_tip_um": {"type": "number"},
                "taper_length_um": {"type": "number"},
                "taper_profile": {"type": "string", "description": "adiabatic, parabolic, linear"},
                "fiber_preset": {"type": "string", "description": "SMF28_1550, SMF28_1310, UHNA_3, Lensed_2p5um, DFB_Laser_Butt"},
                "fiber_angle_deg": {"type": "number"},
                "apodized": {"type": "boolean"},
                "focusing": {"type": "boolean"},
                "box_um": {"type": "number"},
                "export_gds": {"type": "boolean"},
                "include_loopback": {"type": "boolean"},
                "export_interconnect": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_high_q_fdtd",
        "description": "Diagnose High-Q FDTD ring-down non-convergence (tau_ph, required 1e-5 shutoff time, truncated FDTD sinc ripples vs exact infinite-roundtrip TMM) and export non-resonant coupler-only 3D-FDTD cutout .gds/.lsf.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top": {"type": "string"},
                "mat": {"type": "string"},
                "h_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "w_bus_um": {"type": "number"},
                "r_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "alpha_loss_db_per_cm": {"type": "number"},
                "apply_cc": {"type": "boolean"},
                "export_cutout": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_design_mzi",
        "description": "Design & simulate a Mach-Zehnder Interferometer (MZI) with selectable 50:50 splitters (directional_coupler, mmi_2x2, mmi_1x2, y_branch), active tuning (Si PN, Pockels EO, TiN TO, PIN; push_pull or single_arm -> V_pi*L, V_pi, P_pi, 3-dB BW), RAMZI, .GDS mask, and Lumerical INTERCONNECT export without 3D FDTD.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "splitter": {"type": "string", "description": "directional_coupler | mmi_2x2 | mmi_1x2 | y_branch"},
                "mat": {"type": "string", "description": "Core material (Si, Si3N4, LN_Xcut, AlN, 4H-SiC, etc.)"},
                "wl_nm": {"type": "number"},
                "pol": {"type": "string"},
                "h_um": {"type": "number"},
                "slab_um": {"type": "number"},
                "w1_um": {"type": "number"},
                "l1_um": {"type": "number"},
                "dl_um": {"type": "number"},
                "fsr_nm": {"type": "number"},
                "dc_gap_um": {"type": "number"},
                "mmi_width_um": {"type": "number"},
                "y_apex_gap_um": {"type": "number"},
                "ramzi_enabled": {"type": "boolean"},
                "mechanism": {"type": "string", "description": "auto | si_pn_depletion | pockels_eo | thermo_optic | si_pin_injection"},
                "drive_type": {"type": "string", "description": "auto | voltage_v | current_ma | power_mw | delta_temp_k | e_field_v_per_um"},
                "drive_scheme": {"type": "string", "description": "auto | push_pull | single_arm"},
                "bias_values": {"type": "string", "description": "Comma-separated bias points, e.g. '-1,-2,-3,-4'"},
                "export_gds": {"type": "boolean"},
                "export_interconnect": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_eom_multiphysics",
        "description": "Compute 2D Quasi-TEM RF EOM Electrodes (gsg_cpw, t_rail_cpw, gs_coplanar, vertical_microstrip, lumped_ring_gsg -> Z0, n_RF vs n_g, S21_EO(f), V_pi*L, Touchstone .s2p & RF .GDS), 3D Crystal-Tensor EO Wafer Orientation & 0-360 deg Azimuth Finder (wafer_orientation -> optimal azimuth, signed r_eff(phi), SEMI wafer flat, ring split push-pull efficiency), 2D Nonlinear Poisson-Boltzmann Si PN/PIN Doping TCAD (lateral_pn, l_shaped_pn, u_shaped_pn, pin_diode -> Cj(V), Rs, V_pi*L, FCA & implant .GDS), or 2D Steady+Crank-Nicolson Transient Thermal FEM (top_heater, deep_trench, undercut_suspended, doped_si_rib -> P_pi, t_rise/fall, crosstalk & heater+trench .GDS).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "rf_electrode | wafer_orientation | pn_doping | thermal_fem"},
                "topology": {"type": "string", "description": "gsg_cpw | t_rail_cpw | gs_coplanar | vertical_microstrip | lumped_ring_gsg"},
                "wafer_cut": {"type": "string", "description": "auto | x_cut_ln | y_cut_ln | z_cut_ln | x_cut_lt | z_cut_lt | a_cut_bto_si | c_cut_bto | inplane_poled_pzt | c_plane_6mm | a_plane_6mm | cubic_001_43m | cubic_110_43m"},
                "azimuth_deg": {"type": "number", "description": "In-plane waveguide propagation azimuth phi [deg] relative to wafer primary crystal axis"},
                "electrode_field_axis": {"type": "string", "description": "auto | horizontal_ex | vertical_ey"},
                "junction_type": {"type": "string", "description": "lateral_pn | l_shaped_pn | u_shaped_pn | pin_diode"},
                "structure": {"type": "string", "description": "top_heater | deep_trench | undercut_suspended | doped_si_rib"},
                "mat": {"type": "string", "description": "Core material (LN_Xcut, LT_Xcut, BTO, PZT, Si, LN_Zcut, LT_Zcut, AlN, ScAlN_20, 4H-SiC, 3C-SiC, Si3N4)"},
                "substrate": {"type": "string", "description": "hr_si | quartz | undercut_si | std_si | sapphire"},
                "metal": {"type": "string", "description": "Au | Cu | Al"},
                "t_metal_um": {"type": "number"},
                "wl_nm": {"type": "number"},
                "h_um": {"type": "number"},
                "slab_um": {"type": "number"},
                "w_um": {"type": "number"},
                "l_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "ws_um": {"type": "number"},
                "l_rail_um": {"type": "number"},
                "t_period_um": {"type": "number"},
                "t_cap_w_um": {"type": "number"},
                "f_max_ghz": {"type": "number"},
                "na_core_cm3": {"type": "number"},
                "nd_core_cm3": {"type": "number"},
                "pn_offset_nm": {"type": "number"},
                "pn_sink": {"type": "boolean", "description": "Include parasitic W-VIAC/M1 PN contact heat-sinking in 2D Thermal FEM"},
                "export_gds": {"type": "boolean"},
                "export_bundle": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_pdk_drc",
        "description": "Foundry PDK layout verification & native cutline solver (zero Lumerical): build DRC/LVS-clean Co-Integrated PN+TiN Heater Ring Modulator .gds/.lyp/.lyrdb, run headless polygon DRC + 3D multi-metal Union-Find LVS on any .gds/.oas, slice .gds along a 2D cutline to drive native 2D Maxwell/PN/Thermal FEM, or export NDA-safe PDK CSV tables.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "build_pn_heater_ring | verify_gds | cutline_solve | export_csv"},
                "pdk": {"type": "string", "description": "gsip_soi220 | cornerstone_soi220 | siepic_ebeam_soi220 | tfln_xcut_600 | scaln_soi_bilayer | microring_generic"},
                "gds_path": {"type": "string"},
                "pdk_csv_dir": {"type": "string"},
                "r_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "pn_offset_nm": {"type": "number"},
                "wl_nm": {"type": "number"},
            },
        },
    },
    {
        "name": "microring_prism_optimize",
        "description": "PRISM-style Physics Closure Gate (cavity Q-budget decomposition vs FWHM Q_L, TMM passivity, core/slab perturbation split) & Prediction-Gated Closed-Loop Real-2D-FEM Optimizer with online Bayesian surrogate and auditable journal.jsonl.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "optimize | closure_audit"},
                "problem": {"type": "string", "description": "pn_ring_modulator | high_er_pulley_mrr | low_vpil_eom"},
                "max_iterations": {"type": "integer"},
                "seed": {"type": "integer"},
                "mat": {"type": "string"},
                "r_um": {"type": "number"},
                "gap_um": {"type": "number"},
            },
        },
    },
    {
        "name": "microring_wdm_link",
        "description": "Multi-channel Microring WDM Optical Link, NRZ/PAM4 PRBS Eye-Diagram, BER & Closed-Loop PI TiN Heater Wavelength-Locking Simulator driven 100% by native 2D-FEM device physics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string", "description": "nominal | thermal_drift | heater_relocked | dense_crosstalk | low_laser_power"},
                "n_ch": {"type": "integer"},
                "rate_gbps": {"type": "number"},
                "format": {"type": "string", "description": "nrz | pam4"},
                "spacing_ghz": {"type": "number"},
                "dt_k": {"type": "number"},
                "r_um": {"type": "number"},
                "gap_um": {"type": "number"},
                "vpp": {"type": "number"},
            },
        },
    },
    {
        "name": "microring_nonlinear_lle",
        "description": "First-principles Nonlinear Photonics & LLE Microcomb Engine: broadband 2D-FEM D_int(mu) + vector gamma_NL/P_th, GPU/CPU Split-Step Fourier LLE (solitons, dispersive waves, Raman, thermal, Si TPA/FCA), cross-band FWM-BS, Chi(2) SHG, and photothermal-Kerr bistability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "lle | dispersion | fwm_bs | shg | bistability"},
                "mat": {"type": "string", "description": "Core material (Si3N4, 4H-SiC, LN_Xcut, Si, AlN, ScAlN_20, Hydex)"},
                "h_um": {"type": "number"},
                "slab_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "r_um": {"type": "number"},
                "wl_nm": {"type": "number"},
                "mode": {"type": "string", "description": "single_soliton | soliton_sweep | dark_pulse"},
                "p_mw": {"type": "number", "description": "Pump power [mW]"},
                "include_raman": {"type": "boolean"},
                "include_thermal": {"type": "boolean"},
                "include_tpa_fca": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_advanced_mrr",
        "description": "High-order N-ring CROW/SCISSOR filters & group delay, Euler spiral bends, biosensing/athermal, large-signal TD-CMT modulator, MZI-MRR, photonic molecule/EP, Quantum SFWM/squeezing & active MRR lasers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "high_order_filter | crow | scissor | euler_bend | sensor_athermal | td_cmt_modulator | mzi_mrr | photonic_molecule | quantum_sfwm | mrr_laser"},
                "mat": {"type": "string"},
                "r_um": {"type": "number"},
                "n": {"type": "integer", "description": "Filter order N (2..10)"},
                "target_bw_ghz": {"type": "number"},
                "bitrate_gbps": {"type": "number"},
                "modulation_type": {"type": "string", "description": "detuning | coupling"},
                "gain_medium": {"type": "string", "description": "Er_TFLN | Er_Si3N4 | InGaAsP_MQW"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_swg_phc",
        "description": "Subwavelength Grating (SWG) 2nd-order Rytov EMT + 1D Bloch-Floquet microring & Photonic Crystal Ring (PhCR) selective-mode-splitting Bragg resonator + GDS export.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "swg | phcr"},
                "mat": {"type": "string"},
                "h_um": {"type": "number"},
                "w_ring_um": {"type": "number"},
                "r_um": {"type": "number"},
                "swg_period_nm": {"type": "number"},
                "duty_cycle": {"type": "number"},
                "modulation_amplitude_nm": {"type": "number"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_saw_sbs",
        "description": "First-principles 2D Solid Mechanics / Piezoelectric SAW-Lamb acousto-optic modulator FEM (v_open/v_short, K_eff^2, g0, V_pi) & 2D-FEM Stimulated Brillouin Scattering (SBS) gain G_B.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "saw | sbs"},
                "mat": {"type": "string", "description": "AlN | ScAlN_20 | ScAlN_30 | LN_Xcut | LN_Zcut | Si | Si3N4 | 4H-SiC"},
                "substrate_material": {"type": "string", "description": "Si | Sapphire | Air"},
                "h_um": {"type": "number"},
                "idt_period_um": {"type": "number"},
                "acoustic_mode": {"type": "string", "description": "rayleigh_or_s0 | lamb_s0"},
                "sbs_direction": {"type": "string", "description": "backward | forward"},
            },
        },
    },
    {
        "name": "microring_nanoantenna_tcmt",
        "description": "Energy-conserving multi-mode Nanoantenna & Metasurface TCMT: MIM perfect absorber, Fano/EIT/qBIC, Mid-IR SEIRA molecular vibrational coupling & non-Hermitian Exceptional Point.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "regime": {"type": "string", "description": "mim_absorber | fano_eit_qbic | seira_molecular | metasurface_ep"},
                "num_ports": {"type": "integer"},
                "nu0_bright_cm1": {"type": "number"},
                "gamma_rad_bright_cm1": {"type": "number"},
                "gamma_abs_bright_cm1": {"type": "number"},
                "coupling_mu_cm1": {"type": "number"},
                "analyte_preset": {"type": "string", "description": "PMMA | Protein_Amide | CO2 | CH4 | CBP_Polystyrene"},
            },
        },
    },
    {
        "name": "microring_2d_material_mrr",
        "description": "2D Materials on Microring Resonators: Graphene (exact finite-T Kubo SLG/DLG + quantum capacitance), monolayer TMDs (WS2/MoS2/WSe2/MoSe2 exciton gating), Mid-IR hyperbolic hBN & Black Phosphorus.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "material_2d": {"type": "string", "description": "Graphene | WS2 | MoS2 | WSe2 | MoSe2 | hBN | Black_Phosphorus"},
                "capacitor_type": {"type": "string", "description": "SLG | DLG"},
                "mat": {"type": "string", "description": "Core waveguide material (Si | Si3N4 | LN_Xcut | Ge)"},
                "wl_nm": {"type": "number"},
                "r_um": {"type": "number"},
                "spacer_ox_nm": {"type": "number"},
            },
        },
    },
    {
        "name": "microring_photodetector",
        "description": "First-principles 2D-FEM Ge-on-Si PIN/MUTC waveguide photodetector (R, QE, f_3dB, I_dark, GDS) & O-E-O PD->MRM photonic neuron.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "waveguide_pd | oeo_neuron"},
                "pd_architecture": {"type": "string", "description": "vertical_pin | lateral_pin | mutc"},
                "ge_length_um": {"type": "number"},
                "wavelength_nm": {"type": "number"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_polarization_mdm",
        "description": "First-principles 2D-FEM Polarization Splitter-Rotator (PSR TM0<->TE1 hybridization), PBS & MDM (TE0<->TE1/TE2) + GDS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "psr | mdm_te1 | mdm_te2 | pbs_dc"},
                "taper_length_um": {"type": "number"},
                "coupler_gap_um": {"type": "number"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_pcm_photonics",
        "description": "Non-volatile Phase-Change Material (Sb2Se3/Sb2S3/GSST/GST/VO2) multi-level 2D-FEM MRR switch & 2D transient Crank-Nicolson SET/RESET thermal pulse FEM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "pcm_mrr | pcm_pulse"},
                "pcm_material": {"type": "string", "description": "Sb2Se3 | Sb2S3 | GSST | GST | VO2"},
                "pcm_thickness_nm": {"type": "number"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_photonic_computing",
        "description": "Photonic Neural Network matrix engine: N x M MRR Weight Bank (thermal crosstalk Green's matrix & ENOB) & exact N x N Clements/Reck unitary MZI mesh + GDS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "mrr_weight_bank | clements | reck"},
                "mesh_size_n": {"type": "integer"},
                "num_rows": {"type": "integer"},
                "num_cols": {"type": "integer"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_wbg_awg",
        "description": "2D-FEM Waveguide Bragg Gratings (uniform/apodized/pi-shifted/chirped) & Rowland-circle Arrayed Waveguide Grating (AWG 1x8/1x16 WDM) + GDS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "wbg | awg"},
                "grating_type": {"type": "string", "description": "uniform | gaussian_apodized | pi_phase_shifted | chirped"},
                "num_channels": {"type": "integer"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_pic_netlist",
        "description": "Native license-free frequency-domain PIC S-matrix circuit netlist solver (Schur-complement arbitrary graph, passivity check & Touchstone .sNp export).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {"type": "string", "description": "ramzi_linearized | mzi_filter | wbg_fabry_perot | dual_ring_vernier"},
                "export_snp": {"type": "boolean"},
            },
        },
    },
    {
        "name": "microring_inverse_adjoint",
        "description": "Native 2D-FDFD exact Maxwell adjoint-gradient inverse photonic shape optimizer (ultra-compact <0.1 dB Y-branch) + foundry GDS export.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "design_length_um": {"type": "number"},
                "max_iterations": {"type": "integer"},
                "export_gds": {"type": "boolean"},
            },
        },
    },
]

_DISPATCH = {
    "microring_status": lambda _args: agent_status(),
    "microring_width_cutoff": agent_width_cutoff,
    "microring_design_mrr": agent_design_mrr,
    "microring_sweep_mrr": agent_sweep_mrr,
    "microring_inspect_gds": agent_inspect_gds,
    "microring_fit_spectrum": agent_fit_spectrum,
    "microring_fab_tolerance": agent_fab_tolerance,
    "microring_active_tuning": agent_active_tuning,
    "microring_io_coupler": agent_io_coupler,
    "microring_high_q_fdtd": agent_high_q_fdtd,
    "microring_design_mzi": agent_design_mzi,
    "microring_eom_multiphysics": agent_eom_multiphysics,
    "microring_pdk_drc": agent_pdk_drc,
    "microring_prism_optimize": agent_prism_optimize,
    "microring_wdm_link": agent_wdm_link,
    "microring_nonlinear_lle": agent_nonlinear_lle,
    "microring_advanced_mrr": agent_advanced_mrr,
    "microring_swg_phc": agent_swg_phc,
    "microring_saw_sbs": agent_saw_sbs,
    "microring_nanoantenna_tcmt": agent_nanoantenna_tcmt,
    "microring_2d_material_mrr": agent_2d_material_mrr,
    "microring_photodetector": agent_photodetector,
    "microring_polarization_mdm": agent_polarization_mdm,
    "microring_pcm_photonics": agent_pcm_photonics,
    "microring_photonic_computing": agent_photonic_computing,
    "microring_wbg_awg": agent_wbg_awg,
    "microring_pic_netlist": agent_pic_netlist,
    "microring_inverse_adjoint": agent_inverse_adjoint,
}


def handle_mcp_request(msg: dict) -> dict | None:
    """Process a single JSON-RPC 2.0 MCP message and return response dict (or None for notifications)."""
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if req_id is None:
        # JSON-RPC notification (e.g., notifications/initialized)
        return None

    if method == "initialize":
        proto = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "microring-studio", "version": __version__},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        if name not in _DISPATCH:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool '{name}'"},
            }
        try:
            res = _DISPATCH[name](args)
            compact = json.dumps(res, separators=(",", ":"))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": compact}], "isError": False},
            }
        except Exception as exc:
            err_txt = json.dumps({"error": str(exc), "trace": traceback.format_exc()[-300:]}, separators=(",", ":"))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": err_txt}], "isError": True},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method '{method}' not supported"},
    }


def run_mcp_stdio() -> None:
    """Run JSON-RPC 2.0 line-delimited MCP server over stdin/stdout."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_mcp_request(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_stdio()
