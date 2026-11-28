"""Command-Line Interface (`microring`) for `MicroRing Studio`.

Supports human interactive modes AND ultra-low-token AI agent / MCP modes:
  microring                       # Launch native desktop UI window (falls back to browser)
  microring browser               # Open UI in Chrome / Safari / Edge
  microring check                 # Print detected GPU/AMX/CPU engine + 100k-pt benchmark
  microring analyze-gds ring.gds  # Extract MRR topology & dimensions from any .gds/.oas
  microring chat "400 nm Si3N4 pulley all-pass, FSR 150 GHz"
  microring mcp                   # Run stdio MCP server for Claude Code / Codex / Antigravity
  microring agent <tool> ...      # Token-minimal (<150 tokens) local-compute CLI for AI agents
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .agent_tools import (
    agent_active_tuning,
    agent_advanced_mrr,
    agent_design_mrr,
    agent_design_mzi,
    agent_eam_soa,
    agent_eom_multiphysics,
    agent_eom_rf_eye,
    agent_eocomb_opo_ssb,
    agent_fab_tolerance,
    agent_fit_spectrum,
    agent_gnlse_ppln,
    agent_high_q_fdtd,
    agent_inspect_gds,
    agent_io_coupler,
    agent_isolator_mems,
    agent_mrr_mwp,
    agent_nanoantenna_tcmt,
    agent_2d_material_mrr,
    agent_inverse_adjoint,
    agent_mrr_ep,
    agent_nonlinear_lle,
    agent_oam_optomech_topo,
    agent_opa_lidar,
    agent_pcm_photonics,
    agent_pdk_drc,
    agent_photodetector,
    agent_photonic_computing,
    agent_pic_netlist,
    agent_polarization_mdm,
    agent_prism_optimize,
    agent_reservoir_switch_sensing,
    agent_routing_crossings,
    agent_saw_sbs,
    agent_slab_diffractive,
    agent_status,
    agent_sweep_mrr,
    agent_swg_phc,
    agent_vernier_sil_gyro,
    agent_wbg_awg,
    agent_wdm_link,
    agent_width_cutoff,
)
from .copilot import create_initial_session, process_chat_turn
from .desktop_app import launch_studio
from .gds_bridge import analyze_uploaded_gds
from .hardware_accel import detect_hardware_profile, run_compute_benchmark
from .mcp_server import run_mcp_stdio


def _print_compact(data: dict) -> None:
    print(json.dumps(data, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="microring",
        description=f"MicroRing Studio v{__version__} - License-Free Multi-Waveband Photonics Eigenmode, 2D-FEM Active Tuning, CMT, GDSII & AI-Agent MCP Suite",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_app = sub.add_parser("app", help="Launch native desktop UI window")
    p_app.add_argument("--port", type=int, default=None)

    p_web = sub.add_parser("browser", help="Launch UI in default web browser")
    p_web.add_argument("--port", type=int, default=None)

    sub.add_parser("check", help="Inspect hardware accelerator & run 100k-pt TMM benchmark")
    sub.add_parser("mcp", help="Start JSON-RPC 2.0 MCP server on stdin/stdout (for Claude / Codex)")

    p_gds = sub.add_parser("analyze-gds", help="Extract ring/racetrack/pulley dimensions from a .gds/.oas file")
    p_gds.add_argument("gds_file", type=str, help="Path to .gds or .oas layout")

    p_chat = sub.add_parser("chat", help="Run a single conversational design prompt in terminal")
    p_chat.add_argument("prompt", type=str, help="Natural-language design specification")

    # Token-minimal subcommands for Codex / Claude Code / Antigravity
    p_ag = sub.add_parser("agent", help="Ultra-compact JSON tools (<150 tokens) executing 100% locally")
    ag_sub = p_ag.add_subparsers(dest="agent_cmd", required=True)

    ag_sub.add_parser("status", help="Compact local hardware & material/topology keys")

    p_aw = ag_sub.add_parser("width-cutoff", help="Local 2D FEM single-mode width sweep")
    p_aw.add_argument("--mat", default="Si3N4")
    p_aw.add_argument("--h", type=float, default=0.40, dest="h_um")
    p_aw.add_argument("--slab", type=float, default=0.0, dest="slab_um")
    p_aw.add_argument("--wl", type=float, default=1550.0, dest="wl_nm")
    p_aw.add_argument("--pol", default="TE0")
    p_aw.add_argument("--wmin", type=float, default=0.50, dest="w_min_um")
    p_aw.add_argument("--wmax", type=float, default=2.00, dest="w_max_um")
    p_aw.add_argument("--json", type=str, default=None, dest="raw_json")

    p_ad = ag_sub.add_parser("design", help="Local 2D FEM + CMT MRR synthesis (+ optional FSR/CC/GDS)")
    p_ad.add_argument("--top", default="point_circular_ap")
    p_ad.add_argument("--mat", default="Si3N4")
    p_ad.add_argument("--clad", default=None, dest="clad_material")
    p_ad.add_argument("--top-clad", default=None, dest="top_clad_material")
    p_ad.add_argument("--h", type=float, default=None, dest="h_um")
    p_ad.add_argument("--slab", type=float, default=None, dest="slab_um")
    p_ad.add_argument("--w-ring", type=float, default=None, dest="w_ring_um")
    p_ad.add_argument("--w-bus", type=float, default=None, dest="w_bus_um")
    p_ad.add_argument("--r", type=float, default=None, dest="r_um")
    p_ad.add_argument("--gap", type=float, default=None, dest="gap_um")
    p_ad.add_argument("--ls", type=float, default=None, dest="ls_um")
    p_ad.add_argument("--theta", type=float, default=None, dest="theta_deg")
    p_ad.add_argument("--wl", type=float, default=None, dest="wl_nm")
    p_ad.add_argument("--pol", default=None)
    p_ad.add_argument("--loss", type=float, default=None, dest="loss_db_cm", help="Propagation loss [dB/cm]")
    p_ad.add_argument("--fsr", type=float, default=None, dest="target_fsr_ghz")
    p_ad.add_argument("--cc", action="store_true", dest="apply_cc")
    p_ad.add_argument("--gds", action="store_true", dest="export_gds")
    p_ad.add_argument("--json", type=str, default=None, dest="raw_json")

    p_as = ag_sub.add_parser("sweep", help="Local multi-point MRR sweep (cached 2D FEM modes)")
    p_as.add_argument("--param", required=False, default="gap_thru_um", dest="param_name")
    p_as.add_argument("--min", type=float, default=0.16, dest="v_min")
    p_as.add_argument("--max", type=float, default=0.32, dest="v_max")
    p_as.add_argument("--n", type=int, default=7, dest="num_points")
    p_as.add_argument("--top", default="point_circular_ap")
    p_as.add_argument("--mat", default="Si3N4")
    p_as.add_argument("--h", type=float, default=None, dest="h_um")
    p_as.add_argument("--r", type=float, default=None, dest="r_um")
    p_as.add_argument("--array-gds", action="store_true", dest="export_array_gds")
    p_as.add_argument("--json", type=str, default=None, dest="raw_json")

    p_ai = ag_sub.add_parser("inspect-gds", help="Compact GDS polygon geometry extraction")
    p_ai.add_argument("gds_path", type=str)
    p_ai.add_argument("--cell", default=None, dest="cell_name")

    p_af = ag_sub.add_parser("fit", help="Compact local lab spectrum CMT fit")
    p_af.add_argument("csv_path", type=str)
    p_af.add_argument("--model", default="auto", dest="model_type")
    p_af.add_argument("--regime", default=None, dest="known_regime")
    p_af.add_argument("--lrt", type=float, default=250.0, dest="ring_length_um")

    p_at = ag_sub.add_parser("fab-tolerance", help="Cleanroom positive/negative PR dose & CD-bias error analysis")
    p_at.add_argument("--resist", default="positive_pr", dest="resist_tone", help="positive_pr | negative_pr | direct_cd")
    p_at.add_argument("--dose-pct", type=float, default=15.0, dest="max_dose_error_pct")
    p_at.add_argument("--slope", type=float, default=None, dest="dose_slope_nm_per_pct")
    p_at.add_argument("--cd-nm", type=float, default=30.0, dest="max_cd_bias_nm")
    p_at.add_argument("--prox", type=float, default=None, dest="proximity_gap_factor")
    p_at.add_argument("--dh", type=float, default=0.0, dest="delta_h_film_nm")
    p_at.add_argument("--top", default="point_circular_ap")
    p_at.add_argument("--mat", default="Si3N4")
    p_at.add_argument("--h", type=float, default=None, dest="h_um")
    p_at.add_argument("--w-ring", type=float, default=None, dest="w_ring_um")
    p_at.add_argument("--w-bus", type=float, default=None, dest="w_bus_um")
    p_at.add_argument("--r", type=float, default=None, dest="r_um")
    p_at.add_argument("--gap", type=float, default=None, dest="gap_um")
    p_at.add_argument("--wl", type=float, default=None, dest="wl_nm")
    p_at.add_argument("--cc", action="store_true", dest="apply_cc")
    p_at.add_argument("--corners", action="store_true", dest="corners", help="Run Nominal/SS/FF/SF/FS foundry process corners + wafer Monte-Carlo")
    p_at.add_argument("--pdk", type=str, default=None, dest="pdk", help="Foundry PDK profile key")
    p_at.add_argument("--bias-gds", action="store_true", dest="export_bias_gds")
    p_at.add_argument("--json", type=str, default=None, dest="raw_json")

    p_tu = ag_sub.add_parser("active-tuning", help="2D-FEM TO/EO/Si-PN/Si-PIN tuning (V/I/T/E) & Lumerical INTERCONNECT export")
    p_tu.add_argument("--mech", default="auto", dest="mechanism", help="auto | si_pn_depletion | pockels_eo | thermo_optic | si_pin_injection")
    p_tu.add_argument("--drive", default="auto", dest="drive_type", help="auto | voltage_v | current_ma | power_mw | delta_temp_k | e_field_v_per_um")
    p_tu.add_argument("--biases", default=None, dest="bias_values", help="Comma-separated values, e.g. '0,-1,-2,-4,-6'")
    p_tu.add_argument("--top", default="point_circular_ap")
    p_tu.add_argument("--mat", default="Si")
    p_tu.add_argument("--h", type=float, default=None, dest="h_um")
    p_tu.add_argument("--slab", type=float, default=None, dest="slab_um")
    p_tu.add_argument("--w-ring", type=float, default=None, dest="w_ring_um")
    p_tu.add_argument("--w-bus", type=float, default=None, dest="w_bus_um")
    p_tu.add_argument("--r", type=float, default=None, dest="r_um")
    p_tu.add_argument("--gap", type=float, default=None, dest="gap_um")
    p_tu.add_argument("--wl", type=float, default=None, dest="wl_nm")
    p_tu.add_argument("--heater-gap", type=float, default=0.90, dest="heater_gap_um")
    p_tu.add_argument("--box", type=float, default=2.00, dest="box_thickness_um")
    p_tu.add_argument("--eo-gap", type=float, default=None, dest="electrode_gap_um")
    p_tu.add_argument("--na", type=float, default=5.0e17, dest="na_p_cm3")
    p_tu.add_argument("--nd", type=float, default=5.0e17, dest="nd_n_cm3")
    p_tu.add_argument("--pn-offset", type=float, default=25.0, dest="pn_offset_nm")
    p_tu.add_argument("--cc", action="store_true", dest="apply_cc")
    p_tu.add_argument("--interconnect", action="store_true", dest="export_interconnect")
    p_tu.add_argument("--json", type=str, default=None, dest="raw_json")

    p_io = ag_sub.add_parser("io-coupler", help="Design Butt/Edge Coupler or 2D-FDFD Grating Coupler (<150 tokens)")
    p_io.add_argument("--type", type=str, default="grating_coupler", dest="coupler_type")
    p_io.add_argument("--mat", type=str, default="Si", dest="core_material")
    p_io.add_argument("--wl", type=float, default=1550.0, dest="wavelength_nm")
    p_io.add_argument("--pol", type=str, default="TE0", dest="polarization")
    p_io.add_argument("--h", type=float, default=0.22, dest="film_height_um")
    p_io.add_argument("--etch", type=float, default=None, dest="etch_depth_um")
    p_io.add_argument("--w-bus", type=float, default=0.48, dest="w_bus_um")
    p_io.add_argument("--w-tip", type=float, default=None, dest="w_tip_um")
    p_io.add_argument("--l-taper", type=float, default=180.0, dest="taper_length_um")
    p_io.add_argument("--fiber", type=str, default=None, dest="fiber_preset")
    p_io.add_argument("--angle", type=float, default=10.0, dest="fiber_angle_deg")
    p_io.add_argument("--box", type=float, default=2.0, dest="box_thickness_um")
    p_io.add_argument("--uniform", action="store_false", dest="apodized")
    p_io.add_argument("--linear-gc", action="store_false", dest="focusing")
    p_io.add_argument("--gds", action="store_true", dest="export_gds")
    p_io.add_argument("--loopback", action="store_true", dest="include_loopback")
    p_io.add_argument("--interconnect", action="store_true", dest="export_interconnect")
    p_io.add_argument("--json", type=str, default=None, dest="raw_json")

    p_hq = ag_sub.add_parser("high-q", help="Diagnose High-Q FDTD ring-down vs exact infinite-RT TMM (<140 tokens)")
    p_hq.add_argument("--top", type=str, default="pulley_ap", dest="topology")
    p_hq.add_argument("--mat", type=str, default="Si3N4", dest="core_material")
    p_hq.add_argument("--h", type=float, default=0.40, dest="film_height_um")
    p_hq.add_argument("--w-ring", type=float, default=1.20, dest="ring_width_um")
    p_hq.add_argument("--w-bus", type=float, default=0.95, dest="bus_width_um")
    p_hq.add_argument("--r", type=float, default=80.0, dest="ring_radius_um")
    p_hq.add_argument("--gap", type=float, default=0.35, dest="gap_thru_um")
    p_hq.add_argument("--loss", type=float, default=0.08, dest="alpha_loss_db_per_cm")
    p_hq.add_argument("--cc", action="store_true", dest="apply_cc")
    p_hq.add_argument("--cutout", action="store_true", dest="export_cutout")
    p_hq.add_argument("--json", type=str, default=None, dest="raw_json")

    p_mzi = ag_sub.add_parser("mzi", help="Design MZI (DC / MMI / Y-Branch) + Active Tuning + GDS/INTERCONNECT (<150 tokens)")
    p_mzi.add_argument("--splitter", type=str, default="directional_coupler", dest="splitter_type", help="directional_coupler | mmi_2x2 | mmi_1x2 | y_branch")
    p_mzi.add_argument("--mat", type=str, default="Si", dest="core_material")
    p_mzi.add_argument("--wl", type=float, default=1550.0, dest="wavelength_nm")
    p_mzi.add_argument("--pol", type=str, default="TE0", dest="polarization")
    p_mzi.add_argument("--h", type=float, default=0.22, dest="film_height_um")
    p_mzi.add_argument("--slab", type=float, default=0.0, dest="slab_height_um")
    p_mzi.add_argument("--w1", type=float, default=0.48, dest="arm1_width_um")
    p_mzi.add_argument("--w2", type=float, default=None, dest="arm2_width_um")
    p_mzi.add_argument("--l-arm", type=float, default=500.0, dest="arm1_length_um")
    p_mzi.add_argument("--dl", type=float, default=80.0, dest="delta_length_um")
    p_mzi.add_argument("--fsr-nm", type=float, default=None, dest="target_fsr_nm")
    p_mzi.add_argument("--dc-gap", type=float, default=0.20, dest="dc_gap_um")
    p_mzi.add_argument("--mmi-w", type=float, default=None, dest="mmi_width_um")
    p_mzi.add_argument("--y-gap", type=float, default=0.13, dest="y_apex_gap_um")
    p_mzi.add_argument("--ramzi", action="store_true", dest="ramzi_enabled")
    p_mzi.add_argument("--mech", type=str, default="auto", dest="mechanism")
    p_mzi.add_argument("--drive", type=str, default="auto", dest="drive_type")
    p_mzi.add_argument("--scheme", type=str, default="auto", dest="drive_scheme")
    p_mzi.add_argument("--biases", type=str, default=None, dest="bias_values")
    p_mzi.add_argument("--eo-gap", type=float, default=None, dest="electrode_gap_um")
    p_mzi.add_argument("--passive-only", action="store_false", dest="include_active_tuning")
    p_mzi.add_argument("--gds", action="store_true", dest="export_gds")
    p_mzi.add_argument("--interconnect", action="store_true", dest="export_interconnect")
    p_mzi.add_argument("--json", type=str, default=None, dest="raw_json")

    p_eom = ag_sub.add_parser("eom", help="2D-FEM RF EOM Electrode, EO Wafer Orientation, Si PN Doping TCAD, or Thermal FEM (<150 tokens)")
    p_eom.add_argument("--mode", type=str, default="rf_electrode", help="rf_electrode | wafer_orientation | pn_doping | thermal_fem")
    p_eom.add_argument("--topology", type=str, default=None, help="gsg_cpw | t_rail_cpw | gs_coplanar | vertical_microstrip | lumped_ring_gsg")
    p_eom.add_argument("--wafer-cut", type=str, default=None, dest="wafer_cut", help="auto | x_cut_ln | y_cut_ln | z_cut_ln | x_cut_lt | z_cut_lt | a_cut_bto_si | c_cut_bto | inplane_poled_pzt | c_plane_6mm | a_plane_6mm | cubic_001_43m | cubic_110_43m")
    p_eom.add_argument("--azimuth", type=float, default=None, dest="azimuth_deg", help="Waveguide propagation azimuth phi [deg]")
    p_eom.add_argument("--pol", type=str, default=None, dest="pol", help="TE0 | TM0")
    p_eom.add_argument("--field-axis", type=str, default=None, dest="electrode_field_axis", help="auto | horizontal_ex | vertical_ey")
    p_eom.add_argument("--junction", type=str, default=None, dest="junction_type", help="lateral_pn | l_shaped_pn | u_shaped_pn | pin_diode")
    p_eom.add_argument("--structure", type=str, default=None, dest="thermal_structure", help="top_heater | deep_trench | undercut_suspended | doped_si_rib")
    p_eom.add_argument("--mat", type=str, default=None)
    p_eom.add_argument("--substrate", type=str, default=None)
    p_eom.add_argument("--metal", type=str, default=None)
    p_eom.add_argument("--t-metal", type=float, default=None, dest="t_metal_um")
    p_eom.add_argument("--wl", type=float, default=None, dest="wl_nm")
    p_eom.add_argument("--h", type=float, default=None, dest="h_um")
    p_eom.add_argument("--slab", type=float, default=None, dest="slab_um")
    p_eom.add_argument("--w", type=float, default=None, dest="w_um")
    p_eom.add_argument("--l", type=float, default=None, dest="l_um")
    p_eom.add_argument("--gap", type=float, default=None, dest="gap_um")
    p_eom.add_argument("--ws", type=float, default=None, dest="ws_um")
    p_eom.add_argument("--l-rail", type=float, default=None, dest="l_rail_um")
    p_eom.add_argument("--t-period", type=float, default=None, dest="t_period_um")
    p_eom.add_argument("--t-cap-w", type=float, default=None, dest="t_cap_w_um")
    p_eom.add_argument("--fmax", type=float, default=None, dest="f_max_ghz")
    p_eom.add_argument("--na", type=float, default=None, dest="na_core_cm3")
    p_eom.add_argument("--nd", type=float, default=None, dest="nd_core_cm3")
    p_eom.add_argument("--pn-offset", type=float, default=None, dest="pn_offset_nm")
    p_eom.add_argument("--pn-sink", action="store_true", dest="pn_sink", help="Include W-VIAC/M1 PN contact heat-sinking in 2D Thermal FEM")
    p_eom.add_argument("--gds", action="store_true", dest="export_gds")
    p_eom.add_argument("--bundle", action="store_true", dest="export_bundle")
    p_eom.add_argument("--json", type=str, default=None, dest="raw_json")

    p_pdk = ag_sub.add_parser("pdk-drc", help="Foundry PDK Polygon DRC (.lyrdb), 3D Union-Find LVS, PN+Heater Ring GDS, or Native Cutline 2D-FEM (<140 tokens)")
    p_pdk.add_argument("--mode", type=str, default="build_pn_heater_ring", help="build_pn_heater_ring | verify_gds | cutline_solve | export_csv")
    p_pdk.add_argument("--pdk", type=str, default="gsip_soi220", help="gsip_soi220 | cornerstone_soi220 | siepic_ebeam_soi220 | tfln_xcut_600 | scaln_soi_bilayer | microring_generic")
    p_pdk.add_argument("--gds-path", type=str, default=None, dest="gds_path")
    p_pdk.add_argument("--pdk-csv-dir", type=str, default=None, dest="pdk_csv_dir")
    p_pdk.add_argument("--r", type=float, default=12.0, dest="r_um")
    p_pdk.add_argument("--w-ring", type=float, default=0.45, dest="w_ring_um")
    p_pdk.add_argument("--gap", type=float, default=0.20, dest="gap_um")
    p_pdk.add_argument("--pn-offset", type=float, default=20.0, dest="pn_offset_nm")
    p_pdk.add_argument("--wl", type=float, default=1550.0, dest="wl_nm")
    p_pdk.add_argument("--json", type=str, default=None, dest="raw_json")

    p_pr = ag_sub.add_parser("prism-optimize", help="PRISM Physics Closure Gate or Prediction-Gated Closed-Loop Real-2D-FEM Optimizer (<140 tokens)")
    p_pr.add_argument("--mode", type=str, default="optimize", help="optimize | closure_audit")
    p_pr.add_argument("--problem", type=str, default="pn_ring_modulator", help="pn_ring_modulator | high_er_pulley_mrr | low_vpil_eom")
    p_pr.add_argument("--iters", type=int, default=8, dest="max_iterations")
    p_pr.add_argument("--seed", type=int, default=42)
    p_pr.add_argument("--mat", type=str, default=None)
    p_pr.add_argument("--r", type=float, default=None, dest="r_um")
    p_pr.add_argument("--gap", type=float, default=None, dest="gap_um")
    p_pr.add_argument("--json", type=str, default=None, dest="raw_json")

    p_wdm = ag_sub.add_parser("wdm-link", help="Multi-Channel Microring WDM Link, NRZ/PAM4 Eye Diagram & Thermal Locking (<140 tokens)")
    p_wdm.add_argument("--scenario", type=str, default="nominal", help="nominal | thermal_drift | heater_relocked | dense_crosstalk | low_laser_power")
    p_wdm.add_argument("--n-ch", type=int, default=4, dest="n_ch")
    p_wdm.add_argument("--rate", type=float, default=25.0, dest="rate_gbps")
    p_wdm.add_argument("--format", type=str, default="nrz", dest="format", help="nrz | pam4")
    p_wdm.add_argument("--spacing", type=float, default=100.0, dest="spacing_ghz")
    p_wdm.add_argument("--dt", type=float, default=3.0, dest="dt_k")
    p_wdm.add_argument("--r", type=float, default=10.0, dest="r_um")
    p_wdm.add_argument("--gap", type=float, default=0.20, dest="gap_um")
    p_wdm.add_argument("--vpp", type=float, default=2.0, dest="vpp")
    p_wdm.add_argument("--json", type=str, default=None, dest="raw_json")

    p_nl = ag_sub.add_parser("nonlinear", help="First-Principles Nonlinear Photonics & LLE Microcomb Engine (dispersion, LLE, FWM-BS, SHG, bistability) (<150 tokens)")
    p_nl.add_argument("--analysis", type=str, default="lle", help="lle | dispersion | fwm_bs | shg | bistability")
    p_nl.add_argument("--mat", type=str, default="Si3N4")
    p_nl.add_argument("--h", type=float, default=None, dest="h_um")
    p_nl.add_argument("--slab", type=float, default=None, dest="slab_um")
    p_nl.add_argument("--w-ring", type=float, default=None, dest="w_ring_um")
    p_nl.add_argument("--r", type=float, default=None, dest="r_um")
    p_nl.add_argument("--wl", type=float, default=None, dest="wl_nm")
    p_nl.add_argument("--mode", type=str, default=None, help="single_soliton | soliton_sweep | dark_pulse")
    p_nl.add_argument("--p-mw", type=float, default=None, dest="p_mw")
    p_nl.add_argument("--raman", action="store_true", dest="include_raman")
    p_nl.add_argument("--thermal", action="store_true", dest="include_thermal")
    p_nl.add_argument("--tpa", action="store_true", dest="include_tpa_fca")
    p_nl.add_argument("--json", type=str, default=None, dest="raw_json")

    p_adv = ag_sub.add_parser("advanced", help="High-Order CROW/SCISSOR, Euler Bends, Sensor/Athermal, TD-CMT, MZI-MRR, Molecule/EP, Quantum SFWM & Lasers")
    p_adv.add_argument("--analysis", type=str, default="high_order_filter")
    p_adv.add_argument("--mat", type=str, default=None)
    p_adv.add_argument("--r", type=float, default=None, dest="r_um")
    p_adv.add_argument("--n", type=int, default=None)
    p_adv.add_argument("--gds", action="store_true", dest="export_gds")
    p_adv.add_argument("--json", type=str, default=None, dest="raw_json")

    p_swg = ag_sub.add_parser("swg-phc", help="Subwavelength Grating (SWG) & Photonic Crystal Ring (PhCR) Solvers")
    p_swg.add_argument("--analysis", type=str, default="swg", help="swg | phcr")
    p_swg.add_argument("--mat", type=str, default=None)
    p_swg.add_argument("--r", type=float, default=None, dest="r_um")
    p_swg.add_argument("--gds", action="store_true", dest="export_gds")
    p_swg.add_argument("--json", type=str, default=None, dest="raw_json")

    p_saw = ag_sub.add_parser("saw-sbs", help="2D Solid Mechanics / Piezoelectric SAW-Lamb Acousto-Optic & SBS Brillouin FEM")
    p_saw.add_argument("--analysis", type=str, default="saw", help="saw | sbs")
    p_saw.add_argument("--mat", type=str, default=None)
    p_saw.add_argument("--substrate", type=str, default=None, dest="substrate_material")
    p_saw.add_argument("--json", type=str, default=None, dest="raw_json")

    p_na = ag_sub.add_parser("nanoantenna", help="Nanoantenna & Metasurface TCMT (MIM absorber, Fano/qBIC, Mid-IR SEIRA, EP)")
    p_na.add_argument("--regime", type=str, default="seira_molecular")
    p_na.add_argument("--analyte", type=str, default=None, dest="analyte_preset")
    p_na.add_argument("--nu0", type=float, default=None, dest="nu0_bright_cm1")
    p_na.add_argument("--json", type=str, default=None, dest="raw_json")

    p_2d = ag_sub.add_parser("twod-material", help="27-Material 2D vdW Library on MRR (Graphene/BLG, TMDs, Bulk vdW, PtSe2/PdSe2/ReS2, hBN/a-MoO3/a-V2O5, BP/b-AsP/Te, MXenes, Bi2Se3/WTe2, NbOCl2/In2Se3/CIPS/GaSe) (<150 tokens)")
    p_2d.add_argument("--mat-2d", type=str, default="Graphene", dest="material_2d", help="Any of 27 2D vdW catalog keys (use --list to inspect)")
    p_2d.add_argument("--cap", type=str, default="SLG", dest="capacitor_type", help="SLG | DLG")
    p_2d.add_argument("--mat", type=str, default="Si")
    p_2d.add_argument("--wl", type=float, default=None, dest="wl_nm")
    p_2d.add_argument("--r", type=float, default=15.0, dest="r_um")
    p_2d.add_argument("--spacer", type=float, default=10.0, dest="spacer_ox_nm")
    p_2d.add_argument("--angle", type=float, default=0.0, dest="crystal_angle_deg", help="In-plane crystal principal axis angle [deg]")
    p_2d.add_argument("--t2d", type=float, default=None, dest="thickness_nm_override", help="Optional 2D flake thickness [nm]")
    p_2d.add_argument("--list", action="store_true", dest="list_materials", help="List all 27 2D materials")
    p_2d.add_argument("--family", type=str, default=None, help="Optional family filter with --list")
    p_2d.add_argument("--json", type=str, default=None, dest="raw_json")

    p_pd = ag_sub.add_parser("photodetector", help="Ge-on-Si PIN/MUTC waveguide PD & O-E-O photonic neuron")
    p_pd.add_argument("--analysis", type=str, default="waveguide_pd")
    p_pd.add_argument("--arch", type=str, default="vertical_pin", dest="pd_architecture")
    p_pd.add_argument("--len", type=float, default=15.0, dest="ge_length_um")
    p_pd.add_argument("--gds", action="store_true", dest="export_gds")
    p_pd.add_argument("--json", type=str, default=None, dest="raw_json")

    p_pol = ag_sub.add_parser("polarization-mdm", help="PSR polarization rotator, PBS & MDM mode multiplexer")
    p_pol.add_argument("--analysis", type=str, default="psr")
    p_pol.add_argument("--gap", type=float, default=0.20, dest="coupler_gap_um")
    p_pol.add_argument("--gds", action="store_true", dest="export_gds")
    p_pol.add_argument("--json", type=str, default=None, dest="raw_json")

    p_pcm = ag_sub.add_parser("pcm-photonics", help="Non-volatile PCM (Sb2Se3/GSST/GST/VO2) MRR & SET/RESET pulse FEM")
    p_pcm.add_argument("--analysis", type=str, default="pcm_mrr")
    p_pcm.add_argument("--pcm", type=str, default="Sb2Se3", dest="pcm_material")
    p_pcm.add_argument("--gds", action="store_true", dest="export_gds")
    p_pcm.add_argument("--json", type=str, default=None, dest="raw_json")

    p_pnn = ag_sub.add_parser("photonic-computing", help="MRR Weight Bank & Clements/Reck unitary MZI mesh")
    p_pnn.add_argument("--analysis", type=str, default="mrr_weight_bank")
    p_pnn.add_argument("--n", type=int, default=6, dest="mesh_size_n")
    p_pnn.add_argument("--gds", action="store_true", dest="export_gds")
    p_pnn.add_argument("--json", type=str, default=None, dest="raw_json")

    p_wbg = ag_sub.add_parser("wbg-awg", help="Waveguide Bragg Grating (WBG) & Arrayed Waveguide Grating (AWG)")
    p_wbg.add_argument("--analysis", type=str, default="wbg")
    p_wbg.add_argument("--type", type=str, default="pi_phase_shifted", dest="grating_type")
    p_wbg.add_argument("--ch", type=int, default=8, dest="num_channels")
    p_wbg.add_argument("--gds", action="store_true", dest="export_gds")
    p_wbg.add_argument("--json", type=str, default=None, dest="raw_json")

    p_net = ag_sub.add_parser("pic-netlist", help="Native frequency-domain PIC S-matrix circuit netlist simulator")
    p_net.add_argument("--preset", type=str, default="ramzi_linearized")
    p_net.add_argument("--snp", action="store_true", dest="export_snp")
    p_net.add_argument("--json", type=str, default=None, dest="raw_json")

    p_inv = ag_sub.add_parser("inverse-adjoint", help="2D-FDFD exact Maxwell adjoint-gradient inverse shape optimizer")
    p_inv.add_argument("--len", type=float, default=2.40, dest="design_length_um")
    p_inv.add_argument("--iters", type=int, default=30, dest="max_iterations")
    p_inv.add_argument("--gds", action="store_true", dest="export_gds")
    p_inv.add_argument("--json", type=str, default=None, dest="raw_json")

    p_ep = ag_sub.add_parser("mrr-ep", help="MRR Exceptional Point engine: chiral CW/CCW EP2, bus-mirror ES, higher-order PT EP2/EP3/EP4, Anti-PT & dynamic encircling")
    p_ep.add_argument("--regime", type=str, default="chiral_single_ring", help="chiral_single_ring | exceptional_surface | pt_ep_n | anti_pt_bus | ep_encircling")
    p_ep.add_argument("--n", type=int, default=3, dest="order_n", help="Order N=2/3/4 for pt_ep_n")
    p_ep.add_argument("--j", type=float, default=2.0, dest="coupling_j_ghz")
    p_ep.add_argument("--eps", type=float, default=0.05, dest="perturbation_ghz")
    p_ep.add_argument("--fem-gap", action="store_true", dest="synthesize_fem_gap")
    p_ep.add_argument("--gds", action="store_true", dest="export_gds")
    p_ep.add_argument("--json", type=str, default=None, dest="raw_json")

    p_opa = ag_sub.add_parser("opa-lidar", help="Live 1D/2D Optical Phased Array & LiDAR beam-steering engine")
    p_opa.add_argument("--n", type=int, default=16, dest="num_antennas")
    p_opa.add_argument("--pitch", type=float, default=2.0, dest="array_pitch_um")
    p_opa.add_argument("--layout", type=str, default="aperiodic", dest="pitch_layout")
    p_opa.add_argument("--steer", type=float, default=15.0, dest="steer_theta_deg")
    p_opa.add_argument("--gds", action="store_true", dest="export_gds")
    p_opa.add_argument("--json", type=str, default=None, dest="raw_json")

    p_slb = ag_sub.add_parser("slab-diffractive", help="Live 2D ASM Slab Rowland FPR & trainable PDDNN/EO-DPNN engine")
    p_slb.add_argument("--mode", type=str, default="diffractive_pddnn")
    p_slb.add_argument("--task", type=str, default="beam_router", dest="target_task")
    p_slb.add_argument("--layers", type=int, default=3, dest="num_diffractive_layers")
    p_slb.add_argument("--slots", type=int, default=16, dest="num_phase_slots")
    p_slb.add_argument("--gds", action="store_true", dest="export_gds")
    p_slb.add_argument("--json", type=str, default=None, dest="raw_json")

    p_eam = ag_sub.add_parser("eam-soa", help="Live GeSi FKE / InP QCSE EAM & Traveling-Wave SOA / EDWA engine")
    p_eam.add_argument("--analysis", type=str, default="eam", help="eam | soa | edwa")
    p_eam.add_argument("--eam-type", type=str, default="gesi_fke", dest="eam_type")
    p_eam.add_argument("--wl", type=float, default=None, dest="wavelength_nm")
    p_eam.add_argument("--gds", action="store_true", dest="export_gds")
    p_eam.add_argument("--json", type=str, default=None, dest="raw_json")

    p_gnl = ag_sub.add_parser("gnlse-ppln", help="Live single-pass GNLSE Supercontinuum & PPLN QPM SHG/SFG/DFG engine")
    p_gnl.add_argument("--analysis", type=str, default="ppln", help="ppln | gnlse")
    p_gnl.add_argument("--process", type=str, default="shg")
    p_gnl.add_argument("--mat", type=str, default=None)
    p_gnl.add_argument("--gds", action="store_true", dest="export_gds")
    p_gnl.add_argument("--json", type=str, default=None, dest="raw_json")

    p_rt = ag_sub.add_parser("routing-crossings", help="Live 2D-FDFD MMI Crossing, Adiabatic Taper & PIC Auto-Router")
    p_rt.add_argument("--analysis", type=str, default="crossing", help="crossing | taper | router")
    p_rt.add_argument("--gds", action="store_true", dest="export_gds")
    p_rt.add_argument("--json", type=str, default=None, dest="raw_json")

    p_iso = ag_sub.add_parser("isolator-mems", help="Live Ce:YIG/Spatiotemporal Optical Isolator & Photonic MEMS Switch")
    p_iso.add_argument("--analysis", type=str, default="isolator", help="isolator | mems")
    p_iso.add_argument("--iso-type", type=str, default="magneto_optic_ceyig", dest="iso_type")
    p_iso.add_argument("--beam", type=str, default="cantilever", dest="beam_type")
    p_iso.add_argument("--gds", action="store_true", dest="export_gds")
    p_iso.add_argument("--json", type=str, default=None, dest="raw_json")

    p_eye = ag_sub.add_parser("eom-rf-eye", help="Live EOM RF packaging ABCD (pad/taper/wirebond/flip-chip) + NRZ/PAM4 eye diagram, BER, TDECQ & FFE")
    p_eye.add_argument("--launch", type=str, default="gsg_probe", dest="launch_type", help="gsg_probe | wirebond | flip_chip_bump")
    p_eye.add_argument("--fmt", type=str, default="PAM4", dest="modulation_format", help="PAM4 | NRZ")
    p_eye.add_argument("--baud", type=float, default=56.0, dest="baud_rate_gbaud")
    p_eye.add_argument("--vpp", type=float, default=2.5, dest="v_pp_drive_v")
    p_eye.add_argument("--wb-um", type=float, default=350.0, dest="wirebond_length_um")
    p_eye.add_argument("--ffe", type=int, default=5, dest="ffe_taps")
    p_eye.add_argument("--s2p", action="store_true", dest="export_s2p")
    p_eye.add_argument("--json", type=str, default=None, dest="raw_json")

    p_vsg = ag_sub.add_parser("vernier-sil-gyro", help="Live External-Cavity Vernier Laser, SIL Laser + TRN & RFOG Optical Gyroscope")
    p_vsg.add_argument("--analysis", type=str, default="vernier_ecl", help="vernier_ecl | self_injection_locked | rfog")
    p_vsg.add_argument("--gds", action="store_true", dest="export_gds")
    p_vsg.add_argument("--json", type=str, default=None, dest="raw_json")

    p_mwp = ag_sub.add_parser("mrr-mwp", help="Live MRR Microwave Photonics (TTD/OBFN, RF Notch, OEO, Optical Diff/Integrator)")
    p_mwp.add_argument("--mode", type=str, default="ttd_obfn", help="ttd_obfn | rf_notch_filter | oeo_oscillator | temporal_calculus")
    p_mwp.add_argument("--op", type=str, default="differentiator", dest="calculus_op")
    p_mwp.add_argument("--gds", action="store_true", dest="export_gds")
    p_mwp.add_argument("--json", type=str, default=None, dest="raw_json")

    p_eoc = ag_sub.add_parser("eocomb-opo-ssb", help="Live Resonant EO Comb (1R/2R), SFD, Kerr SSB Isolator & chi(3)/chi(2) MRR OPO")
    p_eoc.add_argument("--analysis", type=str, default="coupled_ring_eocomb")
    p_eoc.add_argument("--gds", action="store_true", dest="export_gds")
    p_eoc.add_argument("--json", type=str, default=None, dest="raw_json")

    p_oot = ag_sub.add_parser("oam-optomech-topo", help="Live OAM Vortex MRR, Cavity Optomechanics/Quantum Transducer & Topological MRR Lattices")
    p_oot.add_argument("--analysis", type=str, default="oam_vortex_mrr")
    p_oot.add_argument("--gds", action="store_true", dest="export_gds")
    p_oot.add_argument("--json", type=str, default=None, dest="raw_json")

    p_rss = ag_sub.add_parser("reservoir-switch-sensing", help="Live Silicon MRR Reservoir Computer, N×N Switch Matrix, Nanoparticle Sizing & Dual-Comb DCS")
    p_rss.add_argument("--analysis", type=str, default="photonic_reservoir")
    p_rss.add_argument("--gds", action="store_true", dest="export_gds")
    p_rss.add_argument("--json", type=str, default=None, dest="raw_json")

    args = parser.parse_args()

    if args.cmd in (None, "app"):
        raise SystemExit(launch_studio(browser_only=False, port=getattr(args, "port", None)))

    if args.cmd == "browser":
        raise SystemExit(launch_studio(browser_only=True, port=args.port))

    if args.cmd == "mcp":
        run_mcp_stdio()
        return

    if args.cmd == "check":
        prof = detect_hardware_profile()
        bench = run_compute_benchmark(100_000)
        print(json.dumps({"hardware": prof.to_dict(), "benchmark_100k_pts": bench}, indent=2))
        return

    if args.cmd == "analyze-gds":
        info = analyze_uploaded_gds(args.gds_file)
        summary = {k: v for k, v in info.items() if k != "viewer_layers"}
        print(json.dumps(summary, indent=2))
        return

    if args.cmd == "chat":
        sess = create_initial_session()
        out = process_chat_turn(sess, args.prompt, gds_export_dir=Path.cwd() / "outputs" / "gds_exports")
        print(out["assistant_message"])
        return

    if args.cmd == "agent":
        if args.agent_cmd == "status":
            _print_compact(agent_status())
            return
        if args.agent_cmd == "inspect-gds":
            _print_compact(agent_inspect_gds({"gds_path": args.gds_path, "cell_name": args.cell_name}))
            return
        if args.agent_cmd == "fit":
            _print_compact(agent_fit_spectrum({
                "csv_path": args.csv_path,
                "model_type": args.model_type,
                "known_regime": args.known_regime,
                "ring_length_um": args.ring_length_um,
            }))
            return
        d = {k: v for k, v in vars(args).items() if k not in ("cmd", "agent_cmd", "raw_json") and v is not None}
        if getattr(args, "raw_json", None):
            d.update(json.loads(args.raw_json))
        if args.agent_cmd == "width-cutoff":
            _print_compact(agent_width_cutoff(d))
        elif args.agent_cmd == "design":
            _print_compact(agent_design_mrr(d))
        elif args.agent_cmd == "sweep":
            _print_compact(agent_sweep_mrr(d))
        elif args.agent_cmd == "fab-tolerance":
            _print_compact(agent_fab_tolerance(d))
        elif args.agent_cmd == "active-tuning":
            _print_compact(agent_active_tuning(d))
        elif args.agent_cmd == "io-coupler":
            _print_compact(agent_io_coupler(d))
        elif args.agent_cmd == "high-q":
            _print_compact(agent_high_q_fdtd(d))
        elif args.agent_cmd == "mzi":
            _print_compact(agent_design_mzi(d))
        elif args.agent_cmd == "eom":
            _print_compact(agent_eom_multiphysics(d))
        elif args.agent_cmd == "pdk-drc":
            _print_compact(agent_pdk_drc(d))
        elif args.agent_cmd == "prism-optimize":
            _print_compact(agent_prism_optimize(d))
        elif args.agent_cmd == "wdm-link":
            _print_compact(agent_wdm_link(d))
        elif args.agent_cmd == "nonlinear":
            _print_compact(agent_nonlinear_lle(d))
        elif args.agent_cmd == "advanced":
            _print_compact(agent_advanced_mrr(d))
        elif args.agent_cmd == "swg-phc":
            _print_compact(agent_swg_phc(d))
        elif args.agent_cmd == "saw-sbs":
            _print_compact(agent_saw_sbs(d))
        elif args.agent_cmd == "nanoantenna":
            _print_compact(agent_nanoantenna_tcmt(d))
        elif args.agent_cmd == "twod-material":
            _print_compact(agent_2d_material_mrr(d))
        elif args.agent_cmd == "photodetector":
            _print_compact(agent_photodetector(d))
        elif args.agent_cmd == "polarization-mdm":
            _print_compact(agent_polarization_mdm(d))
        elif args.agent_cmd == "pcm-photonics":
            _print_compact(agent_pcm_photonics(d))
        elif args.agent_cmd == "photonic-computing":
            _print_compact(agent_photonic_computing(d))
        elif args.agent_cmd == "wbg-awg":
            _print_compact(agent_wbg_awg(d))
        elif args.agent_cmd == "pic-netlist":
            _print_compact(agent_pic_netlist(d))
        elif args.agent_cmd == "inverse-adjoint":
            _print_compact(agent_inverse_adjoint(d))
        elif args.agent_cmd == "mrr-ep":
            _print_compact(agent_mrr_ep(d))
        elif args.agent_cmd == "opa-lidar":
            _print_compact(agent_opa_lidar(d))
        elif args.agent_cmd == "slab-diffractive":
            _print_compact(agent_slab_diffractive(d))
        elif args.agent_cmd == "eam-soa":
            _print_compact(agent_eam_soa(d))
        elif args.agent_cmd == "gnlse-ppln":
            _print_compact(agent_gnlse_ppln(d))
        elif args.agent_cmd == "routing-crossings":
            _print_compact(agent_routing_crossings(d))
        elif args.agent_cmd == "isolator-mems":
            _print_compact(agent_isolator_mems(d))
        elif args.agent_cmd == "eom-rf-eye":
            _print_compact(agent_eom_rf_eye(d))
        elif args.agent_cmd == "vernier-sil-gyro":
            _print_compact(agent_vernier_sil_gyro(d))
        elif args.agent_cmd == "mrr-mwp":
            _print_compact(agent_mrr_mwp(d))
        elif args.agent_cmd == "eocomb-opo-ssb":
            _print_compact(agent_eocomb_opo_ssb(d))
        elif args.agent_cmd == "oam-optomech-topo":
            _print_compact(agent_oam_optomech_topo(d))
        elif args.agent_cmd == "reservoir-switch-sensing":
            _print_compact(agent_reservoir_switch_sensing(d))


if __name__ == "__main__":
    main()
