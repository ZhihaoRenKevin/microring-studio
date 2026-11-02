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
    agent_design_mrr,
    agent_design_mzi,
    agent_fab_tolerance,
    agent_fit_spectrum,
    agent_high_q_fdtd,
    agent_inspect_gds,
    agent_io_coupler,
    agent_status,
    agent_sweep_mrr,
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


if __name__ == "__main__":
    main()
