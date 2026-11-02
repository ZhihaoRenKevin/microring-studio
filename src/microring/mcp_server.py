"""Zero-Dependency Model Context Protocol (`MCP`) Stdio Server (`microring mcp`).

Connects `MicroRing Studio` directly to **Claude Code**, **Claude Desktop**, **OpenAI Codex**,
**Cursor**, and **Antigravity** over JSON-RPC 2.0 (`stdin`/`stdout`).

Token-efficiency guarantees:
  - Compact tool schemas (<400 tokens total across all 6 tools).
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

MCP_TOOLS = [
    {
        "name": "microring_status",
        "description": "Probe local GPU/AMX/CPU accelerator & list supported 18 materials and 7 MRR topologies.",
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
