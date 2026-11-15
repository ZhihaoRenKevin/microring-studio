"""Comprehensive Automated Test Suite for MicroRing Studio v3.9.0.

Can be executed either with `pytest` (`uv run pytest`) or directly as a standalone
verification script (`python tests/test_microring_all.py`).
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring import (
    EO_WAFER_CUT_CATALOG,
    EOM_ELECTRODE_CATALOG,
    EOMElectrodeConfig,
    FIBER_LASER_CATALOG,
    FOUNDRY_PDK_CATALOG,
    MATERIAL_CATALOG,
    MRRDesignConfig,
    MZI_SPLITTER_CATALOG,
    MZIDesignConfig,
    PN_JUNCTION_CATALOG,
    PNDopingTCADConfig,
    PRISM_PROBLEM_CATALOG,
    RESIST_TONE_CATALOG,
    THERMAL_STRUCTURE_CATALOG,
    TUNING_MECHANISM_CATALOG,
    T_REF_K,
    ThermalSimConfig,
    WDM_SCENARIO_CATALOG,
    WDMLinkConfig,
    WaveguideCrossSectionSpec,
    agent_active_tuning,
    agent_design_mrr,
    agent_design_mzi,
    agent_eom_multiphysics,
    agent_fab_tolerance,
    agent_fit_spectrum,
    agent_high_q_fdtd,
    agent_inspect_gds,
    agent_io_coupler,
    agent_pdk_drc,
    agent_prism_optimize,
    agent_status,
    agent_sweep_mrr,
    agent_wdm_link,
    analyze_eo_wafer_orientation,
    analyze_high_q_fdtd_convergence,
    analyze_uploaded_gds,
    audit_mrr_and_pn_physics_closure,
    build_pn_heater_ring_modulator_gds,
    create_initial_session,
    design_butt_edge_coupler,
    design_eom_electrode,
    design_grating_coupler,
    detect_hardware_profile,
    export_coupler_cutout_gds,
    export_eom_electrode_gds,
    export_eom_rf_interconnect_bundle,
    export_fab_bias_array_gds,
    export_io_coupler_gds,
    export_io_coupler_interconnect,
    export_lumerical_interconnect_bundle,
    export_mrr_gds,
    export_mrr_gds_with_pdk,
    export_mzi_gds,
    export_mzi_interconnect_bundle,
    export_pdk_csv_tables,
    export_pn_doping_gds,
    export_sweep_array_gds,
    export_tcad_data_bundle,
    export_thermal_shifter_gds,
    extract_3d_electrical_lvs,
    fit_single_resonance_cmt,
    get_foundry_pdk,
    get_material_fn,
    load_pdk_from_csv_dir,
    loss_db_per_cm_to_q_int,
    process_chat_turn,
    q_int_to_loss_db_per_cm,
    run_active_tuning_analysis,
    run_compute_benchmark,
    run_fabrication_error_analysis,
    run_foundry_process_corners,
    run_gds_polygon_drc,
    run_mrr_parameter_sweep,
    run_prism_closed_loop_optimizer,
    run_waveguide_width_sweep,
    simulate_cointegrated_pn_heater_ring,
    simulate_mrr_topology,
    simulate_mzi,
    simulate_pn_doping_tcad,
    simulate_thermal_phase_shifter,
    simulate_wdm_microring_link,
    solve_gds_cutline_multiphysics,
    solve_modes_fem,
)
import microring.app_server as srv
from microring.mcp_server import handle_mcp_request
from microring.mrr_cmt_fit_predict import cmt_split_doublet_power


async def _asgi_request(
    method: str,
    path: str,
    query_string: str = "",
    json_body: dict | None = None,
) -> tuple[int, bytes]:
    """Zero-dependency in-process ASGI caller for testing `microring.app_server.app`."""
    raw_body = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    headers = [(b"host", b"127.0.0.1:8765")]
    if json_body is not None:
        headers.append((b"content-type", b"application/json"))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string.encode("ascii"),
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
    }
    sent = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await srv.app(scope, receive, send)
    return status_code, b"".join(chunks)


def test_materials_and_hardware() -> None:
    assert len(MATERIAL_CATALOG) == 29
    assert {"LT_Xcut", "LT_Zcut", "BTO", "BTO_ccut", "PZT"}.issubset(MATERIAL_CATALOG.keys())
    for _key, meta in MATERIAL_CATALOG.items():
        wl_m = float(meta["default_band_nm"]) * 1e-9
        props = meta["fn"](wl_m, T_REF_K)
        assert props.nx >= 1.0
        assert props.ny >= 1.0

    prof = detect_hardware_profile()
    bench = run_compute_benchmark(20_000)
    assert prof.to_dict()["cpu_cores"] >= 1
    assert bench["throughput_mpts_per_s"] > 0.5


def test_pure_bsd_fem_no_gpl() -> None:
    spec = WaveguideCrossSectionSpec(
        w_bottom_um=1.10,
        h_film_um=0.60,
        h_slab_um=0.0,
        sidewall_deg=82.0,
        core_material_fn=get_material_fn("4H-SiC"),
        clad_material_fn=get_material_fn("SiO2"),
        top_clad_material_fn=get_material_fn("SiO2"),
    )
    modes = solve_modes_fem(spec, 1550e-9, bend_radius_m=40e-6, num_modes=2, core_mesh_res_um=0.045)
    assert "gmsh" not in sys.modules
    assert 2.10 < float(np.real(modes[0].n_eff)) < 2.55
    assert modes[0].te_fraction > 0.85


def test_width_and_mrr_sweeps() -> None:
    w_sw = run_waveguide_width_sweep(
        core_material="Si3N4",
        h_film_um=0.40,
        w_min_um=0.70,
        w_max_um=1.80,
        num_points=4,
    )
    assert len(w_sw["widths_um"]) == 4
    assert 0.70 <= w_sw["single_mode_max_width_um"] <= 1.80

    base = MRRDesignConfig(
        topology="pulley_ap",
        core_material="4H-SiC",
        film_height_um=0.60,
        ring_width_um=1.10,
        bus_width_um=0.82,
        ring_radius_um=40.0,
        gap_thru_um=0.25,
    )
    init = simulate_mrr_topology(base)
    theta_cc = float(init["critical_coupling"]["recommended_value"])
    cc_res = simulate_mrr_topology(replace(base, pulley_angle_thru_deg=theta_cc))
    assert cc_res["metrics"]["ER_thru_dB"] > 35.0

    m_sw = run_mrr_parameter_sweep(base, "gap_thru_um", [0.20, 0.25, 0.30])
    assert len(m_sw["ER_thru_dB"]) == 3


def test_bidirectional_gds_roundtrip() -> None:
    cases = [
        ("point_circular_ap", {}),
        ("racetrack_ad", {"straight_length_um": 14.0}),
        ("pulley_ap", {"pulley_angle_thru_deg": 24.0}),
        ("vernier_crow_ad", {"ring2_radius_um": 43.0, "inter_ring_gap_um": 0.35}),
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        for topology, extra in cases:
            cfg = MRRDesignConfig(
                topology=topology,
                core_material="Si3N4",
                ring_radius_um=40.0,
                ring_width_um=1.20,
                bus_width_um=0.95,
                gap_thru_um=0.25,
                gap_drop_um=0.28,
                **extra,
            )
            gds_file = tmp_path / f"sample_{topology}.gds"
            export_mrr_gds(cfg, gds_file)
            info = analyze_uploaded_gds(gds_file)
            ext = info["extracted_geometry"]

            assert ext["topology"] == topology, f"Expected {topology}, got {ext['topology']}"
            assert abs(ext["ring_radius_um"] - 40.0) < 0.08
            assert abs(ext["ring_width_um"] - 1.20) < 0.04
            assert abs(ext["gap_thru_um"] - 0.25) < 0.02
            if topology == "racetrack_ad":
                assert abs(ext["straight_length_um"] - 14.0) < 0.15

        arr_file = tmp_path / "sweep_arr.gds"
        export_sweep_array_gds(cfg, "gap_thru_um", [0.20, 0.25, 0.30], arr_file)
        assert arr_file.is_file()
        assert arr_file.with_suffix(".lyp").is_file()


def test_conversational_copilot_flow() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # 1. Guided step-by-step wizard (1 question at a time + dB/cm propagation loss)
        s0 = create_initial_session(lang="en")
        g1 = process_chat_turn(s0, "Start step-by-step microring design", gds_export_dir=tmp_path)
        assert g1["wizard_step"] == 2
        g2 = process_chat_turn(g1["session"], "Wavelength = 1550 nm", gds_export_dir=tmp_path)
        assert g2["wizard_step"] == 3
        g3 = process_chat_turn(g2["session"], "Core material: Si3N4", gds_export_dir=tmp_path)
        assert g3["wizard_step"] == 4
        g4 = process_chat_turn(g3["session"], "Confirm stack and solve 2D modes", gds_export_dir=tmp_path)
        assert "width_sweep_result" in g4 and g4["wizard_step"] == 5
        g5 = process_chat_turn(g4["session"], "Propagation loss = 0.85 dB/cm", gds_export_dir=tmp_path)
        assert g5["wizard_step"] == 5 and abs(g5["session"]["config"]["alpha_loss_db_per_cm"] - 0.85) < 1e-4

        # Live EN -> ZH language switch mid-step without losing design state
        g_lang = process_chat_turn(g5["session"], "__switch_lang__", gds_export_dir=tmp_path, lang="zh")
        assert g_lang["session"]["lang"] == "zh"
        assert abs(g_lang["session"]["config"]["alpha_loss_db_per_cm"] - 0.85) < 1e-4

        g6 = process_chat_turn(g_lang["session"], "FSR 200 GHz, 点耦合两端口", gds_export_dir=tmp_path, lang="zh")
        assert "mrr_result" in g6 and g6["wizard_step"] == 6
        assert abs(g6["mrr_result"]["metrics"]["alpha_loss_db_per_cm"] - 0.85) < 1e-3
        fsr_hit = g6["mrr_result"]["metrics"]["FSR_GHz"]
        assert 180.0 <= fsr_hit <= 220.0

        r3 = process_chat_turn(g6["session"], "Apply critical coupling", gds_export_dir=tmp_path, lang="en")
        assert r3["mrr_result"]["metrics"]["ER_thru_dB"] > 30.0

        r_sw_loss = process_chat_turn(r3["session"], "Sweep propagation loss", gds_export_dir=tmp_path, lang="en")
        assert "mrr_sweep_result" in r_sw_loss
        assert r_sw_loss["mrr_sweep_result"]["param_name"] == "alpha_loss_db_per_cm"

        r4 = process_chat_turn(r3["session"], "Export single GDS layout", gds_export_dir=tmp_path)
        assert "gds_export" in r4
        assert Path(r4["gds_export"]["gds_path"]).is_file()

        # 2. Fast-forward Chinese one-shot prompt
        r_zh = process_chat_turn(
            create_initial_session(lang="zh"),
            "220 nm SOI 跑道微环 四端口 损耗 1.8 dB/cm FSR 150 GHz 开始仿真",
            gds_export_dir=tmp_path,
            lang="zh",
        )
        assert r_zh["session"]["config"]["core_material"] == "Si"
        assert r_zh["session"]["config"]["topology"] == "racetrack_ad"
        assert abs(r_zh["mrr_result"]["metrics"]["alpha_loss_db_per_cm"] - 1.8) < 1e-3


def test_cmt_spectrum_fitter() -> None:
    # Exact bidirectional loss [dB/cm] <-> Q_int identity check
    q_calc = loss_db_per_cm_to_q_int(0.75, ng=2.12, wavelength_nm=1550.0)
    alpha_back = q_int_to_loss_db_per_cm(q_calc, ng=2.12, wavelength_nm=1550.0)
    assert abs(alpha_back - 0.75) < 1e-6

    wl = np.linspace(1550.47, 1550.53, 300)
    tr = cmt_split_doublet_power(
        wl, 1550.500, kappa_int_ghz=0.18, kappa_ext_ghz=0.14, mu_split_ghz=0.26
    )
    summary, _, _ = fit_single_resonance_cmt(
        wl, tr, model_type="auto", known_regime="undercoupled", ng_assumed=2.10, ring_length_um=250.0
    )
    assert summary.model_type == "split_doublet"
    assert summary.r_squared > 0.999
    assert summary.alpha_loss_db_per_cm > 0.05

    # Also test logarithmic dB/dBm trace auto-conversion
    tr_dbm = -8.5 + 10.0 * np.log10(np.clip(tr, 1e-5, None))
    summary_db, _, _ = fit_single_resonance_cmt(
        wl, tr_dbm, model_type="split_doublet", known_regime="undercoupled", ng_assumed=2.10
    )
    assert abs(summary_db.alpha_loss_db_per_cm - summary.alpha_loss_db_per_cm) < 0.03


def test_http_endpoints_and_gds_upload() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            old_exp, old_up = srv.GDS_EXPORT_DIR, srv.GDS_UPLOAD_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "exports"
            srv.GDS_UPLOAD_DIR = tmp_path / "uploads"
            try:
                st, body = await _asgi_request("GET", "/")
                assert st == 200 and b"MicroRing Studio" in body and b"setLanguage('zh')" in body

                st, body = await _asgi_request("GET", "/api/materials")
                assert st == 200 and len(json.loads(body)["materials"]) == 29

                st, body_zh = await _asgi_request("POST", "/api/chat_init", query_string="lang=zh")
                assert st == 200 and json.loads(body_zh)["session"]["lang"] == "zh"

                st, body = await _asgi_request("POST", "/api/chat_init")
                init = json.loads(body)

                st, body = await _asgi_request("GET", "/api/ai_backends")
                assert st == 200
                ai_bk = json.loads(body)
                assert ai_bk["offline"] is True and "claude_cli_available" in ai_bk and "codex_cli_available" in ai_bk

                st, body = await _asgi_request(
                    "POST",
                    "/api/chat",
                    json_body={
                        "session": init["session"],
                        "message": "220 nm SOI strip, TE0 at 1550 nm, R = 15 um, simulate",
                        "ai_mode": "offline",
                    },
                )
                assert st == 200
                chat = json.loads(body)
                assert "mrr_result" in chat and chat.get("ai_mode") == "offline"

                st, body = await _asgi_request("POST", "/api/export_gds", json_body=chat["session"]["config"])
                assert st == 200
                exp = json.loads(body)

                q_str = exp["gds_download_url"].split("?", 1)[1]
                st, gds_bytes = await _asgi_request("GET", "/api/download_file", query_string=q_str)
                assert st == 200 and len(gds_bytes) > 100

                b64_gds = base64.b64encode(gds_bytes).decode("ascii")
                st, body = await _asgi_request(
                    "POST",
                    "/api/upload_gds",
                    json_body={"filename": "roundtrip_test.gds", "content_b64": b64_gds, "session": chat["session"]},
                )
                assert st == 200
                up = json.loads(body)
                assert abs(up["gds_info"]["extracted_geometry"]["ring_radius_um"] - 15.0) < 0.08
            finally:
                srv.GDS_EXPORT_DIR, srv.GDS_UPLOAD_DIR = old_exp, old_up

    asyncio.run(_run())


def test_agent_tools_and_mcp_minimal_tokens() -> None:
    # 1. MCP handshake + tools/list
    init_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp is not None and init_resp["result"]["serverInfo"]["name"] == "microring-studio"
    list_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None and len(list_resp["result"]["tools"]) == 15

    # 2. Compact MCP design + critical coupling + GDS export (< 550 chars => < ~140 tokens!)
    call_resp = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "microring_design_mrr",
            "arguments": {
                "top": "pulley_ap",
                "mat": "4H-SiC",
                "h_um": 0.60,
                "w_ring_um": 1.10,
                "w_bus_um": 0.82,
                "r_um": 40.0,
                "gap_um": 0.25,
                "apply_cc": True,
                "export_gds": True,
            },
        },
    })
    assert call_resp is not None and not call_resp["result"]["isError"]
    compact_text = call_resp["result"]["content"][0]["text"]
    assert len(compact_text) < 600, f"Agent response too verbose ({len(compact_text)} chars): {compact_text}"
    parsed = json.loads(compact_text)
    assert parsed["ER_dB"] > 35.0
    assert Path(parsed["gds"]).is_file()

    # 3. Inspect exported GDS & run cached sweep (< 600 chars each)
    gds_kpi = agent_inspect_gds({"gds_path": parsed["gds"]})
    assert gds_kpi["extracted"]["topology"] == "pulley_ap"
    sw_kpi = agent_sweep_mrr({
        "top": "pulley_ap",
        "mat": "4H-SiC",
        "h_um": 0.60,
        "w_ring_um": 1.10,
        "w_bus_um": 0.82,
        "r_um": 40.0,
        "gap_um": 0.25,
        "param_name": "gap_thru_um",
        "v_min": 0.20,
        "v_max": 0.30,
        "num_points": 5,
    })
    assert len(json.dumps(sw_kpi, separators=(",", ":"))) < 450


def test_multiband_vis_1310_midir_and_warnings() -> None:
    # 1. Visible 532 nm Si3N4
    vis = simulate_mrr_topology(
        MRRDesignConfig(
            topology="point_circular_ap",
            core_material="Si3N4",
            clad_material="SiO2",
            top_clad_material="SiO2",
            film_height_um=0.18,
            ring_width_um=0.38,
            bus_width_um=0.34,
            ring_radius_um=25.0,
            gap_thru_um=0.14,
            wavelength_nm=532.0,
            span_nm=2.2,
        )
    )
    assert 1.65 < vis["metrics"]["neff_ring"] < 2.08
    assert len(vis["wavelength_warnings"]) == 0

    # 2. Datacom O-band 1310 nm SOI
    oband = simulate_mrr_topology(
        MRRDesignConfig(
            topology="point_circular_ad",
            core_material="Si",
            clad_material="SiO2",
            top_clad_material="SiO2",
            film_height_um=0.22,
            ring_width_um=0.40,
            bus_width_um=0.38,
            ring_radius_um=15.0,
            gap_thru_um=0.16,
            gap_drop_um=0.20,
            wavelength_nm=1310.0,
            span_nm=10.0,
        )
    )
    assert 2.45 < oband["metrics"]["neff_ring"] < 2.95
    assert oband["metrics"]["ng_ring"] > 3.80
    assert len(oband["wavelength_warnings"]) == 0

    # 3. Mid-IR 4200 nm Ge on CaF2
    mir = simulate_mrr_topology(
        MRRDesignConfig(
            topology="point_circular_ap",
            core_material="Ge",
            clad_material="CaF2",
            top_clad_material="Air",
            film_height_um=1.40,
            slab_height_um=0.30,
            ring_width_um=2.00,
            bus_width_um=1.80,
            ring_radius_um=65.0,
            gap_thru_um=0.40,
            wavelength_nm=4200.0,
            span_nm=35.0,
        )
    )
    assert 3.40 < mir["metrics"]["neff_ring"] < 4.05
    assert len(mir["wavelength_warnings"]) == 0

    # 4. Physical opacity warnings: c-Si at 532 nm & SiO2 cladding at 4500 nm
    bad_vis = simulate_mrr_topology(
        MRRDesignConfig(core_material="Si", wavelength_nm=532.0, film_height_um=0.18, ring_width_um=0.35, bus_width_um=0.30)
    )
    assert len(bad_vis["wavelength_warnings"]) >= 1
    assert bad_vis["metrics"]["Q_int"] < 5000


def test_fabrication_dose_and_pr_tolerance() -> None:
    assert {"positive_pr", "negative_pr", "direct_cd"}.issubset(RESIST_TONE_CATALOG.keys())
    cfg = MRRDesignConfig(
        topology="point_circular_ap",
        core_material="Si",
        film_height_um=0.22,
        ring_width_um=0.48,
        bus_width_um=0.45,
        ring_radius_um=15.0,
        gap_thru_um=0.18,
        wavelength_nm=1550.0,
    )

    # 1. Positive PR: over-dose (+15%, last corner) shrinks W, widens gap, blue-shifts lambda_0
    pos = run_fabrication_error_analysis(cfg, resist_tone="positive_pr", max_dose_error_pct=15.0, num_points=5)
    pos_over = pos["corners"][-1]
    assert pos_over["dose_error_pct"] > 0 and pos_over["delta_cd_nm"] < -15.0
    assert pos_over["gap_fab_um"] > 0.18
    assert pos_over["delta_wl_res_nm"] < -0.5

    # 2. Negative PR: over-dose (+15%, last corner) blooms W, shrinks gap, red-shifts lambda_0
    neg = run_fabrication_error_analysis(cfg, resist_tone="negative_pr", max_dose_error_pct=15.0, num_points=5)
    neg_over = neg["corners"][-1]
    assert neg_over["dose_error_pct"] > 0 and neg_over["delta_cd_nm"] > 15.0
    assert neg_over["gap_fab_um"] < 0.18
    assert neg_over["delta_wl_res_nm"] > 0.5
    assert len(neg["corner_spectra"]["nominal"]["detuning_pm"]) > 100

    # 3. 5-cell pre-biased GDS + Copilot + HTTP + MCP integration
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        gds_meta = export_fab_bias_array_gds(cfg, tmp_path / "opc_bias_test.gds")
        assert Path(gds_meta["gds_path"]).is_file()
        assert Path(gds_meta["lyp_path"]).is_file()
        assert gds_meta["num_cells"] == 5 and len(gds_meta["cells"]) == 5

        s0 = create_initial_session(lang="en")
        c_fab = process_chat_turn(
            s0,
            "check over dose or under dose nagetive PR +/- 15%",
            gds_export_dir=tmp_path,
        )
        assert c_fab["ui_view"] == "fab"
        assert c_fab["fab_result"]["resist_tone"] == "negative_pr"

        c_opc = process_chat_turn(
            c_fab["session"],
            "Export 5-cell pre-biased GDS",
            gds_export_dir=tmp_path,
        )
        assert "gds_export" in c_opc and Path(c_opc["gds_export"]["gds_path"]).is_file()

        async def _check_http() -> None:
            old_exp = srv.GDS_EXPORT_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "srv_exp"
            try:
                st, b_fab = await _asgi_request(
                    "POST",
                    "/api/fab_tolerance",
                    json_body={"config": asdict(cfg), "resist_tone": "positive_pr", "max_dose_error_pct": 12.0, "num_points": 5},
                )
                assert st == 200 and json.loads(b_fab)["resist_tone"] == "positive_pr"
                st2, b_gds = await _asgi_request(
                    "POST",
                    "/api/export_fab_gds",
                    json_body={"config": asdict(cfg)},
                )
                assert st2 == 200 and "gds_download_url" in json.loads(b_gds)
            finally:
                srv.GDS_EXPORT_DIR = old_exp

        asyncio.run(_check_http())

    mcp_fab = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {
            "name": "microring_fab_tolerance",
            "arguments": {
                "mat": "Si",
                "h_um": 0.22,
                "w_ring_um": 0.48,
                "w_bus_um": 0.45,
                "r_um": 15.0,
                "gap_um": 0.18,
                "resist": "positive_pr",
                "dose_pct": 15.0,
                "num_points": 5,
                "export_bias_gds": True,
            },
        },
    })
    assert mcp_fab is not None and not mcp_fab["result"]["isError"]
    mcp_txt = mcp_fab["result"]["content"][0]["text"]
    assert len(mcp_txt) < 620
    mcp_json = json.loads(mcp_txt)
    assert Path(mcp_json["csv"]).is_file() and Path(mcp_json["bias_gds"]).is_file()


def test_active_tuning_2d_fem_and_interconnect() -> None:
    assert {"thermo_optic", "pockels_eo", "si_pn_depletion", "si_pin_injection"}.issubset(
        TUNING_MECHANISM_CATALOG.keys()
    )

    # 1. Silicon Lateral PN Depletion (reverse bias widens W_dep -> red-shifts resonance & lowers modal FCA)
    soi_cfg = MRRDesignConfig(
        topology="point_circular_ap",
        core_material="Si",
        film_height_um=0.22,
        slab_height_um=0.09,
        ring_width_um=0.48,
        bus_width_um=0.44,
        ring_radius_um=20.0,
        gap_thru_um=0.16,
        wavelength_nm=1550.0,
        span_nm=6.0,
    )
    pn = run_active_tuning_analysis(
        soi_cfg, mechanism="si_pn_depletion", drive_type="voltage_v", bias_values=[0.0, -2.0, -5.0]
    )
    assert pn["states"][-1]["resonance_shift_pm"] > 30.0
    assert pn["states"][-1]["alpha_active_only_db_per_cm"] < pn["states"][0]["alpha_active_only_db_per_cm"]
    assert 0.5 < pn["efficiency"]["V_pi_L_v_cm"] < 6.0
    assert pn["geometry_physics"]["f_3db_eo_ghz"] > 5.0

    # 2. Silicon PIN Forward Injection (carrier plasma blue-shift + rising modal FCA)
    pin = run_active_tuning_analysis(
        soi_cfg, mechanism="si_pin_injection", drive_type="current_ma", bias_values=[0.0, 0.5, 1.5]
    )
    assert pin["states"][-1]["resonance_shift_pm"] < -40.0
    assert pin["states"][-1]["alpha_active_only_db_per_cm"] > pin["states"][0]["alpha_active_only_db_per_cm"] + 1.0

    # 3. Anisotropic Pockels EO (2D Laplace electrode gap sensitivity + direct E-field input)
    tfln_cfg = MRRDesignConfig(
        topology="racetrack_ap",
        core_material="LN_Xcut",
        film_height_um=0.60,
        slab_height_um=0.30,
        ring_width_um=1.20,
        bus_width_um=1.00,
        ring_radius_um=60.0,
        straight_length_um=60.0,
        gap_thru_um=0.30,
        wavelength_nm=1550.0,
    )
    eo_narrow = run_active_tuning_analysis(
        tfln_cfg, mechanism="pockels_eo", drive_type="voltage_v", bias_values=[-10.0, 0.0, 10.0], electrode_gap_um=3.2
    )
    eo_wide = run_active_tuning_analysis(
        tfln_cfg, mechanism="pockels_eo", drive_type="voltage_v", bias_values=[-10.0, 0.0, 10.0], electrode_gap_um=5.6
    )
    assert eo_narrow["efficiency"]["V_pi_L_v_cm"] < eo_wide["efficiency"]["V_pi_L_v_cm"]
    eo_efield = run_active_tuning_analysis(
        tfln_cfg, mechanism="pockels_eo", drive_type="e_field_v_per_um", bias_values=[-8.0, 0.0, 8.0]
    )
    assert abs(eo_efield["efficiency"]["wavelength_slope"]) > 30.0

    # 4. Thermo-Optic Microheater (2D Fourier FEM BOX thickness sensitivity + direct temperature input)
    sin_cfg = MRRDesignConfig(
        topology="point_circular_ap",
        core_material="Si3N4",
        film_height_um=0.40,
        ring_width_um=1.20,
        bus_width_um=1.00,
        ring_radius_um=40.0,
        gap_thru_um=0.25,
        wavelength_nm=1550.0,
    )
    to_thin = run_active_tuning_analysis(
        sin_cfg, mechanism="thermo_optic", drive_type="current_ma", bias_values=[0.0, 4.0, 8.0], box_thickness_um=1.8
    )
    to_thick = run_active_tuning_analysis(
        sin_cfg, mechanism="thermo_optic", drive_type="current_ma", bias_values=[0.0, 4.0, 8.0], box_thickness_um=3.6
    )
    assert to_thick["geometry_physics"]["r_th_core_k_per_mw"] > to_thin["geometry_physics"]["r_th_core_k_per_mw"]
    assert to_thick["efficiency"]["P_pi_mw"] < to_thin["efficiency"]["P_pi_mw"]

    to_dt = run_active_tuning_analysis(
        sin_cfg, mechanism="thermo_optic", drive_type="delta_temp_k", bias_values=[0.0, 20.0, 50.0]
    )
    assert to_dt["efficiency"]["wavelength_slope"] > 8.0

    # 5. Lumerical INTERCONNECT export + Copilot + ASGI HTTP + MCP (<150 tokens)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        ic = export_lumerical_interconnect_bundle(soi_cfg, pn, tmp_path / "ic_soi_pn", make_zip=True)
        assert ic["passivity_verified"] is True
        assert Path(ic["spar_dat"]).is_file() and Path(ic["active_neff_table"]).is_file()
        assert Path(ic["lsf_script"]).is_file() and Path(ic["zip_path"]).is_file()

        # Verify strictly ascending frequency in SPAR .dat and non-negative Im(neff) in OMM .txt
        spar_lines = [
            ln.split() for ln in Path(ic["spar_dat"]).read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("(")
        ]
        freqs = [float(r[0]) for r in spar_lines[:20]]
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

        omm_lines = [
            [float(x) for x in ln.split()]
            for ln in Path(ic["active_neff_table"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert all(omm_lines[i][0] < omm_lines[i + 1][0] for i in range(len(omm_lines) - 1))
        assert all(row[2] >= 0.0 for row in omm_lines)

        c_tune = process_chat_turn(
            create_initial_session(lang="en"),
            "Si PN junction tuning spectra at 0, -2, -5 V and export Lumerical INTERCONNECT",
            gds_export_dir=tmp_path,
        )
        assert c_tune["ui_view"] == "tuning"
        assert "tuning_result" in c_tune and "interconnect_export" in c_tune

        async def _check_tune_http() -> None:
            old_exp = srv.GDS_EXPORT_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "srv_ic"
            try:
                st1, b1 = await _asgi_request(
                    "POST",
                    "/api/active_tuning",
                    json_body={"config": asdict(sin_cfg), "drive_type": "delta_temp_k", "bias_values": [0, 25, 50]},
                )
                assert st1 == 200 and json.loads(b1)["efficiency"]["mechanism"] == "thermo_optic"
                st2, b2 = await _asgi_request(
                    "POST",
                    "/api/export_interconnect",
                    json_body={"config": asdict(soi_cfg), "mechanism": "si_pn_depletion", "bias_values": [0, -2, -4]},
                )
                d2 = json.loads(b2)
                assert st2 == 200 and "zip_download_url" in d2
                q_dl = d2["zip_download_url"].split("?", 1)[1]
                st3, zbytes = await _asgi_request("GET", "/api/download_file", query_string=q_dl)
                assert st3 == 200 and len(zbytes) > 500
            finally:
                srv.GDS_EXPORT_DIR = old_exp

        asyncio.run(_check_tune_http())

    mcp_tune = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 108,
        "method": "tools/call",
        "params": {
            "name": "microring_active_tuning",
            "arguments": {
                "mat": "Si",
                "h_um": 0.22,
                "slab_um": 0.09,
                "w_ring_um": 0.48,
                "w_bus_um": 0.44,
                "r_um": 20.0,
                "gap_um": 0.20,
                "mechanism": "si_pn_depletion",
                "biases": [0.0, -2.0, -5.0],
                "export_interconnect": True,
            },
        },
    })
    assert mcp_tune is not None and not mcp_tune["result"]["isError"]
    t_txt = mcp_tune["result"]["content"][0]["text"]
    assert len(t_txt) < 620, f"MCP active_tuning too verbose ({len(t_txt)} chars): {t_txt}"
    t_json = json.loads(t_txt)
    assert Path(t_json["csv"]).is_file() and Path(t_json["interconnect_zip"]).is_file()


def test_io_couplers_and_high_q_fdtd_convergence() -> None:
    assert {"SMF28_1550", "Lensed_2p5um", "DFB_Laser_Butt", "MidIR_InF3"}.issubset(
        FIBER_LASER_CATALOG.keys()
    )

    # 1. Butt / Edge Coupler (2D Vector Maxwell FEM + Poynting Overlap + Adiabatic EME)
    butt_soi = design_butt_edge_coupler(
        core_material="Si",
        wavelength_nm=1550.0,
        film_height_um=0.22,
        w_bus_um=0.45,
        taper_length_um=180.0,
        fiber_preset="Lensed_2p5um",
    )
    assert 0.12 <= butt_soi["w_tip_um"] <= 0.28
    assert butt_soi["metrics"]["total_il_db"] < 1.5
    assert butt_soi["metrics"]["tol_1db_x_um"] > 0.25
    assert butt_soi["metrics"]["return_loss_db"] > 25.0

    butt_dfb = design_butt_edge_coupler(
        core_material="Si3N4",
        wavelength_nm=1310.0,
        film_height_um=0.30,
        w_bus_um=0.90,
        taper_length_um=160.0,
        fiber_preset="DFB_Laser_Butt",
    )
    assert butt_dfb["metrics"]["total_il_db"] < 1.5
    assert butt_dfb["metrics"]["tol_1db_y_um"] > 0.18

    # 2. Surface Grating Coupler (Bragg/Apodized Synthesis + Native 2D Full-Wave Maxwell FDFD)
    gc_unif = design_grating_coupler(
        core_material="Si",
        wavelength_nm=1550.0,
        film_height_um=0.22,
        etch_depth_um=0.07,
        fiber_angle_deg=10.0,
        apodized=False,
        focusing=True,
        box_thickness_um=2.0,
        fiber_preset="SMF28_1550",
    )
    gc_apod = design_grating_coupler(
        core_material="Si",
        wavelength_nm=1550.0,
        film_height_um=0.22,
        etch_depth_um=0.07,
        fiber_angle_deg=10.0,
        apodized=True,
        focusing=True,
        box_thickness_um=2.0,
        fiber_preset="SMF28_1550",
    )
    assert 1.5 < gc_unif["metrics"]["peak_il_db"] < 4.5
    assert 1.5 < gc_apod["metrics"]["peak_il_db"] < gc_unif["metrics"]["peak_il_db"] + 0.1
    assert gc_apod["metrics"]["bw_1db_nm"] > 20.0
    assert gc_apod["metrics"]["s11_db"] < -15.0
    assert gc_apod["field_2d"] is not None and len(gc_apod["field_2d"]["E_norm"]) > 20

    # 3. High-Q & Ultra-High-Q (Q > 1.5e7) Convergence + Coupler Cutout + GDS/INTERCONNECT/Copilot/HTTP/MCP
    hq_seed = MRRDesignConfig(
        topology="pulley_ap",
        core_material="Si3N4",
        film_height_um=0.40,
        ring_width_um=1.30,
        bus_width_um=0.95,
        ring_radius_um=80.0,
        gap_thru_um=0.35,
        pulley_angle_thru_deg=14.0,
        wavelength_nm=1550.0,
        alpha_loss_db_per_cm=0.06,
    )
    hq_scout = simulate_mrr_topology(hq_seed)
    cc_param = hq_scout["critical_coupling"]["parameter"]
    cc_val = float(hq_scout["critical_coupling"]["recommended_value"])
    hq_cfg = replace(hq_seed, **{cc_param: cc_val})

    hq = analyze_high_q_fdtd_convergence(hq_cfg, sim_times_ps=[25.0, 200.0, 1000.0])
    assert hq["Q_loaded"] > 2.0e6
    assert hq["required_fdtd_sim_time_1e5_ns"] > 15.0
    assert hq["true_converged_er_db"] > 25.0
    assert hq["truncated_fdtd_traces"][0]["apparent_er_db"] < hq["true_converged_er_db"] - 10.0
    assert hq["time_domain_speedup_factor"] > 10000

    # Extreme Ultra-High-Q (alpha = 0.01 dB/cm -> Q_int ~ 3.6e7, Q_loaded ~ 1.8e7 at critical coupling)
    uhq_seed = replace(hq_seed, topology="point_circular_ap", alpha_loss_db_per_cm=0.01, gap_thru_um=0.65, span_nm=2.0)
    uhq_cc_gap = float(simulate_mrr_topology(uhq_seed)["critical_coupling"]["recommended_value"])
    uhq_spec = simulate_mrr_topology(replace(uhq_seed, gap_thru_um=uhq_cc_gap))
    assert uhq_spec["metrics"]["Q_loaded"] > 1.5e7
    assert uhq_spec["metrics"]["FWHM_pm"] < 0.15

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        gds_loop = export_io_coupler_gds(
            gc_apod, tmp_path / "gc_loopback.gds", include_ring_loopback=True, ring_config=hq_cfg
        )
        assert Path(gds_loop["gds_path"]).is_file() and Path(gds_loop["lyp_path"]).is_file()
        assert len(gds_loop["cells"]) >= 3

        ic_gc = export_io_coupler_interconnect(gc_apod, tmp_path / "ic_gc")
        assert ic_gc["passivity_verified"] is True
        assert Path(ic_gc["spar_path"]).is_file() and Path(ic_gc["zip_path"]).is_file()

        cut = export_coupler_cutout_gds(hq_cfg, tmp_path / "cutout.gds")
        assert Path(cut["gds_path"]).is_file() and Path(cut["lsf_path"]).is_file()

        # Copilot EN + ZH turns
        c_gc = process_chat_turn(
            create_initial_session(lang="en"),
            "Design apodized focusing grating coupler on 220 nm Si and export 127um loopback GDS",
            gds_export_dir=tmp_path,
        )
        assert c_gc["ui_view"] == "io" and c_gc["io_result"]["coupler_type"] == "grating_coupler"
        assert "gds_export" in c_gc

        c_hq_zh = process_chat_turn(
            create_initial_session(lang="zh"),
            "为什么高Q氮化硅微环在 Lumerical FDTD 中很慢无法收敛？导出耦合区切片 GDS",
            gds_export_dir=tmp_path,
        )
        assert c_hq_zh["ui_view"] == "io" and "high_q_result" in c_hq_zh and "gds_export" in c_hq_zh

        # ASGI HTTP endpoints
        async def _check_io_http() -> None:
            old_exp = srv.GDS_EXPORT_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "srv_io"
            try:
                st1, b1 = await _asgi_request(
                    "POST",
                    "/api/io_coupler",
                    json_body={"coupler_type": "butt_edge", "core_material": "Si", "film_height_um": 0.22, "fiber_preset": "Lensed_2p5um"},
                )
                d1 = json.loads(b1)
                assert st1 == 200 and d1["coupler_type"] == "butt_edge"

                st2, b2 = await _asgi_request(
                    "POST",
                    "/api/export_io_gds",
                    json_body={"coupler_result": d1, "include_ring_loopback": True},
                )
                assert st2 == 200 and "gds_download_url" in json.loads(b2)

                st3, b3 = await _asgi_request(
                    "POST",
                    "/api/export_io_interconnect",
                    json_body={"coupler_result": d1},
                )
                assert st3 == 200 and json.loads(b3)["passivity_verified"] is True

                st4, b4 = await _asgi_request(
                    "POST",
                    "/api/high_q_fdtd",
                    json_body={"config": asdict(hq_cfg), "export_cutout": True},
                )
                d4 = json.loads(b4)
                assert st4 == 200 and "cutout" in d4 and "lsf_download_url" in d4["cutout"]
            finally:
                srv.GDS_EXPORT_DIR = old_exp

        asyncio.run(_check_io_http())

    # Agent CLI helpers & MCP (<150 tokens)
    a_io = agent_io_coupler({"coupler_type": "butt_edge", "mat": "Si", "h_um": 0.22, "export_gds": True})
    assert a_io["IL_total_dB"] < 1.5 and Path(a_io["gds"]).is_file()
    a_hq = agent_high_q_fdtd({"mat": "Si3N4", "r_um": 80.0, "loss_db_cm": 0.08, "apply_cc": True, "export_cutout_gds": True})
    assert a_hq["Q_L"] > 1.5e6 and Path(a_hq["cutout_gds"]).is_file()

    mcp_io = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 112,
        "method": "tools/call",
        "params": {
            "name": "microring_io_coupler",
            "arguments": {"coupler_type": "grating_coupler", "mat": "Si", "h_um": 0.22, "etch_um": 0.07, "export_interconnect": True},
        },
    })
    assert mcp_io is not None and not mcp_io["result"]["isError"]
    io_txt = mcp_io["result"]["content"][0]["text"]
    assert len(io_txt) < 620, f"MCP io_coupler too verbose ({len(io_txt)} chars): {io_txt}"

    mcp_hq = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 113,
        "method": "tools/call",
        "params": {
            "name": "microring_high_q_fdtd",
            "arguments": {"mat": "Si3N4", "r_um": 90.0, "loss_db_cm": 0.08, "gap_um": 0.38, "export_cutout_gds": True},
        },
    })
    assert mcp_hq is not None and not mcp_hq["result"]["isError"]
    hq_txt = mcp_hq["result"]["content"][0]["text"]
    assert len(hq_txt) < 620, f"MCP high_q_fdtd too verbose ({len(hq_txt)} chars): {hq_txt}"


def test_mzi_splitters_tuning_gds_and_interconnect() -> None:
    assert {"directional_coupler", "mmi_2x2", "mmi_1x2", "y_branch"}.issubset(MZI_SPLITTER_CATALOG.keys())

    # 1. All 4 MZI 50:50 splitters without 3D FDTD (2D-FEM Supermode + Multi-Mode EME + Apex Overlap)
    for sp_key in ("directional_coupler", "mmi_2x2", "mmi_1x2", "y_branch"):
        res_sp = simulate_mzi(
            MZIDesignConfig(
                core_material="Si",
                film_height_um=0.22,
                slab_height_um=0.0,
                arm1_width_um=0.45,
                arm1_length_um=300.0,
                delta_length_um=80.0,
                splitter_type=sp_key,
                span_nm=24.0,
                num_spectrum_points=1001,
            ),
            include_active_tuning=False,
        )
        ss = res_sp["splitter"]["summary"]
        mm = res_sp["metrics"]
        assert ss["excess_loss_db"] < 1.35, f"{sp_key} excess loss too high: {ss['excess_loss_db']} dB"
        assert 35.0 <= ss["split_bar_pct_at_wl0"] <= 52.0
        assert 5.5 <= mm["FSR_nm"] <= 9.0
        assert mm["ER_bar_passive_dB"] > 22.0
        # Strict passivity check across entire spectrum
        bar_t = np.asarray(res_sp["passive_spectrum"]["bar_T"], dtype=float)
        cross_t = (
            np.asarray(res_sp["passive_spectrum"]["cross_T"], dtype=float)
            if res_sp["has_cross_port"]
            else np.zeros_like(bar_t)
        )
        assert float(np.max(bar_t + cross_t)) <= 1.0001
        if sp_key in ("mmi_2x2", "mmi_1x2"):
            fmap = res_sp["splitter"]["field_map"]
            assert fmap is not None and len(fmap["E_norm"]) > 20 and ss["guided_modes_solved"] >= 4

    # 2. Active MZI Modulators: Si PN Push-Pull, X-Cut TFLN EO Push-Pull, & Si3N4 Thermo-Optic + RAMZI
    pn_mzi = simulate_mzi(
        MZIDesignConfig(
            core_material="Si",
            film_height_um=0.22,
            slab_height_um=0.09,
            arm1_width_um=0.48,
            arm1_length_um=1500.0,
            delta_length_um=60.0,
            splitter_type="mmi_2x2",
            span_nm=18.0,
        ),
        include_active_tuning=True,
        mechanism="si_pn_depletion",
        drive_scheme="push_pull",
        bias_values=[0.0, -2.0, -4.0],
    )
    act_pn = pn_mzi["active_tuning"]
    assert act_pn is not None
    assert 0.45 < act_pn["v_pi_l_vcm"] < 2.20
    assert act_pn["bandwidth_3db"] > 3.5
    assert act_pn["dynamic_er_bar_db"] > 18.0

    tfln_mzi = simulate_mzi(
        MZIDesignConfig(
            core_material="LN_Xcut",
            film_height_um=0.60,
            slab_height_um=0.30,
            arm1_width_um=1.20,
            arm1_length_um=5000.0,
            delta_length_um=120.0,
            splitter_type="y_branch",
            span_nm=20.0,
        ),
        include_active_tuning=True,
        mechanism="pockels_eo",
        drive_scheme="push_pull",
        bias_values=[-4.0, 0.0, 4.0],
    )
    act_ln = tfln_mzi["active_tuning"]
    assert act_ln is not None
    assert 1.20 < act_ln["v_pi_l_vcm"] < 3.40
    assert act_ln["bandwidth_3db"] > 8.0
    assert act_ln["dynamic_er_bar_db"] > 20.0

    sin_ramzi = simulate_mzi(
        MZIDesignConfig(
            core_material="Si3N4",
            film_height_um=0.40,
            arm1_width_um=1.10,
            arm1_length_um=800.0,
            delta_length_um=90.0,
            splitter_type="directional_coupler",
            ramzi_enabled=True,
            ramzi_ring_radius_um=45.0,
            ramzi_ring_gap_um=0.25,
            span_nm=16.0,
        ),
        include_active_tuning=True,
        mechanism="thermo_optic",
        drive_type="power_mw",
        drive_scheme="single_arm",
        bias_values=[0.0, 40.0, 80.0],
    )
    assert sin_ramzi["ramzi"] is not None and sin_ramzi["ramzi"]["ring_Q_loaded"] > 2000
    assert sin_ramzi["active_tuning"]["p_pi_mw"] > 20.0
    assert sin_ramzi["active_tuning"]["bandwidth_3db"] > 2.0

    # 3. GDS round-trip recognition + Lumerical INTERCONNECT SPAR/OMM/.lsf + Copilot + HTTP + Agent/MCP
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        gds_mzi = export_mzi_gds(pn_mzi, tmp_path / "mzi_pn_test.gds")
        assert Path(gds_mzi["gds_path"]).is_file() and Path(gds_mzi["lyp_path"]).is_file()
        insp = analyze_uploaded_gds(gds_mzi["gds_path"])
        assert insp["extracted_geometry"]["topology"] == "mzi_interferometer"
        assert insp["extracted_geometry"]["is_non_ring"] is True

        ic_mzi = export_mzi_interconnect_bundle(pn_mzi, tmp_path / "mzi_ic_test")
        assert ic_mzi["passivity_verified"] is True
        assert Path(ic_mzi["spar_path"]).is_file() and Path(ic_mzi["omm_path"]).is_file()
        assert Path(ic_mzi["lsf_path"]).is_file() and Path(ic_mzi["zip_path"]).is_file()

        # Copilot EN + ZH MZI turns
        c_mzi_en = process_chat_turn(
            create_initial_session(lang="en"),
            "Design a 220 nm SOI 2x2 MMI push-pull PN MZI modulator and export GDS and INTERCONNECT",
            gds_export_dir=tmp_path,
        )
        assert c_mzi_en["ui_view"] == "mzi" and "mzi_result" in c_mzi_en
        assert "gds_export" in c_mzi_en and "interconnect_export" in c_mzi_en

        c_mzi_zh = process_chat_turn(
            create_initial_session(lang="zh"),
            "设计 Y 分支马赫曾德尔调制器",
            gds_export_dir=tmp_path,
        )
        assert c_mzi_zh["ui_view"] == "mzi" and c_mzi_zh["mzi_result"]["splitter"]["splitter_type"] == "y_branch"

        async def _check_mzi_http() -> None:
            old_exp = srv.GDS_EXPORT_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "srv_mzi"
            try:
                st0, b0 = await _asgi_request("GET", "/")
                assert st0 == 200 and b"pill-mzi" in b0

                st1, b1 = await _asgi_request(
                    "POST",
                    "/api/design_mzi",
                    json_body={
                        "config": {"core_material": "Si", "film_height_um": 0.22, "slab_height_um": 0.09, "arm1_width_um": 0.48, "arm1_length_um": 1200.0, "delta_length_um": 70.0, "splitter_type": "directional_coupler"},
                        "include_active_tuning": True,
                        "mechanism": "si_pn_depletion",
                    },
                )
                d1 = json.loads(b1)
                assert st1 == 200 and d1["splitter"]["splitter_type"] == "directional_coupler"
                assert d1["active_tuning"]["v_pi_l_vcm"] > 0.3

                st2, b2 = await _asgi_request("POST", "/api/export_mzi_gds", json_body={"mzi_result": d1})
                assert st2 == 200 and "gds_download_url" in json.loads(b2)

                st3, b3 = await _asgi_request("POST", "/api/export_mzi_interconnect", json_body={"mzi_result": d1})
                d3 = json.loads(b3)
                assert st3 == 200 and d3["passivity_verified"] is True and "zip_download_url" in d3
            finally:
                srv.GDS_EXPORT_DIR = old_exp

        asyncio.run(_check_mzi_http())

    a_mzi = agent_design_mzi({"mat": "Si", "h_um": 0.22, "slab_um": 0.09, "splitter": "mmi_2x2", "l1_um": 1200.0, "dl_um": 80.0, "export_gds": True})
    assert a_mzi["tuning"]["V_pi_L_Vcm"] > 0.3 and Path(a_mzi["gds"]).is_file()

    mcp_mzi = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 114,
        "method": "tools/call",
        "params": {
            "name": "microring_design_mzi",
            "arguments": {"mat": "LN_Xcut", "h_um": 0.60, "slab_um": 0.30, "w_um": 1.20, "splitter": "y_branch", "l1_um": 4000.0, "dl_um": 100.0, "export_interconnect": True},
        },
    })
    assert mcp_mzi is not None and not mcp_mzi["result"]["isError"]
    mzi_txt = mcp_mzi["result"]["content"][0]["text"]
    assert len(mzi_txt) < 620, f"MCP design_mzi too verbose ({len(mzi_txt)} chars): {mzi_txt}"


def test_eom_electrode_pn_tcad_and_thermal() -> None:
    assert {"gsg_cpw", "t_rail_cpw", "gs_coplanar", "vertical_microstrip", "lumped_ring_gsg"}.issubset(EOM_ELECTRODE_CATALOG.keys())
    assert {
        "x_cut_ln", "y_cut_ln", "z_cut_ln",
        "x_cut_lt", "z_cut_lt",
        "a_cut_bto_si", "c_cut_bto", "inplane_poled_pzt",
        "c_plane_6mm", "a_plane_6mm", "cubic_001_43m", "cubic_110_43m",
    }.issubset(EO_WAFER_CUT_CATALOG.keys())
    assert {"lateral_pn", "l_shaped_pn", "u_shaped_pn", "pin_diode"}.issubset(PN_JUNCTION_CATALOG.keys())
    assert {"top_heater", "deep_trench", "undercut_suspended", "doped_si_rib"}.issubset(THERMAL_STRUCTURE_CATALOG.keys())

    # 0. Exact 3D Crystal-Tensor EO Wafer Orientation & In-Plane Azimuth Finder
    wo_xln = analyze_eo_wafer_orientation(core_material="LN_Xcut", wafer_cut="x_cut_ln", polarization="TE0")
    assert wo_xln["metrics"]["optimal_azimuth_deg"] == 0.0
    assert abs(wo_xln["metrics"]["r_eff_optimal_pm_per_v"] - 30.9) < 0.2
    assert abs(wo_xln["angular_sweep"]["r_eff_pm_per_v"][90] - (-6.8)) < 0.2   # 90 deg rotates to -r22!
    assert abs(wo_xln["angular_sweep"]["r_eff_pm_per_v"][180] - (-30.9)) < 0.2  # 180 deg flips sign -> unsplit ring cancels!
    assert wo_xln["metrics"]["ring_unsplit_efficiency_pct"] < 1.0
    assert wo_xln["metrics"]["ring_split_pushpull_efficiency_pct"] > 60.0

    # Thin-Film Lithium Tantalate (X-cut TFLT, 3m): ~30.5 pm/V along Y (phi=0) with 19x lower birefringence than X-cut TFLN!
    wo_xlt = analyze_eo_wafer_orientation(core_material="LT_Xcut", wafer_cut="x_cut_lt", polarization="TE0")
    assert wo_xlt["metrics"]["optimal_azimuth_deg"] == 0.0
    assert abs(wo_xlt["metrics"]["r_eff_optimal_pm_per_v"] - 30.5) < 0.2
    assert wo_xlt["metrics"]["birefringence_span_dn"] < 0.006
    assert wo_xln["metrics"]["birefringence_span_dn"] > 10.0 * wo_xlt["metrics"]["birefringence_span_dn"]

    # Epitaxial a-oriented BTO-on-Si (4mm): giant shear r42=360 pm/V peaks near diagonal phi~51 deg (|| Si [110] flat), >5x stronger than phi=0!
    wo_bto = analyze_eo_wafer_orientation(core_material="BTO", wafer_cut="a_cut_bto_si", polarization="TE0")
    assert 42.0 <= wo_bto["metrics"]["optimal_azimuth_deg"] <= 56.0
    assert abs(wo_bto["metrics"]["r_eff_optimal_pm_per_v"]) > 250.0
    assert abs(wo_bto["angular_sweep"]["r_eff_pm_per_v"][0]) < 60.0

    # Lanthanide-seeded MPB PZT (4mm electrode-self-poled, Alexander Nat. Commun. 2018): clamped RF r_eff = 67 pm/V -> 100% unsplit 360 deg ring!
    wo_pzt = analyze_eo_wafer_orientation(core_material="PZT", wafer_cut="inplane_poled_pzt", polarization="TE0")
    assert abs(wo_pzt["metrics"]["r_eff_optimal_pm_per_v"] - 67.0) < 1.0
    assert wo_pzt["metrics"]["ring_unsplit_efficiency_pct"] > 99.0

    # Cubic -43m (001) 3C-SiC: [100] cleavage (0 deg) is dead (r_eff=0), [110] SEMI flat (45/135 deg) peaks at |r41|=2.7 pm/V!
    wo_3c = analyze_eo_wafer_orientation(core_material="3C-SiC", wafer_cut="cubic_001_43m", polarization="TE0")
    assert abs(wo_3c["angular_sweep"]["r_eff_pm_per_v"][0]) < 1e-6
    assert wo_3c["metrics"]["optimal_azimuth_deg"] in (45.0, 135.0)
    assert abs(abs(wo_3c["metrics"]["r_eff_optimal_pm_per_v"]) - 2.7) < 0.1

    # C-Plane 6mm Wurtzite AlN & ScAlN (Yang et al., Nat. Commun. 15, 6168, 2024):
    # Opposite-sign AlN r33=+1.50 / r13=-0.92 pm/V; no r33 gain over AlN at <=20% Sc; r13 flips sign near ~16% Sc; enhanced at >=30% Sc!
    wo_aln_te = analyze_eo_wafer_orientation(core_material="AlN", wafer_cut="c_plane_6mm", polarization="TE0")
    wo_aln_tm = analyze_eo_wafer_orientation(core_material="AlN", wafer_cut="c_plane_6mm", polarization="TM0")
    wo_sc10_tm = analyze_eo_wafer_orientation(core_material="ScAlN_10", wafer_cut="c_plane_6mm", polarization="TM0")
    wo_scaln = analyze_eo_wafer_orientation(core_material="ScAlN_20", wafer_cut="c_plane_6mm", polarization="TM0")
    wo_sc20_te = analyze_eo_wafer_orientation(core_material="ScAlN_20", wafer_cut="c_plane_6mm", polarization="TE0")
    wo_sc30_tm = analyze_eo_wafer_orientation(core_material="ScAlN_30", wafer_cut="c_plane_6mm", polarization="TM0")
    assert abs(wo_aln_tm["metrics"]["r_eff_optimal_pm_per_v"] - 1.50) < 0.05
    assert abs(wo_aln_te["metrics"]["r_eff_optimal_pm_per_v"] - (-0.92)) < 0.05
    assert abs(wo_sc10_tm["metrics"]["r_eff_optimal_pm_per_v"]) < abs(wo_aln_tm["metrics"]["r_eff_optimal_pm_per_v"])
    assert abs(wo_scaln["metrics"]["r_eff_optimal_pm_per_v"]) < abs(wo_aln_tm["metrics"]["r_eff_optimal_pm_per_v"])
    assert wo_sc20_te["metrics"]["r_eff_optimal_pm_per_v"] > 0.0  # r13 flips from negative (AlN) to positive above ~16% Sc!
    assert wo_sc30_tm["metrics"]["r_eff_optimal_pm_per_v"] > 1.80
    assert wo_scaln["metrics"]["ring_unsplit_efficiency_pct"] > 99.0

    # 1. RF Traveling-Wave GSG CPW + Azimuthal Rotation Penalty + Slow-Wave T-Rail CPW
    rf_gsg = design_eom_electrode(
        EOMElectrodeConfig(
            electrode_topology="gsg_cpw",
            core_material="LN_Xcut",
            substrate_type="hr_si",
            electrode_gap_um=5.0,
            active_length_um=5000.0,
            run_gap_sweep=True,
        )
    )
    assert 47.0 <= rf_gsg["metrics"]["Z0_ohm"] <= 54.0
    assert 1.5 <= rf_gsg["metrics"]["V_pi_L_Vcm"] <= 3.2
    assert rf_gsg["metrics"]["f_3dBo_GHz"] >= 60.0
    assert rf_gsg["field_map"] is not None and len(rf_gsg["field_map"]["e_mag_v_per_um"]) > 20
    assert rf_gsg["wafer_orientation"]["metrics"]["active_azimuth_deg"] == 0.0

    # Rotating X-cut TFLN waveguide by 90 deg (propagating along Z instead of Y) switches r33 -> r22 and degrades V_pi*L by >3.5x!
    rf_misaligned = design_eom_electrode(
        EOMElectrodeConfig(
            electrode_topology="gsg_cpw",
            core_material="LN_Xcut",
            waveguide_azimuth_deg=90.0,
            substrate_type="hr_si",
            electrode_gap_um=5.0,
            active_length_um=5000.0,
            run_gap_sweep=False,
        )
    )
    assert rf_misaligned["metrics"]["V_pi_L_Vcm"] > 3.5 * rf_gsg["metrics"]["V_pi_L_Vcm"]
    assert bool(rf_misaligned["wafer_orientation"]["mismatch_warning"])

    rf_trail = design_eom_electrode(
        EOMElectrodeConfig(
            electrode_topology="t_rail_cpw",
            core_material="LN_Xcut",
            substrate_type="quartz",
            electrode_gap_um=5.0,
            active_length_um=5000.0,
            run_gap_sweep=False,
        )
    )
    assert 47.0 <= rf_trail["metrics"]["Z0_ohm"] <= 54.5
    assert rf_trail["dimensions"]["t_rail_length_um"] > 1.0
    assert rf_trail["metrics"]["f_3dBo_GHz"] >= 80.0

    # Giant-Pockels BTO & PZT RF EOM Electrodes (ultra-sub-V*cm modulation efficiency!)
    rf_bto = design_eom_electrode(
        EOMElectrodeConfig(
            electrode_topology="gsg_cpw",
            core_material="BTO",
            substrate_type="hr_si",
            film_height_um=0.35,
            slab_height_um=0.12,
            waveguide_width_um=0.90,
            electrode_gap_um=4.0,
            active_length_um=1500.0,
            run_gap_sweep=False,
        )
    )
    assert rf_bto["metrics"]["V_pi_L_Vcm"] < 0.65
    assert 42.0 <= rf_bto["wafer_orientation"]["metrics"]["active_azimuth_deg"] <= 56.0

    rf_pzt = design_eom_electrode(
        EOMElectrodeConfig(
            electrode_topology="gsg_cpw",
            core_material="PZT",
            substrate_type="hr_si",
            film_height_um=0.32,
            slab_height_um=0.10,
            waveguide_width_um=0.85,
            electrode_gap_um=4.0,
            active_length_um=1500.0,
            run_gap_sweep=False,
        )
    )
    assert rf_pzt["metrics"]["V_pi_L_Vcm"] < 1.65

    # 2. 2D Nonlinear Poisson-Boltzmann Silicon PN Doping TCAD (Lateral vs L-Shaped)
    pn_lat = simulate_pn_doping_tcad(
        PNDopingTCADConfig(
            junction_type="lateral_pn",
            bias_voltages_v=[0.0, -1.5, -3.0, -4.5],
            map_bias_v=-3.0,
        )
    )
    pn_l = simulate_pn_doping_tcad(
        PNDopingTCADConfig(
            junction_type="l_shaped_pn",
            bias_voltages_v=[0.0, -1.5, -3.0, -4.5],
            map_bias_v=-3.0,
        )
    )
    # C_j(V) monotonically decreases with reverse bias as depletion region widens
    cjs_lat = [s["cj_ff_per_um"] for s in pn_lat["states"]]
    assert cjs_lat[0] > cjs_lat[1] > cjs_lat[2] > cjs_lat[3]
    assert 0.15 <= pn_lat["metrics"]["Cj_bias_fF_per_um"] <= 0.45
    assert 0.35 <= pn_lat["metrics"]["R_series_ohm_cm"] <= 2.50
    assert pn_lat["metrics"]["f_RC_intrinsic_GHz"] > 25.0
    # L-shaped PN increases optical overlap -> strictly lower V_pi*L and V_pi*L*alpha than lateral PN
    assert pn_l["metrics"]["V_pi_L_Vcm"] < pn_lat["metrics"]["V_pi_L_Vcm"]
    assert pn_l["metrics"]["V_pi_L_alpha_V_dB"] < pn_lat["metrics"]["V_pi_L_alpha_V_dB"]

    # 3. 2D Steady-State + Crank-Nicolson Transient Thermal FEM
    th_top = simulate_thermal_phase_shifter(ThermalSimConfig(thermal_structure="top_heater", core_material="Si"))
    th_dop = simulate_thermal_phase_shifter(ThermalSimConfig(thermal_structure="doped_si_rib", core_material="Si"))
    th_tr = simulate_thermal_phase_shifter(ThermalSimConfig(thermal_structure="deep_trench", core_material="Si"))
    th_uc = simulate_thermal_phase_shifter(ThermalSimConfig(thermal_structure="undercut_suspended", core_material="Si"))
    assert th_tr["metrics"]["P_pi_mW"] < th_top["metrics"]["P_pi_mW"]
    assert th_dop["metrics"]["tau_thermal_us"] < th_top["metrics"]["tau_thermal_us"]
    assert th_tr["metrics"]["crosstalk_at_neighbor_pct"] < th_top["metrics"]["crosstalk_at_neighbor_pct"]
    assert th_uc["metrics"]["P_pi_mW"] < 2.5 and th_uc["metrics"]["P_pi_mW"] < 0.15 * th_top["metrics"]["P_pi_mW"]
    assert th_top["metrics"]["t_rise_10_90_us"] > 2.0 and th_top["metrics"]["t_fall_90_10_us"] > 2.0

    # 4. Cleanroom .GDS + .S2P/TCAD Bundles + Copilot (EN + ZH) + HTTP + Agent/MCP
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        gds_rf = export_eom_electrode_gds(rf_trail, tmp_path / "eom_trail.gds")
        assert Path(gds_rf["gds_path"]).is_file() and Path(gds_rf["lyp_path"]).is_file()
        insp_rf = analyze_uploaded_gds(gds_rf["gds_path"])
        assert insp_rf["extracted_geometry"]["topology"] == "eom_traveling_wave"

        b_rf = export_eom_rf_interconnect_bundle(rf_trail, tmp_path / "rf_bundle")
        assert Path(b_rf["s2p_path"]).is_file() and Path(b_rf["zip_path"]).is_file()

        gds_pn = export_pn_doping_gds(pn_l, tmp_path / "si_pn_l.gds")
        assert {5, 6, 11, 12}.issubset(set(gds_pn["layers"]))
        b_pn = export_tcad_data_bundle(pn_l, tmp_path / "pn_bundle")
        assert Path(b_pn["zip_path"]).is_file()

        gds_th = export_thermal_shifter_gds(th_tr, tmp_path / "th_trench.gds")
        assert 40 in gds_th["layers"]

        c_rf = process_chat_turn(
            create_initial_session(lang="en"),
            "Design slow-wave T-rail EOM electrode on quartz X-cut TFLN and export GDS",
            gds_export_dir=tmp_path,
        )
        assert c_rf["ui_view"] == "eom" and c_rf["eom_result"]["resolved_topology"] == "t_rail_cpw"
        assert "gds_export" in c_rf

        c_pn_zh = process_chat_turn(
            create_initial_session(lang="zh"),
            "仿真硅基 L 型 PN 掺杂结并导出 GDS",
            gds_export_dir=tmp_path,
        )
        assert c_pn_zh["ui_view"] == "eom" and c_pn_zh["eom_result"]["mode"] == "pn_doping_tcad"
        assert "gds_export" in c_pn_zh

        c_th = process_chat_turn(
            create_initial_session(lang="en"),
            "Run undercut thermal phase shifter transient simulation",
            gds_export_dir=tmp_path,
        )
        assert c_th["ui_view"] == "eom" and c_th["eom_result"]["mode"] == "thermal_fem"

        c_wo = process_chat_turn(
            create_initial_session(lang="en"),
            "eo need to find wafer orientation for X-cut TFLN",
            gds_export_dir=tmp_path,
        )
        assert c_wo["ui_view"] == "eom" and c_wo["eom_result"]["mode"] == "wafer_orientation"
        assert c_wo["eom_result"]["metrics"]["optimal_azimuth_deg"] == 0.0

        c_bto = process_chat_turn(
            create_initial_session(lang="en"),
            "find BTO wafer orientation on Si",
            gds_export_dir=tmp_path,
        )
        assert c_bto["ui_view"] == "eom" and c_bto["eom_result"]["mode"] == "wafer_orientation"
        assert abs(c_bto["eom_result"]["metrics"]["r_eff_optimal_pm_per_v"]) > 250.0
        assert c_bto["eom_result"]["metrics"]["eo_overlap_source"] == "solved_2d_fem_cross_section"

        # 5. Strict User-Suggested Parameter Calculation (Zero Preset Lock-In)
        wo_g32 = analyze_eo_wafer_orientation(
            core_material="LT_Xcut",
            film_height_um=0.48,
            slab_height_um=0.18,
            waveguide_width_um=1.05,
            electrode_gap_um=3.2,
            eo_overlap_factor=None,
        )
        wo_g64 = analyze_eo_wafer_orientation(
            core_material="LT_Xcut",
            film_height_um=0.48,
            slab_height_um=0.18,
            waveguide_width_um=1.05,
            electrode_gap_um=6.4,
            eo_overlap_factor=None,
        )
        assert wo_g32["metrics"]["eo_overlap_source"] == "solved_2d_fem_cross_section"
        assert wo_g32["metrics"]["V_pi_L_optimal_Vcm"] < wo_g64["metrics"]["V_pi_L_optimal_Vcm"]

        rf_pzt_custom = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="gsg_cpw",
                core_material="PZT",
                substrate_type="hr_si",
                film_height_um=0.40,
                slab_height_um=0.12,
                waveguide_width_um=0.95,
                electrode_gap_um=4.0,
                active_length_um=1500.0,
                custom_r33_pm_v=220.0,
                run_gap_sweep=False,
            )
        )
        assert abs(rf_pzt_custom["metrics"]["r_eff_pm_per_v"] - 220.0) < 0.5
        assert rf_pzt_custom["metrics"]["V_pi_L_Vcm"] < 0.65 * rf_pzt["metrics"]["V_pi_L_Vcm"]

        th_custom_dndt = simulate_thermal_phase_shifter(
            ThermalSimConfig(
                thermal_structure="top_heater",
                core_material="Si",
                custom_dn_dt_per_k=3.72e-4,
            )
        )
        assert abs(th_custom_dndt["metrics"]["P_pi_mW"] - 0.5 * th_top["metrics"]["P_pi_mW"]) < 1.2

        c_user_bto = process_chat_turn(
            create_initial_session(lang="en"),
            "BTO EOM electrode H=420 nm, slab=150 nm, W=1.05 um, gap=3.2 um, L=2 mm, r42=600 pm/V",
            gds_export_dir=tmp_path,
        )
        assert c_user_bto["ui_view"] == "eom" and c_user_bto["eom_result"]["mode"] == "rf_electrode"
        u_cfg = c_user_bto["eom_result"]["config"]
        assert abs(u_cfg["film_height_um"] - 0.42) < 1e-4
        assert abs(u_cfg["slab_height_um"] - 0.15) < 1e-4
        assert abs(u_cfg["waveguide_width_um"] - 1.05) < 1e-4
        assert abs(u_cfg["electrode_gap_um"] - 3.2) < 1e-4
        assert abs(u_cfg["active_length_um"] - 2000.0) < 1e-2
        assert abs(u_cfg["custom_r51_pm_v"] - 600.0) < 1e-3
        assert abs(c_user_bto["eom_result"]["metrics"]["r_eff_pm_per_v"]) > 400.0
        assert c_user_bto["eom_result"]["metrics"]["f_3dBo_GHz"] > c_user_bto["eom_result"]["metrics"]["f_3dBe_GHz"] > 5.0
        assert "limiting_mechanism" in c_user_bto["eom_result"]["bandwidth_analysis"]

        # 6. Modulator Bandwidth Decomposition + Microring Modulator CMT Optical Peaking
        mrm_slope = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="lumped_ring_gsg",
                core_material="PZT",
                ring_radius_um=30.0,
                racetrack_straight_um=0.0,
                active_length_um=188.5,
                ring_loaded_q=12000.0,
                ring_detuning_norm=0.57735,
                run_gap_sweep=False,
            )
        )
        mrm_peaked = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="lumped_ring_gsg",
                core_material="PZT",
                ring_radius_um=30.0,
                racetrack_straight_um=0.0,
                active_length_um=188.5,
                ring_loaded_q=12000.0,
                ring_detuning_norm=1.45,
                run_gap_sweep=False,
            )
        )
        assert mrm_peaked["bandwidth_analysis"]["optical_peaking_dBe"] > 0.5
        assert mrm_peaked["metrics"]["f_3dBe_GHz"] > 1.35 * mrm_slope["metrics"]["f_3dBe_GHz"]
        assert pn_lat["states"][-1]["f_tw_3dbo_ghz"] > pn_lat["states"][0]["f_tw_3dbo_ghz"]

        c_bw = process_chat_turn(
            create_initial_session(lang="en"),
            "calculate bandwidth of BTO microring modulator Q=12000 detuning=1.4",
            gds_export_dir=tmp_path,
        )
        assert c_bw["ui_view"] == "eom" and c_bw["eom_result"]["resolved_topology"] == "lumped_ring_gsg"
        assert c_bw["eom_result"]["metrics"]["f_3dBe_GHz"] > 10.0

        # Sub-THz 300-GHz LN Slow-Wave T-Rail physics + Bragg cutoff + Cherenkov contrast + 3D GDS DRC/LVS check
        trail_300_qtz = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="t_rail_cpw",
                core_material="LN_Xcut",
                substrate_type="quartz",
                metal_thickness_um=1.50,
                t_rail_period_um=25.0,
                active_length_um=4500.0,
                max_freq_ghz=360.0,
                run_gap_sweep=False,
            )
        )
        m300 = trail_300_qtz["metrics"]
        assert 49.0 <= m300["Z0_ohm"] <= 51.0
        assert abs(m300["n_RF"] - m300["n_g_opt"]) < 0.015
        assert m300["alpha_RF_40GHz_dB_per_cm"] < 2.0
        assert m300["f_3dBe_GHz"] >= 300.0 and m300["S21_EO_300GHz_dBe"] >= -3.0
        assert m300["alpha_cherenkov_300GHz_dB_per_cm"] == 0.0
        assert len(trail_300_qtz["field_map"]["stem_boxes_um"]) == 4

        trail_300_coarse = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="t_rail_cpw",
                core_material="LN_Xcut",
                substrate_type="quartz",
                metal_thickness_um=1.50,
                t_rail_period_um=220.0,
                active_length_um=4500.0,
                max_freq_ghz=360.0,
                run_gap_sweep=False,
            )
        )
        assert trail_300_coarse["metrics"]["f_bragg_cutoff_GHz"] < 340.0
        assert trail_300_coarse["metrics"]["f_3dBe_GHz"] < 300.0
        assert trail_300_coarse["metrics"]["limiting_mechanism"] == "bragg_periodic_cutoff"

        trail_300_hrsi = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="t_rail_cpw",
                core_material="LN_Xcut",
                substrate_type="hr_si",
                metal_thickness_um=1.50,
                t_rail_period_um=25.0,
                active_length_um=4500.0,
                max_freq_ghz=360.0,
                run_gap_sweep=False,
            )
        )
        assert trail_300_hrsi["metrics"]["alpha_cherenkov_300GHz_dB_per_cm"] > 2.0
        assert trail_300_hrsi["metrics"]["f_3dBe_GHz"] < m300["f_3dBe_GHz"]

        import gdstk

        gds_300 = export_eom_electrode_gds(trail_300_qtz, tmp_path / "trail_300ghz_drc.gds")
        lib_300 = gdstk.read_gds(gds_300["gds_path"])
        top_300 = lib_300.top_level()[0]
        from shapely.geometry import Point as _ShpPoint, Polygon as _ShpPoly
        from shapely.ops import unary_union as _shp_union

        def _lyr_geoms(ln: int):
            ps = [_ShpPoly(p.points) for p in top_300.get_polygons() if p.layer == ln and len(p.points) >= 3]
            if not ps:
                return []
            u = _shp_union(ps)
            return list(u.geoms) if hasattr(u, "geoms") else [u]

        core_u = _shp_union(_lyr_geoms(1))
        m1_list = _lyr_geoms(11)
        via_list = _lyr_geoms(12)
        m2_list = _lyr_geoms(13)
        m1_u = _shp_union(m1_list)
        assert m1_u.intersection(core_u).area == 0.0  # Zero Metal-1 overlap over optical waveguide cores!
        nodes_3d = [("M1", g) for g in m1_list] + [("VIA", g) for g in via_list] + [("M2", g) for g in m2_list]
        adj_3d = {i: set() for i in range(len(nodes_3d))}
        for i_n in range(len(nodes_3d)):
            for j_n in range(i_n + 1, len(nodes_3d)):
                ti, gi = nodes_3d[i_n]
                tj, gj = nodes_3d[j_n]
                if (ti == tj or "VIA" in (ti, tj)) and gi.intersects(gj):
                    adj_3d[i_n].add(j_n)
                    adj_3d[j_n].add(i_n)
        xm_chk = 0.5 * (m1_u.bounds[0] + m1_u.bounds[2])
        sig_idx = next(i for i, (t, g) in enumerate(nodes_3d) if t == "M1" and g.contains(_ShpPoint(xm_chk, 0.0)))
        d300 = trail_300_qtz["dimensions"]
        yg_chk = 0.5 * d300["signal_width_um"] + d300["trunk_gap_um"] + 20.0
        gnd_hi_idx = next(i for i, (t, g) in enumerate(nodes_3d) if t == "M1" and g.contains(_ShpPoint(xm_chk, +yg_chk)))
        gnd_lo_idx = next(i for i, (t, g) in enumerate(nodes_3d) if t == "M1" and g.contains(_ShpPoint(xm_chk, -yg_chk)))
        vis_3d = {sig_idx}
        stk_3d = [sig_idx]
        while stk_3d:
            cur_n = stk_3d.pop()
            for nb_n in adj_3d[cur_n]:
                if nb_n not in vis_3d:
                    vis_3d.add(nb_n)
                    stk_3d.append(nb_n)
        assert gnd_hi_idx not in vis_3d and gnd_lo_idx not in vis_3d  # Zero 3D Signal-to-Ground electrical short!

        c_300 = process_chat_turn(
            create_initial_session(lang="en"),
            "Design a 300 GHz LN T-shape electrode modulator",
            gds_export_dir=tmp_path,
        )
        assert c_300["ui_view"] == "eom" and c_300["eom_result"]["resolved_topology"] == "t_rail_cpw"
        assert c_300["eom_result"]["config"]["substrate_type"] == "quartz"
        assert c_300["eom_result"]["metrics"]["f_3dBe_GHz"] >= 300.0

        async def _check_eom_http() -> None:
            old_exp = srv.GDS_EXPORT_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "srv_eom"
            try:
                st0, b0 = await _asgi_request("GET", "/")
                assert st0 == 200 and b"pill-eom" in b0 and b"view-eom" in b0

                st1, b1 = await _asgi_request(
                    "POST",
                    "/api/eom_multiphysics",
                    json_body={"mode": "rf_electrode", "electrode_topology": "gsg_cpw", "core_material": "LN_Xcut", "run_gap_sweep": False},
                )
                d1 = json.loads(b1)
                assert st1 == 200 and d1["mode"] == "rf_electrode" and 46.0 <= d1["metrics"]["Z0_ohm"] <= 55.0

                st2, b2 = await _asgi_request("POST", "/api/export_eom_multiphysics_gds", json_body={"mode": "rf_electrode", "sim_result": d1})
                assert st2 == 200 and "gds_download_url" in json.loads(b2)

                st3, b3 = await _asgi_request("POST", "/api/export_eom_multiphysics_bundle", json_body={"mode": "rf_electrode", "sim_result": d1})
                assert st3 == 200 and "zip_download_url" in json.loads(b3)

                st4, b4 = await _asgi_request(
                    "POST",
                    "/api/eom_multiphysics",
                    json_body={"mode": "pn_doping", "junction_type": "lateral_pn", "bias_voltages_v": "0, -2, -4"},
                )
                assert st4 == 200 and json.loads(b4)["mode"] == "pn_doping_tcad"

                st5, b5 = await _asgi_request(
                    "POST",
                    "/api/eom_multiphysics",
                    json_body={"mode": "thermal_fem", "thermal_structure": "deep_trench", "core_material": "Si"},
                )
                assert st5 == 200 and json.loads(b5)["mode"] == "thermal_fem"

                st6, b6 = await _asgi_request(
                    "POST",
                    "/api/eom_multiphysics",
                    json_body={"mode": "wafer_orientation", "core_material": "3C-SiC", "wafer_cut": "cubic_001_43m", "polarization": "TE0"},
                )
                d6 = json.loads(b6)
                assert st6 == 200 and d6["mode"] == "wafer_orientation" and d6["metrics"]["optimal_azimuth_deg"] in (45.0, 135.0)
            finally:
                srv.GDS_EXPORT_DIR = old_exp

        asyncio.run(_check_eom_http())

    a_rf = agent_eom_multiphysics({"mode": "rf_electrode", "topology": "gsg_cpw", "mat": "LN_Xcut", "export_gds": True})
    assert 46.0 <= a_rf["Z0_ohm"] <= 55.0 and Path(a_rf["gds"]).is_file()

    a_wo = agent_eom_multiphysics({"mode": "wafer_orientation", "mat": "LN_Xcut", "wafer_cut": "x_cut_ln"})
    assert a_wo["opt_azimuth_deg"] == 0.0 and a_wo["ring_unsplit_pct"] < 1.0

    mcp_eom = handle_mcp_request({
        "jsonrpc": "2.0",
        "id": 115,
        "method": "tools/call",
        "params": {
            "name": "microring_eom_multiphysics",
            "arguments": {"mode": "thermal_fem", "structure": "undercut_suspended", "mat": "Si", "export_gds": True},
        },
    })
    assert mcp_eom is not None and not mcp_eom["result"]["isError"]
    eom_txt = mcp_eom["result"]["content"][0]["text"]
    assert len(eom_txt) < 620, f"MCP eom_multiphysics too verbose ({len(eom_txt)} chars): {eom_txt}"


def test_foundry_pdk_drc_lvs_and_cutline_solver() -> None:
    assert len(FOUNDRY_PDK_CATALOG) == 6
    st = agent_status()
    assert "pdks" in st and len(st["pdks"]) == 6

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # 1. NDA-safe CSV PDK export & reload round-trip
        csv_dir = tmp_path / "pdk_csv"
        exp_csv = export_pdk_csv_tables("gsip_soi220", csv_dir)
        assert Path(exp_csv["files"]["process_layers"]).is_file() and Path(exp_csv["files"]["drc_rules"]).is_file()
        pdk_reloaded = load_pdk_from_csv_dir(csv_dir, base_pdk_key="gsip_soi220")
        assert "Custom CSV PDK" in pdk_reloaded["name_en"]
        assert pdk_reloaded["layers"]["core"][:2] == get_foundry_pdk("gsip_soi220")["layers"]["core"][:2]

        # 2. Build Co-Integrated Dual-Drive PN + TiN Heater Ring Modulator (`GSG` North + `H1/H2` South)
        ring_gds = tmp_path / "pn_heater_gsip.gds"
        bld = build_pn_heater_ring_modulator_gds(
            ring_gds,
            pdk="gsip_soi220",
            ring_radius_um=12.0,
            gap_um=0.20,
            apply_slab_fillets=True,
            verify_drc_lvs=True,
        )
        assert ring_gds.is_file() and Path(bld["lyp_path"]).is_file()
        drc = bld["drc"]
        lvs = bld["lvs"]
        assert drc["drc_clean"] is True, f"Unexpected DRC violations: {drc['violations']}"
        assert Path(drc["lyrdb_path"]).is_file() and "<report-database>" in Path(drc["lyrdb_path"]).read_text()
        assert lvs["lvs_clean"] is True, f"Unexpected LVS violations: {lvs}"
        assert lvs["floating_via_count"] == 0
        assert len(lvs["short_violations"]) == 0
        assert lvs["total_nets"] >= 3

        # 3. Deliberate DRC & 3D LVS Failure Injection (must detect narrow core AND S-to-G1 metal short!)
        import gdstk

        bad_lib = gdstk.read_gds(str(ring_gds))
        bad_top = bad_lib.top_level()[0]
        core_ld = get_foundry_pdk("gsip_soi220")["layers"]["core"]
        m2_ld = get_foundry_pdk("gsip_soi220")["layers"]["metal2"]
        # Inject illegal 70 nm narrow core sliver (< 140 nm min width)
        bad_top.add(gdstk.rectangle((-25.0, -5.0), (-24.93, 5.0), layer=core_ld[0], datatype=core_ld[1]))
        # Inject Metal-2 bridge shorting G1 (-100, 74) to S (0, 74)
        bad_top.add(gdstk.rectangle((-105.0, 68.0), (5.0, 80.0), layer=m2_ld[0], datatype=m2_ld[1]))
        bad_gds = tmp_path / "pn_heater_broken.gds"
        bad_lib.write_gds(str(bad_gds))

        bad_drc = run_gds_polygon_drc(bad_gds, pdk="gsip_soi220")
        bad_lvs = extract_3d_electrical_lvs(bad_gds, pdk="gsip_soi220")
        assert bad_drc["drc_clean"] is False and bad_drc["total_violations"] >= 1
        assert bad_lvs["lvs_clean"] is False and len(bad_lvs["short_violations"]) >= 1

        # 4. Native Own-Solver GDS Cutline Layer Builder (10-pt index probe + Maxwell + PB PN + Thermal FEM)
        cut = solve_gds_cutline_multiphysics(ring_gds, pdk="gsip_soi220", wavelength_nm=1550.0)
        assert len(cut["index_probe_10pt"]) == 10
        assert 2.10 <= cut["eigenmode_2d_fem"]["n_eff"] <= 2.85
        assert "pn_tcad_2d_fem" in cut and 0.4 <= cut["pn_tcad_2d_fem"]["V_pi_L_Vcm"] <= 4.5
        assert "thermal_2d_fem" in cut and cut["thermal_2d_fem"]["pn_contact_heat_stealing_pct"] > 5.0

        # 5. Export MRR with PDK + Acute Rib-Slab Wedge Fillets + Thermal Galerkin Flux Conservation
        pdk_mrr = export_mrr_gds_with_pdk(
            MRRDesignConfig(topology="racetrack_ad", core_material="Si", film_height_um=0.22, slab_height_um=0.09, ring_radius_um=14.0, gap_thru_um=0.20, gap_drop_um=0.20),
            tmp_path / "mrr_cornerstone.gds",
            pdk="cornerstone_soi220",
            conformal_rib_slab=True,
            run_drc=True,
        )
        assert pdk_mrr["drc"]["drc_clean"] is True

        th_nosink = simulate_thermal_phase_shifter(ThermalSimConfig(core_material="Si", slab_height_um=0.09, include_pn_contact_sinking=False))
        th_sink = simulate_thermal_phase_shifter(ThermalSimConfig(core_material="Si", slab_height_um=0.09, include_pn_contact_sinking=True))
        assert th_sink["metrics"]["P_pi_mW"] > 1.06 * th_nosink["metrics"]["P_pi_mW"]
        assert th_sink["metrics"]["pn_contact_heat_stealing_pct"] > 6.0
        assert abs(th_sink["metrics"]["flux_balance_ratio"] - 1.0) < 0.03

        coint = simulate_cointegrated_pn_heater_ring(ring_radius_um=12.0, gap_um=0.18, drive_vpp=2.4)
        assert coint["metrics"]["f_3dB_EO_GHz"] > 14.0 and coint["metrics"]["OMA_normalized"] > 0.08, f"Got {coint['metrics']}"


def test_prism_closure_audit_and_real_fem_optimizer() -> None:
    assert set(PRISM_PROBLEM_CATALOG.keys()) == {"pn_ring_modulator", "high_er_pulley_mrr", "low_vpil_eom"}

    # 1. Physics Closure Audit (Q-budget + Unitary Passivity + Core/Slab Perturbation Split)
    aud = audit_mrr_and_pn_physics_closure(
        MRRDesignConfig(topology="point_circular_ad", core_material="Si", film_height_um=0.22, slab_height_um=0.09, ring_radius_um=12.0, gap_thru_um=0.22, gap_drop_um=0.24),
        pn_cfg=PNDopingTCADConfig(bias_voltages_v=[0.0, -2.0]),
    )
    assert aud["closure_pass"] is True, f"Physics closure audit failed: {aud}"
    assert aud["q_budget"]["q_closure_error_pct"] < 6.0
    assert aud["unitary_passivity"]["max_thru_plus_drop_power"] <= 1.0005
    assert aud["perturbation_closure"] is not None
    assert aud["perturbation_closure"]["perturbation_closure_error_pct"] < 6.0

    # 2. Closed-Loop Prediction-Gated Real-2D-FEM Optimizer + Auditable journal.jsonl
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        opt_pn = run_prism_closed_loop_optimizer(
            problem_type="pn_ring_modulator",
            max_iterations=6,
            seed=42,
            output_dir=tmp_path,
        )
        assert opt_pn["iterations_run"] == 6
        assert opt_pn["best_design"]["display_fom"] >= opt_pn["initial_design"]["display_fom"]
        j_path = Path(opt_pn["journal_jsonl_path"])
        assert j_path.is_file()
        lines = [json.loads(ln) for ln in j_path.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 6 and "predicted_sigma" in lines[1]

        opt_pulley = run_prism_closed_loop_optimizer(
            problem_type="high_er_pulley_mrr",
            max_iterations=6,
            seed=7,
            output_dir=tmp_path,
        )
        assert opt_pulley["best_design"]["display_fom"] > opt_pulley["initial_design"]["display_fom"]


def test_wdm_link_eye_thermal_locking_and_foundry_corners() -> None:
    assert len(WDM_SCENARIO_CATALOG) == 5

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # 1. WDM Link: Nominal open eye vs +8 K Thermal Drift collapse vs PI TiN Heater Relocked recovery
        w_nom = simulate_wdm_microring_link(WDMLinkConfig(scenario="nominal", num_channels=4, data_rate_per_ch_gbps=25.0), export_csv_dir=tmp_path)
        w_drift = simulate_wdm_microring_link(WDMLinkConfig(scenario="thermal_drift", num_channels=4, data_rate_per_ch_gbps=25.0))
        w_relock = simulate_wdm_microring_link(WDMLinkConfig(scenario="heater_relocked", num_channels=4, data_rate_per_ch_gbps=25.0))
        w_pam4 = simulate_wdm_microring_link(WDMLinkConfig(scenario="nominal", modulation_format="pam4", num_channels=4, data_rate_per_ch_gbps=50.0))

        assert Path(w_nom["eye_csv_path"]).is_file()
        assert w_nom["metrics"]["Q_factor"] > 5.0 and w_nom["metrics"]["link_open_pass"] is True
        assert w_drift["metrics"]["Q_factor"] < 0.65 * w_nom["metrics"]["Q_factor"]
        assert w_relock["metrics"]["Q_factor"] > 1.35 * w_drift["metrics"]["Q_factor"]
        assert w_relock["metrics"]["heater_lock_power_per_ch_mW"] > 0.5
        assert abs(w_relock["metrics"]["residual_detuning_error_pm"]) < abs(w_drift["metrics"]["residual_detuning_error_pm"]) * 0.15
        assert w_pam4["metrics"]["aggregate_bandwidth_Gbps"] == 200.0 and w_pam4["metrics"]["baud_rate_GBaud"] == 25.0

        # 2. Foundry Process Corners (Nominal / SS / FF / SF / FS) + Wafer Monte-Carlo Yield
        crn = run_foundry_process_corners(
            MRRDesignConfig(topology="point_circular_ap", core_material="Si", film_height_um=0.22, slab_height_um=0.09, ring_radius_um=12.0, gap_thru_um=0.21),
            pdk_key="gsip_soi220",
            mc_trials=120,
        )
        assert [c["corner"] for c in crn["corners"]] == ["Nominal", "SS", "FF", "SF", "FS"]
        assert crn["monte_carlo_summary"]["lambda_res_1sigma_pm"] > 10.0
        assert 0.0 <= crn["monte_carlo_summary"]["yield_ER_gt_10dB_pct"] <= 100.0

        # 3. Copilot NL Triggers + ASGI HTTP Endpoints + Token-Minimal MCP Tools (< 620 chars)
        c_pdk = process_chat_turn(create_initial_session(lang="en"), "Build PN + TiN heater ring with DRC and 3D LVS in GSIP", gds_export_dir=tmp_path)
        assert c_pdk["ui_view"] == "foundry" and c_pdk["foundry_pdk_result"]["drc"]["drc_clean"] is True

        c_wdm = process_chat_turn(create_initial_session(lang="zh"), "仿真 4通道 25Gbps 微环 WDM 眼图 热重锁定", gds_export_dir=tmp_path, lang="zh")
        assert c_wdm["ui_view"] == "foundry" and c_wdm["wdm_result"]["config"]["scenario"] == "heater_relocked"

        async def _check_foundry_http() -> None:
            old_exp = srv.GDS_EXPORT_DIR
            srv.GDS_EXPORT_DIR = tmp_path / "srv_fnd"
            try:
                st0, b0 = await _asgi_request("GET", "/")
                assert st0 == 200 and b"pill-foundry" in b0 and b"view-foundry" in b0

                st1, b1 = await _asgi_request("POST", "/api/pdk_drc_lvs", json_body={"pdk_key": "gsip_soi220", "ring_radius_um": 12.0, "run_cutline_fem": False})
                d1 = json.loads(b1)
                assert st1 == 200 and d1["drc"]["drc_clean"] is True and "lyrdb_download_url" in d1

                st2, b2 = await _asgi_request("POST", "/api/prism_optimize", json_body={"mode": "closure_audit"})
                assert st2 == 200 and json.loads(b2)["closure_pass"] is True

                st3, b3 = await _asgi_request("POST", "/api/wdm_link", json_body={"scenario": "nominal", "num_channels": 4})
                d3 = json.loads(b3)
                assert st3 == 200 and d3["metrics"]["Q_factor"] > 5.0 and "eye_csv_download_url" in d3

                st4, b4 = await _asgi_request("POST", "/api/process_corners", json_body={"pdk_key": "cornerstone_soi220", "mc_trials": 64})
                assert st4 == 200 and len(json.loads(b4)["corners"]) == 5
            finally:
                srv.GDS_EXPORT_DIR = old_exp

        asyncio.run(_check_foundry_http())

        for tool_name, args in [
            ("microring_pdk_drc", {"mode": "build_pn_heater_ring", "pdk": "gsip_soi220", "run_cutline": True}),
            ("microring_prism_optimize", {"mode": "closure_audit"}),
            ("microring_wdm_link", {"scenario": "heater_relocked", "n_ch": 4, "rate_gbps": 25.0}),
        ]:
            mcp_r = handle_mcp_request({"jsonrpc": "2.0", "id": 201, "method": "tools/call", "params": {"name": tool_name, "arguments": args}})
            assert mcp_r is not None and not mcp_r["result"]["isError"]
            txt = mcp_r["result"]["content"][0]["text"]
            assert len(txt) < 620, f"MCP {tool_name} too verbose ({len(txt)} chars): {txt}"

        f_crn = agent_fab_tolerance({"mode": "foundry_corners", "pdk": "gsip_soi220", "mc_trials": 64})
        assert len(json.dumps(f_crn)) < 620 and "yield_ER15_pct" in f_crn


def main() -> None:
    tests = [
        ("Materials & Hardware Probe", test_materials_and_hardware),
        ("Pure-BSD Conformal FEM (Zero GPL)", test_pure_bsd_fem_no_gpl),
        ("Waveguide Width & MRR Parameter Sweeps", test_width_and_mrr_sweeps),
        ("Bidirectional GDSII Round-Trip (4 Topologies + Array)", test_bidirectional_gds_roundtrip),
        ("Conversational Photonics Copilot (EN + ZH)", test_conversational_copilot_flow),
        ("Lab Spectrum CMT Split-Doublet Fitter", test_cmt_spectrum_fitter),
        ("HTTP ASGI Endpoints & GDS Upload/Download", test_http_endpoints_and_gds_upload),
        ("Token-Minimal Codex/Claude Agent CLI & MCP Server", test_agent_tools_and_mcp_minimal_tokens),
        ("Multi-Waveband Physics (Visible 532nm / O-Band 1310nm / Mid-IR 4.2um)", test_multiband_vis_1310_midir_and_warnings),
        ("Cleanroom Lithography Positive/Negative PR Dose & OPC Bias GDS", test_fabrication_dose_and_pr_tolerance),
        ("2D-FEM TO/EO/Si-PN Active Tuning, T/E/V/I Spectra & Lumerical INTERCONNECT", test_active_tuning_2d_fem_and_interconnect),
        ("Photonic IC Butt/2D-FDFD Grating Couplers & High-Q FDTD Convergence", test_io_couplers_and_high_q_fdtd_convergence),
        ("2D-FEM/EME Mach-Zehnder Interferometers (Y-Branch/DC/MMI), Modulators & INTERCONNECT", test_mzi_splitters_tuning_gds_and_interconnect),
        ("2D-FEM EOM RF Electrodes, Si PN Doping TCAD & Steady/Transient Thermal FEM", test_eom_electrode_pn_tcad_and_thermal),
        ("Foundry PDK Profiles, Polygon DRC (.lyrdb), 3D LVS & Native Cutline 2D-FEM", test_foundry_pdk_drc_lvs_and_cutline_solver),
        ("PRISM Physics-Closure Audit & Prediction-Gated Real-2D-FEM Optimizer", test_prism_closure_audit_and_real_fem_optimizer),
        ("Multi-Channel WDM Ring Link, Eye Diagram, PI Heater Locking & Process Corners", test_wdm_link_eye_thermal_locking_and_foundry_corners),
    ]
    if len(sys.argv) > 1:
        toks = [t.strip().lower() for t in sys.argv[1].split(",") if t.strip()]
        tests = [(lbl, fn) for lbl, fn in tests if any(tk in lbl.lower() or tk in fn.__name__.lower() for tk in toks)]
    print("=" * 72)
    print("Running MicroRing Studio v3.9.0 Full Verification Suite")
    print("=" * 72)
    for label, fn in tests:
        print(f"[RUN ] {label} ...", flush=True)
        fn()
        print(f"[PASS] {label}")
    print("=" * 72)
    print(f"ALL {len(tests)}/{len(tests)} TEST SUITES PASSED!")


if __name__ == "__main__":
    main()
