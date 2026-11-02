"""Comprehensive Automated Test Suite for MicroRing Studio v3.6.0.

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
    FIBER_LASER_CATALOG,
    MATERIAL_CATALOG,
    MRRDesignConfig,
    MZI_SPLITTER_CATALOG,
    MZIDesignConfig,
    RESIST_TONE_CATALOG,
    TUNING_MECHANISM_CATALOG,
    T_REF_K,
    WaveguideCrossSectionSpec,
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
    analyze_high_q_fdtd_convergence,
    analyze_uploaded_gds,
    create_initial_session,
    design_butt_edge_coupler,
    design_grating_coupler,
    detect_hardware_profile,
    export_coupler_cutout_gds,
    export_fab_bias_array_gds,
    export_io_coupler_gds,
    export_io_coupler_interconnect,
    export_lumerical_interconnect_bundle,
    export_mrr_gds,
    export_mzi_gds,
    export_mzi_interconnect_bundle,
    export_sweep_array_gds,
    fit_single_resonance_cmt,
    get_material_fn,
    loss_db_per_cm_to_q_int,
    process_chat_turn,
    q_int_to_loss_db_per_cm,
    run_active_tuning_analysis,
    run_compute_benchmark,
    run_fabrication_error_analysis,
    run_mrr_parameter_sweep,
    run_waveguide_width_sweep,
    simulate_mrr_topology,
    simulate_mzi,
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
    assert len(MATERIAL_CATALOG) == 24
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
                assert st == 200 and len(json.loads(body)["materials"]) == 24

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
    assert list_resp is not None and len(list_resp["result"]["tools"]) == 11

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
    ]
    print("=" * 72)
    print("Running MicroRing Studio v3.6.0 Full Verification Suite")
    print("=" * 72)
    for label, fn in tests:
        print(f"[RUN ] {label} ...", flush=True)
        fn()
        print(f"[PASS] {label}")
    print("=" * 72)
    print(f"ALL {len(tests)}/{len(tests)} TEST SUITES PASSED!")


if __name__ == "__main__":
    main()
