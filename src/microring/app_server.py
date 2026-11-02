"""Local Starlette ASGI Backend for `MicroRing Studio v3.0` (`127.0.0.1` Only).

Serves the offline native desktop / browser UI and exposes clean JSON endpoints for:
  - Conversational Photonics Design Copilot (`/api/chat_init`, `/api/chat`)
  - Bidirectional GDSII / OASIS upload, polygon geometry extraction & sweep-array export
    (`/api/upload_gds`, `/api/export_gds`, `/api/export_sweep_gds`, `/api/download_file`)
  - Automated waveguide single-mode width sweeps & MRR parameter sweeps
    (`/api/sweep_width`, `/api/sweep_mrr`)
  - 2D full-vector Maxwell eigenmode solver (`/api/solve_mode`)
  - 7-topology MRR synthesis + critical coupling + EO/TO tuning (`/api/design_mrr`)
  - Experimental lab spectrum CMT fitting (`/api/demo_spectrum`, `/api/fit_spectrum`)
  - 18-material dispersion library & GPU/AMX/CPU benchmark (`/api/materials`, `/api/hardware`)
"""
from __future__ import annotations

import base64
from dataclasses import fields
import math
import os
from pathlib import Path
import re
import time

import anyio
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .active_tuning import (
    TUNING_MECHANISM_CATALOG,
    export_lumerical_interconnect_bundle,
    run_active_tuning_analysis,
)
from .copilot import (
    create_initial_session,
    get_welcome_chips,
    inject_uploaded_gds_into_session,
    probe_ai_backends,
    process_chat_turn,
)
from .eigenmode_solver import (
    WaveguideCrossSectionSpec,
    solve_modes_fde_tidy3d,
    solve_modes_fem,
)
from .fab_tolerance import (
    RESIST_TONE_CATALOG,
    export_fab_bias_array_gds,
    run_fabrication_error_analysis,
)
from .gds_bridge import (
    analyze_uploaded_gds,
    export_sweep_array_gds,
    write_klayout_layer_props,
)
from .hardware_accel import detect_hardware_profile, run_compute_benchmark
from .io_couplers import (
    FIBER_LASER_CATALOG,
    design_butt_edge_coupler,
    design_grating_coupler,
    export_io_coupler_gds,
    export_io_coupler_interconnect,
)
from .materials import (
    MATERIAL_CATALOG,
    T_REF_K,
    WAVEBAND_PRESETS,
    evaluate_material_dispersion_table,
    get_material_fn,
)
from .mrr_cmt_fit_predict import (
    cmt_all_pass_complex_amplitude,
    cmt_fano_background_power,
    cmt_split_doublet_power,
    fit_multi_resonance_spectrum,
    fit_single_resonance_cmt,
    tmm_broadband_mrr_spectrum,
)
from .mrr_topologies import (
    MRRDesignConfig,
    TOPOLOGY_CATALOG,
    analyze_high_q_fdtd_convergence,
    export_coupler_cutout_gds,
    export_mrr_gds,
    get_default_output_root,
    simulate_mrr_topology,
)
from .mzi_interferometer import (
    MZI_SPLITTER_CATALOG,
    MZIDesignConfig,
    export_mzi_gds,
    export_mzi_interconnect_bundle,
    simulate_mzi,
)
from .sweeps import run_mrr_parameter_sweep, run_waveguide_width_sweep
from .ui_frontend import HTML_APP_PAGE

DEFAULT_OUT_ROOT = get_default_output_root()
GDS_EXPORT_DIR = DEFAULT_OUT_ROOT / "gds_exports"
GDS_UPLOAD_DIR = DEFAULT_OUT_ROOT / "gds_uploads"


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return cleaned or "layout.gds"


async def homepage(_request: Request) -> HTMLResponse:
    return HTMLResponse(HTML_APP_PAGE)


async def api_hardware(_request: Request) -> JSONResponse:
    prof = detect_hardware_profile()
    return JSONResponse(prof.to_dict())


async def api_benchmark(_request: Request) -> JSONResponse:
    res = await anyio.to_thread.run_sync(run_compute_benchmark, 100_000)
    return JSONResponse(res)


async def api_materials(_request: Request) -> JSONResponse:
    items = []
    for key, meta in MATERIAL_CATALOG.items():
        ref_wl_m = float(meta["default_band_nm"]) * 1e-9
        props = meta["fn"](ref_wl_m, T_REF_K).to_dict()
        items.append({
            "key": key,
            "name": meta["name"],
            "category": meta["category"],
            "default_band_nm": meta["default_band_nm"],
            "typical_loss_db_cm": meta["typical_loss_db_cm"],
            "ref_props": props,
        })
    return JSONResponse({"materials": items, "topologies": TOPOLOGY_CATALOG, "wavebands": WAVEBAND_PRESETS})


async def api_material_dispersion(request: Request) -> JSONResponse:
    key = request.query_params.get("key", "4H-SiC")
    wmin = float(request.query_params.get("wmin", "380"))
    wmax = float(request.query_params.get("wmax", "10000"))
    try:
        table = evaluate_material_dispersion_table(key, wavelength_min_nm=wmin, wavelength_max_nm=wmax)
        return JSONResponse(table)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def api_chat_init(request: Request) -> JSONResponse:
    lang = request.query_params.get("lang", "en")
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("lang"):
            lang = str(body["lang"])
    except Exception:
        pass
    sess = create_initial_session(lang=lang)
    return JSONResponse({
        "session": sess,
        "assistant_message": sess["history"][-1]["content"],
        "chips": get_welcome_chips(lang=lang),
    })


async def api_ai_backends(_request: Request) -> JSONResponse:
    res = await anyio.to_thread.run_sync(probe_ai_backends)
    return JSONResponse(res)


def _chat_turn_sync(payload: dict) -> dict:
    session = payload.get("session")
    message = str(payload.get("message", "")).strip()
    use_ollama = bool(payload.get("use_ollama", False))
    ai_mode = str(payload.get("ai_mode") or ("ollama" if use_ollama else "offline"))
    ai_api_key = str(payload.get("ai_api_key") or "").strip() or None
    lang = payload.get("lang")
    res = process_chat_turn(
        session,
        message,
        gds_export_dir=GDS_EXPORT_DIR,
        use_ollama=use_ollama,
        ai_mode=ai_mode,
        ai_api_key=ai_api_key,
        lang=str(lang) if lang else None,
    )
    if "gds_export" in res:
        gp = Path(res["gds_export"]["gds_path"])
        lp = Path(res["gds_export"]["lyp_path"])
        res["gds_export"]["gds_download_url"] = f"/api/download_file?name={gp.name}"
        res["gds_export"]["lyp_download_url"] = f"/api/download_file?name={lp.name}"
    if "interconnect_export" in res and res["interconnect_export"].get("zip_path"):
        zp = Path(res["interconnect_export"]["zip_path"])
        res["interconnect_export"]["zip_download_url"] = f"/api/download_file?name={zp.name}"
    return res


async def api_chat(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_chat_turn_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _upload_gds_sync(payload: dict) -> dict:
    fname = _safe_filename(str(payload.get("filename", "uploaded.gds")))
    b64 = str(payload.get("content_b64", ""))
    if "," in b64 and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64)
    GDS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = GDS_UPLOAD_DIR / fname
    saved_path.write_bytes(raw_bytes)

    core_layer = payload.get("core_layer")
    lyr_tuple = (int(core_layer[0]), int(core_layer[1])) if core_layer else None
    cell_name = payload.get("cell_name") or None

    gds_info = analyze_uploaded_gds(saved_path, core_layer=lyr_tuple, cell_name=cell_name)
    lang = str(payload.get("lang") or "en")
    session = payload.get("session") or create_initial_session(lang=lang)
    chat_sync = inject_uploaded_gds_into_session(session, gds_info, lang=lang)
    return {
        "gds_info": gds_info,
        "session": chat_sync["session"],
        "assistant_message": chat_sync["assistant_message"],
        "chips": chat_sync["chips"],
    }


async def api_upload_gds(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_upload_gds_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _solve_mode_sync(payload: dict) -> dict:
    t0 = time.perf_counter()
    core_mat = str(payload.get("core_material", "4H-SiC"))
    core_fn = get_material_fn(
        core_mat,
        sc_fraction=float(payload.get("sc_fraction", 0.20)) if core_mat.startswith("ScAlN") else None,
    )
    clad_fn = get_material_fn(str(payload.get("clad_material", "SiO2")))
    top_fn = get_material_fn(str(payload.get("top_clad_material", "SiO2")))

    wl_nm = float(payload.get("wavelength_nm", 1550.0))
    wl_m = wl_nm * 1e-9
    wl_um = wl_nm * 1e-3
    wl_scale = float(np.clip(wl_nm / 1550.0, 0.35, 4.50))
    w_bot = float(payload.get("w_bottom_um", 1.2))
    box_pad = max(1.8, 1.35 * wl_um)

    spec = WaveguideCrossSectionSpec(
        w_bottom_um=w_bot,
        h_film_um=float(payload.get("h_film_um", 0.4)),
        h_slab_um=float(payload.get("h_slab_um", 0.0)),
        sidewall_deg=float(payload.get("sidewall_deg", 85.0)),
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        domain_width_um=max(3.6 * wl_scale, w_bot + 2.4 * wl_um),
        h_box_um=box_pad,
        h_top_clad_um=box_pad,
    )

    r_um = float(payload.get("bend_radius_um", 0.0))
    bend_m = math.inf if r_um <= 0.0 else (r_um * 1e-6)
    n_modes = int(payload.get("num_modes", 4))
    raw_res = float(payload.get("core_mesh_res_um", 0.040))
    res_um = round(float(np.clip(raw_res * wl_scale, 0.016, 0.160)), 4)
    grid_step = round(float(np.clip(0.040 * wl_scale, 0.018, 0.120)), 4)
    backend = str(payload.get("solver_backend", "fem"))

    if backend == "fde":
        modes = solve_modes_fde_tidy3d(
            spec, wl_m, bend_radius_m=bend_m, num_modes=n_modes, core_dl_um=max(res_um * 0.65, 0.012)
        )
    else:
        modes = solve_modes_fem(
            spec,
            wl_m,
            bend_radius_m=bend_m,
            num_modes=n_modes,
            core_mesh_res_um=res_um,
            clad_mesh_res_um=max(res_um * 5.0, 0.15 * wl_scale),
            eval_grid_step_um=grid_step,
            mesh_backend="bsd_conformal",
        )

    outlines = []
    for name, poly in spec.build_polygons_um().items():
        if name in ("core", "slab", "upper_core", "cap"):
            coords = list(poly.exterior.coords)
            outlines.append([[round(float(x), 4), round(float(y), 4)] for x, y in coords])

    serialized_modes = []
    for m in modes:
        sy = max(1, len(m.y_m) // 110)
        sx = max(1, len(m.x_m) // 130)
        ex_sub = np.abs(m.Ex[::sy, ::sx])
        ey_sub = np.abs(m.Ey[::sy, ::sx])
        ez_sub = np.abs(m.Ez[::sy, ::sx])
        e_tot = np.sqrt(ex_sub ** 2 + ey_sub ** 2 + ez_sub ** 2)
        serialized_modes.append({
            "mode_index": m.mode_index,
            "solver_backend": m.solver_backend,
            "n_eff_real": float(np.real(m.n_eff)),
            "n_eff_imag": float(np.imag(m.n_eff)),
            "te_fraction": round(m.te_fraction, 4),
            "a_eff_um2": round(m.a_eff_um2, 4),
            "core_confinement": round(m.core_confinement, 4),
            "m_azimuthal": round(m.m_azimuthal, 3) if math.isfinite(m.m_azimuthal) else None,
            "x_um": np.round(m.x_m[::sx] * 1e6, 4).tolist(),
            "y_um": np.round(m.y_m[::sy] * 1e6, 4).tolist(),
            "E_abs": np.round(e_tot, 2).tolist(),
            "Ex_abs": np.round(ex_sub, 2).tolist(),
            "Ey_abs": np.round(ey_sub, 2).tolist(),
            "Ez_abs": np.round(ez_sub, 2).tolist(),
        })

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "polygon_outlines_um": outlines,
        "modes": serialized_modes,
    }


async def api_solve_mode(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_solve_mode_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _sweep_width_sync(payload: dict) -> dict:
    return run_waveguide_width_sweep(
        core_material=str(payload.get("core_material", "Si3N4")),
        clad_material=str(payload.get("clad_material", "SiO2")),
        top_clad_material=str(payload.get("top_clad_material", "SiO2")),
        h_film_um=float(payload.get("h_film_um", 0.40)),
        h_slab_um=float(payload.get("h_slab_um", 0.0)),
        sidewall_deg=float(payload.get("sidewall_deg", 85.0)),
        wavelength_nm=float(payload.get("wavelength_nm", 1550.0)),
        w_min_um=float(payload.get("w_min_um", 0.50)),
        w_max_um=float(payload.get("w_max_um", 2.00)),
        num_points=int(payload.get("num_points", 8)),
        polarization=str(payload.get("polarization", "TE0")),
    )


async def api_sweep_width(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_sweep_width_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _dict_to_mrr_config(payload: dict) -> MRRDesignConfig:
    valid_keys = {f.name for f in fields(MRRDesignConfig)}
    filtered = {k: v for k, v in payload.items() if k in valid_keys}
    return MRRDesignConfig(**filtered)


async def api_design_mrr(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        cfg = _dict_to_mrr_config(payload)
        res = await anyio.to_thread.run_sync(simulate_mrr_topology, cfg)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _sweep_mrr_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    param_name = str(payload.get("param_name", "gap_thru_um"))
    if "values" in payload and payload["values"]:
        vals = [float(v) for v in payload["values"]]
    else:
        vmin = float(payload.get("v_min", 0.15))
        vmax = float(payload.get("v_max", 0.35))
        npts = int(payload.get("num_points", 7))
        vals = np.linspace(vmin, vmax, max(3, npts)).tolist()
    return run_mrr_parameter_sweep(cfg, param_name, vals)


async def api_sweep_mrr(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_sweep_mrr_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_single_gds_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload)
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"MicroRing_{cfg.topology}_{cfg.core_material}_R{int(cfg.ring_radius_um)}um.gds"
    out_path = GDS_EXPORT_DIR / fname
    saved = export_mrr_gds(cfg, out_path)
    lyp_saved = write_klayout_layer_props(out_path.with_suffix(".lyp"))
    gds_view = analyze_uploaded_gds(saved)
    return {
        "saved_path": saved,
        "lyp_path": lyp_saved,
        "gds_download_url": f"/api/download_file?name={Path(saved).name}",
        "lyp_download_url": f"/api/download_file?name={Path(lyp_saved).name}",
        "gds_view": gds_view,
    }


async def api_export_gds(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_single_gds_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_sweep_gds_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    param_name = str(payload.get("param_name", TOPOLOGY_CATALOG[cfg.topology]["coupler_param"]))
    if "values" in payload and payload["values"]:
        vals = [float(v) for v in payload["values"]]
    else:
        vmin = float(payload.get("v_min", 0.16))
        vmax = float(payload.get("v_max", 0.32))
        npts = int(payload.get("num_devices", 5))
        vals = [round(float(x), 4) for x in np.linspace(vmin, vmax, max(2, npts))]
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"MicroRing_SweepArray_{cfg.core_material}_{param_name}_{len(vals)}x.gds"
    out_path = GDS_EXPORT_DIR / fname
    saved = export_sweep_array_gds(cfg, param_name, vals, out_path)
    lyp_saved = str(Path(saved).with_suffix(".lyp"))
    gds_view = analyze_uploaded_gds(saved)
    return {
        "saved_path": saved,
        "lyp_path": lyp_saved,
        "values": vals,
        "param_name": param_name,
        "gds_download_url": f"/api/download_file?name={Path(saved).name}",
        "lyp_download_url": f"/api/download_file?name={Path(lyp_saved).name}",
        "gds_view": gds_view,
    }


async def api_export_sweep_gds(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_sweep_gds_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _fab_tolerance_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    return run_fabrication_error_analysis(
        cfg,
        resist_tone=str(payload.get("resist_tone", "positive_pr")),
        max_dose_error_pct=float(payload.get("max_dose_error_pct", 15.0)),
        dose_slope_nm_per_pct=float(payload["dose_slope_nm_per_pct"]) if payload.get("dose_slope_nm_per_pct") not in (None, "") else None,
        max_cd_bias_nm=float(payload.get("max_cd_bias_nm", 30.0)),
        proximity_gap_factor=float(payload["proximity_gap_factor"]) if payload.get("proximity_gap_factor") not in (None, "") else None,
        delta_h_film_nm=float(payload.get("delta_h_film_nm", 0.0)),
        delta_sidewall_deg=float(payload.get("delta_sidewall_deg", 0.0)),
        num_points=int(payload.get("num_points", 7)),
    )


async def api_fab_tolerance(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_fab_tolerance_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_fab_gds_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    prox = float(payload.get("proximity_gap_factor") or 1.0)
    biases = payload.get("cd_prebias_list_nm")
    bias_list = [float(b) for b in biases] if biases else [-20.0, -10.0, 0.0, 10.0, 20.0]
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"MicroRing_OPC_DoseBracket_{cfg.core_material}_{cfg.topology}.gds"
    out_path = GDS_EXPORT_DIR / fname
    info = export_fab_bias_array_gds(
        cfg,
        out_path,
        cd_prebias_list_nm=bias_list,
        proximity_gap_factor=prox,
    )
    saved = info["gds_path"]
    lyp_saved = info["lyp_path"]
    gds_view = analyze_uploaded_gds(saved)
    return {
        "saved_path": saved,
        "lyp_path": lyp_saved,
        "cells": info["cells"],
        "gds_download_url": f"/api/download_file?name={Path(saved).name}",
        "lyp_download_url": f"/api/download_file?name={Path(lyp_saved).name}",
        "gds_view": gds_view,
    }


async def api_export_fab_gds(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_fab_gds_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _parse_bias_payload(raw_b) -> list[float] | None:
    if isinstance(raw_b, str) and raw_b.strip():
        return [float(x.strip()) for x in raw_b.split(",") if x.strip()]
    if isinstance(raw_b, (list, tuple)) and len(raw_b) > 0:
        return [float(x) for x in raw_b]
    return None


def _active_tuning_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    return run_active_tuning_analysis(
        cfg,
        mechanism=str(payload.get("mechanism", "auto")),
        drive_type=str(payload.get("drive_type", "auto")),
        bias_values=_parse_bias_payload(payload.get("bias_values")),
        active_fraction=float(payload.get("active_fraction", 0.78)),
        heater_material=str(payload.get("heater_material", "TiN")),
        heater_width_um=float(payload.get("heater_width_um", 1.60)),
        heater_thickness_um=float(payload.get("heater_thickness_um", 0.12)),
        heater_gap_um=float(payload.get("heater_gap_um", 0.90)),
        heater_offset_um=float(payload.get("heater_offset_um", 0.0)),
        heater_resistance_ohm=float(payload["heater_resistance_ohm"]) if payload.get("heater_resistance_ohm") not in (None, "") else None,
        box_thickness_um=float(payload.get("box_thickness_um", 2.00)),
        top_clad_thickness_um=float(payload.get("top_clad_thickness_um", 1.80)),
        electrode_config=str(payload.get("electrode_config", "auto")),
        electrode_gap_um=float(payload["electrode_gap_um"]) if payload.get("electrode_gap_um") not in (None, "") else None,
        electrode_width_um=float(payload.get("electrode_width_um", 4.00)),
        na_p_cm3=float(payload.get("na_p_cm3", 5.0e17)),
        nd_n_cm3=float(payload.get("nd_n_cm3", 5.0e17)),
        pn_offset_nm=float(payload.get("pn_offset_nm", 25.0)),
        heavy_contact_offset_um=float(payload.get("heavy_contact_offset_um", 0.85)),
        carrier_lifetime_ns=float(payload.get("carrier_lifetime_ns", 1.0)),
    )


async def api_active_tuning(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_active_tuning_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_interconnect_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    tune = payload.get("tuning_result")
    if not isinstance(tune, dict) or "states" not in tune:
        tune = _active_tuning_sync(payload)
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_mat = str(cfg.core_material).replace("-", "")
    mech = str(tune["efficiency"]["mechanism"])
    sub_dir = GDS_EXPORT_DIR / f"interconnect_{safe_mat}_{mech}"
    ic = export_lumerical_interconnect_bundle(cfg, tune, sub_dir, make_zip=True)
    zp = Path(ic["zip_path"]) if ic.get("zip_path") else None
    ic["zip_download_url"] = f"/api/download_file?name={zp.name}" if zp else None
    ic["mechanisms"] = TUNING_MECHANISM_CATALOG
    return ic


async def api_export_interconnect(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_interconnect_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _io_coupler_sync(payload: dict) -> dict:
    ctype = str(payload.get("coupler_type", "grating_coupler")).lower()
    mat = str(payload.get("core_material", "Si"))
    wl = float(payload.get("wavelength_nm", 1550.0))
    pol = str(payload.get("polarization", "TE0"))
    h_um = float(payload.get("film_height_um", 0.22))
    wb = float(payload.get("w_bus_um", 0.48))
    box_um = float(payload.get("box_thickness_um", 2.0))
    fib = str(payload.get("fiber_preset", "Lensed_2p5um" if "butt" in ctype else "SMF28_1550"))

    if "butt" in ctype or "edge" in ctype:
        wt_raw = payload.get("w_tip_um")
        wt_val = float(wt_raw) if wt_raw not in (None, "", "auto", 0, "0") else None
        res = design_butt_edge_coupler(
            core_material=mat,
            clad_material=str(payload.get("clad_material", "SiO2")),
            top_clad_material=str(payload.get("top_clad_material", "SiO2")),
            wavelength_nm=wl,
            polarization=pol,
            film_height_um=h_um,
            slab_height_um=float(payload.get("slab_height_um", 0.0)),
            sidewall_deg=float(payload.get("sidewall_deg", 88.0)),
            w_bus_um=wb,
            w_tip_um=wt_val,
            taper_length_um=float(payload.get("taper_length_um", 180.0)),
            taper_profile=str(payload.get("taper_profile", "adiabatic")),
            fiber_preset=fib,
            mfd_x_um=float(payload["mfd_x_um"]) if payload.get("mfd_x_um") not in (None, "") else None,
            mfd_y_um=float(payload["mfd_y_um"]) if payload.get("mfd_y_um") not in (None, "") else None,
            ar_coating=bool(payload.get("ar_coating", True)),
            box_thickness_um=box_um,
            min_feature_nm=float(payload.get("min_feature_nm", 100.0)),
        )
    else:
        etch_raw = payload.get("etch_depth_um")
        etch_val = float(etch_raw) if etch_raw not in (None, "", "auto", 0, "0") else None
        pitch_raw = payload.get("pitch_um")
        pitch_val = float(pitch_raw) if pitch_raw not in (None, "", "auto", 0, "0") else None
        res = design_grating_coupler(
            core_material=mat,
            clad_material=str(payload.get("clad_material", "SiO2")),
            top_clad_material=str(payload.get("top_clad_material", "SiO2")),
            wavelength_nm=wl,
            polarization=pol,
            film_height_um=h_um,
            etch_depth_um=etch_val,
            fiber_angle_deg=float(payload.get("fiber_angle_deg", 10.0)),
            apodized=bool(payload.get("apodized", True)),
            focusing=bool(payload.get("focusing", True)),
            pitch_um=pitch_val,
            fill_factor=float(payload.get("fill_factor", 0.50)),
            num_periods=int(payload.get("num_periods", 26)),
            grating_width_um=float(payload.get("grating_width_um", 12.0)),
            w_bus_um=wb,
            box_thickness_um=box_um,
            top_clad_thickness_um=float(payload.get("top_clad_thickness_um", 1.5)),
            fiber_preset=fib,
            min_feature_nm=float(payload.get("min_feature_nm", 80.0)),
        )
    res["fiber_catalog"] = FIBER_LASER_CATALOG
    return res


async def api_io_coupler(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_io_coupler_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_io_gds_sync(payload: dict) -> dict:
    coupler = payload.get("coupler_result")
    if not isinstance(coupler, dict) or "coupler_type" not in coupler:
        coupler = _io_coupler_sync(payload)
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    loopback = bool(payload.get("include_ring_loopback", False))
    ring_cfg = _dict_to_mrr_config(payload["ring_config"]) if isinstance(payload.get("ring_config"), dict) else None
    safe_mat = str(coupler["core_material"]).replace("-", "")
    loop_tag = "_RingTestLoop" if loopback else ""
    out_file = GDS_EXPORT_DIR / f"MicroRing_{coupler['coupler_type']}_{safe_mat}{loop_tag}.gds"
    info = export_io_coupler_gds(
        coupler,
        out_file,
        include_ring_loopback=loopback,
        ring_config=ring_cfg,
        fa_pitch_um=float(payload.get("fa_pitch_um", 127.0)),
    )
    saved = Path(info["gds_path"])
    lyp_saved = Path(info["lyp_path"])
    info["saved_path"] = str(saved)
    info["gds_download_url"] = f"/api/download_file?name={saved.name}"
    info["lyp_download_url"] = f"/api/download_file?name={lyp_saved.name}"
    info["gds_view"] = analyze_uploaded_gds(str(saved))
    return info


async def api_export_io_gds(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_io_gds_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_io_interconnect_sync(payload: dict) -> dict:
    coupler = payload.get("coupler_result")
    if not isinstance(coupler, dict) or "coupler_type" not in coupler:
        coupler = _io_coupler_sync(payload)
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_mat = str(coupler["core_material"]).replace("-", "")
    sub_dir = GDS_EXPORT_DIR / f"io_interconnect_{coupler['coupler_type']}_{safe_mat}"
    ic = export_io_coupler_interconnect(coupler, sub_dir)
    # Copy zip to GDS_EXPORT_DIR root if not already there
    zp = Path(ic["zip_path"])
    ic["zip_download_url"] = f"/api/download_file?name={zp.name}"
    return ic


async def api_export_io_interconnect(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_io_interconnect_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _high_q_fdtd_sync(payload: dict) -> dict:
    cfg = _dict_to_mrr_config(payload.get("config", payload))
    sim_times = _parse_bias_payload(payload.get("sim_times_ps"))
    hq = analyze_high_q_fdtd_convergence(cfg, sim_times_ps=sim_times)
    if payload.get("export_cutout"):
        GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        cut = export_coupler_cutout_gds(cfg, GDS_EXPORT_DIR)
        hq["cutout"] = {
            "saved_path": cut["gds_path"],
            "gds_download_url": f"/api/download_file?name={Path(cut['gds_path']).name}",
            "lsf_download_url": f"/api/download_file?name={Path(cut['lsf_path']).name}",
            "gds_view": analyze_uploaded_gds(str(cut["gds_path"])),
            **cut,
        }
    return hq


async def api_high_q_fdtd(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_high_q_fdtd_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _dict_to_mzi_config(payload: dict) -> MZIDesignConfig:
    valid_keys = {f.name for f in fields(MZIDesignConfig)}
    filtered = {}
    for k, v in payload.items():
        if k not in valid_keys:
            continue
        if v in ("", "auto"):
            continue
        filtered[k] = v
    return MZIDesignConfig(**filtered)


def _design_mzi_sync(payload: dict) -> dict:
    cfg = _dict_to_mzi_config(payload.get("config", payload))
    biases = _parse_bias_payload(payload.get("bias_values"))
    egap_raw = payload.get("electrode_gap_um")
    egap = float(egap_raw) if egap_raw not in (None, "", "auto", 0, "0") else None
    res = simulate_mzi(
        cfg,
        include_active_tuning=bool(payload.get("include_active_tuning", True)),
        mechanism=str(payload.get("mechanism", "auto")),
        drive_type=str(payload.get("drive_type", "auto")),
        drive_scheme=str(payload.get("drive_scheme", "auto")),
        bias_values=biases,
        heater_material=str(payload.get("heater_material", "TiN")),
        heater_width_um=float(payload.get("heater_width_um", 1.60)),
        heater_gap_um=float(payload.get("heater_gap_um", 0.90)),
        box_thickness_um=float(payload.get("box_thickness_um", 2.00)),
        electrode_gap_um=egap,
        na_p_cm3=float(payload.get("na_p_cm3", 5.0e17)),
        nd_n_cm3=float(payload.get("nd_n_cm3", 5.0e17)),
        pn_offset_nm=float(payload.get("pn_offset_nm", 25.0)),
    )
    res["splitter_catalog"] = MZI_SPLITTER_CATALOG
    return res


async def api_design_mzi(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_design_mzi_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_mzi_gds_sync(payload: dict) -> dict:
    mzi_res = payload.get("mzi_result")
    if not isinstance(mzi_res, dict) or "splitter" not in mzi_res:
        mzi_res = _design_mzi_sync(payload)
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sp_type = mzi_res["splitter"]["splitter_type"]
    safe_mat = str(mzi_res["config"]["core_material"]).replace("-", "")
    out_file = GDS_EXPORT_DIR / f"MicroRing_MZI_{sp_type}_{safe_mat}.gds"
    info = export_mzi_gds(mzi_res, out_file)
    saved = Path(info["gds_path"])
    lyp_saved = Path(info["lyp_path"])
    info["saved_path"] = str(saved)
    info["gds_download_url"] = f"/api/download_file?name={saved.name}"
    info["lyp_download_url"] = f"/api/download_file?name={lyp_saved.name}"
    info["gds_view"] = analyze_uploaded_gds(str(saved))
    return info


async def api_export_mzi_gds(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_mzi_gds_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _export_mzi_interconnect_sync(payload: dict) -> dict:
    mzi_res = payload.get("mzi_result")
    if not isinstance(mzi_res, dict) or "splitter" not in mzi_res:
        mzi_res = _design_mzi_sync(payload)
    GDS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sp_type = mzi_res["splitter"]["splitter_type"]
    safe_mat = str(mzi_res["config"]["core_material"]).replace("-", "")
    sub_dir = GDS_EXPORT_DIR / f"mzi_ic_{sp_type}_{safe_mat}"
    ic = export_mzi_interconnect_bundle(mzi_res, sub_dir)
    zp = Path(ic["zip_path"])
    ic["zip_download_url"] = f"/api/download_file?name={zp.name}"
    return ic


async def api_export_mzi_interconnect(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_export_mzi_interconnect_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def api_download_file(request: Request) -> Response:
    raw_name = request.query_params.get("name", "")
    safe = _safe_filename(raw_name)
    candidate = (GDS_EXPORT_DIR / safe).resolve()
    if not candidate.is_relative_to(GDS_EXPORT_DIR.resolve()) or not candidate.is_file():
        return JSONResponse({"error": f"File '{safe}' not found in exports."}, status_code=404)
    return Response(
        content=candidate.read_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{candidate.name}"'},
    )


async def api_demo_spectrum(request: Request) -> JSONResponse:
    kind = request.query_params.get("kind", "split_doublet")
    rng = np.random.default_rng(20261031)
    if kind == "lorentzian":
        wl = np.linspace(1549.94, 1550.06, 450)
        amp = cmt_all_pass_complex_amplitude(wl, 1550.000, kappa_int_ghz=0.42, kappa_ext_ghz=0.35)
        tr = (0.98 - 0.03 * (wl - 1550.0)) * (np.abs(amp) ** 2) + rng.normal(0.0, 0.004, len(wl))
        return JSONResponse({
            "wavelength_nm": np.round(wl, 6).tolist(),
            "transmission": np.round(np.clip(tr, 0.001, 1.2), 6).tolist(),
            "ring_length_um": 251.33,
        })

    if kind == "fano":
        wl = np.linspace(1549.90, 1550.10, 450)
        tr = cmt_fano_background_power(
            wl, 1550.000, kappa_int_ghz=0.65, kappa_ext_ghz=0.55,
            fano_phase_rad=1.05, bg_refl_amp=0.18, baseline_level=0.96
        ) + rng.normal(0.0, 0.004, len(wl))
        return JSONResponse({
            "wavelength_nm": np.round(wl, 6).tolist(),
            "transmission": np.round(np.clip(tr, 0.001, 1.3), 6).tolist(),
            "ring_length_um": 251.33,
        })

    if kind == "multi_fsr":
        wl = np.linspace(1542.0, 1558.0, 3600)
        pred = tmm_broadband_mrr_spectrum(
            wl,
            radius_um=42.0,
            neff_at_ref=2.12,
            ng_at_ref=2.385,
            lambda_ref_nm=1550.0,
            dispersion_d2_per_nm2=1.2e-5,
            alpha_loss_db_per_cm=0.55,
            kappa_power_thru=0.035,
        )
        tr = pred["thru_T"] + rng.normal(0.0, 0.003, len(wl))
        return JSONResponse({
            "wavelength_nm": np.round(wl, 5).tolist(),
            "transmission": np.round(np.clip(tr, 0.001, 1.15), 6).tolist(),
            "ring_length_um": round(2.0 * math.pi * 42.0, 3),
        })

    wl = np.linspace(1550.66, 1550.74, 500)
    tr = cmt_split_doublet_power(
        wl,
        lambda0_nm=1550.700,
        kappa_int_ghz=0.18,
        kappa_ext_ghz=0.15,
        mu_split_ghz=0.26,
        baseline_level=0.97,
        baseline_slope_per_nm=-0.04,
    ) + rng.normal(0.0, 0.0035, len(wl))
    return JSONResponse({
        "wavelength_nm": np.round(wl, 6).tolist(),
        "transmission": np.round(np.clip(tr, 0.001, 1.2), 6).tolist(),
        "ring_length_um": 263.89,
    })


def _fit_spectrum_sync(payload: dict) -> dict:
    wl = np.asarray(payload["wavelength_nm"], dtype=float)
    tr = np.asarray(payload["transmission"], dtype=float)
    model_type = str(payload.get("model_type", "auto"))
    known_regime = payload.get("known_regime") or None
    l_rt = float(payload.get("ring_length_um", 250.0))
    ng_assumed = payload.get("ng_assumed")
    ng_val = float(ng_assumed) if ng_assumed not in (None, "") else 2.10

    if model_type == "multi_fsr":
        multi = fit_multi_resonance_spectrum(wl, tr, ring_length_um=l_rt, known_regime=known_regime)
        return {"mode": "multi_fsr", "result": multi}

    summary, y_fit, _ = fit_single_resonance_cmt(
        wl,
        tr,
        model_type=model_type,
        known_regime=known_regime,
        ng_assumed=ng_val,
        ring_length_um=l_rt,
    )
    # Also return normalized linear trace actually fitted (in case input was dB/dBm)
    tr_norm = 10.0 ** ((tr - float(np.percentile(tr, 95))) / 10.0) if float(np.min(tr)) < -0.25 else tr
    return {
        "mode": "single",
        "summary": summary.to_dict(),
        "y_measured_linear": np.round(tr_norm, 6).tolist(),
        "y_fit": np.round(y_fit, 6).tolist(),
        "residuals": np.round(tr_norm - y_fit, 6).tolist(),
    }


async def api_fit_spectrum(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        res = await anyio.to_thread.run_sync(_fit_spectrum_sync, payload)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def create_app() -> Starlette:
    routes = [
        Route("/", homepage, methods=["GET"]),
        Route("/api/hardware", api_hardware, methods=["GET"]),
        Route("/api/ai_backends", api_ai_backends, methods=["GET"]),
        Route("/api/benchmark", api_benchmark, methods=["GET"]),
        Route("/api/materials", api_materials, methods=["GET"]),
        Route("/api/material_dispersion", api_material_dispersion, methods=["GET"]),
        Route("/api/chat_init", api_chat_init, methods=["POST"]),
        Route("/api/chat", api_chat, methods=["POST"]),
        Route("/api/upload_gds", api_upload_gds, methods=["POST"]),
        Route("/api/solve_mode", api_solve_mode, methods=["POST"]),
        Route("/api/sweep_width", api_sweep_width, methods=["POST"]),
        Route("/api/design_mrr", api_design_mrr, methods=["POST"]),
        Route("/api/sweep_mrr", api_sweep_mrr, methods=["POST"]),
        Route("/api/export_gds", api_export_gds, methods=["POST"]),
        Route("/api/export_sweep_gds", api_export_sweep_gds, methods=["POST"]),
        Route("/api/fab_tolerance", api_fab_tolerance, methods=["POST"]),
        Route("/api/export_fab_gds", api_export_fab_gds, methods=["POST"]),
        Route("/api/active_tuning", api_active_tuning, methods=["POST"]),
        Route("/api/export_interconnect", api_export_interconnect, methods=["POST"]),
        Route("/api/io_coupler", api_io_coupler, methods=["POST"]),
        Route("/api/export_io_gds", api_export_io_gds, methods=["POST"]),
        Route("/api/export_io_interconnect", api_export_io_interconnect, methods=["POST"]),
        Route("/api/high_q_fdtd", api_high_q_fdtd, methods=["POST"]),
        Route("/api/design_mzi", api_design_mzi, methods=["POST"]),
        Route("/api/export_mzi_gds", api_export_mzi_gds, methods=["POST"]),
        Route("/api/export_mzi_interconnect", api_export_mzi_interconnect, methods=["POST"]),
        Route("/api/download_file", api_download_file, methods=["GET"]),
        Route("/api/demo_spectrum", api_demo_spectrum, methods=["GET"]),
        Route("/api/fit_spectrum", api_fit_spectrum, methods=["POST"]),
    ]
    return Starlette(debug=False, routes=routes)


app = create_app()
