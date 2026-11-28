"""Token-Minimal Local-Compute Tooling for AI Agents (`Codex`, `Claude`, `Antigravity`, `Ollama`).

Design philosophy:
  1. **100% Local Physics Execution**: 2D vector FEM eigenmodes (with persistent SSD mode
     caching), spatial pulley/point/racetrack CMT integrals, GPU/AMX/CPU broadband TMM
     spectra, GDSII polygon extraction, and BIC curve fitting run entirely on local hardware.
  2. **Zero Array Dumping (<150 Tokens per Call)**: Large arrays (`wavelength_nm`, `thru_dB`,
     2D mode fields, GDS vertex lists) are written to local `.csv`/`.gds` files under
     `outputs/agent_runs/`, while stdout / MCP returns ONLY compact scalar KPIs + file paths.
"""
from __future__ import annotations

from dataclasses import fields, replace
import math
import os
from pathlib import Path
import time

import numpy as np

from .active_tuning import export_lumerical_interconnect_bundle, run_active_tuning_analysis
from .acousto_optic_saw import (
    SAWAcoustoOpticConfig,
    SBSBrillouinConfig,
    simulate_saw_acousto_optic_mrr,
    simulate_sbs_brillouin_mrr,
)
from .advanced_mrr import (
    EulerBendConfig,
    HighOrderFilterConfig,
    LargeSignalTDCMTConfig,
    MZICoupledMRRConfig,
    PhotonicMoleculeConfig,
    SensorAthermalConfig,
    design_euler_racetrack_bend,
    simulate_high_order_mrr_filter,
    simulate_large_signal_td_cmt_modulator,
    simulate_mrr_sensor_and_athermal,
    simulate_mzi_coupled_mrr,
    simulate_photonic_molecule,
)
from .quantum_and_laser_mrr import (
    MRRLaserConfig,
    QuantumSFWMConfig,
    simulate_mrr_laser,
    simulate_quantum_sfwm_mrr,
)
from .swg_and_phc_mrr import (
    PhCMicroringConfig,
    SWGMicroringConfig,
    simulate_phc_microring,
    simulate_swg_microring,
)
from .nanoantenna_tcmt import (
    NanoantennaTCMTConfig,
    fit_nanoantenna_tcmt_spectrum,
    simulate_nanoantenna_tcmt,
)
from .photodetector_oeo import (
    OEONeuronConfig,
    WaveguidePDConfig,
    simulate_oeo_neuron,
    simulate_waveguide_photodetector,
)
from .polarization_mdm import (
    PBSMDMConfig,
    PSRConfig,
    simulate_pbs_or_mdm,
    simulate_psr_rotator,
)
from .pcm_photonics import (
    PCMMRRConfig,
    PCMThermalPulseConfig,
    simulate_pcm_mrr_switch,
    simulate_pcm_thermal_pulse,
)
from .photonic_computing import (
    MRRWeightBankConfig,
    MZIMeshConfig,
    simulate_mrr_weight_bank,
    simulate_mzi_unitary_mesh,
)
from .wbg_and_awg import (
    AWGConfig,
    WaveguideBraggConfig,
    simulate_awg_demux,
    simulate_waveguide_bragg_grating,
)
from .pic_circuit_netlist import PICNetlistConfig, simulate_pic_circuit_netlist
from .inverse_design_adjoint import AdjointInverseConfig, optimize_inverse_photonic_device
from .fab_tolerance import (
    export_fab_bias_array_gds,
    run_fabrication_error_analysis,
    run_foundry_process_corners,
)
from .foundry_pdk import (
    FOUNDRY_PDK_CATALOG,
    build_pn_heater_ring_modulator_gds,
    export_pdk_csv_tables,
    extract_3d_electrical_lvs,
    load_pdk_from_csv_dir,
    run_gds_polygon_drc,
    solve_gds_cutline_multiphysics,
)
from .gds_bridge import (
    analyze_uploaded_gds,
    export_sweep_array_gds,
    write_klayout_layer_props,
)
from .hardware_accel import detect_hardware_profile, run_compute_benchmark
from .io_couplers import (
    design_butt_edge_coupler,
    design_grating_coupler,
    export_io_coupler_gds,
    export_io_coupler_interconnect,
)
from .materials import C0_M_PER_S, MATERIAL_CATALOG
from .mrr_cmt_fit_predict import fit_multi_resonance_spectrum, fit_single_resonance_cmt
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
    MZIDesignConfig,
    export_mzi_gds,
    export_mzi_interconnect_bundle,
    simulate_mzi,
)
from .eom_electrode import (
    EOMElectrodeConfig,
    analyze_eo_wafer_orientation,
    design_eom_electrode,
    export_eom_electrode_gds,
    export_eom_rf_interconnect_bundle,
)
from .multiphysics_tcad import (
    PNDopingTCADConfig,
    ThermalSimConfig,
    export_pn_doping_gds,
    export_tcad_data_bundle,
    export_thermal_shifter_gds,
    simulate_cointegrated_pn_heater_ring,
    simulate_pn_doping_tcad,
    simulate_thermal_phase_shifter,
)
from .prism_optimizer import (
    PRISM_PROBLEM_CATALOG,
    audit_mrr_and_pn_physics_closure,
    run_prism_closed_loop_optimizer,
)
from .sweeps import run_mrr_parameter_sweep, run_waveguide_width_sweep
from .nonlinear_lle import (
    FWMBSConfig,
    LLEConfig,
    SHGConfig,
    ThermalKerrBistabilityConfig,
    compute_resonator_dispersion_and_nl,
    simulate_chi2_shg_resonator,
    simulate_fwm_bragg_scattering,
    simulate_lle_microcomb,
    simulate_thermal_kerr_bistability,
)
from .wdm_link import WDMLinkConfig, simulate_wdm_microring_link
from .twod_materials_mrr import TWOD_MATERIAL_CATALOG, TwoDMaterialMRRConfig, list_2d_materials, simulate_2d_material_mrr
from .ep_microring import MRRExceptionalPointConfig, simulate_mrr_exceptional_point

AGENT_OUT_DIR = Path(os.environ.get("MICRORING_AGENT_OUT", str(get_default_output_root() / "agent_runs")))


def _cfg_from_dict(d: dict) -> MRRDesignConfig:
    valid = {f.name for f in fields(MRRDesignConfig)}
    aliases = {
        "mat": "core_material",
        "material": "core_material",
        "clad": "clad_material",
        "top_clad": "top_clad_material",
        "top": "topology",
        "h_um": "film_height_um",
        "slab_um": "slab_height_um",
        "w_ring_um": "ring_width_um",
        "w_bus_um": "bus_width_um",
        "r_um": "ring_radius_um",
        "r2_um": "ring2_radius_um",
        "gap_um": "gap_thru_um",
        "ls_um": "straight_length_um",
        "theta_deg": "pulley_angle_thru_deg",
        "wl_nm": "wavelength_nm",
        "pol": "polarization",
        "loss_db_cm": "alpha_loss_db_per_cm",
        "dt_k": "delta_temp_k",
    }
    _mat_defaults = {
        "Si3N4": {"film_height_um": 0.40, "ring_width_um": 1.30, "bus_width_um": 1.00, "gap_thru_um": 0.52},
        "4H-SiC": {"film_height_um": 0.60, "ring_width_um": 1.10, "bus_width_um": 0.82, "gap_thru_um": 0.30},
        "LN_Xcut": {"film_height_um": 0.60, "slab_height_um": 0.30, "ring_width_um": 1.20, "bus_width_um": 1.00, "gap_thru_um": 0.38},
        "AlN": {"film_height_um": 0.45, "ring_width_um": 1.10, "bus_width_um": 0.90, "gap_thru_um": 0.35},
        "Ge": {"film_height_um": 2.00, "ring_width_um": 2.20, "bus_width_um": 2.20, "gap_thru_um": 0.45, "clad_material": "Air", "top_clad_material": "Air"},
    }
    mapped: dict = {}
    for k, v in d.items():
        if v is None:
            continue
        target_k = aliases.get(k, k)
        if target_k in valid:
            mapped[target_k] = v
    mat_name = mapped.get("core_material")
    if mat_name in _mat_defaults:
        for dk, dv in _mat_defaults[mat_name].items():
            mapped.setdefault(dk, dv)
    if "gap_thru_um" in mapped and "gap_drop_um" not in mapped:
        mapped["gap_drop_um"] = mapped["gap_thru_um"]
    return MRRDesignConfig(**mapped)


def agent_status() -> dict:
    """Return compact local compute profile + valid material/topology identifiers (~90 tokens)."""
    hw = detect_hardware_profile()
    bench = run_compute_benchmark(50_000)
    return {
        "os": f"{hw.os_name}-{hw.arch}",
        "backend": hw.selected_spectrum_backend,
        "cores": hw.physical_cores,
        "ram_avail_gb": hw.ram_available_gb,
        "cuda_gpu": hw.cuda_device_name,
        "tmm_mpts_s": bench["throughput_mpts_per_s"],
        "materials": list(MATERIAL_CATALOG.keys()),
        "topologies": list(TOPOLOGY_CATALOG.keys()),
        "pdks": list(FOUNDRY_PDK_CATALOG.keys()),
    }


def agent_width_cutoff(params: dict) -> dict:
    """Run local 2D FEM waveguide width sweep; save full CSV locally, return compact cutoff KPIs."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mat = str(params.get("core_material") or params.get("mat") or "Si3N4")
    pol = str(params.get("polarization") or params.get("pol") or "TE0")
    h_um = float(params.get("film_height_um") or params.get("h_um") or 0.40)
    slab_um = float(params.get("slab_height_um") or params.get("slab_um") or 0.0)
    sw_deg = float(params.get("sidewall_deg") or 85.0)
    wl_nm = float(params.get("wavelength_nm") or params.get("wl_nm") or 1550.0)
    w_min = float(params.get("w_min_um") or 0.50)
    w_max = float(params.get("w_max_um") or 2.00)
    n_pts = int(params.get("num_points") or 7)
    top_clad = str(params.get("top_clad_material") or "SiO2")
    clad = str(params.get("clad_material") or "SiO2")

    res = run_waveguide_width_sweep(
        core_material=mat,
        clad_material=clad,
        top_clad_material=top_clad,
        h_film_um=h_um,
        h_slab_um=slab_um,
        sidewall_deg=sw_deg,
        wavelength_nm=wl_nm,
        w_min_um=w_min,
        w_max_um=w_max,
        num_points=n_pts,
        polarization=pol,
    )
    csv_path = AGENT_OUT_DIR / f"width_sweep_{mat}_{pol}.csv"
    ho_clean = [np.nan if v is None else v for v in res["neff_higher_order"]]
    np.savetxt(
        csv_path,
        np.column_stack([
            res["widths_um"],
            res["neff_fundamental"],
            res["ng_fundamental"],
            ho_clean,
            res["core_confinement"],
        ]),
        delimiter=",",
        header="width_um,neff_fund,ng_fund,neff_higher_order,core_confinement",
        comments="",
    )
    idx_rec = int(np.argmin(np.abs(np.asarray(res["widths_um"]) - res["recommended_ring_width_um"])))
    return {
        "mat": mat,
        "pol": pol,
        "sm_max_w_um": res["single_mode_max_width_um"],
        "rec_w_bus_um": res["recommended_bus_width_um"],
        "rec_w_ring_um": res["recommended_ring_width_um"],
        "neff_at_rec": res["neff_fundamental"][idx_rec],
        "ng_at_rec": res["ng_fundamental"][idx_rec],
        "csv": str(csv_path),
    }


def agent_design_mrr(params: dict) -> dict:
    """Synthesize MRR locally (optional target FSR, 1-click critical coupling & GDS export)."""
    t0 = time.perf_counter()
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_dict(params)

    # Optional: auto-solve radius R for requested target_fsr_ghz
    target_fsr_ghz = params.get("target_fsr_ghz")
    if target_fsr_ghz:
        scout = simulate_mrr_topology(replace(cfg, num_spectrum_points=256))
        ng = float(scout["metrics"]["ng_ring"])
        l_rt_m = C0_M_PER_S / (ng * float(target_fsr_ghz) * 1e9)
        ls_um = cfg.straight_length_um if cfg.topology.startswith("racetrack") else 0.0
        r_solved = max(5.0, round((l_rt_m * 1e6 - 2.0 * ls_um) / (2.0 * math.pi), 3))
        cfg = replace(cfg, ring_radius_um=r_solved)

    res = simulate_mrr_topology(cfg)
    if params.get("apply_cc"):
        cc_info = res["critical_coupling"]
        p_cc = cc_info["parameter"]
        cfg = replace(cfg, **{p_cc: float(cc_info["recommended_value"])})
        res = simulate_mrr_topology(cfg)

    m = res["metrics"]
    cc = res["critical_coupling"]
    sp = res["spectrum"]
    tag = f"{cfg.topology}_{cfg.core_material}_R{cfg.ring_radius_um:.1f}"
    csv_path = AGENT_OUT_DIR / f"spec_{tag}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sp["wavelength_nm"], sp["thru_dB"], sp["drop_dB"]]),
        delimiter=",",
        header="wavelength_nm,thru_dB,drop_dB",
        comments="",
    )

    out = {
        "top": cfg.topology,
        "mat": cfg.core_material,
        "R_um": cfg.ring_radius_um,
        "W_ring_um": cfg.ring_width_um,
        "W_bus_um": cfg.bus_width_um,
        "gap_um": cfg.gap_thru_um,
        "loss_dB_cm": round(m["alpha_loss_db_per_cm"], 3),
        "neff": round(m["neff_ring"], 5),
        "ng": round(m["ng_ring"], 5),
        "FSR_GHz": round(m["FSR_GHz"], 2),
        "FSR_nm": round(m["FSR_nm"], 4),
        "Q_L": int(m["Q_loaded"]),
        "Q_int": int(m["Q_int"]),
        "Q_ext": int(m["Q_ext_thru"]),
        "ER_dB": round(m["ER_thru_dB"], 2),
        "regime": cc["regime"],
        "cc_param": cc["parameter"],
        "cc_val": cc["recommended_value"],
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
        "csv": str(csv_path),
    }
    if abs(float(m.get("tuning_shift_pm", 0.0))) > 1e-6:
        out["shift_pm"] = round(m["tuning_shift_pm"], 2)
    if params.get("export_gds"):
        gds_path = export_mrr_gds(cfg, AGENT_OUT_DIR / f"{tag}.gds")
        write_klayout_layer_props(Path(gds_path).with_suffix(".lyp"))
        out["gds"] = gds_path
    return out


def agent_sweep_mrr(params: dict) -> dict:
    """Run local multi-point MRR sweep (cached 2D FEM modes); save CSV/GDS locally, return compact best point."""
    t0 = time.perf_counter()
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_dict(params)
    p_name = str(params.get("param_name") or TOPOLOGY_CATALOG[cfg.topology]["coupler_param"])
    if "values" in params and params["values"]:
        vals = [float(x) for x in params["values"]]
    else:
        vmin = float(params.get("v_min", 0.16))
        vmax = float(params.get("v_max", 0.32))
        n_pts = int(params.get("num_points", 7))
        vals = [round(float(x), 4) for x in np.linspace(vmin, vmax, max(2, n_pts))]

    sw = run_mrr_parameter_sweep(cfg, p_name, vals)
    csv_path = AGENT_OUT_DIR / f"sweep_{cfg.core_material}_{p_name}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sw["values"], sw["ER_thru_dB"], sw["Q_loaded"], sw["resonance_shift_pm"]]),
        delimiter=",",
        header=f"{p_name}_{sw['unit']},ER_thru_dB,Q_loaded,resonance_shift_pm",
        comments="",
    )
    out = {
        "param": p_name,
        "unit": sw["unit"],
        "n_pts": len(vals),
        "range": [vals[0], vals[-1]],
        "best_val": sw["best_extinction_value"],
        "best_ER_dB": sw["best_extinction_dB"],
        "Q_L_at_best": int(sw["Q_loaded"][sw["best_critical_coupling_index"]]),
        "shift_span_pm": [sw["resonance_shift_pm"][0], sw["resonance_shift_pm"][-1]],
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
        "csv": str(csv_path),
    }
    if params.get("export_array_gds"):
        arr_gds = export_sweep_array_gds(cfg, p_name, vals, AGENT_OUT_DIR / f"array_{cfg.core_material}_{p_name}.gds")
        out["array_gds"] = arr_gds
    return out


def agent_inspect_gds(params: dict) -> dict:
    """Extract MRR topology & dimensions from a local `.gds`/`.oas` without dumping polygon vertices."""
    gds_path = str(params["gds_path"])
    lyr = params.get("core_layer")
    core_tuple = (int(lyr[0]), int(lyr[1])) if lyr else None
    info = analyze_uploaded_gds(gds_path, core_layer=core_tuple, cell_name=params.get("cell_name"))
    return {
        "file": info["file_name"],
        "cell": info["cell_name"],
        "core_layer": info["selected_core_layer"],
        "bbox_um": info["bbox_um"],
        "extracted": info["extracted_geometry"],
    }


def agent_fit_spectrum(params: dict) -> dict:
    """Fit a local experimental `.csv`/`.txt` transmission trace via CMT without loading trace into LLM context."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_in = Path(str(params["csv_path"]))
    raw = np.genfromtxt(csv_in, delimiter="," if csv_in.suffix.lower() == ".csv" else None, comments="#")
    if raw.ndim != 2 or raw.shape[1] < 2:
        # Handle header row gracefully
        raw = np.genfromtxt(csv_in, delimiter="," if csv_in.suffix.lower() == ".csv" else None, skip_header=1)
    mask = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])
    wl, tr = raw[mask, 0], raw[mask, 1]
    if np.min(tr) < 0.0:
        # Auto-convert dB transmission to linear power if user passed dB CSV
        tr = 10.0 ** (tr / 10.0)

    model_type = str(params.get("model_type") or "auto")
    regime = params.get("known_regime") or None
    l_rt = float(params.get("ring_length_um") or 250.0)
    ng_in = float(params.get("ng") or params.get("ng_assumed") or 2.10)
    if model_type == "multi_fsr":
        m_res = fit_multi_resonance_spectrum(wl, tr, ring_length_um=l_rt, known_regime=regime)
        return {
            "mode": "multi_fsr",
            "n_peaks": m_res["num_resonances"],
            "FSR_GHz": round(m_res["mean_fsr_ghz"], 3),
            "FSR_nm": round(m_res["mean_fsr_nm"], 4),
            "ng_extracted": round(m_res["ng_extracted"], 5) if m_res["ng_extracted"] else None,
            "mean_Q_L": int(m_res["mean_Q_loaded"]),
            "mean_Q_int": int(m_res["mean_Q_int"]),
            "loss_dB_cm": m_res["alpha_loss_db_per_cm"],
        }

    s, y_fit, _ = fit_single_resonance_cmt(
        wl, tr, model_type=model_type, known_regime=regime, ng_assumed=ng_in, ring_length_um=l_rt
    )
    fit_csv = AGENT_OUT_DIR / f"fitted_{csv_in.stem}.csv"
    np.savetxt(
        fit_csv,
        np.column_stack([wl, tr, y_fit]),
        delimiter=",",
        header="wavelength_nm,measured_linear,cmt_fit_linear",
        comments="",
    )
    return {
        "model": s.model_type,
        "lambda0_nm": round(s.lambda0_nm, 5),
        "FWHM_pm": round(s.fwhm_pm, 3),
        "Q_L": int(s.Q_loaded),
        "Q_int": int(s.Q_int),
        "Q_ext": int(s.Q_ext),
        "loss_dB_cm": round(s.alpha_loss_db_per_cm, 4),
        "loss_under_dB_cm": s.alpha_loss_under_db_per_cm,
        "loss_over_dB_cm": s.alpha_loss_over_db_per_cm,
        "ng_used": s.ng_used,
        "ER_dB": round(s.extinction_ratio_db, 2),
        "split_2g_ghz": round(2.0 * s.mu_split_ghz, 4),
        "regime": s.coupling_regime_status,
        "r2": round(s.r_squared, 5),
        "fit_csv": str(fit_csv),
    }


def agent_fab_tolerance(params: dict) -> dict:
    """Run lithography exposure dose / photoresist-tone / CD-bias error analysis (<150 tokens)."""
    t0 = time.perf_counter()
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_dict(params)
    if params.get("apply_cc"):
        scout = simulate_mrr_topology(cfg)
        cc_info = scout["critical_coupling"]
        cfg = replace(cfg, **{cc_info["parameter"]: float(cc_info["recommended_value"])})

    if params.get("corner_mode") or params.get("corners") or str(params.get("mode") or "").lower() in ("foundry_corners", "corners"):
        pdk_k = str(params.get("pdk") or params.get("pdk_key") or "gsip_soi220")
        pc = run_foundry_process_corners(
            cfg,
            pdk_key=pdk_k,
            delta_w_3sigma_nm=float(params["cd_nm"]) if params.get("cd_nm") is not None else None,
            delta_h_core_3sigma_nm=float(params["dh_nm"]) if params.get("dh_nm") is not None else None,
            mc_trials=int(params.get("mc_trials") or 128),
        )
        mc = pc["monte_carlo_summary"]
        return {
            "mode": "foundry_corners",
            "pdk": pdk_k,
            "lam_3sig_nm": mc["lambda_res_3sigma_nm"],
            "fsr_3sig_pct": mc["fsr_3sigma_spread_pct"],
            "trim_3sig_K": mc["heater_trim_3sigma_K"],
            "yield_ER15_pct": mc["yield_ER_gt_15dB_pct"],
            "ER_p05_p50_p95": mc["ER_dB_p05_p50_p95"],
            "dl_dW_pm_nm": pc["sensitivities"]["dlambda_dW_pm_per_nm"],
            "dl_dHc_pm_nm": pc["sensitivities"]["dlambda_dH_core_pm_per_nm"],
            "ms": round((time.perf_counter() - t0) * 1e3, 1),
        }

    resist = str(params.get("resist_tone") or params.get("resist") or "positive_pr")
    dose_pct = float(params.get("max_dose_error_pct") or params.get("dose_pct") or 15.0)
    slope = params.get("dose_slope_nm_per_pct") or params.get("slope_nm_pct")
    cd_nm = float(params.get("max_cd_bias_nm") or params.get("cd_nm") or 30.0)
    prox = params.get("proximity_gap_factor") or params.get("prox")
    dh_nm = float(params.get("delta_h_film_nm") or params.get("dh_nm") or 0.0)
    n_pts = int(params.get("num_points") or 7)

    fab = run_fabrication_error_analysis(
        cfg,
        resist_tone=resist,
        max_dose_error_pct=dose_pct,
        dose_slope_nm_per_pct=float(slope) if slope is not None else None,
        max_cd_bias_nm=cd_nm,
        proximity_gap_factor=float(prox) if prox is not None else None,
        delta_h_film_nm=dh_nm,
        num_points=n_pts,
    )
    corners = fab["corners"]
    csv_path = AGENT_OUT_DIR / f"fab_{cfg.core_material}_{resist}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([
            [c["dose_error_pct"] for c in corners],
            [c["delta_cd_nm"] for c in corners],
            [c["w_ring_fab_um"] for c in corners],
            [c["gap_fab_um"] for c in corners],
            [c["extinction_ratio_dB"] for c in corners],
            [c["Q_loaded"] for c in corners],
            [c["delta_wl_res_nm"] for c in corners],
        ]),
        delimiter=",",
        header="dose_error_pct,delta_cd_nm,w_ring_fab_um,gap_fab_um,ER_dB,Q_loaded,delta_wl_res_nm",
        comments="",
    )
    s = fab["summary"]
    out = {
        "resist": fab["resist_tone"],
        "nom_ER_dB": s["nominal_er_dB"],
        "under_dose": {
            "dose_pct": s["under_dose_corner"]["dose_error_pct"],
            "dCD_nm": s["under_dose_corner"]["delta_cd_nm"],
            "gap_nm": round(s["under_dose_corner"]["gap_fab_um"] * 1e3, 1),
            "ER_dB": s["under_dose_corner"]["extinction_ratio_dB"],
            "dwl_nm": s["under_dose_corner"]["delta_wl_res_nm"],
            "regime": s["under_dose_corner"]["coupling_regime"],
        },
        "over_dose": {
            "dose_pct": s["over_dose_corner"]["dose_error_pct"],
            "dCD_nm": s["over_dose_corner"]["delta_cd_nm"],
            "gap_nm": round(s["over_dose_corner"]["gap_fab_um"] * 1e3, 1),
            "ER_dB": s["over_dose_corner"]["extinction_ratio_dB"],
            "dwl_nm": s["over_dose_corner"]["delta_wl_res_nm"],
            "regime": s["over_dose_corner"]["coupling_regime"],
        },
        "cd_win_ER15_nm": s["cd_window_er15_nm"],
        "dose_win_ER15_pct": s["dose_window_er15_pct"],
        "dl_dCD_pm_nm": s["sensitivity_dl_dcd_pm_per_nm"],
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
        "csv": str(csv_path),
    }
    if params.get("export_bias_gds"):
        gds_info = export_fab_bias_array_gds(
            cfg,
            AGENT_OUT_DIR / f"opc_bias_array_{cfg.core_material}_{cfg.topology}.gds",
            proximity_gap_factor=fab["proximity_gap_factor"],
        )
        out["bias_gds"] = gds_info["gds_path"]
    return out


def agent_active_tuning(params: dict) -> dict:
    """Run 2D-FEM active tuning (TO/EO/Si-PN/Si-PIN + T/E/V/I sweeps) & optional INTERCONNECT export (<150 tokens)."""
    t0 = time.perf_counter()
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_dict(params)
    if params.get("apply_cc"):
        scout = simulate_mrr_topology(cfg)
        cc_info = scout["critical_coupling"]
        cfg = replace(cfg, **{cc_info["parameter"]: float(cc_info["recommended_value"])})

    raw_b = params.get("bias_values") or params.get("biases")
    bias_list: list[float] | None = None
    if isinstance(raw_b, str) and raw_b.strip():
        bias_list = [float(x.strip()) for x in raw_b.split(",") if x.strip()]
    elif isinstance(raw_b, (list, tuple)) and len(raw_b) > 0:
        bias_list = [float(x) for x in raw_b]

    mech = str(params.get("mechanism") or "auto")
    drv = str(params.get("drive_type") or params.get("drive") or "auto")

    tune = run_active_tuning_analysis(
        cfg,
        mechanism=mech,
        drive_type=drv,
        bias_values=bias_list,
        active_fraction=float(params.get("active_fraction") or 0.78),
        heater_material=str(params.get("heater_material") or "TiN"),
        heater_width_um=float(params.get("heater_width_um") or 1.60),
        heater_gap_um=float(params.get("heater_gap_um") or 0.90),
        box_thickness_um=float(params.get("box_thickness_um") or 2.00),
        electrode_gap_um=float(params["electrode_gap_um"]) if params.get("electrode_gap_um") is not None else None,
        na_p_cm3=float(params.get("na_p_cm3") or 5.0e17),
        nd_n_cm3=float(params.get("nd_n_cm3") or 5.0e17),
        pn_offset_nm=float(params.get("pn_offset_nm") or 25.0),
        heavy_contact_offset_um=float(params.get("heavy_contact_offset_um") or 0.85),
    )

    eff = tune["efficiency"]
    states = tune["states"]
    csv_path = AGENT_OUT_DIR / f"tuning_{cfg.core_material}_{eff['mechanism']}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([
            [s["bias_value"] for s in states],
            [s["delta_neff_active_real"] for s in states],
            [s["alpha_active_db_per_cm"] for s in states],
            [s["delta_wl_pm"] for s in states],
            [s["ER_thru_dB"] for s in states],
        ]),
        delimiter=",",
        header=f"bias_{eff['drive_unit']},delta_neff_active_real,alpha_active_dB_per_cm,delta_wl_pm,ER_thru_dB",
        comments="",
    )

    out = {
        "mech": eff["mechanism"],
        "drive": f"{eff['drive_type']}[{eff['drive_unit']}]",
        "slope": f"{eff['primary_slope']} {eff['primary_slope_unit']}",
        "freq_slope": f"{eff['freq_slope']} {eff['freq_slope_unit']}",
        "pi_fom": f"{eff['pi_figure']} {eff['pi_figure_unit']}",
        "bw_3dB": f"{eff['bandwidth_3db']} {eff['bandwidth_unit']}" if eff.get("bandwidth_3db") else None,
        "dwl_span_pm": [states[0]["delta_wl_pm"], states[-1]["delta_wl_pm"]],
        "loss_span_dB_cm": [states[0]["alpha_active_db_per_cm"], states[-1]["alpha_active_db_per_cm"]],
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
        "csv": str(csv_path),
    }
    if params.get("export_interconnect"):
        ic = export_lumerical_interconnect_bundle(
            cfg,
            tune,
            AGENT_OUT_DIR / f"interconnect_{cfg.core_material}_{eff['mechanism']}",
            make_zip=True,
        )
        out["interconnect_zip"] = ic["zip_path"]
    return out


def agent_io_coupler(params: dict) -> dict:
    """Design Butt/Edge Coupler or 2D-FDFD Surface Grating Coupler locally (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctype = str(params.get("coupler_type") or params.get("type") or "grating_coupler").lower()
    mat = str(params.get("core_material") or params.get("mat") or "Si")
    wl = float(params.get("wavelength_nm") or params.get("wl_nm") or 1550.0)
    h_um = float(params.get("film_height_um") or params.get("h_um") or 0.22)
    pol = str(params.get("polarization") or params.get("pol") or "TE0")
    wb = float(params.get("w_bus_um") or 0.48)
    box_um = float(params.get("box_thickness_um") or params.get("box_um") or 2.0)
    fib = str(params.get("fiber_preset") or params.get("fiber") or ("Lensed_2p5um" if "butt" in ctype or "edge" in ctype else "SMF28_1550"))

    if "butt" in ctype or "edge" in ctype:
        res = design_butt_edge_coupler(
            core_material=mat,
            wavelength_nm=wl,
            polarization=pol,
            film_height_um=h_um,
            w_bus_um=wb,
            w_tip_um=float(params["w_tip_um"]) if params.get("w_tip_um") is not None else None,
            taper_length_um=float(params.get("taper_length_um") or 180.0),
            taper_profile=str(params.get("taper_profile") or "adiabatic"),
            fiber_preset=fib,
            mfd_x_um=float(params["mfd_um"]) if params.get("mfd_um") is not None else None,
            ar_coating=bool(params.get("ar_coating", True)),
            box_thickness_um=box_um,
        )
        m = res["metrics"]
        out = {
            "coupler": "butt_edge",
            "mat": mat,
            "fiber": res["fiber_preset"],
            "w_tip_um": res["w_tip_um"],
            "w_tip_opt_um": res["w_tip_optimal_um"],
            "IL_total_dB": m["total_il_db"],
            "eff_pct": m["total_efficiency_pct"],
            "RL_dB": m["return_loss_db"],
            "tol_1dB_xy_um": [m["tol_1db_x_um"], m["tol_1db_y_um"]],
            "rec_L_taper_um": res["recommended_taper_length_um"],
            "ms": res["elapsed_ms"],
        }
    else:
        etch_in = params.get("etch_depth_um") if params.get("etch_depth_um") is not None else params.get("etch_um")
        res = design_grating_coupler(
            core_material=mat,
            wavelength_nm=wl,
            polarization=pol,
            film_height_um=h_um,
            etch_depth_um=float(etch_in) if etch_in is not None else None,
            fiber_angle_deg=float(params.get("fiber_angle_deg") or 10.0),
            apodized=bool(params.get("apodized", True)),
            focusing=bool(params.get("focusing", True)),
            pitch_um=float(params["pitch_um"]) if params.get("pitch_um") is not None else None,
            num_periods=int(params.get("num_periods") or 26),
            grating_width_um=float(params.get("grating_width_um") or 12.0),
            w_bus_um=wb,
            box_thickness_um=box_um,
            fiber_preset=fib,
        )
        m = res["metrics"]
        out = {
            "coupler": "grating_coupler",
            "mat": mat,
            "apodized": res["apodized"],
            "focusing": res["focusing"],
            "pitch_mean_um": res["synthesis"]["pitch_mean_um"],
            "etch_um": res["etch_depth_um"],
            "peak_IL_dB": m["peak_il_db"],
            "eff_pct": m["peak_efficiency_pct"],
            "peak_wl_nm": m["peak_wavelength_nm"],
            "bw_1dB_3dB_nm": [m["bw_1db_nm"], m["bw_3db_nm"]],
            "S11_dB": m["back_reflection_s11_db"],
            "ms": res["elapsed_ms"],
        }

    if params.get("export_gds") or params.get("include_loopback"):
        ginfo = export_io_coupler_gds(
            res,
            AGENT_OUT_DIR / f"io_{res['coupler_type']}_{mat}.gds",
            include_ring_loopback=bool(params.get("include_loopback", False)),
        )
        out["gds"] = ginfo["gds_path"]
    if params.get("export_interconnect"):
        ic = export_io_coupler_interconnect(res, AGENT_OUT_DIR / f"io_ic_{res['coupler_type']}_{mat}")
        out["interconnect_zip"] = ic["zip_path"]
    return out


def agent_high_q_fdtd(params: dict) -> dict:
    """Diagnose High-Q FDTD non-convergence vs exact infinite-roundtrip TMM + optional coupler cutout GDS (<140 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_dict(params)
    if params.get("apply_cc"):
        scout = simulate_mrr_topology(cfg)
        cc_info = scout["critical_coupling"]
        cfg = replace(cfg, **{cc_info["parameter"]: float(cc_info["recommended_value"])})

    hq = analyze_high_q_fdtd_convergence(cfg)
    out = {
        "Q_L": int(hq["Q_loaded"]),
        "FWHM_pm": hq["FWHM_pm"],
        "tau_ph_ps": hq["photon_lifetime_ps"],
        "req_FDTD_1e5_ns": hq["required_fdtd_sim_time_1e5_ns"],
        "req_roundtrips": hq["required_fdtd_roundtrips_1e5"],
        "coupler_only_ps": hq["coupler_only_fdtd_time_ps"],
        "speedup_x": int(hq["time_domain_speedup_factor"]),
        "true_ER_dB": hq["true_converged_er_db"],
        "truncated_ER_dB": [
            {"t_ps": t["fdtd_sim_time_ps"], "ER_dB": t["apparent_er_db"]}
            for t in hq["truncated_fdtd_traces"]
        ],
    }
    if params.get("export_cutout") or params.get("export_cutout_gds"):
        cut = export_coupler_cutout_gds(cfg, AGENT_OUT_DIR)
        out["cutout_gds"] = cut["gds_path"]
        out["cutout_lsf"] = cut["lsf_path"]
    return out


def agent_design_mzi(params: dict) -> dict:
    """Design & simulate a Mach-Zehnder Interferometer (`Y-Branch`, `DC`, `2x2/1x2 MMI`) + active tuning (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mzi_fields = {f.name for f in fields(MZIDesignConfig)}
    aliases = {
        "mat": "core_material",
        "material": "core_material",
        "clad": "clad_material",
        "top_clad": "top_clad_material",
        "wl_nm": "wavelength_nm",
        "pol": "polarization",
        "h_um": "film_height_um",
        "slab_um": "slab_height_um",
        "splitter": "splitter_type",
        "combiner": "combiner_type",
        "w_um": "arm1_width_um",
        "w1_um": "arm1_width_um",
        "w2_um": "arm2_width_um",
        "l_arm_um": "arm1_length_um",
        "l1_um": "arm1_length_um",
        "dl_um": "delta_length_um",
        "fsr_nm": "target_fsr_nm",
        "fsr_ghz": "target_fsr_ghz",
        "loss_db_cm": "alpha_loss_db_per_cm",
    }
    kw: dict = {}
    for k, v in params.items():
        if v is None:
            continue
        target = aliases.get(k, k)
        if target in mzi_fields:
            kw[target] = v
    cfg = MZIDesignConfig(**kw)

    raw_b = params.get("bias_values") or params.get("biases")
    bias_list: list[float] | None = None
    if isinstance(raw_b, str) and raw_b.strip():
        bias_list = [float(x.strip()) for x in raw_b.split(",") if x.strip()]
    elif isinstance(raw_b, (list, tuple)) and len(raw_b) > 0:
        bias_list = [float(x) for x in raw_b]

    include_tuning = bool(params.get("include_active_tuning", True))
    res = simulate_mzi(
        cfg,
        include_active_tuning=include_tuning,
        mechanism=str(params.get("mechanism") or "auto"),
        drive_type=str(params.get("drive_type") or params.get("drive") or "auto"),
        drive_scheme=str(params.get("drive_scheme") or "auto"),
        bias_values=bias_list,
        heater_material=str(params.get("heater_material") or "TiN"),
        electrode_gap_um=float(params["electrode_gap_um"]) if params.get("electrode_gap_um") is not None else None,
    )

    ps = res["passive_spectrum"]
    sp = res["splitter"]
    m = res["metrics"]
    csv_path = AGENT_OUT_DIR / f"mzi_{sp['splitter_type']}_{cfg.core_material}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([ps["wavelength_nm"], ps["bar_dB"], ps["cross_dB"]]),
        delimiter=",",
        header="wavelength_nm,bar_dB,cross_dB",
        comments="",
    )

    out = {
        "splitter": sp["splitter_type"],
        "mat": cfg.core_material,
        "L1_um": res["arm_eigenmodes"]["arm1"]["length_um"],
        "dL_um": res["arm_eigenmodes"]["delta_length_um"],
        "FSR_nm": m["FSR_nm"],
        "FSR_GHz": m["FSR_GHz"],
        "ER_bar_dB": m["ER_bar_passive_dB"],
        "IL_min_dB": m["IL_min_passive_dB"],
        "splitter_IL_dB": m["splitter_excess_loss_dB"],
        "splitter_imb_dB": m["splitter_imbalance_dB"],
        "ms": res["elapsed_ms"],
        "csv": str(csv_path),
    }
    act = res.get("active_tuning")
    if act:
        out["tuning"] = {
            "mech": act["mechanism"],
            "scheme": act["drive_scheme"],
            "V_pi_V": act["v_pi_v"],
            "V_pi_L_Vcm": act["v_pi_l_vcm"],
            "P_pi_mW": act["p_pi_mw"],
            "bw_3dB": f"{act['bandwidth_3db']} {act['bandwidth_unit']}" if act.get("bandwidth_3db") else None,
            "dyn_ER_dB": act["dynamic_er_bar_db"],
        }
    if params.get("export_gds"):
        ginfo = export_mzi_gds(res, AGENT_OUT_DIR / f"mzi_{sp['splitter_type']}_{cfg.core_material}.gds")
        out["gds"] = ginfo["gds_path"]
    if params.get("export_interconnect"):
        ic = export_mzi_interconnect_bundle(res, AGENT_OUT_DIR / f"mzi_ic_{sp['splitter_type']}_{cfg.core_material}")
        out["interconnect_zip"] = ic["zip_path"]
    return out


def agent_eom_multiphysics(params: dict) -> dict:
    """Run local 2D-FEM RF EOM Electrode (`rf_electrode`), EO Wafer Orientation (`wafer_orientation`), Si PN Doping TCAD (`pn_doping`), or Steady+Transient Thermal FEM (`thermal_fem`) (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = str(params.get("mode") or "rf_electrode").lower().strip()

    if mode in ("wafer_orientation", "orientation", "wafer_cut"):
        q_raw = params.get("query_azimuth_deg", params.get("azimuth_deg"))
        q_deg = None if q_raw in (None, "", "auto") else float(q_raw)
        ovl_raw = params.get("eo_overlap_factor")
        ovl_val = None if ovl_raw in (None, "", "auto") else float(ovl_raw)
        h_raw = params.get("film_height_um", params.get("h_um"))
        s_raw = params.get("slab_height_um", params.get("slab_um"))
        w_raw = params.get("waveguide_width_um", params.get("w_um"))
        res = analyze_eo_wafer_orientation(
            core_material=str(params.get("core_material") or params.get("mat") or "LN_Xcut"),
            wafer_cut=str(params.get("wafer_cut") or "auto"),
            polarization=str(params.get("polarization") or params.get("pol") or "TE0"),
            electrode_field_axis=str(params.get("electrode_field_axis") or "auto"),
            wavelength_nm=float(params.get("wavelength_nm") or params.get("wl_nm") or 1550.0),
            electrode_gap_um=float(params.get("electrode_gap_um") or params.get("gap_um") or 5.0),
            eo_overlap_factor=ovl_val,
            query_azimuth_deg=q_deg,
            film_height_um=float(h_raw) if h_raw is not None else None,
            slab_height_um=float(s_raw) if s_raw is not None else None,
            waveguide_width_um=float(w_raw) if w_raw is not None else None,
            custom_r33_pm_v=float(params["custom_r33_pm_v"]) if params.get("custom_r33_pm_v") is not None else None,
            custom_r13_pm_v=float(params["custom_r13_pm_v"]) if params.get("custom_r13_pm_v") is not None else None,
            custom_r51_pm_v=float(params["custom_r51_pm_v"]) if params.get("custom_r51_pm_v") is not None else None,
            custom_r22_pm_v=float(params["custom_r22_pm_v"]) if params.get("custom_r22_pm_v") is not None else None,
            custom_eps_rf_perp=float(params["custom_eps_rf_perp"]) if params.get("custom_eps_rf_perp") is not None else None,
            custom_eps_rf_c=float(params["custom_eps_rf_c"]) if params.get("custom_eps_rf_c") is not None else None,
        )
        m = res["metrics"]
        return {
            "mode": "wafer_orientation",
            "mat": res["core_material"],
            "wafer_cut": res["wafer_cut"],
            "point_group": res["point_group"],
            "opt_azimuth_deg": m["optimal_azimuth_deg"],
            "r_eff_opt_pm_V": m["r_eff_optimal_pm_per_v"],
            "V_pi_L_opt_Vcm": m["V_pi_L_optimal_Vcm"],
            "active_azimuth_deg": m["active_azimuth_deg"],
            "r_eff_act_pm_V": m["r_eff_active_pm_per_v"],
            "ring_unsplit_pct": m["ring_unsplit_efficiency_pct"],
            "ring_split_pct": m["ring_split_pushpull_efficiency_pct"],
            "flat_label": res["crystal_axes"]["flat_label"],
            "ms": res["elapsed_ms"],
        }

    if mode in ("pn_doping", "pn_doping_tcad", "pn_tcad", "pn"):
        pn_fields = {f.name for f in fields(PNDopingTCADConfig)}
        aliases = {"wl_nm": "wavelength_nm", "pol": "polarization", "h_um": "film_height_um", "slab_um": "slab_height_um", "w_um": "waveguide_width_um", "l_um": "active_length_um"}
        kw = {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in pn_fields}
        res = simulate_pn_doping_tcad(PNDopingTCADConfig(**kw))
        m = res["metrics"]
        out = {
            "mode": "pn_doping_tcad",
            "junction": res["config"]["junction_type"],
            "bias_V": m["operating_bias_V"],
            "V_pi_L_Vcm": m["V_pi_L_Vcm"],
            "V_pi_V": m["V_pi_V"],
            "FCA_dB_cm": m["alpha_FCA_bias_dB_per_cm"],
            "VpiL_alpha_VdB": m["V_pi_L_alpha_V_dB"],
            "Cj_fF_um": m["Cj_bias_fF_per_um"],
            "Rs_ohm_cm": m["R_series_ohm_cm"],
            "f_RC_GHz": m["f_RC_GHz"],
            "f_TW_3dBo_GHz": m["f_TW_3dBo_GHz"],
            "ms": res["elapsed_ms"],
        }
        if params.get("export_gds"):
            out["gds"] = export_pn_doping_gds(res, AGENT_OUT_DIR / f"eom_pn_{res['config']['junction_type']}.gds")["gds_path"]
        if params.get("export_bundle"):
            out["bundle_zip"] = export_tcad_data_bundle(res, AGENT_OUT_DIR / "tcad_pn")["zip_path"]
        return out

    if mode in ("cointegrated_pn_heater", "pn_heater_ring", "dual_drive_ring"):
        gds_out = (AGENT_OUT_DIR / "mrr_pn_heater_cointegrated.gds") if params.get("export_gds") else None
        res = simulate_cointegrated_pn_heater_ring(
            ring_radius_um=float(params.get("ring_radius_um") or params.get("r_um") or 12.0),
            ring_width_um=float(params.get("w_um") or 0.45),
            gap_um=float(params.get("gap_um") or 0.20),
            pn_offset_nm=float(params.get("pn_offset_nm") or 20.0),
            pdk_key=str(params.get("pdk") or "gsip_soi220"),
            output_gds_path=gds_out,
        )
        m = res["metrics"]
        out = {
            "mode": "cointegrated_pn_heater_ring",
            "FSR_GHz": m["FSR_GHz"],
            "Q_L": m["Q_loaded"],
            "OMA": m["OMA_normalized"],
            "ER_dyn_dB": m["ER_dynamic_dB"],
            "f_3dB_EO_GHz": m["f_3dB_EO_GHz"],
            "P_pi_mW": m["P_pi_heater_mW"],
            "pn_heat_steal_pct": m["pn_contact_heat_stealing_pct"],
            "flux_ratio": m["thermal_flux_balance_ratio"],
            "ms": res["elapsed_ms"],
        }
        if "layout_verification" in res:
            lv = res["layout_verification"]
            out["gds"] = lv["gds_path"]
            out["drc_clean"] = lv["drc"]["drc_clean"]
            out["lvs_clean"] = lv["lvs"]["lvs_clean"]
        return out

    if mode in ("thermal", "thermal_fem", "heater"):
        th_fields = {f.name for f in fields(ThermalSimConfig)}
        aliases = {"mat": "core_material", "wl_nm": "wavelength_nm", "pol": "polarization", "h_um": "film_height_um", "slab_um": "slab_height_um", "w_um": "waveguide_width_um", "l_um": "active_length_um", "structure": "thermal_structure", "pn_sink": "include_pn_contact_sinking"}
        kw = {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in th_fields}
        res = simulate_thermal_phase_shifter(ThermalSimConfig(**kw))
        m = res["metrics"]
        out = {
            "mode": "thermal_fem",
            "structure": res["config"]["thermal_structure"],
            "mat": res["config"]["core_material"],
            "P_pi_mW": m["P_pi_mW"],
            "V_pi_V": m["V_pi_V"],
            "R_ohm": m["R_heater_ohm"],
            "f_3dB_kHz": m["f_3dB_thermal_kHz"],
            "t_rise_us": m["t_rise_10_90_us"],
            "t_fall_us": m["t_fall_90_10_us"],
            "Ppi_tau_mW_us": m["FOM_Ppi_tau_mW_us"],
            "xtalk_pct": m["crosstalk_at_neighbor_pct"],
            "pn_steal_pct": m["pn_contact_heat_stealing_pct"],
            "flux_ratio": m["flux_balance_ratio"],
            "ms": res["elapsed_ms"],
        }
        if params.get("export_gds"):
            out["gds"] = export_thermal_shifter_gds(res, AGENT_OUT_DIR / f"eom_thermal_{res['config']['thermal_structure']}.gds")["gds_path"]
        if params.get("export_bundle"):
            out["bundle_zip"] = export_tcad_data_bundle(res, AGENT_OUT_DIR / "tcad_thermal")["zip_path"]
        return out

    rf_fields = {f.name for f in fields(EOMElectrodeConfig)}
    aliases = {
        "topology": "electrode_topology",
        "mat": "core_material",
        "substrate": "substrate_type",
        "metal": "metal_material",
        "wl_nm": "wavelength_nm",
        "pol": "polarization",
        "h_um": "film_height_um",
        "slab_um": "slab_height_um",
        "w_um": "waveguide_width_um",
        "l_um": "active_length_um",
        "gap_um": "electrode_gap_um",
        "ws_um": "signal_width_um",
        "l_rail_um": "t_rail_length_um",
        "t_period_um": "t_rail_period_um",
        "t_cap_w_um": "t_rail_cap_width_um",
        "t_metal_um": "metal_thickness_um",
        "f_max_ghz": "max_freq_ghz",
        "azimuth_deg": "waveguide_azimuth_deg",
    }
    kw = {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in rf_fields}
    kw.setdefault("run_gap_sweep", False)
    res = design_eom_electrode(EOMElectrodeConfig(**kw))
    m = res["metrics"]
    d = res["dimensions"]
    wo_m = res["wafer_orientation"]["metrics"]
    out = {
        "mode": "rf_electrode",
        "topology": res["resolved_topology"],
        "mat": res["config"]["core_material"],
        "wafer_cut": res["wafer_orientation"]["wafer_cut"],
        "azimuth_deg": wo_m["active_azimuth_deg"],
        "WS_um": d["signal_width_um"],
        "G_um": d["inner_gap_um"],
        "Z0_ohm": m["Z0_ohm"],
        "n_RF": m["n_RF"],
        "n_g_opt": m["n_g_opt"],
        "f_3dBo_GHz": m["f_3dBo_GHz"],
        "f_3dBe_GHz": m["f_3dBe_GHz"],
        "bottleneck": m["limiting_mechanism"],
        "PAM4_Gbps": m["max_pam4_gbps"],
        "V_pi_L_Vcm": m["V_pi_L_Vcm"],
        "V_pi_V": m["V_pi_V"],
        "alpha_RF_40G_dB_cm": m["alpha_RF_40GHz_dB_per_cm"],
        "alpha_metal_opt_dB_cm": m["alpha_metal_opt_db_per_cm"],
        "ms": res["elapsed_ms"],
    }
    if res["resolved_topology"] == "t_rail_cpw" or float(res["config"].get("max_freq_ghz", 150.0)) >= 200.0:
        out["L_rail_um"] = d["t_rail_length_um"]
        out["S21_EO_300G_dBe"] = m.get("S21_EO_300GHz_dBe")
        out["alpha_RF_300G_dB_cm"] = m.get("alpha_RF_300GHz_dB_per_cm")
        out["f_Bragg_GHz"] = m.get("f_bragg_cutoff_GHz")
    if params.get("export_gds"):
        out["gds"] = export_eom_electrode_gds(res, AGENT_OUT_DIR / f"eom_rf_{res['resolved_topology']}_{res['config']['core_material']}.gds")["gds_path"]
    if params.get("export_bundle"):
        out["bundle_zip"] = export_eom_rf_interconnect_bundle(res, AGENT_OUT_DIR / f"eom_rf_{res['resolved_topology']}")["zip_path"]
    return out


def agent_pdk_drc(params: dict) -> dict:
    """Foundry PDK Polygon DRC (`.lyrdb`), 3D Union-Find LVS, Co-Integrated PN+Heater Ring GDS, or Native GDS Cutline 2D-FEM (<140 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = str(params.get("mode") or "build_pn_heater_ring").lower().strip()
    pdk_key = str(params.get("pdk") or params.get("pdk_key") or "gsip_soi220")
    pdk_obj: dict | str = load_pdk_from_csv_dir(params["pdk_csv_dir"]) if params.get("pdk_csv_dir") else pdk_key

    if mode in ("export_csv", "pdk_csv"):
        out_dir = Path(params.get("out_dir") or (AGENT_OUT_DIR / f"pdk_{pdk_key}_csv"))
        exp = export_pdk_csv_tables(pdk_obj, out_dir)
        return {
            "mode": "export_csv",
            "pdk": pdk_key,
            "dir": exp["output_dir"],
            "files": len(exp["files"]),
        }

    if mode in ("verify_gds", "drc_lvs", "drc", "lvs"):
        gds_in = str(params["gds_path"])
        cell_n = params.get("cell_name")
        drc = run_gds_polygon_drc(gds_in, pdk_obj, cell_name=cell_n)
        lvs = extract_3d_electrical_lvs(gds_in, pdk_obj, cell_name=cell_n)
        return {
            "mode": "verify_gds",
            "pdk": pdk_key,
            "cell": drc["cell_name"],
            "drc_clean": drc["drc_clean"],
            "drc_errs": drc["total_violations"],
            "lvs_clean": lvs["lvs_clean"],
            "nets": lvs["total_nets"],
            "floating_vias": lvs["floating_via_count"],
            "shorts": len(lvs["short_violations"]),
            "opens": len(lvs["open_violations"]),
            "lyrdb": drc["lyrdb_path"],
        }

    if mode in ("cutline_solve", "cutline", "layer_builder"):
        gds_in = str(params["gds_path"])
        cs = solve_gds_cutline_multiphysics(
            gds_in,
            pdk=pdk_obj,
            wavelength_nm=float(params.get("wl_nm") or params.get("wavelength_nm") or 1550.0),
            polarization=str(params.get("pol") or params.get("polarization") or "TE0"),
            run_pn_tcad=bool(params.get("run_pn_tcad", True)),
            run_thermal_fem=bool(params.get("run_thermal_fem", True)),
        )
        xc = cs["extracted_cross_section"]
        out = {
            "mode": "cutline_solve",
            "pdk": pdk_key,
            "w_core_um": xc["w_core_um"],
            "h_film_um": xc["h_film_um"],
            "h_slab_um": xc["h_slab_um"],
            "neff_2dfem": cs["eigenmode_2d_fem"]["n_eff"],
            "te_frac": cs["eigenmode_2d_fem"]["te_fraction"],
            "probe_pts": len(cs["index_probe_10pt"]),
            "ms": cs["elapsed_ms"],
        }
        if "pn_tcad_2d_fem" in cs:
            out["V_pi_L_Vcm"] = cs["pn_tcad_2d_fem"]["V_pi_L_Vcm"]
            out["f_RC_GHz"] = cs["pn_tcad_2d_fem"]["f_RC_GHz"]
        if "thermal_2d_fem" in cs:
            out["P_pi_mW"] = cs["thermal_2d_fem"]["P_pi_mW"]
            out["pn_steal_pct"] = cs["thermal_2d_fem"]["pn_contact_heat_stealing_pct"]
        return out

    r_um = float(params.get("r_um") or params.get("ring_radius_um") or 12.0)
    gds_target = Path(params["gds_path"]) if params.get("gds_path") else (AGENT_OUT_DIR / f"mrr_pn_heater_{pdk_key}_R{int(round(r_um))}.gds")
    bld = build_pn_heater_ring_modulator_gds(
        gds_target,
        pdk=pdk_obj,
        ring_radius_um=r_um,
        ring_width_um=float(params.get("w_ring_um") or params.get("ring_width_um") or 0.45),
        bus_width_um=float(params.get("w_bus_um") or params.get("bus_width_um") or 0.45),
        gap_um=float(params.get("gap_um") or 0.20),
        pn_offset_nm=float(params.get("pn_offset_nm") or 20.0),
        apply_slab_fillets=bool(params.get("slab_fillets", True)),
        verify_drc_lvs=True,
    )
    drc = bld["drc"]
    lvs = bld["lvs"]
    return {
        "mode": "build_pn_heater_ring",
        "pdk": pdk_key,
        "R_um": bld["ring_radius_um"],
        "drc_clean": drc["drc_clean"],
        "drc_errs": drc["total_violations"],
        "lvs_clean": lvs["lvs_clean"],
        "nets": lvs["total_nets"],
        "vias": lvs["total_vias_checked"],
        "gds": bld["gds_path"],
        "lyrdb": drc["lyrdb_path"],
    }


def agent_prism_optimize(params: dict) -> dict:
    """Run PRISM-style Physics Closure Gate (`closure_audit`) or Prediction-Gated Real-2D-FEM Optimizer (`optimize`) (<140 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = str(params.get("mode") or "optimize").lower().strip()
    if mode in ("closure_audit", "closure", "audit"):
        has_mrr = any(k in params for k in ("mat", "core_material", "r_um", "ring_radius_um", "top", "topology", "gap_um"))
        cfg = _cfg_from_dict(params) if has_mrr else None
        inc_pn = bool(params.get("include_pn", cfg is None or cfg.core_material == "Si"))
        pn_cfg = PNDopingTCADConfig() if inc_pn else None
        aud = audit_mrr_and_pn_physics_closure(cfg=cfg, pn_cfg=pn_cfg)
        qb = aud["q_budget"]
        out = {
            "mode": "closure_audit",
            "closure_pass": aud["closure_pass"],
            "Q_budget": qb["Q_loaded_from_budget"],
            "Q_fwhm": qb["Q_loaded_from_tmm_fwhm"],
            "q_err_pct": qb["q_closure_error_pct"],
            "passivity": aud["unitary_passivity"]["passivity_pass"],
            "ms": aud["elapsed_ms"],
        }
        if aud.get("perturbation_closure"):
            out["pert_err_pct"] = aud["perturbation_closure"]["perturbation_closure_error_pct"]
            out["core_share_pct"] = aud["perturbation_closure"]["core_confinement_share_pct"]
        return out

    prob = str(params.get("problem") or params.get("problem_type") or "pn_ring_modulator")
    res = run_prism_closed_loop_optimizer(
        problem_type=prob,
        max_iterations=int(params.get("max_iterations") or params.get("n_iter") or 8),
        seed=int(params.get("seed") or 42),
        output_dir=AGENT_OUT_DIR,
    )
    return {
        "mode": "prism_optimize",
        "problem": res["problem_type"],
        "iters": res["iterations_run"],
        "recal_gates": res["understanding_gate_recalibrations"],
        "init_fom": res["initial_design"]["display_fom"],
        "best_fom": res["best_design"]["display_fom"],
        "feasible": res["best_design"]["feasible"],
        "best_params": res["best_design"]["params"],
        "journal": res["journal_jsonl_path"],
        "ms": res["elapsed_ms"],
    }


def agent_wdm_link(params: dict) -> dict:
    """Simulate Multi-Channel Microring WDM Optical Link, NRZ/PAM4 Eye Diagram & Closed-Loop PI Thermal Locking (<140 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    wdm_fields = {f.name for f in fields(WDMLinkConfig)}
    aliases = {
        "n_ch": "num_channels",
        "channels": "num_channels",
        "rate_gbps": "data_rate_per_ch_gbps",
        "format": "modulation_format",
        "mod": "modulation_format",
        "spacing_ghz": "channel_spacing_ghz",
        "dt_k": "ambient_drift_k",
        "p_laser_dbm": "laser_power_per_ch_dbm",
        "r_um": "ring_radius_um",
        "w_ring_um": "ring_width_um",
        "w_bus_um": "bus_width_um",
        "wl_nm": "wavelength_nm",
        "vpp": "drive_vpp",
        "bias_v": "dc_bias_v",
    }
    kw = {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in wdm_fields}
    cfg = WDMLinkConfig(**kw)
    res = simulate_wdm_microring_link(cfg, export_csv_dir=AGENT_OUT_DIR)
    m = res["metrics"]
    return {
        "scen": cfg.scenario,
        "fmt": m["modulation_format"],
        "agg_Gbps": m["aggregate_bandwidth_Gbps"],
        "OMA_dBm": m["OMA_dBm"],
        "eye_h_uW": m["eye_height_uW"],
        "eye_w_UI": m["eye_width_UI"],
        "Q": m["Q_factor"],
        "BER": m["BER_estimate"],
        "xtalk_dB": m["crosstalk_penalty_dB"],
        "P_lock_mW": m["total_heater_lock_power_mW"],
        "pn_steal_pct": m["pn_contact_heat_stealing_pct"],
        "E_pJ_bit": m["E_total_pJ_per_bit"],
        "pass": m["link_open_pass"],
        "csv": res.get("eye_csv_path"),
        "ms": res["elapsed_ms"],
    }


def agent_nonlinear_lle(params: dict) -> dict:
    """First-principles Nonlinear Photonics & LLE Microcomb Engine (<150 tokens): dispersion, LLE, FWM-BS, SHG, bistability."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or params.get("kind") or "lle").lower().strip()
    aliases = {
        "mat": "core_material",
        "h_um": "film_height_um",
        "slab_um": "slab_height_um",
        "w_ring_um": "ring_width_um",
        "w_bus_um": "bus_width_um",
        "r_um": "ring_radius_um",
        "gap_um": "gap_thru_um",
        "wl_nm": "wavelength_nm",
        "pol": "polarization",
        "p_mw": "pump_power_mw",
    }

    def _kw_for(cls):
        valid = {f.name for f in fields(cls)}
        return {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in valid}

    if analysis == "dispersion":
        cfg = LLEConfig(**_kw_for(LLEConfig))
        res = compute_resonator_dispersion_and_nl(cfg)
        m = res["metrics"]
        return {
            "analysis": "dispersion",
            "mat": cfg.core_material,
            "FSR_GHz": m["FSR_GHz"],
            "D2_kHz": m["D2_2pi_kHz"],
            "regime": m["dispersion_regime"],
            "beta2_ps2_km": m["beta2_ps2_per_km"],
            "gamma_W_m": m["gamma_NL_per_W_m"],
            "A_eff_um2": m["A_eff_NL_um2"],
            "g0_Hz": m["g0_2pi_Hz"],
            "P_th_mW": m["P_th_OPO_mW"],
            "ZDW_nm": m["zero_dispersion_wls_nm"][:3],
            "ms": res["elapsed_ms"],
        }

    if analysis in ("fwm_bs", "fwm", "qfc"):
        cfg = FWMBSConfig(**_kw_for(FWMBSConfig))
        res = simulate_fwm_bragg_scattering(cfg)
        m = res["metrics"]
        return {
            "analysis": "fwm_bs",
            "mat": cfg.core_material,
            "wl_i_nm": res["wavelengths_nm"]["idler"],
            "gamma_BS": m["gamma_BS_per_W_m"],
            "C_BS": m["cooperativity_C_BS"],
            "eff_pm_pct": m["conversion_eff_phase_matched_pct"],
            "P_opt_mW": m["optimal_pump_geom_mean_mW"],
            "spur_dB": m["spurious_suppression_dB"],
            "ms": res["elapsed_ms"],
        }

    if analysis in ("shg", "chi2"):
        kw = _kw_for(SHGConfig)
        if "wl_nm" in params and "fundamental_wl_nm" not in kw:
            kw["fundamental_wl_nm"] = float(params["wl_nm"])
        cfg = SHGConfig(**kw)
        res = simulate_chi2_shg_resonator(cfg)
        m = res["metrics"]
        return {
            "analysis": "shg",
            "mat": cfg.core_material,
            "QPM_um": m["QPM_period_um"],
            "d_eff_pmV": m["d_eff_pm_per_V"],
            "g2_MHz": m["g2_2pi_MHz"],
            "eta_pct_W": m["eta_SHG_undepleted_pct_per_W"],
            "max_eff_pct": m["max_depleted_eff_pct"],
            "ms": res["elapsed_ms"],
        }

    if analysis in ("bistability", "shark_fin", "thermal_kerr"):
        kw = _kw_for(ThermalKerrBistabilityConfig)
        if "p_mw" in params and "input_power_mw" not in kw:
            kw["input_power_mw"] = float(params["p_mw"])
        cfg = ThermalKerrBistabilityConfig(**kw)
        res = simulate_thermal_kerr_bistability(cfg)
        m = res["metrics"]
        return {
            "analysis": "bistability",
            "bistable": m["is_bistable"],
            "P_bist_mW": m["bistability_threshold_mW"],
            "broaden_x": m["thermal_broadening_factor"],
            "hyst_GHz": m["scan_hysteresis_GHz"],
            "dT_pk_K": m["peak_delta_T_K"],
            "ms": res["elapsed_ms"],
        }

    # Default: Generalized LLE microcomb simulation
    cfg = LLEConfig(**_kw_for(LLEConfig))
    res = simulate_lle_microcomb(cfg)
    m = res["metrics"]
    dm = res["dispersion"]
    csv_path = AGENT_OUT_DIR / f"lle_comb_{cfg.core_material}_R{cfg.ring_radius_um:.1f}um.csv"
    sp = res["spectrum"]
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("mu,wavelength_nm,D_int_2pi_GHz,comb_power_dBm,intracavity_power_dBm\n")
        for mu_v, wl_v, di_v, pc_v, pi_v in zip(
            sp["mu"], sp["wavelength_nm"], sp["D_int_2pi_GHz"], sp["comb_power_dBm"], sp["intracavity_power_dBm"]
        ):
            fh.write(f"{mu_v},{wl_v},{di_v},{pc_v},{pi_v}\n")
    return {
        "analysis": "lle",
        "state": m["comb_state"],
        "backend": res["compute_backend"],
        "D2_kHz": dm["D2_2pi_kHz"],
        "P_th_mW": m["P_th_OPO_mW"],
        "pk_W": m["soliton_peak_power_W"],
        "fwhm_fs": m["pulse_fwhm_fs"],
        "bw3dB_THz": m["comb_3db_bandwidth_THz"],
        "eff_pct": m["comb_conversion_eff_pct"],
        "raman_THz": m["raman_shift_THz"],
        "csv": str(csv_path),
        "ms": res["elapsed_ms"],
    }


def agent_advanced_mrr(params: dict) -> dict:
    """Advanced MRR engines (<150 tokens): high_order_filter (CROW/SCISSOR), euler_bend, sensor_athermal,
    td_cmt_modulator, mzi_mrr, photonic_molecule, quantum_sfwm, or mrr_laser."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or params.get("kind") or "high_order_filter").lower().strip()
    aliases = {
        "mat": "core_material", "h_um": "film_height_um", "slab_um": "slab_height_um",
        "w_ring_um": "ring_width_um", "r_um": "ring_radius_um", "wl_nm": "wavelength_nm",
        "pol": "polarization", "p_mw": "pump_power_mw", "n": "order_n",
    }

    def _kw(cls):
        valid = {f.name for f in fields(cls)}
        return {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in valid}

    if analysis in ("high_order_filter", "crow", "scissor"):
        kw = _kw(HighOrderFilterConfig)
        if analysis in ("crow", "scissor") and "architecture" not in kw:
            kw["architecture"] = analysis
        res = simulate_high_order_mrr_filter(HighOrderFilterConfig(**kw))
        m = res["metrics"]
        return {"analysis": "high_order_filter", "arch": m["architecture"], "N": m["order_N"], "bw3dB_GHz": m["bw_3dB_GHz"], "sf": m["shape_factor_3_to_20dB"], "oob_dB": m["oob_rejection_dB"], "IL_dB": m["drop_IL_dB"], "tau_ps": m["center_group_delay_ps"], "kappas": m["synthesized_kappas_power"], "ms": res["elapsed_ms"]}

    if analysis in ("euler_bend", "euler"):
        kw = _kw(EulerBendConfig)
        if "r_um" in params and "effective_radius_um" not in kw:
            kw["effective_radius_um"] = float(params["r_um"])
        if params.get("export_gds") and not kw.get("export_gds_path"):
            kw["export_gds_path"] = str(AGENT_OUT_DIR / f"euler_racetrack_R{kw.get('effective_radius_um', 5.0):.1f}um.gds")
        res = design_euler_racetrack_bend(EulerBendConfig(**kw))
        m = res["metrics"]
        return {"analysis": "euler_bend", "R_eff": m["effective_radius_um"], "R_min": m["euler_min_radius_um"], "circ_junc_dB": m["circular_junction_mismatch_dB"], "euler_junc_dB": m["euler_junction_mismatch_dB"], "opt_off_nm": m["optimal_lateral_offset_nm"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}

    if analysis in ("sensor_athermal", "sensor", "athermal"):
        res = simulate_mrr_sensor_and_athermal(SensorAthermalConfig(**_kw(SensorAthermalConfig)))
        m = res["metrics"]
        return {"analysis": "sensor_athermal", "S_bulk": m["S_bulk_nm_per_RIU"], "S_surf": m["S_surface_pm_per_nm"], "FoM": m["intrinsic_FoM_per_RIU"], "LOD_RIU": m["LOD_bulk_RIU"], "L_eff_mm": m["effective_gas_path_mm"], "dlam_dT": m["bare_dlambda_dT_pm_per_K"], "h_ath_um": res["athermal"].get("athermal_cap_thickness_um"), "ms": res["elapsed_ms"]}

    if analysis in ("td_cmt_modulator", "td_cmt"):
        res = simulate_large_signal_td_cmt_modulator(LargeSignalTDCMTConfig(**_kw(LargeSignalTDCMTConfig)))
        m = res["metrics"]
        return {"analysis": "td_cmt_modulator", "mod": m["modulation_type"], "Gbps": m["bitrate_Gbps"], "ER_dB": m["dynamic_ER_dB"], "OMA": m["dynamic_OMA_linear"], "eye": m["eye_opening_linear"], "ovsh_pct": m["optical_peaking_overshoot_pct"], "chirp_GHz": m["peak_transient_chirp_GHz"], "alpha_H": m["effective_chirp_alpha_H"], "ms": res["elapsed_ms"]}

    if analysis in ("mzi_mrr", "tunable_kappa"):
        res = simulate_mzi_coupled_mrr(MZICoupledMRRConfig(**_kw(MZICoupledMRRConfig)))
        m = res["metrics"]
        return {"analysis": "mzi_mrr", "FSR_GHz": m["FSR_GHz"], "Q_int": m["Q_int"], "k_cc": m["kappa_critical_coupling"], "phi_cc_rad": m["mzi_phase_critical_coupling_rad"], "Q_range": [m["min_Q_loaded"], m["max_Q_loaded"]], "ms": res["elapsed_ms"]}

    if analysis in ("photonic_molecule", "molecule", "ep"):
        res = simulate_photonic_molecule(PhotonicMoleculeConfig(**_kw(PhotonicMoleculeConfig)))
        m = res["metrics"]
        return {"analysis": "photonic_molecule", "2J_GHz": m["avoided_crossing_2J_GHz"], "EP_4J_GHz": m["exact_EP_loss_contrast_4J_GHz"], "regime": m["pt_regime"], "EP_gain": m["EP_vs_DP_sensitivity_gain_at_1e-3J"], "ms": res["elapsed_ms"]}

    if analysis in ("quantum_sfwm", "sfwm", "quantum", "squeezing"):
        kw = _kw(QuantumSFWMConfig)
        if "wl_nm" in params and "pump_wavelength_nm" not in kw:
            kw["pump_wavelength_nm"] = float(params["wl_nm"])
        res = simulate_quantum_sfwm_mrr(QuantumSFWMConfig(**kw))
        m = res["metrics"]
        return {"analysis": "quantum_sfwm", "R_MHz": m["pair_rate_MHz"], "bright_MHz_mW2": m["brightness_MHz_per_mW2"], "CAR": m["CAR"], "g2_0": m["heralded_g2_0"], "purity": m["JSA_spectral_purity"], "sq_onchip_dB": m["max_squeezing_0p85Pth_onchip_dB"], "ms": res["elapsed_ms"]}

    kw = _kw(MRRLaserConfig)
    if "wl_nm" in params and "lasing_wavelength_nm" not in kw:
        kw["lasing_wavelength_nm"] = float(params["wl_nm"])
    res = simulate_mrr_laser(MRRLaserConfig(**kw))
    m = res["metrics"]
    return {"analysis": "mrr_laser", "medium": m["gain_medium"], "lasing": m["lasing_achieved"], "th": m["threshold_value"], "unit": m["threshold_unit"], "P_out_mW": m["operating_output_power_mW"], "lw_Hz": m["intrinsic_linewidth_Hz"], "SMSR_dB": m["SMSR_dB"], "f_RO_GHz": m["relaxation_osc_f_RO_GHz"], "ms": res["elapsed_ms"]}


def agent_swg_phc(params: dict) -> dict:
    """First-principles Subwavelength Grating (SWG) & Photonic Crystal Ring (PhCR) solver (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or params.get("kind") or "swg").lower().strip()
    aliases = {"mat": "core_material", "h_um": "film_height_um", "w_ring_um": "ring_width_um", "r_um": "ring_radius_um", "wl_nm": "wavelength_nm", "pol": "polarization"}

    def _kw(cls):
        valid = {f.name for f in fields(cls)}
        return {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in valid}

    if analysis in ("phc", "phcr", "photonic_crystal"):
        kw = _kw(PhCMicroringConfig)
        if params.get("export_gds") and not kw.get("export_gds_path"):
            kw["export_gds_path"] = str(AGENT_OUT_DIR / f"phcr_{kw.get('core_material', 'Si3N4')}_A{kw.get('modulation_amplitude_nm', 40):.0f}nm.gds")
        res = simulate_phc_microring(PhCMicroringConfig(**kw))
        m = res["metrics"]
        return {"analysis": "phcr", "m": m["target_azimuthal_order_m"], "period_nm": m["corrugation_period_nm"], "split_GHz": m["selective_mode_split_GHz"], "johnson_GHz": m["johnson_field_split_check_GHz"], "ng_slow": m["band_edge_slow_ng_at_1p15g"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}

    kw = _kw(SWGMicroringConfig)
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / f"swg_ring_P{kw.get('swg_period_nm', 220):.0f}nm.gds")
    res = simulate_swg_microring(SWGMicroringConfig(**kw))
    m = res["metrics"]
    return {"analysis": "swg", "neff_swg": m["neff_swg"], "ng_swg": m["ng_swg"], "bragg_edge_nm": m["bragg_band_edge_nm"], "subwl_ok": m["in_subwavelength_regime"], "FSR_GHz": m["FSR_GHz"], "S_bulk": m["S_bulk_nm_per_RIU"], "pillars": m["num_pillars_around_ring"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}


def agent_saw_sbs(params: dict) -> dict:
    """First-principles 2D Solid Mechanics / Piezoelectric SAW-Lamb Acousto-Optic & SBS Brillouin FEM (<150 tokens)."""
    analysis = str(params.get("analysis") or params.get("kind") or "saw").lower().strip()
    aliases = {"mat": "waveguide_material", "h_um": "film_height_um", "w_ring_um": "ring_width_um", "r_um": "ring_radius_um", "wl_nm": "wavelength_nm", "pol": "polarization"}

    def _kw(cls, extra_alias=None):
        al = dict(aliases)
        if extra_alias:
            al.update(extra_alias)
        valid = {f.name for f in fields(cls)}
        return {al.get(k, k): v for k, v in params.items() if v is not None and al.get(k, k) in valid}

    if analysis in ("sbs", "brillouin"):
        res = simulate_sbs_brillouin_mrr(SBSBrillouinConfig(**_kw(SBSBrillouinConfig, {"mat": "core_material"})))
        m = res["metrics"]
        return {"analysis": "sbs", "dir": m["sbs_direction"], "Omega_GHz": m["Omega_B_2pi_GHz"], "G_B": m["G_B_per_W_m"], "G_pe": m["G_B_photoelastic_only"], "G_mb": m["G_B_moving_boundary_only"], "P_th_mW": m["SBS_ring_threshold_mW"], "ms": res["elapsed_ms"]}

    res = simulate_saw_acousto_optic_mrr(SAWAcoustoOpticConfig(**_kw(SAWAcoustoOpticConfig)))
    m = res["metrics"]
    return {"analysis": "saw", "mode": m["acoustic_mode_type"], "v_open": m["v_open_m_per_s"], "f_SAW_GHz": m["f_SAW_GHz"], "K2_pct": m["K_eff_sq_pct"], "g0_kHz": m["optomechanical_g0_2pi_kHz"], "V_pi_V": m["V_pi_MRR_V"], "P_pi_mW": m["P_pi_MRR_mW"], "ms": res["elapsed_ms"]}


def agent_nanoantenna_tcmt(params: dict) -> dict:
    """First-principles Nanoantenna / Metasurface TCMT solver (<150 tokens): MIM absorber, Fano/EIT/qBIC, Mid-IR SEIRA & EP."""
    valid = {f.name for f in fields(NanoantennaTCMTConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    res = simulate_nanoantenna_tcmt(NanoantennaTCMTConfig(**kw))
    m = res["metrics"]
    return {k: v for k, v in m.items() if v is not None} | {"ms": res["elapsed_ms"]}


def agent_2d_material_mrr(params: dict) -> dict:
    """First-principles 27-Material 2D vdW on MRR solver (<150 tokens): Graphene/BLG, TMDs, Bulk vdW, PtSe2/PdSe2/ReS2, hBN/h10BN/a-MoO3/a-V2O5, BP/b-AsP/Te, MXenes, Bi2Se3/WTe2, NbOCl2/In2Se3/CIPS/GaSe."""
    if params.get("list_materials"):
        mats = list_2d_materials(family=params.get("family"), wavelength_nm=params.get("wl_nm") or params.get("wavelength_nm"))
        return {"count": len(mats), "materials": [f"{m['name']}({m['family']})" for m in mats]}
    aliases = {
        "mat_2d": "material_2d", "mat": "core_material", "wl_nm": "wavelength_nm",
        "r_um": "ring_radius_um", "h_um": "film_height_um", "w_ring_um": "ring_width_um",
        "angle_deg": "crystal_angle_deg", "t_2d_nm": "thickness_nm_override",
    }
    valid = {f.name for f in fields(TwoDMaterialMRRConfig)}
    kw = {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in valid}
    res = simulate_2d_material_mrr(TwoDMaterialMRRConfig(**kw))
    m = res["metrics"]
    return {
        "mat_2d": m["material_2d"],
        "fam": m["family"],
        "cap": m["capacitor_type"],
        "alpha_min": m["alpha_2d_min_dB_per_cm"],
        "alpha_max": m["alpha_2d_max_dB_per_cm"],
        "EA_dB_cm": m["EA_modulation_depth_dB_per_cm"],
        "shift_pm": m["max_resonance_shift_pm"],
        "dn_dk": m["phase_to_loss_ratio_dn_dk"],
        "MRR_ER_dB": m["MRR_dynamic_ER_dB"],
        "f_3dB_GHz": m["f_3dB_total_GHz"],
        "E_fJ": m["energy_per_bit_fJ"],
        "regime": m["dispersion_regime"],
        "ms": res["elapsed_ms"],
    }


def agent_photodetector(params: dict) -> dict:
    """First-principles Ge-on-Si Waveguide PD & O-E-O Photonic Neuron solver (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or "waveguide_pd").lower().strip()
    if analysis in ("oeo", "oeo_neuron", "neuron"):
        valid = {f.name for f in fields(OEONeuronConfig)}
        kw = {k: v for k, v in params.items() if v is not None and k in valid}
        res = simulate_oeo_neuron(OEONeuronConfig(**kw))
        m = res["metrics"]
        return {"analysis": "oeo_neuron", "shape": m["activation_shape"], "gain": m["max_differential_gain_dPout_dPin"], "cascadable": m["optical_cascadable"], "BW_GHz": m["neuron_bandwidth_GHz"], "E_fJ": m["energy_total_fJ_per_op"], "ms": res["elapsed_ms"]}
    valid = {f.name for f in fields(WaveguidePDConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / f"ge_si_pd_{kw.get('pd_architecture', 'vertical_pin')}.gds")
    res = simulate_waveguide_photodetector(WaveguidePDConfig(**kw))
    m = res["metrics"]
    return {"analysis": "waveguide_pd", "arch": m["pd_architecture"], "R_AW": m["responsivity_A_per_W"], "QE_pct": m["quantum_efficiency_pct"], "alpha_dB_um": m["effective_modal_alpha_dB_per_um"], "f_3dB_GHz": m["f_3dB_OE_GHz"], "I_dark_nA": m["dark_current_nA"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}


def agent_polarization_mdm(params: dict) -> dict:
    """First-principles PSR Polarization Rotator, PBS & MDM Mode Multiplexer solver (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or "psr").lower().strip()
    if analysis in ("mdm", "mdm_te1", "mdm_te2", "pbs", "pbs_dc"):
        valid = {f.name for f in fields(PBSMDMConfig)}
        kw = {k: v for k, v in params.items() if v is not None and k in valid}
        if analysis in ("mdm_te1", "mdm_te2", "pbs_dc") and "device_type" not in kw:
            kw["device_type"] = analysis
        elif analysis == "pbs" and "device_type" not in kw:
            kw["device_type"] = "pbs_dc"
        if params.get("export_gds") and not kw.get("export_gds_path"):
            kw["export_gds_path"] = str(AGENT_OUT_DIR / f"{kw.get('device_type', 'mdm_te1')}.gds")
        res = simulate_pbs_or_mdm(PBSMDMConfig(**kw))
        m = res["metrics"]
        return {"analysis": m["device_type"], "W_bus_um": m["phase_matched_w_bus_um"], "L_pi_um": m["coupling_length_L_pi_um"], "IL_dB": m["insertion_loss_dB"], "XT_dB": m["crosstalk_or_PER_dB"], "BW_1dB_nm": m["bw_1dB_nm"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}
    valid = {f.name for f in fields(PSRConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / "psr_rotator.gds")
    res = simulate_psr_rotator(PSRConfig(**kw))
    m = res["metrics"]
    return {"analysis": "psr", "W_hyb_um": m["hybridization_width_um"], "dn_hyb": m["anticrossing_gap_dn"], "conv_pct": m["conversion_efficiency_pct"], "PER_dB": m["PER_dB"], "L99_um": m["L_taper_99pct_um"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}


def agent_pcm_photonics(params: dict) -> dict:
    """First-principles Non-Volatile PCM (Sb2Se3/Sb2S3/GSST/GST/VO2) MRR & SET/RESET Thermal Pulse FEM (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or "pcm_mrr").lower().strip()
    if analysis in ("pcm_pulse", "pulse", "thermal_pulse", "set_reset"):
        valid = {f.name for f in fields(PCMThermalPulseConfig)}
        kw = {k: v for k, v in params.items() if v is not None and k in valid}
        res = simulate_pcm_thermal_pulse(PCMThermalPulseConfig(**kw))
        m = res["metrics"]
        return {"analysis": "pcm_pulse", "pcm": m["pcm_material"], "R_th": m["R_th_PCM_K_per_mW"], "RST_T_K": m["RESET_T_peak_K"], "quench_K_ns": m["RESET_quench_rate_K_per_ns"], "RST_ok": m["RESET_amorphization_pass"], "SET_T_K": m["SET_T_peak_K"], "SET_X": m["SET_crystallized_fraction"], "SET_ok": m["SET_crystallization_pass"], "ms": res["elapsed_ms"]}
    valid = {f.name for f in fields(PCMMRRConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / f"pcm_{kw.get('pcm_material', 'Sb2Se3')}_mrr.gds")
    res = simulate_pcm_mrr_switch(PCMMRRConfig(**kw))
    m = res["metrics"]
    return {"analysis": "pcm_mrr", "pcm": m["pcm_material"], "dn": m["delta_neff_a_to_c"], "FOM": m["modal_FOM_dn_over_dk"], "L_pi_um": m["L_pi_um"], "shift_nm": m["resonance_shift_a_to_c_nm"], "ER_dB": m["MRR_switching_ER_dB"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}


def agent_photonic_computing(params: dict) -> dict:
    """First-principles Photonic Neural Network Matrix Engine: MRR Weight Bank & Clements/Reck MZI Mesh (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or "mrr_weight_bank").lower().strip()
    if analysis in ("mzi_mesh", "clements", "reck", "unitary_mesh"):
        valid = {f.name for f in fields(MZIMeshConfig)}
        kw = {k: v for k, v in params.items() if v is not None and k in valid}
        if analysis in ("clements", "reck") and "architecture" not in kw:
            kw["architecture"] = analysis
        if params.get("export_gds") and not kw.get("export_gds_path"):
            kw["export_gds_path"] = str(AGENT_OUT_DIR / f"mzi_mesh_{kw.get('architecture', 'clements')}_N{kw.get('mesh_size_n', 6)}.gds")
        res = simulate_mzi_unitary_mesh(MZIMeshConfig(**kw))
        m = res["metrics"]
        return {"analysis": "mzi_mesh", "arch": m["architecture"], "N": m["mesh_size_N"], "MZIs": m["num_MZIs"], "fro_err": m["ideal_decomposition_fro_err"], "fidelity": m["mean_unitary_fidelity"], "IL_dB": m["mean_insertion_loss_dB"], "TOPS": m["throughput_TOPS"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}
    valid = {f.name for f in fields(MRRWeightBankConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    res = simulate_mrr_weight_bank(MRRWeightBankConfig(**kw))
    m = res["metrics"]
    return {"analysis": "mrr_weight_bank", "shape": m["matrix_shape"], "xt_pct": m["nearest_neighbor_crosstalk_pct"], "ENOB_raw": m["ENOB_bits_uncompensated"], "ENOB_comp": m["ENOB_bits_precompensated"], "TOPS_mm2": m["compute_density_TOPS_per_mm2"], "fJ_MAC": m["energy_efficiency_fJ_per_MAC"], "ms": res["elapsed_ms"]}


def agent_wbg_awg(params: dict) -> dict:
    """First-principles Waveguide Bragg Grating (WBG) & Arrayed Waveguide Grating (AWG) solver (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = str(params.get("analysis") or "wbg").lower().strip()
    if analysis in ("awg", "awg_demux"):
        valid = {f.name for f in fields(AWGConfig)}
        kw = {k: v for k, v in params.items() if v is not None and k in valid}
        if params.get("export_gds") and not kw.get("export_gds_path"):
            kw["export_gds_path"] = str(AGENT_OUT_DIR / f"awg_1x{kw.get('num_channels', 8)}.gds")
        res = simulate_awg_demux(AWGConfig(**kw))
        m = res["metrics"]
        return {"analysis": "awg", "ch": m["num_channels"], "m_ord": m["diffraction_order_m"], "R_FPR_um": m["FPR_focal_length_um"], "FSR_nm": m["FSR_nm"], "IL_dB": m["center_channel_IL_dB"], "XT_dB": m["adjacent_crosstalk_dB"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}
    valid = {f.name for f in fields(WaveguideBraggConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / f"wbg_{kw.get('grating_type', 'pi_phase_shifted')}.gds")
    res = simulate_waveguide_bragg_grating(WaveguideBraggConfig(**kw))
    m = res["metrics"]
    return {"analysis": "wbg", "type": m["grating_type"], "period_nm": m["bragg_period_nm"], "R_pct": m["peak_reflectivity_pct"], "SB_nm": m["stopband_3dB_nm"], "SLSR_dB": m["sidelobe_suppression_dB"], "pi_Q": m["pi_defect_Q"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}


def agent_pic_netlist(params: dict) -> dict:
    """Native License-Free Frequency-Domain Photonic IC S-Matrix Circuit Netlist Solver (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid = {f.name for f in fields(PICNetlistConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    if params.get("export_snp") and not kw.get("export_snp_path"):
        kw["export_snp_path"] = str(AGENT_OUT_DIR / f"pic_{kw.get('preset', 'ramzi_linearized')}.s4p")
    res = simulate_pic_circuit_netlist(PICNetlistConfig(**kw))
    m = res["metrics"]
    return {"circuit": m["circuit_name"], "comps": m["num_components"], "ports": m["num_external_ports"], "passive": m["passivity_verified"], "min_IL_dB": m["min_insertion_loss_dB"], "max_ER_dB": m["max_extinction_ratio_dB"], "tau_ps": m["peak_group_delay_ps"], "snp": m["snp_path"], "ms": res["elapsed_ms"]}


def agent_inverse_adjoint(params: dict) -> dict:
    """Native 2D-FDFD Exact Adjoint-Gradient Inverse Photonic Shape Optimizer (<150 tokens)."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid = {f.name for f in fields(AdjointInverseConfig)}
    kw = {k: v for k, v in params.items() if v is not None and k in valid}
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / "adjoint_inverse_y_branch.gds")
    res = optimize_inverse_photonic_device(AdjointInverseConfig(**kw))
    m = res["metrics"]
    return {"device": m["device_type"], "T_init_pct": m["initial_transmission_pct"], "IL_init_dB": m["initial_IL_dB"], "T_opt_pct": m["optimized_transmission_pct"], "IL_opt_dB": m["optimized_IL_dB"], "gain_dB": m["IL_improvement_dB"], "evals": m["adjoint_evaluations"], "gds": m["gds_path"], "ms": res["elapsed_ms"]}


def agent_mrr_ep(params: dict) -> dict:
    """First-principles Microring Resonator Exceptional Point (MRR-EP) solver (<150 tokens): Chiral CW/CCW EP2, Bus-Mirror Exceptional Surface (ES), Higher-Order PT EP2/EP3/EP4, Anti-PT Bus EP, Dynamic Encircling & Petermann K_P audit."""
    AGENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    aliases = {"n": "order_n", "mat": "core_material", "wl_nm": "wavelength_nm", "r_um": "ring_radius_um", "j_ghz": "coupling_j_ghz", "eps_ghz": "perturbation_ghz"}
    valid = {f.name for f in fields(MRRExceptionalPointConfig)}
    kw = {aliases.get(k, k): v for k, v in params.items() if v is not None and aliases.get(k, k) in valid}
    if params.get("export_gds") and not kw.get("export_gds_path"):
        kw["export_gds_path"] = str(AGENT_OUT_DIR / f"mrr_ep_{kw.get('regime', 'chiral_single_ring')}.gds")
    res = simulate_mrr_exceptional_point(MRRExceptionalPointConfig(**kw))
    m = res["metrics"]
    out = {"regime": m["regime"], "N": m["order_N"], "ms": res["elapsed_ms"]}
    for k in (
        "fitted_power_law_exponent", "EP_vs_DP_signal_gain", "Petermann_K_P_at_perturbation",
        "SNR_ratio_vs_DP_at_perturbation", "eigenvector_chirality", "reflection_asymmetry_dB",
        "fab_drift_immune_ES", "exact_AntiPT_EP_detuning_GHz", "chiral_switching_contrast",
        "synthesized_gap_EP_um", "gds_path",
    ):
        if k in m:
            out[k] = m[k]
    return out





