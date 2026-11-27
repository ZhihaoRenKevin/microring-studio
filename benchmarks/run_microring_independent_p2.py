"""Independent MicroRing Studio (v3.9.0) Phase-2 Validation Runner (Suites 6-11).

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs_p2.json`
and executes MicroRing Studio v3.9.0 completely independently across:
  - Suite 6A : Butt / Edge Couplers (2D Vector FEM + Complex Poynting Overlap + Alignment Tolerance)
  - Suite 6B : Gaussian-Apodized 26-Tooth Surface Grating Coupler (2D Full-Wave Maxwell FDFD)
  - Suite 7A : 2x2 Paired-Interference MMI Coupler (Multi-Mode 2D-FEM EME)
  - Suite 7B : Low-Q Full Microring Resonator (R = 5 um, gap = 120 nm, broadband spectrum)
  - Suite 7C : INTERCONNECT SPAR Export (2x2 DC MZI, dL = 80 um) & 4-Channel WDM Link Eye/BER
  - Suite 8  : Sub-THz EOM RF Electrodes (GSG CPW HR-Si / Quartz / Undercut-Si, Slow-Wave T-Rail)
               + 3D Crystal-Tensor Wafer Orientation
  - Suite 9  : 2D Nonlinear Poisson-Boltzmann Silicon PN Doping TCAD (Lateral, L-Shaped, U-Shaped)
  - Suite 10 : Advanced Microheaters (XeF2 Undercut, N++ Doped-Si Rib, Co-Integrated PN+TiN Sinking)
  - Suite 11 : Cleanroom Lithography Correlated CD & Gap Bias Drift (+/-20 nm, Pos/Neg PR)

Exports `benchmarks/outputs/microring_independent_results_p2.json` plus INTERCONNECT SPAR `.dat`
files into `benchmarks/outputs/_ic_export/` for closed-loop loading by Lumerical INTERCONNECT.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring import (
    EOMElectrodeConfig,
    MRRDesignConfig,
    MZIDesignConfig,
    PNDopingTCADConfig,
    ThermalSimConfig,
    WDMLinkConfig,
    analyze_eo_wafer_orientation,
    design_butt_edge_coupler,
    design_eom_electrode,
    design_grating_coupler,
    export_io_coupler_interconnect,
    export_mzi_interconnect_bundle,
    run_fabrication_error_analysis,
    simulate_mrr_topology,
    simulate_mzi,
    simulate_pn_doping_tcad,
    simulate_thermal_phase_shifter,
    simulate_wdm_microring_link,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"
OUT_P2_JSON = OUT_DIR / "microring_independent_results_p2.json"
IC_EXPORT_DIR = OUT_DIR / "_ic_export"


def run_suite6a_butt(specs: dict) -> list[dict]:
    out = []
    for c in specs["suite6a_butt_couplers"]:
        t0 = time.perf_counter()
        tip_rows = []
        nom_res = None
        for wt in c["w_tip_sweep_um"]:
            res = design_butt_edge_coupler(
                core_material=c["core_material"],
                clad_material=c["clad_material"],
                top_clad_material=c["top_clad_material"],
                wavelength_nm=c["wavelength_nm"],
                polarization=c["polarization"],
                film_height_um=c["film_height_um"],
                slab_height_um=c["slab_height_um"],
                sidewall_deg=c["sidewall_deg"],
                w_bus_um=c["w_bus_um"],
                w_tip_um=wt,
                sweep_tip_widths=False,
                taper_length_um=c["taper_length_um"],
                taper_profile=c["taper_profile"],
                fiber_preset=c["fiber_preset"],
                mfd_x_um=c["mfd_x_um"],
                mfd_y_um=c["mfd_y_um"],
                n_env=c["n_env"],
                ar_coating=c["ar_coating"],
                box_thickness_um=c["box_thickness_um"],
            )
            m = res["metrics"]
            trec = res["tip_sweep"][0]
            tip_rows.append({
                "w_tip_um": wt,
                "n_eff_tip": trec["n_eff_tip"],
                "mode_overlap_pct": trec["mode_overlap_pct"],
                "substrate_keep_pct": trec["substrate_keep_pct"],
                "facet_il_db": trec["facet_il_db"],
                "taper_il_db": m["taper_il_db"],
                "total_il_db": m["total_il_db"],
                "chip_mfd_x_um": trec["chip_mfd_x_um"],
                "chip_mfd_y_um": trec["chip_mfd_y_um"],
            })
            if abs(wt - c["w_tip_nominal_um"]) < 1e-6:
                nom_res = res
        m_nom = nom_res["metrics"]
        out.append({
            "case_id": c["case_id"],
            "w_tip_nominal_um": c["w_tip_nominal_um"],
            "n_eff_tip_nom": m_nom["n_eff_tip"],
            "n_eff_bus": m_nom["n_eff_bus"],
            "mode_overlap_nom_pct": [r for r in tip_rows if abs(r["w_tip_um"] - c["w_tip_nominal_um"]) < 1e-6][0]["mode_overlap_pct"],
            "facet_il_nom_db": m_nom["facet_il_db"],
            "taper_il_nom_db": m_nom["taper_il_db"],
            "total_il_nom_db": m_nom["total_il_db"],
            "tol_1db_x_um": m_nom["tol_1db_x_um"],
            "tol_1db_y_um": m_nom["tol_1db_y_um"],
            "tol_3db_x_um": m_nom["tol_3db_x_um"],
            "tol_3db_y_um": m_nom["tol_3db_y_um"],
            "tip_sweep": tip_rows,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        })
        print(
            f"[MR Suite 6A] {c['case_id']}: w_tip={c['w_tip_nominal_um']*1e3:.0f}nm "
            f"neff={m_nom['n_eff_tip']:.5f} overlap={out[-1]['mode_overlap_nom_pct']:.2f}% "
            f"facet_IL={m_nom['facet_il_db']:.3f} dB tol1dB=({m_nom['tol_1db_x_um']:.2f}, {m_nom['tol_1db_y_um']:.2f}) um"
        )
    return out


def run_suite6b_apodized_gc(specs: dict) -> dict:
    c = specs["suite6b_apodized_gc"]
    t0 = time.perf_counter()
    res = design_grating_coupler(
        core_material=c["core_material"],
        clad_material=c["clad_material"],
        top_clad_material=c["top_clad_material"],
        substrate_material=c["substrate_material"],
        wavelength_nm=c["wavelength_nm"],
        polarization=c["polarization"],
        film_height_um=c["film_height_um"],
        etch_depth_um=c["etch_depth_um"],
        fiber_angle_deg=c["fiber_angle_deg"],
        apodized=True,
        focusing=False,
        num_periods=c["num_periods"],
        grating_width_um=c["grating_width_um"],
        box_thickness_um=c["box_thickness_um"],
        top_clad_thickness_um=c["top_clad_thickness_um"],
        fiber_preset="SMF28_1550",
        mfd_um=c["mfd_um"],
        n_env=c["n_env"],
    )
    m = res["metrics"]
    sp = res["spectrum"]
    wl_arr = np.asarray(sp["wavelength_nm"], dtype=float)
    il_arr = np.asarray(sp["il_db"], dtype=float)
    s11_arr = np.asarray(sp["s11_db"], dtype=float)
    idx_1550 = int(np.argmin(np.abs(wl_arr - 1550.0)))

    # Also export INTERCONNECT SPAR for Suite 7C closed-loop check
    IC_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ic_gc = export_io_coupler_interconnect(res, IC_EXPORT_DIR / "gc_apodized")

    out = {
        "case_id": c["case_id"],
        "peak_wavelength_nm": m["peak_wavelength_nm"],
        "peak_il_db": m["peak_il_db"],
        "il_at_1550_db": round(float(il_arr[idx_1550]), 3),
        "s11_at_1550_db": round(float(s11_arr[idx_1550]), 2),
        "eta_up_at_1550_pct": m["upward_directionality_pct"],
        "eta_down_at_1550_pct": m["substrate_loss_pct"],
        "bw_1db_nm": m["bw_1db_nm"],
        "bw_3db_nm": m["bw_3db_nm"],
        "spectrum_wavelength_nm": [round(float(x), 3) for x in wl_arr],
        "spectrum_il_db": [round(float(x), 4) for x in il_arr],
        "interconnect_spar_path": str(Path(ic_gc["spar_path"]).relative_to(REPO_ROOT)),
        "interconnect_passivity_ok": bool(ic_gc["passivity_verified"]),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[MR Suite 6B] {c['case_id']}: peak_IL={out['peak_il_db']:.3f} dB @ {out['peak_wavelength_nm']:.1f} nm "
        f"| 1550nm IL={out['il_at_1550_db']:.3f} dB, up={out['eta_up_at_1550_pct']:.2f}%, S11={out['s11_at_1550_db']:.1f} dB"
    )
    return out


def run_suite7_mmi_ring_wdm(specs: dict) -> tuple[dict, dict, dict]:
    # 7A: 2x2 MMI
    c7a = specs["suite7a_mmi_2x2"]
    t0 = time.perf_counter()
    res_mmi = simulate_mzi(
        MZIDesignConfig(
            core_material=c7a["core_material"],
            clad_material=c7a["clad_material"],
            top_clad_material=c7a["top_clad_material"],
            wavelength_nm=c7a["wavelength_nm"],
            film_height_um=c7a["film_height_um"],
            slab_height_um=c7a["slab_height_um"],
            sidewall_deg=c7a["sidewall_deg"],
            arm1_width_um=c7a["arm_width_um"],
            splitter_type="mmi_2x2",
            mmi_width_um=c7a["mmi_width_um"],
            mmi_length_um=c7a["mmi_length_um"],
            mmi_taper_width_um=c7a["mmi_taper_width_um"],
            mmi_taper_length_um=c7a["mmi_taper_length_um"],
            span_nm=30.0,
        ),
        include_active_tuning=False,
    )
    sm = res_mmi["splitter"]["summary"]
    out7a = {
        "case_id": c7a["case_id"],
        "mmi_length_um": c7a["mmi_length_um"],
        "split_bar_pct": sm["split_bar_pct_at_wl0"],
        "split_cross_pct": sm["split_cross_pct_at_wl0"],
        "total_trans_pct": round(sm["split_bar_pct_at_wl0"] + sm["split_cross_pct_at_wl0"], 3),
        "excess_loss_db": sm["excess_loss_db"],
        "imbalance_db": sm["imbalance_at_wl0_db"],
        "phase_diff_deg": sm.get("phase_diff_deg_at_wl0"),
        "beat_length_lpi_um": sm.get("beat_length_lpi_um"),
        "num_guided_modes": sm.get("num_guided_modes"),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[MR Suite 7A] {c7a['case_id']}: Bar/Cross={out7a['split_bar_pct']:.2f}/{out7a['split_cross_pct']:.2f}% "
        f"IL={out7a['excess_loss_db']:.3f} dB Imb={out7a['imbalance_db']:+.3f} dB"
    )

    # 7B: Low-Q Full Microring (R = 5 um, gap = 120 nm)
    c7b = specs["suite7b_full_ring_lowq"]
    t0 = time.perf_counter()
    res_ring = simulate_mrr_topology(
        MRRDesignConfig(
            topology=c7b["topology"],
            core_material=c7b["core_material"],
            clad_material=c7b["clad_material"],
            top_clad_material=c7b["top_clad_material"],
            wavelength_nm=c7b["wavelength_nm"],
            film_height_um=c7b["film_height_um"],
            slab_height_um=c7b["slab_height_um"],
            sidewall_deg=c7b["sidewall_deg"],
            ring_width_um=c7b["ring_width_um"],
            bus_width_um=c7b["bus_width_um"],
            ring_radius_um=c7b["ring_radius_um"],
            gap_thru_um=c7b["gap_thru_um"],
            alpha_loss_db_per_cm=c7b["alpha_loss_db_per_cm"],
            span_nm=c7b["wl_max_nm"] - c7b["wl_min_nm"],
        )
    )
    rm = res_ring["metrics"]
    sp_r = res_ring["spectrum"]
    # Subsample cleanly to ~201 pts for lightweight JSON archiving
    wl_r = np.asarray(sp_r["wavelength_nm"], dtype=float)
    tr_r = np.asarray(sp_r["thru_T"], dtype=float)
    step_r = max(1, len(wl_r) // 250)
    out7b = {
        "case_id": c7b["case_id"],
        "neff_ring": rm["neff_ring"],
        "ng_ring": rm["ng_ring"],
        "FSR_nm": rm["FSR_nm"],
        "kappa_power_thru": rm["kappa_power_thru"],
        "Q_loaded": rm["Q_loaded"],
        "Q_int": rm["Q_int"],
        "FWHM_nm": round(rm["FWHM_pm"] * 1e-3, 5),
        "ER_dB": rm["ER_thru_dB"],
        "spectrum_wavelength_nm": [round(float(x), 4) for x in wl_r[::step_r]],
        "spectrum_thru_power": [round(float(x), 5) for x in tr_r[::step_r]],
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[MR Suite 7B] {c7b['case_id']}: FSR={out7b['FSR_nm']:.3f} nm, Q_L={out7b['Q_loaded']:.1f}, "
        f"kappa^2={out7b['kappa_power_thru']:.4f}, ER={out7b['ER_dB']:.2f} dB"
    )

    # 7C: Asymmetric 2x2 DC MZI INTERCONNECT SPAR Export + 4-Ch WDM Link Scenarios
    c7c = specs["suite7c_interconnect_circuit"]
    t0 = time.perf_counter()
    res_dc_mzi = simulate_mzi(
        MZIDesignConfig(
            core_material="Si",
            wavelength_nm=c7c["mzi_wavelength_nm"],
            film_height_um=0.22,
            slab_height_um=0.0,
            sidewall_deg=90.0,
            arm1_width_um=0.45,
            arm1_length_um=c7c["mzi_arm1_length_um"],
            delta_length_um=c7c["mzi_delta_length_um"],
            splitter_type="directional_coupler",
            dc_gap_um=0.20,
            span_nm=c7c["mzi_span_nm"],
        ),
        include_active_tuning=False,
    )
    ic_mzi = export_mzi_interconnect_bundle(res_dc_mzi, IC_EXPORT_DIR / "mzi_dc_dl80")
    mm = res_dc_mzi["metrics"]
    ps = res_dc_mzi["passive_spectrum"]
    wl_m = np.asarray(ps["wavelength_nm"], dtype=float)
    bar_t = np.asarray(ps["bar_T"], dtype=float)
    crs_t = np.asarray(ps["cross_T"], dtype=float)
    idx_1550 = int(np.argmin(np.abs(wl_m - 1550.0)))

    wdm_rows = []
    for scen in ("nominal", "thermal_drift", "heater_relocked", "dense_crosstalk"):
        wres = simulate_wdm_microring_link(
            WDMLinkConfig(
                num_channels=c7c["wdm_num_channels"],
                data_rate_per_ch_gbps=c7c["wdm_data_rate_gbps"],
                modulation_format="nrz",
                channel_spacing_ghz=c7c["wdm_channel_spacing_ghz"],
                scenario=scen,
                ring_radius_um=c7c["wdm_ring_radius_um"],
                gap_um=c7c["wdm_gap_um"],
            )
        )
        wm = wres["metrics"]
        wdm_rows.append({
            "scenario": scen,
            "Q_factor": wm["Q_factor"],
            "BER": wm["BER_estimate"],
            "OMA_dBm": wm["OMA_dBm"],
            "eye_height_uW": wm["eye_height_uW"],
            "eye_width_ui": wm["eye_width_UI"],
            "heater_power_per_ch_mW": wm["heater_lock_power_per_ch_mW"],
            "energy_pj_per_bit": wm["E_total_pJ_per_bit"],
        })
    out7c = {
        "case_id": c7c["case_id"],
        "mzi_FSR_nm": mm["FSR_nm"],
        "mzi_ER_bar_dB": mm["ER_bar_passive_dB"],
        "mzi_bar_T_at_1550": round(float(bar_t[idx_1550]), 5),
        "mzi_cross_T_at_1550": round(float(crs_t[idx_1550]), 5),
        "mzi_bar_T_max": round(float(np.max(bar_t)), 5),
        "mzi_bar_T_min": round(float(np.min(bar_t)), 6),
        "mzi_spar_path": str(Path(ic_mzi["spar_path"]).relative_to(REPO_ROOT)),
        "mzi_passivity_verified": bool(ic_mzi["passivity_verified"]),
        "wdm_scenarios": wdm_rows,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[MR Suite 7C] MZI FSR={out7c['mzi_FSR_nm']:.3f} nm ER={out7c['mzi_ER_bar_dB']:.1f} dB "
        f"| WDM nominal Q={wdm_rows[0]['Q_factor']:.2f} drift Q={wdm_rows[1]['Q_factor']:.2f} relocked Q={wdm_rows[2]['Q_factor']:.2f}"
    )
    return out7a, out7b, out7c


def run_suite8_rf_electrodes(specs: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    for c in specs["suite8_rf_electrodes"]:
        t0 = time.perf_counter()
        cfg = EOMElectrodeConfig(
            electrode_topology=c["electrode_topology"],
            core_material=c["core_material"],
            substrate_type=c["substrate_type"],
            metal_material=c["metal_material"],
            wavelength_nm=c["wavelength_nm"],
            film_height_um=c["film_height_um"],
            slab_height_um=c["slab_height_um"],
            sidewall_deg=c["sidewall_deg"],
            waveguide_width_um=c["waveguide_width_um"],
            box_thickness_um=c["box_thickness_um"],
            top_clad_thickness_um=c["top_clad_thickness_um"],
            active_length_um=c["active_length_um"],
            electrode_gap_um=c["electrode_gap_um"],
            signal_width_um=c["signal_width_um"],
            ground_width_um=c["ground_width_um"],
            metal_thickness_um=c["metal_thickness_um"],
            t_rail_length_um=c["t_rail_length_um"] if c["t_rail_length_um"] > 0 else None,
            t_rail_period_um=c["t_rail_period_um"],
            t_rail_duty=c["t_rail_duty"],
            t_rail_cap_width_um=c["t_rail_cap_width_um"] if c["t_rail_cap_width_um"] > 0 else None,
            t_rail_stem_width_um=c["t_rail_stem_width_um"] if c["t_rail_stem_width_um"] > 0 else 4.0,
            metal_rms_roughness_nm=0.0,
            max_freq_ghz=320.0,
            run_gap_sweep=False,
        )
        res = design_eom_electrode(cfg)
        m = res["metrics"]
        rf_sw = res["frequency_response"]
        fg = np.asarray(rf_sw["freq_GHz"], dtype=float)
        al = np.asarray(rf_sw["alpha_RF_dB_per_cm"], dtype=float)
        nrf_f = np.asarray(rf_sw["n_RF"], dtype=float)
        z0_f = np.asarray(rf_sw["Z0_ohm"], dtype=float)
        freq_pts = {}
        for fq in c["eval_freqs_ghz"]:
            freq_pts[f"{int(fq)}GHz"] = {
                "Z0_ohm": round(float(np.interp(fq, fg, z0_f)), 3),
                "n_RF": round(float(np.interp(fq, fg, nrf_f)), 4),
                "alpha_RF_dB_per_cm": round(float(np.interp(fq, fg, al)), 3),
            }
        rows.append({
            "case_id": c["case_id"],
            "electrode_topology": c["electrode_topology"],
            "substrate_type": c["substrate_type"],
            "Z0_ohm": m["Z0_ohm"],
            "n_RF": m["n_RF"],
            "n_g_opt": m["n_g_opt"],
            "C_diel_pF_per_m": round(float(m["C_line_pF_per_cm"]) * 100.0, 3),
            "L_ext_nH_per_m": round(float(m["L_line_nH_per_cm"]) * 100.0, 3),
            "V_pi_L_Vcm": m["V_pi_L_Vcm"],
            "f_3dBe_GHz": m["f_3dBe_GHz"],
            "f_bragg_cutoff_GHz": m.get("f_bragg_cutoff_GHz"),
            "alpha_RF_40GHz_dB_per_cm": m["alpha_RF_40GHz_dB_per_cm"],
            "alpha_RF_300GHz_dB_per_cm": m.get("alpha_RF_300GHz_dB_per_cm"),
            "freq_samples": freq_pts,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        })
        print(
            f"[MR Suite 8] {c['case_id']}: Z0={m['Z0_ohm']:.2f} Ohm, n_RF={m['n_RF']:.4f} (n_g={m['n_g_opt']:.4f}), "
            f"a_40G={m['alpha_RF_40GHz_dB_per_cm']:.2f} dB/cm, VpiL={m['V_pi_L_Vcm']:.3f} V*cm, f_3dBe={m['f_3dBe_GHz']:.1f} GHz"
        )

    # 3D Crystal-Tensor Wafer Orientation Check
    orient_rows = []
    for mat, wcut, pol in (
        ("LN_Xcut", "x_cut_ln", "TE0"),
        ("BTO", "a_cut_bto_si", "TE0"),
        ("ScAlN_20", "c_plane_6mm", "TM0"),
        ("3C-SiC", "cubic_001_43m", "TE0"),
    ):
        wo = analyze_eo_wafer_orientation(core_material=mat, wafer_cut=wcut, polarization=pol, wavelength_nm=1550.0)
        wm = wo["metrics"]
        angs = np.asarray(wo["angular_sweep"]["azimuth_deg"], dtype=float)
        reffs = np.asarray(wo["angular_sweep"]["r_eff_pm_per_v"], dtype=float)
        orient_rows.append({
            "core_material": mat,
            "wafer_cut": wcut,
            "polarization": pol,
            "optimal_azimuth_deg": wm["optimal_azimuth_deg"],
            "r_eff_optimal_pm_per_v": wm["r_eff_optimal_pm_per_v"],
            "r_eff_0deg": round(float(np.interp(0.0, angs, reffs)), 4),
            "r_eff_45deg": round(float(np.interp(45.0, angs, reffs)), 4),
            "r_eff_90deg": round(float(np.interp(90.0, angs, reffs)), 4),
            "ring_unsplit_eff_pct": wm["ring_unsplit_efficiency_pct"],
            "ring_split_eff_pct": wm["ring_split_pushpull_efficiency_pct"],
        })
    return rows, orient_rows


def run_suite9_pn_tcad(specs: dict) -> list[dict]:
    rows = []
    for c in specs["suite9_si_pn_tcad"]:
        t0 = time.perf_counter()
        res = simulate_pn_doping_tcad(
            PNDopingTCADConfig(
                junction_type=c["junction_type"],
                wavelength_nm=c["wavelength_nm"],
                polarization=c["polarization"],
                film_height_um=c["film_height_um"],
                slab_height_um=c["slab_height_um"],
                sidewall_deg=c["sidewall_deg"],
                waveguide_width_um=c["waveguide_width_um"],
                active_length_um=c["active_length_um"],
                drive_scheme=c["drive_scheme"],
                na_core_cm3=c["na_core_cm3"],
                nd_core_cm3=c["nd_core_cm3"],
                na_inter_cm3=c["na_inter_cm3"],
                nd_inter_cm3=c["nd_inter_cm3"],
                n_heavy_contact_cm3=c["n_heavy_contact_cm3"],
                pn_offset_nm=c["pn_offset_nm"],
                l_vertical_y_nm=c["l_vertical_y_nm"],
                implant_straggle_nm=c["implant_straggle_nm"],
                inter_implant_offset_um=c["inter_implant_offset_um"],
                heavy_contact_offset_um=c["heavy_contact_offset_um"],
                bias_voltages_v=c["bias_voltages_v"],
                map_bias_v=c["map_bias_v"],
            )
        )
        m = res["metrics"]
        rows.append({
            "case_id": c["case_id"],
            "junction_type": c["junction_type"],
            "neff_intrinsic": m["neff_intrinsic"],
            "V_pi_L_at_m3V_Vcm": m["V_pi_L_Vcm"],
            "alpha_FCA_0V_dB_per_cm": m["alpha_FCA_0V_dB_per_cm"],
            "alpha_FCA_m3V_dB_per_cm": m["alpha_FCA_bias_dB_per_cm"],
            "Cj_0V_fF_per_um": m["Cj_0V_fF_per_um"],
            "Cj_m3V_fF_per_um": m["Cj_bias_fF_per_um"],
            "R_series_ohm_cm": m["R_series_ohm_cm"],
            "f_RC_intrinsic_GHz": m["f_RC_intrinsic_GHz"],
            "states": res["states"],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        })
        print(
            f"[MR Suite 9] {c['case_id']}: VpiL(-3V)={m['V_pi_L_Vcm']:.3f} V*cm, "
            f"Cj(0/-3V)={m['Cj_0V_fF_per_um']:.3f}/{m['Cj_bias_fF_per_um']:.3f} fF/um, "
            f"Rs={m['R_series_ohm_cm']:.3f} Ohm*cm, FCA(-3V)={m['alpha_FCA_bias_dB_per_cm']:.2f} dB/cm"
        )
    return rows


def run_suite10_thermal_adv(specs: dict) -> list[dict]:
    rows = []
    for c in specs["suite10_thermal_adv"]:
        t0 = time.perf_counter()
        res = simulate_thermal_phase_shifter(
            ThermalSimConfig(
                thermal_structure=c["thermal_structure"],
                core_material=c["core_material"],
                wavelength_nm=c["wavelength_nm"],
                film_height_um=c["film_height_um"],
                slab_height_um=c["slab_height_um"],
                sidewall_deg=c["sidewall_deg"],
                waveguide_width_um=c["waveguide_width_um"],
                active_length_um=c["active_length_um"],
                box_thickness_um=c["box_thickness_um"],
                top_clad_thickness_um=c["top_clad_thickness_um"],
                heater_material=c["heater_material"],
                heater_width_um=c["heater_width_um"],
                heater_thickness_um=c["heater_thickness_um"],
                heater_gap_um=c["heater_gap_um"],
                trench_edge_offset_um=c.get("trench_edge_offset_um", 2.80),
                trench_width_um=c.get("trench_width_um", 6.00),
                trench_depth_um=c.get("trench_depth_um", 42.0),
                undercut_width_um=c.get("undercut_width_um", 16.0),
                undercut_depth_um=c.get("undercut_depth_um", 8.0),
                neighbor_pitch_um=c["neighbor_pitch_um"],
                include_pn_contact_sinking=c.get("include_pn_contact_sinking", False),
                pn_contact_clearance_um=c.get("pn_contact_clearance_um", 0.85),
                pn_via_width_um=c.get("pn_via_width_um", 0.45),
                pn_metal_width_um=c.get("pn_metal_width_um", 9.00),
                pn_metal_thickness_um=c.get("pn_metal_thickness_um", 0.45),
                pn_via_longitudinal_fill=c.get("pn_via_longitudinal_fill", 0.40),
            )
        )
        m = res["metrics"]
        rows.append({
            "case_id": c["case_id"],
            "thermal_structure": c["thermal_structure"],
            "R_th_core_K_per_mW": m["R_th_core_K_per_mW"],
            "R_th_bare_K_per_mW": m.get("R_th_bare_without_pn_sinking_K_per_mW"),
            "pn_sinking_penalty_pct": m.get("pn_contact_heat_stealing_penalty_pct"),
            "P_pi_mW": m["P_pi_mW"],
            "t_rise_10_90_us": m["t_rise_10_90_us"],
            "f_3dB_thermal_kHz": m["f_3dB_thermal_kHz"],
            "crosstalk_20um_pct": m["crosstalk_at_neighbor_pct"],
            "power_conservation_err_pct": m["power_conservation_error_pct"],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        })
        print(
            f"[MR Suite 10] {c['case_id']}: R_th={m['R_th_core_K_per_mW']:.4f} K/mW, "
            f"P_pi={m['P_pi_mW']:.3f} mW, t_rise={m['t_rise_10_90_us']:.2f} us, "
            f"xtalk={m['crosstalk_at_neighbor_pct']:.3f}%, flux_err={m['power_conservation_error_pct']:.4f}%"
        )
    return rows


def run_suite11_fab_bias(specs: dict) -> dict:
    c = specs["suite11_fab_bias"]
    t0 = time.perf_counter()
    eta_prox = c["proximity_gap_factor"]
    corners = []
    neff_nom = None
    ng_nom = None
    for dcd in c["delta_cd_nm_list"]:
        w_fab = c["ring_width_nom_um"] + dcd * 1e-3
        g_fab = c["gap_nom_um"] - eta_prox * dcd * 1e-3
        res = simulate_mrr_topology(
            MRRDesignConfig(
                topology="point_circular_ap",
                core_material=c["core_material"],
                clad_material=c["clad_material"],
                top_clad_material=c["top_clad_material"],
                wavelength_nm=c["wavelength_nm"],
                film_height_um=c["film_height_um"],
                slab_height_um=c["slab_height_um"],
                sidewall_deg=c["sidewall_deg"],
                ring_width_um=w_fab,
                bus_width_um=w_fab,
                ring_radius_um=c["ring_radius_um"],
                gap_thru_um=g_fab,
                alpha_loss_db_per_cm=2.0,
            )
        )
        m = res["metrics"]
        if abs(dcd) < 1e-9:
            neff_nom = float(m["neff_ring"])
            ng_nom = float(m["ng_ring"])
        corners.append({
            "delta_cd_nm": dcd,
            "w_fab_nm": round(w_fab * 1e3, 2),
            "gap_fab_nm": round(g_fab * 1e3, 2),
            "neff_bent": m["neff_ring"],
            "ng_bent": m["ng_ring"],
            "kappa_power_cmt": m["kappa_power_thru"],
            "Q_loaded": m["Q_loaded"],
            "ER_dB": m["ER_thru_dB"],
        })
    for row in corners:
        dneff = row["neff_bent"] - neff_nom
        dlam_nm = c["wavelength_nm"] * dneff / ng_nom
        row["delta_lambda_nm"] = round(dlam_nm, 4)
    # Exact central slope dlambda/dCD [pm/nm]
    c_m = [r for r in corners if r["delta_cd_nm"] < 0][0]
    c_p = [r for r in corners if r["delta_cd_nm"] > 0][0]
    slope_pm_per_nm = (c_p["delta_lambda_nm"] - c_m["delta_lambda_nm"]) * 1e3 / (c_p["delta_cd_nm"] - c_m["delta_cd_nm"])

    # Also verify positive vs negative PR sign inversion via run_fabrication_error_analysis
    base_cfg = MRRDesignConfig(
        topology="point_circular_ap",
        core_material=c["core_material"],
        wavelength_nm=c["wavelength_nm"],
        film_height_um=c["film_height_um"],
        sidewall_deg=c["sidewall_deg"],
        ring_width_um=c["ring_width_nom_um"],
        bus_width_um=c["bus_width_nom_um"],
        ring_radius_um=c["ring_radius_um"],
        gap_thru_um=c["gap_nom_um"],
        alpha_loss_db_per_cm=2.0,
    )
    pos_pr = run_fabrication_error_analysis(base_cfg, resist_tone="positive_pr", max_dose_error_pct=10.0, proximity_gap_factor=eta_prox, num_points=3)
    neg_pr = run_fabrication_error_analysis(base_cfg, resist_tone="negative_pr", max_dose_error_pct=10.0, proximity_gap_factor=eta_prox, num_points=3)

    out = {
        "case_id": c["case_id"],
        "dlambda_dCD_pm_per_nm": round(slope_pm_per_nm, 3),
        "positive_pr_overdose_dlam_nm": pos_pr["corners"][-1]["delta_wl_res_nm"],
        "negative_pr_overdose_dlam_nm": neg_pr["corners"][-1]["delta_wl_res_nm"],
        "corners": corners,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[MR Suite 11] {c['case_id']}: dlam/dCD={out['dlambda_dCD_pm_per_nm']:.2f} pm/nm | "
        f"pos_PR(+10% dose)={out['positive_pr_overdose_dlam_nm']:+.3f} nm, neg_PR(+10% dose)={out['negative_pr_overdose_dlam_nm']:+.3f} nm"
    )
    return out


def main() -> None:
    t_all = time.perf_counter()
    specs = json.loads(SPECS_P2_JSON.read_text(encoding="utf-8"))
    s6a = run_suite6a_butt(specs)
    s6b = run_suite6b_apodized_gc(specs)
    s7a, s7b, s7c = run_suite7_mmi_ring_wdm(specs)
    s8_rf, s8_orient = run_suite8_rf_electrodes(specs)
    s9 = run_suite9_pn_tcad(specs)
    s10 = run_suite10_thermal_adv(specs)
    s11 = run_suite11_fab_bias(specs)

    payload = {
        "solver": "MicroRing Studio v3.9.0 (Phase-2 Complete Physical Engine Suite)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite6a_butt_couplers": s6a,
        "suite6b_apodized_gc": s6b,
        "suite7a_mmi_2x2": s7a,
        "suite7b_full_ring_lowq": s7b,
        "suite7c_interconnect_circuit": s7c,
        "suite8_rf_electrodes": s8_rf,
        "suite8_wafer_orientation": s8_orient,
        "suite9_si_pn_tcad": s9,
        "suite10_thermal_adv": s10,
        "suite11_fab_bias": s11,
    }
    OUT_P2_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved MicroRing Phase-2 results to {OUT_P2_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
