"""Neutral Physical Specification Generator for Phase-2 Commercial Solver Validation (Suites 6-11).

Extends Phase 1 (Suites 1-5) so that 100% of MicroRing Studio v3.9.0's physical solver engines
are independently cross-checked against Ansys Lumerical (MODE / EME / 2D & 3D FDTD GPU / CHARGE /
HEAT / INTERCONNECT) and COMSOL Multiphysics 6.4 (Wave Optics ewfd / RF emw / AC-DC es / Heat ht /
Nonlinear PDE Poisson-Boltzmann):
  - Suite 6  : Chip I/O Couplers (6A Butt/Edge Coupler to Lensed Fiber & III-V DFB Laser +
               6B Gaussian-Apodized 26-tooth Surface Grating Coupler).
  - Suite 7  : 2x2 Paired-Interference MMI (3D FDTD GPU), Low-Q Full-Ring 3D FDTD Spectrum +
               Blind CMT Extraction, and Lumerical INTERCONNECT Circuit / WDM Link Verification.
  - Suite 8  : Sub-THz (1-300 GHz) EOM RF Traveling-Wave GSG CPW & Slow-Wave T-Rail Electrodes
               (HR-Si, Quartz, Undercut-Si, 2D + 3D Periodic Unit Cell) & 3D Wafer Orientation.
  - Suite 9  : 2D Silicon PN Carrier-Depletion Semiconductor TCAD (Lateral, L-Shaped, U-Shaped).
  - Suite 10 : Advanced Microheaters (XeF2 Undercut, N++ Doped-Si Rib, Co-Integrated PN+TiN
               Contact Heat-Sinking) + 3D Finite-Length Thermal FEM + Lumerical HEAT 3-Way Check.
  - Suite 11 : Cleanroom Lithography Photoresist Tone & Correlated CD/Gap Bias Drift (+/-20 nm).

ZERO solver outputs are stored in `commercial_validation_specs_p2.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring.active_tuning import (
    HEATER_METAL_CATALOG,
    STATIC_PERMITTIVITY_DC,
    THERMAL_CONDUCTIVITY_W_MK,
    VOLUMETRIC_HEAT_CAPACITY_J_M3K,
)
from microring.eom_electrode import (
    DIELECTRIC_LOSS_TANGENT_RF,
    ELECTRODE_METAL_CATALOG,
    RF_SUBSTRATE_CATALOG,
)
from microring.io_couplers import synthesize_grating_teeth
from microring.materials import T_REF_K, get_material_fn

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"


def _eval_mat_triplet(mat_name: str, wl0_nm: float, dwl_nm: float) -> dict:
    fn = get_material_fn(mat_name)
    out = {}
    for key, wl_nm in (("minus", wl0_nm - dwl_nm), ("center", wl0_nm), ("plus", wl0_nm + dwl_nm)):
        m = fn(wl_nm * 1e-9, T_REF_K)
        out[key] = {
            "wavelength_nm": round(wl_nm, 4),
            "nx": round(float(m.nx), 8),
            "ny": round(float(m.ny), 8),
            "nz": round(float(m.nz), 8),
            "kx": round(float(m.kx), 8),
            "ky": round(float(m.ky), 8),
            "kz": round(float(m.kz), 8),
        }
    m0 = fn(wl0_nm * 1e-9, T_REF_K)
    out["dn_dt_per_k"] = float(m0.dn_dt_per_k)
    out["r33_pm_per_v"] = float(m0.r33_pm_per_v)
    out["r13_pm_per_v"] = float(m0.r13_pm_per_v)
    out["source"] = str(m0.source)
    return out


def build_specs_p2() -> dict:
    si_1550 = _eval_mat_triplet("Si", 1550.0, 12.0)
    sio2_1550 = _eval_mat_triplet("SiO2", 1550.0, 12.0)
    sin_1310 = _eval_mat_triplet("Si3N4", 1310.0, 10.0)
    sio2_1310 = _eval_mat_triplet("SiO2", 1310.0, 10.0)
    ln_xcut_1550 = _eval_mat_triplet("LN_Xcut", 1550.0, 10.0)

    # ==========================================================================
    # Suite 6A: Butt / Edge Couplers (2D Vector FEM + Poynting Overlap + EME)
    # ==========================================================================
    suite6a_butt_couplers = [
        {
            "case_id": "SOI_220nm_Butt_Lensed2p5um",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "polarization": "TE0",
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "w_bus_um": 0.48,
            "w_tip_sweep_um": [0.12, 0.16, 0.20, 0.24, 0.30],
            "w_tip_nominal_um": 0.20,
            "taper_length_um": 160.0,
            "taper_profile": "adiabatic",
            "box_thickness_um": 3.0,
            "fiber_preset": "Lensed_2p5um",
            "mfd_x_um": 2.5,
            "mfd_y_um": 2.5,
            "n_env": 1.0,
            "ar_coating": True,
            "core_index": si_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        },
        {
            "case_id": "Si3N4_400nm_Butt_DFB1310nm",
            "core_material": "Si3N4",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1310.0,
            "polarization": "TE0",
            "film_height_um": 0.40,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "w_bus_um": 0.85,
            "w_tip_sweep_um": [0.20, 0.30, 0.40],
            "w_tip_nominal_um": 0.30,
            "taper_length_um": 150.0,
            "taper_profile": "adiabatic",
            "box_thickness_um": 3.0,
            "fiber_preset": "DFB_Laser_Butt",
            "mfd_x_um": 2.4,
            "mfd_y_um": 1.2,
            "n_env": 1.45,
            "ar_coating": True,
            "core_index": sin_1310,
            "clad_index": sio2_1310,
            "top_index": sio2_1310,
        },
    ]

    # ==========================================================================
    # Suite 6B: Gaussian-Apodized Surface Grating Coupler (26 chirped grooves)
    # ==========================================================================
    gc_apo = synthesize_grating_teeth(
        n_core=si_1550["center"]["nx"],
        n_clad=sio2_1550["center"]["nx"],
        n_top=sio2_1550["center"]["nx"],
        n_env=1.0,
        wl_um=1.55,
        h_film_um=0.22,
        etch_depth_um=0.07,
        fiber_angle_deg=10.0,
        mfd_um=10.4,
        polarization="TE0",
        apodized=True,
        pitch_um=None,
        fill_factor=0.50,
        num_periods=26,
        min_feature_nm=80.0,
    )
    suite6b_apodized_gc = {
        "case_id": "SOI_220nm_Apodized_GC_70nmEtch",
        "core_material": "Si",
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "substrate_material": "Si",
        "wavelength_nm": 1550.0,
        "wavelength_sweep_nm": [1500.0, 1525.0, 1540.0, 1550.0, 1560.0, 1575.0, 1600.0],
        "polarization": "TE0",
        "film_height_um": 0.22,
        "etch_depth_um": 0.07,
        "box_thickness_um": 2.0,
        "top_clad_thickness_um": 1.5,
        "fiber_angle_deg": 10.0,
        "mfd_um": 10.4,
        "n_env": 1.0,
        "grating_width_um": 12.0,
        "num_periods": 26,
        "pitch_mean_um": round(float(gc_apo["pitch_mean_um"]), 6),
        "ff_start": round(float(gc_apo["ff_start"]), 4),
        "ff_end": round(float(gc_apo["ff_end"]), 4),
        "grooves_um": [[round(a, 6), round(b, 6)] for a, b in gc_apo["grooves_um"]],
        "core_index": si_1550,
        "clad_index": sio2_1550,
        "top_index": sio2_1550,
    }

    # ==========================================================================
    # Suite 7: 2x2 MMI (3D FDTD GPU) + Low-Q Full Ring 3D FDTD + INTERCONNECT
    # ==========================================================================
    # Analytic Soldano W_eff for 220 nm SOI W_MMI = 6.0 um -> port offset = W_eff / 6
    n_c_si = si_1550["center"]["nx"]
    n_cl_ox = sio2_1550["center"]["nx"]
    w_eff_6 = 6.0 + (1.55 / math.pi) / math.sqrt(n_c_si ** 2 - n_cl_ox ** 2)
    suite7a_mmi_2x2 = {
        "case_id": "SOI_1550nm_MMI_2x2",
        "splitter_type": "mmi_2x2",
        "core_material": "Si",
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "film_height_um": 0.22,
        "slab_height_um": 0.0,
        "sidewall_deg": 90.0,
        "arm_width_um": 0.48,
        "mmi_width_um": 6.0,
        "mmi_length_um": 40.1,  # Canonical paired-interference 3-dB length for W=6.0 um 220 nm SOI
        "mmi_taper_width_um": 1.10,
        "mmi_taper_length_um": 12.0,
        "port_offset_um": round(w_eff_6 / 6.0, 4),
        "core_index": si_1550,
        "clad_index": sio2_1550,
        "top_index": sio2_1550,
    }

    suite7b_full_ring_lowq = {
        "case_id": "SOI_1550nm_LowQ_FullRing_R5um",
        "topology": "point_circular_ap",
        "core_material": "Si",
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "wl_min_nm": 1510.0,
        "wl_max_nm": 1590.0,
        "film_height_um": 0.22,
        "slab_height_um": 0.0,
        "sidewall_deg": 90.0,
        "ring_width_um": 0.45,
        "bus_width_um": 0.45,
        "ring_radius_um": 5.0,
        "gap_thru_um": 0.12,
        "alpha_loss_db_per_cm": 3.0,
        "core_index": si_1550,
        "clad_index": sio2_1550,
        "top_index": sio2_1550,
    }

    suite7c_interconnect_circuit = {
        "case_id": "SOI_1550nm_DC_MZI_dL80um_and_WDM4ch",
        "mzi_delta_length_um": 80.0,
        "mzi_arm1_length_um": 300.0,
        "mzi_wavelength_nm": 1550.0,
        "mzi_span_nm": 30.0,
        "wdm_num_channels": 4,
        "wdm_data_rate_gbps": 25.0,
        "wdm_channel_spacing_ghz": 100.0,
        "wdm_ring_radius_um": 10.0,
        "wdm_gap_um": 0.18,
    }

    # ==========================================================================
    # Suite 8: Sub-THz EOM RF Traveling-Wave GSG & Slow-Wave T-Rail Electrodes
    # ==========================================================================
    au_spec = ELECTRODE_METAL_CATALOG["Au"]
    suite8_rf_electrodes = [
        {
            "case_id": "TFLN_GSG_CPW_HRSi",
            "electrode_topology": "gsg_cpw",
            "core_material": "LN_Xcut",
            "substrate_type": "hr_si",
            "metal_material": "Au",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.60,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.10,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 0.80,
            "signal_width_um": 20.0,
            "electrode_gap_um": 5.0,
            "ground_width_um": 55.0,
            "metal_thickness_um": 1.10,
            "active_length_um": 5000.0,
            "t_rail_length_um": 0.0,
            "t_rail_period_um": 50.0,
            "t_rail_duty": 1.0,
            "t_rail_cap_width_um": 0.0,
            "t_rail_stem_width_um": 0.0,
            "slab_corridor_half_um": None,
            "eps_core_rf": list(STATIC_PERMITTIVITY_DC["LN_Xcut"]),
            "eps_box_rf": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_sub_rf": float(RF_SUBSTRATE_CATALOG["hr_si"]["eps_rf"]),
            "tan_d_core": float(DIELECTRIC_LOSS_TANGENT_RF["LN_Xcut"]),
            "tan_d_box": float(DIELECTRIC_LOSS_TANGENT_RF["SiO2"]),
            "tan_d_sub": float(RF_SUBSTRATE_CATALOG["hr_si"]["tan_delta"]),
            "sigma_sub_s_per_m": float(RF_SUBSTRATE_CATALOG["hr_si"]["sigma_s_per_m"]),
            "rho_metal_ohm_m": float(au_spec["rho_ohm_m"]),
            "eval_freqs_ghz": [10.0, 40.0, 100.0],
            "core_index": ln_xcut_1550,
            "clad_index": sio2_1550,
        },
        {
            "case_id": "TFLN_GSG_CPW_Quartz",
            "electrode_topology": "gsg_cpw",
            "core_material": "LN_Xcut",
            "substrate_type": "quartz",
            "metal_material": "Au",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.60,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.10,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 0.80,
            "signal_width_um": 26.0,
            "electrode_gap_um": 5.0,
            "ground_width_um": 55.0,
            "metal_thickness_um": 1.50,
            "active_length_um": 5000.0,
            "t_rail_length_um": 0.0,
            "t_rail_period_um": 50.0,
            "t_rail_duty": 1.0,
            "t_rail_cap_width_um": 0.0,
            "t_rail_stem_width_um": 0.0,
            "slab_corridor_half_um": None,
            "eps_core_rf": list(STATIC_PERMITTIVITY_DC["LN_Xcut"]),
            "eps_box_rf": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_sub_rf": float(RF_SUBSTRATE_CATALOG["quartz"]["eps_rf"]),
            "tan_d_core": float(DIELECTRIC_LOSS_TANGENT_RF["LN_Xcut"]),
            "tan_d_box": float(DIELECTRIC_LOSS_TANGENT_RF["SiO2"]),
            "tan_d_sub": float(RF_SUBSTRATE_CATALOG["quartz"]["tan_delta"]),
            "sigma_sub_s_per_m": float(RF_SUBSTRATE_CATALOG["quartz"]["sigma_s_per_m"]),
            "rho_metal_ohm_m": float(au_spec["rho_ohm_m"]),
            "eval_freqs_ghz": [10.0, 40.0, 100.0],
            "core_index": ln_xcut_1550,
            "clad_index": sio2_1550,
        },
        {
            "case_id": "TFLN_SlowWave_TRail_Quartz",
            "electrode_topology": "t_rail_cpw",
            "core_material": "LN_Xcut",
            "substrate_type": "quartz",
            "metal_material": "Au",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.60,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.10,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 0.80,
            "signal_width_um": 82.0,
            "electrode_gap_um": 5.0,
            "ground_width_um": 55.0,
            "metal_thickness_um": 1.50,
            "active_length_um": 4500.0,
            "t_rail_length_um": 10.8,
            "t_rail_period_um": 25.0,
            "t_rail_duty": 0.88,
            "t_rail_cap_width_um": 3.888,
            "t_rail_stem_width_um": 4.0,
            "slab_corridor_half_um": 5.5,
            "eps_core_rf": list(STATIC_PERMITTIVITY_DC["LN_Xcut"]),
            "eps_box_rf": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_sub_rf": float(RF_SUBSTRATE_CATALOG["quartz"]["eps_rf"]),
            "tan_d_core": float(DIELECTRIC_LOSS_TANGENT_RF["LN_Xcut"]),
            "tan_d_box": float(DIELECTRIC_LOSS_TANGENT_RF["SiO2"]),
            "tan_d_sub": float(RF_SUBSTRATE_CATALOG["quartz"]["tan_delta"]),
            "sigma_sub_s_per_m": float(RF_SUBSTRATE_CATALOG["quartz"]["sigma_s_per_m"]),
            "rho_metal_ohm_m": float(au_spec["rho_ohm_m"]),
            "eval_freqs_ghz": [10.0, 40.0, 100.0, 200.0, 300.0],
            "core_index": ln_xcut_1550,
            "clad_index": sio2_1550,
        },
        {
            "case_id": "TFLN_GSG_CPW_UndercutSi",
            "electrode_topology": "gsg_cpw",
            "core_material": "LN_Xcut",
            "substrate_type": "undercut_si",
            "metal_material": "Au",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.60,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.10,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 0.80,
            "signal_width_um": 20.0,
            "electrode_gap_um": 5.0,
            "ground_width_um": 55.0,
            "metal_thickness_um": 1.10,
            "active_length_um": 5000.0,
            "t_rail_length_um": 0.0,
            "t_rail_period_um": 50.0,
            "t_rail_duty": 1.0,
            "t_rail_cap_width_um": 0.0,
            "t_rail_stem_width_um": 0.0,
            "slab_corridor_half_um": None,
            "undercut_cavity_half_w_um": 18.5,
            "undercut_cavity_depth_um": 26.0,
            "eps_core_rf": list(STATIC_PERMITTIVITY_DC["LN_Xcut"]),
            "eps_box_rf": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_sub_rf": float(RF_SUBSTRATE_CATALOG["undercut_si"]["eps_rf"]),
            "tan_d_core": float(DIELECTRIC_LOSS_TANGENT_RF["LN_Xcut"]),
            "tan_d_box": float(DIELECTRIC_LOSS_TANGENT_RF["SiO2"]),
            "tan_d_sub": float(RF_SUBSTRATE_CATALOG["undercut_si"]["tan_delta"]),
            "sigma_sub_s_per_m": float(RF_SUBSTRATE_CATALOG["undercut_si"]["sigma_s_per_m"]),
            "rho_metal_ohm_m": float(au_spec["rho_ohm_m"]),
            "eval_freqs_ghz": [10.0, 40.0, 100.0],
            "core_index": ln_xcut_1550,
            "clad_index": sio2_1550,
        },
    ]

    # ==========================================================================
    # Suite 9: 2D Silicon PN Carrier-Depletion Semiconductor TCAD
    # ==========================================================================
    suite9_si_pn_tcad = []
    for cid, jtype in (
        ("SOI_220nm_Lateral_PN", "lateral_pn"),
        ("SOI_220nm_LShaped_PN", "l_shaped_pn"),
        ("SOI_220nm_UShaped_PN", "u_shaped_pn"),
    ):
        suite9_si_pn_tcad.append({
            "case_id": cid,
            "junction_type": jtype,
            "wavelength_nm": 1550.0,
            "polarization": "TE0",
            "film_height_um": 0.22,
            "slab_height_um": 0.09,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 0.45,
            "active_length_um": 1500.0,
            "drive_scheme": "single_arm",
            "na_core_cm3": 5.0e17,
            "nd_core_cm3": 5.0e17,
            "na_inter_cm3": 2.0e18,
            "nd_inter_cm3": 2.0e18,
            "n_heavy_contact_cm3": 1.0e20,
            "pn_offset_nm": 25.0,
            "l_vertical_y_nm": 110.0,
            "implant_straggle_nm": 18.0,
            "inter_implant_offset_um": 0.35,
            "heavy_contact_offset_um": 0.90,
            "bias_voltages_v": [0.0, -1.0, -2.0, -3.0, -4.0],
            "map_bias_v": -3.0,
            "eps_si_dc": 11.7,
            "eps_ox_dc": 3.9,
            "ni_si_cm3": 1.0e10,
            "temp_k": 300.0,
            "core_index": si_1550,
            "clad_index": sio2_1550,
        })

    # ==========================================================================
    # Suite 10: Advanced Microheaters + 3D Finite-Length + Lumerical HEAT 3-Way
    # ==========================================================================
    suite10_thermal_adv = [
        {
            "case_id": "SOI_220nm_Undercut_Suspended",
            "thermal_structure": "undercut_suspended",
            "core_material": "Si",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 0.45,
            "active_length_um": 200.0,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 1.80,
            "heater_material": "TiN",
            "heater_width_um": 1.80,
            "heater_thickness_um": 0.12,
            "heater_gap_um": 0.95,
            "trench_edge_offset_um": 2.80,
            "trench_width_um": 6.00,
            "trench_depth_um": 42.0,
            "undercut_width_um": 16.0,
            "undercut_depth_um": 8.0,
            "neighbor_pitch_um": 20.0,
            "include_pn_contact_sinking": False,
            "core_index": si_1550,
            "clad_index": sio2_1550,
        },
        {
            "case_id": "SOI_220nm_Doped_Si_Rib",
            "thermal_structure": "doped_si_rib",
            "core_material": "Si",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.09,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 0.45,
            "active_length_um": 200.0,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 1.80,
            "heater_material": "doped_Si",
            "heater_width_um": 1.80,
            "heater_thickness_um": 0.09,
            "heater_gap_um": 0.0,
            "neighbor_pitch_um": 20.0,
            "include_pn_contact_sinking": False,
            "core_index": si_1550,
            "clad_index": sio2_1550,
        },
        {
            "case_id": "SOI_220nm_CoInt_PN_TiN_Heater",
            "thermal_structure": "top_heater",
            "core_material": "Si",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.09,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 0.45,
            "active_length_um": 200.0,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 1.80,
            "heater_material": "TiN",
            "heater_width_um": 1.80,
            "heater_thickness_um": 0.12,
            "heater_gap_um": 0.95,
            "neighbor_pitch_um": 20.0,
            "include_pn_contact_sinking": True,
            "pn_contact_clearance_um": 0.85,
            "pn_via_width_um": 0.45,
            "pn_metal_width_um": 9.00,
            "pn_metal_thickness_um": 0.45,
            "pn_via_longitudinal_fill": 0.40,
            "core_index": si_1550,
            "clad_index": sio2_1550,
        },
    ]
    for tc in suite10_thermal_adv:
        tc["k_core_w_mk"] = float(THERMAL_CONDUCTIVITY_W_MK["Si"])
        tc["k_slab_doped_w_mk"] = 62.0
        tc["k_ox_w_mk"] = float(THERMAL_CONDUCTIVITY_W_MK["SiO2"])
        tc["k_si_sub_w_mk"] = 148.0
        tc["k_tin_w_mk"] = float(HEATER_METAL_CATALOG["TiN"]["kappa_th_w_mk"])
        tc["k_air_w_mk"] = float(THERMAL_CONDUCTIVITY_W_MK["Air"])
        tc["cv_core_j_m3k"] = float(VOLUMETRIC_HEAT_CAPACITY_J_M3K["Si"])
        tc["cv_ox_j_m3k"] = float(VOLUMETRIC_HEAT_CAPACITY_J_M3K["SiO2"])
        tc["cv_si_sub_j_m3k"] = 1.63e6
        tc["cv_tin_j_m3k"] = 2.45e6
        tc["cv_air_j_m3k"] = 1.20e3

    # ==========================================================================
    # Suite 11: Cleanroom Lithography Correlated CD & Gap Bias Drift (+/-20 nm)
    # ==========================================================================
    suite11_fab_bias = {
        "case_id": "SOI_1550nm_PR_CD_Bias_Sweep",
        "core_material": "Si",
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "dwl_nm": 12.0,
        "film_height_um": 0.22,
        "slab_height_um": 0.0,
        "sidewall_deg": 90.0,
        "ring_radius_um": 15.0,
        "ring_width_nom_um": 0.45,
        "bus_width_nom_um": 0.45,
        "gap_nom_um": 0.18,
        "proximity_gap_factor": 1.05,
        "delta_cd_nm_list": [-20.0, 0.0, +20.0],
        "core_index": si_1550,
        "clad_index": sio2_1550,
        "top_index": sio2_1550,
    }

    return {
        "schema_version": "2.0.0",
        "description": "Phase-2 neutral physical benchmark specifications (Suites 6-11) completing 100% physical engine coverage",
        "suite6a_butt_couplers": suite6a_butt_couplers,
        "suite6b_apodized_gc": suite6b_apodized_gc,
        "suite7a_mmi_2x2": suite7a_mmi_2x2,
        "suite7b_full_ring_lowq": suite7b_full_ring_lowq,
        "suite7c_interconnect_circuit": suite7c_interconnect_circuit,
        "suite8_rf_electrodes": suite8_rf_electrodes,
        "suite9_si_pn_tcad": suite9_si_pn_tcad,
        "suite10_thermal_adv": suite10_thermal_adv,
        "suite11_fab_bias": suite11_fab_bias,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_specs_p2()
    SPECS_P2_JSON.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"Wrote Phase-2 neutral physical specifications to {SPECS_P2_JSON}")
    for k, v in specs.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} cases")
        elif isinstance(v, dict):
            print(f"  {k}: 1 case ({v.get('case_id', '')})")


if __name__ == "__main__":
    main()
