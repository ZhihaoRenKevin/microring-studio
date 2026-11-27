"""Neutral Physical Specification Generator for 3-Way Commercial Solver Validation.

Generates `benchmarks/outputs/commercial_validation_specs.json` containing ONLY
first-principles physical geometry, wavelength triplets (lambda0, lambda0 +/- dlambda),
exact literature Sellmeier refractive index tensors, thermal conductivities / heat capacities,
DC permittivity tensors, and Pockels tensors.

ZERO solver outputs are stored in this specification file so that:
  1. MicroRing Studio (`run_microring_independent.py`)
  2. Ansys Lumerical (`run_lumerical_independent.py`)
  3. COMSOL Multiphysics (`run_comsol_independent.py`)
each execute completely independently from the identical physical definition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring.active_tuning import (
    HEATER_METAL_CATALOG,
    STATIC_PERMITTIVITY_DC,
    THERMAL_CONDUCTIVITY_W_MK,
    VOLUMETRIC_HEAT_CAPACITY_J_M3K,
)
from microring.io_couplers import synthesize_grating_teeth
from microring.materials import T_REF_K, get_material_fn

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SPECS_JSON = OUT_DIR / "commercial_validation_specs.json"


def _eval_mat_triplet(mat_name: str, wl0_nm: float, dwl_nm: float) -> dict:
    """Evaluate exact literature Sellmeier index tensor at (wl0 - dwl, wl0, wl0 + dwl)."""
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
    out["crystal_symmetry"] = str(m0.crystal_symmetry)
    out["source"] = str(m0.source)
    return out


def build_specs() -> dict:
    # ==========================================================================
    # Suite 1 & Suite 2A: 5 Canonical Multi-Waveband Photonic Resonators
    # ==========================================================================
    raw_5band = [
        {
            "case_id": "Vis_532nm_Si3N4_Strip",
            "topology": "point_circular_ap",
            "core_material": "Si3N4",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 532.0,
            "dwl_nm": 4.0,
            "film_height_um": 0.18,
            "slab_height_um": 0.0,
            "sidewall_deg": 88.0,
            "ring_width_um": 0.38,
            "bus_width_um": 0.34,
            "ring_radius_um": 25.0,
            "gap_thru_um": 0.14,
            "straight_length_um": 0.0,
            "pulley_angle_thru_deg": 0.0,
        },
        {
            "case_id": "VNIR_780nm_TiO2_Strip",
            "topology": "point_circular_ap",
            "core_material": "TiO2",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 780.0,
            "dwl_nm": 6.0,
            "film_height_um": 0.20,
            "slab_height_um": 0.0,
            "sidewall_deg": 86.0,
            "ring_width_um": 0.42,
            "bus_width_um": 0.36,
            "ring_radius_um": 30.0,
            "gap_thru_um": 0.16,
            "straight_length_um": 0.0,
            "pulley_angle_thru_deg": 0.0,
        },
        {
            "case_id": "OBand_1310nm_SOI_220nm",
            "topology": "point_circular_ad",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1310.0,
            "dwl_nm": 10.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "ring_width_um": 0.40,
            "bus_width_um": 0.38,
            "ring_radius_um": 15.0,
            "gap_thru_um": 0.16,
            "straight_length_um": 0.0,
            "pulley_angle_thru_deg": 0.0,
        },
        {
            "case_id": "CBand_1550nm_4HSiC_Pulley",
            "topology": "pulley_ap",
            "core_material": "4H-SiC",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "dwl_nm": 12.0,
            "film_height_um": 0.60,
            "slab_height_um": 0.0,
            "sidewall_deg": 82.0,
            "ring_width_um": 1.05,
            "bus_width_um": 0.85,
            "ring_radius_um": 40.0,
            "gap_thru_um": 0.25,
            "straight_length_um": 0.0,
            "pulley_angle_thru_deg": 28.0,
        },
        {
            "case_id": "MidIR_4200nm_Ge_CaF2",
            "topology": "racetrack_ap",
            "core_material": "Ge",
            "clad_material": "CaF2",
            "top_clad_material": "Air",
            "wavelength_nm": 4200.0,
            "dwl_nm": 30.0,
            "film_height_um": 1.40,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "ring_width_um": 2.00,
            "bus_width_um": 2.00,
            "ring_radius_um": 65.0,
            "gap_thru_um": 0.35,
            "straight_length_um": 18.0,
            "pulley_angle_thru_deg": 0.0,
        },
    ]

    suite1_eigenmodes = []
    for c in raw_5band:
        wl_nm = c["wavelength_nm"]
        wl_um = wl_nm * 1e-3
        wl_scale = float(np.clip(wl_nm / 1550.0, 0.35, 3.50))
        box_pad = round(max(2.0, 1.35 * wl_um), 3)
        dom_w = round(max(4.2 * wl_scale, c["ring_width_um"] + 2.6 * wl_um), 3)
        fem_res = 0.020 if c["core_material"] == "Si" else round(float(np.clip(0.030 * wl_scale, 0.015, 0.085)), 4)
        fde_dl = (
            0.010 if c["core_material"] == "Si"
            else (0.025 if c["core_material"] == "Ge" else round(float(np.clip(0.018 * wl_scale, 0.012, 0.050)), 4))
        )
        suite1_eigenmodes.append({
            **c,
            "domain_width_um": dom_w,
            "box_pad_um": box_pad,
            "fem_core_res_um": fem_res,
            "fde_dl_um": fde_dl,
            "core_index": _eval_mat_triplet(c["core_material"], wl_nm, c["dwl_nm"]),
            "clad_index": _eval_mat_triplet(c["clad_material"], wl_nm, c["dwl_nm"]),
            "top_index": _eval_mat_triplet(c["top_clad_material"], wl_nm, c["dwl_nm"]),
        })

    # ==========================================================================
    # Suite 2B: SOI 1550 nm Gap-Scaling & Sub-120nm Validity Boundary Audit
    # ==========================================================================
    si_1550 = _eval_mat_triplet("Si", 1550.0, 12.0)
    sio2_1550 = _eval_mat_triplet("SiO2", 1550.0, 12.0)
    suite2b_soi_gap_sweep = []
    for g_nm in (100, 150, 200, 250):
        suite2b_soi_gap_sweep.append({
            "case_id": f"SOI_1550nm_Point_Gap{g_nm}nm",
            "topology": "point_circular_ap",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "dwl_nm": 12.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "ring_width_um": 0.45,
            "bus_width_um": 0.45,
            "ring_radius_um": 15.0,
            "gap_thru_um": round(g_nm * 1e-3, 4),
            "straight_length_um": 0.0,
            "pulley_angle_thru_deg": 0.0,
            "core_index": si_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        })

    # ==========================================================================
    # Suite 3: MZI Splitters (DC Supermodes + 1x2 MMI + Y-Branch) & 2D Grating Coupler
    # ==========================================================================
    suite3a_dc_supermodes = []
    for g_nm in (150, 200, 250):
        g_um = round(g_nm * 1e-3, 4)
        suite3a_dc_supermodes.append({
            "case_id": f"SOI_1550nm_DC_Gap{g_nm}nm",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 0.45,
            "gap_um": g_um,
            "domain_width_um": 5.6,
            "box_pad_um": 2.25,
            "core_index": si_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        })

    suite3b_mzi_splitters = [
        {
            "case_id": "SOI_1550nm_MMI_1x2",
            "splitter_type": "mmi_1x2",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "arm_width_um": 0.48,
            "mmi_width_um": 4.0,
            "mmi_length_um": 14.3,  # Canonical 1x2 symmetric self-imaging length for W=4.0um SOI
            "mmi_taper_width_um": 1.10,
            "mmi_taper_length_um": 12.0,
            "port_offset_um": 1.00,  # +/- W_MMI / 4
            "core_index": si_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        },
        {
            "case_id": "SOI_1550nm_YBranch_1x2",
            "splitter_type": "y_branch",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.22,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "arm_width_um": 0.48,
            "y_apex_gap_um": 0.13,
            "y_taper_length_um": 12.0,
            "y_branch_length_um": 18.0,
            "arm_sep_out_um": 2.40,
            "core_index": si_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        },
    ]

    # Synthesize exact uniform 70-nm etch SOI grating teeth so all 3 solvers share identical teeth!
    gc_synth = synthesize_grating_teeth(
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
        apodized=False,
        pitch_um=None,
        fill_factor=0.50,
        num_periods=26,
    )
    suite3c_grating_coupler = {
        "case_id": "SOI_220nm_Uniform_GC_70nmEtch",
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
        "pitch_um": round(float(gc_synth["pitch_mean_um"]), 6),
        "fill_factor": 0.50,
        "grooves_um": [[round(a, 6), round(b, 6)] for a, b in gc_synth["grooves_um"]],
        "core_index": si_1550,
        "clad_index": sio2_1550,
        "top_index": sio2_1550,
    }

    # ==========================================================================
    # Suite 4: 2D Steady-State & Transient Thermo-Optic Microheater Multiphysics
    # ==========================================================================
    sin_1550 = _eval_mat_triplet("Si3N4", 1550.0, 12.0)
    raw_thermal = [
        {
            "case_id": "SOI_220nm_Top_TiN_Heater",
            "thermal_structure": "top_heater",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
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
            "neighbor_pitch_um": 20.0,
            "core_index": si_1550,
        },
        {
            "case_id": "SOI_220nm_Deep_Air_Trench",
            "thermal_structure": "deep_trench",
            "core_material": "Si",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
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
            "neighbor_pitch_um": 20.0,
            "core_index": si_1550,
        },
        {
            "case_id": "Si3N4_400nm_Top_TiN_BOX2p0",
            "thermal_structure": "top_heater",
            "core_material": "Si3N4",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.40,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.20,
            "active_length_um": 200.0,
            "box_thickness_um": 2.00,
            "top_clad_thickness_um": 2.00,
            "heater_material": "TiN",
            "heater_width_um": 2.00,
            "heater_thickness_um": 0.12,
            "heater_gap_um": 0.95,
            "trench_edge_offset_um": 2.80,
            "trench_width_um": 6.00,
            "trench_depth_um": 42.0,
            "neighbor_pitch_um": 20.0,
            "core_index": sin_1550,
        },
        {
            "case_id": "Si3N4_400nm_Top_TiN_BOX3p5",
            "thermal_structure": "top_heater",
            "core_material": "Si3N4",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "film_height_um": 0.40,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.20,
            "active_length_um": 200.0,
            "box_thickness_um": 3.50,
            "top_clad_thickness_um": 2.00,
            "heater_material": "TiN",
            "heater_width_um": 2.00,
            "heater_thickness_um": 0.12,
            "heater_gap_um": 0.95,
            "trench_edge_offset_um": 2.80,
            "trench_width_um": 6.00,
            "trench_depth_um": 42.0,
            "neighbor_pitch_um": 20.0,
            "core_index": sin_1550,
        },
    ]

    suite4_thermal = []
    for tc in raw_thermal:
        y_h0 = tc["film_height_um"] + tc["heater_gap_um"]
        y_h1 = y_h0 + tc["heater_thickness_um"]
        y_ox_top = max(tc["top_clad_thickness_um"], y_h1 + 0.45)
        y_air_top = y_ox_top + 4.0
        h_sub = max(26.0, tc["trench_depth_um"] + 14.0) if tc["thermal_structure"] == "deep_trench" else 26.0
        y_sub_bot = -tc["box_thickness_um"] - h_sub
        x_dom = max(tc["neighbor_pitch_um"] + 14.0, 36.0)
        suite4_thermal.append({
            **tc,
            "y_heat0_um": round(y_h0, 4),
            "y_heat1_um": round(y_h1, 4),
            "y_ox_top_um": round(y_ox_top, 4),
            "y_air_top_um": round(y_air_top, 4),
            "y_sub_bot_um": round(y_sub_bot, 4),
            "x_dom_um": round(x_dom, 4),
            "k_core_w_mk": float(THERMAL_CONDUCTIVITY_W_MK[tc["core_material"]]),
            "k_ox_w_mk": float(THERMAL_CONDUCTIVITY_W_MK["SiO2"]),
            "k_si_sub_w_mk": 148.0,
            "k_heater_w_mk": float(HEATER_METAL_CATALOG[tc["heater_material"]]["kappa_th_w_mk"]),
            "k_air_w_mk": float(THERMAL_CONDUCTIVITY_W_MK["Air"]),
            "cv_core_j_m3k": float(VOLUMETRIC_HEAT_CAPACITY_J_M3K[tc["core_material"]]),
            "cv_ox_j_m3k": float(VOLUMETRIC_HEAT_CAPACITY_J_M3K["SiO2"]),
            "cv_si_sub_j_m3k": 1.63e6,
            "cv_heater_j_m3k": 2.45e6,
            "cv_air_j_m3k": 1.20e3,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        })

    # ==========================================================================
    # Suite 5: 2D Anisotropic Electrostatic Pockels Electro-Optic Modulators
    # ==========================================================================
    ln_xcut_1550 = _eval_mat_triplet("LN_Xcut", 1550.0, 12.0)
    scaln20_1550 = _eval_mat_triplet("ScAlN_20", 1550.0, 12.0)
    suite5_pockels_eo = [
        {
            "case_id": "TFLN_Xcut_GSG_G5p0um",
            "core_material": "LN_Xcut",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "polarization": "TE0",
            "electrode_config": "inplane_gsg",
            "film_height_um": 0.60,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.20,
            "electrode_gap_um": 5.0,
            "electrode_width_um": 4.0,
            "electrode_thickness_um": 0.45,
            "electrode_voffset_um": 0.75,
            "box_thickness_um": 2.0,
            "top_clad_thickness_um": 1.8,
            "eps_core_dc": list(STATIC_PERMITTIVITY_DC["LN_Xcut"]),
            "eps_clad_dc": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_top_dc": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "core_index": ln_xcut_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        },
        {
            "case_id": "TFLN_Xcut_GSG_G3p2um",
            "core_material": "LN_Xcut",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "polarization": "TE0",
            "electrode_config": "inplane_gsg",
            "film_height_um": 0.60,
            "slab_height_um": 0.30,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.20,
            "electrode_gap_um": 3.2,
            "electrode_width_um": 4.0,
            "electrode_thickness_um": 0.45,
            "electrode_voffset_um": 0.75,
            "box_thickness_um": 2.0,
            "top_clad_thickness_um": 1.8,
            "eps_core_dc": list(STATIC_PERMITTIVITY_DC["LN_Xcut"]),
            "eps_clad_dc": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_top_dc": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "core_index": ln_xcut_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        },
        {
            "case_id": "ScAlN20_Vertical_Plates",
            "core_material": "ScAlN_20",
            "clad_material": "SiO2",
            "top_clad_material": "SiO2",
            "wavelength_nm": 1550.0,
            "polarization": "TM0",
            "electrode_config": "vertical_parallel",
            "film_height_um": 0.50,
            "slab_height_um": 0.0,
            "sidewall_deg": 90.0,
            "waveguide_width_um": 1.00,
            "electrode_gap_um": 2.0,
            "electrode_width_um": 4.0,
            "electrode_thickness_um": 0.45,
            "electrode_voffset_um": 0.75,
            "box_thickness_um": 1.0,
            "top_clad_thickness_um": 1.8,
            "eps_core_dc": list(STATIC_PERMITTIVITY_DC["ScAlN_20"]),
            "eps_clad_dc": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "eps_top_dc": list(STATIC_PERMITTIVITY_DC["SiO2"]),
            "core_index": scaln20_1550,
            "clad_index": sio2_1550,
            "top_index": sio2_1550,
        },
    ]

    return {
        "schema_version": "1.0.0",
        "description": "Neutral physical benchmark specifications for 3-way independent validation (MicroRing Studio v3.9.0 vs Ansys Lumerical 2026R1 vs COMSOL Multiphysics 6.4)",
        "suite1_eigenmodes": suite1_eigenmodes,
        "suite2b_soi_gap_sweep": suite2b_soi_gap_sweep,
        "suite3a_dc_supermodes": suite3a_dc_supermodes,
        "suite3b_mzi_splitters": suite3b_mzi_splitters,
        "suite3c_grating_coupler": suite3c_grating_coupler,
        "suite4_thermal": suite4_thermal,
        "suite5_pockels_eo": suite5_pockels_eo,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    SPECS_JSON.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"Wrote neutral physical specifications to {SPECS_JSON}")
    for k, v in specs.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} cases")
        elif isinstance(v, dict):
            print(f"  {k}: 1 case ({v.get('case_id', '')})")


if __name__ == "__main__":
    main()
