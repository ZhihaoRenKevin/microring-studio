"""Neutral Phase-3 Validation Specification Generator (Suites 12 & 13).

Exports ONLY geometry, wavelength bands, and dispersive literature refractive indices
(zero solver outputs) into `benchmarks/outputs/commercial_validation_specs_p3.json`.

Both `benchmarks/run_microring_independent_p3.py` and
`benchmarks/run_lumerical_independent_p3.py` read ONLY this neutral JSON specification
(plus MicroRing's exported `.gds` mask strictly for the Suite 12B GDS roundtrip test).

Suites covered:
  - Suite 12A: Strictly converged (`status = 2`, `auto_shutoff = 1e-5`) Low-Q Full-Ring
               3D FDTD benchmarks on RTX 5090 GPU across all 5 ring architectures:
                 1. `SOI_LowQ_AddDrop_R5um_g100nm` (4-port Circular Add-Drop, Q_L ~ 2,500)
                 2. `SOI_LowQ_AddDrop_R5um_g120nm` (4-port Circular Add-Drop, Q_L ~ 4,100)
                 3. `SOI_LowQ_Lossy_AllPass_CC_R5um` (Complex-n Lossy All-Pass Critical Coupling)
                 4. `SOI_LowQ_Racetrack_AD_R5um_Ls3um` (4-port Racetrack Add-Drop, Q_L ~ 1,410)
                 5. `SOI_LowQ_Pulley_AD_R5um_th20deg` (4-port Concentric Pulley Add-Drop, Q_L ~ 980)
                 6. `SOI_LowQ_DualRing_CROW_R5um` (2nd-Order Coupled Dual-Ring CROW Add-Drop)
                 7. `SOI_LowQ_PCM_GSST_Switch_R5um` (Non-Volatile `a-GSST` vs `c-GSST` Patch Switch)
  - Suite 12B: End-to-End `.GDS` Export -> Direct Lumerical `gdsimport` 3D FDTD Roundtrip
               + Blind `fit_single_resonance_cmt` Audit on Clean & Sidewall-Bump Split-Doublet
               Full-Wave 3D FDTD Spectra.
  - Suite 13 : Complete 3D FDTD MZI & Full-3D Chip I/O Verification:
                 1. `SOI_1550nm_MMI_2x2_OptLen` (`2x2` MMI at optimal self-imaging length `L = 42.3 um`)
                 2. `SOI_1550nm_DC_2x2_FullSBend` (Full `2x2` Directional Coupler with 12 um S-bends)
                 3. `SOI_1550nm_LowQ_RAMZI_R5um` (Complete Low-Q Ring-Assisted MZI)
                 4. `SOI_1550nm_3D_Focusing_GC` (Full 3D FDTD 20-tooth Curved Focusing Grating Coupler)
                 5. `SOI_1550nm_3D_Butt_Taper_30um` (Full 3D FDTD 30 um Inverse-Taper Butt Coupler)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from microring.materials import T_REF_K, get_material_fn  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SPECS_P3_JSON = OUT_DIR / "commercial_validation_specs_p3.json"


def _mat_sample(mat_key: str, wl_nm: float) -> dict:
    fn = get_material_fn(mat_key)
    m = fn(wl_nm * 1e-9, T_REF_K)
    return {
        "wavelength_nm": wl_nm,
        "nx": round(float(m.nx), 8),
        "ny": round(float(m.ny), 8),
        "nz": round(float(m.nz), 8),
        "kx": round(float(m.kx), 8),
        "ky": round(float(m.ky), 8),
        "kz": round(float(m.kz), 8),
    }


def _mat_triplet(mat_key: str, wl_nm: float = 1550.0, dwl_nm: float = 12.0) -> dict:
    return {
        "minus": _mat_sample(mat_key, wl_nm - dwl_nm),
        "center": _mat_sample(mat_key, wl_nm),
        "plus": _mat_sample(mat_key, wl_nm + dwl_nm),
    }


def build_phase3_specs() -> dict:
    si_1550 = _mat_triplet("Si", 1550.0, 12.0)
    sio2_1550 = _mat_triplet("SiO2", 1550.0, 12.0)
    gsst_a = _mat_sample("GSST_amorphous", 1550.0)
    gsst_c = _mat_sample("GSST_crystalline", 1550.0)

    # Compact 20-tooth curved focusing grating coupler groove radii (70 nm etch on 220 nm SOI, 10 deg SMF-28)
    # Matched to the first 20 apodized teeth of Suite 6B (Rytov Bragg pitch 588 -> 615 nm, ff 0.864 -> 0.48)
    # shifted radially to start at focal distance r0 = 11.50 um!
    specs_p2 = json.loads((OUT_DIR / "commercial_validation_specs_p2.json").read_text(encoding="utf-8"))
    s6b_grooves = specs_p2["suite6b_apodized_gc"]["grooves_um"][:20]
    r0_gc = 11.50
    gc3d_grooves_r_um: list[list[float]] = [
        [round(r0_gc + float(g0), 5), round(r0_gc + float(g1), 5)] for g0, g1 in s6b_grooves
    ]

    specs = {
        "protocol": (
            "Neutral Phase-3 physical specifications (zero solver outputs). "
            "All 3D FDTD full-ring benchmarks enforce auto_shutoff_min = 1e-5 with sufficient "
            "simulation time for strict status = 2 convergence (zero Gibbs truncation ringing)."
        ),
        "materials_1550nm": {
            "Si": si_1550,
            "SiO2": sio2_1550,
            "GSST_amorphous": gsst_a,
            "GSST_crystalline": gsst_c,
        },
        "suite12a_lowq_full_rings": [
            {
                "case_id": "SOI_LowQ_AddDrop_R5um_g100nm",
                "topology": "point_circular_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "gap_thru_um": 0.10,
                "gap_drop_um": 0.10,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "alpha_loss_db_per_cm": 0.5,
                "fdtd_sim_time_ps": 40.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            {
                "case_id": "SOI_LowQ_AddDrop_R5um_g120nm",
                "topology": "point_circular_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "gap_thru_um": 0.12,
                "gap_drop_um": 0.12,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "alpha_loss_db_per_cm": 0.5,
                "fdtd_sim_time_ps": 60.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            {
                "case_id": "SOI_LowQ_Lossy_AllPass_CC_R5um",
                "topology": "point_circular_ap",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "gap_thru_um": 0.12,
                "ring_core_k_imag": 0.000240,  # explicit complex n_core = n_Si + i*k inside ring core (near-CC!)
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "fdtd_sim_time_ps": 60.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            {
                "case_id": "SOI_LowQ_Racetrack_AD_R5um_Ls3um",
                "topology": "racetrack_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "straight_length_um": 3.0,
                "gap_thru_um": 0.16,
                "gap_drop_um": 0.16,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "alpha_loss_db_per_cm": 0.5,
                "fdtd_sim_time_ps": 32.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            {
                "case_id": "SOI_LowQ_Pulley_AD_R5um_th20deg",
                "topology": "pulley_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.40,
                "ring_radius_um": 5.0,
                "pulley_angle_thru_deg": 20.0,
                "pulley_angle_drop_deg": 20.0,
                "gap_thru_um": 0.15,
                "gap_drop_um": 0.15,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "alpha_loss_db_per_cm": 0.5,
                "fdtd_sim_time_ps": 28.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            {
                "case_id": "SOI_LowQ_DualRing_CROW_R5um",
                "topology": "vernier_crow_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "ring2_radius_um": 5.0,
                "gap_thru_um": 0.10,
                "gap_drop_um": 0.10,
                "inter_ring_gap_um": 0.13,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "alpha_loss_db_per_cm": 0.5,
                "fdtd_sim_time_ps": 50.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            {
                "case_id": "SOI_LowQ_PCM_GSST_Switch_R5um",
                "topology": "point_circular_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "gap_thru_um": 0.12,
                "gap_drop_um": 0.12,
                "cap_h_um": 0.025,
                "cap_w_um": 0.45,
                "cap_length_um": 1.50,
                "cap_states": ["GSST_amorphous", "GSST_crystalline"],
                "gsst_amorphous_index": gsst_a,
                "gsst_crystalline_index": gsst_c,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1525.0,
                "wl_max_nm": 1575.0,
                "alpha_loss_db_per_cm": 0.5,
                "fdtd_sim_time_ps": 55.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
        ],
        "suite12b_gds_and_doublet": {
            "gds_roundtrip": {
                "case_id": "GDS_Roundtrip_AddDrop_R5um",
                "source_case_id": "SOI_LowQ_AddDrop_R5um_g100nm",
                "topology": "point_circular_ad",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "gap_thru_um": 0.10,
                "gap_drop_um": 0.10,
                "gds_layer": 1,
                "gds_datatype": 0,
                "wl_min_nm": 1515.0,
                "wl_max_nm": 1585.0,
                "fdtd_sim_time_ps": 40.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            "split_doublet_fdtd": {
                "case_id": "Blind_CMT_Fit_SplitDoublet_3DFDTD",
                "base_case_id": "SOI_LowQ_AddDrop_R5um_g120nm",
                "film_height_um": 0.22,
                "ring_width_um": 0.45,
                "bus_width_um": 0.45,
                "ring_radius_um": 5.0,
                "gap_thru_um": 0.12,
                "gap_drop_um": 0.12,
                # Intentional sidewall scattering bump at theta = 45 deg exciting CW/CCW mode splitting!
                "bump_theta_deg": 45.0,
                "bump_radial_um": 0.055,
                "bump_azimuthal_um": 0.160,
                "wl_min_nm": 1538.0,
                "wl_max_nm": 1562.0,
                "fdtd_sim_time_ps": 65.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
        },
        "suite13_mzi_and_3d_io": {
            "mmi_2x2_optimal": {
                "case_id": "SOI_1550nm_MMI_2x2_OptLen",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "arm_width_um": 0.45,
                "mmi_width_um": 6.0,
                "mmi_length_um": 42.3,
                "mmi_taper_width_um": 1.25,
                "mmi_taper_length_um": 15.0,
                "port_offset_um": 1.0,
                "wavelength_nm": 1550.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            "dc_2x2_full_sbend": {
                "case_id": "SOI_1550nm_DC_2x2_FullSBend",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "arm_width_um": 0.45,
                "dc_gap_um": 0.20,
                "dc_straight_length_um": 8.2,
                "sbend_length_um": 12.0,
                "arm_separation_um": 3.0,
                "wavelength_nm": 1550.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            "lowq_ramzi": {
                "case_id": "SOI_1550nm_LowQ_RAMZI_R5um",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "arm_width_um": 0.45,
                "dc_gap_um": 0.20,
                "dc_straight_length_um": 8.2,
                "sbend_length_um": 10.0,
                "arm_separation_um": 14.0,
                "arm_straight_um": 15.0,
                "ring_radius_um": 5.0,
                "ring_gap_um": 0.10,
                "wl_min_nm": 1525.0,
                "wl_max_nm": 1575.0,
                "fdtd_sim_time_ps": 45.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            "gc_3d_focusing": {
                "case_id": "SOI_1550nm_3D_Focusing_GC",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "etch_depth_um": 0.07,
                "box_thickness_um": 2.0,
                "top_clad_thickness_um": 1.0,
                "input_wg_width_um": 0.45,
                "opening_angle_deg": 34.0,
                "grooves_r_um": gc3d_grooves_r_um,
                "fiber_angle_deg": 10.0,
                "mfd_um": 10.4,
                "n_env": 1.0,
                "wavelength_nm": 1550.0,
                "wl_min_nm": 1510.0,
                "wl_max_nm": 1590.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
            "butt_3d_taper": {
                "case_id": "SOI_1550nm_3D_Butt_Taper_30um",
                "core_material": "Si",
                "clad_material": "SiO2",
                "film_height_um": 0.22,
                "w_tip_um": 0.20,
                "w_bus_um": 0.45,
                "taper_length_um": 30.0,
                "mfd_x_um": 2.5,
                "mfd_y_um": 2.5,
                "n_env": 1.44401866,  # index-matched cladding facet (isolates pure 3D taper + mode overlap)
                "wavelength_nm": 1550.0,
                "core_index": si_1550,
                "clad_index": sio2_1550,
            },
        },
    }
    return specs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_phase3_specs()
    SPECS_P3_JSON.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"Wrote neutral Phase-3 specification JSON -> {SPECS_P3_JSON}")
    print(f"  Suite 12A Low-Q Full Rings : {len(specs['suite12a_lowq_full_rings'])} cases")
    print("  Suite 12B GDS + Doublet Fit: 2 cases")
    print(f"  Suite 13  MZI + 3D Chip I/O: {len(specs['suite13_mzi_and_3d_io'])} cases")


if __name__ == "__main__":
    main()
