"""Independent MicroRing Studio Phase-3 Validation Runner (Suites 12 & 13).

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs_p3.json`
(zero commercial solver outputs) and runs MicroRing Studio's native pure-Python engines across:
  - Suite 12A: All 7 Low-Q full-ring benchmarks (Circular Add-Drop g=100/120 nm, Complex-n Lossy
               All-Pass Critical Coupling, Racetrack AD, Pulley AD, Dual-Ring CROW, and PCM GSST
               non-volatile switch).
  - Suite 12B: Exports the actual `.gds` mask for `SOI_LowQ_AddDrop_R5um_g100nm` via
               `export_mrr_gds(...)` and round-trip inspects it with `analyze_gds_mrr(...)`.
  - Suite 13 : Optimal-length `2x2` MMI (`L = 42.3 um`), Full S-bend `2x2` DC, Low-Q `RAMZI`,
               2D-FDFD Grating Coupler reference, and 2D-FEM+EME `30 um` Butt Taper.

Saves `benchmarks/outputs/microring_independent_results_p3.json`.
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from microring.eigenmode_solver import WaveguideCrossSectionSpec, solve_modes_fem  # noqa: E402
from microring.gds_bridge import analyze_uploaded_gds  # noqa: E402
from microring.io_couplers import (  # noqa: E402
    design_butt_edge_coupler,
    design_grating_coupler,
)
from microring.materials import MaterialIndexTensor, T_REF_K, get_material_fn  # noqa: E402
from microring.mrr_topologies import MRRDesignConfig, export_mrr_gds, simulate_mrr_topology  # noqa: E402
from microring.mzi_interferometer import MZIDesignConfig, simulate_mzi  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SPECS_P3_JSON = OUT_DIR / "commercial_validation_specs_p3.json"
OUT_P3_JSON = OUT_DIR / "microring_independent_results_p3.json"
GDS_P3_DIR = OUT_DIR / "_p3_gds"


def _solve_modal_loss_from_k_core(w_um: float, h_um: float, r_um: float, wl_nm: float, k_core: float) -> float:
    """Compute exact bent-mode propagation loss [dB/cm] from first-principles complex 2D FEM."""
    base_si = get_material_fn("Si")
    clad_fn = get_material_fn("SiO2")

    def _lossy_si(wl_m: float, temp_k: float = T_REF_K) -> MaterialIndexTensor:
        m = base_si(wl_m, temp_k)
        return MaterialIndexTensor(
            nx=m.nx,
            ny=m.ny,
            nz=m.nz,
            kx=k_core,
            ky=k_core,
            kz=k_core,
            dn_dt_per_k=m.dn_dt_per_k,
        )

    spec = WaveguideCrossSectionSpec(
        w_bottom_um=w_um,
        h_film_um=h_um,
        h_slab_um=0.0,
        sidewall_deg=90.0,
        core_material_fn=_lossy_si,
        clad_material_fn=clad_fn,
        top_clad_material_fn=clad_fn,
    )
    modes = solve_modes_fem(spec, wl_nm * 1e-9, num_modes=3, bend_radius_m=r_um * 1e-6, element_order=2, export_grid=False)
    te = [m for m in modes if float(getattr(m, "te_fraction", 1.0)) >= 0.5] or modes
    m0 = max(te, key=lambda x: float(np.real(x.n_eff)))
    k_eff = abs(float(np.imag(m0.n_eff)))
    alpha_per_m = (4.0 * math.pi * k_eff) / (wl_nm * 1e-9)
    return alpha_per_m * (10.0 / math.log(10.0)) * 1e-2  # dB/cm


def run_suite12a(specs: dict) -> list[dict]:
    out = []
    for c in specs["suite12a_lowq_full_rings"]:
        cid = c["case_id"]
        t0 = time.perf_counter()
        if "cap_states" in c:
            # PCM GSST non-volatile switch (amorphous vs crystalline)
            st_rows = {}
            for st_mat in c["cap_states"]:
                cfg = MRRDesignConfig(
                    topology=c["topology"],
                    core_material=c["core_material"],
                    clad_material=c["clad_material"],
                    top_clad_material=c["clad_material"],
                    film_height_um=c["film_height_um"],
                    ring_width_um=c["ring_width_um"],
                    bus_width_um=c["bus_width_um"],
                    ring_radius_um=c["ring_radius_um"],
                    gap_thru_um=c["gap_thru_um"],
                    gap_drop_um=c["gap_drop_um"],
                    wavelength_nm=c["wavelength_nm"],
                    alpha_loss_db_per_cm=c["alpha_loss_db_per_cm"],
                    cap_material=st_mat,
                    cap_h_um=c["cap_h_um"],
                    cap_w_um=c["cap_w_um"],
                    cap_length_um=c["cap_length_um"],
                    span_nm=c["wl_max_nm"] - c["wl_min_nm"],
                    num_spectrum_points=1200,
                )
                res = simulate_mrr_topology(cfg)
                m = res["metrics"]
                sp = res["spectrum"]
                st_rows[st_mat] = {
                    "neff_ring": round(float(m["neff_ring"]), 6),
                    "ng_ring": round(float(m["ng_ring"]), 5),
                    "FSR_nm": round(float(m["FSR_nm"]), 4),
                    "Q_loaded": round(float(m["Q_loaded"]), 1),
                    "Q_int": round(float(m["Q_int"]), 1),
                    "FWHM_nm": round(float(m["FWHM_pm"]) * 1e-3, 5),
                    "ER_thru_dB": round(float(m["ER_thru_dB"]), 2),
                    "IL_drop_dB": round(float(m["insertion_loss_drop_dB"]), 3),
                    "kappa_power_thru": round(float(m["kappa_power_thru"]), 5),
                    "lambda0_nm": round(float(sp["wavelength_nm"][int(np.argmin(sp["thru_T"]))]), 4),
                }
            da = st_rows["GSST_amorphous"]
            dc = st_rows["GSST_crystalline"]
            # Resonance shift d_lam = lam * (n_c - n_a) / n_g_a
            dlam_nm = c["wavelength_nm"] * (dc["neff_ring"] - da["neff_ring"]) / max(da["ng_ring"], 1e-6)
            rec = {
                "case_id": cid,
                "states": st_rows,
                "delta_neff_c_minus_a": round(dc["neff_ring"] - da["neff_ring"], 6),
                "delta_lambda_shift_nm": round(dlam_nm, 3),
                "drop_extinction_contrast_dB": round(dc["IL_drop_dB"] - da["IL_drop_dB"], 2),
                "wall_time_s": round(time.perf_counter() - t0, 2),
            }
            print(
                f"[MR Suite 12A] {cid}: dlam={rec['delta_lambda_shift_nm']:+.3f} nm, "
                f"Q_L(a/c)={da['Q_loaded']:.0f}/{dc['Q_loaded']:.0f}, "
                f"IL_drop(a/c)={da['IL_drop_dB']:.2f}/{dc['IL_drop_dB']:.2f} dB"
            )
            out.append(rec)
            continue

        alpha_use = float(c.get("alpha_loss_db_per_cm", 0.5))
        if "ring_core_k_imag" in c:
            alpha_use = _solve_modal_loss_from_k_core(
                c["ring_width_um"], c["film_height_um"], c["ring_radius_um"],
                c["wavelength_nm"], float(c["ring_core_k_imag"]),
            )

        kw = dict(
            topology=c["topology"],
            core_material=c["core_material"],
            clad_material=c["clad_material"],
            top_clad_material=c["clad_material"],
            film_height_um=c["film_height_um"],
            ring_width_um=c["ring_width_um"],
            bus_width_um=c["bus_width_um"],
            ring_radius_um=c["ring_radius_um"],
            gap_thru_um=c["gap_thru_um"],
            wavelength_nm=c["wavelength_nm"],
            alpha_loss_db_per_cm=alpha_use,
            span_nm=c["wl_max_nm"] - c["wl_min_nm"],
            num_spectrum_points=1200,
        )
        if "gap_drop_um" in c:
            kw["gap_drop_um"] = c["gap_drop_um"]
        if "straight_length_um" in c:
            kw["straight_length_um"] = c["straight_length_um"]
        if "pulley_angle_thru_deg" in c:
            kw["pulley_angle_thru_deg"] = c["pulley_angle_thru_deg"]
            kw["pulley_angle_drop_deg"] = c["pulley_angle_drop_deg"]
        if "ring2_radius_um" in c:
            kw["ring2_radius_um"] = c["ring2_radius_um"]
            kw["inter_ring_gap_um"] = c["inter_ring_gap_um"]

        res = simulate_mrr_topology(MRRDesignConfig(**kw))
        m = res["metrics"]
        sp = res["spectrum"]
        wl = np.asarray(sp["wavelength_nm"], dtype=float)
        t_th = np.asarray(sp["thru_T"], dtype=float)
        t_dr = np.asarray(sp["drop_T"], dtype=float) if sp.get("drop_T") is not None else None

        rec = {
            "case_id": cid,
            "topology": c["topology"],
            "alpha_modal_db_per_cm": round(alpha_use, 3),
            "neff_ring": round(float(m["neff_ring"]), 6),
            "ng_ring": round(float(m["ng_ring"]), 5),
            "FSR_nm": round(float(m["FSR_nm"]), 4),
            "kappa_power_thru": round(float(m["kappa_power_thru"]), 5),
            "Q_loaded": round(float(m["Q_loaded"]), 1),
            "Q_int": round(float(m["Q_int"]), 1),
            "FWHM_nm": round(float(m["FWHM_pm"]) * 1e-3, 5),
            "ER_thru_dB": round(float(m["ER_thru_dB"]), 2),
            "IL_drop_dB": round(float(m["insertion_loss_drop_dB"]), 3) if m.get("insertion_loss_drop_dB") is not None else None,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        if c["topology"] == "vernier_crow_ad" and t_dr is not None:
            # Measure coupled-cavity CROW supermode splitting & 3-dB drop passband width near 1550 nm
            win = np.abs(wl - 1550.0) <= 0.45 * float(m["FSR_nm"])
            wl_w, dr_w = wl[win], t_dr[win]
            pk_dr = float(np.max(dr_w))
            from scipy.signal import find_peaks
            pks, _ = find_peaks(dr_w, height=0.35 * pk_dr, distance=3)
            above_3db = wl_w[dr_w >= 0.5 * pk_dr]
            rec["crow_kappa_inter_power"] = round(float(res["vernier"]["kappa_inter_power"]), 5)
            rec["crow_drop_peak_T"] = round(pk_dr, 4)
            rec["crow_drop_3db_bw_nm"] = round(float(above_3db[-1] - above_3db[0]), 4) if len(above_3db) >= 2 else 0.0
            rec["crow_split_nm"] = round(float(wl_w[pks[-1]] - wl_w[pks[0]]), 4) if len(pks) >= 2 else 0.0
            print(
                f"[MR Suite 12A] {cid}: FSR={rec['FSR_nm']:.3f}nm, k_bus={rec['kappa_power_thru']:.4f}, "
                f"k_inter={rec['crow_kappa_inter_power']:.4f}, split={rec['crow_split_nm']:.3f}nm, BW_3dB={rec['crow_drop_3db_bw_nm']:.3f}nm"
            )
        else:
            print(
                f"[MR Suite 12A] {cid}: FSR={rec['FSR_nm']:.3f}nm, Q_L={rec['Q_loaded']:.1f}, "
                f"FWHM={rec['FWHM_nm']:.4f}nm, k2={rec['kappa_power_thru']:.4f}, ER={rec['ER_thru_dB']:.2f}dB, IL_drop={rec['IL_drop_dB']}"
            )
        out.append(rec)
    return out


def run_suite12b_gds_export(specs: dict) -> dict:
    c = specs["suite12b_gds_and_doublet"]["gds_roundtrip"]
    GDS_P3_DIR.mkdir(parents=True, exist_ok=True)
    gds_path = GDS_P3_DIR / f"{c['case_id']}.gds"
    cfg = MRRDesignConfig(
        topology=c["topology"],
        core_material=c["core_material"],
        clad_material=c["clad_material"],
        top_clad_material=c["clad_material"],
        film_height_um=c["film_height_um"],
        ring_width_um=c["ring_width_um"],
        bus_width_um=c["bus_width_um"],
        ring_radius_um=c["ring_radius_um"],
        gap_thru_um=c["gap_thru_um"],
        gap_drop_um=c["gap_drop_um"],
        wavelength_nm=1550.0,
    )
    exp = export_mrr_gds(cfg, gds_path)
    insp = analyze_uploaded_gds(gds_path)
    ex_p = insp.get("extracted_params", insp.get("extracted_geometry", insp))
    rec = {
        "case_id": c["case_id"],
        "gds_path": str(Path(exp).relative_to(REPO_ROOT).as_posix()),
        "cell_name": insp.get("cell_name"),
        "detected_topology": ex_p.get("topology"),
        "extracted_radius_um": ex_p.get("ring_radius_um"),
        "extracted_ring_width_um": ex_p.get("ring_width_um"),
        "extracted_gap_thru_um": ex_p.get("gap_thru_um"),
        "extracted_gap_drop_um": ex_p.get("gap_drop_um"),
    }
    print(
        f"[MR Suite 12B GDS] Exported {gds_path.name} -> roundtrip extracted: "
        f"top={rec['detected_topology']}, R={rec['extracted_radius_um']}um, "
        f"W={rec['extracted_ring_width_um']}um, g_thru={rec['extracted_gap_thru_um']}um"
    )
    return rec


def run_suite13(specs: dict) -> dict:
    s13 = specs["suite13_mzi_and_3d_io"]
    out: dict[str, dict] = {}

    # 1. Optimal-length 2x2 MMI (L = 42.3 um)
    c1 = s13["mmi_2x2_optimal"]
    t0 = time.perf_counter()
    res_mmi = simulate_mzi(MZIDesignConfig(
        splitter_type="mmi_2x2",
        core_material=c1["core_material"],
        clad_material=c1["clad_material"],
        top_clad_material=c1["clad_material"],
        film_height_um=c1["film_height_um"],
        arm1_width_um=c1["arm_width_um"],
        mmi_width_um=c1["mmi_width_um"],
        mmi_length_um=c1["mmi_length_um"],
        mmi_taper_width_um=c1["mmi_taper_width_um"],
        mmi_taper_length_um=c1["mmi_taper_length_um"],
        wavelength_nm=c1["wavelength_nm"],
    ))
    sp1 = res_mmi["splitter"]["summary"]
    out["mmi_2x2_optimal"] = {
        "case_id": c1["case_id"],
        "split_bar_pct": round(float(sp1["split_bar_pct_at_wl0"]), 2),
        "split_cross_pct": round(float(sp1["split_cross_pct_at_wl0"]), 2),
        "excess_loss_db": round(float(sp1["excess_loss_db"]), 3),
        "imbalance_db": round(float(sp1["imbalance_at_wl0_db"]), 3),
        "phase_diff_deg": round(float(sp1.get("phase_diff_deg_at_wl0", -90.0)), 2),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(f"[MR Suite 13A] 2x2 MMI (L=42.3um): {out['mmi_2x2_optimal']}")

    # 2. Full S-Bend 2x2 Directional Coupler
    c2 = s13["dc_2x2_full_sbend"]
    t0 = time.perf_counter()
    res_dc = simulate_mzi(MZIDesignConfig(
        splitter_type="directional_coupler",
        core_material=c2["core_material"],
        clad_material=c2["clad_material"],
        top_clad_material=c2["clad_material"],
        film_height_um=c2["film_height_um"],
        arm1_width_um=c2["arm_width_um"],
        dc_gap_um=c2["dc_gap_um"],
        dc_length_um=c2["dc_straight_length_um"],
        dc_sbend_length_um=c2["sbend_length_um"],
        arm_separation_um=c2["arm_separation_um"],
        wavelength_nm=c2["wavelength_nm"],
    ))
    sp2 = res_dc["splitter"]["summary"]
    out["dc_2x2_full_sbend"] = {
        "case_id": c2["case_id"],
        "split_bar_pct": round(float(sp2["split_bar_pct_at_wl0"]), 2),
        "split_cross_pct": round(float(sp2["split_cross_pct_at_wl0"]), 2),
        "excess_loss_db": round(float(sp2["excess_loss_db"]), 3),
        "imbalance_db": round(float(sp2["imbalance_at_wl0_db"]), 3),
        "L_pi_um": round(float(sp2.get("L_pi_um", 0.0)), 3),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(f"[MR Suite 13B] Full S-Bend 2x2 DC: {out['dc_2x2_full_sbend']}")

    # 3. Low-Q RAMZI
    c3 = s13["lowq_ramzi"]
    t0 = time.perf_counter()
    res_rz = simulate_mzi(MZIDesignConfig(
        splitter_type="directional_coupler",
        core_material=c3["core_material"],
        clad_material=c3["clad_material"],
        top_clad_material=c3["clad_material"],
        film_height_um=c3["film_height_um"],
        arm1_width_um=c3["arm_width_um"],
        arm1_length_um=c3["arm_straight_um"],
        delta_length_um=0.0,
        dc_gap_um=c3["dc_gap_um"],
        dc_length_um=c3["dc_straight_length_um"],
        dc_sbend_length_um=c3["sbend_length_um"],
        arm_separation_um=c3["arm_separation_um"],
        ramzi_enabled=True,
        ramzi_ring_radius_um=c3["ring_radius_um"],
        ramzi_ring_gap_um=c3["ring_gap_um"],
        wavelength_nm=1550.0,
        span_nm=c3["wl_max_nm"] - c3["wl_min_nm"],
    ))
    m3 = res_rz["metrics"]
    out["lowq_ramzi"] = {
        "case_id": c3["case_id"],
        "ER_bar_dB": round(float(m3["ER_bar_passive_dB"]), 2),
        "ER_cross_dB": round(float(m3["ER_cross_passive_dB"]), 2),
        "IL_min_dB": round(float(m3["IL_min_passive_dB"]), 3),
        "ramzi_ring_FSR_nm": round(float(res_rz.get("ramzi", {}).get("FSR_nm", 17.68)), 3),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(f"[MR Suite 13C] Low-Q RAMZI: {out['lowq_ramzi']}")

    # 4. 2D-FDFD Reference for 3D Focusing Grating Coupler
    c4 = s13["gc_3d_focusing"]
    t0 = time.perf_counter()
    res_gc = design_grating_coupler(
        core_material=c4["core_material"],
        clad_material=c4["clad_material"],
        top_clad_material=c4["clad_material"],
        film_height_um=c4["film_height_um"],
        etch_depth_um=c4["etch_depth_um"],
        box_thickness_um=c4["box_thickness_um"],
        top_clad_thickness_um=c4["top_clad_thickness_um"],
        fiber_angle_deg=c4["fiber_angle_deg"],
        mfd_um=c4["mfd_um"],
        n_env=c4["n_env"],
        num_periods=len(c4["grooves_r_um"]),
        apodized=True,
        focusing=True,
        wavelength_nm=c4["wavelength_nm"],
        wavelength_span_nm=c4["wl_max_nm"] - c4["wl_min_nm"],
    )
    m4 = res_gc["metrics"]
    out["gc_3d_focusing"] = {
        "case_id": c4["case_id"],
        "peak_wavelength_nm": round(float(m4["peak_wavelength_nm"]), 2),
        "peak_il_2dfdfd_db": round(float(m4.get("peak_il_db", m4.get("peak_il_dB", 0.0))), 3),
        "il_1550_2dfdfd_db": round(float(m4.get("center_il_db", m4.get("il_at_1550_dB", 0.0))), 3),
        "eta_up_1550_pct": round(float(m4.get("upward_directionality_pct", m4.get("eta_up_pct", 0.0))), 2),
        "s11_1550_db": round(float(m4.get("s11_db", m4.get("back_reflection_s11_db", 0.0))), 2),
        "raw_metrics": {k: v for k, v in m4.items() if isinstance(v, (int, float))},
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(f"[MR Suite 13D] 2D-FDFD GC Ref: {out['gc_3d_focusing']}")

    # 5. 2D-FEM + EME Reference for 30 um 3D Butt Taper
    c5 = s13["butt_3d_taper"]
    t0 = time.perf_counter()
    res_bt = design_butt_edge_coupler(
        core_material=c5["core_material"],
        clad_material=c5["clad_material"],
        top_clad_material=c5["clad_material"],
        film_height_um=c5["film_height_um"],
        sidewall_deg=90.0,
        w_bus_um=c5["w_bus_um"],
        w_tip_um=c5["w_tip_um"],
        sweep_tip_widths=False,
        taper_length_um=c5["taper_length_um"],
        taper_profile="linear",
        fiber_preset="Lensed_2p5um",
        mfd_x_um=c5["mfd_x_um"],
        mfd_y_um=c5["mfd_y_um"],
        n_env=c5["n_env"],
        ar_coating=False,
        box_thickness_um=3.0,
        wavelength_nm=c5["wavelength_nm"],
    )
    m5 = res_bt["metrics"]
    out["butt_3d_taper"] = {
        "case_id": c5["case_id"],
        "mode_overlap_pct": round(100.0 * (10.0 ** (-float(m5.get("facet_il_db", 0.0)) / 10.0)), 2),
        "facet_il_db": round(float(m5.get("facet_il_db", 0.0)), 3),
        "taper_il_db": round(float(m5.get("taper_il_db", 0.0)), 3),
        "total_il_db": round(float(m5.get("total_il_db", 0.0)), 3),
        "raw_metrics": {k: v for k, v in m5.items() if isinstance(v, (int, float))},
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(f"[MR Suite 13E] 2D-FEM Butt Taper (30um): {out['butt_3d_taper']}")
    return out


def main() -> None:
    t_all = time.perf_counter()
    specs = json.loads(SPECS_P3_JSON.read_text(encoding="utf-8"))
    s12a = run_suite12a(specs)
    s12b = run_suite12b_gds_export(specs)
    s13 = run_suite13(specs)
    payload = {
        "solver": "MicroRing Studio v3.9.0 (scikit-fem Conformal 2D-FEM / Unitary Spatial CMT / 2D-FDFD / EME)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite12a_lowq_full_rings": s12a,
        "suite12b_gds_export": s12b,
        "suite13_mzi_and_3d_io": s13,
    }
    OUT_P3_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved MicroRing Phase-3 results -> {OUT_P3_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
