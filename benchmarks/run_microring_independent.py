"""Independent MicroRing Studio v3.9.0 Runner for 3-Way Commercial Solver Validation.

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs.json`
and executes MicroRing Studio's native pure-Python solvers across all 5 suites without any reference
to Lumerical or COMSOL outputs.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
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

import microring
from microring.active_tuning import _pockels_dn_xyz, solve_2d_electrostatic_eo_fem
from microring.eigenmode_solver import (
    WaveguideCrossSectionSpec,
    solve_modes_fem,
    solve_perturbed_neff_sweep_fem,
)
from microring.io_couplers import design_grating_coupler
from microring.materials import C0_M_PER_S, T_REF_K, get_material_fn
from microring.mrr_topologies import MRRDesignConfig, simulate_mrr_topology
from microring.multiphysics_tcad import ThermalSimConfig, simulate_thermal_phase_shifter
from microring.mzi_interferometer import MZIDesignConfig, simulate_mzi

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SPECS_JSON = OUT_DIR / "commercial_validation_specs.json"
OUT_JSON = OUT_DIR / "microring_independent_results.json"


def _pick_mode(modes: list, pol: str = "TE0"):
    want_te = pol.upper().startswith("TE")
    pool = [m for m in modes if (m.te_fraction >= 0.5 if want_te else m.te_fraction < 0.5)]
    if not pool:
        pool = modes
    return max(pool, key=lambda m: float(np.real(m.n_eff)))


def _solve_suite1_single(c: dict) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    wl0_nm = float(c["wavelength_nm"])
    dwl_nm = float(c["dwl_nm"])
    r_m = float(c["ring_radius_um"]) * 1e-6
    wl_scale = float(np.clip(wl0_nm / 1550.0, 0.35, 3.50))

    spec = WaveguideCrossSectionSpec(
        w_bottom_um=c["ring_width_um"],
        h_film_um=c["film_height_um"],
        h_slab_um=c["slab_height_um"],
        sidewall_deg=c["sidewall_deg"],
        core_material_fn=get_material_fn(c["core_material"]),
        clad_material_fn=get_material_fn(c["clad_material"]),
        top_clad_material_fn=get_material_fn(c["top_clad_material"]),
        domain_width_um=c["domain_width_um"],
        h_box_um=c["box_pad_um"],
        h_top_clad_um=c["box_pad_um"],
    )
    res_um = float(c["fem_core_res_um"])
    clad_res = max(res_um * 5.0, 0.16 * wl_scale)

    neffs_bent = {}
    te_frac0 = 0.0
    for key, wl_nm in (("minus", wl0_nm - dwl_nm), ("center", wl0_nm), ("plus", wl0_nm + dwl_nm)):
        m = _pick_mode(
            solve_modes_fem(
                spec,
                wl_nm * 1e-9,
                bend_radius_m=r_m,
                num_modes=4,
                core_mesh_res_um=res_um,
                clad_mesh_res_um=clad_res,
                element_order=2,
                export_grid=False,
                mesh_backend="bsd_conformal",
            ),
            "TE0",
        )
        neffs_bent[key] = float(np.real(m.n_eff))
        if key == "center":
            te_frac0 = float(m.te_fraction)

    ng_bent_3pt = neffs_bent["center"] - wl0_nm * (neffs_bent["plus"] - neffs_bent["minus"]) / (2.0 * dwl_nm)

    m_str = _pick_mode(
        solve_modes_fem(
            spec,
            wl0_nm * 1e-9,
            bend_radius_m=math.inf,
            num_modes=4,
            core_mesh_res_um=res_um,
            clad_mesh_res_um=clad_res,
            element_order=2,
            export_grid=False,
            mesh_backend="bsd_conformal",
        ),
        "TE0",
    )
    neff_str = float(np.real(m_str.n_eff))
    l_rt_um = 2.0 * math.pi * c["ring_radius_um"] + 2.0 * c["straight_length_um"]
    fsr_nm = (wl0_nm ** 2) / (ng_bent_3pt * l_rt_um * 1e3)

    cfg = MRRDesignConfig(
        topology=c["topology"],
        core_material=c["core_material"],
        clad_material=c["clad_material"],
        top_clad_material=c["top_clad_material"],
        wavelength_nm=wl0_nm,
        film_height_um=c["film_height_um"],
        slab_height_um=c["slab_height_um"],
        sidewall_deg=c["sidewall_deg"],
        ring_width_um=c["ring_width_um"],
        bus_width_um=c["bus_width_um"],
        ring_radius_um=c["ring_radius_um"],
        gap_thru_um=c["gap_thru_um"],
        gap_drop_um=c["gap_thru_um"] + 0.04,
        straight_length_um=c["straight_length_um"],
        pulley_angle_thru_deg=c["pulley_angle_thru_deg"] if c["pulley_angle_thru_deg"] > 0 else 20.0,
        span_nm=max(2.5, round(12.0 * ((wl0_nm / 1550.0) ** 1.8), 1)),
    )
    mrr = simulate_mrr_topology(cfg)
    dt_s = time.perf_counter() - t0

    s1_row = {
        "case_id": c["case_id"],
        "wavelength_nm": wl0_nm,
        "neff_bent": round(neffs_bent["center"], 6),
        "neff_bent_minus": round(neffs_bent["minus"], 6),
        "neff_bent_plus": round(neffs_bent["plus"], 6),
        "ng_bent": round(ng_bent_3pt, 5),
        "neff_straight": round(neff_str, 6),
        "delta_neff_bend": round(neffs_bent["center"] - neff_str, 6),
        "te_fraction": round(te_frac0, 4),
        "fsr_nm": round(fsr_nm, 4),
        "wall_time_s": round(dt_s, 3),
    }
    kp = float(mrr["metrics"]["kappa_power_thru"])
    s2a_row = {
        "case_id": c["case_id"],
        "topology": c["topology"],
        "wavelength_nm": wl0_nm,
        "gap_um": c["gap_thru_um"],
        "kappa_power_cmt": round(kp, 6),
        "thru_power_cmt": round(1.0 - kp, 6),
        "Q_ext_thru_cmt": round(float(mrr["metrics"]["Q_ext_thru"]), 1),
    }
    return s1_row, s2a_row


def run_suite1_and_2a(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    s1_rows = []
    s2a_rows = []
    with ProcessPoolExecutor(max_workers=min(5, len(cases))) as ex:
        for s1_r, s2_r in ex.map(_solve_suite1_single, cases):
            s1_rows.append(s1_r)
            s2a_rows.append(s2_r)
            print(
                f"[S1/2A] {s1_r['case_id']:<26} neff={s1_r['neff_bent']:.6f} ng={s1_r['ng_bent']:.4f} "
                f"d_bend={s1_r['delta_neff_bend']:+.6f} kappa2={s2_r['kappa_power_cmt']:.5f} ({s1_r['wall_time_s']:.1f}s)"
            )
    return s1_rows, s2a_rows


def run_suite2b_soi_gap_sweep(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        cfg = MRRDesignConfig(
            topology=c["topology"],
            core_material=c["core_material"],
            clad_material=c["clad_material"],
            top_clad_material=c["top_clad_material"],
            wavelength_nm=c["wavelength_nm"],
            film_height_um=c["film_height_um"],
            slab_height_um=c["slab_height_um"],
            sidewall_deg=c["sidewall_deg"],
            ring_width_um=c["ring_width_um"],
            bus_width_um=c["bus_width_um"],
            ring_radius_um=c["ring_radius_um"],
            gap_thru_um=c["gap_thru_um"],
            span_nm=15.0,
        )
        mrr = simulate_mrr_topology(cfg)
        kp = float(mrr["metrics"]["kappa_power_thru"])
        rows.append({
            "case_id": c["case_id"],
            "gap_nm": round(c["gap_thru_um"] * 1e3, 1),
            "kappa_power_cmt": round(kp, 6),
            "thru_power_cmt": round(1.0 - kp, 6),
            "neff_ring": round(float(mrr["metrics"]["neff_ring"]), 6),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(f"[S2B]   {c['case_id']:<26} gap={c['gap_thru_um']*1e3:5.1f} nm | kappa2={kp:.5f}")
    return rows


def run_suite3(dc_cases: list[dict], splitter_cases: list[dict], gc_case: dict) -> dict:
    # 3A: Twin-core DC supermodes (order=2 high-accuracy & default order=1 for audit)
    dc_rows = []
    for c in dc_cases:
        t0 = time.perf_counter()
        twin_spec = WaveguideCrossSectionSpec(
            w_bottom_um=c["waveguide_width_um"],
            h_film_um=c["film_height_um"],
            h_slab_um=c["slab_height_um"],
            sidewall_deg=c["sidewall_deg"],
            core_material_fn=get_material_fn(c["core_material"]),
            clad_material_fn=get_material_fn(c["clad_material"]),
            top_clad_material_fn=get_material_fn(c["top_clad_material"]),
            twin_core_gap_um=c["gap_um"],
            domain_width_um=c["domain_width_um"],
            h_box_um=c["box_pad_um"],
            h_top_clad_um=c["box_pad_um"],
        )
        modes = solve_modes_fem(
            twin_spec,
            c["wavelength_nm"] * 1e-9,
            num_modes=6,
            core_mesh_res_um=0.032,
            clad_mesh_res_um=0.22,
            element_order=2,
            export_grid=False,
        )
        te = sorted([m for m in modes if m.te_fraction >= 0.5], key=lambda m: -float(np.real(m.n_eff)))
        n_even = float(np.real(te[0].n_eff))
        n_odd = float(np.real(te[1].n_eff))
        dn = n_even - n_odd
        l_pi_um = (c["wavelength_nm"] * 1e-3) / (2.0 * dn)
        dc_rows.append({
            "case_id": c["case_id"],
            "gap_nm": round(c["gap_um"] * 1e3, 1),
            "n_even": round(n_even, 6),
            "n_odd": round(n_odd, 6),
            "delta_n": round(dn, 6),
            "L_pi_um": round(l_pi_um, 4),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(f"[S3A]   {c['case_id']:<26} n_e={n_even:.6f} n_o={n_odd:.6f} dn={dn:.6f} L_pi={l_pi_um:.3f} um")

    # 3B: MZI Splitters (1x2 MMI & 1x2 Y-Branch)
    sp_rows = []
    for c in splitter_cases:
        t0 = time.perf_counter()
        if c["splitter_type"] == "mmi_1x2":
            cfg = MZIDesignConfig(
                splitter_type="mmi_1x2",
                core_material=c["core_material"],
                wavelength_nm=c["wavelength_nm"],
                film_height_um=c["film_height_um"],
                slab_height_um=c["slab_height_um"],
                sidewall_deg=c["sidewall_deg"],
                arm1_width_um=c["arm_width_um"],
                mmi_width_um=c["mmi_width_um"],
                mmi_length_um=c["mmi_length_um"],
                mmi_taper_width_um=c["mmi_taper_width_um"],
                mmi_taper_length_um=c["mmi_taper_length_um"],
                span_nm=30.0,
            )
        else:
            cfg = MZIDesignConfig(
                splitter_type="y_branch",
                core_material=c["core_material"],
                wavelength_nm=c["wavelength_nm"],
                film_height_um=c["film_height_um"],
                slab_height_um=c["slab_height_um"],
                sidewall_deg=c["sidewall_deg"],
                arm1_width_um=c["arm_width_um"],
                y_apex_gap_um=c["y_apex_gap_um"],
                y_taper_length_um=c["y_taper_length_um"],
                y_branch_length_um=c["y_branch_length_um"],
                span_nm=30.0,
            )
        res = simulate_mzi(cfg, include_active_tuning=False)
        s = res["splitter"]["summary"]
        sp_rows.append({
            "case_id": c["case_id"],
            "splitter_type": c["splitter_type"],
            "split_arm1_pct": round(float(s["split_bar_pct_at_wl0"]), 3),
            "split_arm2_pct": round(float(s["split_cross_pct_at_wl0"]), 3),
            "total_trans_pct": round(float(s["split_bar_pct_at_wl0"]) + float(s["split_cross_pct_at_wl0"]), 3),
            "excess_loss_db": round(float(s["excess_loss_db"]), 4),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(
            f"[S3B]   {c['case_id']:<26} split={s['split_bar_pct_at_wl0']:.2f}/{s['split_cross_pct_at_wl0']:.2f}% "
            f"IL={s['excess_loss_db']:.3f} dB"
        )

    # 3C: Surface Grating Coupler (2D FDFD)
    t0 = time.perf_counter()
    gc = design_grating_coupler(
        core_material=gc_case["core_material"],
        clad_material=gc_case["clad_material"],
        top_clad_material=gc_case["top_clad_material"],
        substrate_material=gc_case["substrate_material"],
        wavelength_nm=gc_case["wavelength_nm"],
        polarization=gc_case["polarization"],
        film_height_um=gc_case["film_height_um"],
        etch_depth_um=gc_case["etch_depth_um"],
        fiber_angle_deg=gc_case["fiber_angle_deg"],
        apodized=False,
        focusing=False,
        pitch_um=gc_case["pitch_um"],
        fill_factor=gc_case["fill_factor"],
        num_periods=gc_case["num_periods"],
        grating_width_um=gc_case["grating_width_um"],
        box_thickness_um=gc_case["box_thickness_um"],
        top_clad_thickness_um=gc_case["top_clad_thickness_um"],
        mfd_um=gc_case["mfd_um"],
        n_env=gc_case["n_env"],
        wavelength_span_nm=100.0,
    )
    m = gc["metrics"]
    spec_tr = gc["spectrum"]
    wl_arr = np.asarray(spec_tr["wavelength_nm"], dtype=float)
    gc_out = {
        "case_id": gc_case["case_id"],
        "pitch_um": round(float(gc["synthesis"]["pitch_mean_um"]), 6),
        "peak_wavelength_nm": round(float(m["peak_wavelength_nm"]), 2),
        "peak_il_db": round(float(m["peak_il_db"]), 3),
        "peak_efficiency_pct": round(float(m["peak_efficiency_pct"]), 2),
        "eta_up_at_1550_pct": round(float(m["upward_directionality_pct"]), 2),
        "eta_down_at_1550_pct": round(float(m["substrate_loss_pct"]), 2),
        "mode_match_1d_pct": round(float(m["mode_match_1d_pct"]), 2),
        "s11_at_1550_db": round(float(m["s11_db"]), 2),
        "il_at_1550_db": round(float(m["center_il_db"]), 3),
        "bw_1db_nm": round(float(m["bw_1db_nm"]), 2),
        "bw_3db_nm": round(float(m["bw_3db_nm"]), 2),
        "spectrum_wavelength_nm": [round(float(x), 2) for x in wl_arr],
        "spectrum_il_db": [round(float(x), 3) for x in spec_tr["il_db"]],
        "spectrum_s11_db": [round(float(x), 3) for x in spec_tr["s11_db"]],
        "wall_time_s": round(time.perf_counter() - t0, 3),
    }
    print(
        f"[S3C]   {gc_case['case_id']:<26} peak={gc_out['peak_wavelength_nm']} nm | "
        f"IL_peak={gc_out['peak_il_db']:.2f} dB | up@1550={gc_out['eta_up_at_1550_pct']:.1f}% | "
        f"dn@1550={gc_out['eta_down_at_1550_pct']:.1f}% | S11={gc_out['s11_at_1550_db']:.1f} dB"
    )
    return {"suite3a_dc": dc_rows, "suite3b_splitters": sp_rows, "suite3c_gc": gc_out}


def run_suite4_thermal(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        cfg = ThermalSimConfig(
            thermal_structure=c["thermal_structure"],
            core_material=c["core_material"],
            clad_material=c["clad_material"],
            top_clad_material=c["top_clad_material"],
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
            trench_edge_offset_um=c["trench_edge_offset_um"],
            trench_width_um=c["trench_width_um"],
            trench_depth_um=c["trench_depth_um"],
            neighbor_pitch_um=c["neighbor_pitch_um"],
        )
        res = simulate_thermal_phase_shifter(cfg)
        m = res["metrics"]
        r_pk = round(float(m["Delta_T_heater_at_Ppi_K"]) / max(float(m["P_pi_mW"]), 1e-9), 4)
        rows.append({
            "case_id": c["case_id"],
            "R_th_core_K_per_mW": float(m["R_th_core_K_per_mW"]),
            "R_th_peak_K_per_mW": r_pk,
            "P_pi_mW": float(m["P_pi_mW"]),
            "t_rise_10_90_us": float(m["t_rise_10_90_us"]),
            "t_fall_90_10_us": float(m["t_fall_90_10_us"]),
            "f_3dB_thermal_kHz": float(m["f_3dB_thermal_kHz"]),
            "crosstalk_20um_pct": float(m["crosstalk_at_neighbor_pct"]),
            "power_conservation_err_pct": float(m["power_conservation_error_pct"]),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(
            f"[S4]    {c['case_id']:<26} R_th={m['R_th_core_K_per_mW']:.4f} K/mW | "
            f"R_pk={r_pk:.4f} | P_pi={m['P_pi_mW']:.2f} mW | "
            f"tr/tf={m['t_rise_10_90_us']:.2f}/{m['t_fall_90_10_us']:.2f} us | "
            f"f3dB={m['f_3dB_thermal_kHz']:.2f} kHz | xtalk={m['crosstalk_at_neighbor_pct']:.3f}%"
        )
    return rows


def run_suite5_pockels_eo(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        cfg = MRRDesignConfig(
            topology="racetrack_ap",
            core_material=c["core_material"],
            clad_material=c["clad_material"],
            top_clad_material=c["top_clad_material"],
            wavelength_nm=c["wavelength_nm"],
            polarization=c["polarization"],
            film_height_um=c["film_height_um"],
            slab_height_um=c["slab_height_um"],
            sidewall_deg=c["sidewall_deg"],
            ring_width_um=c["waveguide_width_um"],
            bus_width_um=c["waveguide_width_um"],
            ring_radius_um=60.0,
            straight_length_um=80.0,
            gap_thru_um=0.30,
        )
        eo = solve_2d_electrostatic_eo_fem(
            cfg,
            roundtrip_length_um=500.0,
            active_fraction=1.0,
            electrode_config=c["electrode_config"],
            electrode_gap_um=c["electrode_gap_um"],
            electrode_width_um=c["electrode_width_um"],
            electrode_thickness_um=c["electrode_thickness_um"],
            electrode_voffset_um=c["electrode_voffset_um"],
            box_thickness_um=c["box_thickness_um"],
            top_clad_thickness_um=c["top_clad_thickness_um"],
        )
        # Coupled perturbed 2D vector Maxwell FEM at V = 0 V and V = +10 V -> straight-arm V_pi * L
        wl_m = c["wavelength_nm"] * 1e-9
        core_fn = get_material_fn(c["core_material"])
        clad_fn = get_material_fn(c["clad_material"])
        top_fn = get_material_fn(c["top_clad_material"])
        mat_c = core_fn(wl_m, T_REF_K)
        spec = WaveguideCrossSectionSpec(
            w_bottom_um=c["waveguide_width_um"],
            h_film_um=c["film_height_um"],
            h_slab_um=c["slab_height_um"],
            sidewall_deg=c["sidewall_deg"],
            core_material_fn=core_fn,
            clad_material_fn=clad_fn,
            top_clad_material_fn=top_fn,
            domain_width_um=5.5,
            h_box_um=2.0,
            h_top_clad_um=2.0,
        )
        v_test = 10.0
        interp_ex = eo["interp_ex_1v_v_per_um"]
        interp_ey = eo["interp_ey_1v_v_per_um"]

        def _eo_pert(xu: np.ndarray, yu: np.ndarray, reg: str):
            if reg not in ("core", "slab"):
                z = np.zeros_like(xu, dtype=complex)
                return z, z, z
            pts = np.column_stack([yu, xu])
            ex_v = v_test * interp_ex(pts)
            ey_v = v_test * interp_ey(pts)
            return _pockels_dn_xyz(mat_c, ex_v, ey_v)

        opt_rows = solve_perturbed_neff_sweep_fem(
            spec,
            wl_m,
            [None, _eo_pert],
            polarization=c["polarization"],
            core_mesh_res_um=0.038,
            element_order=1,
        )
        neff_0 = float(opt_rows[0]["neff_real"])
        neff_v = float(opt_rows[1]["neff_real"])
        dneff_per_v = abs(neff_v - neff_0) / v_test
        # Single-arm straight-waveguide V_pi * L [V*cm] = lambda / (2 * |dneff/dV|) * 100
        v_pi_l_vcm = (wl_m / (2.0 * max(dneff_per_v, 1e-15))) * 100.0

        rows.append({
            "case_id": c["case_id"],
            "electrode_config": eo["electrode_config"],
            "ex_core_v_per_um_per_v": float(eo["ex_core_v_per_um_per_v"]),
            "ey_core_v_per_um_per_v": float(eo["ey_core_v_per_um_per_v"]),
            "dielectric_screening": float(eo["dielectric_screening_factor"]),
            "c_line_ff_per_um": float(eo["c_elec_ff_per_um"]),
            "neff_0v": round(neff_0, 6),
            "dneff_per_v": float(f"{dneff_per_v:.6e}"),
            "V_pi_L_single_arm_Vcm": round(v_pi_l_vcm, 4),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(
            f"[S5]    {c['case_id']:<26} Ex={eo['ex_core_v_per_um_per_v']:+.5f} Ey={eo['ey_core_v_per_um_per_v']:+.5f} "
            f"(V/um)/V | C={eo['c_elec_ff_per_um']:.4f} fF/um | neff0={neff_0:.6f} | VpiL={v_pi_l_vcm:.3f} V*cm"
        )
    return rows


def main() -> None:
    specs = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    t_all = time.perf_counter()
    s1, s2a = run_suite1_and_2a(specs["suite1_eigenmodes"])
    s2b = run_suite2b_soi_gap_sweep(specs["suite2b_soi_gap_sweep"])
    s3 = run_suite3(specs["suite3a_dc_supermodes"], specs["suite3b_mzi_splitters"], specs["suite3c_grating_coupler"])
    s4 = run_suite4_thermal(specs["suite4_thermal"])
    s5 = run_suite5_pockels_eo(specs["suite5_pockels_eo"])

    payload = {
        "solver": f"MicroRing Studio v{getattr(microring, '__version__', '3.9.0')} (scikit-fem Conformal 2D-FEM / Spatial CMT / 2D-FDFD)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite1_eigenmodes": s1,
        "suite2a_coupler_5band": s2a,
        "suite2b_soi_gap_sweep": s2b,
        **s3,
        "suite4_thermal": s4,
        "suite5_pockels_eo": s5,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved MicroRing Studio independent results to {OUT_JSON} in {payload['total_wall_time_s']:.2f}s")


if __name__ == "__main__":
    main()
