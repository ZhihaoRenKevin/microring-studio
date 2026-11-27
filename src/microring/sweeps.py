"""Automated Waveguide Dispersion & MRR Parameter Sweep Engine (`microring.sweeps`).

Provides two high-level design sweeps used by both Python cookbook scripts and
the Conversational Photonics Copilot UI:
  1. `run_waveguide_width_sweep`: Sweeps waveguide width `W` on a given wafer stack
     (`Material`, `H_film`, `H_slab`, `Sidewall`, `Cladding`, `Wavelength`) to find
     `TE0`/`TM0` effective & group indices (`n_eff`, `n_g`), modal confinement,
     and the exact **single-mode cutoff width** (`TE1`/`TM1` onset).
  2. `run_mrr_parameter_sweep`: Sweeps any resonator or coupler parameter (`gap_thru_um`,
     `pulley_angle_thru_deg`, `straight_length_um`, `ring_radius_um`, `ring_width_um`,
     `delta_temp_k`, `voltage_v`) and returns `Q_loaded`, `Q_ext`, `FSR`, `ER`, and
     critical-coupling ratio `C_p`.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .eigenmode_solver import WaveguideCrossSectionSpec, solve_modes_fem
from .materials import T_REF_K, get_material_fn
from .mrr_topologies import MRRDesignConfig, simulate_mrr_topology


def run_waveguide_width_sweep(
    *,
    core_material: str = "Si3N4",
    clad_material: str = "SiO2",
    top_clad_material: str = "SiO2",
    h_film_um: float = 0.40,
    h_slab_um: float = 0.0,
    sidewall_deg: float = 85.0,
    wavelength_nm: float = 1550.0,
    w_min_um: float = 0.50,
    w_max_um: float = 2.00,
    num_points: int = 8,
    polarization: str = "TE0",
    sc_fraction: float = 0.20,
    core_mesh_res_um: float = 0.050,
    custom_n: float | None = None,
    custom_k: float = 0.0,
    custom_clad_n: float | None = None,
    custom_top_clad_n: float | None = None,
) -> dict:
    """Sweep waveguide width to map single-mode cutoff, n_eff(W), n_g(W) & confinement."""
    wl_m = float(wavelength_nm) * 1e-9
    core_fn = get_material_fn(
        core_material,
        sc_fraction=sc_fraction if core_material.startswith("ScAlN") else None,
        custom_n=custom_n,
        custom_k=custom_k,
    )
    clad_fn = get_material_fn(
        "Custom" if custom_clad_n is not None else clad_material,
        custom_n=custom_clad_n,
    )
    top_fn = get_material_fn(
        "Custom" if custom_top_clad_n is not None else top_clad_material,
        custom_n=custom_top_clad_n,
    )

    n_clad_max = max(clad_fn(wl_m, T_REF_K).nx, top_fn(wl_m, T_REF_K).nx)
    guided_margin = n_clad_max + 0.012

    widths = np.linspace(float(w_min_um), float(w_max_um), max(3, int(num_points)))
    want_te = polarization.upper().startswith("TE")

    neff_fund: list[float] = []
    ng_fund: list[float] = []
    neff_ho: list[float | None] = []
    gamma_list: list[float] = []
    aeff_list: list[float] = []
    single_mode_flags: list[bool] = []

    dl_m = max(wl_m * 4e-4, 0.25e-9)
    wl_um = wl_m * 1e6
    wl_scale = float(np.clip(wavelength_nm / 1550.0, 0.32, 4.50))
    eff_mesh_um = (
        round(float(np.clip(0.050 * wl_scale, 0.016, 0.160)), 4)
        if abs(core_mesh_res_um - 0.050) < 1e-6
        else float(core_mesh_res_um)
    )
    box_pad_um = max(1.8, 1.35 * wl_um) if abs(wl_scale - 1.0) >= 0.05 else 2.5

    for w_um in widths:
        spec = WaveguideCrossSectionSpec(
            w_bottom_um=float(w_um),
            h_film_um=float(h_film_um),
            h_slab_um=float(h_slab_um),
            sidewall_deg=float(sidewall_deg),
            core_material_fn=core_fn,
            clad_material_fn=clad_fn,
            top_clad_material_fn=top_fn,
            domain_width_um=max(3.4 * wl_scale, float(w_um) + 2.4 * wl_um),
            h_box_um=box_pad_um,
            h_top_clad_um=box_pad_um,
        )
        modes = solve_modes_fem(
            spec,
            wl_m,
            bend_radius_m=math.inf,
            num_modes=4,
            core_mesh_res_um=eff_mesh_um,
            clad_mesh_res_um=max(eff_mesh_um * 5.0, 0.14 * wl_scale),
            export_grid=False,
            mesh_backend="bsd_conformal",
        )
        same_pol = [m for m in modes if (m.te_fraction >= 0.5) == want_te]
        same_pol.sort(key=lambda m: -float(np.real(m.n_eff)))

        m0 = same_pol[0] if same_pol else modes[0]
        n0 = float(np.real(m0.n_eff))
        neff_fund.append(round(n0, 6))
        gamma_list.append(round(m0.core_confinement, 4))
        aeff_list.append(round(m0.a_eff_um2, 4))

        # Check first higher-order mode of same polarization
        if len(same_pol) >= 2 and float(np.real(same_pol[1].n_eff)) > guided_margin:
            neff_ho.append(round(float(np.real(same_pol[1].n_eff)), 6))
            single_mode_flags.append(False)
        else:
            neff_ho.append(None)
            single_mode_flags.append(True)

        # Group index at this width (warm-started with n_guess=n0 for fast ARPACK convergence)
        m_m = solve_modes_fem(spec, wl_m - dl_m, num_modes=2, n_guess=n0, core_mesh_res_um=eff_mesh_um, clad_mesh_res_um=max(eff_mesh_um * 5.0, 0.14 * wl_scale), export_grid=False)
        m_p = solve_modes_fem(spec, wl_m + dl_m, num_modes=2, n_guess=n0, core_mesh_res_um=eff_mesh_um, clad_mesh_res_um=max(eff_mesh_um * 5.0, 0.14 * wl_scale), export_grid=False)
        pm = [x for x in m_m if (x.te_fraction >= 0.5) == want_te] or m_m
        pp = [x for x in m_p if (x.te_fraction >= 0.5) == want_te] or m_p
        dn_dl = (float(np.real(pp[0].n_eff)) - float(np.real(pm[0].n_eff))) / (2.0 * dl_m)
        ng_fund.append(round(float(n0 - wl_m * dn_dl), 6))

    # Determine single-mode upper bound & recommended widths
    sm_widths = [float(w) for w, sm, nf in zip(widths, single_mode_flags, neff_fund) if sm and nf > guided_margin]
    if sm_widths:
        sm_max_um = max(sm_widths)
        rec_bus_w = round(0.88 * sm_max_um, 3)
        rec_ring_w = round(min(1.12 * sm_max_um, float(w_max_um)), 3)
    else:
        sm_max_um = float(widths[0])
        rec_bus_w = round(float(widths[len(widths) // 3]), 3)
        rec_ring_w = round(float(widths[len(widths) // 2]), 3)

    return {
        "sweep_type": "waveguide_width",
        "polarization": polarization,
        "n_clad_cutoff": round(n_clad_max, 5),
        "widths_um": np.round(widths, 4).tolist(),
        "neff_fundamental": neff_fund,
        "ng_fundamental": ng_fund,
        "neff_higher_order": neff_ho,
        "core_confinement": gamma_list,
        "a_eff_um2": aeff_list,
        "is_single_mode": single_mode_flags,
        "single_mode_max_width_um": round(sm_max_um, 3),
        "recommended_bus_width_um": rec_bus_w,
        "recommended_ring_width_um": rec_ring_w,
    }


def run_mrr_parameter_sweep(
    base_cfg: MRRDesignConfig,
    param_name: str,
    values: list[float] | np.ndarray,
) -> dict:
    """Sweep any `MRRDesignConfig` field and tabulate optical resonator metrics."""
    allowed = {
        "gap_thru_um": "um",
        "gap_drop_um": "um",
        "inter_ring_gap_um": "um",
        "pulley_angle_thru_deg": "deg",
        "straight_length_um": "um",
        "ring_radius_um": "um",
        "ring_width_um": "um",
        "alpha_loss_db_per_cm": "dB/cm",
        "delta_temp_k": "K",
        "voltage_v": "V",
    }
    if param_name not in allowed:
        raise ValueError(f"Unsupported sweep parameter '{param_name}'. Choose from {list(allowed)}")

    vals = [float(v) for v in values]
    q_loaded_arr, q_int_arr, q_ext_arr, fsr_nm_arr, er_db_arr, cp_arr, shift_pm_arr = [], [], [], [], [], [], []

    for v in vals:
        cfg_i = replace(base_cfg, **{param_name: v})
        res = simulate_mrr_topology(cfg_i)
        m = res["metrics"]
        q_loaded_arr.append(m["Q_loaded"])
        q_int_arr.append(m["Q_int"])
        q_ext_arr.append(m["Q_ext_thru"])
        fsr_nm_arr.append(m["FSR_nm"])
        er_db_arr.append(m["extinction_ratio_thru_dB"])
        cp_arr.append(m["coupling_ratio_Cp"])
        shift_pm_arr.append(res["tuning"]["resonance_shift_pm"])

    best_idx = int(np.argmax(er_db_arr))
    return {
        "sweep_type": "mrr_parameter",
        "param_name": param_name,
        "param_unit": allowed[param_name],
        "unit": allowed[param_name],
        "values": [round(v, 5) for v in vals],
        "Q_loaded": q_loaded_arr,
        "Q_int": q_int_arr,
        "Q_ext_thru": q_ext_arr,
        "FSR_nm": fsr_nm_arr,
        "extinction_ratio_dB": er_db_arr,
        "ER_thru_dB": er_db_arr,
        "coupling_ratio_Cp": cp_arr,
        "resonance_shift_pm": shift_pm_arr,
        "best_critical_coupling_index": best_idx,
        "best_extinction_value": round(vals[best_idx], 4),
        "best_extinction_dB": round(er_db_arr[best_idx], 2),
    }
