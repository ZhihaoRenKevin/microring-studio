"""First-Principles Non-Volatile Phase-Change Material (`PCM`: `Sb2Se3`, `Sb2S3`, `GSST`, `GST`, `VO2`)
Photonic MRR Switching & 2D Transient Electro-Thermal `SET`/`RESET` Pulse FEM (`microring.pcm_photonics`).

100% pure-Python, zero commercial solver dependencies:
  1. **Multi-Level Optical Phase/Amplitude State Solver (`simulate_pcm_mrr_switch`)**:
     - Uses Lorentz-Lorenz Effective Medium Theory across intermediate crystallization fractions
       `X in [0, 1]` (2-bit to 4-bit non-volatile photonic memory / synaptic weights).
     - Solves exact **2D Conformal Vector Maxwell FEM** (`WaveguideCrossSectionSpec` conformal
       `cap` PCM patch on Si / Si3N4 / Ge waveguides from 1310 nm to Mid-IR 7.5 um) extracting
       complex `n_eff(X) + i*k_eff(X)`, phase shift `Delta_phi(X)`, excess loss `IL_a / IL_c [dB]`,
       FOM `Delta_n / Delta_k`, non-volatile MRR resonance shift `Delta_lambda(X)`, extinction
       ratio `ER [dB]`, and `N`-level add-drop weight mapping.
  2. **2D Crank-Nicolson Transient Electro-Thermal `SET`/`RESET` Pulse FEM (`simulate_pcm_thermal_pulse`)**:
     - Assembles exact 2D thermal stiffness `K_th` and heat-capacity mass `M_th` (`scikit-fem`
       `ElementTriP1`) across waveguide + PCM cell + `Al2O3` cap + `TiN`/`PIN` microheater + `BOX`
       + Si substrate.
     - Solves implicit Crank-Nicolson transient pulse waveforms `T_PCM(t)` for BOTH:
       * **RESET (Melt-Quench Amorphization)**: checks `T_peak > T_melt` AND computes exact
         quench cooling rate `|dT/dt|` across `[T_m, T_c]` vs critical amorphization threshold!
       * **SET (Crystallization)**: integrates JMAK Arrhenius kinetics `X(t) = 1 - exp(-(K(T)*t)^n)`
         for partial or full crystallization and computes exact switching energies `E_RESET / E_SET [nJ]`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import gdstk
import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTriP0, ElementTriP1, LinearForm, condense

from .active_tuning import (
    THERMAL_CONDUCTIVITY_W_MK,
    VOLUMETRIC_HEAT_CAPACITY_J_M3K,
    _rectilinear_mesh_tri,
)


def _graded_1d_axis(stops: list[float], steps: list[float]) -> np.ndarray:
    """Build sorted 1D axis passing exactly through every material interface stop."""
    pts: list[float] = []
    for i in range(len(stops) - 1):
        a, b = float(stops[i]), float(stops[i + 1])
        if b <= a + 1e-9:
            continue
        n_seg = max(2, int(math.ceil((b - a) / max(float(steps[i]), 1e-4))))
        seg = np.linspace(a, b, n_seg, endpoint=(i == len(stops) - 2))
        pts.extend(seg.tolist())
    return np.unique(np.round(np.asarray(pts, dtype=float), 6))
from .eigenmode_solver import WaveguideCrossSectionSpec, select_target_polarization_mode, solve_modes_fem
from .gds_bridge import MICRORING_LAYERS, write_klayout_layer_props
from .materials import C0_M_PER_S, MaterialIndexTensor, T_REF_K, get_material_fn

KB_EV_PER_K = 8.617333262e-5

# ==============================================================================
# 1. Peer-Reviewed Optical & Thermal PCM Catalog (NIR 1550 nm + Mid-IR)
# ==============================================================================

PCM_MATERIAL_CATALOG: dict[str, dict[str, Any]] = {
    "Sb2Se3": {  # Delaney Sci. Adv. 2020; Fang Nat. Nanotechnol. 2022
        "n_a_1550": 3.285, "k_a_1550": 1.0e-5,
        "n_c_1550": 4.050, "k_c_1550": 1.0e-4,
        "n_a_midir": 3.180, "k_a_midir": 1.0e-5,
        "n_c_midir": 3.880, "k_c_midir": 5.0e-5,
        "t_cryst_k": 473.0, "t_melt_k": 884.0,
        "k_th_a": 0.36, "k_th_c": 1.10, "rho_cp": 1.35e6,
        "ea_cryst_ev": 1.85, "k0_jmak_s1": 2.5e18, "n_avrami": 2.2,
        "quench_crit_k_per_ns": 0.5,
        "non_volatile": True,
    },
    "Sb2S3": {  # Delaney Adv. Funct. Mater. 2020
        "n_a_1550": 2.710, "k_a_1550": 1.0e-5,
        "n_c_1550": 3.310, "k_c_1550": 1.0e-4,
        "n_a_midir": 2.650, "k_a_midir": 1.0e-5,
        "n_c_midir": 3.200, "k_c_midir": 5.0e-5,
        "t_cryst_k": 543.0, "t_melt_k": 823.0,
        "k_th_a": 0.30, "k_th_c": 1.05, "rho_cp": 1.45e6,
        "ea_cryst_ev": 2.00, "k0_jmak_s1": 4.0e18, "n_avrami": 2.3,
        "quench_crit_k_per_ns": 0.6,
        "non_volatile": True,
    },
    "GSST": {  # Zhang Nat. Commun. 2019 (Ge2Sb2Se4Te1); CISM Lab NIR & Mid-IR
        "n_a_1550": 3.326, "k_a_1550": 1.8e-4,
        "n_c_1550": 5.083, "k_c_1550": 0.350,
        "n_a_midir": 3.200, "k_a_midir": 1.0e-4,
        "n_c_midir": 4.650, "k_c_midir": 0.055,
        "t_cryst_k": 553.0, "t_melt_k": 873.0,
        "k_th_a": 0.20, "k_th_c": 0.48, "rho_cp": 1.32e6,
        "ea_cryst_ev": 2.10, "k0_jmak_s1": 6.0e18, "n_avrami": 2.4,
        "quench_crit_k_per_ns": 0.25,
        "non_volatile": True,
    },
    "GST": {  # Rios Nat. Photonics 2015 / 2019 (Ge2Sb2Te5)
        "n_a_1550": 4.300, "k_a_1550": 0.120,
        "n_c_1550": 7.100, "k_c_1550": 1.050,
        "n_a_midir": 3.950, "k_a_midir": 0.020,
        "n_c_midir": 6.100, "k_c_midir": 0.320,
        "t_cryst_k": 433.0, "t_melt_k": 889.0,
        "k_th_a": 0.19, "k_th_c": 0.57, "rho_cp": 1.25e6,
        "ea_cryst_ev": 1.95, "k0_jmak_s1": 1.5e20, "n_avrami": 2.5,
        "quench_crit_k_per_ns": 1.5,
        "non_volatile": True,
    },
    "VO2": {  # Briggs Opt. Express 2010 (volatile Mott IMT at 68 C)
        "n_a_1550": 2.950, "k_a_1550": 0.320,
        "n_c_1550": 1.950, "k_c_1550": 2.600,
        "n_a_midir": 2.850, "k_a_midir": 0.120,
        "n_c_midir": 2.100, "k_c_midir": 4.200,
        "t_cryst_k": 341.0, "t_melt_k": 1800.0,
        "k_th_a": 3.50, "k_th_c": 6.00, "rho_cp": 3.00e6,
        "ea_cryst_ev": 0.40, "k0_jmak_s1": 1.0e12, "n_avrami": 1.5,
        "quench_crit_k_per_ns": 0.0,
        "non_volatile": False,
    },
}


def pcm_effective_index_lorentz_lorenz(
    pcm_material: str,
    crystallinity_x: float,
    wavelength_nm: float = 1550.0,
) -> tuple[float, float]:
    """Compute complex refractive index `(n, k)` at crystallization fraction `X in [0, 1]` via Lorentz-Lorenz EMT."""
    key = pcm_material if pcm_material in PCM_MATERIAL_CATALOG else "Sb2Se3"
    pr = PCM_MATERIAL_CATALOG[key]
    x = float(np.clip(crystallinity_x, 0.0, 1.0))
    # Smooth blend between 1550 nm and Mid-IR (>= 3000 nm) optical constants
    w_mid = float(np.clip((wavelength_nm - 1800.0) / 1400.0, 0.0, 1.0))
    na = (1.0 - w_mid) * pr["n_a_1550"] + w_mid * pr["n_a_midir"]
    ka = (1.0 - w_mid) * pr["k_a_1550"] + w_mid * pr["k_a_midir"]
    nc = (1.0 - w_mid) * pr["n_c_1550"] + w_mid * pr["n_c_midir"]
    kc = (1.0 - w_mid) * pr["k_c_1550"] + w_mid * pr["k_c_midir"]

    eps_a = complex(na, ka) ** 2
    eps_c = complex(nc, kc) ** 2
    ll = (1.0 - x) * (eps_a - 1.0) / (eps_a + 2.0) + x * (eps_c - 1.0) / (eps_c + 2.0)
    eps_eff = (1.0 + 2.0 * ll) / (1.0 - ll)
    n_eff_c = np.sqrt(eps_eff)
    return float(np.real(n_eff_c)), abs(float(np.imag(n_eff_c)))


# ==============================================================================
# 2. Multi-Level PCM-Clad Microring Optical Switch (2D Conformal Vector FEM)
# ==============================================================================

@dataclass
class PCMMRRConfig:
    """Configuration for a Non-Volatile Phase-Change Material (PCM) Integrated MRR Switch."""
    pcm_material: str = "Sb2Se3"            # "Sb2Se3" | "Sb2S3" | "GSST" | "GST" | "VO2"
    core_material: str = "Si"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.22
    slab_height_um: float = 0.0
    ring_width_um: float = 0.48
    ring_radius_um: float = 20.0
    pcm_thickness_nm: float = 30.0          # PCM patch thickness [nm]
    pcm_width_um: float = 0.48              # PCM patch width on waveguide [um]
    pcm_length_um: float = 8.0              # PCM active arc length along ring [um]
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    bare_q_int: float = 40000.0
    q_ext: float = 20000.0
    num_levels: int = 5                     # Multi-bit non-volatile crystallization states (0..1)
    core_mesh_res_um: float = 0.038
    export_gds_path: str | None = None


def simulate_pcm_mrr_switch(cfg: PCMMRRConfig) -> dict[str, Any]:
    """Solve multi-level PCM-on-MRR complex eigenmodes and non-volatile optical switching spectra."""
    t0 = time.perf_counter()
    wl0_nm = float(cfg.wavelength_nm)
    wl0_m = wl0_nm * 1e-9
    k0 = 2.0 * math.pi / wl0_m
    wl_scale = max(wl0_nm / 1550.0, 1.0)
    core_fn = get_material_fn(cfg.core_material)
    clad_fn = get_material_fn(cfg.clad_material)
    top_fn = get_material_fn(cfg.top_clad_material)
    pol_short = "TE" if cfg.polarization.upper().startswith("TE") else "TM"

    # Bare ring mode & group index
    spec_bare = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=cfg.slab_height_um,
        core_material_fn=core_fn, clad_material_fn=clad_fn, top_clad_material_fn=top_fn,
        domain_width_um=max(4.0 * wl_scale, cfg.ring_width_um + 2.5 * wl_scale),
        h_box_um=1.5 * wl_scale, h_top_clad_um=1.5 * wl_scale,
    )
    m_bare = select_target_polarization_mode(
        solve_modes_fem(
            spec_bare, wl0_m, num_modes=4, core_mesh_res_um=cfg.core_mesh_res_um * wl_scale,
            clad_mesh_res_um=0.22 * wl_scale, export_grid=False, mesh_backend="bsd_conformal",
        ),
        polarization=pol_short, fundamental_only=True,
    )
    neff_bare = float(np.real(m_bare.n_eff))
    # Fast 2-point group index
    m_bp = select_target_polarization_mode(
        solve_modes_fem(
            spec_bare, wl0_m * 1.004, num_modes=3, n_guess=neff_bare,
            core_mesh_res_um=cfg.core_mesh_res_um * wl_scale, clad_mesh_res_um=0.22 * wl_scale,
            export_grid=False, mesh_backend="bsd_conformal",
        ),
        polarization=pol_short, fundamental_only=True,
    )
    ng_bare = neff_bare - wl0_m * (float(np.real(m_bp.n_eff)) - neff_bare) / (wl0_m * 0.004)
    l_ring_m = 2.0 * math.pi * cfg.ring_radius_um * 1e-6
    l_pcm_m = min(cfg.pcm_length_um * 1e-6, 0.9 * l_ring_m)

    # Solve exact 2D Conformal Vector FEM with conformal PCM cap across X in [0, 1]
    xs = np.linspace(0.0, 1.0, max(2, int(cfg.num_levels)))
    states: list[dict[str, float]] = []
    neff_a_ref = None
    for x_val in xs:
        n_p, k_p = pcm_effective_index_lorentz_lorenz(cfg.pcm_material, float(x_val), wl0_nm)
        def _pcm_fn(_w: float, _t: float = T_REF_K, _np=n_p, _kp=k_p) -> MaterialIndexTensor:
            return MaterialIndexTensor(nx=_np, ny=_np, nz=_np, kx=_kp, ky=_kp, kz=_kp)

        spec_pcm = WaveguideCrossSectionSpec(
            w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=cfg.slab_height_um,
            core_material_fn=core_fn, clad_material_fn=clad_fn, top_clad_material_fn=top_fn,
            cap_h_um=cfg.pcm_thickness_nm * 1e-3,
            cap_w_um=min(cfg.pcm_width_um, cfg.ring_width_um),
            cap_material_fn=_pcm_fn,
            domain_width_um=max(4.0 * wl_scale, cfg.ring_width_um + 2.5 * wl_scale),
            h_box_um=1.5 * wl_scale, h_top_clad_um=1.5 * wl_scale,
        )
        m_list = solve_modes_fem(
            spec_pcm, wl0_m, num_modes=5, n_guess=neff_bare + 0.15,
            core_mesh_res_um=cfg.core_mesh_res_um * wl_scale,
            clad_mesh_res_um=0.22 * wl_scale, export_grid=False, mesh_backend="bsd_conformal",
        )
        m_st = select_target_polarization_mode(m_list, polarization=pol_short, fundamental_only=True)
        nr = float(np.real(m_st.n_eff))
        ki = abs(float(np.imag(m_st.n_eff)))
        if neff_a_ref is None:
            neff_a_ref = nr
        dn_vs_a = nr - neff_a_ref
        alpha_db_um = 2.0 * k0 * ki * (10.0 / math.log(10.0)) * 1e-6
        il_cell_db = alpha_db_um * cfg.pcm_length_um
        dphi_rad = k0 * l_pcm_m * dn_vs_a
        # Non-volatile resonance wavelength shift Delta_lambda = (lambda / ng) * (L_pcm / L_ring) * Delta_neff
        dwl_nm = (wl0_nm / max(ng_bare, 1.0)) * (l_pcm_m / l_ring_m) * dn_vs_a

        # Loaded MRR thru-port transmission at amorphous resonance wavelength lambda_0
        om0 = 2.0 * math.pi * C0_M_PER_S / wl0_m
        k_int_bare = om0 / max(cfg.bare_q_int, 100.0)
        # Round-trip PCM absorption rate: kappa_pcm = (v_g / L_ring) * (2 * k0 * ki * L_pcm)
        vg = C0_M_PER_S / max(ng_bare, 1.0)
        k_pcm = (vg / l_ring_m) * (2.0 * k0 * ki * l_pcm_m)
        k_ext = om0 / max(cfg.q_ext, 100.0)
        k_tot = k_int_bare + k_pcm + k_ext
        d_om_shift = -(om0 / wl0_nm) * dwl_nm
        # All-pass transmission at probe wavelength lambda_0 (where amorphous state is on resonance!)
        t_thru_at_wl0 = float(abs(1.0 - k_ext / (0.5 * k_tot - 1j * d_om_shift)) ** 2)

        states.append({
            "crystallinity_X": round(float(x_val), 3),
            "neff_real": round(nr, 5),
            "keff_imag": round(ki, 6),
            "delta_neff_vs_a": round(dn_vs_a, 5),
            "alpha_pcm_dB_per_um": round(alpha_db_um, 4),
            "cell_IL_dB": round(il_cell_db, 3),
            "phase_shift_pi": round(dphi_rad / math.pi, 4),
            "resonance_shift_nm": round(dwl_nm, 4),
            "thru_T_at_lambda0": round(t_thru_at_wl0, 4),
        })

    st_a, st_c = states[0], states[-1]
    dn_full = abs(st_c["neff_real"] - st_a["neff_real"])
    dk_full = max(abs(st_c["keff_imag"] - st_a["keff_imag"]), 1e-6)
    l_pi_um = (wl0_nm * 1e-3) / max(2.0 * dn_full, 1e-5)
    il_pi_db = st_c["alpha_pcm_dB_per_um"] * l_pi_um
    t_vals = [s["thru_T_at_lambda0"] for s in states]
    er_switch_db = 10.0 * math.log10(max(max(t_vals), 1e-5) / max(min(t_vals), 1e-5))

    # Full MRR spectra for amorphous (X=0) vs crystalline (X=1)
    span_nm = max(2.5, abs(st_c["resonance_shift_nm"]) * 2.8)
    dwl_grid = np.linspace(-0.4 * span_nm, 0.6 * span_nm, 241)
    om0 = 2.0 * math.pi * C0_M_PER_S / wl0_m
    vg = C0_M_PER_S / max(ng_bare, 1.0)
    k_int_bare = om0 / max(cfg.bare_q_int, 100.0)
    k_ext = om0 / max(cfg.q_ext, 100.0)
    specs_out = {}
    for label, st in (("amorphous", st_a), ("crystalline", st_c)):
        k_p = (vg / l_ring_m) * (2.0 * k0 * st["keff_imag"] * l_pcm_m)
        k_t = k_int_bare + k_p + k_ext
        dom = -(om0 / wl0_nm) * (dwl_grid - st["resonance_shift_nm"])
        t_s = np.abs(1.0 - k_ext / (0.5 * k_t - 1j * dom)) ** 2
        specs_out[label] = np.round(10.0 * np.log10(np.maximum(t_s, 1e-6)), 3).tolist()

    # Optional Cleanroom GDSII Export
    gds_written: str | None = None
    if cfg.export_gds_path:
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell(f"PCM_{cfg.pcm_material.upper()}_MRR")
        r = cfg.ring_radius_um
        w = cfg.ring_width_um
        cell.add(gdstk.ellipse((0.0, 0.0), r + 0.5 * w, inner_radius=r - 0.5 * w, tolerance=0.002, layer=MICRORING_LAYERS["core"][0]))
        # Bus waveguide
        y_bus = -(r + 0.5 * w + 0.20 + 0.5 * w)
        cell.add(gdstk.rectangle((-r - 15.0, y_bus - 0.5 * w), (r + 15.0, y_bus + 0.5 * w), layer=MICRORING_LAYERS["core"][0]))
        # PCM arc patch centered at top of ring (theta = pi/2) on layer 25/0 + TiN heater on layer 10/0
        dtheta = min(cfg.pcm_length_um / max(r, 1.0), math.pi)
        th0, th1 = 0.5 * math.pi - 0.5 * dtheta, 0.5 * math.pi + 0.5 * dtheta
        cell.add(gdstk.ellipse((0.0, 0.0), r + 0.5 * cfg.pcm_width_um, inner_radius=r - 0.5 * cfg.pcm_width_um,
                               initial_angle=th0, final_angle=th1, tolerance=0.002, layer=25, datatype=0))
        cell.add(gdstk.ellipse((0.0, 0.0), r + 0.9 * cfg.pcm_width_um, inner_radius=r - 0.9 * cfg.pcm_width_um,
                               initial_angle=th0 - 0.05, final_angle=th1 + 0.05, tolerance=0.002, layer=MICRORING_LAYERS["heater"][0]))
        out_p = Path(cfg.export_gds_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lib.write_gds(str(out_p))
        write_klayout_layer_props(out_p.with_suffix(".lyp"))
        gds_written = str(out_p)

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "pcm_material": cfg.pcm_material,
            "neff_amorphous": st_a["neff_real"],
            "neff_crystalline": st_c["neff_real"],
            "delta_neff_a_to_c": round(st_c["neff_real"] - st_a["neff_real"], 5),
            "alpha_amorphous_dB_per_um": st_a["alpha_pcm_dB_per_um"],
            "alpha_crystalline_dB_per_um": st_c["alpha_pcm_dB_per_um"],
            "modal_FOM_dn_over_dk": round(dn_full / dk_full, 2),
            "L_pi_um": round(l_pi_um, 2),
            "IL_at_pi_dB": round(il_pi_db, 3),
            "resonance_shift_a_to_c_nm": st_c["resonance_shift_nm"],
            "MRR_switching_ER_dB": round(er_switch_db, 2),
            "num_nonvolatile_levels": len(states),
            "gds_path": gds_written,
        },
        "states": states,
        "spectrum": {
            "detuning_nm": np.round(dwl_grid, 4).tolist(),
            "amorphous_T_dB": specs_out["amorphous"],
            "crystalline_T_dB": specs_out["crystalline"],
        },
    }


# ==============================================================================
# 3. 2D Crank-Nicolson Transient Electro-Thermal SET / RESET Pulse FEM
# ==============================================================================

@dataclass
class PCMThermalPulseConfig:
    """Configuration for 2D Transient Thermal FEM of PCM Microheater `RESET` & `SET` Pulses."""
    pcm_material: str = "Sb2Se3"
    core_material: str = "Si"
    film_height_um: float = 0.22
    ring_width_um: float = 0.50
    pcm_thickness_nm: float = 30.0
    pcm_length_um: float = 8.0
    heater_type: str = "doped_si_pin"       # "doped_si_pin" | "top_tin"
    box_thickness_um: float = 2.0
    oxide_spacer_um: float = 0.15           # Thin Al2O3/SiO2 spacer between heater & PCM [um]
    reset_power_mw: float = 48.0            # High-power short RESET pulse power [mW]
    reset_pulse_ns: float = 400.0           # RESET pulse duration [ns]
    set_power_mw: float = 16.5              # Moderate-power SET pulse power [mW] (T_c < T < T_m)
    set_pulse_ns: float = 5000.0            # SET crystallization pulse duration [ns]


def simulate_pcm_thermal_pulse(cfg: PCMThermalPulseConfig) -> dict[str, Any]:
    """Solve 2D Crank-Nicolson transient thermal FEM for PCM `RESET` melt-quench & `SET` JMAK kinetics."""
    t0 = time.perf_counter()
    key = cfg.pcm_material if cfg.pcm_material in PCM_MATERIAL_CATALOG else "Sb2Se3"
    pr = PCM_MATERIAL_CATALOG[key]

    w_c = float(cfg.ring_width_um)
    h_c = float(cfg.film_height_um)
    t_pcm_um = float(cfg.pcm_thickness_nm) * 1e-3
    h_box = float(cfg.box_thickness_um)
    h_sp = float(cfg.oxide_spacer_um)
    h_htr = 0.08
    w_htr = w_c + 0.6

    # Build snappy conformal rectilinear 2D thermal FEM mesh [um -> m]
    x_um = _graded_1d_axis(
        [-8.0, -3.5, -0.5 * w_htr, -0.5 * w_c, 0.0, 0.5 * w_c, 0.5 * w_htr, 3.5, 8.0],
        [0.55, 0.25, 0.10, 0.07, 0.07, 0.10, 0.25, 0.55],
    )
    y_pts = [-h_box - 3.0, -h_box, 0.0, h_c, h_c + t_pcm_um]
    y_steps = [0.60, 0.22, 0.06, max(0.015, 0.5 * t_pcm_um)]
    if cfg.heater_type == "top_tin":
        y_pts += [h_c + t_pcm_um + h_sp, h_c + t_pcm_um + h_sp + h_htr, h_c + 2.5]
        y_steps += [max(0.04, 0.4 * h_sp), 0.04, 0.35]
    else:
        y_pts += [h_c + t_pcm_um + 0.20, h_c + 2.5]
        y_steps += [0.08, 0.35]
    y_um = _graded_1d_axis(y_pts, y_steps)

    mesh, cx_um, cy_um = _rectilinear_mesh_tri(x_um, y_um)
    basis = Basis(mesh, ElementTriP1())
    p0 = basis.with_element(ElementTriP0())

    k_th = np.full(mesh.t.shape[1], THERMAL_CONDUCTIVITY_W_MK.get("SiO2", 1.38), dtype=float)
    rho_cp = np.full(mesh.t.shape[1], VOLUMETRIC_HEAT_CAPACITY_J_M3K.get("SiO2", 1.63e6), dtype=float)

    # Bottom Si substrate (y < -h_box)
    m_sub = cy_um < -h_box
    k_th[m_sub] = THERMAL_CONDUCTIVITY_W_MK.get("Si", 148.0)
    rho_cp[m_sub] = VOLUMETRIC_HEAT_CAPACITY_J_M3K.get("Si", 1.64e6)

    # Waveguide core (0 <= y <= h_c, |x| <= w_c/2)
    m_core = (cy_um >= 0.0) & (cy_um <= h_c) & (np.abs(cx_um) <= 0.5 * w_c)
    k_th[m_core] = THERMAL_CONDUCTIVITY_W_MK.get(cfg.core_material, 148.0 if cfg.core_material == "Si" else 20.0)
    rho_cp[m_core] = VOLUMETRIC_HEAT_CAPACITY_J_M3K.get(cfg.core_material, 1.64e6)

    # PCM cell (h_c <= y <= h_c + t_pcm_um, |x| <= w_c/2)
    m_pcm = (cy_um >= h_c) & (cy_um <= h_c + t_pcm_um) & (np.abs(cx_um) <= 0.5 * w_c)
    k_th[m_pcm] = 0.5 * (pr["k_th_a"] + pr["k_th_c"])
    rho_cp[m_pcm] = pr["rho_cp"]

    # Heater element mask (unit linear power 1 W/m deposited uniformly across heater area)
    if cfg.heater_type == "top_tin":
        m_htr = (cy_um >= h_c + t_pcm_um + h_sp) & (cy_um <= h_c + t_pcm_um + h_sp + h_htr) & (np.abs(cx_um) <= 0.5 * w_htr)
        k_th[m_htr] = 28.0
        rho_cp[m_htr] = 3.1e6
    else:
        # In-rib doped Si microheater: heat generated directly in Si core & adjacent doped flanks
        m_htr = (cy_um >= 0.0) & (cy_um <= h_c) & (np.abs(cx_um) <= 0.5 * w_htr)
        k_th[m_htr] = 65.0  # Doped Si phonon-impurity reduced thermal conductivity
        rho_cp[m_htr] = 1.64e6

    # Element areas & unit volumetric heat source q_vol [W/m^3 per 1 W/m linear power]
    p_x, p_y = mesh.p[0, mesh.t], mesh.p[1, mesh.t]
    tri_areas = 0.5 * np.abs((p_x[1] - p_x[0]) * (p_y[2] - p_y[0]) - (p_x[2] - p_x[0]) * (p_y[1] - p_y[0]))
    a_htr = float(np.sum(tri_areas[m_htr]))
    q_unit = np.where(m_htr, 1.0 / max(a_htr, 1e-18), 0.0)

    k_interp = p0.interpolate(k_th)
    c_interp = p0.interpolate(rho_cp)
    q_interp = p0.interpolate(q_unit)

    @BilinearForm
    def a_k(u, v, w):
        return w["k"] * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

    @BilinearForm
    def a_m(u, v, w):
        return w["c"] * u * v

    @LinearForm
    def l_q(v, w):
        return w["q"] * v

    K_mat = a_k.assemble(basis, k=k_interp)
    M_mat = a_m.assemble(basis, c=c_interp)
    b_unit = l_q.assemble(basis, q=q_interp)

    # Dirichlet T = T_REF_K at bottom substrate boundary (y == y_min)
    d_dofs = basis.get_dofs(lambda x: x[1] <= float(y_um[0]) * 1e-6 + 1e-9).all()
    K_c, b_c, _, free_i = condense(K_mat, b_unit, D=d_dofs)
    M_c = M_mat[free_i][:, free_i].tocsc()
    K_c = K_c.tocsc()

    # Nodes inside PCM region for exact average & peak PCM temperature tracking
    pcm_node_mask = (mesh.p[1] * 1e6 >= h_c - 1e-4) & (mesh.p[1] * 1e6 <= h_c + t_pcm_um + 1e-4) & (np.abs(mesh.p[0] * 1e6) <= 0.5 * w_c + 1e-4)
    pcm_nodes = np.flatnonzero(pcm_node_mask)
    # Steady-state thermal resistance R_th_PCM [K / (W/m)] and [K / mW]
    dT_ss_full = np.zeros(mesh.p.shape[1], dtype=float)
    dT_ss_full[free_i] = splu(K_c).solve(b_c)
    r_th_pcm_k_per_mw = float(np.mean(dT_ss_full[pcm_nodes])) * (1e-3 / max(cfg.pcm_length_um * 1e-6, 1e-7))

    def _run_cn_pulse(power_mw: float, t_on_ns: float, t_cool_ns: float, n_on: int = 45, n_off: int = 65):
        p_lin_w_per_m = (power_mw * 1e-3) / max(cfg.pcm_length_um * 1e-6, 1e-7)
        b_on = b_c * p_lin_w_per_m
        dt_on = (t_on_ns * 1e-9) / n_on
        dt_off = (t_cool_ns * 1e-9) / n_off

        lu_on = splu((M_c + 0.5 * dt_on * K_c).tocsc())
        rhs_on_mat = (M_c - 0.5 * dt_on * K_c).tocsc()
        lu_off = splu((M_c + 0.5 * dt_off * K_c).tocsc())
        rhs_off_mat = (M_c - 0.5 * dt_off * K_c).tocsc()

        u = np.zeros(len(free_i), dtype=float)
        ts_ns = [0.0]
        t_pcm_k = [T_REF_K]
        full_u = np.zeros(mesh.p.shape[1], dtype=float)

        for step in range(n_on):
            u = lu_on.solve(rhs_on_mat.dot(u) + dt_on * b_on)
            full_u[free_i] = u
            ts_ns.append((step + 1) * dt_on * 1e9)
            t_pcm_k.append(T_REF_K + float(np.mean(full_u[pcm_nodes])))

        for step in range(n_off):
            u = lu_off.solve(rhs_off_mat.dot(u))
            full_u[free_i] = u
            ts_ns.append(t_on_ns + (step + 1) * dt_off * 1e9)
            t_pcm_k.append(T_REF_K + float(np.mean(full_u[pcm_nodes])))

        return np.array(ts_ns, dtype=float), np.array(t_pcm_k, dtype=float)

    # 1. RESET pulse (melt-quench amorphization)
    t_rst_ns, T_rst_k = _run_cn_pulse(cfg.reset_power_mw, cfg.reset_pulse_ns, max(1200.0, 3.0 * cfg.reset_pulse_ns))
    t_peak_rst = float(np.max(T_rst_k))
    i_pk = int(np.argmax(T_rst_k))
    # Quench rate |dT/dt| [K/ns] right after pulse turn-off across cooling window
    dT_dt = np.gradient(T_rst_k, t_rst_ns)
    cool_mask = (np.arange(len(T_rst_k)) > i_pk) & (T_rst_k >= pr["t_cryst_k"]) & (T_rst_k <= pr["t_melt_k"] + 50.0)
    quench_rate_k_ns = float(np.max(-dT_dt[cool_mask])) if np.any(cool_mask) else float(max(-np.min(dT_dt[i_pk:]), 0.0))
    reset_success = bool((t_peak_rst >= pr["t_melt_k"]) and (quench_rate_k_ns >= pr["quench_crit_k_per_ns"]))
    e_reset_nj = cfg.reset_power_mw * 1e-3 * cfg.reset_pulse_ns  # mW * ns = pJ -> *1e-3 = nJ

    # 2. SET pulse (JMAK Arrhenius crystallization kinetics)
    t_set_ns, T_set_k = _run_cn_pulse(cfg.set_power_mw, cfg.set_pulse_ns, max(2000.0, 0.6 * cfg.set_pulse_ns))
    t_peak_set = float(np.max(T_set_k))
    # Integrate JMAK crystallization rate k(T) = k0 * exp(-Ea/(kB*T)) while T_c - 60 < T < T_m
    dt_set_s = np.diff(t_set_ns) * 1e-9
    T_mid = 0.5 * (T_set_k[:-1] + T_set_k[1:])
    active_c = (T_mid >= pr["t_cryst_k"] - 60.0) & (T_mid < pr["t_melt_k"])
    kt_sum = float(np.sum(
        np.where(active_c, pr["k0_jmak_s1"] * np.exp(-pr["ea_cryst_ev"] / (KB_EV_PER_K * np.maximum(T_mid, 300.0))) * dt_set_s, 0.0)
    ))
    x_set_final = float(1.0 - math.exp(-min(kt_sum ** pr["n_avrami"], 50.0)))
    set_success = bool((t_peak_set >= pr["t_cryst_k"]) and (t_peak_set < pr["t_melt_k"]) and (x_set_final >= 0.50))
    e_set_nj = cfg.set_power_mw * 1e-3 * cfg.set_pulse_ns

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "pcm_material": key,
            "heater_type": cfg.heater_type,
            "R_th_PCM_K_per_mW": round(r_th_pcm_k_per_mw, 3),
            "T_cryst_K": pr["t_cryst_k"],
            "T_melt_K": pr["t_melt_k"],
            "RESET_T_peak_K": round(t_peak_rst, 1),
            "RESET_quench_rate_K_per_ns": round(quench_rate_k_ns, 3),
            "RESET_energy_nJ": round(e_reset_nj, 3),
            "RESET_amorphization_pass": reset_success,
            "SET_T_peak_K": round(t_peak_set, 1),
            "SET_crystallized_fraction": round(x_set_final, 4),
            "SET_energy_nJ": round(e_set_nj, 3),
            "SET_crystallization_pass": set_success,
        },
        "reset_waveform": {
            "time_ns": np.round(t_rst_ns, 2).tolist(),
            "T_pcm_K": np.round(T_rst_k, 2).tolist(),
        },
        "set_waveform": {
            "time_ns": np.round(t_set_ns, 2).tolist(),
            "T_pcm_K": np.round(T_set_k, 2).tolist(),
        },
    }
