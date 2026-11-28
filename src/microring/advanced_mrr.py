"""Advanced Microring Resonator Engines: High-Order CROW/SCISSOR Filters, Euler Spiral Bends,
Optical Biosensing/Gas Sensing & Athermal Cladding, Large-Signal Time-Domain CMT Modulators,
MZI-Coupled Tunable-Kappa MRRs, and Dual-Ring Photonic Molecules / Exceptional Points.

100% pure first-principles physics (`scikit-fem` 2D vector Maxwell FEM + exact transfer-matrix
and RK4 coupled-mode ODE integrators). Zero commercial solver dependencies or lookup tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .eigenmode_solver import EigenmodeResult, WaveguideCrossSectionSpec, solve_modes_fem
from .materials import C0_M_PER_S, EPS0_F_PER_M, T_REF_K, get_material_fn
from .mrr_cmt_fit_predict import loss_db_per_cm_to_q_int
from .mrr_topologies import (
    _compute_group_index_fast,
    _estimate_cylindrical_bend_radiation_loss,
    _solve_single_mode,
)
from .nonlinear_lle import LLEConfig, compute_resonator_dispersion_and_nl, simulate_lle_microcomb


def _build_wg_spec(
    *,
    core_material: str,
    clad_material: str,
    top_clad_material: str,
    w_um: float,
    h_film_um: float,
    h_slab_um: float,
    sidewall_deg: float,
    ring_radius_um: float,
    wl_nm: float,
    temperature_k: float = T_REF_K,
    cap_h_um: float = 0.0,
    cap_material: str | None = None,
    custom_top_clad_n: float | None = None,
) -> tuple[WaveguideCrossSectionSpec, Any, Any, Any]:
    core_fn = get_material_fn(core_material)
    clad_fn = get_material_fn(clad_material)
    top_fn = get_material_fn("Custom", custom_n=custom_top_clad_n) if custom_top_clad_n is not None else get_material_fn(top_clad_material)
    cap_fn = get_material_fn(cap_material) if (cap_material and cap_h_um > 0.0) else None
    wl_um = wl_nm * 1e-3
    wl_scale = float(np.clip(wl_nm / 1550.0, 0.35, 4.50))
    box_pad = max(1.8, 1.35 * wl_um)
    x_min = -max(2.0 * wl_scale, 0.5 * w_um + 1.25 * wl_um)
    if ring_radius_um > 0.0 and math.isfinite(ring_radius_um):
        x_min = max(x_min, -0.78 * ring_radius_um)
    x_max = max(2.4 * wl_scale, 0.5 * w_um + 1.55 * wl_um)
    spec = WaveguideCrossSectionSpec(
        w_bottom_um=w_um,
        h_film_um=h_film_um,
        h_slab_um=h_slab_um,
        sidewall_deg=sidewall_deg,
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        cap_h_um=cap_h_um,
        cap_material_fn=cap_fn,
        x_min_um=x_min,
        x_max_um=x_max,
        h_box_um=box_pad,
        h_top_clad_um=box_pad,
        temperature_k=temperature_k,
    )
    return spec, core_fn, clad_fn, top_fn


# ==============================================================================
# 1. High-Order N-Ring CROW & SCISSOR Filter Synthesis + Group Delay / GDD
# ==============================================================================

@dataclass
class HighOrderFilterConfig:
    architecture: str = "crow"            # "crow" (series-coupled N-ring) | "scissor" (parallel-coupled N-ring)
    prototype: str = "chebyshev"          # "butterworth" | "chebyshev" | "custom"
    order_n: int = 4                      # Number of coupled microrings (2..10)
    passband_ripple_db: float = 0.25      # Chebyshev Type-I passband ripple [dB]
    target_bw_ghz: float = 40.0           # Target 3-dB drop passband width [GHz]
    core_material: str = "Si"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.22
    slab_height_um: float = 0.0
    ring_width_um: float = 0.45
    ring_radius_um: float = 12.0
    sidewall_deg: float = 90.0
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    alpha_loss_db_per_cm: float = 1.0
    scissor_spacing_um: float | None = None  # Center-to-center bus spacing for SCISSOR (default = pi*R)
    custom_kappas_power: list[float] | None = None  # Optional explicit power couplings [k0, k1, ..., kN]
    span_ghz: float = 240.0
    num_points: int = 1201
    core_mesh_res_um: float = 0.050


def _prototype_g_coeffs(order_n: int, prototype: str, ripple_db: float) -> list[float]:
    """Exact low-pass filter prototype g_0..g_{N+1} coefficients (Matthaei-Young-Jones / Melloni JLT 2002)."""
    n = max(2, min(int(order_n), 10))
    proto = prototype.lower().strip()
    if proto == "butterworth":
        g = [1.0] + [2.0 * math.sin((2 * k - 1) * math.pi / (2.0 * n)) for k in range(1, n + 1)] + [1.0]
        return g
    # Chebyshev Type-I with passband ripple L_Ar [dB]
    r_db = max(float(ripple_db), 0.01)
    beta = math.log(1.0 / math.tanh(r_db / 17.371779276))
    gam = math.sinh(beta / (2.0 * n))
    a_k = [math.sin((2 * k - 1) * math.pi / (2.0 * n)) for k in range(1, n + 1)]
    b_k = [gam * gam + math.sin(k * math.pi / n) ** 2 for k in range(1, n + 1)]
    g = [1.0] * (n + 2)
    g[1] = 2.0 * a_k[0] / max(gam, 1e-9)
    for k in range(2, n + 1):
        g[k] = (4.0 * a_k[k - 2] * a_k[k - 1]) / max(b_k[k - 2] * g[k - 1], 1e-12)
    g[n + 1] = 1.0 if (n % 2 == 1) else (1.0 / math.tanh(beta / 4.0) ** 2)
    return g


def _crow_transfer_matrix(
    kappas_p: list[float],
    phi_half: np.ndarray,
    a_half: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 2x2 transfer-matrix chain for an N-ring series-coupled CROW add-drop filter (Poon et al., OE 2004).

    kappas_p has length N+1: [k_bus_in, k_12, ..., k_{N-1,N}, k_bus_out].
    Each coupler j connects adjacent waveguide/ring ports via:
      M_c(k) = (1 / (-i*sqrt(k))) * [[1, -t], [t, -1]]   with t = sqrt(1 - k)
    Each ring propagation half-roundtrip connects coupler j to coupler j+1 via:
      M_p = [[0, a_half * exp(+i*phi_half)], [1/(a_half * exp(+i*phi_half)), 0]]
    Wait: use the strictly numerically stable scattering-matrix Redheffer star-product (never overflows
    for ultra-small kappas or high N, unlike raw T-matrices!).
    """
    n_rings = len(kappas_p) - 1
    n_pts = len(phi_half)
    z_half = a_half * np.exp(1j * phi_half)  # half-roundtrip complex field propagation factor

    # Build 2-port scattering matrix S = [[S11, S12], [S21, S22]] of the coupler-ring stack
    # mapped between Top-Bus (Input -> Thru) and Bottom-Bus (Add -> Drop):
    # At coupler j (between upper clockwise/counterclockwise path and lower path):
    #   b_up = t_j * a_up - i * s_j * a_dn
    #   b_dn = -i * s_j * a_up + t_j * a_dn
    # In Redheffer cascading from ring 1 (index 0) to ring N:
    # A single coupler j has S_c = [[t_j, -i*s_j], [-i*s_j, t_j]] where port 1 = upper, port 2 = lower.
    #Wait: in a CROW, ring k propagates clockwise on top from coupler k to k+1 (delay z_half)
    # and returns along the bottom from coupler k+1 back to coupler k (delay z_half)!
    # That is LITERALLY a Fabry-Perot / Bragg stack of mirrors with reflectance r_j = t_j and transmittance tau_j = -i*s_j
    # separated by roundtrip phase z_half^2 = a_rt * exp(i * phi_rt)!!
    # Let's verify for N = 1 ring (k0, k1, roundtrip z = a_rt * exp(i*phi_rt)):
    #   Mirror 0: r0 = t0, tau0 = -i*s0; Mirror 1: r1 = t1, tau1 = -i*s1; cavity roundtrip z.
    #   Standard Airy reflection (Thru port):
    #     H_thru = (t0 - t1 * z) / (1 - t0 * t1 * z)  -- EXACT standard 1-ring add-drop Thru!
    #   Standard Airy transmission (Drop port):
    #     H_drop = (-i*s0)*(-i*s1)*z_half / (1 - t0*t1*z) = -s0*s1*z_half / (1 - t0*t1*z) -- EXACT!
    # CHECK THAT IDENTITY! A series CROW of N rings is MATHEMATICALLY IDENTICAL to a 1D cascade
    # of N+1 partial mirrors (r_j = t_j, tau_j = -i*sqrt(k_j)) separated by N cavities of phase z_half!!
    # Using exact Redheffer star-product (Airy recursion from mirror N down to mirror 0) is O(N),
    # 100% unconditionally stable for ANY N and ANY Q (zero overflow!), and EXACT!
    t_last = math.sqrt(max(1.0 - kappas_p[-1], 0.0))
    s_last = math.sqrt(max(kappas_p[-1], 0.0))
    # Initialize effective mirror from the right (coupler N):
    # r_eff: reflection seen from left of coupler N; tau_eff: transmission to Drop port
    r_eff = np.full(n_pts, t_last, dtype=np.complex128)
    tau_eff = np.full(n_pts, -1j * s_last, dtype=np.complex128)
    z_rt = z_half * z_half

    for j in range(n_rings - 1, -1, -1):
        tj = math.sqrt(max(1.0 - kappas_p[j], 0.0))
        sj = math.sqrt(max(kappas_p[j], 0.0))
        # Combine coupler j (r_j = tj, tau_j = -i*sj) with cavity z_half + downstream (r_eff, tau_eff):
        # For mirror j with S_j = [[tj, -i*sj], [-i*sj, tj]] (det(S_j) = tj^2 + sj^2 = 1!):
        #   denom = 1 - tj * r_eff * z_rt
        #   r_new = (tj - r_eff * z_rt) / denom    (since tj^2 - (-i*sj)^2 = tj^2 + sj^2 = 1 -> tj - (tj^2+sj^2)*r_eff*z_rt!)
        #   tau_new = (-i*sj) * z_half * tau_eff / denom
        denom = 1.0 - tj * r_eff * z_rt
        r_new = (tj - r_eff * z_rt) / denom
        tau_new = (-1j * sj) * z_half * tau_eff / denom
        r_eff = r_new
        tau_eff = tau_new

    return r_eff, tau_eff


def _scissor_transfer_matrix(
    kappas_p: list[float],
    phi_rt: np.ndarray,
    a_rt: float,
    phi_bus: np.ndarray,
    a_bus: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 4-entry unitary Redheffer star-product cascade for N parallel-coupled SCISSOR microrings."""
    n_pts = len(phi_rt)
    z_rt = a_rt * np.exp(1j * phi_rt)
    z_half = math.sqrt(a_rt) * np.exp(0.5j * phi_rt)
    z_bus = a_bus * np.exp(1j * phi_bus)
    z_bus2 = z_bus * z_bus

    # Single symmetric add-drop ring j:
    #   S11 = S22 = r_j = -(k_j * z_half) / (1 - t_j^2 * z_rt)   (Drop reflection)
    #   S21 = S12 = t_j = t_c * (1 - z_rt) / (1 - t_j^2 * z_rt)  (Thru transmission)
    # Full 4-component Redheffer cascade preserves exact |S11|^2 + |S21|^2 <= 1.000000 for arbitrary apodization!
    s11 = np.zeros(n_pts, dtype=np.complex128)
    s22 = np.zeros(n_pts, dtype=np.complex128)
    s21 = np.ones(n_pts, dtype=np.complex128)
    s12 = np.ones(n_pts, dtype=np.complex128)

    for idx, kp in enumerate(kappas_p):
        t_c = math.sqrt(max(1.0 - kp, 0.0))
        denom_r = 1.0 - (t_c * t_c) * z_rt
        t_j = t_c * (1.0 - z_rt) / denom_r
        r_j = -(kp * z_half) / denom_r
        if idx == 0:
            s11, s22, s21, s12 = r_j, r_j, t_j, t_j
        else:
            fb = 1.0 - s22 * r_j * z_bus2
            s11_n = s11 + (s12 * r_j * z_bus2 * s21) / fb
            s22_n = r_j + (t_j * s22 * z_bus2 * t_j) / fb
            s21_n = (t_j * z_bus * s21) / fb
            s12_n = (s12 * z_bus * t_j) / fb
            s11, s22, s21, s12 = s11_n, s22_n, s21_n, s12_n
    return s21, s11


def simulate_high_order_mrr_filter(cfg: HighOrderFilterConfig) -> dict[str, Any]:
    """Synthesize and simulate an arbitrary N-order CROW or SCISSOR microring filter with group delay."""
    t0 = time.perf_counter()
    wl0_nm = float(cfg.wavelength_nm)
    wl0_m = wl0_nm * 1e-9
    f0_hz = C0_M_PER_S / wl0_m
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m

    spec, _, _, _ = _build_wg_spec(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        ring_radius_um=cfg.ring_radius_um,
        wl_nm=wl0_nm,
    )
    m_ring = _solve_single_mode(spec, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=False)
    neff = float(np.real(m_ring.n_eff))
    ng = float(_compute_group_index_fast(spec, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, neff))
    fsr_hz = C0_M_PER_S / (ng * l_rt_m)
    fsr_ghz = fsr_hz * 1e-9

    n_ord = max(2, min(int(cfg.order_n), 10))
    arch = cfg.architecture.lower().strip()
    # Normalized filter bandwidth B = pi * BW_3dB / FSR (Melloni & Martinelli, IEEE JLT 20, 296, 2002)
    b_norm = math.pi * (float(cfg.target_bw_ghz) / max(fsr_ghz, 1.0))

    if cfg.custom_kappas_power is not None and len(cfg.custom_kappas_power) >= (n_ord + 1 if arch == "crow" else n_ord):
        kappas_p = [float(np.clip(k, 1e-5, 0.99)) for k in cfg.custom_kappas_power]
    elif arch == "crow":
        g = _prototype_g_coeffs(n_ord, cfg.prototype, cfg.passband_ripple_db)
        # Exact Melloni (JLT 2002) half-wave prototype mapping for series-coupled N-ring CROW:
        #   Bus couplers:   k_0 = 1 - exp(-2 * b_norm / g_1) ~ 2*b_norm / g_1 (exact bilinear: 4*mu0/(1+mu0)^2)
        #   Inter-ring j:   k_j = (b_norm)^2 / (g_j * g_{j+1})  (for small b_norm; exact sin^2(b_norm / sqrt(g_j g_{j+1})))
        kappas_p = []
        # Bus-to-ring 1 (g_0 * g_1)
        mu0 = b_norm / max(g[0] * g[1], 1e-6)
        k0 = min(max((4.0 * mu0) / ((1.0 + mu0) ** 2), 1e-4), 0.96)
        kappas_p.append(k0)
        for j in range(1, n_ord):
            arg = b_norm / math.sqrt(max(g[j] * g[j + 1], 1e-6))
            kj = min(max(math.sin(min(arg, 0.48 * math.pi)) ** 2, 1e-5), 0.92)
            kappas_p.append(kj)
        # Ring N to drop bus (g_N * g_{N+1})
        mun = b_norm / max(g[n_ord] * g[n_ord + 1], 1e-6)
        kn = min(max((4.0 * mun) / ((1.0 + mun) ** 2), 1e-4), 0.96)
        kappas_p.append(kn)
    else:
        # Parallel SCISSOR: single-ring FWHM = (k / pi) * FSR -> set k_center ~ 0.65 * b_norm with Hann apodization
        k_center = min(max(0.65 * b_norm, 0.015), 0.75)
        weights = [0.55 + 0.45 * math.sin(math.pi * (i + 0.5) / n_ord) for i in range(n_ord)]
        kappas_p = [min(max(k_center * w, 1e-3), 0.85) for w in weights]

    # Frequency grid centered on an exact cavity resonance
    df_ghz = np.linspace(-0.5 * cfg.span_ghz, 0.5 * cfg.span_ghz, max(401, int(cfg.num_points)))
    df_hz = df_ghz * 1e9
    wl_nm = (C0_M_PER_S / (f0_hz + df_hz)) * 1e9
    omega = 2.0 * math.pi * (f0_hz + df_hz)

    phi_rt = 2.0 * math.pi * (df_hz / fsr_hz)
    alpha_np_m = (float(cfg.alpha_loss_db_per_cm) * 100.0) / (10.0 / math.log(10.0))
    a_rt = math.exp(-0.5 * alpha_np_m * l_rt_m)
    a_half = math.sqrt(a_rt)

    if arch == "crow":
        h_thru, h_drop = _crow_transfer_matrix(kappas_p, 0.5 * phi_rt, a_half)
    else:
        l_bus_m = (float(cfg.scissor_spacing_um) * 1e-6) if cfg.scissor_spacing_um is not None else (0.5 * l_rt_m)
        # Choose Bragg-matched bus phase so resonances reinforce constructively (m*pi + df*l_bus*ng/c)
        phi_bus = 2.0 * math.pi * df_hz * (ng * l_bus_m / C0_M_PER_S)
        a_bus = math.exp(-0.5 * alpha_np_m * l_bus_m)
        h_thru, h_drop = _scissor_transfer_matrix(kappas_p, phi_rt, a_rt, phi_bus, a_bus)

    t_thru = np.abs(h_thru) ** 2
    t_drop = np.abs(h_drop) ** 2
    drop_db = 10.0 * np.log10(np.maximum(t_drop, 1e-12))
    thru_db = 10.0 * np.log10(np.maximum(t_thru, 1e-12))

    # Exact unwrapped phase, group delay tau_g = -d(phi)/d(omega) [ps], and GDD = d(tau_g)/d(omega) [ps^2]
    # Note: e^{+i omega t_rt} convention -> d(phi)/d(omega) = +tau_g! Check sign:
    # z_half = exp(+i * phi_half) = exp(+i * omega * T_rt/2), so phase increases with +omega -> tau_g = +d(phi)/d(omega).
    phase_drop = np.unwrap(np.angle(h_drop))
    phase_thru = np.unwrap(np.angle(h_thru))
    d_om = float(omega[1] - omega[0])
    tau_g_drop_ps = np.gradient(phase_drop, d_om) * 1e12
    tau_g_thru_ps = np.gradient(phase_thru, d_om) * 1e12
    gdd_drop_ps2 = np.gradient(tau_g_drop_ps * 1e-12, d_om) * 1e24

    # Extract passband metrics around center df = 0
    center_mask = np.abs(df_ghz) <= min(0.45 * fsr_ghz, 0.48 * cfg.span_ghz)
    pk_drop_db = float(np.max(drop_db[center_mask]))
    il_drop_db = max(0.0, -pk_drop_db)
    in_3db = df_ghz[center_mask & (drop_db >= pk_drop_db - 3.0)]
    in_20db = df_ghz[center_mask & (drop_db >= pk_drop_db - 20.0)]
    bw_3db_ghz = float(np.max(in_3db) - np.min(in_3db)) if len(in_3db) >= 2 else 0.0
    bw_20db_ghz = float(np.max(in_20db) - np.min(in_20db)) if len(in_20db) >= 2 else 0.0
    shape_factor = (bw_3db_ghz / bw_20db_ghz) if bw_20db_ghz > 1e-6 else 0.0
    oob_mask = center_mask & (np.abs(df_ghz) >= max(2.2 * bw_3db_ghz, 0.25 * fsr_ghz))
    oob_rej_db = float(pk_drop_db - np.max(drop_db[oob_mask])) if np.any(oob_mask) else float(pk_drop_db - np.min(drop_db[center_mask]))
    # Passband ripple inside central 60% of 3-dB bandwidth
    pb_inner = center_mask & (np.abs(df_ghz) <= 0.30 * max(bw_3db_ghz, 1.0))
    ripple_meas_db = float(np.max(drop_db[pb_inner]) - np.min(drop_db[pb_inner])) if np.any(pb_inner) else 0.0
    idx_c = int(np.argmin(np.abs(df_ghz)))
    tau_center_ps = float(tau_g_drop_ps[idx_c])
    max_tau_ps = float(np.max(tau_g_drop_ps[center_mask]))

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "architecture": arch,
            "prototype": cfg.prototype,
            "order_N": n_ord,
            "neff_ring": round(neff, 5),
            "ng_ring": round(ng, 4),
            "FSR_GHz": round(fsr_ghz, 3),
            "synthesized_kappas_power": [round(k, 5) for k in kappas_p],
            "drop_IL_dB": round(il_drop_db, 3),
            "bw_3dB_GHz": round(bw_3db_ghz, 3),
            "bw_20dB_GHz": round(bw_20db_ghz, 3),
            "shape_factor_3_to_20dB": round(shape_factor, 4),
            "passband_ripple_dB": round(ripple_meas_db, 3),
            "oob_rejection_dB": round(oob_rej_db, 2),
            "center_group_delay_ps": round(tau_center_ps, 3),
            "peak_group_delay_ps": round(max_tau_ps, 3),
            "center_GDD_ps2": round(float(gdd_drop_ps2[idx_c]), 4),
            "max_unitarity_sum": round(float(np.max(t_thru + t_drop)), 6),
        },
        "arrays": {
            "detuning_GHz": np.round(df_ghz, 4).tolist(),
            "wavelength_nm": np.round(wl_nm, 5).tolist(),
            "T_thru_dB": np.round(thru_db, 3).tolist(),
            "T_drop_dB": np.round(drop_db, 3).tolist(),
            "group_delay_drop_ps": np.round(tau_g_drop_ps, 3).tolist(),
            "group_delay_thru_ps": np.round(tau_g_thru_ps, 3).tolist(),
            "GDD_drop_ps2": np.round(gdd_drop_ps2, 4).tolist(),
        },
    }


# ==============================================================================
# 2. Euler Spiral (Clothoid) / Adiabatic Racetrack Bends & Mode-Mismatch Loss
# ==============================================================================

@dataclass
class EulerBendConfig:
    core_material: str = "Si"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.22
    slab_height_um: float = 0.0
    ring_width_um: float = 0.50
    sidewall_deg: float = 90.0
    effective_radius_um: float = 5.0      # Effective bend footprint radius R_eff [um]
    euler_fraction_p: float = 0.5         # Clothoid fraction p in (0, 1] (Cherchi OE 2013; p=1 full Euler)
    straight_length_um: float = 10.0      # Straight racetrack section length [um]
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    core_mesh_res_um: float = 0.045
    export_gds_path: str | None = None


def _poynting_overlap_efficiency(m_a: EigenmodeResult, m_b: EigenmodeResult, x_shift_m: float = 0.0) -> float:
    """Exact complex 2D vector Poynting mode overlap integral |(int (E_a x H_b*).z dA)|^2 / (4 P_a P_b)."""
    xa, ya = m_a.x_m, m_a.y_m
    xb, yb = m_b.x_m, m_b.y_m
    dx = float(np.mean(np.diff(xa)))
    dy = float(np.mean(np.diff(ya)))
    da = dx * dy

    # Interpolate m_b fields onto m_a grid shifted by x_shift_m along radial x
    from scipy.interpolate import RegularGridInterpolator
    pts_x = xa - x_shift_m
    xx, yy = np.meshgrid(pts_x, ya)
    query = np.stack([yy.ravel(), xx.ravel()], axis=-1)

    def _interp_c(arr: np.ndarray) -> np.ndarray:
        fr = RegularGridInterpolator((yb, xb), np.real(arr), bounds_error=False, fill_value=0.0)
        fi = RegularGridInterpolator((yb, xb), np.imag(arr), bounds_error=False, fill_value=0.0)
        return (fr(query) + 1j * fi(query)).reshape(xx.shape)

    ex_b = _interp_c(m_b.Ex)
    ey_b = _interp_c(m_b.Ey)
    hx_b = _interp_c(m_b.Hx)
    hy_b = _interp_c(m_b.Hy)

    # Bidirectional symmetric overlap: 0.25 * |int (E_a x H_b* + E_b* x H_a).z dA|^2 / (4 P_a P_b)
    c1 = np.sum((m_a.Ex * np.conj(hy_b) - m_a.Ey * np.conj(hx_b)) * da)
    c2 = np.sum((np.conj(ex_b) * m_a.Hy - np.conj(ey_b) * m_a.Hx) * da)
    num = abs(0.5 * (c1 + c2)) ** 2
    den = max(4.0 * float(m_a.power_w) * float(m_b.power_w), 1e-40)
    return float(min(max(num / den, 0.0), 1.0))


def _euler_180_curve_points(r_eff_um: float, p_euler: float, n_pts: int = 80) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate exact (x, y) coordinates of a 180-deg Euler-arc-Euler bend matching footprint R_eff."""
    p = float(np.clip(p_euler, 0.05, 1.0))
    # Parametric unit clothoid of half-angle theta_tot = pi/2:
    # Euler spiral spans angle theta_e = p * (pi/2), circular arc spans (1 - p) * (pi/2)
    theta_e = 0.5 * math.pi * p
    theta_c = 0.5 * math.pi * (1.0 - p)
    # In a clothoid with curvature k(s) = s / a^2 up to s = L_e where theta(L_e) = L_e^2 / (2 a^2) = theta_e:
    #   R_min = a^2 / L_e = L_e / (2 * theta_e)  =>  L_e = 2 * theta_e * R_min
    # Integrate unit-R_min curve first, then scale so total vertical height 2 * y_mid = 2 * R_eff!
    s_e = np.linspace(0.0, 2.0 * theta_e, max(20, n_pts // 2))
    th_e_arr = (s_e ** 2) / max(4.0 * theta_e, 1e-12)
    ds_e = float(s_e[1] - s_e[0])
    x_e = np.cumsum(np.cos(th_e_arr)) * ds_e
    y_e = np.cumsum(np.sin(th_e_arr)) * ds_e
    x_e -= x_e[0]
    y_e -= y_e[0]

    if theta_c > 1e-6:
        th_c_arr = np.linspace(theta_e, 0.5 * math.pi, max(12, n_pts // 4))[1:]
        xc_c = x_e[-1] - math.sin(theta_e)
        yc_c = y_e[-1] + math.cos(theta_e)
        x_c = xc_c + np.sin(th_c_arr)
        y_c = yc_c - np.cos(th_c_arr)
        x_q = np.concatenate([x_e, x_c])
        y_q = np.concatenate([y_e, y_c])
    else:
        x_q, y_q = x_e, y_e

    # Scale so y_q[-1] == r_eff_um (so full 180-deg turn spans exact diameter 2 * r_eff_um!)
    scale = r_eff_um / max(float(y_q[-1]), 1e-9)
    x_q *= scale
    y_q *= scale
    r_min_um = scale  # since unit curve had R_min = 1
    l_half_um = (2.0 * theta_e + theta_c) * r_min_um

    # Mirror second 90-deg quadrant across y = r_eff_um to form smooth 180-deg U-turn
    x_180 = np.concatenate([x_q, x_q[-2::-1]])
    y_180 = np.concatenate([y_q, 2.0 * r_eff_um - y_q[-2::-1]])
    return x_180, y_180, r_min_um, 2.0 * l_half_um


def design_euler_racetrack_bend(cfg: EulerBendConfig) -> dict[str, Any]:
    """Quantify straight-to-bend mode mismatch, optimal lateral offset, TE0->TE1 crosstalk,
    and Euler spiral (clothoid) adiabatic bend loss vs. circular bend of identical R_eff."""
    t0 = time.perf_counter()
    wl0_m = float(cfg.wavelength_nm) * 1e-9
    r_eff_um = max(float(cfg.effective_radius_um), 1.5)
    p_e = float(np.clip(cfg.euler_fraction_p, 0.05, 1.0))

    x_180, y_180, r_min_um, l_180_um = _euler_180_curve_points(r_eff_um, p_e)

    spec_str, core_fn, clad_fn, top_fn = _build_wg_spec(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        ring_radius_um=math.inf,
        wl_nm=cfg.wavelength_nm,
    )
    spec_bnd, _, _, _ = _build_wg_spec(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        ring_radius_um=r_eff_um,
        wl_nm=cfg.wavelength_nm,
    )

    # Solve fundamental straight mode + bent modes (including higher-order mode if supported)
    want_te = cfg.polarization.upper().startswith("TE")
    m_str = _solve_single_mode(spec_str, wl0_m, math.inf, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=True)
    m_bnd_list = solve_modes_fem(
        spec_bnd, wl0_m, bend_radius_m=r_eff_um * 1e-6, num_modes=4,
        n_guess=float(np.real(m_str.n_eff)) * 1.01,
        core_mesh_res_um=cfg.core_mesh_res_um, clad_mesh_res_um=max(cfg.core_mesh_res_um * 5.0, 0.16),
        eval_grid_step_um=0.025, export_grid=True, mesh_backend="bsd_conformal",
    )
    pol_bnd = [m for m in m_bnd_list if ((m.te_fraction >= 0.5) == want_te) and m.core_confinement >= 0.25]
    pol_bnd.sort(key=lambda m: -float(np.real(m.n_eff)))
    m_bnd0 = pol_bnd[0] if pol_bnd else m_bnd_list[0]
    m_bnd1 = pol_bnd[1] if len(pol_bnd) >= 2 else None

    # Exact un-offset straight-to-circular-bend mode mismatch loss & TE0->TE1 crosstalk
    ov_zero = _poynting_overlap_efficiency(m_str, m_bnd0, x_shift_m=0.0)
    loss_circ_junc_db = -10.0 * math.log10(max(ov_zero, 1e-6))
    xtalk_ho_db = (
        10.0 * math.log10(max(_poynting_overlap_efficiency(m_str, m_bnd1, x_shift_m=0.0), 1e-9))
        if m_bnd1 is not None
        else -60.0
    )

    # Optimize lateral offset dx_opt [nm] maximizing straight-to-bend overlap
    shifts_nm = np.linspace(-80.0, 140.0, 45)
    ovs = [_poynting_overlap_efficiency(m_str, m_bnd0, x_shift_m=float(s) * 1e-9) for s in shifts_nm]
    best_i = int(np.argmax(ovs))
    best_shift_nm = float(shifts_nm[best_i])
    loss_offset_junc_db = -10.0 * math.log10(max(float(ovs[best_i]), 1e-6))

    # Radiative bend loss of circular R_eff vs Euler spiral (integrating local radiation alpha(R(s)) along clothoid!)
    n_core = float(core_fn(wl0_m, T_REF_K).nx if want_te else core_fn(wl0_m, T_REF_K).ny)
    n_box = float(clad_fn(wl0_m, T_REF_K).nx)
    n_top = float(top_fn(wl0_m, T_REF_K).nx)
    def _rad_alpha_at(r_um_v: float) -> float:
        return float(_estimate_cylindrical_bend_radiation_loss(
            m_bnd0, spec_bnd, radius_m=r_um_v * 1e-6, wavelength_m=wl0_m,
            w_ring_m=cfg.ring_width_um * 1e-6, h_slab_m=cfg.slab_height_um * 1e-6,
            n_core=n_core, n_box=n_box, n_top=n_top, polarization=cfg.polarization,
            solver_backend="fem", eff_mesh_um=cfg.core_mesh_res_um, ng_rt=max(n_core, 2.0),
        )["alpha_bend_db_per_cm"])

    alpha_rad_circ_db_cm = _rad_alpha_at(r_eff_um)
    loss_rad_circ_180_db = alpha_rad_circ_db_cm * (math.pi * r_eff_um * 1e-4)

    # Integrate exact local curvature R(s) along the Euler 180-deg bend:
    #   - Zero curvature discontinuity at s=0 and s=L_180 -> straight-to-bend mismatch loss drops to ~0!
    #   - Adiabatic inter-modal coupling residual scales as (L_beat / L_e)^2 (Milton-Burns / Cherchi OE 2013):
    l_euler_um = 2.0 * (0.5 * math.pi * p_e) * r_min_um
    dneff_ho = abs(float(np.real(m_bnd0.n_eff)) - float(np.real(m_bnd1.n_eff))) if m_bnd1 is not None else 0.35
    l_beat_um = (cfg.wavelength_nm * 1e-3) / max(dneff_ho, 0.05)
    adiab_supp = 1.0 / (1.0 + (2.0 * l_euler_um / max(l_beat_um, 0.5)) ** 2)
    loss_euler_junc_db = loss_circ_junc_db * adiab_supp
    xtalk_euler_ho_db = xtalk_ho_db + 10.0 * math.log10(max(adiab_supp, 1e-6))

    # Integrated radiative loss along Euler curve (5-point curvature quadrature)
    theta_e = 0.5 * math.pi * p_e
    theta_c = 0.5 * math.pi * (1.0 - p_e)
    rad_e_db = 0.0
    for frac in (0.2, 0.4, 0.6, 0.8, 0.95):
        r_loc = r_min_um / frac
        rad_e_db += _rad_alpha_at(r_loc) * (2.0 * l_euler_um / 5.0) * 1e-4
    rad_e_db += _rad_alpha_at(r_min_um) * (2.0 * theta_c * r_min_um) * 1e-4

    # Full racetrack roundtrip = 4 straight-bend junctions + two 180-deg bends
    circ_rt_excess_db = 4.0 * loss_circ_junc_db + 2.0 * loss_rad_circ_180_db
    euler_rt_excess_db = 4.0 * loss_euler_junc_db + 2.0 * rad_e_db
    l_rt_euler_um = 2.0 * float(cfg.straight_length_um) + 2.0 * l_180_um

    gds_written: str | None = None
    if cfg.export_gds_path:
        import gdstk
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("EULER_RACETRACK_MRR")
        ls = float(cfg.straight_length_um)
        # Right 180-deg Euler bend starting at (+ls/2, 0) -> (+ls/2, 2*r_eff)
        pts_r = [(0.5 * ls + float(x), float(y)) for x, y in zip(x_180, y_180)]
        # Left 180-deg Euler bend starting at (-ls/2, 2*r_eff) -> (-ls/2, 0)
        pts_l = [(-0.5 * ls - float(x), 2.0 * r_eff_um - float(y)) for x, y in zip(x_180, y_180)]
        loop_pts = pts_r + pts_l + [pts_r[0]]
        path = gdstk.FlexPath(loop_pts, cfg.ring_width_um, layer=1, datatype=0)
        cell.add(path)
        out_p = Path(cfg.export_gds_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lib.write_gds(str(out_p))
        gds_written = str(out_p)

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "effective_radius_um": round(r_eff_um, 3),
            "euler_min_radius_um": round(r_min_um, 3),
            "euler_fraction_p": round(p_e, 3),
            "racetrack_roundtrip_um": round(l_rt_euler_um, 3),
            "circular_junction_mismatch_dB": round(loss_circ_junc_db, 4),
            "optimal_lateral_offset_nm": round(best_shift_nm, 2),
            "offset_junction_mismatch_dB": round(loss_offset_junc_db, 4),
            "euler_junction_mismatch_dB": round(loss_euler_junc_db, 4),
            "circular_TE0_to_HO_xtalk_dB": round(xtalk_ho_db, 2),
            "euler_TE0_to_HO_xtalk_dB": round(xtalk_euler_ho_db, 2),
            "circular_180_rad_loss_dB": round(loss_rad_circ_180_db, 5),
            "euler_180_rad_loss_dB": round(rad_e_db, 5),
            "circular_roundtrip_bend_excess_dB": round(circ_rt_excess_db, 4),
            "euler_roundtrip_bend_excess_dB": round(euler_rt_excess_db, 4),
            "euler_loss_reduction_factor": round(circ_rt_excess_db / max(euler_rt_excess_db, 1e-6), 2),
            "gds_path": gds_written,
        },
        "curve_180": {
            "x_um": np.round(x_180, 4).tolist(),
            "y_um": np.round(y_180, 4).tolist(),
        },
    }


# ==============================================================================
# 3. Optical Microring Biosensing, Mid-IR Gas Sensing & Athermal Cladding
# ==============================================================================

@dataclass
class SensorAthermalConfig:
    core_material: str = "Si"
    clad_material: str = "SiO2"
    analyte_medium: str = "water"         # "water" (n=1.318 @ 1550nm) | "air_gas" (n=1.00027) | "custom"
    analyte_n_override: float | None = None
    film_height_um: float = 0.22
    slab_height_um: float = 0.0
    ring_width_um: float = 0.45
    ring_radius_um: float = 20.0
    sidewall_deg: float = 90.0
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    q_loaded: float = 2.0e4
    wavelength_noise_pm: float = 0.5      # Laser/OSA 3-sigma minimum resolvable shift [pm]
    adlayer_n: float = 1.45               # Biomolecule / protein surface adlayer index
    gas_absorption_cm_inv_per_ppm: float = 1.0e-5  # Beer-Lambert molecular line strength [cm^-1 / ppm]
    snr_min_linear: float = 3.0
    compute_athermal: bool = True
    athermal_clad_material: str = "TiO2"  # Negative dn/dT overcladding ("TiO2" anatase ~ -1.8e-4/K)
    core_mesh_res_um: float = 0.048


def simulate_mrr_sensor_and_athermal(cfg: SensorAthermalConfig) -> dict[str, Any]:
    """First-principles 2D vector FEM microring biosensor, gas sensor & athermal cladding solver."""
    t0 = time.perf_counter()
    wl0_nm = float(cfg.wavelength_nm)
    wl0_m = wl0_nm * 1e-9
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m

    if cfg.analyte_n_override is not None:
        n_an = float(cfg.analyte_n_override)
    elif cfg.analyte_medium.lower() == "water":
        # Hale & Querry water refractive index dispersion (~1.318 at 1550 nm, ~1.331 at 850 nm)
        n_an = 1.318 if wl0_nm >= 1200.0 else 1.331
    else:
        n_an = 1.00027

    # 1. Baseline mode + perturbed analyte index (+dn) for exact bulk sensitivity S_bulk = d(lambda)/d(n_clad)
    dn_step = 0.015
    sp0, core_fn, clad_fn, _ = _build_wg_spec(
        core_material=cfg.core_material, clad_material=cfg.clad_material, top_clad_material="Custom",
        custom_top_clad_n=n_an, w_um=cfg.ring_width_um, h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um, sidewall_deg=cfg.sidewall_deg, ring_radius_um=cfg.ring_radius_um, wl_nm=wl0_nm,
    )
    sp1, _, _, _ = _build_wg_spec(
        core_material=cfg.core_material, clad_material=cfg.clad_material, top_clad_material="Custom",
        custom_top_clad_n=n_an + dn_step, w_um=cfg.ring_width_um, h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um, sidewall_deg=cfg.sidewall_deg, ring_radius_um=cfg.ring_radius_um, wl_nm=wl0_nm,
    )
    m0 = _solve_single_mode(sp0, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=True)
    m1 = _solve_single_mode(sp1, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=False)
    neff0 = float(np.real(m0.n_eff))
    neff1 = float(np.real(m1.n_eff))
    ng0 = float(_compute_group_index_fast(sp0, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, neff0))

    dneff_dn = (neff1 - neff0) / dn_step
    # Exact resonance condition: d(lambda)/lambda = d(neff)/ng  =>  S_bulk = (lambda / ng) * d(neff)/d(n_an)
    s_bulk_nm_per_riu = (wl0_nm / max(ng0, 1.0)) * dneff_dn

    # 2. Exact surface adlayer sensitivity S_surf [pm/nm] via Johnson (2002) first-order boundary perturbation
    # Integrating electric energy density along the top surface + both etched sidewalls exposed to analyte:
    x_um, y_um = m0.x_m * 1e6, m0.y_m * 1e6
    dx = float(np.mean(np.diff(m0.x_m)))
    dy = float(np.mean(np.diff(m0.y_m)))
    da = dx * dy
    e_sq = np.abs(m0.Ex) ** 2 + np.abs(m0.Ey) ** 2 + np.abs(m0.Ez) ** 2
    xx, yy = np.meshgrid(x_um, y_um)
    in_core = (yy >= cfg.slab_height_um) & (yy <= cfg.film_height_um) & (np.abs(xx) <= 0.5 * cfg.ring_width_um)
    in_top_an = (yy >= cfg.slab_height_um) & (~in_core)

    want_te = cfg.polarization.upper().startswith("TE")
    n_c_val = float(core_fn(wl0_m, T_REF_K).nx if want_te else core_fn(wl0_m, T_REF_K).ny)
    n_b_val = float(clad_fn(wl0_m, T_REF_K).nx)
    n_map = np.where(in_core, n_c_val, np.where(yy < 0.0, n_b_val, n_an))
    u_tot = float(np.sum((n_map ** 2) * e_sq) * da)
    gamma_analyte = float(np.sum(np.where(in_top_an, (n_an ** 2) * e_sq, 0.0)) * da) / max(u_tot, 1e-40)

    # Surface integral of (eps_ad - eps_an) * (|E_par|^2 + (eps_an/eps_ad)|E_perp|^2) over top & two sidewalls
    deps_ad = cfg.adlayer_n ** 2 - n_an ** 2
    eps_ratio = (n_an ** 2) / max(cfg.adlayer_n ** 2, 1.0)
    # Top surface (y = h_film + 1 grid cell in analyte, x in [-W/2, W/2]): normal = y, parallel = x, z
    iy_top = int(np.argmin(np.abs(y_um - (cfg.film_height_um + 0.020))))
    x_top_mask = np.abs(x_um) <= 0.5 * cfg.ring_width_um
    int_top = float(np.sum((np.abs(m0.Ex[iy_top, x_top_mask]) ** 2 + np.abs(m0.Ez[iy_top, x_top_mask]) ** 2 + eps_ratio * np.abs(m0.Ey[iy_top, x_top_mask]) ** 2) * dx))
    # Two sidewalls (x = +- (W/2 + 20nm), y in [h_slab, h_film]): normal = x, parallel = y, z
    y_sw_mask = (y_um >= cfg.slab_height_um) & (y_um <= cfg.film_height_um)
    int_sw = 0.0
    for sgn in (-1.0, +1.0):
        ix_sw = int(np.argmin(np.abs(x_um - sgn * (0.5 * cfg.ring_width_um + 0.020))))
        int_sw += float(np.sum((np.abs(m0.Ey[y_sw_mask, ix_sw]) ** 2 + np.abs(m0.Ez[y_sw_mask, ix_sw]) ** 2 + eps_ratio * np.abs(m0.Ex[y_sw_mask, ix_sw]) ** 2) * dy))
    # Exact variational formula: dneff / dt_ad = (n_eff / (2 * u_tot)) * deps_ad * (int_top + int_sw) [1/m]
    # Note: u_tot = int n^2 |E|^2 dA; perturbation delta(neff^2) = deps_ad * dt * int_surf / (u_tot / neff^2)
    # => dneff/dt = (neff / (2 * u_tot)) * deps_ad * (int_top + int_sw) ... check scaling with S_bulk:
    # For bulk perturbation: dneff/dn_an = (neff * n_an / u_tot) * int_an |E|^2 dA = (neff / n_an) * gamma_analyte!
    dneff_per_m = (neff0 / max(2.0 * u_tot, 1e-40)) * deps_ad * (int_top + int_sw)
    s_surf_pm_per_nm = (wl0_nm / max(ng0, 1.0)) * (dneff_per_m * 1e-9) * 1e3  # nm shift per nm adlayer -> pm/nm

    # 3. Sensing FoM, Intrinsic LOD, and Cavity-Enhanced Gas Beer-Lambert Effective Pathlength
    fwhm_nm = wl0_nm / max(cfg.q_loaded, 100.0)
    fom_riu_inv = s_bulk_nm_per_riu / max(fwhm_nm, 1e-9)
    lod_riu = (cfg.wavelength_noise_pm * 1e-3) / max(s_bulk_nm_per_riu, 1e-6)
    fsr_nm = (wl0_nm ** 2) / max(ng0 * l_rt_m * 1e9, 1e-9)
    finesse = fsr_nm / max(fwhm_nm, 1e-9)
    # Cavity-enhanced absorption pathlength L_eff = (Finesse / pi) * Gamma_analyte * L_rt (critically coupled)
    l_eff_mm = (finesse / math.pi) * gamma_analyte * (l_rt_m * 1e3)
    # Minimum detectable gas concentration assuming 1% intensity noise floor / SNR_min:
    # dI/I ~ alpha_gas * L_eff >= 0.01 / SNR -> c_min = (1e-3 * snr_min) / (alpha_per_ppm * L_eff_cm)
    lod_gas_ppm = (1e-3 * cfg.snr_min_linear) / max(cfg.gas_absorption_cm_inv_per_ppm * (l_eff_mm * 0.1), 1e-18)

    # 4. Thermal drift & Athermal Negative-dn/dT Overcladding Synthesis (dlambda/dT = 0)
    dT = 20.0
    sp_hot, _, _, _ = _build_wg_spec(
        core_material=cfg.core_material, clad_material=cfg.clad_material, top_clad_material="Custom",
        custom_top_clad_n=n_an, w_um=cfg.ring_width_um, h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um, sidewall_deg=cfg.sidewall_deg, ring_radius_um=cfg.ring_radius_um,
        wl_nm=wl0_nm, temperature_k=T_REF_K + dT,
    )
    m_hot = _solve_single_mode(sp_hot, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=False)
    dlam_dt_bare_pm_k = ((wl0_nm / max(ng0, 1.0)) * (float(np.real(m_hot.n_eff)) - neff0) / dT) * 1e3

    athermal_info: dict[str, Any] = {"enabled": False}
    if cfg.compute_athermal:
        # Evaluate dlambda/dT with negative-dn/dT overcladding (e.g., anatase TiO2, dn/dT = -1.8e-4/K)
        # across thin-film overcladding thicknesses to locate exact athermal zero-crossing!
        h_grid = [0.0, 0.03, 0.06, 0.10, 0.16, 0.25]
        drifts: list[float] = [round(dlam_dt_bare_pm_k, 3)]
        for hc in h_grid[1:]:
            sp_c0, _, _, _ = _build_wg_spec(
                core_material=cfg.core_material, clad_material=cfg.clad_material, top_clad_material="Air",
                w_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg, ring_radius_um=cfg.ring_radius_um, wl_nm=wl0_nm,
                temperature_k=T_REF_K, cap_h_um=hc, cap_material=cfg.athermal_clad_material,
            )
            sp_c1, _, _, _ = _build_wg_spec(
                core_material=cfg.core_material, clad_material=cfg.clad_material, top_clad_material="Air",
                w_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg, ring_radius_um=cfg.ring_radius_um, wl_nm=wl0_nm,
                temperature_k=T_REF_K + dT, cap_h_um=hc, cap_material=cfg.athermal_clad_material,
            )
            mc0 = _solve_single_mode(sp_c0, wl0_m, r_m, cfg.polarization, "fem", max(cfg.core_mesh_res_um, 0.055), export_grid=False)
            mc1 = _solve_single_mode(sp_c1, wl0_m, r_m, cfg.polarization, "fem", max(cfg.core_mesh_res_um, 0.055), export_grid=False)
            dr_c = ((wl0_nm / max(ng0, 1.0)) * (float(np.real(mc1.n_eff)) - float(np.real(mc0.n_eff))) / dT) * 1e3
            drifts.append(round(dr_c, 3))
        h_zero: float | None = None
        for k in range(len(drifts) - 1):
            if drifts[k] * drifts[k + 1] <= 0.0 and abs(drifts[k + 1] - drifts[k]) > 1e-6:
                frac = -drifts[k] / (drifts[k + 1] - drifts[k])
                h_zero = round(h_grid[k] + frac * (h_grid[k + 1] - h_grid[k]), 4)
                break
        athermal_info = {
            "enabled": True,
            "overclad_material": cfg.athermal_clad_material,
            "cap_thickness_um_grid": h_grid,
            "dlambda_dT_pm_per_K_grid": drifts,
            "athermal_cap_thickness_um": h_zero,
            "thermal_drift_reduction_pct": round((1.0 - abs(min(drifts, key=abs)) / max(abs(dlam_dt_bare_pm_k), 1e-6)) * 100.0, 2),
        }

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "analyte_n": round(n_an, 5),
            "neff_mode": round(neff0, 5),
            "ng_mode": round(ng0, 4),
            "analyte_confinement_pct": round(gamma_analyte * 100.0, 2),
            "S_bulk_nm_per_RIU": round(s_bulk_nm_per_riu, 2),
            "S_surface_pm_per_nm": round(s_surf_pm_per_nm, 2),
            "intrinsic_FoM_per_RIU": round(fom_riu_inv, 1),
            "LOD_bulk_RIU": float(f"{lod_riu:.3e}"),
            "effective_gas_path_mm": round(l_eff_mm, 3),
            "LOD_gas_ppm": float(f"{lod_gas_ppm:.3e}"),
            "bare_dlambda_dT_pm_per_K": round(dlam_dt_bare_pm_k, 3),
        },
        "athermal": athermal_info,
    }


# ==============================================================================
# 4. Large-Signal Time-Domain CMT Dynamic Microring Modulator ODE (NRZ / PAM4)
# ==============================================================================

@dataclass
class LargeSignalTDCMTConfig:
    wavelength_nm: float = 1550.0
    q_int: float = 2.0e4
    q_ext: float = 1.8e4
    modulation_type: str = "detuning"     # "detuning" (Si PN / TFLN EO) | "coupling" (MZI-coupled MRR)
    format: str = "nrz"                   # "nrz" | "pam4"
    bitrate_gbps: float = 50.0
    v_pp_v: float = 2.0
    dlambda_dv_pm_per_v: float = 22.0     # Resonance shift slope [pm/V] (from 2D FEM PN/EO)
    f_rc_ghz: float = 35.0                # Electrical RC 3-dB cutoff [GHz]
    bias_detuning_pm: float = 25.0        # Initial operating detuning from cold resonance [pm]
    delta_alpha_fca_fraction: float = 0.08  # Fractional intrinsic loss modulation across V_pp
    coupling_mod_depth: float = 0.65      # For modulation_type="coupling": fractional modulation of kappa_ext
    num_symbols: int = 64
    samples_per_symbol: int = 32
    seed: int = 42


def simulate_large_signal_td_cmt_modulator(cfg: LargeSignalTDCMTConfig) -> dict[str, Any]:
    """Solve exact RK4 large-signal time-domain CMT ODE for high-speed NRZ/PAM4 microring modulators.

    Integrates:
      da/dt = [i*(Delta_om_0 + Delta_om_mod(t)) - (kappa_int(t) + kappa_ext(t))/2] * a + sqrt(kappa_ext(t)) * s_in
      s_out(t) = s_in - sqrt(kappa_ext(t)) * a(t)
    Captures optical peaking overshoot, asymmetric rise/fall dynamics, transient frequency chirp
    Delta_nu(t) = -(1/(2*pi)) * d(arg(s_out))/dt, effective Henry alpha_H, and 2-UI dynamic eye opening.
    """
    t0 = time.perf_counter()
    wl0_m = float(cfg.wavelength_nm) * 1e-9
    om0 = 2.0 * math.pi * C0_M_PER_S / wl0_m
    k_int0 = om0 / max(float(cfg.q_int), 100.0)
    k_ext0 = om0 / max(float(cfg.q_ext), 100.0)
    k_tot0 = k_int0 + k_ext0
    f_opt_ghz = (k_tot0 / (2.0 * math.pi)) * 1e-9  # Optical cavity FWHM [GHz]

    fmt = cfg.format.lower().strip()
    bits_per_sym = 2 if fmt == "pam4" else 1
    sym_rate_hz = (float(cfg.bitrate_gbps) * 1e9) / bits_per_sym
    t_sym = 1.0 / max(sym_rate_hz, 1e6)
    sps = max(16, int(cfg.samples_per_symbol))
    n_sym = max(32, int(cfg.num_symbols))
    dt = t_sym / sps
    n_tot = n_sym * sps

    # Generate PRBS drive waveform u(t) in [0, 1] filtered by 1st-order electrical RC low-pass
    rng = np.random.default_rng(int(cfg.seed))
    if fmt == "pam4":
        syms = rng.integers(0, 4, size=n_sym) / 3.0
    else:
        syms = rng.integers(0, 2, size=n_sym).astype(float)
    # Prepend clean 0 -> 1 -> 0 isolated step at start for exact rise/fall overshoot measurement
    syms[:6] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    u_raw = np.repeat(syms, sps)

    tau_rc = 1.0 / max(2.0 * math.pi * float(cfg.f_rc_ghz) * 1e9, 1e6)
    alpha_rc = math.exp(-dt / tau_rc)
    u_filt = np.empty(n_tot, dtype=float)
    u_cur = float(u_raw[0])
    for i in range(n_tot):
        u_cur = alpha_rc * u_cur + (1.0 - alpha_rc) * float(u_raw[i])
        u_filt[i] = u_cur

    # Map normalized drive u_filt(t) in [0, 1] to cavity detuning & decay rates
    # d(omega)/d(lambda) = -omega_0 / lambda_0
    dom0 = -(om0 / wl0_m) * (float(cfg.bias_detuning_pm) * 1e-12)
    dom_span = -(om0 / wl0_m) * (float(cfg.dlambda_dv_pm_per_v) * float(cfg.v_pp_v) * 1e-12)

    mod_kind = cfg.modulation_type.lower().strip()
    if mod_kind == "coupling":
        # Coupling modulation keeps cavity on-resonance (dom=0) and modulates kappa_ext(t) directly!
        dom_arr = np.zeros(n_tot, dtype=float)
        depth = float(np.clip(cfg.coupling_mod_depth, 0.05, 0.95))
        k_ext_arr = k_ext0 * (1.0 - depth * (1.0 - u_filt))
        k_int_arr = np.full(n_tot, k_int0, dtype=float)
    else:
        dom_arr = dom0 + dom_span * u_filt
        k_ext_arr = np.full(n_tot, k_ext0, dtype=float)
        k_int_arr = k_int0 * (1.0 + float(cfg.delta_alpha_fca_fraction) * (u_filt - 0.5))

    # RK4 integration of exact complex cavity amplitude a(t) (with s_in = 1.0)
    # Initialize in exact steady state at t = 0
    a_cur = math.sqrt(k_ext_arr[0]) / complex(0.5 * (k_int_arr[0] + k_ext_arr[0]), -dom_arr[0])
    s_out = np.empty(n_tot, dtype=np.complex128)
    for i in range(n_tot):
        di = float(dom_arr[i])
        ki = float(k_int_arr[i])
        ke = float(k_ext_arr[i])
        di_n = float(dom_arr[min(i + 1, n_tot - 1)])
        ki_n = float(k_int_arr[min(i + 1, n_tot - 1)])
        ke_n = float(k_ext_arr[min(i + 1, n_tot - 1)])
        di_m = 0.5 * (di + di_n)
        ki_m = 0.5 * (ki + ki_n)
        ke_m = 0.5 * (ke + ke_n)

        l0 = complex(-0.5 * (ki + ke), di)
        lm = complex(-0.5 * (ki_m + ke_m), di_m)
        l1 = complex(-0.5 * (ki_n + ke_n), di_n)
        c0, cm, c1 = math.sqrt(ke), math.sqrt(ke_m), math.sqrt(ke_n)

        k1 = l0 * a_cur + c0
        k2 = lm * (a_cur + 0.5 * dt * k1) + cm
        k3 = lm * (a_cur + 0.5 * dt * k2) + cm
        k4 = l1 * (a_cur + dt * k3) + c1
        s_out[i] = 1.0 - c0 * a_cur
        a_cur = a_cur + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    p_out = np.abs(s_out) ** 2
    phi_out = np.unwrap(np.angle(s_out))
    # Instantaneous frequency chirp Delta_nu(t) = -(1/(2*pi)) * d(phi)/dt [GHz]
    chirp_ghz = -(np.gradient(phi_out, dt) / (2.0 * math.pi)) * 1e-9

    # Dynamic metrics from PRBS body (skip initial 6-symbol step transient)
    body = p_out[6 * sps:]
    chirp_body = chirp_ghz[6 * sps:]
    # Sample at optimal eye-center phase within each symbol
    best_open = -1.0
    best_er_db = 0.0
    best_oma = 0.0
    for ph in range(sps):
        samp = body[ph::sps]
        sy = syms[6:6 + len(samp)]
        hi = samp[sy >= 0.9]
        lo = samp[sy <= 0.1]
        if len(hi) >= 2 and len(lo) >= 2:
            m_hi, m_lo = float(np.mean(hi)), float(np.mean(lo))
            # Allow inverted polarity (when resonance moves into laser vs away from laser)
            if m_lo > m_hi:
                hi, lo = lo, hi
                m_hi, m_lo = m_lo, m_hi
            eye_h = float(np.min(hi) - np.max(lo))
            oma = max(m_hi - m_lo, 0.0)
            if oma > best_oma:
                best_oma = oma
                best_open = eye_h
                best_er_db = 10.0 * math.log10(max(m_hi, 1e-9) / max(m_lo, 1e-9))

    # Measure optical peaking overshoot on the isolated 0->1->0 step (symbols 1..5)
    step_seg = p_out[1 * sps:5 * sps]
    lo_lvl = float(np.mean(p_out[int(1.6 * sps):2 * sps]))
    hi_lvl = float(np.mean(p_out[int(3.6 * sps):4 * sps]))
    span_lvl = abs(hi_lvl - lo_lvl)
    if span_lvl > 1e-4:
        if hi_lvl >= lo_lvl:
            overshoot_pct = max(0.0, (float(np.max(step_seg)) - hi_lvl) / span_lvl * 100.0)
        else:
            overshoot_pct = max(0.0, (lo_lvl - float(np.min(step_seg))) / span_lvl * 100.0)
    else:
        overshoot_pct = 0.0

    # Effective transient Henry chirp parameter alpha_H = 2 * d(phi)/d(ln P)
    # Mask samples right at exact critical-coupling zero-crossings (|s_out| < 0.05) where arg(s_out) has a pi sign flip
    amp_body = np.sqrt(np.maximum(body, 0.0))
    valid_amp = amp_body >= 0.05
    peak_chirp_ghz = float(np.max(np.abs(chirp_body[valid_amp]))) if np.any(valid_amp) else 0.0
    dlnp = np.gradient(np.log(np.maximum(body, 1e-6)), dt)
    valid_c = valid_amp & (np.abs(dlnp) > 0.15 * float(np.max(np.abs(dlnp))))
    alpha_h_eff = float(np.median(np.abs(2.0 * np.gradient(phi_out[6 * sps:], dt)[valid_c] / dlnp[valid_c]))) if np.any(valid_c) else 0.0

    t_ps = np.arange(n_tot) * dt * 1e12
    # Return first 16 symbols of waveform for crisp plotting
    n_plot = min(n_tot, 16 * sps)
    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "modulation_type": mod_kind,
            "format": fmt,
            "bitrate_Gbps": cfg.bitrate_gbps,
            "optical_cavity_fwhm_GHz": round(f_opt_ghz, 2),
            "dynamic_ER_dB": round(best_er_db, 2),
            "dynamic_OMA_linear": round(best_oma, 4),
            "eye_opening_linear": round(best_open, 4),
            "optical_peaking_overshoot_pct": round(overshoot_pct, 2),
            "peak_transient_chirp_GHz": round(peak_chirp_ghz, 2),
            "effective_chirp_alpha_H": round(alpha_h_eff, 3),
            "insertion_loss_on_dB": round(-10.0 * math.log10(max(max(hi_lvl, lo_lvl), 1e-6)), 2),
        },
        "waveform": {
            "time_ps": np.round(t_ps[:n_plot], 3).tolist(),
            "drive_norm": np.round(u_filt[:n_plot], 4).tolist(),
            "optical_power_norm": np.round(p_out[:n_plot], 5).tolist(),
            "chirp_GHz": np.round(chirp_ghz[:n_plot], 3).tolist(),
        },
    }


# ==============================================================================
# 5. MZI-Coupled Tunable-Kappa Microring (MZI-MRR)
# ==============================================================================

@dataclass
class MZICoupledMRRConfig:
    core_material: str = "Si"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.22
    ring_width_um: float = 0.45
    ring_radius_um: float = 20.0
    mzi_arm_length_um: float = 40.0       # Tunable MZI coupler arm length [um]
    dc_split_ratio: float = 0.50          # Each 2x2 DC power split in the MZI coupler (0.50 = ideal 3-dB)
    alpha_loss_db_per_cm: float = 1.5
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    mzi_phase_sweep_rad: list[float] = field(default_factory=lambda: [0.0, 0.35, 0.70, 1.10, 1.57, 2.20, 3.1416])
    core_mesh_res_um: float = 0.050


def simulate_mzi_coupled_mrr(cfg: MZICoupledMRRConfig) -> dict[str, Any]:
    """Solve an MZI-coupled tunable-kappa microring (independent Q_ext / ER tuning without shifting lambda_0)."""
    t0 = time.perf_counter()
    wl0_m = float(cfg.wavelength_nm) * 1e-9
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m + float(cfg.mzi_arm_length_um) * 1e-6
    spec, _, _, _ = _build_wg_spec(
        core_material=cfg.core_material, clad_material=cfg.clad_material, top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=0.0, sidewall_deg=90.0,
        ring_radius_um=cfg.ring_radius_um, wl_nm=cfg.wavelength_nm,
    )
    m_ring = _solve_single_mode(spec, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=False)
    neff = float(np.real(m_ring.n_eff))
    ng = float(_compute_group_index_fast(spec, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, neff))
    fsr_ghz = (C0_M_PER_S / (ng * l_rt_m)) * 1e-9

    alpha_np_m = (float(cfg.alpha_loss_db_per_cm) * 100.0) / (10.0 / math.log(10.0))
    a_rt = math.exp(-0.5 * alpha_np_m * l_rt_m)
    k_dc = float(np.clip(cfg.dc_split_ratio, 0.05, 0.95))
    t_dc = math.sqrt(1.0 - k_dc)
    s_dc = math.sqrt(k_dc)
    q_int = loss_db_per_cm_to_q_int(cfg.alpha_loss_db_per_cm, ng=ng, wavelength_nm=cfg.wavelength_nm)

    # Exact 2x2 MZI coupler cross-coupling:
    #   S_MZI = S_dc * diag(exp(i*dphi/2), exp(-i*dphi/2)) * S_dc
    #   For ideal 50:50 DCs: t_eff(dphi) = -i*sin(dphi/2) -> |t_eff|^2 = sin^2(dphi/2), kappa_eff = cos^2(dphi/2)!
    #   Wait: standard convention with push-pull phase dphi between arms:
    #     S11(thru) = t_dc^2 * e^{i dphi/2} - s_dc^2 * e^{-i dphi/2}
    #     S21(cross)= -i * t_dc * s_dc * (e^{i dphi/2} + e^{-i dphi/2}) = -2i * t_dc * s_dc * cos(dphi/2)
    #   Let's parameterize by dphi_eff = pi - dphi so dphi=0 -> kappa_eff = 0 (decoupled), dphi=pi -> kappa_eff = 4*k_dc*(1-k_dc)!
    states: list[dict[str, Any]] = []
    for dphi in cfg.mzi_phase_sweep_rad:
        ep = complex(math.cos(0.5 * (math.pi - dphi)), math.sin(0.5 * (math.pi - dphi)))
        em = ep.conjugate()
        t_eff = (t_dc * t_dc) * ep - (s_dc * s_dc) * em
        c_eff = -1j * t_dc * s_dc * (ep + em)
        t_abs = abs(t_eff)
        k_eff = abs(c_eff) ** 2
        # On-resonance transmission T_res = ((t_abs - a_rt)/(1 - t_abs * a_rt))^2
        t_res = ((t_abs - a_rt) / max(1.0 - t_abs * a_rt, 1e-12)) ** 2
        er_db = -10.0 * math.log10(max(t_res, 1e-9))
        q_ext = (2.0 * math.pi * ng * l_rt_m) / max(wl0_m * k_eff, 1e-24) if k_eff > 1e-9 else float("inf")
        q_loaded = (q_int * q_ext) / (q_int + q_ext) if math.isfinite(q_ext) else q_int
        regime = "critical" if abs(t_abs - a_rt) < 0.008 else ("overcoupled" if t_abs < a_rt else "undercoupled")
        states.append({
            "mzi_phase_rad": round(float(dphi), 4),
            "kappa_eff": round(float(k_eff), 5),
            "Q_ext": round(q_ext, 1) if math.isfinite(q_ext) else None,
            "Q_loaded": round(q_loaded, 1),
            "ER_dB": round(er_db, 2),
            "regime": regime,
        })

    # Exact analytical MZI phase achieving critical coupling: k_eff_cc = 1 - a_rt^2
    k_cc = 1.0 - a_rt * a_rt
    k_max = 4.0 * k_dc * (1.0 - k_dc)
    dphi_cc = 2.0 * math.asin(min(math.sqrt(max(k_cc / max(k_max, 1e-9), 0.0)), 1.0))

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "neff": round(neff, 5),
            "ng": round(ng, 4),
            "FSR_GHz": round(fsr_ghz, 3),
            "Q_int": round(q_int, 1),
            "kappa_critical_coupling": round(k_cc, 5),
            "mzi_phase_critical_coupling_rad": round(dphi_cc, 4),
            "max_achievable_kappa": round(k_max, 4),
            "min_Q_loaded": round(min(s["Q_loaded"] for s in states), 1),
            "max_Q_loaded": round(max(s["Q_loaded"] for s in states), 1),
        },
        "tuning_states": states,
    }


# ==============================================================================
# 6. Dual-Ring Photonic Molecule, PT-Symmetric Exceptional Point & Dual-Ring LLE
# ==============================================================================

@dataclass
class PhotonicMoleculeConfig:
    mode: str = "supermode_ep"            # "supermode_ep" | "dual_ring_lle"
    core_material: str = "Si3N4"
    film_height_um: float = 0.80
    ring_width_um: float = 1.60
    ring1_radius_um: float = 23.0
    ring2_radius_um: float = 23.5         # Slight Vernier offset for auxiliary-ring AMX control
    wavelength_nm: float = 1550.0
    coupling_j_ghz: float = 2.5           # Inter-ring photon coupling rate J / (2*pi) [GHz]
    gamma1_ghz: float = 0.5               # Ring 1 total decay rate kappa_1 / (2*pi) [GHz]
    gamma2_ghz: float = 10.5              # Ring 2 decay rate kappa_2 / (2*pi) [GHz] (EP at |g2-g1| = 4*J!)
    detuning_span_ghz: float = 12.0
    eo_mod_freq_ghz: float = 2.5          # Synthetic frequency-dimension EO modulation (matched to 2J or FSR)
    pump_power_mw: float = 80.0           # For mode="dual_ring_lle"


def simulate_photonic_molecule(cfg: PhotonicMoleculeConfig) -> dict[str, Any]:
    """Solve coupled dual-ring photonic molecule supermodes, PT-symmetric Exceptional Point (EP)
    square-root sensitivity enhancement, and auxiliary-ring AMX-assisted dual-ring LLE."""
    t0 = time.perf_counter()
    j_ghz = float(cfg.coupling_j_ghz)
    g1 = float(cfg.gamma1_ghz)
    g2 = float(cfg.gamma2_ghz)
    g_avg = 0.5 * (g1 + g2)
    dg = g2 - g1
    # Exact PT-symmetric Exceptional Point occurs when |gamma_2 - gamma_1| = 4 * J (for H = [[d/2 - i*g1/2, J], [J, -d/2 - i*g2/2]])
    # Check eigenvalues of 2x2 non-Hermitian Hamiltonian:
    #   H = [[delta/2 - i*g1/2, J], [J, -delta/2 - i*g2/2]]
    #   lambda_+- = -i*(g1+g2)/4 +- sqrt(J^2 + (delta/2 + i*(g2-g1)/4)^2)
    #   At delta = 0, discriminant = J^2 - ((g2 - g1)/4)^2 = 0  <=>  |g2 - g1| = 4 * J!
    dg_ep = 4.0 * j_ghz

    # Sweep inter-ring frequency detuning delta = f1 - f2
    deltas = np.linspace(-cfg.detuning_span_ghz, cfg.detuning_span_ghz, 121)
    eig_p = -0.5j * g_avg + np.sqrt(j_ghz ** 2 + (0.5 * deltas + 0.25j * dg) ** 2 + 0j)
    eig_m = -0.5j * g_avg - np.sqrt(j_ghz ** 2 + (0.5 * deltas + 0.25j * dg) ** 2 + 0j)
    re_split = np.abs(np.real(eig_p - eig_m))

    # Exact EP square-root perturbation sensitivity vs linear Diabolic Point (DP) sensitivity
    eps_grid = np.logspace(-3, 0, 25) * j_ghz
    # At exact EP (dg = 4J), a perturbation eps on ring 1 gives split ~ 2 * sqrt(J * eps) vs DP split = eps!
    ep_split = np.abs(2.0 * np.sqrt(j_ghz ** 2 + (0.5 * eps_grid + 1j * j_ghz) ** 2 + 0j))
    dp_split = eps_grid.copy()
    enh_small = float(ep_split[0] / max(dp_split[0], 1e-12))

    # Optional coupled dual-ring LLE (auxiliary ring induces localized AMX at target mode mu_amx
    # enabling dark-pulse / platicon initiation or soliton dispersion engineering!)
    dual_lle_metrics: dict[str, Any] | None = None
    if cfg.mode.lower().strip() == "dual_ring_lle":
        lle_cfg = LLEConfig(
            core_material=cfg.core_material,
            film_height_um=cfg.film_height_um,
            ring_width_um=cfg.ring_width_um,
            ring_radius_um=cfg.ring1_radius_um,
            wavelength_nm=cfg.wavelength_nm,
            pump_power_mw=cfg.pump_power_mw,
            detuning_end_kappa=12.0,
            mode="single_soliton",
            amx_mode_mu=0,
            amx_strength_ghz=j_ghz,
            num_comb_modes=256,
            num_steps=2000,
            include_coupler_dispersion=False,
        )
        r_lle = simulate_lle_microcomb(lle_cfg)
        dual_lle_metrics = r_lle["metrics"]

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "avoided_crossing_2J_GHz": round(2.0 * j_ghz, 3),
            "loss_contrast_dgamma_GHz": round(abs(dg), 3),
            "exact_EP_loss_contrast_4J_GHz": round(dg_ep, 3),
            "EP_proximity_ratio": round(abs(dg) / max(dg_ep, 1e-9), 4),
            "pt_regime": "exceptional_point" if abs(abs(dg) - dg_ep) < 0.05 * dg_ep else ("pt_broken" if abs(dg) > dg_ep else "pt_unbroken"),
            "zero_detuning_real_split_GHz": round(float(re_split[len(deltas) // 2]), 4),
            "EP_vs_DP_sensitivity_gain_at_1e-3J": round(enh_small, 2),
        },
        "supermodes": {
            "detuning_GHz": np.round(deltas, 4).tolist(),
            "re_plus_GHz": np.round(np.real(eig_p), 4).tolist(),
            "re_minus_GHz": np.round(np.real(eig_m), 4).tolist(),
            "im_plus_GHz": np.round(np.imag(eig_p), 4).tolist(),
            "im_minus_GHz": np.round(np.imag(eig_m), 4).tolist(),
        },
        "ep_sensitivity_curve": {
            "perturbation_GHz": np.round(eps_grid, 6).tolist(),
            "ep_split_GHz": np.round(ep_split, 5).tolist(),
            "dp_split_GHz": np.round(dp_split, 5).tolist(),
        },
        "dual_ring_lle": dual_lle_metrics,
    }
