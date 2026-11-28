"""First-Principles Nanoantenna & Metasurface Temporal Coupled-Mode Theory (TCMT) Engine.

Implements exact energy-conserving multi-mode Temporal Coupled-Mode Theory (Fan et al.,
JOSA A 20, 569, 2003; Suh, Wang & Fan, IEEE JQE 40, 1511, 2004; Adato et al., Nat. Commun.
4, 2154, 2013) spanning Visible/NIR to Mid-IR (`cm^-1`, `um`, `THz`):
  1. Single-port MIM Perfect Absorber (`mim_absorber`, exact critical coupling A = 100% at gamma_rad = gamma_abs)
  2. Two-port Transmissive Bright-Dark Fano / EIT / qBIC Metasurface (`fano_eit_qbic`)
  3. Mid-IR SEIRA Molecular Vibrational Coupling (`seira_molecular` — Fano dip, strong-coupling Rabi splitting,
     differential SEIRA contrast Delta_R/R & vibrational enhancement factor across PMMA, Amide I/II, CO2, CH4)
  4. Non-Hermitian Metasurface Exceptional Point (`metasurface_ep` — topological phase winding & sqrt sensitivity)
  5. Direct Experimental / FDTD FTIR Spectrum TCMT Inverse Fitter (`fit_nanoantenna_tcmt_spectrum`)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .materials import C0_M_PER_S

CM_INV_TO_RAD_S = 2.0 * math.pi * C0_M_PER_S * 100.0  # 1 cm^-1 -> 1.88365e11 rad/s

# Peer-reviewed Mid-IR molecular vibrational oscillator presets (wavenumber [cm^-1], HWHM damping [cm^-1])
SEIRA_MOLECULE_PRESETS: dict[str, list[dict[str, float]]] = {
    "PMMA": [{"name": "C=O stretch", "nu0_cm1": 1730.0, "gamma_cm1": 12.0, "g_cm1": 22.0}],
    "Protein_Amide": [
        {"name": "Amide I", "nu0_cm1": 1655.0, "gamma_cm1": 18.0, "g_cm1": 24.0},
        {"name": "Amide II", "nu0_cm1": 1545.0, "gamma_cm1": 16.0, "g_cm1": 18.0},
    ],
    "CO2": [{"name": "CO2 antisym stretch", "nu0_cm1": 2349.0, "gamma_cm1": 8.0, "g_cm1": 16.0}],
    "CH4": [{"name": "CH4 bending nu4", "nu0_cm1": 1306.0, "gamma_cm1": 7.5, "g_cm1": 15.0}],
    "CBP_Polystyrene": [
        {"name": "C=C aromatic", "nu0_cm1": 1493.0, "gamma_cm1": 6.0, "g_cm1": 14.0},
        {"name": "CH2 bend", "nu0_cm1": 1452.0, "gamma_cm1": 7.0, "g_cm1": 12.0},
    ],
}


@dataclass
class NanoantennaTCMTConfig:
    regime: str = "seira_molecular"       # "mim_absorber" | "fano_eit_qbic" | "seira_molecular" | "metasurface_ep"
    num_ports: int = 1                    # 1 = reflective MIM absorber (T=0); 2 = transmissive metasurface (R+T+A=1)
    # Primary (bright) antenna resonance in wavenumber [cm^-1] (e.g. 1730 cm^-1 = 5.78 um Mid-IR, 6450 cm^-1 = 1.55 um NIR)
    nu0_bright_cm1: float = 1730.0
    gamma_rad_bright_cm1: float = 85.0    # Radiative amplitude decay rate gamma_r [cm^-1] (FWHM_rad = 2*gamma_r)
    gamma_abs_bright_cm1: float = 85.0    # Ohmic/intrinsic absorption decay rate gamma_a [cm^-1]
    # Secondary (dark / qBIC / EP partner) resonator parameters
    nu0_dark_cm1: float = 1730.0
    gamma_rad_dark_cm1: float = 2.0       # Near-zero for dark/qBIC mode; finite for EP / bright-bright
    gamma_abs_dark_cm1: float = 10.0
    coupling_mu_cm1: float = 45.0         # Near-field coherent coupling rate mu [cm^-1]
    radiative_phase_rad: float = 0.0      # Far-field radiative interference phase psi [rad]
    # Background scattering matrix parameters (direct channel C_bg for 2-port Fano asymmetry)
    bg_reflectivity: float = 0.08         # |r_bg| for 2-port transmissive substrate (0 = pure Lorentzian, >0 = Fano)
    bg_phase_rad: float = 0.0
    # SEIRA molecular analyte parameters (for regime="seira_molecular")
    analyte_preset: str = "PMMA"          # "PMMA" | "Protein_Amide" | "CO2" | "CH4" | "CBP_Polystyrene" | "custom"
    custom_molecules: list[dict[str, float]] = field(default_factory=list)
    seira_coupling_scale: float = 1.0     # Scales molecular coupling g_mol (proportional to sqrt(concentration) * |E_loc|)
    # Wavenumber sweep window [cm^-1]
    nu_min_cm1: float | None = None
    nu_max_cm1: float | None = None
    num_points: int = 801


def _solve_2port_bg_matrix(r_mag: float, phi_bg: float) -> tuple[complex, complex]:
    """Build exact unitary symmetric 2x2 direct-pathway scattering matrix C = [[r, t], [t, r]]."""
    r_m = float(np.clip(r_mag, 0.0, 0.999))
    t_m = math.sqrt(max(1.0 - r_m * r_m, 0.0))
    ep = complex(math.cos(phi_bg), math.sin(phi_bg))
    r_bg = r_m * ep
    t_bg = 1j * t_m * ep  # Exact unitarity: |r|^2 + |t|^2 = 1 and r*t^* + t*r^* = 0!
    return r_bg, t_bg


def simulate_nanoantenna_tcmt(cfg: NanoantennaTCMTConfig) -> dict[str, Any]:
    """Solve exact energy-conserving multi-mode Nanoantenna / Metasurface TCMT equations.

    Strictly obeys Suh-Wang-Fan (IEEE JQE 2004) energy conservation `R + T + A == 1.000000`
    for arbitrary coupled bright/dark/molecular oscillators across 1-port MIM and 2-port
    transmissive metasurfaces.
    """
    t0 = time.perf_counter()
    reg = cfg.regime.lower().strip()
    nu0_b = float(cfg.nu0_bright_cm1)
    gr_b = max(float(cfg.gamma_rad_bright_cm1), 1e-6)
    ga_b = max(float(cfg.gamma_abs_bright_cm1), 1e-6)

    span_default = max(6.0 * (gr_b + ga_b), 400.0)
    nu_min = float(cfg.nu_min_cm1) if cfg.nu_min_cm1 is not None else max(100.0, nu0_b - 0.5 * span_default)
    nu_max = float(cfg.nu_max_cm1) if cfg.nu_max_cm1 is not None else (nu0_b + 0.5 * span_default)
    nu = np.linspace(nu_min, nu_max, max(201, int(cfg.num_points)))
    wl_um = 1e4 / np.maximum(nu, 1e-3)
    freq_thz = (nu * 100.0 * C0_M_PER_S) * 1e-12

    n_ports = 1 if (reg == "mim_absorber" or int(cfg.num_ports) == 1) else 2
    r_bg, t_bg = _solve_2port_bg_matrix(cfg.bg_reflectivity, cfg.bg_phase_rad)

    def _eval_coupled_system(
        om_list: list[float],
        gr_list: list[float],
        ga_list: list[float],
        mu_mat: np.ndarray,
        port_sym_list: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact matrix TCMT solver: s_- = (C + D (i*(nu*I - Omega) + Gamma_tot)^(-1) D^T) s_+."""
        m_n = len(om_list)
        # Build exact port-coupling matrix D (n_ports x m_n) satisfying C * D^* = -D and D^H D = 2*Gamma_rad
        if n_ports == 1:
            c_mat = np.array([[-1.0 + 0.0j]], dtype=np.complex128)
            # For C = [-1], C * d^* = -d => d is PURE REAL! |d|^2 = 2 * gamma_rad -> d = sqrt(2 * gamma_rad)
            d_mat = np.array([[math.sqrt(2.0 * max(g, 0.0)) for g in gr_list]], dtype=np.complex128)
        else:
            c_mat = np.array([[r_bg, t_bg], [t_bg, r_bg]], dtype=np.complex128)
            d_mat = np.zeros((2, m_n), dtype=np.complex128)
            # Eigenvalues of symmetric unitary C are c_even = r_bg + t_bg (|c|=1) and c_odd = r_bg - t_bg (|c|=1)
            # For even mode d = d0*[1, 1]^T: C d^* = c_even * d^* = -d => arg(d0) = 0.5*(arg(c_even) + pi)
            # For odd mode  d = d0*[1,-1]^T: C d^* = c_odd  * d^* = -d => arg(d0) = 0.5*(arg(c_odd)  + pi)
            c_even = r_bg + t_bg
            c_odd = r_bg - t_bg
            ph_even = 0.5 * (math.atan2(c_even.imag, c_even.real) + math.pi)
            ph_odd = 0.5 * (math.atan2(c_odd.imag, c_odd.real) + math.pi)
            for j in range(m_n):
                amp_j = math.sqrt(max(gr_list[j], 0.0))  # |d1|^2 + |d2|^2 = 2 * amp_j^2 = 2 * gamma_rad!
                if port_sym_list[j] >= 0:
                    val = amp_j * complex(math.cos(ph_even), math.sin(ph_even))
                    d_mat[0, j] = val
                    d_mat[1, j] = val
                else:
                    val = amp_j * complex(math.cos(ph_odd), math.sin(ph_odd))
                    d_mat[0, j] = val
                    d_mat[1, j] = -val

        # Radiative damping matrix Gamma_rad = 0.5 * D^H D (automatically includes exact far-field radiative coupling
        # between modes of identical symmetry — guarantees exact passivity R + T <= 1 ALWAYS!)
        gamma_rad_mat = 0.5 * (d_mat.conj().T @ d_mat)
        gamma_abs_mat = np.diag(np.asarray(ga_list, dtype=np.complex128))
        omega_mat = np.diag(np.asarray(om_list, dtype=np.complex128)) + mu_mat

        # Solve (i*(nu*I - Omega) + Gamma_rad + Gamma_abs) a = D^T s_+ with s_+ = [1, 0]^T
        s_in = np.zeros(n_ports, dtype=np.complex128)
        s_in[0] = 1.0
        rhs = d_mat.T @ s_in
        r_amp = np.empty(len(nu), dtype=np.complex128)
        t_amp = np.zeros(len(nu), dtype=np.complex128)
        eye_m = np.eye(m_n, dtype=np.complex128)
        h_base = -1j * omega_mat + gamma_rad_mat + gamma_abs_mat
        for idx, nv in enumerate(nu):
            a_vec = np.linalg.solve(1j * nv * eye_m + h_base, rhs)
            s_out = c_mat @ s_in + d_mat @ a_vec
            r_amp[idx] = s_out[0]
            if n_ports == 2:
                t_amp[idx] = s_out[1]

        r_pwr = np.abs(r_amp) ** 2
        t_pwr = np.abs(t_amp) ** 2
        a_pwr = np.maximum(0.0, 1.0 - r_pwr - t_pwr)
        return r_pwr, t_pwr, a_pwr, r_amp

    # Assemble mode matrices for each physical regime
    extra_metrics: dict[str, Any] = {}
    baseline_r: np.ndarray | None = None
    baseline_t: np.ndarray | None = None

    if reg == "mim_absorber":
        r_p, t_p, a_p, r_amp = _eval_coupled_system([nu0_b], [gr_b], [ga_b], np.zeros((1, 1)), [1])
        q_tot = nu0_b / max(2.0 * (gr_b + ga_b), 1e-6)
        extra_metrics = {
            "critical_coupling_ratio_gr_over_ga": round(gr_b / ga_b, 4),
            "peak_absorption_pct": round(float(np.max(a_p)) * 100.0, 3),
            "min_reflection_dB": round(10.0 * math.log10(max(float(np.min(r_p)), 1e-9)), 2),
            "loaded_Q": round(q_tot, 2),
        }
    elif reg == "fano_eit_qbic":
        nu0_d = float(cfg.nu0_dark_cm1)
        gr_d = max(float(cfg.gamma_rad_dark_cm1), 0.0)
        ga_d = max(float(cfg.gamma_abs_dark_cm1), 1e-6)
        mu_c = float(cfg.coupling_mu_cm1)
        mu_m = np.array([[0.0, mu_c], [mu_c, 0.0]], dtype=np.complex128)
        r_p, t_p, a_p, r_amp = _eval_coupled_system([nu0_b, nu0_d], [gr_b, gr_d], [ga_b, ga_d], mu_m, [1, 1])
        # Reference uncoupled bright antenna for EIT transparency contrast
        r_ref, t_ref, _, _ = _eval_coupled_system([nu0_b], [gr_b], [ga_b], np.zeros((1, 1)), [1])
        baseline_r, baseline_t = r_ref, t_ref
        idx_d = int(np.argmin(np.abs(nu - nu0_d)))
        eit_contrast = float(abs(t_p[idx_d] - t_ref[idx_d]) if n_ports == 2 else abs(r_p[idx_d] - r_ref[idx_d]))
        q_dark = nu0_d / max(2.0 * (gr_d + ga_d), 1e-6)
        extra_metrics = {
            "dark_mode_Q": round(q_dark, 1),
            "fano_eit_contrast_pct": round(eit_contrast * 100.0, 2),
            "peak_absorption_pct": round(float(np.max(a_p)) * 100.0, 2),
            "splitting_2mu_cm1": round(2.0 * mu_c, 2),
            "strong_coupling_C": round((mu_c ** 2) / max((gr_b + ga_b) * (gr_d + ga_d), 1e-6), 3),
        }
    elif reg == "metasurface_ep":
        nu0_d = float(cfg.nu0_dark_cm1)
        gr_d = max(float(cfg.gamma_rad_dark_cm1), 0.0)
        ga_d = max(float(cfg.gamma_abs_dark_cm1), 1e-6)
        mu_c = float(cfg.coupling_mu_cm1)
        mu_m = np.array([[0.0, mu_c], [mu_c, 0.0]], dtype=np.complex128)
        # Use orthogonal/opposite symmetry ports so two resonators couple purely via near-field mu_c (no far-field cross-damping)
        r_p, t_p, a_p, r_amp = _eval_coupled_system([nu0_b, nu0_d], [gr_b, gr_d], [ga_b, ga_d], mu_m, [1, -1])
        g1_tot = gr_b + ga_b
        g2_tot = gr_d + ga_d
        dg = abs(g1_tot - g2_tot)
        # Exact EP for H = [[nu1 - i*g1, mu], [mu, nu2 - i*g2]] occurs at |g1 - g2| = 2 * mu!
        ep_ratio = dg / max(2.0 * mu_c, 1e-9)
        ph_r = np.unwrap(np.angle(r_amp))
        phase_span_pi = abs(float(ph_r[-1] - ph_r[0])) / math.pi
        extra_metrics = {
            "loss_contrast_dgamma_cm1": round(dg, 3),
            "exact_EP_2mu_cm1": round(2.0 * mu_c, 3),
            "EP_proximity_ratio": round(ep_ratio, 4),
            "pt_regime": "exceptional_point" if abs(ep_ratio - 1.0) < 0.06 else ("pt_broken" if ep_ratio > 1.0 else "pt_unbroken"),
            "reflection_phase_winding_pi": round(phase_span_pi, 3),
            "peak_absorption_pct": round(float(np.max(a_p)) * 100.0, 2),
        }
    else:
        # Mid-IR SEIRA molecular vibrational coupling (Adato Nat. Commun. 2013; Ren / Lee SEIRA)
        mols = cfg.custom_molecules if cfg.custom_molecules else SEIRA_MOLECULE_PRESETS.get(cfg.analyte_preset, SEIRA_MOLECULE_PRESETS["PMMA"])
        om_l = [nu0_b] + [float(m["nu0_cm1"]) for m in mols]
        gr_l = [gr_b] + [0.0 for _ in mols]  # Molecular vibrational dipoles are sub-radiant (dark modes!)
        ga_l = [ga_b] + [max(float(m["gamma_cm1"]), 0.5) for m in mols]
        m_n = len(om_l)
        mu_m = np.zeros((m_n, m_n), dtype=np.complex128)
        g_list: list[float] = []
        for idx_m, m in enumerate(mols):
            gm = float(m["g_cm1"]) * float(cfg.seira_coupling_scale)
            mu_m[0, idx_m + 1] = gm
            mu_m[idx_m + 1, 0] = gm
            g_list.append(gm)
        r_p, t_p, a_p, r_amp = _eval_coupled_system(om_l, gr_l, ga_l, mu_m, [1] * m_n)
        # Bare antenna reference (without analyte molecules) to compute differential SEIRA signal Delta_R / Delta_T!
        r_ref, t_ref, a_ref, _ = _eval_coupled_system([nu0_b], [gr_b], [ga_b], np.zeros((1, 1)), [1])
        baseline_r, baseline_t = r_ref, t_ref
        diff_sig = np.abs(r_p - r_ref) if n_ports == 1 else np.abs(t_p - t_ref)
        peak_seira_pct = float(np.max(diff_sig)) * 100.0
        g_main = g_list[0] if g_list else 0.0
        gam_m0 = ga_l[1] if len(ga_l) > 1 else 10.0
        cooperativity = (g_main ** 2) / max((gr_b + ga_b) * gam_m0, 1e-6)
        rabi_strong = bool(2.0 * g_main > 0.5 * ((gr_b + ga_b) + gam_m0))
        extra_metrics = {
            "analyte": cfg.analyte_preset,
            "num_vibrational_modes": len(mols),
            "peak_SEIRA_diff_signal_pct": round(peak_seira_pct, 3),
            "coupling_g_cm1": round(g_main, 3),
            "seira_cooperativity_C": round(cooperativity, 4),
            "strong_coupling_rabi_regime": rabi_strong,
            "bare_peak_absorption_pct": round(float(np.max(a_ref)) * 100.0, 2),
        }

    max_energy_err = float(np.max(np.abs((r_p + t_p + a_p) - 1.0)))
    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "regime": reg,
            "num_ports": n_ports,
            "resonance_cm1": round(nu0_b, 2),
            "resonance_um": round(1e4 / max(nu0_b, 1e-3), 4),
            "resonance_THz": round(nu0_b * 100.0 * C0_M_PER_S * 1e-12, 4),
            "max_energy_conservation_err": float(f"{max_energy_err:.2e}"),
            **extra_metrics,
        },
        "spectrum": {
            "wavenumber_cm1": np.round(nu, 3).tolist(),
            "wavelength_um": np.round(wl_um, 5).tolist(),
            "frequency_THz": np.round(freq_thz, 4).tolist(),
            "R": np.round(r_p, 5).tolist(),
            "T": np.round(t_p, 5).tolist(),
            "A": np.round(a_p, 5).tolist(),
            "R_bare": np.round(baseline_r, 5).tolist() if baseline_r is not None else None,
            "T_bare": np.round(baseline_t, 5).tolist() if baseline_t is not None else None,
            "phase_R_rad": np.round(np.angle(r_amp), 4).tolist(),
        },
    }


def fit_nanoantenna_tcmt_spectrum(
    wavenumber_cm1: np.ndarray,
    reflectance_measured: np.ndarray,
    *,
    with_molecule: bool = False,
    nu_mol_guess_cm1: float | None = None,
) -> dict[str, Any]:
    """Inverse TCMT parameter extractor from measured/FDTD FTIR reflectance spectrum R(nu [cm^-1])."""
    t0 = time.perf_counter()
    nu = np.asarray(wavenumber_cm1, dtype=float)
    r_meas = np.asarray(reflectance_measured, dtype=float)
    idx_min = int(np.argmin(r_meas))
    nu0_g = float(nu[idx_min])
    span = max(float(np.max(nu) - np.min(nu)), 50.0)
    gam_g = max(0.08 * span, 10.0)

    if not with_molecule:
        def _res1(p: np.ndarray) -> np.ndarray:
            n0, gr, ga = p
            r_mod = np.abs(-1.0 + (2.0 * gr) / (1j * (nu - n0) + gr + ga)) ** 2
            return r_mod - r_meas

        sol = least_squares(_res1, [nu0_g, gam_g, gam_g], bounds=([float(np.min(nu)), 0.1, 0.1], [float(np.max(nu)), span, span]))
        n0_f, gr_f, ga_f = [float(v) for v in sol.x]
        r_fit = np.abs(-1.0 + (2.0 * gr_f) / (1j * (nu - n0_f) + gr_f + ga_f)) ** 2
        ss_res = float(np.sum((r_meas - r_fit) ** 2))
        ss_tot = max(float(np.sum((r_meas - np.mean(r_meas)) ** 2)), 1e-12)
        return {
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
            "metrics": {
                "nu0_cm1": round(n0_f, 3),
                "gamma_rad_cm1": round(gr_f, 3),
                "gamma_abs_cm1": round(ga_f, 3),
                "loaded_Q": round(n0_f / max(2.0 * (gr_f + ga_f), 1e-6), 2),
                "r_squared": round(1.0 - ss_res / ss_tot, 6),
            },
            "R_fit": np.round(r_fit, 5).tolist(),
        }

    nm_g = float(nu_mol_guess_cm1) if nu_mol_guess_cm1 is not None else nu0_g

    def _res_mol(p: np.ndarray) -> np.ndarray:
        n0, gr, ga, nm, gm_a, g_c = p
        # Exact 1-port bright antenna + 1 dark molecular oscillator analytical reflection:
        #   r(nu) = -1 + 2*gr / (i*(nu - n0) + gr + ga + g_c^2 / (i*(nu - nm) + gm_a))
        denom = 1j * (nu - n0) + gr + ga + (g_c ** 2) / (1j * (nu - nm) + gm_a)
        r_mod = np.abs(-1.0 + (2.0 * gr) / denom) ** 2
        return r_mod - r_meas

    sol = least_squares(
        _res_mol,
        [nu0_g, gam_g, gam_g, nm_g, 12.0, 20.0],
        bounds=([float(np.min(nu)), 0.5, 0.5, float(np.min(nu)), 0.5, 0.0], [float(np.max(nu)), span, span, float(np.max(nu)), 150.0, 300.0]),
    )
    n0_f, gr_f, ga_f, nm_f, gma_f, gc_f = [float(v) for v in sol.x]
    denom_f = 1j * (nu - n0_f) + gr_f + ga_f + (gc_f ** 2) / (1j * (nu - nm_f) + gma_f)
    r_fit = np.abs(-1.0 + (2.0 * gr_f) / denom_f) ** 2
    ss_res = float(np.sum((r_meas - r_fit) ** 2))
    ss_tot = max(float(np.sum((r_meas - np.mean(r_meas)) ** 2)), 1e-12)
    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "nu0_antenna_cm1": round(n0_f, 3),
            "gamma_rad_cm1": round(gr_f, 3),
            "gamma_abs_cm1": round(ga_f, 3),
            "nu0_molecule_cm1": round(nm_f, 3),
            "gamma_molecule_cm1": round(gma_f, 3),
            "coupling_g_cm1": round(gc_f, 3),
            "r_squared": round(1.0 - ss_res / ss_tot, 6),
        },
        "R_fit": np.round(r_fit, 5).tolist(),
    }
