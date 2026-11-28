"""First-Principles Integrated Nonlinear Photonics & Lugiato-Lefever Equation Engine (`microring.nonlinear_lle`).

100% self-contained pure-Python nonlinear photonic integrated circuit solver
(zero commercial solver licenses, zero empirical fitting constants) built directly
on MicroRing Studio's 2D vector Maxwell FEM eigenmode and spatial CMT engines:

  1. **Broadband Integrated Dispersion & Vector Nonlinearity (`compute_resonator_dispersion_and_nl`)**:
     - Broadband 2D curved Maxwell FEM sweep (`element_order=2`) extracting exact
       resonant mode frequencies `omega_mu`, `D1 (FSR)`, `D2`, `D3`, `D4`, `beta2`,
       zero-dispersion wavelengths (ZDW), and `D_int(mu)` with optional AMX splitting.
     - Full-vector Afshar-Monro (2009) nonlinear effective area `A_eff,NL`,
       waveguide Kerr parameter `gamma_NL [W^-1 m^-1]`, single-photon Kerr shift
       `g0 [Hz]`, OPO parametric threshold `P_th [mW]`, and broadband coupler
       dispersion `kappa_ext(mu)`.
  2. **Generalized Split-Step Fourier LLE Solver (`simulate_lle_microcomb`)**:
     - Solves the dimensional Lugiato-Lefever equation with full `D_int(mu)`,
       broadband `kappa_ext(mu)`, causal Raman self-frequency shift `h_R(t)`,
       coupled photothermal detuning `delta_th(t)`, and Silicon/a-Si TPA + FCD/FCA.
     - Supports `soliton_sweep` (MI -> chaos -> soliton step), `single_soliton`
       (exact analytical DKS relaxation with Cherenkov dispersive waves), and
       `dark_pulse` (normal-dispersion platicon). Automatically dispatches to
       PyTorch CUDA FP64 GPU when available, else vectorized NumPy FP64 CPU.
  3. **Cross-Band Four-Wave Mixing Bragg Scattering (`simulate_fwm_bragg_scattering`)**:
     - 4-wave 2D curved FEM eigenmodes + full-vector third-order tensor overlap
       `gamma_BS`, cavity phase mismatch, cooperativity `C_BS`, quantum frequency
       conversion efficiency `eta_CE`, and spurious sideband suppression ratio.
  4. **Chi(2) Second-Harmonic Generation (`simulate_chi2_shg_resonator`)**:
     - Fundamental + second-harmonic 2D curved FEM modes, crystal-tensor `d_ijk`
       vector overlap, QPM poling period / azimuthal order `M_QPM`, single-photon
       `g2`, undepleted `eta_SHG [%/W]`, and exact pump-depleted SHG transfer curve.
  5. **High-Power Photothermal + Kerr Bistability (`simulate_thermal_kerr_bistability`)**:
     - Exact analytical multi-valued S-curve (stable upper/lower + unstable branch)
       and dynamic blue-to-red / red-to-blue shark-fin laser scan trajectories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .eigenmode_solver import (
    MU0_H_PER_M,
    EigenmodeResult,
    WaveguideCrossSectionSpec,
    solve_modes_fem,
)
from .materials import (
    C0_M_PER_S,
    EPS0_F_PER_M,
    HBAR_J_S,
    MATERIAL_CATALOG,
    T_REF_K,
    Z0_OHM,
    get_material_fn,
)
from .mrr_topologies import (
    MRRDesignConfig,
    _lateral_perturbation_kd_vs_gap,
    _point_coupler_kappa_power_from_gap_curve,
    _solve_single_mode,
    simulate_mrr_topology,
)

try:
    import torch as _torch
    _HAS_TORCH_CUDA = bool(_torch.cuda.is_available())
except Exception:  # pragma: no cover
    _torch = None
    _HAS_TORCH_CUDA = False


# ==============================================================================
# 1. Peer-Reviewed Material Nonlinear Property Catalog (TPA, Raman, Chi2)
# ==============================================================================
# Every entry cites its primary peer-reviewed literature source:
#   - beta_tpa_m_per_w  : Two-photon absorption coefficient at 1550 nm [m/W]
#   - tpa_cutoff_um     : Two-photon absorption bandgap edge [um] (0 above edge)
#   - tau_fc_ns         : Effective free-carrier recombination lifetime [ns]
#   - sigma_fca_1550_m2 : Free-carrier absorption cross-section at 1550 nm [m^2]
#   - f_raman           : Fractional delayed Raman contribution (0..1)
#   - raman_freq_thz    : Dominant optical phonon Raman shift Omega_R / (2*pi) [THz]
#   - raman_fwhm_thz    : Raman linewidth Gamma_R / (2*pi) [THz]
#   - d33_pm_per_v      : Chi(2) nonlinear tensor d_33 [pm/V]
#   - d31_pm_per_v      : Chi(2) nonlinear tensor d_31 [pm/V]
#   - d36_pm_per_v      : Chi(2) nonlinear tensor d_36 (zincblende) [pm/V]
NONLINEAR_MATERIAL_PROPS: dict[str, dict[str, Any]] = {
    "Si": {
        "beta_tpa_m_per_w": 5.0e-12,
        "tpa_cutoff_um": 2.20,
        "tau_fc_ns": 0.50,
        "sigma_fca_1550_m2": 1.45e-21,
        "f_raman": 0.043,
        "raman_freq_thz": 15.60,
        "raman_fwhm_thz": 0.105,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Bristow APL 2007; Lin/Painter/Agrawal OE 2007; Griffith Nat. Commun. 2015",
    },
    "a-Si": {
        "beta_tpa_m_per_w": 2.5e-12,
        "tpa_cutoff_um": 1.65,
        "tau_fc_ns": 0.20,
        "sigma_fca_1550_m2": 1.45e-21,
        "f_raman": 0.04,
        "raman_freq_thz": 14.4,
        "raman_fwhm_thz": 1.80,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Kuyken OE 2011; Grillet OE 2012",
    },
    "Ge": {
        "beta_tpa_m_per_w": 8.0e-11,
        "tpa_cutoff_um": 3.20,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 3.0e-21,
        "f_raman": 0.03,
        "raman_freq_thz": 9.0,
        "raman_fwhm_thz": 0.12,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Zhang/Soref OE 2014 (TPA-free above 3.2 um)",
    },
    "Si3N4": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.48,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.05,
        "raman_freq_thz": 13.5,
        "raman_fwhm_thz": 12.0,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Ikeda OE 2008; Karpov PRL 2016; Brasch Science 2016",
    },
    "SiRich_SiNx": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.90,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.05,
        "raman_freq_thz": 14.0,
        "raman_fwhm_thz": 10.0,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Wang Laser Photon. Rev. 2015 (USRN)",
    },
    "4H-SiC": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.76,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.10,
        "raman_freq_thz": 23.3,
        "raman_fwhm_thz": 0.18,
        "d33_pm_per_v": 11.7,
        "d31_pm_per_v": 6.5,
        "d36_pm_per_v": 0.0,
        "source": "Guidry Optica 2020; Lukin Nat. Photon. 2020; Wang LSA 2022",
    },
    "3C-SiC": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 1.05,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.08,
        "raman_freq_thz": 23.9,
        "raman_fwhm_thz": 0.25,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 17.0,
        "source": "Lu OE 2014; Fan Opt. Lett. 2019",
    },
    "LN_Xcut": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.65,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.16,
        "raman_freq_thz": 18.8,
        "raman_fwhm_thz": 0.72,
        "d33_pm_per_v": 27.0,
        "d31_pm_per_v": 4.3,
        "d36_pm_per_v": 0.0,
        "source": "Shoji JOSA B 1997; He Optica 2019; Lu Optica 2019",
    },
    "LN_Zcut": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.65,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.16,
        "raman_freq_thz": 18.8,
        "raman_fwhm_thz": 0.72,
        "d33_pm_per_v": 27.0,
        "d31_pm_per_v": 4.3,
        "d36_pm_per_v": 0.0,
        "source": "Shoji JOSA B 1997; Lu Optica 2019",
    },
    "LT_Xcut": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.55,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.12,
        "raman_freq_thz": 17.9,
        "raman_fwhm_thz": 0.85,
        "d33_pm_per_v": 13.8,
        "d31_pm_per_v": 0.85,
        "d36_pm_per_v": 0.0,
        "source": "Wang Nature 2024 (LiTaO3); Shoji JOSA B 1997",
    },
    "LT_Zcut": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.55,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.12,
        "raman_freq_thz": 17.9,
        "raman_fwhm_thz": 0.85,
        "d33_pm_per_v": 13.8,
        "d31_pm_per_v": 0.85,
        "d36_pm_per_v": 0.0,
        "source": "Shoji JOSA B 1997",
    },
    "AlN": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.40,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.08,
        "raman_freq_thz": 19.7,
        "raman_fwhm_thz": 0.20,
        "d33_pm_per_v": 4.7,
        "d31_pm_per_v": 0.4,
        "d36_pm_per_v": 0.0,
        "source": "Bruch APL 2018; Jung Opt. Lett. 2013",
    },
    "ScAlN_10": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.46,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.08,
        "raman_freq_thz": 19.1,
        "raman_fwhm_thz": 0.45,
        "d33_pm_per_v": 6.2,
        "d31_pm_per_v": 0.6,
        "d36_pm_per_v": 0.0,
        "source": "Yoshioka Opt. Mater. Express 2022",
    },
    "ScAlN_20": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.52,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.08,
        "raman_freq_thz": 18.4,
        "raman_fwhm_thz": 0.65,
        "d33_pm_per_v": 9.5,
        "d31_pm_per_v": 1.0,
        "d36_pm_per_v": 0.0,
        "source": "Yoshioka Opt. Mater. Express 2022; Friedman Adv. Opt. Mater. 2024",
    },
    "ScAlN_30": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.58,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.08,
        "raman_freq_thz": 17.6,
        "raman_fwhm_thz": 0.90,
        "d33_pm_per_v": 13.5,
        "d31_pm_per_v": 1.5,
        "d36_pm_per_v": 0.0,
        "source": "Friedman Adv. Opt. Mater. 2024",
    },
    "GaN": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.73,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.07,
        "raman_freq_thz": 17.0,
        "raman_fwhm_thz": 0.25,
        "d33_pm_per_v": 10.6,
        "d31_pm_per_v": 5.3,
        "d36_pm_per_v": 0.0,
        "source": "Zheng Nat. Photon. 2022",
    },
    "Diamond": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.45,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.12,
        "raman_freq_thz": 39.95,
        "raman_fwhm_thz": 0.06,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Hausmann Nat. Photon. 2014",
    },
    "SiO2": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.28,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.18,
        "raman_freq_thz": 13.2,
        "raman_fwhm_thz": 9.9,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Stolen JOSA B 1989; Yi Optica 2015",
    },
    "Ta2O5": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.58,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.08,
        "raman_freq_thz": 20.0,
        "raman_fwhm_thz": 8.0,
        "d33_pm_per_v": 0.0,
        "d31_pm_per_v": 0.0,
        "d36_pm_per_v": 0.0,
        "source": "Jung Optica 2021",
    },
    "BTO": {
        "beta_tpa_m_per_w": 0.0,
        "tpa_cutoff_um": 0.78,
        "tau_fc_ns": 1.0,
        "sigma_fca_1550_m2": 0.0,
        "f_raman": 0.10,
        "raman_freq_thz": 15.5,
        "raman_fwhm_thz": 1.5,
        "d33_pm_per_v": 6.8,
        "d31_pm_per_v": 15.7,
        "d36_pm_per_v": 0.0,
        "source": "Abel Nat. Mater. 2019",
    },
}

_DEFAULT_NL_PROP: dict[str, Any] = {
    "beta_tpa_m_per_w": 0.0,
    "tpa_cutoff_um": 0.50,
    "tau_fc_ns": 1.0,
    "sigma_fca_1550_m2": 0.0,
    "f_raman": 0.05,
    "raman_freq_thz": 13.5,
    "raman_fwhm_thz": 10.0,
    "d33_pm_per_v": 0.0,
    "d31_pm_per_v": 0.0,
    "d36_pm_per_v": 0.0,
    "source": "Generic dielectric default",
}


def get_nonlinear_material_props(core_material: str) -> dict[str, Any]:
    """Return peer-reviewed TPA, Raman, and Chi(2) parameters for a catalog material."""
    if core_material in NONLINEAR_MATERIAL_PROPS:
        return dict(NONLINEAR_MATERIAL_PROPS[core_material])
    if core_material.startswith("ScAlN"):
        return dict(NONLINEAR_MATERIAL_PROPS["ScAlN_20"])
    if core_material == "Hydex":
        return dict(NONLINEAR_MATERIAL_PROPS["SiO2"])
    if core_material == "BTO_ccut":
        return dict(NONLINEAR_MATERIAL_PROPS["BTO"])
    return dict(_DEFAULT_NL_PROP)


# ==============================================================================
# 2. Configuration Dataclasses
# ==============================================================================

@dataclass
class LLEConfig:
    """Configuration for Broadband Dispersion + Generalized LLE Microcomb Solver."""
    # Resonator geometry & material
    core_material: str = "Si3N4"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    topology: str = "point_circular_ap"
    film_height_um: float = 0.80
    slab_height_um: float = 0.0
    ring_width_um: float = 1.60
    bus_width_um: float = 1.00
    sidewall_deg: float = 87.0
    ring_radius_um: float = 23.0
    straight_length_um: float = 0.0
    gap_thru_um: float = 0.45
    polarization: str = "TE0"
    wavelength_nm: float = 1550.0
    alpha_loss_db_per_cm: float | None = 0.15
    q_int_override: float | None = None
    q_ext_override: float | None = None
    # Broadband dispersion grid
    num_comb_modes: int = 256                 # Power of 2 (128, 256, 512, 1024)
    disp_wl_min_nm: float | None = None       # Defaults to 0.76 * wavelength_nm
    disp_wl_max_nm: float | None = None       # Defaults to 1.32 * wavelength_nm
    num_disp_samples: int = 11                # Broadband 2D-FEM wavelength anchor points
    include_coupler_dispersion: bool = True   # Broadband kappa_ext(mu) via multi-wl CMT
    core_mesh_res_um: float = 0.055
    # Optional avoided mode crossing (AMX) perturbation
    amx_mode_mu: int | None = None
    amx_strength_ghz: float = 0.0
    amx_detuning_modes: float = 0.25
    # LLE pump & simulation mode
    mode: str = "single_soliton"              # "single_soliton" | "soliton_sweep" | "dark_pulse"
    pump_power_mw: float = 80.0
    detuning_start_kappa: float = -2.0        # Start laser detuning (units of kappa/2)
    detuning_end_kappa: float = 12.0          # End / target detuning (units of kappa/2)
    num_steps: int = 4000                     # SSFM integration steps
    roundtrips_per_step: float | None = None  # Auto-chosen for numerical stability if None
    # Optional nonlinear perturbations
    include_raman: bool = False
    include_thermal: bool = False
    thermal_r_th_k_per_mw: float = 0.80       # Effective modal R_th [K/mW absorbed]
    thermal_tau_us: float = 15.0              # Thermal relaxation time [us]
    include_tpa_fca: bool = False             # Auto-active for Si/a-Si/Ge when True
    # Optional custom overrides
    custom_n2_m2_per_w: float | None = None
    d2_override_khz: float | None = None      # Optional override for analytic benchmark tests
    d3_override_khz: float | None = None
    prefer_gpu: bool = True
    seed: int = 42


@dataclass
class FWMBSConfig:
    """Configuration for Cross-Band Four-Wave Mixing Bragg Scattering (FWM-BS / QFC)."""
    core_material: str = "4H-SiC"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.50
    slab_height_um: float = 0.0
    ring_width_um: float = 1.15
    sidewall_deg: float = 82.0
    ring_radius_um: float = 35.0
    polarization: str = "TE0"
    # Four interacting wavelengths: omega_i = omega_s + (omega_p1 - omega_p2)
    pump1_wl_nm: float = 1550.0
    pump2_wl_nm: float = 1510.0
    signal_wl_nm: float = 750.0
    pump1_power_mw: float = 15.0
    pump2_power_mw: float = 15.0
    q_int_pump: float = 5.0e5
    q_ext_pump: float = 5.0e5
    q_int_sig: float = 3.0e5
    q_ext_sig: float = 1.5e5
    q_int_idl: float = 3.0e5
    q_ext_idl: float = 1.5e5
    core_mesh_res_um: float = 0.050
    custom_n2_m2_per_w: float | None = None


@dataclass
class SHGConfig:
    """Configuration for Chi(2) Second-Harmonic Generation (SHG) in a Microring."""
    core_material: str = "LN_Xcut"
    clad_material: str = "SiO2"
    top_clad_material: str = "Air"
    film_height_um: float = 0.60
    slab_height_um: float = 0.30
    ring_width_um: float = 1.40
    sidewall_deg: float = 65.0
    ring_radius_um: float = 50.0
    fundamental_wl_nm: float = 1550.0
    fh_polarization: str = "TE0"
    sh_polarization: str = "TE0"
    q_int_fh: float = 1.0e6
    q_ext_fh: float = 1.0e6
    q_int_sh: float = 5.0e5
    q_ext_sh: float = 5.0e5
    pump_powers_mw: tuple[float, ...] = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    quasi_phase_matched: bool = True          # True = periodic poling (PPLN/PPLT) or cyclic QPM
    core_mesh_res_um: float = 0.048
    custom_d_eff_pm_per_v: float | None = None


@dataclass
class ThermalKerrBistabilityConfig:
    """Configuration for High-Power Photothermal + Kerr Optical Bistability & Shark-Fin Scan."""
    core_material: str = "Si3N4"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.80
    slab_height_um: float = 0.0
    ring_width_um: float = 1.50
    ring_radius_um: float = 40.0
    gap_thru_um: float = 0.40
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    q_int: float = 1.0e6
    q_ext: float = 1.0e6
    input_power_mw: float = 20.0
    absorption_fraction: float = 0.25         # Fraction of intrinsic loss converted to heat (vs scattering)
    r_th_k_per_mw: float = 1.20               # Modal thermal resistance [K/mW absorbed]
    tau_th_us: float = 12.0                   # Thermal time constant [us]
    scan_span_ghz: float = 8.0
    scan_speed_ghz_per_ms: float = 250.0      # Laser piezo sweep rate [GHz/ms]
    num_scan_points: int = 800
    core_mesh_res_um: float = 0.055


# ==============================================================================
# 3. Full-Vector Kerr Effective Area & Dispersion Extraction
# ==============================================================================

def _build_ring_spec_from_params(
    *,
    core_material: str,
    clad_material: str,
    top_clad_material: str,
    w_um: float,
    h_film_um: float,
    h_slab_um: float,
    sidewall_deg: float,
    ring_radius_um: float,
    wl_ref_nm: float,
) -> tuple[WaveguideCrossSectionSpec, Any, Any, Any]:
    core_fn = get_material_fn(core_material)
    clad_fn = get_material_fn(clad_material)
    top_fn = get_material_fn(top_clad_material)
    wl_um = wl_ref_nm * 1e-3
    wl_scale = float(np.clip(wl_ref_nm / 1550.0, 0.35, 4.50))
    box_pad = max(1.8, 1.35 * wl_um)
    x_min = -max(2.0 * wl_scale, 0.5 * w_um + 1.25 * wl_um)
    if ring_radius_um > 0.0:
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
        x_min_um=x_min,
        x_max_um=x_max,
        h_box_um=box_pad,
        h_top_clad_um=box_pad,
        temperature_k=T_REF_K,
    )
    return spec, core_fn, clad_fn, top_fn


def compute_vector_kerr_gamma(
    mode: EigenmodeResult,
    *,
    w_bottom_um: float,
    h_film_um: float,
    h_slab_um: float,
    sidewall_deg: float,
    n_core: float,
    n_clad: float,
    n_top: float,
    n2_core_m2_per_w: float,
    n2_clad_m2_per_w: float = 2.6e-20,
    n2_top_m2_per_w: float = 0.0,
) -> dict[str, float]:
    """Exact full-vector Afshar-Monro (2009) Kerr waveguide nonlinearity `gamma_NL` & `A_eff,NL`.

    Integrates over the 2D FEM modal cross-section:
      gamma_NL = (2*pi / lambda_0) * int n2(x,y) * (n(x,y)/(2*Z0))^2 * (2|E|^4 + |E.E|^2)/3 dA / P_0^2
    which reduces identically to `2*pi*n2 / (lambda_0 * A)` in the plane-wave limit while
    rigorously capturing longitudinal `E_z`, slow-light `(n_g/n)^2`, and cladding nonlinearity.
    """
    x_um = mode.x_m * 1e6
    y_um = mode.y_m * 1e6
    dx = float(np.mean(np.diff(mode.x_m)))
    dy = float(np.mean(np.diff(mode.y_m)))
    da = dx * dy

    xx, yy = np.meshgrid(x_um, y_um)
    inv_tan = 0.0 if abs(sidewall_deg - 90.0) < 1e-3 else 1.0 / math.tan(math.radians(sidewall_deg))
    half_w_y = 0.5 * w_bottom_um - np.maximum(yy - h_slab_um, 0.0) * inv_tan
    in_core = (yy >= h_slab_um) & (yy <= h_film_um) & (np.abs(xx) <= np.maximum(half_w_y, 0.0))
    in_slab = (yy >= 0.0) & (yy < h_slab_um) if h_slab_um > 1e-6 else np.zeros_like(in_core)
    in_wg = in_core | in_slab
    in_box = yy < 0.0

    n_map = np.where(in_wg, n_core, np.where(in_box, n_clad, n_top))
    n2_map = np.where(in_wg, n2_core_m2_per_w, np.where(in_box, n2_clad_m2_per_w, n2_top_m2_per_w))

    ex, ey, ez = mode.Ex, mode.Ey, mode.Ez
    e_abs2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    e_dot_e = ex * ex + ey * ey + ez * ez
    kerr_tensor_mag = (2.0 * (e_abs2 ** 2) + np.abs(e_dot_e) ** 2) / 3.0

    p0 = float(0.5 * np.real(np.sum(ex * np.conj(mode.Hy) - ey * np.conj(mode.Hx))) * da)
    p0 = max(abs(p0), float(mode.power_w), 1e-30)

    k0 = 2.0 * math.pi / mode.wavelength_m
    num_tot = float(np.sum(n2_map * ((n_map / (2.0 * Z0_OHM)) ** 2) * kerr_tensor_mag) * da)
    gamma_nl = k0 * num_tot / (p0 ** 2)

    # Core-only effective nonlinear area A_eff,NL = (2*pi*n2_core) / (lambda_0 * gamma_core)
    num_core = float(np.sum(np.where(in_wg, 1.0, 0.0) * ((n_core / (2.0 * Z0_OHM)) ** 2) * kerr_tensor_mag) * da)
    a_eff_nl_m2 = (p0 ** 2) / max(num_core, 1e-60)
    a_eff_nl_um2 = a_eff_nl_m2 * 1e12

    # Fractional nonlinear interaction inside core (for TPA/Raman weighting)
    gamma_core_only = k0 * n2_core_m2_per_w * num_core / (p0 ** 2)
    nl_core_share = min(max(gamma_core_only / max(gamma_nl, 1e-30), 0.0), 1.0)

    return {
        "gamma_nl_per_w_m": float(gamma_nl),
        "a_eff_nl_um2": float(a_eff_nl_um2),
        "nl_core_share": float(nl_core_share),
    }


def compute_resonator_dispersion_and_nl(cfg: LLEConfig) -> dict[str, Any]:
    """Compute broadband 2D curved-FEM dispersion `D_int(mu)`, vector `gamma_NL`, `g0`, `P_th`, and `kappa_ext(mu)`."""
    t0 = time.perf_counter()
    wl0_nm = float(cfg.wavelength_nm)
    wl0_m = wl0_nm * 1e-9
    om0 = 2.0 * math.pi * C0_M_PER_S / wl0_m
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_um = 2.0 * math.pi * float(cfg.ring_radius_um) + (
        2.0 * float(cfg.straight_length_um) if cfg.topology.startswith("racetrack") else 0.0
    )
    l_rt_m = l_rt_um * 1e-6
    n_modes = int(cfg.num_comb_modes)

    spec, core_fn, clad_fn, top_fn = _build_ring_spec_from_params(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        ring_radius_um=cfg.ring_radius_um,
        wl_ref_nm=wl0_nm,
    )
    wl_scale = float(np.clip(wl0_nm / 1550.0, 0.35, 4.50))
    mesh_um = float(cfg.core_mesh_res_um) * wl_scale if abs(cfg.core_mesh_res_um - 0.055) < 1e-6 else float(cfg.core_mesh_res_um)
    mesh_um = float(np.clip(mesh_um, 0.022, 0.160))

    # 1. Solve pump 2D bent eigenmode with full exported (E, H) field grid for vector gamma_NL
    m0 = _solve_single_mode(spec, wl0_m, r_m, cfg.polarization, "fem", mesh_um, export_grid=True)
    neff0 = float(np.real(m0.n_eff))

    mat_c0 = core_fn(wl0_m, T_REF_K)
    mat_b0 = clad_fn(wl0_m, T_REF_K)
    mat_t0 = top_fn(wl0_m, T_REF_K)
    is_te = cfg.polarization.upper().startswith("TE")
    n_core0 = float(mat_c0.nx if is_te else mat_c0.ny)
    n2_core = float(cfg.custom_n2_m2_per_w) if cfg.custom_n2_m2_per_w is not None else float(mat_c0.n2_m2_per_w)
    if n2_core <= 0.0:
        n2_core = 2.4e-19

    vec_nl = compute_vector_kerr_gamma(
        m0,
        w_bottom_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        n_core=n_core0,
        n_clad=float(mat_b0.nx),
        n_top=float(mat_t0.nx),
        n2_core_m2_per_w=n2_core,
        n2_clad_m2_per_w=float(mat_b0.n2_m2_per_w),
        n2_top_m2_per_w=float(mat_t0.n2_m2_per_w),
    )

    # 2. Broadband 2D curved-FEM wavelength sweep for exact D1, D2, D3, D4, beta2 & D_int(mu)
    wl_min_nm = float(cfg.disp_wl_min_nm) if cfg.disp_wl_min_nm is not None else max(420.0, 0.78 * wl0_nm)
    wl_max_nm = float(cfg.disp_wl_max_nm) if cfg.disp_wl_max_nm is not None else min(6500.0, 1.30 * wl0_nm)
    n_samp = max(7, int(cfg.num_disp_samples))
    if n_samp % 2 == 0:
        n_samp += 1
    wls_nm = np.linspace(wl_min_nm, wl_max_nm, n_samp)
    # Guarantee pump wavelength wl0_nm is an exact sample anchor
    if float(np.min(np.abs(wls_nm - wl0_nm))) > 1.0:
        wls_nm = np.sort(np.append(wls_nm, wl0_nm))

    # Solve fast grid-free modes across the broadband span (cached via _solve_single_mode)
    om_list: list[float] = []
    neff_list: list[float] = []
    wl_valid_nm: list[float] = []
    for wl_i_nm in wls_nm:
        wl_i_m = float(wl_i_nm) * 1e-9
        if abs(wl_i_nm - wl0_nm) < 1e-6:
            nf_i = neff0
        else:
            try:
                mi = _solve_single_mode(spec, wl_i_m, r_m, cfg.polarization, "fem", mesh_um, export_grid=False)
                nf_i = float(np.real(mi.n_eff))
                nb_i = max(float(clad_fn(wl_i_m, T_REF_K).nx), float(top_fn(wl_i_m, T_REF_K).nx))
                if nf_i <= nb_i + 0.005:
                    continue
            except Exception:
                continue
        wl_valid_nm.append(float(wl_i_nm))
        om_list.append(2.0 * math.pi * C0_M_PER_S / wl_i_m)
        neff_list.append(nf_i)

    # Sort ascending by angular frequency omega
    order = np.argsort(om_list)
    om_arr = np.asarray(om_list, dtype=float)[order]
    neff_arr = np.asarray(neff_list, dtype=float)[order]
    wl_arr_nm = np.asarray(wl_valid_nm, dtype=float)[order]

    # Exact azimuthal mode number m(omega) = omega * L_rt * n_eff(omega) / (2*pi*c)
    m_arr = om_arr * l_rt_m * neff_arr / (2.0 * math.pi * C0_M_PER_S)
    # Invert via monotonic CubicSpline omega(m) for exact discrete cavity resonance frequencies omega_mu
    cs_om_of_m = CubicSpline(m_arr, om_arr, bc_type="not-a-knot", extrapolate=True)
    cs_neff_of_om = CubicSpline(om_arr, neff_arr, bc_type="not-a-knot")

    m0_val = float(om0 * l_rt_m * neff0 / (2.0 * math.pi * C0_M_PER_S))
    d1_rad = float(cs_om_of_m(m0_val, 1))
    d2_rad = float(cs_om_of_m(m0_val, 2))
    d3_rad = float(cs_om_of_m(m0_val, 3))
    # 4th derivative via centered finite difference on 3rd derivative
    dm_fd = 1.0
    d4_rad = float((cs_om_of_m(m0_val + dm_fd, 3) - cs_om_of_m(m0_val - dm_fd, 3)) / (2.0 * dm_fd))

    if cfg.d2_override_khz is not None:
        d2_rad = float(cfg.d2_override_khz) * 1e3 * 2.0 * math.pi
    if cfg.d3_override_khz is not None:
        d3_rad = float(cfg.d3_override_khz) * 1e3 * 2.0 * math.pi

    fsr_ghz = (d1_rad / (2.0 * math.pi)) * 1e-9
    t_rt_s = (2.0 * math.pi) / max(d1_rad, 1e3)
    ng0 = C0_M_PER_S * t_rt_s / l_rt_m
    # Exact propagation constant beta(omega) = omega * neff / c -> beta2 = d^2 beta / d omega^2
    beta_arr = om_arr * neff_arr / C0_M_PER_S
    cs_beta = CubicSpline(om_arr, beta_arr, bc_type="not-a-knot")
    beta2_s2_m = float(cs_beta(om0, 2))
    if cfg.d2_override_khz is not None:
        beta2_s2_m = -ng0 * d2_rad / (C0_M_PER_S * (d1_rad ** 2))
    beta2_ps2_km = beta2_s2_m * 1e27
    disp_ps_nm_km = -(2.0 * math.pi * C0_M_PER_S / (wl0_m ** 2)) * beta2_s2_m * 1e6

    # Broadband beta2(lambda) and Zero-Dispersion Wavelengths (ZDW)
    om_dense = np.linspace(float(om_arr[0]), float(om_arr[-1]), 300)
    wl_dense_nm = (2.0 * math.pi * C0_M_PER_S / om_dense) * 1e9
    b2_dense_ps2_km = cs_beta(om_dense, 2) * 1e27
    zdw_nm_list: list[float] = []
    for k in range(len(b2_dense_ps2_km) - 1):
        if b2_dense_ps2_km[k] * b2_dense_ps2_km[k + 1] <= 0.0 and abs(b2_dense_ps2_km[k + 1] - b2_dense_ps2_km[k]) > 1e-12:
            frac = -b2_dense_ps2_km[k] / (b2_dense_ps2_km[k + 1] - b2_dense_ps2_km[k])
            zdw_nm_list.append(round(float(wl_dense_nm[k] + frac * (wl_dense_nm[k + 1] - wl_dense_nm[k])), 1))
    zdw_nm_list.sort()

    # Exact mode index array mu in [-N/2, ..., N/2 - 1] (FFT frequency-shifted order built later)
    mu_arr = np.arange(-(n_modes // 2), n_modes // 2, dtype=float)
    if cfg.d2_override_khz is not None:
        d_int_rad = 0.5 * d2_rad * (mu_arr ** 2) + (1.0 / 6.0) * d3_rad * (mu_arr ** 3)
    else:
        # Use exact spline omega(m0 + mu) inside solved span, smoothly matched to Taylor outside
        m_q = m0_val + mu_arr
        m_lo, m_hi = float(m_arr[0]), float(m_arr[-1])
        om_mu = np.empty_like(mu_arr)
        in_span = (m_q >= m_lo) & (m_q <= m_hi)
        om_mu[in_span] = cs_om_of_m(m_q[in_span])
        # Smooth quadratic+cubic continuation beyond the FEM span edges to prevent unphysical spline blowup
        if np.any(~in_span):
            for side_mask, m_edge in ((m_q < m_lo, m_lo), (m_q > m_hi, m_hi)):
                if np.any(side_mask):
                    dm = m_q[side_mask] - m_edge
                    f0 = float(cs_om_of_m(m_edge))
                    f1 = float(cs_om_of_m(m_edge, 1))
                    f2 = float(cs_om_of_m(m_edge, 2))
                    om_mu[side_mask] = f0 + f1 * dm + 0.5 * f2 * (dm ** 2)
        d_int_rad = om_mu - (om0 + mu_arr * d1_rad)

    # Optional avoided-mode-crossing (AMX) local dispersion shift
    if cfg.amx_mode_mu is not None and abs(float(cfg.amx_strength_ghz)) > 1e-9:
        g_amx = 2.0 * math.pi * float(cfg.amx_strength_ghz) * 1e9
        dm_amx = mu_arr - float(cfg.amx_mode_mu) + float(cfg.amx_detuning_modes)
        d_int_rad = d_int_rad - (g_amx ** 2) / (d1_rad * np.where(np.abs(dm_amx) < 0.05, np.copysign(0.05, dm_amx), dm_amx))

    # Wavelength vector for each comb mode mu
    om_comb = np.maximum(om0 + mu_arr * d1_rad + d_int_rad, 0.1 * om0)
    wl_comb_nm = (2.0 * math.pi * C0_M_PER_S / om_comb) * 1e9

    # 3. Cavity Q_int, Q_ext(mu) broadband coupler dispersion, g0 & P_th
    alpha_db_cm = (
        float(cfg.alpha_loss_db_per_cm)
        if cfg.alpha_loss_db_per_cm is not None
        else float(MATERIAL_CATALOG.get(cfg.core_material, {}).get("typical_loss_db_cm", 0.5))
    )
    if cfg.q_int_override is not None:
        q_int = float(cfg.q_int_override)
    else:
        alpha_m = alpha_db_cm * 100.0 * (math.log(10.0) / 10.0)
        q_int = (2.0 * math.pi * ng0) / max(alpha_m * wl0_m, 1e-18)

    # Evaluate pump coupler Q_ext and optional broadband kappa_ext(mu)
    if cfg.q_ext_override is not None:
        q_ext0 = float(cfg.q_ext_override)
        kappa_ext_mu = np.full(n_modes, om0 / max(q_ext0, 100.0), dtype=float)
    else:
        mrr_base = simulate_mrr_topology(MRRDesignConfig(
            topology=cfg.topology,
            core_material=cfg.core_material,
            clad_material=cfg.clad_material,
            top_clad_material=cfg.top_clad_material,
            ring_width_um=cfg.ring_width_um,
            bus_width_um=cfg.bus_width_um,
            film_height_um=cfg.film_height_um,
            slab_height_um=cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            ring_radius_um=cfg.ring_radius_um,
            straight_length_um=cfg.straight_length_um,
            gap_thru_um=cfg.gap_thru_um,
            wavelength_nm=wl0_nm,
            polarization=cfg.polarization,
            alpha_loss_db_per_cm=alpha_db_cm,
            span_nm=4.0,
            num_spectrum_points=200,
            core_mesh_res_um=mesh_um,
        ))
        q_ext0 = float(mrr_base["metrics"]["Q_ext_thru"])
        k_ext0 = om0 / max(q_ext0, 100.0)
        if cfg.include_coupler_dispersion:
            # 3-anchor broadband coupler power coupling kappa_p(lambda) across comb span
            w_c_anchors = np.array([max(wl_min_nm, 0.85 * wl0_nm), wl0_nm, min(wl_max_nm, 1.18 * wl0_nm)], dtype=float)
            k_ext_anchors = []
            for w_a in w_c_anchors:
                if abs(w_a - wl0_nm) < 1.0:
                    k_ext_anchors.append(k_ext0)
                else:
                    try:
                        r_a = simulate_mrr_topology(MRRDesignConfig(
                            topology=cfg.topology,
                            core_material=cfg.core_material,
                            clad_material=cfg.clad_material,
                            top_clad_material=cfg.top_clad_material,
                            ring_width_um=cfg.ring_width_um,
                            bus_width_um=cfg.bus_width_um,
                            film_height_um=cfg.film_height_um,
                            slab_height_um=cfg.slab_height_um,
                            sidewall_deg=cfg.sidewall_deg,
                            ring_radius_um=cfg.ring_radius_um,
                            straight_length_um=cfg.straight_length_um,
                            gap_thru_um=cfg.gap_thru_um,
                            wavelength_nm=float(w_a),
                            polarization=cfg.polarization,
                            alpha_loss_db_per_cm=alpha_db_cm,
                            span_nm=4.0,
                            num_spectrum_points=100,
                            core_mesh_res_um=mesh_um,
                        ))
                        om_a = 2.0 * math.pi * C0_M_PER_S / (float(w_a) * 1e-9)
                        k_ext_anchors.append(om_a / max(float(r_a["metrics"]["Q_ext_thru"]), 100.0))
                    except Exception:
                        k_ext_anchors.append(k_ext0)
            # Log-linear interpolation captures exact exponential evanescent decay exp(-gamma(lambda)*gap)
            log_ke = np.log(np.maximum(np.asarray(k_ext_anchors, dtype=float), 1e3))
            log_ke_mu = np.interp(wl_comb_nm, w_c_anchors, log_ke)
            kappa_ext_mu = np.exp(log_ke_mu)
        else:
            kappa_ext_mu = np.full(n_modes, k_ext0, dtype=float)

    kappa_int = om0 / max(q_int, 100.0)
    kappa_ext0 = om0 / max(q_ext0, 100.0)
    kappa_tot0 = kappa_int + kappa_ext0
    q_loaded = om0 / kappa_tot0
    eta_c = kappa_ext0 / kappa_tot0
    finesse = d1_rad / kappa_tot0

    # Exact Kerr parameters: g_NL [rad/(s*W)], single-photon g0 [rad/s], OPO threshold P_th [W]
    gamma_nl = float(vec_nl["gamma_nl_per_w_m"])
    g_nl_rad_per_s_w = (gamma_nl * l_rt_m) / max(t_rt_s, 1e-18)
    g0_rad_per_s = HBAR_J_S * om0 * g_nl_rad_per_s_w / max(t_rt_s, 1e-18)
    p_th_w = (kappa_tot0 ** 3) * HBAR_J_S * om0 / max(8.0 * g0_rad_per_s * kappa_ext0, 1e-40)
    p_th_mw = p_th_w * 1e3

    # Predict Cherenkov dispersive wave (DW) positions where D_int(mu) + delta_soliton = 0
    dw_wls_nm: list[float] = []
    delta_ref = 0.5 * kappa_tot0 * max(float(cfg.detuning_end_kappa), 4.0)
    d_shifted = d_int_rad + delta_ref
    for k in range(len(mu_arr) - 1):
        if abs(mu_arr[k]) > 6 and d_shifted[k] * d_shifted[k + 1] <= 0.0:
            frac = -d_shifted[k] / max(abs(d_shifted[k + 1] - d_shifted[k]), 1e-30) * (1.0 if d_shifted[k + 1] > d_shifted[k] else -1.0)
            wl_dw = float(wl_comb_nm[k] + np.clip(frac, 0.0, 1.0) * (wl_comb_nm[k + 1] - wl_comb_nm[k]))
            if 400.0 <= wl_dw <= 8000.0:
                dw_wls_nm.append(round(wl_dw, 1))

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "wavelength_nm": wl0_nm,
            "neff_pump": round(neff0, 6),
            "ng_pump": round(ng0, 5),
            "FSR_GHz": round(fsr_ghz, 3),
            "D1_2pi_GHz": round(fsr_ghz, 3),
            "D2_2pi_kHz": round((d2_rad / (2.0 * math.pi)) * 1e-3, 3),
            "D2_2pi_MHz": round((d2_rad / (2.0 * math.pi)) * 1e-6, 5),
            "D3_2pi_kHz": round((d3_rad / (2.0 * math.pi)) * 1e-3, 4),
            "D4_2pi_Hz": round((d4_rad / (2.0 * math.pi)), 4),
            "dispersion_regime": "anomalous" if d2_rad > 0.0 else "normal",
            "beta2_ps2_per_km": round(beta2_ps2_km, 2),
            "D_ps_per_nm_km": round(disp_ps_nm_km, 2),
            "zero_dispersion_wls_nm": zdw_nm_list,
            "predicted_dw_wls_nm": dw_wls_nm,
            "n2_m2_per_w": n2_core,
            "A_eff_NL_um2": round(vec_nl["a_eff_nl_um2"], 4),
            "gamma_NL_per_W_m": round(gamma_nl, 4),
            "nl_core_share": round(vec_nl["nl_core_share"], 4),
            "g0_2pi_Hz": round(g0_rad_per_s / (2.0 * math.pi), 4),
            "Q_int": round(q_int, 0),
            "Q_ext": round(q_ext0, 0),
            "Q_loaded": round(q_loaded, 0),
            "finesse": round(finesse, 1),
            "coupling_eta": round(eta_c, 4),
            "kappa_tot_2pi_MHz": round((kappa_tot0 / (2.0 * math.pi)) * 1e-6, 3),
            "P_th_OPO_mW": round(p_th_mw, 4),
            "roundtrip_time_ps": round(t_rt_s * 1e12, 4),
            "roundtrip_length_um": round(l_rt_um, 3),
        },
        "arrays": {
            "mu": mu_arr,
            "wavelength_comb_nm": wl_comb_nm,
            "D_int_rad_per_s": d_int_rad,
            "D_int_2pi_GHz": (d_int_rad / (2.0 * math.pi)) * 1e-9,
            "kappa_ext_mu_rad_per_s": kappa_ext_mu,
            "kappa_tot_mu_rad_per_s": kappa_int + kappa_ext_mu,
            "wl_dense_nm": wl_dense_nm,
            "beta2_dense_ps2_km": b2_dense_ps2_km,
            "wl_fem_anchors_nm": wl_arr_nm,
            "neff_fem_anchors": neff_arr,
        },
        "_internal": {
            "om0": om0,
            "d1_rad": d1_rad,
            "d2_rad": d2_rad,
            "t_rt_s": t_rt_s,
            "l_rt_m": l_rt_m,
            "ng0": ng0,
            "n_core0": n_core0,
            "kappa_int": kappa_int,
            "kappa_ext0": kappa_ext0,
            "kappa_tot0": kappa_tot0,
            "g_nl_rad_per_s_w": g_nl_rad_per_s_w,
            "g0_rad_per_s": g0_rad_per_s,
            "p_th_w": p_th_w,
            "a_eff_nl_m2": vec_nl["a_eff_nl_um2"] * 1e-12,
            "dn_dt_modal": float(m0.core_confinement * mat_c0.dn_dt_per_k + (1.0 - m0.core_confinement) * mat_b0.dn_dt_per_k),
            "core_confinement": float(m0.core_confinement),
        },
    }


# ==============================================================================
# 4. Generalized Split-Step Fourier Lugiato-Lefever Equation (LLE) Solver
# ==============================================================================

def _build_causal_raman_susceptibility(
    mu_fft: np.ndarray,
    d1_rad: float,
    raman_freq_thz: float,
    raman_fwhm_thz: float,
) -> np.ndarray:
    """Exact frequency-domain causal damped-oscillator Raman susceptibility H_R(Omega_mu).

    For causal h_R(tau) = ((om_R^2 + gam_R^2)/om_R) * exp(-gam_R * tau) * sin(om_R * tau) u(tau),
    normalized so H_R(0) = 1. In the co-moving frame theta = phi - D1*t, earlier time lies at
    +theta and later time at -theta, so the retarded convolution with NumPy's e^{-i mu theta}
    FFT convention yields:
      H_R(Omega) = (om_R^2 + gam_R^2) / (om_R^2 + gam_R^2 - Omega^2 - 2i * gam_R * Omega)
    which gives the exact causal Raman self-frequency red-shift (centroid mu < 0).
    """
    om_r = 2.0 * math.pi * max(raman_freq_thz, 0.5) * 1e12
    gam_r = math.pi * max(raman_fwhm_thz, 0.05) * 1e12
    om_0_sq = om_r * om_r + gam_r * gam_r
    om_mu = mu_fft * d1_rad
    return om_0_sq / (om_0_sq - om_mu * om_mu - 2j * gam_r * om_mu)


def simulate_lle_microcomb(
    cfg: LLEConfig,
    *,
    disp_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the Generalized Lugiato-Lefever Equation (LLE) via Split-Step Fourier (SSFM).

    Automatically accelerates on NVIDIA CUDA FP64 GPU (`torch.fft`) when installed and
    enabled (`prefer_gpu=True`), with seamless bit-consistent NumPy FP64 CPU execution.
    """
    t0 = time.perf_counter()
    disp = disp_cache if disp_cache is not None else compute_resonator_dispersion_and_nl(cfg)
    dm = disp["metrics"]
    arr = disp["arrays"]
    inn = disp["_internal"]

    n_modes = int(len(arr["mu"]))
    om0 = float(inn["om0"])
    d1_rad = float(inn["d1_rad"])
    d2_rad = float(inn["d2_rad"])
    t_rt_s = float(inn["t_rt_s"])
    l_rt_m = float(inn["l_rt_m"])
    ng0 = float(inn["ng0"])
    vg = C0_M_PER_S / max(ng0, 1.0)
    kappa_int = float(inn["kappa_int"])
    kappa_ext0 = float(inn["kappa_ext0"])
    kappa_tot0 = float(inn["kappa_tot0"])
    g_nl = float(inn["g_nl_rad_per_s_w"])
    p_th_w = float(inn["p_th_w"])
    a_eff_m2 = float(inn["a_eff_nl_m2"])
    dn_dt_mod = float(inn["dn_dt_modal"])

    # Convert sorted mu = [-N/2 .. N/2-1] to standard FFT order [0, 1, ..., N/2-1, -N/2, ..., -1]
    mu_sorted = np.asarray(arr["mu"], dtype=float)
    d_int_sorted = np.asarray(arr["D_int_rad_per_s"], dtype=float)
    kappa_tot_sorted = np.asarray(arr["kappa_tot_mu_rad_per_s"], dtype=float)
    kappa_ext_sorted = np.asarray(arr["kappa_ext_mu_rad_per_s"], dtype=float)

    mu_fft = np.fft.ifftshift(mu_sorted)
    d_int_fft = np.fft.ifftshift(d_int_sorted)
    kappa_tot_fft = np.fft.ifftshift(kappa_tot_sorted)
    theta = np.linspace(-math.pi, math.pi, n_modes, endpoint=False)

    # Material nonlinear perturbations (Raman, TPA/FCA, Thermal)
    nl_props = get_nonlinear_material_props(cfg.core_material)
    f_raman = float(nl_props["f_raman"]) if cfg.include_raman else 0.0
    h_raman_fft = (
        _build_causal_raman_susceptibility(
            mu_fft, d1_rad, float(nl_props["raman_freq_thz"]), float(nl_props["raman_fwhm_thz"])
        )
        if f_raman > 1e-6
        else None
    )

    wl0_um = cfg.wavelength_nm * 1e-3
    use_tpa = bool(cfg.include_tpa_fca) and float(nl_props["beta_tpa_m_per_w"]) > 0.0 and wl0_um < float(nl_props["tpa_cutoff_um"])
    beta_tpa = float(nl_props["beta_tpa_m_per_w"]) if use_tpa else 0.0
    tau_fc_s = max(float(nl_props["tau_fc_ns"]) * 1e-9, 1e-11)
    sigma_fca = float(nl_props["sigma_fca_1550_m2"]) * ((wl0_um / 1.55) ** 2) if use_tpa else 0.0
    tpa_amp_rate = (beta_tpa * vg) / (2.0 * max(a_eff_m2, 1e-18))  # 1 / (s * W)

    p_in_w = max(float(cfg.pump_power_mw) * 1e-3, 1e-9)
    f_norm_sq = p_in_w / max(p_th_w, 1e-18)
    f_norm = math.sqrt(f_norm_sq)
    s_in = math.sqrt(kappa_ext0 * p_in_w / max(t_rt_s, 1e-18))  # W^(1/2) / s

    # Normalized detuning zeta = 2 * delta_0 / kappa_tot0
    # Maximum single-soliton existence detuning: zeta_max = pi^2 * F^2 / 8 (Herr et al. 2014)
    zeta_max_analytic = (math.pi ** 2 / 8.0) * f_norm_sq
    mode_kind = str(cfg.mode).lower().strip()

    # Clamp target detuning into soliton existence window when running `single_soliton`
    if mode_kind == "single_soliton":
        if d2_rad <= 0.0:
            # Auto-fallback notice if user passed normal-dispersion waveguide to single_soliton
            zeta_target = max(float(cfg.detuning_end_kappa), 2.0)
        else:
            zeta_cap = max(2.2, min(float(cfg.detuning_end_kappa), 0.78 * zeta_max_analytic))
            zeta_target = zeta_cap if f_norm_sq > 2.2 else max(float(cfg.detuning_end_kappa), 1.5)
        zeta_schedule = np.full(int(cfg.num_steps), zeta_target, dtype=float)
    elif mode_kind == "dark_pulse":
        zeta_target = max(1.8, min(float(cfg.detuning_end_kappa), max(2.5, 0.45 * f_norm_sq)))
        zeta_schedule = np.full(int(cfg.num_steps), zeta_target, dtype=float)
    else:
        # Full blue-to-red laser detuning sweep
        z_hi = float(cfg.detuning_end_kappa)
        if z_hi <= float(cfg.detuning_start_kappa):
            z_hi = max(float(cfg.detuning_start_kappa) + 8.0, 0.90 * zeta_max_analytic)
        zeta_schedule = np.linspace(float(cfg.detuning_start_kappa), z_hi, int(cfg.num_steps))

    # Choose slow-time step dt [s] safely inside photon lifetime & nonlinear phase stability
    n_steps = max(400, int(cfg.num_steps))
    tau_ph_s = 1.0 / max(kappa_tot0, 1e3)
    if cfg.roundtrips_per_step is not None and cfg.roundtrips_per_step > 0.0:
        dt_s = float(cfg.roundtrips_per_step) * t_rt_s
    else:
        # Total integration horizon: 24 photon lifetimes for relaxation, 45 for full sweep
        t_horizon = (45.0 if mode_kind == "soliton_sweep" else 24.0) * tau_ph_s
        dt_s = t_horizon / n_steps
    # Guard maximum nonlinear phase shift per step <= 0.12 rad
    p_peak_est = max(2.0 * (0.5 * kappa_tot0 * float(np.max(np.abs(zeta_schedule)))) / max(g_nl, 1e-18), 10.0 * p_in_w * (d1_rad / kappa_tot0))
    dt_nl_cap = 0.12 / max(g_nl * p_peak_est, 1.0)
    dt_s = min(dt_s, max(dt_nl_cap, 0.02 * tau_ph_s / max(f_norm_sq, 1.0)))

    # Initial condition A(theta, 0)
    rng = np.random.default_rng(int(cfg.seed))
    noise_amp = math.sqrt(HBAR_J_S * om0 / max(t_rt_s, 1e-18)) * 2.0
    noise_vec = noise_amp * (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

    if mode_kind == "single_soliton" and d2_rad > 0.0:
        # Exact analytical dissipative Kerr soliton (DKS) seed:
        #   A(theta) = A_cw + sqrt(2*delta/g_NL) * sech(theta / theta_s) * exp(i * phi_s)
        delta_t = 0.5 * kappa_tot0 * float(zeta_schedule[0])
        theta_s = math.sqrt(max(d2_rad / max(2.0 * delta_t, 1e-12), 1e-9))
        a_s = math.sqrt(max(2.0 * delta_t / max(g_nl, 1e-30), 0.0))
        # Exact upper-branch (stable) soliton phase lock:
        #   phi_s = -arccos(sqrt(8 * delta / (pi^2 * kappa * F^2))) in (-pi/2, 0)
        # (same lower half-plane as A_cw = S_in / (kappa/2 + i*delta); +arccos is the unstable branch).
        cos_phi = min(max(math.sqrt(max(float(zeta_schedule[0]) / max(zeta_max_analytic, 1e-9), 0.0)), 0.0), 0.995)
        phi_s = -math.acos(cos_phi)
        a_cw = s_in / complex(0.5 * kappa_tot0, delta_t)
        a_init = a_cw + a_s * (1.0 / np.cosh(np.clip(theta / max(theta_s, 1e-4), -40.0, 40.0))) * complex(math.cos(phi_s), math.sin(phi_s)) + noise_vec
    elif mode_kind == "dark_pulse":
        # Normal-dispersion dark-pulse / platicon seed (two interlocked switching waves)
        delta_t = 0.5 * kappa_tot0 * float(zeta_schedule[0])
        a_hi = math.sqrt(max(delta_t / max(g_nl, 1e-30), 1e-9))
        a_lo = s_in / complex(0.5 * kappa_tot0, delta_t)
        window = 0.5 * (np.tanh((theta + 0.55) / 0.08) - np.tanh((theta - 0.55) / 0.08))
        a_init = a_hi * (1.0 - 0.85 * window) + a_lo * (0.85 * window) + noise_vec
    else:
        # Vacuum-noise cold cavity seed for blue-to-red sweep
        a_init = noise_vec.copy()

    # Select compute backend: PyTorch CUDA FP64 GPU if available & preferred, else NumPy FP64 CPU
    use_cuda = bool(cfg.prefer_gpu) and _HAS_TORCH_CUDA and (_torch is not None)
    backend_name = "cuda_fp64" if use_cuda else "cpu_numpy_fp64"

    n_trace = min(200, n_steps)
    trace_stride = max(1, n_steps // n_trace)
    trace_zeta: list[float] = []
    trace_p_cav_mw: list[float] = []
    trace_p_peak_w: list[float] = []
    trace_delta_th_ghz: list[float] = []

    delta_temp_k = 0.0
    nc_m3 = 0.0
    tau_th_s = max(float(cfg.thermal_tau_us) * 1e-6, 1e-8)
    r_th_k_per_w = float(cfg.thermal_r_th_k_per_mw) * 1e3

    # Strang Split-Step Fourier integration (2nd-order accurate in slow time dt)
    if use_cuda and _torch is not None:
        dev = _torch.device("cuda:0")
        a_t = _torch.tensor(a_init, dtype=_torch.complex128, device=dev)
        d_int_t = _torch.tensor(d_int_fft, dtype=_torch.float64, device=dev)
        ktot_t = _torch.tensor(kappa_tot_fft, dtype=_torch.float64, device=dev)
        h_ram_t = _torch.tensor(h_raman_fft, dtype=_torch.complex128, device=dev) if h_raman_fft is not None else None
        half_dt = 0.5 * dt_s

        for step_i in range(n_steps):
            z_cur = float(zeta_schedule[step_i])
            d0_cur = 0.5 * kappa_tot0 * z_cur
            d_th_cur = -(om0 / max(ng0, 1.0)) * dn_dt_mod * delta_temp_k if cfg.include_thermal else 0.0
            # Silicon FCD blue-shifts cavity (+delta)
            d_fcd = (om0 / max(ng0, 1.0)) * (8.8e-28 * nc_m3 + 8.5e-24 * (max(nc_m3, 0.0) ** 0.8)) if use_tpa else 0.0
            l_op = -0.5 * ktot_t - 1j * (d0_cur + d_th_cur + d_fcd + d_int_t) - (0.5 * sigma_fca * vg * nc_m3)

            # Half linear step + exact exponential Euler pump term on mu=0
            a_k = _torch.fft.fft(a_t)
            exp_h = _torch.exp(l_op * half_dt)
            l0 = l_op[0]
            phi1_0 = (exp_h[0] - 1.0) / l0 if abs(complex(l0.item())) > 1e-12 else half_dt
            a_k = a_k * exp_h
            a_k[0] = a_k[0] + phi1_0 * (s_in * n_modes)
            a_t = _torch.fft.ifft(a_k)

            # Full nonlinear step (Kerr + causal Raman + TPA)
            p_loc = a_t.real * a_t.real + a_t.imag * a_t.imag
            if h_ram_t is not None:
                p_ram = _torch.fft.ifft(_torch.fft.fft(p_loc.to(_torch.complex128)) * h_ram_t).real
                p_eff = (1.0 - f_raman) * p_loc + f_raman * p_ram
            else:
                p_eff = p_loc
            a_t = a_t * _torch.exp((1j * g_nl * p_eff - tpa_amp_rate * p_loc) * dt_s)

            # Second half linear step
            a_k = _torch.fft.fft(a_t)
            a_k = a_k * exp_h
            a_k[0] = a_k[0] + phi1_0 * (s_in * n_modes)
            a_t = _torch.fft.ifft(a_k)

            # Update slow thermal & free-carrier reservoirs
            if cfg.include_thermal or use_tpa or (step_i % trace_stride == 0):
                p_mean_w = float(_torch.mean(a_t.real * a_t.real + a_t.imag * a_t.imag).item())
                if cfg.include_thermal:
                    p_abs_w = 0.25 * kappa_int * t_rt_s * p_mean_w
                    delta_temp_k += (dt_s / tau_th_s) * (r_th_k_per_w * p_abs_w - delta_temp_k)
                if use_tpa:
                    p2_mean = float(_torch.mean((a_t.real * a_t.real + a_t.imag * a_t.imag) ** 2).item())
                    gen_fc = (beta_tpa / (2.0 * HBAR_J_S * om0 * (a_eff_m2 ** 2))) * p2_mean
                    nc_m3 = max(0.0, nc_m3 + dt_s * (gen_fc - nc_m3 / tau_fc_s))
                if step_i % trace_stride == 0:
                    p_pk_w = float(_torch.max(a_t.real * a_t.real + a_t.imag * a_t.imag).item())
                    trace_zeta.append(round(z_cur, 4))
                    trace_p_cav_mw.append(round(p_mean_w * 1e3, 4))
                    trace_p_peak_w.append(round(p_pk_w, 5))
                    trace_delta_th_ghz.append(round((d_th_cur / (2.0 * math.pi)) * 1e-9, 5))

        a_final = a_t.detach().cpu().numpy()
    else:
        a_c = np.asarray(a_init, dtype=np.complex128)
        half_dt = 0.5 * dt_s
        for step_i in range(n_steps):
            z_cur = float(zeta_schedule[step_i])
            d0_cur = 0.5 * kappa_tot0 * z_cur
            d_th_cur = -(om0 / max(ng0, 1.0)) * dn_dt_mod * delta_temp_k if cfg.include_thermal else 0.0
            d_fcd = (om0 / max(ng0, 1.0)) * (8.8e-28 * nc_m3 + 8.5e-24 * (max(nc_m3, 0.0) ** 0.8)) if use_tpa else 0.0
            l_op = -0.5 * kappa_tot_fft - 1j * (d0_cur + d_th_cur + d_fcd + d_int_fft) - (0.5 * sigma_fca * vg * nc_m3)

            # Half linear + exact exponential Euler pump term
            a_k = np.fft.fft(a_c)
            exp_h = np.exp(l_op * half_dt)
            l0 = complex(l_op[0])
            phi1_0 = (exp_h[0] - 1.0) / l0 if abs(l0) > 1e-12 else half_dt
            a_k *= exp_h
            a_k[0] += phi1_0 * (s_in * n_modes)
            a_c = np.fft.ifft(a_k)

            # Full nonlinear step (Kerr + causal Raman + TPA)
            p_loc = a_c.real * a_c.real + a_c.imag * a_c.imag
            if h_raman_fft is not None:
                p_ram = np.fft.ifft(np.fft.fft(p_loc) * h_raman_fft).real
                p_eff = (1.0 - f_raman) * p_loc + f_raman * p_ram
            else:
                p_eff = p_loc
            a_c *= np.exp((1j * g_nl * p_eff - tpa_amp_rate * p_loc) * dt_s)

            # Second half linear step
            a_k = np.fft.fft(a_c)
            a_k *= exp_h
            a_k[0] += phi1_0 * (s_in * n_modes)
            a_c = np.fft.ifft(a_k)

            if cfg.include_thermal or use_tpa or (step_i % trace_stride == 0):
                p_mean_w = float(np.mean(a_c.real * a_c.real + a_c.imag * a_c.imag))
                if cfg.include_thermal:
                    p_abs_w = 0.25 * kappa_int * t_rt_s * p_mean_w
                    delta_temp_k += (dt_s / tau_th_s) * (r_th_k_per_w * p_abs_w - delta_temp_k)
                if use_tpa:
                    p2_mean = float(np.mean((a_c.real * a_c.real + a_c.imag * a_c.imag) ** 2))
                    gen_fc = (beta_tpa / (2.0 * HBAR_J_S * om0 * (a_eff_m2 ** 2))) * p2_mean
                    nc_m3 = max(0.0, nc_m3 + dt_s * (gen_fc - nc_m3 / tau_fc_s))
                if step_i % trace_stride == 0:
                    p_pk_w = float(np.max(a_c.real * a_c.real + a_c.imag * a_c.imag))
                    trace_zeta.append(round(z_cur, 4))
                    trace_p_cav_mw.append(round(p_mean_w * 1e3, 4))
                    trace_p_peak_w.append(round(p_pk_w, 5))
                    trace_delta_th_ghz.append(round((d_th_cur / (2.0 * math.pi)) * 1e-9, 5))

        a_final = a_c

    # Extract final temporal waveform, comb spectrum (coupled to bus waveguide), & soliton metrics
    p_theta_w = np.abs(a_final) ** 2
    a_mu_cav = np.fft.fftshift(np.fft.fft(a_final)) / n_modes  # intracavity modal amplitudes [W^(1/2)]
    p_mu_cav_w = np.abs(a_mu_cav) ** 2

    # Bus output comb line powers: A_out(mu) = sqrt(P_in)*delta_{mu,0} - sqrt(kappa_ext(mu)*T_R)*A_cav(mu)
    a_mu_out = -np.sqrt(np.maximum(kappa_ext_sorted * t_rt_s, 0.0)) * a_mu_cav
    idx_pump = n_modes // 2
    a_mu_out[idx_pump] += math.sqrt(p_in_w)
    p_mu_out_w = np.abs(a_mu_out) ** 2
    p_mu_out_dbm = 10.0 * np.log10(np.maximum(p_mu_out_w * 1e3, 1e-15))

    p_comb_sidebands_w = float(np.sum(p_mu_out_w) - p_mu_out_w[idx_pump])
    conv_eff_pct = (p_comb_sidebands_w / max(p_in_w, 1e-18)) * 100.0

    # Temporal pulse peak, FWHM, contrast & spectral centroid (Raman self-frequency shift!)
    p_peak_w = float(np.max(p_theta_w))
    p_bg_w = float(np.percentile(p_theta_w, 15))
    peak_to_bg = p_peak_w / max(p_bg_w, 1e-15)
    # Center pulse at theta = 0 for sub-sample interpolated FWHM measurement
    i_pk = int(np.argmax(p_theta_w))
    p_centered = np.roll(p_theta_w, n_modes // 2 - i_pk)
    half_lvl = p_bg_w + 0.5 * (p_peak_w - p_bg_w)
    above_half = np.flatnonzero(p_centered >= half_lvl)
    if len(above_half) >= 2 and peak_to_bg > 2.5:
        dth = 2.0 * math.pi / n_modes
        i_l, i_r = int(above_half[0]), int(above_half[-1])
        # Linear sub-bin interpolation on rising and falling edges removes ±1 bin quantization
        if i_l > 0 and p_centered[i_l] > p_centered[i_l - 1]:
            x_l = float(i_l - 1) + float((half_lvl - p_centered[i_l - 1]) / (p_centered[i_l] - p_centered[i_l - 1]))
        else:
            x_l = float(i_l)
        if i_r + 1 < n_modes and p_centered[i_r] > p_centered[i_r + 1]:
            x_r = float(i_r) + float((p_centered[i_r] - half_lvl) / (p_centered[i_r] - p_centered[i_r + 1]))
        else:
            x_r = float(i_r)
        fwhm_rad = max(x_r - x_l, 0.5) * dth
        tau_fwhm_fs = (fwhm_rad / (2.0 * math.pi)) * t_rt_s * 1e15
    else:
        tau_fwhm_fs = None

    # Spectral centroid excluding pump line (detects causal Raman red-shift!)
    sb_w = p_mu_cav_w.copy()
    sb_w[idx_pump] = 0.0
    sb_tot = float(np.sum(sb_w))
    if sb_tot > 1e-18:
        mu_centroid = float(np.sum(mu_sorted * sb_w) / sb_tot)
        shift_thz = (mu_centroid * d1_rad / (2.0 * math.pi)) * 1e-12
        # 3-dB spectral comb bandwidth around peak sideband envelope
        sb_db = 10.0 * np.log10(np.maximum(sb_w, 1e-30))
        pk_sb_db = float(np.max(sb_db))
        in_3db = mu_sorted[sb_db >= pk_sb_db - 3.0]
        bw_3db_thz = float((np.max(in_3db) - np.min(in_3db)) * (d1_rad / (2.0 * math.pi)) * 1e-12) if len(in_3db) >= 2 else 0.0
    else:
        mu_centroid = 0.0
        shift_thz = 0.0
        bw_3db_thz = 0.0

    # Classify final comb state
    if sb_tot < 1e-6 * float(np.sum(p_mu_cav_w)):
        comb_state = "cw_only"
    elif peak_to_bg > 5.0 and tau_fwhm_fs is not None and tau_fwhm_fs < 0.25 * t_rt_s * 1e15:
        comb_state = "dissipative_kerr_soliton"
    elif mode_kind == "dark_pulse" and float(np.min(p_theta_w)) < 0.5 * p_peak_w:
        comb_state = "dark_pulse_platicon"
    else:
        comb_state = "mi_or_turing_comb"

    # Analytical DKS benchmark values including exact coherent CW-background interference
    # (Herr et al., Nat. Photonics 8, 145, 2014 Eq. 11): |A_cw + sqrt(2*delta/g_NL) * exp(-i*arccos(sqrt(zeta/zeta_max)))|^2
    zeta_final = float(zeta_schedule[-1])
    delta_final = 0.5 * kappa_tot0 * zeta_final
    if d2_rad > 0.0 and delta_final > 0.0:
        a_s_an = math.sqrt(2.0 * delta_final / max(g_nl, 1e-30))
        if zeta_final <= zeta_max_analytic and zeta_max_analytic > 0.0:
            a_cw_an = s_in / complex(0.5 * kappa_tot0, delta_final)
            phi_an = -math.acos(min(max(math.sqrt(zeta_final / zeta_max_analytic), 0.0), 1.0))
            analytic_peak_w = float(abs(a_cw_an + a_s_an * complex(math.cos(phi_an), math.sin(phi_an))) ** 2)
        else:
            analytic_peak_w = a_s_an ** 2
        analytic_fwhm_fs = (1.7627 * math.sqrt(d2_rad / max(2.0 * delta_final, 1e-12)) / (2.0 * math.pi)) * t_rt_s * 1e15
    else:
        analytic_peak_w = None
        analytic_fwhm_fs = None

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "compute_backend": backend_name,
        "config": asdict(cfg),
        "dispersion": dm,
        "metrics": {
            "comb_state": comb_state,
            "pump_power_mw": round(cfg.pump_power_mw, 3),
            "P_th_OPO_mW": dm["P_th_OPO_mW"],
            "normalized_pump_F": round(f_norm, 3),
            "max_soliton_detuning_kappa": round(zeta_max_analytic, 3),
            "final_detuning_kappa": round(zeta_final, 3),
            "intracavity_avg_power_mW": round(float(np.mean(p_theta_w)) * 1e3, 3),
            "soliton_peak_power_W": round(p_peak_w, 4),
            "analytic_soliton_peak_W": round(analytic_peak_w, 4) if analytic_peak_w is not None else None,
            "pulse_fwhm_fs": round(tau_fwhm_fs, 2) if tau_fwhm_fs is not None else None,
            "analytic_pulse_fwhm_fs": round(analytic_fwhm_fs, 2) if analytic_fwhm_fs is not None else None,
            "peak_to_bg_ratio": round(peak_to_bg, 2),
            "comb_3db_bandwidth_THz": round(bw_3db_thz, 3),
            "comb_conversion_eff_pct": round(conv_eff_pct, 3),
            "spectral_centroid_mu": round(mu_centroid, 3),
            "raman_shift_THz": round(shift_thz, 4),
            "final_delta_T_K": round(delta_temp_k, 4),
            "final_free_carrier_cm3": float(f"{nc_m3 * 1e-6:.3e}") if use_tpa else 0.0,
        },
        "spectrum": {
            "mu": mu_sorted.astype(int).tolist(),
            "wavelength_nm": np.round(arr["wavelength_comb_nm"], 3).tolist(),
            "D_int_2pi_GHz": np.round(arr["D_int_2pi_GHz"], 4).tolist(),
            "comb_power_dBm": np.round(p_mu_out_dbm, 3).tolist(),
            "intracavity_power_dBm": np.round(10.0 * np.log10(np.maximum(p_mu_cav_w * 1e3, 1e-15)), 3).tolist(),
        },
        "waveform": {
            "theta_rad": np.round(theta, 5).tolist(),
            "time_ps": np.round((theta / (2.0 * math.pi)) * t_rt_s * 1e12, 5).tolist(),
            "power_W": np.round(p_centered, 5).tolist(),
        },
        "sweep_trace": {
            "detuning_kappa": trace_zeta,
            "intracavity_avg_power_mW": trace_p_cav_mw,
            "peak_power_W": trace_p_peak_w,
            "thermal_shift_GHz": trace_delta_th_ghz,
        },
    }


# ==============================================================================
# 5. Cross-Band Four-Wave Mixing Bragg Scattering (FWM-BS / QFC)
# ==============================================================================

def simulate_fwm_bragg_scattering(cfg: FWMBSConfig) -> dict[str, Any]:
    """First-principles 4-wave 2D curved-FEM Four-Wave Mixing Bragg Scattering (FWM-BS) solver.

    Solves exact 2D curved eigenmodes at `pump1`, `pump2`, `signal`, and target `idler`
    (`omega_i = omega_s + omega_p1 - omega_p2`) plus the unwanted parasitic sideband
    (`omega_spur = omega_s - (omega_p1 - omega_p2)`), computes the full-vector 4-wave
    Kerr overlap `gamma_BS`, cavity dispersion mismatch, XPM-compensated cooperativity `C_BS`,
    quantum frequency conversion efficiency `eta_CE`, and sideband suppression ratio.
    """
    t0 = time.perf_counter()
    om_p1 = 2.0 * math.pi * C0_M_PER_S / (cfg.pump1_wl_nm * 1e-9)
    om_p2 = 2.0 * math.pi * C0_M_PER_S / (cfg.pump2_wl_nm * 1e-9)
    om_s = 2.0 * math.pi * C0_M_PER_S / (cfg.signal_wl_nm * 1e-9)
    dom = om_p1 - om_p2
    om_i = om_s + dom
    om_spur = om_s - dom
    if om_i <= 0.0:
        raise ValueError("Invalid FWM-BS wavelengths: target idler frequency became non-positive")

    wl_i_nm = (2.0 * math.pi * C0_M_PER_S / om_i) * 1e9
    wl_spur_nm = (2.0 * math.pi * C0_M_PER_S / max(om_spur, 1e12)) * 1e9

    wl_max_ref = max(cfg.pump1_wl_nm, cfg.pump2_wl_nm, cfg.signal_wl_nm, wl_i_nm)
    spec, core_fn, clad_fn, top_fn = _build_ring_spec_from_params(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        ring_radius_um=cfg.ring_radius_um,
        wl_ref_nm=wl_max_ref,
    )
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m
    is_te = cfg.polarization.upper().startswith("TE")

    # Solve 4-wave 2D curved eigenmodes on a shared uniform evaluation grid for exact overlap
    wls_map = {"p1": cfg.pump1_wl_nm, "p2": cfg.pump2_wl_nm, "s": cfg.signal_wl_nm, "i": wl_i_nm}
    modes: dict[str, EigenmodeResult] = {}
    ngs: dict[str, float] = {}
    min_dim = min(float(cfg.ring_width_um), float(cfg.film_height_um))
    for k, w_nm in wls_map.items():
        w_m = w_nm * 1e-9
        # Quadratic (order=2) FEM converges to <2e-5 neff at h_core ~ 38-55 nm even at 750 nm
        mesh_k = float(np.clip(max(cfg.core_mesh_res_um * (w_nm / 1550.0), 0.038, 0.075 * min_dim), 0.036, 0.090))
        clad_k = max(mesh_k * 6.0, 0.24)
        mk = solve_modes_fem(
            spec, w_m, bend_radius_m=r_m, num_modes=4,
            core_mesh_res_um=mesh_k, clad_mesh_res_um=clad_k,
            eval_grid_step_um=0.030, element_order=2, export_grid=True,
        )
        cands = [m for m in mk if ((m.te_fraction >= 0.5) == is_te) and m.core_confinement >= 0.18]
        m_pick = max(cands, key=lambda m: float(np.real(m.n_eff))) if cands else mk[0]
        modes[k] = m_pick
        # Fast warm-started group index via centered FD
        dw_m = max(w_m * 4e-4, 0.4e-9)
        n_g_seed = float(np.real(m_pick.n_eff))
        mm_l = solve_modes_fem(spec, w_m - dw_m, bend_radius_m=r_m, num_modes=2, n_guess=n_g_seed, core_mesh_res_um=mesh_k, clad_mesh_res_um=clad_k, element_order=2, export_grid=False)
        mp_l = solve_modes_fem(spec, w_m + dw_m, bend_radius_m=r_m, num_modes=2, n_guess=n_g_seed, core_mesh_res_um=mesh_k, clad_mesh_res_um=clad_k, element_order=2, export_grid=False)
        m_m = min(mm_l, key=lambda m: abs(float(np.real(m.n_eff)) - n_g_seed))
        m_p = min(mp_l, key=lambda m: abs(float(np.real(m.n_eff)) - n_g_seed))
        ngs[k] = float(n_g_seed - w_m * (float(np.real(m_p.n_eff)) - float(np.real(m_m.n_eff))) / (2.0 * dw_m))

    # Spurious sideband mode for exact dispersion-engineered suppression calculation
    neff_spur = float(np.real(modes["s"].n_eff))
    if 380.0 <= wl_spur_nm <= 6000.0:
        try:
            mesh_sp = float(np.clip(max(cfg.core_mesh_res_um * (wl_spur_nm / 1550.0), 0.038, 0.075 * min_dim), 0.036, 0.090))
            ms_l = solve_modes_fem(
                spec, wl_spur_nm * 1e-9, bend_radius_m=r_m, num_modes=3,
                n_guess=float(np.real(modes["s"].n_eff)),
                core_mesh_res_um=mesh_sp, clad_mesh_res_um=max(mesh_sp * 6.0, 0.24),
                element_order=2, export_grid=False,
            )
            cs_l = [m for m in ms_l if ((m.te_fraction >= 0.5) == is_te) and m.core_confinement >= 0.18]
            neff_spur = float(np.real((cs_l[0] if cs_l else ms_l[0]).n_eff))
        except Exception:
            pass

    # Exact 4-wave azimuthal momentum mismatch delta_m = (m_i - m_s) - (m_p1 - m_p2)
    m_az = {k: float(modes[k].m_azimuthal) for k in ("p1", "p2", "s", "i")}
    m_spur = (om_spur / C0_M_PER_S) * r_m * neff_spur
    dm_bs = (m_az["i"] - m_az["s"]) - (m_az["p1"] - m_az["p2"])
    # Residual fractional cavity frequency mismatch relative to nearest integer azimuthal conservation
    dm_int_err = dm_bs - round(dm_bs)
    fsr_i_rad = C0_M_PER_S / (ngs["i"] * r_m)
    delta_om_bs_rad = dm_int_err * fsr_i_rad
    # Spurious sideband fractional mismatch relative to exact Bragg-matched pump pair
    dm_spur = (m_az["s"] - m_spur) - (m_az["p1"] - m_az["p2"])
    delta_om_spur_rad = (dm_spur - round(dm_bs)) * fsr_i_rad

    # Exact 4-wave full-vector Kerr overlap integral gamma_BS [W^-1 m^-1]
    mp1, mp2, ms, mi = modes["p1"], modes["p2"], modes["s"], modes["i"]
    dx = float(np.mean(np.diff(mp1.x_m)))
    dy = float(np.mean(np.diff(mp1.y_m)))
    da = dx * dy
    xx, yy = np.meshgrid(mp1.x_m * 1e6, mp1.y_m * 1e6)
    inv_tan = 0.0 if abs(cfg.sidewall_deg - 90.0) < 1e-3 else 1.0 / math.tan(math.radians(cfg.sidewall_deg))
    half_w_y = 0.5 * cfg.ring_width_um - np.maximum(yy - cfg.slab_height_um, 0.0) * inv_tan
    in_core = (yy >= cfg.slab_height_um) & (yy <= cfg.film_height_um) & (np.abs(xx) <= np.maximum(half_w_y, 0.0))
    in_slab = (yy >= 0.0) & (yy < cfg.slab_height_um) if cfg.slab_height_um > 1e-6 else np.zeros_like(in_core)
    in_wg = in_core | in_slab

    mat_ref = core_fn(cfg.pump1_wl_nm * 1e-9, T_REF_K)
    n2_core = float(cfg.custom_n2_m2_per_w) if cfg.custom_n2_m2_per_w is not None else float(mat_ref.n2_m2_per_w)
    if n2_core <= 0.0:
        n2_core = 8.9e-19
    n_geom_4 = (
        float(core_fn(cfg.pump1_wl_nm * 1e-9, T_REF_K).nx)
        * float(core_fn(cfg.pump2_wl_nm * 1e-9, T_REF_K).nx)
        * float(core_fn(cfg.signal_wl_nm * 1e-9, T_REF_K).nx)
        * float(core_fn(wl_i_nm * 1e-9, T_REF_K).nx)
    ) ** 0.5

    def _dot(a: EigenmodeResult, b: EigenmodeResult, conj_a: bool = False, conj_b: bool = False) -> np.ndarray:
        ax, ay, az = (np.conj(a.Ex), np.conj(a.Ey), np.conj(a.Ez)) if conj_a else (a.Ex, a.Ey, a.Ez)
        bx, by, bz = (np.conj(b.Ex), np.conj(b.Ey), np.conj(b.Ez)) if conj_b else (b.Ex, b.Ey, b.Ez)
        return ax * bx + ay * by + az * bz

    # Exact isotropic chi^(3) tensor contraction normalized to 1 for co-polarized waves
    t_4w = (
        _dot(mp1, mp2, True, False) * _dot(ms, mi, False, True)
        + _dot(mp1, ms, True, False) * _dot(mp2, mi, False, True)
        + _dot(mp1, mi, True, True) * _dot(mp2, ms, False, False)
    ) / 3.0
    p_norm_4 = math.sqrt(max(mp1.power_w * mp2.power_w * ms.power_w * mi.power_w, 1e-60))
    ov_4w = abs(complex(np.sum(np.where(in_wg, 1.0, 0.0) * (n_geom_4 / (4.0 * Z0_OHM * Z0_OHM)) * t_4w) * da))
    k_si = math.sqrt(om_s * om_i) / C0_M_PER_S
    # Note: in FWM-BS, degeneracy factor is 2 relative to SPM -> coupling rate 2 * gamma_BS * sqrt(P_p1 * P_p2)!
    gamma_bs = k_si * n2_core * ov_4w / max(p_norm_4, 1e-30)
    a_eff_4w_um2 = (p_norm_4 / max(ov_4w, 1e-60)) * 1e12

    # Intracavity pump powers & coupled-mode quantum conversion efficiency (Li et al. Nat. Photon. 2016)
    k_int_p1, k_ext_p1 = om_p1 / cfg.q_int_pump, om_p1 / cfg.q_ext_pump
    k_int_p2, k_ext_p2 = om_p2 / cfg.q_int_pump, om_p2 / cfg.q_ext_pump
    k_tot_p1, k_tot_p2 = k_int_p1 + k_ext_p1, k_int_p2 + k_ext_p2
    t_rt_p1 = ngs["p1"] * l_rt_m / C0_M_PER_S
    t_rt_p2 = ngs["p2"] * l_rt_m / C0_M_PER_S
    t_rt_s = ngs["s"] * l_rt_m / C0_M_PER_S
    t_rt_i = ngs["i"] * l_rt_m / C0_M_PER_S

    p_cav_p1_w = (4.0 * k_ext_p1 / ((k_tot_p1 ** 2) * t_rt_p1)) * (cfg.pump1_power_mw * 1e-3)
    p_cav_p2_w = (4.0 * k_ext_p2 / ((k_tot_p2 ** 2) * t_rt_p2)) * (cfg.pump2_power_mw * 1e-3)

    # Bragg-scattering parametric coupling rate g_BS [rad/s] = 2 * gamma_BS * L_rt * sqrt(P1*P2) / sqrt(T_s * T_i)
    g_bs_rad = 2.0 * gamma_bs * l_rt_m * math.sqrt(max(p_cav_p1_w * p_cav_p2_w, 0.0)) / math.sqrt(t_rt_s * t_rt_i)

    k_int_s, k_ext_s = om_s / cfg.q_int_sig, om_s / cfg.q_ext_sig
    k_int_i, k_ext_i = om_i / cfg.q_int_idl, om_i / cfg.q_ext_idl
    k_tot_s, k_tot_i = k_int_s + k_ext_s, k_int_i + k_ext_i

    coop_bs = (4.0 * (g_bs_rad ** 2)) / max(k_tot_s * k_tot_i, 1e-30)
    eta_ext_prod = (k_ext_s / k_tot_s) * (k_ext_i / k_tot_i)
    # Exact on-resonance vs dispersion-mismatched conversion efficiencies
    eta_ce_matched = eta_ext_prod * (4.0 * coop_bs) / ((1.0 + coop_bs) ** 2)
    det_norm_sq = (2.0 * delta_om_bs_rad / max(math.sqrt(k_tot_s * k_tot_i), 1e3)) ** 2
    eta_ce_actual = eta_ext_prod * (4.0 * coop_bs) / ((1.0 + coop_bs) ** 2 + det_norm_sq)

    # Spurious sideband conversion & suppression ratio [dB]
    det_spur_sq = (2.0 * delta_om_spur_rad / max(k_tot_s, 1e3)) ** 2
    eta_spur = eta_ext_prod * (4.0 * coop_bs) / ((1.0 + coop_bs) ** 2 + det_spur_sq)
    spur_supp_db = 10.0 * math.log10(max(eta_ce_matched, 1e-12) / max(eta_spur, 1e-12))

    # Optimal pump geometric mean power sqrt(P1*P2) for C_BS = 1 (maximum quantum conversion)
    p_opt_mw = math.sqrt(cfg.pump1_power_mw * cfg.pump2_power_mw) / math.sqrt(max(coop_bs, 1e-12))

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "wavelengths_nm": {
            "pump1": round(cfg.pump1_wl_nm, 3),
            "pump2": round(cfg.pump2_wl_nm, 3),
            "signal": round(cfg.signal_wl_nm, 3),
            "idler": round(wl_i_nm, 3),
            "spurious": round(wl_spur_nm, 3),
        },
        "modes": {
            k: {
                "neff": round(float(np.real(modes[k].n_eff)), 6),
                "ng": round(ngs[k], 5),
                "m_azimuthal": round(m_az[k], 3),
                "confinement": round(modes[k].core_confinement, 4),
            }
            for k in ("p1", "p2", "s", "i")
        },
        "metrics": {
            "A_eff_4wave_um2": round(a_eff_4w_um2, 4),
            "gamma_BS_per_W_m": round(gamma_bs, 4),
            "g_BS_2pi_MHz": round((g_bs_rad / (2.0 * math.pi)) * 1e-6, 3),
            "cooperativity_C_BS": round(coop_bs, 4),
            "conversion_eff_phase_matched_pct": round(eta_ce_matched * 100.0, 3),
            "conversion_eff_as_designed_pct": round(eta_ce_actual * 100.0, 3),
            "max_possible_eff_pct": round(eta_ext_prod * 100.0, 3),
            "optimal_pump_geom_mean_mW": round(p_opt_mw, 3),
            "phase_mismatch_dm": round(dm_int_err, 5),
            "freq_mismatch_GHz": round((delta_om_bs_rad / (2.0 * math.pi)) * 1e-9, 4),
            "spurious_detuning_GHz": round((delta_om_spur_rad / (2.0 * math.pi)) * 1e-9, 3),
            "spurious_suppression_dB": round(spur_supp_db, 2),
        },
    }


# ==============================================================================
# 6. Chi(2) Second-Harmonic Generation (SHG) in a Microring Resonator
# ==============================================================================

def simulate_chi2_shg_resonator(cfg: SHGConfig) -> dict[str, Any]:
    """First-principles 2D curved-FEM Chi(2) Second-Harmonic Generation (SHG) resonator solver.

    Solves exact fundamental (`FH`) and second-harmonic (`SH`) 2D curved eigenmodes, evaluates
    the crystal-tensor vector overlap integral `g2`, exact QPM poling period `Lambda_QPM`,
    undepleted-pump conversion efficiency `eta_SHG [%/W]`, and exact pump-depleted SHG power
    transfer curve `P_SH(P_FH)` (Lu et al., Optica 6, 1455, 2019; Bruch et al., APL 2018).
    """
    t0 = time.perf_counter()
    wl_fh_nm = float(cfg.fundamental_wl_nm)
    wl_sh_nm = 0.5 * wl_fh_nm
    wl_fh_m, wl_sh_m = wl_fh_nm * 1e-9, wl_sh_nm * 1e-9
    om_fh = 2.0 * math.pi * C0_M_PER_S / wl_fh_m
    om_sh = 2.0 * om_fh

    spec, core_fn, _, _ = _build_ring_spec_from_params(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        ring_radius_um=cfg.ring_radius_um,
        wl_ref_nm=wl_fh_nm,
    )
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m

    te_fh = cfg.fh_polarization.upper().startswith("TE")
    te_sh = cfg.sh_polarization.upper().startswith("TE")

    # Solve FH and SH modes on the exact same spatial evaluation grid (quadratic order=2 FEM)
    min_dim_sh = min(float(cfg.ring_width_um), float(cfg.film_height_um))
    fh_mesh = float(np.clip(max(cfg.core_mesh_res_um, 0.045, 0.08 * min_dim_sh), 0.042, 0.095))
    fh_clad = max(fh_mesh * 6.0, 0.28)
    m_fh_list = solve_modes_fem(
        spec, wl_fh_m, bend_radius_m=r_m, num_modes=4,
        core_mesh_res_um=fh_mesh, clad_mesh_res_um=fh_clad,
        eval_grid_step_um=0.030, element_order=2, export_grid=True,
    )
    c_fh = [m for m in m_fh_list if ((m.te_fraction >= 0.5) == te_fh) and m.core_confinement >= 0.20]
    m_fh = max(c_fh, key=lambda m: float(np.real(m.n_eff))) if c_fh else m_fh_list[0]

    sh_mesh = float(np.clip(max(0.75 * fh_mesh, 0.038, 0.068 * min_dim_sh), 0.036, 0.080))
    sh_clad = max(sh_mesh * 6.0, 0.24)
    m_sh_list = solve_modes_fem(
        spec, wl_sh_m, bend_radius_m=r_m, num_modes=6,
        core_mesh_res_um=sh_mesh, clad_mesh_res_um=sh_clad,
        eval_grid_step_um=0.030, element_order=2, export_grid=True,
    )
    c_sh = [m for m in m_sh_list if ((m.te_fraction >= 0.5) == te_sh) and m.core_confinement >= 0.20]
    m_sh = max(c_sh, key=lambda m: float(np.real(m.n_eff))) if c_sh else m_sh_list[0]

    neff_fh = float(np.real(m_fh.n_eff))
    neff_sh = float(np.real(m_sh.n_eff))
    # Warm-started group indices for exact cavity roundtrip times
    dw_fh = max(wl_fh_m * 4e-4, 0.4e-9)
    fh_m = min(solve_modes_fem(spec, wl_fh_m - dw_fh, bend_radius_m=r_m, num_modes=2, n_guess=neff_fh, core_mesh_res_um=fh_mesh, clad_mesh_res_um=fh_clad, element_order=2, export_grid=False), key=lambda m: abs(float(np.real(m.n_eff)) - neff_fh))
    fh_p = min(solve_modes_fem(spec, wl_fh_m + dw_fh, bend_radius_m=r_m, num_modes=2, n_guess=neff_fh, core_mesh_res_um=fh_mesh, clad_mesh_res_um=fh_clad, element_order=2, export_grid=False), key=lambda m: abs(float(np.real(m.n_eff)) - neff_fh))
    ng_fh = float(neff_fh - wl_fh_m * (float(np.real(fh_p.n_eff)) - float(np.real(fh_m.n_eff))) / (2.0 * dw_fh))
    dw_sh = max(wl_sh_m * 4e-4, 0.2e-9)
    sh_m = min(solve_modes_fem(spec, wl_sh_m - dw_sh, bend_radius_m=r_m, num_modes=2, n_guess=neff_sh, core_mesh_res_um=sh_mesh, clad_mesh_res_um=sh_clad, element_order=2, export_grid=False), key=lambda m: abs(float(np.real(m.n_eff)) - neff_sh))
    sh_p = min(solve_modes_fem(spec, wl_sh_m + dw_sh, bend_radius_m=r_m, num_modes=2, n_guess=neff_sh, core_mesh_res_um=sh_mesh, clad_mesh_res_um=sh_clad, element_order=2, export_grid=False), key=lambda m: abs(float(np.real(m.n_eff)) - neff_sh))
    ng_sh = float(neff_sh - wl_sh_m * (float(np.real(sh_p.n_eff)) - float(np.real(sh_m.n_eff))) / (2.0 * dw_sh))

    # Crystal Chi(2) tensor selection
    props = get_nonlinear_material_props(cfg.core_material)
    d33 = float(props["d33_pm_per_v"]) * 1e-12
    d31 = float(props["d31_pm_per_v"]) * 1e-12
    d36 = float(props["d36_pm_per_v"]) * 1e-12
    if cfg.custom_d_eff_pm_per_v is not None:
        d_eff_m_v = float(cfg.custom_d_eff_pm_per_v) * 1e-12
        comp_overlap = "custom"
    elif "Xcut" in cfg.core_material:
        # X-cut crystal: optical axis c lies in-plane along x -> TE (Ex) accesses d33, TM (Ey) accesses d31
        d_eff_m_v = d33 if (te_fh and te_sh) else d31
        comp_overlap = "xxx_d33" if (te_fh and te_sh) else "xyy_d31"
    elif abs(d36) > 0.0:
        d_eff_m_v = d36
        comp_overlap = "xyz_d36"
    else:
        # Z-cut / c-axis vertical (LN_Zcut, AlN, ScAlN, 4H-SiC, GaN): TM (Ey) accesses d33, TE+TM accesses d31
        d_eff_m_v = d33 if ((not te_fh) and (not te_sh)) else d31
        comp_overlap = "yyy_d33" if ((not te_fh) and (not te_sh)) else "yxx_d31"

    # First-order QPM Fourier factor (2/pi) when periodically poled (PPLN/PPLT) vs 1.0 for natural modal PM
    qpm_factor = (2.0 / math.pi) if cfg.quasi_phase_matched else 1.0
    # In circular X-cut rings, azimuthal rotation averages cos^3(phi) -> ~0.64 unless radial poling is patterned
    az_orient = 0.64 if ("Xcut" in cfg.core_material and not cfg.quasi_phase_matched) else 1.0
    d_eff_used = d_eff_m_v * qpm_factor * az_orient

    # Exact 2D vector Chi(2) field overlap integral
    dx = float(np.mean(np.diff(m_fh.x_m)))
    dy = float(np.mean(np.diff(m_fh.y_m)))
    da = dx * dy
    xx, yy = np.meshgrid(m_fh.x_m * 1e6, m_fh.y_m * 1e6)
    inv_tan = 0.0 if abs(cfg.sidewall_deg - 90.0) < 1e-3 else 1.0 / math.tan(math.radians(cfg.sidewall_deg))
    half_w_y = 0.5 * cfg.ring_width_um - np.maximum(yy - cfg.slab_height_um, 0.0) * inv_tan
    in_core = (yy >= cfg.slab_height_um) & (yy <= cfg.film_height_um) & (np.abs(xx) <= np.maximum(half_w_y, 0.0))
    in_slab = (yy >= 0.0) & (yy < cfg.slab_height_um) if cfg.slab_height_um > 1e-6 else np.zeros_like(in_core)
    in_wg = in_core | in_slab

    if comp_overlap == "xxx_d33":
        f_ov = np.conj(m_sh.Ex) * (m_fh.Ex ** 2)
    elif comp_overlap == "yyy_d33":
        f_ov = np.conj(m_sh.Ey) * (m_fh.Ey ** 2)
    elif comp_overlap == "yxx_d31":
        f_ov = np.conj(m_sh.Ey) * (m_fh.Ex ** 2)
    elif comp_overlap == "xyy_d31":
        f_ov = np.conj(m_sh.Ex) * (m_fh.Ey ** 2)
    else:
        f_ov = np.conj(m_sh.Ex) * (m_fh.Ex ** 2) + np.conj(m_sh.Ey) * (m_fh.Ey ** 2)

    # Energy-normalized single-photon coupling g2 [rad/s] (Lu et al., Optica 6, 1455, 2019; Boyd SHG):
    #   Total cavity EM energy U_k = P_k * T_rt,k = L_rt * int (1/2) eps0 * n^2 |E_k|^2 dA
    #   Single-photon field amplitude: e_k = sqrt(hbar * om_k / (4 * U_k)) * E_k
    #   H_int = -eps0 * (2 * d_eff) * L_rt * int (e_sh^* e_fh^2) dA * (b^dag a^2 + h.c.) = -hbar * g2 * (b^dag a^2 + h.c.)
    #   => g2 = eps0 * d_eff * L_rt * ov_int * sqrt(hbar * om_fh^2 * om_sh / (16 * U_fh^2 * U_sh))
    # Exact check in plane-wave critical-coupling limit: eta_ring = eta_wg * L_rt^2 * B_FH^2 * B_SH with coefficient 1.000.
    t_rt_fh = ng_fh * l_rt_m / C0_M_PER_S
    t_rt_sh = ng_sh * l_rt_m / C0_M_PER_S
    u_fh_norm = max(m_fh.power_w * t_rt_fh, 1e-40)
    u_sh_norm = max(m_sh.power_w * t_rt_sh, 1e-40)
    ov_int = abs(complex(np.sum(np.where(in_wg, 1.0, 0.0) * f_ov) * da))
    g2_rad_s = (
        EPS0_F_PER_M * d_eff_used * l_rt_m * ov_int
        * math.sqrt(HBAR_J_S * (om_fh ** 2) * om_sh / (16.0 * (u_fh_norm ** 2) * u_sh_norm))
    )
    # Effective SHG mode area A_eff,SHG [um^2]
    mat_fh = core_fn(wl_fh_m, T_REF_K)
    mat_sh = core_fn(wl_sh_m, T_REF_K)
    n_fh_c = float(mat_fh.nx if te_fh else mat_fh.ny)
    n_sh_c = float(mat_sh.nx if te_sh else mat_sh.ny)
    norm_e_fh = float(np.sum((np.abs(m_fh.Ex) ** 2 + np.abs(m_fh.Ey) ** 2) * da))
    norm_e_sh = float(np.sum((np.abs(m_sh.Ex) ** 2 + np.abs(m_sh.Ey) ** 2) * da))
    a_eff_shg_um2 = ((norm_e_fh ** 2) * norm_e_sh / max(ov_int ** 2, 1e-80)) * 1e12

    # QPM poling period & azimuthal grating order M_QPM = |m_SH - 2*m_FH|
    dn_pm = abs(neff_sh - neff_fh)
    lambda_qpm_um = (wl_fh_m / max(2.0 * dn_pm, 1e-9)) * 1e6
    m_qpm = abs(float(m_sh.m_azimuthal) - 2.0 * float(m_fh.m_azimuthal))

    # Undepleted conversion efficiency eta_SHG = P_SH / P_FH^2 [1/W -> %/W] (Lu et al., Optica 2019 Eq. 1)
    k_int_a, k_ext_a = om_fh / cfg.q_int_fh, om_fh / cfg.q_ext_fh
    k_int_b, k_ext_b = om_sh / cfg.q_int_sh, om_sh / cfg.q_ext_sh
    k_tot_a, k_tot_b = k_int_a + k_ext_a, k_int_b + k_ext_b

    # Exact coupled-mode steady-state equations (on dual resonance):
    #   (k_tot_a / 2) * a + 2 * g2 * a^* * b = sqrt(k_ext_a) * s_in
    #   (k_tot_b / 2) * b - g2 * a^2 = 0  =>  b = (2 * g2 / k_tot_b) * a^2
    #   => (k_tot_a / 2) * a + (4 * g2^2 / k_tot_b) * a^3 = sqrt(k_ext_a * P_in / (hbar * om_fh))
    #   P_SH = hbar * om_sh * k_ext_b * |b|^2 = (8 * g2^2 / (hbar * om_fh)) * (k_ext_b / k_tot_b^2) * (k_ext_a / (k_tot_a/2)^2)^2 * P_in^2
    #   Check: om_sh = 2 * om_fh, |a|^2 = 4 * k_ext_a * P_in / (hbar * om_fh * k_tot_a^2),
    #          |b|^2 = 4 * g2^2 * |a|^4 / k_tot_b^2
    #          P_SH / P_in^2 = (2 * hbar * om_fh) * k_ext_b * (4 * g2^2 / k_tot_b^2) * (16 * k_ext_a^2 / (hbar^2 * om_fh^2 * k_tot_a^4))
    #                        = 128 * g2^2 * k_ext_b * k_ext_a^2 / (hbar * om_fh * k_tot_b^2 * k_tot_a^4)
    eta_shg_per_w = (128.0 * (g2_rad_s ** 2) * k_ext_b * (k_ext_a ** 2)) / max(
        HBAR_J_S * om_fh * (k_tot_b ** 2) * (k_tot_a ** 4), 1e-60
    )
    eta_shg_pct_per_w = eta_shg_per_w * 100.0

    # Exact pump-depleted steady-state curve P_SH(P_FH) by solving exact depressed cubic for |a|
    power_curve: list[dict[str, float]] = []
    for p_mw in cfg.pump_powers_mw:
        p_w = max(float(p_mw) * 1e-3, 1e-12)
        rhs = math.sqrt(k_ext_a * p_w / (HBAR_J_S * om_fh))
        c3 = (4.0 * (g2_rad_s ** 2)) / max(k_tot_b, 1e-12)
        c1 = 0.5 * k_tot_a
        if c3 > 1e-30:
            # Solve c3 * a^3 + c1 * a - rhs = 0 (strictly monotone -> unique positive real root!)
            roots = np.roots([c3, 0.0, c1, -rhs])
            real_pos = [float(r.real) for r in roots if abs(r.imag) < 1e-6 * max(abs(r.real), 1.0) and r.real >= 0.0]
            a_sol = real_pos[0] if real_pos else (rhs / c1)
        else:
            a_sol = rhs / c1
        b_sol = (2.0 * g2_rad_s / max(k_tot_b, 1e-12)) * (a_sol ** 2)
        p_sh_w = HBAR_J_S * om_sh * k_ext_b * (b_sol ** 2)
        power_curve.append({
            "P_FH_mW": round(p_mw, 4),
            "P_SH_mW": round(p_sh_w * 1e3, 6),
            "abs_efficiency_pct": round((p_sh_w / p_w) * 100.0, 3),
        })

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "neff_FH": round(neff_fh, 6),
            "neff_SH": round(neff_sh, 6),
            "ng_FH": round(ng_fh, 5),
            "ng_SH": round(ng_sh, 5),
            "d_eff_pm_per_V": round(d_eff_used * 1e12, 3),
            "tensor_component": comp_overlap,
            "A_eff_SHG_um2": round(a_eff_shg_um2, 4),
            "QPM_period_um": round(lambda_qpm_um, 4),
            "QPM_azimuthal_order_M": round(m_qpm, 2),
            "g2_2pi_MHz": round((g2_rad_s / (2.0 * math.pi)) * 1e-6, 4),
            "eta_SHG_undepleted_pct_per_W": round(eta_shg_pct_per_w, 1),
            "max_depleted_eff_pct": round(max((r["abs_efficiency_pct"] for r in power_curve), default=0.0), 3),
        },
        "power_curve": power_curve,
    }


# ==============================================================================
# 7. High-Power Photothermal + Kerr Bistability & Dynamic Shark-Fin Scan
# ==============================================================================

def simulate_thermal_kerr_bistability(cfg: ThermalKerrBistabilityConfig) -> dict[str, Any]:
    """Solve exact photothermal + Kerr steady-state S-curve and dynamic blue/red laser scans.

    Models the classic high-Q microring triangular 'shark-fin' asymmetric resonance broadening
    (Carmon et al., Opt. Express 12, 4742, 2004) by coupling exact 2D-FEM modal thermo-optic
    shift `dn_eff/dT` and Kerr shift `g_NL` to the intracavity photon and thermal equations.
    """
    t0 = time.perf_counter()
    wl0_m = cfg.wavelength_nm * 1e-9
    om0 = 2.0 * math.pi * C0_M_PER_S / wl0_m
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m

    spec, core_fn, clad_fn, top_fn = _build_ring_spec_from_params(
        core_material=cfg.core_material,
        clad_material=cfg.clad_material,
        top_clad_material=cfg.top_clad_material,
        w_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=87.0,
        ring_radius_um=cfg.ring_radius_um,
        wl_ref_nm=cfg.wavelength_nm,
    )
    m0 = _solve_single_mode(spec, wl0_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=True)
    dw_m = max(wl0_m * 4e-4, 0.4e-9)
    m_m = _solve_single_mode(spec, wl0_m - dw_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=False)
    m_p = _solve_single_mode(spec, wl0_m + dw_m, r_m, cfg.polarization, "fem", cfg.core_mesh_res_um, export_grid=False)
    ng0 = float(np.real(m0.n_eff) - wl0_m * (np.real(m_p.n_eff) - np.real(m_m.n_eff)) / (2.0 * dw_m))
    t_rt_s = ng0 * l_rt_m / C0_M_PER_S

    mat_c = core_fn(wl0_m, T_REF_K)
    mat_b = clad_fn(wl0_m, T_REF_K)
    mat_t = top_fn(wl0_m, T_REF_K)
    is_te = cfg.polarization.upper().startswith("TE")
    dn_dt_mod = float(m0.core_confinement * mat_c.dn_dt_per_k + (1.0 - m0.core_confinement) * mat_b.dn_dt_per_k)
    vec_nl = compute_vector_kerr_gamma(
        m0,
        w_bottom_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=87.0,
        n_core=float(mat_c.nx if is_te else mat_c.ny),
        n_clad=float(mat_b.nx),
        n_top=float(mat_t.nx),
        n2_core_m2_per_w=max(float(mat_c.n2_m2_per_w), 0.0),
        n2_clad_m2_per_w=float(mat_b.n2_m2_per_w),
        n2_top_m2_per_w=float(mat_t.n2_m2_per_w),
    )

    k_int = om0 / max(cfg.q_int, 100.0)
    k_ext = om0 / max(cfg.q_ext, 100.0)
    k_tot = k_int + k_ext
    p_in_w = max(float(cfg.input_power_mw) * 1e-3, 1e-9)

    # Nonlinear frequency shift per Watt of intracavity circulating power P_cav:
    #   delta_omega_Kerr = g_NL * P_cav
    #   delta_omega_th   = (om0 / ng0) * dn_dt_mod * R_th * (f_abs * k_int * T_rt * P_cav)
    g_kerr = (vec_nl["gamma_nl_per_w_m"] * l_rt_m) / max(t_rt_s, 1e-18)
    r_th_k_w = float(cfg.r_th_k_per_mw) * 1e3
    c_th_rad_per_s_w = (om0 / max(ng0, 1.0)) * dn_dt_mod * r_th_k_w * (float(cfg.absorption_fraction) * k_int * t_rt_s)
    c_tot = g_kerr + c_th_rad_per_s_w  # Total red-shift rate [rad/(s*W)]

    # Exact bistability threshold power (normalized shift X = c_tot * P_cav_max / (k_tot/2) > 8 / (3*sqrt(3)) ~ 1.5396)
    p_cav_res_per_w_in = (4.0 * k_ext) / ((k_tot ** 2) * t_rt_s)
    bistab_x = abs(c_tot) * (p_cav_res_per_w_in * p_in_w) / (0.5 * k_tot)
    p_bist_mw = (8.0 / (3.0 * math.sqrt(3.0))) * (0.5 * k_tot) / max(abs(c_tot) * p_cav_res_per_w_in, 1e-30) * 1e3

    # 1. Exact steady-state multi-valued S-curve across laser detuning delta_0 = omega_0 - omega_p
    half_span_ghz = 0.5 * max(float(cfg.scan_span_ghz), abs(c_tot * p_cav_res_per_w_in * p_in_w / (2.0 * math.pi)) * 1e-9 * 2.2, 1.0)
    det_ghz = np.linspace(-0.4 * half_span_ghz, +1.6 * half_span_ghz, int(cfg.num_scan_points))
    det_rad = 2.0 * math.pi * det_ghz * 1e9

    # Exact cubic for P_cav: P_cav * [(k_tot/2)^2 + (delta_0 - c_tot*P_cav)^2] = (k_ext / T_rt) * P_in
    rhs_s = (k_ext / t_rt_s) * p_in_w
    lower_T, upper_T, unstable_T = [], [], []
    for d0 in det_rad:
        if abs(c_tot) > 1e-18:
            roots = np.roots([c_tot ** 2, -2.0 * d0 * c_tot, (0.5 * k_tot) ** 2 + d0 ** 2, -rhs_s])
            real_r = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-6 * max(abs(r.real), 1e-12) and r.real >= 0.0)
        else:
            real_r = [rhs_s / ((0.5 * k_tot) ** 2 + d0 ** 2)]
        # Convert each intracavity power P_cav -> exact through-port transmission T = |1 - (k_ext * T_rt / sqrt(P_in)) * A|^2
        t_vals = []
        for pc in real_r:
            d_eff = d0 - c_tot * pc
            t_amp = 1.0 - k_ext / complex(0.5 * k_tot, d_eff)
            t_vals.append(float(abs(t_amp) ** 2))
        if len(t_vals) == 1:
            lower_T.append(round(t_vals[0], 5))
            upper_T.append(round(t_vals[0], 5))
            unstable_T.append(None)
        else:
            # Lowest P_cav is cold branch, highest P_cav is hot/upper-power branch (deeper dip near resonance)
            lower_T.append(round(t_vals[0], 5))
            unstable_T.append(round(t_vals[1], 5))
            upper_T.append(round(t_vals[-1], 5))

    # 2. Dynamic blue-to-red (increasing delta_0) and red-to-blue (decreasing delta_0) laser scan ODE
    #    Adiabatic optical cavity (tau_ph ~ ns << tau_th ~ 10 us) + exact thermal ODE d(Delta_T)/dt!
    sweep_rate_rad_s2 = 2.0 * math.pi * max(float(cfg.scan_speed_ghz_per_ms), 0.1) * 1e12
    tau_th_s = max(float(cfg.tau_th_us) * 1e-6, 1e-8)

    def _run_dynamic_scan(det_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t_out = np.empty_like(det_seq)
        dt_out = np.empty_like(det_seq)
        d_temp = 0.0
        for idx, d0 in enumerate(det_seq):
            d_th = (om0 / max(ng0, 1.0)) * dn_dt_mod * d_temp
            # Solve fast optical Kerr cubic at fixed thermal shift d_th
            d_k0 = float(d0) - d_th
            if abs(g_kerr) > 1e-18:
                rk = np.roots([g_kerr ** 2, -2.0 * d_k0 * g_kerr, (0.5 * k_tot) ** 2 + d_k0 ** 2, -rhs_s])
                rp = sorted(float(r.real) for r in rk if abs(r.imag) < 1e-6 * max(abs(r.real), 1e-12) and r.real >= 0.0)
                pc_now = rp[-1] if (idx > 0 and d_temp > 0.05) else rp[0]
            else:
                pc_now = rhs_s / ((0.5 * k_tot) ** 2 + d_k0 ** 2)
            d_eff = d_k0 - g_kerr * pc_now
            t_out[idx] = float(abs(1.0 - k_ext / complex(0.5 * k_tot, d_eff)) ** 2)
            dt_out[idx] = d_temp
            if idx < len(det_seq) - 1:
                dt_step = abs(float(det_seq[idx + 1] - d0)) / sweep_rate_rad_s2
                # Semi-implicit Euler step for unconditionally stable thermal relaxation
                p_abs_w = float(cfg.absorption_fraction) * k_int * t_rt_s * pc_now
                t_eq = r_th_k_w * p_abs_w
                alpha_e = math.exp(-dt_step / tau_th_s)
                d_temp = t_eq + (d_temp - t_eq) * alpha_e
        return t_out, dt_out

    thru_blue_to_red, dtemp_b2r = _run_dynamic_scan(det_rad)
    thru_red_to_blue_rev, dtemp_r2b_rev = _run_dynamic_scan(det_rad[::-1])
    thru_red_to_blue = thru_red_to_blue_rev[::-1]
    dtemp_r2b = dtemp_r2b_rev[::-1]

    # Measure apparent broadened FWHM on blue-to-red vs cold cavity FWHM
    cold_fwhm_ghz = (k_tot / (2.0 * math.pi)) * 1e-9
    dip_min = float(np.min(thru_blue_to_red))
    half_t = 0.5 * (1.0 + dip_min)
    idx_b2r = np.flatnonzero(thru_blue_to_red <= half_t)
    app_fwhm_b2r_ghz = float(det_ghz[idx_b2r[-1]] - det_ghz[idx_b2r[0]]) if len(idx_b2r) >= 2 else cold_fwhm_ghz
    idx_r2b = np.flatnonzero(thru_red_to_blue <= half_t)
    app_fwhm_r2b_ghz = float(det_ghz[idx_r2b[-1]] - det_ghz[idx_r2b[0]]) if len(idx_r2b) >= 2 else cold_fwhm_ghz
    hysteresis_ghz = abs(float(det_ghz[int(np.argmin(thru_blue_to_red))]) - float(det_ghz[int(np.argmin(thru_red_to_blue))]))

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "is_bistable": bool(bistab_x > (8.0 / (3.0 * math.sqrt(3.0)))),
            "bistability_threshold_mW": round(p_bist_mw, 4),
            "bistability_parameter_X": round(bistab_x, 3),
            "cold_FWHM_GHz": round(cold_fwhm_ghz, 4),
            "apparent_FWHM_blue_to_red_GHz": round(app_fwhm_b2r_ghz, 4),
            "apparent_FWHM_red_to_blue_GHz": round(app_fwhm_r2b_ghz, 4),
            "thermal_broadening_factor": round(app_fwhm_b2r_ghz / max(cold_fwhm_ghz, 1e-6), 2),
            "scan_hysteresis_GHz": round(hysteresis_ghz, 4),
            "peak_delta_T_K": round(float(np.max(dtemp_b2r)), 3),
            "thermal_vs_kerr_ratio": round(abs(c_th_rad_per_s_w) / max(abs(g_kerr), 1e-18), 2),
        },
        "curves": {
            "detuning_GHz": np.round(det_ghz, 5).tolist(),
            "steady_lower_T": lower_T,
            "steady_upper_T": upper_T,
            "steady_unstable_T": unstable_T,
            "dynamic_blue_to_red_T": np.round(thru_blue_to_red, 5).tolist(),
            "dynamic_red_to_blue_T": np.round(thru_red_to_blue, 5).tolist(),
            "dynamic_delta_T_K": np.round(dtemp_b2r, 4).tolist(),
        },
    }
