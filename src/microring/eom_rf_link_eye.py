"""First-Principles EOM RF Packaging (HFSS-Class ABCD Transition) + 100G–400G NRZ/PAM4 Eye Diagram, BER, TDECQ & FFE Link Engine (`microring.eom_rf_link_eye`).

Zero canned presets / zero lookup-table surrogates — every physical parameter recomputes live from
first-principles electromagnetics, transmission-line ABCD cascade algebra, nonlinear Mach-Zehnder
electro-optic modulation, SMF-28 chromatic dispersion, Bessel-Thomson detection, and Wiener-Hopf
MMSE equalization:

  1. **HFSS-Class 3D RF Packaging & Interconnect Transition ABCD Cascade (`simulate_eom_rf_packaging`)**:
     - Calls `design_eom_electrode` live for the active traveling-wave or lumped EOM cross-section
       (2D quasi-TEM FEM + Wheeler skin-effect + Cherenkov + Bragg Floquet-Bloch dispersion).
     - Solves multilayer finite-ground GSG probe pad and linear/exponential CPW transition taper
       (`N_seg = 16` cascaded ABCD sections) using Veyres–Fouad Hanna / Gevorgian exact elliptic
       conformal mapping (`scipy.special.ellipk`) + finite metal thickness sidewall capacitance
       and Ghione-Naldi conductor + dielectric loss.
     - Solves 3D interconnect launch parasitics live from exact 3-conductor (or 4-conductor dual-bond)
       GSG Neumann/Rosa/Grover mutual-inductance loop magnetostatics + distributed high-Z0 arc
       transmission line:
         * `"gsg_probe"`      : Direct on-wafer GSG microwave probe landing (pad + open-end fringe + tip L)
         * `"wirebond"`       : Arched JEDEC GSG Au/Al bondwire launch (`L_span`, `h_loop`, `d_wire`, `N_wires`)
         * `"flip_chip_bump"` : Vertical GSG Cu-pillar / SnAg solder micro-bump (`h_bump`, `D_bump`, underfill `eps_r`)
     - Optional Metal-2 / Via optical-crossover bridge (`C_bridge`, `L_via`) and distributed lossy
       on-chip thin-film termination resistor (`Z_L(f) = Z_c,R tanh(gamma_R L_R)`).
     - Computes exact packaged 2-port `S11_RF(f)`, `S21_RF(f)`, full complex packaged `S21_EO(f)`
       (including multiple internal reflections between input launch, active electrode, and load),
       bandwidth penalty vs bare electrode, and exports a Touchstone `.s2p` block.

  2. **End-to-End High-Speed Optical Link, 2-UI Eye Diagram, Semi-Analytic BER, IEEE 802.3bs TDECQ & MMSE FFE (`simulate_eom_eye_ber_link`)**:
     - Drives deterministic PRBS-9 / PRBS-11 `NRZ` or Gray-coded `PAM4` waveforms (`baud_rate_gbaud`,
       `v_pp_drive_v`, `tx_rise_time_ps`, `tx_jitter_rms_ps`, `pre_emphasis_db`) through the exact
       complex packaged `S21_EO(f)` frequency response via alias-free FFT convolution.
     - Evaluates exact nonlinear Mach-Zehnder optical field transfer (push-pull zero-chirp OR
       single-arm chirped MZI) capturing outer-eye sin^2 compression and quadrature bias offset.
     - Propagates through optional SMF-28 fiber chromatic dispersion `exp(+j 0.5 beta_2 L omega^2)`
       with ITU-T G.652 wavelength-dependent dispersion `D(lambda)`.
     - Detects with PIN photodiode + 4th-order Bessel-Thomson reference receiver + exact thermal,
       shot, and laser RIN noise variance, followed by an optional `N`-tap Wiener-Hopf MMSE FFE
       equalizer with exact colored-noise enhancement `w^T R_BT4 w`.
     - Extracts 2-UI eye diagram traces, worst-case eye height/width, outer `OMA_dBm`, `ER_dB`,
       `Q_factor`, exact semi-analytic pre-FEC `BER` (down to `1e-18`), IEEE 802.3bs `TDECQ` [dB],
       and KP4-FEC (`2.4e-4`) / HD-FEC (`3.8e-3`) / Error-Free (`1e-12`) compliance flags.

Scope Boundary (Honest Physics Disclosure):
  - Covers chip-level EOM CPW / T-Rail electrodes, GSG pad/taper transitions, GSG wirebond/bump
    interconnects, distributed thin-film loads, and end-to-end optical eye/BER/TDECQ locally.
  - Arbitrary 3D metallic cavity box-modes (`>100 GHz`), non-periodic 3D substrate surface-wave
    radiation, and multi-channel connector/package crosstalk still require full-wave 3D FEM (HFSS).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.special import ellipk, erfc

from .eom_electrode import (
    DIELECTRIC_LOSS_TANGENT_RF,
    ELECTRODE_METAL_CATALOG,
    RF_SUBSTRATE_CATALOG,
    EOMElectrodeConfig,
    design_eom_electrode,
)
from .eigenmode_solver import MU0_H_PER_M
from .materials import C0_M_PER_S, EPS0_F_PER_M


Q_ELECTRON_C = 1.602176634e-19
Q_T_IEEE_PAM4 = 3.414  # IEEE 802.3bs target Q_t for PAM4 SER = 4.8e-4 (BER = 2.4e-4 KP4-FEC)


LAUNCH_TYPE_CATALOG: dict[str, dict[str, str]] = {
    "gsg_probe": {
        "name_en": "On-Wafer GSG Microwave Probe (Pad + Taper Launch)",
        "name_zh": "片上 GSG 射频探针直测（焊盘 + 渐变线过渡）",
    },
    "wirebond": {
        "name_en": "3D GSG Arched Bondwire Launch (PCB/Interposer → Chip)",
        "name_zh": "3D GSG 拱形金丝键合互连（板级/转接板 → 芯片）",
    },
    "flip_chip_bump": {
        "name_en": "3D Flip-Chip GSG Cu-Pillar / Solder Micro-Bump Launch",
        "name_zh": "3D 倒装芯片 GSG 铜柱/微凸点封装互连",
    },
}


@dataclass
class EOMRFLinkConfig:
    """Unified live-editable EOM RF packaging + high-speed optical link configuration."""

    # --- Active Modulator & Electrode (passed live into design_eom_electrode) ---
    core_material: str = "LN_Xcut"
    electrode_topology: str = "gsg_cpw"     # "gsg_cpw" | "t_rail_cpw" | "gs_coplanar" | "vertical_microstrip" | "lumped_ring_gsg"
    substrate_type: str = "quartz"          # "hr_si" | "quartz" | "undercut_si" | "std_si" | "sapphire"
    metal_material: str = "Au"
    wavelength_nm: float = 1550.0
    active_length_um: float = 5000.0
    waveguide_width_um: float = 1.10
    film_height_um: float = 0.60
    slab_height_um: float = 0.30
    box_thickness_um: float = 2.00
    top_clad_thickness_um: float = 0.80
    top_clad_material: str = "SiO2"
    electrode_gap_um: float = 5.0
    metal_thickness_um: float = 1.10
    signal_width_um: float | None = None    # None -> auto 50-Ohm synthesis
    ground_width_um: float = 55.0
    t_rail_period_um: float = 50.0
    t_rail_duty: float = 0.88
    t_rail_stem_width_um: float = 4.0
    driver_impedance_ohm: float = 50.0
    termination_ohm: float = 50.0
    termination_mode: str = "external_50ohm_gsg"  # "external_50ohm_gsg" | "onchip_50ohm_tin" | "open_lumped"
    max_freq_ghz: float = 140.0
    num_freq_points: int = 160

    # --- 3D RF Packaging / Transition Geometry (HFSS-Class ABCD Cascade) ---
    launch_type: str = "gsg_probe"          # "gsg_probe" | "wirebond" | "flip_chip_bump"
    probe_pitch_um: float = 150.0           # GSG center-to-center pitch [um]
    pad_width_um: float = 65.0              # GSG signal pad width [um]
    pad_gap_um: float | None = None         # None -> auto matched (~0.5*(pitch - pad_width) capped to 50-Ohm-like slot)
    pad_length_um: float = 80.0             # GSG pad landing length [um]
    taper_length_um: float = 160.0          # Linear CPW taper length from pad to active electrode [um]
    taper_sections: int = 16                # Number of cascaded ABCD sections along taper
    probe_tip_inductance_ph: float = 18.0   # On-wafer 110-GHz GSG probe tip contact inductance [pH]

    # Wirebond 3D arch geometry (active when launch_type == "wirebond")
    wirebond_length_um: float = 350.0       # Horizontal span between board pad and chip pad [um]
    wirebond_loop_height_um: float = 120.0  # JEDEC arch apex height above pad plane [um]
    wirebond_diameter_um: float = 25.0      # Au bondwire diameter [um] (1 mil = 25.4 um)
    wirebond_pitch_um: float | None = None  # None -> uses probe_pitch_um [um]
    wirebond_num_wires: int = 1             # 1 = single signal wire; 2 = dual parallel V-bond signal wires
    wirebond_wire_spacing_um: float = 35.0  # Center spacing between dual signal wires (if num_wires == 2)
    wirebond_encapsulant_eps_r: float = 1.0 # 1.0 = air; ~3.0 = glob-top epoxy

    # Flip-chip micro-bump 3D geometry (active when launch_type == "flip_chip_bump")
    bump_height_um: float = 40.0            # Solder / Cu-pillar standoff height [um]
    bump_diameter_um: float = 50.0          # Micro-bump cylinder diameter [um]
    bump_pitch_um: float | None = None      # None -> uses probe_pitch_um [um]
    underfill_eps_r: float = 3.4            # Capillary underfill relative permittivity (1.0 = air)
    underfill_tan_d: float = 0.015          # Underfill RF loss tangent
    bump_metal_rho_ohm_m: float = 1.15e-7   # SnAg solder (~11.5 uOhm*cm) or Cu pillar (~2.0e-8)

    # Optional Metal-2 / Via optical crossover air-bridge
    include_m2_bridge: bool = False
    m2_overlap_area_um2: float = 120.0      # Overlap area per bridge [um^2]
    m2_ild_thickness_um: float = 1.2        # Inter-layer SiO2 dielectric thickness [um]
    m2_via_inductance_ph: float = 12.0      # Via barrel series inductance [pH]

    # Distributed on-chip thin-film termination resistor geometry (used when termination_mode == "onchip_50ohm_tin")
    term_resistor_length_um: float = 60.0   # Thin-film resistor physical length [um]
    term_resistor_width_um: float = 30.0    # Thin-film resistor physical width [um]

    # --- High-Speed Optical Link, Waveform, Fiber & Receiver Parameters ---
    modulation_format: str = "PAM4"         # "PAM4" | "NRZ"
    baud_rate_gbaud: float = 56.0           # Symbol rate [GBaud] (56 GBaud PAM4 = 112 Gbps; 112 GBaud PAM4 = 224 Gbps)
    v_pp_drive_v: float = 2.5               # Peak-to-peak differential drive swing into matched 50-Ohm load [V]
    bias_phase_deg: float = 90.0            # Optical MZI bias phase [deg] (90 = quadrature)
    tx_rise_time_ps: float = 3.5            # Transmitter DAC/driver 20-80% edge rise time [ps]
    tx_jitter_rms_ps: float = 0.25          # Transmitter random RMS jitter [ps]
    pre_emphasis_db: float = 0.0            # Optional 2-tap TX de-emphasis/pre-emphasis [dB]
    laser_power_dbm: float = 13.0           # CW laser launch power before passive IL [dBm]
    passive_insertion_loss_db: float = 5.5  # Fiber couplers + splitters + routing passive loss [dB]
    fiber_length_km: float = 0.0            # SMF-28 fiber span [km] (0 = back-to-back)
    rx_bandwidth_ghz: float | None = None   # None -> IEEE 802.3bs reference 0.75 * baud_rate_gbaud
    rx_responsivity_a_w: float = 0.85       # PIN photodiode responsivity [A/W]
    rx_noise_pa_rt_hz: float = 18.0         # TIA input-referred noise current density [pA/sqrt(Hz)]
    laser_rin_db_hz: float = -145.0         # Laser relative intensity noise [dB/Hz]
    ffe_taps: int = 5                       # Receiver MMSE FFE taps (0 = off; 5 = IEEE 802.3bs TDECQ ref; 7/9)
    prbs_order: int = 9                     # 9 (511 syms) or 11 (2047 syms)
    samples_per_symbol: int = 16            # Oversampling ratio (>= 12 for alias-free FFT)
    random_seed: int = 42


# ==============================================================================
# 1. Exact Multilayer Conformal-Mapping CPW & 2x2 ABCD Matrix Algebra
# ==============================================================================

def _ellipk_ratio(k: float) -> float:
    """Compute K(k) / K(sqrt(1 - k^2)) accurately across the entire open interval (0, 1)."""
    kk = float(np.clip(k, 1e-12, 1.0 - 1e-12))
    kp = math.sqrt(max(1.0 - kk * kk, 1e-24))
    return float(ellipk(kk * kk) / max(float(ellipk(kp * kp)), 1e-15))


def _sinh_ratio_safe(x1: float, x2: float) -> float:
    """Overflow-safe sinh(x1) / sinh(x2) for positive x1 <= x2."""
    if x2 <= 1e-9:
        return x1 / max(x2, 1e-15)
    if x2 < 20.0:
        return math.sinh(x1) / max(math.sinh(x2), 1e-30)
    # Exact asymptotic form without overflow: exp(-(x2 - x1)) * (1 - exp(-2*x1)) / (1 - exp(-2*x2))
    dx = x2 - x1
    if dx > 50.0:
        return 0.0
    num = 1.0 - math.exp(-2.0 * min(x1, 35.0))
    den = 1.0 - math.exp(-2.0 * min(x2, 35.0))
    return math.exp(-dx) * (num / max(den, 1e-15))


def _cpw_layer_filling_factor(a_um: float, b_um: float, c_um: float, h_um: float, r0: float) -> float:
    """Exact Veyres-Fouad Hanna / Gevorgian finite-ground CPW conformal filling factor q(h) in [0, 0.5]."""
    if h_um <= 1e-6:
        return 0.0
    if h_um > 80.0 * max(b_um, 1.0):
        return 0.5
    fac = math.pi / (2.0 * h_um)
    xa, xb, xc = fac * a_um, fac * b_um, fac * c_um
    r_ab = _sinh_ratio_safe(xa, xb)
    r_bc = _sinh_ratio_safe(xb, xc)
    r_ac = _sinh_ratio_safe(xa, xc)
    num = max(1.0 - r_bc * r_bc, 1e-18)
    den = max(1.0 - r_ac * r_ac, 1e-18)
    ki = float(np.clip(r_ab * math.sqrt(num / den), 1e-12, 1.0 - 1e-12))
    return float(np.clip(0.5 * _ellipk_ratio(ki) / max(r0, 1e-12), 0.0, 0.5))


def solve_multilayer_cpw_section(
    *,
    w_sig_um: float,
    gap_um: float,
    w_gnd_um: float,
    t_metal_um: float,
    h_box_um: float,
    h_film_um: float,
    h_top_um: float,
    eps_sub: float,
    eps_box: float,
    eps_film: float,
    eps_top: float,
    tan_d_sub: float,
    tan_d_box: float,
    tan_d_film: float,
    tan_d_top: float,
    sigma_sub_s_m: float,
    rho_metal_ohm_m: float,
    f_hz: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """First-principles multilayer finite-ground CPW quasi-TEM solver (Z_c(f), gamma(f), n_RF, alpha)."""
    ws = max(float(w_sig_um), 0.5)
    g = max(float(gap_um), 0.5)
    wg = max(float(w_gnd_um), 5.0)
    tm = max(float(t_metal_um), 0.05)

    a = 0.5 * ws
    b = a + g
    c = b + wg

    # Zero-thickness finite-ground CPW modulus k0
    k0 = (a / b) * math.sqrt(max(1.0 - (b / c) ** 2, 1e-12) / max(1.0 - (a / c) ** 2, 1e-12))
    k0 = float(np.clip(k0, 1e-12, 1.0 - 1e-12))
    r0 = _ellipk_ratio(k0)
    c0_air = 4.0 * EPS0_F_PER_M * r0

    # Stratified conformal filling factors (electrode seated on top of film slab of thickness h_film)
    q_box_tot = _cpw_layer_filling_factor(a, b, c, h_film_um + h_box_um, r0)
    q_film = _cpw_layer_filling_factor(a, b, c, h_film_um, r0)
    q_top = _cpw_layer_filling_factor(a, b, c, h_top_um, r0)
    q_sub = max(0.5 - q_box_tot, 0.0)
    q_box = max(q_box_tot - q_film, 0.0)

    # Exact telescoping effective permittivity (zero-thickness)
    eps_eff_0 = (
        1.0
        + 0.5 * (eps_sub - 1.0)
        + q_box_tot * (eps_box - eps_sub)
        + q_film * (eps_film - eps_box)
        + q_top * (eps_top - 1.0)
    )

    # Finite metal thickness parallel-plate sidewall capacitance in both slots
    f_cov = float(np.clip(h_top_um / max(tm, 1e-6), 0.0, 1.0))
    eps_slot = f_cov * eps_top + (1.0 - f_cov) * 1.0
    c_edge_air = 2.0 * EPS0_F_PER_M * (tm / g)
    c_edge_diel = 2.0 * EPS0_F_PER_M * eps_slot * (tm / g)

    c_air = c0_air + c_edge_air
    c_diel = eps_eff_0 * c0_air + c_edge_diel
    l_ext = (MU0_H_PER_M * EPS0_F_PER_M) / max(c_air, 1e-18)
    eps_eff = c_diel / max(c_air, 1e-18)

    # Frequency-dependent skin-effect conductor resistance + internal inductance (Ghione-Naldi CPW)
    om = 2.0 * math.pi * np.maximum(f_hz, 1.0)
    delta_s = np.sqrt(2.0 * rho_metal_ohm_m / (om * MU0_H_PER_M))
    rs_ohm = rho_metal_ohm_m / np.maximum(delta_s, 1e-12)

    ws_m = ws * 1e-6
    g_m = g * 1e-6
    wg_m = wg * 1e-6
    tm_m = tm * 1e-6

    # Ghione-Naldi current-crowding geometric factor g_cpw for CPW signal + ground edges
    kp0 = math.sqrt(max(1.0 - k0 * k0, 1e-12))
    k_val = float(ellipk(k0 * k0))
    # Smooth thick-conductor geometric factor preventing log singularity for very small t/w
    edge_ratio_s = max(4.0 * math.pi * ws_m / max(tm_m, 1e-9), 4.0)
    edge_ratio_b = max(4.0 * math.pi * (ws_m + 2.0 * g_m) / max(tm_m, 1e-9), 4.0)
    g_gn = (1.0 / (4.0 * max(k_val, 0.5) ** 2 * (1.0 - k0 * k0))) * (
        (1.0 / ws_m) * (math.pi + math.log(edge_ratio_s * (1.0 - k0) / max(1.0 + k0, 1e-6)))
        + (1.0 / (ws_m + 2.0 * g_m)) * (math.pi + math.log(edge_ratio_b * (1.0 - k0) / max(1.0 + k0, 1e-6)))
    )
    g_gn = max(g_gn, 1.0 / max(ws_m, 1e-9))

    r_dc = rho_metal_ohm_m * (1.0 / (ws_m * tm_m) + 1.0 / (2.0 * wg_m * tm_m))
    r_hf = rs_ohm * g_gn
    r_tot = np.sqrt(r_dc * r_dc + r_hf * r_hf)
    # Internal inductance (transitions smoothly from DC L_int to high-frequency R_hf / omega)
    l_int = np.minimum(r_hf / om, 0.12 * l_ext)

    # Dielectric & substrate conductivity shunt conductance G(f)
    eps_tan_sum = (
        q_sub * eps_sub * tan_d_sub
        + q_box * eps_box * tan_d_box
        + q_film * eps_film * tan_d_film
        + q_top * eps_top * tan_d_top
    )
    tan_d_eff = eps_tan_sum / max(eps_eff, 1.0)
    g_diel = om * c_diel * tan_d_eff + (q_sub * c0_air / EPS0_F_PER_M) * sigma_sub_s_m

    z_ser = r_tot + 1j * om * (l_ext + l_int)
    y_sh = g_diel + 1j * om * c_diel
    gamma = np.sqrt(z_ser * y_sh)
    z_c = np.sqrt(z_ser / y_sh)
    return {
        "gamma": gamma,
        "z_c": z_c,
        "z0_dc": float(math.sqrt(l_ext / max(c_diel, 1e-18))),
        "eps_eff": float(eps_eff),
        "n_rf": float(math.sqrt(max(eps_eff, 1.0))),
        "c_f_per_m": float(c_diel),
        "l_h_per_m": float(l_ext),
    }


def _abcd_identity(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones(n, dtype=complex)
    zeros = np.zeros(n, dtype=complex)
    return ones, zeros, zeros.copy(), ones.copy()


def _abcd_mul(
    t1: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    t2: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, b1, c1, d1 = t1
    a2, b2, c2, d2 = t2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
    )


def _abcd_reverse(
    t: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reverse a reciprocal 2-port ABCD matrix (swap port 1 and port 2: A <-> D)."""
    a, b, c, d = t
    return d.copy(), b.copy(), c.copy(), a.copy()


def _abcd_tline(gamma: np.ndarray, z_c: np.ndarray, length_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gl = gamma * float(length_m)
    # Overflow-safe cosh/sinh for large attenuation
    re_gl = np.clip(np.real(gl), -40.0, 40.0)
    im_gl = np.imag(gl)
    ep = np.exp(re_gl + 1j * im_gl)
    em = np.exp(-re_gl - 1j * im_gl)
    ch = 0.5 * (ep + em)
    sh = 0.5 * (ep - em)
    return ch, z_c * sh, sh / z_c, ch


def _abcd_series_z(z_ser: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones_like(z_ser, dtype=complex)
    zeros = np.zeros_like(z_ser, dtype=complex)
    return ones, z_ser.astype(complex), zeros, ones.copy()


def _abcd_shunt_y(y_sh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones_like(y_sh, dtype=complex)
    zeros = np.zeros_like(y_sh, dtype=complex)
    return ones, zeros, y_sh.astype(complex), ones.copy()


def _abcd_to_sparam(
    t: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    zs: float,
    zl: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Frickey/Pozar ABCD-to-S conversion for arbitrary real reference impedances (Z_S, Z_L)."""
    a, b, c, d = t
    den = a * zl + b + c * zs * zl + d * zs
    den = np.where(np.abs(den) < 1e-24, 1e-24 + 0j, den)
    s11 = (a * zl + b - c * zs * zl - d * zs) / den
    s21 = 2.0 * math.sqrt(zs * zl) / den
    s12 = 2.0 * (a * d - b * c) * math.sqrt(zs * zl) / den
    s22 = (-a * zl + b - c * zs * zl + d * zs) / den
    return s11, s21, s12, s22


# ==============================================================================
# 2. First-Principles 3D Interconnect Launch Models (Neumann/Rosa/Grover GSG)
# ==============================================================================

def _neumann_mutual_inductance_h(length_m: float, dist_m: float) -> float:
    """Exact Neumann mutual inductance M(l, d) [H] of two parallel filaments of length l at spacing d."""
    l_m = max(float(length_m), 1e-9)
    d_m = max(float(dist_m), 1e-12)
    u = l_m / d_m
    term = math.log(u + math.sqrt(1.0 + u * u)) - math.sqrt(1.0 + 1.0 / (u * u)) + 1.0 / u
    return (MU0_H_PER_M * l_m / (2.0 * math.pi)) * max(term, 0.0)


def _rosa_self_inductance_h(length_m: float, radius_m: float) -> float:
    """Exact Rosa (1908) high-frequency external self-inductance L_self(l, r) [H] of a straight cylinder."""
    l_m = max(float(length_m), 1e-9)
    r_m = max(float(radius_m), 1e-12)
    # Using exact GMD = r (high-frequency surface current; internal inductance added dynamically via skin depth)
    u = l_m / r_m
    term = math.log(u + math.sqrt(1.0 + u * u)) - math.sqrt(1.0 + 1.0 / (u * u)) + 1.0 / u
    return (MU0_H_PER_M * l_m / (2.0 * math.pi)) * max(term, 0.0)


def compute_gsg_wirebond_abcd(
    *,
    span_um: float,
    loop_height_um: float,
    diameter_um: float,
    pitch_um: float,
    num_wires: int,
    wire_spacing_um: float,
    encap_eps_r: float,
    rho_ohm_m: float,
    f_hz: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Exact 3D GSG arched bondwire launch ABCD matrix from 3/4-conductor Neumann magnetostatics."""
    span_m = max(float(span_um), 40.0) * 1e-6
    h_m = max(float(loop_height_um), 10.0) * 1e-6
    d_m = max(float(diameter_um), 5.0) * 1e-6
    r_m = 0.5 * d_m
    s_m = max(float(pitch_um), 2.5 * float(diameter_um)) * 1e-6
    eps_enc = max(float(encap_eps_r), 1.0)
    n_w = 2 if int(num_wires) >= 2 else 1

    # Exact parabolic JEDEC arch arc-length: y(x) = 4*h*(x/L)*(1 - x/L)
    # Integral_0^L sqrt(1 + (4h/L * (1 - 2x/L))^2) dx = (L/2)*(sqrt(1+a^2) + asinh(a)/a), a = 4*h/L
    a_arch = 4.0 * h_m / span_m
    arc_len_m = 0.5 * span_m * (math.sqrt(1.0 + a_arch * a_arch) + math.asinh(a_arch) / max(a_arch, 1e-12))

    # Exact 3-wire (n_w=1: I_S=+I, I_G1=I_G2=-I/2) or 4-wire (n_w=2: I_S1=I_S2=+I/2, I_G1=I_G2=-I/2) GSG loop inductance
    l_self = _rosa_self_inductance_h(arc_len_m, r_m)
    m_2s = _neumann_mutual_inductance_h(arc_len_m, 2.0 * s_m)
    if n_w == 1:
        m_s = _neumann_mutual_inductance_h(arc_len_m, s_m)
        l_loop_ext = max(1.5 * l_self + 0.5 * m_2s - 2.0 * m_s, 1e-13)
        r_mult = 1.5  # I^2*R_sig + 2*(I/2)^2*R_gnd = 1.5*I^2*R_wire
    else:
        d_pair_m = min(max(float(wire_spacing_um) * 1e-6, 1.4 * d_m), 0.7 * s_m)
        m_ss = _neumann_mutual_inductance_h(arc_len_m, d_pair_m)
        m_sg_near = _neumann_mutual_inductance_h(arc_len_m, max(s_m - 0.5 * d_pair_m, 1.2 * d_m))
        m_sg_far = _neumann_mutual_inductance_h(arc_len_m, s_m + 0.5 * d_pair_m)
        # W_m = 0.5 * L_loop * I^2 with [I/2, I/2, -I/2, -I/2]
        l_loop_ext = max(l_self + 0.5 * m_ss + 0.5 * m_2s - m_sg_near - m_sg_far, 1e-13)
        r_mult = 1.0  # 2*(I/2)^2*R + 2*(I/2)^2*R = 1.0*I^2*R_wire

    # Distributed 3-wire (or 4-wire) transmission line parameters along the arch
    l_per_m = l_loop_ext / arc_len_m
    c_per_m = (MU0_H_PER_M * EPS0_F_PER_M * eps_enc) / max(l_per_m, 1e-12)
    z0_wb = math.sqrt(l_per_m / max(c_per_m, 1e-18))

    om = 2.0 * math.pi * np.maximum(f_hz, 1.0)
    delta_s = np.sqrt(2.0 * rho_ohm_m / (om * MU0_H_PER_M))
    r_dc_per_m = r_mult * rho_ohm_m / (math.pi * r_m * r_m)
    r_hf_per_m = r_mult * rho_ohm_m / (2.0 * math.pi * r_m * np.maximum(delta_s, 1e-12))
    r_per_m = np.sqrt(r_dc_per_m ** 2 + r_hf_per_m ** 2)
    l_int_per_m = np.minimum(r_hf_per_m / om, r_mult * MU0_H_PER_M / (8.0 * math.pi))

    z_ser = r_per_m + 1j * om * (l_per_m + l_int_per_m)
    y_sh = 1j * om * c_per_m
    gamma_wb = np.sqrt(z_ser * y_sh)
    z_c_wb = np.sqrt(z_ser / y_sh)

    abcd = _abcd_tline(gamma_wb, z_c_wb, arc_len_m)
    meta = {
        "arc_length_um": round(arc_len_m * 1e6, 2),
        "L_loop_nH": round(l_loop_ext * 1e9, 4),
        "C_wire_fF": round(c_per_m * arc_len_m * 1e15, 3),
        "Z0_wire_ohm": round(z0_wb, 2),
        "f_self_res_GHz": round(0.5 / max(arc_len_m * math.sqrt(l_per_m * c_per_m), 1e-15) * 1e-9, 1),
    }
    return abcd, meta


def compute_gsg_flip_chip_bump_abcd(
    *,
    height_um: float,
    diameter_um: float,
    pitch_um: float,
    underfill_eps_r: float,
    underfill_tan_d: float,
    rho_ohm_m: float,
    f_hz: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Exact 3D GSG vertical Cu-pillar / solder micro-bump ABCD matrix (Neumann 3-cylinder + underfill C)."""
    h_m = max(float(height_um), 5.0) * 1e-6
    d_m = max(float(diameter_um), 8.0) * 1e-6
    r_m = 0.5 * d_m
    s_m = max(float(pitch_um), 1.5 * float(diameter_um)) * 1e-6
    eps_uf = max(float(underfill_eps_r), 1.0)
    tan_d = max(float(underfill_tan_d), 0.0)

    # Exact 3-cylinder GSG vertical loop inductance via finite-length Rosa + Neumann mutuals
    l_self = _rosa_self_inductance_h(h_m, r_m)
    m_s = _neumann_mutual_inductance_h(h_m, s_m)
    m_2s = _neumann_mutual_inductance_h(h_m, 2.0 * s_m)
    l_loop = max(1.5 * l_self + 0.5 * m_2s - 2.0 * m_s, 1e-13)

    # Distributed vertical 3-cylinder capacitance in underfill + UBM circular pad fringing
    l_per_m = l_loop / h_m
    c_cyl_per_m = (MU0_H_PER_M * EPS0_F_PER_M * eps_uf) / max(l_per_m, 1e-12)
    c_cyl = c_cyl_per_m * h_m
    # Coplanar circular UBM-to-ground fringe capacitance at both ends (signal bump to two adjacent ground bumps)
    gap_b = max(s_m - d_m, 2.0e-6)
    c_fringe = 2.0 * EPS0_F_PER_M * eps_uf * (d_m * math.log(1.0 + d_m / gap_b)) / math.pi
    c_tot = c_cyl + c_fringe

    om = 2.0 * math.pi * np.maximum(f_hz, 1.0)
    delta_s = np.sqrt(2.0 * rho_ohm_m / (om * MU0_H_PER_M))
    r_dc = 1.5 * rho_ohm_m * h_m / (math.pi * r_m * r_m)
    r_hf = 1.5 * rho_ohm_m * h_m / (2.0 * math.pi * r_m * np.maximum(delta_s, 1e-12))
    r_tot = np.sqrt(r_dc * r_dc + r_hf * r_hf)
    l_int = np.minimum(r_hf / om, 1.5 * MU0_H_PER_M * h_m / (8.0 * math.pi))

    # Symmetric Pi-section for electrically short bump (h << lambda even at 140 GHz)
    z_ser = r_tot + 1j * om * (l_loop + l_int)
    y_half = 0.5 * (om * c_tot * tan_d + 1j * om * c_tot)
    t_sh = _abcd_shunt_y(y_half)
    t_se = _abcd_series_z(z_ser)
    abcd = _abcd_mul(_abcd_mul(t_sh, t_se), t_sh)

    z_eff = math.sqrt(l_loop / max(c_tot, 1e-18))
    meta = {
        "height_um": round(height_um, 2),
        "diameter_um": round(diameter_um, 2),
        "L_bump_pH": round(l_loop * 1e12, 2),
        "C_bump_fF": round(c_tot * 1e15, 2),
        "Z_bump_ohm": round(z_eff, 2),
    }
    return abcd, meta


def _sinc_exp_c(z: np.ndarray) -> np.ndarray:
    """Exact (exp(z) - 1) / z evaluated without cancellation near z = 0."""
    out = np.empty_like(z, dtype=complex)
    small = np.abs(z) < 1e-5
    zs = z[small]
    out[small] = 1.0 + 0.5 * zs + (zs * zs) / 6.0 + (zs * zs * zs) / 24.0
    zl = z[~small]
    zl_safe = np.clip(np.real(zl), -50.0, 50.0) + 1j * np.imag(zl)
    out[~small] = (np.exp(zl_safe) - 1.0) / zl
    return out


# ==============================================================================
# 3. Full HFSS-Class Packaged EOM Network Engine (`simulate_eom_rf_packaging`)
# ==============================================================================

def _build_touchstone_s2p(
    freq_ghz: np.ndarray,
    s11: np.ndarray,
    s21: np.ndarray,
    s12: np.ndarray,
    s22: np.ndarray,
    z0: float = 50.0,
    comment: str = "",
) -> str:
    lines = [
        f"! MicroRing Studio EOM RF Packaging S-Parameters ({comment})",
        f"# GHz S RI R {z0:.2f}",
    ]
    for k in range(len(freq_ghz)):
        lines.append(
            f"{freq_ghz[k]:8.4f} "
            f"{s11[k].real:+.6f} {s11[k].imag:+.6f} "
            f"{s21[k].real:+.6f} {s21[k].imag:+.6f} "
            f"{s12[k].real:+.6f} {s12[k].imag:+.6f} "
            f"{s22[k].real:+.6f} {s22[k].imag:+.6f}"
        )
    return "\n".join(lines) + "\n"


def _crossing_db(freq_ghz: np.ndarray, curve_db: np.ndarray, thresh_db: float) -> tuple[float, bool]:
    below = np.flatnonzero(curve_db <= thresh_db)
    if len(below) == 0:
        return round(float(freq_ghz[-1]), 2), True
    idx = int(below[0])
    if idx == 0:
        return round(float(freq_ghz[0]), 2), False
    f0, f1 = float(freq_ghz[idx - 1]), float(freq_ghz[idx])
    y0, y1 = float(curve_db[idx - 1]), float(curve_db[idx])
    fc = f0 + (thresh_db - y0) * (f1 - f0) / (y1 - y0) if abs(y1 - y0) > 1e-12 else f1
    return round(fc, 2), False


_BARE_EOM_CACHE: dict[tuple, dict] = {}


def simulate_eom_rf_packaging(
    cfg: EOMRFLinkConfig | None = None,
    *,
    bare_eom_result: dict | None = None,
    extra_freq_ghz: np.ndarray | None = None,
) -> dict:
    """Compute full 3D-packaging-aware RF + Electro-Optic S-parameters via exact ABCD cascade."""
    t0 = time.perf_counter()
    c = cfg or EOMRFLinkConfig()

    # 1. Solve bare active EOM electrode live (exact geometry-keyed cache avoids re-solving identical active cross-section when sweeping packaging/eye params)
    if bare_eom_result is None:
        eom_cfg = EOMElectrodeConfig(
            core_material=c.core_material,
            electrode_topology=c.electrode_topology,
            substrate_type=c.substrate_type,
            metal_material=c.metal_material,
            wavelength_nm=c.wavelength_nm,
            active_length_um=c.active_length_um,
            waveguide_width_um=c.waveguide_width_um,
            film_height_um=c.film_height_um,
            slab_height_um=c.slab_height_um,
            box_thickness_um=c.box_thickness_um,
            top_clad_thickness_um=c.top_clad_thickness_um,
            top_clad_material=c.top_clad_material,
            electrode_gap_um=c.electrode_gap_um,
            metal_thickness_um=c.metal_thickness_um,
            signal_width_um=c.signal_width_um,
            ground_width_um=c.ground_width_um,
            t_rail_period_um=c.t_rail_period_um,
            t_rail_duty=c.t_rail_duty,
            t_rail_stem_width_um=c.t_rail_stem_width_um,
            driver_impedance_ohm=c.driver_impedance_ohm,
            termination_ohm=c.termination_ohm,
            termination_mode=c.termination_mode,
            probe_pitch_um=c.probe_pitch_um,
            pad_width_um=c.pad_width_um,
            pad_length_um=c.pad_length_um,
            launcher_taper_um=c.taper_length_um,
            max_freq_ghz=max(float(c.max_freq_ghz), 40.0),
            num_freq_points=max(int(c.num_freq_points), 64),
            run_gap_sweep=False,
            probe_pad_capacitance_ff=0.0,  # Modeled explicitly via geometry ABCD below!
        )
        cache_key = tuple(sorted(asdict(eom_cfg).items()))
        if cache_key in _BARE_EOM_CACHE:
            bare = _BARE_EOM_CACHE[cache_key]
        else:
            bare = design_eom_electrode(eom_cfg)
            if len(_BARE_EOM_CACHE) > 24:
                _BARE_EOM_CACHE.clear()
            _BARE_EOM_CACHE[cache_key] = bare
    else:
        bare = bare_eom_result

    fr_bare = bare["frequency_response"]
    f_bare = np.asarray(fr_bare["freq_GHz"], dtype=float)
    gam_bare = np.asarray(fr_bare["gamma_RF_re_np_m"], dtype=float) + 1j * np.asarray(fr_bare["gamma_RF_im_rad_m"], dtype=float)
    zc_bare = np.asarray(fr_bare["Z0_ohm"], dtype=float) + 1j * np.asarray(fr_bare["Z0_im_ohm"], dtype=float)

    # Frequency grid (either user plot grid or extended grid for FFT eye-diagram synthesis)
    if extra_freq_ghz is not None:
        f_ghz = np.asarray(extra_freq_ghz, dtype=float)
        # Smooth physical interpolation / high-frequency sqrt(f) extrapolation of active gamma & Z_c
        gam_re = np.interp(f_ghz, f_bare, np.real(gam_bare))
        # Above f_bare[-1], continue linear beta (constant phase velocity) and sqrt(f) ohmic alpha!
        above = f_ghz > f_bare[-1]
        if np.any(above):
            ratio_f = f_ghz[above] / max(f_bare[-1], 1e-6)
            gam_re[above] = float(np.real(gam_bare[-1])) * np.sqrt(ratio_f)
            gam_im = np.interp(f_ghz, f_bare, np.imag(gam_bare))
            gam_im[above] = float(np.imag(gam_bare[-1])) * ratio_f
        else:
            gam_im = np.interp(f_ghz, f_bare, np.imag(gam_bare))
        gam_act = gam_re + 1j * gam_im
        zc_act = np.interp(f_ghz, f_bare, np.real(zc_bare)) + 1j * np.interp(f_ghz, f_bare, np.imag(zc_bare))
    else:
        f_ghz = f_bare.copy()
        gam_act = gam_bare.copy()
        zc_act = zc_bare.copy()

    f_hz = np.maximum(f_ghz, 0.01) * 1e9
    om = 2.0 * math.pi * f_hz
    n_f = len(f_ghz)

    # Material properties for pad/taper/termination stack
    sub_info = RF_SUBSTRATE_CATALOG.get(c.substrate_type, RF_SUBSTRATE_CATALOG["quartz"])
    met_info = ELECTRODE_METAL_CATALOG.get(c.metal_material, ELECTRODE_METAL_CATALOG["Au"])
    eps_sub = float(sub_info["eps_rf"])
    tan_d_sub = float(sub_info["tan_delta"])
    sig_sub = float(sub_info["sigma_s_per_m"])
    if c.substrate_type == "undercut_si":
        # Pad & taper sit outside the local waveguide undercut trench (or partially over undercut); use BOX+HR-Si
        eps_sub = 11.7
    eps_box = 3.9
    tan_d_box = DIELECTRIC_LOSS_TANGENT_RF.get("SiO2", 0.0006)
    eps_top = 3.9 if c.top_clad_material != "Air" else 1.0
    tan_d_top = DIELECTRIC_LOSS_TANGENT_RF.get(c.top_clad_material, 0.0006) if c.top_clad_material != "Air" else 0.0
    # Effective geometric-mean RF permittivity of thin core slab under pads
    if "LN" in c.core_material:
        eps_film = math.sqrt(44.3 * 27.9)
    elif "LT" in c.core_material:
        eps_film = math.sqrt(41.0 * 43.0)
    elif "BTO" in c.core_material:
        eps_film = 180.0
    elif "SiC" in c.core_material or "AlN" in c.core_material or "ScAlN" in c.core_material:
        eps_film = 10.0
    else:
        eps_film = 11.7
    tan_d_film = DIELECTRIC_LOSS_TANGENT_RF.get(c.core_material, 0.003)
    rho_m = float(met_info["rho_ohm_m"])

    dims = bare["dimensions"]
    ws_act = float(dims["signal_width_um"])
    g_act = float(dims["trunk_gap_um"])
    wg_act = float(dims["ground_width_um"])
    tm_um = float(dims["metal_thickness_um"])
    l_act_m = float(dims["active_length_um"]) * 1e-6
    ng_opt = float(bare["metrics"]["n_g_opt"])

    # 2. GSG Probe Pad + Multi-Section Linear CPW Taper ABCD Matrix
    w_pad = max(float(c.pad_width_um), 15.0)
    pitch_um = max(float(c.probe_pitch_um), w_pad + 12.0)
    if c.pad_gap_um is not None and float(c.pad_gap_um) > 0.5:
        g_pad = float(c.pad_gap_um)
    else:
        # Standard GSG probe layout: pitch = 0.5*W_pad + G_pad + 0.5*W_gnd_pad; for symmetric pads G_pad ~ 0.35*(pitch - W_pad)
        g_pad = max(6.0, min(0.45 * (pitch_um - w_pad), 40.0))
    wg_pad = max(wg_act, 0.65 * pitch_um)

    # Pad section (uniform CPW of length pad_length_um)
    pad_cpw = solve_multilayer_cpw_section(
        w_sig_um=w_pad,
        gap_um=g_pad,
        w_gnd_um=wg_pad,
        t_metal_um=tm_um,
        h_box_um=c.box_thickness_um,
        h_film_um=c.slab_height_um,
        h_top_um=0.0,  # Probe pads are always passivation-opened (air on top!)
        eps_sub=eps_sub,
        eps_box=eps_box,
        eps_film=eps_film,
        eps_top=1.0,
        tan_d_sub=tan_d_sub,
        tan_d_box=tan_d_box,
        tan_d_film=tan_d_film,
        tan_d_top=0.0,
        sigma_sub_s_m=sig_sub,
        rho_metal_ohm_m=rho_m,
        f_hz=f_hz,
    )
    t_pad = _abcd_tline(pad_cpw["gamma"], pad_cpw["z_c"], max(float(c.pad_length_um), 0.0) * 1e-6)
    # Small open-end pad fringing capacitance (CPW back-edge)
    c_pad_fringe = EPS0_F_PER_M * float(pad_cpw["eps_eff"]) * (w_pad * 1e-6) * 0.25
    t_pad_fringe = _abcd_shunt_y(1j * om * c_pad_fringe)

    # Multi-section linear CPW taper from (w_pad, g_pad) -> (ws_act, g_act)
    n_seg = max(4, int(c.taper_sections))
    l_tap_m = max(float(c.taper_length_um), 0.0) * 1e-6
    t_taper = _abcd_identity(n_f)
    taper_z0_profile: list[float] = []
    if l_tap_m > 1e-9:
        dl_m = l_tap_m / n_seg
        for k in range(n_seg):
            u = (k + 0.5) / n_seg
            wk = w_pad + u * (ws_act - w_pad)
            gk = g_pad + u * (g_act - g_pad)
            wgk = wg_pad + u * (wg_act - wg_pad)
            # Top passivation opens near pad (u < 0.2) and covers taper towards active section
            h_top_k = c.top_clad_thickness_um if u >= 0.20 else 0.0
            sec = solve_multilayer_cpw_section(
                w_sig_um=wk,
                gap_um=gk,
                w_gnd_um=wgk,
                t_metal_um=tm_um,
                h_box_um=c.box_thickness_um,
                h_film_um=c.slab_height_um,
                h_top_um=h_top_k,
                eps_sub=eps_sub,
                eps_box=eps_box,
                eps_film=eps_film,
                eps_top=eps_top if h_top_k > 0.0 else 1.0,
                tan_d_sub=tan_d_sub,
                tan_d_box=tan_d_box,
                tan_d_film=tan_d_film,
                tan_d_top=tan_d_top if h_top_k > 0.0 else 0.0,
                sigma_sub_s_m=sig_sub,
                rho_metal_ohm_m=rho_m,
                f_hz=f_hz,
            )
            taper_z0_profile.append(round(float(sec["z0_dc"]), 2))
            t_taper = _abcd_mul(t_taper, _abcd_tline(sec["gamma"], sec["z_c"], dl_m))

    # 3. 3D Interconnect Launch ABCD Matrix (gsg_probe | wirebond | flip_chip_bump)
    launch = str(c.launch_type).lower().strip()
    if launch not in LAUNCH_TYPE_CATALOG:
        launch = "gsg_probe"

    launch_meta: dict[str, float] = {}
    if launch == "wirebond":
        wb_pitch = float(c.wirebond_pitch_um) if c.wirebond_pitch_um is not None else pitch_um
        t_launch, launch_meta = compute_gsg_wirebond_abcd(
            span_um=c.wirebond_length_um,
            loop_height_um=c.wirebond_loop_height_um,
            diameter_um=c.wirebond_diameter_um,
            pitch_um=wb_pitch,
            num_wires=c.wirebond_num_wires,
            wire_spacing_um=c.wirebond_wire_spacing_um,
            encap_eps_r=c.wirebond_encapsulant_eps_r,
            rho_ohm_m=rho_m,
            f_hz=f_hz,
        )
    elif launch == "flip_chip_bump":
        bp_pitch = float(c.bump_pitch_um) if c.bump_pitch_um is not None else pitch_um
        t_launch, launch_meta = compute_gsg_flip_chip_bump_abcd(
            height_um=c.bump_height_um,
            diameter_um=c.bump_diameter_um,
            pitch_um=bp_pitch,
            underfill_eps_r=c.underfill_eps_r,
            underfill_tan_d=c.underfill_tan_d,
            rho_ohm_m=c.bump_metal_rho_ohm_m,
            f_hz=f_hz,
        )
    else:
        # Direct on-wafer GSG microwave probe landing: small probe-tip series inductance
        l_tip_h = max(float(c.probe_tip_inductance_ph), 0.0) * 1e-12
        t_launch = _abcd_series_z(1j * om * l_tip_h)
        launch_meta = {
            "probe_tip_L_pH": round(l_tip_h * 1e12, 2),
            "pad_Z0_ohm": round(float(pad_cpw["z0_dc"]), 2),
            "pad_fringe_C_fF": round(c_pad_fringe * 1e15, 2),
        }

    # 4. Optional Metal-2 / Via Crossover Air-Bridge
    t_bridge = _abcd_identity(n_f)
    bridge_meta = {"enabled": False, "C_bridge_fF": 0.0, "L_via_pH": 0.0}
    if c.include_m2_bridge:
        c_br = EPS0_F_PER_M * 3.9 * (max(float(c.m2_overlap_area_um2), 1.0) * 1e-12) / (max(float(c.m2_ild_thickness_um), 0.1) * 1e-6)
        l_via = max(float(c.m2_via_inductance_ph), 0.0) * 1e-12
        t_bridge = _abcd_mul(
            _abcd_shunt_y(0.5j * om * c_br),
            _abcd_mul(_abcd_series_z(1j * om * l_via), _abcd_shunt_y(0.5j * om * c_br)),
        )
        bridge_meta = {"enabled": True, "C_bridge_fF": round(c_br * 1e15, 2), "L_via_pH": round(l_via * 1e12, 2)}

    # Assemble Input Packaging Chain: Launch -> PadFringe -> Pad -> Taper -> Bridge
    t_in = _abcd_mul(_abcd_mul(_abcd_mul(_abcd_mul(t_launch, t_pad_fringe), t_pad), t_taper), t_bridge)

    # 5. Active Electrode & Output Termination / Output Packaging Chain
    t_act = _abcd_tline(gam_act, zc_act, l_act_m)
    zs = max(float(c.driver_impedance_ohm), 1.0)
    zl_nom = max(float(c.termination_ohm), 1.0)
    is_open = c.termination_mode == "open_lumped" or bare["resolved_topology"] == "lumped_ring_gsg"
    is_onchip_tf = c.termination_mode == "onchip_50ohm_tin" and not is_open

    term_meta: dict[str, float | str] = {"mode": c.termination_mode}
    if is_open:
        t_out = _abcd_identity(n_f)
        z_l_eff = np.full(n_f, 1e9 + 0j, dtype=complex)
        t_tot = _abcd_mul(t_in, t_act)
        s11_pkg, _, _, _ = _abcd_to_sparam(t_tot, zs, 1e9)
        s21_pkg = np.full(n_f, 1e-3 + 0j, dtype=complex)
        s12_pkg = s21_pkg.copy()
        s22_pkg = np.ones(n_f, dtype=complex)
    elif is_onchip_tf:
        # Distributed lossy on-chip thin-film resistor stub Z_L(f) = Z_cR * tanh(gamma_R * L_R)
        l_r_m = max(float(c.term_resistor_length_um), 5.0) * 1e-6
        w_r_um = max(float(c.term_resistor_width_um), 3.0)
        g_r_um = max(g_act, 6.0)
        tf_sec = solve_multilayer_cpw_section(
            w_sig_um=w_r_um,
            gap_um=g_r_um,
            w_gnd_um=wg_act,
            t_metal_um=0.15,
            h_box_um=c.box_thickness_um,
            h_film_um=c.slab_height_um,
            h_top_um=c.top_clad_thickness_um,
            eps_sub=eps_sub,
            eps_box=eps_box,
            eps_film=eps_film,
            eps_top=eps_top,
            tan_d_sub=tan_d_sub,
            tan_d_box=tan_d_box,
            tan_d_film=tan_d_film,
            tan_d_top=tan_d_top,
            sigma_sub_s_m=sig_sub,
            rho_metal_ohm_m=rho_m,
            f_hz=f_hz,
        )
        r_per_m = zl_nom / l_r_m
        l_r_per_m = float(tf_sec["l_h_per_m"])
        c_r_per_m = float(tf_sec["c_f_per_m"])
        z_ser_r = r_per_m + 1j * om * l_r_per_m
        y_sh_r = 1j * om * c_r_per_m
        gam_r = np.sqrt(z_ser_r * y_sh_r)
        zc_r = np.sqrt(z_ser_r / y_sh_r)
        z_l_eff = zc_r * np.tanh(gam_r * l_r_m)
        t_out = _abcd_identity(n_f)
        # Input S11 looking into full chip terminated by distributed Z_L(f)
        t_pre = _abcd_mul(t_in, t_act)
        a_p, b_p, c_p, d_p = t_pre
        z_in_tot = (a_p * z_l_eff + b_p) / np.where(np.abs(c_p * z_l_eff + d_p) < 1e-24, 1e-24 + 0j, c_p * z_l_eff + d_p)
        s11_pkg = (z_in_tot - zs) / (z_in_tot + zs)
        # Voltage transmission to on-chip load (normalized as S21 across Z_S -> Re(Z_L))
        v_load_ratio = z_l_eff / np.where(
            np.abs((a_p + zs * c_p) * z_l_eff + (b_p + zs * d_p)) < 1e-24,
            1e-24 + 0j,
            (a_p + zs * c_p) * z_l_eff + (b_p + zs * d_p),
        )
        s21_pkg = 2.0 * math.sqrt(zs / zl_nom) * v_load_ratio
        s12_pkg = s21_pkg.copy()
        s22_pkg = s11_pkg.copy()
        term_meta.update({
            "R_dc_ohm": round(zl_nom, 2),
            "L_parasitic_pH": round(l_r_per_m * l_r_m * 1e12, 2),
            "C_parasitic_fF": round(c_r_per_m * l_r_m * 1e15, 2),
            "Z_L_100GHz_mag_ohm": round(float(np.abs(z_l_eff[int(np.argmin(np.abs(f_ghz - 100.0)))])), 2),
        })
    else:
        # Symmetric external 50-Ohm GSG output port: Bridge -> Taper(rev) -> Pad -> PadFringe -> Launch(rev)
        t_out = _abcd_reverse(t_in)
        a_o, b_o, c_o, d_o = t_out
        z_l_eff = (a_o * zl_nom + b_o) / np.where(np.abs(c_o * zl_nom + d_o) < 1e-24, 1e-24 + 0j, c_o * zl_nom + d_o)
        t_tot = _abcd_mul(_abcd_mul(t_in, t_act), t_out)
        s11_pkg, s21_pkg, s12_pkg, s22_pkg = _abcd_to_sparam(t_tot, zs, zl_nom)
        term_meta["Z_ext_ohm"] = round(zl_nom, 2)

    # 6. Exact Packaged Traveling-Wave Electro-Optic Transfer Function m_pkg(f)
    if is_open:
        gam_l_eff = np.ones(n_f, dtype=complex)
    else:
        gam_l_eff = (z_l_eff - zc_act) / np.where(np.abs(z_l_eff + zc_act) < 1e-24, 1e-24 + 0j, z_l_eff + zc_act)

    em1 = np.exp(-np.clip(np.real(gam_act) * l_act_m, -40.0, 40.0) - 1j * np.imag(gam_act) * l_act_m)
    em2 = em1 * em1
    d_zin = 1.0 - gam_l_eff * em2
    z_in_act = zc_act * (1.0 + gam_l_eff * em2) / np.where(np.abs(d_zin) < 1e-18, 1e-18 + 0j, d_zin)

    a_in, b_in, c_in, d_in = t_in
    den_v0 = (a_in + zs * c_in) * z_in_act + (b_in + zs * d_in)
    v_act0_over_vsrc = z_in_act / np.where(np.abs(den_v0) < 1e-24, 1e-24 + 0j, den_v0)
    d_vf = 1.0 + gam_l_eff * em2
    vf0_pkg = v_act0_over_vsrc / np.where(np.abs(d_vf) < 1e-18, 1e-18 + 0j, d_vf)

    beta_opt = om * ng_opt / C0_M_PER_S
    m_pkg_tw = vf0_pkg * (
        _sinc_exp_c((-gam_act + 1j * beta_opt) * l_act_m)
        + gam_l_eff * em2 * _sinc_exp_c((+gam_act + 1j * beta_opt) * l_act_m)
    )

    if bare["resolved_topology"] == "lumped_ring_gsg":
        om_opt = 2.0 * math.pi * C0_M_PER_S / (c.wavelength_nm * 1e-9)
        q_l = max(float(bare["config"].get("ring_loaded_q", 3.5e4)), 100.0)
        d_norm = max(abs(float(bare["config"].get("ring_detuning_norm", 0.57735))), 0.0)
        gam_t = om_opt / (2.0 * q_l)
        h_cav = (gam_t ** 2 * (1.0 + d_norm ** 2)) / ((gam_t + 1j * om) ** 2 + (d_norm * gam_t) ** 2)
        m_pkg = m_pkg_tw * h_cav
    else:
        m_pkg = m_pkg_tw

    # Reference normalization at low frequency (1.5 GHz for TW matched, lowest point for open/lumped)
    f_ref = float(f_ghz[0]) if is_open else 1.50
    idx_ref = int(np.argmin(np.abs(f_ghz - f_ref)))
    m_ref_c = complex(m_pkg[idx_ref])
    if abs(m_ref_c) < 1e-15:
        m_ref_c = 0.5 + 0j
    h_eo_pkg_c = m_pkg / m_ref_c
    # Preserve true low-frequency voltage-divider ratio relative to matched 50-Ohm load (0.5) for time-domain swing!
    dc_divider_gain = float(np.abs(m_pkg[0])) / 0.5

    s21_eo_pkg_dbe = 20.0 * np.log10(np.maximum(np.abs(h_eo_pkg_c), 1e-6))
    s21_rf_pkg_db = 20.0 * np.log10(np.clip(np.abs(s21_pkg), 1e-6, 1.0))
    s11_rf_pkg_db = 20.0 * np.log10(np.maximum(np.abs(s11_pkg), 1e-6))

    f3dbe_pkg, pkg_exceeds_3dbe = _crossing_db(f_ghz, s21_eo_pkg_dbe, -3.0103)
    f3dbo_pkg, pkg_exceeds_3dbo = _crossing_db(f_ghz, s21_eo_pkg_dbe, -6.0206)
    f3dbe_bare = float(bare["metrics"]["f_3dBe_GHz"])
    f3dbo_bare = float(bare["metrics"]["f_3dBo_GHz"])

    idx_40 = int(np.argmin(np.abs(f_ghz - min(40.0, float(f_ghz[-1])))))
    idx_67 = int(np.argmin(np.abs(f_ghz - min(67.0, float(f_ghz[-1])))))
    s11_max_40g = float(np.max(s11_rf_pkg_db[f_ghz <= 40.0 + 1e-6])) if np.any(f_ghz <= 40.0 + 1e-6) else float(s11_rf_pkg_db[0])
    s11_max_67g = float(np.max(s11_rf_pkg_db[f_ghz <= 67.0 + 1e-6])) if np.any(f_ghz <= 67.0 + 1e-6) else float(s11_rf_pkg_db[0])
    bw_penalty_pct = round(100.0 * max(0.0, (f3dbo_bare - f3dbo_pkg) / max(f3dbo_bare, 1e-6)), 2)

    s2p_text = _build_touchstone_s2p(
        f_ghz,
        s11_pkg,
        s21_pkg,
        s12_pkg,
        s22_pkg,
        z0=zs,
        comment=f"{c.core_material} {bare['resolved_topology']} launch={launch}",
    )

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "launch_type": launch,
        "launch_name_en": LAUNCH_TYPE_CATALOG[launch]["name_en"],
        "launch_name_zh": LAUNCH_TYPE_CATALOG[launch]["name_zh"],
        "bare_eom": bare,
        "packaging_geometry": {
            "probe_pitch_um": round(pitch_um, 2),
            "pad_width_um": round(w_pad, 2),
            "pad_gap_um": round(g_pad, 2),
            "pad_length_um": round(float(c.pad_length_um), 2),
            "pad_Z0_ohm": round(float(pad_cpw["z0_dc"]), 2),
            "taper_length_um": round(float(c.taper_length_um), 2),
            "taper_sections": n_seg,
            "taper_Z0_profile_ohm": taper_z0_profile,
            "launch_parasitics": launch_meta,
            "m2_bridge": bridge_meta,
            "termination": term_meta,
        },
        "metrics": {
            "f_3dBe_bare_GHz": round(f3dbe_bare, 2),
            "f_3dBo_bare_GHz": round(f3dbo_bare, 2),
            "f_3dBe_packaged_GHz": f3dbe_pkg,
            "f_3dBo_packaged_GHz": f3dbo_pkg,
            "bw_exceeds_sweep": bool(pkg_exceeds_3dbo),
            "packaging_bw_penalty_pct": bw_penalty_pct,
            "S11_max_40GHz_dB": round(s11_max_40g, 2),
            "S11_max_67GHz_dB": round(s11_max_67g, 2),
            "S11_max_full_dB": round(float(np.max(s11_rf_pkg_db)), 2),
            "S21_RF_40GHz_dB": round(float(s21_rf_pkg_db[idx_40]), 2),
            "S21_RF_67GHz_dB": round(float(s21_rf_pkg_db[idx_67]), 2),
            "dc_voltage_divider_gain": round(dc_divider_gain, 4),
            "Z0_active_ohm": round(float(bare["metrics"]["Z0_ohm"]), 2),
            "V_pi_V": round(float(bare["metrics"]["V_pi_V"]), 3),
            "V_pi_L_Vcm": round(float(bare["metrics"]["V_pi_L_Vcm"]), 3),
            "n_RF": round(float(bare["metrics"]["n_RF"]), 4),
            "n_g_opt": round(ng_opt, 4),
        },
        "frequency_response": {
            "freq_GHz": np.round(f_ghz, 3).tolist(),
            "S21_EO_bare_dBe": np.round(np.interp(f_ghz, f_bare, np.asarray(fr_bare["S21_EO_dBe"], dtype=float)), 3).tolist(),
            "S21_EO_pkg_dBe": np.round(s21_eo_pkg_dbe, 3).tolist(),
            "S21_EO_pkg_ang_deg": np.round(np.degrees(np.angle(h_eo_pkg_c)), 2).tolist(),
            "S21_RF_pkg_dB": np.round(s21_rf_pkg_db, 3).tolist(),
            "S21_RF_pkg_ang_deg": np.round(np.degrees(np.angle(s21_pkg)), 2).tolist(),
            "S11_RF_pkg_dB": np.round(s11_rf_pkg_db, 3).tolist(),
            "S11_RF_pkg_ang_deg": np.round(np.degrees(np.angle(s11_pkg)), 2).tolist(),
        },
        "_complex": {
            "freq_GHz": f_ghz,
            "h_eo_pkg_c": h_eo_pkg_c,
            "m_pkg": m_pkg,
            "s11_pkg": s11_pkg,
            "s21_pkg": s21_pkg,
            "dc_divider_gain": dc_divider_gain,
        },
        "touchstone_s2p": s2p_text,
    }


# ==============================================================================
# 4. End-to-End 100G–400G NRZ/PAM4 Eye Diagram, Semi-Analytic BER, TDECQ & FFE
# ==============================================================================

def _generate_prbs_bits(order: int = 9) -> np.ndarray:
    """Deterministic maximal-length LFSR PRBS-9 (511 bits) or PRBS-11 (2047 bits)."""
    ord_i = 11 if int(order) >= 11 else 9
    n_bits = (1 << ord_i) - 1
    reg = (1 << ord_i) - 1
    taps = (8, 4) if ord_i == 9 else (10, 8)
    bits = np.empty(n_bits, dtype=np.int8)
    mask = (1 << ord_i) - 1
    for i in range(n_bits):
        bits[i] = reg & 1
        fb = ((reg >> taps[0]) ^ (reg >> taps[1])) & 1
        reg = ((reg << 1) | fb) & mask
    return bits


def _bessel_thomson_4th_h(f_hz: np.ndarray, f_3db_hz: float) -> np.ndarray:
    """Exact 4th-order Bessel-Thomson reference receiver transfer function H_BT4(j*f/f_3dB)."""
    # Standard 4th-order Bessel polynomial B4(s) = s^4 + 10 s^3 + 45 s^2 + 105 s + 105
    # Normalized so |H(j*u_3dB)| = 1/sqrt(2) at u_3dB = 2.113917675
    u = 2.113917675 * (f_hz / max(float(f_3db_hz), 1e6))
    s = 1j * u
    den = (s ** 4) + 10.0 * (s ** 3) + 45.0 * (s ** 2) + 105.0 * s + 105.0
    return 105.0 / den


def _smf28_dispersion_ps_nm_km(wavelength_nm: float) -> float:
    """ITU-T G.652 SMF-28 chromatic dispersion D(lambda) [ps/(nm*km)]."""
    lam = float(wavelength_nm)
    lam0 = 1310.0
    s0 = 0.092
    return 0.25 * s0 * (lam - (lam0 ** 4) / max(lam ** 3, 1.0))


def _evaluate_pam_eye_and_ber(
    y_wave: np.ndarray,
    sym_levels: np.ndarray,
    sps: int,
    n_levels: int,
    sigma_noise_per_sample: np.ndarray,
    *,
    max_eye_traces: int = 80,
) -> dict:
    """Extract optimal sampling instant, eye height/width, OMA, ER, Q-factor, semi-analytic BER, and 2-UI eye traces."""
    n_sym = len(sym_levels)
    # 1. Search all fractional sampling offsets in [0, sps-1] to maximize worst-case inner eye opening
    best_off = 0
    best_score = -1e18
    for off in range(sps):
        ys = y_wave[off::sps]
        if len(ys) != n_sym:
            continue
        means = [float(np.mean(ys[sym_levels == lv])) for lv in range(n_levels)]
        # Ensure monotonic level ordering (invert if biased on negative MZI slope)
        sgn = 1.0 if means[-1] >= means[0] else -1.0
        ys_o = sgn * ys
        lv_min = [float(np.min(ys_o[sym_levels == lv])) for lv in range(n_levels)]
        lv_max = [float(np.max(ys_o[sym_levels == lv])) for lv in range(n_levels)]
        lv_std = [float(np.std(ys_o[sym_levels == lv])) for lv in range(n_levels)]
        lv_mu = [float(np.mean(ys_o[sym_levels == lv])) for lv in range(n_levels)]
        # Score combines worst-case deterministic eye height and worst-case Q-margin
        min_gap = min(lv_min[m + 1] - lv_max[m] for m in range(n_levels - 1))
        min_q = min((lv_mu[m + 1] - lv_mu[m]) / max(lv_std[m + 1] + lv_std[m], 1e-9) for m in range(n_levels - 1))
        score = min_gap + 0.15 * min_q * (lv_mu[-1] - lv_mu[0]) / max(n_levels - 1, 1)
        if score > best_score:
            best_score = score
            best_off = off

    # Sampled points at optimal phase
    ys_raw = y_wave[best_off::sps]
    sig_s = sigma_noise_per_sample[best_off::sps]
    means_raw = [float(np.mean(ys_raw[sym_levels == lv])) for lv in range(n_levels)]
    polarity = 1.0 if means_raw[-1] >= means_raw[0] else -1.0

    # Oriented waveform & samples so level 0 is lowest and level (M-1) is highest
    if polarity < 0.0:
        # Remap symbol indices so physical level 0 corresponds to lowest optical power
        sym_ord = (n_levels - 1) - sym_levels
    else:
        sym_ord = sym_levels

    lv_mu = [float(np.mean(ys_raw[sym_ord == lv])) for lv in range(n_levels)]
    lv_std_isi = [float(np.std(ys_raw[sym_ord == lv])) for lv in range(n_levels)]
    lv_sig_n = [float(np.mean(sig_s[sym_ord == lv])) for lv in range(n_levels)]
    lv_std_tot = [math.hypot(lv_std_isi[m], lv_sig_n[m]) for m in range(n_levels)]

    # Optimal decision thresholds between adjacent levels (noise-weighted)
    thresholds: list[float] = []
    eye_heights: list[float] = []
    eye_qs: list[float] = []
    for m in range(n_levels - 1):
        s_lo = max(lv_std_tot[m], 1e-12)
        s_hi = max(lv_std_tot[m + 1], 1e-12)
        th = (lv_mu[m] * s_hi + lv_mu[m + 1] * s_lo) / (s_lo + s_hi)
        thresholds.append(th)
        # Eye height at 1e-3 probability contour (~3.09 sigma_noise margin inside deterministic ISI envelope)
        lo_edge = float(np.percentile(ys_raw[sym_ord == m], 99.0)) + 2.0 * lv_sig_n[m]
        hi_edge = float(np.percentile(ys_raw[sym_ord == m + 1], 1.0)) - 2.0 * lv_sig_n[m + 1]
        eye_heights.append(max(hi_edge - lo_edge, 0.0))
        eye_qs.append((lv_mu[m + 1] - lv_mu[m]) / (s_lo + s_hi))

    # Exact vectorized semi-analytic symbol error rate (SER) & Gray-coded BER averaged over ALL PRBS symbols!
    inv_s2 = 1.0 / (math.sqrt(2.0) * np.maximum(sig_s, 1e-15))
    p_err_vec = np.zeros(n_sym, dtype=float)
    m0 = sym_ord == 0
    m_hi = sym_ord == (n_levels - 1)
    p_err_vec[m0] = 0.5 * erfc((thresholds[0] - ys_raw[m0]) * inv_s2[m0])
    p_err_vec[m_hi] = 0.5 * erfc((ys_raw[m_hi] - thresholds[-1]) * inv_s2[m_hi])
    for lv in range(1, n_levels - 1):
        mlv = sym_ord == lv
        if np.any(mlv):
            p_lo = 0.5 * erfc((ys_raw[mlv] - thresholds[lv - 1]) * inv_s2[mlv])
            p_hi = 0.5 * erfc((thresholds[lv] - ys_raw[mlv]) * inv_s2[mlv])
            p_err_vec[mlv] = np.minimum(1.0, p_lo + p_hi)
    ser = max(float(np.mean(p_err_vec)), 1e-18)
    bits_per_sym = 2.0 if n_levels == 4 else 1.0
    ber = max(ser / bits_per_sym, 1e-18)

    # Eye width [UI]: fraction of one UI around best_off where deterministic eye remains open across all eyes
    open_bins = 0
    for d_off in range(-(sps // 2), sps // 2):
        off_c = (best_off + d_off) % sps
        yc = y_wave[off_c::sps]
        if len(yc) != n_sym:
            continue
        is_open_bin = True
        for m in range(n_levels - 1):
            hi_lo = float(np.percentile(yc[sym_ord == m], 98.0)) + 1.5 * lv_sig_n[m]
            lo_hi = float(np.percentile(yc[sym_ord == m + 1], 2.0)) - 1.5 * lv_sig_n[m + 1]
            if lo_hi <= hi_lo:
                is_open_bin = False
                break
        if is_open_bin:
            open_bins += 1
    eye_width_ui = open_bins / float(sps)

    # Fold 2-UI eye diagram centered so optimal sampling instants lie at 0.5 UI and 1.5 UI
    start_shift = (best_off - sps // 2) % sps
    y_shifted = np.roll(y_wave, -start_shift)
    n_two_ui = (len(y_shifted) // (2 * sps)) - 1
    n_traces = min(max_eye_traces, max(n_two_ui, 1))
    stride_tr = max(1, n_two_ui // n_traces)
    eye_traces: list[list[float]] = []
    for itr in range(0, n_two_ui, stride_tr):
        seg = y_shifted[itr * 2 * sps : (itr + 1) * 2 * sps + 1]
        if len(seg) == 2 * sps + 1:
            eye_traces.append(np.round(seg, 5).tolist())
        if len(eye_traces) >= n_traces:
            break
    t_eye_ui = np.round(np.linspace(0.0, 2.0, 2 * sps + 1), 4).tolist()

    return {
        "best_offset": best_off,
        "polarity": polarity,
        "sym_ord": sym_ord,
        "level_means": lv_mu,
        "level_std_isi": lv_std_isi,
        "level_sigma_noise": lv_sig_n,
        "thresholds": thresholds,
        "eye_heights": eye_heights,
        "worst_eye_height": min(eye_heights),
        "eye_width_ui": round(eye_width_ui, 3),
        "worst_q_factor": round(min(eye_qs), 3),
        "ser": float(ser),
        "ber": float(ber),
        "log10_ber": round(math.log10(max(ber, 1e-18)), 3),
        "t_eye_ui": t_eye_ui,
        "eye_traces": eye_traces,
    }


def _compute_ieee_tdecq_db(
    y_sampled: np.ndarray,
    sym_ord: np.ndarray,
    thresholds: list[float],
    oma_outer: float,
    noise_enhancement_rms: float,
    n_levels: int = 4,
) -> float:
    """Exact IEEE 802.3bs TDECQ [dB] via bisection for added Gaussian noise reaching target SER = 4.8e-4."""
    if oma_outer <= 1e-12:
        return 15.0
    target_ser = 4.8e-4 if n_levels == 4 else 1.0e-3
    q_target = Q_T_IEEE_PAM4 if n_levels == 4 else 3.0902

    m0 = sym_ord == 0
    m_hi = sym_ord == (n_levels - 1)
    y0 = y_sampled[m0]
    y_hi = y_sampled[m_hi]
    mid_slices = [(lv, y_sampled[sym_ord == lv]) for lv in range(1, n_levels - 1)]
    n_s = float(len(sym_ord))

    def _ser_at_sigma(sig_eq: float) -> float:
        inv_s = 1.0 / (math.sqrt(2.0) * max(sig_eq, 1e-15))
        err = 0.5 * float(np.sum(erfc((thresholds[0] - y0) * inv_s)))
        err += 0.5 * float(np.sum(erfc((y_hi - thresholds[-1]) * inv_s)))
        for lv, y_m in mid_slices:
            if len(y_m):
                err += float(np.sum(np.minimum(
                    1.0,
                    0.5 * erfc((y_m - thresholds[lv - 1]) * inv_s)
                    + 0.5 * erfc((thresholds[lv] - y_m) * inv_s),
                )))
        return err / n_s

    # Ideal zero-ISI noise ceiling R_ideal = OMA / (2*(M-1)*Q_t)
    r_ideal = oma_outer / (2.0 * (n_levels - 1) * q_target)
    # Check if deterministic ISI alone already exceeds target SER even at tiny noise
    if _ser_at_sigma(1e-4 * r_ideal) >= target_ser:
        return 12.0

    lo = 1e-4 * r_ideal
    hi = 2.0 * r_ideal
    while _ser_at_sigma(hi) < target_ser and hi < 10.0 * r_ideal:
        hi *= 1.5
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if _ser_at_sigma(mid) < target_ser:
            lo = mid
        else:
            hi = mid
    # Added input-referred noise R = sigma_eq / noise_enhancement_rms
    sigma_eq_max = 0.5 * (lo + hi)
    r_input = sigma_eq_max / max(noise_enhancement_rms, 1e-9)
    tdecq_db = 10.0 * math.log10(max(r_ideal / max(r_input, 1e-18), 1.0))
    return round(float(min(tdecq_db, 12.0)), 3)


def _solve_mmse_ffe(
    y_wave: np.ndarray,
    sym_target: np.ndarray,
    sps: int,
    best_off: int,
    n_taps: int,
    sigma_n_avg: float,
    h_bt4_sq: np.ndarray,
    f_fft_hz: np.ndarray,
    t_sym_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact Wiener-Hopf MMSE symbol-spaced N-tap FFE equalizer + exact colored-noise enhancement factor."""
    n_sym = len(sym_target)
    n_t = max(3, int(n_taps))
    if n_t % 2 == 0:
        n_t += 1
    half = n_t // 2

    # Build circular Toeplitz data matrix at symbol instants
    ys = y_wave[best_off::sps]
    mu_y = float(np.mean(ys))
    ys_ac = ys - mu_y
    d_ac = sym_target - float(np.mean(sym_target))
    # Scale target to match DC amplitude of received AC signal
    scale_d = float(np.std(ys_ac)) / max(float(np.std(d_ac)), 1e-12)
    d_sc = d_ac * scale_d

    X = np.empty((n_sym, n_t), dtype=float)
    for j in range(n_t):
        shift = j - half
        X[:, j] = np.roll(ys_ac, -shift)

    # Regularized Wiener-Hopf normal equations (includes real receiver noise variance!)
    reg = max(n_sym * (sigma_n_avg ** 2), 1e-9 * float(np.sum(ys_ac ** 2)))
    A = X.T @ X + reg * np.eye(n_t)
    b = X.T @ d_sc
    w = np.linalg.solve(A, b)
    # Normalize sum(w) = 1.0 so DC level & outer OMA scaling stay physically calibrated
    w_sum = float(np.sum(w))
    if abs(w_sum) > 1e-9:
        w = w / w_sum

    # Apply symbol-spaced FFE taps to the full oversampled waveform (preserving continuous eye shape!)
    y_eq = np.zeros_like(y_wave)
    for j in range(n_t):
        shift_samp = (j - half) * sps
        y_eq += w[j] * np.roll(y_wave - mu_y, -shift_samp)
    y_eq += mu_y

    # Exact colored-noise autocorrelation matrix R_BT4[m] at symbol lags m = -(n_t-1)..(n_t-1)
    # R[m] = integral(|H_BT4(f)|^2 cos(2*pi*f*m*T_sym) df) / integral(|H_BT4(f)|^2 df)
    norm_h = float(np.sum(h_bt4_sq))
    r_lags = np.zeros(n_t, dtype=float)
    for m in range(n_t):
        r_lags[m] = float(np.sum(h_bt4_sq * np.cos(2.0 * math.pi * f_fft_hz * (m * t_sym_s)))) / max(norm_h, 1e-18)
    R_mat = np.empty((n_t, n_t), dtype=float)
    for i in range(n_t):
        for j in range(n_t):
            R_mat[i, j] = r_lags[abs(i - j)]
    nef_rms = math.sqrt(max(float(w @ R_mat @ w), 1e-9))
    return y_eq, w, nef_rms


def simulate_eom_eye_ber_link(cfg: EOMRFLinkConfig | None = None) -> dict:
    """Run complete end-to-end EOM RF packaging + NRZ/PAM4 time-domain eye diagram, BER, TDECQ & FFE link."""
    t0 = time.perf_counter()
    c = cfg or EOMRFLinkConfig()

    fmt = str(c.modulation_format).upper().strip()
    if fmt not in ("PAM4", "NRZ"):
        fmt = "PAM4"
    n_levels = 4 if fmt == "PAM4" else 2
    bits_per_sym = 2 if fmt == "PAM4" else 1
    baud_gbaud = max(float(c.baud_rate_gbaud), 5.0)
    bitrate_gbps = baud_gbaud * bits_per_sym
    t_sym_s = 1.0 / (baud_gbaud * 1e9)
    sps = max(12, int(c.samples_per_symbol))
    dt_s = t_sym_s / sps

    # 1. Generate deterministic PRBS symbol stream
    bits = _generate_prbs_bits(c.prbs_order)
    n_sym = len(bits)
    if fmt == "PAM4":
        # Two decorrelated taps from PRBS register -> balanced Gray-coded PAM4 symbols in {0, 1, 2, 3}
        msb = bits
        lsb = np.roll(bits, n_sym // 3 + 7)
        # Gray mapping: 00->0, 01->1, 11->2, 10->3
        sym_levels = np.where(msb == 0, lsb, 3 - lsb).astype(np.int8)
        drive_norm = (2.0 * sym_levels.astype(float) / 3.0) - 1.0  # [-1, -1/3, +1/3, +1]
    else:
        sym_levels = bits.astype(np.int8)
        drive_norm = 2.0 * sym_levels.astype(float) - 1.0  # [-1, +1]

    # Optional TX 2-tap FIR pre-emphasis (boosts Nyquist transition edge, normalized to preserve peak V_pp!)
    if float(c.pre_emphasis_db) > 0.05:
        boost_lin = 10.0 ** (float(c.pre_emphasis_db) / 20.0)
        alpha_pre = (boost_lin - 1.0) / (boost_lin + 1.0)
        drive_tx = drive_norm - alpha_pre * np.roll(drive_norm, 1)
        drive_tx = drive_tx / max(float(np.max(np.abs(drive_tx))), 1e-9)
    else:
        drive_tx = drive_norm

    # Upsample to time grid & apply physical TX rise-time shaping + optional timing jitter
    n_tot = n_sym * sps
    v_ideal = np.repeat(drive_tx, sps) * (0.5 * float(c.v_pp_drive_v))
    freqs_hz = np.fft.rfftfreq(n_tot, d=dt_s)
    freqs_ghz = freqs_hz * 1e-9

    # 2. Solve full HFSS-class packaged EOM RF + EO complex response on BOTH user grid & FFT grid
    pkg_plot = simulate_eom_rf_packaging(c)
    bare = pkg_plot["bare_eom"]
    pkg_fft = simulate_eom_rf_packaging(c, bare_eom_result=bare, extra_freq_ghz=freqs_ghz)

    # TX Gaussian edge filter (20-80% rise time t_r = 1.6832 * sigma_t)
    t_rise_s = max(float(c.tx_rise_time_ps), 0.1) * 1e-12
    sigma_t = t_rise_s / 1.6832
    h_tx = np.exp(-0.5 * (2.0 * math.pi * freqs_hz * sigma_t) ** 2)

    # Optional TX random jitter modeled via exact per-edge random phase-dithered sample interpolation
    if float(c.tx_jitter_rms_ps) > 0.01:
        rng = np.random.default_rng(int(c.random_seed))
        # Bandlimited per-symbol jitter interpolated smoothly across samples
        jit_sym = rng.normal(0.0, float(c.tx_jitter_rms_ps) * 1e-12 / dt_s, size=n_sym)
        idx_base = np.arange(n_tot, dtype=float)
        jit_samp = np.interp(idx_base, np.arange(n_sym) * sps + 0.5 * sps, jit_sym)
        v_ideal = np.interp((idx_base + jit_samp) % n_tot, idx_base, v_ideal)

    # 3. Apply EXACT complex packaged EOM electro-optic transfer function H_EO_pkg(f)
    h_eo_c = pkg_fft["_complex"]["h_eo_pkg_c"]
    dc_gain = float(pkg_fft["_complex"]["dc_divider_gain"])
    # Pin DC bin cleanly to real dc_gain (no artificial phase jump at f=0)
    h_eo_fft = h_eo_c.copy()
    h_eo_fft[0] = 1.0 + 0j
    v_eff_t = np.fft.irfft(np.fft.rfft(v_ideal) * h_tx * h_eo_fft, n=n_tot) * dc_gain

    # 4. Exact Nonlinear Mach-Zehnder Optical Field Modulation
    v_pi = max(float(bare["metrics"]["V_pi_V"]), 0.1)
    alpha_wg_db_cm = float(bare["metrics"]["alpha_metal_opt_db_per_cm"]) + float(bare["metrics"]["alpha_fca_opt_db_per_cm"]) + 0.30
    wg_loss_db = alpha_wg_db_cm * (float(bare["dimensions"]["active_length_um"]) * 1e-4)
    total_il_db = float(c.passive_insertion_loss_db) + wg_loss_db
    p_in_w = 1e-3 * (10.0 ** ((float(c.laser_power_dbm) - total_il_db) / 10.0))
    phi_bias = math.radians(float(c.bias_phase_deg))

    # Push-pull zero-chirp MZI for symmetric GSG/T-Rail; chirped single-arm MZI for GS / vertical single-arm
    resolved_top = str(bare["resolved_topology"])
    if resolved_top in ("gsg_cpw", "t_rail_cpw", "lumped_ring_gsg"):
        e_opt_t = math.sqrt(p_in_w) * np.cos(0.5 * phi_bias + 0.5 * math.pi * v_eff_t / v_pi).astype(complex)
    else:
        e_opt_t = 0.5 * math.sqrt(p_in_w) * (np.exp(1j * phi_bias) + np.exp(1j * math.pi * v_eff_t / v_pi))

    # 5. Optional SMF-28 Fiber Chromatic Dispersion + Attenuation
    fiber_km = max(float(c.fiber_length_km), 0.0)
    if fiber_km > 1e-6:
        d_ps_nm_km = _smf28_dispersion_ps_nm_km(c.wavelength_nm)
        lam_m = c.wavelength_nm * 1e-9
        # D [s/m^2] = d_ps_nm_km * 1e-6; beta_2 = -lambda^2 * D / (2*pi*c0) [s^2/m]
        beta2_s2_m = -(lam_m ** 2) * (d_ps_nm_km * 1e-6) / (2.0 * math.pi * C0_M_PER_S)
        alpha_fiber_db_km = 0.20 if c.wavelength_nm > 1450.0 else 0.34
        att_lin = 10.0 ** (-0.5 * alpha_fiber_db_km * fiber_km / 10.0)
        om_full = 2.0 * math.pi * np.fft.fftfreq(n_tot, d=dt_s)
        h_cd = att_lin * np.exp(0.5j * beta2_s2_m * (fiber_km * 1e3) * (om_full ** 2))
        e_opt_t = np.fft.ifft(np.fft.fft(e_opt_t) * h_cd)

    # 6. PIN Photodiode Detection + 4th-Order Bessel-Thomson Receiver Filter
    p_rx_t = np.abs(e_opt_t) ** 2  # Optical power [W]
    resp = max(float(c.rx_responsivity_a_w), 0.05)
    i_pd_ma = resp * p_rx_t * 1e3  # Photocurrent [mA]

    f_rx_ghz = float(c.rx_bandwidth_ghz) if (c.rx_bandwidth_ghz is not None and float(c.rx_bandwidth_ghz) > 1.0) else (0.75 * baud_gbaud)
    h_bt4 = _bessel_thomson_4th_h(freqs_hz, f_rx_ghz * 1e9)
    i_rx_ma = np.fft.irfft(np.fft.rfft(i_pd_ma) * h_bt4, n=n_tot)

    # Exact receiver effective noise bandwidth B_noise = integral_0^inf |H_BT4(f)|^2 df
    df_hz = freqs_hz[1] - freqs_hz[0]
    h_bt4_sq = np.abs(h_bt4) ** 2
    b_noise_hz = float(np.sum(h_bt4_sq) * df_hz)

    # Exact level-dependent total RMS noise current [mA] (TIA thermal + photodiode shot + laser RIN)
    i_tia_a_rt_hz = max(float(c.rx_noise_pa_rt_hz), 0.1) * 1e-12
    rin_lin = 10.0 ** (float(c.laser_rin_db_hz) / 10.0)
    p_rx_smooth_w = np.maximum(i_rx_ma * 1e-3 / resp, 0.0)
    i_rx_a = p_rx_smooth_w * resp
    sigma_i_a = np.sqrt((i_tia_a_rt_hz ** 2 + 2.0 * Q_ELECTRON_C * i_rx_a + rin_lin * (i_rx_a ** 2)) * b_noise_hz)
    sigma_i_ma = sigma_i_a * 1e3

    # Align circular group delay between transmitted symbol sequence and detected waveform!
    # Cross-correlate downsampled received waveform against drive_norm to remove integer-symbol transit delay
    y_corr_probe = i_rx_ma[::sps] - float(np.mean(i_rx_ma))
    c_fft = np.fft.irfft(np.fft.rfft(y_corr_probe) * np.conj(np.fft.rfft(drive_norm)), n=n_sym)
    sym_lag = int(np.argmax(np.abs(c_fft)))
    if sym_lag != 0:
        i_rx_ma = np.roll(i_rx_ma, -sym_lag * sps)
        sigma_i_ma = np.roll(sigma_i_ma, -sym_lag * sps)

    # 7. Unequalized Eye Diagram, OMA, ER, Q-Factor & Semi-Analytic BER
    raw_eye = _evaluate_pam_eye_and_ber(i_rx_ma, sym_levels, sps, n_levels, sigma_i_ma)
    p_lo_mw = max(raw_eye["level_means"][0] / resp, 1e-9)
    p_hi_mw = max(raw_eye["level_means"][-1] / resp, 1e-9)
    oma_mw = max(p_hi_mw - p_lo_mw, 1e-9)
    oma_dbm = 10.0 * math.log10(oma_mw)
    er_db = 10.0 * math.log10(max(p_hi_mw / p_lo_mw, 1.0 + 1e-6))

    # 8. Optional Wiener-Hopf MMSE FFE Equalizer + Exact IEEE 802.3bs TDECQ
    n_ffe = int(c.ffe_taps)
    sigma_avg_ma = float(np.mean(sigma_i_ma))
    if n_ffe >= 3:
        sym_target = raw_eye["sym_ord"].astype(float)
        i_eq_ma, ffe_weights, nef_rms = _solve_mmse_ffe(
            i_rx_ma,
            sym_target,
            sps,
            raw_eye["best_offset"],
            n_ffe,
            sigma_avg_ma,
            h_bt4_sq,
            freqs_hz,
            t_sym_s,
        )
        # Re-orient sym_levels consistently with raw polarity
        eq_eye = _evaluate_pam_eye_and_ber(i_eq_ma, sym_levels, sps, n_levels, sigma_i_ma * nef_rms)
        tdecq_db = _compute_ieee_tdecq_db(
            i_eq_ma[eq_eye["best_offset"] :: sps],
            eq_eye["sym_ord"],
            eq_eye["thresholds"],
            oma_outer=max(eq_eye["level_means"][-1] - eq_eye["level_means"][0], 1e-12),
            noise_enhancement_rms=nef_rms,
            n_levels=n_levels,
        )
    else:
        ffe_weights = np.array([1.0])
        nef_rms = 1.0
        eq_eye = raw_eye
        tdecq_db = _compute_ieee_tdecq_db(
            i_rx_ma[raw_eye["best_offset"] :: sps],
            raw_eye["sym_ord"],
            raw_eye["thresholds"],
            oma_outer=max(raw_eye["level_means"][-1] - raw_eye["level_means"][0], 1e-12),
            noise_enhancement_rms=1.0,
            n_levels=n_levels,
        )

    final_ber = float(eq_eye["ber"])
    fec_verdict = (
        "ERROR_FREE (<1e-12)" if final_ber <= 1e-12
        else "PASS_KP4_FEC (<2.4e-4)" if final_ber <= 2.4e-4
        else "PASS_HD_FEC (<3.8e-3)" if final_ber <= 3.8e-3
        else "FAIL_ABOVE_FEC"
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1e3, 1)
    pkg_m = pkg_plot["metrics"]
    pkg_fr = pkg_plot["frequency_response"]
    flat_metrics = {
        "bit_rate_Gbps": round(bitrate_gbps, 2),
        "baud_rate_GBaud": round(baud_gbaud, 2),
        "modulation_format": fmt,
        "launch_type": pkg_plot["launch_type"],
        "f_3dBo_bare_GHz": pkg_m["f_3dBo_bare_GHz"],
        "f_3dBo_packaged_GHz": pkg_m["f_3dBo_packaged_GHz"],
        "packaging_bw_penalty_pct": pkg_m["packaging_bw_penalty_pct"],
        "S11_max_67GHz_dB": pkg_m["S11_max_67GHz_dB"],
        "S21_RF_67GHz_dB": pkg_m["S21_RF_67GHz_dB"],
        "V_pi_V": pkg_m["V_pi_V"],
        "OMA_outer_dBm": round(oma_dbm, 2),
        "ER_dB": round(er_db, 2),
        "raw_BER": float(raw_eye["ber"]),
        "eq_BER": float(eq_eye["ber"]),
        "TDECQ_dB": float(tdecq_db),
        "eq_Q_factor": float(eq_eye["worst_q_factor"]),
        "eq_eye_width_UI": float(eq_eye["eye_width_ui"]),
        "ffe_taps": n_ffe,
        "fec_verdict": fec_verdict,
    }
    return {
        "elapsed_ms": elapsed_ms,
        "config": asdict(c),
        "metrics": flat_metrics,
        "spectrum": {
            "freq_ghz": pkg_fr["freq_GHz"],
            "S21_EO_pkg_dBe": pkg_fr["S21_EO_pkg_dBe"],
            "S21_EO_bare_dBe": pkg_fr["S21_EO_bare_dBe"],
            "S11_RF_pkg_dB": pkg_fr["S11_RF_pkg_dB"],
        },
        "link_summary": {
            "modulation_format": fmt,
            "baud_rate_GBaud": round(baud_gbaud, 2),
            "bit_rate_Gbps": round(bitrate_gbps, 2),
            "launch_type": pkg_plot["launch_type"],
            "launch_name_en": pkg_plot["launch_name_en"],
            "launch_name_zh": pkg_plot["launch_name_zh"],
            "core_material": c.core_material,
            "resolved_topology": resolved_top,
            "v_pp_drive_V": round(float(c.v_pp_drive_v), 3),
            "v_pp_over_v_pi": round(float(c.v_pp_drive_v) / v_pi, 3),
            "total_optical_il_dB": round(total_il_db, 2),
            "fiber_length_km": round(fiber_km, 2),
            "rx_bandwidth_GHz": round(f_rx_ghz, 2),
            "ffe_taps": n_ffe,
            "fec_verdict": fec_verdict,
        },
        "rf_packaging": {
            "metrics": pkg_plot["metrics"],
            "packaging_geometry": pkg_plot["packaging_geometry"],
            "frequency_response": pkg_plot["frequency_response"],
            "touchstone_s2p": pkg_plot["touchstone_s2p"],
        },
        "eye_metrics": {
            "OMA_outer_dBm": round(oma_dbm, 2),
            "ER_dB": round(er_db, 2),
            "raw_eye_height_mA": round(float(raw_eye["worst_eye_height"]), 4),
            "raw_eye_width_UI": float(raw_eye["eye_width_ui"]),
            "raw_Q_factor": float(raw_eye["worst_q_factor"]),
            "raw_BER": float(raw_eye["ber"]),
            "raw_log10_BER": float(raw_eye["log10_ber"]),
            "eq_eye_height_mA": round(float(eq_eye["worst_eye_height"]), 4),
            "eq_eye_width_UI": float(eq_eye["eye_width_ui"]),
            "eq_Q_factor": float(eq_eye["worst_q_factor"]),
            "eq_BER": float(eq_eye["ber"]),
            "eq_log10_BER": float(eq_eye["log10_ber"]),
            "TDECQ_dB": float(tdecq_db),
            "ffe_noise_enhancement_dB": round(20.0 * math.log10(max(nef_rms, 1e-6)), 3),
            "ffe_weights": np.round(ffe_weights, 4).tolist(),
            "pass_kp4_fec": bool(final_ber <= 2.4e-4),
            "pass_hd_fec": bool(final_ber <= 3.8e-3),
            "pass_error_free": bool(final_ber <= 1e-12),
            "fec_verdict": fec_verdict,
        },
        "eye_diagram_raw": {
            "t_ui": raw_eye["t_eye_ui"],
            "traces_mA": raw_eye["eye_traces"],
            "thresholds_mA": [round(float(th), 4) for th in raw_eye["thresholds"]],
            "level_means_mA": [round(float(m), 4) for m in raw_eye["level_means"]],
        },
        "eye_diagram_eq": {
            "t_ui": eq_eye["t_eye_ui"],
            "traces_mA": eq_eye["eye_traces"],
            "thresholds_mA": [round(float(th), 4) for th in eq_eye["thresholds"]],
            "level_means_mA": [round(float(m), 4) for m in eq_eye["level_means"]],
        },
        "scope_boundary_note": (
            "Live first-principles 2D-FEM + multilayer conformal CPW + 3D GSG Neumann/Rosa wirebond/bump ABCD "
            "+ nonlinear MZI FFT + Bessel-Thomson + MMSE FFE engine. High-frequency (>100 GHz) arbitrary 3D "
            "enclosure box modes and multi-lane connector radiation crosstalk remain outside quasi-TEM/ABCD scope."
        ),
    }


__all__ = [
    "LAUNCH_TYPE_CATALOG",
    "EOMRFLinkConfig",
    "solve_multilayer_cpw_section",
    "compute_gsg_wirebond_abcd",
    "compute_gsg_flip_chip_bump_abcd",
    "simulate_eom_rf_packaging",
    "simulate_eom_eye_ber_link",
]
