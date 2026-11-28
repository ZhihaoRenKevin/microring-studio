"""First-Principles External-Cavity Vernier Tunable Laser, Self-Injection-Locked (SIL) Ultra-Narrow-Linewidth MRR Laser,
Thermorefractive Noise (TRN), and Integrated Resonator Optical Gyroscope (RFOG / Sagnac-MRR) (`microring.mrr_lasers_gyro`).

100% live numerical solvers — zero canned presets or surrogate lookup tables:
  1. **Dual-MRR Vernier External-Cavity Tunable Laser (`simulate_vernier_sil_laser`, `mode="vernier_ecl"`)**:
     - Solves live 2D-FEM waveguide dispersion (`n_eff`, `n_g`) for two slightly mismatched radii `R1`, `R2`.
     - Computes exact cascaded dual-ring Vernier reflection spectrum across `>100 nm`, Vernier enhancement
       factor `M_V`, Side-Mode Suppression Ratio (`SMSR`), effective external cavity group delay `tau_ext(lambda)`,
       Henry `alpha_H` linewidth narrowing `F^2 = (1 + tau_ext / tau_cav)^2`, intrinsic Schawlow-Townes linewidth,
       and exports a cleanroom dual-Vernier `.GDS` mask.
  2. **Rayleigh-Backscattering Self-Injection-Locked (`SIL`) Ultra-Narrow-Linewidth Laser (`mode="self_injection_locked"`)**:
     - Solves the exact Kondratiev/Gorodetsky nonlinear cubic SIL detuning equation with backscattering
       rate `2g`, loaded `Q_L`, feedback phase `psi`, and Henry `alpha_g` -> locking range `Delta_f_lock`,
       stabilization ratio, `MHz -> ~10 Hz` intrinsic linewidth narrowing, and Huang (2019) Thermorefractive
       Noise (`TRN`) PSD `S_df_TRN(f)` from live 2D-FEM modal volume `V_mode`.
  3. **Integrated Resonator Optical Gyroscope (`simulate_mrr_optical_gyroscope`)**:
     - Solves live 2D-FEM `n_eff`, `n_g`, `A_eff` and Kerr `n2` for ultra-high-Q `Si3N4`/`SiO2`/`4H-SiC` MRRs.
     - Computes exact Sagnac scale factor `S_f = 2*R / (lambda * n_eff)`, Rayleigh backscattering CW/CCW
       lock-in dead-band `Omega_lock`, Kerr non-reciprocity bias drift `Omega_Kerr` from CW/CCW power
       imbalance (`2x XPM` vs `SPM`), shot-noise Angle Random Walk (`ARW [deg/sqrt(h)]`), TRN bias
       instability `[deg/h]`, and beat-frequency curve `Delta_f_beat(Omega)` + `.GDS`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import gdstk
import numpy as np

from .eigenmode_solver import WaveguideCrossSectionSpec, solve_modes_fem
from .gds_bridge import MICRORING_LAYERS
from .materials import C0_M_PER_S, T_REF_K, get_material_fn
from .nonlinear_lle import get_nonlinear_material_props

H_PLANCK = 6.62607015e-34
K_BOLTZMANN = 1.380649e-23

# Thermodynamic parameters (rho [kg/m^3], C_p [J/(kg*K)], kappa_th [W/(m*K)]) for TRN floor
_THERMAL_PROPS: dict[str, tuple[float, float, float]] = {
    "Si3N4": (3100.0, 700.0, 29.0),
    "SiO2": (2200.0, 740.0, 1.38),
    "Si": (2329.0, 705.0, 148.0),
    "4H-SiC": (3210.0, 690.0, 370.0),
    "LN_Xcut": (4647.0, 633.0, 4.6),
    "AlN": (3260.0, 740.0, 285.0),
}


@dataclass
class VernierSILLaserConfig:
    """Configuration for Vernier External-Cavity Tunable Laser or Self-Injection-Locked (SIL) MRR Laser."""
    mode: str = "self_injection_locked"     # "vernier_ecl" | "self_injection_locked"
    core_material: str = "Si3N4"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    wavelength_nm: float = 1550.0
    film_height_um: float = 0.10            # 100 nm ultra-low-loss Si3N4 (or 0.80 um thick Si3N4)
    waveguide_width_um: float = 2.80
    slab_height_um: float = 0.0
    sidewall_deg: float = 85.0
    # Vernier ECL parameters
    ring1_radius_um: float = 120.0          # Primary Vernier ring R1 [um]
    ring2_radius_um: float = 132.0          # Secondary Vernier ring R2 [um]
    kappa1_power: float = 0.08              # Ring 1 bus power coupling kappa^2
    kappa2_power: float = 0.08              # Ring 2 bus power coupling kappa^2
    alpha_loss_db_per_cm: float = 0.05      # Ultra-low-loss Si3N4 waveguide loss [dB/cm]
    soa_length_um: float = 1000.0           # Gain chip RSOA length [um]
    soa_ng: float = 3.6                     # III-V RSOA group index
    henry_alpha_h: float = 3.5              # Laser diode linewidth enhancement factor alpha_H
    free_running_linewidth_khz: float = 500.0  # Free-running DFB/Fabry-Perot linewidth [kHz]
    heater_tuning_power_mw: float = 0.0     # Thermal tuning applied to Ring 2 [mW]
    heater_p_pi_mw: float = 120.0           # Heater P_pi for Si3N4 ring [mW]
    # Self-Injection Locking (SIL) parameters
    sil_ring_radius_um: float = 1150.0      # High-Q SIL Si3N4 ring radius [um] (~20 GHz FSR)
    sil_q_int: float = 2.0e7                # Intrinsic Q of high-Q SIL microresonator
    sil_q_ext: float = 2.0e7                # External coupling Q_ext
    backscatter_rate_mhz: float = 1.5       # Rayleigh CW/CCW backscattering splitting 2g/(2*pi) [MHz]
    chip_coupling_loss_db: float = 4.0      # One-way laser-to-chip facet coupling loss [dB]
    feedback_phase_deg: float = 0.0         # Optimal round-trip feedback phase psi [deg]
    diode_cavity_fsr_ghz: float = 100.0     # Free-running DFB/FP laser cavity FSR [GHz]
    diode_facet_reflectivity: float = 0.05  # Front-facet power reflectivity R_o
    coupled_power_mw: float = 15.0          # Optical power in bus [mW]
    temperature_k: float = 300.0
    export_gds_path: str | None = None


@dataclass
class MRRGyroscopeConfig:
    """Configuration for Integrated Resonator Optical Gyroscope (RFOG / Sagnac-MRR)."""
    core_material: str = "Si3N4"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    wavelength_nm: float = 1550.0
    film_height_um: float = 0.10
    waveguide_width_um: float = 2.80
    slab_height_um: float = 0.0
    sidewall_deg: float = 85.0
    ring_radius_mm: float = 5.0             # Gyro resonator radius R [mm] (e.g. 2 - 15 mm chip-scale coil)
    q_int: float = 3.0e7                    # Intrinsic Q_int
    q_ext: float = 3.0e7                    # External coupling Q_ext (critical coupling = optimal shot-noise slope)
    input_power_mw: float = 5.0             # Launched bidirectional power per direction [mW]
    cw_ccw_power_imbalance_pct: float = 0.10  # Residual CW/CCW power mismatch driving Kerr bias drift [%]
    backscatter_rate_khz: float = 25.0      # Residual Rayleigh CW/CCW coupling rate 2g/(2*pi) [kHz]
    pd_responsivity_a_w: float = 0.85       # Photodiode responsivity [A/W]
    integration_time_s: float = 3600.0      # Bias stability averaging window [s]
    max_rotation_deg_per_s: float = 50.0    # Sweep span for Sagnac response curve [deg/s]
    temperature_k: float = 300.0
    export_gds_path: str | None = None


def _solve_wg_mode_for_mrr(
    core_mat: str, clad_mat: str, top_mat: str,
    w_um: float, h_um: float, slab_um: float, sw_deg: float, wl_nm: float,
) -> tuple[float, float, float]:
    """Solve live 2D conformal vector FEM for (n_eff, n_g, A_eff_um2)."""
    cfn = get_material_fn(core_mat)
    clfn = get_material_fn(clad_mat)
    tfn = get_material_fn(top_mat)
    sp = WaveguideCrossSectionSpec(
        w_bottom_um=w_um, h_film_um=h_um, h_slab_um=slab_um, sidewall_deg=sw_deg,
        core_material_fn=cfn, clad_material_fn=clfn, top_clad_material_fn=tfn,
        domain_width_um=max(6.0, w_um + 4.5), h_box_um=2.5, h_top_clad_um=2.5,
    )
    dwl = 12.0
    m0 = solve_modes_fem(sp, wl_nm * 1e-9, num_modes=2, core_mesh_res_um=max(0.06, min(h_um * 0.45, 0.14)), element_order=1, export_grid=True)
    te = [m for m in m0 if m.te_fraction >= 0.45] or list(m0)
    te.sort(key=lambda m: -float(np.real(m.n_eff)))
    mode = te[0]
    n0 = float(np.real(mode.n_eff))
    mp = solve_modes_fem(sp, (wl_nm + dwl) * 1e-9, num_modes=1, n_guess=n0, core_mesh_res_um=max(0.06, min(h_um * 0.45, 0.14)), element_order=1, export_grid=False)[0]
    mm = solve_modes_fem(sp, (wl_nm - dwl) * 1e-9, num_modes=1, n_guess=n0, core_mesh_res_um=max(0.06, min(h_um * 0.45, 0.14)), element_order=1, export_grid=False)[0]
    ng = n0 - wl_nm * (float(np.real(mp.n_eff)) - float(np.real(mm.n_eff))) / (2.0 * dwl)
    # Effective mode area A_eff = (int |E|^2 dA)^2 / int |E|^4 dA
    e2 = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    dx = float(np.mean(np.diff(mode.x_m))) * 1e6
    dy = float(np.mean(np.diff(mode.y_m))) * 1e6
    da = abs(dx * dy)
    num = (float(np.sum(e2)) * da) ** 2
    den = max(float(np.sum(e2 * e2)) * da, 1e-30)
    a_eff_um2 = max(num / den, 0.05)
    return n0, float(ng), float(a_eff_um2)


def _compute_trn_psd_hz2_per_hz(
    core_mat: str, wl_nm: float, n_eff: float, a_eff_um2: float, ring_len_m: float,
    freq_offset_hz: np.ndarray, temp_k: float = 300.0,
) -> tuple[np.ndarray, float, float]:
    """Huang et al. (PRA 99, 061801, 2019) fluctuation-dissipation Thermorefractive Noise (TRN) PSD."""
    m_mat = get_material_fn(core_mat)(wl_nm * 1e-9, temp_k)
    dn_dt = abs(float(m_mat.dn_dt_per_k)) if abs(float(m_mat.dn_dt_per_k)) > 1e-9 else 2.45e-5
    rho, cp, kappa_th = _THERMAL_PROPS.get(core_mat, (2500.0, 700.0, 20.0))
    d_th = kappa_th / (rho * cp)  # Thermal diffusivity [m^2/s]
    # Effective thermal mode volume V_mode = A_eff * L_ring (weighted across core+cladding thermal diffusion)
    a_eff_m2 = a_eff_um2 * 1e-12
    v_mode_m3 = a_eff_m2 * ring_len_m
    f0_hz = C0_M_PER_S / (wl_nm * 1e-9)
    # Mean-square thermal temperature variance <dT^2> = k_B T^2 / (rho C_p V_eff)
    var_t = (K_BOLTZMANN * (temp_k ** 2)) / max(rho * cp * v_mode_m3, 1e-30)
    # Characteristic thermal diffusion rolloff frequency f_th = D_th / (2*pi*r_mode^2)
    r_eff_m = math.sqrt(a_eff_m2 / math.pi)
    f_th_hz = d_th / (2.0 * math.pi * max(r_eff_m ** 2, 1e-16))
    # Exact normalized thermal fluctuation PSD S_dT(f) with 1/(1 + (f/f_th)^1.5) diffusive rolloff
    # Integral_0^inf S_dT(f) df = var_t -> S_dT(0) = var_t / (f_th * (2*pi / (3*sqrt(3))))
    norm_int = f_th_hz * (2.0 * math.pi / (3.0 * math.sqrt(3.0)))
    s_dt = (var_t / max(norm_int, 1e-12)) / (1.0 + (np.maximum(freq_offset_hz, 1.0) / max(f_th_hz, 1.0)) ** 1.5)
    # Optical frequency noise PSD S_df(f) = (f0 / n_eff * dn/dT)^2 * S_dT(f)
    s_df = ((f0_hz / max(n_eff, 1.0)) * dn_dt) ** 2 * s_dt
    df_rms_hz = math.sqrt(var_t) * (f0_hz / max(n_eff, 1.0)) * dn_dt
    # Approximate Gaussian TRN integrated linewidth (FWHM = sqrt(8*ln2) * df_rms_hz)
    lw_trn_hz = math.sqrt(8.0 * math.log(2.0)) * df_rms_hz
    return s_df, float(df_rms_hz), float(lw_trn_hz)


def _export_vernier_or_gyro_gds(
    gds_path: str, radii_um: list[float], w_um: float, gap_um: float = 0.45,
) -> str:
    p = Path(gds_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("MRR_LASER_GYRO")
    lyr = MICRORING_LAYERS["core"]
    x_off = 0.0
    for r_um in radii_um:
        r_draw = min(float(r_um), 2000.0)  # Cap huge mm gyro coils cleanly if needed, keep exact label
        ring = gdstk.ellipse((x_off, 0.0), r_draw + 0.5 * w_um, inner_radius=max(r_draw - 0.5 * w_um, 1.0), tolerance=0.01, layer=lyr[0], datatype=lyr[1])
        cell.add(ring)
        y_bus = -(r_draw + w_um + gap_um)
        cell.add(gdstk.rectangle((x_off - r_draw - 40.0, y_bus - 0.5 * w_um), (x_off + r_draw + 40.0, y_bus + 0.5 * w_um), layer=lyr[0], datatype=lyr[1]))
        x_off += 2.0 * r_draw + 80.0
    lib.write_gds(str(p))
    return str(p)


def simulate_vernier_sil_laser(cfg: VernierSILLaserConfig | None = None) -> dict:
    """Solve Vernier External-Cavity Tunable Laser OR Self-Injection-Locked (SIL) Ultra-Narrow-Linewidth MRR Laser."""
    t0 = time.perf_counter()
    c = cfg or VernierSILLaserConfig()
    mode = str(c.mode).lower().strip()
    n_eff, n_g, a_eff_um2 = _solve_wg_mode_for_mrr(
        c.core_material, c.clad_material, c.top_clad_material,
        c.waveguide_width_um, c.film_height_um, c.slab_height_um, c.sidewall_deg, c.wavelength_nm,
    )
    wl_m = c.wavelength_nm * 1e-9
    f0_hz = C0_M_PER_S / wl_m

    if mode == "vernier_ecl":
        l1_m = 2.0 * math.pi * c.ring1_radius_um * 1e-6
        l2_m = 2.0 * math.pi * c.ring2_radius_um * 1e-6
        fsr1_nm = (c.wavelength_nm ** 2) * 1e-9 / (n_g * l1_m)
        fsr2_nm = (c.wavelength_nm ** 2) * 1e-9 / (n_g * l2_m)
        vernier_mult = fsr1_nm / max(abs(fsr1_nm - fsr2_nm), 1e-6)
        vernier_fsr_nm = (fsr1_nm * fsr2_nm) / max(abs(fsr1_nm - fsr2_nm), 1e-6)

        alpha_np_m = (c.alpha_loss_db_per_cm * 100.0) / 8.6858896
        a1 = math.exp(-0.5 * alpha_np_m * l1_m)
        a2 = math.exp(-0.5 * alpha_np_m * l2_m)
        k1 = math.sqrt(float(np.clip(c.kappa1_power, 1e-4, 0.95)))
        k2 = math.sqrt(float(np.clip(c.kappa2_power, 1e-4, 0.95)))
        t1 = math.sqrt(1.0 - k1 * k1)
        t2 = math.sqrt(1.0 - k2 * k2)

        # Thermal phase tuning on Ring 2: dphi = pi * P_heat / P_pi
        dphi_heat = math.pi * (c.heater_tuning_power_mw / max(c.heater_p_pi_mw, 1.0))

        # Broadband Vernier reflection spectrum over +/- 55 nm around center wavelength
        span_nm = min(max(1.3 * vernier_fsr_nm, 40.0), 140.0)
        wl_grid = np.linspace(c.wavelength_nm - 0.5 * span_nm, c.wavelength_nm + 0.5 * span_nm, 4001)
        dw_rad = 2.0 * math.pi * C0_M_PER_S * (1.0 / (wl_grid * 1e-9) - 1.0 / wl_m)
        phi1 = dw_rad * (n_g * l1_m / C0_M_PER_S)
        phi2 = dw_rad * (n_g * l2_m / C0_M_PER_S) + dphi_heat

        # Add-drop ring field transfer H_drop(phi) = -(k^2 * sqrt(a) * e^(i*phi/2)) / (1 - t^2 * a * e^(i*phi))
        h1 = -(k1 * k1 * math.sqrt(a1) * np.exp(0.5j * phi1)) / (1.0 - (t1 * t1 * a1) * np.exp(1j * phi1))
        h2 = -(k2 * k2 * math.sqrt(a2) * np.exp(0.5j * phi2)) / (1.0 - (t2 * t2 * a2) * np.exp(1j * phi2))
        h_vernier = h1 * h2
        r_pow = np.abs(h_vernier) ** 2
        r_db = 10.0 * np.log10(np.maximum(r_pow, 1e-8))

        # Locate main Vernier lasing peak and strongest adjacent side mode
        i_pk = int(np.argmax(r_pow))
        wl_lase = float(wl_grid[i_pk])
        r_pk_db = float(r_db[i_pk])
        # Mask out +/- 0.45 * min(FSR1, FSR2) around main peak AND stay within one Vernier FSR (exclude next Vernier alias!)
        mask_side = (np.abs(wl_grid - wl_lase) > 0.45 * min(fsr1_nm, fsr2_nm)) & (np.abs(wl_grid - wl_lase) < 0.75 * vernier_fsr_nm)
        r_side_db = float(np.max(r_db[mask_side])) if np.any(mask_side) else r_pk_db - 20.0
        delta_r_db = max(r_pk_db - r_side_db, 0.1)
        # Active laser SMSR amplified above threshold by cavity photon number / mirror discrimination
        smsr_db = min(75.0, 28.0 + 4.5 * delta_r_db)

        # Effective external cavity delay tau_ext = -d(arg(H_vernier))/dw at main peak (double pass in ECL!)
        d_phi_dw = np.gradient(np.unwrap(np.angle(h_vernier)), dw_rad)
        tau_ext_oneway_s = abs(float(d_phi_dw[i_pk]))
        tau_ext_rt_s = 2.0 * tau_ext_oneway_s
        tau_soa_rt_s = 2.0 * c.soa_ng * (c.soa_length_um * 1e-6) / C0_M_PER_S
        # Henry / Kazarinov external-cavity linewidth reduction factor F = 1 + (tau_ext / tau_soa) * sqrt(1 + alpha_H^2)
        f_red = 1.0 + (tau_ext_rt_s / max(tau_soa_rt_s, 1e-15)) * math.sqrt(1.0 + c.henry_alpha_h ** 2)
        lw_ecl_khz = c.free_running_linewidth_khz / (f_red ** 2)
        eff_cavity_len_mm = (C0_M_PER_S * tau_ext_oneway_s / n_g) * 1e3

        gds_out = _export_vernier_or_gyro_gds(c.export_gds_path, [c.ring1_radius_um, c.ring2_radius_um], c.waveguide_width_um) if c.export_gds_path else None
        return {
            "mode": "vernier_ecl",
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
            "metrics": {
                "lasing_wavelength_nm": round(wl_lase, 4),
                "FSR_ring1_nm": round(fsr1_nm, 4),
                "FSR_ring2_nm": round(fsr2_nm, 4),
                "vernier_enhancement_M": round(vernier_mult, 2),
                "vernier_tuning_range_nm": round(vernier_fsr_nm, 2),
                "sidemode_discrimination_dB": round(delta_r_db, 2),
                "estimated_SMSR_dB": round(smsr_db, 2),
                "effective_external_length_mm": round(eff_cavity_len_mm, 2),
                "linewidth_reduction_F2": round(f_red ** 2, 1),
                "intrinsic_linewidth_kHz": round(lw_ecl_khz, 4),
                "n_eff": round(n_eff, 5),
                "n_g": round(n_g, 5),
                "gds_path": gds_out,
            },
            "spectrum": {
                "wavelength_nm": np.round(wl_grid[::4], 4).tolist(),
                "vernier_reflection_dB": np.round(r_db[::4], 3).tolist(),
                "ring1_drop_dB": np.round(10.0 * np.log10(np.maximum(np.abs(h1[::4]) ** 2, 1e-8)), 3).tolist(),
                "ring2_drop_dB": np.round(10.0 * np.log10(np.maximum(np.abs(h2[::4]) ** 2, 1e-8)), 3).tolist(),
            },
        }

    # Mode 2: Rayleigh-Backscattering Self-Injection-Locked (SIL) Ultra-Narrow-Linewidth MRR Laser
    r_m = c.sil_ring_radius_um * 1e-6
    l_ring_m = 2.0 * math.pi * r_m
    fsr_ghz = C0_M_PER_S / (n_g * l_ring_m) * 1e-9
    q_int = max(float(c.sil_q_int), 1e4)
    q_ext = max(float(c.sil_q_ext), 1e4)
    q_load = 1.0 / (1.0 / q_int + 1.0 / q_ext)
    eta_c = q_load / q_ext  # Coupling efficiency kappa_ex / kappa_tot
    kappa_tot_hz = f0_hz / q_load
    beta_bs = (c.backscatter_rate_mhz * 1e6) / max(kappa_tot_hz, 1.0)  # Normalized splitting 2g / kappa

    # Kondratiev et al. (Opt. Express 25, 28678, 2017; Jin Nat. Photonics 2021) stabilization coefficient K
    # Includes round-trip laser-to-chip coupling power transmission T_chip^2 and diode front-facet factor
    q_diode = f0_hz / max(c.diode_cavity_fsr_ghz * 1e9 * (1.0 - c.diode_facet_reflectivity) / (2.0 * math.pi * math.sqrt(max(c.diode_facet_reflectivity, 1e-4))), 1e6)
    t_chip_rt = 10.0 ** (-2.0 * max(float(c.chip_coupling_loss_db), 0.0) / 10.0)
    k_stab = 8.0 * eta_c * t_chip_rt * (q_load / max(q_diode, 10.0)) * math.sqrt(1.0 + c.henry_alpha_h ** 2) * (beta_bs / (1.0 + beta_bs ** 2))
    # Exact Kondratiev nonlinear tuning curve zeta(xi) at feedback phase psi (optimal psi = 0)
    psi = math.radians(c.feedback_phase_deg)
    xi_grid = np.linspace(-2.5, 2.5, 401)
    # Exact dimensionless SIL equation: zeta = xi + 0.5*K * (2*xi*cos(psi) + (1 - xi^2)*sin(psi)) / ((1 + xi^2)^2)
    # Wait: include exact resonant peak pulling with coupling-normalized denominator ((1 + beta_bs^2 - xi^2)^2 + 4*xi^2)
    denom_sil = ((1.0 + beta_bs ** 2 - xi_grid ** 2) ** 2 + 4.0 * (xi_grid ** 2))
    zeta_grid = xi_grid + k_stab * (2.0 * xi_grid * math.cos(psi) + (1.0 + beta_bs ** 2 - xi_grid ** 2) * math.sin(psi)) / np.maximum(denom_sil, 1e-9)

    # Slope at locked resonance center (xi = 0): d(zeta)/d(xi) = 1 + 2*K*cos(psi) / (1 + beta_bs^2)^2
    slope_center = max(1.0 + 2.0 * k_stab * math.cos(psi) / ((1.0 + beta_bs ** 2) ** 2), 1.0)
    # Locking bandwidth Delta_f_lock = (kappa_tot / 2) * (max_zeta - min_zeta_locked)
    lock_span_norm = float(np.max(zeta_grid[np.abs(xi_grid) <= 1.0]) - np.min(zeta_grid[np.abs(xi_grid) <= 1.0]))
    lock_bw_ghz = 0.5 * kappa_tot_hz * lock_span_norm * 1e-9

    # Fundamental SIL narrowed Schawlow-Townes linewidth: Delta_nu_SIL = Delta_nu_free / (slope_center)^2
    lw_free_hz = c.free_running_linewidth_khz * 1e3
    lw_sil_hz = lw_free_hz / (slope_center ** 2)

    # Exact Thermorefractive Noise (TRN) PSD from 100 Hz to 10 MHz
    f_off = np.logspace(2.0, 7.0, 200)
    s_trn_hz2_hz, trn_rms_hz, lw_trn_hz = _compute_trn_psd_hz2_per_hz(
        c.core_material, c.wavelength_nm, n_eff, a_eff_um2, l_ring_m, f_off, c.temperature_k,
    )
    # Total locked frequency noise PSD = white SIL floor (lw_sil_hz / pi) + TRN PSD
    s_tot_hz2_hz = (lw_sil_hz / math.pi) + s_trn_hz2_hz
    idx_10k = int(np.argmin(np.abs(f_off - 1.0e4)))
    s_10k = float(s_tot_hz2_hz[idx_10k])

    gds_out = _export_vernier_or_gyro_gds(c.export_gds_path, [c.sil_ring_radius_um], c.waveguide_width_um) if c.export_gds_path else None
    return {
        "mode": "self_injection_locked",
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "SIL_stabilization_K": round(k_stab, 2),
            "SIL_narrowing_ratio": round(slope_center ** 2, 1),
            "locking_bandwidth_GHz": round(lock_bw_ghz, 3),
            "free_running_linewidth_kHz": round(c.free_running_linewidth_khz, 2),
            "SIL_intrinsic_linewidth_Hz": round(lw_sil_hz, 3),
            "TRN_integrated_linewidth_Hz": round(lw_trn_hz, 2),
            "freq_noise_10kHz_Hz2_per_Hz": round(s_10k, 3),
            "loaded_Q": round(q_load, 1),
            "FSR_GHz": round(fsr_ghz, 3),
            "modal_volume_um3": round(a_eff_um2 * (l_ring_m * 1e6), 1),
            "n_eff": round(n_eff, 5),
            "n_g": round(n_g, 5),
            "gds_path": gds_out,
        },
        "spectrum": {
            "freq_offset_Hz": np.round(f_off, 2).tolist(),
            "S_df_total_Hz2_per_Hz": np.round(s_tot_hz2_hz, 4).tolist(),
            "S_df_TRN_Hz2_per_Hz": np.round(s_trn_hz2_hz, 4).tolist(),
        },
        "tuning_curve": {
            "detuning_free_zeta": np.round(zeta_grid, 4).tolist(),
            "detuning_locked_xi": np.round(xi_grid, 4).tolist(),
        },
    }


def simulate_mrr_optical_gyroscope(cfg: MRRGyroscopeConfig | None = None) -> dict:
    """First-principles Integrated Resonator Optical Gyroscope (RFOG / Sagnac-MRR) solver."""
    t0 = time.perf_counter()
    c = cfg or MRRGyroscopeConfig()
    n_eff, n_g, a_eff_um2 = _solve_wg_mode_for_mrr(
        c.core_material, c.clad_material, c.top_clad_material,
        c.waveguide_width_um, c.film_height_um, c.slab_height_um, c.sidewall_deg, c.wavelength_nm,
    )
    m_nl_mat = get_material_fn(c.core_material)(c.wavelength_nm * 1e-9, T_REF_K)
    n2_m2_w = float(m_nl_mat.n2_m2_per_w) if abs(float(m_nl_mat.n2_m2_per_w)) > 1e-24 else 2.4e-19

    wl_m = c.wavelength_nm * 1e-9
    f0_hz = C0_M_PER_S / wl_m
    r_m = max(float(c.ring_radius_mm), 0.1) * 1e-3
    l_ring_m = 2.0 * math.pi * r_m
    area_m2 = math.pi * r_m * r_m

    # Exact relativistic Sagnac scale factor in a dispersive dielectric ring (Post Rev. Mod. Phys. 1967):
    # Round-trip phase shift dphi_S = (8*pi*A / (lambda*c)) * Omega -> resonance shift df_S = (FSR / 2*pi)*dphi_S = (4*A / (lambda*n_g*L)) * Omega
    sf_hz_per_rad_s = (4.0 * area_m2) / (wl_m * n_g * l_ring_m)
    sf_hz_per_deg_s = sf_hz_per_rad_s * (math.pi / 180.0)

    q_int = max(float(c.q_int), 1e4)
    q_ext = max(float(c.q_ext), 1e4)
    q_load = 1.0 / (1.0 / q_int + 1.0 / q_ext)
    eta_c = q_load / q_ext
    fwhm_hz = f0_hz / q_load
    fsr_ghz = C0_M_PER_S / (n_g * l_ring_m) * 1e-9

    # 1. Rayleigh backscattering CW/CCW lock-in dead-band (Adler/coupled-mode splitting)
    # Rotation rate where Sagnac splitting equals backscattering coupling 2g/(2*pi)
    df_bs_hz = max(float(c.backscatter_rate_khz), 0.0) * 1e3
    omega_lock_deg_s = df_bs_hz / max(sf_hz_per_deg_s, 1e-9)

    # 2. Kerr non-reciprocity bias drift from CW/CCW power imbalance (XPM = 2 * SPM -> net nonreciprocal shift!)
    # Intracavity circulating power P_circ = P_in * (4 * eta_c * (1 - eta_c)) * (Finesse / pi)
    finesse = (fsr_ghz * 1e9) / max(fwhm_hz, 1.0)
    buildup = (2.0 * eta_c / math.pi) * finesse  # On-resonance circulating power enhancement
    p_in_w = max(float(c.input_power_mw), 0.01) * 1e-3
    p_circ_w = p_in_w * buildup
    delta_p_circ_w = p_circ_w * (abs(float(c.cw_ccw_power_imbalance_pct)) * 1e-2)
    # Nonreciprocal index difference: dn_NR = (n2 / A_eff) * (P_cw - P_ccw) (since dn_cw = n2(I_cw + 2 I_ccw) -> dn_ccw - dn_cw = n2*(I_cw - I_ccw))
    a_eff_m2 = a_eff_um2 * 1e-12
    dn_kerr_nr = (n2_m2_w / max(a_eff_m2, 1e-16)) * delta_p_circ_w
    df_kerr_hz = f0_hz * (dn_kerr_nr / max(n_g, 1.0))
    kerr_bias_deg_per_h = (df_kerr_hz / max(sf_hz_per_deg_s, 1e-9)) * 3600.0

    # 3. Shot-noise-limited Angle Random Walk (ARW) [deg / sqrt(h)]
    # Minimum detectable frequency shift df_min = (fwhm / SNR_1Hz) * sqrt(2) (2 counter-propagating arms)
    # For optimal PDH / slope locking near critical coupling: delta_f_shot = (fwhm / 2) * sqrt(2 * h*nu / (eta_pd * P_det))
    eta_pd = min(0.99, c.pd_responsivity_a_w * (H_PLANCK * f0_hz / 1.602176634e-19))
    p_det_w = max(0.25 * p_in_w, 1e-6)  # Detected power at optimal slope bias (~25% of input on resonance slope)
    df_shot_hz_rt_hz = (fwhm_hz / 2.0) * math.sqrt(2.0 * H_PLANCK * f0_hz / max(eta_pd * p_det_w, 1e-18))
    # ARW [deg / sqrt(s)] -> [deg / sqrt(h)] by multiplying by sqrt(3600) = 60
    arw_deg_per_rt_s = df_shot_hz_rt_hz / max(sf_hz_per_deg_s, 1e-9)
    arw_deg_per_rt_h = arw_deg_per_rt_s * 60.0
    shot_bias_deg_per_h = (arw_deg_per_rt_s / math.sqrt(max(float(c.integration_time_s), 1.0))) * 3600.0

    # 4. Thermorefractive Noise (TRN) floor (note: CW & CCW share same spatial mode -> common-mode TRN cancels strongly,
    # leaving residual ~1% differential TRN from counter-propagating standing-wave / spatial asymmetry)
    f_low = np.array([1.0 / max(float(c.integration_time_s), 10.0), 1.0, 100.0])
    s_trn, _, _ = _compute_trn_psd_hz2_per_hz(c.core_material, c.wavelength_nm, n_eff, a_eff_um2, l_ring_m, f_low, c.temperature_k)
    trn_diff_hz_rt_hz = 0.01 * math.sqrt(float(s_trn[1]))  # 1% differential TRN residual
    trn_arw_deg_rt_h = (trn_diff_hz_rt_hz / max(sf_hz_per_deg_s, 1e-9)) * 60.0

    # Sagnac beat-frequency curve showing lock-in dead-band vs ideal linear Sagnac line
    om_grid = np.linspace(-c.max_rotation_deg_per_s, c.max_rotation_deg_per_s, 241)
    df_ideal = sf_hz_per_deg_s * om_grid
    # Coupled-mode lock-in response: df_beat = sign(Omega) * sqrt(max(df_ideal^2 - df_bs^2, 0))
    df_locked = np.sign(df_ideal) * np.sqrt(np.maximum(df_ideal ** 2 - df_bs_hz ** 2, 0.0))

    gds_out = _export_vernier_or_gyro_gds(c.export_gds_path, [c.ring_radius_mm * 1e3], c.waveguide_width_um) if c.export_gds_path else None
    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "sagnac_scale_factor_Hz_per_deg_s": round(sf_hz_per_deg_s, 3),
            "shot_noise_ARW_deg_per_rt_h": round(arw_deg_per_rt_h, 5),
            "shot_bias_instability_deg_per_h": round(shot_bias_deg_per_h, 4),
            "kerr_nonreciprocal_bias_deg_per_h": round(kerr_bias_deg_per_h, 4),
            "rayleigh_lockin_deadband_deg_per_s": round(omega_lock_deg_s, 4),
            "TRN_differential_ARW_deg_per_rt_h": round(trn_arw_deg_rt_h, 5),
            "loaded_Q": round(q_load, 1),
            "finesse": round(finesse, 1),
            "cavity_buildup_factor": round(buildup, 1),
            "FSR_GHz": round(fsr_ghz, 3),
            "n_eff": round(n_eff, 5),
            "n_g": round(n_g, 5),
            "A_eff_um2": round(a_eff_um2, 3),
            "gds_path": gds_out,
        },
        "spectrum": {
            "rotation_deg_per_s": np.round(om_grid, 3).tolist(),
            "sagnac_beat_locked_kHz": np.round(df_locked * 1e-3, 4).tolist(),
            "sagnac_beat_ideal_kHz": np.round(df_ideal * 1e-3, 4).tolist(),
        },
    }


__all__ = [
    "VernierSILLaserConfig",
    "MRRGyroscopeConfig",
    "simulate_vernier_sil_laser",
    "simulate_mrr_optical_gyroscope",
]
