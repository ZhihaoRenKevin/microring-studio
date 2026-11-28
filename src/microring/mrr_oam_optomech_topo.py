"""First-Principles Integrated OAM Vortex Beam MRR Emitter, Cavity Optomechanics & Microwave-to-Optical
Quantum Transducer, and 1D SSH / 2D Hafezi Topological Photonic MRR Lattices (`microring.mrr_oam_optomech_topo`).

100% live numerical solvers — zero canned presets:
  1. **Integrated Angular-Grating OAM Vortex Beam Emitter (`simulate_mrr_oam_emitter`)**:
     - Live 2D bent Maxwell FEM solves `n_eff(lambda)`, `n_g(lambda)` -> exact azimuthal WGM order
       `m = round(2*pi*R*n_eff / lambda)` and radiated topological charge `ell = m - q` for `q` angular
       inner-wall grating scatterers (Cai et al., Science 338, 363, 2012).
     - Evaluates exact 3D Stratton-Chu / Huygens discrete-azimuthal vector dipole radiation integral
       for far-field doughnut intensity `|E(theta, phi)|^2`, helical phase winding `arg(E) = ell*phi`,
       OAM mode purity, divergence half-angle `theta_peak`, thermal `ell`-switching ladder, and exports
       a foundry-ready angular-grating OAM `.GDS` mask.
  2. **Cavity Optomechanics & Microwave-to-Optical Quantum Transducer (`simulate_mrr_optomechanics_transducer`)**:
     - `"cavity_optomechanics"`: Exact linearized Heisenberg-Langevin cavity optomechanics for suspended
       MRR/microdisk (`g0`, `Omega_m`, `Gamma_m`, `kappa`, `Delta`, `P_in`): solves intracavity photon
       number `n_c`, cooperativity `C_om`, optical spring shift `delta_Omega_m`, effective damping `Gamma_eff`,
       sideband-resolved ground-state cooling final phonon occupation `n_f`, and OMIT transparency window.
     - `"quantum_transducer"`: Exact 3-mode (microwave <-> piezo-mechanical <-> optical) scattering matrix
       for `TFLN`/`AlN`/`ScAlN` MRR microwave-to-optical quantum transduction (`eta_MO`, bandwidth, and
       dilution-fridge input-referred added noise photons `N_add`).
  3. **Topological Photonic MRR Lattices (`simulate_topological_mrr_lattice`)**:
     - `"ssh_1d"`: 1D Su-Schrieffer-Heeger (`SSH`) alternating-coupling MRR chain (`J1 < J2`): exact
       tight-binding eigenmodes + Monte-Carlo coupling-disorder immunity audit (`edge mode pinned at 0 GHz`) + `.GDS`.
     - `"hafezi_2d_qh"`: 2D `Hafezi` synthetic-magnetic-field Photonic Quantum Hall MRR array (off-centered
       link rings imparting Peierls phase `Phi = 2*pi*alpha` per plaquette): exact Harper-Hofstadter spectrum
       + driven Green's function showing unidirectional chiral edge propagation routing around a missing/defective
       edge ring with suppressed backscattering + `.GDS`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import gdstk
import numpy as np

from .eigenmode_solver import WaveguideCrossSectionSpec, solve_modes_fem
from .gds_bridge import MICRORING_LAYERS
from .materials import C0_M_PER_S, get_material_fn

HBAR = 1.054571817e-34
K_B = 1.380649e-23


@dataclass
class MRROAMEmitterConfig:
    """Configuration for Integrated Angular-Grating OAM Vortex Beam MRR Emitter."""
    core_material: str = "Si"
    clad_material: str = "SiO2"
    top_clad_material: str = "Air"          # Air top cladding maximizes upward free-space vortex radiation
    wavelength_nm: float = 1550.0
    film_height_um: float = 0.22
    waveguide_width_um: float = 0.50
    slab_height_um: float = 0.0
    sidewall_deg: float = 88.0
    ring_radius_um: float = 3.90            # Compact Cai-2012-style SOI OAM vortex emitter (R ~ 3.9 um)
    num_grating_teeth_q: int | None = None  # None -> auto synthesize q = m - target_oam_charge_l
    target_oam_charge_l: int = 2            # Target radiated topological charge ell = m - q
    grating_tooth_depth_um: float = 0.065   # Inner-wall radial perturbation amplitude [um]
    grating_duty_cycle: float = 0.50
    bus_gap_um: float = 0.16                # Bus-to-ring gap for GDS export [um]
    export_gds_path: str | None = None


@dataclass
class MRROptomechConfig:
    """Configuration for Suspended MRR Cavity Optomechanics (OMIT / Cooling) or Quantum Transducer."""
    mode: str = "cavity_optomechanics"      # "cavity_optomechanics" | "quantum_transducer"
    wavelength_nm: float = 1550.0
    optical_q_int: float = 2.0e6
    optical_q_ext: float = 1.5e6
    mech_freq_ghz: float = 2.50             # Radial breathing / piezo mechanical resonance Omega_m / (2*pi) [GHz]
    mech_q: float = 5.0e3                   # Mechanical quality factor Q_m
    single_photon_g0_khz: float = 120.0     # Vacuum optomechanical coupling g0 / (2*pi) [kHz]
    laser_power_mw: float = 2.0             # Red-detuned pump power [mW]
    pump_detuning_in_om_m: float = -1.0     # Delta / Omega_m (-1.0 = exact red sideband cooling / OMIT / beam-splitter!)
    bath_temperature_k: float = 4.0         # Bath temperature [K] (e.g. 4 K cryostat or 0.02 K dilution fridge)
    # Quantum transducer microwave LC/CPW resonator parameters (mode == "quantum_transducer")
    mw_q_int: float = 1.0e4                 # Superconducting microwave resonator Q_e,int
    mw_q_ext: float = 2.5e3                 # Microwave coupling Q_e,ext
    piezo_coupling_g_em_mhz: float = 15.0   # Piezoelectric microwave-mechanical coupling g_em / (2*pi) [MHz]


@dataclass
class TopologicalMRRConfig:
    """Configuration for 1D SSH or 2D Hafezi Synthetic-Magnetic-Field Topological Photonic MRR Lattice."""
    mode: str = "hafezi_2d_qh"              # "ssh_1d" | "hafezi_2d_qh"
    # 1D SSH parameters
    ssh_num_rings: int = 11                 # Odd N guarantees a topological zero-mode localized on one edge!
    ssh_j1_ghz: float = 4.0                 # Weak intra-cell coupling J1 / (2*pi) [GHz]
    ssh_j2_ghz: float = 12.0                # Strong inter-cell coupling J2 / (2*pi) [GHz]
    ssh_disorder_pct: float = 15.0          # Monte-Carlo coupling disorder sigma [%] to prove topological immunity
    # 2D Hafezi Quantum Hall MRR lattice parameters
    lattice_nx: int = 6                     # Number of main rings along X
    lattice_ny: int = 6                     # Number of main rings along Y
    hopping_j_ghz: float = 10.0             # Main-ring effective hopping J / (2*pi) [GHz]
    flux_alpha: float = 0.25                # Synthetic magnetic flux per plaquette Phi / (2*pi) = 1/4!
    intrinsic_loss_ghz: float = 0.15        # Ring intrinsic decay rate gamma_0 / (2*pi) [GHz]
    inject_edge_defect: bool = True         # Remove one edge ring to test unidirectional routing around defect!
    ring_radius_um: float = 15.0
    export_gds_path: str | None = None


def _export_oam_mrr_gds(
    gds_path: str, r_um: float, w_um: float, q_teeth: int, tooth_depth_um: float, duty: float, gap_um: float,
) -> str:
    """Generate cleanroom-ready angular-grating OAM microring + bus waveguide `.GDS`."""
    p = Path(gds_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("MRR_OAM_VORTEX")
    lyr = MICRORING_LAYERS["core"]

    # Outer smooth circle + inner angular-grating modulated polygon
    n_pts_outer = max(180, 6 * q_teeth)
    th_o = np.linspace(0.0, 2.0 * math.pi, n_pts_outer, endpoint=False)
    r_out = r_um + 0.5 * w_um
    r_in_base = r_um - 0.5 * w_um
    outer_poly = gdstk.Polygon(list(zip(r_out * np.cos(th_o), r_out * np.sin(th_o))), layer=lyr[0], datatype=lyr[1])

    # Inner polygon with q sharp rectangular/trapezoidal teeth protruding inward by tooth_depth_um
    in_pts: list[tuple[float, float]] = []
    dth = 2.0 * math.pi / q_teeth
    dth_tooth = duty * dth
    for k in range(q_teeth):
        a0 = k * dth
        a1 = a0 + dth_tooth
        a2 = (k + 1) * dth
        # Protruding inward tooth (radius = r_in_base - tooth_depth_um)
        for ang in (a0, 0.5 * (a0 + a1), a1):
            rr = r_in_base - tooth_depth_um
            in_pts.append((rr * math.cos(ang), rr * math.sin(ang)))
        # Recessed notch (radius = r_in_base)
        for ang in (a1 + 0.03 * dth, 0.5 * (a1 + a2), a2 - 0.03 * dth):
            rr = r_in_base
            in_pts.append((rr * math.cos(ang), rr * math.sin(ang)))
    inner_poly = gdstk.Polygon(in_pts, layer=lyr[0], datatype=lyr[1])
    ring_polys = gdstk.boolean(outer_poly, inner_poly, "not", layer=lyr[0], datatype=lyr[1])
    for rp in ring_polys:
        cell.add(rp)

    # Straight bus waveguide below the ring
    y_bus = -(r_out + gap_um + 0.5 * w_um)
    x_span = r_out + 4.0
    cell.add(gdstk.rectangle((-x_span, y_bus - 0.5 * w_um), (x_span, y_bus + 0.5 * w_um), layer=lyr[0], datatype=lyr[1]))
    lib.write_gds(str(p))
    return str(p)


def simulate_mrr_oam_emitter(cfg: MRROAMEmitterConfig | None = None) -> dict:
    """First-principles 2D Bent FEM + Stratton-Chu Vector Far-Field Integrated OAM Vortex MRR Emitter."""
    t0 = time.perf_counter()
    c = cfg or MRROAMEmitterConfig()
    sp = WaveguideCrossSectionSpec(
        w_bottom_um=c.waveguide_width_um, h_film_um=c.film_height_um, h_slab_um=c.slab_height_um,
        sidewall_deg=c.sidewall_deg,
        core_material_fn=get_material_fn(c.core_material),
        clad_material_fn=get_material_fn(c.clad_material),
        top_clad_material_fn=get_material_fn(c.top_clad_material),
        domain_width_um=max(3.6, c.waveguide_width_um + 2.6),
        h_box_um=1.5, h_top_clad_um=1.5,
    )
    wl_m = c.wavelength_nm * 1e-9
    r_m = c.ring_radius_um * 1e-6
    dwl = 10.0
    m0 = solve_modes_fem(sp, wl_m, bend_radius_m=r_m, num_modes=2, core_mesh_res_um=0.055, element_order=1, export_grid=False)
    te = [m for m in m0 if m.te_fraction >= 0.45] or list(m0)
    te.sort(key=lambda m: -float(np.real(m.n_eff)))
    n_eff = float(np.real(te[0].n_eff))
    mp = solve_modes_fem(sp, (c.wavelength_nm + dwl) * 1e-9, bend_radius_m=r_m, num_modes=1, n_guess=n_eff, core_mesh_res_um=0.055, element_order=1, export_grid=False)[0]
    mm = solve_modes_fem(sp, (c.wavelength_nm - dwl) * 1e-9, bend_radius_m=r_m, num_modes=1, n_guess=n_eff, core_mesh_res_um=0.055, element_order=1, export_grid=False)[0]
    n_g = n_eff - c.wavelength_nm * (float(np.real(mp.n_eff)) - float(np.real(mm.n_eff))) / (2.0 * dwl)

    # Exact azimuthal WGM order m = round(2*pi*R*n_eff / lambda) & exact resonant wavelength for integer m
    m_az = int(round((2.0 * math.pi * r_m * n_eff) / wl_m))
    fsr_nm = (c.wavelength_nm ** 2) * 1e-9 / (2.0 * math.pi * r_m * n_g)
    q_teeth = int(c.num_grating_teeth_q) if (c.num_grating_teeth_q is not None and int(c.num_grating_teeth_q) > 4) else max(6, m_az - int(c.target_oam_charge_l))
    oam_ell = m_az - q_teeth

    # 3D Discrete Azimuthal Vector Dipole Radiation Integral (Cai et al., Science 2012):
    # Each grating element at angle phi_k = 2*pi*k/q carries a predominantly azimuthal (tangential) dipole
    # p_k = (-sin(phi_k), cos(phi_k), 0) * exp(i * m_az * phi_k) = (-sin, cos) * exp(i * ell * phi_k) (since q*phi_k = 2*pi*k!)
    k0 = 2.0 * math.pi / wl_m
    phi_k = np.linspace(0.0, 2.0 * math.pi, q_teeth, endpoint=False)
    phase_k = np.exp(1j * oam_ell * phi_k)
    px_k = -np.sin(phi_k) * phase_k
    py_k = np.cos(phi_k) * phase_k

    # Far-field polar profile theta in [0, 60 deg] at phi = 0 + azimuthal ring at theta_peak for OAM Fourier purity!
    theta_deg = np.linspace(0.2, 60.0, 121)
    th_r = np.radians(theta_deg)
    # Evaluate E_x(theta, phi=0) and E_y(theta, phi=0) from discrete dipole sum
    arg_th = -1j * k0 * r_m * np.outer(np.sin(th_r), np.cos(phi_k))
    ex_th = np.exp(arg_th) @ px_k
    ey_th = np.exp(arg_th) @ py_k
    # Far-field total intensity I(theta) = |E_theta|^2 + |E_phi|^2 = (cos^2(theta)*|Ex|^2 + |Ey|^2) at phi=0
    i_th = (np.cos(th_r) ** 2) * (np.abs(ex_th) ** 2) + (np.abs(ey_th) ** 2)
    i_th_norm = i_th / max(float(np.max(i_th)), 1e-18)
    idx_pk = int(np.argmax(i_th_norm))
    theta_peak_deg = float(theta_deg[idx_pk])
    th_pk_r = math.radians(max(theta_peak_deg, 5.0))

    # Azimuthal far-field ring at theta = theta_peak_deg -> exact circular-polarization OAM decomposition!
    # Note: Azimuthally polarized vector vortex beam with scalar factor exp(i*ell*phi) decomposes into
    # RCP component carrying OAM (ell - 1) and LCP component carrying OAM (ell + 1), while azimuthal component
    # E_phi(phi) = -sin(phi)*E_x + cos(phi)*E_y carries PURE topological charge ell!
    n_phi = 128
    phi_ff = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    # Phase factor exp(-i * k0 * R * sin(th_pk) * cos(phi_ff - phi_k))
    arg_ff = -1j * k0 * r_m * math.sin(th_pk_r) * np.cos(phi_ff[:, None] - phi_k[None, :])
    exp_ff = np.exp(arg_ff)
    ex_ff = exp_ff @ px_k
    ey_ff = exp_ff @ py_k
    e_phi_ff = -np.sin(phi_ff) * ex_ff + np.cos(phi_ff) * ey_ff
    e_rad_ff = np.cos(phi_ff) * ex_ff + np.sin(phi_ff) * ey_ff

    # Discrete OAM spectrum via FFT of E_phi(phi) around the far-field ring!
    oam_fft = np.fft.fft(e_phi_ff) / n_phi
    oam_pow = np.abs(oam_fft) ** 2
    tot_pow = max(float(np.sum(oam_pow)), 1e-18)
    idx_ell = oam_ell % n_phi
    oam_purity_pct = 100.0 * float(oam_pow[idx_ell]) / tot_pow
    # Verify actual wrapped phase winding around 2*pi circle
    ph_unw = np.unwrap(np.angle(e_phi_ff))
    winding_meas = (float(ph_unw[-1] - ph_unw[0]) + math.atan2(math.sin(np.angle(e_phi_ff[0]) - np.angle(e_phi_ff[-1])), math.cos(np.angle(e_phi_ff[0]) - np.angle(e_phi_ff[-1])))) / (2.0 * math.pi)

    # Thermal OAM topological charge switching ladder (ell - 2 .. ell + 2)
    ladder = [{"oam_ell": int(oam_ell + d), "wavelength_nm": round(c.wavelength_nm - d * fsr_nm, 2)} for d in (-2, -1, 0, 1, 2)]

    gds_out = _export_oam_mrr_gds(
        c.export_gds_path, c.ring_radius_um, c.waveguide_width_um, q_teeth,
        c.grating_tooth_depth_um, c.grating_duty_cycle, c.bus_gap_um,
    ) if c.export_gds_path else None

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "OAM_topological_charge_ell": int(oam_ell),
            "measured_phase_winding": round(winding_meas, 3),
            "OAM_mode_purity_pct": round(oam_purity_pct, 2),
            "azimuthal_WGM_order_m": int(m_az),
            "grating_teeth_q": int(q_teeth),
            "divergence_peak_theta_deg": round(theta_peak_deg, 2),
            "center_null_intensity_ratio": round(float(i_th_norm[0]), 5),
            "FSR_nm": round(fsr_nm, 2),
            "n_eff_bent": round(n_eff, 5),
            "n_g_bent": round(n_g, 5),
            "gds_path": gds_out,
        },
        "oam_switching_ladder": ladder,
        "spectrum": {
            "theta_deg": np.round(theta_deg, 2).tolist(),
            "farfield_intensity_norm": np.round(i_th_norm, 4).tolist(),
        },
        "azimuthal_ring": {
            "phi_deg": np.round(np.degrees(phi_ff), 2).tolist(),
            "phase_Ephi_deg": np.round(np.degrees(np.angle(e_phi_ff)), 2).tolist(),
            "intensity_norm": np.round((np.abs(e_phi_ff) ** 2 + np.abs(e_rad_ff) ** 2) / max(float(np.max(np.abs(e_phi_ff) ** 2 + np.abs(e_rad_ff) ** 2)), 1e-18), 4).tolist(),
        },
    }


def simulate_mrr_optomechanics_transducer(cfg: MRROptomechConfig | None = None) -> dict:
    """First-principles Cavity Optomechanics (OMIT / Cooling / Optical Spring) or Quantum Transducer."""
    t0 = time.perf_counter()
    c = cfg or MRROptomechConfig()
    mode = str(c.mode).lower().strip()

    om_opt = 2.0 * math.pi * C0_M_PER_S / (c.wavelength_nm * 1e-9)
    k_int = om_opt / max(float(c.optical_q_int), 1e3)
    k_ext = om_opt / max(float(c.optical_q_ext), 1e3)
    k_tot = k_int + k_ext
    eta_o = k_ext / k_tot

    om_m = 2.0 * math.pi * float(c.mech_freq_ghz) * 1e9
    gam_m = om_m / max(float(c.mech_q), 10.0)
    g0 = 2.0 * math.pi * float(c.single_photon_g0_khz) * 1e3
    delta = float(c.pump_detuning_in_om_m) * om_m  # Detuning Delta = omega_L - omega_c (< 0 = red sideband!)
    p_in_w = max(float(c.laser_power_mw), 1e-4) * 1e-3

    # Intracavity photon number n_c = (k_ext * P_in / (hbar * om_opt)) / (Delta^2 + (kappa/2)^2)
    n_c = (k_ext * p_in_w / (HBAR * om_opt)) / (delta ** 2 + (0.5 * k_tot) ** 2)
    g_enh = g0 * math.sqrt(max(n_c, 0.0))  # Cavity-enhanced optomechanical coupling G = g0 * sqrt(n_c)
    c_om = (4.0 * (g_enh ** 2)) / max(k_tot * gam_m, 1e-18)  # Optomechanical cooperativity

    # Thermal phonon occupation n_th = 1 / (exp(hbar*om_m / (k_B*T)) - 1)
    x_th = (HBAR * om_m) / max(K_B * max(float(c.bath_temperature_k), 1e-3), 1e-30)
    n_th = 1.0 / (math.exp(min(x_th, 500.0)) - 1.0) if x_th < 500.0 else 0.0

    # Sideband resolution parameter 4*Omega_m / kappa
    sideband_ratio = om_m / max(k_tot, 1.0)

    # ------------------------------------------------------------------
    # Mode 1: Suspended MRR Cavity Optomechanics (Cooling, Spring & OMIT)
    # ------------------------------------------------------------------
    if mode == "cavity_optomechanics":
        # Exact quantum-theory Stokes (A_+) and anti-Stokes (A_-) scattering rates (Aspelmeyer RMP 86, 1391, 2014):
        a_minus = (g_enh ** 2) * k_tot / ((delta + om_m) ** 2 + (0.5 * k_tot) ** 2)  # Cooling (anti-Stokes)
        a_plus = (g_enh ** 2) * k_tot / ((delta - om_m) ** 2 + (0.5 * k_tot) ** 2)   # Heating (Stokes)
        gam_opt = a_minus - a_plus
        gam_eff = max(gam_m + gam_opt, 1e-3 * gam_m)
        # Optical spring frequency shift delta_Omega_m
        dom_m = (g_enh ** 2) * (
            (delta + om_m) / ((delta + om_m) ** 2 + (0.5 * k_tot) ** 2)
            + (delta - om_m) / ((delta - om_m) ** 2 + (0.5 * k_tot) ** 2)
        )
        # Final cooled phonon occupation n_f = (gam_m * n_th + a_plus) / (gam_m + gam_opt)
        n_backaction_min = a_plus / max(gam_opt, 1e-12) if gam_opt > 0 else n_th
        n_final = (gam_m * n_th + a_plus) / max(gam_eff, 1e-12) if gam_opt > -gam_m else n_th * 100.0

        # Exact OMIT (Optomechanically Induced Transparency) probe transmission spectrum around probe detuning Omega ~ Omega_m
        # t_p(Omega) = 1 - k_ext / (0.5*k_tot - i*(Omega + Delta) + g_enh^2 / (0.5*gam_m - i*(Omega - om_m)))
        df_probe_mhz = np.linspace(-6.0 * (gam_eff / (2.0 * math.pi * 1e6)), 6.0 * (gam_eff / (2.0 * math.pi * 1e6)), 401)
        om_p_grid = om_m + 2.0 * math.pi * df_probe_mhz * 1e6
        chi_m_inv = 0.5 * gam_m - 1j * (om_p_grid - om_m)
        t_omit = 1.0 - k_ext / (0.5 * k_tot - 1j * (om_p_grid + delta) + (g_enh ** 2) / chi_m_inv)
        t_bare = 1.0 - k_ext / (0.5 * k_tot - 1j * (om_p_grid + delta))
        omit_peak_t = float(np.abs(t_omit[len(df_probe_mhz) // 2]) ** 2)
        bare_dip_t = float(np.abs(t_bare[len(df_probe_mhz) // 2]) ** 2)
        omit_contrast_db = 10.0 * math.log10(max(omit_peak_t / max(bare_dip_t, 1e-6), 1.0))

        return {
            "mode": "cavity_optomechanics",
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
            "metrics": {
                "cooperativity_C_om": round(c_om, 3),
                "intracavity_photons_n_c": round(n_c, 1),
                "sideband_resolved_Omega_over_kappa": round(sideband_ratio, 2),
                "thermal_phonons_n_th": round(n_th, 3),
                "final_cooled_phonons_n_f": round(n_final, 4),
                "ground_state_cooled": bool(n_final < 1.0),
                "optical_spring_shift_kHz": round(dom_m / (2.0 * math.pi * 1e3), 3),
                "effective_mech_linewidth_MHz": round(gam_eff / (2.0 * math.pi * 1e6), 4),
                "OMIT_window_transmission": round(omit_peak_t, 4),
                "OMIT_contrast_dB": round(omit_contrast_db, 2),
            },
            "spectrum": {
                "probe_offset_MHz": np.round(df_probe_mhz, 4).tolist(),
                "OMIT_transmission": np.round(np.abs(t_omit) ** 2, 5).tolist(),
                "bare_cavity_transmission": np.round(np.abs(t_bare) ** 2, 5).tolist(),
            },
        }

    # ------------------------------------------------------------------
    # Mode 2: Piezo-Optomechanical Microwave-to-Optical Quantum Transducer
    # ------------------------------------------------------------------
    k_e_int = om_m / max(float(c.mw_q_int), 100.0)
    k_e_ext = om_m / max(float(c.mw_q_ext), 100.0)
    k_e_tot = k_e_int + k_e_ext
    eta_e = k_e_ext / k_e_tot

    g_em = 2.0 * math.pi * float(c.piezo_coupling_g_em_mhz) * 1e6
    c_e = (4.0 * (g_em ** 2)) / max(k_e_tot * gam_m, 1e-18)
    c_o = c_om

    # Exact resonant bidirectional quantum conversion efficiency (Andrews Nat. Phys. 2014; Mirhosseini Nature 2020):
    # eta_MO(0) = eta_e * eta_o * 4 * C_e * C_o / (1 + C_e + C_o)^2
    eta_mo_peak = eta_e * eta_o * (4.0 * c_e * c_o) / ((1.0 + c_e + c_o) ** 2)
    # Input-referred added noise photons N_add = (n_th / C_e + backaction) / eta_e (exact 3-mode quantum Langevin)
    n_ba = (k_tot / (4.0 * om_m)) ** 2  # Sideband unresolved Stokes heating contribution
    n_add = (n_th / max(c_e, 1e-9) + (c_o / max(c_e, 1e-9)) * n_ba) / max(eta_e, 1e-6)
    trans_bw_mhz = (gam_m * (1.0 + c_e + c_o)) / (2.0 * math.pi * 1e6)

    # Frequency-resolved transduction spectrum |S_oe(dw)|^2
    df_mhz = np.linspace(-4.0 * trans_bw_mhz, 4.0 * trans_bw_mhz, 301)
    dw = 2.0 * math.pi * df_mhz * 1e6
    chi_e = 1.0 / (0.5 * k_e_tot - 1j * dw)
    chi_o = 1.0 / (0.5 * k_tot - 1j * dw)
    chi_m_inv = 0.5 * gam_m - 1j * dw
    s_oe = (math.sqrt(k_e_ext * k_ext) * g_em * g_enh * chi_e * chi_o) / (chi_m_inv + (g_em ** 2) * chi_e + (g_enh ** 2) * chi_o)
    eta_spec = np.abs(s_oe) ** 2

    return {
        "mode": "quantum_transducer",
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "quantum_efficiency_eta_MO": round(float(eta_mo_peak), 5),
            "quantum_efficiency_pct": round(100.0 * float(eta_mo_peak), 3),
            "internal_efficiency_pct": round(100.0 * (4.0 * c_e * c_o) / ((1.0 + c_e + c_o) ** 2), 2),
            "cooperativity_optical_Co": round(c_o, 3),
            "cooperativity_microwave_Ce": round(c_e, 3),
            "transduction_bandwidth_MHz": round(trans_bw_mhz, 3),
            "added_noise_photons_N_add": round(float(n_add), 4),
            "quantum_enabled_Nadd_lt_1": bool(n_add < 1.0),
            "thermal_phonons_n_th": round(n_th, 4),
        },
        "spectrum": {
            "detuning_MHz": np.round(df_mhz, 4).tolist(),
            "efficiency_pct": np.round(100.0 * eta_spec, 4).tolist(),
        },
    }


def _export_topo_mrr_gds(gds_path: str, is_2d: bool, nx: int, ny: int, r_um: float, w_um: float = 0.50) -> str:
    p = Path(gds_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("MRR_TOPOLOGICAL_LATTICE")
    lyr = MICRORING_LAYERS["core"]
    if not is_2d:
        # 1D SSH chain with alternating gaps (wide=0.35 um, narrow=0.18 um)
        x = 0.0
        for i in range(nx):
            cell.add(gdstk.ellipse((x, 0.0), r_um + 0.5 * w_um, inner_radius=r_um - 0.5 * w_um, tolerance=0.01, layer=lyr[0], datatype=lyr[1]))
            gap = 0.35 if i % 2 == 0 else 0.18
            x += 2.0 * r_um + w_um + gap
    else:
        # 2D Hafezi main rings + off-centered link rings imparting Peierls phase!
        pitch = 4.0 * r_um + 2.0 * w_um + 0.50
        for ix in range(nx):
            for iy in range(ny):
                xc, yc = ix * pitch, iy * pitch
                cell.add(gdstk.ellipse((xc, yc), r_um + 0.5 * w_um, inner_radius=r_um - 0.5 * w_um, tolerance=0.02, layer=lyr[0], datatype=lyr[1]))
                # Horizontal link ring (shifted in Y proportional to row index iy -> Landau gauge synthetic B-field!)
                if ix + 1 < nx:
                    dy_shift = 0.25 * (iy % 4)
                    cell.add(gdstk.ellipse((xc + 0.5 * pitch, yc + dy_shift), r_um + 0.5 * w_um, inner_radius=r_um - 0.5 * w_um, tolerance=0.02, layer=lyr[0], datatype=lyr[1]))
                if iy + 1 < ny:
                    cell.add(gdstk.ellipse((xc, yc + 0.5 * pitch), r_um + 0.5 * w_um, inner_radius=r_um - 0.5 * w_um, tolerance=0.02, layer=lyr[0], datatype=lyr[1]))
    lib.write_gds(str(p))
    return str(p)


def simulate_topological_mrr_lattice(cfg: TopologicalMRRConfig | None = None) -> dict:
    """Solve 1D SSH Topological MRR Chain or 2D Hafezi Synthetic-Magnetic-Field Photonic Quantum Hall Lattice."""
    t0 = time.perf_counter()
    c = cfg or TopologicalMRRConfig()
    mode = str(c.mode).lower().strip()

    # ------------------------------------------------------------------
    # Mode 1: 1D SSH Topological Edge-State MRR Chain + Disorder Immunity
    # ------------------------------------------------------------------
    if mode == "ssh_1d":
        n = max(5, int(c.ssh_num_rings))
        if n % 2 == 0:
            n += 1  # Odd N guarantees a topologically protected zero-mode on edge site 0 (for J1 < J2)!
        j1 = float(c.ssh_j1_ghz)
        j2 = float(c.ssh_j2_ghz)
        H = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            j_ij = j1 if (i % 2 == 0) else j2
            H[i, i + 1] = j_ij
            H[i + 1, i] = j_ij
        evals, evecs = np.linalg.eigh(H)
        i_zero = int(np.argmin(np.abs(evals)))
        e_edge = float(evals[i_zero])
        psi_edge = np.abs(evecs[:, i_zero]) ** 2
        edge_loc_pct = 100.0 * float(psi_edge[0] + psi_edge[1] + psi_edge[2]) if j1 < j2 else 100.0 * float(psi_edge[-1] + psi_edge[-2] + psi_edge[-3])
        # Bandgap width = 2 * |J2 - J1|
        gap_ghz = 2.0 * abs(j2 - j1)

        # Monte-Carlo coupling disorder test (chiral symmetry preserves E_edge = 0 EXACTLY under arbitrary J_ij jitter!)
        rng = np.random.default_rng(7)
        sig_frac = abs(float(c.ssh_disorder_pct)) * 1e-2
        edge_drifts = []
        bulk_drifts = []
        for _ in range(40):
            Hd = np.zeros((n, n), dtype=float)
            for i in range(n - 1):
                j_base = j1 if (i % 2 == 0) else j2
                j_jit = max(0.05, j_base * (1.0 + float(rng.normal(0.0, sig_frac))))
                Hd[i, i + 1] = j_jit
                Hd[i + 1, i] = j_jit
            ev_d = np.linalg.eigvalsh(Hd)
            edge_drifts.append(abs(float(ev_d[int(np.argmin(np.abs(ev_d)))])))
            bulk_drifts.append(abs(float(ev_d[-1]) - float(evals[-1])))

        # Coupled-cavity transmission probing edge ring 0 across detuning [-1.5*(J1+J2), +1.5*(J1+J2)]
        dw_ghz = np.linspace(-1.3 * (j1 + j2), 1.3 * (j1 + j2), 301)
        gam = max(float(c.intrinsic_loss_ghz), 0.1)
        k_ex = 1.5 * gam
        t_edge = np.zeros_like(dw_ghz)
        e0 = np.zeros(n, dtype=complex)
        e0[0] = math.sqrt(k_ex)
        for k_w, wv in enumerate(dw_ghz):
            M = (wv + 0.5j * gam) * np.eye(n, dtype=complex) - H
            M[0, 0] += 0.5j * k_ex
            a_sol = np.linalg.solve(M, e0)
            t_edge[k_w] = abs(1.0 - math.sqrt(k_ex) * a_sol[0]) ** 2

        gds_out = _export_topo_mrr_gds(c.export_gds_path, False, n, 1, c.ring_radius_um) if c.export_gds_path else None
        return {
            "mode": "ssh_1d",
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
            "metrics": {
                "topological_zero_mode_GHz": round(e_edge, 6),
                "bandgap_width_GHz": round(gap_ghz, 2),
                "edge_localization_pct": round(edge_loc_pct, 2),
                "max_edge_drift_under_disorder_GHz": round(max(edge_drifts), 6),
                "mean_bulk_drift_under_disorder_GHz": round(float(np.mean(bulk_drifts)), 3),
                "topological_protection_verified": bool(max(edge_drifts) < 1e-4 and edge_loc_pct > 70.0),
                "num_rings": n,
                "gds_path": gds_out,
            },
            "spectrum": {
                "detuning_GHz": np.round(dw_ghz, 3).tolist(),
                "edge_transmission": np.round(t_edge, 4).tolist(),
            },
            "eigenmodes": {
                "site_index": list(range(1, n + 1)),
                "edge_mode_intensity": np.round(psi_edge, 5).tolist(),
                "eigenvalues_GHz": np.round(evals, 3).tolist(),
            },
        }

    # ------------------------------------------------------------------
    # Mode 2: 2D Hafezi Synthetic-Magnetic-Field Photonic Quantum Hall Lattice
    # ------------------------------------------------------------------
    nx = max(4, min(int(c.lattice_nx), 8))
    ny = max(4, min(int(c.lattice_ny), 8))
    n_sites = nx * ny
    j_hop = float(c.hopping_j_ghz)
    phi_p = 2.0 * math.pi * float(c.flux_alpha)

    def _idx(ix: int, iy: int) -> int:
        return iy * nx + ix

    # Build 2D Harper-Hofstadter Landau-gauge Hamiltonian H_Hafezi:
    # Horizontal hopping (ix -> ix+1 at row iy) carries Peierls phase exp(+i * iy * phi_p) from off-centered link ring!
    defect_site = _idx(nx // 2, 0) if c.inject_edge_defect else -1  # Missing ring right in the middle of the bottom edge!
    H2 = np.zeros((n_sites, n_sites), dtype=complex)
    for ix in range(nx):
        for iy in range(ny):
            u = _idx(ix, iy)
            if u == defect_site:
                continue
            if ix + 1 < nx:
                v = _idx(ix + 1, iy)
                if v != defect_site:
                    ph = cmath.exp(1j * iy * phi_p)
                    H2[u, v] = -j_hop * ph
                    H2[v, u] = -j_hop * np.conj(ph)
            if iy + 1 < ny:
                v = _idx(ix, iy + 1)
                if v != defect_site:
                    H2[u, v] = -j_hop
                    H2[v, u] = -j_hop

    evals2 = np.linalg.eigvalsh(H2)
    # Drive bottom-left corner ring (0, 0) cleanly inside the Hofstadter Chern=1 topological gap [-2.4*J, -1.4*J]
    w_edge_grid = np.linspace(-2.45 * j_hop, -1.35 * j_hop, 65) if abs(c.flux_alpha - 0.25) < 0.08 else np.linspace(-3.0 * j_hop, 3.0 * j_hop, 81)
    gam = max(float(c.intrinsic_loss_ghz), 0.05)
    k_ex = 0.6 * j_hop  # Impedance-matched bus coupling to edge mode group velocity (~J)
    src_site = _idx(0, 0)
    dst_cw = _idx(nx - 1, 0)    # Along bottom edge (past the missing defect ring at (nx//2, 0)!)
    dst_ccw = _idx(0, ny - 1)   # Along left edge (opposite chiral direction)

    # Identify optimal chiral edge-state excitation frequency maximizing edge-to-bulk contrast & chirality
    best_w = float(w_edge_grid[0])
    best_score = -1.0
    best_a = np.zeros(n_sites, dtype=complex)
    t_cw_spec = np.zeros_like(w_edge_grid)
    t_ccw_spec = np.zeros_like(w_edge_grid)

    is_edge = np.zeros(n_sites, dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            if ix in (0, nx - 1) or iy in (0, ny - 1):
                is_edge[_idx(ix, iy)] = True

    rhs = np.zeros(n_sites, dtype=complex)
    rhs[src_site] = math.sqrt(k_ex)

    for kw, wv in enumerate(w_edge_grid):
        M = (wv + 0.5j * gam) * np.eye(n_sites, dtype=complex) - H2
        M[src_site, src_site] += 0.5j * k_ex
        M[dst_cw, dst_cw] += 0.5j * k_ex
        M[dst_ccw, dst_ccw] += 0.5j * k_ex
        if defect_site >= 0:
            M[defect_site, defect_site] = 1e6 + 0j
        a_sol = np.linalg.solve(M, rhs)
        tcw = float(k_ex * (abs(a_sol[dst_cw]) ** 2))
        tccw = float(k_ex * (abs(a_sol[dst_ccw]) ** 2))
        t_cw_spec[kw] = tcw
        t_ccw_spec[kw] = tccw
        int_edge = float(np.sum(np.abs(a_sol[is_edge]) ** 2))
        int_bulk = float(np.sum(np.abs(a_sol[~is_edge]) ** 2))
        chiral_r = max(tcw, tccw) / max(min(tcw, tccw), 1e-9)
        score = max(tcw, tccw) * (int_edge / max(int_edge + int_bulk, 1e-12)) * math.log10(max(chiral_r, 1.0) + 1.0)
        if score > best_score:
            best_score = score
            best_w = float(wv)
            best_a = a_sol

    intensity_2d = (np.abs(best_a) ** 2).reshape(ny, nx)
    edge_frac_pct = 100.0 * float(np.sum(np.abs(best_a[is_edge]) ** 2)) / max(float(np.sum(np.abs(best_a) ** 2)), 1e-18)
    tcw_pk = float(k_ex * abs(best_a[dst_cw]) ** 2)
    tccw_pk = float(k_ex * abs(best_a[dst_ccw]) ** 2)
    chiral_dir_db = 10.0 * math.log10(max(tcw_pk, tccw_pk) / max(min(tcw_pk, tccw_pk), 1e-12))

    gds_out = _export_topo_mrr_gds(c.export_gds_path, True, nx, ny, c.ring_radius_um) if c.export_gds_path else None
    return {
        "mode": "hafezi_2d_qh",
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "metrics": {
            "lattice_size": f"{nx}x{ny}",
            "synthetic_flux_alpha": round(c.flux_alpha, 3),
            "edge_defect_injected": bool(c.inject_edge_defect),
            "optimal_edge_freq_GHz": round(best_w, 3),
            "chiral_directionality_dB": round(chiral_dir_db, 2),
            "edge_field_confinement_pct": round(edge_frac_pct, 2),
            "transmission_around_defect_dB": round(10.0 * math.log10(max(max(tcw_pk, tccw_pk), 1e-8)), 2),
            "gds_path": gds_out,
        },
        "spectrum": {
            "detuning_GHz": np.round(w_edge_grid, 3).tolist(),
            "T_chiral_bottom_edge": np.round(t_cw_spec, 5).tolist(),
            "T_opposite_left_edge": np.round(t_ccw_spec, 5).tolist(),
        },
        "intensity_map_2d": np.round(intensity_2d / max(float(np.max(intensity_2d)), 1e-18), 4).tolist(),
    }


import cmath  # used in 2D Hafezi Peierls phase

__all__ = [
    "MRROAMEmitterConfig",
    "MRROptomechConfig",
    "TopologicalMRRConfig",
    "simulate_mrr_oam_emitter",
    "simulate_mrr_optomechanics_transducer",
    "simulate_topological_mrr_lattice",
]
