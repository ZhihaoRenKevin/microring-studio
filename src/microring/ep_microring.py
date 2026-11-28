"""First-Principles Microring Resonator Exceptional Point (MRR-EP) & Non-Hermitian Photonics Engine
(`microring.ep_microring`).

100% pure-Python, zero commercial runtime dependencies. Covers ALL 5 flagship integrated MRR
Exceptional Point regimes from Nature / Science / PRL literature with exact biorthogonal
Petermann noise metrology, forward/backward TCMT spectra, 2D-FEM physical gap synthesis,
and foundry `.GDS` layout generation:

  1. `chiral_single_ring` (Wiersig PRL 2014; Chen Nature 2017):
     Single MRR with dual azimuthal Rayleigh nanoscatterers (alpha_1, alpha_2, phase 2*m*beta)
     inducing asymmetric CW <-> CCW backscattering, exact Chiral EP2 (chirality -> +-1),
     unidirectional reflectionlessness (R_fwd != R_bwd), and sqrt(eps) splitting.
  2. `exceptional_surface` (Zhong PRL 2019; Soleymani Nat. Commun. 2022):
     Single MRR coupled to a bus waveguide with an asymmetric end-mirror (Sagnac loop reflector
     on one side: CCW -> mirror -> CW, zero CW -> CCW back-coupling). Locks strictly onto an
     Exceptional Surface (ES) immune to uniform fabrication resonance & loss drift!
  3. `pt_ep_n` (Hodaei Nature 2017; Peng Nat. Phys. 2014):
     N-ring (N = 2, 3, 4) PT-symmetric / passive-loss-biased coupled MRR chain with exact
     spin-J_x coupling profile J_k = (Delta_gamma / 4) * sqrt(k*(N-k)) realizing exact N-th
     order Exceptional Points (EP2, EP3, EP4) with eps^(1/N) splitting, squared/cubed-Lorentzian
     TCMT transmission spectra, and 2D-FEM twin-core supermode gap synthesis `gap_EP_um`!
  4. `anti_pt_bus` (Peng Nat. Phys. 2016; Zhang Nat. Commun. 2020; Lai Nature 2019):
     Two detuned MRRs coupled dissipatively through a shared bus waveguide (-i*Gamma_c
     imaginary coupling, zero gain required) exhibiting an Anti-PT EP at |Delta| = 2*Gamma_c
     with spontaneous frequency locking, linewidth bifurcation, and Sagnac rotation enhancement.
  5. `ep_encircling` (Doppler Nature 2016; Yoon Nature 2018; Nasari Nature 2022):
     Time-dependent RK4 non-Hermitian Schrödinger evolution encircling the MRR EP along CW vs
     CCW closed loops in (Delta, Delta_gamma) space, demonstrating chiral asymmetric mode
     switching regardless of initial input state.

All regimes compute exact biorthogonal left/right eigenvectors (<L_n|R_m> = delta_nm),
Petermann excess noise factor K_P = <L|L><R|R> / |<L|R>|^2, SNR vs Diabolic Point (DP)
metrology audit, and optional `.GDS` mask export.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import gdstk
import numpy as np

from .eigenmode_solver import WaveguideCrossSectionSpec
from .gds_bridge import MICRORING_LAYERS
from .materials import C0_M_PER_S, T_REF_K, get_material_fn
from .mrr_topologies import _compute_group_index_fast, _solve_single_mode


EP_REGIME_CATALOG: dict[str, dict[str, str]] = {
    "chiral_single_ring": {
        "name_en": "Single-Ring Chiral CW/CCW EP2 (Dual Nanoscatterers)",
        "name_zh": "单环手性 CW/CCW 二阶奇异点 (双纳米散射体, Chen Nature 2017)",
    },
    "exceptional_surface": {
        "name_en": "Single-Ring Bus-Mirror Exceptional Surface (Fab-Immune ES)",
        "name_zh": "单环端面镜奇异面 (抗工艺漂移 ES + 单向无反射, Zhong PRL 2019)",
    },
    "pt_ep_n": {
        "name_en": "N-Ring PT-Symmetric Higher-Order EP (EP2 / EP3 / EP4)",
        "name_zh": "N 环 PT 对称高阶奇异点 (EP2/EP3/EP4, ε^(1/N) 劈裂, Hodaei Nature 2017)",
    },
    "anti_pt_bus": {
        "name_en": "Bus-Coupled Anti-PT MRR EP (Dissipative Locking / Gyro)",
        "name_zh": "总线耗散耦合反 PT 奇异点 (无增益频率锁定/光学陀螺)",
    },
    "ep_encircling": {
        "name_en": "Dynamic CW vs CCW EP Encircling (Chiral Mode Switch)",
        "name_zh": "动态环绕奇异点 (顺/逆时针拓扑手性模式切换, Doppler Nature 2016)",
    },
}


@dataclass
class MRRExceptionalPointConfig:
    regime: str = "chiral_single_ring"      # "chiral_single_ring" | "exceptional_surface" | "pt_ep_n" | "anti_pt_bus" | "ep_encircling"
    order_n: int = 3                        # Order N for pt_ep_n (2, 3, or 4)
    core_material: str = "Si"               # Underlying waveguide core ("Si" | "Si3N4" | "LN_Xcut" | "4H-SiC")
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    film_height_um: float = 0.22
    ring_width_um: float = 0.48
    ring_radius_um: float = 20.0
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    gamma_int_ghz: float = 1.0              # Intrinsic cavity decay rate gamma_0 / (2*pi) [GHz]
    gamma_ext_ghz: float = 1.0              # Bus external coupling decay rate gamma_ext / (2*pi) [GHz]
    coupling_j_ghz: float = 2.0             # Inter-ring coupling J/(2*pi) [GHz] or scatterer strength |eps1| [GHz]
    mirror_reflectivity: float = 0.85       # Bus end-mirror power reflectivity R_m for exceptional_surface
    scatterer_phase_deg: float = 90.0       # Azimuthal phase 2*m*beta [deg] (exact chiral EP2 at 90 deg!)
    scatterer_ratio: float = 1.0            # |eps_2| / |eps_1| ratio (1.0 -> exact unidirectional EP2 at 90 deg!)
    perturbation_ghz: float = 0.05          # Target sensing perturbation eps/(2*pi) [GHz] for SNR/Petermann audit
    bus_phase_rad: float = 0.0              # Inter-ring bus phase phi = beta * L_bus [rad] for anti_pt_bus (0 or pi -> pure dissipative!)
    encircling_time_ns: float = 4.0         # Loop duration T for ep_encircling [ns]
    encircling_radius_ghz: float = 3.0      # Parameter-space loop radius in (delta, dgamma/4) [GHz]
    synthesize_fem_gap: bool = False        # Optional 2D-FEM twin-core supermode solver for physical gap_EP_um
    export_gds_path: str | None = None      # Optional .GDS output path


# ==============================================================================
# 1. Exact Biorthogonal Eigensystem & Petermann Excess Noise Audit
# ==============================================================================

def compute_biorthogonal_petermann(h_mat: np.ndarray) -> dict[str, Any]:
    """Compute exact complex eigenvalues, right/left eigenvectors, Petermann factor K_P, and coalescence angle.

    For a general non-Hermitian matrix H:
      H |R_n> = lambda_n |R_n>,    H^H |L_n> = lambda_n* |L_n>  (equivalently <L_n| H = lambda_n <L_n|)
      K_{P,n} = (<L_n|L_n> <R_n|R_n>) / |<L_n|R_n>|^2 >= 1.
    At an exact EP, |<L_n|R_n>| -> 0 so K_P -> inf (capped numerically at 1e8 for reporting).
    """
    n = h_mat.shape[0]
    evals, vr = np.linalg.eig(h_mat)
    # Sort eigenvalues deterministically by real part then imaginary part
    idx = np.lexsort((np.imag(evals), np.real(evals)))
    evals = evals[idx]
    vr = vr[:, idx]

    # Left eigenvectors: columns of vl satisfy H^H v_L = lambda* v_L
    evals_l, vl = np.linalg.eig(h_mat.conj().T)
    # Match each right eigenvalue lambda_n to closest left eigenvalue lambda_n*
    vl_matched = np.zeros_like(vr)
    used: set[int] = set()
    for i in range(n):
        target = np.conj(evals[i])
        dists = [abs(evals_l[j] - target) if j not in used else 1e30 for j in range(n)]
        j_best = int(np.argmin(dists))
        used.add(j_best)
        vl_matched[:, i] = vl[:, j_best]

    # Normalize each |R_i> and |L_i> to unit Euclidean norm
    kp_list: list[float] = []
    for i in range(n):
        r_vec = vr[:, i] / max(float(np.linalg.norm(vr[:, i])), 1e-30)
        l_vec = vl_matched[:, i] / max(float(np.linalg.norm(vl_matched[:, i])), 1e-30)
        vr[:, i] = r_vec
        vl_matched[:, i] = l_vec
        ov = abs(complex(np.vdot(l_vec, r_vec)))
        kp = 1.0 / max(ov ** 2, 1e-12)
        kp_list.append(float(min(kp, 1e10)))

    # Minimum eigenvector angle between distinct right eigenstates (0 deg at exact EP!)
    min_angle_deg = 90.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_th = min(1.0, abs(complex(np.vdot(vr[:, i], vr[:, j]))))
            ang = math.degrees(math.acos(cos_th))
            min_angle_deg = min(min_angle_deg, ang)

    # Chirality for 2x2 CW/CCW basis: alpha = (|c1|^2 - |c2|^2) / (|c1|^2 + |c2|^2)
    chirality = 0.0
    if n == 2:
        c0 = vr[:, 0]
        chirality = float((abs(c0[0]) ** 2 - abs(c0[1]) ** 2) / max(abs(c0[0]) ** 2 + abs(c0[1]) ** 2, 1e-30))

    return {
        "eigenvalues": evals,
        "right_vecs": vr,
        "left_vecs": vl_matched,
        "petermann_factors": kp_list,
        "max_petermann_K_P": float(max(kp_list)),
        "min_eigenvector_angle_deg": float(min_angle_deg),
        "chirality": float(chirality),
    }


def _synthesize_physical_ring_gap(cfg: MRRExceptionalPointConfig, target_j_ghz: float) -> dict[str, float]:
    """Use native 2D conformal vector FEM twin-core supermodes to solve physical ring-to-ring gap for J_target."""
    wl_m = float(cfg.wavelength_nm) * 1e-9
    core_fn = get_material_fn(cfg.core_material)
    clad_fn = get_material_fn(cfg.clad_material)
    top_fn = get_material_fn(cfg.top_clad_material)
    spec_single = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=0.0,
        sidewall_deg=90.0, core_material_fn=core_fn, clad_material_fn=clad_fn, top_clad_material_fn=top_fn,
        x_min_um=-2.2, x_max_um=2.2, h_box_um=1.8, h_top_clad_um=1.8, temperature_k=T_REF_K,
    )
    m0 = _solve_single_mode(spec_single, wl_m, cfg.ring_radius_um * 1e-6, cfg.polarization, "fem", 0.045, export_grid=False)
    neff0 = float(np.real(m0.n_eff))
    ng0 = float(_compute_group_index_fast(spec_single, wl_m, cfg.ring_radius_um * 1e-6, cfg.polarization, "fem", 0.045, neff0))
    fsr_ghz = (C0_M_PER_S / (ng0 * 2.0 * math.pi * cfg.ring_radius_um * 1e-6)) * 1e-9

    # Target point-coupled ring-to-ring amplitude coupling kappa = 2*pi * J / FSR
    kappa_target = min(0.95, (2.0 * math.pi * target_j_ghz) / max(fsr_ghz, 1.0))

    # Solve twin-core supermode Delta_n_sym_anti at 2 reference gaps (0.20, 0.35 um) -> exact exponential decay length!
    from .eigenmode_solver import solve_modes_fem
    gaps = [0.20, 0.35]
    dns: list[float] = []
    for g_um in gaps:
        spec_twin = WaveguideCrossSectionSpec(
            w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=0.0,
            sidewall_deg=90.0, core_material_fn=core_fn, clad_material_fn=clad_fn, top_clad_material_fn=top_fn,
            twin_core_gap_um=g_um, twin_core_w2_bottom_um=cfg.ring_width_um,
            x_min_um=-(cfg.ring_width_um + 0.5 * g_um + 1.8),
            x_max_um=+(cfg.ring_width_um + 0.5 * g_um + 1.8),
            h_box_um=1.8, h_top_clad_um=1.8, temperature_k=T_REF_K,
        )
        modes = solve_modes_fem(spec_twin, wl_m, num_modes=6, core_mesh_res_um=0.045)
        te_modes = [md for md in modes if md.te_fraction > 0.6]
        te_modes.sort(key=lambda md: float(np.real(md.n_eff)), reverse=True)
        dn = abs(float(np.real(te_modes[0].n_eff)) - float(np.real(te_modes[1].n_eff))) if len(te_modes) >= 2 else 1e-3
        dns.append(max(dn, 1e-7))

    # Exponential decay dn(g) = dn0 * exp(-gamma_g * g), point-ring effective coupling length L_eff = sqrt(pi * R / gamma_g)
    gamma_g_um = math.log(dns[0] / dns[1]) / (gaps[1] - gaps[0])
    gamma_g_um = max(gamma_g_um, 1.0)
    dn_at_0 = dns[0] * math.exp(gamma_g_um * gaps[0])
    l_eff_um = math.sqrt(math.pi * cfg.ring_radius_um / gamma_g_um)
    # kappa(g) = sin(pi * dn(g) * L_eff / wl_um) ~ (pi * dn_at_0 * L_eff / wl_um) * exp(-gamma_g * g)
    wl_um = float(cfg.wavelength_nm) * 1e-3
    k0_amp = (math.pi * dn_at_0 * l_eff_um) / wl_um
    gap_ep_um = math.log(max(k0_amp / max(kappa_target, 1e-6), 1.05)) / gamma_g_um
    gap_ep_um = float(np.clip(gap_ep_um, 0.08, 1.20))

    return {
        "neff_bare": round(neff0, 5),
        "ng_bare": round(ng0, 4),
        "FSR_GHz": round(fsr_ghz, 2),
        "kappa_EP_amplitude": round(kappa_target, 5),
        "coupler_decay_gamma_per_um": round(gamma_g_um, 3),
        "effective_coupling_length_um": round(l_eff_um, 2),
        "synthesized_gap_EP_um": round(gap_ep_um, 4),
    }


def _export_ep_gds(cfg: MRRExceptionalPointConfig, gap_um: float, out_path: str | Path) -> str:
    """Export foundry-ready `.GDS` layout for the chosen MRR Exceptional Point architecture."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(f"MRR_EP_{cfg.regime.upper()}")
    lay_core = MICRORING_LAYERS["core"][0]
    lay_htr = MICRORING_LAYERS["heater"][0]
    lay_m1 = MICRORING_LAYERS["electrode"][0]

    r = float(cfg.ring_radius_um)
    w = float(cfg.ring_width_um)
    g_bus = 0.20

    def _add_ring(cx: float, cy: float, rad: float, with_heater: bool = False) -> None:
        outer = gdstk.ellipse((cx, cy), rad + 0.5 * w, tolerance=0.002, layer=lay_core)
        inner = gdstk.ellipse((cx, cy), rad - 0.5 * w, tolerance=0.002, layer=lay_core)
        for poly in gdstk.boolean(outer, inner, "not", layer=lay_core):
            cell.add(poly)
        if with_heater:
            ho = gdstk.ellipse((cx, cy), rad + 0.7, initial_angle=0.25 * math.pi, final_angle=0.75 * math.pi, tolerance=0.005, layer=lay_htr)
            hi = gdstk.ellipse((cx, cy), rad - 0.7, initial_angle=0.25 * math.pi, final_angle=0.75 * math.pi, tolerance=0.005, layer=lay_htr)
            pts_h = np.vstack([ho.points, hi.points[::-1]])
            cell.add(gdstk.Polygon(pts_h, layer=lay_htr))
            cell.add(gdstk.rectangle((cx - 8.0, cy + rad + 6.0), (cx - 2.0, cy + rad + 14.0), layer=lay_m1))
            cell.add(gdstk.rectangle((cx + 2.0, cy + rad + 6.0), (cx + 8.0, cy + rad + 14.0), layer=lay_m1))

    y_bus = -(r + w + g_bus)
    if cfg.regime == "exceptional_surface":
        _add_ring(0.0, 0.0, r, with_heater=True)
        # Bus waveguide with Sagnac loop reflector on right side (+x) -> unidirectional CCW->CW coupling!
        cell.add(gdstk.rectangle((-r - 20.0, y_bus - 0.5 * w), (r + 18.0, y_bus + 0.5 * w), layer=lay_core))
        # Sagnac mirror loop at x = r + 18..r + 34 um
        loop_cx = r + 26.0
        lo = gdstk.ellipse((loop_cx, y_bus), 8.0 + 0.5 * w, tolerance=0.005, layer=lay_core)
        li = gdstk.ellipse((loop_cx, y_bus), 8.0 - 0.5 * w, tolerance=0.005, layer=lay_core)
        for poly in gdstk.boolean(lo, li, "not", layer=lay_core):
            cell.add(poly)
    elif cfg.regime == "chiral_single_ring":
        _add_ring(0.0, 0.0, r, with_heater=True)
        cell.add(gdstk.rectangle((-r - 20.0, y_bus - 0.5 * w), (r + 20.0, y_bus + 0.5 * w), layer=lay_core))
        # Dual azimuthal Rayleigh nanoscatterer notches at theta = 0 and theta = 45 deg
        for th_deg in (20.0, 20.0 + 0.5 * cfg.scatterer_phase_deg):
            th = math.radians(th_deg)
            sx = (r + 0.5 * w + 0.16) * math.cos(th)
            sy = (r + 0.5 * w + 0.16) * math.sin(th)
            cell.add(gdstk.ellipse((sx, sy), 0.12, tolerance=0.002, layer=lay_core))
    elif cfg.regime == "anti_pt_bus":
        pitch = 2.0 * r + 15.0
        _add_ring(-0.5 * pitch, 0.0, r, with_heater=True)
        _add_ring(+0.5 * pitch, 0.0, r, with_heater=True)
        cell.add(gdstk.rectangle((-pitch - 15.0, y_bus - 0.5 * w), (pitch + 15.0, y_bus + 0.5 * w), layer=lay_core))
    else:  # pt_ep_n or ep_encircling
        n_rings = int(np.clip(cfg.order_n, 2, 4)) if cfg.regime == "pt_ep_n" else 2
        step = 2.0 * r + w + gap_um
        x0 = -0.5 * (n_rings - 1) * step
        for k in range(n_rings):
            _add_ring(x0 + k * step, 0.0, r, with_heater=(k > 0))
        cell.add(gdstk.rectangle((x0 - r - 18.0, y_bus - 0.5 * w), (-x0 + r + 18.0, y_bus + 0.5 * w), layer=lay_core))

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(p))
    return str(p)


# ==============================================================================
# 2. Unified MRR Exceptional Point Physics Engine (All 5 Regimes)
# ==============================================================================

def simulate_mrr_exceptional_point(cfg: MRRExceptionalPointConfig) -> dict[str, Any]:
    """Run first-principles MRR Exceptional Point analysis across all 5 non-Hermitian regimes."""
    t0 = time.perf_counter()
    reg = str(cfg.regime).strip().lower()
    if reg not in EP_REGIME_CATALOG:
        reg = "chiral_single_ring"

    g_int = float(cfg.gamma_int_ghz)
    g_ext = float(cfg.gamma_ext_ghz)
    g_tot = g_int + g_ext
    j_ghz = float(cfg.coupling_j_ghz)
    eps_target = max(float(cfg.perturbation_ghz), 1e-4)

    # Frequency detuning grid for exact TCMT spectra
    span_ghz = max(8.0 * max(j_ghz, g_tot), 12.0)
    f_ghz = np.linspace(-span_ghz, span_ghz, 401)
    eps_sweep = np.logspace(-4, 0, 29) * max(j_ghz, g_tot)

    # Default outputs populated per regime
    t_fwd = np.ones_like(f_ghz)
    r_fwd = np.zeros_like(f_ghz)
    t_bwd = np.ones_like(f_ghz)
    r_bwd = np.zeros_like(f_ghz)
    ep_splits = np.zeros_like(eps_sweep)
    dp_splits = eps_sweep.copy()
    kp_sweep = np.ones_like(eps_sweep)
    snr_ratio_sweep = np.ones_like(eps_sweep)
    extra_curves: dict[str, Any] = {}
    metrics: dict[str, Any] = {
        "regime": reg,
        "order_N": 2,
    }

    # ------------------------------------------------------------------
    # REGIME 1: Single-Ring Chiral CW/CCW EP2 (Dual Rayleigh Scatterers)
    # ------------------------------------------------------------------
    if reg == "chiral_single_ring":
        # H_chiral = [[-i*g_tot/2 + e1 + e2,  A_12], [A_21, -i*g_tot/2 + e1 + e2]]
        # Scatterer complex polarizability: e_k = Re(e_k) - i*Im(e_k) (passive radiative scatterer Im >= 0 -> -i*gamma_s)
        # For exact Chiral EP2 (Chen Nature 2017): phase 2*m*beta = pi/2 & |e1| = |e2| with passive imaginary part
        # Wait: A_12 = e1 + e2 * exp(-i*phi_s), A_21 = e1 + e2 * exp(+i*phi_s).
        # Let e1 = J * (1 - 0.15j) [slightly dissipative scatterer], e2 = r * J * (1j) * exp(+i*(phi_s - pi/2))?
        # Even cleaner & exact (Wiersig PRL 2014 Eq. 3): two scatterers with complex shifts e1 = e_0 (real+loss) and
        # e2 = ratio * e_0 * exp(i*psi_material). When e1 = J*(1 - i), e2 = r*J*(1 + i) -> wait: passive scatterers
        # BOTH satisfy Im(e_k) <= 0 (loss)! To make A_21 = e1 + e2 * exp(+i*phi) = 0 with BOTH passive (Im <= 0):
        # Set e1 = J * (-i), e2 = ratio * J * (+1 - 0.0j)? Wait: e1 = J * exp(-i*pi/4) = J*(0.707 - 0.707i) (passive!),
        # e2 = ratio * J * exp(-i*3*pi/4) = ratio * J * (-0.707 - 0.707i) (ALSO passive, Im < 0!).
        # Then at phi_s = 2*m*beta = pi/2:
        #   A_21 = e1 + e2 * exp(+i*pi/2) = J*(0.707 - 0.707i) + ratio*J*(+0.707 - 0.707i) = ... wait:
        #   exp(+i*pi/2) * (-0.707 - 0.707i) = +0.707 - 0.707i (adds!), while
        #   exp(-i*pi/2) * (-0.707 - 0.707i) = -0.707 + 0.707i = -e1 (EXACT ZERO at ratio=1, phi=90 deg!!)!
        # Let's double check! Both e1 = J*(1-i)/sqrt(2) and e2 = r*J*(-1-i)/sqrt(2) have Im < 0 (100% PASSIVE!),
        # and at phi = 90 deg, ratio = 1:
        #   A_12 = e1 + e2*exp(-i*pi/2) = 0 (EXACT ZERO!)
        #   A_21 = e1 + e2*exp(+i*pi/2) = sqrt(2)*J*(1 - i) (NONZERO! |A_21| = 2*J!)
        # That is 100% physically passive (Wiersig PRA 2016) AND hits exact Chiral EP2 at phi=90 deg, ratio=1.0!
        phi_s = math.radians(float(cfg.scatterer_phase_deg))
        rat = float(cfg.scatterer_ratio)
        e1 = j_ghz * (1.0 - 1.0j) / math.sqrt(2.0)
        e2 = rat * j_ghz * (-1.0 - 1.0j) / math.sqrt(2.0)
        a12 = e1 + e2 * np.exp(-1j * phi_s)   # CCW -> CW coupling (0 at exact EP!)
        a21 = e1 + e2 * np.exp(+1j * phi_s)   # CW -> CCW coupling (nonzero at EP!)
        diag0 = -0.5j * g_tot + e1 + e2

        h0 = np.array([[diag0, a12], [a21, diag0]], dtype=complex)
        bio0 = compute_biorthogonal_petermann(h0)

        # Target perturbation: 3rd nanoparticle scatterer eps on ring adds eps * [[1, 1], [1, 1]]!
        h_pert = h0 + eps_target * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
        bio_p = compute_biorthogonal_petermann(h_pert)

        # Exact TCMT forward (CW driven) and backward (CCW driven) spectra:
        #   -i*(f*I - H) * a = sqrt(g_ext) * s_in  =>  a = i * (f*I - H)^(-1) * sqrt(g_ext) * s_in
        #   s_out_fwd = s_in - sqrt(g_ext) * a_CW,   s_refl_fwd = -sqrt(g_ext) * a_CCW
        #   s_out_bwd = s_in - sqrt(g_ext) * a_CCW,  s_refl_bwd = -sqrt(g_ext) * a_CW
        for i_f, f in enumerate(f_ghz):
            m_inv = np.linalg.inv(f * np.eye(2) - h0)
            a_fwd = 1j * math.sqrt(g_ext) * m_inv[:, 0]
            a_bwd = 1j * math.sqrt(g_ext) * m_inv[:, 1]
            t_fwd[i_f] = abs(1.0 - math.sqrt(g_ext) * a_fwd[0]) ** 2
            r_fwd[i_f] = abs(-math.sqrt(g_ext) * a_fwd[1]) ** 2
            t_bwd[i_f] = abs(1.0 - math.sqrt(g_ext) * a_bwd[1]) ** 2
            r_bwd[i_f] = abs(-math.sqrt(g_ext) * a_bwd[0]) ** 2

        # Perturbation sensitivity & Petermann K_P sweep (3rd Rayleigh target particle eps * [[1,1],[1,1]])
        for i_e, ev in enumerate(eps_sweep):
            hp = h0 + ev * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
            bp = compute_biorthogonal_petermann(hp)
            ep_splits[i_e] = abs(bp["eigenvalues"][1] - bp["eigenvalues"][0])
            kp_sweep[i_e] = bp["max_petermann_K_P"]
            # Noise-corrected SNR ratio vs DP: Signal_gain / sqrt(K_P)
            snr_ratio_sweep[i_e] = (ep_splits[i_e] / max(ev, 1e-12)) / math.sqrt(max(kp_sweep[i_e], 1.0))

        # Fit log-log sensitivity slope over small-eps decade
        slope_fit = float(np.polyfit(np.log10(eps_sweep[:10]), np.log10(np.maximum(ep_splits[:10], 1e-12)), 1)[0])
        refl_asym_db = 10.0 * math.log10((float(np.max(r_fwd)) + 1e-12) / (float(np.max(r_bwd)) + 1e-12))

        metrics.update({
            "A_CW_to_CCW_GHz": round(float(abs(a21)), 4),
            "A_CCW_to_CW_GHz": round(float(abs(a12)), 5),
            "chiral_asymmetry_ratio": round(float(abs(a21) / max(abs(a12), 1e-6)), 2),
            "eigenvector_chirality": round(float(bio0["chirality"]), 5),
            "min_eigenvector_angle_deg": round(float(bio0["min_eigenvector_angle_deg"]), 4),
            "splitting_at_perturbation_GHz": round(float(abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0])), 5),
            "EP_vs_DP_signal_gain": round(float(abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0]) / eps_target), 2),
            "fitted_power_law_exponent": round(float(slope_fit), 3),
            "Petermann_K_P_at_perturbation": round(float(bio_p["max_petermann_K_P"]), 2),
            "SNR_ratio_vs_DP_at_perturbation": round(float((abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0]) / eps_target) / math.sqrt(max(bio_p["max_petermann_K_P"], 1.0))), 3),
            "max_R_fwd": round(float(np.max(r_fwd)), 4),
            "max_R_bwd": round(float(np.max(r_bwd)), 6),
            "reflection_asymmetry_dB": round(float(refl_asym_db), 2),
        })

    # ------------------------------------------------------------------
    # REGIME 2: Single-Ring Bus-Mirror Exceptional Surface (Zhong PRL 2019)
    # ------------------------------------------------------------------
    elif reg == "exceptional_surface":
        # Bus waveguide with end-mirror of amplitude reflectivity r_m = sqrt(R_m) on +x side:
        # CW mode couples into +x bus -> hits mirror -> reflects back into -x bus -> drives CCW mode!
        # Unidirectional coupling: A_{CCW <- CW} = -i * r_m * g_ext, while A_{CW <- CCW} = 0 identically!
        # CRITICAL PROPERTY: Uniform fabrication drift (delta_omega_fab, delta_gamma_fab) shifts the
        # diagonal identically while keeping A_{CW <- CCW} = 0, so the cavity REMAINS LOCKED ON THE ES!
        r_m = math.sqrt(float(np.clip(cfg.mirror_reflectivity, 0.01, 0.999)))
        a_unidir = -1j * r_m * g_ext
        diag0 = -0.5j * g_tot
        h0 = np.array([[diag0, 0.0], [a_unidir, diag0]], dtype=complex)
        bio0 = compute_biorthogonal_petermann(h0)

        # Verify fabrication-drift immunity: add huge +5 GHz resonance drift & +30% loss drift -> STILL exact ES!
        h_fab = np.array([[5.0 - 0.5j * (1.3 * g_tot), 0.0], [a_unidir, 5.0 - 0.5j * (1.3 * g_tot)]], dtype=complex)
        bio_fab = compute_biorthogonal_petermann(h_fab)

        # Target nanoparticle perturbation eps * [[1, 1], [1, 1]] induces nonzero back-coupling A_{CW<-CCW} = eps!
        h_pert = h0 + eps_target * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
        bio_p = compute_biorthogonal_petermann(h_pert)

        # Exact 1-port reflection from mirror-terminated bus (squared-Lorentzian reflection dip!) & forward/back contrast
        for i_f, f in enumerate(f_ghz):
            m_inv = np.linalg.inv(f * np.eye(2) - h0)
            a_fwd = 1j * math.sqrt(g_ext) * m_inv[:, 0]
            a_bwd = 1j * math.sqrt(g_ext) * m_inv[:, 1]
            # Through without mirror vs reflected through mirror port (demonstrates squared-Lorentzian lineshape!)
            t_cw = 1.0 - math.sqrt(g_ext) * a_fwd[0]
            r_ccw_from_cw = -math.sqrt(g_ext) * a_fwd[1]
            t_fwd[i_f] = abs(t_cw) ** 2
            r_fwd[i_f] = abs(r_m * t_cw - math.sqrt(g_ext) * a_fwd[1]) ** 2  # Total port-1 reflection with Sagnac mirror!
            t_bwd[i_f] = abs(1.0 - math.sqrt(g_ext) * a_bwd[1]) ** 2
            r_bwd[i_f] = abs(-math.sqrt(g_ext) * a_bwd[0]) ** 2  # Exactly 0 on the Exceptional Surface!

        for i_e, ev in enumerate(eps_sweep):
            hp = h0 + ev * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
            bp = compute_biorthogonal_petermann(hp)
            ep_splits[i_e] = abs(bp["eigenvalues"][1] - bp["eigenvalues"][0])
            kp_sweep[i_e] = bp["max_petermann_K_P"]
            snr_ratio_sweep[i_e] = (ep_splits[i_e] / max(ev, 1e-12)) / math.sqrt(max(kp_sweep[i_e], 1.0))

        slope_fit = float(np.polyfit(np.log10(eps_sweep[:10]), np.log10(np.maximum(ep_splits[:10], 1e-12)), 1)[0])
        metrics.update({
            "unidirectional_coupling_GHz": round(float(abs(a_unidir)), 4),
            "mirror_reflectivity_R": round(float(r_m ** 2), 4),
            "eigenvector_chirality": round(float(bio0["chirality"]), 5),
            "min_eigenvector_angle_deg": round(float(bio0["min_eigenvector_angle_deg"]), 6),
            "fab_drift_5GHz_angle_deg": round(float(bio_fab["min_eigenvector_angle_deg"]), 6),
            "fab_drift_immune_ES": bool(bio_fab["min_eigenvector_angle_deg"] < 1e-3),
            "splitting_at_perturbation_GHz": round(float(abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0])), 5),
            "EP_vs_DP_signal_gain": round(float(abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0]) / eps_target), 2),
            "fitted_power_law_exponent": round(float(slope_fit), 3),
            "Petermann_K_P_at_perturbation": round(float(bio_p["max_petermann_K_P"]), 2),
            "SNR_ratio_vs_DP_at_perturbation": round(float((abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0]) / eps_target) / math.sqrt(max(bio_p["max_petermann_K_P"], 1.0))), 3),
            "max_R_bwd_on_ES": round(float(np.max(r_bwd)), 8),
        })

    # ------------------------------------------------------------------
    # REGIME 3: N-Ring PT-Symmetric Higher-Order EP (EP2 / EP3 / EP4)
    # ------------------------------------------------------------------
    elif reg == "pt_ep_n":
        n = int(np.clip(cfg.order_n, 2, 4))
        metrics["order_N"] = n
        # Exact spin-S J_x + i*J_z construction (Hodaei Nature 2017; Graefe JPA 2008):
        #   H_0 = -i * g_center/2 * I + kappa_s * (J_x + i * J_z)  (all eigenvalues == -i*g_center/2!)
        # For N=2: J_12 = J, diag losses = [g0, g0 + 4J] -> |dg| = 4J!
        # For N=3: J_12 = J_23 = J, diag losses = [g0, g0 + 2*sqrt(2)*J, g0 + 4*sqrt(2)*J] -> |g3 - g1| = 4*sqrt(2)*J!
        # For N=4: J_12 = J_34 = sqrt(3)/2 * J, J_23 = J, diag losses = [g0, g0+2J, g0+4J, g0+6J]!
        # Let's construct exact J_x & J_z scaled so max coupling = j_ghz, shifted to passive (all Im <= 0)!
        s_spin = 0.5 * (n - 1)
        m_vals = np.array([s_spin - k for k in range(n)], dtype=float)  # +S, S-1, ..., -S
        off_raw = np.array([0.5 * math.sqrt((k + 1) * (n - 1 - k)) for k in range(n - 1)], dtype=float)
        # Scale factor so the first coupling equals j_ghz (for N=2: off_raw=[0.5] -> scale=2*J; N=3: off_raw=[1/sqrt(2)] -> scale=sqrt(2)*J)
        scale = j_ghz / off_raw[0]
        j_couplings = scale * off_raw
        # Passive loss ladder: gamma_k = g_tot + 2 * scale * (S - m_k)  (k=0 has lowest loss g_tot, k=N-1 has highest loss!)
        gammas = np.array([g_tot + 2.0 * scale * (s_spin - m_vals[k]) for k in range(n)], dtype=float)

        h0 = np.diag(-0.5j * gammas).astype(complex)
        for k in range(n - 1):
            h0[k, k + 1] = j_couplings[k]
            h0[k + 1, k] = j_couplings[k]

        bio0 = compute_biorthogonal_petermann(h0)
        # Perturbation eps on Ring 1 (top-left entry H[0,0] += eps)
        e_mat = np.zeros((n, n), dtype=complex)
        e_mat[0, 0] = 1.0
        h_pert = h0 + eps_target * e_mat
        bio_p = compute_biorthogonal_petermann(h_pert)

        # TCMT bus coupled to Ring 1 (all-pass transmission T(f))
        for i_f, f in enumerate(f_ghz):
            m_inv = np.linalg.inv(f * np.eye(n) - h0)
            a_vec = 1j * math.sqrt(g_ext) * m_inv[:, 0]
            t_fwd[i_f] = abs(1.0 - math.sqrt(g_ext) * a_vec[0]) ** 2
            t_bwd[i_f] = t_fwd[i_f]

        # Sweep eps -> exact eps^(1/N) splitting!
        for i_e, ev in enumerate(eps_sweep):
            hp = h0 + ev * e_mat
            bp = compute_biorthogonal_petermann(hp)
            ev_s = bp["eigenvalues"]
            # Max real/complex spread among the N split branches
            ep_splits[i_e] = float(np.max(np.abs(ev_s[:, None] - ev_s[None, :])))
            kp_sweep[i_e] = bp["max_petermann_K_P"]
            snr_ratio_sweep[i_e] = (ep_splits[i_e] / max(ev, 1e-12)) / math.sqrt(max(kp_sweep[i_e], 1.0))

        slope_fit = float(np.polyfit(np.log10(eps_sweep[:10]), np.log10(np.maximum(ep_splits[:10], 1e-12)), 1)[0])
        p_split = float(np.max(np.abs(bio_p["eigenvalues"][:, None] - bio_p["eigenvalues"][None, :])))

        metrics.update({
            "ring_loss_rates_GHz": [round(float(x), 3) for x in gammas],
            "inter_ring_couplings_GHz": [round(float(x), 3) for x in j_couplings],
            "exact_EP_loss_span_GHz": round(float(gammas[-1] - gammas[0]), 3),
            "theoretical_exponent_1_over_N": round(1.0 / n, 4),
            "fitted_power_law_exponent": round(slope_fit, 3),
            "min_eigenvector_angle_deg": round(bio0["min_eigenvector_angle_deg"], 4),
            "splitting_at_perturbation_GHz": round(p_split, 5),
            "EP_vs_DP_signal_gain": round(p_split / eps_target, 2),
            "Petermann_K_P_at_perturbation": round(bio_p["max_petermann_K_P"], 2),
            "SNR_ratio_vs_DP_at_perturbation": round((p_split / eps_target) / math.sqrt(max(bio_p["max_petermann_K_P"], 1.0)), 3),
        })

    # ------------------------------------------------------------------
    # REGIME 4: Dissipatively Bus-Coupled Anti-PT MRR EP (Zero Gain!)
    # ------------------------------------------------------------------
    elif reg == "anti_pt_bus":
        # Two rings coupled via shared bus waveguide with phase phi (phi = m*pi -> pure dissipative -i*Gamma_c!):
        #   H_APT = [[+Delta/2 - i*g_tot/2, -i*g_ext*exp(i*phi)],
        #            [-i*g_ext*exp(i*phi),  -Delta/2 - i*g_tot/2]]
        # At phi = 0 or pi, eigenvalues = -i*g_tot/2 +- sqrt((Delta/2)^2 - g_ext^2)!
        # Exact Anti-PT EP at |Delta_EP| = 2 * g_ext (ZERO optical gain required!!):
        #   |Delta| < 2*g_ext: Anti-PT UNBROKEN (frequencies LOCKED Re(lam1)=Re(lam2)=0, linewidths bifurcate!)
        #   |Delta| > 2*g_ext: Anti-PT BROKEN (frequencies split, linewidths equal!)
        phi_b = float(cfg.bus_phase_rad)
        coup = -1j * g_ext * np.exp(1j * phi_b)
        delta_ep = 2.0 * g_ext
        h0 = np.array([[0.5 * delta_ep - 0.5j * g_tot, coup], [coup, -0.5 * delta_ep - 0.5j * g_tot]], dtype=complex)
        bio0 = compute_biorthogonal_petermann(h0)

        # Sweep inter-ring detuning Delta from 0 to 2.5*Delta_EP to map frequency locking & linewidth bifurcation!
        d_sweep = np.linspace(0.0, 2.5 * delta_ep, 81)
        re_diff = np.zeros_like(d_sweep)
        im_diff = np.zeros_like(d_sweep)
        for i_d, dv in enumerate(d_sweep):
            hd = np.array([[0.5 * dv - 0.5j * g_tot, coup], [coup, -0.5 * dv - 0.5j * g_tot]], dtype=complex)
            ev = np.linalg.eigvals(hd)
            re_diff[i_d] = abs(np.real(ev[0] - ev[1]))
            im_diff[i_d] = abs(np.imag(ev[0] - ev[1]))
        extra_curves["anti_pt_detuning_sweep"] = {
            "detuning_GHz": np.round(d_sweep, 4).tolist(),
            "freq_split_GHz": np.round(re_diff, 4).tolist(),
            "linewidth_split_GHz": np.round(im_diff, 4).tolist(),
        }

        # Perturbation eps added to Delta around exact Anti-PT EP -> sqrt(eps) frequency splitting (optical gyroscope!)
        h_pert = np.array([[0.5 * (delta_ep + eps_target) - 0.5j * g_tot, coup], [coup, -0.5 * (delta_ep + eps_target) - 0.5j * g_tot]], dtype=complex)
        bio_p = compute_biorthogonal_petermann(h_pert)
        for i_e, ev in enumerate(eps_sweep):
            hp = np.array([[0.5 * (delta_ep + ev) - 0.5j * g_tot, coup], [coup, -0.5 * (delta_ep + ev) - 0.5j * g_tot]], dtype=complex)
            bp = compute_biorthogonal_petermann(hp)
            ep_splits[i_e] = abs(bp["eigenvalues"][1] - bp["eigenvalues"][0])
            kp_sweep[i_e] = bp["max_petermann_K_P"]
            snr_ratio_sweep[i_e] = (ep_splits[i_e] / max(ev, 1e-12)) / math.sqrt(max(kp_sweep[i_e], 1.0))

        # Cascaded 2-ring bus transmission spectrum at exact Anti-PT EP
        for i_f, f in enumerate(f_ghz):
            m_inv = np.linalg.inv(f * np.eye(2) - h0)
            s_in = np.array([1.0, np.exp(1j * phi_b)], dtype=complex)
            a_v = 1j * math.sqrt(g_ext) * (m_inv @ s_in)
            t_out = np.exp(1j * phi_b) - math.sqrt(g_ext) * (np.exp(1j * phi_b) * a_v[0] + a_v[1])
            t_fwd[i_f] = min(1.0, abs(t_out) ** 2)
            t_bwd[i_f] = t_fwd[i_f]

        slope_fit = float(np.polyfit(np.log10(eps_sweep[:10]), np.log10(np.maximum(ep_splits[:10], 1e-12)), 1)[0])
        metrics.update({
            "exact_AntiPT_EP_detuning_GHz": round(float(delta_ep), 4),
            "dissipative_coupling_Gamma_c_GHz": round(float(g_ext), 4),
            "locked_freq_split_at_0p5_EP_GHz": round(float(re_diff[16]), 5),
            "min_eigenvector_angle_deg": round(float(bio0["min_eigenvector_angle_deg"]), 4),
            "splitting_at_perturbation_GHz": round(float(abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0])), 5),
            "EP_vs_DP_signal_gain": round(float(abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0]) / eps_target), 2),
            "fitted_power_law_exponent": round(float(slope_fit), 3),
            "Petermann_K_P_at_perturbation": round(float(bio_p["max_petermann_K_P"]), 2),
            "SNR_ratio_vs_DP_at_perturbation": round(float((abs(bio_p["eigenvalues"][1] - bio_p["eigenvalues"][0]) / eps_target) / math.sqrt(max(bio_p["max_petermann_K_P"], 1.0))), 3),
        })

    # ------------------------------------------------------------------
    # REGIME 5: Dynamic CW vs CCW Topological EP Encircling (Doppler 2016)
    # ------------------------------------------------------------------
    else:
        # Parameter space (delta(t), dg_eff(t)) encircling the EP at (delta=0, dg_eff = J) where
        # H(t) = [[delta(t) - i*dg_eff(t), J], [J, -delta(t) + i*dg_eff(t)]] (in rotating frame of average loss).
        # Start/end point at t=0, T: delta=0, dg_eff = 0 (Hermitian symmetric/antisymmetric eigenstates |+>, |->)!
        # Loop center: (delta_c = 0, dg_c = J), radius r_loop = max(J, 1.2*J) passing through (0, 0) at phi=-pi/2!
        r_loop = max(1.05 * j_ghz, float(cfg.encircling_radius_ghz))
        # Center shifted so loop starts & ends strictly at (delta=0, dg_eff = 0) and strictly encloses EP (0, J)!
        dg_center = r_loop  # Since r_loop > J, circle centered at (0, r_loop) encloses (0, J) and starts at (0, 0)!
        t_tot_ns = max(float(cfg.encircling_time_ns), 0.5)
        n_steps = 800
        dt_ns = t_tot_ns / n_steps
        # Scale angular frequency rad/ns = 2*pi * GHz
        twopi = 2.0 * math.pi

        def _h_at(s_norm: float, direction: int) -> np.ndarray:
            # s_norm in [0, 1]; angle starts at -pi/2 -> goes CW (-1) or CCW (+1) by 2*pi
            ang = -0.5 * math.pi + direction * (2.0 * math.pi * s_norm)
            delt = r_loop * math.cos(ang)
            dg_e = dg_center + r_loop * math.sin(ang)
            return twopi * np.array([[delt - 1j * dg_e, j_ghz], [j_ghz, -delt + 1j * dg_e]], dtype=complex)

        # Eigenstates at start/end (0, 0): |+> = [1, 1]/sqrt(2), |-> = [1, -1]/sqrt(2)
        v_plus = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        v_minus = np.array([1.0, -1.0], dtype=complex) / math.sqrt(2.0)

        def _evolve_loop(psi0: np.ndarray, direction: int) -> tuple[float, list[float]]:
            psi = psi0.astype(complex).copy()
            pop_plus: list[float] = []
            for k in range(n_steps):
                s0 = k / n_steps
                sm = (k + 0.5) / n_steps
                s1 = (k + 1.0) / n_steps
                k1 = -1j * (_h_at(s0, direction) @ psi)
                k2 = -1j * (_h_at(sm, direction) @ (psi + 0.5 * dt_ns * k1))
                k3 = -1j * (_h_at(sm, direction) @ (psi + 0.5 * dt_ns * k2))
                k4 = -1j * (_h_at(s1, direction) @ (psi + dt_ns * k3))
                psi = psi + (dt_ns / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                # Renormalize amplitude periodically to prevent exponential overflow while keeping exact state ray!
                nrm = float(np.linalg.norm(psi))
                if nrm > 1e6 or nrm < 1e-6:
                    psi /= max(nrm, 1e-30)
                if k % 16 == 0:
                    p_p = abs(complex(np.vdot(v_plus, psi))) ** 2 / max(float(np.linalg.norm(psi)) ** 2, 1e-30)
                    pop_plus.append(round(p_p, 4))
            p_final_plus = abs(complex(np.vdot(v_plus, psi))) ** 2 / max(float(np.linalg.norm(psi)) ** 2, 1e-30)
            return float(p_final_plus), pop_plus

        cw_from_p, tr_cw_p = _evolve_loop(v_plus, -1)
        cw_from_m, tr_cw_m = _evolve_loop(v_minus, -1)
        ccw_from_p, tr_ccw_p = _evolve_loop(v_plus, +1)
        ccw_from_m, tr_ccw_m = _evolve_loop(v_minus, +1)

        # Evaluate reference EP2 Petermann at target perturbation
        h_ep_ref = np.array([[-1j * j_ghz, j_ghz], [j_ghz, +1j * j_ghz]], dtype=complex)
        bio_p = compute_biorthogonal_petermann(h_ep_ref + eps_target * np.array([[1.0, 0.0], [0.0, 0.0]]))
        for i_e, ev in enumerate(eps_sweep):
            bp = compute_biorthogonal_petermann(h_ep_ref + ev * np.array([[1.0, 0.0], [0.0, 0.0]]))
            ep_splits[i_e] = abs(bp["eigenvalues"][1] - bp["eigenvalues"][0])
            kp_sweep[i_e] = bp["max_petermann_K_P"]
            snr_ratio_sweep[i_e] = (ep_splits[i_e] / max(ev, 1e-12)) / math.sqrt(max(kp_sweep[i_e], 1.0))

        # Reference spectrum at EP point (delta=0, dg=4J)
        h_spec = np.array([[-0.5j * g_tot, j_ghz], [j_ghz, -0.5j * (g_tot + 4.0 * j_ghz)]], dtype=complex)
        for i_f, f in enumerate(f_ghz):
            m_inv = np.linalg.inv(f * np.eye(2) - h_spec)
            a_v = 1j * math.sqrt(g_ext) * m_inv[:, 0]
            t_fwd[i_f] = abs(1.0 - math.sqrt(g_ext) * a_v[0]) ** 2
            t_bwd[i_f] = t_fwd[i_f]

        # Chiral switching fidelity: CW lands in same output branch regardless of input, CCW lands in opposite branch!
        cw_avg_plus = 0.5 * (cw_from_p + cw_from_m)
        ccw_avg_plus = 0.5 * (ccw_from_p + ccw_from_m)
        chiral_switch_contrast = abs(cw_avg_plus - ccw_avg_plus)

        extra_curves["encircling_trajectories"] = {
            "progress": np.round(np.linspace(0.0, 1.0, len(tr_cw_p)), 3).tolist(),
            "CW_from_plus_Pplus": tr_cw_p,
            "CW_from_minus_Pplus": tr_cw_m,
            "CCW_from_plus_Pplus": tr_ccw_p,
            "CCW_from_minus_Pplus": tr_ccw_m,
        }
        metrics.update({
            "CW_from_plus_final_Pplus": round(cw_from_p, 4),
            "CW_from_minus_final_Pplus": round(cw_from_m, 4),
            "CCW_from_plus_final_Pplus": round(ccw_from_p, 4),
            "CCW_from_minus_final_Pplus": round(ccw_from_m, 4),
            "chiral_switching_contrast": round(chiral_switch_contrast, 4),
            "asymmetric_mode_switch_pass": bool(chiral_switch_contrast > 0.85),
            "Petermann_K_P_at_perturbation": round(bio_p["max_petermann_K_P"], 2),
        })

    # Optional 2D-FEM physical ring-to-ring gap synthesis & GDS export
    gap_ep_um = 0.22
    fem_gap_info: dict[str, float] | None = None
    if cfg.synthesize_fem_gap:
        fem_gap_info = _synthesize_physical_ring_gap(cfg, j_ghz)
        gap_ep_um = fem_gap_info["synthesized_gap_EP_um"]
        metrics.update(fem_gap_info)

    gds_out: str | None = None
    if cfg.export_gds_path:
        gds_out = _export_ep_gds(cfg, gap_ep_um, cfg.export_gds_path)
        metrics["gds_path"] = gds_out

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": metrics,
        "spectrum": {
            "detuning_GHz": np.round(f_ghz, 4).tolist(),
            "T_fwd_dB": np.round(10.0 * np.log10(np.maximum(t_fwd, 1e-6)), 3).tolist(),
            "R_fwd_dB": np.round(10.0 * np.log10(np.maximum(r_fwd, 1e-6)), 3).tolist(),
            "T_bwd_dB": np.round(10.0 * np.log10(np.maximum(t_bwd, 1e-6)), 3).tolist(),
            "R_bwd_dB": np.round(10.0 * np.log10(np.maximum(r_bwd, 1e-6)), 3).tolist(),
        },
        "ep_metrology": {
            "perturbation_GHz": [float(f"{x:.4e}") for x in eps_sweep],
            "ep_split_GHz": [float(f"{x:.4e}") for x in ep_splits],
            "dp_split_GHz": [float(f"{x:.4e}") for x in dp_splits],
            "petermann_K_P": [float(f"{x:.3e}") for x in kp_sweep],
            "snr_ratio_vs_dp": np.round(snr_ratio_sweep, 4).tolist(),
        },
        **extra_curves,
    }
